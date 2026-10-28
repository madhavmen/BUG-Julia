
# ── the two half-sweeps of `cbe_bug_step!` (plan: StageBasisBuilder) ──
# Both only build bases (they never evolve a site tensor) and share no data, so they run
# concurrently on their own gauges. `psi` stays L-canonical so `AL`, `AR` and the assembly read one
# gauge; `copy` is safe because every mutation replaces a slot.

"""
    _HalfAcc

Everything ONE half-sweep accumulates, held per half-sweep rather than in closure variables.

⛔ THIS EXISTS BECAUSE THE HALF-SWEEPS CAN NOW RUN CONCURRENTLY. Counters shared between them
(`nmv`, `epre`, the phase timers) are read-modify-write from two tasks, so as closure `Ref`s they
are a data race -- and a racy `+=` on a Float64 does not crash, it silently loses increments, so
the failure would show up as a profile that quietly under-reports and never as an error.
"""
mutable struct _HalfAcc
    nmv::Int
    ngrow::Int
    epre::Float64
    efnl::Float64
    peak::Int
    tcbe::Float64
    tkry::Float64
    tsplit::Float64
    tframe::Float64
    tenv::Float64
    kc::KernelCounters      # this half-sweep's kernel counters (plan T1.4)
end
_HalfAcc() = _HalfAcc(0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, KernelCounters())

"`f(args...)`, an environment build or push, charged to the half-sweep's `tenv` and `kc.b_env`.
The arguments are passed, not captured: a closure over the sweep's reassigned `lch`/`rch` boxes them."
function _env_timed!(acc::_HalfAcc, f, args...; kw...)
    e, dt, db = _measure(() -> f(args...; kw...))
    acc.tenv += dt
    acc.kc.b_env += db
    return e
end

"""
    _HalfSweepCtx

What both half-sweeps of one step READ and never write: the operator, `tau`, the generator shift,
the root bond `c`, the step's allocation and DD-QR expansion settings (`exkw`), the work ceiling,
the `krylov_reuse` ablation, the states the trial space must contain (`keep_states`) and the
`krylov_recycle` cache (`nothing` when off), which round 1 fills.
"""
struct _HalfSweepCtx{M, E, A, K, S, R}
    mpo::M
    tau::ComplexF64
    eshift::E
    c::Int
    alloc::A
    exkw::K
    wcap::Int
    krylov_reuse::Bool
    keep_states::S
    recycle::R
end

"""
Time one bond expansion into the half-sweep's own accumulator: `side`'s frame grows by DD-QR CBE
(`dd_cbe_expand`, plan T2.4) into the bond's own `ledger`; `nn[i]` records what it admitted.
"""
function _expand_bond!(acc::_HalfAcc, cx::_HalfSweepCtx, nn, f, i, lch, rch, side::Symbol,
                       ledger, rng)
    t0 = time_ns()
    ex = dd_cbe_expand(f, cx.mpo, i, lch, rch, side; cx.exkw..., ledger = ledger, rng = rng,
                       kc = acc.kc, version = i)
    acc.tcbe += (time_ns() - t0) / 1e9
    acc.epre = max(acc.epre, ex.err_pre)
    acc.efnl = max(acc.efnl, ex.err_fnl)
    nn[i] = side === :left ? ex.n_new_l : ex.n_new_r
    return ex
end

# Peak is tracked per half-sweep and summed, not scanned across the frame arrays: scanning reads
# arrays the other half-sweep is writing concurrently, and costs O(L) walks per bond. The reported
# number is therefore an upper bound, counting both halves' frames as live at once, which keeps the
# two arms' working sets comparable when they run in parallel.
function _peak!(acc::_HalfAcc, held::Int, ts...)
    acc.peak = max(acc.peak, held + sum(tensor_elements(t) for t in ts; init = 0))
    return nothing
end

"""
Union `Wk` (the basis update's output) with `U0` (the CBE frame), on the LEFT.

`U0` FIRST so the CBE directions are kept verbatim and only the part of `Wk` outside them is
appended -- `_preselect_left` returns `nothing` when there is nothing to append, which is the
ordinary case once the two already span the same space.
"""
function _union_left(U0, Wk)
    Q, _ = _preselect_left(U0, Wk, 1e-14)
    Q === nothing && return U0
    return to_concrete(oplus([U0, to_concrete(setitag(Q, 3, U0.inds[3].itags))], (3,)))
end
"Row mirror of [`_union_left`](@ref): the bond leg is 1 and the direct sum is over it."
function _union_right(V0, Zk)
    Q, _ = _preselect_right(V0, Zk, 1e-14)
    Q === nothing && return V0
    return to_concrete(oplus([V0, to_concrete(setitag(Q, 1, V0.inds[1].itags))], (1,)))
end

"""
    _half_left!(psiL, W, expanded, nn, acc, hledger, rng, cx) -> lenv_root

PHASE B(a), LEFT -> ROOT: writes `W[1:c]`, `expanded[1:c]` and `nn`, logs every bond into
`hledger`, and returns the left environment at link `c` in the NEW basis (captured before the
push through the root frame).

THE SWEEP IS INTERLEAVED: CBE expands bond `i`, then bond `i`'s frame is built and the
environment pushed through it immediately -- the reference's own ordering
(TDVPSweepCBE1Si.m:30/:34), so the environment chain runs through the NEW frames and bond
`i+1`'s selection sees the widening at bond `i`.
"""
function _half_left!(psiL, W, expanded, nn, acc::_HalfAcc, hledger, rng, cx::_HalfSweepCtx)
    mpo, c = cx.mpo, cx.c
    canonical!(psiL, 1)
    _recycle_begin!(cx.recycle, :left, psiL, 0)
    rstack = _env_timed!(acc, right_env_stack, psiL, mpo; downto = 3)
    lch = boundary_channels(mpo)
    lroot = lch
    C = nothing
    kcar = Any[nothing for _ in cx.keep_states]   # each kept state's carry, `(b_i, link)`
    held = 0                          # frames this sweep has accumulated, running total
    for i in 1:c
        K = C === nothing ? psiL[i] : to_concrete!(contract(C, (2,), psiL[i], (1,)))
        _fr = (t0 = time_ns(); f = _frame_from(K, psiL[i + 1]);
               acc.tframe += (time_ns() - t0) / 1e9; f)
        bondlg = AdaptiveLedger()                               # this bond's own ledger (§8)
        ex = _expand_bond!(acc, cx, nn, _fr, i, lch, right_channels(rstack, i + 2), :left,
                           bondlg, rng)
        _peak!(acc, held + _state_stored(psiL), ex.U_ex, ex.V_ex)
        # ── STAGE 3 (plan §4): THE FRAME GROWS FROM `U_ex` BY TESTS C/D ─────────────────
        # The CBE frame is KEPT (old directions verbatim, CBE directions reused) and a Krylov
        # direction is appended only when its weighted residual outside the frame matters.
        M    = to_concrete!(contract(psiL[i + 1], (2, 3), ex.V_ex', (2, 3)))
        Kex  = to_concrete!(contract(K, (3,), M, (1,)))
        rch1 = push_right_channels(right_channels(rstack, i + 2), mpo, ex.V_ex, i + 1)
        H1   = OneSiteOp(shift_h(one_site_h(mpo, i, lch, rch1), cx.eshift), i, acc.kc)
        tk0  = time_ns()
        Wkry, kst = _krylov_enrich(Kex, x -> (acc.nmv += 1; apply_one_site(H1, x)),
                                   cx.krylov_reuse ? ex.U_ex : _fr.U0, :left, cx.tau, cx.alloc;
                                   workcap = cx.wcap)
        cx.krylov_reuse || (Wkry = _merge_frame(Wkry, ex.U_ex, :left))
        _recycle_store!(cx.recycle, :left, i, kst, ex.V_ex, cx.alloc.krylov)
        acc.tkry += (time_ns() - tk0) / 1e9
        _record_krylov!(bondlg, kst, cx.alloc)
        _peak!(acc, held + _state_stored(psiL), Wkry)
        W[i] = to_concrete(setitag(Wkry, 3, _tagW(i)))
        # Union every kept state first: a later union widens `W[i]`, so a carry taken before it
        # would hold a stale bond leg.
        KXs = [kcar[t] === nothing ? X[i] : to_concrete!(contract(kcar[t], (2,), X[i], (1,)))
               for (t, X) in pairs(cx.keep_states)]
        for KX in KXs
            W[i] = _union_left(W[i], KX)
        end
        for (t, KX) in pairs(KXs)
            kcar[t] = to_concrete!(contract(W[i]', (1, 2), KX, (1, 2)))
        end
        _check_frame(W[i], :left, "half-sweep W[$i]")
        expanded[i] = leg_dim(W[i], 3)
        _log_bond!(hledger, bondlg, i, :left, expanded[i])
        held += tensor_elements(W[i])

        # (d) carry the ORIGINAL amplitude into the new basis. `K`, not `Kex`: `W[i]` touches
        # only legs 1-2, so this lands on `(b_i, link_i)` and composes with `psiL[i+1]`.
        C = to_concrete!(contract(W[i]', (1, 2), K, (1, 2)))
        i == c && (lroot = lch)         # captured BEFORE pushing through the root frame
        i < c && (lch = _env_timed!(acc, push_left_channels, lch, mpo, W[i], i))
    end
    _recycle_commit!(cx.recycle, :left, true)
    return lroot
end

"""
    _half_right!(psiR, Z, expanded, nn, acc, hledger, rng, cx) -> (renv_root, AR)

Right-to-root phase: writes `Z[c:L-1]`, `expanded[c+1:L-1]` and `nn`, and returns the right
environment at link `c+2` in the new basis together with the sweep's final carry.

The final carry IS `AR`: the recursion here and an `AR` loop after the sweeps would be the same
recursion over the same tensors in the same gauge, so `AR` is simply where `D` ends. `AL` gets no
such shortcut and is recomputed, since the left carry is built in the 1-canonical gauge and the two
meet over link `c+1`; contracting them would be silently wrong.
"""
function _half_right!(psiR, Z, expanded, nn, acc::_HalfAcc, hledger, rng, cx::_HalfSweepCtx)
    mpo, c = cx.mpo, cx.c
    L = length(psiR)
    canonical!(psiR, L)
    # a shallow copy: the assembly later replaces `psi`'s slots, the cache must keep these
    _recycle_begin!(cx.recycle, :right, cx.recycle === nothing ? nothing : copy(psiR), 0)
    lstack = _env_timed!(acc, left_env_stack, psiR, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    rroot = rch
    D = nothing
    kcar = Any[nothing for _ in cx.keep_states]   # each kept state's carry, `(link, b_j)`
    held = 0
    for j in (L - 1):-1:c
        B = D === nothing ? psiR[j + 1] :
            to_concrete!(contract(psiR[j + 1], (3,), D, (1,)))
        _fr = (t0 = time_ns(); f = _frame_from(psiR[j], B);
               acc.tframe += (time_ns() - t0) / 1e9; f)
        bondlg = AdaptiveLedger()                               # this bond's own ledger (§8)
        ex = _expand_bond!(acc, cx, nn, _fr, j, left_channels(lstack, j), rch, :right,
                           bondlg, rng)
        _peak!(acc, held + _state_stored(psiR), ex.U_ex, ex.V_ex)
        # Row mirror of the left sweep's Stage 3; see the comment there.
        M    = to_concrete!(contract(ex.U_ex', (1, 2), psiR[j], (1, 2)))  # (b_L, link_j)
        Bex  = to_concrete!(contract(M, (2,), B, (1,)))
        lch1 = push_left_channels(left_channels(lstack, j), mpo, ex.U_ex, j)
        H1   = OneSiteOp(shift_h(one_site_h(mpo, j + 1, lch1, rch), cx.eshift), j, acc.kc)
        tk0  = time_ns()
        Zkry, kst = _krylov_enrich(Bex, x -> (acc.nmv += 1; apply_one_site(H1, x)),
                                   cx.krylov_reuse ? ex.V_ex : _fr.V0, :right, cx.tau, cx.alloc;
                                   workcap = cx.wcap)
        cx.krylov_reuse || (Zkry = _merge_frame(Zkry, ex.V_ex, :right))
        _recycle_store!(cx.recycle, :right, j, kst, ex.U_ex, cx.alloc.krylov)
        acc.tkry += (time_ns() - tk0) / 1e9
        _record_krylov!(bondlg, kst, cx.alloc)
        _peak!(acc, held + _state_stored(psiR), Zkry)
        Z[j] = to_concrete(setitag(Zkry, 1, _tagZ(j)))
        BXs = [kcar[t] === nothing ? X[j + 1] : to_concrete!(contract(X[j + 1], (3,), kcar[t], (1,)))
               for (t, X) in pairs(cx.keep_states)]
        for BX in BXs                  # all unions before any carry, as on the left
            Z[j] = _union_right(Z[j], BX)
        end
        for (t, BX) in pairs(BXs)
            kcar[t] = to_concrete!(contract(BX, (2, 3), Z[j]', (2, 3)))
        end
        _check_frame(Z[j], :right, "half-sweep Z[$j]")
        j != c && (expanded[j] = leg_dim(Z[j], 1))
        _log_bond!(hledger, bondlg, j, :right, leg_dim(Z[j], 1))
        held += tensor_elements(Z[j])

        D = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))          # (link_j, b_j)
        j == c && (rroot = rch)
        j > c && (rch = _env_timed!(acc, push_right_channels, rch, mpo, Z[j], j + 1))
    end
    _recycle_commit!(cx.recycle, :right, true)
    return (rroot, D)                   # `D` is (link_{c+1}, b_R)
end
