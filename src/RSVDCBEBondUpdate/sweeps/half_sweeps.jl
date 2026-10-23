
# ── THE TWO HALF-SWEEPS OF `cbe_bug_step!` (plan: StageBasisBuilder) ──────────────────────────
#
# Both are BASIS GENERATORS: CBE widens each bond (`dd_cbe_expand`), the frame grows from the CBE
# output by the Tests C/D Krylov stage (`_krylov_enrich`), the environment is pushed through it,
# and no site tensor is evolved. They share no data -- the left reads the state canonical at 1 and
# writes `W`, the right reads it canonical at `L` and writes `Z` -- so `cbe_bug_step!` can run them
# concurrently. Each carries its own accumulator, ledger and RNG stream for exactly that reason.
#
# ── THE TWO HALF-SWEEPS ARE INDEPENDENT, AND THAT IS NOW EXPLOITED ────────────────────
#
# ⛔ THEY WERE NEVER COUPLED. The left sweep reads the state canonical at 1 and writes
# `W`; the right reads it canonical at `L` and writes `Z`.
# Neither reads the other's output -- the only meeting point is the SINGLE Galerkin solve at
# the root, which needs both and runs after both. The sequential order was an artefact of
# re-gauging ONE state object in place, not a dependence in the method.
#
# ⚠ SO THE LEFT SWEEP GETS ITS OWN COPY. `copy(psi)` copies the tensor VECTOR only, and every
# mutation in this package REPLACES a slot rather than writing into a tensor, so the two
# objects can be re-gauged independently while sharing the tensors neither of them modifies.
#
# ⛔ AND `psi` ITSELF STAYS THE `L`-CANONICAL ONE, deliberately: `AL`, `AR` and the whole
# assembly in `cbe_bug_step!` read `psi`, and they must all see ONE gauge. Mixing `C` (built in
# the 1-canonical gauge) with `D` (built in the L-canonical one) would contract two different
# bases at link `c+1` and is silently wrong -- which is why the step recomputes `AL` against
# `psi` and only `AR` can be taken straight from the right sweep.
#
# ── RUN THEM ─────────────────────────────────────────────────────────────────────────
#
# ⚠ `parallel = true` NEEDS `julia -t 2` OR MORE TO DO ANYTHING. With one thread the tasks are
# interleaved on the same worker and the step is the sequential one plus scheduling overhead,
# so the flag is a pessimisation rather than a no-op -- which is why it is not the default.
#
# ⚠ AND IT COMPETES WITH BLAS. Two tasks each calling into a BLAS already configured for every
# core will oversubscribe; Telum also passes `Threads.nthreads()` to HPTT for every
# `permutedims`, so the nesting is three deep. `benchmarks/rsvd_parallel.jl` reports the
# thread counts it ran under for exactly this reason.
#
# ⛔ WHAT MAKES THIS SAFE, CHECKED RATHER THAN ASSUMED:
#   * `copy(psi)` copies the tensor VECTOR; every mutation in this package REPLACES a slot,
#     so `canonical!` on one copy cannot be seen by the other.
#   * Telum's `svd`/`contract` allocate their HPTT buffers PER CALL (`permute_mat` is a local),
#     so there is no shared scratch to race on. Its only globals are the opt-in cost counters
#     and the display settings.
#   * `KRYLOV_LOG` is shared, and `expv` appends to it under `_KRYLOV_LOCK` (the parallel
#     K/L/S stages run three solves at once). The half-sweeps use `_krylov_enrich`, which
#     does not record.
#   * `_LOCAL_SPACES` is a lazily-filled global `Dict`; it is already populated by the time a
#     state exists, so the sweeps only read it.

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

"""
    _HalfSweepCtx

What both half-sweeps of one step READ and never write: the operator, `tau`, the generator shift,
the root bond `c`, the step's allocation and DD-QR expansion settings (`exkw`), the work ceiling,
the `krylov_reuse` ablation and the states the trial space must contain (`keep_states`).
"""
struct _HalfSweepCtx{M, E, A, K, S}
    mpo::M
    tau::ComplexF64
    eshift::E
    c::Int
    alloc::A
    exkw::K
    wcap::Int
    krylov_reuse::Bool
    keep_states::S
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

# ⛔ PEAK IS TRACKED PER HALF-SWEEP AND SUMMED, NOT SCANNED ACROSS ALL FRAME ARRAYS. Summing every
# array on every call (a) reads arrays the OTHER half-sweep is concurrently writing -- `isassigned`
# racing a `setindex!` -- and (b) is `O(L)` tensor walks per bond, i.e. `O(L^2)` per step of pure
# diagnostic work. Each sweep accumulates its own frames incrementally.
#
# ⚠ THE REPORTED NUMBER IS THEREFORE AN UPPER BOUND: both halves' frames are counted as
# simultaneously live. Under `parallel = true` that is literally true, and it keeps the two arms'
# working sets comparable, which is the point of the column.
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
    rstack = (t0 = time_ns(); e = right_env_stack(psiL, mpo; downto = 3);
              acc.tenv += (time_ns() - t0) / 1e9; e)
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
        acc.tkry += (time_ns() - tk0) / 1e9
        _record_krylov!(bondlg, kst, cx.alloc)
        _peak!(acc, held + _state_stored(psiL), Wkry)
        W[i] = to_concrete(setitag(Wkry, 3, _tagW(i)))
        for (t, X) in pairs(cx.keep_states)
            KX = kcar[t] === nothing ? X[i] : to_concrete!(contract(kcar[t], (2,), X[i], (1,)))
            W[i] = _union_left(W[i], KX)
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
        i < c && ((lch = (t0 = time_ns(); e = push_left_channels(lch, mpo, W[i], i);
                 acc.tenv += (time_ns() - t0) / 1e9; e)))
    end
    return lroot
end

"""
    _half_right!(psiR, Z, expanded, nn, acc, hledger, rng, cx) -> (renv_root, AR)

PHASE B(b), RIGHT -> ROOT: writes `Z[c:L-1]`, `expanded[c+1:L-1]` and `nn`, and returns the right
environment at link `c+2` in the NEW basis and the sweep's final carry.

⛔ ITS FINAL CARRY **IS** `AR`. The `D` recursion here and an `AR` loop after the sweeps would be
the SAME recursion over the SAME tensors in the SAME gauge -- `psi` is not touched between them --
so `AR` is just the value `D` ends on.

⚠ `AL` GETS NO SUCH SHORTCUT and is recomputed by the step. The left sweep's own carry `C` is
built in the 1-canonical gauge while `AR` is in the L-canonical one, and the two meet over link
`c+1` -- different bases for the same link. Contracting them would be silently wrong rather than
an error.
"""
function _half_right!(psiR, Z, expanded, nn, acc::_HalfAcc, hledger, rng, cx::_HalfSweepCtx)
    mpo, c = cx.mpo, cx.c
    L = length(psiR)
    canonical!(psiR, L)
    lstack = (t0 = time_ns(); e = left_env_stack(psiR, mpo; upto = L - 2);
              acc.tenv += (time_ns() - t0) / 1e9; e)
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
        acc.tkry += (time_ns() - tk0) / 1e9
        _record_krylov!(bondlg, kst, cx.alloc)
        _peak!(acc, held + _state_stored(psiR), Zkry)
        Z[j] = to_concrete(setitag(Zkry, 1, _tagZ(j)))
        for (t, X) in pairs(cx.keep_states)
            BX = kcar[t] === nothing ? X[j + 1] :
                 to_concrete!(contract(X[j + 1], (3,), kcar[t], (1,)))
            Z[j] = _union_right(Z[j], BX)
            kcar[t] = to_concrete!(contract(BX, (2, 3), Z[j]', (2, 3)))
        end
        _check_frame(Z[j], :right, "half-sweep Z[$j]")
        j != c && (expanded[j] = leg_dim(Z[j], 1))
        _log_bond!(hledger, bondlg, j, :right, leg_dim(Z[j], 1))
        held += tensor_elements(Z[j])

        D = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))          # (link_j, b_j)
        j == c && (rroot = rch)
        j > c && ((rch = (t0 = time_ns(); e = push_right_channels(rch, mpo, Z[j], j + 1);
                 acc.tenv += (time_ns() - t0) / 1e9; e)))
    end
    return (rroot, D)                   # `D` is (link_{c+1}, b_R)
end
