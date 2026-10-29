# dd_cbe_controller.jl: the incremental DD controller, `dd_cbe_expand` and the step record (from `dd_cbe.jl`, included right after it).

# ── T3.1-T3.4: the incremental DD sketch controller ───────────────────────────────────────

"Plan T3.3: the default independent holdout width `s` (an implementation constant; tests 2, 4, 8)."
const _DD_HOLDOUT = 4

"""
    _dd_controller(frame, side, draw, sketch, project, w0, wcap, ncap, alloc, ledger;
                   kc, s, audit, holdout, trace) -> (Q, status)

Grows the DD basis `Q` of one side block by block until an independent holdout says the rest of
`G = P^D X P^D` fits this side's share `alloc.sample`. Each probe block is streamed through the MPO
once and its action cached; a failed check draws a fresh holdout before promoting the old one.

Statuses: `:met`; `:work_limited` (`wcap` or `ncap` reached); `:basis_limited` (a pass added no
direction); `:empty` (no discarded space). Every continuing pass adds a direction, so the loop ends.
"""
function _dd_controller(frame, side::Symbol, draw, sketch, project, w0::Int, wcap::Int,
                        ncap::Int, alloc::BasisAllocation, ledger::Union{Nothing, AdaptiveLedger};
                        kc::KernelCounters = KernelCounters(), s::Int = _DD_HOLDOUT,
                        audit::Bool = false, holdout = w -> draw(w, nothing),
                        trace::Union{Nothing, Vector} = nothing)
    left = side === :left
    fl, wleg = left ? (3, 1) : (1, 3)
    tags = frame.inds[fl].itags
    aug, Q, n, streamed = frame, nothing, 0, 0
    boost = Dict{Any, Int}()
    function absorb!(Z)
        Qn, _ = _dd_select(aug, Z, side, kc)
        Qn === nothing && return 0
        Qn = to_concrete(setitag(Qn, fl, tags))
        Q = Q === nothing ? Qn : to_concrete(oplus([Q, Qn], (fl,)))
        aug = to_concrete(oplus([aug, Qn], (fl,)))
        return leg_dim(Qn, fl)
    end
    function stream(Om, field::Symbol)
        w = leg_dim(Om, wleg)
        streamed += w
        ledger === nothing || setfield!(ledger, field, getfield(ledger, field) + w)
        return sketch(Om)
    end
    function fresh(w)
        Om = holdout(w)
        return Om === nothing ? nothing : (Om, stream(Om, :n_holdout))
    end
    resid(h) = _sample_residual(Q, project, _ -> h[2], _ -> h[1], 1, wleg, side)
    newsector() = Q !== nothing && any(((q, _),) -> sector_dim(frame, fl, q) == 0, Q.spaces[fl])

    w = clamp(w0, 1, max(wcap, 1))
    last = draw(w, nothing)
    last === nothing && return (nothing, :empty)
    n += absorb!(stream(last, :n_probes))
    h = fresh(s)
    h1 = h
    escalated = false
    while true
        h === nothing && return (Q, :met)            # nothing left to validate against
        est, per, hcols = resid(h)
        ind = indicator(alloc, est)
        trace === nothing || push!(trace, (n, ind, indicator(alloc, resid(h1)[1])))
        ok = ind <= alloc.sample
        if ok && !escalated && (audit || ind > alloc.sample / 2 || newsector())
            escalated = true
            he = fresh(2s)
            he === nothing || (h = he; continue)
        end
        capped = streamed >= wcap || n >= ncap
        if ok || capped
            ledger === nothing || _record_sample!(ledger, side, ind, per, hcols, last, wleg, alloc)
            ok && return (Q, :met)
            if ledger !== nothing
                _unmet!(ledger, :sample)
                ledger.workcap_reached = true
            end
            return (Q, :work_limited)
        end
        nt = _target_sectors!(boost, per, h[1], wleg, alloc)
        ledger === nothing || (ledger.n_targeted += nt)
        hnew = fresh(s)                          # new validation BEFORE the old one is promoted
        added = absorb!(h[2])
        nt == 0 && (w = min(2 * w, wcap))
        blk = draw(w, isempty(boost) ? nothing : boost)
        if blk !== nothing
            last = blk
            added += absorb!(stream(blk, :n_probes))
        end
        n += added
        if added == 0
            ledger === nothing || (_record_sample!(ledger, side, ind, per, hcols, last, wleg,
                                                   alloc); _unmet!(ledger, :sample))
            return (Q, :basis_limited)
        end
        h = hnew
    end
end

# ── QTT plan §1.2 / §1.2A, arm D3: the hybrid predictor controller ────────────────────────

"Bounded oversampling of D3's first block over its predictor: `clamp(cld(pred, 4), 4, 8)` columns."
_dd_oversample(pred::Int) = clamp(cld(pred, 4), 4, 8)

"""
    DDMemory()

Per-bond memory of the D3 hybrid controller (`dd_policy = :hybrid`): the directions accepted last
time at `(side, bond)` -- the PREDICTOR of the next first block, never a correctness condition --
and one trace entry per call (`bond`, `side`, `branch`, `ell`, `s`, `retries`, `m`, `n`, `added`,
`streamed`, `status`) for the first-impressions report (plan §1.2A: reported for every sampled
bond). Locked, since the two half-sweeps may run concurrently.
"""
mutable struct DDMemory
    accepted::Dict{Tuple{Symbol, Int}, Int}
    trace::Vector{NamedTuple}
    lock::ReentrantLock
end
DDMemory() = DDMemory(Dict{Tuple{Symbol, Int}, Int}(), NamedTuple[], ReentrantLock())

_dd_predict(mem::DDMemory, side::Symbol, i::Int) =
    lock(() -> get(mem.accepted, (side, i), nothing), mem.lock)

function _dd_remember!(mem::DDMemory, side::Symbol, i::Int, n::Int, rec::NamedTuple)
    lock(mem.lock) do
        mem.accepted[(side, i)] = n
        push!(mem.trace, rec)
    end
    return mem
end

"""
    _dd_hybrid(frame, opp, side, draw, sketch, sketch_adj, exhaust_opp, exhaust_self, project,
               w0, wcap, cap, m, n, alloc, ledger; pred, rng, kc, s) -> (Q, status, record)

Arm D3 of plan §1.2A, on the same `X = mat(F(psi))`, frames and DD-QR kernels as D1:

1. PREFLIGHT. `m`, `n` are the discarded dimensions of this side and of the probe side. When
   `ell + s + ell_retry >= min(m, n)` sampling cannot undercut the exhaustive DD action (a cost
   heuristic, not a theorem), so that runs instead: the probe side's whole DD basis (`n` columns),
   or for `2m < n` the mirrored route (`m` adjoint columns give the row space, then its rank in
   forward columns). Exhaustive is exact up to the QR floor, so no holdout is drawn.
2. Otherwise ONE block of `ell = pred + _dd_oversample(pred)` columns (`pred` the rank accepted
   here last time, else `w0`), QR, and an INDEPENDENT holdout of `s` columns from its own RNG
   stream, accepted at `alloc.sample` (the time-weighted `eta_DD = h R_DD / ||psi||`).
3. On failure the holdout's actions are absorbed and a bounded block of `max(ell ÷ 2, s)` new
   columns appended, with a fresh holdout, until met, `wcap` streamed columns or `cap` directions
   (`:work_limited`), or a pass adds nothing (`:basis_limited`).
"""
function _dd_hybrid(frame, opp, side::Symbol, draw, sketch, sketch_adj, exhaust_opp, exhaust_self,
                    project, w0::Int, wcap::Int, cap::Int, m::Int, n::Int,
                    alloc::BasisAllocation, ledger::Union{Nothing, AdaptiveLedger};
                    pred::Union{Nothing, Int}, rng::AbstractRNG,
                    kc::KernelCounters = KernelCounters(), s::Int = _DD_HOLDOUT)
    left = side === :left
    fl, wleg = left ? (3, 1) : (1, 3)
    oside = left ? :right : :left
    tags = frame.inds[fl].itags
    aug, Q, nq, streamed = frame, nothing, 0, 0
    function absorb!(Z)
        Qn, _ = _dd_select(aug, Z, side, kc)
        Qn === nothing && return 0
        Qn = to_concrete(setitag(Qn, fl, tags))
        Q = Q === nothing ? Qn : to_concrete(oplus([Q, Qn], (fl,)))
        aug = to_concrete(oplus([aug, Qn], (fl,)))
        return leg_dim(Qn, fl)
    end
    function stream(Om, field::Symbol)
        streamed += leg_dim(Om, wleg)
        ledger === nothing || setfield!(ledger, field, getfield(ledger, field) + leg_dim(Om, wleg))
        return sketch(Om)
    end
    ell = clamp(pred === nothing ? w0 : pred + _dd_oversample(pred), 1, max(wcap, 1))
    retry = max(ell ÷ 2, s)
    rec(branch, st, retries) = (side = side, branch = branch, ell = ell, s = s, retries = retries,
                                m = m, n = n, added = nq, streamed = streamed, status = st)

    if ell + s + retry >= min(m, n)
        if n <= 2m
            B = exhaust_opp()
            B === nothing && return (nothing, :empty, rec(:exhaustive, :empty, 0))
            Bq, _ = _dd_select(opp, B, oside, kc)            # orthonormal DD basis, n columns
            Bq === nothing && return (nothing, :empty, rec(:exhaustive, :empty, 0))
            nq += absorb!(stream(Bq, :n_probes))
            return (Q, :met, rec(:exhaustive, :met, 0))
        end
        Sx = exhaust_self()
        Sx === nothing && return (nothing, :empty, rec(:mirrored, :empty, 0))
        Sq, _ = _dd_select(frame, Sx, side, kc)               # this side's DD basis, m columns
        Sq === nothing && return (nothing, :empty, rec(:mirrored, :empty, 0))
        streamed += leg_dim(Sq, fl)
        ledger === nothing || (ledger.n_probes += leg_dim(Sq, fl))
        Wq, _ = _dd_select(opp, sketch_adj(Sq), oside, kc)   # the row space of A_DD
        Wq === nothing && return (nothing, :met, rec(:mirrored, :met, 0))
        nq += absorb!(stream(Wq, :n_probes))
        return (Q, :met, rec(:mirrored, :met, 0))
    end

    hrng = MersenneTwister(rand(rng, UInt64))                 # the holdout's own stream
    blk = draw(ell, nothing)
    blk === nothing && return (nothing, :empty, rec(:sampled, :empty, 0))
    nq += absorb!(stream(blk, :n_probes))
    retries = 0
    while true
        Oh = _dd_probe(opp, oside, s; rng = hrng, kc = kc)
        Oh === nothing && return (Q, :met, rec(:sampled, :met, retries))
        Zh = stream(Oh, :n_holdout)
        est, per, hcols = _sample_residual(Q, project, _ -> Zh, _ -> Oh, 1, wleg, side)
        ind = indicator(alloc, est)
        if ind <= alloc.sample || streamed >= wcap || nq >= cap
            ledger === nothing || _record_sample!(ledger, side, ind, per, hcols, blk, wleg, alloc)
            ind <= alloc.sample && return (Q, :met, rec(:sampled, :met, retries))
            if ledger !== nothing
                _unmet!(ledger, :sample)
                ledger.workcap_reached = true
            end
            return (Q, :work_limited, rec(:sampled, :work_limited, retries))
        end
        added = absorb!(Zh)                                   # the failed holdout is kept
        w = min(retry, max(wcap - streamed, 0))
        if w > 0
            b2 = draw(w, nothing)
            b2 === nothing || (blk = b2; added += absorb!(stream(b2, :n_probes)))
        end
        nq += added
        retries += 1
        if added == 0
            ledger === nothing || (_record_sample!(ledger, side, ind, per, hcols, blk, wleg, alloc);
                                   _unmet!(ledger, :sample))
            return (Q, :basis_limited, rec(:sampled, :basis_limited, retries))
        end
    end
end

# ── T2.5: one side of the bond expansion ──────────────────────────────────────────────────

const _TAG_W, _TAG_WR = "cbeW,L", "cbeW,R"

"""
    _trim_by_weight(Q, B, side, n) -> Q or nothing

Keep the `n` directions of `Q` (orthonormal, outside the frame) that carry the most of the DD
action `M = P^D X P^D`, given `B = Q' M` (rows `(g, site_r, link_r)` for `:left`, columns
`(link_l, site_l, g)` for `:right`, `g` = `Q`'s leg): Telum's SVD ranks them across ALL sectors
(`Nkeep`), as the final selection of `cbe_expand` does. `_trim_total` instead fills sectors in
storage order, so at a binding work ceiling whole charge sectors lose their expansion (the latent
trap of PLAN-tolerance-control.md §1). MEASURED (dissipative A1 L = 12, dt 0.1, rank probe rp7f):
frame cut at 1024 against 1679 needed, obs error 2.2e-4 against 1.3e-12 uncut, the spurious
components then forcing the compressed rank to maxdim off the root.
"""
function _trim_by_weight(Q, B, side::Symbol, n::Int)
    n <= 0 && return nothing
    left = side === :left
    leg_dim(Q, left ? 3 : 1) <= n && return Q
    U = svd(B, left ? (1,) : (3,), _TAG_W, _TAG_WR; cutoff = 0.0, Nkeep = n).U    # (g, k)
    return left ? to_concrete(contract(Q, (3,), U, (1,))) :     # (link, site, k)
                  to_concrete(contract(U, (1,), Q, (1,)))       # (k, site, link)
end

"""
    dd_cbe_expand(f, mpo, i, lenv, renv, side; alloc, ledger, workcap, growth,
                  exact, rng, kc) -> CBEExpansion

Grows ONE frame at bond `(i, i+1)`: `U0` for `side = :left` (K-sweep), `V0` for `:right` (L-sweep).
The other frame is returned unchanged. Rank comes from the QR; the probe width from
[`_dd_controller`](@ref), or from the numerical rank alone when `alloc === nothing` or `exact`.
Retained directions are bounded by the local room and the work ceiling; past the ceiling
`cap_order = :weight` keeps the heaviest ([`_trim_by_weight`](@ref)), `:sector` the first sectors
in storage order (`_trim_total`).

`dd_width = l > 0` is the fixed-oversampling ablation (xx-efficiency plan Figs. 1A and 4): one
probe of width `l` through the same DD projection, streamed kernel and QR, no holdout.
`dd_policy = :hybrid` is the QTT plan's arm D3 ([`_dd_hybrid`](@ref)), its predictor read from and
written to `dd_memory` (a [`DDMemory`](@ref); `nothing` = no predictor); `:adaptive` is D1; `:off`
adds nothing (the no-CBE control).
"""
function dd_cbe_expand(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink,
                       side::Symbol; alloc::Union{Nothing, BasisAllocation} = nothing,
                       ledger::Union{Nothing, AdaptiveLedger} = nothing,
                       workcap::Int = 0, growth::Float64 = ADAPTIVE_GROWTH,
                       exact::Bool = false,
                       rng::AbstractRNG = MersenneTwister(0x5EED),
                       kc::KernelCounters = KernelCounters(), version::Int = 0,
                       holdout_width::Int = _DD_HOLDOUT, audit::Bool = false,
                       cap_order::Symbol = :sector, dd_width::Int = 0,
                       dd_policy::Symbol = :adaptive, dd_memory = nothing)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    cap_order in (:sector, :weight) || throw(ArgumentError(
        "cap_order must be :sector or :weight, got $cap_order"))
    dd_policy in (:adaptive, :hybrid, :off) || throw(ArgumentError(
        "dd_policy must be :adaptive, :hybrid or :off, got $dd_policy"))
    # `:off` -- the no-CBE control of plan §1.2A (the frame then grows by local Krylov alone)
    dd_policy === :off && return CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)
    left = side === :left
    U0, V0 = f.U0, f.V0
    r = f.old_rank
    frame, opp = left ? (U0, V0) : (V0, U0)
    oside = left ? :right : :left
    fl, wleg = left ? (3, 1) : (1, 3)
    none() = CBEExpansion(U0, V0, 0, 0, 0.0, 0.0)

    # The fused local dimension, as `cbe_expand` reads it (SU(2) counts multiplets).
    fused(t, a, b) = symmetry_mode() === :SU2 ?
        sum(d for (_, d) in reachable_sectors(t, a, b); init = 0) : leg_dim(t, a) * leg_dim(t, b)
    full_self = left ? fused(U0, 1, 2) : fused(V0, 2, 3)
    full_opp  = left ? fused(V0, 2, 3) : fused(U0, 1, 2)
    cap = full_self - r
    wbind = false
    if alloc !== nothing && workcap > 0
        wk = max(workcap - r, 0)
        wbind = wk < cap
        cap = min(cap, wk)
    end
    cap <= 0 && return none()

    dmax = max(leg_dim(U0, 1), r, leg_dim(V0, 3))
    w0 = ceil(Int, 1.2 * max(ceil(Int, growth * dmax) - r, 1))
    wcap = exact ? w0 : full_opp
    (alloc !== nothing && workcap > 0 && !exact) && (wcap = min(wcap, workcap))

    draw(w, b) = exact ? _dd_probe(opp, oside, w; rng = rng, exact = true, kc = kc) :
                         _dd_probe(opp, oside, w; rng = rng, boost = b, kc = kc)
    op = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = !left, version = version,
                         counters = kc)
    out = Ref{Any}(nothing)
    sketch(Om) = (apply_two_site_sketch!(out, Om, op); out[])
    project = left ? (X -> perp_component(U0, X)) : (X -> perp_component_right(V0, X))
    select(Om) = (first(_dd_select(frame, sketch(Om), side, kc)), 0.0)

    # Plan T3.1: with an allocation the incremental controller decides the width (it is always
    # thin); without one, or for the exact reference, the numerical-rank stop of `_range_find`
    # alone.
    Q, n = if dd_width > 0 && !exact
        blk = draw(min(dd_width, full_opp), nothing)
        Qf = blk === nothing ? nothing : first(_dd_select(frame, sketch(blk), side, kc))
        blk === nothing || ledger === nothing || (ledger.n_probes += leg_dim(blk, wleg))
        (Qf, Qf === nothing ? 0 : leg_dim(Qf, fl))
    elseif alloc === nothing || exact
        Qr, _, nr = _certified_range(draw, wleg, select, fl, project, sketch, side, w0, wcap,
                                     cap, nothing, ledger)
        (Qr, nr)
    elseif dd_policy === :hybrid
        opA = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = left, version = version,
                              counters = kc)
        outA = Ref{Any}(nothing)
        # anonymous: a named local function inside a branch is hoisted to the whole method
        sketch_adj = Om -> (apply_two_site_sketch!(outA, Om, opA); outA[])
        exhaust_opp = () -> _dd_probe(opp, oside, 0; rng = rng, exact = true, kc = kc)
        exhaust_self = () -> _dd_probe(frame, side, 0; rng = rng, exact = true, kc = kc)
        pred = dd_memory === nothing ? nothing : _dd_predict(dd_memory, side, i)
        Qh, _, rec = _dd_hybrid(frame, opp, side, draw, sketch, sketch_adj, exhaust_opp,
                                exhaust_self, project, w0, wcap, cap, full_self - r,
                                full_opp - r, alloc, ledger; pred = pred, rng = rng, kc = kc,
                                s = holdout_width)
        nh = Qh === nothing ? 0 : leg_dim(Qh, fl)
        dd_memory === nothing || _dd_remember!(dd_memory, side, i, nh, merge((bond = i,), rec))
        (Qh, nh)
    else
        Qc, _ = _dd_controller(frame, side, draw, sketch, project, w0, wcap, cap, alloc,
                               ledger; kc = kc, s = holdout_width, audit = audit)
        (Qc, Qc === nothing ? 0 : leg_dim(Qc, fl))
    end
    Q === nothing && return none()
    if n > cap
        Q = if cap_order === :weight
            # B = Q' M exactly: the opposite sketch applied to the `n` directions themselves
            opT = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = left, version = version,
                                  counters = kc)
            B = (apply_two_site_sketch!(out, Q, opT); out[])
            _trim_by_weight(Q, left ? perp_component_right(V0, B) : perp_component(U0, B),
                            side, cap)
        else
            _trim_total(Q, fl, cap)
        end
        Q === nothing && return none()
    end
    ledger === nothing || (wbind && n >= cap && (ledger.workcap_reached = true))
    # Insurance only: the QR floor is per sector and absolute, so roundoff in a saturated sector
    # does not become a column the way a globally relative SVD cut lets it (`_guarded_block`).
    Q = _cap_to_local_room(frame, Q, side)
    Q === nothing && return none()
    nq = leg_dim(Q, fl)
    kc.n_sector_new += count(((q, _),) -> sector_dim(frame, fl, q) == 0, Q.spaces[fl])
    ledger === nothing || (ledger.n_selected += nq)
    grown = to_concrete(oplus([frame, to_concrete(setitag(Q, fl, frame.inds[fl].itags))], (fl,)))
    return left ?
        CBEExpansion(_check_frame(grown, :left, "dd_cbe_expand U_ex"), V0, nq, 0, 0.0, 0.0) :
        CBEExpansion(U0, _check_frame(grown, :right, "dd_cbe_expand V_ex"), 0, nq, 0.0, 0.0)
end

# ── T1.4: the typed step record ───────────────────────────────────────────────────────────

"""
    StepRecord

One `cbe_bug_step!`'s account, typed and allocation-light (plan T1.4): which algorithm ran
(`policy`, `qr_backend`, `closure_policy` for the order > 1 frontier sweeps, `order`,
`parallel`), the kernel counters, the ranks `D_old` (largest bond before), `D_stage` (largest
trial-space frame, untruncated), `D_final` (after compression) and the two limits they are held
to, `D_max` (`maxdim`, the final rank) and `D_work` (the working-rank ceiling `work_ceiling`
puts on the frames), root iterations, total seconds and bytes,
the process `peak_rss` (`Sys.maxrss()`, a high-water mark, not per step), and `limits`: which
of `:basis_limited`, `:root_limited`, `:truncation_limited`, `:work_limited`, `:rank_limited`
held the step back (empty when the budget held), and `budget_rule`, how `tol_basis` was
aggregated over the shares (`:linear`; plan T3.2).
"""
struct StepRecord
    policy::Symbol
    qr_backend::Symbol
    closure_policy::Symbol
    order::Int
    parallel::Symbol
    counters::KernelCounters
    D_old::Int
    D_stage::Int
    D_final::Int
    D_max::Int
    D_work::Int
    root_iters::Int
    t_step::Float64
    bytes_step::Int
    peak_rss::Int
    limits::Vector{Symbol}
    budget_rule::Symbol
end

"Plan invariant 4: the separate limit statuses, read off the ledger's failed checks."
function _step_limits(ledger::AdaptiveLedger)
    lim = Symbol[]
    any(u -> u in (:sample, :tail, :krylov, :closure, :projection), ledger.unmet) &&
        push!(lim, :basis_limited)
    :expv in ledger.unmet && push!(lim, :root_limited)
    :trunc in ledger.unmet && push!(lim, :truncation_limited)
    ledger.maxdim_reached && push!(lim, :rank_limited)
    ledger.workcap_reached && push!(lim, :work_limited)
    return lim
end

"""
    _resolve_parallel(p) -> Bool

`parallel` of the plan's API: `:serial`/`false`, `:half_sweeps`/`true`. `:full` also runs the
half-sweeps concurrently; `cbe_bug_step!` then takes the separate parallel K/L/S integrator of
plan Task 6 (`parallel_kls_step!`).
"""
function _resolve_parallel(p)
    p isa Bool && return p
    p === :serial && return false
    p in (:half_sweeps, :full) && return true
    throw(ArgumentError("parallel must be :serial, :half_sweeps, :full or a Bool, got $p"))
end
