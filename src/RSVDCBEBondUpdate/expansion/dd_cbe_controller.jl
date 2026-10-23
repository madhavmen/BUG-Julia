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

# ── T2.5: one side of the bond expansion ──────────────────────────────────────────────────

"""
    dd_cbe_expand(f, mpo, i, lenv, renv, side; alloc, ledger, workcap, growth,
                  exact, rng, kc) -> CBEExpansion

Grows ONE frame at bond `(i, i+1)`: `U0` for `side = :left` (K-sweep), `V0` for `:right` (L-sweep).
The other frame is returned unchanged. Rank comes from the QR; the probe width from
[`_dd_controller`](@ref), or from the numerical rank alone when `alloc === nothing` or `exact`.
Retained directions are bounded by the local room and the work ceiling.
"""
function dd_cbe_expand(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink,
                       side::Symbol; alloc::Union{Nothing, BasisAllocation} = nothing,
                       ledger::Union{Nothing, AdaptiveLedger} = nothing,
                       workcap::Int = 0, growth::Float64 = ADAPTIVE_GROWTH,
                       exact::Bool = false,
                       rng::AbstractRNG = MersenneTwister(0x5EED),
                       kc::KernelCounters = KernelCounters(), version::Int = 0,
                       holdout_width::Int = _DD_HOLDOUT, audit::Bool = false)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
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
    Q, n = if alloc === nothing || exact
        Qr, _, nr = _certified_range(draw, wleg, select, fl, project, sketch, side, w0, wcap,
                                     cap, nothing, ledger)
        (Qr, nr)
    else
        Qc, _ = _dd_controller(frame, side, draw, sketch, project, w0, wcap, cap, alloc,
                               ledger; kc = kc, s = holdout_width, audit = audit)
        (Qc, Qc === nothing ? 0 : leg_dim(Qc, fl))
    end
    Q === nothing && return none()
    if n > cap
        Q = _trim_total(Q, fl, cap)
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
trial-space frame), `D_final` (after compression), root iterations, total seconds and bytes,
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
