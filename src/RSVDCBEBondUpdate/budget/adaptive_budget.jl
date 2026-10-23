
"""
    AdaptiveBudget(basis, trunc, expv)

The three accuracy tolerances of one CBE-BUG step. `basis` governs the trial space (what the step
may look for and must keep), `trunc` the relative cumulative discarded weight of the final
compression, and `expv` the error estimate of the root exponential. They are kept separate on
purpose.
"""
struct AdaptiveBudget
    basis::Float64
    trunc::Float64
    expv::Float64

    function AdaptiveBudget(basis::Real, trunc::Real, expv::Real)
        basis > 0 || throw(ArgumentError("tol_basis must be positive, got $basis"))
        trunc >= 0 || throw(ArgumentError("tol_trunc must be non-negative, got $trunc"))
        expv > 0 || throw(ArgumentError("tol_expv must be positive, got $expv"))
        return new(Float64(basis), Float64(trunc), Float64(expv))
    end
end

"""
    BasisAllocation

`tol_basis` apportioned across the step's bonds and four basis tests, with the residual scale:
`per_bond` (one bond's whole decision), `sample` (Test B), `tail` (Test A), `krylov` (Tests C/D),
`scale` (factor applied to raw Frobenius residuals).

The default `:rss` rule is `per_bond = tol_basis / sqrt(n_bonds * n_frontiers)`, each test `per_bond
/ sqrt(4)`: errors over bonds add in quadrature. That assumes independent residuals across bonds; a
systematic correlation makes it optimistic by up to `sqrt(n_bonds)`. The `:linear` alternative
([`basis_allocation`](@ref)) needs no such assumption. `rule` records which one bounded the step.
"""
struct BasisAllocation
    per_bond::Float64
    sample::Float64
    tail::Float64
    krylov::Float64
    scale::Float64
    rule::Symbol
end
BasisAllocation(per_bond, sample, tail, krylov, scale) =
    BasisAllocation(per_bond, sample, tail, krylov, scale, :rss)

"""
    basis_allocation(budget, nbonds; tau, nrm, nfront = 1) -> BasisAllocation

Apportion `budget.basis` over `nbonds` bonds, `nfront` frontiers and the four tests, and fix the
scale the residuals are measured in. The scale is `|tau| / ||psi||`: the residuals enter the step
through `exp(tau*H)` at O(tau), so omitting `|tau|` over-expands at small steps. Dividing by
`||psi||` keeps `tol_basis` meaningful for unnormalised states. `tau = 0` (basis-only sweeps) gives
scale 1, a pure closure question.
"""
function basis_allocation(budget::AdaptiveBudget, nbonds::Int;
                          tau::Number = 0, nrm::Real = 1.0, nfront::Int = 1,
                          rule::Symbol = :rss)
    nb = max(nbonds, 1)
    nf = max(nfront, 1)
    rule in (:rss, :linear) || throw(ArgumentError("rule must be :rss or :linear, got $rule"))
    at = abs(tau)
    scale = iszero(at) ? 1.0 : at / max(Float64(nrm), eps(Float64))
    if rule === :linear
        per_test = budget.basis / (2 * (nb + 1) * nf)
        return BasisAllocation(2 * per_test, per_test, per_test, per_test, scale, :linear)
    end
    per_bond = budget.basis / sqrt(nb * nf)
    # Four tests (A, B-left/right counted as one sampling decision, C/D as one closure
    # decision) -> sqrt(4) = 2. Kept as an explicit `sqrt(4)` so the count is visible.
    per_test = per_bond / sqrt(4)
    return BasisAllocation(per_bond, per_test, per_test, per_test, scale, :rss)
end

"""
    indicator(alloc, resid) -> Float64

A raw Frobenius residual expressed as the scaled, comparable basis-error indicator. Every
Test A/B/C/D comparison goes through this, so the scaling convention lives in exactly one
place rather than being re-derived (and re-mistaken) at four call sites.
"""
indicator(alloc::BasisAllocation, resid::Real) = alloc.scale * Float64(resid)

# ── THE STEP LEDGER ──
# A solver that cannot meet its tolerance must say so. `maxdim` is a resource limit, never a
# substitute for an accuracy tolerance, so when it binds the status changes rather than the claim.

"""
    AdaptiveLedger

Per-step accounting of what the basis decisions actually cost and whether the budget held.

`tolerance_met` is false as soon as ANY bond exhausts its allowance without passing its
check; `workcap_reached` records that a rank/work ceiling -- not a tolerance -- ended a
search. The two are distinct: hitting the cap with the tolerance already satisfied is a
normal, healthy step.
"""
mutable struct AdaptiveLedger
    max_sample::Float64          # worst Test B indicator over the step, either side
    max_sample_l::Float64        # ...left expansions only (plan §8 `B_left`)
    max_sample_r::Float64        # ...right expansions only (plan §8 `B_right`)
    max_tail::Float64            # worst Test A indicator
    max_krylov::Float64          # worst Test C/D indicator
    max_krylov_c::Float64        # worst Test C (first Krylov direction vs the CBE frame)
    max_krylov_d::Float64        # worst Test D (later Krylov directions vs the grown frame)
    # The final compression's RELATIVE CUMULATIVE discarded norm, `sqrt(sum_b w_b) / ||psi||`
    # over every bond it cut -- the quantity `tol_trunc` bounds (plan §5, UT14). Not the
    # largest single bond's weight, which `CBEBugSweepInfo.discarded` still reports.
    discarded::Float64
    expv_err::Float64            # worst root `expv` error estimate (Saad, relative to ||S0||)
    n_probes::Int                # sketch columns drawn, summed over bonds and sides
    n_holdout::Int               # independent validation probes drawn
    n_selected::Int              # CBE directions retained
    n_krylov::Int                # Krylov directions appended
    n_h1::Int                    # one-site H applications
    n_targeted::Int              # sectors Test B widened after a failed holdout
    # Plan §8: how often Test C finds CBE already captures the first one-site action, and how
    # often Test D must still grow the frame although Test C passed (the UT08 case).
    n_c_pass::Int
    n_d_after_c::Int
    workcap_reached::Bool
    tolerance_met::Bool
    # `maxdim` cut a bond of the final compression past the rank its `tol_trunc` share asked
    # for. Reported even when the cumulative total still fits (plan §7 UT14: a rank-cap
    # violation is reported, not masked).
    maxdim_reached::Bool
    # Plan §5/§8 `state_projection_error`: `||(1 - P) psi|| / ||psi||` for the frames the root
    # solve integrates in. Zero up to roundoff on the ordinary step (the frames contain `psi`);
    # on the midpoint step's second stage it measures how well the half-step frames hold the
    # ORIGINAL state, and it is charged against `tol_basis`.
    proj_err::Float64
    # Plan §8 `r_old`, `r_augmented`, `r_final`: largest bond rank before the step, of the
    # trial space (half-sweep frames), and after the final compression.
    r_old::Int
    r_aug::Int
    r_final::Int
    # Plan §8 `root_expv_iterations`: operator applications of the root solve (all substeps).
    expv_iters::Int
    # WHICH checks failed, so `:budget_unmet` comes with a reason (see `budget_diagnostic`):
    # `:sample` (Test B), `:tail` (Test A), `:krylov` (Tests C/D), `:trunc` (final
    # compression), `:projection` (midpoint frames), `:expv` (root solve), `:closure` (the
    # order-p Krylov frontiers did not close within the pass bound).
    unmet::Vector{Symbol}
    # One entry per Test B verdict (per bond and side): `(side, est, relunc, per, holdout,
    # sketch)` -- the indicator, a bound on its relative sampling error, each sector's scaled
    # indicator, and the holdout and sketch columns per sector. Plan §3 asks for sector-resolved
    # estimates WITH their uncertainty, and §8 for `n_reachable_sectors`/`probes_per_sector`;
    # a step-level maximum cannot say which sector or how sure.
    sample_log::Vector{Any}
    # Plan §8 "track by bond and timestep": one entry per half-sweep bond visit, written by
    # `_log_bond!` -- the bond's own Test A/B/C/D indicators, its counts, its verdict, the
    # width of the frame it produced (`r_augmented`), and the reachable sectors with their
    # probes; after the compression `cbe_bug_step!` adds `r_old`, `r_final` and that bond's
    # relative `discarded` norm. The step-level fields above are the max/sum over these.
    bond_log::Vector{Any}
    # ── General-order Krylov closure (`krylov_closure.jl`; 1 / 0 / empty on `order = 1`) ──
    order::Int                   # the temporal order the step was asked for
    n_passes::Int                # closure passes over the frontiers j = 1..order-1
    n_closure_sweeps::Int        # frontier half-sweep pairs run (enriching or measuring)
    # Final time-weighted outside-space action of each frontier j = 0..order-1,
    # `|tau| ||(I - P) A F_j|| / ||psi||` (upper bound sqrt(left^2 + right^2)), measured on the
    # frames the root solve used; and the integrated-defect estimate `sum_j leak[j]/(j+1)`.
    leak::Vector{Float64}
    defect::Float64
    n_root_frontier::Int         # zero-site applications that built the frontiers
    n_h2_leak::Int               # two-site applications that measured the outside-space action
    n_h1_closure::Int            # one-site applications of the frontier Tests C/D
    # One entry per frontier sweep: pass, degree, leak before, directions added, widths,
    # newly occupied sectors, bonds skipped by the leak gate.
    closure_log::Vector{Any}
    t_closure::Float64
    n_compress::Int              # final `root_out` compressions (plan: exactly one per step)
    # The same frontiers measured on the frames the half-sweeps built, BEFORE any closure
    # enrichment: what the ordinary CBE + local-Krylov step already captures. `leak_initial` is
    # time-weighted like `leak`; `rho_initial[j]` is scale-free, `||(I - P) A F_j|| / ||F_j||`
    # (upper bound). Filled by `order > 1` with `closure_checks` and by `closure_probe` on `order =
    # 1`.
    leak_initial::Vector{Float64}
    rho_initial::Vector{Float64}
    # Plan Task 6: the parallel K/L/S step's account (`_kls_finish`), `nothing` on every other step.
    kls::Any
end

AdaptiveLedger() = AdaptiveLedger(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0, 0, 0, 0, 0, 0, 0, 0, false, true,
                                  false, 0.0, 0, 0, 0, 0, Symbol[], Any[], Any[],
                                  1, 0, 0, Float64[], 0.0, 0, 0, 0, Any[], 0.0, 0,
                                  Float64[], Float64[], nothing)

"""
    _unmet!(ledger, what) -> ledger

Record that the check `what` failed. THE ONE PLACE `tolerance_met` IS CLEARED, so a failed
step always carries the reason with it; `nothing` (the legacy path) is a no-op.
"""
_unmet!(::Nothing, ::Symbol) = nothing
function _unmet!(ledger::AdaptiveLedger, what::Symbol)
    ledger.tolerance_met = false
    what in ledger.unmet || push!(ledger.unmet, what)
    return ledger
end

"""
    budget_status(ledger) -> Symbol

`:met` when every basis, compression and projection check of the step passed its share of
the budget, `:budget_unmet` otherwise (plan §1). `nothing` (the legacy path, which makes no
tolerance claim) gives `:legacy`. [`budget_diagnostic`](@ref) says which check failed and
what to change.
"""
budget_status(::Nothing) = :legacy
budget_status(ledger::AdaptiveLedger) = ledger.tolerance_met ? :met : :budget_unmet

"""
    budget_diagnostic(ledger) -> String

A one-line account of a `:budget_unmet` step: the failed checks with their worst indicators,
and the retry that addresses each (plan §1 "retry diagnostic"). Empty when the budget held.
"""
function budget_diagnostic(ledger::AdaptiveLedger)
    ledger.tolerance_met && return ""
    msg = String[]
    cap = ledger.workcap_reached ? " at the work ceiling (raise maxdim or reduce dt)" :
                                   " with the local space exhausted (reduce dt)"
    :sample in ledger.unmet &&
        push!(msg, "Test B sampling residual $(ledger.max_sample)" * cap)
    :tail in ledger.unmet &&
        push!(msg, "Test A retained-rank tail $(ledger.max_tail)" * cap)
    :krylov in ledger.unmet &&
        push!(msg, "Tests C/D Krylov residual $(ledger.max_krylov)" * cap)
    :trunc in ledger.unmet &&
        push!(msg, "final compression discarded $(ledger.discarded) relative norm" *
                   (ledger.maxdim_reached ? " because maxdim bound (raise maxdim or loosen tol_trunc)" :
                                            " (loosen tol_trunc)"))
    :projection in ledger.unmet &&
        push!(msg, "midpoint frames miss the original state by $(ledger.proj_err) (reduce dt)")
    :expv in ledger.unmet &&
        push!(msg, "root expv error estimate $(ledger.expv_err) after $(ledger.expv_iters) " *
                   "applications (reduce dt)")
    :closure in ledger.unmet &&
        push!(msg, "order-$(ledger.order) Krylov closure did not close after " *
                   "$(ledger.n_passes) passes, frontier leak $(ledger.leak)" * cap *
                   " (or lower the order)")
    return "budget unmet: " * join(msg, "; ")
end

# NOT `Base.merge!`: that method is for associatives, and adding a `TLArray`-free method to it
# from here would be type piracy in a module that already imports a great deal of Base.
"Fold a second ledger into the first. The two half-sweeps each keep their own and merge."
function merge_ledger!(a::AdaptiveLedger, b::AdaptiveLedger)
    a.max_sample = max(a.max_sample, b.max_sample)
    a.max_sample_l = max(a.max_sample_l, b.max_sample_l)
    a.max_sample_r = max(a.max_sample_r, b.max_sample_r)
    append!(a.sample_log, b.sample_log)
    a.max_tail   = max(a.max_tail,   b.max_tail)
    a.max_krylov = max(a.max_krylov, b.max_krylov)
    a.max_krylov_c = max(a.max_krylov_c, b.max_krylov_c)
    a.max_krylov_d = max(a.max_krylov_d, b.max_krylov_d)
    a.n_c_pass    += b.n_c_pass
    a.n_d_after_c += b.n_d_after_c
    a.discarded  = max(a.discarded,  b.discarded)
    a.expv_err   = max(a.expv_err,   b.expv_err)
    a.n_probes  += b.n_probes
    a.n_holdout += b.n_holdout
    a.n_selected += b.n_selected
    a.n_krylov  += b.n_krylov
    a.n_h1      += b.n_h1
    a.n_targeted += b.n_targeted
    a.workcap_reached |= b.workcap_reached
    a.tolerance_met   &= b.tolerance_met
    a.maxdim_reached  |= b.maxdim_reached
    a.proj_err = max(a.proj_err, b.proj_err)
    a.r_old    = max(a.r_old, b.r_old)
    a.r_aug    = max(a.r_aug, b.r_aug)
    a.r_final  = max(a.r_final, b.r_final)
    a.expv_iters += b.expv_iters
    union!(a.unmet, b.unmet)
    append!(a.bond_log, b.bond_log)
    a.order = max(a.order, b.order)
    a.n_passes += b.n_passes
    a.n_closure_sweeps += b.n_closure_sweeps
    append!(a.leak, b.leak)
    a.defect = max(a.defect, b.defect)
    a.n_root_frontier += b.n_root_frontier
    a.n_h2_leak += b.n_h2_leak
    a.n_h1_closure += b.n_h1_closure
    append!(a.closure_log, b.closure_log)
    a.t_closure += b.t_closure
    a.n_compress += b.n_compress
    append!(a.leak_initial, b.leak_initial)
    append!(a.rho_initial, b.rho_initial)
    b.kls === nothing || (a.kls = b.kls)
    return a
end

"""
    _log_bond!(ledger, bl, bond, side, width) -> ledger

Close one bond of a half-sweep: `bl` is the fresh ledger that bond's CBE (Tests A/B) and Krylov
(Tests C/D) decisions wrote into. Its summary goes into `bl.bond_log`, then `bl` is folded into
the half-sweep's `ledger` -- so the step-level maxima and sums are exactly the aggregate of the
logged bonds, by construction rather than by a second bookkeeping path. `nothing` (legacy) is a
no-op.
"""
_log_bond!(::Nothing, ::Nothing, ::Int, ::Symbol, ::Int) = nothing
function _log_bond!(ledger::AdaptiveLedger, bl::AdaptiveLedger, bond::Int, side::Symbol,
                    width::Int)
    # Probes per reachable sector, from the FINAL Test B sketch of each expansion side (Test B
    # logs every widening attempt; the last one is what the frame was built from). Empty on
    # `exact = true`, which draws no sketch.
    final = Dict{Symbol, Any}()
    for s in bl.sample_log
        final[s.side] = s.sketch
    end
    probes = Dict{Symbol, Dict{Any, Int}}(k => v for (k, v) in final)
    push!(bl.bond_log,
          (bond = bond, side = side, sample = bl.max_sample, tail = bl.max_tail,
           krylov_c = bl.max_krylov_c, krylov_d = bl.max_krylov_d,
           n_probes = bl.n_probes, n_holdout = bl.n_holdout, n_selected = bl.n_selected,
           n_targeted = bl.n_targeted, n_krylov = bl.n_krylov, n_h1 = bl.n_h1,
           c_pass = bl.n_c_pass > 0, d_after_c = bl.n_d_after_c > 0,
           workcap = bl.workcap_reached, met = bl.tolerance_met, unmet = copy(bl.unmet),
           width = width, n_sectors = sum(length, values(probes); init = 0),
           probes_per_sector = probes))
    return merge_ledger!(ledger, bl)
end

"""
    work_ceiling(maxdim, r) -> Int

The internal cap on temporary working rank at one bond, from the resource limit and the bond's
current rank. Not a user setting.

Expansion may overshoot `maxdim` while the trial space is built, since that is how a rank-`r` state
discovers rank-`r'` structure. It may not grow without bound, which an uncapped spectrum-driven rule
does on bonds whose singular values never decay. The policy `max(2*maxdim, 2*r, 1)` gives headroom
for the augmented rank-`2r` basis, and the `2r` floor stops a bond already above `maxdim` from being
frozen. It is a tuned policy, not a theorem; `workcap_reached` in the ledger reports when it binds.
"""
work_ceiling(maxdim::Int, r::Int) = max(2 * maxdim, 2 * r, 1)

"""
The adaptive search's STARTING width schedule (`budget = ceil(growth*dmax) - r`, probe `1.2x` that).
USER 2026-10-16: 1.2, not the legacy 2.0 -- at 2.0 the first probe is ~1.2chi columns against a
local space of 2chi, so the RSVD starts as wide as the space it sketches. Only a start: Test B
widens a side that fails (`_certified_range(...; thin = true)`). `cbe_expand`'s own default
(2.0) is the fixed-count selection of the TDVP-CBE baselines.
"""
const ADAPTIVE_GROWTH = 1.2

"Default tolerances of the normal API. Plan §1; starting settings, not certified bounds."
const TOL_DEFAULTS = (basis = 1e-5, trunc = 1e-7, expv = 1e-10)
