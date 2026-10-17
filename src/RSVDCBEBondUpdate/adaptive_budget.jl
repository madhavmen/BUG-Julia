# ── THE STEP'S ACCURACY BUDGET ────────────────────────────────────────────────────────────
#
# Three tolerances and one resource limit, and the whole point of the file is that NOTHING
# ELSE IS A USER KNOB. `dex`, `dover`, `comp_ratio`, `growth`, `krylov_basis`,
# `split_cutoff`, `root_cutoff` and the two `stol_*` were each a GUESS AT A RANK made before
# the run had produced the spectrum that answers it, and a guess made per Hamiltonian: the
# measured `err_fnl = 0.62` in `cbe_expand`'s budget block is one of those guesses discarding
# 62% of the ranked candidate weight because `growth = 2.0` happened to be wrong for that
# bond. The quantities are real and the algorithm still needs them -- they just belong
# DOWNSTREAM of a tolerance, derived from what the sketch actually found, rather than upstream
# of it as inputs.
#
# ⛔ THE THREE ARE NOT INTERCHANGEABLE, and the separation is the design:
#
#     tol_basis   what must be EXPLORED AND RETAINED in the trial space (RSVD + Krylov)
#     tol_trunc   what may be DISCARDED FROM THE EVOLVED STATE (final compression)
#     tol_expv    how accurately the root Galerkin ODE is SOLVED
#
# They fail independently and they are not substitutes. A converged root `expv` does not
# certify that the RSVD found a sufficient Galerkin space, and a poor root exponential is not
# fixed by admitting more CBE directions. Collapsing them into one number -- which is what
# "just set every cutoff to 1e-10" does -- hides which of the three is actually limiting.
#
# ⚠ `tol_basis` IS NOT A CERTIFIED GLOBAL ERROR BOUND, and must not be documented as one.
# The residual indicators below are timestep-scaled, leading-order SURROGATES for basis error,
# and the randomised ones are probabilistic estimates rather than certificates. They are
# calibrated against reference dynamics (`tests/common/free_fermion.jl`), which is evidence,
# not proof.

"""
    AdaptiveBudget(basis, trunc, expv)

The three accuracy tolerances for one CBE-BUG step.

  - `basis` -- RSVD sampling and retained rank, plus one-site Krylov enrichment. Governs the
    TRIAL SPACE: what the step is allowed to look for and obliged to keep.
  - `trunc` -- relative cumulative discarded weight of the final state compression. Governs
    what may be thrown away from the EVOLVED STATE.
  - `expv` -- error estimate of the root `expv` exponential action.

See the block above for why these three are separate and why nothing else is exposed.
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

`tol_basis` apportioned across the step's bonds and across the four basis tests, together
with the relative scale the residuals are measured in.

  - `per_bond` -- allowance for one bond's whole basis decision.
  - `sample`   -- Test B, the RSVD sampling residual (did the sketch reach the range?).
  - `tail`     -- Test A, the projected-Hamiltonian singular tail (did we keep enough?).
  - `krylov`   -- Tests C/D, one-site Krylov closure.
  - `scale`    -- the factor a raw Frobenius residual is multiplied by before comparison.

THE ALLOCATION RULE IS FIXED AND DOCUMENTED, NOT A SETTING, which is the entire reason this
struct exists rather than four more keywords. It is also deliberately the SIMPLEST rule that
is dimensionally honest, so that when it proves too conservative the benchmark says so against
a rule somebody can state in one sentence:

    per_bond = tol_basis / sqrt(n_bonds * n_frontiers)   errors over bonds add in quadrature
    each test = per_bond / sqrt(4)                       four independent checks, likewise

`n_frontiers` is the requested temporal ORDER `p` (`krylov_closure.jl`): the step closes the
Krylov frontiers `j = 0 .. p-1`, each a separate set of basis decisions over every bond. Frontier
`j` enters the local error weighted by `1/(j+1)` (variation of constants), so the integrated
defect is at most `sum_j eta_j/(j+1) <= (H_p / sqrt(p)) tol_basis`, i.e. `1.06, 1.06, 1.04 x
tol_basis` for `p = 2, 3, 4`. `p = 1` is the rule above unchanged.

⚠ QUADRATURE IS AN ASSUMPTION ABOUT INDEPENDENCE, not an identity. Residuals at different
bonds are only approximately orthogonal, and under a strongly correlated error (a systematic
under-sampling that hits every bond the same way) the true accumulation is closer to LINEAR
and this rule is optimistic by `sqrt(n_bonds)`. The conservative alternative -- `tol_basis /
n_bonds` -- costs real rank at every bond to insure against a case that the XX and Heisenberg
benchmarks do not exhibit. Quadrature is the measured-adequate choice, and `IT07`'s tolerance
sweep is what would overturn it.
"""
struct BasisAllocation
    per_bond::Float64
    sample::Float64
    tail::Float64
    krylov::Float64
    scale::Float64
end

"""
    basis_allocation(budget, nbonds; tau, nrm, nfront = 1) -> BasisAllocation

Apportion `budget.basis` over `nbonds` bonds, `nfront` Krylov frontiers (the order, see
[`BasisAllocation`](@ref)) and the four tests, and fix the relative scale residuals are
measured in.

THE SCALE IS `|tau| / ||psi||`, AND BOTH HALVES ARE LOAD-BEARING. The CBE and Krylov
residuals measure the part of `H*Theta` the trial space misses; what that costs the STEP is
that residual propagated through `exp(tau*H)`, i.e. it enters at `O(tau)`. Omitting `|tau|`
makes every indicator look `1/dt` too large and the solver over-expands at small timesteps --
precisely where the basis matters least. Dividing by `||psi||` makes the comparison relative,
so the same `tol_basis` means the same thing for a normalised state and an unnormalised one.

⚠ `tau = 0` IS A REAL CALL, not a degenerate one: it is how a caller asks for the basis
machinery without evolving (`truncate = false` basis-only sweeps, and the `UT`s). The scale is
then 1 -- a pure "is this subspace closed under H" question with no time in it -- rather than
0, which would declare every residual negligible and return the unexpanded frame.
"""
function basis_allocation(budget::AdaptiveBudget, nbonds::Int;
                          tau::Number = 0, nrm::Real = 1.0, nfront::Int = 1)
    nb = max(nbonds, 1)
    per_bond = budget.basis / sqrt(nb * max(nfront, 1))
    # Four tests (A, B-left/right counted as one sampling decision, C/D as one closure
    # decision) -> sqrt(4) = 2. Kept as an explicit `sqrt(4)` so the count is visible.
    per_test = per_bond / sqrt(4)
    at = abs(tau)
    scale = iszero(at) ? 1.0 : at / max(Float64(nrm), eps(Float64))
    return BasisAllocation(per_bond, per_test, per_test, per_test, scale)
end

"""
    indicator(alloc, resid) -> Float64

A raw Frobenius residual expressed as the scaled, comparable basis-error indicator. Every
Test A/B/C/D comparison goes through this, so the scaling convention lives in exactly one
place rather than being re-derived (and re-mistaken) at four call sites.
"""
indicator(alloc::BasisAllocation, resid::Real) = alloc.scale * Float64(resid)

# ── THE STEP LEDGER ───────────────────────────────────────────────────────────────────────
#
# ⛔ A SOLVER THAT CANNOT MEET ITS TOLERANCE MUST SAY SO. The failure this guards against is
# the one the whole plan is aimed at: a run that hits a rank ceiling, silently keeps whatever
# fitted, and reports a converged-looking result that is simply wrong. `maxdim` is a RESOURCE
# LIMIT, never a substitute for satisfying an accuracy tolerance, so when it binds the status
# changes rather than the claim.

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
    # enrichment: what the ordinary CBE + local-Krylov step already captures. `leak_initial`
    # is time-weighted like `leak`; `rho_initial[j]` is scale-free, `||(I - P) A F_j|| /
    # ||F_j||` (upper bound). Filled by `order > 1` with `closure_checks` and by `closure_probe` on `order = 1`.
    leak_initial::Vector{Float64}
    rho_initial::Vector{Float64}
end

AdaptiveLedger() = AdaptiveLedger(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0, 0, 0, 0, 0, 0, 0, 0, false, true,
                                  false, 0.0, 0, 0, 0, 0, Symbol[], Any[], Any[],
                                  1, 0, 0, Float64[], 0.0, 0, 0, 0, Any[], 0.0, 0,
                                  Float64[], Float64[])

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

The internal cap on TEMPORARY working rank at one bond, derived from the resource limit and
the bond's current rank. Not a user setting.

Bond expansion is allowed to overshoot `maxdim` while the trial space is being built -- that
is the entire mechanism by which a rank-`r` state discovers rank-`r'` structure, and clamping
the trial space at `maxdim` would make the final compression a formality that can never
recover anything. What it may NOT do is grow without bound, which is what an uncapped
spectrum-driven rule does on a bond whose singular values never decay (near full rank, every
drawn direction carries weight) -- the sketch then widens to the complete fused basis, which
is exactly the cost the RSVD exists to avoid.

`2*maxdim` with a floor of `2r` is the policy: twice the resource limit is enough headroom for
the augmented rank-`2r` basis the BUG construction asks for, and the `2r` floor keeps a bond
that is ALREADY at or above `maxdim` from being frozen at its current rank. ⚠ IT IS A POLICY,
TUNED BY BENCHMARK, NOT A THEOREM -- and deliberately not another keyword. If it binds often
enough to cost accuracy, `workcap_reached` in the ledger is what reports it.
"""
work_ceiling(maxdim::Int, r::Int) = max(2 * maxdim, 2 * r, 1)

# ── MIGRATING OFF THE OLD KNOBS ───────────────────────────────────────────────────────────
#
# ⛔ NO `adaptive = true` FLAG, DELIBERATELY. A mode switch is itself a tuning knob, and the
# one thing it reliably produces is a run configured in one mode and read as the other. The
# mode is INFERRED instead, from whether the caller named any of the retired parameters --
# which is information already present at the call site and cannot disagree with itself.
#
# This is only detectable because every legacy keyword now defaults to `nothing` ("not asked
# for") rather than to its old value. The old value is still what it RESOLVES to, so a caller
# who passes none of them and none of the tolerances gets exactly the historical step.
#
# ⛔ AND THE TWO SETS MAY NOT BE MIXED. Honouring `tol_basis` and `dex` simultaneously means
# two different answers to "how many directions" with no stated precedence -- the silent-
# contradiction failure that `grow_iters` already has a guard for a few hundred lines down.
# Refused rather than resolved.

"The retired tuning parameters, in the order they are reported to the caller."
const LEGACY_KNOBS = (:dex, :dover, :comp_ratio, :growth, :krylov_basis,
                      :split_cutoff, :split_maxdim, :root_cutoff, :root_maxdim,
                      :trunc_thresh, :stol_pre, :stol_fnl,
                      # Stage 3: the fixed-depth Krylov controls. Tests C/D decide the depth
                      # from `tol_basis` (plan §4), so these are legacy-only like `krylov_basis`.
                      :krylov_tol, :grow_iters, :krylov_grow,
                      # Stage 5: the root solve's iteration controls. `tol_expv` sets its
                      # convergence test and substep split (plan §1 "root expv tolerance").
                      :maxiter, :tol, :root_conv_tol, :root_substeps)

"Historical default of every legacy knob, i.e. what `nothing` resolves to on the legacy path."
const LEGACY_DEFAULTS = (dex = 0, dover = nothing, comp_ratio = 1.0, growth = 2.0,
                         krylov_basis = 30, split_cutoff = 1e-14, split_maxdim = 0,
                         root_cutoff = 0.0, root_maxdim = 0, trunc_thresh = 1e-12,
                         stol_pre = 1e-10, stol_fnl = 1e-13,
                         krylov_tol = 1e-6, grow_iters = nothing, krylov_grow = false,
                         maxiter = 30, tol = 1e-15, root_conv_tol = 0.0, root_substeps = 1)

"""
The adaptive search's STARTING width schedule (`budget = ceil(growth*dmax) - r`, probe `1.2x` that).
USER 2026-10-16: 1.2, not the legacy 2.0 -- at 2.0 the first probe is ~1.2chi columns against a
local space of 2chi, so the RSVD starts as wide as the space it sketches. Only a start: Test B
widens a side that fails (`_certified_range(...; thin = true)`). The legacy path keeps 2.0.
"""
const ADAPTIVE_GROWTH = 1.2

"Default tolerances of the normal API. Plan §1; starting settings, not certified bounds."
const TOL_DEFAULTS = (basis = 1e-5, trunc = 1e-7, expv = 1e-10)

"""
    resolve_budget(tol_basis, tol_trunc, tol_expv, legacy) -> (budget, adaptive, named)

Decide which API the caller used, and refuse the mixture.

`legacy` is a NamedTuple of the retired keywords as the caller left them, `nothing` meaning
"not asked for". Returns the [`AdaptiveBudget`](@ref), whether the ADAPTIVE path is active,
and the legacy keywords that were named (empty on the adaptive path).

`dover` is excluded from the "was it named" test: `nothing` is its own historical default, so
it carries no signal either way -- a legacy caller who wants it is already naming `dex`.
A key absent from `legacy` counts as not named.
"""
function resolve_budget(tol_basis, tol_trunc, tol_expv, legacy::NamedTuple)
    named = Tuple(k for k in LEGACY_KNOBS
                  if k !== :dover && get(legacy, k, nothing) !== nothing)
    tols  = Tuple(n for (n, v) in ((:tol_basis, tol_basis), (:tol_trunc, tol_trunc),
                                   (:tol_expv, tol_expv)) if v !== nothing)

    if !isempty(named) && !isempty(tols)
        throw(ArgumentError(
            "cannot mix the accuracy tolerances $(join(tols, ", ")) with the retired " *
            "tuning parameters $(join(named, ", ")): they are two answers to the same " *
            "question with no precedence between them. Pass only tol_basis/tol_trunc/" *
            "tol_expv/maxdim for the adaptive solver, or only the old parameters for the " *
            "fixed-knob legacy path."))
    end

    budget = AdaptiveBudget(something(tol_basis, TOL_DEFAULTS.basis),
                            something(tol_trunc, TOL_DEFAULTS.trunc),
                            something(tol_expv,  TOL_DEFAULTS.expv))
    return (budget, isempty(named), named)
end

"""
    _legacy_hint(named) -> Union{Nothing, String}

Plan §1 ("may"): the migration hint for a caller who named retired knobs. Naming one is a
legitimate request for the fixed-knob path -- the recorded benchmarks use it -- so this informs
rather than refuses, and only ONCE per session: a scan of thousands of legacy steps should not
print thousands of lines. Returns the message (for tests), `nothing` on the adaptive path.
"""
function _legacy_hint(named)
    isempty(named) && return nothing
    msg = "$(join(named, ", ")) selected the fixed-knob legacy path, which makes no accuracy " *
          "claim (budget_status = :legacy). The adaptive solver takes only tol_basis, " *
          "tol_trunc, tol_expv and maxdim; see README \"Three tolerances and one resource limit\"."
    @warn msg maxlog=1
    return msg
end

"Resolve one legacy keyword to a concrete value, falling back to its historical default."
legacy_or_default(v, name::Symbol) = v === nothing ? getfield(LEGACY_DEFAULTS, name) : v
