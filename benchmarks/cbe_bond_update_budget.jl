# How much expansion budget does the gate-based CBE-BUG actually need?
#
# NOW A LEGITIMATE QUESTION, and it was not before. Every earlier tuning measurement was
# taken on a scheme that did not expand its bonds correctly and is void. The gate path is
# different: the CBE selection now drives the residual of the gate's action on every bond to
# machine zero (`tests/RSVDCBEBondUpdate/test_cbe_bond_update.jl`), rank tracks the validated
# integrator, and the sweep underneath is the validated odd/even Trotter one. So a budget
# sweep here measures the METHOD rather than a bug.
#
# THE ONE NUMBER THIS ANSWERS. At identical settings -- same Strang composition, same gates,
# same truncation -- the gate-CBE XX error was 1.4e-4 against the validated integrator's
# 5.8e-7. The only difference between the two runs is the basis update:
#
#   validated : K/L ODE solves, then admit EVERY direction found, bounded only by Sulz 2r.
#   gate-CBE  : one sketched gate application, budget `ceil(1.1*dmax) - r`, weight-ranked.
#
# The default budget is a ~10% growth schedule, far less than 2r. If the gap closes as `dex`
# rises, the machinery is right and what remains is the accuracy/rank trade-off the method
# exists to exploit -- CBE is supposed to reach a given accuracy at SMALLER rank, and this
# curve is where that claim either shows up or does not. If the gap does NOT close at full
# budget, something other than the budget differs, and the curve says so plainly.
#
# XX free fermions, because the reference is analytic: no truth run to trust, no truncation
# in the reference, and `xx_free_fermion_sz` is already pinned in the validated suite.
#
# SECOND PASS (the run that set the new default). The curve above closed -- `dex = 8`
# saturates at 1.17e-6 against the schedule's 1.43e-4, with `err_fnl = 0.62` on the schedule,
# i.e. it was discarding 62% of the ranked candidate weight for want of budget. So the
# DEFAULT was the limiter, and the default is now `growth = 2.0` (`cbe.jl`) instead of the
# reference's DMRG ratchet of 1.1. This file now measures three things at once:
#
#   * the A/B of the schedule itself, `growth = 1.1` vs `2.0`, at `dex = 0`;
#   * the absolute-budget curve, unchanged, as the yardstick the schedule must reach;
#   * BOTH PATHS -- the Lubich/MPO sweep shares `cbe_expand`, so it inherits the new default
#     and has to be re-measured on it rather than assumed.
#
# What to read: the new default should land on the saturated end of the `dex` curve (~1.2e-6),
# with `err_fnl` at 0 -- nothing ranked out for want of budget -- and wall time within noise
# of the old one, because the sweep dominates the sketch (4.4-4.7s across dex = 1 .. 64).

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf
include("../tests/common/free_fermion.jl")

const L      = 10
const DT     = 0.02
const NSTEPS = 25
const MAXDIM = 64
const TRUNC  = 1e-14
const T_END  = DT * NSTEPS

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(L - 1)]

"Site-resolved <Sz>, NORMALISED. `site_expval` does not divide by the norm and both methods
shed norm under truncation, so an unnormalised profile reports norm loss as accuracy error."
function profile(psi)
    q = copy(psi)
    m = magnetisation(q)
    return m ./ norm(q)^2
end

err_of(psi) = maximum(abs.(profile(psi) .- xx_free_fermion_sz(L, T_END)))

println("XX domain wall, L = $L, dt = $DT, $NSTEPS steps (t = $T_END), " *
        "maxdim = $MAXDIM, trunc = $TRUNC")
println("Reference: xx_free_fermion_sz (analytic).\n")

# ---- the validated integrator, for the bar ----------------------------------
pk = domain_wall_state(L)
tk = @elapsed ik = bond_update_bug!(pk, xx_gates(pk);
                                    opts = BondUpdateOptions(dt = DT, n_steps = NSTEPS,
                                                             maxdim = MAXDIM,
                                                             trunc_thresh = TRUNC))
@printf("%-22s  err %.3e   maxbond %3d   aug %3d   %6.1fs\n",
        "bond_update_bug!", err_of(pk), maximum(ik.max_bond_dims),
        maximum(ik.aug_k_dims), tk)
println()

"One row: run `f` on a fresh domain wall with these expansion controls and report."
function row(label, f; kwargs...)
    p = domain_wall_state(L)
    o = CBEBugOptions(; dt = DT, n_steps = NSTEPS, maxdim = MAXDIM,
                      trunc_thresh = TRUNC, kwargs...)
    t = @elapsed i = f(p, o)
    @printf("%-24s  err %.3e   maxbond %3d   aug %3d   %6.1fs   err_pre %.2e  err_fnl %.2e\n",
            label, err_of(p), maximum(i.max_bond_dims), maximum(i.max_expanded), t,
            maximum(i.err_pre), maximum(i.err_fnl))
end

gate_run(p, o) = cbe_bond_update_bug!(p, xx_gates(p); opts = o)
# The Lubich sweep dresses H with channel environments rather than using per-bond gates,
# so it takes the whole chain rather than a gate list.
const H = xxz_chain(L; J = 1.0, delta = 0.0)
mpo_run(p, o) = cbe_lubich_bug!(p, H; opts = o)

# ---- the schedule A/B, both paths -------------------------------------------
# `dex = 0` hands the budget to the schedule: `ceil(growth*dmax) - r`, neighbourhood-coupled.
# 1.1 is the reference's DMRG ratchet (the old default), 2.0 the new one. `comp_ratio` stays
# at its default: the split governs the PROBE, not what is admitted, and probing with the
# complement alone blinds the sketch to the 1-site component (`RSVDpreBE0SiQS.m:339`).
for g in (1.1, 2.0)
    row("bond_update (growth=$g)", gate_run; growth = g)
end
for g in (1.1, 2.0)
    row("lubich      (growth=$g)", mpo_run; growth = g)
end
println()

# ---- the absolute-budget curve, the yardstick the schedule must reach -------
for dex in (1, 2, 4, 8, 16, 64)
    row("bond_update (dex=$dex)", gate_run; dex = dex)
end
row("lubich      (dex=8)", mpo_run; dex = 8)
