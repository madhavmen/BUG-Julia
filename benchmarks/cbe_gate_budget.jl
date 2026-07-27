# How much expansion budget does the gate-based CBE-BUG actually need?
#
# NOW A LEGITIMATE QUESTION, and it was not before. Every earlier tuning measurement was
# taken on a scheme that did not expand its bonds correctly and is void. The gate path is
# different: the CBE selection now drives the residual of the gate's action on every bond to
# machine zero (`tests/RSVDCBEBondUpdate/test_cbe_gate.jl`), rank tracks the validated
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

# ---- gate-CBE across the budget ---------------------------------------------
# `dex = 0` is the neighbourhood-coupled growth schedule (`ceil(1.1*dmax) - r`); the rest are
# fixed per-side budgets. `comp_ratio` stays at its default: the split governs the PROBE, not
# what is admitted, and probing with the complement alone blinds the sketch to the 1-site
# component (`RSVDpreBE0SiQS.m:339`).
for dex in (0, 1, 2, 4, 8, 16, 64)
    pc = domain_wall_state(L)
    t = @elapsed ic = cbe_gate_bug!(pc, xx_gates(pc);
                                    opts = CBEBugOptions(dt = DT, n_steps = NSTEPS,
                                                         maxdim = MAXDIM,
                                                         trunc_thresh = TRUNC, dex = dex))
    label = dex == 0 ? "cbe_gate (schedule)" : "cbe_gate (dex=$dex)"
    @printf("%-22s  err %.3e   maxbond %3d   aug %3d   %6.1fs   err_pre %.2e  err_fnl %.2e\n",
            label, err_of(pc), maximum(ic.max_bond_dims), maximum(ic.max_expanded), t,
            maximum(ic.err_pre), maximum(ic.err_fnl))
end
