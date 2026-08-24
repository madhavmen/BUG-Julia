# IS THE SELECTION GOVERNED BY A TOLERANCE, OR BY A COUNT? (Jan's points 2 and 3)
#
# THE ASK: "expand only with singular vectors whose singular values are above some specified
# tolerance, say tau_trunc. Make sure all errors are governed by tau_trunc throughout." And,
# separately: "At first, use full SVD rather than rSVD -- it eliminates one source of uncertainty
# and ensures full control of the truncation threshold."
#
# WHAT THE CODE DOES NOW. Two different mechanisms decide what is admitted, and only one of them
# is a tolerance:
#
#   1. `stol_pre`  the cutoff of the PRESELECTION SVD, which is where a direction first has to
#                  earn its place. Defaults to 1e-10.
#   2. `stol_fnl`  the cutoff of the FINAL-SELECTION SVD of `UMU = QL'(H*Theta)QR`. Defaults 1e-13.
#   3. `dex`       a COUNT -- `budget = ceil(growth*dmax) - r`, `growth = 2.0`. Not a tolerance.
#
# ⛔ THE COUNT IS WHAT BINDS, AND IT IS NOT WEIGHT-RANKED WHERE IT BINDS. When a side's budget
# exceeds the joint ranking's reach, `cbe_expand` falls back to `_trim_total(Q, leg, dex)`, which
# walks the charge sectors of `Q` in iteration order and takes `min(d, left)` columns from each.
# The code comment says the candidates "are ALREADY weight-ordered (the preselection SVD's
# singular values are the weights)". THAT IS TRUE OF `res`, BUT `res.U` IS NOT WHAT IS RETURNED:
# `_preselect_left` runs a SECOND SVD for `RepairOrtho` and returns THAT `.U`, whose singular
# values are all ~1 because it is orthonormalising an already-orthonormal `Q`. So the column order
# reaching `_trim_total` carries no weight information at all.
#
# ⚠️ THAT IS A CODE READING, NOT A MEASUREMENT, AND IT IS ONLY A LIVE BUG IF THE COUNT ACTUALLY
# BINDS. If `stol_pre` is raised to `tau_trunc`, everything that survives preselection is already
# above tolerance and the count has nothing left to choose badly -- which is exactly Jan's fix,
# and is why this file tests the fix rather than instrumenting the defect.
#
# THE GATE, and it is stated in the plan: under a tolerance-driven rule `err_fnl` and `err_pre`
# must both fall to `<~ tau_trunc`. `err_fnl` is already documented in `CBEBugSweepInfo` as "the
# honest measure of what the budget cost", so the instrument exists; it has just never been read
# against the tolerance line.
#
# ⛔ EVERY ARM RUNS `exact = true` (Jan's point 3) SO THE SKETCH IS NOT A SECOND VARIABLE. One
# arm at the end re-runs the best tolerance-driven configuration with `exact = false` to price
# the sketch separately, which is the only honest way to read point 3.
#
# Run:  julia --project=. benchmarks/tolerance_driven_selection.jl

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "analytic_reference.jl"))
include(joinpath(@__DIR__, "exact_sparse.jl"))

const L    = 12
const DT   = 0.05
const TMAX = 1.0
const MI   = 16
const CAP  = 64                     # loose: the tau_trunc scan is meaningless if maxdim binds

const TAUS = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10]

"""
Evolve to `TMAX` and report the error against `exact_obs` plus the WORST per-step selection
errors, which are what the tolerance is supposed to govern.

`err_pre` / `err_fnl` are per-step maxima rather than means: a tolerance claim is about the worst
step, and a mean would let one bad step hide behind fifty good ones.
"""
function run_arm(mpo, psi0, obs, exact_obs; tau_trunc, tol_driven::Bool, exact::Bool = true)
    psi, kry = copy(psi0), 0
    epre, efnl = 0.0, 0.0
    # TOLERANCE-DRIVEN: the CBE-internal SVDs are cut at tau_trunc, and `growth` is raised so the
    # COUNT cannot be what binds. BUDGET-DRIVEN: the shipped defaults.
    sel = tol_driven ? (stol_pre = tau_trunc, stol_fnl = tau_trunc, growth = 8.0) :
                       (stol_pre = 1e-10,     stol_fnl = 1e-13,     growth = 2.0)
    for _ in 1:round(Int, TMAX / DT)
        info = cbe_bug_step!(psi, mpo, ComplexF64(-im * DT);
                             exact = exact, maxdim = CAP, trunc_thresh = tau_trunc,
                             maxiter = MI, sel...)
        kry += info.krylov_dims
        epre = max(epre, info.err_pre); efnl = max(efnl, info.err_fnl)
    end
    return (err = maximum(abs.(obs(psi) .- exact_obs)), chi = maximum(bond_dims(psi)),
            kry = kry, epre = epre, efnl = efnl)
end

_sz(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())

function report(title, mpo, psi0, obs, exact_obs)
    println("\n" * "="^96)
    println(title)
    println("="^96)
    @printf("  %-12s %-10s %13s %6s %9s %12s %12s %s\n",
            "rule", "tau_trunc", "err", "chi", "krylov", "err_pre", "err_fnl", "gate")
    for tt in TAUS, (nm, td) in (("budget", false), ("TOLERANCE", true))
        r = run_arm(mpo, psi0, obs, exact_obs; tau_trunc = tt, tol_driven = td)
        # THE GATE: both selection errors under the tolerance line. A rule that admits by count
        # has no reason to satisfy it; a rule that admits by tolerance must.
        ok = r.epre <= 10 * tt && r.efnl <= 10 * tt
        @printf("  %-12s %-10.0e %13.4e %6d %9d %12.3e %12.3e %s\n",
                nm, tt, r.err, r.chi, r.kry, r.epre, r.efnl, ok ? "PASS" : "over the line")
        flush(stdout)
    end
    # POINT 3, PRICED SEPARATELY: same tolerance-driven configuration, randomised sketch on.
    println("  " * "-"^92)
    for tt in (1e-6, 1e-10)
        a = run_arm(mpo, psi0, obs, exact_obs; tau_trunc = tt, tol_driven = true, exact = true)
        b = run_arm(mpo, psi0, obs, exact_obs; tau_trunc = tt, tol_driven = true, exact = false)
        @printf("  sketch @ %-6.0e  full SVD %11.4e (chi %d)   rSVD %11.4e (chi %d)   ratio %.3g\n",
                tt, a.err, a.chi, b.err, b.chi, b.err / max(a.err, 1e-300))
        flush(stdout)
    end
end

set_symmetry!(:U1)
let
    H, states, idx = heisenberg_sparse(L)
    v = expv_sparse(H, ComplexF64(-im * TMAX), neel_vector(L, idx); m = 30, tol = 1e-13)
    report("Heisenberg XXX, L=$L, Neel quench, dt=$DT  (Jan's point 7 puts this model first)",
           xxz_mpo(L; J = 1.0, delta = 1.0), neel_state(L), _sz,
           magnetisation_dense(v, L, states))
end

println("\nREAD IT AS: the TOLERANCE rows must show err_pre and err_fnl tracking tau_trunc down.")
println("If the budget rows sit at a FLOOR instead, the count is what governs them -- which is")
println("the whole of Jan's point 2, and the reason the error stops responding to tau_trunc.")
