# DOES THE KRYLOV STOPPING RULE ACTUALLY STOP? (Jan's point 5)
#
# THE ASK: "Keep track of size of elements of the tridiagonal Krylov Hamiltonian. When these drop
# below some threshold, say tau_Krylov, terminate the Krylov scheme. Choose tau_Krylov to be
# related to tau_trunc, say tau_Krylov = tau_trunc/10."
#
# ⛔ THE OBVIOUS READING OF THAT -- stop when the off-diagonal `beta` gets small -- DOES NOT WORK,
# and this file exists because it was tried first. MEASURED at L=12, dt=0.05 with the `beta` rule:
# Heisenberg and XX returned BIT-IDENTICAL results for a threshold anywhere from 1e-6 to 1e-13.
# The test never fired; every bond ran to the cap. `beta` is a BREAKDOWN detector -- it reports
# that the recursion has no independent direction left, which on a generic `H` does not happen --
# so thresholding it measures the cap, not the convergence.
#
# WHAT REPLACED IT is the quantity Saad's estimate uses, which is also what
# `exact_sparse.jl::_lanczos_expv` already computed for its own error bar:
#
#     contribution of vector m+1  =  beta_m * |[exp(tau*T_m)]_{m,1}|
#
# `beta_m` can be O(1) while `[exp(tau*T_m)]_{m,1} ~ (tau*||H||)^m / m!` is tiny -- which is
# exactly why a short basis suffices at small `tau` and a long one is needed at large `tau`. The
# tridiagonal elements ARE tracked, as Jan asked; they are just weighted by what the propagator
# does with them before being compared to a tolerance.
#
# ⛔ THE GATE IS TWO-SIDED AND BOTH SIDES MATTER. A stopping rule that cuts the depth is only good
# if the ERROR DOES NOT MOVE. This file reports both, so a rule that saves cost by giving up
# accuracy is visible as such rather than being read as a win. The reference row is
# `tol = 0` (run to the cap), which is the most accurate configuration available.
#
# ⛔ AND `krylov` IS PER-SWEEP, NOT CUMULATIVE. `CBEBugSweepInfo.krylov_dims` counts one step's
# operator applications and is non-monotonic across a run; any cost axis must ACCUMULATE it.
#
# SIZED FOR A CORRECTNESS QUESTION. L=12 against exact references, which is where "does the rule
# fire, and does it cost accuracy" is answerable. L=20 timings go to the cluster.
#
# Run:  julia --project=. benchmarks/krylov_stopping.jl

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
const CAP  = 64
const MI   = 16

# `tol = 0.0` disables the contribution test, so the arm runs to `basis`. That is the reference
# row: the most accurate configuration the sweep has, and what every other row is judged against.
const ARMS = [
    ("cap 30, no test",   (krylov_basis = 30, krylov_tol = 0.0)),
    ("tol 1e-4",          (krylov_basis = 30, krylov_tol = 1e-4)),
    ("tol 1e-6 (DEFAULT)",(krylov_basis = 30, krylov_tol = 1e-6)),
    ("tol 1e-8",          (krylov_basis = 30, krylov_tol = 1e-8)),
    ("tol 1e-10",         (krylov_basis = 30, krylov_tol = 1e-10)),
    ("pinned m=3",        (krylov_basis = 3,  krylov_tol = 0.0)),
    ("pinned m=0",        (krylov_basis = 0,  krylov_tol = 0.0)),
]

"Evolve to `TMAX`, accumulating the per-step operator count. Returns `(err, chi, krylov)`."
function run_arm(mpo, psi0, obs, exact_obs, kw)
    psi, kry = copy(psi0), 0
    for _ in 1:round(Int, TMAX / DT)
        info = cbe_bug_step!(psi, mpo, ComplexF64(-im * DT);
                             exact = true, maxdim = CAP, trunc_thresh = 1e-12, maxiter = MI,
                             kw...)
        kry += info.krylov_dims          # PER-SWEEP: accumulate or the cost axis is meaningless
    end
    return (err = maximum(abs.(obs(psi) .- exact_obs)),
            chi = maximum(bond_dims(psi)), kry = kry)
end

_sz(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())

function report(title, mpo, psi0, obs, exact_obs)
    println("\n" * "="^78)
    println(title)
    println("="^78)
    @printf("  %-22s %14s %8s %10s %10s\n", "arm", "err", "chi", "krylov", "vs ref")
    ref = nothing
    for (nm, kw) in ARMS
        r = run_arm(mpo, psi0, obs, exact_obs, kw)
        ref === nothing && (ref = r)
        @printf("  %-22s %14.4e %8d %10d %10s\n", nm, r.err, r.chi, r.kry,
                ref === r ? "--" : @sprintf("%.2fx err, %.2fx cost",
                                            r.err / max(ref.err, 1e-300), r.kry / ref.kry))
        flush(stdout)
    end
end

# ── XX: krylov depth is known to be INERT here, so this is the control ────────────────────
set_symmetry!(:U1)
report("XX nearest neighbour, L=$L  (control: depth known inert -- the rule must not COST here)",
       xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L), _sz,
       xx_free_fermion_sz(L, TMAX; J = 1.0, occupied = 1:(L ÷ 2)))

# ── Heisenberg: the model Jan's point 7 puts first ────────────────────────────────────────
set_symmetry!(:U1)
let
    H, states, idx = heisenberg_sparse(L)
    v = expv_sparse(H, ComplexF64(-im * TMAX), neel_vector(L, idx); m = 30, tol = 1e-13)
    report("Heisenberg XXX, L=$L, Neel quench  (exact sparse reference)",
           xxz_mpo(L; J = 1.0, delta = 1.0), neel_state(L), _sz,
           magnetisation_dense(v, L, states))
end

# ── OAT: the model where depth was worth eight orders ─────────────────────────────────────
set_symmetry!(:none)
report("OAT, L=$L  (depth is worth 8 orders here -- the rule must not stop EARLY)",
       oat_mpo(L), x_polarized_state(L),
       p -> [total_sx(copy(p)) / max(norm(copy(p))^2, eps())],
       [oat_total_sx(L, TMAX)])
set_symmetry!(:U1)

println("\nREAD IT AS: a good rule cuts `krylov` on XX/Heisenberg while leaving `err` at the")
println("reference row's value, and does NOT cut on OAT, where the depth is load-bearing.")
