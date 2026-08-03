# Second symptom: "SAME PHYSICS CHECK ... !! DIVERGED" (1.14e-2 at chi=16, 3.19e-5 at chi=64).
#
# The :none vs :U1 check has a 1e-10 threshold. That threshold is only meaningful when neither
# run is truncating: once the bond hits the cap, the two code paths discard DIFFERENT directions
# (dense SVD vs per-sector block SVD), so they land on genuinely different states. This scans
# chi to see whether the gap is truncation-limited (shrinks with chi) or a real bug (does not).
#
# It also reports each run against a CONVERGED reference (substepped Krylov), not the one-shot
# reference the example uses.
#
# Usage (submit to a compute node; never run on the login node):
#   L=18 TMAX=20.0 DT=0.05 CHIS=16,32,64,128,192 \
#     sbatch scripts/run_julia.sbatch scripts/chi_scan_diag.jl
# Env knobs: L (18), TMAX (20.0), DT (0.05), CHIS (comma list, "16,32,64,128,192").

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime
using LinearAlgebra, Printf

include(joinpath(@__DIR__, "..", "benchmarks", "exact_sparse.jl"))

const L    = parse(Int,     get(ENV, "L",    "18"))
const DT   = parse(Float64, get(ENV, "DT",   "0.05"))
const TMAX = parse(Float64, get(ENV, "TMAX", "20.0"))
const CHIS = [parse(Int, s) for s in split(get(ENV, "CHIS", "16,32,64,128,192"), ",")]
const NSTEP = round(Int, TMAX / DT)

# --- converged reference: substepped, so ||H||*dt per Krylov call is small -----------
A, states, idx = heisenberg_sparse(L)
v0 = domain_wall_vector(L, states, idx)
ref(nsteps) = magnetisation_dense(
    propagate!(A, v0, ComplexF64(-im * TMAX / nsteps), nsteps, (s, x) -> nothing; m = 30),
    L, states)
r1, r2 = ref(800), ref(2000)
@printf("converged reference: |800 vs 2000 substeps| = %.3e\n", maximum(abs.(r1 - r2)))
exact_good = r2
exact_bad  = magnetisation_dense(expv_sparse(A, -im * TMAX, v0), L, states)  # what the example uses
@printf("one-shot reference (example's 'exact') vs converged = %.3e\n\n",
        maximum(abs.(exact_bad - exact_good)))

function run(sym, chi)
    set_symmetry!(sym)
    psi   = domain_wall_state(L)
    gates = bond_gates(psi; J = 1.0, delta = 1.0)
    opts  = CBEBugOptions(; dt = DT, n_steps = NSTEP, maxdim = chi,
                            s_iters = -1, maxiter = 8, exact = false)
    info = RealTime.evolve!(psi, gates; opts = opts)
    return magnetisation(copy(psi)), maximum(bond_dims(psi)), info
end

@printf("%-6s %-8s %-8s %-12s %-12s %-12s\n",
        "chi", "max_bd", "capped?", "err_vs_good", "err_vs_oneshot", "none_vs_U1")
for chi in CHIS
    mn, bn, _ = run(:none, chi)
    mu, bu, _ = run(:U1,   chi)
    @printf("%-6d %-8d %-8s %-12.3e %-12.3e %-12.3e\n",
            chi, bn, bn >= chi ? "YES" : "no",
            maximum(abs.(mu - exact_good)), maximum(abs.(mu - exact_bad)),
            maximum(abs.(mn - mu)))
    flush(stdout)
end
