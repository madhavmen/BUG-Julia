# Is the "exact" reference in examples/heisenberg_domain_wall.jl actually exact at L=18, t=20?
#
# exact_reference() calls expv_sparse(A, -im*TMAX, v) in ONE shot with Krylov dim m=30.
# A single m-dim Krylov space represents exp(-iHt) only while ||H||*t <~ m. Here ||H||*t is
# O(100), so this checks the one-shot answer against a substepped one that is converged.
#
# Usage (submit to a compute node; never run on the login node):
#   L=18 TMAX=20.0 sbatch scripts/run_julia.sbatch scripts/expv_reference_diag.jl
# Env knobs: L (default 18), TMAX (default 20.0).

include(joinpath(@__DIR__, "..", "benchmarks", "exact_sparse.jl"))
using Printf, LinearAlgebra

const L    = parse(Int,     get(ENV, "L",    "18"))
const TMAX = parse(Float64, get(ENV, "TMAX", "20.0"))

A, states, idx = heisenberg_sparse(L)
v0 = domain_wall_vector(L, states, idx)
@printf("L=%d  sector dim=%d  nnz=%d\n", L, length(states), nnz(A))

# crude spectral radius, to show what ||H||*t the one-shot call is being asked to cover
let w = randn(ComplexF64, length(states)); w ./= norm(w)
    for _ in 1:200; w = A * w; w ./= norm(w); end
    @printf("||H||_est = %.3f   =>  ||H||*TMAX = %.1f   (one-shot Krylov dim m=30)\n\n",
            norm(A * w), norm(A * w) * TMAX)
end

# the reference AS THE EXAMPLE COMPUTES IT: one shot, m=30
mag_oneshot = magnetisation_dense(expv_sparse(A, -im * TMAX, v0), L, states)

# converged references: substepped, so ||H||*dt per call is small. Two step sizes + two
# Krylov depths; if these agree with each other they are converged.
function substepped(nsteps, m)
    dt = TMAX / nsteps
    v = propagate!(A, v0, ComplexF64(-im * dt), nsteps, (s, x) -> nothing; m = m)
    return magnetisation_dense(v, L, states)
end

mag_400  = substepped(400,  30)
mag_800  = substepped(800,  30)
mag_2000 = substepped(2000, 20)

@printf("substep self-consistency  |400 vs 800|   = %.3e\n", maximum(abs.(mag_400 - mag_800)))
@printf("substep self-consistency  |800 vs 2000|  = %.3e   <- converged reference\n\n",
        maximum(abs.(mag_800 - mag_2000)))

@printf("ONE-SHOT (what the example calls 'exact') vs converged = %.3e\n",
        maximum(abs.(mag_oneshot - mag_2000)))

@printf("\n   %-4s %-11s %-11s %-11s\n", "site", "one-shot", "converged", "diff")
for j in 1:L
    @printf("   %-4d %+11.5f %+11.5f %+11.5f\n",
            j, mag_oneshot[j], mag_2000[j], mag_oneshot[j] - mag_2000[j])
end

# also: how many substeps does a one-shot call need before it stops moving?
println("\none-shot convergence in the number of chunks (m=30 each):")
for nch in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    mg = substepped(nch, 30)
    @printf("   %4d chunk(s)  (||H||*dt/chunk ~ %6.2f)   err vs converged = %.3e\n",
            nch, 8.0 * TMAX / nch, maximum(abs.(mg - mag_2000)))
end
