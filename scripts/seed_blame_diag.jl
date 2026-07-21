# A: is the step-refinement anomaly caused by the missing-QN seed?
# B: what API gives the left/right charge-pairing constraint?
# Real time only.
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG

const L = 6

function warm(nsweeps, tau)
    psi = domain_wall_state(L); canonical!(psi, 1)
    g = bond_gates(psi)
    for _ in 1:nsweeps, p in (:even, :odd)
        parity_sweep!(psi, g, p, ComplexF64(-tau * im); maxdim = 64,
                      trunc_thresh = 1e-14, rng = MersenneTwister(0x5EED))
    end
    return psi, g
end

function drift(tau, nsteps, fill)
    psi, g = warm(4, 0.05)                     # build some rank first
    e0 = energy(psi, g)
    for _ in 1:nsteps, p in (:even, :odd)
        parity_sweep!(psi, g, p, ComplexF64(-tau * im); maxdim = 64,
                      trunc_thresh = 1e-14, missing_fill = fill,
                      rng = MersenneTwister(0x5EED))
    end
    return abs(energy(psi, g) - e0), bond_dims(psi)
end

println("=== A. energy drift over total time 0.4, warm start ===")
@printf("%-6s %-6s %-6s %14s   %s\n", "fill", "tau", "steps", "|dE|", "bond dims")
for fill in (1, 0), (tau, n) in ((0.04, 10), (0.02, 20), (0.01, 40))
    d, bd = drift(tau, n, fill)
    @printf("%-6d %-6.3f %-6d %14.3e   %s\n", fill, tau, n, d, bd)
end

println("\n=== B. charge-pairing API ===")
psi, _ = warm(3, 0.05)
canonical!(psi, 3)
f = bond_frame(psi, 3)
println("  S0 legs: ", [(ix.itags, ix.dir) for ix in f.S0.inds])
println("  S0 spaces[1] = ", f.S0.spaces[1])
println("  S0 spaces[2] = ", f.S0.spaces[2])
println("  S0 qlabels   = ", f.S0.qlabels)
println("  U0 bond dir=", f.U0.inds[3].dir, " spaces=", f.U0.spaces[3])
println("  V0 bond dir=", f.V0.inds[1].dir, " spaces=", f.V0.spaces[1])

FL = fusion_basis(f.U0, 1, 2; tag = "fL")
FR = fusion_basis(f.V0, 2, 3; tag = "fR")
println("  reachable LEFT  = ", FL.spaces[3])
println("  reachable RIGHT = ", FR.spaces[3])

th = frame_theta(f)
P = to_concrete(contract(contract(FL', (1, 2), th, (1, 2)), (2, 3), FR', (2, 3)))
println("  projected P legs   = ", [(ix.itags, ix.dir) for ix in P.inds])
println("  projected P qlabels= ", P.qlabels)
println("  -> allowed (qL,qR) pairs, i.e. the partner constraint")
