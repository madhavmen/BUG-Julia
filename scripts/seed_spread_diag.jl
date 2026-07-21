# How much of the Python-Julia gap is just the random missing-QN fill?
# If changing only the Julia seed moves the profile by about as much as the
# Python comparison does, the gap is the fill and nothing else needs explaining.
using LinearAlgebra, Printf, JSON, LurCGT, Telum
using BUGJulia.BondUpdateBUG

ref = JSON.parsefile("/home/madhav.menon/BUG-Julia/tests/crosscheck/reference_l6_heisenberg.json")
L = Int(ref["L"]); want = Float64.(ref["sz_final"])

function profile(seed; fill = 1)
    psi = domain_wall_state(L)
    g = bond_gates(psi; J = 1.0, delta = Float64(ref["delta"]))
    bond_update_bug!(psi, g; opts = BondUpdateOptions(
        dt = Float64(ref["dt"]), n_steps = Int(ref["n_steps"]), order = :strang,
        maxdim = Int(ref["maxdim"]), trunc_thresh = Float64(ref["trunc_thresh"]),
        normalize = false, missing_fill = fill, seed = seed))
    return [sz_expectation(psi, j) for j in 1:L]
end

seeds = (0x5EED, 0x1234, 0xABCD, 0x0001)
ps = [profile(s) for s in seeds]
println("=== Julia seed-to-seed spread (same algorithm, different fill directions) ===")
for (i, s) in enumerate(seeds)
    @printf("  seed 0x%04X | vs python %.3e | vs seed[1] %.3e\n",
            s, maximum(abs.(ps[i] .- want)), maximum(abs.(ps[i] .- ps[1])))
end
@printf("  max pairwise Julia spread: %.3e\n",
        maximum(maximum(abs.(a .- b)) for a in ps, b in ps))

println("\n=== with the fill disabled entirely ===")
p0 = profile(0x5EED; fill = 0)
@printf("  fill=0 | vs python %.3e | vs fill=1 %.3e\n",
        maximum(abs.(p0 .- want)), maximum(abs.(p0 .- ps[1])))
