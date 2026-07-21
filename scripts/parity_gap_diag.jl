using LinearAlgebra, Printf, JSON, LurCGT, Telum
using BUGJulia.BondUpdateBUG
ref = JSON.parsefile("/home/madhav.menon/BUG-Julia/tests/crosscheck/reference_l6_heisenberg.json")
L = Int(ref["L"]); want = Float64.(ref["sz_final"])
psi = domain_wall_state(L)
g = bond_gates(psi; J = 1.0, delta = Float64(ref["delta"]))
bond_update_bug!(psi, g; opts = BondUpdateOptions(
    dt = Float64(ref["dt"]), n_steps = Int(ref["n_steps"]), order = :strang,
    maxdim = Int(ref["maxdim"]), trunc_thresh = Float64(ref["trunc_thresh"]),
    normalize = false))
got = [sz_expectation(psi, j) for j in 1:L]
@printf("max |julia - python| = %.4e\n", maximum(abs.(got .- want)))
