# Companion data for the L=6 acceptance gate: where does the error sit?
using LinearAlgebra, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("../tests/common/dense_reference.jl")

L, DMAX = 6, 8
H = dense_heisenberg(L)

@printf("%6s %5s | %12s %12s | %10s %10s | %s\n",
        "dt", "n", "vs Trotter", "vs exact", "max disc", "norm-1", "bond dims")
for (dt, n) in ((0.01, 5), (0.005, 10), (0.01, 20), (0.05, 10))
    psi = domain_wall_state(L)
    v0 = dense_state(psi)
    g = bond_gates(psi)
    info = bond_update_bug!(psi, g; opts = BondUpdateOptions(
        dt = dt, n_steps = n, order = :strang, maxdim = DMAX,
        trunc_thresh = 1e-14, normalize = false))
    got = dense_state(psi)
    tro = dense_trotter_propagate(L, v0, dt, n; order = :strang)
    exa = dense_exact_propagate(H, v0, dt * n)
    @printf("%6.3f %5d | %12.3e %12.3e | %10.2e %10.2e | %s\n",
            dt, n, norm(got - tro), norm(got - exa),
            maximum(info.discarded), abs(1 - info.norms[end]), info.bond_dims[end])
end
