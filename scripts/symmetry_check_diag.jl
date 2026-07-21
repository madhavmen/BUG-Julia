# Is the test model actually exercising U(1), and does the accidental SU(2) at
# delta=1 change the picture?
#
#   delta = 1.0  -> isotropic Heisenberg: U(1) AND SU(2). Degenerate multiplets.
#   delta = 0.5  -> XXZ: U(1) only. No accidental degeneracy.
#   delta = 0.0  -> XX free fermions: U(1) only.
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("../tests/common/dense_reference.jl")

function report(delta)
    L = 6
    H = dense_heisenberg(L; delta = delta)

    # how much of the Sz=0 sector does the spectrum actually resolve?
    ev = sort(real(eigvals(Hermitian(Matrix(H)))))
    degen = count(k -> abs(ev[k + 1] - ev[k]) < 1e-10, 1:(length(ev) - 1))

    for (dt, n) in ((0.01, 5), (0.05, 10))
        psi = domain_wall_state(L)
        v0 = dense_state(psi)
        g = bond_gates(psi; delta = delta)
        info = bond_update_bug!(psi, g; opts = BondUpdateOptions(
            dt = dt, n_steps = n, order = :strang, maxdim = 8,
            trunc_thresh = 1e-14, normalize = false))
        tro = dense_trotter_propagate(L, v0, dt, n; delta = delta, order = :strang)
        exa = dense_exact_propagate(H, v0, dt * n)
        got = dense_state(psi)
        # charge sectors carried by the centre bond
        nsec = length(psi[3].spaces[3])
        @printf("d=%.1f dt=%.3f n=%2d | vs Trot %.3e | vs exact %.3e | bd %s | %d sectors on bond 3\n",
                delta, dt, n, norm(got - tro), norm(got - exa),
                string(info.bond_dims[end]), nsec)
    end
    @printf("        -> %d/%d degenerate eigenvalue pairs in the full spectrum\n\n",
            degen, length(ev) - 1)
end

for d in (1.0, 0.5, 0.0)
    report(d)
end
