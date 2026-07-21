# Is BUG exact at L=6 when the frames are COMPLETE, or is 4e-10 a defect?
#
# If every bond already sits at its ambient rank then U0 alone spans the whole
# local space, the augmented frame is complete, and the Galerkin S-step must
# reproduce the exact two-site propagator to machine precision. If it does, the
# 4e-10 seen from a product state is ordinary BUG projection error on frames that
# are NOT complete -- method, not bug.
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("../tests/common/dense_reference.jl")

L, DMAX = 6, 8
H = dense_heisenberg(L)

ambient(psi, i) = min(leg_dim(psi[i], 1) * 2, 2 * leg_dim(psi[i + 1], 3))

println("=== frame completeness during a cold run (dt=0.01) ===")
psi = domain_wall_state(L); canonical!(psi, 1)
g = bond_gates(psi)
rng = MersenneTwister(0x5EED)
for step in 1:5
    for (parity, frac) in ((:even, 0.5), (:odd, 1.0), (:even, 0.5))
        for i in parity_bonds(L, parity)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            r = kls_bond_update(f, g[i], ComplexF64(-im * 0.01 * frac);
                                maxdim = DMAX, trunc_thresh = 1e-14, rng = rng)
            aL = leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
            aR = leg_dim(f.V0, 2) * leg_dim(f.V0, 3)
            step == 5 && @printf("  bond %d  r=%d  aug_k=%d/%d  aug_l=%d/%d  keep=%d  %s\n",
                    i, f.old_rank, r.aug_k, aL, r.aug_l, aR, r.keep,
                    (r.aug_k == aL && r.aug_l == aR) ? "COMPLETE" : "incomplete")
            psi[i] = r.left_core; psi[i + 1] = r.right_core; psi.center = i + 1
        end
    end
end

println("\n=== accuracy from a WARM (full-rank) start ===")
# drive to full rank first
warm = domain_wall_state(L)
gw = bond_gates(warm)
bond_update_bug!(warm, gw; opts = BondUpdateOptions(
    dt = 0.05, n_steps = 10, order = :strang, maxdim = DMAX,
    trunc_thresh = 1e-14, normalize = false))
println("  warm bond dims = ", bond_dims(warm), "   (ambient = [2,4,8,4,2])")

for (dt, n) in ((0.01, 5), (0.005, 10))
    p = deepcopy(warm)
    v0 = dense_state(p)
    gp = bond_gates(p)
    bond_update_bug!(p, gp; opts = BondUpdateOptions(
        dt = dt, n_steps = n, order = :strang, maxdim = DMAX,
        trunc_thresh = 1e-14, normalize = false))
    tro = dense_trotter_propagate(L, v0, dt, n; order = :strang)
    @printf("  dt=%.3f n=%2d  vs Trotter = %.3e   bond dims %s\n",
            dt, n, norm(dense_state(p) - tro), bond_dims(p))
end
