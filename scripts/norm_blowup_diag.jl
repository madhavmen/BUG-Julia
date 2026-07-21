# Where does the norm blow up: inside a bond update, or between them?
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG

psi = domain_wall_state(6); canonical!(psi, 1)
g = bond_gates(psi)
rng = MersenneTwister(0x5EED)

println("step par bnd |  n(psi)in   n(psi)out |  n(th0)    n(th1)  |  n(S0)   n(Sst)  n(Snew) | Udef     Vdef     LCdef   | r aug_k keep")
for step in 1:3, (parity, frac) in ((:even, 0.5), (:odd, 1.0), (:even, 0.5))
    for i in parity_bonds(6, parity)
        canonical!(psi, i)
        nin = norm(psi)
        f = bond_frame(psi, i)
        th0 = frame_theta(f)
        r = kls_bond_update(f, g[i], ComplexF64(-im * 0.02 * frac);
                            maxdim = 8, trunc_thresh = 1e-12, rng = rng)
        th1 = to_concrete(r.left_core * r.right_core)
        Sst = to_concrete(contract(contract(r.U_aug', (1, 2), th0, (1, 2)),
                                   (2, 3), r.V_aug', (2, 3)))
        psi[i] = r.left_core; psi[i + 1] = r.right_core; psi.center = i + 1
        @printf("%4d %-4s %3d | %9.4f %9.4f | %8.5f %8.5f | %7.4f %7.4f %7.4f | %.2e %.2e %.2e | %d %d %d\n",
                step, string(parity), i, nin, norm(psi), norm(th0), norm(th1),
                norm(f.S0), norm(Sst), norm(r.S_new),
                left_isometry_defect(r.U_aug), right_isometry_defect(r.V_aug),
                left_isometry_defect(r.left_core),
                f.old_rank, r.aug_k, r.keep)
    end
end
