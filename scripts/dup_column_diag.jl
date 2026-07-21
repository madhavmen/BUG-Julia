# U_aug has a duplicated column. Which block supplies it?
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG

psi = domain_wall_state(6); canonical!(psi, 1)
g = bond_gates(psi)
rng = MersenneTwister(0x5EED)

# replay exactly up to the first bad update: step 2, odd sweep, bond 2
sched = [(1, :even, 0.5), (1, :odd, 1.0), (1, :even, 0.5), (2, :even, 0.5)]
for (_, parity, frac) in sched, i in parity_bonds(6, parity)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    r = kls_bond_update(f, g[i], ComplexF64(-im * 0.02 * frac);
                        maxdim = 8, trunc_thresh = 1e-12, rng = rng)
    psi[i] = r.left_core; psi[i + 1] = r.right_core; psi.center = i + 1
end

i = 2
canonical!(psi, i)
f = bond_frame(psi, i)
tau = ComplexF64(-im * 0.02 * 1.0)

# rebuild the K-step by hand so the pieces are visible
K0 = to_concrete(f.U0 * f.S0)
tl, tr = f.site_l.itags, f.site_r.itags
function apply_gk(x)
    th = to_concrete(x * f.V0)
    ev = apply_gate(g[i], th, tl, tr)
    return perp_component(f.U0, to_concrete(ev * f.V0'))
end
K1 = expv(apply_gk, tau, K0; hermitian = false, maxiter = 30, tol = 1e-15)

println("U0 bond spaces : ", f.U0.spaces[3], "  dir=", f.U0.inds[3].dir)
println("K1 bond spaces : ", K1.spaces[3], "  dir=", K1.inds[3].dir)
println("||K1|| = ", norm(K1), "   U0 isometry defect = ", left_isometry_defect(f.U0))

perp = perp_component(f.U0, K1)
println("||perp|| = ", norm(perp))
Q = to_concrete(svd(perp, (1, 2); cutoff = 1e-12).U)
println("Q bond spaces  : ", Q.spaces[3], "  dir=", Q.inds[3].dir)
println("Q isometry defect = ", left_isometry_defect(Q))
println("||U0' Q||  (should be 0) = ",
        norm(to_concrete(contract(f.U0', (1, 2), to_concrete(setitag(Q, 3, f.U0.inds[3].itags)), (1, 2)))))

println("\nsector_report (charge, reachable, u0, k1, range, missing):")
for r in sector_report(f.U0, K1)
    println("  ", r.charge, "  reach=", r.reachable_dim, " u0=", r.u0_cols,
            " k1=", r.k1_cols, " range=", r.range_cols, " MISSING=", r.missing)
end

pl, pr = pairable_charges(f)
println("\npairable left = ", pl)

U_aug, n_new = augmented_left_isometry(f.U0, K1;
                                       missing_fill = 1, seed_charges = pl,
                                       rng = MersenneTwister(0x5EED))
println("\nU_aug spaces = ", U_aug.spaces[3], "  n_new=", n_new,
        "  defect=", left_isometry_defect(U_aug))

println("\nsame, with the seed disabled:")
U_ns, n_ns = augmented_left_isometry(f.U0, K1; missing_fill = 0,
                                     rng = MersenneTwister(0x5EED))
println("U_aug spaces = ", U_ns.spaces[3], "  n_new=", n_ns,
        "  defect=", left_isometry_defect(U_ns))
