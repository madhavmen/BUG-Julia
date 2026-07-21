# Does the Sulz budget starve the missing-QN fill?
# budget = r - n_perp, so once the complement is full rank the fill gets nothing
# and charge sectors that only the fill can open stay closed forever.
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG

L = 6
psi = domain_wall_state(L); canonical!(psi, 1)
g = bond_gates(psi)
rng = MersenneTwister(0x5EED)

@printf("%4s %-5s %4s | %3s %6s %7s %7s %7s | %s\n",
        "step", "par", "bond", "r", "n_perp", "budget", "missing", "seeded", "aug_k")
for step in 1:5, (parity, frac) in ((:even, 0.5), (:odd, 1.0), (:even, 0.5))
    for i in parity_bonds(L, parity)
        canonical!(psi, i)
        f = bond_frame(psi, i)
        tau = ComplexF64(-im * 0.01 * frac)
        tl, tr = f.site_l.itags, f.site_r.itags
        K0 = to_concrete(f.U0 * f.S0)
        K1 = expv(x -> perp_component(f.U0, to_concrete(
                      apply_gate(g[i], to_concrete(x * f.V0), tl, tr) * f.V0')),
                  tau, K0; hermitian = false, maxiter = 30, tol = 1e-15)
        # complement size with no fill at all
        _, n_perp = augmented_left_isometry(f.U0, K1; missing_fill = 0, rng = rng)
        rep = sector_report(f.U0, K1)
        pl, _ = pairable_charges(f)
        nmiss = count(r -> r.missing && r.charge in pl, rep)
        r0 = f.old_rank
        res = kls_bond_update(f, g[i], tau; maxdim = 8, trunc_thresh = 1e-14, rng = rng)
        step in (1, 3, 5) && @printf("%4d %-5s %4d | %3d %6d %7d %7d %7d | %d\n",
                step, string(parity), i, r0, n_perp, r0 - n_perp, nmiss,
                res.n_new_k - n_perp, res.aug_k)
        psi[i] = res.left_core; psi[i + 1] = res.right_core; psi.center = i + 1
    end
end
println("\nfinal bond dims = ", bond_dims(psi))
