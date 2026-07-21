# Does the sweep account for the symmetry, or does it open charge sectors that
# can never carry weight?
#
# A left sector q_l can only hold amplitude if the right frame supplies a
# partner q_r with q_l + q_r equal to the two-site tensor's own charge. If the
# missing-QN seed opens left sectors with no such partner, those directions are
# dead on arrival: the S-step leaves them exactly zero and the truncating SVD
# throws them away -- after they have already consumed maxdim budget.
#
# Real time only.
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG

secs(t, leg) = [q for (q, _) in t.spaces[leg]]
dims(t, leg) = Dict(q => d for (q, d) in t.spaces[leg])

L = 6
tau = ComplexF64(-0.1im)
psi = domain_wall_state(L); canonical!(psi, 1)
g = bond_gates(psi)
rng = MersenneTwister(0x5EED)

println("real-time sweeps, tau = $tau, L = $L, maxdim = 64\n")

for sweep in 1:6, parity in (:even, :odd)
    for i in parity_bonds(L, parity)
        canonical!(psi, i)
        f = bond_frame(psi, i)
        r = kls_bond_update(f, g[i], tau; maxdim = 64, trunc_thresh = 1e-14, rng = rng)

        syms = symm(r.U_aug)
        ldir = r.U_aug.inds[3].dir
        rdir = r.V_aug.inds[1].dir
        # partner test: does V_aug carry the dual of this left sector?
        rsec = Set(align_charge(syms, q, rdir, ldir) for q in secs(r.V_aug, 1))
        aug = secs(r.U_aug, 3)
        kept = Set(secs(r.left_core, 3))
        orphan = [q for q in aug if !(q in rsec)]
        pruned = [q for q in aug if !(q in kept)]

        @printf("swp %d %-4s bond %d | r=%d aug_k=%d aug_l=%d keep=%d | orphan_l=%d pruned=%d",
                sweep, string(parity), i, f.old_rank, r.aug_k, r.aug_l, r.keep,
                length(orphan), length(pruned))
        isempty(orphan) || print("  ORPHANS=", orphan)
        isempty(pruned) || print("  PRUNED=", pruned)
        println()

        psi[i] = r.left_core
        psi[i + 1] = r.right_core
        psi.center = i + 1
    end
end

println("\nfinal bond dims: ", bond_dims(psi))
println("final energy:    ", energy(psi, g))
println("total Sz:        ", total_sz(psi))
