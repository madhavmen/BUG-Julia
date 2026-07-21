# Is the accuracy gap caused by charge sectors the state never opens?
#
# For each bond: which sectors are reachable AND pairable, which the kept bond
# actually carries, and which are absent. Then does widening the fill close the
# gap in the measured error?
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("../tests/common/dense_reference.jl")

function sector_census(fill)
    L = 6
    psi = domain_wall_state(L); canonical!(psi, 1)
    g = bond_gates(psi)
    rng = MersenneTwister(0x5EED)
    for step in 1:5, (parity, frac) in ((:even, 0.5), (:odd, 1.0), (:even, 0.5))
        for i in parity_bonds(L, parity)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            r = kls_bond_update(f, g[i], ComplexF64(-im * 0.01 * frac);
                                maxdim = 8, trunc_thresh = 1e-14,
                                missing_fill = fill, rng = rng)
            if step == 5
                pl, _ = pairable_charges(f)
                kept = Set(q for (q, _) in r.left_core.spaces[3])
                absent = [q for q in pl if !(q in kept)]
                reach = Set(q for (q, _) in fusion_basis(f.U0, 1, 2).spaces[end])
                @printf("  %-4s bond %d: reachable %d, pairable %d, kept %d, absent %s | aug_k=%d keep=%d disc=%.1e\n",
                        string(parity), i, length(reach), length(pl), length(kept),
                        string(absent), r.aug_k, r.keep, r.discarded)
            end
            psi[i] = r.left_core; psi[i + 1] = r.right_core; psi.center = i + 1
        end
    end
    return bond_dims(psi)
end

for fill in (1, 2, 4)
    println("=== missing_fill = $fill ===")
    bd = sector_census(fill)
    println("  final bond dims = ", bd)
end

println("\n=== does widening the fill improve the ERROR? ===")
L = 6; H = dense_heisenberg(L)
for fill in (0, 1, 2, 4, 8)
    psi = domain_wall_state(L)
    v0 = dense_state(psi)
    g = bond_gates(psi)
    info = bond_update_bug!(psi, g; opts = BondUpdateOptions(
        dt = 0.01, n_steps = 5, order = :strang, maxdim = 8,
        trunc_thresh = 1e-14, normalize = false, missing_fill = fill))
    tro = dense_trotter_propagate(L, v0, 0.01, 5; order = :strang)
    @printf("  fill=%d | vs Trotter %.4e | bd %s | max aug_k %d\n",
            fill, norm(dense_state(psi) - tro), string(info.bond_dims[end]),
            maximum(info.aug_k_dims))
end
