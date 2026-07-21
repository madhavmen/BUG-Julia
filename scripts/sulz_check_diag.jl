# Does the augmented rank ever exceed 2r over a real run?
using LinearAlgebra, Random, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG

# NOTE: wrapped in a function. Julia's soft scope turns counters assigned inside
# a top-level `for` into fresh locals -- gotcha 11 in docs/telum_api_contract.md,
# which I walked into again writing this.
function check(nsteps = 8)
    L = 6
    psi = domain_wall_state(L); canonical!(psi, 1)
    g = bond_gates(psi)
    rng = MersenneTwister(0x5EED)
    worst = 0; nviol = 0; total = 0

    for step in 1:nsteps, (parity, frac) in ((:even, 0.5), (:odd, 1.0), (:even, 0.5))
        for i in parity_bonds(L, parity)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            r = kls_bond_update(f, g[i], ComplexF64(-im * 0.01 * frac);
                                maxdim = 8, trunc_thresh = 1e-14, rng = rng)
            for (lbl, aug) in (("k", r.aug_k), ("l", r.aug_l))
                total += 1
                ex = aug - 2 * f.old_rank
                if ex > 0
                    nviol += 1; worst = max(worst, ex)
                    nviol <= 6 && @printf("  VIOLATION step %d %-4s bond %d: aug_%s=%d > 2r=%d\n",
                                          step, string(parity), i, lbl, aug, 2 * f.old_rank)
                end
            end
            psi[i] = r.left_core; psi[i + 1] = r.right_core; psi.center = i + 1
        end
    end
    @printf("\n%d/%d frames exceeded 2r; worst excess = %d\n", nviol, total, worst)
    println("final bond dims = ", bond_dims(psi))
end

check()
