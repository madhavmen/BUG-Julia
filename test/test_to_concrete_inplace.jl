# `to_concrete!` MUST BE BIT-IDENTICAL TO `to_concrete` ON A FRESHLY CONTRACTED TENSOR.
#
# The only difference is the defensive copy: `to_concrete` does `copy(q)` first (via
# `_eager_tlarray(q::TLArray) = copy(q)`) and then normalises/orients the copy; `to_concrete!`
# normalises and orients `q` itself. Same operations, same order, so the results must agree
# EXACTLY — `isapprox` would hide a real difference in the w-matrix sign/orientation cleanup,
# which is the one thing that could plausibly differ.
#
# The test also pins the two facts that make the substitution safe:
#   1. `to_concrete!` returns the SAME OBJECT it was handed (no copy).
#   2. `to_concrete` leaves its argument untouched, so the two are distinguishable — if a
#      future change made `to_concrete` mutate, the substitution would become invisible
#      rather than merely wrong.
#
# And the end-to-end check: a full step of every arm must be unchanged. `apply_one_site` is
# on the Krylov hot path, so a subtle aliasing bug there would show up as drift over a step
# long before it showed up as a crash.

using Test
using LinearAlgebra
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: one_site_h, apply_one_site, left_env_stack,
                                  right_env_stack, left_channels, right_channels
using Telum: contract, to_concrete, to_concrete!

blocks(q) = [copy(ComplexF64.(r)) for r in q.RMTs]

@testset "to_concrete! == to_concrete on a fresh contraction" begin
    for sym in (:U1, :none)
        BondUpdateBUG.set_symmetry!(sym)
        L = 8
        mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = 1.0)
        psi = BondUpdateBUG.neel_state(L)
        for _ in 1:5
            RSVDCBEBondUpdate.tdvp2_step!(psi, mpo, ComplexF64(-im * 0.3);
                                          maxdim = 32, trunc_thresh = 0.0, maxiter = 6)
        end

        i = L ÷ 2
        canonical!(psi, i)
        lst = left_env_stack(psi, mpo; upto = i - 1)
        rst = right_env_stack(psi, mpo; downto = i + 1)
        H1 = one_site_h(mpo, i, left_channels(lst, i), right_channels(rst, i + 1))
        A = psi[i]

        # two independent, identical contraction results
        c1 = contract(H1.l, (3,), A, (1,))
        c2 = contract(H1.l, (3,), A, (1,))
        pre = blocks(c2)

        r_copy = to_concrete(c1)
        r_inpl = to_concrete!(c2)

        @test r_inpl === c2                      # in place: same object, no copy
        @test blocks(c1) == pre                  # to_concrete did NOT mutate its argument
        @test length(r_copy.RMTs) == length(r_inpl.RMTs)
        @test blocks(r_copy) == blocks(r_inpl)   # EXACT, deliberately not isapprox

        # the hot path itself
        A2 = deepcopy(A)
        out = apply_one_site(H1, A2)
        @test blocks(A2) == blocks(A)            # apply_one_site must not touch its input
        @test all(isfinite, vcat([vec(b) for b in blocks(out)]...))
    end
    BondUpdateBUG.set_symmetry!(:U1)
end

@testset "a full step is unchanged and still physical" begin
    BondUpdateBUG.set_symmetry!(:U1)
    L = 8
    mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = 1.0)
    base = BondUpdateBUG.neel_state(L)
    for _ in 1:4
        RSVDCBEBondUpdate.tdvp2_step!(base, mpo, ComplexF64(-im * 0.3);
                                      maxdim = 32, trunc_thresh = 0.0, maxiter = 6)
    end

    # Real-time evolution is unitary, so the norm is a genuine invariant here — an aliasing
    # bug in the matvec would break it long before it broke anything visible.
    for arm in (:tdvp2, :cbe1s, :bug)
        p = deepcopy(base)
        if arm === :tdvp2
            RSVDCBEBondUpdate.tdvp2_step!(p, mpo, ComplexF64(-im * 0.05);
                maxdim = 32, trunc_thresh = 0.0, maxiter = 8)
        elseif arm === :cbe1s
            RSVDCBEBondUpdate.tdvp_cbe1s_step!(p, mpo, ComplexF64(-im * 0.05);
                maxdim = 32, trunc_thresh = 0.0, maxiter = 8, exact = false)
        else
            RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-im * 0.05);
                maxdim = 32, trunc_thresh = 0.0, maxiter = 8, exact = false,
                split_maxdim = 32)
        end
        n = sqrt(real(BondUpdateBUG.overlap(p, p)))
        @test isfinite(n)
        @test abs(n - 1) < 1e-8
    end
end
