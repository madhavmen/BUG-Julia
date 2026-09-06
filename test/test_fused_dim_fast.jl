# THE FAST `fused_dim` MUST AGREE WITH `reachable_sectors` WHEREVER IT IS USED.
#
# `cbe_expand` needs the dimension of the fused `leg_a (x) leg_b` space to decide how much room a
# frame has left. The exact answer is `sum(d for (q,d) in reachable_sectors(t,a,b))`, which builds
# a full `fusion_basis` (`getIdentity` + a complete `svd`) and then throws the basis away. For
# ABELIAN charges that sum is identically `leg_dim(t,a) * leg_dim(t,b)`, so the SVD is pure waste
# -- 116 of them per step at L=30.
#
# This file is the thing that licenses the substitution. It does NOT check "the numbers look
# close"; it checks the two expressions are EQUAL, on states whose bonds actually carry several
# charge sectors, since a vacuum-only bond would agree by accident.
#
# ⛔ AND IT PINS THE NON-ABELIAN SIDE, which is the half that would fail silently. Under SU(2)
# `leg_dim` counts MULTIPLETS, so the product UNDERCOUNTS the fused space badly (a spin-1/2 site
# is one multiplet, so `chi * 1 = chi` where the truth is ~`2*chi`). If the fast path ever leaked
# into SU(2), `room = fused_dim - r` would collapse to ~0, CBE would silently decline to expand,
# and the state would freeze at low rank -- which looks like a physics result, not a bug. So the
# SU(2) assertion here is deliberately the INEQUALITY: the product must be STRICTLY smaller, which
# is what makes "we took the slow path" observable rather than assumed.

using Test
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.BondUpdateBUG: reachable_sectors, leg_dim, symmetry_mode, set_symmetry!
using BUGJulia.RSVDCBEBondUpdate

exact_fused(t, a, b) = sum(d for (_, d) in reachable_sectors(t, a, b); init = 0)
fast_fused(t, a, b)  = leg_dim(t, a) * leg_dim(t, b)

"A state warmed far enough that its bonds carry several charge sectors."
function warm_state(L; sym, steps = 6, maxdim = 32)
    set_symmetry!(sym)
    mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = 1.0)
    psi = BondUpdateBUG.neel_state(L)
    for _ in 1:steps
        RSVDCBEBondUpdate.tdvp2_step!(psi, mpo, ComplexF64(-im * 0.3);
                                      maxdim = maxdim, trunc_thresh = 0.0, maxiter = 6)
    end
    return psi
end

@testset "abelian: the product IS the fused dimension" begin
    for sym in (:U1, :none)
        psi = warm_state(10; sym = sym)
        canonical!(psi, 5)

        # A vacuum-only bond agrees by accident, so assert the test is not vacuous first.
        nsec = maximum(length(psi[i].spaces[3]) for i in 1:(length(psi) - 1))
        if sym === :U1
            @test nsec >= 3        # several charge sectors, so fusion is doing real work
        end

        for i in 1:(length(psi) - 1)
            A = psi[i]             # (link_l, site, link_r)
            B = psi[i + 1]
            # exactly the two calls `cbe_expand` makes
            @test fast_fused(A, 1, 2) == exact_fused(A, 1, 2)
            @test fast_fused(B, 2, 3) == exact_fused(B, 2, 3)
        end
    end
    set_symmetry!(:U1)
end

@testset "SU(2): the product UNDERCOUNTS, so the slow path is mandatory" begin
    # ⛔ NOT `warm_state` HERE. `xxz_mpo` cannot be built under SU(2) at all -- the SU(2) local
    # space exposes only `I` and `S`, while `xxz_chain` (henv.jl:97) reaches for `.Sp` and so
    # raises `FieldError: type NamedTuple has no field Sp`. That is a property of the model
    # builder, not of this fix, and it is why this half uses a state needing no Hamiltonian.
    #
    # `dimer_state` exists under SU(2) and is the SHARPEST case rather than a weaker one: every
    # link carries exactly ONE multiplet, so `leg_dim` reads 1 on both legs and the product says
    # 1 -- while spin-1/2 (x) spin-1/2 fuses to 0 (+) 1, i.e. TWO multiplets. The undercount that
    # would silently freeze CBE is therefore visible with no time evolution at all.
    set_symmetry!(:SU2)
    psi = BondUpdateBUG.dimer_state(10)
    canonical!(psi, 5)

    # At least one bond where the multiplet product is strictly smaller than the true fused
    # dimension. If this ever stops holding, either SU(2) stopped being non-abelian or the
    # branch stopped mattering -- both must fail loudly here rather than as a frozen rank.
    strictly_smaller = 0
    for i in 1:(length(psi) - 1)
        A = psi[i]
        ex, fa = exact_fused(A, 1, 2), fast_fused(A, 1, 2)
        @test fa <= ex                 # the product can never OVERcount
        fa < ex && (strictly_smaller += 1)
    end
    @test strictly_smaller > 0

    set_symmetry!(:U1)
end

# END TO END: the expansion itself must be unchanged. `fused_dim` feeds `room_l`/`room_r`, which
# cap the per-side budget, so an error here shows up as a DIFFERENT RANK rather than an exception.
@testset "cbe1s and bug reach the same rank as before" begin
    set_symmetry!(:U1)
    L = 10
    mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = 1.0)
    for arm in (:cbe1s, :bug)
        psi = BondUpdateBUG.neel_state(L)
        for _ in 1:8
            if arm === :cbe1s
                RSVDCBEBondUpdate.tdvp_cbe1s_step!(psi, mpo, ComplexF64(-im * 0.1);
                    maxdim = 32, trunc_thresh = 1e-12, maxiter = 8, exact = false)
            else
                RSVDCBEBondUpdate.cbe_bug_step!(psi, mpo, ComplexF64(-im * 0.1);
                    maxdim = 32, trunc_thresh = 1e-12, maxiter = 8, exact = false,
                    split_maxdim = 32)
            end
        end
        # The rank has to have GROWN off the product state -- that is the whole thing `room`
        # gates, and a fast path that returned 0 would leave chi = 1 with no error anywhere.
        @test maximum(bond_dims(psi)) > 4
        n = sqrt(real(BondUpdateBUG.overlap(psi, psi)))
        @test abs(n - 1) < 1e-8
    end
end
