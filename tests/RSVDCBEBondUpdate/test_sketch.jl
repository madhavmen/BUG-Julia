using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# Stage 2a: the sketched H-action.
#
# THE REFERENCE IS THE STAGE-1 CODE, and that is the point: `apply_h_two_site` is already
# pinned off-diagonally to the dense matrix element, so requiring
#
#     sketch_h_left(f, …, Om)  ==  contract(apply_h_two_site(Theta, …), Om')
#
# for ARBITRARY `Om` pins the sketch to the same operator without re-deriving anything.
# `Om` is arbitrary rather than Gaussian here on purpose: this test is about the
# contraction order (which never builds the rank-4 block), not about the sampling, which
# is stage 2b's problem.
#
# Two `Om` families are used. `Om = V0` is the degenerate case that must reproduce the
# exact left factor of `H Theta` on the current frame -- it would still pass with the
# sketch dimension confused for the bond dimension, so the real test is a RANDOM `Om`
# with a column count that matches neither `r` nor `d·χ`.

"A random sketch block in `V0`'s layout `(g, site_r, link_r)`, `ncol` columns per sector.

Built from the fusion basis of `(site_r ⊗ link_r)` so the block lives in the sectors the
symmetry allows -- randomising a dense buffer would inject forbidden amplitudes, which is
the failure `random_sector_seed` documents."
function random_right_sketch(f::BondFrame, rng; ncol::Int = 2)
    F = fusion_basis(f.V0, 2, 3)                       # (site_r, link_r, fused)
    R = TLArray(symm(F), copy(F.qlabels), copy(F.wmatdata), copy(F.wmatinfo),
                [randn(rng, ComplexF64, size(r)) for r in F.RMTs], F.inds, F.spaces)
    Q = to_concrete(svd(to_concrete(R), (1, 2); cutoff = 0.0).U)
    Q = _trim_cols(Q, 3, ncol)
    return to_concrete(permutedims(Q, (3, 1, 2)))      # (g, site_r, link_r)
end

"Mirror: a random sketch block in `U0`'s layout `(link_l, site_l, g)`."
function random_left_sketch(f::BondFrame, rng; ncol::Int = 2)
    F = fusion_basis(f.U0, 1, 2)                       # (link_l, site_l, fused)
    R = TLArray(symm(F), copy(F.qlabels), copy(F.wmatdata), copy(F.wmatinfo),
                [randn(rng, ComplexF64, size(r)) for r in F.RMTs], F.inds, F.spaces)
    Q = to_concrete(svd(to_concrete(R), (1, 2); cutoff = 0.0).U)
    return _trim_cols(Q, 3, ncol)
end

"Keep at most `n` columns per charge sector of `leg`."
_trim_cols(Q, leg::Int, n::Int) =
    to_concrete(getsub(Q, leg, s -> (d = sector_dim(Q, leg, s); 1:min(d, n))))

"A state with real entanglement: a few BUG steps off a product state."
function evolved(L::Int; n_steps = 3, dt = 0.05)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = 16))
    return psi
end

@testset "sketched H-action" begin

    @testset "left sketch matches the formed two-site action" begin
        set_symmetry!(:U1)
        rng = Random.MersenneTwister(0x5EED)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                lenv = left_env_stack(psi, h; upto = i - 1)
                renv = right_env_stack(psi, h; downto = i + 2)
                hth = to_concrete(perp_component(
                    f.U0, apply_h_two_site(frame_theta(f), h, i, lenv, renv)))

                for Om in (f.V0, random_right_sketch(f, rng; ncol = 2),
                           random_right_sketch(f, rng; ncol = 1))
                    got = sketch_h_left(f, h, i, lenv, renv, Om)
                    want = to_concrete(contract(hth, (3, 4), Om', (2, 3)))
                    @test got.inds == want.inds
                    @test norm(to_concrete(got - want)) < 1e-10 * max(1.0, norm(want))
                end
            end
        end
    end

    @testset "right sketch matches the formed two-site action" begin
        set_symmetry!(:U1)
        rng = Random.MersenneTwister(0xC0FFEE)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                lenv = left_env_stack(psi, h; upto = i - 1)
                renv = right_env_stack(psi, h; downto = i + 2)
                hth = to_concrete(perp_component_right(
                    f.V0, apply_h_two_site(frame_theta(f), h, i, lenv, renv)))

                for Om in (f.U0, random_left_sketch(f, rng; ncol = 2),
                           random_left_sketch(f, rng; ncol = 1))
                    got = sketch_h_right(f, h, i, lenv, renv, Om)
                    want = to_concrete(contract(Om', (1, 2), hth, (1, 2)))
                    @test got.inds == want.inds
                    @test norm(to_concrete(got - want)) < 1e-10 * max(1.0, norm(want))
                end
            end
        end
    end

    @testset "the sketch is orthogonal to the frame, and the energy still checks out" begin
        # THE PROJECT-FIRST SIGNATURE. `K0 = U0 S0` lies entirely in the KEPT space, and the
        # sketch returns `P_perp (H Theta) Om'`, so their overlap must vanish identically --
        # for ANY probe. That is the property project-first buys by construction, and it is
        # the sharpest single check that the projection is really happening first.
        #
        # The energy handle that used to live here has moved onto `apply_h_two_site`, which
        # still forms the unprojected action: <K0| H Theta> is the Galerkin matrix element
        # and must equal what the environments report. Splitting the two keeps an
        # independent scalar check on the operator while testing the projector separately.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        want = env_energy(psi, h)
        for i in 1:(L - 1)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            lenv = left_env_stack(psi, h; upto = i - 1)
            renv = right_env_stack(psi, h; downto = i + 2)
            K0 = to_concrete(f.U0 * f.S0)
            L0 = to_concrete(f.S0 * f.V0)

            YL = sketch_h_left(f, h, i, lenv, renv, f.V0)      # (link_l, site_l, bond)
            @test abs(tensor_inner(K0, YL)) < 1e-10
            YR = sketch_h_right(f, h, i, lenv, renv, f.U0)     # (bond, site_r, link_r)
            @test abs(tensor_inner(L0, YR)) < 1e-10

            hth = apply_h_two_site(frame_theta(f), h, i, lenv, renv)
            @test abs(tensor_inner(frame_theta(f), hth) - want) < 1e-10
        end
    end

    @testset "no symmetry" begin
        # Without op-legs every term takes the rank-2 branch of contribution (e), which is
        # a different code path from the U(1) XY hop.
        set_symmetry!(:none)
        try
            rng = Random.MersenneTwister(0xBEEF)
            L = 6
            h = xxz_chain(L)
            psi = evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                lenv = left_env_stack(psi, h; upto = i - 1)
                renv = right_env_stack(psi, h; downto = i + 2)
                hth = apply_h_two_site(frame_theta(f), h, i, lenv, renv)
                hthL = to_concrete(perp_component(f.U0, hth))
                hthR = to_concrete(perp_component_right(f.V0, hth))

                Om = random_right_sketch(f, rng; ncol = 2)
                got = sketch_h_left(f, h, i, lenv, renv, Om)
                want = to_concrete(contract(hthL, (3, 4), Om', (2, 3)))
                @test norm(to_concrete(got - want)) < 1e-10 * max(1.0, norm(want))

                OmL = random_left_sketch(f, rng; ncol = 2)
                got2 = sketch_h_right(f, h, i, lenv, renv, OmL)
                want2 = to_concrete(contract(OmL', (1, 2), hthR, (1, 2)))
                @test norm(to_concrete(got2 - want2)) < 1e-10 * max(1.0, norm(want2))
            end
        finally
            set_symmetry!(:U1)
        end
    end
end
