using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG

@testset "BondFrame" begin

    @testset "product state, bond 2" begin
        psi = domain_wall_state(6)
        canonical!(psi, 2)
        f = bond_frame(psi, 2)

        @testset "U0 is a left isometry" begin
            # NOT f.U0' * f.U0 -- `*` is greedy and closes the bond, giving a
            # rank-0 scalar (docs/telum_api_contract.md section 7a).
            g = left_gram(f.U0)
            @test isapprox(norm(g - 1), 0.0; atol = 1e-12)
        end

        @testset "V0 is a right isometry" begin
            g = right_gram(f.V0)
            @test isapprox(norm(g - 1), 0.0; atol = 1e-12)
        end

        @testset "U0*S0*V0 reconstructs the two-site block exactly" begin
            @test isapprox(norm(frame_theta(f) - two_site_block(psi, 2)), 0.0; atol = 1e-12)
        end

        @testset "old_rank matches the middle link dimension" begin
            # dim lives on the tensor's spaces, not on the TLIndex handle
            @test f.old_rank == leg_dim(psi[2], 3)
            @test f.old_rank == sum(d for (_, d) in f.link_mid_space)
        end

        @testset "leg handles name the right legs" begin
            @test f.link_l   == psi[2].inds[1]
            @test f.site_l   == psi[2].inds[2]
            @test f.link_mid == psi[2].inds[3]
            @test f.site_r   == psi[3].inds[2]
            @test f.link_r   == psi[3].inds[3]
        end
    end

    @testset "every interior bond of a genuinely entangled state" begin
        # A product state gives every bond rank 1 and a single charge sector,
        # which leaves all the multi-sector bookkeeping untested. Direct-summing
        # two product states of EQUAL total Sz along the link legs is the
        # standard MPS-sum construction and gives bonds that carry two distinct
        # charge sectors -- built with `oplus`, which is itself sector-aware.
        function mps_sum(a::SymMPS, b::SymMPS)
            L = length(a)
            ts = Any[]
            for i in 1:L
                dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
                push!(ts, to_concrete(oplus(a[i], b[i], dims)))
            end
            return SymMPS(ts, L)
        end

        psi = mps_sum(product_state([:up, :down, :up, :down]),
                      product_state([:down, :up, :up, :down]))
        canonical!(psi, 1)          # establish canonical form from scratch
        canonical!(psi, 4)

        @test any(bond_dims(psi) .> 1)
        @test maximum(length(psi[i].spaces[3]) for i in 1:3) > 1   # multi-sector bond

        for i in 1:3
            canonical!(psi, i)
            f = bond_frame(psi, i)
            @test isapprox(norm(left_gram(f.U0) - 1), 0.0; atol = 1e-12)
            @test isapprox(norm(right_gram(f.V0) - 1), 0.0; atol = 1e-12)
            @test isapprox(norm(frame_theta(f) - two_site_block(psi, i)), 0.0; atol = 1e-12)
            @test f.old_rank == leg_dim(psi[i], 3)
            @test isapprox(norm(f.S0), norm(psi); atol = 1e-12)
        end
    end

    @testset "every interior bond of a product state" begin
        psi = neel_state(6)
        for i in 1:5
            canonical!(psi, i)
            f = bond_frame(psi, i)
            @test isapprox(norm(left_gram(f.U0) - 1), 0.0; atol = 1e-12)
            @test isapprox(norm(right_gram(f.V0) - 1), 0.0; atol = 1e-12)
            @test isapprox(norm(frame_theta(f) - two_site_block(psi, i)), 0.0; atol = 1e-12)
            @test f.old_rank == leg_dim(psi[i], 3)
        end
    end

    @testset "the snapshot is norm-preserving" begin
        psi = domain_wall_state(6)
        canonical!(psi, 3)
        f = bond_frame(psi, 3)
        @test isapprox(norm(frame_theta(f)), norm(psi); atol = 1e-13)
        @test isapprox(norm(f.S0), norm(psi); atol = 1e-13)
    end

    @testset "bond_frame refuses a misplaced centre" begin
        psi = domain_wall_state(6)
        canonical!(psi, 4)
        @test_throws ArgumentError bond_frame(psi, 2)
        @test_throws BoundsError bond_frame(psi, 6)
    end
end
