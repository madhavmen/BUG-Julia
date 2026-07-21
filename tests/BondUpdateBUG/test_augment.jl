using Test, LinearAlgebra, Random, LurCGT, Telum
using BUGJulia.BondUpdateBUG

function mps_sum(a::SymMPS, b::SymMPS)
    L = length(a); ts = Any[]
    for i in 1:L
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(ts, to_concrete(oplus(a[i], b[i], dims)))
    end
    return SymMPS(ts, L)
end

newrng() = MersenneTwister(0x5EED)

@testset "augmented isometry" begin

    psi = domain_wall_state(6); canonical!(psi, 2)
    f = bond_frame(psi, 2)
    K1 = to_concrete(f.U0 * f.S0)

    @testset "U_aug is an isometry" begin
        # NOT U' * U -- greedy `*` closes the bond and returns a scalar.
        U, _ = augmented_left_isometry(f.U0, K1; rng = newrng())
        @test left_isometry_defect(U) < 1e-12
    end

    @testset "U_aug spans U0 (the old frame is retained exactly)" begin
        U, _ = augmented_left_isometry(f.U0, K1; rng = newrng())
        # resid = U0 - U (U' U0), contracting only the (link, site) legs
        resid = perp_component(U, f.U0)
        @test isapprox(norm(resid), 0.0; atol = 1e-12)
    end

    # THE REGRESSION GUARD. complete_column_basis padded every partially
    # populated sector to the full ambient dim(link)*dim(site); the range
    # basis must not.
    @testset "augmented rank is bounded by 2r + opened sectors, below ambient" begin
        U, _ = augmented_left_isometry(f.U0, K1; rng = newrng())
        ambient = leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
        opened = count(r -> r.missing, sector_report(f.U0, K1))
        @test leg_dim(U, 3) <= 2 * f.old_rank + opened
        @test leg_dim(U, 3) <= ambient
    end

    @testset "REGRESSION: rank stays strictly below ambient where there is room" begin
        # At bond 2 of a 6-site product state the ambient space is only
        # 1*2 = 2, so opening the single empty sector necessarily reaches it --
        # the strict bound is vacuous there. The guard against
        # complete_column_basis needs a bond whose ambient space is genuinely
        # larger than 2r + opened.
        ent = mps_sum(product_state([:up, :down, :up, :down, :up, :down]),
                      product_state([:down, :up, :up, :down, :down, :up]))
        canonical!(ent, 1)
        strict = 0
        for i in 2:5
            canonical!(ent, i)
            g = bond_frame(ent, i)
            Ki = to_concrete(g.U0 * g.S0)
            U, _ = augmented_left_isometry(g.U0, Ki; rng = newrng())
            amb = leg_dim(g.U0, 1) * leg_dim(g.U0, 2)
            @test leg_dim(U, 3) <= amb
            leg_dim(U, 3) < amb && (strict += 1)
        end
        @test strict > 0        # at least one bond is genuinely below ambient
    end

    @testset "the rank bound holds at every bond of an entangled state" begin
        ent = mps_sum(product_state([:up, :down, :up, :down, :up, :down]),
                      product_state([:down, :up, :up, :down, :down, :up]))
        canonical!(ent, 1)
        for i in 1:5
            canonical!(ent, i)
            g = bond_frame(ent, i)
            Ki = to_concrete(g.U0 * g.S0)
            U, nnew = augmented_left_isometry(g.U0, Ki; rng = newrng())
            opened = count(r -> r.missing, sector_report(g.U0, Ki))
            @test leg_dim(U, 3) <= 2 * g.old_rank + opened
            @test leg_dim(U, 3) <= leg_dim(g.U0, 1) * leg_dim(g.U0, 2)
            @test left_isometry_defect(U) < 1e-12
            @test isapprox(norm(perp_component(U, g.U0)), 0.0; atol = 1e-12)
            @test nnew == leg_dim(U, 3) - g.old_rank
        end
    end

    @testset "a reachable-but-empty sector is opened by the fill" begin
        missed = filter(r -> r.missing, sector_report(f.U0, K1))
        @test !isempty(missed)                      # the fixture must exercise this
        U, _ = augmented_left_isometry(f.U0, K1; rng = newrng())
        for r in missed
            @test any(s -> first(s) == r.charge, U.spaces[3])
        end
    end

    @testset "missing_fill=0 leaves the empty sectors closed" begin
        U0f, _ = augmented_left_isometry(f.U0, K1; missing_fill = 0, rng = newrng())
        U1f, _ = augmented_left_isometry(f.U0, K1; missing_fill = 1, rng = newrng())
        @test leg_dim(U0f, 3) < leg_dim(U1f, 3)
    end

    @testset "the fill opens ONE direction per empty sector, not the whole sector" begin
        # This is the distinction from complete_column_basis: it padded a
        # sector to its full local dimension.
        missed = filter(r -> r.missing, sector_report(f.U0, K1))
        U, _ = augmented_left_isometry(f.U0, K1; missing_fill = 1, rng = newrng())
        for r in missed
            d = 0
            for (s, dd) in U.spaces[3]
                s == r.charge && (d = dd)
            end
            @test d == min(1, r.reachable_dim)
        end
    end

    @testset "the isometry defect actually detects a non-isometry" begin
        # Negative control: without this, a defect measure that always returns
        # ~0 would pass every isometry test in the suite.
        U, _ = augmented_left_isometry(f.U0, K1; rng = newrng())
        @test left_isometry_defect(U) < 1e-12
        @test left_isometry_defect(to_concrete(2.0 * U)) > 1e-3
        @test right_isometry_defect(f.V0) < 1e-12
        @test right_isometry_defect(to_concrete(3.0 * f.V0)) > 1e-3
    end

    @testset "augment=false returns U0 untouched" begin
        U, nnew = augmented_left_isometry(f.U0, K1; augment = false, rng = newrng())
        @test nnew == 0
        @test leg_dim(U, 3) == f.old_rank
    end

    @testset "max_rank caps the new directions" begin
        U, _ = augmented_left_isometry(f.U0, K1; max_rank = 1, rng = newrng())
        @test leg_dim(U, 3) <= leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
        @test left_isometry_defect(U) < 1e-12
    end

    @testset "the seed is reproducible for a fixed rng" begin
        Ua, _ = augmented_left_isometry(f.U0, K1; rng = newrng())
        Ub, _ = augmented_left_isometry(f.U0, K1; rng = newrng())
        @test isapprox(norm(Ua - Ub), 0.0; atol = 1e-14)
    end

    # ---- right frame ----

    @testset "V_aug is a right isometry spanning V0" begin
        L1 = to_concrete(f.S0 * f.V0)
        V, nnew = augmented_right_isometry(f.V0, L1; rng = newrng())
        @test right_isometry_defect(V) < 1e-12
        @test isapprox(norm(perp_component_right(V, f.V0)), 0.0; atol = 1e-12)
        @test nnew == leg_dim(V, 1) - leg_dim(f.V0, 1)
    end

    @testset "right rank bound, below ambient" begin
        L1 = to_concrete(f.S0 * f.V0)
        V, _ = augmented_right_isometry(f.V0, L1; rng = newrng())
        r0 = leg_dim(f.V0, 1)
        opened = count(r -> r.missing, sector_report_right(f.V0, L1))
        @test leg_dim(V, 1) <= 2 * r0 + opened
        @test leg_dim(V, 1) <= leg_dim(f.V0, 2) * leg_dim(f.V0, 3)
    end
end
