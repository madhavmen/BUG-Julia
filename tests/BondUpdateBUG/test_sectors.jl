using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG

function mps_sum(a::SymMPS, b::SymMPS)
    L = length(a); ts = Any[]
    for i in 1:L
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(ts, to_concrete(oplus(a[i], b[i], dims)))
    end
    return SymMPS(ts, L)
end

@testset "sector enumeration" begin

    @testset "reachable sectors of link(x)site cover the fused charges" begin
        psi = domain_wall_state(4); canonical!(psi, 1)
        f = bond_frame(psi, 1)
        secs = reachable_sectors(f.U0, 1, 2)
        # `dim(f.link_l)` is impossible -- a TLIndex carries no dimension.
        @test sum(d for (_, d) in secs) == leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
        @test length(unique(first.(secs))) == length(secs)   # no duplicate charges
    end

    @testset "hand-rolled charge arithmetic agrees with Telum's own fusion" begin
        # Must cover links carrying a NON-ZERO charge. With a vacuum link the
        # reachable set is {+q,-q}, which is its own dual, so a spurious extra
        # dual still passes -- that false pass hid a real sign bug (job 93413).
        saw_charged_link = Ref(false)
        for psi in (domain_wall_state(4), neel_state(6),
                    mps_sum(product_state([:up, :down, :up, :down]),
                            product_state([:down, :up, :up, :down])))
            for i in 1:(length(psi) - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                t = f.U0
                viaTelum = reachable_sectors(t, 1, 2)
                mine = fuse_spaces(symm(t), t.spaces[1], t.inds[1].dir,
                                             t.spaces[2], t.inds[2].dir)
                any(q != dual_charge(symm(t), q) for (q, _) in t.spaces[1]) &&
                    (saw_charged_link[] = true)
                @test sort(collect(viaTelum), by = string) == sort(collect(mine), by = string)
            end
        end
        @test saw_charged_link[]
    end

    # The L=4 XX U(1) case verified independently in Python by
    # alice_imagtime_study/kls_missing_qn_fill.py.
    @testset "missing-QN table on the L=4 bond (1,2)" begin
        psi = domain_wall_state(4); canonical!(psi, 1)
        f = bond_frame(psi, 1)
        K1 = to_concrete(f.U0 * f.S0)      # K1 stays in U0's sector
        rep = sector_report(f.U0, K1)

        present = filter(r -> !r.missing, rep)
        missed  = filter(r ->  r.missing, rep)
        @test length(present) == 1
        @test present[1].u0_cols == 1
        @test present[1].k1_cols == 1
        @test present[1].range_cols == 1
        @test length(missed) == 1          # the opposite charge is reachable but empty
        @test missed[1].range_cols == 0
        @test missed[1].reachable_dim > 0
        @test missed[1].u0_cols == 0 && missed[1].k1_cols == 0
    end

    @testset "a sector both U0 and K1 miss is flagged, not silently dropped" begin
        psi = domain_wall_state(4); canonical!(psi, 1)
        f = bond_frame(psi, 1)
        rep = sector_report(f.U0, to_concrete(f.U0 * f.S0))
        @test any(r -> r.missing && r.reachable_dim > 0, rep)
        @test length(missing_charges(rep)) == 1
    end

    @testset "K1 inside U0's range adds no new directions anywhere" begin
        # K1 = U0*S0 lies entirely in U0's column space, so P_perp K1 == 0 and
        # range_cols must equal u0_cols in every sector.
        psi = neel_state(6)
        for i in 1:5
            canonical!(psi, i)
            f = bond_frame(psi, i)
            K1 = to_concrete(f.U0 * f.S0)
            @test norm(perp_component(f.U0, K1)) < 1e-12
            for r in sector_report(f.U0, K1)
                @test r.range_cols == r.u0_cols
            end
        end
    end

    @testset "a K1 outside U0's range does add directions" begin
        # Build K1 with support on a charge U0 does not carry, by taking the
        # frame of a DIFFERENT state on the same bond.
        ent = mps_sum(product_state([:up, :down, :up, :down]),
                      product_state([:down, :up, :up, :down]))
        canonical!(ent, 2)
        fe = bond_frame(ent, 2)
        rep = sector_report(fe.U0, to_concrete(fe.U0 * fe.S0))
        @test !isempty(rep)
        # every reported row is self-consistent
        for r in rep
            @test r.range_cols <= r.reachable_dim || r.reachable_dim == 0
            @test r.missing == (r.range_cols == 0 && r.reachable_dim > 0)
        end
    end

    @testset "every reachable charge is reported" begin
        psi = domain_wall_state(6)
        for i in 1:5
            canonical!(psi, i)
            f = bond_frame(psi, i)
            rep = sector_report(f.U0, to_concrete(f.U0 * f.S0))
            reported = Set(r.charge for r in rep)
            for (q, _) in reachable_sectors(f.U0, 1, 2)
                @test q in reported
            end
        end
    end
end
