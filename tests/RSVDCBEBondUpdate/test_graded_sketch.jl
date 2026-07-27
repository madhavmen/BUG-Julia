using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

# Stage 2b: the sector-graded Gaussian sketch.
#
# THE HEADLINE CLAIM LIVES HERE. `augment.jl` needs `random_sector_seed` because under
# U(1), with the opposite frame frozen, the K/L complement can never OPEN a charge sector
# the frame does not already touch. The CBE sketch is supposed to make that fill
# unnecessary, and the reason is arithmetic rather than luck: complement columns are
# allocated per sector in proportion to `DC(q) = Dfull(q) - DT(q)`, so a sector the frame
# misses entirely has `DT(q) = 0` and therefore the LARGEST share of any sector.
#
# So the tests below are, in order of what they are worth:
#
#   1. the sketch's sectors include every reachable-but-missing charge   <- the claim
#   2. `H` acting through that sketch puts real weight in such a sector <- the claim is useful
#   3. both halves of the CompRatio split are actually present, and shift with the ratio
#   4. the mechanical properties (layout, caps, determinism, saturated frame)
#
# Test 1 without test 2 would be hollow: sampling a sector proves nothing if the dynamics
# has nothing to put there.

"Charges reachable on the frame's local space but absent from the frame itself."
function missing_charges_of(frame, side::Symbol)
    a, b = side === :left ? (1, 2) : (2, 3)
    fl   = side === :left ? 3 : 1
    reach = reachable_sectors(frame, a, b)
    have = Set(q for (q, _) in frame.spaces[fl])
    return [q for (q, d) in reach if d > 0 && !(q in have)]
end

"Sectors present on a sketch's own sketch leg."
sketch_charges(Om, side::Symbol) =
    Set(q for (q, _) in Om.spaces[side === :left ? 3 : 1])

@testset "sector-graded sketch" begin

    @testset "samples every reachable-but-missing charge sector" begin
        set_symmetry!(:U1)
        L = 6
        psi = domain_wall_state(L)                # product state: rank 1 everywhere
        for i in 1:(L - 1)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            for (side, frame) in ((:left, f.U0), (:right, f.V0))
                miss = missing_charges_of(frame, side)
                isempty(miss) && continue          # nothing to prove at this bond/side
                Om = sector_graded_sketch(frame, side, 4;
                                          rng = Random.MersenneTwister(0x5EED))
                @test Om !== nothing
                got = sketch_charges(Om, side)
                # This is the claim: no fill, no seeding -- the sketch covers them.
                @test all(q in got for q in miss)
            end
        end
    end

    @testset "H through the sketch populates a missing sector" begin
        # At the domain wall the XY term genuinely rotates weight into the sector the
        # frame is missing, so a nonzero projection there is a statement about the
        # dynamics, not about the sampling. (Away from the wall, e.g. bond 1 of
        # up-up-up-down-down-down, both sites are up and no XY term can act -- the state
        # cannot grow there at all, which is physics, not a failure.)
        set_symmetry!(:U1)
        L = 6
        wall = L ÷ 2                               # bond (wall, wall+1) straddles it
        h = xxz_chain(L)
        psi = domain_wall_state(L)
        canonical!(psi, wall)
        f = bond_frame(psi, wall)
        lenv = left_env_stack(psi, h; upto = wall - 1)
        renv = right_env_stack(psi, h; downto = wall + 2)

        miss = missing_charges_of(f.U0, :left)
        @test !isempty(miss)

        Om = sector_graded_sketch(f.V0, :right, 4; rng = Random.MersenneTwister(0xC0FFEE))
        Y = sketch_h_left(f, h, wall, lenv, renv, Om)
        resid = to_concrete(perp_component(f.U0, Y))     # outside the current frame
        @test norm(resid) > 1e-8

        # ... and the weight is in a sector the frame did not have.
        Q = to_concrete(svd(resid, (1, 2); cutoff = 1e-10).U)
        newq = Set(q for (q, _) in Q.spaces[3])
        @test any(q in newq for q in miss)
    end

    @testset "both halves of the CompRatio split are present" begin
        set_symmetry!(:U1)
        L = 6
        psi = domain_wall_state(L)
        bond_update_bug!(psi, bond_gates(psi);
                         opts = BondUpdateOptions(dt = 0.05, n_steps = 3, maxdim = 16))
        canonical!(psi, 3)
        f = bond_frame(psi, 3)

        function halves(ratio)
            Om = sector_graded_sketch(f.V0, :right, 8; comp_ratio = ratio,
                                      rng = Random.MersenneTwister(0xBEEF))
            Om === nothing && return (0.0, 0.0)
            perp = norm(to_concrete(perp_component_right(f.V0, Om)))
            inside = norm(to_concrete(Om - to_concrete(perp_component_right(f.V0, Om))))
            return (perp, inside)
        end

        p_lo, i_lo = halves(0.25)
        p_hi, i_hi = halves(0.75)
        # Neither half may be missing: a pure-complement sketch is exactly what the
        # reference warns against ("1-site components are also important").
        @test p_lo > 1e-10 && i_lo > 1e-10
        @test p_hi > 1e-10 && i_hi > 1e-10
        # And the ratio has to actually do something.
        @test p_hi / i_hi > p_lo / i_lo
    end

    @testset "layout, caps and determinism" begin
        set_symmetry!(:U1)
        L = 6
        psi = domain_wall_state(L)
        bond_update_bug!(psi, bond_gates(psi);
                         opts = BondUpdateOptions(dt = 0.05, n_steps = 3, maxdim = 16))
        canonical!(psi, 3)
        f = bond_frame(psi, 3)

        OmL = sector_graded_sketch(f.U0, :left, 6; rng = Random.MersenneTwister(1))
        @test OmL !== nothing
        @test length(OmL.inds) == 3
        @test OmL.inds[1] == f.U0.inds[1] && OmL.inds[2] == f.U0.inds[2]

        OmR = sector_graded_sketch(f.V0, :right, 6; rng = Random.MersenneTwister(1))
        @test OmR !== nothing
        @test OmR.inds[2] == f.V0.inds[2] && OmR.inds[3] == f.V0.inds[3]

        # No sector may be sampled beyond its own dimension in the local space.
        full = Dict(reachable_sectors(f.U0, 1, 2))
        for (q, d) in OmL.spaces[3]
            @test d <= get(full, q, 0)
        end

        # Same seed, same sketch.
        again = sector_graded_sketch(f.U0, :left, 6; rng = Random.MersenneTwister(1))
        @test norm(to_concrete(OmL - again)) == 0.0

        # The ENDPOINTS ARE VALID and are the A/B worth running: 1.0 draws only the
        # complement (discarded) space, 0.0 only the isometry (kept) space. An earlier
        # version rejected both AND floored each half at one column, so "complement only"
        # silently still drew a kept-space column and could not be measured at all.
        for cr in (0.0, 1.0)
            Om = sector_graded_sketch(f.U0, :left, 6; comp_ratio = cr,
                                      rng = Random.MersenneTwister(2))
            @test Om !== nothing
            @test leg_dim(Om, 3) >= 1
        end
        # A pure-complement sketch must be orthogonal to the frame; a pure-isometry one
        # must lie inside it. This is what distinguishes the two halves.
        Omc = sector_graded_sketch(f.U0, :left, 6; comp_ratio = 1.0,
                                   rng = Random.MersenneTwister(3))
        @test norm(to_concrete(Omc - to_concrete(perp_component(f.U0, Omc)))) < 1e-10
        Omt = sector_graded_sketch(f.U0, :left, 6; comp_ratio = 0.0,
                                   rng = Random.MersenneTwister(3))
        @test norm(to_concrete(perp_component(f.U0, Omt))) < 1e-10

        @test_throws ArgumentError sector_graded_sketch(f.U0, :left, 6; comp_ratio = -0.1)
        @test_throws ArgumentError sector_graded_sketch(f.U0, :left, 6; comp_ratio = 1.5)
        @test_throws ArgumentError sector_graded_sketch(f.U0, :left, 0)
        @test_throws ArgumentError sector_graded_sketch(f.U0, :sideways, 6)
    end

    @testset "a saturated frame still makes a valid probe" begin
        # At bond 1 the left link is the dim-1 vacuum, so the local space is only
        # 2-dimensional and the frame saturates it after a couple of steps. The sketch must
        # NOT decline there: it is a probe applied to the frame on the OPPOSITE side of the
        # bond, so a full frame is still a usable probe. It returns its isometry half, with
        # no complement columns -- declining instead would refuse to expand one side of a
        # bond because the other was full.
        set_symmetry!(:U1)
        L = 6
        psi = domain_wall_state(L)
        bond_update_bug!(psi, bond_gates(psi);
                         opts = BondUpdateOptions(dt = 0.05, n_steps = 4, maxdim = 16))
        canonical!(psi, 1)
        f = bond_frame(psi, 1)
        saturated = leg_dim(f.U0, 3) == leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
        Om = sector_graded_sketch(f.U0, :left, 4; rng = Random.MersenneTwister(3))
        @test Om !== nothing
        if saturated
            # No complement exists, so everything the sketch returns lies in the frame.
            @test norm(to_concrete(perp_component(f.U0, Om))) < 1e-10
        end
    end

    @testset "no symmetry" begin
        # One dense sector, so there is never a missing sector to open -- the sketch still
        # has to produce a valid complement block, which is what makes rank growth possible
        # without `pad`.
        #
        # The bond is CHOSEN by how much room it has rather than fixed: without symmetry a
        # middle bond saturates its local space quickly (bond dims run to 2*chi), and a
        # saturated frame has no complement to find. Asserting one at a fixed bond tests
        # the state's rank profile, not the sketch.
        set_symmetry!(:none)
        try
            L = 6
            psi = domain_wall_state(L)
            bond_update_bug!(psi, bond_gates(psi);
                             opts = BondUpdateOptions(dt = 0.05, n_steps = 2, maxdim = 16))
            best_i, best_room = 0, 0
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                room = leg_dim(f.V0, 2) * leg_dim(f.V0, 3) - leg_dim(f.V0, 1)
                room > best_room && ((best_i, best_room) = (i, room))
            end
            @test best_room > 0                    # otherwise there is nothing to test
            canonical!(psi, best_i)
            f = bond_frame(psi, best_i)
            Om = sector_graded_sketch(f.V0, :right, 4; rng = Random.MersenneTwister(7))
            @test Om !== nothing
            @test norm(to_concrete(perp_component_right(f.V0, Om))) > 1e-10
        finally
            set_symmetry!(:U1)
        end
    end
end
