using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

# THE COMMON INITIAL STATE, AND THE OBSERVABLE THAT SURVIVES SU(2).
#
# An SU(2)-vs-no-symmetry comparison is only meaningful if both sides evolve THE SAME STATE:
# the claim under test is that symmetry changes the cost, not the physics. Before this existed
# there was no such state -- `product_state`/`neel_state` are guarded off under `:SU2` (a
# definite-Sz product state spans many total-spin sectors) and `dimer_state` was `:SU2`-only.
#
# REGRESSION FOR A SILENT WRONG ANSWER. The first `:none` implementation built the dimer by
# evolving a Néel state on bonds 1,3,5,… with `parity_sweep!`, and returned the UNCHANGED NÉEL
# STATE with nothing raised: `bond_dims = [1,1,1,1,1,1,1]`, every bond energy `-0.25`.
#
# TWO DISTINCT CAUSES produce that identical output, and conflating them cost real time here:
#
#   (a) the parity NAMES are 0-based -- `parity_bonds(L, :even)` is `1,3,5,…` and `:odd` is
#       `2,4,6,…`. Asking for `:odd` while placing gates on `1,3,5,…` leaves every gate in the
#       requested group `nothing`, so the sweep skips every bond and does NOTHING;
#   (b) `kls_bond_update` genuinely cannot grow rank from a product state -- chi=1 in, chi=1 out,
#       and a singlet needs chi=2.
#
# The original failure was (a). It was then misdiagnosed as (b) -- plausibly, since (b) is real
# and is a known property of the standard BUG. (b) is now MEASURED properly, on the correct
# parity group, and is the control in the CBE testset below. The lesson is that "the state did
# not change" does not identify WHY, so a control has to be shown to do something before its
# passing means anything.
#
# The assertions below are therefore on the PHYSICS (bond profile, energy, chi) rather than on
# "it ran": every broken version above ran fine and produced a normalised, plausible state.

@testset "dimer_state and bond_energies" begin

    "Analytic profile: -3/4 on a dimerised bond, 0 between dimers."
    analytic(L) = [isodd(b) ? -0.75 : 0.0 for b in 1:(L - 1)]

    @testset "the dimer is right in :none and :U1, and is not the Neel state" begin
        # Both abelian modes go through `_dimer_state_projected`. They are listed separately
        # rather than assumed equivalent: `:U1` stores the two singlet components in DIFFERENT
        # charge sectors (Sz = +1/2 and -1/2 on the bond) while `:none` stores them in one dense
        # sector, so "chi = 2" is reached by a different route in each.
        for sym in (:none, :U1)
            set_symmetry!(sym)
            for L in (4, 6, 8)
                psi = dimer_state(L)
                g = bond_gates(psi; delta = 1.0)
                @test norm(psi) ≈ 1.0 atol = 1e-12
                # chi = 2 on the dimerised bonds. THE regression assertion: the broken version
                # gave all ones, from a sweep that never touched the state.
                @test bond_dims(psi) == [isodd(b) ? 2 : 1 for b in 1:(L - 1)]
                @test bond_energies(psi, g) ≈ analytic(L) atol = 1e-12
                @test energy(psi, g) ≈ -0.75 * (L ÷ 2) atol = 1e-12
            end
        end
    end

    @testset "the dimer is right in :SU2" begin
        set_symmetry!(:SU2)
        for L in (4, 6, 8)
            psi = dimer_state(L)
            g = bond_gates(psi; delta = 1.0)
            @test norm(psi) ≈ 1.0 atol = 1e-12
            # Every link carries ONE multiplet, so this reads [1,1,...] while representing the
            # same chi=2 state `:none` stores explicitly. Multiplets are not states -- which is
            # exactly why bond dimension may not be compared across symmetries.
            @test bond_dims(psi) == ones(Int, L - 1)
            @test bond_energies(psi, g) ≈ analytic(L) atol = 1e-12
        end
    end

    @testset "SAME PHYSICS: :none, :U1 and :SU2 all agree on the observable" begin
        # The claim the whole L=18 benchmark rests on -- symmetry changes the storage, not the
        # state. Compared through `bond_energies`, which is representation-free and so can equate
        # three states held in completely different formats: one dense sector (:none), charge
        # sectors (:U1), and total-spin multiplets (:SU2).
        for L in (4, 6, 8)
            profiles = Dict{Symbol, Vector{Float64}}()
            dims = Dict{Symbol, Vector{Int}}()
            for sym in (:none, :U1, :SU2)
                set_symmetry!(sym)
                p = dimer_state(L)
                profiles[sym] = bond_energies(p, bond_gates(p; delta = 1.0))
                dims[sym] = bond_dims(p)
            end
            @test maximum(abs.(profiles[:none] - profiles[:U1]))  < 1e-12
            @test maximum(abs.(profiles[:none] - profiles[:SU2])) < 1e-12   # measured 2.2e-16
            @test maximum(abs.(profiles[:U1]   - profiles[:SU2])) < 1e-12
            # ...while the STORAGE differs: SU(2) holds one multiplet where the abelian modes
            # hold two states. This is why bond dimension may not be compared across symmetries.
            @test dims[:none] == dims[:U1]
            @test all(dims[:SU2] .<= dims[:none])
        end
    end

    @testset "bond_energies sums to energy" begin
        for sym in (:none, :U1, :SU2)
            set_symmetry!(sym)
            psi = dimer_state(8)
            g = bond_gates(psi; delta = 1.0)
            @test sum(bond_energies(psi, g)) ≈ energy(psi, g) atol = 1e-12
        end
    end

    @testset "magnetisation is NOT usable under SU(2) -- why bond_energies exists" begin
        # Not a wish: `local_space(:SU2)` exposes `(:I, :S)` with no `Sz`, so the light-cone
        # observable used everywhere else cannot even be constructed here. Physically right too --
        # a total-spin multiplet has <Sz_j> identically 0, so the plot would be blank.
        set_symmetry!(:SU2)
        psi = dimer_state(6)
        @test_throws Exception magnetisation(copy(psi))
        # while the SU(2)-scalar observable is fine on the same state
        @test length(bond_energies(psi, bond_gates(psi; delta = 1.0))) == 5
    end

    @testset "RSVD-CBE GROWS FROM A PRODUCT STATE where the standard BUG cannot" begin
        # THE CAPABILITY CLAIM FOR CBE, TESTED RATHER THAN ASSERTED.
        #
        # `kls_bond_update` (standard BUG) is fixed-rank against a product state: the frames it
        # builds span only what the state already spans, so chi=1 in gives chi=1 out and cooling
        # toward a singlet cannot start. CBE breaks that by widening the bond BEFORE the update,
        # with directions drawn through the gate, so the evolution has somewhere to go.
        #
        # BOTH SELECTION RULES ARE TESTED -- the sketch (`exact=false`, the rsvd_cbe arm) and the
        # complete fused basis (`exact=true`) -- because they are two implementations of one
        # scheme and "the sketch grows rank" is a separate claim from "the exact basis does".
        #
        # THE CONTROL IS LOAD-BEARING AND MUST NOT BE VACUOUS. An earlier version of this testset
        # asked for the `:odd` group while putting gates on bonds 1,3,5,…, so every gate in the
        # group was `nothing`, the sweep skipped every bond, and the control "passed" by doing
        # nothing whatsoever -- and the CBE arms failed for the same reason. `group_gates` below
        # derives the gate list FROM `parity_bonds`, so the two cannot drift apart again.
        L = 8
        target = [isodd(b) ? -0.75 : 0.0 for b in 1:(L - 1)]
        PAR = :even                     # `:even` IS bonds 1,3,5,… -- the names are 0-based

        "Gates only on `parity`'s own bonds, taken from `parity_bonds` so they cannot mismatch."
        function group_gates(psi, parity)
            g = bond_gates(psi; delta = 1.0)
            keep = Set(parity_bonds(length(psi), parity))
            return Any[i in keep ? g[i] : nothing for i in 1:(length(psi) - 1)], g
        end

        function cool(sym, update!; nst = 60, tau = ComplexF64(-0.5))
            set_symmetry!(sym)
            psi = neel_state(L)
            gg, gfull = group_gates(psi, PAR)
            for _ in 1:nst
                update!(psi, gg, tau)
                n = norm(psi)
                n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
            end
            return psi, bond_energies(psi, gfull)
        end

        # Run in BOTH abelian modes. `:SU2` is excluded not by preference but by
        # representability: `neel_state` cannot exist there (a definite-Sz product state spans
        # many total-spin sectors), so there is no product state to bootstrap FROM.
        #
        # WHAT THE STANDARD BUG DOES IS SYMMETRY-DEPENDENT, and stating it without that
        # qualifier was wrong. MEASURED:
        #
        #   :none  kls, every variant           frozen [1,1,…], 5.0e-01 from target
        #   :U1    kls default                  GROWS  [2,1,2,…], 1.1e-16
        #   :U1    kls missing_fill=0           frozen [1,1,…], 5.0e-01
        #   :U1    kls augment=false            frozen [1,1,…], 5.0e-01
        #
        # So the U(1) growth is ENTIRELY `missing_fill`: a RANDOM column seeded into an
        # empty-but-reachable charge sector. A Neel bond carries one U(1) sector and a singlet
        # needs a second, so under `:U1` there is a sector to seed and under `:none` there is
        # not -- one dense sector, nothing empty, nothing to fill.
        #
        # THAT IS THE DISTINCTION WORTH PINNING. The old kernel buys rank by random filling,
        # and only where a symmetry hands it somewhere to put the random column. CBE takes its
        # directions THROUGH THE GATE, so it grows in every mode, with no random fill anywhere
        # (`cbe_core.jl` contains no `missing_fill`/`pad` at all).
        for sym in (:none, :U1)
            # `cbe_rsvd` (exact = false) IS the method; exact CBE is the A/B reference, run
            # beside it to show the sketch selects the same directions rather than merely
            # inflating rank. Measured [2,1,2,1,2,1,2] at 1e-16 for both, in both modes.
            for exact in (false, true)
                psi, be = cool(sym, (p, gs, t) ->
                    cbe_bond_update_sweep!(p, gs, PAR, t;
                                           maxdim = 8, trunc_thresh = 1e-14,
                                           s_iters = -1, exact = exact))
                @test maximum(bond_dims(psi)) >= 2        # rank grew from 1
                @test maximum(abs.(be - target)) < 1e-6   # and grew toward the RIGHT state
                @test norm(psi) ≈ 1.0 atol = 1e-8
            end
        end

        # THE GROWTH IS NOT SEED-DRIVEN, and `:none` above is the proof rather than an aside:
        # a random sector seed needs an empty-but-reachable CHARGE SECTOR to put its column in,
        # `:none` has a single dense sector so no such seed is expressible there, and CBE still
        # reaches the singlet. Whatever supplies that rank, it is not sector filling.
    end

    @testset "HARD CONSTRAINT: the CBE source contains no random fill or padding" begin
        # A behavioural test cannot prove absence -- it can only fail to find it on the states
        # it happens to try. This one is a source invariant: `missing_fill` (random columns into
        # empty-but-reachable sectors) and `pad` (fill to a target rank) must not appear in the
        # CBE path AT ALL, so rank can only come from directions drawn through the gate.
        #
        # Comments are stripped first, because the header legitimately DISCUSSES `missing_fill`
        # to explain what CBE replaces it with -- the ban is on calling it, not naming it.
        src = read(joinpath(dirname(dirname(@__DIR__)),
                            "src", "RSVDCBEBondUpdate", "cbe_core.jl"), String)
        code = join([replace(l, r"#.*$" => "") for l in split(src, '\n')], "\n")
        @test !occursin("missing_fill", code)
        @test !occursin(r"\bpad\s*=", code)
        @test !occursin("augmented_left_isometry", code)
        @test !occursin("augmented_right_isometry", code)
    end

    set_symmetry!(:U1)
end
