using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: GroundState
import Random

# SU(2) support. Under a NON-ABELIAN symmetry the MPS stores reduced matrix elements on
# total-spin multiplets, which changes three things that the rest of the suite may assume:
#
#   * the operator algebra -- the local space exposes `:S` and `:I` ONLY. There is no `:Sp`,
#     `:Sz` or `:Sm`, because none of them is an SU(2) tensor by itself, so the coupling is one
#     irreducible `S.S` term instead of three;
#   * which STATES exist -- every representable state has a definite total spin, so a Neel or
#     domain-wall state is not expressible at any bond dimension;
#   * what `bond_dims` counts -- multiplets, not states. Not comparable with U(1) numbers.
#
# The tests below pin all three, plus the overall constant, which is the one thing that could
# silently be wrong: Telum normalises through Clebsch-Gordan factors, and the `:none` gate
# branch already carries a MEASURED -1 for exactly that reason. Here the constant turned out to
# need no correction, and this is what would catch it if that ever changed.

"Exact ground energy in the half-filling sector, by dense diagonalisation."
function _su2_exact_ground(L::Int; delta::Float64 = 1.0, J::Float64 = 1.0)
    sz(s) = s == 1 ? 0.5 : -0.5
    bit(b, j) = (b >> (j - 1)) & 1
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    idx = Dict(s => k for (k, s) in enumerate(states))
    H = zeros(Float64, length(states), length(states))
    for (k, b) in enumerate(states), j in 1:(L - 1)
        sj, sj1 = bit(b, j), bit(b, j + 1)
        H[k, k] += J * delta * sz(sj) * sz(sj1)
        sj == sj1 || (H[idx[b ⊻ (1 << (j - 1)) ⊻ (1 << j)], k] += J / 2)
    end
    return eigen(Symmetric(H)).values[1]
end

@testset "SU(2)" begin

    @testset "the local space exposes S, not Sp/Sz/Sm" begin
        set_symmetry!(:SU2)
        q = local_space(:SU2)
        @test :S in propertynames(q)
        @test :I in propertynames(q)
        # The reason `xxz_chain` cannot work here, asserted rather than assumed.
        @test !(:Sp in propertynames(q))
        @test !(:Sz in propertynames(q))
        # `S` must be rank 3 with an op-leg -- the SAME shape as `:U1`'s `Sp`, which is why the
        # channel-environment machinery takes it unmodified.
        @test length(q.S.inds) == 3
        @test length(q.I.inds) == 2
        set_symmetry!(:U1)
    end

    @testset "dimer_state is a normalised S = 0 state" begin
        set_symmetry!(:SU2)
        for L in (4, 6, 8)
            d = dimer_state(L)
            @test norm(d) ≈ 1.0 atol = 1e-12
            # One multiplet per link. NOT one state -- the physical state is entangled.
            @test bond_dims(d) == ones(Int, L - 1)
        end
        @test_throws ArgumentError dimer_state(5)      # odd L has no dimer covering
        @test_throws ArgumentError dimer_state(0)
        # `dimer_state` USED TO REFUSE outside `:SU2`, and this line asserted that. It no
        # longer does, deliberately: the L=18 benchmark needs one state constructible in ALL
        # THREE symmetry modes, because "symmetry changes the cost, not the physics" cannot be
        # tested at all without a common starting state. `product_state`/`neel_state` are
        # guarded off under `:SU2`, so the dimer covering is the only candidate, and it is now
        # built by `_dimer_state_projected` in the abelian modes.
        #
        # The assertion is therefore inverted rather than deleted: it now pins that the state
        # IS available everywhere, which is the property the benchmark depends on. The stronger
        # claim -- that all three modes agree on the observable -- lives in `test_dimer.jl`,
        # which measures 2.2e-16 between `:none` and `:SU2`.
        for sym in (:none, :U1)
            set_symmetry!(sym)
            d = dimer_state(4)
            @test norm(d) ≈ 1.0 atol = 1e-12
            # chi = 2 on the dimerised bonds, where SU(2) stores one multiplet.
            @test bond_dims(d) == [2, 1, 2]
        end
        set_symmetry!(:U1)
    end

    @testset "product_state refuses under :SU2 instead of returning nonsense" begin
        set_symmetry!(:SU2)
        # THIS IS A REGRESSION TEST FOR A SILENT WRONG ANSWER, not for an obvious error.
        # Before the guard, the U(1) code path ran to completion here and returned
        # `bond_dims = [1,1,1]`: `SECTOR_UP` is `((1,),)` and the SU(2) spin-1/2 irrep is ALSO
        # `((1,),)`, so `:up` sites matched the getsub filter while `:down` sites selected
        # nothing at all. It looked like a state and was not one.
        @test_throws ArgumentError product_state([:up, :down, :up, :down])
        @test_throws ArgumentError neel_state(4)
        set_symmetry!(:U1)
    end

    @testset "xxz_chain and anisotropic gates refuse under :SU2" begin
        set_symmetry!(:SU2)
        # Fails because it reads `q.Sp`. Kept as a test so the failure stays LOUD if someone
        # later adds an `:SU2` branch to the abelian constructor by mistake.
        @test_throws Exception xxz_chain(6; delta = 1.0)
        set_symmetry!(:U1)
    end

    @testset "heisenberg_su2_chain is one term and reproduces the exact ground energy" begin
        set_symmetry!(:SU2)
        L = 8
        h = heisenberg_su2_chain(L)
        # One irreducible S.S term, against the abelian form's three.
        @test length(hamiltonian_terms(h)) == 1

        p = dimer_state(L)
        info = GroundState.rsvd_cbe_dmrg!(p, h;
            opts = CBEBugOptions(maxdim = 32, trunc_thresh = 1e-13),
            n_sweeps = 40, etol = 1e-12)
        e = info.energies[end]
        e0 = _su2_exact_ground(L)

        # THE CONSTANT. Telum's CG normalisation could put an arbitrary factor here; measured,
        # it does not. A ratio test rather than a difference so a wrong constant reads off
        # directly as the factor instead of as an opaque discrepancy.
        @test e / e0 ≈ 1.0 atol = 1e-10
        @test e ≈ e0 atol = 1e-10
        @test e >= e0 - 1e-10            # still variational under SU(2)
        set_symmetry!(:U1)
    end

    @testset "SU(2) and U(1) agree on the energy, SU(2) with fewer stored numbers" begin
        L = 8
        e0 = _su2_exact_ground(L)

        set_symmetry!(:SU2)
        a = dimer_state(L)
        ia = GroundState.rsvd_cbe_dmrg!(a, heisenberg_su2_chain(L);
            opts = CBEBugOptions(maxdim = 32, trunc_thresh = 1e-13),
            n_sweeps = 40, etol = 1e-12)

        set_symmetry!(:U1)
        b = neel_state(L)
        ib = GroundState.rsvd_cbe_dmrg!(b, xxz_chain(L; delta = 1.0);
            opts = CBEBugOptions(maxdim = 32, trunc_thresh = 1e-13),
            n_sweeps = 40, etol = 1e-12)

        # The physics does not change: same ground energy from both symmetry settings.
        @test ia.energies[end] ≈ ib.energies[end] atol = 1e-9
        @test ia.energies[end] ≈ e0 atol = 1e-9

        # And it is cheaper. NOT a bit-for-bit trajectory comparison: Neel is not representable
        # under SU(2), so the two runs start from different (both bond-dimension-1) states and
        # only the converged state is shared. Measured 126 vs 425 matvecs at L=8; the assertion
        # is loose so it tests the direction of the effect and not one machine's numbers.
        @test sum(ia.krylov_dims) < sum(ib.krylov_dims)
        # maxbd counts MULTIPLETS under SU(2) and STATES under U(1) -- comparing the integers
        # is meaningful only as "the reduced basis is smaller", which is what this asserts.
        @test maximum(ia.max_bond_dims) < maximum(ib.max_bond_dims)
    end
end

set_symmetry!(:U1)      # leave the global mode as the rest of the suite expects
