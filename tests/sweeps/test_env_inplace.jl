# tests/sweeps/test_env_inplace.jl -- the MPO environment steps and the CBE preselection run their
# intermediates through `to_concrete!` / skip a second copy (allocation, bug_profile_2d.jl), and
# must give BIT-IDENTICAL tensors to the copying form while leaving every input untouched.
#
# Exact equality, not isapprox: the in-place w-matrix cleanup is the one thing that could differ,
# and a silent mutation of a stack entry or site tensor would corrupt every later sweep.

using Test
using LinearAlgebra
import Random

@testset "env steps and perp_component: no copy, same tensors" begin
    R = RSVDCBEBondUpdate
    same(a, b) = a.qlabels == b.qlabels && a.wmatdata == b.wmatdata && a.RMTs == b.RMTs &&
                 a.inds == b.inds && a.spaces == b.spaces

    # The copying forms the steps used before (mpo.jl), kept here as the reference.
    function ref_left(E, A, W)
        if E === nothing
            T = to_concrete(contract(W, (2,), A, (2,)))
            T = to_concrete(contract(T, (2, 4), R._bra_left_boundary(A), (2, 1)))
            T = to_concrete(deleteSingleton(T, 1))
            return to_concrete(permutedims(T, (3, 1, 2)))
        end
        T = to_concrete(contract(E, (3,), A, (1,)))
        T = to_concrete(contract(T, (2, 3), W, (1, 2)))
        T = to_concrete(contract(T, (1, 3), R._bra_interior(A), (1, 2)))
        return to_concrete(permutedims(T, (3, 2, 1)))
    end
    function ref_right(E, A, W)
        if E === nothing
            T = to_concrete(contract(W, (2,), A, (2,)))
            T = to_concrete(contract(T, (2, 5), R._bra_right_boundary(A), (2, 3)))
            T = to_concrete(deleteSingleton(T, 2))
            return to_concrete(permutedims(T, (3, 1, 2)))
        end
        T = to_concrete(contract(A, (3,), E, (3,)))
        T = to_concrete(contract(T, (2, 4), W, (2, 4)))
        T = to_concrete(contract(T, (2, 4), R._bra_interior(A), (3, 2)))
        return to_concrete(permutedims(T, (3, 2, 1)))
    end
    function ref_left2(E, A, W)
        Wa = R._mpo_outer(W)
        T = to_concrete(contract(E, (4,), A, (1,)))
        T = to_concrete(contract(T, (3, 4), W, (1, 2)))
        T = to_concrete(contract(T, (2, 4), Wa, (1, 2)))
        T = to_concrete(contract(T, (1, 4), R._bra_interior(A), (1, 2)))
        return to_concrete(permutedims(T, (4, 3, 2, 1)))
    end

    for sym in (:U1, :SU2)
        set_symmetry!(sym)
        L = 8
        h = sym === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; delta = 1.0)
        mpo = R.mpo_from_terms(h)
        # Charged, multi-sector links: a vacuum-link state would hide a sector-order slip.
        psi = dimer_state(L)
        for _ in 1:3
            tdvp2_step!(psi, h, -0.1im; maxdim = 64, trunc_thresh = 1e-14)
        end
        @test maximum(bond_dims(psi)) > 1

        E, E2 = nothing, nothing
        for i in 1:(L - 1)
            A, W = psi[i], mpo[i]
            A0, W0 = deepcopy(A), deepcopy(W)
            E0 = deepcopy(E)
            new = R._mpo_left_step(E, A, W)
            @test same(new, ref_left(E0, A0, W0))
            @test same(A, A0) && same(W, W0)
            E === nothing || @test same(E, E0)
            if E2 !== nothing
                E20 = deepcopy(E2)
                @test same(R._mpo2_left_step(E2, A, W), ref_left2(E20, A0, W0))
                @test same(E2, E20)
            end
            E = new
            E2 = R._mpo2_left_step(E2, A, W)
        end
        E = nothing
        for i in L:-1:2
            A, W = psi[i], mpo[i]
            A0, W0, E0 = deepcopy(A), deepcopy(W), deepcopy(E)
            new = R._mpo_right_step(E, A, W)
            @test same(new, ref_right(E0, A0, W0))
            @test same(A, A0) && same(W, W0)
            E === nothing || @test same(E, E0)
            E = new
        end
        @test mpo_variance(psi, mpo) >= -1e-10

        # perp_component already returns a fresh copy; the preselection no longer copies it
        # again, so it must equal its own `to_concrete` and leave both inputs alone -- including
        # the sectors of `Y` that `U0` lacks, which the Telum sum hands through by reference.
        canonical!(psi, L)
        U0 = psi[L ÷ 2]
        Y = R._randomize_like(U0, Random.MersenneTwister(7))
        U00, Y0 = deepcopy(U0), deepcopy(Y)
        P = perp_component(U0, Y)
        @test same(P, to_concrete(P))
        @test same(U0, U00) && same(Y, Y0)
        @test norm(contract(U0', (1, 2), P, (1, 2))) < 1e-12 * max(norm(Y), 1.0)
    end
    set_symmetry!(:U1)
end
