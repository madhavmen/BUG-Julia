using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# Stage 1b: the right recursion and the two-site effective action.
#
# THE DIAGONAL IS NOT ENOUGH. `tensor_inner(Θ, HΘ) == ⟨ψ|H|ψ⟩` would pass even if
# `apply_h_two_site` returned a wrong operator with the right diagonal element -- and the
# likely failure here (an environment applied to the wrong leg, or a channel closed with
# the wrong half of a term) is exactly that shape. So the main test is OFF-diagonal:
# build a second block `Θ'` inside the SAME frame subspace, splice it back into an MPS,
# and require
#
#     tensor_inner(Θ', H Θ)  ==  ⟨ψ'|H|ψ⟩        (dense reference)
#
# which pins every matrix element of the projected operator, not just one.

"A random centre core in `S0`'s own sparsity pattern -- charge sectors untouched, so
`Θ'` stays in the frame subspace and in the sectors the symmetry allows."
function random_core_block(f::BondFrame, rng)
    S = f.S0
    R = TLArray(symm(S), copy(S.qlabels), copy(S.wmatdata), copy(S.wmatinfo),
                [randn(rng, eltype(S), size(r)) for r in S.RMTs], S.inds, S.spaces)
    return to_concrete((f.U0 * to_concrete(R)) * f.V0)
end

"`psi` with the two-site block at bond `i` replaced by `Theta`."
function splice_block(psi::SymMPS, i::Int, Theta)
    res = svd(Theta, (1, 2); cutoff = 0.0)
    tag = psi[i].inds[3].itags
    out = copy(psi)
    out[i]     = to_concrete(setitag(res.U, 3, tag))
    out[i + 1] = to_concrete(setitag(to_concrete(res.S * res.Vd), 1, tag))
    out.center = i + 1
    return out
end

"A state with real entanglement: a few BUG steps off a product state."
function evolved_state(L::Int; n_steps = 3, dt = 0.05)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = 16))
    return psi
end

@testset "heff (right stack + two-site action)" begin

    @testset "right recursion reproduces the energy" begin
        # Shares no contraction with the left sweep, so this is an independent pin on
        # the mirrored asymmetry: a right channel OPENS with term.right and CLOSES with
        # term.left, with the op-leg arrows the other way round.
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            for psi in (domain_wall_state(L), neel_state(L), evolved_state(L))
                v = dense_state(psi)
                want = v' * (dense_heisenberg(L; delta = delta) * v)
                @test abs(env_energy_right(psi, h) - want) < 1e-11
                @test abs(env_energy_right(psi, h) - env_energy(psi, h)) < 1e-11
            end
        end
    end

    @testset "right identity channel is the identity" begin
        set_symmetry!(:U1)
        L = 6
        psi = evolved_state(L)
        canonical!(psi, 1)                         # right-canonical for sites 2 … L
        st = right_env_stack(psi, xxz_chain(L))
        for i in 2:L
            @test abs(norm(st.id[i])^2 - leg_dim(psi[i], 1)) < 1e-10
        end
    end

    @testset "stack bounds at the ends of the chain" begin
        set_symmetry!(:U1)
        L = 6
        psi = evolved_state(L)
        h = xxz_chain(L)
        # Bond 1 needs nothing on its left; bond L-1 nothing on its right.
        @test left_env_stack(psi, h; upto = 0).done[1] === RSVDCBEBondUpdate.ZERO
        @test right_env_stack(psi, h; downto = L + 1).done[L + 1] === RSVDCBEBondUpdate.ZERO
        @test_throws ArgumentError left_env_stack(psi, h; upto = -1)
        @test_throws ArgumentError right_env_stack(psi, h; downto = L + 2)
    end

    @testset "two-site action: diagonal element" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved_state(L)
            want = env_energy(psi, h)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                th = frame_theta(f)
                lenv = left_env_stack(psi, h; upto = i - 1)
                renv = right_env_stack(psi, h; downto = i + 2)
                hth = apply_h_two_site(th, h, i, lenv, renv)
                @test th.inds == hth.inds                  # legs come back unchanged
                @test abs(tensor_inner(th, hth) - want) < 1e-10
            end
        end
    end

    @testset "two-site action: off-diagonal elements vs dense" begin
        set_symmetry!(:U1)
        rng = Random.MersenneTwister(0xC0FFEE)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            H = dense_heisenberg(L; delta = delta)
            psi = evolved_state(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                th = frame_theta(f)
                lenv = left_env_stack(psi, h; upto = i - 1)
                renv = right_env_stack(psi, h; downto = i + 2)
                hth = apply_h_two_site(th, h, i, lenv, renv)

                for _ in 1:3
                    thp = random_core_block(f, rng)
                    psip = splice_block(psi, i, thp)
                    got = tensor_inner(thp, hth)
                    want = dense_state(psip)' * (H * dense_state(psi))
                    @test abs(got - want) < 1e-10 * max(1.0, abs(want))
                end
            end
        end
    end

    @testset "two-site action, no symmetry" begin
        set_symmetry!(:none)
        try
            rng = Random.MersenneTwister(0xBEEF)
            L = 6
            h = xxz_chain(L)
            H = dense_heisenberg(L)
            psi = evolved_state(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                th = frame_theta(f)
                hth = apply_h_two_site(th, h, i,
                                       left_env_stack(psi, h; upto = i - 1),
                                       right_env_stack(psi, h; downto = i + 2))
                @test abs(tensor_inner(th, hth) - env_energy(psi, h)) < 1e-10
                thp = random_core_block(f, rng)
                psip = splice_block(psi, i, thp)
                @test abs(tensor_inner(thp, hth) -
                          dense_state(psip)' * (H * dense_state(psi))) < 1e-10
            end
        finally
            set_symmetry!(:U1)
        end
    end
end
