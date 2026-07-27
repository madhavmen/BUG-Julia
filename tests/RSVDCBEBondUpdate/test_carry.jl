using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: ChannelSet, ZERO
import Random

# Carrying the channels instead of rebuilding a stack per bond is what makes a step O(L)
# rather than O(L^2). It is only sound if the carry produces EXACTLY the stack, so that is
# what this file checks, link by link.
#
# A stale environment does not throw -- it silently evolves the state with the wrong
# Hamiltonian -- so this cannot be left to the integrator-level tests to notice.

"Compare one channel against a stack entry, tolerating the ZERO / nothing sentinels."
function same_channel(a, b; atol = 1e-12)
    (a === ZERO || b === ZERO) && return a === ZERO && b === ZERO
    (a === nothing || b === nothing) && return a === nothing && b === nothing
    a.inds == b.inds || return false
    return norm(to_concrete(a - b)) <= atol * max(1.0, norm(b))
end

function same_channels(cs::ChannelSet, id, done, open)
    same_channel(cs.id, id) || return false
    same_channel(cs.done, done) || return false
    length(cs.open) == length(open) || return false
    return all(same_channel(cs.open[t], open[t]) for t in eachindex(open))
end

function evolved(L::Int; n_steps = 3, dt = 0.05, maxdim = 16)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

@testset "channel carry" begin

    @testset "left carry reproduces the left stack" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L)
            st = left_env_stack(psi, h; upto = L - 1)

            cs = boundary_channels(h)
            @test same_channels(cs, st.id[1], st.done[1], st.open[1])
            for i in 1:(L - 1)
                cs = push_left_channels(cs, h, psi[i], i; open_next = i < L)
                @test same_channels(cs, st.id[i + 1], st.done[i + 1], st.open[i + 1])
            end
        end
    end

    @testset "right carry reproduces the right stack" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L)
            st = right_env_stack(psi, h; downto = 2)

            cs = boundary_channels(h)
            @test same_channels(cs, st.id[L + 1], st.done[L + 1], st.open[L + 1])
            for i in L:-1:2
                cs = push_right_channels(cs, h, psi[i], i; open_next = i > 1)
                @test same_channels(cs, st.id[i], st.done[i], st.open[i])
            end
        end
    end

    @testset "the kernels agree whether given a carry or a stack" begin
        # `cbe_expand` and `zero_site_h` take the two links they use; the stack-indexed
        # wrappers must be exactly the same call.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        for i in 1:(L - 1)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            lst = left_env_stack(psi, h; upto = i - 1)
            rst = right_env_stack(psi, h; downto = i + 2)
            lch = left_channels(lst, i)
            rch = right_channels(rst, i + 2)

            a = cbe_expand(f, h, i, lst, rst; rng = Random.MersenneTwister(7))
            b = cbe_expand(f, h, i, lch, rch; rng = Random.MersenneTwister(7))
            @test norm(to_concrete(a.U_ex - b.U_ex)) < 1e-12
            @test norm(to_concrete(a.V_ex - b.V_ex)) < 1e-12

            S = to_concrete(contract(contract(a.U_ex', (1, 2), frame_theta(f), (1, 2)),
                                     (2, 3), a.V_ex', (2, 3)))
            ha = zero_site_h(h, i, lst, rst, a.U_ex, a.V_ex)
            hb = zero_site_h(h, i, lch, rch, a.U_ex, a.V_ex)
            @test norm(to_concrete(apply_zero_site(ha, S) -
                                   apply_zero_site(hb, S))) < 1e-12
        end
    end

    @testset "carry is gauge-exact along a sweep" begin
        # The sweep reads the carry after `_absorb_left!` has rewritten psi[i], so the
        # carry must equal a stack rebuilt from the UPDATED state -- this is the property
        # the ordering constraint in `cbe_bug_bond_update` exists to protect.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        canonical!(psi, 1)
        rstack = right_env_stack(psi, h; downto = 3)
        lch = boundary_channels(h)
        c = L ÷ 2
        for i in 1:(c - 1)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            ex = cbe_expand(f, h, i, lch, right_channels(rstack, i + 2);
                            rng = Random.MersenneTwister(3))
            # the right stack must still describe the untouched right side
            fresh_r = right_env_stack(psi, h; downto = i + 2)
            @test same_channel(right_channels(rstack, i + 2).done,
                               right_channels(fresh_r, i + 2).done)
            RSVDCBEBondUpdate._absorb_left!(psi, i, ex.U_ex)
            lch = push_left_channels(lch, h, psi[i], i)
            fresh_l = left_env_stack(psi, h; upto = i)
            @test same_channels(lch, fresh_l.id[i + 1], fresh_l.done[i + 1],
                                fresh_l.open[i + 1])
        end
    end
end
