using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# The Hamiltonian as a genuine MPO and its rank-3 environments -- Eqs. (1.3), (1.7), (1.8)
# of arXiv:2606.28169 -- against the channel recursion of `henv.jl` and against the dense
# matrix element.
#
# WHY THE COMPARISON IS THE TEST. `mpo.jl` and `henv.jl` are two storage layouts of ONE
# operator, and they share no contraction: the MPO path fuses the virtual index into a leg
# and contracts it, the channel path keeps one tensor per automaton state and enumerates
# them. So a leg, arrow or prime slip in either cannot cancel in the other, and
# `apply_h_two_site` agreeing elementwise at every bond is a stronger statement than any
# single scalar. The dense reference then pins BOTH to the analytic matrix element.
#
# The sweep-level test is the one that matters for the integrator: with the same seed, the
# MPO path and the channel path must produce the SAME state, bond dimension for bond
# dimension, because the expansion they drive is the same selection on the same H.

"A state with real entanglement: a few validated BUG steps off a product state."
function _mpo_evolved(L::Int; n_steps = 6, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

@testset "mpo (Eq. 1.3) and its environments (Eq. 1.8)" begin

    @testset "structure: one rank-4 tensor per site, w = 2 + |terms|" begin
        set_symmetry!(:U1)
        for L in (2, 4, 6)
            # Heisenberg: three terms (Sp-pair, Sm-pair, SzSz) -> w = 5.
            mpo = xxz_mpo(L; delta = 1.0)
            @test length(mpo) == L
            for i in 1:L
                @test length(mpo[i].inds) == 4
                # (w_l '+', s_ket '+', s_bra '-', w_r '-'): an operator's own layout with a
                # virtual leg glued to each side.
                @test [x.dir for x in mpo[i].inds] == ['+', '+', '-', '-']
                @test mpo[i].inds[2].itags == "S,$i"
                @test mpo[i].inds[3].itags == "S,$i"
            end
            dims = mpo_virtual_dims(mpo)
            @test length(dims) == L + 1
            @test dims[1] == 1 && dims[end] == 1        # boundary vectors, built in
            @test all(==(5), dims[2:(end - 1)])
            # XX drops SzSz, so the virtual space loses that channel.
            @test all(==(4), mpo_virtual_dims(xxz_mpo(L; delta = 0.0))[2:(end - 1)])
        end
    end

    @testset "mpo_energy == env_energy == dense, in every symmetry mode" begin
        for sym in (:U1, :none)
            set_symmetry!(sym)
            for L in (2, 4, 6), delta in (1.0, 0.0)
                h = xxz_chain(L; delta = delta)
                mpo = mpo_from_terms(h)
                psi = _mpo_evolved(L)
                e_mpo = mpo_energy(psi, mpo)
                e_ch = env_energy(psi, h)
                v = dense_state(psi)
                e_dense = v' * (dense_heisenberg(L; delta = delta) * v)
                @test abs(e_mpo - e_ch) < 1e-10 * max(1.0, abs(e_ch))
                @test abs(e_mpo - e_dense) < 1e-10 * max(1.0, abs(e_dense))
            end
        end
        # SU(2): ONE term, whose S=1 op-leg is the MPO bond, so w = 3 multiplets.
        set_symmetry!(:SU2)
        for L in (2, 4, 6)
            h = heisenberg_su2_chain(L)
            mpo = mpo_from_terms(h)
            @test all(==(3), mpo_virtual_dims(mpo)[2:(end - 1)])
            psi = dimer_state(L)
            @test abs(mpo_energy(psi, mpo) - env_energy(psi, h)) < 1e-10
        end
        set_symmetry!(:U1)
    end

    @testset "carried environments == prebuilt stacks" begin
        # The sweep CARRIES its own side (O(1) per bond) and READS the other from a stack.
        # Both are the same recursion, and this is what says so.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        mpo = mpo_from_terms(h)
        psi = _mpo_evolved(L)
        canonical!(psi, 1)
        lst = left_env_stack(psi, mpo; upto = L - 1)
        cur = boundary_channels(mpo)
        for i in 1:(L - 1)
            cur = push_left_channels(cur, mpo, psi[i], i)
            got, want = cur.E, left_channels(lst, i + 1).E
            @test norm(to_concrete(got - want)) < 1e-12 * max(1.0, norm(want))
        end
        rst = right_env_stack(psi, mpo; downto = 2)
        cur = boundary_channels(mpo)
        for i in L:-1:2
            cur = push_right_channels(cur, mpo, psi[i], i)
            got, want = cur.E, right_channels(rst, i).E
            @test norm(to_concrete(got - want)) < 1e-12 * max(1.0, norm(want))
        end
    end

    @testset "apply_h_two_site: MPO == channels, elementwise at every bond" begin
        set_symmetry!(:U1)
        for L in (2, 4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            mpo = mpo_from_terms(h)
            psi = _mpo_evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                Theta = frame_theta(bond_frame(psi, i))
                want = apply_h_two_site(Theta, h, i,
                                        left_env_stack(psi, h; upto = i - 1),
                                        right_env_stack(psi, h; downto = i + 2))
                got = apply_h_two_site(Theta, mpo, i,
                                       left_env_stack(psi, mpo; upto = i - 1),
                                       right_env_stack(psi, mpo; downto = i + 2))
                @test got.inds == want.inds
                @test norm(to_concrete(got - want)) < 1e-10 * max(1.0, norm(want))
            end
        end
    end

    @testset "0-site Galerkin operator: MPO == the two-site route" begin
        # Eq. (1.7) at a bond. Same claim the channel path makes in `test_cbe_lubich.jl`,
        # re-made on the MPO environments, plus the structural memory claim: nothing in the
        # Krylov loop is rank 4.
        set_symmetry!(:U1)
        rng = Random.MersenneTwister(0x5EED)
        for L in (4, 6), delta in (1.0, 0.0)
            mpo = xxz_mpo(L; delta = delta)
            psi = _mpo_evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                lenv = left_env_stack(psi, mpo; upto = i - 1)
                renv = right_env_stack(psi, mpo; downto = i + 2)
                ex = cbe_expand(f, mpo, i, left_channels(lenv, i),
                                right_channels(renv, i + 2); rng = rng)

                H0 = RSVDCBEBondUpdate.zero_site_h(mpo, i, left_channels(lenv, i),
                                                   right_channels(renv, i + 2),
                                                   ex.U_ex, ex.V_ex)
                @test length(H0.l.inds) == 3 && length(H0.r.inds) == 3

                S = to_concrete(contract(contract(ex.U_ex', (1, 2), frame_theta(f), (1, 2)),
                                         (2, 3), ex.V_ex', (2, 3)))
                got = RSVDCBEBondUpdate.apply_zero_site(H0, S)
                @test length(got.inds) == 2

                th = to_concrete((ex.U_ex * S) * ex.V_ex)
                hth = apply_h_two_site(th, mpo, i, left_channels(lenv, i),
                                       right_channels(renv, i + 2))
                want = to_concrete(contract(contract(ex.U_ex', (1, 2), hth, (1, 2)),
                                            (2, 3), ex.V_ex', (2, 3)))
                @test norm(to_concrete(got - want)) < 1e-10 * max(1.0, norm(want))
            end
        end
    end

    @testset "cbe_lubich_sweep on the MPO path == on the channel path" begin
        # The integrator-level statement: swapping the representation of H changes nothing
        # about the step, because the two representations are one operator.
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            mpo = mpo_from_terms(h)
            tau = ComplexF64(-im * 0.05)
            psi_ch = _mpo_evolved(L; n_steps = 3, maxdim = 8)
            psi_mp = copy(psi_ch)
            for step in 1:3
                info_ch = cbe_lubich_sweep(psi_ch, h, tau;
                                           rng = Random.MersenneTwister(0x5EED + step))
                info_mp = cbe_lubich_sweep(psi_mp, mpo, tau;
                                           rng = Random.MersenneTwister(0x5EED + step))
                @test bond_dims(psi_mp) == bond_dims(psi_ch)
                @test info_mp.expanded == info_ch.expanded
                @test abs(norm(psi_mp) - norm(psi_ch)) < 1e-10
                # NOT equal, and it should not be asserted equal: `krylov_dim` counts
                # operator applications, and the two paths reach the same subspace by
                # different arithmetic. Measured at L=4/6: the MPO path breaks down at
                # dimension 5 while the channel path runs to `maxiter = 30`, because
                # `apply_zero_site` there is a SUM of separately-rounded channel
                # contributions and that noise keeps beta off zero. `lanczos_expv` exits
                # on breakdown only, so the noise costs 6x the matvecs for the same
                # answer -- the state agreement below is the claim, this is a bonus.
                @test info_mp.krylov_dim <= info_ch.krylov_dim
            end
            # Same state, not merely the same diagnostics.
            @test norm(dense_state(psi_mp) - dense_state(psi_ch)) < 1e-9
        end
    end

    @testset "cbe_lubich_bug! builds the MPO internally" begin
        # The driver converts a term list ONCE and every step then runs on `W^[i]`. Handing it
        # the MPO directly must therefore be indistinguishable -- same state, same ranks, same
        # Krylov depth -- which is what says the conversion is where it claims to be and not
        # per step.
        set_symmetry!(:U1)
        L = 4
        h = xxz_chain(L)
        o = CBEBugOptions(dt = 0.05, n_steps = 2, maxdim = 8)
        p1 = domain_wall_state(L); r1 = cbe_lubich_bug!(p1, h; opts = o)
        p2 = domain_wall_state(L); r2 = cbe_lubich_bug!(p2, mpo_from_terms(h); opts = o)
        @test r1.bond_dims == r2.bond_dims
        @test r1.krylov_dims == r2.krylov_dims
        @test norm(dense_state(p1) - dense_state(p2)) < 1e-12
    end

    @testset "exact at full rank, MPO path" begin
        # At L=4 with saturated bonds the centre projector is the identity, so the single
        # 0-site Galerkin step must reproduce exp(-iH dt) exactly. Nothing in the sweep,
        # the environment push or the 0-site apply can hide there.
        set_symmetry!(:U1)
        L, dt = 4, 0.02
        for delta in (1.0, 0.0)
            mpo = xxz_mpo(L; delta = delta)
            psi = _mpo_evolved(L; n_steps = 8, dt = 0.05, maxdim = 32)
            @test bond_dims(psi) == [min(2^i, 2^(L - i)) for i in 1:(L - 1)]
            v0 = dense_state(psi)
            want = exp(-im * dt * Matrix(dense_heisenberg(L; delta = delta))) * v0
            cbe_lubich_sweep(psi, mpo, ComplexF64(-im * dt); maxdim = 32)
            @test norm(dense_state(psi) - want) < 1e-9
        end
    end
end
