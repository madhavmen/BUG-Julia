using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# The 2-site TDVP baseline. Not part of CBE-BUG -- it exists so the memory claim can be
# measured against a real competitor at MATCHED ACCURACY rather than against a proxy, so
# it has to be a correct integrator in its own right.
#
# The sharp test is `apply_one_site`: projecting the two-site effective Hamiltonian onto a
# fixed left isometry IS the one-site effective Hamiltonian at the next site, so
#
#     apply_one_site(H1, M)  ==  A† · apply_h_two_site(A M)
#
# and `apply_h_two_site` is already pinned off-diagonally to the dense matrix element.
# Everything else here is convergence behaviour, which is what a baseline is judged on.

function evolved(L::Int; n_steps = 6, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

@testset "tdvp2 baseline" begin

    @testset "one-site action is the projected two-site action" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L)
            for i in 1:(L - 2)
                canonical!(psi, i + 1)             # psi[i] is a left isometry
                A, M = psi[i], psi[i + 1]
                lenv = left_env_stack(psi, h; upto = i - 1)
                renv = right_env_stack(psi, h; downto = i + 2)

                H1 = RSVDCBEBondUpdate.one_site_h(
                        h, i + 1, left_env_stack(psi, h; upto = i), renv)
                got = RSVDCBEBondUpdate.apply_one_site(H1, M)

                Th = to_concrete(A * M)
                want = to_concrete(contract(A', (1, 2),
                                            apply_h_two_site(Th, h, i, lenv, renv), (1, 2)))
                @test got.inds == want.inds
                @test norm(to_concrete(got - want)) < 1e-10 * max(1.0, norm(want))
            end
        end
    end

    @testset "exact at full rank" begin
        # The projector-splitting integrator is exact when the manifold is the full space,
        # for ANY dt -- so this, not a convergence ratio, is the right unit test. MEASURED
        # 8.2e-14 over 2 steps of dt=0.08 and 1.4e-13 over 4 of dt=0.04 (job 94971): both
        # machine noise, and an order-in-dt test placed here would be comparing two
        # epsilons. Order is measured in benchmarks/cbe_lubich_vs_baselines.jl, where the
        # rank is capped below full so there is a dt error to see.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        H = dense_heisenberg(L)
        psi = evolved(L; n_steps = 4, dt = 0.05, maxdim = 32)
        v0 = dense_state(psi)
        dt = 0.04
        for _ in 1:4
            tdvp2_step!(psi, h, -im * dt; maxdim = 32, trunc_thresh = 1e-14)
        end
        @test norm(dense_state(psi) - dense_exact_propagate(H, v0, dt * 4)) < 1e-11
    end

    @testset "conserves charge and norm" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        sz0, n0 = total_sz(copy(psi)), norm(psi)
        for _ in 1:3
            tdvp2_step!(psi, h, -im * 0.02; maxdim = 32, trunc_thresh = 1e-14)
        end
        @test abs(total_sz(copy(psi)) - sz0) < 1e-10
        @test abs(norm(psi) - n0) < 1e-8
    end

    @testset "reports the two-site block it built" begin
        # The memory figure CBE-BUG is measured against: 2-site TDVP cannot avoid
        # materialising a rank-4 block, and this records how big the largest one was.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        info = tdvp2_step!(psi, h, -im * 0.02; maxdim = 32, trunc_thresh = 1e-14)
        @test info.theta_elements > 0
        @test info.max_bond >= 1
    end
end
