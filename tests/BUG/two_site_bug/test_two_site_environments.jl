# test_two_site_environments.jl
#
# The environments must be updated PROPERLY as the single orthogonality centre
# moves along the chain. Concretely:
#   * after `orthogonalize!(psi, b)` the snapshot's left factor U0 is a left
#     isometry and the right factor V0 is a right isometry ÔÇö i.e. the canonical
#     environment seen by the local KLS step is the identity,
#   * the snapshot factorizes the two-site block exactly: U0┬ÀS0┬ÀV0 == ¤ê[b]┬À¤ê[b+1],
#   * the centre can be transported across every bond (both parities) while the
#     state stays a valid normalized MPS,
#   * the incremental MPO environment builders agree with a from-scratch full
#     recontraction (the environment machinery the BUG infrastructure relies on).

using Test, ITensors, ITensorMPS, LinearAlgebra, Random
include(joinpath(@__DIR__, "two_site_test_setup.jl"))

function _is_left_isometry(U, link_l, site_l, mid; atol = 1e-12)
    M = reshape(ComplexF64.(Array(U, link_l, site_l, mid)),
                dim(link_l) * dim(site_l), dim(mid))
    return norm(M' * M - I(dim(mid))) <= atol
end

function _is_right_isometry(V, mid, site_r, link_r; atol = 1e-12)
    M = reshape(ComplexF64.(Array(V, mid, site_r, link_r)),
                dim(mid), dim(site_r) * dim(link_r))
    return norm(M * M' - I(dim(mid))) <= atol
end

@testset "two-site environment / gauge handling (N=6 XX)" begin
    N = 6
    sites = siteinds("S=1/2", N)
    W_odd, W_even, _ = two_site_xx_parity_mpos(sites; J = 1.0)

    @testset "canonical environment is the identity at every bond" begin
        psi = two_site_rank3_state(sites; seed = 21)
        for b in 1:(N - 1)
            orthogonalize!(psi, b)
            bd = BUG._two_site_bond_snapshot(psi, b)
            @test _is_left_isometry(bd.U0_tens, bd.link_l, bd.site_l, bd.canon_u0)
            @test _is_right_isometry(bd.V0_tens, bd.canon_v0, bd.site_r, bd.link_r)
        end
    end

    @testset "snapshot factorizes the two-site block exactly" begin
        psi = two_site_rank3_state(sites; seed = 22)
        for b in 1:(N - 1)
            orthogonalize!(psi, b)
            bd = BUG._two_site_bond_snapshot(psi, b)
            block_snapshot = bd.U0_tens * bd.S0_tens * bd.V0_tens
            block_direct   = psi[b] * psi[b + 1]
            @test norm(block_snapshot - block_direct) <= 1e-12 * max(norm(block_direct), 1.0)
        end
    end

    @testset "centre transport keeps a valid normalized MPS" begin
        psi = two_site_rank3_state(sites; seed = 23)
        v_ref = two_site_vec(psi)
        for b in 1:(N - 1)            # forward across all bonds (both parities)
            orthogonalize!(psi, b)
            @test abs(norm(psi) - 1.0) <= 1e-10
            # gauge transport is exact: the physical state is unchanged
            @test two_site_infidelity(two_site_vec(psi), v_ref) <= 1e-12
        end
    end

    @testset "incremental MPO environments == full recontraction" begin
        # Reused env machinery: advancing one site at a time must match the
        # batch builders, for both parity MPOs.
        psi = two_site_rank3_state(sites; seed = 24)
        orthogonalize!(psi, 1)
        for W in (W_odd, W_even)
            L_all = TTutils._build_left_envs_mpo(psi, W)
            R_all = TTutils._build_right_envs_mpo(psi, W)

            L = TTutils._left_env_boundary_mpo(psi, W)
            @test norm(L - L_all[1]) <= 1e-12 * max(norm(L_all[1]), 1.0)
            for k in 1:N
                L = TTutils._advance_left_env_mpo(L, psi[k], W[k])
                @test norm(L - L_all[k + 1]) <= 1e-12 * max(norm(L_all[k + 1]), 1.0)
            end

            R = TTutils._right_env_boundary_mpo(psi, W)
            @test norm(R - R_all[N + 2]) <= 1e-12 * max(norm(R_all[N + 2]), 1.0)
            for k in N:-1:1
                R = TTutils._advance_right_env_mpo(R, psi[k], W[k])
                @test norm(R - R_all[k + 1]) <= 1e-12 * max(norm(R_all[k + 1]), 1.0)
            end
        end
    end
end
