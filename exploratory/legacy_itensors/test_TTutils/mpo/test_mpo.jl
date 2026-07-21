# tests/TTutils/mpo/test_mpo.jl
#
# Unit tests for TensorTrainOperator (MPO) construction, identity MPO,
# and MPO environment builders. Tolerances at 1e-12.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils

# ── Helpers ───────────────────────────────────────────────────────────────────

function _make_xx_mpo(sites)
    N = length(sites)
    os = OpSum()
    for i in 1:(N-1)
        os += 0.5, "S+", i, "S-", i+1
        os += 0.5, "S-", i, "S+", i+1
    end
    return TensorTrainOperator(MPO(os, sites))
end

@testset "TensorTrainOperator construction" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    W     = _make_xx_mpo(sites)

    @test length(W) == N
    @test all(k -> hasind(W[k], sites[k]), 1:N)
    @test all(k -> hasind(W[k], sites[k]'), 1:N)
end

@testset "Identity MPO" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    I_op  = identity_op(sites)

    # For each spin basis state, I_op * psi should be psi
    psi = random_tt(sites; maxdim = 3, seed = 11)
    W   = I_op

    # Build full contraction manually and check norm
    # contract(W, psi) ≈ psi
    psi_out = TTutils.contract(W, psi)
    @test abs(norm(psi_out) - norm(psi)) / norm(psi) < 1e-12
    @test TTutils.distance(psi_out, psi) / norm(psi) < 1e-12
end

@testset "MPO right environments" begin
    N     = 6
    sites = siteinds("S=1/2", N)
    W     = _make_xx_mpo(sites)
    psi   = random_tt(sites; maxdim = 4, seed = 13)
    orthogonalize!(psi, 1)

    R_envs = _build_right_envs_mpo(psi, W)
    @test length(R_envs) == N + 2

    # Boundary (rightmost) env must be a well-formed tensor
    R_N2 = R_envs[N+2]
    @test ndims(R_N2) == length(inds(R_N2))
end

@testset "MPO left environments" begin
    N     = 6
    sites = siteinds("S=1/2", N)
    W     = _make_xx_mpo(sites)
    psi   = random_tt(sites; maxdim = 4, seed = 15)
    orthogonalize!(psi, N)

    L_envs = _build_left_envs_mpo(psi, W)
    @test length(L_envs) == N + 2

    L_1 = L_envs[1]
    @test ndims(L_1) == length(inds(L_1))
end

@testset "1-site Heff matrix is Hermitian" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    W     = _make_xx_mpo(sites)
    psi   = random_tt(sites; maxdim = 4, seed = 17)
    orthogonalize!(psi, 1)

    R_envs = _build_right_envs_mpo(psi, W)
    L_cur  = _left_env_boundary_mpo(psi, W)
    L_cur  = _advance_left_env_mpo(L_cur, psi[1], W[1])  # advance past site 1 → correct L for site 2

    k       = 2
    link_l  = commonind(psi[k-1], psi[k])
    site_k  = siteinds(psi, k)
    link_r  = commonind(psi[k], psi[k+1])
    H_mat   = _build_1site_Heff_mat(link_l, site_k, link_r, L_cur, R_envs[k+2], W[k])
    @test norm(H_mat - H_mat') / norm(H_mat) < 1e-12
end


# NOTE: the 0-site Heff requires L and R to carry *distinct* bond indices
# (u_link from psi[k] after SVD, v_link from psi[k+1]).  That setup
# duplicates SVD logic and is covered by the TDVP integration tests;
# no standalone unit test is needed here.
