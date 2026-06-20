# tests/BUG/test_bug_init.jl
#
# Unit tests for BUGInfo struct, composition constants, and bond snapshot helpers.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "BUG", "BUG.jl"))
using .BUG

@testset "BUGInfo defaults" begin
    info = BUGInfo()
    @test isempty(info.bond_dims_before)
    @test isempty(info.bond_dims_after)
    @test info.elapsed == 0.0
    @test isempty(info.aug_sizes_k)
    @test isempty(info.aug_sizes_l)
    @test isempty(info.lanczos_numops)
end

@testset "BUGInfo record helpers" begin
    info = BUGInfo()
    BUG._record_s_step_rank!(info, :fwd, 2, 3)
    @test info.s_step_kept_ranks == [3]
    @test info.s_step_bonds == [2]
end

@testset "_resolve_quantum_step_mode" begin
    mode = :second_order_frf
    cfg = BUG._resolve_quantum_step_mode(mode)
    @test cfg.step_mode === mode
    @test length(cfg.sweep_schedule) >= 2
    @test_throws ErrorException BUG._resolve_quantum_step_mode(:nonexistent)
end

@testset "Canonical bond snapshot" begin
    N     = 5
    sites = siteinds("S=1/2", N)

    function _make_xx_mpo(sites)
        os = OpSum()
        for i in 1:(length(sites)-1)
            os += 0.5, "S+", i, "S-", i+1
            os += 0.5, "S-", i, "S+", i+1
        end
        return TensorTrainOperator(MPO(os, sites))
    end

    psi = random_tt(sites; maxdim = 3, seed = 31)
    W   = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)

    bond = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)

    @testset "snapshot fields present" begin
        @test hasfield(typeof(snap), :U0_tens)
        @test hasfield(typeof(snap), :V0_tens)
        @test hasfield(typeof(snap), :S0_tens)
        @test hasfield(typeof(snap), :canon_u0)
        @test hasfield(typeof(snap), :canon_v0)
    end

    @testset "U0 is left-isometric" begin
        U0  = snap.U0_tens
        mid = snap.canon_u0
        # Flatten (link_l, site_l) → rows; mid → cols; check Q'Q = I
        U0_mat = reshape(ComplexF64.(Array(U0, snap.link_l, snap.site_l, mid)),
                         dim(snap.link_l)*dim(snap.site_l), dim(mid))
        @test norm(U0_mat' * U0_mat - I) < 1e-10
    end

    @testset "V0 is right-isometric" begin
        V0  = snap.V0_tens
        mid = snap.canon_v0
        # Flatten mid → rows; (site_r, link_r) → cols; check QQ' = I
        V0_mat = reshape(ComplexF64.(Array(V0, mid, snap.site_r, snap.link_r)),
                         dim(mid), dim(snap.site_r)*dim(snap.link_r))
        @test norm(V0_mat * V0_mat' - I) < 1e-10
    end

    @testset "theta0 reconstructed from U0*S0*V0" begin
        theta0 = snap.U0_tens * snap.S0_tens * snap.V0_tens
        psi_two = psi[bond] * psi[bond+1]
        @test norm(theta0 - psi_two) / norm(psi_two) < 1e-12
    end
end
