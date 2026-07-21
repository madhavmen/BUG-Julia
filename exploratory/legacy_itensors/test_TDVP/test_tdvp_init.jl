# tests/TDVP/test_tdvp_init.jl
#
# Unit tests for TDVPInfo struct, composition constants, QR/LQ gauge moves.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "TDVP", "TDVP.jl"))
using .TDVP

@testset "TDVPInfo defaults" begin
    info = TDVPInfo()
    @test isempty(info.bond_dims_before)
    @test isempty(info.bond_dims_after)
    @test info.elapsed == 0.0
    @test isempty(info.site_numops)
    @test isempty(info.bond_numops)
    @test info.site_order == 1
end

@testset "TDVP composition constants" begin
    @testset "symmetric_fr sums to 1" begin
        @test sum(abs, getindex.(TDVP.TDVP_SYMMETRIC_FR, 2)) ≈ 1.0 atol=1e-14
    end
    @testset "Yoshida coefficients" begin
        @test TDVP._TDVP_Y1 > 0
        @test TDVP._TDVP_Y0 < 0
        # Full Yoshida sum is 2*(Y1+Y1+Y0) and all pairs contribute 1 full step
        full_sum = sum(getindex.(TDVP.TDVP_FOURTH_ORDER_YOSHIDA_FR, 2))
        @test abs(full_sum - 1.0) < 1e-14
    end
end

@testset "_resolve_tdvp_step_mode" begin
    for mode in (:symmetric_fr, :symmetric_rf, :third_order_frf, :third_order_rfr,
                  :fourth_order_yoshida_fr, :fourth_order_yoshida_rf)
        cfg = TDVP._resolve_tdvp_step_mode(mode)
        @test cfg.step_mode === mode
        @test length(cfg.sweep_schedule) >= 2
    end
    @test_throws ErrorException TDVP._resolve_tdvp_step_mode(:invalid_mode)
end

@testset "_tdvp_forward_qr isometry" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 81)
    orthogonalize!(psi, 1)

    k      = 2
    link_l = commonind(psi[k-1], psi[k])
    site_k = siteinds(psi, k)
    link_r = commonind(psi[k], psi[k+1])
    A      = psi[k]
    Q, C, canon_link = TDVP._tdvp_forward_qr(A, link_l, site_k, "Link,l=$k")
    # Q should be left-isometric: Q† Q = I
    Q_mat = reshape(ComplexF64.(Array(Q, link_l, site_k, canon_link)),
                    dim(link_l) * dim(site_k), dim(canon_link))
    @test norm(Q_mat' * Q_mat - I) < 1e-12
end

@testset "_tdvp_reverse_lq isometry" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 83)
    orthogonalize!(psi, N)

    k      = 3
    link_l = commonind(psi[k-1], psi[k])
    site_k = siteinds(psi, k)
    link_r = commonind(psi[k], psi[k+1])
    A      = psi[k]
    C, Q, canon_link = TDVP._tdvp_reverse_lq(A, site_k, link_r, "Link,l=$(k-1)")
    # Q should be right-isometric: Q Q† = I
    Q_mat = reshape(ComplexF64.(Array(Q, canon_link, site_k, link_r)),
                    dim(canon_link), dim(site_k) * dim(link_r))
    @test norm(Q_mat * Q_mat' - I) < 1e-12
end
