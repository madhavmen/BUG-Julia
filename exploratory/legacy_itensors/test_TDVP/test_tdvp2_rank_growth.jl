# tests/TDVP/test_tdvp2_rank_growth.jl
#
# Rank-adaptive 2-site TDVP should increase bond dimension from a product
# state under an entangling XX evolution.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "TDVP", "TDVP.jl"))
using .TDVP

function _tdvp2_rank_xx_mpo(sites)
    os = OpSum()
    for i in 1:(length(sites) - 1)
        os += 0.5, "S+", i, "S-", i + 1
        os += 0.5, "S-", i, "S+", i + 1
    end
    return TensorTrainOperator(MPO(os, sites))
end

@testset "2-site TDVP grows bond dimension" begin
    N = 6
    sites = siteinds("S=1/2", N)
    W = _tdvp2_rank_xx_mpo(sites)
    psi = TensorTrain(MPS(sites, [i <= N ÷ 2 ? "Up" : "Dn" for i in 1:N]))
    initial_bonds = [dim(commonind(psi[k], psi[k + 1])) for k in 1:(N - 1)]

    info = nothing
    for _ in 1:4
        info = tdvp2_step!(psi, W;
            dt = -0.025im,
            maxdim = typemax(Int),
            cutoff = 1e-12,
            lanczos_tol = 1e-13,
            lanczos_maxiter = 40,
            step_mode = :symmetric_fr,
        )
    end

    @test info.site_order == 2
    @test maximum(info.bond_dims_after) > maximum(initial_bonds)
    @test isfinite(norm(psi))
    @test norm(psi) > 0
end
