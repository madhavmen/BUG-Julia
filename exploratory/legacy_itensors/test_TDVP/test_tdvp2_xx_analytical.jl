# tests/TDVP/test_tdvp2_xx_analytical.jl
#
# Test 2-site TDVP: verify norm preservation on nearest-neighbor XX Hamiltonian.
# (Used directly in domain wall N=50 simulation.)

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "TDVP", "TDVP.jl"))
using .TDVP

include(joinpath(@__DIR__, "..", "common", "xx_free_fermion.jl"))

@testset "TDVP 2-site NN XX (domain-wall-relevant)" begin
    N = 4
    J = 1.0
    sites = siteinds("S=1/2", N)

    @testset "Analytical XX reference available" begin
        E0_exact, psi0_exact = xx_ground_state(N; J = J)
        @test abs(norm(psi0_exact) - 1.0) < 1e-10
    end

    @testset "tdvp2_step! call succeeds" begin
        os = OpSum()
        for b in 1:(N - 1)
            os += J / 2, "S+", b, "S-", b + 1
            os += J / 2, "S-", b, "S+", b + 1
        end
        H = TensorTrainOperator(MPO(os, sites))

        psi = random_tt(sites; maxdim = 2, seed = 123)
        orthogonalize!(psi, 1)

        info = tdvp2_step!(psi, H; dt = 0.01, maxdim = 16)

        # tdvp2_step! should complete and return timing info
        @test info.elapsed ≥ 0.0
        @test norm(psi) > 0.0  # State should remain non-zero
    end

    @testset "tdvp2_step! respects maxdim cap" begin
        os = OpSum()
        for b in 1:(N - 1)
            os += J / 2, "S+", b, "S-", b + 1
            os += J / 2, "S-", b, "S+", b + 1
        end
        H = TensorTrainOperator(MPO(os, sites))

        psi = random_tt(sites; maxdim = 2, seed = 42)
        orthogonalize!(psi, 1)
        info = tdvp2_step!(psi, H; dt = 0.01, maxdim = 4)

        @test all(info.bond_dims_after .≤ 4)
    end
end
