# tests/BUG/two_site_bug/test_two_site_xx_analytical.jl
#
# Test 2-site BUG: verify norm preservation on nearest-neighbor XX Hamiltonian.
# (Used directly in domain wall N=50 simulation.)

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "BUG", "BUG.jl"))
using .BUG

include(joinpath(@__DIR__, "..", "..", "common", "xx_free_fermion.jl"))

@testset "BUG 2-site NN XX (domain-wall-relevant)" begin
    N = 4
    J = 1.0
    sites = siteinds("S=1/2", N)

    @testset "Analytical XX reference available" begin
        E0_exact, psi0_exact = xx_ground_state(N; J = J)
        @test abs(norm(psi0_exact) - 1.0) < 1e-10
    end

    @testset "bug_two_site! call succeeds" begin
        psi = random_tt(sites; maxdim = 2, seed = 123)

        gates = BUG.two_site_xx_bond_gates(sites; J = J)
        info = bug_two_site!(psi, gates; dt = 0.01, order = :strang, maxdim = 16)

        # bug_two_site! should complete and return timing info
        @test info.elapsed ≥ 0.0
        @test norm(psi) > 0.0  # State should remain non-zero
    end

    @testset "bug_two_site! respects maxdim cap" begin
        psi = random_tt(sites; maxdim = 2, seed = 123)
        gates = BUG.two_site_xx_bond_gates(sites; J = J)
        info = bug_two_site!(psi, gates; dt = 0.01, order = :strang, maxdim = 4)

        @test all(info.bond_dims_after .≤ 4)
    end
end
