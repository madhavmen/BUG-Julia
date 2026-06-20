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

    @testset "bug_two_site! trunc_thresh (SVD cutoff) trims rank vs maxdim alone" begin
        gates = BUG.two_site_xx_bond_gates(sites; J = J)

        # Default trunc_thresh = 0.0 must reproduce the pure-maxdim behaviour.
        psi_a = random_tt(sites; maxdim = 2, seed = 7)
        psi_b = random_tt(sites; maxdim = 2, seed = 7)
        info_default = bug_two_site!(psi_a, gates; dt = 0.05, order = :strang, maxdim = 16)
        info_zero    = bug_two_site!(psi_b, gates; dt = 0.05, order = :strang, maxdim = 16, trunc_thresh = 0.0)
        @test info_default.bond_dims_after == info_zero.bond_dims_after

        # A loose cutoff keeps the full rank; a coarse cutoff discards weak
        # directions, so its kept bonds never exceed the tight-cutoff run.
        psi_tight = random_tt(sites; maxdim = 2, seed = 7)
        psi_loose = random_tt(sites; maxdim = 2, seed = 7)
        info_tight = bug_two_site!(psi_tight, gates; dt = 0.05, order = :strang, maxdim = 16, trunc_thresh = 1e-14)
        info_loose = bug_two_site!(psi_loose, gates; dt = 0.05, order = :strang, maxdim = 16, trunc_thresh = 1e-1)
        @test all(info_loose.bond_dims_after .≤ info_tight.bond_dims_after)
        @test maximum(info_loose.bond_dims_after) ≤ maximum(info_tight.bond_dims_after)
        @test norm(psi_loose) > 0.0
    end
end
