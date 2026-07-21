# tests/TTutils/mps/test_mps.jl
#
# Unit tests for TensorTrain (MPS) constructors, canonical forms, algebra,
# and index utilities. All tolerances at 1e-12 or better.

using Test
using ITensors
using LinearAlgebra

# Load module from src
const _SRC = joinpath(@__DIR__, "..", "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils

@testset "TensorTrain construction" begin
    N  = 6
    d  = 2
    χ  = 4
    sites = siteinds("S=1/2", N)

    @testset "random_tt shape" begin
        psi = random_tt(sites; maxdim = χ, seed = 1)
        @test length(psi) == N
        @test all(i -> hasind(psi[i], sites[i]), 1:N)
    end

    @testset "uniform_tt" begin
        psi = uniform_tt(sites, 1.0)
        @test length(psi) == N
        # All entries must be finite
        @test all(i -> all(isfinite, ITensors.array(psi[i])), 1:N)
    end
end

@testset "TensorTrain norms and overlaps" begin
    N     = 6
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 4, seed = 2)
    phi   = random_tt(sites; maxdim = 3, seed = 3)

    @testset "norm consistency" begin
        n = norm(psi)
        @test n > 0
        # Norm via overlap
        @test abs(sqrt(real(dot(psi, psi))) - n) < 1e-12
    end

    @testset "distance(psi, psi) == 0" begin
        @test distance(psi, psi) < 1e-12
    end

    @testset "dot bilinearity" begin
        alpha = 1.7 + 0.3im
        psi_s = tt_scaled_copy(psi, alpha)
        @test abs(dot(psi_s, phi) - conj(alpha) * dot(psi, phi)) < 1e-10
    end
end

@testset "Orthogonalization" begin
    N     = 6
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 6, seed = 5)

    @testset "left-canonical: U†U = I at each site" begin
        orthogonalize!(psi, N)
        for k in 1:(N - 1)
            A = psi[k]
            llink = k == 1 ? nothing : commonind(psi[k-1], psi[k])
            site  = sites[k]
            rlink = commonind(psi[k], psi[k+1])
            # Build identity check: A†A ≈ I on the right bond
            # prime(dag(A), rlink) * A gives (rlink', rlink); compare to identity delta
            AA  = prime(dag(A), rlink) * A
            eye = delta(rlink', rlink)
            @test norm(AA - eye) < 1e-12
        end
    end

    @testset "right-canonical: VV† = I at each site" begin
        orthogonalize!(psi, 1)
        for k in 2:N
            B     = psi[k]
            llink = commonind(psi[k-1], psi[k])
            BB    = prime(dag(B), llink) * B
            eye   = delta(llink', llink)
            @test norm(BB - eye) < 1e-12
        end
    end
end

@testset "linkdims and siteinds" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 2, seed = 7)

    @test length(TTutils.linkinds(psi)) == N + 1
    @test all(1:N) do k
        siteinds(psi, k) == sites[k]
    end
end

@testset "replacelinks" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 9)
    phi   = replacelinks(psi)
    # Should have same norm but different link indices
    @test abs(norm(phi) - norm(psi)) < 1e-12
    for k in 1:(N - 1)
        @test TTutils.linkind(phi, k) != TTutils.linkind(psi, k)
    end
end
