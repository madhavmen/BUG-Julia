# tests/Benchmarks/test_hamiltonians.jl
#
# Smoke tests for benchmark Hamiltonian MPO construction and initial states.
# Checks MPO correctness: exact ground-state energy for solvable models.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils

# ── XY model: H = Σ (S+_i S-_{i+1} + S-_i S+_{i+1}) ─────────────────────────

function make_xy_mpo(sites; Jxy = 1.0)
    N  = length(sites)
    os = OpSum()
    for i in 1:(N-1)
        os += Jxy * 0.5, "S+", i, "S-", i+1
        os += Jxy * 0.5, "S-", i, "S+", i+1
    end
    return TensorTrainOperator(MPO(os, sites))
end

# ── Transverse-field Ising: H = -J Σ ZZ - h Σ X ─────────────────────────────

function make_tfim_mpo(sites; J = 1.0, h = 0.5)
    N  = length(sites)
    os = OpSum()
    for i in 1:(N-1)
        os += -J, "Sz", i, "Sz", i+1
    end
    for i in 1:N
        os += -h, "Sx", i
    end
    return TensorTrainOperator(MPO(os, sites))
end

# ── Heisenberg: H = Σ (0.5 S+S- + 0.5 S-S+ + Sz Sz) ─────────────────────────

function make_heisenberg_mpo(sites; J = 1.0)
    N  = length(sites)
    os = OpSum()
    for i in 1:(N-1)
        os += J * 0.5, "S+", i, "S-", i+1
        os += J * 0.5, "S-", i, "S+", i+1
        os += J,       "Sz", i, "Sz", i+1
    end
    return TensorTrainOperator(MPO(os, sites))
end

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "XY MPO construction" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    W     = make_xy_mpo(sites)
    @test length(W) == N
    @test all(k -> hasind(W[k], sites[k]), 1:N)
    @test all(k -> hasind(W[k], sites[k]'), 1:N)
end

@testset "XY MPO: <↑↓↑↓|H|↑↓↑↓> = 0" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    W     = make_xy_mpo(sites)
    psi   = TensorTrain(MPS(sites, ["Up","Dn","Up","Dn"]))
    Hpsi  = contract(W, psi)
    energy = real(dot(psi, Hpsi))
    @test abs(energy) < 1e-12  # Néel state has zero expectation in XY
end

@testset "TFIM MPO construction" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    W     = make_tfim_mpo(sites)
    @test length(W) == N
end

@testset "Heisenberg MPO construction" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    W     = make_heisenberg_mpo(sites)
    @test length(W) == N
end

@testset "Heisenberg: energy of singlet pair N=2" begin
    # For N=2, H = S1·S2 = (S+S- + S-S+)/2 + SzSz
    # Ground state energy = -3/4
    N     = 2
    sites = siteinds("S=1/2", N)
    W     = make_heisenberg_mpo(sites)
    # Singlet = (|↑↓⟩ - |↓↑⟩) / sqrt(2)
    up = ComplexF64[1, 0];  dn = ComplexF64[0, 1]
    sing = (kron(up, dn) - kron(dn, up)) / sqrt(2)
    psi_sing = MPS(sites; linkdims = [2])
    # Build manually
    lnk = Index(2, "Link,l=1")
    psi_sing[1] = itensor([up[1] up[2]; -dn[1] -dn[2]] ./ sqrt(2), sites[1], lnk)
    psi_sing[2] = itensor([dn[1] dn[2]; up[1] up[2]], lnk, sites[2])
    psi = TensorTrain(psi_sing)
    Hpsi = contract(W, psi)
    energy = real(dot(psi, Hpsi)) / real(dot(psi, psi))
    @test abs(energy - (-0.75)) < 1e-12
end

@testset "MPO has correct bond structure" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    W     = make_heisenberg_mpo(sites)
    # All internal MPO tensors should have 4 indices: site, site', left-MPO, right-MPO
    for k in 2:(N-1)
        @test length(inds(W[k])) == 4
    end
    # Boundary tensors have 3 indices
    @test length(inds(W[1])) == 3
    @test length(inds(W[N])) == 3
end
