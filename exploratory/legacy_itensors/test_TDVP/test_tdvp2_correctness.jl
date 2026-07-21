# tests/TDVP/test_tdvp2_correctness.jl
#
# Exact-comparison test for the new 2-site TDVP implementation.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "TDVP", "TDVP.jl"))
using .TDVP

function _tdvp2_xx_mpo(sites)
    os = OpSum()
    for i in 1:(length(sites) - 1)
        os += 0.5, "S+", i, "S-", i + 1
        os += 0.5, "S-", i, "S+", i + 1
    end
    return TensorTrainOperator(MPO(os, sites))
end

function _tdvp2_exact_evolve(psi0_vec::Vector{ComplexF64}, H::Matrix{ComplexF64}, t::Float64)
    F = eigen(Hermitian(H))
    coeffs = F.vectors' * psi0_vec
    return F.vectors * (exp.(-im .* F.values .* t) .* coeffs)
end

@testset "2-site TDVP matches exact XX evolution on N=6" begin
    N = 6
    dt = 0.01
    nsteps = 5
    sites = siteinds("S=1/2", N)
    W = _tdvp2_xx_mpo(sites)
    psi = TensorTrain(MPS(sites, [isodd(i) ? "Up" : "Dn" for i in 1:N]))

    ITensors.disable_warn_order()
    H = try
        ComplexF64.(TTutils.matrix(W))
    finally
        ITensors.reset_warn_order()
    end
    psi_exact = ComplexF64.(TTutils.vector(psi))
    for _ in 1:nsteps
        psi_exact = _tdvp2_exact_evolve(psi_exact, H, dt)
        tdvp2_step!(psi, W;
            dt = -im * dt,
            maxdim = typemax(Int),
            cutoff = 1e-12,
            lanczos_tol = 1e-13,
            lanczos_maxiter = 40,
            step_mode = :symmetric_fr,
        )
    end

    psi_tdvp2 = ComplexF64.(TTutils.vector(psi))
    rel_err = norm(psi_tdvp2 - psi_exact) / norm(psi_exact)
    fidelity = abs(dot(psi_tdvp2, psi_exact)) / (norm(psi_tdvp2) * norm(psi_exact))
    @test fidelity > 1 - 1e-10
    @test rel_err < 1e-8
end
