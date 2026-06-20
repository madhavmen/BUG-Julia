# tests/TDVP/test_tdvp_integration.jl
#
# Integration tests: 5-step TDVP evolution on a small XX chain with exact
# reference (Jordan-Wigner solvable).

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "TDVP", "TDVP.jl"))
using .TDVP

function _xx_hamiltonian_matrix(N::Int)
    d   = 2^N
    sp  = ComplexF64[0.0 1.0; 0.0 0.0]
    sm  = ComplexF64[0.0 0.0; 1.0 0.0]
    id2 = Matrix{ComplexF64}(I, 2, 2)
    H   = zeros(ComplexF64, d, d)
    for i in 1:(N-1)
        op_p = reduce(kron, [k == i ? sp : k == i+1 ? sm : id2 for k in 1:N])
        H  .+= 0.5 .* op_p
        H  .+= 0.5 .* op_p'
    end
    return H
end

function _exact_evolve(psi0_vec::Vector, H::Matrix, t::Real)
    F  = eigen(Hermitian(H))
    c  = F.vectors' * psi0_vec
    return F.vectors * (exp.(-im .* F.values .* t) .* c)
end

@testset "TDVP integration: XX chain N=4, 5 steps" begin
    N      = 4
    dt     = 0.02
    nsteps = 5
    sites  = siteinds("S=1/2", N)

    os = OpSum()
    for i in 1:(N-1)
        os += 0.5, "S+", i, "S-", i+1
        os += 0.5, "S-", i, "S+", i+1
    end
    W = TensorTrainOperator(MPO(os, sites))

    # Néel initial state |↑↓↑↓⟩
    psi_mps = MPS(sites, ["Up", "Dn", "Up", "Dn"])
    psi     = TensorTrain(psi_mps)

    # Exact reference in the same dense basis/order used by TTutils.
    H_mat = ComplexF64.(TTutils.matrix(W))
    psi0_vec = ComplexF64.(TTutils.vector(psi))
    psi_exact = copy(psi0_vec)
    for _ in 1:nsteps
        psi_exact = _exact_evolve(psi_exact, H_mat, dt)
    end

    # TDVP evolution
    for _ in 1:nsteps
        tdvp_step!(psi, W; dt = -im * dt, step_mode = :symmetric_fr,
            substep_method = :expv, lanczos_tol = 1e-13, lanczos_maxiter = 30,
        )
    end

    psi_tdvp_vec = ComplexF64.(TTutils.vector(psi))
    rel_err = norm(psi_tdvp_vec - psi_exact) / norm(psi_exact)
    fidelity = abs(dot(psi_tdvp_vec, psi_exact)) / (norm(psi_tdvp_vec) * norm(psi_exact))
    @test fidelity > 0.995
    @test rel_err < 0.1
end

@testset "TDVP step returns TDVPInfo" begin
    N     = 4
    sites = siteinds("S=1/2", N)
    os    = OpSum()
    for i in 1:(N-1); os += 0.5, "S+", i, "S-", i+1; os += 0.5, "S-", i, "S+", i+1; end
    W   = TensorTrainOperator(MPO(os, sites))
    psi = TensorTrain(MPS(sites, ["Up", "Dn", "Up", "Dn"]))
    info = tdvp_step!(psi, W; dt = -0.01im)
    @test isa(info, TDVPInfo)
    @test info.site_order == 1
    @test length(info.bond_dims_before) == N - 1
    @test !isempty(info.site_numops)
end
