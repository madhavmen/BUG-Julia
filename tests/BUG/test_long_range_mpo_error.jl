using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl")); using .TTutils

function _lrxx_mpo_error(sites; alpha = 1.5)
    os = OpSum()
    N = length(sites)
    for i in 1:(N - 1), j in (i + 1):N
        c = 0.5 / abs(i - j)^alpha
        os += c, "S+", i, "S-", j
        os += c, "S-", i, "S+", j
    end
    return TensorTrainOperator(MPO(os, sites))
end

function _kron_all(ops::Vector{Matrix{ComplexF64}})
    out = ops[1]
    for k in 2:length(ops)
        out = kron(out, ops[k])
    end
    return out
end

function _dense_lrxx_exact(N::Int; alpha = 1.5)
    sp = ComplexF64[0 1; 0 0]
    sm = ComplexF64[0 0; 1 0]
    id = Matrix{ComplexF64}(I, 2, 2)
    H = zeros(ComplexF64, 2^N, 2^N)

    for i in 1:(N - 1), j in (i + 1):N
        c = 0.5 / abs(i - j)^alpha
        ops_pm = [k == i ? sp : k == j ? sm : id for k in 1:N]
        ops_mp = [k == i ? sm : k == j ? sp : id for k in 1:N]
        H .+= c .* _kron_all(ops_pm)
        H .+= c .* _kron_all(ops_mp)
    end

    return H
end

@testset "long-range XX MPO matches direct dense Hamiltonian at N=6" begin
    N = 6
    alpha = 1.5
    sites = siteinds("S=1/2", N)
    W = _lrxx_mpo_error(sites; alpha = alpha)

    ITensors.disable_warn_order()
    H_mpo = ComplexF64.(TTutils.matrix(W))
    ITensors.reset_warn_order()
    H_exact = _dense_lrxx_exact(N; alpha = alpha)
    ΔH = H_mpo - H_exact

    rel_frob = norm(ΔH) / norm(H_exact)
    rel_op = opnorm(ΔH) / opnorm(H_exact)
    max_abs = maximum(abs, ΔH)

    @info "long-range MPO dense error" rel_frob rel_op max_abs
    @test rel_frob < 1e-12
    @test rel_op < 1e-12
    @test max_abs < 1e-12
end
