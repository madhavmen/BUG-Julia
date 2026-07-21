using Test
using ITensors
using ITensorMPS
using LinearAlgebra

isdefined(@__MODULE__, :QuantumBenchmarkCommon) || include(joinpath(@__DIR__, "..", "..", "benchmarks", "common", "quantum_benchmark_common.jl"))
using .QuantumBenchmarkCommon

const _QB_ROOT = joinpath(@__DIR__, "..", "..")

function _exact_traj(H::AbstractMatrix, psi0::AbstractVector, dt::Float64, nsteps::Int)
    return dense_real_time_trajectory(H, psi0, dt, nsteps)
end

function _l2_profile_error(a::AbstractVector, b::AbstractVector)
    return norm(Float64.(a) .- Float64.(b)) / sqrt(length(a))
end

function _run_bug_profiles(psi0::TensorTrain, W::TensorTrainOperator, dt::Float64, nsteps::Int, measure_profile)
    psi = deepcopy(psi0)
    profiles = Vector{Vector{Float64}}(undef, nsteps + 1)
    profiles[1] = measure_profile(psi, 0.0)
    for step in 1:nsteps
        bug_step!(
            psi,
            W;
            dt = -im * dt,
            maxdim = 64,
            cutoff = 0.0,
            lanczos_tol = 1e-12,
            lanczos_maxiter = 40,
            substep_method = :expv,
        )
        profiles[step + 1] = measure_profile(psi, step * dt)
    end
    return profiles
end

function _run_tdvp_profiles(psi0::TensorTrain, W::TensorTrainOperator, dt::Float64, nsteps::Int, measure_profile)
    psi = deepcopy(psi0)
    profiles = Vector{Vector{Float64}}(undef, nsteps + 1)
    profiles[1] = measure_profile(psi, 0.0)
    for step in 1:nsteps
        tdvp2_step!(
            psi,
            W;
            dt = -im * dt,
            maxdim = 64,
            cutoff = 0.0,
            lanczos_tol = 1e-12,
            lanczos_maxiter = 40,
            substep_method = :expv,
            step_mode = :symmetric_fr,
        )
        QuantumBenchmarkCommon.svd_compress!(psi; maxdim = 64, cutoff = 0.0)
        profiles[step + 1] = measure_profile(psi, step * dt)
    end
    return profiles
end

function _xx_mpo(sites)
    os = OpSum()
    for i in 1:(length(sites) - 1)
        os += 0.5, "S+", i, "S-", i + 1
        os += 0.5, "S-", i, "S+", i + 1
    end
    return TensorTrainOperator(MPO(os, sites))
end

function _long_xy_mpo(sites, alpha::Float64)
    os = OpSum()
    for i in 1:(length(sites) - 1), j in (i + 1):length(sites)
        coeff = 1.0 / abs(i - j)^alpha
        os += coeff, "S+", i, "S-", j
        os += coeff, "S-", i, "S+", j
    end
    return TensorTrainOperator(MPO(os, sites))
end

function _tfim_mpo(sites; J::Float64 = 1.0, h::Float64 = 0.5)
    os = OpSum()
    for i in 1:(length(sites) - 1)
        os += -4 * J, "Sz", i, "Sz", i + 1
    end
    for i in 1:length(sites)
        os += -2 * h, "Sx", i
    end
    return TensorTrainOperator(MPO(os, sites))
end

function _heis_mpo(sites)
    os = OpSum()
    for i in 1:(length(sites) - 1)
        os += 0.5, "S+", i, "S-", i + 1
        os += 0.5, "S-", i, "S+", i + 1
        os += 1.0, "Sz", i, "Sz", i + 1
    end
    return TensorTrainOperator(MPO(os, sites))
end

function _hs_mpo(sites)
    os = OpSum()
    for i in 1:length(sites), j in (i + 1):length(sites)
        coeff = 1.0 / (j - i)^2
        os += coeff * 0.5, "S+", i, "S-", j
        os += coeff * 0.5, "S-", i, "S+", j
        os += coeff, "Sz", i, "Sz", j
    end
    return TensorTrainOperator(MPO(os, sites))
end

@testset "Short-ranged XX domain-wall smoke benchmark" begin
    N = 6
    dt = 0.05
    nsteps = 3
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)
    psi0 = TensorTrain(MPS(sites, [i <= N ÷ 2 ? "Up" : "Dn" for i in 1:N]))

    H = dense_operator_matrix(W)
    psi0_vec = ComplexF64.(TTutils.vector(psi0))
    sz_ops = [dense_local_operator(sites, "Sz", i) for i in 1:N]
    exact = _exact_traj(H, psi0_vec, dt, nsteps)
    exact_profiles = [
        [real(ψ' * (sz_ops[i] * ψ)) / real(ψ' * ψ) for i in 1:N]
        for ψ in exact
    ]
    measure = (psi, _t) -> measure_local_profile(psi, "Sz")

    bug_profiles = _run_bug_profiles(psi0, W, dt, nsteps, measure)
    tdvp_profiles = _run_tdvp_profiles(psi0, W, dt, nsteps, measure)

    @test _l2_profile_error(bug_profiles[end], exact_profiles[end]) < 0.1
    @test _l2_profile_error(tdvp_profiles[end], exact_profiles[end]) < 0.1
end

@testset "Long-ranged XY local-rotation smoke benchmark" begin
    N = 6
    dt = 0.02
    nsteps = 2
    alpha = 1.5
    sites = siteinds("S=1/2", N)
    W = _long_xy_mpo(sites, alpha)
    exact = dense_ground_state(W)
    center = cld(N, 2)
    U = cos(π / 4) * Matrix{ComplexF64}(I, 2^N, 2^N) + 2im * sin(π / 4) * dense_local_operator(sites, "Sy", center)
    psi0_vec = U * exact.state
    psi0 = TensorTrain(psi0_vec, sites; maxdim = 64, cutoff = 1e-12)
    sx_ops = [dense_local_operator(sites, "Sx", i; scale = 2.0) for i in 1:N]
    baseline = [
        real(exact.state' * (sx_ops[i] * exact.state)) / real(exact.state' * exact.state)
        for i in 1:N
    ]
    exact_traj = _exact_traj(exact.hamiltonian, psi0_vec, dt, nsteps)
    exact_profiles = [
        abs.([
            real(ψ' * (sx_ops[i] * ψ)) / real(ψ' * ψ) - baseline[i]
            for i in 1:N
        ])
        for ψ in exact_traj
    ]
    measure = (psi, _t) -> abs.(2.0 .* measure_local_profile(psi, "Sx") .- baseline)

    bug_profiles = _run_bug_profiles(psi0, W, dt, nsteps, measure)
    tdvp_profiles = _run_tdvp_profiles(psi0, W, dt, nsteps, measure)

    @test _l2_profile_error(bug_profiles[end], exact_profiles[end]) < 0.1
    @test _l2_profile_error(tdvp_profiles[end], exact_profiles[end]) < 0.1
end

@testset "TFIM local-Sz smoke benchmark" begin
    N = 6
    dt = 0.05
    nsteps = 2
    sites = siteinds("S=1/2", N)
    W = _tfim_mpo(sites; J = 1.0, h = 0.5)
    exact = dense_ground_state(W)
    center = cld(N, 2)
    phi0 = dense_local_operator(sites, "Sz", center) * exact.state
    phi0 ./= norm(phi0)
    psi0 = TensorTrain(phi0, sites; maxdim = 64, cutoff = 1e-12)
    sz_ops = [dense_local_operator(sites, "Sz", i; scale = 2.0) for i in 1:N]
    baseline = [
        real(exact.state' * (sz_ops[i] * exact.state)) / real(exact.state' * exact.state)
        for i in 1:N
    ]
    exact_traj = _exact_traj(exact.hamiltonian, phi0, dt, nsteps)
    exact_profiles = [
        [
            real(ψ' * (sz_ops[i] * ψ)) / real(ψ' * ψ) - baseline[i]
            for i in 1:N
        ]
        for ψ in exact_traj
    ]
    measure = (psi, _t) -> 2.0 .* measure_local_profile(psi, "Sz") .- baseline

    bug_profiles = _run_bug_profiles(psi0, W, dt, nsteps, measure)
    tdvp_profiles = _run_tdvp_profiles(psi0, W, dt, nsteps, measure)

    @test _l2_profile_error(bug_profiles[end], exact_profiles[end]) < 0.15
    @test _l2_profile_error(tdvp_profiles[end], exact_profiles[end]) < 0.15
end

@testset "Heisenberg local-Sz smoke benchmark" begin
    N = 6
    dt = 0.05
    nsteps = 2
    sites = siteinds("S=1/2", N)
    W = _heis_mpo(sites)
    exact = dense_ground_state(W)
    center = cld(N, 2)
    phi0 = dense_local_operator(sites, "Sz", center) * exact.state
    phi0 ./= norm(phi0)
    psi0 = TensorTrain(phi0, sites; maxdim = 64, cutoff = 1e-12)
    sz_ops = [dense_local_operator(sites, "Sz", i) for i in 1:N]
    baseline = [
        real(exact.state' * (sz_ops[i] * exact.state)) / real(exact.state' * exact.state)
        for i in 1:N
    ]
    exact_traj = _exact_traj(exact.hamiltonian, phi0, dt, nsteps)
    exact_profiles = [
        [
            real(ψ' * (sz_ops[i] * ψ)) / real(ψ' * ψ) - baseline[i]
            for i in 1:N
        ]
        for ψ in exact_traj
    ]
    measure = (psi, _t) -> measure_local_profile(psi, "Sz") .- baseline

    bug_profiles = _run_bug_profiles(psi0, W, dt, nsteps, measure)
    tdvp_profiles = _run_tdvp_profiles(psi0, W, dt, nsteps, measure)

    @test _l2_profile_error(bug_profiles[end], exact_profiles[end]) < 0.15
    @test _l2_profile_error(tdvp_profiles[end], exact_profiles[end]) < 0.15
end

@testset "Haldane-Shastry local-Sz smoke benchmark" begin
    N = 6
    dt = 0.025
    nsteps = 2
    sites = siteinds("S=1/2", N)
    W = _hs_mpo(sites)
    exact = dense_ground_state(W)
    center = cld(N, 2)
    phi0 = dense_local_operator(sites, "Sz", center) * exact.state
    psi0 = TensorTrain(phi0, sites; maxdim = 64, cutoff = 1e-12)
    sz_ops = [dense_local_operator(sites, "Sz", i) for i in 1:N]
    exact_traj = _exact_traj(exact.hamiltonian, phi0, dt, nsteps)
    exact_profiles = Vector{Vector{Float64}}(undef, nsteps + 1)
    for step in 0:nsteps
        ψ = exact_traj[step + 1]
        exact_profiles[step + 1] = [
            real(exp(im * exact.energy * step * dt) * (exact.state' * (sz_ops[i] * ψ)))
            for i in 1:N
        ]
    end
    measure = (psi, t) -> begin
        ψvec = ComplexF64.(TTutils.vector(psi))
        [
            real(exp(im * exact.energy * t) * (exact.state' * (sz_ops[i] * ψvec)))
            for i in 1:N
        ]
    end

    bug_profiles = _run_bug_profiles(psi0, W, dt, nsteps, measure)
    tdvp_profiles = _run_tdvp_profiles(psi0, W, dt, nsteps, measure)

    @test _l2_profile_error(bug_profiles[end], exact_profiles[end]) < 0.15
    @test _l2_profile_error(tdvp_profiles[end], exact_profiles[end]) < 0.15
end
