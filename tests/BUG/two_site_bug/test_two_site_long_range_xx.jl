using Test, ITensors, ITensorMPS, LinearAlgebra, Random
include(joinpath(@__DIR__, "two_site_test_setup.jl"))

function _run_long_range_to_T(sites, terms, dt, nsteps; depth, maxdim = typemax(Int))
    psi = two_site_rank3_state(sites; seed = 4321)
    v0 = two_site_vec(psi)
    info_last = nothing
    for _ in 1:nsteps
        info_last = bug_two_site!(
            psi,
            terms;
            dt = dt,
            order = :strang,
            maxdim = maxdim,
            cutoff = 0.0,
            aug_krylov_depth = depth,
            lanczos_tol = 1e-13,
            lanczos_maxiter = 40,
        )
    end
    return psi, v0, info_last
end

@testset "two-site split-MPO long-range XX (N=6)" begin
    N = 6
    alpha = 1.5
    dt = 0.01
    nsteps = 5
    sites = siteinds("S=1/2", N)
    terms, W_full = two_site_long_range_xx_matching_mpos(sites; alpha = alpha)

    @testset "matching-layer decomposition matches the full MPO" begin
        H_shell = reduce(+, two_site_dense.(terms))
        H_full = two_site_dense(W_full)
        @test norm(H_shell - H_full) / max(norm(H_full), 1.0) < 1e-12
    end

    @testset "local BUG split-step stays within long-range error bounds" begin
        H_full = two_site_dense(W_full)

        psi_d1, v0, info_d1 = _run_long_range_to_T(sites, terms, dt, nsteps; depth = 1)
        psi_d3, _, info_d3 = _run_long_range_to_T(sites, terms, dt, nsteps; depth = 3)

        v_exact = two_site_exact(v0, H_full, dt * nsteps)
        v_d1 = two_site_vec(psi_d1)
        v_d3 = two_site_vec(psi_d3)

        err_d1 = norm(v_d1 - v_exact) / norm(v_exact)
        err_d3 = norm(v_d3 - v_exact) / norm(v_exact)
        fid_d1 = 1 - two_site_infidelity(v_d1, v_exact)
        fid_d3 = 1 - two_site_infidelity(v_d3, v_exact)

        @info "long-range split-MPO two-site BUG metrics" err_d1 err_d3 fid_d1 fid_d3

        @test info_d1.backward_correction_calls == 0
        @test info_d3.backward_correction_calls == 0
        @test two_site_norm_error(v_d3, 1.0) < 5e-3
        @test err_d3 < 8e-2
        @test fid_d3 > 0.998
        @test err_d3 <= 1.05 * err_d1
    end
end
