# test_two_site_trotter_convergence.jl
#
# At full rank (and enough augmentation depth to make every local KLS update
# exact) the ONLY error of the scheme is the odd/even Trotter splitting. This
# error must shrink as dt ÔåÆ 0, at the order set by the composition:
#   * Lie    (odd(dt) even(dt))                    : global O(dt^1),
#   * Strang (odd(dt/2) even(dt) odd(dt/2))        : global O(dt^2).
# We track the relative L2 error against the exact propagator on the N=6 XX
# Hamiltonian with a rank-3 initial state and confirm the convergence rates.

using Test, ITensors, ITensorMPS, LinearAlgebra, Random
include(joinpath(@__DIR__, "two_site_test_setup.jl"))

function _run_to_T(sites, gates, dt, T; order, depth)
    psi = two_site_rank3_state(sites; seed = 1234)
    v0  = two_site_vec(psi)
    nsteps = round(Int, T / dt)
    for _ in 1:nsteps
        bug_two_site!(psi, gates; dt = dt, order = order,
            maxdim = typemax(Int), aug_krylov_depth = depth,
            lanczos_tol = 1e-14, lanczos_maxiter = 60)
    end
    return psi, v0, nsteps
end

@testset "two-site Trotter-error tracking (N=6 XX, rank-3, full rank)" begin
    N = 6
    sites = siteinds("S=1/2", N)
    gates = two_site_xx_bond_gates(sites; J = 1.0)
    _, _, W_full = two_site_xx_parity_mpos(sites; J = 1.0)
    Hfull = two_site_dense(W_full)
    T = 0.5
    dts = (0.1, 0.05, 0.025)

    relerr(psi, v0) = (v = two_site_vec(psi); ve = two_site_exact(v0, Hfull, T);
                       norm(v - ve) / norm(ve))

    @testset "Strang: global O(dt^2), monotone, norm-preserving" begin
        errs = Float64[]
        for dt in dts
            psi, v0, _ = _run_to_T(sites, gates, dt, T; order = :strang, depth = 3)
            push!(errs, relerr(psi, v0))
            @test two_site_norm_error(two_site_vec(psi), 1.0) <= 1e-10
        end
        @info "Strang Trotter errors" dts errs
        # monotone decrease as dt shrinks
        @test errs[1] > errs[2] > errs[3]
        # halving dt reduces error by Ôëê 4 (second order)
        @test 3.2 <= errs[1] / errs[2] <= 4.8
        @test 3.2 <= errs[2] / errs[3] <= 4.8
    end

    @testset "Lie: global O(dt^1), monotone" begin
        errs = Float64[]
        for dt in dts
            psi, v0, _ = _run_to_T(sites, gates, dt, T; order = :lie, depth = 3)
            push!(errs, relerr(psi, v0))
        end
        @info "Lie Trotter errors" dts errs
        @test errs[1] > errs[2] > errs[3]
        # halving dt reduces error by Ôëê 2 (first order)
        @test 1.7 <= errs[1] / errs[2] <= 2.4
        @test 1.7 <= errs[2] / errs[3] <= 2.4
    end

    @testset "Strang is more accurate than Lie at the same dt" begin
        dt = 0.05
        ps, v0, _ = _run_to_T(sites, gates, dt, T; order = :strang, depth = 3)
        pl, _,  _ = _run_to_T(sites, gates, dt, T; order = :lie,    depth = 3)
        @test relerr(ps, v0) < 0.1 * relerr(pl, v0)
    end
end
