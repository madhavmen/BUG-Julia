# test_two_site_local_kls.jl
#
# The BUG KLS local update is applied on each odd and each even bond, with NO
# backward correction. At full rank (with enough augmentation depth to saturate
# the local frames) each update is the exact local evolution, so:
#   * a single parity sweep reproduces exp(-i ¤ä H_par) to machine precision,
#   * one full Strang step reproduces the exact Strang product
#         exp(-i dt/2 H_odd) exp(-i dt H_even) exp(-i dt/2 H_odd),
#   * `backward_correction_calls` is always 0.

using Test, ITensors, ITensorMPS, LinearAlgebra, Random
include(joinpath(@__DIR__, "two_site_test_setup.jl"))

@testset "two-site local KLS update (N=6 XX, full rank)" begin
    N = 6
    sites = siteinds("S=1/2", N)
    J = 1.0
    gates = two_site_xx_bond_gates(sites; J = J)
    W_odd, W_even, _ = two_site_xx_parity_mpos(sites; J = J)
    Hodd  = two_site_dense(W_odd)
    Heven = two_site_dense(W_even)
    dt = 0.05

    @testset "a parity sweep reproduces exp(-i ¤ä H_par)" begin
        for (par, Hpar) in ((:odd, Hodd), (:even, Heven))
            psi = two_site_rank3_state(sites; seed = 7)
            v0 = two_site_vec(psi)
            BUG._with_bug_expv_backend(:native_hermitian_lanczos) do
                BUG._with_bug_time_prefactor(ComplexF64(-im)) do
                    BUG._two_site_parity_sweep!(psi, gates, dt, BUGInfo(); parity = par,
                        maxdim = typemax(Int), augment = true, aug_krylov_depth = 3,
                        lanczos_tol = 1e-14, lanczos_maxiter = 60,
                        substep_method = :expv, matrixfree_sstep = false)
                end
            end
            v  = two_site_vec(psi)
            ve = two_site_exact(v0, Hpar, dt)
            @test two_site_infidelity(v, ve) <= 1e-11
            @test two_site_norm_error(v, norm(v0)) <= 1e-11
        end
    end

    @testset "one full step == exact Strang product (machine precision)" begin
        psi = two_site_rank3_state(sites; seed = 7)
        v0 = two_site_vec(psi)
        info = bug_two_site!(psi, gates; dt = dt, order = :strang,
            maxdim = typemax(Int), aug_krylov_depth = 3,
            lanczos_tol = 1e-14, lanczos_maxiter = 60)
        v = two_site_vec(psi)
        a  = two_site_exact(v0, Hodd, dt / 2)
        b  = two_site_exact(a,  Heven, dt)
        vs = two_site_exact(b,  Hodd, dt / 2)
        @test two_site_infidelity(v, vs) <= 1e-11
        @test info.backward_correction_calls == 0
    end

    @testset "no backward correction over a multi-step run" begin
        psi = two_site_rank3_state(sites; seed = 7)
        total_backward = 0
        for _ in 1:5
            info = bug_two_site!(psi, gates; dt = dt, order = :strang,
                maxdim = typemax(Int), aug_krylov_depth = 3)
            total_backward += info.backward_correction_calls
        end
        @test total_backward == 0
    end
end
