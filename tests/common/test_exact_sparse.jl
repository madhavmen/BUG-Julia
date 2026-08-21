# VALIDATE THE SPARSE-KRYLOV REFERENCE ITSELF, AT LARGE |t|.
#
# `benchmarks/exact_sparse.jl` is what `examples/heisenberg_domain_wall.jl` and
# `benchmarks/l18_matrix.jl` measure the integrator AGAINST, so an error in it is reported as an
# error in the method. It had exactly that failure: `expv_sparse` documented substepping and
# implemented a single degree-`m` projection, which is accurate only while `|t| * ||H|| <~ m`.
# Past that it returns a confidently wrong vector -- and because the Krylov space of a symmetric
# initial condition under symmetric `H` stays in the sector and keeps the spatial symmetry, the
# wrong answer is smooth and correctly antisymmetric. It looks like a converged physics result.
#
# The bug was invisible to every existing test because they all drive the reference through
# `propagate!` in small `dt` steps, where `|t| * ||H||` per call is tiny. So the coverage that
# matters is a ONE-SHOT call at large `|t|`, checked against a genuinely independent reference:
# the DENSE matrix exponential. At L=8 the Sz=0 sector is C(8,4)=70, so `exp(-im*t*Matrix(A))` is
# a 70x70 dense exponential -- exact, and sharing no code with the Lanczos path.

using Test, LinearAlgebra, SparseArrays

include(joinpath(@__DIR__, "..", "..", "benchmarks", "exact_sparse.jl"))

@testset "exact_sparse: expv_sparse" begin
    L = 8
    A, states, idx = heisenberg_sparse(L)
    v0 = domain_wall_vector(L, states, idx)
    Ad = Matrix(A)

    @test issymmetric(Ad)
    nrmA = opnorm(Ad, 2)

    # The regime that used to fail. At t = 20, ||H||*t is far past the default m = 30, so a
    # single projection cannot represent exp(-iHt) no matter how well-conditioned the sector is.
    @testset "one-shot call at |t|*||H|| >> m  (t = $t)" for t in (0.5, 2.0, 20.0, 50.0)
        @test nrmA * t > 0
        exact = exp(-im * t * Ad) * v0            # independent: dense, no Lanczos
        got   = expv_sparse(A, ComplexF64(-im * t), v0)

        @test norm(got - exact) < 1e-9
        # the observable the benchmarks actually report
        @test maximum(abs.(magnetisation_dense(got, L, states) .-
                           magnetisation_dense(exact, L, states))) < 1e-10
        @test abs(norm(got) - 1) < 1e-10          # real time is unitary
    end

    # Substepping must not have broken the small-|t| path the benchmarks rely on, and one
    # one-shot call must agree with the same interval walked in many steps.
    @testset "one-shot == substepped by propagate!" begin
        t, n = 20.0, 400
        walked = propagate!(A, v0, ComplexF64(-im * t / n), n, (s, v) -> nothing)
        @test norm(expv_sparse(A, ComplexF64(-im * t), v0) - walked) < 1e-9
    end

    # Imaginary time: `t` real negative. Decays rather than rotates, so compare the normalised
    # vector -- and it must land on the ground state for large beta.
    @testset "imaginary time" begin
        beta = 20.0
        got   = expv_sparse(A, ComplexF64(-beta), v0)
        exact = exp(-beta * Ad) * v0
        @test norm(got / norm(got) - exact / norm(exact)) < 1e-9

        e0, _ = ground_energy_sparse(A)
        @test abs(real(dot(got, A * got)) / real(dot(got, got)) - e0) < 1e-6
    end

    # An invariant subspace must still be handled exactly (the early-break path), and the
    # degenerate arguments must not loop forever.
    @testset "edge cases" begin
        @test expv_sparse(A, ComplexF64(0), v0) ≈ v0
        @test norm(expv_sparse(A, ComplexF64(-im), zeros(ComplexF64, length(states)))) == 0
        # a single eigenvector: Krylov closes at j = 1
        _, gs = ground_energy_sparse(A)
        g = ComplexF64.(gs)
        @test norm(expv_sparse(A, ComplexF64(-im * 7.0), g) -
                   exp(-im * 7.0 * Ad) * g) < 1e-8
    end

    # A tiny Krylov space must still be correct -- it just needs more substeps. This is the
    # property the old code lacked: accuracy set by `m` instead of by `tol`.
    @testset "accuracy is independent of m" begin
        exact = exp(-im * 20.0 * Ad) * v0
        for m in (4, 8, 30)
            @test norm(expv_sparse(A, ComplexF64(-im * 20.0), v0; m = m) - exact) < 1e-8
        end
    end

    # ── the bridge to `dense_state` ───────────────────────────────────────────────────────────
    #
    # THIS FILE AND `dense_reference.jl` USE MIRRORED BIT CONVENTIONS (here site 1 is the LEAST
    # significant bit and a set bit is UP; there site 1 is the MOST significant and a set bit is
    # DOWN). `sparse_to_dense_index` is the only sanctioned way across, and comparing without it
    # is silently wrong.
    #
    # ⛔ THE STATES MUST BE ASYMMETRIC. The conventions differ by reversal-composed-with-spin-flip,
    # and Neel, the domain wall and the dimer are all FIXED POINTS of that -- they pass with or
    # without the conversion, which is exactly how the bug survived. Anything tested here that is
    # symmetric proves nothing.
    @testset "sparse_to_dense_index bridges the two bit conventions" begin
        for Lb in (4, 6, 8)
            st, _ = sz0_basis(Lb)
            # a bijection onto the same sector, so nothing is dropped or doubled
            mapped = [sparse_to_dense_index(b, Lb) for b in st]
            @test length(unique(mapped)) == length(st)
            @test sort(mapped) == sort(st)      # Sz=0 sector maps onto itself
        end
        # the worked example from the docstring, and its mirror-symmetric neighbours
        @test sparse_to_dense_index(0b001101, 6) == 0b010011   # up{1,3,4} -> down{2,5,6}
        @test sparse_to_dense_index(0b000111, 6) == 0b000111   # domain wall: a FIXED POINT
        @test sparse_to_dense_index(0b010101, 6) == 0b010101   # Neel: also a FIXED POINT
        # an all-up pattern has no down sites, so the dense index is 0
        @test sparse_to_dense_index(0b1111, 4) == 0
        @test sparse_to_dense_index(0b0000, 4) == 0b1111
    end
end
