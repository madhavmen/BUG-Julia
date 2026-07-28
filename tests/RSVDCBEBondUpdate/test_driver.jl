using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# The multi-step driver. It should add bookkeeping and nothing else, so the test that
# matters is that it reproduces a hand-rolled loop threading the SAME rng -- which is also
# what pins the reproducibility contract: one RNG for the whole run, so consecutive steps
# draw different sketches but a rerun draws the same ones.

@testset "cbe_lubich_bug! driver" begin

    @testset "equals a hand-rolled loop with the same rng" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        opts = CBEBugOptions(dt = 0.02, n_steps = 4, maxdim = 16, normalize = false,
                             trunc_thresh = 1e-13, seed = 0xABCD)

        a = domain_wall_state(L)
        cbe_lubich_bug!(a, h; opts = opts)

        b = domain_wall_state(L)
        rng = Random.MersenneTwister(opts.seed)
        for _ in 1:opts.n_steps
            cbe_lubich_sweep(b, h, ComplexF64(-im * opts.dt);
                                maxdim = opts.maxdim, trunc_thresh = opts.trunc_thresh,
                                rng = rng)
        end
        @test norm(dense_state(a) - dense_state(b)) < 1e-12
    end

    @testset "records what it did" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        n = 5
        psi = domain_wall_state(L)
        info = cbe_lubich_bug!(psi, h;
                        opts = CBEBugOptions(dt = 0.02, n_steps = n, maxdim = 8,
                                             record_magnetisation = true,
                                             observe = (p, s, t) -> total_sz(p)))
        @test length(info) == n
        for f in (info.norms, info.bond_dims, info.max_bond_dims, info.expanded,
                  info.max_expanded, info.centre_ranks, info.err_pre, info.err_fnl,
                  info.discarded, info.magnetisations, info.observations)
            @test length(f) == n
        end
        @test info.times ≈ [0.02 * k for k in 1:n]
        @test all(m -> length(m) == L, info.magnetisations)
        @test all(<=(8), info.max_bond_dims)                 # maxdim is respected
        # The expansion happens before the truncation, so the proposed rank is at least
        # the rank that survives -- this is the comparison against `aug_k_dims`.
        @test all(info.max_expanded[k] >= info.max_bond_dims[k] for k in 1:n)
        @test all(x -> abs(x - info.observations[1]) < 1e-10, info.observations)
    end

    @testset "normalisation and charge" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = domain_wall_state(L)
        sz0 = total_sz(copy(psi))
        info = cbe_lubich_bug!(psi, h; opts = CBEBugOptions(dt = 0.02, n_steps = 4, maxdim = 8,
                                                     normalize = true))
        @test abs(norm(psi) - 1.0) < 1e-10
        @test abs(total_sz(copy(psi)) - sz0) < 1e-10
        @test all(n -> n > 0, info.norms)                    # recorded before rescaling
    end

    @testset "reproducible" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        opts = CBEBugOptions(dt = 0.02, n_steps = 3, maxdim = 8, seed = 0x1234)
        a = domain_wall_state(L); cbe_lubich_bug!(a, h; opts = opts)
        b = domain_wall_state(L); cbe_lubich_bug!(b, h; opts = opts)
        @test norm(dense_state(a) - dense_state(b)) == 0.0
    end
end
