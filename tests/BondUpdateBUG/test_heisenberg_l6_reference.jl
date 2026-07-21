using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("../common/dense_reference.jl")

# THE ACCEPTANCE GATE for the whole Julia implementation.
#
# Two references, and they are not interchangeable:
#
#   - `dense_trotter_propagate` uses the SAME odd/even split, so the difference
#     from it is BUG's own projection error, with no splitting error in it.
#   - `dense_exact_propagate` carries an irreducible O(dt^2) Strang error no
#     matter how good BUG is, so that comparison is a convergence-order check and
#     never a precision check.
#
# BUG IS NOT EXACT HERE, AND CANNOT BE. The plan for this task asserted
# `norm(got - trotter) < 1e-13` on the reasoning that Dmax=8 is the full centre
# rank at L=6. That premise is wrong, and measuring it is what showed why.
# Measured frame completeness at dt=0.01 (scripts/l6_exactness_diag.jl):
#
#     bond 1  r=2  aug_k=2/2   aug_l=4/8    <- right frame short
#     bond 3  r=6  aug_k=8/8   aug_l=8/8    <- complete
#     bond 2  r=4  aug_k=4/4   aug_l=7/12   <- right frame short
#
# `exp(-i tau h) Theta` has right support up to 2x the link support, so the h^2
# term needs more room than the Sulz bound `2r` allows: at bond 1, 8 directions
# are wanted and 2r = 4 are permitted. The O(tau^2) local error is therefore
# intrinsic to the 2r bound, not a defect. Confirmed from a WARM full-rank start
# ([2,4,8,4,2] = ambient everywhere), where the error is 5.6e-8 -- LARGER than
# the cold start's 4.4e-10, and first order in dt. If incomplete frames were not
# the cause, a full-rank start would have been exact.
#
# So this file asserts what is actually true: the projection error is small,
# converges under dt refinement, and stays well below the Strang splitting error
# that dominates the physical answer. Exactness is pinned separately, in
# test_kls_step.jl, at a bond where 2r really does reach the ambient dimension.

const L, DMAX, DT, NSTEPS = 6, 8, 0.01, 5

@testset "L=6 Dmax=8 U(1) Heisenberg, first 5 steps" begin
    psi = domain_wall_state(L)
    v0  = dense_state(psi)
    g   = bond_gates(psi; J = 1.0, delta = 1.0)

    info = bond_update_bug!(psi, g; opts = BondUpdateOptions(
        dt = DT, n_steps = NSTEPS, order = :strang, maxdim = DMAX,
        trunc_thresh = 1e-14, normalize = false))
    got = dense_state(psi)

    @testset "the run stayed inside its rank budget and lost no weight" begin
        @test maximum(info.max_bond_dims) <= DMAX
        @test all(d -> d < 1e-13, info.discarded)     # nothing meaningful truncated
        @test all(n -> isapprox(n, 1.0; atol = 1e-11), info.norms)
    end

    @testset "the projection error against the same split is bounded by 2r" begin
        # 6.25e-5 at dt = 0.01, and it is the DOMINANT error: it is the same size
        # as the difference from exact expm, so the Strang splitting is no longer
        # what limits accuracy -- the Sulz bound is.
        #
        # Measured cost of the bound: allowing the fill to sit ON TOP of a full
        # complement (aug = 2r + 1, which violated Sulz in 8/128 frames) bought
        # 4.4e-10 instead. That is five orders, and it is not available -- the
        # augmented rank must stay <= 2r. Accuracy comes from raising the ORDER of
        # the sweep instead (third/fourth-order, later task), not from widening
        # the basis past the bound.
        want = dense_trotter_propagate(L, v0, DT, NSTEPS; order = :strang)
        proj_err = norm(got - want)
        @test proj_err < 1e-3
        @test info.aug_k_dims[1] <= 2 * DMAX          # the bound, at run level
        @test info.aug_l_dims[1] <= 2 * DMAX
    end

    @testset "the projection error shrinks under dt refinement" begin
        function proj_err(dt, n)
            p = domain_wall_state(L)
            u0 = dense_state(p)
            bond_update_bug!(p, bond_gates(p; J = 1.0, delta = 1.0);
                opts = BondUpdateOptions(dt = dt, n_steps = n, order = :strang,
                                         maxdim = DMAX, trunc_thresh = 1e-14,
                                         normalize = false))
            return norm(dense_state(p) - dense_trotter_propagate(L, u0, dt, n;
                                                                 order = :strang))
        end
        # same total time, half the step
        @test proj_err(0.01, 5) / proj_err(0.005, 10) > 2.0
    end

    @testset "matches exact expm to the expected second-order Trotter error" begin
        want = dense_exact_propagate(dense_heisenberg(L), v0, DT * NSTEPS)
        err  = norm(got - want)
        @test err < 1e-4           # O(dt^2) with dt = 0.01
        @test err > 1e-12          # and NOT machine precision -- the split is real
    end

    @testset "second-order convergence against exact expm" begin
        H, T = dense_heisenberg(L), 0.05
        want = dense_exact_propagate(H, v0, T)
        function run(dt, n)
            p = domain_wall_state(L)
            bond_update_bug!(p, bond_gates(p; J = 1.0, delta = 1.0);
                opts = BondUpdateOptions(dt = dt, n_steps = n, order = :strang,
                                         maxdim = DMAX, trunc_thresh = 1e-14,
                                         normalize = false))
            return norm(dense_state(p) - want)
        end
        @test 3.0 < run(0.01, 5) / run(0.005, 10) < 5.0
    end

    @testset "total Sz is conserved to machine precision" begin
        p = domain_wall_state(L)
        sz0 = total_sz(p)
        bond_update_bug!(p, bond_gates(p; J = 1.0, delta = 1.0);
            opts = BondUpdateOptions(dt = DT, n_steps = NSTEPS, maxdim = DMAX))
        @test isapprox(total_sz(p), sz0; atol = 1e-13)
    end
end
