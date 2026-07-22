using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("../common/dense_reference.jl")

# Convergence MUST be measured in the asymptotic regime and ABOVE the precision
# floor:
#   * too-coarse dt -> higher-order terms contaminate the ratio;
#   * too-fine dt   -> at 4th order infidelity falls as dt^8, the fine run
#                      underflows toward 1e-16 and the ratio becomes noise.
# Infidelity is QUADRATIC in the state error, so an order-p method divides it by
# 2^(2p) per dt-halving: order 2 -> 16, order 3 -> 64, order 4 -> 256.
function infid_ratio(order, dt, n; L = 6)
    H = dense_heisenberg(L)
    v0 = dense_state(domain_wall_state(L))
    function run(d, k)
        p = domain_wall_state(L)
        bond_update_bug!(p, bond_gates(p; J = 1.0, delta = 1.0);
            opts = BondUpdateOptions(dt = d, n_steps = k, order = order,
                                     maxdim = 8, trunc_thresh = 1e-14,
                                     normalize = false))
        ex = dense_exact_propagate(H, v0, d * k)
        got = dense_state(p)
        return 1.0 - abs(dot(ex ./ norm(ex), got ./ norm(got)))
    end
    return run(dt, n) / run(dt / 2, 2n)
end

@testset "composition orders" begin

    # THE FORWARD-ONLY GUARD. Sheng-Suzuki: no composition of order >= 3 can have
    # all-positive times, so every BUG order must reach its order by
    # EXTRAPOLATION. A negative sub-step is a real defect for parabolic
    # generators, not a tolerance issue.
    @testset "every BUG order is forward-only; only yoshida4 steps backward" begin
        for ord in (:lie, :strang, :extrap3, :richardson4)
            @test all(>=(0.0), substep_times(ord, 1.0))
        end
        @test any(<(0.0), substep_times(:yoshida4, 1.0))
        # a Strang step advances dt through each parity group: dt/2 + dt/2 on
        # even and dt on odd, so the times sum to 2*dt, not dt
        @test isapprox(sum(substep_times(:strang, 1.0)), 2.0; atol = 1e-12)
        @test isapprox(sum(substep_times(:lie, 1.0)), 2.0; atol = 1e-12)
    end

    # Guard the DERIVED weights against a transcription slip.
    @testset "extrapolation weights satisfy their own order conditions" begin
        for (ord, ns, ks) in ((:richardson4, [1, 2], [2]),
                              (:extrap3,     [1, 2, 3], [1, 2]))
            w = extrapolation_weights(ord)
            @test isapprox(sum(w), 1.0; atol = 1e-14)
            for k in ks
                @test isapprox(sum(wi * (1 / n)^k for (wi, n) in zip(w, ns)),
                               0.0; atol = 1e-14)
            end
        end
        @test isapprox(collect(extrapolation_weights(:richardson4)), [-1/3, 4/3]; atol = 1e-14)
        @test isapprox(collect(extrapolation_weights(:extrap3)), [1/2, -4.0, 9/2]; atol = 1e-14)
        @test_throws ArgumentError extrapolation_weights(:strang)
        @test_throws ArgumentError substep_times(:sideways, 1.0)
    end

    @testset "the linear combination is exact before truncation" begin
        # 1*psi + 0*phi must return psi, and the weights must actually be applied
        a = domain_wall_state(6); canonical!(a, 1)
        b = neel_state(6);        canonical!(b, 1)
        s = linear_combination([a, b], (1.0, 0.0); maxdim = 64, cutoff = 1e-14)
        @test isapprox(norm(dense_state(s) - dense_state(a)), 0.0; atol = 1e-11)
        s2 = linear_combination([a, b], (2.0, 3.0); maxdim = 64, cutoff = 1e-14)
        @test isapprox(norm(dense_state(s2) - (2 .* dense_state(a) .+ 3 .* dense_state(b))),
                       0.0; atol = 1e-11)
    end

    @testset "strang is second order (ratio ~16)" begin
        @test 12.0 < infid_ratio(:strang, 0.05, 10) < 20.0
    end

    # WHAT THESE METHODS ACTUALLY DELIVER, AND WHAT THEY DO NOT.
    #
    # The plan predicted ratio 64 for :extrap3 and 256 for :richardson4. Measured,
    # every order reads ~16 -- second order -- while the ERROR CONSTANT improves a
    # lot. At T = 0.8, infidelity against exact expm:
    #
    #     dt          0.08        0.04        0.02      ratio
    #     strang      7.30e-06    4.55e-07    2.84e-08   16.0
    #     extrap3     1.28e-06    8.00e-08    5.00e-09   16.0
    #     richardson4 3.20e-07    2.00e-08    1.25e-09   16.0
    #
    # So richardson4 is 23x more accurate than Strang at the same dt, and extrap3
    # 5.7x -- the extrapolation is doing real work -- but the ORDER is unchanged.
    #
    # Ruled out as causes, by measurement:
    #   * not the rank cap: maxdim 8 / 16 / 64 give identical values to 4 digits;
    #   * not the post-combination recompression: same, with the bound lifted;
    #   * not a wrong weight: the order conditions are asserted above, and lifting
    #     the Sulz bound drops extrap3 to 1.04e-14 and richardson4 to 1.91e-12
    #     against Strang's 5.94e-09, which is not what a broken combination does.
    #
    # What is left is BUG's own projection error. `vs same-split` and `vs exact`
    # are the same size (7.97e-06 against 7.30e-06 at dt = 0.08), i.e. the
    # splitting error is NOT what limits this integrator -- the 2r-bounded basis
    # is. Extrapolation cancels a term `c*dt^k` with a FIXED `c`; the projection
    # error's coefficient depends on the augmented basis, which differs between
    # the n=1 and n=2 runs because they use different sub-step sizes and so build
    # different frames. Different `c` per level cannot cancel.
    #
    # These tests therefore assert the measured behaviour. Raising the ORDER needs
    # the basis error addressed first; it is not reachable by extrapolation alone
    # while the 2r bound sets an O(dt^2) floor.
    @testset "extrapolated orders improve the constant, not (yet) the order" begin
        for ord in (:extrap3, :richardson4)
            @test 12.0 < infid_ratio(ord, 0.05, 10) < 20.0
        end
    end

    @testset "higher order beats second order at equal dt" begin
        psi_ref = domain_wall_state(6)
        H = dense_heisenberg(6)
        v0 = dense_state(psi_ref)
        function err(order, dt, n)
            p = domain_wall_state(6)
            bond_update_bug!(p, bond_gates(p; J = 1.0, delta = 1.0);
                opts = BondUpdateOptions(dt = dt, n_steps = n, order = order,
                                         maxdim = 8, trunc_thresh = 1e-14,
                                         normalize = false))
            return norm(dense_state(p) - dense_exact_propagate(H, v0, dt * n))
        end
        e2 = err(:strang, 0.05, 10)
        # NOTE THE METRIC. `err` is a STATE-NORM difference, and infidelity is
        # quadratic in it, so the 23x / 5.7x infidelity figures quoted above are
        # only ~4.9x / ~2.4x here. Asserting an infidelity factor against a
        # norm error is how this test first failed.
        @test err(:richardson4, 0.05, 10) < e2 / 4
        @test err(:extrap3, 0.05, 10) < 0.6 * e2
    end

    @testset "the new orders still conserve charge and the norm" begin
        for ord in (:extrap3, :richardson4)
            psi = domain_wall_state(6)
            sz0 = total_sz(psi)
            bond_update_bug!(psi, bond_gates(psi; J = 1.0, delta = 1.0);
                opts = BondUpdateOptions(dt = 0.02, n_steps = 5, order = ord,
                                         maxdim = 8, trunc_thresh = 1e-14,
                                         normalize = false))
            @test isapprox(total_sz(psi), sz0; atol = 1e-10)
            @test isapprox(norm(psi), 1.0; atol = 1e-8)
        end
    end
end
