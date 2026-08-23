# `rexpand` -- THE SHIPPED DEFAULT OF `cbe_bug_step!` -- AND THE ONE DEFECT IT IS PRONE TO.
#
# WHY THIS FILE EXISTS SEPARATELY. `rexpand = true` is the default and was, until this file, the
# only scheme knob with no test. It shipped with a real defect for a while, and the shape of that
# defect is the reason a targeted file is needed rather than another testset in `test_sweeps.jl`:
#
#   THE DEFECT. `rexpand` expands each bond a SECOND time, and the frame it hands that second
#   expansion is `W[i]·(W[i]'K)·psi[i+1]`. That equals the true two-site block ONLY IF `W[i]`
#   spans `K`. It need not -- `W[i]` is built from `svd(K1).U`, the EVOLVED tensor, and
#   `Kex = K·M` matricises through `M`, so its left span can be strictly smaller than `K`'s.
#   Unfixed, the second expansion sketches `H·Theta` against a PROJECTED state and returns the
#   right directions for the WRONG `Theta`. The fix is one `_union_left(svd(K).U, Wk)` per
#   half-sweep -- a REPRESENTATION fix that adds no direction not already needed to represent `K`.
#
# ⛔ AND WHY THE OBVIOUS PLACES DO NOT CATCH IT -- this is the whole design constraint here:
#
#   * NOT at full rank. `test_sweeps.jl` has "cbe_bug_step! is exact at full rank", and once the
#     expansion reaches the whole local space `P_W = I` and the defect VANISHES identically. The
#     existing exactness test is blind to this by construction, which is exactly how it shipped.
#   * NOT at `tau = 0`. Then `K1 == K`, the two spans coincide, and the step is the identity
#     either way.
#   * NOT on OAT or XX. OAT's evolved basis is nearly exact, so the projection loses almost
#     nothing: MEASURED 3.27e-11 -> 4.63e-09 with the defect present, a move no sane threshold
#     would flag. XX did not move at all (8.1911e-05 either way). On TFIM the same defect gives
#     2.508e-01 (`:Z2`) / 5.467e-01 (`:none`) against ~1.4e-05 -- WRONG ANSWERS, not degraded
#     ones. The model dependence IS the tell, so the test must be TFIM and must be CAPPED.
#
# ⛔ THE REFERENCE IS `tfim_x_profile`, THE MAJORANA CLOSED FORM -- NOT A DENSE PROPAGATOR. That
# is not a stylistic preference here, it is what makes the test able to see the thing it is for:
# a 2^L reference caps this at L~10, and the defect is FOUR ORDERS larger at L=16 (2.5e-01) than
# the margin a small-L dense run would leave. The closed form costs O(L³) and is pinned against
# exact diagonalisation in `tests/common/test_analytic_reference.jl`, which is the one place a
# dense object belongs: validating the reference, never the integrator.
#
# The assertions are deliberately of different kinds. The absolute one catches "both arms wrong
# together"; the RATIO one is self-calibrating -- it compares two arms of the same run, so it does
# not drift with `L`, `D`, `dt` or machine, and it carries ~4 orders of margin against the defect
# while sitting ~10x clear of the true `rexpand`-vs-`kaug` spread.

@testset "cbe_bug_step! rexpand (the default)" begin
    set_symmetry!(:Z2)
    # L=16 is the regime the defect was MEASURED in, and is affordable only because the reference
    # is a 32x32 congruence rather than a 65536-dimensional propagator.
    L, J, h, dt, nstep = 16, 1.0, 1.0, 0.05, 20
    D = 16                      # ⛔ MUST BITE. At full expansion the defect is identically zero.

    mpo   = tfim_mpo(L; J = J, h = h)
    dirs  = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    exact = tfim_x_profile(L, dt * nstep, dirs; J = J, h = h)

    function evolve(; kw...)
        psi = ising_kink_state(L)
        for _ in 1:nstep
            cbe_bug_step!(psi, mpo, ComplexF64(-im * dt);
                          maxdim = D, trunc_thresh = 1e-12, maxiter = 16, kw...)
        end
        return maximum(abs.(x_profile(psi) - exact))
    end

    err_rex  = evolve(rexpand = true)
    # ⛔ `rexpand = true` OVERRIDES `kaug`, so the comparison arm MUST pass `rexpand = false` or
    # it silently compares the arm with itself. (Documented on `kaug`; it has bitten before.)
    err_kaug = evolve(rexpand = false, kaug = true)

    @test isfinite(err_rex) && isfinite(err_kaug)

    # (1) ABSOLUTE. Catches both arms being wrong in the same direction, which a ratio cannot.
    #     The defect put this at 2.5e-01; a correct run is orders below the bound.
    @test err_rex < 5e-2

    # (2) RATIO -- the targeted one. `rexpand` and `kaug` restore the same width by different
    #     mechanisms and must land in the same place; the defect separated them by ~4 orders.
    @test err_rex < 20 * err_kaug

    # (3) The knob is REACHED, not silently inert: `rexpand` overriding `kaug` means an arm that
    #     ignored the flag would return the identical number, and both (1) and (2) would pass on
    #     an arm that never ran. `kaug = false, rexpand = false` removes the width restoration
    #     altogether and must be measurably worse than either.
    err_bare = evolve(rexpand = false, kaug = false)
    @test err_bare > err_rex

    set_symmetry!(:U1)
end
