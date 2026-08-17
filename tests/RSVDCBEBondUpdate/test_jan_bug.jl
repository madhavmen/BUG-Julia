using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# Jan BUG: one CBE basis sweep across the whole chain with QR-only bond moves, then ONE 0-site
# bond update at the far end, then one truncation. Direction alternates between steps.
#
# WHAT CAN AND CANNOT BE ASSERTED HERE, because the difference from the other two variants is
# the point:
#
#   * There is NO full-rank exactness test. The connecting tensor sits on a BOUNDARY bond, whose
#     right (resp. left) space is a single site, so the Galerkin problem spans at most `chi*d`
#     directions rather than the whole space -- one step is not `exp(-iH dt)` even at saturated
#     bonds. Asserting otherwise would assert a property the scheme does not claim.
#   * What must hold: the basis sweep leaves the STATE untouched (expansion + QR, both
#     lossless), the step converges to the exact propagator as `dt -> 0`, norm is preserved, the
#     rank bootstraps off a product state, and the alternation leaves the centre where the next
#     sweep begins.
#
# What a boundary-bond Galerkin step with mirrored directions composes to was the open question
# this variant existed to answer, and it is now answered: SECOND order, p = 2.00 measured for
# both delta = 1 and delta = 0, so the exponent is asserted rather than merely logged.

_exact_step(psi::SymMPS, L::Int, dt::Float64; delta = 1.0) =
    exp(-im * dt * Matrix(dense_heisenberg(L; delta = delta))) * dense_state(psi)

function _warm(L::Int; n_steps = 6, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

@testset "jan_bug (full-sweep basis update, boundary-bond Galerkin)" begin

    @testset "tau = 0 is the identity" begin
        # At `tau = 0` the S step is the identity, so `orth(U_ex S0)` is the frame's own range
        # and the discarded `R` carries everything: the whole sweep must reduce to a change of
        # basis. This is the invariant the scheme rests on -- no bond is dynamically evolved on
        # the way out, so nothing may move when the step itself does nothing.
        set_symmetry!(:U1)
        for L in (4, 6), dir in (:right, :left)
            psi = _warm(L)
            v0, bd0 = dense_state(psi), bond_dims(psi)
            info = jan_bug_step!(psi, xxz_mpo(L), ComplexF64(0);
                                 truncate = false, direction = dir,
                                 rng = Random.MersenneTwister(0x5EED))
            @test info.direction === dir
            @test norm(dense_state(psi) - v0) < 1e-9
            @test bond_dims(psi) == bd0                # no growth without evolution
            @test info.end_rank >= 1
        end
    end

    @testset "the S step at each bond grows the rank; the discarded R keeps the state" begin
        # `exp(tau H_eff)` acts from both sides, so `S1` is generically of full rank in the
        # expanded frames even though `S0` carries only `r`: that is where the rank comes from,
        # and it is why the QR of the EVOLVED bond gives a wider basis than the frame did.
        set_symmetry!(:U1)
        L = 6
        psi = domain_wall_state(L)
        info = jan_bug_step!(psi, xxz_mpo(L), ComplexF64(-im * 0.05);
                             truncate = false, rng = Random.MersenneTwister(0x5EED))
        @test maximum(bond_dims(psi)) > 1              # off a product state, in one step
        @test all(>=(1), info.expanded)
        @test abs(norm(psi) - 1) < 1e-2                # see the norm testset below
    end

    @testset "one step converges to the exact propagator as dt -> 0" begin
        set_symmetry!(:U1)
        L = 4
        for delta in (1.0, 0.0)
            mpo = xxz_mpo(L; delta = delta)
            errs = Float64[]
            for dt in (0.02, 0.01, 0.005)
                psi = _warm(L)
                want = _exact_step(psi, L, dt; delta = delta)
                jan_bug_step!(psi, mpo, ComplexF64(-im * dt);
                              maxdim = 32, rng = Random.MersenneTwister(0x5EED))
                push!(errs, norm(dense_state(psi) - want))
            end
            @test issorted(errs; rev = true)           # halving dt must reduce the error
            @info "jan_bug one-step order" delta p = log2(errs[1] / errs[2]) errs
            # SECOND order, measured at p = 2.00 for both deltas. Pinned rather than logged
            # now that it is established: a boundary-bond Galerkin step composing to O(dt^2)
            # is the substantive claim of this variant, and the probe regression above dropped
            # it to exactly 1.00, so this assertion is what catches that class of change.
            @test log2(errs[1] / errs[2]) > 1.8
            @test log2(errs[2] / errs[3]) > 1.8
            @test errs[end] < 1e-4
        end
    end

    @testset "the norm defect is O(dt^2): the price of discarding R" begin
        # THE STEP IS NOT NORM PRESERVING, and that is a property of the scheme rather than a
        # bug, so it is measured rather than tolerated with a loose bound.
        #
        # `_absorb_left!` re-expresses the ORIGINAL amplitude in `range(Q)`, and
        # `Q = orth(U_ex S1)` with `S1 = exp(tau H0) S0`. Since `U_ex S0` has the same column
        # space as `A_i`, `range(Q)` is an O(tau) ROTATION of `range(A_i)` -- so the
        # re-expression is a projection that loses O(tau^2). At `tau = 0` it is exact, which
        # is the `tau = 0` testset above; here the quadratic law is pinned.
        #
        # This is the same O(tau^2) as the K-step approximation every BUG makes; what is
        # unusual is only that it shows up as a norm rather than being absorbed into a
        # subsequent solve, because no evolution is kept at these bonds.
        set_symmetry!(:U1)
        L = 6
        mpo = xxz_mpo(L)
        defects = Float64[]
        for dt in (0.08, 0.04, 0.02)
            psi = _warm(L)
            n0 = norm(psi)
            jan_bug_step!(psi, mpo, ComplexF64(-im * dt);
                          maxdim = 32, rng = Random.MersenneTwister(0x5EED))
            push!(defects, abs(norm(psi) - n0))
        end
        @info "jan_bug norm defect" defects p = log2.(defects[1:2] ./ defects[2:3])
        @test issorted(defects; rev = true)
        for p in log2.(defects[1:2] ./ defects[2:3])
            @test 1.7 < p < 2.3                        # quadratic, not linear and not exact
        end
        @test defects[end] < 1e-4
    end

    @testset "the truncation happens after the sweep, not during it" begin
        # Structural, and the reason it matters: truncating mid-sweep would delete directions
        # that are merely not-yet-populated. With `truncate = false` the expansion survives the
        # step intact, so the bonds come out WIDER than with it on -- that difference is the
        # truncation, and it can only be applied once the far-end update has propagated.
        set_symmetry!(:U1)
        L = 6
        mpo = xxz_mpo(L)
        p1 = _warm(L); jan_bug_step!(p1, mpo, ComplexF64(-im * 0.02); truncate = false,
                                     rng = Random.MersenneTwister(0x5EED))
        p2 = _warm(L); i2 = jan_bug_step!(p2, mpo, ComplexF64(-im * 0.02); truncate = true,
                                          maxdim = 32,
                                          rng = Random.MersenneTwister(0x5EED))
        @test maximum(bond_dims(p1)) >= maximum(bond_dims(p2))
        @test i2.discarded >= 0.0
        # Both are the same physical state to within what the truncation threw away.
        @test norm(dense_state(p1) - dense_state(p2)) < 1e-6
    end

    @testset "the update bond is expanded, not just visited" begin
        # THE REGRESSION THIS EXISTS FOR. The whole step is the Galerkin solve at the far bond,
        # so that bond's frame has to be WIDER than the state's rank there or the step is taken
        # in the subspace the state already occupies and evolves almost nothing.
        #
        # A `two_sided` probe (both Gaussians folded onto the projectors, ranked by the small
        # `M = Q_L'(H Theta)Q_R`) got this wrong and was removed: `M` coupled the two sides, so
        # when the OUTER side saturated -- guaranteed at a boundary bond, where `room = d - r`
        # -- it declined to expand the inner side too. Measured cost: first order instead of
        # second, and `err 2.0e-2` against a `2.2e-2` standstill, i.e. barely moving.
        #
        # `n_new` is the count CBE admitted, before the final SVD trims the bond back to
        # `min(b_L, b_R)`. Asserting on `expanded` instead would NOT catch this: that number is
        # capped at the boundary either way, which is exactly how the bug hid.
        set_symmetry!(:U1)
        L = 6
        psi = _warm(L)
        info = jan_bug_step!(psi, xxz_mpo(L), ComplexF64(-im * 0.02);
                             maxdim = 32, rng = Random.MersenneTwister(0x5EED))
        @test info.n_new[L - 1] > 0                    # the bond the step lives on
        # ONLY the update bond is asserted, and the rest is logged. A bond whose frame already
        # spans its local space has `room = fused_dim - r = 0` and correctly admits nothing, so
        # `all(> 0, n_new)` is not a property of a correct sweep -- it failed on bond 1 of a warm
        # L=6 state (`room_l = 1*d - r = 0` at `r = 2`) and then failed again on an interior bond
        # when narrowed to `2:L-1`. Which bonds are saturated depends on the state, so pinning a
        # pattern here would pin the warm-up rather than the sweep. Global growth is covered by
        # the rank-bootstrap testset, which asserts the thing that actually matters.
        @info "jan_bug n_new / expanded per bond" info.n_new info.expanded
        @test any(>(0), info.n_new)
    end

    @testset "rank bootstraps off a product state" begin
        # Why CBE is mandatory rather than optional: without the expansion a bond update cannot
        # change a bond dimension, so this would sit at chi = 1 forever.
        set_symmetry!(:U1)
        L = 6
        psi = domain_wall_state(L)
        @test all(==(1), bond_dims(psi))
        info = jan_bug!(psi, xxz_mpo(L);
                        opts = CBEBugOptions(dt = 0.05, n_steps = 6, maxdim = 16))
        @test all(>(1), info.max_bond_dims)            # left chi = 1 and never fell back
        # The norm is NOT preserved to machine precision -- discarding `R` projects, at O(dt^2)
        # per step (measured and pinned in the norm testset above). At dt = 0.05 that is a few
        # times 1e-4 per step, so this bounds the defect rather than denying it.
        @test all(n -> abs(n - 1) < 5e-3, info.norms)
    end

    @testset "the driver alternates direction and accepts either H representation" begin
        set_symmetry!(:U1)
        L = 4
        h = xxz_chain(L)
        o = CBEBugOptions(dt = 0.05, n_steps = 4, maxdim = 8)
        p1 = _warm(L); i1 = jan_bug!(p1, h; opts = o)
        p2 = _warm(L); i2 = jan_bug!(p2, mpo_from_terms(h); opts = o)
        @test i1.bond_dims == i2.bond_dims
        @test norm(dense_state(p1) - dense_state(p2)) < 1e-12
        # The post-sweep truncation ends at the end the sweep finished at, which is where the
        # next sweep starts: after an even number of steps beginning :right, that is site 1.
        @test p1.center == 1
    end

    # ── the LOSSLESS variant: no S step away from the root, so nothing is discarded ──────
    #
    # `lossless = true` replaces the per-bond KLS+QR with a pure CBE expansion CARRIED bond to
    # bond, and takes one KLS step at `root`. Two measured facts motivate it, both recorded in
    # `jan_bug.jl`'s header:
    #
    #   * with the QR-discard sweep, every bond AFTER the kept step projects the evolution that
    #     step just performed. Kept step at bond 5/4/3 of L=6: 2.1e-5 / 7.7e-4 / 5.0e-3, and the
    #     dt-order collapses to zero. A discard is only harmless BEFORE the evolution.
    #   * with nothing discarded there is no O(tau^2) loss anywhere, and at the CENTRE root the
    #     step is exact at full rank: 1.2e-14 over 24 steps against the QR-discard's 2.9e-4.
    #
    # The carry is not cosmetic: writing the raw expanded frame into the state leaves a
    # zero-padded remainder that the next bond's `_frame_from` drops again (`cbe_lubich.jl`'s
    # header records L=18 freezing at maxbond 2 that way). The `tau = 0` testset below is the
    # one that would catch a regression there -- a collapsing sweep is not the identity.

    @testset "lossless: tau = 0 is exactly the identity at every root" begin
        # A pure expansion is lossless and a pure gauge change, so with no evolution NOTHING may
        # move -- at any root, either direction. This is also the padding-collapse detector.
        set_symmetry!(:U1)
        for L in (4, 6)
            base = _warm(L)
            v0 = dense_state(base)
            for c in 1:(L - 1), dir in (:right, :left)
                psi = copy(base)
                jan_bug_step!(psi, xxz_mpo(L), ComplexF64(0);
                              direction = dir, root = c, lossless = true, truncate = false,
                              rng = Random.MersenneTwister(0x5EED))
                @test norm(dense_state(psi) - v0) < 1e-9
            end
        end
    end

    @testset "lossless at the centre root is exact at full rank" begin
        # The centre is not merely the best root, it is the structurally right one: the forward
        # sweep refreshes the left half and the reverse sweep the right half, so an alternating
        # pair covers the chain. A boundary root cannot do that -- its reverse sweep runs over a
        # single bond -- which is why it is not tested for exactness here.
        set_symmetry!(:U1)
        for L in (4, 6)
            mpo = xxz_mpo(L)
            H = Matrix(dense_heisenberg(L))
            psi = _warm(L)
            v0 = dense_state(psi)
            dt, n = 0.01, 6
            dir = :right
            for _ in 1:n
                jan_bug_step!(psi, mpo, ComplexF64(-im * dt);
                              direction = dir, root = L ÷ 2, lossless = true, maxdim = 64,
                              rng = Random.MersenneTwister(0x5EED))
                dir = dir === :right ? :left : :right
            end
            err = norm(dense_state(psi) - exp(-im * (n * dt) * H) * v0)
            @info "jan_bug lossless centre" L err
            @test err < 1e-10                           # machine precision, not an order claim
            @test abs(norm(psi) - 1) < 1e-10            # and nothing is discarded, so unitary
        end
    end

    @testset "lossless: an off-centre interior root needs the growing basis" begin
        # Roots either side of the centre have a Galerkin ceiling ABOVE the sector dimension but
        # the plain expansion does not reach it (L=6: b_L = 8 against a ceiling of 16 at root 4).
        # `grow_iters` regrows the frames around each Krylov vector and does reach it --
        # 1.227e-5 -> 2.049e-8 -> 1.8e-15 as grow_iters goes 0 -> 2 -> 4. This pins the
        # DIRECTION of that effect, which is the claim; the exact figures live in the header.
        set_symmetry!(:U1)
        L = 6
        mpo = xxz_mpo(L)
        psi0 = _warm(L)
        v0 = dense_state(psi0)
        H = Matrix(dense_heisenberg(L))
        errs = Float64[]
        for g in (0, 2, 4)
            psi = copy(psi0)
            jan_bug_step!(psi, mpo, ComplexF64(-im * 0.01);
                          direction = :right, root = 4, grow_iters = g, lossless = true,
                          maxdim = 64, rng = Random.MersenneTwister(0x5EED))
            push!(errs, norm(dense_state(psi) - exp(-im * 0.01 * H) * v0))
        end
        @info "jan_bug lossless root=4 vs grow_iters" errs
        @test issorted(errs; rev = true)
        @test errs[end] < errs[1] / 100                 # growth buys orders, not percent
    end

    @testset "against the dense propagator over several steps" begin
        # Not an accuracy claim relative to the other variants -- that is a benchmark. This pins
        # that repeated stepping tracks the exact evolution rather than freezing or drifting,
        # measured against DOING NOTHING over the same interval.
        set_symmetry!(:U1)
        L, dt, n = 4, 0.005, 8
        psi = _warm(L)
        v = dense_state(psi)
        H = Matrix(dense_heisenberg(L))
        jan_bug!(psi, xxz_mpo(L);
                 opts = CBEBugOptions(dt = dt, n_steps = n, maxdim = 32, normalize = false))
        want = exp(-im * (n * dt) * H) * v
        err = norm(dense_state(psi) - want)
        @info "jan_bug multi-step" err standstill = norm(want - v)
        @test err < 0.5 * norm(want - v)
    end
end
