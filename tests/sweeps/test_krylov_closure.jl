# tests/sweeps/test_krylov_closure.jl -- UT1..UT5 of the general-order Krylov-closure plan
# (krylov_closure_cbe_bug_realtime_L16_plan.md §4); UT6..UT9 on full steps:
# test_krylov_closure_steps.jl.
#
#   UT1  the frontier genuinely advances: a dense hopping chain through the production loop,
#        and the single-magnon XX chain through the production MPS closure
#   UT2  matrix-free outside-space action against the exact (I - P) H v, left and right
#   UT3  order moments (dense), and the step's local error order against exp(-iH dt) (MPS)
#   UT4  a projected Krylov breakdown is not accepted
#   UT5  frame invariants after every frontier sweep: nesting, isometry, carries, environments
#   UT5b the frontier embedding carries a frame's defect instead of squaring it (polar frames)
#
# ⛔ THE DENSE TESTS DRIVE THE PRODUCTION LOOP (`_closure_loop!`) AND FRONTIER RECURSION
# (`_projected_frontiers`); only the space is a matrix. Everything MPS-shaped runs the production
# frontier sweep, frames and step.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "krylov_closure_fixtures.jl"))

@testset "Krylov closure (general order): frontiers and frames" begin

@testset "UT1: the frontier genuinely advances" begin
    @testset "dense hopping chain, production loop" begin
        n = 6
        A = zeros(ComplexF64, n, n)
        for k in 1:(n - 1)
            A[k, k + 1] = A[k + 1, k] = 1
        end
        e1 = _kc_unit(n, 1)
        for p in 1:4
            Q, res = _kc_dense_closure(A, e1, reshape(e1, n, 1), ComplexF64(-1im), 1e-12, p)
            @test res.converged
            @test size(Q, 2) == p + 1                    # e1 .. e_{p+1}, no redundant column
            @test norm(Q' * Q - I) < 1e-12
            @test all(norm(Q * (Q' * _kc_unit(n, k)) - _kc_unit(n, k)) < 1e-12 for k in 1:(p + 1))
            @test norm(Q' * _kc_unit(n, p + 2)) < 1e-12   # nothing order p does not need
            # pass 1 discovers ONE new level per frontier, in order -- e2, then e3, ... -- and
            # the confirming pass rediscovers nothing
            p1 = [l for l in res.log if l.pass == 1]
            @test [l.frontier for l in p1] == collect(0:(p - 1))
            @test all(l.added == 1 for l in p1)
            @test all(l.added == 0 for l in res.log if l.pass > 1)
        end
    end

    @testset "single magnon on the XX chain, production MPS closure" begin
        set_symmetry!(:U1)
        L, c = 6, 3
        mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
        psi = _kc_magnon(L, 1)
        tau = ComplexF64(-1im)
        for p in 1:4, checks in (false, true)
            fr = RC.frames_from_state(psi, mpo, c)
            alloc = basis_allocation(AdaptiveBudget(1e-10, 0.0, 1.0), L - 1; tau = tau,
                                     nrm = 1.0, nfront = p)
            cl = RC._krylov_closure!(fr, psi, mpo, RC._project_root(psi, fr), tau, alloc,
                                     _kc_ckw(alloc, 64); order = p, workcap = 64, nrm = 1.0,
                                     include_zero = true, checks = checks)
            @test cl.converged
            # the magnon reached sites 1 .. p+1: each lies in the Galerkin space
            held = [norm(RC._project_root(_kc_magnon(L, k), fr)) for k in 1:(p + 1)]
            @test all(h -> abs(h - 1) < 1e-8, held)
            # every pass-1 frontier sweep found something new...
            p1 = [l for l in cl.log if l.pass == 1]
            @test length(p1) == p && all(l.added > 0 for l in p1)
            # ...so one pass is the whole closure without checks; with them the confirming
            # pass finds nothing
            checks || @test cl.passes == 1 && length(cl.log) == p
            checks && @test all(l.added == 0 for l in cl.log if l.pass == cl.passes) || p == 1
            @test all(left_isometry_defect(fr.W[i]) < 1e-12 for i in 1:c)
            @test all(right_isometry_defect(fr.Z[j]) < 1e-12 for j in c:(L - 1))
            @info "UT1 magnon p=$p checks=$checks: passes $(cl.passes), log $(cl.log), " *
                  "held $(held), root widths $(leg_dim(fr.W[c], 3)) x $(leg_dim(fr.Z[c], 1))"
        end
    end
end

@testset "UT2: outside-space action, matrix-free vs exact" begin
    L = 6
    # Rank-capped warm-up: at full rank the L = 6 Sz = 0 state's Schmidt frames at bond 3 span
    # the whole sector (rank 8), so `(I - P) H v = 0` there and the interior case tests nothing.
    # The cap 3 leaves bonds 2..4 incomplete and the edge bonds (dim 2) complete.
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = neel_state(L)
    for _ in 1:2
        tdvp2_step!(psi, mpo, -0.2im; maxdim = 3, trunc_thresh = 1e-14)
    end
    @test leg_dim(psi[3], 3) < 8
    Hd = dense_heisenberg(L; J = 1.0, delta = 0.0)
    for (c, which) in ((L - 1, :left), (1, :right), (3, :both))
        fr = RC.frames_from_state(psi, mpo, c)
        # the opposite edge frame is COMPLETE, so the exact action is the one side's alone
        which === :left && @test leg_dim(fr.Z[L - 1], 1) == 2
        which === :right && @test leg_dim(fr.W[1], 3) == 2
        S0 = RC._project_root(psi, fr)
        H0 = RC._root_generator(mpo, fr)
        cores = Any[S0, apply_zero_site(H0, S0),
                    RC._randomize_like(S0, Random.MersenneTwister(3))]
        for (k, core) in enumerate(cores)
            est = RC.estimate_global_complement_action(psi, mpo, fr, core)
            v  = dense_state(RC._frontier_mps(psi, fr, core))
            Pv = dense_state(RC._frontier_mps(psi, fr, apply_zero_site(H0, core)))
            exact = norm(Hd * v - Pv)                     # ||(I - P) H v||, P H v = W (H0 core) Z
            @test exact > 1e-3                            # the projector is genuinely incomplete
            if which === :left
                @test isapprox(est.left, exact; rtol = 1e-8)
                @test est.right < 1e-10 * exact
            elseif which === :right
                @test isapprox(est.right, exact; rtol = 1e-8)
                @test est.left < 1e-10 * exact
            else
                @test max(est.left, est.right) <= exact * (1 + 1e-8)
                @test exact <= est.bound * (1 + 1e-8)
            end
            @info "UT2 c=$c core $k: exact $(exact), left $(est.left), right $(est.right)"
            # cores 2, 3 ran on the generation's cached environments (`_side_cache!`): the
            # same per-bond misses as frames built afresh, and the cache is this generation's
            @test fr.lcache.gen == fr.gen && fr.rcache.gen == fr.gen
            fresh = RC.estimate_global_complement_action(psi, mpo,
                                                         RC.frames_from_state(psi, mpo, c), core)
            sc = max(est.left, est.right)
            @test all(abs.(est.per_left .- fresh.per_left) .<= 1e-10 * sc)
            @test all(abs.(est.per_right .- fresh.per_right) .<= 1e-10 * sc)
        end
    end

    @testset "the sampled estimate is unbiased, both sides" begin
        c = 3
        fr = RC.frames_from_state(psi, mpo, c)
        S0 = RC._project_root(psi, fr)
        core = apply_zero_site(RC._root_generator(mpo, fr), S0)
        ex = RC.estimate_global_complement_action(psi, mpo, fr, core)
        ndraw = 120
        sl, sr = zeros(c), zeros(L - c)
        for s in 1:ndraw
            e = RC.estimate_global_complement_action(psi, mpo, fr, core; nsample = 3,
                                                     rng = Random.MersenneTwister(100 + s))
            sl .+= e.per_sampled_left .^ 2 ./ ndraw
            sr .+= e.per_sampled_right .^ 2 ./ ndraw
        end
        for (mean2, exact) in zip(vcat(sl, sr), vcat(ex.per_left, ex.per_right))
            exact < 1e-8 * max(ex.left, ex.right) && continue
            @test isapprox(mean2, exact^2; rtol = 0.25)
        end
        @info "UT2 sampled: per-bond mean square $(vcat(sl, sr)) vs exact " *
              "$(vcat(ex.per_left, ex.per_right) .^ 2)"
    end
end

@testset "UT3: order moments" begin
    @testset "dense: closure through p-1 reproduces A^j psi for j <= p, and not p+1" begin
        n = 24
        rng = Random.MersenneTwister(5)
        X = randn(rng, ComplexF64, n, n)
        A = (X + X') / 2
        A ./= opnorm(A)
        psi = normalize(randn(rng, ComplexF64, n))
        for p in 1:4
            Q, res = _kc_dense_closure(A, psi, reshape(psi, n, 1), ComplexF64(-0.1im),
                                       1e-13, p)
            @test res.converged
            B, s0 = Q' * A * Q, Q' * psi
            @test all(norm(Q * (B^j * s0) - A^j * psi) < 1e-9 for j in 1:p)
            @test norm(Q * (B^(p + 1) * s0) - A^(p + 1) * psi) > 1e-3   # missing frontier p
            errs = [norm(Q * (exp(-1im * dt * B) * s0) - exp(-1im * dt * A) * psi)
                    for dt in (0.1, 0.05, 0.025)]
            slopes = log2.(errs[1:(end - 1)] ./ errs[2:end])
            @test all(abs.(slopes .- (p + 1)) .< 0.2)      # local error O(dt^(p+1))
        end
    end

    @testset "MPS: the step's local error order from a domain wall, against exp(-iH dt)" begin
        # updated_cbe_high_order_parallel.md T5.1: exactly p = 1..5, each MEASURED; an order the
        # floors hide is a failure (T5.5), not a pass.
        for p in 1:5
            errs, floors = _dw_errors(p, (0.4, 0.2, 0.1))
            slope = _finest_order(errs, floors)
            @info "UT3 MPS p=$p: local errors $(errs), floors $(floors), slope $(slope) " *
                  "(target >= $(p + 1))"
            @test slope !== nothing && slope >= p + 0.6
        end
    end
end

@testset "UT4: a projected Krylov breakdown is not accepted" begin
    @testset "dense: root space invariant under B, not under A" begin
        n = 12
        rng = Random.MersenneTwister(9)
        X = randn(rng, ComplexF64, n, n)
        A = (X + X') / 2
        psi = normalize(randn(rng, ComplexF64, n))
        V = Matrix(qr(hcat(psi, A * psi)).Q)[:, 1:2]
        w = randn(rng, ComplexF64, n)
        for _ in 1:2
            w -= V * (V' * w)
        end
        Q0 = hcat(psi, normalize(w))
        B0, s0 = Q0' * A * Q0, Q0' * psi
        @test norm(B0 * s0 - (s0' * B0 * s0) * s0) < 1e-12   # projected Krylov broke down
        @test norm(A * psi - Q0 * (Q0' * A * psi)) > 0.1       # ... while A leaves the space
        Q, res = _kc_dense_closure(A, psi, Q0, ComplexF64(-0.1im), 1e-13, 2;
                                   include_zero = false)
        @test res.log[1].frontier == 1 && res.log[1].added == 1
        @test res.converged
        Bq, sq = Q' * A * Q, Q' * psi
        @test all(norm(Q * (Bq^j * sq) - A^j * psi) < 1e-9 for j in 1:2)
    end

    @testset "MPS: one-dimensional root space (a product state's own frames)" begin
        set_symmetry!(:U1)
        L, c = 8, 4
        psi = domain_wall_state(L)
        tau = ComplexF64(-0.2im)
        alloc = basis_allocation(AdaptiveBudget(1e-10, 0.0, 1.0), L - 1; tau = tau,
                                 nrm = 1.0, nfront = 2)
        # Heisenberg: <DW|H|DW> != 0, so F_1 = tau*lambda*psi is parallel to psi
        mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
        fr = RC.frames_from_state(psi, mpo, c)
        @test leg_dim(fr.W[c], 3) == 1 && leg_dim(fr.Z[c], 1) == 1
        S0 = RC._project_root(psi, fr)
        F1 = RC._projected_frontiers(x -> apply_zero_site(RC._root_generator(mpo, fr), x),
                                     S0, tau, 1)[1]
        @test norm(F1) > 1e-3
        cl = RC._krylov_closure!(fr, psi, mpo, S0, tau, alloc, _kc_ckw(alloc, 64);
                                 order = 2, workcap = 64, nrm = 1.0, include_zero = false)
        @test cl.log[1].frontier == 1 && cl.log[1].added > 0
        @test cl.converged
        est = RC.estimate_global_complement_action(psi, mpo, fr, RC._project_root(psi, fr))
        @test est.bound < 1e-6                 # H psi is now inside, found through F_1 ∥ psi
        # XX: <DW|H|DW> = 0, so F_1 is the ZERO vector -- only the j = 0 sweep can see that
        # psi's action leaves the space, which is why the step always sweeps psi first.
        mpo0 = xxz_mpo(L; J = 1.0, delta = 0.0)
        fr0 = RC.frames_from_state(psi, mpo0, c)
        cl0 = RC._krylov_closure!(fr0, psi, mpo0, RC._project_root(psi, fr0), tau, alloc,
                                  _kc_ckw(alloc, 64); order = 2, workcap = 64, nrm = 1.0,
                                  include_zero = true)
        @test cl0.log[1].frontier == 0 && cl0.log[1].added > 0
        @test cl0.converged
    end
end

@testset "UT5: frame invariants after every frontier sweep (U(1), both sides)" begin
    L, c = 8, 4
    mpo, psi = _xx_warm(L, 2)
    tau = ComplexF64(-0.1im)
    alloc = basis_allocation(AdaptiveBudget(1e-8, 0.0, 1.0), L - 1; tau = tau, nrm = 1.0,
                             nfront = 3)
    ckw = _kc_ckw(alloc, 64; exact = false)              # the RSVD path
    fr = RC.frames_from_state(psi, mpo, c)
    grew = 0
    for (step, j) in enumerate((0, 1, 2, 1, 2))
        S = RC._project_root(psi, fr)
        H0 = RC._root_generator(mpo, fr)
        F = j == 0 ? S : RC._projected_frontiers(x -> apply_zero_site(H0, x), S, tau, j)[j]
        Fmps = RC._frontier_mps(psi, fr, F)
        Wold, Zold = copy(fr.W), copy(fr.Z)
        r = RC._frontier_sweep!(fr, psi, mpo, RC.KrylovFrontier(F, j, fr.gen), tau, alloc,
                                ckw; workcap = 64, nrm = 1.0,
                                rng = Random.MersenneTwister(step))
        grew += r.added
        # the old subspace is kept at EVERY bond, on both sides
        @test _kc_nesting_defect(Wold, fr.W, Zold, fr.Z, c, L) < 1e-10
        @test all(left_isometry_defect(fr.W[i]) < 1e-12 for i in 1:c)
        @test all(right_isometry_defect(fr.Z[k]) < 1e-12 for k in c:(L - 1))
        # carries: psi and the frontier swept are inside the new space, losslessly
        @test abs(norm(RC._project_root(psi, fr)) - norm(psi[psi.center])) < 1e-10
        @test abs(norm(RC._project_root(Fmps, fr)) - norm(F)) < 1e-10 * max(1.0, norm(F))
        # environments: the cached root environments are the fresh ones through the new frames
        lf, rf = _kc_fresh_envs(mpo, fr, L)
        Hc = RC._root_generator(mpo, fr)
        Hf = zero_site_h(mpo, c, lf, rf, fr.W[c], fr.Z[c])
        Xr = RC._randomize_like(RC._project_root(psi, fr), Random.MersenneTwister(step))
        @test norm(to_concrete(apply_zero_site(Hc, Xr) - apply_zero_site(Hf, Xr))) <
              1e-12 * max(1.0, norm(apply_zero_site(Hf, Xr)))
        # a stale frontier is refused once the frames have moved
        r.added > 0 && @test_throws ArgumentError RC._frontier_sweep!(
            fr, psi, mpo, RC.KrylovFrontier(F, j, fr.gen - 1), tau, alloc, ckw)
    end
    @test grew > 0                                       # non-vacuous: frames did grow
end

@testset "UT5b: the frontier embedding does not square a frame's defect (both sides)" begin
    L, c = 8, 4
    mpo, psi = _xx_warm(L, 2)
    rng = Random.MersenneTwister(5)
    fr = RC.frames_from_state(psi, mpo, c)
    # direct measures only: sqrt(||G||^2 - 2||F||^2 + k) cancels to ~sqrt(k eps) ~ 5e-8 (dt7c)
    gram(F, left) = to_concrete(left ? contract(F', (1, 2), F, (1, 2)) : contract(F, (2, 3), F', (2, 3)))
    isodef(F, left) = left ? left_isometry_defect(F) : right_isometry_defect(F)
    for (F0, left) in ((fr.W[c - 1], true), (fr.Z[c + 1], false))
        F = to_concrete(F0 + 1e-7 * RC._randomize_like(F0, rng))
        @test isodef(F, left) > 1e-9                       # premise: a non-isometric frame
        P = RC._polar_frame(F, left ? :left : :right)
        @test isodef(P, left) < 1e-13
        perp = left ? perp_component : perp_component_right
        @test norm(perp(P, F)) < 1e-12 * norm(F)           # same span
        # the transfer map of a bond that keeps its frame: `Pm = P'F` now, `Pm = F'F` before
        G = gram(F, left)
        Pnew = to_concrete(left ? contract(P', (1, 2), F, (1, 2)) : contract(F, (2, 3), P', (2, 3)))
        sq(X) = to_concrete(left ? contract(X', (1,), X, (1,)) : contract(X, (2,), X', (2,)))
        @test norm(to_concrete(sq(Pnew) - G)) < 1e-12      # Pm'Pm = G: the defect is carried
        @test norm(to_concrete(sq(G) - G)) > 1e-9          # CONTROL: Pm'Pm = G^2 = G + O(defect)
    end
    # a sweep over seeded frames hands back isometric ones
    mpo, psi = _xx_warm(L, 2)
    tau = ComplexF64(-0.1im)
    alloc = basis_allocation(AdaptiveBudget(1e-8, 0.0, 1.0), L - 1; tau = tau, nrm = 1.0,
                             nfront = 3)
    ckw = _kc_ckw(alloc, 64; exact = false)
    fr = RC.frames_from_state(psi, mpo, c)
    for i in 1:c
        fr.W[i] = to_concrete(fr.W[i] + 1e-8 * RC._randomize_like(fr.W[i], rng))
    end
    for k in c:(L - 1)
        fr.Z[k] = to_concrete(fr.Z[k] + 1e-8 * RC._randomize_like(fr.Z[k], rng))
    end
    @test maximum(left_isometry_defect(fr.W[i]) for i in 1:c) > 1e-10      # premise
    r = nothing
    for j in (0, 1, 2)                                     # the first frontier that grows the frames
        S = RC._project_root(psi, fr)
        H0 = RC._root_generator(mpo, fr)
        F = j == 0 ? S : RC._projected_frontiers(x -> apply_zero_site(H0, x), S, tau, j)[j]
        r = RC._frontier_sweep!(fr, psi, mpo, RC.KrylovFrontier(F, j, fr.gen), tau, alloc, ckw;
                                workcap = 64, nrm = 1.0, rng = Random.MersenneTwister(j))
        r.added > 0 && break
    end
    @test r.added > 0                                      # non-vacuous: the frames were replaced
    @test all(left_isometry_defect(fr.W[i]) < 1e-12 for i in 1:c)
    @test all(right_isometry_defect(fr.Z[k]) < 1e-12 for k in c:(L - 1))
    @test abs(norm(RC._project_root(psi, fr)) - norm(psi[psi.center])) < 1e-6
end

@testset "projection miss: direct, no sqrt(eps) floor, any gauge" begin
    L, c = 8, 4
    mpo, psi = _xx_warm(L, 2)
    fr = RC.frames_from_state(psi, mpo, c)
    normdiff(p) = sqrt(max(norm(p[p.center])^2 - norm(RC._project_root(p, fr))^2, 0.0))
    m0 = RC._projection_miss(psi, fr.W, fr.Z, c)
    @info "frames holding psi: direct miss $m0, norm difference $(normdiff(psi))"
    @test m0 < 1e-13 * norm(psi[psi.center])
    # a state the frames do not hold: an O(1) miss, where the norm difference is accurate
    _, phi = _xx_warm(L, 4)
    m = RC._projection_miss(phi, fr.W, fr.Z, c)
    @test m > 1e-4 * norm(phi[phi.center])
    @test isapprox(m, normdiff(phi); rtol = 1e-8)
    phi2 = copy(phi)
    canonical!(phi2, 2)
    @test isapprox(RC._projection_miss(phi2, fr.W, fr.Z, c), m; rtol = 1e-10)
end

@testset "parallel closure: concurrent frontiers, final j = 0 measurement" begin
    # With >= RC._FRONTIER_THREADS threads `parallel = true` measures the frontiers concurrently
    # and takes the final j = 0 measurement speculatively; below it, the same calls run in order.
    # `exact = true` draws no random sketch, so serial and parallel must agree to roundoff.
    @info "parallel closure test on $(Threads.nthreads()) threads " *
          "(concurrent frontiers from $(RC._FRONTIER_THREADS))"
    L, c = 8, 4
    mpo, psi = _xx_warm(L, 2)
    tau = ComplexF64(-0.1im)
    alloc = basis_allocation(AdaptiveBudget(1e-8, 0.0, 1.0), L - 1; tau = tau, nrm = 1.0,
                             nfront = 3)
    ckw = _kc_ckw(alloc, 64)
    frs, frp = RC.frames_from_state(psi, mpo, c), RC.frames_from_state(psi, mpo, c)
    S0 = RC._project_root(psi, frs)
    ms = RC._measure_frontiers(frs, psi, mpo, S0, tau, alloc, ckw, 3; parallel = false)
    mp = RC._measure_frontiers(frp, psi, mpo, S0, tau, alloc, ckw, 3; parallel = true)
    @test isapprox(ms.leak, mp.leak; rtol = 1e-10) && ms.met == mp.met
    @test all(>(0), ms.leak)                                  # non-vacuous
    # the measuring sweeps' bonds as concurrent tasks (`_BOND_TASKS`, off by default) vs in order
    mo, mf = try
        RC._BOND_TASKS[] = true
        (RC._measure_frontiers(RC.frames_from_state(psi, mpo, c), psi, mpo, S0, tau, alloc, ckw,
                               3; parallel = true),
         # folded probes draw one RNG per bond under the tasks: an estimate, finite and nonzero
         RC._measure_frontiers(RC.frames_from_state(psi, mpo, c), psi, mpo, S0, tau, alloc, ckw,
                               3; parallel = true, fold = true))
    finally
        RC._BOND_TASKS[] = false
    end
    @test isapprox(mo.leak, mp.leak; rtol = 1e-10) && mo.met == mp.met
    @test all(isfinite, mf.leak) && all(>(0), mf.leak)
    # the default single pass: serial == parallel, j = 0 not measured
    sps = [RC._krylov_closure!(RC.frames_from_state(psi, mpo, c), psi, mpo, S0, tau, alloc, ckw;
                               order = 3, workcap = 64, nrm = 1.0, parallel = par)
           for par in (false, true)]
    @test sps[1].converged == sps[2].converged && sps[1].passes == sps[2].passes == 1
    @test isnan(sps[1].ledger.leak[1]) && isnan(sps[2].ledger.leak[1])
    @test isapprox(sps[1].ledger.leak[2:end], sps[2].ledger.leak[2:end]; rtol = 1e-8)
    # the validation path (checks first, final j = 0)
    cls = RC._krylov_closure!(frs, psi, mpo, S0, tau, alloc, ckw; order = 3, workcap = 64,
                              nrm = 1.0, parallel = false, checks = true)
    clp = RC._krylov_closure!(frp, psi, mpo, S0, tau, alloc, ckw; order = 3, workcap = 64,
                              nrm = 1.0, parallel = true, checks = true)
    @test cls.passes == clp.passes && cls.converged == clp.converged
    @test isapprox(cls.ledger.leak, clp.ledger.leak; rtol = 1e-8)
    @test cls.passes >= 1 && frs.gen > 0                      # non-vacuous: the frames grew
    # the reported final j = 0 leak -- reused, speculative or recomputed -- is the one the FINAL
    # frames give, measured afresh
    for (fr, cl) in ((frs, cls), (frp, clp))
        fresh = RC._measure_frontiers(fr, psi, mpo, cl.S0, tau, alloc, ckw, 1)
        @test isapprox(cl.ledger.leak[1], fresh.leak[1]; rtol = 1e-10)
    end
end

end # Krylov closure
