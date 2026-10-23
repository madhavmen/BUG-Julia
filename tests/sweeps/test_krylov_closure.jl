# tests/sweeps/test_krylov_closure.jl -- UT1..UT9 of the general-order Krylov-closure plan
# (krylov_closure_cbe_bug_realtime_L16_plan.md §4), plus a multi-seed stress test.
#
#   UT1  the frontier genuinely advances: a dense hopping chain through the production loop,
#        and the single-magnon XX chain through the production MPS closure
#   UT2  matrix-free outside-space action against the exact (I - P) H v, left and right
#   UT3  order moments (dense), and the step's local error order against exp(-iH dt) (MPS)
#   UT4  a projected Krylov breakdown is not accepted
#   UT5  frame invariants after every frontier sweep: nesting, isometry, carries, environments
#   UT6  eigenstates: no growth, exact root phase
#   UT7  undersized maxdim: finite, :budget_unmet, nothing leaks into the state
#   UT8  one final compression; tol_trunc respected; cumulative weight != worst bond
#   UT9  order = 1 is the ordinary step bit for bit; refusals
#   closure_checks: off (default) = one enriching sweep per frontier; on = the validation loop
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

"""
The production closure loop on a DENSE space: `Q` holds orthonormal columns, the frontier is
`Q (tau^j/j!) (Q'AQ)^j Q'psi`, and a sweep appends the normalised outside-space action when its
time-weighted size exceeds `allow` -- the dense analogue of the MPS frontier sweep.
"""
function _kc_dense_closure(A, psi, Q0, tau, allow, order; include_zero = true,
                           maxpass = RC._closure_maxpass(order))
    Q = Ref(Matrix{ComplexF64}(Q0))
    s0() = Q[]' * psi
    applyB = x -> Q[]' * (A * (Q[] * x))
    frontier(j) = j == 0 ? s0() : RC._projected_frontiers(applyB, s0(), tau, j)[j]
    function sweep!(F, j, pass)
        Av = A * (Q[] * F)
        r = Av - Q[] * (Q[]' * Av)
        leak = abs(tau) * norm(r) / norm(psi)
        added = 0
        if leak > allow
            for _ in 1:2
                r -= Q[] * (Q[]' * r)
            end
            Q[] = hcat(Q[], r / norm(r))
            added = 1
        end
        return (added = added, leak = leak, met = true)
    end
    res = RC._closure_loop!(frontier, sweep!, order; maxpass = maxpass,
                            include_zero = include_zero)
    return Q[], res
end

_kc_unit(n, k) = (v = zeros(ComplexF64, n); v[k] = 1; v)

"The production `cbe_expand` keywords of an adaptive step, for driving the closure directly."
_kc_ckw(alloc, wcap; exact = true) =
    (dex = 0, growth = 2.0, dover = nothing, comp_ratio = 1.0, sulz_cap = false, rmax = 0,
     preselect_only = false, exact = exact, stol_pre = 0.0, stol_fnl = 1e-13, alloc = alloc,
     workcap = wcap)

"Fresh root environments through the frames' CURRENT tensors (what `fr.lenv/renv` must equal)."
function _kc_fresh_envs(mpo, fr, L)
    lenv = boundary_channels(mpo)
    for i in 1:(fr.c - 1)
        lenv = push_left_channels(lenv, mpo, fr.W[i], i)
    end
    renv = boundary_channels(mpo)
    for j in (L - 1):-1:(fr.c + 1)
        renv = push_right_channels(renv, mpo, fr.Z[j], j + 1)
    end
    return lenv, renv
end

"Is every old frame chain contained in the new one, bond by bond? Max violation of ||O||^2 = r_old."
function _kc_nesting_defect(Wold, Wnew, Zold, Znew, c, L)
    worst = 0.0
    O = nothing
    for i in 1:c
        T = O === nothing ? Wold[i] : to_concrete(contract(O, (2,), Wold[i], (1,)))
        O = to_concrete(contract(Wnew[i]', (1, 2), T, (1, 2)))           # (new_i, old_i)
        worst = max(worst, abs(norm(O)^2 - leg_dim(Wold[i], 3)))
    end
    O = nothing
    for j in (L - 1):-1:c
        T = O === nothing ? Zold[j] : to_concrete(contract(Zold[j], (3,), O, (1,)))
        O = to_concrete(contract(T, (2, 3), Znew[j]', (2, 3)))            # (old_j, new_j)
        worst = max(worst, abs(norm(O)^2 - leg_dim(Zold[j], 1)))
    end
    return worst
end

_kc_magnon(L, k) = product_state([i == k ? :up : :down for i in 1:L])

@testset "Krylov closure (general order)" begin

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

@testset "UT6: eigenstates -- no growth, exact root phase" begin
    set_symmetry!(:U1)
    L, dt = 8, 0.3
    for delta in (0.0, 1.0), checks in (false, true)  # H psi = 0, then H psi = E psi, E != 0
        mpo = xxz_mpo(L; J = 1.0, delta = delta)
        E = delta * (L - 1) / 4
        psi = product_state(fill(:up, L))
        v0 = copy(psi)
        info = cbe_bug_step!(psi, mpo, ComplexF64(-1im * dt); order = 4, tol_basis = 1e-10,
                             tol_trunc = 1e-12, tol_expv = 1e-13, closure_checks = checks)
        lg = info.ledger
        @test bond_dims(psi) == ones(Int, L - 1)
        @test lg.n_selected == 0 && lg.n_krylov == 0
        @test all(l.added == 0 for l in lg.closure_log)
        if checks
            # checked first and already closed: no closure sweep runs at all
            @test lg.n_passes == 0
            @test lg.leak_initial == lg.leak && length(lg.rho_initial) == 4
        else
            # one sweep per frontier j = 1..3, each finding nothing; no measurement
            @test lg.n_passes == 1 && length(lg.closure_log) == 3
            @test isempty(lg.leak_initial) && isnan(lg.leak[1])
        end
        @test budget_status(lg) === :met
        @test abs(overlap(v0, psi) - exp(-1im * dt * E)) < 1e-10
        @test lg.defect < 1e-12
    end
end

@testset "UT7: undersized maxdim terminates and reports" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = neel_state(L)
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.5im); order = 4, tol_basis = 1e-10,
                         tol_trunc = 1e-12, maxdim = 2)
    lg = info.ledger
    @test budget_status(lg) === :budget_unmet
    @test occursin("budget unmet", budget_diagnostic(lg))
    @test lg.workcap_reached
    @test maximum(bond_dims(psi)) <= 2                       # nothing past maxdim survives
    @test lg.r_aug <= work_ceiling(2, 1)                     # frames never passed the ceiling
    @test lg.n_passes <= RC._closure_maxpass(4)
    @test norm(psi) <= 1 + 1e-10
    @info "UT7: unmet $(lg.unmet), passes $(lg.n_passes), r_aug $(lg.r_aug), " *
          "diagnostic: $(budget_diagnostic(lg))"
    # the loop's own bound: a sweep that always grows is stopped and reported
    res = RC._closure_loop!(j -> nothing, (F, j, pass) -> (added = 1, leak = 1.0, met = true),
                            3; maxpass = 4)
    @test res.exhausted && !res.converged && res.passes == 4
    # ...and the working-rank limit (plan step 3): a pass that grows but hits the ceiling with a
    # share unmet stops at once, not converged and not exhausted
    res = RC._closure_loop!(j -> nothing,
                            (F, j, pass) -> (added = 1, leak = 1.0, met = false, capped = true),
                            3; maxpass = 4)
    @test res.capped && !res.converged && !res.exhausted && res.passes == 1
    # control: the same sweep without the ceiling runs to the pass bound
    res = RC._closure_loop!(j -> nothing,
                            (F, j, pass) -> (added = 1, leak = 1.0, met = false, capped = false),
                            3; maxpass = 4)
    @test !res.capped && res.exhausted && res.passes == 4
    # a capped sweep that meets its share is not stopped early
    res = RC._closure_loop!(j -> nothing, (F, j, pass) -> (added = pass == 1 ? 1 : 0, leak = 0.0,
                                                           met = true, capped = true),
                            3; maxpass = 4)
    @test res.converged && !res.capped && res.passes == 2
    # the step reports the stop: UT7's undersized step stopped at the ceiling, not at the bound
    @test lg.n_passes < RC._closure_maxpass(4)
end

@testset "UT8: one root_out compression, tol_trunc, cumulative vs worst" begin
    L = 10
    mpo, psi = _xx_warm(L, 3)
    tol = 1e-3
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.1im); order = 3, tol_basis = 1e-8,
                         tol_trunc = tol, maxdim = 64)
    lg = info.ledger
    @test lg.n_compress == 1
    @test lg.discarded <= tol * (1 + 1e-8)
    # per-bond weights (one value per bond, whichever sweep logged it) rebuild the estimate
    d = Dict{Int, Float64}()
    for e in lg.bond_log
        d[e.bond] = e.discarded
    end
    c = info.root
    left = sqrt(sum(d[b]^2 for b in 1:c; init = 0.0))
    right = sqrt(sum(d[b]^2 for b in (c + 1):(L - 1); init = 0.0))
    @test isapprox(left + right, lg.discarded; rtol = 1e-10)
    @test count(>(0), values(d)) >= 2                       # non-vacuous: several bonds cut
    @test lg.discarded > info.discarded                     # cumulative, not the worst bond
    @test budget_status(lg) === :met
end

@testset "UT9: order = 1 is the ordinary step; refusals" begin
    L = 8
    mpo, psi0 = _xx_warm(L, 2)
    dt = ComplexF64(-0.05im)
    a, b = copy(psi0), copy(psi0)
    ia = cbe_bug_step!(a, mpo, dt; tol_basis = 1e-6, rng = Random.MersenneTwister(7))
    ib = cbe_bug_step!(b, mpo, dt; tol_basis = 1e-6, order = 1,
                       rng = Random.MersenneTwister(7))
    @test all(norm(to_concrete(a[i] - b[i])) == 0.0 for i in 1:L)
    @test ia.krylov_dims == ib.krylov_dims
    @test ib.ledger.n_passes == 0 && isempty(ib.ledger.leak) && ib.ledger.order == 1
    # `closure_probe` only measures: same state bit for bit, and the numbers are what an
    # order-3 closure sees on the same half-sweep frames before it decides to enrich
    pc = copy(psi0)
    ic = cbe_bug_step!(pc, mpo, dt; tol_basis = 1e-6, closure_probe = 3,
                       rng = Random.MersenneTwister(7))
    @test all(norm(to_concrete(a[i] - pc[i])) == 0.0 for i in 1:L)
    @test ic.ledger.n_passes == 0 && isempty(ic.ledger.leak) && length(ic.ledger.rho_initial) == 3
    id = cbe_bug_step!(copy(psi0), mpo, dt; tol_basis = 1e-6, order = 3, closure_checks = true,
                       rng = Random.MersenneTwister(7))
    @test isapprox(ic.ledger.leak_initial, id.ledger.leak_initial; rtol = 1e-12)
    @test isapprox(ic.ledger.rho_initial, id.ledger.rho_initial; rtol = 1e-12)
    @info "UT9 probe: rho_initial $(ic.ledger.rho_initial), leak_initial " *
          "$(ic.ledger.leak_initial); order 3 passes $(id.ledger.n_passes)"
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, dt; closure_probe = 2,
                                             krylov_basis = 3)
    am, bm = copy(psi0), copy(psi0)
    cbe_bug_midpoint_step!(am, mpo, dt; tol_basis = 1e-6, rng = Random.MersenneTwister(7))
    cbe_bug_midpoint_step!(bm, mpo, dt; tol_basis = 1e-6, order = 1,
                           rng = Random.MersenneTwister(7))
    @test all(norm(to_concrete(am[i] - bm[i])) == 0.0 for i in 1:L)
    # the unified Test A/B allocation is unchanged at order 1
    bud = AdaptiveBudget(1e-6, 1e-7, 1e-10)
    @test basis_allocation(bud, L - 1; tau = dt, nrm = 1.0) ===
          basis_allocation(bud, L - 1; tau = dt, nrm = 1.0, nfront = 1)
    @test basis_allocation(bud, L - 1; nfront = 4).per_bond ≈
          basis_allocation(bud, L - 1).per_bond / 2
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, dt; order = 0)
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, dt; order = 2, krylov_basis = 3)
    @test_throws ArgumentError cbe_bug_midpoint_step!(copy(psi0), mpo, dt; order = 2)
end

@testset "closure_checks: off = one enriching sweep per frontier, on = the validation loop" begin
    # USER 2026-10-17: an order-2 step is the two half-sweeps plus ONE left || right enriching
    # sweep of j = 1 -- no check first, no re-check, no final j = 0. The checks stay as a switch.
    # A case where the half-sweep frames do NOT already close j = 1 (the warm XX state at dt 0.1
    # does, job 17332175: then neither path adds anything and the comparison is vacuous).
    L = 8
    mpo, warm = _xx_warm(L, 2)
    stepc(psi, dt, p, checks) = cbe_bug_step!(copy(psi), mpo, ComplexF64(-1im * dt); order = p,
                                            tol_basis = 1e-10, parallel = true,
                                            closure_checks = checks,
                                            rng = Random.MersenneTwister(3))
    for p in (2, 3)
        case = nothing
        for psi0 in (warm, domain_wall_state(L)), dt in (0.1, 0.3, 0.6)
            on = stepc(psi0, dt, p, true)
            if sum(e.added for e in on.ledger.closure_log; init = 0) > 0
                case = (psi0 = psi0, dt = dt, on = on)
                break
            end
        end
        @test case !== nothing                                   # non-vacuous: a case grows
        case === nothing && continue
        off = stepc(case.psi0, case.dt, p, false)
        lo, ln = off.ledger, case.on.ledger
        @test [(e.frontier, e.enrich) for e in lo.closure_log] == [(j, true) for j in 1:(p - 1)]
        @test lo.n_passes == 1 && isempty(lo.leak_initial) && isnan(lo.leak[1])
        @test sum(e.added for e in lo.closure_log) > 0          # non-vacuous: j = 1 enriched
        @test !(:closure in lo.unmet)                         # each enriching sweep met A-D
        # the checks measure first (enrich = false) and end on a measurement of j = 0
        @test !ln.closure_log[1].enrich && length(ln.closure_log) > length(lo.closure_log)
        @test ln.closure_log[end].frontier == 0 && !ln.closure_log[end].enrich
        @info "closure_checks p=$p (dt $(case.dt)): off $(length(lo.closure_log)) sweeps adding " *
              "$(sum(e.added for e in lo.closure_log)), on $(length(ln.closure_log)) sweeps " *
              "(passes $(ln.n_passes)); off: $(budget_diagnostic(lo))"
    end
end

@testset "stress: several RSVD seeds, order 2 and 3, against exp(-iHt)" begin
    L = 10
    mpo, psi0 = _xx_warm(L, 2)
    Hd = dense_heisenberg(L; J = 1.0, delta = 0.0)
    nsteps, dt = 4, 0.1
    vref = exp(-1im * nsteps * dt * Hd) * dense_state(psi0)
    for p in (2, 3)
        errs = Float64[]
        for arm in (:exact, 1, 2, 3)
            psi = copy(psi0)
            met = true
            for _ in 1:nsteps
                info = cbe_bug_step!(psi, mpo, ComplexF64(-1im * dt); order = p,
                                     tol_basis = 1e-7, parallel = true, exact = arm === :exact,
                                     rng = Random.MersenneTwister(arm === :exact ? 0 : arm))
                met &= budget_status(info.ledger) === :met
            end
            @test met
            push!(errs, norm(dense_state(psi) - vref))
        end
        @info "stress p=$p: errors exact/seed1/seed2/seed3 = $(errs)"
        @test maximum(errs) < 1e-3
        @test maximum(abs.(errs[2:end] .- errs[1])) < 1e-5     # sampling is not luck
    end
end

# A nearly saturated frame: the complete local basis of one site tensor of an evolved L=8 chain minus
# ONE column of its largest sector, with 1e-11 orthogonality loss (an accumulated-frame level).
# `side = :left` uses the centre tensor `(link, site, bond)`, `:right` the next one `(bond, site,
# link)`. Returns the frame, the source tensor, the local sector dims, the unsaturated sector, rng.
function saturated_frame(side::Symbol)
    set_symmetry!(:U1)
    L = 8
    psi = neel_state(L)
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    for _ in 1:3
        cbe_bug_step!(psi, mpo, ComplexF64(-0.3); exact = true, tol_basis = 1e-10,
                      tol_trunc = 1e-12, tol_expv = 1e-12, maxdim = 64)
    end
    canonical!(psi, side === :left ? L : 1)
    A = psi[side === :left ? L ÷ 2 : L ÷ 2 + 1]
    F = RC.full_local_basis(A, side)
    fl = side === :left ? 3 : 1
    dims = Dict(q => d for (q, d) in F.spaces[fl])
    qbig = argmax(q -> dims[q], collect(keys(dims)))
    Q = RC._trim_per_sector(F, fl, Dict(q => d - (q == qbig ? 1 : 0) for (q, d) in dims))
    rng = Random.MersenneTwister(11)
    Q = to_concrete(Q + 1e-11 * RC._randomize_like(Q, rng))
    return Q, A, dims, qbig, rng
end

@testset "_extend_frame never exceeds the local space (saturated sector + roundoff)" begin
    # s38test 17341708: a nearly saturated frame (253 of 256) gained 72 > 70 directions in one
    # sector -- roundoff cleared Telum's GLOBALLY relative cutoff because another sector carried a
    # real direction. Reproduced on `saturated_frame`, with a residual with one genuine direction
    # (the unsaturated sector) and roundoff everywhere else.
    Q, A, dims, _, rng = saturated_frame(:left)
    v = RC._randomize_like(A, rng)
    v = to_concrete((1.0 / norm(v)) * v)
    R = to_concrete(RC.perp_component(Q, v))
    Qn, nadd, _ = RC._extend_frame(Q, R, v, :left, 1.0, 0.0)
    nloc = sum(values(dims))
    @info "saturated-frame extension: local $nloc, frame $(RC.leg_dim(Q, 3)) -> $(RC.leg_dim(Qn, 3)) (+$nadd)"
    @test nadd >= 1                                                     # the genuine one is kept
    @test RC.leg_dim(Qn, 3) <= nloc
    RC._frame_sector_dims(Qn, :left, RC.fusion_basis(Qn, 1, 2; tag = "chk"), "test")  # no throw
end

@testset "cbe_expand's appended block never exceeds the local space (_guarded_block)" begin
    # d50f_trap 17346775 / d50f_rad3 17346774 (IT_FRAME_CHECKS, IT-1 L=20 domain wall): the first
    # over-wide frame was `cbe_expand V_ex` (10 > 9) and `U_ex` (253 > 252), whose re-orthogonalised
    # block had no local-room cap. A candidate block spanning the whole local space is appended to
    # `saturated_frame`: only the one free direction may survive.
    for side in (:left, :right)
        Q, _, dims, _, rng = saturated_frame(side)
        fl = side === :left ? 3 : 1
        C = RC._randomize_like(Q, rng)
        # the control: re-orthogonalisation alone turns the saturated sectors' roundoff into columns
        bare = side === :left ? RC._reortho_left(Q, C) : RC._reortho_right(Q, C)
        @test RC.leg_dim(bare, fl) > 1
        U = RC._guarded_block(Q, C, side)
        @test U !== nothing
        @test RC.leg_dim(U, fl) == 1
        # width == the local dimension and no sector over its own: the one column is the free one
        Qn = to_concrete(oplus([Q, to_concrete(setitag(U, fl, Q.inds[fl].itags))], (fl,)))
        @test RC.leg_dim(Qn, fl) == sum(values(dims))
        a, b = side === :left ? (1, 2) : (2, 3)
        RC._frame_sector_dims(Qn, side, RC.fusion_basis(Qn, a, b; tag = "chk"), "test")  # no throw
    end
end

@testset "Telum's dense SVD falls back from gesdd (_robust_svd)" begin
    # d50g16 17346776 / d50g20 17346777: gesdd threw LAPACKException(1) in `_frame_onto_left` on a
    # FINITE exact isometry (every sigma 1); the vendored Telum now retries with QR iteration.
    @test isdefined(Telum, :_robust_svd)
    M = randn(Random.MersenneTwister(3), ComplexF64, 7, 5)
    @test Telum._robust_svd(M).S ≈ svd(M).S
end

end # Krylov closure
