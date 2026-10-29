# tests/sweeps/test_qtt_task2.jl -- qtt_dissipative_cbe_bug_four_task_implementation_plan.md,
# Task 2 gates V01-V08 at unit-test sizes (the order/tolerance/stability studies at plan sizes are
# benchmarks/qtt/task2_study.jl).
#
#   T2.0  dense contractions: frame projection, reduced field, reduced real-linear JVP; the dense
#         TDVP2 (lagged) == the library tdvp2_step! on the same MPO, backward norms included
#   V01   linear GP: the nonlinear stage solver reproduces the dense trapezoid / Radau IIA maps
#         exactly in full frames; BUG p = 2/3/4 converge to expm(h G_grid); analytic modes recorded
#   V02   nonlinear dissipative plane wave: exact discrete rho(t), phi(t); stage-consistent p = 2/3/4
#         orders; frozen density loses order, also on a spatially modulated state (dense RK4 ref)
#   V03   Heston: constant mode exp(-r tau) exact, affine profile vs expm and Riccati, forward
#         adjoint conserves mass and stays positive, call price bounds
#   V07   TDVP2: backward centre substeps amplify (A_max > 1) while the full step contracts;
#         lagged != consistent for g > 0
#   V08   A- vs L-stable: trapezoid does not damp a stiff mode, Radau IIA does

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

"Dense matrix of left-isometry cores (rows: sites, site 1 fastest) / right-isometry cores."
function _wmat(W)
    M = ones(ComplexF64, 1, 1)
    for c in W
        M = reshape(M * reshape(c, size(c, 1), :), :, size(c, 3))
    end
    return M
end
function _zmat(Z)
    M = ones(ComplexF64, 1, 1)
    for c in reverse(Z)
        M = reshape(reshape(c, :, size(c, 3)) * M, size(c, 1), :)
    end
    return M
end

"Slope of log(err) against log(h) between consecutive step sizes."
_slopes(hs, es) = [log(es[k] / es[k + 1]) / log(hs[k] / hs[k + 1]) for k in 1:(length(hs) - 1)]

"Run `n` steps of `f!(psi, t, h)` from `A0`; return the final dense vector."
function _run(f!, A0, h, n)
    psi = qtt_mps(A0)
    for k in 0:(n - 1)
        f!(psi, k * h, h)
    end
    return vec(tt_dense(qtt_cores(psi)))
end

@testset "QTT plan Task 2" begin

@testset "T2.0: dense contractions and the dense TDVP2" begin
    g = _heston_grid(3, 3)
    L = 6
    rng = Random.MersenneTwister(5)
    WA = qtt_cores(qtt_mps(_random_tt(_tt_ranks(L, 4); rng = rng); center = L))
    ZB = qtt_cores(qtt_mps(_random_tt(_tt_ranks(L, 4); rng = rng); center = 1))
    c = 3
    fr = QTTFrames(WA[1:c], ZB[(c + 1):L])
    Wm, Zm = _wmat(fr.W), _zmat(fr.Z)
    @test norm(Wm' * Wm - I) < 1e-12 && norm(Zm * Zm' - I) < 1e-12
    Y = randn(rng, ComplexF64, size(Wm, 2), size(Zm, 1))
    @test _relerr(vec(tt_dense(frame_state(fr, Y))), vec(Wm * Y * Zm)) < 1e-12
    H = heston_op(g, _HP)
    C = _random_tt(_tt_ranks(L, 3); rng = rng)
    ref = Wm' * reshape(op_dense(H) * vec(tt_dense(C)), size(Wm, 1), :) * Zm'
    @test _relerr(frame_project(fr, H, C), ref) < 1e-11
    # the reduced GP field and its real-linear Jacobian against finite differences
    gp = qtt_grid(3, 3; x = (-2.0, 2.0), y = (-2.0, 2.0))
    m = gp_model(gp; gnl = 2.0, gamma = 0.2, mu = 0.3, omega = 1.0, rho_tol = 1e-13)
    fl = GPFlow(m)
    G0, mu, _ = reduced_field(fl, fr, Y, 0.0)
    v = vec(Wm * Y * Zm)
    Hl = op_dense(gp_linear_op(m, 0.0))
    Fv = fl.pref .* (Hl * v .+ m.g .* abs2.(v) .* v)
    @test _relerr(G0, Wm' * reshape(Fv, size(Wm, 1), :) * Zm') < 1e-10
    V = randn(rng, ComplexF64, size(Y))
    J = reduced_jacobian(fl, fr, Y, 0.0, mu)
    e = 1e-6
    fd = (first(reduced_field(fl, fr, Y .+ e .* V, 0.0)) .-
          first(reduced_field(fl, fr, Y .- e .* V, 0.0))) ./ (2e)
    @test _relerr(J(V), fd) < 1e-6
    # dense TDVP2 lagged == the library tdvp2_step! (linear generator, complex tau)
    lin = GPFlow(gp_model(gp; gnl = 0.0, gamma = 0.3, mu = 0.0, omega = 1.0))
    A0 = _random_tt(_tt_ranks(L, 4); rng = rng)
    p1 = qtt_mps(A0)
    r = qtt_tdvp2_step!(p1, lin, 0.0, 0.05; maxdim = 64, tol_trunc = 1e-14, tol_krylov = 1e-13)
    p2 = qtt_mps(A0)
    bn = Float64[]
    tdvp2_step!(p2, qtt_to_mpo(lin.m.H0), ComplexF64(lin.pref * 0.05); maxdim = 64,
                trunc_thresh = 1e-14, hermitian = false, tol = 1e-14, backward_norms = bn)
    @test _relerr(vec(tt_dense(qtt_cores(p1))), vec(tt_dense(qtt_cores(p2)))) < 1e-8
    @test length(bn) == length(r.A_back) == 2 * (L - 2)
    @test maximum(abs.(bn .- r.A_back)) < 1e-6
end

@testset "V01: linear GP -- stage solver exact in full frames, BUG -> expm" begin
    g = qtt_grid(3, 3; x = (-5.0, 5.0), y = (-5.0, 5.0))
    m = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.0, omega = 1.0)
    Gd = (-1im - m.gamma) .* op_dense(m.H0)
    phi0(x) = pi^(-0.25) * exp(-x^2 / 2)
    phi1(x) = sqrt(2) * x * phi0(x)
    A0 = tt_lincomb([(1.0, qtt_separable(g, phi0, phi0)), (0.25, qtt_separable(g, phi1, phi0)),
                     (0.15im, qtt_separable(g, phi0, phi1))])
    v0 = vec(tt_dense(A0))
    h = 0.1
    Rz = Dict(:trapezoid => z -> (1 + z / 2) / (1 - z / 2),
              :radau2 => z -> (1 + z / 3) / (1 - 2z / 3 + z^2 / 6),
              :radau3 => z -> (1 + 2z / 5 + z^2 / 20) / (1 - 3z / 5 + 3z^2 / 20 - z^3 / 60))
    F = eigen(h .* Gd)
    # COMPLETE frames (full-rank isometries), so the Galerkin stage solve is the full RK map
    rng = Random.MersenneTwister(13)
    WA = qtt_cores(qtt_mps(_random_tt(_tt_ranks(6, 64); rng = rng); center = 6))
    ZB = qtt_cores(qtt_mps(_random_tt(_tt_ranks(6, 64); rng = rng); center = 1))
    frc = QTTFrames(WA[1:3], ZB[4:6])
    @test size(frc.W[3], 3) == 8 && size(frc.Z[1], 1) == 8
    S0 = frame_project(frc, nothing, A0)
    for meth in (:trapezoid, :radau2, :radau3)
        Rd = F.vectors * Diagonal(Rz[meth].(F.values)) / F.vectors
        Ys, st = rk_stage_solve(GPFlow(m), frc, S0, 0.0, h, meth; tol = 1e-13)
        @test st.converged && st.newton <= 2                  # linear: Newton exact in one pass
        @test _relerr(vec(tt_dense(frame_state(frc, Ys[end]))), Rd * v0) < 1e-10
    end
    # BUG p = 2/3/4 against the exact discrete flow, two step sizes: the error falls
    T = 0.4
    vT = exp(T .* Gd) * v0
    for p in (2, 3, 4)
        es = Float64[]
        for hh in (0.1, 0.05)
            v = _run((ps, t, hs) -> qtt_bug_step!(ps, GPFlow(m), t, hs; order = p, maxdim = 64,
                                                  tol_basis = 1e-10, tol_trunc = 1e-12), A0, hh,
                     round(Int, T / hh))
            push!(es, norm(v - vT) / norm(vT))
        end
        println("V01 p=$p errors $(es) slope $(_slopes([0.1, 0.05], es))")
        @test es[2] < es[1]
        @test es[2] < 1e-3
    end
    # the analytic R^2 modes: spatial + boundary error of this coarse box, recorded only
    xs, ys = qtt_coords(g)
    E(mn) = mn[1] + mn[2] + 1
    an = [(phi0(x) * phi0(y) * exp((-1im - 0.1) * E((0, 0)) * T) +
           0.25 * phi1(x) * phi0(y) * exp((-1im - 0.1) * E((1, 0)) * T) +
           0.15im * phi0(x) * phi1(y) * exp((-1im - 0.1) * E((0, 1)) * T)) for x in xs, y in ys]
    println("V01 discrete-vs-analytic (spatial) error ",
            _relerr(qtt_values(tt_from_dense(reshape(vT, ntuple(_ -> 2, 6))), g), an))
end

@testset "V02: nonlinear dissipative plane wave and a modulated state" begin
    n = 3
    g = qtt_grid(n, n; x = (0.0, 2pi), y = (0.0, 2pi))
    gam, gn, mu0, rho0 = 0.5, 1.0, 0.0, 0.8
    m = gp_model(g; gnl = gn, gamma = gam, mu = mu0, omega = 0.0, rho_tol = 1e-14)
    kx, ky = 1, 1
    keff2(k, d) = (2 - 2cos(k * d)) / d^2
    alpha = (keff2(kx, g.dx) + keff2(ky, g.dy)) / 2 - mu0          # the DISCRETE alpha
    rhot(t) = rho0 * exp(-2gam * alpha * t) / (1 + gn * rho0 / alpha * (1 - exp(-2gam * alpha * t)))
    phit(t) = log(rhot(t) / rho0) / (2gam)
    A0 = qtt_separable(g, x -> sqrt(rho0) * exp(1im * kx * x), y -> exp(1im * ky * y))
    exact(t) = qtt_separable(g, x -> sqrt(rhot(t)) * exp(1im * (kx * x + phit(t))),
                             y -> exp(1im * ky * y))
    T = 0.4
    vT = vec(tt_dense(exact(T)))
    hs = [0.1, 0.05, 0.025]
    for (p, need) in ((2, 1.8), (3, 2.7), (4, 3.6))
        es = [norm(_run((ps, t, hh) -> qtt_bug_step!(ps, GPFlow(m), t, hh; order = p,
                                                     tol_trunc = 1e-14, tol_stage = 1e-13),
                        A0, hh, round(Int, T / hh)) - vT) / norm(vT) for hh in hs]
        sl = _slopes(hs, es)
        println("V02 plane wave p=$p errors $es slopes $sl")
        @test minimum(sl) > need || maximum(es) < 1e-11
    end
    ef = [norm(_run((ps, t, hh) -> gp_bug_step!(ps, m, t, hh; tol_basis = 1e-10, tol_trunc = 1e-14),
                    A0, hh, round(Int, T / hh)) - vT) / norm(vT) for hh in hs]
    println("V02 plane wave frozen-density errors $ef slopes $(_slopes(hs, ef))")
    @test maximum(_slopes(hs, ef)) < 1.5                         # the lag costs the order
    # a spatially modulated nonlinear state against dense RK4 of the same semi-discrete system
    A1 = tt_lincomb([(1.0, qtt_separable(g, x -> 1 + 0.3cos(x), y -> exp(1im * y))),
                     (0.2, qtt_separable(g, x -> exp(2im * x), y -> cos(y)))])
    Hl = op_dense(gp_linear_op(m, 0.0))
    Fd(v) = (-1im - gam) .* (Hl * v .+ gn .* abs2.(v) .* v)
    v = vec(tt_dense(A1))
    dt = 1e-4
    for _ in 1:round(Int, T / dt)
        k1 = Fd(v); k2 = Fd(v .+ dt / 2 .* k1); k3 = Fd(v .+ dt / 2 .* k2); k4 = Fd(v .+ dt .* k3)
        v = v .+ dt / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    end
    es = [norm(_run((ps, t, hh) -> qtt_bug_step!(ps, GPFlow(m), t, hh; order = 2, maxdim = 64,
                                                 tol_basis = 1e-11, tol_trunc = 1e-13),
                    A1, hh, round(Int, T / hh)) - v) / norm(v) for hh in hs]
    efz = [norm(_run((ps, t, hh) -> gp_bug_step!(ps, m, t, hh; tol_basis = 1e-11, tol_trunc = 1e-13,
                                                 maxdim = 64), A1, hh, round(Int, T / hh)) - v) /
           norm(v) for hh in hs]
    println("V02 modulated consistent $es ($(_slopes(hs, es))) frozen $efz ($(_slopes(hs, efz)))")
    @test minimum(_slopes(hs, es)) > 1.7
    @test maximum(_slopes(hs, efz)) < 1.5
    @test es[end] < efz[end]
end

@testset "V03: Heston exact modes, adjoint mass, call bounds" begin
    g = _heston_grid(3, 3)
    p = _HP
    A = heston_op(g, p)
    fl = LinearFlow(A)
    tau, h = 0.2, 0.05
    one_ = qtt_separable(g, x -> 1.0, v -> 1.0)
    u = _run((ps, t, hh) -> qtt_bug_step!(ps, fl, t, hh; order = 2, maxdim = 32,
                                          tol_trunc = 1e-13), one_, h, 4)
    # the constant mode is an exact eigenvector, so only the trapezoid's own O(h^3) remains
    ztr = -p.r * h
    @test maximum(abs.(u .- ((1 + ztr / 2) / (1 - ztr / 2))^4)) < 1e-12
    @test maximum(abs.(u .- exp(-p.r * tau))) < 1e-8
    a, b = 0.7, -0.5
    B, C = heston_riccati(p, a, b, tau)
    u0 = heston_affine_field(g, a, b, 0.0)
    Ad = op_dense(A)
    uex = exp(tau .* Ad) * vec(tt_dense(u0))
    for pp in (2, 3, 4)
        ub = _run((ps, t, hh) -> qtt_bug_step!(ps, fl, t, hh; order = pp, maxdim = 32,
                                               tol_basis = 1e-10, tol_trunc = 1e-12), u0, h, 4)
        println("V03 affine p=$pp BUG vs expm ", _relerr(ub, uex))
        @test _relerr(ub, uex) < 1e-3
    end
    cont = vec(tt_dense(heston_affine_field(g, a, B, C)))
    println("V03 discrete expm vs exact Riccati (spatial + boundary scheme) ", _relerr(uex, cont))
    @test _relerr(uex, cont) < 0.2
    # forward density: mass is conserved by the Galerkin step only when the ones-state lies in the
    # frames (`keep`, as |I>> made the MPDO step trace-preserving); positivity
    fa = LinearFlow(heston_op(g, p; adjoint = true))
    p0 = qtt_function(g, (x, v) -> exp(-(x / 0.4)^2 - ((v - 0.25) / 0.1)^2))
    m0 = heston_mass(p0, g)
    ones_state = [qtt_mps(one_)]
    pT = _run((ps, t, hh) -> qtt_bug_step!(ps, fa, t, hh; order = 3, maxdim = 32,
                                           tol_trunc = 1e-13, keep = ones_state), p0, 0.01, 5)
    @test abs(qtt_cell(g) * sum(real, pT) - m0) < 1e-9 * m0
    pN = _run((ps, t, hh) -> qtt_bug_step!(ps, fa, t, hh; order = 3, maxdim = 32,
                                           tol_trunc = 1e-13), p0, 0.01, 5)
    println("V03 mass drift without the ones-state kept: ",
            abs(qtt_cell(g) * sum(real, pN) - m0) / m0)
    @test minimum(real, pT) > -1e-3 * maximum(real, pT)
    # the semi-analytic call: positive, decreasing in K, within no-arbitrage bounds
    cs = [heston_call(p, 1.0, K, 0.09, 0.5) for K in (0.8, 1.0, 1.2)]
    @test issorted(cs; rev = true) && all(>(0), cs)
    for (K, c) in zip((0.8, 1.0, 1.2), cs)
        @test max(exp(-p.q * 0.5) - K * exp(-p.r * 0.5), 0) - 1e-6 <= c <= exp(-p.q * 0.5) + 1e-6
    end
end

@testset "V07: TDVP2 backward amplification; lagged vs consistent" begin
    g = qtt_grid(3, 3; x = (-4.0, 4.0), y = (-4.0, 4.0))
    m = gp_model(g; gnl = 0.0, gamma = 1.0, mu = 0.0, omega = 1.0)
    A0 = _random_tt(_tt_ranks(6, 4); rng = Random.MersenneTwister(71))
    psi = qtt_mps(A0)
    n0 = norm(psi)
    r = qtt_tdvp2_step!(psi, GPFlow(m), 0.0, 0.2; maxdim = 64)
    @test r.A_max > 1.0                              # the reversed substep amplifies ...
    @test norm(psi) < n0                             # ... while the full step contracts
    mn = gp_model(g; gnl = 3.0, gamma = 0.1, mu = 0.0, omega = 1.0)
    pl, pc = qtt_mps(A0), qtt_mps(A0)
    rl = qtt_tdvp2_step!(pl, GPFlow(mn), 0.0, 0.1; density = :lagged, maxdim = 64)
    rc = qtt_tdvp2_step!(pc, GPFlow(mn), 0.0, 0.1; density = :consistent, maxdim = 64)
    @test all(==(0), rl.sc_iters) && maximum(rc.sc_iters) >= 2
    @test _relerr(vec(tt_dense(qtt_cores(pl))), vec(tt_dense(qtt_cores(pc)))) > 1e-8
    # the split-dissipation control has no amplifying substep in its dissipative part
    ps = qtt_mps(A0)
    rs = qtt_split_tdvp_step!(ps, m, 0.0, 0.2; maxdim = 64)
    @test rs.A_max < 1 + 1e-8                        # conservative TDVP half: norm-preserving
    @test norm(ps) < n0
end

@testset "V08: A-stable trapezoid vs L-stable Radau IIA on a stiff mode" begin
    @test abs(root_stability(:trapezoid, -1e6)) > 0.99
    @test abs(root_stability(:radau2, -1e6)) < 1e-5
    @test abs(root_stability(:radau3, -1e6)) < 1e-5
    g = qtt_grid(3, 3; x = (0.0, 1.0), y = (0.0, 1.0))
    K = qtt_op(g, [(1.0, qtt_d(g, :x, 2; bc = :periodic), nothing),
                   (1.0, nothing, qtt_d(g, :y, 2; bc = :periodic))])
    # the checkerboard is an eigenmode of the discrete Laplacian, lambda = -8/dx^2 - ... rank 1
    A0 = qtt_separable(g, x -> cos(pi * round(Int, x / g.dx)), y -> cos(pi * round(Int, y / g.dy)))
    lam = -4 / g.dx^2 - 4 / g.dy^2
    h = 1e3 / abs(lam)
    amp = Dict{Symbol, Float64}()
    for (p, meth) in ((2, :trapezoid), (3, :radau2), (4, :radau3))
        psi = qtt_mps(A0)
        qtt_bug_step!(psi, LinearFlow(K), 0.0, h; order = p, maxdim = 16, tol_trunc = 1e-13)
        amp[meth] = norm(psi) / tt_norm(A0)
    end
    println("V08 stiff-mode amplification ", amp)
    @test amp[:trapezoid] > 0.99
    @test amp[:radau2] < 1e-2 && amp[:radau3] < 1e-2
end

end
