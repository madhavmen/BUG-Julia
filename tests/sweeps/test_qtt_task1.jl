# tests/sweeps/test_qtt_task1.jl -- qtt_dissipative_cbe_bug_four_task_implementation_plan.md,
# Task 1 release gates U01-U06 (U07 is the source audit docs/qtt_task1_audit.md, U08 the
# first-impressions study benchmarks/qtt/first_impressions.jl).
#
#   U01 indexing: grouped/interleaved, both endians, every bit pattern, round trips, weights, gauges
#   U02 operators: every QTT MPO against row-by-row sparse stencils on 8x8 and 16x16, adjoint,
#       Heston mass conservation, and the Telum MPO product (the stored-transpose convention)
#   U03 environments: the two-site environment action against the dense projected action, real and
#       complex, Hermitian and not; cache invalidation on a changed state or paddle time
#   U04 DD: streamed P_L X P_R Om and its mirror against the explicit local matrices; D1/D3 true
#       residuals and false accepts over 30 seeds; D3's exhaustive and mirrored branches exact
#   U05 rank: a separable flow keeps the x|y bond at rank 1, a correlated one grows it; tightening
#       tol_basis lowers the error against the dense exponential
#   U06 nonlinear: |psi|^2 and F(psi) against dense, g = 0 is the linear path, the density is rebuilt
#       per trial state, the real-linear JVP against finite differences, frozen != refreshed

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dd_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
RC.clear_fusion_cache!()

const _ORDERS = ((:grouped, :lsb_first), (:grouped, :msb_first), (:interleaved, :lsb_first))

@testset "QTT plan Task 1" begin

@testset "U01: indexing, round trips, weights ($o, $e)" for (o, e) in _ORDERS
    for (nx, ny) in ((2, 2), (3, 2), (3, 4), (4, 4))
        g = qtt_grid(nx, ny; x = (-1.0, 1.0), y = (0.0, 2.0), order = o, endian = e)
        L = qtt_length(g)
        @test sort(vcat(g.sites_x, g.sites_y)) == collect(1:L)
        # every bit pattern maps to a distinct grid point and back
        seen = Set{Tuple{Int, Int}}()
        for ix in 0:(2^nx - 1), iy in 0:(2^ny - 1)
            b = qtt_bits(g, ix, iy)
            @test qtt_point(g, b) == (ix, iy)
            push!(seen, (ix, iy))
        end
        @test length(seen) == 2^L
        @test sort(qtt_site_permutation(g)) == collect(1:2^L)
        # the plan's grouped LSB-first formula, written out
        if o === :grouped && e === :lsb_first
            @test g.sites_x == collect(1:nx) && g.sites_y == collect((nx + 1):L)
        end
        # field -> QTT -> grid values, and single-point evaluation
        f(x, y) = exp(-x^2) * cos(3y) + 0.3im * sin(x * y) + 0.1x
        C = qtt_function(g, f)
        xs, ys = qtt_coords(g)
        M = [f(x, y) for x in xs, y in ys]
        @test _relerr(qtt_values(C, g), M) < 1e-12
        @test abs(qtt_value(C, g, 2^nx - 1, 1) - M[end, 2]) < 1e-12
        # separable construction == full TT-SVD
        S = qtt_separable(g, x -> exp(-x^2), y -> cos(y))
        @test _relerr(qtt_values(S, g), [exp(-x^2) * cos(y) for x in xs, y in ys]) < 1e-12
        @test maximum(tt_ranks(S)[g.nx:g.nx]; init = 1) == 1 || o !== :grouped
        # Telum round trip in several gauges; the physical L2 norm keeps the weight outside
        for c in (1, L ÷ 2 + 1, L)
            psi = qtt_mps(C; center = c)
            @test psi.center == c
            @test _relerr(tt_dense(qtt_cores(psi)), tt_dense(C)) < 1e-12
            @test abs(qtt_l2norm(psi, g) - sqrt(qtt_cell(g)) * norm(M)) < 1e-10 * norm(M)
            @test abs(norm(psi) - norm(M)) < 1e-10 * norm(M)
        end
        psi = qtt_mps(C; center = 1)
        canonical!(psi, L)
        @test all(left_isometry_defect(psi[k]) < 1e-12 for k in 1:(L - 1))
        canonical!(psi, 1)
        @test all(right_isometry_defect(psi[k]) < 1e-12 for k in 2:L)
    end
    # boundary wrap of the periodic shift on grid values
    g = qtt_grid(3, 3; order = o, endian = e)
    C = qtt_function(g, (x, y) -> x + 10y + 0.5x * y)
    Sx = qtt_embed(g; x = qtt_shift(3; bc = :periodic))
    V = qtt_values(C, g)
    @test _relerr(qtt_values(first(tt_compress(op_apply(Sx, C))), g), circshift(V, (-1, 0))) < 1e-12
end

@testset "U02: operators against sparse stencils ($o, $e)" for (o, e) in _ORDERS
    for n in (3, 4)
        N = 2^n
        # periodic GP grid
        g = qtt_grid(n, n; x = (-2.0, 2.0), y = (-2.0, 2.0), order = o, endian = e)
        xs, ys = qtt_coords(g)
        h = g.dx
        for (k, bc) in ((1, :periodic), (2, :periodic), (1, :dirichlet0), (2, :dirichlet0))
            Dx = _ref_d(N, h, k, bc)
            @test _relerr(_grid_matrix(g, op_dense(qtt_embed(g; x = qtt_d(g, :x, k; bc = bc)))),
                          Matrix(_ref2(g, Dx, nothing))) < 1e-12
            @test _relerr(_grid_matrix(g, op_dense(qtt_embed(g; y = qtt_d(g, :y, k; bc = bc)))),
                          Matrix(_ref2(g, nothing, Dx))) < 1e-12
        end
        # coordinates and the mixed derivative
        @test _relerr(_grid_matrix(g, op_dense(qtt_embed(g; x = qtt_coord_diag(g, :x)))),
                      Matrix(_ref2(g, spdiagm(0 => collect(xs)), nothing))) < 1e-12
        Dxy = qtt_embed(g; x = qtt_d(g, :x, 1; bc = :periodic), y = qtt_d(g, :y, 1; bc = :periodic))
        @test _relerr(_grid_matrix(g, op_dense(Dxy)),
                      Matrix(_ref2(g, _ref_d(N, h, 1, :periodic), _ref_d(N, h, 1, :periodic)))) < 1e-12
        # GP H0 with trap, mu and rotation
        m = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.7, omega = 1.3, Omega = 0.4)
        D2 = _ref_d(N, h, 2, :periodic)
        D1 = _ref_d(N, h, 1, :periodic)
        X, Y = spdiagm(0 => collect(xs)), spdiagm(0 => collect(ys))
        Href = -0.5 * (_ref2(g, D2, nothing) + _ref2(g, nothing, D2)) +
               (1.3^2 / 2) * (_ref2(g, X^2, nothing) + _ref2(g, nothing, Y^2)) -
               0.7 * I + 0.4im * (_ref2(g, X, D1) - _ref2(g, D1, Y))
        @test _relerr(_grid_matrix(g, op_dense(m.H0)), Matrix(Href)) < 1e-12
        @test _relerr(op_dense(m.H0), op_dense(op_adjoint(m.H0))) < 1e-12        # Hermitian
        # Heston on its non-periodic grid, generator and forward adjoint
        gh = _heston_grid(n, n; order = o, endian = e)
        xh, vh = qtt_coords(gh)
        p = _HP
        Dx1, Dx2 = _ref_d(N, gh.dx, 1, :onesided), _ref_d(N, gh.dx, 2, :onesided)
        Dv1, Dv2 = _ref_d(N, gh.dy, 1, :onesided), _ref_d(N, gh.dy, 2, :onesided)
        Vd = spdiagm(0 => collect(vh))
        Aref = 0.5 * _ref2(gh, Dx2, Vd) + p.rho * p.sigma * _ref2(gh, Dx1, Vd * Dv1) +
               0.5 * p.sigma^2 * _ref2(gh, nothing, Vd * Dv2) + (p.r - p.q) * _ref2(gh, Dx1, nothing) -
               0.5 * _ref2(gh, Dx1, Vd) + p.kappa * p.theta * _ref2(gh, nothing, Dv1) -
               p.kappa * _ref2(gh, nothing, Vd * Dv1) - p.r * I
        Ah = _grid_matrix(gh, op_dense(heston_op(gh, p)))
        @test _relerr(Ah, Matrix(Aref)) < 1e-11
        Aadj = _grid_matrix(gh, op_dense(heston_op(gh, p; adjoint = true)))
        @test _relerr(Aadj, Matrix(transpose(Aref + p.r * I))) < 1e-11
        @test norm(ones(N^2)' * Aadj) < 1e-9 * norm(Aadj)                         # mass conserved
        # at v = 0 (iy = 0) no diffusion or mixed term survives: those rows carry only the
        # first-order stencils (r - q) u_x + kappa theta u_v - r u, three x and three v entries
        Ad = _grid_matrix(gh, op_dense(heston_op(gh, HestonParams(p.r, p.q, p.kappa, p.theta, 0.0, 0.0))))
        @test _relerr(Ah[1:N, :], Ad[1:N, :]) < 1e-11
        # the Telum MPO applies the same matrix (stored-transpose convention), complex state
        C = _random_tt(_tt_ranks(2n, 4); rng = Random.MersenneTwister(n))
        for W in (heston_op(gh, p), m.H0)
            out = qtt_apply(qtt_to_mpo(W), qtt_mps(C); tol_trunc = 0.0)
            @test _relerr(vec(tt_dense(qtt_cores(out))), op_dense(W) * vec(tt_dense(C))) < 1e-10
            @test _relerr(op_dense(qtt_mpo_cores(qtt_to_mpo(W))), op_dense(W)) < 1e-14
        end
    end
    @test_throws ArgumentError qtt_derivative(1, 0.1; order = 2, bc = :onesided)
end

@testset "U03: environment action vs dense projection, cache invalidation" begin
    g = _heston_grid(3, 3)
    gp = qtt_grid(3, 3; x = (-2.0, 2.0), y = (-2.0, 2.0))
    m = gp_model(gp; gnl = 0.0, gamma = 0.1, mu = 0.3, omega = 1.0, Omega = 0.5)
    cases = ((heston_op(g, _HP), Float64), (heston_op(g, _HP), ComplexF64), (m.H0, ComplexF64))
    for (W, T) in cases, i in (1, 3, 5)
        C = _random_tt(_tt_ranks(6, 4); rng = Random.MersenneTwister(7 + i), T = T)
        b = _qtt_bond(g, W, C, i)
        Th = two_site_block(b.psi, i)
        got = RC._none_block(apply_h_two_site(Th, b.mpo, i, b.lenv, b.renv))
        @test _relerr(got, _dense_eff(b.psi, op_dense(W), i)) < 1e-11
    end
    # a moving paddle or a changed trial state must rebuild H[psi, t]
    pad = GPPaddle(5.0, -1.0, 1.0, 0.5, 0.5, 0.6, 0.8, 0.0, 0.15)
    mg = gp_model(gp; gnl = 1.0, gamma = 0.1, mu = 0.3, omega = 1.0, paddle = pad)
    cache = GPHamiltonianCache()
    psi1 = qtt_mps(_random_tt(_tt_ranks(6, 3); rng = Random.MersenneTwister(3)))
    psi2 = qtt_mps(_random_tt(_tt_ranks(6, 3); rng = Random.MersenneTwister(4)))
    H1, _ = gp_hamiltonian(mg, psi1, 0.0; cache = cache)
    H1b, _ = gp_hamiltonian(mg, psi1, 0.0; cache = cache)
    @test cache.hits == 1 && cache.misses == 1 && H1b === H1
    Ht, _ = gp_hamiltonian(mg, psi1, 0.25; cache = cache)
    @test cache.misses == 2
    @test _relerr(op_dense(qtt_mpo_cores(Ht)), op_dense(qtt_mpo_cores(H1))) > 1e-3   # paddle moved
    H2, _ = gp_hamiltonian(mg, psi2, 0.0; cache = cache)
    @test cache.misses == 3
    @test _relerr(op_dense(qtt_mpo_cores(H2)), op_dense(qtt_mpo_cores(H1))) > 1e-3   # new density
    # after the stop and the ramp the paddle is off
    @test gp_paddle_op(mg, pad.t_stop + pad.ramp + 0.1) === nothing
    @test paddle_state(pad, 10.0)[2] == pad.x_start + pad.v_sweep * pad.t_stop
end

@testset "U04: streamed DD sketches vs explicit local matrices" begin
    g = _heston_grid(5, 5)
    W = heston_op(g, _HP)
    C = _random_tt([2, 3, 5, 10, 8, 4, 2, 1, 1]; rng = Random.MersenneTwister(21))
    for i in (3, 4, 5)
        b = _qtt_bond(g, W, C, i)
        f = b.f
        @test _unsaturated(f)
        dd = _dd_dense(f, b.mpo, i, b.lenv, b.renv)
        rng = Random.MersenneTwister(100 + i)
        out = Ref{Any}(nothing)
        # K orientation: probe in V0's layout, Z = X Om^dagger
        Om = sector_graded_sketch(f.V0, :right, 5; comp_ratio = nothing, rng = rng)
        OmD = perp_component_right(f.V0, Om)
        apply_two_site_sketch!(out, OmD, TwoSiteSketchOp(f, b.mpo, i, b.lenv, b.renv))
        Z = reshape(RC._none_block(out[]), size(dd.Xm, 1), :)
        Omm = reshape(RC._none_block(OmD), leg_dim(OmD, 1), :)
        @test _relerr(Z, dd.Xm * Omm') < 1e-11
        YL = reshape(RC._none_block(perp_component(f.U0, out[])), size(dd.Xm, 1), :)
        @test _relerr(YL, dd.A * reshape(RC._none_block(Om), leg_dim(Om, 1), :)') < 1e-11
        # L orientation (mirror): probe in U0's layout, rows Om^dagger X -- X itself, not H'
        OmL = sector_graded_sketch(f.U0, :left, 5; comp_ratio = nothing, rng = rng)
        OmLD = perp_component(f.U0, OmL)
        apply_two_site_sketch!(out, OmLD,
                               TwoSiteSketchOp(f, b.mpo, i, b.lenv, b.renv; adjoint = true))
        R = reshape(RC._none_block(out[]), leg_dim(OmLD, 3), :)
        OLm = reshape(RC._none_block(OmLD), size(dd.Xm, 1), :)
        @test _relerr(R, OLm' * dd.Xm) < 1e-11
        YR = reshape(RC._none_block(perp_component_right(f.V0, out[])), leg_dim(OmLD, 3), :)
        @test _relerr(YR, reshape(RC._none_block(OmL), size(dd.Xm, 1), :)' * dd.A) < 1e-11
    end

    # D3's deterministic branches are exact; the sampled branch and D1 audited over 30 seeds
    function true_miss(ex, dd, side)
        if side === :left
            U = reshape(RC._none_block(ex.U_ex), size(dd.Xm, 1), :)
            return norm((I - U * U') * dd.A)
        end
        V = reshape(RC._none_block(ex.V_ex), :, size(dd.Xm, 2))
        return norm(dd.A * (I - V' * V))
    end
    # bond 3, left: m = 1, n = 15 -> mirrored; right: m = 15, n = 1 -> exhaustive
    b = _qtt_bond(g, W, C, 3)
    dd = _dd_dense(b.f, b.mpo, 3, b.lenv, b.renv)
    for side in (:left, :right)
        mem = DDMemory()
        al = RC.BasisAllocation(1.0, 1e-12 * norm(dd.A), 1.0, 1.0, 1.0)
        ex = dd_cbe_expand(b.f, b.mpo, 3, b.lenv, b.renv, side; alloc = al, dd_policy = :hybrid,
                           dd_memory = mem, rng = Random.MersenneTwister(5))
        @test only(mem.trace).branch === (side === :left ? :mirrored : :exhaustive)
        @test true_miss(ex, dd, side) < 1e-10 * norm(dd.A)
        @test mem.trace[1].streamed <= (side === :left ? 1 + mem.trace[1].added : 1)
        Q = side === :left ? reshape(RC._none_block(ex.U_ex), size(dd.Xm, 1), :) :
                             copy(reshape(RC._none_block(ex.V_ex), :, size(dd.Xm, 2))')
        @test norm(Q' * Q - I) < 1e-12                                           # QR orthogonality
    end
    # a wide bond for the sampled branch: L = 12, rank 16
    g12 = _heston_grid(6, 6)
    W12 = heston_op(g12, _HP)
    C12 = _random_tt(_tt_ranks(12, 16); rng = Random.MersenneTwister(33))
    b = _qtt_bond(g12, W12, C12, 6)
    dd = _dd_dense(b.f, b.mpo, 6, b.lenv, b.renv)
    share = 0.05 * norm(dd.A)
    al = RC.BasisAllocation(1.0, share, 1.0, 1.0, 1.0)
    stats = Dict(:adaptive => Int[], :hybrid => Int[])
    for pol in (:adaptive, :hybrid), seed in 1:30
        mem = DDMemory()
        lg = RC.AdaptiveLedger()
        ex = dd_cbe_expand(b.f, b.mpo, 6, b.lenv, b.renv, :left; alloc = al, dd_policy = pol,
                           dd_memory = mem, ledger = lg, rng = Random.MersenneTwister(seed))
        pol === :hybrid && @test only(mem.trace).branch === :sampled
        met = lg.tolerance_met
        push!(stats[pol], met && true_miss(ex, dd, :left) > 2 * share ? 1 : 0)
    end
    println("U04 false accepts at 2x share over 30 seeds: D1 ", sum(stats[:adaptive]),
            ", D3 ", sum(stats[:hybrid]))
    @test sum(stats[:adaptive]) <= 8
    @test sum(stats[:hybrid]) <= 8
    # the predictor is read on the next call
    mem = DDMemory()
    for _ in 1:2
        dd_cbe_expand(b.f, b.mpo, 6, b.lenv, b.renv, :left; alloc = al, dd_policy = :hybrid,
                      dd_memory = mem, rng = Random.MersenneTwister(9))
    end
    @test mem.trace[2].ell == mem.trace[1].added + RC._dd_oversample(mem.trace[1].added) ||
          mem.trace[2].branch !== :sampled
end

@testset "U05: rank adapts only where needed" begin
    # separable heat flow: the x|y bond stays rank 1
    g = qtt_grid(4, 4; x = (-3.0, 3.0), y = (-3.0, 3.0))
    Wh = qtt_op(g, [(0.5, qtt_d(g, :x, 2; bc = :periodic), nothing),
                    (0.5, nothing, qtt_d(g, :y, 2; bc = :periodic))])
    psi = qtt_mps(qtt_separable(g, x -> exp(-x^2), y -> exp(-2y^2)); center = 1)
    H = qtt_to_mpo(Wh)
    for _ in 1:4
        cbe_bug_step!(psi, H, ComplexF64(0.05); tol_basis = 1e-8, tol_trunc = 1e-10,
                      maxdim = 32, hermitian = false)
    end
    @test bond_dims(psi)[g.nx] == 1
    # correlated Heston flow from a separable start: the x|v bond grows
    gh = _heston_grid(4, 4)
    u0 = qtt_separable(gh, x -> exp(0.5x), v -> exp(-2v) * (1 + v))
    A = heston_op(gh, _HP)
    Ad = op_dense(A)
    v0 = vec(tt_dense(u0))
    errs = Float64[]
    ranks = Int[]
    for tb in (1e-3, 1e-8)
        p = qtt_mps(u0; center = 1)
        for _ in 1:5
            cbe_bug_step!(p, qtt_to_mpo(A), ComplexF64(0.02); tol_basis = tb, tol_trunc = 1e-12,
                          maxdim = 64, hermitian = false)
        end
        ref = exp(0.1 * Matrix(Ad)) * v0
        push!(errs, norm(vec(tt_dense(qtt_cores(p))) - ref) / norm(ref))
        push!(ranks, bond_dims(p)[gh.nx])
    end
    println("U05 Heston errors (tol_basis 1e-3, 1e-8): ", errs, " x|v ranks ", ranks)
    @test all(>(1), ranks)
    @test errs[2] <= max(errs[1], 1e-10)
    @test errs[2] < 1e-5
end

@testset "U06: nonlinear action, stage refresh, real-linear JVP" begin
    g = qtt_grid(3, 3; x = (-2.0, 2.0), y = (-2.0, 2.0))
    pad = GPPaddle(4.0, -1.0, 1.0, 0.5, 0.5, 0.6, 0.8, 0.0, 0.15)
    m = gp_model(g; gnl = 1.5, gamma = 0.1, mu = 0.4, omega = 1.0, paddle = pad, rho_tol = 1e-13)
    A1 = _random_tt(_tt_ranks(6, 4); rng = Random.MersenneTwister(41))
    A2 = _random_tt(_tt_ranks(6, 4); rng = Random.MersenneTwister(42))
    v1, v2 = vec(tt_dense(A1)), vec(tt_dense(A2))
    t = 0.2
    Hl = op_dense(gp_linear_op(m, t))
    Fd(v) = (-1im - m.gamma) .* (Hl * v .+ m.g .* abs2.(v) .* v)
    # density QTT == elementwise |psi|^2 (zip-up, bounded error)
    rho, info = gp_density(m, A1)
    @test _relerr(vec(tt_dense(rho)), abs2.(v1)) < 1e-11
    @test info.err_bound < 1e-11
    ex, _ = tt_compress(tt_hadamard(A1, tt_conj(A1)); tol = 1e-13)
    @test _relerr(vec(tt_dense(rho)), vec(tt_dense(ex))) < 1e-11
    @test gp_density_audit(m, A1) < 1e-10
    # H[psi, t] and F(psi) against dense
    H1, _ = gp_hamiltonian_op(m, A1, t)
    @test _relerr(op_dense(H1), Hl + m.g * Diagonal(abs2.(v1))) < 1e-11
    @test _relerr(vec(tt_dense(gp_vector_field(m, A1, t))), Fd(v1)) < 1e-10
    # g = 0 is the linear path exactly
    m0 = gp_model(g; gnl = 0.0, gamma = 0.1, mu = 0.4, omega = 1.0, paddle = pad)
    @test _relerr(op_dense(first(gp_hamiltonian_op(m0, A1, t))), Hl) < 1e-13
    # the density follows the TRIAL state; a frozen one does not
    H2, _ = gp_hamiltonian_op(m, A2, t)
    @test _relerr(op_dense(H2), Hl + m.g * Diagonal(abs2.(v2))) < 1e-11
    Hfroz, _ = gp_hamiltonian_op(m, A2, t; rho = rho)
    @test _relerr(op_dense(Hfroz), Hl + m.g * Diagonal(abs2.(v1))) < 1e-11
    @test _relerr(op_dense(Hfroz), op_dense(H2)) > 1e-3
    # the real-linear JVP: finite differences of the dense F, and the conj(dpsi) term
    dA = _random_tt(_tt_ranks(6, 2); rng = Random.MersenneTwister(43))
    dv = vec(tt_dense(dA))
    J = vec(tt_dense(gp_jvp(m, A1, dA, t)))
    eps_ = 1e-6
    fd = (Fd(v1 .+ eps_ .* dv) .- Fd(v1 .- eps_ .* dv)) ./ (2eps_)
    @test _relerr(J, fd) < 1e-7
    Ji = vec(tt_dense(gp_jvp(m, A1, tt_scale(dA, 1im), t)))
    fdi = (Fd(v1 .+ 1im * eps_ .* dv) .- Fd(v1 .- 1im * eps_ .* dv)) ./ (2eps_)
    @test _relerr(Ji, fdi) < 1e-7
    @test _relerr(Ji, 1im .* J) > 1e-3                         # NOT complex-linear
    @test _relerr(vec(tt_dense(gp_jvp(m0, A1, dA, t))), (-1im - 0.1) .* (Hl * dv)) < 1e-11
    # the frozen-density step is labelled and runs; tau = (-i - gamma) dt
    psi = qtt_mps(A1)
    psi[psi.center] = to_concrete(psi[psi.center] / norm(psi))
    info, hinfo = gp_bug_step!(psi, m, t, 0.01; tol_basis = 1e-8, tol_trunc = 1e-10, maxdim = 32)
    @test hinfo.density === :frozen_density
    @test_throws ArgumentError gp_bug_step!(psi, m, t, 0.01; density = :consistent)
end

end
