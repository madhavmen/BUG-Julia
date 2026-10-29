# benchmarks/qtt/common.jl -- shared pieces of the QTT plan's drivers: CSV output, independent
# sparse-grid references (written from the stencils of tests/common/qtt_fixtures.jl, no QTT code),
# and the fixtures every task uses.

using LinearAlgebra, SparseArrays, Printf, Random, Statistics, Serialization
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate
const RC = BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "..", "tests", "common", "qtt_fixtures.jl"))
set_symmetry!(:none)

envs(k, d) = get(ENV, k, d)
envi(k, d) = parse(Int, envs(k, string(d)))
envf(k, d) = parse(Float64, envs(k, string(d)))
envlist(k, d) = [parse(Float64, s) for s in split(envs(k, d), ",") if !isempty(s)]

"Append NamedTuple rows to a CSV (header from the first row; written whole each call)."
function write_csv(path, rows)
    isempty(rows) && return
    ks = collect(keys(rows[1]))
    for r in rows, k in keys(r)
        k in ks || push!(ks, k)
    end
    q(x) = (s = string(x); occursin(',', s) ? "\"" * s * "\"" : s)
    open(path, "w") do io
        println(io, join(ks, ","))
        for r in rows
            println(io, join((q(get(r, k, "")) for k in ks), ","))
        end
    end
end

# ── independent references on the grid index (ix + 1 + Nx iy) ──

"Sparse GP linear operator `-Delta/2 + V_trap - mu - Omega L_z` (+ paddle at t) from stencils."
function gp_sparse_linear(m::GPModel, t::Real)
    g = m.grid
    Nx, Ny = 2^g.nx, 2^g.ny
    xs, ys = qtt_coords(g)
    D2x, D2y = _ref_d(Nx, g.dx, 2, :periodic), _ref_d(Ny, g.dy, 2, :periodic)
    X, Y = spdiagm(0 => collect(xs)), spdiagm(0 => collect(ys))
    H = -0.5 * (_ref2(g, D2x, nothing) + _ref2(g, nothing, D2y)) +
        (m.omega^2 / 2) * (_ref2(g, X^2, nothing) + _ref2(g, nothing, Y^2)) - m.mu * I
    if m.Omega != 0
        D1x, D1y = _ref_d(Nx, g.dx, 1, :periodic), _ref_d(Ny, g.dy, 1, :periodic)
        H = H + 1im * m.Omega * (_ref2(g, X, D1y) - _ref2(g, D1x, Y))
    end
    if m.paddle !== nothing
        amp, xc = paddle_state(m.paddle, t)
        if amp != 0
            p = m.paddle
            bx = RC._box.(xs .- xc, p.width_x, p.smooth)
            by = RC._box.(ys .- p.y_c, p.width_y, p.smooth)
            H = H + amp * spdiagm(0 => vec(bx * transpose(by)))
        end
    end
    return complex(H)
end

"Dense RK4 of `d_t v = (-i - gamma)(H(t) v + g |v|^2 v)` on the grid -- the GP time reference."
function gp_dense_rk4(m::GPModel, v0::AbstractVector, t0::Real, T::Real; dt::Real = 1e-4)
    v = ComplexF64.(v0)
    pref = -1im - m.gamma
    F(t, w) = pref .* (gp_sparse_linear(m, t) * w .+ m.g .* abs2.(w) .* w)
    n = max(1, round(Int, (T - t0) / dt))
    h = (T - t0) / n
    static = m.paddle === nothing
    Hs = static ? gp_sparse_linear(m, t0) : nothing
    Fs(t, w) = static ? pref .* (Hs * w .+ m.g .* abs2.(w) .* w) : F(t, w)
    t = Float64(t0)
    for _ in 1:n
        k1 = Fs(t, v); k2 = Fs(t + h / 2, v .+ h / 2 .* k1)
        k3 = Fs(t + h / 2, v .+ h / 2 .* k2); k4 = Fs(t + h, v .+ h .* k3)
        v = v .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        t += h
    end
    return v
end

"Sparse Heston generator (or its undiscounted transpose) from stencils."
function heston_sparse(g::QTTGrid, p::HestonParams; adjoint::Bool = false)
    Nx, Ny = 2^g.nx, 2^g.ny
    vh = last(qtt_coords(g))
    Dx1, Dx2 = _ref_d(Nx, g.dx, 1, :onesided), _ref_d(Nx, g.dx, 2, :onesided)
    Dv1, Dv2 = _ref_d(Ny, g.dy, 1, :onesided), _ref_d(Ny, g.dy, 2, :onesided)
    Vd = spdiagm(0 => collect(vh))
    A = 0.5 * _ref2(g, Dx2, Vd) + p.rho * p.sigma * _ref2(g, Dx1, Vd * Dv1) +
        0.5 * p.sigma^2 * _ref2(g, nothing, Vd * Dv2) + (p.r - p.q) * _ref2(g, Dx1, nothing) -
        0.5 * _ref2(g, Dx1, Vd) + p.kappa * p.theta * _ref2(g, nothing, Dv1) -
        p.kappa * _ref2(g, nothing, Vd * Dv1)
    return adjoint ? sparse(transpose(A)) : A - p.r * I
end

"Dense RK4 of a linear `d_t v = A v` (sparse A) -- the time reference of linear problems."
function linear_rk4(A, v0, T; dt::Real = 1e-4)
    n = max(1, round(Int, T / dt))
    h = T / n
    v = ComplexF64.(v0)
    for _ in 1:n
        k1 = A * v; k2 = A * (v .+ h / 2 .* k1); k3 = A * (v .+ h / 2 .* k2); k4 = A * (v .+ h .* k3)
        v = v .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    end
    return v
end

"A grid field (matrix) of TT cores, flattened on the grid index."
grid_vec(A, g) = vec(qtt_values(A, g))

"TT cores of a grid vector (fixture sizes)."
grid_tt(v, g; tol = 1e-13, maxdim = typemax(Int)) =
    qtt_function(g, (x, y) -> v[round(Int, (x - g.x0) / g.dx) + 1 +
                                2^g.nx * round(Int, (y - g.y0) / g.dy)]; tol = tol, maxdim = maxdim)

# ── fixtures ──

phi_osc(n::Int, x) = n == 0 ? pi^(-0.25) * exp(-x^2 / 2) :
                     n == 1 ? sqrt(2) * x * phi_osc(0, x) :
                     (sqrt(2 / n) * x * phi_osc(n - 1, x) - sqrt((n - 1) / n) * phi_osc(n - 2, x))

"A rank-rich oscillator superposition (plan §2.4: a nontrivial, entangled start)."
function rank_rich_state(g; nmax = 4, seed = 1)
    rng = MersenneTwister(seed)
    terms = Any[]
    for mx in 0:nmax, my in 0:nmax
        c = (randn(rng) + 1im * randn(rng)) / (1 + mx + my)^2
        push!(terms, (c, qtt_separable(g, x -> phi_osc(mx, x), y -> phi_osc(my, y))))
    end
    return tt_lincomb(terms; tol = 1e-13)
end

"Git commit of the snapshot (`COMMIT` beside the job, or ENV)."
git_commit() = get(ENV, "QTT_COMMIT", isfile("COMMIT") ? strip(read("COMMIT", String)) : "unknown")
