# ── 2D QTT models: damped Gross-Pitaevskii and Heston (QTT plan, Tasks 1.1 and 1.3A) ──
# Every operator is assembled from 1D chains (`qtt_ops1d.jl`) placed by `qtt_embed`, as operator
# cores, and becomes a solver MPO only through `qtt_to_mpo`. The GP equation is NEVER linearised:
#   d_t psi = F(t, psi) = (-i - gamma) H[psi, t] psi,
#   H[psi, t] = -Delta/2 + V_trap + V_pad(t) - mu - Omega L_z + g diag(rho),  rho = C(|psi|^2),
# and `diag(rho)` is rebuilt from the state it is applied to (`gp_hamiltonian`). Holding one rho for
# a whole step is the separately labelled `:frozen_density` ablation (`gp_bug_step!`).
# Heston, in time to maturity: u_tau = A_H u,
#   A_H = v/2 d_xx + rho sigma v d_xv + sigma^2 v/2 d_vv + (r - q - v/2) d_x + kappa (theta - v) d_v - r,
# and the forward density uses the discrete transpose of A_H without the discount, which conserves
# sum_i p_i exactly because every stencil annihilates constants.

# ── shared 1D pieces on a grid ──

_n(g::QTTGrid, c::Symbol) = c === :x ? g.nx : g.ny
_h(g::QTTGrid, c::Symbol) = c === :x ? g.dx : g.dy
_coord1(g::QTTGrid, c::Symbol) = (c === :x ? first(qtt_coords(g)) : last(qtt_coords(g)))

"The 1D derivative chain of order `k` along `c` with boundary fixture `bc`."
qtt_d(g::QTTGrid, c::Symbol, k::Int; bc::Symbol) = qtt_derivative(_n(g, c), _h(g, c); order = k, bc = bc)

"`diag(f(c))` along coordinate `c` as a 1D chain."
qtt_coord_diag(g::QTTGrid, c::Symbol, f = identity) = qtt_diag_1d(f.(_coord1(g, c)))

"""
    qtt_op(g, terms; tol = 1e-13) -> operator cores

`sum_t c_t (X_t (x) Y_t)` on the grid for `terms = [(c, X, Y), ...]` of 1D chains (`nothing` = the
identity on that coordinate), compressed once.
"""
qtt_op(g::QTTGrid, terms; tol::Real = 1e-13) =
    op_lincomb([(c, qtt_embed(g; x = X, y = Y)) for (c, X, Y) in terms]; tol = tol)

# ── Gross-Pitaevskii ──

"""
    GPPaddle(height, x_start, v_sweep, t_stop, ramp, width_x, width_y, y_c, smooth)

A rounded rectangular paddle, `V_pad = A(t) b_x(x - x_p(t)) b_y(y - y_c)` with the smoothed box
`b(u) = (tanh((u + w/2)/s) - tanh((u - w/2)/s)) / 2`, centre `x_p = x_start + v_sweep min(t, t_stop)`
and height ramped linearly to zero over `ramp` after `t_stop`. Separable, so its QTT is two 1D
TT-SVDs at any size. ⚠ AN ENGINEERING FIXTURE: Connor et al.'s Appendix B shape must be read off
the source before Task 3's reproduction run (plan §3.2).
"""
struct GPPaddle
    height::Float64
    x_start::Float64
    v_sweep::Float64
    t_stop::Float64
    ramp::Float64
    width_x::Float64
    width_y::Float64
    y_c::Float64
    smooth::Float64
end

_box(u, w, s) = (tanh((u + w / 2) / s) - tanh((u - w / 2) / s)) / 2

"Height and centre of the paddle at time `t`."
function paddle_state(p::GPPaddle, t::Real)
    amp = t <= p.t_stop ? p.height :
          p.ramp > 0 ? p.height * max(0.0, 1 - (t - p.t_stop) / p.ramp) : 0.0
    return amp, p.x_start + p.v_sweep * min(t, p.t_stop)
end

"""
    GPModel

The damped GP problem on a grid: `g`, `gamma`, `mu`, trap frequency `omega`, rotation `Omega`, an
optional `paddle`, and the STATIC linear part `H0 = -Delta/2 + V_trap - mu - Omega L_z` as operator
cores (kinetic periodic, the plan's GP boundary).
"""
struct GPModel
    grid::QTTGrid
    g::Float64
    gamma::Float64
    mu::Float64
    omega::Float64
    Omega::Float64
    paddle::Union{Nothing, GPPaddle}
    H0::Vector{Array{ComplexF64, 4}}
    rho_tol::Float64
end

"""
    gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.0, omega = 1.0, Omega = 0.0, paddle = nothing,
             rho_tol = 1e-12) -> GPModel

`L_z = -i (x d_y - y d_x)`. `rho_tol` is the relative compression tolerance of the density QTT; its
action error is audited separately (`gp_density_audit`), never assumed.
"""
function gp_model(g::QTTGrid; gnl::Real = 1.0, gamma::Real = 0.1, mu::Real = 0.0,
                  omega::Real = 1.0, Omega::Real = 0.0, paddle = nothing, rho_tol::Real = 1e-12)
    g.periodic_x && g.periodic_y || throw(ArgumentError(
        "gp_model: the GP fixtures are periodic; build the grid with periodic_x = periodic_y = true"))
    Dxx, Dyy = qtt_d(g, :x, 2; bc = :periodic), qtt_d(g, :y, 2; bc = :periodic)
    terms = Any[(-0.5, Dxx, nothing), (-0.5, nothing, Dyy)]
    if omega != 0
        push!(terms, (omega^2 / 2, qtt_coord_diag(g, :x, x -> x^2), nothing))
        push!(terms, (omega^2 / 2, nothing, qtt_coord_diag(g, :y, y -> y^2)))
    end
    mu != 0 && push!(terms, (-mu, nothing, op_identity(g.ny)))
    if Omega != 0
        # -Omega L_z = i Omega (x d_y - y d_x)
        Dx, Dy = qtt_d(g, :x, 1; bc = :periodic), qtt_d(g, :y, 1; bc = :periodic)
        push!(terms, (1im * Omega, qtt_coord_diag(g, :x), Dy))
        push!(terms, (-1im * Omega, Dx, qtt_coord_diag(g, :y)))
    end
    H0 = [ComplexF64.(w) for w in qtt_op(g, terms)]
    return GPModel(g, Float64(gnl), Float64(gamma), Float64(mu), Float64(omega), Float64(Omega),
                   paddle, H0, Float64(rho_tol))
end

"`V_pad(t)` as operator cores (`nothing` without a paddle or once it is off)."
function gp_paddle_op(m::GPModel, t::Real)
    p = m.paddle
    p === nothing && return nothing
    amp, xc = paddle_state(p, t)
    amp == 0 && return nothing
    xs, ys = qtt_coords(m.grid)
    bx = tt_from_samples(_box.(xs .- xc, p.width_x, p.smooth))
    by = tt_from_samples(_box.(ys .- p.y_c, p.width_y, p.smooth))
    return op_scale(op_diag(qtt_embed(m.grid; x = bx, y = by)), amp)
end

"The state-independent part `H0 + V_pad(t)` as operator cores."
function gp_linear_op(m::GPModel, t::Real)
    Vp = gp_paddle_op(m, t)
    Vp === nothing && return m.H0
    return op_lincomb([(1.0, m.H0), (1.0, Vp)])
end

"""
    gp_density(m, psi_cores) -> (rho_cores, info)

`rho = C(|psi|^2)` at `m.rho_tol` (`tt_density`: zip-up, error bound checked, retried once).
"""
gp_density(m::GPModel, A::AbstractVector) = tt_density(A; tol = m.rho_tol)

"""
    gp_hamiltonian_op(m, psi_cores, t; rho = nothing) -> (H cores, info)

`H[psi, t] = H0 + V_pad(t) + g diag(rho(psi))`. Pass `rho` to reuse a density built elsewhere (the
frozen-density ablation); by default it is rebuilt from `psi_cores`. `info` carries the density's
`chi`, error bound and seconds, and the operator's bond dimensions.
"""
function gp_hamiltonian_op(m::GPModel, A::AbstractVector, t::Real; rho = nothing)
    Hl = gp_linear_op(m, t)
    if m.g == 0
        return Hl, (chi_rho = 0, rho_err = 0.0, t_rho = 0.0, retried = false,
                    w = maximum(tt_ranks(Hl); init = 1))
    end
    rinfo = (chi = maximum(tt_ranks(rho === nothing ? A : rho); init = 1), err_bound = 0.0,
             seconds = 0.0, retried = false)
    if rho === nothing
        rho, rinfo = gp_density(m, A)
    end
    H = op_lincomb([(1.0, Hl), (m.g, op_diag(rho))])
    return H, (chi_rho = rinfo.chi, rho_err = rinfo.err_bound, t_rho = rinfo.seconds,
               retried = rinfo.retried, w = maximum(tt_ranks(H); init = 1))
end

"""
    GPHamiltonianCache

Memo of `H[psi, t]` MPOs keyed by `(state tag, t)`: the tag is a hash of the state's core data, so
a changed trial state or stage time is a MISS and rebuilds the density and the MPO -- a stale
nonlinear or moving-paddle operator cannot be reused (plan §1.3A cache tags). `hits`/`misses`
count.
"""
mutable struct GPHamiltonianCache
    store::Dict{Tuple{UInt, Float64}, Any}
    hits::Int
    misses::Int
end
GPHamiltonianCache() = GPHamiltonianCache(Dict{Tuple{UInt, Float64}, Any}(), 0, 0)

"A content hash of a state's cores (gauge-dependent: a re-gauged state is a different tag)."
qtt_state_tag(A::AbstractVector) = foldl((h, c) -> hash(c, h), A; init = hash(length(A)))
qtt_state_tag(psi::SymMPS) = qtt_state_tag(qtt_cores(psi))

"""
    gp_hamiltonian(m, psi, t; cache = nothing, rho = nothing) -> (MPO, info)

The solver MPO of `H[psi, t]` for a `SymMPS` (see [`gp_hamiltonian_op`](@ref)); with `cache`, an
unchanged `(psi, t)` returns the stored MPO.
"""
function gp_hamiltonian(m::GPModel, psi::SymMPS, t::Real; cache = nothing, rho = nothing)
    A = qtt_cores(psi)
    key = (qtt_state_tag(A), Float64(t))
    if cache !== nothing && rho === nothing && haskey(cache.store, key)
        cache.hits += 1
        return cache.store[key]
    end
    H, info = gp_hamiltonian_op(m, A, t; rho = rho)
    out = (qtt_to_mpo(H), info)
    if cache !== nothing && rho === nothing
        cache.misses += 1
        cache.store[key] = out
    end
    return out
end

"""
    gp_vector_field(m, A, t; tol = 1e-12) -> TT

`F(t, psi) = (-i - gamma) H[psi, t] psi` on TT cores, the density rebuilt from `A` itself.
"""
function gp_vector_field(m::GPModel, A::AbstractVector, t::Real; tol::Real = 1e-12)
    H, _ = gp_hamiltonian_op(m, A, t)
    return first(tt_compress(tt_scale(op_apply(H, A), -1im - m.gamma); tol = tol))
end

"""
    gp_jvp(m, A, dA, t; tol = 1e-12) -> TT

The REAL-linear derivative `DF(psi)[dpsi] = (-i - gamma) [(H0 + V_pad - mu + 2 g |psi|^2) dpsi +
g psi^2 conj(dpsi)]` (plan §2.3A). Not complex-linear: `gp_jvp(m, A, im*dA, t) != im*gp_jvp(m, A,
dA, t)` whenever `g != 0`. For Newton-Krylov stage solves only; the CBE always sketches the true
`F`.
"""
function gp_jvp(m::GPModel, A::AbstractVector, dA::AbstractVector, t::Real; tol::Real = 1e-12)
    Hl = gp_linear_op(m, t)
    lin = op_apply(Hl, dA)
    m.g == 0 && return first(tt_compress(tt_scale(lin, -1im - m.gamma); tol = tol))
    rho, _ = gp_density(m, A)
    psi2 = first(tt_compress(tt_hadamard(A, A); tol = m.rho_tol))
    t1 = op_apply(op_diag(rho), dA)
    t2 = op_apply(op_diag(psi2), tt_conj(dA))
    acc = tt_lincomb([(1.0, lin), (2 * m.g, t1), (m.g, t2)]; tol = tol)
    return first(tt_compress(tt_scale(acc, -1im - m.gamma); tol = tol))
end

"""
    gp_density_audit(m, A) -> Float64

`|g| ||(rho_QTT - |psi|^2) psi|| / ||psi||` from dense samples: the plan's action audit of the
density compression (multiply by `h` for the time-weighted share). Small grids only.
"""
function gp_density_audit(m::GPModel, A::AbstractVector)
    v = vec(tt_dense(A))
    rho, _ = gp_density(m, A)
    r = vec(tt_dense(rho))
    return abs(m.g) * norm((r .- abs2.(v)) .* v) / max(norm(v), eps())
end

"""
    gp_bug_step!(psi, m, t, dt; density = :frozen_density, kwargs...) -> (info, hinfo)

One CBE-BUG step of the damped GP equation from `t` to `t + dt` with `H[psi_n, t_n]` held through
the step -- the plan's MODE A, labelled `:frozen_density`: the density is the step input's, so the
nonlinear time order is at most that of the lag, whatever `order` the linear step has. The
stage-consistent MODE B is Task 2. `kwargs` go to `cbe_bug_step!` (`hermitian = false` is forced:
the generator `(-i - gamma) H` is not Hermitian).
"""
function gp_bug_step!(psi::SymMPS, m::GPModel, t::Real, dt::Real;
                      density::Symbol = :frozen_density, cache = nothing, kwargs...)
    density === :frozen_density || throw(ArgumentError(
        "gp_bug_step!: only :frozen_density (plan mode A) exists in Task 1; stage-consistent " *
        "nonlinear stages are Task 2"))
    H, hinfo = gp_hamiltonian(m, psi, t; cache = cache)
    info = cbe_bug_step!(psi, H, ComplexF64((-1im - m.gamma) * dt); kwargs..., hermitian = false)
    return info, merge(hinfo, (density = density,))
end

# ── Heston ──

"""
    HestonParams(r, q, kappa, theta, sigma, rho)

Rate, dividend yield, mean reversion, long-run variance, volatility of variance and correlation.
`feller(p)` is `2 kappa theta >= sigma^2`.
"""
struct HestonParams
    r::Float64
    q::Float64
    kappa::Float64
    theta::Float64
    sigma::Float64
    rho::Float64
end
feller(p::HestonParams) = 2 * p.kappa * p.theta >= p.sigma^2

"""
    heston_op(g, p; adjoint = false, bc = :onesided) -> operator cores

`A_H` on the grid (`y` = variance `v`, which must start at `v = 0`), or with `adjoint = true` the
forward-density operator `A_H'` (discrete transpose, NO discount), which conserves `sum p`. The
mixed term is the product of the two first-derivative chains, so it is exactly `D_x (x) D_v` with the
variance factor on `D_v`'s rows.
"""
function heston_op(g::QTTGrid, p::HestonParams; adjoint::Bool = false, bc::Symbol = :onesided)
    !g.periodic_x && !g.periodic_y || throw(ArgumentError(
        "heston_op: build the grid with periodic_x = periodic_y = false (end points on the grid)"))
    abs(g.y0) < 1e-14 || throw(ArgumentError("heston_op: the variance grid must start at v = 0"))
    Dx, Dxx = qtt_d(g, :x, 1; bc = bc), qtt_d(g, :x, 2; bc = bc)
    Dv, Dvv = qtt_d(g, :y, 1; bc = bc), qtt_d(g, :y, 2; bc = bc)
    V = qtt_coord_diag(g, :y)
    vDv, vDvv = op_compress(op_compose(V, Dv)), op_compress(op_compose(V, Dvv))
    terms = Any[(0.5, Dxx, V), (p.rho * p.sigma, Dx, vDv), (0.5 * p.sigma^2, nothing, vDvv),
                (p.r - p.q, Dx, nothing), (-0.5, Dx, V),
                (p.kappa * p.theta, nothing, Dv), (-p.kappa, nothing, vDv)]
    adjoint || push!(terms, (-p.r, nothing, op_identity(g.ny)))
    A = qtt_op(g, terms)
    return adjoint ? op_transpose(A) : A
end

"`heston_op` as a solver MPO (`tau = dt` integrates `u_tau = A u`; pass `hermitian = false`)."
heston_mpo(g::QTTGrid, p::HestonParams; kwargs...) = qtt_to_mpo(heston_op(g, p; kwargs...))
