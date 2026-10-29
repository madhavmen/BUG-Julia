# ── observables and exact references for the QTT plan (Tasks 2-4) ──
# Physical quantities use the discrete L2 inner product (weight dx dy kept OUTSIDE the cores). Large
# fields are read by selective TT evaluation (`qtt_sample`), never by densifying 2^L values.

# ── normalisation and sampling ──

"Rescale `psi` in place so that `dx dy sum |psi|^2 = N`."
function qtt_normalize!(psi::SymMPS, g::QTTGrid; N::Real = 1.0)
    nrm = qtt_l2norm(psi, g)
    nrm > 0 || throw(ArgumentError("qtt_normalize!: zero state"))
    psi[psi.center] = to_concrete(psi[psi.center] * (sqrt(N) / nrm))
    return psi
end

"""
    qtt_sample(A, g; stride = 1) -> (xs, ys, M)

The field on every `stride`-th grid point in each direction, by one core contraction per point:
reduced-resolution images of a field of any size (plan §4.3).
"""
function qtt_sample(A::AbstractVector, g::QTTGrid; stride::Int = 1)
    ix = 0:stride:(2^g.nx - 1)
    iy = 0:stride:(2^g.ny - 1)
    xs, ys = qtt_coords(g)
    M = [qtt_value(A, g, i, j) for i in ix, j in iy]
    return xs[ix .+ 1], ys[iy .+ 1], M
end

# ── Gross-Pitaevskii ──

const _GP_PARTS = IdDict{Any, Any}()

"The model's separate operators: kinetic `-Delta/2`, trap `V_trap`, rotation `-Omega L_z` (cores)."
function gp_parts(m::GPModel)
    get!(_GP_PARTS, m) do
        g = m.grid
        K = qtt_op(g, [(-0.5, qtt_d(g, :x, 2; bc = :periodic), nothing),
                       (-0.5, nothing, qtt_d(g, :y, 2; bc = :periodic))])
        V = m.omega == 0 ? nothing :
            qtt_op(g, [(m.omega^2 / 2, qtt_coord_diag(g, :x, x -> x^2), nothing),
                       (m.omega^2 / 2, nothing, qtt_coord_diag(g, :y, y -> y^2))])
        R = m.Omega == 0 ? nothing :
            qtt_op(g, [(1im * m.Omega, qtt_coord_diag(g, :x), qtt_d(g, :y, 1; bc = :periodic)),
                       (-1im * m.Omega, qtt_d(g, :x, 1; bc = :periodic), qtt_coord_diag(g, :y))])
        (K = K, V = V, R = R)
    end
end

"`<A| W |A>` for operator cores `W`."
_expect(W, A) = W === nothing ? 0.0 : real(tt_dot(A, op_apply(W, A)))

"""
    gp_observables(m, A, t) -> NamedTuple

Particle number `N`, kinetic / trap / rotation / interaction / paddle energies, the grand-canonical
functional `F = E_kin + E_trap + E_rot + E_pad - mu N + g/2 int |psi|^4` (plan §2.5), the normalised
energy `E = (E_kin + E_trap + E_rot + g/2 int|psi|^4) / N` with chemical potential
`mu[psi] = <psi, H[psi] psi> / <psi, psi>`, the stationarity residual
`R_mu = ||(H[psi] - mu[psi]) psi|| / ||psi||`, and the paddle's external power
`P_ext = int (d_t V_pad) |psi|^2` (central difference in t).
"""
function gp_observables(m::GPModel, A::AbstractVector, t::Real)
    g = m.grid
    w = qtt_cell(g)
    P = gp_parts(m)
    n2 = tt_norm(A)^2
    rho, _ = gp_density(m, A)
    i4 = real(tt_dot(rho, rho))                      # sum |psi|^4 (rho real)
    ek, ev, er = w * _expect(P.K, A), w * _expect(P.V, A), w * _expect(P.R, A)
    Vp = gp_paddle_op(m, t)
    ep = w * _expect(Vp, A)
    eint = m.g / 2 * w * i4
    H, _ = gp_hamiltonian_op(m, A, t)
    HA = first(tt_compress(op_apply(H, A); tol = 1e-12))
    mu = real(tt_dot(A, HA)) / max(n2, eps())
    Rm = tt_norm(tt_lincomb([(1.0, HA), (-mu, A)]; tol = 1e-12)) / sqrt(max(n2, eps()))
    dtv = 1e-4
    Vf, Vb = gp_paddle_op(m, t + dtv), gp_paddle_op(m, max(t - dtv, 0.0))
    pext = (w * _expect(Vf, A) - w * _expect(Vb, A)) / (t + dtv - max(t - dtv, 0.0))
    N = w * n2
    return (N = N, E_kin = ek, E_trap = ev, E_rot = er, E_pad = ep, E_int = eint,
            F_gc = ek + ev + er + ep - m.mu * N + eint,
            E = (ek + ev + er + eint) / max(N, eps()), mu = mu + m.mu, R_mu = Rm, P_ext = pext,
            chi = maximum(tt_ranks(A); init = 1))
end

"""
    vortex_count(M; mask = 0.05) -> (n_plus, n_minus, cores)

Signed winding `q = (1/2pi) sum arg-differences` of `M`'s phase around every grid plaquette, skipping
plaquettes whose density falls below `mask * max|M|^2` (ill-conditioned phase, plan §3.2). `cores`
lists `(i, j, q)` for every charged plaquette.
"""
function vortex_count(M::AbstractMatrix; mask::Real = 0.05)
    nx, ny = size(M)
    dmax = maximum(abs2, M)
    wrap(a) = mod(a + pi, 2pi) - pi
    np, nm = 0, 0
    cores = Tuple{Int, Int, Int}[]
    for i in 1:(nx - 1), j in 1:(ny - 1)
        z = (M[i, j], M[i + 1, j], M[i + 1, j + 1], M[i, j + 1])
        minimum(abs2, z) < mask * dmax && continue
        s = sum(wrap(angle(z[mod1(k + 1, 4)]) - angle(z[k])) for k in 1:4)
        q = round(Int, s / 2pi)
        q == 0 && continue
        push!(cores, (i, j, q))
        q > 0 ? (np += q) : (nm -= q)
    end
    return np, nm, cores
end

# ── Heston references ──

"""
    heston_riccati(p, a, b, tau; nsteps = 20000, discount = true) -> (B, C)

The exact exponential-affine solution `u = exp(a x + B(tau) v + C(tau))` of `u_tau = A_H u` from
`u(0) = exp(a x + b v)` (plan §2.3): `B' = sigma^2 B^2/2 + (rho sigma a - kappa) B + (a^2 - a)/2`,
`C' = (r - q) a + kappa theta B - r` (without `- r` when `discount = false`, the forward-moment
version), integrated by classical RK4 to near machine precision.
"""
function heston_riccati(p::HestonParams, a::Real, b::Real, tau::Real; nsteps::Int = 20000,
                        discount::Bool = true)
    fB(B) = p.sigma^2 * B^2 / 2 + (p.rho * p.sigma * a - p.kappa) * B + (a^2 - a) / 2
    fC(B) = (p.r - p.q) * a + p.kappa * p.theta * B - (discount ? p.r : 0.0)
    B, C = Float64(b), 0.0
    dt = tau / nsteps
    for _ in 1:nsteps
        k1 = fB(B);              l1 = fC(B)
        k2 = fB(B + dt / 2 * k1); l2 = fC(B + dt / 2 * k1)
        k3 = fB(B + dt / 2 * k2); l3 = fC(B + dt / 2 * k2)
        k4 = fB(B + dt * k3);     l4 = fC(B + dt * k3)
        B += dt / 6 * (k1 + 2k2 + 2k3 + k4)
        C += dt / 6 * (l1 + 2l2 + 2l3 + l4)
    end
    return B, C
end

"The affine field `exp(a x + B v + C)` on the grid (rank 1, any size)."
heston_affine_field(g::QTTGrid, a::Real, B::Real, C::Real) =
    qtt_separable(g, x -> exp(a * x + C), v -> exp(B * v))

"""
    heston_call(p, S, K, v, T; umax = 200, n = 4000) -> Float64

The European call by Heston's semi-analytic formula (Gil-Pelaez, the "little Heston trap"
characteristic function, trapezoid on `(0, umax]`): an INDEPENDENT reference for the payoff fixture
(plan §2.3), not a 2D closed form.
"""
function heston_call(p::HestonParams, S::Real, K::Real, v::Real, T::Real; umax::Real = 200.0,
                     n::Int = 4000)
    x = log(S)
    function phi(u)
        d = sqrt((p.rho * p.sigma * 1im * u - p.kappa)^2 + p.sigma^2 * (1im * u + u^2))
        gg = (p.kappa - p.rho * p.sigma * 1im * u - d) / (p.kappa - p.rho * p.sigma * 1im * u + d)
        e = exp(-d * T)
        Cc = (p.r - p.q) * 1im * u * T + p.kappa * p.theta / p.sigma^2 *
             ((p.kappa - p.rho * p.sigma * 1im * u - d) * T - 2 * log((1 - gg * e) / (1 - gg)))
        Dd = (p.kappa - p.rho * p.sigma * 1im * u - d) / p.sigma^2 * (1 - e) / (1 - gg * e)
        return exp(Cc + Dd * v + 1im * u * x)
    end
    du = umax / n
    lk = log(K)
    P1 = 0.0
    P2 = 0.0
    f1 = phi(-1im)
    for k in 1:n
        u = (k - 0.5) * du
        P1 += real(exp(-1im * u * lk) * phi(u - 1im) / (1im * u * f1)) * du
        P2 += real(exp(-1im * u * lk) * phi(u) / (1im * u)) * du
    end
    P1 = 0.5 + P1 / pi
    P2 = 0.5 + P2 / pi
    return S * exp(-p.q * T) * P1 - K * exp(-p.r * T) * P2
end

"Total mass `dx dv sum p` of a forward density (exactly conserved by `heston_op(...; adjoint)`)."
heston_mass(A::AbstractVector, g::QTTGrid) =
    qtt_cell(g) * real(tt_dot(qtt_separable(g, x -> 1.0, v -> 1.0), A))

"`dx dv sum exp(a x + b v) p(x, v)`: the affine moment of a forward density (plan §3.3)."
heston_moment(A::AbstractVector, g::QTTGrid, a::Real, b::Real) =
    qtt_cell(g) * real(tt_dot(qtt_separable(g, x -> exp(a * x), v -> exp(b * v)), A))
