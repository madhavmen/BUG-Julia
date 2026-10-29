# benchmarks/qtt/heston_pilot.jl -- QTT plan §3.3: the 2D Heston pilots.
#
# BACKWARD: a regularised call payoff (e^x - K)_+ (softplus width T3H_EPS) priced to maturity T3H_T
# with rho != 0, against Heston's semi-analytic formula (`heston_call`) at S = 1 and several v0;
# delta, gamma and vega from the QTT derivative chains, at realistic maturities.
# FORWARD: the undiscounted density p_t = A_H' p from a narrow positive Gaussian; mass, positivity
# and the affine moments  sum e^{a x + b v} p_t = sum p0 exp(a x + B(t) v + C(t))  (Riccati, no
# discount) at several times, with the rank profile (it should rise AND fall).
#
# ENV: T3H_OUT, T3H_N (8), T3H_T (1.0), T3H_H (0.02), T3H_FT (5.0), T3H_FH (0.05), T3H_ORDER (3),
#      T3H_MAXDIM (128), T3H_EPS (0.02)

include(joinpath(@__DIR__, "common.jl"))

const OUT = envs("T3H_OUT", "t3h_out")
const N = envi("T3H_N", 8)
const ORD = envi("T3H_ORDER", 3)
const MAXD = envi("T3H_MAXDIM", 128)
mkpath(OUT)
p = HestonParams(0.03, 0.0, 1.5, 0.09, 0.5, -0.7)
println("Feller 2 kappa theta >= sigma^2: ", feller(p))

# ── backward pricing ──
g = qtt_grid(N, N; x = (-3.0, 3.0), y = (0.0, 1.0), periodic_x = false, periodic_y = false)
xs, vs = qtt_coords(g)
K = 1.0
eps_ = envf("T3H_EPS", 0.02)
softplus(z) = z > 30 ? z : log1p(exp(z))
u0 = qtt_separable(g, x -> eps_ * softplus((exp(x) - K) / eps_), v -> 1.0; tol = 1e-12)
fl = LinearFlow(heston_op(g, p))
h = envf("T3H_H", 0.02)
T = envf("T3H_T", 1.0)
psi = qtt_mps(u0)
brows = NamedTuple[]
t = 0.0
wall = @elapsed while t < T - 1e-12
    r = qtt_bug_step!(psi, fl, t, h; order = ORD, maxdim = MAXD, tol_trunc = 1e-10)
    global t += h
    push!(brows, (part = :backward, t = t, chi = r.chi, profile = join(bond_dims(psi), ";")))
end
U = qtt_cores(psi)
ix0 = argmin(abs.(xs))                                 # S = e^x = 1
D1 = qtt_embed(g; x = qtt_d(g, :x, 1; bc = :onesided))
D2 = qtt_embed(g; x = qtt_d(g, :x, 2; bc = :onesided))
Dv = qtt_embed(g; y = qtt_d(g, :y, 1; bc = :onesided))
ux = first(tt_compress(op_apply(D1, U)))
uxx = first(tt_compress(op_apply(D2, U)))
uv = first(tt_compress(op_apply(Dv, U)))
prows = NamedTuple[]
for v0 in (0.04, 0.09, 0.16)
    iv = argmin(abs.(vs .- v0))
    S = exp(xs[ix0])
    ref = heston_call(p, S, K, vs[iv], T)
    pu = real(qtt_value(U, g, ix0 - 1, iv - 1))
    push!(prows, (v0 = vs[iv], S = S, price_qtt = pu, price_ref = ref, err = abs(pu - ref),
                  delta = real(qtt_value(ux, g, ix0 - 1, iv - 1)) / S,
                  gamma = real(qtt_value(uxx, g, ix0 - 1, iv - 1) - qtt_value(ux, g, ix0 - 1, iv - 1)) / S^2,
                  vega_v = real(qtt_value(uv, g, ix0 - 1, iv - 1)), wall = wall))
    println(prows[end])
end
write_csv(joinpath(OUT, "pricing.csv"), prows)

# ── forward density ──
gf = qtt_grid(N, N; x = (-4.0, 4.0), y = (0.0, 1.0), periodic_x = false, periodic_y = false)
p0 = qtt_separable(gf, x -> exp(-(x / 0.1)^2 / 2), v -> exp(-((v - 0.09) / 0.02)^2 / 2); tol = 1e-12)
p0 = tt_scale(p0, 1 / heston_mass(p0, gf))
ffl = LinearFlow(heston_op(gf, p; adjoint = true))
ones_keep = [qtt_mps(qtt_separable(gf, x -> 1.0, v -> 1.0))]
pf = qtt_mps(p0)
fh = envf("T3H_FH", 0.05)
FT = envf("T3H_FT", 5.0)
frows = NamedTuple[]
tf = 0.0
moments = ((0.5, 0.0), (0.0, -1.0), (0.3, -0.5))
while tf < FT - 1e-12
    w = @elapsed r = qtt_bug_step!(pf, ffl, tf, fh; order = ORD, maxdim = MAXD, tol_trunc = 1e-10,
                                   keep = ones_keep)
    global tf += fh
    A = qtt_cores(pf)
    row = Dict{Symbol, Any}(:part => :forward, :t => tf, :mass => heston_mass(A, gf),
                            :min_rel => minimum(real, last(qtt_sample(A, gf; stride = max(1, 2^N ÷ 128)))) /
                                        maximum(real, last(qtt_sample(A, gf; stride = max(1, 2^N ÷ 128)))),
                            :chi => r.chi, :profile => join(bond_dims(pf), ";"), :wall => w)
    for (k, (a, b)) in enumerate(moments)
        B, C = heston_riccati(p, a, b, tf; nsteps = 4000, discount = false)
        pred = qtt_cell(gf) * real(tt_dot(p0, heston_affine_field(gf, a, B, C)))
        row[Symbol("moment$(k)_qtt")] = heston_moment(A, gf, a, b)
        row[Symbol("moment$(k)_ref")] = pred
    end
    push!(frows, (; sort(collect(row); by = first)...))
    println(frows[end]); flush(stdout)
end
write_csv(joinpath(OUT, "forward.csv"), frows)
write_csv(joinpath(OUT, "backward_ranks.csv"), brows)
println("T3H-DONE")
