# benchmarks/qtt/diag_heston_step.jl -- qt1d PH: the Heston forward production smoke (P_N = 5)
# rejected its t = 0 step-doubling checks with e_hat GROWING as h halved (0.013, 0.002, 0.74, 0.78)
# and an unchecked step later took the mass from 1.0 to 5.55, although the Galerkin step conserves
# it exactly when the ones state is in the frames' span. This driver replays those attempts from one
# state (copied as production.jl does) and grades every step against the dense exact exp(h H):
# frame misses of the input and of the kept ones state, frame isometry, mass, and whether the state
# or the kept state is mutated or a repeated step differs (hidden history).
# ENV: DH_N (5), DH_OUT, DH_ORDERS (2,3,4: the nonlinear-BUG orders), DH_HS (0.05,...,0.0015625)

include(joinpath(@__DIR__, "common.jl"))

const N = envi("DH_N", 5)
const OUT = envs("DH_OUT", "diag_heston")
const ORDERS = Int.(envlist("DH_ORDERS", "2,3,4"))
const HS = envlist("DH_HS", "0.05,0.025,0.0125,0.00625,0.003125,0.0015625")
mkpath(OUT)

# the production.jl heston_forward problem, verbatim
g = qtt_grid(N, N; x = (-4.0, 4.0), y = (0.0, 1.0), periodic_x = false, periodic_y = false)
p = HestonParams(0.03, 0.0, 1.5, 0.09, 0.5, -0.7)
Hf = heston_op(g, p; adjoint = true)
fl = LinearFlow(Hf)
A0 = qtt_separable(g, x -> exp(-(x / 0.1)^2 / 2), v -> exp(-((v - 0.09) / 0.02)^2 / 2))
A0 = tt_scale(A0, 1 / heston_mass(A0, g))
keep = [qtt_mps(qtt_separable(g, x -> 1.0, v -> 1.0))]

# ── the semi-discrete operator itself ──
Hd = op_dense(Hf)
n = size(Hd, 1)
ev = eigvals(Matrix(Hd))
println("DENSE n = $n  ||H||_1 = $(opnorm(Matrix(Hd), 1))  max Re(lambda) = $(maximum(real, ev))  " *
        "#Re>1e-8 = $(count(l -> real(l) > 1e-8, ev))  ||H' 1|| = $(norm(transpose(Hd) * ones(n)))")
println("DENSE top eigenvalues by Re: ", sort(ev; by = real, rev = true)[1:min(6, n)])
# the numerical range's right edge: growth a Galerkin projection can show although exp(tH) decays
println("DENSE max eig of (H + H')/2 = $(maximum(eigvals(Hermitian(Matrix(Hd + Hd') / 2))))")
v0 = vec(tt_dense(A0))
exact(h) = exp(Matrix(h .* Hd)) * v0
dense_err(ps, h) = norm(vec(tt_dense(qtt_cores(ps))) .- exact(h)) / norm(exact(h))
mass(ps) = heston_mass(qtt_cores(ps), g)
function dist(a, b)
    A, B = qtt_cores(a), qtt_cores(b)
    return tt_norm(tt_add(A, tt_scale(B, -1.0))) / max(tt_norm(B), eps())
end

psi = qtt_mps(A0)
C0, K0 = deepcopy(qtt_cores(psi)), deepcopy(qtt_cores(keep[1]))
step!(ps, t, h, ord) = qtt_bug_step!(ps, fl, t, h; order = ord, maxdim = 64, tol_basis = 1e-6,
                                     tol_trunc = 1e-8, tol_stage = 1e-10, keep = keep)
rows = NamedTuple[]
first_full = Dict{Int, Any}()
for ord in ORDERS, h in HS
    full, half = copy(psi), copy(psi)
    r1 = step!(full, 0.0, h, ord)
    r2 = step!(half, 0.0, h / 2, ord)
    r3 = step!(half, h / 2, h / 2, ord)
    haskey(first_full, ord) || (first_full[ord] = (h, deepcopy(qtt_cores(full))))
    rs = (r1, r2, r3)
    row = (order = ord, h = h, e_hat = dist(full, half) / (1 - 2.0^(-ord)),
           err_full = dense_err(full, h), err_half = dense_err(half, h),
           mass_full = mass(full), mass_half = mass(half),
           miss_input = maximum(r.miss_input for r in rs), miss_keep = maximum(r.miss_keep for r in rs),
           frame_defect = maximum(r.frame_defect for r in rs),
           fallback = sum(r.basis_fallback for r in rs),
           frames = join((join(r.frame_width, "x") for r in rs), "|"),
           newton = sum(sum(s.newton for s in r.stats) for r in rs),
           stage_res = maximum(r.stats[end].residual for r in rs),
           converged = all(r.converged for r in rs), chi_full = r1.chi, chi_half = r3.chi,
           psi_mutated = qtt_cores(psi) != C0, keep_mutated = qtt_cores(keep[1]) != K0)
    push!(rows, row)
    println(row); flush(stdout)
    write_csv(joinpath(OUT, "diag_heston_step.csv"), rows)
end
# hidden history: the first full step of each order, repeated after all the others
# by key: Pair's isless evaluates the value comparison too (qt1h: isless on core arrays)
for (ord, (h, C)) in sort(collect(first_full); by = first)
    full = copy(psi)
    step!(full, 0.0, h, ord)
    println("HISTORY order $ord h $h: repeat bitwise = $(qtt_cores(full) == C)  " *
            "distance = $(tt_norm(tt_add(qtt_cores(full), tt_scale(C, -1.0))) / tt_norm(C))")
end
println("DIAG-DONE")
