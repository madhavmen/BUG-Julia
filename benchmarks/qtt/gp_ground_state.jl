# benchmarks/qtt/gp_ground_state.jl -- QTT plan §3.1: GP ground states by imaginary time.
# Cases: harmonic g = 0 (exact 2D Gaussian, E = 1 for omega = 1), trapped g = 100, rotating
# Omega = 0.8 g = 100 (Chen et al.; vortex lattice, no analytic state -- graded against the best
# converged run and its stationarity residual). Methods: BUG p = 2/3/4 on a uniform and a geometric
# schedule; the baselines QTT-TDVP2 imaginary time (uniform) and QTT gradient descent only when
# named in T3G_BASELINES (USER 10-30: BUG against the analytic result first, no gradient descent).
# Analytic grading: harmonic -> E - 1 on every step and the final state's infidelity against the
# exact Gaussian exp(-(x^2+y^2)/2) (continuum values; the grid adds O(dx^2)); trapped -> the
# Thomas-Fermi mu_TF = sqrt(g/pi), E_TF = 2 mu_TF / 3 (asymptotic in g, a reference not a gate).
#
# ENV: T3G_OUT, T3G_N (7), T3G_BOX (10), T3G_CASES ("harmonic,trapped,rotating"), T3G_STEPS (60),
#      T3G_H (0.1), T3G_H0 (0.01), T3G_Q (1.3), T3G_HMAX (1.0), T3G_MAXDIM (128), T3G_ORDERS ("2,3,4"),
#      T3G_BASELINES ("": none; "tdvp,gd" adds them), T3G_NORMALIZED (0: -H[psi] psi + renormalise)

include(joinpath(@__DIR__, "common.jl"))

const OUT = envs("T3G_OUT", "t3g_out")
const N = envi("T3G_N", 7)
const BOX = envf("T3G_BOX", 10.0)
const CASES = Symbol.(split(envs("T3G_CASES", "harmonic,trapped,rotating"), ","))
const NST = envi("T3G_STEPS", 60)
const MAXD = envi("T3G_MAXDIM", 128)
const ORDERS = round.(Int, envlist("T3G_ORDERS", "2,3,4"))
const BASELINES = Symbol.(filter(!isempty, split(envs("T3G_BASELINES", ""), ",")))
# USER 10-30: the real-time flow in imaginary time (density MPO rebuilt per stage) + renormalise;
# 1 steps the normalised gradient flow with mu[psi] inside instead
const NORMALIZED = envi("T3G_NORMALIZED", 0) == 1
mkpath(OUT)

g = qtt_grid(N, N; x = (-BOX, BOX), y = (-BOX, BOX))
params = Dict(:harmonic => (gnl = 0.0, Omega = 0.0), :trapped => (gnl = 100.0, Omega = 0.0),
              :rotating => (gnl = 100.0, Omega = 0.8))
# the analytic references (omega = 1): exact for g = 0, Thomas-Fermi for the trapped case
exactE(case) = case === :harmonic ? 1.0 :
               case === :trapped ? 2 / 3 * sqrt(params[case].gnl / pi) : NaN
exactmu(case) = case === :harmonic ? 1.0 :
                case === :trapped ? sqrt(params[case].gnl / pi) : NaN
const GAUSS = qtt_separable(g, x -> exp(-x^2 / 2), y -> exp(-y^2 / 2))
infidelity(A) = 1 - abs2(tt_dot(GAUSS, A)) / (tt_norm(GAUSS)^2 * tt_norm(A)^2)
hist = NamedTuple[]
summary = NamedTuple[]
for case in CASES
    pr = params[case]
    m = gp_model(g; gnl = pr.gnl, gamma = 0.0, mu = 0.0, omega = 1.0, Omega = pr.Omega)
    # start: a wide Gaussian, with a small off-centre charge-1 component to break the symmetry
    A0 = tt_lincomb([(1.0, qtt_separable(g, x -> exp(-x^2 / 4), y -> exp(-y^2 / 4))),
                     (0.1, qtt_separable(g, x -> (x - 0.5) * exp(-x^2 / 4), y -> exp(-y^2 / 4))),
                     (0.1im, qtt_separable(g, x -> exp(-x^2 / 4), y -> y * exp(-y^2 / 4)))])
    runs = Any[]
    for p in ORDERS
        push!(runs, (Symbol("bug$(p)_uniform"), p, :bug, uniform_schedule(envf("T3G_H", 0.1), NST)))
        push!(runs, (Symbol("bug$(p)_geometric"), p, :bug,
                     geometric_schedule(envf("T3G_H0", 0.01), envf("T3G_Q", 1.3),
                                        envf("T3G_HMAX", 1.0), NST)))
    end
    :tdvp in BASELINES &&
        push!(runs, (:tdvp_uniform, 2, :tdvp, uniform_schedule(envf("T3G_H", 0.1), NST)))
    :gd in BASELINES && push!(runs, (:gd, 0, :gd, nothing))
    Ex, mux = exactE(case), exactmu(case)
    for (name, p, meth, sched) in runs
        psi = qtt_mps(A0)
        t0 = time()
        lg = meth === :gd ?
             [merge(e, (accepted = true, rejected = 0)) for e in
              gp_gradient_descent!(psi, m; maxiter = 20 * NST, maxdim = MAXD)] :
             gp_ground_state!(psi, m, sched; method = meth, order = p, maxdim = MAXD,
                              normalized_flow = NORMALIZED)
        for e in lg
            push!(hist, merge((case = case, method = name, p = p), e,
                              (E_err = e.E - Ex, mu_err = e.mu - mux)))
        end
        acc = filter(e -> e.accepted, lg)
        vc = case === :rotating ? vortex_count(last(qtt_sample(qtt_cores(psi), g;
                                                               stride = max(1, 2^N ÷ 128)))) :
             (0, 0, Any[])
        push!(summary, (case = case, method = name, E = acc[end].E, E_exact = Ex,
                        E_err = acc[end].E - Ex, mu = acc[end].mu, mu_exact = mux,
                        mu_err = acc[end].mu - mux,
                        infidelity = case === :harmonic ? infidelity(qtt_cores(psi)) : NaN,
                        R_mu = acc[end].R_mu, chi = acc[end].chi,
                        chi_max = maximum(e.chi for e in lg), wall = time() - t0,
                        accepted = length(acc) - 1, rejected = count(e -> !e.accepted, lg),
                        vortices_plus = vc[1], vortices_minus = vc[2]))
        println(summary[end]); flush(stdout)
        write_csv(joinpath(OUT, "history.csv"), hist)
        write_csv(joinpath(OUT, "summary.csv"), summary)
    end
end
println("T3G-DONE")
