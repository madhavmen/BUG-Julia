# benchmarks/qtt/gp_ground_state.jl -- QTT plan §3.1: GP ground states by imaginary time.
# Cases: harmonic g = 0 (exact 2D Gaussian, E = 1 for omega = 1), trapped g = 100, rotating
# Omega = 0.8 g = 100 (Chen et al.; vortex lattice, no analytic state -- graded against the best
# converged run and its stationarity residual). Methods: BUG p = 2/3/4 on a uniform and a geometric
# schedule, QTT-TDVP2 imaginary time (uniform), QTT gradient descent. Compared on energy and R_mu
# reached per WALL second, with accepted/rejected steps; vortex charges for the rotating case.
#
# ENV: T3G_OUT, T3G_N (7), T3G_BOX (10), T3G_CASES ("harmonic,trapped,rotating"), T3G_STEPS (60),
#      T3G_H (0.1), T3G_H0 (0.01), T3G_Q (1.3), T3G_HMAX (1.0), T3G_MAXDIM (128), T3G_ORDERS ("2,3,4")

include(joinpath(@__DIR__, "common.jl"))

const OUT = envs("T3G_OUT", "t3g_out")
const N = envi("T3G_N", 7)
const BOX = envf("T3G_BOX", 10.0)
const CASES = Symbol.(split(envs("T3G_CASES", "harmonic,trapped,rotating"), ","))
const NST = envi("T3G_STEPS", 60)
const MAXD = envi("T3G_MAXDIM", 128)
const ORDERS = round.(Int, envlist("T3G_ORDERS", "2,3,4"))
mkpath(OUT)

g = qtt_grid(N, N; x = (-BOX, BOX), y = (-BOX, BOX))
params = Dict(:harmonic => (gnl = 0.0, Omega = 0.0), :trapped => (gnl = 100.0, Omega = 0.0),
              :rotating => (gnl = 100.0, Omega = 0.8))
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
    push!(runs, (:tdvp_uniform, 2, :tdvp, uniform_schedule(envf("T3G_H", 0.1), NST)))
    for (name, p, meth, sched) in runs
        psi = qtt_mps(A0)
        t0 = time()
        lg = gp_ground_state!(psi, m, sched; method = meth, order = p, maxdim = MAXD)
        for e in lg
            push!(hist, merge((case = case, method = name, p = p), e))
        end
        acc = filter(e -> e.accepted, lg)
        vc = case === :rotating ? vortex_count(last(qtt_sample(qtt_cores(psi), g;
                                                               stride = max(1, 2^N ÷ 128)))) :
             (0, 0, Any[])
        push!(summary, (case = case, method = name, E = acc[end].E, mu = acc[end].mu,
                        R_mu = acc[end].R_mu, chi = acc[end].chi, wall = time() - t0,
                        accepted = length(acc) - 1, rejected = count(e -> !e.accepted, lg),
                        vortices_plus = vc[1], vortices_minus = vc[2]))
        println(summary[end]); flush(stdout)
        write_csv(joinpath(OUT, "history.csv"), hist)
        write_csv(joinpath(OUT, "summary.csv"), summary)
    end
    psi = qtt_mps(A0)
    t0 = time()
    lg = gp_gradient_descent!(psi, m; maxiter = 20 * NST, maxdim = MAXD)
    for e in lg
        push!(hist, merge((case = case, method = :gd, p = 0, accepted = true), e))
    end
    vc = case === :rotating ? vortex_count(last(qtt_sample(qtt_cores(psi), g;
                                                           stride = max(1, 2^N ÷ 128)))) : (0, 0, Any[])
    push!(summary, (case = case, method = :gd, E = lg[end].E, mu = lg[end].mu, R_mu = lg[end].R_mu,
                    chi = lg[end].chi, wall = time() - t0, accepted = length(lg) - 1, rejected = 0,
                    vortices_plus = vc[1], vortices_minus = vc[2]))
    println(summary[end])
    case === :harmonic && println("harmonic exact E = 1 (continuum); best = ",
                                  minimum(s.E for s in summary if s.case === case))
    write_csv(joinpath(OUT, "history.csv"), hist)
    write_csv(joinpath(OUT, "summary.csv"), summary)
end
println("T3G-DONE")
