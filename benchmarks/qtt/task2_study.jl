# benchmarks/qtt/task2_study.jl -- QTT plan Task 2 at study sizes: V04 spatial convergence, V05
# local/global orders of the WHOLE BUG update (p = 2/3/4, h in {0.1, 0.05, 0.025, 0.0125}, rank-rich
# starts), V06 one-at-a-time tolerance sweeps, V07 TDVP2 backward amplification (lagged /
# consistent / split) against h and gamma. Every error is against an INDEPENDENT reference: sparse
# grid RK4 of the same semi-discrete system (time error) or the exact Riccati solution (spatial).
#
# ENV: T2_PART = orders | spatial | tolerances | tdvp | all; T2_OUT; T2_N (bits per coordinate, 6);
#      T2_HS ("0.1,0.05,0.025,0.0125"); T2_T (0.4); T2_MAXDIM (128).

include(joinpath(@__DIR__, "common.jl"))

const PART = envs("T2_PART", "all")
const OUT = envs("T2_OUT", "t2_out")
const N = envi("T2_N", 6)
const HS = envlist("T2_HS", "0.1,0.05,0.025,0.0125")
const TEND = envf("T2_T", 0.4)
const MAXD = envi("T2_MAXDIM", 128)
mkpath(OUT)

run_arm!(step!, A0, h, n) = (psi = qtt_mps(A0); for k in 0:(n - 1); step!(psi, k * h, h); end; psi)

# ── the problems: (name, grid, initial cores, step constructor(arm, p), reference(T)) ──

function problems()
    P = Any[]
    gg = qtt_grid(N, N; x = (-8.0, 8.0), y = (-8.0, 8.0))
    A0 = rank_rich_state(gg)
    ml = gp_model(gg; gnl = 0.0, gamma = 0.1, mu = 0.0, omega = 1.0)
    push!(P, (name = :gp_linear, g = gg, A0 = A0, flow = GPFlow(ml), model = ml,
              ref = T -> gp_dense_rk4(ml, grid_vec(A0, gg), 0.0, T; dt = 2e-4)))
    mn = gp_model(gg; gnl = 1.0, gamma = 0.1, mu = 0.5, omega = 1.0)
    push!(P, (name = :gp_nonlinear, g = gg, A0 = A0, flow = GPFlow(mn), model = mn,
              ref = T -> gp_dense_rk4(mn, grid_vec(A0, gg), 0.0, T; dt = 2e-4)))
    gh = _heston_grid(N, N)
    u0 = heston_affine_field(gh, 0.7, -0.5, 0.0)
    Ah = heston_sparse(gh, _HP)
    push!(P, (name = :heston_affine, g = gh, A0 = u0, flow = LinearFlow(heston_op(gh, _HP)),
              model = nothing, ref = T -> linear_rk4(Ah, grid_vec(u0, gh), T; dt = 2e-5)))
    return P
end

function err_vs(psi, pr, ref)
    v = grid_vec(qtt_cores(psi), pr.g)
    return norm(v - ref) / norm(ref)
end

rows = NamedTuple[]

if PART in ("orders", "all")
    for pr in problems()
        refT = pr.ref(TEND)
        for p in (2, 3, 4), h in HS
            n = round(Int, TEND / h)
            t0 = time()
            info = Any[]
            psi = run_arm!((ps, t, hh) -> push!(info, qtt_bug_step!(ps, pr.flow, t, hh; order = p,
                                                       maxdim = MAXD, tol_basis = 1e-9,
                                                       tol_trunc = 1e-11, tol_stage = 1e-11)),
                           pr.A0, h, n)
            eg = err_vs(psi, pr, refT)
            # local error: ONE step from the rank-rich start
            ref1 = pr.ref(h)
            p1 = run_arm!((ps, t, hh) -> qtt_bug_step!(ps, pr.flow, t, hh; order = p, maxdim = MAXD,
                                                       tol_basis = 1e-9, tol_trunc = 1e-11,
                                                       tol_stage = 1e-11), pr.A0, h, 1)
            el = err_vs(p1, pr, ref1)
            push!(rows, (part = :orders, problem = pr.name, arm = Symbol("bug$p"), p = p, h = h,
                         err_global = eg, err_local = el, wall = time() - t0,
                         chi = maximum(bond_dims(psi)), newton = sum(i.stats[end].newton for i in info),
                         gmres = sum(i.stats[end].gmres for i in info),
                         converged = all(i.converged for i in info)))
            println(rows[end]); flush(stdout)
        end
        if pr.model !== nothing && pr.model.g != 0                    # the lagged ablations
            for h in HS
                n = round(Int, TEND / h)
                t0 = time()
                psi = run_arm!((ps, t, hh) -> gp_bug_step!(ps, pr.model, t, hh; tol_basis = 1e-9,
                                                           tol_trunc = 1e-11, maxdim = MAXD),
                               pr.A0, h, n)
                push!(rows, (part = :orders, problem = pr.name, arm = :frozen_density, p = 1, h = h,
                             err_global = err_vs(psi, pr, refT), err_local = NaN,
                             wall = time() - t0, chi = maximum(bond_dims(psi))))
                for dens in (:lagged, :consistent)
                    t0 = time()
                    rs = Any[]
                    psi = run_arm!((ps, t, hh) -> push!(rs, qtt_tdvp2_step!(ps, pr.flow, t, hh;
                                                           density = dens, maxdim = MAXD)),
                                   pr.A0, h, n)
                    push!(rows, (part = :orders, problem = pr.name, arm = Symbol("tdvp2_", dens),
                                 p = 2, h = h, err_global = err_vs(psi, pr, refT), err_local = NaN,
                                 wall = time() - t0, chi = maximum(bond_dims(psi)),
                                 A_max = maximum(r.A_max for r in rs)))
                    println(rows[end]); flush(stdout)
                end
            end
        end
        write_csv(joinpath(OUT, "orders.csv"), rows)
    end
end

if PART in ("spatial", "all")
    srows = NamedTuple[]
    a, b, tau = 0.7, -0.5, 0.25
    B, C = heston_riccati(_HP, a, b, tau)
    for n in (6, 7, 8)
        g = _heston_grid(n, n)
        u0 = heston_affine_field(g, a, b, 0.0)
        psi = run_arm!((ps, t, hh) -> qtt_bug_step!(ps, LinearFlow(heston_op(g, _HP)), t, hh;
                                                    order = 4, maxdim = MAXD, tol_trunc = 1e-12),
                       u0, 0.0125, round(Int, tau / 0.0125))
        ex = grid_vec(heston_affine_field(g, a, B, C), g)
        push!(srows, (part = :spatial, problem = :heston_affine, n = n,
                      err = norm(grid_vec(qtt_cores(psi), g) - ex) / norm(ex)))
        gg = qtt_grid(n, n; x = (-8.0, 8.0), y = (-8.0, 8.0))
        m = gp_model(gg; gnl = 0.0, gamma = 0.1, mu = 0.0, omega = 1.0)
        A0 = qtt_separable(gg, x -> phi_osc(0, x) + 0.25 * phi_osc(1, x), y -> phi_osc(0, y))
        pg = run_arm!((ps, t, hh) -> qtt_bug_step!(ps, GPFlow(m), t, hh; order = 4, maxdim = MAXD,
                                                   tol_trunc = 1e-12), A0, 0.0125, 20)
        xs, ys = qtt_coords(gg)
        T = 0.25
        an = vec([(phi_osc(0, x) * exp((-1im - 0.1) * 1 * T) +
                   0.25 * phi_osc(1, x) * exp((-1im - 0.1) * 2 * T)) * phi_osc(0, y)
                  for x in xs, y in ys])
        push!(srows, (part = :spatial, problem = :gp_oscillator, n = n,
                      err = norm(grid_vec(qtt_cores(pg), gg) - an) / norm(an)))
        println(srows[end - 1], "\n", srows[end]); flush(stdout)
    end
    write_csv(joinpath(OUT, "spatial.csv"), srows)
end

if PART in ("tolerances", "all")
    trows = NamedTuple[]
    pr = problems()[2]
    h = 0.05
    n = round(Int, TEND / h)
    refT = pr.ref(TEND)
    base = (tol_basis = 1e-8, tol_stage = 1e-10, tol_trunc = 1e-10)
    for (knob, vals) in ((:tol_basis, (1e-4, 1e-6, 1e-8, 1e-10)), (:tol_stage, (1e-6, 1e-8, 1e-10)),
                         (:tol_trunc, (1e-6, 1e-8, 1e-10))), v in vals
        kw = merge(base, NamedTuple{(knob,)}((v,)))
        infos = Any[]
        t0 = time()
        psi = run_arm!((ps, t, hh) -> push!(infos, qtt_bug_step!(ps, pr.flow, t, hh; order = 3,
                                                                  maxdim = MAXD, kw...)),
                       pr.A0, h, n)
        push!(trows, (part = :tolerances, knob = knob, value = v, err = err_vs(psi, pr, refT),
                      chi = maximum(bond_dims(psi)), wall = time() - t0,
                      capped = any(i.rank_capped for i in infos),
                      converged = all(i.converged for i in infos)))
        println(trows[end]); flush(stdout)
    end
    write_csv(joinpath(OUT, "tolerances.csv"), trows)
end

if PART in ("tdvp", "all")
    drows = NamedTuple[]
    gg = qtt_grid(N, N; x = (-8.0, 8.0), y = (-8.0, 8.0))
    A0 = rank_rich_state(gg)
    for gam in (0.1, 1.0), gn in (0.0, 1.0)
        m = gp_model(gg; gnl = gn, gamma = gam, mu = 0.5, omega = 1.0)
        refT = gp_dense_rk4(m, grid_vec(A0, gg), 0.0, TEND; dt = 2e-4)
        pr = (g = gg,)
        for h in HS
            n = round(Int, TEND / h)
            for arm in (:tdvp2_lagged, :tdvp2_consistent, :split, :bug2, :bug4)
                arm === :tdvp2_consistent && gn == 0 && continue
                t0 = time()
                rs = Any[]
                psi = run_arm!((ps, t, hh) -> push!(rs,
                    arm === :tdvp2_lagged ? qtt_tdvp2_step!(ps, GPFlow(m), t, hh; maxdim = MAXD) :
                    arm === :tdvp2_consistent ? qtt_tdvp2_step!(ps, GPFlow(m), t, hh;
                                                                density = :consistent, maxdim = MAXD) :
                    arm === :split ? qtt_split_tdvp_step!(ps, m, t, hh; maxdim = MAXD) :
                    qtt_bug_step!(ps, GPFlow(m), t, hh; order = arm === :bug2 ? 2 : 4,
                                  maxdim = MAXD, tol_basis = 1e-8, tol_trunc = 1e-10)), A0, h, n)
                push!(drows, (part = :tdvp, gamma = gam, g = gn, h = h, arm = arm,
                              err = err_vs(psi, pr, refT),
                              A_max = hasproperty(rs[1], :A_max) ? maximum(r.A_max for r in rs) : NaN,
                              norm_ratio = norm(psi) / tt_norm(A0), wall = time() - t0,
                              chi = maximum(bond_dims(psi))))
                println(drows[end]); flush(stdout)
            end
        end
    end
    write_csv(joinpath(OUT, "tdvp.csv"), drows)
end
println("T2-DONE")
