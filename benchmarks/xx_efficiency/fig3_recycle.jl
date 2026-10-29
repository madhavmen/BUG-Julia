# benchmarks/xx_efficiency/fig3_recycle.jl -- plan Fig. 3: Krylov recycling (R, `krylov_recycle`)
# against fresh local Krylov (F) over the three order-3 augmentation rounds, same adaptive DD-CBE
# and stop rule; F only has no recycle cache.
#
# Arms: F, R (`recycle_mode = :images`) and Rf (`:frame`, USER 10-29: recycling without the image
# cache). Per step and round: new one-site H applications and local-enrichment seconds. Round 1 is the
# half-sweeps (`ledger.n_h1` minus the frontier sweeps', time `t_kry`); rounds 2-3 are the frontier
# sweeps' `closure_log` entries (frontier j = 1, 2: their `n_h1`, which counts every image refresh,
# and `t`, which also holds the frontier CBE -- the same in both arms). R's transport seconds come
# from `recycle_report`. Accuracy guard: both arms against exact sector propagation of the same
# input amplitudes (`spin_ed_propagate`), as S5.
#
# Env (also common.jl): XXE_CASES (4.0:16:0,4.0:8:0; bond ignored), XXE_H (0.1), XXE_NSTEP (3),
#                       XXE_TOL (1e-5)

include(joinpath(@__DIR__, "common.jl"))

const H = parse(Float64, get(ENV, "XXE_H", "0.1"))
const NSTEP = parse(Int, get(ENV, "XXE_NSTEP", "3"))
const TOL = parse(Float64, get(ENV, "XXE_TOL", "1e-5"))

Hs, basis = spin_xx_sector(L, 1)
# R = the H-image cache; Rf = transported vectors only seed the frame, fresh Krylov (no refresh)
const ARMS = (("F", (krylov_recycle = false,)),
              ("R", (krylov_recycle = true, recycle_mode = :images)),
              ("Rf", (krylov_recycle = true, recycle_mode = :frame)))
rows, errs = Any[], Any[]
for cs in cases("4.0:16:0,4.0:8:0")
    name = "t$(cs.t)_r$(cs.rank)"
    snap = load_snapshot(cs.t, cs.rank)
    v0 = xx_sector_amplitudes(snap, basis)
    # JIT warm-up of every arm, outside every timed region
    for (_, kw) in ARMS
        cbe_bug_step!(copy(snap), MPO_XX, ComplexF64(-H * im); tol_basis = TOL, tol_trunc = 1e-10,
                      tol_expv = 1e-12, maxdim = 256, order = 3, kw...)
    end
    for (arm, kw) in ARMS
        psi, v = copy(snap), v0
        for step in 1:NSTEP
            rep = Ref{Any}(nothing)
            t0 = time()
            info = cbe_bug_step!(psi, MPO_XX, ComplexF64(-H * im); tol_basis = TOL,
                                 tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = 256, order = 3,
                                 kw..., recycle_report = rep,
                                 rng = MersenneTwister(SEEDS[1]))
            tstep = time() - t0
            v = spin_ed_propagate(Hs, v, H)
            err = norm(xx_sector_amplitudes(psi, basis) - v)
            cl = [e for e in info.ledger.closure_log if get(e, :enrich, false) && !get(e, :zero, false)]
            nfr = sum(e -> e.n_h1, cl; init = 0)
            tr = rep[] === nothing ? Dict(:left => Any[], :right => Any[]) : rep[]
            ttrans(rd) = sum(e -> e.round == rd ? e.t_transport : 0.0,
                             vcat(tr[:left], tr[:right]); init = 0.0)
            nrec(rd) = sum(e -> e.round == rd ? e.recycled : 0, vcat(tr[:left], tr[:right]); init = 0)
            base = (figure = "3", case = name, arm = arm, tol = TOL, h = H, seed = SEEDS[1],
                    work_kind = "H1_applications", error = err, error_kind = "state_err_vs_exact",
                    target = "", pass = "", bytes = "")
            push!(rows, merge(base, (side = "both", param = "step=$step,round=1",
                                     work = info.ledger.n_h1 - nfr, columns = "",
                                     status = info.ledger.tolerance_met ? "met" : "unmet",
                                     time_s = info.t_kry)))
            for e in cl
                rd = e.frontier + 1
                push!(rows, merge(base, (side = "both", param = "step=$step,round=$rd",
                                         work = e.n_h1, columns = nrec(rd),
                                         status = e.met ? "met" : "unmet",
                                         time_s = e.t, t_transport = ttrans(rd))))
            end
            push!(errs, (case = name, arm = arm, step = step, err = err, t_step = tstep,
                         n_h1 = info.ledger.n_h1, bonds = bond_dims(psi)))
            @printf("XXE-F3 %s %s step=%d err=%.3e n_h1=%d rounds=%s t=%.2f s maxbond=%d\n", name,
                    arm, step, err, info.ledger.n_h1,
                    string([r.work for r in rows[(end - length(cl)):end]]), tstep,
                    maximum(bond_dims(psi)))
        end
    end
end

write_rows([Dict(String(k) => v for (k, v) in pairs(r)) for r in rows])
open(joinpath(OUT, "fig3_recycle.json"), "w") do io
    JSON.json(io, Dict("rows" => rows, "steps" => errs, "warmed" => true,
                       "timing_job" => get(ENV, "XXE_TIMING", "0") == "1"); allownan = true)
end
println("XXE-F3-ALLDONE $(length(rows)) rows")
