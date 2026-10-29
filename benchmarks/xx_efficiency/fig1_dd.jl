# benchmarks/xx_efficiency/fig1_dd.jl -- plan Fig. 1 on the Fixture M snapshots.
#
# A: adaptive DD stopping against one-shot fixed oversampling, all through the SAME DD projection,
#    streamed kernel and QR (`_dd_select`). Arms: `adaptive` (production `dd_cbe_expand`, its own
#    initial width), `adaptive_b4` (the controller from width 4 in blocks of 4, the plan's spec),
#    `fixed_p` (`dd_cbe_expand(dd_width = k + p)`, k = the expansion's own growth target).
#    x = streamed probe columns incl. the production holdout; y = offline rho_DD.
# B: DD against one-sided projection at equal Gaussian width, same QR; x = selected columns.
#
# The target in rho_DD units is the production test `scale * resid <= sample` read back:
# rho_target = sample / (scale * ||G_DD||). The offline audit (G_DD) is never inside a timed
# region. Times are medians of three warmed repetitions; only a run inside the one timing job is
# authoritative (XXE_TIMING=1 marks it).
#
# Env (also common.jl): XXE_CASES (4.0:16:8,4.0:8:8,4.0:8:5,0.5:8:8), XXE_H (0.1)

include(joinpath(@__DIR__, "common.jl"))

const H = parse(Float64, get(ENV, "XXE_H", "0.1"))
const OVERS = (4, 8, 16)
const WIDTHS = (2, 4, 8, 16)

grown(ex, side) = side === :left ? ex.U_ex : ex.V_ex

"Panel A at one tolerance: every arm's streamed columns and rho_DD."
function panel_a!(rows, cs, name, bc, side, seed, tol, audit, k)
    f, lenv, renv, i = bc.frame, bc.lenv, bc.renv, cs.bond
    alloc = step_alloc(tol, H)
    target = alloc.sample / (alloc.scale * norm(audit.G))
    rho(Fr) = norm(audit.perpQ(Fr, audit.G)) / norm(audit.G)
    row(arm, param, work, cols, rh, status, t, b) =
        (figure = "1A", case = name, side = side, arm = arm, param = param, tol = tol, h = H,
         seed = seed, work = work, work_kind = "stream_cols", columns = cols, error = rh,
         error_kind = "rho_DD", target = target, pass = rh <= target, status = status,
         time_s = t, bytes = b)
    expand(; kw...) = timed() do
        kc, lg = KernelCounters(), RC.AdaptiveLedger()
        ex = dd_cbe_expand(f, MPO_XX, i, lenv, renv, side; alloc = alloc, ledger = lg, kc = kc,
                           rng = MersenneTwister(seed), kw...)
        (ex, kc, lg)
    end
    (ex, kc, lg), t, b = expand()
    push!(rows, row("adaptive", "", kc.n_stream_cols, lg.n_selected, rho(grown(ex, side)),
                    lg.tolerance_met ? "met" : "unmet", t, b))
    (Q, st, c), t, b = timed() do
        c = _dd_closures(f, MPO_XX, lenv, renv, side; i = i, rng = MersenneTwister(seed),
                         audit = false)
        Q, st = RC._dd_controller(c.frame, c.side, (w, bo) -> c.draw(4, bo), c.sketch, c.project,
                                  4, c.wcap, c.ncap, alloc, RC.AdaptiveLedger(); kc = c.kc)
        (Q, st, c)
    end
    push!(rows, row("adaptive_b4", "w0=4,block=4", c.kc.n_stream_cols, _rank(Q, c.fl),
                    rho_dd(audit, Q), st, t, b))
    for p in OVERS
        (ex, kc, lg), t, b = expand(; dd_width = k + p)
        rh = rho(grown(ex, side))
        push!(rows, row("fixed_p$p", "k=$k", kc.n_stream_cols, lg.n_selected, rh,
                        rh <= target ? "met" : "FAILED", t, b))
    end
end

"Panel B: DD against one-sided probes of the same Gaussian width through the same QR."
function panel_b!(rows, cs, name, bc, side, seed, audit)
    for w in WIDTHS, arm in ("DD", "one_sided")
        (Q, c), t, b = timed() do
            c = _dd_closures(bc.frame, MPO_XX, bc.lenv, bc.renv, side; i = cs.bond,
                             rng = MersenneTwister(seed), audit = false)
            Om = arm == "DD" ? c.draw(w, nothing) : one_sided_probe(c, w; rng = MersenneTwister(seed))
            Q = Om === nothing ? nothing :
                first(RC._dd_select(c.frame, c.project(c.sketch(Om)), side, c.kc))
            (Q, c)
        end
        push!(rows, (figure = "1B", case = name, side = side, arm = arm, param = "w=$w", tol = "",
                     h = H, seed = seed, work = c.kc.n_stream_cols, work_kind = "stream_cols",
                     columns = _rank(Q, c.fl), error = rho_dd(audit, Q), error_kind = "rho_DD",
                     target = "", pass = "", status = "", time_s = t, bytes = b))
    end
end

rows = Any[]
for cs in cases("4.0:16:8,4.0:8:8,4.0:8:5,0.5:8:8")
    bc = xx_bond_content(load_snapshot(cs.t, cs.rank), MPO_XX, cs.bond)
    name = "t$(cs.t)_r$(cs.rank)_b$(cs.bond)"
    trivial = bc.dd <= 1e-8 * bc.x          # snapshots.jl's label
    @printf("XXE-F1-CASE %s |X|=%.3e |G_DD|=%.3e |G_DK|=%.3e |G_KD|=%.3e rank_DD=%d%s\n", name,
            bc.x, bc.dd, bc.dk, bc.kd, bc.rank_dd, trivial ? " TRIVIAL" : "")
    trivial && continue                     # plan Sec. 2: never divide by a near-zero ||G_DD||
    f = bc.frame
    dmax = max(leg_dim(f.U0, 1), f.old_rank, leg_dim(f.V0, 3))
    k = max(ceil(Int, RC.ADAPTIVE_GROWTH * dmax) - f.old_rank, 1)
    for side in (:left, :right), seed in SEEDS
        audit = _dd_closures(f, MPO_XX, bc.lenv, bc.renv, side; i = cs.bond,
                             rng = MersenneTwister(seed))
        for tol in TOLS
            panel_a!(rows, cs, name, bc, side, seed, tol, audit, k)
        end
        panel_b!(rows, cs, name, bc, side, seed, audit)
    end
    @printf("XXE-F1-DONE %s rows=%d\n", name, length(rows))
end

write_rows([Dict(String(k) => v for (k, v) in pairs(r)) for r in rows])
open(joinpath(OUT, "fig1_dd.json"), "w") do io
    JSON.json(io, Dict("rows" => rows, "timing_job" => get(ENV, "XXE_TIMING", "0") == "1");
              allownan = true)
end
println("XXE-F1-ALLDONE $(length(rows)) rows")
