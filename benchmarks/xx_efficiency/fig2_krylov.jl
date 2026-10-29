# benchmarks/xx_efficiency/fig2_krylov.jl -- plan Fig. 2 on the Fixture M snapshots.
#
# One frozen CBE-augmented basis Q_CBE (production `dd_cbe_expand` at the same allocation), local
# operator H1_ex and seed K_ex per (case, side, h, tol); then the time-weighted stop (`_krylov_enrich`)
# against exactly `m` steps of the same recurrence (`fixed_depth = m`, m = 8, 16, 32), no recycling.
# y = the independent eps_K (a tightly converged expv, never timed); the pass threshold is the
# stage's own allowance `alloc.krylov` (the same for every arm). Also records the CBE-only eps_K, so
# a case the CBE frame already closes is visible as such.
#
# Env (also common.jl): XXE_CASES (4.0:16:8,4.0:8:8,4.0:8:5), XXE_HS (0.05,0.1,0.2)

include(joinpath(@__DIR__, "common.jl"))

const HS = parse.(Float64, split(get(ENV, "XXE_HS", "0.05,0.1,0.2"), ","))
const DEPTHS = (8, 16, 32)

rows = Any[]
for cs in cases("4.0:16:8,4.0:8:8,4.0:8:5")
    name = "t$(cs.t)_r$(cs.rank)_b$(cs.bond)"
    i = cs.bond
    for side in (:left, :right)
        psi = load_snapshot(cs.t, cs.rank)
        canonical!(psi, side === :left ? i : i + 1)
        f = RC._frame_from(psi[i], psi[i + 1])
        lch, rch = _envs(psi, MPO_XX, i, L)
        for h in HS, tol in TOLS
            tau = ComplexF64(-h * im)
            alloc = step_alloc(tol, h)
            ex = dd_cbe_expand(f, MPO_XX, i, lch, rch, side; alloc = alloc,
                               rng = MersenneTwister(SEEDS[1]))
            kl = _kl_stage(side, MPO_XX, i, psi[i], psi[i + 1], lch, rch, ex)
            kry = alloc.krylov
            e0 = eps_k(kl.Q0, kl.applyH, kl.seed, tau, side)
            base = (figure = "2", case = name, side = side, tol = tol, h = h, seed = SEEDS[1],
                    work_kind = "H1_applications", error_kind = "eps_K", target = kry)
            push!(rows, merge(base, (arm = "cbe_only", param = "", work = 0,
                                     columns = leg_dim(kl.Q0, side === :left ? 3 : 1), error = e0,
                                     pass = e0 <= kry, status = "", time_s = 0.0, bytes = 0)))
            for m in (0, DEPTHS...)
                (Q, st), t, b = timed() do
                    RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, alloc; fixed_depth = m)
                end
                e = eps_k(Q, kl.applyH, kl.seed, tau, side)
                status = m == 0 ? (st.met ? "met" : st.capped ? "capped" : "unmet") :
                         (st.n_h1 < m ? "breakdown" : "")
                push!(rows, merge(base, (arm = m == 0 ? "adaptive" : "fixed_m$m",
                                         param = m == 0 ? "" : "m=$m", work = st.n_h1,
                                         columns = leg_dim(Q, side === :left ? 3 : 1), error = e,
                                         pass = e <= kry, status = status, time_s = t,
                                         bytes = b)))
            end
            @printf("XXE-F2 %s %s h=%.3f tol=%.0e eps_cbe=%.2e allowance=%.2e adaptive n_h1=%d eps=%.2e\n",
                    name, side, h, tol, e0, kry, rows[end - length(DEPTHS)].work,
                    rows[end - length(DEPTHS)].error)
        end
    end
end

write_rows([Dict(String(k) => v for (k, v) in pairs(r)) for r in rows])
open(joinpath(OUT, "fig2_krylov.json"), "w") do io
    JSON.json(io, Dict("rows" => rows, "timing_job" => get(ENV, "XXE_TIMING", "0") == "1");
              allownan = true)
end
println("XXE-F2-ALLDONE $(length(rows)) rows")
