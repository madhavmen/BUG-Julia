# benchmarks/xx_efficiency/fig4_endtoend.jl -- plan Fig. 4: the complete order-3 BUG solver,
# L = 16 from the uncompressed domain wall to J T = 2, three settings at each dt:
#   full          adaptive DD+QR, time-weighted Krylov stop, recycling (`krylov_recycle = true`)
#   norecycle     the same without the recycle cache
#   conventional  fixed DD width and fixed Krylov depth (`dd_width`, `krylov_depth`, predeclared
#                 in XXE_CONV), fresh Krylov
# Records total wall time (median of XXE_REPS warmed repetitions of the whole run), eps_z, the exact
# state error (raw and phase-aligned) and fidelity, H1 applications, streamed probe columns and the
# largest bond. THE TIMING JOB: run alone (plan Sec. 3.4, cluster_compliance.md Sec. 2).
#
# Env (also common.jl): XXE_DTS (0.2,0.1,0.05,0.025), XXE_T (2.0), XXE_REPS (3),
#                       XXE_CONV (16:16 = dd_width:krylov_depth), XXE_TOLS (1e-7,1e-10,1e-12)

include(joinpath(@__DIR__, "common.jl"))

const DTS = parse.(Float64, split(get(ENV, "XXE_DTS", "0.2,0.1,0.05,0.025"), ","))
const TEND = parse(Float64, get(ENV, "XXE_T", "2.0"))
const REPS = parse(Int, get(ENV, "XXE_REPS", "3"))
const CONV = parse.(Int, split(get(ENV, "XXE_CONV", "16:16"), ":"))
const TB, TT, TE = parse.(Float64, split(get(ENV, "XXE_TOLS", "1e-7,1e-10,1e-12"), ","))
const ARMS = Dict("full" => (krylov_recycle = true,),
                  "norecycle" => (krylov_recycle = false,),
                  "conventional" => (krylov_recycle = false, dd_width = CONV[1],
                                     krylov_depth = CONV[2]))

basis = spin_xx_sector(L, 1)[2]
eps_z(psi, t) = sqrt(sum(abs2, spin_sz_profile(psi, 1) .- (real(diag(xx_open_correlation(L, t))) .- 0.5)) / L)

function run_arm(arm, dt)
    psi = domain_wall_state(L)
    acc = (n_h1 = 0, cols = 0, unmet = 0)
    n = round(Int, TEND / dt)
    for k in 1:n
        info = cbe_bug_step!(psi, MPO_XX, ComplexF64(-dt * im); tol_basis = TB, tol_trunc = TT,
                             tol_expv = TE, maxdim = 512, order = 3, ARMS[arm]...,
                             rng = MersenneTwister(SEEDS[1] + k))
        lg = info.ledger
        acc = (n_h1 = acc.n_h1 + lg.n_h1, cols = acc.cols + lg.n_probes + lg.n_holdout,
               unmet = acc.unmet + !lg.tolerance_met)
    end
    return psi, acc
end

# JIT warm-up of every arm on a short run, outside every timed region
for arm in keys(ARMS)
    psi = domain_wall_state(L)
    cbe_bug_step!(psi, MPO_XX, ComplexF64(-0.1im); tol_basis = TB, tol_trunc = TT, tol_expv = TE,
                  maxdim = 512, order = 3, ARMS[arm]...)
end

rows = Any[]
for dt in DTS, arm in ("full", "norecycle", "conventional")
    ts = Float64[]
    psi, acc = nothing, nothing
    for _ in 1:REPS
        GC.gc()
        t = @elapsed ((psi, acc) = run_arm(arm, dt))
        push!(ts, t)
    end
    e = xx_state_error(psi, TEND, basis)
    ez = eps_z(psi, TEND)
    r = (dt = dt, arm = arm, wall_s = sort(ts)[(REPS + 1) ÷ 2], walls = ts, err = e.err,
         err_phase = e.err_phase, fid = e.fid, eps_z = ez, n_h1 = acc.n_h1, stream_cols = acc.cols,
         unmet_steps = acc.unmet, maxbond = maximum(bond_dims(psi)),
         tols = (TB, TT, TE), conv = CONV, threads = Threads.nthreads())
    push!(rows, r)
    @printf("XXE-F4 dt=%.4f arm=%s wall=%.2f s eps_z=%.3e err=%.3e err_phase=%.3e n_h1=%d cols=%d unmet=%d maxbond=%d\n",
            dt, arm, r.wall_s, ez, e.err, e.err_phase, acc.n_h1, acc.cols, acc.unmet, r.maxbond)
end
open(joinpath(OUT, "raw_solver_errors.csv"), "w") do io
    println(io, "dt,arm,wall_s,err,err_phase,fid,eps_z,n_h1,stream_cols,unmet_steps,maxbond")
    for r in rows
        println(io, join((r.dt, r.arm, r.wall_s, r.err, r.err_phase, r.fid, r.eps_z, r.n_h1,
                          r.stream_cols, r.unmet_steps, r.maxbond), ","))
    end
end
open(joinpath(OUT, "fig4_endtoend.json"), "w") do io
    JSON.json(io, Dict("rows" => rows, "timing_job" => get(ENV, "XXE_TIMING", "0") == "1");
              allownan = true)
end
println("XXE-F4-DONE")
