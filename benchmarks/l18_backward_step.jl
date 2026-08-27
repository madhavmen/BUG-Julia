# THE BACKWARD SUBSTEP: WHY TDVP DEGRADES ON A DISSIPATIVE GENERATOR AND CBE-BUG DOES NOT.
#
#   julia -t 2 --project=. benchmarks/l18_backward_step.jl
#   MODELS=xx_open SITES=8 TMAX=1 DTS=0.1,0.4 julia -t 2 --project=. benchmarks/l18_backward_step.jl
#
# ── THE STRUCTURAL FACT, WHICH IS NOT A MEASUREMENT ──────────────────────────────────────────
#
# Both TDVP arms undo their own double counting by evolving BACKWARDS in time. Verbatim:
#
#   tdvp2_baseline.jl:107    expv(... apply_h_two_site ...,  tau, Theta)     # forward  tau/2
#   tdvp2_baseline.jl:124    expv(... apply_one_site  ..., -tau, M)          # BACKWARD tau/2
#   tdvp2_baseline.jl:136    expv(... apply_one_site  ..., -tau, M)          # BACKWARD tau/2
#   tdvp_cbe1s.jl:131,147    expv(... apply_one_site  ...,  half, psi[j])    # forward  tau/2
#   tdvp_cbe1s.jl:163,196    expv(... apply_zero_site ..., -half, C)         # BACKWARD tau/2
#
# and `cbe_bug.jl` has exactly ONE evolution call in the whole step:
#
#   cbe_bug.jl:844           expv(... apply_zero_site ...,  tau, S0)         # forward, and that
#                                                                           # is all of them
#
# ⛔ SO THE CLAIM IS NOT "tdvp2 vs BUG" -- IT IS "BOTH TDVP VARIANTS vs BUG". `tdvp_cbe1s` carries
# the same backward step, in the zero-site tensor rather than the one-site tensor. Predicting the
# effect for `tdvp2` alone would be predicting half of it.
#
# WHY IT MATTERS ONLY WHEN THE GENERATOR IS DISSIPATIVE. For a closed model `tau = -i*dt` is
# IMAGINARY, so `exp(-tau H)` is just the adjoint rotation: unitary, norm preserving, and the
# backward step is as stable as the forward one. For the fused Lindbladian `tau = dt` is REAL and
# the generator's spectrum sits in the LEFT half plane, so `exp(-tau L)` amplifies by
# `exp(+|Re lambda| dt/2)`. The backward substep is ANTI-DISSIPATIVE by construction.
#
# ── THE WITNESSES ARE EXACT AND NEED NO REFERENCE STATE, WHICH IS THE POINT ──────────────────
#
# At L = 18, d = 4 there is no exact solution to compare against -- and using `tdvp2` as the
# reference would beg the entire question. Both witnesses below are CONSERVATION LAWS, so a
# violation is an error in absolute terms with nothing to calibrate against:
#
#   `amp`  = max over steps of `||psi_after|| / ||psi_before||`, taken on the RAW step output
#            before any shift or trace correction. ⛔ DEPHASING IS A UNITAL CHANNEL, so the purity
#            `Tr rho^2 = || |rho>> ||^2` is NON-INCREASING under the exact dynamics. `amp > 1` is
#            therefore a PROVABLE violation, not a tolerance question, and it is exactly what an
#            anti-dissipative substep produces.
#   `dtr`  = max over steps of `|Tr rho - 1|`, read AFTER the zero-body shift and BEFORE
#            `restore_trace!`. `Tr rho = 1` holds for any Lindbladian. ⚠ THIS IS WHY `_open` now
#            exposes `shift!` and `restore!` separately: calling the bundled `post!` first destroys
#            the number being measured, which is how this witness reads a spurious zero.
#
# ⚠ AND `xx` (CLOSED, `:U1`) IS RUN AS THE CONTROL. The mechanism above predicts NO amplification
# there, because the backward step is unitary. If the closed control degrades too, the cause is not
# the backward step and this whole file is measuring something else -- a bug, or the Krylov budget.
# A one-sided experiment that only ran the open models could not tell those apart.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))

const OUT     = joinpath(@__DIR__, "results")
const MODELS  = split(get(ENV, "MODELS", "xx,xx_open,square_open"), ",")
const SITES   = parse(Int,     get(ENV, "SITES", "18"))
const TMAX    = parse(Float64, get(ENV, "TMAX",  "2.0"))
const CAP     = parse(Int,     get(ENV, "CAP",   "48"))
const MAXITER = parse(Int,     get(ENV, "MAXITER", "8"))
# ⛔ THE dt LADDER IS THE INDEPENDENT VARIABLE. The backward step's amplification per step is
# `exp(+|Re lambda| dt/2)`, so the effect must GROW with dt and vanish as dt -> 0. A single dt
# cannot distinguish "the backward step is unstable" from "this arm is just less accurate".
const DTS     = [parse(Float64, s) for s in split(get(ENV, "DTS", "0.025,0.05,0.1,0.2,0.4"), ",")]
const GAMMAS  = [parse(Float64, s) for s in split(get(ENV, "GAMMAS", "0.10"), ",")]

function arm_closures(S, cap)
    herm = (S.d == 2)
    return [
        ("tdvp2", (p, tau) -> tdvp2_step!(p, S.W, tau; maxdim = cap, trunc_thresh = 1e-10,
                                          maxiter = MAXITER, hermitian = herm)),
        ("tdvp_cbe1s", (p, tau) -> tdvp_cbe1s_step!(p, S.W, tau; maxdim = cap,
                                                    trunc_thresh = 1e-10, maxiter = MAXITER,
                                                    hermitian = herm)),
        ("cbe_bug", (p, tau) -> cbe_bug_step!(p, S.W, tau; exact = false, dex = 8, dover = 4,
                                              comp_ratio = 1.0, krylov_basis = 3,
                                              krylov_tol = 0.0, parallel = true,
                                              maxdim = cap, trunc_thresh = 1e-10,
                                              maxiter = MAXITER, hermitian = herm)),
    ]
end

"""
One trajectory, returning the two conservation-law witnesses plus cost.

⚠ A STEP THAT THROWS OR PRODUCES A NON-FINITE STATE IS A RESULT, NOT A CRASH. It is the loudest
form of the effect being measured, so it is caught, recorded as the step it died on, and the
remaining ladder still runs.
"""
function run_arm(S, arm, step!, dt, nsteps)
    psi = copy(S.psi0)
    amp = 0.0; dtr = 0.0; kry = 0; wall = 0.0; died = 0
    for k in 1:nsteps
        n_before = norm(copy(psi))
        t0 = time_ns()
        local info
        try
            info = step!(psi, S.tau)
        catch e
            died = k
            @printf("        %s threw at step %d: %s\n", arm, k,
                    sprint(showerror, e)[1:min(end, 120)])
            break
        end
        wall += (time_ns() - t0) / 1e9
        kry += info.krylov_dims
        n_after = norm(copy(psi))
        if !isfinite(n_after) || n_after == 0
            died = k
            @printf("        %s produced a non-finite state at step %d (norm = %g)\n",
                    arm, k, n_after)
            break
        end
        amp = max(amp, n_after / max(n_before, eps()))
        # ⛔ THE TRACE IS READ BETWEEN THE SHIFT AND THE RESTORE. See the header.
        if S.d == 4
            S.shift!(psi)
            dtr = max(dtr, abs(real(trace_rho(S.imps, psi, S.L; at = 0)) - 1.0))
            S.restore!(psi)
        end
    end
    prof = died == 0 ? S.profile(psi) : Float64[]
    return (amp = amp, dtr = dtr, kry = kry, wall = wall, died = died,
            chi = maximum(bond_dims(psi); init = 0), prof = prof)
end

function main()
    mkpath(OUT)
    csv = open(joinpath(OUT, "l18_backward_step.csv"), "w")
    println(csv, "model,d,arm,L,cap,gamma,dt,nsteps,amp_max,trace_err,krylov,secs," *
                 "maxbond,died_at,dev_vs_bug")
    flush(csv)

    @printf("L=%d  tmax=%.2f  cap=%d  maxiter=%d\n", SITES, TMAX, CAP, MAXITER)
    @printf("dt ladder = %s   gammas = %s\n", string(DTS), string(GAMMAS))
    @printf("julia threads = %d, BLAS threads = %d\n\n",
            Threads.nthreads(), BLAS.get_num_threads())
    flush(stdout)

    for nm in MODELS, g in GAMMAS
        S0 = setup(nm; sites = SITES, dt = DTS[1], gamma = g, cap = CAP)
        @printf("MODEL %-12s %s   (d = %d, MPO dim %d)%s\n", nm, S0.label, S0.d,
                maximum(mpo_virtual_dims(S0.W)),
                S0.d == 2 ? "   <- CLOSED CONTROL: no amplification predicted" : "")
        @printf("   gamma = %.2f\n", g)
        # ⛔ WARMUP. Without it the first arm of the first dt absorbs both TDVP arms' JIT and the
        # `secs` column -- which is HALF THE CLAIM this file makes -- is fiction.
        for (arm, step!) in arm_closures(S0, CAP)
            try
                step!(copy(S0.psi0), S0.tau)
            catch e
                @printf("      warmup: %s threw -- %s\n", arm,
                        sprint(showerror, e)[1:min(end, 160)])
            end
        end
        println("      dt      arm            amp_max    |Tr-1|     krylov      secs   chi   dev_vs_bug")
        flush(stdout)
        for dt in DTS
            S = setup(nm; sites = SITES, dt = dt, gamma = g, cap = CAP)
            nsteps = max(1, round(Int, TMAX / dt))
            res = Dict{String, Any}()
            for (arm, step!) in arm_closures(S, CAP)
                res[arm] = run_arm(S, arm, step!, dt, nsteps)
            end
            for (arm, _) in arm_closures(S, CAP)
                r = res[arm]
                # ⚠ `dev_vs_bug` IS A DEVIATION BETWEEN ARMS, NOT AN ERROR. It is reported because
                # the two witnesses above are about STABILITY and say nothing about where the
                # trajectories end up; it is NOT evidence that `cbe_bug` is the correct one.
                dev = (arm == "cbe_bug" || r.died != 0 || isempty(res["cbe_bug"].prof)) ? NaN :
                      maximum(abs.(r.prof[1:S.L] .- res["cbe_bug"].prof[1:S.L]))
                @printf("   %6.3f   %-12s  %9.4f  %9.2e  %9d  %8.1f  %4d   %.3e%s\n",
                        dt, arm, r.amp, r.dtr, r.kry, r.wall, r.chi, dev,
                        r.died == 0 ? "" : "   DIED@$(r.died)")
                @printf(csv, "%s,%d,%s,%d,%d,%g,%g,%d,%.8g,%.6e,%d,%.4f,%d,%d,%.6e\n",
                        nm, S.d, arm, S.L, CAP, g, dt, nsteps, r.amp, r.dtr, r.kry, r.wall,
                        r.chi, r.died, dev)
                flush(csv)
            end
            flush(stdout)
        end
        println()
    end
    close(csv)
    println("wrote results/l18_backward_step.csv")
end

main()
