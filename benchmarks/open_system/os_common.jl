# Shared machinery of the open-system MODEL drivers (plan Tasks 5-7: OS-1 driven XX + dephasing,
# OS-2 Bose-Hubbard + loss, OS-3 U(1) East + dephasing): the knobs, one arm over one uniform time
# grid on a vectorised density matrix (`liouville.jl`), and its CSVs. A driver supplies the model
# (Liouville space, Lindblad MPO, initial MPDO, observables, reference) and includes this file.
#
# Arms: bugP = cbe_bug_step! at order P on the Liouvillian (tau = +dt real, hermitian = false,
# Arnoldi), with |I>> in the trial space (`keep_states`) when the grading allows it, so the Galerkin
# step is trace-exact and only the one compression moves the trace; tdvp2 = tdvp2_step! with the SAME
# per-cut discard rule (`weight_tol = tol_trunc`) and its backward-substep amplification logged.
# Each step's raw trace is logged, then the state is renormalised to Tr rho = 1 (plan Task 2). A
# step whose ledger is not met is labelled `budget_unmet` with its reasons -- never a converged point.
#
# Env: OS_TMAX (10) OS_DTS ("0.1", a list: the stiff-step sweep) OS_ARMS ("bug2,bug3,bug4,tdvp2")
#      OS_TB ("1e-3,1e-5,1e-7") OS_TT (1e-6) OS_TR (1e-6) OS_MAXDIM (256) OS_ROOT (exp)
#      OS_SEEDS ("1") OS_OBS_EVERY (1: observables every k steps, untimed) OS_BACKAMP (1)
#      OS_CHECKS (1: Hermiticity / positivity at output times, dense when small) OS_OUT
#      OS_FRAME_CHECKS (0, debug)

using LinearAlgebra, Printf, Random
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const OS = (tmax      = parse(Float64, get(ENV, "OS_TMAX", "10")),
            dts       = parse.(Float64, split(get(ENV, "OS_DTS", "0.1"), ",")),
            arms      = split(get(ENV, "OS_ARMS", "bug2,bug3,bug4,tdvp2"), ","),
            tbs       = parse.(Float64, split(get(ENV, "OS_TB", "1e-3,1e-5,1e-7"), ",")),
            tt        = parse(Float64, get(ENV, "OS_TT", "1e-6")),
            tr        = parse(Float64, get(ENV, "OS_TR", "1e-6")),
            maxdim    = parse(Int, get(ENV, "OS_MAXDIM", "256")),
            root      = Symbol(get(ENV, "OS_ROOT", "exp")),
            seeds     = parse.(Int, split(get(ENV, "OS_SEEDS", "1"), ",")),
            obs_every = parse(Int, get(ENV, "OS_OBS_EVERY", "1")),
            backamp   = get(ENV, "OS_BACKAMP", "1") == "1",
            checks    = get(ENV, "OS_CHECKS", "1") == "1",
            out       = get(ENV, "OS_OUT", joinpath(@__DIR__, "results")),
            parallel  = Threads.nthreads() >= 2)

BUGJulia.RSVDCBEBondUpdate._FRAME_CHECKS[] = get(ENV, "OS_FRAME_CHECKS", "0") == "1"

say(msg) = (println(msg); flush(stdout))

"`n` steps of `dt` reaching `tmax` (refused unless `tmax/dt` is an integer: every arm and the
reference share the output times)."
function os_nsteps(dt)
    n = round(Int, OS.tmax / dt)
    abs(n * dt - OS.tmax) < 1e-9 * OS.tmax || error("OS_TMAX/dt must be an integer, got $(OS.tmax)/$dt")
    return n
end

"The output times of a run at `dt`: `t = 0` and every `obs_every`-th step (and the last)."
os_out_steps(n) = [k for k in 0:n if k % OS.obs_every == 0 || k == n]

"Every (arm, tol_basis, seed) combination: tdvp2 has one, with no tol_basis."
os_runs(arm) = [(tb, s) for tb in (startswith(arm, "bug") ? OS.tbs : [NaN])
                for s in (startswith(arm, "bug") ? OS.seeds : OS.seeds[1:1])]

os_purity(rho, ls) = real(overlap(rho, rho)) / abs2(mpdo_trace(rho, ls))

_e2(v) = @sprintf("%.2e", v)
_errstr(errs) = join(["$k=" * _e2(v) for (k, v) in sort(collect(errs))], " ")

const OS_STEP_HEADER = "tag,arm,tol_basis,seed,dt,step,t,chi,wall,status,unmet,trace_raw,purity," *
                       "discarded,backamp,herm,min_eig"
const OS_OBS_HEADER = "tag,arm,tol_basis,seed,dt,step,t,obs,i,value,ref"

"""
One arm on the uniform grid `dt` from `rho0`. `observe(rho) -> Dict(name => Vector)`; `ref[(k, name)]`
the reference vector at output step `k` (`nothing` = no reference). Writes a step row per step into
`sio` and the observables at the output steps into `oio`; returns the final state, the worst
error per observable over the output times, and the counts.
"""
function os_run_arm(sio, oio, tag, ls, W, rho0, Iket, observe, ref, arm, dt, tb, seed)
    n = os_nsteps(dt)
    outs = Set(os_out_steps(n))
    rho = copy(rho0)
    rng = MersenneTwister(seed)
    wall, nfail = 0.0, 0
    errs = Dict{String, Float64}()
    function emit(k)
        obs = observe(rho)
        for (name, v) in obs, (i, x) in pairs(v)
            r = ref === nothing ? NaN : ref[(k, name)][i]
            @printf(oio, "%s,%s,%.0e,%d,%.6g,%d,%.10g,%s,%d,%.14e,%.14e\n", tag, arm, tb, seed, dt,
                    k, k * dt, name, i, x, r)
            ref === nothing || (errs[name] = max(get(errs, name, 0.0), abs(x - r)))
        end
        flush(oio)
    end
    emit(0)
    for k in 1:n
        amp, disc = NaN, NaN
        t0 = time_ns()
        if startswith(arm, "bug")
            info = cbe_bug_step!(rho, W, ComplexF64(dt); order = parse(Int, arm[4:end]),
                                 root_method = OS.root, hermitian = false, tol_basis = tb,
                                 tol_trunc = OS.tt, tol_expv = OS.tr, maxdim = OS.maxdim,
                                 keep_states = Iket === nothing ? Any[] : [Iket],
                                 parallel = OS.parallel, rng = rng)
            st = budget_status(info.ledger)
            um = join(string.(info.ledger.unmet), "+")
            disc = info.ledger.discarded
        else
            rep = OS.backamp ? Ref{Any}(NaN) : nothing
            info = tdvp2_step!(rho, W, ComplexF64(dt); hermitian = false, maxdim = OS.maxdim,
                               maxiter = 30, conv_tol = OS.tr, substeps = 0, weight_tol = OS.tt,
                               backward_report = rep)
            st, um = :tdvp2, ""
            disc = info.cumulative
            amp = rep === nothing ? NaN : Float64(rep[])
        end
        trr = mpdo_normalise_trace!(rho, ls)
        wall += (time_ns() - t0) / 1e9
        nfail += st === :budget_unmet
        herm, mn = NaN, NaN
        if k in outs
            if OS.checks
                c = mpdo_checks(rho, ls)
                herm, mn = c.herm, c.min_eig
            end
            emit(k)
        end
        @printf(sio, "%s,%s,%.0e,%d,%.6g,%d,%.10g,%d,%.3f,%s,%s,%.14e,%.10e,%.6e,%.6e,%.6e,%.6e\n",
                tag, arm, tb, seed, dt, k, k * dt, maximum(bond_dims(rho)), wall, st, um,
                real(trr), os_purity(rho, ls), disc, amp, herm, mn)
        flush(sio)
    end
    say(@sprintf("  %-6s dt %-6g tb %.0e seed %d: chi %d  wall %.1fs  unmet %d  errors %s",
                 arm, dt, tb, seed, maximum(bond_dims(rho)), wall, nfail, _errstr(errs)))
    return (rho = rho, errs = errs, wall = wall, nfail = nfail)
end

"JIT warm-up, untimed: two steps of every arm on the small model `(W6, rho6, Iket6)`."
function os_warmup(ls, W6, rho6, Iket6)
    for arm in OS.arms, _ in 1:2
        r = copy(rho6)
        if startswith(arm, "bug")
            cbe_bug_step!(r, W6, ComplexF64(0.05); order = parse(Int, arm[4:end]),
                          root_method = OS.root, hermitian = false, tol_basis = OS.tbs[1],
                          tol_trunc = OS.tt, tol_expv = OS.tr, maxdim = 16,
                          keep_states = Iket6 === nothing ? Any[] : [Iket6], parallel = OS.parallel)
        else
            tdvp2_step!(r, W6, ComplexF64(0.05); hermitian = false, maxdim = 16, maxiter = 30,
                        conv_tol = OS.tr, substeps = 0, weight_tol = OS.tt,
                        backward_report = OS.backamp ? Ref{Any}(NaN) : nothing)
        end
        mpdo_normalise_trace!(r, ls)
        OS.checks && mpdo_checks(r, ls)
    end
end

"Open the three CSVs of a driver under `OS.out`, with their headers."
function os_open(prefix, tag)
    mkpath(OS.out)
    sio = open(joinpath(OS.out, "$(prefix)_steps_$tag.csv"), "w"); println(sio, OS_STEP_HEADER)
    oio = open(joinpath(OS.out, "$(prefix)_obs_$tag.csv"), "w"); println(oio, OS_OBS_HEADER)
    fio = open(joinpath(OS.out, "$(prefix)_final_$tag.csv"), "w")
    println(fio, "tag,arm,tol_basis,seed,dt,wall,nfail,chi,obs,max_err")
    return sio, oio, fio
end

"One summary row per observable of a finished run."
function os_final!(fio, tag, arm, tb, seed, dt, f)
    for (name, e) in (isempty(f.errs) ? Dict("none" => NaN) : f.errs)
        @printf(fio, "%s,%s,%.0e,%d,%.6g,%.3f,%d,%d,%s,%.6e\n", tag, arm, tb, seed, dt, f.wall,
                f.nfail, maximum(bond_dims(f.rho)), name, e)
    end
    flush(fio)
end
