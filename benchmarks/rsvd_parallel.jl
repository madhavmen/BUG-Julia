# TWO INDEPENDENT SPEEDUPS ON ONE STEP: the RANDOMISED SKETCH and the PARALLEL HALF-SWEEPS.
# Do they each pay, do they COMPOSE, and what do they cost in accuracy against 2-site TDVP?
#
#     exact  serial OLD    the pre-2026-08-27 code (`share_ht = false`) -- the baseline
#     exact  serial        + one shared `H*Theta` per expansion
#     sketch serial        + the randomised sketch
#     exact  parallel      + both half-sweeps at once
#     sketch parallel      everything
#     tdvp2                the ACCURACY REFERENCE, and a second opinion on the seconds
#
# ⛔ EVERYTHING RUNS IN ONE PROCESS, AND THAT IS NOT A STYLE CHOICE. MEASURED before the
# precompile workload was added: a cold `julia` spent 265 s in `setup` and 313 s in the first
# `cbe_bug_step!` against 0.25 s per step afterwards -- ~580 s of JIT per invocation. A sweep that
# launched one process per model spent the overwhelming majority of its wall clock compiling the
# same code repeatedly. `PrecompileTools` in `src/BUGJulia.jl` fixed the per-process cost; running
# one process anyway keeps it paid once.
#
# ── WHAT CHANGED UNDER THIS TABLE ─────────────────────────────────────────────────────────
#
# ⛔ `cbe_expand` USED TO BUILD `H*Theta` THREE TIMES PER BOND. Its two preselections and the joint
# final-selection ranking each called the sketch closures, and each closure ran its own
# `apply_h_two_site`. That is the most expensive object in the expansion and the SKETCH DOES NOT
# TOUCH IT, so `t_cbe` was dominated by work rSVD is incapable of reducing.
#
# ⚠ AND IT MISLED THE EARLIER STUDY. `apply_h_two_site` scales with the MPO's virtual dimension, so
# the redundancy taxed WIDE generators hardest -- 1.63x on the XX chain (MPO dim 4) against 1.07x on
# the square cylinder (MPO dim 14). That was read as "a wide MPO lowers the rSVD ceiling", i.e. as a
# property of the method. It was an artefact of the triple build. `share_ht = false` is kept as an
# arm so the correction is visible in the table rather than asserted in a comment.
#
# ── METHOD, each piece forced by a mistake in an earlier attempt ──────────────────────────
#
# ⛔ EVERY ARM STEPS FROM THE **SAME STATE**. Timing whole trajectories cannot compare arms: they
# build different bases, reach different chi at the same time, and the comparison silently drifts
# to different problem sizes.
#
# ⛔ THE ROOT SOLVE IS PINNED (`tol = 0`, fixed `maxiter`) so every CBE-BUG arm applies the operator
# the same number of times. MEASURED before pinning: `krylov` 77 vs 90 between arms -- the seconds
# were not comparable at all. `kry` is printed so the match is CHECKED, never assumed.
#
# ⛔ ONE SITE COUNT FOR EVERY MODEL (L = 18 by default). The sweep cost is linear in `L` before
# anything else, so a table mixing an 18-site chain with a 32-site cylinder measures the size
# difference as much as the method. The 2D lattices are folded to that length (6x3 cylinder, 3x2
# kagome at 3 sites per cell) and the actual count is printed.
#
# ⚠ `err_t2` IS THE ACCURACY GUARD AND IT IS AGAINST TDVP2, NOT AGAINST ANOTHER CBE ARM. A sketch
# that is faster and worse is not a result, and comparing two of OUR arms would agree just as well
# if both were wrong. `dprof` (against the exact serial arm) additionally isolates the SKETCH error
# from the discretisation error the two integrators share.
#
#   julia --project=. -t 2 benchmarks/rsvd_parallel.jl [models] [chis] [reps] [sites]
#
# ⚠ `-t 2` OR MORE OR THE PARALLEL ARMS MEASURE NOTHING. On one thread the two tasks interleave on
# one worker: the step is the sequential one plus scheduling overhead. The banner says which.

using Printf, LinearAlgebra, Test
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))

const MODELS = length(ARGS) >= 1 ? split(ARGS[1], ",") :
               ["xx", "square", "kagome", "xx_open", "square_open", "lgt"]
const CHIS   = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [64, 128, 256]
const REPS   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
const SITES  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 18
const DT     = 0.05
const MI     = 20        # FIXED root iterations -- identical operator work in every CBE arm
const DEX    = 8         # admitted directions; the budget scan saturates here
const DOVER  = 4         # ⛔ PROBE WIDTH `Dpre = dex + dover = 12`. At Dpre <= 10 the sketch is
                         # 1.02e-07 from exact CBE and inflates the rank 26 -> 28; at 12 it is
                         # 6e-15, at the SAME wall clock. See `examples/common.jl`.

# ⛔ ONE OUTPUT FILE PER INVOCATION, AND THE CSV IS APPENDED, BECAUSE LONG RUNS HERE DIE.
# MEASURED TWICE: a >20-minute Julia run exits 127 with no error, no Windows crash event and (when
# piped) no captured output -- the six-model sweep lost five models after `xx` completed, and the
# test suite lost everything after 27 minutes the same way. Not OOM: 31 GB total, 9.6 GB free.
#
# ⚠ SO THE DRIVER IS BUILT TO SURVIVE IT rather than to explain it: run each model as its own
# short process, name the text log after the model list so one invocation cannot clobber another's
# results, and APPEND to a single CSV (writing the header only when creating it) so the table
# accumulates across invocations. A crash then costs one model, not the study.
const RESDIR = joinpath(@__DIR__, "results"); mkpath(RESDIR)
const TAG = replace(join(MODELS, "-"), "/" => "_")
const IO_ = open(joinpath(RESDIR, "rsvd_parallel_$(TAG).txt"), "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))
const CSVPATH = joinpath(RESDIR, "rsvd_parallel.csv")
const CSV_NEW = !isfile(CSVPATH)
const CSV = open(CSVPATH, "a")
CSV_NEW && println(CSV,
    "model,L,d,mpodim,chi,arm,exact,parallel,share_ht,secs,speedup,dprof,err_t2,kry," *
    "t_cbe,t_kry,t_split,t_frame,t_env,t_root,t_trunc,t_step")

say(@sprintf("julia threads = %d, BLAS threads = %d, cores = %d",
             Threads.nthreads(), BLAS.get_num_threads(), Sys.CPU_THREADS))
Threads.nthreads() >= 2 ||
    say("!! ONE JULIA THREAD: the `parallel` arms cannot overlap and will read as a small " *
        "PESSIMISATION. Re-run with `julia -t 2` for a meaningful parallel column.")
say(@sprintf("L = %d sites, chi in %s, reps = %d, dt = %.3f", SITES, string(CHIS), REPS, DT))
say("")

"""
One step of a named arm. `tol = 0` + fixed `maxiter`: no convergence exit, so the arms differ only
in HOW the sweep is executed, never in how much operator work it does.
"""
step!(p, S, exact::Bool, par::Bool, cap::Int; share_ht::Bool = true) =
    cbe_bug_step!(p, S.W, S.tau; exact = exact, parallel = par, share_ht = share_ht,
                  dex = DEX, dover = DOVER, comp_ratio = 1.0,
                  krylov_basis = 3, krylov_tol = 0.0, hermitian = (S.d == 2),
                  maxdim = cap, trunc_thresh = 1e-10, maxiter = MI, tol = 0.0)

"""
2-site TDVP at the same rank cap -- the independent reference, not another CBE arm.

⛔ `hermitian = (S.d == 2)`, MATCHING THE CBE ARMS. The fused open-system generator is a
Lindbladian and is NOT Hermitian; Lanczos on it does not fail, it returns a plausible vector. A
reference arm silently running Lanczos on a non-Hermitian generator would make every `err_t2` on
the three open models meaningless in a way nothing would flag.

⛔ BUT IT KEEPS ITS OWN `maxiter`/`tol`, UNLIKE THE CBE ARMS. Pinning `tol = 0` here THROWS:
without a convergence exit the two-site Krylov recursion runs the full `maxiter` even on a bond
whose block is smaller than that, and the degenerate basis fails inside `TLArray`
(`MethodError: no method matching TLArray(::Tuple{DataType}, ...)`, MEASURED at L=10 from a Neel
state; the defaults complete fine). The pinning exists so the CBE arms do IDENTICAL operator work
and their seconds are comparable -- TDVP2 is a different integrator whose operator count is not
being compared to anything, so it has nothing to be pinned to and is exempt from the `krylov`
match for the same reason.
"""
step_t2!(p, S, cap::Int) =
    tdvp2_step!(p, S.W, S.tau; maxdim = cap, trunc_thresh = 1e-10,
                hermitian = (S.d == 2))

# ══ PHASE 1. CORRECTNESS OF THE TWO CHANGES, BEFORE ANY SECONDS ARE QUOTED ════════════════
#
# ⛔ THE TWO CLAIMS ARE DIFFERENT AND NEED DIFFERENT TOLERANCES. Sharing `H*Theta` contracts the
# same tensors in the same order and merely stops rebuilding them, so it must be BIT-IDENTICAL --
# a `1e-10` tolerance there would pass a cache that returns a STALE `H*Theta` from a previous bond,
# which is how this class of bug actually fails. The parallel arm gives each sweep its own copy of
# the state to re-gauge, so `canonical!` starts from a different centre and the floating-point path
# genuinely differs; demanding bit-identity there would assert something false.
function validate()
    say("== PHASE 1: correctness ==")
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    function run(; share_ht, parallel, exact, n = 6)
        psi = neel_state(L); nmv = 0
        for _ in 1:n
            info = cbe_bug_step!(psi, mpo, ComplexF64(-im * DT);
                                 exact = exact, share_ht = share_ht, parallel = parallel,
                                 dex = 8, dover = 4, comp_ratio = 1.0,
                                 krylov_basis = 3, krylov_tol = 0.0,
                                 maxdim = 32, trunc_thresh = 1e-10, maxiter = 20, tol = 0.0)
            nmv += info.krylov_dims
        end
        n2 = max(norm(copy(psi))^2, eps())
        return (sz = [real(sz_expectation(copy(psi), j)) for j in 1:L],
                e = real(mpo_energy(copy(psi), mpo)) / n2,
                chi = bond_dims(psi), nmv = nmv)
    end
    ok = true
    for exact in (true, false)
        a = run(; share_ht = true,  parallel = false, exact = exact)
        b = run(; share_ht = false, parallel = false, exact = exact)
        bit = (a.sz == b.sz) && (a.e == b.e) && (a.chi == b.chi) && (a.nmv == b.nmv)
        say(@sprintf("  share_ht bit-identity (exact=%-5s): %s   (max|dsz| = %.3e, dE = %.3e)",
                     exact, bit ? "PASS" : "*** FAIL ***",
                     maximum(abs.(a.sz .- b.sz)), abs(a.e - b.e)))
        ok &= bit
        c = run(; share_ht = true, parallel = true, exact = exact)
        dsz = maximum(abs.(a.sz .- c.sz)); de = abs(a.e - c.e)
        pass = (a.chi == c.chi) && dsz < 1e-10 && de < 1e-10
        say(@sprintf("  parallel == serial   (exact=%-5s): %s   (max|dsz| = %.3e, dE = %.3e)",
                     exact, pass ? "PASS" : "*** FAIL ***", dsz, de))
        ok &= pass
    end
    # ⛔ REPEATABILITY IS THE STRUCTURAL TEST FOR `parallel`. If the right half-sweep read anything
    # the left one wrote, concurrency would be a race and the answer would vary run to run.
    # "Independent" and "usually wins the race" look identical in a single run.
    rs = [run(; share_ht = true, parallel = true, exact = true) for _ in 1:3]
    rep = all(r -> r.sz == rs[1].sz && r.chi == rs[1].chi, rs)
    say("  parallel is deterministic over 3 runs: " * (rep ? "PASS" : "*** FAIL ***"))
    ok &= rep
    say(ok ? "  ALL CORRECTNESS CHECKS PASS -- the seconds below are comparable." :
             "  *** CORRECTNESS FAILED -- do not quote the timings below. ***")
    say("")
    return ok
end

# ══ PHASE 2. THE TIMING TABLE ═════════════════════════════════════════════════════════════

"""
Grow once to `target`. WHICH ARM GROWS IT IS IRRELEVANT -- it is a STATE, not a result, and every
arm is then timed from the identical object. The exact serial arm keeps it reproducible.
"""
function grow_to(S, target::Int; maxsteps::Int = 400)
    p = copy(S.psi0); last = 0; stalled = 0
    for _ in 1:maxsteps
        step!(p, S, true, false, target); S.post!(p)
        chi = maximum(state_bond_dims(p))
        chi >= target && return p, chi
        stalled = (chi == last) ? stalled + 1 : 0
        stalled > 25 && return p, chi
        last = chi
    end
    return p, maximum(state_bond_dims(p))
end

"Time `REPS` steps from the SAME state; keep the BEST (least contended) and ITS phase timers."
function timed(p0, S, cap; exact = true, par = false, share_ht = true, t2 = false)
    best = Inf; kry = 0; prof = nothing; ph = nothing
    for _ in 1:REPS
        p = copy(p0)
        t0 = time_ns()
        info = t2 ? step_t2!(p, S, cap) : step!(p, S, exact, par, cap; share_ht = share_ht)
        el = (time_ns() - t0) / 1e9
        # ⛔ `info.krylov_dims`, NOT `get_krylov_log()`. The log records `expv` calls only, and
        # CBE-BUG calls `expv` EXACTLY ONCE per step (the root Galerkin solve) -- all the operator
        # work in its half-sweeps goes through `_krylov_frame`/`apply_one_site`, which the log
        # never sees. Reading the log gave `20` for CBE against `1980` for TDVP2 and would have
        # been quoted as a 99x reduction in operator applications; the true counts are what BOTH
        # info structs already carry in `krylov_dims`.
        kry = info.krylov_dims
        if el < best
            best = el
            # phase timers from the SAME repetition as the best wall clock, or the shares mix one
            # step's total with another's phases. TDVP2 reports none of them.
            g(f) = hasproperty(info, f) ? getproperty(info, f) : 0.0
            ph = (cbe = g(:t_cbe), kry = g(:t_kry), split = g(:t_split), frame = g(:t_frame),
                  env = g(:t_env), root = g(:t_root), trunc = g(:t_trunc), step = g(:t_step))
        end
        prof = S.profile(p)
    end
    return best, kry, prof, ph
end

#                label                exact  par    share_ht  tdvp2
const ARMS = [("exact  serial OLD",   true,  false, false,    false),
              ("exact  serial",       true,  false, true,     false),
              ("sketch serial",       false, false, true,     false),
              ("exact  parallel",     true,  true,  true,     false),
              ("sketch parallel",     false, true,  true,     false),
              ("tdvp2 (reference)",   true,  false, true,     true)]

function run_model(nm)
    S = setup(nm; sites = SITES, dt = DT, cap = maximum(CHIS))
    mpodim = maximum(mpo_virtual_dims(S.W))
    say(@sprintf("MODEL %-12s %s", nm, S.label))
    say(@sprintf("   %d sites, d = %d, MPO virtual dim %d", S.L, S.d, mpodim))
    let p = copy(S.psi0)                     # warmup: JIT must never land in a timed row
        for (_, ex, pa, sh, t2) in ARMS
            try
                timed_p = copy(p)
                t2 ? step_t2!(timed_p, S, 16) :
                     step!(timed_p, S, ex, pa, 16; share_ht = sh)
            catch e
                say("   warmup of an arm threw: " * sprint(showerror, e)[1:min(end, 200)])
            end
        end
    end
    say(@sprintf("   %-19s %9s %8s %11s %11s %7s | %7s %7s %7s %7s %7s %7s %7s",
                 "arm", "secs", "speedup", "dprof", "err_t2", "krylov",
                 "cbe", "kry", "split", "frame", "env", "root", "trunc"))
    # ⚠ MEASURE AT THE chi THE STATE ACTUALLY REACHES, NOT ONLY AT EXACT TARGETS. A model whose
    # entanglement saturates just below a target still gives a perfectly good timing point at the
    # rank it reached -- MEASURED: XX at L=10 tops out at chi = 30, and demanding 32 threw the row
    # away for a 6% shortfall. Rows are keyed by the ACHIEVED chi and deduplicated, so two targets
    # that saturate to the same rank are measured once rather than reported twice as if they were
    # independent points.
    done_chi = Set{Int}()
    for target in CHIS
        p0, chi = grow_to(S, target)
        if chi in done_chi
            say(@sprintf("   chi %-4d -- saturated at %d, already measured; SKIPPED",
                         target, chi))
            continue
        end
        push!(done_chi, chi)
        say(chi == target ? @sprintf("   --- chi = %d ---", chi) :
                            @sprintf("   --- chi = %d (target %d, saturated) ---", chi, target))
        # ⚠ TDVP2 FIRST so its profile is available as the accuracy reference for every CBE arm.
        # It is REPORTED last, in the ARMS order, so the table still reads baseline-first.
        t2res = try
            timed(p0, S, chi; t2 = true)
        catch e
            say("      tdvp2 threw: " * sprint(showerror, e)[1:min(end, 200)]); nothing
        end
        base = nothing; bprof = nothing; bkry = 0
        for (label, ex, pa, sh, t2) in ARMS
            (t2 && t2res === nothing) && continue
            s, k, pr, ph = t2 ? t2res : timed(p0, S, chi; exact = ex, par = pa, share_ht = sh)
            base === nothing && (base = s; bprof = pr; bkry = k)
            dprof  = maximum(abs.(pr .- bprof))
            err_t2 = t2res === nothing ? NaN : maximum(abs.(pr .- t2res[3]))
            say(@sprintf("   %-19s %9.3f %8.2fx %11.3e %11.3e %7d | %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f",
                         label, s, base / max(s, 1e-12), dprof, err_t2, k,
                         ph.cbe, ph.kry, ph.split, ph.frame, ph.env, ph.root, ph.trunc))
            # TDVP2 legitimately does different operator work; only the CBE arms must match.
            (t2 || k == bkry) ||
                say("      !! krylov MISMATCH ($bkry vs $k) -- arms did different operator " *
                    "work, these seconds are NOT comparable.")
            println(CSV, @sprintf("%s,%d,%d,%d,%d,%s,%s,%s,%s,%.6f,%.4f,%.6e,%.6e,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                                  nm, S.L, S.d, mpodim, chi, replace(strip(label), " " => "_"),
                                  ex, pa, sh, s, base / max(s, 1e-12), dprof, err_t2, k,
                                  ph.cbe, ph.kry, ph.split, ph.frame, ph.env, ph.root,
                                  ph.trunc, ph.step))
            flush(CSV)
        end
    end
    say("")
    return nothing
end

validate()
for nm in MODELS
    try
        run_model(String(nm))
    catch e
        say("MODEL $nm THREW: " * sprint(showerror, e)[1:min(end, 400)])
    end
end
close(CSV)
say("READING IT:")
say("  `speedup` is against `exact serial OLD` -- the pre-fix code -- on the SAME state, so the")
say("     block reads as what each change bought, cumulatively, from where this stood before.")
say("  ⚠ THE PHASE SECONDS ARE CPU TIME SUMMED OVER BOTH HALF-SWEEPS. In the serial arms that is")
say("     also wall clock and they sum to ~`t_step`; in the PARALLEL arms they deliberately do")
say("     not -- two sweeps at once spend more CPU seconds than the step lasts, and phases")
say("     summing past `t_step` is the overlap showing up, not a broken profile.")
say("  `dprof` is against the EXACT SERIAL arm and isolates the SKETCH error. It must stay at")
say("     roundoff, and is NOT expected to be exactly 0 for the parallel arms: each half-sweep")
say("     re-gauges its own copy, so the floating-point path differs while the spaces do not.")
say("  `err_t2` is against 2-site TDVP at the SAME rank cap -- an INDEPENDENT integrator, so it")
say("     bounds the discretisation difference rather than agreeing with our own arms.")
say("  `krylov` is `info.krylov_dims` -- TRUE operator applications, half-sweeps included, not the")
say("     `expv` log (which sees only CBE-BUG's single root solve). It MUST match down the CBE")
say("     rows: the root solve is pinned AND the half-sweep depth is pinned, so every CBE arm")
say("     applies the operator the same number of times and the seconds compare. TDVP2")
say("     legitimately differs and is exempt.")
close(IO_)
