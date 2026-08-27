# DOES THE SKETCH PAY ON THE HARD MODELS? -- square, kagome, and the open systems.
#
# The Schwinger chain is the FAVOURABLE case: a very wide MPO (all-to-all electric term) and
# `d = 4` from the fused (ket, bra) site, so the exact local block `d*chi` is large and there is
# room for a sketch of width `npre` to win. The models here are the ones that test it:
#
#   square       2D cylinder, `:U1`, d = 2   -- MPO width grows with Ly, NOT with Lx
#   kagome       breathing kagome, `:U1`, d = 2, dipolar tail -> the widest CLOSED MPO
#   xx_open      fused Lindblad, `:none`, d = 4, NEAREST-NEIGHBOUR -> the NARROWEST MPO here
#   square_open  fused Lindblad on a 2D lattice, `:none`, d = 4 -- wide AND d = 4
#   lgt          Schwinger + bath, the favourable reference case
#
# ⚠ `xx_open` IS IN THE LIST PRECISELY BECAUSE IT SHOULD LOSE. Its MPO is nearest-neighbour, so
# the contraction the sketch shrinks is small and the overhead should dominate. A study that only
# ran the favourable cases could not tell "rSVD helps" from "we picked helpful models".
#
# ── METHODOLOGY, carried over from `rsvd_lgt.jl` where each piece was forced by a mistake ──
#
# ⛔ BOTH ARMS TIMED FROM THE **SAME STATE** at each chi. Timing whole trajectories cannot locate a
# crossover: the arms build different bases, reach different chi at the same time, and the
# comparison silently drifts to different problem sizes.
#
# ⛔ THE ROOT SOLVE IS PINNED (`tol = 0`, fixed `maxiter`) so both arms apply the operator the SAME
# number of times. MEASURED before pinning: `krylov` 77 (exact) vs 90 (sketched) -- the seconds
# were not comparable at all. `kry` is printed per row so the match is checked, never assumed.
#
# ⛔ AND THE SKETCH USES THE MEASURED-BEST CONFIGURATION, not the defaults: `dex = 8, dover = 0,
# comp_ratio = 1.0`, from the sweep in `rsvd_optimal.jl` (N=8, chi=32: 1.10x best, vs 1.04x at the
# default `dover`, 0.94x at dex 16, 0.64x at dex 32). Measuring a crossover with a mediocre sketch
# would understate rSVD.
#
# ⚠ `dprof` IS THE ACCURACY GUARD and is model-appropriate: bond energies for the closed models,
# `Tr(S^z_j rho)` for the open ones. A faster-and-worse sketch is not a result.
#
# ⚠ `frac = Dpre/(d*chi)` USES EACH MODEL'S OWN `d` -- 2 for the spin-1/2 lattices, 4 for the fused
# chains. Using a single `d` would misstate the speedup ceiling by 2x on half the table.
#
# ⛔ THE AXIS IS THE **MPS** BOND DIMENSION, NOT THE MPO WIDTH. The exact CBE step factorises a
# `d*chi x chi` local block, so its cost grows with `chi` -- a wide MPO only sets how expensive
# each contraction is, while `chi` sets how big the thing being FACTORISED is. `chi` is therefore
# swept and the MPO width is reported as context, not as the variable.
#
# ⛔ AND `t_cbe` DECIDES WHETHER THE QUESTION IS EVEN LIVE. The sketch makes `cbe_expand` cheaper
# and NOTHING ELSE, so by Amdahl the whole-step speedup is capped at `1/(1 - f)` where
# `f = t_cbe / t_step`. MEASURED on the Schwinger chain: 1.10x against a `Dpre/(d*chi)` ceiling of
# ~16x -- which is the signature of a SMALL `f`, not of a slow sketch. Reporting `f` separates
# "the rSVD is inefficient" from "the rSVD is fine and the expansion is 10% of the step"; those
# call for opposite responses, and only the first is a code problem.
#
#   julia --project=. benchmarks/rsvd_models.jl [model] [chi] [reps] [size...]

#
# ⚠ SUPERSEDED FOR THE HEADLINE NUMBER by `benchmarks/rsvd_parallel.jl`, which runs this
# comparison as one of FOUR arms (sketch x parallel half-sweeps) on the same six models. This file
# survives because it reports the phase SHARES and the Amdahl ceiling, which is the diagnostic that
# separates "the sketch is slow" from "the sketch is fine and `cbe_expand` is 10% of the step".
#
# ⛔ AND ITS PRE-2026-08-27 NUMBERS ARE OBSOLETE: `cbe_expand` used to build `H*Theta` THREE times
# per bond (both preselections plus the joint ranking), which is work the sketch cannot reduce, so
# every ratio here was measured against an inflated baseline. See `_sketch_closures`.

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))

const MODEL = length(ARGS) >= 1 ? ARGS[1]                 : "square"
const CHIS  = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) :
                                  [32, 64, 128, 256]
const CHI = maximum(CHIS)
const REPS  = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 3
const SITES = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 18
const DT    = 0.05
const MI    = 20
const DEX   = 8          # the accuracy floor; see the header
const GAMMA = 0.1        # small ON PURPOSE for the open models: dephasing SUPPRESSES operator
                         # entanglement, so a damped run never reaches the interesting chi

const OUT = joinpath(@__DIR__, "results", "rsvd_models_$(MODEL)_chi$(CHI).txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

# ⛔ THE MODELS LIVE IN `study_models.jl` so this file and `rsvd_parallel.jl` measure the
# SAME systems. They used to carry a copy each, which is how two drivers end up quietly
# comparing an 8x4 cylinder against a 6x4 one.

const S = setup(MODEL; sites = SITES, dt = DT, gamma = GAMMA, cap = CHI)
const MPODIM = maximum(mpo_virtual_dims(S.W))

"One step. ⚠ `tol = 0` + fixed `maxiter` = identical operator work in both arms."
step!(p, exact, cap) =
    cbe_bug_step!(p, S.W, S.tau; exact = exact, dex = DEX, dover = 0, comp_ratio = 1.0,
                  krylov_basis = 3, krylov_tol = 0.0,
                  hermitian = (S.d == 2), maxdim = cap, trunc_thresh = 1e-10,
                  maxiter = MI, tol = 0.0)

function grow_to(target::Int; maxsteps::Int = 400)
    p = copy(S.psi0); last = 0; stalled = 0
    for _ in 1:maxsteps
        step!(p, true, target); S.post!(p)
        chi = maximum(state_bond_dims(p))
        chi >= target && return p, chi
        stalled = (chi == last) ? stalled + 1 : 0
        stalled > 25 && return p, chi
        last = chi
    end
    return p, maximum(state_bond_dims(p))
end

function timed(p0, exact, cap)
    best = Inf; kry = 0; prof = nothing; tcbe = 0.0; tk = 0.0; tr = 0.0; tt = 0.0
    tsp = 0.0; tfr = 0.0; tev = 0.0; tstp = 0.0
    for _ in 1:REPS
        p = copy(p0)
        enable_krylov_log()
        t0 = time_ns()
        info = step!(p, exact, cap)
        el = (time_ns() - t0) / 1e9
        kry = sum(get_krylov_log(); init = 0)
        disable_krylov_log()
        if el < best
            best = el
            # ⚠ take the phase timers from the SAME repetition as the best wall clock, or the
            # shares mix a fast step's total with a contended step's phases.
            g(f) = hasproperty(info, f) ? getproperty(info, f) : NaN
            tcbe = g(:t_cbe); tk = g(:t_kry); tr = g(:t_root); tt = g(:t_trunc)
            tsp = g(:t_split); tfr = g(:t_frame); tev = g(:t_env); tstp = g(:t_step)
        end
        prof = S.profile(p)
    end
    return best, kry, prof, tcbe, tk, tr, tt, tsp, tfr, tev, tstp
end

say(@sprintf("MODEL %s -- %s", MODEL, S.label))
say(@sprintf("  %d sites, local dimension d = %d, MPO virtual dim %d", S.L, S.d, MPODIM))
say(@sprintf("  sketch: dex = %d, dover = 0, comp_ratio = 1.0 (the measured-best configuration)",
             DEX))
# warmup so JIT never lands in a timed row
let p = copy(S.psi0)
    step!(p, true, 16); step!(p, false, 16)
end
say("")
say(@sprintf("  %5s %8s %8s %6s | %6s %6s %6s %6s %6s %6s %6s | %6s %6s",
             "chi", "exact", "sketch", "ratio",
             "cbe", "split", "frame", "env", "kry", "root", "trunc", "SUM", "amdahl"))
say("        (shares of the EXACT step. ⛔ SUM must be ~1.0 or the profile is not closed and")
say("         the bottleneck may be in whatever is still unattributed. `split` and `frame` are")
say("         ALSO SVDs, and the sketch does NOT touch them.)")
const CSV = open(joinpath(@__DIR__, "results", "rsvd_models_$(MODEL).csv"), "w")
println(CSV, "model,L,d,mpodim,chi,secs_exact,secs_sketch,ratio,tcbe_exact,tcbe_sketch," *
             "f_cbe,amdahl_cap,dprof,kry_exact,kry_sketch,frac,f_kry,f_root,f_trunc")
for target in CHIS
    p0, chi = grow_to(target)
    if chi < target
        say(@sprintf("  %6d  -- entanglement saturated at chi = %d; SKIPPED", target, chi))
        continue
    end
    se, ke, pe, ce, ke2, re, te, spe, fre, eve, ste = timed(p0, true, chi)
    ss, ks, ps, cs, ks2, rs, ts, sps, frs, evs, sts = timed(p0, false, chi)
    dprof = maximum(abs.(ps .- pe))
    frac  = DEX / (S.d * chi)
    # ⚠ SHARES ARE TAKEN AGAINST `t_step` (measured INSIDE the sweep), not the outer wall clock:
    # the outer timer also catches allocation and GC outside the sweep, which would make the SUM
    # look short for a reason that is not a missing phase.
    base  = isnan(ste) || ste <= 0 ? se : ste
    f_cbe = ce / base;  f_sp = spe / base; f_fr = fre / base; f_ev = eve / base
    f_kry = ke2 / base; f_rt = re / base;  f_tr = te / base
    ssum  = f_cbe + f_sp + f_fr + f_ev + f_kry + f_rt + f_tr
    cap   = 1 / max(1 - f_cbe, 1e-12)           # Amdahl ceiling on the whole-step speedup
    say(@sprintf("  %5d %8.3f %8.3f %6.2f | %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f | %6.3f %6.2f",
                 chi, se, ss, se / max(ss, 1e-12),
                 f_cbe, f_sp, f_fr, f_ev, f_kry, f_rt, f_tr, ssum, cap))
    # ⛔ THE SKETCHED ARM'S PHASES IN ABSOLUTE SECONDS, BESIDE THE EXACT ARM'S. The shares above
    # are of the EXACT step, so they cannot show the thing that actually limits the speedup: the
    # sketched arm GIVING TIME BACK outside `cbe`. MEASURED: the sketch halves `cbe` (2.14x) yet
    # the whole step moves only 1.05x, so ~0.27 of a step is reappearing somewhere -- and only a
    # side-by-side of the SAME phases in BOTH arms can say where.
    say(@sprintf("        exact  s: cbe %7.3f  split %7.3f  frame %7.3f  env %7.3f  kry %7.3f  trunc %7.3f",
                 ce, spe, fre, eve, ke2, te))
    say(@sprintf("        sketch s: cbe %7.3f  split %7.3f  frame %7.3f  env %7.3f  kry %7.3f  trunc %7.3f",
                 cs, sps, frs, evs, ks2, ts))
    say(@sprintf("        delta  s: cbe %+7.3f  split %+7.3f  frame %+7.3f  env %+7.3f  kry %+7.3f  trunc %+7.3f   (sketch - exact)",
                 cs - ce, sps - spe, frs - fre, evs - eve, ks2 - ke2, ts - te))
    ke == ks || say("      ⛔ krylov MISMATCH ($ke vs $ks) -- arms did different operator work, " *
                    "these seconds are NOT comparable.")
    println(CSV, @sprintf("%s,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%.4f,%.6e,%d,%d,%.6f,%.6f,%.6f,%.6f",
                          MODEL, S.L, S.d, MPODIM, chi, se, ss, se / max(ss, 1e-12), ce, cs,
                          f_cbe, cap, dprof, ke, ks, frac, f_kry, f_rt, f_tr))
    flush(CSV)
end
close(CSV)
say("READING IT:")
say("  `f_cbe` IS THE FIRST COLUMN TO READ. It is the share of the step spent inside")
say("     `cbe_expand`, and the sketch makes NOTHING ELSE cheaper -- so the whole-step speedup is")
say("     capped at 1/(1-f_cbe) by Amdahl, no matter how good the sketch is.")
say("     f_cbe = 0.10 -> ceiling 1.11x. f_cbe = 0.50 -> 2.0x. f_cbe = 0.90 -> 10x.")
say("  ⛔ SO A SMALL `ratio` AT SMALL `f_cbe` IS NOT AN INEFFICIENT rSVD -- it is the wrong 10%")
say("     being optimised. Only a small `ratio` at LARGE `f_cbe` indicts the implementation.")
say("  `ratio` > 1 = the sketch is faster. `Dpre/dchi` is the CONTRACTION-level ceiling.")
say("  ⚠ `xx_open` is expected to LOSE -- its MPO is nearest-neighbour, so there is little")
say("  contraction for the sketch to shrink. That row is the control, not a failure.")
close(IO_)
