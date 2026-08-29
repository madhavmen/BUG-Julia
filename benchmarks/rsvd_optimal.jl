# HOW SHOULD THE SKETCH BE BUILT TO MAXIMISE THE SPEEDUP? -- starting from Jan's point 1.
#
# ── WHAT JAN'S PLAN ALREADY SETTLES, so this does not re-ask it ───────────────────────────
#
# POINT 3 ("full SVD before rSVD") is ANSWERED: MEASURED on Heisenberg L=12, `exact = true` and
# `exact = false` give BIT-IDENTICAL error and rank (1.4595e-06 at chi 11; 1.4748e-06 at chi 37;
# ratio 1.000). **The sketch is accuracy-free there**, so there is no accuracy/speed trade to
# navigate -- the only question left is SPEED, and every parameter below should be pushed until
# accuracy actually starts to move.
#
# POINT 1(a): the MATLAB's `0.5*P_kept + 0.5*P_discarded` split is `comp_ratio`, and this repo
# already departs from the reference by defaulting to `1.0` -- the probe drawn ENTIRELY from the
# frame's complement. The MATLAB explicitly rejects that ("do not project into complement space,
# 1-site components are also important for bond update", `RSVDpreBE0SiQS.m:339`). ⚠ That departure
# is a CHOICE and it is on the sweep below, because a deliberate departure has to be re-justified
# whenever the model changes -- here to a d = 4 fused chain with a very wide MPO.
#
# ── WHAT ACTUALLY GOVERNS SPEED, AND WHY IT IS THE *WIDTH* ───────────────────────────────
#
#     budget = dex <= 0 ? ceil(growth*dmax) - r : dex     (cbe_core.jl)
#     npre   = dover === nothing ? ceil(1.2*dex) : dex + dover      (cbe_core.jl:587)
#
# ⚠ `dover` IS AN ABSOLUTE COLUMN COUNT (`Union{Nothing, Int}`), NOT A FRACTION. Passing `0.2`
# throws `TypeError: expected Union{Nothing, Int64}, got Float64` -- which it did on the first
# attempt, taking out two thirds of the sweep. `dover = 0` means NO oversampling at all, which is
# the cheapest probe available and the interesting end of this axis.
#
# The sketched contraction touches `Dpre` columns where the exact one touches `d*chi`. So the
# speedup is bounded by `Dpre / (d*chi)` and the two knobs that move it are `dex` and `dover` --
# `comp_ratio` re-allocates the SAME columns and changes accuracy, not width.
#
# ⚠ SHRINKING `dex` IS NOT FREE ACCURACY-WISE EVEN THOUGH THE SKETCH IS: `dex` is the number of
# directions ADMITTED to the bond, so it changes the ALGORITHM, not just the probe. This repo's
# own budget scan (XX L=10) found `dex = 8` matching `dex = 16` and `dex = 64` to the digit, which
# is why 8 is the floor of the sweep and not something smaller.
#
# ⚠ `dover` IS THE ONE KNOB THAT IS PURELY OVERSAMPLING -- it widens the probe without admitting
# more directions. Cutting it to 0 is the cheapest speed there is, and the risk is a
# rank-deficient sketch missing a direction, which shows up in `dprof` and nowhere else.
#
# ── METHOD ────────────────────────────────────────────────────────────────────────────────
#
# ⛔ ONE STATE, MANY CONFIGURATIONS. The state is grown ONCE to the target chi and every arm steps
# from that identical object, so `secs` and `dprof` differ only by the configuration. Timing along
# separate trajectories would compare different problem sizes, which is the error the first
# crossover attempt made.
#
# ⛔ AND THE ROOT SOLVE IS PINNED (`tol = 0`, fixed `maxiter`) so every arm performs IDENTICAL
# operator work. MEASURED before pinning: `krylov` 77 vs 90 between arms, i.e. the seconds were
# not comparable at all. `kry` is printed so the match is checked and not assumed.
#
#   julia --project=. benchmarks/rsvd_optimal.jl [N] [gamma] [chi] [reps]

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 20
const GAMMA = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.1
const CHI   = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 128
const REPS  = length(ARGS) >= 4 ? parse(Int, ARGS[4])     : 3
const DT    = 0.05
const MI    = 20

set_symmetry!(:none)
iseven(N) || throw(ArgumentError("the staggered-fermion Schwinger chain needs an even N"))

const OUT = joinpath(@__DIR__, "results", "rsvd_optimal_N$(N)_chi$(CHI).txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))
const CSV = open(joinpath(@__DIR__, "results", "rsvd_optimal_N$(N)_chi$(CHI).csv"), "w")
println(CSV, "N,chi,arm,dex,dover,comp_ratio,secs,ratio,kry,dprof,Dpre,frac")

function schwinger_terms(N::Int, x::Float64, mu::Float64, eps0::Float64)
    Jxy = zeros(Float64, N, N)
    for n in 1:(N - 1)
        Jxy[n, n + 1] = 2 * x
    end
    Jzz = zeros(Float64, N, N)
    for k in 1:(N - 1), l in (k + 1):N
        l <= N - 1 || continue
        Jzz[k, l] = 2.0 * (N - l)
    end
    a(n) = eps0 + (isodd(n) ? -0.5 : 0.0)
    hz = [2.0 * sum(a(n) for n in k:(N - 1); init = 0.0) + mu * (isodd(k) ? -1.0 : 1.0)
          for k in 1:N]
    return Jxy, Jzz, hz
end

const JXY, JZZ, HZ = schwinger_terms(N, 1.0, 0.5, 0.0)
const W, SHIFT = fused_lindblad(N, JXY, fill(GAMMA, N); hz = HZ, Jzz = JZZ)
const IMPS = identity_superket(N)
const MPODIM = maximum(mpo_virtual_dims(W))
rho0() = rho0_product([isodd(k) ? :up : :down for k in 1:N])
profile(rho) = [real(trace_rho(IMPS, rho, N; at = j)) for j in 1:N]

"One step under a named configuration. `dover === nothing` takes the reference's 0.2."
function step_cfg!(p, exact::Bool, dex::Int, dover, comp_ratio, cap::Int)
    cbe_bug_step!(p, W, ComplexF64(DT); exact = exact, dex = dex, dover = dover,
                  comp_ratio = comp_ratio, krylov_basis = 3, krylov_tol = 0.0,
                  hermitian = false, maxdim = cap, trunc_thresh = 1e-10,
                  maxiter = MI, tol = 0.0)
end

"Grow once to `CHI`; whichever arm grows it is irrelevant, it is a STATE not a result."
function grow_to(target::Int; maxsteps::Int = 500)
    p = rho0(); last = 0; stalled = 0
    for _ in 1:maxsteps
        step_cfg!(p, true, 8, nothing, 1.0, target)
        SHIFT != 0.0 && apply_shift!(p, exp(DT * SHIFT))
        restore_trace!(IMPS, p, N; maxdim = target, cutoff = 1e-10)
        chi = maximum(state_bond_dims(p))
        chi >= target && return p, chi
        stalled = (chi == last) ? stalled + 1 : 0
        stalled > 30 && return p, chi
        last = chi
    end
    return p, maximum(state_bond_dims(p))
end

function timed(p0, exact, dex, dover, comp_ratio, cap)
    best = Inf; kry = 0; prof = nothing
    for _ in 1:REPS
        p = copy(p0)
        enable_krylov_log()
        t0 = time_ns()
        step_cfg!(p, exact, dex, dover, comp_ratio, cap)
        el = (time_ns() - t0) / 1e9
        kry = sum(get_krylov_log(); init = 0)
        disable_krylov_log()
        el < best && (best = el)
        prof = profile(p)
    end
    return best, kry, prof
end

say(@sprintf("Schwinger N = %d FUSED (d = 4), gamma = %.2f, target chi = %d, reps = %d",
             N, GAMMA, CHI, REPS))
say(@sprintf("MPO virtual dim %d (all-to-all ZZ).", MPODIM))
# warmup: never let JIT land in a timed row
let p = rho0()
    step_cfg!(p, true, 8, nothing, 1.0, 16); step_cfg!(p, false, 8, nothing, 1.0, 16)
end
p0, chi = grow_to(CHI)
say(@sprintf("grown to chi = %d (target %d)%s", chi, CHI,
             chi < CHI ? "  ⚠ SATURATED BELOW TARGET -- speedup room is limited, read `frac`" : ""))
say("")

# the EXACT reference: both the timing baseline and the accuracy reference
sec_ex, kry_ex, prof_ex = timed(p0, true, 8, nothing, 1.0, chi)
say(@sprintf("  reference: exact SVD, dex 8 -> %.4f s, krylov %d", sec_ex, kry_ex))
println(CSV, @sprintf("%d,%d,exact,8,,1.0,%.6f,1.0,%d,0.0,,%.6f",
                      N, chi, sec_ex, kry_ex, 1.0))
say("")
say(@sprintf("  %5s %7s %6s %10s %8s %7s %11s %9s",
             "dex", "dover", "comp", "secs", "ratio", "kry", "dprof", "Dpre/dchi"))
for dex in (8, 16, 32), dover in (nothing, 0, 2), comp in (1.0, 0.5)
    s, k, pr = try
        timed(p0, false, dex, dover, comp, chi)
    catch e
        say("  dex=$dex dover=$dover comp=$comp THREW: " *
            sprint(showerror, e)[1:min(end, 160)]); continue
    end
    # `npre` EXACTLY AS `cbe_core.jl:587` COMPUTES IT -- a re-derived formula here would drift
    # from the code the moment either changes, and this column is the theoretical speedup ceiling.
    Dpre = dover === nothing ? ceil(Int, 1.2 * dex) : dex + dover
    frac = Dpre / (4 * chi)
    dprof = maximum(abs.(pr .- prof_ex))
    say(@sprintf("  %5d %7s %6.1f %10.4f %8.2f %7d %11.3e %9.4f",
                 dex, dover === nothing ? "def" : @sprintf("%.2f", dover), comp,
                 s, sec_ex / max(s, 1e-12), k, dprof, frac))
    println(CSV, @sprintf("%d,%d,sketch,%d,%s,%.1f,%.6f,%.4f,%d,%.6e,%d,%.6f",
                          N, chi, dex, dover === nothing ? "def" : string(dover), comp,
                          s, sec_ex / max(s, 1e-12), k, dprof, Dpre, frac))
    flush(CSV)
end
close(CSV)
say("")
say("READING IT:")
say("  `ratio` > 1 = the sketch is FASTER than the exact SVD at that configuration.")
say("  `kry` MUST match the reference's -- the root solve is pinned so that every arm applies the")
say("     operator the same number of times. A mismatch invalidates the seconds.")
say("  `dprof` is the accuracy cost against the EXACT step FROM THE SAME STATE. Jan's point 3")
say("     found the sketch accuracy-FREE at L=12, so a nonzero `dprof` here is news and bounds")
say("     how far `dex`/`dover` may be cut.")
say("  `Dpre/dchi` is the fraction of the full local block the probe touches -- the theoretical")
say("     ceiling on the speedup. A `ratio` far below `1/frac` means overhead, not the sketch.")
say("")
say("⚠ STILL OPEN from Jan's point 1: the DD-projected ranking governs only when both sides have")
say("  matching headroom (`joint = min(dex_l, dex_r, njl, njr)`); otherwise a side falls back to")
say("  its SINGLY projected candidates. Measuring that FALLBACK RATE needs a counter in")
say("  `cbe_core.jl` and is not instrumented yet.")
close(IO_)
