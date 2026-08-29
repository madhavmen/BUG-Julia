# WHERE IS THE CROSSOVER? EXACT SVD vs RANDOMISED SKETCH AS A FUNCTION OF chi.
#
# The Schwinger chain coupled to a bath is the best case this project has for RSVD-CBE, because
# two things hold at once:
#   1. THE MPO IS WIDE -- the electric term is ALL-TO-ALL (`J_zz[k,l] = 2(N-l)`), so `pair_mpo`
#      carries a transport channel per open bond and the virtual dimension grows with `N`.
#   2. THE PHYSICAL DIMENSION IS 4 -- the fused (ket, bra) site doubles the local space, so the
#      full local block is `d*chi = 4*chi` against a sketch of `npre ~ 1.2*dex`.
#
# ⛔ AND IT ONLY PAYS WITH A FIXED `dex`, NEVER WITH `growth`:
#     budget = dex <= 0 ? ceil(growth*dmax) - r : dex        (cbe_core.jl)
#     npre  ~ 1.2 * dex
# At `growth = 2.0` a bond at rank `chi` gets `budget ~ chi`, so the sketch is a CONSTANT FRACTION
# of full (60% at d=2, 30% at d=4) and can never win however large chi grows. With `dex` FIXED,
#     npre / (d*chi) ~ 1.2*dex / (4*chi)  ->  0
# 1.9% at chi = 128, dex = 8. ⚠ A fixed `dex` is not a handicap: this repo's own budget scan
# (XX L=10) found `dex = 8` giving err 1.17e-06, IDENTICAL to `dex = 16` and `dex = 64`.
#
# ── TWO DESIGN DECISIONS, BOTH FORCED BY WHAT THE FIRST ATTEMPT GOT WRONG ─────────────────
#
# ⛔ 1. TIME BOTH ARMS FROM THE **SAME STATE**, NOT ALONG THEIR OWN TRAJECTORIES. Timing whole
# runs cannot locate a crossover: the arms build different bases, so they reach different chi at
# the same time and the comparison silently drifts to different problem sizes. Here a state is
# grown ONCE to each target chi and both arms are timed stepping from that identical state.
#
# ⛔ 2. PIN THE ROOT SOLVE SO THE OPERATOR WORK IS IDENTICAL. MEASURED in the first attempt:
# `krylov` came out 77 (exact) vs 90 (sketched). With `krylov_tol = 0` the growth frame is
# deterministic, so the difference is the ROOT `expv` exiting after different numbers of
# iterations -- the arms build different bases, so the Krylov solve converges differently. Benign
# in itself, but it means the two arms apply the operator a different number of times and the
# seconds are then not comparable. `tol = 0.0` removes the convergence exit so both burn exactly
# `maxiter`, isolating what the sketch actually changes: the SIZE of the CBE contraction, not the
# NUMBER of applications. The `krylov` columns are printed so the match is checked, not assumed.
#
# ⚠ `dprof` IS THE ACCURACY GUARD. A sketch that is faster and WORSE is not a result. It compares
# the sketched step against the EXACT step FROM THE SAME STATE, so it is a pure sketch error.
#
# ⚠ WALL CLOCK IS THE MEASUREMENT: this must be the only heavy job on the node, threads pinned.
# Operator applications cannot answer the question -- by construction they are now equal.
#
#   julia --project=. benchmarks/rsvd_lgt.jl [N] [gamma] [dex] [reps] [chis...]

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 20
const GAMMA = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.1
const DEX   = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 8   # accuracy floor
const REPS  = length(ARGS) >= 4 ? parse(Int, ARGS[4])     : 3
const CHIS  = length(ARGS) >= 5 ? parse.(Int, split(ARGS[5], ",")) :
                                  [16, 32, 64, 128, 192, 256]
const DT    = 0.05
const MI    = 20          # FIXED iteration count; see decision 2 above

set_symmetry!(:none)
iseven(N) || throw(ArgumentError("the staggered-fermion Schwinger chain needs an even N"))

const OUT = joinpath(@__DIR__, "results", "rsvd_lgt_N$(N).txt"); mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))
const CSV = open(joinpath(@__DIR__, "results", "rsvd_lgt_N$(N).csv"), "w")
println(CSV, "N,dex,chi,secs_exact,secs_sketch,ratio,kry_exact,kry_sketch,dprof,mpodim,frac")

"The Schwinger terms; the electric expansion is derived in `lgt_open.jl`."
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

"""
One step of the arm under test.

⚠ `tol = 0.0` AND A FIXED `maxiter`: no convergence exit, so BOTH arms perform exactly `MI`
operator applications in the root solve and the seconds differ only by the CBE contraction size.
"""
step_arm!(p, exact::Bool, cap::Int) =
    cbe_bug_step!(p, W, ComplexF64(DT); exact = exact, dex = DEX,
                  # ⛔ THE MEASURED-BEST SKETCH, not the defaults. `rsvd_optimal.jl` at N=8,
                  # chi=32 swept dex x dover x comp_ratio with matched krylov:
                  #     dex  8, dover 0, comp 1.0 -> ratio 1.10, dprof 1.09e-07   <- best
                  #     dex  8, dover def, comp 1.0 -> 1.04
                  #     dex 16 -> 0.94,  dex 32 -> 0.64        (`npre` ∝ `dex`)
                  # `comp_ratio` is speed-NEUTRAL but not accuracy-neutral: 1.0 gives 1.09e-07
                  # against 0.5's 2.38e-07, so the pure-complement choice this repo makes AGAINST
                  # the MATLAB's advice is both faster and 2.4x more accurate here.
                  # ⚠ Measuring a crossover with a mediocre sketch would understate rSVD; the
                  # crossover has to be that of the BEST configuration available.
                  dover = 0, comp_ratio = 1.0,
                  krylov_basis = 3, krylov_tol = 0.0, hermitian = false,
                  maxdim = cap, trunc_thresh = 1e-10, maxiter = MI, tol = 0.0)

"""
Grow a state until its largest bond reaches `target` (or the evolution stops growing).

⚠ WHICH ARM GROWS IT DOES NOT MATTER -- it is a STATE, not a result, and both arms are then timed
from the identical object. Using the exact arm keeps it reproducible.
⚠ `gamma` SMALL ON PURPOSE: dephasing SUPPRESSES operator entanglement, so a strongly damped run
never reaches the chi where the sketch could pay, and a null result there would say nothing.
"""
function grow_to(target::Int; maxsteps::Int = 400)
    p = rho0(); last = 0; stalled = 0
    for _ in 1:maxsteps
        step_arm!(p, true, target)
        SHIFT != 0.0 && apply_shift!(p, exp(DT * SHIFT))
        restore_trace!(IMPS, p, N; maxdim = target, cutoff = 1e-10)
        chi = maximum(state_bond_dims(p))
        chi >= target && return p, chi
        stalled = (chi == last) ? stalled + 1 : 0
        stalled > 25 && return p, chi      # entanglement saturated below the target
        last = chi
    end
    return p, maximum(state_bond_dims(p))
end

"Site-resolved `Tr(S^z_j rho)` -- what the two arms must agree on."
profile(rho) = [real(trace_rho(IMPS, rho, N; at = j)) for j in 1:N]

"Time `REPS` steps from the SAME state; returns the best (least contended) time."
function timed(p0, exact::Bool, cap::Int)
    best = Inf; kry = 0; prof = nothing
    for _ in 1:REPS
        p = copy(p0)
        enable_krylov_log()
        t0 = time_ns()
        step_arm!(p, exact, cap)
        el = (time_ns() - t0) / 1e9
        k = sum(get_krylov_log(); init = 0)
        disable_krylov_log()
        el < best && (best = el)
        kry = k; prof = profile(p)
    end
    return best, kry, prof
end

say(@sprintf("Schwinger chain N = %d FUSED sites (d = 4), gamma = %.2f, dex = %d, reps = %d",
             N, GAMMA, DEX, REPS))
say(@sprintf("MPO virtual dim %d (all-to-all ZZ) -- the widest generator in this study.", MPODIM))
say("")
say("⚠ WARMUP FIRST so the first timed row is not a JIT measurement (the previous attempt read")
say("  141.96 s for a step that actually takes 0.4 s).")
let p = rho0()
    step_arm!(p, true, 16); step_arm!(p, false, 16)
end
say("")
say(@sprintf("  %6s %11s %11s %8s %9s %9s %11s %9s",
             "chi", "exact s", "sketch s", "ratio", "kry_ex", "kry_sk", "dprof", "npre/dchi"))
for target in CHIS
    p0, chi = grow_to(target)
    if chi < target
        say(@sprintf("  %6d  -- entanglement saturated at chi = %d; target unreachable, SKIPPED",
                     target, chi))
        continue
    end
    se, ke, pe = timed(p0, true, target)
    ss, ks, ps = timed(p0, false, target)
    dprof = maximum(abs.(ps .- pe))
    frac = 1.2 * DEX / (4 * chi)
    say(@sprintf("  %6d %11.4f %11.4f %8.2f %9d %9d %11.3e %9.4f",
                 chi, se, ss, se / max(ss, 1e-12), ke, ks, dprof, frac))
    println(CSV, @sprintf("%d,%d,%d,%.6f,%.6f,%.4f,%d,%d,%.6e,%d,%.6f",
                          N, DEX, chi, se, ss, se / max(ss, 1e-12), ke, ks, dprof, MPODIM, frac))
    flush(CSV)
end
close(CSV)
say("")
say("READING IT:")
say("  `ratio` > 1 means the SKETCH IS FASTER. The CROSSOVER is the chi where it first exceeds 1.")
say("  `kry_ex` MUST EQUAL `kry_sk` -- that is what the pinned root solve buys. If they differ the")
say("     arms are doing different work and the seconds are not comparable.")
say("  `dprof` is a pure sketch error (same starting state), and must stay negligible or the")
say("     speedup is being bought with accuracy.")
say("  `npre/dchi` is the fraction of the full local block the sketch touches. A null result at")
say("     large `npre/dchi` says nothing about rSVD -- no speedup was available to find.")
close(IO_)
