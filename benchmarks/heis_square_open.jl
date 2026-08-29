# HEISENBERG ON A 5x5 SQUARE CYLINDER, UNDER DEPHASING.
#
# 2D through a SNAKE MAPPING TO AN MPS -- no PEPS. The short direction wraps, so each site is
# 4-coordinate in the bulk, and the snake makes the couplings a mixture of range 1 (along the
# snake) and range ~Ly (the wrap and the rungs). `square_cylinder_couplings` builds the matrix;
# the operator is `pair_mpo`, which takes an arbitrary `J[i,j]`.
#
# ⚠ THE RANGE MIXTURE IS WHY THIS DOES NOT FREEZE, and it is worth stating because the doubled
# chain taught it the hard way. A two-site update cannot start a term whose two operators are not
# both inside the block -- the environment contribution is then a one-site expectation of a
# charge-changing operator, and `<S^-> = 0` on a product state. The square cylinder carries
# RANGE-1 couplings along the snake, which bootstrap the state; once it is entangled the
# longer-range rung and wrap terms activate normally. (The interleaved doubled chain had NO
# range-1 term at all, and never moved: `max|rho_mps(T) - rho(0)| = 0.0000e+00` exactly.)
#
# ⛔ THE FUSED LAYOUT PRESERVES RANGE EXACTLY. Fused sites map 1:1 onto SYSTEM sites, so every
# coupling keeps the range it had in the closed model -- which is what makes this a drop-in.
#
# ⛔ CLUSTER, NOT LOCAL. 25 fused sites at d = 4; wall clock is a headline column. Correctness is
# settled locally at L = 2..4 against the exact dense propagator by `fused_lindblad.jl`.
#
#   julia --project=. benchmarks/heis_square_open.jl [Lx] [Ly] [gamma] [scheme] [tmax] [dt] [cap]

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))

const LX     = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 5
const LY     = length(ARGS) >= 2 ? parse(Int, ARGS[2])     : 5
const GAMMA  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
const SCHEME = length(ARGS) >= 4 ? ARGS[4]                 : "bug_interleaved"
const TMAX   = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 4.0
const DT     = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.05
const CAP    = length(ARGS) >= 7 ? parse(Int, ARGS[7])     : 256
const PERY   = length(ARGS) >= 8 ? parse(Bool, ARGS[8])    : false

# ⛔ TRACE CORRECTION, ON BY DEFAULT, AND THE VARIANT MATTERS.
#   :along_I  add the deficit back ALONG |I>> only  -> exact Tr AND 0.745x the state error
#   :rescale  divide the whole state by Tr          -> exact Tr but 1.91x the state error
#   :none     leave it                              -> Tr drifts at O(dt^2)
# All three MEASURED on the fused L=3 chain at full rank (`benchmarks/trace_preservation.jl`).
# `:along_I` is not a wash -- it is strictly better than doing nothing, because it repairs the
# conservation law WITHOUT multiplying the traceless part, which is what makes `:rescale`
# overcorrect.
const TRACE_FIX = :along_I

set_symmetry!(:none)

const L  = LX * LY
# ⛔ OPEN IN y BY DEFAULT AT 5x5, AND THE REASON IS PHYSICS, NOT CONVENIENCE. Wrapping an ODD
# circumference makes the lattice NON-BIPARTITE: the wrap bond joins `y = Ly` to `y = 1`, and for
# odd `Ly` those carry the SAME sublattice sign, so a Neel pattern is frustrated on every wrap
# bond. The staggered magnetisation would then be measuring a pattern the lattice cannot support
# -- it would still return a number, and the number would still decay under dephasing, which is
# exactly the kind of plausible-but-wrong result to avoid. A 5x5 OPEN patch is bipartite and is
# still a genuine 2D-on-MPS setup with no PEPS. Pass `true` as the 8th argument for a cylinder,
# and use an EVEN `Ly` when doing so.
(PERY && isodd(LY)) && @warn "periodic_y with odd Ly is NON-BIPARTITE: the staggered " *
                             "observable is frustrated on the wrap bond. Use an even Ly."
const JM = square_cylinder_couplings(LX, LY; periodic_y = PERY)

const OUTDIR = joinpath(@__DIR__, "results"); mkpath(OUTDIR)
const TAG = @sprintf("%dx%d_g%.2f_%s", LX, LY, GAMMA, SCHEME)
const CSV = open(joinpath(OUTDIR, "heis_square_open_$(TAG).csv"), "w")
println(CSV, "scheme,gamma,Lx,Ly,t,Mstag,S_op,chi,trace_raw,krylov_cum,elapsed")

"""
One step of the requested scheme; all of them take the same signature.

⛔ ONE BUG SCHEME ACROSS EVERY RUN IN THIS STUDY, AND IT IS `bug_interleaved` -- the same name and
the same settings the closed suite (`cbe_sweeps_l16.jl`) uses, so the open-system and closed-system
BUG numbers are the same method. `cbe_bug`'s sweep IS the interleaved ordering
(`sweeps/cbe_bug.jl:447`, matching `TDVPSweepCBE1Si.m:30/:34`); the DECOUPLED variant no longer
exists -- its flag lived on the K-step path, removed with `kaug`/`rexpand` on 2026-08-24.

⚠ THE SUFFIXED ARMS ARE THE SAME SWEEP AT DIFFERENT KRYLOV DEPTHS, not other integrators. Any
comparison against `tdvp2` / `tdvp_cbe1s` must use the UNSUFFIXED `bug_interleaved`, or it reports
a tuning difference as a method difference.
"""
function stepper(name::AbstractString)
    kw = (; maxdim = CAP, trunc_thresh = 1e-10, maxiter = 30, hermitian = false)
    # ⛔ `growth = 4.0`, NOT THE SHIPPED 2.0, AND THIS SETS THE CONVERGENCE ORDER.
    # `budget = ceil(growth*dmax) - r`; "a bond at the neighbourhood max may DOUBLE" is the right
    # unit for a d = 2 site, and a FUSED d = 4 site can QUADRUPLE a bond. MEASURED at L = 3
    # against the exact `exp(T*L)`:
    #     growth 2, adaptive depth 30 : 2.95e-03, order 1.00   <- the shipped default
    #     growth 4, adaptive depth 30 : 9.56e-06, order 2.00
    # At `growth = 2` the arm collapses to the `m = 0` result even though it computed a depth-30
    # basis: the budget cannot ADMIT the Krylov directions, so the depth is paid for and thrown
    # away. Depth alone does not rescue it and budget alone does not either -- `m = 0` at
    # `growth = 4` is still order 1.00. Both are required.
    # ⚠ Leaving the default here would have made every open-system run silently FIRST order.
    bugkw = (; growth = 4.0)
    name == "tdvp2"      && return (p, W, tau) -> tdvp2_step!(p, W, tau; kw...)
    name == "tdvp_cbe1s" && return (p, W, tau) -> tdvp_cbe1s_step!(p, W, tau; kw...)
    # LIBRARY DEFAULTS (`krylov_basis = 30`, `krylov_tol = 1e-6`), passed by omission so this arm
    # cannot silently fork from the library the day a default changes.
    name == "bug_interleaved" &&
        return (p, W, tau) -> cbe_bug_step!(p, W, tau; exact = true, bugkw..., kw...)
    name == "bug_interleaved_m0" && return (p, W, tau) -> cbe_bug_step!(p, W, tau; exact = true,
                                          krylov_basis = 0, krylov_tol = 0.0, bugkw..., kw...)
    name == "bug_interleaved_m3" && return (p, W, tau) -> cbe_bug_step!(p, W, tau; exact = true,
                                          krylov_basis = 3, krylov_tol = 0.0, bugkw..., kw...)
    # ⚠ `krylov_basis` is the CEILING and `krylov_tol` the stopping rule; a ceiling of 0 would
    # silently turn an "adaptive" arm into m = 0.
    name == "bug_interleaved_tol4" && return (p, W, tau) -> cbe_bug_step!(p, W, tau; exact = true,
                                          krylov_basis = 30, krylov_tol = 1e-4, bugkw..., kw...)
    name == "bug_interleaved_tol6" && return (p, W, tau) -> cbe_bug_step!(p, W, tau; exact = true,
                                          krylov_basis = 30, krylov_tol = 1e-6, bugkw..., kw...)
    throw(ArgumentError("unknown scheme $name"))
end

const W, SHIFT = fused_lindblad(L, JM, fill(GAMMA, L); delta = 1.0)   # Heisenberg: delta = 1
const IMPS = identity_superket(L)

"""
Sublattice sign for chain site `n`.

⚠ THE INDEXING IS READ OFF THE MODEL FILE, NOT GUESSED. `square_cylinder_couplings` uses the plain
COLUMN-MAJOR map `site(x, y) = (x - 1) * Ly + y` -- NOT a boustrophedon. Assuming a snake here
puts the sublattice sign on the wrong sites for every even column, which yields a perfectly
well-behaved number that decays under dephasing and is a DIFFERENT observable.

⚠ AND THE SIGN COMES FROM `(x + y)`, NOT FROM `n`. With `Ly` odd, `(-1)^n` and `(-1)^(x+y)` come
apart at every column boundary.
"""
function stagger(LX, LY)
    s = zeros(Float64, LX * LY)
    for x in 1:LX, y in 1:LY
        n = (x - 1) * LY + y                       # column-major, matching the model file
        s[n] = iseven(x + y) ? 1.0 : -1.0
    end
    return s
end
const SGN = stagger(LX, LY)

"Neel start on the SNAKE, using the same (x, y) sublattice sign the observable uses."
rho0() = rho0_product([SGN[n] > 0 ? :up : :down for n in 1:L])

"Staggered magnetisation, normalised per site."
function mstag(rho)
    s = 0.0
    for j in 1:L
        s += SGN[j] * real(trace_rho(IMPS, rho, L; at = j))
    end
    return s / L
end

say(l) = (println(stdout, l); flush(stdout))
say(@sprintf("Heisenberg %dx%d square cylinder = %d fused sites (d = 4), gamma = %.2f, %s",
             LX, LY, L, GAMMA, SCHEME))
say(@sprintf("MPO virtual dim %d, cap %d, dt = %.3f to t = %.1f",
             maximum(mpo_virtual_dims(W)), CAP, DT, TMAX))
say(@sprintf("  %8s %12s %10s %8s %14s %12s %10s",
             "t", "M_stag", "S_op", "chi", "Tr raw", "krylov", "elapsed"))

step! = stepper(SCHEME)
rho = rho0()
nsteps = round(Int, TMAX / DT)
enable_krylov_log()
t0 = time(); kcum = 0
trace_raw = 1.0        # step 0 has had no step to lose trace in
for n in 0:nsteps
    # ⛔ `global kcum`, AND IT IS NOT OPTIONAL. At top level a `for` body is its OWN SOFT SCOPE,
    # so `kcum += ...` creates a NEW LOCAL and the outer name is never updated -- Julia raises
    # `UndefVarError: kcum not defined in local scope` rather than silently zeroing the counter,
    # which is the one mercy here. This repo has hit the same trap twice before.
    global kcum, trace_raw
    if n > 0
        step!(rho, W, ComplexF64(DT))
        SHIFT != 0.0 && apply_shift!(rho, exp(DT * SHIFT))
        # ⛔ RESTORE THE TRACE ALONG `|I>>`, NOT BY RESCALING. A Galerkin scheme cannot conserve
        # `Tr rho`: it evolves `exp(tau*P L P)` and `Tr(P rho) = <<P I|rho>>`, so the trace
        # survives only if the basis CONTAINS `|I>>`, which CBE/BUG never puts there. Two remedies,
        # both MEASURED at full rank on the fused L=3 chain:
        #     divide by Tr        -> Tr exact, state error 1.91x WORSE (it multiplies the
        #                            TRACELESS part too, and the BUG error is not a pure trace
        #                            deficit, so it OVERCORRECTS)
        #     add eps*|I>>/2^(L/2) -> Tr exact AND state error 0.745x, i.e. 25% BETTER
        # The second leaves the traceless part exactly alone, which is why it is the minimal
        # correction rather than a trade.
        # ⚠ UNDER A BINDING CAP IT IS NOT EXACT: the closing truncation sheds part of the added
        # direction (1-Tr 2.17e-02 -> 2.49e-03 at cap 2, ~9x better but not machine precision).
        # `Tr raw` is therefore still logged -- it is the trace BEFORE the correction, i.e. what
        # the step itself left behind.
        trace_raw = TRACE_FIX === :along_I ?
                        restore_trace!(IMPS, rho, L; maxdim = CAP, cutoff = 1e-10) :
                    TRACE_FIX === :rescale ? normalise_trace!(IMPS, rho, L) :
                                             real(trace_rho(IMPS, rho, L))
        kcum += sum(get_krylov_log(); init = 0)
        enable_krylov_log()          # ⚠ per-sweep counter: accumulate, never read as a total
    end
    if n % 10 == 0 || n == nsteps
        t = n * DT
        m = mstag(rho); S, _ = operator_entanglement(rho)
        tr = trace_raw; el = time() - t0
        say(@sprintf("  %8.2f %12.6f %10.4f %8d %14.10f %12d %10.1f",
                     t, m, S, maximum(state_bond_dims(rho)), tr, kcum, el))
        println(CSV, @sprintf("%s,%.4f,%d,%d,%.4f,%.8f,%.6f,%d,%.10f,%d,%.3f",
                              SCHEME, GAMMA, LX, LY, t, m, S,
                              maximum(state_bond_dims(rho)), tr, kcum, el))
        flush(CSV)
    end
end
disable_krylov_log()
close(CSV)
say("")
say("⛔ `Tr raw` IS THE TRACE THE STEP ITSELF LEFT BEHIND, read BEFORE the correction. The state")
say("   carried forward has the deficit added back ALONG |I>> -- NOT rescaled. Rescaling divides")
say("   the traceless part too and MEASURED 1.91x WORSE; the additive form is 0.745x, i.e. it")
say("   fixes the conservation law and improves accuracy at the same time.")
say("⚠  `tdvp2` conserves Tr exactly (a product of local exponentials, each annihilating <<I|).")
say("   BUG freezes its frames and evolves only the bond matrix, so it lives in the subspace")
say("   {W X Z^dag}; the trace survives only if |I>> lies in that span. See")
say("   benchmarks/bug_exactness_probe.jl for whether that is structural or a defect.")
say("⚠  Do NOT read `Tr raw` -> 1 as accuracy. It is one linear functional of the error.")
