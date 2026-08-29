# XX CHAIN UNDER DEPHASING: DOES BALLISTIC TRANSPORT BECOME DIFFUSIVE?
#
# The literature validation for the open-system machinery (Znidaric, NJP 12 043001; see also
# arXiv:2311.07375). From a DOMAIN WALL, the magnetisation transported across the centre grows
#
#     gamma = 0   ->  ~ t        (ballistic -- the XX chain is integrable and free)
#     gamma > 0   ->  ~ sqrt(t)  (diffusive -- dephasing destroys the ballistic channel)
#
# ⚠ THIS IS A TREND VALIDATION, NOT A NUMBER REPRODUCTION, and the write-up must say so: an
# 18-site open chain over a short window is not the thermodynamic limit the scaling law is stated
# in. What must reproduce is the CHANGE OF EXPONENT with gamma, which is a qualitative signature
# no amount of finite-size error turns on or off.
#
# ⛔ RUN THIS ON THE CLUSTER, NOT LOCALLY. L = 18 fused sites is d = 4 per site and the wall-clock
# column is a headline result; a local run would measure a laptop sharing itself with an editor.
# `submit_xx_open_transport.slurm` puts one (gamma, scheme) per array task so no two schemes ever
# time each other. Locally, correctness is settled at L = 2..4 against the exact dense propagator
# by `fused_lindblad.jl`.
#
#   julia --project=. benchmarks/xx_open_transport.jl [L] [gamma] [scheme] [tmax] [dt] [maxdim]
#
# Schemes: bug_interleaved (THE BUG arm) | tdvp2 | tdvp_cbe1s
#          plus the depth arms bug_interleaved_{m0,m3,tol4,tol6} -- SAME sweep, different Krylov depth

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))

const L      = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 18
const GAMMA  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.0
const SCHEME = length(ARGS) >= 3 ? ARGS[3]                 : "bug_interleaved"
const TMAX   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 8.0
const DT     = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.05
const CAP    = length(ARGS) >= 6 ? parse(Int, ARGS[6])     : 256

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

const OUTDIR = joinpath(@__DIR__, "results"); mkpath(OUTDIR)
const TAG = @sprintf("L%d_g%.2f_%s", L, GAMMA, SCHEME)
const CSV = open(joinpath(OUTDIR, "xx_open_$(TAG).csv"), "w")
println(CSV, "scheme,gamma,L,t,dM,S_op,chi,trace_raw,krylov_cum,elapsed")

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

# ── the model ────────────────────────────────────────────────────────────────────────────
# XX: delta = 0, nearest neighbour, J = 1.
const JM = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
const W, SHIFT = fused_lindblad(L, JM, fill(GAMMA, L); delta = 0.0)
const IMPS = identity_superket(L)
const HALF = L ÷ 2

"`rho0` = a DOMAIN WALL: up on the left half, down on the right."
rho0() = rho0_product([j <= HALF ? :up : :down for j in 1:L])

"""
Magnetisation transported across the centre.

⚠ MEASURED AGAINST `t = 0`, NOT AGAINST ZERO. The right half starts fully down, so
`sum_{j>L/2} <S^z_j>` begins at `-L/4`; subtracting the initial value makes `dM(0) = 0` and makes
the growth exponent readable directly.
"""
function transported(rho)
    s = 0.0
    for j in (HALF + 1):L
        s += real(trace_rho(IMPS, rho, L; at = j))
    end
    return s
end

say(l) = (println(stdout, l); flush(stdout))
say(@sprintf("XX chain, L = %d fused sites (d = 4), gamma = %.2f, scheme = %s", L, GAMMA, SCHEME))
say(@sprintf("MPO virtual dim %d, cap %d, dt = %.3f to t = %.1f",
             maximum(mpo_virtual_dims(W)), CAP, DT, TMAX))
say(@sprintf("  %8s %12s %10s %8s %14s %12s %10s",
             "t", "dM", "S_op", "chi", "Tr raw", "krylov", "elapsed"))

step! = stepper(SCHEME)
rho = rho0()
const DM0 = transported(rho)
nsteps = round(Int, TMAX / DT)
enable_krylov_log()
t0 = time()
kcum = 0
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
        dm = transported(rho) - DM0
        S, nst = operator_entanglement(rho)
        tr = trace_raw
        el = time() - t0
        say(@sprintf("  %8.2f %12.6f %10.4f %8d %14.10f %12d %10.1f",
                     t, dm, S, maximum(state_bond_dims(rho)), tr, kcum, el))
        println(CSV, @sprintf("%s,%.4f,%d,%.4f,%.8f,%.6f,%d,%.10f,%d,%.3f",
                              SCHEME, GAMMA, L, t, dm, S, maximum(state_bond_dims(rho)),
                              tr, kcum, el))
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
