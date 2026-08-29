# LATTICE GAUGE THEORY COUPLED TO A BATH: the SCHWINGER model (1+1D QED) under dephasing.
#
# After a Jordan-Wigner transformation and integrating the gauge field out under open boundaries,
# the Schwinger Hamiltonian becomes a SPIN CHAIN whose electric term is ALL-TO-ALL:
#
#     H = x sum_n (S+_n S-_{n+1} + h.c.)  +  mu sum_n (-1)^n S^z_n  +  sum_{n=1}^{N-1} L_n^2
#     L_n = eps0 + sum_{k<=n} Q_k,   Q_k = S^z_k + (-1)^k / 2
#
# Expanding `sum_n L_n^2` (the derivation is written out in `schwinger_terms` below) gives
#
#     ALL-TO-ALL ZZ   J_zz[k,l] = 2 (N - l)            for k < l
#     ONE-BODY FIELD  h_k = 2 sum_{n>=k} a_n + mu(-1)^k,   a_n = eps0 + (n odd ? -1/2 : 0)
#     a CONSTANT      -- a global phase for the dynamics, dropped
#
# ⛔ THIS IS WHY `pair_mpo` NEEDED ONE COUPLING MATRIX PER VERTEX. The hopping is NEAREST
# NEIGHBOUR and the Coulomb term is ALL-TO-ALL: two different supports, so no single `J` with
# per-vertex scalars reproduces both. `Jzz` is passed separately here.
#
# ⛔ AND THE ONE-BODY FIELD GOES THROUGH A TWO-BODY BUILDER, as `O_j (x) I_{j+1}` on bonds 1..N-1
# plus `I_{N-1} (x) O_N` on the last. ⚠ The last site needs its own term because bond `j` only
# reaches site `j+1`; dropping it evolves a chain whose FINAL site has no field -- a different
# Hamiltonian whose error shrinks with `N`, so small tests miss it.
#
# ⚠ NO FREEZE RISK HERE. The hopping is range 1, so a two-site block contains a whole term and the
# dynamics bootstraps from a product state; the long-range ZZ terms then activate as the state
# entangles. (That is exactly what the interleaved doubled chain lacked, and it never moved.)
#
# VALIDATION TARGET (arXiv:2501.13675): the THERMALISATION TIME RISES with dissipation, with the
# background field `eps0`, and with the mass `mu`. ⚠ A TREND, not a number: that paper's lattice,
# boundary conditions and bath are not ours, and the write-up must say so.
#
# ⛔ CLUSTER, NOT LOCAL -- N = 18 fused sites at d = 4, and the MPO is wide (all-to-all ZZ), which
# is the point: §6 wants a wide generator for the rSVD sketch, and this is the widest one here.
#
#   julia --project=. benchmarks/lgt_open.jl [N] [gamma] [scheme] [x] [mu] [eps0] [tmax] [dt] [cap]

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))

const N      = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 18
const GAMMA  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.0
const SCHEME = length(ARGS) >= 3 ? ARGS[3]                 : "bug_interleaved"
const XHOP   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
const MU     = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.5
const EPS0   = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.0
const TMAX   = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 6.0
const DT     = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 0.05
const CAP    = length(ARGS) >= 9 ? parse(Int, ARGS[9])     : 256

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
iseven(N) || throw(ArgumentError("the staggered-fermion Schwinger chain needs an even N"))

"""
    schwinger_terms(N, x, mu, eps0) -> (Jxy, Jzz, hz)

The Schwinger spin chain, with the electric term expanded.

`L_n = a_n + S_n` where `a_n = eps0 + sum_{k<=n} (-1)^k/2` and `S_n = sum_{k<=n} S^z_k`, so

    sum_n L_n^2 = sum_n a_n^2                                  (constant, dropped)
                + 2 sum_n a_n S_n                              -> one-body field
                + sum_n S_n^2                                  -> ZZ, plus a constant from (S^z)^2

and `sum_n S_n^2 = const + 2 sum_{k<l} (N - l) S^z_k S^z_l`, since the pair `(k,l)` appears in
every `L_n` with `l <= n <= N-1`, which is `N - l` terms.

⚠ `Jxy = 2x` and NOT `x`: `fused_lindblad`'s XY vertices carry coefficient 1/2 each, so a coupling
`J` produces `J (S^xS^x + S^yS^y) = (J/2)(S^+S^- + S^-S^+)`. The Schwinger hopping is
`x (S^+S^- + h.c.)`, hence `J = 2x`.
"""
function schwinger_terms(N::Int, x::Float64, mu::Float64, eps0::Float64)
    Jxy = zeros(Float64, N, N)
    for n in 1:(N - 1)
        Jxy[n, n + 1] = 2 * x
    end
    Jzz = zeros(Float64, N, N)
    for k in 1:(N - 1), l in (k + 1):N
        l <= N - 1 || continue                      # only `n <= N-1` links carry electric energy
        Jzz[k, l] = 2.0 * (N - l)
    end
    a(n) = eps0 + (isodd(n) ? -0.5 : 0.0)
    hz = [2.0 * sum(a(n) for n in k:(N - 1); init = 0.0) + mu * (isodd(k) ? -1.0 : 1.0)
          for k in 1:N]
    return Jxy, Jzz, hz
end

const JXY, JZZ, HZ = schwinger_terms(N, XHOP, MU, EPS0)
const W, SHIFT = fused_lindblad(N, JXY, fill(GAMMA, N); hz = HZ, Jzz = JZZ)
const IMPS = identity_superket(N)

const OUTDIR = joinpath(@__DIR__, "results"); mkpath(OUTDIR)
const TAG = @sprintf("N%d_g%.2f_mu%.2f_e%.2f_%s", N, GAMMA, MU, EPS0, SCHEME)
const CSV = open(joinpath(OUTDIR, "lgt_open_$(TAG).csv"), "w")
println(CSV, "scheme,gamma,N,x,mu,eps0,t,condensate,npart,S_op,chi,trace_raw,krylov_cum,elapsed")

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

"""
The DIRAC VACUUM in the staggered convention: `Q_k = 0` on every site.

`Q_k = S^z_k + (-1)^k/2`, so `Q_k = 0` needs `S^z_k = -(-1)^k/2` -- DOWN on even sites, UP on odd.
⚠ This is the zero-charge reference state the condensate is measured against; starting from the
other staggering describes a completely filled band and thermalises differently.
"""
rho0() = rho0_product([isodd(k) ? :up : :down for k in 1:N])

"Chiral condensate, `(1/N) sum_n (-1)^n <S^z_n>` -- the order parameter that relaxes."
function condensate(rho)
    s = 0.0
    for j in 1:N
        s += (isodd(j) ? -1.0 : 1.0) * real(trace_rho(IMPS, rho, N; at = j))
    end
    return s / N
end

"Particle number, `sum_n <Q_n>` with `Q_n = S^z_n + (-1)^n/2`; zero in the Dirac vacuum."
function nparticles(rho)
    s = 0.0
    for j in 1:N
        s += real(trace_rho(IMPS, rho, N; at = j)) + (isodd(j) ? -0.5 : 0.5)
    end
    return s
end

say(l) = (println(stdout, l); flush(stdout))
say(@sprintf("Schwinger chain N = %d fused sites (d = 4): x = %.2f, mu = %.2f, eps0 = %.2f, gamma = %.2f",
             N, XHOP, MU, EPS0, GAMMA))
say(@sprintf("MPO virtual dim %d (all-to-all ZZ), cap %d, dt = %.3f to t = %.1f, scheme %s",
             maximum(mpo_virtual_dims(W)), CAP, DT, TMAX, SCHEME))
say(@sprintf("  %8s %12s %10s %10s %8s %14s %12s %9s",
             "t", "condensate", "n_part", "S_op", "chi", "Tr raw", "krylov", "elapsed"))

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
                        restore_trace!(IMPS, rho, N; maxdim = CAP, cutoff = 1e-10) :
                    TRACE_FIX === :rescale ? normalise_trace!(IMPS, rho, N) :
                                             real(trace_rho(IMPS, rho, N))
        kcum += sum(get_krylov_log(); init = 0)
        enable_krylov_log()          # ⚠ per-sweep counter: accumulate, never read as a total
    end
    if n % 10 == 0 || n == nsteps
        t = n * DT
        c = condensate(rho); np = nparticles(rho)
        S, _ = operator_entanglement(rho)
        tr = trace_raw; el = time() - t0
        say(@sprintf("  %8.2f %12.6f %10.4f %10.4f %8d %14.10f %12d %9.1f",
                     t, c, np, S, maximum(state_bond_dims(rho)), tr, kcum, el))
        println(CSV, @sprintf("%s,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.8f,%.8f,%.6f,%d,%.10f,%d,%.3f",
                              SCHEME, GAMMA, N, XHOP, MU, EPS0, t, c, np, S,
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
