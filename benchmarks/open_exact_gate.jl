# EVERY open model against an EXACT reference -- the gate that gives `square_open` and `lgt` a
# number instead of a trend.
#
#   MODELS=xx_open,square_open,lgt SITES=6 T=0.5 julia --project=. benchmarks/open_exact_gate.jl
#
# ── WHY THIS FILE EXISTS ─────────────────────────────────────────────────────────────────────
#
# `xx_open` is the only open model with a closed form (`xx_dephasing_sz`: onsite dephasing closes
# the single-particle correlation matrix, so `dC/dt = -i[h,C] - gamma*(1-delta_ij)*C_ij` is an exact
# `L x L` ODE). `square_open` and `lgt` had nothing but published trends, and a trend cannot tell
# "the integrator drifted" from "the generator is wrong" -- which is the distinction the whole
# open-system comparison rests on.
#
# The Liouvillian is a `4^N x 4^N` MATRIX, so at small `N` it can simply be exponentiated. That is
# an exact reference for ANY of these models, sharing no code with the MPO, the sweeps, the
# truncation or the Krylov solver. See `tests/common/dense_lindblad.jl` for why this is an exact
# diagonalisation rather than the our-own-finest-grid mistake.
#
# ── THE CHECKS RUN IN THIS ORDER, AND THE ORDER IS THE ARGUMENT ──────────────────────────────
#
#   A  THE MAPPING, ON AN ASYMMETRIC STATE. `fused_dense_state` and `dense_sz_profile` must agree
#      with `S.profile` on the SAME state. ⛔ TESTED ON AN ASYMMETRIC PRODUCT STATE, NOT NEEL.
#      Dense and MPS bit orderings in this repo have been MIRRORED before, and Neel, domain-wall and
#      dimer states are all invariant under the mirror, so each of them PASSES a broken mapping.
#      Nothing downstream means anything until this check is green.
#
#   B  THE DENSE OPERATOR vs THE ANALYTIC ONE, on `xx_open`. Two exact references, derived
#      independently -- a free-fermion ODE and a dense matrix exponential -- compared against each
#      other. This is the check that PROMOTES the dense operator: once it reproduces the closed form
#      on the one model that has one, it can be trusted on the models that do not.
#
#   C  THE THREE ARMS vs THE DENSE REFERENCE, on a dt ladder, for every requested model. Only now
#      is an arm's error meaningful. `ratio ~ 4` per halving of dt is second order and says THE
#      OPERATOR IS RIGHT; `ratio ~ 1` (error flat in dt) says it is wrong, and no amount of rank or
#      Krylov depth will fix it.
#
# ⚠ THE CAP MUST NOT BIND. At N = 6 fused, full rank is 4^3 = 64, so `CAP = 256` cannot truncate
# anything and every number here is the INTEGRATOR's, not the truncation's. That is the point of
# running the gate small: a rank-limited gate cannot separate the two.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "dense_lindblad.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const OUT     = joinpath(@__DIR__, "results")
const MODELS  = split(get(ENV, "MODELS", "xx_open,square_open,lgt"), ",")
const SITES   = parse(Int,     get(ENV, "SITES", "6"))
const T       = parse(Float64, get(ENV, "T",     "0.5"))
const GAMMA   = parse(Float64, get(ENV, "GAMMA", "0.1"))
const CAP     = parse(Int,     get(ENV, "CAP",   "256"))
const MAXITER = parse(Int,     get(ENV, "MAXITER", "12"))
const DTS     = [parse(Float64, s) for s in split(get(ENV, "DTS", "0.05,0.025,0.0125"), ",")]

mkpath(OUT)
const LOG = open(joinpath(OUT, "open_exact_gate.txt"), "w")
# ⛔ MIRROR EVERY LINE TO A FILE AND FLUSH. Julia BLOCK-BUFFERS a redirected stdout, so a long run
# whose output is piped shows NOTHING until it exits -- and a crash then loses the whole log. This
# repo has already lost a 31-minute run that way.
say(l) = (println(LOG, l); flush(LOG); println(stdout, l); flush(stdout))

function arm_closures(S, cap)
    return [
        ("tdvp2", (p, tau) -> tdvp2_step!(p, S.W, tau; maxdim = cap, trunc_thresh = 1e-12,
                                          maxiter = MAXITER, hermitian = false)),
        ("tdvp_cbe1s", (p, tau) -> tdvp_cbe1s_step!(p, S.W, tau; maxdim = cap,
                                                    trunc_thresh = 1e-12, maxiter = MAXITER,
                                                    hermitian = false)),
        ("cbe_bug", (p, tau) -> cbe_bug_step!(p, S.W, tau; exact = false, dex = 8, dover = 4,
                                              comp_ratio = 1.0, krylov_basis = 3,
                                              krylov_tol = 0.0, maxdim = cap,
                                              trunc_thresh = 1e-12, maxiter = MAXITER,
                                              hermitian = false)),
    ]
end

"""
    propagators(Lsup, base, kmax) -> Vector of exp(Lsup*base*2^i), i = 0..kmax

⛔ ONE MATRIX EXPONENTIAL, THEN SQUARINGS. `exp(L*t)^2 = exp(L*2t)` is an EXACT identity, so every
time point that is a power-of-two multiple of `base` costs one matrix multiply instead of a fresh
`exp`. At N = 6 the operator is 4096 x 4096 and a single `exp` is minutes (scaling-and-squaring is
~20 matmuls of 4096^3); the naive version of this file called it SIX times. The time points below
are therefore chosen as `base * 2^i` rather than round numbers.
"""
function propagators(Lsup, base::Float64, kmax::Int)
    E = exp(Lsup * base)
    out = [E]
    for _ in 1:kmax
        push!(out, out[end] * out[end])
    end
    return out
end

"""
Check A: the dense<->MPS mapping, on an ASYMMETRIC state.

⛔ THE STATE CHOICE IS THE TEST. `[:up, :up, :down, :up, :down, :down]` is not invariant under
reversal, so a mirrored bit convention shows up as a REVERSED profile. Neel would pass either way.
"""
function check_mapping(N)
    spins = [:up, :up, :down, :up, :down, :down][1:N]
    rho = rho0_product(spins)
    imps = identity_superket(N)
    mps_prof = [real(trace_rho(imps, rho, N; at = j)) for j in 1:N]
    v = fused_dense_state(rho)
    dns_prof = dense_sz_profile(v, N)
    want = [s === :up ? 0.5 : -0.5 for s in spins]
    dev_mps = maximum(abs.(mps_prof .- want))
    dev_dns = maximum(abs.(dns_prof .- want))
    dev_mut = maximum(abs.(mps_prof .- dns_prof))
    say(@sprintf("A  mapping on the ASYMMETRIC state %s", string(spins)))
    say(@sprintf("     MPS profile vs analytic  : %.3e", dev_mps))
    say(@sprintf("     dense profile vs analytic: %.3e", dev_dns))
    say(@sprintf("     MPS vs dense             : %.3e   %s",
                 dev_mut, dev_mut < 1e-12 ? "PASS" : "FAIL -- MIRRORED BIT ORDER"))
    say(@sprintf("     dense Tr rho             : %.12f", dense_trace(v, N)))
    say("")
    return dev_mut < 1e-12
end

"""
Check B: the dense Liouvillian against the ANALYTIC free-fermion solution, on `xx_open`.

Two independently derived exact references. Agreement promotes the dense operator to a usable
reference for `square_open` and `lgt`, which have no closed form.
"""
function check_dense_vs_analytic(N, gamma)
    S = setup("xx_open"; sites = N, dt = 0.05, gamma = gamma, cap = CAP)
    p = S.params
    Lsup = dense_lindblad_super(S.L, p.Jm, p.gammas; delta = p.delta, hz = p.hz, Jzz = p.Jzz)
    v0 = fused_dense_state(S.psi0)
    say(@sprintf("B  dense Liouvillian vs the ANALYTIC free-fermion ODE   (xx_open, N = %d, gamma = %.3f)", S.L, gamma))
    say("      t        max|dSz|    |Tr-1| (dense)")
    ok = true
    # Powers of two times T/4, so ONE `exp` covers all of them -- see `propagators`.
    Es = propagators(Lsup, T / 4, 3)
    for (i, E) in enumerate(Es)
        t = (T / 4) * 2^(i - 1)
        vt = E * v0
        got = dense_sz_profile(vt, S.L)
        # The Neel occupation `1:2:L` is the one `rho0_product` builds in `_open`.
        want = xx_dephasing_sz(S.L, t; J = 1.0, gamma = gamma, occupied = 1:2:S.L)
        dev = maximum(abs.(got .- want))
        say(@sprintf("   %6.3f    %.4e    %.2e", t, dev, abs(dense_trace(vt, S.L) - 1.0)))
        ok &= dev < 1e-10
    end
    say(ok ? "     PASS -- the dense operator reproduces the closed form; it is now a REFERENCE." :
             "     FAIL -- the two exact references DISAGREE; fix the operator before reading C.")
    say("")
    return ok
end

"Check C: the three arms against the dense reference, on a dt ladder."
function check_arms(nm, N, gamma)
    S0 = setup(nm; sites = N, dt = DTS[1], gamma = gamma, cap = CAP)
    p = S0.params
    say(@sprintf("C  %s   %s", nm, S0.label))
    say(@sprintf("     N = %d, gamma = %.3f, cap = %d (full rank is %d -- cannot bind), T = %.3f",
                 S0.L, gamma, CAP, 4^(S0.L ÷ 2), T))
    Lsup = dense_lindblad_super(S0.L, p.Jm, p.gammas; delta = p.delta, hz = p.hz, Jzz = p.Jzz)
    v0 = fused_dense_state(S0.psi0)
    vT = exp(Lsup * T) * v0                  # ONE exponential; do not recompute it for the trace
    want = dense_sz_profile(vT, S0.L)
    say(@sprintf("     reference |Tr-1| = %.2e   (exact matrix exponential)",
                 abs(dense_trace(vT, S0.L) - 1.0)))

    # ⛔ WARMUP: without it the first arm of the first dt absorbs both TDVP arms' JIT.
    for (_, step!) in arm_closures(S0, CAP)
        try; step!(copy(S0.psi0), S0.tau); catch; end
    end

    say("      arm              dt      max|dSz|    ratio      |Tr-1|       chi     secs")
    for (arm, _) in arm_closures(S0, CAP)
        prev = NaN
        for dt in DTS
            S = setup(nm; sites = N, dt = dt, gamma = gamma, cap = CAP)
            step! = Dict(arm_closures(S, CAP))[arm]
            psi = copy(S.psi0)
            nsteps = max(1, round(Int, T / dt))
            dtr = 0.0
            t0 = time_ns()
            for _ in 1:nsteps
                step!(psi, S.tau)
                S.shift!(psi)
                dtr = max(dtr, abs(real(trace_rho(S.imps, psi, S.L; at = 0)) - 1.0))
                S.restore!(psi)
            end
            secs = (time_ns() - t0) / 1e9
            dev = maximum(abs.(S.profile(psi)[1:S.L] .- want))
            say(@sprintf("   %-12s %7.4f    %.4e  %7s    %.2e    %6d  %7.1f",
                         arm, dt, dev, isnan(prev) ? "-" : @sprintf("%.2f", prev / dev),
                         dtr, maximum(bond_dims(psi)), secs))
            prev = dev
        end
    end
    say("")
end

function main()
    say(@sprintf("open_exact_gate -- N = %d, T = %.3f, gamma = %.3f, cap = %d, maxiter = %d",
                 SITES, T, GAMMA, CAP, MAXITER))
    say(@sprintf("dt ladder = %s", string(DTS)))
    say("")
    SITES <= 6 || error("the dense reference is 4^N x 4^N: N <= 6 (N = 7 is 4.3 GB)")
    set_symmetry!(:none)

    check_mapping(SITES) || error("check A failed: the dense<->MPS mapping is wrong, so every " *
                                  "number below it would be measured against a mirrored state")
    check_dense_vs_analytic(SITES, GAMMA) ||
        error("check B failed: the dense Liouvillian disagrees with the closed form")

    for nm in MODELS
        check_arms(nm, SITES, GAMMA)
    end
    say("READING IT:")
    say("  ratio ~ 4 per halving of dt -> second order, and THE OPERATOR IS RIGHT.")
    say("  ratio ~ 1 (error FLAT in dt) -> a wrong operator; rank and Krylov depth cannot fix it.")
    say("  The cap cannot bind at N = 6, so these are INTEGRATOR errors with no truncation in them.")
    say(@sprintf("\nwrote %s", joinpath(OUT, "open_exact_gate.txt")))
end

try
    main()
finally
    close(LOG)
end
