# DOES THE FUSED MPDO EVOLUTION REPRODUCE THE **EXACT** DEPHASING XX CHAIN?
#
#   julia --project=. benchmarks/xx_open_exact_gate.jl
#   SITES=6 TMAX=1 DTS=0.02,0.01 GAMMA=0.1 julia --project=. benchmarks/xx_open_exact_gate.jl
#
# ⛔ THIS IS THE GATE THAT PROMOTES `xx_open` FROM A TREND TO AN ERROR. The open XX chain under
# onsite dephasing has a CLOSED FORM: the single-particle correlation matrix obeys
#
#     dC/dt = -i[h, C] - gamma (1 - delta_ij) C_ij
#
# which closes -- the dissipator sends `c_i^dag c_j` to a multiple of itself, generating no higher
# correlator. So `<S^z_j(t)>` is an `L x L` ODE (`tests/common/free_fermion.jl`:
# `xx_dephasing_sz`) and NOTHING of size `4^L` is built. Every other open model in this study is
# validated against a published TREND; this one can be validated against a NUMBER, at the full
# L = 18, which makes it the only place a defect in the open path can be localised exactly.
#
# ── THE THREE CHECKS, IN THE ORDER THAT LOCALISES A FAULT ────────────────────────────────────
#
#   A  gamma = 0, reference vs reference. `xx_dephasing_sz` must reproduce `xx_free_fermion_sz`
#      to machine precision. NO SOLVER INVOLVED -- this tests only that the dissipative generator
#      was assembled the right way round (column-major `vec`, `kron` order, sign).
#      ⚠ It is first because a wrong reference makes every later number meaningless while still
#      looking like a smooth curve.
#
#   B  gamma = 0, MPS vs reference, FULL RANK. Isolates the fused chain's HAMILTONIAN half. A
#      failure here is the doubled-chain construction, not the bath.
#
#   C  gamma > 0, MPS vs reference, FULL RANK, dt REFINED. The whole MPDO path: layout, Liouvillian
#      MPO, zero-body shift, trace restoration. ⛔ THE dt LADDER IS THE POINT -- at full rank the
#      only remaining error is the INTEGRATOR, so the deviation must fall like `dt^2` and go to
#      zero. A deviation that is FLAT in dt is a wrong operator, and no amount of rank or depth
#      will move it. That distinction is exactly what `benchmarks/lindblad_diag.jl` had to
#      establish the hard way for the doubled chain.
#
# ⚠ FULL RANK IS NOT OPTIONAL HERE. With a cap that binds, truncation error masks the thing being
# tested -- `benchmarks/trace_preservation.jl` measured every arm pinned at state dev 1.35e-01 at
# cap 2, tdvp2 included. `L = 6` fused gives a maximum bond of `4^3 = 64`, so `cap = 256` cannot
# bind at any time.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const SITES = parse(Int,     get(ENV, "SITES", "6"))
const TMAX  = parse(Float64, get(ENV, "TMAX",  "0.5"))
const GAMMA = parse(Float64, get(ENV, "GAMMA", "0.1"))
const CAP   = parse(Int,     get(ENV, "CAP",   "256"))
const DTS   = [parse(Float64, s) for s in split(get(ENV, "DTS", "0.05,0.025,0.0125"), ",")]

# ⛔ EVERY LINE IS FLUSHED AND MIRRORED TO A FILE, AND THAT IS NOT TIDINESS. Julia BLOCK-BUFFERS a
# redirected stdout, so a long run that is killed discards everything it printed: this gate was
# first run without it, spent 31 minutes and 2085 CPU-seconds, was killed, and produced `exit 127`
# with a COMPLETELY EMPTY log. Nothing was recoverable. Partial results from a killed run are the
# normal case on a contended laptop, so they have to survive by construction.
const LOG = open(joinpath(@__DIR__, "results", "xx_open_exact_gate.txt"), "w")
say(s) = (println(LOG, s); flush(LOG); println(stdout, s); flush(stdout))

# The Neel start `_open` uses: up on ODD sites, and up is an occupied fermion.
const OCC = 1:2:SITES

function mps_sz(nm, arm, dt, gamma)
    S = setup(nm; sites = SITES, dt = dt, gamma = gamma, cap = CAP)
    L = S.L
    step! = arm == "tdvp2" ?
        ((p, tau) -> tdvp2_step!(p, S.W, tau; maxdim = CAP, trunc_thresh = 1e-14,
                                 maxiter = 30, hermitian = false)) :
        arm == "tdvp_cbe1s" ?
        ((p, tau) -> tdvp_cbe1s_step!(p, S.W, tau; maxdim = CAP, trunc_thresh = 1e-14,
                                      maxiter = 30, hermitian = false)) :
        ((p, tau) -> cbe_bug_step!(p, S.W, tau; exact = true, dex = 8, dover = 4,
                                   comp_ratio = 1.0, krylov_basis = 30, krylov_tol = 0.0,
                                   maxdim = CAP, trunc_thresh = 1e-14, maxiter = 30,
                                   hermitian = false))
    p = copy(S.psi0)
    n = max(1, round(Int, TMAX / dt))
    for _ in 1:n
        step!(p, S.tau)
        S.post!(p)
    end
    return S.profile(p)[1:L], maximum(bond_dims(p)),
           abs(real(trace_rho(S.imps, p, L; at = 0)) - 1.0)
end

say("xx_open exactness gate — L = $SITES fused sites, T = $TMAX, cap = $CAP (cannot bind)")
say("")

# ── A: reference against reference, no solver ────────────────────────────────────────────────
dA = maximum(abs.(xx_dephasing_sz(SITES, TMAX; J = 1.0, gamma = 0.0, occupied = OCC) .-
                  xx_free_fermion_sz(SITES, TMAX; J = 1.0, occupied = OCC)))
say(@sprintf("A  gamma=0, xx_dephasing_sz vs xx_free_fermion_sz : max|d| = %.3e   %s",
             dA, dA < 1e-12 ? "PASS" : "FAIL — the generator is assembled wrong"))
say("")

# ── B and C: the MPDO path against the closed form ───────────────────────────────────────────
for (label, g) in (("B  gamma = 0   (Hamiltonian half only)", 0.0),
                   ("C  gamma = $GAMMA  (full Lindbladian)", GAMMA))
    say(label)
    want = xx_dephasing_sz(SITES, TMAX; J = 1.0, gamma = g, occupied = OCC)
    say(@sprintf("   %-12s %8s %13s %8s %11s %9s %8s",
                 "arm", "dt", "max|dSz|", "ratio", "|Tr-1|", "chi", "secs"))
    for arm in ("tdvp2", "tdvp_cbe1s", "cbe_bug")
        prev = NaN
        for dt in DTS
            t0 = time_ns()
            res = try
                mps_sz("xx_open", arm, dt, g)
            catch e
                say(@sprintf("   %-12s %8.4f  THREW: %s", arm, dt,
                             sprint(showerror, e)[1:min(end, 90)]))
                break
            end
            got, chi, dtr = res
            el = (time_ns() - t0) / 1e9
            d = maximum(abs.(got .- want))
            say(@sprintf("   %-12s %8.4f %13.4e %8s %11.2e %9d %8.1f", arm, dt, d,
                         isnan(prev) ? "-" : @sprintf("%.2f", prev / d), dtr, chi, el))
            prev = d
        end
    end
    say("")
end
say("READING IT:")
say("  `ratio` ~ 4 per halving of dt  -> second order, and the OPERATOR IS RIGHT.")
say("  `ratio` ~ 1 (deviation FLAT in dt) -> a wrong operator; rank and depth cannot fix it.")
say("  At full rank `|Tr-1|` is the GALERKIN trace loss only: exact (1e-14) for the TDVP arms,")
say("  O(dt^2) for cbe_bug -- see benchmarks/trace_preservation.jl.")
close(LOG)
