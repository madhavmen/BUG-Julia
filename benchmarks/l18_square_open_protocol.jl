# `square_open` AT L = 18, FOLLOWING THE PROTOCOL OF arXiv:2410.18468 -- AND SAYING WHERE IT CANNOT.
#
#   julia -t 2 --project=. benchmarks/l18_square_open_protocol.jl
#   TMAX=2 GAMMAS=0.1 SITES=8 julia -t 2 --project=. benchmarks/l18_square_open_protocol.jl
#
# ⛔ READ THIS BEFORE QUOTING ANY NUMBER FROM HERE AGAINST THE PAPER. arXiv:2410.18468 (Lin Zhang,
# "Operator entanglement in SU(2)-symmetric dissipative quantum many-body dynamics") is a protocol
# we can adopt in PART and cannot adopt in part, and the parts are not interchangeable.
#
# ── ADOPTED FROM THE PAPER ───────────────────────────────────────────────────────────────────
#
#   * THE OBSERVABLE AND ITS NORMALISATION. Eq. (3): `S_op = -sum_a lambda_a^2 log_2 lambda_a^2`
#     over the NORMALISED Schmidt values of `|rho>>` at a cut. ⛔ LOG BASE 2, which is why
#     `operator_entanglement` grew a `base` keyword -- it had been returning nats, and quoting nats
#     against the paper's figures understates `S_op` by `ln 2` AND rescales the fitted prefactor by
#     the same factor while leaving the logarithmic shape intact, so the plot still looks right.
#   * THE LATE-TIME LAW to test against: `S_op(t -> inf) = eta log_2(tJ) + S_0`, i.e. the
#     signature is LOGARITHMIC growth, and the thing to extract is the slope in `log_2(tJ)`.
#   * THE DISSIPATION LADDER `gamma/J in {0.05, 0.10, 0.15, 0.20, 0.25}` -- the paper's own set.
#     It doubles as the ladder the ZENO check needs (below), so one sweep serves both targets.
#   * ONE OF THE PAPER'S THREE INITIAL STATES. It runs singlet pairs, NEEL, and triplet pairs;
#     ours is Neel, so this is a genuine point of contact rather than a coincidence.
#
# ── NOT ADOPTED, AND THESE ARE NOT SMALL ─────────────────────────────────────────────────────
#
#   ⛔ THE DISSIPATOR IS A DIFFERENT ONE. The paper's jump operator is the EXCHANGE OPERATOR ON A
#      BOND, `L_i = P_{i,i+1} = 2 S_i . S_{i+1} + 1/2` -- two-site and SU(2)-symmetric. Ours is
#      single-site DEPHASING, `L_j = S^z_j`, which is U(1). The paper's entire point is dissipation
#      BEYOND dephasing, and it treats dephasing as the contrasting case. So this run is the
#      contrast, not a reproduction, and `eta` here has no reason to match `eta` there.
#   ⛔ THE GEOMETRY IS DIFFERENT. The paper is an INFINITE 1D chain (iMPS, two-site unit cell,
#      translation invariant). Ours is an 18-site 2D open patch (6x3). A late-time logarithmic law
#      is an asymptotic statement about the infinite chain; an 18-site cluster saturates instead.
#   ⛔ THE CONVERGENCE SCALE IS DIFFERENT BY THREE ORDERS. Paper: `chi = 4000-50000`, `tJ` to
#      250-300, 4th-order Trotter of `exp(L dt)` by iTEBD with reorthonormalisation. Ours: `chi`
#      <= a few hundred and `tJ` of order 10, by TDVP/BUG. We can therefore test the SHAPE of
#      `S_op(t)` over a window, and must not quote a fitted `eta` as the asymptotic one.
#
# ── THE ACTUAL TARGET: THE PAPER IS A GUIDE, AND THE SIGNATURE IS A CHANGE OF GROWTH LAW ─────
#
# ⛔ REPRODUCTION IS NOT THE GOAL AND NEVER WAS AVAILABLE. What an 18-site cluster CAN decide is
# whether the qualitative ENTANGLEMENT TRANSITION the paper reports is present: operator
# entanglement grows fast (ballistically, set by the Hamiltonian's light cone) under unitary
# dynamics, and once dissipation is on that growth CROSSES OVER to slow -- logarithmic -- growth
# and to a SUPPRESSED plateau. Three consequences, all measurable here, and they must agree or the
# result is not believed:
#
#   1. `S_op(T)` FALLS monotonically with `gamma`.
#   2. On the late window the `log_2 t` fit BEATS the linear fit, and the margin GROWS with
#      `gamma`, while the `gamma = 0` control stays linear. ⛔ The log fit alone proves nothing --
#      over a short window a straight line fits a logarithm well. Only the comparison separates
#      them, which is why `lin_slope` exists next to `log_slope`.
#   3. `chi(T)` saturates, and saturates LOWER for larger `gamma`. This is the INDEPENDENT
#      confirmation: rank is set by the operator entanglement, and it comes out of a different code
#      path (`bond_dims`) than `S_op` does (`svd` + entropy). If (1) holds and (3) does not, one of
#      the two is wrong.
#
# ⇒ WHAT THIS FILE CAN HONESTLY CLAIM: the transition is present or it is not; the three
#   integrators agree on it; and the ZENO trend below is present. Nothing asymptotic, and no
#   fitted `eta` quoted as the paper's.
#
# ── THE SECOND TARGET, WHICH IS THIS MODEL'S OWN ─────────────────────────────────────────────
#
# arXiv:2408.10304 -- ZENO SLOWING: the magnetisation relaxes MORE SLOWLY as the dephasing rate
# rises. ⛔ AND THE OBSERVABLE HAD TO BE CHOSEN AROUND A CONSERVATION LAW: for pure dephasing
# `L_j = S^z_j`, the dissipator contributes EXACTLY ZERO to `d<S^z_k>/dt` (`S^z_k` commutes with
# every `S^z_j`, so the jump and anticommutator terms cancel identically). The bath does not damp
# the magnetisation directly -- it only destroys the coherences the Hamiltonian uses to move it.
# So "dephasing makes M_stag decay" would be measuring the HAMILTONIAN. The signature is the
# derivative of that: LARGER gamma -> SLOWER decay.
#
# ── WHY THE SWEEP IS ONE ARM AND THE COMPARISON IS ONE GAMMA ─────────────────────────────────
#
# A 3-arm x 5-gamma grid at L=18, d=4 is ~20 hours on this machine and answers one question twice.
# Split instead:
#   PART A  all three arms at gamma = 0.10  -> the integrator comparison, and the evidence that
#           the Part B curves do not depend on which integrator drew them.
#   PART B  the cheapest arm across all five gamma -> the two TRENDS, which need the ladder.
# ⚠ Part B is only meaningful BECAUSE Part A passes. If the arms disagree in Part A, Part B is a
# statement about `cbe_bug` and not about the physics, and must not be plotted as the latter.

using LinearAlgebra, Printf, Statistics
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))

const OUT     = joinpath(@__DIR__, "results")
const SITES   = parse(Int,     get(ENV, "SITES", "18"))
const DT      = parse(Float64, get(ENV, "DT",    "0.10"))
const TMAX    = parse(Float64, get(ENV, "TMAX",  "10.0"))
const SNAP    = parse(Float64, get(ENV, "SNAP",  "0.20"))
const CAP     = parse(Int,     get(ENV, "CAP",   "64"))
const MAXITER = parse(Int,     get(ENV, "MAXITER", "8"))
# ⛔ `gamma = 0` IS PREPENDED TO THE PAPER'S LADDER AND IT IS THE MOST IMPORTANT ENTRY. The thing
# being validated is a CHANGE OF GROWTH LAW -- fast/ballistic operator-entanglement growth under
# unitary dynamics, crossing over to slow (logarithmic) growth once dissipation is on. Without the
# `gamma = 0` baseline there is no "before" and every dissipative curve looks like whatever it
# looks like. The paper does not need this row because it can reach `tJ = 250` and read the
# asymptotic slope directly; on 18 sites the CONTRAST between rows is the only thing that carries
# signal, so the control row is mandatory rather than optional.
#
# ⚠ `gamma = 0` STILL TAKES THE NON-HERMITIAN PATH. The fused generator at `gamma = 0` is
# `-i(H (x) I - I (x) H^T)`, which is ANTI-Hermitian, not Hermitian -- so `hermitian = false`
# remains correct and the row is not secretly running a different solver from the others.
const GAMMAS  = [parse(Float64, s) for s in
                 split(get(ENV, "GAMMAS", "0.0,0.05,0.10,0.15,0.20,0.25"), ",")]
const GREF    = parse(Float64, get(ENV, "GREF", "0.10"))    # the gamma Part A runs at

_stag(v) = sum((isodd(j) ? 1.0 : -1.0) * v[j] for j in eachindex(v)) / length(v)

function arm_closures(S, cap)
    return [
        ("tdvp2", (p, tau) -> tdvp2_step!(p, S.W, tau; maxdim = cap, trunc_thresh = 1e-10,
                                          maxiter = MAXITER, hermitian = false)),
        ("tdvp_cbe1s", (p, tau) -> tdvp_cbe1s_step!(p, S.W, tau; maxdim = cap,
                                                    trunc_thresh = 1e-10, maxiter = MAXITER,
                                                    hermitian = false)),
        ("cbe_bug", (p, tau) -> cbe_bug_step!(p, S.W, tau; exact = false, dex = 8, dover = 4,
                                              comp_ratio = 1.0, krylov_basis = 3,
                                              krylov_tol = 0.0, parallel = true,
                                              maxdim = cap, trunc_thresh = 1e-10,
                                              maxiter = MAXITER, hermitian = false)),
    ]
end

"""
Run one (gamma, arm) trajectory, streaming a row per sample. Returns the sampled `(t, S_op)` pairs
so the caller can fit the logarithmic slope.
"""
function trajectory!(csv, S, arm, step!, gamma, nsteps, every)
    L = S.L
    psi = copy(S.psi0)
    kry = 0; wall = 0.0
    ts = Float64[]; sops = Float64[]; mst = Float64[]
    function emit(k)
        t = k * DT
        szv = S.profile(psi)
        sop, nst = operator_entanglement(psi; base = 2)      # arXiv:2410.18468 Eq. (3)
        tr = real(trace_rho(identity_superket(L), psi, L; at = 0))
        @printf(csv, "%s,%g,%d,%d,%g,%g,%d,%.10g,%.8g,%d,%d,%d,%.4f,%.10g\n",
                arm, gamma, L, CAP, DT, t, k, _stag(szv), sop, nst,
                maximum(bond_dims(psi)), kry, wall, tr)
        flush(csv)
        push!(ts, t); push!(sops, sop); push!(mst, _stag(szv))
        return nothing
    end
    emit(0)
    for k in 1:nsteps
        t0 = time_ns()
        info = step!(psi, S.tau)
        wall += (time_ns() - t0) / 1e9
        kry += info.krylov_dims
        S.post!(psi)
        (k % every == 0 || k == nsteps) && emit(k)
    end
    return (ts = ts, sops = sops, mstag = mst, kry = kry, wall = wall,
            chi = maximum(bond_dims(psi)))
end

"""
Least-squares slope of `S_op` against `log_2(tJ)` over the LATE HALF of the window.

⚠ THE SLOPE IS REPORTED WITH ITS RESIDUAL AND OVER TWO SUB-WINDOWS ON PURPOSE. A single slope
always exists; a LOGARITHMIC law is the claim that it stops changing. Comparing the last-half
slope to the last-quarter slope is the cheapest test of that, and if they differ the window is
too short to call the growth logarithmic -- which is the expected answer on 18 sites.
"""
function log_slope(ts, sops; frac = 0.5)
    idx = [i for i in eachindex(ts) if ts[i] > 0 && ts[i] >= frac * ts[end]]
    length(idx) >= 3 || return (NaN, NaN)
    x = [log2(ts[i]) for i in idx]
    y = [sops[i] for i in idx]
    xm, ym = mean(x), mean(y)
    sxx = sum((x .- xm) .^ 2)
    sxx > 0 || return (NaN, NaN)
    eta = sum((x .- xm) .* (y .- ym)) / sxx
    s0 = ym - eta * xm
    res = maximum(abs.(y .- (eta .* x .+ s0)))
    return eta, res
end

"""
Least-squares slope of `S_op` against `t` ITSELF over the same late window, with its residual.

⛔ THIS IS THE OTHER HALF OF THE TEST AND WITHOUT IT THE LOG FIT PROVES NOTHING. A logarithm and a
straight line are both smooth increasing functions, and over a short window either will fit with a
small residual -- so a good `log_2 t` fit is not evidence of logarithmic growth. The claim is
COMPARATIVE: on the late window the log fit must beat the linear fit, and that gap must OPEN UP as
`gamma` rises while the `gamma = 0` control stays linear. Reporting only `eta` would let a purely
ballistic curve be written up as the paper's law.
"""
function lin_slope(ts, sops; frac = 0.5)
    idx = [i for i in eachindex(ts) if ts[i] >= frac * ts[end]]
    length(idx) >= 3 || return (NaN, NaN)
    x = [ts[i] for i in idx]
    y = [sops[i] for i in idx]
    xm, ym = mean(x), mean(y)
    sxx = sum((x .- xm) .^ 2)
    sxx > 0 || return (NaN, NaN)
    a = sum((x .- xm) .* (y .- ym)) / sxx
    b = ym - a * xm
    res = maximum(abs.(y .- (a .* x .+ b)))
    return a, res
end

function main()
    mkpath(OUT)
    csv = open(joinpath(OUT, "l18_square_open_protocol.csv"), "w")
    println(csv, "arm,gamma,L,cap,dt,t,step,mstag,sop_log2,nstates,maxbond,krylov,secs,trace")
    flush(csv)

    nsteps = max(1, round(Int, TMAX / DT))
    every  = max(1, round(Int, SNAP / DT))
    @printf("square_open, L=%d, dt=%.3f, tmax=%.1f (%d steps), cap=%d, maxiter=%d\n",
            SITES, DT, TMAX, nsteps, CAP, MAXITER)
    @printf("julia threads = %d, BLAS threads = %d\n", Threads.nthreads(), BLAS.get_num_threads())
    println("gamma ladder = ", GAMMAS, "   (arXiv:2410.18468's own set)")
    println()
    flush(stdout)

    # ── PART A: do the three integrators agree on this model at all? ────────────────────────
    println("== PART A: three arms at gamma = $GREF ==")
    S = setup("square_open"; sites = SITES, dt = DT, gamma = GREF, cap = CAP)
    @printf("   %s\n   %d sites, d = %d, MPO virtual dim %d\n",
            S.label, S.L, S.d, maximum(mpo_virtual_dims(S.W)))
    flush(stdout)
    # ⛔ WARMUP BEFORE ANY TIMED ARM. `@compile_workload` covers `cbe_bug_step!` only, so without
    # this the first arm in the list absorbs both TDVP arms' specialisation and the seconds column
    # is fiction -- measured 61.6 s against a true 2.7 s on an L=8 smoke run.
    for (arm, step!) in arm_closures(S, CAP)
        try
            step!(copy(S.psi0), S.tau)
        catch e
            @printf("   warmup: %s threw -- %s\n", arm, sprint(showerror, e)[1:min(end, 160)])
        end
    end

    partA = Dict{String, Any}()
    for (arm, step!) in arm_closures(S, CAP)
        r = trajectory!(csv, S, arm, step!, GREF, nsteps, every)
        eta, res = log_slope(r.ts, r.sops)
        partA[arm] = r
        @printf("   %-12s chi %3d  krylov %7d  %8.1f s   S_op(end) = %.4f  eta = %.3f (res %.1e)\n",
                arm, r.chi, r.kry, r.wall, r.sops[end], eta, res)
        flush(stdout)
    end
    if haskey(partA, "tdvp2")
        for arm in ("tdvp_cbe1s", "cbe_bug")
            haskey(partA, arm) || continue
            ds = maximum(abs.(partA[arm].sops .- partA["tdvp2"].sops))
            dm = maximum(abs.(partA[arm].mstag .- partA["tdvp2"].mstag))
            @printf("   %-12s vs tdvp2:  max|dS_op| = %.3e   max|dM_stag| = %.3e\n", arm, ds, dm)
        end
        println("   ⚠ Part B below is a statement about the PHYSICS only if these are small. If " *
                "they are not,\n     it is a statement about cbe_bug.")
    end
    println()

    # ── PART B: the two trends, which need the whole ladder ─────────────────────────────────
    println("== PART B: cbe_bug across the gamma ladder ==")
    # ⚠ `M_stag(T)/M_stag(0)` IS THE ZENO NUMBER, NOT `M_stag(T)` ALONE. The ratio is what
    # "relaxes more slowly" means; the bare endpoint also moves with anything that changes the
    # starting amplitude, and would read as a trend when the starting point was what shifted.
    #
    # THE THREE COLUMNS THAT CARRY THE ENTANGLEMENT-TRANSITION CLAIM, and they must agree:
    #   `S_op(T)`      must FALL with gamma -- dissipation suppresses operator entanglement.
    #   `res_lin/res_log` > 1 means the late window is better described by `log_2 t` than by `t`.
    #                    The claim is that this RATIO GROWS with gamma while the gamma = 0 control
    #                    stays at or below 1.
    #   `chi(T)`       must SATURATE, and saturate LOWER for larger gamma. ⛔ THIS IS THE
    #                    INDEPENDENT CONFIRMATION: bond dimension is set by the operator
    #                    entanglement, so if `S_op` genuinely falls with gamma and `chi` does not,
    #                    one of the two numbers is wrong. Two panels of the same physics computed
    #                    by different code paths is worth more than either alone.
    # ⚠ `chi(T) == cap` MEANS THE ROW IS RANK-LIMITED and its `S_op` is a LOWER BOUND, not a
    #   measurement -- a truncated spectrum cannot report the entropy it was truncated out of.
    println("   gamma   S_op(T)   chi(T)   eta_log   res_log    a_lin    res_lin   " *
            "res_lin/res_log   Mstag(T)/Mstag(0)    secs")
    for g in GAMMAS
        Sg = setup("square_open"; sites = SITES, dt = DT, gamma = g, cap = CAP)
        arm, step! = arm_closures(Sg, CAP)[3]          # cbe_bug
        r = trajectory!(csv, Sg, arm, step!, g, nsteps, every)
        eta, rlog = log_slope(r.ts, r.sops; frac = 0.5)
        a, rlin = lin_slope(r.ts, r.sops; frac = 0.5)
        ratio = abs(r.mstag[1]) > 1e-12 ? r.mstag[end] / r.mstag[1] : NaN
        @printf("   %5.2f  %8.4f   %4d%s  %8.3f  %9.2e  %8.4f  %9.2e   %14.2f   %17.5f  %6.1f\n",
                g, r.sops[end], r.chi, r.chi >= CAP ? "*" : " ", eta, rlog, a, rlin,
                rlin / max(rlog, 1e-30), ratio, r.wall)
        flush(stdout)
    end
    println("   (* = chi hit the cap: that row's S_op is a LOWER BOUND, not a measurement)")
    close(csv)
    println("\nwrote results/l18_square_open_protocol.csv")
end

main()
