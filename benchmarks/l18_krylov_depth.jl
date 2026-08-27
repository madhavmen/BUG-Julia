# HOW DEEP MUST THE CBE-BUG KRYLOV BASIS BE AT L = 18 TO MATCH TDVP?
#
#   julia -t 2 --project=. benchmarks/l18_krylov_depth.jl
#   MODEL=xx SITES=18 TMAX=5 MS=3,8,16,30 julia -t 2 --project=. benchmarks/l18_krylov_depth.jl
#
# ⛔ THE SHIPPED `m = 3` WAS CALIBRATED ON A DIFFERENT PROBLEM AND DOES NOT TRANSFER. It was chosen
# from `docs/PLAN-tolerance-control.md` at **L = 12 over 20 steps of dt = 0.05** -- one revival's
# worth of evolution on a short chain. MEASURED at L = 18 over **100** steps against the EXACT
# free-fermion solution:
#
#     tdvp2        chi 64   krylov 52692   752.8 s   err 1.657e-07
#     tdvp_cbe1s   chi 64   krylov 50995   833.6 s   err 1.657e-07
#     cbe_bug      chi 64   krylov  6196   262.9 s   err 5.055e-02      <- FIVE ORDERS OFF
#
# ⚠ AND THIS IS THE FAILURE MODE THAT DOES NOT LOOK LIKE ONE. A short Krylov basis survives more
# rank, a tighter cutoff and a longer budget -- every knob one would normally reach for -- so it
# never presents as a basis problem. It presents as "the integrator is just less accurate". The
# only way to see it is to vary `m` itself and watch the error move, which is what this file does.
#
# ⛔ ALSO NOTE `krylov_dims` FALLS OUT OF THE COMPARISON HERE. At `m = 3` CBE-BUG used 6196
# applications against TDVP's 52692 -- an 8.5x "saving" that is really an 8.5x SHORTFALL, since the
# answer it bought was five orders worse. An operator count is only a cost when the two arms agree
# on the answer; quoting it otherwise measures how much work was SKIPPED.
#
# ── WHAT COUNTS AS A PASS ────────────────────────────────────────────────────────────────────
#
# `xx` is the one model in the study with a closed form, so `err` here is a TRUE ERROR against
# free fermions -- not a deviation from another arm. The target is `err` within a small factor of
# TDVP's 1.657e-07; the useful output is the SMALLEST `m` that gets there and what it costs.
#
# ⚠ IF NO `m` REACHES IT, THE CAUSE IS NOT DEPTH. At cap 64 with chi pinned AT 64 for every arm,
# a floor common to all three would be the rank, not the Krylov basis -- so the run also reports
# `chi`, and an `m` sweep that plateaus well above TDVP while `chi` sits at the cap means the next
# variable to move is `maxdim`, not `m`.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const OUT     = joinpath(@__DIR__, "results")
const MODEL   = get(ENV, "MODEL", "xx")
const SITES   = parse(Int,     get(ENV, "SITES", "18"))
const DT      = parse(Float64, get(ENV, "DT",    "0.05"))
const TMAX    = parse(Float64, get(ENV, "TMAX",  "5.0"))
const CAP     = parse(Int,     get(ENV, "CAP",   "64"))
const MAXITER = parse(Int,     get(ENV, "MAXITER", "8"))
const MS      = [parse(Int, s) for s in split(get(ENV, "MS", "3,6,10,16,24,30"), ",")]

exact_sz_at(L, t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:2:L)

function main()
    mkpath(OUT)
    csv = open(joinpath(OUT, "l18_krylov_depth_$(MODEL).csv"), "w")
    println(csv, "model,arm,m,L,cap,dt,tmax,err,maxbond,krylov,secs")
    flush(csv)

    S = setup(MODEL; sites = SITES, dt = DT, cap = CAP)
    L = S.L
    nsteps = max(1, round(Int, TMAX / DT))
    herm = (S.d == 2)
    MODEL == "xx" || error("l18_krylov_depth needs a model with a closed form; only `xx` has one")

    @printf("%s  L=%d  dt=%.3f  tmax=%.1f (%d steps)  cap=%d  maxiter=%d\n",
            MODEL, L, DT, TMAX, nsteps, CAP, MAXITER)
    @printf("julia threads = %d, BLAS threads = %d\n\n", Threads.nthreads(), BLAS.get_num_threads())

    gate = maximum(abs.([real(sz_expectation(copy(S.psi0), j)) for j in 1:L] .- exact_sz_at(L, 0.0)))
    @printf("free-fermion convention gate at t=0: max|dSz| = %.3e  %s\n\n",
            gate, gate < 1e-10 ? "PASS" : "FAIL")
    gate < 1e-10 || error("the exact reference describes a different initial state")

    function run(label, step!)
        p = copy(S.psi0); kry = 0; wall = 0.0
        for _ in 1:nsteps
            t0 = time_ns()
            info = step!(p, S.tau)
            wall += (time_ns() - t0) / 1e9
            kry += info.krylov_dims
            S.post!(p)
        end
        sz = S.profile(p)[1:L]
        return (err = maximum(abs.(sz .- exact_sz_at(L, nsteps * DT))),
                chi = maximum(bond_dims(p)), kry = kry, wall = wall)
    end

    mk_t2 = (p, tau) -> tdvp2_step!(p, S.W, tau; maxdim = CAP, trunc_thresh = 1e-10,
                                    maxiter = MAXITER, hermitian = herm)
    mk_bug(m) = (p, tau) -> cbe_bug_step!(p, S.W, tau; exact = false, dex = 8, dover = 4,
                                          comp_ratio = 1.0, krylov_basis = m, krylov_tol = 0.0,
                                          parallel = true, maxdim = CAP, trunc_thresh = 1e-10,
                                          maxiter = MAXITER, hermitian = herm)
    # Warmup: every arm, before anything is timed.
    for f in (mk_t2, mk_bug(first(MS)), mk_bug(last(MS)))
        try; f(copy(S.psi0), S.tau); catch; end
    end

    println("   arm                    m        err vs EXACT   chi     krylov      secs   x TDVP err")
    r2 = run("tdvp2", mk_t2)
    @printf("   %-20s %4s   %14.4e  %4d %10d  %8.1f   %10s\n",
            "tdvp2 (reference)", "-", r2.err, r2.chi, r2.kry, r2.wall, "1.0")
    @printf(csv, "%s,tdvp2,0,%d,%d,%g,%g,%.6e,%d,%d,%.3f\n",
            MODEL, L, CAP, DT, TMAX, r2.err, r2.chi, r2.kry, r2.wall)
    flush(csv); flush(stdout)

    for m in MS
        r = run("cbe_bug", mk_bug(m))
        @printf("   %-20s %4d   %14.4e  %4d %10d  %8.1f   %10.1f\n",
                "cbe_bug", m, r.err, r.chi, r.kry, r.wall, r.err / r2.err)
        @printf(csv, "%s,cbe_bug,%d,%d,%d,%g,%g,%.6e,%d,%d,%.3f\n",
                MODEL, m, L, CAP, DT, TMAX, r.err, r.chi, r.kry, r.wall)
        flush(csv); flush(stdout)
    end
    close(csv)
    println("\nwrote results/l18_krylov_depth_$(MODEL).csv")
    println("⚠ if err PLATEAUS above TDVP while chi sits at the cap, the limit is RANK not depth.")
end

main()
