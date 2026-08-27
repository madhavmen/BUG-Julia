# IS CBE-BUG'S L=18 ERROR A RANK BUDGET, OR THE DIRECTIONS IT SPENDS THE BUDGET ON?
#
#   julia -t 2 --project=. benchmarks/l18_rank_scan.jl
#   CAPS=64,128 DEXS=8 TMAX=2 julia -t 2 --project=. benchmarks/l18_rank_scan.jl
#
# ── THE OBSERVATION THIS EXISTS TO EXPLAIN ───────────────────────────────────────────────────
#
# `xx`, L=18, dt=0.05, cap=64, against the EXACT free-fermion solution (`l18_campaign.jl`):
#
#     t      tdvp2 err / chi      cbe_bug err / chi
#     0.25   3.33e-09 /  8        2.68e-08 / 22
#     1.25   2.86e-07 / 21        2.33e-06 / 39
#     2.75   4.71e-07 / 56        3.99e-03 / 64  <- cap
#     5.00   1.66e-07 / 64        5.06e-02 / 64
#
# ⛔ KRYLOV DEPTH IS ALREADY EXCLUDED. `l18_krylov_depth.jl` swept `m` from 3 to 30 and the final
# error was **5.0554e-02 for every one of them**, identical to five significant figures, with
# `krylov_dims` saturating at ~15028 from m=10 on. It is not a shallow basis.
#
# ⛔ AND "RANK-LIMITED" DOES NOT EXPLAIN IT EITHER, because `tdvp2` sits at the SAME chi = 64 and
# holds 1.66e-07. Sixty-four directions are demonstrably enough; CBE-BUG is keeping the wrong ones.
# The tell is in the chi column long BEFORE the cap binds: at t = 0.25 it already carries 22
# against 8, and at t = 1.25, 39 against 21 -- roughly DOUBLE the rank for an 8x WORSE answer.
#
# ── THE TWO HYPOTHESES, AND THE ONE SCAN THAT SEPARATES THEM ─────────────────────────────────
#
#   (a) BUDGET.     The expansion is right, the cap is simply too small for the directions it
#                   legitimately needs. -> raising `maxdim` recovers TDVP's accuracy.
#   (b) SELECTION.  The expansion ADMITS SPURIOUS DIRECTIONS, inflating chi with noise; once the
#                   cap binds, truncation chooses among mostly-noise directions and discards good
#                   ones. -> raising `maxdim` does NOT converge to TDVP, but LOWERING `dex` does.
#
# ⚠ (b) IS THE SAME SIGNATURE THE `dover` STUDY ALREADY FOUND -- an under-sampled probe admitted
# chi = 28 against an exact 26. That was harmless when rank was free. This is what it costs when
# the cap binds.
#
# So the scan is two-dimensional on purpose: `maxdim` tests (a), `dex` tests (b), and the pair
# tells them apart. A one-dimensional `maxdim` scan would confirm (a) OR leave (b) unfalsified,
# never distinguish them.
#
# ⚠ THE ERROR IS REPORTED AT TWO TIMES, NOT ONE. There is a flat ~8x deficit present from the
# FIRST sample, when no cap is near binding, and a runaway that starts only when it does. Quoting
# the endpoint alone merges an integrator error with a truncation failure and would send the fix
# to the wrong place.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const OUT     = joinpath(@__DIR__, "results")
const SITES   = parse(Int,     get(ENV, "SITES", "18"))
const DT      = parse(Float64, get(ENV, "DT",    "0.05"))
const TMAX    = parse(Float64, get(ENV, "TMAX",  "5.0"))
const TMID    = parse(Float64, get(ENV, "TMID",  "1.25"))   # before any cap binds
const MAXITER = parse(Int,     get(ENV, "MAXITER", "8"))
const CAPS    = [parse(Int, s) for s in split(get(ENV, "CAPS", "64,128,256"), ",")]
const DEXS    = [parse(Int, s) for s in split(get(ENV, "DEXS", "8,4,2"), ",")]
# ⛔ `dover` IS A SEPARATE AXIS FROM `dex`, AND SCANNING ONLY `dex` CANNOT SEPARATE THEM.
# `npre = dex + dover` (`cbe_core.jl:587`) is the PROBE WIDTH while `dex` is the ADMITTED budget, so
# moving `dex` alone moves both at once. The distinction is the whole hypothesis here: an
# UNDER-SAMPLED PROBE does not merely lose accuracy, it ADMITS SPURIOUS DIRECTIONS -- measured on
# Heisenberg at L=10/12, `dover` 0->4 was worth SEVEN ORDERS and dropped chi from 28 to the exact 26
# at no cost in seconds. If `cbe_bug` carries ~2x tdvp2's rank at L=18, an under-sampled probe at
# the larger L is the first thing to rule out, and only a `dover` scan at FIXED `dex` rules it out.
const DOVERS  = [parse(Int, s) for s in split(get(ENV, "DOVERS", "4"), ",")]

exact_sz_at(L, t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:2:L)

function main()
    mkpath(OUT)
    csv = open(joinpath(OUT, "l18_rank_scan.csv"), "w")
    println(csv, "arm,cap,dex,dover,L,dt,t_mid,err_mid,chi_mid,t_end,err_end,chi_end,krylov,secs")
    flush(csv)

    S = setup("xx"; sites = SITES, dt = DT, cap = maximum(CAPS))
    L = S.L
    nsteps = max(1, round(Int, TMAX / DT))
    kmid   = max(1, round(Int, TMID / DT))

    @printf("xx  L=%d  dt=%.3f  tmax=%.1f (%d steps)  mid-probe at t=%.2f\n",
            L, DT, TMAX, nsteps, kmid * DT)
    @printf("caps = %s   dex = %s   maxiter = %d\n", string(CAPS), string(DEXS), MAXITER)
    @printf("julia threads = %d, BLAS threads = %d\n\n", Threads.nthreads(), BLAS.get_num_threads())
    flush(stdout)

    function run(step!)
        p = copy(S.psi0); kry = 0; wall = 0.0
        emid = NaN; cmid = 0
        for k in 1:nsteps
            t0 = time_ns()
            info = step!(p, S.tau)
            wall += (time_ns() - t0) / 1e9
            kry += info.krylov_dims
            S.post!(p)
            if k == kmid
                emid = maximum(abs.(S.profile(p)[1:L] .- exact_sz_at(L, k * DT)))
                cmid = maximum(bond_dims(p))
            end
        end
        return (emid = emid, cmid = cmid,
                eend = maximum(abs.(S.profile(p)[1:L] .- exact_sz_at(L, nsteps * DT))),
                cend = maximum(bond_dims(p)), kry = kry, wall = wall)
    end

    mk_t2(cap) = (p, tau) -> tdvp2_step!(p, S.W, tau; maxdim = cap, trunc_thresh = 1e-10,
                                         maxiter = MAXITER, hermitian = true)
    mk_bug(cap, dex, dover) = (p, tau) -> cbe_bug_step!(p, S.W, tau; exact = false, dex = dex,
                                                 dover = dover, comp_ratio = 1.0,
                                                 krylov_basis = 3, krylov_tol = 0.0,
                                                 parallel = true, maxdim = cap,
                                                 trunc_thresh = 1e-10, maxiter = MAXITER,
                                                 hermitian = true)
    for f in (mk_t2(CAPS[1]), mk_bug(CAPS[1], DEXS[1], DOVERS[1]))
        try; f(copy(S.psi0), S.tau); catch; end
    end

    @printf("   %-22s %5s %4s %6s %5s | %11s %5s | %11s %5s | %9s %8s\n",
            "arm", "cap", "dex", "dover", "Dpre", "err(mid)", "chi", "err(end)", "chi",
            "krylov", "secs")
    for cap in CAPS
        r = run(mk_t2(cap))
        @printf("   %-22s %5d %4s %6s %5s | %11.4e %5d | %11.4e %5d | %9d %8.1f\n",
                "tdvp2", cap, "-", "-", "-", r.emid, r.cmid, r.eend, r.cend, r.kry, r.wall)
        @printf(csv, "tdvp2,%d,0,0,%d,%g,%g,%.6e,%d,%g,%.6e,%d,%d,%.3f\n",
                cap, L, DT, kmid * DT, r.emid, r.cmid, nsteps * DT, r.eend, r.cend, r.kry, r.wall)
        flush(csv); flush(stdout)
        for dex in DEXS, dover in DOVERS
            rb = run(mk_bug(cap, dex, dover))
            @printf("   %-22s %5d %4d %6d %5d | %11.4e %5d | %11.4e %5d | %9d %8.1f\n",
                    "cbe_bug", cap, dex, dover, dex + dover,
                    rb.emid, rb.cmid, rb.eend, rb.cend, rb.kry, rb.wall)
            @printf(csv, "cbe_bug,%d,%d,%d,%d,%g,%g,%.6e,%d,%g,%.6e,%d,%d,%.3f\n",
                    cap, dex, dover, L, DT, kmid * DT, rb.emid, rb.cmid, nsteps * DT, rb.eend,
                    rb.cend, rb.kry, rb.wall)
            flush(csv); flush(stdout)
        end
        println()
    end
    close(csv)
    println("wrote results/l18_rank_scan.csv")
    println("READING IT:")
    println("  err(end) FALLS toward tdvp2 as `cap` rises  -> (a) BUDGET: the cap was too small.")
    println("  err(end) FLAT in `cap` but falls with lower `dex` -> (b) SELECTION: spurious")
    println("     directions; the expansion admits noise and truncation then keeps it.")
    println("  err FALLS with HIGHER `dover` at FIXED `dex`   -> (c) UNDER-SAMPLED PROBE. This is a")
    println("     DEFAULT to change, not a structural defect: `dover` costs nothing in seconds and")
    println("     an under-sampled probe admits spurious directions, so chi should FALL with it.")
    println("  chi(mid) vs tdvp2's chi(mid) is the direct read on (b)/(c): carrying ~2x the rank")
    println("     BEFORE any cap binds is the signature, and the fix must bring that ratio down.")
    println("  err(mid) is measured BEFORE any cap binds, so a deficit there that survives every")
    println("     knob is the INTEGRATOR and is fixed by none of them.")
end

main()
