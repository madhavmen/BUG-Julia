# WHERE DOES THE TIME GO AT chi = 1024 ... 10000?
#
# The campaign so far has measured integrators at chi <= 128, where the step is
# OVERHEAD-DOMINATED (measured: chi=6 costs 1.8 s/step, chi=63 costs 1.08 s/step — the
# cost went DOWN as the rank went up). Nothing learned there transfers to chi = 1024+,
# where the step is dense `zgemm` inside `Telum.contract`. So this benchmark does not
# reuse the trajectory driver; it measures single steps on a state PARKED at the target
# rank ([[random_mps.jl]]) and reports a per-kernel breakdown.
#
# WHAT IT ANSWERS, in order of how much it changes what we do next:
#
#   1. Which Telum entry point owns the step — `contract`, `svd`, or the HPTT permute.
#      Optimisation effort goes where this says, not where intuition says.
#   2. The SECTOR-SIZE DISTRIBUTION at the working bond. This is the batched-gemm
#      question stated numerically: `contract` loops over output sectors SERIALLY
#      (contract.jl:1758) with ONE shared scratch buffer, so if a bond of chi=1024
#      splits into ~20 blocks of ~50, every gemm is far too small to occupy a
#      multithreaded BLAS and the parallelism has to come from batching ACROSS sectors.
#      If instead it splits into 3 blocks of ~340, threaded BLAS inside the gemm is
#      already the right answer and batching buys nothing.
#   3. BLAS thread scaling AT THIS RANK. The 1-6 thread confound measured at chi<=128
#      is not evidence about chi=1024: at small chi more BLAS threads is pure overhead,
#      at large chi it is the only parallelism there is.
#   4. PEAK MEMORY, which is what actually caps the reachable chi. The state is not the
#      problem — the Krylov basis is: `maxiter` two-site blocks held at once, each about
#      twice a site tensor. That product is why chi=10000 may be unreachable regardless
#      of how fast the kernels are, and it is better to learn that from a printed number
#      than from an OOM kill.
#
# ⚠ EVERY NUMBER HERE IS A COST, NEVER AN ACCURACY. The state is random
# ([[random_mps.jl]] header): the Schmidt spectrum is flat, so a truncation threshold
# behaves nothing like it does on a physical state. Runs pin `trunc_thresh = 0.0` and let
# `maxdim` fix the rank, which is also what makes the chi axis controlled.

using LinearAlgebra
using Printf
using Profile
using Random
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using Telum: TELUM_CONTRACT_THREADS

include(joinpath(@__DIR__, "random_mps.jl"))
include(joinpath(@__DIR__, "random_mpo.jl"))

# ── knobs ────────────────────────────────────────────────────────────────────

envint(k, d)   = parse(Int, get(ENV, k, string(d)))
envfloat(k, d) = parse(Float64, get(ENV, k, string(d)))
envbool(k, d)  = parse(Bool, get(ENV, k, string(d)))
envints(k, d)  = [parse(Int, s) for s in split(get(ENV, k, d), ',')]

const L        = envint("LCP_L", 30)
const CHIS     = envints("LCP_CHIS", "1024,2048,4096")
const NSTEPS   = envint("LCP_NSTEPS", 4)
const DT       = envfloat("LCP_DT", 0.01)
const MAXITER  = envint("LCP_MAXITER", 8)
const BLAS_T   = envints("LCP_BLAS", string(max(1, Sys.CPU_THREADS ÷ 2)))
# Tasks `Telum.contract` may spawn across output sectors. A list, so one job can A/B the
# forked threading against the serial path it must reproduce bit for bit.
# ⚠ Capped by `Threads.nthreads()`, so `julia -t 2` makes every entry above 2 identical —
# the two parallelism axes (sectors here, BLAS below) share one thread pool.
const CTHREADS = envints("LCP_CTHREADS", "0")   # 0 = leave the default (unbounded)
const ARMS     = split(get(ENV, "LCP_ARMS", "tdvp2,cbe1s,bug"), ',')
const DELTA    = envfloat("LCP_DELTA", 1.0)
const CONV_TOL = envfloat("LCP_CONV_TOL", 0.0)   # 0 = old breakdown-only behaviour
# MPO virtual bond dimensions to sweep. `0` means the physical `xxz_mpo` (D = 5). Anything
# else builds a random MPO at that D — see random_mpo.jl on why no physical builder here
# reaches D = 1024, and on why the result must never be scored for accuracy.
# ⚠ Doubling overshoots and there is no MPO analogue of a canonical trim, so the D recorded
# in the CSV is the D that came back, not the D requested.
const MPO_DS   = envints("LCP_MPO_D", "0")
# ⛔ THE CBE EXPANSION BUDGET — the knob that inverted the expected solver ordering.
# `cbe_core.jl:646` sets `budget = ceil(growth*dmax) - r`, so the DEFAULT `growth = 2.0`
# asks for `r` NEW directions on top of `r` existing ones: at chi = 1024 that is 1024 per
# side, sketched with `Dpre = ceil(1.2*budget) = 1229` columns against a space of
# `d*chi = 2048`. A 0.6x probe is a width at which no randomised method can beat a direct
# factorisation, and the result is truncated straight back to `maxdim` afterwards.
#
# Measured consequence at chi=1024: cbe1s 246 s/step against tdvp2's 121 s. Both CBE arms
# carry this and tdvp2 does not, which is exactly the wrong way round.
#
# `dex > 0` overrides the schedule with an ABSOLUTE per-side budget, and that is the whole
# point at large rank: +64 directions is a 50% growth at chi=128 (useless, the sketch is
# nearly the whole space) but 3% at chi=1024, where `Dpre = 77` against 2048 is a 0.038x
# probe and the sketch is ~16x cheaper than the expansion it replaces.
# ⚠ `dex` trades against how fast the rank can GROW, so a cost win here is only real if the
# accuracy is scored separately — this file measures cost only.
const DEXS     = envints("LCP_DEX", "0")          # 0 = the growth schedule
# ⚠ DEFAULT 1.1, NOT the library's 2.0. `growth = 1.1` is the reference's own DMRG ratchet
# (cbe_core.jl:900) and asks for ~10% more directions: at chi=1024 that is budget = 103 and
# Dpre = 124 against d*chi = 2048 — a 0.06x probe, which is the regime a randomised sketch
# is actually for. `growth = 2.0` asks for 1024 and probes at 0.6x, where it cannot win.
const GROWTH   = envfloat("LCP_GROWTH", 1.1)
const KRY_TOL  = envfloat("LCP_KRY_TOL", 1e-6)   # BUG half-sweep frame tolerance
# ⛔ BUG'S FRAME DEPTH IS A SEPARATE KNOB FROM `maxiter`, AND ITS DEFAULT IS 30.
# `maxiter` bounds the `expv` solves; `krylov_basis` bounds `_krylov_frame`, which is where
# BUG does nearly all of its operator work — one `apply_one_site` per vector, per bond, per
# half-sweep. Left at the default while the TDVP arms ran `maxiter = 8`, BUG was building
# depth-30 frames against depth-8 solves: ~1700 matvecs a step against ~460. That is the
# same class of unfairness as the `conv_tol` gap, pointing the other way, and it has to be
# matched before any ordering claim means anything.
#
# ⚠ AND THE RANDOM STATE MAKES IT WORSE THAN IT WOULD BE IN PRACTICE. `_krylov_frame` exits
# early once Saad's contribution drops below `krylov_tol`, which on a PHYSICAL state happens
# quickly because the Schmidt spectrum decays. `random_mps` has a flat-ish spectrum by
# construction, so contributions stay large and the frame runs to its cap. Cost measured
# here is therefore an UPPER bound for BUG specifically, in a way it is not for the TDVP
# arms — do not quote a BUG/TDVP ratio from this benchmark without saying so.
# ⚠ AND `krylov_basis` IS NOT THE ANALOGUE OF `maxiter` EITHER — matching them at 8 was
# still wrong, just less wrong. `maxiter` is a Lanczos depth: cost is `maxiter` matvecs.
# `krylov_basis` is how many powers of H enrich the BASIS, and `_krylov_frame` returns that
# many blocks which are then `oplus`ed into a tensor `krylov_basis` times WIDER than the
# state and factorised by `_splitU`. At chi=1024 with m=8 that is an SVD of roughly
# 2048 x 9016 per bond, 58 bonds a step. `m = 1` is the bare CBE frame (the code says so
# outright); the useful range is small.
const KRY_BASIS = envint("LCP_KRY_BASIS", 2)

# ⛔⛔ CAP THE FRAME SPLIT AT `maxdim`, OR BUG SILENTLY RUNS AT TWICE EVERYONE ELSE'S RANK.
# `cbe_bug_step!` defaults to `split_maxdim = 0` (no cap) and `split_cutoff = 1e-14` (no
# truncation), so `_splitU` keeps the whole stacked frame — up to the full local space
# `d*chi = 2048` at chi=1024. Every contraction after that runs at 2048 while the TDVP arms
# run at 1024, which is not a fair race and is not a sensible configuration either.
# This is the rank inflation already established at small chi (chi=80 against TDVP's 35, and
# chi=41 once matched); `trunc_thresh = 0.0` here means nothing else caps it.
# 0 = leave the library default, i.e. uncapped; anything else is an explicit cap, and the
# driver defaults it to `chi` per target rank.
const SPLIT_MAXDIM = envint("LCP_SPLIT_MAXDIM", -1)   # -1 = follow chi

# ⛔ `parallel` DEFAULTS TO **false**, so BUG'S HALF-SWEEP PARALLELISM HAS NEVER BEEN ON in
# any measurement here. It is one of the two structural advantages BUG is supposed to have
# over TDVP (the other being no backward evolution), and TDVP has no equivalent — so leaving
# it off measures BUG with one hand tied and still calls the result a method comparison.
# ⚠ It needs `julia -t 2` or more to do anything at all; with one thread the two tasks
# serialise and the flag is a no-op that quietly reports success.
const PARALLEL = envbool("LCP_PARALLEL", true)

"""
BLAS threads for one arm, given the allocation's `nt`.

⚠ FAIRNESS IS ABOUT TOTAL CORES, NOT THE FLAG. With `parallel = true` BUG runs two
half-sweeps at once, so giving each of them `nt` BLAS threads uses `2*nt` cores on an
`nt`-core allocation — BUG would oversubscribe while TDVP does not, and the resulting
"speedup" would be partly stolen cores and partly contention. Halving BLAS for the parallel
arms keeps every arm at the same core budget, which is the comparison we actually want.
"""
blas_for(arm, nt) = (PARALLEL && (arm == "bug" || arm == "bugmid")) ? max(1, nt ÷ 2) : nt
const DO_PROF  = envbool("LCP_PROFILE", true)
const OUTDIR   = get(ENV, "BUG_OUTDIR", joinpath(@__DIR__, "results"))

const T0 = time()
say(msg) = (@printf("[%8.1fs] %s\n", time() - T0, msg); flush(stdout))
gb(bytes) = bytes / 2^30

# ── kernel-level breakdown ───────────────────────────────────────────────────

const BUCKETS = ["BLAS gemm", "LAPACK svd/qr", "permute/HPTT", "GC/alloc",
                 "contract bookkeeping", "svd bookkeeping", "oplus/sum", "other"]

"""
Classify ONE backtrace by its INNERMOST recognised frame.

The split that matters is arithmetic vs bookkeeping, so `BLAS gemm` is tested before
`contract`: a sample inside `zgemm` called from `contract` is arithmetic and must not be
charged to the contraction machinery, while a sample in `_possible_pair_table` — sector
matching, dict lookups, interval building — is bookkeeping and is exactly the cost that
does NOT shrink when you make the gemms faster. At chi=1024 the first should dominate; if
the second does instead, the fix is restructuring Telum's sector loop, not BLAS tuning.
"""
function classify_bt(bt, lidict)
    for ip in bt                                   # innermost first
        frames = get(lidict, ip, nothing)
        frames === nothing && continue
        for fr in (frames isa Vector ? frames : [frames])
            f = string(fr.func); file = string(fr.file)
            (occursin("gemm", f) || occursin("gemv", f) || occursin("syrk", f) ||
             f == "mul!" || occursin("generic_matmatmul", f)) && return "BLAS gemm"
            (occursin("gesdd", f) || occursin("gesvd", f) || occursin("geqrf", f) ||
             occursin("orgqr", f) || occursin("ungqr", f)) && return "LAPACK svd/qr"
            (occursin("hptt", f) || occursin("permutedims", f)) && return "permute/HPTT"
            (occursin("gc", lowercase(f)) || occursin("jl_alloc", f) ||
             occursin("array_", f)) && return "GC/alloc"
            occursin("svd.jl", file) && return "svd bookkeeping"
            occursin("contract.jl", file) && return "contract bookkeeping"
            (occursin("oplus", f) || occursin("sum_tlarray", file)) && return "oplus/sum"
        end
    end
    return "other"
end

"""
Bucket the sampled profile BY SAMPLE.

⛔ The obvious loop — `for ip in data` — counts stack FRAMES, not samples, so a deep stack
votes many times and the result is dominated by whatever calls the deepest. Julia stores
backtraces in one flat vector separated by zeros, so they have to be split first.
`include_meta = false` strips the per-sample metadata blocks that would otherwise be
mistaken for instruction pointers.
"""
function profile_buckets()
    data = Profile.fetch(include_meta = false)
    lidict = Profile.getdict(data)
    buckets = Dict(b => 0 for b in BUCKETS)
    total = 0
    bt = UInt64[]
    for ip in data
        if ip == 0
            if !isempty(bt)
                buckets[classify_bt(bt, lidict)] += 1
                total += 1
                empty!(bt)
            end
        else
            push!(bt, ip)
        end
    end
    if !isempty(bt)
        buckets[classify_bt(bt, lidict)] += 1; total += 1
    end
    return buckets, total
end

# ── the shapes the gemms actually see ────────────────────────────────────────

"""
The sector-size histogram on the widest bond — item 2 of the header. `contract` does one
gemm per output sector, so this IS the gemm-size distribution, and it decides whether
parallelism belongs inside the gemm (few large blocks) or across them (many small ones).
"""
function report_sector_shapes(psi)
    L = length(psi)
    mid = L ÷ 2
    A = psi[mid]
    sizes = sort([size(r, 1) * size(r, 3) for r in A.RMTs], rev = true)  # rows x cols of each block
    dims  = sort([size(r, 1) for r in A.RMTs], rev = true)
    say(@sprintf("  sector shapes at site %d: %d blocks; left-leg dims %s%s",
                 mid, length(A.RMTs),
                 string(dims[1:min(8, end)]), length(dims) > 8 ? " ..." : ""))
    say(@sprintf("  block areas: max=%d  median=%d  min=%d  (a gemm below ~128x128 cannot fill a multithreaded BLAS)",
                 sizes[1], sizes[(length(sizes)+1) ÷ 2], sizes[end]))
    return length(A.RMTs), dims
end

# ── one arm ──────────────────────────────────────────────────────────────────

function run_arm(arm::AbstractString, psi0, mpo, chi::Int, dex::Int, nt::Int)
    psi = deepcopy(psi0)
    tau = ComplexF64(-im * DT)
    # ⛔ `CONV_TOL` MUST BE SET FOR EVERY ARM OR NONE. CBE-BUG's half-sweeps stop adaptively
    # already (`_krylov_frame` weighs Saad's contribution against `krylov_tol = 1e-6` each
    # iteration), while `tdvp2_step!` and `tdvp_cbe1s_step!` default to `conv_tol = 0.0` and
    # exit on BREAKDOWN only -- which on a generic H never fires, so they burn the full
    # `maxiter` on every solve. Leaving it at the default therefore times an adaptive method
    # against two non-adaptive ones and reads the gap as a property of the algorithm.
    step! = if arm == "tdvp2"
        () -> RSVDCBEBondUpdate.tdvp2_step!(psi, mpo, tau;
                  maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER,
                  conv_tol = CONV_TOL)
    elseif arm == "cbe1s"
        () -> RSVDCBEBondUpdate.tdvp_cbe1s_step!(psi, mpo, tau;
                  maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER, exact = false,
                  conv_tol = CONV_TOL, dex = dex, growth = GROWTH)
    elseif arm == "bug"
        () -> RSVDCBEBondUpdate.cbe_bug_step!(psi, mpo, tau;
                  maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER, exact = false,
                  root_conv_tol = CONV_TOL, krylov_tol = KRY_TOL, krylov_basis = KRY_BASIS,
                  dex = dex, growth = GROWTH, parallel = PARALLEL,
                  split_maxdim = SPLIT_MAXDIM < 0 ? chi : SPLIT_MAXDIM)
    elseif arm == "bugmid"
        () -> RSVDCBEBondUpdate.cbe_bug_midpoint_step!(psi, mpo, tau;
                  maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER, exact = false,
                  root_conv_tol = CONV_TOL, krylov_tol = KRY_TOL, krylov_basis = KRY_BASIS,
                  dex = dex, growth = GROWTH, parallel = PARALLEL,
                  split_maxdim = SPLIT_MAXDIM < 0 ? chi : SPLIT_MAXDIM)
    else
        error("unknown arm $arm")
    end

    BLAS.set_num_threads(blas_for(arm, nt))
    times = Float64[]
    allocs = Float64[]
    for k in 1:NSTEPS
        GC.gc()
        st = @timed step!()
        push!(times, st.time); push!(allocs, st.bytes)
        say(@sprintf("    %-6s step %d/%d  %8.2f s  alloc %6.2f GB  chi=%d",
                     arm, k, NSTEPS, st.time, gb(st.bytes),
                     maximum(BondUpdateBUG.bond_dims(psi))))
    end
    # step 1 carries first-call compilation; measure on the rest when we have them
    meas = length(times) > 1 ? times[2:end] : times
    return (times = times, allocs = allocs, mean = sum(meas) / length(meas),
            chi_out = maximum(BondUpdateBUG.bond_dims(psi)))
end

# ── main ─────────────────────────────────────────────────────────────────────

function main()
    BondUpdateBUG.set_symmetry!(:U1)
    # ⛔ SET THE CONTRACT-THREAD BOUND BEFORE ANYTHING RUNS, NOT INSIDE THE SWEEP LOOP.
    # It used to be set first inside `for ct in CTHREADS`, which sits below `random_mps` —
    # so the STATE BUILD ran at the default while the sweep believed it was pinned. Job
    # 16326783 died in `random_mps` for exactly that reason, with `LCP_CTHREADS=1` set and
    # doing nothing. A knob that only takes effect partway through the run is worse than no
    # knob, because the log claims a setting the measurement did not have.
    TELUM_CONTRACT_THREADS[] = CTHREADS[1] == 0 ? typemax(Int) : CTHREADS[1]
    mkpath(OUTDIR)
    csv = joinpath(OUTDIR, @sprintf("large_chi_L%d.csv", L))
    open(csv, "w") do io
        println(io, "L,chi,mpoD,arm,dex,blas,cthreads,step,seconds,alloc_gb,chi_out,nsectors")
    end

    say(@sprintf("L=%d  chis=%s  nsteps=%d  dt=%g  maxiter=%d  arms=%s",
                 L, string(CHIS), NSTEPS, DT, MAXITER, join(ARMS, ",")))
    say(@sprintf("conv_tol=%g (%s)  krylov_tol=%g", CONV_TOL,
                 CONV_TOL > 0 ? "adaptive Krylov exit, all arms" :
                                "BREAKDOWN-ONLY — TDVP arms burn full maxiter, BUG does not",
                 KRY_TOL))
    say(@sprintf("julia threads=%d  BLAS sweep=%s  CPU_THREADS=%d",
                 Threads.nthreads(), string(BLAS_T), Sys.CPU_THREADS))
    say("output -> $csv")

    say(@sprintf("MPO D sweep = %s (0 = physical xxz_mpo, D=5)", string(MPO_DS)))

    for mpoD in MPO_DS
    mpo = mpoD == 0 ? RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = DELTA) :
                      random_mpo(L, mpoD; seed = 7, delta = DELTA)
    Dact = maximum(mpo_virtual_dims(mpo))
    say("")
    say(repeat("#", 78))
    say(@sprintf("MPO bond dimension D = %d%s", Dact,
                 mpoD == 0 ? " (physical XXZ)" : " (random, requested $mpoD)"))
    say(repeat("#", 78))

    for chi in CHIS
        say("")
        say(repeat("=", 78))
        say(@sprintf("chi = %d", chi))
        say(repeat("=", 78))

        tb = @timed random_mps(L, chi; seed = 1, delta = DELTA)
        psi0 = tb.value
        say(@sprintf("  state built in %.1f s  chi=%d  %d elements  %.2f GB",
                     tb.time, maximum(BondUpdateBUG.bond_dims(psi0)),
                     mps_elements(psi0), gb(16 * mps_elements(psi0))))
        nsec, _ = report_sector_shapes(psi0)

        for ct in CTHREADS
            TELUM_CONTRACT_THREADS[] = ct == 0 ? typemax(Int) : ct
            for nt in BLAS_T
                for dex in DEXS
                    say(@sprintf("  --- BLAS=%d | contract tasks=%s | dex=%s (julia -t %d) ---",
                                 nt, ct == 0 ? "auto" : string(ct),
                                 dex == 0 ? "growth $(GROWTH)" : string(dex),
                                 Threads.nthreads()))
                    for arm in ARMS
                        # tdvp2 has no CBE expansion, so sweeping dex would re-time identical
                        # work. Run it on the first dex only.
                        arm == "tdvp2" && dex != DEXS[1] && continue
                        r = try
                            run_arm(arm, psi0, mpo, chi, dex, nt)
                        catch err
                            say("    $arm FAILED: $(sprint(showerror, err))")
                            continue
                        end
                            say(@sprintf("    %-6s MEAN(steps 2-%d) = %.2f s  [blas=%d%s]",
                                     arm, NSTEPS, r.mean, blas_for(arm, nt),
                                     (PARALLEL && startswith(arm, "bug")) ? ", parallel" : ""))
                        open(csv, "a") do io
                            for (k, t) in enumerate(r.times)
                                @printf(io, "%d,%d,%d,%s,%d,%d,%d,%d,%.6f,%.4f,%d,%d\n",
                                        L, chi, Dact, arm, dex, nt, ct, k, t,
                                        gb(r.allocs[k]), r.chi_out, nsec)
                            end
                        end
                    end
                end
            end
        end
        TELUM_CONTRACT_THREADS[] = typemax(Int)

        if DO_PROF
            say("  --- kernel breakdown (one step per arm, sampled) ---")
            BLAS.set_num_threads(BLAS_T[end])
            for arm in ARMS
                Profile.clear(); Profile.init(n = 10^7, delay = 0.005)
                try
                    psi = deepcopy(psi0)
                    tau = ComplexF64(-im * DT)
                    @profile if arm == "tdvp2"
                        RSVDCBEBondUpdate.tdvp2_step!(psi, mpo, tau;
                            maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER)
                    elseif arm == "cbe1s"
                        RSVDCBEBondUpdate.tdvp_cbe1s_step!(psi, mpo, tau;
                            maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER, exact = false)
                    else
                        RSVDCBEBondUpdate.cbe_bug_step!(psi, mpo, tau;
                            maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER, exact = false)
                    end
                    b, tot = profile_buckets()
                    tot == 0 && (say("    $arm: no samples"); continue)
                    for k in sort(collect(keys(b)), by = x -> -b[x])
                        b[k] == 0 && continue
                        say(@sprintf("    %-6s %-14s %5.1f%%  (%d samples)",
                                     arm, k, 100 * b[k] / tot, b[k]))
                    end
                catch err
                    say("    $arm profile FAILED: $(sprint(showerror, err))")
                end
            end
        end

        say(@sprintf("  peak RSS so far: %.2f GB", gb(Sys.maxrss())))
    end
    end

    say("")
    say("DONE — $csv")
end

main()
