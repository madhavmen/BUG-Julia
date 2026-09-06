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
                 "memcpy/copyto", "TLArray plumbing", "contract bookkeeping",
                 "svd bookkeeping", "oplus/sum", "BUGJulia sweep", "other", "IDLE"]

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
            # ⛔⛔ IDLE THREADS ARE NOT WORK, AND THEY WERE 47% OF THE FIRST BREAKDOWN.
            # Julia's sampler samples EVERY thread, so with OPENBLAS_NUM_THREADS=16 and
            # per-sector gemms too small to occupy them, fifteen workers sit parked in
            # `__futex_abstimed_wait_common` and land in every sample. They diluted the
            # denominator: BLAS gemm read 7.5% at 8 threads and 1.3% at 16, which looks like
            # gemm shrinking and is actually idle padding growing.
            # These are excluded from the total, not bucketed, so every percentage below is a
            # share of ACTIVE samples. The idle count is reported separately — it is a real
            # and useful number (it says the thread pool is oversubscribed for these shapes),
            # just not a share of the work.
            (occursin("futex", f) || occursin("pthread_cond", f) ||
             occursin("sched_yield", f) || occursin("nanosleep", f) ||
             occursin("poll", f) || f == "wait" || f == "uv_run" ||
             occursin("jl_task_get_next", f) || occursin("ijl_task_get_next", f)) &&
                return "IDLE"
            (occursin("gemm", f) || occursin("gemv", f) || occursin("syrk", f) ||
             f == "mul!" || occursin("generic_matmatmul", f)) && return "BLAS gemm"
            (occursin("gesdd", f) || occursin("gesvd", f) || occursin("geqrf", f) ||
             occursin("orgqr", f) || occursin("ungqr", f)) && return "LAPACK svd/qr"
            (occursin("hptt", f) || occursin("permutedims", f)) && return "permute/HPTT"
            (occursin("gc", lowercase(f)) || occursin("jl_alloc", f) ||
             occursin("array_", f)) && return "GC/alloc"
            # Bulk data movement that is NOT a permute: `copyto!`, `similar`, `fill!`,
            # `memmove`. Separated from GC because the fixes differ — GC pressure wants fewer
            # allocations, memcpy wants in-place kernels — and at 40% combined it matters
            # which half is which.
            (occursin("copyto", f) || occursin("memmove", f) || occursin("memcpy", f) ||
             f == "similar" || f == "fill!" || occursin("unsafe_copyto", f)) &&
                return "memcpy/copyto"
            occursin("svd.jl", file) && return "svd bookkeeping"
            occursin("contract.jl", file) && return "contract bookkeeping"
            (occursin("oplus", f) || occursin("sum_tlarray", file)) && return "oplus/sum"
            # Telum's own tensor plumbing outside contract/svd: TLArray construction,
            # to_concrete, itag and index handling. This is the layer `to_concrete` after
            # every contraction lands in.
            (occursin("TLArray.jl", file) || occursin("utils.jl", file) ||
             occursin("permute.jl", file)) && return "TLArray plumbing"
            (occursin("cbe_core.jl", file) || occursin("cbe_bug.jl", file) ||
             occursin("tdvp_cbe1s.jl", file) || occursin("tdvp2_baseline.jl", file) ||
             occursin("mpo.jl", file) || occursin("expv.jl", file)) &&
                return "BUGJulia sweep"
        end
    end
    return "other"
end

"""
The innermost frames that fell through every bucket, most common first.

⛔ WITHOUT THIS THE PROFILE IS UNACTIONABLE. The first chi=1024 breakdown put 39-44% —
the largest single bucket for cbe1s — into `other`, which says only "not one of the things
I thought to name". A bucket list is a hypothesis about where time goes; this prints what
the hypothesis missed, so the next refinement is driven by the data instead of by another
guess.
"""
function unclassified_top(n::Int = 12)
    data = Profile.fetch(include_meta = false)
    lidict = Profile.getdict(data)
    counts = Dict{String, Int}()
    bt = UInt64[]
    function tally(bt)
        classify_bt(bt, lidict) == "other" || return
        for ip in bt
            frames = get(lidict, ip, nothing)
            frames === nothing && continue
            for fr in (frames isa Vector ? frames : [frames])
                key = string(fr.func) * "  @ " * basename(string(fr.file))
                counts[key] = get(counts, key, 0) + 1
                return                      # innermost recognised frame only
            end
        end
    end
    for ip in data
        if ip == 0
            isempty(bt) || (tally(bt); empty!(bt))
        else
            push!(bt, ip)
        end
    end
    isempty(bt) || tally(bt)
    return sort(collect(counts), by = kv -> -kv[2])[1:min(n, length(counts))]
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
    # ACTIVE total: idle samples are excluded from the denominator so each bucket is a share
    # of real work, not a share of "how many threads happened to exist".
    return buckets, total - buckets["IDLE"], total
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

"""
A closure that advances this arm's own state by one step, plus that state.

⛔ SPLIT OUT OF `run_arm` SO THE SWEEP CAN RUN **REP-OUTER**. Running every step of one arm
before starting the next (arm-outer) makes each arm's number a sample of whatever the shared
node was doing during its slot, and the cluster nodes are shared: across two jobs with
IDENTICAL settings, tdvp2 moved 41.0 -> 56.9 s (+39%), cbe1s 46.9 -> 72.8 (+55%) and BUG
28.4 -> 56.2 (+98%). BUG degraded worst because its parallel half-sweeps need 16 concurrent
threads to pay, which is exactly what a busy node cannot give -- so arm-outer does not just
add noise, it BIASES against whichever arm is most contention-sensitive, and that arm is the
one whose advantage we are trying to measure.

Rep-outer interleaves the arms, so a contention episode lands on all of them.
"""
function make_stepper(arm::AbstractString, psi0, mpo, chi::Int, dex::Int)
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

    return step!, psi
end

"""
Run every arm rep-outer and return per-arm times and allocations.

`median` rather than `mean` over the measured steps: contention produces one-sided spikes,
and a single bad slot moves a 2-sample mean by half the spike.
"""
function run_all_arms(arms, psi0, mpo, chi::Int, dex::Int, nt::Int)
    steppers = Dict{String, Any}()
    states = Dict{String, Any}()
    for arm in arms
        s, p = make_stepper(arm, psi0, mpo, chi, dex)
        steppers[arm] = s; states[arm] = p
    end
    times = Dict(arm => Float64[] for arm in arms)
    allocs = Dict(arm => Float64[] for arm in arms)
    failed = String[]

    for k in 1:NSTEPS
        for arm in arms
            arm in failed && continue
            BLAS.set_num_threads(blas_for(arm, nt))
            GC.gc()
            try
                st = @timed steppers[arm]()
                push!(times[arm], st.time); push!(allocs[arm], st.bytes)
                say(@sprintf("    rep %d  %-6s  %8.2f s  alloc %6.2f GB  chi=%d",
                             k, arm, st.time, gb(st.bytes),
                             maximum(BondUpdateBUG.bond_dims(states[arm]))))
            catch err
                say("    rep $k  $arm FAILED: $(sprint(showerror, err))")
                push!(failed, arm)
            end
        end
    end
    return times, allocs, states
end

"Median of the measured steps — rep 1 dropped, since it carries first-call compilation."
function measured_median(v::Vector{Float64})
    isempty(v) && return NaN
    m = length(v) > 1 ? v[2:end] : v
    s = sort(m)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : 0.5 * (s[n ÷ 2] + s[n ÷ 2 + 1])
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
                    # tdvp2 has no CBE expansion, so sweeping dex would re-time identical
                    # work. Run it on the first dex only.
                    arms_here = [a for a in ARMS if a != "tdvp2" || dex == DEXS[1]]
                    times, allocs, states = run_all_arms(arms_here, psi0, mpo, chi, dex, nt)

                    say("    " * repeat("-", 60))
                    for arm in arms_here
                        isempty(times[arm]) && continue
                        med = measured_median(times[arm])
                        lo, hi = extrema(length(times[arm]) > 1 ? times[arm][2:end] :
                                         times[arm])
                        say(@sprintf("    %-6s MEDIAN = %7.2f s  (spread %.2f-%.2f)  [blas=%d%s]",
                                     arm, med, lo, hi, blas_for(arm, nt),
                                     (PARALLEL && startswith(arm, "bug")) ? ", parallel" : ""))
                    end
                    open(csv, "a") do io
                        for arm in arms_here
                            cout = maximum(BondUpdateBUG.bond_dims(states[arm]))
                            for (k, t) in enumerate(times[arm])
                                @printf(io, "%d,%d,%d,%s,%d,%d,%d,%d,%.6f,%.4f,%d,%d\n",
                                        L, chi, Dact, arm, dex, nt, ct, k, t,
                                        gb(allocs[arm][k]), cout, nsec)
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
                Profile.clear(); Profile.init(n = 2 * 10^6, delay = 0.01)
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
                    b, active, tot = profile_buckets()
                    active == 0 && (say("    $arm: no active samples"); continue)
                    say(@sprintf("    %-6s %d active samples of %d (%.0f%% of thread-samples were IDLE workers)",
                                 arm, active, tot, 100 * b["IDLE"] / max(tot, 1)))
                    for k in sort(collect(keys(b)), by = x -> -b[x])
                        (b[k] == 0 || k == "IDLE") && continue
                        say(@sprintf("    %-6s %-22s %5.1f%%  (%d samples)",
                                     arm, k, 100 * b[k] / active, b[k]))
                    end
                    if b["other"] > 0.05 * active
                        say(@sprintf("    %-6s -- what 'other' actually is --", arm))
                        for (name, c) in unclassified_top(12)
                            say(@sprintf("    %-6s    %5.1f%%  %s", arm, 100 * c / active, name))
                        end
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
