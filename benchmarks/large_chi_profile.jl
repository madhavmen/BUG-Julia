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

include(joinpath(@__DIR__, "random_mps.jl"))

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
const ARMS     = split(get(ENV, "LCP_ARMS", "tdvp2,cbe1s,bug"), ',')
const DELTA    = envfloat("LCP_DELTA", 1.0)
const CONV_TOL = envfloat("LCP_CONV_TOL", 0.0)   # 0 = old breakdown-only behaviour
const KRY_TOL  = envfloat("LCP_KRY_TOL", 1e-6)   # BUG half-sweep frame tolerance
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

function run_arm(arm::AbstractString, psi0, mpo, chi::Int)
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
                  conv_tol = CONV_TOL)
    elseif arm == "bug"
        () -> RSVDCBEBondUpdate.cbe_bug_step!(psi, mpo, tau;
                  maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER, exact = false,
                  root_conv_tol = CONV_TOL, krylov_tol = KRY_TOL)
    elseif arm == "bugmid"
        () -> RSVDCBEBondUpdate.cbe_bug_midpoint_step!(psi, mpo, tau;
                  maxdim = chi, trunc_thresh = 0.0, maxiter = MAXITER, exact = false,
                  root_conv_tol = CONV_TOL, krylov_tol = KRY_TOL)
    else
        error("unknown arm $arm")
    end

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
    mkpath(OUTDIR)
    csv = joinpath(OUTDIR, @sprintf("large_chi_L%d.csv", L))
    open(csv, "w") do io
        println(io, "L,chi,arm,blas,step,seconds,alloc_gb,chi_out,nsectors")
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

    mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = DELTA)

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

        for nt in BLAS_T
            BLAS.set_num_threads(nt)
            say(@sprintf("  --- BLAS threads = %d ---", nt))
            for arm in ARMS
                r = try
                    run_arm(arm, psi0, mpo, chi)
                catch err
                    say("    $arm FAILED: $(sprint(showerror, err))")
                    continue
                end
                say(@sprintf("    %-6s MEAN(steps 2-%d) = %.2f s", arm, NSTEPS, r.mean))
                open(csv, "a") do io
                    for (k, t) in enumerate(r.times)
                        @printf(io, "%d,%d,%s,%d,%d,%.6f,%.4f,%d,%d\n",
                                L, chi, arm, nt, k, t, gb(r.allocs[k]), r.chi_out, nsec)
                    end
                end
            end
        end

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

    say("")
    say("DONE — $csv")
end

main()
