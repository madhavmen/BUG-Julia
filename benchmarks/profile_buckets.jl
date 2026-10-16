# KERNEL-LEVEL ATTRIBUTION OF A `Profile` RUN, shared by the profiling drivers
# (`large_chi_profile.jl`, `spectral/bug_profile_2d.jl`). Include AFTER `using Profile`.

using Profile


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
             occursin("mpo.jl", file) || occursin("expv.jl", file) ||
             occursin("krylov_closure.jl", file) || occursin("zero_site_core.jl", file) ||
             occursin("adaptive_budget.jl", file) || occursin("henv.jl", file)) &&
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

"""
    owner_top(n; srcdirs, withline = true) -> Vector{Pair{String,Int}}

The innermost frame of EACH active sample that lies in this package's own source, most common
first -- i.e. which line of OUR code the time was spent under, whatever Telum/BLAS kernel it ended
in. The bucket table says "gemm" or "permute"; this says which call site owns them, which is where
a fix has to go. `withline = false` aggregates by function.
"""
function owner_top(n::Int = 25; srcdirs = ("RSVDCBEBondUpdate", "BondUpdateBUG"),
                   withline::Bool = true, bucket::Union{Nothing, String} = nothing)
    data = Profile.fetch(include_meta = false)
    lidict = Profile.getdict(data)
    counts = Dict{String, Int}()
    bt = UInt64[]
    function tally(bt)
        b = classify_bt(bt, lidict)
        b == "IDLE" && return
        # `bucket = "GC/alloc"` restricts to one kernel bucket: which of OUR lines allocate.
        bucket === nothing || b == bucket || return
        for ip in bt                                   # innermost first
            frames = get(lidict, ip, nothing)
            frames === nothing && continue
            for fr in (frames isa Vector ? frames : [frames])
                file = string(fr.file)
                any(d -> occursin(d, file), srcdirs) || continue
                key = string(fr.func) * "  @ " * basename(file) * (withline ? ":$(fr.line)" : "")
                counts[key] = get(counts, key, 0) + 1
                return
            end
        end
        counts["(no package frame)"] = get(counts, "(no package frame)", 0) + 1
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
