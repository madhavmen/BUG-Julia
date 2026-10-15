#!/usr/bin/env julia
# ── WHERE DOES A FUSED d=4 STEP ACTUALLY SPEND ITS TIME? ──────────────────────────────────────
#
# ⛔ THIS EXISTS BECAUSE A bd=32 STEP TOOK 334 SECONDS AND THAT IS NOT ARITHMETIC. MEASURED
# (job 16341790, `phase_hub4prod`, L=20, chi=32, cutoff 1e-8, `bug`):
#
#     step 1   0.8 s      product state -> bd 32, the cap never binds, nothing is truncated
#     step 2   334.3 s    the first step that has to truncate
#     step 3   > 325 s    still running
#
# Step 2 alone would be a cold-cache artefact -- this repo documents ~313 s for an uncached first
# step, and the fused d=4 path was indeed missing from the precompile workload (now added in
# `src/BUGJulia.jl`). ⛔ BUT STEP 3 IS SLOW TOO, so compilation cannot be the whole story and the
# remaining cost is real work. `krylov` says step 2 used 284 matvecs, i.e. ~1.2 s PER MATVEC at
# bond dimension 32 -- three orders of magnitude more than the arithmetic justifies.
#
# The comparison that decides it, all in ONE process so the cache state is shared:
#
#   * `--scan`  L = 6, 10, 14, 20 at fixed chi -- is this a smooth O(L) cost or a CLIFF? A cliff
#     at fixed chi cannot be arithmetic and points at a per-block overhead that only shows up once
#     enough bonds exist.
#   * the d=2 XXZ chain at the SAME L and chi -- the spin campaigns run ~10 s/step at chi=128, so
#     a large d=4/d=2 ratio localises the defect to the fused model rather than to the sweeps.
#   * a flat profile of the LAST step, which names the function instead of leaving it to inference.
#
# ⚠ THE PRIME SUSPECT IS THIS REPO'S OWN MPO, NOT TELUM. `hubbard_aa_mpo_d4` pre-fills all 36
# slots of its 6x6 automaton with an explicit ZERO block (`Znil`) to dodge an `oplus` space
# inference failure. A sparse Hubbard automaton has ~11 non-zero entries, so every contraction
# carries ~3.3x the block pairs it needs -- and if the virtual leg is built as six ONE-dimensional
# sectors rather than one six-dimensional one, the block-pair count and its per-block overhead
# rise much faster than that. `--mpoblocks` counts them instead of assuming.
#
# ⛔ RUN THIS ON THE CLUSTER. Local runs are banned for this project and a 334 s step would be
# unbearable besides.
#
# usage:  julia -t 2 --project=. benchmarks/hub4_profile.jl [L] [chi] [nsteps] [scheme]

using LinearAlgebra, Printf
using Profile
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const L      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const CHI    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32
const NSTEP  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
const SCHEME = length(ARGS) >= 4 ? ARGS[4] : "bug"
const DT     = 0.1
const CUT    = 1e-8

@printf("host=%s threads=%d L=%d chi=%d nsteps=%d scheme=%s\n",
        gethostname(), Threads.nthreads(), L, CHI, NSTEP, SCHEME); flush(stdout)

# ── the two models, at identical L and chi ───────────────────────────────────────────────────
function d4_case(Lp)
    set_symmetry!(:none)
    occ = hub4_cdw_occupations(Lp; doublon_frac = 0.5, seed = 1)
    psi = hub4_product_state(occ)
    mpo = hubbard_aa_mpo_d4(Lp; J = 1.0, Delta = 8.0, U = 4.0, beta = 532 / 738, phi = 0.0)
    return psi, mpo
end

function d2_case(Lp)
    set_symmetry!(:none)
    psi = neel_state(Lp)
    mpo = xxz_mpo(Lp; J = 1.0, delta = 1.0)
    return psi, mpo
end

stepfn(scheme, mpo, maxdim) =
    scheme == "bug"    ? (p, t) -> cbe_bug_step!(p, mpo, t; maxdim = maxdim,
                              trunc_thresh = CUT, split_cutoff = CUT, maxiter = 30,
                              krylov_basis = 30, root_conv_tol = 0.0, root_substeps = 1) :
    scheme == "bugmid" ? (p, t) -> cbe_bug_midpoint_step!(p, mpo, t; maxdim = maxdim,
                              trunc_thresh = CUT, split_cutoff = CUT, maxiter = 30,
                              krylov_basis = 30, root_conv_tol = 0.0, root_substeps = 1) :
                         (p, t) -> tdvp2_step!(p, mpo, t; maxdim = maxdim,
                              trunc_thresh = CUT, maxiter = 30, conv_tol = 0.0)

"""
    block_report(psi) -> (nmax, ntot, dmax)

How many SECTORS does each bond carry, against how much total dimension?

⛔⛔ THIS IS THE MEASUREMENT THE `maxbond` COLUMN CANNOT MAKE, AND THE CAMPAIGN'S COST ANOMALY IS
SHAPED EXACTLY LIKE ITS FAILURE. `bond_dims` reports `sum(d for (_, d) in t.spaces[3])` -- the
TOTAL dimension of a bond, summed over its blocks. A bond of dimension 32 can therefore be ONE
32x32 block or THIRTY-TWO 1x1 blocks, and the CSV cannot tell the two apart.

They cost wildly different amounts. Block-sparse contraction iterates over block PAIRS, so
fragmenting one bond into `k` blocks turns a single matrix multiply into up to `k^2` tiny ones,
each paying full dispatch and lookup overhead and none of them large enough to amortise it.

MEASURED cost at L=20, chi=32, where the truncated bond dimension never moves off 32:

    step 1     0.8 s
    step 2   334.3 s
    step 3  > 900 s

⇒ THE COST GROWS WHILE `maxbond` IS PINNED. A fixed structural overhead cannot do that; something
is ACCUMULATING that the reported rank does not show. Fragmentation does exactly this if the BUG
augmentation's `oplus` appends new blocks that the closing truncation never MERGES back.

⚠ UNDER `:none` THE ANSWER SHOULD BE 1. There is a single trivial sector, so every leg must carry
exactly one block. Anything above 1 here is a defect, not a tuning question -- and it would also
explain why the spin (d=2) campaigns are unaffected only if they do not fragment, which is why
`--vs-d2` prints this for both.
"""
function block_report(psi)
    ns = [length(psi[i].spaces[3]) for i in 1:(length(psi) - 1)]
    ds = bond_dims(psi)
    return maximum(ns; init = 0), sum(ns; init = 0), maximum(ds; init = 0)
end

"""
Run `n` steps, reporting each one SEPARATELY.

⛔ THE PER-STEP SPLIT IS THE WHOLE POINT. A total, or a mean, would have hidden the 0.8 s / 334 s
split that started this file -- the first step from a product state never truncates and is
therefore not representative of any later one.
"""
function timed(label, psi, mpo, maxdim, n)
    step = stepfn(SCHEME, mpo, maxdim)
    p = copy(psi)
    times = Float64[]
    allocs = Float64[]
    for k in 1:n
        # ⛔ PER STEP, NOT CUMULATIVE. `fusion_basis` is reported because the time profile blamed
        # it; the counters then showed only **28 calls, 26 misses** per step, which is how we know
        # it is neither the bytes nor cacheable.
        #
        # ⛔⛔ `kry` IS THE HYPOTHESIS UNDER TEST. Allocation GROWS 28.5 -> 152.8 -> 233.5 GiB over
        # steps 2-4 while `maxbond` sits at 32 and blocks/bond at 1, so something that is NOT the
        # state size is accumulating. This repo's recorded hottest allocation site is
        # `tensor_inner`, "39.9% of tdvp2's sampled bytes, because reorth'd Lanczos calls it
        # O(maxiter^2) per solve". If the Krylov DEPTH climbs from step to step as the state gets
        # harder, an O(k^2) reorthogonalisation reproduces exactly this growth at pinned rank.
        # `maxiter = 30` here, so there is a lot of room for k to climb into.
        # ⚠ A FLAT `kry` REFUTES IT and sends the search back to the allocation profile.
        reset_fusion_stats!()
        st = time(); kd = Ref(0)
        a = @allocated begin
            info = step(p, ComplexF64(-im * DT))
            kd[] = hasproperty(info, :krylov_dims) ? info.krylov_dims :
                   hasproperty(info, :krylov_dim)  ? info.krylov_dim  : -1
        end
        el = time() - st
        fs = fusion_stats()
        push!(times, el); push!(allocs, a / 2^30)
        nmax, ntot, dmax = block_report(p)
        @printf("  %-14s step %d  %8.2f s  bd=%-5d blocks max=%-3d alloc=%7.2f GiB  kry=%-7d fusion: calls=%-6d miss=%d\n",
                label, k, el, dmax, nmax, a / 2^30, kd[], fs.calls, fs.misses)
        flush(stdout)
    end
    return times, allocs, p
end

# ── 1. the size scan: cliff or slope? ────────────────────────────────────────────────────────
if "--scan" in ARGS
    println("\n=== d=4 SIZE SCAN at chi=$CHI (2 steps each; step 2 is the truncating one) ===")
    for Lp in (6, 10, 14, 20)
        psi, mpo = d4_case(Lp)
        @printf("  L=%-3d w=%d\n", Lp, maximum(mpo_virtual_dims(mpo); init = 0))
        timed("d4 L=$Lp", psi, mpo, CHI, 2)
    end
end

# ── 2. d=4 against d=2 in the SAME process ───────────────────────────────────────────────────
if "--vs-d2" in ARGS
    println("\n=== d=4 vs d=2 at L=$L chi=$CHI ===")
    p4, m4 = d4_case(L); timed("d4", p4, m4, CHI, NSTEP)
    p2, m2 = d2_case(L); timed("d2", p2, m2, CHI, NSTEP)
end

# ── 3. how many blocks is the MPO actually carrying? ─────────────────────────────────────────
#
# The `Znil` prefill is a deliberate hack and this counts its price rather than assuming it.
if "--mpoblocks" in ARGS
    println("\n=== MPO structure ===")
    # ⚠ VIRTUAL DIMS ONLY. Telum does not export a block iterator here, and the flat profile below
    # names the hot function directly -- inventing an accessor to count blocks would be a second
    # unproven thing in series with the one being diagnosed.
    for (name, (_, mpo)) in (("d4", d4_case(L)), ("d2", d2_case(L)))
        @printf("  %-3s virtual dims %s\n", name, string(mpo_virtual_dims(mpo)))
    end
end

# ── 4. the flat profile of a representative (post-truncation) step ───────────────────────────
if "--profile" in ARGS
    println("\n=== flat profile of a truncating d=4 step, L=$L chi=$CHI ===")
    psi, mpo = d4_case(L)
    step = stepfn(SCHEME, mpo, CHI)
    p = copy(psi)
    step(p, ComplexF64(-im * DT))          # step 1: no truncation, and it warms the cache
    step(p, ComplexF64(-im * DT))          # step 2: first truncating step, absorbs any JIT
    # 2e7 slots at 0.005 s is ~10x headroom for a 334 s step (200 samples/s x ~100 frames), and
    # 160 MB rather than the 800 MB a 1e8 buffer would pin for no benefit.
    Profile.init(n = 2 * 10^7, delay = 0.005)
    Profile.clear()
    @profile step(p, ComplexF64(-im * DT)) # step 3: the one we attribute
    Profile.print(IOContext(stdout, :displaysize => (24, 200));
                  format = :flat, sortedby = :count, mincount = 20)
    flush(stdout)
end

# ── 5. WHERE DO THE BYTES GO? (allocation profile, not a time profile) ───────────────────────
#
# ⛔⛔ THE TIME PROFILE ANSWERED A DIFFERENT QUESTION AND I ACTED ON IT ANYWAY. It attributed 45%
# of a step to `fusion_basis`, so `fusion_basis` was memoised -- and the memo is CORRECT (45/45
# unit tests) and CHANGED NOTHING: allocation at L=14 went 15.44 -> 15.52 GiB. The counters said
# why: **28 calls per step, 26 of them misses**. A function called 28 times cannot be holding
# 15 GiB, and 28 calls cannot be cached when every key is distinct.
#
# ⇒ A TIME PROFILE AND AN ALLOCATION PROFILE ARE NOT THE SAME MEASUREMENT. The step is
# allocation-bound (wall clock tracks bytes at a flat ~0.5 GiB/s across every L), so the bytes are
# the thing to attribute, and only `Profile.Allocs` attributes them.
#
# ⚠ `sample_rate` SCALES THE REPORTED BYTES -- at 0.01 the totals are ~1% of the true figure. Read
# the RANKING and the SHARES, never the absolute bytes, and compare against the `@allocated` total
# that `timed` prints for the same step.
# ⚠ Frames are attributed to the FIRST line in our own code or in Telum, walking outward from the
# allocation site: a raw profile blames `Array{ComplexF64}` constructors, which is true and useless.
if "--allocs" in ARGS
    println("\n=== allocation profile of a truncating d=4 step, L=$L chi=$CHI ===")
    psi, mpo = d4_case(L)
    step = stepfn(SCHEME, mpo, CHI)
    p = copy(psi)
    step(p, ComplexF64(-im * DT))          # step 1: no truncation, warms the cache
    step(p, ComplexF64(-im * DT))          # step 2: first truncating step
    try
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=0.01 step(p, ComplexF64(-im * DT))
        res = Profile.Allocs.fetch()
        agg = Dict{String, Vector{Int}}()
        for a in res.allocs
            site = "unattributed"
            for f in a.stacktrace
                s = String(string(f.file))
                if occursin("BUGJulia", s) || occursin("Telum", s)
                    site = string(f.func, " @ ", basename(s), ":", f.line)
                    break
                end
            end
            v = get!(agg, site, [0, 0])
            v[1] += 1
            v[2] += a.size
        end
        tot = sum(v[2] for v in values(agg); init = 0)
        @printf("  %d sampled allocations, %.3f GiB sampled (sample_rate=0.01)\n",
                sum(v[1] for v in values(agg); init = 0), tot / 2^30)
        println("  ── top sites by BYTES ──")
        for (site, v) in first(sort(collect(agg); by = kv -> -kv[2][2]), 25)
            @printf("  %6.2f%%  %9.3f GiB  n=%-8d %s\n",
                    100 * v[2] / max(tot, 1), v[2] / 2^30, v[1], site)
        end
    catch e
        @printf("  allocation profile unavailable: %s\n", sprint(showerror, e)[1:min(end, 200)])
    end
    flush(stdout)
end

# ── 6. THE KRYLOV-DEPTH LADDER: cost AND accuracy together ───────────────────────────────────
#
# ⛔⛔ THE SUSPECTED ROOT CAUSE, AND THE ARITHMETIC THAT NAMES IT. `expv.jl`'s own comment on
# `tensor_inner` says re-orthogonalised Lanczos calls it **O(maxiter^2) per solve** -- "two passes
# over the whole basis every iteration" -- and Telum allocates its contraction buffers PER CALL.
# At `maxiter = 30` that is ~900 inner products per solve; with ~38 solves per step and a two-site
# tensor of 32x4x4x32 (262 KB), the buffers alone come to ~17 GB per step. MEASURED: 15.5 GiB.
#
# And it explains the growth that `maxbond` cannot. If the depth the solver actually uses climbs
# as the state gets harder, cost goes as k^2: a rise from ~10 to ~30 is 9x, against the measured
# 28.5 -> 233.5 GiB over steps 2-4 at PINNED rank.
#
# ⇒ SO THE LADDER MUST REPORT COST **AND** ACCURACY. Cutting `maxiter` is only a fix if the answer
# does not move, and this repo has already measured that it does not: `dn` was BIT-IDENTICAL from
# m = 5 to m = 120, a 24x span ([[krylov-depth-inflates-bug-advantage]] reproduced on this model).
# ⚠ A cost win with a moved answer is NOT a fix, it is a silent accuracy regression -- which is
# why `dmax` below is graded against the m=30 arm rather than reported on its own.
if "--kry" in ARGS
    println("\n=== Krylov-depth ladder at L=$L chi=$CHI: allocation AND accuracy ===")
    psi, mpo = d4_case(L)
    # ⚠ A `Ref`, NOT a bare binding. At top level inside `if`/`for` this is SOFT SCOPE: a plain
    # `ref = ...` inside the loop becomes a NEW LOCAL and the outer value is never seen
    # (`UndefVarError: ref not defined in local scope` -- it killed the first run of this stage).
    # The name is also changed off `ref`, which collides with a Base global.
    dref = Ref(Float64[])
    for m in (30, 16, 8, 5)
        p = copy(psi)
        stp = (q, t) -> cbe_bug_step!(q, mpo, t; maxdim = CHI, trunc_thresh = CUT,
                                      split_cutoff = CUT, maxiter = m, krylov_basis = m,
                                      root_conv_tol = 0.0, root_substeps = 1)
        stp(p, ComplexF64(-im * DT))                       # step 1: no truncation
        a2 = @allocated stp(p, ComplexF64(-im * DT))       # step 2: first truncating step
        t3 = time(); a3 = @allocated stp(p, ComplexF64(-im * DT)); el3 = time() - t3
        dens = [hub4_density(p, i) for i in 1:L]
        isempty(dref[]) && (dref[] = copy(dens))           # m=30 runs first and IS the reference
        dmax = maximum(abs.(dens .- dref[]); init = 0.0)
        @printf("  m=%-4d step2=%7.2f GiB  step3=%7.2f GiB %8.2f s  bd=%-4d  max|dn - dn(m=30)|=%.3e\n",
                m, a2 / 2^30, a3 / 2^30, el3, maximum(bond_dims(p); init = 0), dmax)
        flush(stdout)
    end
    println("  ⇒ if allocation falls ~(30/m)^2 while max|dn| stays at roundoff, the depth is the defect.")
end

println("\ndone."); flush(stdout)
