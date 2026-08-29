# IS THE SKETCH ITSELF FASTER THAN THE EXACT PROBE? -- MEASURED IN ISOLATION.
#
# ⛔ END-TO-END DMRG TIMING CANNOT ANSWER THIS, AND TWO RUNS PROVED IT. `gse_rsvd_vs_exact.jl`
# found RSVD 0.88-0.97x across six configurations, and a `growth` scan found 0.95x even at
# `growth = 1.05` where the sketch is only ~6% of the full local space and MUST win on operation
# count. The reason both failed is visible in their own numbers: at `growth = 1.05` the solve took
# 59.4s and 4896 matvecs against 15s and 1232 at `growth = 2.0` -- starving the budget makes the
# rank grow slowly, so many more sweeps are needed, and THE SKETCH IS A SMALL FRACTION OF TOTAL
# RUNTIME IN EVERY REGIME. A component worth a few percent of the total cannot be measured by
# timing the total; the Krylov solves dominate whatever the probe does.
#
# So this file times the PROBE PATHS ALONE, at controlled dimensions, with nothing else running:
#
#   exact:  full_local_basis(frame, side)                 -> the complete fused basis, d*chi wide
#   RSVD :  sector_graded_sketch(frame, side, npre)       -> a Gaussian, npre wide
#
# and then the operator application and preselection each one feeds, since those are where the
# width is supposed to pay off. The ratio `npre / (d*chi)` is printed on every row because it is
# the quantity that decides whether sketching CAN help, independent of implementation quality.
#
# WHAT A HEALTHY RESULT LOOKS LIKE: time roughly linear in `npre` for the sketch path and flat at
# the `d*chi` cost for the exact path, so the ratio grows as `npre/(d*chi)` falls. If the sketch
# path is NOT much cheaper at `npre/(d*chi) = 0.05`, the cost is inside the sketch construction
# itself -- sector grading, the `RepairOrtho` preselection, or an accidental materialisation of
# the full object -- and that is a fixable implementation problem rather than an asymptotic one.
#
# ══ RESULT: THE SKETCH IS NOT THE DEFECT. THE PROBE IS ALLOCATION-BOUND. ════════════════════
#
# TIME, median of 5 over calibrated inner repetitions:
#
#   chi=16 full=32   npre  2 ->  32    9.81  8.55 10.48 12.13 10.32 ms
#   chi=32 full=54   npre  3 ->  54    9.46 12.53 10.38 12.97 12.00 ms
#   chi=64 full=82   npre  5 ->  82   12.94 14.12 12.21 13.45 13.97 ms
#
# ALLOCATION per `sketch_h_left` call, and how much of it is `H*Theta`:
#
#   chi=16 full=32   npre  2 ->  32   1.530 -> 1.540 MB    H*Theta = 84%
#   chi=64 full=82   npre  5 ->  82   3.472 -> 3.555 MB    H*Theta = 84-86%
#
# FLAT IN `npre` in both time and bytes, over a 16-18x range of sketch width, at every rank. The
# sketch changes nothing because `sketch_h_left` materialises the full `H*Theta` and the full
# `P_perp(H*Theta)` before it ever contracts the Gaussian.
#
# ⛔ AND THE OBVIOUS FIX -- "fold Om into W/R so H*Theta is never formed" -- IS WRONG. `Theta` is
# `n x n` with `n = d*chi`; applying `H` costs `O(n^2 * w)` for `w` MPO channels/terms (3-9 here),
# NOT `O(n^3)`. Pushing `Om` through instead costs `O(n^2 * k)` per term, which is MORE expensive
# than forming `H*Theta` whenever `k > w` -- most of the range. `H*Theta` cannot be avoided
# cheaply, and an earlier version of this file recommended exactly that.
#
# WHERE RSVD'S SAVING ACTUALLY IS: the PRESELECTION SVD. Exact factorises `n x n` at `O(n^3)`;
# the sketch factorises `n x k` at `O(n k^2)`. At n = 32-82 that is 3e4-5e5 flops -- MICROSECONDS
# -- inside a probe that spends 10 ms, 84% of it allocating. The genuine asymptotic win is real
# and is currently invisible beneath allocation overhead.
#
# ⛔ THE OVERHEAD IS A FIXED PER-CALL COST, NOT A DATA COST. From chi=16 to chi=64 the tensor grows
# 16x (16 KB -> 256 KB) while allocation grows 2.3x (1.53 -> 3.47 MB). At chi=16 the probe
# allocates 96x THE SIZE OF THE TENSOR IT OPERATES ON. That is symmetric-block bookkeeping and
# temporaries -- `apply_h_two_site` accumulates `2 + 2*|terms| + 1` full-size tensors, each
# through `to_concrete(acc + x)`.
#
# SO THE ORDER OF WORK IS: (1) cut the allocations in `apply_h_two_site` -- that is where the
# measured time is, and it also accelerates the Krylov solves that are the other ~88% of a solve;
# (2) re-measure this file, where the SVD should then dominate the probe; (3) only then test RSVD
# at chi in the hundreds (L >= 24, cluster), where `n^3` dominates and the sketch can win.
#
# ⛔ AND AT L=12 THE CEILING IS 12% WHATEVER IS DONE HERE. The probe runs ~176 times per solve
# ((L-1) bonds x 2 sides x grow_iters x sweeps) at ~10 ms = ~1.8 s of a 15.3 s solve. DELETING it
# entirely would buy 12%. "Huge speedups" are not available at this size from this component.
#
# Run:  julia --project=. benchmarks/rsvd_probe_microbench.jl

using LinearAlgebra, Printf, Random, Statistics
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
const R = RSVDCBEBondUpdate

# ⛔ ONE CALL PER TIMING IS UNMEASURABLE HERE AND THE FIRST VERSION OF THIS FILE PROVED IT.
# `time()` resolves to about 1 ms on this machine and these probes take ~5 ms, so every reading
# came back an exact multiple of 0.001 -- 3 to 11 TICKS -- and the sketch column was not even
# monotone in `npre` (5, 9, 10, 6, 11 ms), which is impossible for real work. It would have
# reported "1.60x, sketch pays" from a 5-tick against 8-tick comparison.
#
# So each timing runs the operation `inner` times and divides. `calibrate` picks `inner` so the
# timed block lasts at least `floor_s`, i.e. hundreds of ticks, making quantisation negligible.
"Time `f()` with enough inner repetitions that timer granularity is negligible."
function med(f, n = 5; floor_s = 0.20)
    f()                                   # once, untimed: allocate and warm any first-call path
    inner = 1
    while true
        t0 = time(); for _ in 1:inner; f(); end; el = time() - t0
        el >= floor_s && break
        inner = max(inner * 2, ceil(Int, inner * floor_s / max(el, 1e-6)))
        inner > 1 << 22 && break
    end
    ts = Float64[]; out = nothing
    for _ in 1:n
        t0 = time(); for _ in 1:inner; out = f(); end
        push!(ts, (time() - t0) / inner)
    end
    return median(ts), out
end

"""
Build a warm state at bond dimension ~`chi` so the frames are realistic rather than random:
the sketch is graded BY SECTOR, so its cost depends on the charge structure, which a random
tensor would not reproduce.
"""
function warm_state(L, chi)
    set_symmetry!(:U1)
    psi = neel_state(L)
    h = xxz_chain(L; delta = 1.0)
    m = R.mpo_from_terms(h)
    for _ in 1:4
        R.cbe_bug_step!(psi, m, ComplexF64(-0.1); maxdim = chi, trunc_thresh = 1e-12,
                        maxiter = 20)
        psi[psi.center] = to_concrete((1.0 / norm(psi)) * psi[psi.center])
    end
    return psi, h
end

println("PROBE MICROBENCHMARK -- the two candidate-space constructions, timed alone")
println("(median of 5; `full` = d*chi = the exact probe's width)\n")
@printf("%-6s %-6s %6s %6s %7s   %11s %11s %9s\n",
        "L", "chi", "full", "npre", "npre/fu", "sketch (s)", "exact (s)", "speedup")

for (L, chi) in ((12, 16), (12, 32), (16, 64))
    psi, h = warm_state(L, chi)
    i = L ÷ 2
    canonical!(psi, i)
    f = bond_frame(psi, i)
    # `fused_dim` is a LOCAL closure inside `cbe_expand`, not a module function -- inlined here
    # rather than exported, so this benchmark cannot drift from the definition the sweep uses.
    full = sum(d for (_, d) in R.reachable_sectors(f.U0, 1, 2); init = 0)
    lch  = R.left_channels(R.left_env_stack(psi, h; upto = L - 1), i)
    rch  = R.right_channels(R.right_env_stack(psi, h; downto = 2), i + 2)

    for frac in (0.05, 0.1, 0.25, 0.5, 1.0)
        npre = max(1, ceil(Int, frac * full))
        # RSVD path: draw the graded Gaussian, apply H through it, preselect.
        ts, _ = med() do
            Om = R.sector_graded_sketch(f.V0, :right, npre; comp_ratio = 0.5,
                                        rng = MersenneTwister(0x5EED))
            Om === nothing ? nothing : R.sketch_h_left(f, h, i, lch, rch, Om)
        end
        # exact path: the complete fused basis, same downstream call.
        te, _ = med() do
            F = R.full_local_basis(f.V0, :right)
            R.sketch_h_left(f, h, i, lch, rch, F)
        end
        @printf("%-6d %-6d %6d %6d %6.2f    %11.5f %11.5f %8.2fx%s\n",
                L, chi, full, npre, npre / full, ts, te, te / ts,
                te / ts > 1.2 ? "  <- sketch pays" : "")
        flush(stdout)
    end
    println()
end
