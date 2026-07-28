# What does the expansion budget actually COST -- per problem, and in chi?
#
#     julia --project=. benchmarks/cbe_budget_cost.jl [why [rev] | model | chi [L] [t] [caps]]
#
# THE CLAIM THIS FILE WAS BUILT TO EXPLAIN IS RETRACTED. Raising the growth schedule from 1.1
# to 2.0 appeared to make both paths FASTER as well as more accurate (gate 24.4s -> 3.9s, MPO
# 19.7s -> 3.6s, XX L=10). It does not. That was Julia's first-call JIT compilation being
# charged to whichever configuration ran first, and the starved setting was always first.
#
# THE MEASUREMENT THAT SETTLED IT, and the reason `warmup` now runs before any timing: three
# `growth=1.1` runs at caps 8/16/32 were NUMERICALLY IDENTICAL -- error equal to four digits,
# maxbond 4 in all three (the cap never binds), 4.9 iterations per step -- and took 160.6s,
# 4.6s, 4.6s. No property of the method can differ across those rows.
#
# What the counters showed BEFORE the timings were trusted, and what stands independently of
# them (they are collected inside the same runs, so contention cannot touch them):
#
#   * gate path, starved vs generous: 347.9 vs 343.6 iterations per step -- IDENTICAL, with the
#     generous run doing MORE total work (601.8k vs 486.6k iteration-times-size). So the 0-site
#     Krylov was never the cost, and a "poor subspace is harder to converge" story is dead.
#   * MPO path, starved: 6x FEWER iterations on 4x smaller objects. Cheaper by every internal
#     measure, so any wall-time story that made it slower had to be external. It was.
#
# So the honest statement about the budget is: it buys accuracy (1.4e-4 -> 1.2e-6) at
# essentially no steady-state wall-time cost, NOT a speedup. `why` re-measures that with the
# warm-up in place, and `rev` swaps row order so an order-dependent artefact cannot hide.
#
# PROBLEM SPECIFICITY (phase `model`). XX is free fermions and a domain wall is a very special
# initial state, so the same grid runs on Heisenberg (delta=1.0), which is interacting. No
# analytic reference there, so this phase reports COST AND RANK only, with total Sz as the
# invariant that must hold regardless.
#
# SCALING (phase `chi`). The cost question the method lives or dies on. With the ratio
# schedule, `npre ~ 1.2*growth*chi`, so per bond the sketch is O(w d chi^2 npre) = O(chi^3),
# the preselection SVD is O(d chi npre^2) = O(chi^3), and the final SVD is O(npre^3) = O(chi^3)
# -- the SAME order as the canonicalisation the sweep already pays, with no rank-4 object in
# the MPO path. A FIXED `dex` would instead be O(chi^2 dex), asymptotically cheaper but
# measured to lose accuracy once the image outgrows the budget. So the prediction is: time
# ~ chi^3 for both, with the ratio schedule carrying a larger constant and NOT a worse
# exponent. A fitted exponent much above 3 is the blow-up to look for.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf
include("../tests/common/free_fermion.jl")

const PHASE = isempty(ARGS) ? "all" : ARGS[1]

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]
heis_gates(p) = Any[heisenberg_bond_gate(p[i].inds[2], p[i + 1].inds[2]; delta = 1.0)
                    for i in 1:(length(p) - 1)]

"Normalised site-resolved <Sz>; `site_expval` does not divide by the norm."
prof(psi) = (q = copy(psi); magnetisation(q) ./ norm(q)^2)

const HDR = @sprintf("%-26s %10s %8s %8s %9s %9s %9s %9s",
                     "setting", "err", "maxbond", "wall_s", "kry/step", "mean_aug",
                     "work/1e3", "err_fnl")

"""
Compile both paths before ANY timing. Without this, Julia charges the first timed row the JIT
cost of the whole call chain and every conclusion drawn from wall time is about compilation.

THIS IS NOT A PRECAUTION, IT IS A CORRECTION. It was measured: three `growth=1.1` runs at
caps 8/16/32 came out NUMERICALLY IDENTICAL (same error to four digits, same maxbond 4, same
4.9 iterations/step -- the cap never binds) at 160.6s, 4.6s and 4.6s. Nothing about the method
can differ between those rows. The same artefact produced the retracted claim that a generous
expansion budget is *faster*: in `cbe_bond_update_budget.jl` the first CBE row carried ~20s of
residual compilation (24.4s vs a true 3.9s), and `bond_update_bug!` running ahead of it had
already absorbed the rest.
"""
function warmup(L::Int)
    set_symmetry!(:U1)
    o = CBEBugOptions(dt = 0.01, n_steps = 1, maxdim = 8, trunc_thresh = 1e-14)
    p = domain_wall_state(L); cbe_bond_update_bug!(p, xx_gates(p); opts = o)
    p = domain_wall_state(L); cbe_bond_update_bug!(p, heis_gates(p); opts = o)
    q = domain_wall_state(L); cbe_lubich_bug!(q, xxz_chain(L; J = 1.0, delta = 0.0); opts = o)
    q = domain_wall_state(L); cbe_lubich_bug!(q, xxz_chain(L; J = 1.0, delta = 1.0); opts = o)
    # Both `maxiter` settings too: a different `maxiter` is a different specialization only if
    # it changes types, which it does not -- but the Lanczos tail paths still want touching.
    o8 = CBEBugOptions(dt = 0.01, n_steps = 1, maxdim = 8, trunc_thresh = 1e-14, maxiter = 8)
    p = domain_wall_state(L); cbe_bond_update_bug!(p, xx_gates(p); opts = o8)
    q = domain_wall_state(L); cbe_lubich_bug!(q, xxz_chain(L; J = 1.0, delta = 0.0); opts = o8)
    return nothing
end

"""
One row. `work` is `sum(krylov_dims .* mean_aug)`, the iterations-times-size product that
wall time should track if the exponential is where the time goes; printed in thousands.
"""
function row(label, run, err)
    t = @elapsed info = run()
    kry = sum(info.krylov_dims) / length(info.krylov_dims)
    aug = sum(info.mean_aug) / length(info.mean_aug)
    work = sum(info.krylov_dims .* info.mean_aug) / 1e3
    @printf("%-26s %10s %8d %8.1f %9.1f %9.1f %9.1f %9.1e\n",
            label, err === nothing ? "-" : @sprintf("%.3e", err()),
            maximum(info.max_bond_dims), t, kry, aug, work, maximum(info.err_fnl))
    return info
end

# ── phase `why`: iterations vs size, and the maxiter falsifier ────────────────
"""
    why [rev]

`rev` runs the rows in REVERSE order, and it is not cosmetic: the starved setting is measured
first, so ANY decaying background load -- another job finishing, a sync, thermal ramp -- lands
disproportionately on it and manufactures precisely the speedup being measured. MEASURED the
hard way: a first pass of this table came out ~10x slower than the same settings in
`cbe_bond_update_budget.jl` (gate growth=1.1 at 281.6s against 24.4s) because an orphaned L=18 job
was still on the CPU. A ratio that survives the order swap is the method; one that does not is
the machine.
"""
function phase_why(rev::Bool = false)
    L, dt, nsteps, maxdim = 10, 0.02, 25, 64
    tend = dt * nsteps
    exact = xx_free_fermion_sz(L, tend)
    warmup(L)                              # BEFORE any timing; see `warmup`
    println("# why: XX L=$L dt=$dt t=$tend maxdim=$maxdim  (analytic reference)")
    println("# row order: ", rev ? "REVERSED (order-swap control)" : "forward")
    println()
    println(HDR); println(repeat("-", 105))

    gate_rows = (("bond_update growth=1.1", (; growth = 1.1)),
                 ("bond_update growth=2.0", (; growth = 2.0)),
                 ("bond_update dex=2", (; dex = 2)),
                 ("bond_update dex=8", (; dex = 8)),
                 # THE FALSIFIER: if the starved run is Krylov-bound, capping maxiter
                 # must cut its time hard and barely touch the generous one.
                 ("bond_update growth=1.1 maxiter=8", (; growth = 1.1, maxiter = 8)),
                 ("bond_update growth=2.0 maxiter=8", (; growth = 2.0, maxiter = 8)))
    mpo_rows = (("lubich growth=1.1", (; growth = 1.1)),
                ("lubich growth=2.0", (; growth = 2.0)),
                ("lubich growth=1.1 maxiter=8", (; growth = 1.1, maxiter = 8)),
                ("lubich growth=2.0 maxiter=8", (; growth = 2.0, maxiter = 8)))
    order(rows) = rev ? reverse(rows) : rows

    for (lbl, kw) in order(gate_rows)
        p = domain_wall_state(L)
        o = CBEBugOptions(; dt, n_steps = nsteps, maxdim, trunc_thresh = 1e-14, kw...)
        row(lbl, () -> cbe_bond_update_bug!(p, xx_gates(p); opts = o),
            () -> maximum(abs.(prof(p) .- exact)))
    end
    println()
    h = xxz_chain(L; J = 1.0, delta = 0.0)
    for (lbl, kw) in order(mpo_rows)
        p = domain_wall_state(L)
        o = CBEBugOptions(; dt, n_steps = nsteps, maxdim, trunc_thresh = 1e-14, kw...)
        row(lbl, () -> cbe_lubich_bug!(p, h; opts = o),
            () -> maximum(abs.(prof(p) .- exact)))
    end
    println("\nWHY_DONE")
end

# ── phase `model`: is the effect specific to XX free fermions? ────────────────
function phase_model()
    L, dt, nsteps, maxdim = 10, 0.02, 25, 64
    warmup(L)                              # BEFORE any timing; see `warmup`
    println("# model: Heisenberg delta=1.0 (interacting), L=$L dt=$dt maxdim=$maxdim")
    println("# no analytic reference -- COST AND RANK only; total Sz is the invariant.\n")
    println(HDR); println(repeat("-", 105))
    for (lbl, kw) in (("bond_update heis growth=1.1", (; growth = 1.1)),
                      ("bond_update heis growth=2.0", (; growth = 2.0)),
                      ("bond_update heis dex=8", (; dex = 8)))
        p = domain_wall_state(L)
        o = CBEBugOptions(; dt, n_steps = nsteps, maxdim, trunc_thresh = 1e-14, kw...)
        row(lbl, () -> cbe_bond_update_bug!(p, heis_gates(p); opts = o), nothing)
        q = copy(p)
        @printf("    total Sz = %+.3e  (must be 0 for this wall)\n", total_sz(q) / norm(q)^2)
    end
    println()
    h = xxz_chain(L; J = 1.0, delta = 1.0)
    for (lbl, kw) in (("lubich heis growth=1.1", (; growth = 1.1)),
                      ("lubich heis growth=2.0", (; growth = 2.0)))
        p = domain_wall_state(L)
        o = CBEBugOptions(; dt, n_steps = nsteps, maxdim, trunc_thresh = 1e-14, kw...)
        row(lbl, () -> cbe_lubich_bug!(p, h; opts = o), nothing)
        q = copy(p)
        @printf("    total Sz = %+.3e  (must be 0 for this wall)\n", total_sz(q) / norm(q)^2)
    end
    println("\nMODEL_DONE")
end

# ── phase `chi`: does it scale, and where is the blow-up? ─────────────────────
"""
    chi [L] [tmax] [caps]        default 12, 1.5, "8,16,32"

THE CAP MUST ACTUALLY BIND or this measures nothing: at L=10, t=0.5 the state peaks near
chi=14, so a sweep over 16..64 gives identical runs and the "scaling" is pure noise. The
default here is the smallest window where a melting wall outgrows the caps -- L=12 to t=1.5,
caps 8/16/32 -- which keeps it a LOCAL-SIZED run (see docs/current_work.md, which runs go
where).

READ THE EXPONENT AS INDICATIVE, NOT ASYMPTOTIC. At chi = 8..32 the L-dependent constants --
canonicalisation, environment builds, sector bookkeeping -- are a large share of the time, so
a fitted `k` will UNDERSTATE the asymptotic exponent. What a small-size run can settle is
whether the two schedules differ in EXPONENT or only in CONSTANT, and whether anything
super-cubic appears. The asymptotic number needs the cluster: `chi 18 3.0 32,64,128`.
"""
function phase_chi(L::Int = 12, tmax::Float64 = 1.5, caps = (8, 16, 32))
    dt = 0.05
    nsteps = round(Int, tmax / dt)
    tend = dt * nsteps
    exact = xx_free_fermion_sz(L, tend)
    warmup(L)                              # BEFORE any timing; see `warmup`
    println("# chi: XX L=$L dt=$dt t=$tend, caps $(caps) binding, growth=2.0 vs 1.1\n")
    println(HDR); println(repeat("-", 105))
    h = xxz_chain(L; J = 1.0, delta = 0.0)
    times = Dict{Tuple{String, Int}, Float64}()
    for maxdim in caps
        for (tag, kw) in (("growth=1.1", (; growth = 1.1)), ("growth=2.0", (; growth = 2.0)))
            p = domain_wall_state(L)
            o = CBEBugOptions(; dt, n_steps = nsteps, maxdim, trunc_thresh = 1e-14, kw...)
            t0 = time()
            row(@sprintf("lubich chi<=%-3d %s", maxdim, tag),
                () -> cbe_lubich_bug!(p, h; opts = o),
                () -> maximum(abs.(prof(p) .- exact)))
            times[(tag, maxdim)] = time() - t0
        end
    end
    # The exponent, from consecutive doublings: t ~ chi^k  =>  k = log2(t(2chi)/t(chi)).
    println("\n# fitted exponent per doubling of the cap (t ~ chi^k); k ~ 3 predicted,")
    println("# and UNDERSTATED at these chi -- constants dominate. Compare the two schedules.")
    for tag in ("growth=1.1", "growth=2.0")
        for (a, b) in zip(caps[1:(end - 1)], caps[2:end])
            k = log2(times[(tag, b)] / times[(tag, a)]) / log2(b / a)
            @printf("%-12s chi %3d -> %3d :  k = %5.2f\n", tag, a, b, k)
        end
    end
    println("\nCHI_DONE")
end

PHASE in ("why", "all") && phase_why(length(ARGS) >= 2 && ARGS[2] == "rev")
PHASE in ("model", "all") && phase_model()
PHASE in ("chi", "all") && phase_chi(
    length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12,
    length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.5,
    length(ARGS) >= 4 ? Tuple(parse.(Int, split(ARGS[4], ","))) : (8, 16, 32))
PHASE in ("why", "model", "chi", "all") ||
    error("unknown phase $PHASE (why | model | chi | all)")
