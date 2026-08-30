# MAKING `rsvd_cbe_dmrg1s!` FAST -- THE GROUND-STATE KNOB STUDY, SCORED AGAINST EXACT ED.
#
# WHY THIS EXISTS. The campaign's ground state was the bottleneck: a 5x4 square cylinder at
# D=128 ran past 45 minutes without finishing its first solve, where the 20-site CHAIN at the same
# cap took 22 seconds. Some of that is the operator (MPO virtual dim 14 against the chain's 5) and
# the small U(1) blocks, but not a factor of a hundred.
#
# ⛔ THE PRIME SUSPECT IS A DEFAULT INHERITED FROM THE WRONG ALGORITHM, and `cbe_core.jl` says so
# in its own words: "THE CONSTANT IS OURS, NOT THE REFERENCE'S: `growth = 2.0`, not the
# reference's `1.1`. The reference is DMRG, where each sweep re-converges the same state, so a 10%
# ratchet compounds over sweeps and costs only iterations. A TIME INTEGRATOR gets one shot per
# step." `growth = 2.0` was chosen FOR THE TIME INTEGRATOR -- and `dmrg_cbe1s_sweep!` inherits it,
# because `dex = 0` hands the budget to that schedule:
#
#     budget = ceil(growth * dmax) - r        (cbe_core.jl:631)
#     npre   = ceil(1.2 * budget)             the sketch's WIDTH
#
# At `chi = 128` with `dmax = r = 128` that is `budget = 128`, a 154-column sketch, and a bond
# expanded to 256 before the truncation cuts it back to 128. With `dex = 8, dover = 4` it is a
# 12-column sketch and a bond of 136. The DMRG can afford the 1.1 ratchet precisely because it
# re-converges; the time integrator cannot.
#
# ⚠ AND THE EXISTING MEASUREMENT DOES NOT SETTLE IT. The table at `cbe_core.jl:617` ("wall time is
# flat in the budget, 4.4-4.7s from dex=1 to dex=64, because the sweep dominates the sketch") was
# taken at `maxdim = 64` on an L=10 CHAIN for a TIME STEP. Whether the sketch still costs nothing
# when the budget DOUBLES THE BOND at chi=128 on a cylinder is exactly what is unmeasured, so this
# file measures it rather than assuming either way.
#
# ⛔ `err_fnl` IS THE COLUMN THAT SAYS WHETHER THE BUDGET WAS THE LIMITER, and it is why a cheap
# row is not automatically a good one. It is the ranked candidate weight the expansion DISCARDED
# for want of budget: `cbe_core.jl:623` records `err_fnl = 0.62` for the 1.1 schedule, i.e. "the
# cap, not the selection, was the accuracy limiter". A row with small `err_fnl` and low cost is a
# genuine win; a row with small cost and `err_fnl` near 1 is starving.
#
# ⛔ SWEEP 1 IS NOT A COST DATUM -- it carries Julia's compilation of the whole sweep stack (order
# 1e2 s). `DMRGCBERun.sweep_seconds` documents this; `sec_per_sweep` below therefore averages
# `sweep_seconds[2:end]`, and a warm-up solve runs before any row is timed.
#
# Run:  julia --project=. benchmarks/ground_state/dmrg_knobs.jl [chain|square|triangle]
# Env:  DK_LX (5) DK_LY (4) DK_D (128) DK_SWEEPS (12) DK_ETOL (1e-10) DK_J2 (0.0)
#       DK_ROWS  comma list of knob names to scan (default all); use "baseline,dex" to calibrate
# Out:  benchmarks/results/dmrg_knobs_<geom>_<Lx>x<Ly>_D<D>.csv

using LinearAlgebra, SparseArrays, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
# The 2-site CBE-DMRG, needed for the reference's `Nkeep <= D_CBE` bootstrap stage.
using BUGJulia.RSVDCBEBondUpdate.GroundState

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))

const GEOM   = let a = filter(x -> x in ("chain", "square", "triangle"), ARGS)
    isempty(a) ? "square" : a[1]
end
const LX     = parse(Int,     get(ENV, "DK_LX",     "5"))
const LY     = parse(Int,     get(ENV, "DK_LY",     "4"))
const DCAP   = parse(Int,     get(ENV, "DK_D",      "128"))
const NSW    = parse(Int,     get(ENV, "DK_SWEEPS", "12"))
const ETOL   = parse(Float64, get(ENV, "DK_ETOL",   "1e-10"))
const J2     = parse(Float64, get(ENV, "DK_J2",     "0.0"))
const ROWS   = let s = get(ENV, "DK_ROWS", "")
    isempty(s) ? nothing : Set(String.(split(s, ',')))
end
const N      = LX * LY
const OUT    = joinpath(@__DIR__, "..", "results")

"""
    BASE

The campaign's settings (`benchmarks/spectral/bug_knobs.jl` `RSVD_KNOBS`), used here as the
one-factor-at-a-time baseline. `dex = 8` is an ABSOLUTE per-side budget and overrides the growth
schedule entirely; `growth` is carried only so the `dex = 0` rows have something to vary.
"""
const BASE = (dex = 8, dover = 4, comp_ratio = 1.0, growth = 2.0,
              maxiter = 30, restol = 1e-10, exact = false)

function couplings()
    GEOM == "square"   && return square_cylinder_couplings(LX, LY; periodic_y = true)
    # :xc is arXiv:2209.00739's wrapping (Sec. IIB) and a DIFFERENT Hamiltonian from :yc.
    GEOM == "triangle" && return triangular_cylinder_couplings(LX, LY; J2 = J2,
                                                               geometry = :xc, periodic_y = true)
    Jm = zeros(N, N)
    for i in 1:(N - 1)
        Jm[i, i + 1] = 1.0
    end
    return Jm
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE ARMS. Single-stage rows vary one knob; the multi-stage rows below implement what the
# MATLAB reference actually does and we do not.
#
# ⛔ `mainDMRG_CBE.m` DIFFERS FROM OUR DRIVER IN THREE WAYS AT ONCE, and each is a cost decision:
#
#   :81  `Nkeep = DList(isw)`          the bond dimension is RAMPED over sweeps, not fixed.
#   :82  `Dex = ceil(rDex*Nkeep)`      the expansion budget is a FRACTION OF THE CAP -- neither
#                                      our absolute `dex` nor our `growth*dmax` schedule.
#   :64  `D_CBE = round(DList(end)/2)` CBE is switched ON ONLY once `Nkeep > Dmax/2`; below that
#   :109 `else ... 2-site DMRG`        the reference runs plain 2-SITE DMRG.
#
# The third is the one that matters most for us: from a Neel product state the early sweeps have
# tiny rank, where the sketch's fixed overhead is the whole cost and a 2-site update grows rank
# for free. We have a 2-site CBE-DMRG already (`GroundState.solve!`), so the bootstrap costs no
# new machinery -- only the driver-level decision to use it.
# ══════════════════════════════════════════════════════════════════════════════════════════════

"""
    Cost

Accumulated over the stages of one arm, so a ramped or bootstrapped run is directly comparable
with a single-stage one.

⚠ `sec_per_sweep` USES ALL SWEEPS, INCLUDING THE FIRST. That is correct ONLY because `main` runs
a warm-up solve before any row: with compilation already paid, dropping each row's opening sweep
would discard real work -- and would flatter the multi-stage arms most, since they have one
opening sweep per stage.
"""
mutable struct Cost
    sec::Float64
    sweeps::Int
    matvec::Int
    maxexp::Int
    efnl::Float64
    conv::Bool
end
Cost() = Cost(0.0, 0, 0, 0, 0.0, false)

"Fold a `DMRGCBERun` (1-site) into the accumulator."
function absorb!(c::Cost, run::DMRGCBERun, sec::Float64)
    c.sec += sec
    c.sweeps += length(run.energies)
    c.matvec += sum(run.krylov_dims)
    c.maxexp = max(c.maxexp, maximum(run.max_expanded; init = 0))
    c.efnl = max(c.efnl, maximum(run.err_fnl; init = 0.0))
    c.conv = run.converged
    return c
end

"Fold a `GroundStateInfo` (the 2-site solver) in. It reports no `err_fnl`, so that column stays
whatever the 1-site stages set -- a bootstrap's `err_fnl` is a statement about its CBE stages."
function absorb!(c::Cost, info::GroundStateInfo, sec::Float64)
    c.sec += sec
    c.sweeps += length(info.energies)
    c.matvec += sum(info.krylov_dims)
    c.maxexp = max(c.maxexp, maximum(info.max_expanded; init = 0))
    c.conv = info.converged
    return c
end

"One 1-site rSVD-CBE stage, in place on `psi`."
function stage_1s!(c::Cost, psi, W, D::Int, nsw::Int, nt)
    t0 = time()
    run = dmrg_cbe1s!(psi, W; n_sweeps = nsw, etol = ETOL, maxdim = D,
                      trunc_thresh = 1e-12, nt...)
    absorb!(c, run, time() - t0)
end

"One 2-site CBE-DMRG stage (`GroundState.solve!`), in place on `psi`."
function stage_2s!(c::Cost, psi, W, D::Int, nsw::Int)
    opts = CBEBugOptions(; maxdim = D, trunc_thresh = 1e-12, dex = BASE.dex,
                         dover = BASE.dover, comp_ratio = BASE.comp_ratio,
                         maxiter = BASE.maxiter, exact = false)
    t0 = time()
    info = GroundState.solve!(psi, W; opts = opts, n_sweeps = nsw, etol = ETOL,
                              grow_iters = 1)
    absorb!(c, info, time() - t0)
end

"Finish an arm: recompute the energy from the state and package the study's columns."
function finish(psi, W, c::Cost, E0x)
    E = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    return (E = E, sec = c.sec, sweeps = c.sweeps,
            sps = c.sweeps > 0 ? c.sec / c.sweeps : c.sec,
            matvec = c.matvec, chi = maximum(state_bond_dims(psi); init = 0),
            maxexp = c.maxexp, efnl = c.efnl, conv = c.conv)
end

"A single-stage solve at the full cap -- the one-factor-at-a-time rows."
function run_plain(psi0, W, E0x, nt)
    psi = copy(psi0); c = Cost()
    stage_1s!(c, psi, W, DCAP, NSW, nt)
    return finish(psi, W, c, E0x)
end

"""
    run_ramp(psi0, W, E0x, nt; ladder, per)

The reference's `DList`: a few sweeps at each of a rising sequence of caps, converging at the
last. Cheap early sweeps do the coarse work at low rank; only the final stage pays full width.
"""
function run_ramp(psi0, W, E0x, nt; ladder = [DCAP ÷ 4, DCAP ÷ 2, DCAP], per = 3)
    psi = copy(psi0); c = Cost()
    for (k, D) in enumerate(ladder)
        stage_1s!(c, psi, W, D, k == length(ladder) ? NSW : per, nt)
    end
    return finish(psi, W, c, E0x)
end

"""
    run_boot2s(psi0, W, E0x, nt; dboot, per)

The reference's `Nkeep > D_CBE` switch: plain 2-site DMRG while the rank is below `Dmax/2`, then
1-site rSVD-CBE the rest of the way. From a product state the early sweeps are where the sketch
is pure overhead and the 2-site block grows rank for nothing.
"""
function run_boot2s(psi0, W, E0x, nt; dboot = max(4, DCAP ÷ 2), per = 4)
    psi = copy(psi0); c = Cost()
    stage_2s!(c, psi, W, dboot, per)
    stage_1s!(c, psi, W, DCAP, NSW, nt)
    return finish(psi, W, c, E0x)
end

"The reference's recipe with all three ingredients: 2-site bootstrap, ramp, `dex` a fraction of
the cap (`Dex = ceil(rDex*Nkeep)`, `mainDMRG_CBE.m:82`)."
function run_reference(psi0, W, E0x, nt; rdex = 0.1, per = 3)
    psi = copy(psi0); c = Cost()
    stage_2s!(c, psi, W, max(4, DCAP ÷ 4), per)
    for (k, D) in enumerate([DCAP ÷ 2, DCAP])
        stage_1s!(c, psi, W, D, k == 2 ? NSW : per,
                  merge(nt, (dex = max(1, ceil(Int, rdex * D)),)))
    end
    return finish(psi, W, c, E0x)
end

"The scan: (label, knob, value, runner). One field of `BASE` moves per single-stage row; the
`recipe` rows change the SHAPE of the solve instead and are labelled as such."
function rows()
    out = Any[]
    add(lbl, knob, val, nt) = push!(out, (lbl, knob, string(val),
                                          (p, W, E) -> run_plain(p, W, E, nt)))
    addf(lbl, knob, val, f) = push!(out, (lbl, knob, string(val), f))

    add("baseline", "baseline", "-", BASE)

    # ── the reference's own shape, which our driver does not use ────────────────────────────
    addf("ramp", "recipe", "DList", (p, W, E) -> run_ramp(p, W, E, BASE))
    addf("boot2s", "recipe", "D_CBE", (p, W, E) -> run_boot2s(p, W, E, BASE))
    for r in (0.05, 0.1, 0.2)
        addf("ref_rdex$r", "recipe", "rDex=$r",
             (p, W, E) -> run_reference(p, W, E, BASE; rdex = r))
    end

    # ⛔ THE SUSPECT. `dex = 0` hands the budget to the growth schedule; everything else is an
    # absolute per-side budget. This axis is the whole reason the file exists.
    for v in (0, 4, 8, 16, 32)
        add("dex$v", "dex", v, merge(BASE, (dex = v,)))
    end
    # Only meaningful with `dex = 0`. `1.1` is the MATLAB reference's DMRG ratchet.
    for v in (1.1, 1.5, 2.0)
        add("growth$v", "growth", v, merge(BASE, (dex = 0, growth = v)))
    end
    for v in (0, 2, 4, 8)
        add("dover$v", "dover", v, merge(BASE, (dover = v,)))
    end
    for v in (0.25, 0.5, 1.0)
        add("comp$v", "comp_ratio", v, merge(BASE, (comp_ratio = v,)))
    end
    # ⚠ THE LOCAL EIGENSOLVE. `maxiter` is the reference's `K`, `restol` its `Ktol`. A DMRG sweep
    # does NOT need its local solve converged to 1e-10 -- the sweep re-solves the same site next
    # pass, so an over-tight local tolerance buys iterations that the next sweep discards.
    for v in (10, 20, 30)
        add("maxiter$v", "maxiter", v, merge(BASE, (maxiter = v,)))
    end
    for v in (1e-6, 1e-8, 1e-10)
        add("restol$v", "restol", v, merge(BASE, (restol = v,)))
    end
    # The A/B reference arm: the full fused basis instead of the sketch. Not a production setting.
    add("exactcbe", "exact", true, merge(BASE, (exact = true,)))

    ROWS === nothing ? out : filter(r -> r[2] in ROWS, out)
end

function main()
    mkpath(OUT)
    set_symmetry!(:U1)
    Jm = couplings()
    W  = pair_mpo(N, Jm, spin_vertices(N))

    @printf("\n%s\nGROUND-STATE KNOB STUDY  %s  N=%d  %s  D=%d  n_sweeps<=%d  etol=%.0e\n",
            "="^116, uppercase(GEOM), N,
            GEOM == "chain" ? "open chain" : "$(LX)x$(LY) cylinder", DCAP, NSW, ETOL)
    @printf("MPO virtual dim max %d;  full rank at N=%d is %d\n%s\n",
            maximum(mpo_virtual_dims(W)), N, 2^(N ÷ 2), "="^116)
    flush(stdout)

    # ── exact reference ─────────────────────────────────────────────────────────────────────
    t0 = time()
    H, states, = pairs_sparse(N, Jm)
    E0x, _, resid, = sparse_ground(H, length(states))
    resid < 1e-9 || error("Lanczos residual $resid -- reference not converged")
    @printf("exact E0=%.12f  E0/N=%.12f  resid=%.2e  %.1fs\n\n", E0x, E0x / N, resid,
            time() - t0)
    flush(stdout)

    psi0 = neel_state(N)

    # ⛔ WARM-UP BEFORE ANY TIMED ROW. Without it the first row absorbs the compilation of the
    # entire sweep stack and reads as the slowest setting in the table, whatever it is.
    print("warming up (2 sweeps, not recorded) ... "); flush(stdout)
    tw = @elapsed dmrg_cbe1s!(copy(psi0), W; n_sweeps = 2, etol = ETOL, maxdim = DCAP,
                              trunc_thresh = 1e-12, BASE...)
    @printf("%.1fs\n\n", tw); flush(stdout)

    out = joinpath(OUT, "dmrg_knobs_$(GEOM)_$(LX)x$(LY)_D$(DCAP).csv")
    @printf("  %-12s %-11s %-8s %12s %6s %5s %8s %9s %6s %6s %10s\n",
            "arm", "knob", "value", "dev/site", "chi", "swp", "s/sweep", "seconds",
            "maxexp", "matvec", "err_fnl")
    flush(stdout)
    open(out, "w") do io
        # ⚠ NO `spread` COLUMN. `sweep_spread` (the reference's `err_swp`) exists only on
        # `DMRGCBERun`; `GroundStateInfo` from the 2-site bootstrap has no such field, so a
        # multi-stage arm could not fill it and the column would be silently blank for exactly
        # the rows the study is about.
        println(io, "geom,N,Lx,Ly,D,arm,knob,value,E,E_site,E_exact,dev,dev_site,chi," *
                    "sweeps,sec_per_sweep,seconds,maxexp,matvec,err_fnl,converged")
        for (lbl, knob, val, runner) in rows()
            r = runner(psi0, W, E0x)
            dev = r.E - E0x
            @printf("  %-12s %-11s %-8s %12.3e %6d %5d %8.2f %9.1f %6d %6d %10.2e%s\n",
                    lbl, knob, val, dev / N, r.chi, r.sweeps, r.sps, r.sec, r.maxexp,
                    r.matvec, r.efnl, r.conv ? "" : "  (hit sweep cap)")
            flush(stdout)
            @printf(io, "%s,%d,%d,%d,%d,%s,%s,%s,%.12f,%.12f,%.12f,%.6e,%.6e,%d,%d,%.4f,%.3f,%d,%d,%.6e,%d\n",
                    GEOM, N, LX, LY, DCAP, lbl, knob, val, r.E, r.E / N, E0x, dev, dev / N,
                    r.chi, r.sweeps, r.sps, r.sec, r.maxexp, r.matvec, r.efnl,
                    r.conv ? 1 : 0)
            flush(io)
        end
    end
    println("\nwrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
