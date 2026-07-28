# XX domain wall, L=18, U(1), to t=15: CBE-BUG against 2-site TDVP.
#
#   sbatch --job-name=xx18_calib scripts/run_julia.sbatch benchmarks/xx_l18_campaign.jl calib
#   sbatch --job-name=xx18_main  scripts/run_julia.sbatch benchmarks/xx_l18_campaign.jl main
#
# THE REFERENCE IS EXACT, NOT A FINER SIMULATION. The XX chain is free fermions, so a
# domain wall's magnetisation profile is known in closed form -- `xx_free_fermion_sz`
# diagonalises an L x L single-particle matrix. At L=18 a dense many-body propagator would
# be 2^18 x 2^18 and impossible; this is exact and costs nothing, which is why the whole
# campaign is XX rather than XXZ.
#
# MATCHING THE ORDER. 2-site TDVP is second order by construction (symmetric sweep);
# CBE-BUG takes one projected exponential per step, so it has no splitting error to
# symmetrise and no order parameter to set. The two are therefore made comparable the only
# honest way: `calib` finds a dt at which BOTH are dt-converged (halving dt moves the error
# by a few percent at most), so the main run's differences are subspace and rank effects,
# not time-integration order. Reporting a rank comparison at a dt where one method is still
# dt-limited would be measuring the wrong thing.

using LinearAlgebra, Printf
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include("../tests/common/free_fermion.jl")

const L      = 18
const TMAX   = 15.0
const OUTDIR = joinpath(@__DIR__, "..", "results")


"Exact <Sz_j(t)> for the L=18 domain wall (free fermions)."
exact_profile(t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:(L ÷ 2))

"""
L-infinity error of an MPS magnetisation profile against the exact one.

DIVIDES BY THE NORM, and that is not cosmetic. `site_expval` returns the unnormalised
`<psi|Sz_j|psi>`, and BOTH integrators shed norm as their truncations discard weight -- by
t=15 at maxdim=32 that is a percent-level effect. Without the division the plot would
report norm loss as magnetisation error, and would do so unequally between the two methods
(they discard different weight), which is exactly the comparison being made. The raw norm
is recorded separately in the CSV so the loss stays visible rather than hidden by this.
"""
function profile_err(psi, t)
    p = copy(psi)
    n2 = norm(p)^2                     # before measuring: canonical! preserves it anyway
    return maximum(abs.(magnetisation(p) ./ n2 .- exact_profile(t)))
end

"""
One sample of one run.

`theta_elems` is the largest two-site block the step would need. CBE-BUG never builds one,
so it is 0 for it by construction -- see docs/cbe_bug.md section 2b.

`peak_elems` is the WORKING SET, and it is the number the memory claim stands or falls on:
`state + transients alive at that moment`, measured inside the step at a definition both
integrators share (`CBEBugInfo.peak_elements`, `TDVP2Info.peak_elements`). `state_elems`
alone would flatter CBE-BUG, whose basis sweep is transiently wider than what survives the
truncation, and `theta_elems` alone would flatter it further by being 0 -- the honest
comparison is the peak, reported here as the max over the sampling window.
"""
struct Row
    scheme::String
    maxdim::Int
    dex::Int
    dt::Float64
    t::Float64
    err::Float64
    maxbond::Int
    centrebond::Int
    theta_elems::Int
    state_elems::Int
    peak_elems::Int
    norm::Float64
    elapsed::Float64
end

"Total stored elements of the whole MPS -- the memory the state itself occupies."
state_elements(psi) = sum(tensor_elements(psi[i]) for i in 1:length(psi))

# ── the two runs ─────────────────────────────────────────────────────────────

function run_cbe(; dt, maxdim, dex, tmax, sample_every)
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    h = xxz_chain(L; delta = 0.0)
    rows = Row[]
    nsteps = round(Int, tmax / dt)
    c = L ÷ 2
    t0 = time()
    maxex = 0
    peak = 0
    for k in 1:nsteps
        info = cbe_bug_bond_update(psi, h, -im * dt; dex = dex, maxdim = maxdim,
                                   trunc_thresh = 1e-12)
        maxex = max(maxex, maximum(info.expanded; init = 0))
        peak = max(peak, info.peak_elements)
        if k % sample_every == 0 || k == nsteps
            t = k * dt
            bd = bond_dims(psi)
            push!(rows, Row("cbe", maxdim, dex, dt, t, profile_err(psi, t),
                            maximum(bd), bd[c], 0, state_elements(psi), peak,
                            norm(psi), time() - t0))
        end
    end
    return rows, maxex
end

function run_tdvp2(; dt, maxdim, tmax, sample_every)
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    h = xxz_chain(L; delta = 0.0)
    rows = Row[]
    nsteps = round(Int, tmax / dt)
    c = L ÷ 2
    t0 = time()
    theta = 0
    peak = 0
    for k in 1:nsteps
        info = tdvp2_step!(psi, h, -im * dt; maxdim = maxdim, trunc_thresh = 1e-12)
        theta = max(theta, info.theta_elements)
        peak = max(peak, info.peak_elements)
        if k % sample_every == 0 || k == nsteps
            t = k * dt
            bd = bond_dims(psi)
            push!(rows, Row("tdvp2", maxdim, 0, dt, t, profile_err(psi, t),
                            maximum(bd), bd[c], theta, state_elements(psi), peak,
                            norm(psi), time() - t0))
        end
    end
    return rows, theta
end

# ── phases ───────────────────────────────────────────────────────────────────

"""
Calibration: is the comparison dt-limited or rank-limited? Halve dt and watch the error.

    calib [maxdim] [tmax]        default 64 and 2.0

MEASURE IT AT THE MAXDIM THE MAIN RUN REPORTS. The default `maxdim = 64, tmax = 2.0` window
answers only "what is the time-integration error", and MEASURED it answers it cleanly: both
methods are exactly second order there (ratio 4.00 per halving, `err/dt^2` constant to three
digits), with `maxbond` peaking at 28 against a cap of 64 -- nothing truncates, so those
numbers carry no rank information at all and NEITHER method is dt-converged at any of
dt = 0.1, 0.05, 0.025.

That makes the default window the wrong place to pick the main run's dt from: at maxdim=32
out to t=15 the truncation is supposed to dominate, and the dt only has to be small enough
that it is not what is being measured. So run this twice --

    calib 64 2.0        # the order check: is each method O(dt^2)?
    calib 32 6.0        # the choice: at the reported cap, has the error stopped moving?

-- and pick dt from the SECOND. A dt whose halving barely moves the error there is dt-safe
for the main run; one that still halves the error four-fold is not.
"""
function phase_calib(maxdim::Int = 64, tmax::Float64 = 2.0)
    println("# dt calibration, L=$L, XX domain wall, t=$tmax, maxdim=$maxdim")
    @printf("%-8s %8s %12s %9s %9s\n", "scheme", "dt", "err(t=$tmax)", "maxbond", "norm")
    println(repeat("-", 52))
    for dt in (0.1, 0.05, 0.025)
        n = round(Int, tmax / dt)
        r, _ = run_cbe(dt = dt, maxdim = maxdim, dex = 0, tmax = tmax, sample_every = n)
        @printf("%-8s %8.4f %12.4e %9d %9.5f\n",
                "cbe", dt, r[end].err, r[end].maxbond, r[end].norm)
    end
    for dt in (0.1, 0.05, 0.025)
        n = round(Int, tmax / dt)
        r, _ = run_tdvp2(dt = dt, maxdim = maxdim, tmax = tmax, sample_every = n)
        @printf("%-8s %8.4f %12.4e %9d %9.5f\n",
                "tdvp2", dt, r[end].err, r[end].maxbond, r[end].norm)
    end
    println("\nCALIB_DONE")
end

"The A/B the investigation asked for: are the centre's augmented frames needed?"
function phase_centre()
    println("# centre-expansion A/B, L=$L, t=2.0, maxdim=64, dt=0.05")
    set_symmetry!(:U1)
    h = xxz_chain(L; delta = 0.0)
    for ce in (true, false)
        psi = domain_wall_state(L)
        for _ in 1:40
            cbe_bug_bond_update(psi, h, -im * 0.05; maxdim = 64, trunc_thresh = 1e-12,
                                centre_expand = ce)
        end
        @printf("centre_expand=%-5s  err=%.4e  maxbond=%d  centre=%d\n",
                ce, profile_err(psi, 2.0), maximum(bond_dims(psi)),
                bond_dims(psi)[L ÷ 2])
    end
    println("\nCENTRE_DONE")
end

function phase_main(dt)
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, "xx_l18_campaign.csv")
    sample_every = max(1, round(Int, 0.5 / dt))       # a sample every 0.5 time units
    open(path, "w") do io
        println(io, "scheme,maxdim,dex,dt,t,err,maxbond,centrebond,theta_elems," *
                    "state_elems,peak_elems,norm,elapsed")
        function emit(rows)
            for r in rows
                @printf(io, "%s,%d,%d,%g,%g,%.8e,%d,%d,%d,%d,%d,%.8f,%.2f\n",
                        r.scheme, r.maxdim, r.dex, r.dt, r.t, r.err, r.maxbond,
                        r.centrebond, r.theta_elems, r.state_elems, r.peak_elems,
                        r.norm, r.elapsed)
            end
            flush(io)
        end

        for maxdim in (32, 64)
            @printf("running tdvp2 maxdim=%d ...\n", maxdim); flush(stdout)
            rows, theta = run_tdvp2(dt = dt, maxdim = maxdim, tmax = TMAX,
                                    sample_every = sample_every)
            emit(rows)
            @printf("  done: err=%.3e maxbond=%d largest Theta=%d elems  %.1fs\n",
                    rows[end].err, maximum(r.maxbond for r in rows), theta,
                    rows[end].elapsed); flush(stdout)

            for dex in (2, 0)                        # 0 = the full Sulz budget r
                @printf("running cbe maxdim=%d dex=%s ...\n", maxdim,
                        dex == 0 ? "r" : string(dex)); flush(stdout)
                rows, maxex = run_cbe(dt = dt, maxdim = maxdim, dex = dex, tmax = TMAX,
                                      sample_every = sample_every)
                emit(rows)
                @printf("  done: err=%.3e maxbond=%d max expanded=%d  %.1fs\n",
                        rows[end].err, maximum(r.maxbond for r in rows), maxex,
                        rows[end].elapsed); flush(stdout)
            end
        end
    end
    println("wrote $path")
    println("\nMAIN_DONE")
end

# ── entry ────────────────────────────────────────────────────────────────────

const PHASE = isempty(ARGS) ? "calib" : ARGS[1]
if PHASE == "calib"
    phase_calib(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64,
                length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 2.0)
elseif PHASE == "centre"
    phase_centre()
elseif PHASE == "main"
    phase_main(length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.05)
else
    error("unknown phase $PHASE (calib | centre | main)")
end
