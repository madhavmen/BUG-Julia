# WHICH KNOBS BUY WHAT, MEASURED AGAINST THE EXACT CORRELATOR AT N <= 20.
#
# Two studies, and they answer DIFFERENT questions:
#
#   A  BUG's own knobs        m / beta / stol_pre / split_cutoff / root_cutoff / parallel
#                             -> accuracy of G(x,t) AND wall clock
#   B  the rSVD sketch knobs  dex / dover / comp_ratio / growth / exact-vs-sketch
#                             -> COMPUTATION TIME (the sketch is not supposed to change the
#                                answer; a row where it does is a defect, not a trade-off)
#
# ⛔ EVERY ROW IS SCORED AGAINST THE EXACT `G(x,t)`, NOT AGAINST ANOTHER MPS RUN. `sparse_krylov.jl`
# propagates `S^z_c |Om>` in the 184756-state sector exactly, so `err` here is error, not spread.
# This is the property that was missing at 12 sites, where the spectral comparison could only
# report that the three integrators agreed with each other to 1e-11 -- true, and silent about
# whether all three were wrong together.
#
# ⛔ ONE GROUND STATE, SHARED BY EVERY ROW OF A GEOMETRY. The knobs under study belong to the TIME
# STEP; recomputing `|Om>` per row would let a knob that changes the ground state masquerade as a
# knob that changes the evolution. The shared state is converged at `K20_DGS` (default 512, which
# the gate shows is far above what these geometries need) and is checked against the exact vector
# at `t = 0` BEFORE any dynamics -- if `G(x,0)` disagrees, the two paths are not looking at the
# same state and no dynamical row below is meaningful.
#
# ⛔ ONE-FACTOR-AT-A-TIME FROM A DECLARED BASELINE. The baseline row is emitted first and every
# other row changes exactly one field of it. A scan that moves two knobs reports a difference it
# cannot attribute -- which is how `grow_iters = 0` vs `grow_iters = 2` was once read as a growth
# measurement when it had also silently moved `krylov_basis` from 2 to 30.
#
# ⚠ WALL CLOCK ON A SHARED BOX SPREADS. Run this with nothing else competing, and read `seconds`
# only against rows from the SAME invocation. `matvec` is reproducible but INCOMPLETE for BUG:
# `krylov_dims` counts `apply_one_site` and is blind to `cbe_expand`'s sketch work, so it
# understates every BUG row. Both columns are emitted; only wall clock is a cost claim.
#
# Run:  julia -t 2 --project=. benchmarks/spectral/knobs_l20.jl [A|B] <chain|square|triangle> ...
# Env:  K20_LX (5) K20_LY (4) K20_D (128) K20_DGS (512) K20_TMAX (4.0) K20_DT (0.1) K20_J2 (0.0)
# Out:  benchmarks/results/knobs_l20_<study>_<geoms>.csv

using LinearAlgebra, SparseArrays, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))       # pairs_sparse, sz0_basis
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))      # sparse_ground, exact_correlator
include(joinpath(@__DIR__, "lattice2d.jl"))
include(joinpath(@__DIR__, "bug_knobs.jl"))

const STUDY = let i = findfirst(a -> a in ("A", "B", "C"), ARGS); i === nothing ? "A" : ARGS[i] end
const WANT  = let a = filter(x -> x in ("chain", "square", "triangle"), ARGS)
    isempty(a) ? ["square"] : a
end
const LX   = parse(Int,     get(ENV, "K20_LX",   "5"))
const LY   = parse(Int,     get(ENV, "K20_LY",   "4"))
const DCAP = parse(Int,     get(ENV, "K20_D",    "128"))
const DGS  = parse(Int,     get(ENV, "K20_DGS",  "512"))
# ⛔ `TMAX = 8` IS NOT AN ARBITRARY SHORT WINDOW -- IT IS WHERE THE SPECTRAL WEIGHT LIVES. The
# figure runs to `Tmax = 40`, but Eq. (12) multiplies the record by `exp(-eta^2 t^2)`, and at
# `eta^2 = 0.03` that is `0.147` by `t = 8` and `0.013` by `t = 12`. So a knob ranked on `t <= 8`
# is ranked on the part of the trajectory the spectrum is actually built from, at a fifth of the
# cost of ranking it on the full window.
#
# ⚠ WHAT THIS WINDOW CANNOT SEE is a knob whose error GROWS with time -- a slow drift that is
# invisible at `t = 8` and dominant at `t = 40`. The production run at chi=512 therefore still
# reports its own agreement against the other arms; the scan chooses the setting, it does not
# certify the trajectory.
const TMAX = parse(Float64, get(ENV, "K20_TMAX", "8.0"))
const DT   = parse(Float64, get(ENV, "K20_DT",   "0.1"))
const J2   = parse(Float64, get(ENV, "K20_J2",   "0.0"))
const N    = LX * LY
const OUT  = joinpath(@__DIR__, "..", "results")

function couplings(geom::String)
    geom == "square"   && return square_cylinder_couplings(LX, LY; periodic_y = true)
    # :xc is the paper's wrapping (Sec. IIB) and a DIFFERENT Hamiltonian from :yc -- see
    # `lattice2d.jl`. A knob tuned on one must not be loaded for the other; `geom_key` keys on
    # Lx x Ly, so keep this in step with the driver that consumes `bug_optimal_<geom>.json`.
    geom == "triangle" && return triangular_cylinder_couplings(LX, LY; J2 = J2,
                                                               geometry = :xc, periodic_y = true)
    Jm = zeros(N, N)
    for i in 1:(N - 1)
        Jm[i, i + 1] = 1.0
    end
    return Jm
end

"Centre site: the 2D centre for a cylinder, the chain centre for a chain."
centre(geom) = geom == "chain" ? cld(N, 2) : centre_site(LX, LY)

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE ARMS
# ══════════════════════════════════════════════════════════════════════════════════════════════

"""
    mps_correlator(psi0, W, E0, step!, c, nt, dt) -> (G, matvec, seconds, chi)

`G[i,n] = e^{i E0 t_n} <Om| S^z_i |phi(t_n)>` from an MPS arm, in the SAME convention as
`exact_correlator` -- same phase removal, same stepping, same centre.
"""
function mps_correlator(psi0::SymMPS, E0::Float64, step!, c::Int, nt::Int, dt::Float64)
    L = length(psi0)
    phi = deepcopy(psi0)
    apply_sz!(phi, c)
    bras = map(1:L) do i
        b = deepcopy(psi0); apply_sz!(b, i); b
    end
    G = zeros(ComplexF64, L, nt + 1)
    mv = 0; sec = 0.0; chi = 0
    for i in 1:L
        G[i, 1] = overlap(bras[i], phi)
    end
    for n in 1:nt
        t0 = time()
        info = step!(phi, ComplexF64(-im * dt))
        sec += time() - t0
        hasproperty(info, :krylov_dims) ||
            error("integrator info has no krylov_dims: $(typeof(info))")
        kd = info.krylov_dims
        mv += kd isa Number ? Int(kd) : sum(kd)
        chi = max(chi, maximum(bond_dims(phi)))
        ph = cis(E0 * n * dt)
        for i in 1:L
            G[i, n + 1] = ph * overlap(bras[i], phi)
        end
    end
    return G, mv, sec, chi
end

"Largest deviation of an arm's correlator from the exact one, over every site and every time."
gerr(G, Gx) = maximum(abs.(G .- Gx))

# ══════════════════════════════════════════════════════════════════════════════════════════════
# STUDY C -- THE THREE-KNOB CUBE, ON THE 2D GEOMETRIES
# ══════════════════════════════════════════════════════════════════════════════════════════════
#
# The same cube `benchmarks/su2_knob_cube.jl` runs on the chain -- half-sweep `split_cutoff` x
# root-out `root_cutoff` x Krylov depth `m` -- but on the square and triangular cylinders and
# scored against the EXACT `G(x,t)` rather than a bond-energy profile.
#
# ⛔ WHY A CUBE AND NOT THREE LINE SCANS (this is study A's blind spot, not a nicety). A
# one-factor-at-a-time scan holds the other two knobs at values that may THEMSELVES be the binding
# constraint, in which case it measures the floor and reports "this knob does nothing". That is
# exactly how the Krylov depth stayed invisible on the chain: every truncation scan ran at a fixed
# depth that was already the limiter. Study A is a cheap map of one axis; study C is the one that
# can say WHICH axis governs.
#
# Read the heatmaps by where the gradient points -- decided before looking:
#   varies along x only     the HALF-SWEEP governs, the closing pass is cleaning up after it
#   varies along y only     the ROOT-OUT governs, `split_cutoff` has nothing behind it
#   varies along max(x,y)   the two are REDUNDANT: the looser one sets the accuracy
#   flat in both, moves    NEITHER truncation governs -- the KRYLOV DEPTH is the floor
#     between panels
#   genuinely 2D            they cut different things; both must be quoted for any claim
#
# ⚠ THE CAMPAIGN BASELINE SITS INSIDE THIS GRID at `split_cutoff = 1e-6`, `root_cutoff = 1e-8`
# (`BUG_TOL` in `bug_knobs.jl`), so the cube says whether that point is on a plateau or on a cliff.
const CUBE_SPLIT = [1e-4, 1e-6, 1e-8, 1e-10]
const CUBE_ROOT  = [1e-4, 1e-6, 1e-8, 1e-10]
# `mdef` is the package default (cap 30 + tol 1e-6) and is deliberately NOT called "m=30": the
# tolerance usually stops it well short of the cap.
const CUBE_DEPTH = ["m2" => 2, "m3" => 3, "m4" => 4, "mdef" => 0]

"""
    study_C(W, kw) -> Vector{(sweepname, split_cutoff, root_cutoff, stepfactory)}

Every point of the cube. `bug_step_kwargs(0)` is the adaptive-beta/package-default depth, which is
what `mdef` means here and in `plot_knob_cube.py`'s label.
"""
function study_C(W, kw)
    rows = Any[]
    for (nm, m) in CUBE_DEPTH, sc in CUBE_SPLIT, rc in CUBE_ROOT
        # ⛔ `merge`, NOT splat-then-override: `bug_step_kwargs` ALREADY carries both cutoffs, and
        # passing either again is a duplicate keyword argument.
        kv = merge(bug_step_kwargs(m), (split_cutoff = sc, root_cutoff = rc))
        push!(rows, (nm, sc, rc, (p, tau) -> cbe_bug_step!(p, W, tau; kw..., kv...)))
    end
    return rows
end

"""
    cube_tag(geom) -> String

The `PRE` that `plot_knob_cube.py` takes as its third argument. It carries the GEOMETRY AND THE
EVOLUTION CAP, because a cube at D=64 and one at D=128 are different measurements and pooling them
into one panel would compare arms that never saw the same problem.
"""
cube_tag(geom::String) = geom == "chain" ? "ch$(N)D$(DCAP)" :
    (geom == "square" ? "sq" : "tri") * "$(LX)x$(LY)D$(DCAP)"

# ══════════════════════════════════════════════════════════════════════════════════════════════
# STUDY A -- BUG'S OWN KNOBS
# ══════════════════════════════════════════════════════════════════════════════════════════════

"""
    study_A(W) -> Vector{(label, knob, value, stepfactory)}

One-factor-at-a-time around the campaign baseline `bug_step_kwargs(3)`.

The two TDVP arms are included as reference rows, not as knob rows: the campaign's claim is about
BUG's cost against them on the same correlator, so they belong in the same table at the same cap.
"""
function study_A(W, kw)
    base = bug_step_kwargs(3)
    rows = Any[]
    push!(rows, ("tdvp2", "reference", "-", (p, tau) -> tdvp2_step!(p, W, tau; kw...)))
    push!(rows, ("cbe1s", "reference", "-", (p, tau) -> tdvp_cbe1s_step!(p, W, tau; kw...)))
    push!(rows, (bug_label(3), "baseline", "-",
                 (p, tau) -> cbe_bug_step!(p, W, tau; kw..., base...)))

    # ⛔ m = 1 REPRODUCES m = 0 EXACTLY (cbe_expand already spans one power of H), so the depth
    # axis starts at 2. m = 0 here means the ADAPTIVE beta rule, not "no Krylov".
    for m in (2, 4, 6, 8)
        push!(rows, (bug_label(m), "m", string(m),
                     (p, tau) -> cbe_bug_step!(p, W, tau; kw..., bug_step_kwargs(m)...)))
    end
    push!(rows, (bug_label(0), "krylov_tol(beta)", "1e-6",
                 (p, tau) -> cbe_bug_step!(p, W, tau; kw..., bug_step_kwargs(0)...)))

    # ⛔ THE GROWTH PATH, WITH ITS MATCHED-WIDTH CONTROL ALREADY IN THIS TABLE. `grow_iters = n`
    # resolves to `krylov_grow = true, krylov_basis = n` (see `cbe_bug_step!`), so the ONLY
    # legitimate control for it is `krylov_grow = false, krylov_basis = n` -- which is exactly the
    # `m = n` row above. Read `gi2` against `m2` and `gi4` against `m4`; reading either against
    # the m=3 baseline changes width and growth at once.
    #
    # ⛔ `grow_iters = 1` GROWS NOTHING (`_grow_frame` breaks on `k == krylov_basis` before the
    # re-expansion), so the axis starts at 2. A scan whose growth axis starts at 1 reports its
    # first point as "growth on" when `n_grow` was 0.
    for gi in (2, 4)
        kv = merge(base, (grow_iters = gi,))
        push!(rows, ("bug_gi$gi", "grow_iters", string(gi),
                     (p, tau) -> cbe_bug_step!(p, W, tau; kw..., kv...)))
    end

    # ⛔ `merge`, NOT `base..., stol_pre = v`. Splatting a NamedTuple that ALREADY carries the
    # keyword and then passing it again is a duplicate keyword argument -- whether Julia takes the
    # first, the last, or refuses is not something a knob scan should be relying on. `merge`
    # states the override in one object and the call site has each keyword exactly once.
    for v in (1e-1, 1e-2, 1e-3, 1e-4)
        kv = merge(base, (stol_pre = v,))
        push!(rows, ("bug_stolpre$(v)", "stol_pre", string(v),
                     (p, tau) -> cbe_bug_step!(p, W, tau; kw..., kv...)))
    end
    for v in (1e-4, 1e-6, 1e-8, 1e-10)
        kv = merge(base, (split_cutoff = v,))
        push!(rows, ("bug_split$(v)", "split_cutoff", string(v),
                     (p, tau) -> cbe_bug_step!(p, W, tau; kw..., kv...)))
    end
    for v in (0.0, 1e-8, 1e-6, 1e-4)
        kv = merge(base, (root_cutoff = v,))
        push!(rows, ("bug_root$(v)", "root_cutoff", string(v),
                     (p, tau) -> cbe_bug_step!(p, W, tau; kw..., kv...)))
    end
    # ⚠ NEEDS `julia -t 2` TO MEAN ANYTHING. On one thread the two half-sweeps interleave on one
    # worker and `parallel = true` is the sequential step plus scheduling overhead.
    push!(rows, (bug_label(3; parallel = false), "parallel", "false",
                 (p, tau) -> cbe_bug_step!(p, W, tau; kw...,
                                           bug_step_kwargs(3; parallel = false)...)))
    return rows
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# STUDY B -- THE rSVD SKETCH KNOBS
# ══════════════════════════════════════════════════════════════════════════════════════════════

"""
    study_B(W, kw) -> rows

The sketch geometry. `exact = true` is the A/B REFERENCE ARM: it forms the complete fused basis
(cost `d*chi`, precisely what the sketch exists to avoid), so it is the answer to "did the sketch
cost accuracy", never a production setting.

⛔ `dover = 0` ON AN rSVD ARM REPRODUCES THE FROZEN-START SIGNATURE -- converged at chi = 1 in two
sweeps -- and is included so the scan SHOWS that rather than leaving it as folklore. Read its
`err` column, not its `seconds`.
"""
function study_B(W, kw)
    rows = Any[]
    mk(lbl, knob, val, nt) = push!(rows, (lbl, knob, val,
        (p, tau) -> cbe_bug_step!(p, W, tau; kw..., nt...)))

    mk(bug_label(3), "baseline", "-", bug_step_kwargs(3))
    mk("bug_exactcbe", "exact", "true", bug_step_kwargs(3; rsvd = false))
    for v in (4, 8, 16, 32)
        mk("bug_dex$v", "dex", string(v),
           merge(bug_step_kwargs(3), (dex = v,)))
    end
    for v in (0, 2, 4, 8)
        mk("bug_dover$v", "dover", string(v),
           merge(bug_step_kwargs(3), (dover = v,)))
    end
    # ⛔ `comp_ratio` IS VALIDATED TO [0,1] OR `nothing` (cbe_core.jl). A `2.0` row here did not
    # measure an aggressive split -- it THREW, and took the rest of the scan with it. `nothing` is
    # the real third mode: no split at all, probe drawn from the full randomised fusion basis.
    for v in (0.25, 0.5, 1.0, nothing)
        mk("bug_comp$(v === nothing ? "none" : v)", "comp_ratio",
           v === nothing ? "nothing" : string(v),
           merge(bug_step_kwargs(3), (comp_ratio = v,)))
    end

    # ⛔⛔ `growth` IS DEAD CODE UNLESS `dex == 0`, AND THE BASELINE SETS `dex = 8`.
    #     `budget = dex <= 0 ? max(ceil(growth*dmax) - r, 1) : dex`   (cbe_core.jl)
    # So a growth scan run on top of `bug_step_kwargs(3)` produces four rows that are BIT-IDENTICAL
    # to the baseline and reports "growth does nothing" for every geometry. Each growth row must
    # therefore ALSO set `dex = 0`, and it needs `dex = 0` itself in the table as the matched
    # control -- otherwise the growth rows differ from the baseline in two knobs at once.
    # (Same failure as `grow_iters` never running: an unexercised knob is an unimplemented one.)
    mk("bug_dex0", "dex", "0 (growth schedule)",
       merge(bug_step_kwargs(3), (dex = 0,)))
    for v in (1.2, 1.5, 2.0, 3.0)
        mk("bug_growth$v", "growth", string(v),
           merge(bug_step_kwargs(3), (dex = 0, growth = v,)))
    end

    # ⚠ `dex` IS ABSOLUTE AND `growth` IS PROPORTIONAL, which is the whole point of running this
    # at BOTH caps. A `dex` that wins at chi=64 has no claim at chi=128; a `growth` that wins does.
    # Read the two caps' tables together, not separately.
    return rows
end

# ══════════════════════════════════════════════════════════════════════════════════════════════

function run_geometry(io, geom::String)
    # ⛔ SYMMETRY FIRST, THEN THE OPERATOR. `spin_vertices`/`pair_mpo` read the CURRENT local
    # space, so building the MPO before `set_symmetry!` builds it in whatever mode the process
    # happened to be in -- and the mismatch shows up much later, as a state that cannot contract
    # against its own Hamiltonian.
    set_symmetry!(:U1)
    Jm = couplings(geom)
    c  = centre(geom)
    W  = pair_mpo(N, Jm, spin_vertices(N))
    nt = round(Int, TMAX / DT)
    ts = collect(0:nt) .* DT

    @printf("\n%s\n%s  N=%d  %s  centre=%d  D=%d (gs D=%d)  T=%.1f dt=%.2f (%d steps)\n%s\n",
            "="^112, uppercase(geom), N,
            geom == "chain" ? "open chain" : "$(LX)x$(LY) cylinder", c, DCAP, DGS,
            TMAX, DT, nt, "="^112)

    # ── the exact reference ─────────────────────────────────────────────────────────────────
    t0 = time()
    H, states, = pairs_sparse(N, Jm)
    E0x, Om, resid, = sparse_ground(H, length(states))
    resid < 1e-9 || error("Lanczos residual $resid -- reference not converged")
    Gx, apps = exact_correlator(H, states, Om, E0x, c, N, ts)
    @printf("exact: E0=%.12f  E0/N=%.12f  resid=%.2e  %d expv applications  %.1fs\n",
            E0x, E0x / N, resid, apps, time() - t0)

    # ── the shared MPS ground state ─────────────────────────────────────────────────────────
    psi0 = neel_state(N)
    tgs = @elapsed gs = rsvd_cbe_dmrg1s!(psi0, W; n_sweeps = 60, etol = 1e-12, maxdim = DGS,
                                         trunc_thresh = 1e-12, maxiter = 30, restol = 1e-10,
                                         dex = RSVD_KNOBS.dex, dover = RSVD_KNOBS.dover,
                                         comp_ratio = RSVD_KNOBS.comp_ratio,
                                         growth = RSVD_KNOBS.growth)
    E0 = real(mpo_energy(copy(psi0), W)) / max(norm(psi0)^2, eps())
    @printf("MPS  : E0=%.12f  dev=%+.2e  chi=%d  sweeps=%d  %.1fs\n",
            E0, E0 - E0x, maximum(state_bond_dims(psi0)), length(gs.energies), tgs)

    # ⛔ THE STATIC GATE, BEFORE ANY DYNAMICS. `G(x,0) = <Om|S^z_x S^z_c|Om>` is computable both
    # ways with no time evolution at all, so a disagreement here is the STATE or the site
    # convention -- the mirrored-bit landmine in `exact_sparse.jl` -- and not the integrator.
    # Comparing trajectories without this check attributes a state mismatch to the time step.
    G0 = zeros(ComplexF64, N)
    let phi = deepcopy(psi0)
        apply_sz!(phi, c)
        for i in 1:N
            b = deepcopy(psi0); apply_sz!(b, i)
            G0[i] = overlap(b, phi)
        end
    end
    d0 = maximum(abs.(G0 .- Gx[:, 1]))
    @printf("static gate: max|G_mps(x,0) - G_exact(x,0)| = %.3e  %s\n\n",
            d0, d0 < 1e-8 ? "PASS" : "⛔ FAIL -- states or site conventions differ")
    flush(stdout)
    d0 < 1e-8 || error("static correlator mismatch $d0; no dynamical row below is meaningful")

    kw = (maxdim = DCAP, trunc_thresh = 1e-12, maxiter = 8)

    # ── STUDY C: the cube, written in `plot_knob_cube.py`'s own format ──────────────────────
    if STUDY == "C"
        pre = cube_tag(geom)
        @printf("  cube: %d split x %d root x %d depths = %d runs   PRE=%s\n",
                length(CUBE_SPLIT), length(CUBE_ROOT), length(CUBE_DEPTH),
                length(CUBE_SPLIT) * length(CUBE_ROOT) * length(CUBE_DEPTH), pre)
        @printf("  %-6s %10s %10s %11s %6s %9s %9s\n",
                "depth", "split", "root", "err", "chi", "matvec", "secs")
        flush(stdout)
        handles = Dict{String, IO}()
        try
            for (nm, sc, rc, mk) in study_C(W, kw)
                G, mv, sec, chi = mps_correlator(psi0, E0, mk, c, nt, DT)
                e = gerr(G, Gx)
                @printf("  %-6s %10.0e %10.0e %11.3e %6d %9d %9.2f\n", nm, sc, rc, e, chi, mv, sec)
                flush(stdout)
                # One file per depth: `heis_grid<PRE>_<sweep>_L<N>_dt<dt>_T<T>.csv`, which is the
                # name `plot_knob_cube.py` globs. Columns are its schema, not this driver's.
                io2 = get!(handles, nm) do
                    p = joinpath(OUT, @sprintf("heis_grid%s_%s_L%d_dt%g_T%g.csv",
                                               pre, nm, N, DT, TMAX))
                    h = open(p, "w")
                    println(h, "tau_trunc,split_cutoff,t,err_prof,maxbond,krylov,seconds")
                    h
                end
                # `tau_trunc` is the plotter's y axis (root-out) and `t` its "final sample" key;
                # `err_prof` is the max over ALL sites and times, so one row per grid point.
                @printf(io2, "%.6e,%.6e,%.4f,%.6e,%d,%d,%.4f\n", rc, sc, TMAX, e, chi, mv, sec)
                flush(io2)
            end
        finally
            foreach(close, values(handles))
        end
        println("\n  wrote cube CSVs; plot with:")
        println("    python3 benchmarks/plot_knob_cube.py benchmarks/results $N $pre")
        return
    end

    rows = STUDY == "A" ? study_A(W, kw) : study_B(W, kw)

    @printf("  %-22s %-18s %-8s %11s %6s %9s %9s\n",
            "arm", "knob", "value", "err", "chi", "matvec", "secs")
    for (lbl, knob, val, mk) in rows
        G, mv, sec, chi = mps_correlator(psi0, E0, mk, c, nt, DT)
        e = gerr(G, Gx)
        @printf("  %-22s %-18s %-8s %11.3e %6d %9d %9.2f\n", lbl, knob, val, e, chi, mv, sec)
        # ⛔ REDIRECTED STDOUT IS BLOCK-BUFFERED: without this a long scan prints NOTHING until it
        # exits, so "working or wedged?" can only be answered by watching CPU time.
        flush(stdout)
        # ⛔ THE `geom` COLUMN CARRIES THE CIRCUMFERENCE (`geom_key`), not just the lattice name.
        # `select_knobs.py` groups on this column and `fig45_2209.jl` looks the winner up by it,
        # so a bare "triangle" would let a C=3 tuning be loaded for a C=6 run.
        @printf(io, "%s,%s,%d,%d,%d,%s,%s,%s,%d,%.6e,%d,%d,%.4f,%.4f,%.3f\n",
                STUDY, geom_key(geom, LX, LY), N, LX, LY, lbl, knob, val, DCAP, e, chi, mv, sec,
                TMAX, DT)
        flush(io)
    end
end

function main()
    mkpath(OUT)
    # ⛔ THE EVOLUTION CAP IS PART OF THE NAME. `dex` is an ABSOLUTE budget and `growth` a
    # PROPORTIONAL one, so the same scan at chi=64 and chi=128 gives genuinely different winners --
    # which is the point of running both. Without `D` in the filename the second run overwrites the
    # first and the comparison silently becomes one cap measured twice.
    out = joinpath(OUT, "knobs_l$(N)_$(STUDY)_D$(DCAP)_$(join(WANT, '-')).csv")
    open(out, "w") do io
        println(io, "study,geom,N,Lx,Ly,arm,knob,value,D,err,chi,matvec,seconds,tmax,dt")
        for g in WANT
            run_geometry(io, g)
        end
    end
    println("\nwrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
