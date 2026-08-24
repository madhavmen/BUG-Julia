# WHERE DOES THE ERROR ACTUALLY COME FROM? -- the Heisenberg tolerance study.
#
#     H = J Σ_ℓ S_ℓ · S_{ℓ+1}   (XXZ, Δ = 1, open chain),   |Ψ(0)⟩ = |↑↓↑↓…⟩,   L = 20, dt = 0.05
#
#   julia --project=. benchmarks/heisenberg_tolerance.jl phase=tau      # 0a/0b  err vs τ_trunc
#   julia --project=. benchmarks/heisenberg_tolerance.jl phase=dt       # 0c/0d  err vs dt
#   julia --project=. benchmarks/heisenberg_tolerance.jl phase=split    # 6A     root off, split scanned
#   julia --project=. benchmarks/heisenberg_tolerance.jl phase=grid     # 6B     root-out x half-sweep
#   python3 benchmarks/plot_heisenberg_tolerance.py
#
# Full plan and the reasoning behind each phase: docs/PLAN-tolerance-control.md
#
# WHY THIS MODEL, WHEN EVERY REAL-TIME CAMPAIGN SO FAR IS XX. The XX chain is free fermions --
# quadratic under Jordan-Wigner -- so its rank stays modest and basis-mechanism choices do not
# bind. `examples/common.jl` records the consequence at length: mechanisms worth five orders on
# TFIM are BIT-IDENTICAL on XX. XX is a control, not evidence. Δ = 1 turns the interaction on:
# the Néel state's entanglement grows linearly in t, so every scheme is rank-limited within a few
# 1/J and the comparison is about which subspace the truncation keeps.
#
# THE REFERENCE IS EXACT AND IT IS NOT ANALYTIC. Bethe solves the XXX SPECTRUM; it gives no closed
# form for `exp(-iHt)|Néel⟩`, so the standing "Heisenberg vs the analytic reference" rule -- which
# governs GROUND-STATE energies -- has nothing to offer a quench. What it does have is
# `exact_sparse.jl`: the Sz = 0 sector at L = 20 is C(20,10) = 184,756 states, H is sparse, and
# `expv_sparse` Krylov-propagates the vector with an a-posteriori residual check. Exact to solver
# tolerance, not a better-converged MPS, and it costs seconds.
#
# ⛔ THERE ARE TWO TRUNCATIONS IN A `cbe_bug_step!`, NOT THREE, AND THAT IS WHAT MAKES THIS
# MEASURABLE. The root core SVD used to cut as well, unconditionally, at `max(trunc_thresh,1e-14)`
# with `Nkeep = maxdim` -- so `truncate = false` did not give an untruncated root and no tolerance
# statement about the step was complete. It now only SPLITS (`root_cutoff = 0.0`,
# `root_maxdim = 0`). The two that remain are:
#
#     #1  half-sweep split     `split_cutoff`   applied once per bond as the sweep passes
#     #3  closing recursion    `trunc_thresh`   `truncate_recursive!`, root-to-leaves, guarded
#                              + `maxdim`        by `truncate`
#
# Phase `split` turns #3 off and scans #1, so #1 is the only rank control anywhere in the step.

using LinearAlgebra, Printf

include(joinpath(@__DIR__, "..", "examples", "common.jl"))   # parse_params, krylov_depth, steppers
include(joinpath(@__DIR__, "exact_sparse.jl"))               # heisenberg_sparse, neel_vector, expv_sparse

const P = parse_params((
    phase        = "tau",
    L            = 20,
    t_max        = 2.5,      # ≈50 steps at dt=0.05. Néel entanglement grows LINEARLY in t and
                             # the `split` phase runs with no closing truncation, so its ranks
                             # are the largest in the study -- extend only once χ(t) is known.
    dt           = 0.05,
    J            = 1.0,
    maxdim       = 0,        # 0 = UNCAPPED. A cap would floor the τ_trunc scan at the cap's own
                             # error and hide the knee this study exists to find.
    taus         = "1e-4,1e-5,1e-6,1e-7,1e-8,1e-10,1e-12,1e-14",
    dts          = "0.2,0.1,0.05,0.025,0.0125",
    dt_taus      = "1e-4,1e-6,1e-8,1e-10",
    splits       = "1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10",
    # Phase `grid`'s half-sweep axis. Deliberately the SAME ladder as `taus`, so the matrix is
    # square and the diagonal (both thresholds equal) is a readable line through it -- that is
    # the "one tolerance governs the whole step" configuration the plan is aiming at.
    grid_splits  = "1e-4,1e-5,1e-6,1e-7,1e-8,1e-10,1e-12,1e-14",
    # Which half-sweep structures to run the matrix for. See `SWEEPS`.
    grid_sweeps  = "m0,m3,mdef",
    # Which half-sweep truncation the matrix's x-axis is: "split" (the SVD that splits the
    # stacked Krylov frame) or "cbe" (the SVDs inside cbe_expand). Both always present.
    grid_axis    = "split",
    schemes      = "cbe_bug,tdvp_cbe1s,tdvp2",
    sample_every = 0.25,
    symmetry     = "U1",
    maxiter      = 0,        # 0 = derive the Krylov depth from ‖H‖·dt/2
    exact_cbe    = true,     # full SVD, not the randomised sketch (Jan's point 3)
))

const OUTDIR = joinpath(@__DIR__, "results")

# ‖H_XXZ‖ ≤ Σ_bonds ‖S·S‖ = (L-1)·(3/4)·J. The extremal many-body eigenvalues are ~0.44·J·L, so
# this over-estimates by ~2x -- which costs Krylov depth and never accuracy, the right direction
# to be wrong in.
const HNORM = 0.75 * P.J * (P.L - 1)

parse_list(s, T) = T[parse(T, strip(x)) for x in split(s, ',') if !isempty(strip(x))]
parse_syms(s) = String[strip(x) for x in split(s, ',') if !isempty(strip(x))]
depth_for(dt) = P.maxiter == 0 ? krylov_depth(HNORM, dt) : P.maxiter
# `maxdim = 0` means uncapped. Telum takes an Int, and the local space bounds the rank anyway, so
# a number larger than any reachable bond is the same thing and needs no branch downstream.
const CAP = P.maxdim > 0 ? P.maxdim : 1 << (P.L ÷ 2)

# ── the exact reference ───────────────────────────────────────────────────────────────────

"""
Exact `⟨S^z_j(t)⟩` on the sample grid, by sparse Krylov in the `Sz = 0` sector.

Propagating in `sample_every` jumps rather than in `dt` jumps keeps the reference INDEPENDENT of
the integrator's step size. A reference that stepped with `dt` would share the schemes' time grid
and could conceal a dt-dependent error rather than measure it -- which matters here because
phase `dt` varies exactly that.
"""
function exact_profiles(L::Int, J::Float64, t_max::Float64, sample_every::Float64)
    H, states, idx = heisenberg_sparse(L; J = J, delta = 1.0)
    v = neel_vector(L, idx)
    nsamp = round(Int, t_max / sample_every)
    profs = Vector{Vector{Float64}}(undef, nsamp + 1)
    profs[1] = magnetisation_dense(v, L, states)
    for k in 1:nsamp
        v = expv_sparse(H, ComplexF64(-im * sample_every), v; m = 30, tol = 1e-12)
        profs[k + 1] = magnetisation_dense(v, L, states)
    end
    return [k * sample_every for k in 0:nsamp], profs
end

"""
The scalar signal: staggered magnetisation `(1/L) Σ_j (-1)^(j+1) ⟨S^z_j⟩`.

STAGGERED rather than uniform because `S^z_tot` is CONSERVED -- the uniform sum is identically 0
for every scheme at every time and would grade nothing at all. A sum over all sites rather than
one site, so it is not accidentally sensitive to where the front happens to be at a sample time.
Starts at 1/2 for the Néel state and decays as the order melts.
"""
staggered(p::Vector{Float64}) = sum((-1)^(j + 1) * p[j] for j in eachindex(p)) / length(p)

# ── the run ───────────────────────────────────────────────────────────────────────────────

set_symmetry!(Symbol(P.symmetry))
const W    = xxz_mpo(P.L; J = P.J, delta = 1.0)
const PSI0 = neel_state(P.L)

"""
`⟨S^z_j⟩` for every site, NORMALISED.

`magnetisation` returns the unnormalised `⟨ψ|S^z_j|ψ⟩`, and every scheme sheds norm as its
truncation discards weight. Without the division that norm loss is reported as magnetisation
error, and unequally between schemes since they discard different weight. The raw norm goes in
its own column so the loss stays visible rather than hidden by this.
"""
sz_profile(psi) = (p = copy(psi); magnetisation(p) ./ max(norm(p)^2, eps()))

# `COLS_T`, not `COLS`: `examples/common.jl` is included above and already owns that name.
const COLS_T = ["phase", "scheme", "L", "dt", "tau_trunc", "split_cutoff", "cbe_cutoff",
                "root_trunc", "close_trunc", "maxdim", "t", "stag", "stag_exact", "err_stag",
                "err_prof", "maxbond", "norm", "energy", "dE", "err_fnl", "discarded",
                "krylov", "seconds"]

# THE SWEEP HAS ONE STRUCTURE; ONLY THE KRYLOV DEPTH VARIES. `kstep`, `kaug` and `rexpand` were
# removed on 2026-08-24 -- the basis-only sweep at `krylov_basis = 3` beat the K-step machinery on
# every model measured (see `docs/PLAN-tolerance-control.md`). What is left is one knob:
#
#   m0     `krylov_basis = 0`    the bare CBE frame -- ONE power of H. All XX needs; cheapest.
#   m2     `krylov_basis = 2`    the first genuine addition (`m = 1` reproduces `m = 0` exactly,
#                                because CBE's frame already spans one power).
#   m3     `krylov_basis = 3`    a PINNED depth. Heisenberg and OAT both want at least this.
#   mdef   package defaults      `krylov_basis = 30` (a CAP) with `krylov_tol = 1e-6`.
#
# ⛔ `m3` IS NOT THE PACKAGE DEFAULT, and conflating them is how a campaign ends up reporting a
# configuration nobody runs. The shipped default is the CAP 30 plus the `1e-6` breakdown
# tolerance, and because `beta` does not decay on a generic `H` the cap is usually what binds --
# so `mdef` is a DEEPER and more expensive arm than `m3`, not the same one. Measured on
# Heisenberg L=12: 9.61e-07 @ 5580 applications for the defaults against 2.67e-06 @ 1038 at a
# pinned depth 3. Both belong in the matrix; only one of them is what a user gets.
#
# ⚠ A `"""docstring"""` on a `const X = Dict(...)` in an included script is a LOAD ERROR, and it
# fails AFTER the package precompile with EXIT CODE 0 -- so it reads as a completed run that
# wrote no data. Keep this a comment.
const SWEEPS = Dict(
    "m0"   => (krylov_basis = 0,),
    "m2"   => (krylov_basis = 2,),
    "m3"   => (krylov_basis = 3,),
    # empty = pass nothing, so this arm tracks the package defaults even if they change.
    "mdef" => NamedTuple())

"""
    stepper(scheme, tau_trunc, split_cutoff, close, dt; cbe_cut) -> step!

One integrator, configured for one arm.

⛔ ONLY `cbe_bug` HAS `split_cutoff` / `truncate` / `root_*`. `tdvp2` and `tdvp_cbe1s` prune
continuously at every site by construction and expose one tolerance, so the `split` and `grid`
phases are cbe_bug-only BY CONSTRUCTION -- there is nothing on the other two to A/B. They appear
in the `tau` and `dt` phases, where `trunc_thresh` means the same thing for all three.

`root_cutoff` / `root_maxdim` are passed EXPLICITLY at their no-truncation defaults rather than
inherited, so an arm's truncation configuration is legible here and cannot drift with the
package default.
"""
function stepper(scheme::String, tau_trunc::Float64, split_cutoff::Float64,
                 close::Bool, dt::Float64; cbe_cut::Float64 = 0.0)
    m = depth_for(dt)
    if startswith(scheme, "cbe_bug")
        # A bare "cbe_bug" is the PACKAGE DEFAULTS, not a pinned depth -- so the scheme named
        # after the package reports what the package actually does.
        name = scheme == "cbe_bug" ? "mdef" : scheme[length("cbe_bug_") + 1:end]
        haskey(SWEEPS, name) || error("unknown sweep $(repr(name)); have $(keys(SWEEPS))")
        s = SWEEPS[name]
        # `cbe_cut <= 0` keeps the package defaults, so the split-axis runs are unchanged and
        # comparable with everything measured before this argument existed.
        sp = cbe_cut > 0 ? cbe_cut : 1e-10
        sf = cbe_cut > 0 ? cbe_cut : 1e-13
        return (p, tau) -> cbe_bug_step!(p, W, tau;
                                         s...,          # empty for the `mdef` arm
                                         exact = P.exact_cbe,
                                         stol_pre = sp, stol_fnl = sf,
                                         split_cutoff = split_cutoff, split_maxdim = 0,
                                         root_cutoff = 0.0, root_maxdim = 0,
                                         truncate = close,
                                         maxdim = CAP, trunc_thresh = tau_trunc, maxiter = m)
    elseif scheme == "tdvp_cbe1s"
        return (p, tau) -> tdvp_cbe1s_step!(p, W, tau; maxdim = CAP,
                                            trunc_thresh = tau_trunc, maxiter = m)
    elseif scheme == "tdvp2"
        return (p, tau) -> tdvp2_step!(p, W, tau; maxdim = CAP,
                                       trunc_thresh = tau_trunc, maxiter = m)
    end
    error("unknown scheme $(repr(scheme))")
end

"""
One arm, streamed to `io` a row per sample.

`err_stag` is the scalar signal's error and `err_prof` the L-infinity error over the whole
`⟨S^z_j⟩` profile. Both are kept because they can disagree: the staggered sum can cancel a
site-resolved error the profile norm still sees.

⚠ `seconds` is LOCAL wall clock, orientation only -- local timing on this machine spreads 2.6x on
bit-identical work. `krylov` (operator applications) is the cost axis that survives contention.
"""
function run_arm(io, phase, scheme, tau_trunc, split_cutoff, close, dt, profs;
                 cbe_cut::Float64 = 0.0)
    step!  = stepper(scheme, tau_trunc, split_cutoff, close, dt; cbe_cut = cbe_cut)
    psi    = copy(PSI0)
    nsteps = round(Int, P.t_max / dt)
    every  = max(1, round(Int, P.sample_every / dt))
    e0     = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    kry, secs, efnl, disc = 0, 0.0, 0.0, 0.0

    emit(n, t) = begin
        prof = sz_profile(psi)
        s, e = staggered(prof), real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
        @printf(io, "%s,%s,%d,%g,%g,%g,%g,%d,%d,%d,%g,%.10g,%.10g,%.6e,%.6e,%d,%.10g,%.10g,%.6e,%.3e,%.3e,%d,%.2f\n",
                phase, scheme, P.L, dt, tau_trunc, split_cutoff, cbe_cut, 0, close ? 1 : 0,
                P.maxdim, t, s, staggered(profs[n]), abs(s - staggered(profs[n])),
                maximum(abs.(prof .- profs[n])), maximum(bond_dims(psi)), norm(psi),
                e, abs(e - e0), efnl, disc, kry, secs)
        flush(io)
    end
    emit(1, 0.0)

    for k in 1:nsteps
        t0   = time_ns()
        info = step!(psi, ComplexF64(-im * dt))
        secs += (time_ns() - t0) / 1e9
        kry  += info.krylov_dims
        hasproperty(info, :err_fnl)   && (efnl = max(efnl, info.err_fnl))
        hasproperty(info, :discarded) && (disc = max(disc, info.discarded))
        k % every == 0 && emit(k ÷ every + 1, k * dt)
    end

    prof = sz_profile(psi)
    @printf("  %-11s dt=%-7g tau=%-8g split=%-8g close=%-5s | err %.3e  prof %.3e  chi %-4d  kry %-7d  %.0fs\n",
            scheme, dt, tau_trunc, split_cutoff, close, abs(staggered(prof) - staggered(profs[end])),
            maximum(abs.(prof .- profs[end])), maximum(bond_dims(psi)), kry, secs)
    flush(stdout)
end

open_out(name) = begin
    mkpath(OUTDIR)
    io = open(joinpath(OUTDIR, name), "w")
    println(io, join(COLS_T, ","))
    io
end

"""
⚠ `t = 0` IS A GATE, NOT A PRINTOUT, and it is cheap insurance against the two silent failures
this comparison is prone to.

`exact_sparse.jl` indexes site `j` at bit `j-1` with a SET bit meaning UP; `dense_reference.jl`
mirrors both conventions. The Néel state is a FIXED POINT of the transformation that
distinguishes them, so this checks the VALUE, not the convention -- but a wrong Néel phase, a
mis-normalised observable or a `delta` slip all land here rather than looking like a small
integrator error twenty steps later. The energy check is exact: every Néel bond is antiparallel,
so `⟨S·S⟩ = -1/4` per bond and `E₀ = -(L-1)/4` with no tolerance in it.
"""
function preflight(profs)
    s0, sr = staggered(sz_profile(copy(PSI0))), staggered(profs[1])
    @printf("t=0 staggered: mps %.12f  exact %.12f  (diff %.2e)\n", s0, sr, abs(s0 - sr))
    abs(s0 - sr) < 1e-12 || error("t=0 mismatch -- the MPS and the reference are not the same state")
    e0, ref = real(mpo_energy(copy(PSI0), W)) / norm(copy(PSI0))^2, -(P.L - 1) / 4
    @printf("t=0 energy:    %.12f  exact -(L-1)/4 = %.12f  (diff %.2e)\n\n", e0, ref, abs(e0 - ref))
    abs(e0 - ref) < 1e-10 || error("t=0 energy mismatch -- the MPO is not this Hamiltonian")
end

function main()
    @printf("Heisenberg XXZ Δ=1, Néel quench -- L=%d  %s  J=%g  T=%g  dt=%g  maxdim=%s\n",
            P.L, P.symmetry, P.J, P.t_max, P.dt, P.maxdim > 0 ? string(P.maxdim) : "UNCAPPED")
    @printf("phase=%s  exact_cbe=%s  krylov depth=%d (bound %.1e)\n",
            P.phase, P.exact_cbe, depth_for(P.dt), krylov_bound(HNORM, P.dt, depth_for(P.dt)))
    print("building the exact reference (C($(P.L),$(P.L÷2)) states) ... "); flush(stdout)
    t0 = time_ns()
    _, profs = exact_profiles(P.L, P.J, P.t_max, P.sample_every)
    @printf("%d samples, %.1f s\n", length(profs), (time_ns() - t0) / 1e9)
    preflight(profs)

    schemes = parse_syms(P.schemes)
    if P.phase == "tau"
        # 0a / 0b. Is there a knee, and where? Closing truncation ON at τ_trunc, half-sweep split
        # left at its shipped 1e-14 -- i.e. today's configuration, with the one knob swept.
        io = open_out(@sprintf("heis_tau_L%d_dt%g_T%g.csv", P.L, P.dt, P.t_max))
        try
            for tau in parse_list(P.taus, Float64), s in schemes
                run_arm(io, "tau", s, tau, 1e-14, true, P.dt, profs)
            end
        finally; close(io); end

    elseif P.phase == "dt"
        # 0c / 0d. The calibration curve τ_trunc*(dt): where each τ_trunc's line leaves the
        # common envelope is where truncation takes over from time integration.
        io = open_out(@sprintf("heis_dt_L%d_T%g.csv", P.L, P.t_max))
        try
            for dt in parse_list(P.dts, Float64), tau in parse_list(P.dt_taus, Float64), s in schemes
                run_arm(io, "dt", s, tau, 1e-14, true, dt, profs)
            end
        finally; close(io); end

    elseif P.phase == "split"
        # 6A. Closing truncation OFF, root split already non-truncating -> `split_cutoff` is the
        # ONLY rank control in the step. Loosest cutoff FIRST: with nothing closing the step the
        # rank can only ratchet up, so the first arm is what tells us whether T and the cap are
        # affordable for the rest.
        io = open_out(@sprintf("heis_split_L%d_dt%g_T%g.csv", P.L, P.dt, P.t_max))
        try
            for sc in parse_list(P.splits, Float64)
                run_arm(io, "split", "cbe_bug", 0.0, sc, false, P.dt, profs)
            end
        finally; close(io); end

    elseif P.phase == "grid"
        # 6B. THE FULL 2D SCAN: every root-out threshold of phase `tau` crossed with every
        # half-sweep cutoff. This is the experiment that separates the two remaining truncations,
        # and a 2x2 could not do it -- with only two values per axis, "the error follows the row"
        # and "the error follows whichever is looser" are the same picture.
        #
        # WHAT EACH OUTCOME MEANS, decided before the run so the heatmap is read and not
        # rationalised:
        #
        #   error varies down COLUMNS only  -> the ROOT-OUT pass governs; the half-sweeps are
        #                                      inert at their shipped 1e-14 and `split_cutoff`
        #                                      is a knob with nothing behind it.
        #   error varies across ROWS only   -> the HALF-SWEEPS govern; the closing pass is only
        #                                      cleaning up after them.
        #   error follows max(row, col)     -> the LOOSER of the two sets the accuracy, i.e. they
        #                                      are redundant and one of them can go.
        #   the two combine                 -> they cut different things, and both thresholds have
        #                                      to be quoted for any error claim to be reproducible.
        #
        # `truncate = true` throughout: this sweeps the closing pass's THRESHOLD, it does not turn
        # it off. Phase `split` is the arm where it is off.
        # One CSV PER SWEEP, not one shared file. These runs are long and are launched
        # separately; a shared filename means the second launch truncates the first's data,
        # which is how the default arm's grid nearly went missing once already.
        for sw in parse_syms(P.grid_sweeps)
            sch = "cbe_bug_$sw"
            # WHICH HALF-SWEEP TRUNCATION THE x-AXIS IS, and it is NOT the same knob for every
            # sweep -- picking the wrong one scans a variable the sweep never reads.
            #
            #   grid_axis = "split"  ->  `split_cutoff`, the SVD that orthonormalises the stacked
            #                            Krylov blocks into `W[i]` / `Z[j]`. Once per bond.
            #   grid_axis = "cbe"    ->  `stol_pre = stol_fnl`, the SVDs INSIDE `cbe_expand`.
            #
            # BOTH are live in the current sweep -- the retired K-step is what used to make the
            # first one conditional. They sit at different points of the same half-sweep, so a
            # scan of one with the other left at its default is measuring a floor, not the knob.
            axis = P.grid_axis
            axis in ("split", "cbe") || error("grid_axis must be split or cbe, got $(repr(axis))")
            @printf("=== sweep %s  (%s)  x-axis = %s ===\n", sw, sch, axis); flush(stdout)
            io = open_out(@sprintf("heis_grid%s_%s_L%d_dt%g_T%g.csv",
                                   axis == "cbe" ? "cbe" : "", sw, P.L, P.dt, P.t_max))
            try
                for tau in parse_list(P.taus, Float64), x in parse_list(P.grid_splits, Float64)
                    @printf("  -> arm tau=%g %s=%g ...\n", tau, axis, x); flush(stdout)
                    if axis == "split"
                        run_arm(io, "grid", sch, tau, x, true, P.dt, profs)
                    else
                        run_arm(io, "grid", sch, tau, 1e-14, true, P.dt, profs; cbe_cut = x)
                    end
                end
            finally; close(io); end
        end
    else
        error("unknown phase $(repr(P.phase)) -- use tau, dt, split or grid")
    end
    println("\nwrote to ", OUTDIR)
end

main()
