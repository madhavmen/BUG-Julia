# THE CAMPAIGN'S BUG AND rSVD SETTINGS, IN ONE PLACE.
#
# The physics drivers and the knob studies read THIS file, so a setting tuned in a scan is the
# setting the figure runs with. Two copies of a tolerance is how a campaign reports a knob study
# for one configuration and a figure for another.
#
# ══════════════════════════════════════════════════════════════════════════════════════════════
# ⛔ THE NAMES IN THE BRIEF ARE NOT THE NAMES IN THE CODE. This table is the mapping, written out
# because guessing it wrong is silent -- every one of these is a `Float64` keyword that accepts
# any value without complaint.
#
#   brief                              keyword of `cbe_bug_step!`   default   campaign
#   ---------------------------------  --------------------------   -------   --------
#   "truncation threshold for          `stol_pre`                   1e-10     1e-2
#    rsvd_cbe"
#   "half sweep"                       `split_cutoff`               1e-14     1e-6
#   "root_out"                         `root_cutoff`                0.0       1e-8
#   "different m"                      `krylov_basis` (+ `krylov_tol = 0`)  30  scanned
#   "expand_basis, tolerance beta"     `krylov_tol`                 1e-6      1e-6
#
# ⚠ `stol_pre` IS RELATIVE, WHICH IS WHY 1e-2 IS A SENSIBLE NUMBER AND NOT A CATASTROPHE.
# `_preselect_left` normalises the sketched action before its SVD "because `cutoff` is relative to
# the largest singular value" -- the new directions are O(tau)-small in absolute terms, so an
# absolute threshold would either keep everything or nothing. `stol_pre = 1e-2` means "keep
# expansion directions carrying more than 1% of the leading weight", which is the CBE preselection
# tolerance, NOT a truncation of the state. The state's own truncation is `trunc_thresh`, left at
# 1e-12; setting THAT to 1e-2 would discard 1% of the wavefunction every step.
#
# ⚠ `krylov_tol` IS SAAD'S CONTRIBUTION ESTIMATE, NOT A BARE `beta`. The brief calls it beta and
# the estimate is `beta_m * |[exp(tau*T_m)]_{m,1}|`, so beta is in it -- but `beta` ALONE was
# measured to be inert: at L=12, Heisenberg and XX returned BIT-IDENTICAL results for any beta
# threshold from 1e-6 to 1e-13, because a breakdown never happens on a generic H and every bond
# ran to the cap. What decides whether a Krylov vector matters is how much weight the propagator
# puts on it. See `_kry_contribution` in `sweeps/cbe_bug.jl`.
#
# ⛔ `m` AND THE beta RULE ARE ALTERNATIVES, NOT A PAIR. `bug_step_kwargs(m)` with `m >= 1` PINS
# the depth (`krylov_tol = 0.0` disables the test); `m = 0` runs the adaptive rule at
# `krylov_tol = 1e-6` with the cap at 30. Passing a small `krylov_basis` AND a live `krylov_tol`
# measures neither: the cap fires first and the tolerance never gets to decide.
#
# ⛔ `m = 1` REPRODUCES `m = 0` EXACTLY and is not a data point -- `cbe_expand` already spans one
# power of H, so the first genuine addition is `m = 2`. A depth scan that starts at 1 reports its
# first two points as identical and looks like a saturating curve.
# ══════════════════════════════════════════════════════════════════════════════════════════════

"""
    BUG_TOL

The three truncation tolerances the campaign pins, in the code's names. Separate from the depth
knobs because these are FIXED by the brief and the depth is the thing being scanned.
"""
const BUG_TOL = (stol_pre     = 1e-2,      # rsvd_cbe preselection, RELATIVE
                 split_cutoff = 1e-6,      # the half-sweep frame split
                 root_cutoff  = 1e-8)      # the root core split ("root_out")

"""
    RSVD_KNOBS

The rSVD sketch geometry. These are the knobs `rsvd_knobs_l20.jl` scans PER GEOMETRY, because the
sketch's payoff depends on how much contraction there is to shrink -- a nearest-neighbour MPO has
little, a cylinder MPO has a lot.

  - `dex`        directions the expansion is allowed to admit per bond
  - `dover`      oversampling on top of `dex`. ⛔ `dover = 0` on an rSVD arm reproduces the
                 frozen-start signature (converged at chi = 1 in two sweeps) and is not a
                 baseline; `dover = 4` is the measured default.
  - `comp_ratio` sketch width as a fraction of the budget
  - `growth`     per-bond rank growth factor. ⚠ PROBLEM-DEPENDENT: 2.0 is a starting point to be
                 re-derived per geometry from `err_fnl`, never a constant of the method.
"""
const RSVD_KNOBS = (dex = 8, dover = 4, comp_ratio = 1.0, growth = 2.0)

"""
    bug_step_kwargs(m; parallel=true, rsvd=true, knobs=RSVD_KNOBS, tol=BUG_TOL) -> NamedTuple

Keyword arguments for `cbe_bug_step!` under the campaign's settings.

`m >= 1` pins the half-sweep Krylov depth; `m = 0` uses the adaptive contribution rule at
`krylov_tol = 1e-6`.

⚠ `parallel = true` NEEDS `julia -t 2` OR MORE. On one thread the two half-sweeps interleave on
one worker and this is the sequential step plus scheduling overhead -- a small pessimisation, not
a no-op. The result is not bit-identical to the sequential arm either (each sweep re-gauges its
own copy); `tests/rsvd_cbe/` pins the agreement at 1e-10.

⛔ `rsvd = false` IS THE A/B REFERENCE ARM, NOT A FASTER SETTING. It forms the complete fused
basis -- cost `d*chi`, exactly what the sketch exists to avoid -- and makes `comp_ratio`, `dover`
and the RNG inert. It is how "did the sketch lose accuracy" is answered, and it is never the
production choice.
"""
function bug_step_kwargs(m::Int; parallel::Bool = true, rsvd::Bool = true,
                         knobs = RSVD_KNOBS, tol = BUG_TOL)
    m >= 0 || throw(ArgumentError("m must be >= 0, got $m"))
    depth = m >= 1 ? (krylov_basis = m,  krylov_tol = 0.0) :
                     (krylov_basis = 30, krylov_tol = 1e-6)
    return (dex          = knobs.dex,
            dover        = knobs.dover,
            comp_ratio   = knobs.comp_ratio,
            growth       = knobs.growth,
            exact        = !rsvd,
            stol_pre     = tol.stol_pre,
            split_cutoff = tol.split_cutoff,
            root_cutoff  = tol.root_cutoff,
            parallel     = parallel,
            depth...)
end

"A short human label for a BUG configuration, used as the `arm` column so two rows of a scan can
never be confused after the fact."
bug_label(m::Int; parallel::Bool = true, rsvd::Bool = true) =
    "bug_" * (m >= 1 ? "m$m" : "beta") * (rsvd ? "_rsvd" : "_exact") * (parallel ? "_par" : "_seq")

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE chi=128 -> chi=512 HANDOFF
# ══════════════════════════════════════════════════════════════════════════════════════════════

using JSON

"""
    optimal_bug_kwargs(geom; m_default=3, dir=...) -> (kwargs, provenance)

The settings `select_knobs.py` chose from the chi=128 scans, or the campaign defaults if no scan
has been run for `geom` yet. `provenance` is a one-line string the caller MUST print.

⛔ THE PROVENANCE STRING IS NOT DECORATION. The campaign plan is "tune at chi=128, then run
chi=512 with those settings", so the production figure's knobs were measured at a DIFFERENT bond
dimension from the one they are used at. That is a deliberate and reasonable shortcut -- scanning
at chi=512 would cost more than the figure -- but it is an extrapolation, and a reader who assumes
the settings were tuned where they ran would be wrong. Deeper Krylov matters more once the basis
is richer, and the sketch's payoff GROWS with chi because it shrinks work that scales with chi;
neither effect is captured by a chi=128 scan.

⚠ AN ABSENT FILE IS NOT AN ERROR -- it means "not tuned yet", and the defaults are a documented
starting point rather than a measured optimum. The distinction is in the provenance string.
"""
function optimal_bug_kwargs(geom::AbstractString; m_default::Int = 3, D::Int = 0,
                            dir::AbstractString = joinpath(@__DIR__, "..", "results"),
                            parallel::Bool = true, rsvd::Bool = true)
    # ⛔ THE TUNING IS PER (GEOMETRY, CAP). `dex` is an ABSOLUTE per-side budget and `growth` a
    # PROPORTIONAL one, so a scan at chi=64 and one at chi=128 do not have the same winner, and
    # `select_knobs.py` therefore files them separately. Loading the wrong cap's file is exactly
    # the silent cross-contamination `geom_key` already guards against for the circumference.
    # `D = 0` asks for whichever cap is on disk, and SAYS SO in the provenance.
    path = joinpath(dir, "bug_optimal_$(geom)_D$(D).json")
    if D == 0 || !isfile(path)
        cands = filter(f -> startswith(f, "bug_optimal_$(geom)_D") && endswith(f, ".json"),
                       isdir(dir) ? readdir(dir) : String[])
        isempty(cands) && return (bug_step_kwargs(m_default; parallel = parallel, rsvd = rsvd),
                                  "campaign DEFAULTS (no knob scan on disk for '$geom')")
        path = joinpath(dir, first(sort(cands)))
    end
    j = JSON.parsefile(path)
    k = get(j, "knobs", Dict{String, Any}())
    m = haskey(k, "m") ? Int(k["m"]) : m_default
    out = bug_step_kwargs(m; parallel = parallel, rsvd = rsvd)
    # ⛔ EXPLICIT CONVERSION. JSON hands back `Int64` or `Float64` depending on how the number was
    # written; `dex = 8.0` would fail the `::Int` keyword and `dover = 4` is fine either way, so
    # the failure would be intermittent across scans rather than deterministic.
    for (nm, sym) in (("stol_pre", :stol_pre), ("split_cutoff", :split_cutoff),
                      ("root_cutoff", :root_cutoff), ("comp_ratio", :comp_ratio),
                      ("growth", :growth))
        haskey(k, nm) && (out = merge(out, NamedTuple{(sym,)}((Float64(k[nm]),))))
    end
    for (nm, sym) in (("dex", :dex), ("dover", :dover))
        haskey(k, nm) && (out = merge(out, NamedTuple{(sym,)}((Int(k[nm]),))))
    end
    chim = get(get(j, "study_A", Dict()), "chi_measured", get(get(j, "study_B", Dict()),
                                                              "chi_measured", "?"))
    return (out, "TUNED: $(basename(path)), measured at chi=$chim " *
                 "(factor-$(get(j, "target_factor", "?")) Pareto rule)")
end
