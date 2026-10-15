# CAN A RANK-ADAPTIVE BUG SWEEP REPRODUCE THE ED RESULTS THAT DEFINE THE MBL AND SCAR
# LITERATURE, IN THE REGIME WHERE A FIXED-RANK PROJECTION CANNOT?
#
# THE QUESTION, stated so it can come out either way. Two protocols whose published results
# exist only as exact diagonalisation of short chains:
#
#   mbl   the disordered XXZ chain of Rev. Mod. Phys. 91, 021001 (arXiv:1804.11065) Eq. (5)/(6)
#         -- Jordan-Wigner-identical to the interacting fermions of the Bloch-group experiment --
#         quenched from the Néel charge-density wave. Observables: the IMBALANCE `I(t)` (the
#         experiment's own), the half-chain entropy (LOGARITHMIC in the MBL phase, LINEAR in the
#         ergodic one), and the site-resolved `⟨S^z_j⟩`.
#   pxp   the Rydberg blockade chain of arXiv:2011.09486, quenched from `|Z_2⟩`. Observables: the
#         REVIVAL FIDELITY `|⟨Z_2|ψ(t)⟩|²`, the half-chain entropy (which OSCILLATES rather than
#         saturating), and `⟨S^z_j⟩`.
#
# Three integrators, one rank cap ladder, one ED reference per realisation.
#
# ⛔ WHAT WOULD FALSIFY THE HYPOTHESIS, and it must be said before the numbers arrive. The
# recorded result on the Heisenberg chain is that CBE-BUG is ~4x LESS accurate than `tdvp2` at
# IDENTICAL pinned rank, and that this is intrinsic to the local step rather than to the basis
# (measured at chi = 41/64/128 with nothing truncated). So "BUG wins" is NOT the default
# expectation here; the hypothesis is specifically that on these two protocols the ordering
# INVERTS, because both start from a chi = 1 product state and demand a rank schedule that is
# either creeping (MBL, logarithmic) or NON-MONOTONE (scars, oscillating) -- failures of a
# different kind from the local-step constant. If the cap ladder shows `tdvp2` matching or
# beating the BUG at every chi on these models too, that is the answer and it gets reported.
#
# ⛔ THE CAP IS THE INDEPENDENT VARIABLE, NOT A SAFETY NET, which is the opposite of
# `cbe_sweeps_l16.jl`'s configuration. There, rank was tolerance-driven and `maxdim` was set high
# enough never to bind, because the question was cost-to-accuracy for adaptive-rank methods. The
# question HERE is what a method does when it is not allowed enough rank -- that is the entire
# content of "robust" -- so `maxdim` binds by construction, `trunc_thresh` is set well below it,
# and every arm at a given `chi` is doing the same-sized work. All three arms take the cap through
# the SAME closing truncation, so the comparison is between what they put IN the cap.
#
# MATCHED KNOBS, because an unmatched one has already produced an inverted result once in this
# project. Identical across arms: `maxdim`, `trunc_thresh`, `maxiter = 30`, `conv_tol = 0.0` (so
# every local solve burns the same Krylov depth rather than one arm stopping adaptively), and
# `growth` for the two CBE arms. Left at the library default by OMISSION: the BUG's own
# `krylov_basis`/`krylov_tol` basis-update depth -- pinning it here would fork this campaign from
# the library the day a default changes, and depth is separately measured to be accuracy-neutral.
#
# ⚠ WHAT "AT THE SAME chi" DOES AND DOES NOT MEAN, stated because this project has been bitten by
# exactly this. The cap is enforced on the STATE, once per step, by the closing truncation in all
# three arms -- so the states being compared carry the same number of stored numbers, and an
# accuracy comparison at fixed `chi` is fair. The INTERMEDIATE objects are not the same size:
# `tdvp2` forms a `d²χ²` two-site block, and the BUG's half-sweep frames are governed by
# `split_cutoff` (default 1e-14) with `split_maxdim = 0`, i.e. NOT capped at `maxdim`. That is each
# method's own design and is left alone; the consequence is only that "χ = 8" is a statement about
# the state, never about peak memory or work. A cost claim needs `elapsed`, and see below.
#
# ⛔ THE COST COLUMN IS NOT A COST AXIS. `krylov` counts operator applications in the local
# solves and CANNOT see `cbe_expand`, so it under-reports both CBE arms. It is recorded because it
# is informative about the solves; any cost claim must come from `elapsed` on a cluster node with
# one timing job running, and this driver is not that job.
#
# usage:  julia -t N --project=. benchmarks/mbl_scars.jl \
#             [phase] [L] [T] [dt] [seed] [Ws] [chis] [schemes] [deltas]
#         phase   smoke | mbl | pxp | all
#         Ws      comma-separated disorder strengths (mbl only)
#         chis    comma-separated rank caps
#         schemes comma-separated subset of bug,tdvp2,cbe1s
#         deltas  comma-separated anisotropies: 1.0 = interacting (MBL), 0.0 = Anderson control
#   env   BUG_OUTDIR   where the CSV goes. MANDATORY on the cluster -- the default is next to
#                      this file, i.e. inside $HOME, which is code-only storage.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

const HERE = @__DIR__
include(joinpath(HERE, "exact_sparse.jl"))
include(joinpath(HERE, "mbl_scars_exact.jl"))
# The SPINFUL reference, in the fermion occupation basis. Validated independently by
# `benchmarks/hubbard_aa_check.jl` (stdlib only, seconds) BEFORE the MPO that it grades was
# written -- so agreement is evidence rather than a shared assumption.
include(joinpath(HERE, "hubbard_aa_exact.jl"))

const OUTDIR = get(ENV, "BUG_OUTDIR", joinpath(HERE, "results"))
const PHASE  = isempty(ARGS) ? "all" : ARGS[1]
const L      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
const TMAX   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 50.0
const DT     = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.05
const SEED   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
const WS     = length(ARGS) >= 6 ? Tuple(parse.(Float64, split(ARGS[6], ","))) : (1.0, 4.0, 8.0)
const CHIS   = length(ARGS) >= 7 ? Tuple(parse.(Int, split(ARGS[7], ","))) : (8, 16, 32, 64)
# ⚠ `"bugmid"` IS A FOURTH SCHEME AND IS DELIBERATELY NOT IN THIS TUPLE. It is the second-order
# midpoint BUG (`cbe_bug_midpoint_step!`, arXiv:2402.08607); `stepper` accepts it, so ask for it
# by name in ARGS[8]. It is excluded from the default so that nothing already measured moves and
# so the default sweep does not silently double the BUG cost -- the midpoint takes TWO sweeps per
# step. Add it wherever a dt-order claim is being made: the plain `bug` arm was measured at
# dt-ratio 1.98 (i.e. FIRST order) on the doublon cells of `phase_hub4`.
const ALL_SCHEMES = ("bug", "tdvp2", "cbe1s")
const SCHEMES = length(ARGS) >= 8 ? Tuple(String.(split(ARGS[8], ","))) : ALL_SCHEMES
# ⛔ `Δ` IS THE INTERACTION, AND THE Δ = 0 ARM IS A DIFFERENT PHYSICAL PHENOMENON RATHER THAN A
# WEAKER VERSION OF THE SAME ONE. `delta` multiplies `S^z S^z`, which under Jordan-Wigner is the
# density-density interaction `V = J·Δ` of RMP Eq. (5):
#
#   Δ = 1   the INTERACTING chain -- MANY-BODY localisation. Half-chain entropy grows like
#           `log t` and keeps growing over the whole plotted window; the imbalance decays to a
#           non-zero plateau. This is the model the MBL literature is about.
#   Δ = 0   the disordered XX chain -- FREE fermions with on-site disorder, i.e. ANDERSON
#           localisation. Every single-particle orbital is localised, the entropy SATURATES
#           instead of growing, and there is no many-body transition at all.
#
# So Δ = 0 is the CONTROL that says which features of a curve are "many-body": anything present
# in both is single-particle localisation. It is also the arm where the rank demand is bounded,
# which is exactly why it must not be the arm the method is tuned on.
#
# ⚠ Δ = 0 IS ALSO EXACTLY SOLVABLE, AND THAT IS WHAT LIFTS THE SIZE LIMIT. A quadratic
# Hamiltonian has `⟨n_j(t)⟩` and a Slater-determinant entropy from one `L x L` matrix
# exponential, so the non-interacting arm can be graded EXACTLY at `L = 40` (the size
# arXiv:1501.05661's own DMRG uses) or `L = 100`, where no ED exists. That reference is
# `aa_free_fermion` in `mbl_scars_exact.jl`, and the `smoke` phase CROSS-VALIDATES it against the
# `2^L` sector propagation at `L = 12` before it is trusted anywhere ED cannot follow.
#
# Both Δ are still graded against the SAME ED reference inside this driver, which is the honest
# like-for-like comparison at ED-reachable sizes; the large-`L` analytic arm is a separate phase.
const DELTAS = length(ARGS) >= 9 ? Tuple(parse.(Float64, split(ARGS[9], ","))) : (1.0,)

# `trunc_thresh` must NOT bind -- the cap is the variable. Kept above the `split_cutoff` default
# of 1e-14 so the BUG's half-sweep split stays no looser than its closing truncation (the
# recorded hard rule: a looser split inflates rank and makes a fixed-cap comparison meaningless).
const CUTOFF = 1e-12

# ── the truncation-threshold ladder (ARGS[10]) ──────────────────────────────────────────────
#
# `0.0` MEANS NO CUTOFF: rank is then limited ONLY by `maxdim`, so the Schmidt spectrum fills the
# cap and its tail sits at ~1e-17. That is the decisive arm of this campaign -- it is the ONLY
# setting that puts genuinely near-zero singular values in front of the integrators, and it is
# what separates the two methods in principle: 2-site TDVP's backward substep inverts the bond
# matrix, BUG's Galerkin step never forms an inverse at all.
#
# ⛔ COST WARNING. With no cutoff nothing truncates, so the rank fills to `maxdim` within ~10
# steps REGARDLESS of the physics. A 2-site Krylov application is then O(chi^3 d^2 w), so
# `maxdim = 1024` at `L = 40` costs ~1e2-1e3 s PER STEP. With a 1e-6 or 1e-10 cutoff the same cap
# is nearly free because the rank stays physical: an uncapped cell costs the state's REAL rank,
# not the cap. Never launch the (no-cutoff, large-chi) corner without a measured cost probe.
const CUTOFFS = length(ARGS) >= 10 ? Tuple(parse.(Float64, split(ARGS[10], ","))) : (CUTOFF,)

"""
    split_for(cutoff) -> Float64

The BUG half-sweep split threshold that goes with a given closing `trunc_thresh`.

⛔ HARD RULE `split_cutoff <= trunc_thresh`. The library default is `1e-14`, which is fine at
`trunc_thresh = 1e-12` but VIOLATES the rule the moment the cutoff is `0.0` (no truncation): a
split looser than the closing truncation inflates rank and makes a fixed-cap comparison
meaningless. `min` enforces it for every rung of the ladder, including the no-cutoff one.
"""
split_for(cutoff::Float64) = min(1e-14, cutoff)
# ⚠ SUPERSEDED BY THE `RSVD_PRESETS` LADDER BELOW, and kept only as the documented DEFAULT probe:
# `RSVDS` defaults to `("g11",)`, which is this value. Both CBE arms take their width from the
# same preset, so it is still held EQUAL between them -- it is simply no longer a constant.
# 1.1 rather than the library's 2.0 because growth was measured inert on the achieved chi while
# costing accuracy; the ladder is what turns that from an assumption into a measurement.
const GROWTH = 1.1
# One Krylov budget for every local solve in every arm. `conv_tol = 0` everywhere means none of
# them stops early, so a difference cannot be a difference in how hard they tried.
const MAXITER = 30

# ── the Krylov-budget ladder (ARGS[11]) ─────────────────────────────────────────────────────
#
# ⛔ DEFINED AFTER `MAXITER` ON PURPOSE: it defaults to it, and Julia evaluates top level in
# order, so placing this above `MAXITER` is an UndefVarError on load.
#
# `lanczos_expv` exits on BREAKDOWN ONLY, so every local solve burns its full budget and the
# budget is a direct multiplier on cost. Scanning it is how "each method at its most efficient
# settings" gets DEFINED rather than asserted: the best setting is the smallest budget whose
# accuracy still matches the deep-budget arm, found per scheme rather than shared.
const KMAXES = length(ARGS) >= 11 ? Tuple(parse.(Int, split(ARGS[11], ","))) : (MAXITER,)

# ── the rSVD-CBE PROBE ladder (ARGS[13]) ────────────────────────────────────────────────────
#
# THE SKETCH IS THE COST OF THE TWO CBE ARMS, so this is the axis that decides what "BUG at its
# most efficient settings" means. Every preset below is a set of `cbe_expand` keywords, and both
# `cbe_bug_step!` and `tdvp_cbe1s_step!` take the identical set -- so an arm is never compared
# against another arm that was allowed a different probe.
#
# ⛔ `tdvp2` HAS NO rSVD AND MUST NOT BE REPLICATED ACROSS THIS LADDER. It is a plain two-site
# sweep; running it once per preset would multiply its cost by `length(RSVDS)` and, worse, invite
# reading the scatter across those identical runs as an effect. The phase loops run it for the
# FIRST preset only and write `-` in the `rsvd` column; `rsvd_arms` below encodes that rule once.
#
# The axes, in the order they matter for cost:
#
#   * PROBE WIDTH. `growth = g` asks for `ceil(g*dmax) - r` new directions per side, so the probe
#     SCALES WITH THE RANK -- at chi = 1024, `growth = 2.0` is a ~1000-column sketch and the
#     "randomised" SVD is no longer small. `dex = n` is an ABSOLUTE per-side budget and is flat in
#     chi. This is the difference between an O(chi^3) and an O(chi^2 n) selection, and it is the
#     first thing to measure.
#   * THE FOLD. `fold_omega` contracts the probe into the environment chain instead of forming
#     `H*Theta`: O(chi^3) -> O(chi^2) per sketch, but three chains where the shared build pays
#     one, so it wins only for `3*Dpre < chi`. ⛔ IT NEEDS A SMALL `dex` TO WIN AT ALL -- paired
#     with `growth = 2.0` the width is ~chi/2, far past the crossover, which is why every `*f`
#     preset here pins `dex`.
#   * OVERSAMPLING. `dover === nothing` is `Dpre = ceil(1.2*budget)`; `dover = 4` is `budget + 4`,
#     which at `dex = 16` is 20 columns instead of 20 -- the two coincide there and diverge at
#     large budget, where the multiplicative form is the expensive one.
#   * FORM OF THE SELECTION. `preselect_only` skips the final SVD; `comp_ratio` splits the probe
#     between the complement and the isometry (`nothing` = no projector in the probe at all);
#     `sulz_cap` bounds the expanded rank at `2r`.
#   * `exact = true` IS NOT AN rSVD SETTING -- it replaces the thin Gaussian with the complete
#     fused basis, i.e. plain CBE. It is carried as THE CONTROL: the sketch is an approximation to
#     it, so it bounds both the accuracy the sketch can reach and the cost it has to beat.
const RSVD_PRESETS = Dict{String, NamedTuple}(
    # probe width, relative (`dex = 0` hands the budget to the `growth` schedule)
    "g20"   => (growth = 2.0,),
    "g15"   => (growth = 1.5,),
    "g11"   => (growth = 1.1,),          # the driver's previous hardcoded `GROWTH`
    # probe width, absolute. `growth` is left unset ON PURPOSE: `cbe_expand` takes
    # `budget = dex <= 0 ? schedule : dex`, so a positive `dex` makes `growth` dead code, and
    # pinning it here would suggest the two interact.
    "d8"    => (dex = 8,),
    "d16"   => (dex = 16,),
    "d32"   => (dex = 32,),
    "d64"   => (dex = 64,),
    # the fold, always with a pinned width -- see the note above
    "d8f"   => (dex = 8,  fold_omega = true),
    "d16f"  => (dex = 16, fold_omega = true),
    "d32f"  => (dex = 32, fold_omega = true),
    "d64f"  => (dex = 64, fold_omega = true),
    # form of the selection, all at one width so the width is not also moving
    "d16o4" => (dex = 16, dover = 4),
    "d16p"  => (dex = 16, preselect_only = true),
    "d16c5" => (dex = 16, comp_ratio = 0.5),
    "d16cn" => (dex = 16, comp_ratio = nothing),
    "cap"   => (growth = 2.0, sulz_cap = true),
    # THE CONTROL: no sketch at all
    "exact" => (exact = true,),
)

"""
    rsvd_kw(name) -> NamedTuple

The `cbe_expand` keywords for one preset, or a hard error naming the ones that exist.

⛔ AN UNKNOWN NAME MUST NOT FALL BACK TO A DEFAULT. A scan whose preset name is misspelled would
otherwise run `length(RSVDS)` identical arms and report their timing scatter as a probe effect.
"""
function rsvd_kw(name::String)
    haskey(RSVD_PRESETS, name) || error(
        "unknown rsvd preset $(repr(name)); have: " * join(sort(collect(keys(RSVD_PRESETS))), ", "))
    return RSVD_PRESETS[name]
end

const RSVDS = length(ARGS) >= 13 ? Tuple(String.(split(ARGS[13], ","))) : ("g11",)
foreach(rsvd_kw, RSVDS)              # fail on a typo at LOAD time, not 20 minutes in

"""
    rsvd_arms(scheme) -> Tuple of preset names

Which probe settings this scheme is actually swept over: the full ladder for the two CBE arms,
and the first entry ONLY for `tdvp2`, which has no probe. Its rows carry `rsvd = "-"`.
"""
rsvd_arms(scheme::String) = scheme == "tdvp2" ? (RSVDS[1],) : RSVDS
rsvd_tag(scheme::String, r::String) = scheme == "tdvp2" ? "-" : r

# ── where the exact reference stops existing ────────────────────────────────────────────────
#
# The `Sz = 0` sector is `C(L, L/2)`: 12,870 at L=16, 184,756 at L=20, and 1.4e11 at L=40. ED is
# a hard wall, not a slow path. arXiv:1501.05661 hit the same wall and drew the same line: ED for
# its NON-INTERACTING Fig. 3, DMRG (L=40, m_max=100) for everything interacting.
#
# ⛔ ABOVE THIS, `obs_exact` AND `obs_err` ARE `NaN`, DELIBERATELY. Validation at large `L` is
# (i) the paper's published curves, (ii) convergence in `chi` and `cutoff`, and (iii) the two
# witnesses that need no reference: `energy_drift` and `norm`. Never compare a large-`L` arm to
# our own finest grid and call it an error -- the recorded rule is that a no-analytic model needs
# a PUBLISHED anchor.
#
# ⚠ The NON-interacting arm is exempt: `Delta = 0` is quadratic, so `aa_free_fermion` grades it
# EXACTLY at any `L` from one `L x L` matrix exponential. That is why `phase_aa` runs at L=40
# with real `obs_err` while `phase_run` at L=40 cannot.
const ED_MAX = 20

# The PXP exact reference is capped by the CONSTRAINED dimension, not by `L` (see `pxp_dim`).
# 832,040 is `pxp_dim(28)`: the sparse `H` then holds ~L nonzeros per row, i.e. ~23M entries or
# roughly 370 MB, which fits the 16 G these jobs request with room for the Krylov vectors.
# ⚠ Raising this is a MEMORY decision, not a physics one -- `pxp_dim(32)` is 5.7M states and ~6x
# the matrix. Measure `MaxRSS` before going up (cluster_compliance.md section 8).
const PXP_ED_DIM_MAX = 832_040

# ── the paper's stationary-imbalance window ─────────────────────────────────────────────────
#
# arXiv:1501.05661, verbatim: "averaging the imbalance in the evolution time interval from 15τ
# to 30τ". Fig. 2's shaded band is that window.
#
# ⛔ AN EARLIER VERSION OF THIS DRIVER AVERAGED t = 13.7, 17.1, 20.5 INSTEAD. That is neither the
# paper's window nor a subset of it -- 13.7 lies BEFORE it -- so any "stationary imbalance"
# computed that way is not the published quantity and cannot be compared to Fig. 3 or Fig. 4.
#
# ⛔ AND `TMAX` MUST REACH 30. A run stopped at T = 25 averages 15..25 and silently reports it as
# the stationary value; the window is not a target the run can partially hit.
const STAT_T0, STAT_T1 = 15.0, 30.0

# The experiment's initial doublon fraction, "≃34(2)%" (arXiv:1501.05661). ARGS[12] overrides it;
# `0.0` gives the singly-occupied CDW, which is a DIFFERENT physical initial condition and not
# the one Fig. 4 measures.
const DOUBLON_FRAC = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 0.34

"""
    cdw_species(L; doublon_frac, seed) -> Vector{Symbol}

What sits on each EVEN site of the initial charge-density wave: `:up`, `:down`, or `:both`.

The experiment prepares "zero, one or two atoms on each even site" with a doublon fraction of
`≃34(2)%`, and its curves are averaged over ~100 such states. One call here is therefore ONE
MEMBER of that ensemble, selected deterministically from `seed` so the realisation is reproducible
and every integrator arm sees the identical start.

⛔ THE SINGLY-OCCUPIED SITES ALTERNATE `:up`/`:down` RATHER THAN BEING RANDOM. A random spin
assignment would leave the two species imbalanced by O(√L) at these sizes, which shifts `⟨n↑n↓⟩`
and hence the `U` term -- an avoidable source of realisation-to-realisation scatter on top of the
disorder average the paper already takes.
"""
function cdw_species(Lp::Int; doublon_frac::Float64 = DOUBLON_FRAC, seed::Int = 1)
    0.0 <= doublon_frac <= 1.0 ||
        throw(ArgumentError("doublon_frac must be in [0,1], got $doublon_frac"))
    ne = length(2:2:Lp)
    nd = round(Int, doublon_frac * ne)
    rng = Random.MersenneTwister(1000 + seed)
    which = Set(Random.randperm(rng, ne)[1:nd])       # which even sites carry a doublon
    out = Vector{Symbol}(undef, ne)
    k = 0
    for i in 1:ne
        if i in which
            out[i] = :both
        else
            k += 1
            out[i] = isodd(k) ? :up : :down
        end
    end
    return out
end

"""
    stat_step_set(nst) -> (steps_to_sample, stationary_step_set)

Sample points for a trajectory of `nst` steps, guaranteeing the paper's `15τ..30τ` window is
covered by at least `nwin` points ON THE STEP GRID (so no time is ever interpolated), and
returning which of them fall inside it.

⚠ THE CSV KEEPS THE WHOLE TRAJECTORY, so the stationary average is a POST-PROCESSING choice: if
the window is ever revised, the existing rows still answer the new question. The set returned
here only drives the convenience number printed to the log.
"""
function stat_step_set(nst::Int; nwin::Int = 16)
    lo = round(Int, STAT_T0 / DT)
    hi = min(round(Int, STAT_T1 / DT), nst)
    # A short run (e.g. a cost probe at T = 1) never reaches 15τ at all: the window is EMPTY, not
    # reversed. Without this guard `range(lo, hi)` counts DOWNWARDS and invents a bogus window.
    win = lo <= hi ? [n for n in round.(Int, range(lo, hi; length = nwin)) if 1 <= n <= nst] : Int[]
    steps = sort!(unique!(vcat(sample_steps(nst), win)))
    return steps, Set(win)
end

# ⛔⛔ `alloc_gb` IS NOT A CURIOSITY COLUMN -- IT IS THE ONE THAT WOULD HAVE CAUGHT THE d=4
# DEFECT. A fused d=4 step churned 28.5 GiB at bond dimension 32 (job 16342051) and every column
# this file wrote was CORRECT: the imbalance, the energy drift, the norm, even `maxbond`. Only
# `elapsed` moved, and a slow step in a shared-node campaign reads as contention, not as a bug.
# The step was 45% recomputed `fusion_basis` SVDs and the campaign wrote it off as "expensive"
# for days.
#
# ⚠ AND IT IS THE MORE TRUSTWORTHY COST AXIS OF THE TWO. Wall clock is not comparable across jobs
# -- a 1.6x spread on identical work has been measured from node contention alone -- while
# allocated bytes are a deterministic property of the code path. When the two disagree, this one
# is right. (`krylov` is NOT a cost axis at all: it cannot see `cbe_expand`.)
const COLS = "suite,scheme,model,sym,L,W,delta,seed,maxdim,cutoff,kmax,rsvd,dt,t,obs,obs_exact," *
             "obs_err,entropy,entropy_exact,entropy_err,sz_err,energy,energy_drift,norm," *
             "maxbond,centrebond,krylov,elapsed,alloc_gb,smin,sgapmin"

# ⛔ THE FORMAT IS ONE LITERAL, NOT A CONCATENATION. `@printf` resolves its format at MACRO
# EXPANSION, so `@printf(io, "a" * "b", x)` is a LoadError the moment the file is included --
# and it is invisible to a parse-only syntax check, which never expands macros. `COLS` above may
# be concatenated freely; it is an ordinary string.
#
# ⛔⛔ 30 SPECIFIERS, 30 ARGS, 30 NAMES IN `COLS` -- AND NOTHING CHECKS THAT FOR YOU. `@printf`
# validates the count only when the line RUNS, so a mismatch survives parsing, macro expansion
# and a full package load, then aborts the job on the first row written (measured: "25 != 27",
# two minutes into a cluster job, after the gate had already produced its first good line). It
# happened because two columns were appended to `COLS` and to the argument list by a scripted
# edit whose format-string replacement silently did not match -- and the script only verified
# that the file had changed at all, which was true because its OTHER replacements applied.
# Any change here must be counted in all three places by hand.
row(io; suite, scheme, model, sym, l, w, delta, seed, maxdim, cutoff, kmax, rsvd, dt, t, obs,
    obs_exact, obs_err, entropy, entropy_exact, entropy_err, sz_err, energy, energy_drift, nrm,
    maxbond, centrebond, krylov, elapsed, alloc_gb, smin, sgapmin) =
    (@printf(io, "%s,%s,%s,%s,%d,%g,%g,%d,%d,%.1e,%d,%s,%g,%.6f,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e,%.10e,%.6e,%.10f,%d,%d,%d,%.3f,%.4f,%.6e,%.6e\n",
             suite, scheme, model, sym, l, w, delta, seed, maxdim, cutoff, kmax, rsvd, dt, t, obs,
             obs_exact, obs_err, entropy, entropy_exact, entropy_err, sz_err, energy,
             energy_drift, nrm, maxbond, centrebond, krylov, elapsed, alloc_gb, smin, sgapmin);
     flush(io))

# ── shared plumbing ─────────────────────────────────────────────────────────────────────────

"""
Bytes the GC has handed out so far, in GiB.

⚠ `Base.gc_bytes()` is a PROCESS-WIDE RUNNING TOTAL, so it is only meaningful as a difference
between two markers -- which is why every arm captures `ab0` next to its `t0` and reports the
delta. It counts allocation CHURN, not residency: a step that allocates 28 GiB can still peak at
1.5 GB of RSS, so ⛔ never size a `--mem` request from this column (cluster_compliance.md §8).

⚠ Unlike `@allocated` it does not discard the wrapped call's return value, which these loops need
for `info_krylov`.
"""
gc_gb() = Base.gc_bytes() / 2^30

maxstate(psi) = maximum(state_bond_dims(psi); init = 0)
centre_bond(psi) = leg_dim(psi[length(psi) ÷ 2], 3)

"""
One step of each integrator behind one signature.

⛔ `tdvp2` IS THE STRONG BASELINE AND `cbe1s` IS THE STRONGER ONE AT TIGHT TOLERANCE (recorded:
1.6x better than `tdvp2` at equal chi). Both are carried; dropping either would leave the
comparison against whichever happens to lose.
"""
function stepper(scheme::String, mpo, maxdim::Int; cutoff::Float64 = CUTOFF,
                 kmax::Int = MAXITER, rsvd::String = RSVDS[1])
    # ⛔ SPLATTED INTO THE TWO CBE ARMS ONLY. `tdvp2_step!` has no `dex`/`growth`/`fold_omega`, so
    # splatting there is an unmatched-keyword MethodError -- which is the correct outcome, but the
    # loops must not ask for it: see `rsvd_arms`.
    ex = rsvd_kw(rsvd)
    if scheme == "bug"
        # ⛔⛔ `krylov_basis` / `krylov_tol` ARE NOW PASSED, AND THEY WERE THE MISSING HALF OF
        # `MAXITER`. `MAXITER` is documented above as "one Krylov budget for every local solve in
        # every arm ... none of them stops early", but this function only ever handed it to the
        # ROOT solve. The half-sweep basis frames kept the LIBRARY defaults -- `krylov_basis = 30`
        # with `krylov_tol = 1e-6`, which is a genuine early-stopping rule (Saad's contribution
        # estimate) -- so the frames stopped early on every run ever made through this driver,
        # contradicting the stated intent.
        #
        # ⚠ THIS MOVES PREVIOUSLY-MEASURED `bug` NUMBERS, deliberately. It is a fairness fix: a
        # comparison in which `tdvp2` burns its whole budget and BUG's basis sweep quietly stops
        # at Saad's estimate is not a comparison of the two methods at equal effort.
        #
        # ⛔⛔ BUT `krylov_tol` IS **NOT** FORCED TO 0, AND AN EARLIER VERSION OF THIS COMMENT WAS
        # WRONG TO DO SO. Killing the contribution rule makes every bond build the full `kmax`
        # Krylov vectors instead of the ~3-5 the rule stops at, and that is a straight
        # pessimisation: MEASURED (job 16341581, the m-scan this suspicion motivated),
        #
        #     m        5       15       30       60      120
        #     elapsed  0.96s   1.41s    2.13s    4.31s   12.77s
        #     dn       1.276877e-02  then BIT-IDENTICAL 1.276831e-02 from m=15 upward
        #
        # ⇒ cost is linear (worse) in m and accuracy is FLAT. The stopping rule is not an
        # unfairness to be removed, it is an adaptive convergence test that is measurably free.
        # At L=4 the forced depth cost only 1.17x because Lanczos breaks down early in a tiny
        # local space; at L=40 with bd=128 the two-site space is ~2.6e5-dimensional, it never
        # breaks down, and forcing m=30 is ~6-10x the half-sweep work -- which is what stalled
        # job 16341332 at one sampled row in 24 minutes.
        #
        # ⚠ SO THE DEPTH SCAN ALSO ANSWERED ITS OWN QUESTION AND CLOSED IT: on the doublon cells
        # `dn` is BIT-IDENTICAL from m = 5 to m = 120, a 24x span. Krylov depth is NOT what makes
        # plain BUG 1e4x worse than tdvp2 there -- the exponential is already converged at m=5, so
        # the residual is the SCHEME's structural local error (its basis is built from the state
        # at the START of the step). `bugmid` is the fix, not more vectors.
        #
        # ⚠ To FORCE the depth for another such scan, pass `krylov_tol = 0.0` explicitly -- the
        # cap alone measures nothing, because the contribution rule fires first.
        #
        # ⚠ AND THE MATCHING LEVER ON THE OTHER SIDE IS STILL UNPULLED: `tdvp2`/`cbe1s` run with
        # `conv_tol = 0`, i.e. no early stop at all, so "each method at its most efficient
        # settings" is not yet a like-for-like statement. Give them a convergence test before
        # quoting a cost ratio as final.
        #
        # ⛔ THE BUG SWEEP SPELLS IT `root_conv_tol`, NOT `conv_tol`, and the difference is not
        # cosmetic: this sweep has TWO Krylov consumers -- the half-sweep basis frames
        # (`krylov_basis`/`krylov_tol`) and the Galerkin ROOT solve, which is the one that
        # actually applies `exp(tau H_0)`. `conv_tol` would be an unmatched keyword and Julia
        # rejects it, which is how this was caught; the equivalent of `tdvp2`'s `conv_tol = 0`
        # (never stop early) is `root_conv_tol = 0` here.
        #
        # ⛔ `split_cutoff` MUST TRACK `cutoff`. Its library default of 1e-14 is looser than a
        # `cutoff = 0.0` closing truncation, which violates the recorded hard rule and inflates
        # rank exactly in the no-cutoff arm this campaign turns on. `split_for` enforces it.
        return (p, tau) -> cbe_bug_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = cutoff,
                                         split_cutoff = split_for(cutoff),
                                         maxiter = kmax,
                                         krylov_basis = kmax,
                                         root_conv_tol = 0.0, root_substeps = 1, ex...)
    elseif scheme == "bugmid"
        # ⛔ THE SECOND-ORDER BUG, AND IT IS HERE BECAUSE THE FIRST-ORDER ONE WAS MEASURED.
        # `cbe_bug_step!` is formally FIRST order, and the ~4x dt-ratios it shows on some models
        # are cells where the leading first-order term happens to be small rather than absent.
        # MEASURED (job 16341097, `phase_hub4`, fused d=4 Aubry-Andre-Hubbard, L=4, FULL rank):
        #
        #   cell                   bug        cbe1s      tdvp2
        #   U=0 dfrac=0            3.89       3.90       3.90      <- looks second order
        #   U=0 dfrac=0.5 (doublon) 1.98      2.01       8.28      <- both CBE arms drop to FIRST
        #   U=4 dfrac=0.5 (doublon) 1.98      2.04       8.04
        #
        # and in those cells tdvp2 was ~1e4 x more accurate (6.9e-07 vs 4.7e-03). ⛔ IT IS NOT A
        # BASIS SHORTAGE: `cbe1s` reached bd=16, which IS full rank at L=4 (bond 2 is capped at
        # 4^2), so there was nothing left to expand and the residual error is purely temporal.
        #
        # `cbe_bug_midpoint_step!` (arXiv:2402.08607) builds the frames from a half-step state and
        # then takes the full step from the ORIGINAL vector, which restores second order --
        # recorded at 63-70x accuracy for 1.35x wall clock at matched cost.
        #
        # ⚠ DO NOT PASS `truncate` OR `frame_state` HERE: the midpoint wrapper sets
        # `truncate = false` on its basis stage and `frame_state` on its stepping stage, and an
        # explicit value would collide with the one it supplies.
        return (p, tau) -> cbe_bug_midpoint_step!(p, mpo, tau; maxdim = maxdim,
                                                  trunc_thresh = cutoff,
                                                  split_cutoff = split_for(cutoff),
                                                  maxiter = kmax,
                                                  krylov_basis = kmax,
                                                  root_conv_tol = 0.0, root_substeps = 1, ex...)
    elseif scheme == "tdvp2"
        return (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = cutoff,
                                       maxiter = kmax, conv_tol = 0.0)
    elseif scheme == "cbe1s"
        return (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = cutoff,
                                            maxiter = kmax, conv_tol = 0.0, ex...)
    end
    error("unknown scheme $scheme")
end

"""
    warm!(step, psi0, dt) -> nothing

Compile `step` on a THROWAWAY copy of the state, before any clock starts.

⛔⛔ WITHOUT THIS EVERY `elapsed` IS A LIE, AND THE LIE IS NOT SMALL. `stepper` returns a fresh
closure per (scheme, chi, cutoff, kmax, rsvd) cell, so Julia specialises and compiles it on the
FIRST call -- inside the timed loop. MEASURED on this driver: the same comparison read
"BUG 3.3x faster than tdvp2" with the JIT inside the loop and 1.1-1.2x once it was outside, purely
because `bug` ran first and paid the compile that `tdvp2` then inherited warm. The reported factor
was a compile-order artefact, three times the real effect.

⛔⛔ **TWO STEPS, NOT ONE, AND THE OLD "one step is enough -- specialisation is per method, not per
step" CLAIM WAS MEASURABLY WRONG.** It is true that Julia does not specialise on tensor SIZE, but a
single step FROM A PRODUCT STATE never enters the same methods a later step does: starting at
`bd = 1` the augmented rank cannot reach `maxdim`, so the truncate-to-cap branch is never taken and
its kernels stay uncompiled. MEASURED (job 16341790, `hub4prod`, L=20, **chi = 32**):

    step 1   0.8 s      product state -> bd 32, nothing truncated
    step 2   334.3 s    first step that has to truncate

334 s at bond dimension 32 is not arithmetic, it is compilation -- and with a one-step warm-up it
lands INSIDE the timed loop, on whichever arm happens to run first. That is the same compile-order
artefact this docstring already warns about, re-entering through the door the fix left open.
⚠ The real repair is in the package (`src/BUGJulia.jl` now precompiles the fused d=4 path over
THREE steps at a cap that binds); this second step is the driver-side belt to that braces, and it
also covers any model whose first truncating step comes later than the workload's.

Both steps are thrown away, so the timed run still starts from the exact same `psi0` as every
other arm.
"""
function warm!(step, psi0, dt::Float64)
    try
        p = copy(psi0)
        for _ in 1:2
            step(p, ComplexF64(-im * dt))
        end
    catch e
        # A cell that throws on step 1 will throw again inside the loop, where it is reported per
        # arm. Swallowing it here keeps the warm-up from aborting a whole sweep over one bad cell.
        @printf("  warm ⚠ threw %s\n", sprint(showerror, e)[1:min(end, 100)]); flush(stdout)
    end
    return nothing
end

info_krylov(info) = hasproperty(info, :krylov_dims) ? info.krylov_dims :
                    hasproperty(info, :krylov_dim)  ? info.krylov_dim  : 0

"""
    mps_spectrum_stats(psi) -> (; ent, smin, sgapmin)

The centre-bond Schmidt spectrum reduced to three numbers, from ONE canonicalisation.

  * `ent`     -- half-chain entanglement entropy in NATS, via the same `entropy_from_svals` the
                 ED side uses, so the log base cannot differ between the two.
  * `smin`    -- the SMALLEST RETAINED singular value.
  * `sgapmin` -- the smallest RELATIVE gap `(s_i - s_{i+1}) / s_i` over the retained spectrum.

⛔ THE LAST TWO ARE THE MECHANISM, NOT DECORATION, and they are why this function replaced a
bare `mps_entropy`. The claim under test is not merely that a fixed-rank projection is less
accurate -- it is that TDVP has a SPECIFIC structural weakness which the BUG does not:

  * 1-site TDVP inverts the bond matrix `C`, so its error is controlled by `1/smin`. For a
    normalised state `s_1 <= 1`, so `1/smin` IS essentially the condition number.
  * 2-site TDVP avoids the explicit inverse but keeps the projector-splitting structure: a
    forward `exp(-i tau H_2site)` and a BACKWARD `exp(+i tau H_1site)` substep. That backward
    step -- the minus sign -- is what this project already records as unstable in the dissipative
    setting, and it is the part that has no analogue in the BUG.
  * the BUG (Ceruti-Lubich-Walach) has NO backward substep and NO inverse; its published error
    bound is independent of small singular values.

MBL is exactly the setting that stresses this: the entanglement spectrum of a localised state is
near-degenerate with many near-zero values, so `smin` is tiny and `sgapmin` is small. Recording
both per sample turns "the BUG wins" into "the BUG wins WHERE `smin` is small", which is a
mechanism and can be falsified -- if the TDVP arms' error does NOT track `smin`, the explanation
is wrong even if the ranking holds.

⚠ `bond_spectrum` moves the orthogonality centre, which is lossless, so measuring does not
perturb the run -- but it is `O(L chi^3)`, so the three numbers are taken from one call rather
than three.
"""
function mps_spectrum_stats(psi)
    s = bond_spectrum(psi, length(psi) ÷ 2)          # sorted DESCENDING
    isempty(s) && return (; ent = 0.0, smin = NaN, sgapmin = NaN)
    g = Inf
    for i in 1:(length(s) - 1)
        s[i] > 0 && (g = min(g, (s[i] - s[i + 1]) / s[i]))
    end
    return (; ent = entropy_from_svals(s), smin = minimum(s),
            sgapmin = length(s) > 1 ? g : NaN)
end

"""
    imbalance_from_profile(sz) -> Float64

`I(t) = (2/L) Σ_j (-1)^{j+1} ⟨S^z_j⟩`, the same expression [`imbalance_dense`](@ref) evaluates on
the ED vector, so the two are comparable term by term rather than "the same observable".

⚠ IT TAKES THE PROFILE, NOT THE STATE, and that is a cost decision rather than a style one. Each
`sz_expectation(psi, j)` MOVES THE ORTHOGONALITY CENTRE, so asking for all `L` of them is `O(L²)`
canonicalisation -- about 0.2 s per sample at `L = 16, χ = 128`. The sampler already needs the
whole profile for `sz_err`, so an `imbalance_mps(psi)` that recomputed it internally doubled the
observable cost of every sample in the campaign for nothing.
"""
imbalance_from_profile(sz::Vector{Float64}) =
    (2 / length(sz)) * sum((isodd(j) ? 1 : -1) * sz[j] for j in eachindex(sz))

"""
    sample_steps(nst) -> Vector{Int}

Step indices to sample at: LOGARITHMIC in time, plus the endpoint.

⛔ WHY LOG AND NOT EVERY-Nth. The MBL signature IS a logarithm -- `S(t) ~ ξ log t` -- and a
linear grid puts almost every sample in the last decade, where the curve is flattest, and almost
none in the first three, where it is being established. It is also what makes the ED reference
affordable: each sample costs a dense `2^{L/2} × 2^{L/2}` SVD.

⛔ AND THE GRID IS A STEP COUNT, WHICH IS THE ONLY SAFE FORM. This project has already compared
an MPS at `t = 0.2` against an exact vector at `t = 0.25` by building the two grids
independently; here the MPS is sampled AT step `n` and the reference is advanced BY `(n - n_prev)
* dt`, so the two times are the same number by construction and not by agreement.
"""
function sample_steps(nst::Int)
    nst <= 1 && return [max(nst, 1)]
    ks = round.(Int, 10 .^ range(0.0, log10(nst); length = 14 * max(1, ceil(Int, log10(nst)))))
    return sort!(unique!(clamp!(vcat(ks, nst), 1, nst)))
end

"""
    revival_steps(nst; nmax=200) -> Vector{Int}

Step indices on a UNIFORM grid -- the sampling a PERIODIC signal needs.

⛔⛔ **LOG SAMPLING DESTROYS A REVIVAL CURVE, AND IT DOES IT SILENTLY.** `sample_steps` is correct
for MBL, whose signature IS a logarithm, and WRONG for scars, whose signature is periodic. At
`T = 30, dt = 0.05` (`nst = 600`) the log grid's late samples are ~90 steps apart, i.e. ~4.5 time
units, against a `|Z_2>` revival period of ~4.77. That is ONE SAMPLE PER REVIVAL: the peaks would
be aliased into a smooth decay, and the plot would look like ordinary thermalisation rather than
like scars. Nothing would error.

A uniform grid at `nmax = 200` gives ~32 samples per revival at those settings, which resolves
both the peak heights and the decay ENVELOPE -- and the envelope is the part that distinguishes a
scar from a trivial oscillation.

⚠ THE ENDPOINT IS FORCED IN, so the last row is at `t = TMAX` regardless of stride.
⚠ Still a STEP COUNT, not a time grid, for the reason `sample_steps` documents: the reference is
advanced BY `(n - n_prev)*dt`, so MPS and reference share a time by construction.
"""
function revival_steps(nst::Int; nmax::Int = 200)
    nst <= 1 && return [max(nst, 1)]
    stride = max(1, cld(nst, nmax))
    return sort!(unique!(clamp!(vcat(collect(stride:stride:nst), nst), 1, nst)))
end

"""
    exact_reference(H, v0, steps, dt, L, states; want_entropy) -> Vector{NamedTuple}

The ED reference at exactly the sampled times, by sparse Krylov (`expv_sparse`, which asserts its
own residual and substeps until it is met) advanced BETWEEN samples rather than restarted.

One reference per `(model, L, W, seed)`, shared by every arm and every cap -- so no arm can be
graded against a slightly different reference, and the cost is paid once.
"""
function exact_reference(H, v0::Vector{ComplexF64}, steps::Vector{Int}, dt::Float64,
                         Ls::Int, states::Vector{Int}; want_entropy::Bool,
                         kind::Symbol, z2::Union{Nothing, Vector{ComplexF64}} = nothing)
    out = NamedTuple[]
    v = copy(v0)
    prev = 0
    for n in steps
        v = expv_sparse(H, ComplexF64(-im * dt * (n - prev)), v)
        prev = n
        obs = kind === :mbl ? imbalance_dense(v, Ls, states) :
              abs2(dot(z2::Vector{ComplexF64}, v) / norm(v))
        push!(out, (; n = n,
                    obs = obs,
                    sz = sz_profile_dense(v, Ls, states),
                    ent = want_entropy ? half_chain_entropy_sector(v, Ls, states) : NaN))
    end
    return out
end

# ── PHASE smoke: the correctness gate ────────────────────────────────────────────────────────
#
# ⛔ NOTHING FROM THE PRODUCTION PHASES IS TRUSTWORTHY UNTIL THIS PASSES, and it is cheap enough
# to run every time. It separates "the model is wrong" from "the method is limited", which is the
# distinction the whole campaign rests on.
#
# ⛔⛔ THE GATE IS **dt-CONVERGENCE**, NOT AN ABSOLUTE THRESHOLD, AND THE FIRST VERSION OF THIS
# FILE GOT THAT WRONG. It demanded agreement with ED to `1e-10` at full rank, on the premise that
# a full-rank sweep IS `exp(tau H)`. That premise is true for the BUG -- one Galerkin solve on a
# frame that spans everything -- and FALSE for `tdvp2`, whose sweep is a second-order SPLITTING
# over bonds and therefore carries an `O(dt²)` error at any rank. MEASURED at `L = 8`, `T = 1`,
# `dt = 0.05`: `tdvp2` 8.1e-8, `cbe1s` 1.1e-7, `bug` 3.5e-7 -- every arm "failed" a gate that was
# simply mis-specified, while `obs`, `entropy` and `‖ψ‖` all agreed with ED to 6 digits.
#
# So the assertion is the one that actually distinguishes a correct implementation: HALVING `dt`
# MUST CUT THE ERROR BY ~4x. A wrong Hamiltonian does not converge; a correct one does, at its
# own order. That tests the MPO, the reference, the state conventions and the integrator's order
# in one measurement, and it cannot be satisfied by a coincidence of scale.
#
# ⛔ AND `‖ψ‖ = 1` IS ASSERTED SEPARATELY, because it is the check that would have caught the real
# bug this gate did find. `X = S^+ + S^-` is ANTI-Hermitian under Telum's `:none` normalisation
# (see `pxp_operators`), which made `exp(-iHt)` non-unitary: `‖ψ‖` grew to 3.95 while every
# observable stayed finite and plausible. Norm is not a formality here -- it is the only quantity
# that separates "non-Hermitian generator" from "inaccurate integrator".
#
# It also pins the conventions that are silent when wrong:
#   * the `t = 0` energy against a CLOSED FORM that includes the disorder (a field dropped at the
#     boundary sites, or applied to mirrored sites, passes every other check here);
#   * `|Z_2⟩`'s `⟨S^z_1⟩ = +1/2`, i.e. that `|•⟩ = up` on both sides;
#   * for MBL, the `:U1` run against the `:none` run -- the symmetry must change the COST and not
#     the physics, and that is asserted rather than assumed.

"""
    check_order!(fails, tag, errs, dts)

`errs[k]` is the error at `dts[k]`, with `dts` HALVING. Requires the error to fall by at least
`3x` per halving (a second-order step gives `4x`; the margin absorbs the point where the
truncation/Krylov floor starts to bite).

An error already below `1e-11` is at the floor and is passed without a ratio test -- there is
nothing left to converge, and demanding a ratio there would fail on rounding noise.
"""
function check_order!(fails::Vector{String}, tag::String, errs::Vector{Float64},
                      dts::Tuple)
    e1, e2 = errs[1], errs[2]
    e1 < 1e-11 && return                     # at the floor: exact to Krylov tolerance already
    if !(e1 < 1e-3)
        push!(fails, "$tag: error $e1 at dt=$(dts[1]) is far too large for a correct H")
        return
    end
    r = e2 > 0 ? e1 / e2 : Inf
    r >= 3.0 || push!(fails,
        "$tag: halving dt cut the error only $(round(r; digits = 2))x " *
        "(a second-order step must give ~4x) -- errs=$errs at dts=$dts")
end

function phase_smoke(io)
    Ls, T = 8, 1.0
    # TWO STEP SIZES, halving. The gate is the RATIO between them.
    DTS = (0.05, 0.025)
    fails = String[]

    # ---- MBL, against ED in the S^z = 0 sector ----
    #
    # THREE CELLS, each changing EXACTLY ONE THING from the production configuration, so a failure
    # is attributable:
    #   (:U1,   Δ=1)  the interacting model, symmetry-native -- what production runs
    #   (:none, Δ=1)  the same physics with the symmetry switched OFF. U(1) must change the COST
    #                 and not the answer; graded against the SAME ED vector, so this is a real
    #                 assertion and not two independent computations agreeing loosely.
    #   (:U1,   Δ=0)  the NON-INTERACTING (Anderson) control. It is a different phenomenon, not a
    #                 weaker one, so its MPO and its closed-form initial energy need their own
    #                 gate -- `Δ` enters the closed form, and a `Δ` silently ignored by the MPO
    #                 builder would leave the Δ=0 arm secretly interacting.
    hz = disorder_fields(Ls, 4.0; seed = 12345)

    for (sym, delta) in ((:U1, 1.0), (:none, 1.0), (:U1, 0.0))
        Hs, states, idx = disordered_xxz_sparse(Ls, hz; delta = delta)
        v0 = neel_vector(Ls, idx)
        # ONE reference at `t = T`, shared by both step sizes: `steps = [1]` with a "step" of `T`
        # propagates straight to `T`, so neither `dt` is graded against its own grid.
        ref = exact_reference(Hs, v0, [1], T, Ls, states;
                              want_entropy = true, kind = :mbl)[1]
        set_symmetry!(sym)
        mpo = disordered_xxz_mpo(Ls; delta = delta, hz = hz)
        psi0 = neel_state(Ls)
        e0 = real(mpo_energy(psi0, mpo)) / max(norm(psi0)^2, eps())
        ecl = neel_energy_closed(Ls, hz; delta = delta)
        @printf("  smoke mbl %-5s D=%g E(0)=%.12f  closed form=%.12f  diff=%.2e\n",
                sym, delta, e0, ecl, abs(e0 - ecl))
        abs(e0 - ecl) < 1e-10 || push!(fails, "mbl/$sym/D=$delta initial energy vs closed form")
        for scheme in SCHEMES
            errs = Float64[]
            for dt in DTS
                nst = round(Int, T / dt)
                psi = copy(psi0)
                step = stepper(scheme, mpo, 256)      # 256 >= full rank at L=8: no truncation
                t0 = time(); ab0 = gc_gb(); kd = 0
                for _ in 1:nst
                    kd += info_krylov(step(psi, ComplexF64(-im * dt)))
                end
                got = [sz_expectation(psi, j) for j in 1:Ls]
                sz_err = maximum(abs.(got .- ref.sz))
                push!(errs, sz_err)
                ib = imbalance_from_profile(got)
                st = mps_spectrum_stats(psi); ent = st.ent
                nrm = norm(psi)
                @printf("  smoke mbl %-5s D=%g %-6s dt=%-6g sz_err=%.3e S_err=%.3e |psi|=%.10f\n",
                        sym, delta, scheme, dt, sz_err, abs(ent - ref.ent), nrm)
                # ⛔ UNITARITY, asserted separately from accuracy -- see the phase header.
                abs(nrm - 1) < 1e-8 ||
                    push!(fails, "mbl/$sym/D=$delta/$scheme dt=$dt NORM=$nrm " *
                                 "(non-unitary generator, not an accuracy problem)")
                row(io; suite = "smoke", scheme = scheme, model = "mbl", sym = String(sym),
                    cutoff = CUTOFF, kmax = MAXITER, rsvd = rsvd_tag(scheme, RSVDS[1]),
                    l = Ls, w = 4.0, delta = delta, seed = 12345, maxdim = 256, dt = dt,
                    t = T,
                    obs = ib, obs_exact = ref.obs, obs_err = abs(ib - ref.obs),
                    entropy = ent, entropy_exact = ref.ent, entropy_err = abs(ent - ref.ent),
                    sz_err = sz_err,
                    energy = real(mpo_energy(psi, mpo)) / max(nrm^2, eps()),
                    energy_drift = 0.0, nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                smin = st.smin, sgapmin = st.sgapmin)
            end
            check_order!(fails, "mbl/$sym/D=$delta/$scheme", errs, DTS)
            @printf("  smoke mbl %-5s D=%g %-6s dt-ratio=%.2f (want >=3)\n",
                    sym, delta, scheme, errs[2] > 0 ? errs[1] / errs[2] : Inf)
        end
    end

    # ---- THE AUBRY-ANDRÉ FREE-FERMION REFERENCE, against ED ----
    #
    # ⛔ A REFERENCE USED AT `L = 40` MUST BE VALIDATED WHERE A SECOND ONE EXISTS. `aa_free_fermion`
    # is an `L x L` matrix exponential and `disordered_xxz_sparse` is a `2^L` sector propagation;
    # at `U = 0` they compute the SAME three numbers by completely different routes, sharing no
    # code and no basis convention. Agreement to Krylov tolerance is therefore evidence, not a
    # tautology -- and it simultaneously pins the JW factor of 2 in `aubry_andre_xxz_mpo`, the
    # odd-vs-even filling convention, and the Slater-determinant entropy formula.
    #
    # This is what licenses the `U = 0` arm to run at sizes no ED can reach.
    let Lc = 12, tc = 2.0, Dl = 3.0, J = 1.0
        hzq = quasiperiodic_fields(Lc, Dl)
        # `delta = 0` (no S^zS^z) and `J_spin = 2J` -- the fermion chain with hopping `J`.
        Hq, qstates, qidx = disordered_xxz_sparse(Lc, hzq; J = 2J, delta = 0.0)
        vq = neel_vector(Lc, qidx)
        eq = exact_reference(Hq, vq, [1], tc, Lc, qstates;
                             want_entropy = true, kind = :mbl)[1]
        ff = aa_free_fermion(Lc, tc; J = J, Delta = Dl)
        dI = abs(eq.obs - ff.imbalance)
        dS = abs(eq.ent - ff.entropy)
        dn = maximum(abs.((eq.sz .+ 0.5) .- ff.n))
        @printf("  smoke aa  L=%d t=%g D/J=%g: dI=%.3e dS=%.3e dn=%.3e (ED vs free fermions)\n",
                Lc, tc, Dl, dI, dS, dn)
        max(dI, dS, dn) < 1e-9 ||
            push!(fails, "aa free-fermion reference vs ED: dI=$dI dS=$dS dn=$dn")
    end

    # ---- PXP, :none by necessity, against ED in the constrained basis ----
    set_symmetry!(:none)
    Hp, pstates, pidx = pxp_sparse(Ls)
    @printf("  smoke pxp constrained dim=%d of 2^%d=%d (%.1fx)\n",
            length(pstates), Ls, 2^Ls, 2^Ls / length(pstates))
    z2 = z2_vector(Ls, pidx)
    pref = exact_reference(Hp, z2, [1], T, Ls, pstates;
                           want_entropy = true, kind = :pxp, z2 = z2)[1]
    mpo = pxp_mpo(Ls)
    psi0 = z2_state(Ls)
    # `|•⟩ = up` on both sides. A mirrored convention would put the excitations on the even sites,
    # which for `|Z_2⟩` is the OTHER, equally legal, symmetry-related state -- so every scalar
    # observable would still look perfect and only the site-resolved profile would be reversed.
    abs(sz_expectation(psi0, 1) - 0.5) < 1e-12 ||
        push!(fails, "pxp |Z2> convention: <Sz_1> = $(sz_expectation(psi0, 1)), want +1/2")
    for scheme in SCHEMES
        errs = Float64[]
        for dt in DTS
            nst = round(Int, T / dt)
            psi = copy(psi0)
            step = stepper(scheme, mpo, 256)
            t0 = time(); ab0 = gc_gb(); kd = 0
            for _ in 1:nst
                kd += info_krylov(step(psi, ComplexF64(-im * dt)))
            end
            got = [sz_expectation(psi, j) for j in 1:Ls]
            sz_err = maximum(abs.(got .- pref.sz))
            push!(errs, sz_err)
            nrm = norm(psi)
            fid = abs2(overlap(psi0, psi) / max(nrm, eps()))
            st = mps_spectrum_stats(psi); ent = st.ent
            @printf("  smoke pxp none  %-6s dt=%-6g sz_err=%.3e  F=%.6f (exact %.6f)  |psi|=%.10f\n",
                    scheme, dt, sz_err, fid, pref.obs, nrm)
            # ⛔ THE NORM CHECK THAT FOUND THE ANTI-HERMITIAN `X`. See the phase header.
            abs(nrm - 1) < 1e-8 ||
                push!(fails, "pxp/$scheme dt=$dt NORM=$nrm -- exp(-iHt) is not unitary, so " *
                             "the MPO is not Hermitian (this is how S^+ + S^- was caught)")
            # `delta = 0` is a PLACEHOLDER, not a parameter: PXP has no anisotropy knob. The
            # column exists for the MBL rows and every PXP row carries the same value, so
            # grouping the CSV by `delta` never splits a PXP curve.
            row(io; suite = "smoke", scheme = scheme, model = "pxp", sym = "none",
                cutoff = CUTOFF, kmax = MAXITER, rsvd = rsvd_tag(scheme, RSVDS[1]),
                l = Ls, w = 0.0, delta = 0.0, seed = 0, maxdim = 256, dt = dt, t = T,
                obs = fid, obs_exact = pref.obs, obs_err = abs(fid - pref.obs),
                entropy = ent, entropy_exact = pref.ent, entropy_err = abs(ent - pref.ent),
                sz_err = sz_err, energy = real(mpo_energy(psi, mpo)) / max(nrm^2, eps()),
                energy_drift = 0.0, nrm = nrm,
                maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                smin = st.smin, sgapmin = st.sgapmin)
        end
        check_order!(fails, "pxp/$scheme", errs, DTS)
        @printf("  smoke pxp none  %-6s dt-ratio=%.2f (want >=3)\n",
                scheme, errs[2] > 0 ? errs[1] / errs[2] : Inf)
    end

    if isempty(fails)
        println("  SMOKE OK")
    else
        for f in fails
            println("  SMOKE FAIL: ", f)
        end
        error("smoke phase failed ($(length(fails)) checks) -- production phases are not valid")
    end
end

# ── PHASES mbl and pxp ───────────────────────────────────────────────────────────────────────
#
# One loop, because the two differ only in the model, the symmetry, the start state and which
# scalar `obs` holds -- and an accidental difference in the step loop between them would be
# indistinguishable from a physical one.
function phase_run(io, kind::Symbol)
    nst = round(Int, TMAX / DT)
    # ⛔ THE GRID IS MODEL-SPECIFIC, AND GETTING IT WRONG IS INVISIBLE. MBL's signature is a
    # logarithm (log grid); the scar signature is PERIODIC, and a log grid aliases the revivals
    # into a smooth decay that looks exactly like thermalisation. See `revival_steps`.
    steps = kind === :pxp ? revival_steps(nst) : sample_steps(nst)
    # Above `ED_MAX` there is no exact reference and the run is graded against the paper plus
    # convergence -- see the `else` branch below.
    # ⛔ THE PXP LIMIT IS A DIMENSION, NOT A LENGTH, and using `L <= ED_MAX` for both threw away
    # most of the scar range. `ED_MAX = 20` is set by the MBL model's `C(L, L/2)` sector; the
    # Rydberg-blockade basis grows as the GOLDEN RATIO instead, so at L=28 it is 832,040 states
    # against `2^28 = 2.7e8`. Grading PXP by `L` refuses an exact reference at sizes where it is
    # cheap -- and the exact reference is exactly what makes a revival claim checkable.
    graded = kind === :mbl ? L <= ED_MAX : pxp_dim(L) <= PXP_ED_DIM_MAX
    want_entropy = L <= 20            # the ED entropy is a dense 2^{L/2} SVD -- see its docstring
    ws = kind === :mbl ? WS : (0.0,)
    # ⛔⛔ FOR PXP, `delta` IS THE Z_k INDEX, NOT AN ANISOTROPY. PXP has no anisotropy knob, so
    # the column was a constant placeholder; it now carries WHICH INITIAL STATE the row came from
    # (`k = 2` scarred, `k = 3` intermediate, `k = 0` the polarised control). That keeps the
    # start-state ladder on the existing loop and existing column, and grouping the CSV by
    # `delta` separates the curves -- which is what the placeholder comment always promised.
    # ⚠ ANY PXP ANALYSIS MUST GROUP BY `delta`. Pooling k=2 with k=0 averages a reviving curve
    # against a thermalising one and would erase the very effect being measured.
    # ⚠ `DELTAS` DEFAULTS TO `(1.0,)`, WHICH IS AN ILLEGAL `k` (all-up violates the blockade), so
    # PXP falls back to the full contrast ladder rather than inheriting the MBL default and
    # throwing. Pass ARGS[9] to choose explicitly, e.g. `2` for the scarred state alone.
    deltas = kind === :mbl ? DELTAS : (length(ARGS) >= 9 ? DELTAS : (2.0, 3.0, 0.0))

    for delta in deltas, w in ws
        # ---- the reference, once per (Δ, realisation), shared by every arm and cap ----
        hz = kind === :mbl ? disorder_fields(L, w; seed = SEED) : Float64[]
        sym = kind === :mbl ? :U1 : :none
        if graded
            if kind === :mbl
                Hs, states, idx = disordered_xxz_sparse(L, hz; delta = delta)
                v0 = neel_vector(L, idx)
                z2v = nothing
            else
                Hs, states, idx = pxp_sparse(L)
                v0 = zk_vector(L, idx, Int(delta))     # `delta` IS `k` here -- see `deltas` above
                z2v = copy(v0)
            end
            @printf("  %s L=%d W=%g D=%g seed=%d: ED dim=%d, %d samples to t=%g\n",
                    kind, L, w, delta, SEED, length(states), length(steps), nst * DT)
            flush(stdout)
            tref = time()
            ref = exact_reference(Hs, v0, steps, DT, L, states;
                                  want_entropy = want_entropy, kind = kind, z2 = z2v)
            @printf("  %s reference done in %.1fs\n", kind, time() - tref); flush(stdout)
        else
            # ⛔ NO ED ABOVE `ED_MAX`, BY CONSTRUCTION -- NOT AS A FALLBACK. The Sz=0 sector is
            # C(L, L/2): 184,756 at L=20 and 1.4e11 at L=40. There is no exact reference to be
            # had, which is exactly why arXiv:1501.05661 ran ED only for its non-interacting
            # Fig. 3 and DMRG for everything else. Large `L` is validated against THE PAPER'S
            # PUBLISHED CURVES plus convergence in (chi, cutoff); the absolute internal witnesses
            # are `energy_drift` and `norm`, which need no reference at all.
            #
            # ⚠ `obs_exact`/`obs_err`/`entropy_*` are written as NaN here. They are NOT zero and
            # NOT omitted: a 0.0 in an error column would read as perfect agreement, which is the
            # single most dangerous thing this table could claim. Any analysis must drop NaN.
            ref = nothing
            @printf("  %s L=%d W=%g D=%g seed=%d: NO ED (L > %d) -- %d samples to t=%g; graded by convergence + energy drift, obs_exact=NaN\n",
                    kind, L, w, delta, SEED, ED_MAX, length(steps), nst * DT); flush(stdout)
        end

        set_symmetry!(sym)
        mpo = kind === :mbl ? disordered_xxz_mpo(L; delta = delta, hz = hz) : pxp_mpo(L)
        psi0 = kind === :mbl ? neel_state(L) : zk_state(L, Int(delta))
        e0 = real(mpo_energy(psi0, mpo)) / max(norm(psi0)^2, eps())
        if kind === :mbl
            ecl = neel_energy_closed(L, hz; delta = delta)
            abs(e0 - ecl) < 1e-10 ||
                error("initial energy $e0 != closed form $ecl at L=$L W=$w D=$delta")
        end

        # `maxdim` outermost and ASCENDING, `cutoff` next: the cheap, always-informative cells
        # stream out first, so a job that runs out of wall clock still leaves a usable table.
        # ⚠ `rsvd` INNERMOST, so every probe setting for one (chi, cutoff, kmax, scheme) runs
        # back-to-back on the same machine state -- which is what makes their wall clocks
        # comparable to each other. `rsvd_arms` collapses to one entry for `tdvp2`, which has no
        # probe and must not be replicated across the ladder.
        for maxdim in CHIS, cutoff in CUTOFFS, kmax in KMAXES, scheme in SCHEMES,
            rsvd in rsvd_arms(scheme)
            psi = copy(psi0)
            step = stepper(scheme, mpo, maxdim; cutoff = cutoff, kmax = kmax, rsvd = rsvd)
            # ⛔ COMPILE BEFORE THE CLOCK STARTS -- see `warm!`. Without it the first arm of each
            # cell pays the specialisation and every later arm reads as faster than it is.
            warm!(step, psi0, DT)
            # The last SAMPLED values, carried out of the loop so the console line reports the
            # same numbers the CSV recorded rather than recomputing them from the final state --
            # which costs another O(L²) canonicalisation and, worse, would print a quantity taken
            # at a time no reference row exists for.
            t0 = time(); ab0 = gc_gb(); kd = 0; si = 1; n = 0; broke = false
            lastobs = NaN; lastent = NaN; lasterr = NaN; lastserr = NaN
            while n < nst
                n += 1
                try
                    kd += info_krylov(step(psi, ComplexF64(-im * DT)))
                catch e
                    @printf("  %s %s chi=%d cut=%.0e k=%d step %d THREW %s\n", kind, scheme,
                            maxdim, cutoff, kmax, n,
                            sprint(showerror, e)[1:min(end, 120)]); flush(stdout)
                    broke = true
                    break
                end
                (si <= length(steps) && steps[si] == n) || continue
                # `r === nothing` is the no-ED path: every reference-derived field becomes NaN
                # rather than 0.0, so an ungraded row can never be mistaken for a perfect one.
                r = graded ? ref[si] : nothing; si += 1
                got = [sz_expectation(psi, j) for j in 1:L]
                nrm = norm(psi)
                obs = kind === :mbl ? imbalance_from_profile(got) :
                      abs2(overlap(psi0, psi) / max(nrm, eps()))
                st = mps_spectrum_stats(psi); ent = st.ent
                en = real(mpo_energy(psi, mpo)) / max(nrm^2, eps())
                lastobs = obs; lastent = ent
                robs = r === nothing ? NaN : r.obs
                rent = r === nothing ? NaN : r.ent
                rsz  = r === nothing ? NaN : maximum(abs.(got .- r.sz))
                lasterr = abs(obs - robs); lastserr = abs(ent - rent)
                row(io; suite = String(kind), scheme = scheme, model = String(kind),
                    sym = String(sym), l = L, w = w, delta = delta, seed = SEED,
                    maxdim = maxdim, cutoff = cutoff, kmax = kmax,
                    rsvd = rsvd_tag(scheme, rsvd), dt = DT,
                    t = n * DT, obs = obs, obs_exact = robs, obs_err = abs(obs - robs),
                    entropy = ent, entropy_exact = rent, entropy_err = abs(ent - rent),
                    sz_err = rsz, energy = en,
                    energy_drift = abs(en - e0), nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                smin = st.smin, sgapmin = st.sgapmin)
            end
            @printf("  %s %-6s W=%-4g D=%g chi=%-4d cut=%-7.0e k=%-2d %-6s bd=%d obs=%.6f obs_err=%.3e S_err=%.3e %.1fs%s\n",
                    kind, scheme, w, delta, maxdim, cutoff, kmax, rsvd_tag(scheme, rsvd),
                    maximum(bond_dims(psi); init = 0),
                    lastobs, lasterr, lastserr, time() - t0,
                    broke ? "  (BROKE)" : ""); flush(stdout)
        end
    end
end

# ── PHASE hub: THE GATE FOR THE SPINFUL MODEL ────────────────────────────────────────────────
#
# `hubbard_aa_mpo` is the only route to arXiv:1501.05661's Figs. 2, 4 and 5, and it is the most
# error-prone object in this campaign: a THREE-SITE term carrying a Jordan-Wigner string, on an
# interleaved `2L`-site chain, with `U` on odd bonds only and a field that mixes `Delta` and
# `U/2`. A wrong string sign gives the wrong STATISTICS while remaining Hermitian,
# particle-number conserving and energy conserving -- the failure class that has already cost
# this project two cluster jobs.
#
# So it is graded against `hubbard_aa_sparse`, which computes its signs by COUNTING OCCUPIED
# MODES in the fermion basis and shares no code with the MPO. Three things are required:
#
#   1. `U = 0`  the MPO must reproduce the reference. This is the STRING SIGN test: at `U = 0` the
#               dynamics is free-fermionic, and the reference is separately pinned to the
#               single-particle result at 1e-16 (`hubbard_aa_check.jl`).
#   2. `U ≠ 0`  the MPO must reproduce it again. The `U` term is diagonal, so passing (1) and
#               failing (2) localises the defect to the Hubbard bond or its `U/2` field, and
#               passing both means the interacting Hamiltonian is right.
#   3. `‖ψ‖ = 1` and dt-convergence, as everywhere else in this driver.
#
# ⚠ `⟨H⟩` IS NOT COMPARED. `hubbard_aa_mpo` deliberately omits the constant
# `Σ_i Δcos(2πβi+φ) + U·L/4` that expanding `n = S^z + 1/2` produces; it is a global phase and
# invisible in every observable, but it offsets `mpo_energy` from the reference's `⟨H⟩` by exactly
# that amount. Comparing energies would report a "failure" that is a bookkeeping constant, and
# adding the constant to the MPO would spend virtual dimension to represent a phase.
hubbard_density_mps(psi, L::Int) =
    [sz_expectation(psi, 2i - 1) + sz_expectation(psi, 2i) + 1.0 for i in 1:L]

function phase_hub(io)
    T = 1.0
    DTS = (0.05, 0.025)
    fails = String[]
    set_symmetry!(:U1)

    # ⛔ `dfrac` IS PART OF THE GATE, not decoration. `cdw_species` builds the production initial
    # state, so leaving it untested would ship an unvalidated function into the one phase that
    # has no reference to catch it. `dfrac = 0.5` also makes `U` bite hardest -- doublons are the
    # only configurations `n↑n↓` acts on -- so it is the cell that actually tests the Hubbard term.
    for Lp in (4, 6), U in (0.0, 4.0), dfrac in (0.0, 0.5)
        Delta, phi, beta = 2.0, 0.35, 0.721
        species = cdw_species(Lp; doublon_frac = dfrac, seed = 1)
        N = length(species) + count(==(:both), species)
        Hs, hstates, hidx = hubbard_aa_sparse(Lp, N; J = 1.0, Delta = Delta, U = U,
                                              beta = beta, phi = phi)
        v0 = hubbard_cdw_vector(Lp, hstates, hidx; species = species)
        vT = expv_sparse(Hs, ComplexF64(-im * T), v0)
        nex = hubbard_density(vT, Lp, hstates)
        Iex = hubbard_imbalance(nex)

        mpo = hubbard_aa_mpo(Lp; J = 1.0, Delta = Delta, U = U, beta = beta, phi = phi)
        psi0 = hubbard_cdw_state(Lp; species = species)
        n0 = hubbard_density_mps(psi0, Lp)
        @printf("  hub L=%d U=%g dfrac=%g (N=%d, %d doublons): ED dim=%d, I(0)=%.10f (want 1), MPS n(0) vs CDW err=%.2e\n",
                Lp, U, dfrac, N, count(==(:both), species), length(hstates),
                hubbard_imbalance(n0),
                maximum(abs.(n0 .- hubbard_density(v0, Lp, hstates))))
        maximum(abs.(n0 .- hubbard_density(v0, Lp, hstates))) < 1e-12 ||
            push!(fails, "hub L=$Lp U=$U dfrac=$dfrac: initial MPS density != ED initial density")

        for scheme in SCHEMES
            errs = Float64[]
            for dt in DTS
                nst = round(Int, T / dt)
                psi = copy(psi0)
                # 4^Lp is the full rank bound; 1024 never binds at Lp <= 6.
                step = stepper(scheme, mpo, 1024)
                t0 = time(); ab0 = gc_gb(); kd = 0
                for _ in 1:nst
                    kd += info_krylov(step(psi, ComplexF64(-im * dt)))
                end
                nrm = norm(psi)
                nmps = hubbard_density_mps(psi, Lp)
                derr = maximum(abs.(nmps .- nex))
                push!(errs, derr)
                Imps = hubbard_imbalance(nmps)
                st = mps_spectrum_stats(psi)
                @printf("  hub %-6s L=%d U=%-4g d=%g dt=%-6g dn=%.3e  I=%.6f (exact %.6f)  |psi|=%.10f\n",
                        scheme, Lp, U, dfrac, dt, derr, Imps, Iex, nrm)
                abs(nrm - 1) < 1e-8 ||
                    push!(fails, "hub/$scheme L=$Lp U=$U d=$dfrac dt=$dt NORM=$nrm -- " *
                                 "non-Hermitian MPO (a wrong Jordan-Wigner string sign shows up here)")
                # `seed` carries the doublon fraction so the two cells are distinguishable in the
                # CSV; they are otherwise identical in every recorded column.
                row(io; suite = "hub", scheme = scheme, model = "hub", sym = "U1",
                    cutoff = CUTOFF, kmax = MAXITER, rsvd = rsvd_tag(scheme, RSVDS[1]),
                    l = Lp, w = Delta, delta = U, seed = round(Int, 100dfrac),
                    maxdim = 1024, dt = dt, t = T,
                    obs = Imps, obs_exact = Iex, obs_err = abs(Imps - Iex),
                    entropy = st.ent, entropy_exact = NaN, entropy_err = NaN,
                    sz_err = derr,
                    energy = real(mpo_energy(psi, mpo)) / max(nrm^2, eps()),
                    energy_drift = NaN, nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                    smin = st.smin, sgapmin = st.sgapmin)
            end
            check_order!(fails, "hub/$scheme/L=$Lp/U=$U/d=$dfrac", errs, DTS)
            @printf("  hub %-6s L=%d U=%-4g d=%g dt-ratio=%.2f (want >=3)\n",
                    scheme, Lp, U, dfrac, errs[2] > 0 ? errs[1] / errs[2] : Inf)
        end
    end

    if isempty(fails)
        println("  HUB GATE OK -- the spinful MPO reproduces the fermion-basis reference")
    else
        for f in fails
            println("  HUB FAIL: ", f)
        end
        error("hub gate failed ($(length(fails)) checks)")
    end
end

# ── PHASE hub4: THE GATE FOR THE FUSED d = 4 SPINFUL MODEL ───────────────────────────────────
#
# THE SAME PHYSICS AS `phase_hub`, ON A LAYOUT THAT CAN ACTUALLY BE EVOLVED. `hubbard_aa_mpo`
# interleaves ↑₁↓₁↑₂↓₂… on `2L` sites, which makes every hop RANGE-2 -- and a range-2
# Hamiltonian has no term inside any two-site block, so `H_eff ≡ 0` on every bond and the CDW
# start state is an EXACT FIXED POINT of bug, tdvp2 and cbe1s alike (job 16339405: distance-1
# gives bond=5 and evolves, distance-2 gives bond=1 and freezes). `hubbard_aa_mpo_d4` puts one
# physical site on one MPS site, so both species' hops are range-1.
#
# ⛔⛔ `maxbond` IS THE PRIMARY CHECK HERE, AND IT IS THE ONLY ONE THAT COULD HAVE CAUGHT THE
# FREEZE. On the interleaved model `‖ψ‖ = 1`, `energy_drift = 0`, `dt-ratio = 1.00` and
# `⟨H⟩` responded correctly to `U` -- every one of those passed on a state that never moved,
# because the diagonal terms were present and only the kinetic term was inert. A product state
# whose bond dimension is still 1 after a step has not been evolved, whatever else agrees.
#
# THREE INDEPENDENT WITNESSES, in increasing strength:
#
#   1. `maxbond > 1` after ONE step               -- the state moved at all
#   2. `⟨H⟩` vs the reference's `⟨v|H|v⟩`         -- the DIAGONAL part (field + U) is right
#   3. the density profile at `T` vs exact `expv` -- the JORDAN-WIGNER SIGNS are right
#
# ⚠ (2) IS AVAILABLE HERE AND WAS NOT IN `phase_hub`. That gate could not compare energies
# because `hubbard_aa_mpo` drops the constant from expanding `n = S^z + ½`; this MPO is written
# in `n` directly, so no constant is dropped and the two energies must agree ABSOLUTELY. It is a
# non-vacuous check: the CDW state has `⟨H⟩ = Σ_i h_i n_i + U·(#doublons) ≠ 0`, unlike the pure
# hopping expectation that made `⟨H⟩ = 0` vacuous on a product state.
#
# ⚠ AND (3) IS THE ONE THAT CANNOT BE FAKED. A wrong string sign leaves `H` Hermitian and
# particle-number conserving, so it passes (1) and (2) and produces a perfectly smooth, entirely
# wrong imbalance decay. `dfrac = 0.5` is carried because doublons are the only configurations
# `n↑n↓` acts on, so it is the cell where `U` is not inert.
hub4_occupations(Lp::Int, species::Vector{Symbol}) = begin
    occ = fill(:empty, Lp)
    for (k, i) in enumerate(2:2:Lp)
        occ[i] = species[k]
    end
    occ
end

function phase_hub4(io)
    T = 1.0
    DTS = (0.05, 0.025)
    fails = String[]
    # `:none` DELIBERATELY, AND FIRST. The d=4 space has a (:U1 charge x :U1 spin) form, but the
    # fermionic sign is what is under test and a symmetry backend is one more thing that could
    # absorb a sign error. Under `:none` the operators are plain 4x4 blocks and the only thing
    # between the MPO and the reference is the algebra. The symmetric arm is gated against THIS
    # one, once this one passes.
    set_symmetry!(:none)

    for Lp in (4, 6), U in (0.0, 4.0), dfrac in (0.0, 0.5)
        Delta, phi, beta = 2.0, 0.35, 532 / 738
        species = cdw_species(Lp; doublon_frac = dfrac, seed = 1)
        N = length(species) + count(==(:both), species)
        Hs, hstates, hidx = hubbard_aa_sparse(Lp, N; J = 1.0, Delta = Delta, U = U,
                                              beta = beta, phi = phi)
        v0 = hubbard_cdw_vector(Lp, hstates, hidx; species = species)
        vT = expv_sparse(Hs, ComplexF64(-im * T), v0)
        nex  = hubbard_density(vT, Lp, hstates)
        Iex  = hubbard_imbalance(nex)
        Eex  = real(dot(v0, Hs * v0))              # the reference's ⟨H⟩ on the START state

        occ  = hub4_occupations(Lp, species)
        mpo  = hubbard_aa_mpo_d4(Lp; J = 1.0, Delta = Delta, U = U, beta = beta, phi = phi)
        psi0 = hub4_product_state(occ)
        n0   = [hub4_density(psi0, i) for i in 1:Lp]
        nref0 = hubbard_density(v0, Lp, hstates)
        E0   = real(mpo_energy(psi0, mpo)) / max(norm(psi0)^2, eps())

        @printf("  hub4 L=%d U=%-4g dfrac=%g (N=%d, %d dbl): w=%d  I(0)=%.10f (want 1)  n(0) err=%.2e  E(0)=%.10f vs %.10f (err %.2e)\n",
                Lp, U, dfrac, N, count(==(:both), species),
                maximum(mpo_virtual_dims(mpo); init = 0),
                hubbard_imbalance(n0), maximum(abs.(n0 .- nref0)),
                E0, Eex, abs(E0 - Eex)); flush(stdout)

        maximum(abs.(n0 .- nref0)) < 1e-12 ||
            push!(fails, "hub4 L=$Lp U=$U d=$dfrac: initial d=4 density != ED initial density")
        abs(E0 - Eex) < 1e-10 ||
            push!(fails, "hub4 L=$Lp U=$U d=$dfrac: E(0)=$E0 != reference $Eex -- the " *
                         "DIAGONAL part (field or U) of the d=4 MPO is wrong")

        # Every scheme's error at the COARSE dt, so the operator can be judged on the best of
        # them rather than on the worst -- see the note at the bottom of the scheme loop.
        cell_best = Float64[]

        # ⛔ `kmax` IS AN AXIS HERE, NOT A CONSTANT, and it is the FIRST thing to vary before
        # concluding anything about BUG's local error. It now sets the half-sweep basis depth as
        # well as the root solve (see `stepper`), so this ladder is a genuine
        # how-many-Krylov-vectors scan rather than a root-solve-only one. Default `(MAXITER,)`
        # keeps a single rung unless ARGS[11] asks for more.
        for scheme in SCHEMES, kmax in KMAXES
            errs = Float64[]
            for dt in DTS
                nst = round(Int, T / dt)
                psi = copy(psi0)
                # 4^Lp bounds the rank; 1024 never binds at Lp <= 6
                step = stepper(scheme, mpo, 1024; kmax = kmax)
                t0 = time(); ab0 = gc_gb(); kd = 0
                bd1 = 0
                for k in 1:nst
                    kd += info_krylov(step(psi, ComplexF64(-im * dt)))
                    # ⛔ CAPTURED AFTER THE FIRST STEP, NOT AT THE END. This is the freeze
                    # detector, and a state that is an exact fixed point stays at bond 1 forever
                    # -- so reading it at `T` would work too, but reading it at step 1 says
                    # WHEN it failed and costs nothing.
                    k == 1 && (bd1 = maximum(bond_dims(psi); init = 0))
                end
                nrm  = norm(psi)
                nmps = [hub4_density(psi, i) for i in 1:Lp]
                derr = maximum(abs.(nmps .- nex))
                push!(errs, derr)
                # ⚠ `hubbard_imbalance`, the REFERENCE's OWN profile -> imbalance formula, is
                # used for BOTH sides here -- not `hub4_imbalance`, which takes a state. That is
                # deliberate and not a shortcut: the quantity under test is the DENSITY PROFILE
                # (compared directly as `derr`), and sharing the one-line (even-odd)/(even+odd)
                # reduction removes any chance of the two sides disagreeing about the observable
                # rather than about the physics. `hub4_imbalance` takes a `SymMPS` and would need
                # a second canonicalisation to say the same thing.
                Imps = hubbard_imbalance(nmps)
                st = mps_spectrum_stats(psi)
                @printf("  hub4 %-6s L=%d U=%-4g d=%g m=%-3d dt=%-6g bd1=%-3d bd=%-4d dn=%.3e  I=%.6f (exact %.6f)  |psi|=%.10f\n",
                        scheme, Lp, U, dfrac, kmax, dt, bd1,
                        maximum(bond_dims(psi); init = 0), derr, Imps, Iex, nrm); flush(stdout)

                bd1 > 1 || push!(fails,
                    "hub4/$scheme L=$Lp U=$U d=$dfrac m=$kmax dt=$dt FROZEN: maxbond=$bd1 after one " *
                    "step -- the CDW product state did not move, i.e. H_eff == 0 on every " *
                    "bond. This is the range-2 signature and it means the layout, not the MPO.")
                abs(nrm - 1) < 1e-8 ||
                    push!(fails, "hub4/$scheme L=$Lp U=$U d=$dfrac m=$kmax dt=$dt NORM=$nrm -- " *
                                 "non-Hermitian MPO (a wrong Jordan-Wigner sign lands here)")
                row(io; suite = "hub4", scheme = scheme, model = "hub4", sym = "none",
                    cutoff = CUTOFF, kmax = kmax, rsvd = rsvd_tag(scheme, RSVDS[1]),
                    l = Lp, w = Delta, delta = U, seed = round(Int, 100dfrac),
                    maxdim = 1024, dt = dt, t = T,
                    obs = Imps, obs_exact = Iex, obs_err = abs(Imps - Iex),
                    entropy = st.ent, entropy_exact = NaN, entropy_err = NaN,
                    sz_err = derr,
                    energy = real(mpo_energy(psi, mpo)) / max(nrm^2, eps()),
                    energy_drift = abs(real(mpo_energy(psi, mpo)) / max(nrm^2, eps()) - Eex),
                    nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                    smin = st.smin, sgapmin = st.sgapmin)
            end
            # ⛔⛔ `check_order!` IS DELIBERATELY NOT USED HERE, AND THIS IS NOT A RELAXED GATE.
            # Its two rules are "e1 < 1e-3 or the H is wrong" and "ratio >= 3 or it is not second
            # order", and BOTH misattribute on this model:
            #
            #   * MEASURED (job 16341314): on the SAME MPO in the SAME cell, `bugmid` reaches
            #     8.4e-15 and `tdvp2` 9.2e-07 while plain `bug` sits at 1.3e-02. An MPO that lets
            #     one scheme hit MACHINE PRECISION cannot be wrong, so reporting "far too large
            #     for a correct H" against `bug` names the wrong culprit -- and that message is
            #     exactly the kind of false lead that cost this campaign four jobs already.
            #   * `bug` is FIRST order on the doublon cells (ratio 1.98). Demanding ~4 from a
            #     first-order scheme fails it for being what it is.
            #
            # So the two questions are SEPARATED: "is the operator right?" is a property of the
            # CELL and is judged after every scheme has run (below); "does this integrator
            # converge?" is a property of the SCHEME and is judged on its own convergence, not
            # against an assumed order. Nothing is loosened -- the MPO check is now STRICTER,
            # because it demands that some scheme actually reach the reference.
            r = errs[2] > 0 ? errs[1] / errs[2] : Inf
            push!(cell_best, errs[1])
            # A scheme must CONVERGE. Not converging at all (ratio < 1.5 while the error is well
            # above the floor) is a real defect whatever the nominal order.
            (errs[1] > 1e-10 && r < 1.5) && push!(fails,
                "hub4/$scheme/L=$Lp/U=$U/d=$dfrac/m=$kmax DOES NOT CONVERGE: halving dt changed the " *
                "error only $(round(r; digits = 2))x -- errs=$errs at dts=$DTS")
            @printf("  hub4 %-6s L=%d U=%-4g d=%g m=%-3d dt-ratio=%.2f (order %.2f)%s\n",
                    scheme, Lp, U, dfrac, kmax, r,
                    (r > 0 && isfinite(r)) ? log2(r) : NaN,
                    errs[1] <= 1e-10 ? "  [at the floor: ratio is roundoff, not order]" : "")
            flush(stdout)
        end

        # ── IS THE OPERATOR RIGHT? A PROPERTY OF THE CELL, JUDGED ON THE BEST SCHEME ─────────
        #
        # This is the STRICTER form of the old "e1 < 1e-3" rule, not a weaker one: it demands
        # that SOME integrator actually reproduce the independent fermion-basis reference to
        # 1e-5, rather than merely that no integrator be catastrophically bad. A wrong MPO
        # cannot be rescued by a better time-stepper, so if every scheme misses, the operator is
        # the common cause; if one scheme hits machine precision on it, the operator is right and
        # a large error elsewhere is that SCHEME's local error.
        #
        # MEASURED in this exact cell class (job 16341314): bugmid 8.4e-15, tdvp2 9.2e-07, bug
        # 1.3e-02 -- one operator, three very different errors.
        bestc = isempty(cell_best) ? Inf : minimum(cell_best)
        bestc < 1e-5 || push!(fails,
            "hub4 L=$Lp U=$U d=$dfrac: NO scheme reproduced the reference (best density error " *
            "$(bestc) at dt=$(DTS[1])) -- with E(0) and n(0) already exact this points at the " *
            "HOPPING, i.e. a Jordan-Wigner sign, rather than at any one integrator")
        @printf("  hub4 L=%d U=%-4g d=%g  MPO verdict: best scheme reached %.3e (want < 1e-5)\n",
                Lp, U, dfrac, bestc); flush(stdout)
    end

    if isempty(fails)
        println("  HUB4 GATE OK -- the FUSED d=4 spinful MPO reproduces the fermion-basis " *
                "reference AND the state actually moves")
    else
        for f in fails
            println("  HUB4 FAIL: ", f)
        end
        error("hub4 gate failed ($(length(fails)) checks)")
    end
end

# ── PHASE hubdiag: WHY IS THE SPINFUL STATE FROZEN? ─────────────────────────────────────────
#
# Job 16335193 produced `I = 1.000000` at every time, every dt and every scheme, with `bond = 1`
# and a NON-zero energy that responded correctly to `U`. That is a state that never moves: the
# diagonal terms (field, `U`) are present and the HOPPING is absent, so the CDW product state is
# an eigenstate. `dt-ratio = 1.00` exactly -- identical error at both step sizes -- is the tell.
#
# This phase isolates where the hopping is lost, and it is built around ONE discriminator:
# `hubbard_aa_mpo` runs under BOTH `:none` and `:U1`, and under `:none` the hopping channel is
# charge-NEUTRAL, so `_charged_transport` short-circuits to a plain `σ^z` block and never touches
# the charged construction. Therefore:
#
#   * evolves under `:none`, frozen under `:U1`  ⇒  the defect is in the CHARGED transport path
#   * frozen under BOTH                          ⇒  the defect is in the automaton wiring itself
#     (the 0-coefficient blocks, the boundary reshapes, or the row/column indexing)
#
# `bond` and `E` are printed per step: a working hopping term takes `bond` above 1 immediately,
# and that single integer answers the question faster than any norm or imbalance.
function phase_hubdiag(io)
    Lp, dt, nst = 4, 0.05, 20
    M = 2Lp
    q = local_space()
    sigz = to_concrete(2.0 * q.Sz)
    Jnn  = zeros(Float64, M, M); for m in 1:(M - 1); Jnn[m, m + 1] = 1.0; end
    Jnnn = zeros(Float64, M, M); for m in 1:(M - 2); Jnnn[m, m + 2] = 1.0; end

    # ⛔ TWO VERTEX CONSTRUCTIONS, AND THEY ARE NOT INTERCHANGEABLE.
    #
    # `vadj` is what I wrote: `(Sm, Sm')` via the adjoint, in BOTH symmetries.
    # `vcan` is what this repo does everywhere else: `pair_vertices(xxz_chain(...))`, whose
    # `:none` branch writes `XXZTerm(q.Sp, q.Sm, -J)` EXPLICITLY and uses `'` only in the rank-3
    # (`:U1`) branch. Under `:none` the adjoint transposes the site legs, so `vadj` is malformed
    # precisely where `xxz_chain` is careful not to be -- and `spin_vertices`, which every working
    # `pair_mpo` model routes through, IS `pair_vertices(xxz_chain(...))`. `vadj` is therefore the
    # one ingredient here that no validated model uses.
    #
    # `bond > 1` after 20 steps is the ONLY witness. `E0` is VACUOUS for pure hopping: it is zero
    # on a product state whether the kinetic term is present or the MPO is identically zero, which
    # is how a kinetically-empty MPO passed every energy check so far.
    for sym in (:none, :U1)
        set_symmetry!(sym)
        # Built inside the loop: `xxz_chain` branches on `length(q.Sp.inds)`, so the canonical
        # vertices genuinely differ between `:none` (rank 2) and `:U1` (rank 3).
        vcan = pair_vertices(xxz_chain(M; J = 1.0, delta = 0.0))
        # `vadj` (hand-built adjoint vertices) is GONE, not commented out: job 16339352 proved
        # it OOMs inside `pair_mpo` at 32G, which kills the whole diagnostic before the later arms
        # report. The finding is recorded in `hubbard_aa_mpo`; re-running it costs a job to learn
        # nothing new.
        arms = Any[
          ("1  NN aubry_andre (known good)",     () -> aubry_andre_xxz_mpo(M; J = 1.0, Delta = 0.0)),
          ("2a NN pair_mpo CANONICAL verts",     () -> pair_mpo(M, Jnn,  vcan)),
          ("3  NNN pair_mpo CANONICAL, I trans", () -> pair_mpo(M, Jnnn, vcan)),
          ("4  NNN pair_mpo CANONICAL, sigz",    () -> pair_mpo(M, Jnnn, vcan; Itrans = sigz)),
        ]
        species = cdw_species(Lp; doublon_frac = 0.0, seed = 1)
        for (label, build) in arms
            # Progress printed BEFORE the build: job 16339259 was OOM-killed inside a build and
            # the log said nothing at all, because every number came after the 20 steps.
            @printf("  hubdiag %-5s %-38s building...
", sym, label); flush(stdout)
            mpo = try
                build()
            catch e
                @printf("  hubdiag %-5s %-38s THREW %s
", sym, label,
                        sprint(showerror, e)[1:min(end, 140)]); flush(stdout)
                continue
            end
            psi = hubbard_cdw_state(Lp; species = species)
            step = stepper("bug", mpo, 256)
            for _ in 1:nst
                step(psi, ComplexF64(-im * dt))
            end
            nprof = hubbard_density_mps(psi, Lp)
            @printf("  hubdiag %-5s %-38s w=%-3d bond=%-4d |psi|=%.8f I=%+.6f n=%s
",
                    sym, label, maximum(mpo_virtual_dims(mpo)),
                    maximum(bond_dims(psi); init = 0), norm(psi),
                    hubbard_imbalance(nprof), string(round.(nprof; digits = 3)))
            flush(stdout)
        end
    end
end

# ── PHASE hubprod: arXiv:1501.05661 FIGURES 2, 4, 5 AT THE PAPER'S OWN SIZE ──────────────────
#
# The SAME spinful MPO the `hub` gate validates, run at the paper's working point: L = 40
# physical sites (2L = 80 spin sites interleaved), m_max = 100, out to 50 tau.
#
# ⛔ THERE IS NO EXACT REFERENCE HERE AND THERE CANNOT BE. The interacting spinful sector is
# C(2L, N); at L = 40, N = 20 that is C(80,20) ~ 3.5e18. This is precisely why the paper ran ED
# only for its non-interacting Fig. 3 and DMRG for everything else, and it is why every
# reference-derived column below is NaN rather than 0.0.
#
# WHAT VALIDATES A ROW, in descending order of strength:
#   1. THE PAPER'S PUBLISHED CURVES -- Fig. 2 (I vs t at several Delta), Fig. 4 (stationary I
#      over the (U, Delta) plane), Fig. 5 (entropy growth). This is the anchor; the recorded rule
#      is that a model with no analytic solution needs a PUBLISHED one, never our own finest grid.
#   2. CONVERGENCE in `chi` and `cutoff`. Two arms that disagree are both suspect; an arm that
#      stops moving as `chi` grows and `cutoff` falls is converged -- and BUG vs TDVP2 agreeing
#      is meaningful precisely because they fail differently.
#   3. `energy_drift` and `norm`, which need no reference at all. The Hamiltonian is
#      time-independent, so energy is conserved exactly and any drift is integrator error;
#      `‖ψ‖ = 1` is what a wrong Jordan-Wigner string sign breaks first.
#
# `WS` is the Delta/J ladder and `DELTAS` the U/J ladder, matching `phase_aa`'s reuse of `WS`.
function phase_hubprod(io)
    nst = round(Int, TMAX / DT)
    steps, stat_steps = stat_step_set(nst)
    TMAX + 1e-9 < STAT_T1 && @printf("  hubprod ⚠ TMAX=%g < %g: I_stat is a TRUNCATED window, NOT the paper's\n",
                                     TMAX, STAT_T1)
    set_symmetry!(:U1)
    phi, beta = aa_phase_of_seed(SEED), 0.721
    # ⛔ BOTH SPECIES, AND A DOUBLON FRACTION. An all-`:up` state is single-species: no double
    # occupancy is ever possible, `U` is exactly inert, and the run silently measures the
    # NON-INTERACTING model however large `U` is set.
    #
    # ⛔ ZERO DOUBLONS IS ALSO WRONG FOR FIG. 4. The experiment's initial CDW carries a doublon
    # fraction of "≃34(2)%", and the doublon fraction is precisely what decides how much `U` can
    # do. Comparing a 0%-doublon simulation to Fig. 4 is not a like-for-like comparison, and the
    # discrepancy would look like an integrator or convergence problem rather than a state
    # preparation one. `DOUBLON_FRAC` (ARGS[12]) defaults to the paper's value.
    species = cdw_species(L; doublon_frac = DOUBLON_FRAC, seed = SEED)
    nd = count(==(:both), species)
    @printf("  hubprod start: %d even sites, %d doublons (%.1f%%, paper 34(2)%%), N=%d atoms\n",
            length(species), nd, 100nd / max(length(species), 1), length(species) + nd)

    for Delta in WS, U in DELTAS
        mpo  = hubbard_aa_mpo(L; J = 1.0, Delta = Delta, U = U, beta = beta, phi = phi)
        psi0 = hubbard_cdw_state(L; species = species)
        nrm0 = norm(psi0)
        e0   = real(mpo_energy(psi0, mpo)) / max(nrm0^2, eps())
        I0   = hubbard_imbalance(hubbard_density_mps(psi0, L))
        @printf("  hubprod L=%d (2L=%d sites) D/J=%g U/J=%g phi=%.4f: I(0)=%.10f (want 1), E(0)=%.6f, %d samples to t=%g\n",
                L, 2L, Delta, U, phi, I0, e0, length(steps), nst * DT); flush(stdout)
        abs(I0 - 1.0) < 1e-10 ||
            error("hubprod initial imbalance $I0 != 1: the CDW start state is wrong")

        # ⚠ `rsvd` INNERMOST, so every probe setting for one (chi, cutoff, kmax, scheme) runs
        # back-to-back on the same machine state -- which is what makes their wall clocks
        # comparable to each other. `rsvd_arms` collapses to one entry for `tdvp2`, which has no
        # probe and must not be replicated across the ladder.
        for maxdim in CHIS, cutoff in CUTOFFS, kmax in KMAXES, scheme in SCHEMES,
            rsvd in rsvd_arms(scheme)
            psi = copy(psi0)
            step = stepper(scheme, mpo, maxdim; cutoff = cutoff, kmax = kmax, rsvd = rsvd)
            # ⛔ COMPILE BEFORE THE CLOCK STARTS -- see `warm!`. Without it the first arm of each
            # cell pays the specialisation and every later arm reads as faster than it is.
            warm!(step, psi0, DT)
            t0 = time(); ab0 = gc_gb(); kd = 0; si = 1; n = 0; broke = false
            lastI = NaN; lastdrift = NaN; statI = Float64[]
            while n < nst
                n += 1
                try
                    kd += info_krylov(step(psi, ComplexF64(-im * DT)))
                catch e
                    @printf("  hubprod %s chi=%d cut=%.0e k=%d step %d THREW %s\n", scheme,
                            maxdim, cutoff, kmax, n,
                            sprint(showerror, e)[1:min(end, 120)]); flush(stdout)
                    broke = true
                    break
                end
                (si <= length(steps) && steps[si] == n) || continue
                si += 1
                nrm = norm(psi)
                nprof = hubbard_density_mps(psi, L)
                Imb = hubbard_imbalance(nprof)
                st = mps_spectrum_stats(psi)
                en = real(mpo_energy(psi, mpo)) / max(nrm^2, eps())
                lastI = Imb; lastdrift = abs(en - e0)
                n in stat_steps && push!(statI, Imb)
                row(io; suite = "hubprod", scheme = scheme, model = "hub", sym = "U1",
                    l = L, w = Delta, delta = U, seed = SEED, maxdim = maxdim,
                    cutoff = cutoff, kmax = kmax, rsvd = rsvd_tag(scheme, rsvd),
                    dt = DT, t = n * DT,
                    obs = Imb, obs_exact = NaN, obs_err = NaN,
                    entropy = st.ent, entropy_exact = NaN, entropy_err = NaN,
                    sz_err = NaN, energy = en, energy_drift = abs(en - e0), nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                    smin = st.smin, sgapmin = st.sgapmin)
            end
            Ibar = isempty(statI) ? NaN : sum(statI) / length(statI)
            @printf("  hubprod %-6s D/J=%-5g U/J=%-5g chi=%-4d cut=%-7.0e k=%-2d %-6s bd=%-4d I(T)=%.6f I_stat=%.6f drift=%.3e %.1fs%s\n",
                    scheme, Delta, U, maxdim, cutoff, kmax, rsvd_tag(scheme, rsvd),
                    maximum(bond_dims(psi); init = 0), lastI, Ibar, lastdrift,
                    time() - t0, broke ? "  (BROKE)" : ""); flush(stdout)
        end
    end
end

# ── PHASE hub4prod: THE INTERACTING SPINFUL PRODUCTION RUN ───────────────────────────────────
#
# arXiv:1501.05661's Figs. 2, 4 and 5 at the paper's own working point (L = 40, m_max = 100,
# dt = 0.1, 50 tau) and beyond, on the FUSED d = 4 layout that can actually be evolved. This is
# the phase the campaign exists to run; everything else is either a control or a gate for it.
#
# ⛔⛔ NEVER RUN BEFORE `hub4` HAS PASSED ON THE SAME TREE. There is NO reference here -- `L = 40`
# is `C(80,40)`-dimensional and ED is a wall, not a slow path -- so `obs_exact` and `obs_err` are
# `NaN` BY CONSTRUCTION. An MPO defect therefore cannot be caught in this phase at all: it
# produces a smooth, plausible, wrong imbalance decay. That is not hypothetical, it is what the
# interleaved layout did for four jobs.
#
# THE THREE THINGS THAT DO VALIDATE A ROW HERE, none of which needs a reference:
#
#   1. `maxbond` -- and it is FIRST for a reason. A frozen product state reports `I = 1.000000`
#      at every time with perfect norm and zero energy drift. `bd1` is captured after ONE step
#      and the arm is ABANDONED if it is still 1, rather than spending the wall clock producing
#      500 rows of a state that never moved.
#   2. `energy_drift` -- `H` is time-independent, so `⟨H⟩` is conserved exactly and any drift is
#      integrator error. ⚠ THIS IS A GENUINE WITNESS HERE AND WAS VACUOUS FOR THE SPINLESS
#      PRODUCT STATE: hopping has zero expectation on any product state, so `⟨H⟩ = 0` held there
#      whether the kinetic term was present or absent. The CDW state on this model has
#      `⟨H⟩ = Σ_i h_i n_i + U·(#doublons) ≠ 0`, so drift is measured against something real.
#   3. convergence in `chi` and `cutoff` -- the ladders, compared against each other rather than
#      against our own finest grid being relabelled as exact.
#
# ⛔ AND THE PUBLISHED CURVE IS THE ANCHOR, NOT OUR FINEST RUN. The recorded rule for a model
# with no analytic solution is that it needs a PUBLISHED anchor; here that is Fig. 4's stationary
# imbalance versus `U/J`, averaged over the paper's 15-30 tau window (`STAT_T0`/`STAT_T1`), from a
# CDW with the experiment's ~34% doublon fraction.
function phase_hub4prod(io)
    nst = round(Int, TMAX / DT)
    steps, stat_steps = stat_step_set(nst)
    TMAX + 1e-9 < STAT_T1 && @printf("  hub4prod ⚠ TMAX=%g < %g: I_stat is a TRUNCATED window, NOT the paper's\n",
                                     TMAX, STAT_T1)
    # `:none` until the (:U1 charge x :U1 spin) arm is gated against it. The d=4 space has that
    # form, but shipping a symmetric production run whose only validation is a symmetric gate
    # would put two unproven things in series.
    set_symmetry!(:none)
    phi, beta = aa_phase_of_seed(SEED), 532 / 738

    # ⛔ BOTH SPECIES, AND A DOUBLON FRACTION. An all-`:up` state is single-species: no double
    # occupancy is possible, `U` is exactly inert, and the run silently measures the
    # NON-INTERACTING model however large `U` is set. Zero doublons is equally wrong for Fig. 4 --
    # the doublon fraction is precisely what decides how much `U` can do, and comparing a
    # 0%-doublon simulation to that figure is not like-for-like. `DOUBLON_FRAC` (ARGS[12])
    # defaults to the paper's 34%.
    species = cdw_species(L; doublon_frac = DOUBLON_FRAC, seed = SEED)
    ndbl = count(==(:both), species)
    occ  = hub4_occupations(L, species)
    @printf("  hub4prod L=%d (d=4, %d MPS sites) start: %d even sites, %d doublons (%.1f%%, paper 34(2)%%), N=%d atoms\n",
            L, L, length(species), ndbl, 100ndbl / max(length(species), 1),
            length(species) + ndbl); flush(stdout)

    for Delta in WS, U in DELTAS
        mpo  = hubbard_aa_mpo_d4(L; J = 1.0, Delta = Delta, U = U, beta = beta, phi = phi)
        psi0 = hub4_product_state(occ)
        nrm0 = norm(psi0)
        n0   = [hub4_density(psi0, i) for i in 1:L]
        e0   = real(mpo_energy(psi0, mpo)) / max(nrm0^2, eps())
        I0   = hubbard_imbalance(n0)
        @printf("  hub4prod D/J=%-5g U/J=%-5g phi=%.4f w=%d: I(0)=%.10f (want 1), E(0)=%.6f, N=%.4f, %d samples to t=%g\n",
                Delta, U, phi, maximum(mpo_virtual_dims(mpo); init = 0), I0, e0,
                sum(n0), length(steps), nst * DT); flush(stdout)
        abs(I0 - 1) < 1e-10 ||
            error("hub4prod: I(0)=$I0 != 1 -- the CDW start state is not the experiment's")

        for maxdim in CHIS, cutoff in CUTOFFS, kmax in KMAXES, scheme in SCHEMES,
            rsvd in rsvd_arms(scheme)
            psi = copy(psi0)
            step = stepper(scheme, mpo, maxdim; cutoff = cutoff, kmax = kmax, rsvd = rsvd)
            warm!(step, psi0, DT)              # compile before the clock -- see `warm!`
            t0 = time(); ab0 = gc_gb(); kd = 0; si = 1; n = 0; broke = false
            lastI = NaN; lastdrift = NaN; bd1 = 0; statI = Float64[]
            while n < nst
                n += 1
                try
                    kd += info_krylov(step(psi, ComplexF64(-im * DT)))
                catch e
                    @printf("  hub4prod %s D/J=%g U/J=%g chi=%d cut=%.0e k=%d %s step %d THREW %s\n",
                            scheme, Delta, U, maxdim, cutoff, kmax, rsvd, n,
                            sprint(showerror, e)[1:min(end, 120)]); flush(stdout)
                    broke = true
                    break
                end
                if n == 1
                    bd1 = maximum(bond_dims(psi); init = 0)
                    # ⛔ ABANDON THE ARM, DO NOT RECORD 500 ROWS OF A FIXED POINT. A range-2
                    # layout freezes the state exactly, and every other column then looks
                    # perfect: I = 1.000000, |psi| = 1, drift = 0. Spending the wall clock and
                    # writing the rows would produce a table that reads as a physics result.
                    if bd1 <= 1
                        @printf("  hub4prod %s D/J=%g U/J=%g chi=%d cut=%.0e k=%d %s FROZEN after one step (maxbond=%d) -- ABANDONING ARM\n",
                                scheme, Delta, U, maxdim, cutoff, kmax, rsvd, bd1)
                        flush(stdout)
                        broke = true
                        break
                    end
                end
                (si <= length(steps) && steps[si] == n) || continue
                si += 1
                nrm  = norm(psi)
                ns   = [hub4_density(psi, j) for j in 1:L]
                Imb  = hubbard_imbalance(ns)
                en   = real(mpo_energy(psi, mpo)) / max(nrm^2, eps())
                st   = mps_spectrum_stats(psi)
                lastI = Imb; lastdrift = abs(en - e0)
                n in stat_steps && push!(statI, Imb)
                row(io; suite = "hub4prod", scheme = scheme, model = "hub4", sym = "none",
                    l = L, w = Delta, delta = U, seed = SEED, maxdim = maxdim,
                    cutoff = cutoff, kmax = kmax, rsvd = rsvd_tag(scheme, rsvd),
                    dt = DT, t = n * DT,
                    obs = Imb, obs_exact = NaN, obs_err = NaN,
                    entropy = st.ent, entropy_exact = NaN, entropy_err = NaN,
                    sz_err = NaN, energy = en, energy_drift = abs(en - e0), nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                    smin = st.smin, sgapmin = st.sgapmin)
            end
            Ibar = isempty(statI) ? NaN : sum(statI) / length(statI)
            @printf("  hub4prod %-6s D/J=%-5g U/J=%-5g chi=%-4d cut=%-7.0e k=%-2d %-6s bd1=%-3d bd=%-4d I(T)=%.6f I_stat=%.6f drift=%.3e %.1fs%s\n",
                    scheme, Delta, U, maxdim, cutoff, kmax, rsvd_tag(scheme, rsvd), bd1,
                    maximum(bond_dims(psi); init = 0), lastI, Ibar, lastdrift,
                    time() - t0, broke ? "  (BROKE)" : ""); flush(stdout)
        end
    end
end

# ── PHASE aa: arXiv:1501.05661 FIGURE 3, AT THE EXPERIMENT'S OWN SIZE AND BEYOND ─────────────
#
# THE TARGET IS A PUBLISHED, NON-INTERACTING MEASUREMENT, which makes it the strongest sanity
# check available anywhere in this campaign:
#
#   * Fig. 3 is EXPLICITLY `U = 0` -- "to avoid any interaction effects, only a single spin
#     component was used" -- so the model is spinless fermions in the quasiperiodic potential,
#     i.e. a pure XX chain with a `cos(2πβi+φ)` field. `aubry_andre_xxz_mpo(...; U = 0)`.
#   * being quadratic, its EXACT solution is an `L x L` matrix exponential (`aa_free_fermion`),
#     so there is no `2^L` wall: this phase runs at `L = 40` -- the size the paper's OWN DMRG
#     uses -- and at `L = 100`, with an exact reference at both.
#   * the Aubry-André transition sits at `Delta/J = 2` BY SELF-DUALITY. The stationary imbalance
#     must be ~0 below it and finite above it, with nothing fitted. A curve that turns over at 1
#     or at 4 is the Jordan-Wigner factor-of-2 error and looks perfectly reasonable otherwise.
#
# ⚠ WE COMPARE AGAINST THE PAPER'S *GREY* LINE. Fig. 3 shows ED "with (red) and without (grey)
# trap effects"; the experiment has a harmonic trap and our chain does not, so the homogeneous
# grey curve is the like-for-like one. Claiming agreement with the red curve would require the
# trap in the Hamiltonian.
#
# ⚠ AND THIS PHASE IS A CONTROL FOR THE METHOD COMPARISON, NOT EVIDENCE FOR IT. At `U = 0` the
# state is a Slater determinant whose entanglement SATURATES, so the rank demand is bounded and
# every integrator should do well. Its job is to prove the implementation at experiment scale.
# The place a rank-adaptive integrator can win is the INTERACTING problem.
#
# `WS` is read as the `Delta/J` ladder here -- for a quasiperiodic potential `Delta` IS the
# disorder axis, so the CSV's `W` column stays the disorder strength and needs no new column.
#
# THE PHASE `φ` IS THE REALISATION LABEL, and it is swept EVENLY rather than randomly: a
# quasiperiodic potential is deterministic, so averaging over `φ ∈ [0, 2π)` on a uniform grid is
# the exact phase average the paper's "12 phase realizations" approximates. `SEED` selects which
# of the 12 this task runs, so the array parallelises over `φ` exactly as over disorder.
const N_PHI = 12
aa_phase_of_seed(seed::Int) = 2π * mod(seed - 1, N_PHI) / N_PHI

function phase_aa(io)
    nst = round(Int, TMAX / DT)
    phi = aa_phase_of_seed(SEED)
    # The paper's stationary imbalance: the mean over 15τ..30τ, on the step grid so no time is
    # interpolated. See `stat_step_set` -- and note `TMAX` must reach 30 for this to be the
    # published quantity rather than a truncated version of it.
    steps, stat_steps = stat_step_set(nst)
    TMAX + 1e-9 < STAT_T1 && @printf("  aa ⚠ TMAX=%g < %g: I_stat is a TRUNCATED window (%g..%g), NOT the paper's\n",
                                     TMAX, STAT_T1, STAT_T0, min(TMAX, STAT_T1))
    set_symmetry!(:U1)
    @printf("  aa L=%d phi=%.4f (seed %d of %d) T=%g dt=%g, %d samples\n",
            L, phi, SEED, N_PHI, TMAX, DT, length(steps)); flush(stdout)

    for dlt in WS                     # `WS` IS the Delta/J ladder for this phase
        # ---- the EXACT reference: one L x L matrix exponential per sample, no ED ----
        tref = time()
        ref = [aa_free_fermion(L, n * DT; J = 1.0, Delta = dlt, phi = phi) for n in steps]
        @printf("  aa D/J=%-5g reference (free fermions, L=%d) in %.1fs; I_exact(T)=%.6f\n",
                dlt, L, time() - tref, ref[end].imbalance); flush(stdout)

        mpo = aubry_andre_xxz_mpo(L; J = 1.0, Delta = dlt, U = 0.0, phi = phi)
        psi0 = neel_state(L)
        # ⚠ `rsvd` INNERMOST, so every probe setting for one (chi, cutoff, kmax, scheme) runs
        # back-to-back on the same machine state -- which is what makes their wall clocks
        # comparable to each other. `rsvd_arms` collapses to one entry for `tdvp2`, which has no
        # probe and must not be replicated across the ladder.
        for maxdim in CHIS, cutoff in CUTOFFS, kmax in KMAXES, scheme in SCHEMES,
            rsvd in rsvd_arms(scheme)
            psi = copy(psi0)
            step = stepper(scheme, mpo, maxdim; cutoff = cutoff, kmax = kmax, rsvd = rsvd)
            # ⛔ COMPILE BEFORE THE CLOCK STARTS -- see `warm!`. Without it the first arm of each
            # cell pays the specialisation and every later arm reads as faster than it is.
            warm!(step, psi0, DT)
            t0 = time(); ab0 = gc_gb(); kd = 0; si = 1; n = 0
            stat_mps = Float64[]; stat_ex = Float64[]
            while n < nst
                n += 1
                try
                    kd += info_krylov(step(psi, ComplexF64(-im * DT)))
                catch e
                    @printf("  aa %s D/J=%g chi=%d cut=%.0e k=%d step %d THREW %s\n", scheme,
                            dlt, maxdim, cutoff, kmax, n,
                            sprint(showerror, e)[1:min(end, 120)]); flush(stdout)
                    break
                end
                (si <= length(steps) && steps[si] == n) || continue
                r = ref[si]; si += 1
                got = [sz_expectation(psi, j) for j in 1:L]
                nrm = norm(psi)
                obs = imbalance_from_profile(got)
                st = mps_spectrum_stats(psi); ent = st.ent
                if n in stat_steps
                    push!(stat_mps, obs); push!(stat_ex, r.imbalance)
                end
                row(io; suite = "aa", scheme = scheme, model = "aa", sym = "U1",
                    l = L, w = dlt, delta = 0.0, seed = SEED, maxdim = maxdim,
                    cutoff = cutoff, kmax = kmax, rsvd = rsvd_tag(scheme, rsvd), dt = DT,
                    t = n * DT, obs = obs, obs_exact = r.imbalance,
                    obs_err = abs(obs - r.imbalance),
                    entropy = ent, entropy_exact = r.entropy,
                    entropy_err = abs(ent - r.entropy),
                    sz_err = maximum(abs.((got .+ 0.5) .- r.n)),
                    energy = real(mpo_energy(psi, mpo)) / max(nrm^2, eps()),
                    energy_drift = 0.0, nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0, alloc_gb = gc_gb() - ab0,
                    smin = st.smin, sgapmin = st.sgapmin)
            end
            # The paper's own summary statistic, printed so the Fig. 3 curve can be read straight
            # off the log as well as rebuilt from the CSV.
            Ibar   = isempty(stat_mps) ? NaN : sum(stat_mps) / length(stat_mps)
            Ibarex = isempty(stat_ex)  ? NaN : sum(stat_ex)  / length(stat_ex)
            @printf("  aa %-6s D/J=%-5g chi=%-4d cut=%-7.0e k=%-2d %-6s bd=%-4d I_stat=%.6f exact=%.6f err=%.3e %.1fs\n",
                    scheme, dlt, maxdim, cutoff, kmax, rsvd_tag(scheme, rsvd),
                    maximum(bond_dims(psi); init = 0),
                    Ibar, Ibarex, abs(Ibar - Ibarex), time() - t0); flush(stdout)
        end
    end
end

# ── entry ────────────────────────────────────────────────────────────────────────────────────
"""
    check_row_shape()

Write ONE row to a buffer and require its field count to equal `COLS`.

⛔ THIS RUNS BEFORE ANY PHYSICS, AND THAT IS THE ENTIRE POINT. `@printf` checks its specifier
count only when the line executes, so a `COLS`/format/argument mismatch survives parsing, macro
expansion and a full package load, and then aborts the job on the FIRST ROW WRITTEN. Measured:
"25 != 27", two minutes into a cluster job, on a warm cache, after the gate had already emitted a
perfectly good first line -- i.e. the cheapest possible bug found in the most expensive possible
place. Writing a throwaway row costs microseconds and moves that failure to before the queue.
"""
function check_row_shape()
    buf = IOBuffer()
    # ⛔ THIS CALL IS ITSELF A CALL SITE AND MUST BE UPDATED WITH THE OTHERS. Adding `cutoff`/
    # `kmax` to `row` and forgetting it here cost a cluster job an 859-second precompile before
    # `UndefKeywordError: keyword argument `cutoff` not assigned` fired (job 16335097). A grep for
    # `row(io; suite = ` does NOT find this one -- it writes to `buf`, not `io`.
    row(buf; suite = "x", scheme = "x", model = "x", sym = "x", l = 1, w = 0.0, delta = 0.0,
        seed = 0, maxdim = 1, cutoff = 0.0, kmax = 0, rsvd = "x", dt = 0.0, t = 0.0,
        obs = 0.0, obs_exact = 0.0, obs_err = 0.0,
        entropy = 0.0, entropy_exact = 0.0, entropy_err = 0.0, sz_err = 0.0, energy = 0.0,
        energy_drift = 0.0, nrm = 1.0, maxbond = 1, centrebond = 1, krylov = 0, elapsed = 0.0,
        alloc_gb = 0.0, smin = 0.0, sgapmin = 0.0)
    got = length(split(strip(String(take!(buf))), ','))
    want = length(split(COLS, ','))
    got == want || error("row() writes $got fields but COLS names $want -- the format string, " *
                         "the argument list and COLS have drifted apart")
    return want
end

function main()
    ncol = check_row_shape()
    @printf("row shape OK: %d columns\n", ncol)
    mkpath(OUTDIR)
    # ⛔ THE Δ LADDER GOES IN THE FILENAME. One run covers every Δ in `DELTAS` as ROWS, so a
    # single job needs no tag -- but a Δ=0 job and a Δ=1 job launched separately at the same
    # `L/T/dt/seed` would otherwise write to one path and the second would silently win, leaving a
    # file whose `delta` column has one value and whose name claims nothing.
    # The cutoff and Krylov ladders join it for the same reason: two jobs differing ONLY in
    # `ARGS[10]` would otherwise share a path and the second would silently overwrite the first.
    tag = "$(PHASE)_L$(L)_T$(TMAX)_dt$(DT)_s$(SEED)" *
          (length(ARGS) >= 8 ? "_" * replace(ARGS[8], "," => "-") : "") *
          (length(ARGS) >= 9 ? "_D" * replace(ARGS[9], "," => "-") : "") *
          (length(ARGS) >= 10 ? "_c" * replace(ARGS[10], "," => "-") : "") *
          (length(ARGS) >= 11 ? "_k" * replace(ARGS[11], "," => "-") : "") *
          (length(ARGS) >= 13 ? "_r" * replace(ARGS[13], "," => "-") : "")
    path = joinpath(OUTDIR, "mbl_scars_$tag.csv")
    @printf("phase=%s L=%d T=%g dt=%g seed=%d threads=%d -> %s\n",
            PHASE, L, TMAX, DT, SEED, Threads.nthreads(), path); flush(stdout)
    open(path, "w") do io
        println(io, COLS); flush(io)
        # The gate runs before either production phase, always: see `phase_smoke`.
        (PHASE == "all" || PHASE == "smoke") && phase_smoke(io)
        (PHASE == "all" || PHASE == "mbl") && phase_run(io, :mbl)
        (PHASE == "all" || PHASE == "pxp") && phase_run(io, :pxp)
        # `aa` is NOT in `all`: it is a different size regime (L = 40-100 against an analytic
        # reference) and belongs to its own job, not bundled behind the ED phases.
        PHASE == "aa" && phase_aa(io)
        # `hub` is the spinful model's own gate -- its own Hamiltonian, its own reference, and the
        # only route to the paper's interacting figures. Kept out of `all` for the same reason as
        # `aa`: a different model belongs in its own job.
        PHASE == "hub" && phase_hub(io)
        # `hub4` is the SAME model on FUSED d=4 sites, and it is the one that can be evolved:
        # `hub`'s interleaved 2L-site layout makes every hop range-2, which freezes any product
        # state under every two-site sweep. This is the gate that must pass before any
        # interacting spinful production run.
        PHASE == "hub4" && phase_hub4(io)
        # The interacting spinful production run -- Figs. 2/4/5. NOT in `all`, and never run
        # before `hub4` has passed on the same tree: there is no reference at L=40, so an MPO
        # defect here is undetectable and produces plausible wrong physics.
        PHASE == "hub4prod" && phase_hub4prod(io)
        # The paper's own working point (L=40, m_max=100, 50 tau). NOT in `all`, and never run
        # before `hub` has passed on the same tree: it uses the identical MPO with no reference
        # to catch a defect, so an unvalidated MPO here produces plausible wrong physics.
        PHASE == "hubprod" && phase_hubprod(io)
        # Diagnostic for the frozen-state defect; prints only, writes no rows.
        PHASE == "hubdiag" && phase_hubdiag(io)
    end
    println("wrote $path")
    println("MAIN_DONE")
end

main()
