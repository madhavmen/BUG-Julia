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
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))  # xx_free_fermion_sz (EXACT, any L)

const P = parse_params((
    phase        = "tau",
    L            = 20,
    t_max        = 2.5,      # ≈50 steps at dt=0.05. Néel entanglement grows LINEARLY in t and
                             # the `split` phase runs with no closing truncation, so its ranks
                             # are the largest in the study -- extend only once χ(t) is known.
    dt           = 0.05,
    J            = 1.0,
    # ⛔ `delta` SELECTS THE MODEL **AND** THE REFERENCE, and the two are not independent.
    #
    #   delta = 1   XXX / isotropic HEISENBERG. Bethe gives the SPECTRUM, so there is no closed
    #               form for `exp(-iHt)|psi>`; the reference is sparse Krylov ED, which caps this
    #               driver at L ~ 20 (the Sz=0 sector is C(20,10) = 184,756).
    #   delta = 0   the **XX** point. Free fermions under Jordan-Wigner, so
    #               `xx_free_fermion_sz` is EXACT AND ANALYTIC AT ANY L -- L = 50 included, where
    #               no ED of any kind exists.
    #
    # ⚠ delta = 0 IS THE XX MODEL, NOT THE HEISENBERG MODEL. They are different Hamiltonians with
    # different exact anchors: Bethe's E/L = -0.4331 and the des Cloizeaux-Pearson spinon velocity
    # pi/2 belong to delta = 1 ONLY. Labelling a delta = 0 figure "Heisenberg" invites a reader to
    # score it against the wrong number, which is the same class of error as the chain that was
    # once evolved under a triangular coupling builder while carrying a chain's label.
    delta        = 1.0,
    # `neel` = |up dn up dn ...>, `domainwall` = |up..up dn..dn>. Both are Sz = 0 z-basis product
    # states, so both are representable under :U1 and both have an EXACT t=0 energy (see
    # `product_energy`). ⛔ Neither is representable under :SU2 -- `product_state` throws.
    init         = "neel",
    maxdim       = 0,        # 0 = UNCAPPED. A cap would floor the τ_trunc scan at the cap's own
                             # error and hide the knee this study exists to find.
    # ⛔ LADDER TRUNCATED AT 1e-8 (2026-09-01). MEASURED at L=18, chi=64: rows tau <= 1e-7 are
    # identical to THREE DIGITS across the whole plateau (1.32-1.33e-04), because chi is pinned at
    # the cap and neither threshold can move the error below what the rank allows. The 1e-10 /
    # 1e-12 / 1e-14 rows cost ~40% of the grid and restate one number.
    taus         = "1e-4,1e-5,1e-6,1e-7,1e-8",
    dts          = "0.2,0.1,0.05,0.025,0.0125",
    dt_taus      = "1e-4,1e-6,1e-8,1e-10",
    splits       = "1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10",
    # Phase `grid`'s half-sweep axis. Deliberately the SAME ladder as `taus`, so the matrix is
    # square and the diagonal (both thresholds equal) is a readable line through it -- that is
    # the "one tolerance governs the whole step" configuration the plan is aiming at.
    grid_splits  = "1e-4,1e-5,1e-6,1e-7,1e-8",
    # Which half-sweep structures to run the matrix for. See `SWEEPS`.
    grid_sweeps  = "m0,m3,mdef",
    # Which half-sweep truncation the matrix's x-axis is: "split" (the SVD that splits the
    # stacked Krylov frame) or "cbe" (the SVDs inside cbe_expand). Both always present.
    grid_axis    = "split",
    schemes      = "bug_interleaved,tdvp_cbe1s,tdvp2",
    sample_every = 0.25,
    symmetry     = "U1",
    maxiter      = 0,        # 0 = derive the Krylov depth from ‖H‖·dt/2
    # Phase `krylov`'s axis. ⛔ ONE PROCESS, NOT ONE PER DEPTH: `using BUGJulia` costs ~170 s
    # warm, so six depths as six invocations is ~18 min of pure JIT locally and six queue slots
    # on the cluster, for six runs that share every other input.
    maxiters     = "2,3,4,6,8,12",
    # ⛔ FREE SUFFIX ON EVERY OUTPUT FILE, AND IT IS WHAT MAKES CLUSTER SPLITTING POSSIBLE.
    # `knobtag()` carries L, dt, T, maxdim and maxiter but NOT tau/split, so two tasks that run
    # DIFFERENT ROWS of the same grid (`taus=1e-6` vs `taus=1e-7`) write the SAME filename and one
    # silently overwrites the other. A chi=256 grid does not fit in one queue slot, so it has to be
    # split by row -- pass `tag=r1e-6` and the rows land in separate files to be concatenated.
    tag          = "",
    exact_cbe    = true,     # full SVD, not the randomised sketch (Jan's point 3)
    # ⛔ THE 1-SITE CBE BASELINE WAS RUNNING ON PACKAGE DEFAULTS WHILE `cbe_bug` GOT TUNED ONES,
    # WHICH IS NOT A COMPARISON. `tdvp_cbe1s_step!` was called with `maxdim`/`trunc_thresh`/
    # `maxiter` only, so it silently kept `comp_ratio = 0.5` and `exact = false` (the randomised
    # sketch) while the BUG arm ran `exact = exact_cbe` (a full SVD) with `dover = 4` and
    # `comp_ratio = 1.0`. The two arms were therefore using DIFFERENT CBE machinery, and neither
    # the accuracy nor the cost gap between them meant anything.
    #
    # ⚠ `dex = 0` is NOT "no expansion" -- `cbe_core.jl:635` reads `dex <= 0` as "use the
    # `growth` schedule", i.e. `budget = ceil(growth*dmax) - r`. So the old baseline was expanding;
    # it was expanding under a different rule.
    #
    # Defaults here MATCH the BUG arm's, so `tdvp_cbe1s` is a fair baseline out of the box, and
    # each knob stays scannable for the tuning pass.
    cbe1s_dover      = 4,
    cbe1s_comp_ratio = 1.0,
    cbe1s_growth     = 2.0,
    # `-1` = inherit `exact_cbe`, so the two arms cannot silently diverge on the sketch/SVD choice.
    cbe1s_exact      = -1,
))

const OUTDIR = joinpath(@__DIR__, "results")

# ‖H_XXZ‖ ≤ Σ_bonds ‖h_bond‖. The XXZ bond operator has eigenvalues {Δ/4, Δ/4, -Δ/4 ± 1/2}, so
# ‖h_bond‖ = Δ/4 + 1/2 -- which is 3/4 at Δ=1 (the value this was hard-coded to) and 1/2 at Δ=0.
# The extremal many-body eigenvalues are ~0.44·J·L, so this over-estimates by ~2x -- which costs
# Krylov depth and never accuracy, the right direction to be wrong in.
# ⛔ HARD-CODING 3/4 HERE WOULD OVER-DEEPEN EVERY Δ=0 KRYLOV SOLVE BY 50%, quietly making the XX
# arm look more expensive than it is in exactly the cost comparison this driver exists to make.
const HNORM = (P.delta / 4 + 0.5) * P.J * (P.L - 1)

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
function exact_profiles(L::Int, J::Float64, t_max::Float64, sample_every::Float64,
                        delta::Float64, init::String, occ::Vector{Int})
    nsamp = round(Int, t_max / sample_every)
    ts = [k * sample_every for k in 0:nsamp]

    if delta == 0.0
        # ✅ ANALYTIC AND EXACT AT ANY L. Jordan-Wigner makes the XX point of the XXZ family
        # quadratic, so the single-particle propagator `exp(-i h t)` is an L x L matrix
        # exponential and
        #     <Sz_j(t)> = Σ_{k occupied} |[exp(-i h t)]_{jk}|² - 1/2
        # is exact -- no many-body Hilbert space, no truncation, no solver tolerance. This is
        # what makes L = 50 scoreable at all: the Sz=0 sector there is C(50,25) = 1.26e14.
        #
        # ⚠ Each sample is an INDEPENDENT exponential from t = 0, not a product of steps, so the
        # reference cannot accumulate error along the sample grid the way a stepped one would.
        return ts, [xx_free_fermion_sz(L, t; J = J, occupied = occ) for t in ts]
    end

    # Δ ≠ 0: interacting, so there is no explicit closed form for `exp(-iHt)|psi>` even though
    # the model is Bethe-integrable -- the Bethe ansatz gives the SPECTRUM, and the overlaps of a
    # product state with every Bethe state are themselves a 2^L problem. Sparse Krylov ED in the
    # Sz = 0 sector is exact to solver tolerance, and is what caps this branch's size.
    L <= 20 || error("delta=$delta needs the ED reference, infeasible at L=$L " *
                     "(the Sz=0 sector is C($L,$(L÷2))). Use delta=0 for an analytic reference.")
    H, states, idx = heisenberg_sparse(L; J = J, delta = delta)
    v = init == "neel" ? neel_vector(L, idx) : domain_wall_vector(L, states, idx)
    profs = Vector{Vector{Float64}}(undef, nsamp + 1)
    profs[1] = magnetisation_dense(v, L, states)
    for k in 1:nsamp
        v = expv_sparse(H, ComplexF64(-im * sample_every), v; m = 30, tol = 1e-12)
        profs[k + 1] = magnetisation_dense(v, L, states)
    end
    return ts, profs
end

"""
The scalar signal: staggered magnetisation `(1/L) Σ_j (-1)^(j+1) ⟨S^z_j⟩`.

STAGGERED rather than uniform because `S^z_tot` is CONSERVED -- the uniform sum is identically 0
for every scheme at every time and would grade nothing at all. A sum over all sites rather than
one site, so it is not accidentally sensitive to where the front happens to be at a sample time.
Starts at 1/2 for the Néel state and decays as the order melts.
"""
function staggered(p::Vector{Float64})
    L = length(p)
    if P.init == "neel"
        return sum((-1)^(j + 1) * p[j] for j in 1:L) / L
    end
    # ⛔ THE STAGGERED SUM IS THE WRONG SCALAR FOR A DOMAIN WALL -- it is ~0 at t=0 and would
    # grade nothing. The matched signal is the magnetisation still held left of the wall minus
    # that held right of it: it starts at exactly 1/2, like the Néel staggered sum, and decays as
    # the wall melts, so the two start states produce directly comparable curves.
    return (sum(p[1:(L ÷ 2)]) - sum(p[(L ÷ 2 + 1):L])) / L
end

# ── the run ───────────────────────────────────────────────────────────────────────────────

set_symmetry!(Symbol(P.symmetry))
const W    = xxz_mpo(P.L; J = P.J, delta = P.delta)
const PSI0 = P.init == "neel"       ? neel_state(P.L) :
             P.init == "domainwall" ? domain_wall_state(P.L) :
             error("unknown init $(repr(P.init)) -- use neel or domainwall")

# The z-basis spin pattern of the start state, as ±1. Drives both the exact t=0 energy and the
# occupied-site list the analytic reference needs, so the two cannot disagree about the state.
const SPINS = P.init == "neel" ? [isodd(j) ? 1 : -1 for j in 1:P.L] :
                                 [j <= P.L ÷ 2 ? 1 : -1 for j in 1:P.L]
# Occupied fermion sites: up spin is an occupied fermion (Sz = n - 1/2).
const OCC = [j for j in 1:P.L if SPINS[j] > 0]

"""
Exact energy of a z-basis product state: only `S^z S^z` survives, since `S^x S^x + S^y S^y` flips
two spins and takes the state out of itself. So `E = J·Δ·Σ_bonds s_i s_{i+1} / 4`, with no
tolerance in it -- an exact gate at any `Δ` and either start state.

At `Δ = 0` this is `0` for EVERY product state, which is a weaker gate than the `Δ = 1` case: it
would pass even if the sign of `J` or the spin pattern were wrong. `preflight` therefore also
gates the t=0 PROFILE against the reference, which does see the pattern.
"""
product_energy() = P.J * P.delta * sum(SPINS[j] * SPINS[j + 1] for j in 1:(P.L - 1)) / 4

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
                 close::Bool, dt::Float64; cbe_cut::Float64 = 0.0, kry_depth::Int = 0)
    # `kry_depth > 0` overrides both `P.maxiter` and the ‖H‖·dt/2 bound, for phase `krylov`.
    m = kry_depth > 0 ? kry_depth : depth_for(dt)
    # ⚠ `cbe_bug` is accepted as a LEGACY ALIAS so saved commands and old CSV tags still
    # resolve; the canonical name is `bug_interleaved`.
    scheme = replace(scheme, r"^cbe_bug" => "bug_interleaved")
    if startswith(scheme, "bug_interleaved")
        # A bare "bug_interleaved" is the PACKAGE DEFAULTS, not a pinned depth -- so the scheme
        # named after the package reports what the package actually does.
        name = scheme == "bug_interleaved" ? "mdef" :
               scheme[length("bug_interleaved_") + 1:end]
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
        ex = P.cbe1s_exact < 0 ? P.exact_cbe : P.cbe1s_exact > 0
        return (p, tau) -> tdvp_cbe1s_step!(p, W, tau; maxdim = CAP,
                                            trunc_thresh = tau_trunc, maxiter = m,
                                            exact = ex, dover = P.cbe1s_dover,
                                            comp_ratio = P.cbe1s_comp_ratio,
                                            growth = P.cbe1s_growth)
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
                 cbe_cut::Float64 = 0.0, kry_depth::Int = 0, pio = nothing)
    step!  = stepper(scheme, tau_trunc, split_cutoff, close, dt;
                     cbe_cut = cbe_cut, kry_depth = kry_depth)
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
        # The site-resolved observable, for the trajectory plot. Written from the SAME `prof` the
        # error above is computed from, so the figure and the scalar can never disagree.
        if pio !== nothing
            for j in 1:P.L
                @printf(pio, "%s,%s,%g,%g,%g,%d,%.10g,%d,%.10e,%.10e\n",
                        phase, scheme, tau_trunc, split_cutoff, cbe_cut, P.maxdim,
                        t, j, prof[j], profs[n][j])
            end
            flush(pio)
        end
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
    @printf("  %-11s m=%-3d dt=%-7g tau=%-8g split=%-8g close=%-5s | err %.3e  prof %.3e  chi %-4d  kry %-7d  %.0fs\n",
            scheme, kry_depth > 0 ? kry_depth : depth_for(dt), dt, tau_trunc, split_cutoff, close,
            abs(staggered(prof) - staggered(profs[end])),
            maximum(abs.(prof .- profs[end])), maximum(bond_dims(psi)), kry, secs)
    flush(stdout)
end

# ⛔ EVERY OUTPUT NAME CARRIES `maxdim` AND `maxiter`. They are state-determining inputs, so two
# runs that differ only in the rank cap or the Krylov depth are DIFFERENT experiments and must not
# share a filename -- see the note in phase `grid`.
capstr()  = P.maxdim  > 0 ? string(P.maxdim)  : "inf"
iterstr() = P.maxiter > 0 ? string(P.maxiter) : "auto"
knobtag() = string("_D", capstr(), "_m", iterstr(), isempty(P.tag) ? "" : "_" * P.tag)

"""
Per-site `<S^z_j(t)>` for one arm, MPS and exact side by side.

⛔ THE MAIN CSV KEEPS ONLY `err_prof`, WHICH IS A NORM AND CANNOT BE UN-COLLAPSED. An L-infinity
number says how far the profile is from exact but not WHERE, so a front that arrives early, a
boundary reflection and a uniform offset all reduce to the same scalar. The observable plot needs
the profile itself, and re-running a 40-minute arm to recover something the arm already computed
is the wasteful kind of mistake.

One file per phase invocation, with the arm's identity in every row, so a partially-written file
still parses and every row is self-describing.
"""
open_prof(name) = begin
    mkpath(OUTDIR)
    io = open(joinpath(OUTDIR, name), "w")
    println(io, "phase,scheme,tau_trunc,split_cutoff,cbe_cutoff,maxdim,t,site,sz_mps,sz_exact")
    flush(io)   # header out immediately: a 0-byte file must mean "never opened", not "buffered"
    io
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
    e0, ref = real(mpo_energy(copy(PSI0), W)) / norm(copy(PSI0))^2, product_energy()
    @printf("t=0 energy:    %.12f  exact = %.12f  (diff %.2e)\n", e0, ref, abs(e0 - ref))
    abs(e0 - ref) < 1e-10 || error("t=0 energy mismatch -- the MPO is not this Hamiltonian")
    # ⛔ THE t=0 PROFILE GATE IS NOT REDUNDANT WITH THE ENERGY GATE, AND AT Δ=0 IT IS THE ONLY
    # ONE THAT BINDS. `product_energy()` is identically 0 for EVERY product state at Δ=0, so the
    # energy check there would pass with the spin pattern reversed, the wall in the wrong place,
    # or `occupied` mismatched against `SPINS`. The profile sees all three.
    p0, r0 = sz_profile(copy(PSI0)), profs[1]
    @printf("t=0 profile:   max|mps - exact| = %.2e over %d sites\n\n",
            maximum(abs.(p0 .- r0)), P.L)
    maximum(abs.(p0 .- r0)) < 1e-12 ||
        error("t=0 profile mismatch -- the MPS start state and the reference's `occupied` " *
              "list describe different states")
end

function main()
    @printf("XXZ Δ=%g (%s), %s quench -- L=%d  %s  J=%g  T=%g  dt=%g  maxdim=%s\n",
            P.delta, P.delta == 0 ? "XX / free fermions" : P.delta == 1 ? "isotropic Heisenberg" : "anisotropic",
            P.init, P.L, P.symmetry, P.J, P.t_max, P.dt,
            P.maxdim > 0 ? string(P.maxdim) : "UNCAPPED")
    @printf("reference: %s\n", P.delta == 0 ? "EXACT ANALYTIC free fermions (valid at any L)" :
                               "sparse Krylov ED, Sz=0 sector (caps L at ~20)")
    @printf("phase=%s  exact_cbe=%s  krylov depth=%d (bound %.1e)\n",
            P.phase, P.exact_cbe, depth_for(P.dt), krylov_bound(HNORM, P.dt, depth_for(P.dt)))
    print("building the exact reference (C($(P.L),$(P.L÷2)) states) ... "); flush(stdout)
    t0 = time_ns()
    _, profs = exact_profiles(P.L, P.J, P.t_max, P.sample_every, P.delta, P.init, OCC)
    @printf("%d samples, %.1f s\n", length(profs), (time_ns() - t0) / 1e9)
    preflight(profs)

    schemes = parse_syms(P.schemes)
    if P.phase == "tau"
        # 0a / 0b. Is there a knee, and where? Closing truncation ON at τ_trunc, half-sweep split
        # left at its shipped 1e-14 -- i.e. today's configuration, with the one knob swept.
        io  = open_out(@sprintf("heis_tau_L%d_dt%g_T%g%s.csv", P.L, P.dt, P.t_max, knobtag()))
        pio = open_prof(@sprintf("heis_tau_prof_L%d_dt%g_T%g%s.csv", P.L, P.dt, P.t_max, knobtag()))
        try
            for tau in parse_list(P.taus, Float64), s in schemes
                run_arm(io, "tau", s, tau, 1e-14, true, P.dt, profs; pio = pio)
            end
        finally; close(io); close(pio); end

    elseif P.phase == "dt"
        # 0c / 0d. The calibration curve τ_trunc*(dt): where each τ_trunc's line leaves the
        # common envelope is where truncation takes over from time integration.
        io = open_out(@sprintf("heis_dt_L%d_T%g%s.csv", P.L, P.t_max, knobtag()))
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
        io = open_out(@sprintf("heis_split_L%d_dt%g_T%g%s.csv", P.L, P.dt, P.t_max, knobtag()))
        try
            for sc in parse_list(P.splits, Float64)
                run_arm(io, "split", "bug_interleaved", 0.0, sc, false, P.dt, profs)
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
            sch = "bug_interleaved_$sw"
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
            # ⛔ `maxdim` AND `maxiter` ARE PART OF THE FILENAME. Without them a chi=128 grid
            # silently OVERWRITES the chi=64 one -- same L, same dt, same T, same sweep -- and a
            # convergence ladder over (chi x Krylov depth) collapses onto whichever run finished
            # last. `maxdim` is carried as a COLUMN, so the surviving file still looks internally
            # consistent; nothing flags that eight ninths of the campaign is gone. This is the
            # same failure the 2209 driver guards against in `tag`, and the same one the
            # one-CSV-per-sweep note below was added for.
            io  = open_out(@sprintf("heis_grid%s_%s_L%d_dt%g_T%g%s.csv",
                                    axis == "cbe" ? "cbe" : "", sw, P.L, P.dt, P.t_max, knobtag()))
            pio = open_prof(@sprintf("heis_grid%s_%s_prof_L%d_dt%g_T%g%s.csv",
                                     axis == "cbe" ? "cbe" : "", sw, P.L, P.dt, P.t_max, knobtag()))
            try
                for tau in parse_list(P.taus, Float64), x in parse_list(P.grid_splits, Float64)
                    @printf("  -> arm tau=%g %s=%g ...\n", tau, axis, x); flush(stdout)
                    if axis == "split"
                        run_arm(io, "grid", sch, tau, x, true, P.dt, profs; pio = pio)
                    else
                        run_arm(io, "grid", sch, tau, 1e-14, true, P.dt, profs;
                                cbe_cut = x, pio = pio)
                    end
                end
            finally; close(io); close(pio); end
        end
    elseif P.phase == "krylov"
        # WHY THIS PHASE EXISTS. `depth_for` provisions the Lanczos depth from a WORST-CASE bound
        # ‖H‖·dt/2, and at L=8/dt=0.05 that gave depth 12 for a bound of 4.2e-22 -- fourteen
        # orders tighter than any error in the run. ⛔ AND `lanczos_expv` EXITS ONLY ON
        # BREAKDOWN, so every local solve pays ALL of `maxiter` matvecs whether it needed them or
        # not. That makes an over-provisioned depth a pure multiplier on the cost of EVERY arm.
        #
        # ⚠ IT MULTIPLIES ALL THREE ARMS ALIKE, so it is not a knob that flatters one of them --
        # which is exactly why it has to be tuned BEFORE any cross-arm cost claim is quoted, or
        # every ratio is reported at a depth nobody would choose.
        #
        # Read it as: the depth at which the error curve LEAVES the flat floor is the cheapest
        # honest setting. A curve that is flat all the way to the smallest depth means the bound
        # is irrelevant for this model and the shallowest arm wins outright.
        io  = open_out(@sprintf("heis_krylov_L%d_dt%g_T%g_D%s.csv", P.L, P.dt, P.t_max, capstr()))
        pio = open_prof(@sprintf("heis_krylov_prof_L%d_dt%g_T%g_D%s.csv", P.L, P.dt, P.t_max, capstr()))
        try
            for mi in parse_list(P.maxiters, Int), s in schemes
                @printf("  -> maxiter=%d %s ...\n", mi, s); flush(stdout)
                run_arm(io, "krylov", s, 1e-10, 1e-14, true, P.dt, profs;
                        kry_depth = mi, pio = pio)
            end
        finally; close(io); close(pio); end

    else
        error("unknown phase $(repr(P.phase)) -- use tau, dt, split, grid or krylov")
    end
    println("\nwrote to ", OUTDIR)
end

main()
