# cbe_options.jl: prebuilt-stack wrappers, the shared run configuration, the S step and the sketched gate action (from `cbe_core.jl`, included right after it).

# ── prebuilt-stack convenience wrappers ──────────────────────────────────────
#
# The kernels above take the channels on the two links they actually use, so a sweep can
# CARRY them (O(1) per bond) instead of rebuilding a stack (O(L) per bond). These wrappers
# keep the stack-indexed call sites -- the tests, and anything not sweeping -- unchanged.

sketch_h_left(f::BondFrame, h::XXZChain, i::Int,
              lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_left(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

sketch_h_right(f::BondFrame, h::XXZChain, i::Int,
               lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_right(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

cbe_expand(f::BondFrame, h::XXZChain, i::Int,
           lenv::LeftEnvStack, renv::RightEnvStack; kwargs...) =
    cbe_expand(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2); kwargs...)

# ── shared run configuration ──────────────────────────────────────────────────
#
# Lives here, not in a mode submodule, because all three modes and both paths (gate,
# MPO-environment) are configured by the same expansion knobs. Only the fields that mean
# something different per mode are documented per mode.

"""
    CBEBugOptions

Controls for one run of ANY of the three modes, on either path. Mirrors `BondUpdateOptions`
where the meaning is the same, so the integrators can be driven from the same campaign code.

  - `dt`, `n_steps` -- step size and how many to take. `dt` is a REAL time step in
    real time (`tau = -im*dt`) and an INVERSE-TEMPERATURE step in imaginary time
    (`tau = -dt`); the mode
    forms `tau` from it. [`GroundState`](@ref) ignores both and uses `n_sweeps`/`etol`.
  - `maxdim`, `trunc_thresh` -- truncation, applied at the centre and in the final sweep.
  - `normalize` -- rescale after each step; the norm is recorded BEFORE rescaling.
  - `maxiter`, `tol` -- Krylov budget for the single 0-site exponential.
  - `seed` -- one RNG for the whole run, so a run is reproducible while consecutive steps
    still draw different sketches.

NO `order` FIELD on the MPO path. `bond_update_bug!` offers `:lie` / `:strang` because it
Trotterises the even and odd bond groups and has a splitting error to compose away. The
MPO-path CBE-BUG takes one projected exponential per step over a globally expanded subspace
-- there is no operator splitting to symmetrise, so a composition parameter would be
meaningless there. The gate path is always Strang.

WHY THE GATE PATH STORES NO ENVIRONMENTS. Its environments are the IDENTITY, and that is a
property of the splitting rather than an omission. `H = H_odd + H_even` and each group is a
sum of two-site terms on disjoint pairs, so `exp(tau*H_odd)` factorises exactly into one
`exp(tau*h_i)` per bond: the operator at a bond IS the bare gate, and the rest of the chain
is not neglected, it is a different factor of the same product applied at its own bond. The
sweep calls `canonical!(psi, i)` before each bond, which makes everything to the left a left
isometry and everything to the right a right isometry, so `L = R = I` exactly. The MPO path
cannot do this because the full `H` does not factorise, which is what `henv.jl` is for.

The consequence, and it is what fixes the mode/path matrix: the gate path's local operator
has no knowledge of the chain, so a local EIGENSOLVE there would return the ground state of
one two-site term with frozen neighbours -- a different quantity, not an approximation to
the global ground state. Ground state therefore runs on the MPO path only. Imaginary time
runs on both, because `exp(-dt*H)` Trotterises exactly as `exp(-im*dt*H)` does.
"""
Base.@kwdef struct CBEBugOptions
    dt::Float64 = 0.05
    n_steps::Int = 10
    centre_expand::Bool = true
    maxdim::Int = 200
    trunc_thresh::Float64 = 1e-12
    normalize::Bool = true
    maxiter::Int = 30
    tol::Float64 = 1e-15
    # S-step scheme, honoured by the `cbe_bond_update` path only. `s_iters` selects the solver
    # (see `cbe_bond_update`: 1 = one expansion + plain Lanczos, >1 = the outer-loop control,
    # <=0 = the expanding-basis Lanczos with `-s_iters` growth passes). `s_reorth` turns on full
    # reorthogonalisation in the plain Lanczos -- measured ~5.7x fewer matvecs at one bond, not
    # yet a default because it has not been checked across a sweep.
    s_iters::Int = 1
    s_reorth::Bool = false
    # PLAIN CBE instead of RSVD-CBE: the complete fused basis replaces the thin Gaussian, so
    # the preselection sees the exact `H*Theta` and its SVD returns the true leading directions
    # rather than an estimate. This FORMS THE RANK-4 OBJECT (cost `d*chi`, not `1.2*budget`),
    # which is precisely what the sketch exists to avoid -- it is the reference arm of the A/B,
    # not a better default. Makes `seed` and the RNG inert.
    exact::Bool = false
    seed::UInt = 0x5EED
    record_magnetisation::Bool = false
    observe::Union{Nothing, Function} = nothing
end

"""
    CBEBugRunInfo

Per-step record of a `cbe_lubich_bug!` run; every field has length `n_steps`.

`max_expanded` is the headline rank diagnostic and the one to compare against
`BondUpdateInfo.aug_k_dims`: it is the largest bond the basis sweep proposed BEFORE the
truncation, i.e. `r + Dex`, against the K/L augmentation's `2r`.

`krylov_dims` and `mean_aug` are the COST pair, and they are only useful together:
`krylov_dims` counts operator applications in the step's exponential(s) (summed over bonds in
the gate path, the single centre solve in the MPO path) and `mean_aug` is the size those
applications acted on. Wall time is roughly their product, so a setting can be slower either
by doing more iterations or by doing them on bigger objects -- and the budget A/B in
`benchmarks/cbe_budget_cost.jl` turns on telling those two apart.
"""
struct CBEBugRunInfo
    times::Vector{Float64}
    norms::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    expanded::Vector{Vector{Int}}
    max_expanded::Vector{Int}
    centre_ranks::Vector{Int}
    err_pre::Vector{Float64}
    err_fnl::Vector{Float64}
    discarded::Vector{Float64}
    magnetisations::Vector{Vector{Float64}}
    observations::Vector{Any}
    krylov_dims::Vector{Int}
    mean_aug::Vector{Float64}
    # EMPTY unless the caller asked for it. Imaginary time fills it (energy is the convergence
    # signal there); real time leaves it empty rather than paying for a full contraction per
    # step that nothing reads.
    energies::Vector{Float64}
end

CBEBugRunInfo(times, norms, bds, maxbd, expanded, maxexp, centres,
              epre, efnl, disc, mags, obs, kdim, maug) =
    CBEBugRunInfo(times, norms, bds, maxbd, expanded, maxexp, centres,
                  epre, efnl, disc, mags, obs, kdim, maug, Float64[])

Base.length(info::CBEBugRunInfo) = length(info.times)
# ══════════════════════════════════════════════════════════════════════════════════════
# PART II -- CBE AS THE BASIS UPDATE OF THE GATE-BASED bond_update_bug.
#
# Everything above is the SELECTION: sketch, preselect, rank, direct-sum. Everything below
# drives it from a bond frame and a gate, and solves the local problem in the expanded
# basis. The two were separate files until the three modes (real time, imaginary time,
# ground state) were split out into submodules; this file is what all three share.
#
# THE THREE MODES DIFFER IN EXACTLY TWO PLACES, and nowhere else:
#
#   1. WHICH OPERATOR the local problem is posed with -- `_expanding_krylov`'s `applyH`.
#   2. HOW THE TRIDIAGONAL IS REDUCED to an answer -- the `SStepSolver` below.
#
# Everything in between (the growth passes, the sketch, the P_perp projection, the ranking,
# the direct sum, the embedding of stored Krylov vectors into the grown basis, the full
# reorthogonalisation) is byte-for-byte the same code in all three. That is the point: the
# ground-state solver is a DROP-IN REPLACEMENT FOR THE S STEP, sharing the identical Krylov
# basis, not a second algorithm that happens to look similar.

# ── the S step: what to do with the Krylov space once it is built ─────────────

"""
    SStepSolver

How [`_expanding_krylov`](@ref) turns its tridiagonal into the local answer.

The Krylov space, and every growth pass that built it, is identical across subtypes. Only
the reduction differs, which is why the modes share one solver.
"""
abstract type SStepSolver end

"""
    Propagate(tau)

`exp(tau * H_eff)` applied to the current local core. `tau = -im*dt` is real time,
`tau = -dt` is imaginary time; the solver does not care which, and neither does anything
below it.
"""
struct Propagate <: SStepSolver
    tau::ComplexF64
end

"""
    LowestEigen()

The lowest Ritz vector of `H_eff` in the accumulated Krylov space -- a DMRG local solve.

REPLACES the exponential, keeps the basis. Returns a NORMALISED vector: unlike the
propagator, which scales the core by `beta0` to carry the state's weight, an eigenvector
carries no scale and the surrounding sweep is a variational minimisation of a unit-norm
state.

THE OPERATOR THIS IS GIVEN MUST INCLUDE THE ENVIRONMENTS. Handed the bare two-site gate it
would minimise `<h_{i,i+1}>` at each bond independently, whose fixed point is not the global
ground state -- each bond drives its own pair toward a local singlet and the next bond in
the sweep undoes it. DMRG converges because its local operator is `L + h + R`, so a local
minimisation lowers the TOTAL energy. See `apply_h_two_site` in `henv.jl`, and the header of
`CBEBugOptions` for why the gate path cannot supply this.
"""
struct LowestEigen <: SStepSolver end

"""
    _reduce_krylov(solver, alpha, betas, beta0) -> (coeffs, value)

`coeffs[k]` multiplies the `k`-th stored orthonormal Krylov vector to build the answer.
`value` is the Ritz value for [`LowestEigen`](@ref) and `NaN` when the solver has no
eigenvalue to report.
"""
_reduce_krylov(p::Propagate, alpha, betas, beta0) =
    (ComplexF64[beta0 * c for c in hermitian_tridiagonal_exp_coeffs(alpha, betas, p.tau)],
     NaN)

function _reduce_krylov(::LowestEigen, alpha, betas, beta0)
    # `beta0` is deliberately UNUSED: the propagator carries the state's norm through the
    # step, an eigensolve replaces the state outright and the Ritz vector is normalised.
    isempty(betas) && return (ComplexF64[1.0], alpha[1])
    F = eigen(SymTridiagonal(alpha, betas))
    return (ComplexF64.(@view F.vectors[:, 1]), F.values[1])
end
# ══════════════════════════════════════════════════════════════════════════════════════

# CBE as the basis update of the GATE-BASED bond_update_bug -- the validated path.
#
# WHY THIS FILE EXISTS, AND WHY IT COMES FIRST.
#
# `cbe_lubich.jl` puts CBE inside the Lubich/Sulz basis sweep: K-sweep, L-sweep, one centre
# Galerkin, an environment-dressed effective H. That sweep is the eventual target, but it means a failure has four
# possible homes -- the sweep, the channel environments, the CBE selection, or the placement
# of the single Galerkin step. The rank stall could not be pinned to any of them.
#
# `BondUpdateBUG.parity_sweep!` + `kls_bond_update` is the OTHER integrator, and it is
# validated: odd/even Trotter, one commuting bond group at a time, exact within a group,
# full-rank-exact, unitarity-locked. Dropping CBE into THAT removes three of the four
# suspects at once:
#
#   * no environments at all -- the Hamiltonian at a bond IS the gate, purely local, so
#     `henv.jl` and the whole stale/fresh-channel question is out of the picture;
#   * no sweep to get wrong -- `canonical!(psi, i)` before each bond, exactly as the
#     validated sweep does, so every frame is a true canonical snapshot;
#   * no Galerkin-placement question -- there is one Galerkin step per bond and it is the
#     step the validated integrator already takes.
#
# What is left is precisely the piece under suspicion: does the CBE selection admit the
# directions the dynamics needs? If rank grows here, the selection is sound and the fault in
# `cbe_lubich.jl` is in the sweep. If it stalls here too, the selection is the fault, and it is
# now isolated with nothing else to blame.
#
# The selection code is SHARED, not reimplemented: `cbe_expand_bond` calls the same
# `cbe_expand(f, skl, skr; ...)` core as the MPO path, differing only in the two sketch
# closures. So this file is a genuine test of the selection the Lubich sweep depends on.
#
# WHAT IT REPLACES IN `kls_bond_update`. Both of that kernel's basis mechanisms, together:
#
#   * the two K/L ODE solves (`expv` on `P_perp H_K`, non-Hermitian, Arnoldi). CBE supplies
#     the same DIRECTIONS from one sketched application of the gate, with no exponential;
#   * `missing_fill`, the per-sector random seed. The sector-graded sketch reaches an
#     empty-but-reachable sector THROUGH the gate, so what comes back is the direction the
#     dynamics wants rather than a random column the S-step SVD then discards.
#
# The S-step is unchanged and deliberately so -- it is the Galerkin step of the validated
# integrator, `S_hat0 = U_hat' Theta0 V_hat'` with no overlap matrices, Hermitian, Lanczos.

# ── the sketched gate action ──────────────────────────────────────────────────
#
# NEVER FORMS `H Theta`. The rank-4 two-site object is what 2-site TDVP builds and what this
# method promises not to allocate, and `cbe.jl`'s header spells out why sketching after the
# fact would be a strawman: with `H Theta` in hand its exact left SVD beats any sketch of
# it. So `Om` is folded into the gate FIRST -- `gate` is `d^4`, `Om` is thin, and the
# largest intermediate is `O(d^2 * g * r)`.
#
# The guard is `sketch_bond_left(f, gate, Om) == P_perp(contract(apply_gate(gate, Theta), Om'))` for
# arbitrary `Om`, against `apply_gate`, which is itself pinned to the analytic Heisenberg
# block element-for-element (`tests/BondUpdateBUG/test_gates.jl`).

"Check `gate` is tagged for the bond `f` describes, as `apply_gate` does for `theta`."
function _check_gate(f::BondFrame, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    (gate.inds[1].itags == tl && gate.inds[2].itags == tr) || throw(ArgumentError(
        "gate is tagged for ($(gate.inds[1].itags), $(gate.inds[2].itags)), " *
        "bond frame carries ($tl, $tr)"))
    return nothing
end

"""
    sketch_bond_left(f, gate, Om) -> TLArray

`Y = (gate Theta) Om†`, legs `(link_l, site_l, g)` -- the left matricization of the gate
action sketched from the right. `Om` carries `V0`'s layout `(g, site_r, link_r)`, i.e. it is
a candidate right frame with `g` columns instead of `r`; `Om = f.V0` recovers the exact left
factor of `gate Theta`.

Contraction order: absorb `Om` into the gate's right BRA leg, then close the gate's right
KET leg and `link_r` against `V0`, then `K0 = U0 S0`. Nothing rank-4 is built.
"""
function sketch_bond_left(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)       # (link_l, site_l, site_r, link_r)
    c  = contract(f.U0', (1, 2), HT, (1, 2))            # (bond, site_r, link_r)
    A  = to_concrete(HT - to_concrete!(contract(f.U0, (3,), c, (1,))))
    return to_concrete!(contract(A, (3, 4), Om', (2, 3)))          # (link_l, site_l, g)
end

"""
    sketch_bond_right(f, gate, Om) -> TLArray

Mirror of [`sketch_bond_left`](@ref): `Om` carries `U0`'s layout `(link_l, site_l, g)` and
the result is `(g, site_r, link_r)`.
"""
function sketch_bond_right(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)
    c  = contract(HT, (3, 4), f.V0', (2, 3))            # (link_l, site_l, bond)
    A  = to_concrete(HT - to_concrete!(contract(c, (3,), f.V0, (1,))))
    Y  = contract(A, (1, 2), Om', (1, 2))               # (site_r, link_r, g)
    return to_concrete(permutedims(Y, (3, 1, 2)))       # (g, site_r, link_r)
end

"""
    cbe_expand_bond(f, gate; kwargs...) -> CBEExpansion

Controlled bond expansion of the frames at one bond, with the bare two-site gate standing in
for the environment-dressed effective Hamiltonian. Keywords are [`cbe_expand`](@ref)'s,
unchanged -- this is the same selection with different sketch closures.

The sketch probes the PROJECTOR (`P_perp (H Theta)`), which is the only path; there is no
`project_first` switch any more.
"""
cbe_expand_bond(f::BondFrame, gate; kwargs...) =
    cbe_expand(f,
               Om -> sketch_bond_left(f, gate, Om),
               Om -> sketch_bond_right(f, gate, Om); kwargs...)
