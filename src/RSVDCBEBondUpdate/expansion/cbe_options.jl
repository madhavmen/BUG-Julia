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

Controls for one run of any of the three modes, mirroring `BondUpdateOptions`. `dt` is a real step
(`tau = -im*dt`) in real time and an inverse-temperature step (`tau = -dt`) in imaginary time.
`maxdim`/`trunc_thresh` set the truncation; `normalize` rescales after each step (the norm is
recorded first). `maxiter`/`tol` budget the Krylov exponential; `seed` is one RNG per run.

There is no `order` field: the MPO path takes one projected exponential per step, so there is no
splitting to symmetrise. Ground state runs on the MPO path only, since a local eigensolve on a bare
gate would minimise a single term with frozen neighbours.
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

Per-step record of a CBE-BUG run; every field has length `n_steps`. `max_expanded` is the
headline rank diagnostic: the largest bond proposed before truncation (`r + Dex`). `krylov_dims` and
`mean_aug` form the cost pair, operator applications and the size they act on; wall time is roughly
their product.
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
# ── PART II: CBE as the basis update of the gate-based bond_update_bug ──
# The three modes differ only in the operator posed to `_expanding_krylov` and in the S-step solver;
# the growth, sketch, projection, ranking and reorthogonalisation are shared code.

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

The lowest Ritz vector of `H_eff` in the accumulated Krylov space: a DMRG local solve that replaces
the exponential and keeps the basis. The result is normalised, since the sweep minimises a unit-norm
state.

The operator must include the environments (`L + h + R`). Given the bare two-site gate, each bond
would minimise its own pair and the next bond would undo it; see `apply_h_two_site` in `henv.jl`.
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

# ── CBE as the basis update of the validated gate-based integrator ──
# A gate-only path has no environments, so a rank stall here isolates the CBE selection itself, which
# is shared with the MPO path. It replaces the K/L ODE solves and the random sector fill of
# `kls_bond_update`.

# ── the sketched gate action ──
# Never forms H*Theta: `Om` is folded into the gate first, so the largest intermediate is O(d^2 g r).
# Checked against `P_perp` of `apply_gate` on Theta, for arbitrary `Om`.

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
