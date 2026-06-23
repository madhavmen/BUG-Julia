# bug_init.jl
#
# BUG integrator initialization:
#   - BUGInfo diagnostics struct
#   - BUG runtime constants (step rejection, expv backend)
#   - Composition-order constants (2nd, 3rd, 4th order only)
#   - _resolve_quantum_step_mode
#   - Bond snapshot helpers (_canonical_quantum_bond_snapshot,
#     _forward_quantum_bond_snapshot, _reverse_quantum_bond_snapshot,
#     open-boundary lifting)

# ── Diagnostics ───────────────────────────────────────────────────────────────

"""
    BUGInfo

Mutable diagnostics container populated by `bug_step!`.

Fields:
- `bond_dims_before/after`         — MPS bond dimensions
- `elapsed`                        — total wall time
- `forward_sweep_elapsed`          — time in forward (odd) passes
- `reverse_sweep_elapsed`          — time in reverse (even) passes
- `aug_sizes_k / aug_sizes_l`      — augmentation counts per bond update
- `aug_dims_k / aug_dims_l`        — live augmented K/L frame dimensions
- `lanczos_numops`                 — Krylov matvec counts (S-step)
- `backward_correction_calls`      — number of one-site backward solves
- `s_step_*`                       — per-bond truncation diagnostics
- `boundary_trace_*`               — boundary-bond frame diagnostics
"""
mutable struct BUGInfo
    bond_dims_before              :: Vector{Int}
    bond_dims_after               :: Vector{Int}
    elapsed                       :: Float64
    rhs_eval_elapsed              :: Float64
    trial_state_elapsed           :: Float64
    forward_sweep_elapsed         :: Float64
    reverse_sweep_elapsed         :: Float64
    aug_sizes_k                   :: Vector{Int}
    aug_sizes_l                   :: Vector{Int}
    aug_dims_k                    :: Vector{Int}
    aug_dims_l                    :: Vector{Int}
    lanczos_numops                :: Vector{Int}
    backward_correction_calls     :: Int
    s_step_sweeps                 :: Vector{Symbol}
    s_step_bonds                  :: Vector{Int}
    s_step_kept_ranks             :: Vector{Int}
    s_step_full_ranks             :: Vector{Int}
    s_step_tail_norms             :: Vector{Float64}
    s_step_max_discarded_svals    :: Vector{Float64}
    boundary_trace_sweeps         :: Vector{Symbol}
    boundary_trace_bonds          :: Vector{Int}
    boundary_trace_old_ranks      :: Vector{Int}
    boundary_trace_left_caps      :: Vector{Int}
    boundary_trace_right_caps     :: Vector{Int}
    boundary_trace_left_frame_ranks  :: Vector{Int}
    boundary_trace_right_frame_ranks :: Vector{Int}
    boundary_trace_s_full_ranks   :: Vector{Int}
    boundary_trace_n_new_k        :: Vector{Int}
    boundary_trace_n_new_l        :: Vector{Int}
end

BUGInfo() = BUGInfo(
    Int[], Int[], 0.0, 0.0, 0.0, 0.0, 0.0,
    Int[], Int[], Int[], Int[], Int[], 0,
    Symbol[], Int[], Int[], Int[], Float64[], Float64[],
    Symbol[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[],
)

# ── Runtime constants ──────────────────────────────────────────────────────────

const BUG_PROJECTED_LOCAL_DENSE_MAXDIM         = let
    raw = strip(get(ENV, "PS_BUG_PROJECTED_LOCAL_DENSE_MAXDIM", "4096"))
    isempty(raw) ? 4096 : parse(Int, raw)
end
const BUG_DEBUG_REVERSE_SNAPSHOT_ASSERTS       =
    strip(get(ENV, "PS_BUG_DEBUG_REVERSE_SNAPSHOT_ASSERTS", "1")) != "0"

# Discarded-projector BUG (discarded_bug.jl) augmentation tolerance: a Krylov
# direction is admitted into the K/L direct-sum frame only if its residual
# outside the old span exceeds BUG_DEFAULT_AUG_TOL. The faithful KLS path uses
# no such heuristic (see note below); this constant is consumed only by the
# discarded-projector column/row filters in bug_kls.jl.
const BUG_DEFAULT_AUG_TOL = 1e-12

# (BUG_DEFAULT_AUG_TOL removed — augmentation no longer uses a heuristic discard tolerance; new
#  Krylov directions are admitted down to the augmenting QR's machine-precision rank tolerance.)
# Default DENSE S-step. The matrix-free Krylov S-step (matrixfree_sstep=true) is the cure for
# the two-site-BUG O(maxdim^6) dense-S hang (project_twosite_bug_fix) and is used CALLER-SIDE
# by the two-site campaign sweep (faithful_base_sweep). It is NOT made the global default:
# flipping it regressed the single-site/general paths (test_bug_sweep_vs_exact,
# test_bug_multistep_xx_n6 — job 84581), so it stays opt-in where it is proven equivalent.
const BUG_DEFAULT_MATRIXFREE_SSTEP           = false

# ── Diagnostics recording helpers ─────────────────────────────────────────────

function _record_s_step_rank!(info::BUGInfo, sweep::Symbol, bond::Int, keep::Int, svals=nothing)
    push!(info.s_step_sweeps, sweep)
    push!(info.s_step_bonds, bond)
    push!(info.s_step_kept_ranks, keep)
    if isnothing(svals)
        push!(info.s_step_full_ranks, keep)
        push!(info.s_step_tail_norms, 0.0)
        push!(info.s_step_max_discarded_svals, 0.0)
    else
        full_rank = length(svals)
        push!(info.s_step_full_ranks, full_rank)
        if keep < full_rank
            discarded = @view svals[(keep+1):end]
            push!(info.s_step_tail_norms, sqrt(sum(abs2, discarded)))
            push!(info.s_step_max_discarded_svals, maximum(abs, discarded))
        else
            push!(info.s_step_tail_norms, 0.0)
            push!(info.s_step_max_discarded_svals, 0.0)
        end
    end
    return keep
end
_record_s_step_rank!(::Nothing, ::Symbol, ::Int, keep::Int, svals=nothing) = keep

function _augmented_kl_frame_dims(bond_data, candidate)
    left_mid = _left_site_bond_index(candidate.U_aug_tens, bond_data.link_l, bond_data.site_l)
    right_mid = _site_bond_index(candidate.V_aug_tens, bond_data.site_r, bond_data.link_r)
    return (k = dim(left_mid), l = dim(right_mid))
end

function _record_kl_augmentation!(info::BUGInfo, bond_data, candidate)
    dims = _augmented_kl_frame_dims(bond_data, candidate)
    push!(info.aug_sizes_k, candidate.n_new_k)
    push!(info.aug_sizes_l, candidate.n_new_l)
    push!(info.aug_dims_k, dims.k)
    push!(info.aug_dims_l, dims.l)
    return dims
end
_record_kl_augmentation!(::Nothing, bond_data, candidate) = _augmented_kl_frame_dims(bond_data, candidate)

function _record_backward_correction!(info::BUGInfo)
    info.backward_correction_calls += 1
    return info.backward_correction_calls
end
_record_backward_correction!(::Nothing) = 0

function _record_boundary_frame_trace!(info::BUGInfo, sweep::Symbol, bond::Int,
                                       bond_data, candidate, svals)
    left_cap  = dim(bond_data.link_l) * dim(bond_data.site_l)
    right_cap = dim(bond_data.site_r) * dim(bond_data.link_r)
    left_mid  = _left_site_bond_index(candidate.U_aug_tens, bond_data.link_l, bond_data.site_l)
    right_mid = _site_bond_index(candidate.V_aug_tens, bond_data.site_r, bond_data.link_r)
    push!(info.boundary_trace_sweeps, sweep)
    push!(info.boundary_trace_bonds, bond)
    push!(info.boundary_trace_old_ranks, dim(bond_data.link_mid))
    push!(info.boundary_trace_left_caps, left_cap)
    push!(info.boundary_trace_right_caps, right_cap)
    push!(info.boundary_trace_left_frame_ranks, dim(left_mid))
    push!(info.boundary_trace_right_frame_ranks, dim(right_mid))
    push!(info.boundary_trace_s_full_ranks, length(svals))
    push!(info.boundary_trace_n_new_k, candidate.n_new_k)
    push!(info.boundary_trace_n_new_l, candidate.n_new_l)
    return bond
end
_record_boundary_frame_trace!(::Nothing, ::Symbol, ::Int, bond_data, candidate, svals) = nothing

# ── Composition-order constants ───────────────────────────────────────────────
#
# Schedule entries are pairs (primitive_label, dt_coefficient). A `:forward`
# entry runs only the odd-bond left-to-right primitive sweep `F_odd(c*dt)`,
# while `:reverse` runs only the even-bond right-to-left primitive sweep
# `R_even(c*dt)`.
#
# Single-sweep schedule (sum = 1.0). Empirically, at full local saturation
# ONE F sweep advances the global state by approximately exp(-i·dt·H_total),
# not by exp(-i·dt·H_odd). The per-bond H_eff_at_b includes boundary
# contributions from adjacent bonds (H_{b-1,b}, H_{b+1,b+2}) that make
# H_eff_at_b ≈ projection of FULL H onto bond b's 2-site space — so one F
# sweep already captures (approximately) the full Hamiltonian evolution.
#
# Symmetric (2nd-order) FRF composition: forward/reverse/forward with coefficients (0.25, 0.5, 0.25).
const BUG_SECOND_ORDER_FRF = (
    (:forward, 0.25), (:reverse, 0.5), (:forward, 0.25),
)

"""
    _resolve_quantum_step_mode(step_mode) -> NamedTuple

Resolve a public BUG step mode to its sweep schedule. Currently supports the
symmetric 2nd-order FRF composition only.
"""
function _resolve_quantum_step_mode(step_mode::Symbol)
    schedule = step_mode === :second_order_frf   ? BUG_SECOND_ORDER_FRF  :
               error("Unknown BUG step_mode: $step_mode. " *
                     "Supported: :second_order_frf.")
    return (step_mode = step_mode, sweep_schedule = schedule)
end

# ── Helpers from BUGShared (boundary tensor helpers) ─────────────────────────

function _site_bond_index(V_tens::ITensor, s_next::Index, link_r::Index)
    out = nothing
    @inbounds for idx in inds(V_tens)
        if idx != s_next && idx != link_r
            isnothing(out) || error("Expected exactly one site bond index")
            out = idx
        end
    end
    isnothing(out) && error("Could not find site bond index in V tensor")
    return out
end

function _left_site_bond_index(U_tens::ITensor, link_l::Index, site_l::Index)
    out = nothing
    @inbounds for idx in inds(U_tens)
        if idx != link_l && idx != site_l
            isnothing(out) || error("Expected exactly one left-site bond index")
            out = idx
        end
    end
    isnothing(out) && error("Could not find left-site bond index in U tensor")
    return out
end

function _collapse_boundary_left_tensor(tens::ITensor, bond_data)
    hasproperty(bond_data, :boundary_left_collapse) || return tens
    return bond_data.boundary_left_collapse * tens
end

function _collapse_boundary_right_tensor(tens::ITensor, bond_data)
    hasproperty(bond_data, :boundary_right_collapse) || return tens
    return tens * bond_data.boundary_right_collapse
end

function _assert_left_isometry(
    U_tens::ITensor,
    link_l::Index,
    site_l::Index,
    mid::Index;
    atol::Float64 = 1e-10,
    label::AbstractString = "left factor",
)
    U_mat = reshape(
        ComplexF64.(Array(U_tens, link_l, site_l, mid)),
        dim(link_l) * dim(site_l),
        dim(mid),
    )
    err = norm(U_mat' * U_mat - Matrix{ComplexF64}(I, dim(mid), dim(mid)))
    @assert err <= atol "$label isometry check failed: left error = $err, atol = $atol"
    return nothing
end

function _assert_right_isometry(
    V_tens::ITensor,
    mid::Index,
    site_r::Index,
    link_r::Index;
    atol::Float64 = 1e-10,
    label::AbstractString = "right factor",
)
    V_mat = reshape(
        ComplexF64.(Array(V_tens, mid, site_r, link_r)),
        dim(mid),
        dim(site_r) * dim(link_r),
    )
    err = norm(V_mat * V_mat' - Matrix{ComplexF64}(I, dim(mid), dim(mid)))
    @assert err <= atol "$label isometry check failed: right error = $err, atol = $atol"
    return nothing
end

# ── Bond snapshot helpers ─────────────────────────────────────────────────────

function _boundary_env_aux_index(env::ITensor, boundary_link::Index)
    aux_inds = [idx for idx in inds(env)
                if idx != boundary_link && idx != prime(boundary_link)]
    length(aux_inds) == 1 || error("Expected exactly one auxiliary MPO boundary index")
    return only(aux_inds)
end

"""
    _canonical_quantum_bond_snapshot(psi, W, bond, L_mpo_cur, R_mpo_cur; kwargs...)

Extract the two-site bond data (U0, S0, V0, environments) for a BUG local
update at `bond` in canonical gauge.
"""
function _canonical_quantum_bond_snapshot(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bond::Int,
    L_mpo_cur::ITensor,
    R_mpo_cur::ITensor;
    include_theta0::Bool = true,
)
    bond_inds = _tensortrain_bond_indices(psi, bond)
    link_l    = bond_inds.link_l
    link_mid  = bond_inds.link_mid
    link_r    = bond_inds.link_r
    site_l    = bond_inds.site_l
    site_r    = bond_inds.site_r

    U0_tens, S_left_tens = qr(psi[bond], link_l, site_l;
                               tags=join(string.(tags(link_mid)), ","), positive=false)
    canon_u0 = commonind(U0_tens, S_left_tens)

    S_right_tens, V0_tens = lq(psi[bond+1], site_r, link_r)
    canon_v0 = commonind(S_right_tens, V0_tens)
    if tags(canon_v0) != tags(link_mid)
        new_canon_v0 = settags(canon_v0, join(string.(tags(link_mid)), ","))
        S_right_tens = replaceind(S_right_tens, canon_v0, new_canon_v0)
        V0_tens      = replaceind(V0_tens,      canon_v0, new_canon_v0)
        canon_v0     = new_canon_v0
    end

    S0_tens     = S_left_tens * S_right_tens
    theta0_tens = include_theta0 ? (U0_tens * S0_tens * V0_tens) : nothing

    snapshot = (
        link_l    = link_l,
        link_mid  = link_mid,
        link_r    = link_r,
        site_l    = site_l,
        site_r    = site_r,
        U0_tens   = U0_tens,
        V0_tens   = V0_tens,
        S_left_tens  = S_left_tens,
        S_right_tens = S_right_tens,
        S0_tens   = S0_tens,
        theta0_tens  = theta0_tens,
        canon_u0  = canon_u0,
        canon_v0  = canon_v0,
        L_mpo_cur = L_mpo_cur,
        R_mpo_cur = R_mpo_cur,
        W_left    = W[bond],
        W_right   = W[bond+1],
    )
    return snapshot
end

function _svd_quantum_bond_snapshot(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bond::Int,
    L_mpo_cur::ITensor,
    R_mpo_cur::ITensor;
    include_theta0::Bool = true,
)
    bond_inds = _tensortrain_bond_indices(psi, bond)
    link_l    = bond_inds.link_l
    link_r    = bond_inds.link_r
    site_l    = bond_inds.site_l
    site_r    = bond_inds.site_r

    theta = psi[bond] * psi[bond + 1]
    U0_tens, S0_tens, V0_tens = svd(theta, (link_l, site_l); cutoff = 0.0)
    canon_u0 = commonind(U0_tens, S0_tens)
    canon_v0 = commonind(S0_tens, V0_tens)
    theta0_tens = include_theta0 ? theta : nothing

    return (
        link_l = link_l,
        link_mid = canon_u0,
        link_r = link_r,
        site_l = site_l,
        site_r = site_r,
        U0_tens = U0_tens,
        V0_tens = V0_tens,
        S0_tens = S0_tens,
        theta0_tens = theta0_tens,
        canon_u0 = canon_u0,
        canon_v0 = canon_v0,
        L_mpo_cur = L_mpo_cur,
        R_mpo_cur = R_mpo_cur,
        W_left = W[bond],
        W_right = W[bond + 1],
    )
end

function _quantum_bond_snapshot(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bond::Int,
    L_mpo_cur::ITensor,
    R_mpo_cur::ITensor;
    snapshot_mode::Symbol = :qr_lq,
    include_theta0::Bool = true,
)
    if snapshot_mode === :qr_lq
        return _canonical_quantum_bond_snapshot(
            psi, W, bond, L_mpo_cur, R_mpo_cur;
            include_theta0 = include_theta0,
        )
    elseif snapshot_mode === :svd
        return _svd_quantum_bond_snapshot(
            psi, W, bond, L_mpo_cur, R_mpo_cur;
            include_theta0 = include_theta0,
        )
    end
    error("Unknown BUG snapshot_mode: $snapshot_mode. Supported: :qr_lq, :svd.")
end

"""
    _forward_quantum_bond_snapshot(psi, W, bond, L_mpo_cur, R_mpo_cur; reuse_left_isometry, ...)

Forward-sweep specialization: after the first bond, the transported gauge can
be reused directly (skip QR of the left tensor).
"""
function _forward_quantum_bond_snapshot(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bond::Int,
    L_mpo_cur::ITensor,
    R_mpo_cur::ITensor;
    reuse_left_isometry::Bool,
)
    reuse_left_isometry || return _canonical_quantum_bond_snapshot(
        psi, W, bond, L_mpo_cur, R_mpo_cur,
    )

    bond_inds = _tensortrain_bond_indices(psi, bond)
    link_l    = bond_inds.link_l
    link_mid  = bond_inds.link_mid
    link_r    = bond_inds.link_r
    site_l    = bond_inds.site_l
    site_r    = bond_inds.site_r

    canon_u0 = link_mid
    U0_tens  = psi[bond]

    S_right_tens, V0_tens = lq(psi[bond+1], site_r, link_r)
    canon_v0 = commonind(S_right_tens, V0_tens)
    if tags(canon_v0) != tags(link_mid)
        new_canon_v0 = settags(canon_v0, join(string.(tags(link_mid)), ","))
        S_right_tens = replaceind(S_right_tens, canon_v0, new_canon_v0)
        V0_tens      = replaceind(V0_tens,      canon_v0, new_canon_v0)
        canon_v0     = new_canon_v0
    end
    S0_tens = S_right_tens

    snapshot = (
        link_l    = link_l,
        link_mid  = link_mid,
        link_r    = link_r,
        site_l    = site_l,
        site_r    = site_r,
        U0_tens   = U0_tens,
        V0_tens   = V0_tens,
        S0_tens   = S0_tens,
        canon_u0  = canon_u0,
        canon_v0  = canon_v0,
        L_mpo_cur = L_mpo_cur,
        R_mpo_cur = R_mpo_cur,
        W_left    = W[bond],
        W_right   = W[bond+1],
    )
    return snapshot
end

"""
    _reverse_quantum_bond_snapshot(psi, W, bond, L_mpo_cur, R_mpo_cur; reuse_right_isometry, ...)

Reverse-sweep specialization: optionally reuse the transported right isometry.
"""
function _reverse_quantum_bond_snapshot(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bond::Int,
    L_mpo_cur::ITensor,
    R_mpo_cur::ITensor;
    reuse_right_isometry::Bool,
)
    reuse_right_isometry || return _canonical_quantum_bond_snapshot(
        psi, W, bond, L_mpo_cur, R_mpo_cur,
    )

    bond_inds = _tensortrain_bond_indices(psi, bond)
    link_l    = bond_inds.link_l
    link_mid  = bond_inds.link_mid
    link_r    = bond_inds.link_r
    site_l    = bond_inds.site_l
    site_r    = bond_inds.site_r

    U0_tens, S_left_tens = qr(psi[bond], link_l, site_l; tags=join(string.(tags(link_mid)), ","), positive=false)
    canon_u0 = commonind(U0_tens, S_left_tens)
    V0_tens  = psi[bond+1]
    canon_v0 = link_mid
    S0_tens  = S_left_tens

    if BUG_DEBUG_REVERSE_SNAPSHOT_ASSERTS
        _assert_left_isometry(
            U0_tens, link_l, site_l, canon_u0;
            label = "_reverse_quantum_bond_snapshot(U0_tens)",
        )
        _assert_right_isometry(
            V0_tens, canon_v0, site_r, link_r;
            label = "_reverse_quantum_bond_snapshot(V0_tens)",
        )
    end

    snapshot = (
        link_l    = link_l,
        link_mid  = link_mid,
        link_r    = link_r,
        site_l    = site_l,
        site_r    = site_r,
        U0_tens   = U0_tens,
        V0_tens   = V0_tens,
        S0_tens   = S0_tens,
        canon_u0  = canon_u0,
        canon_v0  = canon_v0,
        L_mpo_cur = L_mpo_cur,
        R_mpo_cur = R_mpo_cur,
        W_left    = W[bond],
        W_right   = W[bond+1],
    )
    return snapshot
end
