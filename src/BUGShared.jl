# Shared BUG constants and helpers.
# NOTE: Constants and wrapper functions are now defined in TTutils.krylov_utils.jl
# where they are properly accessible to BUG and TDVP modules.
# This file is retained for reference but is not actively included.

const BUG_DEFAULT_DELTA_SELECTOR = :singular_value

"""
    BUGInfo

Mutable diagnostics container populated by the BUG sweeps.

The fields are grouped by concern:
- bond dimensions before and after the public step
- elapsed timings for RHS assembly, trial states, and forward/reverse sweeps
- augmentation sizes from the K- and L-steps
- Krylov matvec counts
- S-step truncation records per visited bond
"""
mutable struct BUGInfo
    bond_dims_before :: Vector{Int}
    bond_dims_after  :: Vector{Int}
    elapsed          :: Float64
    rhs_eval_elapsed :: Float64
    trial_state_elapsed :: Float64
    forward_sweep_elapsed :: Float64
    reverse_sweep_elapsed :: Float64
    aug_sizes_k      :: Vector{Int}
    aug_sizes_l      :: Vector{Int}
    lanczos_numops   :: Vector{Int}
    s_step_sweeps    :: Vector{Symbol}
    s_step_bonds     :: Vector{Int}
    s_step_kept_ranks:: Vector{Int}
    s_step_full_ranks:: Vector{Int}
    s_step_tail_norms:: Vector{Float64}
    s_step_max_discarded_svals:: Vector{Float64}
    boundary_trace_sweeps::Vector{Symbol}
    boundary_trace_bonds::Vector{Int}
    boundary_trace_old_ranks::Vector{Int}
    boundary_trace_left_caps::Vector{Int}
    boundary_trace_right_caps::Vector{Int}
    boundary_trace_left_frame_ranks::Vector{Int}
    boundary_trace_right_frame_ranks::Vector{Int}
    boundary_trace_s_full_ranks::Vector{Int}
    boundary_trace_n_new_k::Vector{Int}
    boundary_trace_n_new_l::Vector{Int}
end

"""
    BUGInfo()

Construct an empty diagnostics record ready to be filled during a BUG step.
"""
BUGInfo() = BUGInfo(
    Int[], Int[], 0.0, 0.0, 0.0, 0.0, 0.0, Int[], Int[], Int[], Symbol[], Int[], Int[], Int[], Float64[], Float64[], Symbol[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[],
)

function _hermitian_tridiagonal_exp_coeffs(
    alpha::AbstractVector{<:Real},
    beta::AbstractVector{<:Real},
    dt,
)
    T = SymTridiagonal(Vector{Float64}(alpha), Vector{Float64}(beta))
    F = eigen(T)
    weights = exp.(dt .* F.values) .* ComplexF64.(F.vectors[1, :])
    return ComplexF64.(F.vectors * weights)
end

function _native_hermitian_lanczos_exponentiate(
    matvec,
    dt,
    x::AbstractVector;
    tol::Float64,
    krylovdim::Int,
)
    n = length(x)
    n == 0 && return similar(x, ComplexF64, 0), 0

    x_work = Vector{ComplexF64}(undef, n)
    copyto!(x_work, x)
    norm_x = norm(x_work)
    norm_x == 0 && return fill!(similar(x_work), 0.0 + 0.0im), 0

    mmax = min(max(krylovdim, 1), n)
    basis = Matrix{ComplexF64}(undef, n, mmax)
    alpha = Vector{Float64}(undef, mmax)
    beta = Vector{Float64}(undef, max(mmax - 1, 0))
    work = Vector{ComplexF64}(undef, n)
    basis[:, 1] .= x_work ./ norm_x
    numops = 0
    final_dim = 1

    for j in 1:mmax
        vj = @view basis[:, j]
        copyto!(work, matvec(vj))
        numops += 1
        if j > 1
            LinearAlgebra.axpy!(-beta[j - 1], @view(basis[:, j - 1]), work)
        end
        alpha[j] = real(dot(vj, work))
        LinearAlgebra.axpy!(-alpha[j], vj, work)

        if j == mmax
            final_dim = j
            break
        end

        beta_j = norm(work)
        if beta_j <= tol
            final_dim = j
            break
        end
        beta[j] = beta_j
        basis[:, j + 1] .= work ./ beta_j
        final_dim = j + 1
    end

    coeff = _hermitian_tridiagonal_exp_coeffs(alpha[1:final_dim], beta[1:max(final_dim - 1, 0)], dt)
    y = norm_x .* (basis[:, 1:final_dim] * coeff)
    return y, numops
end

_record_rhs_eval_elapsed!(::Nothing, elapsed::Float64) = elapsed
_record_rhs_eval_elapsed!(info::BUGInfo, elapsed::Float64) = (info.rhs_eval_elapsed += elapsed; elapsed)

_record_aug_k!(::Nothing, n_new::Int) = n_new
_record_aug_k!(info::BUGInfo, n_new::Int) = (push!(info.aug_sizes_k, n_new); n_new)

_record_aug_l!(::Nothing, n_new::Int) = n_new
_record_aug_l!(info::BUGInfo, n_new::Int) = (push!(info.aug_sizes_l, n_new); n_new)

_record_lanczos_numops!(::Nothing, numops::Int) = numops
_record_lanczos_numops!(info::BUGInfo, numops::Int) = (push!(info.lanczos_numops, numops); numops)

"""
    _linear_substep(H_or_matvec, dt, x; method, lanczos_tol, lanczos_maxiter, restart=1)

Advance a local linear ODE `x' = H*x` for one BUG micro-step and return
`(x_next, numops)`. The `numops` counter is used as a cheap diagnostic for
how much Krylov work the local step required.

This helper assumes the effective operator is Hermitian when `method=:expv`.
Use `_general_linear_substep` when the symmetry assumption is not valid.
"""
function _linear_substep(
    H::AbstractMatrix,
    dt,
    x::AbstractVector;
    method::Symbol,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    restart::Int = 1,
)
    return _linear_substep(v -> H * v, dt, x;
        method = method,
        lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter,
        restart = restart,
    )
end

function _linear_substep(
    matvec,
    dt,
    x::AbstractVector;
    method::Symbol,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    restart::Int = 1,
)
    if method === :expv
        if _active_bug_expv_backend() === :native_hermitian_lanczos
            try
                return _native_hermitian_lanczos_exponentiate(
                    matvec,
                    dt,
                    x;
                    tol = lanczos_tol,
                    krylovdim = lanczos_maxiter,
                )
            catch err
                if !(err isa LinearAlgebra.LAPACKException)
                    rethrow()
                end
                # Native tridiagonal diagonalization occasionally fails on
                # otherwise valid reduced BUG operators. Fall back to
                # KrylovKit so the local substep still completes.
            end
        end
        y, info = KrylovKit.exponentiate(matvec, dt, x;
            krylovdim = lanczos_maxiter, tol = lanczos_tol,
            maxiter = restart, issymmetric = true, eager = false)
        return y, info.numops
    elseif method === :euler
        y = x + dt * matvec(x)
        return y, 1
    elseif method === :rk4
        k1 = matvec(x)
        k2 = matvec(x + (dt / 2) * k1)
        k3 = matvec(x + (dt / 2) * k2)
        k4 = matvec(x + dt * k3)
        y = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y, 4
    else
        error("Unknown BUGIntegrator substep method: $method. Supported methods are :expv, :euler, and :rk4.")
    end
end

"""
    _general_linear_substep(matvec, dt, x; method, lanczos_tol, lanczos_maxiter, restart=1, issymmetric)

Variant of `_linear_substep` that keeps the caller in control of whether the
effective operator should be treated as symmetric by KrylovKit.
"""
# rename to mat_intergrator()
function _general_linear_substep(
    matvec,
    dt,
    x::AbstractVector;
    method::Symbol,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    restart::Int = 1,
    issymmetric::Bool,
)
    if method === :expv
        if issymmetric && _active_bug_expv_backend() === :native_hermitian_lanczos
            return _native_hermitian_lanczos_exponentiate(
                matvec,
                dt,
                x;
                tol = lanczos_tol,
                krylovdim = min(length(x), max(lanczos_maxiter, 4)),
            )
        end
        y, info = KrylovKit.exponentiate(matvec, dt, x;
            krylovdim = min(length(x), max(lanczos_maxiter, 4)),
            tol = lanczos_tol,
            maxiter = restart,
            issymmetric = issymmetric,
            eager = false,
        )
        return y, info.numops
    elseif method === :euler
        y = x + dt * matvec(x)
        return y, 1
    elseif method === :rk4
        k1 = matvec(x)
        k2 = matvec(x + (dt / 2) * k1)
        k3 = matvec(x + (dt / 2) * k2)
        k4 = matvec(x + dt * k3)
        y = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y, 4
    end

    error("Unknown BUGIntegrator substep method: $method. Supported methods are :expv, :euler, and :rk4.")
end

function _tensortrain_boundary_link(ψ::TensorTrain, side::Symbol)
    N = length(ψ)
    N < 2 && error("_tensortrain_boundary_link requires at least two sites")
    side === :left && return only(uniqueinds(ψ[1], ψ[2]; tags = "Link"))
    side === :right && return only(uniqueinds(ψ[N], ψ[N - 1]; tags = "Link"))
    error("side must be :left or :right")
end

function _tensortrain_site_index(ψ::TensorTrain, k::Int)
    N = length(ψ)
    N < 2 && error("_tensortrain_site_index requires at least two sites")
    if k == 1
        link_l = _tensortrain_boundary_link(ψ, :left)
        link_r = commonind(ψ[1], ψ[2])
    elseif k == N
        link_l = commonind(ψ[N - 1], ψ[N])
        link_r = _tensortrain_boundary_link(ψ, :right)
    else
        link_l = commonind(ψ[k - 1], ψ[k])
        link_r = commonind(ψ[k], ψ[k + 1])
    end
    return only(uniqueinds(ψ[k], [link_l, link_r]))
end

function _tensortrain_bond_indices(ψ::TensorTrain, bond::Int)
    left_tens = ψ[bond]
    right_tens = ψ[bond + 1]
    link_mid = commonind(left_tens, right_tens)
    link_l = only(uniqueinds(left_tens, right_tens; tags = "Link"))
    link_r = only(uniqueinds(right_tens, left_tens; tags = "Link"))
    site_l = only(uniqueinds(left_tens, [link_l, link_mid]))
    site_r = only(uniqueinds(right_tens, [link_mid, link_r]))
    return (; link_l, link_mid, link_r, site_l, site_r)
end

@inline function _complex_tensor_array(T::ITensor, inds...)
    A = Array(T, inds...)
    return eltype(A) <: ComplexF64 ? A : ComplexF64.(A)
end

@inline function _complex_tensor_vec(T::ITensor, inds...)
    A = Array(T, inds...)
    return eltype(A) <: ComplexF64 ? vec(A) : ComplexF64.(vec(A))
end

function _qr_nonzero_diagonal_rank(A::AbstractMatrix)
    isempty(A) && return 0
    F = qr(A)
    ndiag = min(size(A)...)
    ndiag <= 0 && return 0
    scale = zero(Float64)
    diagvals = Vector{Float64}(undef, ndiag)
    @inbounds for i in 1:ndiag
        val = abs(F.R[i, i])
        diagvals[i] = val
        scale = max(scale, val)
    end
    tol = max(size(A)...) * eps(real(eltype(A))) * max(scale, 1.0)
    return count(>(tol), diagvals)
end

function _qr_column_basis(A::AbstractMatrix)
    isempty(A) && return Matrix{eltype(A)}(undef, size(A, 1), 0), 0
    F = qr(A)
    ndiag = min(size(A)...)
    diagvals = Vector{Float64}(undef, ndiag)
    scale = zero(Float64)
    @inbounds for i in 1:ndiag
        val = abs(F.R[i, i])
        diagvals[i] = val
        scale = max(scale, val)
    end
    tol = max(size(A)...) * eps(real(eltype(A))) * max(scale, 1.0)
    rank = count(>(tol), diagvals)
    rank <= 0 && return Matrix{eltype(A)}(undef, size(A, 1), 0), 0
    return Matrix(F.Q[:, 1:rank]), rank
end

function _qr_row_basis(A::AbstractMatrix)
    isempty(A) && return Matrix{eltype(A)}(undef, 0, size(A, 2)), 0
    F = qr(adjoint(A))
    ndiag = min(size(A)...)
    diagvals = Vector{Float64}(undef, ndiag)
    scale = zero(Float64)
    @inbounds for i in 1:ndiag
        val = abs(F.R[i, i])
        diagvals[i] = val
        scale = max(scale, val)
    end
    tol = max(size(A)...) * eps(real(eltype(A))) * max(scale, 1.0)
    rank = count(>(tol), diagvals)
    rank <= 0 && return Matrix{eltype(A)}(undef, 0, size(A, 2)), 0
    return Matrix(adjoint(F.Q[:, 1:rank])), rank
end

function _identity_overlap_matrix(T_el::Type, n::Int)
    out = Matrix{T_el}(undef, n, n)
    fill!(out, zero(T_el))
    @inbounds for i in 1:n
        out[i, i] = one(T_el)
    end
    return out
end

function _complete_column_basis(Q::AbstractMatrix, target_cols::Int)
    current = size(Q, 2)
    current >= target_cols && return Matrix(Q[:, 1:target_cols])

    ambient = size(Q, 1)
    T_el = eltype(Q)
    out = Matrix{T_el}(undef, ambient, target_cols)
    if current > 0
        @views out[:, 1:current] .= Q
    end

    filled = current
    tol = max(ambient, target_cols) * eps(Float64)
    for idx in 1:ambient
        filled >= target_cols && break
        candidate = zeros(T_el, ambient)
        candidate[idx] = one(T_el)
        if filled > 0
            @views candidate .-= out[:, 1:filled] * (out[:, 1:filled]' * candidate)
        end
        nrm = norm(candidate)
        nrm <= tol && continue
        @views out[:, filled + 1] .= candidate ./ nrm
        filled += 1
    end

    filled == target_cols ||
        error("Could not complete column basis from rank $current to requested rank $target_cols")
    return out
end

function _complete_row_basis(B::AbstractMatrix, target_rows::Int)
    current = size(B, 1)
    current >= target_rows && return Matrix(B[1:target_rows, :])
    return Matrix(adjoint(_complete_column_basis(adjoint(B), target_rows)))
end

"""
    _split_s_step_bond(S_new, left_ind, right_ind; maxdim, cutoff)

Split the updated S-tensor for a forward sweep into a left factor `U_s` and a
right factor carrying the singular values. This helper currently keeps the full
SVD rank so the newly added BUG directions survive the immediate write-back.
"""
function _split_s_step_bond(
    S_new::ITensor,
    left_ind::Index,
    right_ind::Index;
    maxdim::Int,
    cutoff::Float64,
)
    S_mat = Array(S_new, left_ind, right_ind)
    F = svd!(S_mat; full = false)
    # Keep full SVD rank — do NOT truncate with maxdim/cutoff.
    # The BUG augmentation grows the basis; truncation here would
    # discard the new directions before they can contribute.
    keep = length(F.S)
    keep > 0 || error("S-step SVD produced zero rank")

    bond_ind = Index(keep, tags(left_ind))
    U_s = itensor(Matrix(@view F.U[:, 1:keep]), left_ind, bond_ind)
    SV_mat = Matrix(@view F.Vt[1:keep, :])
    @inbounds for row in 1:keep
        @views SV_mat[row, :] .*= F.S[row]
    end
    SV_tens = itensor(SV_mat, bond_ind, right_ind)
    return U_s, SV_tens, keep, F.S
end

"""
    _split_s_step_bond_reverse(S_new, left_ind, right_ind; maxdim, cutoff)

Reverse-sweep analogue of `_split_s_step_bond` that keeps the singular values
on the left factor so the gauge can be transported to the previous site.
"""
function _split_s_step_bond_reverse(
    S_new::ITensor,
    left_ind::Index,
    right_ind::Index;
    maxdim::Int,
    cutoff::Float64,
)
    S_mat = Array(S_new, left_ind, right_ind)
    F = svd!(S_mat; full = false)
    # Keep full SVD rank — no truncation (see _split_s_step_bond).
    keep = length(F.S)
    keep > 0 || error("S-step SVD produced zero rank")

    bond_ind = Index(keep, tags(right_ind))
    US_mat = Matrix(@view F.U[:, 1:keep])
    @inbounds for col in 1:keep
        @views US_mat[:, col] .*= F.S[col]
    end
    US_tens = itensor(US_mat, left_ind, bond_ind)
    V_tens = itensor(Matrix(@view F.Vt[1:keep, :]), bond_ind, right_ind)
    return US_tens, V_tens, keep, F.S
end

"""
    _record_s_step_rank!(info, sweep, bond, keep, svals=nothing)

Store per-bond truncation diagnostics from the S-step. When `svals` is
available, the helper also records the discarded tail norm and largest
discarded singular value.
"""
function _record_s_step_rank!(info::BUGInfo, sweep::Symbol, bond::Int, keep::Int, svals = nothing)
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
            discarded = @view svals[(keep + 1):end]
            push!(info.s_step_tail_norms, sqrt(sum(abs2, discarded)))
            push!(info.s_step_max_discarded_svals, maximum(abs, discarded))
        else
            push!(info.s_step_tail_norms, 0.0)
            push!(info.s_step_max_discarded_svals, 0.0)
        end
    end
    return keep
end

_record_s_step_rank!(::Nothing, ::Symbol, ::Int, keep::Int, svals = nothing) = keep

function _record_boundary_frame_trace!(
    info::BUGInfo,
    sweep::Symbol,
    bond::Int,
    bond_data,
    candidate,
    svals,
)
    left_cap = dim(bond_data.link_l) * dim(bond_data.site_l)
    right_cap = dim(bond_data.site_r) * dim(bond_data.link_r)
    left_mid = _left_site_bond_index(candidate.U_aug_tens, bond_data.link_l, bond_data.site_l)
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

function _pick_left_update(
    U0_mat,
    K1_mat;
    aug_tol::Float64,
)
    rm = size(U0_mat, 2)

    # Basis growth is determined by the numerical rank of the combined span.
    # Keep `aug_tol` only as a compatibility kwarg for existing call sites.
    _ = aug_tol
    T_el = promote_type(eltype(U0_mat), eltype(K1_mat))

    # Use the orthogonal projector update only: build an orthonormal basis of
    # the evolved K-step space, then add the previous-time basis and QR the
    # combined span.
    Qk, rk = _qr_column_basis(K1_mat)
    rk <= 0 && return U0_mat, 0, _identity_overlap_matrix(T_el, rm)

    Q_aug_mat, aug_rank = _qr_column_basis(hcat(Qk, U0_mat))
    aug_rank <= 0 && return U0_mat, 0, _identity_overlap_matrix(T_el, rm)
    # No padding: return the augmented basis at its true rank. Truncation back
    # to the old rank or any user cap belongs to the post-S-step SVD, never the
    # K/L augmentation (Ceruti–Kusch–Lubich, arXiv:2304.05660, step 3).
    n_new = max(0, aug_rank - rm)
    return Q_aug_mat, n_new, Q_aug_mat' * U0_mat
end

function _pick_right_update(
    V0_mat,
    L1_mat;
    aug_tol::Float64,
)
    rm = size(V0_mat, 1)

    # Basis growth is determined by the numerical rank of the combined span.
    # Keep `aug_tol` only as a compatibility kwarg for existing call sites.
    _ = aug_tol
    T_el = promote_type(eltype(V0_mat), eltype(L1_mat))

    # Right-basis analogue of the projected update above.
    Ql, rl = _qr_row_basis(L1_mat)
    rl <= 0 && return V0_mat, 0, _identity_overlap_matrix(T_el, rm)

    B_aug_mat, aug_rank = _qr_row_basis(vcat(Ql, V0_mat))
    aug_rank <= 0 && return V0_mat, 0, _identity_overlap_matrix(T_el, rm)
    # No padding — see _pick_left_update.
    n_new = max(0, aug_rank - rm)
    return B_aug_mat, n_new, B_aug_mat * V0_mat'
end

function _left_isometry_from_k(
    U0_tens :: ITensor,
    K1_tens :: ITensor,
    lψ_l    :: Index,
    s_ℓ     :: Index,
    old_mid :: Index;
    maxdim  :: Int = typemax(Int),
    aug_tol :: Float64,
    augmentation_mode::Symbol = :projected,
)
    _ = maxdim
    _ = augmentation_mode
    dl = dim(lψ_l)
    ds = dim(s_ℓ)
    rm = dim(old_mid)
    U0_mat = reshape(_complex_tensor_array(U0_tens, lψ_l, s_ℓ, old_mid), dl * ds, rm)
    K1_mat = reshape(_complex_tensor_array(K1_tens, lψ_l, s_ℓ, old_mid), dl * ds, rm)
    A_aug_mat, n_new, _ = _pick_left_update(U0_mat, K1_mat;
        aug_tol = aug_tol,
    )

    aug_lnk = Index(size(A_aug_mat, 2), tags(old_mid))
    U1_tens = itensor(reshape(A_aug_mat, dl, ds, size(A_aug_mat, 2)), lψ_l, s_ℓ, aug_lnk)
    return U1_tens, n_new
end

function _right_isometry_from_l(
    V0_tens :: ITensor,
    L1_tens :: ITensor,
    old_mid :: Index,
    s_ℓ1    :: Index,
    lψ_r    :: Index;
    aug_tol :: Float64,
)
    rm = dim(old_mid)
    ds1 = dim(s_ℓ1)
    dr1 = dim(lψ_r)
    V0_mat = reshape(_complex_tensor_array(V0_tens, old_mid, s_ℓ1, lψ_r), rm, ds1 * dr1)
    L1_mat = reshape(_complex_tensor_array(L1_tens, old_mid, s_ℓ1, lψ_r), rm, ds1 * dr1)
    B_aug_mat, n_new, _ = _pick_right_update(V0_mat, L1_mat;
        aug_tol = aug_tol,
    )

    keep = size(B_aug_mat, 1)
    keep > 0 || error("Right-isometry QR produced zero rank")
    canon_lnk = Index(keep, tags(old_mid))
    V1_tens = itensor(reshape(B_aug_mat, keep, ds1, dr1), canon_lnk, s_ℓ1, lψ_r)
    return V1_tens, n_new
end

function _right_isometry_seed_from_l(
    L0_tens   :: ITensor,
    basis_mid :: Index,
    s_ℓ1      :: Index,
    lψ_r      :: Index,
)
    L0_mat = Array(L0_tens, basis_mid, s_ℓ1, lψ_r)
    L0_rowmat = reshape(L0_mat, dim(basis_mid), dim(s_ℓ1) * dim(lψ_r))
    F = qr(L0_rowmat')
    keep = min(size(L0_rowmat, 1), size(L0_rowmat, 2))
    keep > 0 || error("Right-seed QR produced zero rank")
    V0_mat = Matrix(F.Q[:, 1:keep])'
    canon_lnk = Index(keep, tags(basis_mid))
    V0_seed_tens = itensor(reshape(V0_mat, keep, dim(s_ℓ1), dim(lψ_r)), canon_lnk, s_ℓ1, lψ_r)
    return V0_seed_tens
end

function _left_isometry_seed_from_k(
    K0_tens   :: ITensor,
    lψ_l      :: Index,
    s_ℓ       :: Index,
    basis_mid :: Index,
)
    U0_seed_tens, qr_rest = qr(K0_tens, lψ_l, s_ℓ; tags = tags(basis_mid), positive = false)
    qr_lnk = commonind(U0_seed_tens, qr_rest)
    canon_lnk = settags(qr_lnk, tags(basis_mid))
    U0_seed_tens = replaceind(U0_seed_tens, qr_lnk, canon_lnk)
    return U0_seed_tens
end

function _augment_left_isometries(
    U_step_tens :: ITensor,
    U_old_tens  :: ITensor,
    lψ_l        :: Index,
    s_ℓ         :: Index,
    step_mid    :: Index,
    old_mid     :: Index,
)
    dl = dim(lψ_l)
    ds = dim(s_ℓ)
    U_step_mat = reshape(_complex_tensor_array(U_step_tens, lψ_l, s_ℓ, step_mid), dl * ds, dim(step_mid))
    U_old_mat = reshape(_complex_tensor_array(U_old_tens, lψ_l, s_ℓ, old_mid), dl * ds, dim(old_mid))
    U_old_perp = U_old_mat .- U_step_mat * (U_step_mat' * U_old_mat)
    F = qr(U_old_perp)
    n_added = _qr_nonzero_diagonal_rank(U_old_perp)
    n_added <= 0 && return U_step_tens, 0

    ΔU = Matrix(F.Q[:, 1:n_added])
    U_aug_mat = hcat(U_step_mat, ΔU)
    aug_lnk = Index(size(U_aug_mat, 2), tags(old_mid))
    U_aug_tens = itensor(reshape(U_aug_mat, dl, ds, size(U_aug_mat, 2)), lψ_l, s_ℓ, aug_lnk)
    return U_aug_tens, n_added
end

function _augment_right_isometries(
    V_step_tens :: ITensor,
    V_old_tens  :: ITensor,
    step_mid    :: Index,
    old_mid     :: Index,
    s_ℓ1        :: Index,
    lψ_r        :: Index,
)
    ds1 = dim(s_ℓ1)
    dr1 = dim(lψ_r)
    V_step_mat = reshape(_complex_tensor_array(V_step_tens, step_mid, s_ℓ1, lψ_r), dim(step_mid), ds1 * dr1)
    V_old_mat = reshape(_complex_tensor_array(V_old_tens, old_mid, s_ℓ1, lψ_r), dim(old_mid), ds1 * dr1)
    V_old_perp = V_old_mat .- (V_old_mat * V_step_mat') * V_step_mat
    F = qr(V_old_perp')
    n_added = _qr_nonzero_diagonal_rank(V_old_perp)
    n_added <= 0 && return V_step_tens, 0

    ΔV = Matrix(F.Q[:, 1:n_added]')
    V_aug_mat = vcat(V_step_mat, ΔV)
    aug_lnk = Index(size(V_aug_mat, 1), tags(old_mid))
    V_aug_tens = itensor(reshape(V_aug_mat, size(V_aug_mat, 1), ds1, dr1), aug_lnk, s_ℓ1, lψ_r)
    return V_aug_tens, n_added
end

function _augmented_left_isometry_from_k(
    U0_tens       :: ITensor,
    K1_tens       :: ITensor,
    lψ_l          :: Index,
    s_ℓ           :: Index,
    old_mid       :: Index,
    evolved_right :: Index;
    aug_tol       :: Float64,
    augmentation_mode::Symbol = :projected,
    return_overlap::Bool = false,
)
    _ = augmentation_mode
    dl = dim(lψ_l)
    ds = dim(s_ℓ)
    U0_mat = reshape(_complex_tensor_array(U0_tens, lψ_l, s_ℓ, old_mid), dl * ds, dim(old_mid))
    K1_mat = reshape(_complex_tensor_array(K1_tens, lψ_l, s_ℓ, evolved_right), dl * ds, dim(evolved_right))
    A_aug_mat, n_new, overlap_mat = _pick_left_update(U0_mat, K1_mat;
        aug_tol = aug_tol,
    )

    aug_lnk = Index(size(A_aug_mat, 2), tags(old_mid))
    U1_tens = itensor(reshape(A_aug_mat, dl, ds, size(A_aug_mat, 2)), lψ_l, s_ℓ, aug_lnk)
    if return_overlap
        overlap_tens = itensor(overlap_mat, aug_lnk, old_mid)
        return U1_tens, n_new, overlap_tens
    end
    return U1_tens, n_new
end

function _augmented_right_isometry_from_l(
    V0_tens      :: ITensor,
    L1_tens      :: ITensor,
    old_mid      :: Index,
    evolved_left :: Index,
    s_ℓ1         :: Index,
    lψ_r         :: Index;
    aug_tol      :: Float64,
    augmentation_mode::Symbol = :projected,
    return_overlap::Bool = false,
)
    _ = augmentation_mode
    ds1 = dim(s_ℓ1)
    dr1 = dim(lψ_r)
    V0_mat = reshape(_complex_tensor_array(V0_tens, old_mid, s_ℓ1, lψ_r), dim(old_mid), ds1 * dr1)
    L1_mat = reshape(_complex_tensor_array(L1_tens, evolved_left, s_ℓ1, lψ_r), dim(evolved_left), ds1 * dr1)
    B_aug_mat, n_new, _ = _pick_right_update(V0_mat, L1_mat;
        aug_tol = aug_tol,
    )

    keep = size(B_aug_mat, 1)
    keep > 0 || error("Right-isometry QR produced zero rank")
    canon_lnk = Index(keep, tags(old_mid))
    V1_tens = itensor(reshape(B_aug_mat, keep, ds1, dr1), canon_lnk, s_ℓ1, lψ_r)
    if return_overlap
        # Transport the old right basis into the new augmented right basis.
        # The order matters: downstream K/S seeds require `V0 * dag(V1)`, not
        # `dag(V1) * V0`.
        overlap_tens = V0_tens * dag(V1_tens)
        return V1_tens, n_new, overlap_tens
    end
    return V1_tens, n_new
end

function _transported_l_seed_data(
    U1_tens::ITensor,
    U0_tens::ITensor,
    S0_tens::ITensor,
    V0_tens::ITensor,
    s_next::Index,
    link_r::Index,
)
    theta0_tens = U0_tens * S0_tens * V0_tens
    L0_tens = dag(U1_tens) * theta0_tens
    link_u1 = commonind(U1_tens, L0_tens)
    V0_seed_tens = _right_isometry_seed_from_l(L0_tens, link_u1, s_next, link_r)
    S_seed_tens = dag(U1_tens) * theta0_tens * dag(V0_seed_tens)
    return (; theta0_tens, L0_tens, link_u1, V0_seed_tens, S_seed_tens)
end

function _transported_k_seed_data(
    V1_tens::ITensor,
    U0_tens::ITensor,
    S0_tens::ITensor,
    V0_tens::ITensor,
    link_l::Index,
    site_l::Index,
)
    theta0_tens = U0_tens * S0_tens * V0_tens
    K0_tens = theta0_tens * dag(V1_tens)
    link_v1 = _left_site_bond_index(K0_tens, link_l, site_l)
    U0_seed_tens = _left_isometry_seed_from_k(K0_tens, link_l, site_l, link_v1)
    seed_mid = _left_site_bond_index(U0_seed_tens, link_l, site_l)
    S_seed_tens = dag(U0_seed_tens) * theta0_tens * dag(V1_tens)
    return (; theta0_tens, K0_tens, link_v1, seed_mid, U0_seed_tens, S_seed_tens)
end

function _transported_s_start_from_augmented_bases(
    U_aug_tens::ITensor,
    U0_tens::ITensor,
    S0_tens::ITensor,
    V0_tens::ITensor,
    V_aug_tens::ITensor,
)
    M_hat = dag(U_aug_tens) * U0_tens
    N_hat = V0_tens * dag(V_aug_tens)
    S_transport = M_hat * S0_tens * N_hat
    return (; M_hat, N_hat, S_transport)
end

"""
    _transported_s_start_from_augmented_bases(U_aug_tens, U0_tens, S0_tens, V0_tens, V_aug_tens)
    _transported_s_start_from_augmented_bases(S0_tens, M_hat, N_hat)

Project the old bond matrix `S0_tens` into the newly augmented left/right bases.
The returned overlap tensors `M_hat` and `N_hat` are useful when the caller also
needs to reuse the basis-change maps for downstream K/L bookkeeping.
"""
function _transported_s_start_from_augmented_bases(
    S0_tens::ITensor,
    M_hat::ITensor,
    N_hat::ITensor,
)
    S_transport = M_hat * S0_tens * N_hat
    return (; M_hat, N_hat, S_transport)
end


# ── Bond index and projected S-start helpers (shared by both sweeps) ────────
_projected_s_start(U1_tens::ITensor, theta0_tens::ITensor, V1_tens::ITensor) =
    dag(U1_tens) * theta0_tens * dag(V1_tens)

function _site_bond_index(V_tens::ITensor, s_next::Index, link_r::Index)
    out = nothing
    @inbounds for idx in inds(V_tens)
        if idx != s_next && idx != link_r
            isnothing(out) || error("Expected exactly one site bond index")
            out = idx
        end
    end
    isnothing(out) && error("Could not find site bond index")
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
    isnothing(out) && error("Could not find left-site bond index")
    return out
end

function _collapse_boundary_left_tensor(tens::ITensor, bond_data)
    if hasproperty(bond_data, :boundary_left_collapse)
        return bond_data.boundary_left_collapse * tens
    end
    return tens
end

function _collapse_boundary_right_tensor(tens::ITensor, bond_data)
    if hasproperty(bond_data, :boundary_right_collapse)
        return tens * bond_data.boundary_right_collapse
    end
    return tens
end

_boundary_left_writeback_index(bond_data) =
    hasproperty(bond_data, :physical_link_l) ? bond_data.physical_link_l : bond_data.link_l

_boundary_right_writeback_index(bond_data) =
    hasproperty(bond_data, :physical_link_r) ? bond_data.physical_link_r : bond_data.link_r
