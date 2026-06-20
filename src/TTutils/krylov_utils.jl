# krylov_utils.jl
#
# Krylov/expv helpers (native Lanczos + KrylovKit dispatch) shared by BUG and TDVP.
# Also contains low-level index and array helpers used across integrators.
#
# Contents drawn from src/BUGShared.jl (the shared integrator utilities).

# ── Krylov / exponential-vector helpers ──────────────────────────────────────

"""
    _hermitian_tridiagonal_exp_coeffs(alpha, beta, dt) -> Vector{ComplexF64}

Compute the Krylov coefficients for exp(dt * T) * e1 where T is the
tridiagonal Lanczos matrix with diagonal `alpha` and off-diagonal `beta`.
"""
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

"""
    _native_hermitian_lanczos_exponentiate(matvec, dt, x; tol, krylovdim) -> (y, numops)

Apply exp(dt * H) to the vector `x` using a native Hermitian Lanczos iteration.
Assumes the operator represented by `matvec` is Hermitian.
"""
function _native_hermitian_lanczos_exponentiate(
    matvec,
    dt,
    x::AbstractVector;
    tol::Float64,
    krylovdim::Int,
)
    n = length(x)
    n == 0 && return similar(x, ComplexF64, 0), 0

    x_work  = Vector{ComplexF64}(undef, n)
    copyto!(x_work, x)
    norm_x  = norm(x_work)
    norm_x == 0 && return fill!(similar(x_work), 0.0 + 0.0im), 0

    mmax  = min(max(krylovdim, 1), n)
    basis = Matrix{ComplexF64}(undef, n, mmax)
    alpha = Vector{Float64}(undef, mmax)
    beta  = Vector{Float64}(undef, max(mmax - 1, 0))
    work  = Vector{ComplexF64}(undef, n)
    basis[:, 1] .= x_work ./ norm_x
    numops    = 0
    final_dim = 1

    for j in 1:mmax
        vj = @view basis[:, j]
        copyto!(work, matvec(vj))
        numops += 1
        if j > 1
            LinearAlgebra.axpy!(-beta[j-1], @view(basis[:, j-1]), work)
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
        beta[j]          = beta_j
        basis[:, j+1]   .= work ./ beta_j
        final_dim        = j + 1
    end

    coeff = _hermitian_tridiagonal_exp_coeffs(
        alpha[1:final_dim], beta[1:max(final_dim-1, 0)], dt,
    )
    y = norm_x .* (basis[:, 1:final_dim] * coeff)
    return y, numops
end

# Global expv backend selector (mirrors BUGShared for backward compat)
const BUG_DEFAULT_EXPV_BACKEND      = :krylovkit
const BUG_ALLOWED_EXPV_BACKENDS     = (:krylovkit, :native_hermitian_lanczos)
const BUG_ACTIVE_EXPV_BACKEND       = Ref{Symbol}(BUG_DEFAULT_EXPV_BACKEND)

"""
    _with_bug_expv_backend(backend, f)

Temporarily set the global expv backend and execute `f()`.
"""
function _with_bug_expv_backend(backend::Symbol, f::Function)
    backend in BUG_ALLOWED_EXPV_BACKENDS ||
        error("Unknown expv backend: $backend")
    prev = BUG_ACTIVE_EXPV_BACKEND[]
    BUG_ACTIVE_EXPV_BACKEND[] = backend
    try
        return f()
    finally
        BUG_ACTIVE_EXPV_BACKEND[] = prev
    end
end

_with_bug_expv_backend(f::Function, backend::Symbol) = _with_bug_expv_backend(backend, f)
_active_bug_expv_backend() = BUG_ACTIVE_EXPV_BACKEND[]

# Global time evolution prefactor selector for classical PDEs vs quantum evolution
const BUG_DEFAULT_TIME_PREFACTOR    = ComplexF64(-im)
const BUG_ACTIVE_TIME_PREFACTOR     = Ref{ComplexF64}(BUG_DEFAULT_TIME_PREFACTOR)

"""
    _with_bug_time_prefactor(c, f)

Temporarily switch the BUG time evolution prefactor while executing `f()`.
For classical real-time PDEs (heat, Burgers), set `c = ComplexF64(1)`.
For quantum evolution (Schrödinger), set `c = -im` (default).
"""
function _with_bug_time_prefactor(c::ComplexF64, f::Function)
    previous = BUG_ACTIVE_TIME_PREFACTOR[]
    BUG_ACTIVE_TIME_PREFACTOR[] = c
    try
        return f()
    finally
        BUG_ACTIVE_TIME_PREFACTOR[] = previous
    end
end

_with_bug_time_prefactor(f::Function, c::ComplexF64) = _with_bug_time_prefactor(c, f)
_active_time_prefactor() = BUG_ACTIVE_TIME_PREFACTOR[]

"""
    _linear_substep(H_or_matvec, dt, x; method, lanczos_tol, lanczos_maxiter, restart=1)

Advance x' = H*x for one micro-step. Returns (x_next, numops).
Assumes H is Hermitian for method=:expv.
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
    if method === :expv
        H_dense = Matrix{ComplexF64}(H)
        x_dense = Vector{ComplexF64}(x)
        return exp(dt * H_dense) * x_dense, 0
    end
    return _linear_substep(v -> H * v, dt, x;
        method = method, lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter, restart = restart,
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
                    matvec, dt, x; tol = lanczos_tol, krylovdim = lanczos_maxiter,
                )
            catch err
                err isa LinearAlgebra.LAPACKException || rethrow()
                # Fallthrough to KrylovKit on LAPACK failure
            end
        end
        y, info = KrylovKit.exponentiate(matvec, dt, x;
            krylovdim = lanczos_maxiter, tol = lanczos_tol,
            maxiter = restart, issymmetric = true, eager = false,
        )
        return y, info.numops
    elseif method === :euler
        return x + dt * matvec(x), 1
    elseif method === :rk4
        k1 = matvec(x)
        k2 = matvec(x + (dt / 2) * k1)
        k3 = matvec(x + (dt / 2) * k2)
        k4 = matvec(x + dt * k3)
        return x + (dt / 6) * (k1 + 2k2 + 2k3 + k4), 4
    else
        error("Unknown substep method: $method. Supported: :expv, :euler, :rk4.")
    end
end

"""
    _general_linear_substep(matvec, dt, x; method, ..., issymmetric)

Like `_linear_substep` but with explicit symmetry flag for KrylovKit dispatch.
"""
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
                matvec, dt, x;
                tol = lanczos_tol, krylovdim = min(length(x), max(lanczos_maxiter, 4)),
            )
        end
        y, info = KrylovKit.exponentiate(matvec, dt, x;
            krylovdim = min(length(x), max(lanczos_maxiter, 4)),
            tol = lanczos_tol, maxiter = restart, issymmetric = issymmetric, eager = false,
        )
        return y, info.numops
    elseif method === :euler
        return x + dt * matvec(x), 1
    elseif method === :rk4
        k1 = matvec(x)
        k2 = matvec(x + (dt / 2) * k1)
        k3 = matvec(x + (dt / 2) * k2)
        k4 = matvec(x + dt * k3)
        return x + (dt / 6) * (k1 + 2k2 + 2k3 + k4), 4
    end
    error("Unknown substep method: $method.")
end

# ── TensorTrain index and array helpers ───────────────────────────────────────

"""
    _tensortrain_boundary_link(psi, side) -> Index

Return the single boundary link index on the left or right end of `psi`.
"""
function _tensortrain_boundary_link(psi::TensorTrain, side::Symbol)
    N = length(psi)
    N < 2 && error("requires at least two sites")
    side === :left  && return only(uniqueinds(psi[1], psi[2]; tags="Link"))
    side === :right && return only(uniqueinds(psi[N], psi[N-1]; tags="Link"))
    error("side must be :left or :right")
end

"""
    _tensortrain_site_index(psi, k) -> Index

Return the physical site index at position `k`.
"""
function _tensortrain_site_index(psi::TensorTrain, k::Int)
    N = length(psi)
    N < 2 && error("requires at least two sites")
    if k == 1
        link_l = _tensortrain_boundary_link(psi, :left)
        link_r = commonind(psi[1], psi[2])
    elseif k == N
        link_l = commonind(psi[N-1], psi[N])
        link_r = _tensortrain_boundary_link(psi, :right)
    else
        link_l = commonind(psi[k-1], psi[k])
        link_r = commonind(psi[k], psi[k+1])
    end
    return only(uniqueinds(psi[k], [link_l, link_r]))
end

"""
    _tensortrain_bond_indices(psi, bond) -> NamedTuple

Extract the five bond indices (link_l, link_mid, link_r, site_l, site_r) for
the two-site window at `bond`.
"""
function _tensortrain_bond_indices(psi::TensorTrain, bond::Int)
    left_tens  = psi[bond]
    right_tens = psi[bond + 1]
    link_mid   = commonind(left_tens, right_tens)
    link_l     = only(uniqueinds(left_tens,  right_tens; tags="Link"))
    link_r     = only(uniqueinds(right_tens, left_tens;  tags="Link"))
    site_l     = only(uniqueinds(left_tens,  [link_l, link_mid]))
    site_r     = only(uniqueinds(right_tens, [link_mid, link_r]))
    return (; link_l, link_mid, link_r, site_l, site_r)
end

"""
    _complex_tensor_array(T, inds...) -> Array{ComplexF64}

Convert an ITensor to a dense ComplexF64 array in the given index order.
"""
@inline function _complex_tensor_array(T::ITensor, inds...)
    A = Array(T, inds...)
    return eltype(A) <: ComplexF64 ? A : ComplexF64.(A)
end

"""
    _complex_tensor_vec(T, inds...) -> Vector{ComplexF64}

Convert an ITensor to a dense ComplexF64 column vector in the given index order.
"""
@inline function _complex_tensor_vec(T::ITensor, inds...)
    A = Array(T, inds...)
    return eltype(A) <: ComplexF64 ? vec(A) : ComplexF64.(vec(A))
end

# ── QR-rank helpers ───────────────────────────────────────────────────────────

"""
    _qr_nonzero_diagonal_rank(A) -> Int

Estimate the numerical rank of A via the number of non-negligible diagonal
entries of its QR R-factor.
"""
function _qr_nonzero_diagonal_rank(A::AbstractMatrix)
    isempty(A) && return 0
    F = qr(A)
    ndiag = min(size(A)...)
    ndiag <= 0 && return 0
    diagvals = Vector{Float64}(undef, ndiag)
    scale    = zero(Float64)
    @inbounds for i in 1:ndiag
        val = abs(F.R[i, i])
        diagvals[i] = val
        scale = max(scale, val)
    end
    tol = max(size(A)...) * eps(real(eltype(A))) * max(scale, 1.0)
    return count(>(tol), diagvals)
end

"""
    _qr_column_basis(A) -> (Q, rank)

Return a basis for the column space of A via QR, plus the numerical rank.
"""
function _qr_column_basis(A::AbstractMatrix)
    isempty(A) && return Matrix{eltype(A)}(undef, size(A, 1), 0), 0
    F     = qr(A)
    ndiag = min(size(A)...)
    diagvals = Vector{Float64}(undef, ndiag)
    scale    = zero(Float64)
    @inbounds for i in 1:ndiag
        val = abs(F.R[i, i])
        diagvals[i] = val
        scale = max(scale, val)
    end
    tol  = max(size(A)...) * eps(real(eltype(A))) * max(scale, 1.0)
    rank = count(>(tol), diagvals)
    rank <= 0 && return Matrix{eltype(A)}(undef, size(A, 1), 0), 0
    return Matrix(F.Q[:, 1:rank]), rank
end

"""
    _qr_row_basis(A) -> (Q, rank)

Return a basis for the row space of A via QR of A†, plus the numerical rank.
"""
function _qr_row_basis(A::AbstractMatrix)
    isempty(A) && return Matrix{eltype(A)}(undef, 0, size(A, 2)), 0
    F     = qr(adjoint(A))
    ndiag = min(size(A)...)
    diagvals = Vector{Float64}(undef, ndiag)
    scale    = zero(Float64)
    @inbounds for i in 1:ndiag
        val = abs(F.R[i, i])
        diagvals[i] = val
        scale = max(scale, val)
    end
    tol  = max(size(A)...) * eps(real(eltype(A))) * max(scale, 1.0)
    rank = count(>(tol), diagvals)
    rank <= 0 && return Matrix{eltype(A)}(undef, 0, size(A, 2)), 0
    return Matrix(adjoint(F.Q[:, 1:rank])), rank
end

"""
    _identity_overlap_matrix(T, n) -> Matrix{T}

Return a n×n identity matrix of element type T.
"""
function _identity_overlap_matrix(T_el::Type, n::Int)
    out = zeros(T_el, n, n)
    @inbounds for i in 1:n
        out[i, i] = one(T_el)
    end
    return out
end

"""
    _complete_column_basis(Q, target_cols) -> Matrix

Complete a partial orthonormal basis Q to have `target_cols` orthonormal columns
by appending standard-basis vectors orthogonalized against existing columns.
"""
function _complete_column_basis(Q::AbstractMatrix, target_cols::Int)
    current = size(Q, 2)
    current >= target_cols && return Matrix(Q[:, 1:target_cols])

    ambient = size(Q, 1)
    T_el    = eltype(Q)
    out     = Matrix{T_el}(undef, ambient, target_cols)
    if current > 0
        @views out[:, 1:current] .= Q
    end

    filled = current
    tol    = max(ambient, target_cols) * eps(Float64)
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
        error("Could not complete column basis from rank $current to rank $target_cols")
    return out
end

"""
    _complete_row_basis(B, target_rows) -> Matrix

Complete a partial orthonormal basis B to have `target_rows` orthonormal rows.
"""
function _complete_row_basis(B::AbstractMatrix, target_rows::Int)
    current = size(B, 1)
    current >= target_rows && return Matrix(B[1:target_rows, :])
    return Matrix(adjoint(_complete_column_basis(adjoint(B), target_rows)))
end
