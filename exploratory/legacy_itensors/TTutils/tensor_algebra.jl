# tensor_algebra.jl
#
# Low-level ITensor QR/LQ/SVD overloads and random-unitary/truncation helpers.
# These are the building blocks for all gauge moves in TensorTrain and
# TensorTrainOperator algorithms.
#
# Exported by TTutils: qr, lq, svd, random_unitary, truncate

# ── QR / LQ ──────────────────────────────────────────────────────────────────

"""
    qr(A::ITensor, Qinds::Index...; tags="Link,qr", positive=false) -> Q, R

Thin QR decomposition of an ITensor. `Qinds` are the indices absorbed into Q;
the remainder go into R. `positive=true` fixes the diagonal of R to be positive.
"""
function LinearAlgebra.qr(A::ITensor, Qinds::Index...; tags::AbstractString="Link,qr", positive::Bool=false)
    Qinds_c = commoninds(A, collect(Qinds))
    Rinds   = uniqueinds(A, Qinds_c)
    leftdim  = prod(dim.(Qinds_c))
    rightdim = prod(dim.(Rinds))

    Amat = reshape(Array(A, Qinds_c..., Rinds...), leftdim, rightdim)
    q, r = _qr_dense(Amat, leftdim, rightdim; positive=positive)
    m = size(q, 2)

    newlink = Index(m, tags)
    Q = itensor(reshape(q, dim.(Qinds_c)..., m), Qinds_c..., newlink)
    R = itensor(reshape(r, m, dim.(Rinds)...), newlink, Rinds...)
    return Q, R
end

"""
    lq(A::ITensor, Qinds::Index...) -> L, Q

Thin LQ decomposition of an ITensor. `Qinds` become the Q part.
"""
function LinearAlgebra.lq(A::ITensor, Qinds::Index...)
    Qinds_c = commoninds(A, collect(Qinds))
    Linds   = uniqueinds(A, Qinds_c)
    leftdim  = prod(dim.(Linds))
    rightdim = prod(dim.(Qinds_c))

    Amat = reshape(Array(A, Linds..., Qinds_c...), leftdim, rightdim)
    l, q = _lq_dense(Amat, leftdim, rightdim)
    m = size(q, 1)

    newlink = Index(m, "Link,lq")
    L = itensor(reshape(l, dim.(Linds)..., m), Linds..., newlink)
    Q = itensor(reshape(q, m, dim.(Qinds_c)...), newlink, Qinds_c...)
    return L, Q
end

"""
    qrq!(A::ITensor, Qinds::Index...)

In-place replacement of `A` with its Q factor from QR (left-isometric).
"""
function qrq!(A::ITensor, Qinds::Index...)
    Qinds_c = commoninds(A, collect(Qinds))
    Rinds   = uniqueinds(A, Qinds_c)
    leftdim  = prod(dim.(Qinds_c))
    rightdim = prod(dim.(Rinds))

    Amat = reshape(Array(A, Qinds_c..., Rinds...), leftdim, rightdim)
    F = qr(Amat)
    m = min(leftdim, rightdim)
    Q = Matrix(F.Q)[:, 1:m]
    A .= itensor(reshape(Q, dim.(Qinds_c)..., dim.(Rinds)...), Qinds_c..., Rinds...)
end

"""
    lqq!(A::ITensor, Qinds::Index...)

In-place replacement of `A` with its Q factor from LQ (right-isometric).
"""
function lqq!(A::ITensor, Qinds::Index...)
    Qinds_c = commoninds(A, collect(Qinds))
    Linds   = uniqueinds(A, Qinds_c)
    leftdim  = prod(dim.(Linds))
    rightdim = prod(dim.(Qinds_c))

    Amat = reshape(Array(A, Linds..., Qinds_c...), leftdim, rightdim)
    F = lq(Amat)
    m = min(leftdim, rightdim)
    Q = Matrix(F.Q)[1:m, :]
    A .= itensor(reshape(Q, dim.(Linds)..., dim.(Qinds_c)...), Linds..., Qinds_c...)
end

# ── SVD ──────────────────────────────────────────────────────────────────────

"""
    svd(A::ITensor, Uinds; maxdim, cutoff, kwargs...) -> U, S, V, spec, u, v

Truncated SVD of an ITensor. `Uinds` go into U; the rest go into V.
Returns the 6-tuple `(U, S, V, spec, u, v)` for compatibility with ITensorMPS.
"""
function ITensors.svd(
    A::ITensor,
    Uinds;
    maxdim::Integer = typemax(Int),
    cutoff::AbstractFloat = 0.0,
    kwargs...,
)
    Uinds_c = commoninds(A, Uinds)
    Vinds   = uniqueinds(A, Uinds_c)
    leftdim  = prod(dim.(Uinds_c))
    rightdim = prod(dim.(Vinds))

    Amat = reshape(Array(A, Uinds_c..., Vinds...), leftdim, rightdim)
    u, s, vt, truncerr = _svd_truncated(Amat, leftdim, rightdim, maxdim, cutoff)
    m = length(s)

    lefttags  = get(kwargs, :lefttags, nothing)
    link_tag  = isnothing(lefttags) ? "Link,SVD" : string(lefttags)
    newlink   = Index(m, link_tag)

    U    = itensor(reshape(u,  dim.(Uinds_c)..., m), Uinds_c..., newlink)
    S    = diag_itensor(s, newlink, newlink')
    V    = itensor(reshape(vt, m, dim.(Vinds)...), newlink', Vinds...)
    spec = Spectrum(Float64.(s .^ 2), truncerr)
    return U, S, V, spec, newlink, newlink'
end

# ── Random unitary ────────────────────────────────────────────────────────────

"""
    random_unitary(T, m, n) -> Matrix{T}

Return an m×n matrix with orthonormal columns drawn from the Haar measure.
"""
function random_unitary(::Type{T}, m::Integer, n::Integer) where {T<:Number}
    A = randn(T, max(m, n), max(m, n))
    F = qr(A)
    return Matrix(F.Q)[1:m, 1:n]
end

random_unitary(m::Integer, n::Integer) = random_unitary(Float64, m, n)
random_unitary(n::Integer)             = random_unitary(Float64, n, n)

# ── Singular value truncation ─────────────────────────────────────────────────

"""
    truncate(s, maxdim, cutoff) -> s_trunc

Truncate a vector of singular values by relative cutoff and maximum count.
"""
function truncate(s::AbstractVector, maxdim::Integer, cutoff::AbstractFloat)
    n = length(s)
    n == 0 && return s
    threshold = cutoff * s[1]
    m = something(findfirst(x -> x < threshold, s), n + 1) - 1
    m = clamp(m, 1, min(maxdim, n))
    return s[1:m]
end

# ── Dense array backends ──────────────────────────────────────────────────────

function _qr_dense(A::AbstractMatrix{T}, leftdim::Integer, rightdim::Integer; positive::Bool=false) where {T}
    F = qr(reshape(A, leftdim, rightdim))
    q = Matrix(F.Q)
    r = Matrix(F.R)
    if positive
        signs = sign.(diag(r))
        signs[signs .== 0] .= 1
        q .*= transpose(signs)
        r .*= signs
    end
    return q, r
end

function _lq_dense(A::AbstractMatrix{T}, leftdim::Integer, rightdim::Integer) where {T}
    F = lq(reshape(A, leftdim, rightdim))
    return Matrix(F.L), Matrix(F.Q)
end

function _svd_truncated(A::AbstractMatrix, leftdim::Integer, rightdim::Integer,
                        maxdim::Integer, cutoff::AbstractFloat)
    Amat = reshape(A, leftdim, rightdim)
    F    = svd(Amat; full=false)
    s    = F.S
    s_trunc = truncate(s, maxdim, cutoff)
    m = length(s_trunc)
    truncerr = m < length(s) ? norm(s[(m + 1):end]) : 0.0
    return F.U[:, 1:m], s_trunc, F.Vt[1:m, :], truncerr
end
