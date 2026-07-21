# Deterministic TT compression and related helpers.

export svd_compress, svd_compress!, compress, compress!, multiply!, square!
export svd_compress_rsvd, svd_compress_rsvd!
export svd_compress_reverse!


"""
    svd_compress(¤ê::TensorTrain; maxdim, cutoff=0.0) -> TensorTrain

Return a compressed copy of `¤ê` by orthogonalizing left-to-right and truncating
each bond with a local SVD.
"""
function svd_compress(¤ê::TensorTrain; kwargs...)
    ¤ò = replacelinks(copy(¤ê))
    svd_compress!(¤ò; kwargs...)
    return ¤ò
end


"""
    svd_compress!(f::TensorTrain; maxdim, cutoff=0.0)

Compress `f` in-place with a single left-to-right truncated-SVD sweep.
"""
function svd_compress!(f::TensorTrain; maxdim::Integer, cutoff::AbstractFloat = 0.0)
    orthogonalize!(f, 1)
    for j in firstindex(f):(lastindex(f) - 1)
        Uinds = uniqueinds(f[j], f[j + 1])
        U, S, V = svd(f[j], Uinds; maxdim, cutoff)
        f[j] = U
        f[j + 1] = (S * V) * f[j + 1]
        old_link = commonind(f[j], f[j + 1])
        if !isnothing(old_link)
            new_link = settags(old_link, "Link,l=$j")
            replaceind!(f[j], old_link, new_link)
            replaceind!(f[j + 1], old_link, new_link)
        end
    end
end


"""
    svd_compress_reverse!(f::TensorTrain; maxdim, cutoff=0.0)

Compress `f` in-place with a single right-to-left truncated-SVD sweep — the
mirror image of `svd_compress!`. Running `svd_compress!` then
`svd_compress_reverse!` (or vice versa) gives a lossless TWO-WAY re-gauge: each
pass is SVD-based (not QR-only), so each direction can independently collapse
any rank-deficiency it finds, leaving every bond at its true minimal Schmidt
rank rather than only the bonds visited by a single direction.
"""
function svd_compress_reverse!(f::TensorTrain; maxdim::Integer, cutoff::AbstractFloat = 0.0)
    N = length(f)
    orthogonalize!(f, N)
    for j in lastindex(f):-1:(firstindex(f) + 1)
        Vinds = uniqueinds(f[j], f[j - 1])
        U, S, V = svd(f[j], Vinds; maxdim, cutoff)
        f[j] = U
        f[j - 1] = f[j - 1] * (S * V)
        old_link = commonind(f[j], f[j - 1])
        if !isnothing(old_link)
            new_link = settags(old_link, "Link,l=$(j - 1)")
            replaceind!(f[j], old_link, new_link)
            replaceind!(f[j - 1], old_link, new_link)
        end
    end
end


"""
    svd_compress_rsvd(¤ê::TensorTrain; maxdim, cutoff=0.0, oversampling=10) -> TensorTrain

Compressed copy of `¤ê` using an oversampled two-pass SVD sweep.
"""
function svd_compress_rsvd(¤ê::TensorTrain; kwargs...)
    ¤ò = replacelinks(copy(¤ê))
    svd_compress_rsvd!(¤ò; kwargs...)
    return ¤ò
end


"""
    svd_compress_rsvd!(f::TensorTrain; maxdim, cutoff=0.0, oversampling=10)

In-place two-pass oversampled compression of `f`.
"""
function svd_compress_rsvd!(
    f::TensorTrain;
    maxdim::Integer,
    cutoff::AbstractFloat = 0.0,
    oversampling::Integer = 10,
)
    k = maxdim + oversampling

    orthogonalize!(f, 1)
    for j in firstindex(f):(lastindex(f) - 1)
        Uinds = uniqueinds(f[j], f[j + 1])
        U, S, V = svd(f[j], Uinds; maxdim = k, cutoff = 0.0)
        f[j] = U
        f[j + 1] = (S * V) * f[j + 1]
        old_link = commonind(f[j], f[j + 1])
        if !isnothing(old_link)
            new_link = settags(old_link, "Link,l=$j")
            replaceind!(f[j], old_link, new_link)
            replaceind!(f[j + 1], old_link, new_link)
        end
    end

    orthogonalize!(f, 1)
    for j in firstindex(f):(lastindex(f) - 1)
        Uinds = uniqueinds(f[j], f[j + 1])
        U, S, V = svd(f[j], Uinds; maxdim = maxdim, cutoff = cutoff)
        f[j] = U
        f[j + 1] = (S * V) * f[j + 1]
        old_link = commonind(f[j], f[j + 1])
        if !isnothing(old_link)
            new_link = settags(old_link, "Link,l=$j")
            replaceind!(f[j], old_link, new_link)
            replaceind!(f[j + 1], old_link, new_link)
        end
    end
end


"""
    _compress_cutoff(kwargs)

Normalize the optional compression cutoff keyword to `Float64`.

Arguments
- `kwargs`: Input parameter used by the local tensor-network calculation.

Returns
- Returns the cutoff value used by deterministic TT compression routines.

Description
- Compression code benefits from a single scalar tolerance type so the truncation logic and comparisons remain type-stable across callers.
"""
_compress_cutoff(kwargs) = Float64(get(kwargs, :cutoff, 0.0))


"""
    compress(f::TensorTrain, bonddim::Integer; cutoff=0.0, kwargs...) -> TensorTrain

Deterministically compress `f` to bond dimension `bonddim` with a truncated
SVD sweep. Legacy sweep-control keyword arguments are accepted for compatibility
and ignored.
"""
function compress(f::TensorTrain, bonddim::Integer; kwargs...)
    return svd_compress(f; maxdim = bonddim, cutoff = _compress_cutoff(kwargs))
end


"""
    compress!(out::TensorTrain, f::TensorTrain; cutoff=0.0, maxdim=maxlinkdim(out), kwargs...) -> TensorTrain

Deterministically overwrite `out` with an SVD-compressed copy of `f`. When
`maxdim` is provided it overrides the buffer's current link dimension, which is
useful for reusable work buffers whose active rank may shrink between calls.
Legacy sweep-control keyword arguments are accepted for compatibility and
ignored.
"""
function compress!(out::TensorTrain, f::TensorTrain; kwargs...)
    length(out) != length(f) &&
        throw(ArgumentError("out and f have different lengths."))
    siteinds(out) == siteinds(f) ||
        throw(ArgumentError("out and f must share the same site indices."))

    target_maxdim = Int(get(kwargs, :maxdim, max(maxlinkdim(out), 1)))
    compressed = compress(f, max(target_maxdim, 1); kwargs...)
    copyto!(out, compressed)
    return out
end


"""
    multiply!(out::TensorTrain, A::TensorTrainOperator, b::TensorTrain; cutoff=0.0, maxdim=maxlinkdim(out), kwargs...)

Compute `out Ôëê A * b` by exact MPO┬ÀMPS contraction followed by deterministic
SVD compression. Passing `maxdim` is useful when `out` is a reusable work
buffer whose current rank should not determine the next compression target.
Legacy sweep-control keyword arguments are accepted for compatibility and
ignored.
"""
function multiply!(out::TensorTrain, A::TensorTrainOperator, b::TensorTrain; kwargs...)
    Ab = contract(A, b)
    compress!(out, Ab; kwargs...)
    return out
end


"""
    square!(out::TensorTrain, u::TensorTrain; cutoff=0.0, product_maxdim=max(4¤ç,16), kwargs...)

Compute the elementwise square `out Ôëê u ÔèÖ u` using `zipup` plus deterministic
SVD compression. Legacy sweep-control keyword arguments are accepted for
compatibility and ignored.
"""
function square!(out::TensorTrain, u::TensorTrain; kwargs...)
    product_maxdim = Int(get(kwargs, :product_maxdim, max(maxlinkdim(out) * 4, 16)))
    sq = zipup(u, u; maxdim = product_maxdim, cutoff = _compress_cutoff(kwargs))
    compress!(out, sq; kwargs...)
    return out
end
