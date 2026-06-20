# tensortrain.jl
#
# Defines the TensorTrain (MPS) struct and all associated constructors.
export TensorTrain
export vector, random_tt, empty_tt, uniform_tt
export orthogonalize, orthogonalize!, filter_singular_values!
export linkdims, coarse, bitreverse


# ÔöÇÔöÇ Struct ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

"""
    TensorTrain <: AbstractTensorTrain

Matrix Product State (MPS) over a set of physical sites.

# Fields
- `data::Vector{ITensor}` ÔÇö the core tensors.
"""
function _ensure_tensortrain_boundary_links(tensors::Vector{ITensor})
    N = length(tensors)
    N == 0 && throw(ArgumentError("TensorTrain requires at least one tensor."))

    data = copy(tensors)
    left_links = filter(hastags("Link"), inds(data[1]))
    if length(left_links) < 2
        data[1] = data[1] * itensor([1.0], Index(1, "Link,l=0"))
    end

    right_links = filter(hastags("Link"), inds(data[end]))
    if length(right_links) < 2
        data[end] = data[end] * itensor([1.0], Index(1, "Link,l=$N"))
    end
    return data
end

struct TensorTrain <: AbstractTensorTrain
    data::Vector{ITensor}

    function TensorTrain(data::Vector{ITensor})
        return new(_ensure_tensortrain_boundary_links(data))
    end
end

TensorTrain(mps::MPS) = TensorTrain(collect(mps))

"""
    TensorTrain(sites::Vector{<:Index}, tensors::Vector{Array{T,3}}) -> TensorTrain

Construct a TensorTrain from a vector of rank-3 arrays and a vector of site
indices.  Auxiliary link indices are created automatically.
"""
function TensorTrain(sites::Vector{<:Index}, tensors::Vector{<:Array{T,3}}) where {T}
    N = length(sites)
    length(tensors) != N && throw(ArgumentError("Length of `sites` and `tensors` must match."))

    dims_l = [size(tensors[k], 1) for k in 1:N]
    dims_r = [size(tensors[k], 3) for k in 1:N]

    links = Vector{Index}(undef, N + 1)
    links[1] = Index(dims_l[1], "Link,l=0")
    for k in 1:N
        links[k+1] = Index(dims_r[k], "Link,l=$k")
    end

    data = [itensor(tensors[k], links[k], sites[k], links[k+1]) for k in 1:N]
    return TensorTrain(data)
end


"""
    TensorTrain(fx::AbstractVector, sites::Vector{<:Index}; kwargs...) -> TensorTrain

Construct a left-canonical TensorTrain from a grid-value vector `fx` via
sequential SVD.  `kwargs` are forwarded to the truncated SVD (e.g. `maxdim`,
`cutoff`).
"""
function TensorTrain(fx::AbstractVector, sites::Vector{<:Index}; kwargs...)
    N     = length(sites)
    d     = dim(sites[1])
    Npts  = d^N
    length(fx) != Npts && throw(ArgumentError("Expected $Npts values, got $(length(fx))."))

    T    = eltype(fx)
    data = Vector{ITensor}(undef, N)

    _maxdim = get(kwargs, :maxdim, typemax(Int))
    _cutoff = Float64(get(kwargs, :cutoff, 0.0))

    links = [Index(1, "Link,l=0")]
    # Start from the dense vector and peel off one site at a time with SVD.
    A     = reshape(Vector{T}(fx), 1, Npts)  # (1, d^N) matrix

    for k in 1:N-1
        lk = links[end]
        m  = dim(lk)
        A  = reshape(A, m * d, d^(N - k))
        F  = LinearAlgebra.svd(A; alg=LinearAlgebra.DivideAndConquer())
        sv = truncate(F.S, _maxdim, _cutoff) # remove singular values 
        r  = length(sv) # rank of matrix after truncation
        U  = F.U[:, 1:r]
        Vt = F.Vt[1:r, :]

        lk1 = Index(r, "Link,l=$k")
        push!(links, lk1)

        data[k] = itensor(reshape(U, m, d, r), lk, sites[k], lk1) # fold back into tensor
        A        = Diagonal(sv) * Vt
    end

    lN  = links[end]
    lN1 = Index(1, "Link,l=$N")
    push!(links, lN1)
    data[N] = itensor(reshape(Matrix(A), dim(lN), d, 1), lN, sites[N], lN1)

    return TensorTrain(data)
end

"""
    siteinds(f::TensorTrain) -> Vector{Index}

Return the physical (site) indices of the tensor train.
"""
function siteinds(f::TensorTrain)
    N  = length(f)
    return [only([i for i in inds(f[k]) if !hastags(i, "Link")]) for k in 1:N]
end

"""
    siteinds(f::TensorTrain, j::Integer) -> Index

Return the physical site index at position `j`.
"""
siteinds(f::TensorTrain, j::Integer) = siteinds(f)[j]

"""
    siteind(f::TensorTrain, j::Integer) -> Index

Alias for `siteinds(f, j)`.
"""
siteind(f::TensorTrain, j::Integer)  = siteinds(f)[j]

"""
    vector(f::TensorTrain) -> Vector

Contract all cores and return the result as a plain Julia vector in the
correct spatial ordering (after bit-reversal).
"""
function vector(f::TensorTrain)
    T  = promote_itensor_eltype(f)
    ls = linkinds(f) # bond indices 
    v  = f[1]
    for k in 2:length(f)
        v = v * f[k]
    end
    # remove boundary terms 
    v = v * itensor(ones(dim(ls[1])),   ls[1])
    v = v * itensor(ones(dim(ls[end])), ls[end])
    sites = siteinds(f)
    arr   = Array(v, sites...)
    return vec(arr)
end


"""
    random_tt([T,] sites, bonddim; gauge_centre=length(sites)) -> TensorTrain

Construct a TensorTrain with random orthonormal cores and a random tensor at
`gauge_centre`.
"""
function random_tt(::Type{T}, sites::Vector{<:Index}, bonddim::Integer;
                   gauge_centre::Integer=length(sites)) where {T<:Number}
    N     = length(sites)
    sdims = dim.(sites)
    ldims = linkdims(sdims, bonddim)

    arrays = Vector{Array{T,3}}(undef, N)
    for k in 1:N
        if k < gauge_centre
            arrays[k] = reshape(
                random_unitary(T, ldims[k] * sdims[k], ldims[k+1]),
                ldims[k], sdims[k], ldims[k+1]
            )
        elseif k == gauge_centre
            arrays[k] = randn(T, ldims[k], sdims[k], ldims[k+1]) # not orthogonalized
        else
            arrays[k] = reshape(
                random_unitary(T, ldims[k+1], ldims[k] * sdims[k])',
                ldims[k], sdims[k], ldims[k+1]
            )
        end
    end
    return TensorTrain(sites, arrays)
end

"""
    random_tt(sites, bonddim, kwargs)

Construct a random tensor-train state with the requested site spaces and bond dimension.

Arguments
- `sites`: Physical site indices defining the tensor-product grid or lattice.
- `bonddim`: Requested uniform bond dimension for the constructed tensor train or MPO.
- `kwargs`: Input parameter used by the local tensor-network calculation.

Returns
- Returns a randomly initialized tensor-train state.

Description
- Random TT states are useful as generic initial conditions and stress tests because they sample the low-rank manifold without privileging a particular canonical basis.
"""
random_tt(sites::Vector{<:Index}, bonddim::Integer; kwargs...) =
    random_tt(Float64, sites, bonddim; kwargs...)


"""
    empty_tt([T,] sites, bonddim) -> TensorTrain

Construct a TensorTrain filled with zeros.
"""
function empty_tt(::Type{T}, sites::Vector{<:Index}, bonddim::Integer) where {T<:Number}
    N     = length(sites)
    sdims = dim.(sites)
    ldims = linkdims(sdims, bonddim)
    arrays = [zeros(T, ldims[k], sdims[k], ldims[k+1]) for k in 1:N]
    return TensorTrain(sites, arrays)
end

"""
    empty_tt(sites, bonddim)

Construct an all-zero tensor-train state with the requested site spaces and bond dimension.

Arguments
- `sites`: Physical site indices defining the tensor-product grid or lattice.
- `bonddim`: Requested uniform bond dimension for the constructed tensor train or MPO.

Returns
- Returns a zero-filled tensor-train state.

Description
- Explicit zero work buffers are convenient in iterative low-rank solvers where output storage is reused but the tensor-network structure must stay compatible with later contractions.
"""
empty_tt(sites::Vector{<:Index}, bonddim::Integer) = empty_tt(Float64, sites, bonddim)


"""
    uniform_tt([T,] sites) -> TensorTrain

Construct a rank-1 TensorTrain whose cores are all ones, representing a
uniform state on the tensor-product grid.
"""
function uniform_tt(::Type{T}, sites::Vector{<:Index}) where {T<:Number}
    N     = length(sites)
    sdims = dim.(sites)
    arrays = [ones(T, 1, sdims[k], 1) for k in 1:N]
    return TensorTrain(sites, arrays)
end

"""
    uniform_tt(sites)

Construct a tensor-train state whose entries are uniform across the represented grid.

Arguments
- `sites`: Physical site indices defining the tensor-product grid or lattice.

Returns
- Returns the tensor-train representation of a spatially uniform state.

Description
- Uniform states are analytically simple fixed directions on the TT manifold and are frequently used as baselines, mean modes, or normalization references.
"""
uniform_tt(sites::Vector{<:Index}) = uniform_tt(Float64, sites)

"""
    orthogonalize(f::TensorTrain, gc::Integer) -> TensorTrain

Return a copy of `f` with gauge centre at site `gc`.
"""
function orthogonalize(f::TensorTrain, gc::Integer)
    g = deepcopy(f)
    orthogonalize!(g, gc)
    return g
end

"""
    orthogonalize!(f::TensorTrain; gc=length(f))

Move the gauge centre of `f` to the rightmost site (default) in-place.
"""
orthogonalize!(f::TensorTrain) = orthogonalize!(f, length(f))

"""
    orthogonalize!(f::TensorTrain, gc::Integer; leftlim=0, rightlim=length(f)+1)

Move the gauge centre of `f` to `gc` in-place using QR (left-moving) and
LQ (right-moving) sweeps.
"""
function orthogonalize!(f::TensorTrain, gc::Integer;
                        leftlim::Integer=0, rightlim::Integer=length(f)+1)
    N = length(f)
    # left-to-right QR sweep: sites leftlim+1 ÔÇª gc-1
    for k in (leftlim+1):(gc-1)
        Uinds = uniqueinds(f[k], f[k+1])
        Q, R  = qr(f[k], Uinds...)
        ts1 = tags(commonind(Q, R))
        ts2 = tags(linkind(f, k))
        replacetags!(Q, ts1, ts2)
        replacetags!(R, ts1, ts2)
        f[k]   = Q
        f[k+1] = R * f[k+1]
    end
    # right-to-left LQ sweep: sites rightlim-1 ÔÇª gc+1.
    # This mirrors the QR pass and moves the orthogonality centre back left.
    for k in (rightlim-1):-1:(gc+1)
        Uinds = uniqueinds(f[k], f[k-1])
        L, Q  = lq(f[k], Uinds...)
        ts1 = tags(commonind(L, Q))
        ts2 = tags(linkind(f, k-1))
        replacetags!(L, ts1, ts2)
        replacetags!(Q, ts1, ts2)
        f[k]   = Q
        f[k-1] = f[k-1] * L
    end
    return f
end


"""
    filter_singular_values!(fn, f::TensorTrain)

Apply function `fn` to the singular values at each bond and update the TT
in-place.
"""
function filter_singular_values!(fn, f::TensorTrain)
    N = length(f)
    orthogonalize!(f, 1)
    for k in 1:N-1
        Uinds = uniqueinds(f[k], f[k+1])
        U, S, V = svd(f[k], Uinds...)
        s = diag(array(S))
        s_new = fn(s)
        S_new = diag_itensor(s_new, inds(S)...)
        f[k]   = U
        f[k+1] = (S_new * V) * f[k+1]
    end
    return f
end


# ÔöÇÔöÇ Link dimension helpers ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

"""
    linkdims(N, d, bonddim) -> Vector{Int}

Return the N+1 auxiliary (link) dimensions for a TensorTrain with `N` sites
of physical dimension `d` and maximum bond dimension `bonddim`.
"""
function linkdims(N::Integer, d::Integer, bonddim::Integer)
    dims = ones(Int, N + 1)
    for k in 1:N-1
        dims[k+1] = min(d^k, bonddim, d^(N-k))
    end
    return dims
end

"""
    linkdims(sitedims::Vector{Int}, bonddim::Integer) -> Vector{Int}

Link dimensions for a TensorTrain with heterogeneous site dimensions.
"""
function linkdims(sitedims::Vector{Int}, bonddim::Integer)
    N = length(sitedims)
    dims = ones(Int, N + 1)
    left_prod = [prod(sitedims[1:k]) for k in 0:N]
    right_prod = [prod(sitedims[k:N]) for k in 1:N+1]
    for k in 1:N-1
        dims[k+1] = min(left_prod[k+1], bonddim, right_prod[k+1])
    end
    return dims
end


"""
    coarse(f::TensorTrain, bits...) -> TensorTrain

Average over the tensor positions listed in `bits`, reducing the number of
sites by `length(bits)`.
"""
function coarse(f::TensorTrain, bits::Integer...)
    g = deepcopy(f)
    for b in sort(collect(bits), rev=true)
        d  = dim(siteind(g, b))
        sc = ITensor([1/d for _ in 1:d], siteind(g, b))
        g[b] = g[b] * sc
        if b > 1
            g[b-1] = g[b-1] * g[b]
            deleteat!(g, b)
        end
    end
    return g
end



"""
    bitreverse(f::TensorTrain) -> TensorTrain

Return a copy of `f` with bit-reversed site ordering.

This is the inverse of the mapping that places grid point `i` at the
multi-index position given by the binary representation of `i`.
"""
function bitreverse(f::TensorTrain)
    return TensorTrain(reverse(deepcopy(f.data)))
end
