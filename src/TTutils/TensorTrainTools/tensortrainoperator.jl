# Defines the TensorTrainOperator (MPO) struct and its constructors,
# matrix contraction, operator application, and utility functions.

export TensorTrainOperator
export matrix, identity_op, empty_op, random_op
export contract, diag, tto_direct_sum

"""
    TensorTrainOperator <: AbstractTensorTrain

Matrix Product Operator (MPO).
"""
struct TensorTrainOperator <: AbstractTensorTrain
    data::Vector{ITensor}
end


"""
    TensorTrainOperator(sites::Vector{<:Vector{<:Index}}, tensors::Vector{Array{T,4}})

Construct a TensorTrainOperator from a nested vector of site indices and
a vector of rank-4 arrays with shape `(left_dim, d, d, right_dim)`.

`sites` is a vector of length N where each element is a one-element vector
containing the physical index for that site.
"""
function TensorTrainOperator(
    sites::Vector{<:Vector{<:Index}},
    tensors::Vector{<:Array{T,4}}
) where {T}
    N = length(sites)
    length(tensors) != N && throw(ArgumentError("`sites` and `tensors` must have the same length."))

    flat_sites = [only(s) for s in sites] # get index
    dims_l = [size(tensors[k], 1) for k in 1:N]
    dims_r = [size(tensors[k], 4) for k in 1:N]

    links = Vector{Index}(undef, N + 1)
    links[1] = Index(dims_l[1], "Link,l=0")
    for k in 1:N
        links[k+1] = Index(dims_r[k], "Link,l=$k")
    end

    data = [
        itensor(tensors[k], links[k], flat_sites[k], flat_sites[k]', links[k+1])
        for k in 1:N
    ]
    return TensorTrainOperator(data)
end


"""
    TensorTrainOperator(sites_vec, WL, WR, tensors_vec)

Construct a TensorTrainOperator from finite-difference MPO stencil data:
- `sites_vec` : length-K vector of site-index vectors (each a row of sites)
- `WL`        : 1├ùbdim left boundary vector
- `WR`        : bdim├ù1 right boundary vector
- `tensors_vec`: nested vector of rank-4 bulk tensors

"""
function TensorTrainOperator(
    sites_vec::Vector{<:Vector{<:Index}},
    WL::AbstractMatrix,
    WR::AbstractMatrix,
    tensors_vec::Vector{<:Vector{<:AbstractArray}}
)
    # flatten sites
    flat_sites = vcat(sites_vec...)
    N = length(flat_sites)

    # flatten tensors
    flat_tensors = vcat(tensors_vec...)
    Nbulk = length(flat_tensors)

    # If only one bulk tensor is provided, broadcast it to all N sites
    if Nbulk == 1 && N > 1
        flat_tensors = fill(flat_tensors[1], N)
    end

    bdim = size(WR, 1)
    d    = dim(flat_sites[1])

    # Shared link indices ÔÇö all bulk bonds have dim bdim;
    # boundary bonds are contracted away via WL/WR below.
    links = [Index(bdim, "Link,l=$k") for k in 0:N]

    data = Vector{ITensor}(undef, N)
    for k in 1:N
        W    = flat_tensors[k]   # shape (bdim, d, d, bdim): (left, bra, ket, right)
        lk   = links[k]          # left  bond of site k (shared with site k-1)
        lkp1 = links[k+1]        # right bond of site k (shared with site k+1)
        # Build ITensor in canonical (left, bra, ket, right) ordering
        data[k] = itensor(W, lk, flat_sites[k], flat_sites[k]', lkp1)
    end

    # Absorb boundary vectors into the first and last tensors.
    # WL is (1, bdim): left-boundary weight vector on links[1].
    # WR is (bdim, 1): right-boundary weight vector on links[N+1].
    l0    = Index(1, "Link,l=0")
    lNend = Index(1, "Link,l=$N")
    WL_it = itensor(WL, l0, links[1])        # (1 ├ù bdim)
    WR_it = itensor(WR, links[N+1], lNend)   # (bdim ├ù 1)
    data[1] = data[1] * WL_it   # left bond: bdim ÔåÆ 1 (l0)
    data[N] = data[N] * WR_it   # right bond: bdim ÔåÆ 1 (lNend)

    return TensorTrainOperator(data)
end


"""
    TensorTrainOperator(mat::AbstractMatrix, sites::Vector{<:Index}; kwargs...)

Construct a TensorTrainOperator from a dense matrix via sequential SVD.
`kwargs` are forwarded to the truncated SVD.
"""
function TensorTrainOperator(mat::AbstractMatrix, sites::Vector{<:Index}; kwargs...)
    N  = length(sites)
    d  = dim(sites[1])
    M  = d^N
    size(mat) == (M, M) || throw(ArgumentError("Matrix must be $(M)├ù$(M)."))

    T    = eltype(mat)
    data = Vector{ITensor}(undef, N)

    # Reshape to (d, d, d, d, ÔÇª) tensor and build the MPO via SVD sweeps
    A = reshape(mat, ntuple(_ -> d, 2N)...)
    # interleave row and column dimensions:  (d_row_1, d_col_1, d_row_2, d_col_2, ÔÇª)
    A = permutedims(A, vcat([(2k-1, 2k) for k in 1:N]...))
    A = reshape(A, 1, d^2N)

    links = [Index(1, "Link,l=0")]
    for k in 1:N
        lk   = links[end]
        m    = dim(lk)
        Amat = reshape(A, m * d^2, d^(2*(N-k)))
        _maxdim = get(kwargs, :maxdim, typemax(Int))
        _cutoff = Float64(get(kwargs, :cutoff, 0.0))
        F    = LinearAlgebra.svd(Amat; alg=LinearAlgebra.DivideAndConquer())
        s    = truncate(F.S, _maxdim, _cutoff)
        r    = length(s)
        lkp1 = Index(r, "Link,l=$k")
        push!(links, lkp1)
        data[k] = itensor(reshape(F.U[:, 1:r] * Diagonal(s), m, d, d, r), lk, sites[k], sites[k]', lkp1)
        A = F.Vt[1:r, :]
    end
    lN  = links[end]
    lN1 = Index(1, "Link,l=$N")
    push!(links, lN1)
    data[N] = itensor(reshape(Matrix(A), dim(lN), d, d, 1), lN, sites[N], sites[N]', lN1)

    return TensorTrainOperator(data)
end

"""
    siteinds(Q::TensorTrainOperator; plev=0) -> Vector{Index}

Return the physical site indices of the MPO.  By default returns the
unprimed (bra) indices.
"""
function siteinds(Q::TensorTrainOperator; plev::Integer=0)
    N  = length(Q)
    return [
        only(filter(i -> ITensors.plev(i) == plev, [x for x in inds(Q[k]) if !hastags(x, "Link")]))
        for k in 1:N
    ]
end

"""
    siteinds(Q, j, plev)

Query the physical site indices associated with a tensor-train operator.

Arguments
- `Q`: Local tensor or vector variable used by the helper routine.
- `j`: Integer position selecting a site, core, or term.
- `plev`: Input parameter used by the local tensor-network calculation.

Returns
- Returns the physical site index or index collection selected by the overload and prime level.

Description
- Matrix-product operators carry both bra and ket physical legs. Accessing them consistently is essential when converting between dense matrices, MPO contractions, and sitewise local updates.
"""
function siteinds(Q::TensorTrainOperator, j::Integer; plev::Integer=0)
    return siteinds(Q; plev)[j]
end

"""
    siteinds(f, j)

Query the physical site indices associated with a tensor-train operator.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `j`: Integer position selecting a site, core, or term.

Returns
- Returns the physical site index or index collection selected by the overload and prime level.

Description
- Matrix-product operators carry both bra and ket physical legs. Accessing them consistently is essential when converting between dense matrices, MPO contractions, and sitewise local updates.
"""
siteinds(f::AbstractTensorTrain, j::Integer) = siteinds(f)[j]

"""
    matrix(Q::TensorTrainOperator) -> Matrix

Contract all cores and return the result as a dense matrix.
"""
function matrix(Q::TensorTrainOperator)
    M = Q[1]
    for k in 2:length(Q)
        M = M * Q[k]
    end
    ls = linkinds(Q)
    M  = M * itensor(ones(dim(ls[1])),   ls[1])
    M  = M * itensor(ones(dim(ls[end])), ls[end])
    bra_sites = siteinds(Q; plev=0)
    ket_sites = siteinds(Q; plev=1)
    d = prod(dim.(bra_sites))
    arr = Array(M, vcat(bra_sites, ket_sites)...)
    return reshape(arr, d, d)
end

"""
    identity_op([T,] sites; bonddim=1) -> TensorTrainOperator

Identity MPO.
"""
function identity_op(::Type{T}, sites::Vector{<:Index}; bonddim::Integer=1) where {T<:Number}
    N    = length(sites)
    d    = dim(sites[1])
    data = Vector{ITensor}(undef, N)
    lL   = Index(1, "Link,l=0")
    lR   = Index(bonddim, "Link,l=1")

    for k in 1:N
        lkp1 = k < N ? Index(bonddim, "Link,l=$k") : Index(1, "Link,l=$N")
        # Each core = identity on (site, site') ├ù identity on bond
        arr = zeros(T, 1, d, d, 1)
        for i in 1:d
            arr[1, i, i, 1] = one(T)
        end
        lk = (k == 1) ? Index(1, "Link,l=0") : lR
        lR = lkp1
        data[k] = itensor(arr, lk, sites[k], sites[k]', lkp1)
    end
    return TensorTrainOperator(data)
end

"""
    identity_op(sites, kwargs)

Construct the identity matrix-product operator on the supplied site spaces.

Arguments
- `sites`: Physical site indices defining the tensor-product grid or lattice.
- `kwargs`: Input parameter used by the local tensor-network calculation.

Returns
- Returns a tensor-train operator representing the identity map.

Description
- The identity MPO is the neutral element for operator composition and the tensorized version of doing nothing on each lattice site while preserving bond connectivity.
"""
identity_op(sites::Vector{<:Index}; kwargs...) = identity_op(Float64, sites; kwargs...)


"""
    empty_op([T,] sites, bonddim) -> TensorTrainOperator

Zero-filled MPO.
"""
function empty_op(::Type{T}, sites::Vector{<:Index}, bonddim::Integer) where {T<:Number}
    N     = length(sites)
    sdims = dim.(sites)
    ldims = linkdims(sdims, bonddim)
    data  = Vector{ITensor}(undef, N)
    links = vcat(Index(1, "Link,l=0"),
                 [Index(ldims[k+1], "Link,l=$k") for k in 1:N-1],
                 Index(1, "Link,l=$N"))
    for k in 1:N
        arr = zeros(T, ldims[k], sdims[k], sdims[k], ldims[k+1])
        data[k] = itensor(arr, links[k], sites[k], sites[k]', links[k+1])
    end
    return TensorTrainOperator(data)
end

"""
    empty_op(sites, bonddim)

Construct an all-zero matrix-product operator on the supplied site spaces.

Arguments
- `sites`: Physical site indices defining the tensor-product grid or lattice.
- `bonddim`: Requested uniform bond dimension for the constructed tensor train or MPO.

Returns
- Returns a zero-filled tensor-train operator.

Description
- Zero MPO buffers are useful as reusable workspaces and as starting points when assembling tensorized differential operators term by term.
"""
empty_op(sites::Vector{<:Index}, bonddim::Integer) = empty_op(Float64, sites, bonddim)


"""
    random_op([T,] sites, bonddim) -> TensorTrainOperator

Random MPO.
"""
function random_op(::Type{T}, sites::Vector{<:Index}, bonddim::Integer) where {T<:Number}
    N     = length(sites)
    sdims = dim.(sites)
    ldims = linkdims(sdims, bonddim)
    data  = Vector{ITensor}(undef, N)
    links = vcat(Index(1, "Link,l=0"),
                 [Index(ldims[k+1], "Link,l=$k") for k in 1:N-1],
                 Index(1, "Link,l=$N"))
    for k in 1:N
        arr = randn(T, ldims[k], sdims[k], sdims[k], ldims[k+1])
        data[k] = itensor(arr, links[k], sites[k], sites[k]', links[k+1])
    end
    return TensorTrainOperator(data)
end

"""
    random_op(sites, bonddim)

Construct a random matrix-product operator on the supplied site spaces.

Arguments
- `sites`: Physical site indices defining the tensor-product grid or lattice.
- `bonddim`: Requested uniform bond dimension for the constructed tensor train or MPO.

Returns
- Returns a randomly initialized tensor-train operator.

Description
- Random MPOs provide generic operator tests for contraction and compression routines because they excite all local channels without hand-designed structure.
"""
random_op(sites::Vector{<:Index}, bonddim::Integer) = random_op(Float64, sites, bonddim)


"""
    contract(A::TensorTrainOperator, b::TensorTrain) -> TensorTrain

Apply an MPO exactly to a tensor-train state before any later compression step.
"""
function contract(A::TensorTrainOperator, b::TensorTrain)
    N      = length(A)
    sites_b = siteinds(b)
    data   = [A[k] * b[k] for k in 1:N]
    # combine the bond indices pairwise
    g = TensorTrain(data)
    for k in 1:N-1
        # fuse the two link indices between sites k and k+1
        l_A = commonind(A[k], A[k+1])
        l_b = commonind(b[k], b[k+1])
        c   = combiner(l_A, l_b; tags=tags(l_b))
        g[k]   = g[k]   * c
        g[k+1] = g[k+1] * dag(c)
    end
    # After the interior loop, boundary tensors each have two dim-1 link
    # indices. Combine them into a single dim-1 boundary index so the result is
    # again a valid TensorTrain.
    left_extra  = filter(i -> hastags(i, "Link"), uniqueinds(g[1], g[2]))
    right_extra = filter(i -> hastags(i, "Link"), uniqueinds(g[N], g[N-1]))
    if length(left_extra) > 1
        g[1] = g[1] * combiner(left_extra...; tags=tags(left_extra[1]))
    end
    if length(right_extra) > 1
        g[N] = g[N] * combiner(right_extra...; tags=tags(right_extra[1]))
    end
    # The MPO introduces primed output site indices. Collapse them back onto the
    # MPS site indices so the result can be used as a state tensor train again.
    for k in 1:N
        g[k] = noprime(g[k], prime(sites_b[k]))
    end
    return g
end


"""
    contract(A::TensorTrainOperator, B::TensorTrainOperator) -> TensorTrainOperator

Exact MPO┬ÀMPO contraction.
"""
function contract(A::TensorTrainOperator, B::TensorTrainOperator)
    N    = length(A)
    data = [A[k] * B[k] for k in 1:N]
    g    = TensorTrainOperator(data)
    for k in 1:N-1
        l_A = commonind(A[k], A[k+1])
        l_B = commonind(B[k], B[k+1])
        c   = combiner(l_A, l_B; tags=tags(l_B))
        g[k]   = g[k]   * c
        g[k+1] = g[k+1] * dag(c)
    end
    left_extra  = filter(i -> hastags(i, "Link"), uniqueinds(g[1], g[2]))
    right_extra = filter(i -> hastags(i, "Link"), uniqueinds(g[N], g[N-1]))
    if length(left_extra) > 1
        g[1] = g[1] * combiner(left_extra...; tags=tags(left_extra[1]))
    end
    if length(right_extra) > 1
        g[N] = g[N] * combiner(right_extra...; tags=tags(right_extra[1]))
    end
    return g
end


"""
    tto_direct_sum(A::TensorTrainOperator, B::TensorTrainOperator) -> TensorTrainOperator

Compute the exact MPO sum whose dense action is `matrix(A) + matrix(B)`.
"""
function tto_direct_sum(A::TensorTrainOperator, B::TensorTrainOperator)
    length(A) == length(B) ||
        throw(ArgumentError("TensorTrainOperators must have the same length."))

    sites_A = siteinds(A; plev=0)
    sites_B = siteinds(B; plev=0)
    sites_A == sites_B ||
        throw(ArgumentError("TensorTrainOperators must share the same site indices."))

    N = length(A)
    B_r = replacelinks(copy(B))
    lA = linkinds(A)
    lB = linkinds(B_r)
    sites_Br = siteinds(B_r; plev=0)
    T = promote_type(promote_itensor_eltype(A), promote_itensor_eltype(B_r))

    arrays = Vector{Array{T,4}}(undef, N)
    for k in 1:N
        aA = Array(A[k],   lA[k], sites_A[k],  sites_A[k]',  lA[k+1])
        aB = Array(B_r[k], lB[k], sites_Br[k], sites_Br[k]', lB[k+1])
        rAl, d1, d2, rAr = size(aA)
        rBl, _,  _,  rBr = size(aB)

        if k == 1
            arr = zeros(T, 1, d1, d2, rAr + rBr)
            arr[1, :, :, 1:rAr] = aA[1, :, :, :]
            arr[1, :, :, rAr+1:end] = aB[1, :, :, :]
        elseif k == N
            arr = zeros(T, rAl + rBl, d1, d2, 1)
            arr[1:rAl, :, :, 1] = aA[:, :, :, 1]
            arr[rAl+1:end, :, :, 1] = aB[:, :, :, 1]
        else
            arr = zeros(T, rAl + rBl, d1, d2, rAr + rBr)
            arr[1:rAl, :, :, 1:rAr] = aA
            arr[rAl+1:end, :, :, rAr+1:end] = aB
        end

        arrays[k] = arr
    end

    return TensorTrainOperator([[s] for s in sites_A], arrays)
end


"""
    bitreverse(A::TensorTrainOperator) -> TensorTrainOperator

Return a copy with reversed site ordering.
"""
bitreverse(A::TensorTrainOperator) = TensorTrainOperator(reverse(deepcopy(A.data)))


"""
    diag(Q::TensorTrainOperator) -> TensorTrain

Extract the diagonal of the MPO as an MPS by replacing each (site, site')
pair with a single delta-contracted site index.
"""
function LinearAlgebra.diag(Q::TensorTrainOperator)
    sites = siteinds(Q; plev=0)
    data  = [Q[k] * delta(sites[k], sites[k]') for k in 1:length(Q)]
    return TensorTrain(data)
end
