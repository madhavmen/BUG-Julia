# abstracttensortrain.jl
#
# Defines the AbstractTensorTrain supertype and all generic methods that work
# for both TensorTrain (MPS) and TensorTrainOperator (MPO).

export AbstractTensorTrain
export promote_itensor_eltype
export linkind, linkinds, replacelinks, replacelinks!
export siteinds, siteind, maxlinkdim
export connect, rescale!, dropdims

"""
    AbstractTensorTrain

Common supertype for tensor-train states (`TensorTrain`) and operators
(`TensorTrainOperator`).

Implementations are expected to store their cores in a `data::Vector{ITensor}`
field so the generic container-style methods defined in this file can operate on
both states and operators.
"""
abstract type AbstractTensorTrain end

"""
    promote_itensor_eltype(m::AbstractTensorTrain) -> Type

Return a scalar element type that can hold entries from every core of `m`.
This is useful when dense temporary arrays are assembled from a TT/MPO whose
individual cores may not all share exactly the same eltype.
"""
function promote_itensor_eltype(m::AbstractTensorTrain)
    return promote_type(eltype.(m.data)...)
end

# Treat tensor trains as light wrappers around a vector of ITensor cores.
"""
    Base.length(f)

Return the number of ITensor cores stored in a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the number of cores in the wrapped tensor-train or MPO container.

Description
- Tensor trains are stored as vectors of local cores. Forwarding `length` to that storage lets the rest of the code treat states and operators as one-dimensional containers without changing their tensor-network meaning.
"""
Base.length(f::AbstractTensorTrain)       = length(f.data)
"""
    Base.size(f)

Expose the tensor train as a one-dimensional Julia container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the one-dimensional container size tuple for the wrapped core vector.

Description
- The TT/MPO object behaves like a linear container of local cores, so `size` reports the container view rather than any physical lattice dimensions.
"""
Base.size(f::AbstractTensorTrain)         = (length(f),)
"""
    Base.ndims(f)

Report the container dimensionality of a tensor train wrapper.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns `1` because the wrapper is presented as a vector of local cores.

Description
- Although the represented state is high-dimensional, the wrapper API deliberately models it as a one-dimensional sequence of tensors so generic container code can iterate over sites.
"""
Base.ndims(f::AbstractTensorTrain)        = 1
"""
    Base.eachindex(f)

Return the valid core indices of a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the iterable index set of the wrapped core storage.

Description
- This forwards Julia indexing semantics to the underlying core vector, which is the natural site ordering used by TT sweeps and contractions.
"""
Base.eachindex(f::AbstractTensorTrain)    = eachindex(f.data)
"""
    Base.firstindex(f)

Return the first valid site index of a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the first index of the wrapped core vector.

Description
- The helper keeps tensor trains compatible with Julia container APIs while preserving the left-to-right ordering used by BUG and compression sweeps.
"""
Base.firstindex(f::AbstractTensorTrain)   = firstindex(f.data)
"""
    Base.lastindex(f)

Return the last valid site index of a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the last index of the wrapped core vector.

Description
- This is the right boundary of the site chain, which many sweep algorithms use when building environments from the right.
"""
Base.lastindex(f::AbstractTensorTrain)    = lastindex(f.data)
"""
    Base.getindex(f, i)

Fetch one local core from a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `i`: Integer position selecting a site, core, or container entry.

Returns
- Returns the requested ITensor core at the specified site position.

Description
- Direct indexed access is the low-level primitive behind local TT updates, environment construction, and gauge manipulations.
"""
Base.getindex(f::AbstractTensorTrain, i)  = f.data[i]
"""
    Base.setindex!(f, v, i)

Overwrite one local core inside a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `v`: Replacement value, local tensor, or test vector supplied to the helper routine.
- `i`: Integer position selecting a site, core, or container entry.

Returns
- Returns the mutated tensor-train container after updating the requested core.

Description
- Local projector-splitting and compression routines repeatedly replace one or two cores at a time. This wrapper forwards those writes to the underlying storage without adding extra tensor logic.
"""
Base.setindex!(f::AbstractTensorTrain, v, i) = setindex!(f.data, v, i)
"""
    Base.iterate(f, state)

Iterate through the local cores of a tensor train in site order.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `state`: Iterator state used by Julia's `iterate` protocol.

Returns
- Returns the next `(core, state)` pair for Julia iteration or `nothing` at the end.

Description
- Treating a tensor train as an iterable sequence of cores keeps utility code generic while respecting the physical left-to-right lattice ordering.
"""
Base.iterate(f::AbstractTensorTrain, state=1) = state > length(f) ? nothing : (f.data[state], state + 1)

"""
    Base.push!(f, v)

Append a new core to the end of a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `v`: Test vector on which the local reduced operator is applied.

Returns
- Returns the mutated tensor-train container after appending the new core.

Description
- This is a low-level container operation used when constructing or reshaping TT objects explicitly from their site tensors.
"""
Base.push!(f::AbstractTensorTrain, v)     = push!(f.data, v)
"""
    Base.pop!(f)

Remove and return the last core from a tensor-train container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the core removed from the right end of the wrapped storage.

Description
- The operation mirrors vector semantics on the underlying core list and is occasionally useful in TT construction utilities.
"""
Base.pop!(f::AbstractTensorTrain)         = pop!(f.data)
"""
    Base.insert!(f, i, v)

Insert a core at a specified site position in the container.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `i`: Integer position selecting a site, core, or container entry.
- `v`: Replacement value, local tensor, or test vector supplied to the helper routine.

Returns
- Returns the mutated tensor-train container after insertion.

Description
- The method exposes structural edits on the core list, which can be useful in debugging or building custom tensor-train layouts.
"""
Base.insert!(f::AbstractTensorTrain, i, v) = insert!(f.data, i, v)
"""
    Base.deleteat!(f, i)

Delete a core at a specified site position.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `i`: Integer position selecting a site, core, or container entry.

Returns
- Returns the mutated tensor-train container after deletion.

Description
- This is a structural container edit on the core sequence; it operates on storage only and leaves any higher-level TT consistency checks to the caller.
"""
Base.deleteat!(f::AbstractTensorTrain, i) = deleteat!(f.data, i)
"""
    Base.append!(f, v)

Append a collection of cores to the tensor-train storage.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `v`: Test vector on which the local reduced operator is applied.

Returns
- Returns the mutated tensor-train container after extending the core list.

Description
- The wrapper preserves Julia container ergonomics for bulk TT construction while keeping the internal representation explicit.
"""
Base.append!(f::AbstractTensorTrain, v)   = append!(f.data, v)
"""
    Base.reverse!(f)

Reverse the order of the stored tensor-train cores in place.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the mutated tensor-train container after reversing the core order.

Description
- Reversing the core list is a structural operation used by utilities such as bit reversal or reverse sweeps that reinterpret the chain orientation.
"""
Base.reverse!(f::AbstractTensorTrain)     = reverse!(f.data)

Base.copy(f::T) where {T<:AbstractTensorTrain}     = T(copy(f.data))
Base.deepcopy(f::T) where {T<:AbstractTensorTrain} = T(deepcopy(f.data))

"""
    Base.copyto!(dest, src)

Copy all cores from one tensor-train container into another.

Arguments
- `dest`: Destination tensor-train container that receives copied data.
- `src`: Source tensor-train container whose data are copied.

Returns
- Returns the destination tensor train after its cores have been overwritten.

Description
- This performs a site-wise copy on the core sequence, which is the correct level of granularity when two TT objects already share the same index structure.
"""
function Base.copyto!(dest::AbstractTensorTrain, src::AbstractTensorTrain)
    length(dest) != length(src) && throw(ArgumentError("TensorTrains have different lengths."))
    for i in eachindex(dest)
        dest[i] = src[i]
    end
    return dest
end

Base.map(fn, f::T) where {T<:AbstractTensorTrain} = T([fn(t) for t in f.data])

"""
    Base.map!(fn, f)

Apply a transformation to every core of a tensor train in place.

Arguments
- `fn`: Input parameter used by the local tensor-network calculation.
- `f`: Tensor-train input container or tensor-train state supplied to the utility.

Returns
- Returns the mutated tensor-train container after mapping the transformation across all sites.

Description
- Many gauge or tagging operations act independently on each core, so an in-place map preserves the TT structure while changing every local tensor consistently.
"""
function Base.map!(fn, f::AbstractTensorTrain)
    for i in eachindex(f)
        f[i] = fn(f[i])
    end
    return f
end

"""
    Base.fill!(f, v)

Fill every stored core tensor with the same scalar value.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `v`: Test vector on which the local reduced operator is applied.

Returns
- Returns the mutated tensor-train container after all entries have been filled.

Description
- This is a storage-level initialization helper used when constructing zero or constant work buffers for TT algorithms.
"""
function Base.fill!(f::AbstractTensorTrain, v)
    for t in f
        fill!(t, v)
    end
    return f
end

# ÔöÇÔöÇ ITensors broadcast overloads ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
ITensors.dag(f::T, args...) where {T<:AbstractTensorTrain} =
    T([ITensors.dag(t, args...) for t in f.data])

for fn in (:prime, :replacetags, :settags, :addtags, :removetags, :setprime, :noprime)
    @eval ITensors.$fn(f::T, args...) where {T<:AbstractTensorTrain} =
        T([ITensors.$fn(t, args...) for t in f.data])

    @eval function ITensors.$(Symbol(fn, :!))(f::AbstractTensorTrain, args...)
        for i in eachindex(f)
            ITensors.$(Symbol(fn, :!))(f[i], args...)
        end
        return f
    end
end


ITensors.prime(fn::Function, f::T, args...) where {T<:AbstractTensorTrain} =
    T([ITensors.prime(fn, t, args...) for t in f.data])

"""
    ITensors.prime!(fn, f, args)

Prime every core index in a tensor-train container in place.

Arguments
- `fn`: Input parameter used by the local tensor-network calculation.
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `args`: Input parameter used by the local tensor-network calculation.

Returns
- Returns the mutated tensor train after the requested prime-level change.

Description
- Prime-level shifts distinguish bra and ket copies of the same physical or bond space. Applying them uniformly across a TT container is fundamental to safe tensor-network contractions.
"""
function ITensors.prime!(fn::Function, f::AbstractTensorTrain, args...)
    for i in eachindex(f)
        ITensors.prime!(fn, f[i], args...)
    end
    return f
end

ITensors.swapprime(f::T, args...) where {T<:AbstractTensorTrain} =
    T([ITensors.swapprime(t, args...) for t in f.data])

"""
    ITensors.swapprime!(f, args)

Swap one prime level for another across all cores in place.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `args`: Input parameter used by the local tensor-network calculation.

Returns
- Returns the mutated tensor train after the prime labels have been exchanged.

Description
- This helper keeps TT-level index bookkeeping consistent when moving between bra/ket or input/output operator conventions.
"""
function ITensors.swapprime!(f::AbstractTensorTrain, args...)
    for i in eachindex(f)
        ITensors.swapprime!(f[i], args...)
    end
    return f
end

ITensors.replaceinds(f::T, args...) where {T<:AbstractTensorTrain} =
    T([ITensors.replaceinds(t, args...) for t in f.data])

"""
    ITensors.replaceinds!(f, args)

Replace multiple indices across every core of a tensor train.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `args`: Input parameter used by the local tensor-network calculation.

Returns
- Returns the mutated tensor train after the requested index substitutions.

Description
- Bulk index replacement is how TT utilities retag or reconnect an existing network without changing the stored tensor data.
"""
function ITensors.replaceinds!(f::AbstractTensorTrain, args...)
    for i in eachindex(f)
        ITensors.replaceinds!(f[i], args...)
    end
    return f
end

ITensors.replaceind(f::T, args...) where {T<:AbstractTensorTrain} =
    T([ITensors.replaceind(t, args...) for t in f.data])

"""
    ITensors.replaceind!(f, args)

Replace one index across every core of a tensor train.

Arguments
- `f`: Tensor-train input container or tensor-train state supplied to the utility.
- `args`: Input parameter used by the local tensor-network calculation.

Returns
- Returns the mutated tensor train after the index substitution.

Description
- This is the TT-wide version of an index relabeling, which is frequently needed when aligning independent tensor networks before contraction.
"""
function ITensors.replaceind!(f::AbstractTensorTrain, args...)
    for i in eachindex(f)
        ITensors.replaceind!(f[i], args...)
    end
    return f
end

# ÔöÇÔöÇ Abstract Tensor utilities ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

"""
    linkind(f::AbstractTensorTrain, j) -> Index

Return the link index shared between tensors `j` and `j+1`.
"""
function linkind(f::AbstractTensorTrain, j::Integer)
    return commonind(f[j], f[j+1])
end

"""
    linkinds(f::AbstractTensorTrain, j) -> Vector{Index}

Return all link indices connecting tensors `j` and `j+1`.
"""
function linkinds(f::AbstractTensorTrain, j::Integer)
    return commoninds(f[j], f[j+1])
end

"""
    linkinds(f::AbstractTensorTrain) -> Vector{Index}

Return the N+1 link indices `[lÔéÇ, lÔéü, ÔÇª, l_N]`, including the
dimension-1 boundary indices at both ends.
"""
function linkinds(f::AbstractTensorTrain)
    N = length(f)
    l0 = only(uniqueinds(f[1], f[2]; tags="Link"))
    # interior bonds
    interior = [linkind(f, j) for j in 1:N-1]
    # right boundary
    lN = only(uniqueinds(f[N], f[N-1]; tags="Link"))
    return vcat(l0, interior, lN)
end


"""
    replacelinks(f::T) -> T

Return a copy of `f` with fresh link (bond) indices ÔÇö useful when the same
tensor train is to appear twice in a contraction.
"""
function replacelinks(f::T) where {T<:AbstractTensorTrain}
    g = deepcopy(f)
    replacelinks!(g)
    return g
end

"""
    replacelinks!(f::AbstractTensorTrain)

Replace all link indices of `f` in-place with freshly created indices.
"""
function replacelinks!(f::AbstractTensorTrain)
    N = length(f)
    for j in 1:N-1
        old     = linkind(f, j)
        new_idx = sim(old)                  # same dim+tags, new unique ID
        replaceind!(f[j],   old, new_idx)
        replaceind!(f[j+1], old, new_idx)
    end
    return f
end


"""
    maxlinkdim(f::AbstractTensorTrain) -> Int

Return the maximum bond (link) dimension.
"""
function maxlinkdim(f::AbstractTensorTrain)
    N = length(f)
    return maximum(dim(linkind(f, j)) for j in 1:N-1)
end


"""
    connect(f::T, g::T) -> T

Concatenate two tensor trains `f` and `g` by contracting their boundary
link indices.
"""
function connect(f::T, g::T) where {T<:AbstractTensorTrain}
    lf = linkinds(f)
    lg = linkinds(g)
    # contract the right boundary of f with the left boundary of g
    fN  = f[end] * delta(lf[end], lg[1])
    return T(vcat(f.data[1:end-1], [fN], g.data[2:end]))
end

# alias used in TTA source
const join = connect


"""
    rescale!(f::AbstractTensorTrain, c; gauge_centre=length(f))

Multiply tensor train `f` by scalar `c` in-place by scaling the tensor at
`gauge_centre`. Default  Gauge center is chosen to be the last site of the TT.
"""
function rescale!(f::AbstractTensorTrain, c::Number; gauge_centre::Integer=length(f))
    f[gauge_centre] .*= c
    return f
end

"""
    dropdims(A::ITensor) -> ITensor

Drop all dimension-1 indices from the ITensor `A`.
"""
function Base.dropdims(A::ITensor)
    trivial = filter(i -> dim(i) == 1, inds(A)) # find the indices with dim=1
    isempty(trivial) && return A
    result = A
    for i in trivial
        result = result * itensor(ones(1), i)
    end
    return result
end

"""
    dropdims(f::T, j::Integer) -> T

Drop dimension-1 link indices at position `j` of tensor train `f`.
"""
function Base.dropdims(f::T, j::Integer) where {T<:AbstractTensorTrain}
    g = deepcopy(f)
    g[j] = dropdims(g[j])
    return g
end

"""
    dropdims(f::T) -> T

Drop all dimension-1 boundary indices from tensor train `f`.
"""
function Base.dropdims(f::T) where {T<:AbstractTensorTrain}
    g = dropdims(f, 1)
    g = dropdims(g, length(g))
    return g
end
