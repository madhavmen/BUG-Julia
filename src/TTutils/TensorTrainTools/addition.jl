# Deterministic summation of TensorTrains.

export add, add!


"""
    add(f::Vector{TensorTrain}, bonddim::Integer; cutoff=0.0, kwargs...) -> TensorTrain

Return a deterministic TT approximation to `sum(f)` by exact direct-sum
construction followed by SVD compression. Legacy sweep-control keyword
arguments are accepted for compatibility and ignored.
"""
function add(f::Vector{TensorTrain}, bonddim::Integer; kwargs...)
    isempty(f) && throw(ArgumentError("Need at least one TensorTrain to add."))
    length(f) == 1 && return compress(f[1], bonddim; kwargs...)

    ref_sites = siteinds(f[1])
    !all(length(tt) == length(f[1]) for tt in f) &&
        throw(ArgumentError("All TensorTrains must have the same length."))
    !all(siteinds(tt) == ref_sites for tt in f) &&
        throw(ArgumentError("All TensorTrains must share the same site indices."))

    summed = deepcopy(f[1])
    for k in 2:length(f)
        summed = _tt_exact_sum(summed, f[k])
    end
    return compress(summed, bonddim; kwargs...)
end


"""
    add!(out::TensorTrain, f::Vector{TensorTrain}; cutoff=0.0, kwargs...) -> TensorTrain

Overwrite `out` with a deterministic compressed approximation to `sum(f)`.
Legacy sweep-control keyword arguments are accepted for compatibility and
ignored.
"""
function add!(out::TensorTrain, f::Vector{TensorTrain}; kwargs...)
    result = add(f, max(maxlinkdim(out), 1); kwargs...)
    copyto!(out, result)
    return out
end


"""
    _tt_exact_sum(¤ê, ¤ò)

Build the exact direct sum of two tensor trains before optional recompression.

Arguments
- `¤ê`: Tensor-train state being queried or updated by the local algorithm.
- `¤ò`: Input parameter used by the local tensor-network calculation.

Returns
- Returns a tensor train representing the algebraically exact sum on an enlarged bond space.

Description
- Adding tensor trains exactly is achieved by block-diagonal concatenation of their local bond spaces. Compression can then be applied afterwards as a separate approximation step.
"""
function _tt_exact_sum(¤ê::TensorTrain, ¤ò::TensorTrain)
    length(¤ê) == length(¤ò) ||
        throw(ArgumentError("TensorTrains must have the same length."))
    siteinds(¤ê) == siteinds(¤ò) ||
        throw(ArgumentError("TensorTrains must share the same site indices."))

    N = length(¤ê)
    sites = siteinds(¤ê)
    ¤ò_r = replacelinks(copy(¤ò))

    l¤ê = linkinds(¤ê)
    l¤ò = linkinds(¤ò_r)
    sites_¤ò = siteinds(¤ò_r)

    A¤ê = [array(¤ê[k], l¤ê[k], sites[k], l¤ê[k + 1]) for k in 1:N]
    A¤ò = [array(¤ò_r[k], l¤ò[k], sites_¤ò[k], l¤ò[k + 1]) for k in 1:N]
    T = promote_type(eltype(A¤ê[1]), eltype(A¤ò[1]))

    arrays = Vector{Array{T,3}}(undef, N)
    for k in 1:N
        a¤ê = A¤ê[k]
        a¤ò = A¤ò[k]
        r¤êl, dk, r¤êr = size(a¤ê)
        r¤òl, _, r¤òr = size(a¤ò)

        if k == 1
            A = zeros(T, 1, dk, r¤êr + r¤òr)
            A[1, :, 1:r¤êr] = a¤ê[1, :, :]
            A[1, :, r¤êr + 1:end] = a¤ò[1, :, :]
        elseif k == N
            A = zeros(T, r¤êl + r¤òl, dk, 1)
            A[1:r¤êl, :, 1] = a¤ê[:, :, 1]
            A[r¤êl + 1:end, :, 1] = a¤ò[:, :, 1]
        else
            A = zeros(T, r¤êl + r¤òl, dk, r¤êr + r¤òr)
            A[1:r¤êl, :, 1:r¤êr] = a¤ê
            A[r¤êl + 1:end, :, r¤êr + 1:end] = a¤ò
        end

        arrays[k] = A
    end

    return TensorTrain(sites, arrays)
end
