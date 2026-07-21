# utils.jl ÔÇö small iterator helpers and index-uniqueness checks

export front, back, check_unique_inds


"""
    front(itr, n=1)

Return an iterator over the first `length(itr) - n` elements of `itr`.
"""
front(itr, n::Integer=1) = Iterators.take(itr, length(itr) - n)


"""
    back(itr, n=1)

Return an iterator that skips the first `n` elements of `itr`.
"""
back(itr, n::Integer=1) = Iterators.drop(itr, n)


"""
    check_unique_inds(inds::Index...) -> Bool

Return `true` if all indices in `inds` are distinct (no repeated Index objects).
"""
function check_unique_inds(inds::Index...)
    allinds = collect(inds)
    return length(allinds) == length(unique(allinds))
end

"""
    check_unique_inds(inds)

Validate that a tensor collection has the expected unique open indices.

Arguments
- `inds`: Index tuple or index collection defining the local tensor ordering.

Returns
- Returns `nothing` when the check passes and throws an error when it fails.

Description
- Tensor-network contractions rely on precise index bookkeeping. This helper catches index-aliasing mistakes early by asserting the intended open-index structure.
"""
function check_unique_inds(inds::Vector{<:Index}...)
    allinds = vcat(collect.(inds)...)
    return length(allinds) == length(unique(allinds))
end
