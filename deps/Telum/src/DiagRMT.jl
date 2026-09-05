"""
    DiagRMT{T,N}(diag, axis)

Reduced-matrix storage for a reshaped diagonal matrix.

Matrix form uses `1 <= axis[1] < axis[2] <= N`: both axes have size
`length(diag)`, all other axes are singleton, and only entries with equal
indices on the two axes are nonzero.

Vectorized form uses `axis[2] == 0`: `axis[1]` has size `length(diag)^2`, all
other axes are singleton, and only vectorized diagonal positions of the logical
`d x d` matrix are nonzero. This form is for prepared contraction data only;
`TLArray` sector storage must reject it.
"""
struct DiagRMT{T, N} <: AbstractArray{T, N}
    diag::Vector{T}
    axis::NTuple{2, Int}

    function DiagRMT{T,N}(diag::Vector{T}, axis::NTuple{2,Int}) where {T,N}
        N >= 2 || throw(ArgumentError("DiagRMT rank must be at least 2"))
        (1 <= axis[1] < axis[2] <= N || (1 <= axis[1] <= N && axis[2] == 0)) ||
            throw(ArgumentError("DiagRMT axis must satisfy 1 <= a < b <= N or b == 0"))
        return new{T,N}(diag, axis)
    end
end

DiagRMT(diag::Vector{T}, ::Val{N}, axis::NTuple{2,Int}) where {T,N} =
    DiagRMT{T,N}(diag, axis)
DiagRMT(diag::AbstractVector{T}, ::Val{N}, axis::NTuple{2,Int}) where {T,N} =
    DiagRMT{T,N}(Vector{T}(diag), axis)

Base.IndexStyle(::Type{<:DiagRMT}) = IndexCartesian()
Base.eltype(::Type{DiagRMT{T,N}}) where {T,N} = T

@inline diag_axis(r::DiagRMT) = r.axis
@inline diag_dim(r::DiagRMT) = length(r.diag)
@inline is_diag_rmt(::Type) = false
@inline is_diag_rmt(::Type{<:DiagRMT}) = true

function Base.size(r::DiagRMT{T,N}) where {T,N}
    d = length(r.diag)
    return ntuple(Val(N)) do i
        if r.axis[2] == 0
            i == r.axis[1] ? d * d : 1
        else
            i == r.axis[1] || i == r.axis[2] ? d : 1
        end
    end
end

Base.axes(r::DiagRMT{T,N}) where {T,N} = ntuple(i -> Base.OneTo(size(r, i)), Val(N))

@inline function _diag_rmt_vectorized_diag_index(p::Int, d::Int)
    1 <= p <= d * d || return 0
    q = p - 1
    step = d + 1
    q % step == 0 || return 0
    return div(q, step) + 1
end

@inline function Base.getindex(r::DiagRMT{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(r, I...)
    if r.axis[2] == 0
        p = I[r.axis[1]]
        diag_index = _diag_rmt_vectorized_diag_index(p, length(r.diag))
        diag_index == 0 && return zero(T)
        @inbounds for n in 1:N
            n == r.axis[1] && continue
            I[n] == 1 || return zero(T)
        end
        return r.diag[diag_index]
    else
        i = I[r.axis[1]]
        i == I[r.axis[2]] || return zero(T)
        @inbounds for n in 1:N
            (n == r.axis[1] || n == r.axis[2]) && continue
            I[n] == 1 || return zero(T)
        end
        return r.diag[i]
    end
end

function Base.Array{T,N}(r::DiagRMT{T,N}) where {T,N}
    out = zeros(T, size(r))
    d = length(r.diag)
    if r.axis[2] == 0
        for i in 1:d
            inds = ntuple(n -> n == r.axis[1] ? (i - 1) * (d + 1) + 1 : 1, Val(N))
            out[inds...] = r.diag[i]
        end
    else
        for i in 1:d
            inds = ntuple(n -> n == r.axis[1] || n == r.axis[2] ? i : 1, Val(N))
            out[inds...] = r.diag[i]
        end
    end
    return out
end

@inline dense_rmt(r::Array) = r
dense_rmt(r::DiagRMT{T,N}) where {T,N} = Array{T,N}(r)

function Base.permutedims(r::DiagRMT{T,N}, perm) where {T,N}
    perm_tuple = Tuple(perm)
    length(perm_tuple) == N || throw(ArgumentError("permutation length must be $N"))
    axis1 = findfirst(==(r.axis[1]), perm_tuple)
    isnothing(axis1) && throw(ArgumentError("perm must contain axis $(r.axis[1])"))
    if r.axis[2] == 0
        return DiagRMT{T,N}(r.diag, (axis1, 0))
    end
    axis2 = findfirst(==(r.axis[2]), perm_tuple)
    isnothing(axis2) && throw(ArgumentError("perm must contain axis $(r.axis[2])"))
    return DiagRMT{T,N}(r.diag, minmax(axis1, axis2))
end

@inline function _diag_scale(r::DiagRMT{T,N}, a) where {T,N}
    Tout = typeof(a * one(T))
    return DiagRMT{Tout,N}(Tout.(a .* r.diag), r.axis)
end

Base.copy(r::DiagRMT{T,N}) where {T,N} = DiagRMT{T,N}(copy(r.diag), r.axis)
Base.conj(r::DiagRMT{T,N}) where {T,N} = DiagRMT(conj.(r.diag), Val(N), r.axis)
Base.:*(a::Number, r::DiagRMT) = _diag_scale(r, a)
Base.:*(r::DiagRMT, a::Number) = _diag_scale(r, a)
Base.:/(r::DiagRMT, a::Number) = _diag_scale(r, inv(a))
Base.:-(r::DiagRMT) = _diag_scale(r, -1)
Base.sum(f::typeof(abs2), r::DiagRMT) = sum(abs2, r.diag)

function _diag_rmt_from_values(vals::AbstractVector{T}, ::Val{N},
                               axis::NTuple{2, Int}, scale = one(T)) where {T,N}
    Tout = typeof(scale * one(T))
    return DiagRMT{Tout,N}(Tout.(scale .* vals), axis)
end

function _dense_diagonal_rmt_from_values(vals::AbstractVector{T}, ::Val{N},
                                         scale = one(T)) where {T,N}
    Tout = typeof(scale * one(T))
    rmt = reshape(Matrix(Diagonal(Tout.(vals))), length(vals), length(vals), ones(Int, N - 2)...)
    rmt[:] .*= scale
    return rmt
end

function _diag_rmt_axes_if_valid(spaces::Tuple{Vararg{AbstractVector, QD}}) where {QD}
    nonsingleton = Int[]
    for leg in 1:QD
        if !(length(spaces[leg]) == 1 && only(spaces[leg])[2] == 1)
            push!(nonsingleton, leg)
        end
    end
    length(nonsingleton) == 2 || return nothing
    a, b = nonsingleton
    spaces[a] == spaces[b] || return nothing
    return (a, b)
end
