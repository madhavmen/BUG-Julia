using Printf
using LinearAlgebra
include("utils.jl")
include("localspaces/localspaces.jl")

# A compile-time tag for a direct product of symmetry groups. Telum records
# symmetry identities in type parameters; use `symm(q)` to retrieve them
# without going through dynamic property access.
abstract type ProductSymm{Syms<:Tuple{Vararg{Symmetry}}} <: Symmetry end

ProductSymm(syms::Type{<:Symmetry}...) = ProductSymm{Tuple{syms...}}

product_symms(::Type{<:ProductSymm{Syms}}) where {Syms} = Tuple(Syms.parameters)
nsymms(::Type{<:ProductSymm{Syms}}) where {Syms} = length(Syms.parameters)

function productsymm(symm::Tuple)
    all(s -> s isa Type{<:Symmetry}, symm) || throw(ArgumentError(
        "all entries of a product symmetry must be symmetry types, got $symm"))
    return ProductSymm(symm...)
end

productsymm(::Type{PS}) where {PS<:ProductSymm} = PS
product_symms(symm::Tuple) = product_symms(productsymm(symm))
nsymms(symm::Tuple) = nsymms(productsymm(symm))

@generated function nonabelian_symmetry_indices(::Type{PS}) where {PS<:ProductSymm}
    syms = product_symms(PS)
    inds = Tuple(i for i in eachindex(syms) if !isabelian(syms[i]))
    return :($inds)
end

@inline n_nonabelian_symmetries(::Type{PS}) where {PS<:ProductSymm} =
    length(nonabelian_symmetry_indices(PS))

@inline is_stored_wmat_symmetry(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n} =
    wmat_tuple_slot(PS, Val(n)) !== nothing

@generated function nonabelian_wmat_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    slot = 0
    for i in eachindex(syms)
        isabelian(syms[i]) || (slot += 1)
        i == n && break
    end
    1 <= n <= length(syms) && !isabelian(syms[n]) ||
        return :(throw(ArgumentError("symmetry index $n is Abelian and has no stored w-matrix slot")))
    return :($slot)
end

function nonabelian_wmat_slot(::Type{PS}, n::Int) where {PS<:ProductSymm}
    syms = product_symms(PS)
    1 <= n <= length(syms) || throw(BoundsError(syms, n))
    slot = 0
    for i in eachindex(syms)
        isabelian(syms[i]) || (slot += 1)
        i == n && break
    end
    isabelian(syms[n]) &&
        throw(ArgumentError("symmetry index $n is Abelian and has no stored w-matrix slot"))
    return slot
end

@generated function wmat_tuple_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    1 <= n <= length(syms) || return :(throw(BoundsError(product_symms(PS), $n)))
    isabelian(syms[n]) && return :(nothing)
    slot = count(i -> !isabelian(syms[i]), 1:n)
    return :($slot)
end

@generated function product_symmetry_index_from_wmat_slot(::Type{PS}, ::Val{slot}) where {PS<:ProductSymm, slot}
    syms = product_symms(PS)
    count = 0
    for i in eachindex(syms)
        isabelian(syms[i]) && continue
        count += 1
        count == slot && return :($i)
    end
    return :(throw(BoundsError(nonabelian_symmetry_indices(PS), $slot)))
end

@inline _trivial_wmat() = ones(1, 1)

const WMatInfo{M} = NTuple{M, NTuple{3, Int}}

@inline _wmat_tuple_width(::Type{PS}) where {PS<:ProductSymm} =
    n_nonabelian_symmetries(PS)

@inline _empty_wmat_info(::Val{M}) where {M} =
    ntuple(_ -> (0, 0, 0), Val(M))

@inline function _wmat_info_present(info::NTuple{3, Int})
    offset, nrow, ncol = info
    return offset != 0 || nrow != 0 || ncol != 0
end

function _validate_wmat_storage(wmatdata::Vector{Float64},
                                wmatinfo::Vector{WMatInfo{M}},
                                nslots::Int) where {M}
    length(wmatinfo) == nslots ||
        throw(ArgumentError("w-matrix info length must equal number of sector slots"))
    ranges = Tuple{Int, Int}[]
    for sector in eachindex(wmatinfo)
        infos = wmatinfo[sector]
        for slot in 1:M
            offset, nrow, ncol = infos[slot]
            if offset == 0 && nrow == 0 && ncol == 0
                continue
            end
            offset >= 1 || throw(ArgumentError("w-matrix sector $sector slot $slot has invalid offset $offset"))
            nrow > 0 && ncol > 0 ||
                throw(ArgumentError("w-matrix sector $sector slot $slot has invalid size ($nrow, $ncol)"))
            last = offset + nrow * ncol - 1
            last <= length(wmatdata) ||
                throw(ArgumentError("w-matrix sector $sector slot $slot range exceeds wmatdata length"))
            push!(ranges, (offset, last))
        end
    end
    sort!(ranges; by=first)
    for i in 2:length(ranges)
        ranges[i][1] <= ranges[i - 1][2] &&
            throw(ArgumentError("w-matrix arena ranges must not overlap"))
    end
    return nothing
end

function _empty_wmat_storage(::Type{PS}) where {PS<:ProductSymm}
    M = _wmat_tuple_width(PS)
    return Float64[], WMatInfo{M}[]
end

@inline _empty_wmat_storage(symm::NTuple) =
    _empty_wmat_storage(productsymm(symm))

function _append_wmat_tuple!(wmatdata::Vector{Float64},
                             wmatinfo::Vector{WMatInfo{M}},
                             wmats::NTuple{M, <:AbstractMatrix}) where {M}
    info = ntuple(Val(M)) do slot
        wmat = wmats[slot]
        len = length(wmat)
        len == 0 && return (0, 0, 0)
        offset = length(wmatdata) + 1
        append!(wmatdata, vec(wmat))
        (offset, size(wmat, 1), size(wmat, 2))
    end
    push!(wmatinfo, info)
    return wmatdata, wmatinfo
end

function _store_wmat_tuple!(wmatdata::Vector{Float64},
                            wmatinfo::Vector{WMatInfo{M}},
                            sector::Int,
                            next_offset::Int,
                            wmats::NTuple{M, <:AbstractMatrix}) where {M}
    wmatinfo[sector] = ntuple(Val(M)) do slot
        wmat = wmats[slot]
        len = length(wmat)
        len == 0 && return (0, 0, 0)
        offset = next_offset
        copyto!(view(wmatdata, offset:offset + len - 1), vec(wmat))
        next_offset += len
        (offset, size(wmat, 1), size(wmat, 2))
    end
    return next_offset
end

function _unit_wmat_storage(::Type{PS}, nslots::Int) where {PS<:ProductSymm}
    M = _wmat_tuple_width(PS)
    total = M * nslots
    data = ones(Float64, total)
    info = Vector{WMatInfo{M}}(undef, nslots)
    offset = 1
    for sector in 1:nslots
        info[sector] = ntuple(Val(M)) do _
            current = offset
            offset += 1
            (current, 1, 1)
        end
    end
    return data, info
end

@inline _unit_wmat_storage(symm::NTuple, nslots::Int) =
    _unit_wmat_storage(productsymm(symm), nslots)

@inline function _wmat_from_storage(wmatdata::Vector{Float64},
                                    wmatinfo::Vector{WMatInfo{M}},
                                    sector::Int,
                                    slot::Int) where {M}
    offset, nrow, ncol = wmatinfo[sector][slot]
    if offset == 0 && nrow == 0 && ncol == 0
        throw(ArgumentError("w-matrix sector $sector slot $slot is not assigned"))
    end
    return reshape(@view(wmatdata[offset:offset + nrow * ncol - 1]), nrow, ncol)
end

@inline function _copy_wmat_to_storage!(wmatdata::Vector{Float64},
                                        wmatinfo::Vector{WMatInfo{M}},
                                        sector::Int,
                                        slot::Int,
                                        wmat::AbstractMatrix) where {M}
    offset, nrow, ncol = wmatinfo[sector][slot]
    if offset == 0 && nrow == 0 && ncol == 0
        throw(ArgumentError("w-matrix sector $sector slot $slot is not assigned"))
    end
    size(wmat) == (nrow, ncol) ||
        throw(DimensionMismatch("w-matrix sector $sector slot $slot expected size ($nrow, $ncol), got $(size(wmat))"))
    copyto!(view(wmatdata, offset:offset + nrow * ncol - 1), vec(wmat))
    return wmatdata
end

# ─── Tag helpers ──────────────────────────────────────────────────────────────
#
# Tags are stored internally as an Itag wrapper around a canonical
# comma-separated String sorted alphabetically, e.g. "bond,site,u1"
# (ITensor convention). Empty string means no tags. Individual tags contain no
# commas or whitespace.
#
# ─────────────────────────────────────────────────────────────────────────────

_itag_text(tags::AbstractString) = String(tags)

# Split tag text into sorted vector of non-empty individual tags.
_parse_itag(tags::AbstractString) = sort!(filter!(!isempty, strip.(split(_itag_text(tags), ','))))

# Canonical form: sorted, comma-joined, no spaces.
_normalize_itag_text(tags::AbstractString) = join(_parse_itag(tags), ',')

struct Itag <: AbstractString
    value::String

    Itag(tags::AbstractString) = new(_normalize_itag_text(tags))
end

Itag(tags::Itag) = tags
_itag_text(tags::Itag) = tags.value
_normalize_itag(tags::AbstractString) = Itag(tags)

Base.String(tags::Itag) = tags.value
Base.convert(::Type{String}, tags::Itag) = tags.value
Base.ncodeunits(tags::Itag) = ncodeunits(tags.value)
Base.codeunit(::Type{Itag}) = codeunit(String)
Base.codeunit(tags::Itag, i::Int) = codeunit(tags.value, i)
Base.isvalid(tags::Itag, i::Int) = isvalid(tags.value, i)
Base.iterate(tags::Itag) = iterate(tags.value)
Base.iterate(tags::Itag, i::Int) = iterate(tags.value, i)
Base.length(tags::Itag) = length(tags.value)
Base.lastindex(tags::Itag) = lastindex(tags.value)
Base.thisind(tags::Itag, i::Int) = thisind(tags.value, i)
Base.nextind(tags::Itag, i::Int) = nextind(tags.value, i)
Base.nextind(tags::Itag, i::Int, n::Int) = nextind(tags.value, i, n)
Base.prevind(tags::Itag, i::Int) = prevind(tags.value, i)
Base.prevind(tags::Itag, i::Int, n::Int) = prevind(tags.value, i, n)
Base.show(io::IO, tags::Itag) = print(io, tags.value)
Base.isempty(tags::Itag) = isempty(tags.value)
Base.hash(tags::Itag, h::UInt) = hash(tags.value, h)
Base.:(==)(a::Itag, b::Itag) = a.value == b.value
Base.:(==)(a::Itag, b::AbstractString) = a.value == _normalize_itag_text(b)
Base.:(==)(a::AbstractString, b::Itag) = _normalize_itag_text(a) == b.value
Base.isequal(a::Itag, b::Itag) = (a == b)
Base.isequal(a::Itag, b::AbstractString) = (a == b)
Base.isequal(a::AbstractString, b::Itag) = (a == b)

# True iff all comma-separated tags in `query` are present in `base`.
function _has_itag(base::AbstractString, query::AbstractString)
    bset = Set(_parse_itag(base))
    return all(t -> t ∈ bset, _parse_itag(query))
end

_matches_itag_selector(base::AbstractString, query::AbstractString) = _has_itag(base, query)
_matches_itag_selector(base::AbstractString, queries::Tuple{Vararg{AbstractString}}) =
    any(query -> _has_itag(base, query), queries)
_matches_itag_selector(base::AbstractString, queries::AbstractVector{<:AbstractString}) =
    any(query -> _has_itag(base, query), queries)

# Add tags from `newtags` to `base`; result is sorted and deduplicated.
_add_itag(base::AbstractString, newtags::AbstractString) =
    Itag(join(sort!(unique!(vcat(_parse_itag(base), _parse_itag(newtags)))), ','))

# Remove every tag listed in `rmtags` from `base`.
function _remove_itag(base::AbstractString, rmtags::AbstractString)
    rm = Set(_parse_itag(rmtags))
    return Itag(join(filter(t -> t ∉ rm, _parse_itag(base)), ','))
end

# Remove tags from `base` using one or more ITensor-style tag queries.
# Each query is applied independently, and its tags are removed only if all
# tags in that query are present in `base`.
function _remove_itag(base::AbstractString,
                      rmtags::Union{Tuple{Vararg{AbstractString}},
                                    AbstractVector{<:AbstractString}})
    base_tags = _parse_itag(base)
    base_set = Set(base_tags)
    rm = Set{String}()
    for query in rmtags
        parsed = _parse_itag(query)
        all(tag -> tag ∈ base_set, parsed) || continue
        union!(rm, parsed)
    end
    return Itag(join(filter(tag -> tag ∉ rm, base_tags), ','))
end

function _replace_itag(base::AbstractString, replacements)
    base_tags = _parse_itag(base)
    base_set = Set(base_tags)
    replaced = Set{String}()
    new_tags = String[]

    for (from, to) in replacements
        from_tags = _parse_itag(from)
        all(tag -> tag ∈ base_set, from_tags) || continue
        union!(replaced, from_tags)
        append!(new_tags, _parse_itag(to))
    end

    append!(new_tags, (tag for tag in base_tags if tag ∉ replaced))
    return Itag(join(sort!(unique!(new_tags)), ','))
end

struct TLIndex
    # Tags associated with the leg, similar to ITensor
    itags::Itag
    # Direction of the leg, '+' for incoming, '-' for outgoing
    dir::Char
    # Prime level similar to ITensor
    plev::Int
    # Lock level. Cannot cntracted if lock > 0. 
    # Decrease lock level by 1 after contraction.
    lock::Int
    # If true, this leg represents the dual space.
    dual::Bool

    TLIndex(itags::AbstractString, dir::Char, plev::Int=0, lock::Int=0, dual::Bool=false) = new(_normalize_itag(itags), dir, plev, lock, dual)
end

TLIndex(dir::Char, plev::Int=0, lock::Int=0) = TLIndex("", dir, plev, lock)

# Two TLIndex objects are equal if they share the same itags, dir, plev, and dual
# (lock is intentionally ignored — it is a transient contraction counter).
Base.:(==)(a::TLIndex, b::TLIndex) = a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.dual == b.dual
Base.isequal(a::TLIndex, b::TLIndex) = (a == b)
Base.hash(a::TLIndex, h::UInt) = hash((a.itags, a.dir, a.plev, a.dual), h)

to_incoming(idx::TLIndex) = TLIndex(idx.itags, '+', idx.plev, idx.lock, idx.dual)
to_outgoing(idx::TLIndex) = TLIndex(idx.itags, '-', idx.plev, idx.lock, idx.dual)
change_dir(idx::TLIndex)  = TLIndex(idx.itags, idx.dir == '+' ? '-' : '+', idx.plev, idx.lock, idx.dual)
dual(idx::TLIndex) = TLIndex(idx.itags, idx.dir, idx.plev, idx.lock, true)
change_dual(idx::TLIndex) = TLIndex(idx.itags, idx.dir, idx.plev, idx.lock, !idx.dual)

# Format a scalar RMT value as a string with consistent width:
# integers print without decimal point; floats use %#.7g which always
# shows 7 significant digits including trailing zeros (e.g. 3.46410 not 3.4641).
function _fmt_scalar_str(v::Real)
    return @sprintf("%#.7g", v)
end

function _fmt_scalar_str(v::Complex)
    re = _fmt_scalar_str(real(v))
    im = _fmt_scalar_str(abs(imag(v)))
    sign = signbit(imag(v)) ? " - " : " + "
    return string(re, sign, im, "im")
end

function _localspace_cgt_fields(data::Vector{Tuple{NTuple{QD, NTuple{N, Tuple{Vararg{Int}}}}, Array{T, RD}}},
                                symm::NTuple{N, Any},
                                spaces::Tuple{Vararg{AbstractVector, QD}}) where {T, QD, N, RD}

    @assert RD == QD + N; @assert QD == 2 || QD == 3
    QT = qlabeltype(symm)
    qlabels = Vector{NTuple{QD, QT}}(undef, length(data))
    PS = productsymm(symm)
    wmatdata, wmatinfo = _empty_wmat_storage(PS)
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    RMTs = Array{T, RD}[]

    for (sector_index, (sector_qlabels, block0)) in pairs(data)
        block = block0
        sector_wmats = Vector{Matrix{Float64}}(undef, N)
        for i in 1:N
            wmat, block, _ = svd_leg(block, QD + i)
            sector_wmats[i] = wmat
        end
        RMT = block
        qlabels[sector_index] = sector_qlabels
        cgt_metadata = ntuple(N) do i
            qforsymm = Tuple(sector_qlabels[j][i] for j in 1:QD)
            if QD == 2
                cgp = (1, 2); legdir = (1, 1)
            elseif QD == 3
                cgp = qforsymm[2] <= qforsymm[3] ? (1, 2, 3) : (1, 3, 2); legdir = (1, 2)
            else error("Invalid TLArray tensor dimension for getLocalSpace: $QD") end

            if QD == 3 && qforsymm[2] > qforsymm[3]
                qforsymm = (qforsymm[1], qforsymm[3], qforsymm[2])
            end
            (qforsymm, cgp, legdir)
        end

        for i in 1:N
            if length(sector_wmats[i]) == 1 && sector_wmats[i][1] < 0
                sector_wmats[i][:] .*= -1
                RMT[:] .*= -one(T)
            end
        end

        _append_wmat_tuple!(wmatdata, wmatinfo,
                            ntuple(slot -> sector_wmats[nonabelian_indices[slot]],
                                   Val(length(nonabelian_indices))))
        push!(RMTs, RMT)
    end

    return (qlabels = qlabels, wmatdata = wmatdata, wmatinfo = wmatinfo, RMTs = RMTs)
end

# ─── TLArray invariant checkers ─────────────────────────────────────────────

# Condition 1: an index with empty itags must have lock == 0
function _check_empty_tag_lock(inds::NTuple{QD, TLIndex}) where QD
    for idx in inds
        if isempty(idx.itags)
            @assert idx.lock == 0 "TLArray: index with empty itag has nonzero lock=$(idx.lock)"
        end
    end
end

# Condition 2: no two TLIndex objects with non-empty itags in the inds tuple
# may be equal (as determined by ==, which compares itags, dir, plev, dual).
function _check_unique_inds(inds::NTuple{QD, TLIndex}) where QD
    tagged = [idx for idx in inds if !isempty(idx.itags)]
    for i in 1:length(tagged), j in i+1:length(tagged)
        @assert tagged[i] != tagged[j] begin
            "Duplicate TLIndex with non-empty itag in TLArray.inds: $(tagged[i])"
        end
    end
end

@inline _rmt_iszero(rmt::AbstractArray) = iszero(sum(abs2, rmt))
@inline _rmt_iszero(rmt::DiagRMT) = iszero(sum(abs2, rmt.diag))

@inline _check_tlarray_rmt_storage(rmt::AbstractArray, ::Val{QD}) where {QD} = nothing

function _check_tlarray_rmt_storage(rmt::DiagRMT, ::Val{QD}) where {QD}
    rmt.axis[2] == 0 && throw(ArgumentError(
        "axis[2] == 0 DiagRMT is only valid for contraction temporaries, not TLArray sector storage"))
    rmt.axis[2] <= QD || throw(ArgumentError(
        "DiagRMT sector axes must refer to TLArray tensor legs, got axis=$(rmt.axis) for tensor rank $QD"))
    return nothing
end

# T: type of element in the RMT array, can be Float64, ComplexF64, etc.
# QD: The rank of tensor (# of legs), N: The number of symmetries
# RD: The rank of RMT array, which is equal to QD + N
# QT: The qlabel type for one leg sector, inferred from the symmetries
abstract type AbstractTLArray{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{T, RD}} <: AbstractArray{T, QD} end

struct TLArray{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{T, RD}} <:
       AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}
    qlabels::Vector{NTuple{QD, QT}}
    wmatdata::Vector{Float64}
    wmatinfo::Vector{WMatInfo{M}}
    RMTs::Vector{RMT}
    isdefined::BitVector
    iszero::BitVector
    inds::NTuple{QD, TLIndex}
    # Space list for each leg: vector of (qlabels, RMT_dim) pairs
    # Similar to leginfo.splist but precomputed for all legs
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}

    function TLArray(symm::NTuple{N, Any},
        qlabels::Vector{QL},
        wmatdata::Vector{Float64},
        wmatinfo::Vector{WMatInfo{M}},
        RMTs::Vector{RMT},
        inds::NTuple{QD, TLIndex},
        spaces::Tuple{Vararg{AbstractVector, QD}}) where {
            N, QD, QL, M, T, RD, RMT<:AbstractArray{T, RD}}

        RD == QD + N ||
            throw(ArgumentError("RMT rank must equal tensor rank plus number of symmetries"))

        PS = productsymm(symm)
        QT = qlabeltype(symm)
        typed_qlabels = Vector{NTuple{QD, QT}}(qlabels)
        M == n_nonabelian_symmetries(PS) ||
            throw(ArgumentError("w-matrix tuple width must equal the number of non-Abelian symmetries"))
        length(typed_qlabels) == length(RMTs) ||
            throw(ArgumentError("qlabels length must equal number of sector slots"))
        _validate_wmat_storage(wmatdata, wmatinfo, length(RMTs))

        sector_isdefined = falses(length(RMTs))
        sector_iszero = falses(length(RMTs))
        for sector_index in eachindex(RMTs)
            rmt_defined = isassigned(RMTs, sector_index)
            if rmt_defined
                _check_tlarray_rmt_storage(RMTs[sector_index], Val(QD))
                sector_isdefined[sector_index] = true
                sector_iszero[sector_index] = _rmt_iszero(RMTs[sector_index])
            else
                sector_isdefined[sector_index] = false
                sector_iszero[sector_index] = true
            end
        end

        typed_spaces = ntuple(l -> convert(Vector{Tuple{QT, Int}}, spaces[l]), QD)
        q = new{T, QD, N, RD, QT, PS, M, RMT}(
            typed_qlabels, wmatdata, wmatinfo, RMTs, sector_isdefined, sector_iszero, inds, typed_spaces)
        _check_unique_inds(q.inds)
        _check_empty_tag_lock(q.inds)
        _normalize_wmats!(q)
        _orient_wmats!(q)
        return q
    end
end

"""
    TLArrayView(arr, conj, scale, perm)

Lazy view of `arr` with exactly one semantic rule:

    view == maybe_conj(scalar_prod(permute(arr, perm), scale), conj)

`perm[new_leg] = old_leg` is applied first to TLArray physical legs and sector
RMT physical axes. `scale::T` is applied next to each nonzero RMT. `conj`
is applied last: it conjugates RMT values, flips TLIndex directions, and uses
the cached w-matrices constructed for the same permutation/conjugation rule.

`materialize(view)` is not eager conversion. It only delegates to
`materialize(view.arr)` so future lazy source arrays can complete undefined
sector payloads. `arr` must not itself be a `TLArrayView`; composed views must
be merged into one layer.
"""
struct TLArrayView{T, QD, N, RD, QT, PS<:ProductSymm, M, RMT<:AbstractArray{T, RD},
                   A<:AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}} <:
       AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}
    arr::A
    conj::Bool
    scale::T
    perm::NTuple{QD, Int}
    wmatdata::Vector{Float64}
    wmatinfo::Vector{WMatInfo{M}}
end

function _qlabel_vector(qlabels::AbstractMatrix{QT}, ::Val{QD}) where {QT, QD}
    size(qlabels, 1) == QD ||
        throw(ArgumentError("qlabels row count must equal number of TLArray legs"))
    return [ntuple(leg -> qlabels[leg, sector], Val(QD)) for sector in axes(qlabels, 2)]
end

function TLArray(symm::NTuple{N, Any},
                 qlabels::AbstractMatrix{QT},
                 wmatdata::Vector{Float64},
                 wmatinfo::Vector{WMatInfo{M}},
                 RMTs::Vector{RMT},
                 inds::NTuple{QD, TLIndex},
                 spaces::Tuple{Vararg{AbstractVector, QD}}) where {
        N, QD, QT, M, T, RD, RMT<:AbstractArray{T, RD}}
    return TLArray(symm, _qlabel_vector(qlabels, Val(QD)), wmatdata, wmatinfo, RMTs, inds, spaces)
end

_compute_spaces(q::TLArray) = q.spaces

function _normalize_wmats!(q::TLArray{T, QD, N}) where {T, QD, N}
    if QD == 0
        for sector_index in sector_slots(q)
            q.iszero[sector_index] && continue
            for i in 1:N
                isabelian(symm(q)[i]) && continue
                wmat = sector_wmat(q, sector_index, i)
                @assert size(wmat) == (1, 1) "0D TLArray must have 1x1 w-matrices"
                w_val = wmat[1]
                if w_val != 1.0
                    wmat[:] .= 1.0
                    sector_rmt(q, sector_index)[:] .*= w_val
                end
            end
        end
    end
    return q
end

productsymm(::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT} = PS
product_symms(q::AbstractTLArray) = product_symms(productsymm(q))
@inline symm(::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT} =
    product_symms(PS)
nsymms(q::AbstractTLArray) = nsymms(productsymm(q))

Base.propertynames(q::TLArray, private::Bool=false) =
    (:qlabels, :wmatdata, :wmatinfo, :RMTs, :isdefined, :iszero, :inds, :spaces)
Base.propertynames(q::TLArrayView, private::Bool=false) =
    (:arr, :conj, :scale, :perm, :wmatdata, :wmatinfo, :inds, :spaces)

function Base.getproperty(q::TLArrayView, name::Symbol)
    name === :inds && return inds(q)
    name === :spaces && return spaces(q)
    return getfield(q, name)
end

@inline Base.eltype(::Type{<:AbstractTLArray{T}}) where {T} = T
@inline Base.ndims(::Type{<:AbstractTLArray{T, QD}}) where {T, QD} = QD
@inline Base.ndims(q::AbstractTLArray{T, QD}) where {T, QD} = QD
@inline Base.size(q::AbstractTLArray{T, QD}) where {T, QD} =
    ntuple(l -> sum(last, spaces(q)[l]; init=0), Val(QD))
@inline Base.size(q::AbstractTLArray, d::Integer) = d <= ndims(q) ? size(q)[d] : 1
@inline Base.length(q::AbstractTLArray) = sector_count(q)
@inline inds(q::TLArray) = q.inds
@inline spaces(q::TLArray) = q.spaces
@inline sector_count(q::TLArray) = length(q.qlabels)
@inline sector_slots(q::TLArray) = eachindex(q.qlabels)
@inline nsectors(q::TLArray) = count(!, q.iszero)
@inline is_sector_defined(q::TLArray, sector::Int) = q.isdefined[sector]
@inline is_sector_zero(q::TLArray, sector::Int) = q.iszero[sector]
@inline is_sector_active(q::TLArray, sector::Int) = !q.iszero[sector]
@inline sector_qlabel(q::TLArray, sector::Int, leg::Int) = q.qlabels[sector][leg]
@inline sector_qlabel(::Type{QT}, q::TLArray, sector::Int, leg::Int) where {QT} =
    q.qlabels[sector][leg]::QT
@inline function sector_wmat(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, n::Int) where {T, QD, N, RD, QT, PS, M, RMT}
    isabelian(symm(q)[n]) && return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, n))
end
@inline function sector_wmat(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, sector::Int, ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, n}
    is_stored_wmat_symmetry(PS, Val(n)) || return _trivial_wmat()
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, nonabelian_wmat_slot(PS, Val(n)))
end
@inline function sector_wmat_slot(q::TLArray, sector::Int, slot::Int)
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, slot)
end
@inline function sector_wmat_slot(q::TLArrayView, sector::Int, slot::Int)
    return _wmat_from_storage(q.wmatdata, q.wmatinfo, sector, slot)
end
@inline function sector_wmat_slot(q::AbstractTLArray, sector::Int, slot::Int)
    n = nonabelian_symmetry_indices(productsymm(q))[slot]
    return sector_wmat(q, sector, n)
end
@inline function sector_rmt_materialized(q::TLArray, sector::Int)
    q.isdefined[sector] || throw(ArgumentError("sector $sector is not evaluated"))
    return q.RMTs[sector]
end
@inline function sector_rmt_axis_dim(q::TLArray, sector::Int, leg::Int)
    return size(sector_rmt_materialized(q, sector), leg)
end
@inline sector_rmt_with_scale(q::TLArray{T}, sector::Int) where {T} =
    (sector_rmt_materialized(q, sector), one(T))
@inline sector_rmt(q::TLArray, sector::Int) = sector_rmt_materialized(q, sector)

@inline _identity_phys_perm(::Val{QD}) where {QD} = ntuple(identity, Val(QD))

function _normalize_phys_perm(perm, ::Val{QD}) where {QD}
    p = Tuple(Int(i) for i in perm)
    length(p) == QD || throw(ArgumentError("permutation length $(length(p)) != TLArray rank $QD"))
    p = p::NTuple{QD, Int}
    _is_valid_perm(p) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    return p
end

@inline function _is_valid_perm(perm::NTuple{QD, Int}) where {QD}
    seen = ntuple(_ -> false, Val(QD))
    for p in perm
        1 <= p <= QD || return false
        seen = Base.setindex(seen, true, p)
    end
    return all(seen)
end

@inline inds(q::TLArrayView{T, QD}) where {T, QD} =
    ntuple(l -> q.conj ? change_dir(inds(q.arr)[q.perm[l]]) : inds(q.arr)[q.perm[l]], Val(QD))
@inline spaces(q::TLArrayView{T, QD}) where {T, QD} =
    ntuple(l -> spaces(q.arr)[q.perm[l]], Val(QD))
@inline sector_count(q::TLArrayView) = sector_count(q.arr)
@inline sector_slots(q::TLArrayView) = sector_slots(q.arr)
@inline nsectors(q::TLArrayView) = nsectors(q.arr)
@inline is_sector_defined(q::TLArrayView, sector::Int) = is_sector_defined(q.arr, sector)
@inline is_sector_zero(q::TLArrayView, sector::Int) = is_sector_zero(q.arr, sector)
@inline is_sector_active(q::TLArrayView, sector::Int) = is_sector_active(q.arr, sector)
@inline sector_qlabel(q::TLArrayView, sector::Int, leg::Int) =
    sector_qlabel(q.arr, sector, q.perm[leg])
@inline sector_qlabel(::Type{QT}, q::TLArrayView, sector::Int, leg::Int) where {QT} =
    sector_qlabel(QT, q.arr, sector, q.perm[leg])
@inline sector_rmt_axis_dim(q::TLArrayView, sector::Int, leg::Int) =
    sector_rmt_axis_dim(q.arr, sector, q.perm[leg])

function Base.hvcat(rows::Tuple{Vararg{Int}}, qs::AbstractTLArray...)
    nrows = length(rows)
    ncols = rows[1]
    all(==(ncols), rows) || throw(ArgumentError("TLArray matrix literal requires rectangular rows"))
    length(qs) == nrows * ncols || throw(ArgumentError("TLArray matrix literal has inconsistent dimensions"))
    out = Matrix{AbstractTLArray}(undef, nrows, ncols)
    pos = 1
    for i in 1:nrows, j in 1:ncols
        out[i, j] = qs[pos]
        pos += 1
    end
    return out
end

function _conj_sector_wmat(q::AbstractTLArray, sector::Int, n::Int, wmat)
    S = symm(q)[n]
    isabelian(S) && return wmat

    ordered_qlabels, _, legdir = _sector_cgt_metadata(q, sector, n)
    return _conj_sector_wmat_from_metadata(S, ordered_qlabels, legdir, wmat)
end

function _conj_sector_wmat_after_perm(q::AbstractTLArray{T, QD}, sector::Int, n::Int,
                                      wmat, perm::NTuple{QD, Int}) where {T, QD}
    S = symm(q)[n]
    isabelian(S) && return wmat

    ordered_qlabels, legdir = _sector_cgt_metadata_after_perm(q, sector, n, perm)
    return _conj_sector_wmat_from_metadata(S, ordered_qlabels, legdir, wmat)
end

function _sector_cgt_metadata_after_perm(q::AbstractTLArray{T, QD}, sector::Int, n::Int,
                                         perm::NTuple{QD, Int}) where {T, QD}
    qinds = inds(q)
    incoming = Int[l for l in 1:QD if qinds[perm[l]].dir == '+']
    outgoing = Int[l for l in 1:QD if qinds[perm[l]].dir == '-']

    sort!(incoming; by = l -> sector_qlabel(q, sector, perm[l])[n], alg = MergeSort)
    sort!(outgoing; by = l -> sector_qlabel(q, sector, perm[l])[n], alg = MergeSort)

    stored_to_view = (incoming..., outgoing...)
    ordered_qlabels = ntuple(i -> sector_qlabel(q, sector, perm[stored_to_view[i]])[n], Val(QD))
    return ordered_qlabels, (length(incoming), length(outgoing))
end

function _conj_sector_wmat_from_metadata(S, ordered_qlabels, legdir, wmat)
    m, k = legdir
    ins, outs = ordered_qlabels[1:m], ordered_qlabels[m+1:m+k]
    ins_, _ = remove_zeros(S, ins)
    outs_, _ = remove_zeros(S, outs)
    if ins_ == outs_
        conjperm = getNsave_Conjperm(S, ins_)
        return wmat[conjperm.perm, :]
    end
    return wmat
end

function _sector_wmat_after_perm(q::TLArrayView{T, QD}, sector::Int, n::Int) where {T, QD}
    if q.perm == _identity_phys_perm(Val(QD))
        return sector_wmat(q.arr, sector, n)
    end
    return _permute_sector_wmat(q.arr, sector, q.perm, n, symm(q))
end

function _view_sector_wmat_slot(arr::AbstractTLArray{T, QD, N, RD, QT, PS}, sector::Int, ::Val{slot},
                                conj_flag::Bool, scale::T,
                                perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, slot}
    n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
    wmat = if perm == _identity_phys_perm(Val(QD))
        sector_wmat_slot(arr, sector, slot)
    else
        _permute_sector_wmat(arr, sector, perm, Val(n))
    end
    conj_flag || return wmat
    return _conj_sector_wmat_after_perm(arr, sector, n, wmat, perm)
end

function _view_wmat_storage(arr::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                            conj_flag::Bool, scale::T,
                            perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    if !conj_flag && perm == _identity_phys_perm(Val(QD))
        return arr.wmatdata, arr.wmatinfo
    end

    wmatdata = copy(arr.wmatdata)
    wmatinfo = copy(arr.wmatinfo)
    for sector in sector_slots(arr), slot in 1:M
        _wmat_info_present(arr.wmatinfo[sector][slot]) || continue
        _copy_wmat_to_storage!(wmatdata, wmatinfo, sector, slot,
                               _view_sector_wmat_slot(arr, sector, Val(slot),
                                                      conj_flag, scale, perm))
    end
    return wmatdata, wmatinfo
end

function sector_wmat(q::TLArrayView{T, QD, N, RD, QT, PS}, sector::Int, n::Int) where {T, QD, N, RD, QT, PS}
    isabelian(symm(q)[n]) && return _trivial_wmat()
    return sector_wmat_slot(q, sector, nonabelian_wmat_slot(PS, n))
end
function sector_wmat(q::TLArrayView, sector::Int, ::Val{n}) where {n}
    return sector_wmat(q, sector, n)
end

@inline function sector_rmt_with_scale(q::TLArrayView{T}, sector::Int) where {T}
    if q.perm == _identity_phys_perm(Val(ndims(q))) && (!q.conj || T <: Real)
        rmt, alpha = sector_rmt_with_scale(q.arr, sector)
        return rmt, alpha * q.scale
    end
    return sector_rmt_materialized(q, sector), one(T)
end

function sector_rmt_materialized(q::TLArrayView, sector::Int)
    rmt, alpha = sector_rmt_with_scale(q.arr, sector)
    if q.perm != _identity_phys_perm(Val(ndims(q)))
        rmt_perm = (q.perm..., ntuple(n -> ndims(q) + n, Val(nsymms(q)))...)
        rmt = permutedims(rmt, rmt_perm)
    end
    scale = alpha * q.scale
    if scale != one(typeof(scale))
        rmt = rmt * scale
    end
    q.conj && (rmt = conj(rmt))
    return rmt
end
@inline sector_rmt(q::TLArrayView, sector::Int) = sector_rmt_materialized(q, sector)

materialize(q::TLArray) = q

materialize(q::TLArrayView) = materialize(q.arr)

_eager_tlarray(q::TLArray) = copy(q)

function _eager_tlarray(q::TLArrayView)
    materialized_arr = materialize(q.arr)
    result = _eager_tlarray(materialized_arr)
    q.perm == _identity_phys_perm(Val(ndims(q))) || (result = _materialized_permutedims(result, q.perm))
    q.scale == one(typeof(q.scale)) || (result = _materialized_scale(result, q.scale))
    q.conj && (result = _materialized_conj(result))
    return result
end

"""
    to_concrete(q::AbstractTLArray) -> TLArray

Return an eager concrete `TLArray` whose storage has the canonical w-matrix/RMT
sign and orientation conventions. This is storage cleanup, not `q / norm(q)`.

`to_concrete` never mutates `q` or arrays reachable from `q`: it first lets the
internal `materialize` hook complete any future undefined lazy sectors, then
copies the represented tensor data, and finally applies the current
`_normalize_wmats!` and `_orient_wmats!` cleanup to the copy.
"""
function to_concrete(q::AbstractTLArray)
    materialize(q)
    result = _eager_tlarray(q)
    _normalize_wmats!(result)
    _orient_wmats!(result)
    return result
end

canonicalize(q::AbstractTLArray) = to_concrete(q)

function _qlabels_from_accessors(q::AbstractTLArray{T, QD, N, RD, QT}) where {T, QD, N, RD, QT}
    return [ntuple(leg -> sector_qlabel(q, sector, leg), Val(QD))::NTuple{QD, QT}
            for sector in sector_slots(q)]
end

function _zero_metadata_tlarray(q::AbstractTLArray{T, QD, N, RD, QT}, ::Type{RT}) where {T, QD, N, RD, QT, RT}
    qlabels = _qlabels_from_accessors(q)
    M = n_nonabelian_symmetries(productsymm(q))
    wmatdata = Float64[]
    wmatinfo = [_empty_wmat_info(Val(M)) for _ in 1:sector_count(q)]
    RMTs = Vector{Array{RT, RD}}(undef, sector_count(q))
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, inds(q), spaces(q))
end

@inline function _normalize_tlarray_view(arr::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT},
                                         conj_flag::Bool,
                                         scale::T,
                                         perm::NTuple{QD, Int}) where {T, QD, N, RD, QT, PS, M, RMT}
    _is_valid_perm(perm) || throw(ArgumentError("perm must be a valid permutation of 1:$QD"))
    arr isa TLArrayView && throw(ArgumentError("TLArrayView layers must be merged before construction"))
    if !conj_flag && scale == one(T) && perm == _identity_phys_perm(Val(QD))
        return arr
    end
    wmatdata, wmatinfo = _view_wmat_storage(arr, conj_flag, scale, perm)
    return TLArrayView{T, QD, N, RD, QT, PS, M, RMT, typeof(arr)}(arr, conj_flag, scale, perm, wmatdata, wmatinfo)
end

function _view_permutedims(q::AbstractTLArray{T, QD}, perm) where {T, QD}
    p = _normalize_phys_perm(perm, Val(QD))
    return _normalize_tlarray_view(q, false, one(T), p)
end

function _view_permutedims(q::TLArrayView{T, QD}, perm) where {T, QD}
    p = _normalize_phys_perm(perm, Val(QD))
    new_perm = ntuple(l -> q.perm[p[l]], Val(QD))
    return _normalize_tlarray_view(q.arr, q.conj, q.scale, new_perm)
end

function _view_conj(q::AbstractTLArray{T, QD}) where {T, QD}
    return _normalize_tlarray_view(q, true, one(T), _identity_phys_perm(Val(QD)))
end

function _view_conj(q::TLArrayView{T, QD}) where {T, QD}
    return _normalize_tlarray_view(q.arr, !q.conj, q.scale, q.perm)
end

function _view_scale(q::AbstractTLArray{T}, fac::Number) where {T}
    RT = promote_type(T, typeof(fac))
    iszero(fac) && return _zero_metadata_tlarray(q, RT)
    RT === T || return _materialized_scale(_eager_tlarray(q), fac)
    return _normalize_tlarray_view(q, false, convert(T, fac), _identity_phys_perm(Val(ndims(q))))
end

function _view_scale(q::TLArrayView{T}, fac::Number) where {T}
    RT = promote_type(T, typeof(fac))
    iszero(fac) && return _zero_metadata_tlarray(q, RT)
    RT === T || return _materialized_scale(_eager_tlarray(q), fac)
    cfac = convert(T, fac)
    new_scale = q.conj ? q.scale * conj(cfac) : q.scale * cfac
    return _normalize_tlarray_view(q.arr, q.conj, new_scale, q.perm)
end

@inline function _stored_position(stored_to_phys::NTuple{QD, Int}, phys_leg::Int) where {QD}
    @inbounds for stored_pos in 1:QD
        stored_to_phys[stored_pos] == phys_leg && return stored_pos
    end
    throw(BoundsError(stored_to_phys, phys_leg))
end

@inline _phys_to_stored_order(stored_to_phys::NTuple{QD, Int}) where {QD} =
    ntuple(phys_leg -> _stored_position(stored_to_phys, phys_leg), Val(QD))

function _stored_leg_order(q::AbstractTLArray{T, QD, N}, sector::Int, n::Int) where {T, QD, N}
    qinds = inds(q)
    incoming = Int[l for l in 1:QD if qinds[l].dir == '+']
    outgoing = Int[l for l in 1:QD if qinds[l].dir == '-']

    sort!(incoming; by = l -> sector_qlabel(q, sector, l)[n], alg = MergeSort)
    sort!(outgoing; by = l -> sector_qlabel(q, sector, l)[n], alg = MergeSort)

    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD))
end

function _stored_leg_order(qlabels::AbstractVector{<:Tuple},
                           inds::NTuple{QD, TLIndex},
                           sector::Int,
                           n::Int) where {QD}
    incoming = Int[l for l in 1:QD if inds[l].dir == '+']
    outgoing = Int[l for l in 1:QD if inds[l].dir == '-']

    sort!(incoming; by = l -> qlabels[sector][l][n], alg = MergeSort)
    sort!(outgoing; by = l -> qlabels[sector][l][n], alg = MergeSort)

    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD))
end

function _sector_cgt_metadata(q::AbstractTLArray{T, QD, N}, sector::Int, n::Int) where {T, QD, N}
    stored_to_phys = _stored_leg_order(q, sector, n)
    qlabels = ntuple(i -> sector_qlabel(q, sector, stored_to_phys[i])[n], Val(QD))

    qinds = inds(q)
    n_in = count(l -> qinds[l].dir == '+', 1:QD)
    return qlabels, _phys_to_stored_order(stored_to_phys), (n_in, QD - n_in)
end

function _sector_cgt_metadata(qlabels::AbstractVector{<:Tuple},
                           inds::NTuple{QD, TLIndex},
                           sector::Int,
                           n::Int) where {QD}
    stored_to_phys = _stored_leg_order(qlabels, inds, sector, n)
    stored_qlabels = ntuple(i -> qlabels[sector][stored_to_phys[i]][n], Val(QD))

    n_in = count(l -> inds[l].dir == '+', 1:QD)
    return stored_qlabels, _phys_to_stored_order(stored_to_phys), (n_in, QD - n_in)
end

function _sector_label_widths(q::TLArray{T, QD, N}, sector_index::Int) where {T, QD, N}
    map(1:N) do n
        vals = (v for l in 1:QD for v in sector_qlabel(q, sector_index, l)[n])
        mxabs = maximum(abs, vals, init=0)
        needs_sign = symm(q)[n] <: U1
        ndigits(max(mxabs, 1)) + (needs_sign ? 1 : 0)
    end
end

function _print_sector_qlabels(io::IO, q::TLArray{T, QD, N}, sector_index::Int, widths) where {T, QD, N}
    print(io, "[")
    for l in 1:QD
        l > 1 && print(io, " ;")
        for n in 1:N
            print(io, " ")
            for v in sector_qlabel(q, sector_index, l)[n]
                print(io, lpad(v, widths[n]))
            end
        end
    end
    print(io, " ]")
end

function _print_sector_cgt_dims(io::IO, q::TLArray{T, QD, N}, sector_index::Int) where {T, QD, N}
    first = true
    for n in 1:N
        S = symm(q)[n]
        isabelian(S) && continue
        dims = [dimension(S, sector_qlabel(q, sector_index, l)[n]) for l in 1:QD]
        first && print(io, "| ")
        first = false
        print(io, join(dims, "x"), "\t")
    end
end

function _sector_cgt_size_2d(q::TLArray{T, 2, N}, sector_index::Int) where {T, N}
    total = 1
    for n in 1:N
        S = symm(q)[n]
        isabelian(S) && continue
        total *= dimension(S, sector_qlabel(q, sector_index, 1)[n])
    end
    return total
end

_tlarray_fields(q::TLArray) = (qlabels = q.qlabels, wmatdata = q.wmatdata, wmatinfo = q.wmatinfo, RMTs = q.RMTs)

Base.ndims(q::TLArray{T, QD}) where {T, QD} = QD

function _sector_cgt_size_2d(symm::NTuple{N, Any},
                             qlabels::Vector{NTuple{QD, QT}},
                             sector_index::Int) where {N, QD, QT}
    total = 1
    for n in 1:N
        S = symm[n]
        isabelian(S) && continue
        total *= dimension(S, qlabels[sector_index][1][n])
    end
    return total
end

function _orient_wmats!(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        for m in 1:M
            wmat = sector_wmat_slot(q, sector_index, m)
            if length(wmat) == 1 && wmat[1] < 0
                wmat[:] .*= -1
                if sector_rmt(q, sector_index) isa DiagRMT
                    q.RMTs[sector_index] = -sector_rmt(q, sector_index)
                else
                    sector_rmt(q, sector_index)[:] .*= -one(T)
                end
            end
        end
    end
    return q
end

function _display_scalar_rmt(q::TLArray, sector_index::Int)
    return only(sector_rmt(q, sector_index)) * _scalar_wmat_product(q, sector_index)
end

function _display_scalar_rmt(q::TLArray{T, 2}, sector_index::Int) where {T}
    return only(sector_rmt(q, sector_index)) * _scalar_wmat_product(q, sector_index) /
           sqrt(_sector_cgt_size_2d(q, sector_index))
end

function _copy_assigned_vector(v::Vector, inds; deep::Bool=false)
    out = similar(v, length(inds))
    for (out_index, in_index) in pairs(collect(inds))
        if isassigned(v, in_index)
            out[out_index] = deep ? deepcopy(v[in_index]) : v[in_index]
        end
    end
    return out
end

@inline _sector_has_wmat_info(q::AbstractTLArray{T, QD, N, RD, QT, PS, M}, sector::Int) where {T, QD, N, RD, QT, PS, M} =
    any(slot -> _wmat_info_present(q.wmatinfo[sector][slot]), 1:M)

_copy_wmat_storage(q::AbstractTLArray; deep::Bool=false) =
    (deep ? copy(q.wmatdata) : q.wmatdata, deep ? copy(q.wmatinfo) : q.wmatinfo)

function _copy_wmat_storage(q::AbstractTLArray{T, QD, N, RD, QT, PS, M},
                            inds; deep::Bool=false) where {T, QD, N, RD, QT, PS, M}
    collected = collect(inds)
    total = 0
    for sector in collected, slot in 1:M
        offset, nrow, ncol = q.wmatinfo[sector][slot]
        offset == 0 && continue
        total += nrow * ncol
    end

    wmatdata = Vector{Float64}(undef, total)
    wmatinfo = Vector{WMatInfo{M}}(undef, length(collected))
    next_offset = 1
    for (out_index, sector) in pairs(collected)
        wmatinfo[out_index] = ntuple(Val(M)) do slot
            offset, nrow, ncol = q.wmatinfo[sector][slot]
            offset == 0 && return (0, 0, 0)
            len = nrow * ncol
            copyto!(view(wmatdata, next_offset:next_offset + len - 1),
                    view(q.wmatdata, offset:offset + len - 1))
            new_offset = next_offset
            next_offset += len
            (new_offset, nrow, ncol)
        end
    end
    return wmatdata, wmatinfo
end

_copy_sector_RMTs(q::TLArray; deep::Bool=false) =
    _copy_assigned_vector(q.RMTs, eachindex(q.RMTs); deep=deep)
_copy_sector_RMTs(q::TLArray, inds; deep::Bool=false) =
    _copy_assigned_vector(q.RMTs, inds; deep=deep)

# Construct a TLArray with the same sectors but with itags replaced.
# itags: Tuple{Vararg{AbstractString, QD}} — one tag per leg; all other TLIndex
# fields are preserved.
function TLArray(q::TLArray{T, QD, N, RD}, itags::Tuple{Vararg{AbstractString, QD}}) where {T, QD, N, RD}
    new_inds = ntuple(l -> TLIndex(itags[l], q.inds[l].dir, q.inds[l].plev,
                                  q.inds[l].lock, q.inds[l].dual), QD)
    return TLArray(symm(q), copy(q.qlabels), _copy_wmat_storage(q; deep=true)...,
                          _copy_sector_RMTs(q; deep=true),
                          new_inds, _copy_spaces_tuple(q.spaces))
end

# Construct a TLArray with the same sectors but with all TLIndex fields replaced.
# inds: NTuple{QD, TLIndex} — one full TLIndex per leg.
# Arrow directions must match the original TLArray (only itags/lock/plev/dual may differ).
function TLArray(q::TLArray{T, QD, N, RD}, inds::NTuple{QD, TLIndex}) where {T, QD, N, RD}
    @assert ntuple(l -> inds[l].dir, QD) == ntuple(l -> q.inds[l].dir, QD) "TLArray(q, inds): arrow directions must match the original TLArray on all legs"
    return TLArray(symm(q), copy(q.qlabels), _copy_wmat_storage(q; deep=true)...,
                          _copy_sector_RMTs(q; deep=true),
                          inds, _copy_spaces_tuple(q.spaces))
end

TLArray(q::TLArrayView{T, QD, N, RD}, inds::NTuple{QD, TLIndex}) where {T, QD, N, RD} =
    TLArray(_eager_tlarray(q), inds)

TLArray(q::TLArrayView{T, QD, N, RD}, itags::Tuple{Vararg{AbstractString, QD}}) where {T, QD, N, RD} =
    TLArray(_eager_tlarray(q), itags)

Base.convert(::Type{TLArray}, q::TLArrayView) = _eager_tlarray(q)

Base.getindex(q::TLArray, i::Int) = TLArray(q, i)
Base.firstindex(q::AbstractTLArray) = 1
Base.lastindex(q::AbstractTLArray) = sector_count(q)

function _normalize_tlarray_sector_index(i::Int, nsectors::Int)
    i == 0 && throw(BoundsError(1:nsectors, i))
    idx = i < 0 ? nsectors + i + 1 : i
    1 <= idx <= nsectors || throw(BoundsError(1:nsectors, i))
    return idx
end

function _normalize_tlarray_sector_selector(selector, nsectors::Int)
    if selector isa Colon
        return collect(1:nsectors)
    elseif selector isa Integer
        return Int[_normalize_tlarray_sector_index(Int(selector), nsectors)]
    elseif selector isa AbstractVector{Bool}
        length(selector) == nsectors || throw(DimensionMismatch(
            "sector selector of length $(length(selector)) does not match number of sectors $nsectors"))
        return findall(selector)
    elseif selector isa AbstractRange{<:Integer}
        return _normalize_tlarray_sector_selector(collect(selector), nsectors)
    elseif selector isa AbstractVector{<:Integer}
        inds = Int[_normalize_tlarray_sector_index(Int(i), nsectors) for i in selector]
        length(unique(inds)) == length(inds) || throw(ArgumentError(
            "sector selector must not contain duplicate indices"))
        return inds
    else
        throw(ArgumentError(
            "sector selector must be :, Int, AbstractRange{<:Integer}, AbstractVector{<:Integer}, or AbstractVector{Bool}"))
    end
end

"""
    TLArray(q::TLArray, selector) -> TLArray

Create a new `TLArray` from a subset of sectors, preserving the original
symmetry tuple, leg indices, and cached leg-space metadata in `q.spaces`.

`selector` may be `:`, an `Int`, an integer range, an integer vector, or a
boolean mask. Negative integer indices count from the end.
"""
function TLArray(q::TLArray{T, QD, N, RD}, selector) where {T, QD, N, RD}
    inds = _normalize_tlarray_sector_selector(selector, sector_count(q))
    qlabels = copy(q.qlabels[inds])
    wmatdata, wmatinfo = _copy_wmat_storage(q, inds; deep=true)
    RMTs = _copy_sector_RMTs(q, inds; deep=true)
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, q.inds, _copy_spaces_tuple(q.spaces))
end

Base.getindex(q::TLArray,
              selector::Union{Colon, AbstractRange{<:Integer},
                              AbstractVector{<:Integer}, AbstractVector{Bool}}) = TLArray(q, selector)
Base.getindex(q::TLArrayView, args...) = getindex(_eager_tlarray(q), args...)

function _scalar_wmat_product(q::AbstractTLArray{T, QD, N, RD, QT, PS, M}, sector_index::Int) where {T, QD, N, RD, QT, PS, M}
    factor = 1.0
    for slot in 1:M
        wmat = sector_wmat_slot(q, sector_index, slot)
        length(wmat) == 1 && (factor *= only(wmat))
    end
    return factor
end

# For 0-dimensional TLArray (scalar), q[] returns the unique RMT element.
function Base.getindex(q::TLArray{T, 0, N, N}) where {T, N}
    @assert nsectors(q) <= 1 "0D TLArray must have zero or one sector"
    if nsectors(q) == 1
        sector_index = findfirst(!, q.iszero)
        @assert length(sector_rmt(q, sector_index)) == 1 "0D TLArray RMT must be a scalar"
        return only(sector_rmt(q, sector_index)) * _scalar_wmat_product(q, sector_index)
    else return zero(T) end
end

function Base.getindex(q::TLArrayView{T, 0, N, N}) where {T, N}
    @assert nsectors(q) <= 1 "0D TLArrayView must have zero or one sector"
    if nsectors(q) == 1
        sector_index = first(sector for sector in sector_slots(q) if is_sector_active(q, sector))
        rmt = sector_rmt_materialized(q, sector_index)
        @assert length(rmt) == 1 "0D TLArrayView RMT must be a scalar"
        return only(rmt) * _scalar_wmat_product(q, sector_index)
    else return zero(T) end
end

# ─── Leg selection utilities ─────────────────────────────────────────────────

"""
    findlegs(q::TLArray, pred::Function) -> Vector{Int}

Find all leg indices where `pred(qindex)` returns true.

# Examples
```julia
findlegs(q, idx -> idx.dir == '-')                    # all outgoing legs
findlegs(q, idx -> occursin("site", idx.itags))       # legs with "site" in tag
findlegs(q, idx -> idx.dir == '-' && idx.plev == 0)   # outgoing, unprimed
```
"""
findlegs(q::AbstractTLArray{T, QD}, pred::Function) where {T, QD} =
    [i for i in 1:QD if pred(inds(q)[i])]

"""
    findleg(q::TLArray, pred::Function) -> Int

Find the first leg index where `pred(qindex)` returns true.
Throws an error if no leg matches.

# Examples
```julia
findleg(q, idx -> idx.itags == "bond")   # first leg with exact tag "bond"
findleg(q, idx -> idx.dir == '+')        # first incoming leg
```
"""
function findleg(q::AbstractTLArray{T, QD}, pred::Function) where {T, QD}
    qinds = inds(q)
    for i in 1:QD pred(qinds[i]) && return i end
    return nothing
end

# Internal: check if a TLIndex matches all specified criteria
function _matches_criteria(idx::TLIndex; dir=nothing, itag=nothing, plev=nothing, lock=nothing)
    (!isnothing(dir)  && idx.dir != dir)                                 && return false
    (!isnothing(itag) && !_matches_itag_selector(idx.itags, itag))       && return false
    (!isnothing(plev) && idx.plev != plev)                               && return false
    (!isnothing(lock) && idx.lock != lock)                               && return false
    return true
end

"""
    findlegs(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Find all leg indices matching the specified criteria. Unspecified criteria match any value.
If `rev=true`, the selection is reversed: legs that do *not* match are returned.

# Arguments
- `dir`: Match direction ('+' for incoming, '-' for outgoing)
- `itag`: Match exact tags. A single string like `"a,b"` means both `a` and `b`
  must be present; a tuple/vector of strings means any one of those tag-sets may match
- `plev`: Match exact prime level
- `lock`: Match exact lock level
- `rev`: If `true`, return legs that do *not* satisfy the criteria (default `false`)

# Examples
```julia
findlegs(q; dir='-')                    # all outgoing legs
findlegs(q; itag="site")                # legs carrying tag "site"
findlegs(q; itag=("a,b", "a,c"))        # legs with tags a+b, or a+c
findlegs(q; dir='+', plev=0)            # incoming, unprimed legs
findlegs(q; lock=0)                     # non-locked legs
findlegs(q; dir='-', rev=true)          # all legs that are NOT outgoing
```
"""
function findlegs(q::AbstractTLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    qinds = inds(q)
    return [i for i in 1:QD if _matches_criteria(qinds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev]
end

"""
    findleg(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Find the first leg index matching the specified criteria.
Returns `nothing` if no leg matches.
If `rev=true`, returns the first leg that does *not* match the criteria.

# Examples
```julia
findleg(q; itag="bond")           # first leg with "bond" in tag
findleg(q; dir='+')               # first incoming leg
findleg(q; lock=0)                # first non-locked leg
findleg(q; dir='+', rev=true)     # first leg that is NOT incoming
```
"""
function findleg(q::AbstractTLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    qinds = inds(q)
    for i in 1:QD
        _matches_criteria(qinds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

function _matching_targets(q::AbstractTLArray; require_unlocked::Bool=false)
    return Set(change_dir(idx) for idx in inds(q)
               if !require_unlocked || idx.lock == 0)
end

_has_target_match(idx::TLIndex, targets; require_unlocked::Bool=false) =
    (!require_unlocked || idx.lock == 0) && (idx in targets)

function _find_matching_legs(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                             dir=nothing, itag=nothing, plev=nothing,
                             lock=nothing, rev::Bool=false,
                             matched::Bool=true,
                             require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    ainds = inds(a)
    return [i for i in 1:QD
            if (_has_target_match(ainds[i], targets; require_unlocked=require_unlocked) == matched) &&
               (_matches_criteria(ainds[i]; dir=dir, itag=itag,
                                  plev=plev, lock=lock) ⊻ rev)]
end

function _find_matching_leg(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                            dir=nothing, itag=nothing, plev=nothing,
                            lock=nothing, rev::Bool=false,
                            matched::Bool=true,
                            require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    ainds = inds(a)
    for i in 1:QD
        (_has_target_match(ainds[i], targets; require_unlocked=require_unlocked) == matched) || continue
        _matches_criteria(ainds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

"""
    matchings(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one matching leg in `b`.

A leg matches when it has the same `itags`, `plev`, and `dual` flag as a leg
of `b`, but with opposite direction. Lock is ignored for the cross-tensor
match test. Keyword arguments filter the returned legs of `a` using the same
rules as `findlegs`.
"""
function matchings(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                   dir=nothing, itag=nothing, plev=nothing,
                   lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true)
end

"""
    matching(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a matching leg in `b`, or `nothing`
if no such leg exists.

Matching ignores lock between tensors; keyword arguments filter the returned
leg of `a` using the same rules as `findleg`.
"""
function matching(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                  dir=nothing, itag=nothing, plev=nothing,
                  lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true)
end

"""
    unmatchings(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any matching leg in `b`.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned legs of `a` using the same rules as `findlegs`.
"""
function unmatchings(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                     dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false)
end

"""
    unmatching(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any matching leg in `b`,
or `nothing` if every leg is matched.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned leg of `a` using the same rules as `findleg`.
"""
function unmatching(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                    dir=nothing, itag=nothing, plev=nothing,
                    lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false)
end

"""
    contractables(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one contractable leg in `b`.

Two legs are contractable when they satisfy the same cross-tensor match rule as
`matchings` and both legs have `lock == 0`. Keyword arguments filter the
returned legs of `a` using the same rules as `findlegs`.
"""
function contractables(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                       dir=nothing, itag=nothing, plev=nothing,
                       lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true, require_unlocked=true)
end

"""
    contractable(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a contractable leg in `b`, or
`nothing` if no such leg exists.

Contractable legs satisfy the same cross-tensor match rule as `matchings`, but
both tensors must have `lock == 0` on the matched legs.
"""
function contractable(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                      dir=nothing, itag=nothing, plev=nothing,
                      lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true, require_unlocked=true)
end

"""
    uncontractables(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any contractable leg in `b`.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs. Keyword arguments filter the returned legs of
`a` using the same rules as `findlegs`.
"""
function uncontractables(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                         dir=nothing, itag=nothing, plev=nothing,
                         lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false, require_unlocked=true)
end

"""
    uncontractable(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any contractable leg in
`b`, or `nothing` if every eligible leg is contractable.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs.
"""
function uncontractable(a::AbstractTLArray{T, QD}, b::AbstractTLArray;
                        dir=nothing, itag=nothing, plev=nothing,
                        lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false, require_unlocked=true)
end

# ─── Lock utilities ──────────────────────────────────────────────────────────
#
# Lock level semantics:
#   lock > 0  : Cannot be contracted. Decreases by 1 after each contraction.
#   lock == 0 : Normal leg, can be contracted.
#   lock == -1: Permanently locked, never changes and cannot be contracted.
#
# ─────────────────────────────────────────────────────────────────────────────

const LegList = Union{AbstractVector{<:Integer}, Tuple{Vararg{Integer}}}

# Internal: apply a lock modification function to selected leg indices.
# legs can be any iterable of integers (Int, Vector, UnitRange, Tuple, etc.)
function _modify_lock(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_lock = modify_fn(idx.lock)
        new_inds[i] = TLIndex(idx.itags, idx.dir, idx.plev, new_lock, idx.dual)
    end
    return TLArray(symm(q), copy(q.qlabels), _copy_wmat_storage(q; deep=true)...,
                          _copy_sector_RMTs(q; deep=true),
                          Tuple(new_inds), _copy_spaces_tuple(q.spaces))
end

# Lock increase function (respects permanent lock)
_lock_inc(current_lock, inc) = current_lock == -1 ? -1 : current_lock + inc

"""
    lock(q::TLArray, leg::Integer; inc::Int=1)

Increase lock level of a single specified leg by `inc` (default 1).
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::TLArray, leg::Integer; inc::Int=1)
    return _modify_lock(q, (leg,), lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray, legs::LegList; inc::Int=1)

Increase lock level of the specified legs by `inc` (default 1).
`legs` can be any vector, range, or tuple of integers, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::TLArray, legs::LegList; inc::Int=1)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray, pred::Function; inc::Int=1)

Increase lock level of legs satisfying predicate by `inc` (default 1).
"""
function Base.lock(q::TLArray, pred::Function; inc::Int=1)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Increase lock level of legs matching criteria by `inc`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lock(q; dir='-')                  # lock all outgoing legs by 1
lock(q; inc=2, itag="bond")       # lock legs with "bond" in tag by 2
lock(q; lock=0)                   # lock all currently-unlocked legs by 1
lock(q; dir='-', rev=true)        # lock all legs that are NOT outgoing
```
"""
function Base.lock(q::TLArray; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lockp(q::TLArray, leg::Integer)

Permanently lock a single specified leg (set lock=-1).
"""
function lockp(q::TLArray, leg::Integer)
    return _modify_lock(q, (leg,), _ -> -1)
end

"""
    lockp(q::TLArray, legs::LegList)

Permanently lock the specified legs (set lock=-1).
`legs` can be any vector, range, or tuple of integers.
"""
function lockp(q::TLArray, legs::LegList)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::TLArray, pred::Function)

Permanently lock legs satisfying predicate.
"""
function lockp(q::TLArray, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Permanently lock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lockp(q; itag="phys")          # permanently lock physical legs
lockp(q; lock=0)               # permanently lock all currently-unlocked legs
lockp(q; itag="phys", rev=true)   # permanently lock all legs except "phys"
```
"""
function lockp(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    unlock(q::TLArray, leg::Integer)

Unlock a single specified leg (set lock=0). Also removes permanent lock.
"""
function Base.unlock(q::TLArray, leg::Integer)
    return _modify_lock(q, (leg,), _ -> 0)
end

"""
    unlock(q::TLArray, legs::LegList)

Unlock the specified legs (set lock=0).
`legs` can be any vector, range, or tuple of integers.
"""
function Base.unlock(q::TLArray, legs::LegList)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::TLArray, pred::Function)

Unlock legs satisfying predicate.
"""
function Base.unlock(q::TLArray, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Unlock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
unlock(q; dir='-')             # unlock all outgoing legs
unlock(q; lock=1)              # unlock all legs currently at lock=1
unlock(q; dir='-', rev=true)   # unlock all legs that are NOT outgoing
```
"""
function Base.unlock(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> 0)
end

# ─── Prime level modification ────────────────────────────────────────────────
#
# ITensor-style prime level modification. Prime level is always non-negative.
# All increment forms use `inc` as an optional keyword argument (default 1).
#
#   prime(q; inc=1, kw...)         – increase plev of selected legs by inc
#   prime(q, leg; inc=1)           – increase plev of a single leg by inc
#   prime(q, legs; inc=1)          – increase plev of specified legs by inc
#   prime(q, pred; inc=1)          – increase plev of legs matching predicate
#   setprime(q, n; kw...)          – set plev of selected legs to n (≥0)
#   setprime(q, legs, n)           – set plev of specified legs to n
#   noprime(q; kw...)              – set plev of selected legs to 0
#   noprime(q, leg)                – set plev of a single leg to 0
#   noprime(q, legs)               – set plev of specified legs to 0
#   noprime(q, pred)               – set plev of legs matching predicate to 0
#
# Keyword selectors for criteria forms (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

# legs can be any iterable of integers
function _modify_plev(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(idx.itags, idx.dir, modify_fn(idx.plev), idx.lock, idx.dual)
    end
    return TLArray(symm(q), copy(q.qlabels), _copy_wmat_storage(q; deep=true)...,
                          _copy_sector_RMTs(q; deep=true),
                          Tuple(new_inds), _copy_spaces_tuple(q.spaces))
end

"""
    prime(q::TLArray; inc::Int=1, dir, itag, plev, lock, rev=false)

Increase the prime level of matching legs by `inc` (default 1).
Prime level is clamped to 0 from below.
With no keyword arguments, all legs are affected.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
prime(q)                        # all legs: plev += 1
prime(q; inc=2)                 # all legs: plev += 2
prime(q; inc=-1)                # all legs: plev -= 1, clamped to 0
prime(q; dir='+')               # incoming legs only
prime(q; itag="site")           # legs whose tag contains "site"
prime(q; plev=0)                # only currently unprimed legs
prime(q; dir='+', rev=true)     # all legs that are NOT incoming
```
"""
function prime(q::TLArray{T, QD}; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::TLArray, leg::Integer; inc::Int=1)

Increase the prime level of a single specified leg by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, 2)                     # leg 2: plev += 1
prime(q, 2; inc=3)              # leg 2: plev += 3
```
"""
function prime(q::TLArray{T, QD}, leg::Integer; inc::Int=1) where {T, QD}
    return _modify_plev(q, (leg,), p -> max(0, p + inc))
end

"""
    prime(q::TLArray, legs::LegList; inc::Int=1)

Increase the prime level of the specified legs by `inc` (default 1).
Prime level is clamped to 0 from below.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.

# Examples
```julia
prime(q, [1, 3])                # legs 1 and 3: plev += 1
prime(q, 1:2; inc=2)            # legs 1–2: plev += 2
prime(q, (1, 3))                # same as prime(q, [1, 3])
```
"""
function prime(q::TLArray{T, QD}, legs::LegList; inc::Int=1) where {T, QD}
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::TLArray, pred::Function; inc::Int=1)

Increase the prime level of legs satisfying predicate by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, idx -> idx.dir == '+')          # incoming legs: plev += 1
prime(q, idx -> idx.plev == 0; inc=2)   # unprimed legs: plev += 2
```
"""
function prime(q::TLArray{T, QD}, pred::Function; inc::Int=1) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    setprime(q::TLArray, n::Int; dir, itag, plev, lock, rev=false)

Set the prime level of matching legs to `n`. `n` must be non-negative.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setprime(q, 2)                  # set all legs to plev=2
setprime(q, 1; dir='-')         # set outgoing legs to plev=1
setprime(q, 0; itag="bond")     # same as noprime(q; itag="bond")
setprime(q, 0; dir='-', rev=true)  # clear prime on all non-outgoing legs
```
"""
function setprime(q::TLArray{T, QD}, n::Int; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> n)
end

"""
    setprime(q::TLArray, legs, n::Int)

Set the prime level of the specified legs to `n`. `n` must be non-negative.
`legs` can be any iterable of leg indices.

# Examples
```julia
setprime(q, [1, 3], 2)          # set legs 1 and 3 to plev=2
setprime(q, 1:2, 0)             # same as noprime(q, 1:2)
```
"""
function setprime(q::TLArray{T, QD}, legs, n::Int) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, legs, _ -> n)
end

"""
    noprime(q::TLArray; dir, itag, plev, lock, rev=false)

Set the prime level of matching legs to 0.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
noprime(q)                      # clear prime on all legs
noprime(q; dir='+')             # clear prime on incoming legs only
noprime(q; plev=1)              # clear all legs currently at plev=1
noprime(q; dir='+', rev=true)   # clear prime on all non-incoming legs
```
"""
function noprime(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::TLArray, leg::Integer)

Set the prime level of a single specified leg to 0.
"""
function noprime(q::TLArray{T, QD}, leg::Integer) where {T, QD}
    return _modify_plev(q, (leg,), _ -> 0)
end

"""
    noprime(q::TLArray, legs::LegList)

Set the prime level of the specified legs to 0.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
"""
function noprime(q::TLArray{T, QD}, legs::LegList) where {T, QD}
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::TLArray, pred::Function)

Set the prime level of legs satisfying predicate to 0.
"""
function noprime(q::TLArray{T, QD}, pred::Function) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, _ -> 0)
end

# ─── Tag modification ────────────────────────────────────────────────────────
#
# ITensor-style tag manipulation on TLArray legs.
# Tags are stored as sorted, comma-separated strings (e.g. "bond,site").
#
#   additag(q, newtags; kw...)     – add tag(s) to matching legs
#   removeitag(q, tags; kw...)     – remove tag(s) from matching legs
#   replaceitag(q, old=>new...)    – replace tag queries on matching legs
#   setitag(q, tags; kw...)        – replace entire tag string of matching legs
#
# Keyword selectors (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

const ITagQuerySpec = Union{AbstractString,
                            Tuple{Vararg{AbstractString}},
                            AbstractVector{<:AbstractString}}
const ITagReplacementPair = Pair{<:AbstractString, <:AbstractString}
const ITagReplacementDict = AbstractDict{<:AbstractString, <:AbstractString}

function _modify_itag(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(modify_fn(idx.itags), idx.dir, idx.plev, idx.lock, idx.dual)
    end
    return TLArray(symm(q), copy(q.qlabels), _copy_wmat_storage(q; deep=true)...,
                          _copy_sector_RMTs(q; deep=true),
                          Tuple(new_inds), _copy_spaces_tuple(q.spaces))
end

"""
    additag(q::TLArray, newtags; dir, itag, plev, lock, rev=false)

Add one or more tags to matching legs. `newtags` may be a comma-separated
string of tags (e.g. `"bond,u1"`). The result is always sorted.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
additag(q, "site")              # add "site" to all legs
additag(q, "phys"; dir='+')     # add "phys" to incoming legs only
additag(q, "u1"; itag="bond")   # add "u1" to legs that already have "bond"
additag(q, "aux"; dir='+', rev=true)  # add "aux" to all non-incoming legs
```
"""
function additag(q::TLArray{T, QD}, newtags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, leg::Integer, newtags)

Add tags to a single specified leg.
"""
function additag(q::TLArray{T, QD}, leg::Integer, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, (leg,), base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, legs::LegList, newtags)

Add tags to the specified legs. `legs` can be any vector, range, or tuple.
"""
function additag(q::TLArray{T, QD}, legs::LegList, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, pred::Function, newtags)

Add tags to legs satisfying predicate.
"""
function additag(q::TLArray{T, QD}, pred::Function, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _add_itag(base, newtags))
end

"""
    removeitag(q::TLArray, tags; dir, itag, plev, lock, rev=false)

Remove one or more tags from matching legs.

`tags` may be either a single comma-separated tag string or a tuple/vector of
such strings. For tuple/vector input, each query is applied independently and
its tags are removed only from legs that contain all tags in that query.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
removeitag(q, "site")                     # remove "site" from all legs
removeitag(q, ("aaa,bbb", "ccc"))       # remove aaa+bbb together, and ccc independently
removeitag(q, "phys"; dir='-')           # remove "phys" from outgoing legs
removeitag(q, "aux"; dir='-', rev=true)  # remove "aux" from all non-outgoing legs
```
"""
function removeitag(q::TLArray{T, QD}, tags::ITagQuerySpec; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, leg::Integer, tags)

Remove tags from a single specified leg.
"""
function removeitag(q::TLArray{T, QD}, leg::Integer, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, (leg,), base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, legs::LegList, tags)

Remove tags from the specified legs. `legs` can be any vector, range, or tuple.
"""
function removeitag(q::TLArray{T, QD}, legs::LegList, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, pred::Function, tags)

Remove tags from legs satisfying predicate.
"""
function removeitag(q::TLArray{T, QD}, pred::Function, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _remove_itag(base, tags))
end

"""
    replaceitag(q::TLArray, replacements::Pair...; dir, itag, plev, lock, rev=false)
    replaceitag(q::TLArray, replacements::AbstractDict; dir, itag, plev, lock, rev=false)

Replace tags on matching legs. Each replacement removes the tags in the source
query and adds the destination tags when the leg contains all tags in the source
query. For example, `"aaa,bbb"=>"ccc"` removes both `aaa` and `bbb` and adds
`ccc` only on legs containing both source tags.

Dictionary input applies each `key => value` replacement in iteration order.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
replaceitag(q, "site"=>"phys")
replaceitag(q, "aaa,bbb"=>"ccc")
replaceitag(q, "site1"=>"left", "site2"=>"right"; dir='-')
replaceitag(q, Dict("site"=>"phys"))
```
"""
function replaceitag(q::TLArray{T, QD}, replacements::ITagReplacementPair...;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, replacements::ITagReplacementDict;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, leg::Integer, replacements)

Replace tags on a single specified leg.
"""
function replaceitag(q::TLArray{T, QD}, leg::Integer,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, leg::Integer,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, legs::LegList, replacements)

Replace tags on the specified legs. `legs` can be any vector, range, or tuple.
"""
function replaceitag(q::TLArray{T, QD}, legs::LegList,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, legs::LegList,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, pred::Function, replacements)

Replace tags on legs satisfying predicate.
"""
function replaceitag(q::TLArray{T, QD}, pred::Function,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, pred::Function,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end

"""
    setitag(q::TLArray, tags; dir, itag, plev, lock, rev=false)

Replace the entire tag string of matching legs with `tags`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setitag(q, "bond")              # set all legs to tag "bond"
setitag(q, ""; dir='+')         # clear tags on incoming legs
setitag(q, "phys"; itag="site") # rename "site" → "phys" (full replacement)
setitag(q, "aux"; dir='+', rev=true)  # set tag on all non-incoming legs
```
"""
function setitag(q::TLArray{T, QD}, tags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::TLArray, leg::Integer, tags)

Set the entire tag string of a single specified leg.
"""
function setitag(q::TLArray{T, QD}, leg::Integer, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, (leg,), _ -> norm)
end

"""    setitag(q::TLArray, legs::LegList, tags)

Set the entire tag string of the specified legs. `legs` can be any vector, range, or tuple.
"""
function setitag(q::TLArray{T, QD}, legs::LegList, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::TLArray, pred::Function, tags)

Set the entire tag string of legs satisfying predicate.
"""
function setitag(q::TLArray{T, QD}, pred::Function, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, findlegs(q, pred), _ -> norm)
end

_view_arr_leg(q::TLArrayView, leg::Integer) = q.perm[Int(leg)]
_view_arr_legs(q::TLArrayView, legs) = Int[_view_arr_leg(q, leg) for leg in legs]
_view_arr_legs(q::TLArrayView, leg::Integer) = (_view_arr_leg(q, leg),)
_view_rewrap(q::TLArrayView, arr::AbstractTLArray) =
    _normalize_tlarray_view(arr, q.conj, q.scale, q.perm)

function Base.lock(q::TLArrayView, leg::Integer; inc::Int=1)
    return _view_rewrap(q, lock(q.arr, _view_arr_leg(q, leg); inc=inc))
end
function Base.lock(q::TLArrayView, legs::LegList; inc::Int=1)
    return _view_rewrap(q, lock(q.arr, _view_arr_legs(q, legs); inc=inc))
end
function Base.lock(q::TLArrayView, pred::Function; inc::Int=1)
    return _view_rewrap(q, lock(q.arr, _view_arr_legs(q, findlegs(q, pred)); inc=inc))
end
function Base.lock(q::TLArrayView; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, Base.lock(q.arr, _view_arr_legs(q, legs); inc=inc))
end

function lockp(q::TLArrayView, leg::Integer)
    return _view_rewrap(q, lockp(q.arr, _view_arr_leg(q, leg)))
end
function lockp(q::TLArrayView, legs::LegList)
    return _view_rewrap(q, lockp(q.arr, _view_arr_legs(q, legs)))
end
function lockp(q::TLArrayView, pred::Function)
    return _view_rewrap(q, lockp(q.arr, _view_arr_legs(q, findlegs(q, pred))))
end
function lockp(q::TLArrayView; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, lockp(q.arr, _view_arr_legs(q, legs)))
end

function Base.unlock(q::TLArrayView, leg::Integer)
    return _view_rewrap(q, unlock(q.arr, _view_arr_leg(q, leg)))
end
function Base.unlock(q::TLArrayView, legs::LegList)
    return _view_rewrap(q, unlock(q.arr, _view_arr_legs(q, legs)))
end
function Base.unlock(q::TLArrayView, pred::Function)
    return _view_rewrap(q, unlock(q.arr, _view_arr_legs(q, findlegs(q, pred))))
end
function Base.unlock(q::TLArrayView; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, unlock(q.arr, _view_arr_legs(q, legs)))
end

function prime(q::TLArrayView; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, prime(q.arr, _view_arr_legs(q, legs); inc=inc))
end
function prime(q::TLArrayView, leg::Integer; inc::Int=1)
    return _view_rewrap(q, prime(q.arr, _view_arr_leg(q, leg); inc=inc))
end
function prime(q::TLArrayView, legs::LegList; inc::Int=1)
    return _view_rewrap(q, prime(q.arr, _view_arr_legs(q, legs); inc=inc))
end
function prime(q::TLArrayView, pred::Function; inc::Int=1)
    return _view_rewrap(q, prime(q.arr, _view_arr_legs(q, findlegs(q, pred)); inc=inc))
end

function setprime(q::TLArrayView, n::Int; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, setprime(q.arr, _view_arr_legs(q, legs), n))
end
function setprime(q::TLArrayView, leg::Integer, n::Int)
    return _view_rewrap(q, setprime(q.arr, _view_arr_legs(q, leg), n))
end
function setprime(q::TLArrayView, legs::LegList, n::Int)
    return _view_rewrap(q, setprime(q.arr, _view_arr_legs(q, legs), n))
end

function noprime(q::TLArrayView; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, noprime(q.arr, _view_arr_legs(q, legs)))
end
function noprime(q::TLArrayView, leg::Integer)
    return _view_rewrap(q, noprime(q.arr, _view_arr_leg(q, leg)))
end
function noprime(q::TLArrayView, legs::LegList)
    return _view_rewrap(q, noprime(q.arr, _view_arr_legs(q, legs)))
end
function noprime(q::TLArrayView, pred::Function)
    return _view_rewrap(q, noprime(q.arr, _view_arr_legs(q, findlegs(q, pred))))
end

function additag(q::TLArrayView, newtags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, additag(q.arr, _view_arr_legs(q, legs), newtags))
end
function additag(q::TLArrayView, leg::Integer, newtags::AbstractString)
    return _view_rewrap(q, additag(q.arr, _view_arr_leg(q, leg), newtags))
end
function additag(q::TLArrayView, legs::LegList, newtags::AbstractString)
    return _view_rewrap(q, additag(q.arr, _view_arr_legs(q, legs), newtags))
end
function additag(q::TLArrayView, pred::Function, newtags::AbstractString)
    return _view_rewrap(q, additag(q.arr, _view_arr_legs(q, findlegs(q, pred)), newtags))
end

function removeitag(q::TLArrayView, tags::ITagQuerySpec; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, removeitag(q.arr, _view_arr_legs(q, legs), tags))
end
function removeitag(q::TLArrayView, leg::Integer, tags::ITagQuerySpec)
    return _view_rewrap(q, removeitag(q.arr, _view_arr_leg(q, leg), tags))
end
function removeitag(q::TLArrayView, legs::LegList, tags::ITagQuerySpec)
    return _view_rewrap(q, removeitag(q.arr, _view_arr_legs(q, legs), tags))
end
function removeitag(q::TLArrayView, pred::Function, tags::ITagQuerySpec)
    return _view_rewrap(q, removeitag(q.arr, _view_arr_legs(q, findlegs(q, pred)), tags))
end

function replaceitag(q::TLArrayView, replacements::ITagReplacementPair...; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_legs(q, legs), replacements...))
end
function replaceitag(q::TLArrayView, replacements::ITagReplacementDict; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_legs(q, legs), replacements))
end
function replaceitag(q::TLArrayView, leg::Integer, replacements::ITagReplacementPair...)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_leg(q, leg), replacements...))
end
function replaceitag(q::TLArrayView, leg::Integer, replacements::ITagReplacementDict)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_leg(q, leg), replacements))
end
function replaceitag(q::TLArrayView, legs::LegList, replacements::ITagReplacementPair...)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_legs(q, legs), replacements...))
end
function replaceitag(q::TLArrayView, legs::LegList, replacements::ITagReplacementDict)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_legs(q, legs), replacements))
end
function replaceitag(q::TLArrayView, pred::Function, replacements::ITagReplacementPair...)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_legs(q, findlegs(q, pred)), replacements...))
end
function replaceitag(q::TLArrayView, pred::Function, replacements::ITagReplacementDict)
    return _view_rewrap(q, replaceitag(q.arr, _view_arr_legs(q, findlegs(q, pred)), replacements))
end

function setitag(q::TLArrayView, tags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _view_rewrap(q, setitag(q.arr, _view_arr_legs(q, legs), tags))
end
function setitag(q::TLArrayView, leg::Integer, tags::AbstractString)
    return _view_rewrap(q, setitag(q.arr, _view_arr_leg(q, leg), tags))
end
function setitag(q::TLArrayView, legs::LegList, tags::AbstractString)
    return _view_rewrap(q, setitag(q.arr, _view_arr_legs(q, legs), tags))
end
function setitag(q::TLArrayView, pred::Function, tags::AbstractString)
    return _view_rewrap(q, setitag(q.arr, _view_arr_legs(q, findlegs(q, pred)), tags))
end

# ─── Pretty-printing for TLArray ──────────────────────────────────────────────
#
# Format:
#
#   TLArray{T}  [Symm1, Symm2, ...]
#     leg 1:  dir  'tag'  (plev=k)
#     leg 2:  dir  'tag'
#     ...
#     1.  <sector inline>
#     2.  <sector inline>
#     ...
#
# When the number of sectors exceeds TLARRAY_DISPLAY_HEAD + TLARRAY_DISPLAY_TAIL,
# only the first TLARRAY_DISPLAY_HEAD and last TLARRAY_DISPLAY_TAIL sectors are shown.
# Set these globals to control the truncation behaviour.
# ─────────────────────────────────────────────────────────────────────────────

const TLARRAY_DISPLAY_HEAD = Ref(5)   # number of first sectors to show
const TLARRAY_DISPLAY_TAIL = Ref(5)   # number of last sectors to show

Base.show(io::IO, qs::TLArray) = show(io, MIME"text/plain"(), qs)
function Base.show(io::IO, qs::TLArrayView)
    concrete = _eager_tlarray(qs)
    show(io, concrete)
    return concrete
end

_qindex_plev_string(plev::Int) =
    plev == 0 ? "" : "p$(plev)"

_qindex_lock_string(lock::Int) = lock == 0 ? "" : "🔒$(lock)"

function _format_qindex(idx::TLIndex)
    return "\"$(idx.itags)$(idx.dir)\"$(_qindex_plev_string(idx.plev))$(_qindex_lock_string(idx.lock))"
end

function _print_tlarray_header(io::IO, qs::TLArray{T, QD, N}) where {T, QD, N}
    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "$(QD)D TLArray, $N symmetries [$symm_names]")
    leg_strs = map(qs.inds) do idx
        raw = _format_qindex(idx)
        idx.dual ? "\e[32m$(raw)\e[0m" : raw
    end
    print(io, "  [", join(leg_strs, ", "), "]")
end

function _print_tlarray_header(io::IO, qs::TLArray{T, 0, N}) where {T, N}
    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "0D TLArray{$T}, $N symmetries [$symm_names]")
end

"""
    printmeta([io::IO], q::TLArray)

Print only the non-RMT metadata line for `q`, matching the non-numerical prefix
of the standard `TLArray` text display.
"""
function printmeta(io::IO, q::TLArray) 
    _print_tlarray_header(io, q)
    println()
end
printmeta(q::TLArray) = printmeta(stdout, q)
printmeta(io::IO, q::TLArrayView) = printmeta(io, _eager_tlarray(q))
printmeta(q::TLArrayView) = printmeta(stdout, q)

# Special pretty-printing for 0-dimensional TLArray (scalar result of full contraction).
function Base.show(io::IO, ::MIME"text/plain", qs::TLArray{T, 0, N, N}) where {T, N}
    _print_tlarray_header(io, qs)
    print(io, ": ", _fmt_scalar_str(qs[]))
end

function Base.show(io::IO, ::MIME"text/plain", qs::TLArray{T, QD, N, RD}) where {T, QD, N, RD}
    # --- Header: symmetries and leg dirs/tags on one line ---
    # Format:  TLArray{...}  [Sym1, Sym2]  ["tag1"+, "tag2"-', ...]
    _print_tlarray_header(io, qs)
    println(io)

    # --- Sectors: one per line with global label width for cross-sector alignment ---
    nr = nsectors(qs)
    if nr == 0
        print(io, "  (empty)")
        return
    end

    # Determine which sector indices to display.
    head = TLARRAY_DISPLAY_HEAD[]
    tail = TLARRAY_DISPLAY_TAIL[]
    truncate = nr > head + tail
    active_indices = [i for i in sector_slots(qs) if !qs.iszero[i]]
    display_indices = if truncate
        vcat(active_indices[1:head], active_indices[nr-tail+1:nr])
    else
        active_indices
    end

    # Compute per-symmetry widths globally across displayed sectors only.
    widths = map(1:N) do n
        maximum(display_indices) do i
            _sector_label_widths(qs, i)[n]
        end
    end
    # Pre-compute scalar width for alignment (only sectors with scalar RMT).
    scalar_width = 0
    for i in display_indices
        rmt = sector_rmt(qs, i)
        length(rmt) == 1 || continue
        scalar_width = max(scalar_width, length(_fmt_scalar_str(_display_scalar_rmt(qs, i))))
    end

    for (k, i) in enumerate(display_indices)
        # Print ellipsis between head and tail blocks.
        if truncate && k == head + 1
            println(io)
            println(io, "  ⋮  ($(nr - head - tail) sectors omitted)")
        end
        rmt = sector_rmt(qs, i)
        om_dim = prod(size(rmt)[QD+1:end]; init=1)
        phys_str = join(size(rmt)[1:QD], "x")
        om_str   = om_dim > 1 ? " @$om_dim" : ""
        print(io, "  $i.\t", phys_str, om_str, "\t")
        _print_sector_cgt_dims(io, qs, i)
        _print_sector_qlabels(io, qs, i, widths)
        length(rmt) == 1 &&
            print(io, "\t", lpad(_fmt_scalar_str(_display_scalar_rmt(qs, i)), scalar_width))
        QD == 2 && print(io, "\t√", _sector_cgt_size_2d(qs, i))
        k < length(display_indices) && println(io)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", qs::TLArrayView)
    concrete = _eager_tlarray(qs)
    show(io, mime, concrete)
    return concrete
end

# Return a TLArray with sectors sorted in dictionary order by physical leg qlabels.
# For each sector the sort key is built leg by leg (1 → QD): at leg l, the key
# is the tuple of qlabels across all symmetries, i.e.
#   (CGT metadata[1].qlabels[cgp₁[l]], CGT metadata[2].qlabels[cgp₂[l]], ...)
# Comparison is then lexicographic over (leg 1 key, leg 2 key, ..., leg QD key).
function sort_sectors(q::TLArray{T, QD, N}) where {T, QD, N}
    sector_indices = collect(sector_slots(q))
    perm = sortperm(sector_indices; by = sector_index -> Tuple(
            sector_qlabel(q, sector_index, l)
        for l in QD:-1:1)
    )
    sorted_indices = sector_indices[perm]
    qlabels = copy(q.qlabels[sorted_indices])
    wmatdata, wmatinfo = _copy_wmat_storage(q, sorted_indices; deep=true)
    RMTs = _copy_sector_RMTs(q, sorted_indices; deep=true)
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, q.inds, _copy_spaces_tuple(q.spaces))
end

# Scalar multiplication and division: only the RMT arrays are scaled.
# CGT metadata (w-matrices, qlabels) are left untouched.
function _materialized_scale(qs::TLArray{T, QD, N, RD}, fac::Number) where {T, QD, N, RD}
    RT = promote_type(T, typeof(fac))
    if iszero(fac)
        RMTs = Vector{Array{RT, RD}}(undef, sector_count(qs))
        return TLArray(symm(qs), copy(qs.qlabels), qs.wmatdata, qs.wmatinfo, RMTs, qs.inds, qs.spaces)
    end
    if RT !== T
        RMTs = eltype(qs.RMTs) <: DiagRMT ? Vector{DiagRMT{RT, RD}}(undef, sector_count(qs)) :
                                            Vector{Array{RT, RD}}(undef, sector_count(qs))
        for sector_index in sector_slots(qs)
            qs.iszero[sector_index] && continue
            RMTs[sector_index] = qs.RMTs[sector_index] * fac
        end
        return TLArray(symm(qs), copy(qs.qlabels), _copy_wmat_storage(qs; deep=true)...,
                       RMTs, qs.inds, _copy_spaces_tuple(qs.spaces))
    end
    result = deepcopy(qs)
    for sector_index in sector_slots(result)
        result.iszero[sector_index] && continue
        result.RMTs[sector_index] = result.RMTs[sector_index] * fac
    end
    return result
end
Base.:*(qs::AbstractTLArray, fac::Number) = _view_scale(qs, fac)
Base.:*(fac::Number, qs::AbstractTLArray) = qs * fac
Base.:/(qs::AbstractTLArray, fac::Number) = qs * (1 / fac)
Base.:-(qs::AbstractTLArray) = qs * -1

# Return a deep copy of a TLArray (CGT metadata, RMTs, indices, spaces all copied).
Base.copy(q::TLArray) = deepcopy(q)
Base.copy(q::TLArrayView) = _eager_tlarray(q)

function _identity_on_tlarray(q::AbstractTLArray{T, QD, N, RD}) where {T, QD, N, RD}
    @assert QD == 2 "Scalar add/subtract is only defined for rank-2 TLArray objects"

    in_legs  = findlegs(q; dir='+')
    out_legs = findlegs(q; dir='-')
    @assert length(in_legs) == 1 && length(out_legs) == 1 "Scalar add/subtract requires exactly one incoming and one outgoing leg"

    in_leg  = only(in_legs)
    out_leg = only(out_legs)
    legspaces = spaces(q)
    qinds = inds(q)
    @assert legspaces[in_leg] == legspaces[out_leg] "Scalar add/subtract requires matching incoming and outgoing spaces"

    id_q = getIdentity((q, out_leg); itag=qinds[out_leg].itags)
    return TLArray(id_q, (qinds[in_leg], qinds[out_leg]))
end

# ─── TLArray norm ─────────────────────────────────────────────────────────────
#
# Exploits the Wigner-Eckart decomposition to compute the Frobenius norm
# directly from the reduced matrix elements (RMTs) without building the full
# dense tensor.
#
# For all ranks:
#   Sectors from different q-label sectors are orthogonal by symmetry.  Within
#   each sector, the CGT canonical basis elements are orthonormal and the
#   w-matrix is left-orthogonal (Uᵀ·U = I), so the M columns of cgt_block
#   are also orthonormal.  Therefore all CGT cross-terms vanish and:
#       ‖A‖² = Σ_r ‖RMT_r‖²
#
# ─────────────────────────────────────────────────────────────────────────────
function LinearAlgebra.norm(q::TLArray{T, QD, N}) where {T, QD, N}
    s = zero(Float64)
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        s += sum(abs2, sector_rmt(q, sector_index))
    end
    return sqrt(s)
end

LinearAlgebra.norm(q::TLArrayView) = abs(q.scale) * norm(q.arr)

function _normalize_oplus_dims(dimensions, QD::Int; sort_dims::Bool=true)
    dims = dimensions isa Integer ? (Int(dimensions),) : Tuple(Int(d) for d in dimensions)
    isempty(dims) && throw(ArgumentError("oplus requires at least one dimension"))
    all(d -> 1 <= d <= QD, dims) || throw(ArgumentError(
        "oplus dimensions must lie in 1:$QD, got $(collect(dims))"))
    length(unique(dims)) == length(dims) || throw(ArgumentError(
        "oplus dimensions must be unique, got $(collect(dims))"))
    return sort_dims ? Tuple(sort(collect(dims))) : dims
end

function _normalize_oplus_matrix_dims(dimensions, QD::Int)
    dimensions isa Tuple && length(dimensions) == 2 || throw(ArgumentError(
        "matrix oplus requires exactly two axis-dimension specifications"))

    dims1 = _normalize_oplus_dims(dimensions[1], QD)
    dims2 = _normalize_oplus_dims(dimensions[2], QD)
    isempty(intersect(dims1, dims2)) || throw(ArgumentError(
        "matrix-axis oplus legs must be disjoint, got $(collect(dims1)) and $(collect(dims2))"))
    return dims1, dims2
end

function _splist_dim_map(splist::Vector)
    dims = Dict{Any, Int}()
    for (qlabels, dim) in splist
        if haskey(dims, qlabels)
            throw(ArgumentError("duplicate qlabel sector encountered in space list: $qlabels"))
        end
        dims[qlabels] = dim
    end
    return dims
end

function _sum_splists_many(splists)
    isempty(splists) && throw(ArgumentError("cannot sum an empty collection of space lists"))

    dims = Dict{Any, Int}()
    seen = Set{Any}()
    result = copy(first(splists))
    empty!(result)

    for splist in splists
        for (qlabels, dim) in splist
            dims[qlabels] = get(dims, qlabels, 0) + dim
            if qlabels ∉ seen
                push!(result, (qlabels, 0))
                push!(seen, qlabels)
            end
        end
    end

    for i in eachindex(result)
        qlabels, _ = result[i]
        result[i] = (qlabels, dims[qlabels])
    end

    return result
end

_copy_spaces_tuple(spaces::NTuple{QD, Vector}) where {QD} = ntuple(l -> copy(spaces[l]), QD)
_tlarray_eltype(::AbstractTLArray{T}) where {T} = T

function _oplus_pad_tlarray(q::TLArray{T, QD, N, RD},
                           result_spaces,
                           dims_tuple,
                           start_dim_maps,
                           result_dim_maps) where {T, QD, N, RD}
    dims_set = Set(dims_tuple)
    qlabels = copy(q.qlabels)
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    # Padding a direct-sum leg generally destroys diagonal RMT structure.
    RMTs = Vector{Array{T, RD}}(undef, sector_count(q))
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        old_sizes = size(sector_rmt(q, sector_index))
        new_phys_sizes = collect(old_sizes[1:QD])
        starts = ones(Int, QD)

        for leg in 1:QD
            leg ∈ dims_set || continue
            qlabel = sector_qlabel(q, sector_index, leg)
            new_phys_sizes[leg] = result_dim_maps[leg][qlabel]
            starts[leg] = get(start_dim_maps[leg], qlabel, 1)
        end

        new_sizes = Tuple(vcat(new_phys_sizes, collect(old_sizes[QD+1:end])))
        new_data = zeros(T, new_sizes)
        fill_inds = ntuple(axis -> begin
            if axis <= QD && axis ∈ dims_set
                start = starts[axis]
                stop = start + old_sizes[axis] - 1
                start:stop
            else
                Colon()
            end
        end, RD)
        new_data[fill_inds...] = sector_rmt(q, sector_index)
        RMTs[sector_index] = new_data
    end
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, q.inds, _copy_spaces_tuple(result_spaces))
end

function _zero_tlarray_with_spaces(symm::NTuple{N, Any},
                                  inds::NTuple{QD, TLIndex},
                                  spaces::NTuple{QD, Vector};
                                  T::Type=Float64) where {N, QD}
    QT = qlabeltype(symm)
    qlabels = Matrix{QT}(undef, QD, 0)
    M = n_nonabelian_symmetries(productsymm(symm))
    wmatdata = Float64[]
    wmatinfo = WMatInfo{M}[]
    RMTs = Array{T, QD + N}[]
    return TLArray(symm, qlabels, wmatdata, wmatinfo, RMTs, inds, _copy_spaces_tuple(spaces))
end

function _accumulate_oplus_starts(qs, dims_tuple, QD::Int)
    dims_set = Set(dims_tuple)
    running = [Dict{Any, Int}() for _ in 1:QD]
    starts = Vector{Vector{Dict{Any, Int}}}(undef, length(qs))

    for (qi, q) in enumerate(qs)
        starts[qi] = [Dict{Any, Int}() for _ in 1:QD]
        for leg in 1:QD
            leg ∈ dims_set || continue
            for (qlabels, dim) in q.spaces[leg]
                start = get(running[leg], qlabels, 1)
                starts[qi][leg][qlabels] = start
                running[leg][qlabels] = start + dim
            end
        end
    end

    return starts
end

_qindex_match_for_oplus(a::TLIndex, b::TLIndex) =
    a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.lock == b.lock

function _inds_match_for_oplus(inds1, inds2)
    length(inds1) == length(inds2) || return false
    return all(_qindex_match_for_oplus(idx1, idx2) for (idx1, idx2) in zip(inds1, inds2))
end

function _find_oplus_leg_permutation(ref_inds, inds, entry::Int)
    length(inds) == length(ref_inds) || throw(ArgumentError(
        "TLArray entry $entry has rank $(length(inds)), expected $(length(ref_inds))"))

    QD = length(ref_inds)
    candidates = Vector{Vector{Int}}(undef, QD)
    for i in 1:QD
        candidates[i] = [j for j in 1:QD if _qindex_match_for_oplus(inds[j], ref_inds[i])]
        isempty(candidates[i]) && throw(ArgumentError(
            "TLArray entry $entry has no leg matching reference leg $i " *
            "(same itag, direction, prime level, and lock required; dual ignored)"))
    end

    results = Vector{NTuple{QD, Int}}()
    _enum_leg_perms!(results, candidates, Int[], Set{Int}(), QD)
    isempty(results) && throw(ArgumentError(
        "TLArray entry $entry has no bijective leg permutation matching the reference indices"))
    length(results) > 1 && throw(ArgumentError(
        "TLArray entry $entry has ambiguous leg permutation matching the reference indices"))
    return results[1]
end

function _align_oplus_inputs(qs)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    first(qs) isa AbstractTLArray || throw(ArgumentError("oplus entry 1 is not a TLArray"))

    ref = _eager_tlarray(first(qs))
    aligned = Vector{TLArray}(undef, length(qs))
    aligned[1] = ref
    for i in 2:length(qs)
        q = qs[i]
        q isa AbstractTLArray || throw(ArgumentError("oplus entry $i is not a TLArray"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "TLArray entry $i has a different symmetry tuple"))
        qinds = inds(q)
        perm = _find_oplus_leg_permutation(ref.inds, qinds, i)
        aligned[i] = _eager_tlarray(
            perm == ntuple(identity, length(ref.inds)) ? q : permutedims(q, perm))
    end
    return aligned
end

function _validate_oplus_common(qs)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    first(qs) isa TLArray || throw(ArgumentError("oplus entry 1 is not a TLArray"))

    ref = first(qs)
    for (i, q) in enumerate(qs)
        q isa TLArray || throw(ArgumentError("oplus entry $i is not a TLArray"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "TLArray entry $i has a different symmetry tuple"))
        length(q.inds) == length(ref.inds) || throw(ArgumentError(
            "TLArray entry $i has rank $(length(q.inds)), expected $(length(ref.inds))"))
        _inds_match_for_oplus(q.inds, ref.inds) || throw(ArgumentError(
            "TLArray entry $i has different indices " *
            "(same itag, direction, prime level, and lock required; dual ignored)"))
    end

    return ref
end

function _oplus_dims_from_keywords(ref::TLArray; dir=nothing, itag=nothing,
                                   plev=nothing, lock=nothing, rev::Bool=false)
    isnothing(dir) && isnothing(itag) && isnothing(plev) && isnothing(lock) &&
        throw(ArgumentError("oplus keyword selection requires at least one selector"))
    dims = Tuple(findlegs(ref; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev))
    isempty(dims) && throw(ArgumentError("oplus keyword selectors did not match any legs"))
    return dims
end

function _build_vector_oplus_spaces(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    QD = length(ref.inds)
    dims_set = Set(dims_tuple)

    return ntuple(leg -> begin
        if leg ∈ dims_set
            _sum_splists_many([q.spaces[leg] for q in qs])
        else
            target = ref.spaces[leg]
            for i in 2:length(qs)
                qs[i].spaces[leg] == target || throw(ArgumentError(
                    "space lists must match on non-oplus leg $leg"))
            end
            copy(target)
        end
    end, QD)
end

function _materialize_vector_oplus(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    result_spaces = _build_vector_oplus_spaces(qs, dims_tuple)
    QD = length(ref.inds)
    result_dim_maps = ntuple(leg -> _splist_dim_map(result_spaces[leg]), QD)
    start_maps = _accumulate_oplus_starts(qs, dims_tuple, QD)
    T = promote_type((_tlarray_eltype(q) for q in qs)...)

    acc = _zero_tlarray_with_spaces(symm(ref), ref.inds, result_spaces; T=T)
    for (q, qstarts) in zip(qs, start_maps)
        padded = _oplus_pad_tlarray(q, result_spaces, dims_tuple, qstarts, result_dim_maps)
        padded = q.inds == ref.inds ? padded : TLArray(padded, ref.inds)
        acc = acc + padded
    end
    return acc
end

function _oplus_matrix_entry(mat, i::Int, j::Int)
    if applicable(isassigned, mat, i, j) && !isassigned(mat, i, j)
        return nothing
    end

    val = mat[i, j]
    if val === nothing || val === missing
        return nothing
    end
    val isa AbstractTLArray || throw(ArgumentError(
        "matrix oplus entry ($i, $j) is neither an AbstractTLArray nor an undefined entry"))
    return _eager_tlarray(val)
end

function _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i::Int, j::Int, QD::Int)
    return ntuple(leg -> begin
        have_first_axis = haskey(first_axis_sources[i], leg)
        have_second_axis = haskey(second_axis_sources[j], leg)
        if have_first_axis && have_second_axis
            first_axis_sources[i][leg] == second_axis_sources[j][leg] || throw(ArgumentError(
                "cannot infer zero TLArray at ($i, $j): inconsistent spaces on leg $leg"))
            copy(first_axis_sources[i][leg])
        elseif have_first_axis
            copy(first_axis_sources[i][leg])
        elseif have_second_axis
            copy(second_axis_sources[j][leg])
        else
            throw(ArgumentError(
                "cannot infer zero TLArray at ($i, $j): missing space information on leg $leg"))
        end
    end, QD)
end

"""
    oplus(qs::AbstractVector, dimensions)

Direct sum of a vector of `TLArray` objects along one or more physical legs.
"""
function oplus(qs::AbstractVector, dimensions)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    any(q -> q === nothing || q === missing, qs) && throw(ArgumentError(
        "vector oplus requires every entry to be well defined"))

    aligned = _align_oplus_inputs(collect(qs))
    ref = _validate_oplus_common(aligned)
    dims_tuple = _normalize_oplus_dims(dimensions, length(ref.inds))
    return _materialize_vector_oplus(aligned, dims_tuple)
end

function oplus(qs::AbstractVector; dir=nothing, itag=nothing, plev=nothing,
               lock=nothing, rev::Bool=false)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    any(q -> q === nothing || q === missing, qs) && throw(ArgumentError(
        "vector oplus requires every entry to be well defined"))

    aligned = _align_oplus_inputs(collect(qs))
    ref = _validate_oplus_common(aligned)
    dims_tuple = _oplus_dims_from_keywords(ref; dir=dir, itag=itag, plev=plev,
                                           lock=lock, rev=rev)
    return _materialize_vector_oplus(aligned, dims_tuple)
end

function oplus(q1::AbstractTLArray, q2::AbstractTLArray, dimensions)
    return oplus(AbstractTLArray[q1, q2], dimensions)
end

function oplus(q1::AbstractTLArray, q2::AbstractTLArray; dir=nothing, itag=nothing,
               plev=nothing, lock=nothing, rev::Bool=false)
    return oplus(AbstractTLArray[q1, q2]; dir=dir, itag=itag, plev=plev,
                 lock=lock, rev=rev)
end

function _complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    size(mat, 1) > 0 && size(mat, 2) > 0 || throw(ArgumentError(
        "matrix oplus requires a non-empty matrix"))

    defined_positions = Tuple{Int, Int}[]
    defined_qs = TLArray[]
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        q === nothing && continue
        push!(defined_positions, (i, j))
        push!(defined_qs, q)
    end
    isempty(defined_qs) && throw(ArgumentError(
        "matrix oplus requires at least one defined TLArray to infer spaces"))

    aligned_qs = _align_oplus_inputs(defined_qs)
    aligned_by_position = Dict(pos => q for (pos, q) in zip(defined_positions, aligned_qs))
    ref = _validate_oplus_common(aligned_qs)
    first_axis_dims, second_axis_dims = _normalize_oplus_matrix_dims(dimensions, length(ref.inds))
    first_axis_dims_set = Set(first_axis_dims)
    second_axis_dims_set = Set(second_axis_dims)

    first_axis_sources = [Dict{Int, Any}() for _ in axes(mat, 1)]
    second_axis_sources = [Dict{Int, Any}() for _ in axes(mat, 2)]

    for ((i, j), q) in zip(defined_positions, defined_qs)
        for leg in 1:length(ref.inds)
            if leg ∉ second_axis_dims_set
                if haskey(first_axis_sources[i], leg)
                    first_axis_sources[i][leg] == q.spaces[leg] || throw(ArgumentError(
                        "sector $i has incompatible spaces on leg $leg"))
                else
                    first_axis_sources[i][leg] = copy(q.spaces[leg])
                end
            end
            if leg ∉ first_axis_dims_set
                if haskey(second_axis_sources[j], leg)
                    second_axis_sources[j][leg] == q.spaces[leg] || throw(ArgumentError(
                        "column $j has incompatible spaces on leg $leg"))
                else
                    second_axis_sources[j][leg] = copy(q.spaces[leg])
                end
            end
        end
    end

    T = promote_type((_tlarray_eltype(q) for q in defined_qs)...)
    filled = Matrix{TLArray}(undef, size(mat, 1), size(mat, 2))
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        if q === nothing
            spaces = _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i, j, length(ref.inds))
            filled[i, j] = _zero_tlarray_with_spaces(symm(ref), ref.inds, spaces; T=T)
        else
            filled[i, j] = aligned_by_position[(i, j)]
        end
    end

    return filled, first_axis_dims, second_axis_dims
end

"""
    complete_oplus_matrix(mat::AbstractMatrix, dimensions)

Validate a matrix input for `oplus`, infer zero `TLArray` objects for undefined
entries, and return the completed `Matrix{TLArray}`.
"""
function complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    filled, _, _ = _complete_oplus_matrix(mat, dimensions)
    return filled
end

function oplus(mat::AbstractMatrix, dimensions)
    filled, first_axis_dims, second_axis_dims = _complete_oplus_matrix(mat, dimensions)

    col_aggregates = Vector{TLArray}(undef, size(filled, 2))
    for j in axes(filled, 2)
        col_aggregates[j] = _materialize_vector_oplus(vec(filled[:, j]), first_axis_dims)
    end

    return _materialize_vector_oplus(col_aggregates, second_axis_dims)
end



# ─── conj / adjoint ──────────────────────────────────────────────────────────
#
# conj(q): conjugate a TLArray object.
#   1. Every leg direction is reversed ('+' ↔ '-').
#   2. Every RMT entry is complex-conjugated.
#   3. When the old incoming qlabels equal the old outgoing qlabels
#      (self-conjugate CGT), the canonical OM ordering of the conjugated CGT
#      differs from the old one by a transposition within each central-space
#      block. This is applied as a sector permutation on the first dimension of
#      the w-matrix.
#
# adjoint(q): for TLArray tensors, defined as conj(q).
# ─────────────────────────────────────────────────────────────────────────────
function _materialized_conj(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    new_inds = ntuple(l -> change_dir(q.inds[l]), QD)

    qlabels = copy(q.qlabels)
    wmatdata = copy(q.wmatdata)
    wmatinfo = copy(q.wmatinfo)
    RMTs = similar(q.RMTs, sector_count(q))
    for sector_index in sector_slots(q)
        if !q.iszero[sector_index]
            RMTs[sector_index] = conj(sector_rmt(q, sector_index))
        end

        for n in 1:N
            isabelian(symm(q)[n]) && continue
            slot = nonabelian_wmat_slot(PS, n)
            _wmat_info_present(q.wmatinfo[sector_index][slot]) || continue
            ordered_qlabels, _, legdir = _sector_cgt_metadata(q, sector_index, n)
            m, k  = legdir

            # 3. Permute the OM (first) axis of the w-matrix when the old
            #    incoming and outgoing qlabels are identical tuples.  In that
            #    case the new CGT references the same canonical CGTom but with
            #    the flat OM ordering transposed within every central-space block:
            #      old flat = start + (upidx-1) + (dnidx-1)*om_up   [upidx fast]
            #      new flat = start + (dnidx-1) + (upidx-1)*om_dn   [dnidx fast]
            S = symm(q)[n]
            ins, outs = ordered_qlabels[1:m], ordered_qlabels[m+1:m+k]
            ins_, _ = remove_zeros(S, ins)
            outs_, _ = remove_zeros(S, outs)
            new_wmat =
                if !isabelian(S) && ins_ == outs_
                    conjperm = getNsave_Conjperm(S, ins_)
                    sector_wmat(q, sector_index, n)[conjperm.perm, :]
                else
                    deepcopy(sector_wmat(q, sector_index, n))
                end

            _copy_wmat_to_storage!(wmatdata, wmatinfo, sector_index, slot, new_wmat)
        end
    end

    # spaces remain the same: physical qlabels at each leg don't change in conj,
    # only the CGT internal structure (incoming/outgoing) changes
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, new_inds, _copy_spaces_tuple(q.spaces))
end

Base.conj(q::AbstractTLArray) = _view_conj(q)
Base.adjoint(q::AbstractTLArray) = conj(q)

getsub(q::TLArray, selector) = TLArray(q, selector)

function _normalize_getsub_index(i::Int, dim::Int, sector, leg::Int)
    i == 0 && throw(ArgumentError(
        "selector for sector $sector on leg $leg cannot contain 0"))
    return i < 0 ? dim + i + 1 : i
end

function _normalize_getsub_indices(raw, dim::Int, sector, leg::Int)
    inds = if raw isa Integer
        [Int(raw)]
    elseif raw isa Tuple && all(i -> i isa Integer, raw)
        Int[Int(i) for i in raw]
    elseif raw isa AbstractRange{<:Integer}
        Int[i for i in raw]
    elseif raw isa AbstractVector{<:Integer}
        Int[i for i in raw]
    else
        throw(ArgumentError(
            "selector for sector $sector on leg $leg must be :, Int, integer tuple, AbstractRange, or AbstractVector{<:Integer}"))
    end

    isempty(inds) && throw(ArgumentError(
        "selector for sector $sector on leg $leg must not be empty"))
    inds = [_normalize_getsub_index(i, dim, sector, leg) for i in inds]
    length(unique(inds)) == length(inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains duplicate indices"))
    all(1 <= i <= dim for i in inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains out-of-bounds indices for dimension $dim"))
    return inds
end
function _normalize_getsub_predicate_pick(raw, dim::Int, sector, leg::Int)
    raw isa Bool && throw(ArgumentError("getsub predicate for sector $sector on leg $leg must not return Bool; use Colon() or nothing explicitly"))
    raw === nothing && return nothing
    raw isa Colon && return Colon()
    return _normalize_getsub_indices(raw, dim, sector, leg)
end

function _collect_getsub_predicate_picks(q::TLArray, positions, pred::Function)
    selected_picks = Dict{Int, Dict{Any, Any}}()
    for leg in positions
        picks = Dict{Any, Any}()
        for (sector, dim) in q.spaces[leg]
            pick = _normalize_getsub_predicate_pick(pred(sector), dim, sector, leg)
            isnothing(pick) && continue
            picks[sector] = pick
        end
        selected_picks[leg] = picks
    end
    return selected_picks
end

function _apply_getsub_picks(q::TLArray{T, QD, N, RD, QT},
                             positions,
                             selected_picks::Dict{Int, Dict{Any, Any}};
                             preserve_space::Bool=false) where {T, QD, N, RD, QT}
    if preserve_space
        for leg in positions
            for (sector, pick) in selected_picks[leg]
                pick isa Colon && continue
                throw(ArgumentError(
                    "preserve_space=true is incompatible with slicing sector $sector on leg $leg; return Colon() or nothing instead"))
            end
        end
    end

    selected_leg_set = Set(positions)
    qlabels_out = NTuple{QD, QT}[]
    source_wmat_sectors = Int[]
    RMTs_out = Array{T, RD}[]
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        picks_by_leg = Dict{Int, Any}()
        keep = true
        for leg in positions
            sector = sector_qlabel(q, sector_index, leg)
            picks = selected_picks[leg]
            haskey(picks, sector) || (keep = false; break)
            picks_by_leg[leg] = picks[sector]
        end
        keep || continue
        selectors = ntuple(d -> get(picks_by_leg, d, Colon()), RD)
        selected_rmt = sector_rmt(q, sector_index)[selectors...]
        iszero(sum(abs2, selected_rmt)) && continue
        push!(qlabels_out, ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD)))
        push!(source_wmat_sectors, sector_index)
        push!(RMTs_out, selected_rmt)
    end

    spaces_out = if preserve_space
        _copy_spaces_tuple(q.spaces)
    else
        ntuple(l -> begin
            if l in selected_leg_set
                out = eltype(q.spaces[l])[]
                picks = selected_picks[l]
                for (sector, dim) in q.spaces[l]
                    haskey(picks, sector) || continue
                    pick = picks[sector]
                    push!(out, (sector, pick isa Colon ? dim : length(pick)))
                end
                out
            else
                copy(q.spaces[l])
            end
        end, QD)
    end

    wmatdata, wmatinfo = _copy_wmat_storage(q, source_wmat_sectors; deep=true)
    return TLArray(symm(q), qlabels_out, wmatdata, wmatinfo, RMTs_out, q.inds, spaces_out)
end

function _normalize_getsub_predicate_legs(q::TLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("getsub requires at least one leg"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "getsub legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "getsub legs must be unique, got $positions"))
    return positions
end

"""
    getsub(q::TLArray, leg::Integer, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sector on `leg` satisfies
`pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), only `q.spaces[leg]` is truncated to
the retained sectors and all other leg-space lists are copied unchanged. If
`preserve_space=true`, all cached leg-space lists are preserved exactly and only
the sectors are filtered. This requires `pred` to keep whole sectors, so any
index-selection return value is rejected when `preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, leg::Integer, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    return getsub(q, (Int(leg),), pred; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, legs, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sectors on every selected leg
satisfy `pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), each selected leg keeps only the
matching entries in its space list, while unselected legs keep copies of their
original space lists. If `preserve_space=true`, all cached leg-space lists are
preserved exactly and only the sectors are filtered. This requires `pred` to keep
whole sectors, so any index-selection return value is rejected when
`preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, legs::LegList, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    positions = _normalize_getsub_predicate_legs(q, legs)
    selected_picks = _collect_getsub_predicate_picks(q, positions, pred)
    return _apply_getsub_picks(q, positions, selected_picks; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, pred::Function; preserve_space::Bool=false, dir=nothing,
           itag=nothing, plev=nothing, lock=nothing, rev=false) -> TLArray

Apply predicate-based `getsub` to every leg selected by the keyword criteria.
The leg selection follows the same matching rules as `findlegs`.
"""
function getsub(q::TLArray{T, QD, N, RD}, pred::Function; preserve_space::Bool=false,
                dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                rev::Bool=false) where {T, QD, N, RD}
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="getsub")
    return getsub(q, legs, pred; preserve_space=preserve_space)
end

"""
    empty_tlarray(symm::NTuple{N, Any}, inds::NTuple{QD, TLIndex}; T::Type=Float64) where {N, QD}

Create an empty rank-`QD` (zero-sector) TLArray over the given symmetries.

`symm` is an `N`-tuple of symmetry types (e.g. `(SU{2}, U1)`); `inds` is a
`QD`-tuple of `TLIndex` objects describing the leg directions, tags, and prime
levels.  All `TLIndex` entries with non-empty tags must be pairwise distinct.

The element type of future sector data defaults to `Float64`; pass `T=ComplexF64`
(or another concrete `<:Number` type) to use a different element type.
"""
function empty_tlarray(symm::NTuple{N, Any}, inds::NTuple{QD, TLIndex};
                      T::Type=Float64) where {N, QD}
    RD = QD + N
    QT = qlabeltype(symm)
    spaces = ntuple(_ -> Vector{Tuple{QT, Int}}(), QD)
    qlabels = NTuple{QD, QT}[]
    M = n_nonabelian_symmetries(productsymm(symm))
    wmatdata = Float64[]
    wmatinfo = WMatInfo{M}[]
    RMTs = Array{T, RD}[]
    return TLArray(symm, qlabels, wmatdata, wmatinfo, RMTs, inds, spaces)
end

function empty_tlarray(q::AbstractTLArray; T::Type=Float64)
    return empty_tlarray(symm(q), inds(q); T=T)
end

function Base.zero(q::AbstractTLArray{T, QD, N, RD}) where {T, QD, N, RD}
    QT = qlabeltype(q)
    qlabels = NTuple{QD, QT}[]
    M = n_nonabelian_symmetries(productsymm(q))
    wmatdata = Float64[]
    wmatinfo = WMatInfo{M}[]
    RMTs = Array{T, RD}[]
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, inds(q), _copy_spaces_tuple(spaces(q)))
end

"""
    qlabeltype(symm::NTuple{N, Any}) where {N}
    qlabeltype(q::TLArray)

Return the qlabel type for one leg sector over the symmetries in `symm` or `q`.

For example, `(U1, SU{3})` returns `Tuple{Tuple{Int}, NTuple{2, Int}}`.
"""
function qlabeltype(symm::NTuple{N, Any}) where {N}
    return Tuple{ntuple(n -> NTuple{nzops(symm[n]), Int}, N)...}
end

qlabeltype(::Type{<:ProductSymm{Syms}}) where {Syms} =
    Tuple{(NTuple{nzops(S), Int} for S in Syms.parameters)...}

qlabeltype(::AbstractTLArray{T, QD, N, RD, QT}) where {T, QD, N, RD, QT} = QT

"""
    zero_qlabels(symm::NTuple{N, Any}) where {N}
    zero_qlabels(q::TLArray)

Return the trivial qlabel for each symmetry in `symm` or `q`.

For example, `(SU{2}, SU{3})` returns `((0,), (0, 0))`.
"""
function zero_qlabels(symm::NTuple{N, Any}) where {N}
    return ntuple(n -> Tuple(0 for _ in 1:nzops(symm[n])), N)
end

zero_qlabels(q::TLArray) = zero_qlabels(symm(q))

function _is_singleton_leg(q::TLArray{T, QD, N}, leg::Int) where {T, QD, N}
    1 <= leg <= QD || throw(ArgumentError("leg must lie in 1:$QD, got $leg"))
    return length(q.spaces[leg]) == 1 && only(q.spaces[leg]) == (zero_qlabels(q), 1)
end

_singleton_legs(q::TLArray{T, QD}) where {T, QD} = [leg for leg in 1:QD if _is_singleton_leg(q, leg)]

function _normalize_delete_singleton_legs(q::TLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one deletion leg must be specified"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "singleton deletion legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "singleton deletion legs must be unique, got $positions"))
    sort!(positions)
    return positions
end


function _delete_singleton_rmt(rmt::AbstractArray{T, RD}, positions, qd::Int, n_symm::Int) where {T, RD}
    new_dims = ntuple(axis -> begin
        if axis <= qd - length(positions)
            old_axis = axis
            for pos in positions
                pos <= old_axis && (old_axis += 1)
            end
            size(rmt, old_axis)
        else
            size(rmt, axis + length(positions))
        end
    end, qd + n_symm - length(positions))
    return reshape(rmt, new_dims)
end

function _delete_singleton_rmt(rmt::DiagRMT{T, RD}, positions, qd::Int, n_symm::Int) where {T, RD}
    any(pos -> pos == rmt.axis[1] || pos == rmt.axis[2], positions) && throw(ArgumentError(
        "cannot preserve DiagRMT when deleting a diagonal axis"))
    shift_axis(axis) = axis - count(pos -> pos < axis, positions)
    return DiagRMT{T,RD - length(positions)}(rmt.diag, (shift_axis(rmt.axis[1]), shift_axis(rmt.axis[2])))
end

function _delete_singleton_impl(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, positions) where {T, QD, N, RD, QT, PS, M, RMT}
    new_qd = QD - length(positions)
    new_rd = RD - length(positions)

    keep_inds = [q.inds[leg] for leg in 1:QD if leg ∉ positions]
    keep_spaces = Vector{Tuple{QT, Int}}[q.spaces[leg] for leg in 1:QD if leg ∉ positions]

    qlabels = Matrix{QT}(undef, new_qd, sector_count(q))
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    RMTs = RMT <: DiagRMT ? Vector{DiagRMT{T, new_rd}}(undef, sector_count(q)) :
                             Vector{Array{T, new_rd}}(undef, sector_count(q))
    keep_legs = [leg for leg in 1:QD if leg ∉ positions]
    for sector_index in sector_slots(q)
        for (new_leg, old_leg) in enumerate(keep_legs)
            qlabels[new_leg, sector_index] = sector_qlabel(q, sector_index, old_leg)
        end
        q.iszero[sector_index] && continue
        RMTs[sector_index] = _delete_singleton_rmt(sector_rmt(q, sector_index), positions, QD, N)
    end

    return TLArray(product_symms(PS), qlabels, wmatdata, wmatinfo, RMTs, Tuple(keep_inds), Tuple(keep_spaces))
end

"""
    deleteSingleton(q::TLArray; dir=nothing, itag=nothing, plev=nothing) -> TLArray

Delete singleton legs matching the supplied criteria.

With no keyword arguments, all singleton legs are deleted.
Only singleton legs are eligible for deletion. If none match, a warning is
emitted and `q` is returned unchanged.
"""
function deleteSingleton(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing) where {T, QD}
    singleton_legs = if isnothing(dir) && isnothing(itag) && isnothing(plev)
        _singleton_legs(q)
    else
        candidate_legs = findlegs(q; dir=dir, itag=itag, plev=plev)
        [leg for leg in candidate_legs if _is_singleton_leg(q, leg)]
    end

    if isempty(singleton_legs)
        if isnothing(dir) && isnothing(itag) && isnothing(plev)
            @warn "deleteSingleton: no singleton legs found"
        else
            @warn "deleteSingleton: no singleton legs matched the requested criteria"
        end
        return q
    end
    return _delete_singleton_impl(q, singleton_legs)
end

"""
    deleteSingleton(q::TLArray, leg::Integer) -> TLArray
    deleteSingleton(q::TLArray, legs::LegList) -> TLArray

Delete the specified singleton legs from `q`.

Every selected leg must be singleton. Otherwise an `ArgumentError` is thrown.
"""
function deleteSingleton(q::TLArray, leg::Integer)
    return deleteSingleton(q, (leg,))
end

function deleteSingleton(q::TLArray{T, QD}, legs::LegList) where {T, QD}
    positions = _normalize_delete_singleton_legs(q, legs)
    bad = [leg for leg in positions if !_is_singleton_leg(q, leg)]
    isempty(bad) || throw(ArgumentError(
        "deleteSingleton requires singleton legs, but legs $bad are not singleton"))
    return _delete_singleton_impl(q, positions)
end

function _expand_singleton_kw(values, count::Int, name::AbstractString)
    if values isa AbstractString || values isa Char || values isa Integer
        return fill(values, count)
    end
    collected = collect(values)
    length(collected) == count || throw(ArgumentError(
        "$name must have length $count, got $(length(collected))"))
    return collected
end

function _singleton_insert_spec(q::AbstractTLArray{T, QD}, legs;
                                itag="", plev=0, lock=0, dir='+') where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one insertion leg must be specified"))

    count = length(positions)
    itag_vec = _expand_singleton_kw(itag, count, "itag")
    plev_vec = _expand_singleton_kw(plev, count, "plev")
    lock_vec = _expand_singleton_kw(lock, count, "lock")
    dir_vec  = _expand_singleton_kw(dir,  count, "dir")

    perm = sortperm(positions)
    positions = positions[perm]
    itag_vec = itag_vec[perm]
    plev_vec = plev_vec[perm]
    lock_vec = lock_vec[perm]
    dir_vec  = dir_vec[perm]

    final_qd = QD + count
    all(p -> 1 <= p <= final_qd, positions) || throw(ArgumentError(
        "singleton insertion legs must lie in 1:$final_qd, got $positions"))
    length(unique(positions)) == count || throw(ArgumentError(
        "singleton insertion legs must be unique, got $positions"))

    for direction in dir_vec
        direction in ('+', '-') || throw(ArgumentError(
            "added leg directions must be '+' or '-', got '$direction'"))
    end

    return positions, itag_vec, plev_vec, lock_vec, dir_vec
end

function _insert_singleton_rmt(rmt::AbstractArray{T, RD},
                               positions,
                               qd::Int,
                               n_symm::Int) where {T, RD}
    old_phys = size(rmt)[1:qd]
    om_dims = size(rmt)[qd+1:qd+n_symm]

    new_phys = Int[]
    sizehint!(new_phys, qd + length(positions))

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:(qd + length(positions))
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            push!(new_phys, 1)
            insert_idx += 1
        else
            push!(new_phys, old_phys[old_leg])
            old_leg += 1
        end
    end

    return reshape(rmt, Tuple(vcat(new_phys, collect(om_dims))))
end

function _insert_singleton_rmt(rmt::DiagRMT{T, RD},
                               positions,
                               qd::Int,
                               n_symm::Int) where {T, RD}
    shift_axis(axis) = axis + count(pos -> pos <= axis, positions)
    return DiagRMT{T,RD + length(positions)}(rmt.diag, (shift_axis(rmt.axis[1]), shift_axis(rmt.axis[2])))
end

"""
    addSingleton(q::TLArray; nlegs=1, itag="", plev=0, lock=0, dir='+')

Append one or more singleton trivial legs to `q`.

The `nlegs` keyword controls how many singleton legs are created. New legs are
added to the end of the leg list. `itag`, `plev`, `lock`, and `dir` may each be
either a scalar applied to every added leg or an iterable with one value per
inserted leg.
"""
function addSingleton(q::TLArray{T, QD}; nlegs::Integer=1,
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    count = Int(nlegs)
    count >= 1 || throw(ArgumentError("nlegs must be at least 1, got $nlegs"))
    positions = (QD + 1):(QD + count)
    return addSingleton(q, positions; itag=itag, plev=plev, lock=lock, dir=dir)
end

"""
    addSingleton(q::TLArray, legs; itag="", plev=0, lock=0, dir='+')

Insert one or more singleton trivial legs into `q`.

`legs` may be a single integer or any iterable of integers. Each value is the
position of an added leg in the output tensor, whose rank is `ndims(q) +
length(legs)`. The original legs keep their relative order.

`itag`, `plev`, `lock`, and `dir` may each be either a scalar applied to
every added leg or an iterable with one value per inserted leg.
"""
function addSingleton(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, legs;
                      itag="", plev=0, lock=0, dir='+') where {T, QD, N, RD, QT, PS, M, RMT}
    positions, itag_vec, plev_vec, lock_vec, dir_vec =
        _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)

    new_qd = QD + length(positions)
    new_rd = RD + length(positions)
    trivial_qlabels = zero_qlabels(q)

    new_inds = Vector{TLIndex}(undef, new_qd)
    new_spaces = Vector{Vector{Tuple{QT, Int}}}(undef, new_qd)
    singleton_space = [(trivial_qlabels, 1)]

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            new_inds[new_leg] = TLIndex(String(itag_vec[insert_idx]), dir_vec[insert_idx],
                                       plev_vec[insert_idx], lock_vec[insert_idx])
            new_spaces[new_leg] = copy(singleton_space)
            insert_idx += 1
        else
            new_inds[new_leg] = q.inds[old_leg]
            new_spaces[new_leg] = q.spaces[old_leg]
            old_leg += 1
        end
    end

    qlabels = Matrix{QT}(undef, new_qd, sector_count(q))
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    RMTs = RMT <: DiagRMT ? Vector{DiagRMT{T, new_rd}}(undef, sector_count(q)) :
                             Vector{Array{T, new_rd}}(undef, sector_count(q))
    for sector_index in sector_slots(q)
        old_leg = 1
        insert_idx = 1
        for new_leg in 1:new_qd
            if insert_idx <= length(positions) && positions[insert_idx] == new_leg
                qlabels[new_leg, sector_index] = trivial_qlabels
                insert_idx += 1
            else
                qlabels[new_leg, sector_index] = sector_qlabel(q, sector_index, old_leg)
                old_leg += 1
            end
        end
        q.iszero[sector_index] && continue
        RMTs[sector_index] = _insert_singleton_rmt(sector_rmt(q, sector_index), positions, QD, N)
    end

    return TLArray(product_symms(PS), qlabels, wmatdata, wmatinfo, RMTs, Tuple(new_inds), Tuple(new_spaces))
end

function addSingleton(q::TLArrayView{T, QD}; nlegs::Integer=1,
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    count = Int(nlegs)
    count >= 1 || throw(ArgumentError("nlegs must be at least 1, got $nlegs"))
    positions = (QD + 1):(QD + count)
    return addSingleton(q, positions; itag=itag, plev=plev, lock=lock, dir=dir)
end

function addSingleton(q::TLArrayView{T, QD}, legs;
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    positions, itag_vec, plev_vec, lock_vec, dir_vec =
        _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)
    internal_dir = q.conj ? [d == '+' ? '-' : '+' for d in dir_vec] : dir_vec
    new_arr = addSingleton(q.arr; nlegs=length(positions),
                           itag=itag_vec, plev=plev_vec, lock=lock_vec, dir=internal_dir)

    new_qd = QD + length(positions)
    new_perm = Vector{Int}(undef, new_qd)
    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            new_perm[new_leg] = QD + insert_idx
            insert_idx += 1
        else
            new_perm[new_leg] = q.perm[old_leg]
            old_leg += 1
        end
    end
    return _normalize_tlarray_view(new_arr, q.conj, q.scale, Tuple(new_perm))
end

"""
    getvac(q::TLArray, itags::Tuple{Vararg{AbstractString, 2}}=("", "")) -> TLArray

Build the rank-2 vacuum TLArray associated with `q`.

The result keeps the same symmetry tuple as `q`, has one incoming leg and one
outgoing leg, and contains exactly one trivial sector with RMT dimension 1 on
each leg. If `itags` is provided, it is used as the tags of the two legs.
"""
function getvac(q::TLArray{T, QD, N, RD},
                itags::Tuple{Vararg{AbstractString, 2}}=("", "")) where {T, QD, N, RD}
    trivial_qlabels = zero_qlabels(q)
    space_entry = (trivial_qlabels, 1)

    qlabels = Matrix{qlabeltype(q)}(undef, 2, 1)
    qlabels[1, 1] = trivial_qlabels
    qlabels[2, 1] = trivial_qlabels
    wmatdata, wmatinfo = _unit_wmat_storage(productsymm(q), 1)
    rmt_data = fill(one(T), ntuple(_ -> 1, N + 2))
    RMTs = Array{T, N + 2}[rmt_data]
    inds = (TLIndex(itags[1], '+'), TLIndex(itags[2], '-'))
    QT = qlabeltype(q)
    space_template = Vector{Tuple{QT, Int}}([space_entry])
    spaces = (copy(space_template), copy(space_template))

    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, inds, spaces)
end

function ⊗(q1::TLArray{T1, QD1, N, RD1},
           q2::TLArray{T2, QD2, N, RD2}) where {T1, T2, QD1, QD2, N, RD1, RD2}
    @assert symm(q1) == symm(q2) "TLArray objects must share the same symmetry tuple"

    q1_ext = addSingleton(q1, QD1 + 1; dir='-')
    q2_ext = addSingleton(q2, 1; dir='+')
    return contract(q1_ext, (QD1 + 1,), q2_ext, (1,))
end

LinearAlgebra.kron(q1::TLArray, q2::TLArray) = q1 ⊗ q2

include("getLocalSpace.jl")
include("getIdentity.jl")
include("contract.jl")
include("sum_tlarray.jl")
include("get1jtensor.jl")
include("svd.jl")
include("eig.jl")
include("permute.jl")
