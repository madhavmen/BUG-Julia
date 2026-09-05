# ─── svd ───────────────────────────────────────────────────────────────
#
# Perform symmetry-adapted SVD of a TLArray object.
#
# Arguments:
#   q          : TLArray to decompose (any rank QD, N symmetries)
#   left_legs  : tuple/vector of leg indices forming the left (U) side
#   left_tag   : itag for the new bond leg on the U/S side   (default "svdL")
#   right_tag  : itag for the new bond leg on the S/Vd side  (default "svdR")
#   cutoff     : keep singular values > cutoff * σ_max  (default 1e-12)
#   Nkeep      : keep the Nkeep largest singular values globally, ignoring
#                degeneracy and counting missing sectors as zero singular values
#
# Returns SVDResult(U, S, Vd, kept_list, trunc_list) where:
#   U  : legs (original left legs...,  bond '+')
#   S  : legs (left_tag '-',  right_tag '-')  [diagonal, singular values]
#   Vd : legs (bond '-', original right legs...)
#   kept_list, trunc_list : singular-value metadata lists when get_lists=true,
#                           otherwise nothing
#
# Convention for bond legs:
#   U  bond =  TLIndex(left_tag,  '+')   — incoming (enters U from the right)
#   S  left  = TLIndex(left_tag,  '-')   — outgoing (leaves S to the left)
#   S  right = TLIndex(right_tag, '-')   — outgoing (leaves S to the right)
#   Vd bond  = TLIndex(right_tag, '-')   — outgoing (leaves Vd to the left)
#
# Legs of U and Vd that come from the original tensor inherit their TLIndex
# properties (itags, lock, plev, dual, direction).
#
# Algorithm:
#   1. Assign each leg of q a unique internal tag at lock=1.
#   2. Build fusing isometries aL / aR via getIdentity.
#   3. Contract q_work with aL/aR, yielding the rank-2 bipartition M.
#   4. Per-sector SVD on M (sL × sR matrix). Truncate either by cutoff alone
#      or by the Nkeep largest singular values, where missing sectors from
#      M.spaces are treated as explicit zero singular values.
#   5. Build rank-2 U, S, Vd from kept singular values.
#   6. Split fused legs of U and Vd by contracting with conjugate of aL/aR.
#   7. Permute legs to desired order and restore original TLIndex properties.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Implement a version that get left_legs by predicates or various keyword arguments
# TODO: Test svd with trunction for TLArray object (This is not rigorous yet)
struct SVDResult{TU, TS, TVD, TKL, TTL}
    U::TU
    S::TS
    Vd::TVD
    kept_list::TKL
    trunc_list::TTL
end

function SVDResult(U::TLArray, S::TLArray, Vd::TLArray, kept_list, trunc_list)
    return SVDResult{typeof(U), typeof(S), typeof(Vd), typeof(kept_list), typeof(trunc_list)}(
        U, S, Vd, kept_list, trunc_list)
end

_svd_sector_qlabels(q::TLArray, sector_index::Int, N::Int) =
    Tuple(sector_qlabel(q, sector_index, 1)[n] for n in 1:N)

@inline function _svd_physical_qlabels(::Type{QT},
                                       qlabels_by_symm::Tuple{Vararg{Tuple, N}},
                                       cgp_by_symm::Tuple{Vararg{NTuple{QD, Int}, N}},
                                       ::Val{QD}) where {QT, N, QD}
    return ntuple(Val(QD)) do leg
        ntuple(n -> qlabels_by_symm[n][cgp_by_symm[n][leg]], Val(N))::QT
    end
end

_svd_dual_sector(symm, sector) = Tuple(get_dualq(symm[n], sector[n]) for n in 1:length(symm))

@generated function _svd_dual_sector(::Type{PS}, sector::QT, ::Val{N}) where {PS<:ProductSymm, QT, N}
    syms = product_symms(PS)
    return Expr(:tuple, [:(get_dualq($(syms[n]), sector[$n])) for n in 1:N]...)
end

_svd_stable_sort_tuple(spaces) = Tuple(sort!(collect(spaces); alg=MergeSort))

_is_zero_array(arr::AbstractArray) = all(iszero, arr)

function _svd_symmetry_stored_leg_order(
    ::Type{QT},
    q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
    sector_index::Int,
    ::Val{n},
) where {T, QD, N, RD, QT, PS, M, RMT, n}
    qlabels_by_phys = ntuple(leg -> sector_qlabel(QT, q, sector_index, leg)[n], Val(QD))

    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for leg in 1:QD
        if q.inds[leg].dir == '+'
            push!(incoming, leg)
        else
            push!(outgoing, leg)
        end
    end

    # Stable sort encodes the TLArray/CGT tie rule: equal direction and qlabel
    # preserve physical tensor-leg order.
    sort!(incoming; by = leg -> qlabels_by_phys[leg], alg=MergeSort)
    sort!(outgoing; by = leg -> qlabels_by_phys[leg], alg=MergeSort)

    nin = length(incoming)
    stored_to_phys = ntuple(i -> i <= nin ? incoming[i] : outgoing[i - nin], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    stored_qlabels = ntuple(i -> qlabels_by_phys[stored_to_phys[i]], Val(QD))
    legdir = (nin, QD - nin)
    return stored_qlabels, phys_to_stored, stored_to_phys, legdir
end

@inline _svd_physical_qlabel_key(::Type{QT},
                                 q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                 sector_index::Int,
                                 ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, n} =
    ntuple(leg -> sector_qlabel(QT, q, sector_index, leg)[n], Val(QD))

@inline _svd_physical_product_key(::Type{QT},
                                  q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                  sector_index::Int) where {T, QD, N, RD, QT, PS, M, RMT} =
    ntuple(leg -> sector_qlabel(QT, q, sector_index, leg), Val(QD))

function _svd_physical_key_stored_order(physical_key::NTuple{QD},
                                        inds::NTuple{QD, TLIndex}) where {QD}
    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for leg in 1:QD
        if inds[leg].dir == '+'
            push!(incoming, leg)
        else
            push!(outgoing, leg)
        end
    end

    sort!(incoming; by = leg -> physical_key[leg], alg=MergeSort)
    sort!(outgoing; by = leg -> physical_key[leg], alg=MergeSort)

    nin = length(incoming)
    stored_to_phys = ntuple(i -> i <= nin ? incoming[i] : outgoing[i - nin], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    stored_qlabels = ntuple(i -> physical_key[stored_to_phys[i]], Val(QD))
    legdir = (nin, QD - nin)
    return stored_qlabels, phys_to_stored, stored_to_phys, legdir
end

function _svd_physical_key_split_args(::Type{S},
                                      physical_key::NTuple{QD, NTuple{NZ, Int}},
                                      inds::NTuple{QD, TLIndex},
                                      left_legs::NTuple{L, Int}) where {S<:NonabelianSymm, QD, NZ, L}
    @assert NZ == nzops(S)
    stored_qlabels, phys_to_stored, _, legdir =
        _svd_physical_key_stored_order(physical_key, inds)
    upsp, dnsp = _svd_cgt_updn(stored_qlabels, legdir)
    left_legs_canon = _svd_to_cgtidx(phys_to_stored, left_legs)::NTuple{L, Int}
    return upsp, dnsp, left_legs_canon
end

@inline function _svd_physical_key_side_signatures(physical_key::NTuple{QD, QT},
                                                   left_legs::NTuple{L, Int},
                                                   right_legs::NTuple{R, Int}) where {QD, QT, L, R}
    left_signature = ntuple(i -> physical_key[left_legs[i]], Val(L))
    right_signature = ntuple(i -> physical_key[right_legs[i]], Val(R))
    return left_signature, right_signature
end

struct _SVDSplitRow{L, R, QT}
    sector_index::Int
    q::QT
    left_signature::NTuple{L, QT}
    right_signature::NTuple{R, QT}
end

struct _SVDSymmetrySplit{L, R, QTN}
    sector_index::Int
    q::QTN
    left_signature::NTuple{L, QTN}
    right_signature::NTuple{R, QTN}
    left_iso::Matrix{Float64}
    right_iso::Matrix{Float64}
    core::Array{Float64, 3}
end

struct _SVDCGTBlockInfo{NZ}
    q::NTuple{NZ, Int}
    omL::Int
    omR::Int
    coeffs::Array{Float64, 3}
end

struct _SVDClassSideInfo{Sig, L, N}
    signature::Sig
    row_index::Int
    sector_index::Int
    phys_dims::NTuple{L, Int}
    om_dims::NTuple{N, Int}
    range::UnitRange{Int}
end

struct _SVDCGTClassMetadata{QT, LI, LR, RI, RR}
    sector::QT
    rows::UnitRange{Int}
    left_infos::LI
    left_ranges::LR
    total_left::Int
    right_infos::RI
    right_ranges::RR
    total_right::Int
end

struct _SVDCGTClassResult{QT, LI, RI, U, S, Vt}
    sector::QT
    left_infos::LI
    right_infos::RI
    U::U
    S::S
    Vt::Vt
end

_svd_class_side_info_type(::Type{QT}, ::Val{L}, ::Val{N}) where {QT, L, N} =
    _SVDClassSideInfo{NTuple{L, QT}, L, N}

function _svd_cgtsvd_class_metadata_type(::Type{QT}, ::Val{NL}, ::Val{NR}, ::Val{N}) where {QT, NL, NR, N}
    LeftSig = NTuple{NL, QT}
    RightSig = NTuple{NR, QT}
    LI = Vector{_svd_class_side_info_type(QT, Val(NL), Val(N))}
    RI = Vector{_svd_class_side_info_type(QT, Val(NR), Val(N))}
    LR = Dict{LeftSig, UnitRange{Int}}
    RR = Dict{RightSig, UnitRange{Int}}
    return _SVDCGTClassMetadata{QT, LI, LR, RI, RR}
end

function _svd_cgtsvd_class_result_type(
    ::Type{_SVDCGTClassMetadata{QT, LI, LR, RI, RR}},
    ::Type{Tmat}) where {QT, LI, LR, RI, RR, Tmat}
    S = Vector{real(Tmat)}
    return _SVDCGTClassResult{QT, LI, RI, Matrix{Tmat}, S, Matrix{Tmat}}
end

function _svd_cgt_updn(qlabels::NTuple{QD}, legdir::Tuple{Int, Int}) where {QD}
    nin = legdir[1]
    upsp = Tuple(qlabels[i] for i in 1:nin)
    dnsp = Tuple(qlabels[i] for i in nin+1:QD)
    return upsp, dnsp
end

function _svd_to_cgtidx(cgp::NTuple{QD, Int}, lidxs) where {QD}
    return Tuple(Int[cgp[l] for l in lidxs])
end

function _svd_abelian_intermediate_q(::Type{S}, 
    qlabels::NTuple{QD}, 
    legdir::Tuple{Int, Int}, 
    left_legs_canon::NTuple{L, Int}) where {S<:AbelianSymm, QD, L}
    nin = legdir[1]
    leftset = Set(left_legs_canon)

    merged = _svd_stable_sort_tuple((
        Tuple(qlabels[i] for i in 1:nin if i in leftset)...,
        Tuple(get_dualq(S, qlabels[i]) for i in nin+1:QD if i in leftset)...,
    ))
    outcomes = combine_qlabels(S, merged)
    @assert length(outcomes) == 1
    return outcomes[1][1]
end

function _svd_cgt_split_spaces(qlabels::NTuple{QD},
    legdir::Tuple{Int, Int},
    left_legs_canon,
    right_legs_canon) where {QD}

    nin = legdir[1]
    QLabel = eltype(qlabels)
    left_up = QLabel[]
    left_dn = QLabel[]
    right_up = QLabel[]
    right_dn = QLabel[]

    for l in left_legs_canon
        if l <= nin push!(left_up, qlabels[l])
        else push!(left_dn, qlabels[l]) end
    end

    for l in right_legs_canon
        if l <= nin push!(right_up, qlabels[l])
        else push!(right_dn, qlabels[l]) end
    end

    left_spaces = (Tuple(left_up), Tuple(left_dn))
    right_spaces = (Tuple(right_up), Tuple(right_dn))
    return left_spaces, right_spaces
end

function _svd_cgt_split_spaces(qlabels::NTuple{QD}, cgp::NTuple{QD, Int},
                               legdir::Tuple{Int, Int}, left_legs_canon) where {QD}
    leftset = Set(left_legs_canon)
    right_legs_canon = Tuple(cgp[l] for l in 1:QD if cgp[l] ∉ leftset)
    return _svd_cgt_split_spaces(qlabels, legdir, left_legs_canon, right_legs_canon)
end

function _get_svd_cgt_split_blocks(S, qlabels::NTuple{QD}, wmat::AbstractMatrix{Float64},
                                   cgp::NTuple{QD, Int}, legdir::Tuple{Int, Int},
                                   left_legs, right_legs, physical_key,
                                   cgtsvd_cache = nothing) where {QD}
    left_legs_canon = _svd_to_cgtidx(cgp, left_legs)
    right_legs_canon = _svd_to_cgtidx(cgp, right_legs)
    NZ = length(qlabels[1])
    BlockInfo = _SVDCGTBlockInfo{NZ}

    if isabelian(S)
        q = _svd_abelian_intermediate_q(S, qlabels, legdir, left_legs_canon)
        coeffs = reshape(copy(wmat), 1, 1, size(wmat, 2))
        _is_zero_array(coeffs) && return BlockInfo[]
        return [BlockInfo(q, 1, 1, coeffs)]
    end

    upsp, dnsp = _svd_cgt_updn(qlabels, legdir)
    cgtsvd = _svd_get_cgtsvd(S, physical_key, upsp, dnsp, left_legs_canon, cgtsvd_cache)

    blocks = BlockInfo[]
    if cgtsvd isa LurCGT.CGTSVD
        coeff_split = cgtsvd.svd_arr * wmat
        offset = 1
        for (q, omL, omR) in cgtsvd.bond_sps
            width = omL * omR
            coeffs = reshape(coeff_split[offset:offset+width-1, :], omL, omR, size(coeff_split, 2))
            !_is_zero_array(coeffs) && push!(blocks, BlockInfo(q, omL, omR, coeffs))
            offset += width
        end
        @assert offset == size(coeff_split, 1) + 1
    else
        q = zero_qlabels((S,))[1]
        om = size(wmat, 1)
        omL, omR = cgtsvd ? (1, om) : (om, 1)
        coeffs = reshape(wmat, omL, omR, size(wmat, 2))
        push!(blocks, BlockInfo(q, omL, omR, coeffs))
    end
    return blocks
end

function _get_svd_cgt_split_blocks(S, qlabels::NTuple{QD}, wmat::AbstractMatrix{Float64},
                                   cgp::NTuple{QD, Int}, legdir::Tuple{Int, Int},
                                   left_legs) where {QD}
    leftset = Set(left_legs)
    right_legs = Tuple(l for l in 1:QD if l ∉ leftset)
    physical_key = ntuple(i -> qlabels[cgp[i]], Val(QD))
    return _get_svd_cgt_split_blocks(S, qlabels, wmat, cgp, legdir, left_legs, right_legs, physical_key)
end

@inline _svd_get_cgtsvd(::Type{S}, physical_key, upsp, dnsp, left_legs_canon, cache) where {S<:AbelianSymm} =
    getNsave_CGTSVD(S, upsp, dnsp, left_legs_canon; save=true)

function _svd_get_cgtsvd(::Type{S}, physical_key, upsp, dnsp, left_legs_canon, cache) where {S<:NonabelianSymm}
    cache === nothing && return getNsave_CGTSVD(S, upsp, dnsp, left_legs_canon; save=true)
    if haskey(cache, physical_key)
        return cache[physical_key]
    end
    cgtsvd = getNsave_CGTSVD(S, upsp, dnsp, left_legs_canon; save=true)
    cache[physical_key] = cgtsvd
    return cgtsvd
end

_cgtsvd_cache_key_type(::Type{QT}, ::Val{n}, ::Val{QD}) where {QT, n, QD} =
    NTuple{QD, fieldtype(QT, n)}

@generated function _svd_cgtsvd_key_tuple_type(::Type{PS}, ::Type{QT}, ::Val{QD}) where {PS<:ProductSymm, QT, QD}
    M = _wmat_tuple_width(PS)
    key_types = Type[]
    for slot in 1:M
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        push!(key_types, _cgtsvd_cache_key_type(QT, Val(n), Val(QD)))
    end
    return :(Tuple{$(key_types...)})
end

@generated function _svd_cgtsvd_row_keys(::Type{PS},
                                         ::Type{QT},
                                         q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                         sector_index::Int,
                                         ::Val{M}) where {T, QD, N, RD, QT, PS<:ProductSymm, M, RMT}
    exprs = Expr[]
    for slot in 1:M
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        push!(exprs, :(_svd_physical_qlabel_key(QT, q, sector_index, Val($n))))
    end
    return Expr(:tuple, exprs...)
end

@generated function _cgtsvd_cache_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    1 <= n <= length(syms) || return :(throw(BoundsError(product_symms(PS), $n)))
    syms[n] <: AbelianSymm && return :(nothing)

    target = syms[n]
    slot = 0
    seen = Type[]
    for S in syms
        S <: AbelianSymm && continue
        if !(S in seen)
            push!(seen, S)
            slot += 1
        end
        S == target && return :($slot)
    end
    return :(throw(ArgumentError("symmetry index $n has no CGTSVD cache")))
end

@inline _cgtsvd_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:AbelianSymm, PS<:ProductSymm, n} = nothing
@inline function _cgtsvd_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:NonabelianSymm, PS<:ProductSymm, n}
    slot = _cgtsvd_cache_slot(PS, Val(n))
    slot === nothing &&
        throw(ArgumentError("symmetry index $n is Abelian and has no CGTSVD cache"))
    return caches[slot]
end

@inline _cgtsvd_cache_for(caches::CT, ::Type{S}, ::Type{PS}, ::Val{n}) where {CT<:Tuple, S, PS<:ProductSymm, n} =
    _cgtsvd_cache_for(S, caches, PS, Val(n))

@inline _irrepdim_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:AbelianSymm, PS<:ProductSymm, n} = nothing
@inline function _irrepdim_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:NonabelianSymm, PS<:ProductSymm, n}
    slot = _cgtsvd_cache_slot(PS, Val(n))
    slot === nothing &&
        throw(ArgumentError("symmetry index $n is Abelian and has no irrep-dimension cache"))
    return caches[slot]
end

@inline _irrepdim_cache_for(caches::CT, ::Type{S}, ::Type{PS}, ::Val{n}) where {CT<:Tuple, S, PS<:ProductSymm, n} =
    _irrepdim_cache_for(S, caches, PS, Val(n))

@generated function _new_cgtsvd_caches(::Type{PS}, ::Type{QT}, ::Val{QD}) where {PS<:ProductSymm, QT, QD}
    syms = product_symms(PS)
    seen = Type[]
    exprs = Expr[]
    for (n, S) in pairs(syms)
        S <: AbelianSymm && continue
        S in seen && continue
        push!(seen, S)
        Key = _cgtsvd_cache_key_type(QT, Val(n), Val(QD))
        # LurCGT standardization can change U/D/L or collapse to a Boolean
        # all-left/all-right shortcut, so the value type is keyed by symmetry.
        Value = Union{Bool, LurCGT.CGTSVD{S}}
        Cache = Dict{Key, Value}
        push!(exprs, :($Cache()))
    end
    return Expr(:tuple, exprs...)
end

_new_cgtsvd_caches(::TLArray{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} =
    _new_cgtsvd_caches(PS, QT, Val(QD))

@generated function _new_irrepdim_caches(::Type{PS}, ::Type{QT}) where {PS<:ProductSymm, QT}
    syms = product_symms(PS)
    seen = Type[]
    exprs = Expr[]
    for (n, S) in pairs(syms)
        S <: AbelianSymm && continue
        S in seen && continue
        push!(seen, S)
        QTN = fieldtype(QT, n)
        Cache = Dict{QTN, Int}
        push!(exprs, :($Cache()))
    end
    return Expr(:tuple, exprs...)
end

_new_irrepdim_caches(::TLArray{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} =
    _new_irrepdim_caches(PS, QT)

@inline _svd_irrep_dimension(::Type{S}, qlabel, cache) where {S<:AbelianSymm} =
    dimension(S, qlabel)

function _svd_irrep_dimension(::Type{S}, qlabel, cache::Dict{QTN, Int}) where {S<:NonabelianSymm, QTN}
    dim = get(cache, qlabel, 0)
    dim != 0 && return dim
    dim = dimension(S, qlabel)
    cache[qlabel] = dim
    return dim
end

@generated function _fill_irrepdim_caches_for_sector!(irrepdim_caches::CT,
                                                      sector::QT,
                                                      ::Type{PS}) where {CT<:Tuple, QT, PS<:ProductSymm}
    syms = product_symms(PS)
    exprs = Expr[]
    for n in eachindex(syms)
        S = syms[n]
        S <: AbelianSymm && continue
        push!(exprs, quote
            qlabel = sector[$n]
            cache = _irrepdim_cache_for(irrepdim_caches, $S, PS, Val($n))
            if !haskey(cache, qlabel)
                cache[qlabel] = dimension($S, qlabel)
            end
        end)
    end
    push!(exprs, :(return irrepdim_caches))
    return Expr(:block, exprs...)
end

function _fill_irrepdim_caches!(irrepdim_caches::CT,
                                class_metadata::AbstractVector,
                                ::Type{PS}) where {CT<:Tuple, PS<:ProductSymm}
    for metadata in class_metadata
        _fill_irrepdim_caches_for_sector!(irrepdim_caches, metadata.sector, PS)
    end
    return irrepdim_caches
end

@inline _svd_cgtsvd_split_count(cgtsvd::LurCGT.CGTSVD) = length(cgtsvd.bond_sps)
@inline _svd_cgtsvd_split_count(::Bool) = 1

function _count_svd_symmetry_splits(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                    sector_index::Int,
                                    left_legs::NTuple{L, Int},
                                    cgtsvd_caches::CT,
                                    ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, L, CT<:Tuple, n}
    S = nth_symm(PS, Val(n))
    isabelian(S) && return 1
    physical_key = _svd_physical_qlabel_key(QT, q, sector_index, Val(n))
    upsp, dnsp, left_legs_canon =
        _svd_physical_key_split_args(S, physical_key, q.inds, left_legs)
    cache = _cgtsvd_cache_for(cgtsvd_caches, S, PS, Val(n))
    cgtsvd = _svd_get_cgtsvd(S, physical_key, upsp, dnsp, left_legs_canon, cache)
    return _svd_cgtsvd_split_count(cgtsvd)
end

function _count_svd_sector_product_splits(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                          sector_index::Int,
                                          left_legs::NTuple{L, Int},
                                          cgtsvd_caches::CT,
                                          ::Val{N}) where {T, QD, N, RD, QT, PS, M, RMT, L, CT<:Tuple}
    count = 1
    for n in 1:N
        count *= _count_svd_symmetry_splits(
            q, sector_index, left_legs, cgtsvd_caches, Val(n))
    end
    return count
end

function _prepare_svd_counted_preassembly(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                          left_legs::NTuple{L, Int},
                                          ::Val{N}) where {T, QD, N, RD, QT, PS, M, RMT, L}
    cgtsvd_caches = _new_cgtsvd_caches(q)
    KeyTuple = _svd_cgtsvd_key_tuple_type(PS, QT, Val(QD))
    nactive = count(sector_index -> !q.iszero[sector_index], sector_slots(q))
    active_sector_indices = Vector{Int}(undef, nactive)
    keys_by_row = Vector{KeyTuple}(undef, nactive)
    sector_row_ranges = Vector{UnitRange{Int}}(undef, nactive)

    total_rows = 0
    active_position = 0
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        active_position += 1
        active_sector_indices[active_position] = sector_index
        keys_by_row[active_position] = _svd_cgtsvd_row_keys(PS, QT, q, sector_index, Val(M))
        nrows = _count_svd_sector_product_splits(
            q, sector_index, left_legs, cgtsvd_caches, Val(N))
        sector_row_ranges[active_position] = total_rows + 1:total_rows + nrows
        total_rows += nrows
    end
    @assert active_position == nactive
    return cgtsvd_caches, active_sector_indices, keys_by_row, sector_row_ranges, total_rows
end

@inline _svd_unit_isometry() = [1.0;;]

function _reduce_svd_cgt_split(block,
                               sector_index::Int,
                               left_signature::NTuple{L, QTN},
                               right_signature::NTuple{R, QTN};
                               tol::Float64 = 1e-12) where {L, R, QTN}
    coeffs = block.coeffs
    if size(coeffs, 1) == 1
        left_iso = _svd_unit_isometry()
        left_reduced = coeffs
    else
        left_iso, left_reduced, _ = svd_leg(coeffs, 1; cutoff=tol)
        size(left_iso, 2) == 0 && return nothing
    end

    if size(left_reduced, 2) == 1
        right_iso = _svd_unit_isometry()
        core = left_reduced
    else
        right_iso, core, _ = svd_leg(left_reduced, 2; cutoff=tol)
        size(right_iso, 2) == 0 && return nothing
    end

    return _SVDSymmetrySplit{L, R, QTN}(
        sector_index, block.q, left_signature, right_signature,
        left_iso, right_iso, core)
end

# Concatenate w-matrices, so the element type is always Float64
# TODO: When GPU support is added, other version should be implemented
function _copy_hcat_mats(mats::Vector{<:AbstractMatrix{Float64}}) 
    nsectors = size(first(mats), 1)
    ncols = sum(size(mat, 2) for mat in mats)
    concat = Matrix{Float64}(undef, nsectors, ncols)

    col = 1
    @views for mat in mats
        width = size(mat, 2)
        copyto!(view(concat, :, col:col+width-1), mat)
        col += width
    end
    @assert col == ncols + 1
    return concat
end

function _qr_shared_isometry(mats::Vector{AW}; tol::Float64 = 1e-12
    ) where {AW<:AbstractMatrix{Float64}}
    nsectors = size(first(mats), 1)
    @assert all(size(mat, 1) == nsectors for mat in mats) "_qr_shared_isometry requires a common sector dimension"

    if nsectors == 1
        Q = similar(first(mats), Float64, (1, 1))
        fill!(Q, 0.0)
        Q[1, 1] = 1.0
        return Q, mats
    end

    concat = _copy_hcat_mats(mats)
    F = qr(concat)
    Qfull = Matrix(F.Q)
    Rfull = Matrix(F.R)

    max_norm = 0.0
    @inbounds for i in axes(Rfull, 1)
        nrm_sq = 0.0
        for j in axes(Rfull, 2)
            nrm_sq += abs2(Rfull[i, j])
        end
        max_norm = max(max_norm, sqrt(nrm_sq))
    end
    max_norm == 0.0 && return nothing
    threshold = tol * max_norm
    used = 1
    @inbounds for i in axes(Rfull, 1)
        nrm_sq = 0.0
        for j in axes(Rfull, 2)
            nrm_sq += abs2(Rfull[i, j])
        end
        sqrt(nrm_sq) > threshold && (used = i)
    end
    Q = Qfull[:, 1:used]
    R = Rfull[1:used, :]

    factors = Vector{Matrix{Float64}}(undef, length(mats))
    col = 0
    for (i, mat) in pairs(mats)
        width = size(mat, 2)
        factors[i] = R[:, col+1:col+width]
        col += width
    end
    @assert col == size(concat, 2)

    return Q, factors
end

function _qr_shared_isometry(tensors::Vector{AW}; tol::Float64 = 1e-12
    ) where {AW<:AbstractArray{Float64, 3}}
    nsectors = size(first(tensors), 1)
    @assert all(size(tensor, 1) == nsectors for tensor in tensors) "_qr_shared_isometry requires a common sector dimension"

    mats = Vector{Matrix{Float64}}(undef, length(tensors))
    for (i, tensor) in pairs(tensors)
        d2, d3 = size(tensor, 2), size(tensor, 3)
        mats[i] = reshape(tensor, nsectors, d2 * d3)
    end

    shared = _qr_shared_isometry(mats; tol=tol)
    isnothing(shared) && return nothing
    Q, mat_factors = shared

    factors = Vector{Array{Float64, 3}}(undef, length(tensors))
    for i in eachindex(tensors)
        d2, d3 = size(tensors[i], 2), size(tensors[i], 3)
        factors[i] = reshape(mat_factors[i], size(mat_factors[i], 1), d2, d3)
    end

    return Q, factors
end

function _sort_svd_splits_by_sector!(splits_by_symm)
    for splits in splits_by_symm
        sort!(splits; by = split -> (split.sector_index, split.q), alg=MergeSort)
    end
    return splits_by_symm
end

@inline _svd_trivial_iso() = ones(Float64, 1, 1)
@inline _svd_trivial_core() = reshape(ones(Float64, 1), 1, 1, 1)

@generated function _svd_payload_tuples(::Type{PS}, sector_splits::ST) where {PS<:ProductSymm, ST<:Tuple}
    M = _wmat_tuple_width(PS)
    N = fieldcount(ST)
    left = Expr[]
    right = Expr[]
    cores = Expr[]
    for slot in 1:M
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        1 <= n <= N || return :(throw(BoundsError(sector_splits, $n)))
        push!(left, :(sector_splits[$n].left_iso))
        push!(right, :(sector_splits[$n].right_iso))
        push!(cores, :(sector_splits[$n].core))
    end
    return :(($(left...),), ($(right...),), ($(cores...),))
end

function _compact_svd_rows_by_valid!(rows::Vector{Row},
                                     left_isos::Vector{NTuple{M, Matrix{Float64}}},
                                     right_isos::Vector{NTuple{M, Matrix{Float64}}},
                                     cores::Vector{NTuple{M, Array{Float64, 3}}},
                                     valid_rows::BitVector) where {Row, M}
    write = 1
    for read in eachindex(rows)
        valid_rows[read] || continue
        if write != read
            rows[write] = rows[read]
            left_isos[write] = left_isos[read]
            right_isos[write] = right_isos[read]
            cores[write] = cores[read]
        end
        write += 1
    end
    last = write - 1
    resize!(rows, last)
    resize!(left_isos, last)
    resize!(right_isos, last)
    resize!(cores, last)
    return rows, left_isos, right_isos, cores
end

function _get_svd_split_rows(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                             splits_by_symm::Tuple{Vararg{AbstractVector, N}},
                             sector_indices::Vector{Int},
                             sector_row_ranges::Vector{UnitRange{Int}},
                             left_legs::NTuple{L, Int},
                             right_legs::NTuple{R, Int},
                             ::Val{N},
                             total_rows::Int = 0) where {T, QD, N, RD, QT, PS, M, RMT, L, R}
    _sort_svd_splits_by_sector!(splits_by_symm)
    Row = _SVDSplitRow{L, R, QT}
    rows = Vector{Row}(undef, total_rows)
    left_isos = Vector{NTuple{M, Matrix{Float64}}}(undef, total_rows)
    right_isos = Vector{NTuple{M, Matrix{Float64}}}(undef, total_rows)
    cores = Vector{NTuple{M, Array{Float64, 3}}}(undef, total_rows)
    (isempty(sector_indices) || any(isempty, splits_by_symm)) &&
        return Row[], NTuple{M, Matrix{Float64}}[],
               NTuple{M, Matrix{Float64}}[], NTuple{M, Array{Float64, 3}}[]
    valid_rows = falses(total_rows)

    nslots = maximum(sector_indices; init=0)
    choices_by_sector = ntuple(n -> [eltype(splits_by_symm[n])[] for _ in 1:nslots], N)
    for n in 1:N
        for split in splits_by_symm[n]
            push!(choices_by_sector[n][split.sector_index], split)
        end
    end

    for (active_position, ri) in pairs(sector_indices)
        next_row = first(sector_row_ranges[active_position])
        if !any(n -> isempty(choices_by_sector[n][ri]), 1:N)
            physical_key = _svd_physical_product_key(QT, q, ri)
            left_signature, right_signature =
                _svd_physical_key_side_signatures(physical_key, left_legs, right_legs)
            choices = ntuple(n -> choices_by_sector[n][ri], N)
            for splits in Iterators.product(choices...)
                sector_splits = Tuple(splits)
                sector = ntuple(n -> sector_splits[n].q, Val(N))::QT
                liso, riso, core = _svd_payload_tuples(PS, sector_splits)
                @assert next_row in sector_row_ranges[active_position]
                rows[next_row] = Row(ri, sector, left_signature, right_signature)
                left_isos[next_row] = liso
                right_isos[next_row] = riso
                cores[next_row] = core
                valid_rows[next_row] = true
                next_row += 1
            end
        end
    end

    _compact_svd_rows_by_valid!(rows, left_isos, right_isos, cores, valid_rows)

    perm = sortperm(eachindex(rows);
        by = i -> (rows[i].q, rows[i].left_signature, rows[i].right_signature,
                   rows[i].sector_index),
        alg=MergeSort)
    return rows[perm], left_isos[perm], right_isos[perm], cores[perm]
end

function _get_svd_split_rows(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                             splits_by_symm::Tuple{Vararg{AbstractVector, N}},
                             sector_indices::Vector{Int},
                             left_legs::NTuple{L, Int},
                             right_legs::NTuple{R, Int},
                             ::Val{N},
                             total_rows::Int = 0) where {T, QD, N, RD, QT, PS, M, RMT, L, R}
    nslots = maximum(sector_indices; init=0)
    counts_by_sector = fill(1, nslots)
    for ri in sector_indices
        counts_by_sector[ri] = 1
    end
    for n in 1:N
        counts_n = zeros(Int, nslots)
        for split in splits_by_symm[n]
            counts_n[split.sector_index] += 1
        end
        for ri in sector_indices
            counts_by_sector[ri] *= counts_n[ri]
        end
    end

    sector_row_ranges = Vector{UnitRange{Int}}(undef, length(sector_indices))
    offset = 0
    for (i, ri) in pairs(sector_indices)
        nrows = counts_by_sector[ri]
        sector_row_ranges[i] = offset + 1:offset + nrows
        offset += nrows
    end
    return _get_svd_split_rows(
        q, splits_by_symm, sector_indices, sector_row_ranges,
        left_legs, right_legs, Val(N), max(total_rows, offset))
end

@generated function _fill_svd_split_row_products!(rows::Vector{Row},
                                                  left_isos::Vector{NTuple{M, Matrix{Float64}}},
                                                  right_isos::Vector{NTuple{M, Matrix{Float64}}},
                                                  cores::Vector{NTuple{M, Array{Float64, 3}}},
                                                  valid_rows::BitVector,
                                                  next_row::Int,
                                                  row_range::UnitRange{Int},
                                                  sector_index::Int,
                                                  left_signature::NTuple{L, QT},
                                                  right_signature::NTuple{R, QT},
                                                  symm_splits::ST,
                                                  ::Type{PS},
                                                  ::Type{QT},
                                                  ::Val{N}) where {Row, M, L, R, QT, ST<:Tuple, PS<:ProductSymm, N}
    idxs = [gensym(:i) for _ in 1:N]
    split_exprs = [:(symm_splits[$n][$(idxs[n])]) for n in 1:N]
    q_exprs = [:(sector_splits[$n].q) for n in 1:N]

    body = quote
        sector_splits = ($(split_exprs...),)
        sector = ($(q_exprs...),)::QT
        liso, riso, core = _svd_payload_tuples(PS, sector_splits)
        @assert next_row in row_range
        rows[next_row] = Row(sector_index, sector, left_signature, right_signature)
        left_isos[next_row] = liso
        right_isos[next_row] = riso
        cores[next_row] = core
        valid_rows[next_row] = true
        next_row += 1
    end

    for n in N:-1:1
        body = quote
            for $(idxs[n]) in eachindex(symm_splits[$n])
                $body
            end
        end
    end

    return quote
        $body
        return next_row
    end
end

function _get_svd_split_rows(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                             active_sector_indices::Vector{Int},
                             keys_by_row,
                             sector_row_ranges::Vector{UnitRange{Int}},
                             cgtsvd_caches::CT,
                             left_legs::NTuple{L, Int},
                             right_legs::NTuple{R, Int},
                             ::Val{N},
                             total_rows::Int;
                             tol::Float64 = 1e-12) where {T, QD, N, RD, QT, PS, M, RMT, L, R, CT<:Tuple}
    Row = _SVDSplitRow{L, R, QT}
    rows = Vector{Row}(undef, total_rows)
    left_isos = Vector{NTuple{M, Matrix{Float64}}}(undef, total_rows)
    right_isos = Vector{NTuple{M, Matrix{Float64}}}(undef, total_rows)
    cores = Vector{NTuple{M, Array{Float64, 3}}}(undef, total_rows)
    isempty(active_sector_indices) &&
        return Row[], NTuple{M, Matrix{Float64}}[],
               NTuple{M, Matrix{Float64}}[], NTuple{M, Array{Float64, 3}}[]
    valid_rows = falses(total_rows)

    for (active_position, ri) in pairs(active_sector_indices)
        next_row = first(sector_row_ranges[active_position])
        symm_splits = ntuple(Val(N)) do n
            _get_svd_symmetry_splits(
                q, ri, active_position, keys_by_row, cgtsvd_caches,
                left_legs, right_legs, Val(n); tol=tol)
        end

        if !any(isempty, symm_splits)
            physical_key = _svd_physical_product_key(QT, q, ri)
            left_signature, right_signature =
                _svd_physical_key_side_signatures(physical_key, left_legs, right_legs)
            next_row = _fill_svd_split_row_products!(
                rows, left_isos, right_isos, cores, valid_rows,
                next_row, sector_row_ranges[active_position], ri,
                left_signature, right_signature, symm_splits, PS, QT, Val(N))
        end
    end

    _compact_svd_rows_by_valid!(rows, left_isos, right_isos, cores, valid_rows)

    perm = sortperm(eachindex(rows);
        by = i -> (rows[i].q, rows[i].left_signature, rows[i].right_signature,
                   rows[i].sector_index),
        alg=MergeSort)
    return rows[perm], left_isos[perm], right_isos[perm], cores[perm]
end

@inline function _svd_tuple_replace(t::NTuple{M, T}, value::T, ::Val{slot}) where {M, T, slot}
    return ntuple(i -> i == slot ? value : t[i], Val(M))
end

@inline _svd_side_qs(signature::NTuple{L, QT}, ::Val{n}) where {L, QT, n} =
    ntuple(i -> signature[i][n], Val(L))

@inline _svd_payload_left_share_key(row, ::Val{n}) where {n} =
    _svd_side_qs(row.left_signature, Val(n))

@inline _svd_payload_right_share_key(row, ::Val{n}) where {n} =
    _svd_side_qs(row.right_signature, Val(n))

function _sort_svd_payloads_by_q!(rows::Vector{Row},
                                  left_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                  right_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                  cores::Vector{NTuple{M, Array{Float64, 3}}}) where {Row, M}
    perm = sortperm(eachindex(rows); by = i -> rows[i].q, alg=MergeSort)
    rows[:] = rows[perm]
    left_payloads[:] = left_payloads[perm]
    right_payloads[:] = right_payloads[perm]
    cores[:] = cores[perm]
    return rows, left_payloads, right_payloads, cores
end

function _share_svd_left_payload_isometries!(rows::Vector{Row},
                                             left_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                             cores::Vector{NTuple{M, Array{Float64, 3}}},
                                             row_range::UnitRange{Int},
                                             ::Val{slot},
                                             ::Val{n},
                                             tol::Float64) where {Row, M, slot, n}
    isempty(row_range) && return rows
    vslot = Val(slot)
    vn = Val(n)
    inds = collect(row_range)
    sort!(inds; by = i -> _svd_payload_left_share_key(rows[i], vn), alg=MergeSort)

    pos = 1
    while pos <= length(inds)
        current_key = _svd_payload_left_share_key(rows[inds[pos]], vn)
        nextpos = pos
        while nextpos <= length(inds) &&
              _svd_payload_left_share_key(rows[inds[nextpos]], vn) == current_key
            nextpos += 1
        end

        group = pos:nextpos-1
        if length(group) > 1
            mats = Matrix{Float64}[left_payloads[inds[j]][slot] for j in group]
            shared = _qr_shared_isometry(mats; tol=tol)
            @assert !isnothing(shared) "_qr_shared_isometry returned zero rank for SVD left payloads"
            common_iso, factors = shared
            for (j, factor) in zip(group, factors)
                row_index = inds[j]
                left_payloads[row_index] =
                    _svd_tuple_replace(left_payloads[row_index], common_iso, vslot)
                cores[row_index] = _svd_tuple_replace(
                    cores[row_index],
                    _contract_om_axis(cores[row_index][slot], factor, 1),
                    vslot)
            end
        end
        pos = nextpos
    end
    return rows
end

function _share_svd_right_payload_isometries!(rows::Vector{Row},
                                              right_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                              cores::Vector{NTuple{M, Array{Float64, 3}}},
                                              row_range::UnitRange{Int},
                                              ::Val{slot},
                                              ::Val{n},
                                              tol::Float64) where {Row, M, slot, n}
    isempty(row_range) && return rows
    vslot = Val(slot)
    vn = Val(n)
    inds = collect(row_range)
    sort!(inds; by = i -> _svd_payload_right_share_key(rows[i], vn), alg=MergeSort)

    pos = 1
    while pos <= length(inds)
        current_key = _svd_payload_right_share_key(rows[inds[pos]], vn)
        nextpos = pos
        while nextpos <= length(inds) &&
              _svd_payload_right_share_key(rows[inds[nextpos]], vn) == current_key
            nextpos += 1
        end

        group = pos:nextpos-1
        if length(group) > 1
            mats = Matrix{Float64}[right_payloads[inds[j]][slot] for j in group]
            shared = _qr_shared_isometry(mats; tol=tol)
            @assert !isnothing(shared) "_qr_shared_isometry returned zero rank for SVD right payloads"
            common_iso, factors = shared
            for (j, factor) in zip(group, factors)
                row_index = inds[j]
                right_payloads[row_index] =
                    _svd_tuple_replace(right_payloads[row_index], common_iso, vslot)
                cores[row_index] = _svd_tuple_replace(
                    cores[row_index],
                    _contract_om_axis(cores[row_index][slot], factor, 2),
                    vslot)
            end
        end
        pos = nextpos
    end
    return rows
end

@generated function _share_svd_payload_slot_isometries!(rows::Vector{Row},
                                                        left_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                                        right_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                                        cores::Vector{NTuple{M, Array{Float64, 3}}},
                                                        ::Type{PS},
                                                        row_range::UnitRange{Int},
                                                        tol::Float64) where {Row, M, PS<:ProductSymm}
    exprs = Expr[]
    for slot in 1:M
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        push!(exprs, :(_share_svd_left_payload_isometries!(
            rows, left_payloads, cores, row_range, Val($slot), Val($n), tol)))
        push!(exprs, :(_share_svd_right_payload_isometries!(
            rows, right_payloads, cores, row_range, Val($slot), Val($n), tol)))
    end
    push!(exprs, :(return rows, left_payloads, right_payloads, cores))
    return Expr(:block, exprs...)
end

function _share_svd_payload_isometries!(rows::Vector{Row},
                                        left_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                        right_payloads::Vector{NTuple{M, Matrix{Float64}}},
                                        cores::Vector{NTuple{M, Array{Float64, 3}}},
                                        ::Type{PS};
                                        tol::Float64 = 1e-12) where {Row, M, PS<:ProductSymm}
    isempty(rows) && return rows, left_payloads, right_payloads, cores
    _sort_svd_payloads_by_q!(rows, left_payloads, right_payloads, cores)

    pos = 1
    while pos <= length(rows)
        q = rows[pos].q
        nextpos = pos + 1
        while nextpos <= length(rows) && rows[nextpos].q == q
            nextpos += 1
        end

        row_range = pos:nextpos-1
        _share_svd_payload_slot_isometries!(
            rows, left_payloads, right_payloads, cores, PS, row_range, tol)
        pos = nextpos
    end
    return rows, left_payloads, right_payloads, cores
end

function _sort_svd_split_rows_by_class!(::Type{QT},
    rows::Vector{Row},
    left_payloads::Vector{NTuple{M, Matrix{Float64}}},
    right_payloads::Vector{NTuple{M, Matrix{Float64}}},
    core_payloads::Vector{NTuple{M, Array{Float64, 3}}},
    ::Val{L},
    ::Val{R}) where {QT, Row, M, L, R}
    LeftSig = NTuple{L, QT}
    RightSig = NTuple{R, QT}
    isempty(rows) && return UnitRange{Int}[]

    perm = Int[]
    class_ranges = UnitRange{Int}[]

    pos = 1
    while pos <= length(rows)
        q = rows[pos].q
        nextpos = pos
        while nextpos <= length(rows) && rows[nextpos].q == q
            nextpos += 1
        end

        row_range = pos:nextpos-1
        sectors = sort!(unique([rows[i].sector_index for i in row_range]); alg=MergeSort)
        rows_by_sector = Dict{Int, Vector{Int}}()
        left_groups = Dict{LeftSig, Vector{Int}}()
        right_groups = Dict{RightSig, Vector{Int}}()
        for i in row_range
            row = rows[i]
            push!(get!(rows_by_sector, row.sector_index, Int[]), i)
            push!(get!(left_groups, row.left_signature, Int[]), row.sector_index)
            push!(get!(right_groups, row.right_signature, Int[]), row.sector_index)
        end
        for group in values(left_groups)
            sort!(group; alg=MergeSort)
            unique!(group)
        end
        for group in values(right_groups)
            sort!(group; alg=MergeSort)
            unique!(group)
        end

        unassigned = Set(sectors)
        classes = Vector{Vector{Int}}()
        for seed in sectors
            seed in unassigned || continue
            component = Int[seed]
            frontier = Int[seed]
            delete!(unassigned, seed)
            seen_left = Set{LeftSig}()
            seen_right = Set{RightSig}()

            while !isempty(frontier)
                ri = pop!(frontier)
                for row_index in rows_by_sector[ri]
                    row = rows[row_index]
                    if row.left_signature ∉ seen_left
                        push!(seen_left, row.left_signature)
                        for rj in left_groups[row.left_signature]
                            rj in unassigned || continue
                            delete!(unassigned, rj)
                            push!(frontier, rj)
                            push!(component, rj)
                        end
                    end
                    if row.right_signature ∉ seen_right
                        push!(seen_right, row.right_signature)
                        for rj in right_groups[row.right_signature]
                            rj in unassigned || continue
                            delete!(unassigned, rj)
                            push!(frontier, rj)
                            push!(component, rj)
                        end
                    end
                end
            end

            sort!(component; alg=MergeSort)
            class_rows = Int[]
            for ri in component
                append!(class_rows, rows_by_sector[ri])
            end
            sort!(class_rows; by = i -> (rows[i].sector_index, rows[i].left_signature,
                                         rows[i].right_signature), alg=MergeSort)
            push!(classes, class_rows)
        end
        sort!(classes; by = cls -> rows[cls[1]].sector_index, alg=MergeSort)
        for class_rows in classes
            start = length(perm) + 1
            append!(perm, class_rows)
            push!(class_ranges, start:length(perm))
        end
        pos = nextpos
    end

    @assert length(perm) == length(rows)
    rows[:] = rows[perm]
    left_payloads[:] = left_payloads[perm]
    right_payloads[:] = right_payloads[perm]
    core_payloads[:] = core_payloads[perm]
    return class_ranges
end

function _get_svd_sector_spaces(q::TLArray{T, QD, N, RD, QT},
    left_legs,
    right_legs) where {T, QD, N, RD, QT}
    active_indices = [sector_index for sector_index in sector_slots(q) if !q.iszero[sector_index]]
    sector_spaces = [begin
        split_spaces = ntuple(N) do n
            qlabels, cgp, _, legdir =
                _svd_symmetry_stored_leg_order(QT, q, sector_index, Val(n))
            left_legs_canon = _svd_to_cgtidx(cgp, left_legs)
            right_legs_canon = _svd_to_cgtidx(cgp, right_legs)
            _svd_cgt_split_spaces(qlabels, legdir, left_legs_canon, right_legs_canon)
        end
        (ntuple(n -> split_spaces[n][1], N), ntuple(n -> split_spaces[n][2], N))
    end for sector_index in active_indices]
    isempty(sector_spaces) && return first.(sector_spaces), last.(sector_spaces)

    left_values = first.(sector_spaces)
    right_values = last.(sector_spaces)
    nslots = maximum(sector_slots(q); init=0)
    left_by_slot = Vector{eltype(left_values)}(undef, nslots)
    right_by_slot = Vector{eltype(right_values)}(undef, nslots)
    for (i, sector_index) in pairs(active_indices)
        left_by_slot[sector_index] = left_values[i]
        right_by_slot[sector_index] = right_values[i]
    end
    return left_by_slot, right_by_slot
end

nth_symm(::Type{ProductSymm{Syms}}, ::Val{n}) where {Syms, n} = 
    product_symms(ProductSymm{Syms})[n]

function _get_svd_symmetry_splits(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                  sector_index::Int,
                                  active_position::Int,
                                  keys_by_row,
                                  cgtsvd_caches::CT,
                                  left_legs::NTuple{L, Int},
                                  right_legs::NTuple{R, Int},
                                  ::Val{n};
                                  tol::Float64 = 1e-12) where {T, QD, N, RD, QT, PS, M, RMT, L, R, CT<:Tuple, n}
    S = nth_symm(PS, Val(n))
    physical_key = if isabelian(S)
        _svd_physical_qlabel_key(QT, q, sector_index, Val(n))
    else
        keys_by_row[active_position][nonabelian_wmat_slot(PS, Val(n))]
    end
    QTN = typeof(physical_key[1])
    splits = Vector{_SVDSymmetrySplit{L, R, QTN}}()
    left_signature, right_signature =
        _svd_physical_key_side_signatures(physical_key, left_legs, right_legs)
    qlabels, cgp, _, legdir =
        _svd_symmetry_stored_leg_order(QT, q, sector_index, Val(n))
    cache = _cgtsvd_cache_for(cgtsvd_caches, S, productsymm(PS), Val(n))
    for block in _get_svd_cgt_split_blocks(S, qlabels, sector_wmat(q, sector_index, Val(n)),
                                           cgp, legdir, left_legs, right_legs, physical_key, cache)
        split = _reduce_svd_cgt_split(
            block, sector_index, left_signature, right_signature; tol=tol)
        isnothing(split) || push!(splits, split)
    end
    return splits
end

function _get_svd_cgt_split_sectors(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                    active_sector_indices::Vector{Int},
                                    keys_by_row,
                                    cgtsvd_caches::CT,
                                    left_legs::NTuple{L, Int},
                                    right_legs::NTuple{R, Int};
                                    tol::Float64 = 1e-12) where {T, QD, N, RD, QT, PS, M, RMT, L, R, CT<:Tuple}
    splits_by_symm = ntuple(Val(N)) do n
        QTN = fieldtype(QT, n)
        Vector{_SVDSymmetrySplit{L, R, QTN}}()
    end
    for (active_position, ri) in pairs(active_sector_indices)
        for n in 1:N
            append!(splits_by_symm[n], _get_svd_symmetry_splits(
                q, ri, active_position, keys_by_row, cgtsvd_caches,
                left_legs, right_legs, Val(n); tol=tol))
        end
    end
    return splits_by_symm
end

function _get_svd_cgt_split_sectors(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                    left_legs::NTuple{L, Int},
                                    right_legs::NTuple{R, Int};
                                    tol::Float64 = 1e-12) where {T, QD, N, RD, QT, PS, M, RMT, L, R}
    cgtsvd_caches, active_sector_indices, keys_by_row, _, _ =
        _prepare_svd_counted_preassembly(q, left_legs, Val(N))
    return _get_svd_cgt_split_sectors(
        q, active_sector_indices, keys_by_row, cgtsvd_caches,
        left_legs, right_legs; tol=tol)
end

@inline _svd_payload_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? nothing : nonabelian_wmat_slot(PS, Val(n))

@inline _svd_left_iso(payload::NTuple{M, Matrix{Float64}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _svd_trivial_iso() : payload[nonabelian_wmat_slot(PS, Val(n))]

@inline _svd_right_iso(payload::NTuple{M, Matrix{Float64}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _svd_trivial_iso() : payload[nonabelian_wmat_slot(PS, Val(n))]

@inline _svd_core(payload::NTuple{M, Array{Float64, 3}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _svd_trivial_core() : payload[nonabelian_wmat_slot(PS, Val(n))]

@generated function _svd_core_tuple(core_payload::CP,
                                    ::Type{PS},
                                    ::Val{N}) where {CP, PS<:ProductSymm, N}
    return Expr(:tuple, [:( _svd_core(core_payload, PS, Val($n)) ) for n in 1:N]...)
end

_svd_core_kron_matrix(::Tuple{}) = ones(Float64, 1, 1)   # N=0 (no symmetry): trivial 1x1 core
function _svd_core_kron_matrix(cores::NTuple{N, Array{Float64, 3}}) where {N}
    core = cores[1]
    mat = reshape(core, size(core, 1) * size(core, 2), size(core, 3))
    for n in 2:N
        core = cores[n]
        mat_n = reshape(core, size(core, 1) * size(core, 2), size(core, 3))
        mat = kron(mat_n, mat)
    end
    return mat
end

@generated function _svd_expanded_core_dims(rmt_size,
                                            cores::CT,
                                            ::Val{QD},
                                            ::Val{N}) where {CT<:Tuple, QD, N}
    exprs = Any[:(rmt_size[$i]) for i in 1:QD]
    for n in 1:N
        push!(exprs, :(size(cores[$n], 1)))
        push!(exprs, :(size(cores[$n], 2)))
    end
    return Expr(:tuple, exprs...)
end

@generated function _svd_expanded_class_perm(left_legs::NTuple{NL, Int},
                                             right_legs::NTuple{NR, Int},
                                             ::Val{QD},
                                             ::Val{N}) where {NL, NR, QD, N}
    exprs = Any[:(left_legs[$i]) for i in 1:NL]
    append!(exprs, [:(QD + 2 * $n - 1) for n in 1:N])
    append!(exprs, [:(right_legs[$i]) for i in 1:NR])
    append!(exprs, [:(QD + 2 * $n) for n in 1:N])
    return Expr(:tuple, exprs...)
end

function _svd_permutedims_to_buffer!(permute_buffer::Matrix{T},
                                     storage::Matrix{T},
                                     dims::NTuple{D, Int},
                                     perm::NTuple{D, Int}) where {T, D}
    data = reshape(view(storage, 1:prod(dims; init=1)), dims)
    output_size = prod(dims; init=1)
    @assert length(permute_buffer) >= output_size
    permute_view = reshape(view(permute_buffer, 1:output_size), ntuple(i -> dims[perm[i]], Val(D)))
    permute_view .= permutedims(data, perm)
    return permute_view
end

function _svd_permutedims_to_buffer!(permute_buffer::Matrix{Float64},
                                     storage::Matrix{Float64},
                                     dims::NTuple{D, Int},
                                     perm::NTuple{D, Int}) where {D}
    output_size = prod(dims; init=1)
    @assert length(permute_buffer) >= output_size
    permute_mat = reshape(view(permute_buffer, 1:output_size), output_size, 1)
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[dims[i] for i in 1:D]
    ccall((:dTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float64, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Float64, Ptr{Float64}, Ptr{Int32}, Cint, Cint),
          perm0, D, 1.0, storage, sizeA, C_NULL,
          0.0, permute_mat, C_NULL, max(1, Threads.nthreads()), 0)
    return reshape(view(permute_buffer, 1:output_size), ntuple(i -> dims[perm[i]], Val(D)))
end

function _svd_permutedims_to_buffer!(permute_buffer::Matrix{ComplexF64},
                                     storage::Matrix{ComplexF64},
                                     dims::NTuple{D, Int},
                                     perm::NTuple{D, Int}) where {D}
    output_size = prod(dims; init=1)
    @assert length(permute_buffer) >= output_size
    permute_mat = reshape(view(permute_buffer, 1:output_size), output_size, 1)
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[dims[i] for i in 1:D]
    ccall((:zTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF64, Cuchar, Ptr{ComplexF64}, Ptr{Int32}, Ptr{Int32},
           ComplexF64, Ptr{ComplexF64}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF64), false, storage, sizeA, C_NULL,
          zero(ComplexF64), permute_mat, C_NULL, max(1, Threads.nthreads()), 0)
    return reshape(view(permute_buffer, 1:output_size), ntuple(i -> dims[perm[i]], Val(D)))
end

function _svd_sector_class_matrix!(dest::AbstractMatrix{Tbuf},
                                   q::TLArray{T, QD, N, RD, QT, PS},
                                   sector_index::Int,
                                   left_payload,
                                   right_payload,
                                   core_payload,
                                   left_legs::NTuple{NL, Int},
                                   right_legs::NTuple{NR, Int},
                                   product_buffer::Matrix{Tbuf},
                                   permute_buffer::Matrix{Tbuf}) where {T, QD, N, RD, QT, PS, NL, NR, Tbuf}
    rmt = sector_rmt(q, sector_index)
    rmt_size = size(rmt)
    phys_dim = prod(rmt_size[i] for i in 1:QD)
    om_dim = prod(rmt_size[QD + n] for n in 1:N; init = 1)   # init=1: N=0 (no symmetry)
    cores = _svd_core_tuple(core_payload, PS, Val(N))
    core_mat = _svd_core_kron_matrix(cores)
    @assert size(core_mat, 2) == om_dim

    expanded_om_dim = size(core_mat, 1)
    product_size = phys_dim * expanded_om_dim
    @assert length(product_buffer) >= product_size
    product_mat = reshape(view(product_buffer, 1:product_size), phys_dim, expanded_om_dim)
    mul!(product_mat, reshape(rmt, phys_dim, om_dim), transpose(core_mat), one(Tbuf), zero(Tbuf))

    expanded_dims = _svd_expanded_core_dims(rmt_size, cores, Val(QD), Val(N))
    perm = _svd_expanded_class_perm(left_legs, right_legs, Val(QD), Val(N))
    left_dim = prod(rmt_size[leg] for leg in left_legs) *
               prod(size(_svd_left_iso(left_payload, PS, Val(n)), 2) for n in 1:N; init = 1)
    right_dim = prod(rmt_size[leg] for leg in right_legs) *
                prod(size(_svd_right_iso(right_payload, PS, Val(n)), 2) for n in 1:N; init = 1)
    @assert size(dest) == (left_dim, right_dim)
    permed = _svd_permutedims_to_buffer!(permute_buffer, product_buffer, expanded_dims, perm)
    dest .+= reshape(permed, left_dim, right_dim)
    return dest
end

@inline _svd_row_signature(row, ::Val{:left}) = row.left_signature
@inline _svd_row_signature(row, ::Val{:right}) = row.right_signature
@inline _svd_signature_field(::Val{:left}) = :left_signature
@inline _svd_signature_field(::Val{:right}) = :right_signature

function _svd_ordered_unique_split_rows(rows::Vector{Row},
                                        class_row_indices::UnitRange{Int},
                                        side) where {Row}
    Sig = fieldtype(Row, _svd_signature_field(side))
    ordered = Int[]
    seen = Set{Sig}()
    for row_index in class_row_indices
        row = rows[row_index]
        sig = _svd_row_signature(row, side)
        sig in seen && continue
        push!(seen, sig)
        push!(ordered, row_index)
    end
    return ordered
end

function _svd_class_side_infos(q::TLArray{T, QD, N, RD, QT, PS},
                               rows::Vector{Row},
                               class_row_indices::UnitRange{Int},
                               left_payloads,
                               right_payloads,
                               legs::NTuple{L, Int},
                               side) where {T, QD, N, RD, QT, PS, L, Row}
    ordered = _svd_ordered_unique_split_rows(rows, class_row_indices, side)
    Sig = fieldtype(Row, _svd_signature_field(side))
    Info = _SVDClassSideInfo{Sig, L, N}
    infos = Info[]
    ranges = Dict{Sig, UnitRange{Int}}()
    offset = 0

    for row_index in ordered
        row = rows[row_index]
        sig = _svd_row_signature(row, side)
        ri = row.sector_index
        rmt_size = size(sector_rmt(q, ri))
        phys_dims = ntuple(i -> rmt_size[legs[i]], L)
        om_dims = ntuple(Val(N)) do n
            side === Val(:left) ?
                size(_svd_left_iso(left_payloads[row_index], PS, Val(n)), 2) :
                size(_svd_right_iso(right_payloads[row_index], PS, Val(n)), 2)
        end
        block_size = prod(phys_dims; init=1) * prod(om_dims; init=1)
        block_range = offset + 1:offset + block_size
        push!(infos, Info(sig, row_index, ri, phys_dims, om_dims, block_range))
        ranges[sig] = block_range
        offset += block_size
    end

    return infos, ranges, offset
end

function _svd_build_side_cgt_metadata(source_qlabels::NTuple{QD},
                                      source_cgp::NTuple{QD, Int},
                                      source_legdir::Tuple{Int, Int},
                                      phys_legs::NTuple{L, Int},
                                      bond_q,
                                      bond_first::Bool,
                                      wmat::AbstractMatrix{Float64}) where {QD, L}
    stored_phys = sort!([(source_cgp[leg], leg) for leg in phys_legs]; by = first, alg=MergeSort)
    nin = source_legdir[1]

    Entry = Tuple{typeof(bond_q), Int}
    incoming = Entry[]
    outgoing = Entry[]
    if bond_first push!(outgoing, (bond_q, 0)) end
    for (stored_pos, leg) in stored_phys
        entry = (source_qlabels[stored_pos], leg)
        if stored_pos <= nin
            push!(incoming, entry)
        else
            push!(outgoing, entry)
        end
    end
    if !bond_first push!(outgoing, (bond_q, length(source_qlabels)+1)) end
    sort!(outgoing; by = first, alg=MergeSort)

    incoming_qlabels = Tuple(entry[1] for entry in incoming)
    outgoing_qlabels = Tuple(entry[1] for entry in outgoing)
    qlabels = (incoming_qlabels..., outgoing_qlabels...)
    legdir = (length(incoming), length(outgoing))

    source_to_stored = Dict{Int, Int}()
    for (stored_pos, (_, source)) in enumerate((incoming..., outgoing...))
        source_to_stored[source] = stored_pos
    end

    final_sources = bond_first ? (0, phys_legs...) : (phys_legs..., length(source_qlabels)+1)
    @assert issorted(final_sources)
    final_cgp = Tuple(source_to_stored[source] for source in final_sources)

    return (qlabels = qlabels, wmat = wmat, cgp = final_cgp, legdir = legdir)
end

function _svd_cgtsvd_class_metadata(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                    rows::Vector{Row},
                                    class_row_indices::UnitRange{Int},
                                    left_payloads,
                                    right_payloads,
                                    left_legs::NTuple{NL, Int},
                                    right_legs::NTuple{NR, Int}) where {T, QD, N, RD, QT, PS, M, RMT, NL, NR, Row}
    left_infos, left_ranges, total_left = _svd_class_side_infos(
        q, rows, class_row_indices, left_payloads, right_payloads, left_legs, Val(:left))

    right_infos, right_ranges, total_right = _svd_class_side_infos(
        q, rows, class_row_indices, left_payloads, right_payloads, right_legs, Val(:right))

    return _SVDCGTClassMetadata(
        rows[first(class_row_indices)].q, class_row_indices,
        left_infos, left_ranges, total_left,
        right_infos, right_ranges, total_right)
end

function _svd_cgtsvd_class_metadata(
    q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
    rows::Vector{Row},
    class_ranges::Vector{UnitRange{Int}},
    left_payloads,
    right_payloads,
    left_legs::NTuple{NL, Int},
    right_legs::NTuple{NR, Int}) where {T, QD, N, RD, QT, PS, M, RMT, NL, NR, Row}
    isempty(class_ranges) && throw(ArgumentError("svd produced no CGTSVD classes"))

    Meta = _svd_cgtsvd_class_metadata_type(QT, Val(NL), Val(NR), Val(N))
    metadata = Vector{Meta}(undef, length(class_ranges))
    max_left = 0
    max_right = 0

    for ci in eachindex(class_ranges)
        item = _svd_cgtsvd_class_metadata(
            q, rows, class_ranges[ci], left_payloads, right_payloads,
            left_legs, right_legs)
        metadata[ci] = item
        max_left = max(max_left, item.total_left)
        max_right = max(max_right, item.total_right)
    end
    return metadata, max_left, max_right
end

function _build_svd_cgtsvd_class(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                 class_metadata::Meta,
                                 rows::Vector{Row},
                                 left_payloads,
                                 right_payloads,
                                 core_payloads,
                                 left_legs::NTuple{NL, Int},
                                 right_legs::NTuple{NR, Int},
                                 irrepdim_caches::ICT,
                                 buffer::AbstractMatrix{Tbuf},
                                 product_buffer::Matrix{Tbuf},
                                 permute_buffer::Matrix{Tbuf}) where {T, QD, N, RD, QT, PS, M, RMT, NL, NR, Row, Meta, ICT<:Tuple, Tbuf}
    sector = class_metadata.sector
    class_row_indices = class_metadata.rows
    left_ranges = class_metadata.left_ranges
    right_ranges = class_metadata.right_ranges
    total_left = class_metadata.total_left
    total_right = class_metadata.total_right

    @assert size(buffer, 1) >= total_left && size(buffer, 2) >= total_right
    mat = @view buffer[1:total_left, 1:total_right]
    fill!(mat, zero(Tbuf))
    for row_index in class_row_indices
        row = rows[row_index]
        lrange = left_ranges[row.left_signature]
        rrange = right_ranges[row.right_signature]
        block = @view mat[lrange, rrange]
        _svd_sector_class_matrix!(
            block, q, row.sector_index, left_payloads[row_index], right_payloads[row_index],
            core_payloads[row_index], left_legs, right_legs, product_buffer, permute_buffer)
    end

    m, n = size(mat)
    k = min(m, n)
    _add_svd_cost!(4 * max(m, n) * k^2 + 8 * k^3)
    F = svd(mat; full=false)
    dimq = Float64(_svd_sector_degeneracy(PS, sector, irrepdim_caches, Val(N)))
    scale = sqrt(dimq)
    Result = _svd_cgtsvd_class_result_type(Meta, Tbuf)
    return Result(
        sector, class_metadata.left_infos, class_metadata.right_infos,
        F.U .* scale, F.S ./ scale, F.Vt .* scale)
end

function _build_svd_cgtsvd_classes(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                   class_metadata::Vector{Meta},
                                   max_left::Int,
                                   max_right::Int,
                                   rows::Vector{Row},
                                   left_payloads,
                                   right_payloads,
                                   core_payloads,
                                   left_legs::NTuple{NL, Int},
                                   right_legs::NTuple{NR, Int},
                                   irrepdim_caches::ICT) where {T, QD, N, RD, QT, PS, M, RMT, Meta, Row, NL, NR, ICT<:Tuple}
    isempty(class_metadata) && throw(ArgumentError("svd produced no CGTSVD classes"))
    Tmat = promote_type(T, Float64)
    buffer = Matrix{Tmat}(undef, max_left, max_right)
    product_buffer = similar(buffer)
    permute_buffer = similar(buffer)
    Result = _svd_cgtsvd_class_result_type(Meta, Tmat)
    class_results = Vector{Result}(undef, length(class_metadata))
    for ci in eachindex(class_metadata)
        class_results[ci] = _build_svd_cgtsvd_class(
            q, class_metadata[ci], rows, left_payloads, right_payloads, core_payloads,
            left_legs, right_legs, irrepdim_caches, buffer, product_buffer, permute_buffer)
    end
    return class_results
end

@generated function _svd_sector_degeneracy(::Type{PS},
                                           sector::QT,
                                           irrepdim_caches::CT,
                                           ::Val{N}) where {PS<:ProductSymm, QT, CT<:Tuple, N}
    expr = :(1)
    syms = product_symms(PS)
    for n in 1:N
        S = syms[n]
        if S <: AbelianSymm
            term = :(dimension($S, sector[$n]))
        else
            term = :(_svd_irrep_dimension($S, sector[$n],
                _irrepdim_cache_for(irrepdim_caches, $S, PS, Val($n))))
        end
        expr = :($expr * $term)
    end
    return expr
end

function _select_svd_cgtsvd_entries(class_results,
                                    ::Type{PS},
                                    irrepdim_caches::ICT,
                                    cutoff::Float64,
                                    Nkeep;
                                    get_lists::Bool = false) where {PS<:ProductSymm, ICT<:Tuple}
    return _select_svd_cgtsvd_entries(
        class_results, PS, irrepdim_caches, cutoff, Nkeep,
        Val(get_lists), Val(length(product_symms(PS))))
end

function _select_svd_cgtsvd_entries(class_results,
                                    ::Type{PS},
                                    irrepdim_caches::ICT,
                                    cutoff::Float64,
                                    Nkeep,
                                    ::Val{false},
                                    ::Val{N}) where {PS<:ProductSymm, ICT<:Tuple, N}
    entries = Tuple{Float64, Int, Int}[]
    sv_global_max = 0.0

    for (ci, result) in enumerate(class_results)
        isempty(result.S) && continue
        for j in eachindex(result.S)
            sv = result.S[j]
            sv_global_max = max(sv_global_max, sv)
            push!(entries, (sv, ci, j))
        end
    end

    thresh = cutoff * sv_global_max
    filter!(x -> x[1] > thresh, entries)
    sort!(entries; by = x -> -x[1], alg=MergeSort)
    if !isnothing(Nkeep)
        resize!(entries, min(Nkeep, length(entries)))
    end

    keep_per_class = [Int[] for _ in 1:length(class_results)]
    for (_, ci, j) in entries
        push!(keep_per_class[ci], j)
    end
    for keep in keep_per_class
        sort!(keep)
    end

    return keep_per_class, nothing, nothing
end

function _select_svd_cgtsvd_entries(class_results,
                                    ::Type{PS},
                                    irrepdim_caches::ICT,
                                    cutoff::Float64,
                                    Nkeep,
                                    ::Val{true},
                                    ::Val{N}) where {PS<:ProductSymm, ICT<:Tuple, N}
    entries = Tuple{Float64, Int, Int}[]
    sv_global_max = 0.0
    Entry = Tuple{Float64, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}
    FullEntry = Tuple{Float64, Int, Int, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}
    full_entries = FullEntry[]
    sector_counts = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()

    for (ci, result) in enumerate(class_results)
        isempty(result.S) && continue
        sector = result.sector::NTuple{N, Tuple{Vararg{Int}}}
        offset = get(sector_counts, sector, 0)
        degeneracy = _svd_sector_degeneracy(PS, sector, irrepdim_caches, Val(N))
        for j in eachindex(result.S)
            sv = result.S[j]
            sv_global_max = max(sv_global_max, sv)
            push!(entries, (sv, ci, j))
            push!(full_entries, (sv, ci, j, degeneracy, sector, offset + j))
        end
        sector_counts[sector] = offset + length(result.S)
    end

    thresh = cutoff * sv_global_max
    filter!(x -> x[1] > thresh, entries)
    sort!(entries; by = x -> -x[1], alg=MergeSort)
    if !isnothing(Nkeep)
        resize!(entries, min(Nkeep, length(entries)))
    end

    keep_per_class = [Int[] for _ in 1:length(class_results)]
    for (_, ci, j) in entries
        push!(keep_per_class[ci], j)
    end
    for keep in keep_per_class
        sort!(keep)
    end

    selected = Set{Tuple{Int, Int}}()
    for (_, ci, j) in entries
        push!(selected, (ci, j))
    end

    kept_list = Entry[]
    trunc_list = Entry[]
    sort!(full_entries; by = x -> -x[1], alg=MergeSort)
    for (sv, ci, j, degeneracy, sector, full_index) in full_entries
        entry = (sv, degeneracy, sector, full_index)
        if (ci, j) in selected
            push!(kept_list, entry)
        else
            push!(trunc_list, entry)
        end
    end
    return keep_per_class, kept_list, trunc_list
end

function _svd_cgtsvd_class_ranges(class_results::AbstractVector, keep_per_class, ::Val{N}) where {N}
    class_ranges = Vector{UnitRange{Int}}(undef, length(class_results))
    Sector = NTuple{N, Tuple{Vararg{Int}}}
    sector_counts = Dict{Sector, Int}()
    sector_order = Sector[]

    for (ci, result) in enumerate(class_results)
        nkeep = length(keep_per_class[ci])
        if nkeep == 0
            class_ranges[ci] = 1:0
            continue
        end

        sector = result.sector::Sector
        if !haskey(sector_counts, sector)
            sector_counts[sector] = 0
            push!(sector_order, sector)
        end
        start = sector_counts[sector] + 1
        stop = sector_counts[sector] + nkeep
        class_ranges[ci] = start:stop
        sector_counts[sector] = stop
    end

    return class_ranges, sector_counts, sector_order
end

_svd_cgtsvd_class_ranges(class_results, keep_per_class) =
    _svd_cgtsvd_class_ranges(class_results, keep_per_class, Val(length(first(class_results).sector)))

function _build_svd_cgtsvd_S(::Type{PS},
                             symmetries::NTuple{N, Any},
                             bond_splist,
                             left_tag::AbstractString,
                             right_tag::AbstractString,
                             sector_values,
                             irrepdim_caches) where {PS<:ProductSymm, N}
    QT = fieldtype(eltype(bond_splist), 1)
    qlabels = Vector{NTuple{2, QT}}(undef, length(bond_splist))
    wmatdata, wmatinfo = _unit_wmat_storage(PS, length(bond_splist))
    inds_S = (TLIndex(left_tag, '+'), TLIndex(right_tag, '+'))
    ET = eltype(bond_splist)
    dual_splist = ET[
        (_svd_dual_sector(PS, sector, Val(N)), dim)
        for (sector, dim) in bond_splist
    ]
    sort!(dual_splist; by = first, alg=MergeSort)
    spaces_S = (bond_splist, dual_splist)
    RMTs = Vector{DiagRMT{Float64, 2 + N}}(undef, length(bond_splist))

    for (sector_index, (sector, _)) in enumerate(bond_splist)
        dual_sector = _svd_dual_sector(PS, sector, Val(N))
        qlabels[sector_index] = (sector, dual_sector)
        svals = sector_values[sector]
        cgt_dim = Float64(_svd_sector_degeneracy(PS, sector, irrepdim_caches, Val(N)))
        scale = sqrt(cgt_dim)
        RMTs[sector_index] = _diag_rmt_from_values(svals, Val(2 + N), (1, 2), scale)
    end

    return TLArray(symmetries, qlabels, wmatdata, wmatinfo, RMTs, inds_S, spaces_S)
end

function _svd_right_legs(::Val{QD}, left_legs::NTuple{L, Int}) where {QD, L}
    return ntuple(Val(QD - L)) do k
        nright = 0
        for leg in 1:QD
            if leg ∉ left_legs
                nright += 1
                nright == k && return leg
            end
        end
        throw(BoundsError())
    end
end

function svd_std(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                 left_legs::NTuple{L, Int},
                 left_tag::AbstractString = "svdL",
                 right_tag::AbstractString = "svdR";
                 cutoff::Float64 = 1e-12,
                 Nkeep::Union{Nothing, Int} = nothing,
                 get_lists::Bool = false) where {T, QD, N, RD, QT, PS, M, RMT, L}
    @assert isnothing(Nkeep) || Nkeep >= 0 "Nkeep must be non-negative"
    @assert issorted(left_legs)
    @assert length(unique(left_legs)) == length(left_legs)
    symmetries = product_symms(PS)

    R = QD - L
    right_legs::NTuple{R, Int} = _svd_right_legs(Val(QD), left_legs)

    cgtsvd_caches, active_sector_indices, keys_by_row, sector_row_ranges, total_rows =
        _prepare_svd_counted_preassembly(q, left_legs, Val(N))
    split_rows, left_payloads, right_payloads, core_payloads =
        _get_svd_split_rows(q, active_sector_indices, keys_by_row, sector_row_ranges,
                            cgtsvd_caches, left_legs, right_legs, Val(N), total_rows;
                            tol=cutoff)
    _share_svd_payload_isometries!(
        split_rows, left_payloads, right_payloads, core_payloads, PS; tol=cutoff)
    split_row_classes = _sort_svd_split_rows_by_class!(
        QT, split_rows, left_payloads, right_payloads, core_payloads, Val(L), Val(R))
    class_metadata, max_left, max_right = _svd_cgtsvd_class_metadata(
        q, split_rows, split_row_classes, left_payloads, right_payloads,
        left_legs, right_legs)
    irrepdim_caches = _fill_irrepdim_caches!(_new_irrepdim_caches(q), class_metadata, PS)
    class_results = _build_svd_cgtsvd_classes(
        q, class_metadata, max_left, max_right, split_rows,
        left_payloads, right_payloads, core_payloads, left_legs, right_legs,
        irrepdim_caches)

    keep_per_class, kept_list, trunc_list =
        _select_svd_cgtsvd_entries(class_results, PS, irrepdim_caches, cutoff, Nkeep;
                                   get_lists=get_lists)
    class_ranges, sector_counts, sector_order =
        _svd_cgtsvd_class_ranges(class_results, keep_per_class, Val(N))

    Tout = promote_type(T, Float64)
    qlabels_U = NTuple{L + 1, QT}[]
    qlabels_Vd = NTuple{R + 1, QT}[]
    RMTs_U = Array{Tout, L + 1 + N}[]
    RMTs_Vd = Array{Tout, R + 1 + N}[]
    nonabelian_indices = nonabelian_symmetry_indices(PS)

    n_wmat_U = 0
    n_wmat_Vd = 0
    total_wmat_U = 0
    total_wmat_Vd = 0
    for (ci, result) in enumerate(class_results)
        isempty(keep_per_class[ci]) && continue
        for info in result.left_infos
            n_wmat_U += 1
            payload = left_payloads[info.row_index]
            for slot in 1:M
                total_wmat_U += length(payload[slot])
            end
        end
        for info in result.right_infos
            n_wmat_Vd += 1
            payload = right_payloads[info.row_index]
            for slot in 1:M
                total_wmat_Vd += length(payload[slot])
            end
        end
    end
    wmatdata_U = Vector{Float64}(undef, total_wmat_U)
    wmatinfo_U = Vector{WMatInfo{M}}(undef, n_wmat_U)
    wmatdata_Vd = Vector{Float64}(undef, total_wmat_Vd)
    wmatinfo_Vd = Vector{WMatInfo{M}}(undef, n_wmat_Vd)
    next_wmat_offset_U = 1
    next_wmat_sector_U = 1
    next_wmat_offset_Vd = 1
    next_wmat_sector_Vd = 1

    Sector = NTuple{N, Tuple{Vararg{Int}}}
    sector_values = Dict{Sector, Vector{Float64}}()
    for sector in sector_order
        svals = Float64[]
        for (ci, result) in enumerate(class_results)
            result.sector == sector || continue
            append!(svals, result.S[keep_per_class[ci]])
        end
        sector_values[sector] = svals
    end

    for (ci, result) in enumerate(class_results)
        keep = keep_per_class[ci]
        isempty(keep) && continue

        sector = result.sector
        dual_sector = _svd_dual_sector(PS, sector, Val(N))
        sector_count = sector_counts[sector]
        class_range = class_ranges[ci]

        for info in result.left_infos
            left_payload = left_payloads[info.row_index]
            part = result.U[info.range, keep]
            full = zeros(eltype(result.U), length(info.range), sector_count)
            full[:, class_range] = part
            tmp = reshape(full, info.phys_dims..., info.om_dims..., sector_count)
            perm = Tuple(vcat(collect(1:L), L + N + 1, collect(L+1:L+N)))
            data = permutedims(tmp, perm)
            rmt_U = data

            cgts_U = ntuple(N) do n
                source_qlabels, source_cgp, _, source_legdir =
                    _svd_symmetry_stored_leg_order(QT, q, info.sector_index, Val(n))
                wmat = isabelian(nth_symm(PS, Val(n))) ? _svd_trivial_iso() :
                    _svd_left_iso(left_payload, PS, Val(n))
                _svd_build_side_cgt_metadata(
                    source_qlabels, source_cgp, source_legdir, left_legs,
                    sector[n], false, wmat)
            end
            push!(qlabels_U,
                  _svd_physical_qlabels(QT,
                                         ntuple(n -> cgts_U[n].qlabels, Val(N)),
                                         ntuple(n -> cgts_U[n].cgp, Val(N)),
                                         Val(L + 1)))
            next_wmat_offset_U = _store_wmat_tuple!(
                wmatdata_U, wmatinfo_U, next_wmat_sector_U, next_wmat_offset_U,
                ntuple(slot -> cgts_U[nonabelian_indices[slot]].wmat, Val(M)))
            next_wmat_sector_U += 1
            push!(RMTs_U, rmt_U)
        end

        for info in result.right_infos
            right_payload = right_payloads[info.row_index]
            part = result.Vt[keep, info.range]
            full = zeros(eltype(result.Vt), sector_count, length(info.range))
            full[class_range, :] = part
            rmt_Vd = reshape(full, sector_count, info.phys_dims..., info.om_dims...)

            cgts_Vd = ntuple(N) do n
                source_qlabels, source_cgp, _, source_legdir =
                    _svd_symmetry_stored_leg_order(QT, q, info.sector_index, Val(n))
                wmat = isabelian(nth_symm(PS, Val(n))) ? _svd_trivial_iso() :
                    _svd_right_iso(right_payload, PS, Val(n))
                _svd_build_side_cgt_metadata(
                    source_qlabels, source_cgp, source_legdir, right_legs,
                    dual_sector[n], true, wmat)
            end
            push!(qlabels_Vd,
                  _svd_physical_qlabels(QT,
                                         ntuple(n -> cgts_Vd[n].qlabels, Val(N)),
                                         ntuple(n -> cgts_Vd[n].cgp, Val(N)),
                                         Val(R + 1)))
            next_wmat_offset_Vd = _store_wmat_tuple!(
                wmatdata_Vd, wmatinfo_Vd, next_wmat_sector_Vd, next_wmat_offset_Vd,
                ntuple(slot -> cgts_Vd[nonabelian_indices[slot]].wmat, Val(M)))
            next_wmat_sector_Vd += 1
            push!(RMTs_Vd, rmt_Vd)
        end
    end

    bond_splist = Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}[
        (sector, sector_counts[sector]) for sector in sector_order
    ]
    dual_bond_splist = Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}[
        (_svd_dual_sector(PS, sector, Val(N)), sector_counts[sector]) for sector in sector_order
    ]
    sort!(dual_bond_splist; by = first, alg=MergeSort)

    inds_U = (ntuple(i -> q.inds[left_legs[i]], Val(L))..., TLIndex(left_tag, '-'))
    inds_Vd = (TLIndex(right_tag, '-'), ntuple(i -> q.inds[right_legs[i]], Val(R))...)

    spaces_U = (ntuple(i -> q.spaces[left_legs[i]], Val(L))..., bond_splist)
    spaces_Vd = (dual_bond_splist, ntuple(i -> q.spaces[right_legs[i]], Val(R))...)

    U_qlabels = Matrix{QT}(undef, L + 1, length(qlabels_U))
    for sector_index in eachindex(qlabels_U), leg in 1:(L + 1)
        U_qlabels[leg, sector_index] = qlabels_U[sector_index][leg]
    end
    Vd_qlabels = Matrix{QT}(undef, R + 1, length(qlabels_Vd))
    for sector_index in eachindex(qlabels_Vd), leg in 1:(R + 1)
        Vd_qlabels[leg, sector_index] = qlabels_Vd[sector_index][leg]
    end

    U = TLArray(symmetries, U_qlabels, wmatdata_U, wmatinfo_U, RMTs_U, inds_U, spaces_U)
    S = _build_svd_cgtsvd_S(PS, symmetries, bond_splist, left_tag, right_tag,
                            sector_values, irrepdim_caches)
    Vd = TLArray(symmetries, Vd_qlabels, wmatdata_Vd, wmatinfo_Vd, RMTs_Vd, inds_Vd, spaces_Vd)
    return SVDResult(U, S, Vd, kept_list, trunc_list)
end

"""
    LinearAlgebra.svd(q, left_legs, left_tag="svdL", right_tag="svdR"; cutoff=1e-12, Nkeep=nothing, get_lists=false)

CGTSVD-based SVD of a `TLArray`, returning `SVDResult(U, S, Vd, kept_list, trunc_list)`.

Unlike `svd`, this path uses the direct split-space class SVD assembly and
returns the bond convention requested for CGTSVD:

- `U` bond leg: outgoing `'-'`
- `S` legs: incoming `'+'`, incoming `'+'`
- `Vd` bond leg: outgoing `'-'`
"""
function LinearAlgebra.svd(q::TLArray{T, QD, N, RD},
                    left_legs,
                    left_tag::AbstractString = "svdL",
                    right_tag::AbstractString = "svdR";
                    cutoff::Float64 = 1e-12,
                    Nkeep::Union{Nothing, Int} = nothing,
                    get_lists::Bool = false) where {T, QD, N, RD}
    left_legs_ = Tuple(_normalize_svd_left_legs(left_legs, QD))
    return svd_std(q, left_legs_, left_tag, right_tag;
                   cutoff=cutoff, Nkeep=Nkeep, get_lists=get_lists)
end

svd_cgtsvd(q::TLArray, args...; kwargs...) = svd(q, args...; kwargs...)

function LinearAlgebra.svd(q::TLArray{T, QD, N, RD},
                    left_tag::AbstractString = "svdL",
                    right_tag::AbstractString = "svdR";
                    dir=nothing,
                    itag=nothing,
                    plev=nothing,
                    lock=nothing,
                    rev::Bool=false,
                    cutoff::Float64=1e-12,
                    Nkeep::Union{Nothing, Int}=nothing,
                    get_lists::Bool=false) where {T, QD, N, RD}
    left_legs = _select_svd_left_legs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return svd(q, left_legs, left_tag, right_tag; cutoff=cutoff, Nkeep=Nkeep, get_lists=get_lists)
end

LinearAlgebra.svd(q::TLArrayView, args...; kwargs...) =
    svd(_eager_tlarray(q), args...; kwargs...)

function _normalize_svd_left_legs(left_legs, rank::Int)
    legs = collect(Int, left_legs)
    isempty(legs) && throw(ArgumentError("svd requires at least one left leg"))
    length(legs) < rank || throw(ArgumentError("svd requires at most $(rank - 1) left legs for rank-$rank input"))
    all(1 .<= legs .<= rank) || throw(ArgumentError("svd left_legs must lie between 1 and $rank"))
    length(unique(legs)) == length(legs) || throw(ArgumentError("svd left_legs must not contain duplicates"))
    return collect(sort(legs))
end

function _select_svd_left_legs(q::TLArray; dir=nothing, itag=nothing, plev=nothing,
                               lock=nothing, rev::Bool=false)
    if isnothing(dir) && isnothing(itag) && isnothing(plev) && isnothing(lock)
        throw(ArgumentError("keyword-based svd requires at least one of dir, itag, plev, or lock"))
    end

    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    rank = length(q.inds)
    1 <= length(legs) <= rank - 1 || throw(ArgumentError("keyword-based svd selected legs $legs, but the number of selected legs must be between 1 and $(rank - 1)"))
    return sort(legs)
end
