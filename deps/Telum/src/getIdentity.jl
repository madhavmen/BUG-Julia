# N: the number of symmetries
# QT: the concrete qlabel type for one sector
# PS: the product symmetry type
struct leginfo{N, QT<:Tuple, PS<:ProductSymm}
    ind::TLIndex
    splist::Vector{Tuple{QT, Int}}
end

function leginfo(symm::NTuple{N, Any}, ind::TLIndex, splist::AbstractVector) where {N}
    QT = qlabeltype(symm)
    PS = productsymm(symm)
    return leginfo{N, QT, PS}(ind, Vector{Tuple{QT, Int}}(splist))
end

leginfo{N}(symm::NTuple{N, Any}, ind::TLIndex, splist::AbstractVector) where {N} =
    leginfo(symm, ind, splist)

# Use precomputed spaces from TLArray
function leginfo(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}, i::Int) where {T, QD, N, RD, QT, PS, M, RMT}
    return leginfo{N, QT, PS}(inds(q)[i], spaces(q)[i])
end

productsymm(::leginfo{N, QT, PS}) where {N, QT, PS} = PS
product_symms(info::leginfo) = product_symms(productsymm(info))
symm(info::leginfo) = product_symms(info)
nsymms(::leginfo{N}) where {N} = N
qlabeltype(::leginfo{N, QT}) where {N, QT} = QT

@inline function Base.getproperty(info::leginfo, name::Symbol)
    name === :symm && return product_symms(info)
    return getfield(info, name)
end

Base.propertynames(::leginfo, private::Bool=false) =
    private ? (:symm, :ind, :splist) : (:symm, :ind, :splist)

# Variadic entry point: accepts multiple (TLArray, Int) pairs as positional arguments
# Keyword arguments control the fused output leg's TLIndex properties
function getIdentity(legs::Vararg{Tuple{AbstractTLArray, Int}};
                     itag::AbstractString="", plev::Int=0, lock::Int=0)
    leginfos = Tuple(leginfo(q, i) for (q, i) in legs)
    return getIdentity(leginfos; itag=itag, plev=plev, lock=lock)
end

function _normalize_getIdentity_legs(q::AbstractTLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("getIdentity requires at least one leg"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "getIdentity legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "getIdentity legs must be unique, got $positions"))
    return Tuple(positions)
end

function getIdentity(q::AbstractTLArray{T, QD}, legs::LegList;
                     itag::AbstractString="", plev::Int=0, lock::Int=0) where {T, QD}
    positions = _normalize_getIdentity_legs(q, legs)
    return getIdentity(((q, leg) for leg in positions)...; itag=itag, plev=plev, lock=lock)
end

function getIdentity(q::AbstractTLArray{T, QD}, leg::Integer, morelegs::Vararg{Integer};
                     itag::AbstractString="", plev::Int=0, lock::Int=0) where {T, QD}
    return getIdentity(q, (Int(leg), (Int(l) for l in morelegs)...); itag=itag, plev=plev, lock=lock)
end
    
# For abelian symmetries: unique outcome is the sum of all qlabels; OM is always 1
# Returns Vector of (qlabel, om) pairs
function combine_qlabels(::Type{S}, 
    qlabels_per_leg::NTuple{D, NTuple{1, Int}}) where {S <: AbelianSymm, D}
    result = qlabels_per_leg[1][1]
    for d in 2:D
        result = add_qn(S, result, qlabels_per_leg[d][1])
    end
    return Tuple{NTuple{1, Int}, Int}[((result,), 1)]
end

# For non-abelian symmetries: use ValidOuts + OMList to enumerate fused irreps with OMs
# Returns Vector of (qlabel, om) pairs
function combine_qlabels(::Type{S}, 
    qlabels_per_leg::NTuple{D, NTuple{NZ, Int}}) where {S <: NonabelianSymm, D, NZ}
    incom = Tuple(sort(collect(qlabels_per_leg)))
    vo = getNsave_validout(S, incom)
    return Tuple{NTuple{NZ, Int}, Int}[
        (q, getNsave_omlist(S, incom, q).totalOM)
        for q in vo.out_spaces
    ]
end

@inline _identity_outcomes(::Type{PS}, qlabels::NTuple{D, QT}) where {D, QT, Syms, PS<:ProductSymm{Syms}} =
    ntuple(Val(length(Syms.parameters))) do n
        combine_qlabels(fieldtype(Syms, n), ntuple(d -> qlabels[d][n], Val(D)))
    end

@inline _identity_cgt_dim(::Type{PS}, fused_qlabels::QT) where {QT, Syms, PS<:ProductSymm{Syms}} =
    foldl(*, ntuple(Val(length(Syms.parameters))) do n
        dimension(fieldtype(Syms, n), fused_qlabels[n])
    end; init=1)

function getIdentity(leginfos::NTuple{D, leginfo{N, QT, PS}};
                     itag::AbstractString="", plev::Int=0, lock::Int=0) where {D, N, QT, Syms, PS<:ProductSymm{Syms}}

    for i in 1:D-1 @assert leginfos[i].symm == leginfos[i+1].symm end
    symm = product_symms(PS)

    # For incoming legs (dir == '+'), dualize every qlabel so that
    # the fusion rule sees all legs as outgoing (charge conservation: sum = 0).
    leginfos_adj = ntuple(Val(D)) do d
        if leginfos[d].ind.dir == '+'
            ET = eltype(leginfos[d].splist)
            new_splist = ET[(ntuple(Val(N)) do n
                                 get_dualq(fieldtype(Syms, n), qlabels[n])
                             end, dim)
                            for (qlabels, dim) in leginfos[d].splist]
            sort!(new_splist; by = x -> x[1])
            leginfo{N, QT, PS}(leginfos[d].ind, new_splist)
        else
            leginfos[d]
        end
    end
    # (sector indices into each leginfo's splist..., total RMT dim, oms...,
    # starting index, ending index along fused axis)
    merged_info = Dict{QT, Vector{NTuple{D+N+3, Int}}}()

    nsectors = ntuple(d -> length(leginfos_adj[d].splist), Val(D))

    for i in CartesianIndices(nsectors)
        I = i.I
        RMT_dims = ntuple(d -> leginfos_adj[d].splist[I[d]][2], Val(D))
        RMT_dim = prod(RMT_dims)
        qlabels = ntuple(d -> leginfos_adj[d].splist[I[d]][1], Val(D))

        # For each symmetry n, fuse the D qlabels to get all valid (qlabel, om) pairs
        outcomes_per_symm = _identity_outcomes(PS, qlabels)
        num_outcomes = ntuple(n -> length(outcomes_per_symm[n]), Val(N))
        for j in CartesianIndices(num_outcomes)
            J = j.I
            fused_qlabels = ntuple(n -> outcomes_per_symm[n][J[n]][1], Val(N))
            oms = ntuple(n -> outcomes_per_symm[n][J[n]][2], Val(N))
            sti = 1
            if !haskey(merged_info, fused_qlabels)
                merged_info[fused_qlabels] = Vector{NTuple{D+N+3, Int}}()
            else sti = merged_info[fused_qlabels][end][end] + 1 end
            total_dim = RMT_dim * prod(oms)
            entry = (I..., RMT_dim, oms..., sti, sti+total_dim-1)
            push!(merged_info[fused_qlabels], entry)
        end
    end

    # Build sectors of the internal fused TLArray.
    # Before the final fixup for originally-incoming legs, the tensor has
    # D selected legs + 1 fused output leg, where selected incoming legs have
    # been dualized so every selected leg can be fused as incoming.
    # RMT shape: (rmts_dims..., space_cnt).
    # For each entry, the block [:,...,:, sti:edi] is an identity matrix
    # (repeated prod_oms times along the outer-multiplicity sub-blocks).
    # Each sector has a D+1-leg qlabel tuple and one OM×OM identity w-matrix
    # per symmetry.
    sector_qlabels = NTuple{D + 1, QT}[]
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    M = length(nonabelian_indices)
    total_wmat_len = 0
    total_wmat_slots = 0
    for entries in values(merged_info)
        total_wmat_slots += length(entries)
        for entry in entries
            oms = entry[D+2:D+N+1]
            for slot in 1:M
                om = oms[nonabelian_indices[slot]]
                total_wmat_len += om * om
            end
        end
    end
    wmatdata = zeros(Float64, total_wmat_len)
    wmatinfo = Vector{WMatInfo{M}}(undef, total_wmat_slots)
    next_wmat_offset = 1
    next_wmat_sector = 1
    RMTs = Array{Float64, D + 1 + N}[]

    for (fused_qlabels, entries) in merged_info
        #println(fused_qlabels)
        space_cnt = entries[end][end]  # last edi across all entries

        # 1. Scale by sqrt of total outgoing CGT dimension
        cgt_dim_out = _identity_cgt_dim(PS, fused_qlabels)

        for entry in entries
            #println("  ", entry)
            orig_ind  = entry[1:D]
            RMT_dim   = entry[D+1]
            oms       = entry[D+2:D+N+1]
            sti       = entry[end-1]
            edi       = entry[end]
            prod_oms  = prod(oms)

            # Dimensions of RMTs of input legs at this sector
            rmts_dims = ntuple(d -> leginfos_adj[d].splist[orig_ind[d]][2], Val(D))

            # Build RMT: zeros everywhere, identity blocks at [:,...,:, sti:edi]
            # The block is laid out as prod_oms copies of a RMT_dim×RMT_dim identity.
            RMT_data = zeros(Float64, rmts_dims..., space_cnt, prod_oms)
            id_block  = reshape(Matrix{Float64}(I, RMT_dim, RMT_dim), rmts_dims..., RMT_dim)
            for m in 1:prod_oms
                cs = sti + (m - 1) * RMT_dim
                RMT_data[ntuple(_ -> (:), D)..., cs:cs+RMT_dim-1, m] = id_block
            end
            RMT_data .*= sqrt(cgt_dim_out)
            RMT_t = reshape(RMT_data, rmts_dims..., space_cnt, oms...)
            wmatinfo[next_wmat_sector] = ntuple(Val(M)) do slot
                om = oms[nonabelian_indices[slot]]
                offset = next_wmat_offset
                next_wmat_offset += om * om
                for j in 1:om
                    wmatdata[offset + (j - 1) * (om + 1)] = 1.0
                end
                (offset, om, om)
            end
            next_wmat_sector += 1

            phys_qlabels = ntuple(d -> d <= D ?
                leginfos_adj[d].splist[orig_ind[d]][1] :
                fused_qlabels, Val(D + 1))
            push!(sector_qlabels, phys_qlabels)
            push!(RMTs, RMT_t)
        end
    end

    # Assemble an internal TLArray whose selected legs are ready for fusion.
    # For originally-incoming legs we expose the dualized leg with dual=true so
    # we can contract a 1j tensor and recover a directly-contractable external leg.
    fused_ind = TLIndex(itag, '-', plev, lock)
    inds = (ntuple(d -> begin
        idx = leginfos[d].ind.dir == '+' ? change_dual(leginfos[d].ind) : leginfos[d].ind
        to_incoming(idx)
    end, Val(D))..., fused_ind)
    
    # Build spaces tuple: first D legs use the adjusted spaces, last leg from merged_info
    # Fused leg space: for each fused_qlabel, RMT dim = total dimension (last edi)
    fused_splist = Vector{Tuple{QT, Int}}()
    for (fused_qlabels, entries) in merged_info
        space_dim = entries[end][end]  # last edi = total RMT dimension for this sector
        push!(fused_splist, (fused_qlabels, space_dim))
    end
    sort!(fused_splist; by = x -> x[1])  # sort by qlabels for consistency
    
    spaces = (ntuple(d -> leginfos_adj[d].splist, Val(D))..., fused_splist)

    qlabels = Matrix{QT}(undef, D + 1, length(sector_qlabels))
    for (sector_index, sector) in enumerate(sector_qlabels), leg in 1:(D + 1)
        qlabels[leg, sector_index] = sector[leg]
    end

    q = TLArray(symm, qlabels, wmatdata, wmatinfo, RMTs, inds, spaces)

    # For an originally-incoming selected leg, attach a 1j tensor so the returned
    # leg is directly contractable with the original tensor leg.
    for d in 1:D
        leginfos[d].ind.dir == '+' || continue
        j = get1jtensor(leginfos[d])
        q = contract(q, (d,), j, (2,); reduce_lock=false)
        perm = (ntuple(i -> i, d - 1)..., D + 1, ntuple(i -> d - 1 + i, D + 1 - d)...)
        q = permutedims(q, perm)
    end

    return q
end
