function _reorder_perm_part(qlabels::NTuple{N, NTuple{NZ, Int}},
    cgp_part::NTuple{N, Int}) where {NZ, N}

    perm = zeros(Int, N)
    st = 1
    while st <= N
        ed = st
        while ed <= N && qlabels[ed] == qlabels[st] ed += 1 end
        perm[st:ed-1] = sortperm(collect(cgp_part[st:ed-1])) .+ (st-1)
        st = ed
    end
    return perm
end

# Compute the reordering permutation for stored positions to satisfy the order constraint.
# Returns reorder where reorder[new_stored_pos] = old_stored_pos.
function _compute_reorder_permutation(qlabels::NTuple{QD, NTuple{NZ, Int}}, 
    cgp::NTuple{QD, Int}, 
    legdir::Tuple{Int, Int}) where {NZ, QD}

    m, k = legdir
    # Build cgp_inv: cgp_inv[stored_pos] = physical_leg
    cgp_inv = zeros(Int, QD)
    for l in 1:QD cgp_inv[cgp[l]] = l end

    # For each direction group, sort by (qlabel, physical_leg) to satisfy constraint.
    # After sorting, same qlabels will have physical legs in increasing order.
    
    # Incoming: positions 1:m
    up_perm = _reorder_perm_part(qlabels[1:m], Tuple(cgp_inv[1:m]))
    # Outgoing: positions m+1:m+k
    dn_perm = _reorder_perm_part(qlabels[m+1:m+k], Tuple(cgp_inv[m+1:m+k]))
    
    # reorder[new] = old: new stored position gets qlabel from old position
    reorder = (up_perm..., (dn_perm .+ m)...)
    
    return reorder
end

function _permuted_sector_wmat(symm_type,
                               qlabels::NTuple{QD, NTuple{NZ, Int}},
                               wmat::AbstractMatrix{Float64},
                               cgp::NTuple{QD, Int},
                               legdir::Tuple{Int, Int},
                               perm::NTuple{QD, Int}) where {QD, NZ}
    new_cgp = Tuple(cgp[perm[l]] for l in 1:QD)
    
    # Need to reorder stored qlabels and apply CGTperm transformation
    m, k = legdir
    reorder = _compute_reorder_permutation(qlabels, new_cgp, legdir)

    upsp = Tuple(qlabels[i] for i in 1:m)    # incoming qlabels, already sorted
    dnsp = Tuple(qlabels[m+i] for i in 1:k)  # outgoing qlabels, already sorted
    
    # New qlabels after reordering
    new_qlabels = Tuple(qlabels[reorder[i]] for i in 1:QD)
    # The permutation 'reorder' should permute only same qlabels
    @assert new_qlabels == qlabels

    cgtperm_obj = getNsave_CGTperm(symm_type, upsp, dnsp, reorder)

    if isnothing(cgtperm_obj)
        # Permutation is identity, or symmetry is Abelian
        return wmat
    end
    
    # Apply CGTperm. CGTperm transforms from old OM basis to new OM basis
    return cgtperm_obj.perm_arr * wmat
end

function _permute_sector_wmat(q::AbstractTLArray{T, QD, N, RD}, sector_index::Int,
                              perm::NTuple{QD, Int}, n::Int, symm) where {T, QD, N, RD}
    qlabels, cgp, legdir = _sector_cgt_metadata(q, sector_index, n)
    return _permuted_sector_wmat(symm[n], qlabels, sector_wmat(q, sector_index, n),
                                 cgp, legdir, perm)
end

function _permute_sector_wmat(q::AbstractTLArray{T, QD, N, RD, QT, PS}, sector_index::Int,
                              perm::NTuple{QD, Int}, ::Val{n}) where {T, QD, N, RD, QT, PS, n}
    qlabels, cgp, legdir = _sector_cgt_metadata(q, sector_index, n)
    return _permuted_sector_wmat(product_symms(PS)[n], qlabels,
                                 sector_wmat(q, sector_index, Val(n)),
                                 cgp, legdir, perm)
end

function _permute_sector_rmt(q::TLArray{T, QD, N, RD}, sector_index::Int,
                             perm::NTuple{QD, Int}) where {T, QD, N, RD}
    rmt_perm = (perm..., ntuple(n -> QD + n, N)...)
    return permutedims(sector_rmt(q, sector_index), rmt_perm)
end

function _permute_wmat_storage(q::TLArray{T, QD, N, RD, QT, PS, M},
                               perm::NTuple{QD, Int},
                               symm) where {T, QD, N, RD, QT, PS, M}
    out_data = copy(q.wmatdata)
    out_info = copy(q.wmatinfo)
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    for slot in 1:M
        n = nonabelian_indices[slot]
        for sector_index in sector_slots(q)
            _wmat_info_present(q.wmatinfo[sector_index][slot]) || continue
            _copy_wmat_to_storage!(out_data, out_info, sector_index, slot,
                                   _permute_sector_wmat(q, sector_index, perm, n, symm))
        end
    end
    return out_data, out_info
end

function _permute_sector_rmts(q::TLArray{T, QD, N, RD}, perm::NTuple{QD, Int}) where {T, QD, N, RD}
    out = similar(q.RMTs, sector_count(q))
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        out[sector_index] = _permute_sector_rmt(q, sector_index, perm)
    end
    return out
end

"""
    permutedims(q::TLArray, perm)

Permute the legs of a TLArray object.

# Arguments
- `q`: TLArray to permute
- `perm`: permutation (tuple or vector) of length equal to the rank of q.
          `perm[new_pos] = old_pos` means the leg at `old_pos` moves to `new_pos`.

# Returns
New TLArray with permuted legs.
"""
function _materialized_permutedims(q::TLArray{T, QD, N, RD}, perm) where {T, QD, N, RD}
    perm = Tuple(perm)
    @assert length(perm) == QD "permutation length $(length(perm)) != TLArray rank $QD"
    @assert sort(collect(perm)) == collect(1:QD) "perm must be a valid permutation of 1:$QD"
    
    # 1. Permute TLIndex tuple
    new_inds = Tuple(q.inds[perm[l]] for l in 1:QD)
    
    # 2. Permute spaces tuple
    new_spaces = Tuple(q.spaces[perm[l]] for l in 1:QD)
    
    # 3 & 4. Permute each sector (CGT metadata and RMT)
    new_qlabels = [ntuple(l -> sector[perm[l]], Val(QD)) for sector in q.qlabels]
    new_wmatdata, new_wmatinfo = _permute_wmat_storage(q, perm, symm(q))
    new_RMTs = _permute_sector_rmts(q, perm)
    
    # 5. Assemble and return new TLArray with precomputed spaces
    return TLArray(symm(q), new_qlabels, new_wmatdata, new_wmatinfo, new_RMTs, new_inds, new_spaces)
end

Base.permutedims(q::AbstractTLArray, perm) = _view_permutedims(q, perm)
