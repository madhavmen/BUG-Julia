# ── getLocalSpace ────────────────────────────────────────────────────────────
#
# Entry point:
#
#   IS, operators... = getLocalSpace(opts::LocalSpaceOptions)
#
# Internal pipeline (common to all option types):
#
#   1. getSymmetryInfo(opts)
#        └─► (symm_tuple, weights, lowering_ops, mwirops)
#              |
#              ▼
#      validate_cross_symmetry_commutation(...)
#              |
#              ▼
#   2. decompose_space(...)          [decomp_space.jl]
#        └─► (ortho_vecs, sector_index_map)   ← symmetry-resolved basis
#              |
#              ▼
#   3. getIROPs(opts, ortho_vecs)
#        └─► NamedTuple of IROP matrices, e.g. (F=..., S=..., Z=..., IS=...)
#              |
#              ▼
#   4. compress_irops(...)           [internal helper below]
#        └─► one TLArray object per IROP
#
# To add a new space type:
#   a) Define a new subtype of LocalSpaceOptions in options.jl.
#   b) Implement getSymmetryInfo(opts::YourOptions) and
#      getIROPs(opts::YourOptions, ortho_vecs, sector_map) below.
#   c) getLocalSpace itself needs no changes.
# ─────────────────────────────────────────────────────────────────────────────

# ── Generic hooks (must be specialised for each option type) ─────────────────

"""
    getSymmetryInfo(opts::LocalSpaceOptions)
        -> (symm, weights, lowering_ops, mwirops)

Return the ingredients needed to decompose the local Hilbert space into
symmetry sectors:

- `symm`         – `NTuple{N, Type{<:Symmetry}}` listing the relevant symmetries.
- `weights`      – `NTuple{N, Vector}`, weight (charge) of each basis state for
                   each symmetry. These are interpreted as the diagonals of the
                   Cartan / charge operators for that symmetry.
- `lowering_ops` – `NTuple{N, Vector{<:AbstractMatrix}}`, lowering operators
                   (empty vector for purely abelian symmetries).
- `mwirops`      – `Dict{Symbol, AbstractMatrix}` mapping the returned operator
                   names to their scaled maximal-weight operator.
"""
function getSymmetryInfo(opts::LocalSpaceOptions)
    error("getSymmetryInfo not implemented for $(typeof(opts))")
end

function _symmetry_name(symm::Type{<:Symmetry})
    return totxt(symm)
end

function _zops_from_weights(symm::Type{<:Symmetry},
    weights::AbstractVector{<:Tuple{Vararg{Int}}},
    symm_idx::Int)

    isempty(weights) && throw(ArgumentError(
        "Symmetry $symm_idx ($(_symmetry_name(symm))) must provide at least one weight tuple"))

    nz = nzops(symm)
    for (state_idx, weight) in pairs(weights)
        length(weight) == nz || throw(ArgumentError(
            "Weight tuple $state_idx for symmetry $symm_idx ($(_symmetry_name(symm))) " *
            "has $(length(weight)) entries, expected $nz"))
    end

    return [spdiagm(0 => Int[weight[zidx] for weight in weights]) for zidx in 1:nz]
end

function _commutator_iszero(A::AbstractMatrix, B::AbstractMatrix)
    comm_res = sparse(comm(A, B))
    droptol!(comm_res, 1e-12)
    return nnz(comm_res) == 0
end

function _symmetry_ops_for_commutation_check(symm::Type{<:Symmetry},
    weights::AbstractVector{<:Tuple{Vararg{Int}}},
    lowering_ops::AbstractVector{<:AbstractMatrix},
    symm_idx::Int,
    spdim::Int)

    ops = Tuple{String, AbstractMatrix}[]
    for (zidx, zop) in pairs(_zops_from_weights(symm, weights, symm_idx))
        size(zop) == (spdim, spdim) || throw(ArgumentError(
            "Weight-derived z-operator $zidx for symmetry $symm_idx ($(_symmetry_name(symm))) " *
            "has size $(size(zop)), expected ($spdim, $spdim)"))
        push!(ops, ("weight[$zidx]", zop))
    end

    for (lopidx, lop) in pairs(lowering_ops)
        lop_sparse = sparse(lop)
        size(lop_sparse) == (spdim, spdim) || throw(ArgumentError(
            "Lowering operator $lopidx for symmetry $symm_idx ($(_symmetry_name(symm))) " *
            "has size $(size(lop_sparse)), expected ($spdim, $spdim)"))
        push!(ops, ("lowering[$lopidx]", lop_sparse))
    end
    return ops
end

function _validate_cross_symmetry_commutation(symm::NTuple{N, Any},
    weights::NTuple{N, Vector{<:Tuple{Vararg{Int}}}},
    lowering_ops::NTuple{N, Vector{<:AbstractMatrix}}) where N

    N <= 1 && return nothing

    spdim = length(weights[1])
    ops_by_symmetry = Vector{Vector{Tuple{String, AbstractMatrix}}}(undef, N)
    for i in 1:N
        length(weights[i]) == spdim || throw(ArgumentError(
            "All symmetries must act on the same local-space dimension; " *
            "symmetry $i ($(_symmetry_name(symm[i]))) has $(length(weights[i])) weights, expected $spdim"))
        ops_by_symmetry[i] = _symmetry_ops_for_commutation_check(
            symm[i], weights[i], lowering_ops[i], i, spdim)
    end

    for i in 1:N-1
        for j in i+1:N
            for (label_i, op_i) in ops_by_symmetry[i]
                for (label_j, op_j) in ops_by_symmetry[j]
                    _commutator_iszero(op_i, op_j) && continue
                    throw(ArgumentError(
                        "Symmetry operations for symmetry $i ($(_symmetry_name(symm[i]))) " *
                        "and symmetry $j ($(_symmetry_name(symm[j]))) must commute, " *
                        "but $label_i and $label_j do not"))
                end
            end
        end
    end
    return nothing
end

# ── Generic pipeline ─────────────────────────────────────────────────────────

"""
    getLocalSpace(opts::LocalSpaceOptions)

Construct the symmetry-adapted local state space and irreducible operator
representations (IROPs) for the physical site described by `opts`.

Returns a named tuple whose fields correspond to the standard operators of the
given space type (e.g. `IS`, `Z`, `F`, `S` for a spinful fermionic site).
"""

function reduceto2d(symm::NTuple{N, Any}, 
    data_full::Vector{Tuple{NTuple{3, NTuple{N, Tuple{Vararg{Int}}}}, Array{Float64, M}}}) where {N, M}

    @assert M == N + 3
    reduced = Vector{Tuple{NTuple{2, NTuple{N, Tuple{Vararg{Int}}}}, Array{Float64, 2+N}}}()

    for (qlabels, block) in data_full
        block_idx = (:, :, 1, (Colon() for _=1:N)...)
        push!(reduced, (qlabels[1:2], block[block_idx...]))
    end
    return reduced
end

function reduce_irop_blocks(symm::NTuple{N, Any}, 
    data_full::Vector{Tuple{NTuple{3, NTuple{N, Tuple{Vararg{Int}}}}, Array{Float64, M}}}) where {N, M}

    @assert M == N + 3

    third_dim_qlabel = first(data_full)[1][3]
    is_third_singleton = all(all(iszero, qlabel) for qlabel in third_dim_qlabel)
    if is_third_singleton data_full = reduceto2d(symm, data_full) end
    return data_full
end

# Convert mult_ind from decompose_space to spaces format for TLArray
# mult_ind: Dict{qlabels => Vector{(start_idx, end_idx)}}
# returns: Vector{Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}} - list of (qlabels, RMT_dim)
# Sorted by qlabels for consistency
function _mult_ind_to_splist(symm::NTuple{N, Any}, 
    mult_ind::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Tuple{Int, Int}}}) where N
    
    splist = Vector{Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}}()
    for (qlabels, ranges) in mult_ind
        # RMT dimension = outer multiplicity (number of ranges) 
        # Each range represents one copy of the irrep, so outer_mult = length(ranges)
        rmt_dim = length(ranges)
        push!(splist, (qlabels, rmt_dim))
    end
    # Sort by qlabels (first element of each tuple)
    sort!(splist; by = x -> x[1])
    return splist
end

function getLocalSpace(opts::LocalSpaceOptions,
    tags::Tuple{Vararg{AbstractString, 3}}=("", "", ""))

    # Step 1 – symmetry operators that define the local Hilbert space structure
    symm, weights, lowering_ops, mwirops = getSymmetryInfo(opts)
    _validate_cross_symmetry_commutation(symm, weights, lowering_ops)

    isempty(symm) && return _getLocalSpace_no_symmetry(mwirops, tags)

    # Step 2 – decompose local space into symmetry sectors
    ortho_vecs, space_list = decompose_space(symm, weights, lowering_ops)
    
    # Convert space_list to splist format for TLArray.spaces field
    # First two legs share the same local space structure
    local_splist = _mult_ind_to_splist(symm, space_list)

    dir = ['+', '-', '-']
    Telum = Dict{Symbol, TLArray}()
    for (name, mwirop) in mwirops
        # Step 3 – get full IROP in the symmetry-adapted basis
        irop_full, irop_qlabel = get_IROP(symm, weights, lowering_ops, mwirop)
        transf_basis!(irop_full, ortho_vecs)
        #println("Tranformed IROP for $name:")
        #display(Array(irop_full))

        # Step 4 – compress each IROP matrix into a TLArray object
        data_full = decompose_irop(symm, irop_full, space_list, irop_qlabel)
        # Reduce to 2D if the third dimension is singleton.
        reduced = reduce_irop_blocks(symm, data_full)

        qd = length(first(reduced)[1])
        inds = Tuple(TLIndex(tags[i], dir[i]) for i in 1:3)
        
        # Build spaces tuple: first two legs use local_splist, third leg (if exists) is read from the decomposed blocks.
        if qd == 2
            spaces = (local_splist, local_splist)
        else
            third_spset = Set{qlabeltype(symm)}()
            third_splist = eltype(local_splist)[]
            for (sector_qlabels, block) in reduced
                qlabel = sector_qlabels[3]
                qlabel in third_spset && continue
                push!(third_spset, qlabel)
                push!(third_splist, (qlabel, size(block, 3)))
            end
            sort!(third_splist; by = first, alg=MergeSort)
            spaces = (local_splist, local_splist, third_splist)
        end
        
        fields = _localspace_cgt_fields(reduced, symm, spaces)
        q = TLArray(symm, fields.qlabels, fields.wmatdata, fields.wmatinfo, fields.RMTs, inds[1:qd], spaces)
        Telum[name] = q
    end
    return NamedTuple(Telum)
end

function _getLocalSpace_no_symmetry(mwirops::Dict{Symbol, <:AbstractMatrix},
                                    tags::Tuple{Vararg{AbstractString, 3}})
    isempty(mwirops) && throw(ArgumentError("no-symmetry local space has no operators"))

    first_op = first(values(mwirops))
    spdim = size(first_op, 1)
    size(first_op, 2) == spdim ||
        throw(ArgumentError("local-space operators must be square"))

    symm = ()
    qlabels = [((), ())]
    spaces = ([(() , spdim)], [(() , spdim)])
    inds = (TLIndex(tags[1], '+'), TLIndex(tags[2], '-'))
    wmatdata = Float64[]
    wmatinfo = [_empty_wmat_info(Val(0))]

    tensors = Dict{Symbol, TLArray}()
    for (name, op) in mwirops
        size(op) == (spdim, spdim) ||
            throw(ArgumentError("operator $name has size $(size(op)), expected ($spdim, $spdim)"))
        RMTs = [Matrix{Float64}(op)]
        tensors[name] = TLArray(symm, qlabels, copy(wmatdata), copy(wmatinfo), RMTs, inds, spaces)
    end
    return NamedTuple(tensors)
end
