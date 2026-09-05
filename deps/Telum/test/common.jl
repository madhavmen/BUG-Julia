using LinearAlgebra
using Random
using SparseArrays
using Test
using LurCGT
using Telum
import Telum: _compute_spaces

include("test_utils.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: reconstruct original array from (U, SV) returned by svd_leg.
#
# U   : (dim_leg, χ)
# SV  : same shape as original array, but with the `leg`-th dim replaced by χ
#
# Reconstruction contracts U[i, χ] with SV[..., χ, ...] over χ,
# placing i back into position `leg`.
# ─────────────────────────────────────────────────────────────────────────────
function reconstruct(U::AbstractArray, SV::AbstractArray, leg::Integer)
    N    = ndims(SV)
    chi  = size(SV, leg)

    # Move χ axis of SV to the front  →  (χ, other dims...)
    other_legs   = [i for i in 1:N if i != leg]
    perm_to_front = (leg, other_legs...)
    SV_front     = permutedims(SV, perm_to_front)

    # Matrix multiply:  U (dim_leg × χ)  *  SV_front_mat (χ × rest)
    rec_mat = U * reshape(SV_front, chi, :)

    # Reshape and permute back to original leg order
    rec_shape = (size(U, 1), [size(SV, i) for i in other_legs]...)
    rec_perm  = invperm(collect(perm_to_front))
    return permutedims(reshape(rec_mat, rec_shape...), rec_perm)
end

function _dense_addSingleton_ref(arr::AbstractArray, legs)
    positions = legs isa Integer ? [Int(legs)] : sort(Int[i for i in legs])
    old_dims = size(arr)
    new_dims = Int[]
    sizehint!(new_dims, ndims(arr) + length(positions))

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:(ndims(arr) + length(positions))
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            push!(new_dims, 1)
            insert_idx += 1
        else
            push!(new_dims, old_dims[old_leg])
            old_leg += 1
        end
    end

    return reshape(arr, Tuple(new_dims))
end

function _dense_tensor_product_ref(a::AbstractArray, b::AbstractArray)
    da = size(a)
    db = size(b)
    ar = reshape(a, da..., ones(Int, ndims(b))...)
    br = reshape(b, ones(Int, ndims(a))..., db...)
    return ar .* br
end

function _normalize_dim_tuple(dimensions, QD::Int)
    dims = dimensions isa Integer ? (Int(dimensions),) : Tuple(Int(d) for d in dimensions)
    return Tuple(sort(collect(dims)))
end

function _normalize_matrix_dim_tuple(dimensions, QD::Int)
    dimensions isa Tuple && length(dimensions) == 2 || error("matrix dimensions must have length 2")
    dims1 = _normalize_dim_tuple(dimensions[1], QD)
    dims2 = _normalize_dim_tuple(dimensions[2], QD)
    return dims1, dims2
end

function _sum_splist_many_ref(splists)
    result = copy(first(splists))
    empty!(result)
    dims = Dict{Any, Int}()
    seen = Set{Any}()

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

function _dense_leg_index_map(symm::Tuple, src_splist::Vector, dst_splist::Vector; offset_sizes=Dict{Any, Int}())
    src_offsets, src_total = get_offset(symm, src_splist)
    dst_offsets, _ = get_offset(symm, dst_splist)
    idxmap = zeros(Int, src_total)

    for (qlabels, src_range) in src_offsets
        dst_range = dst_offsets[qlabels]
        start = first(dst_range) + get(offset_sizes, qlabels, 0)
        for (src_idx, dst_idx) in zip(src_range, start:start+length(src_range)-1)
            idxmap[src_idx] = dst_idx
        end
    end

    return idxmap
end

function _dense_vector_start_maps(qs, dims)
    QD = length(first(qs).inds)
    dims_set = Set(dims)
    running = [Dict{Any, Int}() for _ in 1:QD]
    starts = Vector{Vector{Dict{Any, Int}}}(undef, length(qs))

    for (qi, q) in enumerate(qs)
        starts[qi] = [Dict{Any, Int}() for _ in 1:QD]
        for leg in 1:QD
            leg ∈ dims_set || continue
            leg_offsets, _ = get_offset(symm(q), q.spaces[leg])
            for (qlabels, _) in q.spaces[leg]
                start = get(running[leg], qlabels, 1)
                starts[qi][leg][qlabels] = start
                running[leg][qlabels] = start + length(leg_offsets[qlabels])
            end
        end
    end

    return starts
end

function _dense_vector_oplus_ref(qs, dimensions)
    qs_vec = collect(qs)
    dims = _normalize_dim_tuple(dimensions, length(first(qs_vec).inds))
    dims_set = Set(dims)
    ref = first(qs_vec)

    result_spaces = ntuple(leg -> begin
        leg in dims_set ? _sum_splist_many_ref([q.spaces[leg] for q in qs_vec]) : copy(ref.spaces[leg])
    end, length(ref.inds))

    result_sizes = ntuple(leg -> begin
        get_offset(symm(ref), result_spaces[leg])[2]
    end, length(ref.inds))
    T = promote_type((eltype(Array(to_sparse_array(q))) for q in qs_vec)...)
    result = zeros(T, result_sizes)
    start_maps = _dense_vector_start_maps(qs_vec, dims)

    for (q, qstarts) in zip(qs_vec, start_maps)
        arr = Array(to_sparse_array(q))
        idxmaps = Vector{Vector{Int}}(undef, length(q.inds))
        for leg in 1:length(q.inds)
            if leg in dims_set
                offset_sizes = Dict{Any, Int}(qlabels => start - 1 for (qlabels, start) in qstarts[leg])
                idxmaps[leg] = _dense_leg_index_map(symm(q), q.spaces[leg], result_spaces[leg]; offset_sizes=offset_sizes)
            else
                idxmaps[leg] = collect(1:size(arr, leg))
            end
        end

        for I in CartesianIndices(arr)
            dst = ntuple(d -> idxmaps[d][I[d]], ndims(arr))
            result[dst...] += arr[I]
        end
    end

    return result, result_spaces
end

function _dense_matrix_entry(mat, i::Int, j::Int)
    if applicable(isassigned, mat, i, j) && !isassigned(mat, i, j)
        return nothing
    end
    val = mat[i, j]
    return (val === nothing || val === missing) ? nothing : val
end

function _zero_tlarray_ref_like(ref::TLArray{TQ, QD, N, RD}, spaces; T::Type=Float64) where {TQ, QD, N, RD}
    return Telum._zero_tlarray_with_spaces(symm(ref), ref.inds, spaces; T=T)
end

function _matrix_axis_start_maps(sources, dims, QD::Int, symm::Tuple)
    running = [Dict{Any, Int}() for _ in 1:QD]
    starts = [Dict{Int, Dict{Any, Int}}() for _ in eachindex(sources)]

    for idx in eachindex(sources)
        for leg in dims
            startmap = Dict{Any, Int}()
            leg_offsets, _ = get_offset(symm, sources[idx][leg])
            for (qlabels, _) in sources[idx][leg]
                start = get(running[leg], qlabels, 1)
                startmap[qlabels] = start
                running[leg][qlabels] = start + length(leg_offsets[qlabels])
            end
            starts[idx][leg] = startmap
        end
    end

    return starts
end

function _dense_matrix_oplus_ref(mat, dimensions)
    defined = TLArray[]
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _dense_matrix_entry(mat, i, j)
        q === nothing || push!(defined, q)
    end
    isempty(defined) && error("matrix reference requires at least one defined TLArray")

    ref = first(defined)
    row_dims, col_dims = _normalize_matrix_dim_tuple(dimensions, length(ref.inds))
    row_dims_set = Set(row_dims)
    col_dims_set = Set(col_dims)

    row_sources = [Dict{Int, Any}() for _ in axes(mat, 1)]
    col_sources = [Dict{Int, Any}() for _ in axes(mat, 2)]
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _dense_matrix_entry(mat, i, j)
        q === nothing && continue
        for leg in 1:length(ref.inds)
            if leg ∉ col_dims_set
                haskey(row_sources[i], leg) || (row_sources[i][leg] = copy(q.spaces[leg]))
            end
            if leg ∉ row_dims_set
                haskey(col_sources[j], leg) || (col_sources[j][leg] = copy(q.spaces[leg]))
            end
        end
    end

    result_spaces = ntuple(leg -> begin
        if leg ∈ row_dims_set
            _sum_splist_many_ref([row_sources[i][leg] for i in axes(mat, 1)])
        elseif leg ∈ col_dims_set
            _sum_splist_many_ref([col_sources[j][leg] for j in axes(mat, 2)])
        else
            copy(ref.spaces[leg])
        end
    end, length(ref.inds))

    row_starts = _matrix_axis_start_maps(row_sources, row_dims, length(ref.inds), symm(ref))
    col_starts = _matrix_axis_start_maps(col_sources, col_dims, length(ref.inds), symm(ref))
    T = promote_type((eltype(Array(to_sparse_array(q))) for q in defined)...)
    result_sizes = ntuple(leg -> get_offset(symm(ref), result_spaces[leg])[2], length(ref.inds))
    result = zeros(T, result_sizes)

    for j in axes(mat, 2), i in axes(mat, 1)
        q = _dense_matrix_entry(mat, i, j)
        if q === nothing
            spaces = ntuple(leg -> begin
                if haskey(row_sources[i], leg) && haskey(col_sources[j], leg)
                    copy(row_sources[i][leg])
                elseif haskey(row_sources[i], leg)
                    copy(row_sources[i][leg])
                else
                    copy(col_sources[j][leg])
                end
            end, length(ref.inds))
            q = _zero_tlarray_ref_like(ref, spaces; T=T)
        end

        arr = Array(to_sparse_array(q))
        idxmaps = Vector{Vector{Int}}(undef, length(ref.inds))
        for leg in 1:length(ref.inds)
            if leg ∈ row_dims_set
                offsets = Dict{Any, Int}(qlabels => start - 1 for (qlabels, start) in row_starts[i][leg])
                idxmaps[leg] = _dense_leg_index_map(symm(ref), q.spaces[leg], result_spaces[leg]; offset_sizes=offsets)
            elseif leg ∈ col_dims_set
                offsets = Dict{Any, Int}(qlabels => start - 1 for (qlabels, start) in col_starts[j][leg])
                idxmaps[leg] = _dense_leg_index_map(symm(ref), q.spaces[leg], result_spaces[leg]; offset_sizes=offsets)
            else
                idxmaps[leg] = collect(1:size(arr, leg))
            end
        end

        for I in CartesianIndices(arr)
            dst = ntuple(d -> idxmaps[d][I[d]], ndims(arr))
            result[dst...] += arr[I]
        end
    end

    return result, result_spaces
end

# ─────────────────────────────────────────────────────────────────────────────
