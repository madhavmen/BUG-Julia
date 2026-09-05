# Many-input TLArray summation.
#
# The hot path aligns inputs once, groups all sectors by physical q-labels, and
# compresses each multi-contribution sector in a single QR/RMT pass.

function _enum_leg_perms!(results, candidates, current, used, QD)
    i = length(current) + 1
    if i > QD
        push!(results, Tuple(current))
        return
    end
    for j in candidates[i]
        j ∈ used && continue
        push!(current, j)
        push!(used, j)
        _enum_leg_perms!(results, candidates, current, used, QD)
        pop!(current)
        delete!(used, j)
    end
end

# Find the unique permutation perm such that inds2[perm[i]] == inds1[i] and
# spaces2[perm[i]] == spaces1[i] for all i.
function _find_leg_permutation(inds1::NTuple{QD, TLIndex}, spaces1,
                               inds2::NTuple{QD, TLIndex}, spaces2) where {QD}
    candidates = Vector{Vector{Int}}(undef, QD)
    for i in 1:QD
        cs = Int[]
        for j in 1:QD
            if inds2[j] == inds1[i] && spaces2[j] == spaces1[i]
                push!(cs, j)
            end
        end
        isempty(cs) && error(
            "No leg in TLArray matches leg $i of reference TLArray " *
            "(itag=\"$(inds1[i].itags)\", dir='$(inds1[i].dir)')")
        candidates[i] = cs
    end
    results = Vector{NTuple{QD, Int}}()
    _enum_leg_perms!(results, candidates, Int[], Set{Int}(), QD)
    isempty(results) && error(
        "No valid bijective permutation found to match TLArray leg indices")
    length(results) > 1 && error(
        "Ambiguous permutation: $(length(results)) ways to match TLArray leg indices")
    return results[1]
end

function _align_sum_input(ref::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT1},
                          q::AbstractTLArray{TQ, QD, N, RD, QT, PS, M, RMT2}) where {T, TQ, QD, N, RD, QT, PS, M, RMT1, RMT2}
    if inds(ref) == inds(q) && spaces(ref) == spaces(q)
        return q
    end
    perm = _find_leg_permutation(inds(ref), spaces(ref), inds(q), spaces(q))
    return permutedims(q, perm)
end

function _needs_sum_alignment(ref::AbstractTLArray, q::AbstractTLArray)
    return inds(ref) != inds(q) || spaces(ref) != spaces(q)
end

function _align_sum_inputs(qs::Tuple{AbstractTLArray, Vararg{AbstractTLArray}})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
    ref = qs[1]
    any(q -> _needs_sum_alignment(ref, q), qs) || return qs
    return ntuple(i -> _align_sum_input(ref, qs[i]), Val(length(qs)))
end

function _align_sum_inputs(qs::AbstractVector{<:AbstractTLArray})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
    ref = qs[1]
    if !any(q -> _needs_sum_alignment(ref, q), qs)
        return qs
    end
    aligned = Vector{AbstractTLArray}(undef, length(qs))
    aligned[1] = ref
    for i in 2:length(qs)
        aligned[i] = _align_sum_input(ref, qs[i])
    end
    return aligned
end

@inline _sum_sector_key(q::AbstractTLArray{T, QD}, sector_index::Int) where {T, QD} =
    ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD))

function _sum_sector_table(qs, ::Type{QT}, ::Val{QD}) where {QT, QD}
    total = 0
    for q in qs
        total += nsectors(q)
    end

    table = Vector{Tuple{NTuple{QD, QT}, Int, Int}}(undef, total)
    pos = 1
    for input_index in eachindex(qs)
        q = qs[input_index]
        for sector_index in sector_slots(q)
            is_sector_zero(q, sector_index) && continue
            table[pos] = (_sum_sector_key(q, sector_index), input_index, sector_index)
            pos += 1
        end
    end
    resize!(table, pos - 1)
    sort!(table; by = first)

    unique_count = 0
    pos = 1
    while pos <= length(table)
        unique_count += 1
        key = table[pos][1]
        pos += 1
        while pos <= length(table) && table[pos][1] == key
            pos += 1
        end
    end
    return table, unique_count
end

@inline _sum_rmt_iszero(rmt::AbstractArray) = iszero(sum(abs2, rmt))
@inline _sum_rmt_iszero(rmt::DiagRMT) = iszero(sum(abs2, rmt.diag))

@inline _sum_scale(::Type{T}, alpha) where {T} = convert(T, alpha)

@inline function _sum_processed_rmt_dims(q::TLArray, sector::Int, ::Val{RD}) where {RD}
    return size(sector_rmt_materialized(q, sector))
end

function _sum_processed_rmt_dims(q::TLArrayView{T, QD}, sector::Int, ::Val{RD}) where {T, QD, RD}
    source_dims = size(sector_rmt_materialized(q.arr, sector))
    return ntuple(d -> d <= QD ? source_dims[q.perm[d]] : source_dims[d], Val(RD))
end

@inline function _sum_reference_safe(q::TLArray{T, QD, N, RD}, sector::Int, ::Type{T}) where {T, QD, N, RD}
    rmt = sector_rmt_materialized(q, sector)
    return rmt isa Array{T, RD} || rmt isa DiagRMT
end

@inline _sum_reference_safe(q::TLArray, sector::Int, ::Type{T}) where {T} = false

@inline function _sum_reference_safe(q::TLArrayView{T, QD}, sector::Int, ::Type{RT}) where {T, QD, RT}
    q.perm == _identity_phys_perm(Val(QD)) || return false
    (!q.conj || T <: Real) || return false
    return _sum_reference_safe(q.arr, sector, RT)
end

@inline _sum_needs_buffer(q::AbstractTLArray, sector::Int, ::Type{T}) where {T} =
    !_sum_reference_safe(q, sector, T)

function _sum_prepare_rmt_buffers(qs, ::Type{T}, ::Val{RD}) where {T, RD}
    buffers = Vector{Vector{T}}(undef, length(qs))
    needs_buffer = falses(length(qs))
    for input_index in eachindex(qs)
        q = qs[input_index]
        max_len = 0
        for sector in sector_slots(q)
            is_sector_zero(q, sector) && continue
            _sum_needs_buffer(q, sector, T) || continue
            dims = _sum_processed_rmt_dims(q, sector, Val(RD))
            max_len = max(max_len, prod(dims; init=1))
        end
        if max_len > 0
            buffers[input_index] = Vector{T}(undef, max_len)
            needs_buffer[input_index] = true
        end
    end
    return buffers, needs_buffer
end

function _sum_wmat_reserve_for_entry(qs, entry, ::Val{M}) where {M}
    _, input_index, sector_index = entry
    q = qs[input_index]
    return ntuple(m -> length(sector_wmat_slot(q, sector_index, m)), Val(M))
end

function _sum_wmat_reserve_for_interval(qs, table, interval::UnitRange{Int},
                                        ::Val{M}) where {M}
    return ntuple(Val(M)) do m
        _, first_input, first_sector = table[first(interval)]
        first_wmat = sector_wmat_slot(qs[first_input], first_sector, m)
        nrows = size(first_wmat, 1)
        nrows == 1 && return 1

        ncols = 0
        for pos in interval
            _, input_index, sector_index = table[pos]
            wmat = sector_wmat_slot(qs[input_index], sector_index, m)
            @assert size(wmat, 1) == nrows "_sum_wmat_reserve_for_interval requires a common sector dimension"
            ncols += size(wmat, 2)
        end
        nrows * min(ncols, nrows)
    end
end

function _sum_wmat_arena_size_upper_bound(qs, table, ::Val{M}) where {M}
    total = 0
    pos = 1
    while pos <= length(table)
        last = pos
        while last < length(table) && table[last + 1][1] == table[pos][1]
            last += 1
        end
        reserve = pos == last ?
            _sum_wmat_reserve_for_entry(qs, table[pos], Val(M)) :
            _sum_wmat_reserve_for_interval(qs, table, pos:last, Val(M))
        total += sum(reserve; init=0)
        pos = last + 1
    end
    return total
end

function _sum_store_wmats!(result_wmatdata::Vector{Float64},
                           result_wmatinfo::Vector{WMatInfo{M}},
                           next_offset::Int,
                           sector_wmats::NTuple{M, <:AbstractMatrix},
                           reserve::NTuple{M, Int},
                           ::Val{M}) where {M}
    info = ntuple(Val(M)) do m
        wmat = sector_wmats[m]
        len = length(wmat)
        len == 0 && return (0, 0, 0)
        reserved_len = reserve[m]
        len <= reserved_len ||
            throw(ArgumentError("sum w-matrix slot $m needs $len arena entries but only $reserved_len were reserved"))
        offset = next_offset
        copyto!(view(result_wmatdata, offset:offset + len - 1), vec(wmat))
        next_offset += reserved_len
        (offset, size(wmat, 1), size(wmat, 2))
    end
    push!(result_wmatinfo, info)
    return next_offset
end

function _sum_copy_scaled!(dest::Array{T, RD}, source::AbstractArray, scale) where {T, RD}
    cscale = _sum_scale(T, scale)
    copyto!(dest, source)
    cscale == one(T) || rmul!(dest, cscale)
    return dest
end

function _sum_copy_processed_unscaled!(dest, q::TLArray, sector::Int, ::Type{T}) where {T}
    source = sector_rmt_materialized(q, sector)
    @inbounds for I in CartesianIndices(dest)
        dest[I] = source[I]
    end
    return dest
end

function _sum_copy_processed_unscaled!(dest, q::TLArrayView{T, QD, N, RD}, sector::Int, ::Type{RT}) where {T, QD, N, RD, RT}
    source, _ = sector_rmt_with_scale(q.arr, sector)
    invperm = _phys_to_stored_order(q.perm)
    @inbounds for I in CartesianIndices(dest)
        source_I = ntuple(d -> d <= QD ? I[invperm[d]] : I[d], Val(RD))
        value = source[source_I...]
        q.conj && (value = conj(value))
        dest[I] = value
    end
    return dest
end

function _sum_sector_scale(q::TLArray{T}, sector::Int, ::Type{RT}) where {T, RT}
    return one(RT)
end

function _sum_sector_scale(q::TLArrayView, sector::Int, ::Type{RT}) where {RT}
    _, alpha = sector_rmt_with_scale(q.arr, sector)
    scale = alpha * q.scale
    q.conj && (scale = conj(scale))
    return _sum_scale(RT, scale)
end

function _sum_prepare_sector_rmt!(buffers::Vector{Vector{T}},
                                  needs_buffer::BitVector,
                                  input_index::Int,
                                  q::AbstractTLArray,
                                  sector::Int,
                                  ::Type{T},
                                  ::Val{RD}) where {T, RD}
    if !_sum_needs_buffer(q, sector, T)
        rmt, alpha = sector_rmt_with_scale(q, sector)
        return rmt, _sum_scale(T, alpha), false
    end

    @assert needs_buffer[input_index]
    dims = _sum_processed_rmt_dims(q, sector, Val(RD))
    len = prod(dims; init=1)
    @assert len <= length(buffers[input_index])
    workspace = reshape(view(buffers[input_index], 1:len), dims)
    _sum_copy_processed_unscaled!(workspace, q, sector, T)
    return workspace, _sum_sector_scale(q, sector, T), true
end

function _sum_single_contribution!(result_keys::Vector{NTuple{QD, QT}},
                                   result_wmatdata::Vector{Float64},
                                   result_wmatinfo::Vector{WMatInfo{M}},
                                   next_wmat_offset::Int,
                                   result_RMTs::Vector{Array{T, RD}},
                                   qs,
                                   buffers::Vector{Vector{T}},
                                   needs_buffer::BitVector,
                                   entry::Tuple{NTuple{QD, QT}, Int, Int},
                                   ::Val{QD}) where {T, QD, QT, M, RD}
    key, input_index, sector_index = entry
    q = qs[input_index]
    source_rmt, alpha, workspace_backed =
        _sum_prepare_sector_rmt!(buffers, needs_buffer, input_index, q, sector_index, T, Val(RD))
    if !workspace_backed && alpha == one(T) && source_rmt isa Array{T, RD}
        rmt = source_rmt
    else
        rmt = Array{T, RD}(undef, size(source_rmt))
        _sum_copy_scaled!(rmt, source_rmt, alpha)
    end
    _sum_rmt_iszero(rmt) && return next_wmat_offset
    push!(result_keys, key)
    next_wmat_offset = _sum_store_wmats!(
        result_wmatdata, result_wmatinfo, next_wmat_offset,
        ntuple(m -> sector_wmat_slot(q, sector_index, m), Val(M)),
        _sum_wmat_reserve_for_entry(qs, entry, Val(M)),
        Val(M))
    push!(result_RMTs, rmt)
    return next_wmat_offset
end

function _sum_multi_contribution!(result_keys::Vector{NTuple{QD, QT}},
                                  result_wmatdata::Vector{Float64},
                                  result_wmatinfo::Vector{WMatInfo{M}},
                                  next_wmat_offset::Int,
                                  result_RMTs::Vector{Array{T, RD}},
                                  qs,
                                  buffers::Vector{Vector{T}},
                                  needs_buffer::BitVector,
                                  table,
                                  interval::UnitRange{Int},
                                  key::NTuple{QD, QT},
                                  ::Type{PS},
                                  ::Val{QD},
                                  ::Val{N},
                                  ::Val{M}) where {T, QD, QT, M, RD, N, PS}
    compressed = _compress_sum_sector(qs, table, interval, buffers, needs_buffer, T,
                                      PS, Val(QD), Val(N), Val(M), Val(RD))
    isnothing(compressed) && return next_wmat_offset

    sector_wmats, result_RMT = compressed
    _sum_rmt_iszero(result_RMT) && return next_wmat_offset

    push!(result_keys, key)
    next_wmat_offset = _sum_store_wmats!(
        result_wmatdata, result_wmatinfo, next_wmat_offset, sector_wmats,
        _sum_wmat_reserve_for_interval(qs, table, interval, Val(M)),
        Val(M))
    push!(result_RMTs, result_RMT)
    return next_wmat_offset
end

function _sum_prepare_wmat_slot(qs, table, interval::UnitRange{Int}, m::Int,
                                tol::Float64)
    _, first_input, first_sector = table[first(interval)]
    first_wmat = sector_wmat_slot(qs[first_input], first_sector, m)
    nrows = size(first_wmat, 1)

    if nrows == 1
        factors = Vector{Matrix{Float64}}(undef, length(interval))
        out_pos = 1
        for pos in interval
            _, input_index, sector_index = table[pos]
            factors[out_pos] = sector_wmat_slot(qs[input_index], sector_index, m)
            out_pos += 1
        end
        return [1.0;;], factors
    end

    ncols = 0
    for pos in interval
        _, input_index, sector_index = table[pos]
        wmat = sector_wmat_slot(qs[input_index], sector_index, m)
        @assert size(wmat, 1) == nrows "_sum_prepare_wmat_slot requires a common sector dimension"
        ncols += size(wmat, 2)
    end

    concat = Matrix{Float64}(undef, nrows, ncols)
    col = 1
    @views for pos in interval
        _, input_index, sector_index = table[pos]
        wmat = sector_wmat_slot(qs[input_index], sector_index, m)
        width = size(wmat, 2)
        copyto!(view(concat, :, col:(col + width - 1)), wmat)
        col += width
    end
    @assert col == ncols + 1

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
    factors = Vector{Matrix{Float64}}(undef, length(interval))
    col = 0
    out_pos = 1
    for pos in interval
        _, input_index, sector_index = table[pos]
        width = size(sector_wmat_slot(qs[input_index], sector_index, m), 2)
        factors[out_pos] = R[:, (col + 1):(col + width)]
        col += width
        out_pos += 1
    end
    @assert col == ncols

    return Q, factors
end

function _sum_combined_factor(factors::Matrix{Matrix{Float64}},
                              contribution_index::Int,
                              ::Val{M}) where {M}
    K = ones(Float64, 1, 1)
    rank_dim = 1
    old_dim = 1

    for m in 1:M
        factor = factors[m, contribution_index]
        r_n, old_n = size(factor)
        K_new = Matrix{Float64}(undef, rank_dim * r_n, old_dim * old_n)
        @inbounds for old_idx in 1:old_dim, old_n_idx in 1:old_n,
                      rank_idx in 1:rank_dim, rank_n_idx in 1:r_n
            K_new[rank_idx + (rank_n_idx - 1) * rank_dim,
                  old_idx + (old_n_idx - 1) * old_dim] =
                K[rank_idx, old_idx] * factor[rank_n_idx, old_n_idx]
        end
        K = K_new
        rank_dim *= r_n
        old_dim *= old_n
    end

    return K
end

function _sum_sector_rank_sizes(factors::Matrix{Matrix{Float64}},
                                ::Type{PS},
                                ::Val{N},
                                ::Val{M}) where {PS, N, M}
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    return ntuple(Val(N)) do n
        slot = findfirst(==(n), nonabelian_indices)
        isnothing(slot) ? 1 : size(factors[slot, 1], 1)
    end
end

function _accumulate_sum_rmts!(
    result_mat::Matrix{T},
    qs,
    table,
    interval::UnitRange{Int},
    buffers::Vector{Vector{T}},
    needs_buffer::BitVector,
    factors::Matrix{Matrix{Float64}},
    physical_dim::Int,
    physical_sizes::NTuple{QD, Int},
    source_rank_sizes::NTuple{N, Int},
    ::Val{M},
    ::Val{RD},
) where {T, QD, N, RD, M}
    for (i, pos) in enumerate(interval)
        _, input_index, sector_index = table[pos]
        source, alpha, _ =
            _sum_prepare_sector_rmt!(buffers, needs_buffer, input_index, qs[input_index],
                                     sector_index, T, Val(RD))
        factor = _sum_combined_factor(factors, i, Val(M))
        beta = i == 1 ? zero(T) : one(T)
        _sum_accumulate_one!(result_mat, source, factor, physical_dim,
                             physical_sizes, source_rank_sizes, alpha, beta)
    end
    return result_mat
end

function _accumulate_sum_rmts!(
    result_mat::Matrix{T},
    qs,
    table,
    interval::UnitRange{Int},
    buffers::Vector{Vector{T}},
    needs_buffer::BitVector,
    factors::Matrix{Matrix{Float64}},
    physical_dim::Int,
    physical_sizes::NTuple{QD, Int},
    source_rank_sizes::NTuple{N, Int},
    ::Val{1},
    ::Val{RD},
) where {T, QD, N, RD}
    for (i, pos) in enumerate(interval)
        _, input_index, sector_index = table[pos]
        source, alpha, _ =
            _sum_prepare_sector_rmt!(buffers, needs_buffer, input_index, qs[input_index],
                                     sector_index, T, Val(RD))
        factor = factors[1, i]
        beta = i == 1 ? zero(T) : one(T)
        _sum_accumulate_one!(result_mat, source, factor, physical_dim,
                             physical_sizes, source_rank_sizes, alpha, beta)
    end
    return result_mat
end

function _sum_accumulate_one!(result_mat::Matrix{T},
                              source::AbstractArray,
                              factor::Matrix{Float64},
                              physical_dim::Int,
                              physical_sizes,
                              source_rank_sizes,
                              alpha,
                              beta) where {T}
    source_mat = reshape(source, physical_dim, div(length(source), physical_dim))
    @assert size(source_mat, 2) == size(factor, 2)
    @assert size(result_mat, 2) == size(factor, 1)
    mul!(result_mat, source_mat, transpose(factor), alpha, beta)
    return result_mat
end

function _sum_accumulate_one!(result_mat::Matrix{T},
                              source::DiagRMT{S, RD},
                              factor::Matrix{Float64},
                              physical_dim::Int,
                              physical_sizes::NTuple{QD, Int},
                              source_rank_sizes::NTuple{N, Int},
                              alpha,
                              beta) where {T, S, RD, QD, N}
    beta == zero(T) ? fill!(result_mat, zero(T)) : (beta == one(T) || lmul!(beta, result_mat))
    @assert size(result_mat, 2) == size(factor, 1)
    physical_linear = LinearIndices(physical_sizes)
    rank_linear = LinearIndices(source_rank_sizes)
    ax1, ax2 = source.axis
    @inbounds for diag_index in eachindex(source.diag)
        full_index = ntuple(d -> (d == ax1 || d == ax2) ? diag_index : 1, Val(RD))
        physical_index = ntuple(d -> full_index[d], Val(QD))
        rank_index = ntuple(n -> full_index[QD + n], Val(N))
        row = physical_linear[physical_index...]
        col = rank_linear[rank_index...]
        coeff = alpha * source.diag[diag_index]
        for out_col in axes(result_mat, 2)
            result_mat[row, out_col] += coeff * factor[out_col, col]
        end
    end
    return result_mat
end

function _compress_sum_sector(
    qs,
    table,
    interval::UnitRange{Int},
    buffers::Vector{Vector{T}},
    needs_buffer::BitVector,
    ::Type{T},
    ::Type{PS},
    ::Val{QD},
    ::Val{N},
    ::Val{M},
    ::Val{RD},
    tol::Float64 = 1e-12,
) where {T, RD, PS, QD, N, M}
    K = length(interval)
    sector_wmats = Vector{Matrix{Float64}}(undef, M)
    factors = Matrix{Matrix{Float64}}(undef, M, K)

    for m in 1:M
        prepared = _sum_prepare_wmat_slot(qs, table, interval, m, tol)
        isnothing(prepared) && return nothing
        common_iso, split_factors = prepared
        sector_wmats[m] = common_iso
        for i in 1:K
            factors[m, i] = split_factors[i]
        end
    end

    _, first_input, first_sector = table[first(interval)]
    source_shape = _sum_processed_rmt_dims(qs[first_input], first_sector, Val(RD))
    physical_sizes = ntuple(i -> source_shape[i], Val(QD))
    physical_dim = prod(physical_sizes; init=1)
    source_rank_sizes = ntuple(i -> source_shape[QD + i], Val(N))
    rank_sizes = _sum_sector_rank_sizes(factors, PS, Val(N), Val(M))
    rank_dim = prod(rank_sizes; init=1)
    result_dims::NTuple{RD, Int} = (physical_sizes..., rank_sizes...)
    result_data::Array{T, RD} = Array{T, RD}(undef, result_dims)
    result_mat = reshape(result_data, physical_dim, rank_dim)

    _accumulate_sum_rmts!(result_mat, qs, table, interval, buffers, needs_buffer,
                          factors, physical_dim, physical_sizes, source_rank_sizes,
                          Val(M), Val(RD))

    return ntuple(m -> sector_wmats[m], Val(M)), result_data
end

function _sum_tlarrays_aligned(qs, ::Type{T}, ::Val{QD}, ::Val{N}, ::Type{QT}, ::Type{PS},
                               ::Val{M}, ::Val{RD}) where {T, QD, N, RD, QT, PS, M}
    table, unique_count = _sum_sector_table(qs, QT, Val(QD))
    result_keys = Vector{NTuple{QD, QT}}()
    result_wmatdata = Vector{Float64}(undef, _sum_wmat_arena_size_upper_bound(qs, table, Val(M)))
    result_wmatinfo = Vector{WMatInfo{M}}()
    result_RMTs = Vector{Array{T, RD}}()
    sizehint!(result_keys, unique_count)
    sizehint!(result_wmatinfo, unique_count)
    sizehint!(result_RMTs, unique_count)
    buffers, needs_buffer = _sum_prepare_rmt_buffers(qs, T, Val(RD))

    next_wmat_offset = 1
    pos = 1
    while pos <= length(table)
        key = table[pos][1]
        last = pos
        while last < length(table) && table[last + 1][1] == key
            last += 1
        end
        interval = pos:last
        if pos == last
            next_wmat_offset = _sum_single_contribution!(
                result_keys, result_wmatdata, result_wmatinfo, next_wmat_offset,
                result_RMTs, qs, buffers, needs_buffer, table[pos], Val(QD))
        else
            next_wmat_offset = _sum_multi_contribution!(
                result_keys, result_wmatdata, result_wmatinfo, next_wmat_offset,
                result_RMTs, qs, buffers, needs_buffer, table, interval, key, PS,
                Val(QD), Val(N), Val(M))
        end
        pos = last + 1
    end

    qlabels = Matrix{QT}(undef, QD, length(result_keys))
    for sector_index in eachindex(result_keys), leg in 1:QD
        qlabels[leg, sector_index] = result_keys[sector_index][leg]
    end

    ref = qs[begin]
    return TLArray(symm(ref), qlabels, result_wmatdata, result_wmatinfo, result_RMTs,
                   inds(ref), spaces(ref))
end

function _sum_tlarrays(qs::Tuple{})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
end

function _sum_tlarrays_from_ref(ref::AbstractTLArray{TR, QD, N, RD, QT, PS, M, RMTR}, qs) where {TR, QD, N, RD, QT, PS, M, RMTR}
    for q in qs
        q isa AbstractTLArray || throw(ArgumentError("TLArray sum entries must all be AbstractTLArray objects"))
        q isa AbstractTLArray{<:Number} || throw(ArgumentError("TLArray sum entries must have numeric RMT element types"))
        ndims(q) == QD && nsymms(q) == N && typeof(q).parameters[4] == RD &&
            qlabeltype(q) == QT && productsymm(q) === PS &&
            typeof(q).parameters[7] == M || throw(ArgumentError(
            "TLArray sum requires matching QD, N, RD, qlabel type, product symmetry, " *
            "and w-matrix tuple width; only RMT element/storage may differ"))
    end
    aligned = _align_sum_inputs(qs)
    T = promote_type((_tlarray_eltype(q) for q in aligned)...)
    return _sum_tlarrays_aligned(aligned, T, Val(QD), Val(N), QT, PS, Val(M), Val(RD))
end

function _sum_tlarrays(qs::Union{Tuple{AbstractTLArray, Vararg{AbstractTLArray}},
                                 AbstractVector{<:AbstractTLArray}})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
    return _sum_tlarrays_from_ref(qs[begin], qs)
end

# Keep tuple overloads anchored on concrete tensor containers. A broad
# Tuple{<:AbstractTLArray,...} sum method invalidates Julia's REPL hint path.
Base.sum(qs::Tuple{TLArray, Vararg{AbstractTLArray}}) =
    _sum_tlarrays(qs)
Base.sum(qs::Tuple{TLArrayView, Vararg{AbstractTLArray}}) =
    _sum_tlarrays(qs)
Base.sum(qs::AbstractVector{<:AbstractTLArray}) =
    _sum_tlarrays(qs)

function Base.:+(qs1::TLArray{T1, QD, N, RD, QT, PS}, 
    qs2::TLArray{T2, QD, N, RD, QT, PS}) where {T1, T2, QD, N, RD, QT, PS}
    return _sum_tlarrays((qs1, qs2))
end

Base.:-(qs1::TLArray, qs2::TLArray) = qs1 + (-1 * qs2)
Base.:+(q::TLArray, x::Number) = q + x * _identity_on_tlarray(q)
Base.:+(x::Number, q::TLArray) = q + x
Base.:-(q::TLArray, x::Number) = q + (-x) * _identity_on_tlarray(q)
Base.:-(x::Number, q::TLArray) = x * _identity_on_tlarray(q) + (-q)
Base.:+(qs1::AbstractTLArray, qs2::AbstractTLArray) = _sum_tlarrays((qs1, qs2))
Base.:-(qs1::AbstractTLArray, qs2::AbstractTLArray) = qs1 + (-1 * qs2)
Base.:+(q::AbstractTLArray, x::Number) = q + x * _identity_on_tlarray(q)
Base.:+(x::Number, q::AbstractTLArray) = q + x
Base.:-(q::AbstractTLArray, x::Number) = q + (-x) * _identity_on_tlarray(q)
Base.:-(x::Number, q::AbstractTLArray) = x * _identity_on_tlarray(q) + (-q)
