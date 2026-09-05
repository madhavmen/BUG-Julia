# ─── eigen ────────────────────────────────────────────────────────────────────
#
# Perform symmetry-adapted eigendecomposition of a rank-2 TLArray object.
#
# Arguments:
#   q          : TLArray{T, 2, N, RD} to decompose (rank 2, one incoming and one outgoing leg)
#   eig_tag    : itag for the new eigenvector bond leg (default "eig")
#   hermitian  : if true, assume each RMT block is Hermitian and skip the
#                Hermiticity check. If false (default), compute
#                norm(q - q') / norm(q) and dispatch to the Hermitian or
#                general eigensolver automatically.
#
# Returns `EigenResult(V, D, V_inv, eig_list)` where:
#   D        : diagonal TLArray with legs normalized to ('+', '-')
#   V        : eigenvector TLArray with legs normalized to ('+', '-')
#   V_inv    : `nothing` on the Hermitian path, inverse eigenvector TLArray on
#              the general path
#   eig_list : Vector{Tuple{eigenval_type, Int, sector_qlabels, sector_index}}
#              — (eigenvalue, degeneracy, sector, in-sector index) entries
#              sorted by ascending eigenvalue (real for Hermitian, abs for non-Hermitian).
#              degeneracy = ∏_n dim(irep at leg 1, symm n) for the sector.
#
# Convention:
#   A = V * D * V'   (Hermitian; eigenvectors are orthonormal columns of V)
#
# ─────────────────────────────────────────────────────────────────────────────

_eig_sector_qlabels(q::TLArray{T, 2, N}, sector_index::Int) where {T, N} =
    ntuple(n -> sector_qlabel(q, sector_index, 1)[n], Val(N))

_eig_sort_value(ev::Real, hermitian::Bool) = hermitian ? real(ev) : abs(ev)
_eig_sort_value(ev::Complex, hermitian::Bool) = hermitian ? real(ev) : abs(ev)
const _EIG_HERMITIAN_RTOL = sqrt(eps(Float64))

struct EigenResult{TV, TD, TVI, TL}
    V::TV
    D::TD
    V_inv::TVI
    eig_list::TL
end

function EigenResult(V::TLArray, D::TLArray, V_inv, eig_list)
    return EigenResult{typeof(V), typeof(D), typeof(V_inv), typeof(eig_list)}(
        V, D, V_inv, eig_list)
end

function _append_missing_eig_sectors!(symm::NTuple{N, Any},
                                      spaces,
                                      covered,
                                      qlabels_D,
                                      RMTs_D,
                                      qlabels_V,
                                      RMTs_V,
                                      qlabels_Vinv,
                                      RMTs_Vinv,
                                      eig_list,
                                      cgp::NTuple{2, Int},
                                      ::Type{T_out}) where {N, T_out}
    for (sector_qlabels, dim) in spaces
        sector_qlabels ∈ covered && continue

        cgt_dim = prod(dimension(symm[n], sector_qlabels[n]) for n in 1:N)
        for j in 1:dim
            push!(eig_list, (zero(T_out), cgt_dim, sector_qlabels, j))
        end

        rmt_eye = reshape(Matrix{T_out}(I, dim, dim), dim, dim, ones(Int, N)...)
        rmt_eye[:] .*= sqrt(Float64(cgt_dim))

        push!(qlabels_V, sector_qlabels)
        push!(RMTs_V, rmt_eye)

        if !isnothing(RMTs_Vinv)
            push!(qlabels_Vinv, sector_qlabels)
            push!(RMTs_Vinv, rmt_eye)
        end
    end
end

function _eig_diagonal_rmt(::Vector{<:Array}, vals::AbstractVector{T}, scale, ::Val{RD}) where {T, RD}
    return _dense_diagonal_rmt_from_values(vals, Val(RD), scale)
end

function _eig_diagonal_rmt(::Vector{<:DiagRMT}, vals::AbstractVector{T}, scale, ::Val{RD}) where {T, RD}
    return _diag_rmt_from_values(vals, Val(RD), (1, 2), scale)
end

# TODO: This is not optimized.
function _renumber_eig_entries(eig_entries)
    isempty(eig_entries) && return copy(eig_entries)
    sector_indices = Dict{typeof(eig_entries[1][3]), Vector{Int}}()
    for entry in eig_entries
        push!(get!(sector_indices, entry[3], Int[]), entry[4])
    end

    sector_maps = Dict{typeof(eig_entries[1][3]), Dict{Int, Int}}()
    for (sector, idxs) in sector_indices
        sector_maps[sector] = Dict(old_idx => new_idx for (new_idx, old_idx) in enumerate(sort(unique(idxs))))
    end

    out = similar(eig_entries)
    for (i, entry) in enumerate(eig_entries)
        ev, deg, sector, old_idx = entry
        out[i] = (ev, deg, sector, sector_maps[sector][old_idx])
    end
    return out
end

function _eig_qlabel_matrix(sectors::AbstractVector{QT}) where {QT}
    qlabels = Matrix{QT}(undef, 2, length(sectors))
    for (sector_index, sector) in enumerate(sectors)
        qlabels[1, sector_index] = sector
        qlabels[2, sector_index] = sector
    end
    return qlabels
end

_retag_qindex(idx::TLIndex, tag::AbstractString) = TLIndex(tag, idx.dir, idx.plev, idx.lock, idx.dual)

function _retag_eigen_result(result::EigenResult, eig_tag::AbstractString)
    d_inds = (_retag_qindex(result.D.inds[1], eig_tag), _retag_qindex(result.D.inds[2], eig_tag))
    D = TLArray(result.D, d_inds)

    v_inds = (result.V.inds[1], _retag_qindex(result.V.inds[2], eig_tag))
    V = TLArray(result.V, v_inds)

    V_inv = if isnothing(result.V_inv)
        nothing
    else
        vinv_inds = (_retag_qindex(result.V_inv.inds[1], eig_tag), result.V_inv.inds[2])
        TLArray(result.V_inv, vinv_inds)
    end

    return EigenResult(V, D, V_inv, result.eig_list)
end

function _prepare_eigen_input(q::TLArray{T, 2, N, RD},
                              opname::AbstractString) where {T, N, RD}
    dirs = (q.inds[1].dir, q.inds[2].dir)
    @assert (dirs == ('+', '-') || dirs == ('-', '+')) "$opname requires one incoming ('+') and one outgoing ('-') leg"

    q_work = dirs == ('+', '-') ? q : copy(permutedims(q, (2, 1)))
    @assert q_work.spaces[1] == q_work.spaces[2] "$opname: both legs of input TLArray must have the same space list (same sectors and dimensions)"
    return q_work
end

function _check_hermitian_eigen_legs(q::TLArray,
                                     opname::AbstractString)
    idx1, idx2 = q.inds
    @assert idx1.itags == idx2.itags "$opname: Hermitian eigendecomposition requires both legs to have the same itag"
    @assert idx1.plev == idx2.plev "$opname: Hermitian eigendecomposition requires both legs to have the same plev"
    @assert idx1.lock == idx2.lock "$opname: Hermitian eigendecomposition requires both legs to have the same lock"
    @assert idx1.dual == idx2.dual "$opname: Hermitian eigendecomposition requires both legs to have the same dual flag"
    return nothing
end

function _hermiticity_ratio(q::TLArray{T, 2, N, RD}) where {T, N, RD}
    q_adj = permutedims(q', (2, 1))
    q_adj = TLArray(q_adj, q.inds)

    qnorm = norm(q)
    diffnorm = norm(q - q_adj)
    if iszero(qnorm)
        return iszero(diffnorm) ? 0.0 : Inf
    end
    return diffnorm / qnorm
end

function _select_eig_sectors(template::TLArray{T, 2, N, RD, QT, PS, M, RMT},
                          picks::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}};
                          mode::Symbol) where {T, N, RD, QT, PS, M, RMT}
    qlabels_out = QT[]
    source_wmat_sectors = Int[]
    RMTs_out = mode == :diag && RMT <: DiagRMT ? DiagRMT{T, RD}[] : Array{T, RD}[]
    sector_counts = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()

    for (sector, idxs) in picks
        unique_count = length(unique(sort(idxs)))
        unique_count == 0 && continue
        sector_counts[sector] = unique_count
    end

    for sector_index in sector_slots(template)
        template.iszero[sector_index] && continue
        sector = _eig_sector_qlabels(template, sector_index)
        idxs = get(picks, sector, Int[])
        isempty(idxs) && continue

        idxs_sorted = sort(idxs)
        rmt = sector_rmt(template, sector_index)
        s1, s2 = size(rmt, 1), size(rmt, 2)
        rmt_new = if mode == :diag
            if rmt isa DiagRMT
                DiagRMT(T[rmt.diag[i] for i in idxs_sorted], Val(RD), (1, 2))
            else
                mat = reshape(rmt, s1, s2)
                reshape(mat[idxs_sorted, idxs_sorted], length(idxs_sorted), length(idxs_sorted), ones(Int, N)...)
            end
        elseif mode == :cols
            mat = reshape(rmt, s1, s2)
            reshape(mat[:, idxs_sorted], s1, length(idxs_sorted), ones(Int, N)...)
        elseif mode == :firstdim
            mat = reshape(rmt, s1, s2)
            reshape(mat[idxs_sorted, :], length(idxs_sorted), s2, ones(Int, N)...)
        else
            error("Unknown eig sector selection mode: $mode")
        end

        push!(qlabels_out, sector)
        push!(RMTs_out, rmt_new)
        push!(source_wmat_sectors, sector_index)
    end

    function build_selected_space(base_space, dim_fn)
        selected = eltype(base_space)[]
        for (sector, dim) in base_space
            haskey(sector_counts, sector) || continue
            push!(selected, (sector, dim_fn(dim, sector_counts[sector])))
        end
        return selected
    end

    spaces_out = if mode == :diag
        (
            build_selected_space(template.spaces[1], (_, count) -> count),
            build_selected_space(template.spaces[2], (_, count) -> count),
        )
    elseif mode == :cols
        (
            copy(template.spaces[1]),
            build_selected_space(template.spaces[2], (_, count) -> count),
        )
    elseif mode == :firstdim
        (
            build_selected_space(template.spaces[1], (_, count) -> count),
            copy(template.spaces[2]),
        )
    else
        error("Unknown eig sector selection mode: $mode")
    end

    qlabels_mat = Matrix{QT}(undef, 2, length(qlabels_out))
    for (sector_index, sector) in enumerate(qlabels_out)
        qlabels_mat[1, sector_index] = sector
        qlabels_mat[2, sector_index] = sector
    end
    wmatdata_out, wmatinfo_out = _copy_wmat_storage(template, source_wmat_sectors; deep=true)
    return TLArray(symm(template), qlabels_mat, wmatdata_out, wmatinfo_out, RMTs_out, template.inds, spaces_out)
end

function _effective_eigen_keep_count(eig_entries,
                                     Nkeep::Integer,
                                     tol::Real;
                                     hermitian::Bool)
    total = length(eig_entries)
    nkeep_eff = min(Nkeep, total)
    if !(tol > 0) || nkeep_eff == 0 || nkeep_eff == total
        return nkeep_eff
    end

    candidate_end = min(total, ceil(Int, Nkeep * (1 + Float64(tol))))
    candidate_end <= nkeep_eff && return nkeep_eff
    candidate_end == total && return candidate_end

    sort_vals = [_eig_sort_value(eig_entries[i][1], hermitian) for i in nkeep_eff:candidate_end+1]
    _, gap_idx = findmax(diff(sort_vals))
    return nkeep_eff + gap_idx - 1
end

function _split_eigen_result(result::EigenResult,
                             Nkeep::Integer;
                             tol::Real=0.1,
                             hermitian::Bool = isnothing(result.V_inv))
    @assert Nkeep >= 0 "Nkeep must be non-negative"
    @assert isfinite(tol) && tol >= 0 "tol must be finite and nonnegative"

    eig_entries = copy(result.eig_list)
    sort!(eig_entries; by = x -> _eig_sort_value(x[1], hermitian))

    nkeep_eff = _effective_eigen_keep_count(eig_entries, Nkeep, tol; hermitian=hermitian)
    kept_entries = eig_entries[1:nkeep_eff]
    discarded_entries = eig_entries[nkeep_eff+1:end]

    N = length(symm(result.D))
    kept_picks = Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}}()
    for entry in kept_entries
        push!(get!(kept_picks, entry[3], Int[]), entry[4])
    end

    discarded_picks = Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}}()
    for entry in discarded_entries
        push!(get!(discarded_picks, entry[3], Int[]), entry[4])
    end

    Vkeep = _select_eig_sectors(result.V, kept_picks; mode=:cols)
    Vdiscard = _select_eig_sectors(result.V, discarded_picks; mode=:cols)
    Dkeep = _select_eig_sectors(result.D, kept_picks; mode=:diag)
    Ddiscard = _select_eig_sectors(result.D, discarded_picks; mode=:diag)

    Vinv_keep = isnothing(result.V_inv) ? nothing : _select_eig_sectors(result.V_inv, kept_picks; mode=:firstdim)
    Vinv_discard = isnothing(result.V_inv) ? nothing : _select_eig_sectors(result.V_inv, discarded_picks; mode=:firstdim)

    kept = EigenResult(Vkeep, Dkeep, Vinv_keep, _renumber_eig_entries(kept_entries))
    discarded = EigenResult(Vdiscard, Ddiscard, Vinv_discard, _renumber_eig_entries(discarded_entries))
    return kept, discarded
end

function _eigen_hermitian(q::TLArray{T, 2, N, RD, QT},
                          eig_tag::AbstractString = "eig") where {T, N, RD, QT}

    _check_hermitian_eigen_legs(q, "eigen")

    symmetries = symm(q)
    dirs = (q.inds[1].dir, q.inds[2].dir)
    out_leg = 2
    cgp = (1, 2)

    # ── Decompose each sector ───────────────────────────────────────────────────
    T_out    = promote_type(T, Float64)
    qlabels_D = QT[]
    qlabels_V = QT[]
    RMTs_D = DiagRMT{T_out, 2 + N}[]
    RMTs_V = Array{T_out, 2 + N}[]
    # Eigenvalue, degeneracy, sector qlabels, in-sector index
    eig_list = Tuple{T_out, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]

    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        sector_qlabels = _eig_sector_qlabels(q, sector_index)

        # Degeneracy = product of irrep dims across all symmetries for this sector
        cgt_dim = prod(
            dimension(symmetries[n], sector_qlabels[n])
            for n in 1:N)
        cgt_scale = sqrt(Float64(cgt_dim))

        rmt = sector_rmt(q, sector_index)
        sL, sR = size(rmt, 1), size(rmt, 2)
        @assert sL == sR "eigen: RMT must be square for eigendecomposition, got ($sL, $sR)"

        mat = reshape(rmt, sL, sR) ./ cgt_scale

        F = eigen(Hermitian(mat))
        eigenvalues  = T_out.(F.values)
        eigenvectors = T_out.(F.vectors)

        chi = length(eigenvalues)
        for (j, ev) in enumerate(eigenvalues)
            push!(eig_list, (ev, cgt_dim, sector_qlabels, j))
        end

        rmt_D = _eig_diagonal_rmt(RMTs_D, eigenvalues, cgt_scale, Val(2 + N))
        rmt_V = reshape(eigenvectors, sL, chi, ones(Int, N)...)
        rmt_V[:] .*= cgt_scale

        push!(qlabels_D, sector_qlabels)
        push!(qlabels_V, sector_qlabels)
        push!(RMTs_D, rmt_D)
        push!(RMTs_V, rmt_V)
    end

    covered = Set(_eig_sector_qlabels(q, sector_index) for sector_index in sector_slots(q) if !q.iszero[sector_index])
    _append_missing_eig_sectors!(symmetries, q.spaces[1], covered,
        qlabels_D, RMTs_D,
        qlabels_V, RMTs_V,
        nothing, nothing,
        eig_list, cgp, T_out)

    # Sort eig_list ascending by eigenvalue.
    sort!(eig_list; by = x -> _eig_sort_value(x[1], true))

    # ── Inherit spaces from original q ──────────────────────────────────────
    spaces_D = (q.spaces[1], q.spaces[1])
    spaces_V = (q.spaces[1], q.spaces[1])

    # ── Build output TLIndex tuples ───────────────────────────────────────────
    inds_D = (TLIndex(eig_tag, dirs[1]), TLIndex(eig_tag, dirs[2]))

    orig_out_ind = q.inds[out_leg]
    inds_V = (TLIndex(orig_out_ind.itags, dirs[1], orig_out_ind.plev, orig_out_ind.lock, orig_out_ind.dual),
              TLIndex(eig_tag, dirs[2]))

    wmatdata_D, wmatinfo_D = _unit_wmat_storage(productsymm(q), length(RMTs_D))
    wmatdata_V, wmatinfo_V = _unit_wmat_storage(productsymm(q), length(RMTs_V))
    D = TLArray(symmetries, _eig_qlabel_matrix(qlabels_D),
                wmatdata_D, wmatinfo_D, RMTs_D, inds_D, spaces_D)
    V = TLArray(symmetries, _eig_qlabel_matrix(qlabels_V),
                wmatdata_V, wmatinfo_V, RMTs_V, inds_V, spaces_V)

    return EigenResult(V, D, nothing, eig_list)
end

function _eigen_general(q::TLArray{T, 2, N, RD, QT},
                        eig_tag::AbstractString = "eig") where {T, N, RD, QT}

    symmetries = symm(q)
    dirs = (q.inds[1].dir, q.inds[2].dir)
    cgp = (1, 2)

    in_leg  = 1
    out_leg = 2

    T_out     = promote_type(T, ComplexF64)
    qlabels_D = QT[]
    qlabels_V = QT[]
    qlabels_Vinv = QT[]
    RMTs_D = DiagRMT{T_out, 2 + N}[]
    RMTs_V = Array{T_out, 2 + N}[]
    RMTs_Vinv = Array{T_out, 2 + N}[]
    eig_list  = Tuple{T_out, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]

    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        sector_qlabels = _eig_sector_qlabels(q, sector_index)

        cgt_dim = prod(
            dimension(symmetries[n], sector_qlabels[n])
            for n in 1:N)
        cgt_scale = sqrt(Float64(cgt_dim))

        rmt = sector_rmt(q, sector_index)
        sL, sR = size(rmt, 1), size(rmt, 2)
        @assert sL == sR "eigen_general: RMT must be square"

        mat = reshape(rmt, sL, sR) ./ cgt_scale

        F = eigen(mat)
        eigenvalues      = T_out.(F.values)
        eigenvectors     = T_out.(F.vectors)
        eigenvectors_inv = T_out.(inv(F.vectors))

        chi = length(eigenvalues)
        for (j, ev) in enumerate(eigenvalues)
            push!(eig_list, (ev, cgt_dim, sector_qlabels, j))
        end

        rmt_D    = _eig_diagonal_rmt(RMTs_D, eigenvalues, cgt_scale, Val(2 + N))
        rmt_V    = reshape(eigenvectors, sL, chi, ones(Int, N)...)
        rmt_Vinv = reshape(eigenvectors_inv, chi, sL, ones(Int, N)...)
        rmt_V[:] .*= cgt_scale
        rmt_Vinv[:] .*= cgt_scale

        push!(qlabels_D, sector_qlabels)
        push!(qlabels_V, sector_qlabels)
        push!(qlabels_Vinv, sector_qlabels)
        push!(RMTs_D, rmt_D)
        push!(RMTs_V, rmt_V)
        push!(RMTs_Vinv, rmt_Vinv)
    end

    covered = Set(_eig_sector_qlabels(q, sector_index) for sector_index in sector_slots(q) if !q.iszero[sector_index])
    _append_missing_eig_sectors!(symmetries, q.spaces[1], covered,
        qlabels_D, RMTs_D,
        qlabels_V, RMTs_V,
        qlabels_Vinv, RMTs_Vinv,
        eig_list, cgp, T_out)

    sort!(eig_list; by = x -> _eig_sort_value(x[1], false))

    spaces_D    = (q.spaces[1], q.spaces[1])
    spaces_V    = (q.spaces[1], q.spaces[1])
    spaces_Vinv = (q.spaces[1], q.spaces[1])

    inds_D = (TLIndex(eig_tag, dirs[1]), TLIndex(eig_tag, dirs[2]))

    orig_out_ind = q.inds[out_leg]
    inds_V = (TLIndex(orig_out_ind.itags, dirs[1], orig_out_ind.plev, orig_out_ind.lock, orig_out_ind.dual),
              TLIndex(eig_tag, dirs[2]))

    orig_in_ind = q.inds[in_leg]
    inds_Vinv = (TLIndex(eig_tag, dirs[1]),
                 TLIndex(orig_in_ind.itags, dirs[2], orig_in_ind.plev, orig_in_ind.lock, orig_in_ind.dual))

    wmatdata_D, wmatinfo_D = _unit_wmat_storage(productsymm(q), length(RMTs_D))
    wmatdata_V, wmatinfo_V = _unit_wmat_storage(productsymm(q), length(RMTs_V))
    wmatdata_Vinv, wmatinfo_Vinv = _unit_wmat_storage(productsymm(q), length(RMTs_Vinv))
    D    = TLArray(symmetries, _eig_qlabel_matrix(qlabels_D),
                   wmatdata_D, wmatinfo_D, RMTs_D, inds_D, spaces_D)
    V    = TLArray(symmetries, _eig_qlabel_matrix(qlabels_V),
                   wmatdata_V, wmatinfo_V, RMTs_V, inds_V, spaces_V)
    Vinv = TLArray(symmetries, _eig_qlabel_matrix(qlabels_Vinv),
                   wmatdata_Vinv, wmatinfo_Vinv, RMTs_Vinv, inds_Vinv, spaces_Vinv)

    return EigenResult(V, D, Vinv, eig_list)
end

function LinearAlgebra.eigen(q::TLArray{T, 2, N, RD},
                             eig_tag::AbstractString = "eig";
                             hermitian::Bool = false) where {T, N, RD}
    q_work = _prepare_eigen_input(q, "eigen")

    use_hermitian = hermitian
    if !use_hermitian
        use_hermitian = _hermiticity_ratio(q_work) <= _EIG_HERMITIAN_RTOL
    end

    return use_hermitian ? _eigen_hermitian(q_work, eig_tag) : _eigen_general(q_work, eig_tag)
end

LinearAlgebra.eigen(q::TLArrayView, args...; kwargs...) =
    eigen(_eager_tlarray(q), args...; kwargs...)

"""
    discard_eigen(result::EigenResult, Nkeep, tol, kept_tag, discarded_tag; hermitian=isnothing(result.V_inv))

Keep the `Nkeep` smallest eigenvalues, ignoring degeneracy, and return two
`EigenResult` objects: the kept part and the discarded part.

If `tol > 0`, keep at least `Nkeep` eigenvalues and consider up to
`ceil(Int, Nkeep * (1 + tol))` total next-smallest eigenvalues, capped by the
total number available. The actual cut is chosen from that window by the largest
adjacent gap, so the truncation can move later but never earlier than `Nkeep`.
"""
function discard_eigen(result::EigenResult,
                       Nkeep::Integer,
                       tol::Real,
                       kept_tag::AbstractString = "eigK",
                       discarded_tag::AbstractString = "eigD";
                       hermitian::Bool = isnothing(result.V_inv))
    kept, discarded = _split_eigen_result(result, Nkeep; tol=tol, hermitian=hermitian)
    return _retag_eigen_result(kept, kept_tag), _retag_eigen_result(discarded, discarded_tag)
end

function discard_eigen(result::EigenResult,
                       Nkeep::Integer,
                       kept_tag::AbstractString = "eigK",
                       discarded_tag::AbstractString = "eigD";
                       tol::Real=0.1,
                       hermitian::Bool = isnothing(result.V_inv))
    return discard_eigen(result, Nkeep, tol, kept_tag, discarded_tag; hermitian=hermitian)
end
