get1jtensor(q::TLArray{T, QD, N, RD}, leg::Int) where {T, QD, N, RD} =
get1jtensor(leginfo(q, leg))

function _resolve_unique_leg(q::TLArray; dir=nothing, itag=nothing, plev=nothing,
                             lock=nothing, rev::Bool=false,
                             opname::AbstractString="operation")
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    length(legs) == 1 && return only(legs)
    isempty(legs) && throw(ArgumentError("$opname requires a uniquely specified leg, but no legs matched"))
    throw(ArgumentError("$opname requires a uniquely specified leg, but matched legs $legs"))
end

function _resolve_matching_legs(q::TLArray; dir=nothing, itag=nothing, plev=nothing,
                                lock=nothing, rev::Bool=false,
                                opname::AbstractString="operation")
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    isempty(legs) && throw(ArgumentError("$opname requires at least one matching leg, but no legs matched"))
    return legs
end

function _normalize_legflip_legs(q::TLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("legflip requires at least one leg"))

    for leg in positions
        1 <= leg <= QD || throw(BoundsError(q, leg))
    end

    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "legflip legs must be unique, got $positions"))
    return positions
end

function get1jtensor(q::TLArray; dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false)
    leg = _resolve_unique_leg(q; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, opname="get1jtensor")
    return get1jtensor(q, leg)
end

function get1jtensor(info::leginfo{N, QT, PS}) where {N, QT, PS}
    symm, ind = product_symms(PS), info.ind
    inds1 = (change_dir(ind), change_dir(change_dual(ind)))
    dir1 = inds1[1].dir

    nr = length(info.splist)
    qlabels1 = Matrix{QT}(undef, 2, nr)
    wmatdata1, wmatinfo1 = _unit_wmat_storage(PS, nr)

    # leg 1 = original space, leg 2 = dual space
    ET = eltype(info.splist)
    dual_splist = sort!(ET[(Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N), RMTd)
                           for (qlabels, RMTd) in info.splist], by=x->x[1])
    spaces1 = (info.splist, dual_splist)
    RMTs1 = Vector{DiagRMT{Float64, 2 + N}}(undef, nr)

    for (sector_index, (qlabels, RMTd)) in enumerate(info.splist)
        dual_qlabels = Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N)
        qlabels1[1, sector_index] = qlabels
        qlabels1[2, sector_index] = dual_qlabels

        cgt_dim = 1.0
        for n in 1:N
            cgt_dim *= dimension(symm[n], qlabels[n])
        end
        scale = sqrt(cgt_dim)
        RMTs1[sector_index] = _diag_rmt_from_values(ones(Float64, RMTd), Val(2 + N), (1, 2), scale)
    end

    q1 = TLArray(symm, qlabels1, wmatdata1, wmatinfo1, RMTs1, inds1, spaces1)
    return q1
end

function legflip(q::TLArray{T, QD, N, RD}, leg::Int) where {T, QD, N, RD}
    1 <= leg <= QD || throw(BoundsError(q, leg))
    j = get1jtensor(q, leg)
    q_flip = contract(q, (leg,), j, (1,); reduce_lock=false)
    perm = (ntuple(i -> i, leg - 1)..., QD, ntuple(i -> leg - 1 + i, QD - leg)...)
    return permutedims(q_flip, perm)
end

function legflip(q::TLArray{T, QD, N, RD}, legs::LegList) where {T, QD, N, RD}
    positions = _normalize_legflip_legs(q, legs)
    q_flip = q
    for leg in positions
        q_flip = copy(legflip(q_flip, leg))
    end
    return q_flip
end

function legflip(q::TLArray; dir=nothing, itag=nothing, plev=nothing,
                 lock=nothing, rev::Bool=false)
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="legflip")
    return legflip(q, legs)
end

function legflip(q::TLArrayView, leg::Int)
    return _view_rewrap(q, legflip(q.arr, _view_arr_leg(q, leg)))
end

function legflip(q::TLArrayView, legs::LegList)
    q_flip = q
    for leg in _normalize_legflip_legs(q, legs)
        q_flip = copy(legflip(q_flip, leg))
    end
    return q_flip
end

function legflip(q::TLArrayView; dir=nothing, itag=nothing, plev=nothing,
                 lock=nothing, rev::Bool=false)
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="legflip")
    return legflip(q, legs)
end
