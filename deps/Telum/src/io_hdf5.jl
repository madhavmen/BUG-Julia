const _TLARRAY_HDF5_SCHEMA_VERSION = "1.0"

_h5_attrs(obj) = HDF5.attrs(obj)

function _h5_set_attr!(obj, name::AbstractString, value)
    _h5_attrs(obj)[name] = value
    return obj
end

_h5_attr(obj, name::AbstractString) = _h5_attrs(obj)[name]
_h5_attr_string(obj, name::AbstractString) = String(_h5_attr(obj, name))

function _package_version_string(mod::Module)
    try
        v = Base.pkgversion(mod)
        return v === nothing ? "unknown" : string(v)
    catch
        return "unknown"
    end
end

_eltype_tag(::Type{Float64}) = "Float64"
_eltype_tag(::Type{Float32}) = "Float32"
_eltype_tag(::Type{ComplexF64}) = "ComplexF64"
_eltype_tag(::Type{ComplexF32}) = "ComplexF32"
_eltype_tag(::Type{T}) where {T} =
    throw(ArgumentError("unsupported TLArray HDF5 RMT element type $T"))

function _eltype_from_tag(tag::AbstractString)
    tag == "Float64" && return Float64
    tag == "Float32" && return Float32
    tag == "ComplexF64" && return ComplexF64
    tag == "ComplexF32" && return ComplexF32
    throw(ArgumentError("unsupported TLArray HDF5 RMT element type $tag"))
end

function _symmetry_family_parameter(::Type{S}) where {S<:Symmetry}
    S === U1 && return ("U1", 0)
    S === LurCGT.G2 && return ("G2", 0)
    S <: Z && return ("Z", Int(S.parameters[1]))
    S <: SU && return ("SU", Int(S.parameters[1]))
    S <: SO && return ("SO", Int(S.parameters[1]))
    S <: Sp && return ("Sp", Int(S.parameters[1]))
    throw(ArgumentError("unsupported symmetry type $S for TLArray HDF5 I/O"))
end

function _symmetry_from_family_parameter(family::AbstractString, parameter::Integer)
    family == "U1" && return U1
    family == "G2" && return LurCGT.G2
    family == "Z" && return Z{Int(parameter)}
    family == "SU" && return SU{Int(parameter)}
    family == "SO" && return SO{Int(parameter)}
    family == "Sp" && return Sp{Int(parameter)}
    throw(ArgumentError("unsupported symmetry family $family in TLArray HDF5 file"))
end

function _write_symmetries!(parent, symm::NTuple{N, Any}) where {N}
    g = HDF5.create_group(parent, "symmetries")
    g["count"] = N
    for n in 1:N
        sg = HDF5.create_group(g, string(n))
        family, parameter = _symmetry_family_parameter(symm[n])
        _h5_set_attr!(sg, "family", family)
        _h5_set_attr!(sg, "parameter", parameter)
        _h5_set_attr!(sg, "abelian", isabelian(symm[n]))
    end
    return g
end

function _read_symmetries(parent)
    g = parent["symmetries"]
    N = Int(read(g["count"]))
    return ntuple(Val(N)) do n
        sg = g[string(n)]
        family = _h5_attr_string(sg, "family")
        parameter = Int(_h5_attr(sg, "parameter"))
        _symmetry_from_family_parameter(family, parameter)
    end
end

@inline _qlabel_width(symm) = sum(n -> nzops(symm[n]), eachindex(symm); init=0)

function _encode_oneleg_qlabels(qlabels::AbstractVector, symm)
    width = _qlabel_width(symm)
    data = Matrix{Int}(undef, width, length(qlabels))
    for sector in eachindex(qlabels)
        row = 1
        q = qlabels[sector]
        for n in eachindex(symm)
            qn = q[n]
            for k in 1:nzops(symm[n])
                data[row, sector] = Int(qn[k])
                row += 1
            end
        end
    end
    return data
end

function _decode_oneleg_qlabels(data::AbstractMatrix{<:Integer}, symm)
    width = _qlabel_width(symm)
    size(data, 1) == width ||
        throw(ArgumentError("qlabel matrix row count $(size(data, 1)) does not match expected width $width"))
    QT = qlabeltype(symm)
    result = Vector{QT}(undef, size(data, 2))
    for sector in axes(data, 2)
        row = 1
        q = ntuple(Val(length(symm))) do n
            qn = ntuple(k -> Int(data[row + k - 1, sector]), Val(nzops(symm[n])))
            row += nzops(symm[n])
            qn
        end
        result[sector] = q
    end
    return result
end

function _encode_sector_qlabels(qlabels::AbstractVector{<:NTuple{QD}}, symm) where {QD}
    width = _qlabel_width(symm)
    data = Matrix{Int}(undef, QD * width, length(qlabels))
    for sector in eachindex(qlabels)
        for leg in 1:QD
            row0 = (leg - 1) * width
            row = row0 + 1
            q = qlabels[sector][leg]
            for n in eachindex(symm)
                qn = q[n]
                for k in 1:nzops(symm[n])
                    data[row, sector] = Int(qn[k])
                    row += 1
                end
            end
        end
    end
    return data
end

function _decode_sector_qlabels(data::AbstractMatrix{<:Integer}, symm, ::Val{QD}) where {QD}
    width = _qlabel_width(symm)
    size(data, 1) == QD * width ||
        throw(ArgumentError("sector qlabel matrix row count $(size(data, 1)) does not match expected $(QD * width)"))
    QT = qlabeltype(symm)
    result = Vector{NTuple{QD, QT}}(undef, size(data, 2))
    for sector in axes(data, 2)
        result[sector] = ntuple(Val(QD)) do leg
            row = (leg - 1) * width + 1
            q = ntuple(Val(length(symm))) do n
                qn = ntuple(k -> Int(data[row + k - 1, sector]), Val(nzops(symm[n])))
                row += nzops(symm[n])
                qn
            end
            q
        end
    end
    return result
end

function _write_indices!(parent, inds::NTuple{QD, TLIndex}) where {QD}
    g = HDF5.create_group(parent, "indices")
    g["itags"] = String[String(idx.itags) for idx in inds]
    g["dir"] = UInt8[UInt8(idx.dir) for idx in inds]
    g["plev"] = Int[idx.plev for idx in inds]
    g["lock"] = Int[idx.lock for idx in inds]
    g["dual"] = Bool[idx.dual for idx in inds]
    return g
end

function _read_indices(parent, ::Val{QD}) where {QD}
    g = parent["indices"]
    itags = Vector{String}(read(g["itags"]))
    dirs = Vector{UInt8}(read(g["dir"]))
    plev = Vector{Int}(read(g["plev"]))
    lock = Vector{Int}(read(g["lock"]))
    duals = Vector{Bool}(read(g["dual"]))
    length(itags) == QD && length(dirs) == QD && length(plev) == QD &&
        length(lock) == QD && length(duals) == QD ||
        throw(ArgumentError("index metadata length does not match tensor rank $QD"))
    return ntuple(i -> TLIndex(itags[i], Char(dirs[i]), plev[i], lock[i], duals[i]), Val(QD))
end

function _write_spaces!(parent, spaces::NTuple{QD, <:AbstractVector}, symm) where {QD}
    g = HDF5.create_group(parent, "spaces")
    for leg in 1:QD
        lg = HDF5.create_group(g, string(leg))
        entries = spaces[leg]
        lg["qlabels"] = _encode_oneleg_qlabels([entry[1] for entry in entries], symm)
        lg["dims"] = Int[entry[2] for entry in entries]
    end
    return g
end

function _read_spaces(parent, symm, ::Val{QD}) where {QD}
    g = parent["spaces"]
    QT = qlabeltype(symm)
    return ntuple(Val(QD)) do leg
        lg = g[string(leg)]
        qlabels = _decode_oneleg_qlabels(read(lg["qlabels"]), symm)
        dims = Vector{Int}(read(lg["dims"]))
        length(qlabels) == length(dims) ||
            throw(ArgumentError("space qlabel/dimension length mismatch on leg $leg"))
        Tuple{QT, Int}[(qlabels[i], dims[i]) for i in eachindex(dims)]
    end
end

function _encode_wmatinfo(wmatinfo::Vector{WMatInfo{M}}) where {M}
    data = Vector{Int}(undef, 3 * M * length(wmatinfo))
    p = 1
    for sector in eachindex(wmatinfo), slot in 1:M, item in 1:3
        data[p] = wmatinfo[sector][slot][item]
        p += 1
    end
    return data
end

function _decode_wmatinfo(data::AbstractVector{<:Integer}, sector_count::Int, ::Val{M}) where {M}
    expected = 3 * M * sector_count
    length(data) == expected ||
        throw(ArgumentError("wmat info length $(length(data)) does not match expected $expected"))
    wmatinfo = Vector{WMatInfo{M}}(undef, sector_count)
    p = 1
    for sector in 1:sector_count
        wmatinfo[sector] = ntuple(Val(M)) do _
            info = (Int(data[p]), Int(data[p + 1]), Int(data[p + 2]))
            p += 3
            info
        end
    end
    return wmatinfo
end

function _write_wmat_storage!(parent, q::TLArray{T, QD, N, RD, QT, PS, M}) where {T, QD, N, RD, QT, PS, M}
    g = HDF5.create_group(parent, "wmat")
    g["data"] = q.wmatdata
    g["info"] = _encode_wmatinfo(q.wmatinfo)
    _h5_set_attr!(g, "tuple_width", M)
    _h5_set_attr!(g, "sector_count", length(q.qlabels))
    return g
end

function _read_wmat_storage(parent, sector_count::Int, ::Val{M}) where {M}
    g = parent["wmat"]
    Int(_h5_attr(g, "tuple_width")) == M ||
        throw(ArgumentError("wmat tuple width does not match symmetry metadata"))
    Int(_h5_attr(g, "sector_count")) == sector_count ||
        throw(ArgumentError("wmat sector count does not match sector metadata"))
    return Vector{Float64}(read(g["data"])), _decode_wmatinfo(read(g["info"]), sector_count, Val(M))
end

function _validate_eager_tlarray_for_hdf5(q::TLArray)
    sector_count = length(q.qlabels)
    length(q.RMTs) == sector_count && length(q.isdefined) == sector_count &&
        length(q.iszero) == sector_count ||
        throw(ArgumentError("inconsistent TLArray sector storage lengths"))
    rmt_type = nothing
    for sector in sector_slots(q)
        if q.isdefined[sector]
            isassigned(q.RMTs, sector) ||
                throw(ArgumentError("sector $sector is marked defined but has no assigned RMT"))
            q.iszero[sector] &&
                throw(ArgumentError("sector $sector is both defined and zero"))
            current = typeof(q.RMTs[sector])
            rmt_type === nothing || current === rmt_type ||
                throw(ArgumentError("all defined RMTs in one TLArray must have the same concrete type"))
            rmt_type = current
        else
            q.iszero[sector] ||
                throw(ArgumentError("undefined concrete TLArray sector $sector must be marked zero"))
        end
    end
    return rmt_type
end

@inline _rmt_storage_tag(::Type{<:Array}) = "dense_arena"
@inline _rmt_storage_tag(::Type{<:DiagRMT}) = "diag_arena"
@inline _rmt_storage_tag(::Type{RMT}) where {RMT} =
    throw(ArgumentError("unsupported RMT storage type $RMT for TLArray HDF5 I/O"))

function _write_dense_rmt_arena!(g, q::TLArray{T, QD, N, RD}) where {T, QD, N, RD}
    data = T[]
    info = zeros(Int, 2 + RD, length(q.qlabels))
    for sector in sector_slots(q)
        q.iszero[sector] && continue
        rmt = q.RMTs[sector]
        offset = length(data) + 1
        append!(data, vec(rmt))
        len = length(rmt)
        info[1, sector] = offset
        info[2, sector] = len
        for axis in 1:RD
            info[2 + axis, sector] = size(rmt, axis)
        end
    end
    g["data"] = data
    g["info"] = info
    return g
end

function _write_diag_rmt_arena!(g, q::TLArray{T}) where {T}
    data = T[]
    info = zeros(Int, 4, length(q.qlabels))
    for sector in sector_slots(q)
        q.iszero[sector] && continue
        rmt = q.RMTs[sector]
        offset = length(data) + 1
        append!(data, rmt.diag)
        info[1, sector] = offset
        info[2, sector] = length(rmt.diag)
        info[3, sector] = rmt.axis[1]
        info[4, sector] = rmt.axis[2]
    end
    g["data"] = data
    g["info"] = info
    return g
end

function _write_rmt_storage!(parent, q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    g = HDF5.create_group(parent, "rmt")
    storage = _rmt_storage_tag(RMT)
    _h5_set_attr!(g, "storage", storage)
    _h5_set_attr!(g, "alignment", 1)
    _h5_set_attr!(g, "sector_count", length(q.qlabels))
    if storage == "dense_arena"
        _write_dense_rmt_arena!(g, q)
    else
        _write_diag_rmt_arena!(g, q)
    end
    return g
end

function _check_rmt_range(data_len::Int, offset::Int, len::Int, sector::Int)
    offset > 0 || throw(ArgumentError("defined sector $sector has non-positive RMT arena offset"))
    len > 0 || throw(ArgumentError("defined sector $sector has non-positive RMT arena length"))
    offset + len - 1 <= data_len ||
        throw(ArgumentError("RMT arena range for sector $sector exceeds data length"))
    return nothing
end

function _read_dense_rmts(parent, ::Type{T}, ::Val{RD}, isdefined_bits::BitVector, iszero_bits::BitVector) where {T, RD}
    g = parent["rmt"]
    data = Vector{T}(read(g["data"]))
    info = Matrix{Int}(read(g["info"]))
    sector_count = length(iszero_bits)
    size(info) == (2 + RD, sector_count) ||
        throw(ArgumentError("dense RMT info shape $(size(info)) does not match expected $((2 + RD, sector_count))"))
    RMTs = Vector{Array{T, RD}}(undef, sector_count)
    for sector in 1:sector_count
        offset, len = info[1, sector], info[2, sector]
        if isdefined_bits[sector] && !iszero_bits[sector]
            _check_rmt_range(length(data), offset, len, sector)
            shape = ntuple(axis -> info[2 + axis, sector], Val(RD))
            prod(shape) == len ||
                throw(ArgumentError("dense RMT shape product does not match arena length for sector $sector"))
            RMTs[sector] = Array(reshape(copy(view(data, offset:offset + len - 1)), shape))
        else
            offset == 0 && len == 0 ||
                throw(ArgumentError("undefined zero sector $sector must have zero RMT arena range"))
        end
    end
    return RMTs
end

function _read_diag_rmts(parent, ::Type{T}, ::Val{RD}, isdefined_bits::BitVector, iszero_bits::BitVector) where {T, RD}
    g = parent["rmt"]
    data = Vector{T}(read(g["data"]))
    info = Matrix{Int}(read(g["info"]))
    sector_count = length(iszero_bits)
    size(info) == (4, sector_count) ||
        throw(ArgumentError("diagonal RMT info shape $(size(info)) does not match expected $((4, sector_count))"))
    RMTs = Vector{DiagRMT{T, RD}}(undef, sector_count)
    for sector in 1:sector_count
        offset, len = info[1, sector], info[2, sector]
        if isdefined_bits[sector] && !iszero_bits[sector]
            _check_rmt_range(length(data), offset, len, sector)
            axis = (info[3, sector], info[4, sector])
            RMTs[sector] = DiagRMT(Vector{T}(view(data, offset:offset + len - 1)), Val(RD), axis)
        else
            all(iszero, view(info, :, sector)) ||
                throw(ArgumentError("undefined zero sector $sector must have zero diagonal RMT metadata"))
        end
    end
    return RMTs
end

function _read_rmt_storage(parent, ::Type{T}, ::Val{RD}, isdefined_bits::BitVector, iszero_bits::BitVector) where {T, RD}
    g = parent["rmt"]
    storage = _h5_attr_string(g, "storage")
    Int(_h5_attr(g, "alignment")) == 1 ||
        throw(ArgumentError("unsupported RMT arena alignment in TLArray HDF5 file"))
    Int(_h5_attr(g, "sector_count")) == length(iszero_bits) ||
        throw(ArgumentError("RMT sector count does not match sector metadata"))
    storage == "dense_arena" && return _read_dense_rmts(parent, T, Val(RD), isdefined_bits, iszero_bits)
    storage == "diag_arena" && return _read_diag_rmts(parent, T, Val(RD), isdefined_bits, iszero_bits)
    throw(ArgumentError("unsupported RMT storage $storage in TLArray HDF5 file"))
end

function _write_sector_metadata!(parent, q::TLArray)
    g = HDF5.create_group(parent, "sectors")
    g["qlabels"] = _encode_sector_qlabels(q.qlabels, symm(q))
    g["isdefined"] = collect(Bool, q.isdefined)
    g["iszero"] = collect(Bool, q.iszero)
    return g
end

function _read_sector_metadata(parent, symmetries, ::Val{QD}) where {QD}
    g = parent["sectors"]
    qlabels = _decode_sector_qlabels(read(g["qlabels"]), symmetries, Val(QD))
    isdefined_bits = BitVector(Vector{Bool}(read(g["isdefined"])))
    iszero_bits = BitVector(Vector{Bool}(read(g["iszero"])))
    sector_count = length(qlabels)
    length(isdefined_bits) == sector_count && length(iszero_bits) == sector_count ||
        throw(ArgumentError("sector qlabel/state lengths do not match"))
    for sector in 1:sector_count
        (!isdefined_bits[sector] && !iszero_bits[sector]) &&
            throw(ArgumentError("undefined sector $sector is not marked zero"))
        (isdefined_bits[sector] && iszero_bits[sector]) &&
            throw(ArgumentError("sector $sector is both defined and zero"))
    end
    return qlabels, isdefined_bits, iszero_bits
end

function _write_tlarray_group!(h5, name::AbstractString, q::TLArray{T, QD, N, RD, QT, PS, M, RMT}) where {T, QD, N, RD, QT, PS, M, RMT}
    _validate_eager_tlarray_for_hdf5(q)
    haskey(h5, name) && throw(ArgumentError("HDF5 object $name already exists"))

    _h5_set_attr!(h5, "telum_hdf5_schema_version", _TLARRAY_HDF5_SCHEMA_VERSION)
    _h5_set_attr!(h5, "writer_package", "Telum")
    _h5_set_attr!(h5, "writer_telum_version", _package_version_string(@__MODULE__))
    _h5_set_attr!(h5, "writer_lurcgt_version", _package_version_string(LurCGT))
    _h5_set_attr!(h5, "writer_julia_version", string(VERSION))

    g = HDF5.create_group(h5, name)
    _h5_set_attr!(g, "object_type", "TLArray")
    _h5_set_attr!(g, "schema_version", _TLARRAY_HDF5_SCHEMA_VERSION)
    _h5_set_attr!(g, "eltype", _eltype_tag(T))
    _h5_set_attr!(g, "rmt_storage", _rmt_storage_tag(RMT))
    _h5_set_attr!(g, "qd", QD)
    _h5_set_attr!(g, "nsymms", N)
    _h5_set_attr!(g, "rd", RD)
    _h5_set_attr!(g, "sector_count", length(q.qlabels))
    _h5_set_attr!(g, "nonzero_sector_count", nsectors(q))

    _write_symmetries!(g, symm(q))
    _write_indices!(g, q.inds)
    _write_spaces!(g, q.spaces, symm(q))
    _write_sector_metadata!(g, q)
    _write_wmat_storage!(g, q)
    _write_rmt_storage!(g, q)
    return q
end

function save_tlarray(path::AbstractString, q::AbstractTLArray; name::AbstractString="tensor")
    HDF5.h5open(path, "w") do h5
        save_tlarray(h5, name, q)
    end
    return path
end

save_tlarray(h5, name::AbstractString, q::TLArray) =
    _write_tlarray_group!(h5, name, q)

_hdf5_concrete_tlarray(q::TLArray) = q
_hdf5_concrete_tlarray(q::TLArrayView) = _eager_tlarray(q)
_hdf5_concrete_tlarray(q::AbstractTLArray) = _eager_tlarray(materialize(q))

function save_tlarray(h5, name::AbstractString, q::AbstractTLArray)
    concrete = _hdf5_concrete_tlarray(q)
    return save_tlarray(h5, name, concrete)
end

function _read_tlarray_group(parent)
    _h5_attr_string(parent, "object_type") == "TLArray" ||
        throw(ArgumentError("HDF5 group does not contain a TLArray"))
    schema_version = _h5_attr_string(parent, "schema_version")
    schema_version == _TLARRAY_HDF5_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported TLArray HDF5 schema version $schema_version"))

    QD = Int(_h5_attr(parent, "qd"))
    N = Int(_h5_attr(parent, "nsymms"))
    RD = Int(_h5_attr(parent, "rd"))
    RD == QD + N ||
        throw(ArgumentError("stored RMT rank $RD does not equal QD + N = $(QD + N)"))

    symmetries = _read_symmetries(parent)
    length(symmetries) == N ||
        throw(ArgumentError("symmetry count does not match tensor metadata"))
    M = n_nonabelian_symmetries(productsymm(symmetries))
    T = _eltype_from_tag(_h5_attr_string(parent, "eltype"))

    qlabels, isdefined_bits, iszero_bits = _read_sector_metadata(parent, symmetries, Val(QD))
    sector_count = length(qlabels)
    Int(_h5_attr(parent, "sector_count")) == sector_count ||
        throw(ArgumentError("stored sector_count does not match qlabel metadata"))
    Int(_h5_attr(parent, "nonzero_sector_count")) == count(!, iszero_bits) ||
        throw(ArgumentError("stored nonzero sector count does not match sector state"))

    inds = _read_indices(parent, Val(QD))
    spaces = _read_spaces(parent, symmetries, Val(QD))
    wmatdata, wmatinfo = _read_wmat_storage(parent, sector_count, Val(M))
    RMTs = _read_rmt_storage(parent, T, Val(RD), isdefined_bits, iszero_bits)

    q = TLArray(symmetries, qlabels, wmatdata, wmatinfo, RMTs, inds, spaces)
    q.isdefined == isdefined_bits ||
        throw(ArgumentError("loaded TLArray isdefined bits do not match stored bits"))
    q.iszero == iszero_bits ||
        throw(ArgumentError("loaded TLArray iszero bits do not match stored bits"))
    return q
end

function load_tlarray(path::AbstractString; name::AbstractString="tensor")
    HDF5.h5open(path, "r") do h5
        return load_tlarray(h5, name)
    end
end

load_tlarray(h5, name::AbstractString) = _read_tlarray_group(h5[name])
