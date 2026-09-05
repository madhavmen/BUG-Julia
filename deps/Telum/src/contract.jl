# ─── RMT contraction helpers ─────────────────────────────────────────────────
@inline _identity_perm(::Val{D}) where {D} = ntuple(identity, Val(D))

"""
    TELUM_CONTRACT_THREADS[]

Upper bound on the tasks `contract` spawns across output sectors. Default `typemax(Int)`,
i.e. bounded only by `Threads.nthreads()` and the number of sectors; set to `1` to force the
serial path.

⚠ EXISTS SO THE SPEEDUP CAN BE MEASURED AGAINST ITSELF. A threading change that is only ever
run threaded cannot be told apart from an unrelated regression, and the serial path is also
the reference the bit-identity test compares against. It is a `Ref`, not a keyword, because
the callers that matter are deep inside sweep code that would otherwise have to thread the
option through every layer.
"""
const TELUM_CONTRACT_THREADS = Ref(typemax(Int))

function _hptt_permutedims(A::Array{Float64, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{Float64}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:dTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float64, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Float64, Ptr{Float64}, Ptr{Int32}, Cint, Cint),
          perm0, D, 1.0, A, sizeA, C_NULL,
          0.0, out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims(A::Array{Float32, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{Float32}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:sTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float32, Ptr{Float32}, Ptr{Int32}, Ptr{Int32},
           Float32, Ptr{Float32}, Ptr{Int32}, Cint, Cint),
          perm0, D, Float32(1), A, sizeA, C_NULL,
          Float32(0), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims(A::Array{ComplexF64, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{ComplexF64}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:zTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF64, Cuchar, Ptr{ComplexF64}, Ptr{Int32}, Ptr{Int32},
           ComplexF64, Ptr{ComplexF64}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF64), false, A, sizeA, C_NULL,
          zero(ComplexF64), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims(A::Array{ComplexF32, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{ComplexF32}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:cTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF32, Cuchar, Ptr{ComplexF32}, Ptr{Int32}, Ptr{Int32},
           ComplexF32, Ptr{ComplexF32}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF32), false, A, sizeA, C_NULL,
          zero(ComplexF32), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

@inline function _hptt_permutedims(A::AbstractArray{T, D},
                                   perm::NTuple{D, Int}) where {T, D}
    return Tuple(perm) == _identity_perm(Val(D)) ? A : permutedims(A, perm)
end

# ─── CGTContrInfo ────────────────────────────────────────────────────────────
# Holds the CGT space/leg data needed to call getNsave_Xsymbol for one
# non-abelian symmetry in a matched (r1, r2) sector pair.
struct CGTContrInfo{S<:NonabelianSymm, U1, D1, U2, D2, NZ, M}
    up1sp::NTuple{U1, NTuple{NZ, Int}}   # incoming qlabels of CGT1
    dn1sp::NTuple{D1, NTuple{NZ, Int}}   # outgoing qlabels of CGT1
    up2sp::NTuple{U2, NTuple{NZ, Int}}   # incoming qlabels of CGT2
    dn2sp::NTuple{D2, NTuple{NZ, Int}}   # outgoing qlabels of CGT2
    ctlegs1::NTuple{M, Int}              # contracted stored-qlabel indices in CGT1
    ctlegs2::NTuple{M, Int}              # contracted stored-qlabel indices in CGT2
end

# ─── Output CGT metadata ─────────────────────────────────────────────────────

# Core plain-data function: given two CGTs' qlabel/legdir/cgp data and the
# physical free/contracted legs, compute the resulting CGT's qlabels, cgp,
# and legdir.  This function is independent of the sector/CGT structs.
#
# Arguments (one symmetry at a time):
#   qlabels1/2  : NTuple{QD, NTuple{NZ,Int}} — stored qlabels of each CGT
#   legdir1/2   : (m, k) — number of incoming / outgoing stored qlabels
#   cgp1/2      : NTuple{QD, Int} — physical leg → stored qlabel index
#   free1/2     : physical free leg indices (1-based in each source TLArray)
#   legs1/2     : physical contracted leg indices
#
# Returns (new_qlabels, new_cgp, new_legdir).
function get_new_cgp(qlabels1::NTuple{QD1, NTuple{NZ, Int}}, legdir1,
                     cgp1::NTuple{QD1, Int},
                     free1::NTuple{NF1, Int}, legs1::NTuple{CN, Int},
                     qlabels2::NTuple{QD2, NTuple{NZ, Int}}, legdir2,
                     cgp2::NTuple{QD2, Int},
                     free2::NTuple{NF2, Int}, legs2::NTuple{CN, Int}) where {QD1, QD2, NZ, NF1, NF2, CN}
    m1, _ = legdir1;  m2, _ = legdir2
    QD_out = NF1 + NF2
    ctset1 = ntuple(k -> cgp1[legs1[k]], Val(CN))
    ctset2 = ntuple(k -> cgp2[legs2[k]], Val(CN))

    # (qlabel, output_physical_leg_index) pairs, insertion order = CGT1 then CGT2
    up3 = Vector{Tuple{NTuple{NZ, Int}, Int}}()
    dn3 = Vector{Tuple{NTuple{NZ, Int}, Int}}()
    sizehint!(up3, QD_out)
    sizehint!(dn3, QD_out)

    for (i, l_in) in enumerate(free1)
        sp = cgp1[l_in]
        sp ∈ ctset1 && continue
        if sp <= m1
            push!(up3, (qlabels1[sp], i))
        else
            push!(dn3, (qlabels1[sp], i))
        end
    end
    for (i, l_in) in enumerate(free2)
        sp = cgp2[l_in]
        sp ∈ ctset2 && continue
        if sp <= m2
            push!(up3, (qlabels2[sp], i + NF1))
        else
            push!(dn3, (qlabels2[sp], i + NF1))
        end
    end

    # Stable sort: equal qlabels keep CGT1-before-CGT2 insertion order.
    sort!(up3; by = x -> x[1], alg = MergeSort)
    sort!(dn3; by = x -> x[1], alg = MergeSort)

    m3, k3 = length(up3), length(dn3)
    new_qlabels = ntuple(Val(QD_out)) do i
        i <= m3 ? up3[i][1] : dn3[i - m3][1]
    end
    new_legdir  = (m3, k3)

    cgp3 = zeros(Int, QD_out)
    for (si, (_, l_out)) in enumerate(up3);  cgp3[l_out] = si      end
    for (si, (_, l_out)) in enumerate(dn3);  cgp3[l_out] = m3 + si end

    return (new_qlabels, ntuple(i -> cgp3[i], Val(QD_out)), new_legdir)
end

function get_new_cgp(qlabels1, legdir1, cgp1, free1, legs1,
                     qlabels2, legdir2, cgp2, free2, legs2)
    return get_new_cgp(qlabels1, legdir1, cgp1, Tuple(free1), Tuple(legs1),
                       qlabels2, legdir2, cgp2, Tuple(free2), Tuple(legs2))
end

# ─── merge_new_sector helpers ───────────────────────────────────────────────────

# Mode-product: replace axis `axis` of array A (size k) with the output of
# multiplying matrix M (shape r×k) along that axis.
# Equivalent to: result[..., j, ...] = Σ_l M[j,l] * A[...,l,...] at `axis`.
function _contract_om_axis_data(A::AbstractArray{T1, D}, M::AbstractMatrix{T2}, axis::Int) where {T1, T2<:Real, D}
    dims  = size(A)
    k     = dims[axis]
    r     = size(M, 1)
    @assert size(M, 2) == k "axis size $(k) != M columns $(size(M, 2))"

    findim = Base.setindex(dims, r, axis)
    if r == 1 && k == 1
        return reshape(A .* M[1, 1], findim)
    end

    prod_before = 1
    for i in 1:axis-1
        prod_before *= dims[i]
    end
    prod_after = 1
    for i in axis+1:D
        prod_after *= dims[i]
    end

    # Reshape A to (prod_before, k, prod_after), move k to the back,
    # multiply by M', then restore.  If prod_after == 1, k is already last.
    A_3  = reshape(A, prod_before, k, prod_after)
    if prod_after == 1
        A_mk = reshape(A_3, prod_before, k)
        R_mr = A_mk * transpose(M)
        return reshape(R_mr, findim)
    end

    A_mk = reshape(permutedims(A_3, (1, 3, 2)), prod_before * prod_after, k)
    R_mr = A_mk * transpose(M)
    R_3  = permutedims(reshape(R_mr, prod_before, prod_after, r), (1, 3, 2))
    return reshape(R_3, findim)
end

function _contract_om_axis(A::AbstractArray{T1, D}, M::AbstractMatrix{T2}, axis::Int) where {T1, T2<:Real, D}
    return _contract_om_axis_data(A, M, axis)
end

function _rmt_layout_sizes(rmt::AbstractArray{T, RD},
                           perm::NTuple{RD, Int},
                           nf::Int,
                           cn::Int,
                           nsyms::Int) where {T, RD}
    return _rmt_layout_sizes(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _rmt_layout_sizes(rmt::AbstractArray{T, RD},
                           perm::NTuple{RD, Int},
                           ::Val{NF},
                           ::Val{CN},
                           ::Val{N}) where {T, RD, NF, CN, N}
    dims = size(rmt)
    kept_sizes = ntuple(i -> dims[perm[i]], Val(NF))
    contracted_sizes = ntuple(i -> dims[perm[NF + i]], Val(CN))
    om_sizes = ntuple(i -> dims[perm[NF + CN + i]], Val(N))
    return kept_sizes, contracted_sizes, om_sizes
end

@inline _rmt_array_data(rmt::Array{T, RD}) where {T, RD} = rmt

@inline _permuted_rmt_type(::Type{<:Array{T,RD}}) where {T,RD} = Array{T,3}
@inline _permuted_rmt_type(::Type{<:DiagRMT{T,RD}}) where {T,RD} = DiagRMT{T,3}

@inline function _prepared_axis_group(axis::Int,
                                      perm::NTuple{RD, Int},
                                      ::Val{NF},
                                      ::Val{CN}) where {RD, NF, CN}
    @inbounds for pos in 1:RD
        if perm[pos] == axis
            pos <= NF && return 1
            pos <= NF + CN && return 2
            return 3
        end
    end
    throw(BoundsError(perm, axis))
end

function _permuted_rmt_data(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            nf::Int,
                            cn::Int,
                            nsyms::Int) where {T, RD}
    return _permuted_rmt_data(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _permuted_rmt_data(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    data = reshape(_hptt_permutedims(_rmt_array_data(rmt), perm),
                   prod(kept_sizes; init=1),
                   prod(contracted_sizes; init=1),
                   prod(om_sizes; init=1))
    return data
end

function _permuted_rmt_data(rmt::DiagRMT{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    prod(om_sizes; init=1) == 1 || throw(ArgumentError(
        "DiagRMT prepared contraction data requires singleton OM dimension"))

    if rmt.axis[2] == 0
        prepared_axis = _prepared_axis_group(rmt.axis[1], perm, Val(NF), Val(CN))
        prepared_axis == 3 && throw(ArgumentError(
            "DiagRMT vectorized axis cannot be prepared on the OM dimension"))
        return DiagRMT{T,3}(rmt.diag, (prepared_axis, 0))
    end

    axis1 = _prepared_axis_group(rmt.axis[1], perm, Val(NF), Val(CN))
    axis2 = _prepared_axis_group(rmt.axis[2], perm, Val(NF), Val(CN))
    (axis1 == 3 || axis2 == 3) && throw(ArgumentError(
        "DiagRMT diagonal axes cannot be prepared on the OM dimension"))
    axis = axis1 == axis2 ? (axis1, 0) : minmax(axis1, axis2)
    return DiagRMT{T,3}(rmt.diag, axis)
end

function _cached_permuted_rmt_data!(cache::Vector{Array{T, 3}},
                                    sectors::AbstractVector,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    nf::Int,
                                    cn::Int,
                                    nsyms::Int) where {T, RD}
    return _cached_permuted_rmt_data!(cache, sectors, idx, perm,
                                      Val(nf), Val(cn), Val(nsyms))
end

function _cached_permuted_rmt_data!(cache::Vector{Array{T, 3}},
                                    sectors::AbstractVector,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {T, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = _permuted_rmt_data(sector_rmt(sectors[idx]), perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

function _cached_permuted_rmt_data!(cache::Vector{PRMT},
                                    sectors::AbstractVector,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {PRMT<:DiagRMT, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = _permuted_rmt_data(sector_rmt(sectors[idx]), perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

function _cached_permuted_rmt_data!(cache::Vector{Array{T, 3}},
                                    q::TLArray,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {T, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = _permuted_rmt_data(sector_rmt(q, idx), perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

function _cached_permuted_rmt_data!(cache::Vector{PRMT},
                                    q::TLArray,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {PRMT<:DiagRMT, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = _permuted_rmt_data(sector_rmt(q, idx), perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

function prepared_sector_rmt(q::TLArray{T},
                             idx::Int,
                             perm::NTuple{RD, Int},
                             ::Val{NF},
                             ::Val{CN},
                             ::Val{N}) where {T, RD, NF, CN, N}
    rmt, alpha = sector_rmt_with_scale(q, idx)
    return _permuted_rmt_data(rmt, perm, Val(NF), Val(CN), Val(N)), alpha
end

function prepared_sector_rmt(q::TLArrayView{T, QD},
                             idx::Int,
                             perm::NTuple{RD, Int},
                             ::Val{NF},
                             ::Val{CN},
                             ::Val{N}) where {T, QD, RD, NF, CN, N}
    source_perm = ntuple(pos -> perm[pos] <= QD ? q.perm[perm[pos]] : perm[pos], Val(RD))
    prepared, alpha = prepared_sector_rmt(q.arr, idx, source_perm, Val(NF), Val(CN), Val(N))
    scale = alpha * q.scale
    if q.conj
        T <: Real && return prepared, scale
        return conj(prepared), conj(scale)
    end
    return prepared, scale
end

function _cached_prepared_sector_rmt!(cache::Vector{Tuple{PRMT, T}},
                                      q::AbstractTLArray{T},
                                      idx::Int,
                                      perm::NTuple{RD, Int},
                                      ::Val{NF},
                                      ::Val{CN},
                                      ::Val{N}) where {PRMT, T, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = prepared_sector_rmt(q, idx, perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

function _combined_om_factor_info(factors::NTuple{N, LT}) where {N, LT<:AbstractArray{Float64, 3}}
    rank_sizes = ntuple(n -> size(factors[n], 1), Val(N))
    scale = 1.0

    rank_dim = 1
    om1_dim = 1
    om2_dim = 1
    for n in 1:N
        factor = factors[n]
        r_n, om1_n, om2_n = size(factor)

        if r_n == 1 && om1_n == 1 && om2_n == 1
            scale *= factor[1, 1, 1]
            continue
        end

        rank_dim *= r_n
        om1_dim *= om1_n
        om2_dim *= om2_n
    end

    return rank_sizes, rank_dim, om1_dim, om2_dim, scale
end

function _combined_om_factor_array(factors::NTuple{N, LT},
                                   rank_dim_total::Int,
                                   om1_dim_total::Int,
                                   om2_dim_total::Int) where {N, LT<:AbstractArray{Float64, 3}}
    K = ones(Float64, 1, 1, 1)
    rank_dim = 1
    om1_dim = 1
    om2_dim = 1

    for n in 1:N
        factor = factors[n]
        r_n, om1_n, om2_n = size(factor)

        if r_n == 1 && om1_n == 1 && om2_n == 1
            continue
        end

        F = reshape(factor, r_n, om1_n, om2_n)
        K_new = Array{Float64}(undef, rank_dim * r_n,
                               om1_dim * om1_n,
                               om2_dim * om2_n)
        @inbounds for o2n in 1:om2_n, o2p in 1:om2_dim,
                      o1n in 1:om1_n, o1p in 1:om1_dim,
                      rn in 1:r_n, rp in 1:rank_dim
            K_new[rp + (rn - 1) * rank_dim,
                  o1p + (o1n - 1) * om1_dim,
                  o2p + (o2n - 1) * om2_dim] = K[rp, o1p, o2p] * F[rn, o1n, o2n]
        end
        K = K_new
        rank_dim *= r_n
        om1_dim *= om1_n
        om2_dim *= om2_n
    end

    @assert size(K) == (rank_dim_total, om1_dim_total, om2_dim_total)
    return K
end

function _rmt_factor_array_and_rank_sizes(factors::NTuple{N, LT}) where {N, LT<:AbstractArray{Float64, 3}}
    rank_sizes, rank_dim, om1_dim, om2_dim, scale =
        _combined_om_factor_info(factors)
    K = _combined_om_factor_array(factors, rank_dim, om1_dim, om2_dim)
    scale == 1.0 || (K .*= scale)
    return K, rank_sizes
end

function _prepare_compress_sector(
    new_wmats::NTuple{N, Vector{AW}},
    tol::Float64 = 1e-12,
) where {N, AW<:AbstractArray{Float64, 3}}
    K = length(first(new_wmats))
    @assert K > 0 "_compress_sector requires at least one RMT contract"

    U_mats   = Vector{Matrix{Float64}}(undef, N)
    SV_split = Matrix{AW}(undef, N, K)

    for n in 1:N
        shared = _qr_shared_isometry(new_wmats[n]; tol=tol)
        isnothing(shared) && return nothing
        common_iso::Matrix{Float64},
        factors::Vector{AW} = shared

        U_mats[n] = common_iso
        for i in 1:K
            SV_split[n, i] = factors[i]
        end
    end

    factor_arrays = Vector{Array{Float64, 3}}(undef, K)
    first_rank_sizes = ntuple(_ -> 0, Val(N))
    for i in 1:K
        factor_arrays[i], rank_sizes =
            _rmt_factor_array_and_rank_sizes(ntuple(n -> SV_split[n, i], Val(N)))
        i == 1 && (first_rank_sizes = rank_sizes)
    end

    return U_mats, factor_arrays, first_rank_sizes
end

function _rmt_contract_order(fdim::Int, gdim::Int, cdim::Int,
                             o1dim::Int, o2dim::Int, rdim::Int)
    ab_cost = fdim * gdim * o1dim * o2dim * cdim +
              fdim * gdim * rdim * o1dim * o2dim
    ak_cost = fdim * cdim * rdim * o1dim * o2dim +
              fdim * gdim * rdim * cdim * o2dim
    bk_cost = gdim * cdim * rdim * o1dim * o2dim +
              fdim * gdim * rdim * cdim * o1dim

    if ab_cost <= ak_cost && ab_cost <= bk_cost
        return :AB, ab_cost
    elseif ak_cost <= bk_cost
        return :AK, ak_cost
    else
        return :BK, bk_cost
    end
end

function _rmt_contract_temp_len(fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int, rdim::Int,
                                order::Symbol)
    if order === :AB
        return fdim * o1dim * gdim * o2dim
    elseif order === :AK
        return fdim * cdim * o2dim * rdim
    else
        return gdim * cdim * o1dim * rdim
    end
end

function _rmt_contract_temp_len(fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray)
    rank_dim, om1_dim, om2_dim = size(K)
    rank_dim == 1 && om1_dim == 1 && om2_dim == 1 && return 0

    order, _ = _rmt_contract_order(fdim, gdim, cdim, o1dim, o2dim, rank_dim)
    return _rmt_contract_temp_len(fdim, gdim, cdim, o1dim, o2dim, rank_dim, order)
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:AbstractArray, B<:AbstractArray}
    return _rmt_contract_temp_len(fdim, gdim, cdim, o1dim, o2dim, K)
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:DiagRMT, B<:Array}
    return gdim * cdim * size(K, 1)
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:Array, B<:DiagRMT}
    return 0
end

function _rmt_contract_temp_len(::Type{A}, ::Type{B},
                                fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray
) where {A<:DiagRMT, B<:DiagRMT}
    return 0
end

function _rmt_contract_temp_len(A::AbstractArray,
                                B::AbstractArray,
                                K::AbstractArray)
    fdim, cdim, o1dim = size(A)
    gdim, _, o2dim = size(B)
    return _rmt_contract_temp_len(typeof(A), typeof(B),
                                  fdim, gdim, cdim, o1dim, o2dim,
                                  K)
end

function _rmt_contract_dims(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            nf::Int,
                            cn::Int,
                            nsyms::Int) where {T, RD}
    return _rmt_contract_dims(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _rmt_contract_dims(rmt::AbstractArray{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    return (prod(kept_sizes; init=1),
            prod(contracted_sizes; init=1),
            prod(om_sizes; init=1))
end

@inline function _rmt_temp_view(temp::Vector{T},
                               dims::NTuple{N, Int}) where {T, N}
    @assert length(temp) >= prod(dims; init=1)
    return unsafe_wrap(Array, pointer(temp), dims; own=false)
end

function _accumulate_mkl!(out::AbstractArray{T, 3},
                          A::AbstractArray,
                          B::AbstractArray,
                          K::AbstractArray,
                          order::Symbol,
                          temp::Vector{T},
                          beta::T = one(T)) where {T}
    fdim, cdim, o1dim = size(A)
    gdim = size(B, 1)
    o2dim = size(B, 3)
    rdim = size(K, 1)

    if order === :AB
        tmp = _rmt_temp_view(temp, (fdim, gdim, o1dim, o2dim))
        for i in 1:o1dim
            for j in 1:o2dim
                tmp_view = @view tmp[:, :, i, j]
                A_, B_ = @view(A[:, :, i]), @view B[:, :, j]
                mul!(tmp_view, A_, transpose(B_), one(T), zero(T))
            end
        end
        tmp_ = reshape(tmp, fdim * gdim, o1dim * o2dim)
        K_ = reshape(K, rdim, o1dim * o2dim)
        out_ = reshape(out, fdim * gdim, rdim)
        mul!(out_, tmp_, transpose(K_), one(T), beta)

    elseif order === :AK
        tmp = _rmt_temp_view(temp, (fdim, cdim, o2dim, rdim))
        Kt = permutedims(K, (2, 3, 1))
        A_ = reshape(A, fdim * cdim, o1dim)
        Kt_ = reshape(Kt, o1dim, o2dim * rdim)
        tmp_r1 = reshape(tmp, fdim * cdim, o2dim * rdim)
        mul!(tmp_r1, A_, Kt_, one(T), zero(T))

        B_mat_t = transpose(reshape(B, gdim, cdim * o2dim))
        for r in 1:rdim
            tmp_r2 = reshape(@view(tmp[:, :, :, r]), (fdim, cdim * o2dim))
            out_r = @view(out[:, :, r])
            mul!(out_r, tmp_r2, B_mat_t, one(T), beta)
        end

    else
        tmp = _rmt_temp_view(temp, (gdim, cdim, o1dim, rdim))
        Kt = permutedims(K, (3, 2, 1))
        B_ = reshape(B, gdim * cdim, o2dim)
        Kt_ = reshape(Kt, o2dim, o1dim * rdim)
        tmp_r1 = reshape(tmp, gdim * cdim, o1dim * rdim)
        mul!(tmp_r1, B_, Kt_, one(T), zero(T))

        A_mat = reshape(A, fdim, cdim * o1dim)
        for r in 1:rdim
            tmp_r2 = reshape(@view(tmp[:, :, :, r]), (gdim, cdim * o1dim))
            out_r = @view(out[:, :, r])
            mul!(out_r, A_mat, transpose(tmp_r2), one(T), beta)
        end
    end

    return out
end

function _accumulate_scalar_om!(out::AbstractArray{T, 3},
                                A::AbstractArray,
                                B::AbstractArray,
                                scale,
                                beta::T = one(T)) where {T}
    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    @assert cdim == cdim2
    @assert o1dim == 1 && o2dim == 1 && size(out, 3) == 1

    A_mat = reshape(A, fdim, cdim)
    B_mat = reshape(B, gdim, cdim)
    out_mat = @view out[:, :, 1]
    mul!(out_mat, A_mat, transpose(B_mat), scale, beta)
    return out
end

function _contract_RMT_pair_into_generic!(out::AbstractArray{T, 3},
                                          A::AbstractArray{T1, 3},
                                          B::AbstractArray{T2, 3},
                                          K::AbstractArray,
                                          beta::T = one(T)) where {T, T1, T2}
    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    rdim = size(K, 1)
    @assert cdim == cdim2
    _add_contraction_cost!(fdim * gdim * rdim * o1dim * o2dim * cdim)
    return _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta)
end

function _contract_RMT_pair_into_generic_no_count!(out::AbstractArray{T, 3},
                                                   A::AbstractArray{T1, 3},
                                                   B::AbstractArray{T2, 3},
                                                   K::AbstractArray,
                                                   beta::T = one(T)) where {T, T1, T2}
    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    rdim = size(K, 1)
    @assert cdim == cdim2

    @inbounds for r in 1:rdim
        for g in 1:gdim
            for f in 1:fdim
                acc = zero(T)
                for o2 in 1:o2dim
                    for o1 in 1:o1dim
                        for c in 1:cdim
                            acc += A[f, c, o1] * B[g, c, o2] * K[r, o1, o2]
                        end
                    end
                end
                if iszero(beta)
                    out[f, g, r] = acc
                else
                    out[f, g, r] = acc + beta * out[f, g, r]
                end
            end
        end
    end
    return out
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::AbstractArray{T1, 3},
                                  B::AbstractArray{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
    rank_dim, om1_dim, om2_dim = size(K)

    if rank_dim == 1 && om1_dim == 1 && om2_dim == 1
        fdim, cdim, _ = size(A)
        gdim = size(B, 1)
        _add_contraction_cost!(fdim * gdim * cdim)
        return _accumulate_scalar_om!(out, A, B, K[1, 1, 1], beta)
    end

    fdim, cdim, o1dim = size(A)
    gdim, _, o2dim = size(B)
    order, cost = _rmt_contract_order(fdim, gdim, cdim, o1dim, o2dim, rank_dim)
    _add_contraction_cost!(cost)

    return _accumulate_mkl!(out, A, B, K, order, temp, beta)
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::DiagRMT{T1, 3},
                                  B::DiagRMT{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
    return _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta)
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::DiagRMT{T1, 3},
                                  B::AbstractArray{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
                                  
    if A.axis == (1, 2) && size(A, 3) == 1 && size(K, 2) == 1 && iszero(beta)
        fdim, cdim, o1dim = size(A)
        gdim, cdim2, o2dim = size(B)
        rdim = size(K, 1)
        @assert cdim == cdim2
        @assert o1dim == 1
        @assert size(out) == (fdim, gdim, rdim)

        # Diagonal-left common path: first contract the dense right RMT with K
        # into tmp[g, c, r], then transpose c into the output free-left axis.
        _add_contraction_cost!(gdim * cdim * rdim * o2dim)
        tmp = _rmt_temp_view(temp, (gdim, cdim, rdim))
        B_mat = reshape(B, gdim * cdim, o2dim)
        K_mat = reshape(K, rdim, o2dim)
        tmp_mat = reshape(tmp, gdim * cdim, rdim)
        mul!(tmp_mat, B_mat, transpose(K_mat), one(T), zero(T))

        @inbounds for r in 1:rdim
            for g in 1:gdim
                for f in 1:fdim
                    out[f, g, r] = A.diag[f] * tmp[g, f, r]
                end
            end
        end
        return out
    end

    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    rdim = size(K, 1)
    @assert cdim == cdim2
    _add_contraction_cost!(gdim * cdim * rdim * o1dim * o2dim)
    return _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta)
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::AbstractArray{T1, 3},
                                  B::DiagRMT{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  beta::T = one(T)) where {T, T1, T2}
    if B.axis == (1, 2) && size(B, 3) == 1 && size(K, 3) == 1 && iszero(beta)
        fdim, cdim, o1dim = size(A)
        gdim, cdim2, o2dim = size(B)
        rdim = size(K, 1)
        @assert cdim == cdim2
        @assert o2dim == 1
        @assert size(out) == (fdim, gdim, rdim)

        A_mat = reshape(A, fdim * cdim, o1dim)
        K_mat = reshape(K, rdim, o1dim)
        out_mat = reshape(out, fdim * gdim, rdim)
        _add_contraction_cost!(fdim * cdim * rdim * o1dim)
        mul!(out_mat, A_mat, transpose(K_mat), one(T), zero(T))

        @inbounds for r in 1:rdim
            for g in 1:gdim
                scale = B.diag[g]
                for f in 1:fdim
                    out[f, g, r] *= scale
                end
            end
        end
        return out
    end

    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    rdim = size(K, 1)
    @assert cdim == cdim2
    _add_contraction_cost!(fdim * cdim * rdim * o1dim * o2dim)
    return _contract_RMT_pair_into_generic_no_count!(out, A, B, K, beta)
end

# ─── _compress_sector ────────────────────────────────────────────────────────
# Pure-arithmetic core: given K per-pair (w-matrix, RMT) contributions for one
# output sector, compress the per-symmetry w-matrices into a shared basis and
# accumulate the result into a single (U_mats, result_RMT) pair.
#
#   new_wmats[n][i] : (OM3_n, OM12_n_i)  — w-matrix for pair i, symmetry n
#   new_RMTs[i]     : AbstractArray{T} (sz_free..., OM12_1_i,...,OM12_N_i)
#
# Returns:
#   U_mats[n]  : Matrix{Float64} (OM3_n, r_n)  — new CGT wmat per symmetry
#   result_RMT : AbstractArray{T} (sz_free..., r_1,...,r_N)  — compressed RMT
#   nothing    : when QR reveals that any symmetry contributes no usable
#                columns, so the whole merged sector is identically zero
#
# We use QR-based shared isometries for every sector, including K == 1, so the
# resulting basis is normalized consistently with the multi-contribution case.
function _compress_sector(
    new_wmats ::NTuple{N, Vector{AW}},
    new_RMTs  ::Vector{AR},
    QD_out    ::Int,
    tol       ::Float64 = 1e-12,
) where {T, N, RD, AW<:AbstractArray{Float64, 2}, AR<:AbstractArray{T, RD}}
    K = length(new_RMTs)
    @assert K > 0 "_compress_sector requires at least one RMT"

    # Shared QR basis per symmetry
    U_mats   = Vector{AW}(undef, N)
    # SV pieces per symmetry and pair
    SV_split = Matrix{AW}(undef, N, K)  

    for n in 1:N
        shared = _qr_shared_isometry(new_wmats[n]; tol=tol)
        isnothing(shared) && return nothing
        common_iso::AW, 
        factors::Vector{AW} = shared

        U_mats[n] = common_iso
        for i in 1:K
            SV_split[n, i] = factors[i]
        end
    end

    result_RMT = new_RMTs[1]
    for n in 1:N
        result_RMT = _contract_om_axis(result_RMT, SV_split[n, 1], QD_out + n)
    end

    # Reuse the first contracted contribution as the output buffer.  This avoids
    # allocating a separate zero-filled result, and removes the double allocation
    # in the K == 1 case.
    for i in 2:K
        contrib = new_RMTs[i]
        for n in 1:N
            contrib = _contract_om_axis(contrib, SV_split[n, i], QD_out + n)
        end
        result_RMT .+= contrib
    end

    return U_mats, result_RMT
end

function _compress_sector(
    new_wmats    ::NTuple{N, Vector{AW}},
    sector_pairs    ::AbstractVector{Tuple{Int, Int}},
    work1        ::Dict{Int, AR1},
    work2        ::Dict{Int, AR2},
    kept_sizes1  ::NTuple{NF1, Int},
    kept_sizes2  ::NTuple{NF2, Int},
    tol          ::Float64 = 1e-12,
) where {N, AW<:AbstractArray{Float64, 3}, T1, T2, NF1, NF2,
         AR1<:AbstractArray{T1, 3}, AR2<:AbstractArray{T2, 3}}
    prepared = _prepare_compress_sector(new_wmats, tol)
    isnothing(prepared) && return nothing
    first_i, first_j = first(sector_pairs)
    first_data1 = work1[first_i]
    first_data2 = work2[first_j]

    RT = promote_type(eltype(first_data1), eltype(first_data2), Float64)
    max_temp_len = 0
    for i in eachindex(sector_pairs)
        idx1, idx2 = sector_pairs[i]
        max_temp_len = max(max_temp_len,
                           _rmt_contract_temp_len(work1[idx1], work2[idx2],
                                                  prepared[2][i]))
    end
    temp = Vector{RT}(undef, max_temp_len)

    return _contract_prepared_compress_sector(prepared, sector_pairs, work1, work2,
                                              kept_sizes1, kept_sizes2, temp,
                                              Val(NF1 + NF2 + N))
end

@inline _prepared_cache_rmt(x::Tuple) = x[1]
@inline _prepared_cache_rmt(x::AbstractArray) = x
@inline _prepared_cache_scale(x::Tuple) = x[2]
@inline _prepared_cache_scale(x::AbstractArray{T}) where {T} = one(T)

function _contract_prepared_compress_sector(
    prepared,
    sector_pairs    ::AbstractVector{Tuple{Int, Int}},
    work1,
    work2,
    kept_sizes1  ::NTuple{NF1, Int},
    kept_sizes2  ::NTuple{NF2, Int},
    temp         ::Vector{RT},
    ::Val{RD_out}
) where {NF1, NF2, RT, RD_out}
    first_i, first_j = first(sector_pairs)
    first_data1 = _prepared_cache_rmt(work1[first_i])
    first_data2 = _prepared_cache_rmt(work2[first_j])
    U_mats, factor_arrays, rank_sizes = prepared

    result_shape::NTuple{RD_out, Int} = (kept_sizes1..., kept_sizes2..., rank_sizes...)
    result_data::Array{RT, RD_out} = Array{RT, RD_out}(undef, result_shape)
    result_view = reshape(result_data,
                          size(first_data1, 1),
                          size(first_data2, 1),
                          size(first(factor_arrays), 1))

    for i in eachindex(sector_pairs)
        idx1, idx2 = sector_pairs[i]
        entry1 = work1[idx1]
        entry2 = work2[idx2]
        data1 = _prepared_cache_rmt(entry1)
        data2 = _prepared_cache_rmt(entry2)
        pair_alpha = _prepared_cache_scale(entry1) * _prepared_cache_scale(entry2)
        factors = factor_arrays[i]
        factors_scaled = pair_alpha == one(typeof(pair_alpha)) ? factors : factors .* pair_alpha
        beta = i == firstindex(sector_pairs) ? zero(RT) : one(RT)
        _contract_RMT_pair_into!(
            result_view,
            data1,
            data2,
            factors_scaled,
            temp,
            beta,
        )
    end

    return U_mats, result_data
end

# ─── * operator ──────────────────────────────────────────────────────────────
# Automatically contract two TLArray objects by matching their tagged, unlocked
# indices.  An index on q1 is "contractible" when it has a nonempty tag AND
# lock == 0; same criterion applies to q2.  Two contractible indices are matched
# when they compare equal under TLIndex == (same itags, dir, plev, dual) and
# their precomputed leg spaces are equal. The collected matching pairs define
# legs1 / legs2 passed to `contract`.
#function _contract_matched_legs(q1::TLArray, q2::TLArray, legs1::Vector{Int}, legs2::Vector{Int})
#    length(legs1) == length(legs2) || throw(DimensionMismatch("matched leg vectors must have the same length"))
#    if length(legs1) == 1
#        return contract(q1, (legs1[1],), q2, (legs2[1],); verify_legs=false)
#    elseif length(legs1) == 2
#        return contract(q1, (legs1[1], legs1[2]), q2, (legs2[1], legs2[2]); verify_legs=false)
#    elseif length(legs1) == 3
#        return contract(q1, (legs1[1], legs1[2], legs1[3]), q2, (legs2[1], legs2[2], legs2[3]); verify_legs=false)
#    else
#        return contract(q1, Tuple(legs1), q2, Tuple(legs2); verify_legs=false)
#    end
#end

function Base.:*(q1::AbstractTLArray, q2::AbstractTLArray)
    inds1 = inds(q1)
    inds2 = inds(q2)
    spaces1 = spaces(q1)
    spaces2 = spaces(q2)
    # Collect candidate indices from each TLArray.
    cands1 = [(i, inds1[i]) for i in 1:length(inds1)
              if !isempty(inds1[i].itags) && inds1[i].lock == 0]
    cands2 = [(j, inds2[j]) for j in 1:length(inds2)
              if !isempty(inds2[j].itags) && inds2[j].lock == 0]

    # Match candidates: for each index in cands1, find the unique equal index
    # in cands2.  Raise an error if a tag appears more than once on either side.
    legs1 = Int[]
    legs2 = Int[]
    matched2 = Set{Int}()   # positions in cands2 already consumed

    for (i::Int, idx1) in cands1
        hits = [(pos, j, idx2) for (pos, (j, idx2)) in enumerate(cands2)
                if idx1 == change_dir(idx2) &&
                   spaces1[i] == spaces2[j] &&
                   pos ∉ matched2]::Vector{Tuple{Int, Int, TLIndex}}
        if length(hits) > 1
            error("Ambiguous contraction: tag \"$(idx1.itags)\" matches more than one index in q2")
        end
        if length(hits) == 1
            pos::Int, j::Int, _ = hits[1]
            push!(legs1, i)::Vector{Int}
            push!(legs2, j)::Vector{Int}
            push!(matched2, pos)::Set{Int}
        end
    end

    @assert length(legs1) > 0 "No matching contractible indices found between the two TLArray objects"

    return contract(q1, Tuple(legs1), q2, Tuple(legs2))
end

# ─── contract ────────────────────────────────────────────────────────────────
# Contraction sorted by *contracted* qlabels.  RMT pairs are collected as
# pending contractions per output sector, so the actual RMT contraction happens
# after the w-matrices have been compressed in `_compress_sector`.
#
# Algorithm sketch:
#   1. Build sorted sector-info vectors for each TLArray. Each entry carries the
#      sector index, all physical-leg qlabels, and the contracted-qlabel key.
#   2. Count matching contracted-label runs, fill one possible-pair table, and
#      sort it in-place by output free qlabels to form output-sector intervals.
#   3. Build concrete X-symbol caches and reject impossible output sectors by
#      swap-and-pop while collecting w-matrices and source RMT pairs.
#   4. Merge each output sector (QR compression → final RMT contraction).
#   5. Lock reduction / build result TLArray.

# ── Contracted-label helpers ─────────────────────────────────────────────────
function _free_legs(::Val{QD}, legs::NTuple{CN, Int}) where {QD, CN}
    return ntuple(Val(QD - CN)) do k
        nfree = 0
        for l in 1:QD
            if l ∉ legs
                nfree += 1
                nfree == k && return l
            end
        end
        throw(BoundsError())
    end
end

_contracted_qlabel_type(::Type{QT}, ::Val{CN}) where {QT, CN} = NTuple{CN, Tuple{QT, Int}}

@inline _sector_leg_dim(q::AbstractTLArray, sector::Int, leg::Int) =
    sector_rmt_axis_dim(q, sector, leg)

struct ContractSectorInfo{NF, QT, CQT}
    sector_index::Int
    free_qlabels::NTuple{NF, Tuple{QT, Int}}
    contracted_qlabels::CQT
end

function _contract_sector_infos(::Type{QT}, q::AbstractTLArray{T, QD, N},
                             free_legs::NTuple{NF, Int},
                             legs::NTuple{CN, Int}) where {QT, T, QD, N, NF, CN}
    CQT = _contracted_qlabel_type(QT, Val(CN))
    Info = ContractSectorInfo{NF, QT, CQT}
    infos = Vector{Info}(undef, nsectors(q))
    out_pos = 1
    for i in sector_slots(q)
        is_sector_zero(q, i) && continue
        infos[out_pos] = Info(
            i,
            ntuple(k -> (sector_qlabel(QT, q, i, free_legs[k]), _sector_leg_dim(q, i, free_legs[k])), Val(NF)),
            ntuple(k -> (sector_qlabel(QT, q, i, legs[k]), _sector_leg_dim(q, i, legs[k])), Val(CN))::CQT,
        )
        out_pos += 1
    end
    return sort!(infos; by = info -> info.contracted_qlabels, alg=MergeSort)
end

function _contracted_qlabel_run(infos::AbstractVector{<:ContractSectorInfo{NF, QT, CQT}},
                                first_pos::Int) where {NF, QT, CQT}
    key = infos[first_pos].contracted_qlabels
    next_pos = first_pos + 1
    while next_pos <= lastindex(infos) && infos[next_pos].contracted_qlabels == key
        next_pos += 1
    end
    return key, first_pos:(next_pos - 1), next_pos
end

function _contracted_qlabel_overlaps(infos1::AbstractVector{ContractSectorInfo{NF1, QT, CQT}},
                                     infos2::AbstractVector{ContractSectorInfo{NF2, QT, CQT}}) where {NF1, NF2, QT, CQT}
    OverlapT = Tuple{CQT, UnitRange{Int}, UnitRange{Int}}
    overlaps = Vector{OverlapT}()
    sizehint!(overlaps, min(length(infos1), length(infos2)))
    total = 0
    pos1 = firstindex(infos1)
    pos2 = firstindex(infos2)
    while pos1 <= lastindex(infos1) && pos2 <= lastindex(infos2)
        ckey1 = infos1[pos1].contracted_qlabels
        ckey2 = infos2[pos2].contracted_qlabels

        if isless(ckey1, ckey2)
            _, _, pos1 = _contracted_qlabel_run(infos1, pos1)
        elseif isless(ckey2, ckey1)
            _, _, pos2 = _contracted_qlabel_run(infos2, pos2)
        else
            _, run1, next_pos1 = _contracted_qlabel_run(infos1, pos1)
            _, run2, next_pos2 = _contracted_qlabel_run(infos2, pos2)
            push!(overlaps, (ckey1, run1, run2))
            total += length(run1) * length(run2)
            pos1 = next_pos1
            pos2 = next_pos2
        end
    end
    return overlaps, total
end

function _fill_possible_pairs!(pairs::Vector{PairT},
                               infos1::AbstractVector{<:ContractSectorInfo},
                               infos2::AbstractVector{<:ContractSectorInfo},
                               overlaps::AbstractVector{<:Tuple{CQT, UnitRange{Int}, UnitRange{Int}}}) where {PairT, CQT}
    out_pos = 1
    for (ckey, run1, run2) in overlaps
        for p1 in run1
            info1 = infos1[p1]
            for p2 in run2
                info2 = infos2[p2]
                pairs[out_pos] = (ckey, (info1.free_qlabels, info2.free_qlabels),
                                  info1.sector_index, info2.sector_index)::PairT
                out_pos += 1
            end
        end
    end
    @assert out_pos == length(pairs) + 1
    return pairs
end

function _possible_pair_table(infos1::AbstractVector{ContractSectorInfo{NF1, QT, CQT}},
                              infos2::AbstractVector{ContractSectorInfo{NF2, QT, CQT}}) where {NF1, NF2, QT, CQT}
    FreeKey1 = NTuple{NF1, Tuple{QT, Int}}
    FreeKey2 = NTuple{NF2, Tuple{QT, Int}}
    OutKey = Tuple{FreeKey1, FreeKey2}
    PairT = Tuple{CQT, OutKey, Int, Int}
    overlaps, n_possible = _contracted_qlabel_overlaps(infos1, infos2)
    pairs = Vector{PairT}(undef, n_possible)
    _fill_possible_pairs!(pairs, infos1, infos2, overlaps)
    return sort!(pairs; by = p -> p[2], alg = MergeSort)
end

function _out_key_intervals(pairs::AbstractVector{Tuple{CQT, OutKey, Int, Int}}) where {CQT, OutKey}
    nout = 0
    pos = firstindex(pairs)
    while pos <= lastindex(pairs)
        key = pairs[pos][2]
        nout += 1
        next_pos = pos + 1
        while next_pos <= lastindex(pairs) && pairs[next_pos][2] == key
            next_pos += 1
        end
        pos = next_pos
    end

    out_keys = Vector{OutKey}(undef, nout)
    intervals = Vector{UnitRange{Int}}(undef, nout)
    out_pos = 1
    pos = firstindex(pairs)
    while pos <= lastindex(pairs)
        key = pairs[pos][2]
        next_pos = pos + 1
        while next_pos <= lastindex(pairs) && pairs[next_pos][2] == key
            next_pos += 1
        end
        out_keys[out_pos] = key
        intervals[out_pos] = pos:(next_pos - 1)
        out_pos += 1
        pos = next_pos
    end

    return out_keys, intervals
end

function _out_key_qlabels(out_key::Tuple{NTuple{NF1, QT}, NTuple{NF2, QT}},
                          ::Val{QD_out}) where {NF1, NF2, QT, QD_out}
    return ntuple(Val(QD_out)) do leg
        leg <= NF1 ? out_key[1][leg] : out_key[2][leg - NF1]
    end
end

function _out_key_qlabels(out_key::Tuple{NTuple{NF1, Tuple{QT, Int}}, NTuple{NF2, Tuple{QT, Int}}},
                          ::Val{QD_out}) where {NF1, NF2, QT, QD_out}
    return ntuple(Val(QD_out)) do leg
        leg <= NF1 ? out_key[1][leg][1] : out_key[2][leg - NF1][1]
    end
end

function _sector_pairs_from_interval(pairs::AbstractVector,
                                     interval::UnitRange{Int})
    sector_pairs = Vector{Tuple{Int, Int}}(undef, length(interval))
    out_pos = 1
    for pair_pos in interval
        pair = pairs[pair_pos]
        sector_pairs[out_pos] = (pair[3], pair[4])
        out_pos += 1
    end
    return sector_pairs
end

function _contract_xsym_wmat(wm1::AbstractMatrix{T1},
                             xarr::AbstractArray{T2, 3},
                             wm2::AbstractMatrix{T3}) where {T1, T2, T3}
    OM1, OM2, OM3 = size(xarr)
    size(wm1, 1) == OM1 ||
        throw(DimensionMismatch("wm1 first axis must match x-symbol OM1 axis"))
    size(wm2, 1) == OM2 ||
        throw(DimensionMismatch("wm2 first axis must match x-symbol OM2 axis"))

    d1 = size(wm1, 2)
    d2 = size(wm2, 2)
    WT = promote_type(T1, T2, T3)

    result_perm = Array{WT}(undef, d1, d2, OM3)
    temp = Matrix{WT}(undef, d1, OM2)
    @inbounds for a in 1:OM3
        xmat = @view xarr[:, :, a]
        result_mat = @view result_perm[:, :, a]
        mul!(temp, transpose(wm1), xmat, one(WT), zero(WT))
        mul!(result_mat, temp, wm2, one(WT), zero(WT))
    end
    return permutedims(result_perm, (3, 1, 2))
end

function _fill_contract_wmat_concat!(concat::Matrix{Float64},
                                     xarr::AbstractArray{Float64, 3},
                                     wm1::AbstractMatrix,
                                     wm2::AbstractMatrix,
                                     col_offset::Int)
    OM1, OM2, OM3 = size(xarr)
    size(wm1, 1) == OM1 ||
        throw(DimensionMismatch("wm1 first axis must match x-symbol OM1 axis"))
    size(wm2, 1) == OM2 ||
        throw(DimensionMismatch("wm2 first axis must match x-symbol OM2 axis"))

    d1 = size(wm1, 2)
    d2 = size(wm2, 2)
    @inbounds for c in 1:d2, b in 1:d1, a in 1:OM3
        val = 0.0
        for cc in 1:OM2, bb in 1:OM1
            val += xarr[bb, cc, a] * wm1[bb, b] * wm2[cc, c]
        end
        concat[a, col_offset + b + (c - 1) * d1] = val
    end
    return concat
end

@inline _symmetry_qlabels(qlabels::NTuple{NF, QT}, ::Val{n}) where {NF, QT, n} =
    ntuple(k -> qlabels[k][n], Val(NF))

@inline _symmetry_qlabels(qlabels::NTuple{NF, Tuple{QT, Int}}, ::Val{n}) where {NF, QT, n} =
    ntuple(k -> qlabels[k][1][n], Val(NF))

@inline _symmetry_contracted_qlabel(qlabels::NTuple{CN, Tuple{QT, Int}}, ::Val{n}, ::Val{CN}) where {CN, QT, n} =
    ntuple(k -> qlabels[k][1][n], Val(CN))

_xsym_cache_key_type(::Type{QT}, ::Val{n}, ::Val{NF1}, ::Val{NF2}, ::Val{CN}) where {QT, n, NF1, NF2, CN} =
    Tuple{NTuple{CN, fieldtype(QT, n)},
          NTuple{NF1, fieldtype(QT, n)},
          NTuple{NF2, fieldtype(QT, n)}}

const _ABELIAN_WMAT_3D = reshape([1.0], 1, 1, 1)

@generated function _xsym_cache_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
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
    return :(throw(ArgumentError("symmetry index $n has no X-symbol cache")))
end

@inline _xsym_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:AbelianSymm, PS<:ProductSymm, n} = nothing
@inline function _xsym_cache_for(::Type{S}, caches, ::Type{PS}, ::Val{n}) where {S<:NonabelianSymm, PS<:ProductSymm, n}
    slot = _xsym_cache_slot(PS, Val(n))
    slot === nothing &&
        throw(ArgumentError("symmetry index $n is Abelian and has no X-symbol cache"))
    return caches[slot]
end

@inline _xsym_cache_for(caches::CT, ::Type{S}, ::Type{PS}, ::Val{n}) where {CT<:Tuple, S, PS<:ProductSymm, n} =
    _xsym_cache_for(S, caches, PS, Val(n))

@generated function _new_xsym_caches(::Type{PS}, ::Type{QT},
                                     ::Val{NF1}, ::Val{NF2}, ::Val{CN}) where {PS<:ProductSymm, QT, NF1, NF2, CN}
    syms = product_symms(PS)
    seen = Type[]
    exprs = Expr[]
    for (n, S) in pairs(syms)
        S <: AbelianSymm && continue
        S in seen && continue
        push!(seen, S)
        Key = _xsym_cache_key_type(QT, Val(n), Val(NF1), Val(NF2), Val(CN))
        Cache = Dict{Key, Array{Float64, 3}}
        push!(exprs, :($Cache()))
    end
    return Expr(:tuple, exprs...)
end

@inline _impossible_key(free_qlabels1::NTuple{NF1, QT},
                        free_qlabels2::NTuple{NF2, QT},
                        ::Val{n}) where {NF1, NF2, QT, n} =
    (_symmetry_qlabels(free_qlabels1, Val(n)),
     _symmetry_qlabels(free_qlabels2, Val(n)))

@inline _impossible_key(free_qlabels1::NTuple{NF1, Tuple{QT, Int}},
                        free_qlabels2::NTuple{NF2, Tuple{QT, Int}},
                        ::Val{n}) where {NF1, NF2, QT, n} =
    (_symmetry_qlabels(free_qlabels1, Val(n)),
     _symmetry_qlabels(free_qlabels2, Val(n)))

@generated function _new_impossible_sets(::Type{PS}, ::Type{QT},
                                         ::Val{NF1}, ::Val{NF2}) where {PS<:ProductSymm, QT, NF1, NF2}
    syms = product_symms(PS)
    exprs = Vector{Union{Expr, Symbol}}()
    for (n, S) in pairs(syms)
        if S <: AbelianSymm
            push!(exprs, :(nothing))
        else
            K = Tuple{NTuple{NF1, fieldtype(QT, n)}, NTuple{NF2, fieldtype(QT, n)}}
            push!(exprs, :(Set{$K}()))
        end
    end
    return Expr(:tuple, exprs...)
end

@generated function _out_key_is_impossible(::Type{ProductSymm{Syms}},
                                           impossible_sets,
                                           out_key::Tuple{NTuple{NF1, Tuple{QT, Int}}, NTuple{NF2, Tuple{QT, Int}}}) where {Syms, NF1, NF2, QT}
    stmts = Expr[]
    for n in 1:length(Syms.parameters)
        S = Syms.parameters[n]
        S <: AbelianSymm && continue
        push!(stmts, quote
            (_impossible_key(out_key[1], out_key[2], Val($n)) in impossible_sets[$n]) && return true
        end)
    end
    return quote
        $(stmts...)
        return false
    end
end

@inline function _xsym_cache_key(::Type{QT},
                                 contracted_qlabels::NTuple{CN, Tuple{QT, Int}},
                                 free_qlabels1::NTuple{NF1, Tuple{QT, Int}},
                                 free_qlabels2::NTuple{NF2, Tuple{QT, Int}},
                                 ::Val{n}) where {QT, NF1, NF2, n, CN}
    return (_symmetry_contracted_qlabel(contracted_qlabels, Val(n), Val(CN)),
            _symmetry_qlabels(free_qlabels1, Val(n)),
            _symmetry_qlabels(free_qlabels2, Val(n)))
end

@inline _symmetry_sector_qlabels(q::AbstractTLArray{T, QD, N}, sector::Int, ::Val{n}) where {T, QD, N, n} =
    ntuple(l -> sector_qlabel(q, sector, l)[n], Val(QD))

function _stored_symmetry_leg_order(qlabels::NTuple{QD, NTuple{NZ, Int}},
                                    inds::NTuple{QD, TLIndex}) where {QD, NZ}
    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for l in 1:QD
        if inds[l].dir == '+'
            push!(incoming, l)
        else
            push!(outgoing, l)
        end
    end
    sort!(incoming; by = l -> qlabels[l], alg = MergeSort)
    sort!(outgoing; by = l -> qlabels[l], alg = MergeSort)
    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD)), n_in
end

@inline _stored_contract_legs(phys_to_stored::NTuple{QD, Int}, legs::NTuple{CN, Int}) where {QD, CN} =
    ntuple(k -> phys_to_stored[legs[k]], Val(CN))

function _xsymbol_sector_args(q::AbstractTLArray{T, QD, N},
                              sector::Int,
                              legs::NTuple{CN, Int},
                              ::Val{n}) where {T, QD, N, CN, n}
    qlabels_by_phys = _symmetry_sector_qlabels(q, sector, Val(n))
    stored_to_phys, n_in = _stored_symmetry_leg_order(qlabels_by_phys, inds(q))
    qlabels = ntuple(i -> qlabels_by_phys[stored_to_phys[i]], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    upsp = Tuple(qlabels[i] for i in 1:n_in)
    dnsp = Tuple(qlabels[i] for i in (n_in + 1):QD)
    ctlegs = _stored_contract_legs(phys_to_stored, legs)
    return upsp, dnsp, ctlegs
end

function _load_nonabelian_xarr(::Type{S},
                               q1::AbstractTLArray,
                               i1::Int,
                               q2::AbstractTLArray,
                               i2::Int,
                               legs1::NTuple{CN, Int},
                               legs2::NTuple{CN, Int},
                               ::Val{n}) where {S<:NonabelianSymm, CN, n}
    up1sp, dn1sp, ctlegs1 = _xsymbol_sector_args(q1, i1, legs1, Val(n))
    up2sp, dn2sp, ctlegs2 = _xsymbol_sector_args(q2, i2, legs2, Val(n))
    xsym_obj = getNsave_Xsymbol(S, up1sp, dn1sp, up2sp, dn2sp, ctlegs1, ctlegs2)
    isnothing(xsym_obj) && return nothing
    return xsym_obj.xsym_arr::Array{Float64, 3}
end

@inline function _xarr_for_symmetry(::Type{S},
                                    q1::AbstractTLArray,
                                    i1::Int,
                                    q2::AbstractTLArray,
                                    i2::Int,
                                    contracted_qlabels::NTuple{CN, Tuple{QT, Int}},
                                    free_qlabels1::NTuple{NF1, Tuple{QT, Int}},
                                    free_qlabels2::NTuple{NF2, Tuple{QT, Int}},
                                    xsym_cache::Dict{K, Array{Float64, 3}},
                                    impossible_sets::Tuple,
                                    legs1::NTuple{CN, Int},
                                    legs2::NTuple{CN, Int},
                                    ::Val{n}) where {S<:NonabelianSymm, QT, NF1, NF2, CN, n, K}
    xkey = _xsym_cache_key(QT, contracted_qlabels, free_qlabels1, free_qlabels2, Val(n))
    if haskey(xsym_cache, xkey)
        return xsym_cache[xkey]
    end

    loaded = _load_nonabelian_xarr(S, q1, i1, q2, i2, legs1, legs2, Val(n))
    if loaded === nothing
        push!(impossible_sets[n], _impossible_key(free_qlabels1, free_qlabels2, Val(n)))
        return nothing
    end
    xsym_cache[xkey] = loaded
    return loaded
end

function _qr_contract_concat(concat::Matrix{Float64},
                             pair_shapes::Vector{Tuple{Int, Int}})
    nrows = size(concat, 1)
    if nrows == 1
        Q = ones(1, 1)
        factors = Vector{Array{Float64, 3}}(undef, length(pair_shapes))
        col = 0
        for i in eachindex(pair_shapes)
            d1, d2 = pair_shapes[i]
            width = d1 * d2
            factors[i] = reshape(copy(concat[:, col+1:col+width]), 1, d1, d2)
            col += width
        end
        return Q, factors
    end

    F = qr(concat)
    Qfull = Matrix(F.Q)
    Rfull = Matrix(F.R)

    used = 0
    @inbounds for i in axes(Rfull, 1)
        nrm_sq = 0.0
        for j in axes(Rfull, 2)
            nrm_sq += abs2(Rfull[i, j])
        end
        !iszero(nrm_sq) && (used = i)
    end
    used == 0 && return nothing

    Q = Qfull[:, 1:used]
    R = Rfull[1:used, :]
    factors = Vector{Array{Float64, 3}}(undef, length(pair_shapes))
    col = 0
    for i in eachindex(pair_shapes)
        d1, d2 = pair_shapes[i]
        width = d1 * d2
        factors[i] = reshape(copy(R[:, col+1:col+width]), used, d1, d2)
        col += width
    end
    @assert col == size(R, 2)

    return Q, factors
end

function _prepare_contract_interval_for_symmetry(::Type{S},
                                                 q1::AbstractTLArray,
                                                 q2::AbstractTLArray,
                                                 possible_pairs::AbstractVector,
                                                 interval::UnitRange{Int},
                                                 xsym_caches::Tuple,
                                                 impossible_sets::Tuple,
                                                 legs1::NTuple{CN, Int},
                                                 legs2::NTuple{CN, Int},
                                                 ::Val{n}) where {S<:AbelianSymm, CN, n}
    K = length(interval)
    Q = ones(1, 1)
    factors = Vector{Array{Float64, 3}}(undef, K)
    for i in 1:K
        factors[i] = _ABELIAN_WMAT_3D
    end
    return Q, factors
end

function _prepare_contract_interval_for_symmetry(::Type{S},
                                                 q1::AbstractTLArray,
                                                 q2::AbstractTLArray,
                                                 possible_pairs::AbstractVector,
                                                 interval::UnitRange{Int},
                                                 xsym_caches::Tuple,
                                                 impossible_sets::Tuple,
                                                 legs1::NTuple{CN, Int},
                                                 legs2::NTuple{CN, Int},
                                                 ::Val{n}) where {S<:NonabelianSymm, CN, n}
    K = length(interval)
    xarrs = Vector{Array{Float64, 3}}(undef, K)
    pair_shapes = Vector{Tuple{Int, Int}}(undef, K)
    nrows = 0
    total_width = 0

    i = 1
    for pair_pos in interval
        pair = possible_pairs[pair_pos]
        contracted_qlabels = pair[1]
        free_qlabels1, free_qlabels2 = pair[2]
        i1 = pair[3]
        i2 = pair[4]
        xarr = _xarr_for_symmetry(S, q1, i1, q2, i2,
                                  contracted_qlabels, free_qlabels1, free_qlabels2,
                                  _xsym_cache_for(xsym_caches, S, productsymm(q1), Val(n)),
                                  impossible_sets, legs1, legs2, Val(n))
        xarr === nothing && return nothing
        xarrs[i] = xarr

        wm1 = sector_wmat(q1, i1, Val(n))
        wm2 = sector_wmat(q2, i2, Val(n))
        d1 = size(wm1, 2)
        d2 = size(wm2, 2)
        pair_shapes[i] = (d1, d2)
        nrows == 0 && (nrows = size(xarr, 3))
        @assert nrows == size(xarr, 3)
        total_width += d1 * d2
        i += 1
    end

    concat = Matrix{Float64}(undef, nrows, total_width)
    col = 0
    i = 1
    for pair_pos in interval
        pair = possible_pairs[pair_pos]
        i1 = pair[3]
        i2 = pair[4]
        wm1 = sector_wmat(q1, i1, Val(n))
        wm2 = sector_wmat(q2, i2, Val(n))
        _fill_contract_wmat_concat!(concat, xarrs[i], wm1, wm2, col)
        col += pair_shapes[i][1] * pair_shapes[i][2]
        i += 1
    end
    @assert col == total_width

    return _qr_contract_concat(concat, pair_shapes)
end

@generated function _prepare_contract_interval(::Type{ProductSymm{Syms}},
                                               q1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
                                               q2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
                                               possible_pairs::AbstractVector,
                                               interval::UnitRange{Int},
                                               xsym_caches::CT,
                                               impossible_sets::Tuple,
                                               legs1::NTuple{CN, Int},
                                               legs2::NTuple{CN, Int}) where {Syms, T1, QD1, N, RD1, QT, PS, M, RMT1,
                                                                              T2, QD2, RD2, RMT2, CT<:Tuple, CN}
    stmts = Expr[]
    qnames = Symbol[]
    fnames = Symbol[]
    for n in 1:N
        S = Syms.parameters[n]
        res = Symbol(:prepared_sym_, n)
        q = Symbol(:Q_, n)
        f = Symbol(:factors_, n)
        push!(qnames, q)
        push!(fnames, f)
        push!(stmts, quote
            local $res = _prepare_contract_interval_for_symmetry(
                $S, q1, q2, possible_pairs, interval, xsym_caches,
                impossible_sets, legs1, legs2, Val($n))
            $res === nothing && return nothing
            local $q = $res[1]::Matrix{Float64}
            local $f = $res[2]::Vector{Array{Float64, 3}}
        end)
    end
    factor_tuple = Expr(:tuple, (:(SV_split[$n][i]) for n in 1:N)...)
    zero_rank_sizes = Expr(:tuple, (0 for _ in 1:N)...)

    return quote
        local K = length(interval)
        K > 0 || throw(ArgumentError("contract interval must contain at least one sector pair"))
        $(stmts...)
        local U_mats = Matrix{Float64}[$(qnames...)]
        local SV_split = ($(fnames...),)
        local factor_arrays = Vector{Array{Float64, 3}}(undef, K)
        local first_rank_sizes = $zero_rank_sizes
        for i in 1:K
            factor_arrays[i], rank_sizes =
                _rmt_factor_array_and_rank_sizes($factor_tuple)
            i == 1 && (first_rank_sizes = rank_sizes)
        end
        return U_mats, factor_arrays, first_rank_sizes
    end
end

# ── Convenience overloads ─────────────────────────────────────────────────────
contract(q1, l1::Int, q2, l2::Int) = contract(q1, (l1,), q2, (l2,))

function contract(q1::AbstractTLArray, legs1::AbstractVector{<:Integer},
                  q2::AbstractTLArray, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
end

# ── Main entry point ──────────────────────────────────────────────────────────
function contract(q1::AbstractTLArray{T1, QD1, N, RD1, QT, PS, M, RMT1},
                  legs1::NTuple{CN, Int},
                  q2::AbstractTLArray{T2, QD2, N, RD2, QT, PS, M, RMT2},
                  legs2::NTuple{CN, Int};
                  reduce_lock::Bool=true,
                  verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, QT, PS, M, RMT1, RMT2, CN}

    symmetries = product_symms(PS)
    inds1 = inds(q1)
    inds2 = inds(q2)
    spaces1 = spaces(q1)
    spaces2 = spaces(q2)

    if verify_legs
        for i in 1:CN
            idx1::TLIndex = inds1[legs1[i]::Int]
            idx2::TLIndex = inds2[legs2[i]::Int]
            @assert idx1.dir != idx2.dir "Contracted legs must have opposite arrow directions: " *
                "q1 leg $(legs1[i]) has dir='$(idx1.dir)', q2 leg $(legs2[i]) has dir='$(idx2.dir)'"
            @assert idx1.itags == idx2.itags "Contracted legs must have matching itags: " *
                "q1 leg $(legs1[i]) has itag='$(idx1.itags)', q2 leg $(legs2[i]) has itag='$(idx2.itags)'"
            @assert idx1.dual == idx2.dual "Contracted legs must have matching dual flags: " *
                "q1 leg $(legs1[i]) has dual=$(idx1.dual), q2 leg $(legs2[i]) has dual=$(idx2.dual)"
            @assert spaces1[legs1[i]] == spaces2[legs2[i]] "Contracted legs must have matching space info: " *
                "q1 leg $(legs1[i]) spaces != q2 leg $(legs2[i]) spaces"
        end
    end

    T    = promote_type(T1, T2)

    nf1 = QD1 - CN
    nf2 = QD2 - CN
    free1::NTuple{nf1, Int}  = _free_legs(Val(QD1), legs1)
    free2::NTuple{nf2, Int}  = _free_legs(Val(QD2), legs2)
    QD_out = QD1 + QD2 - 2CN
    RD_out = QD_out + N

    inds_out = (ntuple(i -> inds1[free1[i]], Val(QD1 - CN))...,
                ntuple(i -> inds2[free2[i]], Val(QD2 - CN))...)

    # Fixed permutations: (free..., contracted..., om...).
    perm1 = (free1..., legs1..., ntuple(n -> QD1 + n, Val(N))...)
    perm2 = (free2..., legs2..., ntuple(n -> QD2 + n, Val(N))...)
    #println(perm1)
    #println(perm2)

    PermutedRMT1 = _permuted_rmt_type(RMT1)
    PermutedRMT2 = _permuted_rmt_type(RMT2)
    permuted_rmts1 = Vector{Tuple{PermutedRMT1, T1}}(undef, sector_count(q1))
    permuted_rmts2 = Vector{Tuple{PermutedRMT2, T2}}(undef, sector_count(q2))

    # ── 1. Build sorted sector-info vectors keyed by contracted qlabels ─────────
    sector_infos1 = _contract_sector_infos(QT, q1, free1, legs1)
    sector_infos2 = _contract_sector_infos(QT, q2, free2, legs2)

    # ── 2. Output-sector accumulator ────────────────────────────────────────
    FreeKey1  = NTuple{nf1, Tuple{QT, Int}}
    FreeKey2  = NTuple{nf2, Tuple{QT, Int}}
    OutKey    = Tuple{FreeKey1, FreeKey2}

    SectorPairVec = Vector{Tuple{Int, Int}}

    xsym_caches = _new_xsym_caches(PS, QT, Val(nf1), Val(nf2), Val(CN))
    impossible_sets = _new_impossible_sets(PS, QT, Val(nf1), Val(nf2))

    # ── 3. Collect possible pairs and output intervals ───────────────────────
    possible_pairs = _possible_pair_table(sector_infos1, sector_infos2)
    out_keys, out_intervals = _out_key_intervals(possible_pairs)
    valid_output = trues(length(out_keys))

    # ── 4. Direct-fill QR/K data and one contraction temporary ───────────────
    PreparedSectorT = Tuple{Vector{Matrix{Float64}},
                            Vector{Array{Float64, 3}},
                            NTuple{N, Int}}
    prepared_sectors = Vector{PreparedSectorT}(undef, length(out_keys))
    prepared_sector_pairs = Vector{SectorPairVec}(undef, length(out_keys))
    max_temp_len = 0
    for out_pos in eachindex(out_keys)
        valid_output[out_pos] || continue

        if _out_key_is_impossible(PS, impossible_sets, out_keys[out_pos])
            valid_output[out_pos] = false
            continue
        end

        prepared = _prepare_contract_interval(
            PS, q1, q2, possible_pairs, out_intervals[out_pos],
            xsym_caches, impossible_sets, legs1, legs2)
        if isnothing(prepared)
            valid_output[out_pos] = false
            continue
        end
        _, factor_arrays, _ = prepared

        sector_pairs = _sector_pairs_from_interval(possible_pairs, out_intervals[out_pos])
        @assert length(sector_pairs) == length(factor_arrays)
        for i in eachindex(sector_pairs)
            idx1, idx2 = sector_pairs[i]
            fdim, cdim1, o1dim =
                size(_cached_prepared_sector_rmt!(permuted_rmts1, q1, idx1,
                                                  perm1, Val(nf1), Val(CN), Val(N))[1])
            gdim, cdim2, o2dim =
                size(_cached_prepared_sector_rmt!(permuted_rmts2, q2, idx2,
                                                  perm2, Val(nf2), Val(CN), Val(N))[1])
            @assert cdim1 == cdim2
            max_temp_len = max(max_temp_len,
                               _rmt_contract_temp_len(PermutedRMT1, PermutedRMT2,
                                                      fdim, gdim, cdim1,
                                                      o1dim, o2dim,
                                                      factor_arrays[i]))
        end

        prepared_sectors[out_pos] = prepared
        prepared_sector_pairs[out_pos] = sector_pairs
    end
    contract_temp = Vector{promote_type(T1, T2, Float64)}(undef, max_temp_len)

    # ── 5. Merge each output sector ──────────────────────────────────────────
    RT = promote_type(T1, T2, Float64)
    res_nsectors = count(valid_output)
    out_to_result = zeros(Int, length(out_keys))
    result_pos = 1
    for out_pos in eachindex(out_keys)
        valid_output[out_pos] || continue
        out_to_result[out_pos] = result_pos
        result_pos += 1
    end
    result_qlabels = Matrix{QT}(undef, QD_out, res_nsectors)
    result_RMTs = Vector{Array{RT, RD_out}}(undef, res_nsectors)
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    result_wmatinfo = Vector{WMatInfo{M}}(undef, res_nsectors)
    total_wmat_len = 0
    for out_pos in eachindex(out_keys)
        valid_output[out_pos] || continue
        result_pos = out_to_result[out_pos]
        U_mats = prepared_sectors[out_pos][1]
        result_wmatinfo[result_pos] = ntuple(Val(M)) do m
            n = nonabelian_indices[m]
            U = U_mats[n]
            len = length(U)
            offset = total_wmat_len + 1
            total_wmat_len += len
            (offset, size(U, 1), size(U, 2))
        end
    end
    result_wmatdata = Vector{Float64}(undef, total_wmat_len)

    # ── 5a. THREADED OVER OUTPUT SECTORS ─────────────────────────────────────
    # This loop is where a block-sparse contraction spends its arithmetic: one gemm chain per
    # output sector, and at chi = 1024 on an L = 30 U(1) chain that is 9-15 sectors with
    # left-leg dimensions in the hundreds. Serially, a 128-core node ran it at a few cores'
    # worth of throughput.
    #
    # WHY IT IS SAFE, item by item -- all three had to be checked, because only the first is
    # obvious:
    #   * `contract_temp` WAS the blocker: one scratch buffer, sized to the max over all
    #     sectors, written through by every iteration. Each task now owns one.
    #   * `permuted_rmts1/2` are mutable caches and `_cached_prepared_sector_rmt!` writes on a
    #     miss -- a race if two sectors share an input sector, which is what the cache is for.
    #     ⛔ I FIRST ARGUED THIS AWAY AND WAS WRONG. The argument was that step 4 already calls
    #     the helper for every pair of every valid output sector while sizing `max_temp_len`,
    #     so every entry must be warm by here. Threaded runs then failed with
    #     `UndefRefError: access to undefined reference` and, once, `svd produced no CGTSVD
    #     classes` a few steps downstream -- i.e. some entries were NOT warm, and the ones
    #     that were got read alongside garbage. The pre-warm loop below makes it true instead
    #     of assuming it: serial, unconditional, and free when the entries are already there.
    #     An invariant that has to hold for correctness is worth ten lines, not a comment.
    #   * `result_qlabels`, `result_wmatdata`, `result_RMTs` are written at indices derived
    #     from `result_pos`, which is unique per output sector, and `result_wmatinfo` offsets
    #     were laid out in a serial pass above. Disjoint, so no reduction is needed.
    #
    # Order of accumulation WITHIN a sector is untouched, and sectors write to different slots,
    # so the result is bit-identical to the serial path. That is a test, not a hope --
    # `test_contract_threaded.jl` asserts it.
    #
    # ⚠ Hand-rolled tasks rather than `Threads.@threads`: sector costs vary by more than an
    # order of magnitude (block areas 6144 to 118272 in the measured histogram), so static
    # splitting would leave most threads idle behind the one big sector. An atomic counter
    # gives dynamic balance. `@threads :dynamic` would also balance, but indexing a per-thread
    # buffer by `threadid()` is unsound there -- tasks may migrate between threads.
    todo = Int[op for op in eachindex(out_keys) if valid_output[op]]

    # PRE-WARM THE PERMUTED-RMT CACHES, SERIALLY. `_contract_prepared_compress_sector` reads
    # `work1[idx]` directly and cannot fill a miss, so every entry the threaded loop will
    # touch has to exist before the first task starts. Cache hits are a branch and a load.
    for out_pos in todo
        for (idx1, idx2) in prepared_sector_pairs[out_pos]
            _cached_prepared_sector_rmt!(permuted_rmts1, q1, idx1,
                                         perm1, Val(nf1), Val(CN), Val(N))
            _cached_prepared_sector_rmt!(permuted_rmts2, q2, idx2,
                                         perm2, Val(nf2), Val(CN), Val(N))
        end
    end

    # ⚠ `temp` is deliberately UNANNOTATED. Writing `temp::Vector{TempT}` with a local
    # `TempT = promote_type(...)` is a syntax error -- "local variable cannot be used in
    # closure declaration" -- because the signature is evaluated where the local is not in
    # scope. The element type is fixed at the two `Vector{...}` construction sites below.
    function _merge_out_sector!(out_pos::Int, temp)
        # ⛔⛔ EVERY ONE OF THESE `local`s IS LOAD-BEARING. JULIA CLOSURES CAPTURE, THEY DO NOT
        # SHADOW: assigning a name inside a nested function REBINDS the enclosing local when
        # one exists, and `result_pos`, `U_mats`, `n`, `U` and `offset` are all assigned by the
        # `result_wmatinfo` loop a few lines above. Without `local`, every task writes through
        # the SAME `result_pos` and they scribble over each other's output slots.
        #
        # This was the second of two bugs in this threading. The first (unwarmed RMT caches)
        # announced itself with `UndefRefError`; this one did not throw at all — it produced
        # a plausible-looking TLArray whose sectors had been shuffled, and the failure surfaced
        # several operations downstream as `svd produced no CGTSVD classes`. Serial passed
        # every time. ⚠ A missing `local` in a spawned closure is invisible until the numbers
        # are wrong, which is exactly why `test_contract_threaded.jl` compares with `==`.
        local result_pos = out_to_result[out_pos]
        local out_qlabels = _out_key_qlabels(out_keys[out_pos], Val(QD_out))
        for leg in 1:QD_out
            result_qlabels[leg, result_pos] = out_qlabels[leg]
        end

        local prepared = prepared_sectors[out_pos]
        local sector_pairs = prepared_sector_pairs[out_pos]
        local kept_sizes1 = ntuple(i -> out_keys[out_pos][1][i][2], Val(nf1))
        local kept_sizes2 = ntuple(i -> out_keys[out_pos][2][i][2], Val(nf2))

        local U_mats, result_RMT
        U_mats, result_RMT = _contract_prepared_compress_sector(
            prepared, sector_pairs, permuted_rmts1, permuted_rmts2,
            kept_sizes1, kept_sizes2, temp, Val(RD_out))
        for m in 1:M
            local n = nonabelian_indices[m]
            local U = U_mats[n]
            local offset, nrow, ncol
            offset, nrow, ncol = result_wmatinfo[result_pos][m]
            copyto!(view(result_wmatdata, offset:offset + nrow * ncol - 1), vec(U))
        end
        iszero(sum(abs2, result_RMT)) && return nothing
        result_RMTs[result_pos] = result_RMT
        return nothing
    end

    ntasks = min(length(todo), Threads.nthreads(), TELUM_CONTRACT_THREADS[])
    if ntasks <= 1
        for out_pos in todo
            _merge_out_sector!(out_pos, contract_temp)
        end
    else
        next = Threads.Atomic{Int}(1)
        tasks = map(1:ntasks) do _
            Threads.@spawn begin
                temp = Vector{promote_type(T1, T2, Float64)}(undef, max_temp_len)
                while true
                    i = Threads.atomic_add!(next, 1)
                    i > length(todo) && break
                    _merge_out_sector!(todo[i], temp)
                end
            end
        end
        foreach(wait, tasks)
    end

    #nrow1 = length(permuted_rmts1)
    #used_rows1 = count(isassigned(permuted_rmts1, i) for i in eachindex(permuted_rmts1))
    #nrow2 = length(permuted_rmts2)
    #used_rows2 = count(isassigned(permuted_rmts2, i) for i in eachindex(permuted_rmts2))
    #println("Contracted $(used_rows1)/$(nrow1) sectors from q1, $(used_rows2)/$(nrow2) sectors from q2")
    #println("===============================================================")
    #for i in eachindex(permuted_rmts1)
    #    if !isassigned(permuted_rmts1, i)
    #        sz = size(sector_rmt(q1, i))
    #        println(sz)
    #    end
    #end

    # ── 6. Lock reduction ────────────────────────────────────────────────────
    changed_inds2 = Set(change_dir(inds2[l]) for l in 1:QD2)
    changed_inds1 = Set(change_dir(inds1[l]) for l in 1:QD1)
    final_inds = if reduce_lock
        ntuple(Val(QD_out)) do l
            idx = inds_out[l]
            has_match = l <= nf1 ? (idx ∈ changed_inds2) : (idx ∈ changed_inds1)
            (idx.lock > 0 && has_match) ?
                TLIndex(idx.itags, idx.dir, idx.plev, idx.lock - 1, idx.dual) : idx
        end
    else
        inds_out
    end

    spaces_out = (ntuple(i -> spaces1[free1[i]], Val(QD1 - CN))...,
                  ntuple(i -> spaces2[free2[i]], Val(QD2 - CN))...)

    return TLArray(symmetries, result_qlabels, result_wmatdata, result_wmatinfo, result_RMTs,
                          final_inds, spaces_out)::TLArray{promote_type(T1, T2, Float64), QD_out, N, RD_out, QT, PS, M, Array{promote_type(T1, T2, Float64), RD_out}}
end
