using LinearAlgebra
using SparseArrayKit
using Test
const change_dir = Telum.change_dir
⊗(a, b) = kron(b, a)



# ─── _leg_kron ────────────────────────────────────────────────────────────────
# Kronecker product of two tensors, taken leg-by-leg.
#   A: (d_1a, ..., d_QDa, Ma)
#   B: (d_1b, ..., d_QDb, Mb)
#   Result: (d_1a*d_1b, ..., d_QDa*d_QDb, Ma*Mb)
# The last axis in each operand is treated as a "bond/weight" index and is
# also Kronecker-products so the result carries the full combined bond.
function _leg_kron(A::AbstractArray, B::AbstractArray, QD::Int)
    da = ntuple(l -> size(A, l), QD)
    db = ntuple(l -> size(B, l), QD)
    Ma, Mb = size(A, QD+1), size(B, QD+1)

    # Expand A: insert QD singleton dims between physical and bond, plus a 1 at end
    #   (da..., 1..._QD, Ma, 1)
    Ar = reshape(A, da..., ones(Int, QD)..., Ma, 1)
    # Expand B: insert QD singleton dims at the front, plus a 1 before Mb
    #   (1..._QD, db..., 1, Mb)
    Br = reshape(B, ones(Int, QD)..., db..., 1, Mb)

    # Broadcast multiply → (da..., db..., Ma, Mb)
    outer = Ar .* Br

    # Permute so matching legs are adjacent: (d1a,d1b, d2a,d2b, ..., Ma,Mb)
    perm = vcat([[l, QD+l] for l in 1:QD]..., 2QD+1, 2QD+2)
    outer_p = permutedims(outer, perm)

    # Reshape interleaved pairs into single legs
    return reshape(outer_p, ntuple(l -> da[l]*db[l], QD)..., Ma*Mb)
end

# ─── _legwise_kron ───────────────────────────────────────────────────────────
# Leg-wise Kronecker product of two QD-dimensional tensors (no bond leg).
#   A: (d_1, ..., d_QD),  B: (s_1, ..., s_QD)
#   Result: (d_1*s_1, ..., d_QD*s_QD)
# Convention: kron(A_l, B_l) per leg — A (CGT) is the slow index, B (RMT) fast.
function _legwise_kron(A::AbstractArray, B::AbstractArray)
    QD = ndims(A)
    da = ntuple(l -> size(A, l), QD)
    db = ntuple(l -> size(B, l), QD)
    # Expand A → (d1,1,d2,1,...), B → (1,s1,1,s2,...)
    Ar = reshape(A, Tuple(Iterators.flatten((da[l], 1) for l in 1:QD)))
    Br = reshape(B, Tuple(Iterators.flatten((1, db[l]) for l in 1:QD)))
    outer = Ar .* Br   # shape (d1, s1, d2, s2, ..., dQD, sQD)
    # Keep pair order (dl, sl) so that dl (A) is fast in the reshape.
    # Combined index per leg = (sl-1)*dl + rl  ↔  kron(B_l, A_l) w/ A fast
    perm = Tuple(Iterators.flatten((2l-1, 2l) for l in 1:QD))
    outer_p = permutedims(outer, perm)  # (d1, s1, d2, s2, ...)
    return reshape(outer_p, ntuple(l -> da[l]*db[l], QD)...)
end

# ─── contract_sparse ─────────────────────────────────────────────────────────
# Arbitrary contraction of two arrays over matching leg pairs.
# Remaining legs keep their original relative order:
#   result legs = (free legs of A in original order, free legs of B in original order)
#
#   A     : any AbstractArray  (works with SparseArray, plain Array, …)
#   B     : any AbstractArray
#   legs_A: legs of A to contract (1-based, length M)
#   legs_B: corresponding legs of B (must match sizes pairwise)
function contract_sparse(A::AbstractArray, B::AbstractArray,
                         legs_A::NTuple{M, Int},
                         legs_B::NTuple{M, Int}) where M
    keep_A = [i for i in 1:ndims(A) if !(i in legs_A)]
    keep_B = [i for i in 1:ndims(B) if !(i in legs_B)]

    @assert all(size(A, legs_A[k]) == size(B, legs_B[k]) for k in 1:M) "contracted leg size mismatch"

    # Permute A → (keep_A..., legs_A...)  and reshape to matrix (free × contr)
    Ap = permutedims(A, [keep_A; collect(legs_A)])
    sz_fA = [size(A, i) for i in keep_A]
    Amat  = reshape(Ap, prod(sz_fA; init=1), :)

    # Permute B → (legs_B..., keep_B...)  and reshape to matrix (contr × free)
    Bp = permutedims(B, [collect(legs_B); keep_B])
    sz_fB = [size(B, i) for i in keep_B]
    Bmat  = reshape(Bp, :, prod(sz_fB; init=1))

    # Contract via matrix multiply, then restore leg shapes
    return reshape(Amat * Bmat, sz_fA..., sz_fB...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Compute offset map from a splist (space list) for a given symmetry tuple.
# Returns (leg_offset::Dict{qlabel => UnitRange}, leg_total::Int)
function get_offset(symm::Tuple, splist::Vector)
    N = length(symm)
    leg_offset = Dict{Any, UnitRange{Int}}()
    sector_size = Dict{Any, Int}()
    
    for (qlabels, RMTd) in splist
        irep_dim = prod(
            isabelian(symm[n]) ? 1 :
            dimension(getNsave_irep(symm[n], BigInt, qlabels[n]))
            for n in 1:N)
        sz = irep_dim * RMTd
        sector_size[qlabels] = sz
    end

    # Assign offsets in sorted qlabel order.
    off = 1
    for ql in sort(collect(keys(sector_size)))
        sz = sector_size[ql]
        leg_offset[ql] = off:off+sz-1
        off += sz
    end
    leg_total = off - 1
    return (leg_offset, leg_total)
end


# ─── to_sparse_array ───────────────────────────────────────────────────────────
# Convert a TLArray to a sparse array using its spaces field for offset computation.
function to_sparse_array(q::TLArray{T, QD, N, RD},
    ::Type{FT}) where {T, QD, N, RD, FT}

    symmetries = symm(q)
    # ── Step 1: offset map ──────────────────────────────────────────────────
    # Use spaces field directly for each leg's offset computation.
    leg_info = [get_offset(symmetries, q.spaces[l]) for l in 1:QD]
    leg_offsets = [li[1] for li in leg_info]
    leg_total = [li[2] for li in leg_info]

    # ── Step 2: allocate output array ───────────────────────────────────────
    result = SparseArray(zeros(FT, leg_total...))

    # ── Step 3: accumulate each sector's contribution ────────────────────────
    for sector_index in Telum.sector_slots(q)
        q.iszero[sector_index] && continue
        # For each symmetry n, build the CGT contracted with its w-matrix.
        # After contracting, cgt_wmats[n] has shape
        #   (d_phys_1^(n), ..., d_phys_QD^(n), M_n)
        # where d_phys_l^(n) = irrep dim at physical leg l for symmetry n,
        # and M_n = size(wmat, 2) (compressed bond dimension after SVD).
        cgt_wmats = Vector{Array{FT}}(undef, N)
        for n in 1:N
            S   = symmetries[n]
            qlabels, cgp, legdir = Telum._sector_cgt_metadata(q, sector_index, n)
            wmat = Telum.sector_wmat(q, sector_index, n)
            M   = size(wmat, 2)

            if isabelian(S)
                # Abelian symmetry: irrep dim = 1 at every leg, outer multiplicity = 1.
                # CGT is trivially the scalar 1; the contracted result is just the
                # w-matrix sector reshaped to (1,...,1, M).
                @assert M == 1 "Unexpected bond dimension for abelian symmetry $n"
                cgt_wmats[n] = reshape(FT.(wmat), ones(Int, QD)..., M)
            else
                # a. Extract CGT qlabels from this CGR.
                nin, nout = legdir
                insp  = Tuple(qlabels[i] for i in 1:nin)
                outsp = Tuple(qlabels[i] for i in nin+1:QD)
                insp_, _ = remove_zeros(S, insp)
                outsp_, _ = remove_zeros(S, outsp)

                CGTom = get_CGTom(S, insp_, outsp_)
                om    = CGTom.totalOM
                @assert om == size(wmat, 1) "outer multiplicity mismatch for symmetry $n"

                canbasis = get_canonical_basis(S, insp, outsp, CGTom)

                # b. Allocate (CGT_shape..., om).
                cgt_shape = size(Array(canbasis[1]))   # (d_in_1,...,d_out_NO) in canonical order
                cgt_arr   = zeros(FT, cgt_shape..., om)

                # c. Fill each om-slice with the corresponding canonical basis element.
                for i in 1:om cgt_arr[fill(:, QD)..., i] .= Array(canbasis[i]) end

                # d. Contract last (om) leg of cgt_arr with first (om) leg of wmat.
                #    cgt_arr: (...cgt_shape..., om)  →  flatten to (prod_shape, om)
                #    wmat:    (om, M)
                #    result:  (prod_shape, M)  →  reshaped to (cgt_shape..., M)
                wmat_data   = wmat                                 # (om, M)
                cgt_flat    = reshape(cgt_arr, :, om)              # (prod(cgt_shape), om)
                result_flat = cgt_flat * wmat_data                 # (prod(cgt_shape), M)
                cgt_wmat_canon = reshape(result_flat, cgt_shape..., M)

                # Permute from canonical leg order to physical leg order using cgp.
                # cgp[l] = canonical axis that corresponds to physical leg l,
                # so permutedims with perm = (cgp..., QD+1) maps canonical → physical.
                perm = (cgp..., QD+1)
                cgt_wmats[n] = permutedims(cgt_wmat_canon, perm)
            end
        end

        # e. Kronecker product over symmetries to get the combined CGT block.
        #    cgt_wmats[n]: (d_1^(n), ..., d_QD^(n), M_n)
        #    cgt_block:    (irep_dim_1, ..., irep_dim_QD, M_total)
        cgt_block = cgt_wmats[1]
        for n in 2:N
            cgt_block = _leg_kron(cgt_block, cgt_wmats[n], QD)
        end
        # cgt_block has shape (irep_dim_1, ..., irep_dim_QD, M_total)
        # where irep_dim_l = prod_n d_l^(n)  and  M_total = prod_n M_n.
        chi = size(cgt_block, QD + 1)   # = M_total

        # ── Step 3a: Merge all OM (non-physical) legs of RMT ─────────────────
        # RMT shape: (s_1, ..., s_QD, M_1, ..., M_N)  →  (s_1, ..., s_QD, chi)
        # Julia column-major reshape: M_1 varies fastest, consistent with how
        # _leg_kron built up the bond dimension in cgt_block.
        rmt = Array(Telum.sector_rmt(q, sector_index))
        rmt_merged = reshape(rmt, size(rmt)[1:QD]..., chi)

        # ── Step 3b: Σ_i kron(CGT[:,...,:,i], RMT[:,...,:,i]) ────────────────
        d_phys = ntuple(l -> size(cgt_block,  l), QD)   # irep dims per leg
        s_phys = ntuple(l -> size(rmt_merged, l), QD)   # RMT free dims per leg
        block  = zeros(FT, ntuple(l -> d_phys[l] * s_phys[l], QD))
        col    = fill(:, QD)   # helper for slicing all physical legs
        for i in 1:chi
            block .+= _legwise_kron(cgt_block[col..., i], rmt_merged[col..., i])
        end

        # ── Step 3c: Scatter block into result at the correct qlabel offsets ─
        ranges = Tuple(leg_offsets[l][Telum.sector_qlabel(q, sector_index, l)] for l in 1:QD)
        result[ranges...] .+= block
    end

    return result
end

to_sparse_array(q::TLArray) = to_sparse_array(q, eltype(q))

to_sparse_array(q::TLArrayView, ::Type{FT} = Float64) where {FT} =
    to_sparse_array(Telum._eager_tlarray(q), FT)
to_sparse_array(q::TLArrayView) = to_sparse_array(q, eltype(q))

function _test_contract_matches_sparse_and_preserves_inputs(a::TLArray,
                                                            legs_a::Tuple,
                                                            b::TLArray,
                                                            legs_b::Tuple;
                                                            atol=1e-10,
                                                            rtol=1e-10)
    FT = promote_type(eltype(a), eltype(b), Float64)
    a_sparse = to_sparse_array(a, FT)
    b_sparse = to_sparse_array(b, FT)
    a_before = Array(a_sparse)
    b_before = Array(b_sparse)

    result = contract(a, legs_a, b, legs_b)
    result_sparse = Array(to_sparse_array(result, FT))
    reference = Array(contract_sparse(a_sparse, b_sparse, legs_a, legs_b))

    @test result_sparse ≈ reference atol=atol rtol=rtol
    @test Array(to_sparse_array(a, FT)) == a_before
    @test Array(to_sparse_array(b, FT)) == b_before
    return result
end

function test_contract_sparse_equivalence_diag_rmt()
    symm = (U1,)
    qlabels = [(((0,),), ((0,),))]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0))]
    spaces = ([(((0,),), 2)], [(((0,),), 2)])

    q1 = TLArray(symm, qlabels, wmatdata, wmatinfo,
                 [DiagRMT([1.0, 2.0], Val(3), (1, 2))],
                 (TLIndex("x", '+'), TLIndex("c", '-')), spaces)
    q2 = TLArray(symm, qlabels, wmatdata, wmatinfo,
                 [reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1)],
                 (TLIndex("c", '+'), TLIndex("y", '-')), spaces)

    @test Array(to_sparse_array(q1)) == [1.0 0.0; 0.0 2.0]
    _test_contract_matches_sparse_and_preserves_inputs(q1, (2,), q2, (1,))

    q2_left = TLArray(symm, qlabels, wmatdata, wmatinfo,
                      [reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1)],
                      (TLIndex("x", '+'), TLIndex("c", '-')), spaces)
    q1_right = TLArray(symm, qlabels, wmatdata, wmatinfo,
                       [DiagRMT([1.0, 2.0], Val(3), (1, 2))],
                       (TLIndex("c", '+'), TLIndex("y", '-')), spaces)
    _test_contract_matches_sparse_and_preserves_inputs(q2_left, (2,), q1_right, (1,))

    qdiag_left = TLArray(symm, qlabels, wmatdata, wmatinfo,
                         [DiagRMT([1.5, -2.0], Val(3), (1, 2))],
                         (TLIndex("x", '+'), TLIndex("c", '-')), spaces)
    qdiag_right = TLArray(symm, qlabels, wmatdata, wmatinfo,
                          [DiagRMT([0.5, 4.0], Val(3), (1, 2))],
                          (TLIndex("c", '+'), TLIndex("y", '-')), spaces)
    _test_contract_matches_sparse_and_preserves_inputs(qdiag_left, (2,), qdiag_right, (1,))

    qlabels3 = [(((0,),), ((0,),), ((0,),))]
    spaces3 = ([(((0,),), 2)], [(((0,),), 2)], [(((0,),), 1)])
    q3 = TLArray(symm, qlabels3, wmatdata, wmatinfo,
                 [DiagRMT([1.0, 2.0], Val(4), (1, 2))],
                 (TLIndex("x2", '+'), TLIndex("y2", '+'), TLIndex("c2", '-')),
                 spaces3)
    q4 = TLArray(symm, [(((0,),),)], wmatdata, wmatinfo,
                 [reshape([1.0], 1, 1)],
                 (TLIndex("c2", '+'),), ([(((0,),), 1)],))

    _test_contract_matches_sparse_and_preserves_inputs(q3, (3,), q4, (1,))

    complex_left = TLArray(symm, qlabels, wmatdata, wmatinfo,
                           [reshape(ComplexF64[1 + 2im, 3 - im, -2 + 0.5im, 4im], 2, 2, 1)],
                           (TLIndex("cx", '+'), TLIndex("cc", '-')), spaces)
    complex_right = TLArray(symm, qlabels, wmatdata, wmatinfo,
                            [reshape(ComplexF64[0.5 - im, -1 + 3im, 2 + im, 1 - 0.25im], 2, 2, 1)],
                            (TLIndex("cc", '+'), TLIndex("cy", '-')), spaces)
    _test_contract_matches_sparse_and_preserves_inputs(complex_left, (2,), complex_right, (1,))

    real_right = TLArray(symm, qlabels, wmatdata, wmatinfo,
                         [reshape([2.0, -1.0, 0.5, 3.0], 2, 2, 1)],
                         (TLIndex("cc", '+'), TLIndex("ry", '-')), spaces)
    _test_contract_matches_sparse_and_preserves_inputs(complex_left, (2,), real_right, (1,))

    complex_diag_left = TLArray(symm, qlabels, wmatdata, wmatinfo,
                                [DiagRMT(ComplexF64[1 + im, 2 - im], Val(3), (1, 2))],
                                (TLIndex("dx", '+'), TLIndex("dc", '-')), spaces)
    real_dense_right = TLArray(symm, qlabels, wmatdata, wmatinfo,
                               [reshape([2.0, -1.0, 0.5, 3.0], 2, 2, 1)],
                               (TLIndex("dc", '+'), TLIndex("dy", '-')), spaces)
    _test_contract_matches_sparse_and_preserves_inputs(complex_diag_left, (2,), real_dense_right, (1,))
end

function test_sum_sparse_equivalence_mixed_rmt_eltypes()
    symm = (U1,)
    qlabels = [(((0,),), ((0,),))]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0))]
    spaces = ([(((0,),), 2)], [(((0,),), 2)])
    inds = (TLIndex("x", '+'), TLIndex("y", '-'))

    real_dense = TLArray(symm, qlabels, wmatdata, wmatinfo,
                         [reshape([2.0, -1.0, 0.5, 3.0], 2, 2, 1)],
                         inds, spaces)
    complex_dense = TLArray(symm, qlabels, wmatdata, wmatinfo,
                            [reshape(ComplexF64[1 + 2im, 3 - im, -2 + 0.5im, 4im], 2, 2, 1)],
                            inds, spaces)
    complex_diag = TLArray(symm, qlabels, wmatdata, wmatinfo,
                           [DiagRMT(ComplexF64[1 + im, 2 - im], Val(3), (1, 2))],
                           inds, spaces)

    # Exercise both promotion order and dense/diagonal RMT mixing; the sparse
    # reference keeps the check independent of the internal storage chosen.
    for (left, right) in ((real_dense, complex_dense),
                          (complex_dense, real_dense),
                          (complex_diag, real_dense),
                          (real_dense, complex_diag),)
        result = left + right
        reference = Array(to_sparse_array(left, ComplexF64)) + Array(to_sparse_array(right, ComplexF64))
        @test Array(to_sparse_array(result, ComplexF64)) ≈ reference
        @test eltype(result) == ComplexF64
    end

    complex_view = permutedims(complex_dense, (2, 1))
    scaled_complex_view = (1.5 - 0.5im) * complex_dense
    @test complex_view isa TLArrayView
    @test scaled_complex_view isa TLArrayView

    result_tuple = sum((real_dense, complex_view, scaled_complex_view, complex_diag))
    reference_tuple = Array(to_sparse_array(real_dense, ComplexF64)) +
                      Array(to_sparse_array(complex_dense, ComplexF64)) +
                      (1.5 - 0.5im) .* Array(to_sparse_array(complex_dense, ComplexF64)) +
                      Array(to_sparse_array(complex_diag, ComplexF64))
    @test Array(to_sparse_array(result_tuple, ComplexF64)) ≈ reference_tuple
    @test eltype(result_tuple) == ComplexF64

    result_vector = sum(AbstractTLArray[complex_diag, complex_view, 3 * real_dense])
    reference_vector = Array(to_sparse_array(complex_diag, ComplexF64)) +
                       Array(to_sparse_array(complex_dense, ComplexF64)) +
                       3 .* Array(to_sparse_array(real_dense, ComplexF64))
    @test Array(to_sparse_array(result_vector, ComplexF64)) ≈ reference_vector
    @test eltype(result_vector) == ComplexF64

    q0 = ((0,),)
    q1 = ((1,),)
    q2 = ((2,),)
    spaces_multi = ([(q0, 2), (q1, 2), (q2, 2)],
                    [(q0, 2), (q1, 2), (q2, 2)])
    real_multi = TLArray(symm, [(q0, q0), (q1, q1)], wmatdata,
                         [Telum._empty_wmat_info(Val(0)), Telum._empty_wmat_info(Val(0))],
                         [reshape([1.0, 2.0, -1.0, 0.5], 2, 2, 1),
                          reshape([0.25, -1.5, 2.5, 1.0], 2, 2, 1)],
                         inds, spaces_multi)
    diag_multi = TLArray(symm, [(q1, q1), (q2, q2)], wmatdata,
                         [Telum._empty_wmat_info(Val(0)), Telum._empty_wmat_info(Val(0))],
                         [DiagRMT(ComplexF64[1 + im, -2 + 0.5im], Val(3), (1, 2)),
                          DiagRMT(ComplexF64[0.25 - im, 3 + 2im], Val(3), (1, 2))],
                         inds, spaces_multi)

    result_multi_sector = sum((real_multi, diag_multi))
    reference_multi_sector = Array(to_sparse_array(real_multi, ComplexF64)) +
                             Array(to_sparse_array(diag_multi, ComplexF64))
    @test Array(to_sparse_array(result_multi_sector, ComplexF64)) ≈ reference_multi_sector
    @test eltype(result_multi_sector) == ComplexF64
    @test all(rmt -> rmt isa Array{ComplexF64, 3}, result_multi_sector.RMTs)
    @test Set(result_multi_sector.qlabels) == Set([(q0, q0), (q1, q1), (q2, q2)])
end

function test_contract_tlarrayview_inputs()
    @testset "contract TLArrayView inputs" begin
        symm = (U1,)
        qlabels = [(((0,),), ((0,),))]
        wmatdata = Float64[]
        wmatinfo = [Telum._empty_wmat_info(Val(0))]
        spaces = ([(((0,),), 2)], [(((0,),), 2)])

        left = TLArray(symm, qlabels, wmatdata, wmatinfo,
                       [reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1)],
                       (TLIndex("x", '+'), TLIndex("c", '-')), spaces)
        right = TLArray(symm, qlabels, wmatdata, wmatinfo,
                        [reshape(ComplexF64[0.5 - im, -1 + 3im, 2 + im, 1 - 0.25im], 2, 2, 1)],
                        (TLIndex("c", '+'), TLIndex("y", '-')), spaces)

        left_view = 2.0 * left
        right_view = (1.5 - 0.5im) * right
        @test left_view isa TLArrayView
        @test right_view isa TLArrayView

        # Compare view*dense and view*view contraction against the sparse-array
        # reference so scalar view materialization and eltype promotion are both covered.
        for (a, b) in ((left_view, right), (left_view, right_view))
            result = a * b
            reference = Array(contract_sparse(to_sparse_array(a, ComplexF64),
                                              to_sparse_array(b, ComplexF64),
                                              (2,), (1,)))
            @test Array(to_sparse_array(result, ComplexF64)) ≈ reference atol=1e-10 rtol=1e-10
        end
    end
end

function test_FAcont(option::LocalSpaceOptions)
    q = getLocalSpace(option);
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    qf = TLArray(q.F, ("lur2", "lur2", "op"))
    a = getIdentity((qi1, 2), (qi2, 2));

    ct = _test_contract_matches_sparse_and_preserves_inputs(qf, (2,), a, (2,))

    return q, a, ct
end

function _test_tlarrays_same_sector_storage(a::TLArray, b::TLArray)
    @test a.qlabels == b.qlabels
    @test a.iszero == b.iszero
    @test a.isdefined == b.isdefined
    @test a.spaces == b.spaces
    for sector_index in intersect(Telum.sector_slots(a), Telum.sector_slots(b))
        (a.iszero[sector_index] || b.iszero[sector_index]) && continue
        @test Array(Telum.sector_rmt(a, sector_index)) ≈ Array(Telum.sector_rmt(b, sector_index))
        for n in 1:length(symm(a))
            @test Telum.sector_wmat(a, sector_index, n) ≈ Telum.sector_wmat(b, sector_index, n)
        end
    end
end

function _test_tlarrays_same_sector_storage(a::AbstractTLArray, b::AbstractTLArray)
    return _test_tlarrays_same_sector_storage(copy(a), copy(b))
end

function _test_tlarrays_same_sector_payloads(a::TLArray, b::TLArray)
    @test a.qlabels == b.qlabels
    @test a.iszero == b.iszero
    @test a.isdefined == b.isdefined
    for sector_index in intersect(Telum.sector_slots(a), Telum.sector_slots(b))
        (a.iszero[sector_index] || b.iszero[sector_index]) && continue
        @test Array(Telum.sector_rmt(a, sector_index)) ≈ Array(Telum.sector_rmt(b, sector_index))
        for n in 1:length(symm(a))
            @test Telum.sector_wmat(a, sector_index, n) ≈ Telum.sector_wmat(b, sector_index, n)
        end
    end
end

function _test_tlarrays_same_sector_payloads(a::AbstractTLArray, b::AbstractTLArray)
    return _test_tlarrays_same_sector_payloads(copy(a), copy(b))
end

function test_getIdentity_direct_contract(option::LocalSpaceOptions)
    q = getLocalSpace(option)
    qi = TLArray(q.I, ("lur", "lur"))
    a_pairs = getIdentity((qi, 1); itag="fused")
    a = getIdentity(qi, 1; itag="fused")

    @test a.inds == a_pairs.inds
    @test a.spaces == a_pairs.spaces
    _test_tlarrays_same_sector_storage(a, a_pairs)
    @test a.inds[1] == TLIndex(qi.inds[1].itags, '-', qi.inds[1].plev, qi.inds[1].lock, qi.inds[1].dual)
    @test a.spaces[1] == qi.spaces[1]
    @test a.inds[2] == TLIndex("fused", '-')

    ct = qi * a
    @test length(ct.inds) == 2
    @test ct.inds[1] == qi.inds[2]
    @test ct.inds[2] == TLIndex("fused", '-')
    @test ct.spaces[1] == qi.spaces[2]
    @test !isempty(_test_defined_sector_indices(ct))
end

function test_1jpair(option::LocalSpaceOptions)
    q = getLocalSpace(option);
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))

    q1 = get1jtensor(qi1, 2)
    q2 = permutedims(q1', (2, 1))
    mult = contract(q1, 2, q2, 1)
    arr1 = to_sparse_array(q.I)
    arr2 = to_sparse_array(mult)

    println(norm(arr1 - arr2))
    @test norm(arr1 - arr2) < 1e-10

    a = getIdentity((qi1, 2), (qi2, 2))
    q1 = get1jtensor(a, 3)
    q2 = permutedims(q1', (2, 1))
    mult = contract(q1, 2, q2, 1)

    arr1 = to_sparse_array(mult)
    sz = size(arr1)[1]
    arr2 = SparseArray(Matrix(I, sz, sz))

    println(norm(arr1 - arr2))
    @test norm(arr1 - arr2) < 1e-10

end

function test_get1jtensor_and_legflip_keywords(option::LocalSpaceOptions)
    q0 = getLocalSpace(option, ("site1", "site2", "op"))
    q = q0.F

    function test_same_tlarray_structure(q1::AbstractTLArray, q2::AbstractTLArray)
        @test q1.inds == q2.inds
        @test q1.spaces == q2.spaces
        _test_tlarrays_same_sector_storage(q1, q2)
    end

    j_explicit = get1jtensor(q, 2)
    j_kw = get1jtensor(q; itag="site2")
    test_same_tlarray_structure(j_kw, j_explicit)

    @test_throws ArgumentError get1jtensor(q; plev=0)
    @test_throws ArgumentError get1jtensor(q; itag="missing")

    flipped_explicit = legflip(q, 2)
    flipped_kw = legflip(q; itag="site2")
    test_same_tlarray_structure(flipped_kw, flipped_explicit)

    @test flipped_explicit.inds[1] == q.inds[1]
    @test flipped_explicit.inds[2] == change_dir(Telum.change_dual(q.inds[2]))
    @test flipped_explicit.inds[3] == q.inds[3]

    roundtrip = legflip(flipped_explicit, 2)
    @test roundtrip.inds == q.inds
    @test roundtrip.spaces == q.spaces

    flipped_multi_tuple = legflip(q, (2, 3))
    flipped_multi_vector = legflip(q, [2, 3])
    flipped_multi_kw = legflip(q; dir='-')

    test_same_tlarray_structure(flipped_multi_vector, flipped_multi_tuple)
    test_same_tlarray_structure(flipped_multi_kw, flipped_multi_tuple)

    @test flipped_multi_tuple.inds[1] == q.inds[1]
    @test flipped_multi_tuple.inds[2] == change_dir(Telum.change_dual(q.inds[2]))
    @test flipped_multi_tuple.inds[3] == change_dir(Telum.change_dual(q.inds[3]))

    roundtrip_multi = legflip(flipped_multi_tuple, (2, 3))
    @test roundtrip_multi.inds == q.inds
    @test roundtrip_multi.spaces == q.spaces

    @test_throws ArgumentError legflip(q, Int[])
    @test_throws ArgumentError legflip(q, (2, 2))
    @test_throws ArgumentError legflip(q; itag="missing")
end

function test_contract_verify_legs_checks_dual(option::LocalSpaceOptions)
    q0 = getLocalSpace(option, ("site1", "site2", "op"))
    q = q0.F
    j = get1jtensor(q, 2)

    @test contract(q, (2,), j, (1,); reduce_lock=false) isa TLArray
    @test_throws AssertionError contract(q, (2,), j, (2,); reduce_lock=false)
    @test contract(q, (2,), j, (2,); reduce_lock=false, verify_legs=false) isa TLArray
end

# ─── test_conj ───────────────────────────────────────────────────────────────
# Invariant: conj(q) represented as a dense/sparse array equals the
# elementwise complex conjugate of q's sparse array.
#
# Tested TLArray objects (drawn from existing test helpers):
#   1. q.I       — 2-leg identity operator
#   2. q.F       — 3-leg creation/annihilation operator
#   3. a         — 4-leg identity from getIdentity (tensor product of two I legs)
#   4. ct        — 4-leg contraction result F ⊗ a from test_FAcont
#
# For each, we verify:
#   norm( to_sparse_array(qs)  −  conj(to_sparse_array(conj(qs))) )  < tol
#
# Physical qlabels do not change under conjugation (only stored CGR ordering
# flips), so both sparse arrays have the same shape and qlabel offsets.
# ─────────────────────────────────────────────────────────────────────────────
function test_conj(option::LocalSpaceOptions)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2))
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    # Pairs (label, TLArray) to test
    cases = [
        ("I",  q.I),
        ("F",  q.F),
        ("a",  a),
        ("ct", ct),
    ]

    zero_qlabels = copy(q.F.qlabels)
    push!(zero_qlabels, first(q.F.qlabels))
    zero_wmatdata = copy(q.F.wmatdata)
    zero_wmatinfo = copy(q.F.wmatinfo)
    push!(zero_wmatinfo, Telum._empty_wmat_info(Val(length(q.F.wmatinfo[1]))))
    zero_RMTs = similar(q.F.RMTs, length(q.F.RMTs) + 1)
    for sector_index in Telum.sector_slots(q.F)
        q.F.iszero[sector_index] && continue
        zero_RMTs[sector_index] = q.F.RMTs[sector_index]
    end
    q_with_zero_sector = TLArray(symm(q.F), zero_qlabels, zero_wmatdata, zero_wmatinfo, zero_RMTs,
                                 q.F.inds, q.F.spaces)
    qzc = conj(q_with_zero_sector)
    @test qzc.iszero[end]
    @test !isassigned(qzc.RMTs, length(qzc.RMTs))
    @test qzc.wmatinfo[end] == Telum._empty_wmat_info(Val(length(qzc.wmatinfo[end])))

    for (label, qs) in cases
        qc   = conj(qs)

        # Convert both to plain dense arrays for comparison.
        arr  = Array(to_sparse_array(qs))
        arrc = Array(to_sparse_array(qc))

        # conj(q) as a sparse array should equal conj(q as sparse array).
        diff = norm(arr .- conj.(arrc))
        println("test_conj [$label]: ‖arr − conj(arrc)‖ = $diff")
        @test diff < 1e-10
    end
end

# ─── test_svdQS ────────────────────────────────────────────────────────────────
# Test SVD by reconstructing the original TLArray as U*S*Vd.
# ─────────────────────────────────────────────────────────────────────────────
function test_svdQS(q::TLArray{T, QD, N, RD},
                    left_legs;
                    cutoff::Float64 = 1e-12,
                    tol::Float64 = 1e-9,
                    verbose::Bool = true) where {T, QD, N, RD}
    result = svd(q, left_legs; cutoff=cutoff)
    U, S, Vd = result.U, result.S, result.Vd
    rec = U * S * Vd

    diff_norm = norm(q - rec)
    Uiso_work = U
    for leg in 1:(ndims(U) - 1)
        Uiso_work = additag(Uiso_work, leg, "__test_svdQS_U_$leg")
    end
    Vdiso_work = Vd
    for leg in 2:ndims(Vd)
        Vdiso_work = additag(Vdiso_work, leg, "__test_svdQS_Vd_$leg")
    end
    Uiso = Uiso_work' * lock(Uiso_work, ndims(Uiso_work))
    Vdiso = lock(Vdiso_work, 1) * Vdiso_work'
    Uiso_diff = norm(Uiso - 1)
    Vdiso_diff = norm(Vdiso - 1)

    if verbose
        println("test_svdQS: ‖orig - U*S*Vd‖ = $diff_norm")
        println("test_svdQS: ‖U' * lock(U, bond) - I‖ = $Uiso_diff")
        println("test_svdQS: ‖lock(Vd, bond) * Vd' - I‖ = $Vdiso_diff")
    end

    @assert diff_norm < tol "SVD reconstruction error ‖orig - rec‖ = $diff_norm exceeds tolerance $tol"
    @assert Uiso_diff < tol "SVD U isometry error ‖U' * lock(U, bond) - I‖ = $Uiso_diff exceeds tolerance $tol"
    @assert Vdiso_diff < tol "SVD Vd isometry error ‖lock(Vd, bond) * Vd' - I‖ = $Vdiso_diff exceeds tolerance $tol"

    return diff_norm
end

function _test_svd_diag_rmt_case(q::TLArray; expected_singular_values=nothing)
    @test all(i -> q.iszero[i] || q.RMTs[i] isa DiagRMT, eachindex(q.RMTs))
    diff = test_svdQS(q, (1,); cutoff=0.0, verbose=false)
    @test diff < 1e-10

    result = svd(q, (1,); cutoff=0.0, get_lists=true)
    @test all(i -> result.S.iszero[i] || result.S.RMTs[i] isa DiagRMT, eachindex(result.S.RMTs))

    telum_vals = Float64[]
    # kept_list stores one value per symmetry-degenerate multiplet; expand it
    # before comparing with the flat dense singular-value list.
    for (singular_value, degeneracy, _, _) in result.kept_list
        append!(telum_vals, fill(singular_value, degeneracy))
    end
    sort!(telum_vals; rev=true)
    reference_vals =
        if isnothing(expected_singular_values)
            sort(LinearAlgebra.svdvals(Array(to_sparse_array(q))); rev=true)
        else
            sort(collect(Float64, expected_singular_values); rev=true)
        end
    @test telum_vals ≈ reference_vals atol=1e-10 rtol=1e-10
end

function test_svd_diag_rmt()
    @testset "svd accepts DiagRMT sector storage" begin
        @testset "U1" begin
            symm = (U1,)
            qlabels = [(((0,),), ((0,),))]
            wmatdata = Float64[]
            wmatinfo = [Telum._empty_wmat_info(Val(0))]
            spaces = ([(((0,),), 2)], [(((0,),), 2)])
            q = TLArray(symm, qlabels, wmatdata, wmatinfo,
                        [DiagRMT([3.0, -1.5], Val(3), (1, 2))],
                        (TLIndex("left", '+'), TLIndex("right", '-')), spaces)

            _test_svd_diag_rmt_case(q)
        end

        @testset "SU2" begin
            # get1jtensor builds realistic non-Abelian DiagRMT storage; the
            # spin-1/2 identity contributes two degenerate singular values.
            q = get1jtensor(getLocalSpace(SpinOptions(:SU2, 1)).I, 1)
            _test_svd_diag_rmt_case(q; expected_singular_values=[1.0, 1.0])
        end
    end
end

function _test_multi_sector_matrix_tlarray(matrices::AbstractVector{<:AbstractMatrix{T}};
                                           qlabels=[((i - 1,),) for i in eachindex(matrices)],
                                           tags=("phys", "phys")) where {T}
    @assert length(matrices) == length(qlabels)
    symmetries = (U1,)
    # Each input matrix is stored as one symmetry sector, giving a block-diagonal
    # TLArray whose dense reference has the same per-sector eigenspectrum.
    sector_qlabels = [(qlabel, qlabel) for qlabel in qlabels]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0)) for _ in eachindex(matrices)]
    RMTs = Array{T, 3}[]
    spaces = Tuple{eltype(qlabels), Int}[]
    for (qlabel, matrix) in zip(qlabels, matrices)
        n = size(matrix, 1)
        @assert size(matrix, 2) == n
        push!(RMTs, reshape(Matrix{T}(matrix), n, n, 1))
        push!(spaces, (qlabel, n))
    end
    inds = (TLIndex(tags[1], '+'), TLIndex(tags[2], '-'))
    return TLArray(symmetries, sector_qlabels, wmatdata, wmatinfo, RMTs, inds, (spaces, copy(spaces)))
end

function _test_eigen_dense_matrix(q::TLArray)
    arr = Array(to_sparse_array(q))
    return reshape(arr, size(arr, 1), size(arr, 2))
end

function _test_eigen_expanded_eigenvalues(result)
    vals = mapreduce(entry -> fill(entry[1], entry[2]), append!, result.eig_list;
                     init=eltype(first(result.eig_list)[1])[])
    return sort(vals; by=abs)
end

function _test_eigen_dense_eigenvalues(q::TLArray; hermitian::Bool=false)
    mat = _test_eigen_dense_matrix(q)
    vals = hermitian ? LinearAlgebra.eigvals(Hermitian(mat)) : LinearAlgebra.eigvals(mat)
    return sort(vals; by=abs)
end

function _test_eigen_assert_reconstructs(q::TLArray; hermitian::Bool=false, tol=1e-9)
    result = eigen(q; hermitian)
    # Lock the original matrix leg so multiplication contracts only the eigen bond.
    V = lock(result.V, 1)
    rec = isnothing(result.V_inv) ? V * result.D * result.V' : V * result.D * result.V_inv
    @test norm(q - rec) / max(norm(q), 1.0) < tol
    return result
end

function test_eigen_multi_sector()
    @testset "multi-sector eigenvalues match dense eigendecomposition" begin
        # 5x5 and 10x10 blocks exercise the per-sector eigensolver beyond
        # scalar/small fixtures while keeping the test cheap.
        hermitian_blocks = [
            Matrix(Symmetric(randn(5, 5))),
            Matrix(Symmetric(randn(10, 10))),
        ]
        hermitian_q = _test_multi_sector_matrix_tlarray(hermitian_blocks)
        hermitian_result = _test_eigen_assert_reconstructs(hermitian_q; hermitian=true)
        @test _test_eigen_expanded_eigenvalues(hermitian_result) ≈
              _test_eigen_dense_eigenvalues(hermitian_q; hermitian=true) atol=1e-10 rtol=1e-10

        complex_blocks = [
            randn(ComplexF64, 5, 5),
            randn(ComplexF64, 10, 10),
        ]
        complex_blocks[1][1, 2] += 1.0 - 0.5im
        complex_blocks[1][2, 1] -= 0.25 + 1.0im
        complex_blocks[2][1, 2] += 2.0 - 0.5im
        complex_blocks[2][2, 1] -= 1.0 + 1.5im
        complex_q = _test_multi_sector_matrix_tlarray(complex_blocks)
        complex_result = _test_eigen_assert_reconstructs(complex_q; tol=1e-8)
        @test !isnothing(complex_result.V_inv)
        @test _test_eigen_expanded_eigenvalues(complex_result) ≈
              _test_eigen_dense_eigenvalues(complex_q) atol=1e-8 rtol=1e-8
    end
end

function _test_eigen_reconstructs_view(q::AbstractTLArray; hermitian::Bool=false, tol=1e-9)
    result = eigen(q; hermitian)
    # TLArrayView inputs should reconstruct the represented tensor, not only the
    # eager TLArray produced internally by eigen.
    V = lock(result.V, 1)
    rec = isnothing(result.V_inv) ? V * result.D * result.V' : V * result.D * result.V_inv
    @test norm(q - rec) / max(norm(q), 1.0) < tol
    return result
end

function test_eigen_tlarrayview()
    @testset "eigen accepts TLArrayView input" begin
        # Nonzero scalar multiplication returns TLArrayView for these operands,
        # covering both Hermitian and non-Hermitian eigen paths on lazy inputs.
        hermitian_blocks = [
            Matrix(Symmetric(randn(5, 5))),
            Matrix(Symmetric(randn(10, 10))),
        ]
        hermitian_q = _test_multi_sector_matrix_tlarray(hermitian_blocks)
        hermitian_view = 2.0 * hermitian_q
        @test hermitian_view isa TLArrayView
        hermitian_result = _test_eigen_reconstructs_view(hermitian_view; hermitian=true)
        @test isnothing(hermitian_result.V_inv)

        complex_blocks = [
            randn(ComplexF64, 5, 5),
            randn(ComplexF64, 10, 10),
        ]
        complex_blocks[1][1, 2] += 1.0 - 0.5im
        complex_blocks[1][2, 1] -= 0.25 + 1.0im
        complex_blocks[2][1, 2] += 2.0 - 0.5im
        complex_blocks[2][2, 1] -= 1.0 + 1.5im
        complex_q = _test_multi_sector_matrix_tlarray(complex_blocks)
        complex_view = (1.5 - 0.25im) * complex_q
        @test complex_view isa TLArrayView
        complex_result = _test_eigen_reconstructs_view(complex_view; tol=1e-8)
        @test !isnothing(complex_result.V_inv)
    end
end

# ─── test_norm ───────────────────────────────────────────────────────────────
# Verify norm(q) against two independent reference values:
#
#   (a) norm(to_sparse_array(q))  – dense/sparse Frobenius norm
#   (b) sqrt(abs((q * q')[]))    – scalar obtained by contracting q with conj(q)
#                                   over all legs (Wigner-Eckart inner product)
#
# Cases tested per call:
#   q.I  – rank-2 identity (QD = 2, exercises the dim·‖RMT‖² formula)
#   q.F  – rank-3 operator (QD = 3, standard CGT-orthonormal formula)
#   ct   – rank-4 tensor   (QD = 4, after F * identity contraction)
#
# For (b) to work every pair (q leg l, conj(q) leg l) must form a valid
# contraction pair: same itags, opposite directions.  conj() flips all
# directions, so the pairing is always leg-for-leg given that all legs of q
# already carry distinct TLIndex values (full-rank contraction → 0D scalar).
# ─────────────────────────────────────────────────────────────────────────────
function test_norm(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a   # rank-4: legs (lur1_in, lur2_in, lur2_out, op)

    cases = [
        ("I",  q.I),
        ("F",  q.F),
        ("ct", ct),
    ]

    for (label, qs) in cases
        # ── (a) compare with sparse array norm ─────────────────────────────
        norm_qs     = norm(qs)
        norm_sparse = norm(Array(to_sparse_array(qs)))
        diff_a = abs(norm_qs - norm_sparse)
        println("test_norm [$label]: norm_qs=$norm_qs  norm_sparse=$norm_sparse  Δ=$diff_a")
        @test diff_a < tol

        # ── (b) compare norm² with scalar from qs * qs' ────────────────────
        #   qs' = conj(qs) flips all leg directions; contraction over all
        #   matching legs yields a 0D TLArray whose single element is ‖qs‖².
        scalar_qs     = qs * qs'
        println(qs)
        norm_sq_contr = abs(scalar_qs[])
        diff_b = abs(norm_qs^2 - norm_sq_contr)
        ref    = max(norm_sq_contr, 1.0)
        println("test_norm [$label]: ‖qs‖²=$(norm_qs^2)  (qs·qs')=$norm_sq_contr  Δ=$diff_b  (rel=$(diff_b/ref))")
        @test diff_b < tol * ref
    end
end

# ─── sector helpers ─────────────────────────────────────────────────────────
_test_defined_sector_indices(q::TLArray) =
    [sector_index for sector_index in Telum.sector_slots(q) if !q.iszero[sector_index]]

# ─── test_lock_reduce ─────────────────────────────────────────────────────────
# Verify selective lock-level reduction in contraction:
#
# Rule: a free output leg has its lock decremented by 1 only when the *other*
# tensor in the contraction has at least one leg with the same itag.
#
#   Scenario 1 — match present   : A free leg "free" lock=1 + B has "free" → lock → 0
#   Scenario 2 — no match        : A free leg "unique" lock=1, B has no "unique" → lock stays 1
#   Scenario 3 — lock already 0  : match present but lock=0 → lock stays 0
# ─────────────────────────────────────────────────────────────────────────────
function test_lock_reduce(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    # ── Scenario 1: matching leg present → lock decrements ───────────────────
    # q.I has legs ('+', '-').  A1 is built from q.I (same dirs), B1 from q.I'
    # (dirs flipped to ('-', '+')) so they can be contracted on "ct".
    # A1: ("ct" '+' lock=0,  "free" '-' lock=1)
    # B1: ("ct" '-' lock=0,  "free" '+' lock=0)
    # Contract A1*B1 on "ct"; A1's "free" lock=1 has a match in B1 → lock → 0.
    A1 = TLArray(q.I,
        (TLIndex("ct",   '+', 0, 0),
         TLIndex("free", '-', 0, 1)))
    B1 = TLArray(q.I',
        (TLIndex("ct",   '-', 0, 0),
         TLIndex("free", '+', 0, 0)))

    C1 = A1 * B1
    free_pos1 = findfirst(idx -> idx.itags == "free" && idx.dir == '-', C1.inds)
    @test C1.inds[free_pos1].lock == 0
    println("test_lock_reduce (match):    \"free\" lock=$(C1.inds[free_pos1].lock) (expected 0)")

    # ── Scenario 2: no matching leg → lock unchanged ─────────────────────────
    # A2: ("ct" '+' lock=0,  "unique" '-' lock=1)   — from q.I
    # B2: ("ct" '-' lock=0,  "other"  '+' lock=0)   — from q.I'; "unique" absent
    A2 = TLArray(q.I,
        (TLIndex("ct",     '+', 0, 0),
         TLIndex("unique", '-', 0, 1)))
    B2 = TLArray(q.I',
        (TLIndex("ct",    '-', 0, 0),
         TLIndex("other", '+', 0, 0)))

    C2 = A2 * B2
    unique_pos = findfirst(idx -> idx.itags == "unique", C2.inds)
    @test C2.inds[unique_pos].lock == 1
    println("test_lock_reduce (no match): \"unique\" lock=$(C2.inds[unique_pos].lock) (expected 1)")

    # ── Scenario 3: lock already 0 with match → stays 0 ─────────────────────
    # A3's "free" lock=0 and B1 has a matching "free" leg, but 0 cannot go lower.
    # Must use explicit contract (not *) to avoid auto-contracting both "ct" and "free".
    A3 = TLArray(q.I,
        (TLIndex("ct",   '+', 0, 0),
         TLIndex("free", '-', 0, 0)))
    C3 = contract(A3, (1,), B1, (1,))
    free_pos3 = findfirst(idx -> idx.itags == "free" && idx.dir == '-', C3.inds)
    @test C3.inds[free_pos3].lock == 0
    println("test_lock_reduce (lock=0):   \"free\" lock=$(C3.inds[free_pos3].lock) (expected 0)")
end

# ─── test_contract_requires_matching_spaces_in_star ─────────────────────────
# Verify that automatic contraction via `*` does not match tagged legs when
# their precomputed space metadata differs, even if the TLIndex fields match.
function test_contract_requires_matching_spaces_in_star(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    A = TLArray(q.I,
        (TLIndex("ct", '+', 0, 0),
         TLIndex("",   '-', 0, 0)))
    B = TLArray(q.I',
        (TLIndex("ct", '-', 0, 0),
         TLIndex("",   '+', 0, 0)))

    first_sector, first_dim = first(B.spaces[1])
    bad_leg1 = vcat([(first_sector, first_dim + 1)], B.spaces[1][2:end])
    B_bad = Telum.TLArray(symm(B), copy(B.qlabels), copy(B.wmatdata), copy(B.wmatinfo),
                                 deepcopy(B.RMTs), B.inds, (bad_leg1, B.spaces[2]))

    @test_throws AssertionError A * B_bad
end
