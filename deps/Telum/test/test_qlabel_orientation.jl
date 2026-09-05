function _assert_qlabel_storage(q::TLArray)
    @test eltype(q.qlabels) <: NTuple{ndims(q), Telum.qlabeltype(q)}
    @test length(q.qlabels) == Telum.sector_count(q)
    for sector_index in Telum.sector_slots(q)
        @test q.qlabels[sector_index] ==
              ntuple(leg -> Telum.sector_qlabel(q, sector_index, leg), ndims(q))
    end
end
_assert_qlabel_storage(q::TLArrayView) = _assert_qlabel_storage(copy(q))

function _assert_wmat_storage(q::TLArray)
    nonabelian = Telum.nonabelian_symmetry_indices(Telum.productsymm(q))
    @test length(q.wmatinfo) == Telum.sector_count(q)
    @test eltype(q.wmatinfo) <: NTuple{length(nonabelian), NTuple{3, Int}}
    for sector_index in Telum.sector_slots(q), n in 1:length(symm(q))
        q.iszero[sector_index] && continue
        wmat = Telum.sector_wmat(q, sector_index, n)
        if n in nonabelian
            slot = Telum.nonabelian_wmat_slot(Telum.productsymm(q), n)
            @test wmat == Telum.sector_wmat_slot(q, sector_index, slot)
        else
            @test size(wmat) == (1, 1)
            @test wmat[1] == 1.0
        end
    end
end

function _append_sector_wmat_info!(wmatdata, wmatinfo, q::TLArray, sector_index::Int)
    M = length(q.wmatinfo[sector_index])
    info = ntuple(Val(M)) do slot
        offset0, nrow, ncol = q.wmatinfo[sector_index][slot]
        offset0 == 0 && return (0, 0, 0)
        wmat = Telum.sector_wmat_slot(q, sector_index, slot)
        offset = length(wmatdata) + 1
        append!(wmatdata, vec(wmat))
        (offset, nrow, ncol)
    end
    push!(wmatinfo, info)
    return wmatdata, wmatinfo
end

function _assert_metadata_inferred(q::TLArray)
    Telum.nsectors(q) == 0 && return
    @inferred Telum.sector_qlabel(q, 1, 1)
    @test Telum.sector_wmat(q, 1, 1) isa AbstractMatrix{Float64}
    @inferred Telum._sector_cgt_metadata(q, 1, 1)
end

function _assert_zero_sector_constructor_state(q::TLArray)
    Telum.nsectors(q) == 0 && return

    old_n = Telum.nsectors(q)
    qlabels = copy(q.qlabels)
    push!(qlabels, q.qlabels[1])

    wmatdata = copy(q.wmatdata)
    wmatinfo = copy(q.wmatinfo)
    push!(wmatinfo, Telum._empty_wmat_info(Val(length(q.wmatinfo[1]))))
    RMTs = Vector{eltype(q.RMTs)}(undef, Telum.sector_count(q) + 1)
    for sector_index in Telum.sector_slots(q)
        q.iszero[sector_index] && continue
        RMTs[sector_index] = deepcopy(Telum.sector_rmt(q, sector_index))
    end

    with_zero = TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, q.inds, q.spaces)
    @test Telum.nsectors(with_zero) == old_n
    @test Telum.sector_count(with_zero) == Telum.sector_count(q) + 1
    @test with_zero.isdefined[end] == false
    @test with_zero.iszero[end] == true

    wmatdata_with_extra = copy(q.wmatdata)
    wmatinfo_with_extra = copy(q.wmatinfo)
    _append_sector_wmat_info!(wmatdata_with_extra, wmatinfo_with_extra, q, 1)
    RMTs_bad = copy(RMTs)
    zero_rmt = deepcopy(Telum.sector_rmt(q, 1))
    fill!(zero_rmt, zero(eltype(zero_rmt)))
    RMTs_bad[end] = zero_rmt
    with_zero_rmt = TLArray(symm(q), qlabels, wmatdata_with_extra, wmatinfo_with_extra,
                            RMTs_bad, q.inds, q.spaces)
    @test with_zero_rmt.isdefined[end] == true
    @test with_zero_rmt.iszero[end] == true
end

@testset "wmat slot mappings infer" begin
    PS = ProductSymm(U1, SU{2}, U1, SU{2})
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(1))) === nothing
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(2))) == 1
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(3))) === nothing
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(4))) == 2
    @test @inferred(Telum.product_symmetry_index_from_wmat_slot(PS, Val(1))) == 2
    @test @inferred(Telum.product_symmetry_index_from_wmat_slot(PS, Val(2))) == 4
end

@testset "TLArray qlabel storage orientation" begin
    fermion_u1 = getLocalSpace(FermionOptions(U1))
    fermion_su2 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing), ("in", "out", "op"))

    samples = (
        fermion_u1.I,
        fermion_u1.F,
        fermion_su2.I,
        fermion_su2.F,
        empty_tlarray((U1, SU{2}), (TLIndex("empty_in", '+'), TLIndex("empty_out", '-'))),
        getvac(fermion_su2.I),
    )

    @testset "storage columns are sectors" begin
        for q in samples
            _assert_qlabel_storage(q)
            _assert_wmat_storage(q)
        end
    end

    @testset "constructor marks undefined zero sectors" begin
        for q in samples
            _assert_zero_sector_constructor_state(q)
        end
    end

    @testset "metadata helpers infer and preserve ordering" begin
        for q in samples
            _assert_metadata_inferred(q)
            for sector_index in Telum.sector_slots(q), n in 1:length(symm(q))
                q.iszero[sector_index] && continue
                qlabels, cgp, legdir = Telum._sector_cgt_metadata(q, sector_index, n)
                stored_to_phys = Telum._stored_leg_order(q, sector_index, n)
                @test qlabels == ntuple(i -> Telum.sector_qlabel(q, sector_index, stored_to_phys[i])[n], ndims(q))
                @test cgp == ntuple(leg -> findfirst(==(leg), stored_to_phys), ndims(q))
                @test legdir == (count(leg -> q.inds[leg].dir == '+', 1:ndims(q)),
                                 count(leg -> q.inds[leg].dir == '-', 1:ndims(q)))
            end
        end
    end

    @testset "field operations keep orientation" begin
        q = fermion_su2.F
        _assert_qlabel_storage(copy(q))
        _assert_qlabel_storage(TLArray(q, :))
        _assert_qlabel_storage(permutedims(q, (2, 1, 3)))
        _assert_qlabel_storage(conj(q))
        _assert_qlabel_storage(q + q)
        @test norm(q + q) ≈ 2norm(q)
        _assert_qlabel_storage(sum((q, q, q)))
        @test norm(sum((q, q, q))) ≈ 3norm(q)

        q_perm = permutedims(q, (2, 1, 3))
        @test q_perm isa TLArrayView
        @test conj(q) isa TLArrayView
        @test q * 2.0 isa TLArrayView
        from_permuted_only = sum((zero(q), q_perm))
        _assert_qlabel_storage(from_permuted_only)
        @test norm(from_permuted_only - q) < 1e-10

        sorted = Telum.sort_sectors(q)
        _assert_qlabel_storage(sorted)
        @test sorted !== q
        @test Telum.sector_rmt(sorted, 1) !== Telum.sector_rmt(q, 1)
    end

    @testset "internal TLArrayView materialization" begin
        q = fermion_su2.F
        active = first(i for i in Telum.sector_slots(q) if !Telum.is_sector_zero(q, i))

        vp = Telum._view_permutedims(q, (2, 1, 3))
        @test vp isa TLArrayView
        @test Telum.materialize(vp) === q
        @test norm(copy(vp) - permutedims(q, (2, 1, 3))) < 1e-10

        vc = Telum._view_conj(q)
        @test vc isa TLArrayView
        @test Telum.materialize(vc) === q
        @test norm(copy(vc) - conj(q)) < 1e-10

        vs = Telum._view_scale(q, 2.0)
        @test vs isa TLArrayView
        rmt, alpha = Telum.sector_rmt_with_scale(vs, active)
        @test rmt === Telum.sector_rmt(q, active)
        @test alpha == 2.0
        @test Telum.materialize(vs) === q
        @test norm(copy(vs) - q * 2.0) < 1e-10

        z = Telum._view_scale(q, 0.0)
        @test z isa TLArray
        @test Telum.sector_count(z) == Telum.sector_count(q)
        @test Telum.nsectors(z) == 0
        @test all(!Telum.is_sector_defined(z, i) && Telum.is_sector_zero(z, i)
                  for i in Telum.sector_slots(z))

        @test Telum._view_conj(Telum._view_conj(q)) === q
        @test Telum._view_scale(Telum._view_scale(q, 2.0), 0.5) === q
        @test Telum._view_permutedims(Telum._view_permutedims(q, (2, 1, 3)),
                                      (2, 1, 3)) === q
    end

    @testset "decomposition assembly keeps orientation" begin
        result = svd(fermion_su2.F, (1,))
        U, S, Vd = result.U, result.S, result.Vd
        _assert_qlabel_storage(U)
        _assert_qlabel_storage(S)
        _assert_qlabel_storage(Vd)

        eres = eigen(TLArray(fermion_su2.I, ("eig", "eig")); hermitian=true)
        _assert_qlabel_storage(eres.V)
        _assert_qlabel_storage(eres.D)
    end
end
