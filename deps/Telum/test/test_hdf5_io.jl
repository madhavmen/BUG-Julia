function _assert_tlarray_structural_equal(a::TLArray, b::TLArray)
    @test symm(a) == symm(b)
    @test a.inds == b.inds
    @test a.spaces == b.spaces
end

function _assert_tlarray_roundtrip(a::TLArray, b::TLArray; tol=1e-12)
    _assert_tlarray_structural_equal(a, b)
    @test norm(a - b) / max(norm(a), 1.0) < tol
end

function _with_saved_tlarray(f, q; name="tensor")
    path = tempname() * ".h5"
    try
        save_tlarray(path, q; name)
        loaded = load_tlarray(path; name)
        f(loaded)
    finally
        isfile(path) && rm(path)
    end
end

function _with_zero_sector(q::TLArray)
    qlabels = copy(q.qlabels)
    push!(qlabels, first(q.qlabels))

    wmatdata = copy(q.wmatdata)
    wmatinfo = copy(q.wmatinfo)
    push!(wmatinfo, Telum._empty_wmat_info(Val(length(first(q.wmatinfo)))))

    RMTs = similar(q.RMTs, length(q.RMTs) + 1)
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        RMTs[sector] = deepcopy(Telum.sector_rmt(q, sector))
    end

    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, q.inds, q.spaces)
end

@testset "HDF5 TLArray round trip" begin
    q_nosym = TLArray((),
                      [((), ())],
                      Float64[],
                      [Telum._empty_wmat_info(Val(0))],
                      [reshape([2.0], 1, 1)],
                      (TLIndex("in", '+'), TLIndex("out", '-')),
                      ([(() , 1)], [(() , 1)]))
    @test symm(q_nosym) == ()
    @test q_nosym.qlabels == [((), ())]
    _with_saved_tlarray(q_nosym) do loaded
        _assert_tlarray_roundtrip(q_nosym, loaded)
        @test symm(loaded) == ()
        @test loaded.qlabels == [((), ())]
    end

    q0 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing), ("left", "right", "op"))
    q = q0.F

    q_u1 = getLocalSpace(FermionOptions(U1), ("in", "out", "op")).F
    _with_saved_tlarray(q_u1) do loaded
        _assert_tlarray_roundtrip(q_u1, loaded)
    end

    _with_saved_tlarray(q) do loaded
        _assert_tlarray_roundtrip(q, loaded)
    end

    q_zero = _with_zero_sector(q)
    @test q_zero.iszero[end]
    @test !q_zero.isdefined[end]
    _with_saved_tlarray(q_zero) do loaded
        _assert_tlarray_roundtrip(q_zero, loaded)
        @test loaded.iszero[end]
        @test !loaded.isdefined[end]
    end

    first_active = first(sector for sector in Telum.sector_slots(q) if !q.iszero[sector])
    complex_RMT_type = Array{ComplexF64, ndims(Telum.sector_rmt(q, first_active))}
    complex_RMTs = Vector{complex_RMT_type}(undef, length(q.RMTs))
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        complex_RMTs[sector] = ComplexF64.(Telum.sector_rmt(q, sector))
    end
    q_complex = TLArray(symm(q), copy(q.qlabels), copy(q.wmatdata), copy(q.wmatinfo),
                        complex_RMTs, q.inds, q.spaces)
    _with_saved_tlarray(q_complex) do loaded
        _assert_tlarray_roundtrip(q_complex, loaded)
    end

    q_view = permutedims(q, (2, 1, 3))
    q_view_eager = copy(q_view)
    _with_saved_tlarray(q_view) do loaded
        _assert_tlarray_roundtrip(q_view_eager, loaded)
    end

    q_diag = get1jtensor(q0.I, 1)
    _with_saved_tlarray(q_diag) do loaded
        _assert_tlarray_roundtrip(q_diag, loaded)
    end
end
