@testset "empty_tlarray" begin
    # ── rank-1 to rank-5 construction ────────────────────────────────────────
    @testset "rank-$QD construction" for QD in 1:5
        symm = (SU{2},)
        inds = ntuple(i -> TLIndex("l$i", i == 1 ? '+' : '-'), QD)
        q = empty_tlarray(symm, inds)

        @test isempty(q.qlabels)
        @test isempty(q.RMTs)
        @test Telum.symm(q) == symm
        @test q.inds          == inds
        @test all(isempty(q.spaces[l]) for l in 1:QD)
    end

    # ── multiple symmetries ───────────────────────────────────────────────────
    @testset "multi-symmetry construction" begin
        symm = (U1, SU{2})
        inds = (TLIndex("a", '+'), TLIndex("b", '-'), TLIndex("c", '-'))
        q = empty_tlarray(symm, inds)

        @test Telum.symm(q) == symm
        @test q.inds == inds
        @test length(q.spaces) == 3
        @test all(isempty(q.spaces[l]) for l in 1:3)
    end

    # ── element type keyword ──────────────────────────────────────────────────
    @testset "element type" begin
        symm = (SU{2},)
        inds = (TLIndex("a", '+'), TLIndex("b", '-'))
        qf64 = empty_tlarray(symm, inds; T=Float64)
        qc64 = empty_tlarray(symm, inds; T=ComplexF64)

        @test eltype(qf64.RMTs) <: Array{Float64, 3}
        @test eltype(qc64.RMTs) <: Array{ComplexF64, 3}
    end

    @testset "zero preserves metadata on TLArray" begin
        option = FermionSOptions(3, :U1, :SU2, :SU3)
        q0 = getLocalSpace(option)
        q = TLArray(q0.F, ("site1", "site2", "op"))
        qz = zero(q)

        # zero(q) should clear sector payloads without changing the tensor's leg
        # metadata, so it can still participate in compatible tensor operations.
        @test qz isa TLArray
        @test isempty(qz.qlabels)
        @test isempty(qz.RMTs)
        @test symm(qz) == symm(q)
        @test qz.inds == q.inds
        @test qz.spaces == q.spaces
    end

    # ── show does not error ───────────────────────────────────────────────────
    @testset "show on empty TLArray" begin
        symm = (SU{2},)
        inds = (TLIndex("a", '+'), TLIndex("b", '-'))
        q = empty_tlarray(symm, inds)
        buf = IOBuffer()
        @test (show(buf, MIME"text/plain"(), q); true)
        @test occursin("empty", String(take!(buf)))
    end

    @testset "printmeta on TLArray" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        q = TLArray(q0.F, ("site1", "site2", "op"))

        meta = sprint(printmeta, q)

        @test occursin("3D TLArray", meta)
        @test occursin("site1", meta)
        @test !occursin('\n', meta)
    end
end

@testset "zero_qlabels" begin
    q_empty = empty_tlarray((SU{2}, SU{3}), (TLIndex('+'), TLIndex('-')))
    @test zero_qlabels(q_empty) == ((0,), (0, 0))
    @test zero_qlabels(symm(q_empty)) == ((0,), (0, 0))

    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    @test zero_qlabels(q0.I) == ((0,), (0,))
end

@testset "qlabeltype" begin
    q_empty = empty_tlarray((U1, SU{3}), (TLIndex('+'), TLIndex('-')))
    expected = Tuple{Tuple{Int}, NTuple{2, Int}}
    expected_ps = ProductSymm{Tuple{U1, SU{3}}}

    @test qlabeltype(q_empty) == expected
    @test qlabeltype(symm(q_empty)) == expected
    @test productsymm(q_empty) == expected_ps
    @test @inferred(symm(q_empty)) == (U1, SU{3})
    @test eltype(q_empty.spaces[1]) == Tuple{expected, Int}

    q_multi = empty_tlarray((U1, SU{2}, SU{3}), (TLIndex('+'),))
    @test qlabeltype(q_multi) == Tuple{Tuple{Int}, Tuple{Int}, NTuple{2, Int}}
    @test productsymm(q_multi) == ProductSymm{Tuple{U1, SU{2}, SU{3}}}

    q_local = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).I
    @test eltype(q_local.qlabels) == NTuple{2, qlabeltype(q_local)}
    info = Telum.leginfo(q_local, 1)
    @test qlabeltype(info) == qlabeltype(q_local)
    @test eltype(info.splist) == Tuple{qlabeltype(q_local), Int}
end

@testset "getsub sector slicing" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    candidate = nothing
    # Pick a real local-space tensor with both a full-sector selection and a
    # sector whose multiplicity is large enough to test reordering/slicing.
    for base in values(q0)
        for legcand in 1:length(base.spaces)
            qcand = oplus([base, 2.0 * base, 3.0 * base], legcand)
            reorder_idx = findfirst(i -> qcand.spaces[legcand][i][2] >= 3, eachindex(qcand.spaces[legcand]))
            isnothing(reorder_idx) && continue

            full_idx = findfirst(i -> i != reorder_idx, eachindex(qcand.spaces[legcand]))
            isnothing(full_idx) && continue

            candidate = (
                q = qcand,
                leg = legcand,
                sector_reorder = qcand.spaces[legcand][reorder_idx][1],
                dim_reorder = qcand.spaces[legcand][reorder_idx][2],
                sector_full = qcand.spaces[legcand][full_idx][1],
                dim_full = qcand.spaces[legcand][full_idx][2],
            )
            break
        end
        !isnothing(candidate) && break
    end

    @test !isnothing(candidate)

    q = candidate.q
    leg = candidate.leg
    sector_reorder = candidate.sector_reorder
    dim_reorder = candidate.dim_reorder
    sector_full = candidate.sector_full
    dim_full = candidate.dim_full

    row_sector(qs::TLArray, sector_index::Int) = qs.qlabels[sector_index][leg]
    rows_for_sector(qs::TLArray, sector) =
        [i for i in _test_defined_sector_indices(qs) if row_sector(qs, i) == sector]

    # Keep one sector intact and reverse two multiplicity rows in another; this
    # checks that getsub updates spaces and slices only the selected RMT axis.
    q_pred_pairs = Telum.getsub(q, leg,
                                  sector -> sector == sector_full ? Colon() :
                                            sector == sector_reorder ? (dim_reorder, 1) :
                                            nothing)

    # Expected metadata for the selected leg: the full sector keeps its original
    # multiplicity, while the reordered sector is truncated to the two requested rows.
    expected_spaces_leg = [
        (sector, sector == sector_full ? dim_full : 2)
        for (sector, _) in q.spaces[leg]
        if sector == sector_full || sector == sector_reorder
    ]

    # Only the selected leg's space list changes; all other legs remain compatible
    # with the source tensor.
    @test q_pred_pairs.spaces[leg] == expected_spaces_leg
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_pred_pairs.spaces[other_leg] == q.spaces[other_leg]
    end
    @test Set(row_sector(q_pred_pairs, i) for i in _test_defined_sector_indices(q_pred_pairs)) ==
          Set([sector_full, sector_reorder])

    # Colon() keeps every RMT entry for the full sector unchanged.
    full_rows = rows_for_sector(q_pred_pairs, sector_full)
    orig_full_rows = rows_for_sector(q, sector_full)
    @test length(full_rows) == length(orig_full_rows)
    for (full_idx, orig_full_idx) in zip(full_rows, orig_full_rows)
        @test Telum.sector_rmt(q_pred_pairs, full_idx) == Telum.sector_rmt(q, orig_full_idx)
    end

    # Tuple selectors are applied in the requested order to the selected RMT axis.
    reorder_rows = rows_for_sector(q_pred_pairs, sector_reorder)
    orig_reorder_rows = rows_for_sector(q, sector_reorder)
    @test length(reorder_rows) == length(orig_reorder_rows)
    reorder_inds = [dim_reorder, 1]
    for (reorder_idx, orig_reorder_idx) in zip(reorder_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        reorder_selector = ntuple(d -> d == leg ? reorder_inds : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_pred_pairs, reorder_idx) == orig_rmt[reorder_selector...]
    end

    # Range selectors are normalized to the same explicit RMT-axis selection.
    q_range = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (1:2) : nothing)
    @test q_range.spaces[leg] == [(sector_reorder, 2)]
    range_rows = rows_for_sector(q_range, sector_reorder)
    @test length(range_rows) == length(orig_reorder_rows)
    for (range_idx, orig_reorder_idx) in zip(range_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        range_selector = ntuple(d -> d == leg ? [1, 2] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_range, range_idx) == orig_rmt[range_selector...]
    end

    # Negative indices follow Julia indexing from the end of the multiplicity axis.
    q_negative = Telum.getsub(q, leg, sector -> sector == sector_reorder ? -1 : nothing)
    negative_rows = rows_for_sector(q_negative, sector_reorder)
    @test q_negative.spaces[leg] == [(sector_reorder, 1)]
    @test length(negative_rows) == length(orig_reorder_rows)
    for (negative_idx, orig_reorder_idx) in zip(negative_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        negative_selector = ntuple(d -> d == leg ? [dim_reorder] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_negative, negative_idx) == orig_rmt[negative_selector...]
    end

    # Mixed negative and positive selectors should preserve the explicit order.
    q_mixed = Telum.getsub(q, leg, sector -> sector == sector_reorder ? [-1, 1] : nothing)
    mixed_rows = rows_for_sector(q_mixed, sector_reorder)
    @test q_mixed.spaces[leg] == [(sector_reorder, 2)]
    @test length(mixed_rows) == length(orig_reorder_rows)
    for (mixed_idx, orig_reorder_idx) in zip(mixed_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        mixed_selector = ntuple(d -> d == leg ? [dim_reorder, 1] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_mixed, mixed_idx) == orig_rmt[mixed_selector...]
    end

    # Returning nothing for every sector drops all payloads and clears only the
    # selected leg's space list.
    q_empty = Telum.getsub(q, leg, _ -> nothing)
    @test isempty(_test_defined_sector_indices(q_empty))
    @test isempty(q_empty.spaces[leg])
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_empty.spaces[other_leg] == q.spaces[other_leg]
    end

    # Invalid selectors fail early: bad legs, duplicates, empty selections, zero,
    # out-of-bounds values, unsupported return types, and slicing with preserved spaces.
    @test_throws ArgumentError Telum.getsub(q, 0, _ -> Colon())
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? [1, 1] : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? Int[] : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? 0 : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? (dim_reorder + 1) : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? "bad" : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? 1 : nothing; preserve_space=true)
end

@testset "getsub sector predicate" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    candidate = nothing
    # Use an oplus tensor so at least one leg has multiple sectors for the
    # predicate to filter while the untouched legs keep their original spaces.
    for base in values(q0)
        for legcand in 1:length(base.spaces)
            qcand = oplus([base, 2.0 * base, 3.0 * base], legcand)
            length(qcand.spaces[legcand]) >= 2 || continue
            candidate = (q = qcand, leg = legcand, target_sector = qcand.spaces[legcand][1][1])
            break
        end
        !isnothing(candidate) && break
    end

    @test !isnothing(candidate)

    q = candidate.q
    leg = candidate.leg
    target_sector = candidate.target_sector

    row_sector(qs::TLArray, sector_index::Int) = qs.qlabels[sector_index][leg]
    expected_rows = [i for i in _test_defined_sector_indices(q) if row_sector(q, i) == target_sector]
    expected_leg_spaces = [entry for entry in q.spaces[leg] if entry[1] == target_sector]

    q_component = Telum.getsub(q, leg, sector -> sector[1] == target_sector[1] ? Colon() : nothing)
    expected_component_rows = [i for i in _test_defined_sector_indices(q) if row_sector(q, i)[1] == target_sector[1]]
    expected_component_spaces = [entry for entry in q.spaces[leg] if entry[1][1] == target_sector[1]]
    _test_tlarrays_same_sector_payloads(q_component, TLArray(q, expected_component_rows))
    @test q_component.spaces[leg] == expected_component_spaces

    # preserve_space keeps the original leg metadata even when only a subset of
    # sector payloads remains defined.
    q_preserved = Telum.getsub(q, leg, sector -> sector == target_sector ? Colon() : nothing; preserve_space=true)
    _test_tlarrays_same_sector_payloads(q_preserved, TLArray(q, expected_rows))
    @test q_preserved.spaces == q.spaces

    q_none_preserved = Telum.getsub(q, leg, _ -> nothing; preserve_space=true)
    @test isempty(_test_defined_sector_indices(q_none_preserved))
    @test q_none_preserved.spaces == q.spaces

    @test_throws ArgumentError Telum.getsub(q, leg, _ -> false)
    @test_throws ArgumentError Telum.getsub(q, leg, _ -> true)
end

@testset "getsub multi-leg sector predicate" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    base = getLocalSpace(option, ("sel,left", "sel,right", "op")).I
    q = oplus([base, 2.0 * base, 3.0 * base], (1, 2))
    legs = (1, 2)

    @test length(q.spaces[1]) >= 2
    @test length(q.spaces[2]) >= 2
    @test !isempty(_test_defined_sector_indices(q))

    row_sector_at(sector_index, leg) = q.qlabels[sector_index][leg]
    first_sector = first(_test_defined_sector_indices(q))
    allowed = Set{Any}([row_sector_at(first_sector, 1), row_sector_at(first_sector, 2)])
    pred = sector -> sector in allowed ? Colon() : nothing

    # Filtering both tagged legs must keep only sectors whose labels pass on
    # every selected leg, while non-selected legs retain their original spaces.
    expected_rows = [
        i for i in _test_defined_sector_indices(q)
        if all(row_sector_at(i, leg) in allowed for leg in legs)
    ]
    expected_spaces_1 = [entry for entry in q.spaces[1] if entry[1] in allowed]
    expected_spaces_2 = [entry for entry in q.spaces[2] if entry[1] in allowed]

    q_multi = Telum.getsub(q, legs, pred)
    _test_tlarrays_same_sector_payloads(q_multi, TLArray(q, expected_rows))
    @test q_multi.spaces[1] == expected_spaces_1
    @test q_multi.spaces[2] == expected_spaces_2
    for other_leg in 1:length(q.spaces)
        other_leg in legs && continue
        @test q_multi.spaces[other_leg] == q.spaces[other_leg]
    end

    pred_slice = sector -> sector in allowed ? 1 : nothing
    q_multi_sliced = Telum.getsub(q, legs, pred_slice)
    @test q_multi_sliced.spaces[1] == [(entry[1], 1) for entry in q.spaces[1] if entry[1] in allowed]
    @test q_multi_sliced.spaces[2] == [(entry[1], 1) for entry in q.spaces[2] if entry[1] in allowed]
    for other_leg in 1:length(q.spaces)
        other_leg in legs && continue
        @test q_multi_sliced.spaces[other_leg] == q.spaces[other_leg]
    end
    @test length(_test_defined_sector_indices(q_multi_sliced)) == length(expected_rows)
    for (sliced_idx, orig_idx) in zip(_test_defined_sector_indices(q_multi_sliced), expected_rows)
        orig_rmt = Telum.sector_rmt(q, orig_idx)
        slice_selector = ntuple(d -> d in legs ? [1] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_multi_sliced, sliced_idx) == orig_rmt[slice_selector...]
    end

    q_multi_preserved = Telum.getsub(q, legs, pred; preserve_space=true)
    _test_tlarrays_same_sector_payloads(q_multi_preserved, TLArray(q, expected_rows))
    @test q_multi_preserved.spaces == q.spaces

    q_multi_kw = Telum.getsub(q, pred; itag="sel")
    _test_tlarrays_same_sector_payloads(q_multi_kw, q_multi)
    @test q_multi_kw.spaces == q_multi.spaces

    @test_throws ArgumentError Telum.getsub(q, Int[], pred)
    @test_throws ArgumentError Telum.getsub(q, (1, 1), pred)
    @test_throws ArgumentError Telum.getsub(q, (0, 1), pred)
    @test_throws ArgumentError Telum.getsub(q, legs, pred_slice; preserve_space=true)
    @test_throws ArgumentError Telum.getsub(q, pred; itag="missing")
end

@testset "getvac" begin
    @testset "single trivial sector with default tags" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        vac = getvac(q0.I)

        # The vacuum tensor is a rank-2 identity on the trivial sector, independent
        # of the nontrivial sectors present in the input local space.
        @test symm(vac) == symm(q0.I)
        @test length(_test_defined_sector_indices(vac)) == 1
        @test vac.inds == (TLIndex("", '+'), TLIndex("", '-'))
        @test length(vac.spaces[1]) == 1
        @test length(vac.spaces[2]) == 1

        trivial = zero_qlabels(vac)
        @test vac.spaces[1][1] == (trivial, 1)
        @test vac.spaces[2][1] == (trivial, 1)
        @test Array(to_sparse_array(vac)) == ones(1, 1)
    end

    @testset "optional tags are applied" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        vac = getvac(q0.F, ("vin", "vout"))

        @test vac.inds[1] == TLIndex("vin", '+')
        @test vac.inds[2] == TLIndex("vout", '-')
    end
end

@testset "addSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F
    q_rank2 = TLArray(q0.I, ("lin", "lout"))

    q_default = addSingleton(q, 2)
    @test q_default.inds[1] == q.inds[1]
    @test q_default.inds[2] == TLIndex("", '+')
    @test q_default.inds[3] == q.inds[2]
    @test q_default.inds[4] == q.inds[3]

    trivial = zero_qlabels(q)
    # Added singleton legs always carry the trivial sector with multiplicity one.
    @test q_default.spaces[2] == [(trivial, 1)]

    q_append_default = addSingleton(q)
    @test q_append_default.inds[1:3] == q.inds
    @test q_append_default.inds[4] == TLIndex("", '+')
    @test q_append_default.spaces[4] == [(trivial, 1)]

    q_append_two = addSingleton(q; nlegs=2,
                                itag=("tail_left", "tail_right"),
                                plev=(1, 2),
                                lock=(0, 1),
                                dir=('+', '-'))
    @test q_append_two.inds[1:3] == q.inds
    @test q_append_two.inds[4] == TLIndex("tail_left", '+', 1, 0)
    @test q_append_two.inds[5] == TLIndex("tail_right", '-', 2, 1)
    @test q_append_two.spaces[4] == [(trivial, 1)]
    @test q_append_two.spaces[5] == [(trivial, 1)]
    @test Array(to_sparse_array(q_append_two)) ==
          Array(to_sparse_array(addSingleton(q, (4, 5);
                                             itag=("tail_left", "tail_right"),
                                             plev=(1, 2),
                                             lock=(0, 1),
                                             dir=('+', '-'))))

    @test_throws ArgumentError addSingleton(q; nlegs=0)

    q_added = addSingleton(q, (1, 4);
                           itag=("left_aux", "right_aux"),
                           plev=(2, 3),
                           lock=(0, 1),
                           dir=('-', '+'))

    @test q_added.inds[1] == TLIndex("left_aux", '-', 2, 0)
    @test q_added.inds[2] == q.inds[1]
    @test q_added.inds[3] == q.inds[2]
    @test q_added.inds[4] == TLIndex("right_aux", '+', 3, 1)
    @test q_added.inds[5] == q.inds[3]
    @test q_added.spaces[1] == [(trivial, 1)]
    @test q_added.spaces[4] == [(trivial, 1)]

    arr_ref = _dense_addSingleton_ref(Array(to_sparse_array(q)), (1, 4))
    arr_added = Array(to_sparse_array(q_added))
    # Dense comparison verifies that inserting metadata-only singleton legs did
    # not reorder existing physical axes.
    @test size(arr_added) == size(arr_ref)
    @test norm(arr_added - arr_ref) < 1e-10

    q_rank2_added = addSingleton(q_rank2, 2)
    arr_rank2_ref = _dense_addSingleton_ref(Array(to_sparse_array(q_rank2)), 2)
    arr_rank2_added = Array(to_sparse_array(q_rank2_added))
    @test size(arr_rank2_added) == size(arr_rank2_ref)
    @test norm(arr_rank2_added - arr_rank2_ref) < 1e-10
end

@testset "deleteSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F
    q_rank2 = TLArray(q0.I, ("lin", "lout"))

    q_one = addSingleton(q, 2; itag="aux", plev=3, dir='-')
    q_two = addSingleton(q, (1, 4);
                         itag=("left_aux", "right_aux"),
                         plev=(2, 5),
                         dir=('-', '+'))
    q_rank2_added = addSingleton(q_rank2, 2; itag="mid_aux", dir='+')

    # Deleting singleton legs should restore both metadata and sparse values for
    # explicit leg selection, keyword selection, and default singletons.
    q_deleted_leg = deleteSingleton(q_one, 2)
    @test q_deleted_leg.inds == q.inds
    @test q_deleted_leg.spaces == q.spaces
    @test Array(to_sparse_array(q_deleted_leg)) == Array(to_sparse_array(q))

    q_deleted_legs = deleteSingleton(q_two, (1, 4))
    @test q_deleted_legs.inds == q.inds
    @test q_deleted_legs.spaces == q.spaces
    @test Array(to_sparse_array(q_deleted_legs)) == Array(to_sparse_array(q))

    q_deleted_kw_tag = deleteSingleton(q_two; itag="left_aux")
    @test length(q_deleted_kw_tag.inds) == 4
    @test q_deleted_kw_tag.inds[1] == q.inds[1]
    @test q_deleted_kw_tag.inds[2] == q.inds[2]
    @test q_deleted_kw_tag.inds[3] == TLIndex("right_aux", '+', 5, 0)
    @test q_deleted_kw_tag.inds[4] == q.inds[3]
    @test Array(to_sparse_array(q_deleted_kw_tag)) == Array(to_sparse_array(addSingleton(q, 3; itag="right_aux", plev=5, dir='+')))

    q_deleted_kw_plev = deleteSingleton(q_two; plev=5)
    @test length(q_deleted_kw_plev.inds) == 4
    @test q_deleted_kw_plev.inds[1] == TLIndex("left_aux", '-', 2, 0)
    @test q_deleted_kw_plev.inds[2] == q.inds[1]
    @test q_deleted_kw_plev.inds[3] == q.inds[2]
    @test q_deleted_kw_plev.inds[4] == q.inds[3]

    q_rank2_roundtrip = deleteSingleton(q_rank2_added)
    @test q_rank2_roundtrip.inds == q_rank2.inds
    @test q_rank2_roundtrip.spaces == q_rank2.spaces
    @test Array(to_sparse_array(q_rank2_roundtrip)) == Array(to_sparse_array(q_rank2))

    @test_throws ArgumentError deleteSingleton(q, 1)
    @test_throws ArgumentError deleteSingleton(q_two, (1, 2))
    @test_throws ArgumentError deleteSingleton(q_two, Int[])
    @test_throws ArgumentError deleteSingleton(q_two, (1, 1))
    @test_throws ArgumentError deleteSingleton(q_two, 0)

    @test_logs (:warn, r"no singleton legs found") deleteSingleton(q)
    @test_logs (:warn, r"no singleton legs matched") deleteSingleton(q_two; itag="ain")
end

@testset "TLArray tensor product" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    q1 = TLArray(q0.I, ("l1_in", "l1_out"))
    q2 = TLArray(q0.F, ("l2_in", "l2_out", "l2_op"))

    q12 = Telum.:⊗(q1, q2)
    q12_kron = kron(q1, q2)

    @test symm(q12) == symm(q1) == symm(q2)
    @test q12.inds == (q1.inds..., q2.inds...)
    @test q12.spaces == (q1.spaces..., q2.spaces...)
    @test q12_kron.inds == q12.inds
    @test q12_kron.spaces == q12.spaces

    arr_ref = _dense_tensor_product_ref(Array(to_sparse_array(q1)), Array(to_sparse_array(q2)))
    arr_q12 = Array(to_sparse_array(q12))
    arr_q12_kron = Array(to_sparse_array(q12_kron))
    # Compare against a dense tensor-product reference, and check kron is the
    # same public operation as the unicode tensor product.
    @test size(arr_q12) == size(arr_ref)
    @test norm(arr_q12 - arr_ref) < 1e-10
    @test norm(arr_q12_kron - arr_q12) < 1e-10
end

# Helper: build a rank-4 test TLArray from getIdentity with 3 input legs.
#
#   leg 1 ('+', "l1"), leg 2 ('+', "l2"), leg 3 ('+', "l3"),
#   leg 4 ('-', "fused")
# ─────────────────────────────────────────────────────────────────────────────
function _make_test_tlarray_rank4()
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0  = getLocalSpace(option)
    qi1 = TLArray(q0.I, ("b1a", "b1b"))
    qi2 = TLArray(q0.I, ("b2a", "b2b"))
    qi3 = TLArray(q0.I, ("b3a", "b3b"))
    a4  = getIdentity((qi1, 2), (qi2, 2), (qi3, 2); itag="fused")
    return TLArray(a4, ("l1", "l2", "l3", "fused"))
end


@testset "scalar add/subtract on rank-2 TLArray" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    q = TLArray(q0.I, ("left", "right"))

    # Scalar arithmetic is defined as adding/subtracting a scaled identity on the
    # same rank-2 space, not as touching every stored sector directly.
    idq = getIdentity(q, 2; itag=q.inds[2].itags)
    idq = TLArray(idq, q.inds)

    @test norm((q + 2.5) - (q + 2.5 * idq)) < 1e-10
    @test norm((q - 2.5) - (q - 2.5 * idq)) < 1e-10
    @test norm((2.5 + q) - (q + 2.5 * idq)) < 1e-10
    @test norm((2.5 - q) - (2.5 * idq - q)) < 1e-10

    q_complex_left = (1.0 + 2.0im) * q
    q_complex_right = q * (1.0 + 2.0im)
    arr_q = Array(to_sparse_array(q))
    @test Array(to_sparse_array(q_complex_left)) ≈ (1.0 + 2.0im) .* arr_q
    @test Array(to_sparse_array(q_complex_right)) ≈ (1.0 + 2.0im) .* arr_q
    shown_complex = sprint(show, MIME"text/plain"(), q_complex_left)
    @test occursin("im", shown_complex)

    q_bad_rank = _make_test_tlarray_rank4()
    @test_throws AssertionError q_bad_rank + 1.0

    q_bad_dirs = getIdentity(q, 1, 2; itag="fused")
    @test_throws AssertionError q_bad_dirs + 1.0
end


# ─────────────────────────────────────────────────────────────────────────────
# Tests for arbitrary-rank (rank ≥ 4) TLArray
# ─────────────────────────────────────────────────────────────────────────────

@testset "arbitrary rank TLArray" begin
    q4 = _make_test_tlarray_rank4()

    # ── basic structure ───────────────────────────────────────────────────────
    @testset "rank-4 structure" begin
        @test length(q4.inds)  == 4
        @test length(q4.spaces) == 4
        @test q4.inds[1].dir   == '+'
        @test q4.inds[2].dir   == '+'
        @test q4.inds[3].dir   == '+'
        @test q4.inds[4].dir   == '-'
        @test q4.inds[1].itags == "l1"
        @test q4.inds[4].itags == "fused"
        @test !isempty(_test_defined_sector_indices(q4))
        @test all(!isempty(q4.spaces[l]) for l in 1:4)
    end

    @testset "svd keyword leg selection" begin
        ref = svd(q4, (1, 2, 3), "kwL", "kwR")
        kw = svd(q4, "kwL", "kwR"; dir='+')
        U_ref, S_ref, Vd_ref = ref.U, ref.S, ref.Vd
        U_kw, S_kw, Vd_kw = kw.U, kw.S, kw.Vd

        # Keyword selection by direction should choose the same left legs as the
        # explicit tuple and therefore produce identical factor metadata/values.
        @test U_kw.inds == U_ref.inds
        @test S_kw.inds == S_ref.inds
        @test Vd_kw.inds == Vd_ref.inds

        @test U_kw.spaces == U_ref.spaces
        @test S_kw.spaces == S_ref.spaces
        @test Vd_kw.spaces == Vd_ref.spaces

        @test norm(U_kw - U_ref) < 1e-12
        @test norm(S_kw - S_ref) < 1e-12
        @test norm(Vd_kw - Vd_ref) < 1e-12

        @test_throws ArgumentError svd(q4; itag="missing")
        @test_throws ArgumentError svd(q4; lock=0)
    end

    # ── scalar multiplication ─────────────────────────────────────────────────
    @testset "scalar multiplication on rank-4" begin
        q_scaled = 2.5 * q4
        arr_orig   = Array(to_sparse_array(q4))
        arr_scaled = Array(to_sparse_array(q_scaled))
        @test norm(arr_scaled - 2.5 .* arr_orig) < 1e-10
    end

    # ── addition: q4 + q4 ≈ 2*q4 ─────────────────────────────────────────────
    @testset "addition on rank-4: q + q ≈ 2*q" begin
        q_sum    = q4 + q4
        q_double = 2.0 * q4
        arr_sum    = Array(to_sparse_array(q_sum))
        arr_double = Array(to_sparse_array(q_double))
        @test norm(arr_sum - arr_double) < 1e-10
    end

    # ── conj: conj(conj(q)) ≈ q ───────────────────────────────────────────────
    @testset "double-conj roundtrip on rank-4" begin
        qc  = conj(q4)
        # all leg directions must flip
        @test all(qc.inds[i].dir != q4.inds[i].dir for i in 1:4)
        qcc = conj(qc)
        # directions restored
        @test all(qcc.inds[i].dir == q4.inds[i].dir for i in 1:4)
        # tensor values preserved
        arr_orig = Array(to_sparse_array(q4))
        arr_cc   = Array(to_sparse_array(qcc))
        @test norm(arr_orig - arr_cc) < 1e-10
    end

    @testset "show on rank-4 TLArray" begin
        buf = IOBuffer()
        @test (show(buf, MIME"text/plain"(), q4); true)
        out = String(take!(buf))
        @test occursin("4D TLArray", out)
    end
end
