function _make_test_tlarray_rank4_oplus()
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0  = getLocalSpace(option)
    qi1 = TLArray(q0.I, ("b1a", "b1b"))
    qi2 = TLArray(q0.I, ("b2a", "b2b"))
    qi3 = TLArray(q0.I, ("b3a", "b3b"))
    a4  = getIdentity((qi1, 2), (qi2, 2), (qi3, 2); itag="fused")
    return TLArray(a4, ("l1", "l2", "l3", "fused"))
end

@testset "TLArray direct sum" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    A = TLArray(q0.F, ("site1", "site2", "op"))
    B = 2.0 * A

    @testset "vector selected legs" begin
        qsum = oplus([A, B, 3.0 * A], (2, 3))
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B, 3.0 * A], (2, 3))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], spaces_ref[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector permutes inputs to reference index order" begin
        B_perm = permutedims(B, (2, 1, 3))
        qsum = oplus([A, B_perm], 3)
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "keyword leg selection" begin
        qsum = oplus([A, B]; itag="op")
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector allows different dual fields" begin
        B_dual = TLArray(B, (Telum.change_dual(B.inds[1]), B.inds[2], Telum.change_dual(B.inds[3])))
        qsum = oplus([A, B_dual], 3)
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector mixed eltype RMT storage and views" begin
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
        complex_view = permutedims(complex_dense, (2, 1))

        for (inputs, reference_inputs) in (
            ([real_dense, complex_dense], [real_dense, complex_dense]),
            ([complex_diag, real_dense], [complex_diag, real_dense]),
            ([real_dense, complex_diag], [real_dense, complex_diag]),
            ([real_dense, complex_view], [real_dense, complex_dense]),
        )
            qsum = oplus(inputs, 1)
            arr_ref, spaces_ref = _dense_vector_oplus_ref(reference_inputs, 1)
            arr_qsum = Array(to_sparse_array(qsum, ComplexF64))

            @test eltype(qsum) == ComplexF64
            @test qsum.inds == real_dense.inds
            @test qsum.spaces == (spaces_ref[1], real_dense.spaces[2])
            @test size(arr_qsum) == size(arr_ref)
            @test norm(arr_qsum - arr_ref) < 1e-10
        end

        mat = Matrix{Any}(undef, 2, 2)
        mat[1, 1] = real_dense
        mat[2, 1] = complex_diag
        mat[1, 2] = complex_dense
        mat[2, 2] = nothing

        qsum = oplus(mat, (1, 2))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref(mat, (1, 2))
        arr_qsum = Array(to_sparse_array(qsum, ComplexF64))

        @test eltype(qsum) == ComplexF64
        @test qsum.inds == real_dense.inds
        @test qsum.spaces == spaces_ref
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector validation" begin
        B_bad = TLArray(B, ("other1", "site2", "op"))
        @test_throws ArgumentError oplus([A, nothing], 3)
        @test_throws ArgumentError oplus([A, B_bad], 3)
    end

    @testset "matrix concatenation" begin
        q4 = _make_test_tlarray_rank4_oplus()
        mat = Matrix{TLArray}(undef, 2, 2)
        mat[1, 1] = q4
        mat[2, 1] = 2.0 * q4
        mat[1, 2] = 3.0 * q4
        mat[2, 2] = 4.0 * q4

        qsum = oplus(mat, (3, 4))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref(mat, (3, 4))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == q4.inds
        @test qsum.spaces == (q4.spaces[1], q4.spaces[2], spaces_ref[3], spaces_ref[4])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "matrix permutes entries to reference index order" begin
        q4 = _make_test_tlarray_rank4_oplus()
        mat = Matrix{TLArray}(undef, 2, 2)
        mat[1, 1] = q4
        mat[2, 1] = permutedims(2.0 * q4, (2, 1, 3, 4))
        mat[1, 2] = 3.0 * q4
        mat[2, 2] = 4.0 * q4

        qsum = oplus(mat, (3, 4))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref([q4 3.0 * q4; 2.0 * q4 4.0 * q4], (3, 4))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == q4.inds
        @test qsum.spaces == (q4.spaces[1], q4.spaces[2], spaces_ref[3], spaces_ref[4])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "matrix missing entries and tuple-valued dims" begin
        q4 = _make_test_tlarray_rank4_oplus()
        mat = Matrix{Any}(undef, 2, 2)
        mat[1, 1] = q4
        mat[2, 1] = nothing
        mat[1, 2] = 2.0 * q4
        mat[2, 2] = 3.0 * q4

        qsum = oplus(mat, ((1, 2), 3))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref(mat, ((1, 2), 3))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == q4.inds
        @test qsum.spaces == (spaces_ref[1], spaces_ref[2], spaces_ref[3], q4.spaces[4])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "matrix inference failure" begin
        q4 = _make_test_tlarray_rank4_oplus()
        mat = Matrix{Any}(undef, 2, 2)
        mat[1, 1] = nothing
        mat[2, 1] = nothing
        mat[1, 2] = q4
        mat[2, 2] = 2.0 * q4

        @test_throws ArgumentError oplus(mat, (3, 4))
    end
end
