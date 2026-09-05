function _one_sector_matrix_tlarray(matrix::AbstractMatrix{T};
                                    qlabel=((0,),),
                                    tags=("phys", "phys")) where {T}
    n = size(matrix, 1)
    @assert size(matrix, 2) == n
    symmetries = (U1,)
    qlabels = [(qlabel, qlabel)]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0))]
    rmt = reshape(Matrix{T}(matrix), n, n, 1)
    inds = (TLIndex(tags[1], '+'), TLIndex(tags[2], '-'))
    spaces = ([(qlabel, n)], [(qlabel, n)])
    return TLArray(symmetries, qlabels, wmatdata, wmatinfo, [rmt], inds, spaces)
end

function _dense_matrix(q::TLArray)
    arr = Array(to_sparse_array(q))
    return reshape(arr, size(arr, 1), size(arr, 2))
end

function _eigen_reconstruction(result)
    # Lock the original matrix leg so multiplication contracts only the eigen bond.
    V = lock(result.V, 1)
    return isnothing(result.V_inv) ? V * result.D * result.V' : V * result.D * result.V_inv
end

function _assert_eigen_reconstructs(q::TLArray; hermitian::Bool=false, tol=1e-11)
    result = eigen(q; hermitian)
    rec = _eigen_reconstruction(result)
    @test norm(q - rec) / max(norm(q), 1.0) < tol
    return result
end

function _expanded_eigenvalues(result)
    vals = mapreduce(entry -> fill(entry[1], entry[2]), append!, result.eig_list;
                     init=eltype(first(result.eig_list)[1])[])
    return sort(vals; by=abs)
end

function _dense_eigenvalues(q::TLArray; hermitian::Bool=false)
    vals = hermitian ? LinearAlgebra.eigvals(Hermitian(_dense_matrix(q))) :
                       LinearAlgebra.eigvals(_dense_matrix(q))
    return sort(vals; by=abs)
end

@testset "identity eigenvalues are one" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(3, :U1, :SU2, :SU3),
    )
        q = getLocalSpace(option)
        result = eigen(q.I; hermitian=true)
        vals = _expanded_eigenvalues(result)
        @test !isempty(vals)
        @test all(isapprox.(vals, one(eltype(vals)); atol=1e-10, rtol=1e-10))
        @test isnothing(result.V_inv)
    end

    q10 = _one_sector_matrix_tlarray(Matrix{Float64}(I, 10, 10))
    result10 = eigen(q10; hermitian=true)
    @test _expanded_eigenvalues(result10) ≈ ones(10)
end

@testset "eigen path assigns inverse only for non-Hermitian input" begin
    hermitian_matrix = Symmetric(randn(10, 10))
    hermitian_q = _one_sector_matrix_tlarray(Matrix(hermitian_matrix))
    hermitian_result = _assert_eigen_reconstructs(hermitian_q; hermitian=true)
    @test isnothing(hermitian_result.V_inv)

    general_matrix = randn(10, 10)
    general_matrix[1, 2] += 3.0
    general_matrix[2, 1] -= 2.0
    general_q = _one_sector_matrix_tlarray(general_matrix)
    general_result = _assert_eigen_reconstructs(general_q)
    @test !isnothing(general_result.V_inv)
end

@testset "eigenvalues match dense eigendecomposition" begin
    real_hermitian = Matrix(Symmetric(randn(10, 10)))
    real_hermitian_q = _one_sector_matrix_tlarray(real_hermitian)
    real_hermitian_result = _assert_eigen_reconstructs(real_hermitian_q; hermitian=true)
    @test _expanded_eigenvalues(real_hermitian_result) ≈
          _dense_eigenvalues(real_hermitian_q; hermitian=true) atol=1e-10 rtol=1e-10

    complex_general = randn(ComplexF64, 10, 10)
    complex_general[1, 2] += 2.0 - 0.5im
    complex_general[2, 1] -= 1.0 + 1.5im
    complex_general_q = _one_sector_matrix_tlarray(complex_general)
    complex_general_result = _assert_eigen_reconstructs(complex_general_q; tol=1e-8)
    @test !isnothing(complex_general_result.V_inv)
    @test _expanded_eigenvalues(complex_general_result) ≈
          _dense_eigenvalues(complex_general_q) atol=1e-8 rtol=1e-8
end

test_eigen_multi_sector()
test_eigen_tlarrayview()
