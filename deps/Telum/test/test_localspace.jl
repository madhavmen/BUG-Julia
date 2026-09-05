@testset "re-exported LurCGT symmetries" begin
    @test :Z in names(Telum)
    @test :U1 in names(Telum)
    @test :SU in names(Telum)
    @test :SO in names(Telum)
    @test :Sp in names(Telum)
    @test :printmeta in names(Telum)
    @test :getSymmetryInfo in names(Telum)

    @test Telum.Z === LurCGT.Z
    @test Telum.U1 === LurCGT.U1
    @test Telum.SU === LurCGT.SU
    @test Telum.SO === LurCGT.SO
    @test Telum.Sp === LurCGT.Sp
end

@testset "SpinOptions local operators across symmetries and spins" begin
    for symmetry in (:SU2, :U1, nothing), s2 in 1:5
        q = @test_nowarn getLocalSpace(SpinOptions(symmetry, s2))
        expected_ops = symmetry == :SU2 ? (:S, :I) : (:Sp, :Sz, :Sm, :I)

        for op in expected_ops
            @test hasproperty(q, op)
            tensor = getproperty(q, op)
            @test !isempty(tensor.RMTs)
            @test any(tensor.isdefined)
        end
    end
end

@testset "FermionS full-channel IROP names omit channel suffixes" begin
    q1 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    @test hasproperty(q1, :F)
    @test !hasproperty(q1, :F1)

    q3 = getLocalSpace(FermionSOptions(3, :U1, :SU2, :SU3))
    @test hasproperty(q3, :F)
    @test !hasproperty(q3, :F123)
end

@testset "Fermion local space IROPs" begin
    q1 = getLocalSpace(FermionOptions(U1))
    @test hasproperty(q1, :F)
    @test hasproperty(q1, :Z)
    @test hasproperty(q1, :I)
    @test !hasproperty(q1, :F1)
    @test size(to_sparse_array(q1.I)) == (2, 2)

    q3 = getLocalSpace(FermionOptions(3, :U1, :SU3))
    @test hasproperty(q3, :F)
    @test !hasproperty(q3, :F123)
    @test size(to_sparse_array(q3.I)) == (8, 8)
end

@testset "no-symmetry local spaces" begin
    spin = getLocalSpace(SpinOptions(nothing, 1))
    @test symm(spin.I) == ()
    @test typeof(spin.I).parameters[6] === ProductSymm{Tuple{}}
    @test spin.I.qlabels == [((), ())]
    @test spin.I.spaces == ([(() , 2)], [(() , 2)])
    @test Telum.sector_rmt(spin.I, 1) ≈ Matrix{Float64}(I, 2, 2)
    @test Telum.sector_rmt(spin.Sz, 1) ≈ [0.5 0.0; 0.0 -0.5]
    @test Telum.sector_rmt(spin.Sp, 1) ≈ (-1 / sqrt(2)) * [0.0 1.0; 0.0 0.0]
    @test Telum.sector_rmt(spin.Sm, 1) ≈ (1 / sqrt(2)) * [0.0 0.0; 1.0 0.0]

    fermion = getLocalSpace(FermionOptions(1, nothing, nothing))
    @test symm(fermion.I) == ()
    @test typeof(fermion.I).parameters[6] === ProductSymm{Tuple{}}
    @test fermion.I.qlabels == [((), ())]
    @test fermion.I.spaces == ([(() , 2)], [(() , 2)])
    @test Telum.sector_rmt(fermion.I, 1) ≈ Matrix{Float64}(I, 2, 2)
    @test size(Telum.sector_rmt(fermion.F, 1)) == (2, 2)
    @test size(Telum.sector_rmt(fermion.Z, 1)) == (2, 2)

    spinful = getLocalSpace(FermionSOptions(1, nothing, nothing, nothing))
    @test symm(spinful.I) == ()
    @test typeof(spinful.I).parameters[6] === ProductSymm{Tuple{}}
    @test spinful.I.qlabels == [((), ())]
    @test spinful.I.spaces == ([(() , 4)], [(() , 4)])
    @test Telum.sector_rmt(spinful.I, 1) ≈ Matrix{Float64}(I, 4, 4)
    @test size(Telum.sector_rmt(spinful.Fu, 1)) == (4, 4)
    @test size(Telum.sector_rmt(spinful.Fd, 1)) == (4, 4)
    @test size(Telum.sector_rmt(spinful.Z, 1)) == (4, 4)
end

struct NonCommutingSymmetryOptions <: LocalSpaceOptions end

function Telum.getSymmetryInfo(::NonCommutingSymmetryOptions)
    symm = (U1, SU{2})
    weights = ([(1,), (-1,)], [(1,), (-1,)])
    lowering_ops = (Matrix{Int}[], [sparse([0 0; 1 0])])

    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    mwirops[:I] = spdiagm(0 => ones(Float64, 2))
    return symm, weights, lowering_ops, mwirops
end

@testset "getLocalSpace validates cross-symmetry commutation" begin
    err = try
        getLocalSpace(NonCommutingSymmetryOptions())
        nothing
    catch caught
        caught
    end

    @test err isa ArgumentError
    @test occursin("must commute", sprint(showerror, err))
    @test occursin("weight[1]", sprint(showerror, err))
    @test occursin("lowering[1]", sprint(showerror, err))
end
