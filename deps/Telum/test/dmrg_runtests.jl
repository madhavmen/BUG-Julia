using Test

include("support/dmrg.jl")

@testset "DMRG Hubbard validation" begin
    expected = Dict(
        20 => -82.37493703992092,
        50 => -82.39703156828021,
    )
    results = run_dmrg_validation(; time_blocks=false, nkeeps=[20, 50])

    for result in results
        @test result.energy ≈ expected[result.nkeep]
        @test issorted(result.dimensions; rev=true)
        @test length(result.dimensions) == length(result.qlabels)
    end
end
