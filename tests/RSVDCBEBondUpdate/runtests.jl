# tests/RSVDCBEBondUpdate/runtests.jl
#
# Test runner for the exploratory CBE-BUG submodule (docs/cbe_bug.md).
# Run with: sbatch scripts/run_julia.sbatch tests/RSVDCBEBondUpdate/runtests.jl
#
# Kept separate from tests/runtests.jl while the module is exploratory: a red test
# here must never be mistaken for a regression in the validated `bond_update_bug!`.

using Test
using LinearAlgebra
import Random
Random.seed!(42)

@testset "RSVDCBEBondUpdate" begin
    include("test_henv.jl")
    include("test_heff.jl")
    include("test_sketch.jl")
    include("test_graded_sketch.jl")
    include("test_expand.jl")
end
