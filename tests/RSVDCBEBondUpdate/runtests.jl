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
    # The same H as a genuine MPO (Eq. 1.3 of arXiv:2606.28169) with rank-3 environments
    # (Eq. 1.8). Runs right after the channel environments because it is pinned AGAINST
    # them, element by element, as well as against the dense reference.
    include("test_mpo.jl")
    include("test_heff.jl")
    include("test_carry.jl")
    include("test_sketch.jl")
    include("test_graded_sketch.jl")
    include("test_expand.jl")
    include("test_cbe_bond_update.jl")
    include("test_rank_growth.jl")
    include("test_cbe_lubich.jl")
    # Jan BUG: the KLS step at every bond (S step taken, then QR'd with `R` discarded), with the
    # S step KEPT at the far bond. After test_cbe_lubich.jl because it reuses that sweep's frame
    # and write-back.
    include("test_jan_bug.jl")
    include("test_tdvp2_baseline.jl")
    include("test_driver.jl")
    include("test_modes.jl")
    include("test_tdvp1_cbe.jl")
    include("test_dimer.jl")
    include("test_su2.jl")
end
