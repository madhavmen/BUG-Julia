# tests/runtests.jl -- the `BondUpdateBUG` suite.
#
# Run with:  julia --project=. tests/runtests.jl
#
# ⛔⛔ THIS FILE WAS MISSING, AND `tests/submit_full_test_suite.slurm` HAS BEEN INVOKING IT ALL
# ALONG (`julia --project=. tests/runtests.jl`). Under `set -e` that slurm job died immediately;
# without it, it "passed" having run nothing. Either way the `tests/BondUpdateBUG/` files had NO
# RUNNER, which is why a guard could be written and still never execute.
#
# That is the same class of failure as the defect that prompted this file's creation: a d=4 step
# churning 28.5 GiB at bond dimension 32 went unnoticed for a whole campaign because nothing
# measured it. A test nobody runs and a metric nobody records fail in exactly the same way.
#
# ⚠ ORDERING IS DELIBERATE, matching `tests/sweeps/runtests.jl`'s stated design: primitives
# first, then the things built on them, so a red test localises instead of leaving suspects.
#
# ⚠ AND THE LIST IS EXPLICIT, NOT A GLOB. A glob silently picks up whatever lands in the
# directory, including half-written files, and silently drops a file that gets renamed -- the
# failure mode this file exists to close.
#
# ⚠ PRE-EXISTING STATE UNKNOWN. These files had no runner, so some may be red for reasons that
# predate this suite. `tests/sweeps/runtests.jl` is separately known to be red (4 magnon
# failures). Do not read a failure here as a regression without checking it against the file's
# own history first.

using Test

@testset "BondUpdateBUG" begin
    # 1. primitives: the state container, its frames, and the Krylov exponential.
    @testset "symmetric_mps" begin include("BondUpdateBUG/test_symmetric_mps.jl") end
    @testset "frame"         begin include("BondUpdateBUG/test_frame.jl") end
    @testset "expv"          begin include("BondUpdateBUG/test_expv.jl") end
    @testset "gates"         begin include("BondUpdateBUG/test_gates.jl") end

    # 2. sector bookkeeping, then the memo layered on top of it.
    @testset "sectors"       begin include("BondUpdateBUG/test_sectors.jl") end
    @testset "fusion cache"  begin include("BondUpdateBUG/test_fusion_cache.jl") end

    # 3. an end-to-end physics case.
    @testset "z2 ising"      begin include("BondUpdateBUG/test_z2_ising.jl") end
end
