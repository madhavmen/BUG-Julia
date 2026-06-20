# tests/runtests.jl
#
# Top-level test runner. Loads all submodule tests.
# Run with: julia --project tests/runtests.jl

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

# Set deterministic seed for all tests
import Random
Random.seed!(42)

# --- TTutils ---
@testset "TTutils" begin
    include("TTutils/mps/test_mps.jl")
    include("TTutils/mpo/test_mpo.jl")
end

# --- BUG (faithful-KLS kernel building blocks used by bug_two_site!) ---
@testset "BUG" begin
    include("BUG/test_bug_init.jl")
    include("BUG/test_bug_kls.jl")
    include("BUG/test_bug_local_kls_symmetry.jl")
    include("BUG/test_bug_local_solver.jl")
end

# --- MPO helpers ---
@testset "MPO Helpers" begin
    include("BUG/test_long_range_mpo_error.jl")
end

# --- two-site BUG (faithful-KLS odd/even sweep) ---
@testset "two-site BUG" begin
    include("BUG/two_site_bug/test_two_site_decomposition.jl")
    include("BUG/two_site_bug/test_two_site_environments.jl")
    include("BUG/two_site_bug/test_two_site_local_kls.jl")
    # include("BUG/two_site_bug/test_two_site_long_range_xx.jl")
    include("BUG/two_site_bug/test_two_site_trotter_convergence.jl")
    include("BUG/two_site_bug/test_two_site_xx_analytical.jl")
end

# --- TDVP ---
@testset "TDVP" begin
    include("TDVP/test_tdvp_init.jl")
    include("TDVP/test_tdvp_local.jl")
    include("TDVP/test_tdvp_integration.jl")
    include("TDVP/test_tdvp2_xx_analytical.jl")
end
