# run_two_site_tests.jl
#
# Convenience runner for the two-site BUG odd/even sweep test suite.
# Run with:  julia --project tests/BUG/two_site_bug/run_two_site_tests.jl

using Test, ITensors, ITensorMPS, LinearAlgebra, Random

@testset "two-site BUG odd/even sweep" begin
    include("test_two_site_decomposition.jl")
    include("test_two_site_environments.jl")
    include("test_two_site_local_kls.jl")
    # TODO: rewrite or delete test_two_site_long_range_xx.jl (exercised deleted terms overload)
    # include("test_two_site_long_range_xx.jl")
    include("test_two_site_trotter_convergence.jl")
    include("test_two_site_xx_analytical.jl")
end
