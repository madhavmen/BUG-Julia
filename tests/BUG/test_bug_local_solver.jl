using Test, LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl")); using .TTutils
include(joinpath(_SRC, "BUG", "BUG.jl"));         using .BUG

@testset "local exp solver matches dense exp(dt*H)*x (both backends, real & imag dt)" begin
    n = 16
    A = rand(ComplexF64, n, n)
    H = Matrix(Hermitian(A + A'))
    x = rand(ComplexF64, n)

    for dt in (0.05, -0.05, -0.05im, 0.05im)
        ref = exp(dt .* H) * x
        y, _ = BUG._linear_substep(
            H,
            dt,
            x;
            method = :expv,
            lanczos_tol = 1e-14,
            lanczos_maxiter = n,
            restart = 1,
        )
        @test norm(y - ref) / norm(ref) < 1e-10

        yn, _ = BUG._with_bug_expv_backend(:native_hermitian_lanczos) do
            BUG._linear_substep(
                H,
                dt,
                x;
                method = :expv,
                lanczos_tol = 1e-14,
                lanczos_maxiter = n,
                restart = 1,
            )
        end
        @test norm(yn - ref) / norm(ref) < 1e-10
    end
end
