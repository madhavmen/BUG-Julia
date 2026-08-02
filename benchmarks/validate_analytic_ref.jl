# Does the analytic free-fermion reference agree with the sparse Krylov one?
#
# Both are "exact", so they must agree to roundoff. Asked at L=12/14, where the sparse path is
# still cheap, precisely so the analytic path can be TRUSTED at L=30 where sparse cannot run.
# A new reference validated only by reasoning is not a reference.
using LinearAlgebra, Printf
include(joinpath(@__DIR__, "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

@printf("%-4s %-8s  %-14s  %-14s\n", "L", "t", "max|analytic-sparse|", "rel")
for L in (10, 12, 14)
    A, STATES, IDX = heisenberg_sparse(L; delta = 0.0)   # delta = 0 => XX
    v = domain_wall_vector(L, STATES, IDX)
    dt, nst = 0.05, 60
    worst = 0.0
    propagate!(A, v, ComplexF64(-im * dt), nst, (s, vv) -> begin
        if s % 20 == 0
            got  = magnetisation_dense(vv, L, STATES)
            want = xx_free_fermion_sz(L, s * dt)
            d = maximum(abs.(got .- want))
            worst = max(worst, d)
            @printf("%-4d %-8.2f  %-14.3e  %-14.3e\n", L, s*dt, d, d / maximum(abs.(want)))
        end
    end)
end
