# EXACT finite-size excitation gap of the N=12 clusters, to test whether eps(M) is physics.
#
# LSWT is gapless at the ordering wavevector; a FINITE cluster cannot be. The Anderson tower-of-
# states gap ~ 1/(N*chi_perp) is real physics that vanishes only as N -> oo. If eps(M) from the
# S(q,w) pipeline equals E1 - E0 of the SAME cluster, the gap is the cluster's, not the method's.
using LinearAlgebra, SparseArrays, Printf
using BUGJulia
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "exact_sparse.jl"))

function gap(Jm::Matrix{Float64}, L::Int)
    b = sz0_basis(L)
    H = pairs_sparse(Jm, L, b)
    vals, _ = eigen(Matrix(Hermitian(Matrix(H))))
    return vals[1], vals[2] - vals[1]
end

for (name, Jm) in (("square   3x4", square_cylinder_couplings(3, 4; periodic_y = true)),
                   ("triangle 3x4", triangular_cylinder_couplings(3, 4; J2 = 0.0,
                                                                  geometry = :xc,
                                                                  periodic_y = true)))
    E0, dE = gap(Jm, 12)
    @printf("%s   N=12   E0 = %.10f   E0/N = %.10f   EXACT GAP E1-E0 = %.6f\n",
            name, E0, E0 / 12, dE)
end
