# Analytic XX reference via Jordan-Wigner. Test-side only.
#
# NOT the same thing as tests/common/xx_free_fermion.jl, which despite its name
# is a dense 2^N exact diagonalisation built on ITensors -- the legacy stack this
# refactor removes, and no closed form at all. This file is the actual free-fermion
# solution and touches nothing bigger than an L x L matrix.
#
# That independence is the point. `dense_reference.jl` shares the integrator's own
# view of the model (same gate normalisation, same site ordering, a 2^L propagator);
# if that convention were wrong, both would be wrong together. This reference is
# derived from the Hamiltonian on paper instead, so agreement between the two is
# genuine evidence rather than a tautology.

using LinearAlgebra

"""
    xx_single_particle_hamiltonian(L; J=1.0) -> Matrix{ComplexF64}

The `L x L` hopping matrix of the XX chain after Jordan-Wigner.

`J (Sx Sx + Sy Sy) = (J/2)(S+ S- + S- S+)` maps to `(J/2)(c'_j c_{j+1} + h.c.)`
for nearest neighbours -- the JW string cancels between adjacent sites -- so the
off-diagonal is `J/2` and the diagonal is zero.
"""
function xx_single_particle_hamiltonian(L::Int; J::Real = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    return h
end

"""
    xx_free_fermion_sz(L, t; J=1.0, occupied=1:(L÷2)) -> Vector{Float64}

`⟨Sz_j(t)⟩` for a Slater determinant evolved under the XX chain:

    ⟨Sz_j(t)⟩ = Σ_{k occupied} |[exp(-i h t)]_{jk}|² − ½

`occupied` lists the initially occupied sites. Up spin is an occupied fermion
(`Sz = n − ½`), so the default `1:(L÷2)` is exactly `domain_wall_state(L)`.
"""
function xx_free_fermion_sz(L::Int, t::Real; J::Real = 1.0,
                            occupied = 1:(L ÷ 2))
    U = exp(-im * t * xx_single_particle_hamiltonian(L; J = J))
    return [sum(abs2(U[j, k]) for k in occupied) - 0.5 for j in 1:L]
end
