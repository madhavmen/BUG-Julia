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

# ── TFIM, the same idea one level up: BdG rather than a hopping matrix ────────────────────────
#
# `H = -J Σ Z_j Z_{j+1} - h Σ X_j` is quadratic in MAJORANAS, not in fermions -- the ZZ term
# pairs as well as hops -- so the closed form runs through the 2L x 2L covariance matrix instead
# of an L x L propagator. Still no 2^L object anywhere, which is the whole point: the integrator
# tests can then run at the L where a defect is actually LARGE rather than at whatever L a dense
# propagator happens to fit in.
#
# ⛔ THE BASIS IS σˣ: `X` is DIAGONAL and `Z` is the FLIP. Reading it the usual way describes a
# different Hamiltonian -- see the header of `src/RSVDCBEBondUpdate/models/tfim.jl`.

"""
    tfim_majorana_generator(L; J=1.0, h=1.0) -> Matrix{Float64}

The real antisymmetric `2L x 2L` generator `A` of the TFIM's Majorana flow.

Writing `H = (i/4) Σ A_{mn} a_m a_n` in Majoranas `a_{2j-1}, a_{2j}`, the field `-h X_j` is the
on-site pair `(2j-1, 2j)` and the coupling `-J Z_j Z_{j+1}` is the bond pair `(2j, 2j+1)`. The
factors of two are the Majorana convention and are pinned against exact diagonalisation in
`test_analytic_reference.jl` rather than asserted here.
"""
function tfim_majorana_generator(L::Int; J::Real = 1.0, h::Real = 1.0)
    A = zeros(Float64, 2L, 2L)
    for j in 1:L
        A[2j - 1, 2j] = -2h; A[2j, 2j - 1] = +2h
    end
    for j in 1:(L - 1)
        A[2j, 2j + 1] = -2J; A[2j + 1, 2j] = +2J
    end
    return A
end

"""
    tfim_covariance(L, t, dirs; J=1.0, h=1.0) -> Matrix{Float64}

The Majorana covariance `Γ(t)` of a σˣ PRODUCT state evolved under the TFIM.

`dirs[j]` is `:plus` or `:minus`. Such a state is Gaussian, so it is fully described by
`Γ_{2j-1,2j} = ±1`, and Heisenberg evolution is the congruence

    Γ(t) = R Γ(0) Rᵀ,   R = exp(A t)

Cost is `O(L³)`, independent of any bond dimension. THIS is the primitive: `⟨X_j⟩`, the half-chain
entropy and the connected correlators all read off the SAME `Γ`, so anything needing more than the
profile should take the matrix rather than grow a second copy of this congruence.
"""
function tfim_covariance(L::Int, t::Real, dirs::Vector{Symbol}; J::Real = 1.0, h::Real = 1.0)
    length(dirs) == L || throw(DimensionMismatch("dirs has $(length(dirs)) entries, need $L"))
    G = zeros(Float64, 2L, 2L)
    for (j, d) in enumerate(dirs)
        s = d === :plus ? 1.0 : -1.0
        G[2j - 1, 2j] = s; G[2j, 2j - 1] = -s
    end
    R = exp(tfim_majorana_generator(L; J = J, h = h) * t)
    return R * G * transpose(R)
end

"""
    tfim_x_profile(L, t, dirs; J=1.0, h=1.0) -> Vector{Float64}

`⟨X_j(t)⟩` for a σˣ product state evolved under the TFIM -- the diagonal pair entries of
[`tfim_covariance`](@ref).
"""
function tfim_x_profile(L::Int, t::Real, dirs::Vector{Symbol}; J::Real = 1.0, h::Real = 1.0)
    Gt = tfim_covariance(L, t, dirs; J = J, h = h)
    return [Gt[2j - 1, 2j] for j in 1:L]
end
