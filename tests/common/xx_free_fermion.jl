# tests/common/xx_free_fermion.jl
#
# Analytical reference for the nearest-neighbour XX Hamiltonian via exact diagonalization.
#
# The XX model is integrable and can be solved exactly via Jordan-Wigner mapping to
# free fermions, but for finite systems we compute the full Hamiltonian matrix and
# diagonalize it numerically to high precision (via LinearAlgebra.eigen).

using LinearAlgebra
using ITensors
using ITensorMPS

"""
    xx_hamiltonian_matrix(N; J = 1.0) -> Matrix{ComplexF64}

Construct the full Hamiltonian matrix for the XX chain on 2^N-dimensional Fock space.
H = (J/2) Σ_b (S⁺_b S⁻_{b+1} + S⁻_b S⁺_{b+1})

Basis ordering: computational (binary) with qubit ordering 1,2,...,N.
"""
function xx_hamiltonian_matrix(N::Int; J::Real = 1.0)
    d = 2^N
    H = zeros(ComplexF64, d, d)

    for b in 1:(N - 1)
        # S+ = [[0, 1], [0, 0]] → raises spin
        # S- = [[0, 0], [1, 0]] → lowers spin

        # S+ ⊗ S- at bonds b, b+1
        for s in 0:(d - 1)
            # Extract bits for sites b and b+1 (0-indexed)
            bit_b = (s >> (b - 1)) & 1
            bit_bp1 = (s >> b) & 1

            # S+ S- acts: |↓⟩_b |↑⟩_{b+1} → |↑⟩_b |↓⟩_{b+1}
            if bit_b == 0 && bit_bp1 == 1
                # Create a new state by flipping these bits
                s_new = xor(s, (1 << (b - 1)) | (1 << b))
                H[s_new + 1, s + 1] += (J / 2)
            end

            # S- S+ acts: |↑⟩_b |↓⟩_{b+1} → |↓⟩_b |↑⟩_{b+1}
            if bit_b == 1 && bit_bp1 == 0
                s_new = xor(s, (1 << (b - 1)) | (1 << b))
                H[s_new + 1, s + 1] += (J / 2)
            end
        end
    end

    return H + H'  # Ensure Hermiticity (should already be by construction)
end

"""
    xx_ground_state(N; J = 1.0) -> (E0, psi0_vec)

Ground state energy and wavefunction vector (dense) for the XX chain via exact diagonalization.
"""
function xx_ground_state(N::Int; J::Real = 1.0)
    H = xx_hamiltonian_matrix(N; J = J)
    F = eigen(H)

    # Ground state = smallest eigenvalue
    idx = argmin(F.values)
    E0 = F.values[idx]
    psi0 = F.vectors[:, idx]

    return E0, psi0
end

"""
    xx_time_evolved_state(psi0::Vector, H::Matrix, t::Real) -> Vector

Evolve initial state psi0 under Hamiltonian H to time t: ψ(t) = exp(-iHt) ψ0
Uses exact eigendecomposition: ψ(t) = Σ_n exp(-iE_n t) ⟨n|ψ0⟩ |n⟩
"""
function xx_time_evolved_state(psi0::Vector{ComplexF64}, H::Matrix{ComplexF64}, t::Real)
    F = eigen(H)  # F.values = eigenvalues, F.vectors = eigenvectors (columns)

    # psi0 in eigenbasis
    coeffs = F.vectors' * psi0

    # Apply exp(-i E_n t)
    evolved_coeffs = exp.((-im * t) .* F.values) .* coeffs

    # Back to original basis
    return F.vectors * evolved_coeffs
end

"""
    xx_observable_expectation(psi::Vector, observable::Matrix) -> Complex

Compute expectation value ⟨psi|O|psi⟩ for a dense vector state and observable matrix.
"""
function xx_observable_expectation(psi::Vector{ComplexF64}, observable::Matrix{ComplexF64})
    return real(psi' * observable * psi)
end

"""
    xx_mps_to_vector(psi::MPS) -> Vector{ComplexF64}

Convert an ITensors MPS to a dense vector in computational basis (for small systems only).
"""
function xx_mps_to_vector(psi::MPS)
    N = length(psi)
    2^N > 1_000_000 && error("xx_mps_to_vector: system too large (>20 sites)")

    # Start with |0⟩ ⊗ |0⟩ ⊗ ... ⊗ |0⟩
    d = 2^N
    vec = zeros(ComplexF64, d)
    vec[1] = 1.0  # |0⟩

    # Contract MPS with computational basis
    for n in 1:N
        vec_new = zeros(ComplexF64, d)
        A = psi[n]

        for s in 0:(2^n - 1)
            for sn in 0:1
                s_new = s | (sn << (n - 1))
                # This is a simplified contraction; for production use
                # specialized ITensor contraction. For testing on N ≤ 6 OK.
                idx_old = s + 1
                idx_new = s_new + 1
                for r_left in 1:size(A)[1]
                    for r_right in 1:size(A)[3]
                        # Approximation: use element [r_left, sn+1, r_right]
                        # This is overly simplified; return error for now
                    end
                end
            end
        end
    end

    error("xx_mps_to_vector not fully implemented; use ITensor native contraction instead")
end

"""
    xx_magnetization_z(psi::Vector, site::Int) -> Real

Compute local Z magnetization (⟨Z⟩ = ⟨S_z⟩ = (n_↑ - n_↓)/2) at a single site.
"""
function xx_magnetization_z(psi::Vector{ComplexF64}, site::Int, N::Int)
    # S_z = 0.5 * Z where Z is Pauli Z
    # In computational basis: Z |s⟩ = (-1)^s |s⟩

    Sz_matrix = zeros(ComplexF64, 2^N, 2^N)
    for s in 0:(2^N - 1)
        bit = (s >> (site - 1)) & 1
        sz_val = 0.5 * (1 - 2 * bit)  # +0.5 if bit=0 (spin up), -0.5 if bit=1
        Sz_matrix[s + 1, s + 1] = sz_val
    end

    return xx_observable_expectation(psi, Sz_matrix)
end

"""
    xx_fidelity(psi1::Vector, psi2::Vector) -> Real

Fidelity |⟨psi1|psi2⟩| between two quantum states.
"""
function xx_fidelity(psi1::Vector{ComplexF64}, psi2::Vector{ComplexF64})
    return abs(dot(psi1, psi2)) / (norm(psi1) * norm(psi2))
end
