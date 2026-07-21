# Dense references for validating the symmetric integrator.
#
# TEST-SIDE ONLY. The integrator itself must never densify -- that is what
# reintroduces forbidden amplitudes into a block-sparse state. These helpers
# exist so a 2^L exact answer can be compared against, and nothing here is
# reachable from src/.
#
# TWO DIFFERENT REFERENCES, AND CONFLATING THEM MAKES A TEST THAT CANNOT PASS.
# At L=6 with Dmax=8 the centre bond is already full rank (2^3 = 8), so BUG is
# EXACT and the only residual is the Trotter splitting. Machine precision is
# therefore reachable only against `dense_trotter_propagate`, which uses the very
# same odd/even split. Against `dense_exact_propagate` the error is O(dt^2) by
# construction, however good the integrator is.

using LinearAlgebra
using LurCGT, Telum
using BUGJulia.BondUpdateBUG

# ── site ordering ───────────────────────────────────────────────────────────
# Site 1 is the MOST significant index and local index 1 is spin up, so
# `op(o, i) = kron(I_{2^(i-1)}, o, I_{2^(L-i)})` and the bit pattern in
# `dense_state` agree by construction. Every convention here is pinned by
# `test_dense_reference.jl`, which cross-checks the dense energy against the
# MPS `energy(psi, gates)` -- that single assertion ties the operator ordering
# and the state ordering together, so neither can drift alone.

const _SP = ComplexF64[0 1; 0 0]
const _SM = ComplexF64[0 0; 1 0]
const _SZ = ComplexF64[0.5 0; 0 -0.5]

_embed(o, i, L) = kron(Matrix{ComplexF64}(I, 2^(i - 1), 2^(i - 1)), o,
                       Matrix{ComplexF64}(I, 2^(L - i), 2^(L - i)))

"""
    dense_bond_term(L, i; J=1.0, delta=1.0) -> Matrix{ComplexF64}

`J (Sx Sx + Sy Sy + delta Sz Sz)` on bond `(i, i+1)`, as a `2^L x 2^L` matrix.
The XY part is written `1/2 (S+ S- + S- S+)`, matching `heisenberg_bond_gate`.
"""
function dense_bond_term(L::Int, i::Int; J::Float64 = 1.0, delta::Float64 = 1.0)
    sp1, sm1, sz1 = _embed(_SP, i, L), _embed(_SM, i, L), _embed(_SZ, i, L)
    sp2, sm2, sz2 = _embed(_SP, i + 1, L), _embed(_SM, i + 1, L), _embed(_SZ, i + 1, L)
    return J * (0.5 * (sp1 * sm2 + sm1 * sp2) + delta * (sz1 * sz2))
end

"""
    dense_heisenberg(L; J=1.0, delta=1.0) -> Matrix{ComplexF64}

Open-boundary nearest-neighbour Heisenberg chain, `sum_b h_b`.
"""
function dense_heisenberg(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0)
    H = zeros(ComplexF64, 2^L, 2^L)
    for i in 1:(L - 1)
        H .+= dense_bond_term(L, i; J = J, delta = delta)
    end
    return H
end

"""
    dense_parity_hamiltonian(L, parity; J, delta) -> Matrix{ComplexF64}

The sum of one commuting bond group. Its terms act on disjoint site pairs, so
exponentiating this whole matrix is EXACT for the group -- no splitting error is
introduced inside a group, only between them.
"""
function dense_parity_hamiltonian(L::Int, parity::Symbol;
                                  J::Float64 = 1.0, delta::Float64 = 1.0)
    H = zeros(ComplexF64, 2^L, 2^L)
    for i in parity_bonds(L, parity)
        H .+= dense_bond_term(L, i; J = J, delta = delta)
    end
    return H
end

"`sum_j Sz_j` as a `2^L x 2^L` matrix."
dense_total_sz(L::Int) =
    sum(_embed(_SZ, j, L) for j in 1:L)

# ── state ───────────────────────────────────────────────────────────────────

"Contract a whole `SymMPS` into one tensor: (link_0, S_1 ... S_L, link_{L+1})."
function _full_tensor(psi::SymMPS)
    T = psi[1]
    for j in 2:length(psi)
        T = T * psi[j]
    end
    return to_concrete(T)
end

"""
    dense_state(psi::SymMPS) -> Vector{ComplexF64}

The `2^L` amplitude vector, with site 1 the most significant index and up first.

Computed as one overlap per computational basis configuration rather than by
flattening the tensor. That costs `2^L` cheap contractions, but the index is
built HERE from the spin pattern, so it cannot silently disagree with the
Kronecker ordering above -- which is exactly the class of convention bug that
a `reshape`/`vec` would hide.

A configuration of the wrong total charge has no shared boundary-link space with
`psi` at all, so it is skipped and left as an exact zero rather than contracted.
"""
function dense_state(psi::SymMPS)
    L = length(psi)
    P = _full_tensor(psi)
    v = zeros(ComplexF64, 2^L)
    for idx in 0:(2^L - 1)
        spins = [((idx >> (L - j)) & 1) == 0 ? :up : :down for j in 1:L]
        cfg = product_state(spins)
        psi[L].spaces[3] == cfg[L].spaces[3] || continue    # charge-forbidden
        v[idx + 1] = tensor_inner(_full_tensor(cfg), P)
    end
    return v
end

# ── propagators ─────────────────────────────────────────────────────────────

"""
    dense_exact_propagate(H, v, t) -> Vector{ComplexF64}

`exp(-i t H) v`. The unsplit reference: any Trotter scheme differs from this at
its own order, so a splitting integrator can never match it to machine precision.
"""
dense_exact_propagate(H::AbstractMatrix, v::AbstractVector, t::Real) =
    exp(-im * t * Matrix(H)) * v

"""
    dense_trotter_propagate(L, v, dt, n_steps; J, delta, order=:strang)

The reference the integrator CAN match to machine precision: the same odd/even
split, each group exponentiated exactly, applied in the order the sweep applies
them (even first, then odd; for `:strang`, even at half step on both sides).
"""
function dense_trotter_propagate(L::Int, v::AbstractVector, dt::Real, n_steps::Int;
                                 J::Float64 = 1.0, delta::Float64 = 1.0,
                                 order::Symbol = :strang)
    He = dense_parity_hamiltonian(L, :even; J = J, delta = delta)
    Ho = dense_parity_hamiltonian(L, :odd;  J = J, delta = delta)
    U = if order === :strang
        Uh = exp(-im * (dt / 2) * He)
        Uh * exp(-im * dt * Ho) * Uh        # rightmost acts first
    elseif order === :lie
        exp(-im * dt * Ho) * exp(-im * dt * He)
    else
        throw(ArgumentError("order must be :strang or :lie, got $order"))
    end
    w = ComplexF64.(v)
    for _ in 1:n_steps
        w = U * w
    end
    return w
end
