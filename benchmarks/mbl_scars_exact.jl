# THE ED REFERENCE FOR THE TWO ERGODICITY-BREAKING MODELS -- SYMMETRY-REDUCED, AND BY SPARSE
# KRYLOV RATHER THAN BY DIAGONALISATION.
#
# Requires `exact_sparse.jl` to have been included first: `sz0_basis`, `expv_sparse` and
# `magnetisation_dense` come from there, along with the reason the propagation is Krylov and not
# `eigen` (real time needs `exp(-iHt)|ψ₀⟩`, not the spectrum, and the vector is affordable at
# sizes where the matrix is not) and the SUBSTEPPING that makes the reference detect its own
# failure instead of returning a confidently wrong vector.
#
# ── EXPLOITING THE SYMMETRY IS WHAT PUTS THESE SIZES IN REACH ────────────────────────────────
#
# Both models have a symmetry, they are DIFFERENT symmetries, and each is worth a large factor:
#
#   MBL / disordered XXZ   TOTAL S^z is conserved -- a random longitudinal field commutes with
#                          `Σ S^z_i` exactly -- so the Néel quench never leaves `S^z = 0`:
#                          `C(16,8) = 12870` states instead of `65536`, a 5.1x cut, and `C(20,10)
#                          = 184756` instead of `1048576`, 5.7x. This is the SAME U(1) the MPS
#                          side runs under, so the two sides exploit one symmetry, not two.
#
#   PXP                    NO continuous symmetry at all (`σ^x` conserves nothing), but the
#                          BLOCKADE CONSTRAINT removes every configuration with two adjacent
#                          excitations, leaving `Fib(L+2)` of `2^L`: at `L = 20` that is `17711`
#                          instead of `1048576`, a 59x cut, and it is the only reason the scar
#                          literature reaches `L = 32`. The constrained space is invariant under
#                          `H` (a flip needs both neighbours empty, so no allowed move can create
#                          an adjacent pair), which is what makes restricting to it EXACT rather
#                          than an approximation.
#
# ⚠ AND THE MPS SIDE CANNOT USE PXP's REDUCTION. Block sparsity in an MPS comes from a LOCAL
# ABELIAN CHARGE; the Fibonacci constraint is a local constraint but not a group, and PXP's
# remaining symmetries -- spatial inversion, and the sublattice operator that ANTI-commutes with
# `H` -- are neither local nor charge-like. So the `pxp` arm runs `:none` by necessity while the
# `mbl` arm runs `:U1` natively. That asymmetry is a fact about the models and is reported as
# such; it is not a knob that was left unset.

using LinearAlgebra, SparseArrays

# ── MBL: the disordered XXZ chain in the S^z = 0 sector ──────────────────────────────────────

"""
    disordered_xxz_sparse(L, hz; J = 1.0, delta = 1.0) -> (H, states, idx)

`H = J Σ (S^x S^x + S^y S^y + Δ S^z S^z) + Σ hz[i] S^z_i` on an open chain, sparse, in the
`S^z = 0` sector -- the same operator [`disordered_xxz_mpo`](@ref) builds as an MPO.

`heisenberg_sparse` with the field added to the diagonal. The field is DIAGONAL in the bit basis
(`S^z_i` is), so disorder costs nothing in sparsity and nothing in the sector structure: it is
exactly why the U(1) reduction survives the disorder that destroys integrability.

⚠ SITE CONVENTION, and it is `exact_sparse.jl`'s: bit `j-1` set means site `j` is UP. `hz[j]`
therefore multiplies `+1/2` on a set bit. This is the same indexing `magnetisation_dense` uses,
so `hz` here and `hz` in the MPO are the same vector read the same way round -- but see
`sparse_to_dense_index` for the MIRRORED convention `dense_state` uses, which nothing in this
file touches.
"""
function disordered_xxz_sparse(L::Int, hz::Vector{Float64};
                               J::Float64 = 1.0, delta::Float64 = 1.0)
    length(hz) == L || throw(DimensionMismatch("got $(length(hz)) fields for $L sites"))
    states, idx = sz0_basis(L)
    n = length(states)
    I, Jd, V = Int[], Int[], Float64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states)
        diag = 0.0
        for j in 1:(L - 1)
            sj, sj1 = bit(b, j), bit(b, j + 1)
            diag += J * delta * (sj == sj1 ? 0.25 : -0.25)
            if sj != sj1
                bb = b ⊻ (1 << (j - 1)) ⊻ (1 << j)
                push!(I, idx[bb]); push!(Jd, k); push!(V, J / 2)
            end
        end
        for j in 1:L
            diag += hz[j] * (bit(b, j) - 0.5)
        end
        push!(I, k); push!(Jd, k); push!(V, diag)
    end
    return sparse(I, Jd, V, n, n), states, idx
end

"""
    neel_energy_closed(L, hz; delta = 1.0) -> Float64

`⟨Néel|H|Néel⟩` IN CLOSED FORM, for the `t = 0` gate that catches a wrong Hamiltonian before it
becomes the thing everything is measured against.

On a definite-`S^z` product state the XY terms have zero expectation, so only `S^zS^z` and the
field survive. All `L-1` bonds of `|↑↓↑↓…⟩` are anti-aligned, and site `j` carries
`(-1)^{j+1}/2`:

    E = -(L-1)Δ/4  +  (1/2) Σ_j (-1)^{j+1} hz[j]

⛔ THIS IS THE ONLY NON-VACUOUS `t = 0` CHECK AVAILABLE HERE, and that is why it carries the
disorder term explicitly. A check that only tested `-(L-1)Δ/4` would pass identically for
`hz = 0`, for a field applied to the wrong sites, and for a field silently dropped at the two
boundary sites -- which is the exact failure `field_mpo_from_terms` documents.
"""
neel_energy_closed(L::Int, hz::Vector{Float64}; delta::Float64 = 1.0) =
    -(L - 1) * delta / 4 + sum((isodd(j) ? 0.5 : -0.5) * hz[j] for j in 1:L)

"""
    imbalance_dense(v, L, states) -> Float64

THE OBSERVABLE OF THE BLOCH-GROUP EXPERIMENT (RMP §II), from a sector vector:

    I(t) = (N_odd - N_even) / (N_odd + N_even) = (2/L) Σ_j (-1)^{j+1} ⟨S^z_j⟩

with `n_j = S^z_j + 1/2`. The `1/2`s cancel in the staggered sum for even `L`, and `Σ n_j = L/2`
in the half-filled sector -- so the normalisation is exact rather than measured, and `I(0) = 1`
for the Néel state. `I` stays near 1 in the MBL phase and decays to 0 in the ergodic one, which
is the single number the whole phase diagram is read off.
"""
function imbalance_dense(v::Vector{ComplexF64}, L::Int, states::Vector{Int})
    sz = magnetisation_dense(v, L, states)
    return (2 / L) * sum((isodd(j) ? 1 : -1) * sz[j] for j in 1:L)
end

# ── SCARS: PXP in the blockade-constrained basis ─────────────────────────────────────────────

"""
    pxp_basis(L) -> (states, idx)

The BLOCKADE-CONSTRAINED basis: every bit pattern with no two adjacent set bits, i.e. no two
neighbouring Rydberg excitations. `|Fib(L+2)|` states (2, 3, 5, 8, … for `L = 1, 2, 3, 4`).

Built by EXTENSION rather than by filtering `0:2^L-1`, which matters at the sizes this exists
for: the filter is `2^L` tests (67 M at `L = 26`) to keep 200 k of them, while the extension
touches only states that are already legal.

Site `j` is bit `j-1` and a SET bit is an EXCITED atom (`|•⟩`), matching `exact_sparse.jl`'s
convention and the `|•⟩ = up` convention of [`pxp_operators`](@ref).
"""
function pxp_basis(L::Int)
    L >= 1 || throw(ArgumentError("pxp_basis needs at least one site, got $L"))
    states = Int[0]
    for i in 1:L
        nxt = Int[]
        for b in states
            push!(nxt, b)                                       # site `i` left unexcited
            # Excite `i` only if `i-1` is empty; `i+1` is handled when it is reached.
            (i == 1 || ((b >> (i - 2)) & 1) == 0) && push!(nxt, b | (1 << (i - 1)))
        end
        states = nxt
    end
    sort!(states)
    return states, Dict(s => k for (k, s) in enumerate(states))
end

"""
    pxp_sparse(L; Omega = 1.0) -> (H, states, idx)

`H = Ω Σ_i P_{i-1} X_i P_{i+1}` in the constrained basis -- the same operator [`pxp_mpo`](@ref)
builds, and the operator whose `L = 32` ED gives arXiv:2011.09486 Fig. 2.

PURELY OFF-DIAGONAL: `H` flips one atom and the projectors are diagonal, so the matrix element
between `b` and `b ⊻ 2^{i-1}` is `Ω` whenever both neighbours of `i` are empty in `b`, and `0`
otherwise. Nothing lands outside the basis: exciting `i` requires empty neighbours, and
de-exciting can only reduce the excitation count.

⛔ THE OFF-DIAGONALITY IS ALSO WHY `⟨ψ|H|ψ⟩ = 0` FOR EVERY BASIS STATE, `|Z_2⟩` INCLUDED. So an
energy check at `t = 0` is VACUOUS -- `0` compared against `0` -- and cannot distinguish this
Hamiltonian from a wrong one, or from none at all. The gate that does work is the `smoke` phase
of `benchmarks/mbl_scars.jl`: evolve at FULL RANK, where every integrator is exact, and require
agreement with this propagation to `1e-10`. Energy conservation is then a real check at `t > 0`,
since a truncating run does drift.
"""
function pxp_sparse(L::Int; Omega::Float64 = 1.0)
    L >= 3 || throw(ArgumentError("pxp_sparse needs at least three sites, got $L"))
    states, idx = pxp_basis(L)
    n = length(states)
    I, Jd, V = Int[], Int[], Float64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states)
        for i in 1:L
            # The blockade projectors, evaluated on the configuration: both neighbours empty.
            (i == 1 || bit(b, i - 1) == 0) || continue
            (i == L || bit(b, i + 1) == 0) || continue
            bb = b ⊻ (1 << (i - 1))
            push!(I, idx[bb]); push!(Jd, k); push!(V, Omega)
        end
    end
    return sparse(I, Jd, V, n, n), states, idx
end

"""
    z2_vector(L, idx) -> Vector{ComplexF64}

`|Z_2⟩ = |•∘•∘…⟩` in the constrained basis: odd sites excited, a single basis state, so a unit
column. Matches [`z2_state`](@ref) on the MPS side (`|•⟩ = up`, odd sites up).
"""
function z2_vector(L::Int, idx::Dict{Int, Int})
    b = sum(1 << (j - 1) for j in 1:2:L)
    haskey(idx, b) || error("the Z2 pattern is not in the constrained basis -- convention slip")
    v = zeros(ComplexF64, length(idx))
    v[idx[b]] = 1.0
    return v
end

"""
    sz_profile_dense(v, L, states) -> Vector{Float64}

`⟨S^z_j⟩` per site from a vector in ANY bit-pattern basis (the `S^z=0` sector or the constrained
PXP one) -- `magnetisation_dense` generalised off the half-filled sector, which it is already,
but naming it here keeps the PXP call sites from reading as if they were in a spin sector.
`⟨n_j⟩ = ⟨S^z_j⟩ + 1/2`.
"""
sz_profile_dense(v::Vector{ComplexF64}, L::Int, states::Vector{Int}) =
    magnetisation_dense(v, L, states)

# ── the entanglement entropy, for both models from one function ──────────────────────────────

"""
    entropy_from_svals(s) -> Float64

`S = -Σ p log p` with `p = s²/Σs²`, in NATS. One convention, stated once: the MPS side reads its
`s` off [`bond_spectrum`](@ref) and the ED side off an SVD of the reshaped vector, and a log base
that differs between the two shows up as a constant factor -- which on a LOG-GROWTH plot reads as
a different slope, i.e. as different physics.
"""
function entropy_from_svals(s::AbstractVector)
    p = abs2.(s)
    tot = sum(p)
    tot > 0 || return 0.0
    acc = 0.0
    for q in p ./ tot
        q > 1e-16 && (acc -= q * log(q))
    end
    return acc
end

"""
    half_chain_entropy_sector(v, L, states; nl = L ÷ 2) -> Float64

The exact half-chain entanglement entropy of a vector given in a BIT-PATTERN basis.

Serves both models unchanged, which is the point: the left block is sites `1:nl`, and in this
file's convention those are the LOW bits, so a plain reshape of the embedded vector into
`2^nl × 2^(L-nl)` IS the Schmidt matrix -- no permutation, and no dependence on which basis
subset `states` happens to be (`S^z = 0`, blockade-constrained, or complete).

⚠ COST. The matrix is `2^nl × 2^(L-nl)` DENSE regardless of how sparse the basis is: 256² at
`L = 16`, 1024² at `L = 20`, 4096² (268 MB, and an SVD to match) at `L = 24`. The caller decides
whether to ask; the campaign asks only up to `L = 20`.
"""
function half_chain_entropy_sector(v::Vector{ComplexF64}, L::Int, states::Vector{Int};
                                   nl::Int = L ÷ 2)
    1 <= nl < L || throw(ArgumentError("bipartition needs 1 <= nl < L, got nl=$nl L=$L"))
    M = zeros(ComplexF64, 1 << nl, 1 << (L - nl))
    mask = (1 << nl) - 1
    for (k, b) in enumerate(states)
        M[(b & mask) + 1, (b >> nl) + 1] = v[k]
    end
    return entropy_from_svals(svdvals(M))
end
