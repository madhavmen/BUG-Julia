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
    zk_vector(L, idx, k) -> Vector{ComplexF64}

`|Z_k>` in the constrained basis; `k = 0` is the polarised vacuum `|∘∘∘…>`. The ED partner of
[`zk_state`](@ref), and it must agree with it bit for bit -- `|•> = up` at sites `1, 1+k, 1+2k, …`.

⚠ THE `haskey` CHECK IS THE CONVENTION GATE. If the MPS side and this side ever disagree about
which sites are excited, the pattern simply is not in the constrained basis and this errors,
rather than silently grading the integrator against a DIFFERENT initial state -- which would show
up as a plausible-looking disagreement and be blamed on the integrator.
"""
function zk_vector(L::Int, idx::Dict{Int, Int}, k::Int)
    k == 1 && throw(ArgumentError("zk_vector: k=1 violates the blockade"))
    b = k == 0 ? 0 : sum(1 << (j - 1) for j in 1:k:L)
    haskey(idx, b) ||
        error("the Z$k pattern is not in the constrained basis -- convention slip")
    v = zeros(ComplexF64, length(idx))
    v[idx[b]] = 1.0
    return v
end

"""
    pxp_dim(L) -> Int

Size of the Rydberg-blockade basis, `Fib(L+2)` with `Fib(1)=Fib(2)=1`.

⛔ THIS IS WHY THE PXP EXACT REFERENCE REACHES MUCH FURTHER THAN THE SPIN ONE, and why gating it
on `L` alone throws away most of the paper's range. The constrained space grows as the GOLDEN
RATIO (~1.618^L), not as `2^L`: at `L = 28` it is 832,040 states against `2^28 = 2.7e8`, a factor
of 323. Grading PXP by `L <= ED_MAX = 20` (a limit set by the MBL model's `C(L, L/2)` sector)
would refuse an exact reference at sizes where it is entirely affordable.
"""
function pxp_dim(L::Int)
    a, b = 1, 1                       # Fib(1), Fib(2)
    for _ in 1:L
        a, b = b, a + b
    end
    return b                          # Fib(L+2)
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

# ── THE NON-INTERACTING AUBRY-ANDRÉ CHAIN: EXACT AT ANY `L`, NO ED ───────────────────────────
#
# WHY THIS IS THE MOST VALUABLE REFERENCE IN THE FILE. At `U = 0` the model of arXiv:1501.05661 is
# QUADRATIC, so its dynamics is an `L x L` matrix exponential and the largest object here is
# `L x L` -- never `2^L`. That removes the scaling wall entirely: the `U = 0` arm can be graded
# EXACTLY at `L = 40` (the size the paper's own DMRG uses) or `L = 100`, where no ED exists.
#
# It also carries a PARAMETER-FREE anchor. The Aubry-André model is self-dual at `Delta = 2J`:
# below it every single-particle state is extended, above it every one is localised. So the
# stationary imbalance of Fig. 3 must rise from ~0 to finite AT `Delta/J = 2`, with no fitting.
# That is a published number our curve either hits or does not.
#
# ⚠ WHAT IT CANNOT DO: `U = 0` is ANDERSON-type (single-particle) localisation in a quasiperiodic
# potential. It is the CONTROL, not the physics -- there is no many-body localisation without
# interactions, and the rank a `U = 0` state needs stays bounded, so it is also the arm least able
# to discriminate between integrators. Use it to prove the implementation, not to prove the method.

"""
    aa_hopping(L; J = 1.0, Delta, beta = 0.721, phi = 0.0) -> Matrix{Float64}

The single-particle Hamiltonian of the OPEN Aubry-André chain in EXPERIMENT units:
`-J` on the off-diagonal, `Delta cos(2πβi + φ)` on the diagonal. Same `β = 0.721` and same
`Delta/J` convention as [`aubry_andre_xxz_mpo`](@ref), so a `Delta/J` quoted here and a `Delta/J`
handed to the MPO mean the same thing -- which is the factor-of-2 trap that builder documents.
"""
function aa_hopping(L::Int; J::Float64 = 1.0, Delta::Float64,
                    beta::Float64 = 0.721, phi::Float64 = 0.0)
    h = zeros(Float64, L, L)
    for i in 1:(L - 1)
        h[i, i + 1] = -J
        h[i + 1, i] = -J
    end
    for i in 1:L
        h[i, i] = Delta * cos(2π * beta * i + phi)
    end
    return h
end

"""
    aa_free_fermion(L, t; J, Delta, beta, phi, occupied, nl) -> (; n, imbalance, entropy)

`⟨n_j(t)⟩`, the imbalance and the half-chain entanglement entropy of the CDW quench of the
NON-INTERACTING Aubry-André chain, from one `L x L` matrix exponential.

`occupied` defaults to the ODD sites, matching [`neel_state`](@ref) on the MPS side.

⚠ THE PAPER FILLS THE EVEN SITES AND WE FILL THE ODD ONES. That is a relabelling with no physical
content -- it flips the sign of `I` and nothing else -- and it is safe here only BECAUSE the
potential is incommensurate: `β = 0.721` has period `1/β ≈ 1.386`, so it puts no systematic bias
on either sublattice. Under a COMMENSURATE potential (`β = 1/2`, say) the two choices would be
physically different and the sign convention would stop being a convention.

THE ENTROPY IS THE SLATER-DETERMINANT FORMULA: eigenvalues `ν` of the correlation matrix
restricted to the block give `S = -Σ [ν ln ν + (1-ν) ln(1-ν)]`, in NATS to match
[`entropy_from_svals`](@ref). It equals the SPIN chain's entropy because Jordan-Wigner is local
on a CONTIGUOUS block -- which is exactly why `nl` must be a contiguous left block and not an
arbitrary site subset.
"""
function aa_free_fermion(L::Int, t::Float64; J::Float64 = 1.0, Delta::Float64,
                         beta::Float64 = 0.721, phi::Float64 = 0.0,
                         occupied::Vector{Int} = collect(1:2:L), nl::Int = L ÷ 2)
    h = aa_hopping(L; J = J, Delta = Delta, beta = beta, phi = phi)
    Phi = exp(-im * t * h)[:, occupied]         # evolved occupied orbitals, L x N
    n = [sum(abs2, @view Phi[j, :]) for j in 1:L]
    # Same expression as `imbalance_dense`, with `⟨S^z_j⟩ = n_j - 1/2`.
    imb = (2 / L) * sum((isodd(j) ? 1 : -1) * (n[j] - 0.5) for j in 1:L)
    A = @view Phi[1:nl, :]
    nu = real.(eigvals(Hermitian(A * A')))
    S = 0.0
    for v in nu
        v = clamp(v, 0.0, 1.0)
        (v > 1e-12 && v < 1 - 1e-12) && (S -= v * log(v) + (1 - v) * log(1 - v))
    end
    return (; n = n, imbalance = imb, entropy = S)
end

# ── WHAT WOULD BE NEEDED FOR THE PAPER'S INTERACTING FIGURES ─────────────────────────────────
#
# arXiv:1501.05661 Fig. 4 (the `U/J` vs `Delta/J` map, the "W" shape) and Fig. 5 (entropy growth
# resolved by `U/J`) are SPINFUL: `U Σ_i n_{i↑} n_{i↓}`, an ON-SITE interaction between two
# species. Spinless fermions cannot have one -- `n_i² = n_i` -- so no parameter choice of the
# spinless model in this file reproduces those two figures. What it does reproduce is Fig. 2
# (`I(t)` vs `Delta/J`) and Fig. 3 (non-interacting, where `U` is absent by construction).
#
# THE ROUTE, recorded so the size of the job is not underestimated later. Two species on `L` sites
# = one spin chain on `2L` sites, and the ORDERING decides the cost:
#
#   INTERLEAVED  (↑₁ ↓₁ ↑₂ ↓₂ …)  `U n_{i↑}n_{i↓}` is NEAREST-NEIGHBOUR `S^z S^z`, but a hop
#                                  ↑ᵢ → ↑ᵢ₊₁ jumps over ↓ᵢ, so its Jordan-Wigner string makes it
#                                  a THREE-SITE term `S⁺ σ^z S⁻`. Bond dimension stays O(1), and
#                                  the three-site automaton is exactly the shape `pxp_mpo` already
#                                  builds (`1→2→3→4`, no self-loops).
#   BLOCKED      (↑₁…↑_L ↓₁…↓_L)   every hop is nearest-neighbour and string-free, but `U` becomes
#                                  a distance-`L` `S^z S^z` coupling. `pair_mpo` represents that
#                                  EXACTLY for an arbitrary `J[i,j]`, at the cost of one channel
#                                  per live opening, i.e. `w ~ L`.
#
# Interleaved is the right choice: O(1) bond dimension, and the only new object is a three-site
# vertex. It needs a field on `pair_mpo`/`mpo_from_terms` (the same `id -> done` slot
# `field_mpo_from_terms` uses) and the `σ^z` string operator, which is `2 S^z` and already exists.

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
