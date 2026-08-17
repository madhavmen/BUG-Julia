# THE BENCHMARK MODELS: SU(2) HALDANE-SHASTRY ON A RING, AND HEISENBERG.
#
# Both are built from ONE coupling matrix handed to `pair_mpo`, and the SPIN VERTICES are read
# off the symmetry mode in force. That is the whole point of the arrangement: the operator is
# specified once, as `Σ_{i<j} J[i,j] S_i . S_j`, and `:none`, `:U1` and `:SU2` differ only in
# how many vertices the algebra needs to express `S . S` -- three (`Sp Sm`, `Sm Sp`, `Sz Sz`)
# for the abelian modes, ONE irreducible vertex for SU(2). Same operator, same physics,
# different virtual dimension. Nothing else about the models is symmetry dependent.
#
# HALDANE-SHASTRY, as written in arXiv:2208.10972 (the CBE-TDVP paper, which is where this
# benchmark comes from):
#
#     H_HS = J Σ_{l<l'} [ pi^2 / ( L^2 sin^2( pi (l - l') / L ) ) ] S_l . S_l'
#
# THE RING IS ALREADY IN THE COUPLING, and this is worth being explicit about because it looks
# like something is missing. There is no wrap-around term to add on top: `sin^2` is periodic,
# so the pair `(1, L)` -- nominally the most distant pair on an open chain -- has
# `sin^2(pi (L-1)/L) = sin^2(pi/L)`, i.e. it is as STRONG as a nearest-neighbour pair. The
# chord-distance geometry of the ring is what the coupling matrix says. The MPS itself stays
# an open chain; it is the Hamiltonian that closes the ring, which is exactly why an exact
# all-pairs MPO is needed rather than a nearest-neighbour one.
#
# WHY `pair_mpo` AND NOT AN EXPONENTIAL FIT. `1/sin^2` has no finite-dimensional MPO, so a
# self-loop construction would have to fit it, and the fit error is a property of the
# HAMILTONIAN being simulated -- it is not separable from a result afterwards, and quoting an
# integrator error against a fitted `H` quotes an unstated model. `pair_mpo` spends `O(L)`
# virtual dimension and is EXACT, so every number measured against it is an integrator number.

# ── the spin vertices of the mode in force ───────────────────────────────────

"""
    spin_vertices(L) -> Vector{PairVertex}

The vertices of `S_i . S_j` in whatever symmetry mode is active: the three abelian terms
under `:none`/`:U1`, the single irreducible term under `:SU2`.

Read off the SAME term-list constructors the nearest-neighbour Hamiltonians use
(`xxz_chain`, `heisenberg_su2_chain`), so a long-range model and a nearest-neighbour one
cannot disagree about Telum's operator normalisation -- under SU(2) the generators carry
Clebsch-Gordan factors, and having one source for them is what keeps the overall constant
pinned by the existing `dimer_state` energy test.
"""
spin_vertices(L::Int) = symmetry_mode() === :SU2 ?
    pair_vertices(heisenberg_su2_chain(L; J = 1.0)) :
    pair_vertices(xxz_chain(L; J = 1.0, delta = 1.0))

# ── Haldane-Shastry ──────────────────────────────────────────────────────────

"""
    haldane_shastry_couplings(L; J=1.0) -> Matrix{Float64}

`J[l,l'] = J * pi^2 / ( L^2 sin^2( pi (l - l') / L ) )` on the strict upper triangle, the
Haldane-Shastry ring coupling of arXiv:2208.10972.

Only the strict upper triangle is filled, which is what `pair_mpo` reads, so no pair is
counted twice. The diagonal is zero by construction (`sin(0)` would diverge, and there is no
`l = l'` term in the sum).
"""
function haldane_shastry_couplings(L::Int; J::Float64 = 1.0)
    L >= 2 || throw(ArgumentError("haldane_shastry_couplings needs L >= 2, got $L"))
    Jm = zeros(Float64, L, L)
    for i in 1:(L - 1), j in (i + 1):L
        s = sin(pi * (i - j) / L)
        Jm[i, j] = J * pi^2 / (L^2 * s^2)
    end
    return Jm
end

"""
    haldane_shastry_mpo(L; J=1.0) -> MPO

The SU(2) Haldane-Shastry ring as an exact MPO, in whatever symmetry mode is active.

Virtual dimension is `2 + n*(live openings)` with `n = 1` under SU(2) and `n = 3` under U(1),
so roughly `L` against `3L`. Both represent the same operator exactly -- `test_pair_mpo.jl`
asserts they give the same energy on the same physical state, which is the cross-symmetry
claim, and the SU(2) saving is then a cost statement rather than a physics one.
"""
haldane_shastry_mpo(L::Int; J::Float64 = 1.0) =
    pair_mpo(L, haldane_shastry_couplings(L; J = J), spin_vertices(L))

# ── Heisenberg, open chain and ring ──────────────────────────────────────────

"""
    heisenberg_chain_mpo(L; J=1.0) -> MPO

`H = J Σ_i S_i . S_{i+1}` on an OPEN chain, built through [`pair_mpo`](@ref) rather than
[`mpo_from_terms`](@ref).

Deliberately redundant with `mpo_from_terms`: the two must be the same operator, and that
identity is the pin on the `pair_mpo` automaton against the already-validated builder. Use
`mpo_from_terms` in production for a nearest-neighbour chain -- its virtual dimension is
constant in `L`.
"""
heisenberg_chain_mpo(L::Int; J::Float64 = 1.0) =
    pair_mpo(L, J * nn_coupling_matrix(L), spin_vertices(L))

"""
    heisenberg_ring_couplings(L; J=1.0) -> Matrix{Float64}

Nearest neighbours on a RING: `J[i,i+1] = J` and the closing bond `J[1,L] = J`.

The closing bond is what makes this a ring, and it is a genuinely long-range MPO term -- it
reaches from site 1 to site `L`, so its channel is open across the entire chain. That single
channel is the cheapest possible exercise of the charged transport block over the maximum
distance, which makes this model a useful test in its own right and not just a physics
target.
"""
function heisenberg_ring_couplings(L::Int; J::Float64 = 1.0)
    L >= 3 || throw(ArgumentError("a ring needs at least three sites, got $L"))
    Jm = J * nn_coupling_matrix(L)
    Jm[1, L] += J
    return Jm
end

"""
    heisenberg_ring_mpo(L; J=1.0) -> MPO

`H = J Σ_{i=1}^{L} S_i . S_{i+1}` with `S_{L+1} = S_1`: the periodic Heisenberg ring.
"""
heisenberg_ring_mpo(L::Int; J::Float64 = 1.0) =
    pair_mpo(L, heisenberg_ring_couplings(L; J = J), spin_vertices(L))
