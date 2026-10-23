# Ergodicity-breaking models: disordered / quasiperiodic XXZ (MBL, arXiv:1804.11065) and PXP scars
# (arXiv:2011.09486). Both quench from a product state and are graded against exact diagonalisation.

# ── MBL: the disordered XXZ chain (RMP Eq. 6), and the fermion chain it is (Eq. 5) ───────────

"""
    disorder_fields(L, W; seed) -> Vector{Float64}

`h_i` drawn i.i.d. uniform on `[-W, W]`, the disorder distribution of RMP Eq. (5)/(6).

`seed` IS THE REALISATION LABEL AND IT IS NOT OPTIONAL IN PRACTICE. A disorder average is a
sum over realisations, and every arm of a comparison must see the SAME realisation or the
difference between two integrators is contaminated by the difference between two Hamiltonians --
which at these disorder strengths is far larger than any integrator error. Passing the seed
explicitly (rather than sampling from the global RNG) is what makes one array task reproducible
from its task id alone.
"""
function disorder_fields(L::Int, W::Float64; seed::Int)
    L >= 2 || throw(ArgumentError("disorder_fields needs at least two sites, got $L"))
    W >= 0 || throw(ArgumentError("disorder strength must be non-negative, got $W"))
    rng = Random.MersenneTwister(seed)
    return [W * (2 * rand(rng) - 1) for _ in 1:L]
end

"""
    quasiperiodic_fields(L, Delta; beta = 0.721, phi = 0.0) -> Vector{Float64}

`h_i = Delta * cos(2π β i + φ)`, the Aubry-André potential of arXiv:1501.05661 (`β = 532/738`).
`φ` labels the realisation. The single-particle transition is at `Delta = 2J`.
"""
quasiperiodic_fields(L::Int, Delta::Float64; beta::Float64 = 0.721, phi::Float64 = 0.0) =
    [Delta * cos(2π * beta * i + phi) for i in 1:L]

"""
    aubry_andre_xxz_mpo(L; J = 1.0, Delta, U = 0.0, beta = 0.721, phi = 0.0) -> MPO

Spinless interacting Aubry-André chain in experiment units,
`H = -J Σ (c†_i c_{i+1} + h.c.) + Delta Σ cos(2πβi + φ) n_i + U Σ n_i n_{i+1}`.

Jordan-Wigner gives `J_spin = 2J`, so `Delta/J` matches the paper's axis. `U` is a nearest-neighbour
interaction, not the experiment's on-site one (see [`hubbard_aa_mpo_d4`](@ref)).
"""
function aubry_andre_xxz_mpo(L::Int; J::Float64 = 1.0, Delta::Float64,
                             U::Float64 = 0.0, beta::Float64 = 0.721, phi::Float64 = 0.0)
    hz = quasiperiodic_fields(L, Delta; beta = beta, phi = phi)
    # `J_spin = 2J` (the JW factor) and `delta_zz = U / (2J) * ...`: the ZZ term of `xxz_chain` is
    # `J_spin * delta_zz * S^z S^z`, and `U n_i n_{i+1}` = `U (S^z + 1/2)(S^z + 1/2)` contributes
    # `U S^z S^z` plus a field `U/2 (S^z_i + S^z_{i+1})` and a constant. The field piece is folded
    # into `hz` below -- dropping it would be a DIFFERENT Hamiltonian, not a shifted one.
    if U != 0.0
        for i in 1:L
            nnb = (i > 1 ? 1 : 0) + (i < L ? 1 : 0)     # neighbours on an OPEN chain
            hz[i] += U * nnb / 2
        end
    end
    return field_mpo_from_terms(xxz_chain(L; J = 2J, delta = U / (2J)), hz)
end

"""
    field_mpo_from_terms(h, hz) -> MPO

[`mpo_from_terms`](@ref) plus a site-dependent field `Σ_i hz[i] S^z_i`. The field sits on the
`id -> done` entry, so it costs no virtual dimension; both boundary sites keep their field.
"""
function field_mpo_from_terms(h::XXZChain, hz::Vector{Float64})
    L, terms = length(h), h.terms
    n = length(terms)
    length(hz) == L || throw(DimensionMismatch(
        "field_mpo_from_terms got $(length(hz)) fields for $L sites"))
    # ⛔ NO SU(2) FORM, AND THE REASON IS PHYSICS RATHER THAN PLUMBING. `S^z_i` singles out an
    # axis, so a random longitudinal field breaks the non-abelian symmetry outright; under `:SU2`
    # `local_space` has no rank-2 `S^z` to place here either. The model is `:U1`/`:none` only,
    # which is also what the Néel start requires (a definite-`S^z` product state has weight in
    # many total-spin sectors and is not `:SU2`-representable at any bond dimension).
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "field_mpo_from_terms is :U1/:none only: a longitudinal S^z field breaks SU(2), and " *
        "the Néel start these models quench from is not :SU2-representable either"))
    q = local_space()
    Iloc, Sz = q.I, q.Sz

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid = _mpo_block(Iloc, i, tl, tr, :none)
        mat = Matrix{Any}(nothing, n + 2, n + 2)
        i < L && (mat[1, 1] = Iid)                       # transport the identity
        i > 1 && (mat[n + 2, n + 2] = Iid)               # ... on the far side of a term
        for t in 1:n
            i < L && (mat[1, 1 + t] =
                _mpo_block(terms[t].left, i, tl, tr, _side(terms[t].left, :left)))
            i > 1 && (mat[1 + t, n + 2] = to_concrete(terms[t].coeff *
                _mpo_block(terms[t].right, i, tl, tr, _side(terms[t].right, :right))))
        end
        # The on-site field, at EVERY site including both boundaries -- see the docstring.
        # A zero field is written as an explicit zero block rather than left unset: `oplus`
        # infers zero blocks from the row/column spaces, and a hole in this column has none.
        mat[1, n + 2] = to_concrete(hz[i] * _mpo_block(Sz, i, tl, tr, :none))
        if i == 1
            mat = reshape(mat[1, :], 1, n + 2)           # the `id` row only
        elseif i == L
            mat = reshape(mat[:, n + 2], n + 2, 1)       # the `done` column only
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    disordered_xxz_mpo(L; J = 1.0, delta = 1.0, hz) -> MPO

The MBL chain of arXiv:1804.11065 on an open chain,
`H = J Σ (S^x S^x + S^y S^y + Δ S^z S^z) + Σ_i hz[i] S^z_i`, with `hz` from
[`disorder_fields`](@ref). `S`-operator convention (`W_c ≈ 3.5-3.7`), not the review's Pauli one.
"""
disordered_xxz_mpo(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0, hz::Vector{Float64}) =
    field_mpo_from_terms(xxz_chain(L; J = J, delta = delta), hz)

# ── the spinful Aubry-André-Hubbard chain of arXiv:1501.05661, on 2L interleaved spin sites ──
# Interleaved (↑₁ ↓₁ ↑₂ ↓₂ …) keeps the bond dimension O(1); each hop becomes a three-site term with
# one Jordan-Wigner σ^z, U adds a U/2 field per site, and the constant energy shift is omitted.

"""
    hubbard_aa_mpo(L; J = 1.0, Delta = 0.0, U = 0.0, beta = 0.721, phi = 0.0) -> MPO

The spinful Aubry-André-Hubbard chain of arXiv:1501.05661 on `2L` interleaved spin sites (site
`2i-1` is `(i, ↑)`, `2i` is `(i, ↓)`), in experiment units. Runs under `:U1` and `:none`. Cannot be
evolved from a product state (range-2 hops); use [`hubbard_aa_mpo_d4`](@ref).
"""
function hubbard_aa_mpo(L::Int; J::Float64 = 1.0, Delta::Float64 = 0.0, U::Float64 = 0.0,
                        beta::Float64 = 0.721, phi::Float64 = 0.0)
    L >= 2 || throw(ArgumentError("hubbard_aa_mpo needs at least two physical sites, got $L"))
    sym = symmetry_mode()
    (sym === :U1 || sym === :none) || throw(ArgumentError(
        "hubbard_aa_mpo is :U1/:none only, got $sym: the quasiperiodic potential and the " *
        "Hubbard term both single out S^z, so there is no SU(2) form"))
    q = local_space()
    M = 2L
    # Vertices come from `xxz_chain` (never a hand-built `op'`, whose leg order is wrong under `:none`),
    # and the hop is `pair_mpo`'s transport block with σ^z as the Jordan-Wigner string.
    sigz = to_concrete(2.0 * q.Sz)                 # σ^z, the Jordan-Wigner string


    Jhop = zeros(Float64, M, M)
    for m in 1:(M - 2)
        Jhop[m, m + 2] = J                         # strict upper triangle: `pair_mpo`'s convention
    end
    # `U n↑ n↓` on physical site `i` is a distance-ONE pair on modes `(2i-1, 2i)`. Distance 1 means
    # the channel is never transported, so `Itrans` cannot touch it -- the same `σ^z` is safe for
    # every vertex. Expanding `n = S^z + 1/2` gives `U S^z S^z` plus the `U/2` field folded into
    # `h` below, plus a constant `U·L/4` that is a global phase and is deliberately omitted.
    JU = zeros(Float64, M, M)
    if U != 0.0
        for i in 1:L
            JU[2i - 1, 2i] = U
        end
    end

    # `h[m]`: the on-site field. Δ acts on BOTH spin states of a physical site, and `U` adds `U/2`
    # everywhere (see above) -- so the potential is indexed by PHYSICAL site while the field lives
    # on SPIN sites.
    h = zeros(Float64, M)
    for i in 1:L
        pot = Delta * cos(2π * beta * i + phi) + U / 2
        h[2i - 1] = pot
        h[2i] = pot
    end
    field = [to_concrete(h[m] * _mpo_block(q.Sz, m, "W,$(m - 1)", "W,$m", :none)) for m in 1:M]

    # `xxz_chain(J_spin)` gives `S⁺S⁻` the coefficient `J_spin/2` and the fermion hop is `+J σ^z S⁺S⁻`,
    # so `J_spin = 2J`.
    vhop = pair_vertices(xxz_chain(M; J = 2.0 * J, delta = 0.0))
    length(vhop) == 2 || error("xxz_chain(delta=0) gave $(length(vhop)) vertices, expected 2 " *
                               "(the two XY terms); the term list has changed")
    # `U n↑ n↓` expands to `U S^z S^z` + the `U/2` field already in `h` + a constant `U·L/4` that
    # is a global phase. `(Sz, Sz)` with no adjoint, exactly as `xxz_chain` writes its ZZ term.
    vertices = [vhop[1], vhop[2], PairVertex(q.Sz, q.Sz, 1.0)]
    return pair_mpo(M, [Jhop, Jhop, JU], vertices; Itrans = sigz, field = field)

end

"""
    hubbard_cdw_state(L; species) -> SymMPS

The charge-density-wave start of arXiv:1501.05661 on the interleaved `2L`-site chain: every atom
on an EVEN physical site, nothing on the odd ones. `species[k]` is `:up`, `:down` or `:both` for
the `k`-th even site, matching [`hubbard_cdw_vector`](@ref) on the ED side site for site.

`|occupied⟩ = |↑⟩` since `n = S^z + 1/2`, so this is a `product_state` and has bond dimension 1 --
the whole difficulty of the protocol, as with every other quench in this file.
"""
function hubbard_cdw_state(L::Int; species::Vector{Symbol})
    evens = collect(2:2:L)
    length(species) == length(evens) || throw(DimensionMismatch(
        "got $(length(species)) species for $(length(evens)) even sites"))
    spins = fill(:down, 2L)                        # :down == empty
    for (k, i) in enumerate(evens)
        if species[k] === :up
            spins[2i - 1] = :up
        elseif species[k] === :down
            spins[2i] = :up
        elseif species[k] === :both
            spins[2i - 1] = :up
            spins[2i] = :up
        else
            throw(ArgumentError("species must be :up, :down or :both, got $(species[k])"))
        end
    end
    return product_state(spins)
end

# The per-site density readout needs `sz_expectation`, which this module does not import; it lives
# in `benchmarks/mbl_scars.jl`.
