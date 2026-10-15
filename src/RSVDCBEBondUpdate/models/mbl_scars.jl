# THE TWO MODELS WHERE ERGODICITY BREAKS: MANY-BODY LOCALISATION AND QUANTUM MANY-BODY SCARS.
#
# WHY THESE TWO, TOGETHER, IN ONE FILE. They are the two ways a closed quantum system fails to
# thermalise, and the literature reaches both only by EXACT DIAGONALISATION of small chains:
#
#   MBL    Abanin/Altman/Bloch/Serbyn, Rev. Mod. Phys. 91, 021001 (arXiv:1804.11065), Eq. (5)
#          and Eq. (6): a disordered interacting FERMION chain, and the XXZ spin chain it maps
#          to under Jordan-Wigner. The transition sits near `W ~ 3.5-3.7` and the whole
#          finite-size debate (Lim & Sheng, arXiv:1510.08145) is fought at `L <= 22`.
#   SCARS  Serbyn/Abanin/Papic, Nature Physics 17, 675 (arXiv:2011.09486): the PXP model of a
#          Rydberg blockade chain, whose `|Z2>` quench revives instead of thermalising. ED there
#          reaches `L = 32` only because the blockade CONSTRAINS the Hilbert space to `Fib(L+2)`.
#
# WHAT AN INTEGRATOR HAS TO DO HERE THAT IT DOES NOT HAVE TO DO ON HEISENBERG. Both protocols
# start from a PRODUCT STATE (`chi = 1`) and ask about times long past the point where a fixed
# rank would have been chosen well:
#
#   * MBL entanglement grows LOGARITHMICALLY, so the rank needed creeps up over decades of time
#     and never saturates within the window that is plotted. A method that cannot grow its rank
#     mid-run reports a converged-looking plateau that is its own truncation.
#   * SCARS entanglement OSCILLATES -- it rises and then comes back DOWN at every revival. The
#     rank a step needs is not monotone in time, which is the one thing a fixed-rank schedule
#     cannot express at all.
#
# That is the whole reason this file exists: it is the input to the claim that the rank-adaptive
# BUG sweep is ROBUST on these protocols where a fixed-rank projection is not. The claim is
# TESTED in `benchmarks/mbl_scars.jl` against ED, not asserted here.
#
# ⛔ THE REFERENCE FOR BOTH MODELS IS EXACT DIAGONALISATION, AND THAT IS A DELIBERATE DEPARTURE
# from `benchmarks/cbe_sweeps_l16.jl`, whose header forbids sparse and dense references. That
# rule exists because the Heisenberg chain HAS closed forms (Bethe, free fermions) and agreeing
# with a formula is stronger evidence than agreeing with another computation. Neither model here
# has any closed form -- a random field destroys integrability, and PXP is not integrable at all
# -- so ED is not a second-best substitute for an analytic reference, it IS the reference the
# published results are themselves made of. What must not happen is grading against OUR OWN
# finest MPS grid; the sparse Krylov propagation in `benchmarks/mbl_scars_exact.jl` is exact to
# its own asserted residual and shares no code with the sweeps under test.

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

`h_i = Delta * cos(2π β i + φ)`: the QUASI-RANDOM (Aubry-André) potential of the Bloch-group
experiment, arXiv:1501.05661, whose `β = λ_s/λ_d = 532/738 ≈ 0.721` is a RATIO OF LASER
WAVELENGTHS and therefore not adjustable in the experiment.

⛔ THIS IS NOT WEAK DISORDER, IT IS A DIFFERENT MODEL. A quasiperiodic potential is
deterministic and correlated; it has no rare regions, so the Griffiths physics that dominates
the random-field transition is absent and the two belong to different universality classes.
`φ` -- not a random draw -- is the realization label, and averaging over `φ` is what the
experiment's 12 "disorder realizations" are.

⚠ AND THE SINGLE-PARTICLE PROBLEM HAS AN EXACT TRANSITION AT `Delta = 2J`, which is why the
non-interacting arm is a PARAMETER-FREE anchor rather than a curve to be fitted: the Aubry-André
model is self-dual at `Delta = 2J`, all states localised above it and extended below, and
arXiv:1501.05661 Fig. 3 is a measurement of exactly that. Any implementation that gets the
`Delta/J` axis wrong by the factor of 2 discussed in [`aubry_andre_xxz_mpo`](@ref) puts the
transition at 1 or 4 and still produces a perfectly plausible-looking curve.
"""
quasiperiodic_fields(L::Int, Delta::Float64; beta::Float64 = 0.721, phi::Float64 = 0.0) =
    [Delta * cos(2π * beta * i + phi) for i in 1:L]

"""
    aubry_andre_xxz_mpo(L; J = 1.0, Delta, U = 0.0, beta = 0.721, phi = 0.0) -> MPO

The SPINLESS interacting Aubry-André chain in EXPERIMENT UNITS, i.e. the Jordan-Wigner image of

    H = -J Σ_i (c†_i c_{i+1} + h.c.)  +  Delta Σ_i cos(2πβi + φ) n_i  +  U Σ_i n_i n_{i+1}

so `Delta` and `U` are quoted as multiples of `J` exactly as arXiv:1501.05661 quotes them.

⛔⛔ THE FACTOR OF 2 IN THE HOPPING IS THE WHOLE REASON THIS WRAPPER EXISTS. Under
Jordan-Wigner,

    -J Σ (c†_i c_{i+1} + h.c.) = -2J Σ (S^x_i S^x_{i+1} + S^y_i S^y_{i+1})

so a spin chain written with `J_spin = 1` is a FERMION chain with `J_exp = 1/2`, and every
`Delta/J` on the axis is then wrong by a factor of 2 -- putting the Aubry-André transition at
`Delta = 1` or `Delta = 4` instead of 2. This builder therefore sets `J_spin = 2J` and leaves
`hz` in experiment units. The SIGN of the hopping is NOT a trap: on a bipartite open chain
`c_i -> (-1)^i c_i` flips it, so it cannot change `⟨n_j⟩` or the imbalance.

⚠ `U` IS NOT THE EXPERIMENT'S `U`, AND THIS IS THE MODEL'S REAL LIMITATION. arXiv:1501.05661 uses
SPINFUL fermions with an ON-SITE `U Σ n_{i↑} n_{i↓}`; spinless fermions have no on-site
interaction available at all (`n_i² = n_i`), so the nearest-neighbour `U Σ n_i n_{i+1}` here --
`Delta_zz = U/J` in spin language -- is a DIFFERENT interaction. It is the standard theory proxy
and it reproduces the paper's Fig. 2 and Fig. 3 (which are about `Delta/J`, and Fig. 3 is
non-interacting outright), but it CANNOT be used to reproduce Fig. 4's `U/J` map or Fig. 5's
interaction-resolved entropy growth. Those need the two-species model; see the note in
`benchmarks/mbl_scars_exact.jl`.
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

[`mpo_from_terms`](@ref) plus a SITE-DEPENDENT `S^z` field: `H = h's terms + Σ_i hz[i] S^z_i`.

The field is the `id -> done` entry of the automaton -- it opens no channel, so it costs NOTHING
in virtual dimension, exactly as the transverse field does in [`tfim_mpo`](@ref).

⛔ THE BOUNDARY TRIMMING IS WHY THIS IS NOT A KEYWORD ON `mpo_from_terms`. That builder blanks
`mat[1, n+2]` at BOTH ends -- `Znil` at `i=1` ("nothing can be done yet") and again at `i=L`
("nothing may stay id past the end") -- because for a purely two-body term list that entry is
unused. Here `mat[1, n+2]` IS the field, so keeping those two lines would silently drop `h_1`
and `h_L`. That is a chain with two clean boundary sites: still Hermitian, still `U(1)`, and its
error against ED shrinks as `1/L`, so it would pass every small-`L` smoke test and then quietly
weaken the disorder in exactly the regime the campaign is about.
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

The MBL model of Rev. Mod. Phys. 91, 021001 (arXiv:1804.11065),

    H = J Σ_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + Δ S^z_i S^z_{i+1})  +  Σ_i hz[i] S^z_i

on an OPEN chain, with `hz` from [`disorder_fields`](@ref).

⚠ THE NORMALISATION IS THE `S`-OPERATOR ONE, and it is Lim & Sheng's (arXiv:1510.08145) exactly:
`H = Σ S_i·S_{i+1} - Σ h_i S^z_i` with `J = 1` and `h_i` uniform on `(-h, h)`. The sign of the
field is immaterial for a symmetric distribution. In THIS convention the reported transition sits
at `W_c ≈ 3.5` (Lim & Sheng, ED at `N = 12-22`) and at `W_c ≈ 3.72` (Luitz, Laflorencie & Alet,
`L <= 22`).

⛔ THOSE TWO NUMBERS ARE NOT A UNIT DISAGREEMENT, and treating them as one is a mistake this
docstring previously made. Both are in the `S`-operator convention; the gap between them is
finite-size and method, i.e. the very thing arXiv:1510.08145 is about. The genuine factor-of-2
trap is elsewhere: the RMP writes the same model in PAULI matrices (Eq. 6: `J⊥/2 Σ σσ`,
`Jz/2 Σ σσ`, `h_i σ^z_i`), so a `W` read off an RMP figure is NOT this `W`. The campaign quotes
`W` in the `S` convention throughout. Nothing in the code depends on which is "right" -- a
comparison between integrators at the same `W` is unaffected -- only the label on the axis is,
and only the axis label can be wrong silently.

THIS IS THE FERMION CHAIN OF RMP Eq. (5). Jordan-Wigner maps it to
`Σ_i t(c†_i c_{i+1} + h.c.) + Σ_i V n_i n_{i+1} + Σ_i ε_i n_i` with `t = J/2`, `V = JΔ`, and
`ε_i = hz[i] - (V/2)(neighbour count)` -- i.e. the interacting fermions whose ED the review
reports at `L <= 16` are these spins, and the Néel state is the charge-density wave that the
Bloch-group experiment prepares. That equivalence is why a spin chain is the right object to
test: it is not an analogue of the fermion problem, it is the same operator in another basis.
"""
disordered_xxz_mpo(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0, hz::Vector{Float64}) =
    field_mpo_from_terms(xxz_chain(L; J = J, delta = delta), hz)

# ── THE SPINFUL AUBRY-ANDRÉ-HUBBARD CHAIN: THE ACTUAL EXPERIMENT ──────────────────────────────
#
#     H = -J Σ_{i,σ} (c†_{i+1,σ} c_{i,σ} + h.c.)
#         + Δ Σ_i cos(2πβi + φ) (n_{i↑} + n_{i↓})
#         + U Σ_i n_{i↑} n_{i↓}
#
# This is arXiv:1501.05661's Hamiltonian. Figs. 2, 4 and 5 of that paper are measurements of it;
# the spinless proxy above reaches only Fig. 3, which is explicitly `U = 0` ("only a single spin
# component was used"). Its exact reference is `benchmarks/hubbard_aa_exact.jl`, validated to
# 1e-16 against free fermions at `U = 0` BEFORE this builder was written -- so the MPO is graded
# against something that shares no code with it.
#
# ⛔⛔ THE ORDERING IS INTERLEAVED, AND THE OBVIOUS ALTERNATIVE IS A TRAP. Two species on `L`
# sites become one spin chain on `2L` sites, and there are two natural orderings:
#
#   BLOCKED   (↑₁…↑_L ↓₁…↓_L)   Every hop is nearest-neighbour and STRING-FREE, and `U` becomes a
#                               distance-`L` `S^z S^z` coupling that `pair_mpo` already represents
#                               EXACTLY for an arbitrary `J[i,j]`. So it needs no new machinery at
#                               all -- and it is UNUSABLE. It places every ↑-↓ correlation across
#                               the middle cut, so the bond dimension there has to carry the full
#                               inter-species mutual information: `chi ~ exp(L)`. The MPO would be
#                               cheap and the STATE impossible.
#
#   INTERLEAVED (↑₁ ↓₁ ↑₂ ↓₂ …)  `U n_{i↑}n_{i↓}` is a NEAREST-NEIGHBOUR `S^z S^z` on odd bonds,
#                               and correlated partners sit adjacent, so the entanglement is
#                               physical rather than an artefact of the ordering. The price is
#                               that a hop ↑ᵢ → ↑ᵢ₊₁ jumps over ↓ᵢ and drags a JORDAN-WIGNER
#                               STRING: it is a THREE-SITE term. Bond dimension stays O(1).
#
# THE STRING, derived rather than guessed. With `c_m = (∏_{k<m} σ^z_k) S⁻_m`,
#
#     c†_{m+2} c_m = σ^z_m σ^z_{m+1} S⁺_{m+2} S⁻_m = -σ^z_{m+1} S⁺_{m+2} S⁻_m
#
# because `σ^z_m S⁻_m = -S⁻_m` (S⁻ leaves site `m` down, where `σ^z = -1`). Adding the h.c. and
# the `-J` prefactor, the two signs cancel:
#
#     -J (c†_{m+2}c_m + h.c.) = +J σ^z_{m+1} (S⁺_{m+2} S⁻_m + S⁺_m S⁻_{m+2})
#
# so each hopping vertex carries `+J` -- NOT `-J`, and not `2J`: the factor of 2 that appears in
# the SPINLESS wrapper comes from writing `S^xS^x + S^yS^y` instead of `S⁺S⁻`, and there is no
# such rewriting here. On a bipartite chain the overall sign is a gauge anyway.
#
# `U n_{i↑}n_{i↓} = U S^z S^z + (U/2)(S^z_{2i-1} + S^z_{2i}) + U/4`, so `U` contributes a field of
# `U/2` on EVERY site as well as the `S^zS^z` bond -- dropping that field piece is a different
# Hamiltonian, not a shifted one.
#
# ⚠ THE OMITTED CONSTANT. `Δcos(...)·(n↑+n↓)` and the `U/4` above also produce
# `Σ_i Δcos(2πβi+φ) + U·L/4` times the identity, which this MPO does NOT include. It is a global
# phase: densities, imbalance and entropies are untouched, but `mpo_energy` is offset from
# `hubbard_aa_sparse`'s by exactly that amount, so the two agree on DYNAMICS and not on `⟨H⟩`.

"""
    hubbard_aa_mpo(L; J = 1.0, Delta = 0.0, U = 0.0, beta = 0.721, phi = 0.0) -> MPO

The spinful Aubry-André-Hubbard chain of arXiv:1501.05661 as an MPO on `2L` INTERLEAVED spin
sites: site `2i-1` is `(i, ↑)` and site `2i` is `(i, ↓)`, with `n_{i,σ} = S^z + 1/2`.

`Delta`, `U` and `J` are in EXPERIMENT units -- the paper's own `Delta/J` and `U/J` -- because
this is written in the fermion language directly rather than through an XXZ rewriting.

Virtual dimension 7, independent of `L`:

    1 = id            4 = hop B opened (S⁺ placed, σ^z owed)
    2 = hop A opened  5 = hop B strung (σ^z placed, S⁻ owed)
    3 = hop A strung  6 = U opened (S^z on an odd site, partner owed)
                      7 = done

No self-loop on 2-5, which is what pins the three operators to three CONSECUTIVE sites; the `U`
channel opens only on odd sites, which is what pins it to the two spin states of ONE physical site
rather than to every bond.

Runs under `:U1` (total `S^z` = total particle number, conserved) and `:none`. Under `:U1` the
`S^±` channels are CHARGED, so their string block is built by `_charged_transport` -- the same
verified-by-contraction routine `pair_mpo` uses -- with `σ^z` as the transported site operator
instead of the identity.
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
    # ⛔⛔ NEVER BUILD A `PairVertex` WITH `op'` BY HAND -- GO THROUGH `xxz_chain`.
    # `xxz_chain` BRANCHES ON RANK: `(Sp, Sp')` under `:U1` (rank 3), but `(Sp, Sm, -J)` written
    # EXPLICITLY under `:none` (rank 2), because there the adjoint transposes the site legs. The
    # two are equal as scalars -- `Sp' = -Sm`, so `(Sp,Sp')·(+J) = (Sp,Sm)·(-J)` -- and that
    # equivalence is exactly the trap: it holds for the ALGEBRA and not for the LEG ORDERING that
    # `_mpo_block` consumes. MEASURED side by side in `phase_hubdiag` (job 16339352): canonical
    # vertices evolve (`bond = 5`), the hand-built adjoint ones OOM at 32G inside `pair_mpo`.
    #
    # ⛔ THE HOPPING IS BUILT BY `pair_mpo`, NOT BY A HAND-ROLLED AUTOMATON. A bespoke 7-state
    # automaton was written first and failed twice: once on the adjoint pairing, and then by
    # producing an MPO whose diagonal terms were right while the kinetic term was silently ABSENT
    # -- `I = 1.000000` forever, `bond = 1`, `dt-ratio = 1.00`, a frozen product state (job
    # 16335193, confirmed under BOTH `:none` and `:U1` by `phase_hubdiag`). `pair_mpo` is
    # verified-by-contraction and already carries every other long-range model here, so the
    # hopping goes through it.
    #
    # THE MAPPING. Interleaved ordering `↑₁ ↓₁ ↑₂ ↓₂ …` puts spin `σ` of physical site `i` at mode
    # `2(i-1)+σ`, so a hop between ADJACENT physical sites is a distance-2 move on the spin chain
    # that steps over exactly ONE mode -- the other spin of the same site. Jordan-Wigner therefore
    # contributes a single `σ^z` on that crossed mode, which is precisely the TRANSPORT block:
    #
    #     c†_{m+2} c_m  =  S^+_{m+2} σ^z_{m+1} S^-_m
    #
    # so `Itrans = σ^z` with a coupling supported on `|i-j| = 2` IS the fermionic hopping. The
    # `-J` of the Hamiltonian and the `-1` from `σ^z_m S^-_m = -S^-_m` cancel, leaving `+J`.
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

    # ⛔⛔ VERTICES COME FROM `xxz_chain`, NEVER FROM `op'` BY HAND. `xxz_chain` BRANCHES ON RANK:
    # under `:U1` (rank 3) it uses `(Sp, Sp')`, under `:none` (rank 2) it writes `(Sp, Sm, -J)`
    # EXPLICITLY and never takes an adjoint -- because under `:none` the adjoint transposes the
    # site legs. Hand-writing `PairVertex(q.Sm, to_concrete(q.Sm'), 1.0)` in both symmetries built
    # a malformed vertex: `pair_mpo` then allocated without bound (OOM at 32G in 148 s, jobs
    # 16339248/16339259) and, when it did build, produced an MPO with the DIAGONAL terms right and
    # the kinetic term silently ABSENT -- a frozen product state (`bond = 1`, `I = 1.000000`
    # forever). `spin_vertices`, which every working `pair_mpo` model in this repo routes through,
    # IS `pair_vertices(xxz_chain(...))`; measured side by side in `phase_hubdiag`, the canonical
    # vertices evolve (`bond = 5`) where the adjoint ones do not.
    #
    # AMPLITUDE. `xxz_chain(J_spin)` gives `S^+ S^-` the coefficient `J_spin/2`, and
    #     -J c†_{m+2} c_m = +J σ^z_{m+1} S^+_{m+2} S^-_m
    # (the `σ^z_m S^-_m = -S^-_m` factor cancelling the `-J`), so `J_spin/2 = J` ⇒ `J_spin = 2J`.
    # That is the SAME Jordan-Wigner factor of 2 the spinless `aubry_andre_xxz_mpo` uses, and that
    # wrapper is validated to 1e-7 against free fermions -- so the convention is pinned, not
    # guessed. `Jhop` therefore carries 1.0 and the amplitude lives in `J_spin`.
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

# ⚠ THE PER-PHYSICAL-SITE DENSITY READOUT LIVES IN THE DRIVER, NOT HERE. It is
# `⟨S^z_{2i-1}⟩ + ⟨S^z_{2i}⟩ + 1`, which needs `sz_expectation` -- and that name is NOT among the
# ones this module imports from `BondUpdateBUG` (see the `using ..BondUpdateBUG:` list). Defining
# it here would load and export perfectly happily and then throw `UndefVarError` at the first
# call, i.e. inside a job rather than at include time. `benchmarks/mbl_scars.jl` has the full
# `BondUpdateBUG` in scope and defines it there.

# ── SCARS: the PXP model of a Rydberg blockade chain ──────────────────────────────────────────
#
# THE AUTOMATON, virtual dimension 4 whatever `L` is. A three-site term needs one more state
# than a two-site one: the middle operator has to be remembered as well as the opening.
#
#     1 = id       nothing opened
#     2 = P placed the left projector is down, `X` is owed on the NEXT site
#     3 = X placed the flip is down, the right projector is owed on the NEXT site
#     4 = done     a complete term lies behind us
#
#          ⎡ I   P   X*   0  ⎤       * `id -> X placed` fires ONLY at site 1
#     W =  ⎢ 0   0   X    X† ⎥       † `P placed -> done` fires ONLY at site L
#          ⎢ 0   0   0    P  ⎥
#          ⎣ 0   0   0    I  ⎦
#
# NO SELF-LOOP on states 2 or 3, which is what makes the three operators land on three
# CONSECUTIVE sites rather than at any three increasing positions.
#
# ⛔ THE TWO BOUNDARY ENTRIES ARE THE MODEL, NOT A CORRECTION TO IT. `H = Σ_i P_{i-1} X_i P_{i+1}`
# with `P_0 = P_{L+1} = 1` contains `X_1 P_2` and `P_{L-1} X_L`, which are TWO-site terms and
# cannot be produced by the bulk path. Dropping them gives an open chain whose two end atoms are
# frozen out of the dynamics; the `|Z2>` revival period shifts by O(1/L) and the state stays
# perfectly plausible. They are entered as the starred/daggered blocks above and both are
# exercised by the `L = 3` case in the tests (`H = X_1 P_2 + P_1 X_2 P_3 + P_2 X_3`).

"""
    pxp_operators() -> (; I, P, X)

The local blockade operators, `:none` only.

`P = 1/2 - S^z` projects on the UNEXCITED atom (`|∘>`, spin down) and `X = 2 S^x = σ^x` is the
Rabi flip. The convention is `|•> = up`, so a `|Z2> = |•∘•∘…>` quench starts from
[`neel_state`](@ref) and `⟨n_i⟩ = ⟨S^z_i⟩ + 1/2`.

⛔⛔ `X` IS **NOT** `S^+ + S^-`, AND WRITING IT THAT WAY MAKES `H` NON-HERMITIAN. Telum's `:none`
raising/lowering operators carry a normalisation AND A SIGN that `sx_operator` records as
MEASURED: `S^x = (S^- - S^+)/√2`, with the resulting matrix asserted to be `[0 1/2; 1/2 0]`.
Solving that for the two operators gives

    S^+ = -(1/√2) |↑⟩⟨↓|        S^- = +(1/√2) |↓⟩⟨↑|

so `S^+ + S^-` is the real ANTISYMMETRIC matrix `[0 -1/√2; 1/√2 0]` -- proportional to `-i σ^y`,
hence ANTI-Hermitian. The consequence is not a wrong coefficient, it is a non-unitary
`exp(-iHt)`: MEASURED on the `L = 8` gate, `‖ψ‖` reached 3.95 (BUG) and 5.07 (TDVP2) after 20
steps of `dt = 0.05`, and the fidelity came out 0.25-0.34 against an exact 0.0076. Nothing threw;
Lanczos on a non-Hermitian generator does not fail, it lies.

⚠ THE SIGN ALONE WOULD HAVE BEEN INVISIBLE. `X -> -X` gives `H -> -H`, i.e. `exp(+iHt)`, which
is the complex conjugate evolution and leaves `|⟨Z_2|ψ(t)⟩|²` and every `⟨S^z_j⟩` unchanged. It
is the LOSS OF SYMMETRY, not the sign, that breaks unitarity -- which is why the failure had to
be found through `‖ψ‖` rather than through a wrong-looking observable.

So `X` is built from [`sx_operator`](@ref), which carries its own assertion against the measured
`[0 1/2; 1/2 0]` and throws if Telum's normalisation ever changes. `P` needs no such check: it is
a FUNCTION of `S^z`, so `P = 1/2 - S^z` annihilates the `S^z = +1/2` state by algebra whichever
dense index that happens to be -- the only thing that could break it is `S^z` having eigenvalues
`±1` instead of `±1/2`, and that is asserted below.

⛔ `:none` IS FORCED. `σ^x` does not conserve `S^z`, so there is no U(1) form of this model at
all; under `:U1` the raising and lowering operators are rank-3 (their op-leg carries the ±2
charge) and the sum is not even a well-formed sum of tensors, so the failure is loud rather than
a silent zero. It is still worth naming, because the MBL model in this same file is `:U1`-native
and a driver that sets the symmetry once for both gets one of them wrong.
"""
function pxp_operators()
    symmetry_mode() === :none || throw(ArgumentError(
        "pxp_operators is :none only, got $(symmetry_mode()): σ^x does not conserve S^z, so " *
        "the PXP model has no U(1) or SU(2) form"))
    q = local_space(:none)
    # `S^z` must be the spin-1/2 generator, not the Pauli matrix, or `P = 1/2 - S^z` is not a
    # projector at all (`P² ≠ P`) and every blockade in the model is silently wrong.
    Mz = Matrix(q.Sz.RMTs[1])
    (maximum(abs.(sort(real.(diag(Mz))) .- [-0.5, 0.5])) < 1e-12 &&
     maximum(abs.(Mz .- Diagonal(diag(Mz)))) < 1e-12) || throw(ErrorException(
        "pxp_operators: :none S^z came back as $Mz, expected diagonal with eigenvalues ±1/2. " *
        "`P = 1/2 - S^z` is only the |∘⟩ projector for that normalisation."))
    P = to_concrete(0.5 * q.I - q.Sz)               # projector on |∘⟩ = the S^z = -1/2 state
    X = to_concrete(2.0 * sx_operator())            # σ^x = 2 S^x -- see the docstring
    MX = Matrix(X.RMTs[1])
    maximum(abs.(MX .- [0.0 1.0; 1.0 0.0])) < 1e-12 || throw(ErrorException(
        "pxp_operators: built X = $MX, expected σ^x = [0 1; 1 0]. A non-symmetric X makes the " *
        "PXP MPO non-Hermitian and `exp(-iHt)` non-unitary; ‖ψ‖ then grows without anything " *
        "throwing (measured 3.95 after 20 steps at L=8)."))
    return (; I = q.I, P = P, X = X)
end

"""
    pxp_mpo(L; Omega = 1.0) -> MPO

`H = Ω Σ_{i=1}^{L} P_{i-1} X_i P_{i+1}` (with `P_0 = P_{L+1} = 1`) as an exact MPO of virtual
dimension 4 -- the PXP model of arXiv:2011.09486, the Rydberg blockade chain in the limit where
a neighbouring excitation is forbidden outright rather than merely costly.

`Ω` is the Rabi frequency and sets the time unit. The `|Z2>` quench revives with a period that is
an `O(1)` multiple of `1/Ω` and is NOT `2π/Ω` -- arXiv:2011.09486 makes exactly that point: the
observed frequency "did not coincide with the bare Rabi frequency, signalling the importance of
many-body effects".

⛔ NO PERIOD IS ASSERTED HERE, DELIBERATELY. A specific figure (`4.79/Ω` is the number usually
quoted, and it could not be confirmed from the review itself) hard-coded in a docstring becomes
the thing a plot is aligned to, and then a revival at the wrong period reads as agreement. The
period is MEASURED against the ED propagation in `benchmarks/mbl_scars.jl`, whose reference and
whose MPS arms are both graded on `|⟨Z_2|ψ(t)⟩|²` sampled at the same times -- so the period is an
output of the campaign, not an input to it.

Requires `L >= 3` (at `L = 2` there is no three-site term and the model degenerates) and the
`:none` symmetry mode.
"""
function pxp_mpo(L::Int; Omega::Float64 = 1.0)
    L >= 3 || throw(ArgumentError("pxp_mpo needs at least three sites, got $L"))
    ops = pxp_operators()
    W = 4

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid  = _mpo_block(ops.I, i, tl, tr, :none)
        Pblk = _mpo_block(ops.P, i, tl, tr, :none)
        Xblk = _mpo_block(ops.X, i, tl, tr, :none)
        Znil = to_concrete(0.0 * Iid)
        mat  = Matrix{Any}(nothing, W, W)

        i < L && (mat[1, 1] = Iid)                                  # stay `id`
        i > 1 && (mat[W, W] = Iid)                                  # stay `done`
        i <= L - 1 && (mat[1, 2] = Pblk)                            # open with the left P
        i >= 2 && i <= L - 1 && (mat[2, 3] = Xblk)                  # the bulk flip
        i >= 2 && (mat[3, W] = to_concrete(Omega * Pblk))           # close with the right P
        i == 1 && (mat[1, 3] = Xblk)                                # X_1 P_2: no left P exists
        i == L && (mat[2, W] = to_concrete(Omega * Xblk))           # P_{L-1} X_L: no right P
        # `id -> done` is EMPTY BY PHYSICS -- PXP has no on-site term -- but it is the only entry
        # of column `W` at `i = 1` and of row 1 at `i = L`, and `oplus` cannot infer a zero block
        # for a row/column it has no other entry in. An explicit zero, not a gap.
        mat[1, W] = Znil

        if i == 1
            mat = reshape(mat[1, :], 1, W)
        elseif i == L
            mat = reshape(mat[:, W], W, 1)
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    z2_state(L) -> SymMPS

`|Z_2> = |•∘•∘…>`, the period-2 density wave the scar revivals are measured from.

It IS [`neel_state`](@ref) -- `|•> = up` -- and the alias exists so a PXP driver reads in the
language of the model rather than of the spin chain. Both are `:none`/`:U1` product states of
bond dimension 1, which is the entire difficulty: the integrator has to build every bit of the
rank it will ever need out of a state that has none.
"""
z2_state(L::Int) = product_state([isodd(i) ? :up : :down for i in 1:L])

"""
    zk_state(L, k) -> SymMPS

`|Z_k> = |•∘…∘•∘…>`, an excitation every `k` sites; `k = 0` is the POLARISED VACUUM `|∘∘∘…>`.

⛔⛔ **THE CONTRAST STATE IS NOT OPTIONAL -- IT IS WHAT MAKES A REVIVAL A SCAR.** `|Z_2>` reviving
says nothing on its own: a revival is only evidence of weak ergodicity breaking if a GENERIC
state in the same Hamiltonian, at the same size and the same bond dimension, does NOT revive.
Reporting the `|Z_2>` fidelity alone would be a curve with no control, and it would look
identical whether or not the physics were there.

The ladder the scar literature uses (arXiv:2011.09486 and Turner et al., Nature Physics 14, 745):

  * `k = 2`  `|•∘•∘…>`   STRONG, long-lived revivals -- the scarred state.
  * `k = 3`  `|•∘∘•∘∘…>` WEAKER but still visible revivals -- an intermediate case, and the one
             that shows the effect is not a peculiarity of the period-2 pattern alone.
  * `k = 0`  `|∘∘∘…>`    the polarised state: NO revivals, fast thermalisation. THE CONTROL.

⚠ Every `k >= 2` pattern satisfies the Rydberg blockade (no two adjacent excitations), so all of
these live inside the constrained basis and are directly comparable. `k = 1` would be all-up,
which violates the blockade at every bond, and is rejected rather than silently evolved.

⚠ All are bond-dimension-1 product states, which is the whole difficulty for the integrator: it
must build every bit of rank it will ever need starting from none. That is also why this is the
right place to separate a rank-adaptive integrator from a fixed-rank one -- the scarred
trajectory stays LOW rank while the control's rank grows.
"""
function zk_state(L::Int, k::Int)
    k == 1 && throw(ArgumentError(
        "zk_state: k=1 is all-up, which violates the Rydberg blockade on every bond"))
    k < 0 && throw(ArgumentError("zk_state: k must be >= 0, got $k"))
    k == 0 && return product_state(fill(:down, L))
    return product_state([(i - 1) % k == 0 ? :up : :down for i in 1:L])
end
