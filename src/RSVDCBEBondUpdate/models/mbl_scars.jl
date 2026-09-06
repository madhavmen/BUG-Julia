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

⚠ THE NORMALISATION IS THE `S`-OPERATOR ONE (Pal & Huse; Luitz, Laflorencie & Alet), where the
transition sits at `W_c ≈ 3.7`. The RMP writes the same model in PAULI matrices (its Eq. 6 has
`J⊥/2 Σ σσ` and `h_i σ^z_i`) and quotes `W* ≈ 3.5`; the two differ by factors of 2 that are easy
to carry into a plot and impossible to spot afterwards, so the convention is stated here and the
campaign quotes `W` in THIS one. Nothing in the code depends on which is "right"; a comparison
between integrators at the same `W` is unaffected either way, only the label on the axis is.

THIS IS THE FERMION CHAIN OF RMP Eq. (5). Jordan-Wigner maps it to
`Σ_i t(c†_i c_{i+1} + h.c.) + Σ_i V n_i n_{i+1} + Σ_i ε_i n_i` with `t = J/2`, `V = JΔ`, and
`ε_i = hz[i] - (V/2)(neighbour count)` -- i.e. the interacting fermions whose ED the review
reports at `L <= 16` are these spins, and the Néel state is the charge-density wave that the
Bloch-group experiment prepares. That equivalence is why a spin chain is the right object to
test: it is not an analogue of the fermion problem, it is the same operator in another basis.
"""
disordered_xxz_mpo(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0, hz::Vector{Float64}) =
    field_mpo_from_terms(xxz_chain(L; J = J, delta = delta), hz)

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

`P = 1/2 - S^z` projects on the UNEXCITED atom (`|∘>`, spin down) and `X = S^+ + S^- = σ^x` is
the Rabi flip. The convention is `|•> = up`, so a `|Z2> = |•∘•∘…>` quench starts from
[`neel_state`](@ref) and `⟨n_i⟩ = ⟨S^z_i⟩ + 1/2`.

⛔ `:none` IS FORCED. `σ^x` does not conserve `S^z`, so there is no U(1) form of this model at
all; under `:U1` the raising and lowering operators are rank-3 (their op-leg carries the ±2
charge) and `S^+ + S^-` is not even a well-formed sum of tensors, so the failure is loud rather
than a silent zero. It is still worth naming, because the MBL model in this same file is
`:U1`-native and a driver that sets the symmetry once for both gets one of them wrong.
"""
function pxp_operators()
    symmetry_mode() === :none || throw(ArgumentError(
        "pxp_operators is :none only, got $(symmetry_mode()): σ^x does not conserve S^z, so " *
        "the PXP model has no U(1) or SU(2) form"))
    q = local_space(:none)
    return (; I = q.I,
            P = to_concrete(0.5 * q.I - q.Sz),      # projector on |∘> = spin down
            X = to_concrete(q.Sp + q.Sm))           # σ^x, the Rabi flip
end

"""
    pxp_mpo(L; Omega = 1.0) -> MPO

`H = Ω Σ_{i=1}^{L} P_{i-1} X_i P_{i+1}` (with `P_0 = P_{L+1} = 1`) as an exact MPO of virtual
dimension 4 -- the PXP model of arXiv:2011.09486, the Rydberg blockade chain in the limit where
a neighbouring excitation is forbidden outright rather than merely costly.

`Ω` is the Rabi frequency and sets the time unit; the `|Z2>` revival period is `≈ 4.79/Ω`, which
is NOT `2π/Ω` -- that mismatch is the many-body effect the scar literature is about, and it is
what the `pxp` phase of the campaign measures.

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
