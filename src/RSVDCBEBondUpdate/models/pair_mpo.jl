# AN EXACT MPO FOR AN ARBITRARY TWO-BODY COUPLING MATRIX, INCLUDING CHARGED OPERATORS.
#
# WHY THIS FILE EXISTS. `long_range_mpo` in `mpo.jl` carries a long-range interaction with ONE
# self-loop channel per exponentially decaying term, which is exact for a geometric tail and
# has a virtual dimension independent of `L`. It cannot do Haldane-Shastry, for two separate
# reasons, and both are recorded there:
#
#   1. `J(r) = pi^2 / (L^2 sin^2(pi r / L))` is not a geometric decay, so it would have to be
#      FITTED by a sum of exponentials -- introducing an error that belongs to the Hamiltonian
#      rather than to the integrator, and that cannot be separated from a result afterwards.
#   2. `S . S` is a RANK-3 operator: its op-leg becomes the channel's virtual leg, so that
#      channel's virtual space is CHARGED, and a self-loop on it needs the identity on a
#      charged space rather than `lambda * Iid`. `long_range_mpo` THROWS on rank-3 operators
#      rather than emit a wrong block (`mpo.jl:279-282`).
#
# This file lifts both restrictions the exact way. It spends virtual dimension instead of
# accuracy: ONE CHANNEL PER OPENING SITE, so the coupling `J[i,j]` is applied at the closing
# site and any coupling matrix whatsoever is represented EXACTLY.
#
#     w = 2 + n * (number of live openings),   n = number of two-body vertices
#
# For Haldane-Shastry that is `n = 1` under SU(2) -- one irreducible `S . S` -- against `n = 3`
# under U(1) (`Sp Sm`, `Sm Sp`, `Sz Sz`). So `w ~ L` versus `w ~ 3L`. THAT IS THE ONLY PLACE
# THE TWO SYMMETRIES ARE ALLOWED TO DIFFER: same operator, same physics, different virtual
# dimension and therefore different cost. Nothing downstream inspects `W`'s shape, so the
# environments of Eq. (1.8), `apply_h_two_site`, `zero_site_h` and the CBE sketches are
# untouched.
#
# WHY `O(L)` IS NOT A DEFECT HERE. A coupling with no exponential structure has no
# finite-dimensional MPO; the bond dimension of the exact MPO is the rank of the coupling's
# Hankel matrix, which for `1/sin^2` is full. So `O(L)` is not a wasteful encoding of
# something cheaper -- it is what exactness costs, and at the sizes an exact reference is
# wanted for (`L <= ~24`) it is affordable. `long_range_mpo` remains the right tool when a
# geometric tail really is the model.
#
# THE ONE GENUINELY NEW TENSOR is the CHARGED TRANSPORT BLOCK: `delta_{w_l w_r} (x) I_site`,
# with the virtual legs carrying the operator's op-leg space instead of a dim-1 vacuum. See
# `_charged_transport`, which builds it and then PROVES the arrows are right by the only test
# that cannot lie -- that it contracts against the opening and closing blocks the way the MPO
# will contract it.

# ── the vertices ─────────────────────────────────────────────────────────────

"""
    PairVertex(left, right, coeff)

One two-body vertex: `coeff * left_i * right_j` for every pair `i < j`, to be weighted by a
coupling matrix. `left` and `right` may be rank 2 (charge neutral) or rank 3 (an op-leg
carrying the charge that makes the pair allowed) -- unlike [`LongRangeTerm`](@ref), which
refuses rank 3.

The two halves must join over their op-legs, exactly as in an [`XXZTerm`](@ref): under U(1)
`(Sp, Sm')`-style pairs, under SU(2) the single `(S, S')`.
"""
struct PairVertex
    left::Any
    right::Any
    coeff::Float64
end

PairVertex(t::XXZTerm) = PairVertex(t.left, t.right, t.coeff)

"The vertices of a term list, so a nearest-neighbour `XXZChain` and a long-range coupling can
be built from the SAME operator algebra and cannot disagree about normalisation."
pair_vertices(h::XXZChain) = [PairVertex(t) for t in h.terms]

# ── the charged transport block ──────────────────────────────────────────────

"""
    _outer(A, B) -> TLArray

Tensor product of two tensors with no shared legs, `(A's legs..., B's legs...)`.

Telum has `⊗`, and it is tried first. The fallback needs only `addSingleton` and `contract`:
append a dim-1 leg to `A`, prepend one to `B`, and contract over it -- the contraction of a
one-dimensional index IS the outer product. Both orders are attempted because
`addSingleton`'s arrow choice is not part of the documented contract, and exactly one of the
two will be contractable.
"""
function _outer(A, B)
    try
        return to_concrete(A ⊗ B)
    catch
    end
    A1 = to_concrete(addSingleton(A, (length(A.inds) + 1,)))
    B1 = to_concrete(addSingleton(B, (1,)))
    for (x, y) in ((A1, B1), (B1, A1))
        try
            T = to_concrete(contract(x, (length(x.inds),), y, (1,)))
            # Restore (A's legs..., B's legs...) if the contractable order was the other one.
            return x === A1 ? T :
                   to_concrete(permutedims(T, (ntuple(k -> length(B.inds) + k,
                                                      length(A.inds))...,
                                              ntuple(k -> k, length(B.inds))...)))
        catch
        end
    end
    throw(ArgumentError("_outer: neither ⊗ nor a singleton contraction produced a tensor " *
                        "product; Telum's API has changed"))
end

"""
    _charged_transport(opening, closing, Iloc, i, tl, tr) -> TLArray

The automaton's diagonal entry for a channel that is OPEN across site `i`:
`delta_{w_l w_r} (x) I_site`, as a rank-4 `(w_l '+', s_ket '+', s_bra '-', w_r '-')` block
whose virtual legs carry the channel's own (possibly charged) space.

`opening` is the block that opens the channel (its leg 4 IS the channel's right-facing
virtual leg) and `closing` the one that closes it (its leg 1 is the left-facing one). Both
are passed because the block has to contract against BOTH, and that pair of contractions is
what the construction is verified by.

WHY IT IS VERIFIED RATHER THAN DERIVED. Telum's `getIdentity` flips its input arrows and
reports dual labels (`sectors.jl:77`, and `mpo.jl:243` names this as the reason charged
channels were left unwritten). Reasoning about which of the two legs ends up `'+'` and whether
the labels come back dualised is exactly the kind of step that produces a confident wrong
answer, and the failure is silent: a mislabelled virtual leg does not throw, it traces the
wrong pair. So every candidate leg assignment is built and the one that CONTRACTS CORRECTLY is
returned.

THREE REQUIREMENTS, AND ALL THREE ARE NEEDED. Measured: of the six candidates for a U(1) `Sp`
channel, one satisfies all three, one satisfies only `T -> T`, and one satisfies only
`open -> T`. So testing a subset picks a wrong block that then fails deep inside
`_mpo_left_step` with a space mismatch:

    open -> T      `W^[i]`'s channel column against `W^[i+1]`'s transport row
    T -> T         transport at `i` against transport at `i+1` -- what a distance >= 3 path
                   needs, and the one an earlier version omitted
    T -> close     transport at `i` against the closing block at `i+1`

These ARE the contractions the MPO performs, so passing them is not a proxy for correctness.

THE UNCHARGED SHORT-CIRCUIT TESTS THE SPACE, NOT THE DIMENSION, and that distinction is the
bug this comment exists to prevent recurring. A U(1) op-leg carries ONE sector of dimension
ONE -- `Sp`'s leg is `[((2,), 1)]` -- so `leg_dim(opening, 4) == 1` is true for a CHARGED
channel just as it is for the vacuum, and short-circuiting on it silently returned the vacuum
identity for every charged channel. Distance 1 still passed (no transport happens there) and
`:none` still passed (no charged channel exists there), which is exactly how it survived the
first two validation stages. Comparing against the identity block's own leg-4 space is exact
and symmetry-agnostic.
"""
function _charged_transport(v::PairVertex, Iloc, i::Int,
                            tl::AbstractString, tr::AbstractString)
    tnext = "W,$(i + 1)"
    op_i = _mpo_block(v.left, i, tl, tr, _side(v.left, :left))
    Iid  = _mpo_block(Iloc, i, tl, tr, :none)
    # Charge-NEUTRAL channel: the virtual space is the vacuum, so the plain identity block IS
    # the transport. Tested by SPACE -- see the docstring for why a dimension test is wrong.
    op_i.spaces[4] == Iid.spaces[4] && return Iid

    # EVERY TEST PARTNER LIVES AT SITE `i+1`, and that is not bookkeeping fussiness. `contract`
    # matches on ITAGS as well as arrows, and the site legs of two blocks built for the same site
    # carry the same tag -- so contracting `T(i)` with `T(i)` produces four legs tagged `S,i` and
    # throws on a DUPLICATE INDEX, which has nothing to do with the virtual spaces the test is
    # about. Building the partners at `i+1` is what makes the three tests test what they claim.
    cl_n    = _mpo_block(v.right, i + 1, tr, tnext, _side(v.right, :right))
    Isite_i = _mpo_retag(Iloc, i)
    Isite_n = _mpo_retag(Iloc, i + 1)
    # PLACEHOLDER TAGS, and they are not cosmetic either. `getIdentity(ref, l)` gives BOTH of its
    # legs the source leg's tag -- and from the `closing` source it gives them the same ARROW as
    # well (measured: `[1 '-' (2,)] [2 '-' (-2,)]`). The raw tensor product of that with the site
    # identity therefore carries two identical `TLIndex`es and is an ILLEGAL `TLArray`, throwing
    # inside the constructor rather than at the point of use. Naming the two virtual legs apart
    # before combining removes that whole failure class.
    tA, tB = "pmA,$i", "pmB,$i"

    for (ref, l) in ((op_i, 4), (cl_n, 1))
        Iw = try
            w = to_concrete(getIdentity(ref, l))
            length(w.inds) == 2 || continue
            to_concrete(setitag(setitag(w, 1, tA), 2, tB))
        catch
            continue
        end
        # Either virtual leg may be the left one; try both assignments.
        for perm in ((1, 3, 4, 2), (2, 3, 4, 1))
            Tt, Tn = try
                (to_concrete(setitag(setitag(
                     to_concrete(permutedims(_outer(Iw, Isite_i), perm)), 1, tl), 4, tr)),
                 to_concrete(setitag(setitag(
                     to_concrete(permutedims(_outer(Iw, Isite_n), perm)), 1, tr), 4, tnext)))
            catch
                continue
            end
            ok = try
                contract(op_i, (4,), Tn, (1,))         # open at i  -> transport at i+1
                contract(Tt, (4,), Tn, (1,))           # transport   -> transport
                contract(Tt, (4,), cl_n, (1,))         # transport   -> close at i+1
                true
            catch
                false
            end
            ok && return Tt
        end
    end
    throw(ArgumentError(
        "_charged_transport: could not build a transport block for a channel whose virtual " *
        "space is $(op_i.spaces[4]); no candidate leg assignment satisfied all three of " *
        "open->T, T->T and T->close"))
end

# ── the automaton ────────────────────────────────────────────────────────────

"""
    pair_mpo(L, J, vertices) -> MPO

`H = Σ_{i<j} J[i,j] Σ_t coeff_t * left_t(i) * right_t(j)` as an EXACT MPO.

`J` is any `L x L` matrix; only its strict upper triangle is read, so a symmetric matrix and
its upper triangle give the same operator and a pair is never counted twice. `vertices` is a
vector of [`PairVertex`](@ref); `pair_vertices(h::XXZChain)` produces them from a term list.

The channel a pair travels on remembers WHERE IT OPENED -- that is what lets `J[i,j]` depend
on both ends and is why the virtual dimension grows with `L`. Channels are trimmed per link
to those that are both already open and still able to close, so `w` is `2 + n*(live openings)`
rather than `2 + n*(L-1)` everywhere: at link 1 nothing is open yet and at link `L-1` a
channel that no surviving `J` can close is dropped.

`J[i,i+1]`-only reproduces [`mpo_from_terms`](@ref) elementwise, which is how the
construction is pinned against the already-validated nearest-neighbour MPO.

# One coupling matrix per vertex

`pair_mpo(L, Js::Vector, vertices)` takes ONE MATRIX PER VERTEX, for models whose operator types
have DIFFERENT ranges. The single-matrix form applies the same `J[i,j]` to every vertex scaled by
`vertex.coeff`, which cannot express a Hamiltonian like the lattice Schwinger model, whose hopping
is NEAREST-NEIGHBOUR while its Coulomb term is ALL-TO-ALL:

    aH = x * sum_n (S+_n S-_{n+1} + h.c.)  +  (1/2) sum_{k>n} (N-k-1) Z_n Z_k

-- two different supports, so no single `J` with per-vertex scalars reproduces both. One matrix
per vertex does, at no extra bond dimension: the channel count is set by how many OPENINGS are
still closable, and a vertex whose coupling vanishes past its range stops contributing channels.

⚠️ `Js[t]` PAIRS WITH `vertices[t]` BY POSITION. A permuted list silently builds a different
Hamiltonian -- still Hermitian, still conserving the same charges, so nothing downstream
complains.
"""
function pair_mpo(L::Int, Js::Vector{<:AbstractMatrix}, vertices::Vector{PairVertex})

    length(Js) == length(vertices) || throw(DimensionMismatch(
        "pair_mpo got $(length(Js)) coupling matrices for $(length(vertices)) vertices; " *
        "they pair by position"))
    for (t, J) in pairs(Js)
        size(J) == (L, L) || throw(DimensionMismatch(
            "coupling matrix $t is $(size(J)), expected ($L, $L)"))
    end
    return _pair_mpo(L, collect(Js), vertices)
end

function pair_mpo(L::Int, J::AbstractMatrix, vertices::Vector{PairVertex})
    L >= 2 || throw(ArgumentError("pair_mpo needs at least two sites, got $L"))
    size(J) == (L, L) || throw(DimensionMismatch(
        "coupling matrix is $(size(J)), expected ($L, $L)"))
    return _pair_mpo(L, [J for _ in vertices], vertices)
end

function _pair_mpo(L::Int, Js::Vector, vertices::Vector{PairVertex})
    L >= 2 || throw(ArgumentError("pair_mpo needs at least two sites, got $L"))
    n = length(vertices)
    n >= 1 || throw(ArgumentError("pair_mpo needs at least one vertex"))
    for (k, v) in pairs(vertices)
        nl, nr = length(v.left.inds), length(v.right.inds)
        (nl in (2, 3) && nr in (2, 3)) || throw(ArgumentError(
            "pair_mpo: vertex $k has operators of rank $nl / $nr; only 2 or 3 are operators"))
        nl == nr || throw(ArgumentError(
            "pair_mpo: vertex $k joins a rank-$nl to a rank-$nr operator -- the two halves " *
            "must both carry an op-leg or neither"))
    end
    # The transported identity has to come from the mode's OWN local space. `local_space()`
    # deliberately refuses `:Z2` -- LurCGT cannot derive a mod-2 charge from a commutator, so the
    # Ising space is built below the IROP layer instead (see `z2_ising.jl`) -- and hard-coding
    # `local_space()` here is what previously made every Z2 model unreachable through `pair_mpo`
    # even though `_charged_transport` handles charged channels perfectly well.
    Iloc = symmetry_mode() === :SU2 ? local_space(:SU2).I :
           symmetry_mode() === :Z2  ? ising_local_space(:Z2).I : local_space().I

    # `coupling(o, j, t)` is the weight of the pair FOR VERTEX `t`, strict upper triangle only.
    coupling(o, j, t) = o < j ? Js[t][o, j] : 0.0
    # ...and `coupling(o, j)` is the weight over ANY vertex, which is what decides whether a
    # channel is worth carrying. A channel must stay alive if EITHER operator type still has a
    # partner ahead of it; testing only one would silently drop the longer-ranged term.
    coupling(o, j) = o < j ? maximum(abs(Js[t][o, j]) for t in eachindex(Js)) : 0.0
    # A channel opened at `o` is still useful at link `i` iff some later site can close it.
    closable(o, i) = any(coupling(o, j) != 0.0 for j in (i + 1):L)
    # Openings alive on link `i` (i.e. between sites `i` and `i+1`): opened at `o <= i`.
    alive(i) = [o for o in 1:min(i, L - 1) if closable(o, i)]

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid  = _mpo_block(Iloc, i, tl, tr, :none)
        Znil = to_concrete(0.0 * Iid)

        # Opening/closing/transport blocks for this site, one set per vertex.
        opens  = [_mpo_block(v.left,  i, tl, tr, _side(v.left,  :left))  for v in vertices]
        closes = [_mpo_block(v.right, i, tl, tr, _side(v.right, :right)) for v in vertices]
        # The transport block is built from the VERTEX rather than from `opens`/`closes`, because
        # verifying it needs partner blocks tagged for site `i+1` -- see `_charged_transport`.
        trans  = [_charged_transport(vertices[t], Iloc, i, tl, tr) for t in 1:n]

        prev, cur = alive(i - 1), alive(i)
        # Row index of channel `(o,t)` in the incoming set, column index in the outgoing one.
        rowof = Dict((o, t) => 1 + (k - 1) * n + t
                     for (k, o) in enumerate(prev) for t in 1:n)
        colof = Dict((o, t) => 1 + (k - 1) * n + t
                     for (k, o) in enumerate(cur) for t in 1:n)
        nrow, ncol = 2 + n * length(prev), 2 + n * length(cur)
        mat = Matrix{Any}(nothing, nrow, ncol)

        # `id -> id`: nothing has opened yet. The guard is `i < L` -- exactly `mpo_from_terms`'s
        # -- and NOT `!isempty(cur)`.
        #
        # ⛔ Guarding on `cur` looks right (why carry `id` across a link where nothing is open?)
        # and is wrong twice over. `cur` is the set of channels open ON THIS LINK; a term that
        # opens at a LATER site still needs `id` to reach it, so dropping the block SEVERS the
        # automaton. And because column 1 is then entirely `nothing`, `oplus` has no source for
        # the outgoing space of that column's zero blocks and throws
        #   "cannot infer zero TLArray at (1, 1): missing space information on leg 4"
        # (or leg 1, when row 1 empties too -- same cause, whichever leg Telum checks first).
        #
        # `alive(i)` is empty at an interior link exactly when the coupling graph is DISCONNECTED
        # across that bond. Nearest-neighbour and Haldane-Shastry are connected everywhere, which
        # is why nothing in the suite reached it. Carrying `id` where no term follows costs one
        # virtual dimension and changes no matrix element: `id` can only leave via an opening.
        i < L && (mat[1, 1] = Iid)
        # `done -> done`: a complete pair already lies behind us.
        i > 1 && (mat[nrow, ncol] = Iid)
        # `id -> (i,t)`: OPEN a channel at site `i`.
        for t in 1:n
            haskey(colof, (i, t)) && (mat[1, colof[(i, t)]] = opens[t])
        end
        # `(o,t) -> (o,t)`: TRANSPORT an open channel across site `i`. The charged block.
        for o in prev, t in 1:n
            haskey(colof, (o, t)) && (mat[rowof[(o, t)], colof[(o, t)]] = trans[t])
        end
        # `(o,t) -> done`: CLOSE at site `i`, weighted by `J[o,i]` and the vertex coefficient.
        for o in prev, t in 1:n
            c = coupling(o, i, t) * vertices[t].coeff
            # ⛔ AN EXPLICIT ZERO, NOT A GAP. With ONE matrix per vertex a channel can be alive
            # (some vertex still has a partner ahead) while THIS vertex's coupling vanishes here
            # -- the mixed-range case the per-vertex form exists for. Leaving that entry unset
            # gives `oplus` a hole it cannot infer spaces for:
            #     ArgumentError: cannot infer zero TLArray at (3,1): missing space information
            # Under a SHARED matrix the situation never arose: `J[o,i] == 0` killed every vertex
            # at once, so whole rows were skipped consistently.
            mat[rowof[(o, t)], ncol] = to_concrete((c == 0.0 ? zero(c) : c) * closes[t])
        end

        # Boundaries: site 1 can only be in `id`, site `L` can only end `done`.
        if i == 1
            mat = reshape(mat[1, :], 1, ncol)
            ncol > 1 && (mat[1, ncol] = Znil)         # nothing can be `done` yet
        elseif i == L
            mat = reshape(mat[:, ncol], nrow, 1)
            nrow > 1 && (mat[1, 1] = Znil)            # nothing may stay `id` past the end
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

pair_mpo(L::Int, J::AbstractMatrix, h::XXZChain) = pair_mpo(L, J, pair_vertices(h))

"""
    nn_coupling_matrix(L) -> Matrix{Float64}

`J[i,i+1] = 1`, everything else zero: the nearest-neighbour coupling, so that
`pair_mpo(L, nn_coupling_matrix(L), h)` must equal `mpo_from_terms(h)` as an operator. That
identity is the pin on this whole construction against the already-validated builder.
"""
function nn_coupling_matrix(L::Int)
    J = zeros(Float64, L, L)
    for i in 1:(L - 1)
        J[i, i + 1] = 1.0
    end
    return J
end

"""
    distance_coupling_matrix(L, r) -> Matrix{Float64}

`J[i,i+r] = 1` only: a single interaction distance. `r = 2` is the smallest case that
exercises the CHARGED TRANSPORT BLOCK at all -- a distance-1 coupling never transports
anything -- which is why it is the first thing the tests check after the `r = 1` pin.
"""
function distance_coupling_matrix(L::Int, r::Int)
    r >= 1 || throw(ArgumentError("distance must be at least 1, got $r"))
    J = zeros(Float64, L, L)
    for i in 1:(L - r)
        J[i, i + r] = 1.0
    end
    return J
end
