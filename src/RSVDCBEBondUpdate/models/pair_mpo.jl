# An exact MPO for an arbitrary two-body coupling matrix `J[i,j]`, charged (rank-3) operators
# included: one channel per opening site, virtual dimension `2 + n * (live openings)`.

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
            T = to_concrete!(contract(x, (length(x.inds),), y, (1,)))
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

The diagonal entry `delta_{w_l w_r} (x) I_site` of a channel open across site `i`, with the
channel's (possibly charged) virtual space. Every leg assignment is built and the one that
contracts correctly against opening -> transport, transport -> transport and transport -> closing
is returned. Uncharged channels are detected by comparing spaces, not dimensions.
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
    # Placeholder tags: `getIdentity` gives both legs the same tag and arrow, which would make an
    # illegal TLArray once combined with the site identity.
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
    pair_mpo(L, Js::Vector, vertices) -> MPO

`H = Σ_{i<j} J[i,j] Σ_t coeff_t * left_t(i) * right_t(j)` as an exact MPO. Only the strict upper
triangle of `J` is read; `vertices` come from [`pair_vertices`](@ref). The vector form takes one
coupling matrix per vertex (paired by position), for vertices with different ranges.
"""
function pair_mpo(L::Int, Js::Vector{<:AbstractMatrix}, vertices::Vector{PairVertex};
                  Iloc = nothing, Itrans = nothing, field = nothing, triples = TripleTerm[])

    length(Js) == length(vertices) || throw(DimensionMismatch(
        "pair_mpo got $(length(Js)) coupling matrices for $(length(vertices)) vertices; " *
        "they pair by position"))
    for (t, J) in pairs(Js)
        size(J) == (L, L) || throw(DimensionMismatch(
            "coupling matrix $t is $(size(J)), expected ($L, $L)"))
    end
    return _pair_mpo(L, collect(Js), vertices; Iloc = Iloc, Itrans = Itrans, field = field,
                     triples = triples)
end

"""
    TripleTerm(A, pair, w)

A nearest-neighbour THREE-site term `sum_j w[j] A_j B_{j+1} C_{j+2}` (`j = 1..L-2`), with
`pair = PairVertex(B, C, coeff)` an ordinary (charged or neutral) two-body vertex and `A` a
charge-NEUTRAL rank-2 operator: the facilitation factor `n_{j-1}` of the U(1) East model (plan
OS-3), `n_{j-1} (c'_j c_{j+1} + h.c.)`, which `pair_mpo`'s two-body automaton cannot express.

Because `A` is neutral the term needs no new tensor: in the automaton it is `id -(A)-> pre ->(B
opens, as a pair's left half)-> mid -(C closes)-> done`, and `pre` carries the vacuum virtual space.
Two virtual dimensions per term on the links it crosses. `w` may be complex (a Lindbladian's +-i).
"""
struct TripleTerm
    A::Any
    pair::PairVertex
    w::Vector{ComplexF64}
end

function pair_mpo(L::Int, J::AbstractMatrix, vertices::Vector{PairVertex}; Iloc = nothing,
                  Itrans = nothing, field = nothing, triples = TripleTerm[])
    L >= 2 || throw(ArgumentError("pair_mpo needs at least two sites, got $L"))
    size(J) == (L, L) || throw(DimensionMismatch(
        "coupling matrix is $(size(J)), expected ($L, $L)"))
    return _pair_mpo(L, [J for _ in vertices], vertices; Iloc = Iloc, Itrans = Itrans,
                     field = field, triples = triples)
end

function _pair_mpo(L::Int, Js::Vector, vertices::Vector{PairVertex}; Iloc = nothing,
                   Itrans = nothing, field = nothing, triples = TripleTerm[])
    L >= 2 || throw(ArgumentError("pair_mpo needs at least two sites, got $L"))
    n = length(vertices)
    m = length(triples)
    n + m >= 1 || throw(ArgumentError("pair_mpo needs at least one vertex or triple"))
    for (k, tt) in pairs(triples)
        length(tt.A.inds) == 2 || throw(ArgumentError(
            "pair_mpo: triple $k's first operator must be charge-neutral (rank 2)"))
        length(tt.w) == L - 2 || throw(DimensionMismatch(
            "pair_mpo: triple $k needs $(L - 2) weights (one per j = 1..L-2), got $(length(tt.w))"))
        nl, nr = length(tt.pair.left.inds), length(tt.pair.right.inds)
        (nl in (2, 3) && nl == nr) || throw(ArgumentError(
            "pair_mpo: triple $k's pair joins a rank-$nl to a rank-$nr operator"))
    end
    for (k, v) in pairs(vertices)
        nl, nr = length(v.left.inds), length(v.right.inds)
        (nl in (2, 3) && nr in (2, 3)) || throw(ArgumentError(
            "pair_mpo: vertex $k has operators of rank $nl / $nr; only 2 or 3 are operators"))
        nl == nr || throw(ArgumentError(
            "pair_mpo: vertex $k joins a rank-$nl to a rank-$nr operator -- the two halves " *
            "must both carry an op-leg or neither"))
    end
    # The transported identity comes from the mode's own local space, or `Iloc` when the site is not the
    # mode's default (e.g. a fused d = 4 Liouville site).
    Iloc = Iloc !== nothing ? Iloc :
           symmetry_mode() === :SU2 ? local_space(:SU2).I :
           symmetry_mode() === :Z2  ? ising_local_space(:Z2).I : local_space().I
    # `Itrans` is the operator carried across open channels; `σ^z` with a range-2 coupling gives the
    # interleaved fermionic hop. The identity channels keep `Iloc`.
    Itrans = Itrans !== nothing ? Itrans : Iloc

    # `coupling(o, j, t)` is the weight of the pair FOR VERTEX `t`, strict upper triangle only.
    coupling(o, j, t) = o < j ? Js[t][o, j] : 0.0
    # ...and `coupling(o, j)` is the weight over ANY vertex, which is what decides whether a
    # channel is worth carrying. A channel must stay alive if EITHER operator type still has a
    # partner ahead of it; testing only one would silently drop the longer-ranged term.
    coupling(o, j) = o < j && !isempty(Js) ? maximum(abs(Js[t][o, j]) for t in eachindex(Js)) : 0.0
    # A channel opened at `o` is still useful at link `i` iff some later site can close it.
    closable(o, i) = any(coupling(o, j) != 0.0 for j in (i + 1):L)
    # Openings alive on link `i` (i.e. between sites `i` and `i+1`): opened at `o <= i`.
    alive(i) = [o for o in 1:min(i, L - 1) if closable(o, i)]
    # Triple states on link `i`: `pre` after `A` placed at `i` (a term starting at `j = i`), `mid`
    # after `B` placed at `i` (a term that started at `j = i - 1`). Only where the weight is nonzero.
    tstates(i) = vcat([(t, :pre) for t in 1:m if 1 <= i <= L - 2 && triples[t].w[i] != 0],
                      [(t, :mid) for t in 1:m if 2 <= i <= L - 1 && triples[t].w[i - 1] != 0])

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
        trans  = [_charged_transport(vertices[t], Itrans, i, tl, tr) for t in 1:n]

        prev, cur = alive(i - 1), alive(i)
        # Row index of channel `(o,t)` in the incoming set, column index in the outgoing one.
        rowof = Dict((o, t) => 1 + (k - 1) * n + t
                     for (k, o) in enumerate(prev) for t in 1:n)
        colof = Dict((o, t) => 1 + (k - 1) * n + t
                     for (k, o) in enumerate(cur) for t in 1:n)
        tprev, tcur = tstates(i - 1), tstates(i)
        trow = Dict(s => 1 + n * length(prev) + k for (k, s) in enumerate(tprev))
        tcol = Dict(s => 1 + n * length(cur) + k for (k, s) in enumerate(tcur))
        nrow, ncol = 2 + n * length(prev) + length(tprev), 2 + n * length(cur) + length(tcur)
        mat = Matrix{Any}(nothing, nrow, ncol)

        # `id -> id` on every link but the last (as `mpo_from_terms`): a term opening later still needs it,
        # and an empty column would leave `oplus` without spaces.
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
            # An explicit zero, not a gap: with one matrix per vertex a channel can be alive while this
            # vertex's coupling vanishes, and `oplus` cannot infer a hole.
            mat[rowof[(o, t)], ncol] = to_concrete((c == 0.0 ? zero(c) : c) * closes[t])
        end
        # Triples: `id -(A)-> pre`, `pre -(B)-> mid`, `mid -(w C)-> done` -- the same opening and
        # closing block forms as a pair, with a neutral (vacuum) `pre` state in front.
        for (t, tt) in pairs(triples)
            haskey(tcol, (t, :pre)) && (mat[1, tcol[(t, :pre)]] = _mpo_block(tt.A, i, tl, tr, :none))
            haskey(trow, (t, :pre)) && haskey(tcol, (t, :mid)) &&
                (mat[trow[(t, :pre)], tcol[(t, :mid)]] =
                     _mpo_block(tt.pair.left, i, tl, tr, _side(tt.pair.left, :left)))
            if haskey(trow, (t, :mid))
                c = tt.w[i - 2] * tt.pair.coeff
                mat[trow[(t, :mid)], ncol] = to_concrete((isreal(c) ? real(c) : c) *
                    _mpo_block(tt.pair.right, i, tl, tr, _side(tt.pair.right, :right)))
            end
        end

        # `id -> done`: an ON-SITE term. No channel opens, so it costs no virtual dimension.
        field !== nothing && (mat[1, ncol] = field[i])

        # Boundaries: site 1 can only be in `id`, site `L` can only end `done`.
        if i == 1
            mat = reshape(mat[1, :], 1, ncol)
            # ⛔ `mat[1, ncol]` IS THE FIELD'S SLOT. Blanking it unconditionally drops `h_1` here
            # and `h_L` below -- an error that shrinks like 1/L and therefore reads as a
            # convergence artefact rather than a wrong Hamiltonian. Same trap
            # `field_mpo_from_terms` documents; the guard is why it is not repeated here.
            ncol > 1 && field === nothing && (mat[1, ncol] = Znil)
        elseif i == L
            mat = reshape(mat[:, ncol], nrow, 1)
            nrow > 1 && field === nothing && (mat[1, 1] = Znil)
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
