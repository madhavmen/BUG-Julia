# Channel environments for a nearest-neighbour XXZ chain: the MPO contraction with the virtual leg
# named, so each channel's charge rides on the op-leg of Telum's Sp/Sm.
# The bra leg is primed and the ket unprimed, so the bra tensor differs at each boundary. A wrong
# prime does not throw; it silently traces the wrong pair, which is why `env_energy` is checked
# against the gate energy.

# ── the Hamiltonian as a term list ───────────────────────────────────────────

"""
    XXZTerm(left, right, coeff)

One nearest-neighbour term `coeff · left_i ⊗ right_{i+1}`, repeated on every bond.

`left` and `right` are bare local operators; they are retagged onto their site at
contraction time. Either may be rank-3, carrying an `op` leg whose charge is what
makes the pair U(1)-allowed; when both are, that leg is contracted between them (this
is the XY hop, and it is why the two operators must be a `Sp`/`Sp'` adjoint pair
rather than `Sp`/`Sm` -- see `gates.jl`'s header on the 1/√2 normalisation).
"""
struct XXZTerm
    left::Any
    right::Any
    coeff::Float64
end

"""
    XXZChain(L, terms, J, delta)

`H = J Σ_i (Sx Sx + Sy Sy + delta Sz Sz)` on `L` sites, as a term list.
Built by [`xxz_chain`](@ref); `delta = 0` is XX.
"""
struct XXZChain
    L::Int
    terms::Vector{XXZTerm}
    J::Float64
    delta::Float64
end

Base.length(h::XXZChain) = h.L

"""
    xxz_chain(L; J=1.0, delta=1.0) -> XXZChain

The Hamiltonian of `bond_gates` as a term list instead of a gate per bond. The XY term is
factorised per mode: under `:U1` it is `pair(Sp,Sp) + pair(Sm,Sm)` (the 1/2 is already in the
pair); under `:none` the coefficient is -1, since Telum normalises `Sp = -(1/sqrt2) S+_std`.
Both are measured and pinned by `test_gates.jl`.
"""
function xxz_chain(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0)
    L >= 2 || throw(ArgumentError("xxz_chain needs at least two sites, got $L"))
    q = local_space()
    terms = XXZTerm[]
    if length(q.Sp.inds) == 3
        push!(terms, XXZTerm(q.Sp, to_concrete(q.Sp'), J))
        push!(terms, XXZTerm(q.Sm, to_concrete(q.Sm'), J))
    else
        push!(terms, XXZTerm(q.Sp, q.Sm, -J))
        push!(terms, XXZTerm(q.Sm, q.Sp, -J))
    end
    delta == 0.0 || push!(terms, XXZTerm(q.Sz, q.Sz, J * delta))
    return XXZChain(L, terms, J, delta)
end

"""
    heisenberg_su2_chain(L; J=1.0) -> XXZChain

`H = J sum_i S_i . S_{i+1}` as ONE irreducible term for `symmetry_mode() === :SU2`, where Telum
exposes only `:S` and `:I`. Its two halves are joined by an `S = 1` op-leg, in place of the three
abelian terms; the downstream code is unchanged.

Only the isotropic point exists, so an anisotropic chain runs under `:U1` with [`xxz_chain`](@ref).
The overall constant is measured (Clebsch-Gordan scaling), pinned against dense diagonalisation.
"""
function heisenberg_su2_chain(L::Int; J::Float64 = 1.0)
    L >= 2 || throw(ArgumentError("heisenberg_su2_chain needs at least two sites, got $L"))
    symmetry_mode() === :SU2 || throw(ArgumentError(
        "heisenberg_su2_chain needs symmetry_mode() === :SU2, got $(symmetry_mode()); " *
        "use xxz_chain for :U1 or :none"))
    q = local_space(:SU2)
    # One term, op-leg contracted between the two halves -- the same shape the rank-3 `:U1`
    # branch of `xxz_chain` builds for `Sp`/`Sm`.
    terms = [XXZTerm(q.S, to_concrete(q.S'), J)]
    return XXZChain(L, terms, J, 1.0)
end

"The term list of `h`. `hamiltonian_terms(h)[t]` pairs with `open[i][t]`."
hamiltonian_terms(h::XXZChain) = h.terms

# A channel with no contribution yet is not the boundary identity: `done` starts at ZERO, while `id`
# starts at the boundary identity (`nothing`). Conflating them is a silent factor-of-one bug.

struct ZeroEnv end
const ZERO = ZeroEnv()

_accum(a::ZeroEnv, b) = b
_accum(a, b) = to_concrete(a + b)

# A sweep carries the channels from link i to link i+1 through site i, O(1) per bond instead of an
# O(L) rebuild. `canonical!` rewrites tensors, so a stack built before it is stale: canonicalise
# first, then build the one stack the other side needs, then sweep.

"""
    ChannelSet

The channel environment on ONE link: `id` (the transported identity), `done` (completed
terms) and `open[t]` (term `t`'s dangling half). `id === nothing` is a boundary; `done` or
`open[t]` may be `ZERO`.
"""
struct ChannelSet
    id::Any
    done::Any
    open::Vector{Any}
end

"The channels on a chain boundary: identity transported from nothing, no terms yet."
boundary_channels(h::XXZChain) = ChannelSet(nothing, ZERO, Any[ZERO for _ in h.terms])

"""
    push_left_channels(cs, h, A, i; open_next=true) -> ChannelSet

Carry the left channels on link `i` through site tensor `A` onto link `i+1`.

`open_next=false` suppresses opening new half-terms, for the last site of a chain where a
term opened would never be closed.
"""
function push_left_channels(cs::ChannelSet, h::XXZChain, A, i::Int;
                            open_next::Bool = true)
    terms = h.terms
    acc = cs.done === ZERO ? ZERO : _left_step(cs.done, A, nothing, i)
    for (t, term) in enumerate(terms)
        cs.open[t] === ZERO && continue
        acc = _accum(acc, to_concrete(term.coeff *
                     _left_step(cs.open[t], A, term.right, i)))
    end
    nxt = Any[open_next ? _left_step(cs.id, A, terms[t].left, i) : ZERO
              for t in eachindex(terms)]
    return ChannelSet(_left_step(cs.id, A, nothing, i), acc, nxt)
end

"""
    push_right_channels(cs, h, A, i; open_next=true) -> ChannelSet

Mirror: carry the right channels on link `i+1` through site `i` onto link `i`. A right
channel OPENS with `term.right` and CLOSES with `term.left` -- the asymmetry `_right_step`
documents.
"""
function push_right_channels(cs::ChannelSet, h::XXZChain, A, i::Int;
                             open_next::Bool = true)
    terms = h.terms
    acc = cs.done === ZERO ? ZERO : _right_step(cs.done, A, nothing, i)
    for (t, term) in enumerate(terms)
        cs.open[t] === ZERO && continue
        acc = _accum(acc, to_concrete(term.coeff *
                     _right_step(cs.open[t], A, term.left, i)))
    end
    nxt = Any[open_next ? _right_step(cs.id, A, terms[t].right, i) : ZERO
              for t in eachindex(terms)]
    return ChannelSet(_right_step(cs.id, A, nothing, i), acc, nxt)
end

# ── local pieces ─────────────────────────────────────────────────────────────

"Retag an operator's two site legs onto site `i`, leaving any op-leg alone."
_retag_sites(O, i::Int) = to_concrete(setitag(setitag(O, 1, "S,$i"), 2, "S,$i"))

"""
    _apply_site_op(A, O, i) -> TLArray

`O_i A`, returned in `A`'s own layout with any op-leg appended:
`(link_l, site, link_r)` or `(link_l, site, link_r, op)`.

The ket-facing leg of `O` is found by ARROW, not position: a plain operator has it at
leg 1, an adjoint (`Sp'`) at leg 2, and `contract` asserts opposite arrows, so
hard-coding leg 1 breaks on exactly the adjoint half of every XY term.
"""
function _apply_site_op(A, O, i::Int)
    O === nothing && return A
    Oi = _retag_sites(O, i)
    ket_leg = Oi.inds[1].dir == '+' ? 1 : 2
    T = contract(Oi, (ket_leg,), A, (2,))
    # T = (O's remaining legs..., A's link_l, A's link_r). The surviving site leg of a
    # rank-3 O is always at index < 3, so its remainder is (site_out, op) in order.
    return length(Oi.inds) == 2 ? to_concrete(permutedims(T, (2, 1, 3))) :
                                  to_concrete(permutedims(T, (3, 1, 4, 2)))
end

"Bra tensor for an interior step: both link legs primed, so each pairs with the
environment's bra leg rather than with the ket."
_bra_interior(A) = to_concrete(prime(prime(A, 1), 3)')

"Bra tensor at the left boundary: `link_l` unprimed, because there the bra and ket
links are the same dim-1 vacuum leg and must contract with each other."
_bra_left_boundary(A) = to_concrete(prime(A, 3)')

"Bra tensor closing the right boundary: `link_r` unprimed, for the same reason."
_bra_right_boundary(A) = to_concrete(prime(A, 1)')

# ── one left step ────────────────────────────────────────────────────────────

"""
    _left_step(E, A, O, i) -> TLArray

Push the left environment `E` at link `i` through site `i`, inserting `O` on the ket; `E === nothing`
is the boundary identity. Returns `(bra, ket [, op])`.

Distinguished by rank: rank-2 `E` with an op-free `O` gives rank 2 (transport); rank-2 `E` with a
rank-3 `O` opens a channel; rank-3 `E` with a rank-3 `O` closes one. Rank-3 `E` with an op-free `O`
would carry a half-open term past a site, so it is rejected rather than silently building a
longer-range Hamiltonian.
"""
function _left_step(E, A, O, i::Int)
    AO = _apply_site_op(A, O, i)
    nAO = length(AO.inds)

    if E === nothing
        # Boundary: contract link_l bra-to-ket along with the site leg.
        out = contract(AO, (1, 2), _bra_left_boundary(A), (1, 2))
    else
        nE = length(E.inds)
        if nE == 3 && nAO == 4
            tmp = contract(AO, (1, 4), E, (2, 3))       # close: pair the op-legs too
        elseif nE == 2
            tmp = contract(AO, (1,), E, (2,))
        else
            error("_left_step: rank-$nE environment with a rank-$nAO operator would " *
                  "carry an open channel past site $i")
        end
        # tmp = (site, link_r, [op], bra) -- the bra leg is always last.
        out = contract(tmp, (1, length(tmp.inds)), _bra_interior(A), (2, 1))
    end

    # out = (ket, [op], bra); put the bra first.
    n = length(out.inds)
    return to_concrete(permutedims(out, n == 2 ? (2, 1) : (3, 1, 2)))
end

"""
    _left_close(E, A, O, i) -> ComplexF64

As `_left_step`, but at the last site: `link_r` is the dim-1 vacuum boundary, so it is
contracted bra-to-ket and the result is a scalar. A channel still open here would mean
a term running off the end of the chain, so that is an error.
"""
function _left_close(E, A, O, i::Int)
    AO = _apply_site_op(A, O, i)
    nAO, nE = length(AO.inds), length(E.inds)
    tmp = (nE == 3 && nAO == 4) ? contract(AO, (1, 4), E, (2, 3)) :
          nE == 2               ? contract(AO, (1,), E, (2,))     :
          error("_left_close: rank-$nE environment with a rank-$nAO operator at site $i")
    length(tmp.inds) == 3 || error(
        "_left_close: an operator channel is still open at the right boundary")
    s = to_concrete!(contract(tmp, (1, 2, 3), _bra_right_boundary(A), (2, 3, 1)))
    return ComplexF64(s[])
end
