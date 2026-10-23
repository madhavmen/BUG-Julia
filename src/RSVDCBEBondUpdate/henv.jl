# Channel environments for a nearest-neighbour XXZ chain.
#
# WHY THIS EXISTS. `BondUpdateBUG` is gate-based: `bond_gates` builds one two-site
# gate per bond and `parity_sweep!` Trotterises them. Both halves of CBE-BUG need `H`
# GLOBALLY instead -- the CBE sketch applies `H` through the environments
# (`RSVDpreBE0SiQS.m:336`), and the single centre-bond Galerkin step needs the
# effective Hamiltonian of the whole chain, not of one bond. Telum is a tensor
# library only (no MPS/MPO layer), so this is built here.
#
# NOT A SYMMETRIC MPO. For H = Σ_i Σ_t c_t O^t_i P^t_{i+1} the sum of local terms
# closes under a finite-state recursion with three channel kinds:
#
#   id[i+1]     = A†( id[i] )A                        the transported identity
#   open[i+1,t] = A†( id[i] ⊗ O^t_i )A                half a term, awaiting its partner
#   done[i+1]   = A†( done[i] )A + Σ_t c_t A†( open[i,t] ⊗ P^t_i )A
#
# That is an MPO contraction with the virtual leg NAMED rather than fused, and naming
# it is the point: the charge each channel carries arrives on the op-leg of Telum's
# own `Sp`/`Sm` (`docs/telum_api_contract.md` §1), so there is no charge-graded virtual
# space to assemble by hand and no AutoMPO to write. O(L) build, `2 + |terms|` tensors
# per link.
#
# LEG AND PRIME CONVENTION. Every environment carries `(bra, ket [, op])` with the
# BRA leg primed and the ket leg unprimed, so:
#
#   * the ket leg pairs with the next site tensor's link leg (both unprimed), and
#   * the bra leg pairs with the bra tensor's link leg (both primed).
#
# `contract` asserts equal itags and opposite arrows (§7b), and the two link legs of a
# bond share an itag, so priming is the only thing keeping the bra and ket namespaces
# apart. Consequently the bra tensor differs by position in the chain:
#
#   interior       `prime(prime(A,1),3)'`   both links primed
#   left boundary  `prime(A,3)'`            link_l unprimed -- it pairs bra-to-ket
#   right closure  `prime(A,1)'`            link_r unprimed -- ditto, giving a scalar
#
# Getting this wrong does not throw; it silently traces the wrong pair (§7a). The
# guard is `env_energy == energy(psi, gates)`, the gate path being pinned to the
# analytic Heisenberg block element by element in `test_gates.jl`.

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

The same Hamiltonian `bond_gates` builds, as a term list instead of a gate per bond.

The two symmetry modes need different factorisations of the XY term, exactly as
`heisenberg_bond_gate` does:

  - `:U1` — `Sp`/`Sm` are rank-3 with a ±2 charge on the op-leg, and the physical term
    is the op-leg contraction of an operator with its own adjoint: `pair(Sp,Sp) +
    pair(Sm,Sm)`, which already carries the 1/2 of `½(S⁺S⁻ + S⁻S⁺)`.
  - `:none` — they are plain matrices with no op-leg, Telum normalises them to
    `Sp = -(1/√2)S⁺_std`, and the coefficient reproducing the physical XY term is `-1`.

Both are MEASURED facts about Telum's normalisation, pinned by `test_gates.jl`; this
function only mirrors them, so `env_energy` and `energy(psi, bond_gates(psi))` must
agree to machine precision.
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

`H = J Σ_i S_i · S_{i+1}` on `L` sites, as a SINGLE term, for `symmetry_mode() === :SU2`.

WHY THIS EXISTS SEPARATELY FROM [`xxz_chain`](@ref), rather than as a branch inside it. Under
SU(2) Telum's local space exposes `:S` and `:I` only -- there is no `:Sp`, `:Sz`, `:Sm`, and
there cannot be, because none of those three is an SU(2) tensor on its own. Only the full
vector `S` is, so the coupling is ONE irreducible term whose two halves are joined by an
`S = 1` op-leg, in place of the three (`Sp Sm`, `Sm Sp`, `Sz Sz`) the abelian form needs.

NOTHING DOWNSTREAM HAD TO CHANGE FOR THIS, which is the useful part: the channel environments
already carry operators with op-legs, since that is how `:U1`'s `Sp`/`Sm` work (see the rank-3
branch in `xxz_chain`). So `push_left_channels`, `apply_h_two_site`, `zero_site_h` and
`one_site_h` take this term list unmodified.

ONLY THE ISOTROPIC POINT EXISTS HERE. Any `delta != 1` breaks SU(2) down to U(1), so there is
no `delta` argument -- an anisotropic chain has to be run under `:U1` with `xxz_chain`.

THE OVERALL CONSTANT IS MEASURED, NOT ASSUMED. Telum normalises its spin operators through
Clebsch-Gordan coefficients (`get_SU2_symmops` scales by `sqrt(S(S+1))`), exactly the reason
`heisenberg_bond_gate` carries a measured `-1` on its `:none` branch. The check that pins it is
the ground energy against dense diagonalisation, in `tests/RSVDCBEBondUpdate/test_su2.jl`.
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

# ── zero sentinel ────────────────────────────────────────────────────────────
#
# A channel that has no contribution yet is NOT the same as the boundary identity, and
# conflating them is a silent factor-of-one bug: `done` starts at ZERO (no completed
# terms) while `id` starts at the boundary identity (represented by `nothing`, which
# `_left_step` reads as "contract the bra and ket links together").

struct ZeroEnv end
const ZERO = ZeroEnv()

_accum(a::ZeroEnv, b) = b
_accum(a, b) = to_concrete(a + b)

# ── one link's channels, and how a sweep carries them ────────────────────────
#
# A sweep does NOT need to rebuild an environment stack at every bond. The channels on
# link `i+1` are the channels on link `i` pushed through site `i` -- O(1) per bond, against
# O(L) for a rebuild. Carrying them is what makes a step O(L) instead of O(L^2), and it is
# how the chain BUG in qtci's `exploratory_bug/src/BUG/discarded_bug.jl` works: its K-sweep
# carries `aps`/`aph` site to site and precomputes its environments exactly once.
#
# ORDERING CONSTRAINT, which a rebuild hides. `canonical!` rewrites tensors, so a stack
# precomputed before it is stale. A carried sweep must therefore canonicalise FIRST, then
# build the one stack it needs for the opposite side, then sweep -- see
# `cbe_lubich_sweep`, where each sweep only ever touches sites the other side's stack
# does not depend on.

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

Push the left environment `E` at link `i` through site `i`, inserting `O` on the ket.
`E === nothing` means the left boundary identity. Returns `(bra, ket [, op])`.

Three cases, distinguished by rank alone:

  - `E` rank 2, `O` without an op-leg → rank-2 out (transport, or a `:none` term);
  - `E` rank 2, `O` rank 3 → rank-3 out, the op-leg left dangling (OPEN a channel);
  - `E` rank 3, `O` rank 3 → rank-2 out, the two op-legs contracted (CLOSE it).

The fourth combination (`E` rank 3, `O` without an op-leg) would carry a half-open
term past a site without closing it. For nearest-neighbour terms that never happens,
so it is rejected rather than silently producing a longer-range Hamiltonian.
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
