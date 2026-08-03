# THE TRANSVERSE-FIELD ISING MODEL, WITH ITS Z2 SPIN-FLIP SYMMETRY MADE BLOCK-SPARSE.
#
#     H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i
#
# ── WHY THE SIGMA^X BASIS, AND NOT THE USUAL SIGMA^Z ONE ─────────────────────────────
#
# The symmetry of the TFIM is the GLOBAL SPIN FLIP `P = Π_i X_i`, which commutes with H. In
# the conventional sigma^z basis P is entirely off-diagonal, so nothing is block-diagonal and
# there are no sectors to exploit -- the symmetry is present but invisible to the tensors.
#
# Work in the sigma^x EIGENBASIS instead and the symmetry becomes a charge:
#
#     |+>  (X = +1)  ->  Z2 charge 0
#     |->  (X = -1)  ->  Z2 charge 1
#
# Now `X = diag(+1,-1)` is DIAGONAL, hence charge-preserving, and `Z = [0 1; 1 0]` FLIPS the
# charge by 1. The bond term `Z_i Z_{i+1}` changes the total charge by `1 + 1 = 0 (mod 2)` and
# the field term by 0, so H conserves Z2 charge exactly and every tensor is block-sparse in it.
#
# The price is that the basis is rotated relative to the textbook one: what is called `X` here
# is diagonal and what is called `Z` is the flip. The names refer to the OPERATORS, which are
# basis-independent, so `<X_j>` below is the genuine transverse magnetisation and `<Z_i Z_j>`
# the genuine order-parameter correlator -- only their matrix representations are rotated.
#
# ── WHY THIS FILE TOUCHES TELUM PRIVATES, WHICH IS DELIBERATE AND CONTAINED ──────────
#
# `getLocalSpace` CANNOT build a Z2 space, and not for want of an options type. LurCGT derives
# an operator's charge from the Lie-algebra commutator `[Q, O] = q*O` (`getweight_irop`), and
# the Z2 flip admits no such `q`: it raises 0->1 and lowers 1->0, which are THE SAME charge mod
# 2 but `+1` and `-1` as integers, so the assertion `comm_result ≈ zvals * mwirop` cannot hold.
# No choice of additive weights repairs it -- weights `±1` demand `q = +2` and `q = -2` at once.
# That is intrinsic to mod-N charges, not a bug.
#
# The representation is fine; only the DERIVATION is impossible. With an op leg of charge 1
# both matrix elements of the flip conserve charge,
#
#     <1|Z|0>:  1 = 0 + 1 (mod 2)        <0|Z|1>:  0 = 1 + 1 (mod 2)
#
# which is exactly what the commutator cannot see and the sector bookkeeping can. So the blocks
# are built directly, below the IROP layer, via `Telum._localspace_cgt_fields`.
#
# THAT FUNCTION IS PRIVATE (underscore, no stability guarantee). Every such call in this package
# lives in THIS FILE and nowhere else, so a Telum upgrade breaks one file rather than the code
# base, and `tests/BondUpdateBUG/test_z2_ising.jl` fails loudly if it does: it checks the Pauli
# algebra, charge conservation, and the L=8 ground energy against exact diagonalisation. If
# LurCGT ever grows a Z{N}-aware `getweight_irop`, delete `_z2_local_space` and call
# `getLocalSpace` -- nothing else here changes.

using Telum: TLArray, TLIndex, _localspace_cgt_fields, _getLocalSpace_no_symmetry
using LurCGT: Z
using SparseArrays

# ── the local space ──────────────────────────────────────────────────────────────────

const _Z2_Q0 = ((0,),)      # sigma^x = +1
const _Z2_Q1 = ((1,),)      # sigma^x = -1

# `_localspace_cgt_fields` wants `Vector{Tuple{NTuple{QD, NTuple{N, Tuple{Vararg{Int}}}}, ...}}`
# and Array is INVARIANT in its element type, so `Tuple{Tuple{Int}}` does NOT match
# `Tuple{Tuple{Vararg{Int}}}` and dispatch fails with a confusing MethodError. Spell them out.
const _QT   = Tuple{Vararg{Int}}
const _DAT2 = Tuple{NTuple{2, NTuple{1, _QT}}, Array{Float64, 3}}
const _DAT3 = Tuple{NTuple{3, NTuple{1, _QT}}, Array{Float64, 4}}

const _ISING_SPACES = Dict{Symbol, Any}()

"Charge-preserving (diagonal) rank-2 operator with entries `v0`, `v1` on the two sectors."
function _z2_diag_op(v0::Float64, v1::Float64)
    sp = [(_Z2_Q0, 1), (_Z2_Q1, 1)]
    data = _DAT2[((_Z2_Q0, _Z2_Q0), reshape([v0], 1, 1, 1)),
                 ((_Z2_Q1, _Z2_Q1), reshape([v1], 1, 1, 1))]
    spaces = (sp, sp)
    f = _localspace_cgt_fields(data, (Z{2},), spaces)
    return TLArray((Z{2},), f.qlabels, f.wmatdata, f.wmatinfo, f.RMTs,
                   (TLIndex("s", '+'), TLIndex("s", '-')), spaces)
end

"""
The charge-1 flip `Z`, as a rank-3 operator whose op leg carries charge 1.

BOTH matrix elements share one op-leg charge, which is the whole reason Z2 is representable
here: `1 = 0+1` and `0 = 1+1` are both true mod 2. Under U(1) this would need two different
op-leg charges (`+1` and `-1`) and would not be one irreducible operator at all.
"""
function _z2_flip_op()
    sp = [(_Z2_Q0, 1), (_Z2_Q1, 1)]
    opsp = [(_Z2_Q1, 1)]
    data = _DAT3[((_Z2_Q1, _Z2_Q0, _Z2_Q1), reshape([1.0], 1, 1, 1, 1)),
                 ((_Z2_Q0, _Z2_Q1, _Z2_Q1), reshape([1.0], 1, 1, 1, 1))]
    spaces = (sp, sp, opsp)
    f = _localspace_cgt_fields(data, (Z{2},), spaces)
    return TLArray((Z{2},), f.qlabels, f.wmatdata, f.wmatinfo, f.RMTs,
                   (TLIndex("s", '+'), TLIndex("s", '-'), TLIndex("op", '-')), spaces)
end

"""
    ising_local_space(sym = symmetry_mode()) -> (; I, X, Z)

The spin-½ local space for the TFIM **in the sigma^x basis**, for `:Z2` or `:none`.

`:Z2` carries two charge sectors and a rank-3 `Z`; `:none` is one dense block with a rank-2
`Z`. THE SAME BASIS IN BOTH, deliberately: it is what makes a `:Z2`-vs-`:none` comparison a
statement about storage rather than about two different bases, so the two must agree digit for
digit on every observable.
"""
function ising_local_space(sym::Symbol = symmetry_mode())
    haskey(_ISING_SPACES, sym) && return _ISING_SPACES[sym]
    s = if sym === :Z2
        (; I = _z2_diag_op(1.0, 1.0), X = _z2_diag_op(1.0, -1.0), Z = _z2_flip_op())
    elseif sym === :none
        # Same operators, one dense block. Telum's own no-symmetry builder, so the tensors are
        # constructed exactly as `local_space(:none)`'s are.
        ops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}(
            :I => spdiagm(0 => ones(Float64, 2)),
            :X => spdiagm(0 => Float64[1.0, -1.0]),
            :Z => sparse(Float64[0 1; 1 0]))
        _getLocalSpace_no_symmetry(ops, ("s", "s", "op"))
    else
        throw(ArgumentError(
            "the transverse-field Ising local space is defined for :Z2 and :none, got $sym. " *
            "The TFIM has no U(1) spin symmetry -- Z_i Z_{i+1} does not conserve Sz -- and no " *
            "SU(2) symmetry either, since the transverse field singles out an axis."))
    end
    _ISING_SPACES[sym] = s
    return s
end

# ── the Hamiltonian ──────────────────────────────────────────────────────────────────

"""
    ising_bond_gates(psi; J=1.0, h=1.0) -> Vector

One gate per bond for `H = -J Σ Z_i Z_{i+1} - h Σ X_i`, with the ON-SITE FIELD DISTRIBUTED
over the bonds that touch each site.

THE DISTRIBUTION IS NOT UNIFORM, AND GETTING IT WRONG IS A SILENT ERROR. A bond gate can only
carry two-site operators, so each site's field is split across the bonds it belongs to:
interior site `j` touches bonds `j-1` and `j` and takes `h/2` from each, while sites `1` and
`L` touch ONE bond apiece and must take the FULL `h` there. Splitting every site's field in
half would quietly evolve a chain with weakened boundary fields -- a different Hamiltonian that
still looks entirely plausible, and whose error shrinks with `L` so small-size tests miss it.
"""
function ising_bond_gates(psi::SymMPS; J::Float64 = 1.0, h::Float64 = 1.0)
    L = length(psi)
    L >= 2 || throw(ArgumentError("ising_bond_gates needs at least two sites, got $L"))
    return Any[ising_bond_gate(psi[i].inds[2], psi[i + 1].inds[2];
                               J = J,
                               hl = h / (i == 1 ? 1 : 2),
                               hr = h / (i == L - 1 ? 1 : 2))
               for i in 1:(L - 1)]
end

"""
    ising_bond_gate(site_l, site_r; J=1.0, hl=0.5, hr=0.5) -> TLArray

`-J Z⊗Z - hl X⊗I - hr I⊗X` on one bond, legs `(ket_l, ket_r, bra_l, bra_r)`.

`hl`/`hr` are this bond's SHARE of the on-site field, not the field itself -- see
[`ising_bond_gates`](@ref), which computes the shares.
"""
function ising_bond_gate(site_l, site_r; J::Float64 = 1.0,
                         hl::Float64 = 0.5, hr::Float64 = 0.5)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    tl == tr && throw(ArgumentError(
        "the two sites of a bond need distinct itags, both are '$tl'"))
    q = ising_local_space()

    pair(A, B) = to_concrete(permutedims(
        contract(_retag_site(A, tl), (3,), _retag_site(B, tr)', (3,)), (1, 4, 2, 3)))
    outer(A, B) = to_concrete(permutedims(
        contract(_retag_site(A, tl), (), _retag_site(B, tr), ()), (1, 3, 2, 4)))

    # Under :Z2 the flip is rank-3 (its op leg carries the charge), so Z⊗Z is the op-leg
    # contraction; under :none it is a plain matrix and the same operator is an outer product.
    zz = length(q.Z.inds) == 3 ? pair(q.Z, q.Z) : outer(q.Z, q.Z)
    G = (-J) * zz + (-hl) * outer(q.X, q.I) + (-hr) * outer(q.I, q.X)
    return to_concrete(G * (1.0 + 0.0im))
end

"""
    ising_chain(L; J=1.0, h=1.0) -> XXZChain

The same Hamiltonian as [`ising_bond_gates`](@ref) as a term list, for the MPO/Lubich path.

The transverse field is a ONE-SITE term, which `XXZChain` carries in its `onsite` slot rather
than folded into a bond -- the environment machinery applies it at every site, so no
distribution over bonds and no boundary special case is needed here.
"""
function ising_chain end   # defined in RSVDCBEBondUpdate, which owns XXZChain

# ── states and observables ───────────────────────────────────────────────────────────

"""
    ising_product_state(dirs::Vector{Symbol}) -> SymMPS

Product state in the sigma^x basis; each entry is `:plus` (`X = +1`) or `:minus` (`X = -1`).

Same fusion recursion as `product_state`, but selecting on the Z2 charge (`:Z2`) or on the
dense index (`:none`) -- the two modes select differently because `:none` has ONE sector of
dimension 2, so there is no charge to filter on, only a basis index.
"""
function ising_product_state(dirs::Vector{Symbol})
    L = length(dirs)
    L >= 1 || throw(ArgumentError("ising_product_state needs at least one site"))
    all(d -> d in (:plus, :minus), dirs) || throw(ArgumentError(
        "each entry must be :plus or :minus, got $(filter(d -> !(d in (:plus, :minus)), dirs))"))
    q = ising_local_space()
    z2 = symmetry_mode() === :Z2

    tensors = Any[]
    prev = (getvac(q.I, ("L,0", "L,1")), 2)
    for i in 1:L
        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        A = if z2
            target = dirs[i] === :plus ? _Z2_Q0 : _Z2_Q1
            B = to_concrete(getsub(F, 2, s -> s == target ? Colon() : nothing;
                                   preserve_space = true))
            present = Set(ql[3] for ql in B.qlabels)
            to_concrete(getsub(B, 3, s -> s in present ? Colon() : nothing))
        else
            # `_z2_diag_op`'s ordering is (|+>, |->), and `_getLocalSpace_no_symmetry` keeps it,
            # so index 1 is |+> and 2 is |->.
            to_concrete(getsub(F, 3, s -> dirs[i] === :plus ? 1 : 2))
        end
        A = to_concrete(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end
    return SymMPS(tensors, L)
end

"""
    ising_polarised_state(L) -> SymMPS

`|+ + ... +>`, every spin along `+x`: the exact ground state at `J = 0` and the natural quench
start. Z2 charge 0, `chi = 1` -- a genuine product state, so it is also a test of whether the
integrator can GROW rank at all (without CBE it would sit at `chi = 1` forever).
"""
ising_polarised_state(L::Int) = ising_product_state(fill(:plus, L))

"""
    ising_kink_state(L) -> SymMPS

`|+ ... + - ... ->` with the flip in the middle: a domain wall in the sigma^x basis. Z2 charge
`(L ÷ 2) mod 2`. Gives a light cone under a TFIM quench in the same way the Heisenberg domain
wall does.
"""
ising_kink_state(L::Int) =
    ising_product_state([i <= L ÷ 2 ? :plus : :minus for i in 1:L])

"""
    x_profile(psi) -> Vector{Float64}

`<X_j>` on every site -- the transverse magnetisation, and the TFIM's light-cone observable.

WHY NOT `<Z_j>`: it is identically zero in any Z2-symmetric state, because `Z` changes the
charge and so has no diagonal matrix element at all. That is the symmetry, not a limitation --
the order parameter has to be probed with the CORRELATOR `<Z_i Z_j>`, never a single site.
"""
x_profile(psi::SymMPS) =
    [real(site_expval(psi, j, ising_local_space().X)) for j in 1:length(psi)]
