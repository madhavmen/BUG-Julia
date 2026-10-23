# Two-site bond gates, mirroring bond.py::kernel_gate. Legs are (ket_l '+', ket_r '+', bra_l '-',
# bra_r '-'), as in Python's einsum, so the result drops back into the site legs. Telum separates
# legs by arrow alone. Telum's Sp is normalised so that Sp (x) Sp' = (1/2) S^- (x) S^+: the XY term
# is a plain sum, with no 0.5 factor.

# Itag of a site leg given as a `TLIndex`, an `Itag`, or a plain string.
# A plain string has to be run through `setitag` rather than compared directly:
# `Itag` is a SORTED TAG SET, so the leg written as "S,1" reads back as "1,S"
# and a raw `Itag == String` comparison would reject the very tag that built it.
_site_tag(x::TLIndex) = x.itags
_site_tag(x::AbstractString) = to_concrete(setitag(_tagging_identity(), 1, x)).inds[1].itags
_site_tag(x) = x

# ANY identity with a site leg will do here -- it is used only to normalise a string into an
# `Itag`, never for its values. It must follow the SYMMETRY MODE, though: under `:Z2` the local
# space is the Ising one (`I, X, Z` in the sigma^x basis) and `local_space()` deliberately
# refuses, since there is no spin-1/2 Heisenberg space under Z2. Hard-coding `local_space()`
# here made every Z2 gate application fail inside `apply_gate`.
_tagging_identity() = symmetry_mode() === :Z2 ? ising_local_space().I : local_space().I

"Retag both physical legs of a local operator, leaving any operator leg alone."
_retag_site(O, tag) = to_concrete(setitag(setitag(O, 1, tag), 2, tag))

"""
    heisenberg_bond_gate(site_l, site_r; J=1.0, delta=1.0) -> TLArray

The nearest-neighbour term `J (Sx Sx + Sy Sy + delta Sz Sz)` on one bond, tagged for the two site
legs (`TLIndex` handles or bare itags). `delta = 0` is the XX model, `delta = 1` isotropic
Heisenberg.

Built as `Sp(x)Sp' + Sm(x)Sm' + delta Sz(x)Sz`: the two hops conserve charge in both directions, and
the op-leg charge (+/-2) forces each to pair with its own adjoint.
"""
function heisenberg_bond_gate(site_l, site_r; J::Float64 = 1.0, delta::Float64 = 1.0)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    tl == tr && throw(ArgumentError(
        "the two sites of a bond need distinct itags, both are '$tl'"))
    q = local_space()

    # (ket_l, bra_l, ket_r, bra_r) -> (ket_l, ket_r, bra_l, bra_r).
    pair(A, B) = to_concrete(permutedims(
        contract(_retag_site(A, tl), (3,), _retag_site(B, tr)', (3,)), (1, 4, 2, 3)))

    # ── :SU2 ──
    # One term, not three: under SU(2) only `:S` (a rank-3 op-leg, S = 1 irrep) is exposed, and its
    # op-leg contraction sums the vector index, so one `pair` is the full `S_i . S_j`. `delta != 1`
    # is refused, not built isotropically: anisotropy breaks SU(2) to U(1), so the gate would be the
    # wrong Hamiltonian.
    if symmetry_mode() === :SU2
        delta == 1.0 || throw(ArgumentError(
            "SU(2) admits only the isotropic point, got delta = $delta; anisotropy breaks " *
            "the symmetry down to U(1), so run that case under set_symmetry!(:U1)"))
        return to_concrete(J * pair(q.S, q.S) * (1.0 + 0.0im))
    end
    outer(A, B) = to_concrete(permutedims(
        contract(_retag_site(A, tl), (), _retag_site(B, tr), ()), (1, 3, 2, 4)))

    # Under :U1 the raising/lowering ops are rank 3 and `pair` already folds in the 1/2 (see the
    # header). Under :none they are plain matrices: Telum's Sp = -(1/√2) S⁺ and Sm = (1/√2) S⁻ make
    # the XY term an outer product with coefficient -1. The gate test pins this element for element
    # against :U1.
    xy = if length(q.Sp.inds) == 3
        pair(q.Sp, q.Sp) + pair(q.Sm, q.Sm)
    else
        to_concrete(-1.0 * (outer(q.Sp, q.Sm) + outer(q.Sm, q.Sp)))
    end
    G = delta == 0.0 ? xy : xy + delta * outer(q.Sz, q.Sz)
    return to_concrete(J * G * (1.0 + 0.0im))
end

"""
    xx_bond_gate(site_l, site_r; J=1.0) -> TLArray

`J (Sx Sx + Sy Sy)`, the free-fermion XX term. `heisenberg_bond_gate` with
`delta = 0`.
"""
xx_bond_gate(site_l, site_r; J::Float64 = 1.0) =
    heisenberg_bond_gate(site_l, site_r; J = J, delta = 0.0)

"""
    magnetisation_gate(site_l, site_r) -> TLArray

`Sz (x) I + I (x) Sz`, the two-site magnetisation, in the same leg convention as
[`heisenberg_bond_gate`](@ref). Not part of the dynamics -- it is the observable
that pins U(1) charge conservation across a bond update.
"""
function magnetisation_gate(site_l, site_r)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    q = local_space()
    outer(A, B) = to_concrete(permutedims(
        contract(_retag_site(A, tl), (), _retag_site(B, tr), ()), (1, 3, 2, 4)))
    return to_concrete((outer(q.Sz, q.I) + outer(q.I, q.Sz)) * (1.0 + 0.0im))
end

"""
    apply_gate(gate, theta, site_l, site_r) -> TLArray

Act `gate` on the two-site block `theta`, legs `(link_l, site_l, site_r, link_r)`
in and out. Python's `_apply_gate_named`; the retag Python needs afterwards is
unnecessary here because the gate's bra legs already carry the site tags.

`site_l` / `site_r` are checked against `theta`, not used to build anything: a
gate tagged for the wrong bond otherwise contracts happily against the wrong legs
of a symmetric tensor and returns a plausible, silently wrong answer.
"""
function apply_gate(gate, theta, site_l, site_r)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    (gate.inds[1].itags == tl && gate.inds[2].itags == tr) || throw(ArgumentError(
        "gate is tagged for ($(gate.inds[1].itags), $(gate.inds[2].itags)), " *
        "asked to act on ($tl, $tr)"))
    (theta.inds[2].itags == tl && theta.inds[3].itags == tr) || throw(ArgumentError(
        "gate is tagged for ($tl, $tr) but theta carries " *
        "($(theta.inds[2].itags), $(theta.inds[3].itags))"))
    out = contract(gate, (1, 2), theta, (2, 3))   # (bra_l, bra_r, link_l, link_r)
    return to_concrete(permutedims(out, (3, 1, 2, 4)))
end
