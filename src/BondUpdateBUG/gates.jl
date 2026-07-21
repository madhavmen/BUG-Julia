# Two-site bond gates.
#
# Mirrors bond.py::kernel_gate + bond_hamiltonian. Alice reaches its gate through
# AutoMPO; here the nearest-neighbour term is built directly from the Telum local
# space, which is the same tensor with fewer moving parts.
#
# LEG CONVENTION -- identical to Python's `einsum("LRlr,aLRb->alrb", gate, theta)`:
#
#     gate legs = (ket_l '+', ket_r '+', bra_l '-', bra_r '-')
#
# The two '+' ket legs contract theta's site legs (which are '-'), and the two
# '-' bra legs come out carrying the SAME (itag, arrow) the site legs had, so the
# result drops straight back in. Telum distinguishes the pairs by arrow alone,
# exactly as it does for the bare `q.Sz`, so all four can share the two site tags.
#
# THE 1/sqrt(2) IN Sp. Telum's raising operator is normalised so that
# `Sp (x) Sp'` already equals (1/2) S^- (x) S^+ -- MEASURED, not assumed:
# <du|Sp(x)Sp'|ud> = 0.5, and <uu|Sz(x)Sz|uu> = 0.25 alongside it. The XY term
# 1/2 (S^+S^- + S^-S^+) is therefore the plain SUM of the two adjoint pairings
# with no further factor. Writing the textbook `0.5*(...)` here would be wrong by
# a factor of two; `tests/BondUpdateBUG/test_gates.jl` pins all sixteen matrix
# elements against the analytic Heisenberg block so the convention cannot drift.

# Itag of a site leg given as a `TLIndex`, an `Itag`, or a plain string.
# A plain string has to be run through `setitag` rather than compared directly:
# `Itag` is a SORTED TAG SET, so the leg written as "S,1" reads back as "1,S"
# and a raw `Itag == String` comparison would reject the very tag that built it.
_site_tag(x::TLIndex) = x.itags
_site_tag(x::AbstractString) = to_concrete(setitag(local_space().I, 1, x)).inds[1].itags
_site_tag(x) = x

"Retag both physical legs of a local operator, leaving any operator leg alone."
_retag_site(O, tag) = to_concrete(setitag(setitag(O, 1, tag), 2, tag))

"""
    heisenberg_bond_gate(site_l, site_r; J=1.0, delta=1.0) -> TLArray

The nearest-neighbour term `J (Sx Sx + Sy Sy + delta Sz Sz)` on one bond, tagged
for the two given site legs. `delta = 0` gives the XX model, `delta = 1` isotropic
Heisenberg.

`site_l` / `site_r` may be `TLIndex` handles (e.g. `f.site_l`) or bare itags.

Built as `Sp(x)Sp' + Sm(x)Sm' + delta Sz(x)Sz`: the first two are the U(1)
charge-conserving hop in both directions, contracted over the operator leg, whose
charge (+/-2) is what forces them to pair with their own adjoint rather than with
each other.
"""
function heisenberg_bond_gate(site_l, site_r; J::Float64 = 1.0, delta::Float64 = 1.0)
    tl, tr = _site_tag(site_l), _site_tag(site_r)
    tl == tr && throw(ArgumentError(
        "the two sites of a bond need distinct itags, both are '$tl'"))
    q = local_space()

    # (ket_l, bra_l, ket_r, bra_r) -> (ket_l, ket_r, bra_l, bra_r).
    pair(A, B) = to_concrete(permutedims(
        contract(_retag_site(A, tl), (3,), _retag_site(B, tr)', (3,)), (1, 4, 2, 3)))
    outer(A, B) = to_concrete(permutedims(
        contract(_retag_site(A, tl), (), _retag_site(B, tr), ()), (1, 3, 2, 4)))

    xy = pair(q.Sp, q.Sp) + pair(q.Sm, q.Sm)
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
