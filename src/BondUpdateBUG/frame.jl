# Canonical two-site bond snapshot -- the input the KLS kernel consumes.
#
# Mirrors Alice's two_site_bug/scheme.py::bond_snapshot:
#
#     U0_tens, s_left,  canon_u0 = qr(left,  [link_l, site_l], tag=link_mid.itag)
#     s_right, V0_tens, canon_v0 = lq(right, [site_r, link_r], tag=link_mid.itag)
#     S0_tens = tcontract(s_left, s_right)
#
# Telum has no QR, so SVD stands in (docs/telum_api_contract.md section 4): the
# `U` of `svd(A, left_legs)` is exactly the left isometry a QR would give, and
# `S*Vd` is the R factor. Likewise `Vd` of `svd(B, (1,))` is the right isometry
# an LQ gives, with `U*S` as the L factor. These are exact, truncation-free
# moves -- all rank adaptation belongs to the KLS step and only to it, which is
# why `cutoff=0.0` is passed rather than relying on the 1e-12 default.

"""
    BondFrame

The canonical snapshot at bond `(i, i+1)`:

    theta  ==  U0 * S0 * V0        (exactly, no truncation)

`U0` is a left isometry with legs `(link_l, site_l, bond_u)`, `V0` a right
isometry with legs `(bond_v, site_r, link_r)`, and `S0` the `(bond_u, bond_v)`
centre.

The `link_*` / `site_*` fields are `TLIndex` handles for identifying legs.
**A `TLIndex` carries no dimension** -- that lives in the owning tensor's
`spaces` -- so `old_rank` and `link_mid_space` are stored separately.
"""
struct BondFrame
    U0::Any
    S0::Any
    V0::Any
    link_l::TLIndex
    site_l::TLIndex
    link_mid::TLIndex
    site_r::TLIndex
    link_r::TLIndex
    old_rank::Int
    link_mid_space::Vector
end

"""
    bond_frame(psi, i) -> BondFrame

Snapshot bond `(i, i+1)`. Requires `psi.center == i`.

The two SVDs are given **distinct** tag pairs. With Telum's default tags both
`S` factors would emit a `'+'` leg tagged `"svdL"`, and contracting them would
produce a tensor carrying two identical `TLIndex`es -- Telum rejects that with
"Duplicate TLIndex with non-empty itag".
"""
function bond_frame(psi::SymMPS, i::Int)
    1 <= i < length(psi) || throw(BoundsError(psi, i))
    psi.center == i || throw(ArgumentError(
        "bond_frame requires the orthogonality centre on site $i, got $(psi.center)"))

    left  = psi[i]      # (link_l, site_l, link_mid)
    right = psi[i + 1]  # (link_mid, site_r, link_r)

    res_l = svd(left,  (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    res_r = svd(right, (1,),   "bV,L", "bV,R"; cutoff = 0.0)

    U0 = to_concrete(res_l.U)    # (link_l, site_l, bU,L)   -- left isometry
    V0 = to_concrete(res_r.Vd)   # (bV,R, site_r, link_r)   -- right isometry

    s_left  = res_l.S * res_l.Vd     # (bU,L, link_mid)
    s_right = res_r.U * res_r.S      # (link_mid, bV,R)
    S0 = to_concrete(s_left * s_right)   # (bU,L, bV,R)

    return BondFrame(
        U0, S0, V0,
        left.inds[1], left.inds[2], left.inds[3], right.inds[2], right.inds[3],
        leg_dim(U0, 3),
        copy(left.spaces[3]),
    )
end

"""
    frame_theta(f) -> TLArray

Reassemble the two-site block `U0 * S0 * V0`, legs
`(link_l, site_l, site_r, link_r)`.
"""
frame_theta(f::BondFrame) = to_concrete((f.U0 * f.S0) * f.V0)

"""
    two_site_block(psi, i) -> TLArray

The uncontracted two-site block `psi[i] * psi[i+1]`, same leg order as
`frame_theta`. Used to check the snapshot is exact.
"""
two_site_block(psi::SymMPS, i::Int) = to_concrete(psi[i] * psi[i + 1])
