# Canonical two-site bond snapshot, the input to the KLS kernel; mirrors Alice's bond_snapshot.
# Telum has no QR or LQ, so SVD stands in: the U of svd(A) is the QR's left isometry, and S*Vd its R
# factor (docs/telum_api_contract.md, section 4). These are exact, truncation-free moves, so
# `cutoff=0.0` is passed explicitly; all rank adaptation belongs to the KLS step.

"""
    BondFrame

The canonical snapshot at bond `(i, i+1)`, with `theta == U0 * S0 * V0` exactly (no truncation).
`U0` is a left isometry with legs `(link_l, site_l, bond_u)`, `V0` a right isometry with legs
`(bond_v, site_r, link_r)`, and `S0` the `(bond_u, bond_v)` centre.

`link_*` / `site_*` are `TLIndex` handles. A `TLIndex` carries no dimension, which lives in the
tensor's `spaces`, so `old_rank` and `link_mid_space` are stored separately.
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
