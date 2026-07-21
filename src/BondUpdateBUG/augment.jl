# Rank-<= 2r augmented isometry with a minimal random missing-quantum-number fill.
#
# Mirrors symmetric_completion.py::_symmetric_augmented_{left,right}_isometry_from_{k,l}
# and augment.py::_pick_{left,right}_update.
#
# THE RANK RULE (this is the regression that was fixed in Alice as 1f2de1b).
# The augmented basis is the Sulz range basis of [U0 | K1] at rank <= 2r. It is
# NOT completed to the ambient local dimension dim(link)*dim(site). Alice used
# to call complete_column_basis, which padded every PARTIALLY populated sector
# up to its full local dimension -- the d*r augmented-rank blow-up. Measured
# A/B on the L=6 Heisenberg case: old max augmented rank 16 (= full local dim)
# with 0/160 bonds below full, versus 10 with 58/160 below full.
#
# complete_column_basis was doing two jobs, and only one of them is wanted:
#   (a) OPENING a sector that is completely empty  -> kept, as the minimal
#       random fill below, because without it the sector is unreachable;
#   (b) PADDING a partially populated sector to full -> dropped, because it
#       does not scale and the S-step does not need it.

"""
    random_sector_seed(F, q, n_seed; rng) -> TLArray

`n_seed` random orthonormal directions confined to charge sector `q` of the
fused `(link (x) site)` space, as a rank-3 tensor `(link, site, bond)`.

`F` is the full fusion basis from [`fusion_basis`](@ref) -- NOT raw
`getIdentity`, whose output has flipped arrows and dual labels. Restricting it
to `q` already gives an orthonormal basis of that sector; randomness only matters
when the sector has more directions than we intend to seed, so the block is
randomised and re-orthonormalised, then the first `n_seed` columns are kept.

Only the blocks the symmetry allows are ever touched. Randomising a dense
buffer instead would inject amplitudes into forbidden sectors.
"""
function random_sector_seed(F, q, n_seed::Int; rng::AbstractRNG)
    Fq = to_concrete(getsub(F, 3, s -> s == q ? Colon() : nothing))
    rdim = leg_dim(Fq, 3)
    n = min(n_seed, rdim)
    n <= 0 && return nothing

    # Rebuild with random payloads in exactly Fq's sparsity pattern.
    T = eltype(Fq)
    rand_rmts = [randn(rng, T, size(r)) for r in Fq.RMTs]
    R = TLArray(symm(Fq), copy(Fq.qlabels), copy(Fq.wmatdata), copy(Fq.wmatinfo),
                rand_rmts, Fq.inds, Fq.spaces)

    Q = to_concrete(svd(R, (1, 2); cutoff = 0.0).U)
    leg_dim(Q, 3) <= n && return Q
    return to_concrete(getsub(Q, 3, s -> 1:n))
end

# Numerical-rank guard. NOT a truncation knob and NOT user-facing: it only
# separates genuine directions from the roundoff residue of a floating-point
# subtraction, at the scale of the input. Every direction the K/L step actually
# produces is kept, however small its weight -- the only bound on the K/L
# augmentation is Sulz's 2r.
const _RANK_EPS = 100 * eps(Float64)

"""
    _complement_basis(U0, K1; leg) -> TLArray or nothing

An orthonormal basis of the part of `K1` outside `span(U0)`, exactly orthogonal
to `U0`. `leg` is 3 for a left frame, 1 for a right frame.

NO TOLERANCE. The K/L augmentation admits every direction it finds; the hard
constraint is the Sulz bound, `rank([U0 | K1]) <= 2r`, which is automatic here
because `rank(perp) <= rank(K1) = r` -- asserted below rather than assumed.

TWO SUBTLETIES, BOTH OF WHICH PRODUCED A NORM BLOW-UP. `oplus` concatenates
blocks; it does NOT orthogonalise them, so `[U0 | Q]` is an isometry only if this
function guarantees it, and it originally did not:

 1. A CUTOFF RELATIVE TO `perp` IS MEANINGLESS. `perp` is the O(tau) new
    direction, so its own largest singular value is tiny -- measured 1.5e-6 at
    tau = 0.02. Telum's `cutoff` is relative to that, so the old `cutoff = 1e-12`
    thresholded at 1.5e-18 and kept the roundoff residue of the subtraction at
    ~1e-16. That residue is what was subtracted, so it points ALONG U0: measured
    `||U0' Q|| = 1.0`, one column of Q duplicated a U0 column, and the frame had
    isometry defect exactly sqrt(2). The S-step then read `U_aug' Theta V_aug'`
    off a singular frame and the norm GREW -- 1 -> 2.83 -> 11.3 -> 45.3 -> 128.3
    over five real-time steps. The rank guard is therefore absolute, scaled by
    the input, which is what a numerical rank determination has to be.

 2. RE-PROJECTION IS STILL NEEDED. A direction surviving at the rank guard can
    still carry a U0 component of order eps/guard. Subtracting the U0 component
    again and re-orthonormalising drives the overlap back to machine precision.
"""
function _complement_basis(U0, K1; leg::Int = 3)
    perp_of = leg == 3 ? perp_component : perp_component_right
    split   = leg == 3 ? (1, 2) : (2, 3)
    scale = norm(K1)
    scale == 0 && return nothing
    tol = _RANK_EPS * scale

    function orth(t)
        n = norm(t)
        n <= tol && return nothing
        Q = to_concrete(svd(t, split; cutoff = min(tol / n, 1.0)).U)
        return leg == 3 ? Q : to_concrete(permutedims(Q, (3, 1, 2)))
    end

    Q = orth(perp_of(U0, K1))
    Q === nothing && return nothing
    Q = orth(perp_of(U0, Q))            # exactness pass; see (2) above
    Q === nothing && return nothing

    r = leg_dim(U0, leg)
    leg_dim(Q, leg) <= r || error(
        "Sulz bound violated: $(leg_dim(Q, leg)) new directions against rank $r")
    return Q
end

"""
    augmented_left_isometry(U0, K1; max_rank=typemax(Int), missing_fill=1,
                            augment=true, seed_charges=nothing, rng) -> (U_aug, n_new)

The augmented left frame `[U0 | orth(P_perp K1) | seeds]`, rank <= 2r plus one
minimal seed per empty-but-reachable sector.

`U0` and `K1` both carry `(link, site, bond)`. `U0` is retained **exactly** --
it is the first block of the result -- so the old frame is always spanned.

`max_rank` caps the NEW directions admitted per charge sector, matching
`_kl_truncated_left`'s `keep = min(keep, max_rank - r)`.

`seed_charges`, when given, restricts the fill to those charges. The caller
supplies it because the constraint is not visible from one frame: a left sector
can only ever hold amplitude if the RIGHT frame supplies a partner, and seeding
one that cannot pair produces a column that is structurally zero through the
whole S-step and is then discarded by the truncating SVD -- after inflating
`aug_k`. See `kls_step.jl::pairable_charges`.

`missing_fill = 0` disables the fill entirely. Note Alice cannot do this: its
`n_seed = min(len(rows), max(1, aug_missing_fill))` floors at 1, so
`aug_missing_fill=0` still seeds one column there. The `0` setting is a
Julia-only A/B switch for the regression guard and is **not** part of the
Python parity surface.
"""
function augmented_left_isometry(U0, K1;
                                 max_rank::Int = typemax(Int),
                                 missing_fill::Int = 1,
                                 augment::Bool = true,
                                 seed_charges = nothing,
                                 rng::AbstractRNG = MersenneTwister(0x5EED))
    bond_tag = U0.inds[3].itags
    blocks = Any[U0]
    n_new = 0

    if augment
        # (1) Sulz / discarded augmentation: the part of K1 outside span(U0),
        # orthonormal AND exactly orthogonal to U0 -- see `_complement_basis`.
        Q = _complement_basis(U0, K1; leg = 3)
        if Q !== nothing
            Q = _cap_new_columns(Q, U0, max_rank)
            if Q !== nothing && leg_dim(Q, 3) > 0
                push!(blocks, to_concrete(setitag(Q, 3, bond_tag)))
                n_new += leg_dim(Q, 3)
            end
        end

        # (2) Minimal fill for sectors that are reachable but spanned by neither.
        if missing_fill > 0
            rep = sector_report(U0, K1)
            missed = [r for r in rep if r.missing]
            if !isempty(missed)
                F = fusion_basis(U0, 1, 2; tag = bond_tag)
                for r in missed
                    seed_charges === nothing || r.charge in seed_charges || continue
                    seed = random_sector_seed(F, r.charge,
                                              min(r.reachable_dim, missing_fill); rng = rng)
                    seed === nothing && continue
                    push!(blocks, to_concrete(setitag(seed, 3, bond_tag)))
                    n_new += leg_dim(seed, 3)
                end
            end
        end
    end

    U_aug = length(blocks) == 1 ? U0 : to_concrete(oplus(blocks, (3,)))
    return U_aug, n_new
end

"""
    augmented_right_isometry(V0, L1; kwargs...) -> (V_aug, n_new)

Row mirror of [`augmented_left_isometry`](@ref). `V0` and `L1` carry
`(bond, site, link)`, so the frame leg is leg **1** and the contracted pair is
`(2, 3)`.

Written out rather than obtained by transposing, matching how Alice keeps
`_symmetric_augmented_right_isometry_from_l` explicit -- the arrow and tag
bookkeeping is not symmetric under a naive transpose.
"""
function augmented_right_isometry(V0, L1;
                                  max_rank::Int = typemax(Int),
                                  missing_fill::Int = 1,
                                   augment::Bool = true,
                                  seed_charges = nothing,
                                  rng::AbstractRNG = MersenneTwister(0x5EED))
    bond_tag = V0.inds[1].itags
    blocks = Any[V0]
    n_new = 0

    if augment
        Q = _complement_basis(V0, L1; leg = 1)
        if Q !== nothing
            Q = _cap_new_columns(Q, V0, max_rank; leg = 1)
            if Q !== nothing && leg_dim(Q, 1) > 0
                push!(blocks, to_concrete(setitag(Q, 1, bond_tag)))
                n_new += leg_dim(Q, 1)
            end
        end

        if missing_fill > 0
            rep = sector_report_right(V0, L1)
            missed = [r for r in rep if r.missing]
            if !isempty(missed)
                F = fusion_basis(V0, 2, 3; tag = bond_tag)
                for r in missed
                    seed_charges === nothing || r.charge in seed_charges || continue
                    seed = random_sector_seed(F, r.charge,
                                              min(r.reachable_dim, missing_fill); rng = rng)
                    seed === nothing && continue
                    seed = to_concrete(permutedims(seed, (3, 1, 2)))
                    push!(blocks, to_concrete(setitag(seed, 1, bond_tag)))
                    n_new += leg_dim(seed, 1)
                end
            end
        end
    end

    V_aug = length(blocks) == 1 ? V0 : to_concrete(oplus(blocks, (1,)))
    return V_aug, n_new
end

"""
Cap the new directions admitted in each charge sector at `max_rank` minus the
directions `ref` already supplies there, per `_kl_truncated_left`.
"""
function _cap_new_columns(Q, ref, max_rank::Int; leg::Int = 3)
    max_rank == typemax(Int) && return Q
    syms = symm(Q)
    qdir, rdir = Q.inds[leg].dir, ref.inds[leg].dir
    keep = Dict{Any, Int}()
    for (q, d) in Q.spaces[leg]
        have = sector_dim(ref, leg, align_charge(syms, q, qdir, rdir))
        keep[q] = clamp(max_rank - have, 0, d)
    end
    all(keep[q] == d for (q, d) in Q.spaces[leg]) && return Q
    any(v > 0 for v in values(keep)) || return nothing
    return to_concrete(getsub(Q, leg, s -> (n = get(keep, s, 0); n == 0 ? nothing : 1:n)))
end
