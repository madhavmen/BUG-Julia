# Reachable-sector enumeration and missing-quantum-number detection.
#
# This is the heart of the fix committed to Alice as 1f2de1b. Ports the sector
# bookkeeping of _kernel/kls/frame.py::_left_row_indices_by_flux and the
# missing-sector branch of _kernel/kls/symmetric_completion.py.
#
# WHY A SECTOR CAN BE MISSING. Under U(1) with the opposite frame frozen,
# charge conservation keeps K1 inside U0's charge sector, so orth([U0 | K1])
# can never OPEN a sector the way the dense Sulz BUG's K1 does. A charge that
# is locally reachable on (link (x) site) but populated by neither U0 nor K1 is
# therefore invisible to the augmentation, and the evolution can never rotate
# weight into it. Alice used to fix this with `complete_column_basis`, which
# filled every partially-populated sector to its FULL local dimension -- the
# d*r augmented-rank blow-up. Detecting the genuinely EMPTY sectors is what
# lets the fill stay minimal (Task 9) instead of padding everything.

# ── charge arithmetic ───────────────────────────────────────────────────────

"Per-symmetry dual (charge negation) of a full sector label."
dual_charge(syms, q) = ntuple(n -> get_dualq(syms[n], q[n]), length(syms))

"""
    add_charge(syms, qa, qb)

Per-symmetry fusion of two sector labels. **Abelian only** -- for a
non-Abelian symmetry the fusion of two irreps is a *set*, not a single label,
so use [`reachable_sectors`](@ref), which delegates to Telum's own
`getIdentity` and handles both cases.
"""
function add_charge(syms, qa, qb)
    return ntuple(length(syms)) do n
        s = syms[n]
        isabelian(s) || throw(ArgumentError(
            "add_charge is abelian-only; use reachable_sectors for $(s)"))
        (add_qn(s, qa[n][1], qb[n][1]),)
    end
end

"""
    fuse_spaces(syms, space_a, dir_a, space_b, dir_b) -> Vector{Tuple{QLabel,Int}}

Charges reachable on the fused leg `a (x) b`, with their dimensions, computed
by explicit charge arithmetic.

Telum's convention (measured, job 93413): each **input** leg that is incoming
(`'+'`) is dualised so the fusion rule sees every leg as outgoing, and the
fused label is then simply the SUM -- there is no further dual. Note
`getIdentity` flips the input arrows in its output, so the dirs to pass here
are the ones on the tensor being fused, not the ones on the result.

Getting this wrong is easy to miss: with a vacuum link the reachable set is
`{+q, -q}`, which is its own dual, so an extra dual still "passes". It only
shows up once the link carries a non-zero charge. Abelian only; cross-checked
against [`reachable_sectors`](@ref) on charged links in the tests.
"""
function fuse_spaces(syms, space_a, dir_a::Char, space_b, dir_b::Char)
    acc = Dict{Any, Int}()
    order = Any[]
    for (qa, da) in space_a, (qb, db) in space_b
        ca = dir_a == '+' ? dual_charge(syms, qa) : qa
        cb = dir_b == '+' ? dual_charge(syms, qb) : qb
        q = add_charge(syms, ca, cb)
        haskey(acc, q) || push!(order, q)
        acc[q] = get(acc, q, 0) + da * db
    end
    return [(q, acc[q]) for q in order]
end

"""
    reachable_sectors(t, leg_a=1, leg_b=2) -> Vector{Tuple{QLabel,Int}}

Every charge reachable on the fused `leg_a (x) leg_b` of `t`, with its
dimension. Delegates to Telum's `getIdentity`, so it is correct for
non-Abelian symmetries too.

The total dimension always equals `leg_dim(t, leg_a) * leg_dim(t, leg_b)` --
fusing regroups the product space by charge, it never loses or adds directions.
"""
function reachable_sectors(t, leg_a::Int = 1, leg_b::Int = 2)
    fused = to_concrete(getIdentity((t, leg_a), (t, leg_b); itag = "fused"))
    return [(q, d) for (q, d) in fused.spaces[end]]
end

# ── the missing-quantum-number table ────────────────────────────────────────

"""
    SectorReport

One row of the missing-quantum-number table.

- `charge`         the sector label on the fused `(link (x) site)` leg
- `reachable_dim`  how many directions that charge has there
- `u0_cols`        directions supplied by `U0`
- `k1_cols`        directions supplied by `K1`
- `range_cols`     rank of the combined span, `u0_cols + rank(P_perp K1)`
- `missing`        reachable but spanned by neither: `range_cols == 0 < reachable_dim`
"""
struct SectorReport
    charge::Any
    reachable_dim::Int
    u0_cols::Int
    k1_cols::Int
    range_cols::Int
    missing::Bool
end

"Dimension of sector `q` on leg `l` of `t`, or 0 if the sector is absent."
function sector_dim(t, l::Int, q)
    for (s, d) in t.spaces[l]
        s == q && return d
    end
    return 0
end

"""
    align_charge(syms, q, from_dir, to_dir)

Re-express sector label `q`, read off a leg with arrow `from_dir`, in the
convention of a leg with arrow `to_dir`.

A leg's charge LABEL depends on its arrow: the same physical sector is `q` on
an outgoing leg and `dual(q)` on an incoming one. `U0`'s bond leg comes from
`svd`'s `U` and is `'-'`, while `K1`'s comes from `S0`'s right leg and is `'+'`
(measured, job 93412), so comparing their `spaces` without this conversion
silently mismatches every non-self-dual sector.
"""
align_charge(syms, q, from_dir::Char, to_dir::Char) =
    from_dir == to_dir ? q : dual_charge(syms, q)

"""
    perp_component(U0, K1) -> TLArray

`P_perp K1 = K1 - U0 (U0' K1)`, the part of `K1` outside `U0`'s range,
contracting over the `(link, site)` legs.

This is the discarded projector of the variant this plan implements. Note it is
applied BEFORE the exponential in the K/L generators, which is what makes them
non-Hermitian and forces the Arnoldi path in `expv`.

It also sidesteps a leg-compatibility problem: `U0`'s bond leg and `K1`'s bond
leg generally differ in tag AND arrow (`U0`'s comes from `svd`'s `U`, `K1`'s
from `S0`'s right leg), so they cannot simply be `oplus`ed to measure the
combined rank. Contracting over legs 1 and 2 never touches either bond leg.
"""
function perp_component(U0, K1)
    ov = contract(U0', (1, 2), K1, (1, 2))     # (U0 bond, K1 bond)
    proj = contract(U0, (3,), ov, (1,))        # (link, site, K1 bond)
    return to_concrete(K1 - proj)
end

"""
    sector_report(U0, K1; aug_tol=1e-12) -> Vector{SectorReport}

The missing-quantum-number table for the bond whose left frame is `U0`.

Both tensors carry `(link, site, bond)`. Walks the union of the charges
appearing on the fused `(link (x) site)` leg and on either bond leg, so a
sector reachable but empty is reported rather than silently dropped -- that
omission is exactly the bug this table exists to catch.

`range_cols = u0_cols + rank(P_perp K1)` in that sector, with the rank taken
from a symmetry-native SVD at `aug_tol`. Nothing is densified: densifying a
block-sparse state reintroduces the forbidden amplitudes the symmetry forbids.
"""
function sector_report(U0, K1; aug_tol::Float64 = 1e-12)
    syms = symm(U0)
    ref_dir = U0.inds[3].dir                 # every charge below is in THIS convention
    k_dir = K1.inds[3].dir
    reach = reachable_sectors(U0, 1, 2)
    perp = perp_component(U0, K1)
    new_space = norm(perp) <= aug_tol ? Tuple[] :
                to_concrete(svd(perp, (1, 2); cutoff = aug_tol).U).spaces[3]

    charges = Any[]
    for (q, _) in reach
        q in charges || push!(charges, q)
    end
    for (q, _) in U0.spaces[3]
        q in charges || push!(charges, q)
    end
    for (q, _) in K1.spaces[3]
        qa = align_charge(syms, q, k_dir, ref_dir)
        qa in charges || push!(charges, qa)
    end

    out = SectorReport[]
    for q in charges
        rdim = 0
        for (s, d) in reach
            s == q && (rdim = d)
        end
        u = sector_dim(U0, 3, q)
        k = sector_dim(K1, 3, align_charge(syms, q, ref_dir, k_dir))
        nnew = 0
        for (s, d) in new_space
            s == q && (nnew = d)
        end
        rng = u + nnew
        push!(out, SectorReport(q, rdim, u, k, rng, rng == 0 && rdim > 0))
    end
    return out
end

"Charges that are reachable on `(link (x) site)` but spanned by neither frame."
missing_charges(rep::Vector{SectorReport}) = [r.charge for r in rep if r.missing]
