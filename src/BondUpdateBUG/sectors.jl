# Reachable-sector enumeration and missing-quantum-number detection. Under U(1) with the opposite
# frame frozen, `orth([U0 | K1])` cannot open a new charge sector, so empty reachable sectors are
# detected explicitly instead of padding every sector to full dimension.

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

Charges reachable on the fused leg `a (x) b`, with dimensions, by explicit charge arithmetic in
the frame convention (incoming legs dualised, sum, dualise once more). Abelian only. A vacuum
link cannot tell the sign convention apart, so test on charged links.
"""
function fuse_spaces(syms, space_a, dir_a::Char, space_b, dir_b::Char)
    acc = Dict{Any, Int}()
    order = Any[]
    for (qa, da) in space_a, (qb, db) in space_b
        ca = dir_a == '+' ? dual_charge(syms, qa) : qa
        cb = dir_b == '+' ? dual_charge(syms, qb) : qb
        q = dual_charge(syms, add_charge(syms, ca, cb))
        haskey(acc, q) || push!(order, q)
        acc[q] = get(acc, q, 0) + da * db
    end
    return [(q, acc[q]) for q in order]
end

"""
Memo for [`fusion_basis`](@ref) and its hit/miss counters. The result depends only on the legs'
spaces, arrows, tags and prime levels, so it is memoised; the key carries itags and arrows, a copy
is returned, and the lock (held only around the Dict) makes it safe under the parallel half-sweeps.
"""
const _FUSION_CACHE = Dict{Any, Any}()
const _FUSION_LOCK = ReentrantLock()
const _FUSION_CALLS = Ref(0)
const _FUSION_MISSES = Ref(0)
const _FUSION_ELEMS = Ref(0)

"""
Element budget of the memo. The key holds the legs' exact spaces, which an adaptive run changes
nearly every step, so an unbounded memo grows with the run. An insert that would pass the budget
empties the memo first; within a step the same legs recur, so the reuse the memo exists for
survives. `nothing` = unbounded (the memo before 2026-10-27). 2^28 elements (2-4 GB) holds one
step's legs at D ~ 1024, d = 4 (~200 keys of up to ~6e5 elements), so a step does not thrash it.
"""
const _FUSION_BUDGET = Ref{Union{Nothing, Int}}(2^28)

"Set the memo's element budget (`nothing` = unbounded); returns the previous one."
function set_fusion_budget!(n::Union{Nothing, Integer})
    n === nothing || n > 0 || throw(ArgumentError("fusion budget must be positive, got $n"))
    old = _FUSION_BUDGET[]
    _FUSION_BUDGET[] = n === nothing ? nothing : Int(n)
    return old
end

"Calls and misses since the last `reset_fusion_stats!`; distinct keys and stored elements now."
fusion_stats() = (calls = _FUSION_CALLS[], misses = _FUSION_MISSES[],
                  keys = length(_FUSION_CACHE), elems = _FUSION_ELEMS[])

"Zero the counters (NOT the cache) so a step can be attributed on its own."
function reset_fusion_stats!()
    _FUSION_CALLS[] = 0
    _FUSION_MISSES[] = 0
    return nothing
end

"Drop the memo entirely -- for tests that switch symmetry mode under one process."
function clear_fusion_cache!()
    empty!(_FUSION_CACHE)
    _FUSION_ELEMS[] = 0
    reset_fusion_stats!()
    return nothing
end

_leg_key(t, l::Int) = (t.spaces[l], t.inds[l].dir, string(t.inds[l].itags),
                       t.inds[l].plev, t.inds[l].lock)

"""
    fusion_basis(t, leg_a=1, leg_b=2; tag="fused") -> TLArray

Orthonormal basis of the fused `leg_a (x) leg_b` space in `t`'s own leg convention: legs
`(leg_a, leg_b, bond)` with `t`'s arrows and the bond labelled as `svd(t, (leg_a, leg_b)).U`
labels it. `getIdentity` alone has flipped arrows and dual labels. Memoised (`_FUSION_CACHE`).
"""
function fusion_basis(t, leg_a::Int = 1, leg_b::Int = 2; tag::AbstractString = "fused")
    key = (_leg_key(t, leg_a), _leg_key(t, leg_b), String(tag), symmetry_mode())
    hit = lock(_FUSION_LOCK) do
        _FUSION_CALLS[] += 1
        get(_FUSION_CACHE, key, nothing)
    end
    hit === nothing || return to_concrete(hit)
    lock(_FUSION_LOCK) do
        _FUSION_MISSES[] += 1
    end
    F = to_concrete(getIdentity((t, leg_a), (t, leg_b); itag = tag)')
    # getIdentity's output is always (input_a, input_b, fused), whatever
    # leg numbers the inputs had, so the split is on (1,2) -- NOT (leg_a, leg_b).
    U = to_concrete(svd(F, (1, 2); cutoff = 0.0).U)
    # The SVD relabels the bond to Telum's default, so `tag` is reapplied; the CBE probes need their
    # distinct tags to avoid colliding with a frame's `svdL` bond leg.
    out = to_concrete(setitag(U, 3, tag))
    # Store, then hand out a copy -- see the memo's note on caller ownership.
    n = sum(length(r) for r in out.RMTs; init = 0)
    lock(_FUSION_LOCK) do
        cap = _FUSION_BUDGET[]
        if cap !== nothing && _FUSION_ELEMS[] + n > cap
            empty!(_FUSION_CACHE)
            _FUSION_ELEMS[] = 0
        end
        haskey(_FUSION_CACHE, key) || (_FUSION_ELEMS[] += n)   # a concurrent miss stored it already
        _FUSION_CACHE[key] = out
    end
    return to_concrete(out)
end

"""
    reachable_sectors(t, leg_a=1, leg_b=2) -> Vector{Tuple{QLabel,Int}}

Every charge reachable on the fused `leg_a (x) leg_b` of `t`, with its
dimension, labelled in `t`'s own bond convention (see [`fusion_basis`](@ref)).

The total dimension always equals `leg_dim(t, leg_a) * leg_dim(t, leg_b)` --
fusing regroups the product space by charge, it never loses or adds directions.
"""
function reachable_sectors(t, leg_a::Int = 1, leg_b::Int = 2)
    return [(q, d) for (q, d) in fusion_basis(t, leg_a, leg_b).spaces[end]]
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

`P_perp K1 = K1 - U0 (U0' K1)`, contracting over the `(link, site)` legs, so the two bond legs
(different tags and arrows) never need to match.
"""
function perp_component(U0, K1)
    ov = contract(U0', (1, 2), K1, (1, 2))     # (U0 bond, K1 bond)
    proj = contract(U0, (3,), ov, (1,))        # (link, site, K1 bond)
    # The copying form, not `to_concrete!`: a Telum sum reuses `K1`'s block for any sector
    # `proj` lacks, and the in-place cleanup would rescale it. The result is then a fresh tensor
    # the caller owns -- wrapping it in another `to_concrete` only copies it again.
    return to_concrete(K1 - proj)
end

"""
    sector_report(U0, K1) -> Vector{SectorReport}

The missing-quantum-number table for the bond whose left frame is `U0`: every charge on the fused
`(link (x) site)` leg or either bond leg, with `range_cols = u0_cols + rank(P_perp K1)`.
"""
function sector_report(U0, K1)
    syms = symm(U0)
    ref_dir = U0.inds[3].dir                 # every charge below is in THIS convention
    k_dir = K1.inds[3].dir
    reach = reachable_sectors(U0, 1, 2)
    perp = perp_component(U0, K1)
    # The range must use the same complement basis the augmentation uses, or a sector could be called
    # empty while the complement fills it.
    Q = _complement_basis(U0, K1; leg = 3)
    new_space = Q === nothing ? Tuple[] : Q.spaces[3]

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

# ── right-frame mirrors ─────────────────────────────────────────────────────
# V0 and L1 carry (bond, site, link): the frame leg is 1 and the fused pair is
# (2, 3). Written out rather than obtained by transposing, matching how Alice
# keeps _symmetric_augmented_right_isometry_from_l explicit -- the arrow and tag
# bookkeeping does not survive a naive transpose.

"""
    perp_component_right(V0, L1) -> TLArray

`P_perp L1 = L1 - (L1 V0') V0`, contracting over the `(site, link)` legs.
Row mirror of [`perp_component`](@ref).
"""
function perp_component_right(V0, L1)
    ov = contract(L1, (2, 3), V0', (2, 3))     # (L1 bond, V0 bond)
    proj = contract(ov, (2,), V0, (1,))        # (L1 bond, site, link)
    return to_concrete(L1 - proj)
end

"""
    sector_report_right(V0, L1) -> Vector{SectorReport}

Missing-quantum-number table for a right frame. Row mirror of
[`sector_report`](@ref): charges are enumerated on the fused `(site (x) link)`
leg and counted on frame leg 1.
"""
function sector_report_right(V0, L1)
    syms = symm(V0)
    ref_dir = V0.inds[1].dir
    l_dir = L1.inds[1].dir
    reach = [(q, d) for (q, d) in fusion_basis(V0, 2, 3).spaces[end]]
    perp = perp_component_right(V0, L1)
    Q = _complement_basis(V0, L1; leg = 1)          # see sector_report
    new_space = Q === nothing ? Tuple[] : Q.spaces[1]

    charges = Any[]
    for (q, _) in reach
        q in charges || push!(charges, q)
    end
    for (q, _) in V0.spaces[1]
        q in charges || push!(charges, q)
    end
    for (q, _) in L1.spaces[1]
        qa = align_charge(syms, q, l_dir, ref_dir)
        qa in charges || push!(charges, qa)
    end

    out = SectorReport[]
    for q in charges
        rdim = 0
        for (s, d) in reach
            s == q && (rdim = d)
        end
        v = sector_dim(V0, 1, q)
        l = sector_dim(L1, 1, align_charge(syms, q, ref_dir, l_dir))
        nnew = 0
        for (s, d) in new_space
            s == q && (nnew = d)
        end
        rng = v + nnew
        push!(out, SectorReport(q, rdim, v, l, rng, rng == 0 && rdim > 0))
    end
    return out
end
