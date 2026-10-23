
"""
    _unprime(t, leg) -> TLArray

Drop one prime level from `leg`. An environment's bra leg is primed to keep it distinct
from the ket while both are open (`henv.jl`); once it has replaced a link leg it must
carry that link's original prime level again.
"""
#
# `to_concrete!`, not `to_concrete`: `prime` already returns a deep copy the caller owns, so a second
# copy is pure waste.
_unprime(t, leg::Int) = to_concrete!(prime(t, leg; inc = -1))

"""
    _env_on_link(T, E, leg) -> TLArray

Contract environment `E` onto `leg` of `T`, leaving the bra leg in `leg`'s place,
unprimed. If both carry an op-leg -- `T`'s last, `E`'s third -- they are contracted too
(the channel closes).

`T` is `K0`-shaped `(link_l, site_l, bond [, op])` with `leg = 1`, or `V0`-shaped
`(bond, site_r, link_r [, op])` with `leg = 3`. In both cases the contracted link leg is
at an END of the tensor, so the bra comes back out in exactly the position it vacated
and no permutation is needed beyond the two written below.
"""
function _env_on_link(T, E, leg::Int)
    nT, nE = length(T.inds), length(E.inds)
    out = if nE == 3 && nT == 4
        contract(T, (leg, nT), E, (2, 3))
    elseif nE == 2 && nT == 3
        contract(T, (leg,), E, (2,))
    else
        error("_env_on_link: rank-$nT tensor against a rank-$nE environment leaves an " *
              "operator channel dangling")
    end
    # `out` = T's remaining legs in order, then E's bra leg.
    #   leg == 1 (K0): (site_l, bond, bra) -> bra first
    #   leg == 3 (V0): (bond, site_r, bra) -> already in place
    if leg == 1
        return _unprime(to_concrete(permutedims(out, (3, 1, 2))), 1)
    elseif leg == 3
        return _unprime(to_concrete(out), 3)
    end
    throw(ArgumentError("_env_on_link: leg must be 1 or 3, got $leg"))
end

# ── project-first sketches: `Y = (P_perp (H Theta)) Om'` ──
# `P_perp` and `Om` act on disjoint legs, so the order is exact; `_preselect_*` projects again
# (idempotent) because it also orthonormalises and returns the residual.

# `H*Theta` is built once per `cbe_expand` and shared by its three sketch calls; the arithmetic is
# the same as building it per call.

"""
    _sketch_closures(f, buildHT) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, sharing one lazily built `H*Theta` and one
projection per side. Lazy because `cbe_expand` often returns before sketching (saturated frames).
"""
function _sketch_closures(f::BondFrame, buildHT; share::Bool = true)
    HT = Ref{Any}(nothing)
    AL = Ref{Any}(nothing)          # P_perp^L (H Theta) -- wanted by BOTH `skl` calls
    AR = Ref{Any}(nothing)          # (H Theta) P_perp^R
    # ⚠ `share = false` IS THE OLD BEHAVIOUR, KEPT AS AN A/B AND NOTHING ELSE. A performance claim
    # that can only be reproduced by checking out an old commit is not reproducible in practice;
    # this makes "how much did sharing buy?" a switch that `benchmarks/rsvd_parallel.jl` can flip
    # inside ONE process -- which matters here, because a cold Julia process spends ~10 minutes
    # compiling before it can time anything.
    _ht() = share ? (HT[] === nothing && (HT[] = buildHT()); HT[]) : buildHT()
    function _al()
        share && AL[] !== nothing && return AL[]
        H = _ht()
        c = contract(f.U0', (1, 2), H, (1, 2))             # (bond, site_r, link_r)
        A = to_concrete(H - to_concrete!(contract(f.U0, (3,), c, (1,))))
        share && (AL[] = A)
        return A
    end
    function _ar()
        share && AR[] !== nothing && return AR[]
        H = _ht()
        c = contract(H, (3, 4), f.V0', (2, 3))             # (link_l, site_l, bond)
        A = to_concrete(H - to_concrete!(contract(c, (3,), f.V0, (1,))))
        share && (AR[] = A)
        return A
    end
    skl = Om -> to_concrete!(contract(_al(), (3, 4), Om', (2, 3)))       # (link_l, site_l, g)
    skr = Om -> to_concrete(permutedims(contract(_ar(), (1, 2), Om', (1, 2)), (3, 1, 2)))
    return skl, skr
end

"""
    sketch_h_left(f, h, i, lenv, renv, Om) -> TLArray

`Y = (P_perp (H Theta)) Om†`, legs `(link_l, site_l, g)`: the discarded-space component of
`H Theta` sketched from the right, with `Om` in `V0`'s layout. A standalone call rebuilds
`H*Theta`; sweeps use the cached closures.
"""
sketch_h_left(f::BondFrame, h::XXZChain, i::Int,
              lch::ChannelSet, rch::ChannelSet, Om) =
    first(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), h, i, lch, rch)))(Om)

"""
    sketch_h_right(f, h, i, lenv, renv, Om) -> TLArray

Mirror of [`sketch_h_left`](@ref): the right matricization of `(H Theta) P_perp^R` with
`P_perp^R = I - V0' V0`, sketched from the left. `Om` carries `U0`'s layout
`(link_l, site_l, g)` and the result is `(g, site_r, link_r)`.
"""
sketch_h_right(f::BondFrame, h::XXZChain, i::Int,
               lch::ChannelSet, rch::ChannelSet, Om) =
    last(_sketch_closures(f, () -> apply_h_two_site(frame_theta(f), h, i, lch, rch)))(Om)

# ── the sector-graded Gaussian sketch (`GaussRandMat`, `RSVDpreBE0SiQS.m:557`) ──
# `CompRatio·Dpre` columns in the frame's complement (per sector, by complement dimension) and the
# rest inside the frame, so empty-but-reachable sectors are sampled through `H` by construction.

"""
    _randomize_like(F, rng) -> TLArray

`F`'s sparsity pattern with Gaussian payloads. Only the blocks the symmetry allows are
ever touched.
"""
function _randomize_like(F, rng::AbstractRNG)
    T = eltype(F) <: Complex ? eltype(F) : ComplexF64
    return TLArray(symm(F), copy(F.qlabels), copy(F.wmatdata), copy(F.wmatinfo),
                   [randn(rng, T, size(r)) for r in F.RMTs], F.inds, F.spaces)
end

"""
    _alloc_columns(dims, budget) -> Dict

Split `budget` columns across sectors in proportion to `dims`, capped per sector at its
own dimension. `ceil` per sector as in `GaussRandMat`, so a small budget still reaches
every sector that has room rather than rounding the small ones to zero -- which is the
whole point when the sector we most need is the empty one.
"""
function _alloc_columns(dims::Dict, budget::Int)
    tot = sum(values(dims); init = 0)
    (tot == 0 || budget <= 0) && return Dict(q => 0 for q in keys(dims))
    return Dict(q => min(d, ceil(Int, d / tot * budget)) for (q, d) in dims)
end

"""
    _boost!(cols, boost, cap) -> Dict

Add `boost[q]` columns to `cols[q]`, capped at `cap[q]`, and return what did not fit.

A sector absent from `cap` is not reachable from this side, so it is skipped rather than
created: Test B targets a sector by its residual, and a residual can never name a charge the
fusion forbids, but the guard keeps that true by construction instead of by argument.
"""
function _boost!(cols::Dict, boost, cap::Dict)
    left = Dict{Any, Int}()
    for (q, b) in boost
        haskey(cap, q) || continue
        have = get(cols, q, 0)
        add = clamp(b, 0, max(cap[q] - have, 0))
        cols[q] = have + add
        b > add && (left[q] = b - add)
    end
    return left
end

"Keep `counts[q]` columns of sector `q` on `leg`, dropping sectors allotted none."
function _trim_per_sector(Q, leg::Int, counts::Dict)
    keep = s -> (n = get(counts, s, 0); n <= 0 ? nothing : 1:n)
    any(v > 0 for v in values(counts)) || return nothing
    return to_concrete(getsub(Q, leg, keep))
end

"""
    full_local_basis(frame, side; tag="cbeF") -> TLArray

The complete local basis for one side of a bond, in the frame's layout, with `g` the full fused
dimension. Fed to the same sketch closures it turns RSVD-CBE into plain CBE (the A/B reference).
"""
function full_local_basis(frame, side::Symbol; tag::AbstractString = "cbeF")
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    a, b = side === :left ? (1, 2) : (2, 3)
    F = fusion_basis(frame, a, b; tag = tag)
    # `fusion_basis` returns (a, b, fused); a right frame carries its bond leg first.
    return side === :left ? F : to_concrete(permutedims(F, (3, 1, 2)))
end

"""
    _frame_sector_dims(frame, side, F, where) -> (dfull, dT)

Per charge sector of `F`'s fused leg: the local dimension and the frame's width in it. Throws if
the frame is wider than its local space or holds a sector the space lacks. Labels are aligned to
the fused leg's arrow (`align_charge`).
"""
function _frame_sector_dims(frame, side::Symbol, F, where::AbstractString)
    fl = side === :left ? 3 : 1
    dfull = Dict(q => d for (q, d) in F.spaces[3])
    syms_ = symm(frame)
    fdir, bdir = F.inds[3].dir, frame.inds[fl].dir
    dT = Dict(q => sector_dim(frame, fl, align_charge(syms_, q, fdir, bdir)) for q in keys(dfull))
    for (q, d) in dT
        d <= dfull[q] || throw(ErrorException(
            "$where: the frame holds $d directions in sector $q but the fused local space has " *
            "only $(dfull[q]) -- a frame cannot exceed its local space (frame width " *
            "$(leg_dim(frame, fl)), local $(sum(values(dfull))))"))
    end
    sum(values(dT); init = 0) == leg_dim(frame, fl) || throw(ErrorException(
        "$where: the frame holds $(leg_dim(frame, fl)) directions but only " *
        "$(sum(values(dT); init = 0)) lie in sectors its local space has"))
    return dfull, dT
end

"""
    _cap_to_local_room(Q, U, side) -> U, a trimmed U, or nothing

At most `dim_local(q) - width_Q(q)` leading columns of `U` per sector, so `[Q | U]` cannot exceed
its local space (Telum's cutoff is relative to the global sigma_max, so roundoff in a saturated
sector can otherwise survive). Returns `U` itself when it fits.
"""
function _cap_to_local_room(Q, U, side::Symbol)
    U === nothing && return nothing
    fl = side === :left ? 3 : 1
    a, b = side === :left ? (1, 2) : (2, 3)
    F = fusion_basis(Q, a, b; tag = "cbeRoom")
    dfull, dT = _frame_sector_dims(Q, side, F, "_cap_to_local_room")
    syms_ = symm(Q)
    counts = Dict{Any, Int}()
    over = false
    for (q, d) in dfull
        uq = align_charge(syms_, q, F.inds[3].dir, U.inds[fl].dir)
        du = sector_dim(U, fl, uq)
        du == 0 && continue
        counts[uq] = min(du, d - dT[q])
        over |= counts[uq] < du
    end
    over || return U
    return _trim_per_sector(U, fl, counts)
end

"""
Debug switch, off by default: [`_check_frame`](@ref) asserts at every frame-growth site that the
grown frame still fits its local space. It builds a fusion basis per call, so it is for
localising a non-isometric frame (IT-1 L=16 tol_basis 1e-7, s37test 17341472), not production.
"""
const _FRAME_CHECKS = Ref(false)

"`frame` unchanged; with `_FRAME_CHECKS[]`, first [`_frame_sector_dims`](@ref) at site `where`."
function _check_frame(frame, side::Symbol, where::AbstractString)
    (_FRAME_CHECKS[] && frame !== nothing) || return frame
    a, b = side === :left ? (1, 2) : (2, 3)
    _frame_sector_dims(frame, side, fusion_basis(frame, a, b; tag = "cbeChk"), where)
    return frame
end
