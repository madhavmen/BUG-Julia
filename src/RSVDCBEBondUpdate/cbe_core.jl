
"""
    _unprime(t, leg) -> TLArray

Drop one prime level from `leg`. An environment's bra leg is primed to keep it distinct
from the ket while both are open (`henv.jl`); once it has replaced a link leg it must
carry that link's original prime level again.
"""
#
# ⛔ `to_concrete!`, NOT `to_concrete` -- THIS WAS COPYING THE WHOLE TENSOR TWICE TO CHANGE ONE
# INTEGER. `prime` goes through `_modify_plev`, which alters one `TLIndex`'s prime level and
# then builds its result with `_copy_wmat_storage(q; deep=true)` and
# `_copy_sector_RMTs(q; deep=true)` -- already a FULL DEEP COPY that the caller exclusively
# owns. Wrapping that in the copying `to_concrete` (`_eager_tlarray(q::TLArray) = copy(q)`)
# duplicated every block a second time, for no reader.
#
# Measured at chi=1024 (allocation attribution, job 16333509): this line is 28.5% of bugmid's
# sampled bytes, 9.6% of tdvp2's and 6.1% of cbe1s's. Dropping the second copy halves it.
#
# ⚠ THE FIRST COPY IS STILL THERE, inside `_modify_plev`. Removing it needs a shallow relabel
# that SHARES storage with `t`, which is only sound where the caller owns `t` -- true at these
# call sites (all fresh `contract` results) but not a property of the function. Left for a
# measured follow-up rather than assumed.
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

# ── PROJECT-FIRST SKETCHES: the sketch probes the PROJECTOR, not H*Theta ──────
#
# `Y = (P_perp (H Theta)) Om'`, with `P_perp = I - U0 U0'` applied BEFORE the probe is
# folded in. This is the only sketching path; the fold-Omega-first variant was removed
# on 2026-08-06 at the user's explicit instruction.
#
# THE TWO ORDERINGS AGREE EXACTLY. `P_perp` acts on `(link_l, site_l)` and `Om` acts on
# `(site_r, link_r)` -- disjoint legs, so the operators commute and
#
#     P_perp((H Theta) Om')  ==  (P_perp (H Theta)) Om'
#
# is an identity, not an approximation. So this change moves no numbers; what it changes
# is cost, and only upward. `H Theta` has `d*chi_r` columns against `Om`'s `npre`, so
# projecting first forces the rank-4 two-site object into existence and applies `H` to
# all of it, where folding `Om` in first applied `H` to `npre` columns. That cost was
# raised and the instruction was reaffirmed; it is recorded here so the next reader does
# not rediscover it as a bug.
#
# `_preselect_left/right` still apply `P_perp` to what comes back. That is now redundant
# rather than load-bearing -- `P_perp` is idempotent, so the second application is a
# no-op on an already-projected `Y` -- and it is kept because it also orthonormalises and
# returns the residual the selection needs.

# ⛔ `H*Theta` IS BUILT **ONCE** PER `cbe_expand`, AND THAT IS A FIX, NOT A TIDY-UP.
#
# `cbe_expand` calls its sketch closures THREE times with identical arguments -- `skl(OmR)` and
# `skr(OmL)` for the two preselections, then `skl(QR)` again for the joint final-selection
# ranking (`:664`). Each call used to run its own `apply_h_two_site(frame_theta(f), ...)`, so the
# single most expensive object in the expansion was materialised THREE TIMES per bond per
# half-sweep, and the left projector `P_perp (H Theta)` TWICE.
#
# ⚠ THIS IS WHY THE RANDOMISED SKETCH COULD NOT WIN, and the note in `henv.jl:624` had already
# found the reason without finding the cause: "RSVD reduces flops, and flops were not the cost."
# The sketch shrinks the SVD and the sketch contraction; it does NOT touch `H*Theta`. With the
# expensive part paid three times over, `t_cbe` was dominated by work the sketch is incapable of
# reducing, and the whole-step speedup was capped far below what `Dpre/(d*chi)` promised.
#
# ⚠ AND IT EXPLAINS THE MODEL DEPENDENCE. `apply_h_two_site` scales with the MPO's virtual
# dimension, so the redundancy costs MORE the wider the generator: MEASURED 1.63x on the XX chain
# (MPO dim 4) against 1.07x on the square cylinder (MPO dim 14) -- the wide-MPO models were being
# taxed hardest by the very thing the study was trying to measure.
#
# ⛔ THE ARITHMETIC IS UNCHANGED, BIT FOR BIT: the same tensors are contracted in the same order,
# they are simply not rebuilt. Any accuracy difference after this change would be a bug in it, and
# `tests/rsvd_cbe/` checks exactly that.

"""
    _sketch_closures(f, buildHT) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, sharing ONE `H*Theta` and ONE projection per
side across every call the selection makes.

`buildHT()` returns `H*Theta` with legs `(link_l, site_l, site_r, link_r)`; it is invoked at most
once, and not at all when both sides are already complete and `cbe_expand` returns early.

⚠ THE LAZINESS IS LOAD-BEARING, not stylistic. `cbe_expand` returns before either closure runs
whenever `dex_l <= 0 && dex_r <= 0` (both frames saturated), which at a boundary bond is the
ordinary case -- building `H*Theta` eagerly here would ADD an expensive contraction to exactly the
bonds that previously did none.
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

`Y = (P_perp (H Theta)) Om†`, legs `(link_l, site_l, g)` -- the left matricization of the
DISCARDED-space component of `H Theta`, sketched from the right. This is the candidate
space the left frame is expanded into.

`Om` carries `V0`'s layout `(g, site_r, link_r)`: it IS a candidate right frame, with
`g = Dpre` columns instead of `r`. `Om = f.V0` recovers the exact left factor of
`P_perp (H Theta)` projected on the current right frame.

⚠ A STANDALONE CALL REBUILDS `H*Theta`. Inside a sweep the caching above is what runs; this
method exists for tests and microbenchmarks that want one probe in isolation.
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

# ── the sector-graded Gaussian sketch ────────────────────────────────────────
#
# `GaussRandMat` (`RSVDpreBE0SiQS.m:557`), and the reason CBE removes the need for
# `random_sector_seed`.
#
# The sketch has TWO parts, and the split is deliberate:
#
#   * `Dpre_C = CompRatio·Dpre` columns in the ORTHOGONAL COMPLEMENT of the frame,
#     allocated per charge sector in proportion to `DC(q) = Dfull(q) − DT(q)`;
#   * `Dpre_T = Dpre − Dpre_C` columns INSIDE the frame's own span, allocated in
#     proportion to `DT(q)`.
#
# The complement half is not the whole sketch, on purpose: "do not project into
# complement space, 1-site components are also important for bond update"
# (`RSVDpreBE0SiQS.m:339`).
#
# WHY THIS REPLACES THE RANDOM FILL. A charge sector the frame does not touch has
# `DT(q) = 0`, hence `DC(q) = Dfull(q)` -- the LARGEST complement share of any sector. An
# empty-but-reachable sector is therefore sampled by construction, and it is sampled
# through `H`, so what comes back is the direction the dynamics actually wants rather than
# a random column that the truncating SVD will later discard. Under U(1) with the opposite
# frame frozen the K/L complement can never open such a sector at all (`sectors.jl`
# header), which is why `augment.jl` needs `random_sector_seed` and this does not.
#
# Both halves are drawn from the SAME randomised fusion basis and then separated by the
# projector, which is also how the MATLAB does it: `OmegaC = OmegaC - TO(TO'OmegaC)`
# (`:646`). Randomising a dense buffer instead would inject amplitude into sectors the
# symmetry forbids -- the failure `random_sector_seed` documents.

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

The COMPLETE local basis for one side of a bond, in the frame's own layout: `(link_l, site_l,
g)` for `side = :left` and `(g, site_r, link_r)` for `:right`, exactly as
[`sector_graded_sketch`](@ref) returns, but with `g` the FULL fused dimension and no
randomness anywhere.

THIS IS WHAT MAKES PLAIN CBE A SUBSTITUTION RATHER THAN A SECOND IMPLEMENTATION. Feed this to
the same sketch closure the RSVD path uses and `skl(F)` is the exact `H*Theta` with one side
fused, so the preselection SVD returns the exact leading directions of `P_perp (H Theta)`
instead of an estimate of them. Every downstream step -- preselection, ranking, `_trim_total`,
the orthonormality guard, the direct sum -- is unchanged and unaware of which probe it got.

THE COST IS EXACTLY WHAT THE RSVD EXISTS TO AVOID. `g` is `d*chi` here against the sketch's
`npre ~ 1.2*budget`, so this DOES form the rank-4 two-site object. That is the point of having
both: `cbe_expand(...; exact = true)` is the reference the sketched path approximates, and the
A/B between them is the measurement, not a regression.
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

Per charge sector of `F`'s fused leg (`F = fusion_basis` of the frame's local legs): the local
dimension `dfull[q]` and the frame's width `dT[q]` in it. Throws if the frame is wider than its
local space in any sector, or holds directions in a sector the local space does not have -- an
isometry cannot, so either the frame is not one or the two legs' labels disagree. `where` names
the caller in the message.

⛔ THE FRAME'S BOND LABEL IS READ IN THE FUSED LEG'S ARROW CONVENTION (`align_charge`). A sector's
label depends on the leg's arrow (`q` on one, `dual(q)` on the other, sectors.jl), and a frame
built inside the closure's frontier sweep can carry its bond leg with either arrow.
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

At most `dim_local(q) - width_Q(q)` of the new columns `U` in every charge sector -- the leading
ones, in SVD order -- so `[Q | U]` cannot exceed its local space. `U` unchanged (the same object)
whenever it fits, which is every ordinary step.

⛔ WHY A FRAME COULD OVERFLOW AT ALL. Telum's `svd` cutoff is relative to the GLOBAL sigma_max
across sectors (Telum-fork svd.jl:1672), so once ONE sector of the residual carries a genuine
direction (sigma ~ 1), roundoff in a SATURATED sector (sigma ~ the frame's orthogonality loss)
clears the cut and is renormalised into a fake unit column. MEASURED (s38test 17341708,
IT_FRAME_CHECKS): `_extend_frame` in the closure frontier's Krylov enrichment left 72 directions
in sector 0 against a local space of 70, frame width 253 of 256 -- IT-1 L=16 bug3 tol_basis 1e-7
seed 2, the crash 661c3e7's label alignment did not fix (s37test 17341472: 174 > 92 downstream).
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
