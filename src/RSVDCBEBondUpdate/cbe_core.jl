
"""
    _unprime(t, leg) -> TLArray

Drop one prime level from `leg`. An environment's bra leg is primed to keep it distinct
from the ket while both are open (`henv.jl`); once it has replaced a link leg it must
carry that link's original prime level again.
"""
_unprime(t, leg::Int) = to_concrete(prime(t, leg; inc = -1))

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
        A = to_concrete(H - to_concrete(contract(f.U0, (3,), c, (1,))))
        share && (AL[] = A)
        return A
    end
    function _ar()
        share && AR[] !== nothing && return AR[]
        H = _ht()
        c = contract(H, (3, 4), f.V0', (2, 3))             # (link_l, site_l, bond)
        A = to_concrete(H - to_concrete(contract(c, (3,), f.V0, (1,))))
        share && (AR[] = A)
        return A
    end
    skl = Om -> to_concrete(contract(_al(), (3, 4), Om', (2, 3)))       # (link_l, site_l, g)
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
    sector_graded_sketch(frame, side, npre; comp_ratio=0.5, rng, tag) -> TLArray or nothing

The CBE sketch block for one side of a bond, in the frame's own layout:
`side = :left` gives `(link_l, site_l, g)` like `U0`, `side = :right` gives
`(g, site_r, link_r)` like `V0`.

`npre` is the total column budget `Dpre`. Returns `nothing` only when neither half can be
built at all.

NOTE ON A SATURATED FRAME. If the frame already spans its whole local space there is no
complement, and this returns the isometry half alone -- it does NOT decline. The sketch is
a PROBE applied to the frame on the *opposite* side of the bond, so a saturated right
frame is still a perfectly good probe for expanding the left one. Whether a given side has
room to expand is a separate question, decided by `cbe_expand` from `Dfull - r`; folding
that decision in here would silently refuse to expand one side of the bond because the
other was full.
"""
function sector_graded_sketch(frame, side::Symbol, npre::Int;
                              comp_ratio::Union{Float64, Nothing} = 0.5,
                              rng::AbstractRNG = MersenneTwister(0x5EED),
                              tag::AbstractString = "cbeG")
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    # The ENDPOINTS ARE ALLOWED, and they are the interesting A/B: `1.0` is
    # complement-only (the pure discarded space), `0.0` isometry-only. An earlier version
    # demanded strict inequality and also floored both halves at one column, so
    # "complement only" silently still drew one kept-space column and could not be measured.
    comp_ratio === nothing || 0.0 <= comp_ratio <= 1.0 || throw(ArgumentError(
        "comp_ratio must be in [0, 1] or nothing, got $comp_ratio"))
    npre >= 1 || throw(ArgumentError("npre must be at least 1, got $npre"))

    fl = side === :left ? 3 : 1                       # the frame's bond leg
    a, b = side === :left ? (1, 2) : (2, 3)           # the fused local pair
    sk = side === :left ? 3 : 1                       # the sketch leg, in frame layout
    perp_of = side === :left ? perp_component : perp_component_right
    # `fusion_basis` returns (a, b, fused); a right frame carries its bond leg first.
    tolayout(t) = side === :left ? t : to_concrete(permutedims(t, (3, 1, 2)))

    F = fusion_basis(frame, a, b; tag = tag)
    dfull = Dict(q => d for (q, d) in F.spaces[3])
    dT    = Dict(q => sector_dim(frame, fl, q) for q in keys(dfull))
    dC    = Dict(q => max(dfull[q] - dT[q], 0) for q in keys(dfull))

    # ── comp_ratio = nothing: NO PROJECTOR IN THE PROBE AT ALL ─────────────────────────
    #
    # A plain sector-graded Gaussian on the full local space: columns allocated across charge
    # sectors as usual, but neither projected out of the frame nor into it. The complement is
    # then reached by SAMPLING rather than by construction.
    #
    # WHY THIS IS WORTH HAVING. `comp_ratio` is one of the two hyperparameters the iterative
    # scheme was meant to remove: a fixed split is a GUESS at how much of the probe should look
    # for new directions versus refine old ones, made once, before anything is known. With the
    # expanding-basis solver the probe is REDRAWN at every growth pass from an advancing RNG
    # stream, so successive passes explore different directions in every sector and the
    # coverage accumulates. Sampling over passes replaces guessing once.
    #
    # ORTHOGONALITY IS NOT AT RISK. The probe only decides which directions get LOOKED AT; the
    # preselection still applies the exact `P_perp = I - U0 U0'` to what comes back, and the
    # guard before the direct sum re-establishes orthonormality. So dropping the projection
    # here cannot admit a direction that overlaps the frame.
    #
    # It is also strictly cheaper: two `perp_component` calls per side per pass disappear.
    if comp_ratio === nothing
        R = _trim_per_sector(_randomize_like(F, rng), 3, _alloc_columns(dfull, npre))
        R === nothing && return nothing
        Rl = to_concrete(tolayout(R))
        return norm(Rl) > 0 ? Rl : nothing
    end

    # `n_c + n_t == npre` is only the TARGET split, and either may be zero -- that is what
    # makes the pure cases reachable.
    n_c = clamp(round(Int, npre * comp_ratio), 0, npre)
    n_t = npre - n_c

    # THE SHORTFALL IS REDISTRIBUTED, NOT DROPPED -- MEASURED. `_alloc_columns` caps every
    # sector at its own dimension, so a half whose subspace is small delivers fewer columns
    # than asked. `npre` is the probe's WIDTH, and the expansion can admit at most that many
    # directions however large `dex` is -- they ARE the sketch's columns. Near a chain edge the
    # opposite frame's complement is 0- or 1-dimensional, so a complement-weighted split
    # collapsed the probe to 0-1 columns and the expansion silently declined WITH ROOM TO
    # SPARE: at `comp_ratio = 1.0`, `dex = 64`, an L=8 warm state gave bond 1 `rank 2 -> (2,2)`
    # and bond 3 `6 -> (6,6)`, leaving right residuals of 3.1e-4 and 0.19 untouched. One
    # blocked narrow bond then caps every bond inward -- the staircase profile the MPO sweep
    # froze into. Handing the shortfall to the other half keeps the probe `npre` wide whenever
    # the fused space has the dimensions for it, which is what the RSVD oversampling guarantee
    # assumes in the first place.
    cc = _alloc_columns(dC, n_c)
    n_t += n_c - sum(values(cc); init = 0)        # complement short -> widen the isometry half
    ct = _alloc_columns(dT, n_t)
    short = n_t - sum(values(ct); init = 0)
    short > 0 && (cc = _alloc_columns(dC, n_c + short))  # and back, if that half is short too

    # NEVER RETURN AN EMPTY PROBE while the space has dimensions. The sketch probes the frame
    # on the OPPOSITE side of the bond, so a saturated frame is still a perfectly good probe --
    # `Om = frame` is the EXACT probe. Declining here refuses to expand one side of a bond
    # because the other was full, the edge blockage above in its purest form.
    # (`comp_ratio = 1.0` on a saturated frame hits it: `dC` is all zeros and `n_t` is 0.)
    if sum(values(cc); init = 0) + sum(values(ct); init = 0) == 0
        ct = _alloc_columns(dfull, npre)
    end

    blocks = Any[]

    # Complement half: random, then the frame projected OUT.
    RC = _trim_per_sector(_randomize_like(F, rng), 3, cc)
    if RC !== nothing
        C = to_concrete(perp_of(frame, tolayout(RC)))
        norm(C) > 0 && push!(blocks, C)
    end

    # Isometry half: random, then the frame projected IN. `R - perp(R)` is the projection
    # onto the frame's span, built from the same primitive so the two halves can never
    # disagree about what "the frame's span" is.
    RT = _trim_per_sector(_randomize_like(F, rng), 3, ct)
    if RT !== nothing
        R = tolayout(RT)
        T = to_concrete(R - to_concrete(perp_of(frame, R)))
        norm(T) > 0 && push!(blocks, T)
    end

    isempty(blocks) && return nothing
    length(blocks) == 1 && return blocks[1]
    return to_concrete(oplus(blocks, (sk,)))
end

# ── preselection and final selection ─────────────────────────────────────────
#
# `RSVDpreBE0SiQS.m` in two stages, and the division of labour matters:
#
#   PRESELECT  sketch H, project off the frame, orthonormalise -> Dpre candidates.
#              Cheap and generous; `Dover` (default 20% of Dex) is the oversampling that
#              makes the randomized range finder reliable.
#   FINAL      SVD the small Dpre x Dpre matrix `U0L' (H Theta) V0R'` and keep Dex.
#              This is the step that RANKS candidates by their weight in the two-site
#              action -- the thing the K/L augmentation cannot do, since it admits every
#              direction it finds regardless of weight and leaves the ranking to the
#              S-step SVD.
#
# `RepairOrtho` (`RSVDpreBE0SiQS.m:743`) is folded into `_preselect_*` as a second
# projection pass. It is not optional bookkeeping: `oplus` concatenates blocks without
# orthogonalising them, so `[U0 | Q]` is an isometry only if `Q` is exactly orthogonal to
# `U0`, and a direction surviving the SVD cutoff can still carry a `U0` component of order
# eps/cutoff. The same omission in `augment.jl::_complement_basis` duplicated a column and
# grew the norm by 128x over five steps.

"Relative weight discarded by an SVD, `sqrt(Σcut σ² / Σall σ²)`."
function _trunc_weight(res)
    w(list) = sum(Float64(s)^2 * Float64(deg) for (s, deg, _, _) in list; init = 0.0)
    kept, cut = w(res.kept_list), w(res.trunc_list)
    tot = kept + cut
    return tot == 0.0 ? 0.0 : sqrt(cut / tot)
end

const _CBE_EPS = 100 * eps(Float64)

# DISTINCT BOND TAGS FOR THE TWO PRESELECTIONS. The final-selection matrix is
# `U0L† (HΘ) V0R†`, whose two legs are the bond legs the two preselection SVDs created. On
# Telum's default tags both would be `svdL` with the same arrow, and a TLArray may not
# carry two identical indices -- MEASURED: "Duplicate TLIndex with non-empty itag in
# TLArray.inds: TLIndex(svdL, '+', 0, 0, false)" (job 94959). Exactly the collision
# `bond_frame` avoids by giving its two SVDs distinct tag pairs.
const _TAG_L, _TAG_LR = "cbeL,L", "cbeL,R"
const _TAG_R, _TAG_RR = "cbeR,L", "cbeR,R"
# Distinct again for the final orthonormality guard, for the same reason: its output is a
# fresh bond leg, and reusing a preselection tag risks the duplicate-index error above.
const _TAG_OL, _TAG_OLR = "cbeO,L", "cbeO,R"
const _TAG_OR, _TAG_ORR = "cbeP,L", "cbeP,R"

"""
    _preselect_left(U0, Y, stol) -> (Q, err) or (nothing, 0.0)

Orthonormal basis of the part of the sketched action `Y` lying outside `U0`, with the
`RepairOrtho` second pass. `Y` and `Q` carry `(link_l, site_l, bond)`.

`Y` is normalised before the SVD because `cutoff` is relative to the largest singular
value: the new directions are O(tau)-small in absolute terms, and thresholding relative to
their own scale is what keeps a meaningful tolerance (the trap recorded in `augment.jl`).
"""
function _preselect_left(U0, Y, stol::Float64)
    R = to_concrete(perp_component(U0, Y))
    n = norm(R)
    n <= _CBE_EPS * max(norm(Y), 1.0) && return (nothing, 0.0)
    res = svd(to_concrete((1.0 / n) * R), (1, 2), _TAG_L, _TAG_LR;
              cutoff = stol, get_lists = true)
    Q = to_concrete(res.U)
    R2 = to_concrete(perp_component(U0, Q))            # RepairOrtho
    norm(R2) <= _CBE_EPS * norm(Q) && return (nothing, 0.0)
    Q = to_concrete(svd(R2, (1, 2), _TAG_L, _TAG_LR; cutoff = 1e-13).U)
    return (Q, _trunc_weight(res))
end

"""
    _preselect_right(V0, Y, stol) -> (Q, err) or (nothing, 0.0)

Row mirror: `Y` and `Q` carry `(bond, site_r, link_r)`, the frame leg is 1 and the
orthonormalisation is over `(site_r, link_r)`.
"""
function _preselect_right(V0, Y, stol::Float64)
    R = to_concrete(perp_component_right(V0, Y))
    n = norm(R)
    n <= _CBE_EPS * max(norm(Y), 1.0) && return (nothing, 0.0)
    res = svd(to_concrete((1.0 / n) * R), (2, 3), _TAG_R, _TAG_RR;
              cutoff = stol, get_lists = true)
    Q = to_concrete(permutedims(res.U, (3, 1, 2)))
    R2 = to_concrete(perp_component_right(V0, Q))      # RepairOrtho
    norm(R2) <= _CBE_EPS * norm(Q) && return (nothing, 0.0)
    Q = to_concrete(permutedims(svd(R2, (2, 3), _TAG_R, _TAG_RR; cutoff = 1e-13).U,
                                (3, 1, 2)))
    return (Q, _trunc_weight(res))
end

"""
    CBEExpansion

What [`cbe_expand`](@ref) produced at one bond.

  - `U_ex`, `V_ex`  the expanded frames. They SPAN `[U0 | new]` but are ROTATED within that
    span, so neither contains the old frame as a block. Spanning it is what keeps
    `Ŝ₀ = U_ex† Θ₀ V_ex†` lossless; *not* containing it is what keeps `Ŝ₀` free of an
    identically-zero block, which Telum's strict `sv > cutoff*sv_max` selector would delete
    at the next gauge move. See the rotation block in [`cbe_expand`](@ref).
  - `n_new_l`, `n_new_r`  new directions admitted on each side.
  - `err_pre`  weight the preselection SVDs discarded (both sides combined in quadrature).
  - `err_fnl`  weight the final selection discarded -- the honest measure of what the
    `Dex` cap cost, since it is the tail of the ranked candidate spectrum.
"""
struct CBEExpansion
    U_ex::Any
    V_ex::Any
    n_new_l::Int
    n_new_r::Int
    err_pre::Float64
    err_fnl::Float64
end

"""
    cbe_expand(f, h, i, lenv, renv; kwargs...) -> CBEExpansion

Expand the frames at bond `(i, i+1)` by controlled bond expansion.

This is the BASIS UPDATE of CBE-BUG, and it replaces both of `kls_bond_update`'s
mechanisms at once: there is no K/L ODE (so no frozen opposite frame and nothing
non-Hermitian to Arnoldi), and no random sector fill (the sector-graded sketch reaches
empty-but-reachable sectors through `H` itself).

Keywords:

  - `dex` -- new directions to admit per side. `0` (default) hands the budget to the
    `growth` schedule below; any positive value is an absolute per-side budget and
    overrides the schedule entirely.
  - `growth` -- the schedule's target, as a multiple of the NEIGHBOURHOOD max bond
    dimension: `budget = ceil(growth*dmax) - r`. Default `2.0`, i.e. a bond at the
    neighbourhood max may double. See the budget block in the body for the measurement
    that set it.
  - `dover` -- oversampling for the preselection. `nothing` gives `ceil(0.2*dex)`, i.e.
    `Dpre = ceil(1.2*dex)` as in the reference.
  - `comp_ratio` -- the sketch's complement/isometry split (`CompRatio`). **Default `1.0`,
    i.e. the sketch is drawn ENTIRELY from the frame's orthogonal complement -- the
    discarded space.**

    ⚠️ THIS DEPARTS FROM THE REFERENCE DELIBERATELY. `RSVDpreBE0SiQS.m:339` reads "do not
    project into complement space, 1-site components are also important for bond update",
    i.e. the MATLAB explicitly rejects the pure-complement choice and splits the probe.
    The default here is `1.0` because one knob that is either on or off is easier to reason
    about than a continuum, and the complement is the part that finds NEW directions.
    `0.5` restores the reference's split; `nothing` drops the split entirely (probe drawn
    from the full randomised fusion basis, with the exact `P_perp` still applied downstream
    in preselection, so orthogonality is not at risk either way).
  - `sulz_cap` -- keep the expanded rank at or below `2r`. **Default `false`, which applies
    NO bound** -- growth is limited by `room` (a hard local-space limit) and by `maxdim` at
    the truncation. Note that neither setting reproduces the reference, which caps the
    EXPANSION at `2r` for a total of `3r` (`RSVDpreBE0SiQS.m:77`); `true` caps the TOTAL at
    `2r`, which is tighter than both.
  - `preselect_only` -- skip the final selection and take the preselected candidates
    directly, the reference's `'-p'` mode. Diagnostic: it isolates how much the
    weight-ranking is worth.

THE K/L ODEs ARE NEVER SOLVED. This routine supplies the DIRECTIONS those solves would
have found -- one application of `H`, sketched -- and the per-bond Galerkin update in
`cbe_lubich_sweep` (REMOVED) then builds the Krylov basis on the expanded bond and takes
the step in it. So there is no `expv` per bond on a one-site object, only the `O(χ²)`
matrix-free 0-site Krylov, and no rank-4 tensor anywhere.
"""
cbe_expand(f::BondFrame, h::XXZChain, i::Int,
           lch::ChannelSet, rch::ChannelSet; share_ht::Bool = true, kwargs...) =
    cbe_expand(f,
               _sketch_closures(f,
                   () -> apply_h_two_site(frame_theta(f), h, i, lch, rch);
                   share = share_ht)...; kwargs...)

"""
    cbe_expand(f, skl, skr; kwargs...) -> CBEExpansion

The SELECTION half of controlled bond expansion, with the H-action supplied as two
closures instead of read off a global Hamiltonian:

  - `skl(Om)` -- `(H Theta) Om†` in `U0`'s layout `(link_l, site_l, g)`, for an `Om` in
    `V0`'s layout. The candidate space the LEFT frame expands into.
  - `skr(Om)` -- the mirror, `(g, site_r, link_r)` from an `Om` in `U0`'s layout.

Everything downstream of those two calls -- the growth budget, the sector-graded sketch,
the two preselections with `RepairOrtho`, the weight ranking of the final selection, the
`Empty` branch, the direct sums -- is shared. That is the point of the split: the
gate-based path (`cbe_bond_update.jl`) and the environment-dressed path exercise the SAME selection code,
so confirming CBE works against a validated Trotter sweep also validates the selection the
Lubich sweep depends on. Nothing here knows which caller it has.
"""
function cbe_expand(f::BondFrame, skl, skr;
                    dex::Int = 0,
                    growth::Float64 = 2.0,
                    dover::Union{Nothing, Int} = nothing,
                    comp_ratio::Union{Float64, Nothing} = 0.5,
                    sulz_cap::Bool = false,
                    rmax::Int = 0,
                    preselect_only::Bool = false,
                    exact::Bool = false,
                    stol_pre::Float64 = 1e-10,
                    stol_fnl::Float64 = 1e-13,
                    rng::AbstractRNG = MersenneTwister(0x5EED))
    U0, V0 = f.U0, f.V0
    r = f.old_rank
    none() = CBEExpansion(U0, V0, 0, 0, 0.0, 0.0)

    # Room to expand, per side: the local space minus what the frame already spans.
    #
    # THE FUSED DIMENSION IS ASKED FOR, NOT COMPUTED AS A PRODUCT, and that distinction is what
    # makes this work under a NON-ABELIAN symmetry. `leg_dim(U0,1) * leg_dim(U0,2)` is correct
    # only when every pair of input sectors fuses to exactly one output of the same size, i.e.
    # for abelian charges. Under SU(2) a spin-1/2 site is ONE multiplet, so that product reads
    # `chi * 1 = chi`, giving `room = chi - r ~ 0` -- CBE would decline to expand and the state
    # would freeze at chi = 1, with the same signature as `grow_iters = 0`. The truth is that
    # each bond multiplet `S_a` fuses to `S_a +/- 1/2`, so the fused space has ~2*chi multiplets.
    #
    # `reachable_sectors` goes through `fusion_basis` -> `getIdentity` + `svd`, so Telum applies
    # whichever symmetry the tensors actually carry and this line never needs to know. It
    # reduces EXACTLY to the old product for abelian charges, so it is not a special case.
    #
    # KNOWN COST: this duplicates a `fusion_basis` that `sector_graded_sketch` /
    # `full_local_basis` will build again for the same frame a few lines below -- four calls per
    # expand where two would do. Folding them together means threading the bases through the
    # probe, which is worth doing only once it shows up in a measurement.
    fused_dim(t, a, b) = sum(d for (_, d) in reachable_sectors(t, a, b); init = 0)
    room_l = fused_dim(U0, 1, 2) - r
    room_r = fused_dim(V0, 2, 3) - r
    # GROWTH IS NEIGHBOURHOOD-COUPLED, as in the reference (`RSVDpreBE0SiQS.m:66-77`):
    #
    #   Dmax = max([MDB0(1), MDB1(1), MDB2(1)])   % LEFT, centre and RIGHT bond dims
    #   Dex  = ceil(1.1*Dmax) - MDB1(1)           % target 1.1x the neighbourhood max
    #   Dex  = min(Dex, ceil(2*MDB1(1)))          % cap the EXPANSION at 2r => total 3r
    #
    # The coupling is the point: it lets a WIDE bond pull its narrow neighbours up. A purely
    # local `budget = r` lets a bond sitting at r=1 propose only one direction, so growth
    # crawls one direction per bond per step and stalls -- measured at L=8, `bond_dims`
    # froze at [1,1,2,4,2,1,1] while 2-site TDVP had [2,4,6,8,6,4,2] at the same time, and
    # the centre then hit `room_l = d·χ_{i-1} − r = 0` and could not expand at all.
    #
    # THE CONSTANT IS OURS, NOT THE REFERENCE'S: `growth = 2.0`, not the reference's `1.1`.
    # The reference is DMRG, where each sweep re-converges the same state, so a 10% ratchet
    # compounds over sweeps and costs only iterations. A TIME INTEGRATOR gets one shot per
    # step: a direction the step needs and does not admit is not deferred, it is gone, and
    # the state carries that error forward. Measured (`benchmarks/cbe_bond_update_budget.jl`,
    # XX L=10, dt=0.02, t=0.5, maxdim=64) at the 1.1 schedule, whose budget on a bond at
    # r=13 is `ceil(1.1*13)-13 = 2`:
    #
    #   bond_update (1.1 schedule)  err 1.43e-04   err_fnl 6.2e-01   22.4s
    #   bond_update (dex=4)         err 3.54e-04   err_fnl 0          4.6s
    #   bond_update (dex=8)         err 1.17e-06   err_fnl 0          4.7s
    #   bond_update (dex=16, 64)    err 1.17e-06   err_fnl 0          4.5s   <- saturated
    #   bond_update_bug!         err 5.81e-07                    157.0s
    #
    # `err_fnl = 0.62` is the schedule DISCARDING 62% of the ranked candidate weight for want
    # of budget -- the cap, not the selection, was the accuracy limiter. Wall time is flat in
    # the budget (4.4-4.7s from dex=1 to dex=64) because the sweep dominates the sketch, so a
    # generous default is very nearly free. `growth = 2.0` puts a bond at the neighbourhood
    # max on `budget = r`, past the measured saturation at every rank in that run, and it
    # SCALES -- an absolute floor of 8 would have been tuned to this one chain. Wide bonds are
    # still bounded by `room` (a hard limit) and by `maxdim` at the truncation.
    dmax = max(leg_dim(U0, 1), r, leg_dim(V0, 3))
    budget = dex <= 0 ? max(ceil(Int, growth * dmax) - r, 1) : dex
    # NO SULZ BOUND BY DEFAULT. The `2r` bound belongs to the RANK-ADAPTIVE BUG, and this
    # method is not that -- the rank here is set by CBE's selection, so the bound has nothing
    # to enforce and only throttles. Growth is limited by `room` (the local space, a hard
    # limit) and by `maxdim` at the truncation, which is where a rank ceiling belongs.
    #
    # Two earlier readings of it, both wrong and both measured:
    #   * per-bond `min(budget, r)` -- caps each bond at twice its OWN rank. A bond at r=1
    #     could never exceed 2, the narrow bonds never opened, and `room_l = d·χ_{i-1} − r`
    #     then pinned the centre: L=8 froze at [1,1,2,4,2,1,1] against TDVP2's [2,4,6,8,6,4,2].
    #   * even read globally (`2*rmax`), it is still a rank-adaptive-BUG constraint that does
    #     not apply here.
    # `sulz_cap = true` keeps the global form available purely as an A/B.
    rm = rmax > 0 ? rmax : r
    sulz_cap && (budget = min(budget, max(2 * rm - r, 0)))
    dex_l, dex_r = min(budget, room_l), min(budget, room_r)
    (dex_l <= 0 && dex_r <= 0) && return none()      # both frames already complete

    npre = preselect_only ? max(budget, 1) :
           dover === nothing ? ceil(Int, 1.2 * budget) : budget + dover

    # ---- preselection: sketch H from each side, project off the frame ----------
    #
    # THE PROBE GOES ON `H Theta`, NOT ON THE PROJECTOR, and that ordering is the method. To
    # expand the LEFT frame a Gaussian is contracted into the RIGHT leg only,
    #
    #     Y = (H Theta) Om_R'      ->      Q_L = orth(P_perp Y)
    #
    # so the side being expanded stays EXACT -- `Y` keeps all `d*chi_l` rows and `P_perp` is
    # applied to the true object -- and `Q_L` estimates the RANGE of `P_perp (H Theta)`, i.e.
    # the directions `H` actually points in. The SVD is `O(d*chi_l * Dpre^2)`, so it does grow
    # with the bond dimension.
    #
    # A `two_sided` variant that folded BOTH Gaussians onto the projectors instead
    # (`Q_L = orth(P_perp Om_L)`, ranked by the small `M = Q_L'(H Theta)Q_R`) was implemented
    # and REMOVED. Its SVD was independent of chi, which is a real property, but the candidate
    # space was then a random slice of the complement that `H Theta` only SCORED rather than
    # one it generated, and `M` coupled the two sides so a saturated side vetoed its partner --
    # fatal at a boundary bond, where `room_r = d - r` reaches zero. Measured on `jan_bug!`
    # (XXZ, L=4/6): first order instead of second, 2.0e-2 instead of 1.8e-5 over eight steps,
    # and off a product state the rank never left chi = 1 at all.
    #
    # `exact = true` swaps the thin Gaussian for the COMPLETE fused basis, which turns this
    # from RSVD-CBE into plain CBE: `skl(F)` is then the exact `H*Theta` and the preselection
    # SVD returns the true leading directions of `P_perp (H Theta)` rather than an estimate.
    # Nothing downstream changes or needs to know. `comp_ratio`, `npre`, `dover` and `rng` are
    # all inert on this path -- there is no probe to grade, oversample or draw.
    probe(frame, side) = exact ? full_local_basis(frame, side) :
                         sector_graded_sketch(frame, side, npre;
                                              comp_ratio = comp_ratio, rng = rng)

    QL, err_l = if dex_l > 0
        OmR = probe(V0, :right)
        OmR === nothing ? (nothing, 0.0) : _preselect_left(U0, skl(OmR), stol_pre)
    else
        (nothing, 0.0)
    end

    QR, err_r = if dex_r > 0
        OmL = probe(U0, :left)
        OmL === nothing ? (nothing, 0.0) : _preselect_right(V0, skr(OmL), stol_pre)
    else
        (nothing, 0.0)
    end
    err_pre = sqrt(err_l^2 + err_r^2)

    (QL === nothing && QR === nothing) && return none()

    # ---- final selection: rank the candidates by their weight in H Theta ------
    #
    # PER SIDE, NOT JOINTLY -- and the two-sided expansion is why. The reference's CBE is
    # DIRECTIONAL: `CBE1SiQS(..., direction)` expands one side per call ('>>' or '<<'), so
    # the joint matrix `UMU = QL†(HΘ)QR` only ever ranks the one side being expanded. Reusing
    # it to rank BOTH sides at once imports a coupling that is not in the reference, and it
    # lets a saturated side veto its partner. Two ways, both MEASURED on an L=8 warm state at
    # `dex = 64`:
    #
    #   * `Nkeep = min(dex_l, dex_r)` throttles each side to the OTHER side's headroom. Bond 6
    #     has `room_r = d*chi_r - r = 2*2 - 3 = 1`, so the left was capped at one direction and
    #     its residual stayed at 9.97e-3 while the right reached 7.5e-20. Bond 4 the same.
    #   * the `Empty` verdict was GLOBAL. At bond 3 the left frame already spanned the whole
    #     left image (residual 8.4e-12), so `QL` was numerical noise and `norm(UMU) < 1e-11` --
    #     which returned NOTHING ON EITHER SIDE, abandoning a right residual of 0.19. A
    #     negligible `UMU` means no candidate PAIR carries weight with both sides simultaneously
    #     outside their frames; it says nothing about the one-sided components, which are
    #     exactly the "1-site" contributions the reference insists matter (`:339`).
    #
    # So the joint SVD is used only where it can serve a side's whole budget; otherwise that
    # side falls back to its own preselected candidates, which are ALREADY weight-ordered (the
    # preselection SVD's singular values are the weights in the sketched action). Each side
    # decides independently, and either may come back empty without silencing the other.
    njl = QL === nothing ? 0 : leg_dim(QL, 3)
    njr = QR === nothing ? 0 : leg_dim(QR, 1)

    joint, resj, err_fnl = 0, nothing, 0.0
    if !preselect_only && njl > 0 && njr > 0
        UMU = to_concrete(permutedims(contract(skl(QR), (1, 2), QL', (1, 2)), (2, 1)))
        # EMPTY: the left and right candidate spaces share no charge sector, so no pair of them
        # can carry amplitude (the reference's `Empty` branch, :461; the same constraint
        # `pairable_charges` enforces for the random fill). Now only disables the JOINT ranking.
        if length(UMU.qlabels) > 0 && norm(UMU) >= 1e-11
            joint = min(dex_l, dex_r, njl, njr)
            if joint > 0
                resj = svd(UMU, (1,); Nkeep = joint, cutoff = stol_fnl, get_lists = true)
                err_fnl = _trunc_weight(resj)
            end
        end
    end

    TLC = if njl == 0
        nothing
    elseif resj !== nothing && dex_l <= joint
        to_concrete(contract(QL, (3,), resj.U, (1,)))
    else
        _trim_total(QL, 3, dex_l)
    end
    TRC = if njr == 0
        nothing
    elseif resj !== nothing && dex_r <= joint
        to_concrete(contract(resj.Vd, (2,), QR, (1,)))
    else
        _trim_total(QR, 1, dex_r)
    end

    # Re-establish orthonormality and orthogonality to the frame before the direct sum. See
    # the guard block above `_trim_total` for why this is not redundant with the preselection's
    # `RepairOrtho`: that pass ran on `Q`, this one runs on what the final selection made of it.
    TLC = _reortho_left(U0, TLC)
    TRC = _reortho_right(V0, TRC)

    (TLC === nothing && TRC === nothing) && return CBEExpansion(U0, V0, 0, 0, err_pre, 0.0)

    ltag, rtag = U0.inds[3].itags, V0.inds[1].itags
    U_ex, n_l = if TLC === nothing
        (U0, 0)
    else
        (to_concrete(oplus([U0, to_concrete(setitag(TLC, 3, ltag))], (3,))),
         leg_dim(TLC, 3))
    end
    V_ex, n_r = if TRC === nothing
        (V0, 0)
    else
        (to_concrete(oplus([V0, to_concrete(setitag(TRC, 1, rtag))], (1,))),
         leg_dim(TRC, 1))
    end

    # NO ROTATION HERE, deliberately. An earlier version rotated the expanded frame into the
    # singular basis of `Ŝ₀ + tau·H₀Ŝ₀`, on the theory that the gauge move was deleting the
    # expansion. It was not: an expansion is LOSSLESS, so it cannot change the state's
    # Schmidt rank, and `canonical!` collapsing a widened bond back is correct reporting
    # rather than deletion. Rank grows only where the state is EVOLVED in the widened space,
    # which is what the per-bond Algorithm 7 update in `cbe_lubich_sweep` is for. The
    # rotation was lossless but bought nothing, so it is gone.
    if sulz_cap
        # Only checked when the bound is deliberately switched on, and in its GLOBAL form
        # (`2*rmax`) -- the per-bond reading was wrong, see the budget block above.
        lim = 2 * rm
        leg_dim(U_ex, 3) <= lim || error(
            "CBE expansion broke the global Sulz bound on the left: " *
            "$(leg_dim(U_ex, 3)) > 2*rmax = $lim")
        leg_dim(V_ex, 1) <= lim || error(
            "CBE expansion broke the global Sulz bound on the right: " *
            "$(leg_dim(V_ex, 1)) > 2*rmax = $lim")
    end

    return CBEExpansion(U_ex, V_ex, n_l, n_r, err_pre, err_fnl)
end

# ── orthonormality guard on the appended blocks ──────────────────────────────────────
#
# `[U0 | C_L]` is an isometry only while `C_L` is orthonormal AND `C_L ⊥ U0`, and every
# downstream guarantee rests on that: the losslessness of the expansion, `Ŝ₀ = U_ex† Θ₀ V_ex†`
# reproducing `Θ₀`, and the norm preservation of the S-step. Both properties hold by
# construction -- the preselection returns an orthonormal `Q ⊥ U0`, the ranked branch rotates it
# by an isometry, the fallback slices columns out of it -- so this re-establishes rather than
# repairs. It is insurance on the one place error could enter (the ranked branch contracts over
# a summed index) and it is cheap: one SVD of a `(dχ × n_new)` block with `n_new ≤ dex`.
#
# TELUM HAS NO QR PRIMITIVE, so `svd(...).U` is the orthonormalisation, exactly as in
# `_preselect_left`'s `RepairOrtho` pass.
#
# THE FRAME IS PROJECTED OUT FIRST, and that ordering is the substance. Orthonormalising alone
# would fix the block's internal Gram matrix while leaving any component along `U0` untouched --
# and that component, not the internal one, is what breaks the direct sum's isometry.
#
# The cutoff is `1e-14`: tight enough to keep a column whose only defect is a `1e-16` overlap
# with the frame, loose enough to drop one that genuinely collapses under the projection and so
# carries no direction at all. A dropped column is reported through `n_new_*`, never silently.
_reortho_left(U0, C) = _reortho(C, X -> perp_component(U0, X),
                                R -> svd(R, (1, 2), _TAG_OL, _TAG_OLR; cutoff = 1e-14).U)

_reortho_right(V0, C) = _reortho(C, X -> perp_component_right(V0, X),
                                 R -> permutedims(
                                     svd(R, (2, 3), _TAG_OR, _TAG_ORR; cutoff = 1e-14).U,
                                     (3, 1, 2)))

function _reortho(C, perp, orth)
    C === nothing && return nothing
    R = to_concrete(perp(C))
    norm(R) <= _CBE_EPS * max(norm(C), 1.0) && return nothing
    return to_concrete(orth(R))
end


"Keep at most `n` columns of `Q` on `leg`, walking sectors in Telum's order."
function _trim_total(Q, leg::Int, n::Int)
    n <= 0 && return nothing
    leg_dim(Q, leg) <= n && return Q
    left = n
    keep = Dict{Any, Int}()
    for (q, d) in Q.spaces[leg]
        keep[q] = min(d, left)
        left -= keep[q]
    end
    any(v > 0 for v in values(keep)) || return nothing
    return to_concrete(getsub(Q, leg, s -> (k = get(keep, s, 0); k == 0 ? nothing : 1:k)))
end

# ── prebuilt-stack convenience wrappers ──────────────────────────────────────
#
# The kernels above take the channels on the two links they actually use, so a sweep can
# CARRY them (O(1) per bond) instead of rebuilding a stack (O(L) per bond). These wrappers
# keep the stack-indexed call sites -- the tests, and anything not sweeping -- unchanged.

sketch_h_left(f::BondFrame, h::XXZChain, i::Int,
              lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_left(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

sketch_h_right(f::BondFrame, h::XXZChain, i::Int,
               lenv::LeftEnvStack, renv::RightEnvStack, Om) =
    sketch_h_right(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2), Om)

cbe_expand(f::BondFrame, h::XXZChain, i::Int,
           lenv::LeftEnvStack, renv::RightEnvStack; kwargs...) =
    cbe_expand(f, h, i, left_channels(lenv, i), right_channels(renv, i + 2); kwargs...)

# ── shared run configuration ──────────────────────────────────────────────────
#
# Lives here, not in a mode submodule, because all three modes and both paths (gate,
# MPO-environment) are configured by the same expansion knobs. Only the fields that mean
# something different per mode are documented per mode.

"""
    CBEBugOptions

Controls for one run of ANY of the three modes, on either path. Mirrors `BondUpdateOptions`
where the meaning is the same, so the integrators can be driven from the same campaign code.

  - `dt`, `n_steps` -- step size and how many to take. `dt` is a REAL time step in
    real time (`tau = -im*dt`) and an INVERSE-TEMPERATURE step in imaginary time
    (`tau = -dt`); the mode
    forms `tau` from it. [`GroundState`](@ref) ignores both and uses `n_sweeps`/`etol`.
  - `dex`, `growth`, `dover`, `comp_ratio`, `sulz_cap`, `preselect_only` -- the expansion,
    forwarded to [`cbe_expand`](@ref). `dex = 0` (default) uses the `growth` schedule;
    `growth = 1.1` reproduces the reference's DMRG ratchet, for the A/B.
  - `maxdim`, `trunc_thresh` -- truncation, applied at the centre and in the final sweep.
  - `normalize` -- rescale after each step; the norm is recorded BEFORE rescaling.
  - `maxiter`, `tol` -- Krylov budget for the single 0-site exponential.
  - `seed` -- one RNG for the whole run, so a run is reproducible while consecutive steps
    still draw different sketches.

NO `order` FIELD on the MPO path. `bond_update_bug!` offers `:lie` / `:strang` because it
Trotterises the even and odd bond groups and has a splitting error to compose away. The
MPO-path CBE-BUG takes one projected exponential per step over a globally expanded subspace
-- there is no operator splitting to symmetrise, so a composition parameter would be
meaningless there. The gate path is always Strang.

WHY THE GATE PATH STORES NO ENVIRONMENTS. Its environments are the IDENTITY, and that is a
property of the splitting rather than an omission. `H = H_odd + H_even` and each group is a
sum of two-site terms on disjoint pairs, so `exp(tau*H_odd)` factorises exactly into one
`exp(tau*h_i)` per bond: the operator at a bond IS the bare gate, and the rest of the chain
is not neglected, it is a different factor of the same product applied at its own bond. The
sweep calls `canonical!(psi, i)` before each bond, which makes everything to the left a left
isometry and everything to the right a right isometry, so `L = R = I` exactly. The MPO path
cannot do this because the full `H` does not factorise, which is what `henv.jl` is for.

The consequence, and it is what fixes the mode/path matrix: the gate path's local operator
has no knowledge of the chain, so a local EIGENSOLVE there would return the ground state of
one two-site term with frozen neighbours -- a different quantity, not an approximation to
the global ground state. Ground state therefore runs on the MPO path only. Imaginary time
runs on both, because `exp(-dt*H)` Trotterises exactly as `exp(-im*dt*H)` does.
"""
Base.@kwdef struct CBEBugOptions
    dt::Float64 = 0.05
    n_steps::Int = 10
    dex::Int = 0
    growth::Float64 = 2.0
    dover::Union{Nothing, Int} = nothing
    comp_ratio::Union{Float64, Nothing} = 0.5
    sulz_cap::Bool = false
    preselect_only::Bool = false
    centre_expand::Bool = true
    maxdim::Int = 200
    trunc_thresh::Float64 = 1e-12
    normalize::Bool = true
    maxiter::Int = 30
    tol::Float64 = 1e-15
    # S-step scheme, honoured by the `cbe_bond_update` path only. `s_iters` selects the solver
    # (see `cbe_bond_update`: 1 = one expansion + plain Lanczos, >1 = the outer-loop control,
    # <=0 = the expanding-basis Lanczos with `-s_iters` growth passes). `s_reorth` turns on full
    # reorthogonalisation in the plain Lanczos -- measured ~5.7x fewer matvecs at one bond, not
    # yet a default because it has not been checked across a sweep.
    s_iters::Int = 1
    s_reorth::Bool = false
    # PLAIN CBE instead of RSVD-CBE: the complete fused basis replaces the thin Gaussian, so
    # the preselection sees the exact `H*Theta` and its SVD returns the true leading directions
    # rather than an estimate. This FORMS THE RANK-4 OBJECT (cost `d*chi`, not `1.2*budget`),
    # which is precisely what the sketch exists to avoid -- it is the reference arm of the A/B,
    # not a better default. Makes `comp_ratio`, `dover`, `seed` and the RNG inert.
    exact::Bool = false
    seed::UInt = 0x5EED
    record_magnetisation::Bool = false
    observe::Union{Nothing, Function} = nothing
end

"""
    CBEBugRunInfo

Per-step record of a `cbe_lubich_bug!` run; every field has length `n_steps`.

`max_expanded` is the headline rank diagnostic and the one to compare against
`BondUpdateInfo.aug_k_dims`: it is the largest bond the basis sweep proposed BEFORE the
truncation, i.e. `r + Dex`, against the K/L augmentation's `2r`.

`krylov_dims` and `mean_aug` are the COST pair, and they are only useful together:
`krylov_dims` counts operator applications in the step's exponential(s) (summed over bonds in
the gate path, the single centre solve in the MPO path) and `mean_aug` is the size those
applications acted on. Wall time is roughly their product, so a setting can be slower either
by doing more iterations or by doing them on bigger objects -- and the budget A/B in
`benchmarks/cbe_budget_cost.jl` turns on telling those two apart.
"""
struct CBEBugRunInfo
    times::Vector{Float64}
    norms::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    expanded::Vector{Vector{Int}}
    max_expanded::Vector{Int}
    centre_ranks::Vector{Int}
    err_pre::Vector{Float64}
    err_fnl::Vector{Float64}
    discarded::Vector{Float64}
    magnetisations::Vector{Vector{Float64}}
    observations::Vector{Any}
    krylov_dims::Vector{Int}
    mean_aug::Vector{Float64}
    # EMPTY unless the caller asked for it. Imaginary time fills it (energy is the convergence
    # signal there); real time leaves it empty rather than paying for a full contraction per
    # step that nothing reads.
    energies::Vector{Float64}
end

CBEBugRunInfo(times, norms, bds, maxbd, expanded, maxexp, centres,
              epre, efnl, disc, mags, obs, kdim, maug) =
    CBEBugRunInfo(times, norms, bds, maxbd, expanded, maxexp, centres,
                  epre, efnl, disc, mags, obs, kdim, maug, Float64[])

Base.length(info::CBEBugRunInfo) = length(info.times)
# ══════════════════════════════════════════════════════════════════════════════════════
# PART II -- CBE AS THE BASIS UPDATE OF THE GATE-BASED bond_update_bug.
#
# Everything above is the SELECTION: sketch, preselect, rank, direct-sum. Everything below
# drives it from a bond frame and a gate, and solves the local problem in the expanded
# basis. The two were separate files until the three modes (real time, imaginary time,
# ground state) were split out into submodules; this file is what all three share.
#
# THE THREE MODES DIFFER IN EXACTLY TWO PLACES, and nowhere else:
#
#   1. WHICH OPERATOR the local problem is posed with -- `_expanding_krylov`'s `applyH`.
#   2. HOW THE TRIDIAGONAL IS REDUCED to an answer -- the `SStepSolver` below.
#
# Everything in between (the growth passes, the sketch, the P_perp projection, the ranking,
# the direct sum, the embedding of stored Krylov vectors into the grown basis, the full
# reorthogonalisation) is byte-for-byte the same code in all three. That is the point: the
# ground-state solver is a DROP-IN REPLACEMENT FOR THE S STEP, sharing the identical Krylov
# basis, not a second algorithm that happens to look similar.

# ── the S step: what to do with the Krylov space once it is built ─────────────

"""
    SStepSolver

How [`_expanding_krylov`](@ref) turns its tridiagonal into the local answer.

The Krylov space, and every growth pass that built it, is identical across subtypes. Only
the reduction differs, which is why the modes share one solver.
"""
abstract type SStepSolver end

"""
    Propagate(tau)

`exp(tau * H_eff)` applied to the current local core. `tau = -im*dt` is real time,
`tau = -dt` is imaginary time; the solver does not care which, and neither does anything
below it.
"""
struct Propagate <: SStepSolver
    tau::ComplexF64
end

"""
    LowestEigen()

The lowest Ritz vector of `H_eff` in the accumulated Krylov space -- a DMRG local solve.

REPLACES the exponential, keeps the basis. Returns a NORMALISED vector: unlike the
propagator, which scales the core by `beta0` to carry the state's weight, an eigenvector
carries no scale and the surrounding sweep is a variational minimisation of a unit-norm
state.

THE OPERATOR THIS IS GIVEN MUST INCLUDE THE ENVIRONMENTS. Handed the bare two-site gate it
would minimise `<h_{i,i+1}>` at each bond independently, whose fixed point is not the global
ground state -- each bond drives its own pair toward a local singlet and the next bond in
the sweep undoes it. DMRG converges because its local operator is `L + h + R`, so a local
minimisation lowers the TOTAL energy. See `apply_h_two_site` in `henv.jl`, and the header of
`CBEBugOptions` for why the gate path cannot supply this.
"""
struct LowestEigen <: SStepSolver end

"""
    _reduce_krylov(solver, alpha, betas, beta0) -> (coeffs, value)

`coeffs[k]` multiplies the `k`-th stored orthonormal Krylov vector to build the answer.
`value` is the Ritz value for [`LowestEigen`](@ref) and `NaN` when the solver has no
eigenvalue to report.
"""
_reduce_krylov(p::Propagate, alpha, betas, beta0) =
    (ComplexF64[beta0 * c for c in hermitian_tridiagonal_exp_coeffs(alpha, betas, p.tau)],
     NaN)

function _reduce_krylov(::LowestEigen, alpha, betas, beta0)
    # `beta0` is deliberately UNUSED: the propagator carries the state's norm through the
    # step, an eigensolve replaces the state outright and the Ritz vector is normalised.
    isempty(betas) && return (ComplexF64[1.0], alpha[1])
    F = eigen(SymTridiagonal(alpha, betas))
    return (ComplexF64.(@view F.vectors[:, 1]), F.values[1])
end
# ══════════════════════════════════════════════════════════════════════════════════════

# CBE as the basis update of the GATE-BASED bond_update_bug -- the validated path.
#
# WHY THIS FILE EXISTS, AND WHY IT COMES FIRST.
#
# `cbe_lubich.jl` puts CBE inside the Lubich/Sulz basis sweep: K-sweep, L-sweep, one centre
# Galerkin, an environment-dressed effective H. That sweep is the eventual target, but it means a failure has four
# possible homes -- the sweep, the channel environments, the CBE selection, or the placement
# of the single Galerkin step. The rank stall could not be pinned to any of them.
#
# `BondUpdateBUG.parity_sweep!` + `kls_bond_update` is the OTHER integrator, and it is
# validated: odd/even Trotter, one commuting bond group at a time, exact within a group,
# full-rank-exact, unitarity-locked. Dropping CBE into THAT removes three of the four
# suspects at once:
#
#   * no environments at all -- the Hamiltonian at a bond IS the gate, purely local, so
#     `henv.jl` and the whole stale/fresh-channel question is out of the picture;
#   * no sweep to get wrong -- `canonical!(psi, i)` before each bond, exactly as the
#     validated sweep does, so every frame is a true canonical snapshot;
#   * no Galerkin-placement question -- there is one Galerkin step per bond and it is the
#     step the validated integrator already takes.
#
# What is left is precisely the piece under suspicion: does the CBE selection admit the
# directions the dynamics needs? If rank grows here, the selection is sound and the fault in
# `cbe_lubich.jl` is in the sweep. If it stalls here too, the selection is the fault, and it is
# now isolated with nothing else to blame.
#
# The selection code is SHARED, not reimplemented: `cbe_expand_bond` calls the same
# `cbe_expand(f, skl, skr; ...)` core as the MPO path, differing only in the two sketch
# closures. So this file is a genuine test of the selection the Lubich sweep depends on.
#
# WHAT IT REPLACES IN `kls_bond_update`. Both of that kernel's basis mechanisms, together:
#
#   * the two K/L ODE solves (`expv` on `P_perp H_K`, non-Hermitian, Arnoldi). CBE supplies
#     the same DIRECTIONS from one sketched application of the gate, with no exponential;
#   * `missing_fill`, the per-sector random seed. The sector-graded sketch reaches an
#     empty-but-reachable sector THROUGH the gate, so what comes back is the direction the
#     dynamics wants rather than a random column the S-step SVD then discards.
#
# The S-step is unchanged and deliberately so -- it is the Galerkin step of the validated
# integrator, `S_hat0 = U_hat' Theta0 V_hat'` with no overlap matrices, Hermitian, Lanczos.

# ── the sketched gate action ──────────────────────────────────────────────────
#
# NEVER FORMS `H Theta`. The rank-4 two-site object is what 2-site TDVP builds and what this
# method promises not to allocate, and `cbe.jl`'s header spells out why sketching after the
# fact would be a strawman: with `H Theta` in hand its exact left SVD beats any sketch of
# it. So `Om` is folded into the gate FIRST -- `gate` is `d^4`, `Om` is thin, and the
# largest intermediate is `O(d^2 * g * r)`.
#
# The guard is `sketch_bond_left(f, gate, Om) == P_perp(contract(apply_gate(gate, Theta), Om'))` for
# arbitrary `Om`, against `apply_gate`, which is itself pinned to the analytic Heisenberg
# block element-for-element (`tests/BondUpdateBUG/test_gates.jl`).

"Check `gate` is tagged for the bond `f` describes, as `apply_gate` does for `theta`."
function _check_gate(f::BondFrame, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    (gate.inds[1].itags == tl && gate.inds[2].itags == tr) || throw(ArgumentError(
        "gate is tagged for ($(gate.inds[1].itags), $(gate.inds[2].itags)), " *
        "bond frame carries ($tl, $tr)"))
    return nothing
end

"""
    sketch_bond_left(f, gate, Om) -> TLArray

`Y = (gate Theta) Om†`, legs `(link_l, site_l, g)` -- the left matricization of the gate
action sketched from the right. `Om` carries `V0`'s layout `(g, site_r, link_r)`, i.e. it is
a candidate right frame with `g` columns instead of `r`; `Om = f.V0` recovers the exact left
factor of `gate Theta`.

Contraction order: absorb `Om` into the gate's right BRA leg, then close the gate's right
KET leg and `link_r` against `V0`, then `K0 = U0 S0`. Nothing rank-4 is built.
"""
function sketch_bond_left(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)       # (link_l, site_l, site_r, link_r)
    c  = contract(f.U0', (1, 2), HT, (1, 2))            # (bond, site_r, link_r)
    A  = to_concrete(HT - to_concrete(contract(f.U0, (3,), c, (1,))))
    return to_concrete(contract(A, (3, 4), Om', (2, 3)))          # (link_l, site_l, g)
end

"""
    sketch_bond_right(f, gate, Om) -> TLArray

Mirror of [`sketch_bond_left`](@ref): `Om` carries `U0`'s layout `(link_l, site_l, g)` and
the result is `(g, site_r, link_r)`.
"""
function sketch_bond_right(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    HT = apply_gate(gate, frame_theta(f), tl, tr)
    c  = contract(HT, (3, 4), f.V0', (2, 3))            # (link_l, site_l, bond)
    A  = to_concrete(HT - to_concrete(contract(c, (3,), f.V0, (1,))))
    Y  = contract(A, (1, 2), Om', (1, 2))               # (site_r, link_r, g)
    return to_concrete(permutedims(Y, (3, 1, 2)))       # (g, site_r, link_r)
end

"""
    cbe_expand_bond(f, gate; kwargs...) -> CBEExpansion

Controlled bond expansion of the frames at one bond, with the bare two-site gate standing in
for the environment-dressed effective Hamiltonian. Keywords are [`cbe_expand`](@ref)'s,
unchanged -- this is the same selection with different sketch closures.

The sketch probes the PROJECTOR (`P_perp (H Theta)`), which is the only path; there is no
`project_first` switch any more.
"""
cbe_expand_bond(f::BondFrame, gate; kwargs...) =
    cbe_expand(f,
               Om -> sketch_bond_left(f, gate, Om),
               Om -> sketch_bond_right(f, gate, Om); kwargs...)

# ── the bond update ──────────────────────────────────────────────────────────

"""
    cbe_bond_update(f::BondFrame, gate, tau; kwargs...) -> NamedTuple

One CBE-basis Galerkin update on the bond `f` snapshots -- a drop-in replacement for
`BondUpdateBUG.kls_bond_update` with the K/L solves and the random sector fill both replaced
by [`cbe_expand_bond`](@ref).

Returns the same fields `kls_bond_update` does (`left_core`, `right_core`, `U_aug`, `V_aug`,
`S_new`, `n_new_k`, `n_new_l`, `keep`, `aug_k`, `aug_l`, `svals`, `discarded`) so it slots
into the same sweep, plus `err_pre` / `err_fnl` from the expansion.

Keywords: `maxdim`, `trunc_thresh`, `s_tau`, `maxiter`, `tol` as in `kls_bond_update`;
`expand = false` freezes the frames for a fixed-rank projector step (the analogue of
`augment = false`); everything else forwards to `cbe_expand`.

`s_iters` selects the S-step, and its SIGN selects the scheme:

  - `s_iters = 1` (default) --- one expansion, then a plain Lanczos. The original path.
  - `s_iters = n > 1` --- repeat the expansion `n` times, each pass re-sketching from the
    state the last pass evolved. THE BEST SCHEME MEASURED once the budget is below the room:
    at `growth = 1.25` it takes the error from `1.449e-11` to `1.978e-15` (rank 9 -> 12). Flat
    at `growth = 2.0` only because the budget already saturates the room there and nothing is
    left to find. See the comment at the loop.
  - `s_iters = -g <= 0` --- the EXPANDING-BASIS LANCZOS of [`_expanding_krylov`](@ref), with
    `g` the number of leading Krylov steps that carry a growth pass (`s_iters = 0` grows
    none, which is the plain Lanczos with no expansion at all). The basis grows around each
    Krylov vector before `H` is applied to it, so the bond dimension runs `D`, `growth*D`,
    `growth^2*D`, ..., and the stopping signal is the Lanczos `beta` itself.

`iter_tol` applies only to the `s_iters > 1` control. Returned `n_iter` is the number of
growth passes performed and `iter_weight` is the final `beta` for the expanding solver, the
last new-direction weight for the control.

EXACTNESS AT FULL RANK. `U_ex` contains `U0` as its first block and `V_ex` contains `V0`, so
`S_hat0 = U_ex' Theta0 V_ex'` is lossless and `tau = 0` is the identity on the state. When
the expanded frames span the whole local space the Galerkin generator IS the gate, so the
step is the exact two-site propagator -- the same property the validated kernel has, and the
first thing the tests check.
"""
function cbe_bond_update(f::BondFrame, gate, tau::ComplexF64;
                         maxdim::Int = 200,
                         trunc_thresh::Float64 = 1e-14,
                         s_tau::Union{Nothing, ComplexF64} = nothing,
                         expand::Bool = true,
                         maxiter::Int = 30,
                         tol::Float64 = 1e-15,
                         s_iters::Int = 1,
                         iter_tol::Float64 = 1e-12,
                         s_reorth::Bool = false,
                         s_probe::Symbol = :residual,
                         rng::AbstractRNG = MersenneTwister(0x5EED),
                         expand_kwargs...)
    tau_s = s_tau === nothing ? tau : s_tau
    tl, tr = f.site_l.itags, f.site_r.itags
    theta0 = frame_theta(f)

    # COUNTED, not inferred. `expv` returns only the vector, and it belongs to the validated
    # module, so the iteration count is taken here instead: one increment per operator
    # application IS the Krylov dimension the Lanczos used. It is what distinguishes "this
    # setting does more work per step" from "this setting needs more steps of work", and the
    # budget A/B needs that distinction -- a starved basis can be slower while building a
    # smaller expansion, which no timing alone can attribute.
    nmv = Ref(0)

    # The two closures that are all a mode gets to change. The gate path supplies the bare
    # two-site gate here (its environments are the identity by canonical form -- see
    # `CBEBugOptions`); the MPO path supplies the environment-dressed `apply_h_two_site`.
    applyH = th -> apply_gate(gate, th, tl, tr)
    expand_at = fr -> cbe_expand_bond(fr, gate; rng = rng, expand_kwargs...)

    # The Galerkin step in a GIVEN basis. Note `S_start` is rebuilt from `theta0` every time it
    # is called -- see the loop below for why that is the whole correctness argument.
    function galerkin(U_aug, V_aug)
        S_start = to_concrete(contract(contract(U_aug', (1, 2), theta0, (1, 2)),
                                       (2, 3), V_aug', (2, 3)))
        function apply_s(x)
            nmv[] += 1
            theta = to_concrete((U_aug * x) * V_aug)
            evolved = apply_gate(gate, theta, tl, tr)
            proj = contract(U_aug', (1, 2), evolved, (1, 2))
            return to_concrete(contract(proj, (2, 3), V_aug', (2, 3)))
        end
        return expv(apply_s, tau_s, S_start; hermitian = true, maxiter = maxiter,
                    tol = tol, reorth = s_reorth)
    end

    U_aug, V_aug = f.U0, f.V0
    err_pre, err_fnl, n_iter, iter_weight = 0.0, 0.0, 0, 0.0
    local S_new

    if !expand
        S_new = galerkin(U_aug, V_aug)
    elseif s_iters <= 0
        # ── GROWTH INSIDE THE LANCZOS ────────────────────────────────────────────────
        # `s_iters <= 0` selects the expanding-basis solver; see `_expanding_krylov`.
        U_aug, V_aug, S_new, info = _expanding_krylov(
            f, applyH, expand_at, Propagate(tau_s), theta0, nmv;
            maxiter = maxiter, tol = tol, grow_iters = -s_iters, probe = s_probe)
        err_pre, err_fnl = info.err_pre, info.err_fnl
        n_iter, iter_weight = info.n_grow, info.beta_final
    else
        # ── ONE EXPANSION, THEN A PLAIN LANCZOS (the default) ────────────────────────
        # `s_iters = 1` is the original single expansion and is the default path; `n > 1`
        # repeats it, re-sketching each pass from the state the previous pass evolved.
        #
        # THIS IS THE BEST SCHEME MEASURED, and the reason is NOT that it builds a Krylov
        # space -- it does not. With tau small, exp(tau*H)Theta0 = Theta0 + tau*H*Theta0 +
        # O(tau^2), so each pass re-derives H*Theta0, never H^2*Theta0. What it does instead
        # is COMPOUND THE GROWTH SCHEDULE: the budget is recomputed from the CURRENT rank
        # every pass, so r -> growth*r -> growth^2*r, and each step of that ladder is probed
        # by a sketch whose width is matched to the rank it is probing. A sequence of
        # well-conditioned narrow probes beats one wide draw.
        #
        # MEASURED at L=8 bond (4,5) against the exact two-site propagator, growth = 1.25:
        # one-shot 1.449e-11 at rank 9, three passes 1.978e-15 at rank 12. The expanding
        # Lanczos reaches the SAME rank 12 and stays at 1.450e-11 -- same basis size, four
        # orders worse, so what differs is which directions get admitted and not how many.
        #
        # It is flat at growth = 2.0 only because the budget already saturates the room there
        # (room = d*chi - r -> r, budget -> r for d = 2), so the first pass takes the whole
        # local space and there is nothing left for a second one to find.
        #
        # `galerkin` always rebuilds S_start from the ORIGINAL theta0, never from the previous
        # pass's S_new, so exp(tau*H_eff) is applied exactly once however many passes run.
        probe = f
        for k in 1:s_iters
            ex = expand_at(probe)
            U_prev, V_prev = U_aug, V_aug
            U_aug, V_aug = ex.U_ex, ex.V_ex
            err_pre, err_fnl, n_iter = ex.err_pre, ex.err_fnl, k

            S_new = galerkin(U_aug, V_aug)

            k == s_iters && break
            (ex.n_new_l + ex.n_new_r) == 0 && break
            iter_weight = _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S_new)
            iter_weight < iter_tol && break

            probe = _reframe(f, U_aug, S_new, V_aug)
        end
    end

    res = svd(S_new, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true)
    tag = f.link_mid.itags
    left_core  = to_concrete(setitag(to_concrete(U_aug * res.U), 3, tag))
    right_core = to_concrete(setitag(to_concrete((res.S * res.Vd) * V_aug), 1, tag))

    svals = [s for (s, _, _, _) in res.kept_list]
    return (; left_core, right_core, U_aug, V_aug, S_new,
            n_new_k = leg_dim(U_aug, 3) - f.old_rank,
            n_new_l = leg_dim(V_aug, 1) - f.old_rank,
            keep = leg_dim(left_core, 3),
            aug_k = leg_dim(U_aug, 3), aug_l = leg_dim(V_aug, 1),
            svals, discarded = _trunc_weight(res),
            err_pre, err_fnl, n_matvec = nmv[], n_iter, iter_weight)
end

"""
    _expanding_krylov(f, applyH, expand, solver, theta0, nmv; ...) -> (U, V, S, info)

Lanczos for `exp(tau*H_eff)` whose BASIS GROWS WITH THE KRYLOV SPACE.

The expansion is not a preprocessing step here, it is step 0 of each Lanczos iteration: before
applying `H` to the current Krylov vector `v_k`, the frames are expanded using `v_k` itself as
the probe, so the enlarged basis can hold `H*v_k`. The bond dimension therefore grows
geometrically with the Krylov dimension --- `D`, `growth*D`, `growth^2*D`, ... --- until `room`
or `grow_iters` stops it.

WHY THIS AND NOT AN OUTER LOOP. Re-running a converged exponential and re-sketching from its
result asks for the same subspace every time: `exp(tau*H)Theta = Theta + tau*H*Theta + O(tau^2)`,
so the sketch keeps rediscovering `H*Theta`. Inside the Lanczos, `v_k` genuinely carries `H^k`,
so each growth pass is probing a direction that did not exist before. That is what makes the
basis converge on the relevant subspace rather than having to be tuned onto it by `comp_ratio`
and the oversampling.

STOPPING is the Lanczos beta itself --- literally the small value in the Krylov matrix. When
`beta_k < tol` the Krylov space has closed and no further growth can help.

TWO PROPERTIES WORTH BEING PRECISE ABOUT, because they are approximations and not identities:

  - `alpha_k` is EXACT. `v_k` lies in the range of the projector `P_k` in force at step `k`, so
    `<v_k, P_k H P_k v_k> = <v_k, H v_k>` regardless of how small that basis is.
  - `beta_k` is NOT exact: it measures the residual of `P_k H v_k`, missing whatever part of
    `H v_k` falls outside the step-`k` basis. Making that part small is precisely the growth
    pass's job, which is why the expansion runs BEFORE the operator is applied and not after.
    The resulting tridiagonal is therefore a Galerkin approximation built across a nested
    sequence of subspaces, not the exact tridiagonal of a fixed operator.

FULL REORTHOGONALISATION IS MANDATORY here, not the optional `reorth` of `lanczos_expv`. The
three-term recurrence keeps the basis orthogonal only for a FIXED operator; the projected
operator changes at every growth pass, so earlier vectors stop being exact Krylov vectors of the
current one and the recurrence alone would drift.
"""
function _expanding_krylov(f::BondFrame, applyH, expand, solver::SStepSolver, theta0, nmv;
                           maxiter::Int, tol::Float64, grow_iters::Int,
                           probe::Symbol = :residual)
    project(U, V, th) = to_concrete(contract(contract(U', (1, 2), th, (1, 2)),
                                             (2, 3), V', (2, 3)))

    # Embed a core from the (Uo,Vo) basis into (Un,Vn). `oplus` keeps the old block first so
    # this is `[S; 0]`, but it is COMPUTED as an overlap rather than assumed, which keeps it
    # correct if the orthonormality guard rotates within the appended span.
    #
    # THE OVERLAPS ARE BUILT ONCE PER GROWTH PASS, NOT ONCE PER VECTOR. `PL` and `PR` depend
    # only on the frames, so computing them inside a per-vector `embed` -- as the first version
    # did -- repeated two `O(d*chi*r_old*r_new)` contractions for every stored Krylov vector.
    # With a Krylov dimension of ~30 that is ~30x the frame work per pass, and it grows with
    # chi. Hoisted here; the per-vector cost is now the two `O(chi^3)` core contractions alone.
    embed_ops(Uo, Vo, Un, Vn) =
        (to_concrete(contract(Un', (1, 2), Uo, (1, 2))),         # PL (new_l, old_l)
         to_concrete(contract(Vo, (2, 3), Vn', (2, 3))))         # PR (old_r, new_r)

    embed_with(PL, PR, S) =
        to_concrete(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))

    U, V = f.U0, f.V0
    S0 = project(U, V, theta0)
    beta0 = norm(S0)
    err_pre, err_fnl, n_grow = 0.0, 0.0, 0
    # `value = NaN` is NOT decoration: every other exit carries it (the Ritz value for
    # `LowestEigen`), and a caller that reads `info.value` -- the ground-state solve does --
    # would hit an UndefVarError on this branch if the field were simply absent here.
    beta0 == 0 && return (U, V, S0,
                          (; err_pre, err_fnl, n_grow, beta_final = 0.0, ranks = Int[],
                             value = NaN))

    basis = Any[to_concrete((1.0 / beta0) * S0)]
    alpha, betas, ranks = Float64[], Float64[], Int[leg_dim(U, 3)]
    b_last = 0.0

    image = nothing        # the PREVIOUS iteration's output, H*v_{k-1}, before orthogonalisation

    for it in 1:maxiter
        # ---- step 0: grow the basis around the probe ----
        #
        # WHICH VECTOR DRIVES THE EXPANSION IS THE WHOLE QUESTION, and the two choices are not
        # close numerically:
        #
        #   :residual  (DEFAULT) the orthonormal Lanczos vector `basis[it]`, i.e. the Krylov
        #              vector itself as the centre gauge. `Theta = U*v_k*V` is then in mixed
        #              bond-canonical form -- VERIFIED, `U'U - I` and `VV' - I` both 1.5e-15
        #              and `||S|| = ||U S V|| = 1` at every pass.
        #   :image     the previous step's output `H*v_{k-1}`, before the Lanczos subtraction.
        #              TRIED AND IT MAKES NO DIFFERENCE: at growth = 1.25 both give 1.450e-11,
        #              with :image at rank 11 against :residual's 12. Kept only so the null
        #              result stays reproducible; it is not a knob worth turning.
        #
        # NEITHER IS THE RANDOM GAUSSIAN. The chosen vector becomes the CORE `S0` of the frame
        # handed to `cbe_expand_bond`, which then draws its own sector-graded Gaussian inside
        # `sector_graded_sketch` exactly as on the first pass. Every other step -- preselection,
        # the P_perp projection, the ranking, the direct sum -- is unchanged.
        #
        # The Lanczos RECURRENCE is untouched either way: `H` is still applied to the
        # orthonormal `basis[it]`, so alpha/beta and the tridiagonal are unchanged. Only the
        # basis the next vector is built IN differs.
        #
        # Why H is applied to the ORTHONORMALISED vector and not to the raw output: measured,
        # `H*v_k` overlaps an EARLIER basis vector by 100% of its own norm (k = 2 and 3 at the
        # L=8 test bond), so without the beta subtraction the "next Krylov vector" is
        # numerically the previous one and the iteration stalls. The alpha subtraction, by
        # contrast, IS redundant here: `<v_k, H v_k>` is 1e-15 to 1e-20 on this Hamiltonian.
        if it <= grow_iters
            pv = (probe === :image && image !== nothing) ? image : basis[it]
            ex = expand(_reframe(f, U, pv, V))
            if (ex.n_new_l + ex.n_new_r) > 0
                Uo, Vo = U, V
                U, V = ex.U_ex, ex.V_ex
                PL, PR = embed_ops(Uo, Vo, U, V)
                for k in eachindex(basis)
                    basis[k] = embed_with(PL, PR, basis[k])
                end
                image === nothing || (image = embed_with(PL, PR, image))
                err_pre, err_fnl, n_grow = ex.err_pre, ex.err_fnl, n_grow + 1
            end
        end
        push!(ranks, leg_dim(U, 3))

        nmv[] += 1
        theta = to_concrete((U * basis[it]) * V)
        w = project(U, V, applyH(theta))
        image = w          # this iteration's output, carried to the next expansion

        a = real(tensor_inner(basis[it], w))
        push!(alpha, a)
        w = to_concrete(w + (-a) * basis[it])
        it > 1 && (w = to_concrete(w + (-betas[it - 1]) * basis[it - 1]))
        for _pass in 1:2, u in basis
            w = to_concrete(w - tensor_inner(u, w) * u)
        end

        # `b_last` is the TERMINATING residual, not the last accepted one. Reporting
        # `betas[end]` instead would show the last beta that was large enough to continue --
        # the opposite of the stopping signal it is meant to document.
        b = norm(w)
        b_last = b
        b < tol && break
        it == maxiter && break
        push!(betas, b)
        push!(basis, to_concrete((1.0 / b) * w))
    end

    # THE ONLY MODE-DEPENDENT LINE IN THE WHOLE FUNCTION. `coeff` already carries `beta0` for
    # the propagator and deliberately does not for the eigensolve, so the accumulation below
    # is identical either way.
    coeff, value = _reduce_krylov(solver, alpha, betas, beta0)
    S = to_concrete(coeff[1] * basis[1])
    for k in 2:length(coeff)
        S = to_concrete(S + coeff[k] * basis[k])
    end
    return (U, V, S, (; err_pre, err_fnl, n_grow, beta_final = b_last, ranks, value))
end

"""
    _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S) -> Float64

Fraction of the evolved core's norm living on directions the LAST pass added.

`oplus` puts the old block first, so `U_prev` embeds in `U_aug` and `U_prev' U_aug` is the
projector back onto the old space --- `[I | 0]` in exact arithmetic, and computed rather than
assumed so the measure survives the re-orthonormalisation rotating within the span.

Returns `sqrt(‖S‖² - ‖P S Q‖²)/‖S‖`. Small means the pass produced directions the step puts
no amplitude on, i.e. the Krylov space has stopped delivering, which is the signal to stop.
Everything here is `O(χ²)`: no rank-4 tensor is formed to measure convergence.
"""
function _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S)
    n = norm(S)
    n == 0 && return 0.0
    PL = to_concrete(contract(U_prev', (1, 2), U_aug, (1, 2)))     # (prev_l, new_l)
    PR = to_concrete(contract(V_aug, (2, 3), V_prev', (2, 3)))     # (new_r, prev_r)
    S_old = to_concrete(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))
    return sqrt(max(n^2 - norm(S_old)^2, 0.0)) / n
end

"""
    _reframe(f, U, S, V) -> BondFrame

`f` with its frames and core replaced, so the next sketch probes `H` against the state the
last pass evolved instead of the state the step began with.

`old_rank` is set to the LARGER of the two frame ranks. The two sides can end a pass at
different widths (`n_new_l ≠ n_new_r` is normal --- an edge bond routinely expands on one side
only), and `cbe_expand` carries a single `r` which it uses for `room = dχ - r` and for the
growth schedule. Taking the max makes the computed room a lower bound on the true room per
side, so the next pass can under-ask but never over-ask, and `room` stays the hard ceiling it
is meant to be.
"""
_reframe(f::BondFrame, U, S, V) =
    BondFrame(U, S, V, f.link_l, f.site_l, f.link_mid, f.site_r, f.link_r,
              max(leg_dim(U, 3), leg_dim(V, 1)), f.link_mid_space)

cbe_bond_update(f::BondFrame, gate, tau::Number; kwargs...) =
    cbe_bond_update(f, gate, ComplexF64(tau); kwargs...)

# ── the sweep ────────────────────────────────────────────────────────────────
