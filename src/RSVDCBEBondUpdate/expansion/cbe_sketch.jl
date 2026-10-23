# cbe_sketch.jl: the sector-graded Gaussian sketch, preselection and final selection (from `cbe_core.jl`, included right after it).

"""
    sector_graded_sketch(frame, side, npre; comp_ratio=0.5, rng, tag) -> TLArray or nothing

The CBE sketch block for one side of a bond, in the frame's own layout:
`side = :left` gives `(link_l, site_l, g)` like `U0`, `side = :right` gives
`(g, site_r, link_r)` like `V0`.

`npre` is the total column budget `Dpre`. Returns `nothing` only when neither half can be
built at all.

`boost` (sector => extra columns) is Test B's lever: it widens the probe only where the
holdout found range missing (see [`_certified_range`](@ref)). Extra columns go to the
complement half first, then to the isometry half of the same sector; neither ever exceeds the
sector's dimension.

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
                              tag::AbstractString = "cbeG",
                              boost = nothing)
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
    dfull, dT = _frame_sector_dims(frame, side, F, "sector_graded_sketch")
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
        cols = _alloc_columns(dfull, npre)
        boost === nothing || _boost!(cols, boost, dfull)
        R = _trim_per_sector(_randomize_like(F, rng), 3, cols)
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
    if boost !== nothing
        # A sector whose complement is already fully drawn can still hide range behind the
        # frame's own span, so the remainder goes to the isometry half rather than vanishing.
        _boost!(ct, _boost!(cc, boost, dC), dT)
    end

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
        C = perp_of(frame, tolayout(RC))
        norm(C) > 0 && push!(blocks, C)
    end

    # Isometry half: random, then the frame projected IN. `R - perp(R)` is the projection
    # onto the frame's span, built from the same primitive so the two halves can never
    # disagree about what "the frame's span" is.
    RT = _trim_per_sector(_randomize_like(F, rng), 3, ct)
    if RT !== nothing
        R = tolayout(RT)
        T = to_concrete(R - perp_of(frame, R))
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

"Degeneracy-weighted squared singular weight of a Telum SVD list, `Σ σ² · deg`."
_sq_weight(list) = sum(Float64(s)^2 * Float64(deg) for (s, deg, _, _) in list; init = 0.0)

"Relative weight discarded by an SVD, `sqrt(Σcut σ² / Σall σ²)`."
function _trunc_weight(res)
    kept, cut = _sq_weight(res.kept_list), _sq_weight(res.trunc_list)
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
    _rank_cut(stol, Y, n) -> Float64

The preselection SVD's relative cutoff. `stol > 0` is the legacy absolute-on-normalised cut
(`stol_pre`). `stol <= 0` is the adaptive path's NUMERICAL RANK test (plan §1: `stol_pre` is
replaced by machine-precision rank detection; physical importance is Tests A/B's job).

THE FLOOR IS MEASURED AGAINST `Y`, NOT `R`. `R = P_perp Y` is a difference, and when most of
`Y` lies in the frame the subtraction cancels: rounding of order `eps * ||Y||` survives in an `R`
of size `n`, i.e. relative noise `eps * ||Y|| / n` after normalising. A cut fixed at a few `eps`
would admit that noise as directions whenever `n << ||Y||`; one fixed at 1e-10 drops real
directions whenever `n ~ ||Y||`. `_CBE_EPS` (100 eps) is the same margin the empty-residual test
above already uses.
"""
_rank_cut(stol::Float64, Y, n::Real) =
    stol > 0 ? stol : _CBE_EPS * max(norm(Y), n) / n

"""
    _preselect_left(U0, Y, stol) -> (Q, err) or (nothing, 0.0)

Orthonormal basis of the part of the sketched action `Y` lying outside `U0`, with the
`RepairOrtho` second pass. `Y` and `Q` carry `(link_l, site_l, bond)`.

`Y` is normalised before the SVD because `cutoff` is relative to the largest singular
value: the new directions are O(tau)-small in absolute terms, and thresholding relative to
their own scale is what keeps a meaningful tolerance (the trap recorded in `augment.jl`).
"""
function _preselect_left(U0, Y, stol::Float64)
    R = perp_component(U0, Y)
    n = norm(R)
    n <= _CBE_EPS * max(norm(Y), 1.0) && return (nothing, 0.0)
    res = svd(to_concrete((1.0 / n) * R), (1, 2), _TAG_L, _TAG_LR;
              cutoff = _rank_cut(stol, Y, n), get_lists = true)
    Q = to_concrete(res.U)
    R2 = perp_component(U0, Q)            # RepairOrtho
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
    R = perp_component_right(V0, Y)
    n = norm(R)
    n <= _CBE_EPS * max(norm(Y), 1.0) && return (nothing, 0.0)
    res = svd(to_concrete((1.0 / n) * R), (2, 3), _TAG_R, _TAG_RR;
              cutoff = _rank_cut(stol, Y, n), get_lists = true)
    Q = to_concrete(permutedims(res.U, (3, 1, 2)))
    R2 = perp_component_right(V0, Q)      # RepairOrtho
    norm(R2) <= _CBE_EPS * norm(Q) && return (nothing, 0.0)
    Q = to_concrete(permutedims(svd(R2, (2, 3), _TAG_R, _TAG_RR; cutoff = 1e-13).U,
                                (3, 1, 2)))
    return (Q, _trunc_weight(res))
end

"""
    _range_find(draw, wleg, select, qleg, w0, wcap, ncap) -> (Q, err, n)

Randomised range finder for one side of the preselection: widen the probe until the
preselection SVD CUTS something, then hand back its orthonormal basis `Q`, the weight it
discarded, and `n = leg_dim(Q, qleg)` -- the number of directions that cleared the
tolerance.

`draw(w)` returns a probe of requested width `w` (or `nothing`), with the probe's own width
on leg `wleg`; `select(Y)` is the preselection, returning `(Q, err)`.

WHAT THE STOPPING TEST MEANS, AND WHY IT NEEDS NO TOLERANCE OF ITS OWN. `select` cuts at
`stol_pre`, so a returned `n` STRICTLY BELOW the delivered width is the sketched range
having run past the decay of `P_perp (H Theta)` -- there were directions to rank and the
tolerance ranked some of them out. `n == w` is the opposite reading: the probe is saturated,
every direction it drew still carried weight, so the range was CLIPPED by the width and a
wider probe would find more. That is exactly the question a fixed oversample answers by
guessing, and it is answered here by looking.

⚠️ `ncap` IS WHAT KEEPS THIS AN RSVD. Without it a probe whose spectrum never decays --
a bond near full rank, where every drawn direction carries weight -- would double all the way
to `wcap`, i.e. to the complete fused basis, which is precisely the cost the sketch exists to
avoid. `ncap` is the side's ADMISSION limit (`room`, or an explicit `dex`), and once `n`
reaches it a wider probe cannot admit one further direction: the surplus is trimmed by
`_trim_total` either way. So it is a bound on pointless work, not a tolerance.

The loop only ever widens, by doubling, and also stops at `wcap` (the opposite frame's fused
dimension -- see the call site) or as soon as the probe delivers fewer columns than asked,
which means the local space is exhausted and there is nothing left to widen into. So it
terminates in at most `log2(wcap)` passes, and at an explicit `dover` (`wcap == w0`) it is
one pass and the old behaviour exactly.
"""
function _range_find(draw, wleg::Int, select, qleg::Int, w0::Int, wcap::Int, ncap::Int)
    w = clamp(w0, 1, max(wcap, 1))
    while true
        Om = draw(w)
        Om === nothing && return (nothing, 0.0, 0)
        wd = leg_dim(Om, wleg)
        Q, err = select(Om)
        Q === nothing && return (nothing, 0.0, 0)
        n = leg_dim(Q, qleg)
        (n < wd || n >= ncap || wd < w || w >= wcap) && return (Q, err, n)
        w = min(2 * w, wcap)
    end
end

"""
    _tail_rank(list, scale, allowance, kmax; base = 0.0) -> (k, tail)

TEST A, the tail test: the SMALLEST `k <= kmax` whose discarded tail
`scale * sqrt(base + sum_{j>k} sigma_j^2 * deg_j)` still fits inside `allowance`, together
with the indicator that tail achieves.

`list` is a Telum `kept_list`, `(sigma, degeneracy, ...)` sorted descending. The DEGENERACY
FACTOR IS NOT COSMETIC: under SU(2) one entry stands for `deg` states, and ranking on
`sigma` alone would under-count a multiplet's weight by exactly that factor.

`base` is squared weight the SVD ALREADY discarded (its `trunc_list`). Omitting it would
measure the tail of the surviving list against an allowance meant for the whole spectrum,
and so report a rank as certified when the weight below the SVD's own cut already breached
the budget.

The tail is ABSOLUTE, not relative to the leading singular value. `scale` has already
divided by the state norm (see `basis_allocation`), so the comparison is against a
tolerance on the step's error, which is the quantity the caller actually budgeted. A
relative cut cannot do this: it answers "is this direction small compared to its loudest
neighbour", which at a bond whose whole spectrum is tiny keeps rank that buys no accuracy,
and at a bond with one dominant direction throws away rank that does.

Returns `kmax` when even the full tail does not fit -- the allowance is then unmeetable at
this width, and the caller records `tolerance_met = false` rather than silently believing it.
"""
function _tail_rank(list, scale::Float64, allowance::Float64, kmax::Int; base::Float64 = 0.0)
    n = length(list)
    kmax = min(kmax, n)
    kmax <= 0 && return (0, scale * sqrt(base + _sq_weight(list)))
    w = [Float64(s)^2 * Float64(deg) for (s, deg, _, _) in list]
    # Weight below `kmax` is discarded whatever `k` we choose, so it belongs to the floor.
    tail = base + sum(w[(kmax + 1):n]; init = 0.0)
    k = kmax
    while k > 0
        t = tail + w[k]
        scale * sqrt(t) <= allowance || break
        tail = t
        k -= 1
    end
    return (k, scale * sqrt(tail))
end

"""
    _sample_residual(Q, project, sketch, draw, nh, wleg, side) -> (est, per_sector, cols)

TEST B, the sampling test: a matrix-free estimate of `||(I - QQ†) P_perp (H Theta)||_F`
from HOLDOUT probes that took no part in building `Q`.

The independence is the whole content of the test. The probe that built `Q` is, by
construction, inside its range -- re-measuring with it reports zero residual no matter how
much range was missed, which is why a range finder cannot certify itself from its own
sketch.

⛔ THE NORMALISATION IS PER SECTOR, because the holdout is sector-graded. A column drawn in
sector `q` is Gaussian on that sector alone, so `E||A Om_q||_F^2 = n_q ||A_q||_F^2` and the
unbiased estimate of `||A||_F^2` is `sum_q ||A Om_q||_F^2 / n_q`. Dividing the pooled sum by the
total width instead weights every sector by its SHARE OF THE COLUMNS, and the `ceil`
allocation hands a one-dimensional empty-but-reachable sector one column out of `w` -- so a
range missed there was reported `sqrt(w)` times too small, in exactly the sector the
expansion exists to find.

`project` applies the frame projector `P_perp` (the closure the preselection used), `sketch`
applies `H Theta` to a probe, `draw(w)` returns a probe of width `w` or `nothing`, `wleg` is
the probe's width leg, and `side` (`:left`/`:right`) fixes the frame layout and so which
projector removes `Q`.

Returns the estimate, each sector's SQUARED contribution (they sum to `est^2`; this is what
[`_certified_range`](@ref) targets), and the holdout columns per sector. A zero estimate means
no probe could be drawn or the sketched action vanished: no range left to miss.
"""
function _sample_residual(Q, project, sketch, draw, nh::Int, wleg::Int, side::Symbol)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    per = Dict{Any, Float64}()
    cols = Dict{Any, Int}()
    nh >= 1 || return (0.0, per, cols)
    Om = draw(nh)
    Om === nothing && return (0.0, per, cols)
    for (q, d) in Om.spaces[wleg]
        d > 0 && (cols[q] = d)
    end
    isempty(cols) && return (0.0, per, cols)
    Y = sketch(Om)
    Y === nothing && return (0.0, per, cols)
    R = to_concrete(project(Y))
    norm(R) <= _CBE_EPS && return (0.0, per, cols)
    if Q !== nothing
        off = side === :left ? perp_component(Q, R) : perp_component_right(Q, R)
        # `project` is reapplied so the estimate cannot be polluted by a component along the
        # frame, which `Q` is not meant to cover.
        R = to_concrete(project(to_concrete(off)))
    end
    # The sketch contracts the probe's local legs and leaves its width leg free, so the
    # residual's width sectors ARE the probe's sectors, label for label.
    rleg = side === :left ? 3 : 1
    for (q, d) in R.spaces[rleg]
        n_q = get(cols, q, 0)
        n_q > 0 || error("Test B: residual sector $q has no holdout column -- the sketch " *
                         "did not preserve the probe's width labels, so the per-sector " *
                         "normalisation would be wrong")
        Rq = length(R.spaces[rleg]) == 1 ? R :
             to_concrete(getsub(R, rleg, s -> s == q ? (1:d) : nothing))
        per[q] = norm(Rq)^2 / n_q
    end
    return (sqrt(sum(values(per); init = 0.0)), per, cols)
end

"""
    _target_sectors!(boost, per, Om, wleg, alloc) -> Int

Decide where Test B widens next: every sector whose own share of the missed residual exceeds
a fair `1/S` split of the allowance (`S` = sectors the holdout probed) gets its sketch columns
doubled in `boost`. Returns how many sectors were targeted.

THE PLAN'S `dover`/`comp_ratio` ROLE, MADE SPECIFIC. A uniform doubling spends its new
columns in proportion to sector size, i.e. mostly where the range was ALREADY found; the
holdout says which sector is short, so that is where the columns go. Whenever the total
fails, at least one sector is over its fair share, so a failing test always targets
something.
"""
function _target_sectors!(boost::Dict, per::Dict, Om, wleg::Int, alloc::BasisAllocation)
    isempty(per) && return 0
    have = Om === nothing ? Dict{Any, Int}() : Dict{Any, Int}(q => d for (q, d) in Om.spaces[wleg])
    share = alloc.sample^2 / length(per)
    n = 0
    for (q, s) in per
        alloc.scale^2 * s > share || continue
        boost[q] = get(boost, q, 0) + max(get(have, q, 0), 1)
        n += 1
    end
    return n
end

"""
    _one_sided_tail(M, alloc, kmax, stol, ledger) -> U or nothing

Test A on one side's candidates. `M` carries the candidate leg first (the conjugate of the
side's bond leg) and the opposite local space second; its singular values are the candidates'
weights. Returns the left singular vectors of the smallest rank the tail test allows (the
coefficients that rotate the candidates onto their ranked directions), or `nothing` if no
direction is needed. Weight cut by `kmax` counts toward the tail, as in the joint branch.
"""
function _one_sided_tail(M, alloc::BasisAllocation, kmax::Int, stol::Float64,
                         ledger::Union{Nothing, AdaptiveLedger})
    (length(M.qlabels) == 0 || norm(M) == 0) && return nothing
    res = svd(M, (1,); Nkeep = kmax, cutoff = stol, get_lists = true)
    nk = length(res.kept_list)
    k, tail = _tail_rank(res.kept_list, alloc.scale, alloc.tail, nk;
                         base = _sq_weight(res.trunc_list))
    if ledger !== nothing
        ledger.max_tail = max(ledger.max_tail, tail)
        tail <= alloc.tail || _unmet!(ledger, :tail)
    end
    k == 0 && return nothing
    k < nk && (res = svd(M, (1,); Nkeep = k, cutoff = stol))
    return to_concrete(res.U)
end

"""
    _record_sample!(ledger, side, ind, per, hcols, sketch, wleg, alloc)

Write one Test B verdict into the ledger: the per-side maximum, and a `sample_log` entry with
the per-sector indicators and a bound on the estimate's relative sampling error.

THE UNCERTAINTY BOUND. For a complex Gaussian column `||A w||^2 = sum_i s_i^2 |z_i|^2` with
`|z_i|^2 ~ Exp(1)`, so its variance is `sum_i s_i^4 <= ||A||_F^4`. Averaging `n_q` columns gives
`var(per_q) <= per_q^2 / n_q`, and the relative standard deviation of `est = sqrt(sum per_q)` is
half that of its square: `0.5 * sqrt(sum_q per_q^2 / n_q) / sum_q per_q`. An upper bound, tight
for a rank-1 miss -- the case that matters, a single missed direction.
"""
function _record_sample!(ledger::AdaptiveLedger, side::Symbol, ind::Float64, per::Dict,
                         hcols::Dict, sketch, wleg::Int, alloc::BasisAllocation)
    ledger.max_sample = max(ledger.max_sample, ind)
    if side === :left
        ledger.max_sample_l = max(ledger.max_sample_l, ind)
    else
        ledger.max_sample_r = max(ledger.max_sample_r, ind)
    end
    tot = sum(values(per); init = 0.0)
    relunc = tot > 0 ? 0.5 * sqrt(sum(s^2 / hcols[q] for (q, s) in per; init = 0.0)) / tot : 0.0
    push!(ledger.sample_log,
          (side = side, est = ind, relunc = relunc,
           per = Dict{Any, Float64}(q => alloc.scale * sqrt(s) for (q, s) in per),
           holdout = copy(hcols),
           sketch = sketch === nothing ? Dict{Any, Int}() :
                    Dict{Any, Int}(q => d for (q, d) in sketch.spaces[wleg])))
    return ledger
end
