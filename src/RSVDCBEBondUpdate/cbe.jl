# The sketched H-action: `(H Theta) Omega` without ever forming `H Theta`.
#
# WHY IT MUST BE SKETCHED, NOT SKETCHED-AFTER-THE-FACT. The randomized range finder of
# `RSVDpreBE0SiQS.m` earns its keep only if the rank-4 two-site object is never built:
# with `H Theta` in hand you would take its exact left SVD and beat any sketch of it, in
# both accuracy and cost. So `Omega` has to be folded in DURING the contraction --
# `contractHPsi_R2L` (`RSVDpreBE0SiQS.m:826`) orders its network exactly so, putting the
# sketch next to the right environment first. Building `H Theta` and contracting `Omega`
# onto it afterwards would be a strawman that could never show the cost win.
#
# THE DECOMPOSITION THAT MAKES IT UNIFORM. Every one of the five contributions to `H` at
# a bond (see `apply_h_two_site`) is a product of something acting on the LEFT legs and
# something acting on the RIGHT legs, optionally sharing one op-leg:
#
#     Y[link_l, site_l, g]  =  Σ_bond,op  Kpart[link_l, site_l, bond, op]
#                                        · Vpart[bond, g, op]
#
#   Kpart : the left operator applied to  K0 = U0 S0     (link_l, site_l, bond [, op])
#   Vpart : the right operator applied to V0, then contracted with `Omega`
#                                                          (bond, g [, op])
#
# Nothing bigger than `K0` or `V0` is ever formed, and the sketch dimension `g = Dpre`
# replaces `site_r ⊗ link_r` at the first opportunity. `K0` and `V0` both carry their site
# leg at position 2, the same layout an MPS tensor has, so `_apply_site_op` applies to
# them unchanged.
#
# The guard is `sketch_h_left(..., Om) == contract(apply_h_two_site(Theta, ...), Om')`
# for arbitrary `Om`, against the stage-1 code that is itself pinned to the dense matrix
# element. Nothing here is trusted on inspection.

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

# ── the two halves of every contribution ─────────────────────────────────────

"`Vpart` for a right operator that is the identity: `V0` contracted with the sketch."
_vpart_plain(V0, Om) = to_concrete(contract(V0, (2, 3), Om', (2, 3)))       # (bond, g)

"`Kpart` for the right sketch: `K0` contracted with a left-side sketch."
_kpart_sketched(K0, Om) = to_concrete(contract(Om', (1, 2), K0, (1, 2)))    # (g, bond)

"""
    sketch_h_left(f, h, i, lenv, renv, Om) -> TLArray

`Y = (H Theta) Om†` with the right legs contracted against `Om`, legs
`(link_l, site_l, g)` -- the left matricization of `H Theta` sketched from the right.
This is the candidate space the left frame is expanded into.

`Om` carries `V0`'s layout `(g, site_r, link_r)`: it IS a candidate right frame, with
`g = Dpre` columns instead of `r`. `Om = f.V0` recovers the exact left factor of
`H Theta` projected on the current right frame.
"""
function sketch_h_left(f::BondFrame, h::XXZChain, i::Int,
                       lch::ChannelSet, rch::ChannelSet, Om)
    K0 = to_concrete(f.U0 * f.S0)          # (link_l, site_l, bond)
    V0 = f.V0                              # (bond, site_r, link_r)
    plainV = _vpart_plain(V0, Om)          # (bond, g)

    acc = nothing
    add!(x) = (acc = acc === nothing ? to_concrete(x) : to_concrete(acc + x))
    join2(K, V) = contract(K, (3,), V, (1,))          # (link_l, site_l, g)
    join3(K, V) = contract(K, (3, 4), V, (1, 3))      # ditto, op-leg shared

    # (a) terms entirely to the left of the block.
    lch.done === ZERO ||
        add!(join2(_env_on_link(K0, lch.done, 1), plainV))

    # (b) terms entirely to the right: the operator sits on link_r, which the sketch
    #     already covers, so it is pushed onto V0 before Om is contracted in.
    if rch.done !== ZERO
        Vd = _env_on_link(V0, rch.done, 3)
        add!(join2(K0, to_concrete(contract(Vd, (2, 3), Om', (2, 3)))))
    end

    for (t, term) in enumerate(h.terms)
        # (c) a term straddling link i: closed by term.right at site_l, on the left half.
        o = lch.open[t]
        if o !== ZERO
            K = _env_on_link(_apply_site_op(K0, term.right, i), o, 1)
            add!(to_concrete(term.coeff * join2(K, plainV)))
        end
        # (d) a term straddling link i+2: both of its legs are on the right half.
        o = rch.open[t]
        if o !== ZERO
            Vd = _env_on_link(_apply_site_op(V0, term.left, i + 1), o, 3)
            add!(to_concrete(term.coeff *
                 join2(K0, to_concrete(contract(Vd, (2, 3), Om', (2, 3))))))
        end
        # (e) the bond's own term: one half on each side, sharing the op-leg. Written
        #     from the term list rather than from `bond_gate`, because the gate fuses the
        #     two halves and there would be nothing left to split across the sketch.
        #     Both halves of a term carry an op-leg or neither does (`:U1` XY vs SzSz and
        #     the whole `:none` case), so one rank test covers both sides.
        Ke = _apply_site_op(K0, term.left, i)
        Ve = _apply_site_op(V0, term.right, i + 1)
        if length(Ke.inds) == 4
            Vs = to_concrete(permutedims(contract(Ve, (2, 3), Om', (2, 3)), (1, 3, 2)))
            add!(to_concrete(term.coeff * join3(Ke, Vs)))       # (bond, g, op)
        else
            Vs = to_concrete(contract(Ve, (2, 3), Om', (2, 3))) # (bond, g)
            add!(to_concrete(term.coeff * join2(Ke, Vs)))
        end
    end

    return acc
end

"""
    sketch_h_right(f, h, i, lenv, renv, Om) -> TLArray

Mirror of [`sketch_h_left`](@ref): `Om` carries `U0`'s layout `(link_l, site_l, g)` and
the result is `(g, site_r, link_r)`, the right matricization of `H Theta` sketched from
the left. `Om = f.U0` recovers the exact right factor.
"""
function sketch_h_right(f::BondFrame, h::XXZChain, i::Int,
                        lch::ChannelSet, rch::ChannelSet, Om)
    K0 = to_concrete(f.U0 * f.S0)
    V0 = f.V0
    plainK = _kpart_sketched(K0, Om)                  # (g, bond)

    acc = nothing
    add!(x) = (acc = acc === nothing ? to_concrete(x) : to_concrete(acc + x))
    join2(K, V) = contract(K, (2,), V, (1,))          # (g, site_r, link_r)
    join3(K, V) = contract(K, (2, 3), V, (1, 4))      # ditto, op-leg shared

    if lch.done !== ZERO
        K = _env_on_link(K0, lch.done, 1)
        add!(join2(_kpart_sketched(K, Om), V0))
    end
    rch.done === ZERO ||
        add!(join2(plainK, _env_on_link(V0, rch.done, 3)))

    for (t, term) in enumerate(h.terms)
        o = lch.open[t]
        if o !== ZERO
            K = _env_on_link(_apply_site_op(K0, term.right, i), o, 1)
            add!(to_concrete(term.coeff * join2(_kpart_sketched(K, Om), V0)))
        end
        o = rch.open[t]
        if o !== ZERO
            Vd = _env_on_link(_apply_site_op(V0, term.left, i + 1), o, 3)
            add!(to_concrete(term.coeff * join2(plainK, Vd)))
        end
        Ke = _apply_site_op(K0, term.left, i)                  # (link_l, site_l, bond, op)
        Ve = _apply_site_op(V0, term.right, i + 1)             # (bond, site_r, link_r, op)
        if length(Ke.inds) == 4
            Ks = to_concrete(contract(Om', (1, 2), Ke, (1, 2)))    # (g, bond, op)
            add!(to_concrete(term.coeff * join3(Ks, Ve)))
        else
            add!(to_concrete(term.coeff * join2(_kpart_sketched(Ke, Om), Ve)))
        end
    end

    return acc
end

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
                              comp_ratio::Float64 = 0.5,
                              rng::AbstractRNG = MersenneTwister(0x5EED),
                              tag::AbstractString = "cbeG")
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    0.0 < comp_ratio < 1.0 || throw(ArgumentError(
        "comp_ratio must be strictly between 0 and 1, got $comp_ratio"))
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

    n_c = max(round(Int, npre * comp_ratio), 1)
    n_t = max(npre - n_c, 1)

    blocks = Any[]

    # Complement half: random, then the frame projected OUT.
    RC = _trim_per_sector(_randomize_like(F, rng), 3, _alloc_columns(dC, n_c))
    if RC !== nothing
        C = to_concrete(perp_of(frame, tolayout(RC)))
        norm(C) > 0 && push!(blocks, C)
    end

    # Isometry half: random, then the frame projected IN. `R - perp(R)` is the projection
    # onto the frame's span, built from the same primitive so the two halves can never
    # disagree about what "the frame's span" is.
    RT = _trim_per_sector(_randomize_like(F, rng), 3, _alloc_columns(dT, n_t))
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

  - `U_ex`, `V_ex`  the expanded frames, `[U0 | new]` and `[V0 ; new]`. Both CONTAIN the
    old frame as their first block, which is what makes `Ŝ₀ = U_ex† Θ₀ V_ex†` lossless
    and `tau = 0` the identity on the state.
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

  - `dex` -- new directions to admit per side. `0` (default) means the full Sulz budget
    `r`, so the expanded rank is `2r` and the comparison against `bond_update_bug!` is
    like for like; the point of the method is that a much smaller `dex` should do, which
    is what stage 4 measures.
  - `dover` -- oversampling for the preselection. `nothing` gives `ceil(0.2*dex)`, i.e.
    `Dpre = ceil(1.2*dex)` as in the reference.
  - `comp_ratio` -- the sketch's complement/isometry split (`CompRatio`, default 0.5).
  - `sulz_cap` -- keep the expanded rank at or below `2r` (default `true`). The reference
    instead caps the expansion itself at `2r` (total `3r`, `RSVDpreBE0SiQS.m:77`); the
    tighter bound is kept by default because it is the regime whose order behaviour is
    already characterised.
  - `preselect_only` -- skip the final selection and take the preselected candidates
    directly, the reference's `'-p'` mode. Diagnostic: it isolates how much the
    weight-ranking is worth.
"""
function cbe_expand(f::BondFrame, h::XXZChain, i::Int,
                    lch::ChannelSet, rch::ChannelSet;
                    dex::Int = 0,
                    dover::Union{Nothing, Int} = nothing,
                    comp_ratio::Float64 = 0.5,
                    sulz_cap::Bool = true,
                    preselect_only::Bool = false,
                    stol_pre::Float64 = 1e-10,
                    stol_fnl::Float64 = 1e-13,
                    rng::AbstractRNG = MersenneTwister(0x5EED))
    U0, V0 = f.U0, f.V0
    r = f.old_rank
    none() = CBEExpansion(U0, V0, 0, 0, 0.0, 0.0)

    # Room to expand, per side: the local space minus what the frame already spans.
    room_l = leg_dim(U0, 1) * leg_dim(U0, 2) - r
    room_r = leg_dim(V0, 2) * leg_dim(V0, 3) - r
    budget = dex <= 0 ? r : dex
    sulz_cap && (budget = min(budget, r))
    dex_l, dex_r = min(budget, room_l), min(budget, room_r)
    (dex_l <= 0 && dex_r <= 0) && return none()      # both frames already complete

    npre = preselect_only ? max(budget, 1) :
           dover === nothing ? ceil(Int, 1.2 * budget) : budget + dover

    # ---- preselection: sketch H from each side, project off the frame ----------
    QL, err_l = if dex_l > 0
        OmR = sector_graded_sketch(V0, :right, npre; comp_ratio = comp_ratio, rng = rng)
        OmR === nothing ? (nothing, 0.0) :
            _preselect_left(U0, sketch_h_left(f, h, i, lch, rch, OmR), stol_pre)
    else
        (nothing, 0.0)
    end

    QR, err_r = if dex_r > 0
        OmL = sector_graded_sketch(U0, :left, npre; comp_ratio = comp_ratio, rng = rng)
        OmL === nothing ? (nothing, 0.0) :
            _preselect_right(V0, sketch_h_right(f, h, i, lch, rch, OmL), stol_pre)
    else
        (nothing, 0.0)
    end
    err_pre = sqrt(err_l^2 + err_r^2)

    (QL === nothing && QR === nothing) && return none()

    # ---- final selection: rank the candidates by their weight in H Theta ------
    TLC, TRC, err_fnl = if preselect_only || QL === nothing || QR === nothing
        # With one side empty there is no UMU to form; take what the preselection found.
        (QL === nothing ? nothing : _trim_total(QL, 3, dex_l),
         QR === nothing ? nothing : _trim_total(QR, 1, dex_r),
         0.0)
    else
        UMU = to_concrete(permutedims(
            contract(sketch_h_left(f, h, i, lch, rch, QR), (1, 2), QL', (1, 2)), (2, 1)))
        # EMPTY: the left and right candidate spaces share no charge sector, so no pair of
        # them can carry amplitude (the reference's `Empty` branch, :461; the same
        # constraint `pairable_charges` enforces for the random fill).
        if length(UMU.qlabels) == 0 || norm(UMU) < 1e-11
            (nothing, nothing, 0.0)
        else
            res = svd(UMU, (1,); Nkeep = min(dex_l, dex_r), cutoff = stol_fnl,
                      get_lists = true)
            (to_concrete(contract(QL, (3,), res.U, (1,))),
             to_concrete(contract(res.Vd, (2,), QR, (1,))),
             _trunc_weight(res))
        end
    end

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

    if sulz_cap
        leg_dim(U_ex, 3) <= 2r || error(
            "CBE expansion broke the Sulz bound on the left: $(leg_dim(U_ex, 3)) > 2r = $(2r)")
        leg_dim(V_ex, 1) <= 2r || error(
            "CBE expansion broke the Sulz bound on the right: $(leg_dim(V_ex, 1)) > 2r = $(2r)")
    end

    return CBEExpansion(U_ex, V_ex, n_l, n_r, err_pre, err_fnl)
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
