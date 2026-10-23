# cbe_expand.jl: `cbe_expand`, the two-sided SVD selection (DMRG / TDVP-CBE) (from `cbe_core.jl`, included right after it).

"""
    cbe_expand(f, skl, skr; kwargs...) -> CBEExpansion

The selection half of controlled bond expansion, with the H-action supplied as two closures:
`skl(Om)` gives `(H Theta) Om†` in `U0`'s layout (the left candidates), `skr(Om)` the mirror.
The gate path and the environment path share this selection code. The keywords are internal
settings of the callers (DMRG, TDVP-CBE, the gate path), not a public API.
"""
function cbe_expand(f::BondFrame, skl, skr;
                    # Concrete internal defaults (the TDVP-CBE baselines' fixed-count selection); callers that need
                    # other values pass them.
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
                    # The step's allocation and ledger. `nothing` is the LEGACY path: fixed
                    # counts, no residual tests, exactly the historical selection. An
                    # allocation switches on Tests A/B and the rank comes off the spectrum.
                    alloc::Union{Nothing, BasisAllocation} = nothing,
                    ledger::Union{Nothing, AdaptiveLedger} = nothing,
                    workcap::Int = 0,
                    # Adaptive path only (`alloc !== nothing`): Test B sets the sketch width --
                    # see `_certified_range(...; thin)`. `false` is the old full-rank widening.
                    thin_search::Bool = true,
                    rng::AbstractRNG = MersenneTwister(0x5EED))
    U0, V0 = f.U0, f.V0
    r = f.old_rank
    none() = CBEExpansion(U0, V0, 0, 0, 0.0, 0.0)

    # Room to expand per side: the fused local dimension minus the frame's rank. Abelian charges use the
    # product of leg dimensions (exact there); SU(2) counts multiplets, so it must ask `reachable_sectors`.
    fused_dim(t, a, b) = symmetry_mode() === :SU2 ?
        sum(d for (_, d) in reachable_sectors(t, a, b); init = 0) :
        leg_dim(t, a) * leg_dim(t, b)
    room_l = fused_dim(U0, 1, 2) - r
    room_r = fused_dim(V0, 2, 3) - r
    # Growth is neighbourhood-coupled as in `RSVDpreBE0SiQS.m` (target `growth` x the neighbouring max
    # rank), so a wide bond pulls its narrow neighbours up; `growth = 2.0` rather than the DMRG 1.1.
    dmax = max(leg_dim(U0, 1), r, leg_dim(V0, 3))
    budget = dex <= 0 ? max(ceil(Int, growth * dmax) - r, 1) : dex
    # No `2r` (Sulz) bound by default: growth is limited by `room` and by `maxdim` at the truncation.
    # `sulz_cap = true` keeps the global bound for an A/B.
    rm = rmax > 0 ? rmax : r
    sulz_cap && (budget = min(budget, max(2 * rm - r, 0)))

    # At `dex = 0` / `dover = nothing` the admitted count and the sketch width are read off the
    # preselection spectrum (range finder) instead of fixed guesses; positive values pin them.
    cap_l, cap_r = dex > 0 ? (min(budget, room_l), min(budget, room_r)) : (room_l, room_r)
    if sulz_cap
        lim_ex = max(2 * rm - r, 0)
        cap_l, cap_r = min(cap_l, lim_ex), min(cap_r, lim_ex)
    end
    # PLAN §5: THE WORK CEILING ALSO BOUNDS WHAT IS RETAINED, not only how wide the sketch is.
    # A frame of width `r` may grow to `workcap` and no further; `workcap >= 2r`, so the
    # augmented rank-`2r` basis always fits. Where this cap is the binding one and the
    # candidates fill it, `workcap_reached` is set; whether accuracy suffered is Test A's call
    # (the directions it cut are in its tail), so a cap is reported, never silently accepted.
    wbind_l = wbind_r = false
    if alloc !== nothing && workcap > 0
        wk = max(workcap - r, 0)
        wbind_l, wbind_r = wk < cap_l, wk < cap_r
        cap_l, cap_r = min(cap_l, wk), min(cap_r, wk)
    end
    (cap_l <= 0 && cap_r <= 0) && return none()      # both frames already complete

    npre = preselect_only ? max(budget, 1) :
           dover === nothing ? ceil(Int, 1.2 * budget) : budget + dover
    # The probe for the LEFT expansion lives on the RIGHT frame's fused space and vice versa,
    # so that is where each side's width ceiling comes from. Widening past it buys nothing:
    # `sector_graded_sketch` caps every sector at its own dimension and simply returns fewer
    # columns, which `_range_find` reads as "the local space is exhausted".
    full_l, full_r = room_l + r, room_r + r

    # ── preselection: sketch H from each side, project off the frame ──
    # The probe goes on `H Theta`'s opposite leg, so the expanded side stays exact; `exact = true` uses
    # the complete fused basis (plain CBE) and makes the probe settings inert.
    probe(frame, side, w, boost) = exact ? full_local_basis(frame, side) :
                                   sector_graded_sketch(frame, side, w; boost = boost,
                                                        comp_ratio = comp_ratio, rng = rng)
    # An explicit `dover` pins the width, and `exact` has no width to pin -- the probe is the
    # complete basis, so there is nothing to widen and the first pass already sees everything.
    wcap_l, wcap_r = dover === nothing && !exact ? (full_r, full_l) : (npre, npre)
    # THE WORK CEILING IS THE ONE RESOURCE LIMIT, and it is what keeps the certified loop an
    # RSVD. Test B widens until the residual fits, and at a bond whose spectrum does not decay
    # that is satisfied only by the complete fused basis -- exactly the cost the sketch exists
    # to avoid. `workcap` (`2*max(maxdim, rmax)`, set by the caller from `maxdim`) stops the
    # doubling there and the shortfall is REPORTED as `workcap_reached` rather than paid for.
    if alloc !== nothing && workcap > 0
        wcap_l, wcap_r = min(wcap_l, workcap), min(wcap_r, workcap)
    end
    # The exact probe IS the complete local basis, so there is no range left for Test B to
    # find, and its holdout would be drawn from that same deterministic basis.
    alloc_b = exact ? nothing : alloc

    QL, err_l, njl = if cap_l > 0
        _certified_range((w, b) -> probe(V0, :right, w, b), 1,
                         Y -> _preselect_left(U0, skl(Y), stol_pre), 3,
                         X -> perp_component(U0, X), skl, :left,
                         npre, wcap_l, cap_l, alloc_b, ledger; thin = thin_search)
    else
        (nothing, 0.0, 0)
    end

    QR, err_r, njr = if cap_r > 0
        _certified_range((w, b) -> probe(U0, :left, w, b), 3,
                         Y -> _preselect_right(V0, skr(Y), stol_pre), 1,
                         X -> perp_component_right(V0, X), skr, :right,
                         npre, wcap_r, cap_r, alloc_b, ledger; thin = thin_search)
    else
        (nothing, 0.0, 0)
    end
    err_pre = sqrt(err_l^2 + err_r^2)

    (QL === nothing && QR === nothing) && return none()

    # THE BUDGET IS READ OFF THE SPECTRUM HERE, not guessed before it: `n_j` counts the
    # candidate directions that cleared `stol_pre` in the preselection SVD, and the per-side
    # cap is the hard local limit (plus an explicit `dex`/`sulz_cap` when one was asked for).
    dex_l, dex_r = min(njl, cap_l), min(njr, cap_r)
    ledger === nothing || ((wbind_l && njl >= cap_l) || (wbind_r && njr >= cap_r)) &&
        (ledger.workcap_reached = true)

    # ── final selection: rank the candidates by their weight in H Theta ──
    # Per side, as the reference's directional CBE does: the joint ranking is used only where it can
    # serve a side's whole budget, otherwise the side keeps its own weight-ordered candidates.
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
                # Test A: keep the fewest ranked directions whose discarded weight fits the bond's `tol_basis`
                # share; a shortfall is reported, not hidden.
                if alloc !== nothing
                    k, tail = _tail_rank(resj.kept_list, alloc.scale, alloc.tail, joint;
                                         base = _sq_weight(resj.trunc_list))
                    if ledger !== nothing
                        ledger.max_tail = max(ledger.max_tail, tail)
                        tail <= alloc.tail || _unmet!(ledger, :tail)
                    end
                    if k < joint
                        joint = k
                        resj = k > 0 ?
                               svd(UMU, (1,); Nkeep = k, cutoff = stol_fnl, get_lists = true) :
                               nothing
                        err_fnl = resj === nothing ? 0.0 : _trunc_weight(resj)
                    end
                    # THE TAIL RANK MUST BIND BOTH SIDES, not just the joint rotation. The
                    # per-side fallback below trims to `dex_*`, which is a headroom bound; a
                    # side with more headroom than `k` would otherwise admit directions the
                    # tail test just certified as unnecessary, and the test would silently
                    # apply to one side only.
                    dex_l, dex_r = min(dex_l, joint), min(dex_r, joint)
                end
            end
        end
    end

    # Test A on a side the joint ranking did not cover: rank its candidates by their exact weight against
    # the opposite side's full local basis (only on such bonds).
    oneL = alloc !== nothing && !preselect_only && njl > 0 && resj === nothing && dex_l > 0
    oneR = alloc !== nothing && !preselect_only && njr > 0 && resj === nothing && dex_r > 0
    UL = oneL ? _one_sided_tail(to_concrete(contract(QL', (1, 2),
                                                     skl(full_local_basis(V0, :right)), (1, 2))),
                                alloc, dex_l, stol_fnl, ledger) : nothing
    UR = oneR ? _one_sided_tail(to_concrete(contract(QR', (2, 3),
                                                     skr(full_local_basis(U0, :left)), (2, 3))),
                                alloc, dex_r, stol_fnl, ledger) : nothing

    TLC = if njl == 0
        nothing
    elseif resj !== nothing && dex_l <= joint
        to_concrete!(contract(QL, (3,), resj.U, (1,)))
    elseif oneL
        UL === nothing ? nothing : to_concrete!(contract(QL, (3,), UL, (1,)))
    else
        _trim_total(QL, 3, dex_l)
    end
    TRC = if njr == 0
        nothing
    elseif resj !== nothing && dex_r <= joint
        to_concrete!(contract(resj.Vd, (2,), QR, (1,)))
    elseif oneR
        UR === nothing ? nothing : to_concrete!(contract(UR, (1,), QR, (1,)))
    else
        _trim_total(QR, 1, dex_r)
    end

    # Re-establish orthonormality and orthogonality to the frame before the direct sum. See
    # the guard block above `_trim_total` for why this is not redundant with the preselection's
    # `RepairOrtho`: that pass ran on `Q`, this one runs on what the final selection made of it.
    TLC = _guarded_block(U0, TLC, :left)
    TRC = _guarded_block(V0, TRC, :right)

    if ledger !== nothing
        ledger.n_selected += (TLC === nothing ? 0 : leg_dim(TLC, 3)) +
                             (TRC === nothing ? 0 : leg_dim(TRC, 1))
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

    # No rotation of the expanded frame: an expansion is lossless, and rank grows only where the state
    # is evolved in the widened space.
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

    return CBEExpansion(_check_frame(U_ex, :left, "cbe_expand U_ex"),
                        _check_frame(V_ex, :right, "cbe_expand V_ex"), n_l, n_r, err_pre, err_fnl)
end

# ── orthonormality guard on the appended blocks ──
# Project the frame out, then orthonormalise (Telum has no QR, so `svd(...).U`); cutoff 1e-14
# drops columns that collapse, and a dropped column shows in `n_new_*`.
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

"""
    _guarded_block(frame, C, side) -> the block `C` cleared for `[frame | C]`, or nothing

Orthonormal and orthogonal to `frame` (`_reortho_*`), then at most the local room of each sector
(`_cap_to_local_room`). The second half is not redundant with the first: `_reortho`'s SVD cutoff is
relative to the GLOBAL sigma_max, so in a sector the frame already fills, its orthogonality loss
survives the cut and is renormalised into a fake unit column. MEASURED (IT_FRAME_CHECKS, IT-1 L=20
domain wall, panel, bug4): `cbe_expand V_ex` held 10 directions in a sector of local dimension 9
(d50f_trap 17346775, trapezoid, tol_basis 1e-5) and `cbe_expand U_ex` 253 in the full half-chain
S^z = 0 sector of 252 (d50f_rad3 17346774, radau3, 1e-3); the closure frontier's Krylov enrichment
then refused the frame and the runs crashed (it1r 17345037, 17345039).
"""
_guarded_block(frame, C, side::Symbol) =
    _cap_to_local_room(frame, side === :left ? _reortho_left(frame, C) :
                                               _reortho_right(frame, C), side)


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
