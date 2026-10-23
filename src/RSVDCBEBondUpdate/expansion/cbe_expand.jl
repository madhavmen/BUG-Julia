# cbe_expand.jl: `cbe_expand`, the two-sided SVD selection (DMRG / TDVP-CBE) (from `cbe_core.jl`, included right after it).

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
                    # ⚠ THESE DEFAULT TO THEIR HISTORICAL VALUES, NOT TO `nothing`, and that
                    # is the difference between this layer and `cbe_bug_step!`. The step is
                    # the PUBLIC API and has to tell "not asked for" from "asked for the old
                    # value" in order to pick a path; `cbe_expand` is the kernel it dispatches
                    # to, and by the time control arrives here the path is already chosen and
                    # every value is concrete. Giving the kernel sentinels too would move a
                    # decision that belongs at the boundary into the hot loop.
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
    # ⛔ AND IT HAS NOW SHOWN UP IN A MEASUREMENT, so the abelian case no longer pays for it.
    # `reachable_sectors` builds a FULL `fusion_basis` -- `getIdentity` plus a complete `svd` of a
    # `(d*chi) x (d*chi)` blocked tensor -- and then reads nothing from it but `.spaces[end]`. It
    # is called TWICE per expand, at `2(L-1)` bonds a step: 116 large SVDs per step at L=30,
    # purely to COUNT DIMENSIONS. At chi=1024 `LAPACK svd/qr` measured 4.0% of cbe1s's active
    # work and 6.1% of BUG's, and the discarded `U` is allocation on top of that.
    #
    # For ABELIAN charges the answer is exactly `leg_dim(a) * leg_dim(b)` -- the paragraph above
    # says so ("It reduces EXACTLY to the old product for abelian charges"), because every pair of
    # input sectors fuses to exactly one output of size `da*db`, and `leg_dim` is just the summed
    # sector dimensions. So the product is not an approximation of the fused dimension here, it IS
    # it, and `test_fused_dim_fast.jl` asserts the two agree sector-sum for sector-sum.
    #
    # ⚠ NON-ABELIAN STILL TAKES THE SLOW PATH, and must: under SU(2) `leg_dim` counts MULTIPLETS,
    # so the product reads `chi * 1 = chi` for a spin-1/2 site, `room` collapses to ~0, and CBE
    # would decline to expand -- the chi = 1 freeze described above. That failure is silent, which
    # is why this branches on the symmetry rather than on a tolerance.
    fused_dim(t, a, b) = symmetry_mode() === :SU2 ?
        sum(d for (_, d) in reachable_sectors(t, a, b); init = 0) :
        leg_dim(t, a) * leg_dim(t, b)
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

    # ADAPTIVE BUDGET AND ADAPTIVE SKETCH WIDTH -- NEITHER ADDS A KNOB.
    #
    # `dex` and `dover` were both GUESSES MADE BEFORE ANY SPECTRAL INFORMATION EXISTS: a
    # per-side count of directions to admit, and a fixed 20% oversample on top of it. The
    # preselection SVD already produces the quantity both were guessing at -- the singular
    # values of `P_perp (H Theta) Om`, cut at `stol_pre` -- so both are read off it instead,
    # against the tolerances this routine already carries.
    #
    #   * `dex` (the per-side cap) becomes the hard local limit `room`, and the number of
    #     directions actually admitted is whatever cleared `stol_pre`, i.e. `n_j` below. The
    #     cap stops CAPPING and the spectrum decides. This is not a new behaviour downstream:
    #     `_trim_total(Q, leg, dex_*)` already kept `min(n_j, dex_*)`, so the schedule was only
    #     ever an upper bound, and it was the BINDING one -- `err_fnl = 0.62` in the table
    #     above is that bound discarding 62% of the ranked candidate weight.
    #   * `dover` (the oversample) becomes a randomised-range-finder loop (`_range_find`): the
    #     probe is widened until the preselection SVD CUTS something, which is the signal that
    #     the sketched range has captured the decay rather than ending inside it. A fixed
    #     1.2x cannot know that; a cut spectrum is proof of it.
    #
    # The schedule survives as the range finder's STARTING width only. It sets how many passes
    # the loop needs, not what it converges to -- the loop only ever widens, and the budget is
    # read off the spectrum afterwards either way.
    #
    # Both knobs keep their old meaning when set explicitly: `dex > 0` is still an absolute
    # per-side cap, and an explicit `dover` still pins the width at `budget + dover` with no
    # range finding, which is what the A/B benchmarks and `test_cbe_bug_exact_refs.jl` drive.
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
                # TEST A. The `Nkeep = joint` cut above is a RESOURCE bound -- the smaller
                # side's headroom -- and says nothing about accuracy. The tail test asks the
                # accuracy question of the ranked spectrum it produced: how few of these
                # directions can be kept with the discarded weight still inside the bond's
                # share of `tol_basis`. Keeping more than that buys rank no observable needs;
                # keeping fewer breaks the budget, which is why `k` is capped BELOW by nothing
                # and the shortfall is reported instead.
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

    # TEST A ON A SIDE THE JOINT RANKING DID NOT COVER. With no joint SVD -- the other side
    # has no candidates, or the two share no charge sector -- the fallback below used to keep
    # every preselected direction, so the tail test applied to one kind of bond and not the
    # other. Here the side's candidates are ranked by their EXACT weight against the complete
    # opposite local space, `Q' (P_perp H Theta) F` with `F` that space's full basis -- the
    # same operator the side's preselection sketched (no projection on the far side, exactly
    # as `skl`/`skr` apply it). This contracts against the full local basis, which the sketch
    # otherwise avoids; it runs only on bonds where the joint ranking is unavailable.
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

    return CBEExpansion(_check_frame(U_ex, :left, "cbe_expand U_ex"),
                        _check_frame(V_ex, :right, "cbe_expand V_ex"), n_l, n_r, err_pre, err_fnl)
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
