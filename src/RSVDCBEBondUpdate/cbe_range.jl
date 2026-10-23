# cbe_range.jl: the certified range finder and the expansion record (from `cbe_core.jl`, included right after it).

"""
    _certified_range(draw, wleg, select, qleg, project, sketch, side,
                     w0, wcap, ncap, alloc, ledger) -> (Q, err, n)

One side's candidate basis, widened until TEST B certifies it. `draw(w, boost)` returns a
probe of base width `w` with `boost` (sector => extra columns, or `nothing`) on top.

`_range_find` stops when the preselection SVD cuts something, which says the sketched
spectrum decayed INSIDE the probe. That is necessary but not sufficient: it is a statement
about the directions the probe happened to draw, and a sector the probe under-sampled can
decay convincingly while the range outside it is untouched -- the misleading tail. Test B
asks the independent question, how much of `P_perp (H Theta)` is left outside `Q`, with an
unboosted holdout that took no part in building `Q`, and on failure widens ONLY the sectors
the holdout names ([`_target_sectors!`](@ref)).

On the legacy path (`alloc === nothing`) this is exactly `_range_find` and nothing is drawn.

TERMINATION. Every pass that continues has a strictly wider sketch than the last, and the
width is bounded by `wcap`, so the loop ends. It stops with the test failed when (i) the
sketch reached `wcap` or `n` reached the admission limit `ncap` -- a resource bound,
recorded as `workcap_reached` -- or (ii) targeting could not widen the sketch because the
named sectors are already fully drawn. Both clear `tolerance_met`; neither is accepted
silently.
"""
function _certified_range(draw, wleg::Int, select, qleg::Int, project, sketch, side::Symbol,
                          w0::Int, wcap::Int, ncap::Int,
                          alloc::Union{Nothing, BasisAllocation},
                          ledger::Union{Nothing, AdaptiveLedger};
                          # A SEPARATE DISTRIBUTION ONLY IN TESTS: UT06 cripples the sketch to
                          # miss a sector, and the audit must not share the crippling.
                          holdout = w -> draw(w, nothing),
                          # ⛔ THIN SEARCH: TEST B, NOT THE NUMERICAL RANK, DECIDES THE WIDTH. The
                          # adaptive path runs the preselection at `stol_pre = 0`, so
                          # `_range_find`'s "the SVD cut something" stop fires only at numerical
                          # rank -- and `P_perp (H Theta)` is numerically full rank up to the local
                          # room (~chi), so every bond doubled its probe to ~chi columns before
                          # Test B was asked anything. The sketch was then as wide as the space it
                          # sketches, `tol_basis` bought no speed, and the RSVD could not beat
                          # exact CBE at any chi. `thin = true` draws ONE pass at each width and
                          # widens only when the holdout fails (targeted sectors doubled, else the
                          # base width), so the width -- the cost -- follows `tol_basis`.
                          # `thin = false` is the old full-rank widening, kept as the A/B.
                          thin::Bool = false)
    alloc === nothing &&
        return _range_find(w -> draw(w, nothing), wleg, select, qleg, w0, wcap, ncap)

    boost = Dict{Any, Int}()
    last = Ref{Any}(nothing)       # the last sketch drawn, for its per-sector widths
    lastw = Ref(w0)                # ...and the base width it was asked for
    function drawn(w)
        Om = draw(w, isempty(boost) ? nothing : boost)
        if Om !== nothing
            last[], lastw[] = Om, w
            ledger === nothing || (ledger.n_probes += leg_dim(Om, wleg))
        end
        return Om
    end
    width_of() = last[] === nothing ? 0 : leg_dim(last[], wleg)
    # One pass at width `w` (its own cap is `w`), or the full widening.
    search(w) = thin ? _range_find(drawn, wleg, select, qleg, w, min(w, wcap), ncap) :
                       _range_find(drawn, wleg, select, qleg, w, wcap, ncap)

    Q, err, n = search(w0)
    prev = -1
    while true
        width = width_of()
        # A quarter of the working width, so the holdout stays a fraction of the sketch it
        # audits. `_alloc_columns` still gives every reachable sector at least one column.
        nh = clamp(cld(max(width, 1), 4), 1, max(wcap, 1))
        resid, per, hcols = _sample_residual(Q, project, sketch, holdout, nh, wleg, side)
        ind = indicator(alloc, resid)
        ledger === nothing || (ledger.n_holdout += sum(values(hcols); init = 0))
        # The indicator is recorded for the basis RETURNED, not for the passes it replaced --
        # otherwise a side that recovered would report a breach alongside `tolerance_met`.
        ok = ind <= alloc.sample
        capped = width >= wcap || n >= ncap
        if ok || capped || width <= prev
            ledger === nothing || _record_sample!(ledger, side, ind, per, hcols, last[],
                                                  wleg, alloc)
            ok && return (Q, err, n)
            if ledger !== nothing
                _unmet!(ledger, :sample)
                ledger.workcap_reached |= capped
            end
            return (Q, err, n)
        end
        nt = _target_sectors!(boost, per, last[], wleg, alloc)
        ledger === nothing || (ledger.n_targeted += nt)
        prev = width
        # Thin: the targeted sectors were doubled; if the holdout named none, double the base.
        Q, err, n = search(thin && nt == 0 ? min(2 * lastw[], wcap) : lastw[])
    end
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

  - `dex` -- absolute per-side cap on the directions admitted. `0` (default) is ADAPTIVE:
    there is no count to hit, the cap is the hard local limit `room`, and what is actually
    admitted is whatever cleared `stol_pre` in the preselection SVD. Any positive value
    restores the old behaviour -- a fixed per-side cap, overriding the schedule.
  - `growth` -- the schedule's target, as a multiple of the NEIGHBOURHOOD max bond
    dimension: `budget = ceil(growth*dmax) - r`. Default `2.0`. At `dex = 0` this NO LONGER
    CAPS the expansion; it only seeds the sketch width the range finder starts from, which
    sets how many passes the loop needs and not what it converges to. See the budget block
    in the body.
  - `dover` -- oversampling for the preselection. `nothing` (default) is ADAPTIVE: the probe
    starts at `Dpre = ceil(1.2*budget)` and is WIDENED (doubling) until the preselection SVD
    cuts something, i.e. until the sketched range reaches past the decay rather than ending
    inside it -- see [`_range_find`](@ref). An integer pins the width at `budget + dover`
    with no range finding, as before.
  - `comp_ratio` -- the sketch's complement/isometry split (`CompRatio`). `1.0` draws the
    sketch ENTIRELY from the frame's orthogonal complement -- the discarded space; `0.5`
    restores the reference's split; `nothing` drops the split entirely (probe drawn from
    the full randomised fusion basis, with the exact `P_perp` still applied downstream in
    preselection, so orthogonality is not at risk either way).

    ⚠️ THE DEFAULT IS PER CALLER, AND THEY DISAGREE ON PURPOSE. `cbe_expand` itself and both
    TDVP arms (`tdvp_cbe1s_sweep!`, `tdvp1_cbe_sweep!`) default to `0.5`, the reference's
    split; `cbe_bug_step!` and `dmrg_cbe1s_sweep!` default to `1.0`. Do not "unify" them
    from this docstring -- the `1.0` callers are the two that were MEASURED (see
    `dmrg_cbe1s_sweep!`), and `growth = 2.0` is already an example of one arm's constant
    being adopted everywhere without a measurement to back it.

    ⚠️ `1.0` DEPARTS FROM THE REFERENCE DELIBERATELY. `RSVDpreBE0SiQS.m:339` reads "do not
    project into complement space, 1-site components are also important for bond update",
    i.e. the MATLAB explicitly rejects the pure-complement choice and splits the probe.
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
function cbe_expand(f::BondFrame, h::XXZChain, i::Int,
                    lch::ChannelSet, rch::ChannelSet;
                    share_ht::Union{Nothing, Bool} = nothing,
                    # ⛔ ACCEPTED ONLY TO REFUSE IT WITH A SENTENCE THAT SAYS WHY. `fold_omega`
                    # is implemented for the MPO path only; without this the term-list caller
                    # gets a `MethodError: no method matching cbe_expand(::BondFrame, ::Function,
                    # ::Function; fold_omega=true)` raised inside the two-argument method, which
                    # names neither the flag's restriction nor the caller that set it.
                    fold_omega::Bool = false, kwargs...)
    fold_omega && throw(ArgumentError(
        "fold_omega is implemented for the MPO path only; this is the term-list (XXZChain) " *
        "expansion. Build the Hamiltonian as an MPO, or leave fold_omega = false."))
    return cbe_expand(f,
                      _sketch_closures(f,
                          () -> apply_h_two_site(frame_theta(f), h, i, lch, rch);
                          share = share_ht === nothing ? true : share_ht)...; kwargs...)
end
