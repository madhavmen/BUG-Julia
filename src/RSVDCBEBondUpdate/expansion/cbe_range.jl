# cbe_range.jl: the certified range finder and the expansion record (from `cbe_core.jl`, included right after it).

"""
    _certified_range(draw, wleg, select, qleg, project, sketch, side,
                     w0, wcap, ncap, alloc, ledger) -> (Q, err, n)

One side's candidate basis, widened until TEST B certifies it. `draw(w, boost)` returns a probe of
base width `w`. Stopping because the preselection SVD cut something only says the sketched spectrum
decayed inside the probe, so an under-sampled sector can look converged. Test B checks the rest with
an unboosted holdout and widens only the sectors it names ([`_target_sectors!`](@ref)).

On the legacy path (`alloc === nothing`) this is exactly `_range_find`. A resource bound (`wcap`,
`ncap`) or an exhausted target stops the loop with the test failed, never silently.
"""
function _certified_range(draw, wleg::Int, select, qleg::Int, project, sketch, side::Symbol,
                          w0::Int, wcap::Int, ncap::Int,
                          alloc::Union{Nothing, BasisAllocation},
                          ledger::Union{Nothing, AdaptiveLedger};
                          # A SEPARATE DISTRIBUTION ONLY IN TESTS: UT06 cripples the sketch to
                          # miss a sector, and the audit must not share the crippling.
                          holdout = w -> draw(w, nothing),
                          # Thin search: Test B, not the numerical rank, decides the width.
                          # The adaptive preselection runs at `stol_pre = 0`, so the SVD cut fires only at numerical rank (about
                          # chi), which made every probe full width.
                          # `thin = true` draws one pass per width and widens only when the holdout fails; `thin = false` is the old A/B.
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

  - `U_ex`, `V_ex`: the expanded frames. They span `[U0 | new]`, rotated within that span, which keeps
    the expansion lossless and `S_0` free of an identically zero block.
  - `n_new_l`, `n_new_r`: directions admitted per side. `err_pre`, `err_fnl`: weight discarded
    (`err_fnl` is the cost of the `Dex` cap).
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

The basis update of CBE-BUG: expands the frames at bond `(i, i+1)` with no K/L ODE. Keywords:
`dex` is a per-side cap (0 = adaptive, bounded by the local room); `growth` seeds the sketch width
(default 2.0, not a cap); `dover` sets the oversampling (nothing = adaptive); `comp_ratio` is the
complement/isometry split, 0.5 for the TDVP callers and 1.0 for the DMRG and CBE-BUG callers, as the
MATLAB reference rejects a pure complement; `sulz_cap` caps the total at 2r (off by default);
`preselect_only` skips the final selection.
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
