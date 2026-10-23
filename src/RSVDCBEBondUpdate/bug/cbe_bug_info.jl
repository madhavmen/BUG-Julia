
"""
    CBEBugSweepInfo

What one [`cbe_bug_step!`](@ref) did.

  - `expanded` -- the bond dimension the basis update left at each bond, before the closing
    truncation. In MULTIPLETS under a non-abelian symmetry.
  - `n_new` -- directions CBE admitted per bond; an upper bound on what `expanded` could grow by.
  - `root`, `root_rank` -- which bond took the Galerkin step, and its kept rank afterwards.
  - `err_pre`, `err_fnl` -- largest preselection / final-selection weight discarded. `err_fnl`
    is the one to watch: a large value means the BUDGET, not the selection, limited accuracy.
  - `discarded` -- largest relative weight any single SVD of the closing truncation (or the
    root split) threw away. ⚠ NOT the step's compression error: that is `ledger.discarded`, the
    cumulative relative norm over every bond (plan §5).
  - `ledger` -- the step's [`AdaptiveLedger`](@ref) (Test A/B/C/D indicators, probe counts,
    cumulative compression, projection error, ranks, `tolerance_met`,
    `workcap_reached`). [`budget_status`](@ref) reads
    `:met` / `:budget_unmet` off it and [`budget_diagnostic`](@ref) says why.
  - `krylov_dims` -- operator applications, summed over the `L` K-steps and the one root solve,
    plus, for `order > 1`, the frontier sweeps' one-site and zero-site applications (their
    two-site leak measurements are `ledger.n_h2_leak`).
  - `peak_elements` -- working set (`state + accumulated frames + transients alive at that
    moment`), the same definition `TDVP2Info` uses, so the two are comparable.
    ⚠ **This sweep does NOT have a lower working set than 2-site TDVP.** It never forms a rank-4
    block, which is true and is why its transients are small -- but it holds O(L) FRAMES alive
    until the root solve, and at L=16, chi=32 that measured *larger* than `tdvp2_step!`'s
    `state + one Theta`. The advantage this sweep actually has is in OPERATOR APPLICATIONS and
    wall clock (measured 4.2x fewer applications and 1.35x faster than 2-site TDVP at L=16), not
    in memory.
"""
struct CBEBugSweepInfo
    expanded::Vector{Int}
    n_new::Vector{Int}
    root::Int
    root_rank::Int
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    peak_elements::Int
    # Extra CBE growth passes of the removed fixed-knob path; always 0, kept so the record's
    # layout (and every recorded run) is unchanged.
    n_grow::Int
    # ⛔ THE ROOT'S **RIGHT** FRAME WIDTH, WHICH NOTHING ELSE RECORDED. `expanded[c]` is
    # `leg_dim(W[c], 3)` -- the LEFT frame only -- and the right half-sweep skips the root
    # (`j != c && (expanded[j] = ...)`), so `b_R` was invisible. That matters because the Galerkin
    # space is `b_L x b_R`: judging completeness from `b_L` alone can call a basis complete when it
    # is not, which is exactly the wrong way for a diagnostic to fail.
    root_right::Int
    # ⛔ SECONDS SPENT INSIDE `cbe_expand`, WHICH IS THE ONLY THING THE RANDOMISED SKETCH MAKES
    # CHEAPER. Without it, an rSVD study cannot tell "the sketch is slow" from "the sketch is fast
    # but the expansion is a small share of the step" -- and those call for opposite responses.
    # Amdahl: if the expansion is a fraction `f` of the step, the speedup is capped at `1/(1-f)`
    # however good the sketch gets. MEASURED 1.10x against a `Dpre/(d*chi)` ceiling of ~16x, which
    # is exactly the shape of a small `f`.
    t_cbe::Float64
    # ⛔ A PHASE BREAKDOWN OF THE STEP, because "the sketch did not help" has two very different
    # causes and only one is a code problem. The sketch makes `cbe_expand` cheaper and nothing
    # else, so Amdahl caps the whole-step speedup at `1/(1 - t_cbe/t_step)`. If `t_cbe` is a small
    # share, the answer is not a better sketch -- it is whatever these other phases are doing.
    t_kry::Float64      # building the Krylov frames (`_krylov_enrich`)
    t_root::Float64     # the root Galerkin solve (`expv` on the zero-site generator)
    t_trunc::Float64    # the closing `truncate_recursive!`
    # ⛔ THE PHASES THAT CLOSE THE ACCOUNTING. MEASURED at L=50, chi=64 with only the four timers
    # above: cbe 0.445 + kry 0.128 + root 0.003 + trunc 0.029 = 0.605, leaving ~40% UNATTRIBUTED.
    # A profile that does not add up cannot locate a bottleneck, and the missing block is bigger
    # than every phase except the expansion itself.
    #
    # ⚠ `t_frame` IS AN SVD TOO, and the CBE sketch does not touch it: `_frame_from` factorises
    # both sides of a bond.
    t_split::Float64    # the removed fixed-knob path's frame splits; always 0
    t_frame::Float64    # `_frame_from`, which factorises both sides of the bond
    t_env::Float64      # environment stacks and channel pushes
    t_step::Float64     # the whole call, so the shares can be checked to sum to 1
    # ⛔ THE STEP'S VERDICT. Without it a step that failed Test A/B, or stopped at the work
    # ceiling, returns looking exactly like one that passed -- the silent convergence claim the
    # three-tolerance API exists to rule out.
    ledger::AdaptiveLedger
    # Plan Task 1.4: the typed record of the step -- CBE policy, kernel counters, ranks, limit
    # statuses (`StepRecord`, dd_cbe.jl).
    record::StepRecord
end

"""
    _assemble_and_compress!(psi, W, Z, c, res, ledger, bwts, bd0; maxdim, tol_trunc, truncate)
        -> (root_rank, discarded, t_trunc)

Stage 4 of [`cbe_bug_step!`](@ref): the frames become the new site tensors -- `W[1:c-1]`, the
split root core `W[c] U` and `S Vd Z[c]`, `Z[c+1:L-1]` -- and then ONE compression of the evolved
state (plan §5). `tol_trunc` is a budget on the CUMULATIVE discarded norm over all bonds, shared
out by `truncate_recursive!` into `bwts`; the ledger records it and closes the per-bond log.
`discarded` is the largest single relative weight the root split or the compression threw away.
"""
function _assemble_and_compress!(psi, W, Z, c::Int, res, ledger::AdaptiveLedger, bwts, bd0;
                                 maxdim::Int, tol_trunc::Float64, truncate::Bool)
    L = length(psi)
    discarded = _trunc_weight(res)
    for i in 1:(c - 1)
        psi[i] = _relink(W[i], psi[i].inds[1].itags, psi[i].inds[3].itags)
    end
    t1c, t3c = psi[c].inds[1].itags, psi[c].inds[3].itags
    t3r = psi[c + 1].inds[3].itags
    psi[c]     = _relink(to_concrete(W[c] * res.U), t1c, t3c)
    psi[c + 1] = _relink(to_concrete((res.S * res.Vd) * Z[c]), t3c, t3r)
    for j in (c + 1):(L - 1)
        psi[j + 1] = _relink(Z[j], psi[j + 1].inds[1].itags, psi[j + 1].inds[3].itags)
    end
    psi.center = c + 1
    root_rank = leg_dim(psi[c], 3)

    ttr = 0.0
    if truncate
        t0 = time_ns()
        _, wtr, cumw, capped = truncate_recursive!(psi, c + 1; maxdim = maxdim,
                                                   weight_tol = tol_trunc, bond_weights = bwts)
        ttr = (time_ns() - t0) / 1e9
        discarded = max(discarded, wtr)
        ledger.n_compress += 1
        ledger.discarded = cumw
        ledger.maxdim_reached = capped
        # Relative slack for the rounding in the per-bond shares, which sum to `tol` exactly.
        cumw > tol_trunc * (1 + 1e-8) && _unmet!(ledger, :trunc)
    end
    bd1 = bond_dims(psi)
    ledger.r_final = maximum(bd1; init = 0)
    # Close the per-bond log with what only exists after the root solve and compression.
    ledger.bond_log = Any[merge(e, (r_old = bd0[e.bond], r_final = bd1[e.bond],
                                    discarded = bwts[e.bond])) for e in ledger.bond_log]
    return (root_rank, discarded, ttr)
end
