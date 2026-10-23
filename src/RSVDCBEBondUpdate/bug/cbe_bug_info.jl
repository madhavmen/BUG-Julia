
"""
    CBEBugSweepInfo

What one [`cbe_bug_step!`](@ref) did:

  - `expanded`, `n_new`: bond dims after the update (multiplets under SU(2)); directions admitted.
  - `root`, `root_rank`: the Galerkin bond and its kept rank.
  - `err_pre`, `err_fnl`: largest discarded weight in preselection / final selection.
  - `discarded`: largest weight one closing SVD discarded (cumulative: `ledger.discarded`).
  - `ledger`: the [`AdaptiveLedger`](@ref), read by [`budget_status`](@ref).
  - `krylov_dims`, `peak_elements`: applications and working set (not below 2-site TDVP's).
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
    # Seconds inside `cbe_expand`, the only phase the randomised sketch makes cheaper. Without this
    # timer an rSVD study cannot tell a slow sketch from a fast sketch inside a small expansion
    # share, and those call for opposite responses: by Amdahl, the speedup is capped at 1/(1-f) for
    # an expansion share f.
    t_cbe::Float64
    # ⛔ A PHASE BREAKDOWN OF THE STEP, because "the sketch did not help" has two very different
    # causes and only one is a code problem. The sketch makes `cbe_expand` cheaper and nothing
    # else, so Amdahl caps the whole-step speedup at `1/(1 - t_cbe/t_step)`. If `t_cbe` is a small
    # share, the answer is not a better sketch -- it is whatever these other phases are doing.
    t_kry::Float64      # building the Krylov frames (`_krylov_enrich`)
    t_root::Float64     # the root Galerkin solve (`expv` on the zero-site generator)
    t_trunc::Float64    # the closing `truncate_recursive!`
    # The timers that close the step's time accounting: without them a profile leaves a large
    # unattributed share. `t_frame` is an SVD too (`_frame_from` factorises both sides of a bond),
    # so it is not part of the CBE sketch. `t_split` is the removed fixed-knob path's frame splits
    # and is always 0.
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
