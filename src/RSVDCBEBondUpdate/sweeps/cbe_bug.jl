
_tagW(i::Int) = "cbeW,$i"
_tagZ(j::Int) = "cbeZ,$j"

# The residual below which the Krylov space has genuinely CLOSED -- an invariant subspace, not a
# convergence judgement. Kept separate from `krylov_tol` on purpose: one is a property of the
# operator and the seed, the other is an accuracy request from the caller.
const _KRY_BREAKDOWN = 1e-13

"""
    _kry_contribution(alpha, betas, tau) -> Float64

Saad's estimate of what the NEXT Krylov vector would add to `exp(tau*H)*v0`:

    beta_m * |[exp(tau*T_m)]_{m,1}|

with `T_m` the tridiagonal built from `alpha[1..m]` and `betas[1..m-1]`. `betas[m]` is the
residual that produced the vector just accepted.

⛔ THIS, NOT `beta`, IS THE CONVERGENCE MEASURE. `beta` alone is a BREAKDOWN detector: it says the
recursion could not produce another independent direction, which on a generic `H` simply does not
happen -- MEASURED at L=12, Heisenberg and XX return BIT-IDENTICAL results for a `beta` threshold
anywhere from `1e-6` to `1e-13`, because the test never fires and every bond runs to the cap. What
decides whether a vector MATTERS is how much weight the propagator actually puts on it, and that
is the product above: `beta_m` can be O(1) while `[exp(tau*T_m)]_{m,1}` is `~(tau*||H||)^m/m!`,
which is what makes a short basis sufficient at small `tau` and a long one necessary at large.

`T_m` is `m x m` with `m` at most `krylov_basis`, so the dense `exp` here is negligible against
one application of `H` to a bond tensor.
"""
function _kry_contribution(H::Matrix{ComplexF64}, betas::Vector{Float64}, m::Int,
                          tau::ComplexF64)
    m == 0 && return Inf
    length(betas) >= m || return Inf
    E = exp(tau * @view H[1:m, 1:m])
    return abs(betas[m]) * abs(E[m, 1])
end

"""
    _krylov_frame(v0, applyH, maxdepth, tol, tau) -> Vector

The ORTHONORMAL Krylov vectors `{v0, v1, ...}` of `applyH` seeded at `v0`, stopped when the next
vector's CONTRIBUTION to `exp(tau*H)*v0` falls below `tol` (see [`_kry_contribution`](@ref)), when
the recursion breaks down, or when `maxdepth` is reached.

⛔ NO SVD PER ITERATION. The vectors are accumulated whole and split ONCE by the caller. A
selection inside the loop -- which is what the REMOVED `cbe_lubich_sweep`'s `grow_iters` does, re-running the
CBE ranking on every pass -- re-truncates by budget at each step and throws away exactly the
directions the next application of `H` needs: MEASURED on OAT it bought 1.8x and then saturated
flat (2.13e-02 -> 1.16e-02) no matter how many passes were allowed.

⛔ FULL RE-ORTHOGONALISATION IS NOT OPTIONAL, and this is where a plain power iteration fails.
`H^k v` converges to the dominant eigenvector, so by `k ~ 4` the raw powers are numerically
parallel and the closing SVD deletes them as dependent -- the basis would saturate for a
CONDITIONING reason that looks exactly like a physical one. Two passes, against the whole basis:
one pass of modified Gram-Schmidt loses orthogonality once the vectors start to align.

⚠ The projection coefficients are taken DURING the orthogonalisation and accumulated across both
passes, so column `k` of `Hm` is the true projection of `H v_k` onto the basis. `Hm[k,k]` is then
the Rayleigh quotient `<v_k, H v_k>` and is kept COMPLEX -- for a Hermitian `H` its imaginary part
is roundoff and the Hessenberg reduces to the tridiagonal, so closed-system numbers are unchanged.
"""
function _krylov_frame(v0, applyH, maxdepth::Int, tol::Float64, tau::ComplexF64)
    b0 = norm(v0)
    b0 <= 0 && return Any[v0]
    vs = Any[to_concrete((1.0 / b0) * v0)]
    betas = Float64[]
    # ⛔ THE FULL UPPER-HESSENBERG PROJECTION, NOT alpha/beta. The recursion already
    # orthogonalises against EVERY previous vector (two passes), so it is Arnoldi -- the earlier
    # version simply DISCARDED the off-diagonal coefficients and rebuilt a SYMMETRIC tridiagonal
    # from `real(<v,Hv>)`. That is correct only for `H = H'`; on a non-Hermitian generator it is
    # silently wrong and nothing errors. Keeping the coefficients costs an `m x m` matrix with
    # `m <= krylov_basis`, and for a Hermitian `H` the Hessenberg IS the tridiagonal, so the
    # closed-system numbers are unchanged bit for bit.
    Hm = zeros(ComplexF64, maxdepth + 1, maxdepth + 1)
    for k in 1:maxdepth
        w = applyH(vs[end])
        for _pass in 1:2, (i, u) in pairs(vs)
            c = tensor_inner(u, w)
            # accumulate across BOTH passes: the second pass's correction is part of the
            # projection, not a separate quantity
            Hm[i, k] += c
            w = to_concrete(w - c * u)
        end
        b = norm(w)
        # BREAKDOWN comes first and is unconditional: with no independent direction left there is
        # nothing for the contribution test to weigh, and `1/b` would not be finite.
        b <= _KRY_BREAKDOWN && break
        push!(betas, b)
        Hm[k + 1, k] = b
        push!(vs, to_concrete((1.0 / b) * w))
        tol > 0 && _kry_contribution(Hm, betas, k, tau) < tol && break
    end
    # The seed carries the state's own scale back in: the caller's SVD cutoff is relative to the
    # largest singular value, and a basis of unit vectors alone would let the closing truncation
    # rank the high-order directions against each other rather than against the state.
    vs[1] = to_concrete(b0 * vs[1])
    return vs
end

"""
    _HalfAcc

Everything ONE half-sweep accumulates, held per half-sweep rather than in closure variables.

⛔ THIS EXISTS BECAUSE THE HALF-SWEEPS CAN NOW RUN CONCURRENTLY. Counters shared between them
(`nmv`, `epre`, the phase timers) are read-modify-write from two tasks, so as closure `Ref`s they
are a data race -- and a racy `+=` on a Float64 does not crash, it silently loses increments, so
the failure would show up as a profile that quietly under-reports and never as an error.
"""
mutable struct _HalfAcc
    nmv::Int
    ngrow::Int
    epre::Float64
    efnl::Float64
    peak::Int
    tcbe::Float64
    tkry::Float64
    tsplit::Float64
    tframe::Float64
    tenv::Float64
end
_HalfAcc() = _HalfAcc(0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

"""
    CBEBugSweepInfo

What one [`cbe_bug_step!`](@ref) did.

  - `expanded` -- the bond dimension the basis update left at each bond, before the closing
    truncation. In MULTIPLETS under a non-abelian symmetry.
  - `n_new` -- directions CBE admitted per bond; an upper bound on what `expanded` could grow by.
  - `root`, `root_rank` -- which bond took the Galerkin step, and its kept rank afterwards.
  - `err_pre`, `err_fnl` -- largest preselection / final-selection weight discarded. `err_fnl`
    is the one to watch: a large value means the BUDGET, not the selection, limited accuracy.
  - `discarded` -- largest relative weight the closing truncation threw away.
  - `krylov_dims` -- operator applications, summed over the `L` K-steps and the one root solve.
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
    # ⛔ EXTRA `cbe_expand` PASSES (Jan's point 4), AND IT IS HERE BECAUSE `krylov_dims` CANNOT
    # SEE THEM. `krylov_dims` counts `apply_one_site` calls; a growth pass is a `cbe_expand`,
    # whose operator work happens inside `sketch_h_*` and is NOT counted there. Reporting a
    # growing sweep on `krylov_dims` alone would show its accuracy and hide its cost -- the same
    # blindness that made the retired `rexpand` look free. Zero when `krylov_grow = false`.
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
    t_kry::Float64      # building the Krylov frames (`apply_one_site` through `_krylov_frame`)
    t_root::Float64     # the root Galerkin solve (`expv` on the zero-site generator)
    t_trunc::Float64    # the closing `truncate_recursive!`
    # ⛔ THE PHASES THAT CLOSE THE ACCOUNTING. MEASURED at L=50, chi=64 with only the four timers
    # above: cbe 0.445 + kry 0.128 + root 0.003 + trunc 0.029 = 0.605, leaving ~40% UNATTRIBUTED.
    # A profile that does not add up cannot locate a bottleneck, and the missing block is bigger
    # than every phase except the expansion itself.
    #
    # ⚠ `t_split` AND `t_frame` ARE **ALSO SVDs**, and the randomised sketch does NOT touch them.
    # `_splitU`/`_splitV` factorise once per bond per half-sweep and `_frame_from` factorises BOTH
    # sides of a bond -- so the sweep can be SVD-bound while `cbe_expand` is under half of it, and
    # sketching only CBE would then be incapable of winning however good the sketch is.
    t_split::Float64    # `_splitU` / `_splitV`, the half-sweep frame splits
    t_frame::Float64    # `_frame_from`, which factorises both sides of the bond
    t_env::Float64      # environment stacks and channel pushes
    t_step::Float64     # the whole call, so the shares can be checked to sum to 1
end

"""
    cbe_bug_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

ONE step of the CBE-BUG sweep, in place. `tau = -im*dt` is real time and `tau = -dt` imaginary
time; nothing here cares which, and the caller renormalises.

THERE IS ONE SWEEP. Both half-sweeps are BASIS GENERATORS -- CBE widens each bond, the frame is
built from that bond's Krylov space, the environment is pushed through it, and no site tensor is
ever evolved. The single Galerkin solve at the root bond is the only evolution in the step.

  left -> root   `U_ex` becomes `W[i]`;  right -> root   `V_ex` becomes `Z[j]`

so every bond is widened from BOTH sides before the root sees it.

⛔ THE K-STEP, `kaug` AND `rexpand` ARE GONE (2026-08-24). They evolved a one-site tensor per bond
and then restored the width that the split cost. MEASURED at L=12, dt=0.05, against exact
references, this sweep at `krylov_basis = 3` beats that machinery on every model tested:

    model        this sweep              old K-step sweep        
    XX           1.6948e-04 @   289      1.6948e-04 @  2888   identical, 10x cheaper (m=0)
    Heisenberg   2.6697e-06 @  1038      2.7869e-06 @  3423   better, 3.3x cheaper
    OAT          7.0610e-14 @  1036      5.2449e-06 @  3740   8 orders better, 3.6x cheaper

`git log` holds the old path; `docs/PLAN-tolerance-control.md` holds the measurements that
retired it, including the three hypotheses that were wrong on the way.

Keywords beyond the [`cbe_expand`](@ref) ones (`dex`, `growth`, `dover`, `comp_ratio`,
`sulz_cap`, `preselect_only`, `exact`, `stol_pre`, `stol_fnl`):

  - `krylov_tol` -- **THE STOPPING RULE, `1e-6`.** The half-sweep stops adding Krylov vectors when
    the next one's CONTRIBUTION to `exp(tau*H_1)*Theta` falls below it -- Saad's estimate
    `beta_m*|[exp(tau*T_m)]_{m,1}|`, see [`_kry_contribution`](@ref). `0.0` disables the test and
    runs to the cap. Pairs with `trunc_thresh` the way `tau_Krylov = tau_trunc/10` does.
  - `krylov_basis` -- the CAP on that iteration, default `30`.

    ⚠ **THE CRITERION USED TO BE `beta` ALONE AND THAT DID NOT WORK.** MEASURED at L=12 before the
    change: Heisenberg and XX returned BIT-IDENTICAL results for a `beta` threshold anywhere from
    `1e-6` to `1e-13` -- the test never fired and every bond ran to the cap. `beta` is a BREAKDOWN
    detector: it reports that no independent direction is left, which on a generic `H` does not
    happen. It survives here only as the `_KRY_BREAKDOWN` guard, where it belongs.

    MEASURED WITH THE CONTRIBUTION RULE, L=12, dt=0.05, T=1.0, against exact references
    (`benchmarks/krylov_stopping.jl`; `krylov` = accumulated operator applications):

        arm            XX err / krylov        Heisenberg err / krylov   OAT err / krylov
        cap, no test   1.6948e-04 /  1770     1.0800e-06 /  2280        5.9952e-15 / 2158
        tol 1e-4       1.6948e-04 /   952     2.4032e-06 /  1384        1.6431e-14 / 1552
        tol 1e-6  DEF  1.6948e-04 /  1180     1.4748e-06 /  1622        1.0880e-14 / 1766
        tol 1e-10      1.6948e-04 /  1550     1.1190e-06 /  1976        6.2172e-15 / 2000
        pinned m=3     1.6948e-04 /   950     2.6697e-06 /  1038        7.0610e-14 / 1036
        pinned m=0     1.6948e-04 /   289     8.8424e-05 /   308        2.1329e-02 /  313

    Read three things off it. **XX: the rule cuts 1.9x for free** -- bit-identical error at every
    tolerance, so it correctly finds that depth buys nothing here. **OAT: the rule refuses to cut**
    -- every tolerance stays at ~1e-14 where `m = 0` is 2.1e-02, which is the safety property that
    matters, since this is the model where depth is worth eight orders. (The apparent ordering
    among the OAT tolerance rows is roundoff against roundoff and means nothing.)
    **Heisenberg: it is a genuine dial, not a free lunch** -- `1e-6` buys 0.71x the cost for 1.37x
    the error, and `1e-10` is nearly free at 1.04x error for 0.87x cost. That monotone trade is
    the deliverable; `beta` gave one point and no dial at all.

    ⚠ **`1e-6` WAS CHOSEN WHILE THIS KNOB WAS INERT**, so the number predates the rule that now
    reads it. It is kept as the default deliberately, but a caller who wants the accurate end
    should use `1e-10` rather than assume the default is conservative.

    Set `krylov_basis` to a small fixed number (2-3) and `krylov_tol = 0.0` to pin the depth.

    ⛔ **NOT CONSIDERING ENOUGH KRYLOV VECTORS IS A REAL AND EASILY-MISSED SOURCE OF ERROR.**
    `krylov_basis = 0` uses `cbe_expand`'s frame alone -- ONE power of `H` -- and that is enough
    on XX and badly short on Heisenberg and OAT. The failure does NOT look like a basis problem:
    it survives more budget, more rank, and turning the closing truncation off entirely, because
    the basis is short of the right DIRECTIONS rather than of room. `m = 1` reproduces `m = 0`
    exactly (CBE already spans one power), so the first genuine addition is `m = 2`.

    ⛔ ONE SVD, AT THE END -- see [`_krylov_frame`](@ref). Selecting inside the loop (which is what
    the REMOVED `cbe_lubich_sweep`'s `grow_iters` does) re-truncates by budget every pass and discards exactly
    what the next application of `H` needs: measured 1.8x and then flat on OAT.
  - `root` -- which bond takes the Galerkin step. Default `L/2`, a fixed point of the reflection
    `b -> L - b` on an even chain, so the sweep is already spatially symmetric there.
  - `split_cutoff`, `split_maxdim` -- truncation on the frame split, applied once per bond as the
    sweep passes.
  - `root_cutoff`, `root_maxdim` -- truncation on the ROOT CORE SPLIT. **Defaults `0.0`/`0`, i.e.
    NONE**: that SVD splits the Galerkin core into the two centre tensors and nothing more.
    There are exactly TWO truncations in a step -- the half-sweep split and the closing pass --
    and this used to be an unswitchable third.
  - `maxdim`, `trunc_thresh` -- the closing [`truncate_recursive!`](@ref). `truncate = false`
    disables it; with the root defaults above that leaves `split_cutoff` as the only rank control
    anywhere in the step, which is a study configuration and not a production one.
  - `maxiter`, `tol`, `reorth` -- Krylov budget for the root solve.
"""
function cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                       dex::Int = 0,
                       growth::Float64 = 2.0,
                       dover::Union{Nothing, Int} = nothing,
                       comp_ratio::Union{Float64, Nothing} = 1.0,
                       sulz_cap::Bool = false,
                       preselect_only::Bool = false,
                       exact::Bool = false,
                       stol_pre::Float64 = 1e-10,
                       stol_fnl::Float64 = 1e-13,
                       root::Union{Nothing, Int} = nothing,
                       krylov_basis::Int = 30,
                       krylov_tol::Float64 = 1e-6,
                       krylov_grow::Bool = false,
                       # ⛔ `grow_iters` IS THE SINGLE-KNOB SPELLING OF THE GROWTH PATH, and it is
                       # this sweep's OWN mechanism -- not an import from the retired
                       # `cbe_lubich_sweep`. `grow_iters = n` means `krylov_grow = true` with `n`
                       # growth passes; `grow_iters = 0` is the ordinary one-SVD Krylov frame.
                       # The count was always there, spelled as two parameters: `_grow_frame`
                       # loops `for k in 1:krylov_basis`, so `krylov_basis` IS the pass count once
                       # `krylov_grow` is on. Two knobs meaning one thing is how a scan ends up
                       # reporting a growth study that never grew.
                       #
                       # ⛔ THE TWO PATHS ARE NOT THE SAME ALGORITHM, so do not read `grow_iters`
                       # as lubich's. There the pass RE-RUNS THE CBE SELECTION, re-truncating by
                       # budget and discarding exactly what the next application of `H` needs --
                       # MEASURED on OAT it bought 1.8x and then went flat (2.13e-02 -> 1.16e-02)
                       # however many passes it was allowed. Here the rule is ACCUMULATE, NEVER
                       # RE-RANK: nothing already accepted is touched and the space only grows.
                       # ⚠ Its cost is invisible in `krylov_dims` (each pass is a `cbe_expand`,
                       # whose operator work lives in `sketch_h_*`); `n_grow` counts the passes
                       # for exactly that reason.
                       grow_iters::Union{Nothing, Int} = nothing,
                       split_cutoff::Float64 = 1e-14,
                       split_maxdim::Int = 0,
                       truncate::Bool = true,
                       root_cutoff::Float64 = 0.0,
                       root_maxdim::Int = 0,
                       maxdim::Int = 200,
                       trunc_thresh::Float64 = 1e-12,
                       maxiter::Int = 30,
                       tol::Float64 = 1e-15,
                       reorth::Bool = true,
                       # ⛔ THE GALERKIN SOLVE IS THE ONE THAT EVOLVES THE STATE. The half-sweeps
                       # only supply DIRECTIONS -- a basis that is short there costs accuracy in
                       # the frame, but the root `expv` is what actually applies `exp(tau*H_0)`,
                       # so its depth is the one that limits the step. `maxiter` alone cannot
                       # express that, because `lanczos_expv`'s exit is BREAKDOWN-only: it burns
                       # the whole budget and returns the projection whether or not it converged.
                       #
                       #   `root_conv_tol > 0`  stop on Saad's contribution estimate instead --
                       #                        a real convergence test.
                       #   `root_substeps = 0`  split `tau` adaptively, so a large step needs no
                       #                        larger `maxiter` AND the exponent stays
                       #                        representable. In imaginary time `exp(tau*lambda)`
                       #                        at `tau = -50`, `lambda ~ -20` is `exp(1000)` --
                       #                        past the Float64 ceiling, which is how a too-large
                       #                        step returned `obs = 0` with no error raised.
                       #
                       # Both default to the historical behaviour so nothing already measured
                       # moves; see `BondUpdateBUG/expv.jl` for the full argument.
                       root_conv_tol::Float64 = 0.0,
                       root_substeps::Int = 1,
                       # Build the bases from THIS state instead of from `psi`, while still
                       # evolving `psi` over the full `tau`. `nothing` is the ordinary step.
                       # Exists for [`cbe_bug_midpoint_step!`](@ref); see the comment at `psiL`.
                       frame_state = nothing,
                       # ⛔ `hermitian = false` SWITCHES THE ROOT SOLVE TO ARNOLDI. Lanczos on a
                       # non-Hermitian generator -- an open system's Lindbladian or a
                       # non-Hermitian H_eff -- returns a plausible vector and no error at all.
                       # The half-sweep basis needs no flag: it already builds the full
                       # upper-Hessenberg projection, which is the tridiagonal when H = H'.
                       hermitian::Bool = true,
                       # ⛔ RUN THE TWO HALF-SWEEPS CONCURRENTLY. They share no data: the left
                       # reads the state canonical at 1 and writes `W`, the right reads it
                       # canonical at `L` and writes `Z`, and neither reads the other's output.
                       # The single Galerkin solve that needs both runs after both.
                       #
                       # ⚠ NEEDS `julia -t 2`. On one thread the tasks interleave on one worker
                       # and this is the sequential step plus scheduling overhead -- a small
                       # pessimisation, not a no-op, which is why it is off by default.
                       #
                       # ⚠ THE RESULT IS NOT BIT-IDENTICAL TO THE SEQUENTIAL ARM, and that is
                       # expected rather than a defect: each sweep now re-gauges its OWN copy of
                       # the state, so `canonical!(psiL, 1)` runs from the incoming centre instead
                       # of from wherever the other sweep left it. Same spaces, different
                       # floating-point path. `tests/rsvd_cbe/` pins the agreement at 1e-10.
                       parallel::Bool = false,
                       # ⚠ AN A/B FOR ONE PERFORMANCE FIX, NOT A TUNING KNOB. `false` restores the
                       # pre-2026-08-27 behaviour where each of `cbe_expand`'s three sketch calls
                       # rebuilt `H*Theta` from scratch. Results are IDENTICAL either way -- only
                       # the seconds move -- so the only reason to set it is to measure how much
                       # the sharing bought, which `benchmarks/rsvd_parallel.jl` does.
                       share_ht::Bool = true,
                       rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("cbe_bug_step! needs at least two sites"))
    c = root === nothing ? max(1, min(L - 1, L ÷ 2)) : root
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))


    # ── `grow_iters` RESOLVES ONTO THE TWO KNOBS IT IS SHORTHAND FOR ────────────────────────────
    # `nothing` leaves `krylov_grow`/`krylov_basis` exactly as passed, so every existing call and
    # every recorded number is untouched. An explicit value WINS over both, and contradicting them
    # is refused rather than silently resolved -- `grow_iters = 4, krylov_grow = false` is a caller
    # who believes something false about the run they are about to do, and a scan that quietly
    # picked one of the two would report a growth study that never grew.
    if grow_iters !== nothing
        grow_iters >= 0 || throw(ArgumentError("grow_iters must be >= 0, got $grow_iters"))
        (krylov_grow && grow_iters == 0) && throw(ArgumentError(
            "grow_iters = 0 disables the growth path but krylov_grow = true was also passed"))
        krylov_grow = grow_iters > 0
        krylov_grow && (krylov_basis = grow_iters)
    end

    cut = max(trunc_thresh, 1e-14)
    # `krylov_tol = 0` falls back to `tau_trunc/10`; the DEFAULT is an explicit `1e-6`.
    ktol = krylov_tol > 0 ? krylov_tol : cut / 10

    # TRUNCATION AT THE HALF-SWEEP SPLIT. Historically this was a hard `cutoff = 1e-14` with no
    # `Nkeep` at all, so the half-sweeps pruned essentially nothing and every rank decision was
    # deferred to the root solve and `truncate_recursive!`. `tdvp_cbe1s` instead splits with
    # `cutoff = cut, Nkeep = maxdim` at every site. These two closures make that difference a
    # knob rather than a hard-coded constant; the defaults reproduce the historical behaviour
    # exactly, so `split_cutoff = 1e-14, split_maxdim = 0` is a no-op.
    _splitU(T) = split_maxdim > 0 ?
        to_concrete(svd(T, (1, 2); cutoff = split_cutoff, Nkeep = split_maxdim).U) :
        to_concrete(svd(T, (1, 2); cutoff = split_cutoff).U)
    _splitV(T) = split_maxdim > 0 ?
        to_concrete(svd(T, (1,); cutoff = split_cutoff, Nkeep = split_maxdim).Vd) :
        to_concrete(svd(T, (1,); cutoff = split_cutoff).Vd)

    expanded = zeros(Int, L - 1)
    # ⚠ ONE ARRAY PER HALF-SWEEP: bond `c` is written by BOTH (`n_new_l` from the left sweep,
    # `n_new_r` from the right), so a single shared array is a genuine write conflict there --
    # benign when the sweeps ran in sequence and last-writer-wins, undefined when they run
    # concurrently. Merged after the join with the RIGHT sweep winning at `c`, which is exactly
    # what the sequential order used to produce.
    n_newL   = zeros(Int, L - 1)
    n_newR   = zeros(Int, L - 1)
    accL = _HalfAcc()
    accR = _HalfAcc()

    # `rmax` read ONCE, globally, before anything widens.
    # `stol_pre` / `stol_fnl` are THREADED THROUGH rather than left to `cbe_expand`'s defaults.
    # They are the only truncations that exist inside the half-sweep besides the frame split -- there
    # is no `_splitU` on that path at all -- so leaving them at a hardcoded 1e-10 / 1e-13 made the
    # basis-only sweep ungovernable by any tolerance the caller sets, and made `split_cutoff` look
    # like the half-sweep's knob when for that sweep it is not used at all.
    exkw = (dex = dex, growth = growth, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact,
            stol_pre = stol_pre, stol_fnl = stol_fnl, share_ht = share_ht, rng = rng)

    troot = Ref(0.0); ttrunc = Ref(0.0)
    _t0step = time_ns()
    """Time one `cbe_expand` into the calling half-sweep's own accumulator."""
    function timed_expand(acc::_HalfAcc, args...; kw...)
        t0 = time_ns()
        ex = cbe_expand(args...; kw...)
        acc.tcbe += (time_ns() - t0) / 1e9
        return ex
    end
    function record!(acc::_HalfAcc, nn, ex, i, side::Symbol)
        acc.epre = max(acc.epre, ex.err_pre)
        acc.efnl = max(acc.efnl, ex.err_fnl)
        nn[i] = side === :left ? ex.n_new_l : ex.n_new_r
        return ex
    end

    """
        _grow_frame(v0, Vopp, side, i, lch, rch) -> (blocks, Vopp_final)

    Jan's point 4: RE-EXPAND THE BOND BETWEEN LANCZOS STEPS instead of once per bond.

    The fixed-width builder [`_krylov_frame`](@ref) widens the bond once, builds `H1` against that
    frame and runs the whole recursion inside it -- so any part of `H^k*Theta` lying outside the
    initial expansion is projected away at every step. Here the OPPOSITE frame is re-expanded
    after each accepted vector, the environment and `H1` are rebuilt against it, and the vectors
    already accumulated are EMBEDDED into the widened space rather than recomputed.

    ⛔ THE EMBEDDING IS AN OVERLAP, NOT AN ASSUMPTION. `cbe_expand` returns
    `V_ex = oplus([V0, TRC])` where `V0` is its own factorisation of what was handed in, which
    need not be the tensor that went in -- the orthonormality guard may rotate within the appended
    span. Computing `PR` as the true overlap `<V_old, V_new>` is correct under that rotation;
    assuming `[I 0]` is not. (Same reasoning as `_expanding_krylov`'s `embed_ops`.)

    ⛔ ACCUMULATE, NEVER RE-RANK. Jan's bookkeeping is "new kept = old kept PLUS expanded", and
    that word is load-bearing: the REMOVED `cbe_lubich_sweep`'s `grow_iters` re-runs the CBE SELECTION each
    pass, which re-truncates by budget and discards exactly what the next application of `H`
    needs -- MEASURED on OAT it bought 1.8x and then went flat. Nothing here touches the vectors
    already accepted; the space only ever grows.

    ⚠ COST DOES NOT APPEAR IN `krylov_dims`. Each pass is a `cbe_expand`, whose operator work is
    inside `sketch_h_*`. `ngrow` counts the passes for exactly this reason -- see the note on
    `CBEBugSweepInfo.n_grow`.
    """
    function _grow_frame(acc::_HalfAcc, v0, Vopp, side::Symbol, i::Int, lch, rch,
                         dex_pass::Int)
        # ⚠ THE TWO SIDES ARE NOT MIRROR IMAGES IN LEG INDEX, and getting this backwards is silent:
        # the shapes still contract, they just contract the wrong leg.
        #   :left   opposite frame is `V_ex = (g, site, link)`  -> its bond leg is 1
        #           the Krylov vectors are `(link, site, g)`    -> their bond leg is 3
        #   :right  opposite frame is `U_ex = (link, site, g)`  -> its bond leg is 3
        #           the Krylov vectors are `(g, site, link)`    -> their bond leg is 1
        leg = side === :left ? 1 : 3                  # the OPPOSITE frame's bond leg
        b0 = norm(v0)
        b0 <= 0 && return (Any[v0], Vopp)
        V  = Vopp
        vs = Any[to_concrete((1.0 / b0) * v0)]
        betas = Float64[]
        # ⛔ THE SAME UPPER-HESSENBERG PROJECTION AS [`_krylov_frame`](@ref), for the same reason:
        # `real(<v,Hv>)` plus a symmetric tridiagonal is correct only for `H = H'`. This loop
        # already orthogonalises against every previous vector, so it is Arnoldi and the
        # coefficients are simply kept. Leaving the old `alpha`/`betas` pair here would ALSO have
        # been a hard error rather than a silent one -- `_kry_contribution` no longer has a
        # three-argument method -- but only on the `krylov_grow = true` + `ktol > 0` path, which
        # nothing in the suite exercised.
        Hm = zeros(ComplexF64, krylov_basis + 1, krylov_basis + 1)
        for k in 1:krylov_basis
            H1 = side === :left ?
                 one_site_h(mpo, i, lch, push_right_channels(rch, mpo, V, i + 1)) :
                 one_site_h(mpo, i + 1, push_left_channels(lch, mpo, V, i), rch)
            acc.nmv += 1
            w = apply_one_site(H1, vs[end])
            # ⛔ EVERY NAME ASSIGNED HERE MUST BE ONE `cbe_bug_step!`'s BODY DOES NOT USE. This is a
            # nested closure, so an assignment to a name that is already a local of the enclosing
            # function does NOT shadow it -- it REBINDS IT. `ov` was written `c` and silently
            # overwrote the ROOT BOND INDEX (line ~387) with a Gram-Schmidt overlap, so the whole
            # `krylov_grow`/`grow_iters` path died on `i < c` with `isless(::Int64, ::ComplexF64)`
            # -- and would have evolved about the wrong root had the types happened to compare.
            # The loop index is `t` and not `i` for exactly this reason; `c` was the one missed.
            for _pass in 1:2, (t, u) in pairs(vs)
                ov = tensor_inner(u, w)
                Hm[t, k] += ov
                w = to_concrete(w - ov * u)
            end
            b = norm(w)
            b <= _KRY_BREAKDOWN && break
            push!(betas, b)
            Hm[k + 1, k] = b
            push!(vs, to_concrete((1.0 / b) * w))
            ktol > 0 && _kry_contribution(Hm, betas, k, tau) < ktol && break
            k == krylov_basis && break

            # ── expand against the NEWEST direction, then embed everything into the result ──
            #
            # ⛔ NOT `_frame_from`. That helper factorises BOTH sides and joins them through
            # `res_l.S * res_l.Vd`, which needs the two bond legs to meet as an ordinary pair. On
            # the ALREADY-EXPANDED bond they do not: the Krylov vector's bond leg was produced by
            # contracting against `V_ex'`, so `S` comes back 0-dimensional and that product has
            # nothing to contract (`AssertionError: No matching contractible indices`).
            #
            # It is also unnecessary work. One side here is ALREADY an isometry -- it is the
            # opposite CBE frame -- so only the other side needs factorising, and the core is then
            # a single overlap rather than a product of four SVD outputs.
            # `spl`/`core`, not `res`/`S0`: both of those ARE locals of the enclosing
            # `cbe_bug_step!` (the root split and the root core), and this closure would rebind
            # them rather than shadow them -- see the note on `ov` above. They survive today only
            # because the body happens to rewrite both after the sweeps and before reading them,
            # which is an ordering no signature enforces.
            fr = if side === :left
                spl  = svd(vs[end], (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
                U0   = to_concrete(spl.U)
                core = to_concrete(contract(U0', (1, 2), vs[end], (1, 2)))     # (bU, g)
                BondFrame(U0, core, V,
                          vs[end].inds[1], vs[end].inds[2], vs[end].inds[3],
                          V.inds[2], V.inds[3],
                          leg_dim(U0, 3), copy(vs[end].spaces[3]))
            else
                spl  = svd(vs[end], (1,), "bV,L", "bV,R"; cutoff = 0.0)
                V0   = to_concrete(spl.Vd)
                core = to_concrete(contract(vs[end], (2, 3), V0', (2, 3)))     # (g, bV)
                BondFrame(V, core, V0,
                          V.inds[1], V.inds[2], V.inds[3],
                          vs[end].inds[2], vs[end].inds[3],
                          leg_dim(V, 3), copy(V.spaces[3]))
            end
            # ⛔ AN EXPLICIT `dex` PER PASS, OR THIS WHOLE BRANCH IS A NO-OP. `cbe_expand`'s
            # default budget is `ceil(growth*dmax) - r`, and the FIRST expansion already took the
            # bond to `growth*dmax` -- so on every later pass `r` has grown, the budget collapses
            # to zero, and the call returns the frame it was given. MEASURED before this line
            # existed: 178-238 expansion passes per run that changed the error by NOTHING (XX,
            # Heisenberg and OAT all bit-identical to `grow = false`). Each pass gets its own
            # allowance instead, so "old kept PLUS expanded" can actually add something.
            exk = timed_expand(acc, fr, mpo, i, lch, rch; exkw..., dex = dex_pass)
            Vn = side === :left ? exk.V_ex : exk.U_ex
            # `ngrow` counts ACTUAL WIDENINGS, not calls. Counting calls made a no-op look like
            # work and let the smoke test's "did it grow?" gate pass on a branch that grew nothing.
            if leg_dim(Vn, leg) > leg_dim(V, leg)
                acc.ngrow += 1
                P = side === :left ?
                    to_concrete(contract(V, (2, 3), Vn', (2, 3))) :      # (old_g, new_g)
                    to_concrete(contract(Vn', (1, 2), V, (1, 2)))       # (new_g, old_g)
                for t in eachindex(vs)
                    vs[t] = side === :left ?
                            to_concrete(contract(vs[t], (3,), P, (1,))) :
                            to_concrete(contract(P, (2,), vs[t], (1,)))
                end
                V = Vn
            end
        end
        vs[1] = to_concrete(b0 * vs[1])
        return (vs, V)
    end

    # ⛔ PEAK IS TRACKED PER HALF-SWEEP AND SUMMED, NOT SCANNED ACROSS ALL SIX ARRAYS.
    # The old version summed `W, Z, Vwide, Uwide, Wcbe, Zcbe` on every call, which (a) reads
    # arrays the OTHER half-sweep is concurrently writing -- `isassigned` racing a `setindex!` --
    # and (b) is `O(L)` tensor walks per bond, i.e. `O(L^2)` per step of pure diagnostic work.
    # Each sweep now accumulates its own frames incrementally.
    #
    # ⚠ THE REPORTED NUMBER IS THEREFORE AN UPPER BOUND on the old one: both halves' frames are
    # counted as simultaneously live. Under `parallel = true` that is literally true, and it keeps
    # the two arms' working sets comparable, which is the point of the column.
    function peak!(acc::_HalfAcc, held::Int, ts...)
        acc.peak = max(acc.peak, held + sum(tensor_elements(t) for t in ts; init = 0))
        return nothing
    end

    W = Vector{Any}(undef, c)          # augmented LEFT frames,  sites 1 … c
    Z = Vector{Any}(undef, L - 1)      # augmented RIGHT frames, Z[j] absorbs site j+1

    Vwide = Vector{Any}(undef, c)                # widened RIGHT frame, bonds 1 … c
    Uwide = Vector{Any}(undef, L - 1)        # widened LEFT  frame, bonds c … L-1
    # The CBE frame. It IS the basis at `krylov_basis = 0`, and the seed the Krylov space grows
    # into the basis) -- so it is stored whenever the basis is going to reference it.
    # THE SWEEP IS INTERLEAVED: CBE expands bond `i`, then bond `i`'s frame is built and the
    # environment pushed through it immediately -- the reference's own ordering
    # (TDVPSweepCBE1Si.m:30/:34), so the environment chain runs through the NEW frames and bond
    # `i+1`'s selection sees the widening at bond `i`.
    Wcbe  = Vector{Any}(undef, c)
    Zcbe  = Vector{Any}(undef, L - 1)

    """
    Union `Wk` (the basis update's output) with `U0` (the CBE frame), on the LEFT.

    `U0` FIRST so the CBE directions are kept verbatim and only the part of `Wk` outside them is
    appended -- `_preselect_left` returns `nothing` when there is nothing to append, which is the
    ordinary case once the two already span the same space.
    """
    function _union_left(U0, Wk)
        Q, _ = _preselect_left(U0, Wk, 1e-14)
        Q === nothing && return U0
        return to_concrete(oplus([U0, to_concrete(setitag(Q, 3, U0.inds[3].itags))], (3,)))
    end
    "Row mirror of [`_union_left`](@ref): the bond leg is 1 and the direct sum is over it."
    function _union_right(V0, Zk)
        Q, _ = _preselect_right(V0, Zk, 1e-14)
        Q === nothing && return V0
        return to_concrete(oplus([V0, to_concrete(setitag(Q, 1, V0.inds[1].itags))], (1,)))
    end

    # ── THE TWO HALF-SWEEPS ARE INDEPENDENT, AND THAT IS NOW EXPLOITED ────────────────────
    #
    # ⛔ THEY WERE NEVER COUPLED. The left sweep reads the state canonical at 1 and writes
    # `W / Vwide / Wcbe`; the right reads it canonical at `L` and writes `Z / Uwide / Zcbe`.
    # Neither reads the other's output -- the only meeting point is the SINGLE Galerkin solve at
    # the root, which needs both and runs after both. The sequential order was an artefact of
    # re-gauging ONE state object in place, not a dependence in the method.
    #
    # ⚠ SO THE LEFT SWEEP GETS ITS OWN COPY. `copy(psi)` copies the tensor VECTOR only, and every
    # mutation in this package REPLACES a slot rather than writing into a tensor, so the two
    # objects can be re-gauged independently while sharing the tensors neither of them modifies.
    #
    # ⛔ AND `psi` ITSELF STAYS THE `L`-CANONICAL ONE, deliberately: `AL`, `AR` and the whole
    # assembly below read `psi`, and they must all see ONE gauge. Mixing `C` (built in the
    # 1-canonical gauge) with `D` (built in the L-canonical one) would contract two different
    # bases at link `c+1` and is silently wrong -- which is why `AL` is still recomputed here
    # against `psi` and only `AR` can be taken straight from the right sweep.
    # ⛔ `frame_state` DECOUPLES *WHERE THE BASES COME FROM* FROM *WHAT IS BEING EVOLVED*, which
    # is the one structural thing the midpoint BUG integrator (arXiv:2402.08607) needs and the
    # standard step does not. There the bases come from a HALF-STEP state `Y_{1/2}` while the
    # Galerkin ODE is still integrated over the FULL step starting from the ORIGINAL `Y_0`.
    #
    # ⛔ AND THE START VECTOR IS THE TRAP. `S0` is `<frames | state>`; feeding the half-step state
    # into BOTH would integrate the second half twice and quietly produce a first-order scheme
    # that still looks convergent. `AL` is already recomputed against `psi` below, but `AR` is
    # taken from the right sweep's CARRY -- so with a separate frame state that carry projects the
    # WRONG state, and `AR` has to be rebuilt against `psi` too. See the `AR` block below.
    #
    # When `frame_state === nothing`, `psiR` IS `psi` (aliased, not copied) and every line below is
    # the historical one.
    psiL = copy(frame_state === nothing ? psi : frame_state)
    psiR = frame_state === nothing ? psi : copy(frame_state)

    lenv_root = boundary_channels(mpo)   # left environment at link c, in the NEW basis
    renv_root = boundary_channels(mpo)   # right environment at link c+2, in the NEW basis
    AR = nothing                         # the right sweep's carry IS `AR`; see below

    # ── PHASE B(a). LEFT -> ROOT. Writes `W`, `Vwide`, `Wcbe`, `expanded[1:c]`, `n_newL`. ──
    function half_left!()
        canonical!(psiL, 1)
        rstack = (t0 = time_ns(); e = right_env_stack(psiL, mpo; downto = 3);
                  accL.tenv += (time_ns() - t0) / 1e9; e)
        lch = boundary_channels(mpo)
        C = nothing
        held = 0                          # frames this sweep has accumulated, running total
    for i in 1:c
        K = C === nothing ? psiL[i] : to_concrete(contract(C, (2,), psiL[i], (1,)))
        _fr = (t0 = time_ns(); f = _frame_from(K, psiL[i + 1]);
               accL.tframe += (time_ns() - t0) / 1e9; f)
        ex = record!(accL, n_newL, timed_expand(accL, _fr, mpo, i, lch,
                                right_channels(rstack, i + 2); exkw...), i, :left)
        peak!(accL, held + _state_stored(psiL), ex.U_ex, ex.V_ex)
        Vwide[i] = ex.V_ex
        Wcbe[i]  = ex.U_ex
        held += tensor_elements(ex.U_ex) + tensor_elements(ex.V_ex)
        if krylov_basis > 0
            # ── BASIS FROM THE KRYLOV SPACE, NOT FROM ONE APPLICATION OF `H` ───────────────
            #
            # `krylov_basis = 0` builds the frame from `cbe_expand` alone, i.e. from
            # `span{Theta, H*Theta}` -- ONE power of `H`. That is enough wherever `tau*||H_loc||`
            # is small (XX and Heisenberg: measured bit-identical to the shipped sweep on XX),
            # and badly short where it is not.
            #
            # Here the frame is instead `orth span{Kex, H1*Kex, ..., H1^m*Kex}`: real
            # applications of the one-site effective Hamiltonian, so the basis carries `m` Krylov
            # vectors rather than one. `m = 0` is the bare CBE frame; `m = 1` reproduces it exactly.
            #
            # The vectors are NORMALISED before the direct sum. Without that, `H1^k*Kex` grows
            # like `||H1||^k` and the SVD's relative `cutoff` -- which is measured against the
            # LARGEST singular value -- deletes the low-order vectors that carry the state
            # itself. `Kex` is left unnormalised so it keeps the state's own scale.
            M   = to_concrete(contract(psiL[i + 1], (2, 3), ex.V_ex', (2, 3)))
            Kex = to_concrete(contract(K, (3,), M, (1,)))
            blocks = if krylov_grow
                # PER-PASS BUDGET = what the FIRST expansion admitted on this bond. Keeping the
                # increment constant makes the space grow linearly in the Lanczos step, which is
                # what "old kept plus expanded" asks for; reusing `growth` would re-derive a
                # budget from the rank that expansion just raised, i.e. zero.
                bl, Vfin = _grow_frame(accL, Kex, ex.V_ex, :left, i, lch,
                                       right_channels(rstack, i + 2), max(ex.n_new_r, 1))
                Vwide[i] = Vfin          # accounting must see the width the growth reached
                bl
            else
                rch1 = push_right_channels(right_channels(rstack, i + 2), mpo, ex.V_ex, i + 1)
                H1 = one_site_h(mpo, i, lch, rch1)
                (t0 = time_ns();
                 bl = _krylov_frame(Kex, x -> (accL.nmv += 1; apply_one_site(H1, x)),
                                    krylov_basis, ktol, tau);
                 accL.tkry += (time_ns() - t0) / 1e9; bl)
            end
            peak!(accL, held + _state_stored(psiL), blocks...)
            stacked = length(blocks) == 1 ? blocks[1] : to_concrete(oplus(blocks, (3,)))
            W[i] = (t0 = time_ns(); r = to_concrete(setitag(_splitU(stacked), 3, _tagW(i)));
                accL.tsplit += (time_ns() - t0) / 1e9; r)
        else
            W[i] = Wcbe[i]                                          # reproduces cbe_lubich
        end
        expanded[i] = leg_dim(W[i], 3)
        held += tensor_elements(W[i])

        # (d) carry the ORIGINAL amplitude into the new basis. `K`, not `Kex`: `W[i]` touches
        # only legs 1-2, so this lands on `(b_i, link_i)` and composes with `psiL[i+1]`.
        C = to_concrete(contract(W[i]', (1, 2), K, (1, 2)))
        i == c && (lenv_root = lch)     # captured BEFORE pushing through the root frame
        i < c && ((lch = (t0 = time_ns(); e = push_left_channels(lch, mpo, W[i], i);
                 accL.tenv += (time_ns() - t0) / 1e9; e)))
    end
        return nothing
    end

    # ── PHASE B(b). RIGHT -> ROOT. Writes `Z`, `Uwide`, `Zcbe`, `expanded[c+1:end]`, `n_newR`.
    #
    # ⛔ ITS FINAL CARRY **IS** `AR`. The `D` recursion here and the `AR` loop that used to follow
    # the sweeps are the SAME recursion over the SAME tensors in the SAME gauge -- `psi` is not
    # touched between them -- so the second one was a redundant pass over half the chain,
    # contracting against the (widened) `Z` frames for a result already in hand. Deleted; `AR` is
    # now just the value `D` ends on.
    #
    # ⚠ `AL` GETS NO SUCH SHORTCUT and must still be recomputed below. The left sweep's own carry
    # `C` is built in the 1-canonical gauge while `AR` is in the L-canonical one, and the two meet
    # over link `c+1` -- different bases for the same link. Contracting them would be silently
    # wrong rather than an error.
    function half_right!()
        canonical!(psiR, L)
        lstack = (t0 = time_ns(); e = left_env_stack(psiR, mpo; upto = L - 2);
                  accR.tenv += (time_ns() - t0) / 1e9; e)
        rch = boundary_channels(mpo)
        D = nothing
        held = 0
    for j in (L - 1):-1:c
        B = D === nothing ? psiR[j + 1] :
            to_concrete(contract(psiR[j + 1], (3,), D, (1,)))
        _fr = (t0 = time_ns(); f = _frame_from(psiR[j], B);
               accR.tframe += (time_ns() - t0) / 1e9; f)
        ex = record!(accR, n_newR, timed_expand(accR, _fr, mpo, j,
                                left_channels(lstack, j), rch; exkw...), j, :right)
        peak!(accR, held + _state_stored(psiR), ex.U_ex, ex.V_ex)
        Uwide[j] = ex.U_ex
        Zcbe[j]  = ex.V_ex
        held += tensor_elements(ex.U_ex) + tensor_elements(ex.V_ex)
        if krylov_basis > 0
            # Row mirror of the K half-sweep's Krylov basis; see the comment there.
            M   = to_concrete(contract(ex.U_ex', (1, 2), psiR[j], (1, 2)))    # (b_L, link_j)
            Bex = to_concrete(contract(M, (2,), B, (1,)))
            blocks = if krylov_grow
                bl, Ufin = _grow_frame(accR, Bex, ex.U_ex, :right, j,
                                       left_channels(lstack, j), rch, max(ex.n_new_l, 1))
                Uwide[j] = Ufin
                bl
            else
                lch1 = push_left_channels(left_channels(lstack, j), mpo, ex.U_ex, j)
                H1 = one_site_h(mpo, j + 1, lch1, rch)
                (t0 = time_ns();
                 bl = _krylov_frame(Bex, x -> (accR.nmv += 1; apply_one_site(H1, x)),
                                    krylov_basis, ktol, tau);
                 accR.tkry += (time_ns() - t0) / 1e9; bl)
            end
            peak!(accR, held + _state_stored(psiR), blocks...)
            stacked = length(blocks) == 1 ? blocks[1] : to_concrete(oplus(blocks, (1,)))
            Z[j] = (t0 = time_ns(); r = to_concrete(setitag(_splitV(stacked), 1, _tagZ(j)));
                accR.tsplit += (time_ns() - t0) / 1e9; r)
        else
            Z[j] = Zcbe[j]
        end
        j != c && (expanded[j] = leg_dim(Z[j], 1))
        held += tensor_elements(Z[j])

        D = to_concrete(contract(B, (2, 3), Z[j]', (2, 3)))          # (link_j, b_j)
        j == c && (renv_root = rch)
        j > c && ((rch = (t0 = time_ns(); e = push_right_channels(rch, mpo, Z[j], j + 1);
                 accR.tenv += (time_ns() - t0) / 1e9; e)))
    end
        AR = D                          # (link_{c+1}, b_R) -- exactly the deleted `AR` loop
        return nothing
    end

    # ── RUN THEM ─────────────────────────────────────────────────────────────────────────
    #
    # ⚠ `parallel = true` NEEDS `julia -t 2` OR MORE TO DO ANYTHING. With one thread the tasks are
    # interleaved on the same worker and the step is the sequential one plus scheduling overhead,
    # so the flag is a pessimisation rather than a no-op -- which is why it is not the default.
    #
    # ⚠ AND IT COMPETES WITH BLAS. Two tasks each calling into a BLAS already configured for every
    # core will oversubscribe; Telum also passes `Threads.nthreads()` to HPTT for every
    # `permutedims`, so the nesting is three deep. `benchmarks/rsvd_parallel.jl` reports the
    # thread counts it ran under for exactly this reason.
    #
    # ⛔ WHAT MAKES THIS SAFE, CHECKED RATHER THAN ASSUMED:
    #   * `copy(psi)` copies the tensor VECTOR; every mutation in this package REPLACES a slot,
    #     so `canonical!` on one copy cannot be seen by the other.
    #   * Telum's `svd`/`contract` allocate their HPTT buffers PER CALL (`permute_mat` is a local),
    #     so there is no shared scratch to race on. Its only globals are the opt-in cost counters
    #     and the display settings.
    #   * `KRYLOV_LOG` IS a shared unsynchronised `Vector` -- but `expv` runs only in the ROOT
    #     solve, which is outside both tasks. The half-sweeps use `_krylov_frame`, which does not
    #     record. Moving an `expv` call into a half-sweep would make the log racy.
    #   * `_LOCAL_SPACES` is a lazily-filled global `Dict`; it is already populated by the time a
    #     state exists, so the sweeps only read it.
    if parallel
        tR = Threads.@spawn half_right!()
        half_left!()
        wait(tR)
    else
        half_left!()
        half_right!()
    end

    # `n_new` at bond `c` is written by BOTH sweeps; the right one wins, which is what the old
    # sequential order produced.
    n_new = vcat(n_newL[1:(c - 1)], n_newR[c:(L - 1)])
    epre = max(accL.epre, accR.epre)
    efnl = max(accL.efnl, accR.efnl)

    # ⛔ A SEPARATE `frame_state` MEANS THE RIGHT SWEEP NEVER TOUCHED `psi`, and two things the
    # default path got for free now have to be done explicitly:
    #
    #   1. GAUGE. `half_right!` used to leave `psi` L-canonical, and `AL`/`AR` below must both be
    #      contracted in ONE gauge -- the file's own warning at the sweep is that mixing two is
    #      "silently wrong rather than an error".
    #   2. `AR`. The default path takes it from the right sweep's carry `D`, which is
    #      `<Z | frame_state>`. For the midpoint scheme the Galerkin step must START FROM `Y_0`,
    #      so it needs `<Z | psi>`. Reusing the carry would integrate the first half-step twice
    #      and quietly reduce the method to first order while still converging -- the failure
    #      that would be hardest to spot afterwards.
    if frame_state !== nothing
        canonical!(psi, L)
        AR = nothing
        for j in (L - 1):-1:c                    # mirrors `half_right!`'s `D` recursion on `psi`
            B = AR === nothing ? psi[j + 1] :
                to_concrete(contract(psi[j + 1], (3,), AR, (1,)))
            AR = to_concrete(contract(B, (2, 3), Z[j]', (2, 3)))     # (link_j, b_j)
        end
    end

    AL = nothing
    for i in 1:c
        T = AL === nothing ? psi[i] : to_concrete(contract(AL, (2,), psi[i], (1,)))
        AL = to_concrete(contract(W[i]', (1, 2), T, (1, 2)))         # (b_L, link_{i+1})
    end
    S0 = to_concrete(contract(AL, (2,), AR, (1,)))                   # (b_L, b_R)
    peak = accL.peak + accR.peak +
           sum(tensor_elements(t) for t in (W[c], Z[c], S0); init = 0)

    H0 = zero_site_h(mpo, c, lenv_root, renv_root, W[c], Z[c])
    nroot = Ref(0)
    _t0root = time_ns()
    S1 = expv(x -> (nroot[] += 1; apply_zero_site(H0, x)), tau, S0;
              hermitian = hermitian, maxiter = maxiter, tol = tol, reorth = reorth,
              conv_tol = root_conv_tol, substeps = root_substeps)
    troot[] = (time_ns() - _t0root) / 1e9

    # ⛔ THE ROOT CORE SPLIT DOES NOT TRUNCATE BY DEFAULT. This SVD exists to SPLIT `S1` into
    # the two site tensors below; cutting here was a third, redundant truncation on top of the
    # half-sweep splits and the closing `truncate_recursive!`, and it was the only one with no
    # switch. Rank control now lives in exactly two places -- `split_cutoff` on the half-sweeps
    # and `trunc_thresh`/`maxdim` on the closing pass -- so a tolerance statement about the step
    # has two knobs to account for instead of three.
    #
    # `root_cutoff = 0.0` keeps every singular value with `sv > 0` (Telum's selector is relative
    # to `sv_max`), i.e. the split is lossless up to exact zeros; `root_maxdim = 0` means no
    # `Nkeep`. Both are still settable, so the old behaviour is an A/B (`root_cutoff = cut`,
    # `root_maxdim = maxdim`) rather than something that has been deleted.
    #
    # ⚠ WITH `truncate = false` AND THESE DEFAULTS THERE IS NO RANK CONTROL AT THE ROOT AT ALL,
    # which is deliberate -- it is what makes "root truncation off" a reachable arm -- but it
    # means the closing pass is the only thing enforcing `maxdim` in the shipped configuration.
    res = svd(S1, (1,); cutoff = root_cutoff,
              Nkeep = root_maxdim > 0 ? root_maxdim : nothing, get_lists = true)
    discarded = _trunc_weight(res)
    expanded[c] = leg_dim(W[c], 3)

    # ── 4. ASSEMBLE AND TRUNCATE ─────────────────────────────────────────────────────────

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

    if truncate
        _t0tr = time_ns()
        _, wtr = truncate_recursive!(psi, c + 1; maxdim = maxdim, cutoff = cut)
        ttrunc[] = (time_ns() - _t0tr) / 1e9
        discarded = max(discarded, wtr)
    end

    # ⚠ THE PHASE TIMERS ARE SUMS ACROSS BOTH HALF-SWEEPS, i.e. CPU time, while `t_step` is WALL
    # clock. Under `parallel = false` they are the same clock and the shares sum to ~1 as before.
    # Under `parallel = true` they DELIBERATELY DO NOT: two sweeps running at once spend more CPU
    # seconds than the step takes, and a share above 1 is the speedup showing up, not a bug. Keep
    # comparing phases arm-to-arm in seconds; only the serial arm's shares are fractions of a step.
    return CBEBugSweepInfo(expanded, n_new, c, root_rank, epre, efnl,
                           discarded, accL.nmv + accR.nmv + nroot[], peak,
                           accL.ngrow + accR.ngrow, leg_dim(Z[c], 1),
                           accL.tcbe + accR.tcbe, accL.tkry + accR.tkry, troot[], ttrunc[],
                           accL.tsplit + accR.tsplit, accL.tframe + accR.tframe,
                           accL.tenv + accR.tenv, (time_ns() - _t0step) / 1e9)
end

cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_step!(psi, mpo_from_terms(h), tau; kwargs...)

"""
    cbe_bug_midpoint_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

One MIDPOINT BUG step (Ceruti-Kusch-Lubich, arXiv:2402.08607), in place.

TWO STAGES, and the second is an ordinary [`cbe_bug_step!`](@ref) with its bases taken from the
first:

  1. a HALF STEP, `tau/2`, whose only product is a midpoint state `Y_{1/2}`;
  2. a FULL step over `tau` whose frames are built from `Y_{1/2}` but which integrates the
     Galerkin ODE from the ORIGINAL `Y_0`, then truncates once.

⛔ IT IS EXPLICIT, DESPITE THE NAME. The midpoint *rule* is an implicit ODE method; this
integrator approximates the midpoint with a half-step of the augmented BUG integrator instead of
solving for it, so no linear system is ever formed. It is a two-stage EXPLICIT scheme, and it
does NOT remove the Krylov exponential -- stage 2's Galerkin ODE over the full step IS
`exp(tau*H_0)` on the core, i.e. the same `expv` root solve the ordinary step makes. What it buys
is Theorem 4.2: a robust SECOND-ORDER global error bound whose constants are INDEPENDENT OF THE
SINGULAR VALUES, with improved dependence on the normal component of the vector field.

⛔ STAGE 1 MUST NOT TRUNCATE. Its job is to supply DIRECTIONS, and the paper's midpoint state is
the rank-2r augmented one; cutting it back to rank r here would discard exactly what the
augmented basis of stage 2 is supposed to contain. `truncate = false` is therefore forced, not a
default, and rank control happens once at the end of stage 2.

⚠ COST IS ~2x THE ORDINARY STEP -- two sweeps and two root solves. Whether that pays has to be
MEASURED against simply halving `dt` on the ordinary sweep, which buys second-order accuracy at
the same 2x. This sweep is already measured at `O(dt^2)`, so what is on offer here is the error
CONSTANT and the proven bound, not an extra order; `benchmarks/midpoint_bug.jl` is where that
comparison lives.
"""
function cbe_bug_midpoint_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64; kwargs...)
    half = copy(psi)
    # stage 1: bases only. `truncate = false` keeps the augmented rank -- see the docstring.
    cbe_bug_step!(half, mpo, tau / 2; kwargs..., truncate = false)
    # stage 2: full step, frames from the midpoint, START VECTOR still `psi`.
    return cbe_bug_step!(psi, mpo, tau; kwargs..., frame_state = half)
end

cbe_bug_midpoint_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_midpoint_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_midpoint_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_midpoint_step!(psi, mpo_from_terms(h), tau; kwargs...)
