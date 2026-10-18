
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
        # ⛔⛔ ONE MATERIALISATION PER PASS, NOT ONE PER VECTOR -- THIS IS THE ALLOCATION DEFECT
        # THAT BLOCKED THE SPINFUL CAMPAIGN. The previous form updated `w` inside the inner loop:
        #
        #     for _pass in 1:2, (i, u) in pairs(vs)
        #         c = tensor_inner(u, w); Hm[i,k] += c; w = to_concrete(w - c * u)   # allocates
        #
        # so one solve performed `sum_k 2k ~ m^2` full-tensor materialisations. MEASURED at
        # L=20, chi=32, d=4, m=30: ~930 per solve x ~38 solves x a 262 KB two-site tensor, and the
        # step allocated **28.6 GiB rising to 237 GiB by step 5** with `maxbond` pinned at 32.
        # The allocation profile named it exactly: `_compress_sum_sector` 24.6% of bytes (the
        # `w - c*u` being compressed) and `copy` 37.9% (the `to_concrete`), against ~6.6% for
        # every SVD site combined -- which is why the TIME profile, where SVD is 68.6%, pointed
        # somewhere else entirely.
        #
        # Accumulating the correction and subtracting ONCE per pass makes it `2m`
        # materialisations per solve instead of `m^2` -- 15x fewer at m=30 -- because Telum's `+`
        # and `*` are LAZY, so `corr` is a symbolic tree that costs nothing until it is forced.
        #
        # ⚠ THIS IS THE CLASSICAL (CGS2) PROJECTION RATHER THAN THE MODIFIED (MGS) ONE: within a
        # pass every inner product is taken against the SAME `w`, not against a partially updated
        # one. That is a deliberate and standard trade -- CGS **repeated twice** is as stable as
        # MGS ("twice is enough"), and the two passes were already here. The coefficients are
        # still accumulated across both passes, so `Hm` means what it did before.
        #
        # ⚠ AND `tensor_inner` IS NEVER CALLED ON THE SYMBOLIC TREE. `w` stays concrete for the
        # whole pass and only `corr` is lazy, so this does NOT reintroduce the recorded O(k^3)
        # trap where a symbolic `w` was re-forced one term longer on every inner product.
        for _pass in 1:2
            corr = nothing
            for (i, u) in pairs(vs)
                c = tensor_inner(u, w)
                # accumulate across BOTH passes: the second pass's correction is part of the
                # projection, not a separate quantity
                Hm[i, k] += c
                corr = corr === nothing ? c * u : corr + c * u
            end
            corr === nothing || (w = to_concrete(w - corr))
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

# ── STAGE 3: RESIDUAL-DRIVEN K/L KRYLOV ENRICHMENT (plan §4, Tests C and D) ──────────────────
#
# The adaptive path's replacement for `_krylov_frame`'s fixed `krylov_basis`/`krylov_tol`.
#
# ⛔ TWO SPACES, AND THE TESTS LIVE IN THE SMALLER ONE. The Krylov recursion runs on ONE-SITE
# TENSORS (`(link, site, g)` on the left), exactly as `_krylov_frame` does. What the half-sweep
# keeps is a FRAME -- an isometry on the local legs `(link, site)` -- and the plan's `Q` with
# its projector `I - QQ'` is that frame: a Krylov vector adds something only through the part
# of its local range the frame does not already span. `Q0` is the CBE output `U_ex = [U_old |
# U_CBE]`, so the old directions are kept VERBATIM and the CBE-selected ones are reused; the
# seed `K_ex` already lies in `U_old` (it is the state's own site tensor).

const _TAG_K, _TAG_KR = "cbeK,L", "cbeK,R"

"Safety bound on the adaptive Krylov depth. Reaching it with weight left is REPORTED, not hidden."
const _KRY_MAXDEPTH = 64

# The norm-difference `sqrt(||psi||^2 - ||S0||^2)` cannot resolve a projection error below
# ~sqrt(eps): cancellation. A frame that contains `psi` exactly reads ~1e-8 here, so a
# `tol_basis` tighter than this floor is compared against the floor, not against roundoff.
const _PROJ_FLOOR = 1e-7

# The root solve's Krylov basis per substep on the adaptive path. A NUMERICAL cap, not an
# accuracy one: `substeps = 0` splits `tau` whenever this basis cannot reach `tol_expv`.
const _ROOT_MAXITER = 30

"""
    _kry_weight(Hm, k, tau) -> Float64

The weight the newest Krylov vector `v_{k+1}` carries in `exp(tau*H) v_1`, from the Arnoldi
Hessenberg after `k` steps: `|tau * beta_k * [phi_1(tau*H_k)]_{k,1}|`, read off ONE `(k+1)x(k+1)`
exponential of `[tau*H_k 0; tau*beta_k*e_k' 0]`. This is Saad's a posteriori error of the
`k`-dimensional Krylov approximation, so it is also the CLOSURE test: once it is below the
allowance, no later direction can matter more (for `k = 1` it is `|tau| beta_1 (1 + O(tau))`,
the plan's Test C normalisation `|dt| ||H1 K_ex|| / ||K_ex||`).
"""
function _kry_weight(Hm::Matrix{ComplexF64}, k::Int, tau::ComplexF64)
    A = zeros(ComplexF64, k + 1, k + 1)
    A[1:k, 1:k] .= tau .* @view Hm[1:k, 1:k]
    A[k + 1, k] = tau * Hm[k + 1, k]
    real(tau) == 0 && return abs(exp(A)[k + 1, 1])
    # ⛔ IMAGINARY TIME: RELATIVE TO THE PROPAGATED SEED, AND SHIFTED TO STAY REPRESENTABLE. The
    # propagator is not unitary, so `|[exp(A)]_{k+1,1}|` carries the seed's own growth
    # `||exp(tau*H_k) e_1||` -- `exp(dbeta*|E|)` for an unshifted generator, i.e. `e^44` at L=100,
    # dbeta=1 -- and every direction would look that much more important than it is to the
    # NORMALISED state the caller keeps (and overflow past dbeta*|E| ~ 700). The ratio is the
    # weight the direction carries in the normalised propagated seed; the scalar shift cancels in
    # it exactly. In real time the propagator is unitary and the old expression is kept bit for bit.
    s = maximum(real, eigvals(A[1:k, 1:k]))
    E = exp(A - s * I)
    return abs(E[k + 1, 1]) / max(norm(@view E[1:k, 1]), floatmin(Float64))
end

"""
    _extend_frame(Q, R, v, side, weight, allowance; room) -> (Q, n_added, fits)

Append to the frame `Q` the local directions of the residual `R = (I - QQ') v` that the
allowance requires: the smallest number whose weighted discarded tail `weight * sqrt(sum s^2)`
fits (Test A's rule, `_tail_rank`). Orthogonalised within each charge sector by the block SVD,
cut only at the numerical floor `_rank_cut`, re-orthogonalised against `Q` before the direct
sum -- the same guard `cbe_expand` applies to its own appended block.

`room` is what the work ceiling still allows on the frame leg (plan §5): at most that many are
appended, and `fits = false` reports that the allowance needed more.
"""
function _extend_frame(Q, R, v, side::Symbol, weight::Float64, allowance::Float64;
                       room::Int = typemax(Int))
    legs = side === :left ? (1, 2) : (2, 3)
    cut = _rank_cut(0.0, v, norm(R))
    res = svd(R, legs, _TAG_K, _TAG_KR; cutoff = cut, get_lists = true)
    nk = length(res.kept_list)
    k, tail = _tail_rank(res.kept_list, weight, allowance, min(nk, room);
                         base = _sq_weight(res.trunc_list))
    fits = tail <= allowance
    k == 0 && return (Q, 0, fits)
    k < nk && (res = svd(R, legs, _TAG_K, _TAG_KR; Nkeep = k, cutoff = cut))
    U = side === :left ? _reortho_left(Q, to_concrete(res.U)) :
                         _reortho_right(Q, to_concrete(permutedims(res.U, (3, 1, 2))))
    U === nothing && return (Q, 0, fits)
    fl = side === :left ? 3 : 1
    Qn = to_concrete(oplus([Q, to_concrete(setitag(U, fl, Q.inds[fl].itags))], (fl,)))
    return (Qn, leg_dim(U, fl), fits)
end

"""
    _krylov_enrich(v0, applyH, Q0, side, tau, alloc; workcap = 0, maxdepth = _KRY_MAXDEPTH)
        -> (Q, stats)

Grow the frame `Q0` by the Krylov directions of `applyH` seeded at `v0` that the step needs.
`side` is `:left` (frame legs `(link, site, g)`, local legs 1-2) or `:right` (`(g, site, link)`,
local legs 2-3). Each Arnoldi direction `v_k` is scored

    eta_k = weight_k * ||(I - QQ') v_k||          (weight_k from `_kry_weight`)

`eta_1` is TEST C (does the CBE frame already hold the first one-site action?), `eta_{k>=2}`
TEST D (is the grown frame closed under `H1` on the active frontier?). A score above
`alloc.krylov` appends that residual's directions. A small `eta_1` is NOT a stop: the plan's
counterexample (`H e1 = e2`, `H e2 = e1 + e3`, `Q0 = [e1, e2]`) has `eta_1 = 0` and a needed
`e3`. The loop stops when the propagator weight itself is below the allowance -- every
remaining direction is then negligible however new -- or when the recursion closes exactly.
Directions with no amplitude on the seed's Krylov space are never visited, so dormant ones are
not forced closed.

Returns the grown frame and `stats = (eta_c, eta_d, added, n_h1, d_after_c, capped, met)`.
`capped` means the work ceiling or the depth bound stopped the search with weight left;
`met = false` then, and the caller records it -- never a silent pass.

`amp` is the seed's norm relative to the STATE's (`||v0|| / ||psi||`). The weights above are
relative to the unit seed, which is the state's own scale on the ordinary half-sweep (`amp = 1`);
a Krylov frontier (`krylov_closure.jl`) is a small vector, and its directions matter only in
proportion to its size.
"""
function _krylov_enrich(v0, applyH, Q0, side::Symbol, tau::ComplexF64,
                        alloc::BasisAllocation; workcap::Int = 0,
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0)
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    perp = side === :left ? perp_component : perp_component_right
    fl = side === :left ? 3 : 1
    Q = Q0
    eta_c, eta_d, added, added_d, nh = 0.0, 0.0, 0, 0, 0
    capped, closed = false, false
    b0 = norm(v0)
    if b0 > 0
        # `tau = 0` asks the unit-time closure question, as `basis_allocation` does.
        teff = iszero(tau) ? one(tau) : tau
        vs = Any[to_concrete((1.0 / b0) * v0)]
        Hm = zeros(ComplexF64, maxdepth + 1, maxdepth + 1)
        for k in 1:maxdepth
            w = applyH(vs[end])
            nh += 1
            # CGS2 against every previous vector, as in `_krylov_frame` (see the notes there).
            for _pass in 1:2
                corr = nothing
                for (t, u) in pairs(vs)
                    ov = tensor_inner(u, w)
                    Hm[t, k] += ov
                    corr = corr === nothing ? ov * u : corr + ov * u
                end
                corr === nothing || (w = to_concrete(w - corr))
            end
            b = norm(w)
            b <= _KRY_BREAKDOWN && (closed = true; break)    # invariant: nothing further exists
            Hm[k + 1, k] = b
            v = to_concrete((1.0 / b) * w)
            push!(vs, v)
            weight = amp * _kry_weight(Hm, k, teff)
            R = to_concrete(perp(Q, v))
            eta = weight * norm(R)
            k == 1 ? (eta_c = eta) : (eta_d = max(eta_d, eta))
            if eta > alloc.krylov
                # The ceiling bounds the frame WIDTH, not the number of extensions: a block
                # may not carry the frame past it either.
                room = workcap > 0 ? workcap - leg_dim(Q, fl) : typemax(Int)
                room <= 0 && (capped = true; break)
                Q, nadd, fits = _extend_frame(Q, R, v, side, weight, alloc.krylov; room = room)
                added += nadd
                k >= 2 && (added_d += nadd)
                fits || (capped = true; break)
            end
            weight <= alloc.krylov && (closed = true; break)
        end
        closed || (capped = true)                             # depth bound with weight left
    else
        closed = true
    end
    return (Q, (eta_c = eta_c, eta_d = eta_d, added = added, n_h1 = nh,
                d_after_c = eta_c <= alloc.krylov && added_d > 0,
                capped = capped, met = closed && !capped))
end

"Fold one `_krylov_enrich` verdict into the step ledger (plan §8 Krylov diagnostics)."
function _record_krylov!(ledger::Union{Nothing, AdaptiveLedger}, st, alloc::BasisAllocation)
    ledger === nothing && return nothing
    ledger.max_krylov_c = max(ledger.max_krylov_c, st.eta_c)
    ledger.max_krylov_d = max(ledger.max_krylov_d, st.eta_d)
    ledger.max_krylov   = max(ledger.max_krylov, st.eta_c, st.eta_d)
    ledger.n_krylov += st.added
    ledger.n_h1     += st.n_h1
    st.eta_c <= alloc.krylov && (ledger.n_c_pass += 1)
    st.d_after_c && (ledger.n_d_after_c += 1)
    st.capped && (ledger.workcap_reached = true)
    st.met || _unmet!(ledger, :krylov)
    return nothing
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
  - `discarded` -- largest relative weight any single SVD of the closing truncation (or the
    root split) threw away. ⚠ NOT the step's compression error: on the adaptive path that is
    `ledger.discarded`, the cumulative relative norm over every bond (plan §5).
  - `ledger` -- the adaptive path's [`AdaptiveLedger`](@ref) (Test A/B/C/D indicators, probe
    counts, cumulative compression, projection error, ranks, `tolerance_met`,
    `workcap_reached`); `nothing` on the legacy fixed-knob path. [`budget_status`](@ref) reads
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
    # ⛔ THE ADAPTIVE PATH'S VERDICT, `nothing` on the legacy path. Without it a step that failed
    # Test A/B, or stopped at the work ceiling, returns looking exactly like one that passed --
    # the silent convergence claim the three-tolerance API exists to rule out.
    ledger::Union{Nothing, AdaptiveLedger}
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

  - `order` -- the temporal order `p` (default `1`, the ordinary step). For `p > 1` the frames
    the half-sweeps built are grown until the root generator `B = P(tau H)P` reproduces the
    global Krylov moments `B^j psi = (tau H)^j psi`, `j <= p`: each projected frontier
    `F_j = (tau^j/j!) B^j S0`, `j = 1..p-1`, is embedded as an MPS and swept ONCE with the same
    CBE / Tests C/D kernels, which enrich every bond whose outside-space action exceeds its share
    of `tol_basis` (`krylov_closure.jl`). `closure_checks = true` adds the validation measurements
    (check first, repeat until a pass adds nothing, final `j = 0`). One root `expv`, one
    compression, as always. `ledger.leak`,
    `ledger.defect`, `ledger.n_passes` and `ledger.closure_log` report it; failing to close is
    `:budget_unmet` (`:closure`). An order is a CLAIM TO BE MEASURED, not a guarantee: finite
    tolerances, the rank cap and the final compression set an error floor.

  ⛔ ON THE ADAPTIVE PATH (no retired knob named) THE HALF-SWEEP FRAME IS BUILT BY
  [`_krylov_enrich`](@ref): it starts from the CBE frame `U_ex` and appends Krylov directions only
  when Tests C/D (plan §4) find their weighted residual outside the frame above `tol_basis`'s
  allowance; the depth follows from that, not from a count. `krylov_basis`, `krylov_tol`,
  `krylov_grow` and `grow_iters` below are the LEGACY path's controls, and naming any of them
  together with a tolerance is refused.

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
    sweep passes. LEGACY ONLY: the adaptive frame has no split (plan §5), only the numerical
    rank floor inside `_krylov_enrich`.
  - `root_cutoff`, `root_maxdim` -- truncation on the ROOT CORE SPLIT. **Defaults `0.0`/`0`, i.e.
    NONE**: that SVD splits the Galerkin core into the two centre tensors and nothing more.
    There are exactly TWO truncations in a step -- the half-sweep split and the closing pass --
    and this used to be an unswitchable third. The adaptive path pins them to `0.0`/`0`.
  - `maxdim`, `trunc_thresh` -- the closing [`truncate_recursive!`](@ref). `maxdim` defaults to
    128 on the adaptive path (plan §1) and 200 on the legacy path. On the adaptive path
    `tol_trunc` replaces `trunc_thresh` and is a budget on the CUMULATIVE relative discarded
    norm (`weight_tol`); exceeding it, e.g. because `maxdim` binds, makes the step
    `:budget_unmet` rather than silently accepted. `truncate = false`
    disables it; with the root defaults above that leaves `split_cutoff` as the only rank control
    anywhere in the step, which is a study configuration and not a production one.
  - `maxiter`, `tol`, `reorth` -- Krylov budget for the root solve. `maxiter`, `tol`,
    `root_conv_tol` and `root_substeps` are LEGACY knobs: on the adaptive path `tol_expv` is the
    convergence test and the substep split is adaptive (`ledger.expv_err`, `expv_iters`).
"""
function cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                       # ── THE NORMAL API: three tolerances and one resource limit ────────
                       #
                       # `nothing` means "not asked for", which is what lets the step tell an
                       # adaptive caller from a legacy one without a mode flag. See
                       # `resolve_budget` in `adaptive_budget.jl` for why the two sets of
                       # keywords are mutually exclusive rather than merged.
                       tol_basis::Union{Nothing, Float64} = nothing,
                       tol_trunc::Union{Nothing, Float64} = nothing,
                       tol_expv::Union{Nothing, Float64} = nothing,
                       # The temporal order `p` -- a METHOD choice, not a fourth tolerance.
                       # `p > 1` closes the global Krylov frontiers `j = 1..p-1` of the root
                       # generator before the one root solve (`krylov_closure.jl`); `1` is the
                       # ordinary step, unchanged. Adaptive path only.
                       order::Int = 1,
                       # ⚠ INSTRUMENTATION, NOT A METHOD CHOICE. `m > 0` on an `order = 1`
                       # step MEASURES (never enriches) the frontiers `j = 0..m-1` on the frames
                       # the ordinary step solves in -- `ledger.leak_initial`/`rho_initial`,
                       # what CBE + local Krylov already capture of the global Krylov sequence.
                       # The evolved state is unchanged. `order > 1` always records them.
                       closure_probe::Int = 0,
                       # ⚠ VALIDATION, NOT A METHOD CHOICE (USER 2026-10-17: off by default).
                       # `false`: an `order = p` step is the two half-sweeps plus ONE enriching
                       # sweep per frontier j = 1..p-1. `true`: the closure as it was validated --
                       # measure every frontier first, loop until a pass adds nothing, measure
                       # j = 0 on the final frames (`_krylov_closure!(...; checks)`).
                       closure_checks::Bool = false,
                       # ── IMAGINARY TIME (plan: cbe_bug_imaginary_open_benchmark, Task 1) ──
                       # The root integrator (`ROOT_METHODS`): `:exp` is the Saad-converged
                       # exponential (the reference); `:trapezoid`/`:radau2`/`:radau3` apply that
                       # implicit method's stability function in one stage. A METHOD choice.
                       root_method::Symbol = :exp,
                       # The step integrates `H - eshift` (`shift_h`): the root and every Krylov
                       # generator, so the basis weights measure the change of the NORMALISED
                       # state rather than its `exp(-dbeta*E)` growth. `0.0` is bit-identical.
                       eshift::Real = 0.0,
                       # ⚠ INSTRUMENTATION. A `Ref` receives the root's account: energies of the
                       # state before and after the root solve (`<S|H|S>/<S|S>`, exact in the
                       # Galerkin space), the solve's report, and with `outside_residual = true`
                       # the outside-Galerkin action `||(I - P) H psi_1|| / ||psi_1||` of the
                       # evolved state (`estimate_global_complement_action`, one measuring sweep).
                       root_report::Union{Nothing, Base.RefValue} = nothing,
                       outside_residual::Bool = false,
                       # ── THE TRIAL SPACE IS CONSTRAINED TO CONTAIN THESE STATES (plan Task 2) ──
                       # Each half-sweep carries every `keep_states` MPS through the new frames, as
                       # it carries `psi`, and unions its local part into the frame, so the root
                       # space contains each of them exactly. For an MPDO, keeping `|I>>` makes the
                       # Galerkin step TRACE-PRESERVING before the compression: `<<I|P = <<I|` and
                       # `<<I|L = 0` give `d/dt <<I|rho>> = <<I|P L P rho>> = 0`. The states must
                       # share `psi`'s site legs; `[]` is the ordinary step, bit for bit.
                       keep_states::AbstractVector = Any[],
                       # ── THE RETIRED TUNING PARAMETERS ─────────────────────────────────
                       #
                       # ⛔ EACH OF THESE IS A GUESS AT A RANK MADE BEFORE THE SPECTRUM THAT
                       # ANSWERS IT EXISTS, and each had to be re-guessed per Hamiltonian.
                       # They survive as a FIXED-KNOB LEGACY PATH so the benchmark suite can
                       # still drive the historical behaviour for A/B comparison; they are not
                       # the normal API and naming one opts the whole call out of the adaptive
                       # solver. `nothing` resolves to the historical default, so a call that
                       # names none of them and none of the tolerances is unchanged.
                       dex::Union{Nothing, Int} = nothing,
                       growth::Union{Nothing, Float64} = nothing,
                       dover::Union{Nothing, Int} = nothing,
                       comp_ratio::Union{Float64, Nothing} = nothing,
                       sulz_cap::Bool = false,
                       preselect_only::Bool = false,
                       exact::Bool = false,
                       stol_pre::Union{Nothing, Float64} = nothing,
                       stol_fnl::Union{Nothing, Float64} = nothing,
                       root::Union{Nothing, Int} = nothing,
                       krylov_basis::Union{Nothing, Int} = nothing,
                       krylov_tol::Union{Nothing, Float64} = nothing,
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
                       #
                       # ⛔ `grow_iters = 1` GROWS NOTHING -- THE MINIMUM USEFUL VALUE HERE IS 2.
                       # `_grow_frame` breaks on `k == krylov_basis` BEFORE it reaches the
                       # re-expansion, so one pass builds a second Krylov vector and never widens
                       # a frame. MEASURED (L=16 chain, 10 imaginary-time steps): `n_grow` is 0 at
                       # `grow_iters = 1` against 114 at 2 and 203 at 3. It is not an error -- it
                       # is a slower spelling of `krylov_grow = false, krylov_basis = 1` -- but a
                       # scan that starts its growth axis at 1 reports its first point as "growth
                       # on" when nothing grew. ⚠ AND `GroundState.solve!` HAS A DIFFERENT
                       # `grow_iters` WITH DIFFERENT SEMANTICS: it drives `_expanding_krylov`,
                       # where 1 IS viable and is the documented default. Same name, two paths.
                       #
                       # ⛔ `grow_iters = 0` IS NOT A CONTROLLED BASELINE FOR THE OTHERS. It only
                       # clears `krylov_grow`; `krylov_basis` keeps whatever it had, which is the
                       # DEFAULT 30. So `grow_iters = 0` against `grow_iters = 2` compares a
                       # 30-vector fixed-width builder with a 2-vector growth path and changes two
                       # things at once. To isolate growth, hold the width: pass
                       # `krylov_grow = false, krylov_basis = n` as the control against
                       # `grow_iters = n`.
                       grow_iters::Union{Nothing, Int} = nothing,
                       split_cutoff::Union{Nothing, Float64} = nothing,
                       split_maxdim::Union{Nothing, Int} = nothing,
                       truncate::Bool = true,
                       root_cutoff::Union{Nothing, Float64} = nothing,
                       root_maxdim::Union{Nothing, Int} = nothing,
                       maxdim::Union{Nothing, Int} = nothing,
                       trunc_thresh::Union{Nothing, Float64} = nothing,
                       maxiter::Union{Nothing, Int} = nothing,
                       tol::Union{Nothing, Float64} = nothing,
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
                       root_conv_tol::Union{Nothing, Float64} = nothing,
                       root_substeps::Union{Nothing, Int} = nothing,
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
                       # ⛔ `nothing` MEANS "NOT ASKED FOR", AND THE SWEEP PASSES `exkw` TO EVERY
                       # `cbe_expand` UNCONDITIONALLY -- so a `Bool` default here would hand
                       # `cbe_expand` an EXPLICIT `share_ht` on every call and make its
                       # fold_omega/share_ht contradiction check fire on runs that never asked
                       # for sharing at all. `nothing` is resolved to `true` at the bottom of the
                       # chain, where the two flags are actually reconciled.
                       share_ht::Union{Nothing, Bool} = nothing,
                       # ⚠ FOLD THE PROBE INTO THE CONTRACTION INSTEAD OF BUILDING `H*Theta`.
                       # See `mpo.jl` -- the saving is `O(chi^3) -> O(chi^2)` per sketch but it
                       # costs THREE chains where the shared build pays one, so it only wins for
                       # `3*Dpre < chi`. ⛔ SET `dex` SMALL WITH IT OR IT LOSES: at the default
                       # `dex = 0` the width comes from `growth = 2.0` and is ~chi/2, far past the
                       # crossover. MPO path only.
                       fold_omega::Bool = false,
                       # Adaptive path: Test B, not the numerical rank, sets the sketch width
                       # (`_certified_range`). `false` is the old full-rank widening, an A/B.
                       thin_search::Bool = true,
                       rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("cbe_bug_step! needs at least two sites"))
    c = root === nothing ? _bug_root_link(L) : root
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))

    # ── WHICH API DID THE CALLER USE? ───────────────────────────────────────────────────────
    #
    # `resolve_budget` refuses the mixture and infers the mode from what was named; everything
    # below then works with CONCRETE values, so no downstream code has to know which path it
    # is on. `legacy_names` is non-empty exactly on the legacy path and is carried into the
    # ledger's status so a benchmark can report which solver produced its numbers.
    budget, adaptive, legacy_names = resolve_budget(
        tol_basis, tol_trunc, tol_expv,
        (dex = dex, dover = dover, comp_ratio = comp_ratio, growth = growth,
         krylov_basis = krylov_basis, split_cutoff = split_cutoff,
         split_maxdim = split_maxdim, root_cutoff = root_cutoff,
         root_maxdim = root_maxdim, trunc_thresh = trunc_thresh,
         stol_pre = stol_pre, stol_fnl = stol_fnl,
         # `krylov_grow` is a Bool, so only `true` is a request; `false` is its default.
         krylov_tol = krylov_tol, grow_iters = grow_iters,
         krylov_grow = krylov_grow ? true : nothing,
         maxiter = maxiter, tol = tol, root_conv_tol = root_conv_tol,
         root_substeps = root_substeps))
    _legacy_hint(legacy_names)
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    order == 1 || adaptive || throw(ArgumentError(
        "order = $order needs the adaptive solver; $(join(legacy_names, ", ")) selected the " *
        "first-order legacy path, which has no tolerance to close the Krylov frontiers against"))
    order == 1 || frame_state === nothing || throw(ArgumentError(
        "order > 1 is a single-step method and does not combine with the midpoint predictor"))
    closure_probe >= 0 && (closure_probe == 0 || adaptive) || throw(ArgumentError(
        "closure_probe = $closure_probe needs the adaptive solver and a non-negative count"))

    dex          = legacy_or_default(dex, :dex)
    growth       = adaptive ? ADAPTIVE_GROWTH : legacy_or_default(growth, :growth)
    comp_ratio   = legacy_or_default(comp_ratio, :comp_ratio)
    krylov_basis = legacy_or_default(krylov_basis, :krylov_basis)
    krylov_tol   = legacy_or_default(krylov_tol, :krylov_tol)
    split_cutoff = legacy_or_default(split_cutoff, :split_cutoff)
    split_maxdim = legacy_or_default(split_maxdim, :split_maxdim)
    # Plan §5: the root structural SVD is LOSSLESS on the adaptive path (keeps every nonzero
    # singular value, no `Nkeep`); written out rather than inherited from `LEGACY_DEFAULTS` so
    # a change to the legacy defaults cannot put a physical cut back on the adaptive root.
    # Plan §1/§4 (Stage 5): `tol_expv` IS the root solve's convergence test (Saad's estimate) and
    # drives its substep split (`substeps = 0`), so the iteration count follows from it. The
    # per-substep basis cap and the breakdown threshold are numerical, not accuracy, settings.
    maxiter       = adaptive ? _ROOT_MAXITER : legacy_or_default(maxiter, :maxiter)
    tol           = adaptive ? 1e-15 : legacy_or_default(tol, :tol)
    root_conv_tol = adaptive ? budget.expv : legacy_or_default(root_conv_tol, :root_conv_tol)
    root_substeps = adaptive ? 0 : legacy_or_default(root_substeps, :root_substeps)
    root_cutoff  = adaptive ? 0.0 : legacy_or_default(root_cutoff, :root_cutoff)
    root_maxdim  = adaptive ? 0 : legacy_or_default(root_maxdim, :root_maxdim)
    # The adaptive path passes `0.0`: the preselection then cuts only at the numerical rank
    # floor (`_rank_cut`), and which directions MATTER is decided by Tests A/B against
    # `tol_basis` -- plan §1 removes `stol_pre` as a physical threshold, not just as a keyword.
    stol_pre     = adaptive ? 0.0 : legacy_or_default(stol_pre, :stol_pre)
    # Plan §1 signature: `maxdim = 128`. The legacy path keeps its historical 200 so recorded
    # fixed-knob benchmarks are unchanged.
    maxdim       = something(maxdim, adaptive ? 128 : 200)
    stol_fnl     = legacy_or_default(stol_fnl, :stol_fnl)
    # ⚠ `trunc_thresh` IS THE ONE THE ADAPTIVE PATH OVERRIDES rather than defaults. It is the
    # final compression tolerance, which on the normal API is `tol_trunc` BY DEFINITION -- the
    # plan's "one shared compression budget" instead of a separate per-step cutoff.
    trunc_thresh = adaptive ? budget.trunc : legacy_or_default(trunc_thresh, :trunc_thresh)

    # The per-bond, per-test allowances `tol_basis` buys, plus the `|tau|/||psi||` scale every
    # residual is measured in. Unused on the legacy path, which answers the same questions
    # with fixed counts.
    # ⚠ THE NORM COMES FROM THE CENTRE TENSOR, which carries the state's whole norm only while
    # the state is canonical -- true on entry for every caller here, since the sweeps gauge
    # before they read. Guarded anyway: an out-of-range `center` gives 1.0 (an unnormalised
    # relative scale) rather than a bounds error from a diagnostic.
    _nrm = (1 <= psi.center <= L) ? norm(psi[psi.center]) : 1.0
    alloc  = basis_allocation(budget, L - 1; tau = tau, nrm = _nrm, nfront = order)
    ledger = AdaptiveLedger()
    ledger.order = order
    bd0 = bond_dims(psi)                 # per-bond `r_old` for the bond log (plan §8)
    ledger.r_old = maximum(bd0; init = 0)
    bwts = adaptive ? zeros(L - 1) : nothing   # per-bond discarded norm of the compression

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
    # The ONE work ceiling, shared by the CBE sketch and the Krylov enrichment (plan §4/§5).
    wcap_step = work_ceiling(maxdim, maximum(bond_dims(psi); init = 0))
    exkw = (dex = dex, growth = growth, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact,
            stol_pre = stol_pre, stol_fnl = stol_fnl, share_ht = share_ht,
            fold_omega = fold_omega, thin_search = thin_search,
            # `nothing` on the legacy path keeps `cbe_expand`'s fixed-count selection exactly
            # as it was; an allocation switches on the residual tests.
            alloc = adaptive ? alloc : nothing,
            workcap = wcap_step)

    # ⛔ THE LEDGER AND THE RNG ARE PER HALF-SWEEP, because under `parallel = true` the two run
    # concurrently: a shared `AdaptiveLedger` loses `+=` updates, and a shared `MersenneTwister`
    # is not thread-safe at all. The right sweep gets a stream SEEDED FROM `rng` only when
    # parallel, so the serial path draws exactly the sequence it always did. The two ledgers are
    # folded into the step's after the join.
    ledgerL, ledgerR = AdaptiveLedger(), AdaptiveLedger()
    rngR = parallel ? MersenneTwister(rand(rng, UInt64)) : rng
    halfkw(acc) = acc === accL ? (ledger = adaptive ? ledgerL : nothing, rng = rng) :
                                 (ledger = adaptive ? ledgerR : nothing, rng = rngR)

    troot = Ref(0.0); ttrunc = Ref(0.0)
    _t0step = time_ns()
    """Time one `cbe_expand` into the calling half-sweep's own accumulator. An explicit
    `ledger` (the bond's own, see `_log_bond!`) wins over the half-sweep's."""
    function timed_expand(acc::_HalfAcc, args...; kw...)
        t0 = time_ns()
        ex = cbe_expand(args...; halfkw(acc)..., kw...)
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
            H1 = shift_h(side === :left ?
                         one_site_h(mpo, i, lch, push_right_channels(rch, mpo, V, i + 1)) :
                         one_site_h(mpo, i + 1, push_left_channels(lch, mpo, V, i), rch), eshift)
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
                core = to_concrete!(contract(U0', (1, 2), vs[end], (1, 2)))     # (bU, g)
                BondFrame(U0, core, V,
                          vs[end].inds[1], vs[end].inds[2], vs[end].inds[3],
                          V.inds[2], V.inds[3],
                          leg_dim(U0, 3), copy(vs[end].spaces[3]))
            else
                spl  = svd(vs[end], (1,), "bV,L", "bV,R"; cutoff = 0.0)
                V0   = to_concrete(spl.Vd)
                core = to_concrete!(contract(vs[end], (2, 3), V0', (2, 3)))     # (g, bV)
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
                    to_concrete!(contract(V, (2, 3), Vn', (2, 3))) :      # (old_g, new_g)
                    to_concrete!(contract(Vn', (1, 2), V, (1, 2)))       # (new_g, old_g)
                for t in eachindex(vs)
                    vs[t] = side === :left ?
                            to_concrete!(contract(vs[t], (3,), P, (1,))) :
                            to_concrete!(contract(P, (2,), vs[t], (1,)))
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
        kcarL = Any[nothing for _ in keep_states]   # each kept state's carry, `(b_i, link)`
        held = 0                          # frames this sweep has accumulated, running total
    for i in 1:c
        K = C === nothing ? psiL[i] : to_concrete!(contract(C, (2,), psiL[i], (1,)))
        _fr = (t0 = time_ns(); f = _frame_from(K, psiL[i + 1]);
               accL.tframe += (time_ns() - t0) / 1e9; f)
        bondlg = adaptive ? AdaptiveLedger() : nothing         # this bond's own ledger (§8)
        ex = record!(accL, n_newL, timed_expand(accL, _fr, mpo, i, lch,
                                right_channels(rstack, i + 2); exkw..., ledger = bondlg), i, :left)
        peak!(accL, held + _state_stored(psiL), ex.U_ex, ex.V_ex)
        Vwide[i] = ex.V_ex
        Wcbe[i]  = ex.U_ex
        held += tensor_elements(ex.U_ex) + tensor_elements(ex.V_ex)
        if adaptive
            # ── STAGE 3 (plan §4): THE FRAME GROWS FROM `U_ex` BY TESTS C/D ───────────────
            # Same seed and same `H1` as the legacy branch below; what changes is that the CBE
            # frame is KEPT (old directions verbatim, CBE directions reused) and a Krylov
            # direction is appended only when its weighted residual outside the frame matters.
            M    = to_concrete!(contract(psiL[i + 1], (2, 3), ex.V_ex', (2, 3)))
            Kex  = to_concrete!(contract(K, (3,), M, (1,)))
            rch1 = push_right_channels(right_channels(rstack, i + 2), mpo, ex.V_ex, i + 1)
            H1   = shift_h(one_site_h(mpo, i, lch, rch1), eshift)
            tkL0 = time_ns()
            Wkry, kstL = _krylov_enrich(Kex, x -> (accL.nmv += 1; apply_one_site(H1, x)),
                                        ex.U_ex, :left, tau, alloc; workcap = wcap_step)
            accL.tkry += (time_ns() - tkL0) / 1e9
            _record_krylov!(bondlg, kstL, alloc)
            peak!(accL, held + _state_stored(psiL), Wkry)
            W[i] = to_concrete(setitag(Wkry, 3, _tagW(i)))
        elseif krylov_basis > 0
            M   = to_concrete!(contract(psiL[i + 1], (2, 3), ex.V_ex', (2, 3)))
            Kex = to_concrete!(contract(K, (3,), M, (1,)))
            blocks = if krylov_grow
                bl, Vfin = _grow_frame(accL, Kex, ex.V_ex, :left, i, lch,
                                       right_channels(rstack, i + 2), max(ex.n_new_r, 1))
                Vwide[i] = Vfin          # accounting must see the width the growth reached
                bl
            else
                rch1 = push_right_channels(right_channels(rstack, i + 2), mpo, ex.V_ex, i + 1)
                H1 = shift_h(one_site_h(mpo, i, lch, rch1), eshift)
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
        for (t, X) in pairs(keep_states)
            KX = kcarL[t] === nothing ? X[i] : to_concrete!(contract(kcarL[t], (2,), X[i], (1,)))
            W[i] = _union_left(W[i], KX)
            kcarL[t] = to_concrete!(contract(W[i]', (1, 2), KX, (1, 2)))
        end
        expanded[i] = leg_dim(W[i], 3)
        _log_bond!(halfkw(accL).ledger, bondlg, i, :left, expanded[i])
        held += tensor_elements(W[i])

        # (d) carry the ORIGINAL amplitude into the new basis. `K`, not `Kex`: `W[i]` touches
        # only legs 1-2, so this lands on `(b_i, link_i)` and composes with `psiL[i+1]`.
        C = to_concrete!(contract(W[i]', (1, 2), K, (1, 2)))
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
        kcarR = Any[nothing for _ in keep_states]   # each kept state's carry, `(link, b_j)`
        held = 0
    for j in (L - 1):-1:c
        B = D === nothing ? psiR[j + 1] :
            to_concrete!(contract(psiR[j + 1], (3,), D, (1,)))
        _fr = (t0 = time_ns(); f = _frame_from(psiR[j], B);
               accR.tframe += (time_ns() - t0) / 1e9; f)
        bondlg = adaptive ? AdaptiveLedger() : nothing         # this bond's own ledger (§8)
        ex = record!(accR, n_newR, timed_expand(accR, _fr, mpo, j,
                                left_channels(lstack, j), rch; exkw..., ledger = bondlg), j, :right)
        peak!(accR, held + _state_stored(psiR), ex.U_ex, ex.V_ex)
        Uwide[j] = ex.U_ex
        Zcbe[j]  = ex.V_ex
        held += tensor_elements(ex.U_ex) + tensor_elements(ex.V_ex)
        if adaptive
            # Row mirror of the left sweep's Stage 3 branch; see the comment there.
            M    = to_concrete!(contract(ex.U_ex', (1, 2), psiR[j], (1, 2)))  # (b_L, link_j)
            Bex  = to_concrete!(contract(M, (2,), B, (1,)))
            lch1 = push_left_channels(left_channels(lstack, j), mpo, ex.U_ex, j)
            H1   = shift_h(one_site_h(mpo, j + 1, lch1, rch), eshift)
            tkR0 = time_ns()
            Zkry, kstR = _krylov_enrich(Bex, x -> (accR.nmv += 1; apply_one_site(H1, x)),
                                        ex.V_ex, :right, tau, alloc; workcap = wcap_step)
            accR.tkry += (time_ns() - tkR0) / 1e9
            _record_krylov!(bondlg, kstR, alloc)
            peak!(accR, held + _state_stored(psiR), Zkry)
            Z[j] = to_concrete(setitag(Zkry, 1, _tagZ(j)))
        elseif krylov_basis > 0
            # Row mirror of the K half-sweep's Krylov basis; see the comment there.
            M   = to_concrete!(contract(ex.U_ex', (1, 2), psiR[j], (1, 2)))    # (b_L, link_j)
            Bex = to_concrete!(contract(M, (2,), B, (1,)))
            blocks = if krylov_grow
                bl, Ufin = _grow_frame(accR, Bex, ex.U_ex, :right, j,
                                       left_channels(lstack, j), rch, max(ex.n_new_l, 1))
                Uwide[j] = Ufin
                bl
            else
                lch1 = push_left_channels(left_channels(lstack, j), mpo, ex.U_ex, j)
                H1 = shift_h(one_site_h(mpo, j + 1, lch1, rch), eshift)
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
        for (t, X) in pairs(keep_states)
            BX = kcarR[t] === nothing ? X[j + 1] :
                 to_concrete!(contract(X[j + 1], (3,), kcarR[t], (1,)))
            Z[j] = _union_right(Z[j], BX)
            kcarR[t] = to_concrete!(contract(BX, (2, 3), Z[j]', (2, 3)))
        end
        j != c && (expanded[j] = leg_dim(Z[j], 1))
        _log_bond!(halfkw(accR).ledger, bondlg, j, :right, leg_dim(Z[j], 1))
        held += tensor_elements(Z[j])

        D = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))          # (link_j, b_j)
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
    if adaptive
        merge_ledger!(ledger, ledgerL)
        merge_ledger!(ledger, ledgerR)
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
        AR = _overlap_right(psi, Z, c)           # mirrors `half_right!`'s `D` recursion on `psi`
    end

    AL = _overlap_left(psi, W, c)                                     # (b_L, link_{c+1})
    S0 = to_concrete!(contract(AL, (2,), AR, (1,)))                   # (b_L, b_R)

    # ── GENERAL ORDER (plan: krylov_closure_cbe_bug_realtime_L16_plan.md) ───────────────────
    # The half-sweeps closed the frontier j = 0 (psi itself). For `order > 1` the frontiers
    # j = 1..order-1 of the ROOT generator are closed against the GLOBAL operator, growing `W`
    # and `Z` in place; the root environments and `S0` come back rebuilt in the final frames,
    # and the single root solve below runs in them.
    nclose = 0
    if adaptive && order > 1
        fr = KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing, Float64(eshift))
        ckw = Base.structdiff(exkw, NamedTuple{(:share_ht, :fold_omega)})
        cl = _krylov_closure!(fr, psi, mpo, S0, tau, alloc, ckw; order = order,
                              workcap = wcap_step, nrm = _nrm, parallel = parallel, rng = rng,
                              ledger = ledger, fold = fold_omega, checks = closure_checks)
        lenv_root, renv_root, S0 = fr.lenv, fr.renv, cl.S0
        for i in 1:(c - 1)
            expanded[i] = leg_dim(W[i], 3)
        end
        for j in (c + 1):(L - 1)
            expanded[j] = leg_dim(Z[j], 1)
        end
        nclose = cl.n_h1 + cl.n_root
    elseif adaptive && closure_probe > 0
        # A copied `rng` and untouched frames: the step itself stays bit-identical.
        pr = _measure_frontiers(KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing,
                                         Float64(eshift)), psi, mpo, S0, tau,
                                alloc, NamedTuple(), closure_probe; nrm = _nrm,
                                parallel = parallel, rng = copy(rng), fold = fold_omega)
        merge_ledger!(ledger, pr.ledger)
    end
    # PLAN §5 `state_projection_error`: how much of `psi` the frames miss. `W`/`Z` are
    # isometries, so `||S0|| = ||P psi||` and the miss is `sqrt(||psi||^2 - ||S0||^2)`. `psi` is
    # canonical at `L` here on both paths (`half_right!` or the `frame_state` re-gauge above).
    # On the ordinary step the frames CONTAIN `psi` and this is roundoff -- a K/L cut of a
    # physical direction would show up here; on the midpoint's second stage it is the genuine
    # error of projecting `Y_0` onto frames built from `Y_{1/2}`, charged against `tol_basis`.
    if adaptive
        npsi = norm(psi[psi.center])
        ledger.r_aug = max(maximum(expanded; init = 0), leg_dim(Z[c], 1))
        ledger.proj_err = npsi > 0 ? sqrt(max(npsi^2 - norm(S0)^2, 0.0)) / npsi : 0.0
        ledger.proj_err > max(budget.basis, _PROJ_FLOOR) && _unmet!(ledger, :projection)
    end
    peak = accL.peak + accR.peak +
           sum(tensor_elements(t) for t in (W[c], Z[c], S0); init = 0)

    H0 = shift_h(zero_site_h(mpo, c, lenv_root, renv_root, W[c], Z[c]), eshift)
    nroot = Ref(0)
    rootrep = Ref{Any}(nothing)
    _t0root = time_ns()
    S1 = expv(x -> (nroot[] += 1; apply_zero_site(H0, x)), tau, S0;
              hermitian = hermitian, maxiter = maxiter, tol = tol, reorth = reorth,
              conv_tol = root_conv_tol, substeps = root_substeps, report = rootrep,
              root_method = root_method)
    troot[] = (time_ns() - _t0root) / 1e9
    if root_report !== nothing
        # Rayleigh quotients in the Galerkin space: `S0` and `S1` ARE the state before the root
        # and after it (before the compression), and `<S|H_0|S> = <psi|P H P|psi> = <psi|H|psi>`
        # for `psi` in the frames' span, so these are exact energies, not projections of them.
        rq(S) = (n2 = norm(S)^2; n2 > 0 ? real(tensor_inner(S, apply_zero_site(H0, S))) / n2 +
                                          Float64(eshift) : NaN)
        e_pre, e_root = rq(S0), rq(S1)
        out = outside_residual ?
            estimate_global_complement_action(psi, mpo,
                KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing, Float64(eshift)),
                S1; rng = copy(rng)) : nothing
        n1 = max(norm(S1), eps())
        root_report[] = (root_method = root_method, eshift = Float64(eshift), e_pre = e_pre,
                         e_root = e_root, solve = rootrep[], n_root = nroot[],
                         outside = out === nothing ? NaN : out.bound / n1,
                         outside_left = out === nothing ? NaN : out.left / n1,
                         outside_right = out === nothing ? NaN : out.right / n1,
                         # the step's outside-Galerkin defect, `|tau| ||(I-P) H psi_1|| / ||psi_1||`
                         resid = out === nothing ? NaN : abs(tau) * out.bound / n1)
    end
    if adaptive && rootrep[] !== nothing
        ledger.expv_err   = rootrep[].err
        ledger.expv_iters = rootrep[].iterations
        # Relative slack: each substep is accepted at `tol_expv`, so only a sum over several
        # substeps (or a step that could not advance) can exceed it.
        rootrep[].err > budget.expv * (1 + 1e-8) && _unmet!(ledger, :expv)
    end

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
        # Plan §5: ONE compression of the evolved state. On the adaptive path `tol_trunc` is a
        # budget on the CUMULATIVE discarded norm over all bonds, shared out by
        # `truncate_recursive!`; the legacy path keeps its per-bond relative cut.
        _, wtr, cumw, capped = adaptive ?
            truncate_recursive!(psi, c + 1; maxdim = maxdim, weight_tol = budget.trunc,
                                bond_weights = bwts) :
            truncate_recursive!(psi, c + 1; maxdim = maxdim, cutoff = cut)
        ttrunc[] = (time_ns() - _t0tr) / 1e9
        discarded = max(discarded, wtr)
        if adaptive
            ledger.n_compress += 1
            ledger.discarded = cumw
            ledger.maxdim_reached = capped
            # Relative slack for the rounding in the per-bond shares, which sum to `tol` exactly.
            cumw > budget.trunc * (1 + 1e-8) && _unmet!(ledger, :trunc)
        end
    end
    if adaptive
        bd1 = bond_dims(psi)
        ledger.r_final = maximum(bd1; init = 0)
        # Close the per-bond log with what only exists after the root solve and compression.
        ledger.bond_log = Any[merge(e, (r_old = bd0[e.bond], r_final = bd1[e.bond],
                                        discarded = bwts[e.bond])) for e in ledger.bond_log]
    end

    # ⚠ THE PHASE TIMERS ARE SUMS ACROSS BOTH HALF-SWEEPS, i.e. CPU time, while `t_step` is WALL
    # clock. Under `parallel = false` they are the same clock and the shares sum to ~1 as before.
    # Under `parallel = true` they DELIBERATELY DO NOT: two sweeps running at once spend more CPU
    # seconds than the step takes, and a share above 1 is the speedup showing up, not a bug. Keep
    # comparing phases arm-to-arm in seconds; only the serial arm's shares are fractions of a step.
    return CBEBugSweepInfo(expanded, n_new, c, root_rank, epre, efnl,
                           discarded, accL.nmv + accR.nmv + nroot[] + nclose, peak,
                           accL.ngrow + accR.ngrow, leg_dim(Z[c], 1),
                           accL.tcbe + accR.tcbe, accL.tkry + accR.tkry, troot[], ttrunc[],
                           accL.tsplit + accR.tsplit, accL.tframe + accR.tframe,
                           accL.tenv + accR.tenv, (time_ns() - _t0step) / 1e9,
                           adaptive ? ledger : nothing)
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
    get(kwargs, :order, 1) == 1 || throw(ArgumentError(
        "order > 1 is a single-step method (cbe_bug_step!); the midpoint scheme is its own " *
        "second-order construction and does not take an order"))
    half = copy(psi)
    # stage 1: bases only. `truncate = false` keeps the augmented rank -- see the docstring.
    info1 = cbe_bug_step!(half, mpo, tau / 2; kwargs..., truncate = false)
    # stage 2: full step, frames from the midpoint, START VECTOR still `psi`.
    info2 = cbe_bug_step!(psi, mpo, tau; kwargs..., frame_state = half)
    # Stage 1 built the directions stage 2 integrates in, so a basis test it failed is the
    # step's failure too, not a discarded intermediate.
    if info2.ledger !== nothing
        rfin = info2.ledger.r_final        # the PREDICTOR's rank is uncompressed, not final
        # Both stages sweep every bond; say which visit each per-bond entry is.
        for (lg, st) in ((info1.ledger, 1), (info2.ledger, 2))
            lg.bond_log = Any[merge(e, (stage = st,)) for e in lg.bond_log]
        end
        merge_ledger!(info2.ledger, info1.ledger)
        info2.ledger.r_final = rfin
    end
    return info2
end

cbe_bug_midpoint_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_midpoint_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_midpoint_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_midpoint_step!(psi, mpo_from_terms(h), tau; kwargs...)
