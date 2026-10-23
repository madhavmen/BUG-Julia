# cbe_bond_update.jl: the gate-based CBE bond update and its sweep (from `cbe_core.jl`, included right after it).

# ── the bond update ──────────────────────────────────────────────────────────

"""
    cbe_bond_update(f::BondFrame, gate, tau; kwargs...) -> NamedTuple

One CBE-basis Galerkin update on the bond `f` snapshots -- a drop-in replacement for
`BondUpdateBUG.kls_bond_update` with the K/L solves and the random sector fill both replaced
by [`cbe_expand_bond`](@ref).

Returns the same fields `kls_bond_update` does (`left_core`, `right_core`, `U_aug`, `V_aug`,
`S_new`, `n_new_k`, `n_new_l`, `keep`, `aug_k`, `aug_l`, `svals`, `discarded`) so it slots
into the same sweep, plus `err_pre` / `err_fnl` from the expansion.

Keywords: `maxdim`, `trunc_thresh`, `s_tau`, `maxiter`, `tol` as in `kls_bond_update`;
`expand = false` freezes the frames for a fixed-rank projector step (the analogue of
`augment = false`); everything else forwards to `cbe_expand`.

`s_iters` selects the S-step, and its SIGN selects the scheme:

  - `s_iters = 1` (default) --- one expansion, then a plain Lanczos. The original path.
  - `s_iters = n > 1` --- repeat the expansion `n` times, each pass re-sketching from the
    state the last pass evolved. THE BEST SCHEME MEASURED once the budget is below the room:
    at `growth = 1.25` it takes the error from `1.449e-11` to `1.978e-15` (rank 9 -> 12). Flat
    at `growth = 2.0` only because the budget already saturates the room there and nothing is
    left to find. See the comment at the loop.
  - `s_iters = -g <= 0` --- the EXPANDING-BASIS LANCZOS of [`_expanding_krylov`](@ref), with
    `g` the number of leading Krylov steps that carry a growth pass (`s_iters = 0` grows
    none, which is the plain Lanczos with no expansion at all). The basis grows around each
    Krylov vector before `H` is applied to it, so the bond dimension runs `D`, `growth*D`,
    `growth^2*D`, ..., and the stopping signal is the Lanczos `beta` itself.

`iter_tol` applies only to the `s_iters > 1` control. Returned `n_iter` is the number of
growth passes performed and `iter_weight` is the final `beta` for the expanding solver, the
last new-direction weight for the control.

EXACTNESS AT FULL RANK. `U_ex` contains `U0` as its first block and `V_ex` contains `V0`, so
`S_hat0 = U_ex' Theta0 V_ex'` is lossless and `tau = 0` is the identity on the state. When
the expanded frames span the whole local space the Galerkin generator IS the gate, so the
step is the exact two-site propagator -- the same property the validated kernel has, and the
first thing the tests check.
"""
function cbe_bond_update(f::BondFrame, gate, tau::ComplexF64;
                         maxdim::Int = 200,
                         trunc_thresh::Float64 = 1e-14,
                         s_tau::Union{Nothing, ComplexF64} = nothing,
                         expand::Bool = true,
                         maxiter::Int = 30,
                         tol::Float64 = 1e-15,
                         s_iters::Int = 1,
                         iter_tol::Float64 = 1e-12,
                         s_reorth::Bool = false,
                         s_probe::Symbol = :residual,
                         rng::AbstractRNG = MersenneTwister(0x5EED),
                         expand_kwargs...)
    tau_s = s_tau === nothing ? tau : s_tau
    tl, tr = f.site_l.itags, f.site_r.itags
    theta0 = frame_theta(f)

    # COUNTED, not inferred. `expv` returns only the vector, and it belongs to the validated
    # module, so the iteration count is taken here instead: one increment per operator
    # application IS the Krylov dimension the Lanczos used. It is what distinguishes "this
    # setting does more work per step" from "this setting needs more steps of work", and the
    # budget A/B needs that distinction -- a starved basis can be slower while building a
    # smaller expansion, which no timing alone can attribute.
    nmv = Ref(0)

    # The two closures that are all a mode gets to change. The gate path supplies the bare
    # two-site gate here (its environments are the identity by canonical form -- see
    # `CBEBugOptions`); the MPO path supplies the environment-dressed `apply_h_two_site`.
    applyH = th -> apply_gate(gate, th, tl, tr)
    expand_at = fr -> cbe_expand_bond(fr, gate; rng = rng, expand_kwargs...)

    # The Galerkin step in a GIVEN basis. Note `S_start` is rebuilt from `theta0` every time it
    # is called -- see the loop below for why that is the whole correctness argument.
    function galerkin(U_aug, V_aug)
        S_start = to_concrete!(contract(contract(U_aug', (1, 2), theta0, (1, 2)),
                                       (2, 3), V_aug', (2, 3)))
        function apply_s(x)
            nmv[] += 1
            theta = to_concrete((U_aug * x) * V_aug)
            evolved = apply_gate(gate, theta, tl, tr)
            proj = contract(U_aug', (1, 2), evolved, (1, 2))
            return to_concrete!(contract(proj, (2, 3), V_aug', (2, 3)))
        end
        return expv(apply_s, tau_s, S_start; hermitian = true, maxiter = maxiter,
                    tol = tol, reorth = s_reorth)
    end

    U_aug, V_aug = f.U0, f.V0
    err_pre, err_fnl, n_iter, iter_weight = 0.0, 0.0, 0, 0.0
    local S_new

    if !expand
        S_new = galerkin(U_aug, V_aug)
    elseif s_iters <= 0
        # ── GROWTH INSIDE THE LANCZOS ────────────────────────────────────────────────
        # `s_iters <= 0` selects the expanding-basis solver; see `_expanding_krylov`.
        U_aug, V_aug, S_new, info = _expanding_krylov(
            f, applyH, expand_at, Propagate(tau_s), theta0, nmv;
            maxiter = maxiter, tol = tol, grow_iters = -s_iters, probe = s_probe)
        err_pre, err_fnl = info.err_pre, info.err_fnl
        n_iter, iter_weight = info.n_grow, info.beta_final
    else
        # ── ONE EXPANSION, THEN A PLAIN LANCZOS (the default) ────────────────────────
        # `s_iters = 1` is the original single expansion and is the default path; `n > 1`
        # repeats it, re-sketching each pass from the state the previous pass evolved.
        #
        # THIS IS THE BEST SCHEME MEASURED, and the reason is NOT that it builds a Krylov
        # space -- it does not. With tau small, exp(tau*H)Theta0 = Theta0 + tau*H*Theta0 +
        # O(tau^2), so each pass re-derives H*Theta0, never H^2*Theta0. What it does instead
        # is COMPOUND THE GROWTH SCHEDULE: the budget is recomputed from the CURRENT rank
        # every pass, so r -> growth*r -> growth^2*r, and each step of that ladder is probed
        # by a sketch whose width is matched to the rank it is probing. A sequence of
        # well-conditioned narrow probes beats one wide draw.
        #
        # MEASURED at L=8 bond (4,5) against the exact two-site propagator, growth = 1.25:
        # one-shot 1.449e-11 at rank 9, three passes 1.978e-15 at rank 12. The expanding
        # Lanczos reaches the SAME rank 12 and stays at 1.450e-11 -- same basis size, four
        # orders worse, so what differs is which directions get admitted and not how many.
        #
        # It is flat at growth = 2.0 only because the budget already saturates the room there
        # (room = d*chi - r -> r, budget -> r for d = 2), so the first pass takes the whole
        # local space and there is nothing left for a second one to find.
        #
        # `galerkin` always rebuilds S_start from the ORIGINAL theta0, never from the previous
        # pass's S_new, so exp(tau*H_eff) is applied exactly once however many passes run.
        probe = f
        for k in 1:s_iters
            ex = expand_at(probe)
            U_prev, V_prev = U_aug, V_aug
            U_aug, V_aug = ex.U_ex, ex.V_ex
            err_pre, err_fnl, n_iter = ex.err_pre, ex.err_fnl, k

            S_new = galerkin(U_aug, V_aug)

            k == s_iters && break
            (ex.n_new_l + ex.n_new_r) == 0 && break
            iter_weight = _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S_new)
            iter_weight < iter_tol && break

            probe = _reframe(f, U_aug, S_new, V_aug)
        end
    end

    res = svd(S_new, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true)
    tag = f.link_mid.itags
    left_core  = to_concrete(setitag(to_concrete(U_aug * res.U), 3, tag))
    right_core = to_concrete(setitag(to_concrete((res.S * res.Vd) * V_aug), 1, tag))

    svals = [s for (s, _, _, _) in res.kept_list]
    return (; left_core, right_core, U_aug, V_aug, S_new,
            n_new_k = leg_dim(U_aug, 3) - f.old_rank,
            n_new_l = leg_dim(V_aug, 1) - f.old_rank,
            keep = leg_dim(left_core, 3),
            aug_k = leg_dim(U_aug, 3), aug_l = leg_dim(V_aug, 1),
            svals, discarded = _trunc_weight(res),
            err_pre, err_fnl, n_matvec = nmv[], n_iter, iter_weight)
end

"""
    _expanding_krylov(f, applyH, expand, solver, theta0, nmv; ...) -> (U, V, S, info)

Lanczos for `exp(tau*H_eff)` whose BASIS GROWS WITH THE KRYLOV SPACE.

The expansion is not a preprocessing step here, it is step 0 of each Lanczos iteration: before
applying `H` to the current Krylov vector `v_k`, the frames are expanded using `v_k` itself as
the probe, so the enlarged basis can hold `H*v_k`. The bond dimension therefore grows
geometrically with the Krylov dimension --- `D`, `growth*D`, `growth^2*D`, ... --- until `room`
or `grow_iters` stops it.

WHY THIS AND NOT AN OUTER LOOP. Re-running a converged exponential and re-sketching from its
result asks for the same subspace every time: `exp(tau*H)Theta = Theta + tau*H*Theta + O(tau^2)`,
so the sketch keeps rediscovering `H*Theta`. Inside the Lanczos, `v_k` genuinely carries `H^k`,
so each growth pass is probing a direction that did not exist before. That is what makes the
basis converge on the relevant subspace rather than having to be tuned onto it by `comp_ratio`
and the oversampling.

STOPPING is the Lanczos beta itself --- literally the small value in the Krylov matrix. When
`beta_k < tol` the Krylov space has closed and no further growth can help.

TWO PROPERTIES WORTH BEING PRECISE ABOUT, because they are approximations and not identities:

  - `alpha_k` is EXACT. `v_k` lies in the range of the projector `P_k` in force at step `k`, so
    `<v_k, P_k H P_k v_k> = <v_k, H v_k>` regardless of how small that basis is.
  - `beta_k` is NOT exact: it measures the residual of `P_k H v_k`, missing whatever part of
    `H v_k` falls outside the step-`k` basis. Making that part small is precisely the growth
    pass's job, which is why the expansion runs BEFORE the operator is applied and not after.
    The resulting tridiagonal is therefore a Galerkin approximation built across a nested
    sequence of subspaces, not the exact tridiagonal of a fixed operator.

FULL REORTHOGONALISATION IS MANDATORY here, not the optional `reorth` of `lanczos_expv`. The
three-term recurrence keeps the basis orthogonal only for a FIXED operator; the projected
operator changes at every growth pass, so earlier vectors stop being exact Krylov vectors of the
current one and the recurrence alone would drift.
"""
function _expanding_krylov(f::BondFrame, applyH, expand, solver::SStepSolver, theta0, nmv;
                           maxiter::Int, tol::Float64, grow_iters::Int,
                           probe::Symbol = :residual)
    project(U, V, th) = to_concrete!(contract(contract(U', (1, 2), th, (1, 2)),
                                             (2, 3), V', (2, 3)))

    # Embed a core from the (Uo,Vo) basis into (Un,Vn). `oplus` keeps the old block first so
    # this is `[S; 0]`, but it is COMPUTED as an overlap rather than assumed, which keeps it
    # correct if the orthonormality guard rotates within the appended span.
    #
    # THE OVERLAPS ARE BUILT ONCE PER GROWTH PASS, NOT ONCE PER VECTOR. `PL` and `PR` depend
    # only on the frames, so computing them inside a per-vector `embed` -- as the first version
    # did -- repeated two `O(d*chi*r_old*r_new)` contractions for every stored Krylov vector.
    # With a Krylov dimension of ~30 that is ~30x the frame work per pass, and it grows with
    # chi. Hoisted here; the per-vector cost is now the two `O(chi^3)` core contractions alone.
    embed_ops(Uo, Vo, Un, Vn) =
        (to_concrete!(contract(Un', (1, 2), Uo, (1, 2))),         # PL (new_l, old_l)
         to_concrete!(contract(Vo, (2, 3), Vn', (2, 3))))         # PR (old_r, new_r)

    embed_with(PL, PR, S) =
        to_concrete!(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))

    U, V = f.U0, f.V0
    S0 = project(U, V, theta0)
    beta0 = norm(S0)
    err_pre, err_fnl, n_grow = 0.0, 0.0, 0
    # `value = NaN` is NOT decoration: every other exit carries it (the Ritz value for
    # `LowestEigen`), and a caller that reads `info.value` -- the ground-state solve does --
    # would hit an UndefVarError on this branch if the field were simply absent here.
    beta0 == 0 && return (U, V, S0,
                          (; err_pre, err_fnl, n_grow, beta_final = 0.0, ranks = Int[],
                             value = NaN))

    basis = Any[to_concrete((1.0 / beta0) * S0)]
    alpha, betas, ranks = Float64[], Float64[], Int[leg_dim(U, 3)]
    b_last = 0.0

    image = nothing        # the PREVIOUS iteration's output, H*v_{k-1}, before orthogonalisation

    for it in 1:maxiter
        # ---- step 0: grow the basis around the probe ----
        #
        # WHICH VECTOR DRIVES THE EXPANSION IS THE WHOLE QUESTION, and the two choices are not
        # close numerically:
        #
        #   :residual  (DEFAULT) the orthonormal Lanczos vector `basis[it]`, i.e. the Krylov
        #              vector itself as the centre gauge. `Theta = U*v_k*V` is then in mixed
        #              bond-canonical form -- VERIFIED, `U'U - I` and `VV' - I` both 1.5e-15
        #              and `||S|| = ||U S V|| = 1` at every pass.
        #   :image     the previous step's output `H*v_{k-1}`, before the Lanczos subtraction.
        #              TRIED AND IT MAKES NO DIFFERENCE: at growth = 1.25 both give 1.450e-11,
        #              with :image at rank 11 against :residual's 12. Kept only so the null
        #              result stays reproducible; it is not a knob worth turning.
        #
        # NEITHER IS THE RANDOM GAUSSIAN. The chosen vector becomes the CORE `S0` of the frame
        # handed to `cbe_expand_bond`, which then draws its own sector-graded Gaussian inside
        # `sector_graded_sketch` exactly as on the first pass. Every other step -- preselection,
        # the P_perp projection, the ranking, the direct sum -- is unchanged.
        #
        # The Lanczos RECURRENCE is untouched either way: `H` is still applied to the
        # orthonormal `basis[it]`, so alpha/beta and the tridiagonal are unchanged. Only the
        # basis the next vector is built IN differs.
        #
        # Why H is applied to the ORTHONORMALISED vector and not to the raw output: measured,
        # `H*v_k` overlaps an EARLIER basis vector by 100% of its own norm (k = 2 and 3 at the
        # L=8 test bond), so without the beta subtraction the "next Krylov vector" is
        # numerically the previous one and the iteration stalls. The alpha subtraction, by
        # contrast, IS redundant here: `<v_k, H v_k>` is 1e-15 to 1e-20 on this Hamiltonian.
        if it <= grow_iters
            pv = (probe === :image && image !== nothing) ? image : basis[it]
            ex = expand(_reframe(f, U, pv, V))
            if (ex.n_new_l + ex.n_new_r) > 0
                Uo, Vo = U, V
                U, V = ex.U_ex, ex.V_ex
                PL, PR = embed_ops(Uo, Vo, U, V)
                for k in eachindex(basis)
                    basis[k] = embed_with(PL, PR, basis[k])
                end
                image === nothing || (image = embed_with(PL, PR, image))
                err_pre, err_fnl, n_grow = ex.err_pre, ex.err_fnl, n_grow + 1
            end
        end
        push!(ranks, leg_dim(U, 3))

        nmv[] += 1
        theta = to_concrete((U * basis[it]) * V)
        w = project(U, V, applyH(theta))
        image = w          # this iteration's output, carried to the next expansion

        a = real(tensor_inner(basis[it], w))
        push!(alpha, a)
        w = to_concrete(w + (-a) * basis[it])
        it > 1 && (w = to_concrete(w + (-betas[it - 1]) * basis[it - 1]))
        for _pass in 1:2, u in basis
            w = to_concrete(w - tensor_inner(u, w) * u)
        end

        # `b_last` is the TERMINATING residual, not the last accepted one. Reporting
        # `betas[end]` instead would show the last beta that was large enough to continue --
        # the opposite of the stopping signal it is meant to document.
        b = norm(w)
        b_last = b
        b < tol && break
        it == maxiter && break
        push!(betas, b)
        push!(basis, to_concrete((1.0 / b) * w))
    end

    # THE ONLY MODE-DEPENDENT LINE IN THE WHOLE FUNCTION. `coeff` already carries `beta0` for
    # the propagator and deliberately does not for the eigensolve, so the accumulation below
    # is identical either way.
    coeff, value = _reduce_krylov(solver, alpha, betas, beta0)
    S = to_concrete(coeff[1] * basis[1])
    for k in 2:length(coeff)
        S = to_concrete(S + coeff[k] * basis[k])
    end
    return (U, V, S, (; err_pre, err_fnl, n_grow, beta_final = b_last, ranks, value))
end

"""
    _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S) -> Float64

Fraction of the evolved core's norm living on directions the LAST pass added.

`oplus` puts the old block first, so `U_prev` embeds in `U_aug` and `U_prev' U_aug` is the
projector back onto the old space --- `[I | 0]` in exact arithmetic, and computed rather than
assumed so the measure survives the re-orthonormalisation rotating within the span.

Returns `sqrt(‖S‖² - ‖P S Q‖²)/‖S‖`. Small means the pass produced directions the step puts
no amplitude on, i.e. the Krylov space has stopped delivering, which is the signal to stop.
Everything here is `O(χ²)`: no rank-4 tensor is formed to measure convergence.
"""
function _new_direction_weight(U_prev, V_prev, U_aug, V_aug, S)
    n = norm(S)
    n == 0 && return 0.0
    PL = to_concrete!(contract(U_prev', (1, 2), U_aug, (1, 2)))     # (prev_l, new_l)
    PR = to_concrete!(contract(V_aug, (2, 3), V_prev', (2, 3)))     # (new_r, prev_r)
    S_old = to_concrete!(contract(contract(PL, (2,), S, (1,)), (2,), PR, (1,)))
    return sqrt(max(n^2 - norm(S_old)^2, 0.0)) / n
end

"""
    _reframe(f, U, S, V) -> BondFrame

`f` with its frames and core replaced, so the next sketch probes `H` against the state the
last pass evolved instead of the state the step began with.

`old_rank` is set to the LARGER of the two frame ranks. The two sides can end a pass at
different widths (`n_new_l ≠ n_new_r` is normal --- an edge bond routinely expands on one side
only), and `cbe_expand` carries a single `r` which it uses for `room = dχ - r` and for the
growth schedule. Taking the max makes the computed room a lower bound on the true room per
side, so the next pass can under-ask but never over-ask, and `room` stays the hard ceiling it
is meant to be.
"""
_reframe(f::BondFrame, U, S, V) =
    BondFrame(U, S, V, f.link_l, f.site_l, f.link_mid, f.site_r, f.link_r,
              max(leg_dim(U, 3), leg_dim(V, 1)), f.link_mid_space)

cbe_bond_update(f::BondFrame, gate, tau::Number; kwargs...) =
    cbe_bond_update(f, gate, ComplexF64(tau); kwargs...)

# ── the sweep ────────────────────────────────────────────────────────────────
