
# ── THE HALF-SWEEP KRYLOV STAGE OF `cbe_bug_step!` (plan §4, Tests C and D) ─────────────────

_tagW(i::Int) = "cbeW,$i"
_tagZ(j::Int) = "cbeZ,$j"

# The residual below which the Krylov space has genuinely CLOSED -- an invariant subspace, not a
# convergence judgement.
const _KRY_BREAKDOWN = 1e-13

# ── STAGE 3: RESIDUAL-DRIVEN K/L KRYLOV ENRICHMENT (plan §4, Tests C and D) ──────────────────
#
# ⛔ TWO SPACES, AND THE TESTS LIVE IN THE SMALLER ONE. The Krylov recursion runs on ONE-SITE
# TENSORS (`(link, site, g)` on the left). What the half-sweep
# keeps is a FRAME -- an isometry on the local legs `(link, site)` -- and the plan's `Q` with
# its projector `I - QQ'` is that frame: a Krylov vector adds something only through the part
# of its local range the frame does not already span. `Q0` is the CBE output `U_ex = [U_old |
# U_CBE]`, so the old directions are kept VERBATIM and the CBE-selected ones are reused; the
# seed `K_ex` already lies in `U_old` (it is the state's own site tensor).

const _TAG_K, _TAG_KR = "cbeK,L", "cbeK,R"

"Safety bound on the adaptive Krylov depth. Reaching it with weight left is REPORTED, not hidden."
const _KRY_MAXDEPTH = 64

# The root solve's Krylov basis per substep. A NUMERICAL cap, not an
# accuracy one: `substeps = 0` splits `tau` whenever this basis cannot reach `tol_expv`.
const _ROOT_MAXITER = 30

"""
    _kry_weight(Hm, k, tau; state_seed = true) -> Float64

The weight the newest Krylov vector `v_{k+1}` carries in `exp(tau*H) v_1`, from the Arnoldi
Hessenberg after `k` steps: `|tau * beta_k * [phi_1(tau*H_k)]_{k,1}|`, read off ONE `(k+1)x(k+1)`
exponential of `[tau*H_k 0; tau*beta_k*e_k' 0]`. This is Saad's a posteriori error of the
`k`-dimensional Krylov approximation, so it is also the CLOSURE test: once it is below the
allowance, no later direction can matter more (for `k = 1` it is `|tau| beta_1 (1 + O(tau))`,
the plan's Test C normalisation `|dt| ||H1 K_ex|| / ||K_ex||`).

`state_seed`: `v_1` is the state's own tensor (the half-sweeps), not a closure frontier. It
matters only for a non-unitary propagator (below).
"""
function _kry_weight(Hm::Matrix{ComplexF64}, k::Int, tau::ComplexF64; state_seed::Bool = true)
    A = zeros(ComplexF64, k + 1, k + 1)
    A[1:k, 1:k] .= tau .* @view Hm[1:k, 1:k]
    A[k + 1, k] = tau * Hm[k + 1, k]
    real(tau) == 0 && return abs(exp(A)[k + 1, 1])
    # ⛔ NON-UNITARY (imaginary time, Lindblad): RELATIVE TO THE STATE, SHIFTED TO STAY
    # REPRESENTABLE. `|[exp(A)]_{k+1,1}|` carries the seed's own growth `g = ||exp(tau*H_k) e_1||`
    # -- `exp(dbeta*|E|)` for an unshifted generator, i.e. `e^44` at L=100, dbeta=1 -- and every
    # direction would look that much more important than it is to the state the caller keeps (and
    # overflow past dbeta*|E| ~ 700). A half-sweep seed IS the state's tensor, so dividing by `g`
    # gives the weight in the propagated state; the shift `s` cancels in that ratio exactly.
    s = maximum(real, eigvals(A[1:k, 1:k]))
    E = exp(A - s * I)
    state_seed && return abs(E[k + 1, 1]) / max(norm(@view E[1:k, 1]), floatmin(Float64))
    # A closure FRONTIER is not the state: it is made of excitations and DECAYS under the shifted
    # generator (`g = e^{-dbeta h11}`), so dividing by ITS growth inflated its directions by
    # `e^{dbeta h11}` -- 1e8 to 3e15 at dbeta = 8 on a converged L=16 chain (d49); `_extend_frame`
    # could not fit and a converged step was flagged as capped. Its weight is the absolute one in
    # the shifted frame, `|E[k+1,1]| e^s`: relative to a state whose growth over the step is 1 --
    # >= 1 in imaginary time (Jensen, the shift is its energy), so this never under-weights there.
    w = abs(E[k + 1, 1])
    return w == 0 ? 0.0 : exp(log(w) + s)
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
    # roundoff in a saturated sector must not become a column (`_cap_to_local_room`)
    U = _cap_to_local_room(Q, U, side)
    U === nothing && return (Q, 0, fits)
    fl = side === :left ? 3 : 1
    Qn = to_concrete(oplus([Q, to_concrete(setitag(U, fl, Q.inds[fl].itags))], (fl,)))
    return (_check_frame(Qn, side, "_extend_frame"), leg_dim(U, fl), fits)
end

"""
    _krylov_enrich(v0, applyH, Q0, side, tau, alloc; workcap = 0, maxdepth = _KRY_MAXDEPTH,
                   amp = 1.0, state_seed = true)
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
proportion to its size -- and, for a non-unitary propagator, never relative to its own decay
(`state_seed = false`, see `_kry_weight`).
"""
function _krylov_enrich(v0, applyH, Q0, side::Symbol, tau::ComplexF64,
                        alloc::BasisAllocation; workcap::Int = 0,
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                        state_seed::Bool = true)
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
            # ⛔ CGS2 against EVERY previous vector, the correction accumulated lazily and subtracted
            # ONCE per pass: `2m` materialisations per solve instead of `m^2` (a per-vector update
            # allocated 28.6 GiB per step at L=20, chi=32, d=4). Classical Gram-Schmidt repeated
            # twice is as stable as modified ("twice is enough"); the coefficients accumulate
            # across both passes, so column `k` of `Hm` is the true projection of `H v_k`.
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
            weight = amp * _kry_weight(Hm, k, teff; state_seed = state_seed)
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

"""
    _merge_frame(Q, U, side) -> Q'

`Q` grown by the part of the frame `U` it does not span, cut only at the numerical floor (the
`krylov_reuse = false` ablation: the CBE directions join after the Krylov stage instead of
seeding it).
"""
function _merge_frame(Q, U, side::Symbol)
    R = to_concrete(side === :left ? perp_component(Q, U) : perp_component_right(Q, U))
    # `U` inside `Q` (no CBE direction this step): Telum drops the cancelled blocks and its SVD of
    # a blockless tensor throws (hq34b 17436300)
    norm(R) == 0 && return Q
    return first(_extend_frame(Q, R, U, side, 1.0, 0.0))
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
