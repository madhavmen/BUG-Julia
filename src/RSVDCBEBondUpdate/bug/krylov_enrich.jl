
# ── THE HALF-SWEEP KRYLOV STAGE OF `cbe_bug_step!` (plan §4, Tests C and D) ─────────────────

_tagW(i::Int) = "cbeW,$i"
_tagZ(j::Int) = "cbeZ,$j"

# The residual below which the Krylov space has genuinely CLOSED -- an invariant subspace, not a
# convergence judgement.
const _KRY_BREAKDOWN = 1e-13

# ── STAGE 3: residual-driven K/L Krylov enrichment (plan §4, Tests C and D) ──
# Krylov vectors live on one-site tensors, but the plan's projector `I - QQ'` acts on the frame (an
# isometry on `(link, site)`): a vector adds only through the part of its local range the frame
# lacks. `Q0` is the CBE output `U_ex`, so old directions are kept verbatim; the seed `K_ex` already
# lies in `U_old`.

const _TAG_K, _TAG_KR = "cbeK,L", "cbeK,R"

"Safety bound on the adaptive Krylov depth. Reaching it with weight left is REPORTED, not hidden."
const _KRY_MAXDEPTH = 64

# The root solve's Krylov basis per substep. A NUMERICAL cap, not an
# accuracy one: `substeps = 0` splits `tau` whenever this basis cannot reach `tol_expv`.
const _ROOT_MAXITER = 30

# Relative Hermitian part of `tau*H_k` below which the Krylov propagator counts as unitary: CGS2
# leaves a Hermitian generator's Hessenberg Hermitian to ~1e-15, and a loss term far below this
# gives the same weight in both branches anyway.
const _KRY_SKEW_TOL = 1e-12

"""
    _kry_weight(Hm, k, tau; state_seed = true) -> Float64

The weight the newest Krylov vector `v_{k+1}` carries in `exp(tau*H) v_1`, from the Arnoldi
Hessenberg after `k` steps: `|tau * beta_k * [phi_1(tau*H_k)]_{k,1}|`, read from one `(k+1)x(k+1)`
exponential. This is Saad's a posteriori error of the `k`-dimensional approximation, so it is also
the closure test: below the allowance, no later direction matters more.

`state_seed` is false when `v_1` is a closure frontier rather than the state's own tensor; it
matters only for a non-unitary propagator.

The propagator is unitary when `tau*H_k` is skew-Hermitian, not whenever `real(tau) == 0`: a
no-jump generator `H - (i/2)Q` at `tau = -im*dt` contracts, and its weight must be relative to the
decayed state, not to the start norm.
"""
function _kry_weight(Hm::Matrix{ComplexF64}, k::Int, tau::ComplexF64; state_seed::Bool = true)
    A = zeros(ComplexF64, k + 1, k + 1)
    A[1:k, 1:k] .= tau .* @view Hm[1:k, 1:k]
    A[k + 1, k] = tau * Hm[k + 1, k]
    Ak = @view A[1:k, 1:k]
    norm(Ak + Ak') <= _KRY_SKEW_TOL * norm(Ak) && return abs(exp(A)[k + 1, 1])
    # Non-unitary (imaginary time, Lindblad, no-jump loss): weigh relative to the state's own
    # growth `g = ||exp(tau*H_k) e_1||`, which is `exp(dbeta*|E|)` unshifted and overflows at large dbeta. The
    # shift `s` cancels in that ratio exactly.
    s = maximum(real, eigvals(A[1:k, 1:k]))
    E = exp(A - s * I)
    state_seed && return abs(E[k + 1, 1]) / max(norm(@view E[1:k, 1]), floatmin(Float64))
    # A closure frontier is not the state: it decays under the shifted generator, so dividing by its
    # own growth inflated its directions by `e^{dbeta h11}`, enough to block a converged step. Its
    # weight is the absolute one, `|E[k+1,1]| e^s`, in the shifted frame, which never under-weights
    # in imaginary time (the shift is the energy there).
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
                   amp = 1.0, state_seed = true, recycled = ())
        -> (Q, stats)

Grow the frame `Q0` by the Krylov directions of `applyH` from seed `v0` that the step needs. Each
direction is scored by `weight * ||(I - QQ') v_k||`: `eta_1` is Test C, `eta_{k>=2}` Test D. Scores
above `alloc.krylov` are appended. A small `eta_1` is not a stop (`H e1 = e2`, `H e2 = e1 + e3` with
`Q0 = [e1, e2]`). `capped` with `met = false` means the work ceiling or depth bound stopped the
search with weight left. `amp` is the seed's norm relative to the state. `stats.basis` and
`stats.weights` are the Arnoldi vectors and their weights, the last vector the active frontier: what
a later round recycles. `recycled` augments the recurrence ([`_krylov_basis`](@ref)).
"""
function _krylov_enrich(v0, applyH, Q0, side::Symbol, tau::ComplexF64,
                        alloc::BasisAllocation; workcap::Int = 0,
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                        state_seed::Bool = true, recycled = ())
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    perp = side === :left ? perp_component : perp_component_right
    fl = side === :left ? 3 : 1
    Q = Q0
    eta_c, eta_d, added, added_d = 0.0, 0.0, 0, 0
    function score(v, weight, first::Bool)          # Tests C/D on one direction; false = capped
        R = to_concrete(perp(Q, v))
        eta = weight * norm(R)
        first ? (eta_c = eta) : (eta_d = max(eta_d, eta))
        eta > alloc.krylov || return true
        # The ceiling bounds the frame WIDTH, not the number of extensions: a block may not carry
        # the frame past it either.
        room = workcap > 0 ? workcap - leg_dim(Q, fl) : typemax(Int)
        room <= 0 && return false
        Q, nadd, fits = _extend_frame(Q, R, v, side, weight, alloc.krylov; room = room)
        added += nadd
        first || (added_d += nadd)
        return fits
    end
    kb = _krylov_basis(v0, applyH, tau, alloc.krylov, score; recycled = recycled,
                       maxdepth = maxdepth, amp = amp, state_seed = state_seed)
    return (Q, (eta_c = eta_c, eta_d = eta_d, added = added, n_h1 = kb.n_h1,
                d_after_c = eta_c <= alloc.krylov && added_d > 0,
                capped = kb.capped, met = kb.closed && !kb.capped, basis = kb.basis,
                weights = kb.weights, n_recycled = kb.n_recycled, rec_closed = kb.rec_closed))
end

# The recurrence runs on the local tensors and, for the matrix tests, on plain vectors.
_kinner(a, b) = tensor_inner(a, b)
_kinner(a::Vector{<:Number}, b::Vector{<:Number}) = dot(a, b)
_kconc(x) = to_concrete(x)
_kconc(x::Vector{<:Number}) = x

"A recycled vector whose part outside the block is below this fraction of it adds nothing new."
const _RECYCLE_FLOOR = 1e-8

# A recycled direction the chain has absorbed down to this fraction is dropped: its image is a
# difference of stored images, accurate only to `eps ||H|| / fraction`. Dropping only ever delays
# the joint-basis stop; the chain's own stop still holds.
const _RECYCLE_DEFLATE = 1e-6

"""
    _krylov_basis(v0, applyH, tau, allowance, score; recycled = (), maxdepth, amp, state_seed)
        -> (basis, weights, n_h1, closed, capped, n_recycled, rec_closed)

The local Krylov recurrence of the frame stages. `score(v, weight, first) -> fits` takes each
direction to the frame (`first` = Test C's direction), `false` stopping the search as capped.

Without `recycled` it is Arnoldi from `v0`: each new vector weighed by Saad's estimate
(`_kry_weight`), closed once that weight is within `allowance`.

With `recycled` (plan §1.2: "transport/reorthogonalize old useful directions into the new gauge, and
extend only from active frontiers"; §1.3: `tol_basis` carries "the recycled local Krylov basis
defect"), only `v0`'s own Arnoldi chain is ever extended -- its newest vector is the active
frontier -- so its span always holds the Krylov space of `v0` and its stop is the fresh one. The
recycled directions are reorthogonalised against the chain as it grows, their images following by
linearity (one `H1` each, once). After every chain step the defect of the joint basis `B = [chain,
recycled']`, `d = tau (HB - B A) phi_1(tau A) e1` with `A = B'HB`, is tested as well: within the
allowance, the recycled directions join the frame weighted by their coefficient in
`exp(tau A) e1`, and the search stops early (`rec_closed`). Otherwise they never reach the frame.
At most `length(recycled)` `H1` more than the fresh stage, and when the chain closes it appends
exactly what the fresh stage appends.

Extending along `d` instead (one vector per step from a block whose every vector leaves a
residual) is a block recurrence advancing one direction at a time: in the matrix tests it ran to the
depth bound with errors 1e-5..1e-2 at a 1e-7 allowance whenever the recycled vectors were not the
seed's own.
"""
function _krylov_basis(v0, applyH, tau::ComplexF64, allowance::Float64, score;
                       recycled = (), maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                       state_seed::Bool = true)
    nh, capped, closed, rec_closed = 0, false, false, false
    b0 = norm(v0)
    b0 > 0 || return (basis = Any[], weights = Float64[], n_h1 = 0, closed = true,
                      capped = false, n_recycled = 0, rec_closed = false)
    # `tau = 0` asks the unit-time closure question, as `basis_allocation` does.
    teff = iszero(tau) ? one(tau) : tau
    vs, ws = Any[_kconc((1.0 / b0) * v0)], Float64[amp]
    X = Any[]                                   # recycled, orthonormal, orthogonal to the seed
    for c in recycled
        x = _cgs2(vcat(vs, X), c)
        nx = norm(x)
        nx > _RECYCLE_FLOOR * max(norm(c), eps()) && push!(X, _kconc((1.0 / nx) * x))
    end
    nrec = length(X)
    HX = Any[]
    for x in X
        push!(HX, _kconc(applyH(x)))
        nh += 1
    end
    HP, Rp, gp = Any[], Any[], Float64[]        # chain images; recycled' and weights at last test
    Hm = zeros(ComplexF64, maxdepth + 1, maxdepth + 1)
    for k in 1:maxdepth
        w = applyH(vs[end])
        nh += 1
        isempty(X) || push!(HP, _kconc(w))
        # ⛔ CGS2 against EVERY previous vector, the correction accumulated lazily and subtracted
        # ONCE per pass: `2m` materialisations per solve instead of `m^2` (a per-vector update
        # allocated 28.6 GiB per step at L=20, chi=32, d=4). Classical Gram-Schmidt repeated
        # twice is as stable as modified ("twice is enough"); the coefficients accumulate
        # across both passes, so column `k` of `Hm` is the true projection of `H v_k`.
        # Never against the recycled block: the chain must stay `v0`'s Krylov sequence.
        for _pass in 1:2
            corr = nothing
            for (t, u) in pairs(vs)
                ov = _kinner(u, w)
                Hm[t, k] += ov
                corr = corr === nothing ? ov * u : corr + ov * u
            end
            corr === nothing || (w = _kconc(w - corr))
        end
        b = norm(w)
        b <= _KRY_BREAKDOWN && (closed = true; break)    # invariant: nothing further exists
        if !isempty(X)
            Rp, HRp = _rk_deflate!(X, HX, vs[k], HP[k])
            if !isempty(Rp)
                A, nd, g = _joint_defect(vs, HP, Rp, HRp, Hm, k, teff)
                gp = amp .* abs.(g[(k + 1):end])
                if amp * nd <= allowance
                    closed = rec_closed = true
                    for (l, r) in pairs(Rp)
                        score(r, gp[l], false) || (capped = true; break)
                    end
                    break
                end
            end
        end
        Hm[k + 1, k] = b
        v = _kconc((1.0 / b) * w)
        push!(vs, v)
        weight = amp * _kry_weight(Hm, k, teff; state_seed = state_seed)
        push!(ws, weight)
        score(v, weight, k == 1) || (capped = true; break)
        weight <= allowance && (closed = true; break)
    end
    closed || (capped = true)                             # depth bound with weight left
    # the chain's newest vector stays last: the active frontier a later round keeps first
    basis = isempty(Rp) ? vs : vcat(vs[1:1], Rp, vs[2:end])
    weights = isempty(Rp) ? ws : vcat(ws[1:1], gp, ws[2:end])
    return (basis = basis, weights = weights, n_h1 = nh, closed = closed, capped = capped,
            n_recycled = nrec, rec_closed = rec_closed)
end

"""
    _rk_deflate!(X, HX, p, Hp) -> (R, HR)

Remove the new chain vector `p` from the recycled directions `X` (kept unnormalised, so no rounding
compounds across steps) and their images `HX` by linearity, drop what the chain has absorbed, and
return the rest orthonormalised with their images.
"""
function _rk_deflate!(X, HX, p, Hp)
    for i in eachindex(X), _pass in 1:2
        o = _kinner(p, X[i])
        X[i] = _kconc(X[i] - o * p)
        HX[i] = _kconc(HX[i] - o * Hp)
    end
    keep = [norm(x) > _RECYCLE_DEFLATE for x in X]
    keepat!(X, keep)
    keepat!(HX, keep)
    R, HR = Any[], Any[]
    for (x, hx) in zip(X, HX)
        for _pass in 1:2, (r, hr) in zip(R, HR)
            o = _kinner(r, x)
            x = _kconc(x - o * r)
            hx = _kconc(hx - o * hr)
        end
        nx = norm(x)
        nx > _RECYCLE_DEFLATE || continue
        push!(R, _kconc((1.0 / nx) * x))
        push!(HR, _kconc((1.0 / nx) * hx))
    end
    return R, HR
end

"""
    _joint_defect(P, HP, R, HR, Hm, k, tau) -> (A, ||d||, g)

The Galerkin block `A = B'HB` of `B = [P[1:k], R]` (the chain block read from the Arnoldi `Hm`), the
norm of its defect `d = tau (HB - B A) phi_1(tau A) e1`, and `g = exp(tau A) e1`.
"""
function _joint_defect(P, HP, R, HR, Hm, k::Int, tau::ComplexF64)
    n = k + length(R)
    B = vcat(P[1:k], R)
    HB = vcat(HP[1:k], HR)
    A = zeros(ComplexF64, n, n)
    A[1:k, 1:k] .= @view Hm[1:k, 1:k]
    for a in 1:n, c in 1:n
        (a <= k && c <= k) || (A[a, c] = _kinner(B[a], HB[c]))
    end
    g, ph = _exp_phi1_e1(tau .* A)
    d = _kconc(tau * (_lincomb(HB, ph) - _lincomb(B, A * ph)))
    return A, norm(d), g
end

"`x` with its components along the orthonormal `B` removed, classical Gram-Schmidt twice."
function _cgs2(B, x)
    for _pass in 1:2
        corr = nothing
        for u in B
            ov = _kinner(u, x)
            corr = corr === nothing ? ov * u : corr + ov * u
        end
        corr === nothing || (x = _kconc(x - corr))
    end
    return x
end

"`sum_k c[k] vs[k]`."
function _lincomb(vs, c)
    acc = nothing
    for (k, v) in pairs(vs)
        iszero(c[k]) && continue
        acc = acc === nothing ? c[k] * v : acc + c[k] * v
    end
    return acc === nothing ? _kconc(0.0 * vs[1]) : _kconc(acc)
end

"`(exp(A) e1, phi_1(A) e1)` from one augmented exponential: `exp([A e1; 0 0])`."
function _exp_phi1_e1(A::AbstractMatrix)
    n = size(A, 1)
    Z = zeros(ComplexF64, n + 1, n + 1)
    Z[1:n, 1:n] .= A
    Z[1, n + 1] = 1
    E = exp(Z)
    return E[1:n, 1], E[1:n, n + 1]
end

"""
    _krylov_enrich_recycled(v0, recycled, applyH, Q0, side, tau, alloc; workcap, maxdepth, amp)
        -> (Q, stats)

The local Krylov stage of a closure round that holds the previous round's useful directions at
this bond, already transported into this round's gauge: [`_krylov_enrich`](@ref) with `recycled`
augmenting `v0`'s own chain ([`_krylov_basis`](@ref)). Closure frontiers only (`state_seed =
false`: absolute weights).
"""
_krylov_enrich_recycled(v0, recycled::AbstractVector, applyH, Q0, side::Symbol, tau::ComplexF64,
                        alloc::BasisAllocation; workcap::Int = 0, maxdepth::Int = _KRY_MAXDEPTH,
                        amp::Float64 = 1.0) =
    _krylov_enrich(v0, applyH, Q0, side, tau, alloc; workcap = workcap, maxdepth = maxdepth,
                   amp = amp, state_seed = false, recycled = recycled)

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
