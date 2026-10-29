
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
a later round recycles. With `recycled` only the complement of the previous solve's input is
extended ([`_krylov_basis`](@ref)).
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
                weights = kb.weights, n_recycled = kb.n_recycled, rec_closed = kb.rec_closed,
                rec_overlap = kb.rec_overlap, rec_defect = kb.rec_defect))
end

# The recurrence runs on the local tensors and, for the matrix tests, on plain vectors.
_kinner(a, b) = tensor_inner(a, b)
_kinner(a::Vector{<:Number}, b::Vector{<:Number}) = dot(a, b)
_kconc(x) = to_concrete(x)
_kconc(x::Vector{<:Number}) = x

"A recycled vector whose part outside the block is below this fraction of it adds nothing new."
const _RECYCLE_FLOOR = 1e-8

"""
    _krylov_basis(v0, applyH, tau, allowance, score; recycled = (), maxdepth, amp, state_seed)
        -> (basis, weights, n_h1, closed, capped, n_recycled, rec_closed, rec_overlap, rec_defect)

The local Krylov recurrence of the frame stages. `score(v, weight, first) -> fits` takes each
direction to the frame (`first` = Test C's direction), `false` stopping the search as capped.
`basis[1]` is always the solve's input `v0 / ||v0||`, the last vector the active frontier.

Without `recycled` it is Arnoldi from `v0` ([`_arnoldi_chain`](@ref)). With `recycled` -- the
previous solve's input first, then its kept Krylov vectors, all in this round's gauge -- the input
is split as `v0 = alpha p + v_c` and only the complement `v_c` is extended
([`_rk_complement`](@ref)); when the previous block no longer holds `p`'s evolution it is Arnoldi
from `v0` after all.
"""
function _krylov_basis(v0, applyH, tau::ComplexF64, allowance::Float64, score;
                       recycled = (), maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                       state_seed::Bool = true)
    b0 = norm(v0)
    b0 > 0 || return (basis = Any[], weights = Float64[], n_h1 = 0, closed = true,
                      capped = false, n_recycled = 0, rec_closed = false, rec_overlap = 0.0,
                      rec_defect = 0.0)
    # `tau = 0` asks the unit-time closure question, as `basis_allocation` does.
    teff = iszero(tau) ? one(tau) : tau
    u0 = _kconc((1.0 / b0) * v0)
    rc = (n_h1 = 0, n_recycled = 0, rec_overlap = 0.0, rec_defect = 0.0)
    if !isempty(recycled)
        rc = _rk_complement(u0, recycled, applyH, teff, allowance, score; maxdepth = maxdepth,
                            amp = amp, state_seed = state_seed)
        haskey(rc, :basis) && return rc
    end
    ch = _arnoldi_chain(u0, applyH, teff, allowance, score; maxdepth = maxdepth, amp = amp,
                        state_seed = state_seed)
    return (basis = ch.basis, weights = ch.weights, n_h1 = rc.n_h1 + ch.n_h1, closed = ch.closed,
            capped = ch.capped, n_recycled = rc.n_recycled, rec_closed = false,
            rec_overlap = rc.rec_overlap, rec_defect = rc.rec_defect)
end

"""
    _arnoldi_chain(u, applyH, tau, allowance, score; maxdepth, amp, state_seed)
        -> (basis, weights, n_h1, closed, capped)

Arnoldi from the unit seed `u`: each new vector weighed by `amp` times Saad's estimate
(`_kry_weight`) and offered to `score`, closed once that weight is within `allowance`.
"""
function _arnoldi_chain(u, applyH, tau::ComplexF64, allowance::Float64, score;
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                        state_seed::Bool = true)
    nh, capped, closed = 0, false, false
    vs, ws = Any[u], Float64[amp]
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
            for (t, x) in pairs(vs)
                ov = _kinner(x, w)
                Hm[t, k] += ov
                corr = corr === nothing ? ov * x : corr + ov * x
            end
            corr === nothing || (w = _kconc(w - corr))
        end
        b = norm(w)
        b <= _KRY_BREAKDOWN && (closed = true; break)    # invariant: nothing further exists
        Hm[k + 1, k] = b
        v = _kconc((1.0 / b) * w)
        push!(vs, v)
        weight = amp * _kry_weight(Hm, k, tau; state_seed = state_seed)
        push!(ws, weight)
        score(v, weight, k == 1) || (capped = true; break)
        weight <= allowance && (closed = true; break)
    end
    closed || (capped = true)                             # depth bound with weight left
    return (basis = vs, weights = ws, n_h1 = nh, closed = closed, capped = capped)
end

"""
    _rk_complement(u0, recycled, applyH, tau, allowance, score; maxdepth, amp, state_seed)

The recycled solve (plan §1.2: "transport/reorthogonalize old useful directions into the new gauge,
and extend only from active frontiers"; §1.3: `tol_basis` carries "the recycled local Krylov basis
defect"). `recycled[1]` is the previous solve's input `p`, the rest its kept Krylov vectors. By
linearity `exp(tau H) u0 = alpha exp(tau H) p + exp(tau H) v_c` with `alpha = <p, u0>` and the
complement `v_c = u0 - alpha p`: the previous block `B = orth[p, recycled...]` already spans the
first term if its defect for `p` under THIS round's `H1`, `d = tau (HB - BA) phi_1(tau A) e1` with
`A = B'HB`, is small, and only `v_c` is new.

- `amp |alpha|` within the allowance: `p` carries nothing, nothing is tested (no `H1`).
- `amp |alpha| ||d|| <= allowance / 2` (one `H1` per block vector): the block joins the frame
  weighted by `amp |alpha| |exp(tau A) e1|`, and a fresh Arnoldi chain runs from `v_c` alone with
  `amp ||v_c||` and the allowance left -- unless `v_c`'s whole evolution fits in it (`2 amp ||v_c||`
  bounds it for a contraction), when `v_c` only joins the frame. Returns the solve, `rec_closed`.
- otherwise the block no longer holds `p`'s evolution (the generator moved between the rounds) and
  the caller runs Arnoldi from `u0`; returns only the counts.

The previous block is only ever tested, never extended: a block whose every vector leaves a
residual, continued one direction per step, ran to the depth bound in the matrix tests. Skipping
the test is no option either: with the generator moved by 1e-6 relative between the solves, the
complement's chain alone left errors 10x the allowance (1e-3: 1e4x).
"""
function _rk_complement(u0, recycled, applyH, tau::ComplexF64, allowance::Float64, score;
                        maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0,
                        state_seed::Bool = true)
    none = (n_h1 = 0, n_recycled = 0, rec_overlap = 0.0, rec_defect = 0.0)
    np = norm(recycled[1])
    np > 0 || return none
    p = _kconc((1.0 / np) * recycled[1])
    al = _kinner(p, u0)
    ampr = amp * abs(al)
    ampr > allowance || return merge(none, (rec_overlap = ampr,))
    B = Any[p]
    for c in Iterators.drop(recycled, 1)
        x = _cgs2(B, c)
        nx = norm(x)
        nx > _RECYCLE_FLOOR * max(norm(c), eps()) && push!(B, _kconc((1.0 / nx) * x))
    end
    HB = Any[_kconc(applyH(x)) for x in B]
    nh = length(B)
    A = ComplexF64[_kinner(B[r], HB[c]) for r in eachindex(B), c in eachindex(B)]
    g, ph = _exp_phi1_e1(tau .* A)
    dr = ampr * norm(_kconc(tau * (_lincomb(HB, ph) - _lincomb(B, A * ph))))
    counts = (n_h1 = nh, n_recycled = length(B), rec_overlap = ampr, rec_defect = dr)
    dr <= allowance / 2 || return counts
    gw = ampr .* abs.(g)
    capped, closed = false, true
    for (k, x) in pairs(B)
        score(x, gw[k], false) || (capped = true; break)
    end
    basis, weights = vcat(Any[u0], B), vcat(amp, gw)
    vc = _kconc(u0 - al * p)
    nc = norm(vc)
    rest = allowance - dr
    if !capped && nc > 0
        uc = _kconc((1.0 / nc) * vc)
        if 2 * amp * nc <= rest
            score(uc, amp * nc, false) || (capped = true)
            push!(basis, uc)
            push!(weights, amp * nc)
        else
            ch = _arnoldi_chain(uc, applyH, tau, rest, score; maxdepth = maxdepth,
                                amp = amp * nc, state_seed = state_seed)
            nh += ch.n_h1
            capped, closed = ch.capped, ch.closed
            append!(basis, ch.basis)
            append!(weights, ch.weights)
        end
    end
    return (basis = basis, weights = weights, n_h1 = nh, closed = closed, capped = capped,
            n_recycled = length(B), rec_closed = true, rec_overlap = ampr, rec_defect = dr)
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

The local Krylov stage of a closure round that holds the previous round's solve at this bond --
its input first, then its kept Krylov vectors -- already transported into this round's gauge:
[`_krylov_enrich`](@ref) extending only the complement of that input ([`_rk_complement`](@ref)).
Closure frontiers only (`state_seed = false`: absolute weights).
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
