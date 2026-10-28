
# ── RECYCLED LOCAL KRYLOV ACROSS AUGMENTATION ROUNDS (`krylov_recycle`, an ABLATION) ──────────
# xx-efficiency plan Fig. 3 / S5. An order-p step runs p augmentation rounds: the half-sweeps
# (round 1, frontier j = 0) and one enriching frontier sweep per j = 1..p-1. Each round's local
# Krylov basis at a bond is cached; the next round transports it into its own gauge -- the frame
# leg by the map `Pm` the frontier sweep already carries the frames with, the opposite leg by the
# overlap of the old and the new tail state -- re-orthonormalises it against its seed, REFRESHES
# every image with its own `H1` (a stale `H1 v` is never reused) and continues the Arnoldi
# recurrence from that block under the same time-weighted stop. Off by default.

"""
    KrylovRecycle()

One step's recycle cache, per side: the last committed round's entries `bond => (basis, weights,
frame)`, the state whose tail the entries' opposite leg lives in, and the frame generation their
frame leg is written in. A sweep writes `pend` and its entries become usable only when its frames
are committed ([`_recycle_commit!`](@ref)). `log` has one row per recycled bond visit.
"""
mutable struct KrylovRecycle
    cur::Dict{Symbol, Any}      # side => (entries, tail, gen)
    pend::Dict{Symbol, Any}
    log::Dict{Symbol, Vector{Any}}
end
KrylovRecycle() = KrylovRecycle(Dict{Symbol, Any}(:left => nothing, :right => nothing),
                                Dict{Symbol, Any}(:left => nothing, :right => nothing),
                                Dict{Symbol, Vector{Any}}(:left => Any[], :right => Any[]))

# The two half-sweeps (and the two halves of a frontier sweep) may run concurrently: each touches
# only its own side's slots, which `Dict` assignment to an EXISTING key does not resize.
_recycle_begin!(::Nothing, ::Symbol, tail, gen) = nothing
_recycle_begin!(rec::KrylovRecycle, side::Symbol, tail, gen::Int) =
    (rec.pend[side] = (entries = Dict{Int, Any}(), tail = tail, gen = gen); nothing)

"""
Store bond `i`'s round: the vectors that carried weight above `keep` (they mattered to the
response) and the last one, the active frontier the recurrence would continue from.
"""
_recycle_store!(::Nothing, args...) = nothing
function _recycle_store!(rec::KrylovRecycle, side::Symbol, i::Int, st, frame, keep::Float64)
    p = rec.pend[side]
    (p === nothing || isempty(st.basis)) && return nothing
    n = length(st.basis)
    sel = [k for k in 1:n if k == n || st.weights[k] > keep]
    p.entries[i] = (basis = st.basis[sel], weights = st.weights[sel], frame = frame)
    return nothing
end

_recycle_commit!(::Nothing, ::Symbol, ::Bool) = nothing
function _recycle_commit!(rec::KrylovRecycle, side::Symbol, keep::Bool)
    keep && (rec.cur[side] = rec.pend[side])
    rec.pend[side] = nothing
    return nothing
end

"The committed entries of `side` usable in frames generation `gen`, or `nothing`."
_recycle_entries(::Nothing, ::Symbol, ::Int) = nothing
function _recycle_entries(rec::KrylovRecycle, side::Symbol, gen::Int)
    c = rec.cur[side]
    return c === nothing || c.gen != gen ? nothing : c
end

"`t` with its internal link legs moved into the private `RB,k` namespace (as `overlap` does)."
function _rb(t, k::Int, L::Int)
    k > 1 && (t = setitag(t, 1, "RB,$k"))
    k < L && (t = setitag(t, 3, "RB,$(k + 1)"))
    return to_concrete(t)
end

"""
    _tail_overlaps(Anew, Aold, side) -> E

The overlap environments of two states with the same site legs, `Anew` the bra: `side = :left`
gives the right stack `E[k]` = sites `k..L` with legs `(bra link_k, ket link_k)` (both states
right-canonical there), `:right` the left stack `E[k]` = sites `1..k-1`. Bra links are `RB,k`.
"""
function _tail_overlaps(Anew::SymMPS, Aold::SymMPS, side::Symbol)
    L = length(Anew)
    E = Vector{Any}(nothing, L + 1)
    if side === :left
        for k in L:-1:3
            T = E[k + 1] === nothing ? Aold[k] : to_concrete!(contract(Aold[k], (3,), E[k + 1], (2,)))
            E[k] = to_concrete!(contract(_rb(Anew[k], k, L)', (2, 3), T, (2, 3)))
        end
    else
        for k in 1:(L - 2)
            T = E[k] === nothing ? Aold[k] : to_concrete!(contract(E[k], (2,), Aold[k], (1,)))
            E[k + 1] = to_concrete!(contract(_rb(Anew[k], k, L)', (1, 2), T, (1, 2)))
        end
    end
    return E
end

"""
    _recycle_transport(e, side, Pm, Fnew, E, i) -> Vector

The entry `e` of bond `i` in the current round's local space: the frame leg through `Pm` (as the
frames themselves), the opposite leg through `<new frame + tail | old frame + tail>`. A projection,
not an isometry: what the new opposite space lacks is dropped (the vectors are re-orthonormalised
by the solver).
"""
function _recycle_transport(e, side::Symbol, Pm, Fnew, E, i::Int)
    Fold = e.frame
    if side === :left                               # opposite frame: (bond, site_{i+1}, link_{i+1})
        Ek = E[i + 2]
        Fn = setitag(Fnew, 1, "RCb")
        Ek === nothing || (Fn = setitag(Fn, 3, "RB,$(i + 2)"))
        X = Ek === nothing ? Fold : to_concrete!(contract(Fold, (3,), Ek, (2,)))
        M = to_concrete!(contract(to_concrete(Fn)', (2, 3), X, (2, 3)))      # (new*, old)
        tag = Fnew.inds[1].itags
        return map(e.basis) do v
            w = Pm === nothing ? v : to_concrete!(contract(Pm, (2,), v, (1,)))
            to_concrete(setitag(to_concrete!(contract(w, (3,), M, (2,))), 3, tag))
        end
    end
    Ek = E[i]                                       # opposite frame: (link_{i-1}, site_i, bond)
    Fn = setitag(Fnew, 3, "RCb")
    Ek === nothing || (Fn = setitag(Fn, 1, "RB,$i"))
    X = Ek === nothing ? Fold : to_concrete!(contract(Ek, (2,), Fold, (1,)))
    M = to_concrete!(contract(to_concrete(Fn)', (1, 2), X, (1, 2)))          # (new*, old)
    tag = Fnew.inds[3].itags
    return map(e.basis) do v
        w = to_concrete!(contract(M, (2,), v, (1,)))
        Pm === nothing || (w = to_concrete!(contract(w, (3,), Pm, (1,))))
        to_concrete(setitag(w, 1, tag))
    end
end

"`x` minus its projection on the orthonormal `B`, CGS2 (`_krylov_enrich`'s orthogonalisation)."
function _orth_against(B, x)
    for _pass in 1:2
        corr = nothing
        for u in B
            ov = tensor_inner(u, x)
            corr = corr === nothing ? ov * u : corr + ov * u
        end
        corr === nothing || (x = to_concrete(x - corr))
    end
    return x
end

"`phi_1(A) e_1` from one augmented exponential."
function _phi1_e1(A::AbstractMatrix)
    n = size(A, 1)
    Z = zeros(ComplexF64, n + 1, n + 1)
    Z[1:n, 1:n] .= A
    Z[1, n + 1] = 1
    return exp(Z)[1:n, n + 1]
end

"""
    _krylov_enrich_recycled(v0, applyH, Q0, side, tau, alloc, C; workcap, maxdepth, amp)
        -> (Q, stats)

`_krylov_enrich` started from the block `B = orth[v0, C]` instead of `v0` alone. Every column of
`B` gets a FRESH image (`n_refresh` applications); the residual block `R = H B - B (B'HB)` then
plays the role of the Arnoldi remainder: the time-weighted defect `d = tau R phi_1(tau B'HB) e_1`
has norm `|tau beta_k [phi_1]_{k1}|` -- Saad's weight -- when `C` is empty, so the stop is the
production one. While `amp ||d||` exceeds `alloc.krylov` the recurrence continues along `d`
(one new application each). Block vectors are scored by their Galerkin coefficient, defect
directions by the defect weight, and appended to the frame by the same rule (`_extend_frame`).
"""
function _krylov_enrich_recycled(v0, applyH, Q0, side::Symbol, tau::ComplexF64,
                                 alloc::BasisAllocation, C::AbstractVector; workcap::Int = 0,
                                 maxdepth::Int = _KRY_MAXDEPTH, amp::Float64 = 1.0)
    perp = side === :left ? perp_component : perp_component_right
    fl = side === :left ? 3 : 1
    Q = Q0
    teff = iszero(tau) ? one(tau) : tau
    b0 = norm(v0)
    added, nref, nnew, eta_c, eta_d = 0, 0, 0, 0.0, 0.0
    capped, closed = false, false
    B, ws = Any[to_concrete((1.0 / b0) * v0)], Float64[amp]
    for c in C
        x = _orth_against(B, c)
        nx = norm(x)
        nx > 1e-8 * max(norm(c), eps()) && push!(B, to_concrete((1.0 / nx) * x))
    end
    nrec = length(B) - 1
    function score!(v, weight, first)
        R = to_concrete(perp(Q, v))
        eta = weight * norm(R)
        first ? (eta_c = max(eta_c, eta)) : (eta_d = max(eta_d, eta))
        eta > alloc.krylov || return true
        room = workcap > 0 ? workcap - leg_dim(Q, fl) : typemax(Int)
        room <= 0 && return false
        Q, nadd, fits = _extend_frame(Q, R, v, side, weight, alloc.krylov; room = room)
        added += nadd
        return fits
    end
    nmax = length(B) + maxdepth
    Hm = zeros(ComplexF64, nmax + 1, nmax + 1)
    Rs = Any[]
    function image!(k)                                   # column k of Hm and its residual
        y = applyH(B[k])
        for j in 1:k
            Hm[j, k] = tensor_inner(B[j], y)
        end
        r = _orth_against(B[1:k], y)
        for j in 1:k                                     # the CGS2 correction, folded back
            Hm[j, k] += tensor_inner(B[j], r)
        end
        push!(Rs, _orth_against(B[1:k], r))
    end
    # the block: images of the seed and the recycled vectors (refreshed, never reused)
    for k in 1:length(B)
        image!(k)
        nref += 1
        # later block vectors' residuals must also be orthogonal to the earlier block vectors
    end
    for k in 1:length(B), j in (k + 1):length(B)
        ov = tensor_inner(B[j], Rs[k])
        Hm[j, k] += ov
        Rs[k] = to_concrete(Rs[k] - ov * B[j])
    end
    n = length(B)
    if nrec > 0
        cG = exp(teff .* Hm[1:n, 1:n])[:, 1]
        for k in 2:n
            w = amp * abs(cG[k])
            push!(ws, w)
            score!(B[k], w, k == 2) || (capped = true)
        end
    end
    while !capped
        ph = _phi1_e1(teff .* Hm[1:n, 1:n])
        d = nothing
        for k in 1:n
            t = (teff * ph[k]) * Rs[k]
            d = d === nothing ? t : d + t
        end
        d = to_concrete(d)
        nd = norm(d)
        weight = amp * nd
        nd <= _KRY_BREAKDOWN && (closed = true; break)          # invariant: nothing further
        if weight <= alloc.krylov
            # kept, never applied: the active frontier a next round would continue from (as
            # `_krylov_enrich` keeps its last vector)
            push!(B, to_concrete((1.0 / nd) * d))
            push!(ws, weight)
            closed = true
            break
        end
        n >= nmax && (capped = true; break)
        bn = to_concrete((1.0 / nd) * _orth_against(B, d))
        bn = to_concrete((1.0 / norm(bn)) * bn)
        push!(B, bn)
        push!(ws, weight)
        for k in 1:n                                     # old residuals lose their bn part
            ov = tensor_inner(bn, Rs[k])
            Hm[n + 1, k] += ov
            Rs[k] = to_concrete(Rs[k] - ov * bn)
        end
        image!(n + 1)
        nnew += 1
        n += 1
        score!(bn, weight, n == 2) || (capped = true)
    end
    return (Q, (eta_c = eta_c, eta_d = eta_d, added = added, n_h1 = nref + nnew,
                d_after_c = eta_c <= alloc.krylov && added > 0, capped = capped,
                met = closed && !capped, basis = B, weights = ws, n_recycled = nrec,
                n_refresh = nref, n_new = nnew))
end

"""
    _recycled_or_fresh(rec, side, i, e, transport, v0, applyH, Q0, tau, alloc; kw...)

The bond's local Krylov stage: recycled from the entry `e` (transported by `transport()`) when
there is one and the step is unitary, `_krylov_enrich` otherwise. Logs the visit.
"""
function _recycled_or_fresh(rec, side::Symbol, i::Int, e, transport, v0, applyH, Q0,
                            tau::ComplexF64, alloc::BasisAllocation; round::Int, kw...)
    if rec === nothing || e === nothing || real(tau) != 0
        return _krylov_enrich(v0, applyH, Q0, side, tau, alloc; kw...)
    end
    t0 = time_ns()
    C = transport()
    tt = (time_ns() - t0) / 1e9
    Q, st = _krylov_enrich_recycled(v0, applyH, Q0, side, tau, alloc, C;
                                    workcap = get(kw, :workcap, 0),
                                    amp = get(kw, :amp, 1.0))
    push!(rec.log[side], (round = round, bond = i, cached = length(e.basis),
                          recycled = st.n_recycled, n_refresh = st.n_refresh, n_new = st.n_new,
                          added = st.added, t_transport = tt))
    return Q, st
end
