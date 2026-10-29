
# ── RECYCLED LOCAL KRYLOV DIRECTIONS ACROSS THE ROUNDS OF ONE STEP (plan §1.2) ────────────────
# An order-p step builds its frames in p rounds: the half-sweeps (round 0, frontier j = 0) and one
# enriching sweep per closure frontier j = 1..p-1. Each round runs a local Krylov stage at every
# bond it enriches. Without recycling, every round restarts that stage from its own seed, so a
# closure round re-derives a whole local Krylov sequence and appends whatever of it scores above
# the allowance (A1 L = 24, rp7q: frontier 1 appended 1000-2500 local Krylov directions per bond
# against 10-90 DD-QR CBE directions, 22k in all, and the frame outgrew the work ceiling).
#
# Here a round keeps, per bond, the Krylov vectors that carried weight above the allowance and its
# last vector (the active frontier). The next committed round transports them into its own gauge
# -- the frame leg by the map `Pm` its sweep already carries the frames with, the opposite leg by
# the overlap of the old and new expanded frame and tail -- and augments its stage's chain with
# them ([`_krylov_basis`](@ref)). Nothing survives the step: every step starts from the state.

"Recycled vectors kept per bond, heaviest first (plan §1.2 work caps: a bounded recycled set)."
const _RECYCLE_KEEP = 8

"""
    RecycledKrylov()

One step's recycled local Krylov directions, per side (`1` = left, `2` = right): the last
committed round `cur[s] = (entries, tail, gen)` -- `entries[i] = (vecs, frame)`, `tail` the state
whose tail the vectors' opposite leg lives in, `gen` the frame generation their frame leg is
written in -- and the round being built, `pend[s]`. A round's entries become usable only once its
frames are committed. The halves of a sweep touch only their own slot, so they may run
concurrently. `n_offered` / `n_used` count transported vectors and bonds that recycled.
"""
mutable struct RecycledKrylov
    cur::Vector{Any}
    pend::Vector{Any}
    n_offered::Vector{Int}
    n_used::Vector{Int}
end
RecycledKrylov() = RecycledKrylov(Any[nothing, nothing], Any[nothing, nothing], [0, 0], [0, 0])

_rk_slot(side::Symbol) = side === :left ? 1 : 2

_rk_begin!(::Nothing, ::Symbol) = nothing
_rk_begin!(rk::RecycledKrylov, side::Symbol) = (rk.pend[_rk_slot(side)] = Dict{Int, Any}(); nothing)

"""
Keep bond `i`'s useful vectors of this round: those with weight above `keep` and the last one,
at most `_RECYCLE_KEEP`, heaviest first; `frame` is the expanded opposite frame their opposite leg
is written in (`V_ex` on the left, `U_ex` on the right).
"""
_rk_keep!(::Nothing, args...) = nothing
function _rk_keep!(rk::RecycledKrylov, side::Symbol, i::Int, st, frame, keep::Float64)
    p = rk.pend[_rk_slot(side)]
    (p === nothing || isempty(st.basis)) && return nothing
    n = length(st.basis)
    sel = [k for k in 2:(n - 1) if st.weights[k] > keep]   # the seed is this round's own state
    sort!(sel; by = k -> -st.weights[k])
    sel = vcat(n >= 2 ? [n] : Int[], sel)                  # the active frontier first
    length(sel) > _RECYCLE_KEEP && (sel = sel[1:_RECYCLE_KEEP])
    isempty(sel) || (p[i] = (vecs = st.basis[sel], frame = frame))
    return nothing
end

"Commit the round being built when its frames are (`keep`), with its tail state and generation."
_rk_commit!(::Nothing, args...) = nothing
function _rk_commit!(rk::RecycledKrylov, side::Symbol, tail, gen::Int, keep::Bool)
    s = _rk_slot(side)
    keep && (rk.cur[s] = (entries = rk.pend[s], tail = tail, gen = gen))
    rk.pend[s] = nothing
    return nothing
end

"The committed round of `side` whose frame legs are generation `gen`, or `nothing`."
_rk_round(::Nothing, ::Symbol, ::Int) = nothing
function _rk_round(rk::RecycledKrylov, side::Symbol, gen::Int)
    r = rk.cur[_rk_slot(side)]
    return r === nothing || r.gen != gen || isempty(r.entries) ? nothing : r
end

# The bra's links go into a private namespace, as `overlap` does, so the transfer tensors never
# carry two identical indices.
_rk_tag(k::Int) = "RKB,$k"
function _rk_bra(t, k::Int, L::Int)
    k > 1 && (t = setitag(t, 1, _rk_tag(k)))
    k < L && (t = setitag(t, 3, _rk_tag(k + 1)))
    return to_concrete(t)
end

"""
    _rk_overlaps(new, old, side) -> Vector

Transfer tensors of `<new | old>` over the tails the transport needs. `side = :left`: `E[k]` =
sites `k..L`, legs `(bra link_k, ket link_k)` (both tails right-canonical); `:right`: `E[k]` = sites
`1..k`, legs `(bra link_{k+1}, ket link_{k+1})` (both heads left-canonical). Bra links `RKB,k`.
"""
function _rk_overlaps(new::SymMPS, old::SymMPS, side::Symbol)
    L = length(new)
    E = Vector{Any}(nothing, L)
    if side === :left
        for k in L:-1:3
            b = _rk_bra(new[k], k, L)
            T = k == L ? old[k] : to_concrete!(contract(old[k], (3,), E[k + 1], (2,)))
            E[k] = to_concrete!(contract(b', (2, 3), T, (2, 3)))
        end
    else
        for k in 1:(L - 2)
            b = _rk_bra(new[k], k, L)
            E[k] = k == 1 ? to_concrete!(contract(b', (1, 2), old[1], (1, 2))) :
                   to_concrete!(contract(to_concrete!(contract(E[k - 1], (1,), b', (1,))),
                                         (1, 2), old[k], (1, 2)))
        end
    end
    return E
end

"""
    _rk_transport(e, side, Pm, frame_new, E, i) -> Vector

Bond `i`'s recycled vectors `e.vecs` in this round's local space. The frame leg goes through the
sweep's frame map `Pm` (`(new, old)` on the left, `(old, new)` on the right; `nothing` at the chain's
end), exactly as the frames themselves; the opposite leg through `<frame_new + tail | e.frame +
old tail>` from the overlaps `E`. That map is a projection: what the new opposite space lacks is
dropped, and the solver re-orthonormalises.
"""
function _rk_transport(e, side::Symbol, Pm, frame_new, E, i::Int)
    if side === :left                         # opposite frame (b, site_{i+1}, link_{i+1})
        Ek = i + 2 <= length(E) ? E[i + 2] : nothing
        X = Ek === nothing ? e.frame : to_concrete!(contract(e.frame, (3,), Ek, (2,)))
        Fn = Ek === nothing ? frame_new : to_concrete(setitag(frame_new, 3, _rk_tag(i + 2)))
        M = to_concrete!(contract(Fn', (2, 3), X, (2, 3)))                  # (b_new*, b_old)
        return map(e.vecs) do v
            w = Pm === nothing ? v : to_concrete!(contract(Pm, (2,), v, (1,)))
            to_concrete!(contract(w, (3,), M, (2,)))
        end
    end
    Ek = i - 1 >= 1 ? E[i - 1] : nothing      # opposite frame (link_{i-1}, site_i, b)
    X = Ek === nothing ? e.frame : to_concrete!(contract(Ek, (2,), e.frame, (1,)))
    Fn = Ek === nothing ? frame_new : to_concrete(setitag(frame_new, 1, _rk_tag(i)))
    M = to_concrete!(contract(Fn', (1, 2), X, (1, 2)))                      # (b_new*, b_old)
    return map(e.vecs) do v
        w = Pm === nothing ? v : to_concrete!(contract(v, (3,), Pm, (1,)))
        to_concrete!(contract(M, (2,), w, (1,)))
    end
end
