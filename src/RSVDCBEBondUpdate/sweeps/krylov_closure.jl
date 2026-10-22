# ── GENERAL-ORDER KRYLOV CLOSURE (krylov_closure_cbe_bug_realtime_L16_plan.md) ──────────────
#
# THE ORDER CONDITION. With `A = tau*H`, `P = P(W, Z)` the Galerkin projector of the K/L frames
# and `B = P A P` the root generator, ONE root exponential has local error O(|tau|^(p+1)) once
#
#     (P A P)^j psi = A^j psi   (j = 1..p)     <=>     (I - P) A B^j psi = 0   (j = 0..p-1)
#
# (induction: given B^j psi = A^j psi, B^(j+1) psi = P A (A^j psi) equals A^(j+1) psi iff the j-th
# frontier action leaves nothing outside P). The ordinary half-sweeps of `cbe_bug_step!` close
# j = 0 -- CBE and Tests C/D on psi itself. This file closes j = 1..p-1.
#
# ⛔ REPEATED LOCAL H1 ACTIONS DO NOT CERTIFY ORDER. Tests C/D close the ONE-SITE Krylov space of
# each bond, a property of `H1` with the far environment frozen; the order condition is about the
# GLOBAL operator. So a frontier is judged by its outside-space action `(I - P) A F_j`, never by a
# root Krylov residual: a root space invariant under `B` (projected breakdown) says nothing about
# `A` (UT4).
#
# THE FRONTIER. `F_j = (tau^j / j!) B^j S0`, built at the root by zero-site applications and
# embedded as an MPS in the CURRENT frames (`_frontier_mps`). The time weight lives in the vector:
# variation of constants gives the local error as `sum_j (tau/(j+1)) (I - P) A F_j + ...`, so
# measuring `F_j` with the step's own `|tau|/||psi||` scale reads off each frontier's contribution
# (the `1/(j+1)` is slack).
#
# THE SWEEP. A frontier is swept with the SAME kernels as psi -- `cbe_expand` (RSVD Tests A/B) and
# `_krylov_enrich` (Tests C/D) -- with the current frame, embedded into the already-grown basis of
# the bond before it, as the old subspace kept VERBATIM (`U_ex = [U0 | new]`). Before any
# selection the bond's exact outside-space action `||(I - U0 U0') H Theta_F||` is read off the
# two-site action CBE builds anyway; a bond within its share of `tol_basis` is skipped. For a
# nearest-neighbour H these per-bond numbers are exactly the telescoping pieces of the global
# complement action (`estimate_global_complement_action`).
#
# ⚠ THE ORDINARY HALF-SWEEP IS NOT RE-ROUTED THROUGH THIS CODE: `order = 1` keeps `cbe_bug_step!`
# bit for bit (UT9), and the closure starts from the frames it built.

"A frontier below this fraction of the state norm is the zero vector (e.g. `H psi = 0`)."
const _FRONTIER_FLOOR = 1e-14

"Internal bound on closure passes. Reaching it with the frames still growing is `:closure`."
_closure_maxpass(order::Int) = 2 * order + 2

# Each enriching frontier sweep re-tags its frames with a new generation, so a frame and the
# frame it grew from never share a tag inside one overlap tensor.
_tagWg(i::Int, g::Int) = g == 0 ? _tagW(i) : "cbeW$g,$i"
_tagZg(j::Int, g::Int) = g == 0 ? _tagZ(j) : "cbeZ$g,$j"

"""
    KLFrames(W, Z, c, lenv, renv, gen)

The K/L frames of one step and the root environments they imply: `W[1..c]` left frames,
`Z[c..L-1]` right frames (`Z[j]` absorbs site `j+1`), `lenv` the environment at link `c` through
`W[1..c-1]`, `renv` the one at link `c+2` through `Z[c+1..L-1]`. `gen` counts the enrichments,
so a [`KrylovFrontier`](@ref) built in older frames is refused rather than silently misread.

`lcache` / `rcache` hold the environments every frontier sweep of ONE generation shares
([`_side_cache!`](@ref)); each half-sweep writes only its own, so the parallel halves never
write the same field.
"""
mutable struct KLFrames
    W::Vector{Any}
    Z::Vector{Any}
    c::Int
    lenv::Any
    renv::Any
    gen::Int
    lcache::Any
    rcache::Any
    # The generators' energy shift (`shift_h`): the root `B - eshift` and the frontier sweeps'
    # one-site Krylov operators. `0.0` is the unshifted step.
    eshift::Float64
end

KLFrames(W, Z, c, lenv, renv, gen) = KLFrames(W, Z, c, lenv, renv, gen, nothing, nothing, 0.0)

"""
Threads below which the closure keeps its frontier sweeps in sequence. Each sweep already runs
its two halves concurrently under `parallel = true`, so two concurrent sweeps need four threads
to gain anything; below that the 2-thread production path stays exactly what it was.
"""
const _FRONTIER_THREADS = 4

_frontier_parallel(parallel::Bool) = parallel && Threads.nthreads() >= _FRONTIER_THREADS

"""
Whether a measuring sweep runs its bonds as concurrent tasks ([`_bond_measurements`](@ref)) when
[`_frontier_parallel`](@ref) holds. A switch, not a tolerance: it changes no result at
`fold = false` and exists so a timing job can A/B it in one build (`bug_profile_2d.jl`,
`BP_BOND_TASKS`). Off by default: at 4 threads, D = 512 it gained nothing (job 17287169).
"""
const _BOND_TASKS = Ref(false)

"""
    _prime_caches!(fr, psi, mpo, core)

Build this generation's shared environments ([`_side_cache!`](@ref)) before frontier sweeps run
concurrently, so the concurrent sweeps only READ them. `core` is any root core: the cached
parts do not depend on it. The two sides are separate fields and are built concurrently.
"""
function _prime_caches!(fr::KLFrames, psi::SymMPS, mpo, core)
    Fm = _frontier_mps(psi, fr, core)
    t = Threads.@spawn _side_cache!(fr, Fm, mpo, :right)
    _side_cache!(fr, Fm, mpo, :left)
    fetch(t)
    return fr
end

"""
    _side_cache!(fr, Fm, mpo, side) -> (chain, stack)

The environments of one half-sweep that do not depend on the frontier, built once per frame
generation. ⛔ EVERY FRONTIER OF A GENERATION IS `(W ⊗ Z) core` WITH ONLY THE CORE DIFFERENT, so
re-gauging `Fm` to site 1 (left half) changes sites `1..c` only: the right stack on links
`c+1..L+1` and the frame chain `<W[1..i-1]| H |W[1..i-1]>` (the non-enriching sweep's `lch`)
are the same for every frontier. Rebuilding them per sweep was ~1/3 of a measuring sweep
(bug4 10x4 D=256 profile, job 17283685). `chain[i]` is the left channels at bond `i` (`i = 1..c`),
`stack` the right-stack entries on links `>= c+1`; the right half mirrors both (`chain[j]` the
right channels at bond `j = c..L-1`, `stack` the left-stack entries on links `<= c`).
"""
function _side_cache!(fr::KLFrames, Fm::SymMPS, mpo, side::Symbol)
    L, c = length(Fm), fr.c
    cur = side === :left ? fr.lcache : fr.rcache
    valid = cur !== nothing && cur.gen == fr.gen
    chain = valid ? cur.chain : nothing          # an enriching sweep seeds the chain only
    E = valid ? cur.stack : nothing
    (chain !== nothing && E !== nothing) && return (chain, E)
    if chain === nothing
        chain = Vector{Any}(undef, L - 1)
        ch = boundary_channels(mpo)
        if side === :left
            for i in 1:c
                chain[i] = ch
                i < c && (ch = push_left_channels(ch, mpo, fr.W[i], i))
            end
        else
            for j in (L - 1):-1:c
                chain[j] = ch
                j > c && (ch = push_right_channels(ch, mpo, fr.Z[j], j + 1))
            end
        end
    end
    E = Any[missing for _ in 1:(L + 1)]
    env = nothing
    if side === :left
        E[L + 1] = nothing
        for k in L:-1:(c + 1)
            env = _mpo_right_step(env, Fm[k], mpo[k])
            E[k] = env
        end
        fr.lcache = (gen = fr.gen, chain = chain, stack = E)
    else
        E[1] = nothing
        for i in 1:(c - 1)
            env = _mpo_left_step(env, Fm[i], mpo[i])
            E[i + 1] = env
        end
        fr.rcache = (gen = fr.gen, chain = chain, stack = E)
    end
    return (chain, E)
end

"""
    KrylovFrontier(core, degree, gen)

A projected Krylov frontier `F_j = (tau^j / j!) B^j S0`: its coefficients `core` in the root
basis of frames generation `gen` (a block-sparse `(b_L, b_R)` tensor, so its U(1) sectors travel
with it) and its degree `j`. The cached environments are the frames' own.
"""
struct KrylovFrontier
    core::Any
    degree::Int
    gen::Int
end

"""
    frames_from_state(psi, mpo, c) -> KLFrames

The state's OWN Schmidt frames at root bond `c` -- the subspace every step starts from, before
any expansion. Used to start a closure from scratch (`include_zero = true`) and by the tests.
"""
function frames_from_state(psi::SymMPS, mpo, c::Int)
    L = length(psi)
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))
    p = copy(psi)
    canonical!(p, c)
    W = Vector{Any}(undef, c)
    for i in 1:(c - 1)
        W[i] = p[i]
    end
    W[c] = to_concrete(svd(p[c], (1, 2); cutoff = 0.0).U)
    q = copy(psi)
    canonical!(q, c + 1)
    Z = Vector{Any}(undef, L - 1)
    Z[c] = to_concrete(svd(q[c + 1], (1,); cutoff = 0.0).Vd)
    for j in (c + 1):(L - 1)
        Z[j] = q[j + 1]
    end
    # Frame tags on both ends of every frame link, as the half-sweeps write them.
    for i in 1:c
        W[i] = to_concrete(setitag(W[i], 3, _tagW(i)))
        i < c && (W[i + 1] = to_concrete(setitag(W[i + 1], 1, _tagW(i))))
    end
    for j in c:(L - 1)
        Z[j] = to_concrete(setitag(Z[j], 1, _tagZ(j)))
        j > c && (Z[j - 1] = to_concrete(setitag(Z[j - 1], 3, _tagZ(j))))
    end
    lenv = boundary_channels(mpo)
    for i in 1:(c - 1)
        lenv = push_left_channels(lenv, mpo, W[i], i)
    end
    renv = boundary_channels(mpo)
    for j in (L - 1):-1:(c + 1)
        renv = push_right_channels(renv, mpo, Z[j], j + 1)
    end
    return KLFrames(W, Z, c, lenv, renv, 0)
end

"`<W[1..c] | psi>` on the left of the root: `(b_L, link_{c+1})`. A full contraction, any gauge."
function _overlap_left(psi::SymMPS, W, c::Int)
    AL = nothing
    for i in 1:c
        T = AL === nothing ? psi[i] : to_concrete!(contract(AL, (2,), psi[i], (1,)))
        AL = to_concrete!(contract(W[i]', (1, 2), T, (1, 2)))         # (b_L, link_{i+1})
    end
    return AL
end

"`<Z[c..L-1] | psi>` on the right of the root: `(link_{c+1}, b_R)`. A full contraction, any gauge."
function _overlap_right(psi::SymMPS, Z, c::Int)
    AR = nothing
    for j in (length(psi) - 1):-1:c
        B = AR === nothing ? psi[j + 1] : to_concrete!(contract(psi[j + 1], (3,), AR, (1,)))
        AR = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))         # (link_j, b_j)
    end
    return AR
end

"""
    _projection_miss(psi, W, Z, c) -> Float64

`||(I - P) psi||` for `P` the projector onto the root space of `W[1..c]`, `Z[c..L-1]`, measured
DIRECTLY. `I - P = (I - P_L) ⊗ 1 + P_L ⊗ (I - P_R)`, and each side telescopes into orthogonal
one-site pieces `(1 - W_i W_i') T_i` / `B_j (1 - Z_j' Z_j)` that are formed explicitly, so the
roundoff floor is ~eps. The difference of squares `||psi||^2 - ||P psi||^2` cancels to
~sqrt(eps), the size of a tight `tol_basis`. The rest of `psi` beyond each piece enters through
a square-root factor of its Gram matrix (an SVD's `U S` / `S V'`), so any gauge works.
"""
function _projection_miss(psi::SymMPS, W, Z, c::Int)
    L = length(psi)
    R = Vector{Any}(undef, L)                    # R[k] R[k]' = Gram of psi[k..L] on link_k
    for k in L:-1:2
        T = k == L ? psi[L] : to_concrete!(contract(psi[k], (3,), R[k + 1], (1,)))
        res = svd(T, (1,); cutoff = 0.0)
        R[k] = to_concrete(res.U * res.S)
    end
    miss2 = 0.0
    AL = nothing
    for i in 1:c
        T = AL === nothing ? psi[i] : to_concrete!(contract(AL, (2,), psi[i], (1,)))
        miss2 += norm(to_concrete!(contract(perp_component(W[i], T), (3,), R[i + 1], (1,))))^2
        AL = to_concrete!(contract(W[i]', (1, 2), T, (1, 2)))         # (b_i, link_{i+1})
    end
    G = Vector{Any}(undef, L - 1)                # G[j]' G[j] = Gram of P_L psi on link_{j+1}
    res = svd(AL, (1,); cutoff = 0.0)
    G[c] = to_concrete(res.S * res.Vd)
    for j in (c + 1):(L - 1)
        res = svd(to_concrete!(contract(G[j - 1], (2,), psi[j], (1,))), (1, 2); cutoff = 0.0)
        G[j] = to_concrete(res.S * res.Vd)
    end
    AR = nothing
    for j in (L - 1):-1:c
        B = AR === nothing ? psi[j + 1] : to_concrete!(contract(psi[j + 1], (3,), AR, (1,)))
        miss2 += norm(to_concrete!(contract(G[j], (2,), perp_component_right(Z[j], B), (1,))))^2
        AR = to_concrete!(contract(B, (2, 3), Z[j]', (2, 3)))         # (link_{j+1}, b_j)
    end
    return sqrt(miss2)
end

"`S = (W ⊗ Z)' psi`, the root coefficients of `P psi` (plan: `project_root`)."
_project_root(psi::SymMPS, fr::KLFrames) =
    to_concrete!(contract(_overlap_left(psi, fr.W, fr.c), (2,),
                          _overlap_right(psi, fr.Z, fr.c), (1,)))

"`B = P H P` at the root, from the frames' cached environments (plan: `root_generator`)."
_root_generator(mpo, fr::KLFrames) =
    shift_h(zero_site_h(mpo, fr.c, fr.lenv, fr.renv, fr.W[fr.c], fr.Z[fr.c]), fr.eshift)

_scaled(a::Number, x::Array{<:Number}) = a .* x
_scaled(a::Number, x) = to_concrete(a * x)

"""
    _projected_frontiers(applyB, S0, tau, m) -> [F_1, ..., F_m]

`F_j = (tau^j / j!) B^j S0` by the recursion `F_j = (tau/j) B F_{j-1}`. Generic in the vector
type (root cores here, plain vectors in the dense unit tests).
"""
function _projected_frontiers(applyB, S0, tau::Number, m::Int)
    Fs = Any[]
    F = S0
    for j in 1:m
        F = _scaled(tau / j, applyB(F))
        push!(Fs, F)
    end
    return Fs
end

"""
    _frontier_mps(psi, fr, core) -> SymMPS

The Galerkin vector `(W ⊗ Z) core` as an MPS on `psi`'s link tags, centre at the root -- the
same assembly `cbe_bug_step!` uses for the evolved core.
"""
function _frontier_mps(psi::SymMPS, fr::KLFrames, core)
    L, c = length(psi), fr.c
    T = Vector{Any}(undef, L)
    for i in 1:(c - 1)
        T[i] = _relink(fr.W[i], psi[i].inds[1].itags, psi[i].inds[3].itags)
    end
    T[c] = _relink(to_concrete(fr.W[c] * core), psi[c].inds[1].itags, psi[c].inds[3].itags)
    for j in c:(L - 1)
        T[j + 1] = _relink(fr.Z[j], psi[j + 1].inds[1].itags, psi[j + 1].inds[3].itags)
    end
    return SymMPS(T, c)
end

"""
    _frame_onto_left(U0, K, right) -> BondFrame

`_frame_from` with the LEFT isometry prescribed: `U0` is the current frame (embedded in the grown
basis of the bond before), which contains the range of the carry `K`, so `U0 S0 V0` is the exact
two-site block. `old_rank` is `U0`'s width -- the side being grown.
"""
function _frame_onto_left(U0, K, right)
    res_r = svd(right, (1,), "bV,L", "bV,R"; cutoff = 0.0)
    V0 = to_concrete(res_r.Vd)
    S0 = to_concrete(to_concrete!(contract(U0', (1, 2), K, (1, 2))) *
                     to_concrete(res_r.U * res_r.S))
    return BondFrame(U0, S0, V0, K.inds[1], K.inds[2], K.inds[3], right.inds[2],
                     right.inds[3], leg_dim(U0, 3), copy(K.spaces[3]))
end

"Row mirror of [`_frame_onto_left`](@ref): the right isometry `V0` is prescribed."
function _frame_onto_right(left, B, V0)
    res_l = svd(left, (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    U0 = to_concrete(res_l.U)
    S0 = to_concrete(to_concrete(res_l.S * res_l.Vd) *
                     to_concrete!(contract(B, (2, 3), V0', (2, 3))))
    return BondFrame(U0, S0, V0, left.inds[1], left.inds[2], left.inds[3], B.inds[2],
                     B.inds[3], leg_dim(V0, 1), copy(left.spaces[3]))
end

"`||(I - U0 U0') X||` for a two-site `X = (link_l, site_l, site_r, link_r)`."
_left_leak(U0, X) = norm(perp_component(U0, X))

"`||X (I - V0' V0)||` for a two-site `X`, the right frame's miss."
function _right_leak(V0, X)
    ov = contract(X, (3, 4), V0', (2, 3))                               # (link_l, site_l, bond)
    return norm(to_concrete(X - to_concrete!(contract(ov, (3,), V0, (1,)))))
end

"""
    _sampled_leak(f, skl, skr, side, n, rng) -> Float64

The randomised estimate of the same miss: `n` sector-graded Gaussian columns on the FULL opposite
local space (no projector in the probe), through the production sketch closures `(skl, skr)` --
shared or folded -- and Test B's per-sector unbiased normalisation (`_sample_residual`). An
estimate, not a bound.
"""
function _sampled_leak(f::BondFrame, skl, skr, side::Symbol, n::Int, rng::AbstractRNG)
    if side === :left
        draw = w -> sector_graded_sketch(f.V0, :right, w; comp_ratio = nothing, rng = rng)
        return first(_sample_residual(nothing, Y -> perp_component(f.U0, Y), skl, draw, n,
                                      1, :left))
    end
    draw = w -> sector_graded_sketch(f.U0, :left, w; comp_ratio = nothing, rng = rng)
    return first(_sample_residual(nothing, Y -> perp_component_right(f.V0, Y), skr, draw, n,
                                  3, :right))
end

"""
    _bond_action(f, mpo, i, lch, rch, U0, V0, side, fold, rng) -> (skl, skr, leak, nh2)

One frontier bond's sketch closures and outside-space action on `side`.

`fold = false`: the exact two-site action `X = H Theta_F` is formed once, the leak is exact and the
closures share `X` (`nh2 = 1`). `fold = true` (`cbe_bug_step!(...; fold_omega = true)`): `X` is
NEVER formed -- the probe is folded into the environment contractions (`_fold_sketch_closures`)
and the leak is Test B's holdout estimate at a quarter of the frame width, the same convention
`_certified_range` uses. ⛔ This is the large-chi path: `X` lives on the frame width ~2chi, so the
exact action costs `O((2chi)^3 d^2 D_w)` per bond per frontier sweep, the dominant cost of an
order-p step at large chi, while the folded probe costs `O((2chi)^2 d n D_w)`.
"""
function _bond_action(f::BondFrame, mpo, i::Int, lch, rch, U0, V0, side::Symbol, fold::Bool,
                      rng::AbstractRNG)
    if fold
        skl, skr = _fold_sketch_closures(f, mpo, i, lch, rch)
        n = max(8, cld(leg_dim(side === :left ? U0 : V0, side === :left ? 3 : 1), 4))
        return (skl, skr, _sampled_leak(f, skl, skr, side, n, rng), 0)
    end
    X = apply_h_two_site(frame_theta(f), mpo, i, lch, rch)
    skl, skr = _sketch_closures(f, () -> X)
    return (skl, skr, side === :left ? _left_leak(U0, X) : _right_leak(V0, X), 1)
end

"""
    _bond_measurements(Fm, mpo, fr, chain, stack, side; fold, nsample, rng) -> Vector

A MEASURING half-sweep's bonds, CONCURRENTLY (parallel plan (b)). With the frames fixed, the only
bond-to-bond dependency is the carry (`C = W[i]' K` on the left, `D = B Z[j]'` on the right): a
cheap sequential chain. Each bond's frame, two-site action and leak (~1/3 of a measuring sweep,
`_bond_action`) then depend on nothing else, so they run one task per bond, each with its own RNG
drawn in bond order from `rng`. Entry `i` (a bond of `side`) is `(K, leak, nh2, sleak)`, `K` the
carry the sequential loop would have built there; `sleak = 0` unless `nsample > 0`.
"""
function _bond_measurements(Fm::SymMPS, mpo, fr::KLFrames, chain, stack, side::Symbol;
                            fold::Bool, nsample::Int, rng::AbstractRNG)
    L, c = length(Fm), fr.c
    bonds = side === :left ? (1:c) : ((L - 1):-1:c)
    carry = Vector{Any}(undef, L - 1)
    X = nothing
    for i in bonds
        if side === :left
            carry[i] = X === nothing ? Fm[i] : to_concrete!(contract(X, (2,), Fm[i], (1,)))
            X = to_concrete!(contract(fr.W[i]', (1, 2), carry[i], (1, 2)))
        else
            carry[i] = X === nothing ? Fm[i + 1] :
                       to_concrete!(contract(Fm[i + 1], (3,), X, (1,)))
            X = to_concrete!(contract(carry[i], (2, 3), fr.Z[i]', (2, 3)))
        end
    end
    rngs = Vector{Any}(undef, L - 1)
    for i in bonds
        rngs[i] = MersenneTwister(rand(rng, UInt64))
    end
    function one(i)
        f, a = if side === :left
            f = _frame_onto_left(fr.W[i], carry[i], Fm[i + 1])
            (f, _bond_action(f, mpo, i, chain[i], right_channels(stack, i + 2), fr.W[i],
                             nothing, :left, fold, rngs[i]))
        else
            f = _frame_onto_right(Fm[i], carry[i], fr.Z[i])
            (f, _bond_action(f, mpo, i, left_channels(stack, i), chain[i], nothing, fr.Z[i],
                             :right, fold, rngs[i]))
        end
        skl, skr, leak, nh2 = a
        sl = nsample > 0 ? _sampled_leak(f, skl, skr, side, nsample, rngs[i]) : 0.0
        return (K = carry[i], leak = leak, nh2 = nh2, sleak = sl)
    end
    tasks = Dict(i => Threads.@spawn(one(i)) for i in bonds)
    out = Vector{Any}(undef, L - 1)
    for i in bonds
        out[i] = fetch(tasks[i])
    end
    return out
end

"How many charge sectors `new` carries on `leg` that `old` did not (plan §3.5)."
function _new_sectors(old, new, leg::Int)
    have = Set(first(s) for s in old.spaces[leg])
    return count(s -> !(first(s) in have), new.spaces[leg])
end

"""
    _frontier_expand(f, skl, skr, mpo, i, lch, rch, side, ckw, ledger, rng) -> CBEExpansion

A frontier bond's CBE under the step's `cbe_policy` (in `ckw`, `cbe_bug_step!`'s; plan T4.1): the
DD-QR policies grow `side`'s frame by [`dd_cbe_expand`](@ref), exactly as the half-sweeps do, so
the Krylov stage after it starts from those directions; `:svd_select` runs `cbe_expand` on the
frontier's sketch closures.
"""
function _frontier_expand(f::BondFrame, skl, skr, mpo, i::Int, lch, rch, side::Symbol,
                          ckw::NamedTuple, ledger::AdaptiveLedger, rng::AbstractRNG)
    policy = get(ckw, :cbe_policy, :svd_select)
    policy === :svd_select &&
        return cbe_expand(f, skl, skr;
                          Base.structdiff(ckw, NamedTuple{(:cbe_policy, :krylov_reuse)})...,
                          ledger = ledger, rng = rng)
    return dd_cbe_expand(f, mpo, i, lch, rch, side; policy = policy, ledger = ledger, rng = rng,
                         alloc = get(ckw, :alloc, nothing), workcap = get(ckw, :workcap, 0),
                         growth = get(ckw, :growth, ADAPTIVE_GROWTH),
                         thin_search = get(ckw, :thin_search, true),
                         exact = get(ckw, :exact, false), version = i)
end

# Tag every bond entry a frontier sweep logged, after `_log_bond!` wrote it.
_tag_last!(lg::AdaptiveLedger, extra::NamedTuple) =
    (lg.bond_log[end] = merge(lg.bond_log[end], extra); lg)

"""
    _frontier_left(Fm, mpo, fr, tau, alloc, ckw; enrich, gen, workcap, amp, rng, nsample)

The left half-sweep of one frontier MPS `Fm` (centre at the root, built from `fr`): bond by bond
the current frame `fr.W[i]` is embedded into the grown basis of bond `i-1` (`U0 = Pm W[i]`, an
overlap, not an assumption), the exact outside-space action of the frontier's two-site block is
measured, and -- if it exceeds the bond's share and `enrich` -- the step's CBE
([`_frontier_expand`](@ref)) and `_krylov_enrich` append validated directions to `U0`, which is
kept verbatim. Returns the new frames (re-tagged to generation `gen`), the root environment, the
bond ledger and the per-bond misses.

A MEASURING sweep (`enrich = false`) grows nothing, so it runs in the frames themselves: `U0 =
fr.W[i]` with no overlap `Pm` (identity there) and the generation's cached frame chain for `lch`
([`_side_cache!`](@ref)). `chain` returns the `lch` an enriching sweep built from the new frames,
which is the next generation's chain. With `bonds` its bonds are measured concurrently
([`_bond_measurements`](@ref)) and the loop only logs them.
"""
function _frontier_left(Fm::SymMPS, mpo, fr::KLFrames, tau::ComplexF64,
                        alloc::BasisAllocation, ckw::NamedTuple; enrich::Bool, gen::Int,
                        workcap::Int, amp::Float64, rng::AbstractRNG, nsample::Int = 0,
                        fold::Bool = false, bonds::Bool = false)
    c, L = fr.c, length(Fm)
    canonical!(Fm, 1)
    chain, zst = _side_cache!(fr, Fm, mpo, :left)
    E = copy(zst)                                # links >= c+1: shared by every frontier
    env = E[c + 1]
    for k in c:-1:3                              # links 3..c: this frontier's own gauge
        env = _mpo_right_step(env, Fm[k], mpo[k])
        E[k] = env
    end
    rstack = MPORightEnvStack(E)
    pre = bonds && !enrich ? _bond_measurements(Fm, mpo, fr, chain, rstack, :left; fold = fold,
                                                nsample = nsample, rng = rng) : nothing
    lch = boundary_channels(mpo)
    lenv = lch
    lchs = Vector{Any}(undef, L - 1)
    C, Pm = nothing, nothing
    Wn = Vector{Any}(undef, c)
    lg = AdaptiveLedger()
    leak, sleak = zeros(c), zeros(c)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for i in 1:c
        enrich || (lch = chain[i])
        U0 = _check_frame(Pm === nothing ? fr.W[i] : to_concrete!(contract(Pm, (2,), fr.W[i], (1,))),
                          :left, "frontier U0 = Pm W[$i] (gen $(fr.gen))")
        if pre === nothing
            K = C === nothing ? Fm[i] : to_concrete!(contract(C, (2,), Fm[i], (1,)))
            f = _frame_onto_left(U0, K, Fm[i + 1])
            rch = right_channels(rstack, i + 2)
            skl, skr, leak[i], k2 = _bond_action(f, mpo, i, lch, rch, U0, nothing, :left, fold,
                                                 rng)
            nsample > 0 && (sleak[i] = _sampled_leak(f, skl, skr, :left, nsample, rng))
        else
            K, leak[i], k2, sleak[i] = pre[i]
        end
        nh2 += k2
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[i]) <= alloc.per_bond
        Wi = if enrich && !quiet
            ex = _frontier_expand(f, skl, skr, mpo, i, lch, rch, :left, ckw, bl, rng)
            M = to_concrete!(contract(Fm[i + 1], (2, 3), ex.V_ex', (2, 3)))
            Kex = to_concrete!(contract(K, (3,), M, (1,)))
            H1 = shift_h(one_site_h(mpo, i, lch, push_right_channels(rch, mpo, ex.V_ex, i + 1)),
                         fr.eshift)
            reuse = get(ckw, :krylov_reuse, true)
            Q, st = _krylov_enrich(Kex, x -> (nh1[] += 1; apply_one_site(H1, x)),
                                   reuse ? ex.U_ex : U0, :left, tau, alloc; workcap = workcap,
                                   amp = amp, state_seed = false)
            _record_krylov!(bl, st, alloc)
            reuse ? Q : _merge_frame(Q, ex.U_ex, :left)
        else
            nskip += 1
            U0
        end
        Wn[i] = enrich ? to_concrete(setitag(Wi, 3, _tagWg(i, gen))) : U0
        _check_frame(Wn[i], :left, "frontier Wn[$i] (gen $gen)")
        nadd = leg_dim(Wn[i], 3) - leg_dim(U0, 3)
        nsec = _new_sectors(U0, Wn[i], 3)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, i, :left, leg_dim(Wn[i], 3))
        _tag_last!(lg, (leak = indicator(alloc, leak[i]), skipped = quiet || !enrich,
                        added = nadd, new_sectors = nsec, width_before = leg_dim(U0, 3)))
        enrich && (Pm = to_concrete!(contract(Wn[i]', (1, 2), U0, (1, 2))))  # (new_i, old_i)
        pre === nothing && (C = to_concrete!(contract(Wn[i]', (1, 2), K, (1, 2))))
        lchs[i] = lch
        i == c && (lenv = lch)
        enrich && i < c && (lch = push_left_channels(lch, mpo, Wn[i], i))
    end
    return (frames = Wn, env = lenv, chain = lchs, ledger = lg, leak = leak, sleak = sleak,
            added = added, new_sectors = newsec, skipped = nskip, n_h2 = nh2, n_h1 = nh1[])
end

"Row mirror of [`_frontier_left`](@ref): grows `fr.Z[L-1..c]`."
function _frontier_right(Fm::SymMPS, mpo, fr::KLFrames, tau::ComplexF64,
                         alloc::BasisAllocation, ckw::NamedTuple; enrich::Bool, gen::Int,
                         workcap::Int, amp::Float64, rng::AbstractRNG, nsample::Int = 0,
                         fold::Bool = false, bonds::Bool = false)
    c, L = fr.c, length(Fm)
    canonical!(Fm, L)
    chain, wst = _side_cache!(fr, Fm, mpo, :right)
    E = copy(wst)                                # links <= c: shared by every frontier
    env = E[c]
    for i in c:(L - 2)                           # links c+1..L-1: this frontier's own gauge
        env = _mpo_left_step(env, Fm[i], mpo[i])
        E[i + 1] = env
    end
    lstack = MPOLeftEnvStack(E)
    pre = bonds && !enrich ? _bond_measurements(Fm, mpo, fr, chain, lstack, :right; fold = fold,
                                                nsample = nsample, rng = rng) : nothing
    rch = boundary_channels(mpo)
    renv = rch
    rchs = Vector{Any}(undef, L - 1)
    D, Pm = nothing, nothing
    Zn = Vector{Any}(undef, L - 1)
    lg = AdaptiveLedger()
    leak, sleak = zeros(L - 1), zeros(L - 1)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for j in (L - 1):-1:c
        enrich || (rch = chain[j])
        V0 = _check_frame(Pm === nothing ? fr.Z[j] : to_concrete!(contract(fr.Z[j], (3,), Pm, (1,))),
                          :right, "frontier V0 = Z[$j] Pm (gen $(fr.gen))")
        if pre === nothing
            B = D === nothing ? Fm[j + 1] : to_concrete!(contract(Fm[j + 1], (3,), D, (1,)))
            f = _frame_onto_right(Fm[j], B, V0)
            lch = left_channels(lstack, j)
            skl, skr, leak[j], k2 = _bond_action(f, mpo, j, lch, rch, nothing, V0, :right, fold,
                                                 rng)
            nsample > 0 && (sleak[j] = _sampled_leak(f, skl, skr, :right, nsample, rng))
        else
            B, leak[j], k2, sleak[j] = pre[j]
        end
        nh2 += k2
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[j]) <= alloc.per_bond
        Zj = if enrich && !quiet
            ex = _frontier_expand(f, skl, skr, mpo, j, lch, rch, :right, ckw, bl, rng)
            M = to_concrete!(contract(ex.U_ex', (1, 2), Fm[j], (1, 2)))
            Bex = to_concrete!(contract(M, (2,), B, (1,)))
            H1 = shift_h(one_site_h(mpo, j + 1, push_left_channels(lch, mpo, ex.U_ex, j), rch),
                         fr.eshift)
            reuse = get(ckw, :krylov_reuse, true)
            Q, st = _krylov_enrich(Bex, x -> (nh1[] += 1; apply_one_site(H1, x)),
                                   reuse ? ex.V_ex : V0, :right, tau, alloc; workcap = workcap,
                                   amp = amp, state_seed = false)
            _record_krylov!(bl, st, alloc)
            reuse ? Q : _merge_frame(Q, ex.V_ex, :right)
        else
            nskip += 1
            V0
        end
        Zn[j] = enrich ? to_concrete(setitag(Zj, 1, _tagZg(j, gen))) : V0
        _check_frame(Zn[j], :right, "frontier Zn[$j] (gen $gen)")
        nadd = leg_dim(Zn[j], 1) - leg_dim(V0, 1)
        nsec = _new_sectors(V0, Zn[j], 1)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, j, :right, leg_dim(Zn[j], 1))
        _tag_last!(lg, (leak = indicator(alloc, leak[j]), skipped = quiet || !enrich,
                        added = nadd, new_sectors = nsec, width_before = leg_dim(V0, 1)))
        enrich && (Pm = to_concrete!(contract(V0, (2, 3), Zn[j]', (2, 3))))  # (old_j, new_j)
        pre === nothing && (D = to_concrete!(contract(B, (2, 3), Zn[j]', (2, 3))))
        rchs[j] = rch
        j == c && (renv = rch)
        enrich && j > c && (rch = push_right_channels(rch, mpo, Zn[j], j + 1))
    end
    return (frames = Zn, env = renv, chain = rchs, ledger = lg, leak = leak, sleak = sleak,
            added = added,
            new_sectors = newsec, skipped = nskip, n_h2 = nh2, n_h1 = nh1[])
end

_frame_width(fr::KLFrames, L::Int) =
    max(maximum(leg_dim(fr.W[i], 3) for i in 1:fr.c),
        maximum(leg_dim(fr.Z[j], 1) for j in fr.c:(L - 1)))

"""
    _frontier_sweep!(fr, psi, mpo, F, tau, alloc, ckw; enrich, workcap, nrm, parallel, rng,
                     pass, nsample) -> NamedTuple

Test one frontier's outside-space action on both half-sweeps and, when `enrich`, grow `fr` by the
validated directions (plan: `enrich_KL_from_frontiers!`). The frames are replaced -- and their
generation advanced -- only if something was added, so frontiers built in them stay valid
otherwise. `leak` is the time-weighted outside-space action `|tau| sqrt(left^2 + right^2) /
||psi||` of the frontier BEFORE this sweep enriched; `met` is whether every bond ended within
its share.
"""
function _frontier_sweep!(fr::KLFrames, psi::SymMPS, mpo, F::KrylovFrontier, tau::ComplexF64,
                          alloc::BasisAllocation, ckw::NamedTuple; enrich::Bool = true,
                          workcap::Int = 0, nrm::Float64 = 1.0, parallel::Bool = false,
                          rng::AbstractRNG = MersenneTwister(0x5EED), pass::Int = 0,
                          nsample::Int = 0, fold::Bool = false)
    F.gen == fr.gen || throw(ArgumentError(
        "stale frontier: built in frames generation $(F.gen), the frames are at $(fr.gen) -- " *
        "the root operator and frontiers must be rebuilt whenever the frames change"))
    t0 = time_ns()
    L, c = length(psi), fr.c
    nF = norm(F.core)
    zero_leaks = (zeros(c), zeros(L - 1))
    if nF <= _FRONTIER_FLOOR * max(nrm, eps())
        lg = AdaptiveLedger()
        push!(lg.closure_log, (pass = pass, frontier = F.degree, enrich = enrich,
                               norm = nF / max(nrm, eps()), leak = 0.0, added = 0,
                               zero = true, t = 0.0))
        return (added = 0, leak = 0.0, leak_left = 0.0, leak_right = 0.0, met = true,
                capped = false, ledger = lg, leaks_left = zero_leaks[1], leaks_right = zero_leaks[2],
                sleaks_left = zero_leaks[1], sleaks_right = zero_leaks[2])
    end
    Fm = _frontier_mps(psi, fr, F.core)
    FmR = copy(Fm)
    gen = fr.gen + 1
    amp = nF / max(nrm, eps())
    rngR = parallel ? MersenneTwister(rand(rng, UInt64)) : rng
    kw(r) = (enrich = enrich, gen = gen, workcap = workcap, amp = amp, rng = r,
             nsample = nsample, fold = fold,
             bonds = !enrich && _BOND_TASKS[] && _frontier_parallel(parallel))
    rl, rr = if parallel
        tR = Threads.@spawn _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...)
        l = _frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...)
        (l, fetch(tR))
    else
        (_frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...),
         _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...))
    end
    lg = AdaptiveLedger()
    for half in (rl.ledger, rr.ledger)
        half.bond_log = Any[merge(e, (pass = pass, frontier = F.degree)) for e in half.bond_log]
        merge_ledger!(lg, half)
    end
    lL, lR = sqrt(sum(abs2, rl.leak)), sqrt(sum(abs2, rr.leak))
    leak = indicator(alloc, sqrt(lL^2 + lR^2))
    met = enrich ? lg.tolerance_met :
          all(indicator(alloc, x) <= alloc.per_bond for x in vcat(rl.leak, rr.leak))
    added = rl.added + rr.added
    wbefore = _frame_width(fr, L)
    if enrich && added > 0
        for i in 1:c
            fr.W[i] = rl.frames[i]
        end
        for j in c:(L - 1)
            fr.Z[j] = rr.frames[j]
        end
        fr.lenv, fr.renv, fr.gen = rl.env, rr.env, gen
        # the channels this sweep carried through the new frames ARE the new generation's chain
        fr.lcache = (gen = gen, chain = rl.chain, stack = nothing)
        fr.rcache = (gen = gen, chain = rr.chain, stack = nothing)
    end
    lg.n_closure_sweeps = 1
    lg.n_h2_leak = rl.n_h2 + rr.n_h2
    lg.n_h1_closure = rl.n_h1 + rr.n_h1
    push!(lg.closure_log,
          (pass = pass, frontier = F.degree, enrich = enrich, norm = amp, leak = leak,
           leak_left = indicator(alloc, lL), leak_right = indicator(alloc, lR), met = met,
           added = added, added_left = rl.added, added_right = rr.added,
           new_sectors = rl.new_sectors + rr.new_sectors,
           skipped = rl.skipped + rr.skipped, bonds = c + (L - c),
           width_before = wbefore, width_after = _frame_width(fr, L),
           n_h2 = lg.n_h2_leak, n_h1 = lg.n_h1_closure, n_probes = lg.n_probes,
           n_selected = lg.n_selected, n_krylov = lg.n_krylov, zero = false,
           capped = enrich && lg.workcap_reached, t = (time_ns() - t0) / 1e9))
    # `capped`: an enrichment of this sweep stopped at the work ceiling (`_closure_loop!`).
    return (added = added, leak = leak, leak_left = indicator(alloc, lL),
            leak_right = indicator(alloc, lR), met = met,
            capped = enrich && lg.workcap_reached, ledger = lg,
            leaks_left = rl.leak, leaks_right = rr.leak,
            sleaks_left = rl.sleak, sleaks_right = rr.sleak)
end

"""
    estimate_global_complement_action(psi, mpo, fr, core; nsample = 0, rng) -> NamedTuple

The outside-space action `(I - P) H v` of the Galerkin vector `v = (W ⊗ Z) core`, matrix-free,
from the per-bond two-site misses of a non-enriching frontier sweep (no time weight).

Writing `P^(i)` for the projector onto the left frame chain up to bond `i`, `I - P_L =
sum_i (P^(i-1) - P^(i))` is a sum of ORTHOGONAL pieces, and bond `i`'s miss
`||(I - U0 U0') H Theta||` is exactly the norm of piece `i` whenever every term of `H` that acts
left of the cut also stops at site `i+1` (nearest neighbour): the right environment then spans
everything the term leaves on the right. So, for a nearest-neighbour `H`,

    left  = ||(I - P_L ⊗ 1) H v||,     right = ||(1 ⊗ (I - P_R)) H v||   (exactly)
    max(left, right) <= ||(I - P) H v|| <= bound = sqrt(left^2 + right^2)

For longer-range terms the per-bond numbers miss what a term leaves outside the right
environment's span, i.e. they become an estimate. `nsample > 0` adds the randomised per-bond
estimates (`sampled_*`). Returns `left`, `right`, `bound`, `per_left[1..c]`, `per_right[c..L-1]`
and, if sampled, `sampled_left`, `sampled_right`, `per_sampled_left`, `per_sampled_right`.
"""
function estimate_global_complement_action(psi::SymMPS, mpo, fr::KLFrames, core;
                                           nsample::Int = 0,
                                           rng::AbstractRNG = MersenneTwister(0x5EED))
    L, c = length(psi), fr.c
    alloc = basis_allocation(AdaptiveBudget(1.0, 0.0, 1.0), L - 1)    # scale 1: raw norms
    r = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(core, 0, fr.gen), ComplexF64(0), alloc,
                         NamedTuple(); enrich = false, nrm = max(norm(core), eps()),
                         nsample = nsample, rng = rng)
    pl, pr = r.leaks_left, r.leaks_right[c:(L - 1)]
    sl, sr = r.sleaks_left, r.sleaks_right[c:(L - 1)]
    left, right = sqrt(sum(abs2, pl)), sqrt(sum(abs2, pr))
    return (left = left, right = right, bound = sqrt(left^2 + right^2),
            per_left = pl, per_right = pr,
            sampled_left = sqrt(sum(abs2, sl)), sampled_right = sqrt(sum(abs2, sr)),
            per_sampled_left = sl, per_sampled_right = sr)
end

"""
    _measure_frontiers(fr, psi, mpo, S0, tau, alloc, ckw, m; nrm, parallel, rng, pass = 0)
        -> NamedTuple

Measure -- never enrich -- the frontiers `j = 0..m-1` of `S0` on the frames `fr` as they stand:
what the frames already capture of the global Krylov sequence. On the half-sweep frames this is
the ordinary step's own closure, `rho_j = ||(I - P) A F_j|| / ||F_j||` (the bound
`sqrt(left^2 + right^2)`), with no closure sweep involved. `fr` is untouched. Returns the
time-weighted `leak[j+1]`, the scale-free `rho[j+1]`, `met` (every bond of every frontier
`j >= 1` within its share -- `j = 0` is the half-sweeps' own test), the merged `ledger` and
`n_root` zero-site applications; `met0` is the `j = 0` verdict.

The frontiers are independent measurements on one set of frames, so with `parallel` and at
least [`_FRONTIER_THREADS`](@ref) threads they run CONCURRENTLY (each with its own RNG stream
drawn from `rng`, after the shared environments are built once by [`_prime_caches!`](@ref));
otherwise in order, exactly as before. The results are merged in `j` order either way.
"""
function _measure_frontiers(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                            alloc::BasisAllocation, ckw::NamedTuple, m::Int;
                            nrm::Float64 = 1.0, parallel::Bool = false,
                            rng::AbstractRNG = MersenneTwister(0x5EED), pass::Int = 0,
                            workcap::Int = 0, fold::Bool = false)
    H0 = _root_generator(mpo, fr)
    nroot = Ref(0)
    F = m > 1 ? _projected_frontiers(x -> (nroot[] += 1; apply_zero_site(H0, x)), S0, tau,
                                     m - 1) : Any[]
    lg = AdaptiveLedger()
    leak, rho, met, met0 = zeros(m), zeros(m), true, true
    cores = Any[j == 0 ? S0 : F[j] for j in 0:(m - 1)]
    measure(j, r) = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(cores[j + 1], j, fr.gen), tau,
                                     alloc, ckw; enrich = false, workcap = workcap, nrm = nrm,
                                     parallel = parallel, rng = r, pass = pass, fold = fold)
    rs = if _frontier_parallel(parallel) && m > 1
        _prime_caches!(fr, psi, mpo, S0)
        rngs = [MersenneTwister(rand(rng, UInt64)) for _ in 1:m]
        fetch.([Threads.@spawn(measure(j, rngs[j + 1])) for j in 0:(m - 1)])
    else
        [measure(j, rng) for j in 0:(m - 1)]
    end
    for j in 0:(m - 1)
        core, r = cores[j + 1], rs[j + 1]
        merge_ledger!(lg, r.ledger)
        nF = norm(core)
        raw = sqrt(sum(abs2, r.leaks_left) + sum(abs2, r.leaks_right))
        leak[j + 1] = r.leak
        rho[j + 1] = nF > 0 ? raw / nF : 0.0
        j >= 1 ? (met &= r.met) : (met0 = r.met)
    end
    lg.leak_initial, lg.rho_initial = leak, rho
    lg.n_root_frontier = nroot[]
    return (leak = leak, rho = rho, met = met, met0 = met0, ledger = lg, n_root = nroot[])
end

"""
    _closure_loop!(frontier, sweep!, order; maxpass, include_zero = true) -> NamedTuple

The termination logic of the order-`p` closure, independent of how a space is stored.
`frontier(j)` builds frontier `j` in the CURRENT space; `sweep!(F, j, pass)` tests it and
enriches, returning at least `(added, leak, met)`.

Pass 1 sweeps `j = 0..p-1` (`j = 1..p-1` when the caller already closed `j = 0`); later passes
sweep `j = 1..p-1` only -- the spaces are nested, so a closed `j = 0` (whose frontier is `psi`
itself and does not depend on `B`) stays closed. Each frontier is rebuilt immediately before its
sweep, so it reflects every enrichment before it. The loop is CONVERGED after a pass in which
nothing was added and every sweep met its share: every frontier was then built from the final
`B` and tested against the final frames. A pass that adds nothing but fails a share stops too
(not converged), so every non-final pass grows the space by at least one direction: termination
is guaranteed by the work ceiling, and `maxpass` is a safety bound whose hit is reported.

THE WORKING-RANK LIMIT ENDS THE LOOP TOO (plan step 3: "until every ... estimate fits the
internally allocated `tol_basis` budget OR A DERIVED WORKING-RANK LIMIT IS REACHED"). A pass in
which some enriching sweep hit the work ceiling (`r.capped`) and a share was still unmet stops
the loop, not converged (`capped = true`): the bonds that leak cannot widen, so another pass
re-measures the same leak (10x4 D=256/512 order 2 and 4, jobs 17284922/17285201: every pass
after the first added nothing and left the leak unchanged). The final `j = 0` measurement still
runs; the `j >= 1` leaks are the last measured, before that pass's enrichment.
"""
function _closure_loop!(frontier, sweep!, order::Int; maxpass::Int,
                        include_zero::Bool = true)
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    log = Any[]
    later = 1:(order - 1)
    initial = include_zero ? (0:(order - 1)) : later
    isempty(initial) &&
        return (passes = 0, converged = true, exhausted = false, capped = false, log = log)
    ok0 = true
    for pass in 1:maxpass
        grew, allmet, capped = false, true, false
        for j in (pass == 1 ? initial : later)
            r = sweep!(frontier(j), j, pass)
            push!(log, (pass = pass, frontier = j, added = r.added, leak = r.leak, met = r.met))
            grew |= r.added > 0
            allmet &= r.met
            capped |= get(r, :capped, false)
            j == 0 && (ok0 = r.met)
        end
        (!grew || isempty(later)) &&
            return (passes = pass, converged = allmet && ok0, exhausted = false,
                    capped = false, log = log)
        capped && !allmet &&
            return (passes = pass, converged = false, exhausted = false, capped = true,
                    log = log)
    end
    return (passes = maxpass, converged = false, exhausted = true, capped = false, log = log)
end

"""
    _krylov_closure!(fr, psi, mpo, S0, tau, alloc, ckw; order, workcap, nrm, parallel, rng,
                     ledger, include_zero = false, maxpass, checks = false) -> NamedTuple

Close the frontiers `j = 0..order-1` of the frames `fr` (in place) for the state `psi` with root
coefficients `S0`.

`checks = false` (THE DEFAULT, USER 2026-10-17): ONE enriching sweep per frontier, `j = 1..order-1`
in order (`0..order-1` with `include_zero`), each frontier rebuilt in the frames the previous one
grew. Nothing is measured before or after: an enriching sweep measures every bond's leak itself
and enriches only the bonds over their share, so an order-2 step is the two half-sweeps plus one
left || right sweep of `j = 1`. The measured cost of the checks (10x4, D = 512, bug2, job
17299494): check first 12 s, re-check + final `j = 0` 16-18 s, of a 46-52 s closure in a
78-84 s step (plan §8). `converged` is then every enriching sweep's own Tests A-D verdict; `ledger.leak[j]` is
frontier `j`'s leak BEFORE its enrichment (`j = 0`, not measured, is `NaN`) and `defect` sums the
measured ones; `leak_initial`/`rho_initial` stay empty.

`checks = true` is the validation path the closure was built and tested with: every frontier is
first MEASURED on the incoming frames
([`_measure_frontiers`](@ref), `ledger.leak_initial`/`rho_initial`); when they already close
every frontier (`n_passes = 0`) nothing is swept and `fr` is returned untouched -- the ordinary
CBE + local-Krylov frames are then already of order `p` to the tolerance. Otherwise the
enrichment loop runs; after every enriching sweep the root generator and `S0` are rebuilt in the
new frames, and once the loop stops `j = 0` is MEASURED (not enriched) on the final frames, so
`ledger.leak` holds every frontier's final time-weighted outside-space action and
`ledger.defect = sum_j leak[j]/(j+1)` the integrated-defect estimate (a bound for a
nearest-neighbour `H` and exact per-bond measurements; with RSVD selections the leftovers they
measure are those of sampled directions). Not converging -- pass bound or work ceiling -- is
`:closure`, never a silent pass; `capped` says the loop stopped at the working-rank limit
([`_closure_loop!`](@ref)). Returns the final `S0`, the loop's verdict and its counters.
"""
function _krylov_closure!(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                          alloc::BasisAllocation, ckw::NamedTuple; order::Int,
                          workcap::Int = 0, nrm::Float64 = 1.0, parallel::Bool = false,
                          rng::AbstractRNG = MersenneTwister(0x5EED),
                          ledger::Union{Nothing, AdaptiveLedger} = nothing,
                          include_zero::Bool = false,
                          maxpass::Int = _closure_maxpass(order), fold::Bool = false,
                          checks::Bool = false)
    t0 = time_ns()
    lg = AdaptiveLedger()
    lg.order = order
    sweepkw = (workcap = workcap, nrm = nrm, parallel = parallel, rng = rng, fold = fold)
    checks || return _single_pass_closure!(fr, psi, mpo, S0, tau, alloc, ckw, lg, t0;
                                           order = order, include_zero = include_zero,
                                           ledger = ledger, sweepkw...)
    # CHECK FIRST: what the incoming frames (the ordinary CBE + local-Krylov half-sweeps)
    # already capture of the global Krylov sequence. If they close every frontier, no closure
    # sweep runs and the frames stay exactly as the half-sweeps built them.
    pre = _measure_frontiers(fr, psi, mpo, S0, tau, alloc, ckw, order; sweepkw...)
    merge_ledger!(lg, pre.ledger)
    nroot = Ref(pre.n_root)
    S = Ref{Any}(S0)
    final_leak = Dict{Int, Float64}(j - 1 => pre.leak[j] for j in 1:order)
    gen0 = fr.gen
    res = if pre.met && (pre.met0 || !include_zero)
        (passes = 0, converged = true, exhausted = false, capped = false, log = Any[])
    else
        H0 = Ref{Any}(_root_generator(mpo, fr))
        applyB = x -> (nroot[] += 1; apply_zero_site(H0[], x))
        frontier = j -> KrylovFrontier(j == 0 ? S[] :
                                       _projected_frontiers(applyB, S[], tau, j)[j], j, fr.gen)
        # ⚠ SPECULATIVE FINAL MEASUREMENT (>= _FRONTIER_THREADS threads). A later pass usually
        # adds nothing and is then the last; its last sweep cannot change the frames unless it
        # adds. So `j = 0` is measured CONCURRENTLY on a snapshot of the frames, and kept only
        # if the loop ends in that generation -- otherwise it is discarded and redone below.
        spec = Ref{Any}(nothing)
        r0kw = (workcap = workcap, nrm = nrm, parallel = parallel, fold = fold, enrich = false)
        sweep! = (F, j, pass) -> begin
            if _frontier_parallel(parallel) && pass >= 2 && j == order - 1
                spec[] === nothing || fetch(spec[].task)
                snap = KLFrames(copy(fr.W), copy(fr.Z), fr.c, fr.lenv, fr.renv, fr.gen,
                                fr.lcache, fr.rcache, fr.eshift)
                F0 = KrylovFrontier(S[], 0, snap.gen)
                rs0 = MersenneTwister(rand(rng, UInt64))
                spec[] = (gen = snap.gen, task = Threads.@spawn(
                    _frontier_sweep!(snap, psi, mpo, F0, tau, alloc, ckw; r0kw...,
                                     rng = rs0, pass = pass + 1)))
            end
            r = _frontier_sweep!(fr, psi, mpo, F, tau, alloc, ckw; sweepkw..., pass = pass)
            merge_ledger!(lg, r.ledger)
            final_leak[j] = r.leak
            if r.added > 0
                H0[] = _root_generator(mpo, fr)
                S[] = _project_root(psi, fr)
            end
            r
        end
        loop = _closure_loop!(frontier, sweep!, order; maxpass = maxpass,
                              include_zero = include_zero)
        sr = spec[] === nothing ? nothing : (gen = spec[].gen, r = fetch(spec[].task))
        if fr.gen == gen0
            # Nothing grew: the frames and `S0` are the ones `pre` measured `j = 0` on, so its
            # measurement IS the final one (`final_leak[0]` already holds it).
        elseif sr !== nothing && sr.gen == fr.gen
            merge_ledger!(lg, sr.r.ledger)
            final_leak[0] = sr.r.leak
        else
            r0 = _frontier_sweep!(fr, psi, mpo, frontier(0), tau, alloc, ckw; r0kw...,
                                  rng = rng, pass = loop.passes + 1)
            merge_ledger!(lg, r0.ledger)
            final_leak[0] = r0.leak
        end
        loop
    end
    lg.n_passes = res.passes
    lg.leak = [final_leak[j] for j in 0:(order - 1)]
    lg.defect = sum(lg.leak[j + 1] / (j + 1) for j in 0:(order - 1))
    lg.n_root_frontier = nroot[]
    res.converged || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - t0) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (S0 = S[], passes = res.passes, converged = res.converged,
            exhausted = res.exhausted, capped = res.capped, log = res.log, ledger = lg,
            n_root = nroot[],
            n_h1 = lg.n_h1_closure)
end

"""
    _single_pass_closure!(fr, psi, mpo, S0, tau, alloc, ckw, lg, t0; order, include_zero,
                          ledger, sweepkw...) -> NamedTuple

[`_krylov_closure!`](@ref) with `checks = false`: one enriching sweep per frontier, each rebuilt
in the frames the previous sweep grew, and nothing measured before or after. Same return fields.
"""
function _single_pass_closure!(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                               alloc::BasisAllocation, ckw::NamedTuple, lg::AdaptiveLedger,
                               t0::UInt64; order::Int, include_zero::Bool,
                               ledger::Union{Nothing, AdaptiveLedger}, sweepkw...)
    nroot = Ref(0)
    S = Ref{Any}(S0)
    H0 = Ref{Any}(_root_generator(mpo, fr))
    applyB = x -> (nroot[] += 1; apply_zero_site(H0[], x))
    leak = fill(NaN, order)                      # NaN: not measured (j = 0 unless include_zero)
    log = Any[]
    met, capped = true, false
    for j in (include_zero ? 0 : 1):(order - 1)
        core = j == 0 ? S[] : _projected_frontiers(applyB, S[], tau, j)[j]
        r = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(core, j, fr.gen), tau, alloc, ckw;
                             sweepkw..., enrich = true, pass = 1)
        merge_ledger!(lg, r.ledger)
        push!(log, (pass = 1, frontier = j, added = r.added, leak = r.leak, met = r.met))
        leak[j + 1] = r.leak                     # before this sweep's enrichment
        met &= r.met
        capped |= r.capped
        if r.added > 0
            H0[] = _root_generator(mpo, fr)
            S[] = _project_root(psi, fr)
        end
    end
    lg.n_passes = 1
    lg.leak = leak
    lg.defect = sum((leak[j + 1] / (j + 1) for j in 0:(order - 1) if !isnan(leak[j + 1]));
                    init = 0.0)
    lg.n_root_frontier = nroot[]
    met || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - t0) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (S0 = S[], passes = 1, converged = met, exhausted = false, capped = capped && !met,
            log = log, ledger = lg, n_root = nroot[], n_h1 = lg.n_h1_closure)
end
