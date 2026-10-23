"""
    RSVDCBEBondUpdate.GroundState

Two-site DMRG with RSVD-CBE expansion and the same expanding Krylov local solve as the
time-evolution modes (`_expanding_krylov`). The operator is `apply_h_two_site` (`L + h + R`) and the
reduction is `LowestEigen`, so it is a drop-in S step. MPO path only: gate environments are
identities and would miss the global ground state. Growth passes build the frames, then a clean
Lanczos in fixed frames keeps the Ritz value variational.
"""
module GroundState

using LinearAlgebra, Random
using LurCGT, Telum

using ..RSVDCBEBondUpdate: CBEBugOptions, LowestEigen, _expanding_krylov, _reframe,
                          cbe_expand, XXZChain, MPO, apply_h_two_site,
                          truncate_recursive!,
                          boundary_channels, left_channels, right_channels,
                          push_left_channels, push_right_channels,
                          left_env_stack, right_env_stack
using ...BondUpdateBUG: SymMPS, canonical!, bond_frame, frame_theta, bond_dims, leg_dim

"""
    GroundStateInfo

Per-sweep record. `energies[k]` is the lowest Ritz value seen during sweep `k`, which is a
variational upper bound on the ground energy (phase B makes it so -- see the module
docstring). `max_bond_dims` and `max_expanded` mirror `CBEBugRunInfo`.
"""
struct GroundStateInfo
    energies::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    max_expanded::Vector{Int}
    krylov_dims::Vector{Int}
    grow_passes::Vector{Int}
    converged::Bool
end

Base.length(info::GroundStateInfo) = length(info.energies)

"""
    _bond_solve(f, applyH, expand_at, nmv; grow_iters, maxiter, tol) -> (U, V, S, energy, n_grow)

The local eigenproblem at one bond: phase A grows the frames, phase B solves exactly in them.
"""
function _bond_solve(f, applyH, expand_at, nmv; grow_iters::Int, maxiter::Int, tol::Float64)
    theta0 = frame_theta(f)

    # ---- phase A: grow the frames -------------------------------------------------
    # `maxiter = grow_iters` because this phase is here for its BASIS, not its answer; the
    # vector it returns is discarded and only `U`, `V` are kept. Running it longer would buy
    # iterations of an inconsistent tridiagonal.
    U, V, S_g, ga = grow_iters <= 0 ? (f.U0, f.V0, nothing, (; n_grow = 0)) :
        _expanding_krylov(f, applyH, expand_at, LowestEigen(), theta0, nmv;
                          maxiter = grow_iters, tol = tol, grow_iters = grow_iters)

    # ---- phase B: exact Rayleigh-Ritz in the FIXED frames -------------------------
    # `grow_iters = 0` makes `expand_at` unreachable, so the frames cannot move and the
    # tridiagonal is the true Galerkin matrix of the projected operator -- variational again.
    # `theta0` is the ORIGINAL two-site object: the grown frames contain the originals as
    # their first block, so projecting it in is lossless and it is the right start vector.
    fg = grow_iters <= 0 ? f : _reframe(f, U, S_g, V)
    U, V, S, info = _expanding_krylov(fg, applyH, identity, LowestEigen(), theta0, nmv;
                                      maxiter = maxiter, tol = tol, grow_iters = 0)
    return U, V, S, info.value, ga.n_grow
end

"""
    _write_bond!(psi, i, f, U, V, S, moving; maxdim, trunc_thresh) -> (rank, expanded)

Truncate the solved core and put it back into the state, leaving the centre where the sweep
is heading: `:right` makes `psi[i]` a left isometry and parks the centre on `psi[i+1]`,
`:left` mirrors it. Getting this backwards silently breaks the next bond's `bond_frame`,
which assumes a true canonical snapshot.
"""
function _write_bond!(psi::SymMPS, i::Int, f, U_aug, V_aug, S, moving::Symbol;
                      maxdim::Int, trunc_thresh::Float64, defer::Bool = false)
    # `defer = true` keeps every direction in this split, so rank control happens once at the end of
    # `solve!`. The CBE pass and the growth iterations find directions that a truncation at the next
    # bond could discard before the eigensolve sees them. The variational bound is unaffected, but
    # sweeps-to-convergence is not.
    res = defer ? svd(S, (1,); cutoff = 0.0, get_lists = true) :
                  svd(S, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
                      get_lists = true)
    tag = f.link_mid.itags
    if moving === :right
        psi[i]     = to_concrete(setitag(to_concrete(U_aug * res.U), 3, tag))
        psi[i + 1] = to_concrete(setitag(to_concrete((res.S * res.Vd) * V_aug), 1, tag))
        psi.center = i + 1
    else
        psi[i]     = to_concrete(setitag(to_concrete(U_aug * to_concrete(res.U * res.S)),
                                         3, tag))
        psi[i + 1] = to_concrete(setitag(to_concrete(res.Vd * V_aug), 1, tag))
        psi.center = i
    end
    return leg_dim(psi[i], 3), max(leg_dim(U_aug, 3), leg_dim(V_aug, 1))
end

"""
    _half_sweep!(psi, h, moving, exkw, nmv; ...) -> (energy, max_expanded, n_grow)

One pass over every bond, left to right (`moving = :right`) or right to left.

`O(L)` environment work: the side the sweep moves away from is CARRIED bond to bond with one
`push_*_channels` each, and the side it moves toward is read from a single stack built at the
start of the half sweep. That stack stays valid for exactly this direction, because a
left-to-right pass only ever rewrites sites at or behind the bond it is on, and the stack is
read ahead of it.
"""
function _half_sweep!(psi::SymMPS, h::Union{XXZChain, MPO}, moving::Symbol, exkw, nmv;
                      grow_iters::Int, maxiter::Int, tol::Float64,
                      maxdim::Int, trunc_thresh::Float64, defer::Bool = false)
    L = length(psi)
    e_best = Inf
    maxexp = 0
    grows = 0

    bonds = moving === :right ? (1:(L - 1)) : ((L - 1):-1:1)
    canonical!(psi, moving === :right ? 1 : L)

    # The stack for the side being swept TOWARD; the other side is carried.
    stack = moving === :right ? right_env_stack(psi, h; downto = 3) :
                                left_env_stack(psi, h; upto = L - 1)
    carried = boundary_channels(h)

    for i in bonds
        lch = moving === :right ? carried : left_channels(stack, i)
        rch = moving === :right ? right_channels(stack, i + 2) : carried

        # Moving right, the centre is already on the bond's left site; moving left it is one site
        # off, so one QR move fixes either direction (idempotent when in place). It is safe against
        # the prebuilt stacks: moving right rewrites sites <= i while the right stack is read at
        # i+2, and moving left rewrites sites >= i while the left stack is read at 1..i-1.
        canonical!(psi, i)

        f = bond_frame(psi, i)
        applyH    = Theta -> apply_h_two_site(Theta, h, i, lch, rch)
        expand_at = fr -> cbe_expand(fr, h, i, lch, rch; exkw...)

        U, V, S, e, ng = _bond_solve(f, applyH, expand_at, nmv;
                                     grow_iters = grow_iters, maxiter = maxiter, tol = tol)
        grows += ng
        isnan(e) || (e_best = min(e_best, real(e)))

        _, exp_here = _write_bond!(psi, i, f, U, V, S, moving;
                                   maxdim = maxdim, trunc_thresh = trunc_thresh,
                                   defer = defer)
        maxexp = max(maxexp, exp_here)

        # Carry the environment across the bond just written, in the sweep's direction.
        carried = moving === :right ? push_left_channels(carried, h, psi[i], i) :
                                      push_right_channels(carried, h, psi[i + 1], i + 1)
    end
    return e_best, maxexp, grows
end

"""
CBE growth passes per local solve, the ground-state analogue of `s_iters = -g`. One is the minimum
viable setting: with zero, the frames never widen, the eigensolve minimises over a one-dimensional
space, and the sweep stays at the Neel product state while reporting convergence. The growth pass
does all of the rank generation, so `info.converged` is not evidence by itself. Raise it if a harder
model stalls.
"""
const _GROW_ITERS = 1

"""
    solve!(psi, h; opts=CBEBugOptions(), n_sweeps=20, etol=1e-10) -> GroundStateInfo

Ground state of `h` in place by two-site DMRG sweeps with RSVD-CBE expansion. `opts` gives the
truncation (`maxdim`, `trunc_thresh`), the Krylov budget and `exact`; the expansion uses
[`_GROW_ITERS`](@ref) growth passes per local solve. `opts.dt` and `opts.n_steps` are ignored.
Sweeps alternate direction and stop when the sweep-best energy moves by less than `etol`.

The result is a variational upper bound (the local solve's second phase runs in fixed frames), so a
reported energy below the true ground energy is a bug. U(1) pins the particle-number sector of
`psi`.
"""
function solve!(psi::SymMPS, h::Union{XXZChain, MPO}; opts::CBEBugOptions = CBEBugOptions(),
                n_sweeps::Int = 20, etol::Float64 = 1e-10, defer_truncation::Bool = false)
    L = length(psi)
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("ground-state search needs at least two sites"))

    rng = MersenneTwister(opts.seed)
    exkw = (rmax = maximum(bond_dims(psi); init = 0), exact = opts.exact, rng = rng)

    energies = Float64[]; bds = Vector{Int}[]; maxbd = Int[]
    maxexp = Int[]; kdim = Int[]; grows = Int[]
    converged = false
    prev = Inf

    for sweep in 1:n_sweeps
        nmv = Ref(0)
        e, mx, ng = _half_sweep!(psi, h, isodd(sweep) ? :right : :left, exkw, nmv;
                                 grow_iters = _GROW_ITERS, maxiter = opts.maxiter,
                                 tol = opts.tol, maxdim = opts.maxdim,
                                 trunc_thresh = opts.trunc_thresh,
                                 defer = defer_truncation)
        push!(energies, e)
        d = bond_dims(psi); push!(bds, d); push!(maxbd, maximum(d; init = 0))
        push!(maxexp, mx); push!(kdim, nmv[]); push!(grows, ng)

        # Compared against the PREVIOUS SWEEP, not the previous bond: within a sweep the
        # energy falls monotonically by construction, so a per-bond test would stop early on
        # any flat stretch.
        if abs(e - prev) < etol
            converged = true
            break
        end
        prev = e
    end

    # ── deferred truncation: must not be skipped ──
    # One recursive root-to-leaves pass cuts each bond once (`Theta_tau`, arXiv:2201.10291 Alg. 7).
    # Recorded energies are unaffected (variational in the space used); the returned state is cut.
    if defer_truncation
        canonical!(psi, max(1, min(L - 1, L ÷ 2)))
        truncate_recursive!(psi, max(1, min(L - 1, L ÷ 2));
                            maxdim = opts.maxdim, cutoff = opts.trunc_thresh)
        # OVERWRITE, not push: these arrays are indexed BY SWEEP and an extra entry would
        # invent a sweep that never ran.
        d = bond_dims(psi); bds[end] = d; maxbd[end] = maximum(d; init = 0)
    end

    return GroundStateInfo(energies, bds, maxbd, maxexp, kdim, grows, converged)
end

"""
    cbe_dmrg!(psi, h; opts=CBEBugOptions(), kwargs...) -> GroundStateInfo

Plain CBE-DMRG: the expansion uses the COMPLETE fused basis, so the preselection sees the
exact `H*Theta` and its SVD returns the true leading complement directions.

This is the reference arm. It forms the rank-4 two-site object (cost `d*chi` per side against
the sketch's `~1.2*budget`), which is exactly what the randomised version exists to avoid --
so it is the thing to measure against, not a better default.
"""
cbe_dmrg!(psi::SymMPS, h::Union{XXZChain, MPO}; opts::CBEBugOptions = CBEBugOptions(),
          kwargs...) =
    solve!(psi, h; opts = _with_exact(opts, true), kwargs...)

"""
    rsvd_cbe_dmrg!(psi, h; opts=CBEBugOptions(), kwargs...) -> GroundStateInfo

RSVD-CBE-DMRG: the sketched expansion, and [`solve!`](@ref)'s default. Named so the A/B
against [`cbe_dmrg!`](@ref) reads as two arms of one comparison at the call site.
"""
rsvd_cbe_dmrg!(psi::SymMPS, h::Union{XXZChain, MPO}; opts::CBEBugOptions = CBEBugOptions(),
               kwargs...) =
    solve!(psi, h; opts = _with_exact(opts, false), kwargs...)

# Kept private: callers set `exact` in the options directly. The copy runs over `fieldnames`, not a
# hand-written field list, so a field added later is not silently reset to its default while its
# neighbours keep their values.
_with_exact(o::CBEBugOptions, e::Bool) = CBEBugOptions(;
    (f => (f === :exact ? e : getfield(o, f)) for f in fieldnames(CBEBugOptions))...)

export GroundStateInfo, solve!, cbe_dmrg!, rsvd_cbe_dmrg!

end # module GroundState
