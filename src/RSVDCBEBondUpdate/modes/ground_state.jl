"""
    RSVDCBEBondUpdate.GroundState

Two-site DMRG whose bond expansion is RSVD-CBE and whose local solve is the SAME expanding
Krylov iteration the time-evolution modes use.

WHAT IS SHARED, AND IT IS ALMOST EVERYTHING. The local problem here is solved by
`_expanding_krylov`, the identical function real- and imaginary-time evolution
call. Same growth passes, same sector-graded sketch, same `P_perp` projection, same ranking,
same direct sum, same embedding of stored Krylov vectors into the grown basis, same full
reorthogonalisation. Two things are swapped:

  - the OPERATOR: `apply_h_two_site` (environment-dressed `L + h + R`) instead of a gate;
  - the REDUCTION: `LowestEigen` -- the lowest Ritz vector of the tridiagonal -- instead of
    `Propagate` -- `exp(tau*H)` applied to the current core.

So the ground-state solver is a drop-in replacement for the S step, not a second algorithm.

WHY IT IS MPO-PATH ONLY. The gate path's environments are the identity by canonical form,
which is exact for a Trotter factor but makes the local operator the bare two-site term.
Minimising `<h_{i,i+1}>` at each bond independently does not have the global ground state as
its fixed point -- each bond drives its own pair toward a local singlet and the next bond in
the sweep undoes it. DMRG converges precisely because `L + h + R` makes a LOCAL minimisation
lower the TOTAL energy. See `CBEBugOptions` for the longer argument.

HOW THIS IMPROVES ON ONE-SHOT CBE-DMRG. Standard CBE picks the expansion directions from a
single sketch of `H*Theta`, so the enlarged basis can only ever contain directions reachable
by ONE application of `H`. Here pass `k` sketches around the `k`-th Krylov vector, which
carries `H^k`, so the basis accumulates directions from successive powers -- and the
stopping signal is the Lanczos `beta`, a quantity the iteration already computes, rather
than an oversampling parameter that has to be tuned onto the problem.

THE TWO-PHASE LOCAL SOLVE, and why the growth and the eigensolve are not one loop.
`_expanding_krylov`'s tridiagonal is a Galerkin approximation across a NESTED SEQUENCE of
subspaces: `alpha_k` is exact, but `beta_k` is measured against the projector in force at
step `k` and misses whatever part of `H v_k` fell outside the then-current basis. For a
short-time exponential that is a small perturbation of an already-small correction. For an
EIGENSOLVE it is not acceptable -- the Ritz value is the answer, and one computed from an
inconsistent tridiagonal is not variational, so it can sit BELOW the true ground energy and
report false convergence. Hence:

  phase A  grow the frames, `grow_iters` passes driven by successive Krylov vectors;
  phase B  a clean Lanczos in the now-FIXED frames, which is an exact Rayleigh-Ritz and so
           variational again.

Phase B is `_expanding_krylov` with `grow_iters = 0`, so it is the same code once more and
not a second Lanczos to keep in step. Phase A costs `grow_iters` matvecs (2-3 in practice),
which is why buying back the variational guarantee is cheap.
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
    # ⛔ `defer = true` MAKES THIS SPLIT LOSSLESS. The SVD still has to happen -- it is what
    # separates `S` into the two site tensors -- but it keeps every direction instead of cutting
    # to `maxdim` here. Rank control then happens ONCE, at the end of `solve!`, so the eigensolve
    # is never handed a state that a previous bond's truncation has already narrowed.
    #
    # WHY THAT MATTERS FOR A GROUND-STATE SEARCH SPECIFICALLY: the CBE pass and the growth
    # iterations spend operator applications finding directions, and truncating at the very next
    # bond can discard them before the eigensolve has ever seen them weighted. The variational
    # bound is unaffected either way -- a smaller space still gives an upper bound -- but the
    # SWEEPS-TO-CONVERGENCE is not, which is the quantity being optimised here.
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

        # `bond_frame` demands the orthogonality centre on the bond's LEFT site. Moving right
        # that is already true (`_write_bond!` parked it on `i+1`, which is the next bond's
        # left site); moving left it is one site off, because the write parks the centre on
        # `i` and the next bond is `i-1`. One QR move either way, and idempotent when already
        # in place, so the same call serves both directions.
        #
        # SAFE AGAINST THE PREBUILT STACK in both directions, and that is not an accident:
        # moving right, `canonical!` rewrites sites <= i while the right stack is read at
        # i+2; moving left it rewrites sites >= i while the left stack is read at 1..i-1.
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
    solve!(psi, h; opts=CBEBugOptions(), n_sweeps=20, etol=1e-10, grow_iters=2)
        -> GroundStateInfo

Find the ground state of `h` in place, by two-site DMRG sweeps whose bond expansion is
RSVD-CBE and whose local solve is the shared expanding Krylov iteration.

`opts` supplies the expansion knobs (`growth`, `comp_ratio`, `dex`, ...) and the truncation
(`maxdim`, `trunc_thresh`); `opts.dt` and `opts.n_steps` are IGNORED -- there is no time step
here. `grow_iters` is the number of CBE growth passes per local solve, the ground-state
analogue of `s_iters = -g`.

`grow_iters = 1` IS THE DEFAULT AND THE MINIMUM VIABLE SETTING, and the two facts are
related. Measured at L=8, delta=1, half filling:

    grow_iters   E - E_exact    maxbd   matvec
    0            +1.625e+00     1       14      <- FROZEN at the product state
    1            -1.776e-15     16      425
    2            -3.553e-15     16      598
    3            -1.332e-15     16      631

At `grow_iters = 0` the frames never widen, so the local eigensolve minimises over a
one-dimensional space and the sweep cannot leave the start state: it reports the Neel
product-state energy `(L-1)*(-0.25) = -1.75` at bond dimension 1, and "converges" in two
sweeps. THE CBE GROWTH PASS IS DOING ALL OF THE RANK GENERATION and the eigensolve none of
it, which also means `info.converged` is not evidence of anything by itself.

One pass then reaches machine precision, and further passes only cost matvecs -- hence the
default. That is one easy model, so raise it if a harder one stalls; `benchmarks/` is where
that would show.

Sweeps alternate direction and stop when the sweep-best energy moves by less than `etol`.

THE ANSWER IS A VARIATIONAL UPPER BOUND, and stays one because the local solve's phase B
runs in fixed frames -- see the module docstring for why a single-phase expanding Lanczos
would not give that. A reported energy below the true ground energy is therefore a bug, not
a lucky sweep.

U(1) NOTE: the symmetry pins the particle-number sector to whatever `psi` starts in, so this
finds the ground state OF THAT SECTOR. `neel_state` is half filling.
"""
function solve!(psi::SymMPS, h::Union{XXZChain, MPO}; opts::CBEBugOptions = CBEBugOptions(),
                n_sweeps::Int = 20, etol::Float64 = 1e-10, grow_iters::Int = 1,
                defer_truncation::Bool = false)
    L = length(psi)
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("ground-state search needs at least two sites"))

    rng = MersenneTwister(opts.seed)
    exkw = (dex = opts.dex, growth = opts.growth, dover = opts.dover,
            comp_ratio = opts.comp_ratio, sulz_cap = opts.sulz_cap,
            rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = opts.preselect_only, exact = opts.exact, rng = rng)

    energies = Float64[]; bds = Vector{Int}[]; maxbd = Int[]
    maxexp = Int[]; kdim = Int[]; grows = Int[]
    converged = false
    prev = Inf

    for sweep in 1:n_sweeps
        nmv = Ref(0)
        e, mx, ng = _half_sweep!(psi, h, isodd(sweep) ? :right : :left, exkw, nmv;
                                 grow_iters = grow_iters, maxiter = opts.maxiter,
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

    # ⛔ THE DEFERRED TRUNCATION, AND IT MUST NOT BE SKIPPED. With `defer_truncation` the sweeps
    # never cut, so `maxdim` and `trunc_thresh` have had NO effect up to this point and the state
    # carries whatever rank the expansions produced. One recursive root-to-leaves pass applies
    # them, cutting each bond exactly once -- the same `Theta_tau` the time-evolution sweeps close
    # with (arXiv:2201.10291 section 4.2, Algorithm 7), not an edge-to-edge pass that would cut
    # every bond twice.
    #
    # The energies already recorded are unaffected: each was a variational bound in the space the
    # solve actually used. This changes the STATE that is returned, so a caller measuring the
    # energy afterwards can see it rise -- that is the truncation being paid for, once, instead of
    # L-1 times per sweep.
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
cbe_dmrg!(psi::SymMPS, h::XXZChain; opts::CBEBugOptions = CBEBugOptions(), kwargs...) =
    solve!(psi, h; opts = _with_exact(opts, true), kwargs...)

"""
    rsvd_cbe_dmrg!(psi, h; opts=CBEBugOptions(), kwargs...) -> GroundStateInfo

RSVD-CBE-DMRG: the sketched expansion, and [`solve!`](@ref)'s default. Named so the A/B
against [`cbe_dmrg!`](@ref) reads as two arms of one comparison at the call site.
"""
rsvd_cbe_dmrg!(psi::SymMPS, h::XXZChain; opts::CBEBugOptions = CBEBugOptions(), kwargs...) =
    solve!(psi, h; opts = _with_exact(opts, false), kwargs...)

# `CBEBugOptions` is a kwdef struct with no mutation, so switch one field by rebuilding. Kept
# private: callers should set `exact` in the options directly, and the two names above exist to
# make the comparison legible rather than to add a second way to configure it.
_with_exact(o::CBEBugOptions, e::Bool) = CBEBugOptions(;
    dt = o.dt, n_steps = o.n_steps, dex = o.dex, growth = o.growth, dover = o.dover,
    comp_ratio = o.comp_ratio, sulz_cap = o.sulz_cap, preselect_only = o.preselect_only,
    centre_expand = o.centre_expand, maxdim = o.maxdim, trunc_thresh = o.trunc_thresh,
    normalize = o.normalize, maxiter = o.maxiter, tol = o.tol, s_iters = o.s_iters,
    s_reorth = o.s_reorth,
    exact = e, seed = o.seed, record_magnetisation = o.record_magnetisation,
    observe = o.observe)

export GroundStateInfo, solve!, cbe_dmrg!, rsvd_cbe_dmrg!

end # module GroundState
