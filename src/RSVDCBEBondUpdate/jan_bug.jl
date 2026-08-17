# Jan BUG: one CBE basis sweep across the WHOLE chain, then one 0-site bond update at the far
# end, then one truncation.
#
# THE SCHEME, as specified:
#
#   1. bond-canonical form at the first bond
#   2. at each bond in turn the FULL KLS step: the RSVD-CBE expansion supplies the K/L basis
#      update, and the S step is taken as well -- then QR the evolved bond and DISCARD `R`, so
#      the basis moves where the dynamics points but no time evolution accumulates. That is
#      what "we are not dynamically evolving the bonds" means: the amplitude is thrown away and
#      the ORIGINAL state, re-expressed in the new basis, is what travels on.
#   3. update the environment through the basis just written, and carry on
#   4. at the LAST bond there is no QR: the S step is KEPT, and that is the step
#   5. sweep the chain with SVDs to control the rank, now that the sweep is complete
#   6. that is one sweep = one step of `tau`. The next step sweeps back and updates bond 1.
#
# WHERE IT SITS BETWEEN THE OTHER TWO. All three are BUG -- augment the basis first, then ONE
# Galerkin solve, no projector splitting and no backward substep. They differ in which bond
# carries the update and how the bases are found:
#
#                     connecting tensor        bases from             evolutions per step
#   Lubich (ours)     centre bond (0-site)     CBE, two half-sweeps    1
#   Mendl (paper)     centre SITE (1-site)     local ODE per tensor    1 (+ L-1 basis solves)
#   Jan (here)        END bond   (0-site)      CBE, one full sweep     1
#
# So this keeps the 0-site bargain -- pay once to widen the bond, then update at 0-site cost --
# and moves the single update to the end of a full-chain basis sweep.
#
# BOND-CANONICAL FORM NEEDS NO NEW MACHINERY. `canonical!(psi, 1)` puts every amplitude at the
# left end, and `_frame_from(psi[1], psi[2]) = U0 S0 V0` factors the first bond: `S0` IS the bond
# tensor of bond-canonical form. Each `_absorb_left!` then carries that form one bond to the
# right. Nothing stores an explicit bond tensor, which is why no rank-4 object is ever built.
#
# WHY THE RANK GROWS, the question to ask of any BUG variant, and here it grows at EVERY bond
# rather than only at the far end. `exp(tau H_eff)` is not a left multiplication:
# `H_eff S = L S + S R + sum_t ol_t S or_t` acts from both sides, so it does not preserve the
# rank of `S`, and `S1` is generically of full rank in the expanded frames even though `S0`
# carries only the state's rank `r`. `orth(U_ex S1)` therefore has more columns than `U0` did.
# Expansion alone could never do this -- it is lossless, hence rank-preserving. The environment
# carried past bond `i` must be pushed through the NEW basis for the following bonds to see it.
#
# WHAT THE DISCARDED `R` COSTS, stated plainly rather than hidden. `range(orth(U_ex S1))` equals
# `range(U_ex)` whenever `S1` has full row rank, and then re-expressing the state in it is
# lossless. When the evolved core is wider than it is tall the new basis is a genuine subspace
# and the re-expression PROJECTS -- that projection is the K-step's approximation, the same one
# every BUG makes, and it is why the S step has to be redone in the final basis rather than
# accumulated along the way.
#
# THE WRITE-BACK IS SAFE HERE, and it is worth saying why, because `cbe_lubich.jl`'s header
# records `_absorb_left!` breaking that sweep. There the failure was re-deriving the NEXT frame
# from a ZERO-PADDED tensor with `svd(cutoff = 0)` while the only evolution sat at the centre, so
# the collapse propagated back into the bond just expanded and L=18 froze at maxbond 2. Here
# nothing is ever zero-padded: what is written back is `orth(U_ex S1)`, a basis every direction
# of which the S step gave weight to.
#
# TRUNCATION IS ONCE PER STEP, AFTER THE FULL SWEEP, never during it: rank control is a sweep of
# SVDs at the end, and the factorisations inside the sweep use a numerical-zero cutoff only. The
# round trip runs both directions and ends at the end the sweep finished at, so the next
# (opposite) sweep starts exactly there with no extra canonicalisation.
#
# THE PROBE GOES ON `H Theta`, NOT ON THE PROJECTOR, and for this variant that is not a
# preference. `cbe_expand` contracts a Gaussian into the RIGHT leg to expand the LEFT frame,
#
#     Y = (H Theta) Om_R'    ->    Q_L = orth(P_perp Y)
#
# so the side being expanded stays EXACT and the candidates estimate the RANGE of
# `P_perp (H Theta)` -- where `H` actually points. A `two_sided` variant that folded both
# Gaussians onto the projectors instead, ranking by the small `M = Q_L'(H Theta)Q_R`, was
# implemented, measured HERE and removed. Its ranking matrix was independent of `chi`, but `M`
# COUPLES THE SIDES: at a BOUNDARY bond the outer side is a single site, so `room_r = d - r`
# hits zero at `r = d`, and the branch then declined to expand the bond AT ALL -- precisely the
# bond this scheme takes its step on. Off a product state it was worse still: `r = 1` leaves each
# slice one random vector and `M` needs both non-negligible at once, so nothing was ever
# admitted. Measured on XXZ, L=4/6: first order instead of second, `2.0e-2` instead of `1.8e-5`
# over eight steps (against a standstill of `2.2e-2`, i.e. the step barely moved the state), and
# the rank never left `chi = 1`.
#
# MPO ONLY, deliberately. Everything here goes through `mpo.jl`'s `W^[i]` and rank-3
# environments, so a long-range or site-dependent `H` needs a different `MPO` and no change to
# this file. The channel path could not express one.

"""
    JanBugInfo

What one [`jan_bug_step!`](@ref) did.

  - `direction` -- `:right` (the update lands on bond `L-1`) or `:left` (on bond 1).
  - `expanded` -- the bond dimension the basis update left at each bond, i.e. the rank of
    `orth(U_ex S1)` after the QR, before the final truncation. `n_new` is what CBE admitted
    into the frame at that bond, which is an upper bound on it.
  - `end_rank` -- kept rank at the updated bond after its SVD.
  - `err_pre`, `err_fnl` -- largest preselection / final-selection weight discarded.
  - `discarded` -- largest relative weight the post-sweep truncation threw away.
  - `krylov_dim` -- Krylov dimension of the single 0-site exponential.
  - `peak_elements` -- working set: `state + transients alive at that moment`, the definition
    `CBEBugInfo.peak_elements` and `TDVP2Info.peak_elements` use, so the three are comparable.
    The transients here are one expanded frame and one bond tensor; never a rank-4 block.
"""
struct JanBugInfo
    direction::Symbol
    expanded::Vector{Int}
    n_new::Vector{Int}
    end_rank::Int
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dim::Int
    peak_elements::Int
end

"""
    _truncate_round_trip!(psi; maxdim, cutoff, ending) -> Float64

Compress every bond once the sweep is complete (Sulz Algorithm 1), and leave the orthogonality
centre at `ending` (`:right` = site `L`, `:left` = site 1) -- the end the next, opposite sweep
starts from. Returns the largest relative weight discarded.

Two passes, because a bond's truncation decision depends on the gauge: the first pass starts at
the centre where the sweep left it, the second returns. `truncate_sweep!` is the same algorithm
with the passes fixed in the other order, which would leave the centre at the wrong end here and
cost an extra canonicalisation.
"""
function _truncate_round_trip!(psi::SymMPS; maxdim::Int, cutoff::Float64, ending::Symbol)
    L = length(psi)
    worst = 0.0

    "Right to left: truncate bond j-1, centre ends at site 1."
    function down!()
        for j in L:-1:2
            res = svd(psi[j], (1,); cutoff = cutoff, Nkeep = maxdim, get_lists = true)
            worst = max(worst, _trunc_weight(res))
            tag = psi[j - 1].inds[3].itags
            M = to_concrete(res.U * res.S)
            psi[j]     = to_concrete(setitag(res.Vd, 1, tag))
            psi[j - 1] = to_concrete(setitag(contract(psi[j - 1], (3,), M, (1,)), 3, tag))
            psi.center = j - 1
        end
    end

    "Left to right: truncate bond i, centre ends at site L."
    function up!()
        for i in 1:(L - 1)
            res = svd(psi[i], (1, 2); cutoff = cutoff, Nkeep = maxdim, get_lists = true)
            worst = max(worst, _trunc_weight(res))
            tag = psi[i].inds[3].itags
            M = to_concrete(res.S * res.Vd)
            psi[i]     = to_concrete(setitag(res.U, 3, tag))
            psi[i + 1] = to_concrete(setitag(contract(M, (2,), psi[i + 1], (1,)), 1, tag))
            psi.center = i + 1
        end
    end

    if ending === :right
        down!(); up!()                     # starts at site L, ends at site L
    else
        up!(); down!()                     # starts at site 1, ends at site 1
    end
    return worst
end

"Put a frame's two bond legs back on the state's own link tags. Both, not one: adjacent tensors
in a `SymMPS` must agree on the shared link tag, and retagging one side only leaves
`psi[i].inds[3] != psi[i+1].inds[1]`, which `contract` rejects on the next sweep."
_relink(T, t1, t3) = to_concrete(setitag(setitag(T, 1, t1), 3, t3))

"""
    jan_bug_step!(psi, mpo, tau; direction=:right, kwargs...) -> JanBugInfo

One Jan-BUG step of `tau = -im*dt`, in place.

`direction = :right` starts bond-canonical at bond 1, re-bases every bond on the way out and
takes the 0-site update on bond `L-1`; `:left` is the mirror and updates bond 1. Alternating them
is one step each, and [`jan_bug!`](@ref) does the alternation.

Keywords: `dex`, `growth`, `dover`, `comp_ratio`, `sulz_cap`, `preselect_only`, `exact` forward
to [`cbe_expand`](@ref). `maxdim` and `trunc_thresh` control the post-sweep truncation
(`truncate = false` disables it, to watch the rank ratchet by `Dex` per bond per step); `maxiter`
and `tol` bound the single 0-site Krylov exponential.

`O(L)` environment work: the side the sweep travels away from is CARRIED bond to bond through
the expanded frames, the other is read from a stack prebuilt once. The sweep only ever writes to
sites that stack does not depend on -- reading a stale environment does not throw, it silently
evolves with the wrong Hamiltonian.
"""
function jan_bug_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                       direction::Symbol = :right,
                       dex::Int = 0,
                       growth::Float64 = 2.0,
                       dover::Union{Nothing, Int} = nothing,
                       comp_ratio::Union{Float64, Nothing} = 0.5,
                       sulz_cap::Bool = false,
                       preselect_only::Bool = false,
                       exact::Bool = false,
                       root::Union{Nothing, Int} = nothing,
                       grow_iters::Int = 0,
                       lossless::Bool = false,
                       truncate::Bool = true,
                       maxdim::Int = 200,
                       trunc_thresh::Float64 = 1e-12,
                       maxiter::Int = 30,
                       tol::Float64 = 1e-15,
                       rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("jan_bug_step! needs at least two sites"))
    direction in (:right, :left) || throw(ArgumentError(
        "direction must be :right or :left, got $direction"))

    expanded = zeros(Int, L - 1)
    n_new    = zeros(Int, L - 1)
    err_pre = 0.0
    err_fnl = 0.0
    peak = 0
    cut = max(trunc_thresh, 1e-14)

    # `rmax` is read globally, once, before the sweep widens anything -- as in
    # `cbe_lubich_sweep`, since the (optional) Sulz bound is a statement about the state.
    exkw = (dex = dex, growth = growth, dover = dover,
            comp_ratio = comp_ratio, sulz_cap = sulz_cap,
            rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only,
            exact = exact, rng = rng)

    peak!(transients...) = (peak = max(peak, _state_stored(psi) +
                                       sum(tensor_elements(t) for t in transients; init = 0)))
    function record!(ex, i, side::Symbol)
        err_pre = max(err_pre, ex.err_pre)
        err_fnl = max(err_fnl, ex.err_fnl)
        expanded[i] = side === :left ? leg_dim(ex.U_ex, 3) : leg_dim(ex.V_ex, 1)
        n_new[i] = side === :left ? ex.n_new_l : ex.n_new_r
        return ex
    end

    nmv = Ref(0)

    """
    The KLS step at ONE bond: the CBE expansion supplies the K/L basis update, and this is the
    S step in the expanded frames.

    `S0 = U_ex'(A_i A_{i+1})V_ex'` is evaluated as `(U_ex'A_i)(A_{i+1}V_ex')`, so the two-site
    block is never formed; it reproduces the bond tensor exactly because each expansion contains
    its frame. `exp(tau H_eff)` is NOT a left multiplication -- `H_eff S = L S + S R + sum_t
    ol_t S or_t` acts from both sides -- so `S1` is generically of full rank even though `S0`
    has the state's rank `r`. That is where the rank comes from.
    """
    function s_step(i, ex, lch, rch; grow::Int = 0)
        A = to_concrete(contract(ex.U_ex', (1, 2), psi[i], (1, 2)))       # (b_L, link)
        B = to_concrete(contract(psi[i + 1], (2, 3), ex.V_ex', (2, 3)))   # (link, b_R)
        S0 = to_concrete(contract(A, (2,), B, (1,)))                      # (b_L, b_R)
        peak!(ex.U_ex, ex.V_ex, S0)
        grow <= 0 && begin
            H0 = zero_site_h(mpo, i, lch, rch, ex.U_ex, ex.V_ex)
            return (expv(x -> (nmv[] += 1; apply_zero_site(H0, x)), tau, S0;
                         hermitian = true, maxiter = maxiter, tol = tol),
                    ex.U_ex, ex.V_ex)
        end
        # EXPANDING-BASIS LANCZOS, for the KEPT bond only. `maxiter` adds vectors to a fixed
        # subspace; this regrows the subspace around each vector, which is a different thing --
        # measured at `root = 4`, L=6 full rank: `1.227e-5 -> 2.049e-8 -> 2.504e-15` as
        # `grow_iters` goes `0 -> 2 -> 4`, while at a BOUNDARY root the rank climbs 4 -> 12 and
        # the error does not move at all (`benchmarks/root_position.jl`). Worth paying for only
        # where the root's ceiling `dim(U_ex) x dim(V_ex)` is not already below the space.
        #
        # It costs the 0-site bargain: `_expanding_krylov` forms the rank-4 block each
        # iteration, so the matvec is `O(chi^2 d^2)` rather than `O(chi^2 w)`.
        f_root = BondFrame(ex.U_ex, S0, ex.V_ex,
                           ex.U_ex.inds[1], ex.U_ex.inds[2], ex.U_ex.inds[3],
                           ex.V_ex.inds[2], ex.V_ex.inds[3],
                           max(leg_dim(ex.U_ex, 3), leg_dim(ex.V_ex, 1)),
                           copy(ex.U_ex.spaces[3]))
        theta0 = to_concrete(to_concrete(ex.U_ex * S0) * ex.V_ex)
        peak!(theta0)
        U_aug, V_aug, S_new, _ = _expanding_krylov(
            f_root,
            th -> apply_h_two_site(th, mpo, i, lch, rch),
            fr -> cbe_expand(fr, mpo, i, lch, rch; exkw...),
            Propagate(tau), theta0, nmv;
            maxiter = maxiter, tol = tol, grow_iters = grow)
        return (S_new, U_aug, V_aug)
    end

    """
    QR the evolved bond and DISCARD `R`.

    `Q = orth(U_ex S1)` keeps the direction the S step moved the basis into and nothing of how
    far it moved: the amplitude lives entirely in the discarded `R`, so no time evolution
    accumulates along the sweep. Telum has no QR primitive, so `svd(...).U` is the
    orthonormalisation -- the same substitution `_reortho_left` makes -- and the cutoff is
    numerical-zero only, because rank control is the job of the sweep at the END of the step,
    not of this factorisation.
    """
    _basis_left(U_ex, S1) =
        to_concrete(svd(to_concrete(contract(U_ex, (3,), S1, (1,))), (1, 2);
                        cutoff = 1e-14).U)

    "Mirror: `Q = orth(S1 V_ex)` as a right isometry, `R` discarded."
    _basis_right(V_ex, S1) =
        to_concrete(svd(to_concrete(contract(S1, (2,), V_ex, (1,))), (1,);
                        cutoff = 1e-14).Vd)

    """
    The KEPT bond: `S1` is written into the state instead of being reduced to a basis. Split by
    an exact SVD (rank control is the final sweep's job) with the amplitude on the side the
    sweep is travelling TOWARDS, so the sweep can carry on through it.

    `U`, `V` are passed rather than read off `ex` because an expanding-basis solve returns
    frames WIDER than the ones it was given.
    """
    function write_kept!(i, U, V, S1, amplitude::Symbol)
        res = svd(S1, (1,); cutoff = 1e-14, get_lists = true)
        t1, t3 = psi[i].inds[1].itags, psi[i].inds[3].itags
        t3r = psi[i + 1].inds[3].itags
        if amplitude === :right
            psi[i]     = _relink(to_concrete(U * res.U), t1, t3)
            psi[i + 1] = _relink(to_concrete((res.S * res.Vd) * V), t3, t3r)
            psi.center = i + 1
        else
            psi[i]     = _relink(to_concrete(U * to_concrete(res.U * res.S)), t1, t3)
            psi[i + 1] = _relink(to_concrete(res.Vd * V), t3, t3r)
            psi.center = i
        end
        return leg_dim(psi[i], 3)
    end

    # WHICH BOND KEEPS ITS S STEP. Exactly one may: if every bond kept it the state would be
    # advanced by `tau` once per bond, i.e. by `(L-1) tau` in a step meant to be `tau` -- the
    # over-counting `cbe_lubich.jl`'s header records as already tried and abandoned.
    #
    # `root = nothing` keeps it at the FAR bond, which is the variant as originally specified.
    # An explicit `root` keeps it there instead and the sweep CARRIES ON to the end afterwards,
    # so the bonds past it still get their basis refreshed -- against the now-evolved state,
    # which is what the returning sweep will start from.
    #
    # WHY THE CHOICE MATTERS, measured rather than argued (`benchmarks/root_position.jl`, L=6,
    # full rank, one step at dt=0.01): the single kept evolution is a Galerkin problem in
    # `span(U_ex) (x) span(V_ex)`, and at a BOUNDARY bond the outer factor is one site, so that
    # space is 16-dimensional inside a 20-dimensional Sz=0 sector -- incomplete at any rank,
    # any Krylov budget. Moving one bond in lifts the ceiling to 64, and there growth reaches
    # it: `1.745e-5` at the boundary against `2.5e-15` at bond `L-2` with `grow_iters = 4`.
    c = root === nothing ? (direction === :right ? L - 1 : 1) : root
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))

    """
    Solve at the KEPT bond from an explicit `S0`, honouring `grow_iters`. Split out because the
    lossless sweep builds `S0` from a CARRY rather than from `psi[i]`.
    """
    function root_solve(i, ex, lch, rch, S0)
        peak!(ex.U_ex, ex.V_ex, S0)
        if grow_iters <= 0
            H0 = zero_site_h(mpo, i, lch, rch, ex.U_ex, ex.V_ex)
            return (expv(x -> (nmv[] += 1; apply_zero_site(H0, x)), tau, S0;
                         hermitian = true, maxiter = maxiter, tol = tol),
                    ex.U_ex, ex.V_ex)
        end
        f_root = BondFrame(ex.U_ex, S0, ex.V_ex,
                           ex.U_ex.inds[1], ex.U_ex.inds[2], ex.U_ex.inds[3],
                           ex.V_ex.inds[2], ex.V_ex.inds[3],
                           max(leg_dim(ex.U_ex, 3), leg_dim(ex.V_ex, 1)),
                           copy(ex.U_ex.spaces[3]))
        theta0 = to_concrete(to_concrete(ex.U_ex * S0) * ex.V_ex)
        peak!(theta0)
        U_aug, V_aug, S_new, _ = _expanding_krylov(
            f_root,
            th -> apply_h_two_site(th, mpo, i, lch, rch),
            fr -> cbe_expand(fr, mpo, i, lch, rch; exkw...),
            Propagate(tau), theta0, nmv;
            maxiter = maxiter, tol = tol, grow_iters = grow_iters)
        return (S_new, U_aug, V_aug)
    end

    end_rank = 0
    if lossless
        # ── LOSSLESS BASIS SWEEP + ONE KLS AT THE ROOT ────────────────────────────────
        #
        # No S step away from the root, therefore no `R` to discard and no projection: the
        # sweep is a pure CBE expansion carried bond to bond, which is exactly lossless
        # (`oplus` puts `U0` first, so `range(U_ex) contains range(A_i)`).
        #
        # WHY THE CARRY AND NOT `_absorb_left!`. Writing the raw expanded frame into the state
        # leaves a ZERO-PADDED remainder in `psi[i+1]` -- the new directions have no weight yet
        # -- and the next bond's `_frame_from` SVDs that tensor and drops them again.
        # `cbe_lubich.jl`'s header records that exact experiment: the widening was undone bond
        # by bond and L=18 froze at maxbond 2. Carrying `C = U_ex'(C psi[i])` instead means no
        # frame is ever re-derived from a padded tensor, so the widening chains.
        #
        # WHY THE SWEEP STOPS AT THE ROOT rather than continuing to the end. MEASURED: with the
        # QR-discard sweep, every bond AFTER the kept step projects the evolution that step just
        # performed, and it costs orders -- L=6, one step at dt=0.01, kept step at bond
        # 5/4/3: 2.1e-5 / 7.7e-4 / 5.0e-3, and the dt-order collapses to 0. A discard is only
        # harmless BEFORE the evolution. Here there is no discard at all, but there is also
        # nothing for a post-root pass to do: a lossless expansion past the root is a pure gauge
        # change that the closing truncation undoes.
        Wf = Vector{Any}(undef, L)
        if direction === :right
            canonical!(psi, 1)
            rstack = right_env_stack(psi, mpo; downto = 3)
            lch = boundary_channels(mpo)
            C = nothing
            for i in 1:c
                K = C === nothing ? psi[i] :
                    to_concrete(contract(C, (2,), psi[i], (1,)))
                rch = right_channels(rstack, i + 2)
                f = _frame_from(K, psi[i + 1])
                ex = record!(cbe_expand(f, mpo, i, lch, rch; exkw...), i, :left)
                if i == c
                    A = to_concrete(contract(ex.U_ex', (1, 2), K, (1, 2)))
                    B = to_concrete(contract(psi[i + 1], (2, 3), ex.V_ex', (2, 3)))
                    S0 = to_concrete(contract(A, (2,), B, (1,)))
                    S1, U, V = root_solve(i, ex, lch, rch, S0)
                    res = svd(S1, (1,); cutoff = 1e-14, get_lists = true)
                    for j in 1:(i - 1)
                        psi[j] = _relink(Wf[j], psi[j].inds[1].itags, psi[j].inds[3].itags)
                    end
                    t1, t3 = psi[i].inds[1].itags, psi[i].inds[3].itags
                    t3r = psi[i + 1].inds[3].itags
                    psi[i]     = _relink(to_concrete(U * res.U), t1, t3)
                    psi[i + 1] = _relink(to_concrete((res.S * res.Vd) * V), t3, t3r)
                    psi.center = i + 1
                    end_rank = leg_dim(psi[i], 3)
                    expanded[i] = end_rank
                else
                    Wf[i] = ex.U_ex
                    expanded[i] = leg_dim(ex.U_ex, 3)
                    C = to_concrete(contract(ex.U_ex', (1, 2), K, (1, 2)))
                    lch = push_left_channels(lch, mpo, ex.U_ex, i)
                end
            end
        else
            canonical!(psi, L)
            lstack = left_env_stack(psi, mpo; upto = L - 2)
            rch = boundary_channels(mpo)
            D = nothing
            for i in (L - 1):-1:c
                B = D === nothing ? psi[i + 1] :
                    to_concrete(contract(psi[i + 1], (3,), D, (1,)))
                lch = left_channels(lstack, i)
                f = _frame_from(psi[i], B)
                ex = record!(cbe_expand(f, mpo, i, lch, rch; exkw...), i, :right)
                if i == c
                    A = to_concrete(contract(ex.U_ex', (1, 2), psi[i], (1, 2)))
                    Bp = to_concrete(contract(B, (2, 3), ex.V_ex', (2, 3)))
                    S0 = to_concrete(contract(A, (2,), Bp, (1,)))
                    S1, U, V = root_solve(i, ex, lch, rch, S0)
                    res = svd(S1, (1,); cutoff = 1e-14, get_lists = true)
                    for j in (i + 2):L
                        psi[j] = _relink(Wf[j], psi[j].inds[1].itags, psi[j].inds[3].itags)
                    end
                    t1, t3 = psi[i].inds[1].itags, psi[i].inds[3].itags
                    t3r = psi[i + 1].inds[3].itags
                    psi[i]     = _relink(to_concrete(U * to_concrete(res.U * res.S)), t1, t3)
                    psi[i + 1] = _relink(to_concrete(res.Vd * V), t3, t3r)
                    psi.center = i
                    end_rank = leg_dim(psi[i], 3)
                    expanded[i] = end_rank
                else
                    Wf[i + 1] = ex.V_ex
                    expanded[i] = leg_dim(ex.V_ex, 1)
                    D = to_concrete(contract(B, (2, 3), ex.V_ex', (2, 3)))
                    rch = push_right_channels(rch, mpo, ex.V_ex, i + 1)
                end
            end
        end
    elseif direction === :right
        # Bond-canonical at bond 1: all amplitude at the left end, so `_frame_from`'s `S0` is
        # that bond's tensor. The right stack is prebuilt ONCE: at bond `i` it is read at link
        # `i+2`, which depends only on sites `>= i+2`, and the sweep has written only `psi[i]`
        # and `psi[i+1]` by then -- true whether bond `i` kept its S step or not.
        canonical!(psi, 1)
        rstack = right_env_stack(psi, mpo; downto = 3)
        lch = boundary_channels(mpo)

        for i in 1:(L - 1)
            rch = right_channels(rstack, i + 2)
            f = _frame_from(psi[i], psi[i + 1])
            ex = record!(cbe_expand(f, mpo, i, lch, rch; exkw...), i, :left)
            S1, U, V = s_step(i, ex, lch, rch; grow = (i == c ? grow_iters : 0))
            if i == c
                end_rank = write_kept!(i, U, V, S1, :right)      # the S step is KEPT here
                expanded[i] = end_rank
            else
                # QR, R discarded: `psi[i]` becomes the evolved BASIS and the ORIGINAL
                # amplitude, re-expressed in it, goes into `psi[i+1]`. No time evolution
                # accumulates -- that is what "not dynamically evolving the bonds" means.
                _absorb_left!(psi, i, _basis_left(ex.U_ex, S1))
                expanded[i] = leg_dim(psi[i], 3)
            end
            i < L - 1 && (lch = push_left_channels(lch, mpo, psi[i], i))
        end
    else
        canonical!(psi, L)
        lstack = left_env_stack(psi, mpo; upto = L - 2)
        rch = boundary_channels(mpo)

        for i in (L - 1):-1:1
            lch = left_channels(lstack, i)
            f = _frame_from(psi[i], psi[i + 1])
            ex = record!(cbe_expand(f, mpo, i, lch, rch; exkw...), i, :right)
            S1, U, V = s_step(i, ex, lch, rch; grow = (i == c ? grow_iters : 0))
            if i == c
                end_rank = write_kept!(i, U, V, S1, :left)       # the S step is KEPT here
                expanded[i] = end_rank
            else
                _absorb_right!(psi, i, _basis_right(ex.V_ex, S1))
                expanded[i] = leg_dim(psi[i + 1], 1)
            end
            i > 1 && (rch = push_right_channels(rch, mpo, psi[i + 1], i + 1))
        end
    end

    # ONE truncation, now that the sweep is complete and the update has propagated into every
    # direction the expansion added. Ends at the end the sweep finished at.
    discarded = truncate ?
        _truncate_round_trip!(psi; maxdim = maxdim, cutoff = cut, ending = direction) : 0.0

    return JanBugInfo(direction, expanded, n_new, end_rank,
                      err_pre, err_fnl, discarded, nmv[], peak)
end

jan_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    jan_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

jan_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    jan_bug_step!(psi, mpo_from_terms(h), tau; kwargs...)

# ── driver ───────────────────────────────────────────────────────────────────

"""
    jan_bug!(psi, h; opts=CBEBugOptions(), start=:right) -> CBEBugRunInfo

Evolve `psi` in place for `opts.n_steps` steps of `opts.dt`, ALTERNATING the sweep direction:
step 1 sweeps out and updates bond `L-1`, step 2 sweeps back and updates bond 1, and so on. Each
sweep is a full step of `tau = -im*dt`.

The alternation is not cosmetic. The post-sweep truncation leaves the centre at the end the sweep
finished at, which is where the next sweep begins, so no extra canonicalisation is needed -- and
consecutive steps are mirror images, which is the only structural reason to expect anything
better than first order from a scheme whose Galerkin problem sits on a boundary bond.

A term list is converted to an MPO ONCE, here, as in [`cbe_lubich_bug!`](@ref); pass an `MPO` to
skip the conversion. `opts` is shared with the other integrators so one run script can switch
between them.
"""
function jan_bug!(psi::SymMPS, h::Union{XXZChain, MPO};
                  opts::CBEBugOptions = CBEBugOptions(), start::Symbol = :right,
                  root::Union{Nothing, Int} = nothing, grow_iters::Int = 0,
                  lossless::Bool = false)
    mpo = h isa MPO ? h : mpo_from_terms(h)
    start in (:right, :left) || throw(ArgumentError(
        "start must be :right or :left, got $start"))
    rng = MersenneTwister(opts.seed)

    times = Float64[]; norms = Float64[]
    bdims = Vector{Int}[]; maxb = Int[]
    expanded = Vector{Int}[]; maxex = Int[]
    cranks = Int[]; epre = Float64[]; efnl = Float64[]; disc = Float64[]
    mags = Vector{Float64}[]; obs = Any[]
    kdim = Int[]; maug = Float64[]

    dir = start
    for step in 1:opts.n_steps
        info = jan_bug_step!(psi, mpo, ComplexF64(-im * opts.dt);
                             direction = dir,
                             dex = opts.dex, growth = opts.growth,
                             dover = opts.dover,
                             comp_ratio = opts.comp_ratio,
                             sulz_cap = opts.sulz_cap,
                             preselect_only = opts.preselect_only,
                             exact = opts.exact,
                             root = root, grow_iters = grow_iters, lossless = lossless,
                             maxdim = opts.maxdim,
                             trunc_thresh = opts.trunc_thresh,
                             maxiter = opts.maxiter, tol = opts.tol, rng = rng)
        dir = dir === :right ? :left : :right

        n = norm(psi)                                # recorded BEFORE renormalising
        if opts.normalize && n > 0
            psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center])
        end

        bd = bond_dims(psi)
        push!(times, step * opts.dt); push!(norms, n)
        push!(bdims, bd); push!(maxb, maximum(bd; init = 0))
        push!(expanded, info.expanded); push!(maxex, maximum(info.expanded; init = 0))
        push!(cranks, info.end_rank)                 # the updated bond, not a centre bond
        push!(epre, info.err_pre); push!(efnl, info.err_fnl)
        push!(disc, info.discarded)
        push!(kdim, info.krylov_dim)
        push!(maug, isempty(info.expanded) ? 0.0 :
                    sum(info.expanded) / length(info.expanded))

        opts.record_magnetisation && push!(mags, magnetisation(copy(psi)))
        opts.observe === nothing || push!(obs, opts.observe(copy(psi), step, step * opts.dt))
    end

    return CBEBugRunInfo(times, norms, bdims, maxb, expanded, maxex, cranks,
                         epre, efnl, disc, mags, obs, kdim, maug)
end
