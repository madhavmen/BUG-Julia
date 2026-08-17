# THE FINAL SWEEP: RSVD-CBE BOND EXPANSION + THE FIXED-RANK BUG SWEEP.
#
# In the same spirit as CBE-TDVP -- widen the bond first, then update -- but the update is the
# BUG integrator of arXiv:2606.28169 §1 instead of a one-site TDVP.
#
# NOTE ON THE REFERENCE. The request named §2; the fixed-rank BUG sweep for MPS is §1, "The BUG
# algorithm": Eq. (6) the tensor ODE `d/dt T^[c] = -i H_eff^[c] T^[c]`, Eq. (7) `H_eff` as one
# contraction, Eq. (8) the rank-3 environment recursion, Eq. (9) the augmentation projector,
# root `r = floor(L/2)`, and **Remark 1** the frozen-environment assumption. §2 is the HYBRID
# quantum/classical layer and contains no MPS sweep. Two deliberate departures from §1, both
# requested and both recorded here rather than left to be discovered:
#
#   * ENVIRONMENTS ARE NOT FROZEN. Remark 1 freezes them so every node update is independent,
#     which is what buys the paper its parallelism. This sweep is sequential, so the left
#     environment is pushed through the basis just written before the next bond is touched.
#     There is nothing to gain from freezing in a serial code and there is accuracy to lose.
#   * FIXED RANK, i.e. `augment = false`. Eq. (9) augments to `orth([K(tau) | A_i])`, which
#     guarantees the new basis CONTAINS the old one. Here the basis is `orth(K(tau))` alone, and
#     the rank comes ONLY from the CBE expansion in step 2. That is the standing constraint that
#     `cbe_rsvd` IS the method -- rank growth is the expansion's job and nothing else's.
#
# THE STEP, at root bond `c = floor(L/2)`:
#
#   1. LEFT -> ROOT, bonds 1 … c. At bond `i`:
#        a. RSVD-CBE expansion of bond `i`. Lossless, so the state is untouched; it only gives
#           the K-step below a wider bond to move into.
#        b. the BUG K-STEP: evolve the one-site object `K = C·A_i`, widened onto the expanded
#           bond, under `H^{1s}_i` for the FULL step `tau`.
#        c. `W_i = orth(K(tau))`, **R discarded**. The basis moves where the dynamics points and
#           NO evolution accumulates.
#        d. carry `C <- W_i† K(0)` -- the ORIGINAL amplitude re-expressed in the new basis.
#           That is what "do not dynamically evolve the basis" means: `K(tau)` chooses the
#           subspace, `K(0)` supplies what travels on.
#   2. RIGHT -> ROOT, bonds L-1 … c, the mirror, giving `Z_c … Z_{L-1}`.
#   3. ONE Galerkin update at the root bond: seed `S0 = <W,Z|psi(t0)>` by full contraction, then
#      one 0-site `exp(tau H_eff)`. THE ONLY EVOLUTION THAT IS KEPT, hence no over-counting and
#      no backward substep.
#   4. one truncation round trip (Sulz Algorithm 1).
#
# WHY THE K-STEP IS EVOLVED FOR THE FULL `tau` AND NOT `tau/2`. BUG is not a splitting. The
# K-step's amplitude is thrown away, so integrating it over the whole step cannot double-count
# anything -- it is a basis-finding calculation whose only output is a subspace, and the
# subspace wanted is the one the state occupies at the END of the step.
#
# WHY THE CARRY USES THE UNWIDENED `K(0)`. `W_i` involves only the `(b_{i-1}, s_i)` legs, so
# contracting it against the unwidened `K(0)` gives `(b_i, link_i)` -- a bond leg the NEXT site
# tensor can be contracted with. Carrying the widened object instead leaves the expanded bond
# dangling on the wrong side and the chain does not compose.
#
# WHERE THIS SITS AGAINST THE OTHER TWO SWEEPS. All three are BUG -- augment the basis, then ONE
# Galerkin solve, no projector splitting and no backward substep -- and each differs from this
# one along exactly ONE axis, which is what makes a comparison between them attributable:
#
#                     basis update at a bond          root         evolutions kept
#   cbe_lubich        CBE expansion ALONE             centre bond  1
#   jan_bug           CBE + a 0-site S step, QR'd     end bond     1
#   THIS              CBE + the BUG 1-site K step     centre bond  1
#
# `kstep = false` turns step 1b/1c off and takes `W_i = U_ex` instead, which reproduces
# `cbe_lubich_sweep`. It exists so the K-step's contribution can be measured with everything
# else held fixed, rather than argued about.
#
# AT FULL RANK THIS IS STILL EXACT, which is worth stating because a fixed-rank BUG sounds like
# it should not be. If the expanded bond reaches the whole local space then `orth(K(tau))` spans
# it whatever `K(tau)` is, so the basis is complete, the seed is lossless and the root solve is
# `exp(tau H)` on the full space. The `augment = false` approximation only bites once the
# expansion is capped -- which is the regime the method is for, and where the O(dt^2) local
# error of the BUG basis update is the thing being paid.

"""
    CBEBugSweepInfo

What one [`cbe_bug_step!`](@ref) did.

  - `expanded` -- the bond dimension the basis update left at each bond, before the closing
    truncation. In MULTIPLETS under a non-abelian symmetry.
  - `n_new` -- directions CBE admitted per bond; an upper bound on what `expanded` could grow by.
  - `root`, `root_rank` -- which bond took the Galerkin step, and its kept rank afterwards.
  - `err_pre`, `err_fnl` -- largest preselection / final-selection weight discarded. `err_fnl`
    is the one to watch: a large value means the BUDGET, not the selection, limited accuracy.
  - `discarded` -- largest relative weight the closing truncation threw away.
  - `krylov_dims` -- operator applications, summed over the `L` K-steps and the one root solve.
  - `peak_elements` -- working set (`state + transients alive at that moment`), the same
    definition `CBEBugInfo` and `TDVP2Info` use, so the three are comparable. The transients
    here are one expanded frame, one one-site object and one bond tensor -- **never a rank-4
    block**, which is the memory claim this sweep inherits from the 0-site root.
"""
struct CBEBugSweepInfo
    expanded::Vector{Int}
    n_new::Vector{Int}
    root::Int
    root_rank::Int
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    peak_elements::Int
end

"""
    cbe_bug_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

One RSVD-CBE + fixed-rank-BUG step of `tau`, in place. `tau = -im*dt` is real time and
`tau = -dt` imaginary time; nothing here cares which, and the caller renormalises.

Keywords beyond the [`cbe_expand`](@ref) ones (`dex`, `growth`, `dover`, `comp_ratio`,
`sulz_cap`, `preselect_only`, `exact`):

  - `root` -- the bond that keeps the Galerkin step. `nothing` is `floor(L/2)`, the paper's
    root, and it is not a free parameter in practice: the root solve explores exactly
    `b_L x b_R` directions, and at the CENTRE both factors are a full half-chain while at bond
    `L-1` the right factor is a SINGLE SITE. Measured on the sibling sweep
    (`benchmarks/root_position.jl`, L=6, full rank, one step at dt=0.01): centre `8.5e-16`,
    off-centre interior `1.2e-5`, boundary `1.7e-5` and unreachable at any budget, because
    there the ceiling `8 x 2 = 16` sits inside a 20-dimensional sector. Kept as a keyword so
    the root position can be varied as a controlled experiment with everything else fixed.
  - `kstep` -- `true` (default) does the BUG basis update; `false` takes the CBE frame directly
    and reproduces `cbe_lubich_sweep`. The A/B that isolates the K-step's contribution.
  - `maxdim`, `trunc_thresh` -- the closing truncation. `truncate = false` disables it, to watch
    the rank ratchet.
  - `maxiter`, `tol` -- Krylov budget for the K-steps and the root solve.
"""
function cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                       dex::Int = 0,
                       growth::Float64 = 2.0,
                       dover::Union{Nothing, Int} = nothing,
                       comp_ratio::Union{Float64, Nothing} = 0.5,
                       sulz_cap::Bool = false,
                       preselect_only::Bool = false,
                       exact::Bool = false,
                       root::Union{Nothing, Int} = nothing,
                       kstep::Bool = true,
                       truncate::Bool = true,
                       maxdim::Int = 200,
                       trunc_thresh::Float64 = 1e-12,
                       maxiter::Int = 30,
                       tol::Float64 = 1e-15,
                       reorth::Bool = true,
                       rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("cbe_bug_step! needs at least two sites"))
    c = root === nothing ? max(1, min(L - 1, L ÷ 2)) : root
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))

    cut = max(trunc_thresh, 1e-14)
    expanded = zeros(Int, L - 1)
    n_new    = zeros(Int, L - 1)
    epre = 0.0; efnl = 0.0; peak = 0
    nmv = Ref(0)

    # `rmax` read ONCE, globally, before anything widens.
    exkw = (dex = dex, growth = growth, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact, rng = rng)

    function record!(ex, i, side::Symbol)
        epre = max(epre, ex.err_pre)
        efnl = max(efnl, ex.err_fnl)
        n_new[i] = side === :left ? ex.n_new_l : ex.n_new_r
        return ex
    end
    "Working set at this instant: the state as it stands plus the transients still alive."
    peak!(ts...) = (peak = max(peak, _state_stored(psi) +
                               sum(tensor_elements(t) for t in ts; init = 0)))

    W = Vector{Any}(undef, c)          # augmented LEFT frames,  sites 1 … c
    Z = Vector{Any}(undef, L - 1)      # augmented RIGHT frames, Z[j] absorbs site j+1

    # ── 1. LEFT -> ROOT ──────────────────────────────────────────────────────────────────
    #
    # `psi` canonical at 1, so sites 2…L are right isometries and the left block's amplitude
    # rides in the carry `C`. The right stack is prebuilt once and read at link `i+2`. It cannot
    # go stale, and for a stronger reason than in the other sweeps: THIS PASS NEVER WRITES TO
    # THE STATE AT ALL -- frames are accumulated in `W`/`Z` and `psi` is assembled only after
    # the root solve. (Writing each frame back as it is found is the failure `cbe_lubich.jl`'s
    # header records: it leaves a zero-padded remainder in `psi[i+1]` which the next bond's
    # `_frame_from` then SVDs away again, and L=18 froze at maxbond 2.)
    canonical!(psi, 1)
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    lenv_root = lch                    # left environment at link c, in the NEW basis
    C = nothing
    for i in 1:c
        K = C === nothing ? psi[i] : to_concrete(contract(C, (2,), psi[i], (1,)))
        rch = right_channels(rstack, i + 2)
        f  = _frame_from(K, psi[i + 1])
        ex = record!(cbe_expand(f, mpo, i, lch, rch; exkw...), i, :left)
        peak!(ex.U_ex, ex.V_ex)

        if kstep
            # (a) widen `K`'s bond leg onto the expanded bond, LOSSLESSLY: `psi[i+1] V_ex†` is
            # the change of basis, and `range(V_ex) ⊇ range(psi[i+1])` because `oplus` puts the
            # old frame first. This is where the extra room for the K-step comes from, and it
            # is the ONLY thing the expansion does here.
            M   = to_concrete(contract(psi[i + 1], (2, 3), ex.V_ex', (2, 3)))   # (link_i, b_R)
            Kex = to_concrete(contract(K, (3,), M, (1,)))          # (b_{i-1}, s_i, b_R)
            # (b) the right environment must be the WIDENED one -- one push through `V_ex`,
            # not a rebuild.
            rch1 = push_right_channels(rch, mpo, ex.V_ex, i + 1)
            H1 = one_site_h(mpo, i, lch, rch1)
            peak!(Kex)
            K1 = expv(x -> (nmv[] += 1; apply_one_site(H1, x)), tau, Kex;
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)
            # (c) basis only -- `R` discarded. Telum has no QR, so `svd(...).U` is the
            # orthonormaliser, with a numerical-zero cutoff: rank control belongs to the
            # closing truncation, not to this factorisation.
            W[i] = to_concrete(svd(K1, (1, 2); cutoff = 1e-14).U)
        else
            W[i] = ex.U_ex                                          # reproduces cbe_lubich
        end
        expanded[i] = leg_dim(W[i], 3)

        # (d) carry the ORIGINAL amplitude into the new basis. `K`, not `Kex`: `W[i]` touches
        # only legs 1-2, so this lands on `(b_i, link_i)` and composes with `psi[i+1]`.
        C = to_concrete(contract(W[i]', (1, 2), K, (1, 2)))
        i == c && (lenv_root = lch)     # captured BEFORE pushing through the root frame
        i < c && (lch = push_left_channels(lch, mpo, W[i], i))
    end

    # ── 2. RIGHT -> ROOT ─────────────────────────────────────────────────────────────────
    # The mirror, and it needs the opposite gauge: canonical at `L` makes sites 1…L-1 left
    # isometries, so the right block's amplitude rides in `D`. Re-gauging does not change the
    # state and `W` stays valid -- `W_1 … W_c` names a SUBSPACE of the left half's Hilbert
    # space, which no gauge move touches, and the seed below is a full contraction.
    canonical!(psi, L)
    lstack = left_env_stack(psi, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    renv_root = rch                    # right environment at link c+2, in the NEW basis
    D = nothing
    for j in (L - 1):-1:c
        B = D === nothing ? psi[j + 1] :
            to_concrete(contract(psi[j + 1], (3,), D, (1,)))
        lch_j = left_channels(lstack, j)
        f  = _frame_from(psi[j], B)
        ex = record!(cbe_expand(f, mpo, j, lch_j, rch; exkw...), j, :right)
        peak!(ex.U_ex, ex.V_ex)

        if kstep
            M   = to_concrete(contract(ex.U_ex', (1, 2), psi[j], (1, 2)))       # (b_L, link_j)
            Bex = to_concrete(contract(M, (2,), B, (1,)))           # (b_L, s_{j+1}, b_{j+1})
            lch1 = push_left_channels(lch_j, mpo, ex.U_ex, j)
            H1 = one_site_h(mpo, j + 1, lch1, rch)
            peak!(Bex)
            B1 = expv(x -> (nmv[] += 1; apply_one_site(H1, x)), tau, Bex;
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)
            Z[j] = to_concrete(svd(B1, (1,); cutoff = 1e-14).Vd)
        else
            Z[j] = ex.V_ex
        end
        j != c && (expanded[j] = leg_dim(Z[j], 1))

        D = to_concrete(contract(B, (2, 3), Z[j]', (2, 3)))          # (link_j, b_j)
        j == c && (renv_root = rch)
        j > c && (rch = push_right_channels(rch, mpo, Z[j], j + 1))
    end

    # ── 3. THE SEED AND THE SINGLE GALERKIN STEP ─────────────────────────────────────────
    # `S0 = <W, Z | psi>` by full contraction from each end. No overlap matrices and no gauge
    # requirement on `psi`: this is a complete contraction of the state against the frames, so
    # the re-gauge between the two passes cannot have disturbed it.
    AL = nothing
    for i in 1:c
        T = AL === nothing ? psi[i] : to_concrete(contract(AL, (2,), psi[i], (1,)))
        AL = to_concrete(contract(W[i]', (1, 2), T, (1, 2)))         # (b_L, link_{i+1})
    end
    AR = nothing
    for j in (L - 1):-1:c
        T = AR === nothing ? psi[j + 1] : to_concrete(contract(psi[j + 1], (3,), AR, (1,)))
        AR = to_concrete(contract(T, (2, 3), Z[j]', (2, 3)))         # (link_{j+1}, b_R)
    end
    S0 = to_concrete(contract(AL, (2,), AR, (1,)))                   # (b_L, b_R)
    peak!(W[c], Z[c], S0)

    H0 = zero_site_h(mpo, c, lenv_root, renv_root, W[c], Z[c])
    S1 = expv(x -> (nmv[] += 1; apply_zero_site(H0, x)), tau, S0;
              hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

    res = svd(S1, (1,); cutoff = cut, Nkeep = maxdim, get_lists = true)
    discarded = _trunc_weight(res)
    expanded[c] = leg_dim(W[c], 3)

    # ── 4. ASSEMBLE AND TRUNCATE ─────────────────────────────────────────────────────────
    # BOTH bond legs of every frame are retagged, not one: adjacent tensors in a `SymMPS` must
    # agree on the shared link tag, and retagging one side only leaves
    # `psi[i].inds[3] != psi[i+1].inds[1]`, which `contract` rejects on the next sweep.
    for i in 1:(c - 1)
        psi[i] = _relink(W[i], psi[i].inds[1].itags, psi[i].inds[3].itags)
    end
    t1c, t3c = psi[c].inds[1].itags, psi[c].inds[3].itags
    t3r = psi[c + 1].inds[3].itags
    psi[c]     = _relink(to_concrete(W[c] * res.U), t1c, t3c)
    psi[c + 1] = _relink(to_concrete((res.S * res.Vd) * Z[c]), t3c, t3r)
    for j in (c + 1):(L - 1)
        psi[j + 1] = _relink(Z[j], psi[j + 1].inds[1].itags, psi[j + 1].inds[3].itags)
    end
    psi.center = c + 1
    root_rank = leg_dim(psi[c], 3)

    # Compress ONCE, now that the evolved root core has propagated out through every retained
    # frame -- so a direction carrying no weight is genuinely unwanted rather than merely
    # not-yet-populated. Run before that, it would delete the expansion it is meant to control.
    truncate && truncate_sweep!(psi; maxdim = maxdim, cutoff = cut)

    return CBEBugSweepInfo(expanded, n_new, c, root_rank, epre, efnl,
                           discarded, nmv[], peak)
end

cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_step!(psi, mpo_from_terms(h), tau; kwargs...)
