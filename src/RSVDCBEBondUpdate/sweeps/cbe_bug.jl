
_tagW(i::Int) = "cbeW,$i"
_tagZ(j::Int) = "cbeZ,$j"

"""
    _krylov_frame(v0, applyH, maxdepth, tol) -> Vector

The ORTHONORMAL Krylov vectors `{v0, v1, …}` of `applyH` seeded at `v0`, stopped when the
residual `beta` falls below `tol` (the Krylov space has closed) or `maxdepth` is reached.

⛔ NO SVD PER ITERATION. The vectors are accumulated whole and split ONCE by the caller. A
selection inside the loop -- which is what `cbe_lubich_sweep`'s `grow_iters` does, re-running the
CBE ranking on every pass -- re-truncates by budget at each step and throws away exactly the
directions the next application of `H` needs: MEASURED on OAT it bought 1.8x and then saturated
flat (2.13e-02 → 1.16e-02) no matter how many passes were allowed.

⛔ FULL RE-ORTHOGONALISATION IS NOT OPTIONAL, and this is where a plain power iteration fails.
`H^k v` converges to the dominant eigenvector, so by `k ≈ 4` the raw powers are numerically
parallel and the closing SVD deletes them as dependent -- the basis would saturate for a
CONDITIONING reason that looks exactly like a physical one. Two passes, against the whole basis:
one pass of modified Gram-Schmidt loses orthogonality once the vectors start to align.

`beta` here is the norm of the residual after orthogonalisation, i.e. the same quantity
`lanczos_expv` exits on -- so `tol` is a Krylov tolerance in the ordinary sense and pairs with
`trunc_thresh` the way `tau_Krylov = tau_trunc/10` does.
"""
function _krylov_frame(v0, applyH, maxdepth::Int, tol::Float64)
    b0 = norm(v0)
    b0 <= 0 && return Any[v0]
    vs = Any[to_concrete((1.0 / b0) * v0)]
    for _ in 1:maxdepth
        w = applyH(vs[end])
        for _pass in 1:2, u in vs
            w = to_concrete(w - tensor_inner(u, w) * u)
        end
        b = norm(w)
        b < tol && break
        push!(vs, to_concrete((1.0 / b) * w))
    end
    # The seed carries the state's own scale back in: the caller's SVD cutoff is relative to the
    # largest singular value, and a basis of unit vectors alone would let the closing truncation
    # rank the high-order directions against each other rather than against the state.
    vs[1] = to_concrete(b0 * vs[1])
    return vs
end

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
  - `peak_elements` -- working set (`state + accumulated frames + transients alive at that
    moment`), the same definition `TDVP2Info` uses, so the two are comparable.
    ⚠ **This sweep does NOT have a lower working set than 2-site TDVP.** It never forms a rank-4
    block, which is true and is why its transients are small -- but it holds O(L) FRAMES alive
    until the root solve, and at L=16, chi=32 that measured *larger* than `tdvp2_step!`'s
    `state + one Theta`. The advantage this sweep actually has is in OPERATOR APPLICATIONS and
    wall clock (measured 4.2x fewer applications and 1.35x faster than 2-site TDVP at L=16), not
    in memory.
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

ONE step of the CBE-BUG sweep, in place. `tau = -im*dt` is real time and `tau = -dt` imaginary
time; nothing here cares which, and the caller renormalises.

THERE IS ONE SWEEP. Both half-sweeps are BASIS GENERATORS -- CBE widens each bond, the frame is
built from that bond's Krylov space, the environment is pushed through it, and no site tensor is
ever evolved. The single Galerkin solve at the root bond is the only evolution in the step.

  left -> root   `U_ex` becomes `W[i]`;  right -> root   `V_ex` becomes `Z[j]`

so every bond is widened from BOTH sides before the root sees it.

⛔ THE K-STEP, `kaug` AND `rexpand` ARE GONE (2026-08-24). They evolved a one-site tensor per bond
and then restored the width that the split cost. MEASURED at L=12, dt=0.05, against exact
references, this sweep at `krylov_basis = 3` beats that machinery on every model tested:

    model        this sweep              old K-step sweep        
    XX           1.6948e-04 @   289      1.6948e-04 @  2888   identical, 10x cheaper (m=0)
    Heisenberg   2.6697e-06 @  1038      2.7869e-06 @  3423   better, 3.3x cheaper
    OAT          7.0610e-14 @  1036      5.2449e-06 @  3740   8 orders better, 3.6x cheaper

`git log` holds the old path; `docs/PLAN-tolerance-control.md` holds the measurements that
retired it, including the three hypotheses that were wrong on the way.

Keywords beyond the [`cbe_expand`](@ref) ones (`dex`, `growth`, `dover`, `comp_ratio`,
`sulz_cap`, `preselect_only`, `exact`, `stol_pre`, `stol_fnl`):

  - `krylov_tol` -- **THE DEFAULT STOPPING RULE, `1e-6`.** The Krylov iteration stops when the
    Lanczos residual `beta` falls below it. `0.0` falls back to `tau_trunc/10`.
  - `krylov_basis` -- the CAP on that iteration, default `30`.

    ⚠ **THE CAP OFTEN IS THE DEPTH, because `beta` does not decay on a generic `H`.** MEASURED at
    L=12: Heisenberg and XX give BIT-IDENTICAL results for `krylov_tol` anywhere from `1e-6` to
    `1e-13` -- the criterion never fires and every bond runs to the cap. Cost follows: Heisenberg
    9.61e-07 @ 5580 applications against a fixed depth of 3's 2.67e-06 @ 1038. OAT is the one
    model where it binds (4104 @ `1e-6` against 5983 @ `1e-13`).

    `beta` is a BREAKDOWN detector, not a convergence measure; the sharper criterion is the
    contribution `beta_m*|coeff_m|` that `exact_sparse.jl::_lanczos_expv` computes, and swapping
    to it is what would make the depth genuinely adaptive. Set `krylov_basis` to a small fixed
    number (2-3) to pin the depth instead.

    ⛔ **NOT CONSIDERING ENOUGH KRYLOV VECTORS IS A REAL AND EASILY-MISSED SOURCE OF ERROR.**
    `krylov_basis = 0` uses `cbe_expand`'s frame alone -- ONE power of `H` -- and that is enough
    on XX and badly short on Heisenberg and OAT. The failure does NOT look like a basis problem:
    it survives more budget, more rank, and turning the closing truncation off entirely, because
    the basis is short of the right DIRECTIONS rather than of room. `m = 1` reproduces `m = 0`
    exactly (CBE already spans one power), so the first genuine addition is `m = 2`.

    ⛔ ONE SVD, AT THE END -- see [`_krylov_frame`](@ref). Selecting inside the loop (which is what
    `cbe_lubich_sweep`'s `grow_iters` does) re-truncates by budget every pass and discards exactly
    what the next application of `H` needs: measured 1.8x and then flat on OAT.
  - `root` -- which bond takes the Galerkin step. Default `L/2`, a fixed point of the reflection
    `b -> L - b` on an even chain, so the sweep is already spatially symmetric there.
  - `split_cutoff`, `split_maxdim` -- truncation on the frame split, applied once per bond as the
    sweep passes.
  - `root_cutoff`, `root_maxdim` -- truncation on the ROOT CORE SPLIT. **Defaults `0.0`/`0`, i.e.
    NONE**: that SVD splits the Galerkin core into the two centre tensors and nothing more.
    There are exactly TWO truncations in a step -- the half-sweep split and the closing pass --
    and this used to be an unswitchable third.
  - `maxdim`, `trunc_thresh` -- the closing [`truncate_recursive!`](@ref). `truncate = false`
    disables it; with the root defaults above that leaves `split_cutoff` as the only rank control
    anywhere in the step, which is a study configuration and not a production one.
  - `maxiter`, `tol`, `reorth` -- Krylov budget for the root solve.
"""
function cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                       dex::Int = 0,
                       growth::Float64 = 2.0,
                       dover::Union{Nothing, Int} = nothing,
                       comp_ratio::Union{Float64, Nothing} = 1.0,
                       sulz_cap::Bool = false,
                       preselect_only::Bool = false,
                       exact::Bool = false,
                       stol_pre::Float64 = 1e-10,
                       stol_fnl::Float64 = 1e-13,
                       root::Union{Nothing, Int} = nothing,
                       krylov_basis::Int = 30,
                       krylov_tol::Float64 = 1e-6,
                       split_cutoff::Float64 = 1e-14,
                       split_maxdim::Int = 0,
                       truncate::Bool = true,
                       root_cutoff::Float64 = 0.0,
                       root_maxdim::Int = 0,
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
    # `krylov_tol = 0` falls back to `tau_trunc/10`; the DEFAULT is an explicit `1e-6`.
    ktol = krylov_tol > 0 ? krylov_tol : cut / 10

    # TRUNCATION AT THE HALF-SWEEP SPLIT. Historically this was a hard `cutoff = 1e-14` with no
    # `Nkeep` at all, so the half-sweeps pruned essentially nothing and every rank decision was
    # deferred to the root solve and `truncate_recursive!`. `tdvp_cbe1s` instead splits with
    # `cutoff = cut, Nkeep = maxdim` at every site. These two closures make that difference a
    # knob rather than a hard-coded constant; the defaults reproduce the historical behaviour
    # exactly, so `split_cutoff = 1e-14, split_maxdim = 0` is a no-op.
    _splitU(T) = split_maxdim > 0 ?
        to_concrete(svd(T, (1, 2); cutoff = split_cutoff, Nkeep = split_maxdim).U) :
        to_concrete(svd(T, (1, 2); cutoff = split_cutoff).U)
    _splitV(T) = split_maxdim > 0 ?
        to_concrete(svd(T, (1,); cutoff = split_cutoff, Nkeep = split_maxdim).Vd) :
        to_concrete(svd(T, (1,); cutoff = split_cutoff).Vd)

    expanded = zeros(Int, L - 1)
    n_new    = zeros(Int, L - 1)
    epre = 0.0; efnl = 0.0; peak = 0
    nmv = Ref(0)

    # `rmax` read ONCE, globally, before anything widens.
    # `stol_pre` / `stol_fnl` are THREADED THROUGH rather than left to `cbe_expand`'s defaults.
    # They are the only truncations that exist inside the half-sweep besides the frame split -- there
    # is no `_splitU` on that path at all -- so leaving them at a hardcoded 1e-10 / 1e-13 made the
    # basis-only sweep ungovernable by any tolerance the caller sets, and made `split_cutoff` look
    # like the half-sweep's knob when for that sweep it is not used at all.
    exkw = (dex = dex, growth = growth, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact,
            stol_pre = stol_pre, stol_fnl = stol_fnl, rng = rng)

    function record!(ex, i, side::Symbol)
        epre = max(epre, ex.err_pre)
        efnl = max(efnl, ex.err_fnl)
        n_new[i] = side === :left ? ex.n_new_l : ex.n_new_r
        return ex
    end

    _frames_stored() = sum(sum(tensor_elements(v[k]) for k in eachindex(v)
                               if isassigned(v, k); init = 0)
                           for v in (W, Z, Vwide, Uwide, Wcbe, Zcbe); init = 0)
    peak!(ts...) = (peak = max(peak, _state_stored(psi) + _frames_stored() +
                               sum(tensor_elements(t) for t in ts; init = 0)))

    W = Vector{Any}(undef, c)          # augmented LEFT frames,  sites 1 … c
    Z = Vector{Any}(undef, L - 1)      # augmented RIGHT frames, Z[j] absorbs site j+1

    Vwide = Vector{Any}(undef, c)                # widened RIGHT frame, bonds 1 … c
    Uwide = Vector{Any}(undef, L - 1)        # widened LEFT  frame, bonds c … L-1
    # The CBE frame. It IS the basis at `krylov_basis = 0`, and the seed the Krylov space grows
    # into the basis) -- so it is stored whenever the basis is going to reference it.
    # THE SWEEP IS INTERLEAVED: CBE expands bond `i`, then bond `i`'s frame is built and the
    # environment pushed through it immediately -- the reference's own ordering
    # (TDVPSweepCBE1Si.m:30/:34), so the environment chain runs through the NEW frames and bond
    # `i+1`'s selection sees the widening at bond `i`.
    Wcbe  = Vector{Any}(undef, c)
    Zcbe  = Vector{Any}(undef, L - 1)

    """
    Union `Wk` (the basis update's output) with `U0` (the CBE frame), on the LEFT.

    `U0` FIRST so the CBE directions are kept verbatim and only the part of `Wk` outside them is
    appended -- `_preselect_left` returns `nothing` when there is nothing to append, which is the
    ordinary case once the two already span the same space.
    """
    function _union_left(U0, Wk)
        Q, _ = _preselect_left(U0, Wk, 1e-14)
        Q === nothing && return U0
        return to_concrete(oplus([U0, to_concrete(setitag(Q, 3, U0.inds[3].itags))], (3,)))
    end
    "Row mirror of [`_union_left`](@ref): the bond leg is 1 and the direct sum is over it."
    function _union_right(V0, Zk)
        Q, _ = _preselect_right(V0, Zk, 1e-14)
        Q === nothing && return V0
        return to_concrete(oplus([V0, to_concrete(setitag(Q, 1, V0.inds[1].itags))], (1,)))
    end

    canonical!(psi, 1)
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    C = nothing

    # ── PHASE B(a). LEFT -> ROOT. THE EVOLUTION, IDENTICAL UNDER BOTH SCHEMES ─────────────
    # Only WHERE `Vwide[i]` comes from differs, which is what makes a measured difference
    # between the two schemes attributable to the decoupling and to nothing else.
    lch = boundary_channels(mpo)
    lenv_root = lch                    # left environment at link c, in the NEW basis
    C = nothing
    for i in 1:c
        K = C === nothing ? psi[i] : to_concrete(contract(C, (2,), psi[i], (1,)))
        ex = record!(cbe_expand(_frame_from(K, psi[i + 1]), mpo, i, lch,
                                right_channels(rstack, i + 2); exkw...), i, :left)
        peak!(ex.U_ex, ex.V_ex)
        Vwide[i] = ex.V_ex
        Wcbe[i]  = ex.U_ex
        if krylov_basis > 0
            # ── BASIS FROM THE KRYLOV SPACE, NOT FROM ONE APPLICATION OF `H` ───────────────
            #
            # `krylov_basis = 0` builds the frame from `cbe_expand` alone, i.e. from
            # `span{Theta, H*Theta}` -- ONE power of `H`. That is enough wherever `tau*||H_loc||`
            # is small (XX and Heisenberg: measured bit-identical to the shipped sweep on XX),
            # and badly short where it is not.
            #
            # Here the frame is instead `orth span{Kex, H1*Kex, ..., H1^m*Kex}`: real
            # applications of the one-site effective Hamiltonian, so the basis carries `m` Krylov
            # vectors rather than one. `m = 0` is the bare CBE frame; `m = 1` reproduces it exactly.
            #
            # The vectors are NORMALISED before the direct sum. Without that, `H1^k*Kex` grows
            # like `||H1||^k` and the SVD's relative `cutoff` -- which is measured against the
            # LARGEST singular value -- deletes the low-order vectors that carry the state
            # itself. `Kex` is left unnormalised so it keeps the state's own scale.
            M   = to_concrete(contract(psi[i + 1], (2, 3), ex.V_ex', (2, 3)))
            Kex = to_concrete(contract(K, (3,), M, (1,)))
            rch1 = push_right_channels(right_channels(rstack, i + 2), mpo, ex.V_ex, i + 1)
            H1 = one_site_h(mpo, i, lch, rch1)
            blocks = _krylov_frame(Kex, x -> (nmv[] += 1; apply_one_site(H1, x)),
                                   krylov_basis, ktol)
            peak!(blocks...)
            stacked = length(blocks) == 1 ? blocks[1] : to_concrete(oplus(blocks, (3,)))
            W[i] = to_concrete(setitag(_splitU(stacked), 3, _tagW(i)))
        else
            W[i] = Wcbe[i]                                          # reproduces cbe_lubich
        end
        expanded[i] = leg_dim(W[i], 3)

        # (d) carry the ORIGINAL amplitude into the new basis. `K`, not `Kex`: `W[i]` touches
        # only legs 1-2, so this lands on `(b_i, link_i)` and composes with `psi[i+1]`.
        C = to_concrete(contract(W[i]', (1, 2), K, (1, 2)))
        i == c && (lenv_root = lch)     # captured BEFORE pushing through the root frame
        i < c && (lch = push_left_channels(lch, mpo, W[i], i))
    end


    canonical!(psi, L)
    lstack = left_env_stack(psi, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    D = nothing

    # ── PHASE B(b). RIGHT -> ROOT. THE EVOLUTION ─────────────────────────────────────────
    rch = boundary_channels(mpo)
    renv_root = rch                    # right environment at link c+2, in the NEW basis
    D = nothing
    for j in (L - 1):-1:c
        B = D === nothing ? psi[j + 1] :
            to_concrete(contract(psi[j + 1], (3,), D, (1,)))
        ex = record!(cbe_expand(_frame_from(psi[j], B), mpo, j,
                                left_channels(lstack, j), rch; exkw...), j, :right)
        peak!(ex.U_ex, ex.V_ex)
        Uwide[j] = ex.U_ex
        Zcbe[j]  = ex.V_ex
        if krylov_basis > 0
            # Row mirror of the K half-sweep's Krylov basis; see the comment there.
            M   = to_concrete(contract(ex.U_ex', (1, 2), psi[j], (1, 2)))     # (b_L, link_j)
            Bex = to_concrete(contract(M, (2,), B, (1,)))
            lch1 = push_left_channels(left_channels(lstack, j), mpo, ex.U_ex, j)
            H1 = one_site_h(mpo, j + 1, lch1, rch)
            blocks = _krylov_frame(Bex, x -> (nmv[] += 1; apply_one_site(H1, x)),
                                   krylov_basis, ktol)
            peak!(blocks...)
            stacked = length(blocks) == 1 ? blocks[1] : to_concrete(oplus(blocks, (1,)))
            Z[j] = to_concrete(setitag(_splitV(stacked), 1, _tagZ(j)))
        else
            Z[j] = Zcbe[j]
        end
        j != c && (expanded[j] = leg_dim(Z[j], 1))

        D = to_concrete(contract(B, (2, 3), Z[j]', (2, 3)))          # (link_j, b_j)
        j == c && (renv_root = rch)
        j > c && (rch = push_right_channels(rch, mpo, Z[j], j + 1))
    end

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

    # ⛔ THE ROOT CORE SPLIT DOES NOT TRUNCATE BY DEFAULT. This SVD exists to SPLIT `S1` into
    # the two site tensors below; cutting here was a third, redundant truncation on top of the
    # half-sweep splits and the closing `truncate_recursive!`, and it was the only one with no
    # switch. Rank control now lives in exactly two places -- `split_cutoff` on the half-sweeps
    # and `trunc_thresh`/`maxdim` on the closing pass -- so a tolerance statement about the step
    # has two knobs to account for instead of three.
    #
    # `root_cutoff = 0.0` keeps every singular value with `sv > 0` (Telum's selector is relative
    # to `sv_max`), i.e. the split is lossless up to exact zeros; `root_maxdim = 0` means no
    # `Nkeep`. Both are still settable, so the old behaviour is an A/B (`root_cutoff = cut`,
    # `root_maxdim = maxdim`) rather than something that has been deleted.
    #
    # ⚠ WITH `truncate = false` AND THESE DEFAULTS THERE IS NO RANK CONTROL AT THE ROOT AT ALL,
    # which is deliberate -- it is what makes "root truncation off" a reachable arm -- but it
    # means the closing pass is the only thing enforcing `maxdim` in the shipped configuration.
    res = svd(S1, (1,); cutoff = root_cutoff,
              Nkeep = root_maxdim > 0 ? root_maxdim : nothing, get_lists = true)
    discarded = _trunc_weight(res)
    expanded[c] = leg_dim(W[c], 3)

    # ── 4. ASSEMBLE AND TRUNCATE ─────────────────────────────────────────────────────────

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

    if truncate
        _, wtr = truncate_recursive!(psi, c + 1; maxdim = maxdim, cutoff = cut)
        discarded = max(discarded, wtr)
    end

    return CBEBugSweepInfo(expanded, n_new, c, root_rank, epre, efnl,
                           discarded, nmv[], peak)
end

cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_step!(psi, mpo_from_terms(h), tau; kwargs...)
