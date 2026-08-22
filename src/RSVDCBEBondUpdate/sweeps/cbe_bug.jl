
_tagW(i::Int) = "cbeW,$i"
_tagZ(j::Int) = "cbeZ,$j"

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
  - `kaug` -- whether the frame union ran. Recorded so a result can be attributed from the info
    struct alone, without reference to the call site.
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
    kaug::Bool
end

"""
    cbe_bug_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

One RSVD-CBE + fixed-rank-BUG step of `tau`, in place. `tau = -im*dt` is real time and
`tau = -dt` imaginary time; nothing here cares which, and the caller renormalises.

Keywords beyond the [`cbe_expand`](@ref) ones (`dex`, `growth`, `dover`, `comp_ratio`,
`sulz_cap`, `preselect_only`, `exact`):

  - `growth` -- `1.1` HERE, against `2.0` in both [`cbe_expand`](@ref) and `CBEBugOptions`.
    The divergence is deliberate, and it is worth being explicit that this sweep ships the
    REFERENCE'S DMRG RATCHET -- the constant the budget block in `cbe_core.jl` argues against
    for a time integrator, on the grounds that a direction the step needs and does not admit
    is not deferred, it is gone.

    IT STANDS HERE BECAUSE CBE IS NOT THE ONLY SOURCE OF NEW DIRECTIONS IN THIS SWEEP. The
    K-step supplies its own, so the expansion has a narrower job than in `tdvp_cbe1s`, where
    CBE is the sole source. Two sweeps wanting different constants is that difference, not an
    oversight.

    MEASURED (`benchmarks/rsvd_param_scan.jl`, XX L=16, maxdim=24, t=6): the final error moves
    by 5.0e-7 RELATIVE across the whole `dex`/`growth` grid -- 4.59041166e-05 to
    4.59041397e-05 -- while `err_fnl`, the fraction of ranked candidate weight DISCARDED for
    want of budget, spans 0.0 to 0.59. Raising `growth` buys nothing on that model and costs
    sketch work.

    THAT NUMBER IS RESOLVED, NOT ROUNDING. Phase 1 measures the seed noise floor on the same
    arm: `cbe_bug` returns 4.5904e-05 over six seeds with a spread of 1.807e-13, i.e. 3.9e-9
    relative, so the budget effect sits 128x above the floor. It is real and it is negligible.
    For scale, `tdvp_cbe1s` on the identical grid moves 6.1e-3 against its own floor of 9.9e-7
    -- a budget response four orders of magnitude larger than this sweep's.

    ⛔ EVERY NUMBER ABOVE IS FROM A CAP-BOUND RUN, which is the regime where the budget CANNOT
    matter: `maxdim` throws away what the expansion admits. The argument for `2.0` bites only
    where the cap is slack, and the one slack run measured -- OAT at full rank, above the
    exact rank 9 -- is already machine-exact at `1.1`. So nothing yet argues for changing it,
    and the test that could is a slack-cap model, not a wider grid on these two.
  - `rexpand` -- EXPAND EVERY BOND TWICE PER HALF-SWEEP INSTEAD OF UNIONING. The split that
    produces `W[i]` is bounded by the evolve, so it comes out narrower than the CBE frame just
    paid for -- and `W[i]` is what sets site `i+1`'s LEFT bond. `kaug` restores that width by
    merging the frame back in; `rexpand` restores it by EXPANDING BOND `i` A SECOND TIME against
    the state as it then stands, so each bond is expanded once as the right bond of site `i` and
    once as the left bond of site `i+1`. That is the pattern `tdvp_cbe1s` gets for free by
    expanding on both its forward and backward passes, and it means every site is evolved with
    BOTH of its bonds widened. No union is performed: `rexpand = true` overrides `kaug`.

    THE COST IS A SECOND `cbe_expand` PER BOND PER HALF-SWEEP, with no extra `expv` -- the
    re-expansion happens after the exponential, not around it. Whether that buys accuracy is the
    open question this flag exists to answer; it is `false` by default until measured.
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
  - `kaug` -- UNION the basis update with the CBE frame instead of letting it REPLACE it:
    `W[i] = orth([U_ex | svd(K1).U])`, and the mirror for `Z[j]`. Only meaningful with
    `kstep = true`. `true` is the DEFAULT; `false` reproduces the historical behaviour and is
    kept only as the A/B that exhibits the defect.

    MEASURED on OAT (L=10, T=2, the `tests/sweeps/test_oat.jl` dt-order configuration), as
    `max_t |S^x_tot - closed form|`:

        tdvp_cbe1s          4.3475e-03   ratio 4.03   2nd order
        tdvp2               4.3475e-03   ratio 4.03   2nd order
        cbe_bug kaug=false  1.8271e-02   ratio 4.07   2nd order   <- WORST arm of the four
        cbe_bug kaug=true   2.2901e-06   ratio 19.6   4th order

    The old default was not merely behind `tdvp2`, it was 4.2x WORSE than it -- which is what
    justifies changing a library default rather than leaving the union opt-in.

    ⚠️ THE FOURTH ORDER IS OAT-SPECIFIC AND DOES NOT GENERALISE. `H_OAT` is diagonal in the
    computational basis and all its terms COMMUTE, so a splitting integrator incurs no Trotter
    error on it. On Heisenberg the union still beats the replaced-frame arm (2.31e-10 vs
    1.54e-09 infidelity, L=8) but stays second order and loses to `tdvp2` (1.08e-11). Quote the
    OAT number as an OAT number.

    ⛔ WHY THIS EXISTS. The bases ARE the state -- the assembly below writes `psi[i] = W[i]`
    and `psi[j+1] = Z[j]` -- so a direction that is not in `W`/`Z` cannot reach the state at
    all. But each half-sweep keeps the CBE frame on the OPPOSITE side from the basis it is
    building (`Vwide[i] = ex.V_ex` while building `W[i]`; `Uwide[j] = ex.U_ex` while building
    `Z[j]`) and the frame that would BECOME the state is discarded. So CBE's admitted
    directions are admitted into a tensor that is then thrown away, and the basis is whatever
    one short-time local solve happened to populate on top of `Kex`, whose rank in the very
    split being SVD'd is bounded by the OLD bond because `Kex` contracts over `link_i`.

    MEASURED on OAT (L=10, product start, step 1): `n_new = [1,2,2,2,2,2,2,2,1]` identically
    with and without the K-step, but `expanded` is `[2,2,2,...]` with it and `[2,3,3,3,...]`
    without -- CBE proposes two new directions per interior bond and one is discarded. The
    consequence is an energy error of 1.561e-3 that then FREEZES (1.565e-3 after a full 4π
    run), against 4.2e-15 for 2-site TDVP. Contrast `tdvp_cbe1s`, which has the SAME
    `svd(M,(1,2)).U` and is unaffected: its `expand_bond!` absorbs the widened frame INTO
    `psi`, so the widening persists in the state and compounds along the sweep.

    ⚠ Not the closing truncation (`trunc_thresh = 0` is bit-identical, `discarded ~1e-6`) and
    not `comp_ratio` (`nothing`/0.0/0.25/0.5/0.75/1.0 all bit-identical).
  - `maxdim`, `trunc_thresh` -- the closing truncation. `truncate = false` disables it, to watch
    the rank ratchet.
    The closing truncation is [`truncate_recursive!`](@ref) -- `Θ_τ` of arXiv:2201.10291 §4.2
    Algorithm 7, the recursive ROOT-TO-LEAVES truncation with each bond cut exactly once. Both
    values of `decoupled` share it, so rank control is identical between the two orderings.
  - `maxiter`, `tol` -- Krylov budget for the K-steps and the root solve.
"""
function cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                       dex::Int = 0,
                       growth::Float64 = 1.1,
                       dover::Union{Nothing, Int} = nothing,
                       comp_ratio::Union{Float64, Nothing} = 1.0,
                       sulz_cap::Bool = false,
                       preselect_only::Bool = false,
                       exact::Bool = false,
                       root::Union{Nothing, Int} = nothing,
                       kstep::Bool = true,
                       kaug::Bool = true,
                       rexpand::Bool = false,
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

    _frames_stored() = sum(sum(tensor_elements(v[k]) for k in eachindex(v)
                               if isassigned(v, k); init = 0)
                           for v in (W, Z, Vwide, Uwide, Wcbe, Zcbe); init = 0)
    peak!(ts...) = (peak = max(peak, _state_stored(psi) + _frames_stored() +
                               sum(tensor_elements(t) for t in ts; init = 0)))

    W = Vector{Any}(undef, c)          # augmented LEFT frames,  sites 1 … c
    Z = Vector{Any}(undef, L - 1)      # augmented RIGHT frames, Z[j] absorbs site j+1

    Vwide = Vector{Any}(undef, kstep ? c : 0)      # widened RIGHT frame, bonds 1 … c
    Uwide = Vector{Any}(undef, kstep ? L - 1 : 0)  # widened LEFT  frame, bonds c … L-1
    # The CBE frame. Needed for `kstep = false` (it IS the basis) and for `kaug` (it is unioned
    # into the basis) -- so it is stored whenever the basis is going to reference it.
    # THE SWEEP IS INTERLEAVED: CBE expands bond `i`, then site `i` is evolved immediately --
    # the reference's own ordering (TDVPSweepCBE1Si.m:30/:34), so the environment chain runs
    # through the EVOLVED frames and bond `i+1`'s selection sees the update at site `i`. This is
    # also what makes `kaug` well defined: `U_ex` and `svd(K1).U` are built from the SAME `K` in
    # the same iteration, so they are two bases of one space and the union is meaningful.
    _keepcbe = !kstep || kaug
    Wcbe  = Vector{Any}(undef, _keepcbe ? c : 0)
    Zcbe  = Vector{Any}(undef, _keepcbe ? L - 1 : 0)

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
        kstep && (Vwide[i] = ex.V_ex)
        _keepcbe && (Wcbe[i] = ex.U_ex)
        if kstep

            M   = to_concrete(contract(psi[i + 1], (2, 3), Vwide[i]', (2, 3)))  # (link_i, b_R)
            Kex = to_concrete(contract(K, (3,), M, (1,)))          # (b_{i-1}, s_i, b_R)
            # (b) the right environment must be the WIDENED one -- one push through `V_wide`,
            # not a rebuild.
            rch1 = push_right_channels(right_channels(rstack, i + 2), mpo, Vwide[i], i + 1)
            H1 = one_site_h(mpo, i, lch, rch1)
            peak!(Kex)
            K1 = expv(x -> (nmv[] += 1; apply_one_site(H1, x)), tau, Kex;
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)


            Wk = to_concrete(svd(K1, (1, 2); cutoff = 1e-14).U)
            # UNION, not replace -- see `kaug` in the docstring. `W[i]` becomes `psi[i]`, so a
            # CBE direction absent from it never reaches the state.
            Wk = (kaug && !rexpand) ? _union_left(Wcbe[i], Wk) : Wk
            W[i] = to_concrete(setitag(Wk, 3, _tagW(i)))
        else
            W[i] = Wcbe[i]                                          # reproduces cbe_lubich
        end

        # ── `rexpand`: WIDEN BOND `i` AGAIN, AFTER THE SPLIT ─────────────────────────────
        #
        # The split above is bounded by the evolve, so `W[i]` comes out NARROWER than the CBE
        # frame that was just paid for -- and `W[i]` is what sets site `i+1`'s LEFT bond. Under
        # `kaug` that width is restored by unioning the frame back in; here it is restored by
        # EXPANDING BOND `i` A SECOND TIME, against the state as it now stands. Every bond is
        # therefore expanded twice per half-sweep -- once as the right bond of site `i`, once as
        # the left bond of site `i+1` -- which is the pattern `tdvp_cbe1s` gets for free by
        # expanding on both its forward and backward passes. No union anywhere.
        #
        # The frame is `(W[i]) * (C_new · psi[i+1])`: `W[i]` is an isometry out of the split and
        # the carry holds the amplitude, which is exactly the one-carries-amplitude invariant
        # `_frame_from` documents. Tags already agree -- `C_new`'s first leg comes from `W[i]'`,
        # so it is `_tagW(i)` on both sides of the contraction.
        if kstep && rexpand
            Cn = to_concrete(contract(W[i]', (1, 2), K, (1, 2)))     # (b_i, link_i)
            ex2 = record!(cbe_expand(_frame_from(W[i],
                                                 to_concrete(contract(Cn, (2,), psi[i + 1], (1,)))),
                                     mpo, i, lch, right_channels(rstack, i + 2); exkw...),
                          i, :left)
            peak!(ex2.U_ex)
            W[i] = to_concrete(setitag(ex2.U_ex, 3, _tagW(i)))
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
        kstep && (Uwide[j] = ex.U_ex)
        _keepcbe && (Zcbe[j] = ex.V_ex)
        if kstep
            M   = to_concrete(contract(Uwide[j]', (1, 2), psi[j], (1, 2)))     # (b_L, link_j)
            Bex = to_concrete(contract(M, (2,), B, (1,)))           # (b_L, s_{j+1}, b_{j+1})
            lch1 = push_left_channels(left_channels(lstack, j), mpo, Uwide[j], j)
            H1 = one_site_h(mpo, j + 1, lch1, rch)
            peak!(Bex)
            B1 = expv(x -> (nmv[] += 1; apply_one_site(H1, x)), tau, Bex;
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)
            Zk = to_concrete(svd(B1, (1,); cutoff = 1e-14).Vd)
            # Mirror of the K half-sweep: `Z[j]` becomes `psi[j+1]`.
            Zk = (kaug && !rexpand) ? _union_right(Zcbe[j], Zk) : Zk
            Z[j] = to_concrete(setitag(Zk, 1, _tagZ(j)))
        else
            Z[j] = Zcbe[j]
        end

        # Row mirror of the K half-sweep's `rexpand` block: bond `j` is expanded a second time
        # after the split, so `Z[j]` -- which sets site `j`'s RIGHT bond on the next iteration --
        # carries the widened frame rather than the evolve-limited one. Frame is
        # `(psi[j] · D_new) * Z[j]`: the carry holds the amplitude, `Z[j]` is the isometry.
        if kstep && rexpand
            Dn = to_concrete(contract(B, (2, 3), Z[j]', (2, 3)))      # (link_j, b_j)
            ex2 = record!(cbe_expand(_frame_from(to_concrete(contract(psi[j], (3,), Dn, (1,))),
                                                 Z[j]),
                                     mpo, j, left_channels(lstack, j), rch; exkw...),
                          j, :right)
            peak!(ex2.V_ex)
            Z[j] = to_concrete(setitag(ex2.V_ex, 1, _tagZ(j)))
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

    res = svd(S1, (1,); cutoff = cut, Nkeep = maxdim, get_lists = true)
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
                           discarded, nmv[], peak, kaug)
end

cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_step!(psi, mpo_from_terms(h), tau; kwargs...)
