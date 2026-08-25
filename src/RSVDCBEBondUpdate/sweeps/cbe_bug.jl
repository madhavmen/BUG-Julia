
_tagW(i::Int) = "cbeW,$i"
_tagZ(j::Int) = "cbeZ,$j"

# The residual below which the Krylov space has genuinely CLOSED -- an invariant subspace, not a
# convergence judgement. Kept separate from `krylov_tol` on purpose: one is a property of the
# operator and the seed, the other is an accuracy request from the caller.
const _KRY_BREAKDOWN = 1e-13

"""
    _kry_contribution(alpha, betas, tau) -> Float64

Saad's estimate of what the NEXT Krylov vector would add to `exp(tau*H)*v0`:

    beta_m * |[exp(tau*T_m)]_{m,1}|

with `T_m` the tridiagonal built from `alpha[1..m]` and `betas[1..m-1]`. `betas[m]` is the
residual that produced the vector just accepted.

⛔ THIS, NOT `beta`, IS THE CONVERGENCE MEASURE. `beta` alone is a BREAKDOWN detector: it says the
recursion could not produce another independent direction, which on a generic `H` simply does not
happen -- MEASURED at L=12, Heisenberg and XX return BIT-IDENTICAL results for a `beta` threshold
anywhere from `1e-6` to `1e-13`, because the test never fires and every bond runs to the cap. What
decides whether a vector MATTERS is how much weight the propagator actually puts on it, and that
is the product above: `beta_m` can be O(1) while `[exp(tau*T_m)]_{m,1}` is `~(tau*||H||)^m/m!`,
which is what makes a short basis sufficient at small `tau` and a long one necessary at large.

`T_m` is `m x m` with `m` at most `krylov_basis`, so the dense `exp` here is negligible against
one application of `H` to a bond tensor.
"""
function _kry_contribution(alpha::Vector{Float64}, betas::Vector{Float64}, tau::ComplexF64)
    m = length(alpha)
    m == 0 && return Inf
    length(betas) >= m || return Inf
    T = zeros(Float64, m, m)
    @inbounds for k in 1:m
        T[k, k] = alpha[k]
        if k < m
            T[k, k + 1] = betas[k]
            T[k + 1, k] = betas[k]
        end
    end
    E = exp(tau * T)
    return abs(betas[m]) * abs(E[m, 1])
end

"""
    _krylov_frame(v0, applyH, maxdepth, tol, tau) -> Vector

The ORTHONORMAL Krylov vectors `{v0, v1, ...}` of `applyH` seeded at `v0`, stopped when the next
vector's CONTRIBUTION to `exp(tau*H)*v0` falls below `tol` (see [`_kry_contribution`](@ref)), when
the recursion breaks down, or when `maxdepth` is reached.

⛔ NO SVD PER ITERATION. The vectors are accumulated whole and split ONCE by the caller. A
selection inside the loop -- which is what `cbe_lubich_sweep`'s `grow_iters` does, re-running the
CBE ranking on every pass -- re-truncates by budget at each step and throws away exactly the
directions the next application of `H` needs: MEASURED on OAT it bought 1.8x and then saturated
flat (2.13e-02 -> 1.16e-02) no matter how many passes were allowed.

⛔ FULL RE-ORTHOGONALISATION IS NOT OPTIONAL, and this is where a plain power iteration fails.
`H^k v` converges to the dominant eigenvector, so by `k ~ 4` the raw powers are numerically
parallel and the closing SVD deletes them as dependent -- the basis would saturate for a
CONDITIONING reason that looks exactly like a physical one. Two passes, against the whole basis:
one pass of modified Gram-Schmidt loses orthogonality once the vectors start to align.

⚠ `alpha` is taken BEFORE the orthogonalisation, which is what makes it the Rayleigh quotient
`<v_k, H v_k>` rather than a residual artefact. `H` is Hermitian here, so it is real up to
roundoff and the imaginary part is discarded rather than carried.
"""
function _krylov_frame(v0, applyH, maxdepth::Int, tol::Float64, tau::ComplexF64)
    b0 = norm(v0)
    b0 <= 0 && return Any[v0]
    vs = Any[to_concrete((1.0 / b0) * v0)]
    alpha, betas = Float64[], Float64[]
    for _ in 1:maxdepth
        w = applyH(vs[end])
        push!(alpha, real(tensor_inner(vs[end], w)))
        for _pass in 1:2, u in vs
            w = to_concrete(w - tensor_inner(u, w) * u)
        end
        b = norm(w)
        # BREAKDOWN comes first and is unconditional: with no independent direction left there is
        # nothing for the contribution test to weigh, and `1/b` would not be finite.
        b <= _KRY_BREAKDOWN && break
        push!(betas, b)
        push!(vs, to_concrete((1.0 / b) * w))
        tol > 0 && _kry_contribution(alpha, betas, tau) < tol && break
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
    # ⛔ EXTRA `cbe_expand` PASSES (Jan's point 4), AND IT IS HERE BECAUSE `krylov_dims` CANNOT
    # SEE THEM. `krylov_dims` counts `apply_one_site` calls; a growth pass is a `cbe_expand`,
    # whose operator work happens inside `sketch_h_*` and is NOT counted there. Reporting a
    # growing sweep on `krylov_dims` alone would show its accuracy and hide its cost -- the same
    # blindness that made the retired `rexpand` look free. Zero when `krylov_grow = false`.
    n_grow::Int
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

  - `krylov_tol` -- **THE STOPPING RULE, `1e-6`.** The half-sweep stops adding Krylov vectors when
    the next one's CONTRIBUTION to `exp(tau*H_1)*Theta` falls below it -- Saad's estimate
    `beta_m*|[exp(tau*T_m)]_{m,1}|`, see [`_kry_contribution`](@ref). `0.0` disables the test and
    runs to the cap. Pairs with `trunc_thresh` the way `tau_Krylov = tau_trunc/10` does.
  - `krylov_basis` -- the CAP on that iteration, default `30`.

    ⚠ **THE CRITERION USED TO BE `beta` ALONE AND THAT DID NOT WORK.** MEASURED at L=12 before the
    change: Heisenberg and XX returned BIT-IDENTICAL results for a `beta` threshold anywhere from
    `1e-6` to `1e-13` -- the test never fired and every bond ran to the cap. `beta` is a BREAKDOWN
    detector: it reports that no independent direction is left, which on a generic `H` does not
    happen. It survives here only as the `_KRY_BREAKDOWN` guard, where it belongs.

    MEASURED WITH THE CONTRIBUTION RULE, L=12, dt=0.05, T=1.0, against exact references
    (`benchmarks/krylov_stopping.jl`; `krylov` = accumulated operator applications):

        arm            XX err / krylov        Heisenberg err / krylov   OAT err / krylov
        cap, no test   1.6948e-04 /  1770     1.0800e-06 /  2280        5.9952e-15 / 2158
        tol 1e-4       1.6948e-04 /   952     2.4032e-06 /  1384        1.6431e-14 / 1552
        tol 1e-6  DEF  1.6948e-04 /  1180     1.4748e-06 /  1622        1.0880e-14 / 1766
        tol 1e-10      1.6948e-04 /  1550     1.1190e-06 /  1976        6.2172e-15 / 2000
        pinned m=3     1.6948e-04 /   950     2.6697e-06 /  1038        7.0610e-14 / 1036
        pinned m=0     1.6948e-04 /   289     8.8424e-05 /   308        2.1329e-02 /  313

    Read three things off it. **XX: the rule cuts 1.9x for free** -- bit-identical error at every
    tolerance, so it correctly finds that depth buys nothing here. **OAT: the rule refuses to cut**
    -- every tolerance stays at ~1e-14 where `m = 0` is 2.1e-02, which is the safety property that
    matters, since this is the model where depth is worth eight orders. (The apparent ordering
    among the OAT tolerance rows is roundoff against roundoff and means nothing.)
    **Heisenberg: it is a genuine dial, not a free lunch** -- `1e-6` buys 0.71x the cost for 1.37x
    the error, and `1e-10` is nearly free at 1.04x error for 0.87x cost. That monotone trade is
    the deliverable; `beta` gave one point and no dial at all.

    ⚠ **`1e-6` WAS CHOSEN WHILE THIS KNOB WAS INERT**, so the number predates the rule that now
    reads it. It is kept as the default deliberately, but a caller who wants the accurate end
    should use `1e-10` rather than assume the default is conservative.

    Set `krylov_basis` to a small fixed number (2-3) and `krylov_tol = 0.0` to pin the depth.

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
                       krylov_grow::Bool = false,
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
    ngrow = Ref(0)

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

    """
        _grow_frame(v0, Vopp, side, i, lch, rch) -> (blocks, Vopp_final)

    Jan's point 4: RE-EXPAND THE BOND BETWEEN LANCZOS STEPS instead of once per bond.

    The fixed-width builder [`_krylov_frame`](@ref) widens the bond once, builds `H1` against that
    frame and runs the whole recursion inside it -- so any part of `H^k*Theta` lying outside the
    initial expansion is projected away at every step. Here the OPPOSITE frame is re-expanded
    after each accepted vector, the environment and `H1` are rebuilt against it, and the vectors
    already accumulated are EMBEDDED into the widened space rather than recomputed.

    ⛔ THE EMBEDDING IS AN OVERLAP, NOT AN ASSUMPTION. `cbe_expand` returns
    `V_ex = oplus([V0, TRC])` where `V0` is its own factorisation of what was handed in, which
    need not be the tensor that went in -- the orthonormality guard may rotate within the appended
    span. Computing `PR` as the true overlap `<V_old, V_new>` is correct under that rotation;
    assuming `[I 0]` is not. (Same reasoning as `_expanding_krylov`'s `embed_ops`.)

    ⛔ ACCUMULATE, NEVER RE-RANK. Jan's bookkeeping is "new kept = old kept PLUS expanded", and
    that word is load-bearing: `cbe_lubich_sweep`'s `grow_iters` re-runs the CBE SELECTION each
    pass, which re-truncates by budget and discards exactly what the next application of `H`
    needs -- MEASURED on OAT it bought 1.8x and then went flat. Nothing here touches the vectors
    already accepted; the space only ever grows.

    ⚠ COST DOES NOT APPEAR IN `krylov_dims`. Each pass is a `cbe_expand`, whose operator work is
    inside `sketch_h_*`. `ngrow` counts the passes for exactly this reason -- see the note on
    `CBEBugSweepInfo.n_grow`.
    """
    function _grow_frame(v0, Vopp, side::Symbol, i::Int, lch, rch, dex_pass::Int)
        # ⚠ THE TWO SIDES ARE NOT MIRROR IMAGES IN LEG INDEX, and getting this backwards is silent:
        # the shapes still contract, they just contract the wrong leg.
        #   :left   opposite frame is `V_ex = (g, site, link)`  -> its bond leg is 1
        #           the Krylov vectors are `(link, site, g)`    -> their bond leg is 3
        #   :right  opposite frame is `U_ex = (link, site, g)`  -> its bond leg is 3
        #           the Krylov vectors are `(g, site, link)`    -> their bond leg is 1
        leg = side === :left ? 1 : 3                  # the OPPOSITE frame's bond leg
        b0 = norm(v0)
        b0 <= 0 && return (Any[v0], Vopp)
        V  = Vopp
        vs = Any[to_concrete((1.0 / b0) * v0)]
        alpha, betas = Float64[], Float64[]
        for k in 1:krylov_basis
            H1 = side === :left ?
                 one_site_h(mpo, i, lch, push_right_channels(rch, mpo, V, i + 1)) :
                 one_site_h(mpo, i + 1, push_left_channels(lch, mpo, V, i), rch)
            nmv[] += 1
            w = apply_one_site(H1, vs[end])
            push!(alpha, real(tensor_inner(vs[end], w)))
            for _pass in 1:2, u in vs
                w = to_concrete(w - tensor_inner(u, w) * u)
            end
            b = norm(w)
            b <= _KRY_BREAKDOWN && break
            push!(betas, b)
            push!(vs, to_concrete((1.0 / b) * w))
            ktol > 0 && _kry_contribution(alpha, betas, tau) < ktol && break
            k == krylov_basis && break

            # ── expand against the NEWEST direction, then embed everything into the result ──
            #
            # ⛔ NOT `_frame_from`. That helper factorises BOTH sides and joins them through
            # `res_l.S * res_l.Vd`, which needs the two bond legs to meet as an ordinary pair. On
            # the ALREADY-EXPANDED bond they do not: the Krylov vector's bond leg was produced by
            # contracting against `V_ex'`, so `S` comes back 0-dimensional and that product has
            # nothing to contract (`AssertionError: No matching contractible indices`).
            #
            # It is also unnecessary work. One side here is ALREADY an isometry -- it is the
            # opposite CBE frame -- so only the other side needs factorising, and the core is then
            # a single overlap rather than a product of four SVD outputs.
            fr = if side === :left
                res = svd(vs[end], (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
                U0  = to_concrete(res.U)
                S0  = to_concrete(contract(U0', (1, 2), vs[end], (1, 2)))      # (bU, g)
                BondFrame(U0, S0, V,
                          vs[end].inds[1], vs[end].inds[2], vs[end].inds[3],
                          V.inds[2], V.inds[3],
                          leg_dim(U0, 3), copy(vs[end].spaces[3]))
            else
                res = svd(vs[end], (1,), "bV,L", "bV,R"; cutoff = 0.0)
                V0  = to_concrete(res.Vd)
                S0  = to_concrete(contract(vs[end], (2, 3), V0', (2, 3)))      # (g, bV)
                BondFrame(V, S0, V0,
                          V.inds[1], V.inds[2], V.inds[3],
                          vs[end].inds[2], vs[end].inds[3],
                          leg_dim(V, 3), copy(V.spaces[3]))
            end
            # ⛔ AN EXPLICIT `dex` PER PASS, OR THIS WHOLE BRANCH IS A NO-OP. `cbe_expand`'s
            # default budget is `ceil(growth*dmax) - r`, and the FIRST expansion already took the
            # bond to `growth*dmax` -- so on every later pass `r` has grown, the budget collapses
            # to zero, and the call returns the frame it was given. MEASURED before this line
            # existed: 178-238 expansion passes per run that changed the error by NOTHING (XX,
            # Heisenberg and OAT all bit-identical to `grow = false`). Each pass gets its own
            # allowance instead, so "old kept PLUS expanded" can actually add something.
            exk = cbe_expand(fr, mpo, i, lch, rch; exkw..., dex = dex_pass)
            Vn = side === :left ? exk.V_ex : exk.U_ex
            # `ngrow` counts ACTUAL WIDENINGS, not calls. Counting calls made a no-op look like
            # work and let the smoke test's "did it grow?" gate pass on a branch that grew nothing.
            if leg_dim(Vn, leg) > leg_dim(V, leg)
                ngrow[] += 1
                P = side === :left ?
                    to_concrete(contract(V, (2, 3), Vn', (2, 3))) :      # (old_g, new_g)
                    to_concrete(contract(Vn', (1, 2), V, (1, 2)))       # (new_g, old_g)
                for t in eachindex(vs)
                    vs[t] = side === :left ?
                            to_concrete(contract(vs[t], (3,), P, (1,))) :
                            to_concrete(contract(P, (2,), vs[t], (1,)))
                end
                V = Vn
            end
        end
        vs[1] = to_concrete(b0 * vs[1])
        return (vs, V)
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
            blocks = if krylov_grow
                # PER-PASS BUDGET = what the FIRST expansion admitted on this bond. Keeping the
                # increment constant makes the space grow linearly in the Lanczos step, which is
                # what "old kept plus expanded" asks for; reusing `growth` would re-derive a
                # budget from the rank that expansion just raised, i.e. zero.
                bl, Vfin = _grow_frame(Kex, ex.V_ex, :left, i, lch,
                                       right_channels(rstack, i + 2), max(ex.n_new_r, 1))
                Vwide[i] = Vfin          # accounting must see the width the growth reached
                bl
            else
                rch1 = push_right_channels(right_channels(rstack, i + 2), mpo, ex.V_ex, i + 1)
                H1 = one_site_h(mpo, i, lch, rch1)
                _krylov_frame(Kex, x -> (nmv[] += 1; apply_one_site(H1, x)),
                              krylov_basis, ktol, tau)
            end
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
            blocks = if krylov_grow
                bl, Ufin = _grow_frame(Bex, ex.U_ex, :right, j,
                                       left_channels(lstack, j), rch, max(ex.n_new_l, 1))
                Uwide[j] = Ufin
                bl
            else
                lch1 = push_left_channels(left_channels(lstack, j), mpo, ex.U_ex, j)
                H1 = one_site_h(mpo, j + 1, lch1, rch)
                _krylov_frame(Bex, x -> (nmv[] += 1; apply_one_site(H1, x)),
                              krylov_basis, ktol, tau)
            end
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
                           discarded, nmv[], peak, ngrow[])
end

cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_step!(psi, mpo_from_terms(h), tau; kwargs...)
