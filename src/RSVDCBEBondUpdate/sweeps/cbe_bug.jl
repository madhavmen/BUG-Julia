
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

  - `growth` -- the expansion budget, `budget = ceil(growth*dmax) - r`. **`2.0`, matching
    [`cbe_expand`](@ref) and `CBEBugOptions`.** This shipped `1.1` -- the reference's DMRG
    ratchet -- until the `rexpand` measurements below; all three now agree.

    ⛔ `rexpand` NEEDS THE LARGER BUDGET AND `kaug` DOES NOT, and the reason is structural
    rather than empirical. `rexpand`'s SECOND expansion runs against
    `U0 = orth([svd(K).U | svd(K1).U])`, which is strictly larger than the first expansion's
    `U0 = svd(K).U` that `kaug` extends. `r` is that rank, and it is SUBTRACTED from the
    budget -- so at fixed `growth` the same schedule starves `rexpand` and not `kaug`.

    MEASURED on TFIM L=16, D=32, the model where `rexpand` trailed:

        kaug     growth=1.1   1.4293e-05        rexpand  growth=1.1   3.6578e-05
        kaug     dex=8        1.6735e-04        rexpand  dex=8        4.8117e-05
                                                rexpand  dex=16       7.0810e-05
                                                rexpand  growth=2.0   1.5135e-05

    `growth = 2.0` takes `rexpand` from 2.6x worse to parity. `dex` cannot: it is an ABSOLUTE
    override that ignores `r` entirely, which is why its sweep is non-monotone and never
    reaches the schedule's result. And the control rules out "more budget helps everyone" --
    `kaug` at `dex = 8` is 12x WORSE.

    NO REGRESSION ELSEWHERE. OAT L=10 D=6 and XX L=16 D=24 are unchanged for BOTH mechanisms
    at `2.0` (OAT 3.2748e-11, XX 8.1911e-05), and `rexpand` at `2.0` becomes bit-identical to
    `kaug` on OAT rather than differing in the last digit. Cost is +2.7% Krylov on OAT and
    +0.3% on XX.

    An earlier note here argued `1.1` should stand because the K-step is a second source of
    directions, with the caveat that every supporting number came from a CAP-BOUND run where
    the budget cannot matter. That reasoning was right for `kaug` -- which still does not want
    more budget -- and the caveat is exactly where `rexpand` broke it.
  - `rexpand` -- **THE DEFAULT.** EXPAND EVERY BOND TWICE PER HALF-SWEEP INSTEAD OF UNIONING.
    The split that produces `W[i]` is bounded by the evolve, so it comes out narrower than the
    CBE frame just paid for -- and `W[i]` is what sets site `i+1`'s LEFT bond. `kaug` restores
    that width by merging the frame back in; `rexpand` restores it by EXPANDING BOND `i` A SECOND
    TIME against the state as it then stands, so each bond is expanded once as the right bond of
    site `i` and once as the left bond of site `i+1`. Every site is therefore evolved with BOTH
    of its bonds widened, which is what `tdvp_cbe1s` gets for free by expanding on both its
    forward and backward passes. No basis is ever formed by merging two others:
    **`rexpand = true` overrides `kaug`**, so an A/B on `kaug` must pass `rexpand = false`.

    THE COST IS A SECOND `cbe_expand` PER BOND PER HALF-SWEEP AND NO EXTRA `expv` -- the
    re-expansion happens after the exponential, not around it, so it is QR and matmul rather
    than Krylov. MEASURED (`benchmarks/arm4_rexpand.jl`, OAT L=10, D=6 = the exact ceiling, so
    rank is not the limiter and the bond profile is a pass/fail with no tolerance in it):

        arm                infidelity    krylov  bond profile
        tdvp2              1.9927e-06     20178  [2,3,6,6,6,6,6,4,2]
        tdvp_cbe1s         1.9927e-06     16844  [2,3,6,6,6,6,6,4,2]
        bug kaug=true      3.2748e-11      5149  [2,3,4,5,6,5,4,3,2]  exact
        bug rexpand=true   3.2749e-11      5135  [2,3,4,5,6,5,4,3,2]  exact
        bug kaug=false     3.9758e-05      5196  [2,3,6,6,6,6,6,3,2]
        bug kstep=false    6.3963e-05       560  [2,3,4,5,6,5,4,3,2]  exact

    ACROSS THE THREE EXAMPLE MODELS, against `kaug`:

        OAT   L=10 D=6     3.2749e-11  vs  3.2748e-11    matches to 5 figures
        XX    L=16 D=24    8.1911e-05  vs  8.1911e-05    identical
        TFIM  L=16 :Z2     1.413e-05   vs  6.563e-06     2.2x worse
        TFIM  L=16 :none   1.241e-05   vs  3.838e-06     3.2x worse

    So the two mechanisms agree wherever the evolved basis is already good, and `kaug` keeps a
    real edge on TFIM. `rexpand` is the default for the STRUCTURE rather than that number: no
    basis is ever formed by merging two others, and every site is evolved with both of its bonds
    widened, which is the reference sweep's ordering.

    ⛔ THE `_union_left(svd(K).U, Wk)` ABOVE IS NOT PART OF THE WIDENING AND MUST NOT BE REMOVED
    AS REDUNDANT. It adds no direction that was not already needed to represent `K`; it only
    makes `W[i]·W[i]'·K == K`, so that the second expansion sketches `H·Theta` against the TRUE
    two-site block. Without it the frame is `P_W·K·psi[i+1]`, a PROJECTION -- `svd(K1).U` comes
    from the evolved tensor and `Kex = K·M` matricises through `M`, so its left span can be
    strictly smaller than `K`'s. The second expansion then returns the right directions for the
    wrong `Theta`. MEASURED before the fix: TFIM 2.508e-01 (`:Z2`) and 5.467e-01 (`:none`) --
    wrong answers, not degraded ones -- while OAT, whose evolved basis is nearly exact so the
    projection loses almost nothing, moved only from 3.27e-11 to 4.63e-09. The model dependence
    is the tell, and it is why this cannot be caught on OAT alone.

    Note also that BOTH TDVPs over-fill the OAT profile to `[2,3,6,6,6,6,6,4,2]`. Landing on the
    exact Schmidt profile is a property of the BUG half-sweeps, not of `kaug` or `rexpand`.

  - `basis_frac` -- how far the HALF-SWEEPS evolve, as a fraction of `tau`. The root Galerkin step
    always takes the full `tau`. `1.0` (default) is the endpoint basis; `0.5` would be the
    MIDPOINT BUG, the standard route to second order for BUG-type integrators.

    ⛔ MEASURED, AND IT DOES NOT WORK HERE -- LEAVE IT AT `1.0`. The knob is kept only because it
    is the instrument that established that, and because the XX column below is a live check on a
    structural property of the half-sweeps. `benchmarks/midpoint_basis.jl`, L=12, three `dt`:

        XX   D=32   basis_frac 1.0 / 0.75 / 0.5 / 0.25 -> BIT-IDENTICAL, 6.4211e-05, ratio 3.99
        TFIM D=32   1.0    4.8595e-05  3.1491e-06  1.9996e-07   ratio 15.43   <- best, 4th order
                    0.75   2.0043e-04  2.6416e-05  3.3209e-06   ratio  7.59
                    0.5    4.1845e-04  5.4065e-05  6.6882e-06   ratio  7.74
                    0.25   6.3668e-04  8.1329e-05  1.0051e-05   ratio  7.83

    Shortening the basis evolution COSTS AN ORDER on TFIM -- 4th (ratio ~15.5) down to 3rd (~7.8)
    -- and is 21x worse at the finest `dt`. It is not neutral; it is strictly harmful.

    WHY, AND IT IS THE SAME REASON THE SWEEP IS A BUG AND NOT A TDVP: the half-sweeps are BASIS
    GENERATORS -- `C = contract(W[i]', K)` uses `K`, not `K1`, so the evolved AMPLITUDE is thrown
    away and only the COLUMN SPAN is kept. To leading order that span is

        col(Kex) + col(exp(btau·H)·Kex) = col(Kex) + col(H·Kex) + O(btau)

    which is INDEPENDENT of how far the half-sweep evolved. So where nothing truncates the union,
    `basis_frac` is exactly inert -- which is what the bit-identical XX column IS, measured rather
    than argued. Where the union IS truncated, `basis_frac` no longer selects a different span,
    only a worse-resolved one: the `O(btau)` content that distinguishes the good directions from
    the marginal ones shrinks with `btau`, so the truncation cuts on less information.

    A midpoint rule needs the basis to CARRY the half-step state. This one cannot: it discards
    amplitude by construction. The midpoint BUG is not available to a sweep built this way, and
    that is a property of the scheme rather than a tuning failure.

    WHY THERE IS A KNOB HERE AT ALL, and why it is the only symmetrisation left: the K and L
    half-sweeps do NOT need symmetrising -- both read the FROZEN start-of-step state (`rstack` and
    `lstack` are built once, before either runs) and write into independent `W[]` / `Z[]`, so
    swapping their order is a literal no-op. That leaves the ROOT as the only spatial asymmetry --
    and the root is a BOND, `1:L-1`, on which reflection acts as `b -> L - b`. The default root
    `L/2` is a FIXED POINT of that map on an even chain. So at its default the sweep is ALREADY
    spatially reflection-symmetric and a reflected composition has nothing to cancel.

    MEASURED (`benchmarks/symmetric_composition.jl`, XX L=16 D=24): composing `S(tau/2)` at root
    `c` with `S(tau/2)` at root `L-c` is BIT-IDENTICAL to plain substepping at the same root --
    2.0258e-05 at 43546 Krylov for both. That is the fixed point showing up as arithmetic, and it
    doubles as a measurement of the commuting claim above: were the half-sweeps order-dependent,
    the K-before-L ordering would still break the symmetry at a fixed-point root and the two arms
    would differ. They agree to the last digit.

    Spatial symmetrisation being exhausted, `basis_frac` is the TEMPORAL one, and it is the cheap
    kind -- see the `btau` comment in the body for why it costs 1x rather than 2x.
  - `split_cutoff`, `split_maxdim` -- TRUNCATION AT THE HALF-SWEEP SPLIT, i.e. on `svd(K1).U`
    and `svd(B1).Vd` rather than at the root. Defaults `1e-14` and `0` (no cap) reproduce the
    historical behaviour exactly, in which the half-sweeps prune essentially nothing and EVERY
    rank decision is deferred to the root solve and `truncate_recursive!`.

    THIS IS THE ONE STRUCTURAL DIFFERENCE FROM `tdvp_cbe1s` THAT IS NOT ABOUT WIDENING. That
    sweep splits with `cutoff = cut, Nkeep = maxdim` at every site, so it prunes continuously;
    this one does not prune at all until the end.

    MEASURED AT THE CURRENT DEFAULTS (`benchmarks/split_truncation.jl`, TFIM L=16 D=32, `rexpand`
    with `growth = 2.0`): the whole grid spans 1.47e-05 to 1.72e-05 against a 1.5135e-05
    baseline -- 17%, with `cutoff = 1e-12` marginally the best at 1.4714e-05 -- and XX is
    BIT-IDENTICAL across every setting. Deferring rank control to the root is therefore very
    nearly free, not merely defensible, and the default is left alone because 2.8% does not
    justify another knob in the shipped configuration.

    ⛔ AN EARLIER VERSION OF THIS NOTE SAID PRUNING WAS ACTIVELY HARMFUL AND SHOULD NOT BE
    RETRIED. That was measured at `growth = 1.1`, which -- as the `growth` entry above records --
    STARVES `rexpand`. Under starvation the grid spanned 3.66e-05 to 9.48e-05 and `Nkeep = maxdim`
    cost 2.6x, because trimming could only remove directions the expansion was already short of.
    The harm was an artefact of the budget, not a property of the split, and it vanished when the
    budget did. Recorded because the confound is the reusable lesson: a knob measured under a
    starved sweep tells you about the starvation.

    ⛔ THIS DOES NOT REACH THE TRUE-STATE UNION. `_union_left(svd(K).U, Wk)` keeps its own
    `1e-14`, deliberately: its job is to make `W[i]W[i]'K == K` so the second expansion sees the
    true block, and truncating it harder reintroduces exactly the projection defect that made
    TFIM return 2.5e-01. Prune the evolve's basis, never the state's.
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
    `kstep = true`. `false` reproduces the historical behaviour and is kept only as the A/B that
    exhibits the defect.

    ⛔ INERT UNDER THE DEFAULT. `rexpand = true` restores the same width by re-expanding and
    overrides this flag entirely, so varying `kaug` alone changes NOTHING unless you also pass
    `rexpand = false`. Every A/B on `kaug` must do so or it silently compares an arm with itself.

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
                       growth::Float64 = 2.0,
                       dover::Union{Nothing, Int} = nothing,
                       comp_ratio::Union{Float64, Nothing} = 1.0,
                       sulz_cap::Bool = false,
                       preselect_only::Bool = false,
                       exact::Bool = false,
                       root::Union{Nothing, Int} = nothing,
                       kstep::Bool = true,
                       kaug::Bool = true,
                       rexpand::Bool = true,
                       basis_frac::Float64 = 1.0,
                       split_cutoff::Float64 = 1e-14,
                       split_maxdim::Int = 0,
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

    # THE MIDPOINT-BUG KNOB. The half-sweeps evolve by `btau`, the root Galerkin step always by
    # the full `tau`. `basis_frac = 1.0` is the endpoint basis (bases and root agree) and is the
    # historical behaviour; `0.5` builds the bases at the MIDPOINT and is the standard route to
    # second order for BUG-type integrators -- the bases are the O(tau) object, so centring them
    # cancels the leading term the same way a midpoint rule does.
    #
    # ⛔ THIS IS NOT A COMPOSITION AND DOES NOT DOUBLE THE COST. Same half-sweep count, same root
    # solve; only the argument of the half-sweep `expv` changes (a shorter `tau` is if anything
    # slightly CHEAPER in Krylov). So the bar is 1x, not the ~4x that spending the same compute on
    # substepping already buys -- which is the bar any 2x-cost symmetrisation would have to clear.
    btau = basis_frac == 1.0 ? tau : ComplexF64(basis_frac) * tau

    cut = max(trunc_thresh, 1e-14)

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
            K1 = expv(x -> (nmv[] += 1; apply_one_site(H1, x)), btau, Kex;
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)


            Wk = _splitU(K1)
            # UNION, not replace -- see `kaug` in the docstring. `W[i]` becomes `psi[i]`, so a
            # CBE direction absent from it never reaches the state.
            Wk = (kaug && !rexpand) ? _union_left(Wcbe[i], Wk) : Wk

            # ⛔ `rexpand` MUST RE-EXPAND FROM THE TRUE STATE, AND THAT COSTS THIS ONE LINE.
            # The frame handed to the second expansion below is `W[i]·(W[i]'K)·psi[i+1]`, which
            # equals the true two-site block ONLY IF `W[i]` spans `K`. It need not: `svd(K1).U`
            # comes from the EVOLVED tensor, and `Kex = K·M` already matricises through `M`, so
            # its left span can be strictly smaller than `K`'s. Left unfixed, the second
            # expansion sketches `H·Theta` against a PROJECTED state and returns the right
            # directions for the wrong `Theta` -- measured on TFIM L=16 as 2.5e-01 (`:Z2`) and
            # 5.5e-01 (`:none`) against 6.6e-06 for the union, while OAT, whose evolved basis is
            # nearly exact so the projection loses almost nothing, barely moved.
            #
            # Unioning `svd(K).U` in is a REPRESENTATION fix, not a widening mechanism: it only
            # restores exactness of the frame. Every new direction still comes from `cbe_expand`
            # below, never from a merge -- which is the property that distinguishes this from
            # `kaug`.
            rexpand && (Wk = _union_left(to_concrete(svd(K, (1, 2); cutoff = 1e-14).U), Wk))
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
            B1 = expv(x -> (nmv[] += 1; apply_one_site(H1, x)), btau, Bex;
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)
            Zk = _splitV(B1)
            # Mirror of the K half-sweep: `Z[j]` becomes `psi[j+1]`.
            Zk = (kaug && !rexpand) ? _union_right(Zcbe[j], Zk) : Zk
            # Row mirror of the K half-sweep's true-state fix: `Z[j]` must span `B` or the
            # re-expansion below sees `B·P_Z` instead of the true block. Representation only --
            # the widening still comes from `cbe_expand`.
            rexpand && (Zk = _union_right(to_concrete(svd(B, (1,); cutoff = 1e-14).Vd), Zk))
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
