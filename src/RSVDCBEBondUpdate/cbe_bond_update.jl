# CBE as the basis update of the GATE-BASED bond_update_bug -- the validated path.
#
# WHY THIS FILE EXISTS, AND WHY IT COMES FIRST.
#
# `cbe_lubich.jl` puts CBE inside the Lubich/Sulz basis sweep: K-sweep, L-sweep, one centre
# Galerkin, an environment-dressed effective H. That sweep is the eventual target, but it means a failure has four
# possible homes -- the sweep, the channel environments, the CBE selection, or the placement
# of the single Galerkin step. The rank stall could not be pinned to any of them.
#
# `BondUpdateBUG.parity_sweep!` + `kls_bond_update` is the OTHER integrator, and it is
# validated: odd/even Trotter, one commuting bond group at a time, exact within a group,
# full-rank-exact, unitarity-locked. Dropping CBE into THAT removes three of the four
# suspects at once:
#
#   * no environments at all -- the Hamiltonian at a bond IS the gate, purely local, so
#     `henv.jl` and the whole stale/fresh-channel question is out of the picture;
#   * no sweep to get wrong -- `canonical!(psi, i)` before each bond, exactly as the
#     validated sweep does, so every frame is a true canonical snapshot;
#   * no Galerkin-placement question -- there is one Galerkin step per bond and it is the
#     step the validated integrator already takes.
#
# What is left is precisely the piece under suspicion: does the CBE selection admit the
# directions the dynamics needs? If rank grows here, the selection is sound and the fault in
# `cbe_lubich.jl` is in the sweep. If it stalls here too, the selection is the fault, and it is
# now isolated with nothing else to blame.
#
# The selection code is SHARED, not reimplemented: `cbe_expand_bond` calls the same
# `cbe_expand(f, skl, skr; ...)` core as the MPO path, differing only in the two sketch
# closures. So this file is a genuine test of the selection the Lubich sweep depends on.
#
# WHAT IT REPLACES IN `kls_bond_update`. Both of that kernel's basis mechanisms, together:
#
#   * the two K/L ODE solves (`expv` on `P_perp H_K`, non-Hermitian, Arnoldi). CBE supplies
#     the same DIRECTIONS from one sketched application of the gate, with no exponential;
#   * `missing_fill`, the per-sector random seed. The sector-graded sketch reaches an
#     empty-but-reachable sector THROUGH the gate, so what comes back is the direction the
#     dynamics wants rather than a random column the S-step SVD then discards.
#
# The S-step is unchanged and deliberately so -- it is the Galerkin step of the validated
# integrator, `S_hat0 = U_hat' Theta0 V_hat'` with no overlap matrices, Hermitian, Lanczos.

# ── the sketched gate action ──────────────────────────────────────────────────
#
# NEVER FORMS `H Theta`. The rank-4 two-site object is what 2-site TDVP builds and what this
# method promises not to allocate, and `cbe.jl`'s header spells out why sketching after the
# fact would be a strawman: with `H Theta` in hand its exact left SVD beats any sketch of
# it. So `Om` is folded into the gate FIRST -- `gate` is `d^4`, `Om` is thin, and the
# largest intermediate is `O(d^2 * g * r)`.
#
# The guard is `sketch_bond_left(f, gate, Om) == contract(apply_gate(gate, Theta), Om')` for
# arbitrary `Om`, against `apply_gate`, which is itself pinned to the analytic Heisenberg
# block element-for-element (`tests/BondUpdateBUG/test_gates.jl`).

"Check `gate` is tagged for the bond `f` describes, as `apply_gate` does for `theta`."
function _check_gate(f::BondFrame, gate)
    tl, tr = f.site_l.itags, f.site_r.itags
    (gate.inds[1].itags == tl && gate.inds[2].itags == tr) || throw(ArgumentError(
        "gate is tagged for ($(gate.inds[1].itags), $(gate.inds[2].itags)), " *
        "bond frame carries ($tl, $tr)"))
    return nothing
end

"""
    sketch_bond_left(f, gate, Om) -> TLArray

`Y = (gate Theta) Om†`, legs `(link_l, site_l, g)` -- the left matricization of the gate
action sketched from the right. `Om` carries `V0`'s layout `(g, site_r, link_r)`, i.e. it is
a candidate right frame with `g` columns instead of `r`; `Om = f.V0` recovers the exact left
factor of `gate Theta`.

Contraction order: absorb `Om` into the gate's right BRA leg, then close the gate's right
KET leg and `link_r` against `V0`, then `K0 = U0 S0`. Nothing rank-4 is built.
"""
function sketch_bond_left(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    # gate legs (ket_l '+', ket_r '+', bra_l '-', bra_r '-'); Om' flips Om's arrows.
    G = contract(gate, (4,), Om', (2,))          # (ket_l, ket_r, bra_l, g, link_r)
    T = contract(G, (2, 5), f.V0, (2, 3))        # (ket_l, bra_l, g, bond)
    K0 = to_concrete(f.U0 * f.S0)                # (link_l, site_l, bond)
    return to_concrete(contract(K0, (2, 3), T, (1, 4)))   # (link_l, site_l, g)
end

"""
    sketch_bond_right(f, gate, Om) -> TLArray

Mirror of [`sketch_bond_left`](@ref): `Om` carries `U0`'s layout `(link_l, site_l, g)` and
the result is `(g, site_r, link_r)`. `Om = f.U0` recovers the exact right factor.
"""
function sketch_bond_right(f::BondFrame, gate, Om)
    _check_gate(f, gate)
    G = contract(gate, (3,), Om', (2,))          # (ket_l, ket_r, bra_r, link_l, g)
    T = contract(G, (1, 4), f.U0, (2, 1))        # (ket_r, bra_r, g, bond)
    L0 = to_concrete(f.S0 * f.V0)                # (bond, site_r, link_r)
    Y = contract(L0, (1, 2), T, (4, 1))          # (link_r, site_r, g)
    return to_concrete(permutedims(Y, (3, 2, 1)))         # (g, site_r, link_r)
end

"""
    cbe_expand_bond(f, gate; kwargs...) -> CBEExpansion

Controlled bond expansion of the frames at one bond, with the bare two-site gate standing in for
the environment-dressed effective Hamiltonian. Keywords are [`cbe_expand`](@ref)'s, unchanged -- this is the same
selection with different sketch closures.
"""
cbe_expand_bond(f::BondFrame, gate; kwargs...) =
    cbe_expand(f,
               Om -> sketch_bond_left(f, gate, Om),
               Om -> sketch_bond_right(f, gate, Om); kwargs...)

# ── the bond update ──────────────────────────────────────────────────────────

"""
    cbe_bond_update(f::BondFrame, gate, tau; kwargs...) -> NamedTuple

One CBE-basis Galerkin update on the bond `f` snapshots -- a drop-in replacement for
`BondUpdateBUG.kls_bond_update` with the K/L solves and the random sector fill both replaced
by [`cbe_expand_bond`](@ref).

Returns the same fields `kls_bond_update` does (`left_core`, `right_core`, `U_aug`, `V_aug`,
`S_new`, `n_new_k`, `n_new_l`, `keep`, `aug_k`, `aug_l`, `svals`, `discarded`) so it slots
into the same sweep, plus `err_pre` / `err_fnl` from the expansion.

Keywords: `maxdim`, `trunc_thresh`, `s_tau`, `maxiter`, `tol` as in `kls_bond_update`;
`expand = false` freezes the frames for a fixed-rank projector step (the analogue of
`augment = false`); everything else forwards to `cbe_expand`.

EXACTNESS AT FULL RANK. `U_ex` contains `U0` as its first block and `V_ex` contains `V0`, so
`S_hat0 = U_ex' Theta0 V_ex'` is lossless and `tau = 0` is the identity on the state. When
the expanded frames span the whole local space the Galerkin generator IS the gate, so the
step is the exact two-site propagator -- the same property the validated kernel has, and the
first thing the tests check.
"""
function cbe_bond_update(f::BondFrame, gate, tau::ComplexF64;
                         maxdim::Int = 200,
                         trunc_thresh::Float64 = 1e-14,
                         s_tau::Union{Nothing, ComplexF64} = nothing,
                         expand::Bool = true,
                         maxiter::Int = 30,
                         tol::Float64 = 1e-15,
                         rng::AbstractRNG = MersenneTwister(0x5EED),
                         expand_kwargs...)
    tau_s = s_tau === nothing ? tau : s_tau
    tl, tr = f.site_l.itags, f.site_r.itags

    ex = expand ? cbe_expand_bond(f, gate; rng = rng, expand_kwargs...) :
                  CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)
    U_aug, V_aug = ex.U_ex, ex.V_ex

    theta0 = frame_theta(f)
    S_start = to_concrete(contract(contract(U_aug', (1, 2), theta0, (1, 2)),
                                   (2, 3), V_aug', (2, 3)))

    # COUNTED, not inferred. `expv` returns only the vector, and it belongs to the validated
    # module, so the iteration count is taken here instead: one increment per operator
    # application IS the Krylov dimension the Lanczos used. It is what distinguishes "this
    # setting does more work per step" from "this setting needs more steps of work", and the
    # budget A/B needs that distinction -- a starved basis can be slower while building a
    # smaller expansion, which no timing alone can attribute.
    nmv = Ref(0)
    function apply_s(x)
        nmv[] += 1
        theta = to_concrete((U_aug * x) * V_aug)
        evolved = apply_gate(gate, theta, tl, tr)
        proj = contract(U_aug', (1, 2), evolved, (1, 2))
        return to_concrete(contract(proj, (2, 3), V_aug', (2, 3)))
    end

    S_new = expv(apply_s, tau_s, S_start; hermitian = true, maxiter = maxiter, tol = tol)

    res = svd(S_new, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true)
    tag = f.link_mid.itags
    left_core  = to_concrete(setitag(to_concrete(U_aug * res.U), 3, tag))
    right_core = to_concrete(setitag(to_concrete((res.S * res.Vd) * V_aug), 1, tag))

    svals = [s for (s, _, _, _) in res.kept_list]
    return (; left_core, right_core, U_aug, V_aug, S_new,
            n_new_k = ex.n_new_l, n_new_l = ex.n_new_r,
            keep = leg_dim(left_core, 3),
            aug_k = leg_dim(U_aug, 3), aug_l = leg_dim(V_aug, 1),
            svals, discarded = _trunc_weight(res),
            err_pre = ex.err_pre, err_fnl = ex.err_fnl, n_matvec = nmv[])
end

cbe_bond_update(f::BondFrame, gate, tau::Number; kwargs...) =
    cbe_bond_update(f, gate, ComplexF64(tau); kwargs...)

# ── the sweep ────────────────────────────────────────────────────────────────

"""
    cbe_bond_update_sweep!(psi, gates, parity, tau; kwargs...) -> NamedTuple

`BondUpdateBUG.parity_sweep!` with [`cbe_bond_update`](@ref) at each bond. Returns
`(; aug_k, aug_l, discarded, err_pre, err_fnl)`, each the maximum over the group's bonds.

The sweep itself is copied unchanged from the validated one, including the reason it needs no
re-canonicalisation: the bond update returns `left_core` already a left isometry with the
centre in `right_core`, so `psi.center = i + 1` is exact and the next bond of the group is one
move to the right. Bonds of one parity act on disjoint site pairs, so the group is an EXACT
factor of the Trotter step.
"""
function cbe_bond_update_sweep!(psi::SymMPS, gates, parity::Symbol, tau::ComplexF64; kwargs...)
    aug_k = 0; aug_l = 0
    discarded = 0.0; err_pre = 0.0; err_fnl = 0.0
    n_matvec = 0; aug_sum = 0; nbond = 0
    for i in parity_bonds(length(psi), parity)
        gates[i] === nothing && continue
        canonical!(psi, i)
        f = bond_frame(psi, i)
        r = cbe_bond_update(f, gates[i], tau; kwargs...)
        n_matvec += r.n_matvec
        aug_sum += r.aug_k * r.aug_l          # the SIZE each of those matvecs acted on
        nbond += 1
        psi[i] = r.left_core
        psi[i + 1] = r.right_core
        psi.center = i + 1
        aug_k = max(aug_k, r.aug_k)
        aug_l = max(aug_l, r.aug_l)
        discarded = max(discarded, r.discarded)
        err_pre = max(err_pre, r.err_pre)
        err_fnl = max(err_fnl, r.err_fnl)
    end
    return (; aug_k, aug_l, discarded, err_pre, err_fnl, n_matvec, aug_sum, nbond)
end

cbe_bond_update_sweep!(psi::SymMPS, gates, parity::Symbol, tau::Number; kwargs...) =
    cbe_bond_update_sweep!(psi, gates, parity, ComplexF64(tau); kwargs...)

# ── the driver ───────────────────────────────────────────────────────────────

"""
    cbe_bond_update_bug!(psi, gates; opts=CBEBugOptions()) -> CBEBugRunInfo

Evolve `psi` in place with the gate-based CBE-BUG for `opts.n_steps` steps of `opts.dt`.

`opts.order` is not a field of `CBEBugOptions`; the composition here is Strang
(`even tau/2, odd tau, even tau/2`), second order, matching `bond_update_bug!`'s default.
One RNG threads the whole run, so a run is reproducible while consecutive steps still draw
different sketches.

REAL TIME ONLY: `tau = -im*dt`, formed here.
"""
function cbe_bond_update_bug!(psi::SymMPS, gates; opts::CBEBugOptions = CBEBugOptions())
    rng = MersenneTwister(opts.seed)
    tau = -im * opts.dt
    kw = (; maxdim = opts.maxdim, trunc_thresh = opts.trunc_thresh,
            maxiter = opts.maxiter, tol = opts.tol, rng = rng,
            dex = opts.dex, growth = opts.growth,
            dover = opts.dover, comp_ratio = opts.comp_ratio,
            sulz_cap = opts.sulz_cap, preselect_only = opts.preselect_only)

    times = Float64[]; norms = Float64[]
    bds = Vector{Int}[]; maxbd = Int[]
    expanded = Vector{Int}[]; maxexp = Int[]; centres = Int[]
    epre = Float64[]; efnl = Float64[]; disc = Float64[]
    mags = Vector{Float64}[]; obs = Any[]
    kdim = Int[]; maug = Float64[]

    for step in 1:opts.n_steps
        a = cbe_bond_update_sweep!(psi, gates, :even, tau / 2; kw...)
        b = cbe_bond_update_sweep!(psi, gates, :odd,  tau;     kw...)
        c = cbe_bond_update_sweep!(psi, gates, :even, tau / 2; kw...)

        nrm = norm(psi)
        opts.normalize && nrm > 0 && (psi[psi.center] =
            to_concrete((1.0 / nrm) * psi[psi.center]))

        push!(times, step * opts.dt); push!(norms, nrm)
        d = bond_dims(psi); push!(bds, d); push!(maxbd, maximum(d; init = 0))
        push!(centres, d[center_bond(psi)])
        aug = [a.aug_k, a.aug_l, b.aug_k, b.aug_l, c.aug_k, c.aug_l]
        push!(expanded, aug); push!(maxexp, maximum(aug; init = 0))
        push!(epre, max(a.err_pre, b.err_pre, c.err_pre))
        push!(efnl, max(a.err_fnl, b.err_fnl, c.err_fnl))
        push!(disc, max(a.discarded, b.discarded, c.discarded))
        # SUMMED over the three parity sweeps: the gate path solves one exponential per bond,
        # so the step's cost is the total, not a maximum. `mean_aug` is the mean expanded
        # AREA (aug_k*aug_l) those solves acted on, in the same units for both paths.
        push!(kdim, a.n_matvec + b.n_matvec + c.n_matvec)
        nb = a.nbond + b.nbond + c.nbond
        push!(maug, nb == 0 ? 0.0 : (a.aug_sum + b.aug_sum + c.aug_sum) / nb)
        opts.record_magnetisation && push!(mags, magnetisation(copy(psi)))
        opts.observe === nothing || push!(obs, opts.observe(copy(psi), step, step * opts.dt))
    end

    return CBEBugRunInfo(times, norms, bds, maxbd, expanded, maxexp, centres,
                         epre, efnl, disc, mags, obs, kdim, maug)
end
