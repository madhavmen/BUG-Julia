# ── one-site CBE-DMRG: a line-for-line port of DMRGSweepCBE1Si.m ──
# There is no 0-site backward update, unlike the TDVP sweep: DMRG minimises, so a backward step would
# not converge to anything.
# The expansion is mandatory, since a one-site update cannot change a bond dimension.

"""
    DMRGCBEInfo

What one [`dmrg_cbe1s_sweep!`](@ref) half-sweep pair did.

  - `energy`: the lowest Ritz value in the sweep, a variational upper bound. Below the true ground
    energy it is a bug, not a lucky sweep.
  - `energies`: the Ritz value at each local solve; their spread is the convergence signal.
  - `bond_dims` (multiplets), `max_expanded`, `n_new`, `err_pre`, `err_fnl`, `discarded`,
    `krylov_dims`, and `ledger` ([`AdaptiveLedger`](@ref)).
"""
struct DMRGCBEInfo
    energy::Float64
    energies::Vector{Float64}
    bond_dims::Vector{Int}
    max_expanded::Int
    n_new::Vector{Int}
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    ledger::AdaptiveLedger
end

"""
Default `tol_basis` / `tol_trunc` of the ADAPTIVE CBE-DMRG -- medium-loose, NOT the time step's
`TOL_DEFAULTS` (1e-5 / 1e-7). A DMRG sweep is not a one-shot step: every bond is re-expanded and
re-solved on the next sweep, so a direction a loose sweep skips is offered again, and a tight
per-sweep budget mostly buys rank and matvecs the next sweep throws away. PROVISIONAL until
`benchmarks/ground_state/adaptive_dmrg_tolerance.jl` measures how loose each may be at a given
energy accuracy (results: `ADAPTIVE_DMRG_TOLERANCE_RESULTS.md`).
"""
const DMRG_TOL_DEFAULTS = (basis = 1e-3, trunc = 1e-4)

"""
    dmrg_etol(tol_basis, tol_trunc, N) -> Float64

The adaptive CBE-DMRG's sweep-to-sweep stopping threshold, DERIVED from its two tolerances
rather than set as a fourth accuracy knob. MEASURED (ADAPTIVE_DMRG_TOLERANCE_RESULTS.md, N = 20
Heisenberg / XX chains and 5x4 square cylinder): the returned state's energy error obeys
`dE/N <~ 2.5e-3 tol_basis^2 + 0.2 tol_trunc^2` (largest coefficients seen). The run stops once a
sweep changes the energy by less than a tenth of that, since further sweeps cannot buy more than
the tolerances allow. Floor `1e-12 N` (round-off). A fixed `etol = 1e-10` made loose settings
sweep up to the 40-sweep limit (34-40 sweeps at `tol_basis = 0.1`).
"""
dmrg_etol(tb::Real, tt::Real, N::Int) =
    max(0.1 * N * (2.5e-3 * tb^2 + 0.2 * tt^2), 1e-12 * N)

"""
    dmrg_cbe1s_sweep!(psi, mpo; kwargs...) -> DMRGCBEInfo

One full CBE-DMRG sweep (left to right, then right to left), in place, with the RSVD expansion engine
of `cbe_bug_step!`. Accuracy is set by `tol_basis`, `tol_trunc` and `maxdim`; the centre ends on site 1.

`tol_basis` is taken at unit time, so `tol_basis^2 / gap` bounds the energy error a met sweep can hide.

`restol = 1e-6` and `comp_ratio = 1.0` measured best. Do not copy `restol` to a time integrator: DMRG
re-solves each site on the next sweep, whereas a time step gets one shot and needs the Krylov depth.
"""
function dmrg_cbe1s_sweep!(psi::SymMPS, mpo::MPO;
                           tol_basis::Union{Nothing, Float64} = nothing,
                           tol_trunc::Union{Nothing, Float64} = nothing,
                           exact::Bool = false,
                           maxdim::Int = 128,
                           maxiter::Int = 30,
                           restol::Float64 = 1e-6,
                           seed_expanded::Bool = false,
                           # Adaptive search width from Test B (`_certified_range`); `false` = old.
                           thin_search::Bool = true,
                           rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("dmrg_cbe1s_sweep! needs at least two sites"))

    # The GROUND-STATE defaults, not the time step's: a sweep is repeated until the energy
    # settles, so each one need not be tight (see `DMRG_TOL_DEFAULTS`).
    budget = AdaptiveBudget(something(tol_basis, DMRG_TOL_DEFAULTS.basis),
                            something(tol_trunc, DMRG_TOL_DEFAULTS.trunc), TOL_DEFAULTS.expv)
    energies = Float64[]
    n_new = zeros(Int, L - 1)
    maxexp = 0; epre = 0.0; efnl = 0.0; disc = 0.0; nmv = 0

    rmax = maximum(bond_dims(psi); init = 0)    # read before the gauge, as it always was
    canonical!(psi, 1)
    ledger = AdaptiveLedger()
    # Scale-free indicators: the Lanczos solves return unit vectors anyway, so normalising the
    # start only makes the first bond's residual comparable with the rest.
    n0 = norm(psi[1])
    n0 > 0 && (psi[1] = to_concrete((1.0 / n0) * psi[1]))
    alloc = basis_allocation(budget, L - 1; tau = 1.0, nrm = 1.0)
    share = budget.trunc / sqrt(L - 1)          # each split's relative discarded-norm share
    swept = [0.0, 0.0]                          # squared discarded norm, per half-sweep

    # `rmax` read ONCE, before anything widens, so the work ceiling is a statement about the
    # state rather than about whichever bond is being looked at. The preselection cuts only at
    # the numerical rank floor (`stol_pre = 0`, plan §1); Tests A/B against `tol_basis` decide
    # what matters.
    exkw = (growth = ADAPTIVE_GROWTH, comp_ratio = 1.0, stol_pre = 0.0, rmax = rmax,
            exact = exact, thin_search = thin_search, rng = rng, alloc = alloc,
            workcap = work_ceiling(maxdim, rmax))

    function record!(info)
        maxexp = max(maxexp, info.dim_out)
        n_new[info.bond] += info.n_new
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
    end

    # The budgeted split after the local solve. Logs the bond (Tests A/B from its expansion, plus
    # what the split discarded) into the sweep ledger.
    function split!(A, legs, i, bondlg, info, half)
        res, capped = _budget_svd(A, legs, maxdim, n2 -> share * sqrt(n2))
        capped && (bondlg.maxdim_reached = true; _unmet!(bondlg, :trunc))
        w = _trunc_weight(res)
        disc = max(disc, w)
        swept[half] += w^2
        _log_bond!(ledger, bondlg, i, info.dir, info.dim_out)
        ledger.bond_log[end] = merge(ledger.bond_log[end],
                                     (r_old = info.dim_in, r_final = length(res.kept_list),
                                      discarded = w))
        return res
    end

    """
    The one-site eigensolve at site `j`, in the environments given.

    `seed_expanded` changes only the Lanczos start vector, never the operator or the space, so the Ritz
    value stays a variational upper bound (see [`lanczos_lowest`](@ref)). The default start is the
    zero-padded old vector, with no component along the directions the expansion just admitted;
    seeding with `v + eta * P_new(H v)` puts them in the space from the first vector.
    """
    function solve1(j, lch, rch; expanded::Bool = false)
        H1 = one_site_h(mpo, j, lch, rch)
        apply = x -> apply_one_site(H1, x)
        x0 = psi[j]
        if expanded && seed_expanded
            w = apply(x0)                      # the one extra application, counted below
            nmv += 1
            # Project out the current direction: what remains is what H mixes in from outside the old frame.
            # The denominator must be real: `tensor_inner` returns ComplexF64, and `max` on complex numbers is a
            # MethodError, so `norm(x0)^2` is used.
            ip = tensor_inner(x0, w) / max(norm(x0)^2, eps())
            r = to_concrete(w + (-ip) * x0)
            nr = norm(r)
            if nr > 1e-14
                # 0.1 of the current norm: enough to enter the Krylov space at iteration 1,
                # small enough that the start is still essentially the current state.
                x0 = to_concrete(x0 + (0.1 * norm(x0) / nr) * r)
            end
        end
        A, e, kd = lanczos_lowest(apply, x0; maxiter = maxiter, restol = restol)
        nmv += kd
        push!(energies, e)
        return A
    end

    # ── left to right ──
    # The right stack is prebuilt once and read at link i+2, which depends only on sites >= i+2. Reading it
    # backwards silently minimises the wrong Hamiltonian, since a stale environment does not throw.
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    for i in 1:(L - 1)
        bondlg = AdaptiveLedger()                           # this bond's Tests A/B (§8)
        rch1, info = expand_bond!(psi, mpo, i, :right, lch,
                                  right_channels(rstack, i + 2); exkw..., ledger = bondlg)
        record!(info)

        A = solve1(i, lch, rch1; expanded = true)
        # Split with the amplitude moving RIGHT, truncating: this is the only place rank is
        # removed, and `Nkeep` is in MULTIPLETS under a non-abelian symmetry.
        res = split!(A, (1, 2), i, bondlg, info, 1)
        tag = psi[i].inds[3].itags
        C = to_concrete(res.S * res.Vd)
        psi[i]     = to_concrete(setitag(res.U, 3, tag))
        psi[i + 1] = to_concrete(setitag(contract(C, (2,), psi[i + 1], (1,)), 1, tag))
        psi.center = i + 1

        lch = push_left_channels(lch, mpo, psi[i], i)
    end
    # Site L: the reference's `'vv'` -- solve and keep, no split, because there is no bond to
    # its right to move an orthogonality centre across.
    psi[L] = solve1(L, lch, boundary_channels(mpo))
    psi.center = L

    # ======== right to left ========
    # Built AFTER the closing solve, which rewrote only `psi[L]`; this stack reads sites
    # `1 … L-2`.
    lstack = left_env_stack(psi, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    for i in (L - 1):-1:1
        bondlg = AdaptiveLedger()
        lch1, info = expand_bond!(psi, mpo, i, :left,
                                  left_channels(lstack, i), rch; exkw..., ledger = bondlg)
        record!(info)

        A = solve1(i + 1, lch1, rch; expanded = true)
        res = split!(A, (1,), i, bondlg, info, 2)
        tag = psi[i].inds[3].itags
        C = to_concrete(res.U * res.S)
        psi[i + 1] = to_concrete(setitag(res.Vd, 1, tag))
        psi[i]     = to_concrete(setitag(contract(psi[i], (3,), C, (1,)), 3, tag))
        psi.center = i

        rch = push_right_channels(rch, mpo, psi[i + 1], i + 1)
    end
    psi[1] = solve1(1, boundary_channels(mpo), rch)
    psi.center = 1

    # Each half-sweep cuts every bond once, against a unit-norm centre; the larger of the two
    # accumulations is what the sweep's splits cost the state (the `tol_trunc` budget).
    ledger.discarded = sqrt(maximum(swept))
    ledger.r_old = rmax
    ledger.r_aug = maxexp
    ledger.r_final = maximum(bond_dims(psi); init = 0)
    return DMRGCBEInfo(minimum(energies; init = Inf), energies, bond_dims(psi),
                       maxexp, n_new, epre, efnl, disc, nmv, ledger)
end

dmrg_cbe1s_sweep!(psi::SymMPS, h::XXZChain; kwargs...) =
    dmrg_cbe1s_sweep!(psi, mpo_from_terms(h); kwargs...)
