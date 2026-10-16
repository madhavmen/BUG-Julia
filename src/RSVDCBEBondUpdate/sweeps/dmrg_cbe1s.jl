# DMRG WITH CONTROLLED BOND EXPANSION, ONE SITE AT A TIME.
#
# A LINE-FOR-LINE PORT of `cbe_matlab/DMRGSweepCBE1Si.m`. The correspondence is exact and
# worth writing down, because "does our code do what the reference does" is a question that
# should be answerable by reading rather than by measuring:
#
#   MATLAB (`>>` half-sweep, si = 1 … L-1)                     here
#   -------------------------------------------------------    ------------------------------
#   VR = loadV(EnV.VR{si+1})                                   right_channels(rstack, i+2)
#   [TL,TR,VR,InfoCBE] = CBE(TL,TR,VL,VR,HL,HR,'>>')           expand_bond!(…, :right, …)
#   [TL,VL,OrthC,Info1Si] = Update1Si(TL,VL,VR,HL,'>>')        lanczos_lowest + truncated svd
#   MPS.T{si} = TL;  EnV.VL{si+1} = VL                         psi[i] = res.U;  push_left…
#   TL = AbsorbOrthCent(OrthC, TR, '>>')                       C absorbed into psi[i+1]
#   … closing `[TL,Info1Si] = Update1Si(TL,VL,VR,HL,'vv')`     the site-L solve, no split
#
# WHAT DMRG DOES **NOT** DO, and it is the one structural difference from the TDVP sweep next
# door: THERE IS NO 0-SITE BACKWARD UPDATE. `DMRGSweepCBE1Si.m` goes straight from `Update1Si`
# to `AbsorbOrthCent` (lines 31→40), where `TDVPSweepCBE1Si.m` puts an `Update0Si` between them
# (line 42). The reason is not an oversight in either: TDVP's backward half-step exists to
# cancel the double counting of the bond between two consecutive forward site updates, which is
# a statement about a TIME step. DMRG has no time step to over-count -- each local solve simply
# minimises, and the sweep's fixed point is the variational minimum. Adding a 0-site step here
# would not be a small error; it would not converge to anything.
#
# WHY THE EXPANSION IS MANDATORY, not an accelerator. A one-site update CANNOT change a bond
# dimension: it varies a single tensor with both its bond legs fixed. Without CBE this sweep is
# frozen at whatever rank it started with -- from a product state, at `chi = 1` forever. That is
# the same failure the `grow_iters = 0` measurement in `modes/ground_state.jl` records (energy
# frozen at the Neel product value `-1.75`, "converged" in two sweeps). The expansion is where
# ALL rank generation happens, so `info.converged` is not by itself evidence of anything.
#
# ONE-SITE, NOT TWO-SITE, IS THE POINT. Two-site DMRG gets its rank from the two-site block and
# pays `O(d^2 chi^3)` per solve; this pays `O(d chi^3)` at one site plus a thin sketch, and gets
# its rank from the expansion instead. That trade is the whole reason CBE exists
# (`mainDMRG_CBE.m:84`: CBE is switched on only once `Nkeep > D_CBE`, i.e. once the bond
# dimension is large enough for the two-site cost to hurt).

"""
    DMRGCBEInfo

What one [`dmrg_cbe1s_sweep!`](@ref) half-sweep pair did.

  - `energy` -- the LOWEST Ritz value seen anywhere in the sweep. A variational upper bound on
    the ground energy, because every local solve is an exact Rayleigh-Ritz in a fixed subspace
    (see [`lanczos_lowest`](@ref)). **A reported energy below the true ground energy is a bug,
    not a lucky sweep.**
  - `energies` -- the Ritz value at every local solve, in sweep order. The spread across a
    sweep is the reference's `err_swp` (`mainDMRG_CBE.m:131-137`, `std(egList)`) and is the
    honest convergence signal: a converged sweep reports the same number at every bond.
  - `bond_dims` -- after the sweep, in MULTIPLETS.
  - `max_expanded` -- the widest any bond was expanded to before truncation.
  - `n_new` -- directions admitted per bond, summed over both half-sweeps.
  - `err_pre`, `err_fnl`, `discarded` -- the largest preselection / final-selection /
    truncation weight over the sweep.
  - `krylov_dims` -- total operator applications in the local eigensolves.
  - `ledger` -- the adaptive path's [`AdaptiveLedger`](@ref) for the sweep (Tests A/B per bond in
    `bond_log`, the split's discarded norm, `:budget_unmet` reasons); `nothing` on the legacy path.
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
    ledger::Union{Nothing, AdaptiveLedger}
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

One full CBE-DMRG sweep (left to right, then right to left), in place.

## Adaptive path (the default): `tol_basis`, `tol_trunc`, `maxdim`

The SAME RSVD engine `cbe_bug_step!` uses (simulation_plan_2 Task 4: "the same adaptive RSVD
discovery ... may be shared with the ground-state optimizer"). At every bond `cbe_expand` runs
with the step's [`BasisAllocation`](@ref): sector coverage, Test B (holdout sampling residual,
targeted widening) decides how much to SAMPLE and Test A (projected singular tail) how many
directions to KEEP, under the [`work_ceiling`](@ref) derived from `maxdim`. No `dex`, `dover`,
`growth`, `comp_ratio` or `stol_pre`.

⚠ WHAT `tol_basis` MEASURES HERE. There is no time step, so the allocation is taken at unit
time (`tau = 1`): a bond's indicator is `||(I - P) H Theta|| / ||psi||` in ENERGY units of `H`
-- the part of `H`'s action that leaves the expanded two-site space. Summed in quadrature over
bonds that is the two-site variance estimate, so `tol_basis^2 / gap` is the scale of energy
error a met sweep can still hide. A ground state at finite `maxdim` keeps a nonzero two-site
variance, so a `tol_basis` tighter than that reports `:budget_unmet` (Test A at the work
ceiling) instead of pretending -- the variational energy is unaffected either way.

`tol_trunc` budgets each split's discarded RELATIVE norm at `tol_trunc / sqrt(L - 1)` (the
rule of [`_budget_svd`](@ref)); `maxdim` binding past that share is `:trunc` in the ledger.
`restol` / `maxiter` stay: the local stopping test of an EIGENSOLVE is the Ritz residual, not a
basis tolerance (plan Task 4).

## Legacy path

Naming any of `dex`, `growth`, `dover`, `comp_ratio`, `trunc_thresh` selects the historical
fixed-knob selection (`ledger = nothing`), exactly as before; mixing them with `tol_basis` /
`tol_trunc` is refused. `exact = true` forms the full fused basis (the A/B reference arm) on
either path. `maxiter` / `restol` bound the local eigensolve (the reference's `K` / `Ktol`).

Leaves the orthogonality centre at site 1, so consecutive sweeps chain with no extra
canonicalisation.

## Two defaults that differ from the other CBE sweeps, and the measurement behind each

Both were set on the square 5x4 `N = 20` cylinder at `D = 128` (`benchmarks/ground_state/`),
scored against the exact `S^z = 0` sparse ground state, `E0/N = -0.649788994793`.

`restol = 1e-6`, not the `1e-10` the TDVP arms use. Accuracy is FLAT across that axis
(`dev/site` 5.023e-07 / 5.039e-07 / 5.037e-07 at `1e-6` / `1e-8` / `1e-10`) while the local
eigensolve cost halves (2704 vs 5467 matvec) -- a free 2x.

⚠️ THE SAME KNOB POINTS THE OPPOSITE WAY IN A TIME INTEGRATOR, so do not copy this to
`tdvp_cbe1s_sweep!`. DMRG re-solves every site on the next sweep, so an over-tight local
tolerance is thrown away; a time step gets ONE shot per site and the Krylov depth is the
accuracy -- see the `krylov-depth-is-a-hidden-error-source` finding.

`comp_ratio = 1.0`, not the reference's `0.5`. Best on BOTH deterministic columns here:
`dev/site` 5.037e-07 vs 5.080e-07 (`0.5`) and 5.303e-07 (`0.25`), and fewest matvec
(5467 vs 5585 / 5720). It also agrees with `cbe_bug_step!`, which already used `1.0`.

⚠️ `dex` STAYS AT `0` (i.e. the `growth` schedule) EVEN THOUGH `dex = 8` MEASURED FASTER
here (74.3 s vs 183.6 s at `dex = 16`, same energy). `dex` is an ABSOLUTE per-side budget,
so a value tuned at one cap does not scale: `err_fnl` was already 0.556 at `D = 128`,
meaning the expansion was discarding half its ranked candidates, and it starves harder at
larger `D`. The reference's own rule is proportional (`Dex = ceil(rDex*Nkeep)`, a fraction
of the CAP). Set `dex` at the DRIVER, where the cap is known; do not bake it in here.
"""
function dmrg_cbe1s_sweep!(psi::SymMPS, mpo::MPO;
                           tol_basis::Union{Nothing, Float64} = nothing,
                           tol_trunc::Union{Nothing, Float64} = nothing,
                           # Legacy knobs: `nothing` = not named (see `resolve_budget`).
                           dex::Union{Nothing, Int} = nothing,
                           growth::Union{Nothing, Float64} = nothing,
                           dover::Union{Nothing, Int} = nothing,
                           comp_ratio::Union{Nothing, Float64} = nothing,
                           trunc_thresh::Union{Nothing, Float64} = nothing,
                           sulz_cap::Bool = false,
                           preselect_only::Bool = false,
                           exact::Bool = false,
                           maxdim::Union{Nothing, Int} = nothing,
                           maxiter::Int = 30,
                           restol::Float64 = 1e-6,
                           seed_expanded::Bool = false,
                           rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("dmrg_cbe1s_sweep! needs at least two sites"))

    # `maxiter`/`restol` are the EIGENSOLVE's controls, not the BUG root solve's legacy
    # `maxiter`, so they are not offered to `resolve_budget`.
    budget, adaptive, legacy_names = resolve_budget(tol_basis, tol_trunc, nothing,
        (dex = dex, dover = dover, comp_ratio = comp_ratio, growth = growth,
         trunc_thresh = trunc_thresh))
    _legacy_hint(legacy_names)
    # The GROUND-STATE defaults, not the time step's: a sweep is repeated until the energy
    # settles, so each one need not be tight (see `DMRG_TOL_DEFAULTS`).
    adaptive && (budget = AdaptiveBudget(something(tol_basis, DMRG_TOL_DEFAULTS.basis),
                                         something(tol_trunc, DMRG_TOL_DEFAULTS.trunc),
                                         budget.expv))
    maxdim = something(maxdim, adaptive ? 128 : 200)
    cut = max(legacy_or_default(trunc_thresh, :trunc_thresh), 1e-14)
    energies = Float64[]
    n_new = zeros(Int, L - 1)
    maxexp = 0; epre = 0.0; efnl = 0.0; disc = 0.0; nmv = 0

    rmax = maximum(bond_dims(psi); init = 0)    # read before the gauge, as it always was
    canonical!(psi, 1)
    ledger = adaptive ? AdaptiveLedger() : nothing
    if adaptive
        # Scale-free indicators: the Lanczos solves return unit vectors anyway, so normalising
        # the start only makes the first bond's residual comparable with the rest.
        n0 = norm(psi[1])
        n0 > 0 && (psi[1] = to_concrete((1.0 / n0) * psi[1]))
    end
    alloc = basis_allocation(budget, L - 1; tau = 1.0, nrm = 1.0)
    share = budget.trunc / sqrt(L - 1)          # each split's relative discarded-norm share
    swept = [0.0, 0.0]                          # squared discarded norm, per half-sweep

    # `rmax` read ONCE, before anything widens, so an (optional) global rank bound is a
    # statement about the state rather than about whichever bond is being looked at.
    exkw = (dex = legacy_or_default(dex, :dex), growth = legacy_or_default(growth, :growth),
            dover = dover, comp_ratio = legacy_or_default(comp_ratio, :comp_ratio),
            sulz_cap = sulz_cap, rmax = rmax, preselect_only = preselect_only, exact = exact,
            rng = rng,
            # Plan §1: on the adaptive path the preselection cuts only at the numerical rank
            # floor; Tests A/B against `tol_basis` decide what matters.
            stol_pre = adaptive ? 0.0 : legacy_or_default(nothing, :stol_pre),
            alloc = adaptive ? alloc : nothing,
            workcap = adaptive ? work_ceiling(maxdim, rmax) : 0)

    function record!(info)
        maxexp = max(maxexp, info.dim_out)
        n_new[info.bond] += info.n_new
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
    end

    # The split after the local solve: Telum's relative cut on the legacy path, the budgeted
    # rule on the adaptive one. Logs the bond (Tests A/B from its expansion, plus what the split
    # discarded) into the sweep ledger.
    function split!(A, legs, i, bondlg, info, half)
        res = if adaptive
            r, capped = _budget_svd(A, legs, maxdim, n2 -> share * sqrt(n2))
            capped && (bondlg.maxdim_reached = true; _unmet!(bondlg, :trunc))
            r
        else
            svd(A, legs; cutoff = cut, Nkeep = maxdim, get_lists = true)
        end
        w = _trunc_weight(res)
        disc = max(disc, w)
        if adaptive
            swept[half] += w^2
            _log_bond!(ledger, bondlg, i, info.dir, info.dim_out)
            ledger.bond_log[end] = merge(ledger.bond_log[end],
                                         (r_old = info.dim_in, r_final = length(res.kept_list),
                                          discarded = w))
        end
        return res
    end

    """
    The one-site eigensolve at site `j`, in the environments given.

    `seed_expanded` changes ONLY the Lanczos START VECTOR, never the operator or the space, so
    the Ritz value stays a variational upper bound either way (see [`lanczos_lowest`](@ref)).

    WHY A DIFFERENT START HELPS. `expand_bond!` widens the bond "leaving the state unchanged" --
    the admitted directions are appended with ZERO amplitude. So the default start `psi[j]` is
    the old vector zero-padded, and it has no component along precisely the directions the
    expansion just paid to find. Lanczos still reaches them, but only through the operator: they
    first appear in `H v`, i.e. from the SECOND Krylov vector, and the Ritz vector needs further
    iterations to weight them properly. Seeding with `v + eta * P_new(H v)` puts them in the
    space from the FIRST vector.

    ⚠ It is not free: the seed costs ONE extra operator application per solve, counted in `nmv`
    below like any other, so a fair reading of the result is applications-to-accuracy and not
    sweeps alone. `eta` is small and the seed is renormalised, so the start stays dominated by
    the current state -- this is a perturbation of the start direction, not a different ansatz.
    """
    function solve1(j, lch, rch; expanded::Bool = false)
        H1 = one_site_h(mpo, j, lch, rch)
        apply = x -> apply_one_site(H1, x)
        x0 = psi[j]
        if expanded && seed_expanded
            w = apply(x0)                      # the one extra application, counted below
            nmv += 1
            # Project out the current direction: what remains is exactly the content H mixes in
            # from OUTSIDE the old frame, which is where the new bond directions live.
            #
            # ⛔ THE DENOMINATOR MUST BE REAL. `tensor_inner` returns `ComplexF64` even for a
            # self-overlap whose imaginary part is zero by construction, and `max(::Float64,
            # ::ComplexF64)` is a MethodError, not a promotion -- complex numbers have no total
            # order. `norm(x0)^2` is the same quantity, real by type rather than by accident.
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

    # ======== left to right ========
    #
    # The right stack is prebuilt ONCE and read at link `i+2`, which depends only on sites
    # `>= i+2`; the sweep has written only `psi[i]` and `psi[i+1]` by then. Getting this
    # backwards reads a STALE environment, which does not throw -- it silently minimises the
    # wrong Hamiltonian.
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    for i in 1:(L - 1)
        bondlg = adaptive ? AdaptiveLedger() : nothing      # this bond's Tests A/B (§8)
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
        bondlg = adaptive ? AdaptiveLedger() : nothing
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

    if adaptive
        # Each half-sweep cuts every bond once, against a unit-norm centre; the larger of the
        # two accumulations is what the sweep's splits cost the state (the `tol_trunc` budget).
        ledger.discarded = sqrt(maximum(swept))
        ledger.r_old = rmax
        ledger.r_aug = maxexp
        ledger.r_final = maximum(bond_dims(psi); init = 0)
    end
    return DMRGCBEInfo(minimum(energies; init = Inf), energies, bond_dims(psi),
                       maxexp, n_new, epre, efnl, disc, nmv, ledger)
end

dmrg_cbe1s_sweep!(psi::SymMPS, h::XXZChain; kwargs...) =
    dmrg_cbe1s_sweep!(psi, mpo_from_terms(h); kwargs...)

# ── driver ───────────────────────────────────────────────────────────────────

"""
    DMRGCBERun

Per-sweep history of [`dmrg_cbe1s!`](@ref). `energies[k]` is sweep `k`'s best Ritz value and
`sweep_spread[k]` the standard deviation across that sweep's local solves -- the reference's
`err_swp`, which is the convergence signal to read rather than the energy change alone: a
sweep whose bonds disagree has not converged even if its best value stopped moving.
"""
struct DMRGCBERun
    energies::Vector{Float64}
    sweep_spread::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    # Under SU(2) `bond_dims` counts MULTIPLETS, so a reader comparing ranks across symmetry
    # modes needs the state count too -- recorded per sweep, because taking it from the final
    # `psi` afterwards reports one number for the whole history and hides the growth.
    state_bond_dims::Vector{Vector{Int}}
    max_state_bond_dims::Vector{Int}
    # Wall-clock seconds for each sweep on its own. The FIRST entry of a fresh session carries
    # Julia's compilation of the whole sweep stack (order 10^2 s here) and is NOT a cost datum.
    sweep_seconds::Vector{Float64}
    max_expanded::Vector{Int}
    err_pre::Vector{Float64}
    err_fnl::Vector{Float64}
    discarded::Vector{Float64}
    krylov_dims::Vector{Int}
    converged::Bool
    # Per sweep: `budget_status` (`:met` / `:budget_unmet` / `:legacy`) and the sweep's ledger
    # (`nothing` on the legacy path), so a converged energy carries its basis verdict with it.
    status::Vector{Symbol}
    ledgers::Vector{Union{Nothing, AdaptiveLedger}}
end

Base.length(r::DMRGCBERun) = length(r.energies)

"Sample standard deviation, written out rather than pulling in `Statistics` for one call."
function _spread(v::Vector{Float64})
    n = length(v)
    n < 2 && return 0.0
    m = sum(v) / n
    return sqrt(sum((x - m)^2 for x in v) / (n - 1))
end

"""
    dmrg_cbe1s!(psi, h; n_sweeps=20, etol=nothing, kwargs...) -> DMRGCBERun

Ground state of `h` by CBE-DMRG sweeps, in place. `h` may be an `MPO` or an `XXZChain`.

Convergence is tested on the SWEEP-BEST energy against the previous sweep, not bond to bond:
within a sweep the energy falls by construction, so a per-bond test stops early on any flat
stretch.

`maxdim` may be raised between sweeps by passing a schedule -- see the reference's `DList`
(`mainDMRG_CBE.m:81`), which ramps the bond dimension over sweeps and only switches CBE on
once `Nkeep > D_CBE`. Here the schedule is the caller's business; this function runs one
`maxdim`.

**Under U(1) the particle-number sector is pinned by the initial state**, and under SU(2) the
total spin is, so this finds the ground state OF THAT SECTOR. `dimer_state` is the `S = 0`
sector, which is where the Heisenberg and Haldane-Shastry ground states live.
"""
function dmrg_cbe1s!(psi::SymMPS, h::Union{XXZChain, MPO};
                     n_sweeps::Int = 20, etol::Union{Nothing, Float64} = nothing, kwargs...)
    mpo = h isa MPO ? h : mpo_from_terms(h)
    kw = values(kwargs)
    # The stopping threshold, decided after the first sweep says which path ran: an explicit
    # `etol` always wins; the legacy path keeps its historical 1e-10; the adaptive path derives it
    # from its own tolerances (`dmrg_etol`).
    etol_of(info) = etol !== nothing ? etol :
        info.ledger === nothing ? 1e-10 :
        dmrg_etol(something(get(kw, :tol_basis, nothing), DMRG_TOL_DEFAULTS.basis),
                  something(get(kw, :tol_trunc, nothing), DMRG_TOL_DEFAULTS.trunc), length(psi))
    es = Float64[]; spread = Float64[]; bds = Vector{Int}[]; maxbd = Int[]
    sbds = Vector{Int}[]; maxsbd = Int[]; secs = Float64[]
    maxexp = Int[]; epre = Float64[]; efnl = Float64[]; disc = Float64[]; kd = Int[]
    status = Symbol[]; ledgers = Union{Nothing, AdaptiveLedger}[]
    converged = false
    prev = Inf

    for _ in 1:n_sweeps
        tsw = time()
        info = dmrg_cbe1s_sweep!(psi, mpo; kwargs...)
        push!(secs, time() - tsw)
        push!(es, info.energy)
        push!(spread, _spread(info.energies))
        push!(bds, info.bond_dims); push!(maxbd, maximum(info.bond_dims; init = 0))
        # `psi` is the state THIS sweep produced, so this is a per-sweep reading.
        sbd = state_bond_dims(psi)
        push!(sbds, sbd); push!(maxsbd, maximum(sbd; init = 0))
        push!(maxexp, info.max_expanded)
        push!(epre, info.err_pre); push!(efnl, info.err_fnl)
        push!(disc, info.discarded); push!(kd, info.krylov_dims)
        push!(status, budget_status(info.ledger)); push!(ledgers, info.ledger)
        if abs(info.energy - prev) < etol_of(info)
            converged = true
            break
        end
        prev = info.energy
    end
    return DMRGCBERun(es, spread, bds, maxbd, sbds, maxsbd, secs, maxexp, epre, efnl, disc,
                      kd, converged, status, ledgers)
end

"""
    rsvd_cbe_dmrg1s!(psi, h; kwargs...) -> DMRGCBERun

CBE-DMRG with the **RSVD** selection: the expansion is chosen from a thin Gaussian sketch of
`H*Theta`. **This is the method** -- `dmrg_cbe1s!`'s default, named so an A/B against the exact
arm reads as two arms of one comparison at the call site.
"""
rsvd_cbe_dmrg1s!(psi::SymMPS, h::Union{XXZChain, MPO}; kwargs...) =
    dmrg_cbe1s!(psi, h; exact = false, kwargs...)

"""
    exact_cbe_dmrg1s!(psi, h; kwargs...) -> DMRGCBERun

CBE-DMRG with the **exact** selection: the expansion uses the complete fused basis, so it
forms the rank-4 `H*Theta` -- the cost the randomised version exists to avoid. The A/B
reference arm, not a better default and not a fallback.
"""
exact_cbe_dmrg1s!(psi::SymMPS, h::Union{XXZChain, MPO}; kwargs...) =
    dmrg_cbe1s!(psi, h; exact = true, kwargs...)

"""
    rsvd_cbe_dmrg1s_seeded!(psi, h; kwargs...) -> DMRGCBERun

CBE-DMRG with the RSVD selection **and the local eigensolve started from the EXPANDED basis**
rather than from the zero-padded current tensor (`seed_expanded = true`).

The distinction it isolates: `expand_bond!` widens the bond leaving the state unchanged, so the
admitted directions arrive with zero amplitude and the default Lanczos start cannot see them
until the operator has been applied once. This arm gives the start a small component along them
up front. Everything else -- the selection, the operator, the truncation, the sweep order -- is
identical to [`rsvd_cbe_dmrg1s!`](@ref), so a difference between the two is attributable to the
start vector alone.

⚠ It buys the earlier basis at the price of ONE extra operator application per local solve, and
that application is counted in `krylov_dims` like any other. Read the comparison as
applications-to-accuracy, not sweeps-to-accuracy: fewer sweeps at a higher per-sweep cost is not
automatically a win, and which way it lands is a measurement, not a prediction.
"""
rsvd_cbe_dmrg1s_seeded!(psi::SymMPS, h::Union{XXZChain, MPO}; kwargs...) =
    dmrg_cbe1s!(psi, h; exact = false, seed_expanded = true, kwargs...)
