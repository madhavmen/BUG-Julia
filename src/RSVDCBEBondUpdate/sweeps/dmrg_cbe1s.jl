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
end

"""
    dmrg_cbe1s_sweep!(psi, mpo; kwargs...) -> DMRGCBEInfo

One full CBE-DMRG sweep (left to right, then right to left), in place.

Keywords: `dex`, `growth`, `dover`, `comp_ratio`, `sulz_cap`, `preselect_only`, `exact`
forward to [`cbe_expand`](@ref) -- `exact = false` (the default) is the **RSVD** selection and
is the method; `exact = true` forms the full fused basis and is the A/B reference arm.
`maxdim` / `trunc_thresh` bound the truncation, `maxiter` / `restol` the local eigensolve
(the reference's `K` / `Ktol`).

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
                           dex::Int = 0,
                           growth::Float64 = 2.0,
                           dover::Union{Nothing, Int} = nothing,
                           comp_ratio::Union{Float64, Nothing} = 1.0,
                           sulz_cap::Bool = false,
                           preselect_only::Bool = false,
                           exact::Bool = false,
                           maxdim::Int = 200,
                           trunc_thresh::Float64 = 1e-12,
                           maxiter::Int = 30,
                           restol::Float64 = 1e-6,
                           seed_expanded::Bool = false,
                           rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("dmrg_cbe1s_sweep! needs at least two sites"))

    cut = max(trunc_thresh, 1e-14)
    energies = Float64[]
    n_new = zeros(Int, L - 1)
    maxexp = 0; epre = 0.0; efnl = 0.0; disc = 0.0; nmv = 0

    # `rmax` read ONCE, before anything widens, so an (optional) global rank bound is a
    # statement about the state rather than about whichever bond is being looked at.
    exkw = (dex = dex, growth = growth, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact, rng = rng)

    function record!(info)
        maxexp = max(maxexp, info.dim_out)
        n_new[info.bond] += info.n_new
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
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
    canonical!(psi, 1)
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    for i in 1:(L - 1)
        rch1, info = expand_bond!(psi, mpo, i, :right, lch,
                                  right_channels(rstack, i + 2); exkw...)
        record!(info)

        A = solve1(i, lch, rch1; expanded = true)
        # Split with the amplitude moving RIGHT, truncating: this is the only place rank is
        # removed, and `Nkeep` is in MULTIPLETS under a non-abelian symmetry.
        res = svd(A, (1, 2); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
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
        lch1, info = expand_bond!(psi, mpo, i, :left,
                                  left_channels(lstack, i), rch; exkw...)
        record!(info)

        A = solve1(i + 1, lch1, rch; expanded = true)
        res = svd(A, (1,); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        C = to_concrete(res.U * res.S)
        psi[i + 1] = to_concrete(setitag(res.Vd, 1, tag))
        psi[i]     = to_concrete(setitag(contract(psi[i], (3,), C, (1,)), 3, tag))
        psi.center = i

        rch = push_right_channels(rch, mpo, psi[i + 1], i + 1)
    end
    psi[1] = solve1(1, boundary_channels(mpo), rch)
    psi.center = 1

    return DMRGCBEInfo(minimum(energies; init = Inf), energies, bond_dims(psi),
                       maxexp, n_new, epre, efnl, disc, nmv)
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
    dmrg_cbe1s!(psi, h; n_sweeps=20, etol=1e-10, kwargs...) -> DMRGCBERun

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
                     n_sweeps::Int = 20, etol::Float64 = 1e-10, kwargs...)
    mpo = h isa MPO ? h : mpo_from_terms(h)
    es = Float64[]; spread = Float64[]; bds = Vector{Int}[]; maxbd = Int[]
    sbds = Vector{Int}[]; maxsbd = Int[]; secs = Float64[]
    maxexp = Int[]; epre = Float64[]; efnl = Float64[]; disc = Float64[]; kd = Int[]
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
        if abs(info.energy - prev) < etol
            converged = true
            break
        end
        prev = info.energy
    end
    return DMRGCBERun(es, spread, bds, maxbd, sbds, maxsbd, secs, maxexp, epre, efnl, disc,
                      kd, converged)
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
