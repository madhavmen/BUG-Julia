# dmrg_cbe1s_driver.jl: the CBE-DMRG driver (from `dmrg_cbe1s.jl`, included right after it).

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
    # Per sweep: `budget_status` (`:met` / `:budget_unmet`) and the sweep's ledger, so a
    # converged energy carries its basis verdict with it.
    status::Vector{Symbol}
    ledgers::Vector{AdaptiveLedger}
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
    # The stopping threshold: an explicit `etol` wins, otherwise it is derived from the sweep's
    # own tolerances (`dmrg_etol`).
    etol = something(etol,
        dmrg_etol(something(get(kw, :tol_basis, nothing), DMRG_TOL_DEFAULTS.basis),
                  something(get(kw, :tol_trunc, nothing), DMRG_TOL_DEFAULTS.trunc), length(psi)))
    es = Float64[]; spread = Float64[]; bds = Vector{Int}[]; maxbd = Int[]
    sbds = Vector{Int}[]; maxsbd = Int[]; secs = Float64[]
    maxexp = Int[]; epre = Float64[]; efnl = Float64[]; disc = Float64[]; kd = Int[]
    status = Symbol[]; ledgers = AdaptiveLedger[]
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
        if abs(info.energy - prev) < etol
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
