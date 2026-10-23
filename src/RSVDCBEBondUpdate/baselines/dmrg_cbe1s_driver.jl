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

Ground state of `h` (an `MPO` or `XXZChain`) by CBE-DMRG sweeps, in place. Convergence is tested on
the sweep-best energy, not bond to bond, since within a sweep the energy falls by construction.

`maxdim` is one value per call; a schedule is the caller's business. The particle number (U(1)) or
total spin (SU(2)) is pinned by the initial state, so this finds the ground state of that sector.
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

CBE-DMRG with the RSVD selection and the local eigensolve started from the expanded basis
(`seed_expanded = true`). Everything else matches [`rsvd_cbe_dmrg1s!`](@ref), so a difference is
attributable to the start vector alone.

The seed costs one extra operator application per local solve, counted in `krylov_dims`; compare
applications-to-accuracy, not sweeps.
"""
rsvd_cbe_dmrg1s_seeded!(psi::SymMPS, h::Union{XXZChain, MPO}; kwargs...) =
    dmrg_cbe1s!(psi, h; exact = false, seed_expanded = true, kwargs...)
