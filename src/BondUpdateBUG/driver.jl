# The public entry point: run the odd/even sweep for n_steps and report what it did.
#
# Mirrors the loop Alice's two_site_bug.py runs around scheme.parity_sweep.
#
# REAL TIME ONLY. `dt` is a real time step and the driver forms `tau = -im*dt`
# itself. Alice carries an `imaginary_time` mode through a global prefactor; this
# integrator is validated as a real-time propagator and does not expose one.

"""
    BondUpdateOptions

Controls for one `bond_update_bug!` run.

  - `dt`, `n_steps` -- real time step and how many to take.
  - `order` -- `:strang` (even dt/2, odd dt, even dt/2, second order) or `:lie`
    (even dt, odd dt, first order). These are the only orders: under the strict
    Sulz <=2r bound the rank-2r Galerkin bond update caps the global order at 2,
    so composition and extrapolation cannot raise it (see `SUPPORTED_ORDERS`).
  - `maxdim` -- hard bond-dimension cap. Python's default is 200; same here.
  - `trunc_thresh` -- singular-value cutoff for the S-step split.
  - `normalize` -- rescale to unit norm after each step. The norm is recorded
    BEFORE rescaling, so `info.norms` still shows what the step did.
  - `augment`, `missing_fill` -- rank-adaptation controls, forwarded. There is
    no K/L tolerance: that augmentation is bounded only by Sulz's `2r`.
  - `lanczos_tol`, `lanczos_maxiter` -- Krylov budget for all three substeps.
  - `seed` -- one RNG is seeded with it for the WHOLE run, so a run is
    reproducible while consecutive steps still draw different fill directions.
"""
Base.@kwdef struct BondUpdateOptions
    dt::Float64 = 0.05
    n_steps::Int = 10
    order::Symbol = :strang
    maxdim::Int = 200
    trunc_thresh::Float64 = 1e-12
    normalize::Bool = true
    augment::Bool = true
    missing_fill::Int = 1
    lanczos_tol::Float64 = 1e-15
    lanczos_maxiter::Int = 30
    seed::UInt = 0x5EED
end

"""
    BondUpdateInfo

Per-step record of a run. Every field has length `n_steps`.

  - `times` -- elapsed time after each step.
  - `norms` -- state norm after the step, before any renormalisation.
  - `bond_dims` -- the full bond-dimension profile after each step.
  - `max_bond_dims` -- its maximum, the usual headline number.
  - `aug_k_dims`, `aug_l_dims` -- largest PROPOSED augmented rank in the step,
    i.e. old rank plus new directions, before the truncating split. Now counts
    only sectors that can actually pair (`kls_step.jl::pairable_charges`).
  - `discarded` -- largest relative weight thrown away by a bond's S-step split.
"""
struct BondUpdateInfo
    times::Vector{Float64}
    norms::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    aug_k_dims::Vector{Int}
    aug_l_dims::Vector{Int}
    discarded::Vector{Float64}
end

Base.length(info::BondUpdateInfo) = length(info.times)

"The orders `bond_update_bug!` accepts. Under the strict Sulz <=2r bound the
rank-2r Galerkin bond update caps the achievable global order at 2, so the
integrator offers exactly the forward-only parity Trotter of the even/odd bond
groups: `:strang` (second order) and `:lie` (first)."
const SUPPORTED_ORDERS = (:lie, :strang)

"The (parity, fraction-of-dt) schedule of one Trotter step."
function _trotter_schedule(order::Symbol)
    order === :strang && return ((:even, 0.5), (:odd, 1.0), (:even, 0.5))
    order === :lie    && return ((:even, 1.0), (:odd, 1.0))
    throw(ArgumentError("order must be :strang or :lie, got $order"))
end

"""
Advance `psi` by `dt` under `order`, in place, accumulating the sweep diagnostics.

A parity group's bonds are disjoint, so each group is an exact factor of the step;
the only splitting error is between the even and odd groups. `:strang` (even dt/2,
odd dt, even dt/2) is second order, `:lie` (even dt, odd dt) is first. Both are
forward-only -- every sub-step time is positive.
"""
function _advance!(psi::SymMPS, gates, order::Symbol, dt::Float64, kw, acc)
    for (parity, frac) in _trotter_schedule(order)
        s = parity_sweep!(psi, gates, parity, ComplexF64(-im * dt * frac); kw...)
        acc[1] = max(acc[1], s.aug_k); acc[2] = max(acc[2], s.aug_l)
        acc[3] = max(acc[3], s.discarded)
    end
    return psi
end

"""
    bond_update_bug!(psi, gates; opts=BondUpdateOptions()) -> BondUpdateInfo

Evolve `psi` in place for `opts.n_steps` real-time steps of `opts.dt`.

A parity group is an exact factor of the step -- its bonds are disjoint -- so the
only splitting error is between the even and odd groups, which is what `order`
selects.
"""
function bond_update_bug!(psi::SymMPS, gates;
                          opts::BondUpdateOptions = BondUpdateOptions())
    opts.order in SUPPORTED_ORDERS || throw(ArgumentError(
        "order must be one of $(SUPPORTED_ORDERS), got $(opts.order)"))
    rng = MersenneTwister(opts.seed)
    canonical!(psi, 1)

    kw = (maxdim = opts.maxdim, trunc_thresh = opts.trunc_thresh,
          augment = opts.augment,
          missing_fill = opts.missing_fill,
          maxiter = opts.lanczos_maxiter, tol = opts.lanczos_tol, rng = rng)

    times = Float64[]; norms = Float64[]
    bdims = Vector{Int}[]; maxb = Int[]
    augk = Int[]; augl = Int[]; disc = Float64[]

    for step in 1:opts.n_steps
        acc = Any[0, 0, 0.0]
        _advance!(psi, gates, opts.order, opts.dt, kw, acc)
        ak, al, dd = acc[1], acc[2], acc[3]

        n = norm(psi)                            # recorded BEFORE renormalising
        if opts.normalize && n > 0
            psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center])
        end

        bd = bond_dims(psi)
        push!(times, step * opts.dt); push!(norms, n)
        push!(bdims, bd); push!(maxb, maximum(bd; init = 0))
        push!(augk, ak); push!(augl, al); push!(disc, dd)
    end

    return BondUpdateInfo(times, norms, bdims, maxb, augk, augl, disc)
end
