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
  - `pad` -- complete each active bond's frame to the Sulz `2r` budget with
    orthogonal random directions. Off by default (U(1) grows rank via the
    minimal missing-sector fill). Needed WITHOUT symmetry: with a single dense
    sector the missing-fill never fires, so from a product state the rank-`r`
    Galerkin core cannot see the off-diagonal generator and the state freezes;
    padding lets rank double per step (the truncating SVD prunes bonds still
    ahead of the entanglement front). This is the no-symmetry analogue of the
    fill, and the rank growth 2-site TDVP gets from its 2-site SVD.
  - `criterion2` -- targeted alternative to `pad`: enrich each frame with
    the residual of the FULL 2-site update `HΘ` (Ceruti-Kusch-Lubich), i.e.
    exactly the direction the dynamics needs, capped at Sulz `2r`. Grows rank
    by the minimal physical amount (no wasted random columns), so it is the
    efficient no-symmetry rank-growth path. Prefer over `pad`.
    AUTO-DEFAULT (NO-SYMMETRY ONLY): when `symmetry_mode() === :none`, the
    initial state is a product state (every bond is dimension 1), and the caller
    has chosen neither `pad` nor `criterion2`, `criterion2` is switched on for
    the whole run. A product state is exactly the case the rank-1 freeze strands
    when there is a single dense sector, and criterion 2 is the minimal,
    physically correct way out -- so it is the safe default rather than
    something the caller must know to request. Under U(1) this does NOT fire:
    the missing-sector fill already grows a product state, and the symmetric
    path is left exactly as validated. Set `pad = true` to override with
    padding, or pass a non-product state to skip the auto-enable.
  - `lanczos_tol`, `lanczos_maxiter` -- Krylov budget for all three substeps.
  - `seed` -- one RNG is seeded with it for the WHOLE run, so a run is
    reproducible while consecutive steps still draw different fill directions.

  Diagnostics (all opt-in, all recorded from a `copy` of the state so they can
  never perturb the run):

  - `record_magnetisation` -- store the full `⟨Sz⟩` site profile after every
    step in `info.magnetisations` (for light-cone heat-maps / error metrics).
  - `spectrum_steps` -- step indices at which to store the FULL entanglement
    spectrum (all bonds) in `info.spectra`. Use a handful of timestamps, not
    every step -- it is far heavier than the scalar diagnostics.
  - `observe` -- an arbitrary `f(psi, step, time)` run after each step on a copy
    of the state; its return is pushed to `info.observations`. The escape hatch
    for anything the built-in recorders do not cover.
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
    pad::Bool = false
    criterion2::Bool = false
    lanczos_tol::Float64 = 1e-15
    lanczos_maxiter::Int = 30
    seed::UInt = 0x5EED
    record_magnetisation::Bool = false
    spectrum_steps::Vector{Int} = Int[]
    observe::Union{Nothing,Function} = nothing
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
  - `center_bond_dims` -- bond dimension of the central cut after each step
    (`bond_dims[center_bond]`), always recorded.

Opt-in per-step diagnostics (empty unless the matching option was set):

  - `magnetisations` -- the `⟨Sz⟩` site profile after each step, one
    `Vector{Float64}` of length `length(psi)` per step (`record_magnetisation`).
  - `spectra` -- `step => entanglement_spectrum` at the requested `spectrum_steps`;
    the value is the all-bonds Schmidt spectrum (`Vector{Vector{Float64}}`).
  - `observations` -- the return of `observe(psi, step, time)` for each step.
"""
struct BondUpdateInfo
    times::Vector{Float64}
    norms::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    aug_k_dims::Vector{Int}
    aug_l_dims::Vector{Int}
    discarded::Vector{Float64}
    center_bond_dims::Vector{Int}
    magnetisations::Vector{Vector{Float64}}
    spectra::Dict{Int,Vector{Vector{Float64}}}
    observations::Vector{Any}
end

Base.length(info::BondUpdateInfo) = length(info.times)

"A product state -- every interior bond is dimension 1. This is the state the
rank-1 K/L freeze strands, so it is what triggers the `criterion2` auto-default."
_is_product_state(psi::SymMPS) = maximum(bond_dims(psi); init = 1) <= 1

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

    # crit2 defaults ON from a product state, NO-SYMMETRY ONLY: the rank-1 freeze
    # (see kls_step.jl) strands a product initial state when there is a single
    # dense sector. Under U(1) the missing-sector fill already pads a product
    # state into growth, so the symmetric path needs nothing here. Only
    # auto-enable when the caller has not already chosen a rank-growth path (pad,
    # or explicit crit2), and only for THIS invocation's initial state (a chunked
    # driver wanting crit2 across an evolved, non-product restart must ask).
    criterion2 = opts.criterion2 ||
        (!opts.pad && symmetry_mode() === :none && _is_product_state(psi))

    kw = (maxdim = opts.maxdim, trunc_thresh = opts.trunc_thresh,
          augment = opts.augment,
          missing_fill = opts.missing_fill,
          pad = opts.pad,
          criterion2 = criterion2,
          maxiter = opts.lanczos_maxiter, tol = opts.lanczos_tol, rng = rng)

    times = Float64[]; norms = Float64[]
    bdims = Vector{Int}[]; maxb = Int[]
    augk = Int[]; augl = Int[]; disc = Float64[]
    cbd = Int[]
    mags = Vector{Float64}[]
    spectra = Dict{Int,Vector{Vector{Float64}}}()
    obs = Any[]
    cb = center_bond(psi)                        # fixed: the central interior bond

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
        push!(cbd, bd[cb])

        # Opt-in diagnostics. Each reads a `copy` so `canonical!` moving the
        # centre can never disturb the live state the next step evolves --
        # identical behaviour under U(1) and with no symmetry.
        if opts.record_magnetisation
            push!(mags, magnetisation(copy(psi)))
        end
        if step in opts.spectrum_steps
            spectra[step] = entanglement_spectrum(copy(psi))
        end
        if opts.observe !== nothing
            push!(obs, opts.observe(copy(psi), step, step * opts.dt))
        end
    end

    return BondUpdateInfo(times, norms, bdims, maxb, augk, augl, disc,
                          cbd, mags, spectra, obs)
end
