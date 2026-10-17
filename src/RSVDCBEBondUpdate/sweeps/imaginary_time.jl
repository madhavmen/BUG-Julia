# NORMALISED IMAGINARY-TIME CBE-BUG (cbe_bug_imaginary_open_benchmark plan, Task 1).
#
# One step is `psi(beta + dbeta) ~ exp[-dbeta (H - Eshift)] psi(beta)` taken by ONE
# `cbe_bug_step!` at `tau = -dbeta`: the augmented root space keeps the old state (`proj_err`), the
# root integrator (`root_method`) evolves inside it, the state is renormalised, and the step's one
# compression follows. Nothing here changes the validated CBE-RSVD / Krylov logic; this file is the
# imaginary-time driver around it:
#
#   * the energy shift `Eshift` (default: the state's own energy), passed to the step's generators
#     (`shift_h`) so the basis weights measure the change of the NORMALISED state;
#   * the energies before the root, after the root / before truncation, and after truncation
#     (`root_report`), so `Delta E_root` and `Delta E_compression` are separable;
#   * the OUTSIDE-Galerkin defect of the step, `dbeta ||(I - P) H psi_1|| / ||psi_1||`, measured on
#     the evolved state (`outside_residual`) and charged against `tol_basis` -- distinct from the
#     root solver's own projected residual (`tol_root`) and from the discarded weight (`tol_trunc`);
#   * reject / re-enrich / retry: a step whose ledger or outside test fails is redone from the saved
#     state, first with the order-p closure run to closure (re-enrich), then as two half steps
#     (reject), and is FLAGGED (or throws) if it still fails -- never a silent success (IT-U6);
#   * the time grids: the logarithmic panels of arXiv:2606.02930 and the uniform control.
#
# The three solver tolerances keep the meanings they have in real time: `tol_basis` the basis
# budget per step (Tests A-D and the outside defect), `tol_trunc` the cumulative discarded norm of
# the one compression (same per-cut rule as 2-site TDVP, `_cut_divisor`), `tol_root` the root
# solve's convergence (`tol_expv` of `cbe_bug_step!`; for an implicit root, the FOM residual of its
# shifted solves). The ORDER and the GRID are method choices, not tolerances.

"""
    log_panel_steps(beta_max, P; per_panel = 4) -> Vector{Float64}

The step sizes of arXiv:2606.02930's logarithmic PANEL grid: `P` dyadic panels with endpoints
`0, beta_max/2^(P-1), ..., beta_max/2, beta_max`, each split into `per_panel` uniform steps (four is
the paper's trapezoidal reference). The first two panels are equal and every later one doubles, so
the early steps are `2^(P-1)` times finer than the last. The largest step is
`beta_max / (2 per_panel)` WHATEVER `P` is: `P` refines the start, `per_panel` the coarsest step.
`sum(steps) == beta_max` up to roundoff. (Same construction as `benchmarks/imaginary_time`'s
`panel_steps`, whose `n` is likewise the TOTAL step count, here `P * per_panel`.)
"""
function log_panel_steps(beta_max::Real, P::Int; per_panel::Int = 4)
    beta_max > 0 || throw(ArgumentError("beta_max must be positive, got $beta_max"))
    P >= 1 || throw(ArgumentError("log_panel_steps needs P >= 1, got $P"))
    per_panel >= 1 || throw(ArgumentError("per_panel must be >= 1, got $per_panel"))
    edges = vcat(0.0, [Float64(beta_max) / 2.0^(P - j) for j in 1:P])
    steps = Float64[]
    for j in 1:P
        append!(steps, fill((edges[j + 1] - edges[j]) / per_panel, per_panel))
    end
    return steps
end

"`n` equal steps reaching `beta_max`: the uniform-grid control."
function uniform_beta_steps(beta_max::Real, n::Int)
    beta_max > 0 || throw(ArgumentError("beta_max must be positive, got $beta_max"))
    n >= 1 || throw(ArgumentError("n must be >= 1, got $n"))
    return fill(Float64(beta_max) / n, n)
end

"The grid points `0, b_1, ..., beta_max` a step list reaches (`cumsum` with the origin)."
beta_grid(steps::AbstractVector{<:Real}) = vcat(0.0, cumsum(Float64.(steps)))

"`<psi|H|psi> / <psi|psi>`, any gauge (`mpo_energy` is exact in every gauge)."
_ite_energy(psi::SymMPS, mpo::MPO) = real(mpo_energy(psi, mpo)) / max(real(overlap(psi, psi)), eps())

"Renormalise a canonical state in place through its centre tensor; returns the old norm."
function _ite_normalise!(psi::SymMPS)
    n = norm(psi)
    (isfinite(n) && n > 0) || throw(ErrorException(
        "imaginary-time step produced a state of norm $n -- not representable"))
    psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center])
    return n
end

# The ledger reasons a retry can address; `:outside` is this file's own (the outside-Galerkin
# defect). `:projection` belongs to the midpoint step, which this driver does not use.
const _ITE_REASONS = (:sample, :tail, :krylov, :closure, :expv, :trunc, :outside)

"""
One attempt at one step: the state's energy, the step at `tau = -dbeta` with the shifted
generators and the root report, renormalisation, the energy after, and the verdicts.
"""
function _ite_attempt!(psi::SymMPS, mpo::MPO, dbeta::Float64; order::Int, root_method::Symbol,
                       tol_basis::Float64, tol_trunc::Float64, tol_root::Float64, maxdim::Int,
                       eshift, outside_residual::Bool, closure_checks::Bool, parallel::Bool,
                       rng::AbstractRNG, kwargs...)
    t0 = time_ns()
    e0 = _ite_energy(psi, mpo)
    es = eshift === :energy ? e0 : Float64(eshift)
    rep = Ref{Any}(nothing)
    info = cbe_bug_step!(psi, mpo, ComplexF64(-dbeta); order = order, root_method = root_method,
                         eshift = es, tol_basis = tol_basis, tol_trunc = tol_trunc,
                         tol_expv = tol_root, maxdim = maxdim, root_report = rep,
                         outside_residual = outside_residual, closure_checks = closure_checks,
                         parallel = parallel, rng = rng, kwargs...)
    growth = _ite_normalise!(psi)
    e1 = _ite_energy(psi, mpo)
    r = rep[]
    unmet = copy(info.ledger.unmet)
    # `!(resid <= tol)` and not `resid > tol`: a NaN defect is a failed measurement, not a pass.
    outside_residual && !(r.resid <= tol_basis) && push!(unmet, :outside)
    return (dbeta = dbeta, ok = isempty(unmet), unmet = unmet, eshift = es,
            e_before = e0, e_pre_root = r.e_pre, e_root = r.e_root, e_after = e1,
            de_root = r.e_root - r.e_pre, de_compression = e1 - r.e_root,
            outside = r.outside, resid = r.resid, root_err = info.ledger.expv_err,
            root_iters = info.ledger.expv_iters, discarded = info.ledger.discarded,
            growth = growth, chi = maximum(bond_dims(psi); init = 0),
            r_aug = info.ledger.r_aug, closure_checks = closure_checks,
            wall = (time_ns() - t0) / 1e9, info = info)
end

"""
    ite_step!(psi, mpo, dbeta; order = 2, root_method = :exp, tol_basis = 1e-5,
              tol_trunc = 1e-6, tol_root = 1e-6, maxdim = 128, eshift = :energy,
              outside_residual = true, retries = 2, on_fail = :flag, parallel = false,
              rng, kwargs...) -> NamedTuple

ONE normalised imaginary-time step of CBE-BUG over `dbeta`, in place; `psi` is renormalised on
entry and exit. `eshift = :energy` uses the state's own energy as `Eshift` (any number fixes it).
`kwargs` reach [`cbe_bug_step!`](@ref) (e.g. `closure_probe`, `fold_omega`, `thin_search`).

VERDICT. The step is accepted when its ledger is `:met` and, with `outside_residual`, the
outside-Galerkin defect `dbeta ||(I-P) H psi_1|| / ||psi_1||` is within `tol_basis`. Otherwise it is
redone from the saved state:

  1. `order > 1` and the closure or the outside test failed: once more with the closure run to
     closure (`closure_checks = true`) -- RE-ENRICH, same `dbeta`;
  2. then, while `retries > 0`, as two half steps of `dbeta/2` (each with `retries - 1`) -- REJECT.

A step that still fails is `status = :failed` with its reasons in `unmet`, and `on_fail = :error`
throws instead. Never a silent success.

Returns `status` (`:met`, `:retried`, `:failed`), `unmet`, the energies `e_before`, `e_pre_root`,
`e_root`, `e_after` and the split `de_root` (root solve) / `de_compression` (truncation), the
outside defect `resid` (and the scale-free `outside`), `root_err`, `discarded`, `chi`, `nsub` (the
substeps finally taken), `attempts` (every attempt's record, including the rejected ones) and
`wall`.
"""
function ite_step!(psi::SymMPS, mpo::MPO, dbeta::Real; order::Int = 2,
                   root_method::Symbol = :exp, tol_basis::Float64 = 1e-5,
                   tol_trunc::Float64 = 1e-6, tol_root::Float64 = 1e-6, maxdim::Int = 128,
                   eshift = :energy, outside_residual::Bool = true, retries::Int = 2,
                   on_fail::Symbol = :flag, parallel::Bool = false,
                   rng::AbstractRNG = MersenneTwister(0x5EED), kwargs...)
    dbeta > 0 || throw(ArgumentError("dbeta must be positive, got $dbeta"))
    root_method in ROOT_METHODS || throw(ArgumentError(
        "root_method must be one of $(ROOT_METHODS), got :$root_method"))
    on_fail in (:flag, :error) || throw(ArgumentError("on_fail must be :flag or :error"))
    retries >= 0 || throw(ArgumentError("retries must be >= 0, got $retries"))
    t0 = time_ns()
    _ite_normalise!(psi)
    saved = copy(psi)         # every mutation replaces a slot, so this snapshot is stable
    akw = (order = order, root_method = root_method, tol_basis = tol_basis,
           tol_trunc = tol_trunc, tol_root = tol_root, maxdim = maxdim, eshift = eshift,
           outside_residual = outside_residual, parallel = parallel, rng = rng)
    restore!() = (for i in 1:length(psi); psi[i] = saved[i]; end; psi.center = saved.center)

    attempts = Any[]
    a = _ite_attempt!(psi, mpo, Float64(dbeta); akw..., closure_checks = false, kwargs...)
    push!(attempts, a)
    if !a.ok && order > 1 && any(in((:closure, :outside)), a.unmet)
        restore!()
        a = _ite_attempt!(psi, mpo, Float64(dbeta); akw..., closure_checks = true, kwargs...)
        push!(attempts, a)
    end
    a.ok && return _ite_result(length(attempts) == 1 ? :met : :retried, [a], attempts, t0)

    if retries > 0
        restore!()
        halves = Any[]
        for _ in 1:2
            h = ite_step!(psi, mpo, dbeta / 2; akw..., retries = retries - 1, on_fail = :flag,
                          kwargs...)
            append!(attempts, h.attempts)
            push!(halves, h)
        end
        ok = all(h.status !== :failed for h in halves)
        res = _ite_result(ok ? :retried : :failed, halves, attempts, t0)
        ok || on_fail === :flag ||
            throw(ErrorException("imaginary-time step dbeta=$dbeta failed after retries: " *
                                 "unmet $(res.unmet)"))
        return res
    end
    on_fail === :error &&
        throw(ErrorException("imaginary-time step dbeta=$dbeta failed: unmet $(a.unmet); " *
                             budget_diagnostic(a.info.ledger)))
    return _ite_result(:failed, [a], attempts, t0)
end

"""
Fold the accepted pieces of a step (one attempt, or the two halves of a split) into one record:
first energy before, last after, root and compression changes summed, worst defects.
"""
function _ite_result(status::Symbol, pieces, attempts, t0)
    nsub(p) = hasproperty(p, :nsub) ? p.nsub : 1
    unmet = unique(vcat(Symbol[], (p.unmet for p in pieces)...))
    return (status = status, unmet = unmet, dbeta = sum(p.dbeta for p in pieces),
            e_before = pieces[1].e_before, e_pre_root = pieces[1].e_pre_root,
            e_root = pieces[end].e_root, e_after = pieces[end].e_after,
            de_root = sum(p.de_root for p in pieces),
            de_compression = sum(p.de_compression for p in pieces),
            resid = maximum(p.resid for p in pieces), outside = maximum(p.outside for p in pieces),
            root_err = maximum(p.root_err for p in pieces),
            discarded = maximum(p.discarded for p in pieces), chi = pieces[end].chi,
            nsub = sum(nsub(p) for p in pieces), attempts = attempts,
            wall = (time_ns() - t0) / 1e9)
end

"""
    ite_evolve!(psi, mpo, steps; observe = nothing, kwargs...) -> Vector{NamedTuple}

Normalised imaginary-time evolution over the step list `steps` (`log_panel_steps`,
`uniform_beta_steps`, or any positive vector), one [`ite_step!`](@ref) per entry with `kwargs`.
Returns one record per step: `beta` reached plus the step's record, and `obs = observe(psi)` when
an observable is given. A `:failed` step is recorded and the evolution continues
(`on_fail = :error` stops it instead).
"""
function ite_evolve!(psi::SymMPS, mpo::MPO, steps::AbstractVector{<:Real};
                     observe = nothing, kwargs...)
    log = Any[]
    beta = 0.0
    for db in steps
        r = ite_step!(psi, mpo, db; kwargs...)
        beta += db
        push!(log, merge((beta = beta,), r,
                         observe === nothing ? NamedTuple() : (obs = observe(psi),)))
    end
    return log
end
