# ── QUANTUM-JUMP TRAJECTORIES (dissipative_cbe_bug_implementation_plan.md, Task 4) ──
# A trajectory is an unnormalised no-jump segment `dψ/dt = Gψ` until the survival `||ψ||^2` reaches a
# random threshold `u`, then a jump `ψ <- L_μ ψ / ||L_μ ψ||` with `p_μ ∝ ||L_μ ψ||^2` (plan eq. 8).
# The segment is renormalised every step and its log survival carried explicitly, so the threshold
# test is exact and never underflows. The step is any no-jump integrator, BUG by default.

"""
    apply_jump(ch, psi; tol_trunc = 0.0, maxdim = 10_000) -> (SymMPS, discarded)

`L_μ ψ` for the channel `ch` ([`JumpChannel`](@ref)), unnormalised, each factor applied exactly
and compressed once ([`apply_mpo`](@ref)). The returned discarded norm is the largest factor's.
"""
function apply_jump(ch::JumpChannel, psi::SymMPS; tol_trunc::Float64 = 0.0, maxdim::Int = 10_000)
    phi, worst = psi, 0.0
    for m in ch.mpos
        phi, d = apply_mpo(m, phi; tol_trunc = tol_trunc, maxdim = maxdim)
        worst = max(worst, d)
    end
    return phi, worst
end

"`||L_μ ψ||^2`, exact: the uncompressed product, read by `overlap` (any gauge)."
function jump_weight(ch::JumpChannel, psi::SymMPS)
    phi = psi
    for (k, m) in pairs(ch.mpos)
        phi = mpo_product(m, phi)
        k < length(ch.mpos) || break
        n2 = real(overlap(phi, phi))
        n2 > 0 || return 0.0
        canonical!(phi, 1)           # a gauged input for the next factor (exact, nothing cut)
    end
    return max(real(overlap(phi, phi)), 0.0)
end

"Scale the canonical `psi` to unit norm in place; returns the squared norm it had."
function _jt_normalise!(psi::SymMPS)
    n2 = norm(psi)^2
    (isfinite(n2) && n2 > 0) || throw(ErrorException(
        "quantum-jump segment: the state has norm^2 $n2 -- not representable"))
    psi[psi.center] = to_concrete((1.0 / sqrt(n2)) * psi[psi.center])
    return n2
end

"""
    jump_trajectory(psi0, G, channels, tgrid; dt, seed, observe = nothing, step = nothing,
                    step_kw = NamedTuple(), event_tol = 1e-10, max_event_evals = 40,
                    growth_tol = 1e-10, tol_trunc = 1e-10, maxdim = 256) -> NamedTuple

One quantum-jump trajectory from `psi0` (normalised first) on the output grid `tgrid` (starting at
0). The no-jump segment runs `step(psi, h)`, by default `cbe_bug_step!(psi, G, -im h;
hermitian = false, order = 3, tol_trunc, maxdim, step_kw...)`, in steps of at most `dt` that land
on every grid time.

A step whose survival falls below the threshold `u` is redone from its saved start: the jump time
`s` solves `log P0(t + s) = log u` by the Illinois regula falsi (`log P0` falls monotonically),
each evaluation one step of size `s`, to `|log P0 - log u| <= event_tol` or `max_event_evals`. The
channel is then drawn with `p_μ ∝ ||L_μ ψ_c||^2` ([`jump_weight`](@ref)), the state jumps and
compresses at `tol_trunc`/`maxdim`, `log P0` restarts at 0 and a new `u` is drawn. The step has no
history to reset: every BUG step starts from the state alone.

Every random number comes from `MersenneTwister(seed)` (thresholds, then a channel draw per jump), so
a seed reproduces its trajectory. `observe(psi)` (a vector) is read on the NORMALISED conditional
state at each grid time. A step that RAISES the norm by more than `growth_tol` is recorded in
`flags` (`:norm_growth`): physically `P0` cannot grow, so this separates a defect from the
per-step renormalisation, which is bookkeeping and never flagged.

Returns `times`, `obs`, `logp` (log survival since the last jump, at each grid time), `njumps`
(cumulative, at each grid time), `events` (`t`, `channel`, `weights`, `evals`, `resid`,
`discarded`), `flags`, `nsteps` (accepted steps), `nevals` (extra event-location steps), `seed`.
"""
function jump_trajectory(psi0::SymMPS, G::MPO, channels::AbstractVector{JumpChannel},
                         tgrid::AbstractVector{<:Real}; dt::Real, seed::Integer,
                         observe = nothing, step = nothing, step_kw = NamedTuple(),
                         event_tol::Float64 = 1e-10, max_event_evals::Int = 40,
                         growth_tol::Float64 = 1e-10, tol_trunc::Float64 = 1e-10,
                         maxdim::Int = 256)
    isempty(channels) && throw(ArgumentError("jump_trajectory needs at least one channel"))
    (length(tgrid) >= 1 && tgrid[1] == 0 && issorted(tgrid)) || throw(ArgumentError(
        "tgrid must be sorted and start at 0"))
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    stepfn = step !== nothing ? step :
        (p, h) -> cbe_bug_step!(p, G, ComplexF64(-im * h); hermitian = false, order = 3,
                                tol_trunc = tol_trunc, maxdim = maxdim, step_kw...)
    rng = MersenneTwister(seed)
    psi = copy(psi0)
    _jt_normalise!(psi)
    logp, logu = 0.0, log(rand(rng))
    t, nj, nsteps, nevals = 0.0, 0, 0, 0
    events, flags = NamedTuple[], NamedTuple[]
    obs, logps, njs = Any[], Float64[], Int[]
    record!(p) = (push!(obs, observe === nothing ? nothing : observe(p)); push!(logps, logp);
                  push!(njs, nj))
    record!(psi)
    for k in 2:length(tgrid)
        tnext = Float64(tgrid[k])
        while tnext - t > 1e-12 * max(1.0, tnext)
            h = min(Float64(dt), tnext - t)
            saved = copy(psi)
            stepfn(psi, h)
            n2 = norm(psi)^2
            n2 > 1 + growth_tol && push!(flags, (t = t + h, kind = :norm_growth, value = n2 - 1))
            if logp + log(n2) > logu
                _jt_normalise!(psi)
                logp += log(n2)
                t += h
                nsteps += 1
                continue
            end
            # ── the survival crossed `u` inside (t, t + h]: locate it (Illinois regula falsi) ──
            a, fa = 0.0, logp - logu                    # > 0
            b, fb, best = h, logp + log(n2) - logu, psi  # <= 0
            side, ev = 0, 0
            while abs(fb) > event_tol && ev < max_event_evals && b - a > 1e-14 * h
                s = clamp((a * fb - b * fa) / (fb - fa), a + 1e-3 * (b - a), b - 1e-3 * (b - a))
                trial = copy(saved)
                stepfn(trial, s)
                ev += 1
                fs = logp + log(norm(trial)^2) - logu
                if fs <= 0
                    b, fb, best = s, fs, trial
                    side == -1 && (fa /= 2)
                    side = -1
                else
                    a, fa = s, fs
                    side == 1 && (fb /= 2)
                    side = 1
                end
            end
            nevals += ev
            nsteps += 1
            psi = best
            _jt_normalise!(psi)
            w = [jump_weight(ch, psi) for ch in channels]
            W = sum(w)
            W > 0 || throw(ErrorException(
                "jump at t = $(t + b): every channel weight is zero, but the survival fell -- " *
                "the channels do not match the generator's Q"))
            r = rand(rng) * W
            mu = something(findfirst(>=(r), cumsum(w)), length(w))
            psi, disc = apply_jump(channels[mu], psi; tol_trunc = tol_trunc, maxdim = maxdim)
            _jt_normalise!(psi)
            t += b
            nj += 1
            push!(events, (t = t, channel = channels[mu].name, weights = w, evals = ev,
                           resid = fb, discarded = disc))
            logp, logu = 0.0, log(rand(rng))
        end
        t = tnext
        record!(psi)
    end
    return (times = collect(Float64, tgrid), obs = obs, logp = logps, njumps = njs,
            events = events, flags = flags, nsteps = nsteps, nevals = nevals, seed = seed)
end

"""
    jump_ensemble(psi0, G, channels, tgrid; ntraj, seed = 1, observe, kwargs...) -> NamedTuple

`ntraj` independent trajectories with seeds `seed, seed + 1, ...` (plan eq. 9), `kwargs` passed to
[`jump_trajectory`](@ref). Returns the trajectories and, at each grid time, the ensemble mean and
standard error of every component of `observe` and of the cumulative jump count, over the
independent trajectories.
"""
function jump_ensemble(psi0::SymMPS, G::MPO, channels::AbstractVector{JumpChannel},
                       tgrid::AbstractVector{<:Real}; ntraj::Int, seed::Integer = 1,
                       observe, kwargs...)
    ntraj >= 2 || throw(ArgumentError("jump_ensemble needs ntraj >= 2 for an error bar, got $ntraj"))
    trajs = [jump_trajectory(psi0, G, channels, tgrid; seed = seed + r - 1, observe = observe,
                             kwargs...) for r in 1:ntraj]
    nt = length(tgrid)
    O = [reduce(hcat, [Float64.(tr.obs[k]) for tr in trajs]) for k in 1:nt]   # (ncomp, ntraj)
    N = [Float64[tr.njumps[k] for tr in trajs] for k in 1:nt]
    avg(x) = sum(x) / length(x)
    sem(x) = sqrt(sum(abs2, x .- avg(x)) / (length(x) - 1) / length(x))
    rows(o) = [o[c, :] for c in 1:size(o, 1)]
    return (trajs = trajs, times = collect(Float64, tgrid),
            mean = [avg.(rows(o)) for o in O], stderr = [sem.(rows(o)) for o in O],
            jumps_mean = avg.(N), jumps_stderr = sem.(N))
end
