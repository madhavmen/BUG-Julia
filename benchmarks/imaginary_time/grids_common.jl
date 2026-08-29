# TIME GRIDS AND THE CSV SCHEMA -- shared by every driver that compares grid constructions.
#
# ⛔ THESE LIVE IN ONE FILE BECAUSE A SECOND COPY WOULD DRIFT, AND THE DRIFT WOULD BE INVISIBLE.
# `plot_logtime.py`'s `annotate_nyquist` REBUILDS the log grid in Python from this construction to
# draw the Nyquist bound; a driver carrying its own `log_steps` would put a bound on the figure
# that does not describe the runs plotted under it. The same applies to `panel_steps`, whose
# `n` convention (TOTAL steps, not steps per panel) is the single thing most easily got wrong
# between this repo and arXiv:2606.02930.
#
# Included by `logtime_suite.jl` (closed models) and `dissipative_xx.jl` (the open one). Requires
# `LinearAlgebra`, `Printf`, and BUGJulia's `bond_dims` / `to_concrete` from the includer.

const COLS = ["model", "symmetry", "L", "grid", "nsteps", "scheme", "obs", "ref",
              "err", "matvec", "chi", "seconds", "maxnorm", "finite"]

renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

"Step sizes of a geometric grid reaching `t_max` in `n` steps, starting at `t_0`."
function log_steps(t_0, t_max, n)
    r = (t_max / t_0)^(1 / (n - 1))
    ts = [t_0 * r^(k - 1) for k in 1:n]
    return vcat(ts[1], diff(ts))
end
uniform_steps(t_max, n) = fill(t_max / n, n)

"""
    panel_steps(t_max, P; per_panel = 4) -> Vector{Float64}

arXiv:2606.02930's PANEL grid: `P` dyadic panels with endpoints `0, t_max/2^(P-1), …, t_max/2,
t_max`, each subdivided into `per_panel` UNIFORM steps.

⛔ THEIR `n` IS NOT OUR `n`, AND CONFLATING THE TWO MISREADS BOTH. In the paper `n` is the number
of steps **per panel** (`n = 4`), so the total is `P * n`; everywhere else in this repo `n` is the
TOTAL step count. An "n = 4" row of ours against an "n = 4" of theirs compares 4 steps with `4P`.

⛔ AND THE STRUCTURE IS PIECEWISE-UNIFORM WITH A DOUBLING ENVELOPE, NOT GEOMETRIC. That has one
consequence worth stating before any scan, because it decides what `P` can and cannot buy:

    largest step = (t_max - t_max/2) / per_panel = t_max / (2 * per_panel)   -- INDEPENDENT OF P

So **`P` does not control the coarsest step at all** -- `per_panel` does. `P` refines the grid
near `tau = 0`, where the first step is `t_max / (2^(P-1) * per_panel)`. Raising `P` to fix an
accuracy problem caused by a too-large FINAL step cannot work, and raising `per_panel` to resolve
an initial transient is exponentially more expensive than raising `P`. The two knobs are not
interchangeable and a scan that moves only one of them will attribute the wrong cause.

The first two gaps are equal (`t_max/2^(P-1)` each) and every later one doubles; that is the
dyadic construction, not an off-by-one.
"""
function panel_steps(t_max::Real, P::Int; per_panel::Int = 4)
    P >= 1 || throw(ArgumentError("panel_steps needs P >= 1, got $P"))
    per_panel >= 1 || throw(ArgumentError("per_panel must be >= 1, got $per_panel"))
    edges = vcat(0.0, [t_max / 2.0^(P - j) for j in 1:P])
    dts = Float64[]
    for j in 1:P
        append!(dts, fill((edges[j + 1] - edges[j]) / per_panel, per_panel))
    end
    return dts
end

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

"""
Run `step!(psi, dt)` over `steps` and return `(observable, matvec, chi, seconds, maxnorm, finite)`.

`imag_time = true` renormalises after every step; a real-time run must NOT, because the norm is
the physics there and rescaling would hide a diverging integrator.

⛔ AN ARM THAT BREAKS MUST NOT TAKE THE SUITE WITH IT, and that is not defensive padding -- a
huge step is EXPECTED to break here and the breakdown is part of the result. MEASURED on the
Heisenberg chain at L=20: the log grid's last step is `dt ~ 54` at `n = 4` and `dt ~ 46` at
`n = 6`, and both fail -- BUG silently, returning `obs = 0`, and `tdvp2` by throwing
`UndefRefError` out of `_state_stored` (a step that truncates every sector away leaves an
undefined tensor slot). The first version of this file let that exception abort the run, so
models 2-4 were never reached because model 1's smallest arm broke.

Returning `NaN` records the failure as data instead: the row is written, the plot shows the
ceiling, and the remaining arms still run.

⚠ `post!`, when given, runs after EVERY step and before the observable. The open-system drivers
need it for the zero-body shift and `restore_trace!`; the closed ones pass nothing.
"""
function run_arm(step!, psi0, steps, observe; imag_time::Bool, trace = nothing, post! = nothing)
    psi = copy(psi0); kry = 0; t0 = time()
    # `trace`, when given, collects `(t, observe(psi))` at every step boundary. See
    # `trajectory_error` for why an ENDPOINT metric cannot see aliasing at all.
    tnow = 0.0
    if trace !== nothing
        push!(trace, (0.0, observe(psi)))
    end
    # ⛔ THE NORM IS RECORDED BEFORE EVERY RENORMALISATION, and it is a first-class result rather
    # than a diagnostic. In imaginary time `||psi||` grows like `exp(-tau*E_0)` -- MEASURED 3.9e37
    # after one `dt = 10` step at L=20 -- so it is the quantity that decides whether a step is
    # REPRESENTABLE at all, and a non-finite one is how a step fails silently. Renormalising
    # without looking at it first is exactly how the earlier `obs = 0` failures hid.
    maxnorm = 0.0; finite = true
    try
        for dt in steps
            kry += _kry(step!(psi, dt))
            post! === nothing || post!(psi, dt)
            nz = norm(psi)
            isfinite(nz) || (finite = false)
            maxnorm = max(maxnorm, isfinite(nz) ? nz : Inf)
            imag_time && renorm!(psi)
            tnow += dt
            trace === nothing || push!(trace, (tnow, observe(psi)))
        end
        o = observe(psi)
        # `obs == 0` exactly is BUG's breakdown signature at a too-large step, not a value.
        ok = isfinite(o) && o != 0.0
        return (ok ? o : NaN), kry, maximum(bond_dims(psi)), time() - t0, maxnorm, finite
    catch err
        @info "arm broke (recorded as NaN)" nsteps=length(steps) dtmax=maximum(steps) err
        return NaN, kry, 0, time() - t0, maxnorm, false
    end
end

"""
    trajectory_error(trace, exact; npts = 400) -> Float64

Largest deviation of the arm's observable AS A CURVE from `exact(t)`: the trace's samples are
joined piecewise-linearly and compared on `npts` uniform points.

⛔ THIS EXISTS BECAUSE AN ENDPOINT METRIC CANNOT SEE ALIASING, AND MEASURING ONE PRODUCED THE
OPPOSITE CONCLUSION. Scored on `m_s(t_max)` alone (L=10, t_max=8) the log grid at `n = 12` BEAT a
100-step uniform grid on two of three arms -- `bug` 2.872e-08 against 2.382e-06. That is neither a
fluke nor a defect in the grid: `expv` is EXACT for the local generator, so one enormous step
still lands close to the right state AT THAT INSTANT. What a step past Nyquist destroys is not the
endpoint but the ABILITY TO REPORT THE CURVE -- there are no samples in between.

So the two regimes here ask genuinely different questions, which is also why the log grid winning
in the imaginary-time models and losing in the real-time ones is not a contradiction:

    imaginary time   only the tau -> infinity ENDPOINT is wanted   -> log grid wins
    real time        the whole TRAJECTORY is the result            -> Nyquist binds

⚠ The reconstruction is deliberately the crudest one, LINEAR interpolation between samples. A
spline would flatter a coarse grid by inventing curvature it never measured, which is precisely
the thing under test.
"""
function trajectory_error(trace, exact; npts::Int = 400)
    length(trace) < 2 && return NaN
    ts = [p[1] for p in trace]
    os = [p[2] for p in trace]
    (all(isfinite, os) && all(isfinite, ts)) || return NaN
    tmax = ts[end]
    tmax > 0 || return NaN
    worst = 0.0
    for k in 0:npts
        t = tmax * k / npts
        i = clamp(searchsortedlast(ts, t), 1, length(ts) - 1)
        w = ts[i + 1] > ts[i] ? (t - ts[i]) / (ts[i + 1] - ts[i]) : 0.0
        worst = max(worst, abs((os[i] + w * (os[i + 1] - os[i])) - exact(t)))
    end
    return worst
end

function emit(io, model, sym, L, grid, n, scheme, obs, ref, kry, chi, sec,
              maxnorm = NaN, finite = true; err = nothing, tag = "")
    e = err === nothing ? abs(obs - ref) : err
    @printf(io, "%s,%s,%d,%s,%d,%s,%.10g,%.10g,%.6e,%d,%d,%.3f,%.6e,%d\n",
            model, sym, L, grid, n, scheme, obs, ref, e, kry, chi, sec,
            maxnorm, finite ? 1 : 0)
    flush(io)
    @printf("  %-14s %-6s n=%-4d %-9s err=%-11.3e mv=%-7d chi=%-4d %7.1fs |psi|max=%-10.2e%s%s\n",
            model, grid, n, scheme, e, kry, chi, sec, maxnorm, tag,
            finite ? "" : "  <- NON-FINITE NORM")
    flush(stdout)
end
