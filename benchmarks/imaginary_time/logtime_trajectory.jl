# WALL TIME AND BOND-DIMENSION GROWTH ALONG AN IMAGINARY-TIME TRAJECTORY -- the figures of
# arXiv:2606.02930, for BUG against 2-site TDVP.
#
# Their paper plots cost and rank AS A FUNCTION OF IMAGINARY TIME, which a summary table cannot
# show: the whole argument for a logarithmic grid is about how the LATE, HUGE steps behave, and an
# end-of-run number hides whether the rank exploded at step 3 and was then truncated back.
#
# So this file records ONE ROW PER STEP: `tau`, `dt`, the bond dimension, cumulative wall time,
# the energy and its error against the analytic Bethe reference. Three plots come out of that
# (`plot_logtime_traj.py`): wall time vs tau, chi vs tau, error vs tau.
#
# ⛔ SECTION 2 IS A ROBUSTNESS PROBE, AND IT EXISTS BECAUSE OF A THEORETICAL CLAIM THAT MY OWN
# MEASUREMENT APPEARED TO CONTRADICT. BUG's defining property (Ceruti-Lubich) is that its error
# bound is INDEPENDENT OF THE SMALLEST SINGULAR VALUE, where projector-splitting TDVP carries a
# `1/sigma_min`. BUG should therefore NOT fail where 2-site TDVP fails. But on the log grid at
# L=20 both broke at the same place -- `dt ~ 54` (n=4) and `dt ~ 46` (n=6): BUG returned exactly
# `obs = 0`, TDVP2 threw `UndefRefError` from `_state_stored`.
#
# Two very different explanations, and they are distinguishable:
#
#   (a) a ROBUSTNESS failure -- the integrator genuinely cannot take the step. This would
#       contradict the theory and would be a real defect worth chasing.
#   (b) FLOATING-POINT OVERFLOW in `expv` -- `exp(-dt*H)` grows like `exp(-dt*E_min)`, and with
#       `E_min ~ -10` and `dt ~ 50` that is `exp(500)`, near the Float64 ceiling of `exp(709)`.
#       Nothing to do with the integrator; a SHIFT of `H` fixes it and the theory is untouched.
#
# Section 2 reports the NORM after a single large step, which separates them outright: a
# non-finite norm is (b). Reporting `obs = 0` alone could never have told them apart, which is
# why the first version of this comparison should not have been read as evidence about BUG.
#
# ⛔ MPO ONLY, NEVER GATES -- `apply_h_two_site(Theta, h::XXZChain, ...)` closes the on-bond term
# with `apply_gate`, so every arm here passes an `MPO`.
#
# Run:  julia --project=. benchmarks/imaginary_time/logtime_trajectory.jl
# Out:  benchmarks/results/logtime_traj.csv  ->  benchmarks/imaginary_time/plot_logtime_traj.py

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))

const OUT  = joinpath(@__DIR__, "..", "results", "logtime_traj.csv")
const COLS = ["model", "scheme", "grid", "nsteps", "step", "tau", "dt",
              "chi", "bondsum", "seconds", "energy", "err"]

renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)
_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

function log_steps(t_0, t_max, n)
    r = (t_max / t_0)^(1 / (n - 1))
    ts = [t_0 * r^(k - 1) for k in 1:n]
    return vcat(ts[1], diff(ts))
end

"""
Walk `steps` and write one CSV row per step.

⛔ WALL TIME IS CUMULATIVE AND EXCLUDES THE OBSERVABLE. `mpo_energy` is measured every step for
the plot, which is NOT work the integrator does -- charging it to the integrator would make a
scheme that needs fewer steps look artificially better, because it pays fewer measurements.
"""
function trajectory(io, model, scheme, step!, psi0, W, eref, steps, gridname)
    # ⛔ COMPILE FIRST, THEN TIME. MEASURED without this: BUG's step 1 read 213.5s and 2-site
    # TDVP's read 50.5s, against 5-25s for every later step -- almost all of it Julia specialising
    # the SU(2) code path, which each scheme pays ONCE per process. Left in, the cumulative
    # wall-time curve would have shown BUG four times more expensive than TDVP2 at tau_0 purely
    # because BUG compiles more code, and that is the headline figure of this file.
    #
    # One throwaway step on a COPY, at the smallest dt in the run, then discard it.
    try
        step!(copy(psi0), first(steps))
    catch err
        @info "warm-up threw (timing may include compilation)" model scheme err
    end

    psi = copy(psi0); tau = 0.0; wall = 0.0
    for (k, dt) in enumerate(steps)
        t0 = time()
        ok = true
        try
            step!(psi, dt); renorm!(psi)
        catch err
            @info "step broke" model scheme k dt err
            ok = false
        end
        wall += time() - t0                       # integrator only; measurement excluded
        tau  += dt
        bd   = ok ? bond_dims(psi) : Int[]
        e    = ok ? real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps()) : NaN
        isfinite(e) && e != 0.0 || (e = NaN)
        @printf(io, "%s,%s,%s,%d,%d,%.6g,%.6g,%d,%d,%.4f,%.10g,%.6e\n",
                model, scheme, gridname, length(steps), k, tau, dt,
                isempty(bd) ? 0 : maximum(bd), isempty(bd) ? 0 : sum(bd),
                wall, e, abs(e - eref))
        flush(io)
        ok || break
    end
    @printf("  %-12s %-6s %-4s n=%-3d  tau=%.3g  chi=%d  %.1fs  err=%.3e\n",
            model, scheme, gridname, length(steps), tau,
            maximum(bond_dims(psi)), wall,
            abs(real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps()) - eref))
    flush(stdout)
end

# ── SECTION 2: is the large-step failure robustness, or overflow? ────────────────────────────
function robustness_probe(W, psi0, D, maxiter)
    println("\n-- large single steps from a warm state: norm tells overflow from breakdown --")
    warm = copy(psi0)
    for _ in 1:3
        cbe_bug_step!(warm, W, ComplexF64(-0.1); maxdim = D, trunc_thresh = 1e-12,
                      maxiter = maxiter, exact = true)
        renorm!(warm)
    end
    # ⛔ THE KRYLOV DEPTH IS SCANNED, BECAUSE IT IS THE THIRD CANDIDATE EXPLANATION AND THE ONE
    # THE CODE ITSELF POINTS AT. `cbe_bug_step!` caps the half-sweep basis at `krylov_basis = 30`
    # and stops early on `krylov_tol`; `exp(-tau*H)` at `tau ~ 50` with `||H|| ~ 20` is
    # `tau*||H|| ~ 1000`, which a 30-vector basis may simply not represent. If a DEEPER basis
    # fixes the large step, then the ceiling was a parameter choice of mine and BUG's robustness
    # (Ceruti-Lubich: error bound independent of sigma_min) is untouched -- which is what the
    # theory predicts and what a single `obs = 0` could never have distinguished.
    @printf("  %-6s %-14s %-14s %-14s %s\n", "dt", "scheme", "norm after", "E after", "verdict")
    for dt in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0)
        arms = Any[]
        for kb in (30, 80, 200)
            push!(arms, ("bug kb=$kb", (p, d) -> cbe_bug_step!(p, W, ComplexF64(-d);
                         maxdim = D, trunc_thresh = 1e-12, maxiter = maxiter, exact = true,
                         krylov_basis = kb, krylov_tol = 1e-12)))
        end
        push!(arms, ("tdvp2", (p, d) -> tdvp2_step!(p, W, ComplexF64(-d); maxdim = D,
                                                     trunc_thresh = 1e-12, maxiter = maxiter)))
        for (nm, st) in arms
            p = copy(warm)
            try
                st(p, dt)
                nz = norm(p)
                e = isfinite(nz) && nz > 0 ? real(mpo_energy(copy(p), W)) / nz^2 : NaN
                v = !isfinite(nz) ? "OVERFLOW -> exp(-dt*H) exceeded Float64" :
                    nz == 0.0     ? "UNDERFLOW -> norm collapsed to zero" : "ok"
                @printf("  %-6.0f %-8s %-14.4e %-14.6f %s\n", dt, nm, nz, e, v)
            catch err
                @printf("  %-6.0f %-8s %-14s %-14s THREW: %s\n", dt, nm, "-", "-",
                        first(split(sprint(showerror, err), '\n')))
            end
            flush(stdout)
        end
    end
end

function main(; L = 20, D = 64, tau_max = 60.0, tau_0 = 0.05, maxiter = 40)
    set_symmetry!(:SU2)
    W = heisenberg_su2_mpo(L)
    psi0 = dimer_state(L)
    eref, _, res = bethe_xxx_open_energy(L)
    res < 1e-12 || error("Bethe residual $res")
    @printf("HEISENBERG SU(2) CHAIN L=%d D=%d tau_max=%g  E_exact=%.10f\n", L, D, tau_max, eref)

    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, join(COLS, ","))
        for n in (8, 12, 24)
            trajectory(io, "heis_chain", "bug", (p, dt) -> cbe_bug_step!(p, W,
                       ComplexF64(-dt); maxdim = D, trunc_thresh = 1e-12,
                       maxiter = maxiter, exact = true), psi0, W, eref,
                       log_steps(tau_0, tau_max, n), "log")
            trajectory(io, "heis_chain", "tdvp2", (p, dt) -> tdvp2_step!(p, W,
                       ComplexF64(-dt); maxdim = D, trunc_thresh = 1e-12,
                       maxiter = maxiter), psi0, W, eref,
                       log_steps(tau_0, tau_max, n), "log")
        end
    end
    println("wrote ", OUT)
    robustness_probe(W, psi0, D, maxiter)
end

main()
