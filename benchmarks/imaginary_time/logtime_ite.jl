# IMAGINARY TIME ON A LOGARITHMIC GRID -- arXiv:2606.02930, and whether our BUG needs their
# implicit solver to follow one.
#
# THEIR CLAIM. Long imaginary-time dynamics is resolved by a LOGARITHMIC time grid, giving an
# exponential reduction in the number of steps against uniform stepping; and A-stable IMPLICIT
# stepping makes any step size stable, at the price of a linear solve per step. Benchmarked
# against uniform-step TDVP, they report several orders of magnitude.
#
# THE QUESTION FOR US. The log grid is a property of the PROBLEM (imaginary time damps
# monotonically, so late times carry little new information), not of their integrator. Anyone can
# use it. What their scheme buys is STABILITY at huge steps, which they get from implicitness.
# Our sweeps solve each local problem with `expv` -- a Krylov EXPONENTIAL, which is exact for the
# local generator at any step size and needs no linear solve. So the question is whether an
# explicit-but-exponential integrator already has the stability that motivates their implicit one.
#
# ⛔ MPO ONLY, NEVER GATES. `apply_h_two_site(Theta, h::XXZChain, ...)` closes the on-bond term
# with `apply_gate(bond_gate(h, ...))` -- passing an `XXZChain` silently takes the gate path.
# Every arm here passes an `MPO` object (`heisenberg_su2_mpo`), and the MPO method of
# `apply_h_two_site` is pure environment contraction.
#
# ⛔ THE REFERENCE IS THE ANALYTIC BETHE ENERGY, never a dense or sparse diagonalisation, and its
# RESIDUAL is asserted before use -- the solver converges to a residual and a converged-but-wrong
# root is indistinguishable from a right one in the energy alone.
#
# THE GRID. `tau_k = tau_0 * r^k` up to `tau_max`, so `dt_k = tau_k - tau_{k-1}` grows
# geometrically. The last steps are ENORMOUS (order `tau_max`), which is exactly where an explicit
# scheme is expected to fail and where their implicitness is supposed to pay.
#
# WHAT IS MEASURED: energy error against Bethe, versus NUMBER OF STEPS and versus operator
# applications. Steps are the axis their claim is about; matvecs are the honest cost axis, because
# a huge step costs more Krylov vectors and a step count alone would hide that.
#
# Run:  julia --project=. benchmarks/imaginary_time/logtime_ite.jl

using LinearAlgebra, Printf, Statistics
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))

renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

"Logarithmic grid on `[tau_0, tau_max]`: the step sizes `dt_k = tau_k - tau_{k-1}`."
function log_steps(tau_0, tau_max, n)
    r = (tau_max / tau_0)^(1 / (n - 1))
    taus = [tau_0 * r^(k - 1) for k in 1:n]
    return vcat(taus[1], diff(taus))          # first step reaches tau_0 from 0
end

uniform_steps(tau_max, n) = fill(tau_max / n, n)

"""
Evolve in imaginary time over `steps`, renormalising each step, and report the final energy.

`step!(psi, dt)` takes a POSITIVE `dt` and applies `exp(-dt*H)` approximately.
"""
function evolve(step!, psi0, mpo, steps)
    psi = copy(psi0); kry = 0; peak = 0
    for dt in steps
        info = step!(psi, dt)
        kry += hasproperty(info, :krylov_dims) ? info.krylov_dims : 0
        renorm!(psi)
        peak = max(peak, maximum(bond_dims(psi)))
    end
    e = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
    return e, kry, peak, maximum(bond_dims(psi))
end

function run_chain(; L = 20, D = 64, tau_max = 60.0, tau_0 = 0.05, maxiter = 40)
    set_symmetry!(:SU2)
    W = heisenberg_su2_mpo(L)                       # MPO. Never `heisenberg_su2_chain`.
    psi0 = dimer_state(L)                           # :SU2 admits the dimer, not Neel

    eref, _, res = bethe_xxx_open_energy(L)
    res < 1e-12 || error("Bethe residual $res -- reference not trustworthy")
    @printf("HEISENBERG SU(2) CHAIN  L=%d  maxdim=%d  tau_max=%g   E_exact = %.12f\n",
            L, D, tau_max, eref)
    @printf("  %-26s %6s %12s %10s %5s %8s\n",
            "arm", "steps", "E - E_exact", "matvec", "chi", "time")

    bug(p, dt)  = cbe_bug_step!(p, W, ComplexF64(-dt); maxdim = D, trunc_thresh = 1e-12,
                                maxiter = maxiter, exact = true)
    tdvp(p, dt) = tdvp2_step!(p, W, ComplexF64(-dt); maxdim = D, trunc_thresh = 1e-12,
                              maxiter = maxiter)

    arms = []
    for n in (12, 24, 48)
        push!(arms, ("bug   LOG  n=$n",  bug,  log_steps(tau_0, tau_max, n)))
        push!(arms, ("tdvp2 LOG  n=$n",  tdvp, log_steps(tau_0, tau_max, n)))
    end
    # uniform stepping at the same dt as the log grid's FIRST step -- what their comparison uses
    for n in (200, 1200)
        push!(arms, ("tdvp2 UNIF n=$n", tdvp, uniform_steps(tau_max, n)))
        push!(arms, ("bug   UNIF n=$n", bug,  uniform_steps(tau_max, n)))
    end

    for (nm, st, steps) in arms
        t0 = time()
        e, kry, _, chi = try
            evolve(st, psi0, W, steps)
        catch err
            @printf("  %-26s %6d %12s %10s %5s %7.1fs\n", nm, length(steps),
                    "FAILED", "-", "-", time() - t0)
            @info "arm failed" nm err
            continue
        end
        @printf("  %-26s %6d %12.4e %10d %5d %7.1fs\n",
                nm, length(steps), e - eref, kry, chi, time() - t0)
        flush(stdout)
    end
end

run_chain()
