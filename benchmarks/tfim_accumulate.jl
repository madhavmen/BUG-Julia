# WHY DOES `cbe_bug`'s TFIM ERROR COMPOUND WHERE `tdvp_cbe1s`'s CANCELS?
#
# WHAT IS ALREADY EXCLUDED (`tfim_anomaly.jl`, `tfim_bisect.jl`), so this file does not retest it:
#   * NOT a defective step. One step at dt=1e-2 is 9.76e-13 and FALLS to 4.55e-15 at dt=1e-3 --
#     214x for a 10x cut, no floor. The step is correct.
#   * NOT any one component. maxiter, truncate, trunc_thresh, kaug-vs-rexpand, exact-vs-sketch
#     and growth all return BIT-IDENTICAL 9.7622e-13 for that step.
#   * NOT truncation, rank, symmetry, interaction range, or dt order -- see those two files.
#
# WHAT IS LEFT. Single step: 73x behind `tdvp_cbe1s`. Ten steps: 9500x. `cbe_bug` went
# 9.76e-13 -> 2.12e-10, a factor of 217 where LINEAR accumulation predicts 10, while
# `tdvp_cbe1s` sat at machine precision the whole way. So the per-step error COMPOUNDS for one
# scheme and CANCELS for the other, and that is the thing to explain.
#
# THE DISCRIMINATOR IS THE GROWTH EXPONENT, so this traces error against STEP COUNT rather than
# reporting a final number:
#
#     err ~ n        linear accumulation -- an ordinary constant, nothing to fix
#     err ~ n^2      amplification -- each step degrades the basis the next one is built from
#     err ~ exp(n)   instability
#
# Two companions are traced beside it because they separate those cases:
#   * NORM. The Galerkin step is unitary IN ITS BASIS, so norm loss can only come from the basis
#     failing to contain the evolved state. Norm decaying while error grows means the basis is
#     the leak.
#   * ENERGY. Conserved exactly by the true dynamics. Drift that tracks the error says the
#     scheme is leaving the energy shell, which a frozen-basis Galerkin step can do and a
#     tangent-space projection cannot.
#
# ⛔ ENERGY IS A DIAGNOSTIC HERE, NEVER A RANKING. This repo's own rule: a scheme can conserve
# energy to machine precision and still be an order of magnitude further from the exact state.
# It is traced to explain the error, not to score the schemes.
#
# ⛔ RANK IS LEFT UNCONSTRAINED WHEREVER THE SIZE ALLOWS IT. At L=8 the Schmidt ceiling is 16, so
# maxdim=1024 can never bind and NOTHING is truncated at any step -- any growth seen there is
# pure scheme. The larger sizes then ask whether it survives when the cap does bind.
#
# Run:  julia --project=. benchmarks/tfim_accumulate.jl
#       julia --project=. benchmarks/tfim_accumulate.jl part=small

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

"""
Trace one scheme step by step. Prints `n`, the error, the norm defect, the energy drift and
`chi` at every `sample`-th step, then the fitted growth exponent.

The exponent is a least-squares slope of `log(err)` against `log(n)` over the LAST HALF of the
trace, so the transient at small `n` -- where the error is still near machine precision and the
fit would be dominated by round-off -- does not bias it.
"""
function trace(nm, step!, psi0, nsteps, sample, exact_at, measure, mpo)
    psi = copy(psi0)
    e0  = real(mpo_energy(copy(psi), mpo))
    ns, es = Int[], Float64[]
    @printf("    %-14s %6s %12s %11s %11s %5s\n", nm, "n", "err", "1-norm", "dE", "chi")
    for n in 1:nsteps
        step!(psi)
        if n % sample == 0
            err = measure(psi, exact_at(n))
            push!(ns, n); push!(es, err)
            @printf("    %-14s %6d %12.4e %11.2e %11.2e %5d\n", "", n, err,
                    1 - norm(psi), real(mpo_energy(copy(psi), mpo)) - e0,
                    maximum(bond_dims(psi)))
        end
    end
    # slope of log(err) vs log(n) over the second half
    h = max(1, length(ns) ÷ 2)
    x, y = log.(ns[h:end]), log.(max.(es[h:end], 1e-300))
    xm, ym = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- xm) .* (y .- ym)) / max(sum((x .- xm) .^ 2), eps())
    @printf("    %-14s -> err ~ n^%.2f  (1 = accumulation, 2 = amplification)\n\n", "", slope)
    return slope
end

function study(title, sym, mpo_f, psi0_f, exact_f, measure, dt, nsteps, sample, D, maxiter)
    println("\n" * "="^84)
    @printf("%s   maxdim=%d  dt=%g  %d steps\n", title, D, dt, nsteps)
    println("="^84)
    set_symmetry!(sym)
    mpo, psi0 = mpo_f(), psi0_f()
    tau = ComplexF64(-im * dt)
    for (nm, st) in ["tdvp2"      => p -> tdvp2_step!(p, mpo, tau; maxdim = D,
                                                      trunc_thresh = 1e-14, maxiter = maxiter),
                     "tdvp_cbe1s" => p -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = D,
                                                           trunc_thresh = 1e-14,
                                                           maxiter = maxiter),
                     "cbe_bug"    => p -> cbe_bug_step!(p, mpo, tau; maxdim = D,
                                                        trunc_thresh = 1e-14,
                                                        maxiter = maxiter, exact = true)]
        trace(nm, st, psi0, nsteps, sample, n -> exact_f(n * dt), measure, mpo)
        flush(stdout)
    end
end

tfim_study(L, D; dt = 0.01, nsteps = 200, sample = 20, maxiter = 30) =
    study("TFIM  L=$L  :Z2  (h = J, critical)", :Z2,
          () -> tfim_mpo(L; J = 1.0, h = 1.0), () -> ising_kink_state(L),
          t -> tfim_x_profile(L, t, [i <= L ÷ 2 ? :plus : :minus for i in 1:L]),
          (p, r) -> maximum(abs.(x_profile(copy(p)) - r)), dt, nsteps, sample, D, maxiter)

xx_study(L, D; dt = 0.01, nsteps = 200, sample = 20, maxiter = 30) =
    study("XX  L=$L  :U1  (the control)", :U1,
          () -> xxz_mpo(L; J = 1.0, delta = 0.0), () -> domain_wall_state(L),
          t -> xx_free_fermion_sz(L, t; J = 1.0),
          (p, r) -> maximum(abs.(magnetisation(copy(p)) - r)), dt, nsteps, sample, D, maxiter)

let part = "all"
    for a in ARGS
        startswith(a, "part=") && (part = split(a, '='; limit = 2)[2])
    end
    if part in ("small", "all")
        # L=8: the Schmidt ceiling is 16, so maxdim=1024 NEVER binds and nothing is truncated.
        tfim_study(8, 1024)
        xx_study(8, 1024)
    end
    if part in ("large", "all")
        # L=12 and L=16 with a generous cap: does it survive to larger systems and ranks?
        tfim_study(12, 512; nsteps = 200, sample = 25)
        xx_study(12, 512;   nsteps = 200, sample = 25)
        tfim_study(16, 256; nsteps = 150, sample = 25, dt = 0.02)
        xx_study(16, 256;   nsteps = 150, sample = 25, dt = 0.02)
    end
    println()
end
