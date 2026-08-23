# THE FAIR COMPARISON: ERROR AT EQUAL COST, NOT AT EQUAL dt.
#
# ⛔ EVERY COMPARISON IN THIS REPO SO FAR HAS BEEN AT EQUAL `dt`, AND THAT IS NOT A FAIR TEST.
# `cbe_bug` uses 3.5-4.4x fewer Krylov applications per step than `tdvp_cbe1s` -- measured
# repeatedly: 14763 vs 50917 (TFIM), 21591 vs 75383 (XX), 2946 vs 9962 (long-range XX). Judging
# it against `tdvp_cbe1s` at the same `dt` therefore hands `tdvp_cbe1s` 3.5x the compute and
# reports the difference as accuracy.
#
# THE ARITHMETIC THAT MOTIVATES THIS. On XX at D=24, `cbe_bug` is ~5x behind at equal `dt` and
# ~3.5x cheaper per step. Second order means halving `dt` quarters the error, so spending the
# saved compute on 3.5x more substeps should drop the error ~12x -- putting `cbe_bug` roughly
# 2.4x AHEAD at identical cost. That is a prediction, and this file tests it.
#
# THE OUTPUT IS A PARETO FRONT: rows sorted by `krylov`, so "which scheme is more accurate for
# this compute budget" can be read off directly. A scheme wins where its row sits below and left
# of the others.
#
# ⛔ SUBSTEPPING IS NOT FREE ONCE THE RANK CAP BINDS. Below the point where truncation dominates
# the time-step error, more substeps buy NOTHING -- the error floors at the truncation level and
# only the cost keeps rising. So the front is expected to bend flat, and WHERE it bends is the
# useful number: it says how much of `cbe_bug`'s cost advantage is actually spendable.
#
# MODELS ARE CHOSEN SO THE BASELINE IS NOT EXACT. TFIM at full rank is useless for this -- the
# TDVP family is EXACT there by the projector-splitting property (measured: machine precision at
# every state and observable tried, while `cbe_bug` is a consistent dt^4), so any ratio against
# it is unbounded and means nothing. Both models below run with the cap BINDING.
#
# Run:  julia --project=. benchmarks/cost_fair.jl
#       julia --project=. benchmarks/cost_fair.jl model=xx

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

function tfim_majorana_generator(L::Int; J = 1.0, h = 1.0)
    A = zeros(Float64, 2L, 2L)
    for j in 1:L
        A[2j - 1, 2j] = -2h; A[2j, 2j - 1] = +2h
    end
    for j in 1:(L - 1)
        A[2j, 2j + 1] = -2J; A[2j + 1, 2j] = +2J
    end
    return A
end

function tfim_x_exact(L::Int, t::Real, dirs::Vector{Symbol}; J = 1.0, h = 1.0)
    G = zeros(Float64, 2L, 2L)
    for (j, d) in enumerate(dirs)
        s = d === :plus ? 1.0 : -1.0
        G[2j - 1, 2j] = s; G[2j, 2j - 1] = -s
    end
    R  = exp(tfim_majorana_generator(L; J = J, h = h) * t)
    Gt = R * G * transpose(R)
    return [Gt[2j - 1, 2j] for j in 1:L]
end

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

"""
Sweep `dt` for every scheme and print the rows sorted by cost.

`maxiter` is held FIXED across `dt`, deliberately. `lanczos_expv` exits on breakdown only, so
every solve burns its whole budget and `krylov` is then proportional to the number of solves --
which is what makes the cost axis comparable between schemes at different `dt`.
"""
function pareto(title, mpo, psi0, T, dts, D, maxiter, exact_at, measure)
    println("\n" * "="^80)
    @printf("%s   maxdim=%d (BINDING)  T=%g\n", title, D, T)
    println("Sorted by cost. A scheme wins where its row is lower AND left.")
    println("="^80)
    rows = Tuple{Int, Float64, String, Float64, Int}[]
    for (nm, mk) in ["tdvp2"      => (t) -> ((p) -> tdvp2_step!(p, mpo, t; maxdim = D,
                                              trunc_thresh = 1e-12, maxiter = maxiter)),
                     "tdvp_cbe1s" => (t) -> ((p) -> tdvp_cbe1s_step!(p, mpo, t; maxdim = D,
                                              trunc_thresh = 1e-12, maxiter = maxiter)),
                     "cbe_bug"    => (t) -> ((p) -> cbe_bug_step!(p, mpo, t; maxdim = D,
                                              trunc_thresh = 1e-12, maxiter = maxiter))]
        for dt in dts
            n = round(Int, T / dt)
            st = mk(ComplexF64(-im * dt))
            psi, kry = copy(psi0), 0
            for _ in 1:n
                kry += _kry(st(psi))
            end
            push!(rows, (kry, measure(psi, exact_at(T)), nm, dt, maximum(bond_dims(psi))))
        end
        flush(stdout)
    end
    sort!(rows; by = first)
    @printf("\n  %10s  %-12s %8s %12s  %s\n", "krylov", "scheme", "dt", "err", "chi")
    for (k, e, nm, dt, c) in rows
        @printf("  %10d  %-12s %8g %12.4e  %d\n", k, nm, dt, e, c)
    end
end

function run_xx(; L = 16, T = 6.0, D = 24, maxiter = 12)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    pareto("XX domain wall  L=$L  :U1", mpo, domain_wall_state(L), T,
           (0.1, 0.05, 0.025, 0.0125), D, maxiter,
           t -> xx_free_fermion_sz(L, t; J = 1.0),
           (p, r) -> maximum(abs.(magnetisation(copy(p)) - r)))
end

function run_tfim(; L = 16, T = 3.0, D = 16, maxiter = 16)
    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    mpo  = tfim_mpo(L; J = 1.0, h = 1.0)
    # ⛔ D=16 AT L=16 BINDS HARD -- the Schmidt ceiling here is 256. That is the point: at a
    # generous cap the TDVP family is exact and the comparison is vacuous.
    pareto("TFIM  L=$L  :Z2  (h = J, critical)", mpo, ising_kink_state(L), T,
           (0.1, 0.05, 0.025, 0.0125), D, maxiter,
           t -> tfim_x_exact(L, t, dirs),
           (p, r) -> maximum(abs.(x_profile(copy(p)) - r)))
end

let model = "both"
    for a in ARGS
        startswith(a, "model=") && (model = split(a, '='; limit = 2)[2])
    end
    model in ("xx", "both")   && run_xx()
    model in ("tfim", "both") && run_tfim()
    println()
end
