# Does the expanding-basis Lanczos blow up in chi?
#
#     julia --project=. benchmarks/cbe_chi_scaling.jl
#
# The scheme grows the basis inside the Krylov iteration, so the worry is a runtime that
# degrades faster in chi than the one-shot it replaces. Three things are measured:
#
#   1. WALL TIME vs maxdim, as MIN OF THREE repeats. Minimum, not mean: timing noise on this
#      box is one-sided (interference only ever adds), and the identical configuration has
#      already been seen to swing 2.6x (docs 7.6.4). The minimum is the robust statistic.
#   2. The fitted exponent p in t ~ chi^p between consecutive maxdim values. If the expanding
#      solver's p exceeds the one-shot's, it degrades in chi and the feature is a liability at
#      scale however good it looks at chi = 12.
#   3. The REALISED rank, to check the structural claim that growth cannot compound. `room =
#      d*chi_neighbour - r` is measured against FIXED neighbour bond dimensions, so once a pass
#      saturates it the next pass has room = 0 and cannot grow. Rank should be bounded by
#      d*chi_neighbour no matter how many passes are allowed -- NOT D, alpha*D, alpha^2*D, ...
#
# L = 12 keeps this local (standing rule: L <= ~12 here, larger goes to the cluster) while
# still reaching chi = 64 at the centre.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf

set_symmetry!(:U1)

const L     = parse(Int, get(ENV, "L", "12"))
const DT    = 0.05
const TMAX  = 0.4
const NSTEP = round(Int, TMAX / DT)
const REPS  = 3

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

function analytic_sz(L::Int, t::Float64, n0::Vector{Float64}; J::Float64 = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    U = exp(-im * h * t)
    return [sum(abs2(U[j, k]) * n0[k] for k in 1:L) - 0.5 for j in 1:L]
end

const N0  = magnetisation(domain_wall_state(L)) .+ 0.5
const REF = analytic_sz(L, TMAX, N0)

function run_once(maxdim, nsteps; kwargs...)
    psi = domain_wall_state(L)
    gates = xx_gates(psi)
    info = cbe_bond_update_bug!(psi, gates;
                                opts = CBEBugOptions(; dt = DT, n_steps = nsteps,
                                                     maxdim = maxdim, trunc_thresh = 1e-12,
                                                     kwargs...))
    return psi, info
end

function measure(maxdim; kwargs...)
    run_once(maxdim, 2; kwargs...)                       # warm-up, untimed
    best = Inf
    local psi, info
    for _ in 1:REPS
        t = @elapsed ((psi, info) = run_once(maxdim, NSTEP; kwargs...))
        best = min(best, t)
    end
    err = norm(magnetisation(copy(psi)) - REF)
    return best, err, sum(info.krylov_dims), maximum(info.max_bond_dims),
           maximum(info.max_expanded)
end

const CAPS = [8, 16, 32, 64]

@printf("L = %d, dt = %.3f, t = %.2f (%d steps), min of %d repeats\n\n",
        L, DT, TMAX, NSTEP, REPS)

function scan(name; kwargs...)
    println(name)
    @printf("  %-8s %-9s %-7s %-12s %-9s %-8s %s\n",
            "maxdim", "wall(s)", "p", "err", "matvec", "maxbd", "max expanded")
    prevt, prevc = NaN, NaN
    for c in CAPS
        t, err, mv, bd, ex = measure(c; kwargs...)
        p = isnan(prevt) ? NaN : log(t / prevt) / log(c / prevc)
        @printf("  %-8d %-9.2f %-7s %-12.4e %-9d %-8d %d\n",
                c, t, isnan(p) ? "--" : @sprintf("%.2f", p), err, mv, bd, ex)
        prevt, prevc = t, c
    end
    println()
end

scan("one-shot default (growth=2.0)")
scan("one-shot + reorth"; s_reorth = true)
scan("expanding grow=2"; s_iters = -2)
scan("expanding grow=8"; s_iters = -8)

println("CHI_SCAN_DONE")
