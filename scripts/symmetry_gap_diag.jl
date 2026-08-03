# Is `max |<Sz>_none - <Sz>_U1| = 2.17e-04` at L=8 a STORAGE BUG, or two approximations
# differing within their own accuracy?
#
# The example asserts the gap must be < 1e-10 whenever neither run hit `maxdim`. That premise is
# wrong for a rank-GROWING method: the state starts at chi=1, and what limits its rank at an
# intermediate step is the `growth` schedule and the CBE selection, not `maxdim`. Two runs that
# select different expansion directions land on different states even when `maxdim` never binds.
# `:none` sketches with one dense Gaussian block, `:U1` with a block-diagonal one per charge
# sector -- different draws, hence different subspaces.
#
# THE DECISIVE QUESTION. If the gap is approximation error it must vanish as the approximation is
# removed, and vanish at the same rate as the error itself. If it plateaus while the error falls,
# the two storage paths really do disagree and it IS a bug.
#
# So drive all three knobs to convergence at once:
#   dt      -> 0        removes the Strang splitting error
#   exact   = true      removes the RSVD's random draw (deterministic CBE selection)
#   maxdim  = full      at L=8 the centre bond maxes at 16, so 64 can never bind
#   growth  large       lets the rank reach full in as few steps as possible
#
# At L=8 with no rank restriction anywhere the MPS is FULL RANK and can represent the state
# exactly, so a converged run must agree with the closed form AND across symmetries to round-off.
#
# Usage (submit to a compute node; never run on the login node):
#   L=8 TMAX=2.0 DELTA=0 sbatch scripts/run_julia.sbatch scripts/symmetry_gap_diag.jl
# Env knobs: L (default 8), TMAX (default 2.0), DELTA (default 0.0).

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime
using LinearAlgebra, Printf
import Random

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const L     = parse(Int,     get(ENV, "L",     "8"))
const TMAX  = parse(Float64, get(ENV, "TMAX",  "2.0"))
const DELTA = parse(Float64, get(ENV, "DELTA", "0.0"))

want = xx_free_fermion_sz(L, TMAX; J = 1.0)

function run(sym, dt, chi, exact, growth)
    Random.seed!(1234)                     # same draw for both symmetries where one is used
    set_symmetry!(sym)
    psi   = domain_wall_state(L)
    gates = bond_gates(psi; J = 1.0, delta = DELTA)
    opts  = CBEBugOptions(; dt = dt, n_steps = round(Int, TMAX / dt), maxdim = chi,
                            s_iters = -1, maxiter = 30, exact = exact, growth = growth)
    info = RealTime.evolve!(psi, gates; opts = opts)
    return magnetisation(copy(psi)), maximum(info.max_bond_dims)
end

@printf("L=%d  delta=%.1f  t=%.1f   reference: xx_free_fermion_sz (closed form)\n", L, DELTA, TMAX)
@printf("centre bond maxes at %d, so maxdim=64 cannot bind\n\n", 2^(L ÷ 2))

@printf("%-7s %-7s %-8s %-7s %-11s %-11s %-11s %-9s\n",
        "exact", "growth", "dt", "max_bd", "err_none", "err_U1", "|none-U1|", "ratio")
for exact in (true, false), growth in (2.0, 8.0)
    prev = NaN
    for dt in (0.05, 0.025, 0.0125, 0.00625)
        mn, bn = run(:none, dt, 64, exact, growth)
        mu, bu = run(:U1,   dt, 64, exact, growth)
        en = maximum(abs.(mn .- want))
        eu = maximum(abs.(mu .- want))
        gap = maximum(abs.(mn .- mu))
        @printf("%-7s %-7.1f %-8.5f %-7d %-11.3e %-11.3e %-11.3e %-9s\n",
                exact, growth, dt, max(bn, bu), en, eu, gap,
                isnan(prev) ? "-" : @sprintf("%.2f", prev / gap))
        prev = gap
        flush(stdout)
    end
    println()
end

# A ratio near 4 means the gap is 2nd order in dt -- i.e. it IS the splitting error and shares
# the integrator's convergence order. A ratio near 1 means it is a floor that refinement cannot
# reach, which would be the signature of a genuine storage disagreement.
println("""
ratio = gap(previous dt) / gap(this dt).  ~4 => the gap is O(dt^2), the same order as the
integrator, so it is approximation error and NOT a storage bug.  ~1 => a floor, i.e. real.""")
