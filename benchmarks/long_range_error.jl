# Error analysis of the two BUG sweeps on a LONG-RANGE Hamiltonian.
#
#   jan_bug!         KLS at every bond, R discarded, the kept S step on the END bond,
#                    direction alternating between steps.
#   cbe_lubich_bug!  the MPS reading of the Lubich TTN BUG: two basis half-sweeps and one
#                    Galerkin step on the CENTRE bond.
#
# WHY A LONG-RANGE MODEL SEPARATES THEM AT ALL. With nearest-neighbour H every bond's
# effective Hamiltonian sees its neighbours only, so where the single Galerkin step sits
# matters least at the ends. A geometric tail couples every pair, so the centre bond
# carries interactions that cross it from both halves and the boundary bond does not --
# which is exactly the structural difference between the two schemes.
#
# TWO HAMILTONIANS, AND THE DISTINCTION IS THE POINT:
#
#   exact   `long_range_zz_mpo`  — a geometric tail is carried EXACTLY by one self-loop, so
#                                  the measured error is the INTEGRATOR's, full stop.
#   fitted  `power_law_zz_mpo`   — 1/r^alpha fitted by n geometrics, so the error has a
#                                  MODEL floor that no dt and no rank can go below.
#
# Reporting only the second would confuse an integrator error with a Hamiltonian error, so
# the order study runs on the first and the floor is measured separately.
#
# LOCAL RUN, SMALL L, CORRECTNESS ONLY. L=6 so a dense 2^L propagator is the reference. No
# wall-clock is reported: timing and ranking belong on the cluster, and a JIT-dominated
# local run would say nothing about either.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "tests", "common", "dense_reference.jl"))

set_symmetry!(:U1)

const L      = 6
const LAMBDA = 0.5
const ALPHA  = 3.0
const TFINAL = 0.24

"A state with real entanglement to start from, the same one for every arm."
function warm(L::Int; n_steps = 6, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

const METHODS = [("jan", jan_bug!), ("lubich/mendl", cbe_lubich_bug!)]

"Run one arm and return (error against `want`, final bond dims, max bond dim, norm)."
function arm(run!, mpo, psi0, want; dt, n, maxdim)
    psi = copy(psi0)
    info = run!(psi, mpo; opts = CBEBugOptions(dt = dt, n_steps = n, maxdim = maxdim,
                                               normalize = false))
    v = dense_state(psi)
    return (err = norm(v - want), bd = bond_dims(psi),
            maxb = maximum(info.max_bond_dims; init = 0), nrm = norm(psi))
end

function section(title)
    println()
    println(title)
    println(repeat("-", length(title)))
end

psi0 = warm(L)
v0   = dense_state(psi0)

# ── 1. order in dt, on the EXACT long-range H (no model error) ────────────────
section("1. global error vs dt   |   H = XY + Jz sum_{i<j} $(LAMBDA)^(r-1) Sz Sz   (exact MPO)")

H_ex  = dense_long_range_zz(L; lambda = LAMBDA)
want  = exp(-im * TFINAL * H_ex) * v0
mpo_ex = long_range_zz_mpo(L; lambda = LAMBDA)
println(@sprintf("  standstill (doing nothing) = %.3e", norm(want - v0)))
println(@sprintf("  %-14s %8s %12s %12s %10s", "method", "dt", "err", "order", "max chi"))
prev = Dict{String, Tuple{Float64, Float64}}()
for dt in (0.04, 0.02, 0.01, 0.005)
    n = round(Int, TFINAL / dt)
    for (name, run!) in METHODS
        r = arm(run!, mpo_ex, psi0, want; dt = dt, n = n, maxdim = 32)
        p = haskey(prev, name) ? log2(prev[name][2] / r.err) / log2(prev[name][1] / dt) : NaN
        prev[name] = (dt, r.err)
        println(@sprintf("  %-14s %8.4f %12.3e %12s %10d",
                         name, dt, r.err, isnan(p) ? "-" : @sprintf("%.2f", p), r.maxb))
    end
end

# ── 2. error at FIXED RANK ────────────────────────────────────────────────────
section("2. error at fixed rank (dt = 0.01, exact long-range H); full rank = [2,4,8,4,2]")
println(@sprintf("  %-14s %8s %12s %20s", "method", "maxdim", "err", "final bond dims"))
for maxdim in (2, 4, 8, 16, 32)
    for (name, run!) in METHODS
        r = arm(run!, mpo_ex, psi0, want; dt = 0.01, n = round(Int, TFINAL / 0.01),
                maxdim = maxdim)
        println(@sprintf("  %-14s %8d %12.3e %20s", name, maxdim, r.err, string(r.bd)))
    end
end

# ── 3. the MODEL floor: a fitted power law ────────────────────────────────────
section("3. fitted 1/r^$(ALPHA): the fit error is a floor no dt can cross")
println(@sprintf("  %-6s %12s %12s %14s %14s",
                 "n_exp", "fit max rel", "fit l2 rel", "err jan", "err lubich"))
H_pl  = dense_power_law_zz(L; alpha = ALPHA)
wantp = exp(-im * TFINAL * H_pl) * v0
for n_exp in (1, 2, 3, 4, 6)
    mpo, fit = power_law_zz_mpo(L; alpha = ALPHA, n_exp = n_exp)
    row = [arm(run!, mpo, psi0, wantp; dt = 0.005,
               n = round(Int, TFINAL / 0.005), maxdim = 32).err
           for (_, run!) in METHODS]
    println(@sprintf("  %-6d %12.3e %12.3e %14.3e %14.3e",
                     n_exp, fit.max_rel_err, fit.l2_rel_err, row[1], row[2]))
end

# ── 4. how far the tail reaches, as a control ─────────────────────────────────
section("4. control: the tail really is long-ranged")
for lam in (0.0, 0.3, 0.5, 0.9)
    H = dense_long_range_zz(L; lambda = lam)
    println(@sprintf("  lambda = %.1f   ||H - H_nn|| = %.4f   E0 = %.6f",
                     lam, norm(H - dense_long_range_zz(L; lambda = 0.0)),
                     real(dot(v0, H * v0))))
end

println()
println("LR_ERROR_DONE")
