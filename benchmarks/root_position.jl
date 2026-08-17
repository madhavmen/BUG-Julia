# WHY IS THE BOUNDARY-ROOT SWEEP LESS ACCURATE THAN THE CENTRE-ROOT ONE?
#
# Measured on the long-range model at L=6, full rank: `cbe_lubich_bug!` reaches 2.3e-15 and
# `jan_bug!` 7.4e-5 -- ten orders apart on two schemes that look almost the same. The two
# differ in TWO ways at once, and the whole purpose of this script is to stop guessing which
# one costs what by varying them ONE AT A TIME:
#
#   (A) WHERE THE GALERKIN STEP SITS. The single evolution explores exactly `b_L x b_R`
#       directions. At the CENTRE both factors are a full half-chain, so at full rank
#       `U_ex (x) V_ex` spans the whole space and the step is the exact propagator. At bond
#       `L-1` the right factor is ONE SITE, so `b_R <= d = 2` however wide the state is: the
#       variational space is smaller by a factor `chi/d` and can never be complete.
#
#   (B) WHETHER THE BASIS SWEEP TOUCHES THE STATE. `cbe_lubich_sweep` accumulates frames in
#       `W`/`Z` and never writes `psi` until the end -- lossless. `jan_bug_step!` writes at
#       every bond: `Q = orth(U_ex S1)` and the ORIGINAL amplitude re-expressed in it, which
#       PROJECTS, losing O(tau^2) per bond (the measured norm defect).
#
# THE DESIGN. `cbe_lubich_sweep` now takes `root`, so the middle row below is the centre
# scheme moved onto jan's bond with everything else identical:
#
#   1. lubich, root = centre     both (A) and (B) favourable        <- the reference
#   2. lubich, root = L-1        (A) as jan, (B) as lubich          <- isolates (A)
#   3. jan                       both as jan                        <- adds (B)
#
# So `2 - 1` is the cost of the root position and `3 - 2` the cost of discarding R. If row 2
# lands on row 3 the root explains everything; if row 2 stays near row 1, it does not and
# the write-back is the culprit. Either way the answer is read off, not argued.
#
# L=6, dense reference, correctness only -- no timing (that belongs on the cluster).

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "tests", "common", "dense_reference.jl"))

set_symmetry!(:U1)

const L = 6

function warm(L::Int; n_steps = 6, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

"One `cbe_lubich_sweep` at a chosen root, repeated `n` times."
function lubich_at(psi0, mpo, dt, n; root, maxdim = 32)
    psi = copy(psi0)
    for _ in 1:n
        cbe_lubich_sweep(psi, mpo, ComplexF64(-im * dt);
                         root = root, maxdim = maxdim,
                         rng = Random.MersenneTwister(0x5EED))
    end
    return psi
end

"`jan_bug_step!` repeated `n` times, direction alternating as the driver does."
function jan_at(psi0, mpo, dt, n; maxdim = 32, alternate = true)
    psi = copy(psi0)
    dir = :right
    for _ in 1:n
        jan_bug_step!(psi, mpo, ComplexF64(-im * dt);
                      direction = dir, maxdim = maxdim,
                      rng = Random.MersenneTwister(0x5EED))
        alternate && (dir = dir === :right ? :left : :right)
    end
    return psi
end

import Random

function section(t)
    println(); println(t); println(repeat("-", length(t)))
end

psi0 = warm(L)
v0 = dense_state(psi0)
println("bond dims of the start state: ", bond_dims(psi0))

for (hname, mpo, H) in (
        ("nearest-neighbour XXZ", xxz_mpo(L), Matrix(dense_heisenberg(L))),
        ("long-range (lambda=0.5)", long_range_zz_mpo(L; lambda = 0.5),
         Matrix(dense_long_range_zz(L; lambda = 0.5))))

    section("H = $hname")

    # ── one step, so nothing accumulates and (A) is seen bare ────────────────
    println("  ONE step, full rank (maxdim = 32)")
    println(@sprintf("    %-26s %12s %12s %12s", "scheme", "dt=0.04", "dt=0.02", "dt=0.01"))
    rows = Any[("lubich root=centre", (p, dt) -> lubich_at(p, mpo, dt, 1; root = nothing)),
               ("lubich root=$(L-1) (jan's bond)", (p, dt) -> lubich_at(p, mpo, dt, 1; root = L - 1)),
               ("lubich root=1", (p, dt) -> lubich_at(p, mpo, dt, 1; root = 1)),
               ("jan (root=$(L-1), R discarded)", (p, dt) -> jan_at(p, mpo, dt, 1))]
    for (name, f) in rows
        errs = [norm(dense_state(f(psi0, dt)) - exp(-im * dt * H) * v0)
                for dt in (0.04, 0.02, 0.01)]
        println(@sprintf("    %-26s %12.3e %12.3e %12.3e", name, errs...))
    end

    # ── the norm defect: (B) in isolation, since a lossless sweep has none ───
    println()
    println("  norm defect after ONE step (a lossless basis sweep has none)")
    for (name, f) in rows
        d = [abs(norm(f(psi0, dt)) - norm(psi0)) for dt in (0.04, 0.02, 0.01)]
        println(@sprintf("    %-26s %12.3e %12.3e %12.3e", name, d...))
    end

    # ── many steps at fixed T ────────────────────────────────────────────────
    T = 0.24
    println()
    println(@sprintf("  %d steps to T = %.2f, full rank", round(Int, T / 0.01), T))
    want = exp(-im * T * H) * v0
    println(@sprintf("    standstill = %.3e", norm(want - v0)))
    for (name, f) in (("lubich root=centre",
                       () -> lubich_at(psi0, mpo, 0.01, 24; root = nothing)),
                      ("lubich root=$(L-1)",
                       () -> lubich_at(psi0, mpo, 0.01, 24; root = L - 1)),
                      ("jan alternating",
                       () -> jan_at(psi0, mpo, 0.01, 24)),
                      ("jan one-directional",
                       () -> jan_at(psi0, mpo, 0.01, 24; alternate = false)))
        psi = f()
        println(@sprintf("    %-26s err = %.3e   chi = %s", name,
                         norm(dense_state(psi) - want), string(bond_dims(psi))))
    end
end

# ── the variational dimension, computed rather than asserted ─────────────────
section("the Galerkin space at each root, full rank (this is claim (A), as a number)")
psi = warm(L)
println("  bond dims: ", bond_dims(psi))
println(@sprintf("  %-8s %10s %10s %14s", "root", "b_L", "b_R", "b_L * b_R"))
for c in 1:(L - 1)
    # The frame at bond c before any expansion: b_L <= d*chi_{c-1}, b_R <= d*chi_{c+1}.
    bl = leg_dim(psi[c], 1) * leg_dim(psi[c], 2)
    br = leg_dim(psi[c + 1], 2) * leg_dim(psi[c + 1], 3)
    println(@sprintf("  %-8d %10d %10d %14d", c, bl, br, bl * br))
end
println("  (the U(1) Sz=0 sector of L=$L has dimension ", binomial(L, L ÷ 2), ")")

println()
println("ROOT_POSITION_DONE")
