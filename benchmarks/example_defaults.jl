# THE WIDTH-RESTORATION TABLE, AT THE EXAMPLE SCRIPTS' OWN SETTINGS.
#
# `examples/common.jl` justifies `rexpand = true` as the examples' scheme with a measured table.
# That table has to be re-measured whenever a default it depends on moves, and one has: `growth`
# went 1.1 -> 2.0, which is exactly the knob that was STARVING `rexpand` (its second expansion
# runs against a larger `U0`, and `budget = ceil(growth*dmax) - r` SUBTRACTS that `r`). The old
# table's "kaug keeps a 2-3x edge" was measured on the starved side of that change.
#
# ⛔ RUN AT THE EXAMPLES' SETTINGS, NOT THE BENCHMARKS'. TFIM here is L=16 maxdim=64 trunc=1e-10,
# not the L=16 D=32 trunc=1e-12 the sweep benchmarks use. Pasting a benchmark number into an
# example's docstring would be reporting a different configuration than the one the reader runs.
#
# The three arms are the three ways the half-sweep split's lost width can be handled:
#   rexpand  -- expand every bond a SECOND time (the default; no basis formed by merging)
#   kaug     -- union the CBE frame back in
#   neither  -- nothing restores it; the failure is SILENT (exit 0, CSV written, answer wrong)
#
# Run:  julia --project=. benchmarks/example_defaults.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

# ⛔ EVERY ARM PASSES `exact = true`, BECAUSE THE EXAMPLE SCRIPTS DO. The package default is
# `exact = false` (the randomised sketch), so an arm that simply omits the flag is NOT running
# what `examples/common.jl` runs -- and a table captioned "at these scripts' own settings" would
# then be measuring a different code path than the one it claims. The first version of this file
# omitted it.
const ARMS = [("rexpand = true (default)", (rexpand = true,  exact = true)),
              ("kaug    = true",          (rexpand = false, kaug = true, exact = true)),
              ("neither",                 (rexpand = false, kaug = false, exact = true)),
              # ⛔ THE SECOND FLAG THE EXAMPLES OVERRIDE, AND ITS JUSTIFICATION IS ALSO STALE.
              # `steppers` passes `exact = true` (the package default is `false`), justified in
              # its docstring as free -- "bit-identical to the exact SVD, relative difference
              # 0.000e+00", from `rsvd_param_scan.jl` phase 0. That campaign ran under `kaug` at
              # `growth = 1.1`, the same superseded defaults that made the width-restoration
              # table wrong. The claim may well still hold; it is simply not currently backed by
              # a measurement of the code AS IT SHIPS, so it gets re-pinned here alongside.
              ("rexpand + randomised SVD", (rexpand = true, exact = false))]

# ⛔ ONE LINE PER ARM, PRINTED THE MOMENT THAT ARM FINISHES. The first version of this file
# collected a whole symmetry column into a dict and printed only when BOTH were done, so a run
# that takes 40 minutes shows NOTHING for 40 minutes and cannot be stopped early with partial
# results in hand. Progress visibility is not a nicety on a run this size: it is what lets a
# too-slow configuration be recognised as too slow instead of as hung.
function arm_errors(tag, mpo, psi0, nst, tau, D, trunc, maxiter, measure)
    out = String[]
    for (lbl, kw) in ARMS
        t0 = time()
        psi = copy(psi0)
        for _ in 1:nst
            cbe_bug_step!(psi, mpo, tau; maxdim = D, trunc_thresh = trunc,
                          maxiter = maxiter, kw...)
        end
        e = measure(psi)
        push!(out, @sprintf("%.4e", e))
        # the elapsed column is for SIZING this benchmark, never for comparing schemes -- local
        # wall time spreads 2.6-2.8x on bit-identical work.
        @printf("  %-10s %-26s %12.4e   chi=%-4d  [%.0fs]\n", tag, lbl, e,
                maximum(bond_dims(psi)), time() - t0)
        flush(stdout)
    end
    return out
end

# ── TFIM, the model the table is about: L=16, maxdim=64, both symmetry modes ───────────────
function tfim_table(; L = 16, T = 3.0, dt = 0.05, D = 64, trunc = 1e-10, J = 1.0, h = 1.0,
                    maxiter = 16)
    nst  = round(Int, T / dt)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    ref  = tfim_x_profile(L, nst * dt, dirs; J = J, h = h)
    println("\nTFIM  L=$L  maxdim=$D  trunc=$trunc  t=$T  dt=$dt   (examples/tfim_z2.jl defaults)")
    flush(stdout)
    rows = Dict{Symbol, Vector{String}}()
    for sym in (:Z2, :none)
        set_symmetry!(sym)
        rows[sym] = arm_errors(String(sym), tfim_mpo(L; J = J, h = h), ising_kink_state(L), nst,
                               ComplexF64(-im * dt), D, trunc, maxiter,
                               p -> maximum(abs.(x_profile(copy(p)) - ref)))
    end
    println("\n  -- the table, as it goes into examples/common.jl --")
    @printf("  %-26s %12s %12s\n", "mechanism", ":Z2", ":none")
    for (i, (lbl, _)) in enumerate(ARMS)
        @printf("  %-26s %12s %12s\n", lbl, rows[:Z2][i], rows[:none][i])
    end
    flush(stdout)
end

# ── XX and OAT: the CONTROLS. Both were already at parity and must stay there. ─────────────
function xx_table(; L = 16, T = 6.0, dt = 0.05, D = 64, trunc = 1e-10, J = 1.0, maxiter = 12)
    set_symmetry!(:U1)
    nst = round(Int, T / dt)
    ref = xx_free_fermion_sz(L, nst * dt; J = J)
    println("\nXX domain wall  L=$L  :U1  maxdim=$D  trunc=$trunc  t=$T  dt=$dt")
    arm_errors("U1", xxz_mpo(L; J = J, delta = 0.0), domain_wall_state(L), nst,
               ComplexF64(-im * dt), D, trunc, maxiter,
               p -> maximum(abs.(magnetisation(copy(p)) - ref)))
end

function oat_table(; L = 10, T = 10.0 * pi, dt = 0.05, trunc = 1e-14, maxiter = 30)
    set_symmetry!(:none)
    D    = min(L ÷ 2, L - L ÷ 2) + 1                 # the exact ceiling the example uses
    nst  = round(Int, T / dt)
    psi0 = x_polarized_state(L)
    v0   = dense_state(psi0)
    vex  = exp.(-im * (nst * dt) .* ((diag(dense_total_sz(L)) .^ 2) ./ 2 .- L / 8)) .* v0
    println("\nOAT  L=$L  :none  maxdim=$D (exact ceiling)  trunc=$trunc  t=$(T/pi)π  dt=$dt")
    @printf("  %-26s %12s  %s\n", "mechanism", "infidelity", "bond profile")
    for (lbl, kw) in ARMS
        psi = copy(psi0)
        for _ in 1:nst
            cbe_bug_step!(psi, oat_mpo(L), ComplexF64(-im * dt); maxdim = D,
                          trunc_thresh = trunc, maxiter = maxiter, kw...)
        end
        vn = dense_state(psi)
        # ⛔ THE BOND PROFILE IS THE TELL ON OAT, NOT THE ERROR. The exact Schmidt profile is
        # [2,3,4,5,6,5,4,3,2] for all time; an unrestored arm over-fills the interior and sits at
        # the cap, which reads deceptively like the cap simply binding.
        @printf("  %-26s %12.4e  %s\n", lbl,
                1 - abs2(dot(vex, vn)) / real(dot(vex, vex) * dot(vn, vn)), bond_dims(psi))
        flush(stdout)
    end
end

tfim_table(); xx_table(); oat_table()
println()
