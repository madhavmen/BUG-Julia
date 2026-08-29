# THE L = 8 MPDO VALIDATION, AT PARAMETERS THAT FINISH IN MINUTES.
#
# ⛔ THIS IS A SCHEDULING CHOICE, NOT A WEAKER TEST. `dissipative_xx.jl calibrate` runs `dt = 0.005`
# out to `t = 3.0` -- 600 steps per arm, and `evolve_arm` returns the whole trajectory before
# anything is printed, so the first row costs the full 600. MEASURED: >30 CPU-minutes for arm 1
# alone once the jump-convention fix made the state actually EVOLVE (it was frozen at `chi = 1`,
# and therefore instant, before). That is the right run to have and the wrong one to wait on while
# deciding whether the generator is correct at all.
#
# ⚠ WHAT IS AND IS NOT GIVEN UP. Shorter horizon and larger `dt` mean this cannot speak to LONG-TIME
# accuracy or to the steady state -- `ness` is the run for those and it belongs on the cluster. What
# it does test is exactly what "is the MPDO right" asks: at `L = 8` the superket has 8 fused d=4
# sites and truncation is live (unlike the L=2/L=3 gates, where the block IS the whole space and a
# two-site update is exact by construction), so a wrong generator, a wrong observable or a wrong
# frame update has somewhere to show.
#
# ⛔ THE REFERENCE IS THE ANALYTIC 8x8 LYAPUNOV SOLVE, exact at EVERY `t`, `O(N^3)`, no dense object.
# See `tests/common/free_fermion.jl` (`xx_boundary_*`), certified against a small dense Liouvillian
# at L = 2..5 to 7.2e-16 by `benchmarks/imaginary_time/certify_xx_reference.jl`.
#
# ⚠ `order` HERE IS A CONSISTENCY CHECK, NOT AN ORDER MEASUREMENT, and must not be quoted as one.
# At L = 8 with `cap = 128` the run is far from rank-limited, so a `dt`-halving scan measures the
# roundoff floor rather than the scheme's exponent -- the standing rule that a dt-order test needs a
# RANK-LIMITED regime. What it does catch is the failure this file exists because of: an error that
# does NOT move with `dt` is a state that is not moving at all.
#
# Run:  julia -t 2 --project=. benchmarks/imaginary_time/mpdo_validate_fast.jl

# `dissipative_xx.jl` guards its driver with `abspath(PROGRAM_FILE) == @__FILE__`, so including it
# here brings in the model, the arms and the checks WITHOUT firing its `all` dispatch.
include(joinpath(@__DIR__, "dissipative_xx.jl"))

println("\n#### FAST L=8 MPDO VALIDATION ####")
flush(stdout)

# ⛔ EVERY `dt` HERE DIVIDES EVERY SNAPSHOT TIME EXACTLY, AND THAT IS A REQUIREMENT, NOT TIDINESS.
# `evolve_arm` takes `round(Int, (t - tnow)/dt)` steps, so a `dt` that does not divide the gap stops
# SHORT. That is now scored honestly (the reference is closed-form at the time actually reached, and
# `order` prints `t_act`), but a scan whose rows sit at DIFFERENT times is still comparing unlike
# things -- the ratio between them is not a clean order estimate.
#
# MEASURED with the earlier `dts = (0.04, 0.02, 0.01, 0.005)` at `t = 0.25`: `round(6.25) = 6` and
# `round(12.5) = 12` (Julia rounds half to EVEN) put the first TWO rows both at `t = 0.24`, so they
# agreed to four digits and reported `order 0.00`. The gate grades on the FIRST ratio, so it
# printed `ORDER FAILED` for arms whose real orders are 2.99 (tdvp2) and 1.98 (bug_rsvd).
#
#   0.25 / 0.05 = 5    0.25 / 0.025 = 10    0.25 / 0.0125 = 20    0.25 / 0.00625 = 40
#   0.50 / 0.025 = 20  (calibrate's second snapshot, from the first)
ok = calibrate(; L = 8, ts = (0.25, 0.5), dt = 0.025)
ok &= order(; L = 8, t = 0.25, dts = (0.05, 0.025, 0.0125, 0.00625))

println(ok ? "\nFAST VALIDATION PASSED" : "\nFAST VALIDATION FAILED")
flush(stdout)
