# ONE-AXIS TWISTING under the paper's own Z2 symmetry -- arXiv:2208.10972 Fig. 2.
#
#     H = (Σ_ℓ S^z_ℓ)² / 2 ,     |Ψ(0)⟩ = ⊗_ℓ |→⟩ ,     observable  S^x_tot(t)
#
# WHY THIS MODEL IS THE REFERENCE TEST. Everything about it is closed form, with no
# diagonalisation and no second integrator anywhere:
#
#     S^x_tot(t) = (L/2)·cos^(L-1)(t/2)
#     exact Schmidt rank at the cut after site n  =  min(n, L-n) + 1,  at GENERIC t
#
# The rank being constant in time is what makes `maxdim` meaningful here rather than a tolerance:
# at or above the ceiling the sweeps are exact and what remains is the time-step error, below it
# the truncation is genuine. Both regimes are worth running, and a comparison that shows only one
# of them is measuring roundoff or truncation rather than the integrator.
#
# ⛔ BUT "FOR ALL TIME" IS WRONG, AND THIS FILE SAID SO UNTIL 2026-08-27. THE RANK COLLAPSES AT
# t = π. `H = (S^z_tot)²/2`, so a state of definite `m` picks up `exp(-i t m²/2)`; at `t = π` that
# is `exp(-iπ m²/2)`, and `m² mod 4 ∈ {0, 1}`, so the phase takes only **TWO** distinct values --
# one for even `m`, one for odd. The state is therefore a TWO-BRANCH superposition at `t = π`: a
# cat state, of Schmidt rank EXACTLY 2 at every cut, not `min(n, L-n) + 1`.
#
# ⚠ THIS WAS READ AS A BUG IN `cbe_bug`, AND IT IS THE OPPOSITE. MEASURED, L=10, t=π, maxdim=6:
#
#     tdvp2        final err 3.816e-06   chi = 6   krylov 25914
#     tdvp_cbe1s   final err 3.816e-06   chi = 6   krylov 22935
#     cbe_bug      final err 8.271e-15   chi = 2   krylov  2684
#
# `cbe_bug` FINDS the rank-2 structure and is nine orders more accurate for a tenth of the work;
# the TDVP arms carry rank 6 with four near-zero Schmidt values. A `chi` that DROPS here is the
# correct answer, and reading the bond dimension as "did not grow" inverts the result. The run loop
# in `common.jl` now prints the whole bond profile for exactly this reason -- one number cannot
# tell a collapse-to-the-true-rank from a freeze.
#
# THE Z2 IS THE PAPER'S. The conserved quantity is the global spin flip P = Π_ℓ 2S^x_ℓ, NOT
# S^z_tot -- H is even in S^z. In the σ^x basis that flip is diagonal, so S^x is measurable and
# ⊗|→⟩ is a single charge-0 sector at χ=1 (the paper's "an MPS with D = 1"). `symmetry=none` is
# the unsymmetric control; the two agree to ~1e-12.
#
# The state revives every 4π, so the default 10π covers 2.5 revivals -- enough to tell a bounded
# error from an accumulating one, which a single period cannot.
#
#   julia --project=. examples/oat_z2.jl
#   julia --project=. examples/oat_z2.jl L=12 t_over_pi=4 dt=0.05 maxdim=16 symmetry=none
#   python3 examples/plot_examples.py

include(joinpath(@__DIR__, "common.jl"))

# ── PARAMETERS -- EDIT THESE (or override as name=value on the command line) ──────────────
const P = parse_params((
    L            = 10,      # sites
    t_over_pi    = 10.0,    # total time in units of π. 4π is one full revival cycle.
    dt           = 0.05,    # time step
    maxdim       = 0,       # 0 = use the exact ceiling min(n,L-n)+1; any other value caps there
    trunc_thresh = 1e-14,   # CBE root-to-leaves truncation threshold at the end of each sweep
    maxiter      = 0,       # 0 = derive the Krylov depth from ‖H‖·dt/2 (see below)
    sample_every = 0.2,     # record a row every this much physical time
    symmetry     = "Z2",    # "Z2" (the paper's) or "none" (the dense control)
))
# ──────────────────────────────────────────────────────────────────────────────────────────

const T_MAX = P.t_over_pi * π


const SPP = let s = max(4, round(Int, π / P.dt)); s + (4 - s % 4) % 4 end
const DT  = π / SPP                     # ~0.0491 for the requested 0.05; divides π exactly

const CEIL  = min(P.L ÷ 2, P.L - P.L ÷ 2) + 1        # the exact rank ceiling
const MAXDIM = P.maxdim == 0 ? CEIL : P.maxdim

# ‖H_OAT‖ ≈ L²/8: the largest eigenvalue of (Σ S^z)²/2 is (L/2)²/2. It grows QUADRATICALLY, so
# a depth fixed at 12 bounds 1e-15 at L=10 but only 1e-10 at L=16 and 3e-8 at L=20 -- see
# `krylov_depth` in common.jl for why a Krylov-limited run is the quiet failure to fear here.
const HNORM   = P.L^2 / 8
const MAXITER = P.maxiter == 0 ? krylov_depth(HNORM, DT) : P.maxiter

# The closed form, written out here rather than imported from `tests/common/analytic_reference.jl`
# so this example stands on its own -- a downloaded copy of the package should not need the test
# tree to run. Derivation: each spin picks up a phase from the (L-1) others, and the product of
# those independent cosines is what collapses and revives.
oat_sx_exact(L::Int, t::Real) = (L / 2) * cos(t / 2)^(L - 1)

set_symmetry!(Symbol(P.symmetry))

const MPO_OAT = oat_mpo(P.L)
const PSI0    = x_polarized_state(P.L)

# ⛔ REPORT THE STEP THAT RAN, NOT THE ONE REQUESTED. Printing `P.dt` here would reintroduce
# exactly the defect `SPP` fixes: a header naming a time grid the run did not use.
@printf("OAT  L=%d  symmetry=%s  t=%gπ (%.3f)  dt=π/%d=%.6g  maxdim=%d%s  trunc=%g\n",
        P.L, P.symmetry, P.t_over_pi, T_MAX, SPP, DT, MAXDIM,
        P.maxdim == 0 ? " (exact ceiling)" : "", P.trunc_thresh)
@printf("krylov depth = %d  (arg %.3f, Lanczos bound %.1e -- must sit under %g)\n",
        MAXITER, HNORM * DT / 2, krylov_bound(HNORM, DT, MAXITER), P.trunc_thresh)
@printf("exact rank ceiling = %d   initial chi = %d   S^x_tot(0) = %.6f\n\n",
        CEIL, maximum(bond_dims(PSI0)), total_sx(copy(PSI0)))

const OUT = joinpath(RESULTS,
                     @sprintf("oat_L%d_%s_D%d_dt%g_T%gpi.csv",
                              P.L, P.symmetry, MAXDIM, DT, P.t_over_pi))
io = open_csv(OUT)
try
    run_model(io, "OAT", P.symmetry, PSI0, MPO_OAT,
              psi -> total_sx(psi),                  # observable
              t   -> oat_sx_exact(P.L, t),            # its closed form
              (; t_max = T_MAX, dt = DT, maxdim = MAXDIM,
                 trunc_thresh = P.trunc_thresh, maxiter = MAXITER,
                 sample_every = P.sample_every))
finally
    close(io)
end
println("\nwrote ", OUT)
