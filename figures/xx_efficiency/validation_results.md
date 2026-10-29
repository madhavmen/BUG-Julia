# Validation results (S1-S8)

The tests live in `tests/sweeps/test_xx_efficiency.jl` on branch `plan/xx-efficiency`.
S1-S5 run for spin-1/2 (L = 12), spin-1 (L = 10) and spin-2 (L = 8) XX chains. The spin-S
reference is sector ED (`spin_ed.jl`).

**Final run:** xe6 (Slurm 17583863) at ae3ee31: **1765 / 1765 pass**, 103 min.

The regression suites `krylov_closure`, `adaptive_krylov`, `dd_cbe` and `dd_cbe_controller` passed in xe4 / xe5.

| ID | Result | Evidence |
|---|---|---|
| S1 | **pass** | Slater amplitudes equal ED, including signs, for L = 4, 6, 8, 16. ∑⟨S^z⟩ = 0, and norm and energy are conserved. The MPS basis is pinned to Slater (spin-1/2) or ED (spin-1: 8e-12, spin-2: 4e-12) |
| S2 | **pass** | Streamed P_L^D X P_R^D Ω and the mirrored adjoint match explicit contractions up to a rounding floor of eps·‖X‖·‖Ω‖ (bond 4: ‖G_DD‖/‖X‖ = 7e-8). No dense projectors. Empty reachable sectors are found. `dd_width = l` streams between l and l + #sectors columns, because `_alloc_columns` rounds each sector up |
| S3 | **pass** | 480 seeded runs (80 per bond, 2 bonds × 3 spins). 479 report `met` and 1 reports a limit (2s = 1, bond 6). There are **0 false accepts** (exact DD residual > share). Holdout cost is counted: mean streamed columns 17-27 at spin-1/2, 12-42 at spin-1, 59-72 at spin-2 |
| S4 | **pass** | The time-weighted defect is consistent with an independent ε_K. On the delayed-frontier fixture the need for H1² is detected. The fixed depths m = 1, 2, 4, 8 are monotone (fixed via `fixed_depth`; the earlier fixture was vacuous) |
| S5 | **pass** | Recycled (`:images`, `:frame`) and fresh runs reach the same response accuracy (errors < 1e-3, within 10 % of fresh). Images are refreshed and charge orthogonality is kept. Cost: `:frame` equals fresh (232 / 379 / 478 H1 at 2s = 1 / 2 / 4); `:images` costs 402 / 1652 / 2032 |
| S6 | **pass** | The recurrence is genuine: rounds 2-3 are enriching frontier sweeps on the advanced state (source), not repeats. The step-1 local error has slope exactly p + 1 for p = 1..4 (xfs: order 3 gives 1.02e-5, 6.4e-7, 4.0e-8, 2.5e-9). The global error from the domain wall has slope 4.0, not 3, because it is one first-step error out of the rank-1 product state; steps 2+ add < 1e-15. Started from the t = 0.5 reference, the global error is 1.5-3.8e-11 for every dt, so the h³ term is below the tolerance floor. Local errors from the references (xs6) sit at the input floor of 1e-12 |
| S7 | **pass** | Tightening `tol_basis` (1e-3 → 1e-9) lowers the error 1.4e-8 → 3.0e-11, reaching a floor at the loosest settings. `tol_trunc` 1e-6 → 1e-12 gives 2.7e-7 → 1.0e-11 and rank 24 → 69. No caps are hit. Every step reports `tolerance_met` |
| S8 | **pass, with caveats** | Paired arms run in one job on the same serialised snapshots, seeds, bonds and hardware class (`x86-64-v3&intel`). Every CSV row carries case, seed, side, param and status. H1, probe and column counts come from the solver ledgers. Figs. 3 and 4 come from dedicated timing jobs (xr3, xf4; `timing_job = true`, warmed, each alone). Fig. 1-2 times are within-process medians from a non-timing job |

## Open items

- Diagnosed in xr3: the 3 Fig. 4 steps with `tolerance_met = false` (`:images` recycling, dt = 0.2) are round-2 visits whose block-augmented recurrence hit the 64-step depth bound. The unmet flag is correct reporting, and the final error is unchanged. The `:images` mode is default-off and not recommended; `:frame` is identical to fresh. Changing the `:images` recurrence to a block continuation is a design change and is not made here.
- `closure_checks` is off by USER decision (10-17). The step-1 closure defect (1.1e-2 at dt = 0.2) is therefore recorded but not gated.
- The figures are spin-1/2 only. The spin-1 and spin-2 results are test-level (S1-S5).
