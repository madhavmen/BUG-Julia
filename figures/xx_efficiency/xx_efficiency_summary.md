# XX domain-wall efficiency study: implementation, results, settings

This summarises the work done under `xx_domain_wall_cbe_bug_efficiency_standalone_plan.md`.
- Per-figure captions: [methods_efficiency_XX.md](methods_efficiency_XX.md).
- Test-by-test evidence: [validation_results.md](validation_results.md).
- Raw data: `raw_component_costs.csv` (Figs 1-3) and `raw_solver_errors.csv` (Fig 4) in this folder, plus per-job JSON in `runs_xe4/`, `runs_xr3/` and `runs_xf4/`.

## 1. What was implemented

**Ablations.** All are switched off by default, so production behaviour is unchanged.

| Feature | Where | What it does |
|---|---|---|
| H-image recycle cache | `src/RSVDCBEBondUpdate/bug/krylov_recycle.jl`, `cbe_bug_step!(krylov_recycle = true, recycle_mode = :images)` | Carries a round's Krylov vectors into the next order-3 augmentation round. It gauge-transports them, re-orthogonalises them, and re-applies the new local H1 to every transported vector. The Krylov recurrence then continues from that augmented block. |
| Frame-only recycling | `recycle_mode = :frame` | The transported vectors only seed the next round's frame. A fresh Krylov runs on top, with no image refresh. |
| Fixed Krylov depth | `_krylov_enrich(fixed_depth = m)`, `cbe_bug_step!(krylov_depth = m)` | Takes exactly `m` Krylov steps with no weight stop. Before this, `_kalloc(0.0)` stopped every "fixed depth" after one application. |
| Fixed DD width | `dd_cbe_expand(dd_width = l)`, `cbe_bug_step!(dd_width = l)` | Draws one one-shot DD sketch through the same DD+QR kernel. It replaces the adaptive holdout controller. |
| Recycle verdict log | `recycle_report` entries | Records each visit's `met`, `capped`, `eta_c` and `eta_d`, which made it possible to diagnose the unmet steps. |

**Oracles and fixtures** (in `tests/common/`):
- Slater-determinant amplitudes (`xx_slater_amplitudes`), with signs checked against ED.
- Sector-ED references for spin-1 and spin-2.
- Snapshots of the reference state compressed per U(1) sector.
- Offline audits: ρ_DD (the DD range error, built from an explicit G_DD that production never forms), ε_K (the local Krylov response error) and the DK/DD content of each bond.

**Tests and drivers:**
- Tests S1-S5 are in `tests/sweeps/test_xx_efficiency.jl`. They run at spin-1/2, spin-1 and spin-2.
- The drivers are in `benchmarks/xx_efficiency/`: `snapshots`, `fig1_dd`, `fig2_krylov`, `fig3_recycle`, `fig4_endtoend`, `s6_s7_order`, `s6_first_step`, `diag_recycle_unmet`, and the plotting script `plot_figs.py`.

## 2. Results

| Plan question | Result | Claim allowed? |
|---|---|---|
| DD projection before QR | **Supported** on the active bond. At equal width 8, DD reaches ρ_DD 1.5e-2 against 1.9e-1 for one-sided sampling. At width 16, DD reaches 2e-13 against 1.5e-3. On quiet bonds the two give the same result, and DD only adds overhead. | Yes, on active bonds |
| Adaptive RSVD vs fixed oversampling | **Not shown.** At tol 1e-7, the adaptive controller streams 32 columns (10.4 ms). Fixed k+16 needs 16 (2.6 ms) and also passes 6/6. Fixed k+4 and k+8 fail (0/6 and 2/6). Adaptive never fails, so its value here is robustness, not saved work. On L = 16 the discarded space is only 8-32 columns. | No |
| Time-weighted Krylov stop | **Supported.** The stop takes 4-6 H1 applications against fixed m = 8/16/32. That is 1.3-2× fewer applications and about 2× less time at the same ε_K. It passes 51/54 cases. The 3 failures are all on the active rank-16 bond at tol 1e-3, overshooting by 1.44×, 1.60× and 1.00×. | Yes, with those cases stated |
| Krylov recycling | **Negative.** See section 3. | No |
| End-to-end speedup (Fig 4) | **Supported without recycling.** At matched exact error (dt = 0.2, 0.1, 0.05), it is 1.5-1.9× faster than conventional 16:16 controls, with 1.8-2.8× fewer H1 applications. The full arm with recycling is 2.5× slower at dt = 0.2 and 4-10% slower elsewhere. | Yes, for the no-recycling setup |
| Third order (S6) | **Local order shown; global h³ not observed.** The first-step error falls with slope exactly p+1 for p = 1..4. The global error from the domain wall falls as h⁴, not h³, because all of it is made in step 1, out of the product state; steps 2+ add < 1e-15. Started from the t = 0.5 reference, the global error is 1.5-3.8e-11 at every dt, below the tolerance floor. So the h³ convergence window the plan asks for was never seen. | Order ≥ 3 behaviour; no measured global h³ slope |

**Tests:**
- Final run xe6: S1-S5 pass 1765/1765 across all three spins.
- Earlier jobs: the regression suites `krylov_closure`, `adaptive_krylov`, `dd_cbe` and `dd_cbe_controller` pass.
- S7: tightening a tolerance lowers the error until a floor, and no caps are hit.

**Recommendation:** adaptive DD+QR, the time-weighted Krylov stop, and **no recycling**. These are the current defaults, so nothing changes.

## 3. Observation: recycled vectors widen the basis instead of saving work

Recycling is meant to avoid rediscovering Hamiltonian directions in rounds 2-3. In practice, the transported vectors make the round-2 frame wider, and every extra column has to be refreshed and orthogonalised against. Measured in xr3 (Fig 3, J t = 4 snapshots, h = 0.1, tol_basis 1e-5):

| Snapshot | Arm | Round-2 columns carried in (step 1) | Round-2 H1 | Step-1 H1 | Final bond dims, bonds 7-9 (step 3) |
|---|---|---|---|---|---|
| rank 16 | fresh (F) | 0 | 10 | 82 | 35, 42, 33 |
| rank 16 | `:images` (R) | 12 | 18 | 90 | 35, 42, 35 |
| rank 8 | fresh (F) | 0 | 20 | 92 | 31, 31, 30 |
| rank 8 | `:images` (R) | 24 | 41 | 113 | 32, 32, 31 |

- **More work for the same accuracy.** R uses 4-9% more H1 and 8-17% more local-enrichment time over three steps. The final exact errors are the same within 10% (rank 16: 5.34e-7 for F, 4.96e-7 for R; rank 8: 5.43e-7 for F, 5.68e-7 for R).
- **Slow convergence on wide blocks.** After refreshing a block of 8-9 transported vectors, R extends along one defect direction per step. On some round-2 visits it hits the 64-step depth bound with weight left. This causes the 3 steps with `tolerance_met = false` in Fig 4 at dt = 0.2, which cost about 74 H1 each. Those visits are most of the 2.5× slowdown at that dt, and R needs 1299 H1 against 988 for no recycling.
- **Frame-only recycling brings nothing.** `:frame` seeds 0 new directions and does exactly fresh Krylov's work in every run. The transported directions already lie in the carried MPS frames.
- **Larger local dimension is worse.** In S5 (order 3 against the exact reference), `:images` costs 402 / 1652 / 2032 H1 at spin-1/2 / spin-1 / spin-2, against 232 / 379 / 478 for fresh and `:frame`. The penalty grows from 1.7× at spin-1/2 to about 4.3× at spin-1 and spin-2, because the transported block gets wider with d.

**Conclusion.** On the XX domain wall, at every spin tested, recycled Krylov vectors are not efficient. The changed gauge and local operator erase most of their value, and the vectors that survive widen the frame and the refresh cost. Recycling stays off.

## 4. Tolerances and settings per simulation

**Common settings:**
- Model: open XX chain, J = 1, domain-wall product state. L = 16 for the figures.
- Solver: `cbe_bug_step!` with `order = 3` (round 1 = half-sweeps; rounds 2-3 = one enriching frontier sweep each) and `closure_checks` off.
- Seeds 11, 22, 33.
- Hardware and threads: 1 BLAS thread on `x86-64-v3&intel`.
- Cluster output: `/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-29-<job>/`.

| Simulation (job) | Purpose | Settings |
|---|---|---|
| Snapshots (xe3) | Fixture M | Ramped order-4 TDVP2 reference, dt 0.025, to J t = 0.5 and 4. Compressed to rank 8 / 16 / 32. Bonds 8 (active) and 5 (quiet). Labelled trivial if ‖G_DD‖ ≤ 1e-8 ‖X‖. |
| Fig 1 (xe4) | Adaptive DD vs fixed oversampling; DD vs one-sided | h = 0.1. tol_basis ∈ {1e-3, 1e-5, 1e-7}. Adaptive = production controller, plus a w0 = 4 variant growing in blocks of 4. Fixed = one-shot ℓ = k + p with p ∈ {4, 8, 16}. Panel B widths 2, 4, 8, 16. 3 seeds × 2 sides. Not a timing job (relative times only). |
| Fig 2 (xe4) | Krylov stop vs fixed depth | Frozen CBE basis. h ∈ {0.05, 0.1, 0.2} × tol_basis ∈ {1e-3, 1e-5, 1e-7}, 54 cases. Fixed depths m = 8, 16, 32. Seed 11. Not a timing job. |
| Fig 3 (xr3, 17596414) | Recycling F / R / Rf | J t = 4, rank 8 and 16 snapshots. 3 order-3 steps, h = 0.1. tol_basis 1e-5, tol_trunc 1e-10, tol_expv 1e-12, maxdim 256. Seed 11. Timing job, with every arm JIT-warmed first. |
| Fig 4 (xf4, 17583862) | End-to-end, three arms | L = 16, J T = 2, dt ∈ {0.2, 0.1, 0.05, 0.025}. tol_basis 1e-7, tol_trunc 1e-10, tol_expv 1e-12, maxdim 512 (non-binding; max bond 24). Conventional arm: dd_width = 16, krylov_depth = 16. Timing job: median of 3 warmed repetitions. |
| Recycle diagnosis (xr3) | Fig 4's unmet steps | dt = 0.2, J T = 2. tol_basis 1e-7, tol_trunc 1e-10, tol_expv 1e-12, maxdim 512. Arms: fresh, `:images`, `:frame`. |
| S6 / S7 (xs6) | Order and tolerance response | Tight settings: tol_basis 1e-10, tol_trunc 1e-12, tol_expv 1e-13, maxdim 512. S6: one-step errors from the references, plus a global run from the domain wall to J T = 2 at dt ∈ {0.2, 0.1, 0.05, 0.025}. S7 at h = 0.1 from t = 4: tol_basis swept 1e-3 → 1e-9, and tol_trunc swept 1e-6 → 1e-12. |
| First-step diagnosis (xfs) | Global h⁴ explained | The same tight settings. Per-step errors and ledgers from the product state, a first-step order sweep p = 1..4, and a global run from the t = 0.5 reference. |
| Tests S1-S5 (xe6, 17583863) | Validation | Spin and chain length (2s, L): (1, 12), (2, 10), (4, 8). S1 also at L = 4, 6, 8, 16. S3: 80 seeds per bond, 2 bonds per spin. S5 (order 3 against the exact reference): tol_basis 1e-8, tol_trunc 1e-10, tol_expv 1e-12, maxdim 64 at spin-1/2 and 128 otherwise. |

## 5. Open items (USER decisions)

- Redesigning `:images` so its recurrence grows a whole block of directions per step. This would fix the slow convergence but not the widening, so recycling stays off.
- Running Fig 1 at L = 24 or 32, where the discarded space is large enough for adaptive sketching to pay off.
- An S6 run with a resolvable global h³ window: tighter tolerances or a longer time T.
- Spin-1 and spin-2 figures. These spins are currently covered at test level only.
