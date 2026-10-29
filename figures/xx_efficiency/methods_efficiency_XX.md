# Methods efficiency on the open XX domain wall

Executes `xx_domain_wall_cbe_bug_efficiency_standalone_plan.md` (branch `plan/xx-efficiency`).
Validation of S1-S8: [validation_results.md](validation_results.md).

## Settings

| Item | Value |
|---|---|
| Model | open spin-1/2 XX chain, `L = 16`, domain-wall product state, exact reference = Slater determinant (`tests/common/free_fermion.jl`) |
| Solver | `cbe_bug_step!`, `order = 3` (round 1 = half-sweeps, rounds 2-3 = one enriching frontier sweep each), `closure_checks` off (default) |
| Snapshots (Figs 1-2) | `benchmarks/xx_efficiency/snapshots.jl`: TDVP2 references at `J t = 0.5, 4`, truncated to rank 8/16; bonds 5 (quiet) and 8 (active) |
| Ablation switches (default off) | `dd_width = l` (one-shot DD sketch of width `l`), `krylov_depth = m` / `_krylov_enrich(fixed_depth = m)`, `krylov_recycle` + `recycle_mode = :images / :frame` |
| Fig 4 tolerances | `tol_basis = 1e-7`, `tol_trunc = 1e-10`, `tol_expv = 1e-12`, `maxdim = 512` (non-binding, max bond 24), `T = 2`, `dt ∈ {0.2, 0.1, 0.05, 0.025}` |
| Conventional arm | `dd_width = 16`, `krylov_depth = 16`, fresh Krylov (declared before the run) |
| Timing | 1 thread, `x86-64-v3&intel`, each run as the only timing job of its time. Fig 4 = xf4 (Slurm 17583862): median of 3 warmed repetitions of the whole run. Fig 3 = xr3 (17596414): JIT warm-up of every arm, then one paired run. Fig 1-2 times are medians over seeds and sides from xe4, which was not a timing job; they are relative costs inside one process |
| Raw data | `raw_component_costs.csv` (Figs 1-3; the Fig 3 rows come from xr3) and `raw_solver_errors.csv` (Fig 4) sit beside this file. Per-job JSON is in `runs_xe4/`, `runs_xr3/` and `runs_xf4/`. Plot with `python benchmarks/xx_efficiency/plot_figs.py RUNDIR OUTDIR` |

Error kinds are kept apart in all figures. ρ_DD is an independently audited range error of the
discarded-discarded block G_DD = P_L^D X P_R^D, built offline and never by the production path.
ε_K is the independent local response error of the Krylov basis. ε_z and ‖ψ − ψ_ex‖ are exact
physical errors against the Slater solution.

## Fig. 1 — adaptive DD-RSVD and DD selection (`fig1_rsvd_dd.pdf`)

**Hypothesis.** Adaptive sketching spends probes only where the bond needs them. DD selection
spends new bond columns on directions that matter to G_DD.

**Comparison.** Active bond (J t = 4, rank 16, bond 8), 3 seeds × 2 sides, same DD + QR kernel.
- A: the production adaptive controller (holdout of 4) against one-shot ℓ = k + p, p ∈ {4, 8, 16}.
- B: DD against one-sided P_L^D X at equal Gaussian width.

**Achieved error, Panel A.** `tol_basis = 1e-7` is the only target that binds on this bond. The
1e-3 and 1e-5 targets lie above ρ_DD = 1 (see `figS1a`).

| arm (tol 1e-7) | streamed columns | time [ms] | bytes | pass |
|---|---|---|---|---|
| adaptive (production) | 32 | 10.4 | 3.36 MB | 6/6 |
| adaptive (w0 = 4, blocks of 4) | 30 | 10.0 | 3.22 MB | 6/6 |
| fixed k + 4 | 8 | 2.5 | 0.78 MB | 0/6 (ρ 1.5e-2) |
| fixed k + 8 | 14 | 2.6 | 0.85 MB | 2/6 (ρ 1.9e-3) |
| fixed k + 16 | 16 | 2.6 | 0.87 MB | 6/6 |

**Saved work.** None at matched range error. Fixed k + 16 meets every target with half the
columns and a quarter of the time. On the quiet bond (bond 5), every arm reaches ρ ≈ 1e-13, and
adaptive streams 19 columns against 7.5-8. The adaptive controller's value here is
robustness: it never failed, while k + 4 and k + 8 did.

**Panel B.** At equal width, DD reaches a smaller ρ_DD at fewer selected columns: width 8 gives
1.5e-2 (DD) against 1.9e-1 (one-sided), and width 16 gives 2e-13 against 1.5e-3. The two
coincide on the rank-8 bond 8 at full width.

**Where it fails.** The claim "adaptive DD-RSVD saves MPO work at matched range error" is
**not demonstrated** on these L = 16 bonds. The discarded space is only 8-32 columns, so
p ≥ 8 is already nearly exhaustive. The minimum adaptive cost (initial block + holdout + one
growth) is above the cost of a sufficient one-shot sketch. Supplementary: `figS1a` (both bonds × three tolerances), `figS1b` (three bonds).

## Fig. 2 — time-weighted Krylov stop (`fig2_krylov_stopping.pdf`)

**Hypothesis.** The time-weighted stop avoids unnecessary H1 applications. Fixed depth cannot
detect when it is insufficient.

**Comparison.** A frozen CBE basis on each snapshot × side × h ∈ {0.05, 0.1, 0.2} × tol_basis ∈ {1e-3, 1e-5, 1e-7}
(54 cases), with the same QR and seed. There is no recycling in any arm.

| arm | median H1 applications | median time [ms] | pass (ε_K ≤ allowance) |
|---|---|---|---|
| time-weighted stop | 4 / 5 / 6 (h = 0.05 / 0.1 / 0.2) | 4.9 / 6.0 / 7.0 | 51/54 |
| fixed m = 8 | 8 | 12 | 54/54 |
| fixed m = 16 | 16 | 27.5 | 54/54 |
| fixed m = 32 | 32 | 76 | 54/54 |
| CBE only (no Krylov) | 0 | — | 22/54 |

**Saved work.** 1.3-2× fewer H1 applications than m = 8, and about 2× less time. The fixed depths pass with
ε_K many orders below the allowance, so they are over-solved.

**Where it fails.** In 3 of 54 cases ε_K exceeds the allowance. All three are on the active bond
(rank 16) at tol_basis = 1e-3, after 3-4 applications: 1.44× and 1.60× at h = 0.2, and 1.00× at h = 0.05. The
stop's weight estimate is optimistic by less than a factor of 2 there. The CBE-only row shows that the Krylov stage is needed.

## Fig. 3 — Krylov recycling (`fig3_krylov_recycling.pdf`)

**Hypothesis.** Reusing earlier Krylov directions saves H1 applications in rounds 2-3.

**Comparison.** Three order-3 steps from the J t = 4 rank-8 and rank-16 snapshots
(h = 0.1, tol_basis = 1e-5). The arms:
- F: fresh Krylov.
- R: the H-image cache (`recycle_mode = :images`; gauge-transported vectors, every image refreshed with the new H1).
- Rf: `recycle_mode = :frame`, added at the USER's request. The transported vectors only seed the frame, and a fresh Krylov runs with no image refresh.

The MPS frames are identical in all three arms. Timing comes from the timing job xr3.

**Achieved error (accuracy mark in Panel A).** An arm passes when every round met its
tolerance and its exact state error after step 3 is within 10 % of F's. All three arms pass on both
snapshots:
- rank 16: F 5.34e-7, R 4.96e-7, Rf 5.34e-7
- rank 8: F 5.43e-7, R 5.68e-7, Rf 5.43e-7

**Saved work.** None. Round 1 costs 72 H1 per step in every arm. Rounds 2-3 do work only in step 1:
- rank 16: R 18 vs F 10
- rank 8: R 41 vs F 20

Over three steps, R costs 4-9 % more H1 and 8-17 % more local-enrichment time (rank 16: 0.89 s vs
0.82 s, rank 8: 0.96 s vs 0.82 s). Refreshing the transported images costs more than rediscovering
the directions. Rf seeds 0 new directions and does exactly F's H1 work. Its time differs from F's by
±10 %, which is single-run noise. The transported directions already lie in the carried frame, and S5
shows the same at 2s = 1, 2, 4 (n_h1 232 / 379 / 478 for both F and Rf, against 402 / 1652 / 2032 for R).

**Where it fails.** Gauge changes and the changed local operator erase the value of old images. R has a
further defect, measured in job xr3 (`diag_recycle_unmet.jl`). It refreshes a block of 8-9 transported
vectors and then extends along one defect direction per step, so the block's residuals shrink by one
direction at a time. On some round-2 visits the defect never falls below the allowance before the
64-step depth bound (`n_new = 64`, `capped`); this is the source of the Fig. 4 unmet steps. Following the
plan, the recommendation is **recycling off (current default)**. Rf is harmless but brings no benefit.

## Fig. 4 — complete solver (`fig4_total_solver.pdf`)

**Hypothesis.** The local wins survive environments, root Lanczos and SVDs end to end.

**Comparison.** L = 16, domain wall, J T = 2, order 3, three arms, dt varied.

| dt | arm | wall [s] | ε_z | ‖ψ − ψ_ex‖ | H1 apps | unmet steps |
|---|---|---|---|---|---|---|
| 0.2 | full (recycling, `:images`) | 10.80 | 4.0e-7 | 1.02e-5 | 1299 | 3 |
| 0.2 | no recycling | **4.31** | 4.0e-7 | 1.02e-5 | 988 | 0 |
| 0.2 | conventional 16:16 | 6.62 | 4.0e-7 | 1.02e-5 | 1760 | 0 |
| 0.1 | full | 8.23 | 2.7e-8 | 6.38e-7 | 1490 | 0 |
| 0.1 | no recycling | **7.47** | 2.7e-8 | 6.38e-7 | 1432 | 0 |
| 0.1 | conventional | 13.26 | 2.7e-8 | 6.37e-7 | 3400 | 0 |
| 0.05 | full | 14.98 | 1.7e-9 | 4.6e-8 | 2447 | 0 |
| 0.05 | no recycling | **14.19** | 1.7e-9 | 4.5e-8 | 2441 | 0 |
| 0.05 | conventional | 26.86 | 1.7e-9 | 4.0e-8 | 6806 | 0 |
| 0.025 | full | 28.48 | 4.9e-11 | 3.5e-8 | 4491 | 0 |
| 0.025 | no recycling | 27.41 | 4.9e-11 | 3.8e-8 | 4487 | 0 |
| 0.025 | conventional | 52.82 | 1.1e-10 | 2.6e-9 | 13690 | 0 |

**Saved work.** At matched physical error (dt = 0.2, 0.1, 0.05: equal ε_z and state error),
adaptive without recycling is **1.5-1.9× faster** than the conventional controls, with 1.8-2.8×
fewer H1 applications. The conventional arm fixes both the DD width and the Krylov depth, so Fig. 4
cannot attribute the gain to one of them. Figs. 1-2 suggest it comes from the Krylov depth. The recycling
cache never pays off: it is 2.5× slower at dt = 0.2 and 4-10 % slower elsewhere.

**Order (Panel B).** The state error falls as h⁴ (1.02e-5 → 6.4e-7 → 4.0e-8), which is steeper than the h³
reference. The cause was diagnosed (job xfs): all of the global error is made in step 1, out of the
rank-1 product state. Steps 2+ add < 1e-15, and the first-step error has slope exactly p + 1 for
p = 1..4. A one-time local error O(h⁴) therefore dominates the global O(h³) term. Started from the t = 0.5
reference, the global error stays at 1.5-3.8e-11 for every dt.

**Where it fails.**
- At dt = 0.025 both adaptive arms sit on the tol_basis = 1e-7 floor (state error 3.5-3.8e-8). The conventional arm's fixed width and depth over-solve to 2.6e-9, so there is no matched-error pair at that dt. The fair comparison is conventional at dt = 0.05 (4.0e-8, 26.9 s) against no-recycling at dt = 0.05 (4.5e-8, 14.2 s).
- ε_z keeps falling (4.9e-11) while the state error plateaus. ε_z is the less sensitive metric.
- The full arm reports 3 steps with `tolerance_met = false` at dt = 0.2 (steps 7, 8 and 10), while fresh and `:frame` report none. The cause was diagnosed in xr3. In each of those steps one round-2 visit (bond 7 or 9) ran the `:images` recurrence to its 64-step depth bound, as described under Fig. 3. The solver reports these visits as unmet instead of hiding them, and the final error is unchanged. The extra applications (each capped visit about 74 H1) are most of the 2.5× slowdown at that dt.

## Recommendation

Use adaptive DD + QR with the time-weighted Krylov stop and **no recycling** (the current defaults).

- The end-to-end speedup over fixed controls is genuine at matched error, and order ≥ 3 holds.
- Adaptive DD sketching is not cheaper than a sufficient fixed oversampling on these small
  bonds. Keep it for robustness, and do not claim it saves MPO work here.
- Recycling, in either mode, does not save work on the XX domain wall.
