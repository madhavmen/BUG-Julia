# Agent Transfer Notes (Paused)

Date: 2026-10-13  
Scope: Local BUG update in `RSVDCBEBondUpdate` (adaptive RSVD/CBE controller), cluster-only validation on LMU HPC.

## 1) What was requested

- Continue from Stage 1 and **wire in Stage 2 RSVD logic**:
  - Test A (retained-rank tail test),
  - Test B (holdout sampling residual test),
  - keep compatibility with parallel half-sweeps.
- Keep cluster compliance workflow (`cluster_compliance.md`) before submissions.
- Run code/tests on cluster (no local Julia execution).

## 2) Completed work in this pause window

## 2.1 Read/verification work

- Confirmed `CBEBugOptions` path is in `modes/ground_state.jl`, not forcing `cbe_bug_step!` into legacy mode.
- Monitored previous Stage 1 jobs and captured outcomes.
- Re-checked cluster compliance checklist prior to new submissions.

## 2.2 Source changes made

### A) `src/RSVDCBEBondUpdate/cbe_core.jl`

Added/updated adaptive RSVD internals:

1. Added shared singular-weight helper:
   - `_sq_weight(list)` (degeneracy-weighted `sum(sigma^2 * deg)`).

2. Kept `_trunc_weight(res)` but refactored to use `_sq_weight`.

3. Added **Test A helper**:
   - `_tail_rank(list, scale, allowance, kmax; base=0.0) -> (k, tail_indicator)`
   - Includes weight already cut by SVD (`base`) in the tail budget check.

4. Added **Test B helper**:
   - `_sample_residual(Q, project, sketch, draw, nh, wleg, side)`
   - Uses independent holdout probes and returns residual estimate `||miss||/sqrt(nh)`.

5. Added certified widening wrapper:
   - `_certified_range(...)`
   - Uses `_range_find` first, then holdout residual loop against `alloc.sample`.
   - Records ledger status (`max_sample`, `n_holdout`, tolerance/workcap flags).

6. Wired `_certified_range` into `cbe_expand`:
   - Replaced direct `_range_find` calls for left/right preselection.

7. Wired **Test A** into joint final ranking block:
   - After SVD on `UMU`, applies `_tail_rank(...)` with `alloc.tail`.
   - Tightens retained joint rank where possible.
   - Propagates reduced rank to per-side caps (`dex_l`, `dex_r`) so fallback path obeys same tail decision.
   - Updates ledger (`max_tail`, `tolerance_met`).

8. Wired `workcap` into range-width caps when adaptive path is active:
   - clamps widening ceilings `wcap_l/wcap_r` by `workcap`.

9. Added ledger accounting:
   - `ledger.n_selected` updated from admitted left/right directions after re-orthonormalization.

### B) `tests/sweeps/test_adaptive_rsvd.jl` (new)

Added Stage 2 tests:

- **UT03**: sector allocation invariants for `_alloc_columns`.
- **UT04**: retained-rank tail logic (`_tail_rank`) incl. degeneracy and base-cut effects.
- **UT05**: independent holdout rationale/counterexample and estimator sanity.
- **UT06**: end-to-end adaptive step checks on XX/U(1), with `parallel=true` and monotonic behavior expectations.

### C) `tests/sweeps/runtests.jl`

- Included `test_adaptive_rsvd.jl` after `test_adaptive_api.jl`.

## 3) Cluster executions and status

## 3.1 Completed Stage 1 targeted job

- Job: `17169317`
- Result: **COMPLETED**
- Output summary:
  - `LOAD OK`
  - `UT01`: 56/56 pass
  - `UT02`: 7/7 pass
- Efficiency:
  - CPU util: 39.3%
  - Mem util: max ~6.9G (requested 24G)

## 3.2 Full sweeps regression run

- Job: `17169504`
- Observed summary:
  - `Pass 707 / Fail 2 / Error 2 / Total 711`
  - `suite rc=1`
- Failure/error set appears to match known pre-existing reds (not newly introduced by Stage 1 migration).

## 3.3 Stage 2 targeted UT run (in progress at pause)

- Synced updated files to cluster:
  - `src/RSVDCBEBondUpdate/cbe_core.jl`
  - `tests/sweeps/test_adaptive_rsvd.jl`
  - `tests/sweeps/runtests.jl`
- Submitted job: `17188205`
  - Script runs:
    1) `tests/sweeps/test_adaptive_api.jl`
    2) `tests/sweeps/test_adaptive_rsvd.jl`
  - Threads: `julia -t 2`
  - Memory: 16G
  - Output directory:
    - `/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-13-stage2/`
  - **Status at pause: still RUNNING** on node `th-cl-dell07`, ~10.5 min elapsed.
    Only the startup banner had been written; no UT results yet.

    This is expected rather than a hang: the job landed on a node that had not
    yet built `BUGJulia`, and a cold precompile of this package has been measured
    at **~13-14 min** (768 s on job `17169317`, 804 s on job `17169504`). Julia
    also block-buffers stdout when it is redirected to a file, so the UT summaries
    appear only once each `julia` invocation exits. An empty `.out` beyond the
    banner therefore carries no information until the job leaves the queue.

## 4) Important caveats / follow-up checks

1. **Collect final result of job `17188205`**:
   - `ut06_17188205.out/.err`
   - confirm UT03..UT06 pass/fail details.

2. If any failures:
   - likely first suspects are new helper wiring in `cbe_core.jl` and assertions in UT06.

3. After targeted pass, rerun broader suite as needed and diff against accepted baseline failure set.

4. Keep enforcing:
   - `parallel=true` coverage in adaptive path tests,
   - cluster-only execution,
   - compliance checklist before every `sbatch`.

## 4b) 2026-10-14: job 17188205 result, plan audit, and Stage 2 fixes

Job `17188205` COMPLETED: UT01 56/56, UT02 7/7, UT03 35/35, UT04 18/18, UT05 4/4, UT06 6/6
(seffx: 38% CPU of 4, 7.3G of 16G, cold precompile 674 s).

The green run did NOT mean Stage 2 matched plan §3/§7. Audit findings, all fixed:

1. **Ledger discarded.** `cbe_bug_step!` built an `AdaptiveLedger` and never returned it, so
   `tolerance_met = false` was invisible. Now `CBEBugSweepInfo.ledger` (`nothing` on legacy);
   the midpoint step merges its predictor's ledger.
2. **Parallel race.** Both half-sweeps mutated ONE ledger and ONE `MersenneTwister`
   concurrently. Now per-half ledgers (merged after the join) and, when `parallel`, a right-sweep
   RNG seeded from `rng` (serial path draws the identical sequence as before).
3. **Test B estimator biased low.** The holdout is sector-graded but was normalised by the total
   width, under-reporting a missed small sector by `sqrt(w/n_q)`. Now normalised per sector;
   `_sample_residual` returns `(est, per_sector, cols)`.
4. **No sector-targeted widening** (plan §3 "add probes to the worst sectors"). Now
   `_target_sectors!` + `boost` kwarg on `sector_graded_sketch`; `_certified_range` widens only
   the named sectors and stops (reported) on exhaustion or the work cap. `n_targeted` in ledger.
5. `max_sample` recorded the final accepted indicator, not failed intermediate passes.
6. Tests renumbered to the plan: UT03 coverage on a real frame (+ boost), UT04 Test B on the
   production sketch (self-certification counterexample, per-sector unbiasedness vs exact,
   targeting), UT05 Test A incl. plan spectrum (10,9,3,0.1), UT06 misleading tail (A passes,
   B fails, controller repairs; exhaustion and work cap reported), plus end-to-end ledger /
   parallel-reproducibility / midpoint checks. Telum `kept_list` is globally sorted
   (`deps/Telum/src/svd.jl:1741`), so `_tail_rank` is valid.

Known weakness kept (now covered by Test B): `_range_find` stops on a GLOBAL count
`n < wd`, so surplus columns in one sector can mask a miss in another.

Validation: `17188345` OOM-killed in precompile at `--mem=9G` (`JULIA_CPU_TARGET` multiversion
raises the peak; cancelled). `17188740` (16G, `.../2026-10-14-stage2b/`): COMPLETED 1h19, MaxRSS
10.8G, CPU 69% of 2. Sweeps 824/3/2: reds = the 4 baseline ones (`test_cbe_bug_exact_refs.jl:310`,
`test_physics.jl:82,116,153`, unchanged from 17169504) + UT04 "a sketch cannot certify its own
range" -- a TEST precondition (each sector's range has rank <= 1 at that bond, so a narrow sketch
missed nothing), fixed. UT05 22/22, UT06 16/16, end-to-end 13/13, `share_ht and parallel` 18/18.

## 4c) 2026-10-14: plan audit ledger and the Stage 1/2 open items

USER: audit code vs plan after EVERY stage -> `ADAPTIVE_PLAN_AUDIT.md`. Open in-scope rows fixed:
1.2 `maxdim` default 128 on the adaptive path (legacy 200); 1.13 UT01 through the public entry;
1.14/1.15 adaptive preselection uses the numerical floor `_rank_cut` (stol = 0) and UT02 tests the
production `_preselect_left`; 2.5/2.14 ledger `max_sample_l/_r` + `sample_log` (per-sector, with
uncertainty bound); 2.10 Test A on one-sided expansions (`_one_sided_tail`). Test files wrapped
in one outer testset; shared fixture `tests/common/adaptive_fixtures.jl`.
Validation job `17190027` (`.../2026-10-14-stage2c/`, 13G): COMPLETED 1h45. Stage 1 70/70,
Stage 2 225/225, full sweeps 939/2/2 with ONLY the 4 baseline reds. MaxRSS 11.5G (88% of 13G),
so request 14G next. **Stages 1-2 closed; next is Stage 3 (K/L Krylov Tests C/D, UT07-UT10),
audited into `ADAPTIVE_PLAN_AUDIT.md` when done.**

## 4d) 2026-10-14: Stage 3 -- residual-driven K/L Krylov (plan §4, Tests C/D)

`cbe_bug.jl`: `_krylov_enrich` (+ `_kry_weight`, `_extend_frame`, `_record_krylov!`), called from
BOTH half-sweeps on the adaptive path (`if adaptive ... elseif krylov_basis > 0` legacy). The frame
starts as the CBE `U_ex`/`V_ex` (old + CBE directions verbatim); each Arnoldi direction of `H1`
on the seed is scored `weight_k * ||(I-QQ')v_k||` (weight = Saad's `|tau beta phi_1|`, k=1 is
Test C, k>=2 Test D) and appended when above `alloc.krylov`; it stops on propagator weight <=
allowance or exact closure. No split SVD on the adaptive K/L frame. `krylov_tol`, `grow_iters`
and `krylov_grow` are now LEGACY_KNOBS (refused with a tolerance). Ledger gains
`max_krylov_c/_d`, `n_c_pass`, `n_d_after_c`. Tests: `tests/sweeps/test_adaptive_krylov.jl`
(UT07-UT10 + an end-to-end XX L=12 free-fermion check). Audit: `ADAPTIVE_PLAN_AUDIT.md` Stage 3.
Validation job `17192249` (`.../2026-10-14-stage3/`, 14G).

## 4e) 2026-10-14: Stage 4 -- rank cap and one budgeted compression (plan §5, §9.4), NOT YET RUN

Written locally while `17192249` (Stage 3) runs; NOT synced to `~/BUG-Julia-rsvdadapt` (each stage
of that job reloads the source). Changes:
- `truncate_recursive!(...; weight_tol)`: `tol_trunc` is now a budget on the cumulative relative
  error -- per-bond share `tol/(sqrt(n_L)+sqrt(n_R))`, minimal rank via `_tail_rank`, estimate
  `(sqrt(Σ_L w)+sqrt(Σ_R w))/||psi||` (Pythagoras within a subtree, triangle across). Before, the
  adaptive path passed `tol_trunc` as Telum's relative singular-value cutoff (audit 4.5).
  Returns `(psi, worst, cumulative, capped)`; legacy `cutoff` mode unchanged.
- Ledger: `maxdim_reached`, `proj_err`, `r_old/r_aug/r_final`, `unmet::Vector{Symbol}`;
  `_unmet!` is the only place `tolerance_met` is cleared; exported `budget_status`
  (`:met`/`:budget_unmet`/`:legacy`) and `budget_diagnostic`.
- Work ceiling now also caps RETAINED CBE directions (`r + n <= workcap`) and the Krylov frame
  width (`_extend_frame(...; room)`; it could overshoot by one block).
- Adaptive root split pinned lossless; midpoint projection error charged to `tol_basis`
  (`_PROJ_FLOOR = 1e-7`, the norm-difference resolution).
- Tests: `tests/sweeps/test_adaptive_compression.jl` (UT11-UT14 + midpoint T=1 regression),
  included after `test_adaptive_krylov.jl`. Audit: `ADAPTIVE_PLAN_AUDIT.md` Stage 4.

## 4f) 2026-10-14: Stages 4-5 run; Stage 5 (`tol_expv`); Stage 6 campaign submitted

- Stage 3 CLOSED (17192249): UT07-10 68/68, sweeps 1007/2/2 = baseline reds only.
- Stage 5: `expv(...; report)` -> `(iterations, substeps, err)`; adaptive root uses
  `conv_tol = tol_expv`, `substeps = 0`, `maxiter = _ROOT_MAXITER = 30`; `maxiter`, `tol`,
  `root_conv_tol`, `root_substeps` are LEGACY_KNOBS. Ledger `expv_err`, `expv_iters`, `:expv`.
  UT15 in `tests/sweeps/test_adaptive_expv.jl`. simulation_plan's `dt = 0, truncate = false`
  reconstruction check added to UT13.
- 17193333 (S4+S5, 13G next; MaxRSS 10.7G cold): UT01-10 green (UT10 now non-vacuous: CBE alone
  misses 1.8e-3/5.5e-3, Krylov closes it to 1e-16); UT11-14 65/67, UT15 24/25 -- all three reds
  were TEST preconditions (fixed, rerun 17195151); sweeps 1100/5/2 = 4 baseline + those 3.
- Stage 6: `benchmarks/adaptive_xx_l20.jl` (phases profiles/sweeps/baseline, streamed CSV,
  free-fermion graded) + `benchmarks/analyze_adaptive_xx.py` (tables, loosest-tolerance-per-
  target). L=8 smoke 180 rows OK. L=20 jobs (`.../2026-10-14-l20/`, sbatch `l20.sbatch`,
  MODE/PHASE/TMAX via --export): 17195152 profiles, 17195153 sweeps, 17195154 baseline (T=0.5),
  17195155 sweeps T=2.0.

## 4g) 2026-10-14: ALL SIX STAGES DONE; results

- UT11-15 rerun 17195151: 67/67, 25/25. Stages 4-5 closed.
- L=20 campaign complete (profiles/sweeps/baseline at T=0.5; sweeps/combos/maxdim at T=2):
  `ADAPTIVE_XX_L20_RESULTS.md` (findings + tables + recommended settings). README now documents
  the three-tolerance API, `budget_status`/`budget_diagnostic`, and links the results.
- Headlines: RSVD == exact CBE (no accuracy cost; 1.6x MORE CBE time at d*chi <= 256);
  tol_basis is the accuracy knob (err ~ tol_basis); tol_trunc can be 10-100x looser and is the
  speed knob; tol_expv 1e-2 is enough at dt 0.05; midpoint sets a 1000x lower floor.
- 17195549: full sweeps on FINAL code 1103/2/2 = 4 baseline reds only. 17195550 matched-accuracy
  (one process): adaptive beats legacy kry3 on trial width/rank/wall at matched error (audit 6.9
  DONE). Next experiment: a non-free-fermion model (Heisenberg/OAT) and a
  large-chi case where the RSVD sketch could actually beat `exact = true`.
- Committed (user chose the whole tested tree): branch `feature/adaptive-cbe-krylov`, commit 0555fb4.
  RSVD crossover (17195922): RSVD/exact CBE time 1.8 at chi 29 -> parity at chi ~250-400. Not pushed.

## 4h) 2026-10-15: robustness + quickstart

- `examples/adaptive_quickstart.jl` (17197997): QUICKSTART-OK (defaults 1.6e-6 chi 43; loose 4.4e-5
  chi 4; maxdim 4 -> 20/20 `:budget_unmet` with a readable diagnostic).
- Driver takes `[model]` (`xx` | `heis`, exact sparse Sz=0 reference in `benchmarks/exact_sparse.jl`)
  and `dt`. 17197998 Heisenberg sweeps+combos, 17197999 XX dt 0.1 sweeps (T=2): rules hold; finding
  9 + "Robustness" tables in `ADAPTIVE_XX_L20_RESULTS.md`. Gate 9 clean (MaxRSS 2.5/1.6 GB).
- Re-reading §7/§8 found two gaps, closed with row 1.5 in job 17198444 (audit R.1-R.3):
  `ledger.bond_log` (one entry per half-sweep bond visit; step fields = its max/sum; driver
  `*_bonds.csv`, analyzer "Per bond"), `neel_state(L; eltype = Float64)` etc. + Float64 tests,
  `_legacy_hint` (one warning per session). New tests 83/83; full sweeps 1186/2/2 = 4 baseline reds.
  EVERY ROW OF THE PLAN AUDIT IS NOW DONE.
- simulation_plan.md checked line by line (audit S.1-S.2): `adaptive = true` deliberately absent
  (main plan's API wins); §5 per-bond list completed -- bond entries carry `n_sectors`,
  `probes_per_sector`, `r_old`, `r_final`, `discarded` (`truncate_recursive!(...; bond_weights)`).
  Job 17199363: relevant tests 170/170. USER: run only relevant test files from now on.

## 5) Quick resume commands (cluster)

```bash
squeue --job=17188205
tail -120 /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-13-stage2/ut06_17188205.out
tail -80  /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-13-stage2/ut06_17188205.err
seffx -j 17188205
```

