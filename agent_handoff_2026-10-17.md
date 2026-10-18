# Agent handoff — CBE-BUG vs 2-site TDVP under the new fair settings

Date: 2026-10-17 23:20. Repo: BUG-Julia, branch `feature/krylov-closure-order`, HEAD `8cd253b`.
**All of this session's code changes are UNCOMMITTED** (§3). The user has not yet said to commit.

Earlier context:
- `agent_handoff_2026-10-17_morning.md`: the morning handoff, with the session log appended.
- `AGENTS_HANDOVER_5.md`: the detailed record, with figures in `handover5_figs/`.

---

## 0. Standing rules (binding)

- **Cluster.**
  - Read `../cluster_compliance.md` IN FULL before every `sbatch`, and walk its 9 gates with
    named evidence.
  - `--time=7-00:00:00`; `--mem` from a measured MaxRSS + 10–20 %; 4 CPUs with `julia -t 4`;
    exclude the cip nodes and th-cl-dell03.
  - Output under `$SCRATCH/rsvd_adaptive/<date>-<tag>/`.
  - **Gate 9 is three checks:** `seffx -j`, `grep -iE 'oom|out of memory'`, and
    `grep -E 'EXIT-[A-Z0-9x-]+=[1-9]'`. The batch script always exits 0.
- **Run nothing locally** (Julia). Python grading and plotting locally is fine.
- **Staging.**
  - `tar --force-local` of `git ls-files -co --exclude-standard` → `scp` →
    `~/bugjl/stage.sh ~/bugjl/<name>.tar <name>`. It prints `SNAP` and the content-addressed
    `PKG`.
  - A `src` change gives a new PKG, so the first job pays a ~10 min cold precompile.
- **One timing job at a time,** and keep it off nodes that run your other jobs (§4, `fair10`).
- **The ssh wait-loop trap.** A loop on `squeue | grep -q` exits on an ssh failure. Loop on
  `sacct -X -n -o State` until it shows a final state.
- **Fairness (USER 2026-10-16/17).**
  - Partners: bug2 ↔ `tdvp2_step!` (bugmid is not tdvp2's partner); same dt; JIT counted.
  - `tol_trunc` = `tol_expv` = 1e-6.
  - **Both integrators discard singular values by the identical per-cut rule (§2.2), not by an
    equal per-step budget.**
  - Grade against a reference (or exact) before calling anything faster or rank-better.
- **USER priority (2026-10-17):**
  1. comparable errors under fair testing;
  2. bond-dimension growth χ(t) of CBE-BUG ≤ or ≈ 2-site TDVP at a NON-binding cap;
  3. only then, efficiency of general-p CBE-BUG.

  The augmentation (`tol_basis`, closure, RSVD) may be made as loose and cheap as possible while
  the error stays similar.

## 1. Definitions

- **Phase 6 / references:** 10x4 square Heisenberg cylinder (40 spins), S^z quench at the centre,
  G(x,t) = ⟨Ω|S^z_x(t) S^z_c|Ω⟩. Error = max_x |G − G_ref|; G(c,0) = 1/4.
- **nb24 (this session):** 6x4 square (24 spins), exact sparse reference (dim 2,704,156). Cap
  4096 = 2^12, the full rank of the middle bond, so the cap can never bind.
- **Arms:**
  - `tdvp2e`: `tdvp2_step!`, Saad-stopped exponentials at `tol_expv`, `weight_tol = tol_trunc`.
  - `bug2`: `cbe_bug_step!(...; order = 2)`.
  - `cbe1se`: 1-site CBE-TDVP with a RELATIVE cutoff. It has no budgeted rule (open flag).

## 2. Decisions taken this session (USER)

### 2.1 An order-p step is the half-sweeps plus ONE enriching sweep per frontier

> "for a second order method, why are we doing 4 half sweeps? we should only be doing 2 ... just
> do stage 1 and stage 3. the checks were there only for the beginning to validate stuff -- keep
> it as a knob, default off."

- `cbe_bug_step!(...; closure_checks = false)` → `_krylov_closure!(...; checks)` →
  `_single_pass_closure!`. An order-2 step is now the augment half-sweeps (L ‖ R) plus one
  enriching sweep (L ‖ R) of `j = 1`.
- `closure_checks = true` restores the validation path:
  - check first: measure every frontier;
  - loop until a pass adds nothing;
  - measure `j = 0` on the final frames.
- Ledger with the default:
  - `n_passes = 1`;
  - `leak[j]` = frontier `j`'s leak before its enrichment, and `leak[0] = NaN` (not measured);
  - `leak_initial`/`rho_initial` are empty (`closure_probe` still measures them at order 1);
  - `:closure` is unmet when an enriching sweep fails Tests A–D.
- **Why:** kscreen2 17299494 (10x4, D = 512) — the closure took 46–52 s of a 78–84 s step:
  - check first: 12 s;
  - the enriching sweep: 12–16 s;
  - the re-check ‖ final j = 0: 16–18 s.

  The enriching sweep (the only closure sweep that runs Tests A/B) costs only 0.2–3.7 s more than
  a measuring sweep. Recorded in **closure plan §8** (`krylov_closure_cbe_bug_realtime_L16_plan.md`).
- This overrides handover 5's "check first stays (UT6/UT9)".

### 2.2 The same discarded singular values per cut, not per step

> "we should not discard the norm per step. we need to make sure the singular values that are
> being discarded for the time evolution step is the same, so the S-step for the BUG and the per
> bond time update for the 2sitetdvp."

- `_cut_divisor(L, root)` and `_bug_root_link(L)` live in `zero_site_core.jl`, and both
  integrators use them:
  - `truncate_recursive!` (the BUG's S-step compression);
  - `tdvp2_step!(...; weight_tol)` (every two-site update).
- Every cut is allowed `tol_trunc·‖ψ‖/_cut_divisor(L, L÷2+1)`.
- Before (≤ s25), tdvp2 cut each bond with `tol/(2√(L−1))`, about 1.4x tighter. **Every tdvp2e
  result before s26 used the old rule** (ref1024t, kscreen/kscreen2, the Phase 6 runs, the
  earlier references).
- tdvp2's `info.cumulative` is now reported, not bounded (it can reach ≈ √2·`tol_trunc`). The BUG
  is unchanged.

### 2.3 Other user questions, answered from the code

- **Tests A/B have no knob of their own:** both get `tol_basis/(2·sqrt((L−1)·p))`, scaled by
  `|dt|/‖ψ‖` (`basis_allocation`, `adaptive_budget.jl`).
- **Test B** stops on an absolute holdout-residual threshold, not machine precision.
  - Holdout = ¼ of the sketch width, drawn fresh, with a per-sector unbiased estimate.
  - On failure it doubles the sectors the holdout named and **re-draws the whole sketch from
    scratch**.
  - The old wide search (`thin_search = false`) widened to numerical rank (`stol_pre = 0`) before
    Test B was ever consulted.
- **Test A** runs once per bond, after B passes on both sides: the smallest k whose tail fits, on
  the SVD of the small `M = Q_L† G Q_R`. All LAPACK svd/qr was 3.8 % of a D = 256 step (pair2
  17285246).
- The user suggested checking Test B only every 2–3 iterations and a rank-revealing QR for
  Test A. I showed the data (they are not the closure's cost) and proposed reuse instead. **Not
  decided or implemented.** See §5.

## 3. Code changes (uncommitted; snapshots s24–s27 on the cluster hold the exact files)

| file | change |
|---|---|
| `src/RSVDCBEBondUpdate/sweeps/krylov_closure.jl` | `checks` kwarg, `_single_pass_closure!` |
| `src/RSVDCBEBondUpdate/sweeps/cbe_bug.jl` | `closure_checks` kwarg; docstring; root via `_bug_root_link` |
| `src/RSVDCBEBondUpdate/zero_site_core.jl` | `_bug_root_link`, `_cut_divisor`; `truncate_recursive!` uses them |
| `src/RSVDCBEBondUpdate/tdvp2/tdvp2_baseline.jl` | per-cut allowance = `_cut_divisor` (was `2√(L−1)`); docs |
| `src/RSVDCBEBondUpdate/adaptive_budget.jl` | comment only |
| `tests/sweeps/test_krylov_closure.jl` | UT1/UT6 under both settings; parallel test and UT9 use `checks = true`; new testset "closure_checks" |
| `tests/sweeps/test_tdvp2_truncation.jl` | asserts the identical per-cut allowance for both integrators |
| `benchmarks/spectral/fig45_2209.jl` | steps CSV + G CSV written per step and flushed; `bonds` column (all bond dims); `fig45_G_<tag>_exact.csv` |
| `benchmarks/spectral/growth_vs_exact.py` (new) | error vs exact, χ_max(t), discarded per step, per arm; partial records OK; `--plot` |
| `krylov_closure_cbe_bug_realtime_L16_plan.md` | §8 amendment (decision + timing table) |

Also uncommitted from before this session: `TASK4_VALIDATION_RESULTS.md` (modified),
`benchmarks/spectral/order_ladder_2d.jl`, `AGENTS_HANDOVER_5.md`, `handover5_figs/`.

**Verification (all on PKG `40863a843a5f`):**
- **s26test 17332175** (gate 9 clean, 4.7 G):
  - tdvp2 truncation 11/11 — at `weight_tol` 1e-2: share 2.13e-3, tdvp2 worst cut 1.74e-3 (above
    the old share 1.51e-3), BUG worst cut 0.97e-3;
  - adaptive search 38/38, adaptive parallel 32/32, driver smoke OK;
  - closure 301/305 — the 4 failures were a vacuous fixture in the new testset;
  - the order-2/3 stress test vs exp(−iHt) with checks off: 4.5e-8, seed-independent.
- **s27test 17332524** (test-only fix: the testset searches for a case that grows): closure
  **307/307**. Order 2, checks off: 1 sweep adding 4 directions; checks on: 5 sweeps.

**Snapshots:**
- s24: fig45 only, old PKG `9f6d412f1cf5`;
- s25: closure only, PKG `6634428a80e6`, never tested;
- **s26 = s27: current code, PKG `40863a843a5f` (compiled, warm).**

## 4. Results this session

### 4.1 nb24 scan — fair-rule error and χ(t) at a non-binding cap (6x4, vs exact, T = 1)

Jobs 17332220–24, `2026-10-17-nb24/scan/<arm>/`, graded in `scan/graded/`. Figure:
`handover5_figs/fig10_nb24_scan.png`. dt 0.05; GS D_GS 1024 (E0 = −15.7284509714, χ 954,
DMRG − exact = +7.8e-11). Gate 9 clean.

| arm | err vs exact at t = 1 | χ at t = 0.5 / 1 | wall to t = 1* | MaxRSS |
|---|---|---|---|---|
| tdvp2e | 9.9e-9 | 1411 / 1920 | 2260 s | 7.6 G |
| bug2 `tol_basis` 1e-3 | 1.2e-5 | 969 / 1215 | 924 s | 7.6 G |
| bug2 `tol_basis` 1e-4 | 3.5e-7 | 1268 / 1775 | 1458 s | 11.2 G |
| bug2 `tol_basis` 1e-5 | **1.0e-8 (matched)** | 1569 / 2044 | 2959 s | 15.1 G |
| bug2 `tol_basis` 1e-6 | 1.35e-9 | 1657 / 2173 | 5077 s | 17.4 G |

\*All five ran on one node (th-cl-rome03n4), so walls are indicative.

- **At matched error (`tol_basis` 1e-5), bug2 carries 6–11 % MORE χ than tdvp2e and is about
  1.3x slower.** So far this does NOT show better rank control or speed for the BUG.
- bug2's error ≈ 0.018·`tol_basis` per unit time: it is basis-limited down to `tol_basis` ≈ 1e-5.
- tdvp2e is truncation-limited near 1e-8. It makes 2(L−1) cuts per step against the BUG's L−1,
  under the same per-cut rule, and discards about 1.5x the norm.
- At 1e-6, bug2 is about 7x more accurate than tdvp2e, with 13 % more χ.

### 4.2 fair10 — timing, 10x4, binding caps, t = 10 (job 17332661, `2026-10-17-fair10/`)

- The job ran on th-cl-rome03n3, away from my other jobs.
- **It was OOM-killed at 20 G** in bug2 `tol_basis` 1e-3 at D = 512, step 17 (t = 1.7). The same
  arm and cap died at 32 G in ref1024b, so the closure frames grow.
- **Completed parts:**
  - D = 256, all arms;
  - D = 512 tdvp2e: 100 steps at 77–83 s/step;
  - D = 512 bug2 1e-3: steps 1–17 at about 156 s/step.

  G records: `bp_G_10x4_*.csv`.
- D = 256 (wall per unit time, JIT included):

  | arm | wall | vs tdvp2e |
  |---|---|---|
  | tdvp2e | 321 s | — |
  | bug2 `tol_basis` 1e-3 | 479 s | 0.67x (slower) |
  | bug2 `tol_basis` 1e-4 | 440 s | 0.73x |

  - Infidelity vs tdvp2e at t = 10: 0.11 and 0.39. Both arms are truncation-dominated at
    D = 256: tdvp2e discards about 1e-2 per step late on.
  - **Not yet graded** against ref1024t (+ ref1536t when done) with `matched_error.py`.
- At D = 512, bug2 1e-3 is about 0.5x tdvp2e's speed at steps 15–17.

### 4.3 Other jobs that finished

| job | result |
|---|---|
| p6sq_cbe1se_256 17331889 | COMPLETED (MaxRSS 3.5 G) |
| p6sq_cbe1se_512 17331891 | COMPLETED (3.7 G) |
| p6sq_cbe1se_1024 17331893 | **OOM at 6 G** (MaxRSS seen 6.2 G) → rerun at ~8 G |
| p6sq_tdvp2e_256 17331888 | COMPLETED, OLD tdvp2 rule (before s26) |
| p6j1_test 17331902 | COMPLETED, all EXIT 0. MaxRSS per arm on J1-J2 10x4 D = 1024: tdvp2e 5.2 G, cbe1se 5.3 G, **bug2 15.4 G** (old closure) |
| p6gs_kagome 17305633 | **OOM in the variance, after caching the state.** `p6/gs/kagome/gs_kagome_4x3_J20p0000_Dgs1024_adaptive.jls` (120 MB, 17:34) exists → rerun the variance-only job (`gs_kagome.sbatch`) at 32 G |
| nb24 J1-J2 GS 17331994 | E0 = −12.4399615001 (−0.5183317292 per site), χ 941, cached in `2026-10-17-nb24/gs/j1j2/` |

**Cancelled as superseded by the s26 settings:** pareto 17331886 (its dt 0.2 process had OOMed
at 12 G; old-closure bug2 at dt 0.2 took 181–295 s vs tdvp2e 121 s/step), t10knob 17331901,
p6sq tdvp2e 512/1024, p6sq bug2 256/512.

## 5. Jobs running at handoff

| job | what | dir |
|---|---|---|
| 17331887 `ref1536t` | second reference, tdvp2e D = 1536 to t = 10 (OLD tdvp2 rule; fine as a reference). At step 44, ~970 s/step → ~15 h left | `2026-10-17-ref1536/` |

Other jobs on the account (`long_r-*`, `rs3*`, `rs308*`, the held `ms*`/`gr*`) belong to the
Hubbard track. Leave them alone.

## 6. Next tasks (in the user's order)

1. **Ask the user** about:
   - committing §3 (tests pass);
   - what to rerun of the pre-s26 tdvp2e results;
   - how to make bug2 competitive (item 3).
2. **Fair errors and rank:**
   - grade `fair10` D = 256 and partial D = 512 against ref1024t, then against ref1024t +
     ref1536t when ref1536t finishes;
   - repeat the nb24 scan for **J1-J2** (GS cached; 24 G per job, bug2 at 1e-6 needed 17.4 G →
     use 20–24 G);
   - optionally a longer T (2–4) at `tol_basis` 1e-5 vs tdvp2e on 6x4 to see whether the χ gap
     grows or closes.
3. **bug2 efficiency at matched error (`tol_basis` ≈ 1e-5).**
   - **Profile first** (`bug_profile_2d.jl` profile mode, 10x4 D = 512 or 6x4 at 1e-5). The
     6x4 per-step walls are in `fig45_steps_*.csv`.
   - Candidates that skip no check:
     - (a) **incremental Test B re-sketch:** keep `Y`, apply H only to the added columns, and fold
       the failed holdout into the sketch (`_certified_range`, `cbe_core.jl:791`);
     - (b) drop the redundant second SVD in Test A (`cbe_core.jl:1267`), `_one_sided_tail` and
       `_budget_svd` — slice the first SVD instead;
     - (c) the closure's per-bond leak measurement and frame SVDs (`_bond_action`,
       `_frame_onto_left/right`).
   - Decoupling the A/B/C/D or closure shares needs user approval (plan §1/§2).
   - **Memory:** bug2 needs >20 G at 10x4 D = 512 by t ≈ 1.7, because its closure frames grow
     toward 2χ. Fix before any t = 10 D ≥ 512 bug2 run.
4. **Phase 6 square reruns on s26** (tdvp2e, bug2 at `tol_basis` 1e-5, caps 256/512/1024, T = 8,
   dt 0.05). Size bug2 from p6j1_test (15.4 G at D = 1024) and the memory issue above. Rerun
   cbe1se_1024 at ~8 G. Then J1-J2 and kagome (after the kagome variance), S(q,ω), and the plan
   §8 figures.
5. Older open items:
   - U22 (bug2/bug4 order on 6x4 on the current PKG);
   - `cbe1s` has no budget rule.

## 7. Where things are

- **Drivers:**
  - `benchmarks/spectral/fig45_2209.jl` — Phase 6, nb24, validation; `F_*` env; per-step CSVs;
  - `benchmarks/spectral/bug_profile_2d.jl` — timing and profiling; `BP_*` env;
    `BP_CLOSURE_LOG=1`.
- **Graders:**
  - `benchmarks/spectral/growth_vs_exact.py DIR TAG --times ... --plot out.png` — vs exact; it
    imports `matched_error.py`, so run it from `benchmarks/spectral/`. The cluster has no
    matplotlib: copy the CSVs locally to plot;
  - `benchmarks/spectral/matched_error.py --ref A B TEST...` — vs the two-reference mean.
- **Cluster dirs** (`$SCRATCH/rsvd_adaptive/`):
  - `2026-10-17-nb24/{test,gs/j1j2,scan}`;
  - `2026-10-17-fair10`;
  - `2026-10-17-s26`, `2026-10-17-s27` (test logs);
  - `2026-10-16-p6/...`;
  - `2026-10-16-ref1024/tdvp2e`, `2026-10-17-ref1536`.
- **Key code:**
  - closure: `src/RSVDCBEBondUpdate/sweeps/krylov_closure.jl`;
  - Tests A/B: `cbe_core.jl` (`_certified_range`, `_sample_residual`, `_tail_rank`);
  - allocation: `adaptive_budget.jl` (`basis_allocation`);
  - truncation: `zero_site_core.jl` (`_budget_svd`, `_cut_divisor`, `truncate_recursive!`).
- **Memory notes:**
  - `order-p-step-is-half-sweeps-plus-one-enriching-sweep-checks-off`;
  - `fair-truncation-is-the-same-per-cut-rule-not-a-per-step-budget`;
  - `focus-shift-errors-and-bond-growth-before-walltime`.

## 8. Continuation (2026-10-17 late)

**USER answers to §6.1:** commit §3 (done); rerun the Phase 6 tdvp2e arms only (references stay
on the old rule); bug2: profile first, then (a) + (b).

- **Commits:** `86dea31` code + tests + plan §8, `5dfd528` benchmarks, `f588ef9` docs/figures.
  s27 matches HEAD (md5).
- **fair10 graded** vs the mean of ref1024t and ref1536t (t ≤ 4.5; refs agree to ≤ 9e-7):

  | arm | D | wall/unit t | err t=1 | t=2 | t=4.5 |
  |---|---|---|---|---|---|
  | tdvp2e | 256 | 321 s | 7.9e-6 | 3.0e-5 | 2.4e-4 |
  | bug2 1e-3 | 256 | 479 s | 2.9e-4 | 2.1e-3 | 1.5e-2 |
  | bug2 1e-4 | 256 | 440 s | 7.2e-4 | 7.4e-3 | 6.6e-2 |
  | tdvp2e | 512 | 797 s | 2.8e-7 | 2.7e-6 | 2.2e-5 |
  | bug2 1e-3 | 512 (to t=1.7) | 1207 s | 2.2e-5 | — | — |

  The per-step discarded norm is equal (1.4e-2 vs 1.1e-2 at D = 256), and bug2's `closure` is
  unmet on every step: at a binding cap bug2 is basis-limited, and tighter `tol_basis` is WORSE
  (not diagnosed).
- **Jobs:**
  - prof24 17339380 (TIMING, th-cl-hua07; `2026-10-17-prof24/`): bug_profile_2d 6x4, cap
    4096, dt 0.05, 10 steps; tdvp2e vs bug2 `tol_basis` 1e-5, fold 0 and 1; serial-step
    profiles; closure log.
  - Phase 6 reruns (`2026-10-17-p6s26/square/D*/`, they exclude hua07):
    - 17339381–83: p6r_tdvp2e 256/512/1024 (s27, 4/5/6 G);
    - 17339384: p6r_cbe1se_1024 (s23 like its siblings, 8 G).

### 8.1 prof24 result (17339380, th-cl-hua07 Broadwell, MaxRSS 11.6 G)

6x4, cap 4096 (non-binding), dt 0.05, 10 steps (t = 0.5), `tol_basis` 1e-5:

| arm | mean s/step | chi at t=0.5 | notes |
|---|---|---|---|
| tdvp2e | 123.8 | 1411 | GC 55 % of samples |
| bug2 fold 0 | 168.9 | 1572 | |
| bug2 fold 1 | 171.1 | 1573 | infidelity vs fold 0 1.6e-11 |

- bug2 runs at 0.72x tdvp2e's speed. The infidelity between the two at t = 0.5 is 5.7e-11. bug2 carries 11 % more chi (as in nb24).
- bug2 serial-step phases (343 s total):

  | phase | time | share |
  |---|---|---|
  | closure (frontier j=1 enriching sweep) | 149 s | 43 % |
  | kry (Tests C/D) | 62 s | 18 % |
  | cbe (sketch + Test B) | 56 s | 16 % |
  | env | 19 s | |
  | root | 12.5 s | |
  | trunc | 12 s | |
  | frame | 8.5 s | |

- So (b), Test A's re-SVD (`_budget_svd` + `_extend_frame`:197), costs < 3 %.
- (a), the Test B re-sketch, is bounded by the cbe share.
- **The closure sweep is the cost.** This is PARKED: the USER moved to the imaginary-time/open-system plan on 2026-10-18 (Task 1 on `plan/imag-open-t1`).

## 9. Imaginary-time / open-system plan (USER 2026-10-18)

The plan is `cbe_bug_imaginary_open_benchmark_research (1).md`: seven tasks, done in order, one
branch per task, each merged into `feature/krylov-closure-order` once confirmed. Tolerances:
`tol_trunc` = `tol_root` (= `tol_expv`) = 1e-6, with `tol_basis` as the knob.

- **Task 1** — normalised imaginary time. Merged `3d94792`; s28test 17340070 + s31test 17340175.
  - `expv(...; root_method = :exp|:trapezoid|:radau2|:radau3)`: the Padé roots run on the
    Ritz-shifted Krylov space.
  - `shift_h` / `eshift` on the generators.
  - `_kry_weight` in imaginary time is now relative to the propagated seed and shifted (before:
    an e^(dβ|E|) over-weight).
  - `cbe_bug_step!(...; root_report, outside_residual)`.
  - `sweeps/imaginary_time.jl`: `ite_step!` / `ite_evolve!` (Eshift = E, energy split
    root/compression, outside defect vs `tol_basis`, re-enrich → split → flag), panel/uniform
    grids.
  - IT-U1..U6 pass. The order test needs the domain wall: from Néel, p = 1 and p = 2 are
    identical. Slopes are p+1.
  - The first step from a product state is flagged `:outside`.
- **Task 2** — symmetry-aware MPDO. Merged `1aba5b5`; s33test 17340370 + s34test 17340453.
  - `models/liouville.jl`: weak / strong / none Liouville sites built below the IROP layer; the
    covariance check; the stored-transpose rule; `lindblad_mpo` = `pair_mpo` + `field`; block-walk
    trace / expectation / RDM / dense.
  - `cbe_bug_step!(...; keep_states)` with `mpdo_identity` makes the step trace-exact (1e-15 vs
    0.37 drift without it).
  - OS-U1..U7 pass. The MPO equals the dense Liouvillian to 2e-16.
- **Task 3** — IT-1 Heisenberg chain, branch `plan/imag-open-t3`.
  - Driver: `benchmarks/imaginary_time/it1_heisenberg_chain.jl`.
  - Smoke 17340494, L = 12, β = 8: bug2 vs exact E(β) 1e-7 (panel) / 2e-7 (uniform), against
    tdvp2 3e-7 / 2.4e-6 on the same grids.
  - Production 17340511–13 (L = 12/16/20, β = 64, `panel:9:4` + `uniform:256`; bug2/3/4 ×
    `tol_basis` × 3 seeds; tdvp2), in `2026-10-18-it1/L*/`.
  - Next:
    - plots;
    - L = 100 (DMRG reference; a TIMING job, so it runs alone). The BUG wall includes the
      `ite_step!` energy and outside measurements; separate them before quoting a timing.
    - USER decision pending: the flagged Néel first step — accept-and-label (current) or
      pre-grow the rank.
- **Staging:** use `git -c core.autocrlf=false archive HEAD`. A branch switch rewrites files with
  CRLF and changes the PKG hash for identical code.
