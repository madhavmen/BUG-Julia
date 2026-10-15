# Plan audit: adaptive CBE-Krylov (`adaptive_cbe_krylov_implementation_plan_first_test.md`)

One row per plan requirement, checked BY CALL SITE (the code that runs on the production path,
not the code that exists). Re-done at the end of every §9 stage, before that stage is called
done. Status:

- `DONE` -- implemented on the production path AND exercised by a test that calls that code
- `PARTIAL` -- implemented, but a named piece is missing or untested
- `DIVERGES` -- the code does something other than what the plan says
- `DEFERRED(Sn)` -- the plan schedules it for a later §9 stage; current behaviour stated

Line numbers are as of the audit date; symbols are the durable reference.

---

## Stage 1 -- clean API and diagnostics (§1, §2, §9.1; UT01-UT02). Audited 2026-10-14

| # | Requirement (plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| 1.1 | `cbe_bug_step!` takes `tol_basis`, `tol_trunc`, `tol_expv`, `maxdim` (§1) | `cbe_bug.jl:332-334,400` | DONE | |
| 1.2 | Default `maxdim = 128` (§1 signature) | `cbe_bug_step!`: `maxdim = something(maxdim, adaptive ? 128 : 200)` | DONE | Fixed 2026-10-14. The legacy path keeps 200 so recorded benchmarks are unchanged. |
| 1.3 | Midpoint uses the same accuracy controls (§1) | `cbe_bug_midpoint_step!` forwards `kwargs...` | DONE | Ledger now merged across both stages (`cbe_bug.jl:1140`). |
| 1.4 | Retired knobs cannot be mixed with the tolerances; no silent double honouring (§1) | `resolve_budget` `adaptive_budget.jl:251`, called `cbe_bug.jl:483` | DONE | Refused with `ArgumentError`. |
| 1.5 | Old names: optional deprecation warning with migration hint (§1, "may") | `_legacy_hint` in `cbe_bug_step!` (see R.3) | DONE (R.3) | Closed 2026-10-15, job 17198444. |
| 1.6 | Normal API has ONLY three tolerances + `maxdim` as accuracy knobs (§1, UT01, acceptance) | `krylov_tol`, `grow_iters`, `krylov_grow` (S3) and `maxiter`, `tol`, `root_conv_tol`, `root_substeps` (S5) are in `LEGACY_KNOBS`: refused alongside a tolerance | DONE (S3, S5) | Remaining public keywords are method/diagnostic inputs, not accuracy knobs: `root`, `truncate`, `hermitian`, `parallel`, `rng`, `share_ht`, `fold_omega`, `reorth`, `frame_state`. |
| 1.7 | `tol_expv` drives the root `expv` (§1, §4) | adaptive: `root_conv_tol = budget.expv`, `root_substeps = 0` in `cbe_bug_step!` | DONE (S5) | See 5.1. |
| 1.8 | `tol_trunc` is the final compression tolerance (§1) | `cbe_bug.jl:504` `trunc_thresh = budget.trunc` on the adaptive path | DONE | Single final pass is S4. |
| 1.9 | One fixed, documented allocation of `tol_basis` (§2) | `basis_allocation` `adaptive_budget.jl:110`: `per_bond = tol/sqrt(L-1)`, each test `per_bond/2` | DONE | |
| 1.10 | Scale `|dt|/||psi||`; `dt == 0` without division by zero (§2) | `basis_allocation`; `_nrm` `cbe_bug.jl:513` | DONE | `tau = 0` gives scale 1. |
| 1.11 | Step-level ledger, max and aggregate indicators (§2) | `AdaptiveLedger`, returned as `CBEBugSweepInfo.ledger` | DONE | Fixed 2026-10-14; it was built and discarded before. |
| 1.12 | `:budget_unmet` / retry diagnostic when infeasible (§1, §3) | `budget_status(ledger)` -> `:met`/`:budget_unmet`/`:legacy`; `budget_diagnostic` names each failed check (`ledger.unmet`), its worst indicator and the retry | DONE (S4) | Closed in Stage 4, see 4.1. |
| 1.13 | UT01: inspect accepted kwargs / migration errors (§7) | UT01 "the PUBLIC entry refuses the mixture before touching the state" (`cbe_bug_step!` and the midpoint step) | DONE | Fixed 2026-10-14; job 17190027 passed. |
| 1.14 | `stol_pre` replaced by scale-aware numerical rank detection (§1 table, UT02) | the adaptive path passes `stol_pre = 0.0`, so `_rank_cut` gives a floor of `100 eps ||Y|| / ||P_perp Y||` | DONE | Fixed 2026-10-14. The floor is measured against the pre-projection size, so cancellation cannot pass noise off as directions. Test A decides physical importance. |
| 1.15 | UT02 tests the preselection (§7) | UT02 calls the production `_preselect_left` on a real bond: amplitude invariance, a redundant direction, a weak direction, never below the legacy cut | DONE | Fixed 2026-10-14; job 17190027 passed. |
| 1.16 | `stol_fnl` replaced by Test A (§1 table) | joint SVD still cuts at `stol_fnl = 1e-13` before Test A; Test A counts that cut weight as `base` (`cbe_core.jl:1155`) | DONE | The residual cut is accounted for, not hidden. |

## Stage 2 -- sector coverage + RSVD Tests A/B (§3, §9.2; UT03-UT06). Audited 2026-10-14

| # | Requirement (plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| 2.1 | ≥1 probe per nonempty admissible complement sector; never a forbidden one (§3) | `_alloc_columns` ceil `cbe_core.jl:228`; `sector_graded_sketch` `:312` over `fusion_basis` sectors | DONE | UT03 real-frame test, made non-vacuous by an empty-but-reachable sector. |
| 2.2 | Probes inside charge blocks; matrix-free, no dense two-site matrix (§3) | `_randomize_like` on fusion blocks; `skl`/`skr` closures | DONE | `exact = true` (the full basis) is the A/B reference, and Test B is skipped on it (`cbe_core.jl:1085`). |
| 2.3 | Test B: independent holdout, BOTH sides (§3) | `_certified_range` `:718` for left and right (`:1088`, `:1097`); holdout drawn fresh, never boosted | DONE | |
| 2.4 | Test B is an unbiased matrix-free estimate (§3) | `_sample_residual` `:632`, per-sector normalised | DONE | Fixed 2026-10-14 (it was pooled, biased low). UT04 checks 200 draws against the exact residual. |
| 2.5 | Sector-resolved estimates AND sampling uncertainty recorded (§3) | `_record_sample!` -> `ledger.sample_log` (per-sector indicator, holdout and sketch columns, relative-error bound) | DONE | Fixed 2026-10-14; checked in the end-to-end test. |
| 2.6 | On B failure, add probes to the worst sector(s) (§3, UT04) | `_target_sectors!` `:681` + `sector_graded_sketch(...; boost)` | DONE | Fixed 2026-10-14; it was a uniform doubling. |
| 2.7 | Loop checks rank exhaustion, complement exhaustion, work cap, unmet status (§6) | `_certified_range`: `n >= ncap`, `width <= prev`, `width >= wcap` | DONE | Reported through `tolerance_met`/`workcap_reached`; see 1.12 for the missing `:budget_unmet`. |
| 2.8 | Reliability over multiple RNG seeds (§3, §7) | UT04 averages 200 holdout draws; IT03: 5 seeds x {Neel, DW} x {ordinary, midpoint} x {loose, tight} at L=20 | DONE (S6) | Seed spread of err(T) <= 5e-10 in every cell (job 17195152). |
| 2.9 | Test A: smallest `k_keep` with projected tail ≤ budget, global across charge blocks, degeneracy-weighted (§3) | `_tail_rank` `:587` on Telum's GLOBALLY sorted `kept_list` (`deps/Telum/src/svd.jl:1741`) | DONE | UT05 includes the plan's (10, 9, 3, 0.1). |
| 2.10 | Test A applies to EVERY retained selection (§3) | joint branch, plus `_one_sided_tail` when no joint SVD exists | DONE | Fixed 2026-10-14. UT05 checks both chain-edge bonds: monotone in the allowance, and nothing kept when everything fits. `preselect_only` remains a legacy study arm. |
| 2.11 | Decision order B, then A; a spurious small A tail is never accepted (§3, UT06) | `_certified_range` runs before the joint SVD | DONE | UT06: blind sketch passes A, fails B, then is repaired. |
| 2.12 | No holdout or unused oversampling in the physical frames (§3) | holdout feeds only the residual; Test A trims the joint rank | DONE | Except the fallback branch, see 2.10. |
| 2.13 | Work-rank ceiling derived from `maxdim` (§5) | `work_ceiling` caps the probe width AND (Stage 4) the retained width `r + n <= workcap` in `cbe_expand` | DONE (S4) | Closed in Stage 4, see 4.2. |
| 2.14 | Diagnostics: `n_reachable_sectors`, `probes_per_sector`, holdout count, selected, A, B-left, B-right (§8) | `max_sample_l`/`max_sample_r`; sector counts and probes per sector in each `sample_log` entry | DONE | Fixed 2026-10-14. The Krylov and root diagnostics are S3/S5. |
| 2.15 | Compatible with parallel half-sweeps (handoff requirement) | per-half ledger + RNG stream, `cbe_bug.jl:582`, merged `:991` | DONE | Fixed 2026-10-14 (data race). Reproducibility test in the end-to-end set. |

## Stage 3 -- residual-driven K/L Krylov, Tests C/D (§4, §5 K/L, §9.3; UT07-UT10). Audited 2026-10-14, before submission

Interpretation, stated because the plan leaves it implicit: the Krylov recursion runs on ONE-SITE
tensors, and the plan's `Q`/`I - QQ'` is the half-sweep FRAME (the isometry on the local legs).
A Krylov vector adds directions only through the part of its local range outside the frame.

| # | Requirement (plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| 3.1 | Selected CBE directions transported into the seed's one-site space and charge blocks; `Q0 = orth[U_old, U_CBE, K_ex]` (§4) | `Q0 = ex.U_ex` (`= [U0 | TLC]`, already orthonormal) in `half_left!`; `ex.V_ex` in `half_right!`; `K_ex`'s local range lies in `U_old` | DONE | UT10: `Q0`, the old frame and the seed all lie in the final frame, on both sides. |
| 3.2 | Test C `eta_C = |dt| ||(I-Q0Q0')H1 K_ex|| / ||K_ex||`; NOT a stopping test alone (§4) | `_krylov_enrich`, `k = 1`: `weight_1 * ||(I-QQ')v_1||`, with `weight_1 = |tau beta_1 phi_1(tau alpha_1)|` | DONE | Equal to the plan's `eta_C` to `O(tau^2)`, since `(I-QQ')K_ex = 0`. UT07 (zero, nothing appended); UT08 (zero, loop continues). |
| 3.3 | Test D: frontier residual `(I-Q_mQ_m')H1 V_m`; significant -> orthogonalise per sector, append, continue; stop when negligible AND the contribution check passes (§4) | `_krylov_enrich`, `k >= 2`: `weight_k * ||(I-QQ')v_k||`; `_extend_frame` (block SVD, tail rule, reortho); stop on `weight_k <= allowance` or exact closure | DONE | Uses the Arnoldi vector `v_{k+1}`, i.e. `H1 V_m` with the already-scored earlier directions removed. UT08 finds `e3`; UT09 adds rank only for the significant part. |
| 3.4 | No fixed `krylov_basis`, no public `tol_krylov` (§4, §1 table) | the adaptive branch ignores `krylov_basis`; `krylov_tol`, `grow_iters`, `krylov_grow` are in `LEGACY_KNOBS` and refused alongside a tolerance | DONE | Closes the `krylov_tol` half of audit row 1.6. Test: "the fixed-depth Krylov knobs are legacy-only". |
| 3.5 | Shares the rank/work ceiling and the ledger with CBE (§4) | `wcap_step` (one value, given to `cbe_expand` and `_krylov_enrich`); `_record_krylov!` into the half-sweep's ledger | DONE | Depth bound `_KRY_MAXDEPTH = 64` is a safety stop; reaching it is reported (`capped`, `met = false`). |
| 3.6 | Test D does not force closure of dormant, zero-amplitude directions (§4) | only the seed's Krylov space is visited, each direction weighted by its propagator weight | DONE | |
| 3.7 | Cache `H` actions, reorthogonalise (§4) | one `applyH` per Krylov vector; CGS2 against the whole basis, as in `_krylov_frame` | DONE | |
| 3.8 | During K/L: keep the old subspace, orthogonality, symmetry; only CBE plus validated Krylov directions; numerical cleanup only, no `split_cutoff` (§5) | the adaptive frame IS `Q`; no `_splitU`/`_splitV` on the adaptive branch; numerical floor `_rank_cut` | DONE | Brought forward from Stage 4 for K/L. `split_cutoff` remains a legacy-path knob. |
| 3.9 | Diagnostics: `n_krylov_added`, C, D, `n_H1`; count C-pass and D-after-C (§8) | `n_krylov`, `max_krylov_c`, `max_krylov_d`, `n_h1`, `n_c_pass`, `n_d_after_c` | DONE | Printed by the Stage 3 end-to-end test. |
| 3.10 | UT07-UT10 (§7) | `tests/sweeps/test_adaptive_krylov.jl` | DONE | Job 17192249: 68/68; end to end XX L=12 err 1.69e-4 (bar 5e-4), `tolerance_met`, no work cap; full sweeps 1007/2/2 = the 4 baseline reds only. ⚠ UT10 WAS VACUOUS there: the default CBE frame filled the local space (16 -> 16, CBE alone missed 1.8e-16). Fixed in the S4 job: `dex = 1` CBE frame plus a non-vacuity precondition (CBE alone misses > 100x the allowance) and `added >= 1`. |
| 3.11 | Each direction's leftover is held to the allowance, so a bond's total K/L leftover is at most (depth x allowance) | per-direction test | DONE (S6) | Checked against IT07 (job 17195153): err(T=0.5) ~ 0.07-0.15 x tol_basis across five decades, i.e. the allocation is CONSERVATIVE, not over-spent. Kept. |

## Stage 4 -- rank cap and compression cleanup (§5, §9.4; UT11-UT14). Audited 2026-10-14, before submission

| # | Requirement (plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| 4.1 | Infeasible accuracy gives an explicit `:budget_unmet` with a useful diagnostic, never a silent pass (§1, §6 (iv), UT11) | `_unmet!(ledger, what)` is now the ONLY place `tolerance_met` is cleared: `:sample` (`_certified_range`), `:tail` (joint and one-sided Test A), `:krylov` (`_record_krylov!`), `:trunc`, `:projection` (`cbe_bug_step!`); `budget_status`, `budget_diagnostic` (exported) | DONE | `grep "tolerance_met = false" src` is empty. UT11: ledger-level, CBE-level, Krylov-level and step-level. |
| 4.2 | Work ceiling caps sketch sizes AND the retained CBE/Krylov directions; report it (§5) | `cbe_expand`: `cap_* = min(cap_*, workcap - r)`, `workcap_reached` when that cap binds and fills; `_krylov_enrich`: `room = workcap - width` passed to `_extend_frame`, which appends at most `room` and returns `fits` | DONE | The Krylov frame could overshoot the ceiling by one block before (checked only BEFORE extending). UT11 bounds both and the step's `r_aug <= work_ceiling(maxdim, r_old)`. |
| 4.3 | During K/L: no physical Schmidt truncation, numerical cleanup only (§5) | adaptive branch has no `_splitU`/`_splitV` (3.8); `_extend_frame` cuts at `_rank_cut` only | DONE | Evidence on the production step: UT12 predictor `proj_err < 1e-7` (frames hold all of `psi`) and bond dims == frame widths. |
| 4.4 | Root structural SVD lossless (§5, UT13) | `root_cutoff = adaptive ? 0.0 : ...`, `root_maxdim = adaptive ? 0 : ...` in `cbe_bug_step!` | DONE | Pinned explicitly, no longer inherited from `LEGACY_DEFAULTS`. UT13: `discarded == 0`, norm conserved to 1e-10 in real time. |
| 4.5 | ONE final compression with `tol_trunc` and `maxdim`, tracking the cumulative discarded weight; a conservative estimate where one canonical pass does not justify Pythagoras (§5) | `truncate_recursive!(...; weight_tol = budget.trunc)`: per-bond share `tol/(sqrt(n_L)+sqrt(n_R))`, minimal rank by `_tail_rank`, `cumulative = (sqrt(Σ_L w)+sqrt(Σ_R w))/||psi||`; `ledger.discarded = cumulative` | DONE | ⚠ DIVERGED BEFORE: the adaptive path passed `tol_trunc` as Telum's RELATIVE singular-value cutoff per bond and reported only the worst single bond. Pythagoras holds within a subtree (each later state lies in the range the earlier cut kept), not across the two, hence the triangle combination. Legacy `cutoff` mode unchanged. |
| 4.6 | `maxdim` violations reported, not masked by the largest single SVD (§5, UT14) | `capped` from `truncate_recursive!` -> `ledger.maxdim_reached`; `cumulative > tol_trunc` -> `:trunc` | DONE | UT14: with `maxdim = 2`, `cum > worst` and `cum > tol`. |
| 4.7 | Midpoint: predictor keeps its enlarged basis; frames from `Y_{1/2}`, START from `Y_0`; projection error checked and charged to `tol_basis` (§5, UT12) | `truncate = false` forced in stage 1; `frame_state` rebuilds `AR` against `psi`; `ledger.proj_err` vs `max(tol_basis, _PROJ_FLOOR)` -> `:projection`; midpoint keeps stage 2's `r_final` | DONE | Norm-difference formula: resolution floor ~1e-8 (`_PROJ_FLOOR = 1e-7`). UT12 accuracy over T = 1 (a `Y_{1/2}` start would evolve to 1.5 T; the test checks it can tell). |
| 4.8 | Diagnostics `r_old`, `r_augmented`, `r_final`, `state_projection_error`, `cumulative_discarded_weight` (§8) | `ledger.r_old/r_aug/r_final/proj_err/discarded` | DONE | |
| 4.9 | UT11-UT14 + midpoint regression (§7, §9.4) | `tests/sweeps/test_adaptive_compression.jl` | DONE | Rerun 17195151: 67/67. Job 17193333: 65/67. The two reds were ONE testset's precondition (`room >= 2`): the default CBE frame already filled the local space, the UT10 situation. Fixed with `dex = 1`; rerun 17195151. Measured: UT11 diagnostic names `maxdim`; UT12 midpoint projection error 3.7e-8; midpoint XX L=10 T=1 dt=0.05 err 3.0e-7 (ordinary L=12: 1.7e-4). Full sweeps 1100/5/2 = 4 baseline reds + these 3 test-precondition reds: no regression from the new compression rule. |
| 4.10 | `tol_trunc` changes the adaptive compression semantics; the legacy path is unchanged | `adaptive ? weight_tol : cutoff` | DONE | Expect adaptive ranks to move slightly vs Stage 3 (the per-bond share is an absolute weight, not `1e-7 * sigma_max`); the full sweeps run is the regression check. |

## Stage 5 -- root `expv` bound to `tol_expv` (§1, §4 "Root Lanczos", §9.5; UT15). Audited 2026-10-14, before submission

| # | Requirement (plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| 5.1 | Root `expv` convergence is automatic from `tol_expv`; iterations/restarts adapt (§1, §4) | adaptive path: `conv_tol = budget.expv` (Saad's estimate stops the basis), `substeps = 0` (splits `tau` when the per-substep basis `_ROOT_MAXITER = 30` cannot reach it) | DONE | Existing, already-measured machinery (`BondUpdateBUG/expv.jl`); no new solver, per §0 "out of scope". |
| 5.2 | A posteriori residual reported; too large -> more iterations/restarts internally, else an explicit status (§4) | `expv(...; report)` -> `(iterations, substeps, err)`, `err` = Saad's tail summed over substeps; `ledger.expv_err`, `ledger.expv_iters`; `err > tol_expv` -> `_unmet!(:expv)` with a diagnostic | DONE | |
| 5.3 | `tol_expv` does not control RSVD or K/L Krylov (§1 table) | `budget.expv` read only at the root `expv` call | DONE | |
| 5.4 | Old root controls cannot silently act on the adaptive path (§1) | `maxiter`, `tol`, `root_conv_tol`, `root_substeps` in `LEGACY_KNOBS`, resolved by `legacy_or_default` | DONE | Every in-repo caller that passes them also names another legacy knob (`trunc_thresh`/`krylov_basis`), so none changes path. |
| 5.5 | UT15: tiny Hermitian matrix, iterations adapt to `tol_expv`, agree with the dense exponential (§7) | `tests/sweeps/test_adaptive_expv.jl`: known `M` realised on orthonormal block-sparse tensors; restart case at `maxiter = 4`; step-level ledger and refusal checks | DONE | Rerun 17195151: 25/25. Job 17193333: 24/25. The red was `its[1] < n`: the random `M` had `||tau M|| ~ 1`, so every tolerance ran to closure of the 6-dim space (vacuous). `M` now normalised to `||M|| = 0.2`. Restart and step-level checks passed. Rerun 17195151. |

## Stage 6 -- XX L=20 benchmark (§8, §9.6; simulation_plan §4; IT01-IT07). Audited 2026-10-14

Driver `benchmarks/adaptive_xx_l20.jl` (graded per step against the exact free-fermion occupations),
summarised by `benchmarks/analyze_adaptive_xx.py`; results in `ADAPTIVE_XX_L20_RESULTS.md`.
Jobs in `$SCRATCH/rsvd_adaptive/2026-10-14-l20/`: 17195152 profiles, 17195153 sweeps, 17195154
baseline (T = 0.5), 17195155 sweeps, 17195335 combos, 17195336 maxdim (T = 2).

| # | Requirement (plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| 6.1 | IT01 occupations after 1 and 10 steps vs exact; tight better than loose (§8) | profiles: Neel ordinary 1.5e-4 (loose) -> 5.3e-7 (tight); DW midpoint 1.8e-6 -> 4.3e-8 | DONE | |
| 6.2 | IT02 U(1), norm, invariants (§8) | Sz drift <= 1.6e-14, norm drift <= 1.9e-11 over all profiles | DONE | Fusion/old-subspace invariants: UT10, UT12. |
| 6.3 | IT03 >= 5 seeds, Neel and DW; reproducible diagnostics (§8) | 5 seeds per cell; spread <= 5e-10; 0 unmet runs | DONE | |
| 6.4 | IT04 all-up: negligible needless enrichment (§8) | chi stays 1, err 9e-16, 0 Krylov directions | DONE | Residual cost: ~21 H applications/step (one Test C/D probe per bond plus the root). |
| 6.5 | IT05 old fixed-knob vs adaptive at matched accuracy (§8) | baseline vs sweeps/profiles, operator counts | DONE | Ordinary Neel at ~2e-7: legacy default 1416 applications (chi 29), adaptive tol_basis 1e-6 1094 (chi 42); legacy `krylov_basis = 3` 891 (chi 32). Midpoint at ~1e-9: legacy kry3 900, adaptive (tol_basis 1e-7) 1222. MIXED: cheaper than the legacy default, not than the hand-tuned shallow frame. Wall clock not compared (different jobs/nodes). |
| 6.6 | IT06 ordinary vs midpoint (§8) | profiles + `mid` sweep; UT12 | DONE | Midpoint reaches 1.6e-9 where the ordinary step floors at 2e-7 (its O(dt) error). |
| 6.7 | IT07 tolerance sweep, smooth response, `maxdim` failures visible (§8) | sweeps (per component, T = 0.5 and 2); maxdim phase: caps 4/8/16 all `:budget_unmet` (`trunc`, plus `krylov`/`tail`) with err 0.78 / 0.03-0.08 / 1e-3 | DONE | T = 2 (17195155): err ~ 0.2-0.9 x tol_basis; tol_trunc 1e-4 reaches the floor at chi 12 vs 128 and 3x faster; tol_expv 1e-2 within 1.3x of the floor. RSVD == exact CBE to every digit, RSVD 1.6x slower in CBE time (same job). Combos (17195335) all `:met`, roughly additive. Per-component looseness tables in `ADAPTIVE_XX_L20_RESULTS.md`. RSVD crossover (17195922, maxdim 256/512, T = 4): RSVD/exact CBE time 1.7-1.9 at chi 29 falling to 0.85-1.0 at chi 224-256, accuracy identical. |
| 6.8 | Acceptance: XX L=20 accuracy for multiple seeds and both profiles; explicit unmet status; only 3 tolerances + maxdim | 6.1-6.7; UT01 | DONE | |
| 6.9 | Acceptance: fewer samples/retained directions OR fewer H applications than the fixed baseline at comparable accuracy (simulation_plan §6) | matched-accuracy job 17195550 (legacy and adaptive arms in ONE process) | DONE | Ordinary BUG, Neel/DW, T = 0.5/2: at matched or better error, 5-8x fewer trial directions, 2-6x less rank, H applications within +8% to 2x fewer, faster wall clock than legacy `krylov_basis = 3` (and better than the legacy default on every count). Midpoint at ~1e-9: more H applications than kry3 (1222 vs 900) but a narrower trial space (58 vs 94), so the "or" holds. |

## Re-read of §1/§7/§8 (2026-10-15): requirements outside the stage rows. Audited before submission; job 17198444

| # | Requirement (plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| R.1 | §8 "Track by bond and timestep" | each half-sweep bond gets its own `AdaptiveLedger` (`bondlg` in `half_left!`/`half_right!`), passed to `cbe_expand` (explicit `ledger` wins in `timed_expand`) and `_record_krylov!`; `_log_bond!` pushes `(bond, side, sample, tail, krylov_c, krylov_d, counts, c_pass, d_after_c, workcap, met, unmet, width)` into `bond_log` and merges into the half-sweep ledger; midpoint tags `stage = 1/2` | DONE | Step-level fields are the max/sum of the bond entries BY CONSTRUCTION (every A-D counter is written inside `cbe_expand`/`_record_krylov!`, i.e. per bond). Driver writes `*_bonds.csv` (one row per run, step, visit); analyzer "Per bond" table. Legacy path: `bondlg = nothing`, `_log_bond!` no-op, `cbe_expand` kwargs unchanged. `test_adaptive_diagnostics.jl` (83/83, job 17198444): L visits per step, every step counter = sum/max of the bond entries, widths = built frames, failed bonds named, midpoint stages tagged. Driver smoke wrote `_bonds.csv` (688 rows); analyzer table mirror-symmetric on XX/Neel as it must be. |
| R.2 | §7 "Use both Float64 and complex state tensors where applicable" | `product_state`/`neel_state`/`domain_wall_state(...; eltype = ComplexF64)`; `test_adaptive_diagnostics.jl` runs a Float64 Neel through ordinary and midpoint, real and imaginary time, against its complex twin, and the RSVD path against free fermions | DONE | The step itself works in `ComplexF64` (`tau` is promoted), so "where applicable" = the START state the step receives. Default unchanged: every existing caller still gets complex tensors. 83/83 (job 17198444): Float64 Neel, ordinary + midpoint, real + imaginary time, all `:met`, Sz within 1e-6 of the complex twin; RSVD path from a real state within 1e-4 of free fermions. |
| R.3 | §1 old names: deprecation warning with migration hint (row 1.5) | `_legacy_hint(legacy_names)` right after `resolve_budget` in `cbe_bug_step!`: one `@warn` per session (`maxlog=1`) naming the knobs and the four adaptive keywords | DONE | Informs, does not refuse: the legacy path is a supported study arm. Test asserts the message, the public-entry warning, and no warning on an adaptive step. 83/83 (job 17198444): message, public-entry warning, no warning on an adaptive step. |

## `simulation_plan.md` checked line by line (2026-10-15). Audited before submission; job 17199363

Its tasks, UT1-UT8 and IT1-IT5 are the main plan's §3-§5, UT03-UT14 and IT01-IT07 (rows above).
What it says that the main plan does not:

| # | Requirement (simulation_plan §) | Evidence | Status | Note |
|---|---|---|---|---|
| S.1 | §1 signature shows `adaptive = true` | no such keyword: the path is inferred from whether a retired knob is named (`resolve_budget`) | DIVERGES (by decision) | The main plan's §1 signature (the governing plan) has no flag, and §1 forbids silently honouring two configurations; a mode flag is one more knob that can disagree with the knobs passed. Stated in `adaptive_budget.jl` ("NO `adaptive = true` FLAG, DELIBERATELY"). |
| S.2 | §5 per bond AND timestep: `n_reachable_sectors`, `n_rsvd_samples`, `n_rsvd_selected`, `n_krylov_added`, `r_old`, `r_augmented`, `r_final`, sampling residual, retained tail, Krylov residual, `discarded_weight`, `n_hamiltonian_actions` | `bond_log` entries (R.1) gain `n_sectors` + `probes_per_sector` (final Test B sketch per side, `_log_bond!`), and after compression `r_old`, `r_final`, `discarded` (that bond's relative discarded norm, from `truncate_recursive!(...; bond_weights)`); `width` is `r_augmented` | DONE | Test: per-bond `r_old`/`r_final` equal the state's bond dims before/after; the per-bond discarded norms rebuild the step's cumulative estimate `sqrt(Σ_left d²) + sqrt(Σ_right d²)` to 1e-10 at `tol_trunc = 1e-3` (non-vacuous); sampled bonds report >= 1 sector. CSV/analyzer columns added. Job 17199363: `test_adaptive_diagnostics.jl` + `test_adaptive_compression.jl` 170/170 (5m24s); driver smoke writes the new columns. |

## Open items, by when they must close

- **Before Stage 3:** none of Stage 1/2 scope. **Stages 1-2 are closed** (job 17190027): UT01-02
  70/70; UT03-06 plus end-to-end 225/225; full `tests/sweeps` 939/2/2. Its reds are exactly the
  pre-existing four (`test_cbe_bug_exact_refs.jl:310`, `test_physics.jl:82,116,153`, also red on
  baseline 17169504), so there is no regression.
- **Final regression (job 17195549, all code final):** full `tests/sweeps` 1103/2/2 -- ONLY the 4 baseline reds (`exact_refs:310`, `physics:82,116,153`).
- **Stage 3 closed** (job 17192249, see 3.10). 1.6 `krylov_tol` -> closed by 3.4.  **Stage 4**: 1.12 and 2.13 -> closed by 4.1/4.2.
  **Stage 5**: 1.6/1.7 -> closed by 5.1/5.4. **Stages 4 and 5 closed** (17193333 + rerun 17195151).  **Stage 6 closed** (6.9 by matched-accuracy job 17195550). 2.8 closed (IT03).
- **Robustness of the Stage 6 conclusions (2026-10-15, jobs 17197997-99):** quickstart example
  runs as written (QUICKSTART-OK); per-component sweeps re-run on isotropic Heisenberg (exact
  sparse reference) and on XX at dt = 0.1 -- the looseness rules hold, see finding 9 in
  `ADAPTIVE_XX_L20_RESULTS.md`.
- **Found by a line-by-line re-read of §7/§8:** per-bond tracking (§8) and Float64 states (§7)
  were missing; added as rows R.1/R.2 with the row-1.5 hint as R.3.
- **R.1-R.3 closed (job 17198444):** new tests 83/83; full `tests/sweeps` 1186/2/2 = the 4 baseline
  reds only (`exact_refs:310`, `physics:82,116,153`); MaxRSS 11.4 GB (cold precompile).
- **The four standing reds are pre-existing and unrelated, shown by value, not by count.** In
  the baseline job 17169504 (2026-10-13), before any of this work changed physics, they are the same
  four with the same numbers: `exact_refs:310` OAT rank `12 == 7`; `physics:153` Haldane-Shastry
  magnon `0.0042542611127878716` (job 17198444: `0.004254261112787816`; also measured 4.2543e-03 on
  2026-08-25, `docs/PLAN-tolerance-control.md` §7); `physics:82,116` `UndefVarError: jan_bug_step!`
  (deleted in 3024066). All four run the LEGACY path (`trunc_thresh` named) or a deleted function.
  User decision 2026-10-15: not in scope (Haldane-Shastry skipped); run only the relevant test files.
- **Job 17199914 (2026-10-15), edited files only:** HS physics gated behind `BUGJULIA_TEST_HS=1`
  (user: skip Haldane-Shastry), dead `jan_bug_step!` arms removed, OAT rank `@test_broken`.
  90 pass, 1 broken, 2 FAIL: the Heisenberg-RING magnon arm, which never ran before (the
  `jan_bug_step!` error aborted the testset first). It fails for `cbe_bug` (5.0e-3) AND for
  `tdvp_cbe1s` (2.7e-3) -- TDVP reaches 2.6e-8 on the HS magnon -- so it is not the adaptive or
  BUG code (the reference was re-read and is right: `E_F = L/4` for the NN ring). User decision
  2026-10-15: skip it -- opt-in via `BUGJULIA_TEST_RING_MAGNON=1`. Job 17200031: `test_physics.jl`
  18/18. With 17199914 (`exact_refs`: all pass, OAT rank broken), both previously red files pass.
- Kept by design: `_range_find` stops on a
  global count, and Test B covers the gap that leaves.
