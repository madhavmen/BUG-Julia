# Plan audit: general-order Krylov-closure CBE-BUG (`krylov_closure_cbe_bug_realtime_L16_plan.md`)

One row per plan requirement, checked BY CALL SITE (the code on the production path,
`cbe_bug_step!(...; order = p)`), with the test that calls that code. Status:

- `DONE` -- on the production path AND exercised by a test that calls that code
- `PARTIAL` -- implemented, a named piece missing or untested
- `DIVERGES` -- the code does something other than the plan says (with the reason)
- `PENDING` -- implemented, validation job not yet back

Code: `src/RSVDCBEBondUpdate/sweeps/krylov_closure.jl` (new), `sweeps/cbe_bug.jl`
(`order` keyword, closure call before the root solve, `_krylov_enrich(...; amp)`),
`adaptive_budget.jl` (`basis_allocation(...; nfront)`, ledger fields). Tests:
`tests/sweeps/test_krylov_closure.jl`. Integration: `benchmarks/krylov_closure_l16.jl`,
`benchmarks/analyze_krylov_closure.py`.

## Design decisions the plan leaves open (stated so they can be disagreed with)

- **Frontiers are scaled raw powers** `F_j = (tau^j/j!) B^j S0`, not orthonormal Arnoldi vectors.
  The time weight then lives in the vector: measuring `F_j` with the step's `|tau|/||psi||` scale
  gives frontier `j`'s contribution to the local error (variation of constants: `sum_j
  tau/(j+1) (I-P) A F_j`). For `p <= 4` the raw powers are well conditioned.
- **The outside-space action is measured exactly per bond, not sampled.** The frontier's
  two-site action `H Theta` is the object CBE builds anyway (shared through `_sketch_closures`),
  so `||(I - U0 U0') H Theta||` costs no extra contraction. For a nearest-neighbour `H` the
  per-bond numbers are exactly the orthogonal telescoping pieces of `(I - P_L) H F` (proof in
  `estimate_global_complement_action`'s docstring), so `max(left, right) <= ||(I-P) H F|| <=
  sqrt(left^2 + right^2)`. The sampled estimate (`nsample > 0`, Test B's estimator on the full
  opposite local space) exists for diagnostics and UT2.
- **A bond within its share is skipped** (no CBE, no Krylov) -- the gate is the bond's whole
  allowance `alloc.per_bond`, while the enrichment targets the per-test shares, so an enriched
  bond lands under the gate and the confirming pass does not ping-pong.
- **The ordinary half-sweeps are not re-routed through the frontier code.** `order = 1` is the
  existing step bit for bit (UT9); the frontier sweeps duplicate the half-sweep LOOP but share
  every kernel (`cbe_expand`, `_krylov_enrich`, `_record_krylov!`, `_log_bond!`, environment
  pushes, `_relink`).
- **`tol_basis` is split over the `p` frontiers** (`per_bond = tol_basis / sqrt(n_bonds * p)`),
  so the integrated defect `sum_j eta_j/(j+1) <= (H_p/sqrt(p)) tol_basis ~ 1.05 tol_basis`.
  `p = 1` is unchanged.
- **The midpoint step refuses `order > 1`** (it is its own second-order construction).
- **Check first, enrich only if needed** (revision 2026-10-15, after review: the ordinary step's
  CBE + local-Krylov frames may already contain the global Krylov directions, since
  `P_i H^2 K = P_i H P_i H K + P_i H (I - P_i) H K` and CBE may already have made the second
  term negligible). `_krylov_closure!` now first MEASURES every frontier `j = 0..p-1` on the
  incoming half-sweep frames (`_measure_frontiers`, `ledger.leak_initial` / `rho_initial`, with
  `rho_j = ||(I-P) H F_j|| / ||F_j||`). If every bond of every frontier is already within its
  share, no closure sweep runs (`n_passes = 0`) and the frames are those of the `order = 1`
  step. The closed case costs `p` measurement sweeps, as before; the open case costs `p` extra
  measurement sweeps.
- **`closure_probe = m`** (instrumentation, not a method choice) runs the same measurement on
  an `order = 1` step without changing the evolved state (copied `rng`, frames untouched; UT9
  checks bit identity and that it equals what an order-3 closure measures before enriching).
  The `rho` phase of `benchmarks/krylov_closure_l16.jl` uses it to measure `rho_0..rho_3` of
  the unchanged solver at L = 16 next to its dt convergence. It runs BEFORE any closure claim.

## §1 Interface and invariants

| # | Requirement | Evidence (call site) | Status | Note |
|---|---|---|---|---|
| 1.1 | `order` keyword; only new user-facing method choice; three tolerances + `maxdim` unchanged | `cbe_bug_step!(...; order::Int = 1)`; not in `LEGACY_KNOBS`; no other new METHOD keyword | DONE | UT9. `closure_probe` is an added instrumentation keyword (measures only; state bit-identical, UT9) -- a deliberate exception, listed under design decisions |
| 1.2 | No new accuracy knobs (`n_passes`, per-frontier tolerances, RSVD sizes not exposed) | `_closure_maxpass(order)` internal; gate and shares derived from `basis_allocation(...; nfront = order)` | DONE | |
| 1.3 | Internal allocations may depend on order/dt | `basis_allocation(budget, L-1; tau, nrm, nfront = order)` in `cbe_bug_step!` | DONE | UT9: `nfront = 1` identical (`===`), `nfront = 4` halves `per_bond` |
| 1.4 | Preserve original subspace and U(1) labels | `U_ex = [U0 | new]` with `U0 = Pm W_old[i]` in `_frontier_left/_right` | DONE | UT5 nesting at every bond, both sides |
| 1.5 | K/L sweeps remain basis constructors; ONE final full-step root exponential | the closure only grows `W`, `Z`; the single `expv` after it in `cbe_bug_step!` | DONE | |
| 1.6 | `:budget_unmet` if order/accuracy cannot fit the working-rank cap | `_unmet!(lg, :closure)` when the loop does not converge; CBE/Krylov `:tail/:sample/:krylov` + `workcap_reached` from frontier sweeps merged into the step ledger | DONE | UT7 |

## §2 Order target

| # | Requirement | Evidence | Status | Note |
|---|---|---|---|---|
| 2.1 | `P psi = psi` | frames start from the half-sweep frames, which contain psi; `ledger.proj_err` | DONE | UT5 carries |
| 2.2 | frontiers `v_j` projected at the root, embedded with `W`, `Z` | `_projected_frontiers` (zero-site applications), `_frontier_mps` | DONE | UT1, UT5 |
| 2.3 | outside-space action `d_j = (I - P) A v_j`, j = 0..p-1 | j = 0 by the ordinary half-sweeps (and MEASURED on the final frames by `_krylov_closure!`), j >= 1 by `_frontier_sweep!` | DONE | UT2 (exact vs matrix-free) |
| 2.4 | time-weighted leakage / integrated defect | `ledger.leak[j]` = `|tau| sqrt(l^2 + r^2)/||psi||` of `F_j` on the final frames; `ledger.defect = sum_j leak[j]/(j+1)` | DONE | integrated defect via the triangle bound, not quadrature nodes (plan: "and/or") |
| 2.5 | a zero first-root residual is not sufficient | every frontier is judged by its FULL-space action; a projected breakdown is irrelevant | DONE | UT4 (dense and MPS) |
| 2.6 | recompute root operator/frontiers whenever frames change | `_krylov_closure!`: `H0`, `S0` rebuilt after every enriching sweep; `KrylovFrontier.gen` refused when stale | DONE | UT5 stale-frontier refusal |

## §3 Implementation sequence

| # | Requirement | Evidence | Status | Note |
|---|---|---|---|---|
| 3.1 | `KrylovFrontier` with root coefficients, sector metadata, environment references; CBE / K/L helpers act on an arbitrary frontier MPS; `order = 1` compatible | `KrylovFrontier(core, degree, gen)`, `KLFrames` (frames + root environments); `_frame_onto_left/_right` feed `cbe_expand` / `_krylov_enrich` a frontier's carry with the CURRENT frame prescribed | DONE | `_krylov_enrich(...; amp)` scales a small frontier's Tests C/D to the state norm (default 1.0: bit-identical) |
| 3.2 | `estimate_global_complement_action` with matrix-free contractions; exact reference only in tests; Tests A/B reused per frontier | `estimate_global_complement_action` (non-enriching frontier sweep); enrichment through `cbe_expand` (Tests A/B) on every enriched bond | DONE | UT2 compares with the dense `(I - P) H v` |
| 3.3 | repeated K/L sweeps; append only validated independent directions; reorthogonalise; transport carries; rebuild envs and frontiers; termination; do not propagate oversampling | `_closure_loop!` + `_frontier_sweep!`; `_reortho_*` inside `cbe_expand`/`_extend_frame`; `Pm`, `C`/`D` carries; envs pushed through the new frames; work ceiling; `maxpass` reported | DONE | UT1, UT5, UT7 |
| 3.4 | project ORIGINAL psi into final frames; one full-step `expv` at `tol_expv`; lossless root split; one `root_out` compression | `S0 = _project_root(psi, fr)`; existing root/assembly/`truncate_recursive!` unchanged; `ledger.n_compress` | DONE | UT8 |
| 3.5 | instrumentation per sweep/bond | `ledger.closure_log` (pass, degree, leak before, added, widths, new sectors, skipped, counts, time), bond entries tagged `pass/frontier/leak/skipped/added/new_sectors/width_before`, `n_root_frontier`, `n_h1_closure`, `n_h2_leak`, `t_closure`, `n_compress`; benchmark adds wall time and allocations per step | DONE | |

## §4 Unit tests

Job 17207253 (code with check-first + `closure_probe`): closure file **210/210 pass**
(EXIT-CLOSURE=0); regression (`test_adaptive_api/krylov/diagnostics/compression`) **312/312**
(EXIT-REGRESSION=0).

| # | Test | Where | Status |
|---|---|---|---|
| UT1 | frontier advances, no redundant columns | dense hopping chain (production `_closure_loop!`), single magnon XX L=6 (production MPS closure) | DONE -- magnon held at sites 1..p+1 to 1e-8, one addition per frontier in pass 1, none in the confirming pass |
| UT2 | outside-space action vs exact, left and right | XX L=6 (rank-capped warm-up), own Schmidt frames at c = 5 / 1 / 3; sampled estimate unbiased per bond | DONE -- one-sided cases match dense to 1e-15 rel.; c = 3: exact 1.00885 vs bound 1.00885, max(l, r) 0.713 (core 1). Job 17204568's c = 3 failure was a fixture defect (full-rank frames span the sector), fixed |
| UT3 | order moments | dense n=24, p = 1..4 (moments, next moment fails, slope p+1); MPS DW L=10 local error slope | DONE -- MPS local-error slopes 2.00 / 3.00 / 4.00 for p = 1 / 2 / 3 (target p+1): from a domain wall the ORDINARY step is first order, the closure supplies the order |
| UT4 | projected breakdown not accepted | dense invariant-under-B space; MPS 1-dim root space (Heisenberg DW), XX zero-frontier case | DONE |
| UT5 | frame invariants after every pass | XX L=8, RSVD path, 5 frontier sweeps: nesting, isometry, carries, envs, stale refusal | DONE |
| UT6 | no growth on eigenstates; exact root phase | all-up XX (H psi = 0) and XXZ (H psi = E psi), order 4 | DONE -- closed at the check (`n_passes = 0`), `leak_initial == leak` |
| UT7 | undersized maxdim | Neel L=10, maxdim 2, order 4: `:budget_unmet`, bounded widths and passes | DONE -- unmet `krylov+closure+trunc`, r_aug 4, diagnostic names all three |
| UT8 | one root_out compression, cumulative vs worst | warm XX L=10, order 3, tol_trunc 1e-3 | DONE |
| UT9 | order = 1 reproduces the ordinary step; `closure_probe` | bit-identical ordinary and midpoint; probe bit-identical and equal to order-3's pre-closure measurement; allocation `===`; refusals | DONE -- warm XX L=8, dt 0.05: rho_0..2 ~ 1e-15 on the order-1 frames, order 3 ran 0 passes (near-complete frames at L = 8; not evidence for L = 16) |
| stress | multi-seed | orders 2, 3: exact CBE + 3 RSVD seeds against `exp(-iHt)` | DONE -- seeds agree with the exact arm to 1e-15 abs. (4.48e-8 / 4.44e-8) |

## §5 L=16 integration (IT1-IT5)

`benchmarks/krylov_closure_l16.jl` phases `it1`..`it5`, graded against `heisenberg_sparse(16;
delta = 0)` + `expv_sparse` in the 12870-state sector, occupations cross-checked against free
fermions; the MPS is read out in the sector basis (complex wavefunction, raw and phase-aligned).
A `check` phase (run before every phase) aborts on any convention mismatch.

| # | Test | Status |
|---|---|---|
| rho | unchanged order-1 step: `rho_0..rho_3` on its own frames vs observed dt order (tight and default tolerances) | DONE (job 17207911, see below) |

**rho result (job 17207911, L = 16 XX domain wall, T = 0.8; first run 17207253 stopped at a
convention check whose 1e-8 threshold sat below order 2's own local error 2.9e-8 -- fixed to
"< 1e-6 and 1000x below the reversed propagator", which gave 2.9e-8 vs 1.0e-2):**

- Global error 7.22e-3 / 1.78e-3 / 4.43e-4 / 1.11e-4 at dt 0.2 / 0.1 / 0.05 / 0.025 (slope 2.0),
  IDENTICAL at tight (1e-12) and default (1e-8) tolerances, so it is time discretisation.
- **98-100 % of it is the FIRST step.** The order-1 frames out of the product state have rank 2
  (psi and H psi): rho_0 = 2.5e-16, but **rho_1 = 0.71** at every dt. The first-step error equals
  the defect estimate `leak_1/2` (7.07e-3 predicted, 7.06e-3 measured at dt = 0.2; leak_1 scales
  as dt^2). The first step's rho_2 = 1e-16 is a projected breakdown, not closure: <DW|H|DW> = 0,
  so B^2 S0 is parallel to S0 in the 2-dimensional root space. Only rho_1 detects the gap (UT4's case).
- Later steps: rho_0 <= 1e-11 (tight) / 1e-7 (default). rho_1 has median 1e-9 (tight, dt <= 0.05)
  and 2e-6 to 4e-6 (default); its max falls ~linearly with dt (0.07 to 0.009). rho_2 and rho_3
  stay open (max 0.41 / 0.55; medians 1e-8 and 1e-6 tight). Time-weighted, later leak_1 <= 2.8e-6
  at dt = 0.025.
- Verdict for the review's three outcomes: **outcome 2 at the first step** (CBE captures H psi,
  not H^2 psi), and close to outcome 1 for j = 1 afterwards. The closure is needed where the
  frames are poor (low entanglement), and check-first keeps it cheap elsewhere. The observed slope 2
  of "order 1" is the one-off O(dt^2) first-step error, not a second-order method.
| IT1 | exact state t = 0.5/1/2, orders 1/2/4 | DONE (17208037) -- raw wf error at dt 0.05, default tolerances: order 1 4.43e-4 and order 2 3.68e-6, CONSTANT from t = 0.5 to 2 (first-step error, later steps add nothing visible); order 4 4.4e-9 to 1.7e-8 (grows with t: the tol_basis 1e-8 floor). max dSz order 4 3.5e-11 at t = 2. All steps met |
| IT2 | temporal convergence T = 0.8, dt 0.2..0.025, orders 1-4 (+ midpoint calibration) | DONE (17208038) -- raw wf error slopes **2.0 / 3.0 / 4.0 / 5.0 (4.93 last)** for p = 1..4; order 4 reaches 3.8e-12 at dt 0.025. Slope p+1 at fixed T is the one-off first-step local error O(dt^(p+1)) (shown directly for p = 1 by the rho phase, and by IT1's t-independent error for p = 1, 2), NOT the generic global order p. Midpoint = order 2 to 7 digits at dt <= 0.1 although the runs differ (r_aug, n_h1): both solve the first step in the same Krylov space span{psi, H psi, H^2 psi}. Closure defect at the final step <= 2.6e-13; passes 0-2 |
| IT3 | conservation and n_j(t) to t = 2 | DONE (17208039) -- norm 3.9e-14, energy 5.9e-15, N_up 5.8e-15, isometry 4.2e-15; 40/40 steps met (orders 2, 4) |
| IT4 | maxdim 32/64/128 rank adaptation, T = 4 | DONE (17208040) -- md64/128: chi 43-45 < cap, D_work 57, 80/80 met, identical results (cap not binding). md32: cap binds, 13-14 of 80 steps `:budget_unmet (trunc)`, discarded 1.3e-9, error unchanged (3.7e-6 order 2) -- reported, not silent. Closure directions added over 80 steps: 12 (order 2), 46 (order 4) |

⚠ The phase-aligned error in IT1-IT4 (first runs) came from `sqrt(|v|^2 + |vex|^2 - 2|<vex,v>|)`,
which cancels to a ~1e-8 floor (it printed 0 and non-monotone values for order 4). Use the raw
error, which is an upper bound on it. Fixed in `wf_grade` (rotate, then subtract) before IT5.
| IT5 | work at matched accuracy, one process | DONE (17208119) -- at ~17.5 s wall: order 4 (dt 0.2) 1.2e-7 vs midpoint (dt 0.05) 3.7e-6 vs order 1 (dt 0.05) 4.4e-4; order 2 = midpoint accuracy at 1.5-1.8x its cost; order 4 below dt 0.2 sits on the tol_basis floor; closure 55-72 % of order-4 wall, and the check-first measurement alone is 34 % of order 2 at dt 0.05. Details: `KRYLOV_CLOSURE_L16_RESULTS.md` |
