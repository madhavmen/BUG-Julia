# General-order Krylov-closure CBE-BUG: XX L = 16 domain-wall results

Plan: `krylov_closure_cbe_bug_realtime_L16_plan.md` (§5, IT1-IT5), plus the `rho` diagnostic
added after review. Requirement-by-requirement evidence: `KRYLOV_CLOSURE_AUDIT.md`.

Model: `H = (1/2) sum_j (S+_j S-_{j+1} + h.c.)`, open, L = 16, U(1), start state
`|up^8 down^8>`. Every number is graded against a sparse exact reference in the N_up = 8 sector
(12870 states, `expv_sparse` at 1e-13), with occupations cross-checked against free fermions.
The comparison is on the full complex wavefunction (raw error, global phase included). A
convention check runs before every phase (readout 0, sparse vs free fermions 7.9e-15, one
order-2 step 2.9e-8 vs 1.0e-2 for the reversed propagator).

Driver `benchmarks/krylov_closure_l16.jl`, tables `benchmarks/analyze_krylov_closure.py`.
Jobs (2026-10-15, `$SCRATCH/rsvd_adaptive/2026-10-15-klclose/`, code `~/BUG-Julia-klclose`, files
sha256-identical to the local working tree): unit tests 17207253, rho 17207911, IT1-IT4
17208037-40, IT5 17208119. Every job: 2 threads, 1.0 G MaxRSS (warm cache), no OOM.
Figures: `benchmarks/results/krylov_closure_l16/`.

## 1. What the unchanged order-1 step already captures (rho phase)

Question from review: does the ordinary CBE + local-Krylov step already contain the global
Krylov directions, i.e. is `rho_j = ||(I-P) H F_j|| / ||F_j||` small on its own frames?
`closure_probe = 4` measures this without changing the evolved state (UT9: bit-identical).

| dt | error at T = 0.8 | share from step 1 | step 1: rho_0 / rho_1 | steps >= 2: max rho_1 | all steps: median rho_1 (tight) |
|---|---|---|---|---|---|
| 0.2 | 7.22e-3 | 98 % | 2.5e-16 / 0.71 | 7.0e-2 | 7.0e-2 |
| 0.1 | 1.78e-3 | 99 % | 2.5e-16 / 0.71 | 3.5e-2 | 2.3e-4 |
| 0.05 | 4.43e-4 | 100 % | 2.5e-16 / 0.71 | 1.8e-2 | 3.4e-10 |
| 0.025 | 1.11e-4 | 100 % | 2.5e-16 / 0.71 | 8.8e-3 | 9.9e-10 |

- The error is identical at tight (tol_basis 1e-12) and default (1e-8) tolerances, so it is
  time discretisation, not basis truncation.
- **Out of the product state the order-1 frames have rank 2**: they contain psi and H psi
  (rho_0 = 1e-16) but not H^2 psi (rho_1 = 0.71 at every dt). That step's error equals the
  defect estimate `leak_1/2` (7.07e-3 predicted, 7.06e-3 measured at dt = 0.2), and it is
  essentially all of the error at T.
- Step 1's rho_2 = 1e-16 is not closure. `<DW|H|DW> = 0` puts `B^2 S0` parallel to `S0` in the
  2-dimensional root space, so only rho_1 shows the gap. This is why every frontier is tested
  against the full H (UT4).
- Once the state is entangled, the ordinary frames nearly close j = 1 (median rho_1 ~ 1e-9 at
  tight tolerances, dt <= 0.05). rho_2 and rho_3 stay open in the worst step (0.41 and 0.55),
  but they enter weighted by dt^2 and dt^3.

**Verdict.** Outcome 2 of the review at the first step: CBE captures the first Hamiltonian
action, not the second. Afterwards the j = 1 behaviour is close to outcome 1. The apparent
slope 2 of order 1 is the one-off O(dt^2) first-step error, not a second-order method. The
closure is needed where the frames are poor. Because `_krylov_closure!` measures before it
enriches, closed steps run no closure sweep (`n_passes = 0`).

## 2. Temporal convergence (IT2, T = 0.8, tol_basis 1e-12, tol_expv 1e-13, maxdim 512)

| arm | dt = 0.2 | 0.1 | 0.05 | 0.025 | slopes |
|---|---|---|---|---|---|
| order 1 | 7.22e-3 | 1.78e-3 | 4.43e-4 | 1.11e-4 | 2.02 / 2.01 / 2.00 |
| order 2 | 2.35e-4 | 2.95e-5 | 3.68e-6 | 4.60e-7 | 3.00 / 3.00 / 3.00 |
| order 3 | 1.02e-5 | 6.38e-7 | 3.99e-8 | 2.49e-9 | 4.00 / 4.00 / 4.00 |
| order 4 | 1.18e-7 | 3.68e-9 | 1.15e-10 | 3.78e-12 | 5.00 / 5.00 / 4.93 |
| midpoint | 2.35e-4 | 2.95e-5 | 3.68e-6 | 4.60e-7 | 3.00 / 3.00 / 3.00 |

![IT2 convergence](benchmarks/results/krylov_closure_l16/it2_convergence.png)

- **Each order buys exactly one power of dt.** The slopes are p+1, not the textbook global p,
  because the error on this problem is the first step's local error O(dt^(p+1)), made once.
  This is shown directly for p = 1 (section 1) and for p = 1 and 2 by IT1, whose error does not
  change from t = 0.5 to t = 2. It is not checked separately for p = 3 and 4.
- Order 4 at dt = 0.025 (3.8e-12) is close to the tolerance floor (tol_basis 1e-12, tol_expv
  1e-13). The last slope of 4.93 is the first sign of it.
- The midpoint step equals order 2 to 7 digits for dt <= 0.1, although the runs differ (rank 28
  vs 30, 98 vs 186 one-site applications at dt = 0.05). Both solve the decisive first step in the
  same space span{psi, H psi, H^2 psi}.
- Closure defect at the final step <= 2.6e-13 on every arm; every step met its tolerances.

## 3. Long-time accuracy, conservation, rank (IT1, IT3, IT4; dt = 0.05, default tolerances)

IT1, raw wavefunction error against exact:

| arm | t = 0.5 | 1 | 1.5 | 2 | max dSz at t = 2 |
|---|---|---|---|---|---|
| order 1 | 4.43e-4 | 4.43e-4 | 4.43e-4 | 4.43e-4 | 2.6e-4 |
| order 2 | 3.68e-6 | 3.68e-6 | 3.68e-6 | 3.68e-6 | 1.2e-6 |
| order 4 | 4.4e-9 | 8.1e-9 | 1.4e-8 | 1.7e-8 | 3.5e-11 |

Orders 1 and 2 carry their first-step error unchanged to t = 2. Order 4 sits on the tol_basis =
1e-8 floor and grows slowly with it.

IT3 (orders 2 and 4, 40 steps each): norm drift 3.9e-14, energy 5.9e-15, N_up 5.8e-15,
isometry defect 4.2e-15, every step `:met`. The front `n_j(t)` follows the exact profile
(`it3_front_order*.png`).

IT4 (T = 4): with maxdim 64 or 128 the cap never binds (chi 43-45, working rank 57), so the
results are identical and all 80/80 steps are met. With maxdim 32 the cap binds:
13-14 of 80 steps report `:budget_unmet (trunc)`, the discarded weight is 1.3e-9, and the error
is unchanged (3.7e-6 for order 2). The steps report the cap rather than passing silently. Over
80 steps the closure added 12 directions (order 2) and 46 (order 4), mostly at the start.

## 4. Work at matched accuracy (IT5, ONE process, T = 2, default tolerances)

| dt | order 1 | midpoint | order 2 | order 3 | order 4 |
|---|---|---|---|---|---|
| 0.2 | 7.2e-3 / 4.5 s | 2.4e-4 / 4.8 s | 2.4e-4 / 8.4 s | 1.0e-5 / 14.1 s | **1.2e-7 / 17.6 s** |
| 0.1 | 1.8e-3 / 8.5 s | 2.9e-5 / 9.2 s | 2.9e-5 / 14.3 s | 6.4e-7 / 22.6 s | 1.1e-8 / 27.0 s |
| 0.05 | 4.4e-4 / 17.1 s | 3.7e-6 / 17.3 s | 3.7e-6 / 26.3 s | 4.6e-8 / 32.5 s | 1.7e-8 / 40.3 s |

Entries are wavefunction error / wall time.

- **At equal wall time (~17.5 s), order 4 at dt = 0.2 is 31x more accurate than the midpoint
  step at dt = 0.05 (1.2e-7 vs 3.7e-6), and 3700x more accurate than order 1.** All of these
  are measured; none is extrapolated.
- Order 3 at dt = 0.1 is 6x more accurate than midpoint at dt = 0.05 for 1.3x the time.
- At the same error, **order 2 costs 1.5-1.8x the midpoint step**. The midpoint is the better
  second-order method here.
- Order 4 below dt = 0.2 only adds steps: 1.1e-8 and 1.7e-8 are the tol_basis = 1e-8 floor.
  At a fixed tolerance the useful dt is the largest one the order allows.
- The closure is 55-72 % of order 4's wall time. Checking first is not free: order 2 at
  dt = 0.05 runs 0.1 enriching passes per step, yet spends 8.9 of 26.3 s in the closure, nearly
  all of it in the p measurement sweeps. Measuring j = 0 again on unchanged frames is
  redundant with the half-sweeps' own test, so it is the obvious saving (follow-up, not
  implemented).

## 5. Limits of these results

- One model and one start state. On this domain wall the first step dominates, which favours
  high order. A state whose later steps carry the error would show the global order p and
  different matched-accuracy ratios.
- For a nearest-neighbour H the `rho` values are upper bounds `sqrt(left^2 + right^2)` built
  from exact per-bond pieces (UT2 checks them against dense `(I-P) H v`). For longer-range H
  they are estimates.
- The phase-aligned error printed by the first IT1-IT4 runs used a cancelling formula with a
  1e-8 floor. Only raw errors are quoted here. The formula is fixed (`wf_grade`) and IT5 used
  the fixed one.
