# Adaptive CBE-BUG on the XX chain, L = 20: what each tolerance controls

Stage 6 of `adaptive_cbe_krylov_implementation_plan_first_test.md` (§8, IT01-IT07) and of
`simulation_plan.md` §4. Open XX chain, U(1), J = 1, dt = 0.05, maxdim = 128, `parallel = true`.
Every number is graded against the EXACT free-fermion occupations (`tests/common/free_fermion.jl`):
`err = max_i |<Sz_i>(t) - exact|`. Driver `benchmarks/adaptive_xx_l20.jl`, tables by
`benchmarks/analyze_adaptive_xx.py`. Jobs (2026-10-14, `$SCRATCH/rsvd_adaptive/2026-10-14-l20/`):
17195152 profiles, 17195153 sweeps, 17195154 baseline (T = 0.5); 17195155 sweeps, 17195335 combos,
17195336 maxdim (T = 2). Code validated by UT01-UT15 (jobs 17193333, 17195151).

## Findings

**1. The RSVD's randomisation costs no accuracy here -- and no time either, it costs MORE.**
The `exact` arm (complete local basis, no random sketch, `exact = true`) reproduces the RSVD arm
to every printed digit at every `tol_basis`, at T = 0.5 and T = 2 (chi up to 128), and five RNG
seeds agree to <= 5e-10. Test B's certified sketch therefore always recovers the full CBE range
on this model. In the SAME job (comparable clocks) the RSVD arm spent ~1.6x the CBE time of the
exact arm (T = 2: 59 s vs 36 s of CBE time at tol_basis 1e-6): with `d*chi <= 256` the exact local basis is
cheap, and certifying a sketch (holdout probes, targeted widening) is overhead. The sketch can
only pay off where `d*chi` is large; on XX L=20 `exact = true` is the faster equally-accurate
choice.

WHERE THE CROSSOVER IS (job 17195922: Neel, T = 4, maxdim 256 and 512, RSVD vs exact in ONE
process; errors identical at every step). Per-step CBE time, RSVD / exact:

| chi | 29 | 52-83 | 120-185 | 224-256 | 281-398 |
|---|---|---|---|---|---|
| t_cbe ratio | 1.7-1.9 | 1.3-1.7 | 1.1-2.4 (noisy) | 0.85-1.0 | 1.0-1.4 |

The sketch's overhead falls steadily with chi and reaches PARITY around chi ~ 250-400 (`d*chi` ~
500-800); at chi = 256 the RSVD arm was slightly faster (0.85-0.92). Over the whole run it is
still 1.1-1.5x slower (most steps sit at small chi). The sketch draws about half the columns of
the exact local basis; its overhead is the Test B certification (holdout probes, re-sketching),
whose relative cost shrinks as the exact basis grows like `d*chi`. So: on spin-1/2 chains use
`exact = true` below chi ~ 250; the RSVD is the choice for larger chi or larger local dimension.

**2. `tol_basis` (RSVD Tests A/B + Krylov Tests C/D, i.e. the trial space) is the accuracy knob.**
The error tracks it over five decades: err(T = 0.5) ~ 0.07-0.15 x tol_basis, err(T = 2) ~
0.2-0.9 x tol_basis, until the time-step floor of the scheme (ordinary BUG: 2e-7 at T = 0.5,
4.7e-6 at T = 2; midpoint: 1.6e-9 / 9e-9). The fixed allocation rule is CONSERVATIVE (the
measured error stays below tol_basis), not over-spent. It is also the main COST driver of the
half-sweeps: H applications, sketch probes and Krylov directions all grow with it.

**3. `tol_trunc` (final compression) can be loose -- it is the speed knob.** At T = 2 tol_trunc
= 1e-4 already reaches the scheme's floor (5.1e-6) with a final rank of 12 instead of 128, and the
run is 3x faster than at 1e-10 (45 s vs 141 s). err ~ 0.05-0.5 x tol_trunc above the floor.
Tight tol_trunc with tight tol_basis is what drives chi into `maxdim`.

**4. `tol_expv` (root Galerkin solve) does not matter at dt = 0.05.** 1e-2 gives 6.2e-6 vs 4.7e-6
at 1e-12 (T = 2) with 2.5x fewer root iterations; the root solve is < 2% of the step time. Its
only visible side effect: a sloppy root solve leaves more rank to keep (chi 59 vs 45 at T = 0.5).

**5. The BUG integrator is the floor.** Ordinary vs midpoint is the one choice no tolerance
substitutes for: midpoint reaches 1e-9 where the ordinary step stops at its O(dt) floor, at ~2x
the cost per step.

**6. Loose settings combine roughly additively (mild compounding).** E.g. ordinary, T = 2:
(tol_basis, tol_trunc, tol_expv) = (1e-5, 1e-3, 1e-2) gives 6.7e-5 at chi 8; (1e-4, 1e-5, 1e-2)
gives 3.0e-5 at chi 46; (1e-4, 1e-3, 1e-2) gives 2.2e-4 -- ~2.5x the sum of the individual
errors. Every combination stayed `:met`.

**7. Failures are reported.** Every run whose rank hit `maxdim` (the tight T = 2 sweeps, and the
maxdim = 4/8/16 phase) ends `:budget_unmet` with the reason (`trunc`, plus `krylov`/`tail` when
the work ceiling binds), and its error is correspondingly large (0.78 / 0.03-0.08 / 1e-3 at
maxdim 4 / 8 / 16). Every run that stayed below the cap -- all profiles, all combos -- met its
budget.

**8. Cost vs the legacy fixed-knob solver at matched accuracy (IT05; job 17195550, ONE process,
so wall clocks compare).** Adaptive settings `tol_trunc = 1e-4, tol_expv = 1e-2`:

| ordinary BUG | err adaptive / legacy kry3 | H applications | trial width r_aug | chi | wall |
|---|---|---|---|---|---|
| Neel T=2, tol_basis 1e-5 | 3.9e-6 / 4.7e-6 | 3636 / 3591 | 39 / 262 | 12 / 68 | 35 s / 51 s |
| DW T=2, tol_basis 1e-4 | 1.4e-4 / 2.6e-4 | 1611 / 3312 | 10 / 79 | 7 / 20 | 12 s / 22 s |
| Neel T=0.5, tol_basis 1e-6 | 2.2e-7 / 2.1e-7 | 962 / 891 | 21 / 104 | 5 / 31 | 7.7 s / 8.5 s |
| DW T=0.5, tol_basis 1e-4 | 4.9e-5 / 5.0e-5 | 339 / 632 | 5 / 27 | 4 / 9 | 2.2 s / 3.4 s |

At matched or better error the adaptive step keeps 5-8x fewer trial directions and 2-6x less rank,
needs at most ~8% more (and up to 2x fewer) H applications, and is faster; the legacy default is
worse on every count. The MIDPOINT step at ~1e-9 needs tight tol_trunc and then costs more H
applications than legacy kry3 (1222 vs 900, T = 0.5) while still keeping a narrower trial space
(58 vs 94). `exact = true` is the fastest arm throughout, with results identical to the RSVD arm.

## Recommended starting settings (XX-like dynamics, dt = 0.05)

Measured at T = 2 (combos table below; error in parentheses):

| target err | ordinary BUG | midpoint BUG |
|---|---|---|
| 1e-3 | tol_basis 1e-3, tol_trunc 1e-3, tol_expv 1e-2 (9.2e-4, chi 10) | tol_basis 1e-4, tol_trunc 1e-3, tol_expv 1e-2 (1.5e-4, chi 8) |
| 1e-4 | tol_basis 1e-5, tol_trunc 1e-3, tol_expv 1e-2 (6.7e-5, chi 8) | tol_basis 1e-5, tol_trunc 1e-3, tol_expv 1e-2 (7.0e-5, chi 8) |
| 1e-5 | tol_basis 1e-5, tol_trunc 1e-5, tol_expv 1e-6 (5.5e-6, chi 18; just above the 4.7e-6 floor) | tol_basis 1e-5, tol_trunc 1e-5, tol_expv 1e-6 (1.1e-6, chi 16) |

Rule of thumb from this model: tol_basis ~ target, tol_trunc ~ 10-100 x target, tol_expv 1e-2 is
enough at this dt; prefer `exact = true` while d*chi is a few hundred. One model, one dt: re-check
on a non-free-fermion Hamiltonian before relying on the constants.

## T = 0.5 (plan settings)

### Sweeps: one tolerance at a time (Neel, seed 1; others tight)

| sweep | tol | err(1 step) | err(T) | chi | r_aug | H applications | probes | Krylov dirs | C pass / D after C | expv iters | discarded | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| basis | 1e-01 | 2.99e-06 | 7.03e-03 | 9 | 9 | 471 | 2356 | 60 | 144 / 0 | 71 | 0.0e+00 | met |
| basis | 1e-02 | 1.31e-06 | 1.51e-03 | 16 | 16 | 653 | 3466 | 126 | 116 / 0 | 97 | 0.0e+00 | met |
| basis | 1e-03 | 2.30e-07 | 7.01e-05 | 23 | 23 | 779 | 5312 | 236 | 78 / 0 | 99 | 9.7e-12 | met |
| basis | 1e-04 | 1.95e-07 | 8.95e-06 | 30 | 30 | 884 | 6984 | 334 | 68 / 0 | 100 | 2.5e-11 | met |
| basis | 1e-05 | 4.44e-10 | 9.38e-07 | 36 | 36 | 1000 | 8178 | 482 | 64 / 0 | 100 | 3.4e-11 | met |
| basis | 1e-06 | 2.90e-10 | 2.33e-07 | 42 | 43 | 1094 | 9648 | 734 | 70 / 2 | 100 | 4.0e-11 | met |
| basis | 1e-07 | 2.90e-10 | 2.04e-07 | 45 | 50 | 1198 | 10712 | 1082 | 72 / 0 | 100 | 5.3e-11 | met |
| exact | 1e-01 | 2.99e-06 | 7.03e-03 | 9 | 9 | 471 | 0 | 60 | 144 / 0 | 71 | 0.0e+00 | met |
| exact | 1e-02 | 1.31e-06 | 1.51e-03 | 16 | 16 | 653 | 0 | 126 | 116 / 0 | 97 | 0.0e+00 | met |
| exact | 1e-03 | 2.30e-07 | 7.01e-05 | 23 | 23 | 779 | 0 | 236 | 78 / 0 | 99 | 9.7e-12 | met |
| exact | 1e-04 | 1.95e-07 | 8.95e-06 | 30 | 30 | 884 | 0 | 334 | 68 / 0 | 100 | 2.5e-11 | met |
| exact | 1e-05 | 4.44e-10 | 9.38e-07 | 36 | 36 | 1000 | 0 | 482 | 64 / 0 | 100 | 3.4e-11 | met |
| exact | 1e-06 | 2.90e-10 | 2.33e-07 | 42 | 43 | 1094 | 0 | 734 | 70 / 2 | 100 | 4.0e-11 | met |
| exact | 1e-07 | 2.90e-10 | 2.04e-07 | 45 | 50 | 1198 | 0 | 1082 | 72 / 0 | 100 | 5.3e-11 | met |
| mid | 1e-01 | 2.99e-06 | 6.07e-03 | 8 | 8 | 473 | 4652 | 54 | 350 / 0 | 135 | 0.0e+00 | met |
| mid | 1e-02 | 1.31e-06 | 1.53e-03 | 16 | 16 | 664 | 6988 | 114 | 308 / 0 | 181 | 0.0e+00 | met |
| mid | 1e-03 | 2.30e-07 | 5.70e-05 | 24 | 24 | 793 | 10030 | 226 | 282 / 0 | 187 | 7.2e-12 | met |
| mid | 1e-04 | 1.95e-07 | 4.12e-06 | 30 | 32 | 888 | 13054 | 344 | 252 / 0 | 189 | 3.7e-11 | met |
| mid | 1e-05 | 1.43e-10 | 2.89e-07 | 41 | 42 | 1008 | 17364 | 645 | 197 / 2 | 190 | 4.6e-11 | met |
| mid | 1e-06 | 7.10e-11 | 4.24e-08 | 47 | 48 | 1112 | 20456 | 1060 | 184 / 0 | 190 | 4.2e-11 | met |
| mid | 1e-07 | 1.45e-11 | 1.61e-09 | 49 | 58 | 1222 | 22973 | 1548 | 190 / 4 | 190 | 5.7e-11 | met |
| trunc | 1e-02 | 2.90e-10 | 5.26e-05 | 2 | 19 | 1088 | 4820 | 1674 | 38 / 0 | 100 | 8.2e-03 | met |
| trunc | 1e-03 | 2.90e-10 | 5.68e-06 | 4 | 25 | 1108 | 5580 | 1806 | 38 / 0 | 100 | 8.0e-04 | met |
| trunc | 1e-04 | 2.90e-10 | 2.28e-07 | 4 | 27 | 1116 | 5808 | 1798 | 38 / 0 | 100 | 4.9e-05 | met |
| trunc | 1e-05 | 2.90e-10 | 2.04e-07 | 6 | 27 | 1128 | 5976 | 1790 | 38 / 0 | 100 | 7.5e-06 | met |
| trunc | 1e-06 | 2.90e-10 | 2.05e-07 | 6 | 27 | 1128 | 6116 | 1774 | 38 / 0 | 100 | 8.1e-07 | met |
| trunc | 1e-07 | 2.90e-10 | 2.06e-07 | 11 | 27 | 1160 | 6576 | 1694 | 54 / 0 | 100 | 6.5e-08 | met |
| trunc | 1e-08 | 2.90e-10 | 2.05e-07 | 27 | 40 | 1198 | 9248 | 1536 | 54 / 0 | 100 | 6.3e-09 | met |
| trunc | 1e-09 | 2.90e-10 | 2.07e-07 | 41 | 46 | 1190 | 10168 | 1241 | 68 / 2 | 100 | 5.1e-10 | met |
| trunc | 1e-10 | 2.90e-10 | 2.04e-07 | 45 | 50 | 1198 | 10712 | 1082 | 72 / 0 | 100 | 5.3e-11 | met |
| expv | 1e-02 | 1.75e-09 | 2.28e-07 | 59 | 61 | 1198 | 12158 | 795 | 74 / 0 | 40 | 4.3e-11 | met |
| expv | 1e-04 | 3.01e-10 | 2.06e-07 | 58 | 58 | 1228 | 12156 | 804 | 72 / 2 | 50 | 4.3e-11 | met |
| expv | 1e-06 | 2.90e-10 | 2.05e-07 | 54 | 59 | 1190 | 11958 | 965 | 72 / 0 | 70 | 4.8e-11 | met |
| expv | 1e-08 | 2.90e-10 | 2.04e-07 | 44 | 49 | 1176 | 10621 | 1086 | 72 / 0 | 80 | 5.5e-11 | met |
| expv | 1e-10 | 2.90e-10 | 2.04e-07 | 45 | 50 | 1188 | 10716 | 1084 | 72 / 0 | 90 | 5.5e-11 | met |
| expv | 1e-12 | 2.90e-10 | 2.04e-07 | 45 | 50 | 1198 | 10712 | 1082 | 72 / 0 | 100 | 5.3e-11 | met |

#### Loosest tolerance reaching each accuracy target (max |Sz - exact| over the run)

| sweep | err <= 1e-02 | err <= 1e-03 | err <= 1e-04 | err <= 1e-05 |
|---|---|---|---|---|
| basis | 1e-01 | 1e-03 | 1e-03 | 1e-04 |
| exact | 1e-01 | 1e-03 | 1e-03 | 1e-04 |
| mid | 1e-01 | 1e-03 | 1e-03 | 1e-04 |
| trunc | 1e-02 | 1e-02 | 1e-02 | 1e-03 |
| expv | 1e-02 | 1e-02 | 1e-02 | 1e-02 |

Error floor over all sweep runs: 1.61e-09 (what the tightest settings reach at this dt -- the integrator/time-step error no tolerance removes).

### Profiles (IT01-IT04, IT06): mean +- spread over seeds

| profile | scheme | state | seeds | err(1 step) | err(T) mean | err(T) max | chi | H applications | norm drift | Sz drift | unmet runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loose | midpoint | dw | 5 | 8.48e-12 | 1.77e-06 +- 0.0e+00 | 1.77e-06 | 3 | 340 | 1.3e-12 | 2.0e-15 | 0 |
| loose | midpoint | neel | 5 | 2.30e-07 | 5.92e-05 +- 4.0e-12 | 5.92e-05 | 15 | 744 | 1.9e-11 | 1.2e-15 | 0 |
| loose | ordinary | dw | 5 | 1.63e-07 | 4.89e-05 +- 0.0e+00 | 4.89e-05 | 3 | 332 | 3.2e-13 | 1.6e-15 | 0 |
| loose | ordinary | neel | 5 | 2.30e-07 | 1.52e-04 +- 0.0e+00 | 1.52e-04 | 15 | 718 | 1.5e-11 | 1.2e-15 | 0 |
| tight | midpoint | allup | 1 | 3.33e-16 | 8.88e-16 +- 0.0e+00 | 8.88e-16 | 1 | 210 | 0.0e+00 | 0.0e+00 | 0 |
| tight | midpoint | dw | 5 | 1.02e-11 | 4.26e-08 +- 0.0e+00 | 4.26e-08 | 6 | 438 | 8.7e-15 | 8.0e-15 | 0 |
| tight | midpoint | neel | 5 | 1.43e-10 | 4.42e-07 +- 4.5e-10 | 4.42e-07 | 28 | 1000 | 1.3e-14 | 1.1e-15 | 0 |
| tight | ordinary | allup | 1 | 3.33e-16 | 8.88e-16 +- 0.0e+00 | 8.88e-16 | 1 | 210 | 0.0e+00 | 0.0e+00 | 0 |
| tight | ordinary | dw | 5 | 1.63e-07 | 5.01e-05 +- 0.0e+00 | 5.01e-05 | 6 | 413 | 5.8e-15 | 5.6e-15 | 0 |
| tight | ordinary | neel | 5 | 4.44e-10 | 5.32e-07 +- 6.3e-14 | 5.32e-07 | 26 | 988 | 8.2e-15 | 1.6e-15 | 0 |

### Baseline (IT05): legacy fixed-knob solver

| arm | scheme | state | err(1 step) | err(T) | chi | H applications | t_step sum |
|---|---|---|---|---|---|---|---|
| legacy_cut1e-8 | midpoint | dw | 1.02e-11 | 4.32e-08 | 6 | 776 | 4.5s |
| legacy_cut1e-8 | ordinary | dw | 1.63e-07 | 5.01e-05 | 6 | 679 | 3.4s |
| legacy_cut1e-8 | midpoint | neel | 1.30e-10 | 1.18e-09 | 9 | 1462 | 20.0s |
| legacy_cut1e-8 | ordinary | neel | 1.81e-10 | 2.03e-07 | 12 | 1412 | 9.6s |
| legacy_default | midpoint | dw | 1.02e-11 | 4.32e-08 | 10 | 780 | 5.3s |
| legacy_default | ordinary | dw | 1.63e-07 | 5.01e-05 | 10 | 688 | 4.1s |
| legacy_default | midpoint | neel | 1.30e-10 | 1.17e-09 | 30 | 1462 | 25.2s |
| legacy_default | ordinary | neel | 1.81e-10 | 2.03e-07 | 29 | 1416 | 18.0s |
| legacy_kry3 | midpoint | dw | 1.02e-11 | 4.32e-08 | 9 | 758 | 4.7s |
| legacy_kry3 | ordinary | dw | 1.63e-07 | 5.01e-05 | 9 | 636 | 3.6s |
| legacy_kry3 | midpoint | neel | 1.97e-12 | 9.00e-10 | 23 | 900 | 12.0s |
| legacy_kry3 | ordinary | neel | 7.05e-10 | 2.07e-07 | 32 | 891 | 9.6s |


## T = 2 (longer evolution, chi up to maxdim)

### Sweeps: one tolerance at a time (Neel, seed 1; others tight)

| sweep | tol | err(1 step) | err(T) | chi | r_aug | H applications | probes | Krylov dirs | C pass / D after C | expv iters | discarded | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| basis | 1e-01 | 2.99e-06 | 8.93e-02 | 25 | 25 | 1970 | 18306 | 170 | 638 / 0 | 370 | 9.7e-16 | met |
| basis | 1e-02 | 1.31e-06 | 6.78e-03 | 72 | 72 | 2693 | 39574 | 574 | 442 / 0 | 397 | 1.6e-16 | met |
| basis | 1e-03 | 2.30e-07 | 2.48e-04 | 111 | 111 | 3179 | 57672 | 864 | 378 / 0 | 399 | 1.5e-11 | met |
| basis | 1e-04 | 1.95e-07 | 2.23e-05 | 128 | 138 | 3604 | 73830 | 1246 | 364 / 0 | 400 | 1.7e-08 | budget_unmet (trunc) |
| basis | 1e-05 | 4.44e-10 | 5.80e-06 | 128 | 149 | 4090 | 89655 | 2179 | 378 / 0 | 400 | 1.5e-08 | budget_unmet (trunc) |
| basis | 1e-06 | 2.90e-10 | 4.64e-06 | 128 | 162 | 4510 | 98315 | 3734 | 401 / 2 | 400 | 3.3e-09 | budget_unmet (trunc) |
| basis | 1e-07 | 2.90e-10 | 4.66e-06 | 128 | 174 | 4844 | 100767 | 6164 | 404 / 0 | 400 | 8.1e-10 | budget_unmet (trunc) |
| exact | 1e-01 | 2.99e-06 | 8.93e-02 | 25 | 25 | 1970 | 0 | 170 | 638 / 0 | 370 | 4.6e-16 | met |
| exact | 1e-02 | 1.31e-06 | 6.78e-03 | 72 | 72 | 2693 | 0 | 574 | 442 / 0 | 397 | 1.6e-16 | met |
| exact | 1e-03 | 2.30e-07 | 2.48e-04 | 111 | 111 | 3179 | 0 | 864 | 378 / 0 | 399 | 1.5e-11 | met |
| exact | 1e-04 | 1.95e-07 | 2.23e-05 | 128 | 138 | 3604 | 0 | 1246 | 364 / 0 | 400 | 1.7e-08 | budget_unmet (trunc) |
| exact | 1e-05 | 4.44e-10 | 5.80e-06 | 128 | 149 | 4080 | 0 | 2172 | 380 / 0 | 400 | 1.5e-08 | budget_unmet (trunc) |
| exact | 1e-06 | 2.90e-10 | 4.66e-06 | 128 | 160 | 4521 | 0 | 3711 | 402 / 2 | 400 | 3.6e-09 | budget_unmet (trunc) |
| exact | 1e-07 | 2.90e-10 | 4.66e-06 | 128 | 174 | 4844 | 0 | 6145 | 403 / 0 | 400 | 6.6e-10 | budget_unmet (trunc) |
| mid | 1e-01 | 2.99e-06 | 1.00e-01 | 29 | 29 | 1973 | 36292 | 190 | 1428 / 0 | 705 | 0.0e+00 | met |
| mid | 1e-02 | 1.31e-06 | 6.10e-03 | 72 | 72 | 2704 | 80212 | 586 | 1218 / 0 | 751 | 1.6e-16 | met |
| mid | 1e-03 | 2.30e-07 | 1.70e-04 | 109 | 109 | 3181 | 113116 | 864 | 1188 / 0 | 757 | 9.0e-12 | met |
| mid | 1e-04 | 1.95e-07 | 9.77e-06 | 128 | 143 | 3568 | 147670 | 1351 | 1130 / 0 | 759 | 2.0e-08 | budget_unmet (trunc) |
| mid | 1e-05 | 1.43e-10 | 2.26e-06 | 128 | 159 | 4038 | 183094 | 2908 | 973 / 2 | 760 | 2.1e-08 | budget_unmet (trunc) |
| mid | 1e-06 | 7.10e-11 | 4.87e-08 | 128 | 177 | 4567 | 203551 | 5579 | 933 / 0 | 760 | 3.8e-09 | budget_unmet (trunc) |
| mid | 1e-07 | 1.45e-11 | 9.26e-09 | 128 | 190 | 4918 | 208089 | 8565 | 924 / 4 | 760 | 8.6e-10 | budget_unmet (trunc) |
| trunc | 1e-02 | 2.90e-10 | 5.46e-03 | 6 | 44 | 4596 | 29360 | 9978 | 202 / 0 | 400 | 8.9e-03 | met |
| trunc | 1e-03 | 2.90e-10 | 6.50e-05 | 8 | 57 | 4656 | 35460 | 11186 | 224 / 0 | 400 | 8.5e-04 | met |
| trunc | 1e-04 | 2.90e-10 | 5.14e-06 | 12 | 68 | 4672 | 39026 | 11416 | 234 / 0 | 400 | 7.3e-05 | met |
| trunc | 1e-05 | 2.90e-10 | 4.23e-06 | 16 | 69 | 4686 | 41254 | 11364 | 236 / 0 | 400 | 7.8e-06 | met |
| trunc | 1e-06 | 2.90e-10 | 4.53e-06 | 20 | 75 | 4696 | 43826 | 11602 | 238 / 0 | 400 | 8.1e-07 | met |
| trunc | 1e-07 | 2.90e-10 | 4.58e-06 | 28 | 76 | 4780 | 47896 | 11790 | 254 / 0 | 400 | 7.3e-08 | met |
| trunc | 1e-08 | 2.90e-10 | 4.60e-06 | 66 | 107 | 4846 | 65138 | 10007 | 308 / 0 | 400 | 6.5e-09 | met |
| trunc | 1e-09 | 2.90e-10 | 4.62e-06 | 108 | 142 | 4836 | 83001 | 7757 | 369 / 2 | 400 | 6.2e-10 | met |
| trunc | 1e-10 | 2.90e-10 | 4.66e-06 | 128 | 174 | 4844 | 100767 | 6164 | 404 / 0 | 400 | 8.1e-10 | budget_unmet (trunc) |
| expv | 1e-02 | 1.75e-09 | 6.16e-06 | 128 | 170 | 4968 | 113540 | 4680 | 414 / 0 | 160 | 5.0e-09 | budget_unmet (trunc) |
| expv | 1e-04 | 3.01e-10 | 4.54e-06 | 128 | 171 | 5064 | 111743 | 4612 | 418 / 2 | 200 | 3.0e-09 | budget_unmet (trunc) |
| expv | 1e-06 | 2.90e-10 | 4.67e-06 | 128 | 177 | 4867 | 109309 | 5741 | 407 / 0 | 280 | 9.4e-10 | budget_unmet (trunc) |
| expv | 1e-08 | 2.90e-10 | 4.66e-06 | 128 | 176 | 4762 | 100315 | 6134 | 403 / 0 | 320 | 5.8e-10 | budget_unmet (trunc) |
| expv | 1e-10 | 2.90e-10 | 4.66e-06 | 128 | 174 | 4804 | 100895 | 6167 | 405 / 0 | 360 | 5.3e-10 | budget_unmet (trunc) |
| expv | 1e-12 | 2.90e-10 | 4.66e-06 | 128 | 174 | 4844 | 100767 | 6164 | 404 / 0 | 400 | 8.1e-10 | budget_unmet (trunc) |

#### Loosest tolerance reaching each accuracy target (max |Sz - exact| over the run)

| sweep | err <= 1e-02 | err <= 1e-03 | err <= 1e-04 | err <= 1e-05 |
|---|---|---|---|---|
| basis | 1e-02 | 1e-03 | 1e-04 | 1e-05 |
| exact | 1e-02 | 1e-03 | 1e-04 | 1e-05 |
| mid | 1e-02 | 1e-03 | 1e-04 | 1e-04 |
| trunc | 1e-02 | 1e-03 | 1e-03 | 1e-04 |
| expv | 1e-02 | 1e-02 | 1e-02 | 1e-02 |

Error floor over all sweep runs: 9.26e-09 (what the tightest settings reach at this dt -- the integrator/time-step error no tolerance removes).

### Combined loose settings (do they compound?)

| scheme | tol_basis | tol_trunc | tol_expv | maxdim | err(T) | chi | H applications | discarded | status |
|---|---|---|---|---|---|---|---|---|---|
| midpoint | 1e-03 | 1e-03 | 1e-02 | 128 | 1.31e-03 | 8 | 2962 | 8.0e-04 | met |
| midpoint | 1e-03 | 1e-03 | 1e-06 | 128 | 1.30e-03 | 8 | 2881 | 8.0e-04 | met |
| midpoint | 1e-03 | 1e-05 | 1e-02 | 128 | 2.03e-04 | 59 | 2970 | 5.8e-06 | met |
| midpoint | 1e-03 | 1e-05 | 1e-06 | 128 | 1.87e-04 | 47 | 3016 | 5.2e-06 | met |
| midpoint | 1e-04 | 1e-03 | 1e-02 | 128 | 1.46e-04 | 8 | 3554 | 8.5e-04 | met |
| midpoint | 1e-04 | 1e-03 | 1e-06 | 128 | 1.45e-04 | 8 | 3476 | 8.5e-04 | met |
| midpoint | 1e-04 | 1e-05 | 1e-02 | 128 | 2.57e-05 | 47 | 3651 | 6.9e-06 | met |
| midpoint | 1e-04 | 1e-05 | 1e-06 | 128 | 1.31e-05 | 41 | 3454 | 7.2e-06 | met |
| midpoint | 1e-05 | 1e-03 | 1e-02 | 128 | 6.98e-05 | 8 | 3970 | 8.5e-04 | met |
| midpoint | 1e-05 | 1e-03 | 1e-06 | 128 | 6.53e-05 | 8 | 3880 | 8.5e-04 | met |
| midpoint | 1e-05 | 1e-05 | 1e-02 | 128 | 3.37e-06 | 30 | 3964 | 7.4e-06 | met |
| midpoint | 1e-05 | 1e-05 | 1e-06 | 128 | 1.09e-06 | 16 | 3924 | 8.1e-06 | met |
| ordinary | 1e-03 | 1e-03 | 1e-02 | 128 | 9.23e-04 | 10 | 2618 | 8.1e-04 | met |
| ordinary | 1e-03 | 1e-03 | 1e-06 | 128 | 9.18e-04 | 10 | 2698 | 7.6e-04 | met |
| ordinary | 1e-03 | 1e-05 | 1e-02 | 128 | 1.62e-04 | 57 | 2950 | 5.1e-06 | met |
| ordinary | 1e-03 | 1e-05 | 1e-06 | 128 | 4.49e-04 | 41 | 2999 | 6.1e-06 | met |
| ordinary | 1e-04 | 1e-03 | 1e-02 | 128 | 2.23e-04 | 8 | 3146 | 8.5e-04 | met |
| ordinary | 1e-04 | 1e-03 | 1e-06 | 128 | 2.22e-04 | 8 | 3266 | 8.5e-04 | met |
| ordinary | 1e-04 | 1e-05 | 1e-02 | 128 | 3.04e-05 | 46 | 3603 | 6.2e-06 | met |
| ordinary | 1e-04 | 1e-05 | 1e-06 | 128 | 6.04e-05 | 38 | 3346 | 6.6e-06 | met |
| ordinary | 1e-05 | 1e-03 | 1e-02 | 128 | 6.72e-05 | 8 | 3614 | 8.5e-04 | met |
| ordinary | 1e-05 | 1e-03 | 1e-06 | 128 | 6.62e-05 | 8 | 3734 | 8.5e-04 | met |
| ordinary | 1e-05 | 1e-05 | 1e-02 | 128 | 9.38e-06 | 30 | 3934 | 7.2e-06 | met |
| ordinary | 1e-05 | 1e-05 | 1e-06 | 128 | 5.51e-06 | 18 | 3792 | 7.2e-06 | met |

### Rank cap binding (IT07: failures from maxdim are visible)

| scheme | tol_basis | tol_trunc | tol_expv | maxdim | err(T) | chi | H applications | discarded | status |
|---|---|---|---|---|---|---|---|---|---|
| midpoint | 1e-06 | 1e-08 | 1e-10 | 4 | 7.81e-01 | 4 | 1989 | 1.9e-02 | budget_unmet (krylov+tail+trunc) |
| midpoint | 1e-06 | 1e-08 | 1e-10 | 8 | 7.91e-02 | 8 | 2414 | 1.8e-02 | budget_unmet (krylov+tail+trunc) |
| midpoint | 1e-06 | 1e-08 | 1e-10 | 16 | 8.19e-04 | 16 | 3137 | 2.3e-04 | budget_unmet (krylov+projection+tail+trunc) |
| ordinary | 1e-06 | 1e-08 | 1e-10 | 4 | 7.52e-01 | 4 | 1970 | 9.2e-03 | budget_unmet (krylov+trunc) |
| ordinary | 1e-06 | 1e-08 | 1e-10 | 8 | 2.95e-02 | 8 | 2569 | 1.2e-02 | budget_unmet (krylov+trunc) |
| ordinary | 1e-06 | 1e-08 | 1e-10 | 16 | 9.72e-04 | 16 | 3412 | 2.5e-04 | budget_unmet (krylov+trunc) |

## Matched accuracy, one process (job 17195550)

T = 0.5:

| scheme | state | arm | tol_basis | err(T) | H applications | trial width r_aug | chi | t_step sum | status |
|---|---|---|---|---|---|---|---|---|---|
| midpoint | dw | legacy_default |  | 4.32e-08 | 780 | 45 | 10 | 4.9s | legacy |
| midpoint | dw | legacy_kry3 |  | 4.32e-08 | 758 | 32 | 9 | 4.5s | legacy |
| midpoint | dw | adaptive | 1e-06 | 1.23e-06 | 438 | 9 | 4 | 2.7s | met |
| midpoint | dw | adaptive_exact | 1e-06 | 1.23e-06 | 438 | 9 | 4 | 2.2s | met |
| midpoint | dw | adaptive | 1e-05 | 1.23e-06 | 402 | 7 | 4 | 2.6s | met |
| midpoint | dw | adaptive_exact | 1e-05 | 1.23e-06 | 402 | 7 | 4 | 2.1s | met |
| midpoint | dw | adaptive | 1e-04 | 1.23e-06 | 344 | 6 | 4 | 2.3s | met |
| midpoint | dw | adaptive_exact | 1e-04 | 1.23e-06 | 344 | 6 | 4 | 1.8s | met |
| midpoint | neel | legacy_kry3 |  | 8.94e-10 | 900 | 94 | 23 | 10.0s | legacy |
| midpoint | neel | legacy_default |  | 1.16e-09 | 1464 | 241 | 30 | 21.9s | legacy |
| midpoint | neel | adaptive_exact | 1e-06 | 2.26e-07 | 1160 | 26 | 5 | 7.4s | met |
| midpoint | neel | adaptive | 1e-06 | 2.26e-07 | 1160 | 26 | 5 | 9.2s | met |
| midpoint | neel | adaptive | 1e-05 | 5.46e-07 | 976 | 23 | 5 | 8.0s | met |
| midpoint | neel | adaptive_exact | 1e-05 | 5.46e-07 | 976 | 23 | 5 | 6.5s | met |
| midpoint | neel | adaptive | 1e-04 | 4.76e-06 | 880 | 15 | 4 | 7.3s | met |
| midpoint | neel | adaptive_exact | 1e-04 | 4.76e-06 | 880 | 15 | 4 | 5.9s | met |
| ordinary | dw | adaptive | 1e-04 | 4.90e-05 | 339 | 5 | 4 | 2.2s | met |
| ordinary | dw | adaptive_exact | 1e-04 | 4.90e-05 | 339 | 5 | 4 | 1.8s | met |
| ordinary | dw | adaptive | 1e-05 | 4.90e-05 | 375 | 6 | 4 | 2.4s | met |
| ordinary | dw | adaptive_exact | 1e-05 | 4.90e-05 | 375 | 6 | 4 | 1.8s | met |
| ordinary | dw | adaptive | 1e-06 | 4.90e-05 | 399 | 8 | 4 | 2.4s | met |
| ordinary | dw | adaptive_exact | 1e-06 | 4.90e-05 | 399 | 8 | 4 | 2.0s | met |
| ordinary | dw | legacy_default |  | 5.01e-05 | 688 | 36 | 10 | 3.9s | legacy |
| ordinary | dw | legacy_kry3 |  | 5.01e-05 | 632 | 27 | 9 | 3.4s | legacy |
| ordinary | neel | legacy_default |  | 2.03e-07 | 1416 | 169 | 29 | 15.9s | legacy |
| ordinary | neel | legacy_kry3 |  | 2.07e-07 | 891 | 104 | 31 | 8.5s | legacy |
| ordinary | neel | adaptive | 1e-06 | 2.24e-07 | 962 | 21 | 5 | 7.7s | met |
| ordinary | neel | adaptive_exact | 1e-06 | 2.24e-07 | 962 | 21 | 5 | 6.6s | met |
| ordinary | neel | adaptive | 1e-05 | 2.88e-07 | 878 | 18 | 5 | 7.2s | met |
| ordinary | neel | adaptive_exact | 1e-05 | 2.88e-07 | 878 | 18 | 5 | 5.9s | met |
| ordinary | neel | adaptive | 1e-04 | 1.37e-05 | 766 | 13 | 6 | 19.9s | met |
| ordinary | neel | adaptive_exact | 1e-04 | 1.37e-05 | 766 | 13 | 6 | 5.6s | met |


T = 2:

| scheme | state | arm | tol_basis | err(T) | H applications | trial width r_aug | chi | t_step sum | status |
|---|---|---|---|---|---|---|---|---|---|
| midpoint | dw | legacy_kry3 |  | 1.18e-06 | 3456 | 87 | 20 | 25.8s | legacy |
| midpoint | dw | legacy_default |  | 1.18e-06 | 3788 | 130 | 28 | 31.9s | legacy |
| midpoint | dw | adaptive | 1e-04 | 1.30e-04 | 1612 | 11 | 7 | 12.8s | met |
| midpoint | dw | adaptive_exact | 1e-04 | 1.30e-04 | 1612 | 11 | 7 | 9.9s | met |
| midpoint | dw | adaptive | 1e-06 | 1.32e-04 | 2156 | 19 | 7 | 15.4s | met |
| midpoint | dw | adaptive_exact | 1e-06 | 1.32e-04 | 2156 | 19 | 7 | 12.3s | met |
| midpoint | dw | adaptive_exact | 1e-05 | 1.32e-04 | 1958 | 15 | 7 | 11.2s | met |
| midpoint | dw | adaptive | 1e-05 | 1.32e-04 | 1958 | 15 | 7 | 14.3s | met |
| midpoint | neel | legacy_default |  | 1.57e-09 | 5847 | 537 | 54 | 197.6s | legacy |
| midpoint | neel | legacy_kry3 |  | 3.25e-09 | 3600 | 264 | 49 | 68.8s | legacy |
| midpoint | neel | adaptive | 1e-06 | 3.87e-06 | 4698 | 73 | 12 | 43.1s | met |
| midpoint | neel | adaptive_exact | 1e-06 | 3.87e-06 | 4698 | 73 | 12 | 35.7s | met |
| midpoint | neel | adaptive | 1e-05 | 5.73e-06 | 3976 | 55 | 12 | 38.2s | met |
| midpoint | neel | adaptive_exact | 1e-05 | 5.73e-06 | 3976 | 55 | 12 | 31.6s | met |
| midpoint | neel | adaptive | 1e-04 | 6.93e-05 | 3552 | 39 | 12 | 34.6s | met |
| midpoint | neel | adaptive_exact | 1e-04 | 6.93e-05 | 3552 | 39 | 12 | 28.3s | met |
| ordinary | dw | adaptive | 1e-05 | 1.35e-04 | 1901 | 13 | 7 | 13.5s | met |
| ordinary | dw | adaptive_exact | 1e-05 | 1.35e-04 | 1901 | 13 | 7 | 10.7s | met |
| ordinary | dw | adaptive | 1e-06 | 1.36e-04 | 2049 | 17 | 7 | 14.2s | met |
| ordinary | dw | adaptive_exact | 1e-06 | 1.36e-04 | 2049 | 17 | 7 | 11.4s | met |
| ordinary | dw | adaptive_exact | 1e-04 | 1.36e-04 | 1611 | 10 | 7 | 9.6s | met |
| ordinary | dw | adaptive | 1e-04 | 1.36e-04 | 1611 | 10 | 7 | 12.3s | met |
| ordinary | dw | legacy_default |  | 2.56e-04 | 3648 | 112 | 28 | 27.1s | legacy |
| ordinary | dw | legacy_kry3 |  | 2.56e-04 | 3312 | 79 | 20 | 21.9s | legacy |
| ordinary | neel | adaptive | 1e-06 | 3.84e-06 | 4000 | 58 | 12 | 38.7s | met |
| ordinary | neel | adaptive_exact | 1e-06 | 3.84e-06 | 4000 | 58 | 12 | 32.1s | met |
| ordinary | neel | adaptive | 1e-05 | 3.86e-06 | 3636 | 39 | 12 | 35.5s | met |
| ordinary | neel | adaptive_exact | 1e-05 | 3.86e-06 | 3636 | 39 | 12 | 29.3s | met |
| ordinary | neel | legacy_default |  | 4.67e-06 | 5680 | 545 | 69 | 116.2s | legacy |
| ordinary | neel | legacy_kry3 |  | 4.68e-06 | 3591 | 262 | 68 | 51.1s | legacy |
| ordinary | neel | adaptive | 1e-04 | 1.80e-04 | 3180 | 31 | 12 | 46.0s | met |
| ordinary | neel | adaptive_exact | 1e-04 | 1.80e-04 | 3180 | 31 | 12 | 27.0s | met |

