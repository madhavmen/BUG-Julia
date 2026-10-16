# How loose can the adaptive CBE-DMRG tolerances be?

The ground-state counterpart of [ADAPTIVE_XX_L20_RESULTS.md](ADAPTIVE_XX_L20_RESULTS.md)
(simulation_plan_2 Task 4). The script is `benchmarks/ground_state/adaptive_dmrg_tolerance.jl` and
the tables come from `analyze_dmrg_tolerance.py`. Every run starts from Neel at N = 20 and is
scored against the exact ground energy in the S^z = 0 sector (sparse Lanczos, 184756 states, residual
< 1e-13). It runs on 2 threads, and wall times are comparable only within one model.

The error reported is the energy of the RETURNED state, after its truncation:
`dE/N = (E - E0)/N`. The exact variance (`mpo_variance`) is reported beside it.

## Heisenberg chain, open, N = 20 (maxdim 256; job 17208958)

| `tol_basis` (`tol_trunc` 1e-8) | 1e-1 | 3e-2 | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|---|---|---|
| dE/N | 2.0e-5 | 1.6e-6 | 1.6e-7 | 1.5e-8 | 1.4e-9 | 1.2e-11 | 1.1e-13 | 9e-16 |
| variance | 9.0e-4 | 9.2e-5 | 8.4e-6 | 9.6e-7 | 1.1e-7 | 9.7e-10 | 1.0e-11 | 9e-14 |
| chi | 9 | 16 | 19 | 22 | 35 | 44 | 65 | 91 |

| `tol_trunc` (`tol_basis` 1e-5) | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-8 |
|---|---|---|---|---|---|---|---|
| dE/N | 8.2e-6 | 6.9e-7 | 6.3e-8 | 1.1e-9 | 9.2e-12 | 2.1e-13 | 1.1e-13 |
| chi | 11 | 17 | 20 | 36 | 44 | 63 | 65 |

Loosest tolerance reaching each target:

| target dE/N | loosest `tol_basis` | chi | loosest `tol_trunc` | chi |
|---|---|---|---|---|
| 1e-4 | 1e-1 | 9 | 1e-2 | 11 |
| 1e-6 | 1e-2 | 19 | 3e-3 | 17 |
| 1e-8 | 1e-3 | 35 | 1e-4 | 36 |
| 1e-10 | 1e-4 | 44 | 1e-5 | 44 |

Findings:

1. **Both tolerances act quadratically on the energy.** Over six decades the fits are
   `dE/N ~ 1.5e-3 * tol_basis^2`, `dE/N ~ 0.08 * tol_trunc^2` and `variance ~ 0.1 * tol_basis^2`.
   This is the variational square law: the energy error is second order in the state error, so a
   ground state tolerates far looser settings than a time step does. The time step's error is
   linear in `tol_basis`.
2. **The two tolerances balance at `tol_trunc ~ tol_basis / 7`.** The provisional defaults
   `1e-3 / 1e-4` sit on that line: dE/N 1.9e-9, variance 1.6e-7, chi 32.
3. **They do not compound.** In the combined runs the looser of the two sets the error:
   `(1e-2, 1e-3)` gives the `tol_basis = 1e-2` result, and `(1e-3, 1e-3)` the `tol_trunc = 1e-3` one.
4. **Every run ends `:met`, so no run claims an accuracy it lacks.**
5. **At this size, loose settings buy RANK, not time.** Every run takes 11-15 s with about 1,000-1,300
   matvecs. The loosest settings need more sweeps (7-8) and more matvecs, up to 1,934 at
   `tol_trunc = 1e-2`. The legacy fixed-knob run reaches round-off at chi 162 in 13.3 s; the
   adaptive run at `1e-6 / 1e-8` gets the same accuracy at chi 91 in 12.3 s.
6. **The exact selection is faster than the RSVD sketch here** (9.7 s vs 12.3 s, identical
   energies). This is the same crossover as the BUG: the sketch pays only at large chi.

## XX chain, open, N = 20 (maxdim 256; job 17208958)

| `tol_basis` (`tol_trunc` 1e-8) | 1e-1 | 3e-2 | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|---|---|---|
| dE/N | 2.7e-5 | 1.5e-6 | 3.2e-7 | 1.9e-8 | 2.5e-9 | 1.8e-11 | 2.2e-13 | 3e-15 |
| chi | 10 | 12 | 18 | 22 | 27 | 43 | 58 | 87 |
| sweeps / wall s | 34 / 59.9 | 6 / 12.4 | 14 / 26.4 | 6 / 12.5 | 6 / 12.6 | 6 / 13.1 | 5 / 11.3 | 5 / 11.7 |

| `tol_trunc` (`tol_basis` 1e-5) | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-8 |
|---|---|---|---|---|---|---|---|
| dE/N | 2.4e-6 | 4.8e-7 | 5.5e-8 | 7.7e-10 | 5.8e-12 | 2.8e-13 | 2.2e-13 |
| chi | 12 | 16 | 21 | 31 | 46 | 57 | 58 |
| sweeps / wall s | 6 / 15.6 | 14 / 28.6 | 10 / 21.8 | 5 / 11.5 | 6 / 14.1 | 5 / 11.5 | 5 / 11.3 |

7. **The same square law holds:** `dE/N ~ 2.5e-3 * tol_basis^2` and `~ 0.08 * tol_trunc^2`. The
   defaults `1e-3 / 1e-4` give 2.5e-9 at chi 27, against the legacy run's chi 163.
8. **Very loose settings converge SLOWLY** (34 sweeps at `tol_basis = 0.1`; 10-14 sweeps at
   `1e-2` / `3e-3` / `1e-3` on either knob). The cause is the fixed sweep-to-sweep stopping test
   `etol = 1e-10`: a state truncated at a loose tolerance fluctuates by more than that between
   sweeps, so the run keeps sweeping long after reaching the accuracy its tolerances allow. A fixed
   `etol` is a fourth accuracy knob that contradicts the other two. Next step: derive the stopping
   threshold from `tol_basis` / `tol_trunc` using the square law above.

## Square cylinder 5x4, N = 20 (maxdim 512; job 17208958)

| `tol_basis` (`tol_trunc` 1e-8) | 1e-1 | 3e-2 | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|---|---|---|---|
| dE/N | 9.5e-6 | 5.5e-7 | 4.6e-8 | 2.8e-9 | 2.7e-10 | 1.4e-11 | 1.3e-11 |
| chi | 81 | 138 | 221 | 335 | 420 | 512 (cap) | 512 (cap) |
| sweeps / wall s | 40 / 114 | 40 / 149 | 9 / 42 | 7 / 40 | 7 / 47 | 7 / 58 | 7 / 58 |

| `tol_trunc` (`tol_basis` 1e-5) | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|---|---|
| dE/N | 2.4e-5 | 1.9e-6 | 2.3e-7 | 1.7e-9 | 2.1e-11 |
| chi | 62 | 107 | 169 | 357 | 512 (cap) |
| sweeps / wall s | 12 / 54 | 16 / 78 | 11 / 52 | 9 / 64 | 7 / 59 |

Legacy fixed-knob run: dE/N 1.3e-11, chi 512 (the cap), 7 sweeps, 62.3 s.

9. **In 2D the adaptive solver wins wall time, not only rank.** At matched-or-better
   targets: `tol_basis = 3e-3` reaches 2.8e-9 at chi 335 in 39.9 s (0.64x legacy). The defaults
   `1e-3 / 1e-4` reach 2.0e-9 at chi 351 in 49.9 s (0.80x); `1e-3 / 1e-5` reaches 2.8e-10 at chi 417
   in 45.9 s (0.74x).
10. **The same square law holds, with 2D coefficients:** `dE/N ~ 3e-4 tol_basis^2` and
    `~ 0.2 tol_trunc^2`. The truncation side dominates the defaults in 2D (1.7e-9 against 2.7e-10).
11. **Tight settings hit the cap and say so.** From `tol_basis <= 1e-4` or `tol_trunc <= 1e-5` the
    rank reaches `maxdim = 512`, the status is `:budget_unmet`, and the energy floors at the
    legacy value 1.3e-11.
12. **The stopping-test problem is worse in 2D.** `tol_basis = 0.1` and `3e-2` ran to the 40-sweep
    limit.

### The stopping threshold is now derived (`dmrg_etol`)

When the caller names no `etol`, the adaptive DMRG stops once a sweep changes the energy by less
than a tenth of what its tolerances allow, using the measured law with the largest coefficients seen:
`etol = 0.1 N (2.5e-3 tol_basis^2 + 0.2 tol_trunc^2)`, floor `1e-12 N`. An explicit `etol` wins,
and the legacy path keeps 1e-10. Verification reruns the same grid with the derived threshold
(`adaptive_dmrg_tolerance.jl MODEL DIR auto`, job 17265883).

**Chains, derived threshold (job 17265883).** Every adaptive run now converges in 4-6 sweeps, and
every energy matches the fixed-`etol` run to the printed digits, so stopping earlier costs no
accuracy:

| Run | fixed `etol` = 1e-10 | derived `dmrg_etol` | dE/N |
|---|---|---|---|
| XX, `tol_basis` 0.1 | 34 sweeps, 59.9 s | 5 sweeps, 10.2 s | 2.65e-5 |
| XX, `tol_basis` 1e-2 | 14 sweeps, 26.4 s | 5 sweeps, 10.5 s | 3.17e-7 |
| XX, `tol_trunc` 3e-3 | 14 sweeps, 28.6 s | 4 sweeps, 9.0 s | 4.8e-7 |
| XX, `tol_trunc` 1e-3 | 10 sweeps, 21.8 s | 4 sweeps, 9.1 s | 5.5e-8 |
| Heisenberg, `tol_basis` 0.1 | 7 sweeps, 12.7 s | 4 sweeps, 8.0 s | 2.0e-5 |
| Heisenberg, defaults `1e-3 / 1e-4` | 6 sweeps, 12.3 s | 5 sweeps, 10.7 s | 1.9e-9 |

On the chains the loose and medium settings now beat the legacy run on wall time (12.9 s), at
0.65-0.85x; with the fixed `etol` they did not.

**2D cylinders, derived threshold (same job).** Every adaptive run converges in 5-8 sweeps, with
the same energies as before:

| Run | fixed `etol` = 1e-10 | derived `dmrg_etol` | dE/N |
|---|---|---|---|
| square, `tol_basis` 0.1 | 40 sweeps, 114 s | 5 sweeps, 17.0 s | 1.0e-5 |
| square, `tol_basis` 3e-2 | 40 sweeps, 149 s | 6 sweeps, 23.6 s | 5.7e-7 |
| square, defaults `1e-3 / 1e-4` | 8 sweeps, 49.9 s | 6 sweeps, 34.6 s | 2.0e-9 |
| J1-J2, `tol_basis` 3e-3 | 18 sweeps, 123 s | 6 sweeps, 31.5 s | 5.8e-9 |
| J1-J2, defaults `1e-3 / 1e-4` | 7 sweeps, 44.6 s | 6 sweeps, 34.5 s | 2.7e-9 |
| J1-J2, defaults, `exact = true` | 7 sweeps, 39.4 s | 6 sweeps, 29.7 s | 2.7e-9 |

Legacy runs: square 65.4 s, J1-J2 73.2 s (dE/N ~1.4e-11, chi 512 = the cap).

## J1-J2 square cylinder 5x4, J2/J1 = 0.5, N = 20 (maxdim 512; job 17208958)

| `tol_basis` (`tol_trunc` 1e-8) | 1e-1 | 3e-2 | 1e-2 | 3e-3 | 1e-3 | 1e-4 |
|---|---|---|---|---|---|---|
| dE/N | 2.5e-5 | 9.7e-7 | 9.9e-8 | 5.2e-9 | 3.7e-10 | 1.6e-11 |
| chi | 47 | 115 | 173 | 274 | 404 | 512 (cap) |
| sweeps / wall s | 21 / 52 | 40 / 138 | 15 / 67 | 18 / 123 | 8 / 62 | 7 / 64 |

| `tol_trunc` (`tol_basis` 1e-5) | 1e-2 | 3e-3 | 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|---|---|
| dE/N | 2.9e-5 | 1.6e-6 | 2.6e-7 | 2.2e-9 | 2.9e-11 |
| chi | 46 | 99 | 156 | 342 | 512 (cap) |
| sweeps / wall s | 10 / 41 | 9 / 45 | 14 / 66 | 7 / 50 | 7 / 67 |

Legacy fixed-knob run: dE/N 1.4e-11, chi 512 (the cap), 7 sweeps, 75.2 s. The defaults `1e-3 / 1e-4`
reach **2.7e-9 at chi 321 in 44.6 s (0.59x legacy)**, and 39.4 s with `exact = true`.

13. **The frustrated model follows the same law:** `dE/N ~ 3.7e-4 tol_basis^2` and
    `~ 0.26 tol_trunc^2`. The truncation coefficient is slightly above the 0.2 in `dmrg_etol`;
    the threshold is a tenth of the predicted error, so it stays conservative.
14. **Its loose runs are the slowest to converge under the fixed `etol`** (40 sweeps at
    `tol_basis = 3e-2`, 28 sweeps at `(1e-2, 1e-3)`), which is what the derived threshold addresses.

## Summary: what to set for a ground state

| Use | `tol_basis` | `tol_trunc` | Typical dE/N (chain / 2D) | chi at N = 20, 2D | Cost vs legacy (2D) |
|---|---|---|---|---|---|
| loose: starting states, quick scans | 1e-2 | 1e-3 | 3e-7 / 3e-7 | ~150-170 | 0.33-0.39x |
| **medium: the default** | **1e-3** | **1e-4** | **2e-9 / 2-3e-9** | **~320-350** | **0.47-0.53x** |
| tight | 1e-4 | 1e-5 | 2e-11 / cap-limited | maxdim | ~1x (cap binds) |

- Both knobs act QUADRATICALLY on the energy (variational square law), across chains, a 2D
  cylinder and the frustrated J1-J2 point: `dE/N ~ (0.3-2.5)e-3 tol_basis^2 + (0.08-0.26) tol_trunc^2`.
- Keep `tol_trunc ~ tol_basis / 10`: in 2D the truncation term dominates at equal settings.
- Every run is honest. Settings that cannot be met within `maxdim` end `:budget_unmet` and floor at
  the legacy value; no run reports `:met` with an energy worse than its tolerances allow.
- With the derived stopping threshold (`dmrg_etol`) every setting converges in 4-8 sweeps; the
  fixed `etol = 1e-10` had cost the loose settings up to 40 sweeps.
- The exact local basis is 10-15% faster than the RSVD sketch at these sizes (chi <= 512), the
  same crossover as the BUG.
