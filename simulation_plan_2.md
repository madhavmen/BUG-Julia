# Adaptive CBE-BUG: Implementation and 2D Quantum-Magnet Benchmark Plan

**Status:** implementation and experiment specification (proposed changes, not yet implemented)  
**Core objective:** build a symmetry-aware, accuracy-controlled CBE-BUG solver with minimal model-specific tuning, reuse CBE directions during one-site Krylov enrichment, and demonstrate accurate real-time spin dynamics on nontrivial 2D quantum magnets.

## Executive scope and deliverables

This plan has **three distinct validation levels**. They must not be conflated:

1. **Unit tests:** small synthetic tensors and symmetry-blocked Hamiltonian actions prove that RSVD, Krylov-reuse, rank allocation, root propagation, and truncation logic work.
2. **Small-system physics validation:** exact diagonalization or accurate sparse Krylov calculations verify ground states, time correlations, symmetries, and spectral sum rules.
3. **Actual proof of principle:** produce resolved dynamical structure factors and interpretable real-space dynamics for 2D systems too large for dense exact diagonalization. Simultaneously demonstrate the behavior and computational cost of adaptive CBE-BUG against 2-site TDVP and 1-site CBE-TDVP.

The **three main Hamiltonian families** are all first-class requirements of the project:

| Family | Hamiltonian | Validation | Main proof of principle | Stretch |
|---|---|---|---|---|
| Square Heisenberg | `H = J sum_<ij> S_i . S_j`, `J>0` | 4x4, then 6x4 | **10x4 or 12x4 cylinder** | 6x6 cylinder; optional wider or periodic geometry |
| Square J1-J2 Heisenberg | `H = J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j` | 4x4, then 6x4 | **10x4 cylinder, J2/J1=0.50** | 12x4 or 6x6; selected ratios 0.40, 0.50, 0.60 |
| Kagome Heisenberg | `H = J sum_<ij> S_i . S_j`, `J>0` | 12/18 sites, then 24 | **36-site cylinder with explicit 3-site unit-cell geometry** | Longer cylinder (e.g. 48 sites); 36-site torus only if feasible |

A `10x4` cylinder is 40 spins; `12x4` is 48 spins. They are not dense-ED benchmarks. A longer width-4 cylinder offers better longitudinal momentum resolution than an 8x4 cylinder at approximately the same circumference-related MPS entanglement challenge. Wider cylinders are harder even when they contain fewer sites. The precise kagome embedding, wrapping, ordering, and accessible momenta must be recorded and checked before a 36-site spectrum is interpreted.

**Common protocol:** prepare the *same* converged CBE-DMRG ground-state MPS for each geometry, then time evolve the *same* spin-excited initial state with (i) 2-site TDVP, (ii) 1-site CBE-TDVP with RSVD, and (iii) adaptive CBE-BUG (ordinary and midpoint variants where practical).

---

## 1. Minimal public solver API and removal of old knobs

Expose three physical/numerical accuracy controls plus one resource cap:

```julia
cbe_bug_step!(
    psi, mpo, dt;
    tol_basis = 1e-5,  # CBE discovery/selection and K/L Krylov enrichment
    tol_trunc = 1e-7,  # Shared final MPS compression budget
    tol_expv  = 1e-10, # Root exponential-action accuracy
    maxdim    = 512,   # Final rank/resource cap
)
```

The timestep `dt`, real-time versus imaginary-time mode, and ordinary versus midpoint BUG are method inputs, not extra accuracy cutoffs. Preserve explicit RNG seeds and diagnostics options for reproducibility. A later non-exponential root solve can generalize `tol_expv` to `tol_root` without adding a fourth accuracy knob.

**Remove these parameters from the normal public interface:** `stol_pre`, `stol_fnl`, `dex`, `dover`, `comp_ratio`, `growth`, `krylov_basis`, `split_cutoff`, `split_maxdim`, `root_cutoff`, `root_maxdim`, and `trunc_thresh`.

| Old controls | Replacement |
|---|---|
| `stol_pre` | Internal machine-precision-scaled numerical rank/reorthogonalization guard; **not** a tunable physical cutoff |
| `stol_fnl`, `dex` | RSVD Test A chooses smallest retained expansion rank meeting `tol_basis` allocation |
| `dover`, `comp_ratio` | RSVD Test B chooses extra sector-aware sketches/probes automatically |
| `growth` | Derived work-rank cap and explicit `budget_unmet` reporting |
| `krylov_basis` | Krylov Tests C/D decide whether and how far to enrich in the K/L sweeps |
| `split_cutoff`, `split_maxdim` | Preserve old K/L frames; only remove numerical linear dependence; work cap derives from `maxdim` |
| `root_cutoff`, `root_maxdim` | Root split is structural and lossless by default |
| `trunc_thresh` | Single final compression controlled by `tol_trunc`; report cumulative error indicators |

Do not retain old knobs as undocumented independent tolerances. If backward compatibility is needed, provide migration/deprecation warnings and an explicitly labeled legacy test mode.

**Important:** internal allocations for Tests A/B/C/D should be derived by a documented fixed policy from `tol_basis`, elapsed step size, state norm, number of bonds, and symmetry blocks. These sub-allocations are *not* new public knobs. They are error indicators (some probabilistic or first-order), not automatically rigorous global solution-error bounds.

---

## 2. Task 1 - Sector-aware adaptive RSVD (Tests A and B)

**Code location:** `cbe_expand` and its symmetry-blocked sketch/final-selection helpers.

**What exists according to the algorithm audit:** randomized complement-projected Hamiltonian sketches, separate allowed symmetry sectors, left/right candidate orthogonalization, a projected Hamiltonian SVD, direct-sum augmentation of the original bond frame, optional `comp_ratio` sampling. CBE selects a *basis*, not an updated state. The code-level mapping of `dover` and `comp_ratio` to sketch sizes still needs checking before rewriting allocation.

### Mandatory algorithm

1. Enumerate quantum-number sectors *reachable through valid local fusion and MPO charges*, including currently empty but accessible virtual-bond sectors. Exclude sectors with zero admissible complement dimension.
2. Draw **at least one vector within each reachable complement block** for left and right sampling, subject to block compatibility. This is sector coverage, not a retained-rank promise.
3. Apply the complement-projected Hamiltonian action through existing environment/tensor contractions; do not form a dense global two-site matrix.
4. Orthonormalize against old frames and candidate vectors, preserving U(1) charges and discarding only numerical null directions.
5. Use Test B first to decide whether the *candidate space* missed significant Hamiltonian-action directions. Target extra probes to unresolved charge blocks.
6. Once Test B passes, use Test A to retain only important directions from the small symmetry-blocked projected SVD. Append those selected directions to the old bond frame by direct sum.
7. Never permanently propagate unused random/validation probes into K/L frames.

### Test B - Sampling/range residual: adapt *exploration*

For a complement-projected action `G` and sampled spaces `Q_L`, `Q_R`, estimate by **independent holdout probes**, on both sides,

```math
eta_B,L ~ |dt| ||(I-Q_L Q_L^dagger) G|| / ||psi||,
eta_B,R ~ |dt| ||G (I-Q_R Q_R^dagger)|| / ||psi||.
```

These are conceptual norms; actual matrix-free randomized estimators must specify normalization, variance/confidence, and the number of independent holdouts. Large residual in charge sector `q` -> add sketch directions **in `q`**. A nearly zero tail inside the sampled subspace is not evidence that unsampled sectors are irrelevant.

### Test A - Projected singular-value tail: adapt *retention*

For the projected operator `M = Q_L^dagger G Q_R`, singular values `sigma_j` and global candidate retention rank `k`, use

```math
eta_A(k) = |dt| sqrt(sum_{j>k} sigma_j^2) / ||psi||.
```

Select the **smallest** symmetry-consistent `k_keep` that meets the internally allocated basis budget. Invariant: the old bond subspace remains represented exactly (up to numerical orthogonality error).

**Stopping/failure:** both A and B must satisfy the budget. If the complement is exhausted or working-rank cap blocks convergence, return a recorded `budget_unmet` status; do not silently accept. In the time-dependent 2D models, do not equate a large projected Hamiltonian singular value with a Schmidt weight.

**Required telemetry:** reachable/occupied charge sectors; candidate and holdout probes per sector; extra probes added; retained directions by charge; Tests A/B and their budgets; old/selected/working rank; local `H*Theta` contraction counts; numerical orthogonality.

---

## 3. Task 2 - Reuse CBE directions during one-site Krylov enrichment (Tests C and D)

**This task is mandatory, not an optional future optimization.** It is a distinct deliverable after Task 1, and must be implemented for *both* `half_left!` (K sweep) and `half_right!` (L sweep).

### Why it matters

The audit describes two enlargements in K/L sweeps:

- CBE-RSVD selects directions from the complement-projected Hamiltonian action.
- Optional one-site Krylov constructs `K_ex, H1*K_ex, H1^2*K_ex, ...` (and the right-sweep analogue).

These two mechanisms can rediscover overlapping local directions. Keeping their independent fixed ranks can cause unnecessary **temporary and final bond growth**, orthogonalization, environment-updating work, and root-expv cost.

We therefore want **one joint basis**, not `CBE + fixed number of Krylov vectors`.

### Implementation actions

1. Extend `cbe_expand` to return the **selected** charge-resolved CBE directions in a reusable form (plus minimal gauge/index metadata); do not return or reuse raw random test vectors.
2. In `half_left!`, transport/lift those *bond* directions into the same **one-site tensor coordinate system** as `K_ex` and `H1*K_ex`. Do the analogous operation in `half_right!` for `B_ex`. Verify physical, bond, MPO-environment, gauge, and symmetry indices; the arrays cannot simply be concatenated in their original forms.
3. Construct the initial combined orthonormal one-site basis, schematically

   ```math
   Q0 = orth([U_old, lift(U_CBE, local_frame), K_ex]),
   ```

   with symmetry-blocked orthogonalization and explicit old-subspace preservation.
4. Compute/capture `A0 = H1*K_ex` with the existing one-site Hamiltonian contraction. **Reuse cached action tensors when valid**, but do not assume the earlier two-site CBE `H*Theta` contraction equals `H1*K_ex`.
5. Apply Test C to measure whether CBE already represents the **first** Krylov direction. Skip the redundant addition if its residual is negligible.
6. Apply Test D to the **active Krylov frontier** to check higher-order leakage; even if Test C passes, a later Krylov direction may still be missing. Add only new independent directions that pass the budgeted test.
7. Stop adaptively when the contributions/closure are sufficient or return an explicit rank-cap failure. Propagate only the chosen joint basis to the carry and root environments.
8. Keep the **root** Lanczos `expv` separate. Task 2 enriches *basis frames* in the K/L sweeps; it does not introduce an augmented block-Lanczos time integrator at the root.

### Test C - First-action overlap

For the combined one-site frame `Q0`, local effective Hamiltonian `H1`, and local seed `K_ex`, measure

```math
R_C = (I-Q0 Q0^dagger) H1 K_ex,
eta_C = |dt| ||R_C|| / ||K_ex||.
```

A small `eta_C` means that appending the **first** Hamiltonian-action vector would add little new information. This is an early-exit/overlap diagnostic, **not** by itself a proof of Krylov closure.

### Test D - Krylov-frontier leakage and stopping

For the current orthonormal joint basis `Q_m` and the dynamically active frontier `V_m`, evaluate

```math
R_D = (I-Q_m Q_m^dagger) H1 V_m.
```

Use a timestep-scaled contribution/closure indicator derived from `tol_basis` to decide whether to append orthogonalized `R_D` directions. Advance the frontier and continue until the contribution falls below budget or a work/rank limit is reached. In a multi-site wavefunction, avoid forcing closure of arbitrary dormant random vectors. Store which terms were already captured by CBE vs actually required Krylov augmentation.

**Important nontrivial test:** let `H1*e1 = e2`, `H1*e2 = e1+e3`, and `Q0 = span{e1,e2}` with seed `e1`. The first residual `R_C` is zero; Test D must still discover `e3` through action on the active `e2` frontier. A design that stops after Test C alone is incorrect.

### Why this must be shared with the RSVD budget

Use the **same `tol_basis`** to allocate RSVD Tests A/B and one-site Krylov Tests C/D. Increasing CBE retained rank can make subsequent one-site Krylov vectors redundant. Conversely, if CBE cannot economically capture enough of `H1` action, Krylov may add the missing directions. The joint working rank, not separate `dex` and `krylov_basis`, determines how much the root space grows.

### Task 2 unit tests and ablation

| ID | Setup | Expected result |
|---|---|---|
| K1 | Selected CBE direction maps into a local tensor charge block | Correct shape, charge, gauge, and frame transport in both sweeps |
| K2 | `H1*K_ex` fully in combined `Q0` | Test C ~ 0; no duplicate vector appended |
| K3 | `H1*K_ex` orthogonal to `Q0` | First Krylov residual added exactly once |
| K4 | Test C vanishes but second active Krylov direction escapes | Test D adds missing `e3` in the example above |
| K5 | Same candidate occurs from CBE and Krylov | Final rank increments only by independent directions |
| K6 | Sector-coupled local operator | Added directions have valid U(1) labels; old frame preserved; left and right cases agree |
| K7 | Vary `tol_basis` and working-rank cap | Residual decreases as available work increases; cap failure is explicit |
| K8 | Turn reuse off in **benchmark-only ablation** | Physical errors agree at high accuracy; count added directions, `H1` actions, root rank and wall time separately |

**Task 2 deliverable:** a per-bond trace

```text
D_old -> D_CBE_selected -> D_joint_Krylov -> D_after_final_compression
```

plus Tests C/D residuals and one-site Hamiltonian actions saved relative to the legacy fixed-`krylov_basis` implementation. This should be featured in the main CBE-BUG performance plots, not just in test logs.

---

## 4. Task 3 - Preserve sweep frames, unify truncation, and constrain working rank

- K/L sweeps build bases; they do not integrate the state. Keep old subspaces and the selected CBE/Krylov directions. Remove only numerical redundancy during local frame splits, not physical state weight with an independent `split_cutoff`.
- Derive a **temporary working-rank ceiling** from final `maxdim` and the algorithm's staging requirements; it may exceed final `maxdim` but must remain bounded. If inadequate, report an unmet accuracy target, rather than silently dropping old/new directions.
- Keep the root structural SVD lossless by default. Do not use another tunable `root_cutoff`.
- After root evolution, do one budgeted final MPS compression using `tol_trunc` and `maxdim`.
- Track the **sum of absolute squared discarded norms** in a compatible orthogonal/canonical one-pass compression; if conditions are not met, use a conservative accumulated bound. Do not treat the largest local discarded weight as the global error.
- For midpoint BUG, an intermediate `truncate=false` half-step must not hide physical truncation in the K/L splits. Build midpoint frames from the predictor, but project the **original** `Y0` into those frames before the full-step root solve.

**Important audit detail:** the existing reported `discarded=max(w_root,w_close)` is a worst single-SVD relative discarded weight; it is **not** a cumulative step error. Preserve it for legacy comparison but add bondwise and accumulated diagnostics.

---

## 5. Task 4 - Ground-state preparation and consistent spectroscopy

### Ground states

Use CBE-DMRG to prepare `psi0` and `E0` for all models, with independently converged energy, variance and spin correlations. Use several initial MPS guesses for strongly frustrated models and report suspected metastability. The same adaptive RSVD discovery and selected-direction reuse ideas may be shared with the ground-state optimizer, but its **local Lanczos stopping criterion is a Ritz/eigenvector residual**, not the BUG root exponential error. Do not assume the ground-state code already implements Task 2; add an independent transport/overlap check if reusing the kernels there.

### Real-time excitation and observable

For the longitudinal dynamical spin structure factor,

```math
phi_q(0) = S^z_q psi0,
phi_q(t) = exp[-i (H-E0) t] phi_q(0),
C_q(t) = <phi_q(0) | phi_q(t)>,
S^{zz}(q,omega) = Fourier_t[C_q(t)]
```

where `S_q^z = (1/sqrt(N)) sum_j exp(i q.r_j) S_j^z`, with boundary-appropriate spatial transform and kagome three-sublattice offsets. Start with `Szz`, since it preserves total U(1) charge. Add transverse `S^{+-}` as a symmetry-changing test after validation. Alternatively evolve a local excitation and reconstruct space-time correlators; verify boundary reflections and the spatial Fourier convention.

**Ensure common conditions** for all three time integrators: identical lattice ordering, MPO, ground state, excitation, charge sector, time grid, total simulation time, momentum choices, window/broadening, and resource ceilings. A momentum-frequency plot is *not* an algorithmic benchmark unless those are matched.

---

## 6. Task 5 - Unit tests and small-cluster physics validation

### Numerical unit tests (Julia `Test`, deterministic seeds)

| ID | Target | Acceptance condition |
|---|---|---|
| U01 | API and old knobs | Only three accuracy tolerances + `maxdim` in normal mode; no hidden `stol_pre`/`dex` etc. |
| U02 | Numerical candidate rank | Scale-aware QR/SVD null detection, no physical cutoff in K/L |
| U03 | Sector coverage | Sample currently empty but reachable U(1) sectors; reject forbidden charges |
| U04 | RSVD Test B | Missed singular direction/sector is detected by independent probes; targeted resampling |
| U05 | RSVD Test A | Select smallest rank matching projected tail budget across charge blocks |
| U06 | Misleading projected tail | Test A can pass with incomplete sketch but Test B fails; solver does not accept |
| U07 | Left/right frame preservation | Old subspaces, canonical orthogonality, and symmetry remain valid |
| U08 | Task 2 Krylov Test C | No duplicate direction if CBE already contains first action |
| U09 | Task 2 Krylov Test D | Detect missing `e3` after a zero first residual |
| U10 | Task 2 mapping/gauges | Selected bond CBE vectors map correctly to both one-site K/L spaces |
| U11 | Task 2 contribution/rank accounting | Difference between CBE-selected and new Krylov directions is logged exactly |
| U12 | Rank cap and errors | `budget_unmet` on insufficient work rank, no silent tolerance pass |
| U13 | Single final compression | Cumulative discarded-norm estimator behaves as expected on prescribed Schmidt values |
| U14 | Root split and `truncate=false` | Lossless root factorization and no hidden half-sweep state truncation |
| U15 | Midpoint invariant | Midpoint frames originate from predictor, root initial state from original `Y0` |
| U16 | Root `expv` | Tiny Hermitian root matches dense matrix exponential and tolerance-response test |
| U17 | Lattice/MPO bonds | Independent edge-list checks for nearest-neighbor, J2 diagonals, kagome triangles, cylinder and torus wrap |
| U18 | Small-system operator consistency | U(1), Hermiticity, energies, and Hamiltonian actions against explicit small dense/sparse matrices |
| U19 | Fourier/source normalization | `C_q(0)=<psi0|S_-q^z S_q^z|psi0>` and correct kagome sublattice positions |
| U20 | Short-time expansion and sum rules | Correlator derivatives and integrated spectral weight agree with independent reference |
| U21 | Diagnostics schema/seed | Same RNG seed reproduces sector probes/selection; logs distinguish predictor and final sweeps |
| U22 | Order checks | Ordinary vs midpoint BUG error-versus-dt with spatial errors suppressed; do not assume second order without measurement |

### Physics-validation gates

- **Square Heisenberg and J1-J2:** `4x4`, then `6x4` where sparse reference calculations remain practical.
- **Kagome:** 12/18-site geometries with exact references; use 24 sites as a bridge where computing resources permit.
- Compare ground-state energy and variance, full complex `C_q(t)`, `S(q,omega)`, norm, energy drift, total `S^z`, and equal-time spectral-weight sum rules.
- Test decreasing time steps and tightening `tol_basis`, `tol_trunc`, `tol_expv` independently. Use multiple fixed RSVD seeds to test robustness.
- Treat these as **validation**, not headline proof-of-principle results.

---

## 7. Task 6 - High-quality 2D magnet proof-of-principle simulations

### Spectral-quality goals (initial proposals)

| Setting | Pilot validation | Main spectra | Stretch spectra |
|---|---|---|---|
| Time step `dt*J` | 0.05, check 0.025 | 0.05, representative checks at 0.025 | 0.025-0.05 |
| Maximum time `tmax*J` | 4-8 | **20** | **30** |
| Gaussian damping `w(t)=exp(-(eta*t)^2/2)` | `eta/J ~ 0.3-0.5` | **eta/J = 0.15** | **eta/J = 0.10** |
| Finite-window frequency scale | `2*pi/tmax` | ~0.31 J | ~0.21 J |
| Gaussian spectral FWHM | `2.355*eta` | ~0.35 J | ~0.24 J |
| Rank settings | resource ladder, not assumed converged | e.g. `maxdim=256,512,1024` as available | raise only if accuracy warrants |

These are targets, not promised spectral precision or costs. Zero-padding smooths images but does **not** improve intrinsic resolution. Long-time local-source correlations may reach open boundaries and reflect; track propagation versus system length and assess genuine finite-size effects. Avoid calling finite-cluster spectral lines a thermodynamic continuum without finite-size/broadening checks.

### Priority runs

1. **Square Heisenberg, 10x4 or 12x4 cylinder:** magnon dispersion, especially modes near accessible representatives of `(pi,pi)` and `(pi,0)`; `S(q,omega)` heatmap, momentum cuts, and real-space spin propagation.
2. **Square J1-J2, 10x4 cylinder, `J2/J1=0.50`:** compare distributed/broad spectral response and entanglement-driven bond growth. Stretch: ratios 0.40 and 0.60, plus a longer or wider cylinder.
3. **Kagome, 36-site cylinder:** check broad low-/high-energy response, plot actual cluster-compatible q path and fixed-q spectral cuts, respect 3-site basis. Verify convergence with a 24-site pilot and one longer/wider shape when possible.
4. **Stretch widths/boundaries:** 6x6 square cylinder; 36-site kagome torus only if cost allows. Do not make a large torus essential to the deliverable.

### Essential physics plots

- `S^{zz}(q,omega)` maps for each method on *identical axes, momentum sets, color scales and damping*; difference map versus an adequately converged numerical reference where available.
- Fixed-q line shapes with peak positions, widths, spectral weight and error bars or numerical uncertainty.
- Real/imaginary parts of `C_q(t)` through the full accepted time interval; optional `|C(r,t)|` propagation heatmap following a local spin quench.
- Check `int d(omega) S^{zz}(q,omega) = <psi0|S_-q^z S_q^z|psi0>` subject to normalization, connected subtraction, and finite-window/broadening effects.

---

## 8. Required diagnostic logging and CBE-BUG performance plots

**Do not rely on a final `discarded` scalar or the overall max bond dimension alone.** Log adaptive decisions per bond and timestep, with midpoint predictor/final phases separated.

### Per-bond, per-step output schema

```julia
(
    step, time, dt, sweep, bond, mode,
    r_old, r_cbe_selected, r_joint_krylov, r_final,
    n_reachable_sectors, n_new_sectors,
    probes_per_sector, holdouts_per_sector,
    selected_cbe_per_sector,
    rsvd_tail_A, rsvd_sample_B_left, rsvd_sample_B_right,
    krylov_first_overlap_C, krylov_frontier_leakage_D,
    n_krylov_added, n_Htheta_actions, n_H1_actions,
    root_expv_iterations, n_root_H0_actions, root_expv_error,
    w_root, w_close_max, discarded_per_bond, discarded_cumulative,
    psi_projection_error, energy_drift, norm_drift, sz_error,
    elapsed_rsvd, elapsed_krylov, elapsed_root, elapsed_compress,
    peak_memory, tolerance_met, workcap_reached
)
```

Inspect existing instrumentation before adding duplicative counters. `n_Htheta_actions`, `n_H1_actions` and `n_root_H0_actions` are different workloads; report them **separately** rather than treating every operator call as identical cost. Count root `expv` work and midpoint predictor work fully. Use compact storage (CSV summaries + Arrow/HDF5/etc. bondwise logs) and save raw correlator timeseries.

### Mandatory numerical-performance figures

1. **Bond growth vs time:** `max_i D_i(t)` for all three methods; for BUG overlay maximum augmented working rank and final rank. Include `D_i(t)` heatmaps across bond and time.
2. **Where rank comes from:** plots of `r_old -> r_cbe_selected -> r_joint_krylov -> r_final` at representative bonds; CBE versus truly new one-site Krylov directions over time.
3. **RSVD efficiency:** samples, holdouts and retained directions by sector; Test A tail and Test B left/right residuals versus their allocated tolerances; fractions of newly activated sectors.
4. **Krylov reuse:** Test C/D residuals and number of H1 actions; compare reuse ON to a **benchmark-only reuse-OFF ablation**, at matched physical error.
5. **Discarded weight:** max local weight (`discarded` legacy), per-bond discarded weight and cumulative compression indicator, each against time; mark rank-cap and tolerance failures.
6. **Accuracy:** correlator differences `C_q(t)-C_ref(t)`, integrated spectral and peak-position errors, norm/energy/U(1) conservation, static sum-rule residual.
7. **Cost-vs-accuracy:** wall time, peak memory, and stage-by-stage runtime against observed physical error; compare at equal `maxdim` *and* at matched accuracy.

### Matched-error performance protocol

- Same initial ground-state MPS, geometry, MPO, excitation, momentum conventions, and spectral window for all three methods.
- Keep dt/tmax/broadening fixed for direct comparison; also run separate timestep convergence ladders.
- Report equal-cap performance **and** physical-error-matched performance. Identical `maxdim`/nominal tolerances alone do not establish efficiency.
- Where dense ED is unavailable, create a converged higher-rank reference from at least two independent methods/settings and report its uncertainty. Avoid normalizing relative errors by a near-zero correlator.
- Include 5 reproducible seeds on small stochastic tests and several seeds on at least one challenging trajectory. Show median/range where sampling influences rank and cost.
- Preserve legacy fixed-`dex`/`krylov_basis` as an **ablation** only, not as the user-facing solver API.

---

## 9. Development sequence with blocking gates

| Phase | Mandatory implementation | Tests / deliverable | Gate to proceed |
|---|---|---|---|
| 0 | Baseline counters and 3-tolerance API | U01-U02, diagnostics inventory | No hidden conflicting truncation knobs |
| 1 | **Task 1: sector-aware RSVD Tests A/B** | U03-U06, sector plots on XX U(1) L=20 | No unsampled reachable-sector failure, explicit rank-cap errors |
| 2 | **Task 2: CBE-to-one-site-Krylov reuse, Tests C/D** in `half_left!` and `half_right!` | K1-K8, U08-U11; before/after rank and H1-action logs | Correct transport, old-frame preservation, independent-direction accounting |
| 3 | **Task 3: unified K/L/frame and final compression policy** | U07, U12-U15; midpoint invariant | No hidden sweep truncation, cumulative discarded-weight accounting |
| 4 | Root and order validation | U16, U22; XX L=20 and 4x4 square | Accuracy, symmetry and observed order verified, not assumed |
| 5 | **2D small-cluster physics validation** | U17-U20; 4x4 square/J1-J2 and kagome 12/18 | `C_q(t)` and sum rules agree with reference |
| 6 | **2D main proof of principle** | Square 10/12x4, J1-J2 10x4, kagome 36 cylinder | Resolved `S(q,omega)` and full algorithmic performance plots |
| 7 | Stretch and extension | 6x6 square; longer kagome; higher order / imaginary time as resources permit | Clear new resolution/convergence insight, not site count alone |

## 10. Future extension: imaginary-time and open-system BUG

This is a **later stage**, after stable real-time benchmarks and verified CBE-Krylov reuse. The general BUG basis machinery and three-tolerance philosophy should be reusable, but real-time root `expv` and imaginary-time/open-system root propagation are not identical.

- **Imaginary time:** use an appropriate shifted dissipative generator `-(H-E_shift)` with normalization, energy/variance checks and a root operator that is stable for long imaginary steps. Benchmark existing exponential action against A-stable trapezoidal and L-stable rational alternatives (e.g. SDIRK2), potentially with logarithmic step panels. A stable root solve does not certify the *basis* error: estimate the physical-space residual and enrich using CBE when needed. Rename `tol_expv` to `tol_root` if multiple root methods are supported, keeping only three user-facing accuracy controls.
- **Ground-state search:** share RSVD/Krylov-reuse kernels with CBE-DMRG where coordinate compatibility holds, but use an eigenvector/Ritz residual for local ground-state convergence.
- **Open systems:** quantum trajectories first (effective non-Hermitian evolution and jumps); generic non-Hermitian root operators may need Arnoldi rather than Hermitian Lanczos. Direct density-operator evolution introduces trace/positivity requirements and local `d^2` operator-space dimensions.

### Final definition of success

A successful project delivers **(i)** a robust minimal-knob CBE-BUG solver, **(ii)** a tested and *explicit* reuse of CBE directions in **both** one-site K/L Krylov enrichments, **(iii)** high-quality physical `S(q,omega)` plots on the square, square J1-J2 and kagome models, and **(iv)** detailed RSVD sampling, Krylov-reuse, bond-growth, discarded-weight, accuracy, memory and runtime evidence against two-site TDVP and one-site CBE-TDVP at matched physical quality.
