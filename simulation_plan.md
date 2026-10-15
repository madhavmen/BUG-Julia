# Implementation Plan: Adaptive CBE–Krylov Enrichment

**Goal:** Make the CBE-BUG solver automatically discover relevant U(1) symmetry sectors, reuse RSVD information during one-site Krylov enrichment, and limit bond growth without introducing many Hamiltonian-dependent parameters.

The main benchmark will be the **XX spin-½ chain with U(1) symmetry, \(L=20\)**.

## 1. Simplify the solver interface

Expose three accuracy tolerances and one resource limit:

```julia
cbe_bug_step!(
    psi, mpo, dt;
    tol_basis = 1e-5,
    tol_trunc = 1e-7,
    tol_expv  = 1e-10,
    maxdim    = 128,
    adaptive  = true
)
```

| Parameter | Controls |
|---|---|
| `tol_basis` | RSVD discovery, retained expansion rank, and additional one-site Krylov enrichment |
| `tol_trunc` | Total discarded wavefunction weight during final MPS compression |
| `tol_expv` | Accuracy of the root Lanczos/exponential evolution |
| `maxdim` | Final bond-dimension/resource limit, not an error tolerance |

Internally, make `dex`, `dover`, `comp_ratio`, and `krylov_basis` adaptive. Keep `stol_pre` as a numerical rank-deficiency safeguard and replace `stol_fnl` with a tolerance-derived selection rule.

The root-core SVD should remain lossless by default. Final state truncation should use one shared compression budget rather than independent `root_cutoff` and `trunc_thresh` decisions.

## 2. Implementation tasks

### Task 1 — Sector-aware adaptive RSVD

Modify `cbe_expand`:

1. Enumerate reachable symmetry sectors, including currently empty bond sectors with valid complement spaces.
2. Sample one random vector in each reachable sector.
3. Construct the projected Hamiltonian and perform the existing final singular-direction selection.
4. Use independent residual probes to identify sectors that need more samples (`dover`).
5. Increase retained expansion rank (`dex`) if the projected singular-value tail exceeds its budget.

Do not propagate unused random vectors into the MPS.

### Task 2 — Reuse CBE directions during one-site Krylov enrichment

Modify `half_left!` and `half_right!`.

After RSVD, transport the selected CBE directions to the compatible one-site tensor space and combine them with the old frame and \(K_{\mathrm{ex}}\):

\[
Q_0=\operatorname{orth}\left(
[U_{\mathrm{old}},\,U_{\mathrm{CBE}}^{\mathrm{local}},\,K_{\mathrm{ex}}]
\right).
\]

Compute the first one-site Krylov residual:

\[
R=(I-Q_0Q_0^\dagger)H_1K_{\mathrm{ex}}.
\]

If its timestep-scaled norm is below the allocated `tol_basis` budget, skip additional Krylov enrichment. Otherwise, append orthogonalized residual directions and continue adaptively.

Reuse computed \(H_1\) actions where possible. Keep the existing root `expv` solver unchanged.

### Task 3 — Control rank growth and truncation

- Preserve the old K/L subspaces during basis construction.
- Give CBE and Krylov enrichment a shared per-bond working-rank budget.
- Never retain every randomized probe merely because it was sampled.
- Allow temporary working dimensions above `maxdim`, but impose an internally derived working-rank ceiling.
- Perform physical-state compression after the root evolution using a cumulative discarded-weight budget.

If an accuracy target cannot be reached under the working-rank ceiling, return an explicit failure/retry diagnostic instead of silently accepting an inaccurate basis.

These changes preserve the existing separation between CBE basis construction, K/L frame building, and the root Galerkin evolution.

## 3. Unit tests to implement

Use Julia's `Test` framework, with deterministic random seeds and small synthetic block matrices for the algorithmic tests.

| Test | Setup | Expected result |
|---|---|---|
| **UT1: Sector coverage** | Symmetry-blocked input with an empty but reachable charge sector | RSVD samples at least one vector in that sector |
| **UT2: Adaptive `dover`** | Synthetic block \(G_{q=0}=\operatorname{diag}(10,9)\), initially undersampled | Independent residual detects the missing direction; an additional probe is generated in \(q=0\) |
| **UT3: Adaptive `dex`** | Projected singular strengths \((10,9,3,0.1)\) across several sectors | Retained rank increases until the tail meets `tol_basis`; selected directions have correct charge labels |
| **UT4: CBE–Krylov overlap** | \(H_1K_{\mathrm{ex}}\) entirely inside \(Q_0\) | No additional Krylov direction is generated |
| **UT5: New Krylov direction** | \(H_1K_{\mathrm{ex}}\) has a known component outside \(Q_0\) | Exactly the required independent direction is added in the synthetic example |
| **UT6: Bond growth and orthogonality** | Repeated K/L expansions | Old subspaces are preserved, \(Q^\dagger Q\approx I\), and the working-rank ceiling is respected |
| **UT7: Truncation budget** | MPS with prescribed Schmidt spectra | Cumulative discarded weight satisfies `tol_trunc`, unless an explicit rank cap makes the requested tolerance infeasible |
| **UT8: Midpoint consistency** | Midpoint BUG with nonzero timestep | Root initial tensor is projected from the original state, not the half-step state |

For UT4 and UT5, use small Hermitian matrices so the expected answer is exact. For example, with \(K_{\mathrm{ex}}=e_1\) and \(Q_0=[e_1,e_2]\), a diagonal \(H_1\) produces no new direction, whereas \(H_1=e_1e_3^\dagger+e_3e_1^\dagger\) produces the new direction \(e_3\).

Also add a `truncate=false` test verifying that a pure K/L basis-construction sweep reconstructs the original state to numerical precision.

## 4. Physical benchmark: XX chain, U(1), L=20

Use the open-boundary XX Hamiltonian

\[
H=\frac{J}{2}\sum_{i=1}^{19}
\left(S_i^+S_{i+1}^-+S_i^-S_{i+1}^+\right)
\]

with \(J=1\), using the standard spin-½ ladder-operator convention.

**Initial state:** Néel product state \(|\uparrow\downarrow\uparrow\downarrow\cdots\uparrow\downarrow\rangle\), containing 10 spin-up and 10 spin-down sites. The total U(1) charge is conserved.

This is a good benchmark because hopping immediately generates new virtual-charge sectors and nontrivial entanglement.

### Reference and test configuration

Use two benchmark regimes:

| Parameter | Loose | Tight |
|---|---:|---:|
| `tol_basis` | \(10^{-3}\) | \(10^{-5}\) |
| `tol_trunc` | \(10^{-5}\) | \(10^{-7}\) |
| `tol_expv` | \(10^{-8}\) | \(10^{-10}\) |
| `maxdim` | 128 | 128 |
| \(\Delta t\) | 0.05 | 0.05 |
| Final time | 0.5 | 0.5 |

These are initial test settings, not guaranteed sufficient tolerances.

For an independent reference, exploit the fact that the XX chain maps to free fermions. Evolve the \(20\times20\) single-particle correlation matrix and obtain exact site occupations. This is substantially cheaper than evolving the full 20-site wavefunction. For selected full-state comparisons, the fixed-\(N=10\) Hilbert space has dimension \(\binom{20}{10}=184756\), which is also feasible for a sparse Krylov reference.

### Integration tests

**IT1 — Physical accuracy:** Compare local spin occupations \(\langle n_i(t)\rangle\) against the exact XX solution after one timestep and after ten timesteps. Verify that the tight configuration improves the accuracy without pathological rank growth. Calibrate a fixed regression threshold against the reference.

**IT2 — Symmetry conservation:** Verify total particle number remains 10, that all virtual-charge fusion rules are satisfied, and that norm drift remains within the integration and compression error budget.

**IT3 — Robustness across random seeds:** Run the adaptive RSVD using at least five fixed seeds. Check that the accuracy requirements remain satisfied, even when the sampled and retained sector dimensions differ.

**IT4 — Efficiency:** Compare against the current solver at comparable physical accuracy, using fixed `dex`, `dover`, and `krylov_basis`. Measure Hamiltonian applications, peak working bond dimension, final bond dimension, and root Krylov iterations.

**IT5 — Midpoint regression:** Run both ordinary and midpoint CBE-BUG. Verify the midpoint state is used for frame construction while the original state supplies \(S_0\). Compare both against the XX reference.

## 5. Diagnostics to add

For each bond and timestep, record:

```julia
(
    n_reachable_sectors,
    n_rsvd_samples,
    n_rsvd_selected,
    n_krylov_added,
    r_old,
    r_augmented,
    r_final,
    rsvd_sampling_residual,
    rsvd_retained_tail,
    krylov_enrichment_residual,
    discarded_weight,
    n_hamiltonian_actions
)
```

These diagnostics are essential for determining whether the optimization genuinely saves work.

In particular, measure how often the CBE-selected directions already capture \(H_1K_{\mathrm{ex}}\). If that happens frequently, residual-aware Krylov enrichment should significantly reduce unnecessary basis growth.

## 6. Acceptance criteria

Consider the implementation successful when:

- The XX \(L=20\) accuracy tests pass for multiple random seeds and both tolerance profiles.
- All symmetry-sector and old-subspace preservation tests pass.
- No unnecessary Krylov vectors are added when the CBE basis already captures the tested Hamiltonian action.
- Working bond dimensions remain bounded, and the solver explicitly reports when it cannot satisfy a tolerance.
- At comparable accuracy, the adaptive implementation uses fewer total samples/retained directions or fewer local Hamiltonian applications than the fixed-parameter baseline.
- The solver uses only the three main accuracy tolerances and `maxdim` in normal operation.

## 7. Suggested implementation order

1. **Adaptive sector RSVD** — implement Task 1 and UT1–UT3.
2. **Residual-aware Krylov enrichment** — implement Task 2 and UT4–UT6.
3. **Shared truncation budget** — implement Task 3 and UT7–UT8.
4. **XX L=20 validation** — run IT1–IT5, compare accuracy and work against the existing implementation.

## 8. Key design decision

Do **not** immediately implement a new block-Lanczos solver. First reuse the RSVD-selected subspace to remove redundant one-site Krylov directions. This is a smaller change to `half_left!` and `half_right!`, keeps the existing root solver intact, and directly targets unnecessary bond growth while preserving the single-site-cost approach.

**Follow-up numerical question:** How often does the optional K/L Krylov enrichment actually find significant directions outside the CBE-selected space? Measuring that overlap should be one of the first experiments.
