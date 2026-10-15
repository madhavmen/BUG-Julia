# Implementation Plan: Adaptive CBE-Krylov with Minimal Tuning Knobs

## Goal

Make the CBE-BUG solver **self-adapting across Hamiltonians** while preserving symmetry-sector discovery at approximately single-site costs. Specifically:

1. Remove independent, model-sensitive RSVD, K/L split, Krylov, and root-truncation knobs from the normal solver API.
2. Use two RSVD tests to decide **how much to sample** and **how many discovered directions to keep**.
3. Reuse selected CBE directions during one-site K/L Krylov enrichment, and use residual tests to decide whether further Krylov directions are necessary.
4. Keep the physical input subspace intact during K/L basis construction; control temporary rank growth locally, then compress the evolved state once with a shared budget.
5. Validate accuracy, symmetry preservation, robust stopping, and runtime against the XX U(1) chain at **L = 20**.

**Out of scope for the first implementation:** rewriting the root exponential as a block/recycled Lanczos solver. Keep the current `expv` implementation and make its convergence automatic from `tol_expv`.

## 1. Target public API: three tolerances, one resource limit

```julia
cbe_bug_step!(
    psi, mpo, dt;
    tol_basis = 1e-5,   # RSVD + K/L Krylov basis quality
    tol_trunc = 1e-7,   # Final state compression
    tol_expv  = 1e-10,  # Root expv time-evolution solve
    maxdim    = 128     # Final bond dimension / resource limit
)
```

The midpoint variant should use the **same accuracy controls**. The timestep `dt`, Hamiltonian, and choice of ordinary/midpoint BUG are physical/method inputs, not tuning thresholds. An `rng`/seed and diagnostics level may be exposed for reproducibility, but are not accuracy knobs.

| User-facing setting | Responsibility | What it must NOT control |
|---|---|---|
| `tol_basis` | RSVD sampling residual (Test B), RSVD singular tail (Test A), K/L Krylov residual/closure (Tests C/D), frame projection checks | Final Schmidt compression, root expv iteration accuracy |
| `tol_trunc` | Relative cumulative physical-state compression error | Basis discovery or RSVD preselection |
| `tol_expv` | Error estimate of root `expv`/Lanczos exponential action | RSVD rank or one-site K/L Krylov basis size |
| `maxdim` | Allowed final MPS bond rank and derived temporary working-rank ceiling | A substitute for satisfying an accuracy tolerance |

**Do not add `tol_rsvd_pre`, `tol_rsvd_fnl`, `tol_krylov`, `tol_sampling`, or `tol_sector` to the public API.** The controller apportions `tol_basis` internally. If the requested accuracy is infeasible under the rank/work limits, return a clear `:budget_unmet` / retry diagnostic; do not silently claim convergence.

### Remove or internalize the existing knobs

The names below come from the current algorithm audit; exact code-level semantics of `dover` and `comp_ratio` must be verified in `cbe_expand` before the migration.

| Current parameter | New behavior | Normal public API |
|---|---|---|
| `stol_pre` | Replace with scale-aware machine-precision rank/orthogonality detection; physical importance is assessed later by residual tests | **Remove** |
| `stol_fnl` | Replace with Test A projected singular-tail rank selection | **Remove** |
| `dex` | Internal **selected expansion rank** `k_keep`, grown by Test A | **Remove** |
| `dover` | Internal **extra sketch vectors**, grown by Test B | **Remove** |
| `comp_ratio` | Internal complement/sector probe allocation and coverage policy; no hand-tuned ratio | **Remove** |
| `growth` | Replace by work-rank ceiling and explicit budget-unmet status | **Remove** |
| `krylov_basis` | Replace fixed count with Test C/D residual-driven K/L Krylov depth | **Remove** |
| `split_cutoff`, `split_maxdim` | No independent physical truncation in K/L frame construction; allow numerical rank cleanup only, and enforce derived work ceiling | **Remove** |
| `root_cutoff`, `root_maxdim` | Structural root SVD stays lossless in default method | **Remove** |
| `trunc_thresh` | Replaced by shared final compression tolerance `tol_trunc` | **Rename/replace** |
| `maxdim` | Final MPS rank cap; also used to derive a conservative work-rank ceiling | **Keep** |
| root `expv` tolerance | Driven by `tol_expv`; no separate Krylov-iteration tuning | **Replace with `tol_expv`** |

For a safe migration, the old parameter names may generate *temporary deprecation errors/warnings* with a migration hint. Do not silently continue to honor old and new cutoffs simultaneously. A fixed-parameter legacy path may remain in the test harness for benchmark comparisons, not as the default API.

## 2. One internal accuracy budget, not more knobs

Implement an internal step controller from `(tol_basis, tol_trunc, tol_expv, maxdim, |dt|, ||psi||, L)`:

```julia
struct AdaptiveBudget
    basis::Float64
    trunc::Float64
    expv::Float64
end

# The actual controller also tracks spent/estimated error,
# symmetry-sector work, the current bond, and any rank-cap failures.
```

`tol_basis` is distributed internally across the left/right sweeps, bonds, and RSVD/Krylov checks. Choose **one documented, fixed allocation rule** initially (not additional user settings). Collect the local indicators in a step-level ledger and report the maximum and aggregate indicators. For example, use a fixed allocation across A/B/C/D and an L-dependent local allowance, then benchmark whether it is overly conservative.

**Numerical caveat:** timestep-scaled CBE and Krylov residuals are leading-order / surrogate basis-error indicators, not rigorous global bounds on MPS time-evolution error. Independent randomized probes can provide probabilistic sampling estimates, not an unconditional certificate. Calibrate the policy against reference dynamics; do not label `tol_basis` a certified global error bound until a proof/validated estimator exists.

For all basis checks, consistently use an appropriate relative scale such as `|dt| / ||psi||` (or the norm of the transported local seed for one-site Krylov). Special-case `dt == 0` without division by zero.

## 3. Adaptive RSVD: sector coverage + two tests

Target routine: `cbe_expand`.

**Mandatory coverage:** enumerate *reachable and symmetry-compatible* bond charges, including currently empty charge sectors, and allocate **one probe per nonempty admissible complement sector**. Left and right probe spaces must obey their respective fusion/charge constraints. Never sample an inaccessible charge merely to meet a vector count.

Probe vectors live **inside individual charge blocks**; do not draw one unstructured random vector across all charges. Apply the local Hamiltonian/complement action with the existing one-site-like environment contractions, without constructing a dense full two-site residual matrix.

Let `G` denote the relevant complement-projected local Hamiltonian action and `QL`, `QR` the sampled left/right candidate spaces. Let

\[
M = Q_L^\dagger G Q_R, \qquad M = U\Sigma V^\dagger.
\]

### Test B: Was the important range actually sampled?

Use *independent holdout vectors*, separate from sketch construction, and check both sides. Schematically estimate

\[
\eta_{B,L} \sim \frac{|dt|}{\|\psi\|}\|(I-Q_LQ_L^\dagger)G\|_F, \quad
\eta_{B,R} \sim \frac{|dt|}{\|\psi\|}\|G(I-Q_RQ_R^\dagger)\|_F.
\]

These norms must be **estimated by matrix-free probes**, not evaluated using the full `G`. Record sector-resolved estimates and sampling uncertainty. If Test B fails, add probes specifically to the sector(s) with the largest missed residual (internal role of `dover`/`comp_ratio`). Repeat the sketch. Use a small fixed internal number of independent validation probes and verify reliability over multiple RNG seeds.

### Test A: Were enough of the discovered directions retained?

For sector-blocked singular values `sigma`, find the smallest feasible `k_keep` with

\[
\eta_A(k) = \frac{|dt|}{\|\psi\|}
\left(\sum_{j>k}\sigma_j^2\right)^{1/2}
\leq \text{allocated basis budget}.
\]

This is a **projected-Hamiltonian tail indicator**, not a Schmidt-weight cutoff. Retain the globally strongest `k_keep` CBE directions across charge blocks (preserving the correct charge labels and symmetry degeneracies).

**Decision order:** if B fails, improve discovery first. If B passes but A fails, grow `k_keep` (internal role of `dex`). If B fails for a sector that is exhausted or the work-rank cap is hit, report budget unmet / request a smaller timestep; do not accept a spurious small projected tail.

Do **not** carry holdout or unused oversampling vectors into the physical sweep frames.

## 4. Adaptive one-site K/L Krylov: two more tests

Targets: `half_left!` and `half_right!`. Their optional Krylov construction builds **basis vectors**, not an ODE solution. The root `expv` solve is separate.

After CBE final selection, map the selected bond directions to the *same one-site tensor space and charge blocks* used by the local seed `K_ex` (or the right-sweep counterpart). Then form an orthonormal combined basis, retaining old directions exactly:

\[
Q_0 = \operatorname{orth}([U_{\mathrm{old}},\; U_{\mathrm{CBE}}^{\mathrm{local}},\;K_{\mathrm{ex}}]).
\]

The mapping and index compatibility must be tested explicitly. CBE bond vectors cannot automatically be concatenated with local tensors without this transport.

### Test C: Does CBE already capture the first Krylov direction?

\[
R_0=(I-Q_0Q_0^\dagger)H_1K_{\mathrm{ex}},\qquad
\eta_C=\frac{|dt|\,\|R_0\|_F}{\|K_{\mathrm{ex}}\|_F}.
\]

If small, the first one-site Hamiltonian action introduces little or nothing new. This is a useful early-exit indicator, but **it is NOT a sufficient stopping test by itself**.

### Test D: Is the combined Krylov basis closed enough under `H1`?

A first residual can vanish while the *next* one does not. Example: `H1*e1=e2`, `H1*e2=e1+e3`, and `Q0=[e1,e2]`. Test C returns zero, but `e3` is still missing.

Therefore also measure the projected action of `H1` on the dynamically relevant current Krylov *frontier block* `V_m`:

\[
R_{m+1}=(I-Q_mQ_m^\dagger)H_1V_m.
\]

Use the same allocated `tol_basis` to decide whether that residual contributes a significant new local direction (with the appropriate timestep/seed normalization). If significant, orthogonalize the residual block within each charge sector, append it to `Q_m`, and continue. If negligible and the relevant active subspace passes the closure/contribution check, stop. Cache Hamiltonian actions and reorthogonalize as needed.

**No fixed `krylov_basis` depth and no extra public `tol_krylov`.** Krylov enrichment shares the rank/work ceiling and basis-error ledger with CBE. Test D must not force closure of every dormant, zero-amplitude/randomly sampled direction; operate on the active transported seed/selected frontier.

### Root Lanczos is still independent

At the root, run the existing matrix-free `expv(H0, S0, dt)` with `tol_expv`. If its a posteriori residual is too large, increase root Krylov iterations/restarts internally. A converged root exponential does **not** certify that RSVD/K/L found a sufficient Galerkin space. Conversely, a poor root exponential is not solved by indiscriminately increasing `dex`.

## 5. Controlled bond growth and a single final truncation

- **During K/L:** maintain the old left/right subspace, orthogonality, and symmetry; keep only CBE-selected plus validated new Krylov directions. Remove numerical linear dependence using a machine-epsilon-derived threshold. No independent physical Schmidt truncation at `split_cutoff`.
- **Working rank:** derive an internal, conservative maximum working dimension from `maxdim` and the current rank, with a small fixed allowance for temporary augmentation (tune the policy by benchmark, not with another user knob). Cap local sketch sizes and the number of retained CBE/Krylov directions. Report unmet accuracy if the cap prevents convergence.
- **At root:** keep the root structural SVD lossless (up to numerical zeros). Do not use `root_cutoff` as an extra error decision.
- **After full root evolution:** do one recursive MPS compression with `tol_trunc` and `maxdim`, tracking the cumulative squared discarded Schmidt weight when justified by the canonical one-pass contraction order. Otherwise report a conservative accumulated error estimate. Do not confuse a **maximum individual discarded weight** with a full-step compression error.
- **Midpoint:** preserve the predictor half-step's enlarged basis (`truncate=false` in the current API must not allow an untracked K/L physical truncation). Construct midpoint frames from the half-step state but project the **original** state into them before the full root solve. Check the projection error and include it in `tol_basis` accounting.

Keep the current matrix-free/symmetry-blocked Hamiltonian contractions and projected small SVDs. The intended work is proportional to the **number of candidate/validation/Krylov vectors** rather than a dense two-site SVD. One-per-sector screening has a real cost when there are many reachable sectors; benchmark `n_sectors`, `n_probes`, and `n_H1_actions` to ensure it remains competitive.

## 6. Core controller pseudocode (illustrative)

```julia
function adaptive_bond_enrichment(local, H1, dt, budget, workcap)
    # Step 1: at least one left/right sketch probe in each
    # reachable complement charge sector.
    sketch = seed_one_probe_per_reachable_sector(local)

    while true
        QL, QR = build_and_orthogonalize_sketch(sketch, local)

        # Test B: independent holdout residuals on BOTH sides,
        # reported per charge sector.
        sample_error = holdout_sampling_residual(local, QL, QR, dt)
        if sample_error > budget.rsvd_sample
            add_probes_in_worst_sectors!(sketch, workcap)
            continue
        end

        M = projected_cbe_matrix(local, QL, QR)
        # Test A: select minimum globally retained rank from
        # charge-block SVD tail. Do not retain unused probes.
        selected = select_min_rank_from_tail(M, dt, budget.rsvd_tail)
        break
    end

    Q = transport_and_combine_old_cbe_and_seed(local, selected)

    # Tests C/D: residual-aware one-site Krylov enrichment.
    # A small first residual alone is not a stopping certificate.
    Q = enrich_krylov_until_closure!(Q, H1, local.seed, dt;
                                     budget=budget.krylov,
                                     workcap=workcap)
    return Q
end
```

The `budget.rsvd_sample`, `budget.rsvd_tail`, and `budget.krylov` fields are **internally derived allocations from the single user-provided `tol_basis`**, not new public tolerances. Every loop must check for (i) numerical rank exhaustion, (ii) physical complement exhaustion, (iii) rank/work cap, and (iv) an explicit unmet-tolerance status.

## 7. Unit tests (Julia `Test`, deterministic seeds)

| ID | Test | Expected behavior |
|---|---|---|
| UT01 | **Public API cleanup:** inspect accepted kwargs / migration errors | Normal API has only `tol_basis`, `tol_trunc`, `tol_expv`, `maxdim` as tuning knobs; old names cannot silently affect results |
| UT02 | **Numerical preselection:** near-dependent vectors at two amplitude scales | Numerical QR/SVD filtering works without `stol_pre`; true small physical directions are judged by tolerance-driven residual, not arbitrary absolute cutoff |
| UT03 | **Empty-but-reachable charge sector** | One left/right probe allocated in the relevant admissible complement block; forbidden sectors are never sampled |
| UT04 | **RSVD Test B / internal oversampling** using `G_q=diag(10,9)` and an initially one-dimensional sketch | Independent residual detects missed rank and adds sector-specific probes; no full two-site matrix is needed in the production path |
| UT05 | **RSVD Test A / internal retained rank** with singular strengths `(10,9,3,0.1)` split across sectors | Smallest `k_keep` satisfying the projected-tail budget is selected; charges and degeneracies are respected |
| UT06 | **Misleading projected tail:** sketch misses a strong entire sector | Test A may pass, Test B fails, algorithm expands the correct sector rather than accepting |
| UT07 | **CBE-Krylov overlap:** `H1*Kex` already in transported `Q0` | First new residual is zero; do not add a redundant vector |
| UT08 | **Second Krylov direction missed by Test C:** `H1*e1=e2`, `H1*e2=e1+e3`, `Q0=[e1,e2]` | Test C zero but Test D detects and adds `e3` (subject to budget) |
| UT09 | **Genuinely new local Krylov direction** | New basis block is orthogonalized and has correct charge; rank grows only by independently significant directions |
| UT10 | **Old-subspace preservation and index transport** on both `half_left!`/`half_right!` | Old physical subspace preserved; orthogonality / fusion rules correct; no invalid CBE-to-one-site concatenation |
| UT11 | **Working rank ceiling / infeasible accuracy** | No unbounded growth or silent pass; explicit `:budget_unmet` with a useful diagnostic |
| UT12 | **No hidden K/L truncation; midpoint invariant** | With predictor `truncate=false`, no physical-state cutoff remains active in K/L; final-stage `S0` projects original `psi`, not midpoint `half` |
| UT13 | **Root structural SVD** | With final compression disabled, root split is lossless to floating-point accuracy |
| UT14 | **Cumulative compression** with prescribed Schmidt spectra | `tol_trunc` controls accumulated discarded norm; rank-cap violations are reported, not masked by the largest-single-SVD diagnostic |
| UT15 | **Root `expv` tolerance** on a tiny Hermitian matrix | Iterations/restarts adapt to `tol_expv` and agree with dense matrix exponential |

Use both `Float64` and complex state tensors where applicable. Tests on randomized sampling should use a fixed reproducible `rng`, plus an additional multi-seed robustness test; do not make a single random draw a deterministic mathematical guarantee.

## 8. XX U(1) integration benchmark: L = 20

### Hamiltonian and initial states

Open boundary XX chain, `J = 1`, `L = 20`:

\[
H = \frac{J}{2}\sum_{i=1}^{19}
    (S_i^+S_{i+1}^- + S_i^-S_{i+1}^+).
\]

- **Primary quench:** Neel product state `up,down,up,down,...` (10 up spins, fixed U(1) sector `N_up=10`).
- **Additional sector stress test:** domain-wall product state (10 up followed by 10 down), same total charge but different sector-discovery pattern.
- **No-growth control:** all-up state. It is annihilated by XX hopping; the controller should avoid pointless expansion and Krylov growth.

### Independent reference

Map the XX chain to free fermions and use the exact `20 x 20` single-particle evolution for local occupations. In the chosen ladder-operator convention the one-particle hopping amplitude is `J/2`. With `U(t)=exp(-im*t*h)`, the correlation matrix evolves as `C(t)=U(t)*C(0)*U(t)'` and `n_i(t)=real(C(t)[i,i])`. Optionally compare selected full-state runs to a sparse fixed-charge exact Krylov reference (`binomial(20,10) = 184756` states).

### Initial benchmark matrix

| Setting | Loose | Tight |
|---|---:|---:|
| `tol_basis` | `1e-3` | `1e-5` |
| `tol_trunc` | `1e-5` | `1e-7` |
| `tol_expv` | `1e-8` | `1e-10` |
| `maxdim` | `128` | `128` |
| `dt` | `0.05` | `0.05` |
| Final time | `0.5` | `0.5` |

These are **starting settings**, not certified error guarantees. Run at least five fixed RNG seeds and compare ordinary versus midpoint BUG.

### Integration tests and metrics

| ID | Benchmark | Pass criterion |
|---|---|---|
| IT01 | XX occupations after 1 and 10 steps | Agree with exact free-fermion occupations; tighter controls improve typical accuracy at comparable resources |
| IT02 | U(1), norm, old-subspace invariants | Correct charge sectors and fusion; norm drift consistent with solver and compression errors |
| IT03 | Multiple seeds; Neel and domain wall | No silent bad-rank decisions; reproducible diagnostics for a fixed seed |
| IT04 | All-up stationary state | Negligible needless CBE and K/L Krylov enrichment |
| IT05 | Old fixed-knob solver vs new adaptive solver | Match observable accuracy before comparing cost; measure total H applications, probes, allocation, runtime, and peak/final rank |
| IT06 | Ordinary vs midpoint | Midpoint starts full root evolution from original state and preserves basis-projection invariant |
| IT07 | Tolerance sweep over 2-3 decades | Basis/sketch/Krylov work responds smoothly; failures from `maxdim` are visible |

### Required diagnostics

```julia
(
    n_reachable_sectors,
    probes_per_sector,
    n_rsvd_holdout_probes,
    n_rsvd_selected,
    rsvd_tail_indicator_A,
    rsvd_sampling_indicator_B_left,
    rsvd_sampling_indicator_B_right,
    n_krylov_added,
    krylov_overlap_indicator_C,
    krylov_closure_indicator_D,
    n_H1_applications,
    root_expv_iterations,
    root_expv_error_estimate,
    r_old,
    r_augmented,
    r_final,
    state_projection_error,
    cumulative_discarded_weight,
    workcap_reached,
    tolerance_met
)
```

Track by bond and timestep. In the first benchmark, count how often Krylov Test C finds that CBE already captures the next one-site action, and how often Test D still needs to grow the basis despite Test C passing.

## 9. Implementation sequence and acceptance

1. **Clean API and diagnostics:** introduce the three-tolerance budget object and old-knob migration; add UT01-UT02. Do not change physics yet.
2. **Sector coverage plus RSVD A/B:** change `cbe_expand`; add UT03-UT06. Validate one-vector-per-sector behavior, independent probes, and adaptive retained rank.
3. **Residual-driven K/L Krylov C/D:** change `half_left!`, `half_right!`; add UT07-UT10. Confirm CBE directions are transported correctly, eliminate redundant Krylov enrichment, and catch second-step leakage.
4. **Rank cap and compression cleanup:** remove independent K/L/root physical cutoffs, implement common final truncation ledger; add UT11-UT14 and midpoint regression.
5. **Keep root `expv`, bind tolerance:** implement `tol_expv`-driven convergence/failure status; add UT15.
6. **XX L=20 benchmark:** run IT01-IT07 and compare to the old fixed-knob implementation **at matched accuracy**.

**Acceptance criteria:**

- Normal user API has **only three accuracy tolerances and `maxdim`**; no `stol_pre`, `stol_fnl`, `dex`, `dover`, `comp_ratio`, `growth`, `krylov_basis`, `split_cutoff`, or `root_cutoff` to tune.
- RSVD A/B both pass their internally allocated checks, or the solver returns an explicit unmet-budget status; no important reachable charge is silently excluded.
- Krylov C/D either show why enrichment is necessary or skip it; no hard-coded one-site Krylov depth is needed.
- Symmetry sectors, old-subspace preservation, and midpoint projection invariants remain correct.
- A lossless root split and a single budgeted final compression prevent hidden multiple physical truncations.
- XX L=20 observable accuracy matches the reference and stays robust across several seeds, while peak rank and contraction counts remain competitive with the existing fixed-knob solver.

**Core design principle:** `tol_basis` determines *what must be explored and retained in the trial space*; `tol_trunc` determines *what can be discarded from the evolved state*; `tol_expv` determines *how accurately the root evolution is solved*. The legacy rank/sampling/Krylov settings are consequences of those decisions, not user inputs.
