# Implementation plan: general-order Krylov-closure CBE-BUG (real time)

**Scope:** Extend the *already implemented* adaptive, symmetry-aware CBE/RSVD solver with repeated K/L enrichment sweeps driven by new Krylov frontiers. **Do not redo** the unified RSVD selection or reintroduce `dex`, `dover`, `stol_pre`, `stol_fnl`, `krylov_basis`, etc. as public knobs. Validate with the **open-boundary XX U(1) chain, L=16, domain-wall initial state**. Imaginary time and open systems are **out of scope**.

**Claim to establish, not assume:** The code supports a requested temporal order `p` (initially 1–4) when its combined Galerkin space captures the required global Krylov moments to sufficient accuracy. Repeating local `H1` actions alone does **not** certify higher order. A finite tolerance, truncated MPS, or rank ceiling may cause an error floor.

## 1. Minimal interface and invariants

```julia
cbe_bug_step!(psi, mpo, dt;
    order = 4,             # method choice, not another tolerance
    tol_basis = 1e-7,      # existing RSVD A/B + K/L Krylov closure C/D
    tol_trunc = 1e-10,     # final root_out / recursive MPS compression
    tol_expv = 1e-11,      # final root exponential action
    maxdim = 128           # resource cap
)
```

Use the repository's existing keyword names if they differ; keep only these **three accuracy tolerances** and `maxdim`. Internal allocations may depend on `order`, `dt`, and step budget, but are **not new public knobs**. Preserve the original MPS subspace and U(1) charge labels. Keep the K/L sweeps as basis constructors; there is **one final full-step root exponential**. Return `:budget_unmet` (or the repo's equivalent) if the requested order/accuracy cannot fit the working-rank cap.

## 2. Mathematical order target

Set `A = -im * H`, `P = P(W,Z)`, and `B = P*A*P`. The original state must satisfy `P*psi = psi`.

For temporal order `p`, require the final trial space to reproduce the needed global Krylov moments:

\[
P\psi=\psi, \qquad (PAP)^j\psi \approx A^j\psi,\quad j=1,\ldots,p.
\]

A practical way to check this is to construct **projected root Krylov frontiers** `v_j` (embedded as MPS using `W`, `Z`) and estimate the outside-space action

\[
d_j=(I-P)A v_j,\qquad j=0,\ldots,p-1.
\]

For exact moment closure, a single accurately computed root exponential has local error `O(|dt|^(p+1))` and global order `p`. For approximate/randomized closure, use **time-weighted** frontier leakage and/or the integrated defect of the projected root trajectory,

\[
\eta_{\mathrm{defect}}=\int_0^{|dt|}\|(I-P)A\, e^{sB}\psi\|\,ds,
\]

estimated using a few internal quadrature points and sector-aware matrix-free contractions. This is a useful unitary-evolution error indicator; the randomized version is an **estimate**, not a deterministic proof. A first-root-Krylov residual of zero is **not** sufficient unless later frontier actions also close. Recompute the root operator/frontiers whenever the frames change.

## 3. Small implementation sequence

1. **Frontier abstraction, without new knobs.** Add an internal `KrylovFrontier` (or adapt existing root tensor objects) that carries coefficients in the current root basis, U(1) sector metadata, and references to cached environments. Allow `cbe_expand`/the K/L augmentation helpers to act on an **arbitrary frontier MPS**, not only the original `psi` or its local K/L carry. Preserve backward compatibility for `order=1`.
2. **Outside-space action.** Implement `estimate_global_complement_action(frontier, W, Z, mpo)` using the existing sector-resolved, matrix-free CBE contractions; start with a slow exact MPO-apply/projection reference *only in tests*. Compare left/right sampled residuals with exact small-system residuals. Reuse the already implemented **RSVD Test B (missed range)** and **Test A (retained singular tail)** for each new frontier.
3. **Repeated K/L closure sweeps.** Initialize the old state frames. Build projected Krylov frontiers up to the requested `order`; for each significant `(I-P) A v_j`, run left/right CBE sweeps and append **only validated independent directions**. Reorthogonalize with old frames, transport gauges/carries, preserve U(1), and rebuild root environments and Krylov frontiers after augmentation. Continue until every relevant order-frontier / integrated-defect estimate fits the *internally allocated* `tol_basis` budget or a derived working-rank limit is reached. Do **not** propagate all oversampling vectors. Keep an explicit termination condition and log any unmet budget.
4. **Single root solve and truncation.** Project the **original** initial MPS into the final frames; run existing `expv` for the **full** `dt` at `tol_expv`. Split the root structurally (without an independent root cutoff); compress `root_out` once using `tol_trunc` and `maxdim` with the existing cumulative discarded-weight ledger. Avoid early physical truncation in K/L, beyond machine-precision redundancy removal.
5. **Instrument the new work.** At every closure sweep/bond record `order`, `n_passes`, frontier degree, Test A/B residuals, outside-space frontier residuals, integrated-defect estimate, number of CBE samples/selected directions, new K/L Krylov directions, newly occupied charge sectors, `D_before/D_work/D_final`, `maxdim` hits, root `expv` iterations, and per-bond/cumulative `root_out` discarded weight. Count one-site MPO contractions and root actions separately; track wall time and allocations.

### Pseudocode (design only)

```julia
W, Z = frames_containing_original_state(psi)
while true
    S0 = project_root(psi, W, Z)
    B  = root_generator(-im, mpo, W, Z)
    frontiers = projected_krylov_frontiers(B, S0, order)
    missing   = test_global_frontier_leakage(frontiers, W, Z, mpo, dt)
    if order_aware_basis_budget_met(missing, tol_basis, order, dt)
        break
    end
    W, Z = enrich_KL_from_frontiers!(W, Z, frontiers, missing;
                                     reuse_existing_rsvd_tests=true)
    check_orthogonality_symmetry_and_workcap!(W, Z)
end
S1 = expv(root_generator(-im, mpo, W, Z),
          project_root(psi, W, Z), dt; tol=tol_expv)
root_out = assemble_losslessly(W, S1, Z)
truncate_budgeted!(root_out; tol=tol_trunc, maxdim=maxdim)
```

A new `order` API is justified; **do not expose** `n_passes`, per-frontier tolerances, or RSVD sampling sizes. At finite tolerances, do not claim certified order until the tests below demonstrate it.

## 4. Unit tests (write these before touching the core sweeps)

| Test | Fixture | What must pass |
|---|---|---|
| **UT1 — Frontier genuinely advances** | Small 4-/5-level Hermitian hopping chain with `H*e1=e2`, `H*e2=e1+e3`, etc. | Pass 1 discovers `e2`, next pass `e3` rather than repeatedly rediscovering `e2`; no redundant columns retained. |
| **UT2 — Outside-space action** | L=4–6 XX, sparse exact `H` and a deliberately incomplete projector | Matrix-free/sampled `(I-P)Av_j` agrees with exact complement action within justified stochastic tolerance; both left/right sectors tested. |
| **UT3 — Order moments** | Small exactly known `A`, `psi`, nested projectors | If all frontiers through `p-1` close, projected exponential reproduces `A^j psi` for `j<=p`; missing higher frontier fails the order test. |
| **UT4 — Artificial projected Krylov breakdown** | Projector that closes under `B=PAP` but not `A` | Root Krylov breakdown does **not** cause false acceptance; full-space residual triggers another K/L pass. |
| **UT5 — Frame invariants** | Both left and right sweeps with U(1) charge blocks | Old subspace preserved; isometric frames; correct charge labels/carries/environments after every pass. |
| **UT6 — No-growth case** | Exact eigenstate or `H*psi=0` | No unnecessary augmentation; exact root phase; no rank growth (up to numerical noise). |
| **UT7 — Rank cap and tolerance** | Deliberately undersized `maxdim` | Finite termination, explicit `budget_unmet`, no silent success or oversampling leakage into final MPS. |
| **UT8 — Root_out accounting** | Prescribed singular spectra | One final compression, `tol_trunc` respected when feasible, cumulative weight distinct from max local discarded weight. |
| **UT9 — Backward compatibility** | Existing order-1 CBE-BUG and midpoint fixtures | With `order=1`, reproduce legacy reference within expected noise; unified RSVD A/B tolerances unchanged. |

Use deterministic RNGs for algorithm tests; run a small multi-seed stress test to avoid confusing lucky sampling with correctness.

## 5. L=16 XX U(1) domain-wall integration tests

**Model:** Open boundary, `J=1`,

\[
H = \frac{J}{2}\sum_{j=1}^{15}
  (S_j^+S_{j+1}^- + S_j^-S_{j+1}^+),
\qquad
|\psi_0\rangle = |\uparrow\rangle^{\otimes 8}|\downarrow\rangle^{\otimes 8}.
\]

Preserve total `N_up=8`. Its fixed-number sector has dimension `binomial(16,8)=12870`, small enough for an **independent sparse exact reference** (exponential action on the state); cross-check local occupations with the free-fermion `16 x 16` correlation-matrix evolution. Align spin/hopping conventions before comparison.

| Test | Setup | Acceptance |
|---|---|---|
| **IT1 — Exact real-time state** | `t=0.5, 1.0, 2.0`, orders 1, 2, 4 | Complex wavefunction (phase-aligned), site occupations, domain-wall magnetization/current match sparse exact reference within reported basis/trunc/root budgets. |
| **IT2 — Temporal convergence** | `T=0.8`, `dt=0.2, 0.1, 0.05, 0.025`, `order=1,2,3,4` | With compression effectively disabled, sufficiently large working space and tight `tol_basis/tol_expv`, observed wavefunction error decreases at least as `O(dt^p)` in a resolved asymptotic regime (or faster if the Krylov space is effectively exact). Do **not** infer order from infidelity alone. |
| **IT3 — Conservation and physical plots** | Domain-wall evolution to `t=2` | Norm, energy, `N_up`, orthogonality maintained; plot `n_j(t)`/magnetization front and compare against exact result. |
| **IT4 — Rank adaptation** | `maxdim` ladder, e.g. 32/64/128; shared tol profiles | Chart `D_max(t)`, per-bond/pass `D_work`, sector additions, RSVD residuals, Krylov closure residuals, discarded weight and cap flags; report unmet accuracy rather than silently passing. |
| **IT5 — Work at matched accuracy** | Current order-1/midpoint BUG versus general-order at several `dt` | Report runtime, allocations, CBE sketch/holdout actions, K/L frontier applications, root `expv` actions, and observable/state error. Include *total repeated-sweep cost*, not only final root solve. |

For order claims, the small exact-matrix tests are the first gate. L=16 validates the complete MPS, symmetry, RSVD, K/L and truncation implementation; adaptive tolerances can create an error floor so slopes must be measured above it.

## 6. Skills and workflow for the Claude implementation agent

**First inspect which skills are installed** and read `CLAUDE.md`, repository instructions, and the existing test/benchmark entrypoints. Claude Code supports project skills in `.claude/skills/<name>/SKILL.md`; the skill names below are **recommended roles**, not assertions that those custom skills are already installed. Built-in skills such as `/code-review`, `/debug`, and `/verify` can be used if available (check the local command menu). Official reference: https://code.claude.com/docs/en/skills.

| Skill / agent role | When to invoke | Required output |
|---|---|---|
| **Repo explorer / codebase mapping** (`repo-map`, custom if available) | Before edits | Call graph of `cbe_expand`, `half_left!`, `half_right!`, root projection/`expv`, truncation, diagnostic structures; minimal edit locations. |
| **Julia TDD** (`julia-tdd`, custom) | Start of every milestone | Add failing `Test.@testset` fixtures UT1–UT9 first; then smallest implementation that makes them pass. Reuse repo dependencies; `julia --project -e 'using Pkg; Pkg.test()'`. |
| **MPS/U(1) invariants** (`mps-symmetry`, custom) | Any code changing frame/gauge/carry/charge blocks | Check canonicality, old-subspace containment, zero-sector handling and exact charge conservation. |
| **Krylov order verification** (`krylov-order`, custom) | Frontier and stopping-rule implementation | Review global-vs-local operator distinction, moment tests, defect estimates and timestep-convergence evidence; prevent unjustified order claims. |
| **Julia profiling** (`julia-performance`, custom) | After correct tests, never before | `BenchmarkTools`/`@allocated`/`Profile`: count MPO applications, root actions, reorthogonalizations, allocations and `D_work` growth; avoid premature micro-optimization. |
| **Built-in `/code-review`, `/debug`, `/verify`** | After each milestone or failure | Independent review, targeted debugging and execution of the real repo test command, including the L=16 integration test. |

**Efficient delegation:** (a) repo mapping and exact XX reference/test fixtures can proceed in parallel without overlapping edits; (b) one agent integrates the frontier API and K/L sweeps, assisted by U(1)/order checks; (c) profiling follows correctness. Do not have multiple agents modify `half_left!`/`half_right!` concurrently. Reuse existing tests, RSVD sampling machinery and logging. Avoid creating elaborate skills unless these roles are genuinely recurring; concise project `SKILL.md` files or task instructions are sufficient.

## 7. Definition of done

- Existing minimal-knob CBE-RSVD behavior remains intact; **only `order` is a new user-facing method choice**.
- General-order code path can request `p=1..4`; closure detects new frontiers and recognizes missing global Hamiltonian action, not merely local `H1` closure.
- All unit tests and the L=16 domain-wall reference tests pass; orders are **measured**, with honest reporting of truncation/basis floors.
- One final root `expv` and one controlled `root_out` compression; rank growth stays bounded or returns a documented unmet-budget status.
- Logs and plots show numerical errors **and** RSVD selections, per-sweep rank growth, root truncation and the actual cost of repeated K/L sweeps.
