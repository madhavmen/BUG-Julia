# Standalone Method-Efficiency Study: Why Adaptive DD-RSVD and Recycled Krylov Are Worth Using

**Scope:** One exactly solvable model, four publication-style figures, a compact set of correctness checks. This is a *methods justification*, not an open-system physics campaign. No Lindblad operators, quantum trajectories, MPDOs, collective decay, or large production sweeps.

**Algorithm under test:** U(1)-symmetric, third-order-target CBE-BUG with implicit **double-discarded (DD)** randomized bond expansion; McCulloch–Osborne-style streamed Gaussian MPO contractions and **QR-only** CBE direction selection; adaptive independent-holdout range checks; time-weighted local Krylov stopping; gauge-transported Krylov recycling; and one matrix-free **Hermitian Lanczos** root exponential. Its physical order must be *measured*, not inferred from three enrichment rounds.

## 1. The four questions the figures must answer

| Design choice | Why it might be worth using | What would count as convincing evidence? | What it **does not** show |
|---|---|---|---|
| **DD projection before QR sketching** | Search for genuinely two-site-entangling (DD) directions instead of spending limited bond rank on one-site-accessible DK/KD components | At *equal* selected rank, DD+QR captures more of the **true DD action** than one-sided QR on a bond with significant DK content | DD is not guaranteed to win on every XX snapshot or for every BUG stage |
| **Adaptive RSVD rather than fixed oversampling** | Quiet/low-rank bonds should require fewer Gaussian probes; hard bonds should receive more | Less *total* candidate-plus-holdout MPO work/time to meet the **same independently audited DD range error** | This does not imply the randomized sketch has a faster mathematical convergence rate per probe |
| **Time-weighted Krylov stopping** | A fixed Krylov depth wastes applications when the local exponential response is already represented; sometimes a fixed depth is insufficient | Comparable local response error with fewer **new one-site Hamiltonian applications**; stopping and validation overhead included | Early stopping does not make Krylov vectors intrinsically converge faster at fixed iteration count |
| **Recycled Krylov frontier** | The next higher-order sweep should not regenerate useful Hamiltonian directions already discovered in earlier sweeps | Fewer *new* H applications and lower total time over rounds 1–3 at equal independent response accuracy, **including gauge transport and operator refresh** | Recycling may be unhelpful when local effective operators/gauges change substantially |

**End-to-end question:** Do the savings remain after environment updates, root Lanczos, MPS reconstruction, and compression? Figure 4 answers this against the exactly known XX evolution. All claims are conditional on observed data; a negative result is useful and should change the default policy.

## 2. One toy model and an exact oracle

Use the **open-boundary spin-1/2 XX chain**, with `J=1` and `d=2`:

\[
H=J\sum_{j=1}^{L-1}(S_j^xS_{j+1}^x+S_j^yS_{j+1}^y)
 =\frac J2\sum_{j=1}^{L-1}(S_j^+S_{j+1}^-+S_j^-S_{j+1}^+),
\quad |\psi_0\rangle=|\uparrow\rangle^{\otimes L/2}|\downarrow\rangle^{\otimes L/2}.
\]

The U(1) charge is the conserved total \(S^z\); for even \(L\), the initial state has \(N=L/2\) spin-up fermions. Under Jordan–Wigner, the open XX Hamiltonian becomes a **quadratic** hopping operator with one-particle matrix \(h_{j,j+1}=h_{j+1,j}=J/2\). Define

\[
u(t)=e^{-iht},\qquad
V(t)=u(t)[:,1{:}L/2],\qquad
C(t)=V(t)V(t)^\dagger.
\]

Then the **finite-open-chain reference is exact** (up to numerical matrix-exponential rounding):

\[
\langle S_j^z(t)\rangle=C_{jj}(t)-\tfrac12,
\quad N_{\mathrm{left}}(t)=\sum_{j\le L/2}C_{jj}(t).
\]

For \(L=16\), the fixed-charge Hilbert-space sector has only \(\binom{16}{8}=12{,}870\) states. Build an independent exact full-state/sector oracle when needed: the amplitude of occupied configuration \(I\) is the corresponding **Slater determinant** \(\det V_{I,:}\), with the fermion/spin basis sign convention checked against small-chain sparse ED. This permits state fidelity and one-step error, not just observable error. An exact one-particle correlation matrix alone is *not* a state-fidelity oracle for an arbitrary truncated MPS input.

**Minimal fixtures:**

- **Fixture S (correctness and end-to-end):** `L=16`, start from the *uncompressed* domain-wall product state, evolve to `JT=2` (extend to `JT=4` only if needed to expose nontrivial rank growth). Use exact Slater/sector-state comparisons.
- **Fixture M (component efficiency):** `L=16` and optionally `L=24` or `32` for timing, at **two predetermined snapshots** `Jt*=0.5` and `Jt*=4`, testing bonds `i=L/2` (front) and `i=L/2-3` (away from center). Build high-accuracy reference snapshots, **then deliberately compress them in U(1) sectors** to modest retained ranks (e.g. 8 or 16) before extracting the identical local frames for every ablation. This leaves real discarded directions to discover. The compression is a fixture-generation step, not timed algorithm work. For `L=24/32`, exact *observables* still come from \(C(t)\); do not claim to hold an exact full state in memory.
- If the chosen snapshot has \(\|G_{DD}\|\simeq0\) or lacks an informative DK/DD mixture, label it **trivial** and use the other preselected snapshot/bond; never divide by a near-zero \(\|G_{DD}\|\), and never cherry-pick an undisclosed favorable bond.

**Why this model is enough:** early domain-wall motion produces highly uneven activity across bonds, and later dynamics populate new U(1) sectors and growing entanglement. The exact solution makes it possible to distinguish true accuracy improvements from favorable but misleading tolerance settings. A very low-rank/short-time example is deliberately expected to favor adaptivity; a later, harder snapshot checks its overhead and failure modes.

## 3. Benchmark protocol: fair comparisons with minimal runs

1. **One change at a time.** Every component ablation uses the same local MPO, exact same incoming (compressed) MPS snapshot, bond/gauge, symmetry sectors, step \(h\), root scheme and work cap. The only switched feature is the one named in the figure. For Fig. 1, both fixed and adaptive candidates use **the same DD projection, streamed kernel and QR**, so the adaptive-width effect is not confounded with using an SVD.
2. **Match achieved quality, not configured knobs.** Compare each method at the same independent range error (RSVD), local exponential response error (Krylov), or exact XX evolution error (full solver). In tables/CSV, mark configurations that *fail* the error target; do not include failed ones in a claimed speedup ratio.
3. **Count everything.** DD candidate **and validation** probes, projection, MPO-channel contractions, QR, local H actions, orthogonalization, basis transport, changed-operator image refresh, environment updates, root Lanczos, final SVD, allocations. Do not present sketch width, Krylov iteration count or wall time alone as proof.
4. **Simple repeatability.** Use three paired random sketch seeds (`11, 22, 33`), three warmed timing repetitions, fixed BLAS threads and hardware, and the same independent audit probes/reference for every baseline. Store raw counts and medians; increase repetitions only if speed differences are smaller than timing noise. Local exact/matrix tests are cheap and can use more seeds as needed.
5. **Minimal tolerance sweep.** For each local choice use *at most* three representative attainable target levels, for example `tol_basis ∈ {1e-3, 1e-5, 1e-7}`; use a fixed preregistered per-bond/round allocation of `tol_basis` for DD and Krylov. Keep `tol_root` and `tol_trunc` tight enough not to set the measured floor, and record `maxdim` and **achieved** rank. These target values are trial settings, not promised achievable accuracies.
6. **Independent audits outside timing.** The production holdout test is timed; the independent final accuracy audit is **not** included as runtime because it is not done in production. For small fixed tensors, compute the true DD residual using an offline explicit \(G_{DD}\); the production implementation must *never* allocate that matrix. For larger blocks, use a fixed, wider, independently seeded audit sketch and label it an estimate.

## 4. Four clean publication figures

### Figure 1 — “Do we spend Gaussian probes and new bond directions where they matter?”

**Purpose:** justify *both* DD selection and adaptive RSVD without merging them into one unfair comparison. Use one representative active XX bond, plus the predefined quieter bond as a small inset/table if needed.

**Panel A: adaptive stopping vs fixed oversampling (same DD+QR kernel).**

- **Adaptive:** initial sketch `ell=4`, add blocks of `4`, stop using a small independent holdout and the preallocated `tol_basis` criterion; reuse previous candidate and holdout actions appropriately.
- **Standard:** one-shot fixed \(\ell=k+p_{\rm over}\), with `p_over ∈ {4,8,16}`; keep `k` fixed/predeclared per sector for this baseline. Use **QR-only** for both, no reduced SVD. Show different `k` only if necessary to match the target; document it.
- **x-axis:** **total streamed MPO-channel contractions** (including production holdout). **y-axis:** independently audited \(\rho_{DD}(Q)=\|(I-QQ^\dagger)G_{DD}\|_F/\|G_{DD}\|_F\). Mark the target with a light horizontal line. Label failed fixed configurations. Report total time and bytes in the small data table/figure caption.

**Panel B: DD vs one-sided projection (same width and QR).**

- Compare \(G_{DD}=P_L^D X P_R^D\) to the one-sided \(G_{1s}=P_L^DX\), *using the same Gaussian input width* and reporting **how well both resulting left bases capture \(G_{DD}\)**.
- **x-axis:** selected new bond columns; **y-axis:** independently audited DD residual \(\rho_{DD}\). This asks whether one-sided sampling spends finite rank on directions useful only to one-site/Krylov updates.
- **Important:** on pure DD snapshots the curves may coincide; if so, report that the extra DD projector adds overhead without benefit **in this case**. Do not claim universal superiority.

**Expected interpretation:** adaptive sketching reduces work on easy bonds and pays for more probes only on hard bonds; DD selection can avoid DK redundancy. **These are hypotheses, not acceptance conditions.**

### Figure 2 — “Why not just set the Krylov iteration count to 16 or 32?”

**Purpose:** isolate the *time-weighted stopping rule*, without recycling.

- Freeze the same CBE-augmented basis \(Q_{\rm CBE}\), local Hermitian one-site operator \(H_{1,\mathrm{ex}}\), starting tensor \(K_{\rm ex}\), and step \(h\).
- **Adaptive:** enlarge the active Krylov frontier until the time-weighted defect/validated local response meets its allocated `tol_basis` budget. **Fixed-depth:** exactly the same QR/orthogonalization and seed with predetermined `m=8,16,32` (or earlier exact breakdown); no recycling in either method.
- An independent small-system reference checks the **representability** of the local exponential response:
  \[
  \epsilon_K(Q)=\frac{\|(I-QQ^\dagger)e^{-ihH_{1,\mathrm{ex}}}K_{\rm ex}\|_F}{\|K_{\rm ex}\|_F}.
  \]
  For medium local problems, replace explicit \(e^{-ihH_1}\) with a tightly converged **independent** reference action; never include this audit in the production solver.
- **Panel A:** x = *new one-site Hamiltonian applications*, y = independent \(\epsilon_K\) (log scale); show adaptive accepted points and the three fixed depths.
- **Panel B:** x = representative \(h\) (e.g. `0.05, 0.1, 0.2`), y = selected Krylov applications, with a compact indication of which fixed depths pass the **same** response-error threshold. Plot execution time in a caption/table.

**Expected interpretation:** the same Krylov recurrence is being used—the advance is **avoiding unnecessary iterations and detecting insufficient fixed-depth choices**, not changing the per-iteration convergence polynomial.

### Figure 3 — “Do we really avoid recomputing Hamiltonian directions in higher-order BUG?”

**Purpose:** isolate *recycling*, keeping the adaptive stopping criterion and DD-CBE directions **identical**.

- On the **same** XX starting state and three verified order-3 augmentation rounds, compare (R) **transport and reuse** of previous relevant CBE/Krylov columns and their active frontier with (F) **fresh local Krylov** on each round. Both retain the same previously assembled MPS frames; F only clears the *recycle cache*.
- After any gauge/environment change, gauge-transport old vectors, reorthogonalize and **reapply the new local H** where operator images need refresh. Never reuse stale Lanczos tridiagonals or stale \(H_1v\) products. Keep the root solve out of this isolated timing.
- **Panel A:** grouped bars: **new one-site H applications** in rounds `1,2,3`, R vs F.
- **Panel B:** cumulative **total local-enrichment wall time** through rounds `1,2,3`, including transport, reorthogonalization, operator refresh, and basis handling.
- **Accuracy guard:** compare the independent local response defects and attained final augmented spans (via subspace projections or physical response error). Show an accuracy-pass mark; if F or R fails, no speedup is claimed.

**Expected interpretation:** savings should appear in rounds 2–3 if earlier Krylov directions remain relevant. If gauge changes erase their value, the correct conclusion is that recycling should be made conditional.

### Figure 4 — “Does any of this actually make the complete BUG solver faster?”

**Purpose:** ensure local wins are not erased by environment reconstruction, root Lanczos, and final SVDs.

- Run `L=16`, domain-wall initial product state, `JT=2` with the same third-order-target root solver and three-round basis recurrence. Compare only **three BUG settings**:
  1. **Full:** adaptive DD+QR, time-weighted Krylov stop, recycling.
  2. **No recycling:** same adaptive DD+QR and stop rule, fresh Krylov each round.
  3. **Conventional controls:** fixed DD sketch width and fixed Krylov depth, fresh Krylov; baseline settings selected *in advance*, then refined only to meet the same physical error.
- Add **one** ordinary two-site TDVP trace only as an optional external accuracy/cost check; the three BUG ablations are the causal comparisons for the design choices. TDVP2 generally has different order and should be independently timestep-tuned at **matched physical accuracy**.
- **Panel A:** x = total measured wall time, y = global root-mean-square exact magnetization error
  \[
  \epsilon_z(T)=\sqrt{\frac1L\sum_j\bigl(\langle S_j^z(T)\rangle_{\rm MPS} -[C_{jj}(T)-\tfrac12]\bigr)^2},
  \]
  as `dt` is varied. On `L=16`, separately audit global state error/fidelity using the exact Slater-state coefficients, with phase-aligned comparison; keep secondary metrics off the figure.
- **Panel B:** x = `dt`, y = exact **global state error** at fixed `T` for the *full* method, alongside a light \(h^3\) reference slope. Verify a one-step \(h^4\) slope in the regression report, not as a fifth figure. Run `dt ∈ {0.2, 0.1, 0.05, 0.025}` *if* this is a clean asymptotic region; use a tighter interval if not.

**Expected interpretation:** the preferred solver is justified only if it produces the **same physical answer for less work**, while retaining the claimed third-order accuracy. No basis or root tolerance can replace a genuine timestep convergence test.

## 5. Minimal test suite and stop/go gates

These are fast **smoke and unit tests**, not additional simulation campaigns.

| ID | Test | Pass requirement |
|---|---|---|
| **S1** | Exact XX reference, `L=4,6,8,16` | \(\sum_j\langle S_j^z\rangle=0\), norm/energy conserved, Slater correlation and small-sector ED agree; basis signs checked |
| **S2** | Both streamed DD sketches and U(1) sectors on tiny tensors | Streamed \(P_L^DXP_R^D\Omega_R\) and mirrored adjoint equal explicit contractions; no dense production projectors; empty-but-reachable sectors discovered |
| **S3** | Adaptive DD stopping vs offline exact residual | In repeated inexpensive seeded tests, meet allocated DD accuracy or report `basis_limited`; count holdout cost and record false accepts |
| **S4** | Local Krylov stopping on a delayed-frontier fixture | Time-weighted defect is consistent with independent \(\epsilon_K\); need for \(H_1^2\) is detected even if initial \(H_1K_{\rm ex}\) lies in the existing space |
| **S5** | Recycling with gauge rotation and changed local operator | Recycled and fresh runs attain the same response accuracy; operator images refreshed and charge orthogonality preserved |
| **S6** | BUG order-3 actual recurrence and exact XX evolution | Source confirms a genuine higher-order predictor/frontier—not just three repeats of the same input. Then observe local \(O(h^4)\) and global \(O(h^3)\) in a real convergence window with other errors suppressed |
| **S7** | One-at-a-time tolerance changes | Tightening `tol_basis` improves the measured local defects (or reaches a floor); tight `tol_root` and `tol_trunc` do not mask comparisons; cap failures flagged, not reported as successes |
| **S8** | Paired ablation reproducibility | Same snapshot/bond/generator/seeds/hardware; matched-achieved-error comparisons and truthful matvec/probe/time accounting; every plotted point traceable to its configuration |

**Release rule:** S1–S6 are prerequisites for making the four **solver-performance** claims. If third order fails, report that as a method-development problem and **do not** label Figure 4 an order-3 performance validation. A component figure may still be useful independently if its operator-level tests pass.

## 6. Efficient execution sequence (one short methods sprint)

1. **Exact oracle and snapshots:** implement/check the finite-open-chain fermion solution; prepare the four predetermined bond snapshots and intentional U(1) compressions; record \(\|G_{DD}\|\), DK/DD ratio and achieved ranks to ensure each comparison is meaningful.
2. **Local reference audits:** implement offline true DD residual on tiny blocks and independent local exponential-response accuracy; instrument candidate+validation probe cost and new Krylov H applications. Run S1–S5.
3. **Figures 1–3:** run the limited paired baselines, one feature toggled at a time; save raw CSV/JSON and a single compact table of achieved error, cost, storage and status. If Figure 1 is trivial on an inactive bond, report it as such rather than hiding it.
4. **Validate order:** verify the actual higher-order stage recurrence and S6–S7 using the exact `L=16` solution. Do not assume three sweeps imply third order.
5. **Figure 4:** short end-to-end run from the same uncompressed initial domain wall, varying only timestep and the three predefined BUG variants. Optionally include one tuned TDVP2 cost reference.
6. **Write four concise captions:** for each: *hypothesis → controlled comparison → achieved-error check → saved work → where it fails*. If speedup disappears end to end, state that and change the recommendation.

## 7. Deliverables and claim discipline

**Files:** `methods_efficiency_XX.md` (methods/settings), `fig1_rsvd_dd.pdf`, `fig2_krylov_stopping.pdf`, `fig3_krylov_recycling.pdf`, `fig4_total_solver.pdf`, `raw_component_costs.csv`, `raw_solver_errors.csv`, and a short `validation_results.md`. PDF plotting is a later output stage, not required to execute this plan.

**Plots:** max two panels per figure; direct, descriptive axis labels with units; medians and small seed bands only where stochastic; identical markers/colors for the same method across figures; no dense in-plot parameter tables. A narrow caption should state the mechanism and cost metric, and explicitly distinguish estimated DD-range error from exact XX physical error.

**Strong claims allowed only if demonstrated:**

- “Adaptive DD-RSVD saves MPO work **at matched independent range error** on heterogeneous bonds.”
- “Time-weighted Krylov stopping avoids unnecessary local Hamiltonian actions while satisfying the same physical-response criterion.”
- “Recycling prevents rediscovery of useful local directions in later augmentation rounds, **net of its bookkeeping overhead**.”
- “The combined choices improve **end-to-end time-to-accuracy** for a verified third-order CBE-BUG method.”

**Do not claim:** that all individual improvements always win; that a holdout is a deterministic error guarantee; that CBE residual directly equals full-state error; that fixed-order Krylov literally converges more slowly per iteration; that recycling is free when local operators change; or that three augmentation sweeps by themselves establish \(O(h^4)\) local error.

**References (background):** McCulloch–Osborne, arXiv:2403.00562 (streamed Gaussian range finding/QR); Gleis–Li–von Delft, arXiv:2501.12291 (DD versus one-sided CBE); Martinsson–Tropp, *Randomized Numerical Linear Algebra* (Gaussian range finding/validation); finite-chain XX/Jordan–Wigner free-fermion solution (use the exact finite open-chain matrix \(h\) above, not an infinite-chain approximation).
