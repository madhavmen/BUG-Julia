# Seven-task plan: higher-order symmetry-aware CBE-BUG for imaginary time and open systems

**Status:** implementation and experiment plan, not results. **Scope:** two setup tasks plus **two imaginary-time** and **three open-system** model tasks. The 2D real-time quantum-magnet runs are already underway and are not repeated here.

## Scientific question and comparison rules

Demonstrate, on physically meaningful and symmetry-resolved problems, whether higher-order adaptive CBE-BUG reaches **a given physical accuracy faster and at larger stable timesteps** than **symmetry-preserving two-site projector-splitting TDVP**. The rationale from Lubich and collaborators is specific: BUG constructs enlarged K/L bases without backward dissipative integration, evolves forward in a Galerkin root space, and admits robust rank-adaptive error/dissipation estimates under stated assumptions. **It is not a theorem that this particular randomized MPS implementation is always faster or that generic MPDO compression preserves positivity.**

**Solver controls:** `tol_basis`, `tol_trunc`, `tol_root`, `maxdim`, `order`, and a physical time grid. `order` and the grid are method choices, not extra unexplained rank-selection tolerances. Keep RSVD Test A (retained singular tail), Test B (independent sampling residual), Krylov Test C (first-action CBE overlap), and Test D (active-frontier closure). No public `dex`, `dover`, `comp_ratio`, `stol_pre`, `stol_fnl` or fixed `krylov_basis`.

**Order-/cost-fair comparisons:**

- **Matched-order test:** second-order midpoint BUG versus a genuinely second-order symmetric two-site TDVP sweep; match the global temporal order, not merely the local Krylov/exponential tolerance. Compare at identical `dt`, initial state, geometry, final rank cap, and output times.
- **Higher-order advantage:** BUG3/BUG4 versus a converged two-site TDVP reference at **matched observable error**, allowing TDVP to use smaller `dt`. If a verified third-/fourth-order *global* TDVP variant is available, add a like-order comparison. Do not label a second-order TDVP sweep “fourth order” merely because its local exponential is accurately solved.
- **Grid ablation:** both methods on the **same** logarithmic or uniform imaginary-time grid; separately show an optimized-grid comparison. Otherwise log-grid gains are confounded with BUG gains.
- **Stability test:** sweep stiff step sizes and measure **actual observable error, solver failure, and transient backward-substep amplification** for TDVP. Boundedness of BUG alone is not evidence of accurate evolution. The exact root exponential is already stable for dissipative Hermitian generators; implicit solvers must justify themselves through lower cost at large steps.
- **Full-cost accounting:** predictor and every high-order stage, RSVD probes/holdouts, K/L Krylov work, root actions or shifted linear solves, environments, truncation, memory, and retries. Report wall time **versus achieved physical error**, not only steps/second.

## Symmetry requirement for all five models

| ID | Model | Exploited symmetry | Representation |
|---|---|---|---|
| IT-1 | 1D spin-1/2 Heisenberg | **Strong U(1)**, fixed total `Sz=0` | Pure-state MPS |
| IT-2 | Square-lattice spin-1/2 Heisenberg cylinder | **Strong U(1)**, fixed total `Sz=0` | Pure-state MPS |
| OS-1 | Boundary-driven XX chain + dephasing | **Weak U(1)** Liouvillian symmetry, conserved ket-minus-bra charge | Vectorized MPDO |
| OS-2 | Bose-Hubbard + local two-body loss | **Weak U(1)** Liouvillian symmetry, conserved ket-minus-bra charge | Vectorized MPDO |
| OS-3 | **Particle-conserving East-type constrained hopping + dephasing** | **Strong U(1)** physical-number conservation (and Liouville block labels) | Vectorized MPDO |

**Important change from the earlier plan:** the coherently driven open quantum East model of Causer, Banuls and Garrahan (2025) does **not** have the desired generic U(1). To meet the all-models symmetry requirement, OS-3 uses the *particle-conserving quantum East Hamiltonian* of Brighi, Ljubotina and Serbyn (SciPost Phys. 15, 093, 2023), plus number-preserving Lindblad dephasing. **This is a proposed open-system extension, not the same model as the 2025 paper.** The published undamped East dynamics offer a reference at `gamma=0`; the damped extension requires independent exact small-system validation. Do not compare its curves directly to the coherently driven 2025 East results.

---

## Task 1 — Imaginary-time BUG with logarithmic panels: infrastructure + unit tests

Implement normalized imaginary-time propagation, a geometrically expanding panel grid, and interchangeable root integrators **without changing the validated CBE-RSVD/Krylov logic**:

- `psi(beta+dbeta) ~ exp[-dbeta*(H-Eshift)] psi(beta)` inside the augmented root space, followed by normalization and one controlled compression. Preserve the old state in the root Galerkin space; log energy **before root**, **after root/before truncation**, and **after truncation**.
- Use the logarithmic-panel schedule from Zima, Stoudenmire, White, Parcollet and Kaye (2026): geometrically increasing panel endpoints, initially **four substeps per panel** for a trapezoidal reference, with denser early-time sampling. Permit reproducible uniform-grid controls and adaptive rejection/retry when residuals or caps fail.
- Root methods: converged Lanczos exponential **as reference**; implicit trapezoidal (A-stable, order 2); two-stage Radau IIA (L-stable, root order 3); for BUG4, a sufficiently accurate exponential or a higher-order L-stable root scheme. Root temporal order alone does **not** establish the global BUG order: retain all required stage/midpoint derivative directions via adaptive CBE and one-site Krylov reuse.
- Estimate the **outside-Galerkin residual** of the full imagined-time step, not only the projected linear-solver residual. Re-enrich/re-solve or reject if the basis is insufficient; distinguish basis error from root residual and discarded weight.

**Small unit tests (IT-U1–U6):**
1. **IT-U1 — Grid:** no gaps/duplicates, correct endpoints and geometric panel growth; early-time resolution and reproducibility.
2. **IT-U2 — Root solvers:** scalar stiff `y'=-lambda*y`, exact small Hermitian matrix, expected amplification; trapezoidal A-stability, Radau L-stability, order tests for all enabled root schemes.
3. **IT-U3 — Global order:** phase-aligned MPS error for a small XX/Heisenberg chain across `dbeta,dbeta/2,...`; verify observed BUG2/3/4 **global** order with spatial compression effectively disabled.
4. **IT-U4 — Energy:** normalized energy decreases during an accurately solved **uncompressed** projected exponential step; attribute possible post-truncation increases to the measured compression error.
5. **IT-U5 — Symmetry and residual:** preserve total `Sz`, previous state and K/L subspaces; compare projected versus full residual and verify extra RSVD/Krylov enrichment when the latter fails.
6. **IT-U6 — Fail safely:** insufficient working rank, failed shifted solve, or unresolved RSVD A/B / Krylov C/D triggers a flagged retry/failure, never a silent successful step.

## Task 2 — Symmetry-aware MPDO CBE-BUG: infrastructure + unit tests

Implement `|rho>> = vec(rho)` as a Liouville-space MPS, with the correct Lindblad MPO `L` for `rho' = L(rho)`. Reuse sector-aware CBE-RSVD and K/L one-site Krylov enrichment (Tests A–D), but **make all tensor charge rules explicit for ket and bra**. Use `tol_root` for non-Hermitian Arnoldi exponential or an appropriate implicit root solver; higher-order stages require independently established whole-method order.

- Vectorized site dimension `d^2`, with local weak-symmetry charge `q_ket-q_bra`. Where every `H` and jump commutes with number, support **strong U(1)** sectors `(N_ket,N_bra)` as well. Validate the charge-covariance of **each** Hamiltonian and jump before enabling sectors.
- Track trace (`<<I|rho>>`), Hermiticity, and positivity (exact minimum full-density eigenvalue on small systems; local reductions only for larger systems). A generic truncated MPDO is **not** automatically CPTP. Do not infer Frobenius-norm contractivity for nonunital Lindblad generators.
- Keep the identity/trace vector in, or explicitly constrain, the final trial space if needed for trace accuracy; check whether any trace correction changes positivity/observable accuracy. Apply one accounted-for MPDO compression and log the largest local and accumulated discarded-weight estimates.

**Small unit tests (OS-U1–U7):**
1. **OS-U1 — Vectorization:** exact identity `vec(A*rho*B) = (transpose(B) kron A)*vec(rho)` with declared index convention.
2. **OS-U2 — MPO:** compare local Lindblad/MPO action against an explicitly assembled dense Liouvillian (small spin and small cutoff boson chains).
3. **OS-U3 — Physics invariants:** exact `Tr[L(rho)]=0`, Hermiticity preservation, and positivity of dense exact propagation; measure approximate MPDO violations explicitly.
4. **OS-U4 — Symmetry:** boundary-drive/particle-loss weak U(1) charge-difference blocks, number-preserving East strong U(1) blocks; forbidden blocks are not sampled, and initially empty reachable blocks can be discovered.
5. **OS-U5 — Contractive control:** for purely Hermitian dephasing jumps, verify `d Tr(rho^2)/dt = -sum ||[L_mu,rho]||_F^2`; test nonunital counterexample separately without assuming purity monotonicity.
6. **OS-U6 — Root and time order:** small Liouvillian compare Arnoldi/Radau against dense `exp(t*L)`; verify whole-MPDO BUG temporal order and dependence on tolerances.
7. **OS-U7 — Adaptive reuse:** CBE selected directions are correctly transported to the one-site Liouville Krylov space; C/D skip redundant directions and detect higher-action leakage; rank-cap and failed-residual paths are explicit.

---

## Task 3 — Imaginary time IT-1: 1D Heisenberg chain

**Model:** `H = J sum_{j=1}^{L-1} S_j.S_{j+1}`, `J=1`, open ends; initialize a Neel MPS in `Sz=0`. **Sizes:** small `L=12,16,20` (sparse exact reference), **main `L=100`**, stretch `L=200`. Match the Stoudenmire logarithmic-stepping Heisenberg setup.

**Physics:** energy per site and variance vs **log beta**, central/staggered spin correlations, convergence toward accurate DMRG ground state. **Algorithm:** compare uniform and log grids for BUG2/3/4 and symmetric 2-site TDVP2; sweep grid refinement and stiffness to probe the backward-step limitation. Plot actual ground-state error and CPU time to each accuracy target, not merely beta reached. **Model check:** exact small-chain state/energy and large-chain DMRG agreement; both energy before and after compression.

## Task 4 — Imaginary time IT-2: square Heisenberg cylinder

**Model:** `H = J sum_<ij> S_i.S_j`, `J=1`, cylindrical boundary conditions; Neel initial MPS at total `Sz=0`. **Sizes:** small `4x4`, `6x4`; **main `8x4`, `10x4`**; stretch `6x6` conditional on convergence.

**Physics:** `E(beta)/N`, energy variance, `S(pi,pi)` (with the correct finite-cylinder transform), spin/bond-energy spatial maps and ground-state correlators. Use DMRG and **finite-geometry** Sandvik QMC data if matching the exact boundary conditions; do not compare a finite cylinder directly against a thermodynamic energy density. **Algorithm:** compare cost/error and stable panel sizes for BUG2/3/4 versus matched-order or error-matched TDVP; show `Dmax(beta)` and individual RSVD/Krylov work over the full relaxation.

## Task 5 — Open system OS-1: boundary-driven XX + bulk dephasing

**Model:** `H = J sum_j (sigma+_j sigma-_{j+1} + h.c.)`; boundary gain/loss jumps with driving imbalance, optional Hermitian local dephasing `L_j=sqrt(gamma)*n_j`. Make spin and Lindblad-rate conventions explicit before comparing formulas. **Symmetry:** weak Liouville U(1). **Sizes:** small `L=4,6,8`; **main `L=32,64`**; stretch `L=100`.

**Physics:** `m_j(t)` heatmap, boundary-to-bulk front, current `j(t)` (log time to steady state), steady-state `j_NESS(L)` and magnetization profile. **References:** exact small density evolution; Žnidaric's analytical steady current/magnetization for the matching dissipator convention; exact transient results from the nondephasing boundary-driven XX literature where directly applicable. **Stability stress:** dephasing-only (purity law) then driven dissipative case; compare `gamma*dt`, actual late-time error and backward TDVP stage amplification.

## Task 6 — Open system OS-2: Bose-Hubbard with two-body loss

**Model:** `H = -J sum_j(b_j^dag b_{j+1}+h.c.) + U/2 sum_j n_j(n_j-1)`; `L_j=sqrt(gamma)*b_j^2`. **Symmetry:** weak Liouville U(1), despite physical loss. **Sizes:** small `L=3-6` with `nmax=2,3`; **main `L=16,24`**, stretch `L=32` with cutoff convergence.

**Physics:** remaining particle density `N(t)/N(0)`, double occupancy, `n_j(t)` heatmap from an inhomogeneous state, and the quantum-Zeno slowing of depletion as `gamma` becomes large. **Exact identity:** `d<N>/dt = -2*gamma sum_j <n_j(n_j-1)>` for these jump conventions. **References:** exact small-cluster Lindblad propagation and Rossini et al.'s strong-loss/Zeno trends; do not infer quantitative agreement at unmatched fillings/cutoffs. **Performance:** operator-local dimension `(nmax+1)^2`, memory and cost-at-error; separate physical boson-cutoff error from MPDO bond error.

## Task 7 — Open system OS-3: U(1)-conserving East-type chain with dephasing

**Model:** use the published **particle-conserving facilitated-hopping Hamiltonian** (hard-core bosons/spinless fermions)

`H_E = J sum_{j=2}^{L-1} n_{j-1} (c_j^dag c_{j+1} + c_{j+1}^dag c_j)`;

add the *proposed* open-system jumps `L_j=sqrt(gamma)*n_j`. Each `H_E` and `L_j` commutes with total particle number, so the Lindbladian has strong U(1); restrict initial states to a nontrivial mobile fragment, rather than an exactly frozen configuration. **Sizes:** small `L=4,6,8`; **main `L=24,40`**; stretch `L=60` if long-time error and cost remain controlled.

**Physics:** site-resolved density `n_j(t)` heatmap, imbalance/density autocorrelation over logarithmic time, particle spreading and relaxation within the allowed constrained fragment. **References:** Brighi, Ljubotina and Serbyn (2023) for the undamped U(1)-conserving Hamiltonian and fragmentation/slow unitary dynamics; **the dephased dynamics are a new variant** checked against exact small-L Lindblad solves, not the coherent open-East 2025 curves. **Model checks:** verify `[H,N]=0`, `[L_j,N]=0`, frozen configurations remain frozen, and only allowed facilitated hops are generated. This provides an explicit U(1)-symmetric long-time open-system test of BUG's dissipative handling.

---

## Shared analysis and mandatory plots (all five model tasks)

Use **three basis settings** (suggested `tol_basis = 1e-3, 1e-5, 1e-7`), with independently converged `tol_trunc`, `tol_root`, boson cutoff and rank caps. If a tolerance/rank cap cannot be met, **label that run as a failure**, not a converged solution. For randomized RSVD, use several fixed seeds on small and representative medium runs, log the spread and store seeds.

1. **Physical observables:** a clearly resolved space-time heatmap or spatial profile, plus accurate long-time curves against exact/literature/DMRG/QMC references, as appropriate to each model. Use the same output times and observable normalization across integrators.
2. **Error growth and timestep/order:** `error(t)` or `error(beta)` at each `tol_basis`, against an **independent** reference; local/global convergence slope for BUG orders 2/3/4; accepted `dt`/`dbeta`, retries and TDVP breakdown times; equal-grid matched-order (p=2) and matched-physical-error comparisons.
3. **Bond-dimension growth:** `Dmax_final` and `Dmax_working` versus time for every method/tolerance; bond-index/time heatmaps. Distinguish rank after CBE, after extra K/L Krylov directions, and after compression; report peak MPDO ranks and memory.
4. **Why adaptive BUG saves work:** per-sector RSVD samples/selected ranks; Test-A tail and Test-B independent missing-range estimate; Krylov C/D overlap/closure residuals and added vectors; one-site Hamiltonian actions with/without CBE–Krylov reuse. Plot maximum per-bond discarded Schmidt weight **separately** from cumulative compression error estimates.
5. **Why dissipative BUG is stable:** intermediate backward-TDVP norm-amplification factor, final physical error, root residual and failure/restart counts. In imaginary time, distinguish `Delta E_root` from `Delta E_compression`; for MPDOs log trace, Hermiticity, small-L negative eigenvalues, and purity *only where its monotonicity is mathematically valid*.
6. **Performance:** wall time and peak memory **versus physical error**, with breakdown (CBE sample/holdout, K/L Krylov, root solves, environments, SVD/truncation) and comparison on identical timestepping grids. Plot both the same-global-order BUG2/TDVP2 pair and BUG3/4 versus converged TDVP at matched physical accuracy.

**Acceptance:** all infrastructure tests pass; the selected physics is reproduced within documented error; temporal order is observed with basis/truncation floors suppressed; rank growth and approximation indicators behave consistently across tolerances; and any claimed BUG advantage is supported by **matched-error cost** and a reproducible large-step stability margin, not inferred merely from Lubich's theoretical properties.

## Primary literature to cite in the study

- Ceruti & Lubich, *An unconventional robust integrator for dynamical low-rank approximation* (2022), https://doi.org/10.1007/s10543-021-00873-0 — forward-only basis update vs backward projector-splitting stage.
- Ceruti, Kusch & Lubich, *A rank-adaptive robust integrator for dynamical low-rank approximation* (2022), https://doi.org/10.1007/s10543-021-00907-7 — small-singular-value robustness and gradient dissipation up to truncation.
- Ceruti, Lubich & Sulz, *Rank-Adaptive Time Integration of Tree Tensor Networks* (2023), https://doi.org/10.1137/22M1473790 — MPS/TTN context.
- Ceruti, Einkemmer, Kusch & Lubich, *A robust second-order low-rank BUG integrator based on the midpoint rule* (2024), https://doi.org/10.1007/s10543-024-01032-x — midpoint basis conditions.
- Hauck, Kusch & Schotthoefer, *High-order robust basis-update & Galerkin integrators* (2026), https://arxiv.org/abs/2608.27749 — general-order construction; root solver order alone is insufficient.
- Zima, Stoudenmire, White, Parcollet & Kaye, *Fast Tensor Network Imaginary Time Evolution by Implicit Stepping on Logarithmic Grids* (2026), https://arxiv.org/html/2606.02930 — logarithmic panels and chain benchmark.
- Sandvik, *High-precision ground state parameters of the two-dimensional spin-1/2 Heisenberg model on the square lattice* (2026), https://arxiv.org/html/2601.20189 — exact geometry-dependent ground-state references.
- Žnidaric, *Exact solution for a diffusive nonequilibrium steady state of an open quantum chain* (2010), https://arxiv.org/abs/1005.1271 — driven XX current/magnetization.
- Rossini et al., *Strong correlations in lossy one-dimensional quantum gases: From the quantum Zeno effect to the generalized Gibbs ensemble* (2021), https://doi.org/10.1103/PhysRevA.103.L060201 — boson loss.
- Brighi, Ljubotina & Serbyn, *Hilbert space fragmentation and slow dynamics in particle-conserving quantum East models* (2023), https://doi.org/10.21468/SciPostPhys.15.3.093 — **number-conserving East Hamiltonian**; its dephased open extension here is proposed, not a direct reproduction.
- Causer, Banuls & Garrahan, *Dynamical heterogeneity and large deviations in the open quantum East glass model from tensor networks* (2025), https://doi.org/10.1103/PhysRevB.111.L060303 — a **different, generally non-U(1)-symmetric** open-East model; retained as background, **not** a direct data reference for OS-3.
