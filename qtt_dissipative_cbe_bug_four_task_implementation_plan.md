# Four-Task Implementation Plan: Rank-Adaptive QTT–CBE–BUG for 2D Dissipative Gross–Pitaevskii and Heston Dynamics

**Status:** updated 2026-10-29 with a source-faithful **Appendix A.2 TDVP implementation flow**, a mandatory early no-symmetry CBE break-even experiment, and nonlinear stage-consistency tests; checked against arXiv:2507.01149v1 and arXiv:2507.04279v5. This is an implementation specification and proposed experiments, **not** a claim that the code already supports QTT PDEs, nonlinear stages, or fourth-order dissipative BUG.  
**Models:** (G) dissipative 2D Gross–Pitaevskii equation (GPE); (H) 2D Heston backward pricing/forward diffusion operators.  
**Primary hypothesis:** without physically linearizing the GPE (except in independent exact-reference tests), forward-only, rank-adaptive CBE–BUG can achieve **better time-to-accuracy at larger stable timesteps** than a conventional projector-splitting two-site TDVP implementation on sufficiently dissipative, structured PDEs. The claimed advantage must be **measured**, not assumed.  
**Target production run:** 30 quantics bits (15 bits per coordinate), final time `T = 500`, **maximum allowed** step `h_max = 0.1`, orders `p = 2, 3, 4`, with staged accuracy, stability, bond-rank, and resource gates. If `h=0.1` is not physically accurate, reduce the step; do not present a stable but inaccurate run as a success.

## Scientific framing and source-derived constraints

1. The **full GPE is nonlinear** because of `g |psi|^2 psi`. Its `g=0` limit is linear. There are exact 2D tests for the linear problem and special exact **nonlinear** solutions, but the general vortex-shedding solution is **not analytically known**; use a converged independent numerical reference.
2. The 2D Heston PDE is **linear, diffusive, and generally nonnormal**. It admits exponential-affine exact solutions for special terminal profiles and semi-analytic Fourier-transform references for option prices. Its discount term `-r u` is not equivalent to generic decay of every Euclidean norm; use positivity/maximum-principle and accurate-reference diagnostics.
3. Connor, Duncan & Daley, *Tensor network methods for the Gross-Pitaevskii equation on fine grids*, arXiv:2507.01149v1, use the damped GPE with `g=1`, `gamma=0.1` and paddle-driven 2D vortex shedding. Their representative `256 x 256` comparison uses `dt=0.001`. This paper is an **experimental physics/dynamics benchmark**, not an exact oracle. `https://arxiv.org/html/2507.01149v1`
4. Chen, Liu, Li & Chung, *Solving the Gross-Pitaevskii Equation with Quantic Tensor Trains: Ground States and Nonlinear Dynamics*, arXiv:2507.04279v5 / SciPost Phys. 21, 035 (2026), compare QTT imaginary-time TDVP and QTT gradient descent for 2D harmonic/rotating traps and find gradient descent faster for the ground-state examples. Compare to **both** where reproducible. `https://arxiv.org/html/2507.04279v5`
5. Basis-update & Galerkin (BUG) avoids the explicitly backward S-substep of standard projector splitting; this is a plausible stability advantage for dissipative generators, but **some stabilized/implicit projector-splitting schemes also remain stable**. Compare against such baselines and do not claim all TDVP variants must blow up. `https://link.springer.com/article/10.1007/s10543-021-00873-0`, `https://arxiv.org/abs/2507.15192`
6. Prior internal XX study: active-bond DD sampling substantially improved the DD range; adaptive Krylov stopping improved cost; the old adaptive sketch used 32 columns versus 16 for a successful fixed width; full-image Krylov recycling increased cost by approximately 1.7–4.3x in the spin tests. **Carry forward DD and adaptive stopping; redesign sampling; keep full-image recycling disabled.** Sources: uploaded `xx_efficiency_summary.md`, `validation_results.md`, and `xx_domain_wall_cbe_bug_efficiency_standalone_plan(1).md`.

## Appendix A.2 audit: no physical linearization of the GPE is required

**Direct answer to the implementation question:** No. Keep the **full nonlinear** damped GP equation in all meaningful vortex, ground-state, and solver-comparison runs. Setting `g=0` remains useful **only as a separate analytic validation fixture**, not as the production equation. For `hbar=m=1`, the paper's nonlinear equation is

\[
\boxed{\partial_t\psi=F(t,\psi)=(-i-\gamma)
 \Big[-\tfrac12\Delta+V(x,y,t)-\mu+g\operatorname{diag}(|\psi|^2)\Big]\psi.}
\]

The diagonal multiplication operator depends on the **current trial wavefunction**. Assembling an MPO from a temporarily known density does **not** make the physical PDE linear; it is simply a representation or lagged evaluation of a nonlinear map. Jacobian linearization *within Newton--Krylov* is allowed as a numerical device, provided the full nonlinear stage residual is driven to tolerance.

**What Connor--Duncan--Daley actually state in [arXiv:2507.01149v1, Appendix A.2](https://arxiv.org/html/2507.01149v1):**

1. They use **ITensorTDVP / ITensors** and construct `H = -1/2 Delta + H_d`, with `H_d = V - mu + g|psi|^2` as a **real-space diagonal MPO**. The kinetic Laplacian may be built from finite differences or `-F^{-1}|k|^2 F`; they select the **QFT-based** Laplacian for TDVP because it gave lower errors in their comparison.
2. They evolve a **nonlinear GP** vortex-shedding problem; Appendix A.2/Figure 26 compare TDVP, finite-difference RK4 and split-step MPS dynamics against **XMDS2** on a `256 x 256` grid, `gamma=0.1`, `v_sweep=1.5`, `dt=0.001`. The paper finds comparable TDVP and split-step errors at a cutoff `1e-12`, while split-step wins at the tighter cutoff `1e-15`. This is **not** an example in which TDVP was demonstrated to blow up.
3. **Not specified in Appendix A.2:** whether the reported run uses one-site or two-site TDVP; how often the state-dependent density MPO is refreshed; how the nonlinear MPO is updated *inside* a TDVP sweep, or whether a self-consistent implicit nonlinear solve is used. **Do not attribute any particular refresh/linearization strategy to that paper without checking its code or an additional source.** Their use of a state-dependent diagonal MPO does not by itself specify temporal order for the nonlinear equation.
4. Their grouped/sequential bit order (all `x` bits followed by all `y` bits) is the proposed default here. In their QFT example, this yielded a lower maximal **operator** bond dimension than alternating `x_1,y_1,x_2,y_2,...`; this is a useful reason to retain grouped ordering, but wavefunction ranks still need direct measurement.

**Independent complementary implementation guidance:** [Chen et al., arXiv:2507.04279v5, Sec. 3.1](https://arxiv.org/html/2507.04279v5) explicitly say that the interaction `g|psi|^2` is **reconstructed after each time step**, using the newly evolved wavefunction; they keep the one-body and nonlinear QTTO terms separately, build a density QTT from an initially quadratic-rank Hadamard product, and **variationally compress** it before applying it as a diagonal operator. They report this density compression as an important cost driver (roughly `O(D^4)` in their setup). This does **not** by itself imply a fourth-, third-, or even second-order **time treatment of the nonlinearity**. For our implicit high-order BUG implementation we must update the density at every **trial implicit stage**, not only after each accepted step.

**Proposed fair baseline, distinguished from what the appendix actually documents:** implement `TDVP2_lagged`, using one density MPO built from `psi_n` at step start, and separately `TDVP2_consistent`, updating the nonlinear density for each substep or nonlinear iteration until stage consistency is reached. Both use the **same full nonlinear PDE**, but the first is an intentionally lagged approximation whose observed order may be lower. Compare the actual cost of refreshing `|psi|^2`, rebuilding its MPO/environments, applying `H_eff`, and solving local stages for **both BUG and TDVP2**. Include a QFT split-step nonlinear reference (or dense-grid FFT split-step) where appropriate. A large-`dt` victory over the lagged baseline **alone** is not the intended result.

### Reproducible nonlinear TDVP2 update flow (our proposed baseline; **not** a claim about the authors' undocumented implementation)

Appendix A.2 supplies the MPO and says TDVP evolves tensors/bonds through local sweeps; it does **not** give an explicit nonlinear substep schedule. The following is the implementation we should benchmark against nonlinear CBE--BUG:

1. **Input:** current canonical QTT `psi_n` with grouped `x`-then-`y` bits, moving-paddle time `t_n`, and separate static kinetic and potential MPO/environment caches.
2. **Construct the density:** `rho = compress_QTT(psi .* conj(psi))`; evaluate its action error on small grids. Form `H[psi,t] = T_QFT + V_pad(t) + V_trap - mu + g * Diag(rho)`. The Hamiltonian is **state dependent**; the diagonal MPO is a linear map only for the particular trial state at which it has been constructed.
3. **Local TDVP2 sweeps:** contract left/right environments to obtain the local projected RHS. Perform the standard two-site local evolution and center/bond updates, including any backward projector-splitting center step. Instrument the center-step norm and any subsequent cancellation. Build the same spatial operator and use the same final compression criterion as BUG.
4. **Two transparent nonlinear update policies:**
   - **Lagged (`TDVP2_lagged`)**: hold `rho_n` fixed during a full time step, rebuild it using `psi_{n+1}` for the next step. This gives a *well-defined, cheaper approximation* of the nonlinear flow, but its true order must be measured and it must not be our only TDVP baseline.
   - **Stage-consistent (`TDVP2_consistent`)**: evaluate `rho=|psi_trial|^2` and `V_pad(t_stage)` for each relevant trial substep; if the local method is implicit, iterate the full projected nonlinear equations to tolerance. Rebuild affected nonlinear environments or update them incrementally **with an independent residual check**. Do not claim the stage-consistent splitting automatically has order 2; measure the complete method.
5. **Accept and audit:** canonicalize/compress, save `psi_{n+1}`, record density-construction time/rank, environment rebuilds, backward-stage amplification, discarded weights and the independent physical error against a dense spectral reference. Repeat at `h/2` to distinguish time lag from projector/truncation error.

For a nonlinear BUG root stage, `Y_j = S_n + h sum_l a[j,l] K_l` and `K_j = W^dagger F(t_j, W Y_j Z^dagger) Z`. Thus **each** `F(t_j,...)` reconstructs the nonlinear density at that stage/iterate. A Newton--Krylov **Jacobian-vector product** is a *numerical linearization for solving these equations*, not a linear approximation of the GP model. For real-time vortex shedding neither `g=0` nor frozen `g|psi_n|^2` is an acceptable substitute for the full model unless explicitly labeled as a separate controlled ablation.

**Source-vs-implementation acceptance test:** keep a short `paper_appendix_A2_audit.md` recording (i) what Appendix A.2 actually says, (ii) details it omits (one-site/two-site, nonlinear refresh schedule, time order, exact local solver), and (iii) our explicitly chosen variants. Never imply the paper demonstrated projector-splitting instability: its Figure 26 reports **stable, competitive TDVP errors** on a small-step vortex benchmark.

### One-screen task map

| Task | Development target | Mandatory gate before moving on |
|---|---|---|
| **1. QTT core and nonsymmetric adaptive CBE–BUG** | Grouped 2D binary ordering; MPO/TT operators; environment contractions; hybrid DD rank selection **plus mandatory early no-symmetry first-impressions study**; stage-aware nonlinear action | Tiny-grid QTT/MPO equivalence, environments and nonlinear density parity, **measured CBE cost-vs-DD-error**, no hidden dense allocations |
| **2. Exact GP/Heston tests and order/stability analysis** | Analytic 2D references; **stage-consistent full nonlinear GP for BUG and TDVP2**; TDVP2 backward-stage instrumentation; p=2/3/4 root integrators | Exact `g>0` stage checks, error vs `h`, actual global orders, backwards amplification, matched-error stability |
| **3. Imaginary-time GP and finite-time dissipative dynamics** | Ground states (uniform vs logarithmic-time schedule), nonlinear vortex shedding, Heston diffusive evolution, adaptive rank characterization, competitive baselines | Physics converged against grid/time/rank changes and reference runs; measured time-to-accuracy advantage or transparent negative finding |
| **4. L=30, T=500 production and observables** | 30-bit grouped QTT, `h<=0.1`, long-time rank-adaptive BUG, checkpoints, detailed observables and cost vs TDVP | Pilot accepted `h=0.1` where used, error/restart checks, peak ranks/memory inside budget, no uncaught loss of stability |

---

# TASK 1 — Adapt the environment-based CBE–BUG solver to nonsymmetric, grouped-ordering 2D QTT

**Outcome:** an operator-agnostic, nonsymmetry-resolved QTT integrator with controlled rank expansion and no dependence on the old spin-chain U(1) sector selection. Reuse existing BUG, implicit trapezoidal, Radau IIA and GMRES implementations only after a source audit verifies compatible interfaces; the uploaded plans describe these as present, but do not establish that their general-PDE/nonlinear integrations already work.

## 1.1 Binary geometry, ordering, normalization, and MPO construction

For `n_x+n_y=L_QTT`, define

\[
 x_{i_x}=x_{\min}+i_x\Delta x,\quad i_x=\sum_{k=0}^{n_x-1}2^k b_{x,k},\qquad
 y_{i_y}=y_{\min}+i_y\Delta y,\quad i_y=\sum_{k=0}^{n_y-1}2^k b_{y,k}.
\]

**Default: grouped ordering**

\[
 (b_{x,0},\ldots,b_{x,n_x-1}\mid b_{y,0},\ldots,b_{y,n_y-1}),\quad
 \psi_{i_x,i_y}=A^{[1]b_{x,0}}\cdots A^{[n_x]b_{x,n_x-1}}
 A^{[n_x+1]b_{y,0}}\cdots A^{[L]b_{y,n_y-1}}.
\]

The physical coordinate name `y` means `y` for GP and **variance `v`** for Heston. The intra-coordinate bit endian convention must be explicit and unit-tested; grouped versus interleaved is **not** equivalent at fixed attainable TT rank. Use grouped as required; add an inexpensive interleaved ordering comparison on small grids solely as a diagnostic if grouped ranks explode.

`L=30` with `n_x=n_y=15` corresponds to `2^15 x 2^15 = 32,768 x 32,768`, i.e. `2^30 = 1,073,741,824` grid values. **Never materialize that full field in the production hot loop.** A field QTT has storage approximately `sum_i 2 chi_(i-1) chi_i` scalars; MPOs, Krylov bases, preconditioners, and environments can have much larger working memory.

For uniform quadrature, distinguish the raw Euclidean TT norm from the physical L2 norm:

\[
\|\psi\|_{L^2,h}^2 = \Delta x\Delta y\sum_{i_x,i_y}|\psi_{i_x,i_y}|^2.
\]

Either encode a global `sqrt(Delta x Delta y)` scaling into the TT state consistently or keep the weight outside all contractions. Do not mix the two conventions in tolerances or benchmark errors. For Heston's nonuniform weight/boundary treatments, document the chosen discrete inner product and when operator similarity transforms are employed.

Implement and test the following **matrix-free QTT MPOs**, without constructing a giant grid matrix:

- `D_x`, `D_y` and `D_xx`, `D_yy` from binary carry/shift operators or a validated spectral MPO; periodic GP boundaries and Heston-specific finite-domain boundaries must be *separate operator fixtures*.
- `D_xv` for Heston (correlated Brownian motion), and diagonal QTT factors `x`, `v`, `v^2` / `V(x,y,t)`.
- GP kinetic `-1/2 (D_xx + D_yy)`, harmonic `V_trap=(omega^2/2)(x^2+y^2)`, optionally `-Omega L_z`, plus time-dependent paddle `V_pad(x,y,t)` and chemical-potential shift.
- Nonlinear GP action `n(psi) = |psi|^2`, `N(psi)=g n(psi) psi`: use elementwise TT products with controlled compression, **refreshing** `n(psi)` at the appropriate implicit stage and time, and audit `h |g| ||(n_exact-n_QTT) psi||` on tractable grids. Do not silently treat this as a state-independent linear MPO.
- Heston log-price backward generator in time to maturity `tau`:

\[
\boxed{\mathcal A_H =\tfrac12 v\partial_{xx}+\rho\sigma v\partial_{xv}
 +\tfrac12\sigma^2v\partial_{vv}+(r-q-\tfrac12v)\partial_x
 +\kappa(\theta-v)\partial_v-r.}
\]

Here `kappa` = mean-reversion rate, `theta` = long-run variance, `sigma` = volatility of variance, `rho` = correlation, `r` = rate, `q` = dividend yield. Track the variance boundary at `v=0`, the choice of Feller regime `2 kappa theta >= sigma^2`, and exact/specified finite-box traces. For long-time forward densities, also implement the properly discretized **adjoint** `A_H^*` with conservative flux handling and explicit boundaries.

**Environment-based local action:** for MPO `W`, maintain left/right environments `E_L^[i]`, `E_R^[i]` and contract

\[
G_{i,\rm eff}[\Theta]
  =E_L^{[i-1]}\,W^{[i]}W^{[i+1]}\,\Theta\,E_R^{[i+2]}
\]

as tensor contractions with indices as appropriate. This notation denotes contractions, **not** a dense matrix product. Build separate environments for each summand, caching static kinetic/potential operators. Invalidate time-dependent `V_pad` environments and nonlinear-density environments when their source tensors change. No full `d^L x d^L` matrix or two-site dense operator is allowed in production.

## 1.2 Nonsymmetric DD-controlled bond expansion with a *hybrid* sketch controller

At each bond `i`, reshape the current local two-site state to

\[
\Theta_i\in\mathbb C^{(2\chi_{i-1})\times(2\chi_{i+1})},\quad
\Theta_i\simeq U_i S_i V_i^\dagger,
\]

and let `U_i,V_i` denote the current retained left/right subspaces in their respective local physical spaces. The projectors onto currently missing directions are

\[
 P^D_L=I-U_iU_i^\dagger,\qquad P^D_R=I-V_iV_i^\dagger.
\]

For `X_i = mat[F_eff(Theta_i)]` (`F_eff` may be a nonlinear/state-dependent stage action), the **double-discarded action** is

\[
 A_{DD}=P^D_L X_i P^D_R.
\]

Compute its *sketched* left/right ranges, without explicitly allocating `X_i` or either projector:

\[
Y_L=P_L^D\big[X_i(P_R^D\Omega_R)\big],\qquad
Y_R=P_R^D\big[X_i^\dagger(P_L^D\Omega_L)\big].
\]

The mirrored right sketch must apply **the conjugate transpose of the reshaped current stage vector-field tensor** `X_i`, i.e. `X_i^dagger`, implemented by appropriate conjugated/transpose streaming contractions. It is **not** in general the same as a simple action of `G_eff^dagger` on the MPS, and for nonlinear GPE it is **not** the Fréchet-adjoint `DF(psi)^dagger`. The nonlinear Jacobian is a distinct operator used for Newton--Krylov implicit stages; always audit this distinction against an explicitly formed small-grid `X_i`.

**Controller redesign based on negative XX results:**

1. Compute attainable discarded dimensions `dL = 2 chi_(i-1)-rank(U_i)` and `dR = 2 chi_(i+1)-rank(V_i)`, with `min(dL,dR)` as the obvious range-rank bound. No artificial U(1) sectors; if a numerical symmetry block is introduced later, this is a separate optional extension.
2. If the DD space is small, compare exhaustive streamed discarded actions with sampling; choose whichever has lower **measured** cost. No large dense two-site action.
3. For larger DD spaces, choose an initial sketch width using the previously accepted local rank (a **predictor**, not a correctness condition) and a bounded initial oversampling block. Start at a useful rank rather than repeatedly growing from a tiny sketch.
4. QR and remove dependent columns. Reserve a small **independent holdout** with its own RNG stream. Estimate squared missing action by

\[
 \widehat R_{DD}^2 = \frac1s\left\|(I-Q Q^\dagger)A_{DD}\Omega_{\rm test}\right\|_F^2,
 \qquad \mathbb E[\widehat R_{DD}^2]=\|(I-QQ^\dagger)A_{DD}\|_F^2
\]

for i.i.d. variance-one Gaussian probes (normalization is needed if the probes are scaled differently).
5. Accept only when a suitably conservative **time-weighted absolute** residual fits the allocated part of `tol_basis`:

\[
\eta_{DD}=\frac{h\widehat R_{DD}}{\max(\|\psi_n\|,\epsilon_0)}
 \le c_{DD}\,\texttt{tol_basis}.
\]

The `h` scaling is a leading-response engineering criterion, **not** a full physical error bound. Use safety factors/independent audits to calibrate false accepts. Include both left and right missing ranges in the audit; either can limit the augmented space.
6. If unmet, **append** a bounded block of genuinely new columns, orthogonalize incrementally, and retest. Count all candidate and holdout actions; cap work, return `basis_limited`/`rank_limited` honestly. Preserve only one user-facing `tol_basis`.

The fixed initial width, cap, and validation width are internal algorithm/resource parameters, not new physical tolerances. Benchmark `current-adaptive`, `one-shot fixed`, and `hybrid-with-holdout` on identical QTT snapshots and independent offline DD audits. A guaranteed speedup is **not assumed**.

## 1.2A Mandatory early "first impressions" experiment: randomized DD-CBE with **no symmetry sectors**

**Do this before implementing expensive p=4 GP stages or launching medium/large trajectories.** The 1D XX study exposed cases where adaptive sampling cost **more** than a successful fixed-width sketch. In QTT there is no `U(1)` block or per-sector column rounding: each two-site DD candidate is just a complex (GPE) or real (Heston) rectangular range-finding problem. This simplification may remove overhead, **but it does not guarantee that randomized sampling is faster**. Smooth QTT fields may have very small discarded spaces, favoring deterministic range extraction.

For a canonical TT bond with physical bit dimension `d=2` and retained rank `chi_i`, the geometric discarded dimensions are, before nullities from actual frames,

\[
 n_L^D=2\chi_{i-1}-\chi_i,\qquad
 n_R^D=2\chi_{i+1}-\chi_i,\qquad
 \operatorname{rank}(A_{DD})\le\min(n_L^D,n_R^D).
\]

When `n_L^D,n_R^D` are small, an **exact streamed discarded-basis action + QR** (or a tiny local SVD only if demonstrably cheaper) may beat *all* RSVD variants. Neither the sample length nor `L_QTT=30` alone determines whether randomized sampling pays off. The decisive parameters are actual **local DD dimension, numerical singular-value tail, MPO-channel cost, and how often the action changes**.

### First-impressions cost model and expected decision (test **before** building the full solver)

For the no-symmetry local DD matrix `A_DD` with `m=n_L^D`, `n=n_R^D`, a **right** random sketch of candidate width `ell` and an independent holdout of width `s` requires approximately `(ell+s)` streamed DD operator applications (plus retries and QR), whereas exhaustively applying `A_DD` to a basis on its `n`-dimensional right side requires `n` columns. When `n` is larger than `m`, the *left/mirrored* exhaustive action may be cheaper. Therefore, a simple **preflight rejection screen** is

\[
\boxed{\ell+s+\ell_{\rm retry}\ \gtrsim\ \min(m,n)
\quad\Longrightarrow\quad
\text{benchmark exact/deterministic DD first}.}
\]

This is a **cost heuristic**, not a theorem: projected MPO contractions, the two mirrored sketches, temporary tensor shapes, caching, and QR can change the actual work. The controller must choose based on real measured costs after the preflight calibration. Report `ell`, `s`, retries, `m`, `n`, useful added rank and total candidate+holdout actions **for every sampled bond**.

**Expected first impressions, not measured results:** dropping U(1) removes sector bookkeeping and sector-rounded probe allocations, which should simplify the implementation. It does not automatically make randomized selection worthwhile. For smooth GP Gaussians and exponential-affine Heston solutions, a few low-rank discarded directions and small DD dimensions may favor streamed deterministic expansion. The best chance for randomization is a larger, spectrally compressible DD space at genuinely active vortex/variance-coupling bonds, especially with higher attained `chi`. We must measure whether that regime appears *before* investing in a sophisticated controller.

For nonlinear GP the QTT density `|psi|^2` can initially have TT rank as large as `chi_psi^2` and must be compressed. Chen et al. (arXiv:2507.04279v5, Sec. 3.1) report this operation as a major approximately `O(D^4)` cost in their setup. **A faster CBE sketch cannot accelerate the dominant nonlinear-density rebuild if that is where the wall time goes.** In addition to component speedups, record

\[
 f_{\rm CBE}=\frac{t_{\rm CBE}}{t_{\rm whole\ step}},\qquad
 f_{\rho}=\frac{t_{\rm density\ build+compression+environments}}{t_{\rm whole\ step}},
\]

and an achieved **whole-step** speedup at matched physical error. A hypothetical 2x CBE speedup yields at most `1/(1 - f_CBE/2)` overall speedup if everything else is unchanged. Use this estimate as a quick stopping criterion when CBE is only a tiny part of the cost. An optional localized density-QTT update may be explored *only after* its action error is independently tested; it is not automatically accurate for a moving-paddle or implicit stage.

**Very first run:** take one GP quiet bond, one GP complex-vortex bond, one Heston smooth bond and one Heston correlated/diffusive bond, all with the **same incoming tensors** for every CBE arm. Benchmark `D0`/`D2`/`D3` on these four local problems (3 seeds, 3 warm timings), audit true DD residual offline, then run 10--20 full QTT steps on the two difficult fixtures. This produces the `first impressions` report **before** attempting p=4 or `L=30`. Expand the seed/audit count and parameter grid only if the initial results show a credible gain. **Do not invent quantitative savings in advance.**

### Prespecified small experiment matrix

| Fixture | Bits / local rank | Role | How to construct reproducibly |
|---|---|---|---|
| GP-quiet | `L=12,16` (`64^2`, `256^2`), target ranks 8/16 | Near-separable, low-DD control | 2D harmonic Gaussian or exact linear oscillator superposition; compress to defined TT rank |
| GP-structured | `L=16,20`, target ranks 16/32/64 if attainable | Nonlinear and phase-structured DD | Complex two-vortex field `psi = f(x,y) prod_j [(x-x_j)+i(y-y_j)]` with regularized cores; then one or two short source-consistent paddle evolution snapshots once available |
| Heston-smooth | `L=12,16`, ranks 8/16 when attainable | Near-exact low-rank control | Exact exponential-affine profile with nonzero `rho` and mixed derivative |
| Heston-hard | `L=16,20`, ranks 16/32/64 if attained | Larger DD tail | Regularized call payoff / correlated forward density snapshot, compressed to the same prespecified incoming rank across every arm |

`L=20` is `1024^2` grid values, **not** a required dense full-state oracle for production. Exact small *local* `A_DD` matrices may be formed **offline** for these tests only, where feasible. Record the true DD rank and spectral tail rather than forcing unattainable target ranks. Measure an active bond near the grouped `x|y` interface and a quiet bond away from it. The generated test cases are engineering fixtures, **not published vortex-shedding physics**.

### Exactly matched arms and metrics

- **D0:** deterministic streamed DD range (small sectors/dimensions only), rank-revealing QR; use as an offline truth reference when feasible.
- **D1:** current adaptive holdout controller from the XX implementation, generalized to no sectors.
- **D2:** fixed one-shot DD+QR sketch at `ell in {8,16,32}` where admissible (record whether a **separate validation** is included in *production* cost; do not compare certified and uncertified methods as though they had equal guarantees).
- **D3:** proposed hybrid predictor + one-shot sketch + **independent** 4–8-column holdout + incremental enlargement only on failure, always operating on the same `X=mat(F(psi))` and same frames.
- **Optional D4:** one-sided sketch at **matched retained rank** to quantify the DD projection's actual benefit on QTT. This is distinct from adaptive-width savings and must not be mixed into D1--D3 speedup claims.

For every fixture and arm, compute the independent exact/offline missing-range diagnostics

\[
 \rho_L=\frac{\|(I-Q_LQ_L^\dagger)A_{DD}\|_F}{\|A_{DD}\|_F},\qquad
 \rho_R=\frac{\|A_{DD}(I-Q_RQ_R^\dagger)\|_F}{\|A_{DD}\|_F},
\]

plus **absolute time-weighted** misses `h||residual||_F/||psi||`. When `||A_DD||` is tiny, **do not normalize** or interpret `rho`; report it as a trivial/inactive bond. Measure independent local Galerkin response error too, because a low DD Frobenius residual does not guarantee low full-step physical error.

**Count real work:** candidate-and-holdout projected MPO-channel contractions; streamed columns; QR/reorthogonalization; number of **useful independent directions**; left and right frame growth; rank and rank cap; density-QTT build/compression for GP; environment rebuild; local and root Krylov applications; allocations, peak working memory and warmed wall time. Time the whole CBE call (not just QR) and a 10–20-step solver loop **with and without** CBE after the local experiments. Use `3` paired seeds/repetitions for first timing impressions, then `>=30` random probe seeds for independently audited acceptance on the small matrices, expanding if the failure-rate estimate is ambiguous. Zero false accepts in this finite test is **not a mathematical guarantee**.

At most three attainable tolerance targets (`tol_basis = 1e-3, 1e-5, 1e-7` as starting values). Compare **equal achieved residual and equal final physical error**, not just equal nominal tolerance. For no-sector QTT, an input field of small actual rank must not be artificially padded merely to make RSVD appear valuable.

### Early decision gate, before Task 2

1. First plot: **independent true DD residual versus total streamed candidate+holdout actions** at easy and hard GP/Heston bonds, with all unsuccessful target points marked.
2. Second plot: **CBE call wall time versus achieved DD residual**, together with selected rank and peak bytes. Include the cheap deterministic branch.
3. Third compact plot/table: **10–20 step whole-solver time-to-accuracy and rank growth**, to see whether local gains survive the root and nonlinear MPO costs.
4. **Go / no-go:** keep the randomized hybrid only for measured regimes where it is reliably cheaper than the best **accuracy-matched** deterministic or fixed-width control (a suggested engineering target is at least ~20% lower CBE time, not a promised outcome). Otherwise use a **size-aware deterministic/fixed-width fallback** and keep DD projection itself. Record first impressions in `qtt_cbe_nosym_first_impressions.md` and `qtt_cbe_nosym_ablation.csv`; no large production campaign is needed to make this decision.

**Nonlinearity distinction:** the sketched object is the **current vector field**, `A_DD=P_L^D mat(F(t,psi)) P_R^D`. Its right sketch is `A_DD^dagger Omega_L`, i.e. the *conjugate transpose of that reshaped vector-field tensor*. It is **not** automatically `DF(psi)^dagger Omega_L` and is not obtained merely by substituting an adjoint Hamiltonian for a nonlinear Jacobian. A separate Jacobian-vector product is required only for the implicit nonlinear stage solver (Task 2).

## 1.3 Adaptive local Krylov and recycling policy

For a *linear* non-Hermitian local generator `G_i`, use Arnoldi rather than relying on Hermitian Lanczos:

\[
 \mathcal K_m(G_i,v)=\operatorname{span}\{v,G_iv,\ldots,G_i^{m-1}v\},\qquad
 \varepsilon_K=\frac{\|(I-Q_mQ_m^\dagger)e^{hG_i}v\|}{\|v\|}.
\]

Continue adaptive **time-weighted** stopping, calibrated against the independent response error. The existing XX fixture exceeded its error target in 3 of 54 cases, so add a safety margin/verification guard; do not add a public fixed iteration-depth knob.

**Do not enable the old full-image recycling.** If selective recycling is evaluated at all, first project transported old vectors outside the frame already retained by BUG,

\[
 U_{\rm cand}=(I-F_{\rm new}F_{\rm new}^\dagger)\,TQ_{\rm old}.
\]

Retain only one or two independent vectors with demonstrable response relevance. Use a **cost gate**, in units of full operator applications plus transport/orthogonalization:

\[
 C_{\rm transport}+C_{\rm refresh}+C_{\rm orth}+C_{\rm continue}
 < C_{\rm fresh}.
\]

Otherwise execute fresh adaptive Arnoldi. Preserve the option of **GMRES deflation/recycling across repeated implicit root stage systems** with sufficiently similar operators; this is a different, more plausible reuse setting than K/L gauge-changing half-sweeps. Never reuse stale `G_i q` after local operator or frame changes.

## 1.3A Exact nonlinear action and stage-aware environment implementation (no `g=0` shortcut)

Use `rho_psi=psi .* conj(psi)` in QTT form, **variationally compress** this density where available, and build `W_nl = g diag(rho_psi)` as a matrix-free diagonal MPO. Keep the kinetic/paddle/nonlinear terms separate so that **static** operator contractions can be reused; cache tags must include `psi`/stage and `t` to avoid reuse of stale density or moving-paddle environments. The raw uncompressed Hadamard product may have rank up to `chi_psi^2`, so record `chi_rho`, compression error and the full density-build cost. This cost is a potential **bottleneck**, independently of CBE.

For a trial stage state `Y_j` at `t_n+c_j h`, evaluate

\[
 \rho_j=\mathcal C_{\varepsilon_\rho}(|Y_j|^2),\quad
 H_j=H_{\rm kin}+V(t_n+c_jh)-\mu+g\operatorname{diag}(\rho_j),\quad
 F_j=(-i-\gamma)H_jY_j.
\]

The compression error should satisfy an independently checked *action* budget, e.g. `h |g| ||(rho_j-|Y_j|^2)Y_j||` below its allocated share of `tol_basis`/stage accuracy. The nonlinear TT product is not allowed to be truncated in an uncontrolled additional step. Make `F_j` matrix-free through ordinary TT/MPO environments and use the **true** stage `F_j` when computing CBE enrichment; use a different **linearized JVP** only in Newton--Krylov as needed.

**Two implementation modes:**

- **A / lagged exploratory mode:** build `rho_n` once, hold `H[psi_n]` fixed through a step, apply existing linear root solve; label results `frozen_density`, never claim p2/p3/p4 solely from the linear stage solver's formal order. This is a cheap smoke and ablation, not the preferred high-order nonlinear solver.
- **B / consistent nonlinear mode (production candidate):** recompute `rho_j` for every nonlinear stage iterate; solve the nonlinear Radau (or other validated) stage equations to an independent convergence tolerance; use `DF` with the `psi^2 delta conj(psi)` contribution for Jacobian actions; update the BUG basis with the stage/predictor **nonlinear action**. Verify order after the entire basis-plus-root update, not by the root solver alone.

A static Hamiltonian MPO action can use standard linear Arnoldi, while `F(psi)` must be evaluated as a **nonlinear vector field**. There is no single global `exp(h H)` operator for the full GP flow with general `g != 0`.

## 1.4 Test matrix, measured outputs and Task-1 release gates

- **U01 / indexing:** at `n_x,n_y in {2,3,4}`, grouped QTT -> dense samples -> QTT round-trip, every bit pattern, boundary wrap, grid weights and tensor canonical gauges.
- **U02 / operators:** every `D_x`, `D_y`, `D_xx`, `D_yy`, `D_xv`, coordinates, GP/Heston MPO and adjoint equals independent sparse-grid application at `8x8` and `16x16` to floating-point accuracy (within the discretization's algebraic representation).
- **U03 / environments:** a single-site and two-site environment-based action matches explicit small-grid contraction for **both real and complex** tensors, including non-Hermitian adjoints. Confirm invalidation when `psi` or the moving potential changes.
- **U04 / DD:** streamed nonsymmetric `P_D^L X P_D^R Omega` and mirrored adjoint match explicit small-matrix sketches. Audit DD true residual, candidate plus holdout counts, independent random seeds, false acceptances, peak memory, QR orthogonality.
- **U05 / rank:** starting from a rank-1 separable field, an intentionally nonseparable 2D test grows ranks only as needed; a separable solution does not waste rank, and tightening `tol_basis` decreases missing-action error before the floor.
- **U06 / nonlinear:** `|psi|^2 psi` via QTT compression agrees with elementwise dense evaluation; controlled `g=0` reduces identically to the linear path; **density MPO rebuilt at trial stages**, cached tags invalidated, correct real-linear derivative and Jacobian-vector products; static vs refreshed nonlinear environments produce the expected different results.
- **U07 / code audit:** confirm existing `implicit trapezoidal`, `Radau IIA`, `matrix-free GMRES`, 2/3/4-order BUG dispatch and nonsymmetric DD kernels. Mark *present / needs adaptation / missing* with actual source file and method names. No large-scale runs before results are logged.
- **U08 / no-symmetry DD first impressions:** four prespecified GP/Heston fixtures, independent true residuals/false-acceptance audits, full cost accounting, robust rank adaptation and honest deterministic fallback; no sector-rounded probe allocations.

**Pass:** all small-grid parity tests pass, no dense large-grid fallback, bounded sampled columns and rank working space, and the hybrid CBE mode does not falsify tolerance success. **Mandatory pre-Task-2 evidence:** `qtt_cbe_nosym_first_impressions.md` and ablation CSV, with **actual** GP/Heston non-sector DD-size distributions, a comparison to cheap deterministic selection, and separate totals for nonlinear-density construction. The randomized route is optional when those numbers disfavor it.

---

# TASK 2 — Analytically validated 2D GP/Heston evolution; identify TDVP2 backward amplification; verify BUG order 2/3/4

**Outcome:** independent exact/semianalytic oracles for both PDEs, verified time orders, and an honest error/stability/time-to-accuracy map for TDVP2 vs CBE–BUG.

## 2.1 GP exact 2D oracle A: damped linear harmonic oscillator

Use `hbar=m=omega=1`, `g=0`, 2D harmonic trap `H_0=-1/2 Delta + (x^2+y^2)/2`, and the **specific damping convention of Connor et al.**:

\[
\boxed{ i\partial_t\psi=(1-i\gamma)(H_0-\mu)\psi,\quad
 \partial_t\psi=(-i-\gamma)(H_0-\mu)\psi. }
\]

On `R^2`, let `phi_{mn}(x,y)=phi_m(x) phi_n(y)` be normalized oscillator eigenmodes with `E_mn=m+n+1`. Set e.g.

\[
 \psi(0)=\mathcal N\,[\phi_{00}+0.25\phi_{10}+0.15i\phi_{01}],\qquad
 \boxed{\psi(t)=\mathcal N\sum_{(m,n)}c_{mn}
 e^{(-i-\gamma)(E_{mn}-\mu)t}\phi_{mn}. }
\]

Choose `gamma in {0,0.1,1}` and `mu=0` for a genuinely contractive test; add `mu=E_00` for a stationary component plus decaying excitations. Monitor exact complex state, mode coefficients, density, norm decay, and error at fixed `T`. The analytic eigenfunctions live on `R^2`; on a periodic finite box they are only an **asymptotic spatial reference** if tails and their derivatives are negligible at the boundary. Independently compare to `expm(h G_grid)` on a small discrete grid to separate **time-integration** error from spatial/boundary error.

## 2.2 GP exact 2D oracle B: *nonlinear* dissipative plane wave

Use a periodic two-coordinate domain with `V=0`, arbitrary allowed wave-vector `k=(k_x,k_y)` and `g>0`:

\[
 \psi(x,y,t)=\sqrt{\rho(t)}\;e^{i(k_xx+k_yy+\varphi(t))},\qquad
 \alpha=\tfrac12(k_x^2+k_y^2)-\mu.
\]

For the same damping convention `psi_t=(-i-gamma)[-Delta/2-mu+g|psi|^2]psi`, density and phase obey

\[
 \boxed{\dot\rho=-2\gamma(\alpha+g\rho)\rho,\qquad
 \dot\varphi=-(\alpha+g\rho).}
\]

For `gamma>0` and `alpha != 0`, the exact solution is

\[
 \boxed{\rho(t)=\frac{\rho_0 e^{-2\gamma\alpha t}}
 {1+\frac{g\rho_0}{\alpha}(1-e^{-2\gamma\alpha t})},\quad
 \varphi(t)=\varphi_0+\frac{1}{2\gamma}\ln\frac{\rho(t)}{\rho_0}.}
\]

For `alpha=0`, `rho(t)=rho_0/(1+2 gamma g rho_0 t)`; for `gamma=0`, density is constant and phase advances at `-(alpha+g rho_0)t`. This test covers the *actual nonlinear action* and dissipative phase, including the case where norm need not monotonically fall for arbitrary `mu`. It does **not** probe vortex formation or rank growth; those are different tests.

## 2.3 Heston exact 2D oracle: exponential-affine profiles

In time to maturity `tau`, solve `u_tau=A_H u`. Start with the intentionally variance-dependent test function `u(x,v,0)=exp(a x+b v)` with nonzero `a,b` and nonzero `rho`, and make the ansatz

\[
\boxed{u(x,v,\tau)=\exp(a x+B(\tau)v+C(\tau)).}
\]

Substitution into the full 2D Heston PDE yields the exact Riccati system

\[
\begin{aligned}
 B' &= \tfrac12\sigma^2 B^2+(\rho\sigma a-\kappa)B+\tfrac12(a^2-a), &B(0)&=b,\\
 C' &= (r-q)a+\kappa\theta B-r, &C(0)&=0.
\end{aligned}
\]

Integrate these scalar equations to near machine precision (or use the equivalent closed affine formula, with correct branch choices) to obtain the **exact function on the 2D continuous domain**, up to the one-dimensional reference evaluation. Use `a in {0.3,0.7}` and small negative `b` for initial fixtures, `rho in {0,-0.4,-0.7}`, and positive `kappa,theta,sigma` in a documented Feller regime. Set **finite-box boundary values from this exact analytic function**, including the degenerate `v=0` boundary treatment. Do not confuse the exact continuum solution with `exp(t A_h)` of the discrete grid operator: use both references to isolate spatial and temporal errors.

Additional Heston tests:

- `u(x,v,0)=1` gives `u(x,v,tau)=exp(-r tau)` for constant rate, a simple discounted constant-mode invariant.
- `u(x,v,0)=exp(x)` with `q=0` gives `u=exp(x)` (discounted asset-price martingale mode); include as an operator/price sanity check.
- For a standard European call payoff `(exp(x)-K)_+`, compare at **ordinary maturities** against Heston's independent **semi-analytic Fourier integration** or a highly refined 2D grid reference. The general call value is not an elementary 2D closed-form function.
- For long-time forward densities, compare moment generating functions `int exp(a x+b v) p(x,v,t) dx dv` to the same affine solution where integrable. Check positivity and probability conservation for the undiscounted forward operator with its boundary fluxes.

## 2.3A Nonlinear TDVP2 and BUG must solve the **same** GP equation

For the full GP define the state-dependent operator `H[psi,t] = H_kin+V(t)-mu+g diag(|psi|^2)` and `F(psi,t)=(-i-gamma)H[psi,t]psi`.

**What is implemented as a *new* method, not claimed by Connor Appendix A.2:**

1. `TDVP2_lagged`: at step start, assemble density QTT from `psi_n`, compress, build real-space diagonal `H_d[psi_n]`, and run a standard two-site TDVP projector-splitting sweep using this **temporarily frozen** operator. Recompute the nonlinear MPO on the next accepted step. Explicitly monitor the error induced by lagging; method may have reduced temporal order even if the *linear* local integrator is high order.
2. `TDVP2_consistent`: use the same two-site TDVP geometry, but recompute the density and local environments as the trial `psi` changes in a **self-consistent nonlinear substep / predictor-corrector iteration**, including the negative/backward center stages. Solve each nonlinear local subflow to its allotted residual, rather than applying one frozen `exp(hH)` and calling the nonlinear update exact. Establish its order numerically; the word "TDVP2" refers to **two-site** updates, not automatically **second-order** integration in time.
3. `BUG_nonlinear`: evaluate `F(t_j,Y_j)` at every Radau stage and use it for the DD basis predictor as well as for the Galerkin root solve. Update `rho_j`, its compression and its environments whenever a nonlinear iterate changes. Use full Newton/Picard convergence checks and the correct *real*-linear JVP. In particular,

\[
 \mathrm{D}F(\psi)[\delta\psi]=(-i-\gamma)
 \Big[\big(H_{\rm kin}+V-\mu+2g|\psi|^2\big)\delta\psi
       +g\psi^2\overline{\delta\psi}\Big],
\]

with the time-dependent potential held at the chosen stage time. Treat complex conjugation explicitly in a doubled-real or Wirtinger-block implementation; do **not** incorrectly use a complex-linear Jacobian. A Jacobian is a numerical *linearization inside the iterative solve*, not an approximation to the underlying PDE.
4. A dense-grid FFT split-step or high-order RK reference must evaluate the **same** `g, gamma, mu, V(t)` and domain, using sufficiently fine `dt` and grid. An **independent** numerical reference, not the Connor Appendix's TDVP itself, determines whether our large timestep is accurate.

**Mandatory anti-shortcut unit test:** compare `frozen_density` with `stage_refreshed` on the exactly solvable **nonlinear dissipative plane wave** and on a smooth spatially modulated nonlinear state at several step sizes. Check that the full `p=2,3,4` method retains its measured order only with its required stage consistency; if freezing happens to be exact on a special plane-wave case, the second, spatially structured fixture must expose the distinction. The GP oracle with `g=0` remains a separate correctness test and cannot certify nonlinear order.

**Source interpretation:** Connor et al.'s Appendix A.2 does **not** document the refresh schedule or specify the two-site variant; the two baselines above are **our controlled implementations**. The QTT paper of Chen et al. explicitly refreshes and compresses density at the end of each step, which is a directly reproducible *step-lagged baseline*, not independent evidence for higher-order nonlinear stages.

## 2.4 Real and imaginary time stepping at p=2,3,4

Denote the reduced root frame by `W,Z` and the root state `S`:

\[
 \mathcal G_{0,t}(S)=W^\dagger F_t(WSZ^\dagger) Z,
 \qquad S_n=W^\dagger \psi_n Z,
\]

where `F_t` is the correct **linear operator** (Heston, linear GP) or **nonlinear vector field** (full GP), evaluated consistently at every implicit stage. The root begins from the **original step input** projected into the newly constructed augmented frames, not a predictor state.

Implement and independently validate:

| Target BUG order | Implicit root method | Property/limitation |
|---|---|---|
| `p=2` | Existing implicit trapezoidal / Crank–Nicolson | Order 2, **A-stable but not L-stable**; stiff negative modes are not strongly damped for very large `h |lambda|` |
| `p=3` | Existing **2-stage Radau IIA** | Order 3, **L-stable**; suitable stiff baseline |
| `p=4` | **3-stage Radau IIA** root (classical order 5) plus a separately order-4-verified BUG basis construction | L-stable root of order 5 supports an overall order-4 method; **it does not establish order 4 by itself** |

For linear root `S'=G_0 S`, an implicit Runge–Kutta step solves

\[
 K_j=G_0\left(S_n+h\sum_{\ell=1}^s a_{j\ell}K_\ell\right),\quad
 S_{n+1}=S_n+h\sum_{j=1}^s b_jK_j.
\]

Use **matrix-free block GMRES**, recycling/deflation only if beneficial, with preconditioning based on dominant separable diffusion/kinetic terms and no explicit root matrix. For nonlinear GP, stages instead satisfy

\[
 K_j=F_{t_n+c_jh}\left(S_n+h\sum_{\ell=1}^s a_{j\ell}K_\ell\right).
\]

Use Newton–Krylov or converged Picard **inside the stage equations**. The real-linear derivative of the GP nonlinearity is

\[
 \boxed{D\big(|\psi|^2\psi\big)[\delta\psi]
 =2|\psi|^2\delta\psi+\psi^2\delta\bar\psi,}
\]

so a complex-linear Jacobian that ignores `delta conj(psi)` is incorrect; use two real fields or the appropriate Wirtinger block representation. At each stage evaluate `V_pad(t_n+c_j h)`, update the nonlinear-density MPO, and ensure its compression error is below the stage residual budget. Merely freezing `|psi_n|^2` once per step can reduce physical time order.

For every `p`, verify actual local and global convergence of the **whole BUG update** (basis predictor/enrichment, root stage, truncation) rather than assuming `p` augmentation rounds imply order `p`:

\[
 \|\Phi_h(\psi)-\psi(t+h)\|=O(h^{p+1}),\qquad
 \|\Phi_h^{T/h}(\psi_0)-\psi(T)\|=O(h^p).
\]

Try `h in {0.1,0.05,0.025,0.0125}` in a regime where errors are measurable and the spatial/truncation/GMRES errors are smaller. Use a **nontrivial, already entangled/rank-rich** initial state too: the previous XX domain-wall product-state experiment never resolved the desired global third-order slope.

## 2.5 Dissipativity, backward-stage amplification, and fair TDVP2 comparison

For an exact linear GP test with `H_0-mu` positive semidefinite, the continuous generator is `G=(-i-gamma)(H_0-mu)`, with

\[
 \frac{d}{dt}\|\psi\|^2=-2\gamma\langle\psi,(H_0-\mu)\psi\rangle\le0.
\]

For full nonlinear GP with static potential, define the grand-canonical Lyapunov functional

\[
 \mathcal F[\psi]=\int\left[\tfrac12|\nabla\psi|^2+(V-\mu)|\psi|^2+\tfrac g2|\psi|^4\right]dx\,dy.
\]

Writing `A(psi)=(-Delta/2+V-mu+g|psi|^2)psi`, the continuous damped flow obeys

\[
 \boxed{\frac{d\mathcal F}{dt}=-2\gamma\|A(\psi)\|_{L^2}^2\le0}
\]

for time-independent `V` and compatible boundaries. **With a moving paddle**, add the external power `int (partial_t V_pad)|psi|^2 dxdy`; therefore do not demand monotonically decreasing `F` during driving. A fixed `mu` does **not** ensure monotonic particle number for every initial state.

A conventional projector-splitting/2site-TDVP tangent projector has components symbolically

\[
 P_T=P_{\rm left}+P_{\rm right}-P_{\rm overlap}.
\]

The subtraction produces a **backward center/S substep**. For a dissipative test eigenmode `G v=-lambda v` with `lambda>0`, the isolated reversed substep amplifies as `exp(+h lambda)`. Whether that causes full-method instability depends on discretization, cancellation, truncation, time integrator and stabilization. Instrument **actual** per-substep amplification:

\[
 A_{\rm back}(n,i)=\frac{\|S^{\rm(after\,negative\,stage)}_{n,i}\|}
 {\|S^{\rm(before\,negative\,stage)}_{n,i}\|},\qquad
 A_{\max}(h)=\max_{n,i}A_{\rm back}(n,i).
\]

Record local center stage norm, global physical norm/contractive observable, accepted-step status, local residual, final error, internal Krylov/GMRES counts and peak working rank. **Never diagnose a blow-up solely from one intermediate stage norm** if the full step remains accurate and stable.

**Required baselines** (same spatial generator/MPO where relevant, same state and rank/tolerance semantics): conventional two-site projector-splitting TDVP with explicit backward stage, a stiff-stable/implicit or split-diffusion TDVP control, BUG at p2/3/4, and an appropriate independent dense/spectral small-grid reference. Compare both at equal `h` for diagnosis and at **matched achieved physical error** for the performance claim. If the stabilized TDVP baseline wins or survives `h=0.1`, report it.

## 2.6 Task-2 tests and release gates

- **V01** linear 2D GP eigenfunction reference (`gamma=0,0.1,1`) + exact discrete expm reference; mode coefficients, complex phase, norm, error.
- **V02** nonlinear dissipative plane-wave GP reference, including `alpha=0` and `alpha!=0`; density, phase, compressed nonlinearity, **frozen vs stage-refreshed MPO** and stage consistency. Include a non-plane-wave smooth `g>0` case to detect temporal-order loss due to lagging.
- **V03** exact Heston affine profile, nonzero `rho`, including mixed derivative and adjoint; constant and asset modes, exact boundary traces; positivity for appropriate payoff fixtures.
- **V04** spatial convergence `n=6,7,8` bits/coordinate and time convergence separately; never attribute boundary/discretization error to BUG time stepping.
- **V05** full BUG `p=2,3,4`: local slopes approximately `p+1`, global slopes approximately `p`, within a documented fit window; no root/truncation floor. If order 4 fails, label it **experimental** and block order-4 production.
- **V06** independent `tol_basis`, `tol_root`, `tol_trunc` sweeps; rank caps and false accepts explicitly flagged; no hidden Krylov fixed-depth or stale nonlinear MPOs.
- **V07** TDVP stage amplification vs `h`, `gamma`/diffusion stiffness, PDE modes and time; **lagged vs stage-consistent nonlinear TDVP2**, full-step error, stability, and wall time. Compare a stabilized TDVP baseline and do not equate an unstable **substage** with an unstable complete step.
- **V08** `A-stable vs L-stable`: stiff scalar/matrix/2D diffusion eigenmode tests showing why trapezoidal is not an adequate sole stiff p2 solution. Optional alternative order-2 L-stable root may be needed if oscillatory stiff remainders block p2.

**Pass:** exact nonlinear/linear and Heston tests, verified requested order for each released solver, honest independent error budgets, and a reproducible stability mechanism and cost comparison. The hypothesis that `h=0.1` is safe and accurate is tested, not inserted as a hard-coded success condition.

---

# TASK 3 — GP imaginary-time ground states; nonlinear vortex-shedding and Heston diffusion pilots; rank growth and competitive baselines

**Outcome:** reproduce recognizable source-level physics, demonstrate controlled QTT rank adaptivity, and determine when large timesteps are **accurate**, not merely numerically survivable.

## 3.1 GP imaginary-time evolution and logarithmically expanding steps

For stationary trap `H_0=-Delta/2+V_trap-Omega L_z`, minimize the normalized GPE energy

\[
 E[\psi]=\langle\psi,H_0\psi\rangle+\tfrac g2\int|\psi|^4\,dxdy,
 \qquad \|\psi\|_{L^2}=1.
\]

The chemical potential and normalized imaginary-time flow are

\[
 \mu[\psi]=\langle\psi,(H_0+g|\psi|^2)\psi\rangle,
 \qquad \boxed{\partial_\tau\psi=-\left(H_0+g|\psi|^2-\mu[\psi]\right)\psi.}
\]

In exact arithmetic with a time-independent trap and this normalized gradient flow,

\[
 \frac{dE}{d\tau}
 =-2\left\|(H_0+g|\psi|^2-\mu[\psi])\psi\right\|^2\le0.
\]

Compute the **stationarity residual** `R_mu=||(H_0+g|psi|^2-mu[psi])psi||`, energy, chemical potential, norm, bond dimension and imaginary time. For trapped/rotating systems, start with `g=0, Omega=0` (exact 2D Gaussian), then `g=100, Omega=0` (nonlinear 2D trapped ground state), then a rotating case such as `Omega=0.8, g=100` from Chen et al. The latter requires an independently converged dense/spectral or variational reference, not an invented analytic state. Check vortex-lattice topological charges, energy and `mu`, not only an image comparison.

**Two timestep policies**, with the same implicit BUG p2/3/4 implementation, allowed rank cap, requested final accuracy, and effective operator:

1. **Uniform:** `Delta tau_k=h_const`, tuned across the same admissible step ceiling.
2. **Logarithmically/geometrically expanding:** `Delta tau_k=min(h_max, h_0 q^k)` with `q>1`, or equivalent logarithmically spaced time nodes. Specify the schedule in the job manifest; `q` is an experiment-level scheduling parameter, **not** a physical accuracy tolerance.

Both policies must reject/reduce steps if the nonlinear implicit residual, discrete energy condition, truncation, or reference error is unacceptable. Periodically re-normalize in the physical quadrature. Do not compare schedules at equal *number of steps* only; compare CPU time, total nonlinear/GMRES actions, number of accepted/rejected steps, **energy and stationary residual reached**.

**Competition with arXiv:2507.04279:** include QTT-TDVP imaginary time, **QTT gradient descent** (the paper's faster ground-state baseline), and when feasible a standard BEFD/BESP reference. The meaningful contribution is *time to a target ground-state energy/stationarity residual at controlled ranks*, not merely that BUG can take a larger `Delta tau`. Confirm whether logarithmic steps improve time-to-accuracy over a tuned constant-step implicit method; do not presuppose improvement.

## 3.2 2D dissipative nonlinear GP vortex-shedding test with rank growth

Follow the **equation and physical setup**, not its exact method, from Connor et al.:

\[
 \boxed{\partial_t\psi=(-i-\gamma)\left[-\tfrac12\Delta
 +V_{\rm trap}(x,y)+V_{\rm pad}(x,y,t)-\mu+g|\psi|^2\right]\psi,}
\]

with a rounded moving paddle `V_pad`, e.g. `x_pad(t)=x_0+v_sweep t` until a recorded stopping point, followed by a **specified** ramp-down to prevent perpetual re-entry through a periodic domain. Implement the actual rounded paddle shape/potential from the selected source fixture, not an arbitrary discontinuous wall that introduces unresolvable grid high-frequency content.

**Paper-reproduction pilot:** dimensionless `g=1`, `gamma=0.1`, `v_sweep=1.5`, box length `L_box=64`, `256x256` points, source-style `dt=0.001` high-resolution spectral/split-step reference. Record domain boundaries, `mu`, paddle height/width/smoothing and exact motion from source figure/appendix before finalizing the fixture. For a trapped four-paddle late-time alternative the source uses `L_box=35`, `omega=0.15 sqrt(2)`, `g=1`, `gamma=0.1`, a ramp-down and sweep speeds `0.5/1.5`: do not mix the single- and four-paddle setups accidentally.

**Observe** `n=|psi|^2`, `phase=arg(psi)`, quantized circulation and signed vortex count,

\[
 q_\Gamma=\frac{1}{2\pi}\oint_\Gamma\nabla\arg(\psi)\cdot d\ell\in\mathbb Z,
\]

excluding poorly conditioned density-zero pixels in phase-unwrapping; vortex-core trajectories/annihilation, local density dip, gradients, kinetic and interaction energies, grand-canonical functional (after forcing stops), and QTT bond dimension `chi_i(t)` at every bond. For a moving paddle with `V_pad(t)`, include the external work term before interpreting dissipation.

**Rank-adaptive hypothesis:** as shed vortices and fine phase/density structure appear, **some** TT bonds may need to grow. Plot `max_i chi_i(t)`, full bond profile, QTT singular spectra, cumulative discarded weights and the controlled DD range defect, together with **vortex count and characteristic healing-scale resolution**. Rank growth is not guaranteed to be monotone, and rank saturation is physically plausible; distinguish physical structure generation from unnecessary rank inflation.

**Time-step test:** start from source-like small `h` for the reference and step through `{0.005,0.01,0.025,0.05,0.1}` as **candidate** BUG steps, with equal-accuracy TDVP2 and stabilized/split-step comparisons. Do not force `h=0.1` during rapid paddle motion if it fails vortex-count, phase, force-motion, or solution-error checks. If `h=0.1` is only accurate **after** the stirring ramp, that is a valid late-time result. Record each period's actual accepted-step histogram.

To compete credibly with the existing papers, compare **against their published time scale, grid size, physical quantities and baseline algorithms**, acknowledging differences in derivative discretization (finite difference vs QFT/spectral) and ordering. An observed difference caused only by switching spatial accuracy is not evidence of a better integrator.

## 3.3 2D Heston dissipative pilot, including forward density for long-time dynamics

**Backward PDE validation/medium pilot**: choose nonzero mixed derivative `rho != 0`, smooth variance-dependent affine initial data with exact Riccati reference; then an option payoff `(e^x-K)_+` with its kink, where new small-scale features are smoothed by diffusion. Report discounted price, delta `partial_x u`, gamma `partial_xx u`, vega `partial_v u`, and mixed derivative. For a true pricing setup use realistic maturities to show financial relevance; do not present `T=500` as a typical option maturity.

**Forward-density optional but recommended for meaningful late dynamics:** evolve `p_t=A_H^* p` **without the pricing discount**, from a normalized localized positive density (e.g. product of narrow Gaussian in log price and truncated positive variance density). This is still the *same Heston stochastic model*, not a third model. Verify

\[
 \int p\,dx\,dv=1,\quad p\ge0,\quad
 \int e^{a x+b v}p(x,v,t)\,dx\,dv
 =\mathbb E[e^{aX_t+bV_t}]
\]

against affine-transform moments when finite. Handle degenerate `v=0` through the correct zero-flux/Feller treatment; require that the x-domain follows the widening/drifting distribution (moving window or sufficiently wide range). For a discount-augmented density, mass decays as `e^{-r t}`; factor this out before claiming high relative accuracy at large `t`.

**Heston rank hypothesis:** nonseparable correlation `rho`, initial strike kink, boundaries and x–v coupling may increase QTT rank transiently; diffusion often **decreases** spatial complexity eventually. A growth-only rank plot is scientifically inappropriate. Show rank *adaptation both upwards and downwards*, with the same independent error controls.

## 3.4 Mandatory controlled comparisons and minimum figures

At one representative medium 2D grid (`2^8 x 2^8` for GP source reproduction, then `2^10 x 2^10` or larger for QTT scaling as affordable), compare:

- `BUG_p2`, `BUG_p3`, `BUG_p4` with the **same hybrid DD range controller**, adaptive local Krylov, implicit root, and no full-vector recycle.
- Conventional `TDVP2` with measured negative center step **and** a stabilized/implicit TDVP/split-dissipation control where available.
- GP: independent small-step spectral split-step/RK reference and QTT gradient-descent reference for ground states; Heston: exact affine and dense-grid/transform references where applicable.
- At matched physical solution error, report accepted `h`, total wall time, total operator/Jacobian actions, peak **working** and final `chi`, error and backward amplification. Equal timestep alone is not a fair order-2-vs-order-4 performance comparison.

**Minimum publication-quality pilot figure set** (keep raw arrays):

1. **Accuracy vs timestep and time-to-accuracy:** 2D exact GP/Heston plus `p=2/3/4`, TDVP2 and stabilized controls; separate spatial from temporal error.
2. **Dissipative stability mechanism:** TDVP backward-stage amplification, physical norm/Lyapunov/positivity, accepted-step size, rejected steps and exact solution error. Do **not** label a method unstable if it merely has an amplifying internal stage but stable accurate full steps.
3. **Imaginary-time GP:** energy and stationarity residual vs **wall time** for uniform vs log-expanding implicit BUG and QTT-TDVP/GD; vortex ground-state density/phase as a paired physics panel.
4. **Rank-adaptive physics:** GP density/phase/vortex count and `chi_i(t)` during shedding and relaxation; Heston price/density, Greeks or distribution moments and `chi_i(t)`.

**Pass:** physically converged medium runs, a meaningful rank-adaptive response, and a measured timestep–accuracy–cost envelope. A case where TDVP2 remains competitive is a documented result, not a failure to be hidden.

---

# TASK 4 — Gate and execute 30-bit, T=500, h_max=0.1 long-time QTT campaigns

**Outcome:** independent, reproducible, long-time dissipative GP and Heston results with confidence in physical observables, controlled ranks, and a fair comparison of the *largest accurate timestep* and time-to-accuracy.

## 4.1 Production configuration and staged release

Use grouped `n_x=n_y=15` quantics bits (`2^30` grid values, never dense). Default candidate `T=500` and `h_max=0.1`; a constant `h=0.1` run would require **5,000 accepted timesteps** if it remains accurate throughout. Build identical initial conditions and spatial MPOs for all solver arms and record physical grid spacing; increasing binary resolution alone does not verify that the vortex healing length or option payoff boundary layer is resolved.

**Stage A / 2D correctness:** `8x8`, `16x16`, `64x64` tiny/dense-parity and exact references; verify full solver p2/3/4 and environmental updates.

**Stage B / physical medium pilots:** `256x256` source-matched GP vortex shedding and Heston affine/payoff cases, then one `1024x1024` or equivalent higher-resolution pilot. Measure working rank (not just final compressed rank), operator rank, Radau/GMRES solve counts, stage nonlinearity iterations, and time-to-accuracy.

**Stage C / QTT scaling:** `L in {20,24,28,30}`, always 50/50 bits per coordinate where possible, same physical domain, resolution and operator conventions, with a **short production-shaped trajectory**, checkpoint/restart, memory, and relevant quantics rank per bond. Large `L` is not automatically beneficial if physical small scales are already resolved.

**Stage D / T=500:** run the selected GP vortex-shedding-plus-relaxation case and the selected Heston forward-density (and optionally backward discounted) case only after accepted accuracy/stability/resource checks. The paper's GP stirring phase is short compared with `T=500`; specify when the paddle stops and ramps down, then track genuine free relaxation and late-time vortex annihilation.

**Suggested experimental tolerances (not guaranteed achievable):** `tol_basis in {1e-5,1e-7}`, `tol_trunc in {1e-6,1e-8}`, `tol_root <=1e-10`, initial rank caps `chi_max in {128,256,512}` as resources permit. Require one-at-a-time convergence checks and `tol_nonlinear`, if separately implemented, strictly below the target time error. Do not implicitly identify `tol_root` with timestep error; use an independent step-check target.

## 4.2 Large-step acceptance and when a claim is earned

For each proposed step `h<=0.1`, use occasional **full vs two half-step** checks under matched MPO/nonlinear controls, plus stage residual and physical diagnostics:

\[
 \delta_h=\|\Phi_h(\psi_n)-\Phi_{h/2}(\Phi_{h/2}(\psi_n))\|,
 \qquad \widehat e_h\approx\frac{\delta_h}{1-2^{-p}}
\]

when the method is actually global order `p` and is in the smooth asymptotic regime. This estimate can fail near truncation floors, rapidly varying forcing, and nonlinear-stage inaccuracies. Calibrate against stronger reference segments. Accept only if the actual model's **observable error** and numerical residuals satisfy a predeclared quality target; otherwise reject/reduce `h`. For GP use density/phase and vortex trajectories/counts, not just normalized state overlap. For Heston use exact affine/option or moment error and positivity/conservation.

**The primary success criterion is not “BUG runs at 0.1 and TDVP2 crashes.”** It is

\[
\boxed{\text{BUG reaches the same target physical error at larger accepted }h
\text{ and lower total wall time/peak work than the relevant TDVP baseline.}}
\]

If `h=0.1` is accurate only for smooth late-time dynamics, state that. If accurate GP vortex shedding requires `h<<0.1`, use the validated smaller steps during stirring and only enlarge during relaxation. If TDVP2 can be stabilized with implicit/splitting remedies, include those cost/accuracy results. In the Heston long-time pricing case, factor out `e^{-r t}` when evaluating meaningful relative accuracy; avoid claiming dynamics from a trivially vanishing price.

## 4.3 Observables and figures for T=500

### GP: shedding, coarsening, and damping

Save **QTT-compressed** state/checkpoints and produce spatial images by selective TT evaluation or validated reduced-resolution sampling, not a dense `2^30` conversion. Plot at predeclared snapshots, e.g. `t={0,5,10,20,50,100,200,350,500}` (adjust the early snapshots to the actual paddle schedule):

- `|psi(x,y,t)|^2` and phase `arg(psi)` with core masks; optionally velocity/current fields built from QTT derivative contractions.
- Signed vortex count, vortex-pair separations, formation/annihilation rate, vortex-position time traces, and density-minimum diagnostics.
- Particle number, kinetic/trap/interaction/grand-canonical energy, **correct dissipative energy balance** after the paddle is off, and computed external work while it is moving.
- Peak and profile `chi_i(t)`, discarded weight, DD holdout/missing-action estimate, accepted `h(t)`, GMRES/Arnoldi counts, accumulated runtimes/peak bytes, and whether rank saturates or decreases.
- At selected checkpoints rerun a strict subset with `h/2`, tighter rank tolerance, higher spatial resolution (where tractable), and at least one independent spatial-method reference for the early physical interval.

### Heston: correlated diffusion and eventual smoothing

For the **backward** run save price surfaces, delta, gamma, vega, volatility skew/smile at finite informative maturities, errors against exact affine/mature option reference, and discounted norm scales. For a **forward** `T=500` run save:

- Marginal log-price and variance distributions, mean/variance/covariance, selected integrated probability tails, and affine-transform moments with independent reference.
- Joint density heatmaps at several times, positivity violations and mass/flux-balance residual; use domains wide enough to avoid artificial boundary reflection.
- Peak/per-bond QTT rank vs time and operator/stage work, highlighting rank reduction from diffusion as well as early rank growth.
- An explicit note that `T=500` is primarily a numerical **long-time/stiffness demonstration** for Heston, rather than an economically standard option maturity; meaningful financial comparisons remain at shorter maturities.

### Shared efficiency and stability figures

For both models show `p=2/3/4` BUG curves; conventional TDVP2 where it can complete with correct accuracy, and stabilized TDVP or split-step reference where relevant:

1. **Tolerance/time-to-accuracy frontiers:** error vs seconds and error vs **accepted timestep** (not just requested timestep).
2. **Mechanism:** per-step backward-stage norm gains of TDVP; full-step positivity/norm/Lyapunov and observable error for both methods.
3. **Long-time quality:** physical fields/observables at `T<=500`, with convergence bands from rank/step refinements and documented failure time of any method that does not finish.
4. **Resource scalability:** `L=20,24,28,30` CPU time, peak memory, MPO/rank, rejected steps; final and temporary bond ranks, not cherry-picked final ranks.

## 4.4 Reproducible jobs, smoke checks and release criteria

Every job manifest must record `git commit`, actual source methods, model and PDE operator hash, bit-order map, domain and boundary conditions, MPO/TT approximations, stage family/order, nonlinear solver policy, time-step controller, initial state hash, RNG seeds for DD probes, tolerances, rank cap, tolerances actually met, CPU/BLAS/threads and host class.

**Minimum per-step ledger:** `t,h,p,accepted,reason`, state norm or Heston weighted mass, GP grand-canonical energy and external work, maximum backward TDVP center-core gain if applicable, total `chi_max`, temporary rank peak, DD sampled/holdout columns, per-side true/estimated DD residual (where available), Krylov/GMRES iterations, nonlinear stage residual and iteration count, root residual, discarded weights, environment recomputes, wall time, allocated/peak bytes.

**Before committing large cluster resources:**

- Run 10–20 representative accepted steps at `L=30` with exact same MPO, solver, stage and nonlinear settings as production; confirm checkpoint/restart bitwise or within deterministic tolerance and measure peak memory.
- Establish a validated window where `h=0.1` passes **independent** state/observable error and contractivity/positivity requirements. If not, do not launch a `h=0.1` result; revise schedule or solver.
- Require an **attained order** certificate for each released p2/3/4 method, including nonlinear GPE; p4 remains gated if its predictor recurrence is not proven by numerical tests.
- Require nontrivial comparison with *a stabilized* TDVP/control and independent physical reference, plus a stated accuracy target. No claim of universal superiority or inevitable TDVP2 blow-up.

**Deliverables:** a short source-audit report **including the limits of Connor Appendix A.2**; `qtt_cbe_nosym_first_impressions.md` and `qtt_cbe_nosym_ablation.csv` with immediate go/no-go; a 2D QTT operator and nonlinear test suite; three exact-reference suites (`GP linear`, `GP nonlinear plane wave`, `Heston affine`); p2/3/4 order/stability plots; imaginary-time GP uniform-vs-log schedule report; vortex-shedding and Heston medium reference cases; and two `L=30, T=500` production datasets with saved TT snapshots, physics figures, rank/work ledgers, and qualified BUG-vs-TDVP speedups.

---

# Lean execution checklist (exactly four top-level tasks)

- [ ] **Task 1 complete:** grouped 2D nonsymmetric QTT + matrix-free environments, correct GP/Heston MPOs and adjoints, **early first-impressions ablation (U08) showing whether randomized DD-CBE pays**, hybrid/deterministic selection policy, verified *full nonlinear* local action, old image-recycling off, U01–U08 pass.
- [ ] **Task 2 complete:** GP `g=0` exact 2D oscillator and **`g>0` full nonlinear** exact plane wave plus a structured nonlinear test, Heston 2D affine exact; verified p2/3/4 full-method convergence with **stage-refreshed nonlinear density**, backward-stage diagnostics and **both lagged and consistent nonlinear TDVP2**, V01–V08 pass.
- [ ] **Task 3 complete:** GP ground-state uniform/geometric imaginary-time comparison including QTT GD baseline; source-matched 2D vortex shedding with rank and vortex observables; Heston correlated diffusion pricing/forward density with independent references; useful time-to-accuracy curves.
- [ ] **Task 4 complete:** staged production gate then `L=30, T=500, h<=0.1`, paired checks, physical observables, late-time snapshots, rank/error/time/memory plots, reproducible job reports; describe *observed* limitations rather than forcing a predetermined speedup claim.

## Scientific references

1. R. Connor, C. Duncan, A. Daley, *Tensor network methods for the Gross-Pitaevskii equation on fine grids*, arXiv:2507.01149v1 — **Appendix A.2** (TDVP MPO and QFT Laplacian), **Appendix B** (moving paddle), **Section III.2 / Fig. 26** (nonlinear dissipative vortex comparison); https://arxiv.org/html/2507.01149v1
2. Q.-C. Chen, I.-K. Liu, J.-W. Li, C.-M. Chung, *Solving the Gross-Pitaevskii Equation with Quantic Tensor Trains: Ground States and Nonlinear Dynamics*, arXiv:2507.04279v5, SciPost Phys. 21, 035 (2026) — **Sec. 3.1** (density-QTT reconstruction/compression per step), **Sec. 4.3** (nonlinear dynamics), https://arxiv.org/html/2507.04279v5
3. G. Ceruti, C. Lubich, *An unconventional robust integrator for dynamical low-rank approximation* — https://doi.org/10.1007/s10543-021-00873-0
4. S. Zhang, J. Hu, *On the stability of the low-rank projector-splitting integrator for hyperbolic and parabolic equations* — https://arxiv.org/abs/2507.15192
5. Heston exact/affine structure and semi-analytic option valuation — https://mathworld.wolfram.com/HestonModel.html and https://xilinx.github.io/Vitis_Libraries/quantitative_finance/2019.2/methods/cf-ht.html
6. 2-stage Radau IIA (order 3) and 3-stage Radau IIA (order 5) tableaux — https://juliagni.github.io/RungeKutta.jl/dev/radau2/
7. Uploaded internal methods evidence: `xx_efficiency_summary.md`, `validation_results.md`, `xx_domain_wall_cbe_bug_efficiency_standalone_plan(1).md`, `dissipative_cbe_bug_implementation_plan(1).md`.
