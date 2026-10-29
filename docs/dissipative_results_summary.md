# Dissipative CBE-BUG: what was implemented, results, settings

Plan: `dissipative_cbe_bug_implementation_plan.md`, Tasks 1-7.
Branch history: `plan/dissip-task-1` … `plan/dissip-task-6`, merged into `main`.
The full audit, with file-level evidence and every job ID, is `docs/dissipative_plan_audit.md`. This file is the short version: what exists, what the runs showed, and with which settings.

Status on 2026-10-29:
- Tasks 1-6 are done.
- Task 7 is done except one A2 L = 32 run, which is still running (dr6v5, with resume job dr6v6).
- The planned rerun of the Task 7 BUG arms at tuned settings was not made: the knob study found no setting that fixes A1 (section 5.5).

## 1. What was implemented

| Task | Implemented | Gate |
|---|---|---|
| 1 | No-jump generator `G = -iH - Q/2` as the MPO `H - (i/2)Q` (`no_jump_mpo`), run at `tau = -i dt`. Non-Hermitian Arnoldi for the root `expv`, the local one-site Krylov and two-site TDVP. Radau IIA(2) + GMRES root (`:radau2_gmres`). Order-3 dispatch (half-sweeps, then one enriching frontier sweep per frontier j = 1, 2). Loss-aware Krylov weight. Folded sketch for the frontier leaks | G0: D1.1-D1.8, 51/51 |
| 2 | Pair-loss `Q` (`doublon_loss_mpo`). Collective-emission `Q` (`collective_emission_mpo`, virtual dim 6 at any L). d = 4 Hubbard under U(1). Byte counters. Unit tests U01-U18. `global_krylov_states` (keeps Gψ, G²ψ, G³ψ on the product-state step of nonlocal-Q models) | G0 184/184; G1/G2 67/67 |
| 3 | A1 and A2 fixtures with exact Lindblad references. `closure_checks = true` for every nonlocal-Q run (two-site windows cannot see long-range S⁺S⁻ hops from a U(1) product state) | A1 276/276, A2 138/138 |
| 4 | Quantum-jump engine (`jump_trajectory`, `jump_ensemble`): log survival, Illinois event location, channel draw ∝ ‖L_μψ‖², seeds stored. B1 junction model and current MPOs. B2 collective channels | t4 54/54, t4_b 452/452, t4_conv 8/8 |
| 5 | Driver `benchmarks/dissipative_run.jl` (JSON per run). Warm start `DR_WARM`. Step-doubling controller (eq. 7) with a backward-stable MPS distance; the first step of a segment is always checked. Cap marking in the figures (plan §4.4) | t5 24/24; G2 passed (USER 10-25) |
| 6 | Per-step profiling (phase timers, `D_stage`/`D_work`/`D_final`, sectors, RSS, closure sweeps, holdout, Krylov defect). Checkpoint / restart (refuses changed solver or tolerances; a resumed run equals the uninterrupted one bitwise). Fix: `_dd_select` admitted in-frame roundoff (A2 crash) | t6 R1-R3; restart checks 15/15 |
| 7 | Fix: the frontier embedding squared the frame defect (`_polar_frame`, `3fa0e78`). Fix: the unbounded `fusion_basis` memo (B1 OOMs; element budget, `3a12dca`). Parallel K/L/S and concurrent half-sweeps (U18). `tol_krylov` knob (`ef6c30d`). Work ceiling `d·maxdim` (`6a8b742`). Rank probe `benchmarks/dissipative_rank_probe.jl`. Recycled local Krylov vectors across the rounds of a step (final form `3a07c7c`; since the merge with `main`, `recycle_mode = :complement` in `krylov_recycle.jl`) | U18 35/35; rk7y 1279/1279 |

Re-audit of §1.2 / §1.3 on `3a07c7c`: every item is IMPLEMENTED except plan l.128, "step rejection / retry". Reason codes exist, but a `:work_limited` step is accepted and logged; only step doubling rejects. A smaller-dt retry would not rescue A1: at dt 0.025 the frame still fills the ceiling.

## 2. Models

| Model | Hamiltonian | Loss | d |
|---|---|---|---|
| A1 | Hubbard `H = -J Σ (c†c + h.c.) + U Σ n↑n↓` | `L_j = √κ c_j↓ c_j↑` on every site | 4 |
| B1 | Hubbard junction (bond `J_link`, bias) | `L = √κ c_j0↓ c_j0↑` at the junction j0 only | 4 |
| A2 | XXZ, `Jz` | collective `L± = √Γ± Σ_j e^{±ikx_j} S_j^-`, ka = π/3 | 2 |
| B2 | XXZ, finite density ν (subradiance) | collective, as A2 | 2 |

## 3. Settings shared by all runs

Unless a table says otherwise, every run used these settings.

**Solver:**
- BUG order 3;
- `tol_root` 1e-12, `tol_basis` 1e-8;
- `tol_krylov` = none (falls back to `tol_basis`);
- `cap_order = :sector`, `fold_omega` on;
- one thread, BLAS on one thread.

**Nonlocal-Q models (A2/B2):**
- `closure_checks = true`;
- the exact Gψ…G³ψ are kept on product-state steps (`DR_KEEP_KRYLOV = 3`).

**Integrators:**
- TDVP2 is standard two-site TDVP on the same MPO with `hermitian = false`;
- both integrators truncate with the same per-cut rule (`_cut_divisor`).

**Start:**
- `warm 0.2` means the run starts from the state evolved to t = 0.2 by tight BUG steps;
- every arm of a point shares that state.

**Work ceiling:**
- `2·maxdim` up to `6a8b742`;
- `d·maxdim` after it (dq7e-i, rank probes, the recycling work).

**Recycling:**
- every production and study run in sections 4-5 ran **without** recycled Krylov vectors: the Task 7 and knob-study commits (`0a4fc8f`, `86d7e9c`, `fb7b3a7`, `6a8b742`) predate the recycling code;
- recycling was measured only in tests R5/R6 and the rank probe rp7x (section 6).

**Cluster:** `x86-64-v3&intel` nodes. Array walls are indicative only; timing comes from single sequential timing jobs (dr6a, dp7a/b/c).

## 4. Results by task

### Task 5 (G2): order, ladders, controller

| Run | Settings | Result |
|---|---|---|
| dr5d (46 runs, `68a0db8`) | A1 L = 12, dbl_rich, U = 4, κ ∈ {1, 8, 20}, maxdim 1024 (not binding), T = 0.5. Reference: BUG dt 0.003125 at 1e-11 / 1e-13 / 1e-13 | One-step slopes: BUG 3.95-4.08, TDVP2 3.0 (κ = 1). Ladders monotone in their own tolerance: tol_basis 1e-8/6/4 → 1.3e-10 / 2.1e-9 / 1.7e-7; tol_root → 8e-14 … 2.9e-10; tol_trunc → 1e-11 … 1.8e-7. No component lowers the order. Cost about equal at matched error |
| dr5e (24 runs, `bf4cf5f`) | warm start 0.2, A1 L = 12 and A2 L = 20 block, maxdim 1024 | After the first step no time error accumulates (5-9e-13 for dt 0.1 … 0.0125). Controller with τ = 1e-5 / 7 / 9 gives final errors 5.6e-9 / 1.0e-10 / 6.4e-12 (fixed dt 0.1: 1.4e-5) |

The work ceiling of 2·maxdim binds once D_final ≈ 0.65 maxdim (G3 constraint, found here).

### Task 6 (G3): pilots and profiling

| Run | Settings | Result |
|---|---|---|
| dr6b (references) | A1 L = 12, U = 4, κ = 2, T = 1, maxdim 3072, tol_basis 1e-10 | tol_trunc ladder at dt 0.05 monotone: 2.0e-10 / 1.6e-8 / 3.6e-6 for 1e-9 / 7 / 5 |
| dr6a (timing job) | A1 L = 12, κ = 2, product start, Jt = 1, maxdim 2048, tol_trunc 1e-9 | Neither capped. BUG dt 0.05 / 0.1 / 0.2: 2.7e-6 / 3.3e-5 / 6.7e-4 at 332 / 350 / 433 s. TDVP2 about 10% cheaper at matched error |
| dr6c (binding cap) | A1 L = 12 κ = 2 and L = 16 κ = 8, maxdim 512, tol_trunc 1e-9 | BUG is work-limited from t = 0.4-0.45 (ceiling 2·maxdim binds at D_final 0.38 maxdim); TDVP2 from t = 0.75-0.95. After that BUG is less accurate |
| dr6s (stability) | A1 L = 12, κ = 20, warm 0.2, Jt = 4, maxdim 1024, tol_trunc 1e-9. Reference: BUG dt 0.025 at 1e-10 / 1e-12 / 1e-11 | No instability in either integrator. BUG reaches 1e-8 at dt 0.2, TDVP2 at dt 0.025 (about 3.5× the wall) |
| dr6ta (A2 time to accuracy) | A2 L = 24, 6 exc., block, warm 0.2, Γ = 5, maxdim 512, tol 1e-8 / 1e-12 / 1e-9 | BUG 1.9-2.3× cheaper than TDVP2 at matched error |
| dr6j / dr6k (pilots) | B1 L = 12, U ∈ {±4} × κ ∈ {0.5, 2, 10}, maxdim 256; B2 L = 24, ν ∈ {1/8, 1/4} × Γ ∈ {0.5, 2}; 20 trajectories each | F08 / F11 pilots: a 5% half-width needs about 700-3800 trajectories |

### Task 7 (G4): production points

All rows: tol_trunc ladder {1e-5, 1e-7, 1e-9} plus a dt-halved arm at 1e-7.

| Point | BUG settings | TDVP2 settings | Result |
|---|---|---|---|
| **A2 L = 48** (dr7b, `0a4fc8f`) | 2 exc., block, warm 0.2, Jz = 1, Γ = 1, maxdim 256, dt 0.1, Jt = 20 | maxdim 256, dt 0.05 | Nothing capped (D_final ≤ 26). BUG's ladder is converged (1e-7 vs 1e-9: 7.6e-9) and its dt-halving difference is 4.8e-7. TDVP2's dt-halving difference is 4.1e-4: **BUG about 1000× more accurate** at a wall ratio of 1.07-1.54 |
| **A1 L = 24** (dr7a/c/m/n, `0a4fc8f`) | U = 4, κ = 2, dbl_rich, warm 0.2, Jt = 2, dt 0.1, maxdim 512 and 1024 | dt 0.05, maxdim 512 and 1024 | **BUG fails.** It is work-limited from the first steps (20/20 at maxdim 512, 18/20 at 1024), and its observables are off by 0.3-0.6. Before the cap it agrees with TDVP2 to 4e-10. TDVP2 is converged (512 vs 1024: 1.3e-4) and is the result |
| A1 L = 24, tol_basis 1e-5 (dr7a5, dr7m5, `86d7e9c`) | as above, tol_basis 1e-5 | — | Still fails (0.3-0.6) |
| **B2 L = 32**, no-jump (dr7w rows 1-8, `86d7e9c`) | 4 exc. (ν = 1/8), alt start, warm 0.2, Jz = 1, Γ = 2, maxdim 512, dt 0.1, Jt = 10 | maxdim 512, dt 0.05 | Nothing capped (D_final 154, D_stage 540-890). BUG's dt-halving difference is 9.3e-7, TDVP2's 1.5e-4. BUG wall 6.3-9.1× TDVP2's, 81-89% of it in the closure checks. At matched error about 1e-6, TDVP2 would need dt ≈ 0.0035 (projected 33000 s vs BUG 21858 s) |
| **B2 L = 32**, 20 trajectories (dr7w rows 9-10, row 9 resumed as dr7wr9) | as above, alt product start (no warm), tol_trunc 1e-7, dt 0.1, T = 10 | dt 0.05 | **TDVP2 is wrong from the product state.** Exact P0(0.1) = 0.7495, BUG 0.7488, TDVP2 0.408. TDVP2 jumps early on 7/20 seeds; mean jumps 0.35 (BUG) vs 0.55 (TDVP2). BUG gives the correct jump statistics |
| **B1 L = 20**, 20 trajectories (dr7xb, dr7y, `86d7e9c`) | U = −4, κ = 2, J_link = 1, biased start, maxdim 512, **tol_basis 1e-5** (probe dr7z), dt 0.05, T = 6 | maxdim 256 (and 512, 5 traj, T = 3), dt 0.025 | Before BUG's maxdim cap (t ≈ 0.9-2.15) the two agree to TDVP2's time error (3e-5). After it BUG drifts (0.14-0.18 at t = 6). TDVP2 at its cap stays converged and is the B1 result: late j_c left 0.183 ± 0.019, right −0.214 ± 0.021 |
| A2 L = 32, finite density (dr6v5, `a0c20f4`) | 8 exc., block, Γ = 1, maxdim 1024, dt 0.1, T = 1, tol_basis 1e-10, tol_trunc 1e-12, crash-state config | — | **Running** (1 d 21 h on 10-29, 14.5 G). Resume job dr6v6 is queued behind it |

### Task 7: parallel variants and memory

| Run | Settings | Result |
|---|---|---|
| dp7a (`048eca6`) | A2 L = 48, 2 exc., maxdim 256, tol_trunc 1e-7, Jt = 20; serial / half_sweeps / full × dt {0.2, 0.1, 0.05} | D ≤ 26: parallel is 1-20% slower and burns 2.7-3.3× the CPU |
| dp7b (`c75b983`) | B2 L = 32, 4 exc., maxdim 512, tol_trunc 1e-7, Jt = 2, dt 0.1 | D_stage ≈ 550: half_sweeps 1.52×, full 1.56× per step. No accuracy cost (≤ 1.7e-10 vs serial) |
| dp7c (timing job) | A1 L = 12, maxdim 64 … 1024 × serial / serial BLAS2 / half / half BLAS2 / full; B2 L = 32, tol_trunc 1e-5 / 7 / 9 | half_sweeps gains 1.1-1.8× only from D_stage ≈ 600-1000; nothing below. `full` never pays |
| dm7a (`3a12dca`) | B1 row-1 point, 2 trajectories, T = 2.5 | Unbounded memo 12.3 G at step 106; budget 2^28 levels off at 5 G, bitwise equal results, +6% wall |

## 5. Knob study (USER 10-28): `tol_krylov` and `tol_basis`

All rows: no-jump, tol_trunc 1e-7, production dt and maxdim. `tol_krylov` replaces `tol_basis` for the local Krylov share only.

| Run | Point, commit | Grid | Result |
|---|---|---|---|
| dk7a / dk7b (`ef6c30d`) | A2 L = 24, 6 exc., Γ = 5, maxdim 512, Jt = 4, tol 1e-8 / 1e-12 / 1e-9 | tol_krylov {none, 1e-7 … 1e-4} × dt {0.1, 0.05}; reference dt 0.025 | At dt 0.1 the default's error (4.8e-6) is the ceiling's. tol_krylov 1e-6 gives 9.0e-8 at 11% less wall. At dt 0.05 it makes no difference. Default unchanged |
| dq7a (`fb7b3a7`) | A1 L = 24, maxdim 512 | tol_basis {1e-8, 1e-5} × tol_krylov {none, 1e-6 … 1e-2} | Even tol_krylov 1e-2 (n_krylov 45k → 4.3k) stays work-limited 19-20/20, error about 0.5 |
| dq7b (`fb7b3a7`) | B1 L = 20, maxdim 512, dt 0.05, T = 3; TDVP2 references 1024 / 512 at dt 0.0125 / 0.025 | tol_basis {1e-8, 1e-5} × tol_krylov | At ceiling 2·maxdim: error 3-6e-3, work-limited 37-49/60 |
| dq7c (`fb7b3a7`) | A2 L = 48, maxdim 256 | tol_krylov × 5 | tol_krylov none reproduces dr7b exactly. 1e-5 is best: −25% wall, error 4.8e-7 → 1.2e-6 |
| dq7d (`fb7b3a7`) | B2 L = 32, maxdim 512, T = 5 | tol_krylov × 5 | Same-node: 1e-3 is 1.8× faster, error 9.6e-7 → 3.5e-6; 1e-4 no gain |
| dq7e / dq7f (`6a8b742`) | A2 L = 48 / B2 L = 32 | tol_basis {1e-4 … 1e-7} × tol_krylov {none, 1e-5, 1e-4} | tol_basis does not set D_final (A2 26, B2 154 at every value; tol_trunc sets it). It sets D_stage and Krylov work: B2 at 1e-4 has D_stage 392 / n_krylov 125k vs 664 / 260k, same error. A2 at 1e-4 costs more closure sweeps |
| dq7g (`6a8b742`, ceiling d·maxdim) | B1 L = 20, maxdim 512 | tol_basis {1e-4 … 1e-8} × tol_krylov {none, 1e-5, 1e-4, 1e-3} | **B1 fixed by the d·maxdim ceiling:** all 19 rows accurate to 7.1e-6 (the reference floor), still work-limited on 11-40/60 steps. Cheapest: tol_krylov 1e-3 (n_krylov 22k vs 277k) |
| dq7h (`6a8b742`) | A1 L = 24, maxdim 512 (ceiling 2048) | same 19-row grid | **All 19 fail**, observables off by 0.24-0.28. No tolerance lifts it |
| dq7i (`6a8b742`) | A1 L = 24, maxdim 1024 (ceiling 4096) | defaults | Still fails: obs error 0.21, log P0 1.29 vs TDVP2 maxdim 1024; work-limited 17/20; 35451 s, 16.8 G |
| dq7j (`6a8b742`) | A1 L = 24, maxdim 512 (ceiling 2048) | tol_basis 1e-8 × tol_krylov {1e-2, 3e-2, 1e-1, 1}: Krylov allowance per test 6.9e-5 … 6.9e-3 (vs 6.9e-11) | **All 4 fail.** Obs error 0.27 / 0.26 / 0.26 / 0.19, log P0 2.2 / 2.4 / 2.1 / 0.85; work-limited 16 / 16 / 16 / 15 of 20; n_krylov 28k → 14k (dq7h at 1e-5: 52k). Details in §7.5 |

### Why A1 blows up (rank probes)

These probes start from BUG's own clean state at t = 0.5 (A1 L = 24, maxdim 512).
- **rp7q:** frontier 1 adds 25573 directions (22k local Krylov against 3.5k DD-QR CBE), frontier 2 adds 13220.
  - At ceiling 8192 the step is clean (D 214-299 against TDVP2's 201-234), but it costs 6098 s against TDVP2's 74 s.
- **rp7r:** order 1 runs away as well. With no closure rounds, the half-sweeps alone add 1000-1300 Krylov and 520-710 CBE directions per bond.

Section 7 explains where these directions come from and why the bond runs away. The blow-up is specific to the d = 4 Hubbard generator with loss on every site. B1 (loss on one site) saturates but stays accurate. The d = 2 spin chains stay accurate.

## 6. Recycled local Krylov vectors (plan §1.2)

**What it is.** Within one order-3 step:
- each round (half-sweeps, frontier 1, frontier 2) keeps, per bond, its local Krylov input, its active frontier and up to 8 of its heaviest vectors;
- the next round transports them into its own gauge and reuses them.

**Final form (`3a07c7c`):**
- The next round splits its input as u0 = αp + v_c, where p is the previous input.
- The recycled block is reused only if it certifies p's evolution under *this* round's generator: its defect must be ≤ half the Krylov allowance.
- If it passes, Arnoldi extends only the complement v_c. Otherwise the fresh solve runs.
- Nothing survives the step.

The option is `krylov_recycle` with `recycle_mode = :complement`, default off. In the merge with `main` it shares one knob with the XX-efficiency study's `:images` / `:frame` modes. `DR_RECYCLE` is the driver knob.

**Measurements across the three versions:**

| Version | Test | Closure H1, recycle on / off | Directions added / frame width | Reuse |
|---|---|---|---|---|
| `bc25565` (block-defect continuation) | R5, A1 L = 6 | 576 / 112 (5×) | equal (χ 56 / 56) | the continuation was a block recurrence; it hit the depth bound in the matrix test |
| `c2b2478` (augmented Arnoldi) | R5, A1 L = 6 | 190 / 112 | 105 / 105, frame 64 / 64 | — |
| `c2b2478` | R6 matrix test, generator moved 1e-5 / 1e-3 | — | **10 / 12 directions against 7 fresh: wider** | never fewer than fresh |
| `c2b2478` | rp7x, A1 L = 12, t0 = 0.1, maxdim 512 | wall 42.3 s / 19.1 s (order 2: 38.4 / 12.2) | 1183 / 1183 added, frame 994 / 994 | — |
| `3a07c7c` (complement, final) | R5, A1 L = 6 | 216 / 112 (2×) | 105 / 105, frame 64 / 64 | **0 of 104 bonds** |
| `3a07c7c` | R6 matrix test | more H1 than fresh in every case | equal to fresh | only when the generator is unchanged; a relative change ≥ 1e-6 rejects it |

**Observations:**
- **The generator moves between rounds.** The frontier sweeps advance the frames, so the one-site generator a bond sees in frontier 2 is not the one it saw in frontier 1. Once the certification test is applied, the kept vectors almost never pass it. Without the test, the reused complement errs by 10× the allowance at a 1e-6 generator change and by 10⁴× at 1e-3.
- **Recycling never narrowed the frame.** No version added fewer directions than the fresh solve. The augmented form (`c2b2478`) added more (10-12 against 7) when the generator moved. The final form adds exactly the fresh directions.
- **It always costs more.** Each recycled vector needs its image under the new generator: 1.7-5× the closure H1 applications, and 2.2-3× the wall at L = 12.
- **It cannot reach the A1 blow-up.** The runaway already happens at order 1 (rp7r), where there are no closure rounds to recycle into. The width comes from the local Krylov enrichment itself: each Krylov vector's out-of-frame range is appended.

**Conclusion:**
- Recycled local Krylov vectors are not efficient here. They do not reduce bond growth, and the variant that reused them most widened the frame. The final variant leaves the width unchanged for 2× the operator applications.
- What decides the frame width is the local Krylov enrichment and the work ceiling. Recycling does not change either.
- Recycling stays implemented as the plan specifies, but it brings no gain on these models.

## 7. Why the bond dimension grows

Two different growths are involved, with different causes:
- **the working frame** that BUG builds inside a step is far wider than the state it produces;
- **the final bond** (after truncation) runs away to maxdim. That happens only once the frame is cut short by the ceiling, or when `tol_basis` is much looser than `tol_trunc`.

The numbers below come from the per-bond logs of the rank probes (rp7q, rp7r) and the A1 grid (dq7h). The two causal links marked *inferred* follow from these numbers but have not been measured directly.

### 7.1 Size of the gap

Every row is one step on A1 L = 24, from BUG's own clean state at t0 = 0.5 (rp7q):

| Arm | Frame at the middle bond | Final bonds | Error (dist) |
|---|---|---|---|
| TDVP2, maxdim 512 | — | 201-234 | 1.9e-7 |
| BUG, half-sweeps only | 2299 = ~196 old + 561 DD-QR CBE + 1547 local Krylov | — | — |
| BUG, ceiling 8192 (both closure rounds) | 8192 (round 1 adds 25 573 directions in total, round 2 adds 13 220) | 214-299 | 7.0e-8 |
| BUG, ceiling 2048 | 2048, filled by the half-sweeps alone | 241-**512** | **7.4e-5** |

The frame is 10-40× wider than the state it produces.

### 7.2 Why the working frame is so wide

1. **Each Krylov vector brings its whole range, not one direction.**
   - The local Krylov stage (Tests C/D) hands each Krylov vector to `_extend_frame` (`bug/krylov_enrich.jl`). That function takes the vector's part outside the frame, `R = (I - QQ') v_k`, and appends singular directions of `R` until `weight × (discarded tail) ≤ allowance`.
   - The allowance per test is `tol_basis / (2(nb+1)·nf)` (`basis_allocation`, rule `:linear`). At L = 24, order 3 and `tol_basis` 1e-8 that is **6.9e-11**.
   - So the leading Krylov vectors, whose weights are 0.1-1, are admitted almost to their numerical rank.
   - Each application of the one-site generator spreads the range further, so successive vectors bring new directions.
   - At the middle bond this adds 950-1550 directions, about 6× the state's own rank.
2. **The closure rounds compound it.**
   - Each closure round writes its frontier as an MPS in the current frames (`_frontier_mps`, `bug/krylov_closure.jl`). The frontier's bonds are therefore as wide as the frames (2000+), not as wide as the state (≈ 200).
   - That round's Krylov vectors have rank ≈ 2000, and admitting their ranges adds tens of thousands of directions.
   - Wider frames make a wider frontier, which makes the next round's frames wider still. This is why order 3 needs about 4× what the half-sweeps alone need (rp7q: 2299 → 8192).
3. **It also compounds along the sweep.** Bond i's frame is chosen inside (bond i-1's frame) × d. With d = 4 the frame widens bond by bond (t0 = 0.3: 256 → 810 → 1141 → … → 1366).
4. **Why A1 and not the spin chains:**
   - The MPO width is not the difference: the generator `G` has bond dimension 8 for A1/B1 and 11 for A2/B2.
   - What differs is the local dimension (d = 4 against 2) and the size of the state (D ≈ 200 for A1 against 26 for A2). B2 (d = 2, D_final 154) stays at D_stage 640.
   - B1 (d = 4) also fills the ceiling, on 11-40 of 60 steps. A1 is where filling it costs accuracy.
5. **Recycling cannot help here.** Order 1 runs away too (rp7r), and order 1 has no closure rounds to recycle into (section 6).

### 7.3 Why the final bond runs away

1. **The ceiling cuts by order, not by importance.** *(inferred)*
   - Room in the frame is handed out first come, first served: DD-QR CBE first, then the Krylov vectors in Arnoldi order, then closure round 1, then round 2.
   - At t0 = 0.5 and ceiling 2048, the half-sweeps alone fill the frame. Round 1 then gets 387 directions and **round 2 gets 0**, although its leak (6.5e-8) is still there.
   - The step becomes a projection onto an incomplete space: its error is 7.4e-5, against 7.0e-8 with the frame uncapped.
2. **The truncation cannot remove an error that large.**
   - The final cut may discard only about `tol_trunc` = 1e-7 of weight. An error 700× larger, spread over many directions, therefore stays in the state as extra rank: the bonds jump to 512.
   - The next step starts from 512-wide bonds, so its frames fill the ceiling sooner. The loop feeds itself.
   - dq7h, `tol_basis` 1e-5: D_final is 138, 206, 297, then **466 at the first work-limited step**, then 512 on every later step. TDVP2 at the same times stays near 140-220.
3. **The same happens without any cap when `tol_basis` ≫ `tol_trunc`.**
   - dq7h at `tol_basis` 1e-4 has D_final **298 at step 1**, before any step is capped. At `tol_basis` 1e-5 it is 138, and TDVP2 is ≈ 141.
   - The looser basis leaves a step error above the truncation level, and the truncation keeps it. This is the "Galerkin error of the looser basis kept by the tighter cut" found earlier with rp7h (`docs/dissipative_plan_audit.md`, Task 7 A1).

### 7.4 What would confirm it, and the fixes that follow

**Confirming probe (not run).** One A1 L = 24 step from the t0 = 0.5 state, on one cluster job of about 2 h. For every appended direction it would record:
- its source: DD-QR CBE, Krylov depth k, or closure round j;
- its weight in the final state.

That measures directly the two links marked *inferred* above: how few of the appended directions matter, and whether the order-based cut drops the ones that do.

**Fixes this points to** (none implemented):
- **Cut by weight across all sources** when the ceiling binds, instead of first come, first served. Today `cap_order = :weight` ranks only the DD-QR CBE directions.
- **Compress each frontier to its own rank** before its closure sweep, instead of sweeping it at frame width. This breaks the round-to-round compounding.
- **Revisit the per-vector allowance rule.** Admitting each Krylov vector's full residual range at 6.9e-11 spends the whole basis budget on directions that the final cut mostly discards. *Tested in dq7j (§7.5): loosening it alone does not fix A1.*

### 7.5 Loosening the per-test Krylov allowance at the d·maxdim ceiling (dq7j)

dq7j raises the Krylov allowance per test from 6.9e-11 to 6.9e-5 … 6.9e-3 (`tol_krylov` 1e-2 … 1, `tol_basis` 1e-8), at A1 L = 24, maxdim 512, ceiling 2048, same code and snapshot as dq7h. Reference: TDVP2 maxdim 1024 (dr7n).

| tol_krylov | Allowance | Obs err | log P0 err | D_final, steps 1-5 | D_stage, steps 1-5 | First step at 512 / work-limited | Krylov dirs, step 1 / total | DD-QR CBE dirs, step 1 / total | Wall |
|---|---|---|---|---|---|---|---|---|---|
| dq7h 1e-5 | 6.9e-8 | 0.26 | 2.45 | 176 250 340 512 512 | 1134 1250 1726 2048 2048 | 4 / 4 | 2502 / 51.7k | 8218 / 382k | 5884 s |
| dq7h 1e-3 | 6.9e-6 | 0.27 | 2.40 | 322 488 512 512 512 | 1042 1270 1771 2048 2048 | 3 / 4 | 595 / 32.9k | 8974 / 386k | 6224 s |
| 1e-2 | 6.9e-5 | 0.27 | 2.17 | 292 402 512 512 512 | 1005 1112 1356 1930 2048 | 3 / 5 | 263 / 28.1k | 8918 / 379k | 7701 s |
| 3e-2 | 2.1e-4 | 0.26 | 2.37 | 273 341 487 512 512 | 988 1052 1187 1663 2048 | 4 / 5 | 157 / 24.3k | 8885 / 376k | 7721 s |
| 1e-1 | 6.9e-4 | 0.26 | 2.12 | 269 301 428 512 512 | 983 1033 1120 1541 2048 | 4 / 5 | 100 / 21.7k | 8965 / 370k | 4894 s |
| 1 | 6.9e-3 | 0.19 | 0.85 | 270 270 316 512 512 | 1000 1019 1027 1429 1997 | 4 / 6 | 44 / 14.3k | 8994 / 357k | 4816 s |

The allowance per test is `tol_krylov/(2(nb+1)nf)`; with `tol_krylov` unset it is the `tol_basis` share, 6.9e-11. Walls are from different nodes (rows 3-4 on h05n4, rows 1-2 on h02n2) and are not a timing comparison. TDVP2 maxdim 512 vs 1024 differ by 1.3e-4; D_final of TDVP2 is ≈ 141 at step 1.

**Observations:**
1. **The local Krylov stage is not what fills the ceiling at this point.** At `tol_krylov` 1 it adds 44 directions at step 1 (against 2502 at 1e-5), yet D_stage at step 1 is still 1000 (against 1134). Over every step the DD-QR CBE appends 7-25× more directions than the Krylov stage (≈ 19k per step in every row). The frame widening of §7.2 item 1 is the Krylov share in rp7q's uncapped half-sweeps; at the 2048 ceiling the DD-QR CBE and the round-to-round compounding (§7.2 items 2-3) fill the frame on their own.
2. **A looser Krylov allowance inflates D_final from step 1, before any cap.** D_final at step 1 is 269-292 against 176 at 1e-5 and TDVP2's ≈ 141, with discarded weight still ≈ 7e-8. This is §7.3 item 3 again: the looser basis leaves a Galerkin error above `tol_trunc`, and the truncation keeps it as rank. The bonds reach 512 at step 3-4, as before.
3. **Every row then becomes work-limited (from step 5-6) and stays so.** Loosening delays the first capped step by 1-2 steps; it does not stop the runaway.
4. **`tol_krylov` 1 is the least bad row** (obs 0.19, log P0 0.85 against 0.26 / 2.2-2.4). Fewer Krylov directions leave room under the ceiling for the closure rounds (§7.3 item 1, *inferred*). It is still 10³× worse than the TDVP2 512-vs-1024 difference, so it is not a fix.

**Conclusion:** the per-test allowance is not the lever for A1. Loosening it cuts Krylov work up to 3.6× but leaves the frame at the ceiling (filled by the DD-QR CBE and the compounding) and raises D_final through the looser Galerkin error. The fixes that remain are the first two of §7.4: cut by weight across all sources when the ceiling binds, and compress each frontier to its own rank.

## 8. Open items

1. **dr6v5 (A2 L = 32):** still running; resume job dr6v6 (17599988) is queued behind it.
2. **`krylov_recycle` default:** set to `false` in the merge with `main` (USER 10-29: one knob, with this plan's variant as `recycle_mode = :complement`); the knob is kept.
3. **Plan l.128:** step rejection / retry on `:work_limited` is PARTIAL.
4. **A1 needs a ceiling of about 16·maxdim** to run clean (rp7q, at about 80× TDVP2's cost per step). Until then TDVP2 is the A1 result (plan §4.5: a regime where TDVP2 wins is a result).
5. **Task 7 BUG reruns at tuned settings were not made.** The candidates are tol_krylov 1e-5 (A2) and 1e-3 (B2, B1), but none changes a verdict.
