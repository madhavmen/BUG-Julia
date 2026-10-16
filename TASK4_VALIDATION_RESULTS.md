# simulation_plan_2 Phase 5: small-cluster validation of the integrators

This is the S^zz real-time correlator `G(x,t) = e^{i E0 t} <Om| S^z_x e^{-iHt} S^z_c |Om>` on 2D
cylinders. Each arm is scored against the EXACT correlator (sparse Krylov in the S^z = 0 sector,
same coupling matrix as the MPO): `max_{x,t} |G - G_exact|`. Every arm starts from the SAME adaptive
CBE-DMRG ground state, solved at the evolution cap (`F_DGS = F_D`). The arms are:

- `tdvp2`: 2-site TDVP;
- `cbe1s`: 1-site CBE-TDVP with RSVD;
- `bug1` / `bugmid`: adaptive CBE-BUG, ordinary and midpoint;
- `bug2` / `bug4`: Krylov-closure orders 2 and 4, with `bug4` at a 4x larger step.

The BUG tolerances are `1e-5 / 1e-4 / 1e-4` unless stated, and all arms use 2 threads with
parallel half-sweeps. Driver: `benchmarks/spectral/fig45_2209.jl`.

## 4x4 square Heisenberg cylinder, T = 8, dt = 0.05 (bug4: 0.2)

| cap | ground state dE (variance) | tdvp2 | cbe1s | bug1 | bugmid | bug2 | bug4 |
|---|---|---|---|---|---|---|---|
| 64 (job 17261124) | 1.5e-4 (1.2e-3) | 1.85e-3 / 518 s | 1.85e-3 / 545 s | 5.2e-2 / 219 s | 4.8e-2 / 424 s | 5.2e-2 / 679 s | 4.2e-2 / 305 s |
| 128 (job 17261125) | 1.3e-6 (1.2e-5) | 4.26e-5 / 662 s | 4.26e-5 / 731 s | **4.04e-5 / 311 s** | 4.04e-5 / 583 s | 4.04e-5 / 512 s | **2.44e-5 / 192 s** |
| 256 (job 17261133) | 1.0e-8 (1.3e-7) | 2.4e-8 / 764 s | 2.4e-8 / 793 s | 1.0e-5 / 323 s | 1.0e-5 / 580 s | 1.0e-5 / 532 s | 1.3e-6 / 188 s |

Entries are `max|G - G_exact|` / wall time.

- **At cap 256 (essentially full rank for 16 sites) every step is `:met`.** The BUG is then limited by
  its own tolerances. Each step's compression discards about 1.8e-5, inside `tol_trunc = 1e-4`,
  and over 160 steps that sums to 1.0e-5, while TDVP is near exact. Order 4 at 4x the step does a
  quarter of the compressions and is 8x more accurate (1.3e-6) in 188 s. A matched-accuracy
  comparison at this rank needs tighter BUG tolerances: see the tight dt ladder.

- **At cap 64 the BUG is truncation-limited.** The cap binds hard: every BUG step is
  `:budget_unmet` (`:trunc`, `:krylov`) and discards about 1.4e-3 of the state's norm per step. The
  error accumulates like TEBD's, while TDVP, which projects onto the fixed-rank manifold, stays at
  1.85e-3. The ledger reports the failure on every step.
- **At cap 128 the BUG matches TDVP's accuracy at 0.47x its wall time** (bug1). Order 4 at 4x the
  step is MORE accurate (2.44e-5 against 4.26e-5) at **0.29x** the wall time: a quarter of the
  steps means a quarter of the compressions. The only unmet reason left is `:trunc` (about
  1.2e-4 discarded per step against a 1e-4 budget).

## dt ladder, 4x4 square, cap 256 (not binding), T = 4 (job 17261140)

At the arm tolerances `1e-5 / 1e-4 / 1e-4`:

| dt | steps | tdvp2 | bug1 = bugmid = bug2 = bug4 |
|---|---|---|---|
| 0.2 | 20 | 2.0e-8 | 6.3e-7 |
| 0.1 | 40 | 1.9e-8 | 3.1e-6 |
| 0.05 | 80 | 1.9e-8 | 9.1e-6 |

**The BUG error grows as dt shrinks, identically for all four schemes.** Time discretization is not
what limits it. Each step's single compression discards about 1.8e-5, inside the 1e-4 budget so the
step is `:met`, and these losses accumulate per step. At fixed tolerances, fewer and larger steps
are more accurate. The "`tol_trunc` 10-100x looser than the target" rule from the XX study assumed a
~1e-6 target, and TDVP is at 2e-8 here. A tight-tolerance ladder (`1e-8 / 1e-8 / 1e-10`, dt 0.4 to
0.05) is running to expose the schemes' orders (job 17282199).

## 4x4 J1-J2 square cylinder, J2/J1 = 0.5, T = 8, dt = 0.05 (bug4: 0.2)

| cap | ground state dE (variance), 3 starts | tdvp2 | cbe1s | bug1 | bugmid | bug2 | bug4 |
|---|---|---|---|---|---|---|---|
| 64 (job 17261134) | 4.1e-5 (2.2e-4), spread 2.3e-7 | 1.76e-3 / 558 s | 1.76e-3 / 581 s | 9.1e-2 / 228 s | 7.8e-2 / 442 s | 9.6e-2 / 706 s | 8.0e-2 / 329 s |
| 128 (job 17261135) | 1.7e-6 (1.3e-5), spread 8.2e-10 | 5.5e-5 / 718 s | 5.5e-5 / 799 s | **5.1e-5 / 327 s** | 5.1e-5 / 612 s | 5.1e-5 / 998 s | **3.2e-5 / 388 s** |
| 256 (job 17261136) | 2.0e-8 (2.0e-7), spread 8.7e-10 | 5.3e-8 / 868 s | 5.3e-8 / 874 s | 5.9e-6 / 344 s | 5.9e-6 / 628 s | 6.0e-6 / 556 s | 2.3e-6 / 216 s |

At cap 64 this is the same truncation-limited picture as the square: every BUG step is
`:budget_unmet` (`:trunc`, `:krylov`). The three ground-state starts (Neel plus two random S^z = 0
products) agree to 2.3e-7, so there is no metastability at this rank.

At cap 128 the frustrated model repeats the square's result: bug1 matches TDVP2's accuracy (5.1e-5
against 5.5e-5) at 0.46x its wall time, and bug4 is more accurate (3.2e-5) at 0.54x. Order 2 is the
slowest arm here (998 s): its closure sweeps cost time but buy nothing while the compression, not
the time step, sets the error. At cap 256 every step is `:met` and the BUG is tolerance-limited
(5.9e-6, bug4 2.3e-6) against TDVP's near-exact 5.3e-8, as for the square.

## Kagome cylinder, 2x3 cells (18 sites), T = 8, dt = 0.05 (bug4: 0.2)

| cap | ground state dE (variance), 3 starts | tdvp2 | cbe1s | bug1 | bugmid | bug2 | bug4 |
|---|---|---|---|---|---|---|---|
| 64 (job 17261137) | 4.2e-3 (1.7e-2), spread 4.0e-8 | 1.65e-2 / 581 s | 1.65e-2 / 633 s | 1.24e-1 / 232 s | 1.27e-1 / 416 s | 1.22e-1 / 798 s | 1.14e-1 / 357 s |
| 128 | pending | | | | | | |
| 256 | pending | | | | | | |

Kagome is the hardest of the three models at a given rank. At cap 64 even the ground state is
rank-limited (4.2e-3 above exact), and every arm inherits that floor. The BUG arms are
truncation-limited as on the square (every step `:budget_unmet`, about 6e-3 discarded per step).
Two random starts ran to the 60-sweep cap because the driver passed a fixed `etol = 1e-12`; the
driver now leaves the adaptive ground state to `dmrg_etol`.
