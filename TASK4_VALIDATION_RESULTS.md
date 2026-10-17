# simulation_plan_2 Phase 5: small-cluster validation of the integrators

## Summary

On three 2D models (4x4 square, 4x4 J1-J2 at J2 = 0.5, 18-site kagome) against the EXACT
correlator:

1. **There is a cap below which the BUG is truncation-limited and worse than TDVP.** Each BUG step
   evolves in a 2chi trial space and compresses back to chi. When the cap binds hard, every step
   discards 1e-3 to 6e-3 of the state, the error accumulates, and the BUG is 25-50x worse than TDVP
   (which projects onto the fixed-rank manifold). The ledger flags every such step
   `:budget_unmet` (`:trunc`, `:krylov`), so the failure is never silent.
2. **Above that cap the adaptive BUG matches TDVP's accuracy at 0.46-0.56x its wall time, and
   order 4 at 4x the step is the most accurate arm** (0.29-0.72x the wall time). The crossover cap is
   set by the model's entanglement: 128 for the square and J1-J2, 256 for kagome.
3. **With no binding cap the BUG is limited by its own tolerances**, because every step's
   compression discards up to `tol_trunc`'s share and the losses add per step. At the loose
   `1e-5 / 1e-4` the BUG is therefore worse at SMALLER dt, the same for all schemes. Tight
   tolerances remove this (1.85e-8 at every dt), and fewer, larger steps always help.
4. **The CBE-DMRG ground states are sound**: three starts agree to 1e-10 to 2e-7 on every model and
   cap, and the energies sit above the exact ones by the rank-limited amount.
5. **The order of the schemes is NOT measured here.** At 16 sites cap 256 is full rank and the BUG
   is exact at any dt. Measuring order on 2D needs a larger cylinder (Phase 6).

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

## Order of the schemes on 2D (U22): `benchmarks/spectral/order_ladder_2d.jl`

The setup is a square cylinder started from the Neel product state, at short T = 0.8 and tight
tolerances (`1e-10 / 1e-12 / 1e-13`), with a cap far above the reached rank. It is scored against
the exact state from sparse Krylov in the S^z = 0 sector. The error is `max_j |<S^z_j>(T) - exact|`,
and a t = 0 gate checks that the MPS and the sparse basis use the same site convention.

**4x3 (N = 12) smoke, job 17282573:**

| arm | dt = 0.2 | dt = 0.1 | slope | wall (0.2 / 0.1) |
|---|---|---|---|---|
| tdvp2 | 9.9e-2 | 4.7e-2 | 1.1 | 16 / 42 s |
| bug1 | 1.6e-1 | 8.8e-2 | 0.9 | 2.3 / 5.1 s |
| bugmid | 8.8e-2 | 4.6e-2 | 0.9 | 5.4 / 10.9 s |
| **bug2** | **6.0e-7** | **7.4e-8** | **3.0** | 7.3 / 11.5 s |
| **bug4** | **5.4e-10** | **1.4e-11** | **5.3** | 9.9 / 15.6 s |

- **From a product state, every fixed-frame scheme is first order and badly off (5-16%).** That
  covers TDVP2 and the ordinary and midpoint BUG steps. Their frames cannot represent the second
  Krylov moment of a low-rank state. This is the gap the chain closure study measured on its first
  step (rho_1 = 0.71); here it recurs while the state stays low-rank.
- **The Krylov closure removes it.** Order 2 is ~1e5 times and order 4 ~1e8 times more accurate at
  dt = 0.2. The slopes are p + 1, the local order O(dt^(p+1)) when the early steps dominate, as on
  the chain. Order 4 costs 2-4x order 1 in wall time.
- Two steps reported `:budget_unmet` (one each for bug2 and bug4); their reasons will be read
  from the 6x4 run.

*6x4 (N = 24, sector dim 2.7M), dt 0.2 to 0.025: running (same job).*

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

### Tight tolerances `1e-8 / 1e-8 / 1e-10`, cap 256, T = 4 (job 17282199)

| dt | tdvp2 | bug1 = bugmid = bug2 = bug4 | wall time tdvp2 / bug1 / bug4 |
|---|---|---|---|
| 0.4 | 2.5e-7 | **1.85e-8** | 48.8 / 39.9 / 54.1 s |
| 0.2 | 1.97e-8 | 1.85e-8 | 94.0 / 59.9 / 96.6 s |
| 0.1 | 1.85e-8 | 1.85e-8 | 184.3 / 99.0 / 181.8 s |
| 0.05 | 1.86e-8 | 1.86e-8 | 367.4 / 167.8 / 357.7 s |

Every BUG step discards nothing (`discarded 0.0`), with final rank 256. The common 1.85e-8 floor is
the ground state's own error (1e-8 in energy), shared by every arm.

**This ladder cannot measure the schemes' orders, and the reason is the system size.** At 16 sites,
cap 256 IS the full space at the centre bond (2^8). The BUG trial space then contains everything,
its projected evolution is exact at any dt, and all four schemes agree to the digit. The BUG at
dt = 0.4 is therefore exact while TDVP2 shows its own step error (2.5e-7). As a cost statement this
is real: the BUG at dt = 0.4 matches TDVP2 at dt = 0.2 in 40 s against 94 s. But it is a
small-system artefact, not a general result. Order differences need a cap BELOW full rank and a
truncation tolerance BELOW the step error. The L = 16 XX closure study met both conditions and
measured slopes 2/3/4/5 (KRYLOV_CLOSURE_AUDIT.md IT2); the 2D version needs a cylinder too large
for full rank at the working cap (Phase 6).

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
| 128 (job 17261138) | 2.0e-4 (1.1e-3), spread 8.8e-11 | 3.2e-3 / 825 s | 3.2e-3 / 980 s | 6.3e-2 / 392 s | 6.5e-2 / 763 s | 6.3e-2 / 1400 s | 7.0e-2 / 676 s |
| 256 (job 17261139) | 5.8e-7 (4.0e-6), spread 1.9e-8 | 4.9e-5 / 1287 s | 4.9e-5 / 1634 s | **4.6e-5 / 721 s** | 4.6e-5 / 1417 s | 4.6e-5 / 1226 s | **3.3e-5 / 924 s** |

Kagome is the hardest of the three models at a given rank. At cap 64 even the ground state is
rank-limited (4.2e-3 above exact), and every arm inherits that floor. The BUG arms are
truncation-limited as on the square (every step `:budget_unmet`, about 6e-3 discarded per step).
Two random starts ran to the 60-sweep cap because the driver passed a fixed `etol = 1e-12`; the
driver now leaves the adaptive ground state to `dmrg_etol`.

At cap 128 the kagome cylinder is still truncation-limited for the BUG: every step is unmet
(`:trunc`, `:krylov`) and discards about 1.7e-3, against about 1.2e-4 on the square at the same cap.
The cap at which the BUG overtakes TDVP therefore depends on the model's entanglement growth. On
the square and J1-J2 it is 128; on kagome it is 256. At cap 256 (half the 512 full rank of 18 sites)
the kagome BUG matches TDVP2 (4.6e-5 against 4.9e-5) at 0.56x its wall time, and bug4 beats it
(3.3e-5) at 0.72x. Only `:trunc` remains unmet (about 1.1e-4 per step against 1e-4).
