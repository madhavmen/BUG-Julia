# RSVD-CBE for BUG — current work

Branch `feature/cbe-bug`. Read this before touching `src/RSVDCBEBondUpdate/`; it records what
was fixed, what was **ruled out** (so it is not re-investigated), and what is left.

---

## 0. The two paths, and what to call them

| | `cbe_bond_update` (`cbe_bond_update.jl`) | `cbe_lubich_sweep` (`cbe_lubich.jl`) |
|---|---|---|
| `H` at a bond | the **bare two-site gate** | an **effective operator dressed with channel environments**, carrying every term of `H` |
| composition | odd/even Trotter, Strang | none — one step, no splitting |
| Galerkin steps per step | one per bond (`L−1`) | **one**, at the centre |
| basis update | `canonical!` before each bond | K-sweep + L-sweep with carries |
| built from | `xx_bond_gate`, `heisenberg_bond_gate` | `xxz_chain` + `henv.jl` |
| rank-4 tensor | yes, one per matvec inside `apply_s` | **never** |
| drivers | `cbe_bond_update_sweep!`, `cbe_bond_update_bug!` | `cbe_lubich_bug!` |
| status | the **validated** path; the diagnostic vehicle | the **production target** |

**Neither path ever applies a global `H` to the state**, and earlier wording in these docs that
said so was wrong. Every application is an EFFECTIVE LOCAL operator: a two-site effective `H`
for the basis sketch (`sketch_h_left/right`, built from the left channels at link `i`, the right
channels at link `i+2`, and the local terms) and a zero-site effective `H` for the Galerkin step
(`zero_site_h` contracted with the augmented frames). The only defensible sense of "global" is
*which terms the effective operator contains* — all of `H` for `cbe_lubich_sweep`, versus only
the local bond term for `cbe_bond_update`, where the rest of the chain reaches the bond solely
through the canonical gauge. `henv.jl:10` says it outright: **NOT A SYMMETRIC MPO** — it is an
MPO contraction with the virtual leg *named* rather than fused.

Older names, for reading past commits: `cbe_gate_bond_update` → `cbe_bond_update`,
`cbe_gate_sweep!` → `cbe_bond_update_sweep!`, `cbe_gate_bug!` → `cbe_bond_update_bug!`,
`cbe_expand_gate` → `cbe_expand_bond`, `sketch_gate_*` → `sketch_bond_*`,
`cbe_bug_bond_update` → `cbe_lubich_sweep`, `cbe_bug!` → `cbe_lubich_bug!`.

---

## 1. Headline

**RSVD-CBE now works, in BOTH integrator paths.** The long-standing rank stall — bond
dimensions frozen at `[1,1,2,4,2,1,1]` on L=8 while 2-site TDVP reached `[2,4,6,8,6,4,2]` —
was **two defects in the CBE selection**, not in the sweep. Once fixed, the gate-based and the
Lubich/MPO sweeps both reach the same profile and both beat the validated `bond_update_bug!`.

```
XX domain wall, L=8, dt=0.02, t=0.5, maxdim=32, dex=8   (benchmarks/lubich_vs_bond_update_rank.jl)

bond_update_bug!      err 2.09e-04   profile [2,4,7,12,8,4,2]   172.7s
cbe_bond_update_bug! (gate)  err 1.37e-06   profile [2,4,8,12,8,4,2]    31.1s
cbe_lubich_bug!   (Lubich)   err 8.44e-06   profile [2,4,8,12,8,4,2]    32.2s
```

(That run set `dex = 8` by hand. Since §7.1 the DEFAULT reaches the same regime, so
`lubich_vs_bond_update_rank.jl`'s explicit `DEX = 8` is now redundant rather than load-bearing.)

Tests green: **1019/1019** `tests/RSVDCBEBondUpdate/runtests.jl`, **722/722**
`tests/runtests.jl` (the validated integrator, untouched).

---

## 2. What was done, and why in that order

The stall had four candidate homes — the Lubich basis sweep, the channel environments, the CBE
selection, the placement of the single centre Galerkin — and could not be pinned. The move that
broke the deadlock was the user's: **put the same CBE selection inside the VALIDATED gate-based
odd/even Trotter sweep first.** That removes three suspects at once:

* the Hamiltonian at a bond **is** the gate → no channel environments at all;
* `canonical!(psi, i)` runs before every bond → no basis sweep to get wrong;
* one Galerkin per bond → the step the validated integrator already takes.

Only the selection is left, and it is **shared code, not a reimplementation** — which is what
made the result transfer back to the Lubich path for free.

`cbe_expand` was refactored to take **sketch closures**:

```julia
cbe_expand(f::BondFrame, skl, skr; kwargs...)          # the selection core
cbe_expand(f, h, i, lch, rch; kwargs...)               # MPO wrapper  (cbe.jl)
cbe_expand_bond(f, gate; kwargs...)                    # gate wrapper (cbe_bond_update.jl)
```

Both existing call signatures are unchanged.

### New file: `src/RSVDCBEBondUpdate/cbe_bond_update.jl`

| function | role |
|---|---|
| `sketch_bond_left/right(f, gate, Om)` | `(gate Θ) Ω†` **without ever forming the rank-4 `HΘ`** — `Ω` is folded into the gate first; largest intermediate is `O(d²·g·r)` |
| `cbe_expand_bond(f, gate; …)` | the shared selection with gate sketch closures |
| `cbe_bond_update(f, gate, tau; …)` | drop-in for `BondUpdateBUG.kls_bond_update`; same return fields |
| `cbe_bond_update_sweep!` / `cbe_bond_update_bug!` | the validated `parity_sweep!` / Strang driver, CBE basis update |

It replaces **both** of `kls_bond_update`'s basis mechanisms at once: the two K/L ODE solves
(`expv` on the non-Hermitian `P⊥H_K`) and `missing_fill`'s random per-sector seeds. The S-step
is deliberately untouched — it is the validated integrator's Galerkin step.

---

## 3. The two bugs that were the whole cause

Both have the same shape: **one side of a bond vetoing the other.**

### 3.1 `sector_graded_sketch` dropped the probe-width shortfall

`_alloc_columns` caps every sector at its own dimension, so a half whose subspace is small
silently returned fewer than `npre` columns. **`npre` is the expansion's ceiling** — the
candidates *are* the sketch's columns — so however large `dex` is, a narrow probe caps growth.
Near a chain edge the *opposite* frame's complement is 0- or 1-dimensional, so the probe
collapsed and the expansion **declined with room to spare**. Measured at `comp_ratio=1.0`,
`dex=64` on an L=8 warm state: bond 1 `rank 2 → (2,2)`, bond 3 `6 → (6,6)`, leaving right
residuals of `3.1e-4` and `0.19` untouched.

**Fix:** hand each half's shortfall to the other, and never return an empty probe while the
fused space has dimensions.

### 3.2 The final selection coupled the two sides

* `Nkeep = min(dex_l, dex_r)` throttled each side to the **other's** headroom. Bond 6 has
  `room_r = d·χ_r − r = 2·2 − 3 = 1`, so the left was capped at one direction and its residual
  stayed at `9.97e-3` while the right reached `7.5e-20`.
* The `Empty` verdict was **global**. At bond 3 the left frame already spanned its whole left
  image (residual `8.4e-12`), so `QL` was numerical noise, `norm(UMU) < 1e-11` fired, and
  **nothing** came back on either side — abandoning a right residual of `0.19`.

Root cause of the design error: the reference's CBE is **directional** —
`CBE1SiQS(TL, TR, VL, VR, HL, HR, direction, …)` expands one side per call — so its joint
matrix `UMU` never had to rank two sides. Our two-sided expansion imported a coupling that is
not in the reference.

**Fix:** the joint SVD serves a side only when it can cover that side's whole budget; otherwise
that side falls back to its own preselected candidates, which are already weight-ordered (the
preselection SVD's singular values *are* the weights in the sketched action). Each side decides
independently and either may come back empty without silencing the other.

**Result:** every per-bond residual of the gate's action now reaches machine zero —
`0.19 → 4.8e-12`, `9.97e-3 → 3.1e-16`, `3.1e-4 → 2.7e-19`, `1.35e-5 → 1.0e-16`.

---

## 4. The diagnostic that did the work — keep using it

```julia
left_residual(U, f, gate) = ‖HΘ − U U† HΘ‖ / ‖HΘ‖
```

Printed as a **number**, per bond, next to `r → (dim_l, dim_r)` before and after. "Refused with
room to spare" is then visible at a glance; a pass/fail assertion hides *which side* failed.
Lives in `tests/RSVDCBEBondUpdate/test_cbe_bond_update.jl`; the ~1 min regression probe is
`benchmarks/probe_edge_refusal.jl` (vs ~10 min for the suite).

**Method note that would have saved several cycles:** after any change, check the bond profile
for **bitwise identity** against baseline first. Removing the Sulz bound, changing the growth
schedule, and 4×/8× the candidate budget each produced bit-identical profiles — that check
separates "no effect" from "small effect" in seconds.

---

## 5. RULED OUT — do not re-investigate

| hypothesis | how it died |
|---|---|
| **Stale environments** — the sweep selects against unexpanded far-side envs | Checked against `CBE1SiQS.m` (the CBE-TDVP path, bound at `Run_TDVP.m:129`), *not* the DMRG `RSVDpreBE0SiQS.m`. `PushVR(TRex, HR, VR, …)` takes the **old** `VR` through the **expanded** `TRex` and returns it as an output. CBE feeds unexpanded envs **in** by design — at bond `i` the envs at links `i` and `i+2` describe sites the expansion does not touch. A refactor to "fix" this broke the frame/env pairing and was **reverted**. |
| **The Sulz `2r` bound** | It is on the state's **max** bond dimension, not per-bond, and belongs to the *rank-adaptive* BUG. Removed (`sulz_cap=false`); profile was bit-identical. |
| **Candidate count / `dex` too small** | `dex=8` and `dex=16` gave **bit-identical** profiles to `dex=r`. Not the cause. (`dex` *does* matter for accuracy — see §7.1 — just not for the stall.) |
| **Rotating the expanded frame** | An expansion is **lossless**, so it cannot change the state's Schmidt rank; `canonical!` collapsing a widened bond is correct reporting, not deletion. `CBE1SiQS.m:727-733` makes the losslessness explicit (`TLex·TRex == TL·TR`). Removed. |
| **Per-bond Galerkin in the Lubich sweep** | Over-counts `tau` — it applies the global `H` at `L−1` bonds per sweep (`exact at full rank: 0.0224`). **BUG has no backward correction step**, so projector-splitting is not available as a fix. Abandoned; the carry-based sweep was the real fix. |
| **`comp_ratio = 1.0` = "keep only the discarded space"** | `comp_ratio` governs the **probe**, not what is admitted — `_preselect_*` projects the frame out regardless, so the admitted block is *always* the discarded space. Setting 1.0 instead **blinds the sketch to the 1-site component** ("do not project into complement space, 1-site components are also important for bond update", `RSVDpreBE0SiQS.m:339`) and caps the probe at the opposite frame's complement dimension. **Leave it at the default.** |
| **`stol_pre` / `stol_fnl` tolerances** | Never implicated; they are relative cutoffs and are fine. |

---

## 6. Current status per path

### Gate-based BUG — **validated**
`cbe_bond_update_bug!`. Sits inside the odd/even Trotter sweep, which is exact within a parity group.
103/103 tests, including: sketch pinned against a formed-`HΘ` reference; isometry +
containment + losslessness; **exactness at full rank against `expv` on the block directly**;
`tau=0` is the identity; residual→machine-zero at full budget; rank leaves 1 on step 1 and
keeps growing; U(1) charge conserved; XX profile vs `xx_free_fermion_sz`.

### Lubich/Sulz BUG — **working, unchanged, benefits from the shared fixes**
`cbe_lubich_bug!`. The sweep is the MPS specialisation of Sulz's TTN BUG (K-sweep, L-sweep, seed from
the sweep carries with no M/N overlap matrices, one centre Galerkin, then truncate), matching
`exploratory_bug/src/BUG/discarded_bug.jl:16-31`. It needed **no change of its own** —
`centre rank` now climbs `2,3,5,6,6,6,8,8,9,10,…,12` with `err_fnl = 0`.

> **Correction to an intermediate conclusion in this session:** it was stated at one point that
> "the selection is sound, so the fault in `cbe_lubich.jl` is the sweep." The first half held, the
> second did **not**. There was never a separate sweep bug. Do not act on that earlier claim.

---

## 7. Remaining tasks

### 7.1 The default growth schedule — DONE, re-measured on both paths
**The default is now `growth = 2.0`**: `budget = max(ceil(growth*dmax) - r, 1)`, exposed as a
keyword on `cbe_expand` and a field on `CBEBugOptions`, so both paths inherit it and
`growth = 1.1` reproduces the old behaviour for the A/B. The evidence is the table below.

Why 2.0 and not the reference's 1.1: the reference is **DMRG**, where every sweep
re-converges the same state, so a 10% ratchet compounds over sweeps and costs only
iterations. A **time integrator gets one shot per step** — a direction the step needs and
does not admit is not deferred, it is gone, and the state carries the error forward. At
`r = 13` the 1.1 schedule proposes `ceil(1.1*13) - 13 = 2` directions.

Why a ratio and not an absolute floor of 8: 8 is where *this* chain saturates. The ratio
scales, and wide bonds stay bounded by `room` (a hard limit) and by `maxdim` at the
truncation, which is where a rank ceiling belongs.

#### `growth = 2.0` is a STARTING POINT, not a constant of the method

It was measured on one chain (XX, L=10, a domain wall, `t = 0.5`), and a budget schedule is
exactly the kind of parameter that should be expected to move with the problem — how fast the
entanglement grows, how many channels `H` has, how close `maxdim` is to binding. Treat it as
per-problem and **check it rather than inherit it**. The three diagnostics needed to decide are
now recorded on every run, so this is a procedure and not a judgement call:

| what you see | what it means | what to do |
|---|---|---|
| `err_fnl > 0` | the ranking is discarding candidate weight for want of budget — the cap is the accuracy limiter | raise `growth` (or set `dex` above the saturation point) |
| `err_fnl == 0`, error still too high | budget is *not* the limiter; the subspace or `dt` is | leave `growth` alone; look at `dt`, `maxdim`, `comp_ratio` |
| `err_fnl == 0` and time matters | the budget is past saturation and being paid for nothing | lower `growth` until `err_fnl` first leaves 0 |
| `krylov_dims` high with small `mean_aug` | iterations, not size, are the cost — a poor subspace is hard to converge | raise `growth`; a bigger budget can be *cheaper* |
| `krylov_dims` flat, time still rising | cost is in the sketch/SVDs, which scale with `npre` | lower `growth`, or pin `dex` |

The saturation point is cheap to find — the `dex` curve in `benchmarks/cbe_bond_update_budget.jl`
costs a few seconds per point at these sizes — so for a new problem, sweep `dex` once, read
off where the error stops improving and `err_fnl` reaches 0, and set `growth` so the schedule
lands there at the ranks that problem actually reaches.

`benchmarks/cbe_bond_update_budget.jl` now measures the schedule A/B (`growth = 1.1` vs `2.0`) on
**both** paths alongside the absolute-`dex` curve, since `cbe_lubich_bug!` shares `cbe_expand` and
must be re-measured on the new default rather than assumed to follow.

Measured (`benchmarks/cbe_bond_update_budget.jl`, XX L=10, dt=0.02, t=0.5, maxdim=64; run on a
laptop, one thread, so wall times are ~1.5x the cluster's but internally comparable):

```
bond_update_bug!        err 5.814e-07   maxbond 14   aug 17   141.9s

bond_update (growth=1.1)   err 2.347e-04   maxbond 14   aug 16    24.4s   err_fnl 5.26e-01
bond_update (growth=2.0)   err 1.165e-06   maxbond 13   aug 19     3.9s   err_fnl 0
lubich (growth=1.1)   err 6.972e-06   maxbond  4   aug  4    19.7s   err_fnl 0
lubich (growth=2.0)   err 8.445e-06   maxbond 14   aug 22     3.6s   err_fnl 3.5e-15

bond_update (dex=1)        err 2.148e-03   maxbond 14   aug 15     4.1s   err_fnl 6.71e-01
bond_update (dex=2)        err 1.209e-03   maxbond 12   aug 14     4.8s   err_fnl 1.90e-03
bond_update (dex=4)        err 5.240e-04   maxbond 14   aug 18     6.0s   err_fnl 1.27e-04
bond_update (dex=8)        err 1.165e-06   maxbond 13   aug 19     4.1s   err_fnl 0
bond_update (dex=16)       err 1.165e-06   maxbond 13   aug 19     5.9s   err_fnl 0
bond_update (dex=64)       err 1.165e-06   maxbond 13   aug 19    13.2s   err_fnl 0
lubich (dex=8)        err 8.445e-06   maxbond 13   aug 21    16.5s   err_fnl 0
```

What it says:

* **The new default lands exactly on the saturated end of the curve.** `growth = 2.0` gives
  `1.165e-06`, the same value to all printed digits as `dex = 8, 16, 64`, with `err_fnl = 0`.
  Same on the MPO path: `growth = 2.0` and `dex = 8` both give `8.445e-06`. The schedule is
  no longer the limiter on either path.
* `err_fnl = 0.53` on the old default was the weight ranking **discarding half the candidate
  spectrum for want of budget** — the cap, not the selection, was the accuracy limiter.
* ~~**The generous default is FASTER, not merely affordable**: gate `24.4s → 3.9s`, MPO
  `19.7s → 3.6s`.~~ **RETRACTED — this was a measurement artefact, not the method.** See
  §7.5. The wall times in the table above are the first-of-their-kind rows and carry Julia's
  JIT compilation; the budget buys accuracy at essentially **no** steady-state time cost,
  which is a weaker and correct claim. All `err`, `maxbond` and `err_fnl` columns stand.
* **The 1.1 schedule still stalls the Lubich path outright**: `maxbond 4` against the gate
  path's 14 at the same setting. §3's fixes removed the *selection's* veto; the ratchet was a
  second, independent throttle, and it was hurting the MPO sweep far more than the gate one.
* **A real residual gap remains between the two paths**: gate `1.17e-06` vs Lubich
  `8.44e-06`, ~7x, with both at `err_fnl ≈ 0` and the same maxbond. Budget is now ruled out as
  its cause, so it is the sweep-vs-Trotter difference itself. Next thing to chase after §7.2.
* `bond_update_bug!` is still the most accurate at `5.81e-07`, and the gate path reaches
  `1.17e-06` for **36x less wall time** (3.9s vs 141.9s) at one lower maxbond.

### 7.2 The L=18 XX campaign — calibration done, `main` still to run
`benchmarks/xx_l18_campaign.jl` + `plot_xx_l18_campaign.py`. Previously **voided** by the user
because the scheme was wrong; the scheme is now validated on both paths, so these measurements
mean something. Note `profile_err` must normalise (`site_expval` does not divide by the norm,
and both methods shed norm under truncation).

**`calib 64 2.0` result — CBE-BUG is second order, measured.** L=18, `maxdim = 64`:

```
scheme      dt     err(t=2)   maxbond        ratio per halving
cbe     0.1000   1.0235e-03        28
cbe     0.0500   2.5648e-04        28        3.99
cbe     0.0250   6.4211e-05        25        3.99
tdvp2   0.1000   2.0640e-04        28
tdvp2   0.0500   5.1784e-05        25        3.99
tdvp2   0.0250   1.2969e-05        24        3.99
```

`err/dt²` constant to three digits on both. This closes the open question in
`docs/cbe_bug.md` §3 — the single centre Galerkin does **not** cost an order, and the failure
mode named there (first order) did not happen. CBE-BUG does carry a ~5x larger error
*constant* at equal `dt` and equal untruncated rank; that is a subspace statement, and the
rank/memory comparison has to be read against it.

**But this window cannot pick the main run's `dt`.** `maxbond` peaks at 28 against a cap of
64, so nothing truncates and neither method is dt-converged at any of these `dt` — a `main`
run at `dt = 0.05` would compare two dt-limited runs and report time-integration constants as
if they were rank effects. `phase_calib` now takes `maxdim` and `tmax`
(`calib [maxdim] [tmax]`), so the choice is made where the main run actually reports:

```
calib 64 2.0      # the order check (above)
calib 32 6.0      # the dt choice: at the reported cap, has the error stopped moving?
```

`calib 32 6.0` is **running**; pick `dt` from it, then `main`. A `dt` whose halving barely
moves the error there is dt-safe; one that still quarters it is not.

### 7.3 Substantiate the memory claim — now plumbed end to end, needs the run
`peak_elements` was already wired into both `CBEBugInfo` and `TDVP2Info` at a matched
definition (state + transients alive) but **nothing reported it**. Now it is carried through:
`xx_l18_campaign.jl` records it per sample as a `peak_elems` CSV column, and
`plot_xx_l18_campaign.py` panels (c)/(d) and the printed table read that column instead of
the old `state_elems + theta_elems` proxy.

Dropping the proxy matters for honesty, not tidiness — it flattered CBE-BUG **twice**: its
`theta_elems` is 0 by construction, and its basis sweep is transiently *wider* than the state
that survives truncation, so the state alone understates its true peak. The design thesis is
that CBE-BUG needs a smaller bond dimension than 2-site TDVP at matched accuracy because it
is a 1-site-cost solver that still sees other symmetry sectors. §1 and §7.1 support it on
rank and wall time (13 vs 14, 33× faster); the peak-working-set figure now comes out of the
§7.2 run automatically.

### 7.5 RETRACTED: "the generous budget is faster" — it was JIT compilation

**Every wall-time claim about the expansion budget was wrong, and the mechanism is banal.**
Julia compiles a call chain on first use, `@elapsed` charges that to the first timed row, and
in every one of these benchmarks the starved setting ran first.

The measurement that made it unarguable — three `growth=1.1` runs at caps 8/16/32 in
`cbe_budget_cost.jl chi`:

```
mpo chi<=8  growth=1.1   err 5.710e-03  maxbond 4  wall 160.6s  kry/step 4.9  aug 1.6
mpo chi<=16 growth=1.1   err 5.710e-03  maxbond 4  wall   4.6s  kry/step 4.9  aug 1.6
mpo chi<=32 growth=1.1   err 5.710e-03  maxbond 4  wall   4.6s  kry/step 4.9  aug 1.6
```

Identical error to four digits, identical maxbond (4 — the cap never binds, so the runs are
the *same computation*), identical iteration count, **35x** the wall time on the first. It also
reproduces quantitatively elsewhere: the first gate row is ~280s in both `why` and `model`
(281.6s, 278.8s), the first MPO row 80–160s, every later row 4–25s. And it explains the
original numbers exactly — in `cbe_bond_update_budget.jl`, `bond_update_bug!` ran first and absorbed
most of the shared Telum compilation, leaving the first CBE row ~20s of residue:
`24.4s ≈ 3.9s + 20s`, `19.7s ≈ 3.6s + 16s`.

**The Krylov hypothesis is dead too, and independently of the timings** — the counters are
collected inside the runs, so contention cannot reach them:

* gate path, starved vs generous: `347.9` vs `343.6` iterations per step — **identical** — with
  the generous run doing **more** total iteration-times-size work (601.8k vs 486.6k).
* MPO path, starved: **6x fewer** iterations on **4x smaller** objects, i.e. cheaper by every
  internal measure. Nothing internal could have made it slower.

**Fixes applied.** `cbe_budget_cost.jl` now calls `warmup(L)` — one throwaway step per path and
per Hamiltonian — before any timing, and `why rev` reverses row order so an order-dependent
artefact cannot survive unnoticed. **Any future timing table in this repo needs the same
warm-up; without it the first row is a compilation measurement.**

**What still stands:** the accuracy case for `growth = 2.0` is untouched (`err_fnl 0.53 → 0`,
`1.4e-4 → 1.2e-6`, and the Lubich sweep stalling at maxbond 4 under the 1.1 ratchet).

**The corrected cost table**, warmed up, and run in BOTH row orders on a verified-idle box
(XX L=10, dt=0.02, t=0.5, maxdim=64; forward / reversed):

```
setting                      err        maxbond   fwd    rev    kry/step
cbe_bond_update growth=1.1   2.347e-04     14    10.5s   9.1s     347.9
cbe_bond_update growth=2.0   1.165e-06     13     9.5s   9.4s     343.6
cbe_bond_update  maxiter=8   1.165e-06     13     6.3s   5.7s     105.8
cbe_lubich      growth=1.1   6.972e-06      4     3.5s   3.1s       4.8
cbe_lubich      growth=2.0   8.445e-06     14     6.8s   5.6s      27.8
cbe_lubich       maxiter=8   8.445e-06     14     6.1s   5.5s       7.6
```

* **On `cbe_bond_update` the budget is free** — 9.5 vs 10.5s forward, 9.4 vs 9.1s reversed, i.e.
  1.0 ± 0.1 either way — for a **200x** accuracy gain.
* **On `cbe_lubich` the budget costs ~1.9x** (3.5 → 6.8s), which is the direction a cost model
  predicts. The starved run is cheap only because it is STALLED at maxbond 4 and is not doing
  the physics: at L=12, t=1.5 it is 22x less accurate (`5.71e-03` vs `2.64e-04`).
* **`maxiter = 30` is the wasteful default, not the budget — but only on one path.** Capping at
  8 cuts `cbe_bond_update` by **~36%** (9.7 → 6.2s) with the error *unchanged to all printed
  digits* (`1.165e-06`); on `cbe_lubich` it saves **nothing** (5.6 → 6.0s, inside noise). That
  asymmetry is structural and worth keeping in mind: `cbe_bond_update` solves one exponential
  per BOND (~344 iterations per step, so ~36% of its cost), while `cbe_lubich` solves ONE at
  the centre (27.8 iterations per step — a rounding error against its sweeps). **New task:
  sweep `maxiter` properly and re-default it for the bond-update path.**
* `dex = 64` at 13.2s against `dex = 8` at 4.1s says cost does rise with budget, so an
  unboundedly generous budget is not free either.

**Note on L=10 as a discriminator:** at t=0.5 the Lubich path's accuracy does NOT separate the
schedules (`6.97e-06` starved, nominally *better* than `8.44e-06` generous) because both are
dt-limited there. The accuracy case for that path rests on the longer window. Do not read a
short-window run as evidence about rank.

### 7.4 Deferred by the user: fixed-rank vs rank-adaptive
For that comparison, reinstate the bound as *max rank of the new orthonormal K/L matrices ≤ 2r*.
`sulz_cap = true` is already wired in the **global** `2·rmax` form. Explicitly "for later".

### 7.6 Iterating the basis instead of tuning the probe — MEASURED, mostly negative

**The idea.** The one-shot expansion must guess the right subspace in a single draw, helped by
two hyperparameters: the oversampling (`dover`, default `1.2×`) and the complement/isometry
split (`comp_ratio`, default `0.5`). The proposal was to stop guessing — take a narrow probe,
re-expand repeatedly, and let the basis converge on the relevant subspace, stopping when the
Krylov matrix goes small.

Two schemes were built and measured (`benchmarks/cbe_s_iterations.jl`, L=8, bond (4,5), r=7,
against the **exact** two-site propagator):

**(a) Outer loop** (`s_iters = n > 1`): re-sketch from the state the previous pass evolved.
**Plateaus after two passes**, five orders short of the one-shot.

| setting | err vs exact | rank | passes |
|---|---|---|---|
| `dex=1 s_iters=1` | 7.78e-10 | 8 | 1 |
| `dex=1 s_iters=2…8` | 5.50e-10 | 9 | stops at 2 |
| `growth=2.0` one-shot | **5.60e-16** | 11 | 1 |

Why it cannot work: `exp(τH)Θ₀ = Θ₀ + τHΘ₀ + O(τ²)`, so re-sketching a *converged exponential*
keeps rediscovering `HΘ₀` and never reaches `H²Θ₀`. An outer loop cannot build a Krylov space.

**(b) Expanding-basis Lanczos** (`s_iters = -g`, `_expanding_lanczos`): growth becomes **step 0
of each Krylov iteration** — expand around `v_k` *before* applying `H` to it, so the bond
dimension runs `D`, `growth·D`, `growth²·D`, …  Here `v_k` genuinely carries `H^k`, so the
objection to (a) does not apply.

| setting | err vs exact | rank | matvec |
|---|---|---|---|
| `dex=1 grow=1` | 7.778e-10 | 8 | 22 |
| `dex=1 grow=6` | 7.777e-10 | **12** | 17 |
| `growth=2.0 grow=1..3` | 4.682e-16 | 11 | **3** |
| `growth=2.0` one-shot, `reorth=false` | 5.600e-16 | 11 | 17 |
| `growth=2.0` one-shot, **`reorth=true`** | 4.675e-16 | 11 | **3** |

**Two findings, and the second kills the headline.**

1. **A starved probe cannot be rescued by Krylov depth.** At `dex=1` the rank grows 8 → 12 —
   *larger* than the 11 the default uses — and the error does not move (7.78e-10 vs 4.68e-16).
   Directions drawn from a 2-column probe are poor, and more of them does not help. The
   oversampling is a **reliability threshold, not a tunable to trade against depth**.

2. **The 17 → 3 matvec drop is full reorthogonalisation, not the expanding basis.** The control
   (`s_reorth = true`, one-shot, same expansion) reaches 3 matvecs on its own. The expanding
   basis must not be credited for it.

**β is not a sufficiency criterion.** `dex=1 grow=1` terminates with β = 2.8e-24 — as small as a
stopping signal ever gets — and is still wrong by 7.8e-10. β measures whether the Krylov space
has closed *inside the current basis*; it says nothing about whether that basis contains the
right directions. A starved basis closes early, confidently, and wrongly.

**Also measured: the hyperparameters barely matter at this bond**, so there was little to
replace. `comp_ratio` 0.0 → 1.0 moves the error only 1.63e-15 → 5.60e-16; `dover` 0 → 8 moves
nothing. The tuning burden the scheme was meant to remove is not, on this evidence, large.

`tau = 0` stays the identity for both schemes (1.29e-16 / 2.53e-16), so neither double-evolves.

**Actionable:** `s_reorth = true` is a ~5.7× matvec saving at no accuracy cost on this bond, and
it compounds with the known `maxiter` waste (§7.1). **Not yet a default** — it needs a
sweep-level check across all bonds and a longer window before changing one.

#### 7.6.1 Why none of it could discriminate — `growth = 2.0` saturates room when `d = 2`

This is the finding that reframes the rest, and it is structural rather than an artefact of
`L = 8`. At a bond whose neighbours are the same width (`χ = r`, so `dmax = r`):

```
room   = d·χ − r              →  r      (d = 2)
budget = ⌈growth·dmax⌉ − r    →  r      (growth = 2.0)
dex    = min(budget, room)    =  room
```

**The expansion therefore spans the entire local space.** The ranking never has to choose,
`err_fnl = 0` at every bond, and every variant — every `comp_ratio`, every `dover`, every
iteration scheme — is computing the *exact two-site propagator* and must agree to machine
precision. Confirmed directly: `dex = 64` (no cap at all) gives 7.44e-16, the same as the
default. **No experiment run at `growth = 2.0` on a spin-½ chain can discriminate between
selection strategies**, because there is no selection happening.

Consequences worth carrying forward:

* Re-read §7.1: at `d = 2`, `growth = 2.0` does not mean "a generous budget", it means
  "expand to the complete local space". That is why the budget looked free and 200× more
  accurate, and why the RSVD's width advantage is eroded (§Cost in the write-up) — the probe
  is as wide as the thing it is sketching.
* The discriminating knob is **`growth`, not system size**. `growth < 2.0`, `d > 2`, or a
  `maxdim` that binds hard enough to make neighbouring bonds uneven are the regimes where the
  ranking, the hyperparameters, and any iteration scheme can differ at all.
* Section G of `benchmarks/cbe_s_iterations.jl` runs `growth ∈ {1.1, 1.25, 1.5}` for this
  reason, and it is the only part of that benchmark whose result carries information about the
  selection.

---

## 8. Gotchas that cost real time

* **Telum has no QR primitive** — only `svd` + `getIdentity`. `frame.jl` uses `svd(…; cutoff=0.0)`
  as a rank-revealing stand-in, which is *why* re-deriving a frame from a zero-padded tensor
  collapses it. That was the original cause of the maxbond-2 freeze: the basis sweep was
  **mutating the state**; fixed by the carry-based sweep (`_frame_from` on carried tensors,
  frames accumulated in `W`/`Z`, `psi` assembled only after the Galerkin step).
* **Telum's SVD selector is strict** (`sv > cutoff*sv_max`) — exactly-zero directions are dropped
  even at `cutoff = 0.0`.
* **Tag collisions**: a `TLArray` may not carry two identical non-empty itags. The two
  preselections need distinct tag pairs (`_TAG_L`/`_TAG_R`), as `bond_frame` already does.
  On assembly, retag **both** bond legs — adjacent tensors must agree on the shared link tag.
* **`round(Int, 2*0.25) == 0`** (banker's rounding) — interior `comp_ratio` values are degenerate
  at narrow `npre`. State endpoint claims at `npre = min(ndc, ndt)`.
* **`npre ≥ dim(fused space)`** forces the probe to take everything, so `comp_ratio` provably
  cannot change the mix there. Measure the split at a width the space can apportion.
* **Test the sector claim against the physics.** On a product domain wall only the ONE bond
  straddling the wall can hop; elsewhere `P⊥HΘ = 0` exactly and refusing to expand is *correct*.
  Assert against `P⊥HΘ`, never against an assumed bond list.
* **Total Sz of a half-filled domain wall is 0**, so a charge-conservation test on it passes on a
  zero target. Use 5 up / 3 down (Sz = +1).
* **A cancelled job can keep running.** Killing a wrapper script does NOT kill the `julia`
  child it launched — it keeps a core and silently taxes whatever is measured next. This cost
  two tables in one session: a "cancelled" L=18 calibration inflated the first cost table ~10x,
  and a "cancelled" re-run inflated the second by ~2.2x. **Before any timing run, verify the
  box is idle** (`Get-Process julia` / `pgrep julia`) rather than assuming a cancel took, and
  kill leftovers by PID.
* **Never trust one timing pass.** The order-swap control (`why rev`) is what caught the second
  contamination: settings that agreed between passes were real, the ones that disagreed by 2x
  were the machine. Rows that are the SAME COMPUTATION (e.g. a stalled run at three different
  caps that never bind) are the sharpest check available — they must agree to noise, and when
  they read 160.6s/4.6s/4.6s the table is measuring something other than the method.
* **`find` on this cluster is `bfs`** — `pgrep -f "^find …"` does not match a running tree walk and
  a kill silently does nothing. Match on the command line and *verify*. And never walk the NFS
  home: `/home` is `10.1.0.110:/volume1/slurm-home`.

---

## 9. How to run things

```bash
# fast: the gate path only (~5 min)
sbatch --job-name=cbegate --partition=short scripts/run_julia.sbatch \
       tests/RSVDCBEBondUpdate/test_cbe_bond_update.jl

# the CBE module suite (~10 min) and the validated suite (~7 min)
sbatch --partition=main scripts/run_julia.sbatch tests/RSVDCBEBondUpdate/runtests.jl
sbatch --partition=main scripts/run_julia.sbatch tests/runtests.jl

# the two measurements in this document
sbatch --partition=short scripts/run_julia.sbatch benchmarks/cbe_bond_update_budget.jl
sbatch --partition=short scripts/run_julia.sbatch benchmarks/lubich_vs_bond_update_rank.jl

# ~1 min regression probe for "refused with room to spare"
sbatch --partition=short scripts/run_julia.sbatch benchmarks/probe_edge_refusal.jl

# ── QUEUED, all cluster work (see WHICH RUNS GO WHERE below) ────────────────────
# why is the generous budget FASTER, is it XX-specific, and does it scale in chi?
sbatch --job-name=cbecost --partition=short scripts/run_julia.sbatch \
       benchmarks/cbe_budget_cost.jl why
sbatch --job-name=cbemodel --partition=short scripts/run_julia.sbatch \
       benchmarks/cbe_budget_cost.jl model
sbatch --job-name=cbechi --partition=main scripts/run_julia.sbatch \
       benchmarks/cbe_budget_cost.jl chi

# the L=18 campaign: choose dt where the cap binds, then the main run
sbatch --job-name=xx18_calib32 --partition=short scripts/run_julia.sbatch \
       benchmarks/xx_l18_campaign.jl calib 32 6.0
sbatch --job-name=xx18_main --partition=main scripts/run_julia.sbatch \
       benchmarks/xx_l18_campaign.jl main <dt from calib 32 6.0>
```

**WHICH RUNS GO WHERE.** Local is for **small system sizes only** — the test suites, and
L ≤ ~12 probes that finish in seconds to a couple of minutes. Everything else goes to the
cluster via `sbatch`: L ≥ 16, long time windows, `maxdim` sweeps, the L=18 campaign, and **any
wall-clock comparison** (a laptop shares cores with whatever else is running, so timings taken
next to another job are not comparable). Measured the hard way this session: `calib 32 6.0`
at L=18 ran long enough locally to block the timing benchmarks it was queued ahead of, and
had to be killed and requeued. Size the local run to the question — a mechanism that breaks
at L≥4 does not need L=18 to show it.

**Running it off the cluster.** When the cluster is unreachable (VPN down — `ssh login-node`
to `10.1.0.120` just times out) the whole suite runs on a laptop, and the L=8/L=10
measurements in this document are small enough to be worth doing there:

```bash
# once: a project outside the repo, so instantiating never dirties the Manifest
cp -r BUG-Julia /tmp/bugjl && cd /tmp/bugjl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. deps/build.jl        # the Telum :none SVD patch, NOT `Pkg.build`
julia --project=. --threads=1 tests/RSVDCBEBondUpdate/test_cbe_bond_update.jl
```

Two gotchas, both measured: `Pkg.build("BUGJulia")` re-resolves in a sandbox and dies on
`JSON = "1.6.1"` against an older local registry — run `deps/build.jl` directly instead, it
is the same patch without the sandbox. And first-time precompilation of Telum/HDF5/MKL takes
several minutes during which the log stays EMPTY (block-buffered to a file), so an empty log
is not a hung run. Timings are ~2x the cluster's on one thread; run wall-clock comparisons
one at a time, never alongside a test suite.

**Standing constraints.** On the cluster, run everything via `sbatch` on compute nodes — never
Julia on the login node (4 cores, 11 GB, `/home` on NFS). Never write `--time`; inherit the partition max
(`short` 1 h, `main` 7 d, `long` 30 d). Stay under 300 GB memory *and* 50 cores across all jobs;
`short_qos`/`long_qos` cap `cpu=32` per user. Real time only: `tau = -im*dt`. Do not modify
`src/BUG` / `src/BondUpdateBUG` unitarity-critical components without an explicit request. The
naming is **`bond_update_bug`**, never "two-site BUG".

## 10. Reference files

Cloned via `https://gitlab.physik.uni-muenchen.de/LDAP_ls-vondelft/QSMPSLib.git` (the `lmu.de`
URL hits a login page).

* `CBE1SiQS.m` — **the CBE-TDVP path**, bound at `Run_TDVP.m:129`. Use this one, not the DMRG file.
  `:76`/`:319` `PushVR`; `:727-733` `ExpandTR` derives the padding via the overlap `M = TRex†TR`.
* `RSVDpreBE0SiQS.m` — the **DMRG** path. `:66-77` growth schedule; `:339` the 1-site-components
  warning; `:461` the `Empty` branch; `:516` has a `%!!!!! wrong!` note on direct-sum ordering.
  The two references disagree on complement-first vs original-first; forming `Ŝ₀ = U_ex†Θ₀V_ex†`
  directly (as we do) sidesteps it.
* `Run_TDVP.m:125-128` — CBE-TDVP's forward 1-site `-dTime/2*1j` + backward 0-site `+dTime/2*1j`
  pair. **BUG replaces that whole mechanism with the single centre Galerkin — take the
  environment handling from here, explicitly not the forward/backward pair.**
* `exploratory_bug/src/BUG/discarded_bug.jl:16-31` — the authoritative chain-BUG sweep.
