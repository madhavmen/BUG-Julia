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
At `growth = 2.0` it looks useless — flat after two passes. **That reading was wrong**, and
§7.6.1 explains why: at `growth = 2.0` the first pass already takes the whole local space.
Measured where the budget is *below* the room, it is the best scheme tested (§7.6.2).

It does **not** build a Krylov space — `exp(τH)Θ₀ = Θ₀ + τHΘ₀ + O(τ²)`, so each pass
re-derives `HΘ₀`, never `H²Θ₀`. What it does is **compound the growth schedule**: the budget is
recomputed from the *current* rank each pass, so `r → growth·r → growth²·r`, and every rung of
that ladder is probed by a sketch whose width matches the rank it is probing. A sequence of
well-conditioned narrow probes beats one wide draw.

**(b) Expanding-basis Lanczos** (`s_iters = -g`, `_expanding_krylov`): growth becomes **step 0
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

#### 7.6.2 The discriminating run — and it inverts two conclusions above

`growth ∈ {1.1, 1.25, 1.5}` at the same bond, budget now **below** room (`room_l = room_r = 5`,
`r = 7`). Error is against the exact two-site propagator.

| growth | budget | scheme | err | rank | matvec |
|---|---|---|---|---|---|
| 1.10 | 1 | one-shot | 7.778e-10 | 8 | 30 |
| 1.10 | 1 | `comp_ratio` 0.0 / 0.5 / 1.0 | 7.778e-10 (all) | 8 | 30 |
| 1.10 | 1 | outer `s_iters=3` | 5.501e-10 | 9 | 60 |
| 1.10 | 1 | expanding `grow=4` | 7.777e-10 | 10 | 23 |
| **1.25** | 2 | one-shot | 1.449e-11 | 9 | 30 |
| **1.25** | 2 | `comp_ratio` 0.0 / 0.5 / 1.0 | 1.440–1.449e-11 | 9 | 30 |
| **1.25** | 2 | **outer `s_iters=3`** | **1.978e-15** | **12** | 60 |
| **1.25** | 2 | expanding `grow=4` | 1.450e-11 | **12** | 11 |
| 1.50 | 4 | one-shot | 1.401e-15 | 11 | 30 |
| 1.50 | 4 | `comp_ratio=0.0` | 2.476e-14 | 10 | 20 |
| 1.50 | 4 | `comp_ratio=1.0` | 2.671e-16 | 11 | 20 |
| 1.50 | 4 | one-shot `+ reorth` | 4.675e-16 | 11 | **3** |
| 1.50 | 4 | expanding `grow=1` | 4.682e-16 | 11 | **3** |

**Corrections to §7.6, both material:**

1. **The outer loop is not useless — it is the best scheme measured.** At `growth = 1.25` it
   takes the error from `1.449e-11` to `1.978e-15`, a factor of **7300**, by growing rank 9 →
   12. The earlier "plateaus, five orders short" reading came entirely from `growth = 2.0`,
   where the first pass already takes the whole local space.
2. **The expanding-basis Lanczos does not work, and rank is not the reason.** At `growth = 1.25`
   it reaches the *same rank 12* as the outer loop and stays at `1.450e-11` — four orders worse
   at identical basis size. What differs is *which* directions get admitted, not how many.
   Likely cause: its probe is the Krylov residual `v_k`, not the physical evolved state, and its
   tridiagonal is assembled across changing subspaces (the `beta_k` approximation flagged in
   `_expanding_krylov`). Not yet isolated.

**Hyperparameters still barely matter, with one exception.** At `growth` 1.1 and 1.25,
`comp_ratio` 0.0 → 1.0 and `dover` 0 → 4 move the error by <1%. Only at `growth = 1.5` does
`comp_ratio` bite: `2.476e-14` (0.0, and it loses a direction — rank 10) vs `2.671e-16` (1.0), a
100× spread. Below that the probe is too narrow (`npre = ⌈1.2·1⌉ = 2`) for the split to mean
anything.

**So the original proposal is vindicated, in its first form.** Iterating the expansion does
replace hyperparameter tuning — but it is the *outer* loop compounding the growth schedule that
does it, not growth inside the Lanczos.

**Caveats before acting on this:** one bond, one state, one `dt`, three `growth` values. The
7300× needs a sweep-level and longer-window confirmation before `s_iters > 1` becomes a
recommended setting, and the cost is real (60 matvecs vs 30). The obvious follow-up is
`s_iters = n` with `growth ≈ 1.25` versus one-shot `growth = 2.0` at matched *total* work.

#### 7.6.3 The sweep, against an ANALYTIC reference — the outer-loop result does not survive

Everything in §7.6.2 is one bond of a warmed state judged against an exact two-site propagator.
That is a diagnostic, not a decision. `benchmarks/cbe_sweep_cost.jl` runs whole evolutions
(L=8, dt=0.02, t=0.5, maxdim=32), warms up before any timing, and scores against the **exact**
answer rather than another integrator.

**The reference is now analytic.** The XX chain is free-fermionic: Jordan–Wigner on
`H = J Σ (SxSx + SySy)` gives `H = (J/2) Σ (c†ⱼcⱼ₊₁ + h.c.)`, a tridiagonal single-particle
matrix, so for a Fock initial state `nⱼ(t) = Σₖ |[exp(−iht)]ⱼₖ|² nₖ(0)` is **exact for the
finite open chain** — no infinite-chain or short-time assumption, unlike the Bessel-function
domain-wall profile. Two conventions cannot go wrong: the hopping's *sign* is irrelevant (it
enters as `|U|²`), and the wall's orientation is read off the MPS rather than assumed. Checks:
`‖analytic(0) − ψ(0)‖ = 0.000e+00`, and `‖tdvp2(t) − analytic(t)‖ = 1.76e-06`, dt-limited.

| scheme | err | wall(s) | matvec | maxbd |
|---|---|---|---|---|
| growth=2.0 one-shot (default) | 1.940e-06 | 6.38 | 6523 | 12 |
| growth=2.0 + reorth | 1.940e-06 | 5.97 | **2672** | 12 |
| growth=1.25 one-shot | 1.537e-05 | 6.35 | 6639 | 12 |
| growth=1.25 + reorth | 1.537e-05 | 6.05 | 2805 | 12 |
| growth=1.25 **outer `s_iters=3`** | 1.199e-05 | **10.55** | 12850 | 12 |
| growth=1.25 expanding grow=2 | 9.256e-06 | 5.92 | 1986 | 12 |
| growth=1.5 one-shot | 1.940e-06 | 5.67 | 6606 | 12 |
| growth=1.5 expanding grow=2 | **1.399e-06** | 5.42 | **1452** | 12 |
| growth=1.5 expanding grow=8 | 1.399e-06 | 5.76 | 1155 | 12 |

**The outer loop reverses.** Best scheme at one bond (§7.6.2, 7300× better at `growth = 1.25`);
across a full evolution it is the **worst** — 6.2× the default's error, 1.65× the wall time,
1.97× the matvecs. A single-bond improvement against an exact local propagator did not survive
composition over 25 steps and all bonds. **Do not promote a bond-level result again without
this run.**

**`s_reorth` confirms — with an important qualification.** Identical error, 59% fewer matvecs
(0.41×), but only ~6% less wall time. So **matvecs are not the dominant cost**: the expansion
work (sketches, preselection and ranking SVDs) is. Any scheme sold on a matvec count is
overstating its case, including the ones above.

#### 7.6.4 With the control in: the expanding solver DOES earn its place

The row that decides it is `growth = 1.5 + reorth` — same reorthogonalisation the expanding
solver does, without the expanding basis. Adding it, and a `growth = 2.0` expanding row:

| scheme | err | matvec | err vs default | matvec vs default |
|---|---|---|---|---|
| growth=2.0 one-shot (default) | 1.940e-06 | 6523 | 1.00× | 1.00× |
| growth=2.0 + reorth | 1.940e-06 | 2672 | 1.00× | 0.41× |
| growth=1.5 one-shot | 1.940e-06 | 6606 | 1.00× | 1.01× |
| **growth=1.5 + reorth (CONTROL)** | **1.940e-06** | **2693** | **1.00×** | **0.41×** |
| growth=1.5 expanding grow=2 | **1.399e-06** | 1452 | **0.72×** | 0.22× |
| growth=2.0 expanding grow=2 | **1.399e-06** | **1386** | **0.72×** | **0.21×** |

**Reorthogonalisation alone changes the error by nothing** (1.940e-06, identical to one-shot at
every `growth`); it only halves the matvecs. **The expanding basis lowers the error to
1.399e-06 and halves the matvecs again** relative to the control. So the accuracy gain is
attributable to the expanding basis, not to reorthogonalisation — which reverses the bond-level
verdict in §7.6 that it "buys nothing and pays an extra expansion per bond". At sweep level it
is the best scheme measured on both reliable axes.

**Two caveats that stop this being a decision yet.**

1. **Wall-clock is unusable, worse than §7.6.3 suggested.** The *identical* default
   configuration timed **6.38 s** in one run and **16.45 s** in the next — a 2.6× swing on the
   same computation, despite a warm-up pass. Every "relative wall" figure in that run is
   measured against an inflated baseline and must be ignored. This is §7.5's artefact
   resurfacing; a 2-step warm-up does not compile every scheme's path. **Only `err` and
   `matvec` are trustworthy here** (both deterministic).
2. **The errors sit at the dt-limited floor.** 2-site TDVP reaches 1.761e-06 at this `dt`, and
   the schemes span 1.399e-06 – 1.940e-06 — all the same order, all at maxbd 12. The 28% gap
   may be a Trotter-error prefactor rather than basis quality. **A `dt` scan settles it** —
   and it did: see §7.6.5. It is not a prefactor.

#### 7.6.5 The `dt` scan — the expanding basis RESTORES second order

`benchmarks/cbe_dt_scan.jl`, fixed `t = 0.5`, halving `dt`, against the analytic reference. A
second-order scheme quarters its error per halving (ratio 4).

| dt | one-shot / +reorth | ratio | expanding grow=2 | ratio |
|---|---|---|---|---|
| 0.020 | 1.9402e-06 | — | 1.3986e-06 | — |
| 0.010 | 5.6045e-07 | **3.46** | 3.4964e-07 | **4.00** |
| 0.005 | 1.8639e-07 | **3.01** | 8.7411e-08 | **4.00** |

*(the `dt = 0.04` row is outside the asymptotic regime — err 6.3e-03 for every scheme — so only
the last two ratios carry information.)*

**The expanding solver is cleanly second order; the one-shot is not.** Its ratio decays
3.46 → 3.01, drifting away from 4 as `dt` shrinks — the signature of a **basis floor**, an error
that does not scale as `dt²` and so dominates more as the Trotter error falls. Reorthogonalisation
does not touch it (the control is bit-identical to the one-shot at every `dt`).

**So the 28% was not a prefactor.** A prefactor holds the ratio at 4 and keeps the gap constant;
instead the gap *widens*: 1.39× at `dt=0.02`, 1.60× at 0.01, **2.13×** at 0.005. The one-shot
expansion leaves physics unresolved and the in-Lanczos growth recovers it.

**And it is cheaper, increasingly so.** At `dt = 0.005`: 4748 matvecs against 27255 for the
one-shot (0.17×) and 10739 for the reorth control (0.44×).

`growth` is irrelevant to the expanding solver (1.5 vs 2.0 gives 3.4956e-07 vs 3.4964e-07),
consistent with §7.6.1 — once the growth happens inside the Krylov iteration, the one-shot
budget stops mattering.

**Status: the strongest result in this section, and the one worth acting on.** Remaining before
`s_iters = -2` becomes a default: confirm on a non-free-fermion model (XXZ with `delta ≠ 0`,
where the analytic reference is unavailable and 2-site TDVP must serve), and at a larger `L`
where `maxdim` binds hard — both cluster runs. Wall-clock must be re-measured there with
repeats, since it is unusable on this box (§7.6.4).

**Wall-clock generally is weak evidence on this box.** Trust the matvec column; treat wall
differences below ~1.3× as noise until repeated, and note that even 2.6× turned out to be an
artefact here.

### 7.6.6 The χ-scan that measured nothing, and why

`benchmarks/cbe_chi_scaling.jl` was written to answer "does the expanding solver blow up at
large χ". Its first version used a **melting domain wall**, and it produced this:

| maxdim | wall(s) | p | err | matvec | maxbd |
|--------|---------|-------|-------------|--------|-------|
| 8 | 1.33 | — | 8.1005e-06 | 2532 | 8 |
| 16 | 1.21 | −0.13 | 8.1005e-06 | 2549 | 10 |
| 32 | 1.28 | 0.08 | 8.1005e-06 | 2549 | 10 |
| 64 | 1.29 | 0.01 | 8.1005e-06 | 2549 | 10 |

Identical error, identical matvecs, identical rank from `maxdim = 16` up: **the cap never
bound**, so χ = 32 and χ = 64 were the same run as χ = 16 and the fitted exponent was pure
jitter on ~1.2 s runs. All four configurations looked the same because they *were* the same.

**Cause: a domain wall is the one free-fermion quench whose entanglement grows only
logarithmically**, so at L = 12 it saturates at χ = 10. A **Néel** quench grows it linearly.
Measured at L = 12, `maxdim = 64`: χ = 25 at t = 1, 44 at t = 2, 64 (saturated) at t = 4. The
benchmark now uses Néel at t = 2, where the natural χ is 44 so caps 8/16/32 bind and 64 does
not — the only configuration in which "wall time versus χ" is a statement about χ.

**Generalise the lesson:** a benchmark in which every arm agrees to the last digit is not
evidence of equivalence, it is evidence that the knob under test was never engaged.

---

## 7.7 Three modes on one engine — real time, imaginary time, ground state

**Layout.** `cbe.jl` + `cbe_bond_update.jl` merged into **`cbe_core.jl`** (the shared engine;
Part I is the RSVD selection, Part II drives it and solves the local problem), with
`CBEBugOptions` / `CBEBugRunInfo` moved in from `cbe_lubich.jl` since all three modes are
configured by them. The modes are thin submodules in `src/RSVDCBEBondUpdate/modes/`.

**The modes differ in exactly two places.** `_expanding_lanczos` became
`_expanding_krylov(f, applyH, expand, solver, theta0, nmv; …)`, and the growth passes, sketch,
`P⊥` projection, ranking, direct sum, Krylov-vector embedding and reorthogonalisation are one
copy of one function shared by all three. What varies:

| mode | `applyH` at a bond | S-step reduction | path |
|---|---|---|---|
| `RealTime` | gate, or env-dressed `H` | `exp(-i·dt·H)` | gate + MPO |
| `ImaginaryTime` | gate, or env-dressed `H` | `exp(-dt·H)` | gate + MPO |
| `GroundState` | env-dressed `H` **only** | lowest Ritz vector | MPO |

So the ground-state solver is a **drop-in replacement for the S step**, not a second algorithm.
Real and imaginary time now differ by one argument to `cbe_gate_evolve!`, and
`cbe_bond_update_bug!` is a one-line alias at `tau = -im*dt` so every existing caller is
untouched.

### 7.7.1 Why the gate path has no environments, and what that costs

It *has* environments — they are the **identity**, by construction rather than by neglect.
`H = H_odd + H_even`, each group is a sum of two-site terms on disjoint pairs, so
`exp(tau·H_odd)` factorises exactly into one `exp(tau·h_i)` per bond: the operator at a bond
IS the bare gate, and the rest of the chain is not approximated away, it is a different factor
of the same product applied at its own bond. `canonical!(psi, i)` before each bond makes
everything left a left isometry and everything right a right isometry, so `L = R = I` exactly.
Canonicalisation *is* the environment storage, and it costs a QR sweep instead of a stack.

**The consequence, and it is what fixes the mode/path matrix:** the gate path's local operator
has no knowledge of the chain, so a local *eigensolve* there returns the ground state of one
two-site term with frozen neighbours — a different quantity, not an approximation to the
global ground state. Each bond would drive its own pair toward a local singlet and the next
bond would undo it. DMRG converges precisely because `L + h + R` makes a local minimisation
lower the **total** energy. Ground state is therefore MPO-path only. Imaginary time is
unaffected: Trotterisation does not care whether `tau` is real or imaginary.

### 7.7.2 The two-phase local solve — buying back the variational bound

`_expanding_krylov`'s tridiagonal is a Galerkin approximation across a **nested sequence of
subspaces**: `alpha_k` is exact, but `beta_k` is measured against the projector in force at
step `k` and misses whatever part of `H·v_k` fell outside the then-current basis. For a
short-time exponential that is a small correction to a small correction. **For an eigensolve it
is not tolerable** — the Ritz value *is* the answer, and one from an inconsistent tridiagonal
is not variational, so it can sit *below* the true ground energy and report false convergence.

Hence the local solve is two phases: **A** grows the frames (`grow_iters` passes driven by
successive Krylov vectors), **B** runs a clean Lanczos in the now-**fixed** frames, which is an
exact Rayleigh–Ritz. Phase B is the same function with `grow_iters = 0`, so there is still only
one Lanczos to maintain, and phase A costs 2–3 matvecs — cheap for a variational guarantee.
`E − E_exact ≥ 0` is therefore a **correctness assertion**, not a hope.

### 7.7.3 Validation

**Real time — bit-identical across the refactor**, on both solver paths, and
`RealTime.evolve!` returns exactly what `cbe_bond_update_bug!` does (`==`, not `isapprox`):

| variant | err | pre-refactor |
|---|---|---|
| one-shot | 1.172340e-06 | 1.172340e-06 |
| expanding | 9.119571e-07 | 9.119570e-07 |

**Imaginary time — and a trap worth naming.** There are two independent errors and reading a
residual without separating them misattributes it, because both shrink under "run it harder":

* incomplete cooling `~ exp(-2β·gap)` — falls with **β**, flat in `dt`
* Trotter splitting `~ dt²` — falls with **`dt`**, flat in β

L = 8, δ = 1, half filling, exact `E₀ = -3.3749325987`:

| β | dt | E − E_exact | ratio |
|---|-----|-------------|-------|
| 5 | 0.05 | 1.219e-02 | — |
| 10 | 0.05 | 2.476e-04 | 49.3 |
| 20 | 0.05 | 1.691e-07 | 1464 |
| 40 | 0.05 | 7.307e-08 | 2.31 |
| 40 | 0.10 | 1.167e-06 | — |
| 40 | 0.025 | 4.568e-09 | — |

At β = 10 the residual is **2.48e-04 at both `dt = 0.1` and `dt = 0.05`** — unmoved by halving
`dt`, so it is entirely the cooling term and says nothing whatever about the splitting error.
(An earlier draft of the check labelled it `O(dt²)`; the flat column is what refuted that.)
The β ratio collapsing from 1464 to **2.31** at β = 20→40 is the cooling term crossing below
the Trotter floor: past that point extra β buys nothing and only `dt` moves the number.

**The `dt` ratios at β = 40 are 15.98 and 15.99 — fourth order, and expected rather than
suspicious.** The energy is stationary about the ground state, so a state error of `O(dt²)`
shows up in the *energy* as `O(dt⁴)`. Do not read this as the integrator being fourth order:
the state is still second order, as §7.6.5 measures in real time.

**Ground state — exact to machine precision, and one control that matters more than the
result.** L = 8, δ = 1, half filling, `maxdim = 32`, `etol = 1e-12`:

| `grow_iters` | energy | E − E_exact | sweeps | maxbd | matvec |
|---|---|---|---|---|---|
| 0 | −1.7500000000 | 1.625e+00 | 2 | **1** | 14 |
| 1 | −3.3749325987 | −1.776e-15 | 6 | 16 | **425** |
| 2 | −3.3749325987 | −3.553e-15 | 6 | 16 | 598 |
| 3 | −3.3749325987 | −1.332e-15 | 6 | 16 | 631 |

The residuals are negative at 2–3 ulp of `|E| ≈ 3.4`, i.e. the variational bound of §7.7.2
holds to machine precision. Anything larger and negative would be a broken Rayleigh–Ritz.

**`grow_iters = 0` freezes at exactly the Néel product-state energy** — `(L−1)·(−0.25) = −1.75`
at `maxbd = 1` — and "converges" in two sweeps because it never leaves the start state. This is
the DMRG analogue of the standard-BUG χ=1 freeze recorded elsewhere: **the CBE growth pass is
doing all of the rank generation**, and the local eigensolve contributes none of it. It also
means `info.converged` is worthless as evidence on its own, which is now asserted in
`test_modes.jl` rather than left as a comment.

**`grow_iters = 1` suffices here and is 30% cheaper** than the default 2 (425 vs 598 matvecs,
identical energy to all digits). Not yet changed: one easy model is not grounds for moving a
default, and the cluster runs below are where a harder case would show whether extra passes
earn their cost. Same shape of reasoning as §7.1 for `growth`.

---

## 7.8 The method matrix — seven arms, and two new ones

| | arm | selection | status |
|---|---|---|---|
| **time** | `rsvd_cbe_bond_update` gate | sketched | verified, bit-identical across the §7.7 refactor |
| | `rsvd_cbe_bond_update` lubich | sketched | in the 1046-test suite |
| | `tdvp2` | — | baseline, **no discarded projectors** by design |
| | `1site_tdvp_cbe_rsvd` | sketched | **new**, verified |
| | `1site_tdvp_cbe` | exact | **new**, verified |
| **ground** | `rsvd_cbe_dmrg` | sketched | verified, machine precision |
| | `cbe_dmrg` | exact | **new**, verified, machine precision |

`benchmarks/method_matrix.jl` runs all of them against **exact diagonalisation** rather than
against another integrator: the Sz=0 sector at L=12 is C(12,6) = 924 states, so one dense
eigendecomposition supplies both the real-time propagator and the ground energy. Past
`MAXDENSE` it falls back to 2-site TDVP and *labels the column as consistency-only* — several
comparisons earlier in this document used an integrator as its own reference and so measured
agreement rather than correctness.

### 7.8.1 Plain CBE is a substitution, not a second implementation

`full_local_basis(frame, side)` returns the complete fused isometry **in the same frame layout**
`sector_graded_sketch` uses, so the whole exact/sketched distinction is one line:

```julia
probe(frame, side) = exact ? full_local_basis(frame, side) :
                     sector_graded_sketch(frame, side, npre; comp_ratio, rng)
```

With the full basis `skl(F)` *is* the exact `H·Θ`, so the preselection SVD returns the true
leading directions of `P⊥(HΘ)`. Preselection, ranking, `_trim_total`, the orthonormality guard
and the direct sum are untouched and never learn which probe they got. `exact = true` makes
`comp_ratio`, `npre`, `dover` and the RNG inert. One flag therefore produces four arms:
`cbe_dmrg`, `rsvd_cbe_dmrg`, `1site_tdvp_cbe`, `1site_tdvp_cbe_rsvd`.

**Ground state, L = 8, δ = 1, half filling** (`E₀ = -3.3749325987`):

| arm | E − E_exact | sweeps | maxbd | maxexp | matvec |
|---|---|---|---|---|---|
| `cbe_dmrg` (exact) | −4.885e-15 | 5 | 16 | 16 | 318 |
| `rsvd_cbe_dmrg` | −1.776e-15 | 6 | 16 | 16 | 425 |

**`maxexp` is IDENTICAL, which is a null result and was predicted wrongly.** The comparison was
run at `growth = 2.0`, where the budget saturates `room` at `d = 2` (§7.6.1), so both arms
expand to the whole local space and only the *ordering* of candidates differs — visible as one
fewer sweep, not as a wider basis. **Discriminating the two selections needs `growth < 2.0`**,
still to do. Third instance in this document of a comparison run with the knob under test
disengaged; see also §7.6.1 and §7.6.6.

Also: `matvec` **understates the exact arm**, which counts Krylov applications and not the
rank-4 formation inside its own preselection. Fewer matvecs there does not mean cheaper.

### 7.8.2 One-site TDVP + CBE, and why CBE is mandatory rather than optional

Structure follows QSMPSLib `TDVPSweepCBE1Si.m`: per bond, CBE → forward **one-site** at `+τ` →
backward **zero-site** at `−τ` → absorb. Two half-sweeps at `τ/2`. Note that 2-site TDVP's
backward substep is *one*-site where this one is *zero*-site.

**Two arms, and only two: `1site_tdvp_cbe` (exact CBE) and `1site_tdvp_cbe_rsvd` (sketched).**
Bare 1-site TDVP is *not* an arm and is not benchmarked anywhere — no bare stepper exists in
`src/`. It appears below only as the reason CBE is mandatory.

**Bare 1-site TDVP is fixed-rank.** Splitting a `(χ_l, d, χ_r)` site tensor cannot yield a bond
wider than `χ_r`, because the bond tensor must contract back into a neighbour whose leg is
`χ_r`. From a product state it stays a product state forever. CBE is what supplies the rank: it
widens the bond *before* the update by padding the neighbour with complement directions and the
current site with matching zeros, and the one-site evolution then fills them. That is the appeal
over 2-site TDVP — rank adaptivity while every solve stays one-site `O(dχ³)` or zero-site
`O(χ²)`, and the rank-4 block is never formed (provided the expansion is sketched).

The zero-site back-step is equally non-optional: evolving site `i` and then `i+1` counts the
shared bond twice, and the `−τ` zero-site step removes exactly that. Dropping it does not
degrade the order, it makes the scheme wrong.

### 7.8.2.1 The boundary bug, and the test design that hid it

**The first version omitted the closing one-site update at each pass's far end**, and the
consequence is worth spelling out because it is a whole-step error that a plausible test suite
reported as success.

Count the sites. The rightward loop runs over **bonds** `1..L-1` and evolves each bond's *left*
site, so it covers sites `1..L-1` and never site `L`. The leftward loop evolves each bond's
*right* site, covering `L..2` and never site `1`. So without a closing update per pass, **sites
1 and L receive `τ/2` where every interior site receives `τ`** — the boundary integrated at half
the rate of the bulk. `TDVPSweepCBE1Si.m` closes each pass with `Update1Si(…, 'vv')` precisely
for this; there is no back-step there because no shared bond extends past the chain end.

**What it measured, with the bug present:**

| test | result | verdict |
|---|---|---|
| L=8 XX, domain wall, vs analytic | 2.09e-06 | **passed — blind** |
| dt ratios, same setup | 3.92, 3.96 | **passed — blind** |
| L=12 Heisenberg, Néel, vs exact | **5.73e-02** vs tdvp2's 6.99e-09 | caught it |

**A domain wall cannot see this bug at all**: its edges are frozen, carry no amplitude, and
mis-integrating them changes nothing — so both the accuracy check and the order-in-`dt` check
passed cleanly with a broken scheme. A Néel start makes every site active and the error is
immediate and gross: seven orders, *while carrying more bond dimension than 2-site TDVP* (25 vs
20), which is the signature of a wrong scheme rather than a starved basis.

**Lesson worth generalising:** an initial state with frozen degrees of freedom silently removes
those degrees of freedom from every test built on it. The domain wall was chosen because it has
an analytic reference — and that convenience is exactly what made it blind. `test_tdvp1_cbe.jl`
now carries a Néel/`δ=1` case with explicit assertions on the *end sites*, compared against an
independent scheme.

This is the second time in this document a comparison passed for the wrong reason; see §7.6.6,
where a domain wall's logarithmic entanglement growth meant a χ-scan never engaged its cap. See
§7.8.2.2 — it happened twice more before the day was out.

**Bond growth remains demonstrated:** `maxbd = 10`, `maxexp = 12` from a bond-dimension-1 start
(L=8 XX, dt=0.02, 20 steps), i.e. the expansion is what supplies the rank.

Note this figure *did* move with the fix — it was `8` pre-fix — so the earlier claim that bond
growth is "unaffected by the bug" was wrong. The boundary sites were being under-evolved, which
suppressed the entanglement they generate, so the reachable rank was lower. What is true is the
weaker statement: growth was already non-trivial before the fix, so it never signalled the bug.

**Post-fix, confirmed.** L=12, Heisenberg, Néel, dt=0.02, t=0.4, against exact
diagonalisation of the 924-state `Sz=0` sector:

| arm | pre-fix | post-fix |
|---|---|---|
| tdvp2 | 6.989474e-09 | 6.989474e-09 |
| `1site_tdvp_cbe_rsvd` | **5.733650e-02** | **6.989480e-09** |
| `1site_tdvp_cbe` (exact) | **5.733650e-02** | **6.989867e-09** |
| `rsvd_cbe_bond_update` gate | 5.513000e-06 | 5.513000e-06 |
| `rsvd_cbe_bond_update` lubich | *crashed on a kwarg* | 5.257197e-06 |

Imaginary time, L=8, dt=0.05, against exact `E0 = -3.3749325987`. The bug was **inflating the
cooling residual**, which is why those numbers were discarded rather than reused: cooling
converges to whatever fixed point the scheme has, and a boundary integrated at half the bulk
rate moves that fixed point.

| β | pre-fix (buggy) | post-fix |
|---|---|---|
| 5 | 5.248e-02 | **1.220e-02** |
| 10 | 6.008e-03 | **2.478e-04** |
| 20 | — | **9.625e-08** |

Cross-checked against an independent scheme: cooled at identical β=10/dt=0.05, `1site_tdvp_cbe`
and 2-site TDVP agree on the residual to four digits (2.478e-04 both) with **bit-identical bond
dimensions** `[2, 4, 8, 16, 8, 4, 2]`. That is the strongest correctness evidence for this arm —
energy alone is a weak probe here (see below), so the state was compared too.

**The residual at β=20 is still flat in `dt`** (9.626e-08 at `dt=0.1` vs 9.625e-08 at
`dt=0.05`), which refutes the prediction previously recorded in `imaginary_time.jl` that the
`dt²` scaling would emerge once β was large enough. It does not, because the ground state is
stationary: an `O(dt²)` error in the *state* enters the *energy* at `O(dt⁴)` and stays under the
cooling term. Measuring a splitting error through the energy is the wrong instrument.

`test_tdvp1_cbe.jl` is 18/18 green, including the Néel boundary regression case.

### 7.8.2.2 The same blind spot, twice more, after writing it down

The §7.8.2.1 lesson was written, and then immediately violated twice. Recording both because
the pattern — not the individual slip — is the finding.

**(a) The method-matrix headline table cannot rank anything.** At L=12, t=0.4, `maxdim=48` the
state's widest bond is 20–22. The cap never engages, nothing truncates, and all three TDVP arms
land on a common floor — they agree to *six significant digits* (6.989474e-09 / 6.989480e-09 /
6.989867e-09). Six-digit agreement between a 2-site block evolve and a 1-site+0-site splitting
is not a coincidence; it means the test is saturated. That table is a **correctness** check — it
is what caught the boundary bug at 5.73e-02 — and it is *not* an accuracy comparison. It now
says so in its own printed output, and `benchmarks/method_matrix.jl` carries a second
**fixed-rank** table (hard caps χ ∈ {4,8,16} at t=2.0) for the actual ranking.

**(b) The exact-vs-sketched A/B, written one step after (a), repeated (a).** Set up at L=10,
t=0.4, `MAXDIM=64` where `maxbd` is 20–27. Result: error flat at ~8.1e-09 for every `growth`
from 1.05 to 2.0, while `maxexp` moved 27→31. The knob was demonstrably working and the metric
could not respond, because a state that is already representable cannot be damaged by a
narrower candidate space. Re-run at `MAXDIM=8`, `t=2.0`.

**The rule, stated operationally.** `growth`, `maxdim`, `dex` and `comp_ratio` are all
allocation knobs: they decide *what to keep when you cannot keep everything*. Every one of them
is inert unless the run is actually rank-starved. So **before reading an error column, check
that `maxbd` reached the cap** — if it did not, the row is measuring the integrator's floor, not
the knob. Both benchmark scripts now flag a non-binding row explicitly rather than leaving that
inference to the reader.

Four instances now share one shape: a configuration chosen for convenience (an analytic
reference, a short run, a round cap) removed the very degree of freedom under test, and the
green result was indistinguishable from a meaningful one.

### 7.8.3 SU(2) — the operator algebra changes, the environments do not

**`:Sp`, `:Sz`, `:Sm` do not exist under `:SU2`**, and cannot: none is an SU(2) tensor by
itself. Telum exposes `:S` and `:I` only, so the coupling is ONE irreducible `S·S` term with an
`S = 1` op-leg in place of three. Since `q.S` is rank-3 with an op-leg — **measured: the same
shape as `:U1`'s `Sp`** — `push_left_channels`, `apply_h_two_site`, `zero_site_h` and
`one_site_h` all take the single-term list unmodified. The expensive part was already done.

Three consequences that are physics, not missing features:

* **Only δ = 1 exists.** Anisotropy singles out the z axis and breaks SU(2) down to U(1), so
  `heisenberg_su2_chain` has no `delta` argument and the SU(2) gate refuses `delta ≠ 1`.
* **Néel and domain-wall states are unrepresentable at any bond dimension** — they carry weight
  across many total-`S` sectors. `dimer_state(L)` (a product of nearest-neighbour singlets,
  `S = 0`, norm 1) is the representable start.
* **`magnetisation` is useless** — it needs `Sz`, and is identically 0 for a singlet anyway.
  Use the energy or a bond correlator.

**`room` had to be fixed first, and it would have failed silently-ish.** `cbe_expand` computed
`room_l = leg_dim(U0,1) * leg_dim(U0,2) - r`, and that product is only correct when every pair
of input sectors fuses to exactly one output — i.e. for abelian charges. Under SU(2) a spin-½
site is *one* multiplet, so it reads `χ·1 − r ≈ 0` and CBE would decline to expand, freezing at
χ=1 with the same signature as `grow_iters = 0`. Now `sum(d for (_,d) in reachable_sectors(...))`,
which goes through `fusion_basis → getIdentity + svd` and so applies whatever symmetry the
tensors carry. Reduces exactly to the old product for abelian charges, so it is not a special
case. Known cost: duplicates a `fusion_basis` the probe rebuilds a few lines later.

**Measured, L = 8, δ = 1, half filling:**

| symmetry | ground energy | E − E_exact | sweeps | maxbd | matvec |
|---|---|---|---|---|---|
| `:SU2` | −3.3749325987 | 4.441e-16 | 5 | **6** multiplets | **126** |
| `:U1` | −3.3749325987 | −1.776e-15 | 6 | 16 states | 425 |

Same energy to 4e-16, 3.4× fewer matvecs. **The overall constant needed no correction** —
`XXZTerm(q.S, q.S', J)` is exactly `J·S·S`, unlike the `:none` gate branch which carries a
measured `−1`. The ratio `E_su2/E₀ = 1.0000000000` is the test that pins it.

Two caveats on reading that table. `maxbd` counts **multiplets** under SU(2) and **states**
under U(1), so the integers are not comparable except as "the reduced basis is smaller". And
the two runs start from *different* states (Néel is unrepresentable under SU(2)), so it is the
same converged state reached from different starts — the energy agreement is a clean result,
the cost ratio is indicative rather than a controlled A/B.

---

## 8. Gotchas that cost real time

* **`contract` verifies itags, not just arrow directions.** Two tensors whose contracted legs
  span the same space but carry different tags are REJECTED. Combined with the rule that a
  single tensor may not carry a duplicate non-empty itag, this is a squeeze: in
  `tdvp1_cbe.jl` the bond tensor `C` has legs `(new, bond_ex)` which are different spaces (so
  they must be tagged differently) while `apply_zero_site` needs them to match the environment
  built from the site tensors (so they must be tagged the same). The only way through is to
  keep the SVD's own tag through the zero-site step and rename BOTH ends of the bond together
  afterwards — which is what `cbe_lubich_sweep` already does with `U_ex`/`V_ex`. Cost two
  failed runs.

* **The RANK of what comes out of an `svd` decides whether a tag line is safe.** `tdvp2` writes
  `setitag(res.S * res.Vd, 1, tag)` safely because its SVD splits a rank-4 `Theta`, so `Vd` is
  `(new, site_r, link_r)` and contains no bond leg. The identical line in a 1-site sweep splits
  a rank-3 site tensor, `Vd` is `(new, bond_ex)`, and the bond tag collides. Copying a line
  between the two is not safe.

* **`product_state` under `:SU2` returned a plausible-looking wrong answer** rather than
  failing: `SECTOR_UP` is `((1,),)` and the SU(2) spin-½ irrep is *also* `((1,),)`, so `:up`
  sites matched the `getsub` filter while `:down` sites (`((-1,),)`, nonexistent under SU(2))
  selected nothing. It reported `bond_dims = [1,1,1]`. Now guarded with a throw. The general
  shape: a sector-label coincidence between symmetries turns a type error into a silent one.

* **A probe that fails for the wrong reason proves nothing.** The SU(2) probe's `xxz_chain`
  test reported `UndefVarError: xxz_chain not defined in Main` — the script had imported only
  `BondUpdateBUG`. That is not evidence the function refuses under SU(2); the real check
  (`FieldError: no field Sp, available: I, S`) came later.

* **JIT dominates short local runs, so a timeout is not evidence of a hang.** First-call
  compilation of the tensor stack costs ~3 min on the dev box (Néel probe: 211 s for run 1
  against 14.9 s for run 2), and `Test` buffers output until a testset closes, so a suite can
  print nothing for 28 minutes and still be healthy. Diagnose with a longer timeout before
  concluding a deadlock.

* **`bond_dims` counts MULTIPLETS, not states, so χ means different things under different
  symmetries.** `leg_dim` sums sector *multiplicities* (`sum(d for (_, d) in t.spaces[l])`).
  Under U(1) each sector is one state and the two coincide; under SU(2) a spin-`S` multiplet
  stands for `2S+1` states, and `maxdim`/`Nkeep` cap multiplets. So an SU(2) run at χ=8 is
  holding materially more physical information than a U(1) run at χ=8, and **the fixed-rank
  table in `method_matrix.jl` must not be read across symmetries** without converting to state
  counts first — SU(2) would look better at equal χ for free. The dimer state is the extreme
  illustration: every link carries one multiplet, `bond_dims` reports all ones, and the state is
  genuinely entangled.

* **An allocation knob is inert unless the run is rank-starved, so check `maxbd` before reading
  the error.** `growth`, `maxdim`, `dex`, `comp_ratio` all answer "what do you keep when you
  cannot keep everything". If `maxbd` never reached the cap, nothing was discarded and the error
  column is the integrator's floor, identical across arms — see §7.8.2.2, where this produced
  six-digit agreement between three different schemes and a `growth` scan that was flat to four
  digits while the knob visibly moved `maxexp`. Both benchmark scripts now print a
  "CAP DID NOT BIND" marker on such rows. Cost four wasted comparisons across two days, twice
  *after* the lesson was written down.

* **Harness scripts need `using Telum` for `to_concrete`** — hit three separate times
  (`cbe_project_first.jl`, an ad-hoc probe, and `benchmarks/method_matrix.jl`). It surfaces late
  and looks like a library bug: in `method_matrix.jl` the real-time section passed cleanly and
  only the imaginary-time section died, because `to_concrete` is reached solely on the
  `normalise = true` path. `Hint: a global variable of this name also exists in Telum` is the
  tell.

* **A background job started before a fix landed will happily report post-fix-looking numbers.**
  The imaginary-time β scan was launched at 11:21, the boundary fix landed at 11:55, and the job
  was still running at 12:06 — its output would have read as a fresh measurement. Julia loads
  the package once at startup and never sees later edits. Check a job's start time against
  `git`/file mtimes before trusting its output, and kill anything that predates the change.

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
