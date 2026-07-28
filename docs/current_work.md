# RSVD-CBE for BUG — session handoff (2026-07-27)

Branch `feature/cbe-bug`, pushed to `origin`. Read this before touching
`src/RSVDCBEBondUpdate/`; it records what was fixed, what was **ruled out** (so it is not
re-investigated), and what is left.

---

## 1. Headline

**RSVD-CBE now works, in BOTH integrator paths.** The long-standing rank stall — bond
dimensions frozen at `[1,1,2,4,2,1,1]` on L=8 while 2-site TDVP reached `[2,4,6,8,6,4,2]` —
was **two defects in the CBE selection**, not in the sweep. Once fixed, the gate-based and the
Lubich/MPO sweeps both reach the same profile and both beat the validated `bond_update_bug!`.

```
XX domain wall, L=8, dt=0.02, t=0.5, maxdim=32, dex=8   (benchmarks/lubich_vs_gate_rank.jl)

bond_update_bug!      err 2.09e-04   profile [2,4,7,12,8,4,2]   172.7s
cbe_gate_bug! (gate)  err 1.37e-06   profile [2,4,8,12,8,4,2]    31.1s
cbe_bug!   (Lubich)   err 8.44e-06   profile [2,4,8,12,8,4,2]    32.2s
```

(That run set `dex = 8` by hand. Since §7.1 the DEFAULT reaches the same regime, so
`lubich_vs_gate_rank.jl`'s explicit `DEX = 8` is now redundant rather than load-bearing.)

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
cbe_expand_gate(f, gate; kwargs...)                    # gate wrapper (cbe_gate.jl)
```

Both existing call signatures are unchanged.

### New file: `src/RSVDCBEBondUpdate/cbe_gate.jl`

| function | role |
|---|---|
| `sketch_gate_left/right(f, gate, Om)` | `(gate Θ) Ω†` **without ever forming the rank-4 `HΘ`** — `Ω` is folded into the gate first; largest intermediate is `O(d²·g·r)` |
| `cbe_expand_gate(f, gate; …)` | the shared selection with gate sketch closures |
| `cbe_gate_bond_update(f, gate, tau; …)` | drop-in for `BondUpdateBUG.kls_bond_update`; same return fields |
| `cbe_gate_sweep!` / `cbe_gate_bug!` | the validated `parity_sweep!` / Strang driver, CBE basis update |

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
Lives in `tests/RSVDCBEBondUpdate/test_cbe_gate.jl`; the ~1 min regression probe is
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
`cbe_gate_bug!`. Sits inside the odd/even Trotter sweep, which is exact within a parity group.
103/103 tests, including: sketch pinned against a formed-`HΘ` reference; isometry +
containment + losslessness; **exactness at full rank against `expv` on the block directly**;
`tau=0` is the identity; residual→machine-zero at full budget; rank leaves 1 on step 1 and
keeps growing; U(1) charge conserved; XX profile vs `xx_free_fermion_sz`.

### Lubich/Sulz BUG — **working, unchanged, benefits from the shared fixes**
`cbe_bug!`. The sweep is the MPS specialisation of Sulz's TTN BUG (K-sweep, L-sweep, seed from
the sweep carries with no M/N overlap matrices, one centre Galerkin, then truncate), matching
`exploratory_bug/src/BUG/discarded_bug.jl:16-31`. It needed **no change of its own** —
`centre rank` now climbs `2,3,5,6,6,6,8,8,9,10,…,12` with `err_fnl = 0`.

> **Correction to an intermediate conclusion in this session:** it was stated at one point that
> "the selection is sound, so the fault in `cbe_bug.jl` is the sweep." The first half held, the
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

`benchmarks/cbe_gate_budget.jl` now measures the schedule A/B (`growth = 1.1` vs `2.0`) on
**both** paths alongside the absolute-`dex` curve, since `cbe_bug!` shares `cbe_expand` and
must be re-measured on the new default rather than assumed to follow.

Measured (`benchmarks/cbe_gate_budget.jl`, XX L=10, dt=0.02, t=0.5, maxdim=64; run on a
laptop, one thread, so wall times are ~1.5x the cluster's but internally comparable):

```
bond_update_bug!        err 5.814e-07   maxbond 14   aug 17   141.9s

cbe_gate (growth=1.1)   err 2.347e-04   maxbond 14   aug 16    24.4s   err_fnl 5.26e-01
cbe_gate (growth=2.0)   err 1.165e-06   maxbond 13   aug 19     3.9s   err_fnl 0
cbe_bug  (growth=1.1)   err 6.972e-06   maxbond  4   aug  4    19.7s   err_fnl 0
cbe_bug  (growth=2.0)   err 8.445e-06   maxbond 14   aug 22     3.6s   err_fnl 3.5e-15

cbe_gate (dex=1)        err 2.148e-03   maxbond 14   aug 15     4.1s   err_fnl 6.71e-01
cbe_gate (dex=2)        err 1.209e-03   maxbond 12   aug 14     4.8s   err_fnl 1.90e-03
cbe_gate (dex=4)        err 5.240e-04   maxbond 14   aug 18     6.0s   err_fnl 1.27e-04
cbe_gate (dex=8)        err 1.165e-06   maxbond 13   aug 19     4.1s   err_fnl 0
cbe_gate (dex=16)       err 1.165e-06   maxbond 13   aug 19     5.9s   err_fnl 0
cbe_gate (dex=64)       err 1.165e-06   maxbond 13   aug 19    13.2s   err_fnl 0
cbe_bug  (dex=8)        err 8.445e-06   maxbond 13   aug 21    16.5s   err_fnl 0
```

What it says:

* **The new default lands exactly on the saturated end of the curve.** `growth = 2.0` gives
  `1.165e-06`, the same value to all printed digits as `dex = 8, 16, 64`, with `err_fnl = 0`.
  Same on the MPO path: `growth = 2.0` and `dex = 8` both give `8.445e-06`. The schedule is
  no longer the limiter on either path.
* `err_fnl = 0.53` on the old default was the weight ranking **discarding half the candidate
  spectrum for want of budget** — the cap, not the selection, was the accuracy limiter.
* **The generous default is FASTER, not merely affordable**: gate `24.4s → 3.9s`, MPO
  `19.7s → 3.6s`. That was not the expectation (the sketch does strictly more work) and it is
  not the sketch: `dex = 64` costs 13.2s, so cost *does* rise with budget. The starved runs
  are paying somewhere else — most plausibly the 0-site Krylov, which has `maxiter = 30` and
  a much worse subspace to converge in. **Not yet confirmed; record the Krylov dimension per
  step before believing it.**
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

### 7.4 Deferred by the user: fixed-rank vs rank-adaptive
For that comparison, reinstate the bound as *max rank of the new orthonormal K/L matrices ≤ 2r*.
`sulz_cap = true` is already wired in the **global** `2·rmax` form. Explicitly "for later".

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
* **`find` on this cluster is `bfs`** — `pgrep -f "^find …"` does not match a running tree walk and
  a kill silently does nothing. Match on the command line and *verify*. And never walk the NFS
  home: `/home` is `10.1.0.110:/volume1/slurm-home`.

---

## 9. How to run things

```bash
# fast: the gate path only (~5 min)
sbatch --job-name=cbegate --partition=short scripts/run_julia.sbatch \
       tests/RSVDCBEBondUpdate/test_cbe_gate.jl

# the CBE module suite (~10 min) and the validated suite (~7 min)
sbatch --partition=main scripts/run_julia.sbatch tests/RSVDCBEBondUpdate/runtests.jl
sbatch --partition=main scripts/run_julia.sbatch tests/runtests.jl

# the two measurements in this document
sbatch --partition=short scripts/run_julia.sbatch benchmarks/cbe_gate_budget.jl
sbatch --partition=short scripts/run_julia.sbatch benchmarks/lubich_vs_gate_rank.jl

# ~1 min regression probe for "refused with room to spare"
sbatch --partition=short scripts/run_julia.sbatch benchmarks/probe_edge_refusal.jl
```

**Running it off the cluster.** When the cluster is unreachable (VPN down — `ssh login-node`
to `10.1.0.120` just times out) the whole suite runs on a laptop, and the L=8/L=10
measurements in this document are small enough to be worth doing there:

```bash
# once: a project outside the repo, so instantiating never dirties the Manifest
cp -r BUG-Julia /tmp/bugjl && cd /tmp/bugjl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. deps/build.jl        # the Telum :none SVD patch, NOT `Pkg.build`
julia --project=. --threads=1 tests/RSVDCBEBondUpdate/test_cbe_gate.jl
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
