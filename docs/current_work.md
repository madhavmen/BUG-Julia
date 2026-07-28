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

### 7.1 The default growth schedule is now the accuracy limiter — highest value
`budget = max(ceil(1.1*dmax) - r, 1)` (`cbe.jl`, from `RSVDpreBE0SiQS.m:66-77`) is a ~10%
schedule. Measured (`benchmarks/cbe_gate_budget.jl`, XX L=10, dt=0.02, t=0.5, maxdim=64):

```
bond_update_bug!      err 5.81e-07   maxbond 14   157.0s
cbe_gate (schedule)   err 1.43e-04   maxbond 14    22.4s   err_fnl 6.2e-01
cbe_gate (dex=1)      err 2.28e-03   maxbond 14     3.8s   err_fnl 6.7e-01
cbe_gate (dex=2)      err 1.21e-03   maxbond 12     4.2s   err_fnl 1.1e-02
cbe_gate (dex=4)      err 3.54e-04   maxbond 15     4.6s   err_fnl 0
cbe_gate (dex=8)      err 1.17e-06   maxbond 13     4.7s   err_fnl 0
cbe_gate (dex=16)     err 1.17e-06   maxbond 13     4.5s   err_fnl 0
cbe_gate (dex=64)     err 1.17e-06   maxbond 13     4.4s   err_fnl 0
```

`err_fnl = 0.62` on the default means the weight ranking is **discarding 62% of the candidate
spectrum for want of budget**. `dex = 8` saturates: 16 and 64 are identical, and cost the same
wall time as `dex = 1` because the sweep dominates. The default was left alone only because the
MPO path shares it and changing it mid-investigation would have confounded the comparison.
**Decide the new default and re-measure both paths.**

### 7.2 Re-run the L=18 XX campaign and the 2-site TDVP comparison
`benchmarks/xx_l18_campaign.jl` + `plot_xx_l18_campaign.py` are ready. Previously **voided** by
the user because the scheme was wrong; the scheme is now validated on both paths, so these
measurements mean something. Note `profile_err` must normalise (`site_expval` does not divide
by the norm, and both methods shed norm under truncation).

### 7.3 Substantiate the memory claim
`peak_elements` is wired into both `CBEBugInfo` and `TDVP2Info` at a matched definition (state +
transients alive). The design thesis is that CBE-BUG needs a smaller bond dimension than 2-site
TDVP at matched accuracy because it is a 1-site-cost solver that still sees other symmetry
sectors. §1 and §7.1 support it on rank and wall time (13 vs 14, 33× faster); the **peak working
set** figure has not been reported yet.

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

**Standing constraints.** Run everything via `sbatch` on compute nodes — never Julia on the
login node (4 cores, 11 GB, `/home` on NFS). Never write `--time`; inherit the partition max
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
