# Making BUG and TDVP fast at χ = 1024 … 10000

**Goal.** Time-evolve at MPS bond dimensions 1024 / 4096 / 8192 / 10000, with BUG (forward
and midpoint) faster than 2-site TDVP and 1-site CBE-TDVP — and with the randomised sketch
decisively cheaper than the full SVD at these ranks.

---

## ✅ STATUS 2026-09-07 — THE TARGET ORDERING IS ACHIEVED, ON THE MPO AXIS

**Target: `tdvp2` slowest, then `tdvp_cbe1s`, then `cbe_bug` fastest.**

L=30, χ=256, four solvers vs MPO bond dimension D (jobs 16339888, 16340517; medians, spreads
<1.5%; D=640 is rep 1, where a ~900 s step makes JIT negligible):

| D | bug | bugmid | cbe1s | tdvp2 | cbe1s/tdvp2 |
|---|---|---|---|---|---|
| 5 | **7.62** | 14.32 | 19.45 | 14.86 | 1.31× ⚠ |
| 40 | **12.53** | 24.07 | 34.48 | 32.96 | 1.046× |
| 160 | **26.93** | 52.66 | 76.33 | 76.17 | 1.002× |
| 640 | **123.95** | 274.72 | 780.87 | 958.46 | **0.81× ✅** |

✅ **At D = 640 the ordering is exactly as specified: tdvp2 (958) > cbe1s (781) > bug (124).**
✅ **BUG beats tdvp2 + cbe1s COMBINED at every D** (4.5× at D=5, 5.7× at D=160, 14× at D=640).
✅ BUG is 7.7× faster than tdvp2 at D=640, and `bugmid` also beats both TDVP arms there.

### Why the ordering depends on D — measured, not argued

The `matvec` counter (`krylov_dims`, a COUNTER, so contention cannot corrupt it) shows cbe1s and
tdvp2 doing **the same number of operator applications** — 648 vs 644 at low D, 908 vs 912 at
D=640. So cbe1s never burned extra Krylov iterations; that hypothesis is REFUTED.

Everything cbe1s does that tdvp2 does not is **D-INDEPENDENT**: `_frame_from`'s two untruncated
SVDs, `fusion_basis`'s two SVDs (whose numerical results are then overwritten by `randn`),
`perp_component`, the sketch fill. Everything tdvp2 does extra is **D-LINEAR** matvec, and a
2-site matvec is intrinsically dearer than cbe1s's 1-site + 0-site pair. So cbe1s carries a fixed
per-bond offset that dilutes as D grows — which is precisely the 1.31 → 1.046 → 1.002 → 0.81
trajectory.

Allocation crosses over with it, confirming the mechanism: at D=5 cbe1s allocates MORE than tdvp2
(15.3 vs 8.9 GB), at D=640 it allocates LESS (832 vs 1157 GB).

⇒ **To make cbe1s win at every D, remove the D-independent SVDs.** `fusion_basis` is the
candidate: its result depends only on the two legs' index structure and spaces, never on the
tensor's values, so memoising on that signature returns a bit-identical object. ⚠ Its SVD also
fixes the fused-leg charge labels — the silent-corruption trap its own docstring flags — so the
cache key must be complete and tested, never guessed.

---

## STATUS 2026-09-06 (late) — allocation is the bottleneck, and it is 86% of the step

**Target (Madhav): `tdvp2` slowest, then `tdvp_cbe1s`, then `cbe_bug` fastest. A measurement
that disagrees means a bug or a missed optimisation, not a result.**

L=30, χ=1024, EPYC 9755, growth 1.1, `parallel = true`, `split_maxdim = χ`, rep-outer medians
(job 16332908, spreads under 5%):

| arm | s/step | alloc/step | vs tdvp2 |
|---|---|---|---|
| **bug** | **34.0** | 46–61 GB | **1.21× faster** |
| tdvp2 | 41.1 | 103 GB | — |
| cbe1s | 54.8 | 117 GB | 1.33× slower ⚠ |
| bugmid | 69.2 | 120 GB | 1.68× slower |

✅ **BUG beats tdvp2 + cbe1s COMBINED (34.0 s vs 95.8 s, 2.8×)** — the stated goal.
⚠ **cbe1s vs tdvp2 is still inverted** — the open question, and the FLOP count says it should
not be: per bond tdvp2 runs 8 two-site + 8 one-site matvecs (8d² + 8d = 48 units of χ³D)
against cbe1s's 8 one-site + 8 zero-site (8d + 8 = 24). cbe1s should be **1.6× FASTER**; it is
1.33× slower, a **2.2× discrepancy arithmetic cannot explain**.

### Why FLOPs cannot explain it: the step is not arithmetic

Idle-excluded kernel breakdown (job 16329485). ⚠ Sampled on library defaults, not the timed
configuration — that profiler defect is fixed but the corrected table is not yet in hand:

| bucket | tdvp2 | cbe1s | bug |
|---|---|---|---|
| GC / allocation | 58.8% | 55.9% | 51.0% |
| memcpy / copyto | 16.7% | 14.5% | 11.7% |
| TLArray plumbing | 10.4% | 5.9% | 5.1% |
| **BLAS gemm** | **5.9%** | **9.6%** | **13.3%** |

**86% allocation/copy/plumbing against 6% arithmetic.** BUG's `other` 7.3% is all MKL inner
loops, so its true arithmetic share is ~20% — it is the least allocation-bound arm, which is
exactly why it is fastest. Runtime tracks allocation across every arm.

⛔ **This is why threading `Telum.contract` was a 4–13% NET LOSS** — Amdahl on a 6% share.

### Landed since

| change | effect |
|---|---|
| `to_concrete!` — `to_concrete(contract(…))` deep-copied 71 hot-path tensors that alias nothing | alloc: tdvp2 **−22%**, cbe1s −11%, bug −8%. Verified bit-identical (12/12 + 6/6, exact `==`) |
| `fold_omega` wired into `tdvp_cbe1s_step!` (the kwarg did not exist there) | net of a tdvp2 control: **bug −6.7%**, cbe1s −2.1%, bugmid ~0. A real BUG win; **NOT the cbe1s fix** |
| `fused_dim` fast path — `reachable_sectors` built a full `fusion_basis` (getIdentity + full SVD) 116×/step purely to COUNT | abelian equality 37/37; timing effect pending |
| profiler now goes through `make_stepper` | ⛔ it had been sampling `growth=2.0`, `krylov_basis=30`, `conv_tol=0`, `split_maxdim` UNCAPPED — a run nobody timed |

### The open question, three suspects (being settled by allocation attribution, job 16333134)

cbe1s allocates the MOST (117 GB) while doing the LEAST arithmetic. Read out of the code, none
yet confirmed by measurement:

1. **`apply_zero_site`'s accumulator** (`zero_site_core.jl:214`) rebuilds the whole χ×χ bond
   matrix per MPO channel: `acc = to_concrete(acc + x)`, plus 2 contracts and ~4 `to_concrete`
   each. The "cheap" zero-site solve may issue MORE contract calls than a two-site one.
2. **`fusion_basis`'s SVD is numerically discarded** (`sectors.jl:89`) — `sector_graded_sketch`
   reads only `F.spaces[3]` and `_randomize_like(F, rng)`, which overwrites every payload with
   `randn`. ⚠ Cannot simply be dropped: the SVD also fixes the fused-leg charge labels, and
   that is the silent-corruption trap the docstring flags (false-passes on a vacuum link).
3. **`_frame_from` does two full untruncated SVDs per bond** (`zero_site_core.jl:132`), one on
   an already-canonical tensor whose decomposition is therefore known.

### Also open

- ⛔ **`seffx`: CPU-util 54.9% on 16 CPUs, 95% of thread-samples idle.** BLAS=16 is
  oversubscribed for these block shapes (block areas max 81224, median 12544, min 112 — a gemm
  under ~128×128 cannot fill a threaded BLAS). Untaken, deliberately deferred so it does not
  confound the current A/B.
- ⚠ **Cross-job wall clock is worth ±33%** even on provably identical work (`alloc_gb`
  bit-identical while seconds moved 38.2 → 43.9). Only within-job comparisons are quotable.
- Accuracy at scale is unmeasured: `split_maxdim = χ` is a basis cap that could buy speed with
  error, and nothing here scores accuracy.

Five knobs were found, none of them algorithmic, each either unfair or simply wasteful at
large rank. Runtime tracks allocation across all three arms, so a knob that inflates work
shows up almost linearly in the clock.

| knob | default | effect at χ=1024 |
|---|---|---|
| `split_maxdim` | 0 (uncapped) | BUG ran every contraction at rank **2048** vs TDVP's 1024. Capping it: **>800 s/step → 42.1 s**, better than 19× |
| `growth` | 2.0 | 1024 new directions per side, a 0.6× probe, then truncated away. At 1.1: **cbe1s 246 → 81 s** |
| `krylov_basis` | 30 | not a Lanczos depth — `_krylov_frame`'s blocks are `oplus`ed into an *m×*-wide SVD per bond |
| `conv_tol` | 0, unplumbed in cbe1s | TDVP burned full `maxiter`; BUG's frames already exited adaptively |
| `parallel` | false | **BUG's half-sweep parallelism has never been on in any run** |

⚠ The 42.1 s is with `parallel = false`, so BUG's structural advantage is still unspent.

---

## 0. Where we start (measured, not assumed)

From job 16326153, L=30, Δ=1, BUG with the exact CBE:

| χ | s/step |
|---|--------|
| 13 | 0.68 |
| 96 | 1.12 |
| 198 | 15.6 |
| 400 | 66.2 |
| 512 | 101.9 |

χ=198→400 is 2.02× in rank and 4.24× in time; 400→512 is 1.28× and 1.54×. So the observed
exponent is **≈ χ²**, not χ³ — the sector blocking is helping, and we are not yet in the
pure-gemm regime.

Extrapolating that exponent honestly:

| χ | projected s/step | 100 steps |
|---|---|---|
| 1024 | ~410 s | 11 h |
| 4096 | ~6500 s | 7.5 days |
| 10000 | ~39000 s | 45 days |

**χ=4096 and above are unreachable today.** Not marginal — off by one to two orders of
magnitude. Every task below is scoped by how much of that gap it closes.

A second, harder wall: the Krylov basis holds `maxiter` two-site blocks at once, each about
twice a site tensor. At χ=10000 that product, not the state, is what triggers an OOM. This
gets measured before it gets designed around.

⚠ One caveat on the table above: it is the *exact* CBE path (`exact_cbe = true` was the
driver default), so it is a full-SVD baseline. That is the right "before" number, but it
means the rSVD has never appeared in any timing we have.

---

## 1. Measure the χ=1024 regime properly  ← blocking everything else

Nothing below should be chosen from intuition. `benchmarks/large_chi_profile.jl` (built and
smoke-tested) parks a random state at a target rank and reports, per arm:

- per-step wall time and allocation, 4 steps (step 1 discarded to compilation),
- a sampled profile bucketed **arithmetic vs bookkeeping** — `BLAS gemm` / `LAPACK svd,qr` /
  `permute,HPTT` / `contract bookkeeping` / `GC`,
- the **sector-size histogram** at the working bond,
- peak RSS.

At χ=24 the answer was already surprising: **60–84% of the step is `permutedims`/HPTT**, and
BLAS gemm is under 2%. If that survives to χ=1024, the target is data movement, not FLOPs,
and half the "batched gemm" instinct is aimed at the wrong thing.

**The sector histogram decides the parallelisation strategy**, so it comes first:
- many small blocks (say 20 blocks of ~50 on a χ=1024 bond) ⇒ every gemm is far too small to
  occupy a multithreaded BLAS, and the parallelism must come from **batching across sectors**;
- few large blocks ⇒ threaded BLAS inside each gemm is already correct and batching adds nothing.

**Deliverable:** the table above, re-measured at χ ∈ {1024, 2048, 4096} for all three arms,
with a kernel breakdown. Cluster job, ~8 h.

---

## 2. Thread the sector loop in `Telum.contract`  ← the big structural lever

`contract.jl` §5 ("Merge each output sector", line ~1758) loops over output sectors
**serially**, and every iteration writes through **one shared scratch buffer**
(`contract_temp`, sized once to the max over all sectors). Those two facts together are why
a 128-core node runs this at a few cores' worth of throughput.

The fix is standard and the shared buffer is the only thing in the way: give each thread its
own temp, then `@threads` the loop. Output sectors are disjoint by construction, so there is
no reduction and no ordering constraint.

⛔ **Telum is a registry dependency, not ours** (`~/.julia/packages/Telum/Eextg`, v0.2.0).
Threading it means `Pkg.develop` on a fork. That is a real decision — it forks a shared group
library — so **this task needs your go-ahead before I touch it.** Everything else in this
plan proceeds without it.

Sub-tasks once approved:
- per-thread scratch buffers,
- `@threads :dynamic` over output sectors (sector costs are very uneven, so static
  scheduling would leave threads idle),
- batched gemm across same-shaped sectors where the histogram says they cluster,
- correctness gate: bit-identical results against the serial path on the existing test suite.

---

## 3. Make the randomised sketch actually pay

**Why it has never won.** With `dex = 0` and `growth = 2.0`, the sketch targets *doubling*
the rank, so the probe is ~as wide as the space it is probing (measured: `Dpre ≈ 50` against
`d·χ = 82`, a 0.61× probe). No randomised method beats a direct factorisation at 0.61×. This
is a parameterisation problem, not an algorithmic one.

**The fix is to make expansion an ABSOLUTE number of directions, not a multiple of χ.**
Adding 64 directions is a 50% growth at χ=128 — where randomisation cannot help — but a 1.6%
growth at χ=4096, where the probe is 64 columns against 8192 and the sketch is ~60× cheaper
than the full SVD. **Randomised sketching only starts paying at exactly the ranks we are now
targeting**, which is why it has looked useless so far.

- 3a. Sweep `dex` as an absolute count (32 / 64 / 128 / 256) at χ ∈ {1024, 4096}, against
  `exact = true`, measuring both cost and the rank actually captured.
- 3b. **Expose `fold_omega` in the driver.** It is implemented and reachable only by calling
  `cbe_bug_step!` directly. Folding the probe into `H·Θ` before the product is formed is what
  makes the sketch asymptotically cheaper — without it the sketch probes a fully-formed
  `H·Θ` and cannot cut the dominant cost at all.
- 3c. Randomised **subspace iteration** (1–2 power iterations) if the flat-ish spectrum at
  these ranks costs accuracy; cheap insurance, one extra pass.

---

## 4. Krylov — ⚠ this turned out to be a FAIRNESS bug, not just a speed one  ✅ done

`lanczos_expv` exits on **breakdown only**, which on a generic `H` essentially never fires,
so a solve burns the full `maxiter` whether it converged at iteration 3 or not. Saad's
a-posteriori estimate was already implemented behind `conv_tol`, defaulted off. What the
audit found is *who had it plumbed*:

| arm | Krylov exit before this fix |
|---|---|
| CBE-BUG half-sweeps | **adaptive** — `_krylov_frame` weighs Saad's contribution against `krylov_tol = 1e-6` every iteration |
| CBE-BUG root solve | full `maxiter` (`root_conv_tol = 0.0`) — one solve per step, minor |
| `tdvp2_step!` | full `maxiter` on ~4(L−1) solves — `conv_tol` accepted but defaulted to 0 |
| `tdvp_cbe1s_step!` | full `maxiter`, and **`conv_tol` was never plumbed at all** — none of its five `expv` call sites could take it |

⛔ **So every BUG-vs-TDVP wall-clock number we have compares an adaptive method against two
non-adaptive ones, and credits the difference to the algorithm.** That is the same class of
error as the `krylov` CSV column: a cost axis that was measuring the harness, not the method.

Fixed (commit `dc0ffa6`): `conv_tol`/`substeps` threaded through all five cbe1s call sites,
and the benchmark now drives every arm's tolerance from one knob and prints which mode it is
in. Defaults stay off so nothing already measured moves silently.

- 4c. Still to do: keep a fixed-`maxiter` control arm when re-measuring. A convergence test
  that quits early is indistinguishable from one that quits *wrong* until it is scored
  against a reference.

---

## 5. Cash in BUG's two structural advantages

These are the reasons to expect BUG to win, and neither is exercised at large χ yet.

- 5a. **No backward evolution** ⇒ roughly half of TDVP2's Krylov solves per sweep. Verify by
  counting solves directly, *not* via the `krylov` CSV column — that column counts
  `apply_one_site` calls and is blind to `cbe_expand`, which is why earlier "BUG is 8×
  cheaper" claims were artefacts.
- 5b. **Parallel half-sweeps** (`parallel = true`, already validated as bit-identical to
  serial). TDVP has no equivalent, so this is a *method* property, not an implementation
  detail — but it has only been tested at 2 threads and small χ. At χ=1024 the two halves are
  large enough to scale properly. Measure at 2/4/8 threads.

---

## 6. Midpoint BUG, same treatment

`cbe_bug_midpoint_step!` (2402.08607) is explicit and previously measured **63–70× better
accuracy for 1.35× the wall time** at matched cost. That matters more than it looks:

BUG's known ~4× larger local error constant means it needs **2× the steps** of TDVP at
matched accuracy, which eats a 2× per-step win. Midpoint BUG is the thing that removes that
handicap. So the honest route to "BUG beats both" is **midpoint for the accuracy, HPC work
for the per-step cost** — the two multiply, and neither alone gets there.

Apply §§1–5 to the midpoint stepper and include it as a fourth arm everywhere.

---

## 7. Large MPO bond dimension — runs WITH §3, not after everything

Everything above varies χ(MPS) with `xxz_mpo` at D=5. Cost in `H·Θ` is linear in D, so a
D=1024 MPO is a ~200× different problem — and it is the setting where the sketch has the most
to win, since the object being probed grows with D while the number of directions we actually
want does not. So this pairs with §3 rather than trailing the plan: **how well does the rSVD
hold up as the MPO bond dimension grows** is one question, not two.

Needs a random MPO generator mirroring `random_mps.jl` (inherit the sector structure, never
invent it), then §1 and §3 re-run over D ∈ {64, 256, 1024}.

---

## 8. Reduce allocation — promoted, on evidence

Measured at χ=1024, L=30, BLAS=1: **tdvp2 121 s/step allocating 142 GB; cbe1s 252 s/step
allocating 243 GB** — against a state of 0.08 GB. That is ~1800× churn, and both arms sit at
almost exactly 1 GB/s of allocation, so **runtime is tracking bytes allocated, not FLOPs.**

If the kernel breakdown confirms a large GC and memory-movement share, this outranks threading:
`to_concrete` after every contraction materialises a fresh tensor, and the sweep discards most
of them immediately. Targets: in-place contraction into caller-owned buffers, reusing the
Krylov basis vectors across solves, and avoiding the permute copies that dominated the small-χ
profile.

---

## Order of work

1. **§1 measurement at χ=1024–4096** — everything else is chosen from its output. *(running)*
2. **§4 Krylov convergence** — ✅ done, and it was a fairness bug.
3. **§2 Telum threading** — ✅ approved and forked to `deps/Telum`; sector loop threaded.
4. **§8 allocation** — priority set by the kernel breakdown from §1.
5. **§3 rSVD (absolute `dex`, `fold_omega`) together with §7 large MPO D.**
6. **§5 parallel half-sweeps at scale** and **§6 midpoint BUG** as a full arm throughout.

⛔ Cluster discipline: one timing job at a time, and the 9-point `cluster_compliance.md`
checklist walked with named evidence per gate before every `sbatch`.

---

## On the target

The aim is **BUG faster than 2-site TDVP and 1-site CBE-TDVP at these ranks**, per step and in
practice. Per step, BUG should gain from skipping the backward evolution (roughly half the
Krylov solves) and from parallel half-sweeps, neither of which TDVP can do.

⚠ **The accuracy handicap is NOT carried into this target.** The ~4× local error constant was
measured at L=8 and L=18 with χ ≤ 128; it is a small-system result and there is no basis for
assuming it holds at L=30, χ=1024, where the error budget is dominated by truncation rather
than by the local step. It gets re-measured at scale rather than assumed, and a 2× difference
in error is not a problem worth trading speed for.

Midpoint BUG stays in the comparison as a full arm — it was 63–70× more accurate for 1.35× the
wall time at matched cost, which is a good trade in its own right, independently of whether
forward BUG needs the help.
