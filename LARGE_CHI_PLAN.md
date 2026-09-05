# Making BUG and TDVP fast at χ = 1024 … 10000

**Goal.** Time-evolve at MPS bond dimensions 1024 / 4096 / 8192 / 10000, with BUG (forward
and midpoint) faster than 2-site TDVP and 1-site CBE-TDVP — and with the randomised sketch
decisively cheaper than the full SVD at these ranks.

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
