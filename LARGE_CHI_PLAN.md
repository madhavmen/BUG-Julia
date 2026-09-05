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

## 4. Cut Krylov cost — the largest BUG-side lever needing no fork

`lanczos_expv`'s exit condition is **breakdown-only**, so *every* solve burns the full
`maxiter` regardless of whether it converged ten iterations ago. Since the step is
essentially `maxiter × (H·Θ)`, a genuine residual-based convergence test is a direct
multiplier on every arm — and it is ours to change, in `BondUpdateBUG/expv.jl`.

- 4a. Real convergence test (residual estimate from the Lanczos coefficients), with the
  breakdown check kept as the floor.
- 4b. Per-solve adaptive `maxiter`.
- 4c. Re-measure — with a fixed-`maxiter` arm retained as the control, because a convergence
  test that quits early is indistinguishable from one that quits wrong until it is scored
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

## 7. Large MPO bond dimension

Everything above varies χ(MPS) with `xxz_mpo` at D=5. Cost in `H·Θ` is linear in D, so a
D=1024 MPO is a ~200× different problem. Needs a random MPO generator mirroring
`random_mps.jl` (inherit the sector structure, never invent it), then re-run §1 over
D ∈ {64, 256, 1024}.

---

## Order of work

1. **§1 measurement at χ=1024–4096** — everything else is chosen from its output.
2. **§4 Krylov convergence** — biggest win available without forking Telum.
3. **§3 rSVD as absolute `dex` + `fold_omega`** — where randomisation finally pays.
4. **§5 parallel half-sweeps at scale** + **§6 midpoint**.
5. **§2 Telum threading** — pending your decision on forking.
6. **§7 large MPO.**

⛔ Cluster discipline: one timing job at a time, and the 9-point `cluster_compliance.md`
checklist walked with named evidence per gate before every `sbatch`.

---

## On the target

"BUG faster than both" is reachable, and it is worth being precise about the arithmetic so we
know when we get there. Per step, BUG should gain ~2× from skipping the backward evolution
and ~2× from parallel half-sweeps. Against that, forward BUG needs 2× the steps at matched
accuracy. Net: roughly 2× — a real win, but short of beating both arms *combined*.

Closing the rest is what §6 is for: midpoint BUG removes the 2× step handicap, and then the
per-step gains apply in full. That is the configuration to aim the optimisation at.
