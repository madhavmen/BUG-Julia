# Handoff — RSVD-CBE at L = 18, as of 2026-07-31

State of the L=18 benchmark campaign, what is proven, what is *not*, and what to do next.
Written for an agent picking this up cold. Companion to `docs/current_work.md` §7.9 / §7.9.1,
which hold the full derivations; this file is the operational summary.

---

## 1. The one-line status

The benchmark could not run as specified — 84 rows × ~69 min ≈ **97 h**. The cause was found and
fixed at the driver level (Krylov depth 30 → 8: **3.64× faster, accurate to 2.9e-13 with the cap
binding**), and a subsequent cost-vs-χ scan showed cost is roughly **linear** in χ here, not
cubic. Together those put the real-time grid at **~6 h with the cutoff axis dropped** — an
overnight laptop run.

**Nothing is blocked on code.** The grid is ready to relaunch (§5); the only judgement call left
is whether to drop `CUTOFFS=1e-12`, which needs one L=18 pair to confirm rather than the L=8
agreement currently standing in for it.

⚠️ Note this reverses an earlier conclusion in this campaign that "the grid must be trimmed".
That was based on a *guessed* χ-scaling exponent (~3× per doubling); measured, it is ~1.9× per
doubling of χ for ~1.86× the time, i.e. linear.

## 2. What is PROVEN (measured, not argued)

| claim | evidence |
|---|---|
| exact reference is correct | 1.4e-14 vs dense `expm` at t=15; 1.8e-15 vs dense `eigen`; dimer 3.3e-16 |
| `:none` / `:U1` / `:SU2` are the same physics | identical to every printed digit, real + imaginary, dimer + domain wall |
| no random rank growth anywhere | no `missing_fill` / `pad` / KLS augmenter in the CBE path, pinned by a **source-level** test |
| CBE grows from a product state | reaches `[2,1,2,…]` at 1e-16 in `:none` AND `:U1`, both selection rules |
| the growth is not seed-driven | it grows under `:none`, where **no empty charge sector exists to seed** |
| sketch ≈ exact CBE | agree to every printed digit on all time arms; part only where the cap binds (`dmrg_cbe_rsvd` 7.53e-07 vs `dmrg_cbe` 1.78e-15 at χ=8, gone at χ=16) |
| `maxiter=8` is safe | 2.17e-13 / 2.97e-13 / 2.90e-13 at t=2/4/6, χ=64 cap **binding** (`reached_bd=64`) |
| root cause of the cost | main-thread profile: 97% of a `tdvp2` step is `expv`; mean Krylov dim **27.4 of a 30 cap** |

## 3. What is NOT known — do not claim these

- **No valid timing data exists for ANY arm.** Every timing this campaign has produced ran
  contended. Arm-vs-arm ranking is **unmeasured**. Do not quote `seconds` from any JSON as a
  comparison.
- **Cost scaling in χ is measured but the scan is INCOMPLETE** — see §4a. It reached step 100 of
  200 (χ=103 of a 128 cap) before being stopped, so the plateau at the cap was never observed and
  the 300-step extrapolation below is an estimate, not a measurement.
- **Whether `cbe_rsvd` costs accuracy at a binding cap at L=18.** The one place exact and
  sketched CBE ever diverged was a binding cap. At L=18 the caps *do* bind (`maxbd=32` at cap 32).
  This is the single most interesting open number in the campaign.
- **Imaginary-time and ground-state rows at L=18.** Never run; only `MODES=real` was attempted.
- **SU(2) at L=18.** Never run. Note SU(2) cannot do the domain wall — the state is
  unrepresentable and `<Sz_j>` is identically zero. Use `INIT=dimer` for any SU(2) comparison.

## 4. The results file

`benchmarks/l18_dw.json` — **archive, `maxiter=30`**, 2 rows + the exact reference:

```
real  none  tdvp2           chi=32  ct=0  err_end=3.991e-05  maxbd=32  4161.8s
real  none  1site_tdvp_cbe  chi=32  ct=0  err_end=5.605e-05  maxbd=32  3677.9s
```

Both have `maxbd` == cap, so they are genuinely rank-starved and their errors are meaningful.
These rows use the OLD key format (no `|mi30` suffix, no `maxiter` field). **Do not resume a
`maxiter=8` run into this file** — the rows would be recomputed correctly (the key now carries
the depth) but would still be *plotted* together, mixing 4159 s rows with ~1150 s ones.
`plot_l18.py` prints a warning if it sees mixed depths. Use a fresh `OUT=`.

Result JSONs are untracked by convention — no result file has ever been committed here.

## 4a. Cost vs χ — the projection is much better than first feared

`tdvp2`, L=18, `maxiter=8`, cap 128, domain wall, one step at a time (`scratchpad/chi_scaling.jl`).
**Steps 1–40 overlapped another job and are contended; steps 60–100 ran alone.**

| step | χ reached | s/step | cum s |
|------|-----------|--------|-------|
| 1 | 4 | 1.264 | 1.3 |
| 20 | 21 | 1.473 | 27.6 |
| 40 | 34 | 2.610 | 65.2 |
| 60 | 53 | 3.468 | 128.1 |
| 80 | 76 | 7.491 | 219.4 |
| 100 | 103 | 6.470 | 358.0 |

**Cost is roughly LINEAR in χ over this range, not χ³**: χ 53 → 103 is 1.94× the bond dimension
for 1.86× the time. That is the signature of a regime dominated by per-contraction overhead and
dispatch rather than by BLAS arithmetic — consistent with the profile, where no `gemm` appears at
all. It is also why the earlier ~3×-per-doubling guess overestimated the grid so badly.

**Revised estimate** (extrapolating the remaining 200 steps at ~7–8 s): `tdvp2` at cap 128 for
300 steps ≈ **30–35 min**, versus 69 min at cap 32 with `maxiter=30`. Scaling the other arms by
their measured per-step ratios (`1site` ≈ 0.7×, `gate` ≈ 0.15×, `lubich` ≈ 0.07× of `tdvp2`,
measured at `maxiter=30` so treat as indicative):

| grid | estimate |
|------|----------|
| one sym, one cutoff, 7 arms × 3 caps | ~3 h |
| `SYMS=none,U1`, `CUTOFFS=0.0` | **~6 h** |
| `SYMS=none,U1`, both cutoffs | ~12 h |

So with the cutoff axis dropped the real-time grid is an **overnight laptop run, not a four-day
one**. Caveats that must travel with these numbers: the scan never reached the cap, the arm
ratios come from a `maxiter=30` probe, and none of it is arm-vs-arm timing evidence.

## 5. How to relaunch

```bash
cd BUG-Julia
INIT=domainwall SYMS=none,U1 MODES=real L=18 TMAX=15.0 DT=0.05 SNAP=0.25 \
  MAXDIMS=32,64,128 CUTOFFS=0.0 MAXITER=8 OUT=benchmarks/l18_dw_mi8.json \
  julia --project=. benchmarks/l18_matrix.jl 2>&1 | tee benchmarks/l18_dw_mi8.log
```

- `MAXITER=8` is the default; stated explicitly so the log records it.
- `CUTOFFS=0.0` **is a proposed trim, not a validated one.** The reasoning is that when `maxdim`
  binds, `trunc_thresh` has nothing left to discard, and `maxbd` hit the cap in both rows so far.
  It agreed at L=8. That is an *extrapolation from a small size*, which is exactly the move that
  produced several wrong numbers in this campaign — **measure one L=18 pair before relying on it.**
- Plots: `python3 benchmarks/plot_l18.py benchmarks/l18_dw_mi8.json benchmarks/figs_dw`

**Run ONE Julia job at a time.** See §7 below.

## 6. Open work, in priority order

1. **Relaunch the real-time grid.** Per §4a this is now an **~6 h overnight run** with
   `CUTOFFS=0.0` — no trim beyond the cutoff axis appears necessary, which reverses the earlier
   "the grid must be cut" conclusion. Confirm the cutoff drop with one L=18 pair first (§5).
2. **Finish `scratchpad/chi_scaling.jl`** on an idle machine if a firmer projection is wanted —
   it stopped at step 100 of 200, so the plateau at the cap was never seen.
3. **Cache the bond gate.** `apply_h_two_site` calls `bond_gate(h, tl, tr)` →
   `heisenberg_bond_gate` on **every matvec** (`src/RSVDCBEBondUpdate/henv.jl:639`), rebuilding a
   tensor that depends only on `(site_l, site_r, J, delta)`. ~107/1564 ≈ 7% of a step. Pure
   memoisation, numerics-identical. Not done: the user scoped the fix to "driver only".
4. **The real Krylov fix**, if the user ever authorises it: a convergence test in `lanczos_expv`
   so the depth adapts to `dt`, `L` and the spectrum instead of a constant that is only right
   for this problem. **`src/BondUpdateBUG` is not to be modified without an explicit request**,
   and it would change numerics for every arm, so the whole suite would need re-running.
5. Imaginary-time and ground-state modes; then the SU(2) dimer run.

## 7. Traps this campaign actually fell into

Every one of these produced a *plausible, confident, wrong* answer. They are listed because each
cost real time and each is easy to repeat.

- **Profiling all threads.** MKL parks a 12-thread pool; the sampler counts them sitting in
  `ZwDelayExecution` as 21.8% of "time". That led to a BLAS-threading diagnosis that
  `BLAS.set_num_threads(1)` then refuted (1.03× / 0.98× / 0.93× / 1.07×). **Always profile
  `threads = 1:1`.**
- **Extrapolating from one sample.** Timing one matvec at one bond and multiplying produced a
  bogus "66% of the step is not Krylov". Counting iterations at one bond produced a bogus "5×
  fewer matvecs". Both were wrong; the module's `enable_krylov_log` gives the real distribution.
- **Timing under contention — three times.** The last instance was launching a probe *while* an
  accuracy validation ran, yielding 2.51× / **0.79×** / 1.47× for maxiter 12 / 8 / 6 —
  non-monotonic, i.e. less work taking longer. **One timing job at a time.**
- **Reading a benchmark whose cap never bound.** A knob is inert unless the run is rank-starved:
  always check `maxbd` reached the cap before reading an error column.
- **A control that passed by doing nothing.** `parity_bonds(:even)` is `1,3,5,…` and `:odd` is
  `2,4,6,…` (0-based names). Asking for the wrong group makes every gate `nothing`, so the sweep
  silently does nothing and returns the input unchanged — indistinguishable from a real rank
  freeze. **A control must be shown to do something before its passing means anything.**
- **JIT in a timing column.** First-call rows carry minutes of compilation (`1site_tdvp_cbe`
  87.5 s vs `1site_tdvp_cbe_rsvd` 4.8 s at L=8 is compilation reuse, not an 18× speedup).

## 8. Hard constraints (from the user, standing)

- **`cbe_rsvd` IS the method.** Rank grows *only* via RSVD-CBE. No random sector seeds, no
  `missing_fill`, no `pad`, no KLS. Exact CBE is an A/B reference only. KLS is not an arm, not a
  fallback, not a baseline.
- **Do not modify `src/BUG` / `src/BondUpdateBUG`** unitarity-critical components without an
  explicit request.
- Naming is **`bond_update_bug`**, never "two-site BUG".
- Commits authored Madhav, **no Claude co-author line**. Branches pushed only on confirmation.
- Cluster work goes through `sbatch`, never Julia on the login node; never write `--time`.
  **This benchmark is the exception** — the user explicitly scoped it to their local laptop.

## 9. Environment notes

- `feature/cbe-bug` (41 commits) is **merged into `main`** (merge `92fc033`, no conflicts) and
  **both branches are pushed** — `origin/main` at `6f664c1` or later.
- **Test status, measured.** `tests/RSVDCBEBondUpdate/runtests.jl` ran **1161/1162** (45 min).
  The single failure was `test_su2.jl` asserting that `dimer_state` refuses outside `:SU2` — a
  guard removed on purpose when the L=18 benchmark needed one state constructible in all three
  symmetry modes. It had been red since `50662c7`, i.e. **`main` was already failing when this
  branch was merged**, because the suite was not re-run for that merge. Fixed by inverting the
  assertion; SU(2) testset now **29/29**.
- ⚠️ `tests/runtests.jl` (the *outer* suite, covering `BondUpdateBUG` too) has still not been
  run end to end in this campaign. The module suite above is the strongest evidence there is.
- The module suite takes **~45 min**; the SU(2) testset alone is ~17 min. Budget for that
  rather than assuming a hang.
- Laptop: 14 logical cores, 31 GB, MKL with a 12-thread pool.
- A Julia process (PID 31952) has been resident since **2026-07-27** holding ~700 MB. It was
  flagged to the user twice and left alone — it may be their REPL. Not the agent's to kill.
- JIT is ~3 min on the first call of each arm; a silent log is not a hang.
