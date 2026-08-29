# The three-model campaign — imaginary time, log grids, and one open system

Everything here serves **one claim**: `cbe_bug` is more robust than 2-site TDVP, shown on
logarithmic time grids where a log grid is legitimate and on a uniform grid where it is not, with
symmetry as our addition.

Three models, and the third exists to break the pattern of the first two:

| # | model | time | grid | symmetry | result |
|---|---|---|---|---|---|
| 1 | Heisenberg chain, L=16 | imaginary | log | SU(2) ✅, U(1) ✅ | **BUG 8.8× fewer matvec, 4 orders more accurate** vs Bethe |
| 2 | Heisenberg cylinder, L=16 | imaginary | log | SU(2) ✅, U(1) ✅ | **BUG 12.2–12.4× fewer matvec**; QMC anchor ⛔ not yet quotable |
| 3 | Boundary-driven XX chain, open | real | uniform | ⛔ `:none` only | **two regimes** — tdvp2 wins at `eps=1`; tdvp2 **blows up** at `eps ≳ 24`, BUG does not |

The reference is **analytic in every case** — Bethe ansatz for 1 and 2, a closed-form Lyapunov
solution for 3. No `2^L` or `4^L` object is built anywhere except to certify a reference once, at
sizes where that is cheap.

`logtime_suite.jl` carries a fourth model, `neel` — Barmettler's real-time Néel quench. It is not
one of the three, but it makes the model-3 grid argument in a *closed* system: the same log grid
that wins in imaginary time needs ~6× **more** steps than uniform once the dynamics oscillates.

Full write-up, with the arguments and the failures behind each number:
[`docs/PLAN-open-systems.md` §7](../../docs/PLAN-open-systems.md).

## Files

**Drivers** — each writes CSV to `../results/`, each guarded so it can be `include`d without running.

| file | what it produces |
|---|---|
| [`logtime_suite.jl`](logtime_suite.jl) | `chain` and `cylinder` (models 1–2), plus `neel` — error vs *number of steps*, log grid against uniform, per symmetry |
| [`dissipative_xx.jl`](dissipative_xx.jl) | model 3: generator validation, χ(t), the `(eps, dt)` robustness scan, the NESS |
| [`logtime_trajectory.jl`](logtime_trajectory.jl) | wall clock and χ **along** the trajectory — the figures of arXiv:2606.02930 |
| [`bug_knob_study.jl`](bug_knob_study.jl) | where the rSVD speedup comes from, and how the two truncations interact |
| [`logtime_ite.jl`](logtime_ite.jl) | the narrow question: does our BUG need their A-stable implicit stepper to follow a log grid? |
| [`krylov_depth_scan.jl`](krylov_depth_scan.jl) | how many Krylov vectors one imaginary-time step needs as a function of `dt` |
| [`expv_substep_check.jl`](expv_substep_check.jl) | the large-`dt` failure a log grid forces at its last step, and what splitting `tau` costs |
| [`certify_xx_reference.jl`](certify_xx_reference.jl) | certifies model 3's analytic reference — **the only place a dense Liouvillian is built** |
| [`mpdo_validate_fast.jl`](mpdo_validate_fast.jl) | the same validation at parameters that finish in minutes |
| [`dissipative_ising.jl`](dissipative_ising.jl) | Cui–Cirac–Bañuls (PRL 114, 220601) — the dark-state argument; ⚠ built and certified, **not run** |

**Shared** — [`grids_common.jl`](grids_common.jl) holds the grid constructors, `run_arm`,
`trajectory_error` and the CSV schema. It is one file because a second copy would drift and the
drift would be invisible: `plot_logtime.py` reads the columns it defines.

**Reproducers** for the U(1) defect that blocked every symmetry result until it was fixed —
[`u1_cylinder_repro.jl`](u1_cylinder_repro.jl) (the failure, with a stack trace, and the
before/after verification) and [`fusion_tag_probe.jl`](fusion_tag_probe.jl) (the five-line question
that found the cause).

**Figures** — `plot_logtime.py`, `plot_logtime_traj.py` and `plot_xx_chigrowth.py` plot the CSVs
the drivers here write. `plot_spectrum.py` draws the "why an open system cannot use a log grid"
figure, but ⚠ its CSV comes from `benchmarks/dissipative_heisenberg.jl spectrum`, **outside this
folder** — see the grid-argument note below before quoting its numbers for model 3.

**Cluster** — `submit_logtime_suite.slurm`, `submit_dissipative_xx.slurm`,
`submit_bug_knob_study.slurm`. All three `cd` to the repo root and take paths from there.

## Running

Local runs are for **correctness against an exact reference at L ≤ ~12**. Anything at L ≥ 16, any
sweep, and *every* wall-clock number belongs on the cluster.

```bash
# models 1 and 2. Positional args select models and mode; `smoke` shrinks all three to L<=12 and
# is the ONLY thing that should run locally. Naming a model APPENDS, so a resumed run keeps rows.
julia -t 2 --project=. benchmarks/imaginary_time/logtime_suite.jl smoke chain
julia -t 2 --project=. benchmarks/imaginary_time/logtime_suite.jl l16 cylinder   # the quoted size
python3 benchmarks/imaginary_time/plot_logtime.py

# model 3 -- cheapest check first; `calibrate` answers "is the MPDO right at all"
julia -t 2 --project=. benchmarks/imaginary_time/dissipative_xx.jl calibrate
XX_L=8 XX_DT=0.02 XX_TMAX=0.8 julia -t 2 --project=. \
    benchmarks/imaginary_time/dissipative_xx.jl chigrowth
python3 benchmarks/imaginary_time/plot_xx_chigrowth.py

# the robustness result -- ONE PROCESS PER eps; the default eps=1 finds nothing (see below)
for e in 16 24 32 64; do
  XX_EPS=$e XX_D=32 XX_TMAX=0.8 julia -t 2 --project=. \
      benchmarks/imaginary_time/dissipative_xx.jl dtrobust
done
```

⚠ `bug_par` launches its two half-sweeps as tasks. On one thread it measures the sequential sweep
**plus** scheduling overhead, and the driver warns about it — use `-t 2` or more before quoting any
parallel number.

⚠ One timing job at a time. A contaminated wall-clock column is worse than a missing one.

## What the numbers mean, and what they do not

### ⛔ `matvec` is not a cost proxy for BUG

`_record_krylov` is called **only from `expv.jl`**, so the `matvec` column counts time-evolution
Krylov solves and *nothing else*. BUG's CBE expansion — the rSVD sketch, the QR, the SVDs — is
invisible to it. On model 3 the undercount implied by wall clock is ≈ 40×:

| arm | matvec | seconds | χ | err |
|---|---|---|---|---|
| 2-site TDVP | 8,271 | **138.9** | 64 | **4.98e-08** |
| BUG (rSVD-CBE) | **315** | 204.7 | 64 | 4.90e-05 |

26× fewer matvec and still slower. Models 1–2 are matvec-bound and the ratio is closer to honest
there, but it is never exactly honest. Where wall clock and `matvec` disagree about which arm is
cheaper, say which one you are quoting and why.

### ⛔ The log-grid argument is *degeneracy*, not aliasing

For models 1–2 the generator `−H` is Hermitian, so `max|Im λ| = 0` — every mode is a monotone
decay, nothing can alias, and a log grid's exponentially growing steps cost nothing.

For model 3 the Lindbladian is non-Hermitian, so its modes are decaying **oscillations** and
Nyquist caps the *largest* step no matter how many points a grid has. The sharp form of the claim
is therefore **not** "the log grid aliases" — that invites "then use more log points". It is that a
log grid spends its resolution at small `t`, so satisfying Nyquist at large `t` forces it to a step
count *worse than uniform*. It is not merely wrong here, it is worse than the thing it was meant to
beat.

⚠ **Model 3's observable is QUADRATIC in the modes**, so it beats pairs of rapidities together: the
fastest angular frequency is `2 max|Im μ| ~ 4J` and the Nyquist step is `π/4`, **not** `π/2`.
Measured at N=30: `n = 2000` log steps overshoot by **79×** and `n = 12` by **8406×**, against
~12,000 *uniform* steps to reach `t_max`.

⚠ `plot_spectrum.py` draws this argument, but on the **dissipative Heisenberg** chain
(`max|Im λ| = 10.55` at L=3, `14.29` at L=4) — a *different* open model, whose uniform dissipation
gives it an N-independent gap ≈ 0.50 and a steady state at `t ≈ 6`. Its producer is
`benchmarks/dissipative_heisenberg.jl spectrum`, outside this folder. Do not attribute those two
numbers to the boundary-driven XX chain; its gap closes as `N⁻³` and it reaches no steady state at
all here.

### ⛔ Model 3 has two regimes — never quote one alone

**Weak dissipation (`eps = 1`)**: tdvp2 is more accurate, and it is an **order** gap (tdvp2 ~2.98,
bug_rsvd ~1.98 on the fused MPDO), which *widens* as `dt` shrinks. Three excuses were tested and
all three eliminated — Krylov depth (`maxiter` 8→30 moved the error from 5.3e-05 to 5.644e-05),
rank budget (already 1.7e-05 at χ=12 against a cap of 64), and CBE `growth` (saturated at 4).

**Strong dissipation (`eps ≳ 24`)**: tdvp2 undoes the double-counting of its two-site sweep with a
**backward** substep (`tdvp2_baseline.jl:124,136`). On a Lindbladian `tau` is real and the spectrum
sits in the left half-plane, so that step carries `exp(+|Re λ|·dt/2)` and is *anti-dissipative by
construction*. `cbe_bug.jl:844` has one forward `expv` and none backward.

| eps | dt | tdvp2 err | tdvp2 raw `Tr ρ` | BUG err | BUG `Tr ρ` |
|---|---|---|---|---|---|
| 16 | 0.4 | 7.85e-04 | 1.000000 | 2.64e-02 | 1.000000 |
| 16 | 0.8 | **2.30e-01** | 1.000000 | **6.04e-02** | 1.000000 |
| 24 | 0.8 | **TRACE LOST** | **6.27e+07** | 4.47e-02 | 1.000000 |
| 32 | 0.4 | **TRACE LOST** | **2.82e+09** | 1.45e-02 | 1.000000 |
| 64 | 0.8 | **TRACE LOST** | **1.98e+31** | 1.90e-02 | 1.000000 |

`Tr ρ = 1.98e+31` is the result, not the `err` column: `exp(Lt)` is trace-preserving *exactly*, so
any deviation is pure integrator error and 1e31 cannot be truncation. It grows monotonically in
both `eps` and `dt` — the signature `exp(+|Re λ|·dt/2)` predicts.

⚠ **BUG's own failure mode, which must be reported beside it.** At `dt = 0.8` its χ collapses to
**1** at every `eps` — CBE cannot build rank out of one huge step — so its survival there is a
product state carrying 2–6e-02 error. At `dt = 0.4` it keeps χ = 13..32 and is genuinely usable.
Its `Tr` reads a perfect 1.000000 throughout, so a **trace-only gate would score the collapsed
state a pass**.

⚠ `eps ≳ 16 J` is the strong-measurement / Zeno regime. Physically legitimate — but name the regime
rather than stating the claim flat.

⚠ **A null from an under-powered scan is a statement about the parameter, not the method.** Three
separate scans (purity at γ=0.1, `dt` to 0.8, long-time at fixed rank to t=2) all came back null at
`eps = 1`, because `|Re λ| ~ eps` and no reachable `dt` makes the exponent O(1) there. This was
briefly written up as "the hypothesis is unsupported". It was not; the scan was.

### ⛔ The NESS is not reached and must not be claimed

Model 3's Liouvillian gap closes as `N⁻³` — already `2.5412e-02` at N=8, so a true steady state
needs `t ≈ 236` (~12,000 steps at dt=0.02), and `t ≈ 1300` at N=16. Everything measured locally is
the **transient**, which is exactly where χ grows and where the integrators can differ. The
closed-form `j = 0.4` is drawn as an asymptote for orientation only. Charging an integrator for the
residual physical relaxation would report physics as integrator error.

### ⛔ No open model here has a symmetry arm

`fused_space()` builds the d=4 site through `_getLocalSpace_no_symmetry`, so **every fused superket
in this repo is `:none`**, whatever the model. This is sharpest for the XX chain, which physically
*does* conserve `Sᶻ`. A symmetry arm needs a new d=4 local space graded by `q = n_ket − n_bra` with
sectors `{−1, 0, 0, +1}` — real work, not a parameter change.

### ⚠ χ is not comparable across symmetries

Same state, same step: `χ = 6` under SU(2), `16` under U(1)/`:none`, because SU(2) counts
**multiplets**. Measured on the L=16 cylinder, `tdvp2` at `:U1, maxdim=128` gives `err = 2.9e-02`
against `2.76e-03` at `:SU2, maxdim=70` — the U(1) run is *under-resolved, not worse-behaved*. A
fair comparison must match the effective dimension, never `maxdim`.

## Open

* **The cylinder QMC anchor is not quotable.** All four arms agree on `e_bulk = −0.6805250308` to
  ten digits, but `resid = 8.0e-03` and the driver's own "FIT NOT LINEAR, do not quote" guard
  fires — three short lengths cannot reach the bulk. Needs `Lxs = (4,6,8)`, which is
  `submit_logtime_suite.slurm` task 1.
* **BUG vs TDVP at matched accuracy on model 3 is extrapolated, not measured.** Second order ⇒
  `dt/31.6` for 1000×, giving ~10,000 matvec against tdvp2's 8,271. Measure before quoting.
* **The dark-state demonstration has not been run.** `dissipative_ising.jl` is built and certified
  to 1.4e-14; the argument it supports is §7.5 of the plan.
* **The `(eps, dt)` stability figure** — raw `Tr ρ` on a log axis — is not yet drawn. The 1e31 is
  the entire argument in one number.
