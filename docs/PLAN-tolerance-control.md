# PLAN — putting every error under one tolerance

Addressing Jan's seven points, the question that prompted them — **why does the error not depend
on the truncation threshold?** — and the follow-up: **where** in the step the truncation is
applied, half-sweeps versus root.

Fixed experimental setting for the whole investigation, so every plot is comparable:

| | |
|---|---|
| model | Heisenberg, `XXZ Δ=1`, **open chain**, `J=1`, `xxz_mpo(20; J=1.0, delta=1.0)` |
| symmetry | `:U1` (point 7 — SU(2) and Haldane-Shastry come after) |
| **L** | **20** |
| **dt** | **0.05** |
| initial state | `neel_state(20)` — a product state, `χ=1`, so rank growth is genuine |
| reference | **exact**, sparse Krylov in the `Sz=0` sector (`benchmarks/exact_sparse.jl`) |
| observables | staggered magnetisation; `L∞` error over the `⟨S^z_j⟩` profile; `χ(t)`; operator applications |

**The reference is affordable at L=20 and this is what makes the whole plan possible.** The
`Sz=0` sector is `C(20,10) = 184,756` states, `H` is sparse with ~`L/2` off-diagonal entries per
row, and `expv_sparse` carries an a-posteriori residual check. So `exp(-iHt)|Néel⟩` is exact to
solver tolerance — not a better-converged MPS — at a cost of seconds. Bethe gives the XXX
*spectrum* and has nothing to say about a quench, so this is the only exact real-time option, and
it is a good one.

⚠ The standing "Heisenberg vs the analytic reference" rule governs **ground-state energies**. It
does not apply here and cannot: there is no closed form for this quantity. Every error quoted
below is against the sparse-Krylov propagation, and every plot must say so.

---

## 0. First: reproduce and explain the τ_trunc insensitivity

This is the observation that motivates everything else, so it is measured before anything is
changed.

**Jan's hypothesis:** `τ_trunc ∈ {1e-8, 1e-10}` sits far below the time-integration error, so the
truncation never binds and the error is set by `dt` alone.

**Confound already recorded in this repo, and it must be excluded first.** At *full rank* these
schemes are exact for any `dt` — one step is `exp(τH)` on the complete space — so a `dt` scan
there measures roundoff, and a `dt` scan at too small a rank has no window either
(`docs/current_work.md`, the L=8 dtscan). The L=20 Néel quench should sit in the rank-limited
window by construction, but that has to be shown, not assumed: **control arm = uncapped `maxdim`
with `τ_trunc = 1e-14`**, which is the closest thing to exact this code can produce.

**Experiment 0.** `cbe_bug`, `tdvp_cbe1s`, `tdvp2`; `maxdim` uncapped; sweep

- `τ_trunc ∈ {1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 1e-14}` at `dt = 0.05`
- `dt ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}` at each of `τ_trunc ∈ {1e-4, 1e-6, 1e-8, 1e-10}`

> **Plot 0a** — `err(T)` vs `τ_trunc`, log-log, one line per scheme. The prediction is a **knee**:
> flat below it (dt-limited, which is the reported insensitivity) and rising above it
> (truncation-limited). *The knee's position is the answer to Jan's question.*
>
> **Plot 0b** — `χ(T)` vs `τ_trunc` on the same x-axis. Below the knee this is the cost paid for
> nothing, which is the other half of the argument.
>
> **Plot 0c** — `err(T)` vs `dt`, one line per `τ_trunc`. Where the lines separate from the
> common envelope is where truncation takes over from time integration. **The crossing locus is
> the calibration curve `τ_trunc*(dt)`** — the largest threshold that does not raise the error.
>
> **Plot 0d** — the recommendation checked: is `τ_trunc = dt × 10⁻² = 5e-4` on, above, or below
> `τ_trunc*(0.05)`?

**Deliverable:** a defensible default rule for `τ_trunc(dt)` on this model, and a statement of
whether it is `dt`-proportional (Jan's `dt × 10⁻²`) or scales with some power of `dt` instead. Every
later phase runs at that τ_trunc.

---

### ✅ ANSWERED (2026-08-24, L=12) — and the flat region is NOT dt-limited

The knee is real, but **the prediction above about what lies below it is wrong**, and that matters
more than the knee's position. Below the knee the error is **Krylov-depth-limited**, not
`dt`-limited. Two measurements, which meet exactly:

| | held fixed | scanned | result |
|---|---|---|---|
| `tolerance_driven_selection.jl` | `krylov_tol = 1e-6` (default) | `τ_trunc` 1e-4 → 1e-10 | error **flat at 1.4748e-06** for `τ_trunc ≤ 1e-8`, while χ climbs 11 → 24 → 37 |
| `krylov_stopping.jl` | `trunc_thresh = 1e-12` (deep in the flat region) | `krylov_tol` | error **moves**: 2.4032e-06 → 1.4748e-06 → 1.0800e-06 |

**The two tables agree to five digits and to the operator count.** `tolerance_driven_selection`'s
`τ_trunc = 1e-10` row is `1.4748e-06 @ 1622 krylov`; `krylov_stopping`'s Heisenberg
`tol 1e-6` row is `1.4748e-06 @ 1622 krylov`. Same number, same cost — so at `τ_trunc ≤ 1e-8` the
run is **entirely** Krylov-limited and truncation contributes nothing at all.

So the answer to *"why does the error not depend on the truncation threshold?"* is:

1. Above `τ_trunc ≈ 1e-5` it does — that region is truncation-limited and behaves as expected.
2. Below it the error is pinned by the **depth of the half-sweep Krylov basis**, a knob that was
   not in the original list of suspects and was until now **inert** (thresholding `β` never fired,
   §5). Tightening `τ_trunc` there buys 3.4× the rank (χ 11 → 37) for **zero** accuracy.
3. The floor is therefore moved by `krylov_tol`, not by `τ_trunc`, `growth`, `dex` or `maxdim`.

⚠️ **This makes `τ_trunc = dt × 10⁻²` a rule for the wrong knob in the flat region.** The
recommendation's *direction* is right where truncation binds; it simply has no purchase below the
knee, because nothing in the truncation path is what limits the answer there.

---

## What the code already does — three of the seven points are partly done

Worth stating up front, because it changes the size of the work.

**Point 6 is already an SVD, not a QR.** `cbe_bug.jl:320-325` splits with
`svd(T, (1,2); cutoff = split_cutoff)`, `split_cutoff = 1e-14`, `split_maxdim = 0`. So the change
is **the cutoff value, not the factorisation** — see §6, which is now the largest single piece of
this plan.

**Point 4 already exists, in the wrong sweep.** `_expanding_krylov` (`cbe_core.jl:1316`) does
exactly what Jan describes: expansion is *step 0 of each Lanczos iteration*, the probe is the
current Krylov vector `v_k`, the stored basis is re-embedded into the widened frames
(`embed_ops`), and full reorthogonalisation is mandatory because the projected operator changes at
every pass. Its own docstring makes Jan's argument for it — an outer loop keeps rediscovering
`H·Θ`, whereas `v_k` genuinely carries `H^k`. **But it is wired into `cbe_lubich_sweep` and
`GroundState`, not into the production `cbe_bug_step!`.** So point 4 is a *wiring* job plus the
environment re-push, not a new algorithm.

**Point 1 is half-done and the docstring is wrong.** `cbe_expand`'s docstring (`cbe_core.jl:447`)
claims `comp_ratio` defaults to `1.0`; the actual keyword at `cbe_core.jl:501` is **`0.5`**.
`cbe_bug_step!` passes `1.0` explicitly (`cbe_bug.jl:275`) so the production sweep is already
complement-only, but `CBEBugOptions` (`cbe_core.jl:856`) still carries `0.5`. **Fix the docstring
or the default — they cannot both stand.** Note also that `comp_ratio` is a property of the
*randomised probe*; with full SVD (point 3) there is no probe, so **points 1 and 3 together delete
the knob rather than retune it**.

---

## 1. Drop the ½·P_kept + ½·P_discarded split; project D on **both** legs

**RE-READ AGAINST THE CODE (2026-08-24) — most of this point is already implemented, and the
plan's earlier "Now" was wrong about it.** Two separate things were conflated:

**(a) The ½/½ split is already gone.** `comp_ratio` is the MATLAB's `0.5·P_kept + 0.5·P_discarded`,
and it splits the *probe's columns* between the frame's complement and its span. `cbe_bug_step!`
defaults it to **`1.0` — the probe is drawn ENTIRELY from the complement, i.e. the discarded
space.** The MATLAB explicitly rejects that choice (`RSVDpreBE0SiQS.m:339`, "do not project into
complement space, 1-site components are also important for bond update"); this repo already
departs from it in exactly the direction asked for, and has for some time.

**(b) The DD projection already exists, in the final selection.** `_preselect_left` applies
`P⊥^L = I − U0U0†` and `_preselect_right` applies `P⊥^R = I − V0†V0`, so `QL ⊥ U0` and `QR ⊥ V0`.
The final-selection matrix is then

```julia
UMU = QL† (HΘ) QR        # cbe_core.jl, "final selection" block
```

which is the **doubly projected** object — `D` on both legs of the two-site update tensor,
precisely as requested — and it is what the ranking SVD runs on.

**⛔ THE REAL GAP IS THAT THE DD RANKING IS NOT ALWAYS USED.** `joint = min(dex_l, dex_r, njl, njr)`,
and a side falls back to its own **singly** projected preselected candidates whenever its budget
exceeds `joint`, or when `UMU` is empty / `norm(UMU) < 1e-11`. The fallback is deliberate and well
argued in the code (a saturated side must not veto its partner, and the one-sided components are
the "1-site contributions" the reference insists matter) — but it means the DD-projected ranking
governs only when both sides have matching headroom.

**Change:** measure how often the fallback fires at L=20, and make the DD path the one that
governs where it can. The arms below stand, with `{joint DD ranking, per-side fallback}` added as
the axis that actually differs.

**Arms:** `{comp_ratio 0.5, 1.0, nothing}` × `{exact, sketched}`, reporting the **fallback rate**
per bond per step alongside the error.

> **Plot 1a** — `err(t)` per arm.
> **Plot 1b** — `err` vs `χ` (accuracy at equal rank), which is the comparison that matters given
> the recorded finding that *at fixed rank every integrator shares one error floor*.

⚠ The MATLAB explicitly rejects pure-complement (`RSVDpreBE0SiQS.m:339`, "1-site components are
also important for bond update"). We are departing from it deliberately; the plot is what
justifies that.

---

## 2. Tolerance-driven selection, not budget-driven

**Now:** the number of directions admitted is a **count** — `budget = ceil(growth·dmax) − r` with
`growth = 2.0`. (`stol_pre` / `stol_fnl` are no longer hardcoded: they are threaded through
`cbe_bug_step!` as of the sweep refactor, so the CBE-internal SVDs are already caller-governed.
Their *defaults* are still 1e-10 / 1e-13.)

⛔ **AND ON THE FALLBACK PATH THE SELECTION IS NOT EVEN WEIGHT-ORDERED — VERIFY THIS FIRST.**
`_trim_total(Q, leg, n)` walks `Q.spaces[leg]` and takes `min(d, left)` columns from each charge
sector in iteration order until the budget `n` is spent. Within one sector the preselection SVD's
ordering is by singular value, but ACROSS sectors it is by sector order, so a large singular value
in a later sector can be dropped in favour of a small one in an earlier sector. The code comment
asserts the candidates "are ALREADY weight-ordered"; that holds per sector, not globally.

This is the single cheapest thing to check in the whole plan and it sits underneath §1 and §2
both — a tolerance rule cannot govern a selection that is not ranked by weight. **Measure before
changing:** print, per bond, the singular values `_trim_total` kept against the ones it dropped,
and see whether the sets actually interleave. If they do not (e.g. one sector dominates in
practice), this is a latent trap rather than a live bug, and should be documented as such rather
than fixed.

**Change:** admit every singular vector of the DD-projected `HΘ` with `σ > τ_trunc`; retire
`growth` / `dex` / `sulz_cap` from this path; set `stol_pre = stol_fnl = τ_trunc`.

**The pass/fail criterion is already instrumented.** `CBEExpansion.err_fnl` is "the weight the
final selection discarded" and the code already calls it *the honest measure of what the `Dex` cap
cost*. Under a tolerance-driven rule it must fall to `≲ τ_trunc` — that is literally "make sure
all errors are governed by `τ_trunc` throughout".

> **Plot 2a** — `χ(t)`, budget-driven vs tolerance-driven.
> **Plot 2b** — `err_fnl`, `err_pre` and `discarded` per step vs `t`, **with the `τ_trunc` line
> drawn on the axes**. Everything must sit under the line; anything above it names the component
> that is not yet tolerance-governed.

---

### ⛔ MEASURED (2026-08-24, Heisenberg L=12) — THE PREMISE OF THIS POINT DOES NOT HOLD HERE

| rule | τ_trunc | err | χ | krylov | err_pre | err_fnl |
|---|---|---|---|---|---|---|
| budget | 1e-4 | 4.0632e-05 | 7 | 1546 | 1.458e-16 | **0.000e+00** |
| TOLERANCE | 1e-4 | 4.0632e-05 | 7 | 1544 | 1.038e-04 | 9.297e-05 |
| budget | 1e-6 | 1.4595e-06 | 11 | 1584 | 4.882e-11 | **3.186e-14** |
| TOLERANCE | 1e-6 | 1.4595e-06 | 11 | 1580 | 1.702e-06 | 2.878e-07 |
| budget | 1e-10 | 1.4748e-06 | 37 | 1622 | 1.669e-10 | **3.469e-14** |
| TOLERANCE | 1e-10 | 1.4748e-06 | 37 | 1622 | 1.669e-10 | 6.858e-11 |

**The two rules give the SAME error, the SAME χ and the same cost to within 0.3%, at every
tolerance.** And the reason is in the `err_fnl` column: under the budget rule it is **0.000e+00**
at `τ_trunc = 1e-4` and ~1e-14 elsewhere. *The final selection is discarding nothing.* At
`growth = 2.0` the count is never the binding constraint on this model, so making the selection
tolerance-driven has nothing to bite on.

⚠️ **The `_trim_total` ordering defect is therefore LATENT, not live** — it can only mis-select
when the count binds, and here it does not. It stays documented (`tolerance_driven_selection.jl`
header) as a trap to fix before anything raises `growth` pressure or lowers `dex`, not as the
cause of anything currently measured.

**Where the tolerance rule DOES change things** is that it makes `err_pre` / `err_fnl` track
`τ_trunc` as Jan asked (1.038e-04 at 1e-4 down to 6.858e-11 at 1e-10) instead of sitting at a
hardcoded floor. That is worth having for *diagnosis* — those columns now mean what the plan says
they mean — but it does not move the answer.

### ✅ POINT 3 ANSWERED AT THE SAME TIME: the sketch is free

| τ_trunc | full SVD (`exact = true`) | rSVD (`exact = false`) | ratio |
|---|---|---|---|
| 1e-6 | 1.4595e-06 (χ 11) | 1.4595e-06 (χ 11) | **1.000** |
| 1e-10 | 1.4748e-06 (χ 37) | 1.4748e-06 (χ 37) | **1.000** |

Identical error and identical rank. Jan's "use full SVD first to eliminate one source of
uncertainty" is satisfied trivially on this model: **there is no uncertainty to eliminate.** The
randomised sketch can stay on, and the plan's phase-6 "re-introduce rSVD" step is a no-op here.
(This is a Heisenberg L=12 statement; it is exactly the kind of claim to re-check at L=20 and
under SU(2), where sector dimensions change what the sketch is sampling.)

---

## 3. Full SVD before rSVD

`exact = true` already exists in `cbe_expand` (`cbe_core.jl:615`) and swaps the thin Gaussian for
the complete fused basis — `full_local_basis` — so preselection returns the *true* leading
directions of `P⊥(HΘ)` rather than an estimate. The example scripts already pass it. Make it the
default for this investigation and quantify what it costs.

> **Plot 3** — `err` and cost (operator applications; wall clock cluster-only) for exact vs RSVD,
> across `τ_trunc`.

⚠ The repo records exact and RSVD as **bit-identical on OAT and XX** (`rsvd_param_scan.jl` phase
0, relative difference `0.000e+00`). That was measured on models the same repo calls *controls,
not evidence*. It has to be re-checked on Heisenberg at L=20 before it is repeated.

---

## 4. Expand the bond at **every** Lanczos step

**RE-TARGETED (2026-08-24): there is no K-step any more.** The place this applies is now
`_krylov_frame`, the half-sweep basis builder. It currently expands the bond ONCE (via
`cbe_expand`), builds `H1` against that fixed `ex.V_ex`, and runs the whole Krylov recursion at
that width. Jan's §4 is to widen between iterations instead.

✅ **The pattern already exists in this repo — `_expanding_krylov` (`cbe_core.jl`) does exactly
this for the ROOT solve**, including the part that is easy to get wrong: it re-runs `cbe_expand`
and then EMBEDS the accumulated basis into the widened space (`embed_ops` / `embed_with`) rather
than restarting. So §4 is "give the half-sweep the treatment the root already gets", not a new
mechanism.

⛔ **AND THERE IS A MEASURED TRAP IN THE OBVIOUS IMPLEMENTATION.** `cbe_lubich_sweep`'s
`grow_iters` re-runs the CBE *selection* on every pass, which re-truncates by budget and discards
exactly the directions the next application of `H` needs: MEASURED on OAT it bought 1.8× and then
went flat (2.13e-02 → 1.16e-02) no matter how many passes were allowed. **Jan's formulation avoids
this by construction** — "new kept = old kept **plus** expanded" is accumulation, not reselection —
so a faithful §4 must widen the space without re-ranking what is already in it. Any implementation
that saturates is reproducing the `grow_iters` bug, not discovering a limit of the method.

Jan's bookkeeping maps onto the existing structure as follows, and one part needs no new code:

- *new kept = old kept ∪ expanded* → `oplus([U0, TLC], (3,))`, already how the frame grows.
- *new discarded = old discarded ∖ expanded* → **automatic**. The discarded space here is not
  stored; it is `P⊥ = I − U U†` of the current frame, so widening `U` shrinks it by exactly the
  admitted directions. No separate machinery — worth saying so we don't build redundant state.
- *environments must be updated* → **this is the real work.** The half-sweep builds `H1` from
  `push_right_channels(right_channels(rstack, i+2), mpo, ex.V_ex, i+1)` once per bond; once the
  frame grows *inside* the Krylov loop, that push and the `one_site_h` build must happen per
  growth pass. That is also where the cost is, so Plot 4c is the one that decides this.

### ⛔ MEASURED (2026-08-25, L=12) — IMPLEMENTED, CORRECT, AND IT BUYS NOTHING

`krylov_grow = true` on `cbe_bug_step!`. Ten steps, dt=0.05, against exact references:

| model | m | grow=false | grow=true | χ (false / true) | widenings |
|---|---|---|---|---|---|
| XX | 3 | 5.0060e-05 | 5.0060e-05 | 10 / 10 | 96 |
| Heisenberg | 3 | 4.5282e-07 | 4.5282e-07 | 35 / 35 | 166 |
| OAT | 3 | 3.7303e-14 | 1.1546e-14 | 7 / 7 | 133 |

**Bit-identical error, identical final rank.** OAT's difference is roundoff against roundoff —
both arms are at machine precision. The frame demonstrably DOES widen (`n_grow` counts actual
widenings, not calls), and the answer does not move.

**WHY, and it is the same root cause as §2's null result.** At `growth = 2.0` the first
`cbe_expand` on a bond already discards nothing — measured there as `err_fnl = 0.000e+00`. The
selection is not budget-limited, so the initial frame already spans every direction that carries
weight, and a second expansion can only admit directions with none. Whatever extra width the
growth passes add mid-recursion is then removed again by the frame split and the closing
truncation, which is why the final χ is unchanged to the integer.

**Two implementation traps found on the way, both worth keeping:**

1. `_frame_from` **cannot be used on the already-expanded bond.** It factorises both sides and
   joins them through `res_l.S * res_l.Vd`; the Krylov vector's bond leg was produced against
   `V_ex'` and carries the conjugate arrow, so `S` comes back 0-dimensional and the product has
   nothing to contract. One side is already an isometry, so the core is a single overlap and the
   `BondFrame` can be built from parts — less work, not more.
2. **The budget collapses on the second pass.** `budget = ceil(growth·dmax) − r`, and the first
   expansion already took the bond to `growth·dmax`. Before this was fixed, 178–238 expansion
   passes per run changed the error by *nothing* — a no-op that a call-counter reported as work.
   Each pass now gets its own `dex`, set to what the first expansion admitted on that bond.

⚠️ **The null result is only established where the selection is not budget-limited.** If a future
configuration raises growth pressure or lowers `dex` far enough that `err_fnl` becomes non-zero,
§4 becomes worth re-testing — that is the regime it was designed for, and this measurement does
not cover it.

> **Plot 4a** — `err` vs `grow_iters ∈ {0,1,2,3,4,…}` at fixed `τ_trunc`.
> **Plot 4b** — `χ` per Lanczos step within one bond update, showing the geometric growth
> `D, growth·D, growth²·D, …` the docstring predicts, and where `room` stops it.
> **Plot 4c** — `err` vs total operator applications, against the `grow_iters = 0` baseline. This
> is where the mechanism has to pay for itself.

Do this **last** among the algorithmic changes — it perturbs the most and is cleanest to read once
the tolerances are already under control.

---

## 5. Terminate Krylov on the tridiagonal elements, `τ_Krylov = τ_trunc/10`

⛔ **THERE ARE TWO KRYLOV DEPTHS IN A STEP, NOT ONE, AND THEY ARE SEPARATE KNOBS.** This section
originally described only the second; conflating them is why the first went unnoticed.

| # | where | knobs | status |
|---|---|---|---|
| 1 | the **half-sweep basis** each bond builds, `span{Θ, HΘ, …}` | `krylov_basis`, `krylov_tol` | ✅ **DONE 2026-08-24** |
| 2 | the **root Galerkin solve**, `_expanding_krylov` | `maxiter`, `tol` | ⏳ open |

**(1) — done.** Jan's literal criterion was implemented first and **it does not fire**: thresholding
the off-diagonal `β` gave BIT-IDENTICAL results at L=12 for anything from `1e-6` to `1e-13`, on
both Heisenberg and XX, because `β` is a BREAKDOWN detector and a generic `H` never breaks down.
Replaced with the Saad contribution the section already named:

```
contribution of vector m+1  =  β_m · |[exp(τ·T_m)]_{m,1}|
```

The tridiagonal elements ARE tracked, exactly as asked; they are just weighted by what the
propagator does with them before meeting the tolerance. `β` survives as `_KRY_BREAKDOWN = 1e-13`,
which is the job it can actually do.

**The estimate was validated before being trusted**, on dense Hermitian problems against the true
Krylov truncation error: it **overestimates** by 4×–160× and never underestimates outside
roundoff. That is the safe direction — stopping below `tol` guarantees the true error is below
`tol`, at a cost of ~1–2 extra vectors. `test_cbe_bug_exact_refs.jl` pins the *direction* of that
inequality, which is the property that makes the rule sound.

#### MEASURED — L=12, dt=0.05, T=1.0, `benchmarks/krylov_stopping.jl`

| arm | XX err / krylov | Heisenberg err / krylov | OAT err / krylov |
|---|---|---|---|
| cap, no test | 1.6948e-04 / 1770 | 1.0800e-06 / 2280 | 5.9952e-15 / 2158 |
| tol 1e-4 | 1.6948e-04 / **952** | 2.4032e-06 / 1384 | 1.6431e-14 / 1552 |
| **tol 1e-6 (default)** | 1.6948e-04 / 1180 | 1.4748e-06 / 1622 | 1.0880e-14 / 1766 |
| tol 1e-10 | 1.6948e-04 / 1550 | 1.1190e-06 / 1976 | 6.2172e-15 / 2000 |
| pinned m=3 | 1.6948e-04 / 950 | 2.6697e-06 / 1038 | 7.0610e-14 / 1036 |
| pinned m=0 | 1.6948e-04 / 289 | 8.8424e-05 / 308 | **2.1329e-02** / 313 |

✅ **XX — the rule cuts 1.9× for free.** Bit-identical error at every tolerance: it correctly finds
that depth buys nothing on this model.

✅ **OAT — the rule refuses to cut.** Every tolerance row stays at ~1e-14 where `m = 0` is 2.1e-02.
This is the safety property that matters, and it is the model where getting it wrong costs eight
orders. (The ordering *among* the OAT tolerance rows is roundoff against roundoff — ignore it.)

⚠️ **Heisenberg — it is a genuine dial, not a free lunch.** `1e-6` buys 0.71× the cost for 1.37×
the error; `1e-10` is nearly free at 1.04× error for 0.87× cost. **That monotone trade IS the
deliverable** — `β` gave one point and no dial at all, which is what "the criterion never fires"
means in practice.

⚠️ **`1e-6` was chosen while this knob was inert**, so the number predates the rule that now reads
it. Kept as the default deliberately; anyone wanting the accurate end should pass `1e-10` rather
than assume the default is conservative.

**(2) — open, and there is a measured 3.64× sitting in it.** `_expanding_krylov` exits on
`b < tol` with `cbe_bug_step!` passing `tol = 1e-15`, so every root solve burns `maxiter` (recorded
elsewhere as 27.4 of 30 used, with `maxiter = 8` giving **3.64× fewer applications at 2e-13**). The
same substitution applies and is cheaper to justify here, because `_reduce_krylov` **already
returns the `coeff` vector** the estimate needs. Deferred only because `_expanding_krylov` is
shared with `cbe_lubich_sweep` and the ground-state mode, so it is a wider blast radius than (1)
and wants (1)'s measurement in hand first.

> **Plot 5a** — mean/max Krylov depth vs `τ_Krylov`, with the `maxiter` ceiling drawn.
> **Plot 5b** — `err` vs total operator applications. Prior measurement says `maxiter = 8` gives
> **3.64× fewer applications at 2e-13**, so there is real headroom here.
> **Plot 5c** — `β` against the Saad estimate at termination, per solve; this is what says whether
> `τ_Krylov = τ_trunc/10` is the right factor or whether it should be tied to the sharper estimate.

---

## 6. Where the truncation happens — half-sweeps vs the root

The largest piece of this plan, and the one the follow-up question is about.

### There were THREE truncations in one `cbe_bug_step!`. There are now TWO. ✅ DONE

| # | where | code | controlled by | status |
|---|---|---|---|---|
| 1 | **half-sweep split**, at every site as the sweep passes | `_splitU(K1)` / `_splitV(B1)`, `cbe_bug.jl:320-325, 416, 499` | `split_cutoff` (`1e-14`), `split_maxdim` (`0` = uncapped) | **kept** — prunes essentially nothing at today's default |
| 2 | ~~**root core SVD**, on the Galerkin core `S1` at bond `c`~~ | `cbe_bug.jl:548` | `root_cutoff` (`0.0`), `root_maxdim` (`0`) | **REMOVED as a truncation.** The SVD still *splits*; it no longer cuts |
| 3 | **closing recursive truncation**, root-to-leaves over all bonds | `truncate_recursive!` | `trunc_thresh`, `maxdim`, guarded by `truncate` | **kept** — the only thing enforcing `maxdim` in production |

Each bond is split exactly once by the half-sweeps — the left sweep owns bonds `1…c` (`W[i]`), the
right sweep owns `c…L-1` (`Z[j]`) — so there is no double-cut there. Bond `c` is the exception in
a different sense: **both** of its frames come from split truncations, so the root bond is the one
most directly exposed to `split_cutoff`.

### What changed, and why it was blocking the experiment

**#2 was the only truncation with no switch.** `truncate = false` disabled the closing pass and
left the root core SVD running at `cut = max(trunc_thresh, 1e-14)` with `Nkeep = maxdim` — and
because of that `max`, **`trunc_thresh = 0` did not mean zero, it floored at `1e-14`**. So "no
root truncation" was not reachable, and any claim that a step's error is governed by one
tolerance had a third, unnamed cutoff inside it.

`cbe_bug_step!` now takes `root_cutoff = 0.0` and `root_maxdim = 0` **by default**: the SVD of
`S1` splits the Galerkin core into the two centre site tensors and does nothing else. Rank
control lives in exactly two places — `split_cutoff` on the half-sweeps and `trunc_thresh`/`maxdim`
on the closing pass — which is the two-knob contract the rest of this plan depends on. The old
behaviour is an A/B, not a deletion: `root_cutoff = max(trunc_thresh, 1e-14)`, `root_maxdim = maxdim`
restores it exactly.

⚠ Two consequences to carry into the measurements. `root_cutoff = 0.0` keeps every `sv > 0`
(Telum's selector is relative to `sv_max`), so the centre bond can transiently carry
numerically-negligible directions until the closing pass removes them — **expect the peak working
set at the root to rise slightly**, and check it rather than assume it. And with `truncate = false`
the shipped defaults now leave `split_cutoff` as the *only* rank control anywhere in the step,
which is exactly what makes 6A a clean experiment and exactly why it is not a production setting.

### Experiment 6A — the requested sweep: root off, half-sweep cutoff scanned

Root truncation **off** — now just `truncate = false`, since #2 is off by default — and

    split_cutoff ∈ {1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10}

The half-sweep cutoff is then the **only** rank control in the step, so if this scheme is
tolerance-governed at all, this is where it shows.

> **Plot 6a** — `err(T)` vs `split_cutoff`, log-log. A clean monotone line means the half-sweeps
> govern the accuracy. A flat line means they do not, and the error is set elsewhere.
> **Plot 6b** — `χ(t)` per `split_cutoff`. With no root truncation the rank can only ratchet
> **up**; the question is whether the half-sweep cutoff still holds it. Runaway `χ` at flat error
> is the diagnostic signature of a cutoff that is not binding.
> **Plot 6c** — the Pareto: `err` vs `χ`, all cutoffs on one curve. This is the plot that says
> whether half-sweep truncation is an efficient place to spend rank.
> **Plot 6d** — `discarded` per step vs `t` for each cutoff, with the cutoff line drawn.

### Experiment 6B — the full 2D scan: root-out × half-sweep

**Every root-out threshold of Phase 0 crossed with every half-sweep cutoff**, on the same ladder
both ways: `trunc_thresh ∈ {1e-4 … 1e-14}` × `split_cutoff ∈ {1e-4 … 1e-14}`, 64 arms, CBE-BUG
only (the two TDVPs prune continuously at every site by construction and expose one tolerance —
there is nothing on them to A/B). The closing pass stays **on** throughout; this sweeps its
threshold, it does not disable it. 6A is the arm where it is off.

⛔ **A 2×2 could not answer this and has been dropped.** With two values per axis, "the error
follows the row" and "the error follows whichever threshold is looser" produce the same picture.
The square ladder makes the diagonal — both thresholds equal, i.e. *one tolerance governing the
whole step* — a readable line through the matrix.

**What each outcome means, decided before the run so the heatmap is read and not rationalised:**

| pattern | reading |
|---|---|
| error varies **down columns** only | the **root-out** pass governs; the half-sweeps are inert at their shipped `1e-14` and `split_cutoff` is a knob with nothing behind it |
| error varies **across rows** only | the **half-sweeps** govern; the closing pass is only cleaning up after them |
| error follows **max(row, col)** | the **looser** of the two sets the accuracy — they are redundant and one can go |
| the two **combine** | they cut different things, and both thresholds must be quoted for any error claim to be reproducible |

> **Plot 6e** — heatmaps of `err` and `χ` over the matrix, diagonal marked, **plus** a line family
> `err` vs `split_cutoff` with one line per root-out threshold. Both, from the same data: a
> heatmap alone invites reading a colour gradient that is really one decade of noise, while the
> line panel puts it on a log axis where a flat line is unambiguously flat.

### Experiment 6C — does the root's *recursive* form still matter with the root off?

`truncate_recursive!` exists because edge-to-edge `truncate_sweep!` cuts every bond twice. With #3
disabled that machinery is out of the loop entirely, so 6A/6B also test whether the recursive
closing pass is load-bearing or whether half-sweep pruning subsumes it.

> **Plot 6f** — `{truncate_recursive!, truncate_sweep!, none}` at the best `split_cutoff` from 6A.

⚠ Prior measurement: the split-truncation grid spans 17% at `growth = 2.0` and XX was
**bit-identical across every setting** — but that was measured *with the root truncation active*,
which is exactly the arm 6A removes, and at a `growth` that §2 retires. **That finding does not
transfer.** Re-measure, do not assume.

---

---

## MEASURED — L=12, dt=0.05, T=1.5, Néel, uncapped `maxdim` (2026-08-24)

Local, small-`L`, correctness-grade. Cost is operator applications; wall clock is not quoted.

### Phase 0 — the knee is real, and Jan's value is on the wrong side of it

| τ_trunc | TDVP err | TDVP χ | CBE-BUG err | CBE-BUG χ |
|---|---|---|---|---|
| 1e-4 | 8.394e-05 | 8 | 5.075e-05 | 10 |
| 1e-5 | 2.869e-06 | 12 | **4.099e-06** | 13 |
| **1e-6** | **6.724e-07** | **18** | 5.161e-06 | 19 |
| 1e-8 | 8.421e-07 | 27 | 5.282e-06 | 32 |
| 1e-10 | 8.425e-07 | 40 | 5.282e-06 | 48 |
| 1e-14 | 8.425e-07 | 64 | 5.282e-06 | 64 |

**Knee at τ_trunc ≈ 1e-6 (TDVP), ≈ 1e-5 (CBE-BUG).** Past it the error is flat to three digits
over seven decades while χ goes 18 → 64: at τ=1e-10, **2.2–2.7× the rank for zero accuracy**; at
1e-14, 3.6×. ⇒ **the reported insensitivity is real and is dt-limitation.**

⚠ **`dt × 10⁻² = 5e-4` is LOOSER than the knee** — at 1e-4 the error is 8.4e-5, 100× above the
floor. Measured τ_trunc\* ≈ 1e-6 at dt=0.05, i.e. `dt × 2e-5`. Whether the RATIO is the invariant
is what the `dt` phase decides; one (L, dt) point cannot establish a scaling.

⚠ The optimum is **non-monotone**: TDVP's minimum is 6.724e-07 at 1e-6 and tightening makes it
*worse* (8.425e-07, stable to three digits from 1e-7 to 1e-14). Same pattern the L=16 ITE campaign
recorded, now in real time on the interacting model.

⚠ At the dt-limited floor **CBE-BUG is 6.3× less accurate than either TDVP** (5.28e-6 vs 8.43e-7)
at **3.2–3.6× fewer applications** (4,138 vs 13,430 / 15,042). First measurement behind the
`USAGE.md` §6 claim; true on accuracy at matched tolerance, false on cost.

⚠ L=12 reaches FULL RANK (χ=64) at tight τ, so the flat floor is genuinely the dt error. At L=20
full rank is unreachable and truncation binds throughout — expect the knee LOOSER and the floor
possibly never reached.

### 6B — the half-sweep cutoff is inert in the default sweep, and the cause is in the code

The 8×8 matrix varies **down columns only**: every row constant across all eight `split_cutoff`
values, error *and* χ, including the extreme corner (root-out 1e-14 × split 1e-4, ten decades
apart) → χ=64, err 5.3e-6. ⇒ **the root-out pass governs; `split_cutoff` is a knob with nothing
behind it.**

**Cause, confirmed by the sweep comparison rather than argued:** `rexpand` restores what the split
discarded one line later —

```julia
Wk = _splitU(K1)                                                    # truncates at split_cutoff
rexpand && (Wk = _union_left(svd(K, (1,2); cutoff = 1e-14).U, Wk))  # puts it straight back
```

In `historic`, which has **no union anywhere**, `split_cutoff` is **live**: χ spans 10 → 64 and the
error moves. ⇒ the union's **hardcoded `1e-14`** is the tolerance actually in charge — a third
hidden constant of the same species as the root cut. In `historic` the rank follows whichever of
the two thresholds is LOOSER, so both must be quoted there for a result to reproduce.

### Sweep structures — sweep 2 is dominated, sweep 1 is a genuine Pareto point

Diagonal (both thresholds equal), floor values:

| sweep | error | applications | |
|---|---|---|---|
| `default` | **5.3e-06** | ~4,138 | |
| `cbe_only` (sweep 1, OLD definition = `lubich`) | 1.9e-04 | **~250** | 20× cheaper, 35× worse |
| `historic` (sweep 2) | 2.3e-04 | ~4,000 | same cost, **40× worse — dominated** |

⇒ on this model the width restoration is worth **~40× in accuracy for essentially zero extra
Krylov**, which is a far stronger case for `kaug`/`rexpand` than the TFIM numbers in the docstring
and it is on the interacting model rather than a control.

⚠ **The `cbe_only` row above is the SUPERSEDED definition** (`kstep = false`, `V_ex` discarded,
left direction only) and is now the `lubich` arm. Sweep 1 was corrected to keep BOTH frames
(`evolve_sites = false`); its numbers are pending.

### ⭐ SWEEP 1 + KRYLOV BASIS BEATS THE SHIPPED SWEEP BY EIGHT ORDERS AT 3.6× LESS COST

The single most important result so far. OAT, L=12, dt=0.05, `krylov_basis` = the number of
Krylov vectors put into the half-sweep frame:

| `krylov_basis` | err | χ | krylov |
|---|---|---|---|
| 0 (plain sweep 1) | 2.1329e-02 | 7 | 313 |
| 1 | 2.1329e-02 | 7 | 553 |
| 2 | 1.1358e-04 | 7 | 800 |
| **3** | **6.9500e-14** | **7** | **1040** |
| 4 | 3.4417e-14 | 7 | 1280 |
| 8 | 9.9920e-15 | 7 | 2240 |
| SHIPPED | 5.2449e-06 | 9 | 3740 |

χ stays pinned at the exact analytic Schmidt ceiling (7) throughout — the right rank AND the
right basis. The jump from `m=2` to `m=3` is a CLIFF, not a slope: the signature of the local
Krylov space CLOSING, which is what should happen for a diagonal `H` with finitely many `m²`.

**⛔ TWO OF MY OWN CONCLUSIONS ARE RETRACTED BY THIS.** (1) "The OAT gap is an irreducible order
effect from the commuting Hamiltonian that no basis change recovers" — WRONG, a basis change
recovers it completely and then some. (2) That `grow_iters` saturating proved extra powers do not
help — WRONG: `grow_iters` saturates because it re-runs the CBE SELECTION each pass and so
re-truncates by budget, discarding exactly what the next application of `H` needs. Accumulating
the vectors and splitting ONCE at the end is a different thing entirely.

**THE MECHANISM, and it is what the whole sweep-1 argument turned on.** `kstep = false` carries
ONE power of `H` (`cbe_expand` gives `span{Θ, H·Θ}`). The shipped K-step's `exp(τH₁)Θ` carries
all of them. Adding the powers explicitly closes the gap without ever evolving a site tensor —
so "we are just not time evolving during the half sweep" was the right reading, and the fix is to
put the Krylov vectors in the basis, not to reinstate the evolution.

`_krylov_frame` in `cbe_bug.jl`: fully re-orthogonalised Lanczos (two Gram-Schmidt passes; a raw
power iteration collapses onto the dominant eigenvector by `k≈4` and the closing SVD then deletes
the vectors as dependent — saturation for a CONDITIONING reason that looks physical), stopped on
the residual `β < τ_Krylov = τ_trunc/10`, ONE SVD at the end.

### Sweep 1 VALIDATED before the matrix — `tests/sweeps/test_cbe_bug_exact_refs.jl`

16/16, L=12, against EXACT references on three models. What the sweep is: CBE widens the bond,
the expansion frame ITSELF becomes the basis, no one-site `expv` and no split in the half-sweeps,
root Galerkin step is the only evolution. Both frames are used across the step — `U_ex` → `W[i]`
on the left half-sweep, `V_ex` → `Z[j]` on the right.

| model | reference | dt=0.1 | dt=0.05 | ratio | χ | dE |
|---|---|---|---|---|---|---|
| XX nearest-neighbour | free-fermion closed form | 6.529e-04 | 1.695e-04 | **3.85** | 1 → 15 | 4.6e-15 |
| XX long-range `1/r²` | exact sparse, `Sz=0`, 924 states | 3.157e-03 | 1.546e-03 | **2.04** | 1 → 64 | 1.4e-15 |
| OAT | `(L/2)cos^(L−1)(t/2)` | 8.583e-02 | 2.133e-02 | **4.02** | 1 → **7** | 1.5e-14 |

**THE RANK GROWS FROM A χ=1 PRODUCT START in every case**, which is the property that matters: a
basis-only sweep has no evolved tensor to split, so only the CBE frame can carry new directions
in, and χ=1 is a fixed point of any sweep that does not genuinely widen. OAT lands **exactly** on
the analytic Schmidt ceiling `oat_rank(12,6,1.0) = 7`, not one above — and both TDVP arms
over-fill that profile, so it is a real property of the BUG half-sweeps.

⚠ **LONG-RANGE XX CONVERGES AT ONLY ~FIRST ORDER** (ratio 2.04 against 3.85 and 4.02), measured
at χ=63/64 where truncation is NOT the limiter — so it is a time-integration effect, not a rank
artefact. The basis-only sweep appears to lose an order once the interaction is long ranged.
Open, and worth its own dt scan.

⚠ **IT IS A LOW-ACCURACY INTEGRATOR.** OAT's 2.1e-02 is ~1.5% of a signal of order 1.44, matching
the `cbe_bug.jl` docstring table (`kstep=false` 6.4e-05 vs the shipped sweep's 3.3e-11 — six
orders). The test asserts CONVERGENCE and STRUCTURE, not competitiveness, and says so.

⚠ **Long-range XY is NOT free fermions** — the Jordan-Wigner string cancels only for nearest
neighbours, so the NN arm gets the single-particle propagator and the long-range arm gets an
exact sparse propagation. Using `xx_free_fermion_sz` at range would have been a silent error.

**HYPOTHESIS, NOT A RESULT:** if sweep 1's error is dt-limited and second order, matching the
default's accuracy needs ~6× smaller dt → ~1,500 applications, still ~2.7× cheaper. Neither the
order nor the dt-limitation is established for that sweep. The measurement that settles it is a
dt scan on sweep 1 alone, which is cheap given its cost.

---

## 6d. THE THREE-KNOB CUBE — how the knobs actually interact

`split_cutoff` (half-sweep) × `trunc_thresh` (root-out) × `krylov_basis` (`m`), Néel quench,
dt=0.05, T=0.5, `maxdim = 64`. `benchmarks/heisenberg_tolerance.jl phase=grid`, plotted by
`benchmarks/plot_knob_cube.py`.

| L | depth | best err | χ | krylov | x-spread (half-sweep) | y-spread (root-out) |
|---|---|---|---|---|---|---|
| 12 | m0 | 4.293e-05 | 29 | 112 | **1.0** | 3.4 |
| 12 | m2 | 3.171e-06 | 13 | 360 | 6.6 | 5.2 |
| 12 | m3 | 4.511e-07 | 15 | 478 | 8.1 | 16.8 |
| 12 | mdef | 2.632e-07 | **8** | 744 | 8.2 | 28.5 |
| 16 | m0 | 7.018e-05 | 31 | 112 | **1.0** | 9.8 |
| 16 | m2 | 3.900e-06 | 18 | 440 | 11.1 | 8.9 |
| 16 | m3 | 4.955e-07 | **8** | 598 | 16.0 | 15.3 |
| 16 | mdef | 4.780e-07 | 9 | 994 | 6.7 | 6.7 |

*(spread = worst max/min ratio of the error along that axis; 1.0 means the knob changed nothing)*

### ⛔ THE KNOBS ARE NOT INDEPENDENT: `m` GATES WHETHER `split_cutoff` EXISTS

At `m = 0` the half-sweep cutoff is **exactly inert** — spread `1.0`, i.e. bit-identical error and
bit-identical χ across four decades, at BOTH sizes:

```
L=16, m0:  split_cutoff →   1e-4       1e-6       1e-8       1e-10
  root 1e-4              6.872e-04  6.872e-04  6.872e-04  6.872e-04    χ = 6, 6, 6, 6
  root 1e-8              7.019e-05  7.019e-05  7.019e-05  7.019e-05    χ = 22 ×4
```

**The mechanism is structural, not numerical.** At `m = 0` the half-sweep frame is a SINGLE CBE
block, so `_splitU` has nothing to discard — the knob is attached to nothing. At `m ≥ 2` the frame
is a STACK of `m+1` Krylov blocks and `split_cutoff` is what truncates that stack. Depth is what
creates the structure the knob cuts, which is why BOTH truncation spreads grow monotonically with
`m` (1.0 → 6.6 → 8.1 → 8.2 at L=12; 1.0 → 11.1 → 16.0 at L=16).

### The hierarchy, and what it means for quoting a tolerance

1. **`m` sets the floor.** 163× better error at L=12 (4.29e-05 → 2.63e-07) and 141× at L=16,
   while *reducing* χ from 29 → 8 and 31 → 8. Depth buys accuracy AND rank, for ~6× the operator
   applications.
2. **`trunc_thresh` governs until it saturates, and the knee MOVES WITH `m`.** At `m = 0` it
   flattens past 1e-8 (χ 22 → 31 for nothing); at `m = 3` it has already flattened at 1e-6.
   **Quoting a `τ_trunc` without quoting `m` is meaningless** — the knee is not a property of the
   model alone.
3. **`split_cutoff` matters only at `m ≥ 2`**, and then it is worth up to 16×.

### ⚠️ AND THEY ARE NOT REDUNDANT — a loose half-sweep makes the root-out knob HARMFUL

The L=16 `mdef` panel is the one place the cube is genuinely two-dimensional:

```
L=16 mdef:  split →     1e-4        1e-6        1e-8       1e-10
  root 1e-6         1.193e-06   4.788e-07   4.780e-07   4.780e-07    χ = 21,  9,  9,  9
  root 1e-8         2.774e-06   4.791e-07   4.791e-07   4.791e-07    χ = 30, 26, 18, 18
  root 1e-10        3.183e-06   4.793e-07   4.791e-07   4.791e-07    χ = 36, 39, 41, 27
```

Down the LOOSE `split = 1e-4` column, tightening the root-out threshold makes the error **worse**
— 1.193e-06 → 2.774e-06 → 3.183e-06 — while χ climbs 21 → 36. More rank, worse answer. Along
every tighter column the same scan is flat. So "the looser of the two sets the accuracy" is
**false** here: a half-sweep cutoff that is too loose leaves junk directions in the frame, and a
tighter closing pass then keeps more of them rather than fewer.

**Practical consequence:** tighten the half-sweep FIRST. A τ_trunc scan run at a loose
`split_cutoff` measures that interaction, not the truncation error.

### Cost note

At L=16, `m3` reaches 4.955e-07 @ 598 applications against `mdef`'s 4.780e-07 @ 994 — within 4%
of the accuracy for **40% less cost**. The shipped default is not the cost-optimal point on this
model; it is the safe one.

⚠️ This is why the earlier one-knob scans read as "the tolerance does nothing": every one of them
was run at a fixed depth that was itself the binding constraint. A cube was required, and a
one-axis scan through this cube would have reproduced the same wrong conclusion at any fixed `m`.

---

## 7. Model ladder

### ⛔ MEASURED (2026-08-25) — `benchmarks/model_ladder.jl`

**Arm A: the standing red magnon test is NOT a Krylov-depth problem, and NOT a rank problem.**
`tests/sweeps/test_physics.jl:153` has been failing on `cbe_bug` for the Haldane-Shastry magnon
since before this work started. Scanned against depth for the first time (it could not be scanned
before, because thresholding `β` never fired):

| arm | max\|dSz\| | χ | krylov |
|---|---|---|---|
| `tdvp_cbe1s` (the bar) | **2.5994e-08** | 2 | — |
| `cbe_bug` DEFAULTS | 4.2543e-03 | 2 | 706 |
| `cbe_bug` `krylov_tol=0` (to cap) | 4.2543e-03 | 2 | 817 |
| `cbe_bug` m=2 / 3 / 8 / 16 pinned | 4.2543e-03 | 2 | 424 / 609 / 817 / 817 |

**Bit-identical to five significant figures from m=0 to m=16.** Depth is eliminated. So is rank:
`tdvp_cbe1s` reaches 2.60e-08 at the SAME χ = 2. The one-magnon state is exactly representable at
χ = 2, so both schemes represent it perfectly and one is five orders worse — **the defect is in
the evolution, not in the basis or the rank.** Two suspects down; this is where the next
investigation of that failure should start.

**Arm B: SU(2) reproduces U(1), and the multiplet/state trap is real.**

| symmetry | E(0) | E(T) | max χ | max states |
|---|---|---|---|---|
| `:U1` | −3.0000000000 | −3.0000000000 | 16 | 16 |
| `:SU2` | −3.0000000000 | −3.0000000000 | **6** | 16 |

Identical energies, both conserved exactly, from the `dimer_state` start (Néel is not
SU(2)-representable). SU(2) carries the same 16 STATES in 6 MULTIPLETS — 2.7× fewer blocks. Read
`max χ` alone and SU(2) looks like it lost information; that is the trap, and it is why any
cross-symmetry rank claim must quote `state_bond_dims`.

**Arm C: Haldane-Shastry conserves energy at every depth** — 3.29e-14 (defaults), 2.53e-14 (m=3),
1.78e-14 (m=16), at L=12 to T=1.0. Consistent with Arm A. ⚠️ χ hit the cap of 64 here, so this is
a CONSERVATION check and not an accuracy one — on this repo's models the two orderings disagree.

---

1. **Heisenberg `:U1`, L=20** — everything above.
2. **Heisenberg `:SU2`** — `heisenberg_su2_mpo(20)`. Two traps: `maxdim` counts **multiplets**, so
   any cross-symmetry plot must index by `state_bond_dims`, never `maxdim`; and **Néel is not
   SU(2)-representable** (a definite-`Sz` product state spans many total-spin sectors), so this arm
   needs `dimer_state` + bond energies and a re-run of the U(1) arm on the same start to be
   comparable.
### ✅ SU(2) CUBE MEASURED (2026-08-25) — `benchmarks/su2_knob_cube.jl`, L=12, dimer start

The full three-knob cube re-run under `:SU2`, and again under `:U1` on the SAME dimer start so the
two are the same physics. Bond-energy observable against an exact 924-state sector reference.

| depth | τ_trunc | err SU(2) | err U(1) | states | mult SU(2) | mult U(1) | saving |
|---|---|---|---|---|---|---|---|
| m3 | 1e-06 | 4.4692e-07 | 4.4692e-07 | 11 | 5 | 11 | 2.20× |
| m3 | 1e-08 | 4.4948e-07 | 4.4948e-07 | 16 | 6 | 16 | 2.67× |
| m3 | 1e-10 | 4.4948e-07 | 4.4948e-07 | 26 | 9 | 26 | 2.89× |
| mdef | 1e-08 | 4.4953e-07 | 4.4953e-07 | 16 | 6 | 16 | 2.67× |

**Worst relative error difference between the two symmetry modes over all 64 cube points:
9.3e-07, and the STATE counts agree EXACTLY at every point.** SU(2) is the same physics stored in
2.2–2.9× fewer blocks. The knob hierarchy carries over unchanged: saturation at `τ_trunc = 1e-6`
with `m3`/`mdef`, and tightening past it buys rank (states 11 → 26) for nothing.

⚠️ **`krylov` DOES NOT SEE SU(2)'s BENEFIT.** Both modes report identical operator-application
counts (660, 670) — the NUMBER of applications is the same and the TENSORS are smaller. The cost
axis used everywhere else in this study is blind to exactly the saving SU(2) provides, the same
way it was blind to the retired `rexpand`'s extra work. Any SU(2) speed claim needs wall clock
from an isolated run; that is UNMEASURED.

⚠️ **The observable had to change, and this is not a detail.** A dimer has `<S^z_j> = 0` on every
site and SU(2)-symmetric evolution keeps it there, so the chain cube's Sz profile would have
reported a flawless `0.000e+00` for every arm at every tolerance — a measurement that cannot fail.
The bond-energy profile is built through `pair_mpo`, so it is literally the same operator in every
symmetry mode.

3. **Haldane-Shastry** — `haldane_shastry_mpo`, long-range, `O(L)` virtual dimension, and no exact
   real-time reference; graded on energy conservation and the `S(k,ω)` identities.

---

## 8. THE CYLINDER — where the scheme ranking REVERSES

`benchmarks/su2_cylinder.jl`, Heisenberg on a square-lattice cylinder from a dimer start,
bond-energy observable against an exact sector reference.

### FULL RANK (4×3, 12 sites, maxdim 128 — nothing truncates)

| scheme | err | krylov | sec |
|---|---|---|---|
| tdvp2 | **7.2309e-03** | 5287 | 80.4 |
| tdvp_cbe1s | **7.2309e-03** | 4385 | 101.6 |
| cbe_bug defaults | 1.0949e-02 | 552 | 14.2 |
| cbe_bug m=3 | 1.0949e-02 | 394 | 4.2 |
| cbe_bug m=0 | 1.1124e-02 | 123 | 2.6 |

TDVP wins on accuracy by 1.51×; `cbe_bug` m=3 is **19× faster in wall clock** and 13× cheaper.
`states = 64 = 2^(L/2)` is the FULL rank, so nothing truncated and the residual is pure
time-integration error — the long-range leg and wrap bonds, on which a 2-site update is not exact.
Depth spans only 1.6% across the whole m range while the gap to TDVP is 51%: **a plateau, not a
trend.**

### RANK-LIMITED (4×4, 16 sites, maxdim 48 — every arm pinned at the ceiling)

| scheme | err | krylov | sec |
|---|---|---|---|
| tdvp2 | 4.3128e-02 | 7383 | 96.8 |
| tdvp_cbe1s | 4.3128e-02 | 6313 | 131.2 |
| **cbe_bug defaults** | **2.8095e-02** | **772** | **28.9** |
| **cbe_bug m=3** | **2.8385e-02** | **490** | **15.6** |
| cbe_bug m=0 | 3.6936e-02 | 123 | 6.8 |

**THE RANKING FLIPS.** `cbe_bug` is 1.53× MORE accurate than both TDVPs at 9.6× fewer operator
applications and 3.3× less wall clock. Even `m = 0` — one power of `H` — beats both TDVPs
(3.69e-02 against 4.31e-02) at **60× fewer applications**.

⛔ **SO "WHICH `m` IS NEEDED TO BE COMPETITIVE" HAS TWO OPPOSITE ANSWERS, AND THE REGIME DECIDES:**

* **rank-limited: none.** `m = 0` already beats both TDVPs; depth then buys a further 1.3× for 4×
  the cost. This is the regime 2D is actually run in, and the one a rank-adaptive BUG exists for.
* **full rank: no `m` is enough.** The gap is in the scheme's COMPOSITION — one Galerkin solve at
  the root against a per-bond sweep — and depth cannot reach it.

⚠️ Any single-regime benchmark of these schemes is therefore misleading, and the earlier chain
results were all taken at or near full rank.

### ✅ AND THIS SETTLES POINTS 1, 2 AND 4

`err_fnl` is **3.0e-16 / 3.1e-14 / 0.0 even when fully rank-starved** (`states = 48/48` on every
arm). The CBE selection discards essentially nothing *even here*. The rank limit is enforced by
the **closing** truncation, not by the selection — so tolerance-driving the selection (§2),
expanding inside the recursion (§4), and the `_trim_total` ordering defect (§1) all have nothing
to bite on in 2D either. **All three are null for a structural reason: the selection is never the
lossy step.**

⚠️ The first 4×3 run reported "the budget does not bind even in 2D" while sitting at FULL RANK,
where `err_fnl = 0` only means "there was nothing to discard". `su2_cylinder.jl` now computes the
binding ceiling `min(2^(L/2), maxdim)` and prints **"inconclusive — not rank-starved"** rather
than a verdict when the arms did not reach it. That is this repo's own §6 rule — a knob is inert
only if the run is rank-starved — applied to the benchmark itself.

---

## Sequencing

| phase | content | gate to pass |
|---|---|---|
| 0 | reference + harness at L=20; reproduce the insensitivity | `t=0` gate exact; knee visible in Plot 0a |
| 1 | ~~§6 root-cutoff switch~~ ✅, ~~sweep comparison + Krylov basis~~ ✅ **shipped as the only sweep**, then Experiments 6A/6B/6C at L=20 | 6a monotone or explained; peak working set checked |
| 2 | §3 full SVD, then §1 DD projection | no accuracy regression; `comp_ratio` gone |
| 3 | §2 tolerance-driven selection | `err_fnl` and `discarded` both `≲ τ_trunc` (Plot 2b) |
| 4 | §5 Krylov tolerance | depth drops, error unchanged (Plot 5b) |
| 5 | §4 expansion inside Lanczos | Plot 4c beats `grow_iters = 0` at equal cost |
| 6 | §7 SU(2), Haldane-Shastry; re-introduce rSVD | rSVD matches full SVD at fixed `τ_trunc` |

§6 is promoted to **phase 1** — ahead of the algorithmic changes — because it needs no new
algorithm, it answers a question that is already on the table, and its root-cutoff switch is a
prerequisite for stating that any *later* result is tolerance-governed.

**Every phase re-runs the Phase-0 plot set**, so what each change bought is visible rather than
argued.

---

## Operational constraints

- **L=20 timings go to the cluster.** Locally this is a correctness exercise only; the cost axis
  quoted in every plot is **operator applications**, which is contention-immune. Wall clock is
  reported only from an isolated cluster run — local wall clock has been measured to spread 2.6×
  on bit-identical work, and one contaminated timing column is worse than a missing one.
- **One timing job at a time.**
- **Uncapped `maxdim` is what makes the `τ_trunc` scan meaningful**, and it gets expensive as the
  Néel quench's entanglement grows linearly in `t`. Start at `T = 2.5` (≈50 steps) and extend once
  the ranks are known; record where a cap had to bind and exclude those points from the knee fit.
  ⚠ Experiment 6A runs with **no root truncation at all**, so its ranks will be the largest in the
  plan — run its loosest cutoffs first and let the measured `χ(t)` set the budget for the rest.
- Julia block-buffers stdout to a file — every diagnostic print needs an explicit
  `flush(stdout)`, or a long run is a zero-byte log with no way to tell slow from hung.
- Plot scripts run under **`python3`** (`python` has no matplotlib here).

## Files

| what | where |
|---|---|
| driver ✅ | `benchmarks/heisenberg_tolerance.jl` — `phase=tau｜dt｜split｜grid` |
| exact reference | `benchmarks/exact_sparse.jl` — already present, `heisenberg_sparse` + `neel_vector` + `expv_sparse` |
| plots ✅ | `benchmarks/plot_heisenberg_tolerance.py` |
| cluster ✅ | `benchmarks/submit_heisenberg_tolerance.slurm` — `--array=0-3%1`, one phase per task, serialised |
| code change ✅ | `src/RSVDCBEBondUpdate/sweeps/cbe_bug.jl` — `root_cutoff` / `root_maxdim`, defaulting to **no truncation** at the root core split |
| code change ✅ | same file — **one sweep only**. `kstep`, `kaug`, `rexpand` and `basis_frac` removed; the half-sweeps build a Krylov basis (`krylov_basis = 30`, `krylov_tol = 1e-6`) instead of evolving a one-site tensor per bond. Better on every model tested, at 3.3–10× fewer operator applications |

`benchmarks/heisenberg_study.jl` was an earlier draft of the same harness and has been **deleted**
rather than left alongside: two near-identical Heisenberg drivers is exactly the ambiguity this
repo's own docs warn about. Its `t=0` gate and exact-reference plumbing live on in
`heisenberg_tolerance.jl`.
