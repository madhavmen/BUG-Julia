# Handoff — MPO layer, long-range H, and the three CBE-BUG sweep variants, as of 2026-08-17

Written for an agent picking this up cold. Scope: everything added to
`src/RSVDCBEBondUpdate` since the L=18 benchmark campaign — a genuine MPO layer, long-range
Hamiltonians, the `jan_bug` variant, and the root-position/lossless investigation.

Companion documents:

- `docs/current_work.md` §12.0 / §12.2 — the root table and the lossless variant, with derivations.
- `docs/HANDOFF.md` — the **earlier, separate** L=18 benchmark campaign handoff. Its §7 (traps)
  and §8 (hard constraints) still apply verbatim; its §6 (open work) is superseded by §7 here.
- Sweep audit (rendered): https://claude.ai/code/artifact/68a294fe-dcdb-49d1-a152-ff719ed1a241
  — Part 0 shared machinery, walkthroughs of all four sweeps with bond rails, seven findings.

`C:\BUGJulia` is a **different repository** (`feature/discarded-bug-global-sweep`, the
`discarded_bug` work) and has no `RSVDCBEBondUpdate`. Nothing here needs mirroring into it.

---

## 1. One-line status

The MPO layer and long-range Hamiltonians are done and tested. The `jan_bug` variant works and
its error has been **fully decomposed by measurement** — the answer is that where the single kept
Galerkin step sits dominates everything else, and the cheapest correct configuration is
`lossless = true, root = centre`. That configuration is **machine-exact at full rank but breaks
down at `maxdim = 4`, and that breakdown is undiagnosed.** `cbe_lubich` remains the better
scheme at every rank actually worth running.

**Nothing is committed.** See §2.

## 2. Working-tree state — everything is uncommitted

Branch `main`, base `80570b1`. `git status`:

```
 M benchmarks/cbe_probe_variants.jl
 M docs/cbe_lubich_sweep.tex
 M docs/current_work.md
 M src/RSVDCBEBondUpdate/RSVDCBEBondUpdate.jl
 M src/RSVDCBEBondUpdate/cbe_core.jl
 M src/RSVDCBEBondUpdate/cbe_lubich.jl
 M src/RSVDCBEBondUpdate/modes/ground_state.jl
 M tests/RSVDCBEBondUpdate/runtests.jl
 M tests/common/dense_reference.jl
?? src/RSVDCBEBondUpdate/jan_bug.jl
?? src/RSVDCBEBondUpdate/mpo.jl
?? tests/RSVDCBEBondUpdate/test_jan_bug.jl
?? tests/RSVDCBEBondUpdate/test_long_range.jl
?? tests/RSVDCBEBondUpdate/test_mpo.jl
?? benchmarks/long_range_error.jl
?? benchmarks/root_position.jl
?? benchmarks/chi_scan_maxiter8.log, benchmarks/l18_dw.{json,log}   ← result files, never committed
```

`+641 / -86` across the tracked files. **No branch has been created and nothing has been pushed**
— the user was away from the machine and outward-facing actions were deliberately held.

## 3. What was added

### 3.1 `src/RSVDCBEBondUpdate/mpo.jl` — genuine MPO + rank-3 environments

Eq (1.3) of arXiv:2606.28169 as an actual `MPO`, with Eq (1.8) rank-3 environment recursions
(`left_env_stack`, `right_env_stack`, `boundary_channels`, `push_left_channels`,
`push_right_channels`). The sweeps are polymorphic in the `H` representation; the channel
(`XXZChain`) path is kept as an elementwise witness that the MPO path agrees.

**Long-range, added in this round.** `LongRangeTerm`, `long_range_mpo`, `long_range_zz_mpo`,
`LongRangeFit`, `default_decays`, `fit_long_range`, `long_range_terms`, `power_law_zz_mpo` — all
exported (`RSVDCBEBondUpdate.jl:113-114`).

- **A geometric tail is ONE self-loop channel and it is EXACT.** `Σ_{i<j} c λ^(j-i-1) A_i B_j`
  needs no per-distance channel: give the half-open channel a `λ·I` self-loop, and a path that
  opens at `i`, loops over `i+1…j-1` and closes at `j` accumulates `λ^(j-i-1)`. So
  `w = 2 + |nn| + |lr|`, **constant in `L` and in the interaction range**. `λ = 0` reproduces the
  nearest-neighbour MPO elementwise.
- **Hard limit: charge-neutral (rank-2) operators only.** A rank-3 operator (`Sp` under U(1), `S`
  under SU(2)) puts its op-leg on the *virtual* leg, so the channel's virtual space is charged and
  the self-loop would need the identity on a charged space, not `λ·Iid`. Telum's `getIdentity`
  flips input arrows and reports duals (`sectors.jl:77`). `long_range_mpo` **throws** rather than
  emitting a wrong block. `Sz Sz` covers the long-range Ising/XXZ family.
- **Power laws have no finite-`w` MPO.** `fit_long_range` / `power_law_zz_mpo` return
  `(MPO, LongRangeFit)`, and the two-tuple is deliberate: the fit error is a property of the
  *Hamiltonian being simulated*, is not separable afterwards, and quoting a result without it
  quotes an unstated model. Decay rates are **prescribed (log-spaced), not fitted** — solving for
  `λ_k` too (Prony/matrix-pencil) is sharper but ill-conditioned as `n` grows and returns complex
  rates, which would make every MPO tensor complex for a real Hermitian `H`.
- Gotcha when reading fit errors: with `n_exp >= L-1` the fit *interpolates* every distance
  exactly (L=6 has 5 distances, so `n_exp = 6` gives 4.7e-16 — not a statement about power laws).
  Use `L >> n_exp`.

### 3.2 `src/RSVDCBEBondUpdate/jan_bug.jl` — the third variant

One CBE basis sweep across the whole chain; at each bond the full KLS step is taken and then the
evolved bond is QR'd with `R` **discarded**, so the basis moves where the dynamics points but no
evolution accumulates; the S step is **kept** at exactly one bond; one truncation round-trip at
the end. `jan_bug!` alternates direction every step.

Knobs beyond the CBE ones: `root` (which bond keeps the step), `grow_iters` (expanding-basis
Lanczos at the kept bond only), `lossless` (no QR-discard at all — see §5), `truncate = false`
(watch the rank ratchet).

### 3.3 `cbe_core.jl` — `two_sided` removed entirely

The probe now always goes on `H·Θ`: `Y = (HΘ)Ω_R'`, `Q_L = orth(P⊥ Y)`, so the side being
expanded stays exact and the candidates estimate the range of `P⊥(HΘ)`. The removed variant
folded both Gaussians onto the projectors and ranked by the small `M = Q_L†(HΘ)Q_R`. **Why it was
removed, measured:** `M` couples the sides, so at a boundary bond the outer side is a single site,
`room_r = d − r` hits zero at `r = d`, and the branch declined to expand the bond **at all** —
precisely the bond `jan_bug` takes its step on. XXZ L=4/6: first order instead of second,
`2.0e-2` instead of `1.8e-5` over eight steps against a standstill of `2.2e-2`, rank never left
χ=1. The dead `_assemble` went with it, and a prose block at `cbe_core.jl:61-76` records this so
it is not re-tried.

### 3.4 `cbe_lubich.jl` — `root` and `grow_iters`

`root = nothing` keeps the default centre `max(1, min(L-1, L÷2))`. `grow_iters > 0` switches the
root solve from the 0-site `expv` to `_expanding_krylov`, which returns **wider frames**, so the
assembly uses the returned `Wc`/`Zc` rather than `W[c]`/`Z[c]`.

### 3.5 Tests and benchmarks

- `tests/RSVDCBEBondUpdate/test_long_range.jl` — **144/144**. Virtual dim constant in `L`; energy
  matches an explicit pair sum at every distance; the tail is really there; `λ=0` IS the NN MPO
  elementwise; one full-rank step reproduces `exp(-iHdt)`; integrators track the propagator;
  charged channel refused; multiple LR terms coexist; fit error shrinks with `n_exp`; the MPO
  carries the *fitted* H exactly; the fit error propagates and a 4× finer `dt` does not cross it.
- `tests/common/dense_reference.jl` — `dense_zz_tail`, `dense_xy_chain`, `dense_long_range_zz`,
  `dense_power_law_zz`. Explicit pair loops sharing **nothing** with the automaton, which is the
  only reason they are a reference.
- `benchmarks/long_range_error.jl` — dt-order on the exact MPO, error at fixed rank, the fitted
  power-law floor, a long-range control.
- `benchmarks/root_position.jl` — the root-position vs discard decomposition of §4.

## 4. PROVEN by measurement — do not re-investigate

L=6, long-range ZZ MPO, full rank, `Sz=0` sector dimension **20**, one step at `dt=0.01` unless
stated (`benchmarks/root_position.jl`).

**The root position is 82% of jan's error; the discarded `R` is 18%.** The kept step is a Galerkin
problem in `span(U_ex) ⊗ span(V_ex)`. Three regimes — not centre-vs-boundary:

| regime | ceiling `b_L·b_R` | error | note |
|---|---|---|---|
| centre | ≥ 20, reached | **8.5e-16** | frames already span both halves ⇒ projector is the identity ⇒ the step IS `exp(-iH dt)` |
| off-centre interior | 64 ≥ 20, **half reached** (`b_L = 8` of 16) | 1.227e-5 | exactness *reachable*; `grow_iters` 0→2→4 gives `1.227e-5 → 2.049e-8 → 1.8e-15` |
| boundary | `8×2 = 16` **< 20** | 1.745e-5 | unreachable at any budget; `grow_iters` climbs rank 4→12 and the error is **bit-identical** |

**Ruled out, decisively:**

- **Krylov depth.** Flat to 7 significant figures from `maxiter = 4` to 60; unmoved by `tol` over
  six orders. The clincher is that `cbe_lubich` at root 3 vs root 5 uses the *same* solver and one
  is machine-exact.
- **Rank.** `2.935e-4` at `maxdim` 4 / 8 / 16 / 32 alike — a structural error, not a rank error.

**A discard is only harmless BEFORE the evolution.** Keeping the S step at bond `c` and letting
the QR-discard sweep continue past it makes every later bond project the step just taken: error at
kept bond 5 / 4 / 3 = `2.1e-5 / 7.7e-4 / 5.0e-3`, and the **dt-order collapses to 0** (the scheme
becomes inconsistent). This is why `lossless` stops the sweep at the root.

**Order.** jan's *local* one-step error is O(dt²) ⇒ **global order 1** at fixed `T`. An earlier
claim of "second order" in this work confused the two; the test file's header comment still says
`p = 2.00`, which is the local exponent. The norm defect is O(dt²) **by design** (the
`_absorb_left!` projection is not norm-preserving), not a bug.

**The two expansions are mutually load-bearing.** `_basis_left` contracts `U_ex`'s bond leg away
and leaves `b_R`, so the new **left** rank is bounded by `min(d·r_l, b_R)` — the *right*
expansion's width. Mirror on the left sweep. `V_ex`'s *vectors* never reach the state at a
non-kept bond — `_absorb_left!` re-expresses using the original `psi[i+1]` — but its *width* is
what makes `S1` wide, and a wide `S1` is the only reason `orth(U_ex·S1)` has more columns than
`U0` did. Expansion alone is lossless, hence rank-preserving; the rank comes from
`H_eff S = LS + SR + Σ_t ol_t S or_t` acting from **both** sides.

Measured, L=8 domain wall warmed 6 steps, right sweep instrumented bond by bond
(`scratchpad/which_side_caps.jl`):

| bond | `r_l` | `d·r_l` | `b_L` | `b_R` | achieved | `cap` | binding |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2 | 2 | 4 | **2** | 2 | `d·r_l` — saturated |
| 2 | 2 | 4 | 4 | 7 | **4** | 4 | `d·r_l` — saturated |
| 3 | 4 | 8 | 8 | 11 | 6 | 8 | `d·r_l` |
| 4 | 6 | 12 | 12 | 12 | 10 | 12 | tie |
| 5 | 10 | 20 | 11 | 8 | 6 | 8 | `b_R` (right expansion) |
| 6 | 6 | 12 | 8 | 4 | **4** | 4 | `b_R` — saturated |

So **which** cap binds varies along the chain: the site dimension on the left half, where the state
is still narrow and the right block large; `b_R` on the right half, where it is hit exactly at
bond 6. The *necessary* condition is unconditional, though — with no right expansion `b_R = r`, so
`cap = min(d·r_l, r) = r` at **every** bond and the rank is frozen chain-wide. That is precisely
the `two_sided` failure mode above, and bonds 1–2 show the mechanism from the other side: `b_R` of
4 and 7 against `r_l` of 1 and 2 is the headroom that lets the rank double per bond.

Corollary for anyone tempted to halve the CBE cost by probing one side only at non-kept bonds:
**it cannot be done.** The saving is real (two sketches and four SVDs per bond, of which only one
branch's descendant reaches the state) but the right probe is not overhead — it is the pump.

## 5. The `lossless` variant — best cheap configuration, with a defect

`lossless = true, root = centre`. No S step away from the root, so nothing is discarded and
nothing is projected: a pure CBE expansion **carried** bond to bond (`C = U_ex'(C·psi[i])`) plus
one KLS at the centre. **1.203e-14 over 24 steps** against the QR-discard variant's `2.935e-4`.
Needs no `grow_iters`.

Two implementation facts that are load-bearing:

- **The carry is mandatory.** Writing the raw expanded frame back into the state leaves a
  zero-padded remainder in `psi[i+1]` (the new directions have no weight yet) which the next
  bond's `_frame_from` SVDs away again. L=18 froze at maxbond 2 that way. The frames are parked in
  `Wf[]` and committed only after the root solve.
- **The centre is structural, not merely best.** A forward sweep refreshes the left half and a
  reverse sweep the right, so an alternating pair covers the chain; a boundary root's reverse sweep
  runs over one bond and the alternation degenerates (`5.8e-2`). This is Lubich's K/L split reached
  from the other direction — but with ONE half-sweep and one env stack per step.

### ⛔ The half-cost economy does NOT survive truncation

Full-rank equality was misleading. L=8, T=0.2, 20 steps, error vs `maxdim`:

| variant | χ=4 | χ=8 | χ=16 | χ=32 | χ=64 |
|---|---|---|---|---|---|
| jan, QR-discard, far root | 2.59e-4 | 2.59e-4 | 2.59e-4 | 2.59e-4 | 2.59e-4 |
| jan, `lossless`, centre | **9.999e-1** | 6.56e-10 | 3.80e-10 | 3.80e-10 | 3.80e-10 |
| `cbe_lubich` | 1.558e-6 | 5.35e-10 | **3.67e-12** | 3.67e-12 | 3.67e-12 |

So `lossless`+centre beats QR-discard by six orders, but **`cbe_lubich` beats it by ~100× at
χ ≥ 16** — the second half-sweep buys real accuracy — and at **χ=4 `lossless` breaks down**
(error ≈ ‖target‖, i.e. essentially no state left) where lubich is still at 1.6e-6. That is a
**defect, not a trade-off, and it is UNDIAGNOSED.** Do not use `lossless` at small `maxdim`.
`cbe_lubich` remains the scheme to prefer.

An earlier conclusion in this work — "same accuracy, half the work" — was **wrong**; it
generalised from L=6 at full rank, where the variable is invisible. The table above overturns it.

## 6. What is NOT known — do not claim these

- **Why `lossless` collapses at χ=4.** The top open item.
- **The full suite has not been re-run since the last test edit.** The `n_new` assertion in
  `test_jan_bug.jl` was weakened after the run that reported it (see §8), so its current state is
  unverified.
- **Time symmetry.** The adjoint test `Ψ_h ∘ Ψ_{-h}` vs the identity was offered and never run.
  Until it is, whether order-4 composition is available for these variants is open — and note the
  standing constraint that a **backward substep is banned for dissipative problems**, so a
  triple-jump is not automatically usable even if the test passes.
- Any **timing or arm-vs-arm ranking**. None was attempted here. `docs/HANDOFF.md` §3 explains why
  no valid timing data exists for any arm.

## 7. Open work, in priority order

1. **Diagnose the χ=4 `lossless` collapse** (§5). Start by checking whether the carry `C` or the
   parked-frame commit survives an aggressive truncation of the *previous* step's output — the
   `Wf[]` commit re-imposes tags on frames derived from a state that the closing truncation has
   since narrowed.
2. **Register `test_long_range.jl` in `tests/RSVDCBEBondUpdate/runtests.jl`.** 144 passing
   assertions currently sit **outside** the suite; an earlier edit adding it was rejected before
   the file was finished and was never re-applied. Put it after `test_mpo.jl`.
3. **Re-run both suites.** They are disjoint: `tests/runtests.jl` covers `BondUpdateBUG`,
   `tests/RSVDCBEBondUpdate/runtests.jl` covers this module. Budget ~45 min and ~15 min.
4. **Decide Finding 04 — the structural cause of the ~100× gap.** One pass cannot widen the
   root's right frame, because that frame is built from the *original* `psi[c+1]`. Two options,
   both honest: (a) add a reverse lossless pass before the root — this **reproduces Lubich** and
   the cost saving disappears; (b) accept a one-sided root frame, which is ~100× worse at moderate
   rank and unsafe at low rank. This may also resolve item 1, so consider them together.
5. **Commit and branch.** Nothing is committed. Suggested split: (i) MPO long-range + tests,
   (ii) `two_sided` removal, (iii) `jan_bug` + root/grow/lossless + tests, (iv) docs. Commits
   authored Madhav, **no Claude co-author line**; push only on the user's confirmation.
6. Optional: fix the ten pre-existing gate-path sketch errors (§8).

## 8. Known-red tests, and they are NOT this work's fault

`test_sketch.jl` has **2 errors** (10 across the suite in the fuller count), in the *test-side*
helper `reference_sketch_right` → `perp_component_right` — an itag mismatch on the **gate** path.

**Verified pre-existing** by checking out a clean HEAD `git worktree` (at `/c/bl`, because a
worktree inside the scratchpad path failed with `Filename too long`) and running the same file:
identical `111 pass / 2 errors`. Do not spend time attributing these to the MPO or jan work.

Leftover: `git worktree remove` failed with `Permission denied` on `.git/worktrees/bl`. `/c/bl`
was deleted and `git worktree list` shows only the main worktree, so it is effectively pruned —
harmless metadata.

Also weakened during this work, deliberately: `test_jan_bug.jl`'s expansion assertion went
`all(>(0), info.n_new)` → `all(>(0), n_new[2:L-1])` → finally `any(>(0), info.n_new)` plus an
`@info` log, after two failures. The reason is real and worth keeping: **which bonds saturate
depends on the warm-up state**, so a per-bond claim is not a property of the scheme. The testset's
actual claim — the update bond gets expanded — is what is now asserted. Norm assertions similarly
moved from `abs(n-1) < 1e-6` to asserting the O(dt²) **scaling** (`1.7 < p < 2.3`), because the
defect is by design.

## 9. Traps hit during this round

Each of these produced a confident wrong answer.

- **Quoting the local order as the global order.** jan is locally O(dt²) and globally order 1.
  Everything built on the "second order" reading — including triple-jump and Richardson advice —
  was retracted.
- **Using a pre-expansion ceiling as if the expansion reached it.** Predicted roots 2 and 4 exact
  because `b_L·b_R = 64 ≥ 20`; measured 1.227e-5. `benchmarks/root_position.jl` showed roots 4/5
  reach only *half* their ceiling. **Measure the achieved dimension, never the ceiling.**
- **Answering the wrong question about Krylov.** "More Krylov vectors" meant `_expanding_krylov`
  (regrow the subspace around each vector), not `maxiter` (more vectors in a *fixed* subspace).
  They are different mechanisms and only the first can help.
- **Generalising a fixed-rank conclusion from a full-rank run** — see §5. At full rank the rank
  variable is invisible, so an equality there says nothing about truncated behaviour.
- **Proposing a fix aimed at the small term.** A "union-QR" (`orth([U_ex·S1 | A_i])`) targeted the
  18% discard term when the decomposition said 82% was root position. The measurement said don't
  bother, and it didn't get built.
- **Julia soft scope in scratchpad scripts.** A top-level `for` loop reassigning a global
  (`lch = push_left_channels(...)`) creates a *new local* and throws `UndefVarError`. Wrap probe
  scripts in a `function main()`.
- **Empty Julia logs are block buffering plus first-call JIT**, not a hang. Background with a long
  timeout.

## 10. Hard constraints still standing

From `docs/HANDOFF.md` §8, unchanged and still in force:

- **`cbe_rsvd` IS the method.** Rank grows *only* via RSVD-CBE. No random sector seeds, no
  `missing_fill`, no `pad`, no KLS augmenter. Exact CBE is an A/B reference only.
- **Do not modify `src/BUG` / `src/BondUpdateBUG`** without an explicit request.
- Naming: **`bond_update_bug`**, never "two-site BUG". The user's names for the variants are
  **Gate-Based-BUG**, **2-Site-BUG**, and for the sweeps here: "Mendl BUG" = the MPS version of
  the Lubich BUG = `cbe_lubich_bug!`.
- `growth = 2.0` is a **per-problem starting point**, not a default to trust; re-derive it from
  `err_fnl` (decision table in `docs/current_work.md` §7.1).
- **Local runs are small sizes only** (`L ≲ 12`) and the question there is **correctness against an
  exact reference**, never ranking or timing. `L ≥ 16`, sweeps, and all wall-clock go to the
  cluster via `sbatch`. **One timing job at a time.**
- Commits authored Madhav, **no Claude co-author line**. Branches pushed only on confirmation.
