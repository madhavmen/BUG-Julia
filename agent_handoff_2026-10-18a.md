# Agent handoff — imaginary-time / open-system CBE-BUG plan (2026-10-18)

- **Repo:** BUG-Julia.
- **Branch:** `plan/imag-open-t3` (Task 3), forked from `feature/krylov-closure-order`, which
  already holds the merged Tasks 1 and 2.
- **Working tree:** clean; everything is committed.
- **Previous handoff:** `agent_handoff_2026-10-17.md`, about CBE-BUG vs 2-site TDVP real time
  under the fair settings. Its standing rules (§0) still apply and are summarised in §1.
- **The plan:** `cbe_bug_imaginary_open_benchmark_research (1).md` (USER 2026-10-18: "with what
  was understood now about the tolerances for the rsvd, augmentation and s step, we proceed with
  plan …"). It has seven tasks, done in order:
  1. imaginary-time infrastructure;
  2. MPDO infrastructure;
  3. IT-1 Heisenberg chain;
  4. IT-2 square cylinder;
  5. OS-1 driven XX + dephasing;
  6. OS-2 Bose-Hubbard + two-body loss;
  7. OS-3 U(1) East + dephasing.

---

## 1. Standing rules (binding; details in `agent_handoff_2026-10-17.md` §0)

- **Run nothing locally in Julia.** Python grading and plotting locally is fine; there is no
  pandas, so use `csv` + matplotlib.
- **Cluster jobs:**
  - Read `../cluster_compliance.md` in full before every `sbatch` and walk its 9 gates.
  - `--partition=small,cluster --nodes=1 --cpus-per-task=4`, `julia -t 4`,
    `--constraint=x86-64-v3`, `--time=7-00:00:00`.
  - Exclude the cip nodes and th-cl-dell03.
  - `--mem` from a measured MaxRSS + 10–20 %. A cold-PKG compile peaks at about 5 G.
  - Output under `$SCRATCH/rsvd_adaptive/<date>-<tag>/`, with a PROVENANCE file.
  - Gate 9 afterwards: `seffx -j`, the OOM grep, and the `EXIT-…=` grep.
- **One timing job at a time.** Accuracy-only jobs may run concurrently.
- **Staging (CHANGED this session):**
  - Build the tarball with `git -c core.autocrlf=false archive --format=tar -o X.tar HEAD`, then
    `scp X.tar lmu-hpc:bugjl/` and run `~/bugjl/stage.sh ~/bugjl/X.tar <name>`. It prints `SNAP`
    and the content-addressed `PKG`.
  - Do NOT tar the working tree. A `git checkout` rewrites files with CRLF on this OneDrive
    checkout, so identical code gets a new PKG hash and a pointless 10-minute cold compile.
  - A `src` change still needs one cold compile.
- **Test files:**
  - Run several files in one process with
    `$SCRATCH/rsvd_adaptive/run_tests.jl f1.jl f2.jl …` from SNAP, with `--project=$PKG`.
  - Run only the tests relevant to the change.
  - No `const` inside a top-level `@testset` (it is local scope).
- **`@printf`:**
  - The format must be ONE literal.
  - `$SCRATCH/rsvd_adaptive/gate_printf.py` checks the specifier count statically and cannot
    count a splat (`row...`), so spell the arguments out.
- **Git:**
  - Use explicit paths with `git commit`. `-am` swept two externally deleted handoff files into
    `655b53d`; they are recoverable from `f588ef9`.
- **Fairness (USER 2026-10-16/17):**
  - bug2 ↔ `tdvp2_step!` at the same dt.
  - Identical per-cut discard rule (`_cut_divisor`; tdvp2 via `weight_tol = tol_trunc`).
  - `tol_trunc = tol_root (= tol_expv) = 1e-6`; `tol_basis` is the knob.
  - Grade against exact / reference before claiming faster or rank-better.
  - Report a flagged (`:failed`) step as a failure, never as a converged point.

## 2. Task status

### Task 1 — imaginary-time infrastructure: MERGED `3d94792`

- **Confirmed by:** s28test 17340070 (regressions) and s31test 17340175 (IT-U1..U6).
- **Root methods:** `expv(...; root_method = :exp | :trapezoid | :radau2 | :radau3)`, in
  `src/BondUpdateBUG/expv.jl`.
  - Each implicit method's stability function (Padé) is applied once on the Lanczos/Arnoldi space
    after shifting to the extreme Ritz value, so z ≤ 0 and the method is A-/L-stable.
  - `tol_expv` is the FOM tail.
  - Helpers: `root_order`, `root_stability`.
- **Energy shift:** `shift_h` / `ShiftedH` (`mpo.jl`) wrap only the generators: the root, and the
  half-sweep and closure one-site Krylov operators. `eshift = 0` is bit-identical. `KLFrames`
  carries `eshift`.
- **`_kry_weight` (`cbe_bug.jl`)** in imaginary time is now relative to the propagated seed, and
  shifted. Before, it over-weighted directions by e^(dβ|E|) and overflowed. Real time is
  unchanged.
- **`cbe_bug_step!(...; root_method, eshift, root_report = Ref(), outside_residual)`.**
  - The report holds `e_pre`, `e_root` and the outside-Galerkin defect
    `|τ| ‖(I−P)Hψ₁‖ / ‖ψ₁‖`, measured with `estimate_global_complement_action`.
- **`sweeps/imaginary_time.jl`:**
  - `ite_step!`: renormalises; Eshift = the state's energy; logs `e_before`, `e_pre_root`,
    `e_root`, `e_after` (`de_root` / `de_compression`); outside defect vs `tol_basis`. The retry
    ladder is re-enrich (`closure_checks`) → split dβ → `:failed` (or `on_fail = :error`).
  - Also `ite_evolve!`, `log_panel_steps(beta_max, P; per_panel = 4)` (arXiv:2606.02930),
    `uniform_beta_steps`, `beta_grid`.
- **Findings:**
  - The order test needs the DOMAIN WALL. From Néel, p = 1 and p = 2 are bit-identical.
  - Observed slopes are p+1 (1.95 / 2.96 / 3.97 / 5.0).
  - Radau2 caps the order at 3, the trapezoid at 2.
  - ITE equals dense E(β) to 5e-11.
  - **The first step from a product state is flagged `:outside`:** χ = 1 frames cannot hold Hψ₁
    at any dβ. That is the fail-safe working. **USER decision still pending:** accept-and-label
    (current) or pre-grow the rank.

### Task 2 — symmetry-aware MPDO: MERGED `1aba5b5`

- **Confirmed by:** s33test 17340370 and s34test 17340453 (OS-U1..U7 and the `keep_states` test).
- **`src/RSVDCBEBondUpdate/models/liouville.jl`:**
  - `liouville_space(q; sym = :weak|:strong|:none)` builds the d² site with
    `Telum._localspace_cgt_fields`. Labels are `q_ket − q_bra` (U1), `(q_ket, q_bra)` (U1×U1), or
    a single sector.
  - `liouville_op` refuses a non-covariant superoperator and STORES `transpose(M)`, because the
    MPO / `mpdo_apply!` apply the transpose — the old `jump_transpose` finding, made a rule.
  - Superoperator builders: `ket_super`, `bra_super`, `dissipator_super`.
  - `lindblad_mpo(ls, L; pairs, onsite_h, jumps)` is a `pair_mpo` with complex couplings, with
    the on-site H and dissipators in `field`.
  - States: `mpdo_product_state` (keeps the whole physical sector and cuts the LINK — Telum
    refuses an in-sector slice with `preserve_space`), `mpdo_identity`.
  - Contractions by walking the abelian blocks: `mpdo_trace`, `mpdo_expect`, `mpdo_rdm`,
    `mpdo_dense`, `mpdo_checks`.
- **`cbe_bug_step!(...; keep_states = [mpdo_identity(ls, L)])`** unions |I⟩⟩ into the frames, so
  the Galerkin step is trace-exact: drift 1.1e-15 vs 0.37 without it, at `tol_basis` 1e-2.
- **Findings:**
  - The MPO equals the dense Liouvillian to 2e-16 (XX + dephasing + drive; Bose-Hubbard + loss).
  - The roots equal dense exp/Padé to 1e-15, but only from a FULL-RANK bond: product-state frames
    are not the whole space, even at L = 2.
  - MPDO order slopes are 2/3/4.
  - weak == strong to 2e-17.
  - The purity law holds to 1e-15.
  - The strong total charge reads (−N, −N) on the last link (arrow sign).

### Task 3 — IT-1 Heisenberg chain: IN PROGRESS on `plan/imag-open-t3`

- **Driver:** `benchmarks/imaginary_time/it1_heisenberg_chain.jl` (env `IT_*`, documented in its
  header).
  - Arms: bug2/3/4 via `ite_step!`, and tdvp2 as the fair order-2 partner.
  - Grids: `panel:P:k` and `uniform:n`, the same grid for every arm.
  - References: exact (L ≤ 20: E0, E(β), the correlation row) or DMRG (cached
    `it1_dmrg_L*_D*.jls`).
  - Output CSVs: `it1_steps_L*.csv`, `it1_final_L*.csv`, `it1_ref_L*.csv`.
  - `6a0b595` added an untimed JIT warm-up, and the tdvp2 wall now includes its energy readout.
    The production runs below predate this.
- **Plots:** `benchmarks/imaginary_time/plot_it1.py DIR L --tb 1e-05` draws small multiples:
  trajectory error, E − E0, χ(β), cost-to-accuracy; open rings mark flagged steps.
- **Production 2026-10-18** (`$SCRATCH/rsvd_adaptive/2026-10-18-it1/L{12,16,20}/`; β = 64,
  `panel:9:4` + `uniform:256`; bug2/3/4 × `tol_basis` {1e-3, 1e-5, 1e-7} × seeds 1–3, plus tdvp2;
  maxdim 1024):

  | L | job | outcome | MaxRSS |
  |---|---|---|---|
  | 12 | 17340511 | COMPLETED | 1.35 G |
  | 16 | 17340512 | CRASHED at bug3 `tol_basis` 1e-7 panel, seed 2 | 1.70 G |
  | 20 | 17340513 | CRASHED at bug3 `tol_basis` 1e-7 panel (after seed 1) | 5.98 G of 6 G — use 8 G next |

  - **Crash:** `ArgumentError: selector for sector … on leg 3 out-of-bounds`, in
    `sector_graded_sketch` → `_trim_per_sector`, inside the order-p closure frontier sweep.
  - **Root cause:** the frame's bond-sector dimension was read with the fusion basis's label
    without arrow alignment, so it read the CONJUGATE sector. That fails only once the sector
    dimensions stop being symmetric under q → −q.
  - **Fix `661c3e7`** (`cbe_core.jl`): `align_charge(symm(frame), q, F.inds[3].dir,
    frame.inds[fl].dir)`, plus an assertion that a frame never exceeds its local space. This is a
    real pre-existing bug in shared CBE code, not in the new code.
- **s37test 17341472 (RUNNING at handoff;** `$SCRATCH/rsvd_adaptive/2026-10-18-s37/`; PKG
  `46c903a8ef8f`, cold**):**
  1. Reproducer: L = 16, bug3, `tol_basis` 1e-7, `panel:9:4`, seeds 1–3, in `repro/`.
  2. Regressions: `test_adaptive_rsvd`, `test_adaptive_search`, `test_krylov_closure`,
     `test_imaginary_time`.

  Check `EXIT-REPRO`, `EXIT-TESTS` and the `FILE-` lines.
- **L = 12 results** (local copy and figure in the session scratchpad; regenerate with
  `plot_it1.py`):
  - **Every arm reaches the exact E0 to about 1e-12 at β = 64.** The exception is uniform with
    `tol_basis` 1e-3: BUG stalls at 5–9e-8, a basis floor that the panel grid does not show.
  - **Trajectory accuracy (|E − E_exact(β)|):** on the uniform grid, tdvp2 is ~3e-4 early, bug2
    ~1e-5, bug3/4 1e-7–1e-9. On the panel grid, **tdvp2 departs from the exact trajectory**: a
    6e-6 spike at β ≈ 20, and it reaches E0 by β ≈ 20 while the exact relaxation needs β ≈ 50.
    With panel steps of dβ = 4–8 the projector splitting acts like a DMRG sweep. That is fast
    ground-state search, not accurate integration.
  - **Cost-to-accuracy (indicative only: concurrent jobs, JIT in the first arm):** tdvp2 is
    faster at L = 12 (χ ≤ 38). Panel: tdvp2 34 s to 1e-10, bug 44–53 s. Uniform: tdvp2 144 s,
    bug 205–224 s. The panel grid is 3–4x cheaper than uniform for every arm.
  - **χ(β) is about the same for all arms.**
  - **Flagged steps:** the Néel first step for bug2. Many late steps are flagged at
    `tol_basis` 1e-7 on the panel grid (L = 20 bug3: 14) — the outside test vs very large dβ.
  - **Report both metrics** (ground-state error vs cost, AND trajectory error vs exact E(β)).
    The plan's time-to-accuracy metric alone rewards tdvp2's non-integrator behaviour.

## 3. Next steps (in order)

1. **Read s37test.**
   - If the reproducer completes and the four regression files pass, rerun L = 16 and L = 20
     production on the fixed PKG (`46c903a8ef8f`), with L = 20 at `--mem=8G`. The L = 12 data
     need not be rerun: the fix only changes behaviour where labels were misaligned, but L = 12
     can be rerun for one consistent PKG.
   - If it fails, read the new assertion's message: it names the sector and the dimensions.
2. **Merge `plan/imag-open-t3`** into `feature/krylov-closure-order` only after Task 3's
   deliverables exist. Until then the fix lives on the t3 branch. Or cherry-pick `661c3e7`
   earlier if other work needs it.
3. **Task 3 main size L = 100** (stretch L = 200): the DMRG reference `IT_DMRG_D` (512) is cached
   by the driver. This is a TIMING job: alone, on its own node, using the warm-up driver.
   - Size `--mem` from L = 20 × the χ growth (measure a short test first).
   - Do the plan's grid-refinement / stiffness sweep (backward-step limitation of TDVP) and the
     implicit roots (`IT_ROOT=radau2` etc.).
4. **Plots for all sizes:** `plot_it1.py`, plus a `tol_basis` comparison and the seed spread.
5. **Tasks 4–7** in order: IT-2 square cylinder (4x4, 6x4; main 8x4, 10x4), then OS-1/2/3 with
   `liouville.jl`.
   - OS-3 (East) needs a 3-body term `n_{j−1}(c†_j c_{j+1} + h.c.)`, which `pair_mpo` (two-body)
     cannot build — it needs an MPO builder extension.
   - Strong-U(1) |I⟩⟩ is not an MPS, so `keep_states` there needs an |I_N⟩⟩ construction.

## 4. Other jobs and results from the real-time track (parked by the USER on 2026-10-18)

- **prof24 17339380** (`2026-10-17-prof24/`, 6x4, non-binding cap, `tol_basis` 1e-5):
  - bug2 is 0.72x tdvp2e's speed and carries +11 % χ.
  - bug2 serial step: closure sweep 43 %, Krylov 18 %, CBE 16 %.
  - So Test B incremental re-sketch (a) and Test A's second SVD (b) are small; the closure sweep
    is the cost. Details in `agent_handoff_2026-10-17.md` §8.1.
- **fair10** graded against ref1024t + ref1536t (t ≤ 4.5): at a binding cap bug2 is 40–280x less
  accurate than tdvp2e.
- **ref1536t 17331887:** still RUNNING, step ~79/100 at ~1010 s/step (~6 h left) —
  `2026-10-17-ref1536/`.
- **Phase 6 reruns on the s26 rules** (`2026-10-17-p6s26/square/`):

  | job | arm | state | MaxRSS |
  |---|---|---|---|
  | 17339381 | p6r_tdvp2e_256 | COMPLETED | 2.87 G |
  | 17339382 | p6r_tdvp2e_512 | COMPLETED | 2.99 G |
  | 17339383 | p6r_tdvp2e_1024 | RUNNING | |
  | 17339384 | p6r_cbe1se_1024 | OUT_OF_MEMORY again at 8 G | 8.31 G |

  - p6r_cbe1se_1024 needs ≥ 10 G and is still pending a rerun.
- **Hubbard-track jobs** on the account (`long_r-*`, `rs3*`, `ms*` / `gr*`) are NOT ours — leave
  them alone.

## 5. Where things are

| what | where |
|---|---|
| plan | `cbe_bug_imaginary_open_benchmark_research (1).md` |
| imaginary-time step / grids | `src/RSVDCBEBondUpdate/sweeps/imaginary_time.jl` |
| root integrators | `src/BondUpdateBUG/expv.jl` (block above `_tail`) |
| eshift wrapper | `src/RSVDCBEBondUpdate/mpo.jl` (`shift_h`) |
| MPDO | `src/RSVDCBEBondUpdate/models/liouville.jl` |
| keep_states / root_report | `src/RSVDCBEBondUpdate/sweeps/cbe_bug.jl` (kwargs block, half-sweeps, after root `expv`) |
| sketch label fix | `src/RSVDCBEBondUpdate/cbe_core.jl` (`sector_graded_sketch`) |
| tests | `tests/sweeps/test_imaginary_time.jl` (IT-U1..U6), `tests/sweeps/test_liouville.jl` (OS-U1..U7, keep_states) |
| IT-1 driver / plots | `benchmarks/imaginary_time/it1_heisenberg_chain.jl`, `plot_it1.py` |
| exact sparse helpers | `benchmarks/exact_sparse.jl`, `benchmarks/sparse_krylov.jl` |
| cluster dirs | `$SCRATCH/rsvd_adaptive/2026-10-18-{s28..s37,it1}` |
| memory notes | `imag-open-plan-task1-design`, `imag-open-plan-task2-mpdo`, `bugjulia-cold-compile-fix-staging-and-workload` |
