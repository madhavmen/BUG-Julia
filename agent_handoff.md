# Agent handoff — imaginary-time / open-system CBE-BUG plan (2026-10-18, third handoff)

- **Repo:** BUG-Julia. **Branch:** `plan/imag-open-t3`, HEAD `a3bb0f6` (+ this handoff). It is
  forked from `feature/krylov-closure-order`, which holds the merged Tasks 1 and 2.
- **Working tree:** clean.
- **Earlier handoffs, still valid for detail:**
  - `agent_handoff_2026-10-18b.md` — this session's bugs and fixes (§3 there), task status, file map.
  - `agent_handoff_2026-10-18a.md` — the session start; the standing rules (§1 there).
  - `agent_handoff_2026-10-17.md` — the real-time track.
- **The plan:** `cbe_bug_imaginary_open_benchmark_research (1).md`. Seven tasks, done in order:
  1. imaginary-time infrastructure (MERGED);
  2. MPDO infrastructure (MERGED);
  3. IT-1 Heisenberg chain;
  4. IT-2 square cylinder;
  5. OS-1 driven XX + dephasing;
  6. OS-2 Bose-Hubbard + loss;
  7. OS-3 U(1) East + dephasing.

---

## 0. Two jobs running at handoff: read them first

### s40test 17342415 (`$SCRATCH/rsvd_adaptive/2026-10-18-s40/`)

- **Build:** PKG `9207c690e1c7` = `033225a`, all src fixes of this session. Request `--mem=7G`
  (cold compile).
- **Stages done:**

| stage | result |
|---|---|
| `EXIT-PARSE` / `EXIT-PRINTF` | 0 / 0 |
| `EXIT-REPRO=0` | **The frame fix is confirmed.** IT-1 L=16, bug3, `tol_basis` 1e-7, `panel:9:4`, seeds 1–3, with `IT_FRAME_CHECKS=1`, all complete (this crashed in 17340512, s37 and s38). E − E0 = 0.99e-12 / 1.08e-12 / 1.18e-12; χ 56–58. |
| `EXIT-IT1DW=0` | IT-1 L=12 from the domain wall, β=8, `panel:5:4`, `tol_basis` 1e-6: bug2 E − E(β) 9.0e-4, **0 failed** (13 retried); tdvp2 1.3e-3. |
| `EXIT-OS1=0` | OS-1 L=6, γ=0.3, t=2, dt=0.1: the max error vs the exact covariance is bug2 j 5.7e-5 / m 4.5e-5 (1 step unmet: `projection`, step 1); tdvp2 4.8e-6 / 5.3e-6. **bug2 is ~10x worse than tdvp2**; see §2.3. |
| `EXIT-OS2=1`, `EXIT-OS3=1` | Crashed in MY reference code (`sparse_lindblad.jl`: `isless(ComplexF64, Float64)` in the Arnoldi breakdown test). Fixed in `a3bb0f6`; rerun as s41 below. |

- **Still RUNNING at handoff:** `EXIT-TESTS`, i.e. `test_imaginary_time`, `test_liouville`,
  `test_pair_mpo`, `test_krylov_closure`, `test_adaptive_rsvd`, `test_adaptive_search`.
  - Every testset added this session runs here for the first time: tdvp2 eshift/backward, the
    first-step split, the capped step not split, the OS-3 three-site MPO, |I_N⟩⟩, the OS-1
    reference, and the saturated-frame cap.
  - Grep `FILE-OK` / `FILE-FAILED`, then the `Test Failed` / `Expression` lines in `.out` / `.err`.
  - **If a test fails, read the code before weakening the test** (standing rule).
- **Gate 9 afterwards:** `seffx -j 17342415`, the OOM grep, the `EXIT-` grep.

### s41os 17342676 (`2026-10-18-s41/`)

- **Build:** `a3bb0f6` on the SAME PKG `9207c690e1c7` (a script-only change, so warm). `--mem=4G`.
- **Runs:** the OS-2 smoke (L=3, nmax 2, exact sparse reference) and the OS-3 smoke (L=6, STRONG
  U(1) with |I_N⟩⟩, exact sparse reference).
- **Check:** `EXIT-OS2`, `EXIT-OS3`, and the per-arm error lines.
- **CSVs:** `os2/`, `os3/`; plot with `benchmarks/open_system/plot_os.py`.

## 1. Standing rules

These are unchanged; details in `agent_handoff_2026-10-18a.md` §1.

- **Run nothing locally in Julia.**
- **Before every `sbatch`:**
  - Read `../cluster_compliance.md` in full.
  - `--partition=small,cluster --nodes=1 --cpus-per-task=4`, `julia -t 4`,
    `--constraint=x86-64-v3`, `--time=7-00:00:00`; exclude cip and th-cl-dell03.
- **Memory:** cold compile 5.0 G peak (`--mem=7G`); L=20 production 8 G.
- **Staging:** `git -c core.autocrlf=false archive HEAD` → `~/bugjl/stage.sh`. A script-only change
  keeps the PKG (no recompile).
- **Other:** explicit paths with `git commit`; one timing job at a time; fairness as before
  (`tol_trunc = tol_root = 1e-6`).

**USER decisions on 2026-10-18:**

- The Néel first-step flag is a BUG. Fixed: `576a46b`, the defect-sized split.
- Use a domain-wall initial state as well (`IT_INIT=dw`).
- OS-3 is strong U(1), as the plan states (`mpdo_identity_sector`).

## 2. Open findings: investigate BEFORE production

### 2.1 Flagged steps in the L=16 reproducer (s40)

- **Counts:** seeds 1/2/3 have 9/13/9 failed steps (6/3/6 retried), yet every run converges to
  ~1e-12.
- **Reasons, from `repro/it1_steps_L16.csv`:**
  - **`projection`:** steps 13–20 (dβ 0.25–0.5), steps 29/32 (dβ 4) and step 33 (dβ 8).
  - **`krylov+closure`:** steps 34–36 (dβ 8), the last panel.
- **Suspected cause of `projection`: the check's own noise floor.**
  - `proj_err = sqrt(||psi||² − ||S0||²)/||psi||` (`cbe_bug.jl`, near
    `ledger.proj_err > max(budget.basis, _PROJ_FLOOR)`) is a difference of squares, so its
    roundoff floor is ~sqrt(eps) ≈ 1e-8.
  - The threshold at `tol_basis` 1e-7 is `max(1e-7, _PROJ_FLOOR = 1e-7)`, only ~5–10x above that
    floor.
  - On the ordinary (non-midpoint) step the frames contain psi by construction.
  - **Not yet proven.** Next: log `proj_err` per step (add it to the driver row), and compute
    `||(1−P)psi||` DIRECTLY (overlap of psi with the frames, or a perp contraction) to compare with
    the cancellation formula.
  - If it is the floor, the fix is the direct measurement, not loosening the threshold.
- **`krylov+closure` at dβ = 8:** the Krylov / closure tests unmet on the largest panel steps.
  - These split (they are not capped), so they are probably the large-step regime.
  - Check `nsub`, the resid and `workcap_reached` in those rows before deciding.
- **Prior state:** these flags were also present in the earlier production (L=20 bug3 had 14
  flagged late steps), so they predate this session's changes.

### 2.2 Same-pattern audit of the frame cap (open)

- `_cap_to_local_room` is applied in `_extend_frame` only.
- `cbe_expand`'s `_reortho_left/right(U0, TLC/TRC)` and `_union_left/right` (keep_states) use the
  same globally relative Telum cutoff.
- They never tripped `_FRAME_CHECKS` in s38/s40, but that is not proof.
- Applying the cap there costs one `fusion_basis` per call (timing impact).
- **Decide after the §2.1 investigation.** Running a production job with `IT_FRAME_CHECKS=1` is a
  cheap way to watch for it.

### 2.3 OS-1: bug2 ~10x less accurate than tdvp2 at dt 0.1 (s40 smoke)

- **Data:** L=6, γ=0.3, `tol_basis` 1e-6, `tol_trunc` 1e-6; one `projection` unmet at step 1 (the
  same check as §2.1, here at threshold 1e-6).
- **Unknown yet:**
  - whether this is the time-step error (compare against bug3/bug4 and smaller dt);
  - the basis tolerance (try `tol_basis` 1e-8);
  - or a defect of the MPDO / keep_states path.
- **Plan rule:** grade against the exact reference at matched error before any claim.
- **Next:** an `OS_DTS=0.1,0.05,0.025` × bug2/3/4 × `tol_basis` {1e-6, 1e-8} run at L=6. It is
  exact and cheap, and gives the order slopes and the floor.

## 3. What this session fixed (details: `agent_handoff_2026-10-18b.md` §3)

| commit | what |
|---|---|
| `033225a` | `_cap_to_local_room`: the frame non-isometry behind the IT-1 L=16/20 crash. Telum's `svd` cutoff is relative to the GLOBAL σ_max across sectors, so roundoff in a saturated sector became fake columns in `_extend_frame`. **VERIFIED by s40's reproducer.** |
| `576a46b` | `ite_step!` splits by the measured outside defect (`_ite_nsplit`, defect ∝ dβ^(p+1), from fp1 job 17342386), and never splits a cap-bound step (`_ite_capped`). This is the Néel first-step "failure". `root_report.per_left/per_right`. |
| `6ceb0eb` | `tdvp2_step!(...; eshift, backward_report)`: the shift avoids overflow at large L, and the plan's backward-substep amplification metric is reported. |
| `a19c027` | `_FRAME_CHECKS` / `IT_FRAME_CHECKS` debug assertions at every frame-growth site. |
| `4288d32` | `TripleTerm`: three-site MPO terms (the OS-3 East hop) in `pair_mpo` / `lindblad_mpo(...; triples)`. |
| `d08731f` | `mpdo_identity_sector` (\|I_N⟩⟩ under strong U(1)); OS-3 defaults to strong. |
| `19b0b8c`, `92e5a98`, `45476cf`, `d4a3360` | `it_common.jl` (IT-1 refactor), the IT-2 driver, the OS-1/2/3 drivers, `sparse_lindblad.jl`, `xx_open_covariance` (exact OS-1 with dephasing), plots, `IT_INIT=dw`. |
| `a3bb0f6` | sparse_lindblad `abs()` fix (s40 crash). |

## 4. Next steps (in order)

1. **Read s40's tests and s41.** Fix any failures properly (the root cause, not the test).
2. **Investigate §2.1** (`projection` floor, `krylov+closure` at dβ 8), and §2.3 if it is cheap.
   Then decide §2.2.
3. **Merge `plan/imag-open-t3` into `feature/krylov-closure-order`** once the tests pass.
4. **Task 3 production on one PKG:**
   - L = 12/16/20 from BOTH Néel and the domain wall: β=64, `panel:9:4` + `uniform:256`,
     bug2/3/4 × `tol_basis` {1e-3, 1e-5, 1e-7} × seeds 1–3, plus tdvp2, maxdim 1024, `IT_BACKAMP=1`.
     L=20 at 8 G; the old sbatch files are in `2026-10-18-it1/L*/run.sbatch`.
   - Then L=100: a TIMING job, alone, with `IT_BACKAMP=0` and the DMRG reference `IT_DMRG_D=512`.
     Run the stiffness / grid-refinement sweep and `IT_ROOT=radau2` as separate accuracy jobs.
   - Plots: `plot_it1.py DIR 16 [--tb ...]`; `16_dw` for the domain wall.
5. **Task 4:** IT-2 at 4x4 / 6x4 (exact; 6x4 needs ~10 G for the 2.7e6-state reference), with
   **maxdim ≥ 256** (64 binds at 4x4), both starts. Then 8x4 / 10x4 against DMRG.
   - Still to write: spatial-map and S(π,π) plots.
6. **Tasks 5–7:** OS-1 at L=32/64 (exact) plus the `OS_DTS` stiffness sweep; OS-2 at L=3–6 with
   nmax 2/3 against exact, then L=16/24 plus the γ (Zeno) sweep; OS-3 at L=4/6/8 against exact,
   then L=24/40.
   - Use 3 `tol_basis` values and several seeds (the plan's shared analysis).

## 5. Where things are

As in `agent_handoff_2026-10-18b.md` §7. Additions:

- **OS-2/OS-3 exact reference:** `benchmarks/open_system/sparse_lindblad.jl`.
- **First-step probe:** `benchmarks/imaginary_time/first_step_probe.jl`; its output is
  `2026-10-18-fp1/first_step_probe.csv`.
- **Memory notes:** `imag-open-plan-tasks3-7-progress`,
  `user-10-18-neel-first-step-flag-is-a-bug-dw-start-strong-u1-per-plan`.

| job | result |
|---|---|
| s37 17341472 | 661c3e7 insufficient; 4 test files OK; MaxRSS 5.0 G cold |
| s38 17341708 | frame checks localised `_extend_frame` 72 > 70; IT-1/IT-2 smokes OK (old PKG) |
| fp1 17342386 | the first-step probe |
| s39 17342409 | cancelled |
| **s40 17342415** | RUNNING: reproducer FIXED; IT-1 dw OK; OS-1 OK; OS-2/3 crashed (fixed); tests pending |
| **s41 17342676** | RUNNING: OS-2 / OS-3 smoke rerun |
