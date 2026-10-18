# Agent handoff — imaginary-time / open-system CBE-BUG plan (2026-10-18, second session)

- **Repo:** BUG-Julia.
- **Branch:** `plan/imag-open-t3`, HEAD `033225a` (+ this handoff). Forked from
  `feature/krylov-closure-order`, which holds the merged Tasks 1 and 2. Nothing has been merged
  since.
- **Working tree:** clean apart from this handoff commit.
- **Previous handoffs:**
  - `agent_handoff_2026-10-18a.md` — the start of this session: Tasks 1–2 merged, the Task 3
    production runs and the first sketch fix.
  - `agent_handoff_2026-10-17.md` — the standing rules, §0.
- **The plan:** `cbe_bug_imaginary_open_benchmark_research (1).md`. Seven tasks, done in order:
  1. imaginary-time infrastructure — MERGED;
  2. MPDO infrastructure — MERGED;
  3. IT-1 Heisenberg chain;
  4. IT-2 square cylinder;
  5. OS-1 driven XX + dephasing;
  6. OS-2 Bose-Hubbard + two-body loss;
  7. OS-3 U(1) East + dephasing.

---

## 0. Read first: one job is running and decides the next step

**s40test 17342415** (`$SCRATCH/rsvd_adaptive/2026-10-18-s40/`; PKG `9207c690e1c7` = HEAD
`033225a`, cold compile) verifies everything below. Its stages, in order:

| stage | grep | what it shows |
|---|---|---|
| gates | `EXIT-PARSE`, `EXIT-PRINTF` | parse and printf gates |
| reproducer | `EXIT-REPRO` | IT-1 L=16, bug3, `tol_basis` 1e-7, `panel:9:4`, seeds 1–3, `IT_FRAME_CHECKS=1`. This crashed in 17340512, s37 and s38. **This stage is the verdict on the frame fix.** |
| smoke | `EXIT-IT1DW` | IT-1 from the domain wall |
| smokes | `EXIT-OS1/2/3` | the first cluster runs of the OS drivers, including OS-3 under strong U(1) with \|I_N⟩⟩ |
| tests | `EXIT-TESTS`, `FILE-` | `test_imaginary_time`, `test_liouville`, `test_pair_mpo`, `test_krylov_closure`, `test_adaptive_rsvd`, `test_adaptive_search` |

- Gate 9 afterwards: `seffx -j 17342415`, the OOM grep, the `EXIT-` grep.
- **None of the new src code below has run on the cluster yet.** Its tests run for the first time
  in s40. If a test fails, read the code before weakening the test (the standing rule).
- **s38test 17341708** may still be in its test stage. It ran on the OLD PKG `0da0827126fb`
  (diagnostic build), so its `EXIT-TESTS` is informative only.

## 1. Standing rules

These are unchanged; `agent_handoff_2026-10-18a.md` §1 has the details.

- **Run nothing locally in Julia.** Python is fine (`csv` + matplotlib, no pandas).
- **Cluster jobs:**
  - Read `../cluster_compliance.md` in full before every `sbatch`.
  - `--partition=small,cluster --nodes=1 --cpus-per-task=4`, `julia -t 4`,
    `--constraint=x86-64-v3`, `--time=7-00:00:00`.
  - Exclude the cip nodes and th-cl-dell03.
- **Memory:**
  - A cold compile peaks at 5.0 G (s37: 5025740K), so cold jobs request `--mem=7G`.
  - L=20 production: 8 G.
- **Staging:** `git -c core.autocrlf=false archive HEAD` → `~/bugjl/stage.sh`. Never tar the
  working tree.
- **Git:** explicit paths with `git commit`.
- **One timing job at a time.**
- **Fairness:** bug2 ↔ tdvp2 at the same dt, the same per-cut discard rule, and
  `tol_trunc = tol_root = 1e-6`.

## 2. USER decisions this session (2026-10-18)

1. *"Continue with the rest of the tasks while waiting."* That is why the Task 4–7 drivers exist
   already.
2. *"For the Néel product state, [the first-step flag] hints at a bug, fix it. Also use a domain
   wall initial state."* Done: §3.2 and `IT_INIT=dw`.
3. *"For the strong U(1), follow what the plan states."* The plan's table gives OS-3 strong U(1),
   so OS-3 now defaults to `:strong` with |I_N⟩⟩.
4. *"Finish the current task now, then create a new agent handoff file."* This file.

## 3. Bugs found and fixed (verification pending in s40)

### 3.1 The IT-1 L=16/20 crash was a NON-ISOMETRIC frame, not a label convention

- `661c3e7` (align_charge) did **not** fix it. s37 still crashed with *"frame holds 174 directions
  in sector ((0,),) but local space has only 92"*, and sector 0 is self-conjugate.
- `a19c027` added `_FRAME_CHECKS` (`IT_FRAME_CHECKS=1`): a frame-fits-local-space check at every
  frame-growth site, off by default.
- **s38 localised it:** `_extend_frame` inside the closure frontier's `_krylov_enrich` left 72
  directions in sector 0 against 70 local (frame 253 of 256).
- **Mechanism:**
  - Telum's `svd` cutoff is relative to the GLOBAL σ_max across sectors (Telum-fork
    `svd.jl:1672`).
  - So roundoff in a saturated sector clears the cut whenever another sector carries a real
    direction.
  - `_reortho` then renormalises it into a fake column.
- **Fix `033225a`:** `_cap_to_local_room` (`cbe_core.jl`) keeps at most the sector's local room of
  new columns, the leading ones in SVD order.
  - It returns `U` unchanged whenever it fits, so the ordinary step is bit-identical.
  - It costs one `fusion_basis` per extension.
  - Test: `test_krylov_closure.jl`, "_extend_frame never exceeds the local space".
- **Same pattern still to audit:** `cbe_expand`'s `_reortho_left/right(U0, TLC/TRC)` and
  `_union_left/right` (keep_states) use the same globally relative cutoff. They never tripped the
  check in s38, but nothing proves they cannot.
- **Do next if s40 passes:** decide whether to call `_cap_to_local_room` there too. Mind the cost
  in timing jobs.

### 3.2 The flagged first ITE step from Néel (USER: "a bug")

The probe was `first_step_probe.jl`, run as fp1 job 17342386 (L=10, `2026-10-18-fp1/`).

- **The old explanation was wrong.** The handoff said "χ=1 frames cannot hold Hψ₁ at any dβ". In
  fact the first-step defect shrinks like dβ^(p+1):
  - domain wall: p=1 slope 2, p=2 slope 3;
  - Néel: ~3 already at p=1, because the half-sweep Krylov holds the closure's moments (p1 ≡ p2
    bit-identical, as recorded before).
  - The constant is 400–1000× the second step's, because the energy fluctuation is maximal at a
    product state.
- **The bug was the fixed halving in `ite_step!`.** `retries=2` reaches dβ/4, but the panel's
  first step (dβ=0.0625, `tol_basis` 1e-7) needs about dβ/7.
- **A second defect:** it also split steps that failed on a resource cap (`maxdim_reached` /
  `workcap_reached`). That only multiplied the truncations: the IT-2 4x4 smoke at maxdim 64 had
  bug2 E−E(β) = 6e-2 with 4 sub-steps per step.
- **Fix `576a46b`:**
  - `_ite_nsplit` gives n = 1.2·(resid/tol_basis)^(1/(order+1)), capped at 64.
  - `_ite_capped` means capped steps are flagged at once, not split.
  - `root_report` also carries the per-bond defect (`per_left/per_right`).
- **Tests:** the physics test now asserts NO failed step. There is a new first-step test (Néel
  and domain wall, `tol_basis` 1e-7) and a capped-no-split test.

### 3.3 tdvp2 in imaginary time could not handle large L (`6ceb0eb`)

- Unshifted, a step grows the state by e^(dβ|E|). The squared weights reach the Float64 limit near
  dβ|E| ≈ 350 (L=100, panel dβ=8).
- **Fix:**
  - `tdvp2_step!(...; eshift, backward_report)`; `eshift=0` is bit-identical.
  - The drivers pass Eshift = the state's energy, as `ite_step!` does.
  - `backward_report` is the plan's backward-substep amplification,
    `log(‖M'‖/‖M‖) − Re(−τ)⟨H1⟩_M`: ≥ 0 by Jensen and shift-invariant.
  - It costs one extra H1 action per bond when requested. Use `IT_BACKAMP=0` / `OS_BACKAMP=0` in
    timing jobs.

## 4. Task status

### Task 3 — IT-1 Heisenberg chain (driver ready; production blocked on s40)

- **Driver:** `it1_heisenberg_chain.jl` on the shared `it_common.jl`.
  - Output is unchanged except for two appended columns, `backamp` and `var`.
  - New knobs: `IT_VAR_EVERY` (variance vs β, untimed), `IT_FRAME_CHECKS`, and `IT_INIT=neel|dw`
    (tag `L<L>_dw`).
- **Smokes passed in s38:** L=12, β=4. The trajectory error at β=4 is 1.5e-6 for bug2 and 1.75e-5
  for tdvp2.
- **If s40 passes,** rerun L=12/16/20 production on PKG `9207c690e1c7`, from BOTH Néel and the
  domain wall:
  - β=64, `panel:9:4` + `uniform:256`, bug2/3/4 × `tol_basis` {1e-3, 1e-5, 1e-7} × seeds 1–3,
    plus tdvp2, maxdim 1024;
  - L=20 at 8 G. The old sbatch files are in `2026-10-18-it1/L*/run.sbatch`.
- **Then L=100** (stretch 200):
  - This is a TIMING job: alone, with `IT_BACKAMP=0`, and the DMRG reference `IT_DMRG_D=512`
    cached.
  - Run the grid-refinement / stiffness sweep (several `uniform:n` and `panel:P:k`) with
    `IT_BACKAMP=1` in a separate accuracy job.
  - Implicit roots via `IT_ROOT=radau2`.
- **Plots:** `plot_it1.py DIR 16 [--tb 1e-05]`; for a domain-wall run, pass `16_dw` as L. A 5th row
  shows tdvp2's backward amplification.

### Task 4 — IT-2 square cylinder (driver ready, smoke passed)

- **Driver:** `it2_square_cylinder.jl` (`IT_LX`, `IT_LY` even, `IT_INIT`).
  - Exact reference for N ≤ 24 (`pairs_sparse`, Lanczos m=60). For 6x4, C(24,12) = 2.7e6 states:
    request about 10 G.
  - DMRG reference for larger N, cached per geometry.
  - Output: site/bond maps and the finite-cylinder S^zz(π,π) (`it2_maps_*.csv`, `it2_final_*`).
- **4x4 exact:** E0 = −10.264289620979 (E0/N = −0.6415), S^zz(π,π) = 1.36977.
- **The 4x4 smoke used maxdim 64, which binds.** Production at 4x4 needs maxdim ≥ 256.
  - At the cap BUG is truncation-limited (known).
  - The 6e-2 drift was also the split-on-cap defect fixed in §3.2.
- **Next:** 4x4 and 6x4 production, both starts, then 8x4 and 10x4 (DMRG reference).
- **Not yet done:** spatial-map plots and S(π,π) plots; `plot_it1.py --model it2` covers energies,
  χ and cost only.

### Tasks 5–7 — open systems (drivers ready; first cluster run is s40)

Shared machinery:

- `benchmarks/open_system/os_common.jl` (`OS_*` knobs).
  - Arms: bugP via `cbe_bug_step!(hermitian=false, keep_states=[trace vector])` and tdvp2.
  - The raw trace is logged, then the state is renormalised.
  - Unmet steps are labelled. Checks (Hermiticity, min eigenvalue) run at output times.
- `OS_DTS` takes a list, for the stiffness sweep.
- Plots: `plot_os.py DIR os1|os2|os3 TAG`, tested on synthetic data only.

**OS-1 (`os1_driven_xx.jl`).** XX + boundary drive + dephasing, weak U(1).

- The EXACT reference at every L ≤ 64 is `xx_open_covariance`, added to
  `tests/common/free_fermion.jl`: dephasing keeps the two-point ODE closed.
- It is pinned against the dense Liouvillian at L=4 in `test_liouville.jl`. At γ=0 the NESS
  current has a closed form.
- Conventions are stated in the header. Rate g means D[√g J].

**OS-2 (`os2_bose_hubbard_loss.jl`).** Bose-Hubbard + loss.

- Exact sparse-Lindblad reference (`sparse_lindblad.jl`: substepped Arnoldi plus Saad's estimate)
  for d^L ≤ `OS_REF_MAXD`.
- Logs N and `loss_rate` for the exact identity dN/dt = −2γ Σ⟨n(n−1)⟩.

**OS-3 (`os3_east_dephasing.jl`).** East + dephasing.

- The three-site term is built with the new `TripleTerm` in `pair_mpo` / `lindblad_mpo(...;
  triples)` (`4288d32`). Test: dense-Liouvillian equality under weak and strong grading, [H,N]=0,
  frozen states, facilitated hops only.
- Default `OS_SYM=strong`, with `mpdo_identity_sector` (|I_N⟩⟩, `d08731f`) in the trial space.
  Test: trace = C(L,N), ⟨⟨I_N|ρ⟩⟩ = Tr ρ, a trace-exact loose step.

**Next:**

- OS-1 production: L=32, 64 (exact), stretch 100, γ ∈ {0, γ>0}, and the stiffness sweep in
  `OS_DTS`.
- OS-2: L=3–6, nmax 2/3 against the exact reference, then L=16/24 plus the γ sweep (Zeno).
- OS-3: L=4/6/8 exact, then L=24/40.
- Use 3 `tol_basis` values and several seeds (plan "Shared analysis").

## 5. Open items / caveats

- **The audit of §3.1** (cbe_expand / `_union_*` reortho) is open.
- **The IT drivers' DMRG reference** has not run at L=100.
- **The OS drivers have not run on the cluster** before s40. Expect small fixes. Things to watch:
  - `mpdo_checks` cost at large L with small `OS_OBS_EVERY` (local RDMs: 16 walks per bond);
  - `xx_open_covariance` building a 4096² `exp` at L=64 (≈ 1 GB).
- **`plan/imag-open-t3`** should be merged into `feature/krylov-closure-order` once s40 passes and
  the Task 3 deliverables exist. Until then everything lives only on the t3 branch.
- **Parked real-time-track jobs** (USER 10-18): `agent_handoff_2026-10-18a.md` §4 lists them. The
  ref1536t and p6r jobs should be finished by now; `p6r_cbe1se_1024` still needs a rerun at
  ≥ 10 G.

## 6. Jobs this session

| job | dir | result |
|---|---|---|
| s37test 17341472 | `2026-10-18-s37` | reproducer FAILED (174 > 92, the 661c3e7 fix is insufficient); 4 regression test files OK; MaxRSS 5.0 G (cold) |
| s38test 17341708 | `2026-10-18-s38` | gates OK; IT-1 smoke OK; IT-2 4x4 smoke OK (maxdim 64 binds); frame-check reproducer localised `_extend_frame` 72 > 70; tests: check `EXIT-TESTS` (old PKG) |
| fp1 17342386 | `2026-10-18-fp1` | first-step probe, CSV `first_step_probe.csv` (§3.2) |
| s39test 17342409 | `2026-10-18-s39` | CANCELLED (superseded by s40) |
| **s40test 17342415** | `2026-10-18-s40` | **RUNNING — §0** |

## 7. Where things are

| what | where |
|---|---|
| shared IT driver core / IT-1 / IT-2 | `benchmarks/imaginary_time/{it_common,it1_heisenberg_chain,it2_square_cylinder}.jl` |
| first-step probe | `benchmarks/imaginary_time/first_step_probe.jl` |
| OS drivers, shared core, sparse reference, plots | `benchmarks/open_system/{os_common,os1_driven_xx,os2_bose_hubbard_loss,os3_east_dephasing,sparse_lindblad}.jl`, `plot_os.py` |
| plots IT | `benchmarks/imaginary_time/plot_it1.py` (`--model it2`) |
| frame cap / frame checks | `cbe_core.jl` (`_cap_to_local_room`, `_frame_sector_dims`, `_check_frame`, `_FRAME_CHECKS`) |
| ITE retry ladder | `sweeps/imaginary_time.jl` (`_ite_nsplit`, `_ite_capped`) |
| tdvp2 eshift / backward amplification | `tdvp2/tdvp2_baseline.jl` (`_tdvp2_backward`) |
| three-site MPO terms | `models/pair_mpo.jl` (`TripleTerm`), `models/liouville.jl` (`triples`) |
| \|I_N⟩⟩ | `models/liouville.jl` (`mpdo_identity_sector`) |
| exact OS-1 reference | `tests/common/free_fermion.jl` (`xx_open_generator/ness/covariance`) |
| new tests | `test_imaginary_time.jl` (tdvp2 partner, first-step split), `test_liouville.jl` (OS-3 model, \|I_N⟩⟩, OS-1 reference), `test_krylov_closure.jl` (saturated frame) |
| memory notes | `imag-open-plan-tasks3-7-progress`, `user-10-18-neel-first-step-flag-is-a-bug-dw-start-strong-u1-per-plan` |
