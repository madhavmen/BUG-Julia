# Audit: `dissipative_cbe_bug_implementation_plan.md`, Task 1 (G0)

Branch `plan/dissip-task-1` (cut from `dissipative_benchmarks` = `main` @ `3139c6e`), commit
`c0a42e2` (fixes `b8837f0`, `d276134`). Status per plan item: **IMPLEMENTED** (on the production call path, with a test that
calls it), **PARTIAL**, **MISSING**, **DIVERGES** (exists but contradicts the plan text),
**LATER(Tn)** (belongs to a later task). Evidence is `file:symbol`. Tests:
`tests/sweeps/test_dissipative_t1.jl` (D1.1–D1.6), regressions `test_dd_cbe.jl`,
`test_hq_root.jl`, `test_liouville.jl` (cluster job dt1a 17471485).

**How a dissipative run is expressed.** The integrators advance `exp(tau * mpo)`. The no-jump
generator `G = -iH - Q/2` is therefore the MPO `H_eff = H - (i/2) Q` at `tau = -im*dt`
(`no_jump_mpo(H, Q)`), with `hermitian = false`. Nothing else in the step changes.

## 1. Locate / verify the existing modules

| plan item | status | evidence |
|---|---|---|
| `half_left!` / `half_right!` | IMPLEMENTED | `_half_left!` / `_half_right!` (bug/half_sweeps.jl): interleaved DD-QR CBE then local Krylov per bond; basis-only, the carry is the ORIGINAL amplitude |
| three-round stage predictor | IMPLEMENTED | `order = 3`: the half-sweeps close frontier `j = 0`, then `_single_pass_closure!` (bug/krylov_closure_loop.jl) runs one enriching `_frontier_sweep!` per frontier `F_j = (tau^j/j!) B^j S0`, `j = 1, 2` (bug/frontier_sweeps.jl). Whether it attains order 3 on `G` is U10 (Task 2) |
| MPS root reconstruction | IMPLEMENTED | `svd(S1, (1,); cutoff = 0.0)` lossless split, then `_assemble_and_compress!` (bug/cbe_bug_info.jl): the one compression |
| implicit DD+QR MPO sketch | IMPLEMENTED | `dd_cbe_expand` → `_dd_probe` (P^D on the thin probe), `TwoSiteSketchOp` / `_stream_left` / `_stream_right` (MPO-streamed, `Θ` and `X` never formed), `_dd_select` / `_thin_qr` (sectorwise Householder, two passes), `_dd_controller` (independent holdout) (expansion/dd_cbe*.jl) |
| Krylov recycling | PARTIAL | `_krylov_enrich` (bug/krylov_enrich.jl) is seeded with the CBE frame `U_ex` (`krylov_reuse = true`) and frontier frames are transported by the overlap `Pm`; no Krylov VECTOR of an earlier bond/round is transported. Same finding as `hq_plan_audit.md` T4 acceptance row (PARTIAL) |
| matrix-free root operation | IMPLEMENTED | `RootOp` / `zero_site_h` / `apply_zero_site` (operators/zero_site_core.jl) inside `expv`; the root matrix is never formed |

## 2. Hermitian `H` → `G = -iH - Q/2`

| plan item | status | evidence |
|---|---|---|
| MPO / operator dispatch for `G` | IMPLEMENTED (new) | `no_jump_mpo(H, Q)` = `mpo_sum(H, Q; b = -i/2)`, `onsite_mpo(L, ops)` (operators/mpo.jl). Direct sum of the virtual legs: any builder's `H` and `Q` combine without a new automaton. D1.1: on a root space the MPO action is the dense `H - iQ/2`, and `B - B' = -iQ` |
| adjoint for the right sketch | IMPLEMENTED (unchanged) | `_stream_right` contracts `Ω'` into the bra side of the SAME MPO, i.e. the rows `Ω_L' X`; no step assumes `H = H'`. The plan's `X'` is the matrix adjoint of `X = mat[G Θ]`, not `G'`. Explicit tiny check is U03 (Task 2) |
| complex U(1) charge flow | IMPLEMENTED | states are complex by default; `mpo_sum` promotes every site tensor to `ComplexF64`; `Q` here is charge-neutral. Complex cross-site phases (collective emission) are U02 / Task 3 |
| complex normalisation | PARTIAL | the budget scale is `‖ψ‖` at the step's start (`_nrm`, bug/cbe_bug.jl). `_kry_weight` takes its unitary branch whenever `real(tau) == 0`, so at `tau = -im*dt` with a non-Hermitian `H_eff` the Krylov weights are absolute (start norm), not relative to the decaying state. Exact for unitary steps; for strong loss (`κ h` large) it under-weights by the step's decay factor. Not changed here; see §6 |
| operator / gauge cache invalidation | IMPLEMENTED | operators rebuilt per bond visit (`OneSiteOp`, `TwoSiteSketchOp` with `version`, symmetry check); stale frontiers refused (`_frontier_sweep!`). Rebuild after a jump: LATER(T4) |
| backward-compatible loss-free evolution | IMPLEMENTED | D1.4: `hermitian = false` on Hermitian `H` equals the Hermitian step; `no_jump_mpo(H, 0)` equals it and keeps the norm |

## 3. Non-Hermitian Krylov

| plan item | status | evidence |
|---|---|---|
| root `expv` | IMPLEMENTED | `hermitian = false` → `arnoldi_expv` (BondUpdateBUG/expv_root_methods.jl): CGS2, Saad tail `conv_tol = tol_root`, adaptive substeps. D1.2 against `exp(tau U'GU)`; Lanczos on the same root is wrong by > 1e-3 (control) |
| local one-site Krylov | IMPLEMENTED | `_krylov_enrich` always builds the full Hessenberg (CGS2 against every vector); its weight `_kry_weight` reads `exp` of the Hessenberg, not a tridiagonal |
| non-Hermitian two-site TDVP | IMPLEMENTED | `tdvp2_step!(...; hermitian = false)` reaches both the forward two-site and backward one-site `expv` (baselines/tdvp2.jl); D1.3 (L = 2 exact, Lanczos control wrong). Same MPO object as BUG |
| Radau IIA(2) + GMRES | MISSING as specified | `root_method = :radau2` applies Radau IIA's stability function `R(z)` on the Arnoldi space (one stage, no coupled-stage solve, no GMRES). Usable as the stiffness benchmark the plan asks for; the default stays `:exp` |

## 4. `order = 3` dispatch

| plan item | status | evidence |
|---|---|---|
| explicit dispatch, recorded recurrence | IMPLEMENTED | `cbe_bug_step!(...; order = 3)`; `record.order`, `record.closure_policy = :dd_qr_streamed`, `ledger.closure_log` (one entry per frontier sweep: `frontier`, `leak`, `added`). D1.5: frontiers {1, 2} swept |
| root starts from the ORIGINAL state | IMPLEMENTED | `S0 = _project_root(psi, fr)` after every enrichment (`_single_pass_closure!`), `ledger.proj_err = ‖(I-P)ψ‖/‖ψ‖`. D1.5: `proj_err < 1e-12` |
| `order = 3` on parallel K/L/S | IMPLEMENTED (dispatch) | `parallel_kls_step!` = `cbe_bug_step!(...; kls)`: the same order-p closure runs before K/L/S; the K, L, S stages are full `expv` exponentials of their projected generators. Observed order: U18 (Task 2) |

## 5. Tolerances, ranks, status, profiling, RNG

| plan item | status | evidence |
|---|---|---|
| three tolerances | IMPLEMENTED | `TOL_DEFAULTS`, `AdaptiveBudget`, `basis_allocation` (budget/adaptive_budget.jl); `tol_root` is the plan's name for `tol_expv` |
| `tol_trunc` semantics (§1.3) | IMPLEMENTED | relative 2-norm `‖ψ - ψ_trunc‖/‖ψ‖`, equal per-bond shares (`truncate_recursive!`); `tdvp2_step!(...; weight_tol)` applies the same per-cut rule (`_cut_divisor`) |
| peak working-rank cap | IMPLEMENTED | `work_ceiling(maxdim, r) = max(2 maxdim, 2r, 1)` on every frame |
| temporary vs final vs allowed rank | IMPLEMENTED (extended) | `StepRecord.D_stage` (untruncated frames), `D_final`; NEW `D_max` (`maxdim`) and `D_work` (the ceiling). D1.6 |
| status codes | IMPLEMENTED | `StepRecord.limits` (`:basis_limited`, `:root_limited`, `:truncation_limited`, `:work_limited`, `:rank_limited`), `budget_status` / `budget_diagnostic` |
| profiling | IMPLEMENTED | `KernelCounters` (streamed actions, input/output DD, QR, one-site, root: counts, seconds, bytes); `CBEBugSweepInfo` phase timers (CBE, Krylov, root, compression, environments). Frontier-sweep CBE calls are not in `record.counters` (`hq_plan_audit.md` T4) |
| deterministic RNG streams | IMPLEMENTED | `rng` keyword; under `parallel` the right sweep and each frontier get streams drawn from it. Per-trajectory seeding is the caller's: LATER(T4) |

## 6. Deliverable

| check | status | evidence |
|---|---|---|
| `model = :lossless` reproduces the Hermitian tests | see results | D1.4 + `test_dd_cbe.jl`, `test_hq_root.jl` (T5.2 Hermitian root), `test_liouville.jl` |
| `model = :nonhermitian` root `expv` = small-matrix exponential | see results | D1.2 |
| no dense two-site action allocated by CBE | IMPLEMENTED for the half-sweeps; **DIVERGES** for `order > 1` | the half-sweep CBE has no `apply_h_two_site` call site and `ledger.n_h2_leak == 0` (D1.5). The order > 1 frontier sweeps measure each bond's leak with `_bond_action`, which forms `X = H Θ_F` once per bond unless `fold_omega = true` (then a Test-B holdout estimate, `n_h2_leak == 0`). Selection stays QR-only either way. The README records `fold_omega = true` slower at every χ ≤ 256 measured, which is why it is off |

### Open decisions (not changed in Task 1)

1. **`fold_omega` default for `order = 3`.** The plan (§1.2) forbids dense `X`; the default
   forms it in the frontier leak test. Flipping the default is plan-conformant and measured
   slower at χ ≤ 256.
2. **Krylov weight under loss.** `_kry_weight`'s unitary branch keys on `real(tau) == 0` rather
   than on the propagator being unitary. Using the non-unitary branch for `hermitian = false`
   would make the local Krylov test relative to the decayed state.
3. **Radau IIA(2) + GMRES.** Only the stability-function form exists. The plan wants it as a
   benchmark only, so it can wait for Task 5's stiffness profile.

## Results

| job | commit | file | result |
|---|---|---|---|
| dt1a 17471485 | `c0a42e2` | `test_dissipative_t1.jl` | ERROR: `onsite_mpo`'s `Any[Iid O]` literal concatenated the TLArrays (fixed `b8837f0`, also in `mpo_sum`) |
| dt1a 17471485 | `c0a42e2` | `test_dd_cbe.jl` | 131/131 |
| dt1a 17471485 | `c0a42e2` | `test_hq_root.jl` | 46/46 |
| dt1a 17471485 | `c0a42e2` | `test_liouville.jl` | (see job) |
| dt1b 17471544 | `b8837f0` | `test_dissipative_t1.jl` | 29/31: two vacuous fixture checks (fixed `d276134`) |
| dt1c 17471567 | `d276134` | `test_dissipative_t1.jl` | 30/31, MaxRSS 1.9 G |

- D1.1 3/3, D1.2 9/9 (root error 1.1e-12 at `tol_root` 1e-6 and 1e-10, Saad estimate 1.8e-11, root
  dim 20 = the whole N = 3 sector; Lanczos on the same root 0.67), D1.4 3/3, D1.5 11/11, D1.6 3/3.
- D1.3: Arnoldi TDVP2 at L = 2 exact (< 1e-12). **The Lanczos CONTROL fails, open:** with
  `maxiter = 30` and the breakdown-only exit, the three-term recurrence still satisfies
  `G V = V T + b v e'` for any `alpha`, so `V exp(tau T) e1` is a degree-30 polynomial in `G` and
  converges for `||tau G|| ~ 1` even though `real(alpha)` drops the loss: Lanczos agrees with
  Arnoldi to 6e-15 (L = 2) and 7e-14 (L = 6). The flag IS wired (`_tdvp2_bond!`, `_tdvp2_backward`
  pass `hermitian`). D1.2's root Lanczos is wrong because it stops early on Saad's estimate of
  the wrong `T`. A control that can fail needs an early stop (`conv_tol`) or a larger `||tau G||`.
