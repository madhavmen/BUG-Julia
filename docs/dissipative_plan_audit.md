# Audit: `dissipative_cbe_bug_implementation_plan.md`, Task 1 (G0)

Branch `plan/dissip-task-1` (cut from `dissipative_benchmarks` = `main` @ `3139c6e`), commit
`c0a42e2` (fixes `b8837f0`, `d276134`; decisions `7c7adb6`, `39e9cea`). Status per plan item: **IMPLEMENTED** (on the production call path, with a test that
calls it), **PARTIAL**, **MISSING**, **DIVERGES** (exists but contradicts the plan text),
**LATER(Tn)** (belongs to a later task). Evidence is `file:symbol`. Tests:
`tests/sweeps/test_dissipative_t1.jl` (D1.1–D1.8), regressions `test_dd_cbe.jl`,
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
| complex normalisation | IMPLEMENTED (`7c7adb6`) | the budget scale is `‖ψ‖` at the step's start (`_nrm`, bug/cbe_bug.jl). `_kry_weight` (bug/krylov_enrich.jl) takes its unitary branch only when `tau*H_k` is skew-Hermitian (`_KRY_SKEW_TOL`), so a lossy step at `tau = -im*dt` weighs each Krylov direction relative to the decayed state. D1.7 |
| operator / gauge cache invalidation | IMPLEMENTED | operators rebuilt per bond visit (`OneSiteOp`, `TwoSiteSketchOp` with `version`, symmetry check); stale frontiers refused (`_frontier_sweep!`). Rebuild after a jump: LATER(T4) |
| backward-compatible loss-free evolution | IMPLEMENTED | D1.4: `hermitian = false` on Hermitian `H` equals the Hermitian step; `no_jump_mpo(H, 0)` equals it and keeps the norm |

## 3. Non-Hermitian Krylov

| plan item | status | evidence |
|---|---|---|
| root `expv` | IMPLEMENTED | `hermitian = false` → `arnoldi_expv` (BondUpdateBUG/expv_root_methods.jl): CGS2, Saad tail `conv_tol = tol_root`, adaptive substeps. D1.2 against `exp(tau U'GU)`; Lanczos on the same root is wrong by > 1e-3 (control) |
| local one-site Krylov | IMPLEMENTED | `_krylov_enrich` always builds the full Hessenberg (CGS2 against every vector); its weight `_kry_weight` reads `exp` of the Hessenberg, not a tridiagonal |
| non-Hermitian two-site TDVP | IMPLEMENTED | `tdvp2_step!(...; hermitian = false)` reaches both the forward two-site and backward one-site `expv` (baselines/tdvp2.jl); D1.3: at L = 2 Arnoldi is exact at `maxiter` 30 and 2, Lanczos at `maxiter = 2` is wrong (> 1e-3). Same MPO object as BUG |
| Radau IIA(2) + GMRES | IMPLEMENTED (`7c7adb6`) | `root_method = :radau2_gmres` → `radau2_gmres_expv` (BondUpdateBUG/expv_root_methods.jl): the coupled stage system `(I - τA⊗B)Y = 1⊗S0`, decoupled by `A = PΛP⁻¹` into two shifted systems that share ONE Arnoldi basis; GMRES to `tol_root` on the true stage residual, at most `_RADAU_MAXCYCLE` restarts, then `root_limited`. Plugs into the same root call, so the CBE, the local Krylov and the order-3 frontier enrichment are unchanged. D1.8 (order 1 and 3: both stages = the dense coupled solve, end state = stage 2, ≠ exp, = `:radau2` when converged). `:radau2` (stability function on the Arnoldi space) stays. Not recycled yet: the frontier vectors as GMRES start space, GMRES subspaces across steps |

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
| no dense two-site action allocated by CBE | IMPLEMENTED (`39e9cea`) | the half-sweep CBE has no `apply_h_two_site` call site and `ledger.n_h2_leak == 0` (D1.5). The order > 1 frontier sweeps measure each bond's leak with `_bond_action`; `fold_omega` now defaults to `order > 1 && !exact`, so they use the folded sketch and a Test-B holdout estimate (`n_h2_leak == 0`, D1.5 default and `true`). `fold_omega = false` keeps the exact `H Θ_F` leak (D1.5 control, `n_h2_leak > 0`); `exact = true` keeps it too, since the folded leak is sampled. The README records the fold slower at every χ ≤ 256 measured |

### Decisions (USER 2026-10-24)

1. **`fold_omega` for `order > 1`:** on by default (`39e9cea`).
2. **Krylov weight under loss:** implemented (`7c7adb6`, D1.7).
3. **Radau IIA(2) + GMRES:** implemented with the same CBE / Krylov / frontier enrichment
   (`7c7adb6`, D1.8).
4. **D1.3 control:** rebuilt at `maxiter = 2` (`7c7adb6`).

## Results

| job | commit | file | result |
|---|---|---|---|
| dt1a 17471485 | `c0a42e2` | `test_dissipative_t1.jl` | ERROR: `onsite_mpo`'s `Any[Iid O]` literal concatenated the TLArrays (fixed `b8837f0`, also in `mpo_sum`) |
| dt1a 17471485 | `c0a42e2` | `test_dd_cbe.jl` | 131/131 |
| dt1a 17471485 | `c0a42e2` | `test_hq_root.jl` | 46/46 |
| dt1a 17471485 | `c0a42e2` | `test_liouville.jl` | 91/92: OS-U3 `min_eig = -1.76e-4 > -1e-4` fails, the pre-existing regression parked by the USER 2026-10-22 (not on Task 1's code path) |
| dt1b 17471544 | `b8837f0` | `test_dissipative_t1.jl` | 29/31: two vacuous fixture checks (fixed `d276134`) |
| dt1c 17471567 | `d276134` | `test_dissipative_t1.jl` | 30/31, MaxRSS 1.9 G |
| dt1d 17471631 | `7c7adb6` | -- | cancelled after 6.5 min, superseded by `39e9cea` |
| dt1e 17471646 | `39e9cea` | `test_dissipative_t1.jl` | 51/51 (D1.1-D1.8) |
| dt1e 17471646 | `39e9cea` | `test_dd_cbe.jl`, `test_krylov_closure_steps.jl`, `test_adaptive_search.jl`, `test_adaptive_parallel.jl`, `test_hq_root.jl` | 131/131, 102/102, 32/32, 32/32, 46/46 |
| dt1e 17471646 | `39e9cea` | `test_liouville.jl` | 91/92: OS-U3 only (parked) |
| dt1e 17471646 | `39e9cea` | `test_imaginary_time.jl` | 203/205: IT-U6 `:radau2` at `tol_root = 1e-30` reports `:met` (expected `:failed`, `:expv`). Last full run of this file was 10-18 (s49); control dt1m 17471800 runs it at `main` `3139c6e`. 54 min, MaxRSS 2.5 G |
| dt1m 17471800 | `main` `3139c6e` | `test_imaginary_time.jl` | 203/205, the same two IT-U6 checks: **pre-existing on `main`, not Task 1.** `main` touched this path after 10-18 in `1a1779a` (legacy `cbe_bug_step!` path removed), `dfa2543`, `3139c6e`; not bisected (outside this plan). MaxRSS 1.9 G |

- D1.1 3/3, D1.2 9/9 (root error 1.1e-12 at `tol_root` 1e-6 and 1e-10, Saad estimate 1.8e-11, root
  dim 20 = the whole N = 3 sector; Lanczos on the same root 0.67), D1.4 3/3, D1.5 11/11, D1.6 3/3.
- D1.3: Arnoldi TDVP2 at L = 2 exact (< 1e-12). **The Lanczos CONTROL fails, open:** with
  `maxiter = 30` and the breakdown-only exit, the three-term recurrence still satisfies
  `G V = V T + b v e'` for any `alpha`, so `V exp(tau T) e1` is a degree-30 polynomial in `G` and
  converges for `||tau G|| ~ 1` even though `real(alpha)` drops the loss: Lanczos agrees with
  Arnoldi to 6e-15 (L = 2) and 7e-14 (L = 6). The flag IS wired (`_tdvp2_bond!`, `_tdvp2_backward`
  pass `hermitian`). D1.2's root Lanczos is wrong because it stops early on Saad's estimate of
  the wrong `T`. A control that can fail needs an early stop (`conv_tol`) or a larger `||tau G||`.

# Task 2 (G0 → G1): hard mathematical unit tests

Branch `plan/dissip-task-2`, cut from `plan/dissip-task-1`. The G0 gate (U01-U05, U08, U12, U17)
is `tests/sweeps/test_dissipative_t2.jl`; G1/G2 (U06, U07, U09-U11, U13-U16, U18) are not written
yet, and U07, U13-U15 test the jump engine of plan Task 4.

| item | status | where |
|---|---|---|
| pair-loss `Q` (A1/B1) | NEW | `doublon_loss_mpo` (`models/dissipative.jl`): `Σ κ_j n↑n↓`, virtual dim 2 |
| collective-emission `Q` (A2/B2) | NEW | `collective_emission_mpo`: one transported channel per (hop direction, phase), virtual dim 6 at any `L`. c_ij sits on vertex `(Sm, Sm')`, which the MPO applies as `S⁺_i S⁻_j` (transpose rule, `spin_xx.jl`) |
| MPO as a matrix on small systems | NEW | `mpo_matrix_element(bra, mpo, ket)`; `mpo_energy` is its `bra = ket` case |
| bytes per phase (U17) | NEW | `KernelCounters.b_one_site`, `b_root`, `b_env`, `b_compress` |
| d = 4 Hubbard under `:U1` | FIXED | had never run: `HUB4_*` labels were `((n-1, s),)`, Telum's are `((n-1,), (s,))`; channel B's op-legs (`f'Z`, `f`) needed `legflip` and the op-leg last (`_hub4_half`); the automaton's zero pre-fill is `:none`-only (`91e8c6a`) |

| job | commit | file | result |
|---|---|---|---|
| dt2a 17471813 | `fd00dc4` | `test_dissipative_t2.jl` | 180/184. U02 Hubbard `:U1`: ERROR (Duplicate TLIndex, `f'Z` is `(s, op, s)`); U02 emission: `MQ = Q(-k)` exactly (3 fails, c_ij on the wrong vertex). Fixed `ff2473b`, labels `d76cec2`. U03-U17 all pass |
| dt2a 17471813 | `fd00dc4` | `test_dissipative_t1.jl`, `test_dd_cbe.jl`, `test_krylov_closure.jl` | 51/51, 131/131, 222/222. MaxRSS 2.2 G (cold compile), 21 min |
| dt2b 17471868 | `ff2473b` | -- | cancelled: the `:U1` labels (`d76cec2`) were found before it ran |
| dt2c 17471888 | `d76cec2` | `test_dissipative_t2.jl` | 183/184: emission U01/U02 8/8 (fix confirmed). Hubbard `:U1`: ERROR in `oplus`, "column 2 has incompatible spaces on leg 4": the automaton's vacuum-zero pre-fill contradicts a charged hop column. Fixed `91e8c6a` |
| dt2d 17471938 | `91e8c6a` | `test_dissipative_t2.jl` | **184/184: the G0 gate passes.** MaxRSS 2.0 G, 14 min |

# Task 3 (G1): A1 / A2 model fixtures and exact smoke tests

Branch `plan/dissip-task-3`, cut from `plan/dissip-task-2` @ `91e8c6a`. Tests
`tests/sweeps/test_dissipative_t3_a1.jl` (Hubbard chain, bulk doublon loss) and
`test_dissipative_t3_a2.jl` (XXZ chain, collective emission); shared fixtures
`tests/common/dissipative_fixtures.jl`; exact Lindblad references through
`lindbladian_from(H, Xs)` (`benchmarks/open_system/sparse_lindblad.jl`). `src/` equals Task 2's.
The trajectory averages and jump counts of A1/A2 need the jump engine: LATER(T4).

| item | status | where |
|---|---|---|
| baseline | USER 2026-10-24 | standard two-site TDVP (`tdvp2_step!`, `hermitian = false`, same MPO) is the plan's only baseline; the split-loss TDVP of `6faaa23` is removed (`9b99150`) |
| bond-charge decoders | FIXED `4f2d96c` | the outgoing bond leg carries the NEGATED left-block charge; before the fix the A1 sector-rank checks passed vacuously |
| A2 order 3 from a product state | `2530c55` (tests) | see below; the A2 BUG settings set `closure_checks = true`, the solver default stays OFF (USER 10-17). USER 2026-10-24: every nonlocal-`Q` run (A2/B2 pilot and production) sets it too; the plan's driver sketch and Task 6/7 items say so |

**A2 at L = 8 (dt3a, dt3b): the first step missed `Q`'s long-range hops.** The error is all in
step 1 and scales with `dt Γ`. dt3e (`dt3diag2.jl`, one step from the product state, Γ = 2,
dt = 0.025) separates the cause:

| settings | root dim (sector 56) | `‖BX − PGPX‖/‖X‖` | step vs own Galerkin | `Q_LR ψ₀` outside frames | step error |
|---|---|---|---|---|---|
| A2 smoke | 44 | 5.1e-14 | 1.2e-15 | 49% | 4.0e-2 |
| `tol_basis` 1e-12 | 44 | 5.8e-14 | 1.2e-15 | 49% | 4.0e-2 |
| + `exact = true` (DD sketch) | 44 | 3.7e-14 | 1.4e-15 | 49% | 4.0e-2 |
| + `closure_checks = true` | 56 | 1.4e-13 | 1.6e-15 | 1.4e-15 | 1.3e-15 |

The Telum contractions are exact. `Hψ₀`, the onsite part and the nearest-neighbour hops of `Qψ₀`
all lie in the frames (≤ 1.5e-15). The S⁺S⁻ hops of range ≥ 2 do not: from a U(1) product
state, the two-site environments cannot see them, since the charge makes every
⟨frame|S±|frame⟩ zero. The single enriching pass per frontier therefore cannot capture them
either, and its per-bond leak estimate is exact only for nearest-neighbour terms.
`dt‖(I−P)Gψ₀‖ = 4.9e-2` accounts for the 4.0e-2 step error. Tighter tolerances and the exact
sketch change nothing. The validated closure (`closure_checks = true`) repeats the pass until
it adds nothing. The second pass starts from frames that already mix charge sectors, so the
root grows to the whole sector and the step is exact.

| job | commit | file | result |
|---|---|---|---|
| dt3a 17471939 | `6faaa23` | A1, A2 | A1 physics green (rank checks vacuous, see `4f2d96c`); A2 L = 6 exact (1e-14), L = 8 smoke fails ED by 3e-3 (Γ = 0.2) to 4.6e-2 (Γ = 2) |
| dt3d 17472019 | `6faaa23` | diagnostic | MPO exact at L = 7, 8; H-only 4e-9; error all in step 1, ∝ `dt Γ` (dt/4 → /3.8); same with `tol_basis` 1e-11, exact DD, TDVP2 (7e-3) |
| dt3b 17472046 | `4f2d96c` | A1, A2 | A1 293/293; A2 134/138 (the four L = 8 smoke accuracy checks only) |
| dt3e 17472284 | `4f2d96c` | `dt3diag2.jl` | the table above (17472280: script path mangled, cancelled) |
| dt3c 17472355 | `2530c55` | A1, A2 | **A1 276/276, A2 138/138.** A1's 17 fewer = the removed split arm (3 Strang checks, 2 refinement checks, 2 per dt × 3 dt × 2 κ in the loop). MaxRSS 2.0 G of 3.9 G, 17 min, CPU 49.7% (one core) |

# Task 2 (G1/G2): U06, U09, U10, U11, U16

Branch `plan/dissip-task-2` (`fadedf8`, `85e9916`, `fb152fb`), `tests/sweeps/test_dissipative_t2_g12.jl`.
U07, U13-U15 test the jump engine and are in Task 4's files; U18 (optional parallel K/L/S) is not
run, because the parallel backend is not used in production (plan §1.2).

| test | dt2f 17473173 (`85e9916`) | note |
|---|---|---|
| U06 contractivity, hazard | 6/6 | `P0` never grows before or after the recorded compression; `-Δ log P0` against `∫⟨Q⟩_c dt` |
| U09 balances | 5/5 | eq. (2), (3) against ED finite differences at dt 0.01 (at 0.05 the FD error was above the tolerance) |
| U10 local loss | 3/3 | from the L = 10 domain wall (HQ T5.5 method: slopes only where error ≥ 10× floor): global 4.3e-5 → 2.7e-6 → 1.7e-7 (slope 4.0), one-step 3.76, order-1 control 2.04 |
| U10 collective | **1/3, open** | global 2.2e-2 → 5.4e-3 → 1.0e-3 (slopes 2.0, 2.4), see below |
| U11 ladders | 17/17 | each tolerance moves its own error and cost |
| U16 fairness | 31/32 | the lossy arm's BUG floor was `tol_trunc`'s: `fb152fb` sets 1e-10 (to be confirmed) |

**U10 collective: frontier j = 2 is not closed from a U(1) product state.** The order-3 step closes
frontiers j = 1, 2 (`_krylov_closure!`). Closing frontier j should put G^{j+1}ψ in the root space.
dt5k (`67eff34`, L = 10 domain wall, closure_checks, one step, dt 0.4/0.2/0.1):

| trial space | one-step errors | slopes |
|---|---|---|
| closure_checks | 4.6e-2, 1.7e-2, 4.1e-3 | 1.42, 2.06 |
| + keep Gψ, G²ψ | identical to 4 digits | 1.43, 2.06 |
| + keep Gψ, G²ψ, G³ψ | 2.0e-2, 4.2e-3, 5.3e-4 | 2.23, 2.99 |

Gψ and G²ψ are in the frames already; G³ψ is not. The closure measures each frontier's leak in
two-site windows inside the environments. A hop S⁺_a S⁻_b with a and b far apart moves both blocks
out of their frames' charge sectors at once, and every window projects that away. The loop
therefore stops on "nothing added" while the global vector still leaks. Keeping the exact global
vectors (`apply_mpo`, the existing `keep_states`) supplies them. Passing more than one kept state
exposed a crash: a later union widened the frame after an earlier carry was taken. That is fixed
in `67eff34`. dt5m measures whether the defect is first-step only, and the asymptotic slopes.

# Task 4 (G1 → G2): the quantum-jump engine, B1/B2 smoke

Branch `plan/dissip-task-4`, cut from `plan/dissip-task-3`.

| item | status | where |
|---|---|---|
| `L_μψ` | NEW | `mpo_product`, `site_op_product`, `apply_factor` (`operators/mpo_apply.jl`): exact product, a gauge sweep, one `truncate_recursive!` at `tol_trunc`; a zero product is refused |
| jump channels | NEW | `JumpChannel(name, factors)`; `jump_mpo` (charged MPO), `SiteOp` (local, no MPO); `pair_loss_channels`, `collective_emission_channels`, `site_emission_channels` (`models/dissipative.jl`) |
| eq. (8) trajectory | NEW | `jump_trajectory` (`bug/quantum_jumps.jl`): log survival carried with per-step renormalisation; the event located by Illinois regula falsi on `log P0 = log u` to `event_tol`; channel `∝ ‖L_μψ‖²`; `:norm_growth` flags; seeds stored with events; `jump_ensemble` (eq. 9) with standard errors |
| B1 junction | NEW | `hubbard_aa_mpo_d4` with bond `J_j` and `V_jσ`; `hub4_current_mpo`; `pair_loss_junction` (one channel at `j0`) |
| conventions (traps) | FIXED `86e9533`, `d205a4a` | the MPO and `_apply_site_op` apply the TRANSPOSE of the stored tensor (store `Sp` to apply S⁻, `f'` to apply f); Telum's `Sp = -S⁺/√2`, so jumps are scaled to a unit matrix element by a two-site probe |

| job | commit | file | result |
|---|---|---|---|
| dt4a | `6a599bb` | t4.jl | engine reproduces the ED event time (1.5e-14); U14 A2 vs Lindblad |
| dt4c | `a54866e` | t4_b.jl | B2 smoke: MPS vs ED 1e-14; interaction identity holds at N_exc = 1 (1e-14), breaks at N_exc = 2; counts [32, 209, 59] vs master equation [39.0, 202.5, 58.5]; the independent-emitter control fails, as it should |
| dt4g 17474462 | `d205a4a` | t4.jl | 50/52: J0 (spin, pair loss with signs), U07, U13 (event 0.18173997222066 vs ED …6403, residual 1e-14; channel draw 120 vs 110.8 ± 5.6), U14 A2 L = 4, A1 L = 2, 4 vs Lindblad. The 2 fails: the pair-loss STRING CONTROL could not fail (site 1 still ~1e-4 from doubly occupied at t = 0.6) → `1ab7d0e` (t = 3, asserts the string is not a global sign) |
| dt4g | `d205a4a` | t4_conv.jl | 6/7: U15b weak convergence 1.2e-7 → 7.0e-9 → 3.2e-9. U15a: located (errors 1.1e-5, 8.5e-7, 5.8e-8, order 3.7-3.9, vs snapping 0.31, 0.11, 0.008) but the absolute 1e-8 bound was a wrong premise: the remainder is the segment's time error in P0 → `9de020d` |
| dt4g MaxRSS | | | 2.5 G, 55 min |
