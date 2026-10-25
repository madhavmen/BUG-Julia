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
| U10 collective | 1/3 in dt2f; **passes in dt5n** after `6f34c9f` | dt2f: global 2.2e-2 → 5.4e-3 → 1.0e-3 (slopes 2.0, 2.4). dt5n 17477888: global 1.1e-4 → 8.7e-6 → 6.2e-7 (slopes 3.65, 3.82); without the kept vectors (control) the one-step slopes are 2.48, 2.73. See below |
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
in `67eff34`.

dt5m 17477143 found that the defect is confined to the first step. Keeping G..G³ψ on step 1 only
gives the same errors as keeping them on every step, and from an entangled start the step is exact.
`6f34c9f` adds `global_krylov_states` (`operators/mpo_apply.jl`). U10 collective keeps those vectors
on the product-state step, at dt 0.1, 0.05 and 0.025, and runs a control without them. The driver
keeps them by default on product-state steps of A2/B2 (`DR_KEEP_KRYLOV`, ablation
`DR_KEEP_EVERY`). In dt5n (`6f34c9f`), `test_dissipative_t2_g12.jl` passes 67/67.

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
| dt4h | `d566af8` | t4_b.jl | 443/452. **B1 current sign:** the MPO and the Fock reference both measured the flow b+1 → b, so they agreed and were both wrong → `618b9a6` (j = iJ(c'_{b+1} c_b − h.c.)). B1 smoke L = 4 deviated 3.6 from Lindblad, with the ensemble closer to the no-jump control. A pathwise ED replica (dt5p, `67eff34`) showed the engine is exact path by path. The deviation was the test's error model: the ensemble error was floored by the jump count's Poisson error → `2a54806` |
| dt5n 17477888 | `6f34c9f` | t4.jl | 50/54: the J0 string control still cannot fail at t = 3 from (N↑, N↓) = (1, 1), because there (−1)^{n_1↑} is a global sign → `7aad6f5` starts at (2, 2) |
| dt5n | `6f34c9f` | t4_conv.jl | 8/8 (U15a with `9de020d`) |
| **dt5q 17481802** | `09af23e` | t4_b.jl | **452/452**: B1 currents MPO = Fock to 1e-16; B1 L = 4 worst deviation 0.37 (observables) and 0.34 (jumps) in units of 4 se + 2e-3, against 1.90 for the no-jump control; B1 conditional L = 6 7.0e-7; B2 MPS vs ED ≤ 7e-14; counts [32, 209, 59] vs master equation [39.0, 202.5, 58.5]. 79 min |
| **dt5r 17481814** | `b051384` | t4.jl | **54/54**: J0 with the (2, 2) start; U13 event 0.18173997222066 vs ED …6403; U14 A2 L = 4, A1 L = 2, 4 within 0.70 (4 se + 2e-3), no-jump controls 6.6-17.8. 60 min |

**Task 4 status: G1 → G2 passes** for `t4.jl`, `t4_b.jl` and `t4_conv.jl` (dt5r, dt5q, dt5n).
seffx reports no wasted allocation for dt5q or dt5r.

# Task 5 (G2): order, tolerances, step control vs TDVP2

Branch `plan/dissip-task-5`, which has Tasks 2 and 4 merged in. The driver is `benchmarks/dissipative_run.jl`
(one run per call, JSON out) and the figures come from `benchmarks/dissipative_figures.py` (F01-F04, F10).
The runs live in `figures/dissipative_t5/runs*/` (configs, commit, sbatch, JSON).

## Arrays and why the first three were not valid

| array | point | finding |
|---|---|---|
| dr5a (96 runs) | A1 L = 10/12, A2 L ≤ 16, maxdim 256 | **invalid.** A1: maxdim 256 binds (a cap that prevents the tolerance is a failed gate, plan §1.3). A2: the closure grows the root space to the full sector, so BUG is exact vs ED (1e-10) and the order is vacuous. The A2 ladders are valid and monotone |
| dr5b | A1 L = 8, maxdim 1024 | **degenerate:** the bonds reach full rank at once, so BUG steps are exact (step doubling 0) and order, controller and ladders are vacuous |
| dr5c | A2 L = 12 bright, ED, closure off / keep 3 | **degenerate**, as dr5b |
| dr5d 17478244 (46 runs) | A1 L = 12 dbl_rich U = 4, maxdim 1024 (not binding: peak 748), T = 0.5 | valid. Results below |

## dr5d results (`68a0db8`)

The error is the largest |Δ log P0| or observable difference over the output times, against a BUG
reference at dt 0.003125 with tolerances 1e-11/1e-13/1e-13.

| item | result |
|---|---|
| F01 order, κ = 1 | BUG 1.4e-5, 4.4e-7, 2.7e-8, 1.7e-9 at dt 0.1 … 0.0125: slopes 4.98, 4.02, 3.95. TDVP2 2.3e-6 … 4.4e-9 at dt 0.05 … 0.00625: slopes 3.00, 3.00, 3.00 |
| F01 ablation | lossless (exact DD) 4.02 below dt 0.1; tol_basis 1e-6, tol_root 1e-8 and tol_trunc 1e-7 each leave the slopes at 3.95-4.04. No single component lowers the order |
| F04 strong loss | κ = 8: BUG 4.08, 4.06, TDVP2 3.0. κ = 20: BUG 3.39, 3.75, TDVP2 3.0. TDVP2's backward-substep amplification is 0.15-1.5 (∝ dt κ); BUG has none. No P0 growth flags |
| F02 ladders (vs a same-dt tight run) | tol_basis 1e-8/1e-6/1e-4: 1.3e-10, 2.1e-9, 1.7e-7. tol_root 1e-10/1e-8/1e-6: 8e-14, 1.3e-11, 2.9e-10. tol_trunc 1e-9/1e-7/1e-5: 1e-11, 6.4e-10, 1.8e-7. Each is monotone in its own tolerance |
| F10 cost | at 2.7e-8, BUG 346 s vs TDVP2 (3.5e-8) 387 s; at 4.4e-7, BUG 342 s vs TDVP2 (2.8e-7) 462 s. These walls come from an accuracy array, so they are indicative only |
| F03 controller | **defective**, see below |

**The slopes are one order above the global orders.** BUG (order 3) shows about 4 and TDVP2 (order 2)
shows 3.0. That is the one-step signature: from the product state, the first step's O(dt^{p+1})
error dominates a 5-to-40-step run (U10 local loss has the same pattern, global 4.0). So the plan's
global-order criterion (2.7-3.3) has not actually been tested here. `bf4cf5f` adds `DR_WARM`: the
run starts from the state evolved to t_w by tight BUG steps, and every run of a point uses the same
warm state. dr5e repeats F01 from t_w = 0.2 for A1 L = 12 and for A2 L = 20 (block start,
closure_checks plus first-step keep, maxdim 1024).

## The eq. (7) controller estimated 0

In dr5d the checks reported err = 0.0 at t = 0.1 and 0.3, with no rejections. dt5s 17481834 traced
this to the distance, not the control flow. The dense distances between the full and doubled steps
were 2.1e-12, 3.1e-12 and 3.8e-12. `_mps_distance` returned 0, 0 and 1.49e-8, because the overlap
form ‖a‖² + ‖b‖² − 2Re⟨a,b⟩ cancels to about eps, and a converged step-doubling estimate sits below
√eps. `feb64a9` computes ‖a − b‖ from the direct-sum MPS of a − b (block-diagonal bonds, as
`mpo_sum`), swept to the last site by backward-stable SVDs.

`tests/sweeps/test_dissipative_t5.jl`, on L = 10 local loss from the domain wall:

- **C1:** the distance resolves ε‖X‖ down to ε = 1e-12.
- **C2:** on real steps it equals the dense distance. The old formula is replayed as a control and
  must fail by more than 50%. The eq. (7) estimate of the first step's error matches the dense
  error within a factor 3.
- **C3:** in `jump_trajectory`, the first step is rejected and shrunk, never forced through at
  dt_min. A check is accepted exactly when err ≤ τ, every estimate is > 0, and the step grows to
  ≥ 2dt. The final error is < 1e-7 and below 1/100 of the fixed step's, which must be > 1e-7 (the
  control).

The first two runs failed for test reasons, not code. dt5t: C1's relative bound sat below the SVD
roundoff (1.1e-15 at ε = 1e-12), so `0aea55d` adds a 5e-15 absolute floor. C3's `segment` closure
also captured the testset's own `tr`, so the controlled result came back as the fixed run's; it is
now `local`. dt5u (21/22): the controller rejects the domain-wall step three times (0.1 → 0.05 →
0.025 → 0.0158, err 5.8e-7 → 4.2e-10). Once the steps are exact (err ~4e-12) it doubles every check
to 0.4. Final error: 2.8e-11 in 8 steps vs 6.4e-7 for the fixed step. The control threshold of 1e-6
was a guess; C2 measures the first-step error at 5.4e-7, so `bbeade2` sets it to 1e-7.

dr5d's controller design was also wrong. `jump_trajectory` lands on every output time, and
DR_EVERY = dt = 0.1 capped every step at 0.1. dr5e reruns the controller with one output at T = 0.5:
τ ∈ {1e-5, 1e-7, 1e-9} with checks every step, plus τ = 1e-7 checking every 4 steps, and τ ∈
{1e-7, 1e-9} from the warm start. The fixed-step reference is dr5d's (product start) or dr5e's
(warm start).

| job | commit | content | result |
|---|---|---|---|
| dt5s 17481834 | `b051384` | distance diagnosis | above |
| dt5t 17491550 | `feb64a9` | t5.jl C1-C3 | 14/16: test defects (C1 floor, C3 capture) |
| dt5u 17491733 | `0aea55d` | t5.jl | 21/22: C3 control threshold |
| dt5v 17491774 | `bbeade2` | t5.jl | **22/22**: the controller tests pass (MaxRSS 1.7G of 2.5G) |
| dr5e 17491590 (24 runs) | `bf4cf5f` | warm-start F01 (A1 L = 12, A2 L = 20), F03 rerun | below; peak MaxRSS 12.3G (reference), most runs ~5G (seffx: 39% of the 16G requests wasted) |
| dt5w 17492054 | `e45bf94` | t5.jl + check_every = 4 first-step check | **24/24**: with check_every = 4 the first check is at t = 0 and is rejected 3 times; final error 2.9e-11 (unchecked: 1.4e-5); MaxRSS 1.8G |

## dr5e results (`bf4cf5f`, figures `figures/dissipative_t5/dr5e`, `e45bf94`)

**Rank limits were not marked, and they decide the warm A1 slopes.** In an order-3 BUG step the
augmented basis grows to about 3r. The internal working-rank ceiling `work_ceiling` = max(2 maxdim,
2r) = 2048 therefore binds once D_final passes about 700 at maxdim 1024. From then on the ledger
reports the tolerance as missed (`budget_met` = 0). The solver flags this (plan U12), but
`dissipative_figures.py` fitted slopes through such runs. `e45bf94` marks them, as plan §4.4
requires: for BUG the first step with `budget_met` = 0, for TDVP2 the first step with a bond at
maxdim. Capped points are drawn open, and every row with a cap also reports the error over the
output times before the first capped step (the run's own or its reference's).

| item | result |
|---|---|
| warm A1 L = 12 BUG (t_w = 0.2) | before the first capped step, the error is 5-9e-13 for every dt from 0.1 to 0.0125: **no time error accumulates after the first step**. Step doubling agrees (estimate 1.4e-11 at h = 0.1 and 0.2). The full-window errors 6.2e-6 (dt 0.1) and 5.8e-7 (dt 0.05) appear only after the cap binds (t = 0.4 and 0.5) |
| warm A1 L = 12 TDVP2 | 5e-13 to 4e-12 (dt 0.05 is capped from t = 0.4) |
| A2 L = 20 block (both) | **degenerate**: every bond is 1024 = 2^10, the full rank (and maxdim), so every step is exact (3e-13). TDVP2's cap marks here are the full rank, not truncation |
| dr5d product start, re-marked | BUG is never capped: slopes 4.98, 4.02, 3.95. TDVP2 at dt 0.05 and 0.025 hits maxdim; before the cap its slopes are 3.0, 2.5, 2.06, falling toward order 2 |
| F04, re-marked | BUG is uncapped: κ = 8 slopes 4.08, 4.06; κ = 20 slopes 3.75, 3.39. TDVP2 before its cap: κ = 8 1.49, 2.01; κ = 20 1.82, 1.91 |
| F03 product start | τ = 1e-5, 1e-7, 1e-9 give final errors 5.6e-9, 1.0e-10, 6.4e-12 (fixed dt 0.1: 1.4e-5), with 2, 3 and 5 rejections; the step grows to 0.17-0.23. **check_every = 4 is defective**: the first check came at step 4, the product-state step went unchecked, and the run kept the fixed step's 1.4e-5. `89e59e7`: the first step of a segment (and after each jump) is always a check; test C3 asserts it (dt5w) |
| F03 warm start | τ = 1e-7 and 1e-9: final errors 7.4e-13 and 8.4e-13. The rejections fall exactly on the capped steps (estimates 9e-5 at h = 0.2): shrinking h lowers the working rank a step needs below the ceiling |

## G2 assessment

- **Order.** BUG's only visible time error is the first step from a weakly entangled state. Every
  later step is exact to the 1e-12 tolerances whenever the basis budget is met, even at dt 0.1. So a
  run's global error is its first step's error, and the measured slopes are the one-step slopes:
  3.95-4.08 at κ = 1 and 8 (3.39-3.75 at κ = 20, where dt κ ≥ 0.25). That meets "one-step slope
  compatible with 4". The plan's global criterion asks for e ∝ dt³ **before error floors**; after
  the first step the error is already at the floor for every dt, so no dt³ regime exists to fit.
  This is the same finding as U10 local loss (global 4.0, Task 2) and HQ T5.5. TDVP2 shows its order
  2 once its capped steps are excluded.
- **Tolerances:** monotone, each ladder in its own tolerance (dr5d F02). **Ablation:** no component
  lowers the order. **Stability:** no P0 growth flags; TDVP2's backward amplification reaches 1.5 at
  dt κ = 1 and BUG has none.
- **Controller:** C1-C3 pass (dt5w 24/24, including the first-step check). The τ ladder is monotone
  with errors below τ.
- **Cost:** BUG and TDVP2 are comparable at matched error on A1 L = 12 (dr5d F10, an accuracy array).
  TDVP2's dt 0.05 and 0.025 points are capped there.
- **G3 constraint found here:** BUG at order 3 stays uncapped only while D_final ≲ 0.65 maxdim, because
  the working ceiling is 2 maxdim. TDVP2 only needs D_final < maxdim. Pilot runs must pick maxdim
  from BUG's working rank, and report `budget_met` and TDVP2's cap hits.

**G2 passes** on this reading of the order criterion (USER 2026-10-25), with dt5w confirming the
first-step check. Task 5 is on `plan/dissip-task-5` (head `eaf8a1f`) and is not merged to main.

# Task 6 (G3): pilots and profiling

Branch `plan/dissip-task-6`, cut from the Task 5 head. USER 2026-10-25 chose the "G3 essentials"
scale (about 300-500 CPU-h) over the plan's full grid. Measured costs rule the grid out:

- A1 at L = 12-16 reaches maxdim 512 by Jt ≈ 0.7, and BUG's work ceiling by Jt ≈ 0.4. A
  non-binding run to Jt ≈ 8 is not feasible.
- B1 at 100 trajectories per point costs about 0.25 CPU-h per trajectory at L = 12, or about
  1800 CPU-h for the grid (dr6t).

The scale consists of:

- one non-binding BUG-vs-TDVP2 time-to-accuracy curve each for A1 and A2, run as sequential single
  timing jobs;
- a tol_trunc ladder subset;
- the plan's A1/A2 points at a binding cap, for cost and bond profiling;
- B1/B2 on 4-6 points with about 20 trajectories each.
- USER 2026-10-25, added: a stability run for the plan's Task 5 stability bullet, which Task 5
  only covered at Jt ≤ 1. The points are A1 L = 12 κ = 20 and A2 L = 24, 6 excitations, Γ = 5
  (A2 at L = 12 and 20 sat at full rank). Both use a warm start, so TDVP2's product-state
  blindness to long-range hops is not read as instability. BUG runs at dt 0.4-0.05 and TDVP2 at
  dt 0.4-0.025, to Jt = 4, against a tight BUG reference, plus one controller run (τ = 1e-6) per
  model.

  The F04 energy-balance check used to be the finite-difference residual, which came out at
  0.7-24, the same for both integrators and every dt: it measured the output grid. It is now the
  integrated balance, compared with the reference on the same grid. On dr5d this "excess" tracks
  the state error (BUG κ = 8: 1.5e-9, 2.0e-8, 3.0e-7 at dt 0.0125, 0.025, 0.05), while the
  own-grid residual stays at its floor of 0.045. F04 also reports error against t, the largest dt
  per error target (1e-3, 1e-4, 1e-6) and the steps that raised P0.

## Pre-submission audit of the driver against the plan (call sites in `benchmarks/dissipative_run.jl`)

| plan item | where | status |
|---|---|---|
| Task 6 "measure before scaling": time_CBE_DD, time_QR, local Krylov, Arnoldi root, compress, environments | `t_dd`, `t_qr`, `t_kry`, `t_root`, `t_trunc`, `t_env` (plus `t_cbe`, `t_frame`, `t_step`) | logged since `dc729ae` |
| MPO matvecs | `matvecs` = `info.krylov_dims` (operator applications, cbe_bug_info.jl) | logged |
| probe / holdout columns | `n_probes`, `n_stream_cols`; `n_holdout` | **added**: `n_holdout` |
| reused Krylov vectors | `n_keep` (global Krylov states kept on product-state steps) | **added** |
| rejected steps | `nreject`, `checks` (controller runs) | logged |
| peak working D, final D, sectors | `D_stage`, `D_work`, `D_final`; `sectors` per step | **added**: per-step `sectors` |
| CPU time, peak memory | `wall` per step (1 thread); `peak_rss` per step | **added**: per-step `peak_rss` (BUG and TDVP2) |
| §3 accuracy status | `budget_met`, plus the `StepRecord.limits` flags (`eaf8a1f`); figures mark caps | logged |
| §3 DD holdout estimate, local Krylov defect, root residual | `holdout_est` (ledger `max_sample`), `krylov_defect` (`max_krylov`), `root_err` (`expv_err`) | **added** |
| §3 norm / log survival per step | `norm2` (unnormalised norm after the step) | **added** |
| §3 Hamiltonian / jump-operator hash | `config.operator_hash` (model, L, start, parameters; with the commit it fixes H and Q) | **added** |
| Task 6 A2: compressed collective MPO width | `config.mpo_dims` (H, Q, G virtual dims) | **added** |
| Task 6 A2: closure passes per step and their cost | `closure_sweeps`; `t_closure` (ledger; in none of the phase timers) | **added**: `t_closure` |
| §3 per trajectory: RNG seed, events | `traj[].seed`, `traj[].events` (t, channel) | **added**: `seed` |
| Task 4/6 B1: currents on both sides of the sink, lead imbalance | `jup/jdn_left/right` (`hub4_current_mpo` on bonds j0-1 and j0), `imb_up/dn` | **added** |
| Task 4 B1: preparation convergence | `params.prep` (β, status, energy, residual of each ITE step) | **added** |
| §4.3 checkpoint / restart | `jump_trajectory` / `jump_ensemble` `checkpoint`, `resume`; driver `DR_CKPT`, `DR_CKPT_SEC`, `DR_STOP_AT` (`1787e44`) | **added and confirmed**, see below: dt6s (t6 R1-R3), dt6r (t4 54/54, t5 24/24 unchanged), dr6y (15/15 restart checks) |
| B2: dark-state overlap | none | optional, not added |

Two Julia scope traps were avoided while adding these. Local methods of one function share a name
space, so B1's `lead(n, s)`, which builds the start state, would have been replaced by a new
`lead(n, r)`. The A1/B1 observable is `hub_obs`, not a second `obs(p)`.

## Checkpoint / restart (plan §4.3, `1787e44`)

The plan asks for the MPS and charge sector, the controller history, the trajectory RNG and jump
threshold, the accumulated log survival, and the observables and diagnostics. It also asks that a
restart be refused if the MPO or the solver semantics changed. Mapping:

| plan item | implementation |
|---|---|
| MPS, charge sector | the normalised `psi` (its tensors carry the sectors) |
| controller history | `hcur`, `since`, `nreject`, `steps`, `checks` |
| RNG, jump threshold, log survival | the `MersenneTwister` itself (the threshold `logu` is already drawn from it), `logu`, `logp` |
| observables, diagnostics | `obs`, `logp_grid`, `njumps`, `events`, `flags`; the driver adds its step log |
| ensembles | `(done, current)`: the finished trajectories plus the running one's state |
| refuse an inconsistent restart | the driver's header is the run configuration (operator hash, MPO widths, tolerances, integrator) without `T`, plus a hash of the solver source; `jump_trajectory` refuses another seed or a grid that does not extend the checkpoint's |

A BUG step seeds its own sketch (`MersenneTwister(0x5EED)` per call), so it depends only on its
input state. A resumed run should therefore reproduce the uninterrupted run exactly, and the
tests check exact equality, with no tolerance.

- **The hook:** the state is handed over as live references. A copy at every grid time would
  double the memory of a maxdim-2048 run. The driver serialises it at most every `DR_CKPT_SEC`
  (default 1800 s) and removes it once the JSON is written.
- **Growing `T`:** `T` may grow on a resume, so a validated run can be extended. The plan asks
  for this in A2: "validate ... to Jt ~ 20 before requesting longer runs".
- **Tests (`test_dissipative_t6.jl`):**
  - R1: a resume from a real serialisation round trip, with jumps and the controller, from the grid
    time before the last jump; CONTROL: a fresh RNG draws different jumps.
  - R2: another seed, a different grid and a truncated grid are each refused, and a longer grid
    extends the run.
  - R3: ensembles resumed inside a trajectory and between trajectories.
- **Driver check (`benchmarks/dissipative_restart_check.sh`, the Task 7 "dry-run output and
  restart"):** each run goes uninterrupted, then is stopped at checkpoint n and resumed. The two
  JSONs must be equal apart from measurements. CONTROL: a resume with a changed `tol_trunc` must
  be refused.

**Results:**

- dt6s (`b173c8e`): t6 passes. R1 resumes from t = 0.45, between the jumps at 0.18 and 0.50.
  The first version (dt6r) stopped at its own premise guard: on a grid of 0.5, both jumps fall
  before the first checkpoint. The grid was refined; the guard and exact equality are unchanged.
- dt6r: t4 54/54 and t5 24/24, so the jump engine and controller are unchanged by the hooks.
- dr6y: all three configurations pass all five checks. They are A2 no_jump with the controller and
  an ED reference, A2 traj stopped inside trajectory 2, and B1 traj with the recomputed
  preparation. Each check confirms:
  - the stopped run wrote the checkpoint and no JSON;
  - the changed `tol_trunc` was refused;
  - the resume removed the checkpoint;
  - the resumed JSON equals the uninterrupted one.

  The first version (dr6x) read "no JSON yet" after the resume had written it, a harness bug.

## Profiling: dr6p / dr6q / dr6r

- **B1, L = 20, maxdim 256, two trajectories to Jt = 6 (dr6r):** ~14 s per step, ~30 min per
  trajectory, MaxRSS 6.3 G.
  - The cap binds almost at once: 96 % of steps are work-limited and 86 % rank-limited.
  - The root error is at most 1e-12, and the holdout estimate is nonzero (up to 1e-3).
  - A B1 pilot at 20 trajectories × 4-6 points costs ~40-60 CPU-h, rank-limited and marked so.
- **A2, L = 32, nexc 8, maxdim 1024: BUG crashes (dr6q rows 1 and 3, reproduced by dr6z):**
  - Both crash in the step after the first, a NaN in the root Arnoldi Hessenberg (fixed step) and
    an empty root SVD (controller). It is deterministic.
  - Each crashing step ran ~45 min before failing, starting from D ≈ 74 with the budget met. The
    first step (from the product state, 170 s) left P0 = 0.52: at finite density the collective
    `Q` is large (‖Q‖ ~ Γ L²/4).
  - TDVP2 on the same point and B2 at L = 24 run normally.
  - The driver now writes `DR_OUT.partial` and, with `DR_CRASH_STATE`, the failing step's input.
- **A1, L = 12, maxdim 2048, Jt = 1, tol_trunc 1e-12 (dr6p):** BUG (dt 0.1) 868 s, TDVP2
  (dt 0.05) 1628 s. Both reach the cap: TDVP2 at Jt ≈ 0.6, after which its discarded norm grows
  from 8e-13 to 2.7e-11 per step; BUG at Jt = 1.0. log P0 agrees to 3.7e-5. MaxRSS 4.9 G / 3.7 G.
  At this tolerance the point is borderline-binding, so the Task 6 curve uses tol_trunc 1e-9 arms
  and separate tight references.

## The A2 crash: `_dd_select` admitted in-frame roundoff (fixed, `a2d8c78`)

Replays of the two crash states localised it in steps. The replays (rp6z to rp6v) are scripts that
include the driver with `DR_T = 0` and redefine solver functions with instrumentation; they are
kept with the crash states in `$SCRATCH/rsvd_adaptive/2026-10-25-dr6z`.

1. The root's Arnoldi start vector already had norm 5e76, so the root frames were not isometric.
2. The half-sweep frames are fine (defect ‖F'F − I‖ ≈ 1e-7). The first bad frame is a
   `dd_cbe_expand` result in a closure frontier sweep (`_frontier_left`, `closure_checks = true`)
   at site 13, width 637 of a local space of 814, defect 14. The frontier embeds the old frame as
   `Pm·W[i]`, so one non-isometric frame compounds site by site, to 1e26 at the root.
3. The first selection whose `Q` overlaps its frame (frontier sweep, site 10; serialised as
   `out/1.json.crash.sel`): `‖frame'Q‖ = 1.41` = two unit columns inside the frame. Pass 1 of
   `_dd_select` saw `Y = P^D Z` with residuals 3.0e-14 in sectors the frame fills (its blocks
   there are full: 36 × 45, 9 × 10), against a floor `100 eps · max(‖Z‖, ‖Y‖)` of 2.65e-14. QR
   normalised that roundoff to unit columns. Pass 2, the "twice is enough" reprojection, then
   compared their residual with an ABSOLUTE `100 eps` and kept them; renormalised, they lie in the
   frame.

The plan's T2.4 reprojection is the classical DGKS / Kahan-Parlett test, which accepts a
reorthogonalised column only if it keeps a fixed fraction of its norm (1/√2). `_dd_select` now does
exactly that (`_REORTH_KEEP`): a genuine direction keeps ≈ 1, while roundoff that cleared the first
floor collapses to the frame's orthogonality level. Nothing physical is cut, since the test is
scale-free. Test "T2.4: roundoff in a sector the frame fills is not a direction, both sides"
(`test_dd_cbe.jl`) builds such a case on both sides. Its CONTROL asserts that the old floor keeps
an in-frame column; the new selection must be isometric, orthogonal to the frame, keep `Y`'s range
and touch no filled sector.

**Confirmed on the crash states (rp6u 17496050, `a2d8c78`, frame checks after every frame-producing
call):** both failing steps now complete, in 20-21 min. Before the fix they ran about 45 min and
then failed. Every frame stays at a defect of at most 1.2e-6, which is the measurement floor
√(k eps) of the defect formula at widths k of 1400-2048; before the fix the defect was 14, and 1e26
at the root. In that one step at dt 0.1 the centre bonds grow from 74 to the 1024 cap, so A2 L = 32
at finite density is cap-bound from step 2 at tol_trunc 1e-12. MaxRSS 8.4 G. `test_dd_cbe.jl`
passes, including the new test (dt6t 17496046).

`_reortho` (`cbe_expand.jl`, the RSVD-CBE path) has the same single-pass shape with a relative SVD
cutoff of 1e-14. Its frames stayed at a defect of about 1e-7 in every replay, so it is unchanged.

The dr6q B2 runs predate the fix. Every step after the first ran the maximum 6 closure sweeps at
the work ceiling (`D_stage` = 1024, `basis_limited` and `work_limited` in 49 of 50 steps), with a
local Krylov defect of 0.1-2.8, even at L = 24, where D_final settled at 378 with zero discarded
weight. That looked like fake columns inflating the frames, so B2 L = 24 was rerun on the fixed
code (dr6u row 3). **The hypothesis is refuted:** the rerun is identical. log P0 agrees to 1e-8,
and the bonds, the 6 closure sweeps per step and the work-ceiling flags are all the same; wall
4115 s against 3885 s, MaxRSS 4.8 G. It is a cost property of the closure for collective `Q` at
finite density, not a defect. The state needs D = 378 with zero discarded weight, yet the frames
sit at the 1024 ceiling every step. The closure runs its maximum 6 sweeps because the local Krylov
defect stays at 0.1-1.7, so it never converges.

## Task 6 results on the fixed code (`4a76d77`)

**References and the tol_trunc ladder (dr6b 17496157; A1 L = 12, U = 4, κ = 2, T = 1, maxdim 3072).**
None of the runs hit a cap; the peak D_final was 1579.

- The tol_trunc ladder at dt 0.05, measured against the same-dt tight run, gives 2.0e-10, 1.6e-8
  and 3.6e-6 for tol_trunc 1e-9, 1e-7 and 1e-5. It is monotone in its own tolerance, so it is not
  hidden by tol_basis (1e-10) or the root solve. This is the G3 condition. Against the dt 0.025
  reference all three are flat at 2.7e-6, which is the dt 0.05 time error.
- At tol_trunc 1e-5 the ledger marks 7 steps `work_limited` + `basis_limited` at D_final 44-180.
  That flag comes from the DD controller's probe budget (`streamed >= wcap` or `n >= ncap`), not
  from maxdim. With tol_basis held at 1e-10, each step's truncation discards about 7e-6, and the
  basis budget asks for more than the sketch is allowed to stream. It is an honest flag on a
  deliberately mismatched point.
- The dt 0.025 and dt 0.05 references differ by 2.7e-6 in log P0. With the first step dominating
  (one-step order 4), the reference error is about 2e-7, which would floor TDVP2's finest arms.
  dr6d 17496371 adds BUG references at dt 0.0125 and dt 0.00625.
- MaxRSS 6.7 G against 16 G requested (§7 one-off over-request); later references request 8 G.

**Binding cap: BUG's working-rank ceiling binds far below maxdim (dr6c row 1 against dr6t row 1;
A1 L = 12, κ = 2, maxdim 512, tol_trunc 1e-9).**

- The fix changes nothing here before the cap binds: pre-fix and post-fix BUG agree to their own
  error up to t = 1. After that both are truncation-dominated and drift apart, reaching 0.16 in
  log P0 at Jt = 4.
- BUG's order-3 working basis is 5-9 times its input rank for d = 4 (input rank 76 gives D_stage
  620). `work_ceiling` = max(2 maxdim, 2r) = 1024 therefore binds at D_final = 195 (step 4), which
  is 0.38 maxdim. From that step on the basis budget is unmet (`basis_limited`, `work_limited`).
  maxdim itself binds from t = 0.7.
- Against the dr6b reference:

  | t | BUG error (log P0 / obs) | TDVP2 error (log P0 / obs) |
  |---|---|---|
  | 0.6 | 3e-5 / 5e-4 | 2.5e-6 / 3e-6 |
  | 1.0 | 6e-3 / 4.7e-2 | 2e-6 / 1.9e-6 |

  TDVP2 needs only D_final < maxdim, and it is at D = 294 at t = 0.6.
- Both integrators apply the same per-cut rule (discarded 2-norm `weight_tol ||Θ|| / _cut_divisor`:
  `truncate_recursive!` for BUG, `_budget_svd` for TDVP2). Before the cap, BUG simply carries
  about 1.5 times TDVP2's rank (t = 0.4: 197 against 146).
- At equal maxdim, BUG's capacity is set by the working ceiling, a "tuned policy, not a theorem"
  (adaptive_budget.jl). The plan's Task 1 asks for a centralised peak-working-rank cap separate
  from maxdim. It is centralised, but fixed at a factor of 2 and not exposed to callers.
- **USER 2026-10-25: report as is.** The solver is unchanged. BUG is marked work-limited from the
  step its working ceiling binds, and BUG and TDVP2 are compared at matched error and cost, never
  at equal maxdim alone.

**Run bookkeeping.** dr6u rows 1-2 (A2 L = 32) and both dr6d references were OOM-killed at their
10 G and 8 G requests (sampled MaxRSS 10.0 G and 8.0-8.3 G). They were resubmitted as dr6v
17496930 (16 G) and dr6d 17496931 (12 G; row 2 resumes from its checkpoint). The A2 stability
rows run at maxdim 512 (dr6w 17496935), the class B2 L = 24 just profiled; at 1024 the cost would
be about 4-8 times as high. The stability array reused this morning's smoke directory name
(`2026-10-25-dr6s`); it overwrote the smoke's four JSONs, which no figure or table used, and the
smoke logs remain. The fixture tests missing from dt6t (wrong file name `test_dissipative_t3.jl`)
run as dt6u 17496932 (`t3_a1`, `t3_a2`, `t2_g12`).

**A1 stability run (dr6s 17496308; A1 L = 12, U = 4, κ = 20, warm start t_w = 0.2, Jt = 4,
maxdim 1024, tol_trunc 1e-9; figures `figures/dissipative_t6/a1_stability`).** The reference is
BUG at dt 0.025 with tolerances 1e-10/1e-12/1e-11. The walls come from an accuracy array, so
they are indicative only.

| arm | dt | max error | wall [s] | TDVP2 backward amp | note |
|---|---|---|---|---|---|
| BUG | 0.05 / 0.1 / 0.2 / 0.4 | 4.1e-9 / 3.3e-10 / 1.1e-10 / 2.0e-7 | 1186 / 672 / 643 / 406 | - | dt 0.2 and 0.4 work-limited from t = 0.4 |
| TDVP2 | 0.025 / 0.05 / 0.1 / 0.2 / 0.4 | 1.8e-9 / 2.6e-8 / 2.9e-8 / 8.6e-7 / 1.6e-6 | 2248 / 1446 / 799 / 610 / 497 | 0.35 / 0.69 / 1.3 / 2.5 / 4.7 | never capped |

- **No instability in either integrator:** no step raised P0, there are no P0-growth flags, and
  the error against t grows at most slowly. TDVP2's backward amplification is κ dt / 1.7, yet it
  stays bounded up to κ dt = 8.
- **Largest dt per target:** at an error of 1e-8, BUG reaches dt 0.2 (0.1 uncapped) and TDVP2
  dt 0.025, about 3.5 times the wall. At 1e-6, BUG reaches 0.4 (work-limited) and TDVP2 0.2.
- **BUG's error is set by truncation, not by dt:** it falls as dt grows (4e-9 at 0.05, 1e-10 at
  0.2) because fewer steps accumulate fewer truncations at tol_trunc 1e-9.
- **Energy balance:** the excess over the reference tracks the state error (BUG dt 0.4 3.4e-7,
  TDVP2 dt 0.4 4.4e-6). The reference's own fine-grid residual peaks in the fast transient
  (t < 0.7) and then stays at 3e-3, so no violation accumulates.
- **Controller (τ = 1e-6):** the step grows from 0.1 to 0.4 with no rejections; final error
  2.7e-9, 13 accepted steps, of which 12 (with check substeps) are work-limited.
- **Reading:** under strong loss (κ dt up to 8) TDVP2 does not go unstable; it loses accuracy
  instead. BUG keeps 1e-8 to 1e-10 accuracy at steps 4-8 times larger. The A2 counterpart (Γ = 5)
  is dr6w.

**A1 time-to-accuracy (dr6a 17496168: one job, every arm in sequence on th-cl-dell06, a Xeon
Gold 6130, with JIT warm-up 2; A1 L = 12, U = 4, κ = 2, product start, Jt = 1, maxdim 2048,
tol_trunc 1e-9; figures `a1_tta`).** Neither integrator is capped. Errors are measured against the
dr6b reference, whose own error of about 2e-7 floors TDVP2's finest arm; dr6d refines it.

| arm | dt | error | wall [s] |
|---|---|---|---|
| BUG | 0.05 / 0.1 / 0.2 | 2.7e-6 / 3.3e-5 / 6.7e-4 | 332 / 350 / 433 |
| TDVP2 | 0.0125 / 0.025 / 0.05 / 0.1 | (2.6e-7) / 4.5e-7 / 3.0e-6 / 2.3e-5 | 605 / 368 / 300 / 307 |

At matched error (3e-6 and 3e-5) TDVP2 is about 10% cheaper on this point. BUG's error is its
first step out of the product state (one-step order 4). Its wall hardly changes with dt, because a
larger step carries a larger D (1108 at dt 0.2 against 854 at 0.05). TDVP2's slope is 2.7-3.0
above the reference floor.

**Binding cap (dr6c 17496162, figures `a1_bindcap`).**

| point | BUG capped from | TDVP2 capped from | discarded total BUG / TDVP2 | wall BUG / TDVP2 [s] | MaxRSS BUG / TDVP2 |
|---|---|---|---|---|---|
| L = 12, κ = 2 | t = 0.4 (work) | t = 0.75 | 0.040 / 0.150 | 948 / 2024 | 4.5 / 2.6 G |
| L = 16, κ = 8 | t = 0.45 (work) | t = 0.95 | 9.0e-6 / 2.0e-5 | 2990 / 6308 | 7.8 / 2.9 G |

- The DD sketch time scales as D^2.2 (L = 12) and D^1.4 (L = 16) for D ≥ maxdim/4, close to the
  plan's "quadratic-in-D fixed-width regime". TDVP2's step time scales as D^1.0.
- There is no reference beyond Jt = 1, so after t = 1 the accuracy at the cap is read only from
  the discarded norms. Before t = 1, the dr6b comparison above applies: BUG is work-limited
  earlier and less accurate.

**A2 time-to-accuracy point (USER 2026-10-25).** A2 at L = 32 and finite density (8 excitations)
reaches maxdim 1024 by Jt ≈ 0.25 under both integrators: TDVP2 dr6q row 2 took 7.1 h to Jt = 2
and discarded up to 1.8e-4 per step after the cap. A non-binding curve there is not feasible. The
A2 curve uses A2 at L = 24, ν = 1/4 (6 excitations), maxdim 512, where the state needs D = 378
(dr6u row 3). The dr6w accuracy ladder fixes the arms, and the matched arms rerun as one
sequential timing job, as dr6a did for A1.

**Pilots submitted:**

- dr6j 17496962, B1 L = 12: U ∈ {-4, +4} × κ ∈ {0.5, 2, 10}, 20 trajectories each, maxdim 256.
- dr6k 17496964, B2 L = 24: ν ∈ {1/8, 1/4} × Γ ∈ {0.5, 2}, 20 trajectories plus a conditional
  no-jump run each.

# Task 7 (G4): scheduler smoke and the tol_trunc study

USER 2026-10-25 chose "G4 essentials" (about 400 CPU-h) over the plan's grid. The grid (A1
L = 24-40, A2 L = 48-96, B1 L = 24-40 with CI-resolving ensembles, B2 L = 48-64, each at three
tol_trunc) is far beyond the measured costs: B1 L = 20 costs 0.5 CPU-h per trajectory, and A2 at
L = 32 and finite density is cap-bound at once.

The scale is the scheduler smoke, then the three-point study tol_trunc ∈ {1e-5, 1e-7, 1e-9} at one
validated point per model. tol_basis, tol_root and the time step are held fixed and tighter than
the differences compression causes. The points are:

- A1 L = 24, κ near the Zeno crossover;
- A2 L = 48, dilute (2 excitations), to Jt ≈ 20;
- B1 L = 20, at the pilots' transport extremum, 20-40 trajectories;
- B2 L = 32, ν = 1/8, conditional plus 20 trajectories.

BUG and TDVP2 both run the deterministic (no-jump) points.

**Scheduler smoke (dr7s 17496996, `4a76d77`; plan: "one short trajectory, dry-run output and
restart, memory budget check").** `benchmarks/dissipative_restart_check.sh` runs at each production
point at production size over a short window:

- A1 L = 24 κ = 8, maxdim 1024;
- A2 L = 48 with 2 excitations, maxdim 256;
- B1 L = 20, two trajectories;
- B2 L = 32 ν = 1/8, two trajectories.

Each check compares an uninterrupted run with one stopped at a checkpoint and resumed, plus the
changed-tol_trunc refusal control. The memory budget is read from each run's printed peak RSS.

**Fix regression tests complete.** dt6t 17496046 passed `test_dd_cbe` (with the new T2.4 test),
`test_dd_cbe_controller` and `test_dissipative_t2`, `t4` and `t5`. dt6u 17496932 passed
`test_dissipative_t3_a1`, `t3_a2` and `t2_g12` (U10 order). Every relevant test file passes on
`a2d8c78` / `4a76d77`.

**A1 references converged (dr6d 17496931).** Successive halvings differ by 2.2e-7 (dt 0.025 to
0.0125) and then 1.4e-8 (0.0125 to 0.00625), a ratio of 15 (one-step order 4). The dt 0.00625
reference is therefore good to about 1e-9; none of them was capped (peak D 1407). Against it,
TDVP2 gives 4.6e-8, 3.7e-7, 3.0e-6 and 2.4e-5 at dt 0.0125-0.1 (slopes 3.0, 3.0, 2.98), and BUG
gives 2.9e-6, 3.3e-5 and 6.7e-4 at dt 0.05-0.2. The final timing curve, dr6e 17497216, reruns
every arm in one job and adds BUG dt 0.025 and 0.0125, so that BUG also reaches TDVP2's two most
accurate arms.
