# QTT four-task plan -- source audit (U07) and plan-to-code map

Plan: `qtt_dissipative_cbe_bug_four_task_implementation_plan.md`. Branch `plan/qtt-task-1` (all four
tasks; the checkout is shared with the dissipative-plan session, so commits go to the branch ref
without switching HEAD). Status words: **present** (used as is), **adapted** (changed for this plan),
**new**, **missing**.

## U07 -- what existed before this plan

| plan item | status | source |
|---|---|---|
| implicit trapezoid root | present (LINEAR only) | `BondUpdateBUG/expv_root_methods.jl` `:trapezoid` -- the rational stability function in one Krylov stage |
| 2-stage Radau IIA | present (LINEAR only) | `:radau2` (rational), `:radau2_gmres` (coupled stages, `radau2_gmres_expv`) |
| 3-stage Radau IIA | present (LINEAR only) | `:radau3` (rational) |
| matrix-free GMRES | present (linear, shifted) | `_shifted_gmres` in `expv_root_methods.jl` |
| order 2/3/4 BUG dispatch | present (linear generator) | `cbe_bug_step!(...; order)` -> `krylov_closure.jl` frontier closure |
| nonsymmetric DD kernels | present; verified under `:none` (U04) | `dd_cbe.jl` `_stream_left/_stream_right` (the mirror applies `X` itself, `Om' X`, not `H'`), `_dd_select`, `_dd_controller` |
| Arnoldi for non-Hermitian | present | `arnoldi_expv`, `hermitian = false` |
| full-image recycling | present, default OFF (kept off) | `krylov_recycle.jl`, `krylov_recycle = false` |
| nonlinear stage solve (GP) | **missing -> new** | `qtt/qtt_flows.jl` `rk_stage_solve` (Newton-GMRES, doubled-real) |
| QTT geometry, operators | **missing -> new** | `qtt/qtt_grid.jl`, `qtt_cores.jl`, `qtt_ops1d.jl`, `qtt_models.jl` |
| `:none` dense tensor bridge | **missing -> new** | `qtt/qtt_telum.jl` |
| TDVP2 backward-stage raw amplification | adapted | `tdvp2_step!(...; backward_norms)` |

## Plan-to-code map

### Task 1
- 1.1 grouped ordering (LSB first, plan formula), interleaved/MSB diagnostics: `qtt_grid`, `QTTGrid.sites_x/sites_y`; weights kept outside (`qtt_cell`, `qtt_l2norm`).
- 1.1 MPOs: `qtt_shift` (carry automaton), `qtt_derivative` (`:periodic`, `:dirichlet0`, `:onesided`), `qtt_coord_diag`, `qtt_embed` (`D_xv` = product of chains), `gp_model` (kinetic, trap, `-Omega L_z`, `-mu`), `GPPaddle`/`gp_paddle_op`, `heston_op` (+ `adjoint = true`, mass-conserving transpose).
- 1.1 environment action: the solver's own MPO environments (`apply_h_two_site`); the summands are direct-summed into one MPO (`op_lincomb`), whose environments are the direct sum of the summands' -- cached per step by the solver, rebuilt when `H[psi, t]` changes.
- 1.2 hybrid controller D3: `_dd_hybrid`, `DDMemory`, `dd_policy = :hybrid` (preflight exhaustive / mirrored, predictor one-shot, independent holdout, bounded enlargement); D0 `exact = true`; D1 `:adaptive`; D2 `dd_width`; no-CBE control `dd_policy = :off`.
- 1.2A first impressions: `benchmarks/qtt/first_impressions.jl` -> `qtt_cbe_nosym_ablation.csv`, `dims.csv`, `steps.csv`; report `docs/qtt_cbe_nosym_first_impressions.md`.
- 1.3 Arnoldi + time-weighted stop: unchanged (present); recycling off.
- 1.3A density and nonlinear action: `tt_density` (zip-up, bounded error, retry), `gp_hamiltonian_op/gp_hamiltonian` (cache keyed by state tag and t: `GPHamiltonianCache`), `gp_vector_field`, `gp_jvp` (real-linear), `gp_density_audit`; mode A `gp_bug_step!` (`:frozen_density`), mode B `qtt_bug_step!`.
- 1.4 tests: `tests/sweeps/test_qtt_task1.jl` (U01-U06).

### Task 2
- 2.1-2.3 oracles: oscillator modes (`test_qtt_task2.jl` V01), discrete-exact plane wave (V02), `heston_riccati`, `heston_affine_field`, `heston_call`, `heston_mass`, `heston_moment` (V03).
- 2.3A the three nonlinear methods: `qtt_tdvp2_step!` (`:lagged`, `:consistent`), `qtt_bug_step!`; dense RK4 reference of the same semi-discrete system (`benchmarks/qtt/common.jl` `gp_dense_rk4`).
- 2.4 roots: `RK_TABLEAUX` (`:trapezoid`, `:radau2`, `:radau3`), `qtt_root_for_order`; the step input projected (`frame_project(fr, nothing, psi_n)`), basis predictor-corrector passes (`npc`).
- 2.5 amplification `A_back` (dense TDVP2 and `tdvp2_step!(...; backward_norms)`), stabilized control `qtt_split_tdvp_step!`, global `tt_expv`.
- 2.6 gates: V01-V03, V07, V08 in `test_qtt_task2.jl`; V04-V07 at study sizes `benchmarks/qtt/task2_study.jl`.

### Task 3
- 3.1 `gp_ground_state!` (uniform/geometric schedules, rejection), `gp_gradient_descent!`, QTT-TDVP imaginary time (`method = :tdvp`), `gp_observables` (E, mu, R_mu); driver `benchmarks/qtt/gp_ground_state.jl`.
- 3.2 `benchmarks/qtt/vortex_shedding.jl` (paddle, dense reference, vortex charges `vortex_count`, energies, `P_ext`, chi profiles).
- 3.3 `benchmarks/qtt/heston_pilot.jl` (pricing vs `heston_call`, Greeks, forward moments, rank up/down).

### Task 4
- `benchmarks/qtt/production.jl` (manifest, ledger, step-doubling acceptance, checkpoint/resume, snapshots by `qtt_sample`), `benchmarks/qtt/restart_check.jl`.

## Defects found in existing code (reported, not patched in the core)
- **Order-p frontier closure under `:none` near saturation** (job qt1c 17603921, GP oscillator,
  L = 12, order 4): `_frontier_right` builds `V0 = Z[j] Pm` with the OLD frame's width (36) on a site
  whose new right link leaves a 16-dimensional local space, and the frame guard throws in
  `_krylov_enrich -> _extend_frame -> _cap_to_local_room`. Not reproduced in the U(1) campaigns.
  `qtt_bug_step!` catches exactly this guard and builds the equivalent order-p basis from order-1
  frames that keep the global Krylov vectors `(tau H)^j psi`, `j < p` (`global_krylov_states`,
  dissipative plan U10); every use is counted in the step's `basis_fallback`.
- `qtt_bug_step!`'s frame-building step underflowed its linear root to an exactly zero matrix at
  `h lambda = -1e3` (V08) -> Telum's root SVD refused it; fixed by shifting the frames' generator by
  `<H>` (Krylov spaces unchanged).

## Known limitations (stated, not hidden)
- Second-order FD Laplacian, not the paper's QFT Laplacian.
- The paddle shape is an engineering fixture (Appendix B gives no parameters).
- The imaginary-time Jacobian omits `d mu` (the residual is exact; Newton converges linearly there).
- `qtt_tdvp2_step!` rebuilds environments per substep (O(L^2) per sweep); it is a reference
  baseline, the library `tdvp2_step!` is the fast lagged TDVP2.
