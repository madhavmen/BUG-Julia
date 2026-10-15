# Plan audit: `simulation_plan_2.md` (2D quantum-magnet benchmark)

One row per plan requirement, checked BY CALL SITE (the code on the production path). Status:

- `DONE` -- on the production path AND exercised by a test or validation job that calls it
- `PENDING` -- implemented, the test / validation job that calls it has not come back yet
- `CARRIED` -- implemented and audited under an earlier plan (reference given), unchanged here
- `OPEN` -- not implemented yet

Tasks 1-3 of this plan are the same deliverables as `simulation_plan.md` /
`adaptive_cbe_krylov_implementation_plan_first_test.md`; they are audited in
`ADAPTIVE_PLAN_AUDIT.md` (Stages 1-6, all DONE) and are `CARRIED` here. The general-order
Krylov-closure step used by the `bug2` / `bug4` arms is audited in `KRYLOV_CLOSURE_AUDIT.md`.

## §1-§4, Tasks 1-3 (API, RSVD Tests A/B, Krylov reuse C/D, compression)

| # | Requirement | Evidence | Status |
|---|---|---|---|
| 1 | Three tolerances + `maxdim`, legacy knobs refused alongside them | `resolve_budget`; ADAPTIVE_PLAN_AUDIT 1.1-1.11 | CARRIED |
| 2 | Task 1: sector coverage, Test B holdout, Test A tail | ADAPTIVE_PLAN_AUDIT Stage 2 | CARRIED |
| 3 | Task 2: CBE directions reused in K/L Krylov, Tests C/D | ADAPTIVE_PLAN_AUDIT Stage 3 | CARRIED |
| 4 | Task 3: lossless root, one budgeted compression, midpoint invariant | ADAPTIVE_PLAN_AUDIT Stages 4-5 | CARRIED |

## §5 Task 4 -- ground states and spectroscopy (audited 2026-10-15)

| # | Requirement | Evidence (call site) | Status | Note |
|---|---|---|---|---|
| 4.1 | CBE-DMRG ground state shares the adaptive RSVD discovery/selection (Tests A/B) | `dmrg_cbe1s_sweep!`: `resolve_budget` -> `basis_allocation(...; tau = 1)` -> `expand_bond!(...; exkw..., ledger = bondlg)` with `alloc`, `workcap = work_ceiling(maxdim, rmax)`, `stol_pre = 0` | PENDING | job 17208254, `test_adaptive_dmrg.jl` |
| 4.1b | Ground-state tolerances medium-loose by default, looseness measured (user decision 2026-10-16) | `DMRG_TOL_DEFAULTS` (provisional 1e-3 / 1e-4) applied after `resolve_budget`; `benchmarks/ground_state/adaptive_dmrg_tolerance.jl` vs exact N = 20 (heis, xx, square 5x4, J1-J2 5x4) | PENDING | jobs 17208876 (smoke + tests) -> 17208877 (study); defaults to be set from its table |
| 4.2 | Its local stopping criterion is a Ritz residual, not the BUG root error | `lanczos_lowest(...; restol)` unchanged; `restol`/`maxiter` not routed through `resolve_budget` | DONE | design statement in the docstring |
| 4.3 | Budgeted split, no hidden truncation knob; `maxdim` binding reported | `split!` -> `_budget_svd(A, legs, maxdim, n2 -> tol_trunc/sqrt(L-1)*sqrt(n2))`; `capped` -> `:trunc`, `maxdim_reached` | PENDING | the same `_budget_svd` now serves `truncate_recursive!` (refactor, `test_adaptive_compression.jl` in the same job) |
| 4.4 | Per-bond telemetry (Test A/B, probes, widths, discarded) per sweep | `_log_bond!` + `(r_old, r_final, discarded)`; `DMRGCBERun.status` / `.ledgers` | PENDING | |
| 4.5 | Legacy fixed-knob DMRG unchanged | every existing caller names `trunc_thresh` -> legacy; `rmax` read before the gauge as before | PENDING | `test_sweeps.jl` DMRG sets in the same job |
| 4.6 | Converged energy AND variance | `mpo_variance` (two-layer MPO environment), printed by the driver (`VAR0`) and written to the cost CSV | PENDING | test vs dense `<H^2>-<H>^2` (chain, 3x3 J1-J2) in `test_adaptive_dmrg.jl` -- added after 17208254 was submitted, runs in the next job |
| 4.7 | Several initial guesses for frustrated models, metastability reported | `fig45_2209.jl`: `NSTART` (default 3 when `J2 != 0` or kagome), Neel + seeded random S^z = 0 products, spread printed | PENDING | validation job |
| 4.8 | `S^zz(q,omega)` from `C(t)`, U(1)-preserving excitation | `fig45_2209.jl` `correlator` (local `S^z_c` source, plan's allowed alternative) + Eq. (14) transform | CARRIED | campaign arXiv:2209.00739 |
| 4.9 | Common conditions for all integrators (same GS, MPO, source, grid, window, cap) | one `psi0`, one `W`, one `ETA2`, one `DCAP` for every arm; only `bug4` changes dt (user decision: 4th order at larger dt), recorded per row (`dt` column) | DONE | |
| 4.10 | Adaptive CBE-BUG arms, ordinary and midpoint | `steps["bug1"]`, `steps["bugmid"]` with `akw = (tol_basis, tol_trunc, tol_expv, maxdim)` only | PENDING | |
| 4.11 | Higher order (§9 Phase 7, pulled forward by user): order 2 and 4, larger dt for order 4 | `steps["bug2"]`, `steps["bug4"]` (`order = 4`, `armdt = F_DT4`) | PENDING | |

## §6 Task 5 -- validation tests used so far

| # | Requirement | Evidence | Status | Note |
|---|---|---|---|---|
| U17 | Independent edge checks: NN, J2 diagonals, kagome, cylinder wrap | `bond_length_gate` (every lattice, run before physics in the driver); `test_adaptive_dmrg.jl` square-J2 edge list; `test_kagome.jl` (existing) | PENDING | |
| U18 | Small-system operator consistency vs sparse | `pairs_sparse` and the MPO from ONE `Jm`; DMRG E0 vs exact, refused if below | PENDING | |
| U19 | Fourier/source normalisation | `G(c,0) = 1/4`, `sum_i G(i,0) = 0` gates (existing) | CARRIED | |
| U20 | Correlator vs independent reference | `exact_correlator` per arm on its own grid -> `gerr_max`, `gerr_T`; exact `S(q,w)` written as arm `exact` | PENDING | N <= `F_EXACT_N` (20) |
| U22 | Order checks, measured not assumed | `KRYLOV_CLOSURE_AUDIT.md` IT2 (chain); on 2D: `gerr` of bug1/bug2/bug4 vs dt | OPEN | needs a dt ladder on the 4x4 square |

## Models

| Model | Code | Status |
|---|---|---|
| Square Heisenberg | `square_cylinder_couplings` | CARRIED |
| Square J1-J2 | `square_cylinder_couplings(...; J2)` diagonals `(x,y)-(x+1,y+-1)`, wrapped | PENDING |
| Kagome | `kagome_cylinder_couplings`; `lattice2d.jl` `kagome_lattice` (positions with sublattice offsets, extended-zone path, `nsites`, `centre_site(lat, ...)`) | PENDING |
