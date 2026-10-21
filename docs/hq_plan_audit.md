# Audit: `cbe_bug_high_order_rsvd_parallel_tasks.md`, Tasks 1–2

Branch `plan/hq-tasks-1-2`, commit `d717376` (cut from `plan/imag-open-t3` @ `882dfe0`).
Status per plan item: **DONE** (code on the production call path + a test that calls it),
**PARTIAL**, **DIVERGES**, **DEFERRED(Tn)** (belongs to a later task of the plan).
Evidence is `file:symbol` at `d717376`. Test evidence: `tests/sweeps/test_dd_cbe.jl` (cluster
job 17409190; baseline comparison 17409191 on `882dfe0`, results at the end).

## T1.1 Call graph of one `cbe_bug_step!` (adaptive path, `order = 1`)

| stage | function (file) | status |
|---|---|---|
| entry, API resolution | `cbe_bug_step!` ([sweeps/cbe_bug.jl](../src/RSVDCBEBondUpdate/sweeps/cbe_bug.jl)), `resolve_budget`, `_resolve_parallel`, policy resolution | existing (+ new `tol_root`, `parallel` symbols, `cbe_policy`) |
| K half-sweep, left → root | `half_left!` (closure in `cbe_bug_step!`) | existing |
| frame snapshot per bond | `_frame_from` (zero_site_core.jl) | existing |
| **CBE (K-sweep, A grows)** | `timed_expand(...; side = :left)` → `dd_cbe_expand(..., :left)` (dd_cbe.jl) → `_certified_range` (cbe_core.jl) → `_dd_probe` / `apply_two_site_sketch!` (`_stream_left`) / `_dd_select` (`_thin_qr` ×2) | **new** (was `cbe_expand`, both sides, SVD) |
| local one-site Krylov (Tests C/D) | `_krylov_enrich` (cbe_bug.jl) on `OneSiteOp` (`apply_one_site` → `MPOOneSiteH`) | existing; operator wrapped (new) |
| environment push | `push_left_channels` (mpo.jl) | existing |
| L half-sweep, right → root | `half_right!`; **CBE (B grows)** `dd_cbe_expand(..., :right)` → `_stream_right`; `_krylov_enrich` | new CBE / existing Krylov |
| original-state projection | `_overlap_left` / L-sweep carry `D` → `S0 = AL·AR`; `_projection_miss` → `ledger.proj_err` | existing |
| higher-order stage predictor | `_krylov_closure!` / `_frontier_sweep!` (krylov_closure.jl), `order > 1` only | existing; still `cbe_expand` (SVD) → see T2.5 row |
| root | `expv` (BondUpdateBUG/expv.jl) on `RootOp` (`apply_zero_site` → `zero_site_h`) | existing; operator wrapped (new) |
| final compression | `truncate_recursive!` (zero_site_core.jl) | existing |
| step record | `StepRecord` (dd_cbe.jl) in `CBEBugSweepInfo.record` | **new** |

"Recycled Krylov" (the T1.1 question): **not gauge-transported subspace reuse.** Each bond's
`_krylov_enrich` starts from `Q0 = U_ex` (the CBE frame) and a fresh seed; no Krylov vector of a
previous bond, half-sweep or order round is transported. Reuse of the CBE directions as the
initial frame is real (`_krylov_enrich(Kex, ..., ex.U_ex, ...)`); cross-round recycling is
DEFERRED(T4).

## T1.2 Tensor conventions

| object | legs | gauge | notes |
|---|---|---|---|
| `U0` / `A` (left frame) | `(link_l, site_l, bond)` | left isometry, `A'A = I` over legs 1–2 | bond leg arrow from `svd`'s `U` |
| `V0` / `B` (right frame) | `(bond, site_r, link_r)` | right isometry, `B B' = I` over legs 2–3 | stored as ROWS: `B_aug' = [B', Q]` means rows `[B; Q_rows]` |
| `S0` | `(bond_l, bond_r)` | centre core | `Θ = U0·S0·V0` (`frame_theta`, never formed by the DD path) |
| probe `Ω_R` | `V0` layout `(g, site_r, link_r)` | — | `Ω'` (conjugate) is contracted into the environments |
| probe `Ω_L` | `U0` layout `(link_l, site_l, g)` | — | |
| `Z = X Ω_R` | `(link_l, site_l, g)` | — | `_stream_left` |
| `Z = X' Ω_L` | rows `(g, site_r, link_r)` = `(Ω_L' X)` | — | `_stream_right`; the row space is the column space of `X'Ω_L` |
| MPO env | `(bra, w, ket)`, bra primed | built in the bond's canonical gauge | boundary env is `nothing` (dim-1 legs dropped) |
| charges | a sector label depends on the leg arrow (`align_charge`) | | `_thin_qr` builds its new leg in the frame's arrow convention |

Root identity: `S_{0,p} = (U_p' U_n) S_n (V_n' V_p)` is evaluated as `S0 = _overlap_left(psi, W, c) ·
AR`, contracting the final frames with the ORIGINAL `psi` (not a predictor); `ledger.proj_err =
||(1-P)psi|| / ||psi||` checks it. Test: `test_dd_cbe.jl` "T1.2" (complex warmed state, centre at
site 1 and at site L, `proj_err < 1e-12`, norm preserved to `1e-10`). **DONE** for the K/L
frames of `order = 1`; `order > 1` uses the same `_project_root` path (not re-tested here).

## Task 1

| item | status | evidence |
|---|---|---|
| T1.1 call graph, existing/partial/missing | DONE | table above |
| T1.1 recycled Krylov = real reuse? | DONE (answer: no) | above; `_krylov_enrich` seeds from `U_ex` only |
| T1.2 conventions documented | DONE | table above |
| T1.2 projection identity with original state, complex, both gauges | DONE | test "T1.2" |
| T1.2 randomized U(1) blocks | PARTIAL | warmed TDVP2 states (complex, several sectors), not random-block MPS |
| T1.3 `apply_two_site_sketch!`, `apply_one_site!`, `apply_root!` | DONE (interface) | dd_cbe.jl `TwoSiteSketchOp`/`OneSiteOp`/`RootOp`; wired: CBE, both half-sweep Krylovs, root |
| T1.3 explicit adjoint | DONE | `TwoSiteSketchOp(...; adjoint)`; test U01 both orientations |
| T1.3 symmetry tag | DONE | `op.symmetry` checked on every apply |
| T1.3 stage/environment version | PARTIAL | `version` = bond recorded; invalidation is by immutability (operators are rebuilt per visit), no version comparison |
| T1.3 scratch buffers | DIVERGES | Telum contractions allocate their result; outputs go to a caller `Ref`, no reusable block buffer |
| T1.4 counts: MPO contractions, one-site, root, QR ranks, sector additions | DONE | `KernelCounters`; test "T1.4" |
| T1.4 stage/temporary/final D, root iterations | DONE | `StepRecord.D_old/D_stage/D_final/root_iters` |
| T1.4 bytes allocated, time per kernel | DONE | `t_*`/`b_*` per kernel (`Base.gc_bytes` deltas, process-wide) |
| T1.4 peak live memory | PARTIAL | `Sys.maxrss()` = process high-water mark, not per step |
| T1.4 one typed step record | DONE | `CBEBugSweepInfo.record::StepRecord` |
| T1.4 no logging allocations in hot loop | DONE | counters are plain mutable fields; no logging in kernels |
| Acceptance: record reproducible from fixed seed | DONE | test "T1.4" (same `rng` ⇒ identical counters, ranks, state) |
| Invariant 1: `tol_root`, `parallel = :half_sweeps`, `:full` only after acceptance | DONE | `tol_root` alias, `_resolve_parallel` refuses `:full`; test "API" |
| Invariant 4: separate limit statuses | DONE | `StepRecord.limits` from `_step_limits` |

## Task 2

| item | status | evidence |
|---|---|---|
| T2.1 project the thin probe first: `Ω − B'(BΩ)` | DONE | `_dd_probe` → `perp_component_right(V0, ·)`; test U01 (`P_R^D Ω~ = Ω~`, `BΩ~ = 0`) |
| T2.1 streamed action, then left projection of the thin result | DONE | `_stream_left` then `_dd_select` → `perp_component(U0, Z)` |
| T2.1 mirrored adjoint path | DONE | `_stream_right`, `perp_component(U0, ·)` on the input, `perp_component_right` on the output |
| T2.1 check vs explicit small `P_L^D X P_R^D Ω` | DONE | test U01, bonds 1, 4, 7 (both boundaries), both directions, `< 1e-12` |
| T2.1 no full action tensors / complement bases | DONE | `Θ` and `HΘ` never formed (probe → `V0` → `U0 S0`); projectors applied as two thin contractions |
| T2.2 same projected probe for every MPO channel | DONE | channels summed inside the contraction over `w`; test U01 equals the materialised full-`H` action on a w = 5 MPO |
| T2.2 right projection once, not per channel | DONE | one `perp` per probe in `_dd_probe` |
| T2.2 reusable U(1)-blocked accumulation buffers | DIVERGES | Telum allocates per contraction (as T1.3) |
| T2.3 probes in fused sectors incl. zero kept multiplicity | DONE | `_dd_probe` allots by discarded dim `dfull − dkept`; test "T2.3" (Neel: new sector found, `n_sector_new ≥ 1`) |
| T2.3 nothing in forbidden / zero-discarded blocks | DONE | sectors with `dC = 0` get no columns; `nothing` when the whole discarded space is empty |
| T2.3 adaptive extra probes where DD residual is large | DONE | `_certified_range` Test B targets sectors by holdout residual (`_target_sectors!`) with DD-projected holdouts |
| T2.3 split mixed-charge sketch columns before QR | DONE (structural) | probe columns carry one fused charge each (block-sparse); `_thin_qr` works per sector |
| T2.4 sectorwise thin Householder QR, numerical rank | DONE (abelian) | `_thin_qr`: `qr(M, ColumnNorm())` per sector, keep `|R_kk| > 100 eps max(‖Z‖,‖Y‖)` |
| T2.4 reorthogonalisation against old frame | DONE | second `P^D` + QR pass in `_dd_select`; test U03 `A'Q ≈ 0`, `BQ_R ≈ 0` |
| T2.4 no physics cutoff | DONE | only the machine-precision floor; nothing ranked |
| T2.4 `Q'Q ≈ I`, range kept | DONE | test U03 |
| T2.4 SU(2) | PARTIAL | non-abelian blocks are not plain matrices: falls back to `_preselect_*` at `stol = 0` (numerical-rank SVD orthonormalisation), reported as `record.qr_backend = :svd_orth` |
| T2.4 right QR a mirror, one QR per orientation | DONE | `dd_cbe_expand(..., :left)` in the K-sweep, `:right` in the L-sweep; the other frame is returned untouched (test "T2.3") |
| T2.5 production path `dd_qr_streamed` | DONE for the K/L half-sweeps | default on the adaptive path; `record.policy` |
| T2.5 exclude 5-SVD shrewd / `Q'G` SVD / dense DD on production | DONE for the K/L half-sweeps | `dd_cbe_expand` calls none; test U02 asserts Telum SVD cost 0 inside a (cache-warm) DD expansion |
| T2.5 `one_sided_qr_streamed` ablation | DONE | `cbe_policy = :one_sided_qr_streamed` (unprojected probe, everything else shared) |
| T2.5 full-DD exact tiny reference, tests only | DONE | `dd_reference_sketch` (materialised `HΘ`); `dd_cbe_expand(...; exact = true)` |
| T2.5 original SVD selection for ablation | DONE | `cbe_policy = :svd_select`; also forced on the legacy path |
| T2.5 record policy, never switch silently | DONE | `record.policy`, `record.qr_backend`, `record.closure_policy`; legacy + DD policy refused |
| T2.5 `order > 1` frontier sweeps | **DIVERGES → DEFERRED(T4/T5)** | `_frontier_sweep!` still calls `cbe_expand` (two-sided SVD selection), reported as `record.closure_policy = :svd_select` |
| Acceptance: DK-dominant / DD-small fixture | PARTIAL | test "DK/DD": a kept-space probe has DK ≠ 0 one-sided and exactly 0 DD; the exact DD range is captured whole; one-sided keeps ≥ as many directions. A fixture where DK ≫ DD by construction is not built |
| Acceptance: complex adjoints | DONE | complex states and probes, test U01 right direction |
| Acceptance: missing-sector discovery | DONE | test "T2.3" and the two-spin test |
| Acceptance: no CBE SVD | DONE (abelian) | test U02 |
| Acceptance: no full tensors/projectors, no O(D³) scratch | PARTIAL | no `Θ`/`HΘ`/projector formed; `fusion_basis` (cached per leg structure) is a `(dD)²` identity used to build probes — bookkeeping, measured separately in P06 |
| Acceptance: near-quadratic sketch slope at fixed d, w, ℓ | DEFERRED(P06) | `KernelCounters.t_stream`/`b_stream` record it per step; the D-sweep timing job is not run |
| End-to-end physics | DONE | two-spin entangling step exact to `1e-10`; L = 8 Heisenberg vs exact propagation for all three policies |

## Behaviour changes to note

* The adaptive K-sweep no longer widens the right frame (and the L-sweep the left one). The
  one-site Krylov at bond `i` of the K-sweep therefore runs with its right environment through
  `V0` rather than a CBE-widened `V_ex` (the standard BUG K-step operator); the DK directions it
  used to reach through `V_ex` come from the one-site Krylov and the DD ones from the QR.
* `CBEBugSweepInfo` has a new last field `record`.

## Confirmation results

* hq12br 17409190 (`d717376`): 73 pass / 9 fail / 1 error. U01 bond 4, U03, DK/DD and T1.4 failed
  because the fixture `_hb_bond` was warmed at an unbinding cap: the L = 8 centre bond was full
  (D_4 = 16), both `P^D` vanished and every check compared roundoff. Fixed in `94e35fc` (warm-up
  `maxdim = 4`, `_unsaturated` asserted at every DD fixture).
* hq12b 17409443 (`94e35fc`): **97 pass / 2 fail**. Every fixture test passes; the two failures
  are the end-to-end assertions `err(dd) < 1e-4` and `err(dd) <= 10 err(svd_select)`.
* HQGAP (`benchmarks/hq_dd_gap_l8.jl`, Heisenberg L = 8 from Néel, T = 0.2, maxdim 64, error vs
  exact):

  | dt | dd_qr_streamed | one_sided_qr_streamed | svd_select |
  |---|---|---|---|
  | 0.1   | 6.13e-3 | 6.13e-3 | 9.30e-5 |
  | 0.05  | 1.53e-3 | 1.53e-3 | 1.16e-5 |
  | 0.025 | 3.83e-4 | 3.83e-4 | 1.46e-6 |

  Identical (to ~9 digits) at `tol_basis` 1e-6, 1e-8, 1e-10; no limit status fires; D_final 14
  (DD) vs 11 (SVD). DD converges as h² exactly (ratio 4.00), SVD as h³ (ratio 8.0). DD ≡
  one-sided, so the opposite-leg projector is not the cause. Reading: not a basis-selection
  defect (tolerance-independent, ledger silent, DD range certified) but the K/L-step
  operator: on the SVD path the one-site Krylov at bond `i` runs through the CBE-widened `V_ex`
  (a two-site-like K-step reaching `H²` two-site directions); on the DD path it runs through `V0`
  (the standard BUG K-step), and the DD QR adds only first-order `P_L^D H Θ P_R^D` directions.
  This is the leakage T3.5 (Taylor-leak) and the order-p rounds of T4/T5 address.
