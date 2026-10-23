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
| T2.5 `order > 1` frontier sweeps | **DIVERGES → DEFERRED(T4/T5)** | `_frontier_sweep!` still calls `cbe_expand` (two-sided SVD selection), reported as `record.closure_policy = :svd_select`. Resolved in T4.1 (`b04e5da`) |
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
* USER 2026-10-21: keep the plan design. `c05c87c` grades the end-to-end test on what the plan
  supports (svd < 1e-4, DD dt-convergence ratio ≥ 3.5, tol_basis-independence, DD ≡ one-sided)
  and marks the 10×-svd parity `@test_broken` (T3.5/T4).
* hq12c 17410287 (`c05c87c`): **test_dd_cbe.jl 113 pass / 1 broken / 0 fail**, MaxRSS 1.6 G.
  Tasks 1–2 CONFIRMED.

## Task 3

Branch `plan/hq-task-3` (stacked on `plan/hq-tasks-1-2`), commits `22aca99`, `c043e90`, `c570c3c`,
`092684e`.
Tests: `tests/sweeps/test_dd_cbe.jl` "T3.*" on `_hb_wide()` (L = 12, cap 16, bond 6: at L = 8 cap 4
the discarded probe space is 4-dim and a width-4 holdout spans it).

| item | status | evidence |
|---|---|---|
| T3.1 small Gaussian block, `P_R^D` on the probe, `P_L^D` on the thin output | DONE | `_dd_controller` (dd_cbe.jl) through `_dd_probe` / `_stream_*` / `_dd_select` |
| T3.1 append only when the DD holdout fails | DONE | controller loop; test "T3.1" (`length(trace) ≥ 2`, `:met`) |
| T3.1 promote holdouts only after fresh validation columns | DONE | holdout → candidate on a failed check, a new holdout drawn first; test "T3.1" |
| T3.1 never re-stream a cached column | DONE | test "T3.1": `n_stream_cols == n_probes + n_holdout` |
| T3.1 QR-only, rank from the stopping rule | DONE | `_thin_qr` numerical rank only; test U02 (no SVD) |
| T3.2 one `tol_basis`, linear shares over bonds × rounds | DONE | `basis_allocation(...; rule = :linear)` for the DD policies; test "T3.2" (shares sum to `tol_basis`) |
| T3.2 RSS only where its assumptions hold | DONE | `:rss` kept for `svd_select`; `StepRecord.budget_rule` logs it; no new public knob |
| T3.3 holdout width s small (2, 4, 8), projections on thin objects only | DONE | `RC._DD_HOLDOUT`; test "T3.3" |
| T3.3 escalation on failed/borderline check, new sector | DONE | sector targeting by holdout residual (`_target_sectors!`) inside the controller |
| T3.3 false acceptance on missed-sector fixtures | DONE | test "T3.3": blind-holdout control fails, honest holdout s = 8 has 0/10 false acceptances |
| T3.3 no full `‖G_DD‖` at production D | DONE | `G` formed only in the test closures (`_dd_closures`), never in `dd_cbe_expand` |
| T3.4 explicit `:met` / `:work_limited` / `:basis_limited` / `:empty`, bounded loop | DONE | test "T3.4" (unreachable share, work ceiling, empty discarded space; streamed columns bounded) |
| T3.4 timestep rejection/retry on a limit | PARTIAL | limits reach `StepRecord.limits`; no retry driver (none in the plan's call path yet) |
| T3.5 Taylor leak in small tests | DONE | test "T3.5": dense L = 8, `η_m = h^m/m! ‖(I−P)H^m ψ‖` for m ≤ 3 on the step's root space |
| T3.5 tighter `tol_basis` exposes the temporal limit | DONE | first-order leak at the rounding floor at tol 1e-4 and 1e-8; the dominant leak is m = 3 (1.9e-11 at 1e-8) |
| Acceptance: holdout residual nonincreasing as Q grows | DONE | test "T3.1" |
| Acceptance: range error converges with the share | DONE | test "T3 acceptance" (share 1e-1 … 1e-5, both sides) |
| Acceptance: deterministic seed | DONE | test "T1.4" (same `rng` ⇒ same record and state) |

`thin_search` is the A/B knob of `_certified_range`, so it now only acts on `svd_select` (and the
`no_alloc`/exact path): the DD controller is always thin.

### Confirmation results

* hq13 17418199 (`22aca99`): test_dd_cbe.jl failed only on T3.5 `leak[2] > leak[1]`. At tol 1e-8
  the frames hold H² ψ as well, so both terms sat at the rounding floor (3.4e-16, 5.1e-17, then
  m = 3 at 1.9e-11). Fixed in `c570c3c`: the dominant leak is a higher-order term and exceeds 100× the
  first-order one.
* hq13a 17418200 (`22aca99`), the five adaptive files:
  * test_adaptive_search.jl "thin < wide probes" failed because `thin_search` no longer reaches the
    DD controller. Fixed in `c570c3c`: the A/B now runs on `cbe_policy = :svd_select`.
  * test_adaptive_rsvd.jl "a tighter basis tolerance never buys less rank" failed on the final max
    bond, 14 at tol 1e-8 against 15 at 1e-3.
* hqrank2 17419263 (`benchmarks/hq_rank_monotone_l12.jl`, XX L = 12, 4 × dt 0.05, exact-CBE
  reference with bonds up to 13): the frames are monotone in `tol_basis`, with DD D_stage 18 at
  1e-3 and 37 at 1e-8. The extra final rank at 1e-3 is the looser step's error kept above
  `tol_trunc` (reading (a)), not a controller defect. `092684e` grades the step's frame
  (`record.D_stage`) per step. MaxRSS 2.0 G.
* hq13b 17419262 (`c570c3c`): **test_dd_cbe.jl 202 pass / 1 broken** (the svd parity, as in
  Tasks 1–2), test_adaptive_search.jl 38 / 38, MaxRSS 1.7 G. T3.5 rows:
  * tol 1e-4: dim 57, leak 2.8e-16, 3.5e-7, 1.6e-6;
  * tol 1e-8: dim 69, leak 3.4e-16, 5.1e-17, 1.9e-11.
* hq13c 17419418 (`092684e`): **test_adaptive_rsvd.jl 177 / 177**, MaxRSS 1.7 G.

**Task 3 CONFIRMED.**

## Task 4

Branch `plan/hq-task-4` (stacked on `plan/hq-task-3`), commit `b04e5da`. Most of Task 4 was
already in the adaptive Krylov stage (`_krylov_enrich`, cbe_bug.jl) and the closure
(krylov_closure.jl). The gap was the T2.5 deferral: the `order > 1` frontier sweeps still ran
`cbe_expand`.

| item | status | evidence |
|---|---|---|
| T4.1 DD directions reused before Krylov, both half-sweeps | DONE | `half_left!` / `half_right!` seed `_krylov_enrich` with `ex.U_ex` / `ex.V_ex` (old frame plus DD directions, kept verbatim, charge labels from `oplus`); test "T4.1: Krylov after DD-CBE" (DD directions verbatim, a recycled round adds 0, both sides) |
| T4.1 frontier sweeps under the step's policy | DONE (`b04e5da`) | `_frontier_expand` dispatches on `ckw.cbe_policy`; `record.closure_policy = policy`; tests "T4.1: frontier CBE runs the step's policy", "order-3 steps == exact CBE" |
| T4.1 no direction regenerated | DONE | `_krylov_enrich` scores `(I − QQ†) v_k` against the grown frame; UT07 (CBE holds `H1 K_ex`: 0 added) |
| T4.2 transport basis and active frontier to the next stage | DONE | the frontier sweep embeds `fr.W[i]` by the overlap `Pm` (not `[I 0]`); frontiers `j` rebuilt from the root generator in the current frames; UT1, UT5 (old subspace nested at every bond, ψ and the frontier carried losslessly) |
| T4.2 propagated seed tested by its new response, not appended | DONE | Arnoldi vectors of the seed are propagated unprojected, only their residual `(I − QQ†) v` is appended; UT08 (`η_1 = 0`, `e3` still found) |
| T4.3 termination by a time-weighted residual, not a Ritz value | DONE | `_kry_weight`: `|τ β_k [φ_1(τ H_k)]_{k,1}|`, Saad's estimate, the residual integrated over `[0, τ]`; no eigenvalue is read |
| T4.4 operator images invalidated on a gauge or frame change | DONE | `H1` is rebuilt from the current environments at every bond visit (`OneSiteOp`/`TwoSiteSketchOp` carry `version`); a frontier from an older frame generation is refused (UT5) |
| T4.5 share `tol_basis` with CBE, work/rank stop | DONE | `alloc.krylov` is a share of the same allocation (T3.2); `workcap` bounds the frame; `capped` → `:work_limited` |
| T4.5 basis-only sweeps do not advance ψ | DONE | the carry is `W' K`, not `W' K_ex` (cbe_bug.jl, "carry the ORIGINAL amplitude"); the root starts from `S0 = AL·AR` of ψ (T1.3) |
| Acceptance: synthetic 3-level delayed frontier | DONE | UT08 |
| Acceptance: duplicate CBE/Krylov vector added once | DONE | UT07; test "T4.1: Krylov after DD-CBE" |
| Acceptance: left/right gauge-transport invariance | DONE | UT5; UT10 (old subspace and transport on both half-sweeps) |
| Acceptance: Krylov counts reduced with recycling | PARTIAL | a recycled round appends nothing (T4.1 test), but `n_h1` does not drop: the depth is set by the propagator weight, which does not depend on `Q` |
| Acceptance: stale operator images invalidated | DONE | UT5 stale-frontier refusal |
| Acceptance: no full-frame `H1 Q` closure scan | DONE | only the seed's Arnoldi vectors are applied; the frame `Q` is never multiplied by `H1` |

The frontier `dd_cbe_expand` calls are NOT counted in `record.counters`: the closure has no
per-half-sweep `KernelCounters`, and one shared across the concurrent frontier sweeps would race.
Their probes are in the bond ledgers.

Each row above was checked against the code before the confirmation.

### Confirmation (`b04e5da`)

* hq14a 17419908: **test_krylov_closure.jl 313 / 313** (14 min), test_adaptive_krylov.jl 72 / 72,
  test_adaptive_search.jl 38 / 38. MaxRSS 2.1 G.
* hq14b 17419909: test_adaptive_parallel.jl, test_adaptive_compression.jl and
  test_adaptive_expv.jl all pass.
* hq14 17419907 ran out of memory at 2 G. The rerun, hq14r 17420173 at 3 G, gave
  **test_dd_cbe.jl 227 pass / 1 broken** (the svd parity, as in Tasks 1–3), 23 min, MaxRSS 1.8 G.

**Task 4 CONFIRMED.**

Out of scope, parked by the user on 2026-10-22: hq14b's test_liouville.jl gave 91 / 92. The
failure is OS-U3 positivity (L = 6 Lindblad, order 2, `hermitian = false`, 4 compressed steps):
`min_eig = -1.76e-4` against the test's `-1e-4`. Every run on 2026-10-18 (default
`:svd_select`) gave `-2.9e-8`. This is the first run of the file since T1–2 made
`:dd_qr_streamed` the default. hq14b was cancelled before test_imaginary_time.jl.

## Task 5

Branch `plan/hq-task-5` (stacked on `plan/hq-task-4`), tests `dc5275a`. No source change was
needed: the root, the closure and the compression already met the plan. The gaps were in the
tests:
* p = 4 and 5 had never run;
* UT3 accepted `errs[3] < 1e-12` in place of a slope;
* there was no exact root-matrix comparison, no root ablation and no global order test.

| item | status | evidence |
|---|---|---|
| T5.1 stage maps are the Krylov closure, not predictor stages | DONE | order p closes the frontiers `F_j = (τ^j/j!) B^j S0`, j = 1..p−1 (`_projected_frontiers`), each with one enriching frontier sweep (`_single_pass_closure!`), in the frames the previous sweep grew; there is no `Ỹ_j → Ỹ_{j+1}` predictor |
| T5.1 root starts from the original state | DONE | `S0 = AL·AR` of ψ (cbe_bug.jl:1405); for `order > 1` `cl.S0`, re-projected `_project_root(psi, fr)` after every sweep that added (krylov_closure.jl:980, :1043) |
| T5.1 test exactly 1..5, order measured | TEST | UT3 (test_krylov_closure.jl): `_dw_errors(p, (0.4, 0.2, 0.1))`, `_finest_order >= p + 0.6`, for p = 1..5; an order the floors hide fails |
| T5.2 root terminated by a norm-scaled residual from `tol_root` | DONE | `tol_root → tol_expv → budget.expv → expv(conv_tol)`; `report.err` is Saad's estimate relative to ‖x‖ |
| T5.2 estimator vs the exact small root-matrix exponential | TEST | test_hq_root.jl T5.2: `U'HU` on the root space of a real step's frames (`_root_basis`, m ≤ 10); `est ≤ tol_root`, `err ≤ 10 tol_root`, `err ≤ 10 est`, the root action count grows with the tolerance |
| T5.2 root action count in the benchmark | DEFERRED to T8 | `record.root_iters` is recorded per step |
| T5.3 root ablation q < p, q = p, exponential | TEST | T5.3 at p = 3: trapezoid (q = 2) is capped at local order 3; radau2, radau3 and exp reach 4 |
| T5.4 lossless root split | DONE / TEST | `root_cutoff = 0`; T5.4, p = 1..5 with `truncate = false`: discarded < 1e-14, `proj_err < 1e-12`, `n_compress == 0`, norm kept to 1e-10 |
| T5.4 per-bond `w_i` and cumulative diagnostic, max ≠ cumulative | DONE | `bond_log[].discarded` per bond, `ledger.discarded` cumulative; UT8 rebuilds the cumulative figure from the per-bond weights and asserts it exceeds the worst bond |
| T5.5 global order at fixed T, valid range only | TEST / PARTIAL | T5.5: T = 0.8, p = 1..5; `_observed_orders` reads `log2(e(h)/e(h/2))` only where both errors are 10× above their floor (`tol_basis` + `expv_err` + discarded + `proj_err`), else `nothing`, which fails. **First-step dominated at L = 10** (hq15a): the fixed-T errors equal T5.3's one-step errors to 9 digits, so steps 2..n are exact (the frames fill the reachable sector after one step) and p_obs = p + 1. It bounds the fixed-T error but does not exercise accumulation; that is P08 (Task 8, L = 16, analytic reference) |
| Acceptance: original-state projection identity | TEST | T5.2 `‖v0 − UU'v0‖ < 1e-12`; T5.4 `proj_err` |
| Acceptance: root-operator Hermiticity, equivalence to dense | TEST | T5.2 at `tol_root = 1e-4` |

### Confirmation (`dc5275a`)

* hq15b 17421133: **test_krylov_closure.jl 314 / 314** (UT3 now p = 1..5 by slope), test_dd_cbe.jl
  227 pass / 1 broken (the svd parity, as before). MaxRSS 2.2 G.
* hq15a 17421132: **test_hq_root.jl 46 / 46**, 86 min, MaxRSS 2.4 G.
  * T5.3, local errors at h = 0.4, 0.2, 0.1, p = 3: trapezoid slope 2.99 (capped at q + 1 = 3);
    radau2 4.00, radau3 4.00, exp 4.00.
  * T5.5, observed orders at the two finest pairs: p = 1: 2.07, 2.02; p = 2: 3.00, 3.00;
    p = 3: 3.99, 4.00; p = 4: 4.99, 5.00; p = 5: 5.99, 6.00. All floors are below 1e-11. These are
    p + 1, not p: see the T5.5 row.

**Task 5 CONFIRMED**, with T5.5's accumulation test carried to Task 8 (P08).

## Task 6

Branch `plan/hq-task-6` (stacked on `plan/hq-task-5`): integrator and tests `7c6195a`;
`parallel = :full` and the Schmidt-frame kernel tests `abd2f02`; the T6.3 fixture `bb26a0c`.
Kernel `sweeps/parallel_kls.jl`; entry `parallel_kls_step!` = `cbe_bug_step!(...; kls = schedule)`,
or `cbe_bug_step!(...; parallel = :full)`.

The K/L stages start right after the preliminary half-sweeps (and, for `order = p > 1`, the frontier
closure), before the root solve, and are merged after it. S(h) IS the standard root solve. With
`kls = nothing` (the default) no line of the standard step's arithmetic changes.

| item | status | evidence |
|---|---|---|
| T6.1 separate implementation, shared kernels | DONE | `_kls_start` / `_kls_finish`: `one_site_h`, `push_*_channels`, `_preselect_*`, `_stream_left`, `zero_site_h`, the step's root `expv` settings, the one compression; the default path is the standard step |
| T6.1 p preliminary half-sweeps kept; K/L time-integrating | DONE | the K/L stages run in the frames the half-sweeps and the closure built; each is `expv` of `tau` at `tol_root`, not a basis-only sweep |
| T6.2 all three initial values from Y_n in the same frames | TEST | T6.2: `K0`, `L0` dense-equal ψ; `W[c]'K0 = S0 = L0 Z[c]'`; ψ untouched |
| T6.2 shapes, complex adjoints, charges | TEST | K/L operators Hermitian on random complex tensors (1e-11); leg dimensions; block-sparse U(1) legs carry the charges |
| T6.2 stage order >= p | DONE / TEST | converged exponentials at `tol_root`: K(h), L(h) equal the dense `exp(τ Q'HQ)` on their one-site spaces to 1e-10 |
| T6.3 omitted block measured on every step | DONE / TEST | `M11 = τ U_+' H Θ_n V_+` streamed against the thin `V_+` (no `H Θ`); `missing = ‖M11‖/‖ψ‖` in `account`, compared with `tol_missing = budget.basis` at the call site (cbe_bug.jl). T6.3, on the Schmidt frames of a truncated XXZ (Δ = 0.5) state at h = 1e-3: the exact new-new block minus `M11` falls as h^2 |
| T6.3 never silently accept the zero block | DONE / TEST | `missing > tol_basis` → the augmented root Galerkin step in `[W[c],U_+] ⊗ [Z[c];V_+]`, `account.fallback`; T6.3 checks it against the dense augmented Galerkin to 1e-10; four-block core has no new-new component (< 1e-12) |
| T6.3 observed order | TEST | "local order of the K/L/S recurrence" p = 1..5, `_finest_order >= p + 0.6` |
| T6.4 the two-spin DD test | TEST | Z2: unenriched frames miss the O(h) amplitude (< 1e-14); the full step gives `cos(Jh)`, `sin(Jh)` to 1e-12 with no new K/L directions; U(1) analogue against the dense exponential to 1e-12 |
| T6.5 three schedules, one state | TEST | the step: `:sequential`, `:kl_then_s`, `:concurrent` agree to 1e-10, `:concurrent` reproducible to 1e-12, `parallel = :full` (concurrent half-sweeps) agrees to 1e-10; the kernel on Schmidt frames, where U_+ and V_+ exist on both sides: the three states agree to 1e-10 |
| T6.5 standard vs K/L/S at matched physical error | DEFERRED to T8 (P12) | both are `cbe_bug_step!` keywords on the same model |
| T6.6 immutable frames, ψ not mutated before the merge | DONE | `_kls_start` writes nothing; `W[c]`/`Z[c]` replaced only after the root solve; ψ is written by the root split |
| T6.6 separate environments, scratch, counters per worker | DONE | `HK`/`HL` own new environments (`push_*_channels`), `KernelCounters`, report `Ref`s; `KRYLOV_LOG` appends under `_KRYLOV_LOCK` |
| T6.6 per-worker RNG streams | N/A | the K/L/S stages are deterministic `expv` and draw no random numbers |
| T6.6 no BLAS oversubscription | PARTIAL | not pinned inside the library (as for the half-sweeps, cbe_bug.jl "IT COMPETES WITH BLAS"); jobs run `-t 2` on 2 CPUs; P11 must record the BLAS/thread counts |
| T6.6 memory overhead of three solvers | MEASURED (job) / P11 | MaxRSS below; per-stage allocation is not separable under concurrency (`Base.gc_bytes` is process-wide) |
| critical-path model, measured speedup | DEFERRED to T8 (P11) | `ledger.kls.t_K`, `t_L`, the root counters' `t_root` are recorded per step |

**What the step does at the test sizes (hq16a).** At L ≤ 10 the real preliminary frames already
contain K(h) and L(h) to numerical rank, because the half-sweeps' local Krylov runs on the same
one-site spaces and fills them. So `n_new_l = n_new_r = 0`, the four-block core is `S(h)`, and the
K/L/S step equals the standard step: T6.5's ledger shows 0 new directions, and the K/L/S local
errors at p = 3 and 5 equal the standard ones to every digit. The local-order tests therefore
show the order of the recurrence but do not tell it apart from standard BUG at these sizes.
The kernel tests (T6.3, the kernel half of T6.5) run on the state's own Schmidt frames, which are
strictly smaller than the one-site spaces, so U_+ and V_+ exist there. The comparison that can
tell the two methods apart is P12 at L = 16 (Task 8).

**T6.3's fixture.** The guard `rems[1] < ‖dM‖` (the leading term dominates the remainder) is
ours, not the plan's. The plan asks for the block to be measured and for the fallback; both are
tested unchanged. The fixture was changed twice:
* On the XX Néel Schmidt frames (hq17d 17429012) the exact new-new block has no O(h) term:
  `‖dM‖ = 5e-14`, while the h^2 slopes passed. The kernel is correct; the fixture is degenerate.
* Every truncated XXZ fixture has a real O(h) block, but at h = 0.2 the remainder is 35–1600
  times the leading term, and the ratio is ∝ h (hq19b 17429851).
* By the user's decision (2026-10-22), T6.3 now runs at h = 1e-3 (`bb26a0c`), assertions unchanged.

### Confirmation

* hq16b 17422382 (`7c6195a`): test_expv 25, test_adaptive_parallel 32, test_dd_cbe 227 pass /
  1 broken (the svd parity, as before). MaxRSS 1.8 G.
* hq17c 17427751 (`abd2f02`): test_dd_cbe 228 pass / 1 broken, at 3 G.
* hq17d 17429012 (`abd2f02`): test_parallel_kls 41 / 1 failed, the T6.3 guard on the XX fixture
  above.
* hq20a 17430527 (`bb26a0c`): **test_parallel_kls.jl 42 / 42**, 63 min. MaxRSS 2.3 G at 4 G,
  no `oom_kill`. T6.3: `missing = 1.41e-6`; the remainders 2.50e-7, 6.24e-8, 1.56e-8, 3.90e-9
  fall at slopes 2.000, 2.000, 2.000; remainder / leading term = 0.18.

**Task 6 CONFIRMED**, with the speed-up and the matched-error comparison carried to Task 8 (P11, P12).

## Task 7

Branch `plan/hq-task-7` (stacked on `plan/hq-task-6`):
* the models and references `4b77177`;
* the generalized Jordan–Wigner map `657a399`, `16c400b`;
* the current's sign `56b5dce`;
* the reference solver's first-step split `5fa2a85`;
* the checkpointed reference run and common-horizon band `e84f50f`.

Models `models/spin_xx.jl`: `spin_space`, `spin_xx_mpo`, `spin_current_mpo`,
`spin_domain_wall_state`, `spin_sz_profile`. Tests `test_spin_xx.jl` (T7.0–T7.3) and
`test_spin_jw.jl` (U31). L = 16 reference `benchmarks/hq_xx_reference.jl` with
`hq_xx_reference_band.jl`.

| item | status | evidence |
|---|---|---|
| (XX), (DW): one exchange convention, OBC, d = 2s + 1 | DONE / TEST | `spin_xx_mpo(L, twoS)` = (J/2) Σ (S⁺S⁻ + S⁻S⁺) on bonds 1..L−1; T7.0: d = 2, 7 sectors, s = 1/2 equals `xxz_mpo(delta = 0)` on an evolved state (1e-12), `<H> = 0` and the exact product-state variance |
| T7.1 closed-form open-chain propagator | TEST | `xx_open_propagator` equals `exp(-i t h)` (1e-12) at t = 0.5, 3, 8; `<S^z_i> = C_ii − 1/2`, `tr C = 8`, `tr hC = 0`, continuity of the centre current by central difference |
| T7.1 converged solver against it, L = 16 to Jt = 8 | TEST | order-4 TDVP2, dt = 0.025, D ≤ 512, `trunc 1e-14`: err_m, err_j < 1e-6; norm, E, Var(H) − 1/4, S^z_tot, sector on every record |
| T7.2 / U31 generalized JW map | TEST | Batista–Ortiz map, 2s flavours, any vacuum: `T'T = I`; `H_JW T = T H_s`, S⁺, S^z, the centre current, S^z_tot commute (1e-12) |
| T7.2 / U31 "is the mapped H quadratic?" | TEST: NO for s = 3 | least-squares fit over every quadratic fermion operator: s = 1/2 residual < 1e-10 (the control); s = 3 residual 0.84–1.0 on the full range and on the domain wall's Krylov space, every vacuum (hq26a 138/138). Hence no analytical s = 3 curve (plan line 454) |
| T7.2 (a) symmetry and short-time identities | TEST | T7.3 below; reflection × spin flip `m_{L/2+1} = −m_{L/2}`; S^z_tot = 0 |
| T7.2 (b) sector ED, L = 4, 6 | TEST | `spin_ed.jl` (shares nothing with the package): profile and centre current < 1e-6 to Jt = 1; the ED is unitary and its current obeys continuity, which fixes the current's sign |
| T7.2 (c) L = 16 from ≥ 2 methods/settings | BENCHMARK | hq28a–e: D_max 512/256, dt 0.025/0.0125, trunc 1e-10/1e-8, order 4/2 TDVP2; band = largest deviation from the finest |
| T7.3 t² coefficients 1/4 and 9 | TEST | Richardson in t = 0.01, 0.005 on the MPS (L = 16) and the ED (L = 4): relative error < 1e-5 |
| T7.3 constancy of `<H>`, Var(H), norm, S^z_tot | TEST | on every T7.1/T7.2 record; Var(H) needs the reference's first-step split (below) |
| T7.4 frozen conventions | DONE | header of `hq_xx_reference.jl`; the run's `config` carries twoS, L, J, dt, order, maxdim, trunc, ramp, solver |
| T7.4 converged in D_max, dt, truncation, solver; band | BENCHMARK | `hq_xx_reference_band.jl` refuses fewer than two varied axes; it covers the times every setting reached and prints that horizon |
| Acceptance: no spin-3 free-fermion curve is claimed | DONE | `hq_xx_reference.jl` labels the s = 3 reference numerical; U31 asserts the s = 3 fit fails |

**The current's sign (`56b5dce`).** `pair_mpo` applies the transpose of each vertex matrix. The
current's imaginary antisymmetric vertex therefore came out as −j. It now enters with the
opposite sign, and T7.2's ED-continuity check and the JW map both pin it.

**The reference solver's first step (`5fa2a85`).** From the domain wall, the first projected
TDVP2 step leaves Var(H) off by ∝ dt², and no later step removes it:
* s = 3, L = 16: Var(H) − s² = 2.18e-3, 5.44e-4, 1.36e-4 at dt = 0.025, 0.0125, 0.00625 (hq22a/b/c), flat afterwards;
* T7.3 asks for Var(H) constancy, so this is a defect of the reference, not a loose threshold.

`tests/common/ramped_tdvp.jl` replaces step 1 by dt/2^k, dt/2^k, dt/2^(k−1), …, dt/2. hq26b,
at t = 0.2:

| k | s = 3, L = 6 vs ED: Var(H) − s² | err_j | s = 1/2, L = 16 vs closed form |
|---|---|---|---|
| 0 | 3.5e-4 | 5.5e-5 | 1.7e-6 |
| 4 | 1.4e-6 | 2.1e-7 | 6.6e-9 |
| 8 | 5.3e-9 | 8.4e-10 | 2.6e-11 |
| 12 | 1.9e-11 | 2.9e-12 | ~1e-13 |

The error falls as 4^-k. Tests and the reference use k = 12.

### Confirmation

* hq26a 17432427 (`56b5dce`): **test_spin_jw.jl 138 / 138** (U31: s = 1/2 control quadratic,
  s = 3 not quadratic on every vacuum). MaxRSS 4.1 G.
* hq27a 17433092 (`5fa2a85`): **test_spin_xx.jl 65 / 65** (T7.0–T7.3 with the ramped reference),
  380 min. MaxRSS 1.85 G at 4 G, no `oom_kill`.
* T7.2 (c) / T7.4, the L = 16 s = 3 band: the ladder hq28a–e (`e84f50f`) agreed only to t ≈ 0.9
  before the D = 256 members bound (t = 1.1–1.2); it was cancelled with the rest of the s = 3 work
  by the user's decision of 2026-10-22 (Task 8 runs s = 1/2 only), and no band was computed.

**Task 7 CONFIRMED** for its tests; the s = 3 L = 16 band is not delivered (s = 3 out of scope).

## Task 8

Branch `plan/hq-task-8` (stacked on `plan/hq-task-7`). One setting per run:
`benchmarks/hq_bench.jl` (L = 16 domain wall, per-step telemetry and records, `HQB_TWIN = 1` runs
the standard step next to K/L/S, `HQB_STATE_TIMES` grades the state, and the twin, against sector
ED). Figures: `benchmarks/hq_task8_figures.py RUNDIR[,RUNDIR...] OUTDIR [S3REF] [--figs=...]`.
P06: `benchmarks/hq_dd_scaling.jl`. Commits `82fb919`, `eac2083`, `344877f`, `3a78fa1`, `e901769`
(`krylov_reuse` knob, default on), `5563947` (`_merge_frame` guard), `ead126f`, `4dae685`,
`1ce5e48`, `bd4f261`, `808bfa1`, `b33f65b`. Figures and their run JSONs: `figures/hq_task8/`.

**Scope: spin 1/2 only (user, 2026-10-22).** The plan runs every real-time plot to Jt = 8 for both
spins. At s = 3 neither side reaches it: the reference ladder (hq28a–e) agreed only to t ≈ 0.9
(the D = 256 members bind at t = 1.1–1.2), and the MPS bond reaches ~780 at t = 0.4. By the user's
decision Task 8 runs s = 1/2 only for now; the s = 3 jobs (hq28 ladder, hq41 campaign, the s = 3
P11 stage, the band) were cancelled. The s = 3 K/L/S scan already measured (hq31a, hq37) is kept
below as a finding; its JSONs are in `figures/hq_task8/runs/*_s3`, outside the figures.

| item | status | evidence |
|---|---|---|
| P01 dynamics | DONE (s = 1/2) | `P01_dynamics_s1.png` |
| P02 observable errors by tol_basis, order | DONE (s = 1/2) | `P02_error_s1.png` |
| P03 norm / E / S^z_tot / Var(H) | DONE (s = 1/2) | `P03_conservation.png`; E(0) = 0, absolute drift |
| P04 bond stages, per-bond heatmap, three tol_basis | DONE (s = 1/2) | `P04_bonds_s1.png` |
| P05 DD-QR vs one-sided vs svd_select | DONE (s = 1/2) | hq33: p = 2 all equal; p = 4, h = 0.1, tb 1e-5: DD-QR 1.2e-7, one-sided 2.9e-9, svd_select 3.3e-6 (L2 error at t = 8) |
| P06 DD-QR kernel scaling vs D | DONE (s = 1/2, D 32–256, hq43t) | `P06_dd_scaling.png`; every kernel's local exponent γ is −0.1 … 0.9 at fixed w: no hidden cubic work (overhead-dominated at these D). The s = 3 run (hq32a, D 32–512) is superseded |
| P07 Krylov reuse on/off at matched error | DONE | hq36b vs hq33, p = 1: reuse saves 16–19 % Krylov directions (378/448 at h = 0.2, 412/489 at h = 0.4), same H1 count, err at t = 8 equal to 3e-9 |
| P08 order 1–5, floors marked | DONE: order p NOT observed | below |
| P09 tol_basis × h grids | DONE (s = 1/2) | `P09_controls_s1_p*.png` |
| P10 time to accuracy | DONE | below |
| P11 three schedules × workers 1/2/3 | DONE: no speedup at this size | below |
| P12 parallel vs standard, separate reference errors | DONE | below; at matched error K/L/S is 10–15 % slower |
| P13 failure modes, seeds | PARTIAL | hq33 seeds 1–3; limits and unmet steps logged per run |

**P08: the state error is set by the first step.** At L = 16, T = 8, tol_basis 1e-7 (hq38), `e_p`
falls as h^(p+1), not h^p: fitted slopes 2.33, 3.00, 3.99, 4.97, 5.96 for p = 1..5 (p = 1 is 2.0
below h = 0.4; the observable slopes are the same). The error does not grow after the first step:
at h = 0.4 it is 4.194e-6 at both t = 0.4 and 0.8 (p = 4), and 6.353e-7 at t = 0.4, 4 and 8 (p = 5).
Mechanism, from the step-1 ledger: from the D = 1 domain wall each of the p Krylov rounds reaches
one site further, and `ledger.leak` per round (0.08, 9.8e-3, 1.07e-3 at p = 4, h = 0.4) is the
exact amplitude (Jt)^d/d! at distance d = 2, 3, 4. The amplitude at distance p + 1, ~h^(p+1)/(p+1)!,
is left out and carried unitarily; step 2 starts from the wider frames, reaches 2p sites and adds
< 1e-9. `tolerance_met` holds on every step: it covers frontiers 0..p−1, and the remainder is the
order-p local error, so it scales with h and p as it must. It is the same pattern as T5.5 at L = 10.
Below ~2e-8 (growing with the step count) the tol_basis floor takes over; those points are drawn
open and not fitted (`808bfa1`).

**Timing (hq43t, one job, one node, 3 CPUs, BLAS pinned to 1 thread).** Step 1 of every process
carries the JIT compilation (145–165 s against < 1 s of physics at D = 1), so integrator time is
the sum of `t_step` over steps 2..n (`3c32c39`).

**P10: higher order wins on time to accuracy.** Standard serial step, one worker, s = 1/2,
tb 1e-7, Jt = 8 (`P10_time_to_accuracy.png`). Integrator time to reach max|Δm(8)| ≲ 1e-6:

| p | cheapest run at ≲ 1e-6 | time |
|---|---|---|
| 1 | not reached (6.4e-5 at h = 0.05, 68 s) | — |
| 2 | h = 0.05, 9.2e-7 | 85.5 s |
| 3 | h = 0.1, 1.0e-7 | 54.2 s |
| 4 | h = 0.4, 1.2e-6 (h = 0.2: 3.6e-8, 37 s) | 24.3 s |
| 5 | h = 0.4, 3.8e-8 | 24.7 s |

p = 4–5 reach it 3.5× faster than p = 2. Operator applications follow the same ordering; peak RSS
is 2.0–2.4 G for every run; |Δ‖ψ‖²| and |ΔE| at t_f stay below 3e-12 at every h.

**P12 speed at matched error.** At h = 0.1 the K/L/S state has the standard step's error (p = 1, 2,
3, 5 identical; p = 4 equal to 6 %), so the speeds compare at matched error. The fastest K/L/S run
of each order (always one worker) against the standard serial step: 0.88, 0.90, 0.89, 0.90, 0.85
for p = 1..5. K/L/S does the standard step's work plus the K and L solves and the merge, and at
this size it gains nothing back from the extra workers (P11).

**P11: more workers are slower.** s = 1/2, tb 1e-7, h = 0.1, Jt = 8, `parallel = :full`
(`P11_schedules.png`). Integrator time at 1 / 2 / 3 workers, range over the three schedules:

| p | 1 | 2 | 3 | speedup at 3 |
|---|---|---|---|---|
| 1 | 44–46 s | 52–53 s | 59–62 s | 0.74–0.77 |
| 2 | 53 s | 64–66 s | 76–79 s | 0.68–0.70 |
| 3 | 61–64 s | 75–77 s | 88–91 s | 0.70–0.72 |
| 4 | 69–71 s | 86–88 s | 101–106 s | 0.66–0.69 |
| 5 | 78–80 s | 98–99 s | 117–120 s | 0.65–0.67 |

The three schedules agree to ~4 % at every worker count, including `:sequential`, which runs no
K/L/S stage concurrently: the slowdown is the threaded rest of the step (the `:full` half-sweeps),
not K/L/S. The K, L and root solves are ~8 s of a 45–80 s step, so even perfect three-way overlap
bounds the gain at ~10 %. At s = 1/2, L = 16 the bond stays below 200 and each thread gets too
little work; the `small` partition allows at most 4 CPUs. Peak RSS 2.3–2.8 G throughout.

**P12: K/L/S against the exact state and against the standard step.**
* s = 1/2 (hq39, tb 1e-7, t = 0.8): K/L/S equals the standard step at p = 1, 2, 3, 5. At p = 4,
  the only order where it finds new directions, it is 3.5–5× closer to the exact state (4.19e-6
  vs 1.90e-5 at h = 0.4). Both arms have slope p + 1. The K/L/S norm excess (3.5e-10 at p = 4,
  h = 0.4) is real amplitude toward the exact state, not error.
* s = 3 (hq31a, hq37, tb 1e-5, t = 0.4): the distance to the standard step falls as h^(p+1)
  (slopes 3.5, 3.9, 4.8, 5.7 for p = 2..5) and the norm excess as its square. The excess equals
  ‖U₊'K‖² + ‖LV₊'‖², as the plan's four-block merge implies; it is created in step 1 and then
  flat. At h = 0.2 it is not usable for p ≤ 3 (p = 2: norm² − 1 = 0.05 and Var(H) − s² = +14.3
  on a conserved 9); at h ≤ 0.1 and p ≥ 3 it is below 1e-4.
* A measurement floor of 3–5e-9 in the state error (max|Δm| is 1e-14 there) is not fitted.
