# BUG-Julia — `cbe_bug`

A symmetry-native, rank-adaptive **Basis-Update-and-Galerkin** time integrator for 1D tensor
networks, built on Telum / LurCGT so that U(1), ℤ₂ and non-Abelian symmetric tensors stay
block-sparse throughout.

Rank grows only through **RSVD controlled bond expansion** — directions drawn through the
Hamiltonian itself. There is no random sector seeding and no padding anywhere.

## Install and run

Julia 1.10+, and `matplotlib` if you want the figures.

```bash
git clone https://github.com/madhavmen/BUG-Julia.git && cd BUG-Julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/oat_z2.jl          # a full benchmark, ~1 min
python3 examples/plot_examples.py             # figures for every CSV produced
```

`instantiate` pulls two dependencies that are **not in the General registry** and come straight
from GitHub — [`LurCGT.jl`](https://github.com/lurlurlurrrrr/LurCGT.jl) (symmetric-tensor
bookkeeping) and [`Telum.jl`](https://github.com/ssblee/Telum.jl) (the tensor layer). Both are
pinned by commit in the tracked `Manifest.toml`, so the versions are reproducible; if
`instantiate` fails, it is almost always because one of those two could not be reached, not
because of anything in this repository.

The first run compiles the package and takes a couple of minutes; later runs are fast.

That is the whole quick start. **[examples/](examples/)** is the place to begin — three
self-contained scripts, each one model, each writing a CSV and three figures, with every
parameter in an editable block at the top:

| script | model | what it exercises |
|---|---|---|
| [`oat_z2.jl`](examples/oat_z2.jl) | one-axis twisting, ℤ₂ | rank is **constant** in time — the calibration case |
| [`xx_domain_wall.jl`](examples/xx_domain_wall.jl) | XX chain, U(1) | rank **grows** — adaptivity under load |
| [`tfim_z2.jl`](examples/tfim_z2.jl) | transverse-field Ising, ℤ₂ | rank grows, and a ℤ₂-charged coupling |

Every one is checked against a **closed form or a free-fermion solution**, never a second
integrator and never a `2^L` diagonalisation — so the error column is an error, not a
disagreement, and stays available at sizes where exact diagonalisation is not.
See [examples/README.md](examples/README.md).

## The three integrators

All take an `MPS` and an `MPO` and advance it by `tau`, so any difference between them is the
integrator and nothing else.

```julia
using BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

set_symmetry!(:U1)                                # :none | :U1 | :SU2 | :Z2 — set it FIRST
psi = domain_wall_state(8)                        # |↑↑↑↑↓↓↓↓⟩, χ = 1
W   = xxz_mpo(8; J = 1.0, delta = 1.0)

for _ in 1:40
    info = cbe_bug_step!(psi, W, ComplexF64(-im * 0.05); tol_basis = 1e-5, maxdim = 32)
    budget_status(info.ledger) === :met || @warn budget_diagnostic(info.ledger)
end

bond_dims(psi)          # rank grew from a product state, as far as tol_trunc/maxdim allow
mpo_energy(psi, W)
```

| function | what it is |
|---|---|
| `cbe_bug_step!` | RSVD-CBE + rank-adaptive BUG. `K`/`L` half-sweeps build the bases, one Galerkin step at the centre root. |
| `tdvp_cbe1s_step!` | 1-site TDVP with controlled bond expansion ([arXiv:2208.10972](https://arxiv.org/abs/2208.10972)) — the baseline to beat. |
| `tdvp2_step!` | 2-site TDVP. The conventional reference. |

### Three tolerances and one resource limit

`cbe_bug_step!` and `cbe_bug_midpoint_step!` are driven by

| keyword | default | controls |
|---|---|---|
| `tol_basis` | `1e-5` | the trial space: RSVD sampling and retained rank (Tests A/B) and one-site Krylov enrichment (Tests C/D) |
| `tol_trunc` | `1e-7` | the ONE final compression: cumulative discarded norm of the evolved state |
| `tol_expv` | `1e-10` | the root Galerkin `expv` (convergence test and substep split) |
| `maxdim` | `128` | final bond dimension; also sets the internal working-rank ceiling |

Everything else (`dex`, `dover`, `comp_ratio`, `growth`, `krylov_basis`, `krylov_tol`,
`split_*`, `root_*`, `trunc_thresh`, `stol_*`, `maxiter`, `tol`) belongs to the **legacy
fixed-knob path**: naming any of them selects it (with a one-time warning saying so), and mixing
them with a tolerance is an error.
`info.ledger` (`nothing` on the legacy path) records every indicator, per step and, in
`info.ledger.bond_log`, per half-sweep bond visit (Tests A-D, probes, Krylov directions, frame
width, verdict); `budget_status(info.ledger)`
is `:met` or `:budget_unmet`, and `budget_diagnostic` says which check failed and what to change
(smaller `dt`, larger `maxdim`, looser `tol_trunc`) -- a rank cap is reported, never silently
accepted. `exact = true` replaces the random sketch by the complete local basis. Start states
are complex by default; `neel_state(L; eltype = Float64)` (likewise `domain_wall_state`,
`product_state`) gives a real one.

**The bond expansion is DD-projected, MPO-streamed and QR-only** on the adaptive path
(`cbe_policy = :dd_qr_streamed`, [cbe_bug_high_order_rsvd_parallel_tasks.md](cbe_bug_high_order_rsvd_parallel_tasks.md)
Task 2): the K-sweep grows only the left frame from `P_L^D X P_R^D Ω`, the L-sweep only the right
frame from the mirror, with the probe projected before the MPO contraction, the output projected
after, and a sectorwise Householder QR (numerical rank only) choosing the directions -- no
`H*Theta`, no projector and no SVD in the selection. `:one_sided_qr_streamed` (drops the
opposite-leg projector) and `:svd_select` (the original two-sided SVD selection, and what the
legacy path runs) are ablations. The `order > 1` frontier sweeps still use `:svd_select`
(`info.record.closure_policy`). `info.record` is the step's typed `StepRecord`: policy, QR
backend, kernel counters (streamed actions, one-site and root applications, QR ranks, new
sectors, seconds and bytes per kernel), `D_old`/`D_stage`/`D_final` and `limits`
(`:basis_limited`, `:root_limited`, `:truncation_limited`, `:work_limited`, `:rank_limited`).
`tol_root` is accepted as the plan's name for `tol_expv`, and `parallel = :half_sweeps` /
`:serial` for `true` / `false`.

**`order = p`** (default `1`) is the one method choice on top of the three tolerances. For
`p > 1` the step closes the global Krylov frontiers of the root generator before its single root
`expv`: each `F_j = (tau^j/j!) B^j S0`, `j = 1..p-1` (`B = P (tau H) P`), is embedded as an MPS and
swept with the same CBE / Tests C/D kernels until its outside-space action `(I - P) H F_j` fits
its share of `tol_basis`, so `B^j psi = (tau H)^j psi` for `j <= p` (local error `O(dt^(p+1))`).
`info.ledger.leak` (per frontier), `.defect` (integrated-defect estimate), `.n_passes` and
`.closure_log` report it; failing to close is `:budget_unmet` (`:closure`). The frontiers are
first MEASURED on the ordinary step's frames (`.leak_initial`, `.rho_initial`); when those
already close them no closure sweep runs (`n_passes = 0`). `closure_probe = m` records the same
numbers on an `order = 1` step without changing it. The order is a claim to be measured:
tolerances, `maxdim` and the final compression set an error floor
([krylov_closure_cbe_bug_realtime_L16_plan.md](krylov_closure_cbe_bug_realtime_L16_plan.md),
conformance in [KRYLOV_CLOSURE_AUDIT.md](KRYLOV_CLOSURE_AUDIT.md), XX L=16 results in
[KRYLOV_CLOSURE_L16_RESULTS.md](KRYLOV_CLOSURE_L16_RESULTS.md)).

What each tolerance buys, measured on XX L=20 against the exact solution, and starting
settings per target error: [ADAPTIVE_XX_L20_RESULTS.md](ADAPTIVE_XX_L20_RESULTS.md). In short:
`tol_basis` sets the accuracy (err ~ tol_basis), `tol_trunc` can be 10-100x looser and is the
speed knob, `tol_expv = 1e-2` suffices at `dt = 0.05` (use `1e-4` near 1e-5 on interacting
models), and the ordinary/midpoint choice sets the floor; re-checked on isotropic Heisenberg and
at `dt = 0.1`. Plan conformance: [ADAPTIVE_PLAN_AUDIT.md](ADAPTIVE_PLAN_AUDIT.md).

**Search width and large χ.** The RSVD probe starts at `ADAPTIVE_GROWTH = 1.2` times the bond's
budget and is widened only where Test B's holdout finds range missing (`thin_search = true`, the
default; `false` keeps the old doubling, which ran every bond up to its numerical rank before
Test B was asked). At large χ `tol_basis` is the speed knob: on the 10x4 Heisenberg cylinder at
χ = 256 (`benchmarks/spectral/bug_profile_2d.jl`, `tol_trunc = tol_expv = 1e-6`), `tol_basis`
1e-4 → 1e-3 cut the first-order step from 31 to 20 s at an infidelity of 5e-6 after t = 0.3.
Timing against TDVP is only quoted between arms of the same order at the same dt (bug2/bugmid
vs `tdvp2_step!`, bug4 vs `tdvp2_composed_step!(...; order = 4)`) or at matched physical error.
Measured that way (10x4, dt 0.1, 3 steps, 2 CPUs, job 17285246), order-2 BUG at `tol_basis =
1e-3` runs at 0.43x `tdvp2_step!`'s speed at χ = 256 (139 vs 60 s/step) and 0.95x at χ = 512
(227 vs 216 s/step), landing within 1e-7 infidelity of it; `tol_basis = 1e-4` costs 3x more.
`tdvp2_composed_step!(psi, h, tau; order = 4, composition = :suzuki)` composes the symmetric
`tdvp2_step!` (Suzuki 5 stages or Yoshida 3) and is of order 4 *in the step* only where the step
error dominates: on an L = 24 XXZ quench with the rank set by `trunc_thresh = 1e-10`
(`benchmarks/tdvp2_order_probe.jl`, `tests/sweeps/test_tdvp2_order.jl`), max|ΔS^z| falls with
slope 2.1 (order 2) and 5.6 (Suzuki order 4) from dt 0.2 to 0.1, while Yoshida needs a finer dt
(2.8, then 4.8). At a binding cap the truncation sets the error and composing buys nothing (L = 8
cap 6: order-4 errors no smaller than order 2; 6x4 D=512: `tdvp2_step!` slopes 0.95–0.99).
`fold_omega = true` folds the probe into the environment contractions instead of forming
`H Theta` (with `order > 1` the closure's leak estimate then uses the folded sketches and a Test-B
holdout); it pays only when the probe is much thinner than χ (3·width < χ), was slower at every
χ ≤ 256 measured, and is off by default.

**Ground states use the same engine.** `dmrg_cbe1s!(psi, mpo; tol_basis, tol_trunc, maxdim)`
(and `rsvd_cbe_dmrg1s!`) runs the one-site CBE-DMRG with the adaptive Test A/B selection and the
`maxdim`-derived work ceiling. With no time step, a bond's indicator is
`||(I - P) H Theta|| / ||psi||` in energy units: summed over bonds, that is the two-site variance
estimate. Each split keeps the smallest rank whose discarded norm fits `tol_trunc / sqrt(L - 1)`.
`r.status` and `r.ledgers` record the per-sweep verdict and the per-bond logs. When no
tolerance is named, the defaults are medium-loose (`DMRG_TOL_DEFAULTS`, `1e-3` / `1e-4`), not the
time step's tight ones. A sweep is repeated until the energy settles, so a direction one loose
sweep skips is offered again on the next. These defaults are provisional until the tolerance
study (`benchmarks/ground_state/adaptive_dmrg_tolerance.jl`) has measured them. The local
eigensolve keeps its Ritz-residual stop (`restol`). Naming `dex`, `growth`, `dover`, `comp_ratio`
or `trunc_thresh` selects the old fixed-knob DMRG unchanged. `mpo_variance(psi, mpo)` is the
exact `<H^2> - <H>^2`, from a two-layer MPO environment. 2D spectra (square, square `J2`, kagome,
triangular) with every integrator arm, the adaptive ground state and an exact reference up to
N = 20 are produced by `benchmarks/spectral/fig45_2209.jl`. Conformance:
[SIMULATION_PLAN_2_AUDIT.md](SIMULATION_PLAN_2_AUDIT.md).

### The legacy fixed-knob path

Its accuracy knob is the **depth of the Krylov basis** the half-sweeps build:

- **`krylov_basis`** (default `30`) — the *cap* on how many orthonormal vectors
  `span{Θ, HΘ, H²Θ, …}` the half-sweep builds per bond. `0` stops at one application of `H`, which
  is what the CBE frame alone gives you.
- **`krylov_tol`** (default `1e-6`) — stops the recursion when the next vector's *contribution*
  to `exp(τH)Θ` drops below it (Saad's estimate `β_m·|[exp(τT_m)]_{m,1}|`). `0.0` runs to the cap.

⛔ **`krylov_basis` is the knob to scan when an answer is less accurate than it should be**, and it
is the one that looks like nothing is wrong. Extra rank, extra expansion budget and switching the
closing truncation off entirely all leave the error *unmoved*, because none of them add a
**direction** — only depth does. Measured at L=12, `dt=0.05`:

| model | one power of `H` | depth 3 |
|---|---|---|
| XX | 1.6948e-04 | 1.6948e-04 — extra vectors are pure cost |
| Heisenberg | 8.8424e-05 | 2.6697e-06 |
| OAT | 2.1329e-02 | 7.0610e-14 — **eight orders** |

The two schemes agree to O(τ), so they are the same *order* and can look identical (XX: bit
identical). The difference is entirely in the prefactor, and it appears wherever `τ‖H_loc‖` is not
small.

The sweep is always **interleaved** (expand bond `i`, then immediately evolve site `i`) — the
ordering of the reference implementation.

Imaginary time is the same call with a real negative `tau`; rescale the state each step.
`GroundState.rsvd_cbe_dmrg!` is the variational ground-state mode.

## Symmetries

One call, before building any tensors:

| mode | models | charge | notes |
|---|---|---|---|
| `:none` | all | — | one dense block; the control every symmetric run is checked against |
| `:U1` | XXZ / Heisenberg / XX | `S^z` | the default |
| `:SU2` | Heisenberg (isotropic) | total spin | χ counts **multiplets**, not states |
| `:Z2` | transverse-field Ising, OAT | spin-flip parity | its own local space, gates, states and MPO |

ℤ₂ works in the **σˣ basis**, where the global flip `P = Π X_i` becomes a diagonal charge and the
tensors go block-sparse (in the usual σᶻ basis it is off-diagonal and buys nothing). The trade is
that **`X` is diagonal and `Z` is the flip** — the reverse of the usual convention — so `X` is the
measurable operator and `Z` carries the charge.

```julia
set_symmetry!(:Z2)                                # H = -J Σ Z_i Z_{i+1} - h Σ X_i
psi = ising_kink_state(8)                         # a domain wall in the X basis
W   = tfim_mpo(8; J = 1.0, h = 1.0)               # explicit MPO, virtual dimension 3
cbe_bug_step!(psi, W, ComplexF64(-im * 0.05); maxdim = 32)
x_profile(psi)                                    # ⟨X_j⟩; ⟨Z_j⟩ is identically zero by symmetry
```

## Hamiltonians

Every model is an `MPO`, so all of them drive the same environments, effective Hamiltonians and
Krylov solves — a run differs from another run only in the operator.

| builder | Hamiltonian | virtual dim |
|---|---|---|
| `xxz_mpo(L; J, delta)` | `J Σ (SxSx + SySy + delta·SzSz)`, `delta = 0` is XX | 5 (U(1)) |
| `heisenberg_su2_mpo(L; J)` | isotropic Heisenberg | 3 multiplets |
| `tfim_mpo(L; J, h)` | `-J Σ Z_i Z_{i+1} - h Σ X_i` | 3 |
| `oat_mpo(L)` | `(Σ S^z)² / 2` | — |
| `pair_mpo(L, J, vertices)` | **any** two-body `J[i,j]`, exactly, including charged operators | `O(L)` |
| `long_range_mpo` | exponentially decaying tails, one channel per decay | independent of `L` |

`pair_mpo` is what makes Haldane-Shastry exact: `1/sin²` has no finite-dimensional MPO, so `O(L)`
is what exactness costs rather than a wasteful encoding.

## Tests

```bash
julia --project=. tests/sweeps/runtests.jl        # the sweeps, the models, the physics
julia --project=. tests/runtests.jl               # BondUpdateBUG
```

Three **disjoint** runners — `tests/runtests.jl`, `tests/sweeps/runtests.jl` and
`tests/RSVDCBEBondUpdate/runtests.jl` — so running the wrong one proves nothing about the other
two.

`tests/sweeps/runtests.jl` is ordered so a red test localises instead of leaving suspects: the
analytic references are pinned against exact diagonalisation first, then the operators
(`pair_mpo`, `tfim_mpo`) with no integrator involved, then the sweeps, then cross-symmetry
parity, then the physics.

Two independent references, each validated before it is relied on:
`tests/common/dense_reference.jl` (shares the integrator's conventions) and
`tests/common/free_fermion.jl` (derived from the Hamiltonian on paper, no `2^L` object anywhere).

⛔ **Rank the schemes by error against a reference, never by energy drift.** They disagree
outright on these models: a scheme can conserve energy to machine precision and still be an order
of magnitude further from the exact solution.

## Documentation

📖 **[docs/USAGE.md](docs/USAGE.md)** — symmetry switching, the options that matter and why, how
to read the diagnostics, the Krylov-depth result (§6b), and the gotchas.

📁 **[examples/README.md](examples/README.md)** — the three example scripts, their parameters,
the CSV columns, and what the randomised sketch does and does not affect.

📊 **[benchmarks/imaginary_time/README.md](benchmarks/imaginary_time/README.md)** — the three-model
campaign: Heisenberg chain and cylinder on logarithmic imaginary-time grids, and a boundary-driven
open XX chain on a uniform one. Every arm scored against an analytic reference, and the caveats
that have to travel with the numbers (`matvec` is not a cost proxy for BUG; model 3 has two
regimes; no open model here has a symmetry arm).

## Retired paths

`exploratory/` holds the global-sweep discarded BUG and the pre-refactor ITensors tree. None of
it loads against this package's dependencies; it is a record, not a working state.
