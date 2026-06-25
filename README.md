# BUG (Bond-Update-and-Galerkin) Quantum Integrator

## Overview

The BUG module provides canonically validated, rank-adaptive time-integration
methods for quantum spin-chain systems. Two integrators are available, both
two-site and both rank-adaptive; they differ only in how the local bond update
grows the basis.

**`bug_two_site!`** — 2-Site-BUG method (faithful KLS)
- Applies 2-site-local layers via faithful-KLS with adaptive bond dimension.
- Each parity (odd/even) group is a commuting set of nearest-neighbour terms, so
  the group is an exact factor of the Trotter step.
- Recommended for systems where rank growth is a concern.

**`discarded_bug_step!`** — discarded-projector BUG variant
- Derived from the faithful CKL scheme; differs *only* in the local bond update.
  The discarded (orthogonal-complement) projector is applied to the K/L generator
  *before* the exponential (`project-before`), and the basis is grown by a plain
  direct sum `[U0 | Qk]` / `[V0 ; Ql]` — **no augmented overlap matrices** `M̂`/`N̂`.
- Advances against a Hamiltonian **MPO** by a symmetric forward+reverse sweep
  (Strang), so with the rank free to grow it reproduces `exp(-i dt H)` to high order.
- The project-before generator is non-Hermitian, so the K/L substep uses a
  non-Hermitian Krylov exponential (`KrylovKit.exponentiate`, `issymmetric=false`);
  the Hermitian S-step reuses the faithful Lanczos path.

## Core Concepts

### Faithful Simultaneous K+L+S Update

The KLS (Krylov-Lanczos-SVD) local bond update is the foundation of rank-adaptive
integration:
- **K-step:** Right-orthogonal basis extension (Krylov)
- **L-step:** Left-orthogonal basis extension (co-Krylov)
- **S-step:** Schmidt value truncation with error control
- **Augmentation:** Optional enrichment of trial spaces with residual directions

Implemented via `_faithful_kls_local_bond_candidate(...)` with tunable tolerances
and iteration limits. The substeps exponentiate the *projected* effective
Hamiltonian internally (Krylov / Lanczos) — no pre-formed gate is applied — so the
local step is the faithful KLS update, exact at full rank.

### Strang Splitting

The step uses the symmetric (Strang) odd/even composition

```
U_odd(dt/2) ∘ U_even(dt) ∘ U_odd(dt/2)
```

applied as a sweep of bond-local KLS updates over each parity group.

## Public API

### Quantum 2-Site Method

```julia
info = bug_two_site!(psi, gates; dt, maxdim, order=:strang, kwargs...)
```
- `psi::MPS` — the state (modified in-place)
- `gates::Vector{ITensor}` — per-bond two-site Hamiltonian terms `h_{b,b+1}`
- `dt::Float64` — timestep
- `maxdim::Int` — bond-dimension cap for truncation
- `order::Symbol` — `:lie` (1st) or `:strang` (2nd, default)
- Returns `BUGInfo` with convergence metrics

### Discarded-Projector Variant

```julia
info = discarded_bug_step!(psi, W; dt, order=:symmetric, maxdim=typemax(Int),
                           time_prefactor=ComplexF64(-im), kwargs...)
```
- `psi::TensorTrain` — the state (modified in-place)
- `W::TensorTrainOperator` — the Hamiltonian **MPO** (not per-bond gates)
- `dt::Number` — timestep
- `order::Symbol` — `:symmetric` (Strang forward+reverse, 2nd order, default) or
  `:forward` (single forward sweep, 1st order; diagnostics)
- `maxdim::Int` — bond-dimension cap (`typemax(Int)` = grow freely)
- `time_prefactor::ComplexF64` — `-im` for real time (default); pass `ComplexF64(1)`
  for imaginary time / parabolic PDEs
- Returns `BUGInfo` with the before/after bond dimensions and convergence metrics

The kernel `discarded_bug_local_update(bond_data, HW_env; dt, maxdim, ...)` mirrors
the call surface of `_faithful_kls_local_bond_candidate`, so the sweep can swap
kernels without any other change.

### KLS Kernel (Direct Access)

```julia
candidate = _faithful_kls_local_bond_candidate(
    bond_data;
    dt, s_dt, augment=true,
    aug_krylov_depth=1,
    lanczos_tol=1e-15,
    lanczos_maxiter=60,
    substep_method=:expv,
    HW_env_override=nothing,
)
```
For cases where direct bond-by-bond update control is needed (rare).

### Helpers

- `two_site_xx_parity_mpos(sites; J)` → reference `(W_odd, W_even, W_full)` MPOs
  for the nearest-neighbour XX Hamiltonian
- `two_site_xx_bond_gates(sites; J)` → per-bond two-site term generators for the same

## Tuning

### Rank Adaptation

- **Increase `maxdim`** for finer accuracy at higher cost


### Composition Error

- Lie is 1st order; Strang is 2nd order
- For quantum short-time dynamics (small dt), 2nd order often suffices

## References

- Ceruti, Kusch & Lubich (2022), *BIT* — rank-adaptive Basis-Update & Galerkin (arXiv:2304.05660)
- Lubich (1994) — From Quantum to Classical
- Hochbruck & Lubich (1997) — Exponential integrators for large systems
- This codebase uses 2-site local terms (not full-chain) with faithful KLS for efficiency

## Files

- `bug_init.jl` — MPS/MPO initialization utilities and `BUGInfo` bookkeeping
- `bug_kls.jl` — Faithful simultaneous K+L+S implementation
- `discarded_bug.jl` — Discarded-projector BUG variant (`discarded_bug_step!`):
  project-before K/L generators, direct-sum basis growth, MPO-driven Strang sweep
- `two_site_bug/two_site_bug.jl` — Two-site odd/even Strang sweep and XX helpers
