# BUG-Julia — `bond_update_bug`

A symmetry-native, rank-adaptive Basis-Update-and-Galerkin time integrator for
1D tensor networks, built on [Telum](https://github.com/) / LurCGT so that U(1)
and non-Abelian symmetric tensors stay block-sparse throughout.

**There is exactly one supported integrator: `bond_update_bug!`.** Everything
else that used to live here is under `exploratory/` or at the tag
`archive/pre-bond-update-legacy`.

## Quick start

```julia
using BUGJulia.BondUpdateBUG

psi  = domain_wall_state(6)                       # |up up up down down down>
gates = bond_gates(psi; J = 1.0, delta = 1.0)     # delta = 0 gives XX
info = bond_update_bug!(psi, gates; opts = BondUpdateOptions(
    dt = 0.01, n_steps = 5, order = :strang, maxdim = 8))

[sz_expectation(psi, j) for j in 1:6]             # the evolved profile
info.max_bond_dims                                 # rank growth, per step
```

Real time only: `dt` is real and the driver forms `tau = -im*dt` itself.

## The algorithm

One local bond update, mirroring the verified Python kernel
(`Alice/.../kls/discarded_candidate.py`), in six steps:

1. **K-step.** `H_K x = V0'·gate(x ⊗ V0)`, then project *before* the exponential:
   `G_K x = H_K x − U0(U0'·H_K x)`. That makes `G_K` **non-Hermitian**, so the K
   and L substeps take an Arnoldi exponential, not Lanczos.
2. **L-step.** The mirror, with `P⊥_V0` applied on the right.
3. **Augmentation.** `Q = orth([U0 | K1])` per charge sector. **No tolerance is
   applied here** — every direction the step finds is kept, and the only
   constraint is the Sulz bound `rank([U0|K1]) ≤ 2r`.
4. **Missing-quantum-number fill.** Under U(1) with the opposite frame frozen,
   `K1` stays inside `U0`'s sectors and can never *open* a new one, so a reachable
   sector that neither populates is seeded with a minimal random orthonormal
   block. Only sectors whose dual is reachable on the other side are seeded — an
   unpaired one is structurally zero and would be dead weight. The fill draws
   from the same `2r` budget as the complement, and goes **first**: starve it and
   the state freezes.
5. **S-step.** `Ŝ0 = Û'·Θ0·V̂'` directly — **no overlap matrices** `M̂`/`N̂`. The
   Galerkin generator on the augmented bases is Hermitian, so this one is Lanczos.
6. **Truncate.** A symmetry-blocked SVD of `S1` sets the new bond dimension and
   prunes any seeded sector the dynamics left empty.

## `BondUpdateOptions`

| field | default | |
|---|---|---|
| `dt`, `n_steps` | 0.05, 10 | real time step and count |
| `order` | `:strang` | `:strang` (even ½, odd, even ½) or `:lie` |
| `maxdim` | 200 | hard bond-dimension cap |
| `trunc_thresh` | 1e-12 | singular-value cutoff for the S-step split |
| `normalize` | true | rescale after each step; the norm is recorded *before* |
| `augment`, `missing_fill` | true, 1 | rank adaptation; there is deliberately **no** K/L tolerance |
| `lanczos_tol`, `lanczos_maxiter` | 1e-15, 30 | Krylov budget for all three substeps |
| `seed` | `0x5EED` | one RNG for the whole run, so a run is reproducible |

`bond_update_bug!` returns a `BondUpdateInfo` with `times`, `norms`,
`bond_dims`, `max_bond_dims`, `aug_k_dims`, `aug_l_dims` and `discarded`.

## Accuracy

L=6 Heisenberg, `Dmax=8`, `dt=0.01`, against a dense propagator using the *same*
odd/even split:

| | |
|---|---|
| projection error | 6.25e-5, converging second order in `dt` |
| vs the Alice reference kernel (`⟨Sz_j⟩`) | 6.33e-8 |
| vs Alice with the Sulz bound relaxed | **4.27e-11** |
| XX vs the free-fermion analytic solution | < 1e-6 |

BUG is **not** exact at `Dmax=8`: `exp(-iτh)Θ` has right support up to twice the
link support, so the `h²` term wants more room than `2r` permits, and the
resulting O(τ²) local error is intrinsic to the bound rather than a defect. The
6.33e-8 gap from Alice is *entirely* the strict `2r` enforcement — the port
itself agrees to 4.27e-11, and the fill's RNG contributes exactly nothing (four
seeds, zero spread). Accuracy is to come from raising the order of the sweep,
never from widening the basis past `2r`.

## Tests

```bash
sbatch --job-name=t_all --mem=32G scripts/run_julia.sbatch tests/runtests.jl
```

720 tests. The Python parity test consumes
`tests/crosscheck/reference_l6_heisenberg.json`, regenerated with:

```bash
sbatch scripts/run_python.sbatch tests/crosscheck/export_python_reference.py
```

Two independent references are used, and each is validated before it is relied
on: `tests/common/dense_reference.jl` (shares the integrator's conventions) and
`tests/common/free_fermion.jl` (derived from the Hamiltonian on paper, no 2^L
object anywhere). They agree to 1e-11.

## Retired paths

`exploratory/` holds the global-sweep discarded BUG and the pre-refactor
ITensors tree (TDVP, TTutils, the faithful-KLS kernel). None of it loads against
this package's dependencies; it is a record, not a working state.
