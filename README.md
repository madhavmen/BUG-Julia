# BUG-Julia — `bond_update_bug`

A symmetry-native, rank-adaptive Basis-Update-and-Galerkin time integrator for
1D tensor networks, built on Telum / LurCGT so that U(1)
and non-Abelian symmetric tensors stay block-sparse throughout.


## Quick start

The bond update is **RSVD controlled bond expansion** (`rsvd_cbe`) — rank grows only from
directions drawn through the gate, with no random sector seeding anywhere.

```julia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime

set_symmetry!(:U1)                                # :none | :U1 | :SU2 — set it FIRST
psi   = domain_wall_state(8)                      # |up up up up down down down down>
gates = bond_gates(psi; J = 1.0, delta = 1.0)     # delta = 0 gives XX

info = RealTime.evolve!(psi, gates; opts = CBEBugOptions(
    dt = 0.05, n_steps = 40, maxdim = 32, s_iters = -1, maxiter = 8))

magnetisation(copy(psi))                          # the evolved profile
bond_dims(psi)                                    # rank growth: 1 -> [2,4,8,16,8,4,2]
info.max_bond_dims                                # per step
```

**Two paths, same expansion and same S-step**, differing only in how `H` is applied:

```julia
RealTime.evolve!(psi, gates; opts)                       # gate-based: Strang over bond gates
RealTime.evolve_mpo!(psi, xxz_chain(8; delta = 1.0); opts)   # Lubich: MPO envs, no Trotter error
```

Also `ImaginaryTime.evolve!` / `cool!` and `GroundState.rsvd_cbe_dmrg!` — same engine, the mode
only fixes `tau` and what is recorded.

📖 **[docs/USAGE.md](docs/USAGE.md)** — the full guide: symmetry switching, both paths, the
three modes, which options matter and why, how to read the diagnostics, and the gotchas.

▶️ **[examples/heisenberg_domain_wall.jl](examples/heisenberg_domain_wall.jl)** — runnable L=8
Heisenberg domain wall in `:none` and `:U1`, both paths, checked against an exact
sparse-Krylov reference:

```bash
julia --project=. examples/heisenberg_domain_wall.jl
```
```
── symmetry = :none ──          ── symmetry = :U1 ──
   bond dims : [2,4,8,16,8,4,2]    bond dims : [2,4,8,16,8,4,2]
   max |error vs exact| = 9.901e-05

SAME PHYSICS CHECK  max |<Sz>_none - <Sz>_U1| = 4.746e-15
```

### The earlier KLS bond update

`bond_update_bug!` with `BondUpdateOptions` is the original K/L/S kernel, described below. It
is **superseded** by the RSVD-CBE path above and is no longer the default: its rank growth
under U(1) comes entirely from `missing_fill` — a random column seeded into an empty charge
sector — and it cannot grow rank at all under `:none`, where there is no empty sector to seed.
RSVD-CBE needs no such thing and grows in every mode.

## The algorithm

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
5. **S-step.** `Ŝ0 = Û'·Θ0·V̂'` .
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
