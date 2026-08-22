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
    cbe_bug_step!(psi, W, ComplexF64(-im * 0.05); maxdim = 32, trunc_thresh = 1e-10)
end

bond_dims(psi)          # rank grew from a product state: 1 -> [2,4,8,16,8,4,2]
mpo_energy(psi, W)
```

| function | what it is |
|---|---|
| `cbe_bug_step!` | RSVD-CBE + rank-adaptive BUG. `K`/`L` half-sweeps build the bases, one Galerkin step at the centre root. |
| `tdvp_cbe1s_step!` | 1-site TDVP with controlled bond expansion ([arXiv:2208.10972](https://arxiv.org/abs/2208.10972)) — the baseline to beat. |
| `tdvp2_step!` | 2-site TDVP. The conventional reference. |

`cbe_bug_step!` has **three scheme knobs**, all defaulting to on:

- **`kstep`** — do the BUG basis update. `false` takes the CBE frame directly.
- **`rexpand`** — expand every bond **twice per half-sweep**, once as site `i`'s right bond and
  once as site `i+1`'s left bond, so every site is evolved with **both** of its bonds widened.
  This is the default mechanism for keeping the width the half-sweep split would otherwise lose.
- **`kaug`** — the older mechanism for the same thing: UNION the basis update with the CBE frame
  rather than letting it replace it. **`rexpand` overrides it**, so an A/B on `kaug` must pass
  `rexpand = false`.

Turning off *both* is not a tuning choice — it is a wrong answer. On the TFIM example it returns
`1.4e+00` against `6.6e-06`, while still exiting 0 and writing its CSV. See
[docs/USAGE.md §6b](docs/USAGE.md) and `benchmarks/arm4_rexpand.jl`.

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
to read the diagnostics, the `kaug` result (§6b), and the gotchas.

📁 **[examples/README.md](examples/README.md)** — the three example scripts, their parameters,
the CSV columns, and what the randomised sketch does and does not affect.

## Retired paths

`exploratory/` holds the global-sweep discarded BUG and the pre-refactor ITensors tree. None of
it loads against this package's dependencies; it is a record, not a working state.
