# Examples

Runnable scripts that reproduce the benchmarks this package exists to support. Each writes a CSV
to `examples/results/` and is plotted by `plot_examples.py`.

## Running one

```bash
julia --project=. examples/oat_z2.jl          # defaults
python3 examples/plot_examples.py             # figures for every CSV present
```

The three models:

```bash
julia --project=. examples/oat_z2.jl          # one-axis twisting, Z2        (constant rank)
julia --project=. examples/xx_domain_wall.jl  # XX chain from a domain wall  (growing rank)
julia --project=. examples/tfim_z2.jl         # transverse-field Ising, Z2   (growing rank)
```

Every script takes its parameters from a block at the top of the file marked
`PARAMETERS -- EDIT THESE`. You can also override them on the command line:

```bash
julia --project=. examples/oat_z2.jl L=12 t_over_pi=4 dt=0.05 trunc_thresh=1e-10 maxdim=16
julia --project=. examples/xx_domain_wall.jl L=24 t_max=10 maxdim=128 symmetry=none
julia --project=. examples/tfim_z2.jl L=20 t_max=5 h=0.5 maxdim=128
```

Unrecognised names are rejected rather than ignored, so a typo stops the run instead of silently
producing a figure from the defaults.

### Parameters every script has

| name | meaning |
|---|---|
| `L` | number of sites |
| `t_max` *(XX, TFIM)* | total simulated time, in units of `1/J` |
| `t_over_pi` *(OAT only)* | total simulated time, in units of π |
| `dt` | time step |
| `maxdim` | bond-dimension cap |
| `trunc_thresh` | CBE root-to-leaves truncation threshold at the end of each sweep |
| `maxiter` | Krylov depth. `0` derives it from `‖H‖·dt/2` — leave it there unless you are testing the solver |
| `sample_every` | how much physical time between recorded rows |
| `symmetry` | the symmetric run, or `none` for the dense control |

`oat_z2.jl` additionally accepts `maxdim=0`, meaning "use the exact rank ceiling".
`xx_domain_wall.jl` has `J`; `tfim_z2.jl` has `J` and `h`.

⛔ **Only OAT takes its time in units of π**, because only OAT has a period — the state revives
every `4π`, so a run is naturally quoted as a fraction of a revival. XX and TFIM have no such
structure; their scales come from `J` (and `h`), and the landmark that matters for XX is the
front reaching the boundary at `t ≈ L/(2J)`. Quoting those two in π would hide the one number a
reader needs, so they take a plain `t_max`.

## The three integrators

All three advance the same MPS under the same MPO, so any difference between them is the
integrator and nothing else.

| name | what it is |
|---|---|
| `tdvp2` | 2-site TDVP. The usual reference. Rank grows from the 2-site block. |
| `tdvp_cbe1s` | 1-site TDVP with controlled bond expansion (arXiv:2208.10972). Matches 2-site accuracy at 1-site cost. |
| `cbe_bug` | RSVD-CBE + rank-adaptive BUG. `K`/`L` half-sweeps build the bases, one Galerkin step at the centre root. |

`cbe_bug` has three scheme knobs. These scripts run it at its **defaults**, with the one
simplification that the randomised sketch is switched off (`exact = true`, measured free — see
below):

- **`kstep`** (default `true`) — do the BUG basis update.
- **`rexpand`** (default `true`, **what these scripts pass**) — expand every bond **twice per
  half-sweep**, once as site `i`'s right bond and once as site `i+1`'s left bond, so every site is
  evolved with both of its bonds widened. No basis is ever formed by merging two others. This is
  the same ordering `tdvp_cbe1s` gets for free from its forward and backward passes.
- **`kaug`** (default `true`) — the older mechanism for the same thing: UNION the basis update
  with the CBE frame instead of letting it replace it. **`rexpand` overrides it**, so an A/B on
  `kaug` must pass `rexpand = false` or it silently compares an arm with itself.

Both mechanisms exist because the half-sweep split is bounded by the evolve, so the frame comes
out NARROWER than the CBE expansion that was just paid for — and that frame sets the next site's
left bond. `rexpand` restores the width by expanding again; `kaug` by merging the frame back in.

⛔ **Restoring the width is not optional — but which mechanism does it barely matters, and on one
model neither matters at all.** Measured at these scripts' own settings
(`benchmarks/example_defaults.jl`, L=16 `maxdim=64` for TFIM and XX; OAT L=10 at the exact
ceiling, t=10π):

| mechanism | TFIM `:Z2` | TFIM `:none` | XX `:U1` | OAT infidelity | OAT bond profile |
|---|---|---|---|---|---|
| **`rexpand = true`** (default) | **3.8455e-06** | 4.3036e-06 | 8.1911e-05 | **3.2748e-11** | `[2,3,5,5,6,5,4,3,2]` |
| `kaug = true` | 4.2717e-06 | **4.0839e-06** | 8.1911e-05 | 3.2750e-11 | `[2,3,6,6,6,6,6,3,2]` |
| neither | 1.9699e-01 | 1.9474e-01 | 8.0282e-05 | 3.9347e-05 | `[2,3,6,6,6,6,6,3,2]` |

Read three things off it:

1. **`rexpand` and `kaug` are at parity on accuracy** — within ~11%, and the direction flips
   between symmetry modes. `rexpand` is the default for its *structure*, not for a win.
2. **`neither` is catastrophic on TFIM and OAT and free on XX.** The run still exits 0 and still
   writes its CSV, so on TFIM the failure is silent at five orders. **Never validate a
   basis-mechanism change on XX** — it stays low-rank enough not to discriminate, and has now
   failed to show three separate real effects.
3. **The bond profile is not the sensitive diagnostic it was documented to be.** `kaug` and
   `neither` have *identical* profiles with errors six orders apart. The profile separates
   `rexpand` — which nearly tracks the exact `min(k, L−k)+1` = `[2,3,4,5,6,5,4,3,2]` — from the
   two that sit pinned at the cap, and that is the one place `rexpand` is clearly ahead: equal
   accuracy on a leaner state.

`docs/USAGE.md` §6b has the defect analysis behind both mechanisms.

The sweep is always interleaved (expand bond `i`, then immediately evolve site `i`) — the
ordering of the reference implementation.

### The randomised sketch — measured, and switched off

These scripts run `cbe_bug` with **`exact = true`**: the exact SVD, no randomised sketch. That
costs nothing, and the measurement is the reason.

At L=16 on **both** OAT and XX (`benchmarks/rsvd_param_scan.jl`, phase 0), the sketch and the
exact SVD give `cbe_bug` results that are **bit-identical** — relative difference `0.000e+00`,
not "agrees to N digits":

| model | sketch | exact SVD | rel. diff |
|---|---|---|---|
| OAT | 7.755e-01 | 7.755e-01 | **0.000e+00** |
| XX | 4.5904130e-05 | 4.5904130e-05 | **0.000e+00** |

The sketch is **saturated**: it already spans the whole relevant subspace, so a different draw is
just a different basis of the same space. That is why `dover`, `comp_ratio` and the RNG seed are
all inert for this scheme — and why the oversampling knob was removed from these scripts rather
than left as a dial that does nothing.

This was checked on XX specifically because its rank *grows*, which is where a sketch should come
under pressure. It doesn't.

⛔ **The randomisation is a cost optimisation, never an accuracy one — and it does not always
win.** On OAT at `maxdim = 6` the exact SVD was *faster* (68.9s vs 72.8s): at small rank the
sketch's fixed overhead never amortises. It pays off when the target rank is much smaller than
the matrix dimension.

The budget knobs (`dex`, `growth`) were measured too, and on XX they move the error by **0.6%
across their entire range** — from discarding 59% of ranked candidate weight down to 0%. At fixed
`maxdim` the error is set by the rank, not by the search. **To improve accuracy, raise `maxdim`.**

## What comes out

Each CSV has one row per sample time per scheme, with columns:

| column | meaning |
|---|---|
| `model`, `scheme`, `symmetry`, `L`, `maxdim`, `dt` | what was run |
| `t` | physical time |
| `obs`, `obs_exact` | the observable and its exact value |
| `err` | `abs(obs - obs_exact)` — the error plot's y-axis |
| `maxbond` | largest bond dimension at that time |
| `energy`, `dE` | energy and its drift from `t = 0` (a conservation check, NOT an accuracy check) |
| `krylov` | cumulative operator applications — the honest cost axis |

`plot_examples.py` produces one figure per run, with three panels:

1. **observable vs time**, with the exact curve underneath
2. **error vs time**, log scale
3. **max bond dimension vs time**, with the cap drawn in

⛔ **Rank the schemes by `err`, never by `dE`.** They disagree outright on these models: a scheme
can conserve energy to machine precision and still be an order of magnitude worse against the
exact solution. `dE` is in the CSV as a conservation check and is deliberately not plotted.

## The models

Every exact reference below is a **closed form or a free-fermion solution**, never a second
numerical integrator and never a `2^L` diagonalisation. That is what makes the error column an
error rather than a disagreement, and it keeps the reference available at sizes where exact
diagonalisation is not.

### `oat_z2.jl` — one-axis twisting

```
H = (Σ_ℓ S^z_ℓ)² / 2 ,   |Ψ(0)⟩ = ⊗_ℓ |→⟩ ,   observable S^x_tot(t)
```

Everything is closed form: `S^x_tot(t) = (L/2)·cos^(L-1)(t/2)`, and the exact Schmidt rank at the
cut after site `n` is `min(n, L-n)+1` **for all time**. So `maxdim ≥ min(n,L-n)+1` is full rank and
anything below is a genuine truncation — both regimes are worth running.

Runs under the paper's own **ℤ₂ spin symmetry** (global spin flip, σˣ basis), where `S^x` is
diagonal and the polarised start is a single charge-0 sector at χ=1. Set `symmetry=none` for the
unsymmetric control; the two agree to ~1e-12.

The state revives every `4π`, so `t_over_pi = 10` covers 2.5 revivals — enough to tell a bounded
error from one that accumulates, which a single period cannot.

The rank being **constant in time** is what makes this the calibration model: the bond-dimension
panel is a flat line, so nothing in the comparison is about rank adaptivity.

### `xx_domain_wall.jl` — XX chain from a domain wall

```
H = J Σ_ℓ (S^x_ℓ S^x_{ℓ+1} + S^y_ℓ S^y_{ℓ+1}) ,   |Ψ(0)⟩ = |↑…↑↓…↓⟩
observable  Σ_{ℓ ≤ L/2} ⟨S^z_ℓ⟩   — magnetisation still in the left half
```

Jordan-Wigner makes this free fermions, so a Slater determinant stays one and the exact answer
lives in an `L × L` matrix — exact at finite `L`, open boundaries, any time, `O(L³)`.

Runs with and without **U(1)** (`S^z_tot` conservation); the physics is identical and only the
block structure differs, so a gap between them is a bookkeeping bug rather than physics.

Unlike OAT there is **no rank ceiling** — the wall spreads and entanglement grows — so `maxbond`
climbing into the cap is expected and the bond-dimension panel is the interesting one. The front
travels at `≈ J` sites per unit time and reaches the boundary at `t ≈ L/(2J)`; the script prints
which side of that the run ends on.

### `tfim_z2.jl` — transverse-field Ising from a domain wall

```
H = -J Σ_ℓ Z_ℓ Z_{ℓ+1} - h Σ_ℓ X_ℓ ,   |Ψ(0)⟩ = |+…+−…−⟩
observable  Σ_{ℓ ≤ L/2} ⟨X_ℓ⟩   — transverse magnetisation left of the wall
```

Uses `tfim_mpo`, an explicit virtual-dimension-3 MPO, so this model drives the same environment
machinery and Krylov solves as the other two.

⛔ **The basis is σˣ, so `X` is diagonal and `Z` is the flip** — the opposite of the usual
convention. That is what makes the ℤ₂ charge (the global spin flip) diagonal and the symmetric run
possible; it also means `X` is the measurable one and `Z` carries the charge. Same trade as OAT.

The exact reference is free fermions again, via the Majorana form: an X-basis product state is
Gaussian and stays Gaussian, so the solution is a `2L × 2L` real orthogonal evolution. Validated
against dense diagonalisation at L=8 to 7e-14 on symmetric and asymmetric states alike, plus the
analytic pin `⟨+…+|H|+…+⟩ = -h·L` for any `J`.

`h = J` is the critical point of the infinite chain, and is the default.

There is **no U(1) or SU(2) form** — `Z_ℓ Z_{ℓ+1}` does not conserve `S^z` and the field picks an
axis — which is why this is the ℤ₂ example and XX is the U(1) one. `symmetry=none` is the control.
