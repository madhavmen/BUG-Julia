# Two-site BUG odd/even sweep

This folder tests the two-site (single-moving-orthogonality-center) BUG
integrator implemented in
[`src/BUG/two_site_bug/two_site_bug.jl`](../../../src/BUG/two_site_bug/two_site_bug.jl).

## The scheme

Split the Hamiltonian into two parity groups of nearest-neighbour terms

```
H = H_odd + H_even,   H_odd = Σ_{b odd} h_{b,b+1},   H_even = Σ_{b even} h_{b,b+1}
```

Odd bonds touch the disjoint pairs (1,2),(3,4),(5,6),…; even bonds touch
(2,3),(4,5),…. One public `bug_two_site!` step approximates `exp(-i dt H)` by a
symmetric (Strang) product

```
exp(-i dt H) ≈ U_odd(dt/2) · U_even(dt) · U_odd(dt/2) + O(dt³)   per step,
```

and each parity evolution `U_par(τ) = exp(-i τ H_par) = Π_{b∈par} exp(-i τ h_{b,b+1})`
is realised **bond-by-bond by the BUG KLS local update** (K-augment, L-augment,
S-evolve).

### Key design points (and why)

* **We apply the Hamiltonian, not its exponential.** The KLS step is fed the
  Hamiltonian *term* `h_{b,b+1}` (the slice of the parity MPO on that bond) and
  exponentiates the projected effective Hamiltonian *internally* (Krylov/Lanczos
  in the K/L/S substeps). We never form `exp(-i τ H_par)` or a per-bond gate
  matrix.
* **Per-bond term with a trivial environment — no double counting.** Within one
  parity the other terms act on *disjoint* sites. Contracting the whole parity
  MPO into the bond's effective Hamiltonian would drag those other terms into
  the environment and apply them a second time at their own bonds — a flat error
  floor (~0.10 for N=6 XX) that does **not** shrink with dt. Instead each bond
  sees only its own term; the canonical environment is the identity, so the
  local update reproduces `exp(-i τ h_{b,b+1})` exactly at full rank.
* **Single moving center — no diagonal inverse.** We keep one orthogonality
  center and move it along the chain with QR/LQ (`orthogonalize!`). At the active
  bond that gives exactly the left-isometry / right-isometry condition the KLS
  step needs, for that one bond only. No simultaneous-center canonical form is
  used, so no diagonal inverse ever appears in the update.
* **No backward correction.** The two-site BUG has no backward (single-site,
  backward-in-time) sub-step; the disjoint parity structure removes the
  shared-site double counting that the 2-site TDVP backward step exists to
  cancel. `BUGInfo.backward_correction_calls` stays 0.

## Error: Trotter only, tracked vs dt

At full rank, with enough augmentation depth to saturate the local frames, every
bond update is the *exact* local evolution. The only remaining error is the
odd/even **Trotter splitting**, set by `[H_odd, H_even] ≠ 0`. It vanishes as
`dt → 0` at the composition order. Measured (N=6 XX, rank-3, T=0.5, relative L2
error vs the exact propagator):

| dt     | Strang error | ratio | Lie error | ratio |
|--------|-------------:|:-----:|----------:|:-----:|
| 0.1    | 1.35e-4      |   –   | 9.31e-3   |   –   |
| 0.05   | 3.36e-5      | 4.00  | 4.65e-3   | 2.00  |
| 0.025  | 8.41e-6      | 4.00  | 2.33e-3   | 2.00  |

Strang halves to 1/4 (global O(dt²)); Lie halves to 1/2 (global O(dt¹)) — exactly
the expected Trotter orders, with no floor.

## Tests

* **`test_two_site_decomposition.jl`** — Hamiltonian decomposed properly:
  `matrix(W_odd)+matrix(W_even)==matrix(W_full)`; within-parity terms commute
  while the two parities do not (the genuine Trotter source); the per-bond gate
  generators sum to `H`; the dressed local effective Hamiltonian acts on the
  2-site block exactly like the bare gate (identity environment).
* **`test_two_site_environments.jl`** — environments updated properly: after
  `orthogonalize!`, `U0`/`V0` are left/right isometries (the local environment is
  the identity); the snapshot factorizes the block exactly (`U0·S0·V0 == ψ[b]·ψ[b+1]`);
  center transport keeps a valid normalized MPS; and the incremental MPO env
  builders match a from-scratch full recontraction.
* **`test_two_site_local_kls.jl`** — sweep structure / KLS correctness: a parity
  sweep reproduces `exp(-i τ H_par)` to machine precision; one full step equals
  the exact Strang product to machine precision; `backward_correction_calls == 0`
  over single and multi-step runs.
* **`test_two_site_trotter_convergence.jl`** — Trotter-error tracking: Strang
  O(dt²), Lie O(dt¹), monotone decrease, norm preservation, Strang ≪ Lie at equal
  dt.

Run the suite:

```
julia --project tests/BUG/two_site_bug/run_two_site_tests.jl
```

## Public API

```julia
gates = two_site_xx_bond_gates(sites; J = 1.0)        # per-bond 2-site terms h_{b,b+1}
info  = bug_two_site!(psi, gates; dt, order = :strang, # or :lie
                      maxdim, aug_krylov_depth, ...)   # info.backward_correction_calls == 0
```

`two_site_xx_parity_mpos(sites; J)` returns `(W_odd, W_even, W_full)` as a
reference decomposition for verification (the integrator consumes the per-bond
gates).
