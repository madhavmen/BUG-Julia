# exploratory/

Research paths kept for reference but **not** the supported integrator.
The one supported BUG implementation is `src/BondUpdateBUG/` (`bond_update_bug!`).

- `global_sweep/` — the discarded-projector BUG as a single global sweep
  (`phi = H*psi` formed once, augmented bases spanning `range(psi) + range(H psi)`).
  Measured first-order in the state; superseded by the per-bond `bond_update_bug`.
  Not covered by `tests/runtests.jl`.

Everything here is built on **ITensors**, which is no longer a dependency of this
package, so none of it loads as-is. It is a record, not a working state. The full
pre-refactor tree is at the tag `archive/pre-bond-update-legacy`.

- `legacy_itensors/` — the pre-refactor tree: `ProjectorSplitting.jl` (the old
  entry point, whose `include("BUG/BUG.jl")` now dangles), `TDVP/`, `TTutils/`
  and their tests. Parked rather than deleted because TDVP2 is still the
  comparison method for rank-growth studies, but it is ITensors-based and does
  not load against this package's dependency set.
