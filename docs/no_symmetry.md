# Running `bond_update_bug` with and without symmetry

`bond_update_bug!` runs either symmetry-native (U(1) charge sectors, the default)
or without any symmetry (one dense block). Set the mode once, before building the
state and its gates -- they must share it:

```julia
set_symmetry!(:U1)    # default: U(1)-symmetric, block-sparse
set_symmetry!(:none)  # no symmetry: a single dense sector

psi   = domain_wall_state(20)
gates = bond_gates(psi; J = 1.0, delta = 0.0)   # XX
info  = bond_update_bug!(psi, gates; opts = BondUpdateOptions(
            dt = 0.05, n_steps = 600, maxdim = 64, criterion2 = true))
```

Every tensor is self-describing (`symm(t)`), so the kernel dispatches on the
tensors' own symmetry; only the local space, the product-state builder, and the
bond gate branch on the mode.

## Rank growth without symmetry

Under U(1) a product state grows rank through the missing-sector fill: the
augmentation opens a reachable charge sector that neither frame populates yet.
Without symmetry there is a single sector that is never "missing", so from a
product state the rank-r Galerkin core cannot see the off-diagonal generator
(each K/L half-step holds the opposite frame fixed and projects the flip to zero)
and the state freezes at bond dimension 1. Two options restore growth:

- `criterion2 = true` (recommended) -- Ceruti-Kusch-Lubich residual enrichment:
  enrich each frame with the residual of the FULL 2-site update HTheta, i.e.
  exactly the direction the dynamics needs, capped at the Sulz 2r. Grows rank by
  the minimal physical amount -- no wasted columns. This is the rank growth 2-site
  TDVP gets from its 2-site SVD.
- `pad = true` -- complete each frame to 2r with orthogonal random directions.
  Simpler but less efficient: it does the S-step on a doubled core every step and
  the truncating SVD prunes the directions that never gain weight.

For XX at L=8 (exact, chi<=16) both track the U(1) run to ~1.6e-4; criterion 2
and padding agree to ~8e-5.

## Telum patch required for no symmetry

Telum's SVD was never exercised with zero symmetries and errors on the empty
`prod(... for n in 1:N)` reductions at N=0. The one-file fix (add `init = 1`, plus
a trivial N=0 core-kron method) is in `telum_nosym_svd.patch`. It is applied to
the local package copy; re-apply after any Pkg operation that re-extracts Telum,
or carry it via a Telum fork / upstream PR.
