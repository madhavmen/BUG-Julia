# Telum Quick Reference

This file summarizes the core data model and the public functions exported by
`Telum/src/Telum.jl`. It is intended as a fast orientation note for future
coding sessions.

## Mental Model

`TLArray` stores a symmetry-adapted tensor as a list of symmetry sectors. Each
sector row contains:

- symmetry metadata (`CGR` objects) describing the Clebsch-Gordan tensor side;
- a reduced matrix tensor (`RMT`) containing the numerical degrees of freedom.

The physical tensor is therefore not stored densely. Operations such as
contraction, SVD, eigendecomposition, addition, and direct sums operate on this
sector representation.

## Core Types

### `TLArray{T, QD, N, RD, QT, PS, CGRS}`

Concrete symmetry-adapted tensor container.

Type parameters:

- `T`: element type of each reduced matrix tensor, such as `Float64` or
  `ComplexF64`.
- `QD`: physical tensor rank, i.e. the number of external legs.
- `N`: number of symmetry factors in the product symmetry.
- `RD`: rank of the stored reduced matrix tensor. In normal TLArray rows this is
  `QD + N`: physical axes followed by one outer-multiplicity axis per symmetry.
- `QT`: type of one physical leg sector label. For `(U1, SU{3})` this is
  `Tuple{Tuple{Int}, NTuple{2, Int}}`.
- `PS`: compile-time `ProductSymm` type encoding the symmetry tuple.
- `CGRS`: concrete tuple type of the row's `CGR` metadata.

Fields:

- `rows::Vector{row{T, QD, N, RD, CGRS}}`
  Stores the nonzero symmetry sectors. Each row carries one tuple of CGR
  metadata and one reduced matrix tensor. A row is identified by the qlabels on
  its physical legs after applying each CGR's `cgp` permutation.
- `inds::NTuple{QD, TLIndex}`
  Stores the external leg identities. These are the ITensor-like leg descriptors
  used for leg selection, matching, contraction, priming, locking, tagging, and
  display. `TLArray(q, inds)` may replace indices without changing rows, but
  arrow directions must match the original tensor.
- `spaces::NTuple{QD, Vector{Tuple{QT, Int}}}`
  Cached space list for each physical leg. Each entry is `(qlabels, RMT_dim)`,
  where `qlabels` is one product-symmetry sector and `RMT_dim` is that sector's
  reduced multiplicity dimension on the leg. Contraction, identity building,
  direct sums, singleton tests, and subspace extraction rely on this cache.

Constructor side effects and invariants:

- converts rows and spaces to concrete typed containers;
- normalizes rank-2 and rank-0 rows with `normalize_tlarray!`;
- orients one-element negative w-matrices by moving the sign into `RMT`;
- drops tiny rows using `TLARRAY_ROW_CUTOFF`;
- checks CGR qlabel ordering, unique non-empty indices, and lock rules.

### `TLIndex`

Descriptor for one external leg.

Fields:

- `itags::Itag`: sorted comma-separated tag set. Empty means untagged.
- `dir::Char`: `'+'` for incoming or `'-'` for outgoing.
- `plev::Int`: prime level.
- `lock::Int`: contraction lock level. `0` is contractable, positive values
  block contraction until reduced, and `-1` is permanent lock.
- `dual::Bool`: marks the dual/green version of a space.

Equality ignores `lock`, so matching compares tags, direction, prime level, and
dual flag.

### `Itag`

Canonical tag wrapper around a string. Tags are split by comma, stripped,
deduplicated by sorting behavior in helper operations, and stored as a sorted
comma-separated string.

### `row`

One sector row in a `TLArray`.

Fields:

- `cgrs`: one `CGR` per symmetry factor.
- `RMT`: the reduced matrix tensor for this sector.

### `CGR`

Clebsch-Gordan representation metadata for one symmetry factor in one row.

Fields:

- `qlabels`: qlabels in canonical CGT order.
- `wmat`: outer-multiplicity mixing matrix.
- `cgp`: maps each physical TLArray leg to the stored qlabel position.
- `legdir`: `(# incoming, # outgoing)` in the stored CGT ordering.

### `LurTensor`

Thin `AbstractArray` wrapper around `data::AbstractArray`. Most arithmetic and
array behavior delegates to the wrapped `data`.

### `ProductSymm`

Compile-time tag for a tuple of symmetry groups. `ProductSymm(U1, SU{2})`
builds a type encoding that product symmetry.

## Local Space Options

- `LocalSpaceOptions`: abstract supertype for objects accepted by
  `getLocalSpace`.
- `SpinOptions`: pure spin local space. Fields are `symmetry` and `spin`.
- `FermionOptions`: spinless fermion local space. Fields are `nchannels`,
  `charge_symm`, and `channel_symm`.
- `FermionSOptions`: spinful fermion local space. Fields are `nchannels`,
  `charge_symm`, `spin_symm`, and `channel_symm`.

`Z`, `U1`, `SU`, `SO`, and `Sp` are re-exported from `LurCGT`.

## Public Function Summary

### Symmetry Metadata

- `productsymm(symm)`: convert a tuple of symmetry types, or an existing product
  symmetry type, to a `ProductSymm` type.
- `product_symms(x)`: recover the underlying tuple of symmetry types from a
  `ProductSymm`, symmetry tuple, `TLArray`, or `leginfo`.
- `symm(x)`: return the symmetry tuple for a `TLArray`, `CGR`, or `leginfo`.
- `nsymms(x)`: return the number of symmetry factors.
- `qlabeltype(symm_or_q)`: return the concrete qlabel type for one leg sector.
- `zero_qlabels(symm_or_q)`: return the trivial qlabel for each symmetry factor.

### Construction And Local Operators

- `getLocalSpace(opts, tags=("", "", ""))`: build symmetry-adapted local-space
  operators for `SpinOptions`, `FermionOptions`, or `FermionSOptions`. Returns a
  named tuple such as `IS`, `Z`, `F`, `S`, depending on the option type.
- `empty_tlarray(symm, inds; T=Float64)`: create a zero-row `TLArray` over a
  symmetry tuple and index tuple.
- `empty_tlarray(q; T=Float64)`: create an empty tensor with `q`'s symmetries and
  indices.
- `getvac(q, itags=("", ""))`: build a rank-2 vacuum tensor with one incoming
  and one outgoing trivial leg.
- `getIdentity((q, leg)...; itag="", plev=0, lock=0)`: build a fused identity
  tensor for one or more selected legs.
- `getIdentity(q, legs; itag="", plev=0, lock=0)`: same as above, selecting legs
  from one tensor.
- `get1jtensor(q, leg)` or `get1jtensor(q; selectors...)`: build the 1j tensor
  that converts a leg to its dual direction-compatible counterpart.
- `legflip(q, leg_or_legs)` or `legflip(q; selectors...)`: contract with 1j
  tensors to flip selected legs.

### Leg Selection And Matching

The selector keywords used throughout are `dir`, `itag`, `plev`, `lock`, and
`rev`. `rev=true` selects the complement.

- `findlegs(q, pred)` / `findlegs(q; selectors...)`: return all matching leg
  positions.
- `findleg(q, pred)` / `findleg(q; selectors...)`: return the first matching
  leg, or `nothing` if none match.
- `matchings(a, b; selectors...)`: legs of `a` that have a matching leg in `b`.
  Matching means same tags, prime level, and dual flag, with opposite direction.
- `matching(a, b; selectors...)`: first matching leg of `a`, or `nothing`.
- `unmatchings(a, b; selectors...)`: legs of `a` with no matching leg in `b`.
- `unmatching(a, b; selectors...)`: first unmatched leg of `a`, or `nothing`.
- `contractables(a, b; selectors...)`: matching legs whose lock level is `0` on
  both tensors.
- `contractable(a, b; selectors...)`: first contractable leg of `a`, or
  `nothing`.
- `uncontractables(a, b; selectors...)`: legs of `a` with no unlocked
  contractable partner in `b`.
- `uncontractable(a, b; selectors...)`: first non-contractable leg of `a`, or
  `nothing`.

### Index Metadata Editing

These functions return new `TLArray` objects with modified `inds`; rows and
spaces are reused through the constructor.

- `prime(q; inc=1, selectors...)`, `prime(q, leg_or_legs; inc=1)`,
  `prime(q, pred; inc=1)`: increase selected prime levels, clamped at `0`.
- `setprime(q, n; selectors...)`, `setprime(q, legs, n)`: set selected prime
  levels to `n`.
- `noprime(q; selectors...)`, `noprime(q, leg_or_legs)`,
  `noprime(q, pred)`: set selected prime levels to `0`.
- `lock(q; inc=1, selectors...)`, `lock(q, leg_or_legs; inc=1)`,
  `lock(q, pred; inc=1)`: increase lock level, preserving permanent locks.
- `lockp(q; selectors...)`, `lockp(q, leg_or_legs)`, `lockp(q, pred)`: set
  selected locks to `-1`.
- `unlock(q; selectors...)`, `unlock(q, leg_or_legs)`, `unlock(q, pred)`: set
  selected locks to `0`.
- `additag(q, newtags; selectors...)`, plus leg/list/predicate overloads: add
  comma-separated tags.
- `removeitag(q, tags; selectors...)`, plus leg/list/predicate overloads:
  remove tags. A tuple/vector of tag queries applies each query independently.
- `replaceitag(q, old=>new...; selectors...)`, plus dictionary and
  leg/list/predicate overloads: replace matched tag queries.
- `setitag(q, tags; selectors...)`, plus leg/list/predicate overloads: replace
  the whole tag string on selected legs.

### Tensor Operations

- `contract(q1, legs1, q2, legs2; reduce_lock=true, verify_legs=true)`: contract
  selected legs. Legs must have opposite directions, matching tags, matching
  dual flags, and matching `spaces` when verification is enabled.
- `q1 * q2`: automatically contract all tagged, unlocked, contractable matching
  legs between two tensors.
- `permutedims(q, perm)`: reorder physical legs. `perm[new_pos] = old_pos`.
- `conj(q)` / `adjoint(q)`: conjugate RMT data and reverse leg directions while
  updating CGR metadata.
- `norm(q)`: Frobenius norm computed directly from sector RMTs and CGT
  normalization.
- `q1 + q2`, `q1 - q2`: add/subtract compatible tensors, permuting the second
  operand if needed to match the first tensor's indices and spaces.
- `q * number`, `number * q`, `q / number`, `-q`: scale RMT data only.
- `q + number`, `number + q`, `q - number`, `number - q`: scalar shifts for
  rank-2 operator-like tensors using an identity tensor on the same space.
- `copy(q)`: deep copy.
- `q[i]`, `q[selector]`, `TLArray(q, selector)`: select rows by index, range,
  integer vector, boolean mask, or `:`.
- `getsub(q, selector)`: row-subset shorthand.
- `getsub(q, leg_or_legs, pred; preserve_space=false)`: keep sectors selected
  by a predicate on sector labels, optionally slicing RMT indices inside a
  sector.
- `getsub(q, pred; selectors..., preserve_space=false)`: select legs by index
  metadata and then apply predicate-based subsetting.
- `addSingleton(q; nlegs=1, itag="", plev=0, lock=0, dir='+')`: append trivial
  singleton legs.
- `addSingleton(q, legs; itag="", plev=0, lock=0, dir='+')`: insert trivial
  singleton legs at specified output positions.
- `deleteSingleton(q; dir=nothing, itag=nothing, plev=nothing)`: delete matching
  singleton trivial legs, or all singleton legs with no selectors.
- `deleteSingleton(q, leg_or_legs)`: delete explicitly selected singleton legs.
- `oplus(qs, dimensions)` / `oplus(q1, q2, dimensions)`: direct sum a vector or
  pair of tensors along one or more physical legs.
- `oplus(qs; selectors...)` / `oplus(q1, q2; selectors...)`: choose direct-sum
  legs using index selectors.
- `oplus(mat, (row_dims, col_dims))`: matrix-style block direct sum, filling
  `nothing`, `missing`, or unassigned entries with inferred zero tensors.
- `complete_oplus_matrix(mat, dimensions)`: validate a matrix direct-sum input
  and return the filled matrix before taking the direct sum.
- `q1 ⊗ q2` / `kron(q1, q2)`: tensor product implemented by adding singleton
  legs and contracting them.

### Decompositions

- `svd_leg(arr, leg; cutoff=1e-12, maxdim=nothing)`: SVD an ordinary array by
  isolating one leg and merging the rest. Returns `(U, SV, S)`.
- `svd_leg(t::LurTensor, leg; cutoff=1e-12, maxdim=nothing)`: same operation for
  `LurTensor`, returning wrapped `U` and `SV`.
- `svd(q, left_legs, left_tag="svdL", right_tag="svdR"; cutoff=1e-12,
  Nkeep=nothing)`: symmetry-adapted CGTSVD of a `TLArray`, returning
  `(U, S, Vd)`.
- `svd(q, left_tag="svdL", right_tag="svdR"; selectors..., cutoff=1e-12,
  Nkeep=nothing)`: choose SVD left legs using index selectors.
- `svd_cgtsvd(q, args...; kwargs...)`: exported alias for `svd(q, ...)`.
- `eigen(q, eig_tag="eig"; hermitian=false)`: eigendecompose a rank-2
  `TLArray`. Auto-detects Hermitian input unless forced.
- `discard_eigen(result, Nkeep, tol=0.1, kept_tag="eigK",
  discarded_tag="eigD"; hermitian=isnothing(result.V_inv))`: split an
  `EigenResult` into kept and discarded eigen-subspaces.

### Contraction Internals Exposed For Advanced Use

- `get_new_cgp(qlabels1, legdir1, cgp1, free1, legs1, qlabels2, legdir2, cgp2,
  free2, legs2)`: compute the canonical CGR permutation and leg direction for a
  contraction output sector. This is mainly useful when working on the
  contraction implementation.

### Module-Level Helpers

`Telum/src/Telum.jl` also defines, but does not export:

- `comm(A, B) = A * B - B * A`
- `a ⊗ b` for ordinary matrices/vectors, implemented as `kron(b, a)`.
