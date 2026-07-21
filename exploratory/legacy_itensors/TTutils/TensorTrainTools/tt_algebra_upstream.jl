## tt_algebra.jl
#
# Custom tensor-train algebraic operations defined in this package.
# These functions extend TensorTrainAlgebra with operations specific to the
# dynamical low-rank integrators implemented here.
#
# Exported:
#   tt_direct_sum(¤ê, ¤å) -> TensorTrain
#   tt_scaled_copy(¤ê, ╬▒) -> TensorTrain

# ÔöÇÔöÇ Direct sum (block-diagonal TT construction) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

"""
    tt_direct_sum(¤ê::TensorTrain, ¤å::TensorTrain) -> TensorTrain

Compute the **exact direct sum** (block-diagonal core construction) of two
tensor trains `¤ê` and `¤å` that share the same site (physical) indices.

## Mathematical definition

Let `¤ê = A┬╣ A┬▓ Ôï» Aß┤║` and `¤å = B┬╣ B┬▓ Ôï» Bß┤║` be TTs with bond dimensions
`r_¤ê` and `r_¤å` respectively.  The direct sum `╬À = tt_direct_sum(¤ê, ¤å)` is the
TT whose cores are:

| Site | Core shape | Construction |
|------|------------|--------------|
| k = 1 | `(1, d, r_¤ê + r_¤å)` | `╬À[1, :, 1:r_¤ê] = ¤ê[1, :, :]` ; `╬À[1, :, r_¤ê+1:end] = ¤å[1, :, :]` |
| 1 < k < N | `(r_¤ê_l + r_¤å_l, d, r_¤ê_r + r_¤å_r)` | block diagonal: upper-left = ¤ê core, lower-right = ¤å core |
| k = N | `(r_¤ê + r_¤å, d, 1)` | `╬À[1:r_¤ê, :, 1] = ¤ê[:, :, 1]` ; `╬À[r_¤ê+1:end, :, 1] = ¤å[:, :, 1]` |

**Consequence:** `vector(tt_direct_sum(¤ê, ¤å)) == vector(¤ê) + vector(¤å)` exactly,
with bond dimensions `r_╬À[k] = r_¤ê[k] + r_¤å[k]`.

## Connection to BUG augmentation

In the matrix-valued rank-r dynamical low-rank integrator (CerutiÔÇôLubich 2022)
the K-step augmentation reads

    K_aug = [KÔéü | X]     (column concatenation of KÔéü = K + dt┬ÀRhsK  with the old basis X)

The TT direct sum plays the same role site-by-site:

    ╬À_aug = tt_direct_sum(tt_direct_sum(¤ê, dt┬ÀF), ¤ê_old)
          Ôåö  [KÔéü | X]

## Prerequisites

- `¤ê` and `¤å` **must share the same site (physical) index objects** (same ITensors
  `Index` identities, not just matching dimensions).  This is always true when
  `¤å` is derived from `¤ê` (e.g. a copy or the output of `rhs_fn(¤ê)`).
- Link indices of `¤å` are replaced internally with fresh indices
  (via `replacelinks`) to avoid conflicts with `¤ê`'s bond indices.

## Example
```julia
¤ê   = random_tt(sites, 4)            # bond dim 4
F   = rhs_fn(¤ê)                       # bond dim Ôëñ 4
╬À   = tt_direct_sum(¤ê, F)             # bond dim Ôëñ 8
@assert norm(vector(╬À) - (vector(¤ê) + vector(F))) < 1e-12
```
"""
function tt_direct_sum(¤ê::TensorTrain, ¤å::TensorTrain)
    N     = length(¤ê)
    sites = siteinds(¤ê)

    # Give ¤å fresh link indices to avoid index conflicts with ¤ê.
    ¤å_r = replacelinks(copy(¤å))

    l¤ê = linkinds(¤ê)    # [lÔéÇ, lÔéü, ÔÇª, l_N]   (N+1 elements; lÔéÇ,l_N have dim 1)
    l¤å = linkinds(¤å_r)

    # ¤å may have been produced by a different construction path and therefore
    # carries its OWN site Index objects (same tags/dims but different IDs).
    # Use siteinds(¤å_r) ÔÇô not sites = siteinds(¤ê) ÔÇô when extracting its arrays.
    sites_¤å = siteinds(¤å_r)

    # Extract rank-3 arrays in fixed (left_link, site, right_link) ordering.
    A¤ê = [array(¤ê[k],   l¤ê[k], sites[k],   l¤ê[k+1]) for k in 1:N]
    A¤å = [array(¤å_r[k], l¤å[k], sites_¤å[k], l¤å[k+1]) for k in 1:N]

    # Infer floating-point type.
    T = promote_type(eltype(A¤ê[1]), eltype(A¤å[1]))

    arrays = Vector{Array{T,3}}(undef, N)

    for k in 1:N
        a¤ê = A¤ê[k]
        a¤å = A¤å[k]
        r¤êl, dk, r¤êr = size(a¤ê)
        r¤ål, _,  r¤år = size(a¤å)

        if k == 1
            # k=1: both left bonds are dim-1 ÔÇö concatenate along the RIGHT bond.
            # Result shape: (1, d, r¤êr + r¤år)
            A = zeros(T, 1, dk, r¤êr + r¤år)
            A[1, :, 1:r¤êr]     = a¤ê[1, :, :]
            A[1, :, r¤êr+1:end] = a¤å[1, :, :]

        elseif k == N
            # k=N: both right bonds are dim-1 ÔÇö stack along the LEFT bond.
            # Result shape: (r¤êl + r¤ål, d, 1)
            A = zeros(T, r¤êl + r¤ål, dk, 1)
            A[1:r¤êl,     :, 1] = a¤ê[:, :, 1]
            A[r¤êl+1:end, :, 1] = a¤å[:, :, 1]

        else
            # Interior sites: block-diagonal (upper-left = ¤ê, lower-right = ¤å).
            # Result shape: (r¤êl + r¤ål, d, r¤êr + r¤år)
            A = zeros(T, r¤êl + r¤ål, dk, r¤êr + r¤år)
            A[1:r¤êl,     :, 1:r¤êr]      = a¤ê
            A[r¤êl+1:end, :, r¤êr+1:end]  = a¤å
        end

        arrays[k] = A
    end

    return TensorTrain(sites, arrays)
end

# ÔöÇÔöÇ Scaled copy ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

"""
    tt_scaled_copy(¤ê::TensorTrain, ╬▒) -> TensorTrain

Return a deep copy of `¤ê` scaled by the scalar `╬▒`.

Equivalent to `deepcopy(¤ê); rescale!(copy, ╬▒)` but returns the result
rather than mutating in-place.

Used in the Heun RK2 BUG step to form weighted averages without modifying
the intermediate TTs:

    K_heun = tt_direct_sum(tt_scaled_copy(¤ê, 0.5), tt_scaled_copy(KÔéé, 0.5))
"""
function tt_scaled_copy(¤ê::TensorTrain, ╬▒)
    ¤å = deepcopy(¤ê)
    gauge_centre = length(¤å)
    ¤å[gauge_centre] = ╬▒ * ¤å[gauge_centre]
    return ¤å
end

# ÔöÇÔöÇ Controlled Bond Expansion (CBE) augmentation ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

"""
    tt_cbe_augment(¤ê::TensorTrain, F_scaled::TensorTrain, p::Int;
                   rsvd::Bool=false, rsvd_oversampling::Int=10) -> TensorTrain

**Controlled Bond Expansion (CBE)** augmentation for the BUG integrator.

Instead of the full `tt_direct_sum(¤ê, F_scaled)` which always doubles the bond
dimension from `r` to `2r`, CBE enriches ¤ê with only the top-`p` directions
of `F_scaled`.  The result has bond dimension at most `r + p`.

## Algorithm

Compresses `F_scaled` to at most `p` bond dimensions, then forms a direct sum
with `¤ê`:

    F_small = svd_compress(F_scaled; maxdim=p)     # bond Ôëñ p  (or rSVD if rsvd=true)
    ╬À       = tt_direct_sum(¤ê, F_small)            # bond Ôëñ r_¤ê + p

The result `╬À` has `vector(╬À) = vector(¤ê) + vector(F_small)` exactly, with bond
dimension at most `r_¤ê + p` instead of the `r_¤ê + r_F` that a plain
`tt_direct_sum(¤ê, F_scaled)` would give.  The subsequent `svd_compress!` in the
BUG step then truncates back to `r_max`.

This is equivalent to the "Naive CBE" strategy: select the `p` most energetic
mode directions of F before augmenting.  The rSVD variant uses a randomized
range finder for the pre-truncation, reducing cost from `O(r_F┬│)` to
`O(r_F┬▓ ┬À p)` per bond.

## Arguments
- `¤ê`              : current TensorTrain (bond dim r)
- `F_scaled`       : `dt ┬À F(¤ê)` as a TensorTrain (same sites, bond dim r_F)
- `p`              : maximum bond dimension kept from F (ÔëÑ 1); intermediate
                     bond after augmentation is Ôëñ r + p
- `rsvd`           : if true, use randomized SVD for the F pre-truncation
- `rsvd_oversampling`: oversampling parameter for rSVD (default 10)

## Returns
A new `TensorTrain` with bond dims `Ôëñ r + p`.

## References
- Gleis, Haegeman & Pollmann (2022). *Controlled bond expansion TDVP.*
  arXiv:2207.14712.
- Li, Qiaoyi (2023). FiniteMPS.jl, `src/Algorithm/CBE/NaiveCBE.jl`.
"""
function tt_cbe_augment(
    ¤ê              :: TensorTrain,
    F_scaled       :: TensorTrain,
    p              :: Int;
    rsvd           :: Bool = false,
    rsvd_oversampling :: Int = 10,
)
    p > 0 || throw(ArgumentError("p must be ÔëÑ 1"))

    # Pre-truncate F to bond dim p ÔÇö the "controlled" part of CBE.
    # rsvd=true uses a randomized SVD sketch for this step (cheaper for large r_F).
    if rsvd
        F_small = svd_compress_rsvd(F_scaled; maxdim = p, cutoff = 0.0,
                                    oversampling = rsvd_oversampling)
    else
        F_small = svd_compress(F_scaled; maxdim = p, cutoff = 0.0)
    end

    # Augment: ¤ê gets p extra bond channels from the top-p modes of F.
    # vector(result) = vector(¤ê) + vector(F_small)
    # bond dim of result = bond(¤ê) + bond(F_small) Ôëñ r + p
    return tt_direct_sum(¤ê, F_small)
end

# ÔöÇÔöÇ Exports ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
export tt_direct_sum
export tt_scaled_copy
export tt_cbe_augment
