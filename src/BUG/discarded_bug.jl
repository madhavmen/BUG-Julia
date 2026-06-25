# discarded_bug.jl
#
# Discarded-projector BUG: a rank-adaptive two-site integrator derived from the
# Ceruti–Kusch–Lubich basis-update-and-Galerkin (BUG) scheme, but with the basis
# growth driven by the DISCARDED (orthogonal-complement) projectors and WITHOUT
# building the augmented overlap matrices M, N.
#
# Local 2-site update at a bond (state Θ0 = U0 · S0 · V0)
# ------------------------------------------------------
# K-step — grow the LEFT bond space:
#   1. K0 = U0 · S0 ; integrate K̇ = -i H_K K under the right-projected effective
#      Hamiltonian H_K (right environment = the OLD right isometry V0) → K1.
#   2. Act the LEFT DISCARDED projector  P⊥_U0 = I_(link_l,site_l) − U0 U0†  on the
#      output (link_l, site_l) legs of K1 → only the part of K1 OUTSIDE span(U0).
#   3. QR that complement piece → orthonormal new columns Qk (⊥ U0 by construction).
#   4. Augment by DIRECT SUM:  Û = [ U0 | Qk ].   (No overlap matrix M is formed.)
#
# L-step — grow the RIGHT bond space (mirror of the K-step):
#   1. L0 = S0 · V0 ; integrate under the left-projected H_L (left env = OLD U0) → L1.
#   2. Act the RIGHT DISCARDED projector  P⊥_V0 = I_(site_r,link_r) − V0† V0  on the
#      output (site_r, link_r) legs of L1.
#   3. QR (row space) → orthonormal new rows Ql (⊥ V0).
#   4. Direct sum:  V̂ = [ V0 ; Ql ].   (No overlap matrix N is formed.)
#
# S-step:
#   - DO NOT transport the core via M, N. Project the CURRENT two-site state
#     directly onto the augmented bases:  Ŝ0 = Û† Θ0 V̂† .
#     (This equals M S0 N because Û ⊇ U0 and V̂ ⊇ V0 exactly, but is built without
#     ever forming M or N.)
#   - Integrate Ŝ̇ = -i (Û† H V̂-projected) Ŝ in the augmented basis → Ŝ1.
#   - SVD Ŝ1 and keep up to `maxdim` singular values → the new, possibly larger,
#     bond rank. The K/L direct sums GROW the frame; this SVD sets the kept rank.
#
# Correctness
# -----------
# span([U0 | P⊥_U0 K1]) = span([U0 | K1]) because K1 = U0 (U0† K1) + P⊥_U0 K1 and
# the first term already lies in span(U0). Hence the augmented SUBSPACES equal
# those of the faithful CKL BUG, so the Galerkin S-step yields the same evolution;
# the discarded projector only isolates the genuinely-new directions and lets the
# direct sum stay orthonormal without an overlap matrix. The scheme is therefore a
# faithful rank-adaptive integrator: with the rank free to grow and a symmetric
# (forward+backward) sweep, it reproduces exp(-i dt H) to machine precision.
#
# Sweeping (two-site-TDVP-like, gauge transported, no inverse)
# ------------------------------------------------------------
# A forward L→R sweep updates bonds 1,2,…,N-1. After the bond-b SVD we hold
# Û_b (left isometry, the new psi[b]) and (Ŝ1 V̂_b) on the right. To move the
# orthogonality centre to bond b+1 we contract the centre with the right isometry,
# QR the result to expose a new left isometry on site b+1, and carry the
# upper-triangular factor rightward — exactly the TDVP/DMRG gauge transport, done
# with QR (no Λ^{-1}). A symmetric step does L→R with dt/2 then R→L with dt/2.

# ── Discarded-projector left isometry (K-step) ────────────────────────────────

"""
    _discarded_left_isometry(U0_tens, K1_tens, link_l, site_l, canon_u0, mid_k)
        -> (U_aug_tens, n_new, aug_ind)

Grow the left frame by the LEFT discarded projector. `K1_tens` is the integrated
K tensor on `(link_l, site_l, mid_k)`. Returns the augmented left isometry
`Û = [U0 | Qk]` (direct sum), the number of new columns, and its right index.
"""
function _discarded_left_isometry(
    U0_tens  :: ITensor,
    K1_tens  :: ITensor,
    link_l   :: Index,
    site_l   :: Index,
    canon_u0 :: Index,
    mid_k    :: Index;
    aug_tol  :: Float64 = BUG_DEFAULT_AUG_TOL,
)
    dl = dim(link_l);  dsl = dim(site_l);  old_rank = dim(canon_u0);  d_mid_k = dim(mid_k)
    U0_mat = reshape(_complex_tensor_array(U0_tens, link_l, site_l, canon_u0), dl * dsl, old_rank)
    K1_mat = reshape(_complex_tensor_array(K1_tens, link_l, site_l, mid_k),    dl * dsl, d_mid_k)

    # Left discarded projector P⊥_U0 = I − U0 U0†, applied to the integrated K1.
    K1_perp = K1_mat - U0_mat * (U0_mat' * K1_mat)
    Qk = _filter_left_aug_columns(U0_mat, _qr_column_basis(K1_perp)[1]; aug_tol = aug_tol)

    if size(Qk, 2) == 0
        aug_ind = Index(old_rank, tags(canon_u0))
        U_aug = itensor(reshape(U0_mat, dl, dsl, old_rank), link_l, site_l, aug_ind)
        return U_aug, 0, aug_ind
    end
    # Direct sum (U0 first); a final QR drops any residual dependence and fixes a
    # canonical orientation of the new columns.
    U_aug_mat, aug_rank = _qr_column_basis(hcat(U0_mat, Qk))
    aug_ind = Index(aug_rank, tags(canon_u0))
    U_aug = itensor(reshape(U_aug_mat, dl, dsl, aug_rank), link_l, site_l, aug_ind)
    return U_aug, max(0, aug_rank - old_rank), aug_ind
end

# ── Discarded-projector right isometry (L-step) ───────────────────────────────

"""
    _discarded_right_isometry(V0_tens, L1_tens, canon_v0, mid_l, site_r, link_r)
        -> (V_aug_tens, n_new, aug_ind)

Mirror of `_discarded_left_isometry` on the right frame. `L1_tens` is the
integrated L tensor on `(mid_l, site_r, link_r)`. Returns `V̂ = [V0 ; Ql]`.
"""
function _discarded_right_isometry(
    V0_tens  :: ITensor,
    L1_tens  :: ITensor,
    canon_v0 :: Index,
    mid_l    :: Index,
    site_r   :: Index,
    link_r   :: Index;
    aug_tol  :: Float64 = BUG_DEFAULT_AUG_TOL,
)
    dsr = dim(site_r);  dr = dim(link_r);  old_rank = dim(canon_v0);  d_mid_l = dim(mid_l)
    V0_row = reshape(_complex_tensor_array(V0_tens, canon_v0, site_r, link_r), old_rank, dsr * dr)
    L1_row = reshape(_complex_tensor_array(L1_tens, mid_l, site_r, link_r),    d_mid_l, dsr * dr)

    # Right discarded projector P⊥_V0 = I − V0† V0 on (site_r, link_r), applied to L1.
    L1_perp = L1_row - (L1_row * V0_row') * V0_row
    Ql = _filter_right_aug_rows(V0_row, _qr_row_basis(L1_perp)[1]; aug_tol = aug_tol)

    if size(Ql, 1) == 0
        aug_ind = Index(old_rank, tags(canon_v0))
        V_aug = itensor(reshape(V0_row, old_rank, dsr, dr), aug_ind, site_r, link_r)
        return V_aug, 0, aug_ind
    end
    V_aug_mat, aug_rank = _qr_row_basis(vcat(V0_row, Ql))
    aug_ind = Index(aug_rank, tags(canon_v0))
    V_aug = itensor(reshape(V_aug_mat, aug_rank, dsr, dr), aug_ind, site_r, link_r)
    return V_aug, max(0, aug_rank - old_rank), aug_ind
end

# ── Local 2-site discarded-BUG update ─────────────────────────────────────────

"""
    discarded_bug_local_update(bond_data, HW_env; dt, maxdim, ...) -> NamedTuple

One discarded-projector BUG update on the two-site window described by
`bond_data` (fields `U0_tens, S0_tens, V0_tens`, the four indices, and the
canonical bond indices `canon_u0/canon_v0`). `HW_env` is the dressed two-site
effective Hamiltonian (`L_mpo · W_left · W_right · R_mpo`, or a per-bond gate
dressed with identities on the link legs).

The block-evolution form (a faithful port of the Python `block_local_update`):
the two-site block is evolved **once** under the two-site effective Hamiltonian,

    Θ1 = exp(pref·dt·HW_env) · Θ0       (Θ0 = U0·S0·V0, Hermitian ⇒ Lanczos),

and the augmented frames are read directly off `Θ1` — the left frame from its
`(link_l, site_l)` column space and the right frame from its `(site_r, link_r)`
row space — each DIRECT-SUMMED onto the old frame with the DISCARDED projector
(`Û = [U0 | colspace(Θ1)]`, `V̂ = [V0 ; rowspace(Θ1)]`; never an `M`/`N`
overlap matrix). The Galerkin core is the projection of the already-evolved
block onto the augmented frames, `Ŝ = Û† Θ1 V̂†` — there is NO separate K/L/S
sub-evolution; the time evolution is already in `Θ1`.

Why grow from the evolved block (and not a frozen-neighbour generator): acting
with `H` on the two-site window is what creates the new Schmidt direction (a
domain-wall interface block has Schmidt rank 2, so the bond MUST grow 1→2 in one
step). A K-step that froze the right subsystem at the old single-state frame `V0`
(projecting `H·Θ0` onto `V0 V0†`) would annihilate exactly that direction, since
the new content is orthogonal to `V0`. Reading the frames off the full `Θ1` keeps
the physical legs free, so the genuine entanglement growth survives.

Returns `(U_new, S_kept, V_new, n_new_k, n_new_l, keep, svals)` where
`Θ1 ≈ U_new · S_kept · V_new`, `U_new` is a left isometry on
`(link_l, site_l, kept)`, `V_new` a right isometry on `(kept, site_r, link_r)`,
and `kept ≤ maxdim` is the new (rank-adaptive) bond dimension.
"""
function discarded_bug_local_update(
    bond_data,
    HW_env :: ITensor;
    dt               :: Number,
    maxdim           :: Int     = typemax(Int),
    lanczos_tol      :: Float64 = 1e-15,
    lanczos_maxiter  :: Int     = 40,
    substep_method   :: Symbol  = :expv,
    matrixfree_sstep :: Bool    = false,   # unused: a single block evolution, kept for call-site compat
    aug_tol          :: Float64 = BUG_DEFAULT_AUG_TOL,
)
    _ = matrixfree_sstep
    link_l = bond_data.link_l;  site_l = bond_data.site_l
    site_r = bond_data.site_r;  link_r = bond_data.link_r
    U0 = bond_data.U0_tens;     V0 = bond_data.V0_tens;  S0 = bond_data.S0_tens
    canon_u0 = bond_data.canon_u0;  canon_v0 = bond_data.canon_v0
    pref = _active_time_prefactor()

    # ---- Evolve the two-site block ONCE under the two-site effective Hamiltonian ----
    # Θ1 = exp(pref·dt·HW_env)·Θ0 (Hermitian ⇒ native Hermitian Lanczos, issymmetric=true).
    # Matrix-free apply: noprime(HW_env·Θ) maps unprimed→unprimed on (link_l,site_l,site_r,link_r),
    # exactly the 2-site TDVP apply — no dense H is formed.
    theta0 = U0 * S0 * V0
    th_inds = (link_l, site_l, site_r, link_r)
    th_dims = dim.(th_inds)
    theta0_vec = _complex_tensor_vec(theta0, th_inds...)
    function h_matvec(v::AbstractVector)
        th = itensor(reshape(ComplexF64.(v), th_dims...), th_inds...)
        return _complex_tensor_vec(noprime(HW_env * th), th_inds...)
    end
    theta1_vec, _ = _general_linear_substep(h_matvec, pref * dt, theta0_vec;
        method = substep_method, lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter, issymmetric = true)
    theta1 = itensor(reshape(theta1_vec, th_dims...), th_inds...)

    # ---- Grow the LEFT frame from Θ1's (link_l, site_l) column space (discarded sum) ----
    # QR of Θ1 over (link_l, site_l) exposes its column-space isometry K1 on (link_l,site_l,mid_k);
    # _discarded_left_isometry direct-sums [U0 | colspace(K1)] (P⊥_U0 isolates the genuinely-new
    # part; the leading U0 keeps the old frame exactly inside). No M overlap matrix.
    K1_tens, mid_k = _column_space_isometry(theta1, link_l, site_l, canon_u0)
    U_aug, n_new_k, _ = _discarded_left_isometry(
        U0, K1_tens, link_l, site_l, canon_u0, mid_k; aug_tol = aug_tol)

    # ---- Grow the RIGHT frame from Θ1's (site_r, link_r) row space (discarded sum) ----
    L1_tens, mid_l = _row_space_isometry(theta1, site_r, link_r, canon_v0)
    V_aug, n_new_l, _ = _discarded_right_isometry(
        V0, L1_tens, canon_v0, mid_l, site_r, link_r; aug_tol = aug_tol)

    # ---- Galerkin core: project the already-evolved block onto the augmented frames ----
    # Ŝ = Û† Θ1 V̂†  (the DISCARDED projector — no separate S-evolution, no M/N transport).
    S_new = dag(U_aug) * theta1 * dag(V_aug)

    # ---- truncate: SVD sets the new (rank-adaptive) bond dimension ----
    s_inds = inds(S_new)
    U_s, SV_tens, keep, svals = _truncate_quantum_s_step(
        S_new, s_inds[1], s_inds[2]; maxdim = maxdim)

    U_new = U_aug * U_s          # (link_l, site_l, kept)
    V_new = SV_tens * V_aug      # (kept, site_r, link_r)
    return (
        U_new = U_new, V_new = V_new, S_kept = SV_tens,
        n_new_k = n_new_k, n_new_l = n_new_l, keep = keep, svals = svals,
    )
end

# ── Column/row-space isometries of the evolved block (feed the discarded augmenters) ──

"""
    _column_space_isometry(theta1, link_l, site_l, canon_u0) -> (K1_tens, mid_k)

QR of `theta1` over `(link_l, site_l)`: returns the column-space isometry `K1`
on `(link_l, site_l, mid_k)` (its right index tagged like `canon_u0`) that spans
`colspace(theta1 | link_l, site_l)`. Mirrors the Python
`decomp(theta1, axes=[0,1], mode='QR', itag=u0.itags[2])`.
"""
function _column_space_isometry(theta1::ITensor, link_l::Index, site_l::Index, canon_u0::Index)
    dl = dim(link_l);  dsl = dim(site_l)
    other = uniqueinds(theta1, (link_l, site_l))
    rest  = prod(dim.(other))
    M     = reshape(_complex_tensor_array(theta1, link_l, site_l, other...), dl * dsl, rest)
    Q, r  = _qr_column_basis(M)
    mid_k = Index(r, tags(canon_u0))
    return itensor(reshape(Q, dl, dsl, r), link_l, site_l, mid_k), mid_k
end

"""
    _row_space_isometry(theta1, site_r, link_r, canon_v0) -> (L1_tens, mid_l)

QR of `theta1` over `(site_r, link_r)`: returns the row-space isometry `L1` on
`(mid_l, site_r, link_r)` (its left index tagged like `canon_v0`) that spans
`rowspace(theta1 | site_r, link_r)`. Mirrors the Python
`decomp(theta1, axes=[2,3], mode='QR')` (reordered to the right-isometry layout).
"""
function _row_space_isometry(theta1::ITensor, site_r::Index, link_r::Index, canon_v0::Index)
    dsr = dim(site_r);  dr = dim(link_r)
    other = uniqueinds(theta1, (site_r, link_r))
    rest  = prod(dim.(other))
    # rows indexed by (site_r, link_r), columns by the rest → row space lives on (site_r,link_r).
    M     = reshape(_complex_tensor_array(theta1, other..., site_r, link_r), rest, dsr * dr)
    Qrow, r = _qr_row_basis(M)
    mid_l = Index(r, tags(canon_v0))
    return itensor(reshape(Qrow, r, dsr, dr), mid_l, site_r, link_r), mid_l
end

# ── Gauge retag (mirror of the TDVP2 split retag) ─────────────────────────────

function _discarded_retag_split_pair(left_tens::ITensor, right_tens::ITensor, bond::Int)
    shared = commonind(left_tens, right_tens)
    isnothing(shared) && error("discarded_bug split lost the shared bond index at bond $bond.")
    canon = settags(shared, "Link,l=$bond")
    return replaceind(left_tens, shared, canon), replaceind(right_tens, shared, canon), canon
end

# ── Recursive-bisection step (the MPS realisation of the Lubich TTN-BUG) ──────
#
# This is the MPS specialisation of the Lubich tree-tensor-network BUG: the
# reference builds a balanced binary tree by RECURSIVE BISECTION of the 1D modes
# (leaves = physical sites), so the MPS realisation recursively bisects the chain
# and performs ONE two-site node update at each bisection bond. There is NO
# forward/backward Strang sweep and NO backward (negative-time) substep — BUG is
# inverse-free by design; every bond is a tree node, so the bond dimension grows
# along the whole chain (the full light cone). A naive node-order-reversed pass
# does NOT lift the order, so a single bisection is first order in dt (a 2nd-order
# symmetric composition is left to future work).

"""
    _update_bond!(psi, W, mid, dt, info; maxdim, ...) -> kept

Apply one Lubich node update at bond `(mid, mid+1)`: move the orthogonality
centre to `mid` (so the snapshot is canonical there), build the MPO environments
bracketing the two-site window from the boundaries, run the discarded
`discarded_bug_local_update`, and write the two new cores back (centre at
`mid+1`). Returns the kept bond dimension. Mirrors the Python `_update_bond`.
"""
function _update_bond!(
    psi    :: TensorTrain,
    W      :: TensorTrainOperator,
    mid    :: Int,
    dt     :: Number,
    info   :: Union{BUGInfo,Nothing};
    maxdim           :: Int,
    lanczos_tol      :: Float64,
    lanczos_maxiter  :: Int,
    substep_method   :: Symbol,
    matrixfree_sstep :: Bool,
    aug_tol          :: Float64,
)
    N = length(psi)
    orthogonalize!(psi, mid)

    # Build the MPO environments bracketing the two-site window (mid, mid+1) from the
    # boundaries — rebuilt per node like the Python `_update_bond` (the bisection visits
    # bonds out of order, so there is no single running env to advance).
    L_cur = _left_env_boundary_mpo(psi, W)
    for k in 1:(mid - 1)
        L_cur = _advance_left_env_mpo(L_cur, psi[k], W[k])
    end
    R_cur = _right_env_boundary_mpo(psi, W)
    for k in N:-1:(mid + 2)
        R_cur = _advance_right_env_mpo(R_cur, psi[k], W[k])
    end

    bond_data = _canonical_quantum_bond_snapshot(psi, W, mid, L_cur, R_cur)
    HW_env = L_cur * bond_data.W_left * bond_data.W_right * R_cur

    cand = discarded_bug_local_update(bond_data, HW_env;
        dt = dt, maxdim = maxdim, lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter, substep_method = substep_method,
        matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)

    _record_kl_local!(info, mid, :bisection, cand)

    # Write the two new cores; the centre is left at mid+1 (psi[mid] left isometry,
    # psi[mid+1] = S_kept · V_new carries the singular values).
    U_new, carry, _ = _discarded_retag_split_pair(cand.U_new, cand.V_new, mid)
    psi[mid] = U_new
    psi[mid + 1] = carry
    return cand.keep
end

"""
    _discarded_bisect!(psi, W, dt, info, lo, hi; maxdim, ...) -> kept

Recursive bisection of the sub-chain on sites `[lo, hi]` (its bonds are
`lo … hi-1`). Updates the bisection-bond node `mid = (lo+hi) ÷ 2`, then recurses
into the left half `[lo, mid]` and the right half `[mid+1, hi]` — the MPS
realisation of the Lubich balanced-binary-tree `Step` (each bond is a tree node).
Returns the maximum kept bond dimension in the subtree. Mirrors the Python
`_bisect`.
"""
function _discarded_bisect!(
    psi    :: TensorTrain,
    W      :: TensorTrainOperator,
    dt     :: Number,
    info   :: Union{BUGInfo,Nothing},
    lo     :: Int,
    hi     :: Int;
    maxdim           :: Int,
    lanczos_tol      :: Float64,
    lanczos_maxiter  :: Int,
    substep_method   :: Symbol,
    matrixfree_sstep :: Bool,
    aug_tol          :: Float64,
)
    hi - lo < 1 && return 1
    mid = (lo + hi) ÷ 2
    kept = _update_bond!(psi, W, mid, dt, info; maxdim = maxdim,
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        substep_method = substep_method, matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)
    kept_l = _discarded_bisect!(psi, W, dt, info, lo, mid; maxdim = maxdim,
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        substep_method = substep_method, matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)
    kept_r = _discarded_bisect!(psi, W, dt, info, mid + 1, hi; maxdim = maxdim,
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        substep_method = substep_method, matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)
    return max(kept, kept_l, kept_r)
end

function _record_kl_local!(info::BUGInfo, bond::Int, sweep::Symbol, cand)
    push!(info.aug_sizes_k, cand.n_new_k)
    push!(info.aug_sizes_l, cand.n_new_l)
    _record_s_step_rank!(info, sweep, bond, cand.keep, cand.svals)
    return nothing
end
_record_kl_local!(::Nothing, ::Int, ::Symbol, cand) = nothing

# ── Public API ────────────────────────────────────────────────────────────────

"""
    discarded_bug_step!(psi, W; dt, kwargs...) -> BUGInfo

Advance `psi` by one discarded-projector BUG step against the MPO `W`, via
RECURSIVE BISECTION of the chain — the MPS realisation of the Lubich
tree-tensor-network BUG `Step` on a balanced binary tree (the reference builds
the tree by recursive bisection of the 1D modes). Canonicalises to `center == 1`,
then recursively bisects (`_discarded_bisect!`): at each bisection bond it applies
one discarded two-site `discarded_bug_local_update` (the basis grows from the
evolved two-site block with the DISCARDED projector — no `M`/`N` overlap matrix),
then recurses into the two halves. Because every bond is a tree node, every bond's
basis is updated, so the bond dimension grows along the whole chain (the full
light cone). The step is first order in `dt`; there is no backward (negative-time)
substep — BUG is inverse-free by design, and the validated property is the rank
growth / light-cone spread.

`maxdim` caps the rank-adaptive bond growth (`typemax(Int)` = grow freely).
`time_prefactor = -im` (real time) by default; pass `ComplexF64(1)` for imaginary
time / parabolic PDEs. The `order`/`matrixfree_sstep` keywords are retained for
call-site compatibility but no longer select a scheme (a single bisection is the
only scheme; there is no forward/reverse Strang variant).
"""
function discarded_bug_step!(
    psi :: TensorTrain,
    W   :: TensorTrainOperator;
    dt               :: Number,
    order            :: Symbol  = :bisection,
    maxdim           :: Int     = typemax(Int),
    lanczos_tol      :: Float64 = 1e-15,
    lanczos_maxiter  :: Int     = 40,
    substep_method   :: Symbol  = :expv,
    matrixfree_sstep :: Bool    = false,
    aug_tol          :: Float64 = BUG_DEFAULT_AUG_TOL,
    expv_backend     :: Symbol  = :auto,
    time_prefactor   :: ComplexF64 = ComplexF64(-im),
)
    _ = order
    N = length(psi)
    N < 2 && error("discarded_bug_step! requires at least 2 sites")

    effective_backend = expv_backend === :auto ? :native_hermitian_lanczos : expv_backend
    effective_backend in (:krylovkit, :native_hermitian_lanczos) ||
        error("Unknown discarded_bug expv_backend: $expv_backend.")

    info = BUGInfo()
    info.bond_dims_before = [dim(linkind(psi, k)) for k in 1:(N - 1)]

    info.elapsed = @elapsed begin
        _with_bug_expv_backend(effective_backend) do
            _with_bug_time_prefactor(time_prefactor) do
                orthogonalize!(psi, 1)
                _discarded_bisect!(psi, W, dt, info, 1, N; maxdim = maxdim,
                    lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
                    substep_method = substep_method, matrixfree_sstep = matrixfree_sstep,
                    aug_tol = aug_tol)
                # Full re-gauge after the bisection so every bond's charge-sector order is
                # consistent (the augmenter can reorder a grown bond's sectors).
                orthogonalize!(psi, 1)
            end
        end
    end

    info.bond_dims_after = [dim(linkind(psi, k)) for k in 1:(N - 1)]
    return info
end
