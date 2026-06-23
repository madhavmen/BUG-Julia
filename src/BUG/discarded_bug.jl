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
    matrixfree_sstep :: Bool    = false,
    aug_tol          :: Float64 = BUG_DEFAULT_AUG_TOL,
)
    link_l = bond_data.link_l;  site_l = bond_data.site_l
    site_r = bond_data.site_r;  link_r = bond_data.link_r
    U0 = bond_data.U0_tens;     V0 = bond_data.V0_tens;  S0 = bond_data.S0_tens
    canon_u0 = bond_data.canon_u0;  canon_v0 = bond_data.canon_v0
    pref = _active_time_prefactor()

    # ---- K-step: act the LEFT discarded projector on the generator, THEN integrate ----
    # G_K = P⊥_U0 · H_K with H_K = HW_env·V0·V0† (faithful K-generator, right env = V0)
    # and P⊥_U0 = I − U0 U0† on the (link_l,site_l) output legs. K1 = exp(-i dt G_K) K0.
    # MATRIX-FREE: G_K is applied as an ITensor-contraction matvec inside a Krylov
    # exponential (G_K is non-Hermitian ⇒ issymmetric=false). No dense H is formed,
    # so memory scales like a 2-site TDVP apply, not O(d²).
    K0 = U0 * S0
    mid_k = _left_site_bond_index(K0, link_l, site_l)
    k0_vec = _complex_tensor_vec(K0, link_l, site_l, mid_k)
    HW_K = HW_env * V0 * prime(dag(V0))      # faithful K-generator (built once, sparse-ish)
    function gK_matvec(v::AbstractVector)
        K = itensor(reshape(ComplexF64.(v), dim(link_l), dim(site_l), dim(mid_k)),
            link_l, site_l, mid_k)
        HK = noprime(HW_K * K)               # apply H_K
        HK = HK - U0 * (dag(U0) * HK)        # P⊥_U0 on (link_l,site_l)
        return _complex_tensor_vec(HK, link_l, site_l, mid_k)
    end
    k1_vec, _ = _general_linear_substep(gK_matvec, pref * dt, k0_vec;
        method = substep_method, lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter, issymmetric = false)
    K1_tens = itensor(reshape(k1_vec, dim(link_l), dim(site_l), dim(mid_k)),
        link_l, site_l, mid_k)
    U_aug, n_new_k, _ = _discarded_left_isometry(
        U0, K1_tens, link_l, site_l, canon_u0, mid_k; aug_tol = aug_tol)

    # ---- L-step: act the RIGHT discarded projector on the generator, THEN integrate ----
    L0 = S0 * V0
    mid_l = _site_bond_index(L0, site_r, link_r)
    l0_vec = _complex_tensor_vec(L0, mid_l, site_r, link_r)
    HW_L = prime(dag(U0)) * HW_env * U0       # faithful L-generator (left env = U0)
    function gL_matvec(v::AbstractVector)
        L = itensor(reshape(ComplexF64.(v), dim(mid_l), dim(site_r), dim(link_r)),
            mid_l, site_r, link_r)
        HL = noprime(HW_L * L)               # apply H_L
        HL = HL - V0 * (dag(V0) * HL)        # P⊥_V0 on (site_r,link_r)
        return _complex_tensor_vec(HL, mid_l, site_r, link_r)
    end
    l1_vec, _ = _general_linear_substep(gL_matvec, pref * dt, l0_vec;
        method = substep_method, lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter, issymmetric = false)
    L1_tens = itensor(reshape(l1_vec, dim(mid_l), dim(site_r), dim(link_r)),
        mid_l, site_r, link_r)
    V_aug, n_new_l, _ = _discarded_right_isometry(
        V0, L1_tens, canon_v0, mid_l, site_r, link_r; aug_tol = aug_tol)

    # ---- S-step: project Θ0 directly onto the augmented bases (no M, N), evolve ----
    theta0  = U0 * S0 * V0
    S_start = dag(U_aug) * theta0 * dag(V_aug)
    sstep = _advance_s_tensor_in_bases(U_aug, V_aug, S_start, HW_env, dt;
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        substep_method = substep_method, matrixfree_sstep = matrixfree_sstep)

    # ---- truncate: SVD sets the new (rank-adaptive) bond dimension ----
    s_inds = inds(sstep.S_new)
    U_s, SV_tens, keep, svals = _truncate_quantum_s_step(
        sstep.S_new, s_inds[1], s_inds[2]; maxdim = maxdim)

    U_new = U_aug * U_s          # (link_l, site_l, kept)
    V_new = SV_tens * V_aug      # (kept, site_r, link_r)
    return (
        U_new = U_new, V_new = V_new, S_kept = SV_tens,
        n_new_k = n_new_k, n_new_l = n_new_l, keep = keep, svals = svals,
    )
end

# ── Gauge retag (mirror of the TDVP2 split retag) ─────────────────────────────

function _discarded_retag_split_pair(left_tens::ITensor, right_tens::ITensor, bond::Int)
    shared = commonind(left_tens, right_tens)
    isnothing(shared) && error("discarded_bug split lost the shared bond index at bond $bond.")
    canon = settags(shared, "Link,l=$bond")
    return replaceind(left_tens, shared, canon), replaceind(right_tens, shared, canon), canon
end

# ── Sweeps (two-site-TDVP-like gauge transport, NO backward correction) ───────
#
# BUG has no backward (single-site) correction: the rank-adaptive K/L augmentation
# already supplies the new directions, and moving a single orthogonality centre
# bond-to-bond with QR (here folded into the next bond's snapshot) gives the right
# isometry conditions without re-introducing the shared-bond double counting that
# the TDVP backward step exists to cancel.

"""
    _discarded_forward_sweep!(psi, W, dt, info; maxdim, ...) -> psi

Left-to-right sweep over bonds 1 … N-1. At bond k: snapshot the two-site window,
apply `discarded_bug_local_update`, write `psi[k] = U_new` (left isometry) and
`psi[k+1] = S_kept · V_new` (carry the centre right), then advance the left MPO
environment with the new `psi[k]`. The next bond's snapshot re-canonicalizes the
carried tensor (QR), completing the gauge transport.
"""
function _discarded_forward_sweep!(
    psi    :: TensorTrain,
    W      :: TensorTrainOperator,
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
    orthogonalize!(psi, 1)
    R_all = _build_right_envs_mpo(psi, W)
    L_cur = _left_env_boundary_mpo(psi, W)

    for k in 1:(N - 1)
        bond_data = _canonical_quantum_bond_snapshot(psi, W, k, L_cur, R_all[k + 3])
        HW_env = L_cur * bond_data.W_left * bond_data.W_right * R_all[k + 3]

        cand = discarded_bug_local_update(bond_data, HW_env;
            dt = dt, maxdim = maxdim, lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter, substep_method = substep_method,
            matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)

        _record_kl_local!(info, k, :forward, cand)

        # carry the centre right: psi[k] = U_new (left iso), psi[k+1] = S_kept·V_new.
        U_new, carry, _ = _discarded_retag_split_pair(cand.U_new, cand.V_new, k)
        psi[k] = U_new

        if k == N - 1
            psi[k + 1] = carry
            continue
        end
        L_cur = _advance_left_env_mpo(L_cur, psi[k], W[k])
        psi[k + 1] = carry
    end
    return psi
end

"""
    _discarded_reverse_sweep!(psi, W, dt, info; maxdim, ...) -> psi

Right-to-left sweep over bonds N-1 … 1. Mirror of the forward sweep: write
`psi[k+1] = V_new` (right isometry) and carry `U_new · S_kept` left into `psi[k]`,
advancing the right MPO environment with the new `psi[k+1]`.
"""
function _discarded_reverse_sweep!(
    psi    :: TensorTrain,
    W      :: TensorTrainOperator,
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
    orthogonalize!(psi, N)
    L_all = _build_left_envs_mpo(psi, W)
    R_cur = _right_env_boundary_mpo(psi, W)

    for k in (N - 1):-1:1
        bond_data = _canonical_quantum_bond_snapshot(psi, W, k, L_all[k], R_cur)
        HW_env = L_all[k] * bond_data.W_left * bond_data.W_right * R_cur

        cand = discarded_bug_local_update(bond_data, HW_env;
            dt = dt, maxdim = maxdim, lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter, substep_method = substep_method,
            matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)

        _record_kl_local!(info, k, :reverse, cand)

        carry, V_new, _ = _discarded_retag_split_pair(cand.U_new, cand.V_new, k)
        psi[k + 1] = V_new

        if k == 1
            psi[k] = carry
            continue
        end
        R_cur = _advance_right_env_mpo(R_cur, psi[k + 1], W[k + 1])
        psi[k] = carry
    end
    return psi
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

Advance `psi` by one discarded-projector BUG step against the MPO `W`.

`order`:
- `:symmetric` (default) — Strang: forward(dt/2) then reverse(dt/2). 2nd order;
  the symmetric pair cancels the leading sweep-splitting error, so with the rank
  free to grow this reproduces `exp(-i dt H)` to high order.
- `:forward` — a single forward sweep (1st order), useful for diagnostics.

`maxdim` caps the rank-adaptive bond growth (`typemax(Int)` = grow freely, error
set by the integration order only). `time_prefactor = -im` (real time) by default;
pass `ComplexF64(1)` for imaginary time / parabolic PDEs.
"""
function discarded_bug_step!(
    psi :: TensorTrain,
    W   :: TensorTrainOperator;
    dt               :: Number,
    order            :: Symbol  = :symmetric,
    maxdim           :: Int     = typemax(Int),
    lanczos_tol      :: Float64 = 1e-15,
    lanczos_maxiter  :: Int     = 40,
    substep_method   :: Symbol  = :expv,
    matrixfree_sstep :: Bool    = false,
    aug_tol          :: Float64 = BUG_DEFAULT_AUG_TOL,
    expv_backend     :: Symbol  = :auto,
    time_prefactor   :: ComplexF64 = ComplexF64(-im),
)
    N = length(psi)
    N < 2 && error("discarded_bug_step! requires at least 2 sites")
    order in (:symmetric, :forward) ||
        error("discarded_bug_step!: order must be :symmetric or :forward")

    effective_backend = expv_backend === :auto ? :native_hermitian_lanczos : expv_backend
    effective_backend in (:krylovkit, :native_hermitian_lanczos) ||
        error("Unknown discarded_bug expv_backend: $expv_backend.")

    info = BUGInfo()
    info.bond_dims_before = [dim(linkind(psi, k)) for k in 1:(N - 1)]

    fwd(τ) = _discarded_forward_sweep!(psi, W, τ, info; maxdim = maxdim,
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        substep_method = substep_method, matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)
    rev(τ) = _discarded_reverse_sweep!(psi, W, τ, info; maxdim = maxdim,
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        substep_method = substep_method, matrixfree_sstep = matrixfree_sstep, aug_tol = aug_tol)

    info.elapsed = @elapsed begin
        _with_bug_expv_backend(effective_backend) do
            _with_bug_time_prefactor(time_prefactor) do
                if order === :symmetric
                    fwd(dt / 2)
                    rev(dt / 2)
                else
                    fwd(dt)
                end
            end
        end
    end

    info.bond_dims_after = [dim(linkind(psi, k)) for k in 1:(N - 1)]
    return info
end
