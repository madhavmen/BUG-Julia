# bug_kls.jl
#
# Faithful KLS local bond candidate:
#   - Augmentation helpers (_pick_left/right_update, _augmented_*_isometry_from_k/l)
#   - Projected Hamiltonian helpers (2-site, left/right projected, bond-center projected)
#   - S-step advance (_advance_s_tensor_in_bases)
#   - S-step rank selection and truncation
#   - Step rejection
#   - Faithful forward and reverse KLS candidates
#   - _run_independent_kl_pair (threaded K+L pair)
#
# ONLY faithful KLS is exposed. 

# ── Augmentation ──────────────────────────────────────────────────────────────

# NOTE: the augmentation tolerance has been fully removed (per request). New Krylov directions are
# admitted with NO heuristic discard threshold; the augmenting QR (`_qr_column_basis`/`_qr_row_basis`)
# still drops exactly-dependent columns at its machine-precision rank tolerance, so the basis stays a
# well-defined orthonormal isometry. Final rank control happens only at the post-S-step SVD truncation
# to maxdim (Ceruti–Kusch–Lubich step 3).

function _pick_left_update(U0_mat, K1_mat;
        augment::Bool = true,
        max_rank::Union{Nothing,Int} = nothing,
    )
    rm    = size(U0_mat, 2)
    T_el  = promote_type(eltype(U0_mat), eltype(K1_mat))
    # No-aug branch: skip the [U0 | K1] concatenation entirely. The basis
    # update reduces to "use U0 unchanged", and S_transport = I * S0 = S0
    # (the overlap matrix is the identity). This is the fixed-rank parallel
    # basis-update without rank adaptation (Ceruti–Lubich).
    if !augment
        return U0_mat, 0, _identity_overlap_matrix(T_el, rm)
    end
    Qk, rk = _qr_column_basis(K1_mat)
    rk <= 0 && return U0_mat, 0, _identity_overlap_matrix(T_el, rm)
    size(Qk, 2) == 0 && return U0_mat, 0, _identity_overlap_matrix(T_el, rm)
    # Order matters: place U0 BEFORE Qk in the QR so the first `rm` columns of
    # Q_aug equal U0 exactly (rather than the K1-derived QR pivot, which depends
    # on sign(dt)). The new directions enter as columns rm+1 … aug_rank.
    Q_aug_mat, aug_rank = _qr_column_basis(hcat(U0_mat, Qk))
    aug_rank <= 0 && return U0_mat, 0, _identity_overlap_matrix(T_el, rm)
    # No padding: return the augmented basis at its true rank. Truncation back
    # to old_rank or any user cap happens after the post-S-step SVD, never here
    # (Ceruti–Kusch–Lubich, arXiv:2304.05660, step 3). `max_rank` is honored as
    # a hard ceiling on the augmented rank for memory safety only.
    if !isnothing(max_rank) && aug_rank > max_rank
        Q_aug_mat = Matrix(Q_aug_mat[:, 1:max_rank])
        aug_rank = max_rank
    end
    n_new = max(0, aug_rank - rm)
    return Q_aug_mat, n_new, Q_aug_mat' * U0_mat
end

function _pick_right_update(V0_mat, L1_mat;
        augment::Bool = true,
        max_rank::Union{Nothing,Int} = nothing,
    )
    rm    = size(V0_mat, 1)
    T_el  = promote_type(eltype(V0_mat), eltype(L1_mat))
    if !augment
        return V0_mat, 0, _identity_overlap_matrix(T_el, rm)
    end
    Ql, rl = _qr_row_basis(L1_mat)
    rl <= 0 && return V0_mat, 0, _identity_overlap_matrix(T_el, rm)
    size(Ql, 1) == 0 && return V0_mat, 0, _identity_overlap_matrix(T_el, rm)
    # Order matters: V0 BEFORE Ql so the first `rm` rows of B_aug equal V0
    # (sign(dt)-independent). New L1 directions enter as rows rm+1 … aug_rank.
    B_aug_mat, aug_rank = _qr_row_basis(vcat(V0_mat, Ql))
    aug_rank <= 0 && return V0_mat, 0, _identity_overlap_matrix(T_el, rm)
    # No padding — see _pick_left_update.
    if !isnothing(max_rank) && aug_rank > max_rank
        B_aug_mat = Matrix(B_aug_mat[1:max_rank, :])
        aug_rank = max_rank
    end
    n_new = max(0, aug_rank - rm)
    return B_aug_mat, n_new, B_aug_mat * V0_mat'
end

function _augmented_left_isometry_from_k(
    U0_tens       :: ITensor,
    K1_tens       :: ITensor,
    lψ_l          :: Index,
    s_ℓ           :: Index,
    old_mid       :: Index,
    evolved_right :: Index;
    augment        :: Bool    = true,
    max_rank       :: Union{Nothing,Int} = nothing,
    return_overlap :: Bool    = false,
)
    dl  = dim(lψ_l);  ds = dim(s_ℓ)
    U0_mat = reshape(_complex_tensor_array(U0_tens, lψ_l, s_ℓ, old_mid), dl * ds, dim(old_mid))
    K1_mat = reshape(_complex_tensor_array(K1_tens, lψ_l, s_ℓ, evolved_right), dl * ds, dim(evolved_right))
    A_aug_mat, n_new, overlap_mat = _pick_left_update(U0_mat, K1_mat;
        augment = augment, max_rank = max_rank)
    aug_lnk  = Index(size(A_aug_mat, 2), tags(old_mid))
    U1_tens  = itensor(reshape(A_aug_mat, dl, ds, size(A_aug_mat, 2)), lψ_l, s_ℓ, aug_lnk)
    return_overlap && return U1_tens, n_new, itensor(overlap_mat, aug_lnk, old_mid)
    return U1_tens, n_new
end

function _augmented_right_isometry_from_l(
    V0_tens      :: ITensor,
    L1_tens      :: ITensor,
    old_mid      :: Index,
    evolved_left :: Index,
    s_ℓ1         :: Index,
    lψ_r         :: Index;
    augment        :: Bool    = true,
    max_rank       :: Union{Nothing,Int} = nothing,
    return_overlap :: Bool    = false,
)
    ds1 = dim(s_ℓ1);  dr1 = dim(lψ_r)
    V0_mat = reshape(_complex_tensor_array(V0_tens, old_mid, s_ℓ1, lψ_r), dim(old_mid), ds1 * dr1)
    L1_mat = reshape(_complex_tensor_array(L1_tens, evolved_left, s_ℓ1, lψ_r), dim(evolved_left), ds1 * dr1)
    B_aug_mat, n_new, _ = _pick_right_update(V0_mat, L1_mat;
        augment = augment, max_rank = max_rank)
    keep = size(B_aug_mat, 1)
    keep > 0 || error("Right-isometry QR produced zero rank")
    canon_lnk = Index(keep, tags(old_mid))
    V1_tens   = itensor(reshape(B_aug_mat, keep, ds1, dr1), canon_lnk, s_ℓ1, lψ_r)
    if return_overlap
        overlap_tens = V0_tens * dag(V1_tens)
        return V1_tens, n_new, overlap_tens
    end
    return V1_tens, n_new
end

# ── Transported S-start ───────────────────────────────────────────────────────

function _transported_s_start_from_augmented_bases(
    U_aug_tens :: ITensor,
    U0_tens    :: ITensor,
    S0_tens    :: ITensor,
    V0_tens    :: ITensor,
    V_aug_tens :: ITensor,
)
    M_hat      = dag(U_aug_tens) * U0_tens
    N_hat      = V0_tens * dag(V_aug_tens)
    S_transport = M_hat * S0_tens * N_hat
    return (; M_hat, N_hat, S_transport)
end

function _transported_s_start_from_augmented_bases(
    S0_tens    :: ITensor,
    M_hat      :: ITensor,
    N_hat      :: ITensor,
)
    return (; M_hat, N_hat, S_transport = M_hat * S0_tens * N_hat)
end

# ── Projected local operators ─────────────────────────────────────────────────

function _left_projected_local_operator(
    HW_env::ITensor,
    right_basis::ITensor,
    link_l::Index,
    site_l::Index,
    mid::Index,
)
    _ = link_l
    _ = site_l
    _ = mid
    return HW_env * right_basis * prime(dag(right_basis))
end

function _right_projected_local_operator(
    HW_env::ITensor,
    left_basis::ITensor,
    mid::Index,
    site_r::Index,
    link_r::Index,
)
    _ = mid
    _ = site_r
    _ = link_r
    return prime(dag(left_basis)) * HW_env * left_basis
end

function _bond_center_projected_local_operator(HW_env::ITensor, left_basis::ITensor, right_basis::ITensor)
    projected_ket = HW_env * left_basis * right_basis
    return prime(dag(left_basis)) * projected_ket * prime(dag(right_basis))
end

function _apply_projected_local_operator(projected_env::ITensor, v::AbstractVector, out_inds::Vararg{Index})
    local_state = itensor(reshape(v, (dim.(out_inds))...), out_inds...)
    return _complex_tensor_vec(noprime(projected_env * local_state), out_inds...)
end

function _apply_projected_local_operator(projected_mat::AbstractMatrix, v::AbstractVector, out_inds::Vararg{Index})
    return projected_mat * v
end

# ── Matrix-free local-gate projected applies (scales to large bonds) ───────────
# For a LOCAL even/odd gate the per-bond effective Hamiltonian is `gate ⊗ I_link_l ⊗
# I_link_r`: the gate touches only the two physical legs, the bond legs are pure
# identities. Materialising `HW_env = gate * δ_link_l * δ_link_r` (or projecting it)
# densifies those identities → an O(χ⁴) dense tensor that OOMs at the 2D x|y cut
# (χ≈maxdim). These apply the SAME projected operator matrix-free in O(χ³): lift the
# local vector through the frame(s), apply the (cheap) gate on the site legs with the
# bond legs passing straight through, then project back. Numerically identical to the
# dense `_left/right/bond_center_projected_local_operator` path (validated ≤1e-12), but
# never forms HW_env or the projected operator. `gate` maps unprimed→primed site legs.
function _apply_local_gate_left_projected(gate::ITensor, right_basis::ITensor,
        v::AbstractVector, out_inds::Vararg{Index})           # K-step: ⟨V0|gate|V0⟩ on (link_l, site_l, mid)
    x  = itensor(reshape(collect(ComplexF64, v), (dim.(out_inds))...), out_inds...)
    Θ  = right_basis * x                                       # lift mid → (site_r, link_r)
    gΘ = noprime(gate * Θ)                                     # gate on sites; bonds pass through
    return _complex_tensor_vec(dag(right_basis) * gΘ, out_inds...)   # project (site_r, link_r) back
end

function _apply_local_gate_right_projected(gate::ITensor, left_basis::ITensor,
        v::AbstractVector, out_inds::Vararg{Index})           # L-step: ⟨U0|gate|U0⟩ on (mid, site_r, link_r)
    x  = itensor(reshape(collect(ComplexF64, v), (dim.(out_inds))...), out_inds...)
    Θ  = left_basis * x                                       # lift mid → (link_l, site_l)
    gΘ = noprime(gate * Θ)
    return _complex_tensor_vec(dag(left_basis) * gΘ, out_inds...)    # project (link_l, site_l) back
end

function _apply_local_gate_center_projected(gate::ITensor, left_basis::ITensor, right_basis::ITensor,
        v::AbstractVector, out_inds::Vararg{Index})           # S-step: ⟨U|⟨V|gate|U⟩|V⟩ on (mid_k, mid_l)
    x  = itensor(reshape(collect(ComplexF64, v), (dim.(out_inds))...), out_inds...)
    Θ  = left_basis * x * right_basis                         # lift both → (link_l, site_l, site_r, link_r)
    gΘ = noprime(gate * Θ)
    return _complex_tensor_vec(dag(left_basis) * gΘ * dag(right_basis), out_inds...)
end

function _prepare_projected_local_operator(projected_env::ITensor, out_inds::Vararg{Index})
    local_dim = prod(dim.(out_inds))
    if local_dim <= BUG_PROJECTED_LOCAL_DENSE_MAXDIM
        bra_inds = ntuple(i -> prime(out_inds[i]), length(out_inds))
        projected_arr = _complex_tensor_array(projected_env, bra_inds..., out_inds...)
        return reshape(projected_arr, local_dim, local_dim)
    end
    return projected_env
end


# ── S-step advance ─────────────────────────────────────────────────────────────

function _advance_s_tensor_in_bases(
    U_basis_tens :: ITensor,
    V_basis_tens :: ITensor,
    S_start      :: ITensor,
    HW_env       :: Union{Nothing,ITensor},
    dt           :: Number;
    lanczos_tol      :: Float64,
    lanczos_maxiter  :: Int,
    lanczos_restart  :: Int = 1,
    substep_method   :: Symbol,
    matrixfree_sstep :: Bool,
    local_gate       :: Union{Nothing,ITensor} = nothing,
)
    s_inds   = inds(S_start)
    d_k      = dim(s_inds[1]);  d_l = dim(s_inds[2])
    d_s      = d_k * d_l
    s_old_vec = _complex_tensor_vec(S_start, s_inds[1], s_inds[2])
    # MATRIX-FREE local-gate path takes precedence: apply gate⊗I⊗I projected onto both
    # augmented frames without forming the dense center operator (the O(χ⁴) wall).
    use_local_gate = local_gate !== nothing
    use_dense = !use_local_gate && (!matrixfree_sstep || (substep_method === :expv && d_s <= 16))

    if use_local_gate
        s_new_vec, numops_s = _linear_substep(
            v -> _apply_local_gate_center_projected(local_gate, U_basis_tens, V_basis_tens, v,
                    s_inds[1], s_inds[2]),
            _active_time_prefactor() * dt,
            s_old_vec;
            method = substep_method,
            lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter,
            restart = lanczos_restart,
        )
    elseif !use_dense
        H_eff_proj = _prepare_projected_local_operator(
            _bond_center_projected_local_operator(HW_env, U_basis_tens, V_basis_tens),
            s_inds[1], s_inds[2],
        )
        s_new_vec, numops_s = _linear_substep(
            v -> _apply_projected_local_operator(H_eff_proj, v, s_inds[1], s_inds[2]),
            _active_time_prefactor() * dt,
            s_old_vec;
            method = substep_method,
            lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter,
            restart = lanczos_restart,
        )
    else
        bra_L  = prime(dag(U_basis_tens))
        bra_R  = prime(dag(V_basis_tens))
        H_eff_mat = zeros(ComplexF64, d_s, d_s)
        e_vec     = zeros(ComplexF64, d_s)
        for col in 1:d_s
            fill!(e_vec, 0.0);  e_vec[col] = 1.0
            S_basis   = itensor(reshape(e_vec, d_k, d_l), s_inds[1], s_inds[2])
            HΘ_col    = HW_env * (U_basis_tens * S_basis * V_basis_tens)
            HS_col    = noprime(bra_L * HΘ_col * bra_R)
            H_eff_mat[:, col] = _complex_tensor_vec(HS_col, s_inds[1], s_inds[2])
        end
        s_new_vec, numops_s = _linear_substep(H_eff_mat, _active_time_prefactor() * dt, s_old_vec;
            method = substep_method, lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter, restart = lanczos_restart,
        )
    end

    return (
        S_new    = itensor(reshape(s_new_vec, d_k, d_l), s_inds[1], s_inds[2]),
        numops_s = numops_s,
    )
end

# ── S-step rank selection and truncation ──────────────────────────────────────

function _s_step_truncation_tail_norm(svals::AbstractVector, keep::Int)
    keep >= length(svals) && return 0.0
    discarded = @view svals[(keep + 1):end]
    return sqrt(sum(abs2, discarded))
end

# Robust thin SVD. Julia's default `svd`/`svd!` uses LAPACK gesdd (divide-and-conquer), which can
# throw LAPACKException(1) (bdsdc non-convergence) on ill-conditioned / clustered-spectrum matrices —
# e.g. a low maxdim cap applied to a high-rank state (the nb20 vortex at md64). Fall back to gesvd
# (QRIteration, the QR-based SVD), which converges where gesdd fails — the same cascade NDTensors uses
# internally. Same factorisation, only a stabler algorithm. The first attempt is non-mutating so the
# original `S_mat` survives intact for the gesvd fallback.
function _robust_thin_svd(S_mat::AbstractMatrix)
    try
        return svd(S_mat; full = false)                                            # gesdd (D&C, fast)
    catch err
        err isa LinearAlgebra.LAPACKException || rethrow(err)
        return svd!(S_mat; full = false, alg = LinearAlgebra.QRIteration())         # gesvd (QR, robust)
    end
end

# Number of singular values to keep from `svals` (sorted descending) under a
# rank cap `maxdim` and a relative weight `cutoff`. The cutoff is measured
# against the largest singular value (`thresh = cutoff·|svals[1]|`), keeping every
# value strictly above it — identical to the Python two_site_bug trunc_thresh
# rule. `cutoff = 0.0` disables the threshold (pure maxdim cap, the default).
function _svd_keep_count(svals::AbstractVector, maxdim::Int, cutoff::Real)
    n = length(svals)
    n == 0 && return 0
    keep = n
    if cutoff > 0
        thresh = cutoff * abs(svals[1])
        keep = max(count(>(thresh), abs.(svals)), 1)
    end
    return min(keep, maxdim, n)
end

function _truncate_quantum_s_step(
    S_new     :: ITensor,
    left_ind  :: Index,
    right_ind :: Index;
    maxdim    :: Int,
    cutoff    :: Float64 = 0.0,
)
    S_mat  = Array(S_new, left_ind, right_ind)
    F      = _robust_thin_svd(S_mat)
    keep   = _svd_keep_count(F.S, maxdim, cutoff)
    p_ind  = Index(keep, tags(left_ind))
    P_tens = itensor(Matrix(@view F.U[:, 1:keep]), left_ind, p_ind)
    SV_mat = Matrix(@view F.Vt[1:keep, :])
    @inbounds for row in 1:keep
        @views SV_mat[row, :] .*= F.S[row]
    end
    return P_tens, itensor(SV_mat, p_ind, right_ind), keep, F.S
end

function _truncate_quantum_s_step_reverse(
    S_new     :: ITensor,
    left_ind  :: Index,
    right_ind :: Index;
    maxdim    :: Int,
    cutoff    :: Float64 = 0.0,
)
    S_mat = Array(S_new, left_ind, right_ind)
    F     = _robust_thin_svd(S_mat)
    keep  = _svd_keep_count(F.S, maxdim, cutoff)
    bond_ind = Index(keep, tags(right_ind))
    US_mat   = Matrix(@view F.U[:, 1:keep])
    @inbounds for col in 1:keep
        @views US_mat[:, col] .*= F.S[col]
    end
    return itensor(US_mat, left_ind, bond_ind), itensor(Matrix(@view F.Vt[1:keep, :]), bond_ind, right_ind), keep, F.S
end

# ── Parallel K+L helper ───────────────────────────────────────────────────────

function _kl_parallel_enabled(checkerboard_threads::Integer)
    checkerboard_threads >= 1 ||
        error("checkerboard_threads must be >= 1; got $checkerboard_threads")
    return min(checkerboard_threads, Threads.nthreads()) >= 2
end

function _run_independent_kl_pair(
    k_job::F,
    l_job::G;
    checkerboard_threads::Integer = Threads.nthreads(),
) where {F<:Function,G<:Function}
    _kl_parallel_enabled(checkerboard_threads) || return k_job(), l_job()
    k_task = Threads.@spawn k_job()
    l_result = l_job()
    return fetch(k_task), l_result
end

# ── Faithful forward KLS candidate ───────────────────────────────────────────

"""
    _faithful_kls_local_bond_candidate(bond_data; ...) -> NamedTuple

Paper-faithful simultaneous K+L augmentation followed by S-step.
Returns `(U_aug_tens, V_aug_tens, n_new_k, n_new_l, numops_s, S_new)`.
"""
function _faithful_kls_local_bond_candidate(
    bond_data;
    dt               :: Number,
    s_dt             :: Number = dt,
    augment          :: Bool   = true,
    aug_krylov_depth :: Int    = 1,
    lanczos_tol      :: Float64,
    lanczos_maxiter  :: Int,
    lanczos_restart  :: Int = 1,
    substep_method   :: Symbol,
    kl_substep_method :: Symbol = substep_method,
    s_substep_method  :: Symbol = substep_method,
    matrixfree_sstep :: Bool,
    checkerboard_threads::Integer = Threads.nthreads(),
    HW_env_override  :: Union{Nothing,ITensor} = nothing,
    local_gate       :: Union{Nothing,ITensor} = nothing,
)
    aug_krylov_depth >= 1 ||
        error("aug_krylov_depth must be >= 1; got $aug_krylov_depth")
    dl    = dim(bond_data.link_l)
    ds_l  = dim(bond_data.site_l)
    ds_r  = dim(bond_data.site_r)
    dr    = dim(bond_data.link_r)
    old_rank  = dim(bond_data.link_mid)
    left_cap  = dl * ds_l
    right_cap = ds_r * dr
    rank_cap  = min(left_cap, right_cap)
    # Saturation is directional: a boundary bond can max out one Schmidt cap
    # while the opposite frame still has room to absorb a useful K/L update.
    # Gate the left and right augmentations independently so we retain any
    # one-sided frame growth that survives the final truncation back to rank_cap.
    augment_left_here  = augment && old_rank < left_cap
    augment_right_here = augment && old_rank < right_cap
    # MATRIX-FREE local-gate path: when `local_gate` is given we apply `gate ⊗ I_link_l ⊗
    # I_link_r` matrix-free (K/L/S below) and NEVER build the dense HW_env — that is the only
    # way this scales past χ≈64 at the 2D x|y cut. HW_env stays `nothing` in that path.
    use_local_gate = local_gate !== nothing
    HW_env = use_local_gate ? nothing :
        (isnothing(HW_env_override) ?
            (bond_data.L_mpo_cur * bond_data.W_left * bond_data.W_right * bond_data.R_mpo_cur) :
            HW_env_override)

    K0_tens = bond_data.U0_tens * bond_data.S0_tens
    mid_k   = _left_site_bond_index(K0_tens, bond_data.link_l, bond_data.site_l)
    d_mid_k = dim(mid_k)
    k0_vec  = _complex_tensor_vec(K0_tens, bond_data.link_l, bond_data.site_l, mid_k)

    L0_tens = bond_data.S0_tens * bond_data.V0_tens
    mid_l   = _site_bond_index(L0_tens, bond_data.site_r, bond_data.link_r)
    d_mid_l = dim(mid_l)
    l0_vec  = _complex_tensor_vec(L0_tens, mid_l, bond_data.site_r, bond_data.link_r)

    # The augmentation pass keeps the full 2r basis (span of [K1 | U0]).
    # Truncation back to the requested runtime cap happens only in the
    # post-S-step SVD. Capping U_aug at old_rank here
    # would let augmentation order columns by the K1-first QR pivot, which makes
    # U_aug ≈ K1's column basis (NOT span(U0)) and breaks sign-reversibility of
    # the local update.
    max_aug_rank = nothing

    k_result, l_result = _run_independent_kl_pair(
        function ()
            # Unified matvec for the K-substep: matrix-free local-gate apply (no dense HW_env)
            # when use_local_gate, else the dense projected operator. Both feed _linear_substep
            # and the deeper-Krylov power iteration identically.
            apply_k = use_local_gate ?
                (w -> _apply_local_gate_left_projected(local_gate, bond_data.V0_tens, w,
                        bond_data.link_l, bond_data.site_l, mid_k)) :
                let H1_k = _prepare_projected_local_operator(
                        _left_projected_local_operator(
                            HW_env, bond_data.V0_tens,
                            bond_data.link_l, bond_data.site_l, mid_k,
                        ),
                        bond_data.link_l, bond_data.site_l, mid_k,
                    )
                    (w -> _apply_projected_local_operator(H1_k, w,
                            bond_data.link_l, bond_data.site_l, mid_k))
                end
            k1_vec, _ = _linear_substep(apply_k, _active_time_prefactor() * dt, k0_vec;
                method = kl_substep_method, lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter, restart = lanczos_restart,
            )
            if aug_krylov_depth == 1
                K1_tens = itensor(reshape(k1_vec, dl, ds_l, d_mid_k),
                    bond_data.link_l, bond_data.site_l, mid_k)
                return _augmented_left_isometry_from_k(
                    bond_data.U0_tens, K1_tens,
                    bond_data.link_l, bond_data.site_l, bond_data.canon_u0, mid_k;
                    augment = augment_left_here, return_overlap = true,
                )
            end
            # Deeper Krylov augmentation: stack [K1, H_K·K1, H_K²·K1, …,
            # H_K^(m-1)·K1] as the augmentation columns. Each H_K application
            # is a matrix-vector product (or ITensor apply) reusing H1_k.
            cols = Vector{Vector{ComplexF64}}(undef, aug_krylov_depth)
            cols[1] = k1_vec
            v = k1_vec
            for j in 2:aug_krylov_depth
                v = apply_k(v)
                cols[j] = v
            end
            K1_block_mat = reduce(hcat,
                reshape(cols[j], dl * ds_l, d_mid_k) for j in 1:aug_krylov_depth)
            mid_k_ext = Index(aug_krylov_depth * d_mid_k, tags(mid_k))
            K1_tens = itensor(reshape(K1_block_mat, dl, ds_l, aug_krylov_depth * d_mid_k),
                bond_data.link_l, bond_data.site_l, mid_k_ext)
            return _augmented_left_isometry_from_k(
                bond_data.U0_tens, K1_tens,
                bond_data.link_l, bond_data.site_l, bond_data.canon_u0, mid_k_ext;
                augment = augment_left_here, return_overlap = true,
            )
        end,
        function ()
            apply_l = use_local_gate ?
                (w -> _apply_local_gate_right_projected(local_gate, bond_data.U0_tens, w,
                        mid_l, bond_data.site_r, bond_data.link_r)) :
                let H1_l = _prepare_projected_local_operator(
                        _right_projected_local_operator(
                            HW_env, bond_data.U0_tens,
                            mid_l, bond_data.site_r, bond_data.link_r,
                        ),
                        mid_l, bond_data.site_r, bond_data.link_r,
                    )
                    (w -> _apply_projected_local_operator(H1_l, w,
                            mid_l, bond_data.site_r, bond_data.link_r))
                end
            l1_vec, _ = _linear_substep(apply_l, _active_time_prefactor() * dt, l0_vec;
                method = kl_substep_method, lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter, restart = lanczos_restart,
            )
            if aug_krylov_depth == 1
                L1_tens = itensor(reshape(l1_vec, d_mid_l, ds_r, dr),
                    mid_l, bond_data.site_r, bond_data.link_r)
                return _augmented_right_isometry_from_l(
                    bond_data.V0_tens, L1_tens,
                    bond_data.canon_v0, mid_l, bond_data.site_r, bond_data.link_r;
                    augment = augment_right_here, return_overlap = true,
                )
            end
            # Deeper Krylov augmentation for L: stack [L1, L1·H_L, …,
            # L1·H_L^(m-1)] as the augmentation rows.
            rows = Vector{Vector{ComplexF64}}(undef, aug_krylov_depth)
            rows[1] = l1_vec
            v = l1_vec
            for j in 2:aug_krylov_depth
                v = apply_l(v)
                rows[j] = v
            end
            L1_block_mat = reduce(vcat,
                reshape(rows[j], d_mid_l, ds_r * dr) for j in 1:aug_krylov_depth)
            mid_l_ext = Index(aug_krylov_depth * d_mid_l, tags(mid_l))
            L1_tens = itensor(reshape(L1_block_mat, aug_krylov_depth * d_mid_l, ds_r, dr),
                mid_l_ext, bond_data.site_r, bond_data.link_r)
            return _augmented_right_isometry_from_l(
                bond_data.V0_tens, L1_tens,
                bond_data.canon_v0, mid_l_ext, bond_data.site_r, bond_data.link_r;
                augment = augment_right_here, return_overlap = true,
            )
        end,
        checkerboard_threads = checkerboard_threads,
    )
    U_aug_tens, n_new_k, M_hat = k_result
    V_aug_tens, n_new_l, N_hat = l_result

    S_start = _transported_s_start_from_augmented_bases(bond_data.S0_tens, M_hat, N_hat).S_transport
    sstep   = _advance_s_tensor_in_bases(U_aug_tens, V_aug_tens, S_start, HW_env, s_dt;
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        lanczos_restart = lanczos_restart, substep_method = s_substep_method,
        matrixfree_sstep = matrixfree_sstep, local_gate = local_gate,
    )
    return (
        U_aug_tens = U_aug_tens, V_aug_tens = V_aug_tens,
        n_new_k = n_new_k, n_new_l = n_new_l,
        numops_s = sstep.numops_s, S_new = sstep.S_new,
    )
end

# ── Faithful reverse KLS candidate ───────────────────────────────────────────

"""
    _faithful_reverse_kls_local_bond_candidate(bond_data; ...) -> NamedTuple

Reverse-parity local update. The sweep traverses bonds right-to-left, but
the local KLS rule is identical to the forward case: K, L, S all use the
old bases (`U0`, `V0`) and the same `dt` sign — the adjoint structure lives
in the sweep ordering, not a local sign or basis flip.
"""
function _faithful_reverse_kls_local_bond_candidate(
    bond_data;
    dt               :: Number,
    s_dt             :: Number = dt,
    augment          :: Bool   = true,
    aug_krylov_depth :: Int    = 1,
    lanczos_tol      :: Float64,
    lanczos_maxiter  :: Int,
    lanczos_restart  :: Int = 1,
    substep_method   :: Symbol,
    kl_substep_method :: Symbol = substep_method,
    s_substep_method  :: Symbol = substep_method,
    matrixfree_sstep :: Bool,
    checkerboard_threads::Integer = Threads.nthreads(),
    HW_env_override  :: Union{Nothing,ITensor} = nothing,
    local_gate       :: Union{Nothing,ITensor} = nothing,
)
    return _faithful_kls_local_bond_candidate(
        bond_data;
        dt = dt,
        s_dt = s_dt,
        augment = augment,
        aug_krylov_depth = aug_krylov_depth,
        lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter,
        lanczos_restart = lanczos_restart,
        substep_method = substep_method,
        kl_substep_method = kl_substep_method,
        s_substep_method = s_substep_method,
        matrixfree_sstep = matrixfree_sstep,
        checkerboard_threads = checkerboard_threads,
        HW_env_override = HW_env_override,
        local_gate = local_gate,
    )
end
