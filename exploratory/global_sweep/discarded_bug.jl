# discarded_bug.jl
#
# Rank-adaptive discarded-projector BUG integrator — the MPS realisation of the
# rank-adaptive tree-tensor-network BUG of Ceruti–Lubich–Walach / Sulz (thesis,
# Algorithms 5–7), specialised to the linear (MPS) tree. The Julia mirror of the
# validated Python `sweep.py`. Two defining choices:
#
#   * the basis growth is driven by the DISCARDED (orthogonal-complement) projector,
#     applied EXPLICITLY (P_perp = I − U0 U0^dagger) and PER basis matrix, never by
#     forming the augmented overlap matrices M, N; and
#   * the augmentation direction is read from the FULL Hamiltonian image
#     phi = H · psi (computed once as an MPS), NOT from a local two-site block — so
#     the augmented bases span range(psi) ⊕ range(H psi) (the exact rank-adaptive BUG
#     basis) rather than a local approximation.
#
# One step (`discarded_bug_step!` → `_dbug_global_step!`)
# ------------------------------------------------------
#   1. Image.     phi = H · psi as an MPS (TTutils.contract; align the two trivial
#                 boundary links to psi so phi and psi are contractible site-by-site).
#   2. K-sweep    (left→right, `_dbug_k_sweep`). Build the augmented LEFT isometries
#                 W_1 … W_cl. At each bond keep psi's left frame EXACTLY (U0 = qr(psi
#                 part)), admit only the discarded part of phi, (I − U0 U0^dagger) phi,
#                 SVD-truncating that complement to the remaining budget maxdim − rank(U0).
#                 Keeping psi exact is what makes truncation rank-STABLE.
#   3. L-sweep    (right→left, `_dbug_l_sweep`). Mirror: augmented RIGHT isometries
#                 Z_cr … Z_N.
#   4. Galerkin   core (Alg. 7). Seed S_start = <W, Z | psi> from the sweep carries
#                 (no M/N overlap matrices) and integrate the single centre connecting
#                 tensor over the full step under the two-site effective Hamiltonian
#                 E_left · W_cl · W_cr · E_right. This is the only time evolution.
#   5. Truncate   the new centre and return the orthogonality centre to site 1.
#
# Forward-only / inverse-free: a single Galerkin core evolution and a single
# truncation, with NO backward (−tau) substep and no overlap-matrix inverse. At full
# bond dimension the step is EXACT; the truncation error converges monotonically as
# the bond dimension is raised. Measured (XX, exact-diagonalisation reference): the
# single-step infidelity is CURRENTLY O(dt^2) (ratio ~4 per dt-halving) ⇒ the step is
# 1ST ORDER in the state for now and CONVERGENT (no forward-only floor). The
# Ceruti-Kusch-Lubich rank-adaptive-BUG theory argues the augmented Galerkin core
# should give 2nd order; the implementation does not yet reproduce that (tracked as
# a follow-up, not re-derived here) — treat this as a first-order method for now.

# ── per-site helpers (index extraction; trivial boundary links) ────────────────
_dbug_siteind(t::ITensor)         = only(filter(ix -> hastags(ix, "Site"), inds(t)))
_dbug_leftlink(psi, i::Int)       = only(uniqueinds(psi[i], psi[i + 1]; tags = "Link"))
_dbug_rightlink(psi, i::Int)      = only(uniqueinds(psi[i], psi[i - 1]; tags = "Link"))

# ── K-sweep: augmented LEFT isometries W_1 … W_cl ──────────────────────────────
# Returns (Wv::Vector{ITensor}, aps_c, n_new::Vector{Int}) where aps_c = <W_cl | psi>
# on (aug_cl, psi_centre_bond) and n_new[i] is the number of admitted phi directions.
function _dbug_k_sweep(psi, phi, cl::Int, maxdim::Int, cutoff::Float64)
    Wv    = Vector{ITensor}(undef, cl)
    n_new = zeros(Int, cl)
    aps = nothing;  aph = nothing                       # carries: (aug_{i-1}, psi/phi link_{i-1})
    for i in 1:cl
        s_i = _dbug_siteind(psi[i])
        if aps === nothing
            psit = psi[i];  phit = phi[i];  prevb = _dbug_leftlink(psi, i)
        else
            psit = aps * psi[i];  phit = aph * phi[i];  prevb = commonind(aps, psit)
        end
        u0, _r  = qr(psit, prevb, s_i; tags = "Link,l=$i")          # keep psi frame EXACTLY
        mid_psi = commonind(u0, _r)
        proj     = dag(u0) * phit                                    # U0^dagger phi
        phi_perp = phit - u0 * proj                                  # (I − U0 U0^dagger) phi
        W_i = u0
        # Augment to 2r (budget = rpsi, Sulz Alg. 5): admit r extra discarded phi
        # directions so the propose bases can span new (incl. high-|charge|) sectors.
        # Capping off-central frames at maxdim-rpsi instead starves the augmentation
        # and stalls cooling. The redundant 2r scaffolding is collapsed back to the
        # true minimal Schmidt rank by the lossless re-gauge in _dbug_global_step!.
        budget = dim(mid_psi)
        if budget > 0
            Uc, Sc, _Vc = svd(phi_perp, (prevb, s_i); maxdim = budget, cutoff = cutoff)
            qb = commonind(Uc, Sc)
            if dim(qb) > 0
                db = dim(prevb);  ds = dim(s_i)
                u0m = reshape(_complex_tensor_array(u0, prevb, s_i, mid_psi), db * ds, dim(mid_psi))
                qcm = reshape(_complex_tensor_array(Uc, prevb, s_i, qb),      db * ds, dim(qb))
                Wm, r = _qr_column_basis(hcat(u0m, qcm))
                aug = Index(r, "Link,l=$i")
                W_i = itensor(reshape(Wm, db, ds, r), prevb, s_i, aug)
                n_new[i] = max(0, r - dim(mid_psi))
            end
        end
        aps = dag(W_i) * psit;  aph = dag(W_i) * phit
        Wv[i] = W_i
    end
    return Wv, aps, n_new
end

# ── L-sweep: augmented RIGHT isometries Z_cr … Z_N (mirror of K-sweep) ──────────
# Returns ({i => Z_i}, bps_c, n_new). Z_i is stored on (site_i, right_link, left_aug).
function _dbug_l_sweep(psi, phi, cr::Int, maxdim::Int, cutoff::Float64)
    N = length(psi)
    Z = Dict{Int,ITensor}();  n_new = Dict{Int,Int}()
    bps = nothing;  bph = nothing                       # carries: (psi/phi link_i, zaug_{i+1})
    for i in N:-1:cr
        s_i = _dbug_siteind(psi[i])
        if bps === nothing
            psit = psi[i];  phit = phi[i];  nextb = _dbug_rightlink(psi, i)
        else
            psit = psi[i] * bps;  phit = phi[i] * bph;  nextb = commonind(bps, psit)
        end
        v0, _r  = qr(psit, s_i, nextb; tags = "Link,l=$(i-1)")       # keep psi frame EXACTLY
        mid_psi = commonind(v0, _r)
        proj     = dag(v0) * phit
        phi_perp = phit - v0 * proj
        V_i = v0;  n_new[i] = 0
        # Mirror of _dbug_k_sweep: augment to 2r (budget = rpsi). See note there.
        budget = dim(mid_psi)
        if budget > 0
            Uc, Sc, _Vc = svd(phi_perp, (s_i, nextb); maxdim = budget, cutoff = cutoff)
            qb = commonind(Uc, Sc)
            if dim(qb) > 0
                ds = dim(s_i);  dn = dim(nextb)
                v0m = reshape(_complex_tensor_array(v0, s_i, nextb, mid_psi), ds * dn, dim(mid_psi))
                qcm = reshape(_complex_tensor_array(Uc, s_i, nextb, qb),      ds * dn, dim(qb))
                Vm, r = _qr_column_basis(hcat(v0m, qcm))
                zaug = Index(r, "Link,l=$(i-1)")
                V_i = itensor(reshape(Vm, ds, dn, r), s_i, nextb, zaug)
                n_new[i] = max(0, r - dim(mid_psi))
            end
        end
        bps = psit * dag(V_i);  bph = phit * dag(V_i)
        Z[i] = V_i
    end
    return Z, bps, n_new
end

# ── One global discarded-projector BUG step (mutates psi in place) ─────────────
function _dbug_global_step!(
    psi  :: TensorTrain,
    W    :: TensorTrainOperator,
    tau  :: Number;
    maxdim          :: Int,
    cutoff          :: Float64,
    lanczos_tol     :: Float64,
    lanczos_maxiter :: Int,
    info            :: Union{BUGInfo,Nothing} = nothing,
)
    N  = length(psi);  cl = N ÷ 2;  cr = cl + 1
    orthogonalize!(psi, 1)
    phi = TTutils.contract(W, psi)
    # Align phi's trivial boundary links to psi's (contract mints its own dim-1 ends).
    phi[1] = replaceind(phi[1], _dbug_leftlink(phi, 1),  _dbug_leftlink(psi, 1))
    phi[N] = replaceind(phi[N], _dbug_rightlink(phi, N), _dbug_rightlink(psi, N))

    Wv, aps_c, n_new_k = _dbug_k_sweep(psi, phi, cl, maxdim, cutoff)
    Z,  bps_c, n_new_l = _dbug_l_sweep(psi, phi, cr, maxdim, cutoff)

    u0c = Wv[cl];  v0c = Z[cr]
    mid_u = commonind(u0c, aps_c)
    mid_v = commonind(v0c, bps_c)
    s_start = aps_c * bps_c                                          # (mid_u, mid_v) = <W,Z|psi>

    # MPO environments in the augmented basis: left from W_1..W_{cl-1}, right from Z_{cr+1}..Z_N.
    Lenv = _left_env_boundary_mpo(psi, W)
    for k in 1:(cl - 1);  Lenv = _advance_left_env_mpo(Lenv, Wv[k], W[k]);  end
    Renv = _right_env_boundary_mpo(psi, W)
    for k in N:-1:(cr + 1);  Renv = _advance_right_env_mpo(Renv, Z[k], W[k]);  end
    HW = Lenv * W[cl] * W[cr] * Renv

    # Galerkin core: integrate the centre connecting tensor under the 2-site effective H.
    apply_s = function (v::AbstractVector)
        S   = itensor(reshape(ComplexF64.(v), dim(mid_u), dim(mid_v)), mid_u, mid_v)
        th  = u0c * S * v0c
        hth = noprime(HW * th)
        sl  = dag(u0c) * hth
        return _complex_tensor_vec(dag(v0c) * sl, mid_u, mid_v)
    end
    s_vec = _complex_tensor_vec(s_start, mid_u, mid_v)
    s_new_vec, _ = _general_linear_substep(apply_s, tau, s_vec; method = :expv,
                        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter, issymmetric = true)
    S_new = itensor(reshape(s_new_vec, dim(mid_u), dim(mid_v)), mid_u, mid_v)

    U_s, SV, keep, _sv = _truncate_quantum_s_step(S_new, mid_u, mid_v; maxdim = maxdim, cutoff = cutoff)
    left_core  = u0c * U_s
    right_core = SV  * v0c

    cores = ITensor[Wv[k] for k in 1:(cl - 1)]
    push!(cores, left_core);  push!(cores, right_core)
    for k in (cr + 1):N;  push!(cores, Z[k]);  end
    psi2 = TensorTrain(cores)
    # The K/L sweeps augment every bond to 2r; those redundant directions must be
    # collapsed back to the true minimal Schmidt rank each step, otherwise the 2r
    # scaffolding accumulates and the off-central bonds double every step. A lossless
    # SVD sweep (tiny cutoff, no maxdim cap) does this without discarding weight; the
    # user-facing maxdim truncation stays confined to the central S-step above. This
    # mirrors the Alice TWO-WAY canonical(L-1,trunc=None); canonical(0,trunc=None)
    # re-gauge: BOTH passes must be SVD-based (not QR-only), or the direction NOT
    # covered by the SVD sweep is left at its full augmented (2r) rank — the Python
    # investigation (2026-07-08) measured this asymmetry blow up the untruncated half
    # to full Hilbert rank at L>=20. `orthogonalize!` is QR-only and cannot reduce
    # rank, so the second pass uses `svd_compress_reverse!` (mirror of
    # `svd_compress!`), giving a genuine two-way SVD re-gauge.
    TTutils.svd_compress!(psi2; maxdim = typemax(Int), cutoff = 1e-14)
    TTutils.svd_compress_reverse!(psi2; maxdim = typemax(Int), cutoff = 1e-14)
    orthogonalize!(psi2, 1)
    for k in 1:N;  psi[k] = psi2[k];  end                           # mutate psi in place

    if info !== nothing
        append!(info.aug_sizes_k, n_new_k)
        append!(info.aug_sizes_l, [n_new_l[i] for i in cr:N])
    end
    return maximum(Int[dim(commonind(psi[k], psi[k + 1])) for k in 1:(N - 1)])
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    discarded_bug_step!(psi, W; dt, kwargs...) -> BUGInfo

Advance `psi` by one rank-adaptive discarded-projector BUG step against the MPO `W`
(the global sweep of `_dbug_global_step!`): form `phi = H·psi`, build the augmented
left/right isometries by the per-matrix discarded-projector sweeps (keeping `psi`
exact and admitting only `phi`'s complement), integrate the single Galerkin centre
connecting tensor over the full step, truncate to `maxdim`, and return the
orthogonality centre to site 1. The bond dimension grows along the whole chain (the
light cone) as the wall melts; at full bond dimension the step is EXACT and the
truncation error converges monotonically as `maxdim` is raised. There is NO backward
(negative-time) substep — BUG is inverse-free by design (`backward_correction_calls`
is always 0).

`maxdim` caps the rank-adaptive bond growth (`typemax(Int)` = grow freely); `cutoff`
is the relative singular-value threshold of the SVD truncations. `time_prefactor =
-im` (real time) by default; pass `ComplexF64(1)` for imaginary time. The
`order`/`matrixfree_sstep`/`aug_tol` keywords are retained for call-site
compatibility but no longer select a scheme (the global sweep is the only scheme).

Currently a FIRST-order method in practice (measured single-step infidelity ~O(dt^2),
i.e. state error ~O(dt); see `discarded_bug.jl` header) — treat it as first order for
now, not the second order the rank-adaptive-BUG theory argues for.
"""
function discarded_bug_step!(
    psi :: TensorTrain,
    W   :: TensorTrainOperator;
    dt               :: Number,
    order            :: Symbol  = :sweep,
    maxdim           :: Int     = typemax(Int),
    cutoff           :: Float64 = 0.0,
    lanczos_tol      :: Float64 = 1e-15,
    lanczos_maxiter  :: Int     = 40,
    substep_method   :: Symbol  = :expv,
    matrixfree_sstep :: Bool    = false,
    aug_tol          :: Float64 = BUG_DEFAULT_AUG_TOL,
    expv_backend     :: Symbol  = :auto,
    time_prefactor   :: ComplexF64 = ComplexF64(-im),
)
    _ = (order, matrixfree_sstep, aug_tol, substep_method)          # retained for compatibility
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
                _dbug_global_step!(psi, W, time_prefactor * dt;
                    maxdim = maxdim, cutoff = cutoff, lanczos_tol = lanczos_tol,
                    lanczos_maxiter = lanczos_maxiter, info = info)
            end
        end
    end

    info.bond_dims_after = [dim(linkind(psi, k)) for k in 1:(N - 1)]
    return info
end
