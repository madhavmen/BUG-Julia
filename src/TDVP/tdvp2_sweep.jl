# tdvp2_sweep.jl
#
# Minimal 2-site TDVP implementation:
#   - Dense 2-site local evolution
#   - 1-site backward correction on the carried tensor
#   - 2nd-order Strang composition via forward/reverse half-sweeps

function _build_2site_Heff_mat(
    link_l::Index,
    site_l::Index,
    site_r::Index,
    link_r::Index,
    L_mpo::ITensor,
    R_mpo::ITensor,
    W_l::ITensor,
    W_r::ITensor,
)
    HW_env = complex(L_mpo) * W_l * W_r * R_mpo
    d_l = dim(link_l)
    d_sl = dim(site_l)
    d_sr = dim(site_r)
    d_r = dim(link_r)
    d_tot = d_l * d_sl * d_sr * d_r
    H_mat = zeros(ComplexF64, d_tot, d_tot)
    bra_inds = (prime(link_l), prime(site_l), prime(site_r), prime(link_r))
    e_vec = zeros(ComplexF64, d_tot)
    for col in 1:d_tot
        fill!(e_vec, 0.0 + 0.0im)
        e_vec[col] = 1.0
        theta_col = itensor(
            reshape(e_vec, d_l, d_sl, d_sr, d_r),
            link_l, site_l, site_r, link_r,
        )
        Htheta_col = HW_env * theta_col
        H_mat[:, col] = vec(Array(Htheta_col, bra_inds...))
    end
    return H_mat
end

function _tdvp2_retag_split_pair(
    left_tens::ITensor,
    right_tens::ITensor,
    bond::Int,
)
    shared = commonind(left_tens, right_tens)
    isnothing(shared) && error("2-site TDVP split lost the shared bond index at bond $bond.")
    canon = settags(shared, "Link,l=$bond")
    return replaceind(left_tens, shared, canon), replaceind(right_tens, shared, canon), canon
end

# ── Matrix-free effective-Hamiltonian apply ──────────────────────────────────
# Apply the effective Hamiltonian WITHOUT ever forming the contracted
# HW_env = L·W_l·W_r·R tensor. That object carries indices
# (link_l, link_l', site_l, site_l', site_r, site_r', link_r, link_r') ⇒ χ⁴·d⁴
# entries (68 GB at χ=128, d=2 — the N=50 XX OOM). By associativity the same
# linear map H·θ is obtained by contracting the four tensors onto θ one at a
# time, whose largest intermediate is only ~χ²·d·(MPO bond). Cost per apply
# drops from χ⁴ to χ³ and peak memory from χ⁴ to χ²; results are identical to
# the old dense-HW_env path up to contraction-order rounding (~1e-15).
# See [[project_twosite_bug_fix]] for the analogous BUG matrixfree_sstep fix.
function _heff2_matvec_closure(L_mpo::ITensor, W_l::ITensor, W_r::ITensor, R_mpo::ITensor,
        link_l::Index, site_l::Index, site_r::Index, link_r::Index)
    dl, dsl, dsr, dr = dim(link_l), dim(site_l), dim(site_r), dim(link_r)
    ket = (link_l, site_l, site_r, link_r)
    bra = (prime(link_l), prime(site_l), prime(site_r), prime(link_r))
    Lc = complex(L_mpo)
    return function (v::AbstractVector)
        th = itensor(reshape(ComplexF64.(v), dl, dsl, dsr, dr), ket...)
        return vec(Array(((((Lc * th) * W_l) * W_r) * R_mpo), bra...))
    end
end

function _heff1_matvec_closure(L_mpo::ITensor, W_site::ITensor, R_mpo::ITensor,
        link_l::Index, site_k::Index, link_r::Index)
    dl, ds, dr = dim(link_l), dim(site_k), dim(link_r)
    ket = (link_l, site_k, link_r)
    bra = (prime(link_l), prime(site_k), prime(link_r))
    Lc = complex(L_mpo)
    return function (v::AbstractVector)
        a = itensor(reshape(ComplexF64.(v), dl, ds, dr), ket...)
        return vec(Array((((Lc * a) * W_site) * R_mpo), bra...))
    end
end

function _tdvp2_two_site_update(
    theta::ITensor,
    dt::Number,
    link_l::Index,
    site_l::Index,
    site_r::Index,
    link_r::Index,
    L_mpo::ITensor,
    R_mpo::ITensor,
    W_l::ITensor,
    W_r::ITensor;
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
)
    matvec = _heff2_matvec_closure(L_mpo, W_l, W_r, R_mpo, link_l, site_l, site_r, link_r)
    theta_vec = _complex_tensor_vec(theta, link_l, site_l, site_r, link_r)
    theta_new_vec, numops = _linear_substep(matvec, dt, theta_vec;
        method = substep_method,
        lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter,
        restart = lanczos_restart,
    )
    theta_new = itensor(
        reshape(theta_new_vec, dim(link_l), dim(site_l), dim(site_r), dim(link_r)),
        link_l, site_l, site_r, link_r,
    )
    return theta_new, numops
end

function _tdvp2_site_backward_tensor(
    A::ITensor,
    dt::Number,
    left_ind::Index,
    site_ind::Index,
    right_ind::Index,
    L_mpo::ITensor,
    R_mpo::ITensor,
    W_site::ITensor;
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
)
    matvec = _heff1_matvec_closure(L_mpo, W_site, R_mpo, left_ind, site_ind, right_ind)
    a_vec = _complex_tensor_vec(A, left_ind, site_ind, right_ind)
    a_new_vec, numops = _linear_substep(matvec, -dt, a_vec;
        method = substep_method,
        lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter,
        restart = lanczos_restart,
    )
    A_new = itensor(
        reshape(a_new_vec, dim(left_ind), dim(site_ind), dim(right_ind)),
        left_ind, site_ind, right_ind,
    )
    return A_new, numops
end

function _tdvp2_forward_sweep!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    maxdim::Int,
    cutoff::Float64,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
    npartitions::Int = 1,
)
    N = length(psi)
    orthogonalize!(psi, 1)
    R_all = nothing
    info.env_advance_elapsed += @elapsed begin
        R_all = _build_right_envs_mpo(psi, W)
    end
    L_cur = _left_env_boundary_mpo(psi, W)

    for krange in _partition_ranges(N - 1, npartitions)
    for k in krange
        link_l = k == 1 ? linkinds(psi)[1] : commonind(psi[k - 1], psi[k])
        site_l = siteinds(psi, k)
        site_r = siteinds(psi, k + 1)
        link_r = k == N - 1 ? linkinds(psi)[N + 1] : commonind(psi[k + 1], psi[k + 2])
        theta = psi[k] * psi[k + 1]

        theta_new = nothing
        numops_two = 0
        info.site_update_elapsed += @elapsed begin
            theta_new, numops_two = _tdvp2_two_site_update(
                theta, dt,
                link_l, site_l, site_r, link_r,
                L_cur, R_all[k + 3], W[k], W[k + 1];
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.site_numops, numops_two)

        U = nothing
        S = nothing
        V = nothing
        info.gauge_qr_elapsed += @elapsed begin
            U, S, V = svd(
                theta_new,
                (link_l, site_l);
                maxdim = max(maxdim, 1),
                cutoff = max(cutoff, 0.0),
            )
        end
        carry = S * V
        U, carry, carry_link = _tdvp2_retag_split_pair(U, carry, k)
        psi[k] = U

        if k == N - 1
            psi[k + 1] = carry
            continue
        end

        info.env_advance_elapsed += @elapsed begin
            L_cur = _advance_left_env_mpo(L_cur, psi[k], W[k])
        end
        carry_new = nothing
        numops_back = 0
        info.bond_backward_elapsed += @elapsed begin
            carry_new, numops_back = _tdvp2_site_backward_tensor(
                carry, dt, carry_link, site_r, link_r,
                L_cur, R_all[k + 3], W[k + 1];
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.bond_numops, numops_back)
        psi[k + 1] = carry_new
    end
    end  # krange
    return psi
end

function _tdvp2_reverse_sweep!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    maxdim::Int,
    cutoff::Float64,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
    npartitions::Int = 1,
)
    N = length(psi)
    orthogonalize!(psi, N)
    L_all = nothing
    info.env_advance_elapsed += @elapsed begin
        L_all = _build_left_envs_mpo(psi, W)
    end
    R_cur = _right_env_boundary_mpo(psi, W)

    for krange in reverse(_partition_ranges(N - 1, npartitions))
    for k in reverse(collect(krange))
        link_l = k == 1 ? linkinds(psi)[1] : commonind(psi[k - 1], psi[k])
        site_l = siteinds(psi, k)
        site_r = siteinds(psi, k + 1)
        link_r = k == N - 1 ? linkinds(psi)[N + 1] : commonind(psi[k + 1], psi[k + 2])
        theta = psi[k] * psi[k + 1]

        theta_new = nothing
        numops_two = 0
        info.site_update_elapsed += @elapsed begin
            theta_new, numops_two = _tdvp2_two_site_update(
                theta, dt,
                link_l, site_l, site_r, link_r,
                L_all[k], R_cur, W[k], W[k + 1];
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.site_numops, numops_two)

        U = nothing
        S = nothing
        V = nothing
        info.gauge_qr_elapsed += @elapsed begin
            U, S, V = svd(
                theta_new,
                (link_l, site_l);
                maxdim = max(maxdim, 1),
                cutoff = max(cutoff, 0.0),
            )
        end
        carry = U * S
        carry, V, carry_link = _tdvp2_retag_split_pair(carry, V, k)
        psi[k + 1] = V

        if k == 1
            psi[k] = carry
            continue
        end

        info.env_advance_elapsed += @elapsed begin
            R_cur = _advance_right_env_mpo(R_cur, psi[k + 1], W[k + 1])
        end
        carry_new = nothing
        numops_back = 0
        info.bond_backward_elapsed += @elapsed begin
            carry_new, numops_back = _tdvp2_site_backward_tensor(
                carry, dt, link_l, site_l, carry_link,
                L_all[k], R_cur, W[k];
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.bond_numops, numops_back)
        psi[k] = carry_new
    end
    end  # krange
    return psi
end

# ── Concurrent-partition TDVP2 forward sweep ─────────────────────────────────
# Builds all envs from frozen start-of-step ψ, spawns one Task per partition,
# then does serial seam re-gauge + backward correction.
function _tdvp2_concurrent_forward_sweep!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    maxdim::Int,
    cutoff::Float64,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
    npartitions::Int,
)
    N = length(psi)
    orthogonalize!(psi, 1)
    R_all = _build_right_envs_mpo(psi, W)
    L_all = _build_left_envs_mpo(psi, W)
    psi_frozen = TensorTrain([copy(psi.data[k]) for k in 1:N])

    ranges = _partition_ranges(N - 1, npartitions)

    tasks = map(ranges) do krange
        Threads.@spawn begin
            local_results = Vector{NamedTuple}()
            for k in krange
                bond_inds = _tensortrain_bond_indices(psi_frozen, k)
                link_l = bond_inds.link_l
                site_l = bond_inds.site_l
                site_r = bond_inds.site_r
                link_r = bond_inds.link_r
                theta = psi_frozen[k] * psi_frozen[k + 1]

                theta_new, numops_two = _tdvp2_two_site_update(
                    theta, dt, link_l, site_l, site_r, link_r,
                    L_all[k], R_all[k + 3], W[k], W[k + 1];
                    lanczos_tol = lanczos_tol,
                    lanczos_maxiter = lanczos_maxiter,
                    lanczos_restart = lanczos_restart,
                    substep_method = substep_method,
                )

                U, S, V = svd(
                    theta_new, (link_l, site_l);
                    maxdim = max(maxdim, 1),
                    cutoff = max(cutoff, 0.0),
                )
                carry = S * V
                U, carry, carry_link = _tdvp2_retag_split_pair(U, carry, k)

                push!(local_results, (
                    k           = k,
                    U           = U,
                    carry       = carry,
                    carry_link  = carry_link,
                    site_r      = site_r,
                    link_r      = link_r,
                    numops_two  = numops_two,
                ))
            end
            local_results
        end
    end

    all_results = Dict{Int,NamedTuple}()
    for t in tasks
        for res in fetch(t)
            all_results[res.k] = res
        end
    end

    # Serial assembly: write U for each bond, carry only for the last bond.
    # Each bond was computed from psi_frozen independently. U_k has right link = carry_link_k
    # (new), and U_{k+1} has left link = psi_frozen[k]'s right link (original frozen ID).
    # Relink to unify adjacent indices, then re-gauge.
    for (part_idx, krange) in enumerate(ranges)
        for k in krange
            res = all_results[k]
            info.site_update_elapsed += 0.0
            push!(info.site_numops, res.numops_two)
            psi[k] = res.U
            if k == N - 1
                psi[k + 1] = res.carry
            end
        end
    end

    for k in 1:(N - 2)
        carry_link_k    = all_results[k].carry_link
        frozen_left_kp1 = only(commoninds(psi[k + 1], psi_frozen[k]; tags="Link"))
        psi[k + 1] = replaceind(psi[k + 1], frozen_left_kp1, carry_link_k)
    end

    orthogonalize!(psi, N)
    return psi
end

# ── Concurrent-partition TDVP2 reverse sweep ─────────────────────────────────
function _tdvp2_concurrent_reverse_sweep!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    maxdim::Int,
    cutoff::Float64,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
    npartitions::Int,
)
    N = length(psi)
    orthogonalize!(psi, N)
    L_all = _build_left_envs_mpo(psi, W)
    R_all = _build_right_envs_mpo(psi, W)
    psi_frozen = TensorTrain([copy(psi.data[k]) for k in 1:N])

    rev_ranges = reverse(_partition_ranges(N - 1, npartitions))

    tasks = map(rev_ranges) do krange
        Threads.@spawn begin
            local_results = Vector{NamedTuple}()
            for k in reverse(collect(krange))
                bond_inds = _tensortrain_bond_indices(psi_frozen, k)
                link_l = bond_inds.link_l
                site_l = bond_inds.site_l
                site_r = bond_inds.site_r
                link_r = bond_inds.link_r
                theta = psi_frozen[k] * psi_frozen[k + 1]

                theta_new, numops_two = _tdvp2_two_site_update(
                    theta, dt, link_l, site_l, site_r, link_r,
                    L_all[k], R_all[k + 3], W[k], W[k + 1];
                    lanczos_tol = lanczos_tol,
                    lanczos_maxiter = lanczos_maxiter,
                    lanczos_restart = lanczos_restart,
                    substep_method = substep_method,
                )

                U, S, V = svd(
                    theta_new, (link_l, site_l);
                    maxdim = max(maxdim, 1),
                    cutoff = max(cutoff, 0.0),
                )
                carry = U * S
                carry, V, carry_link = _tdvp2_retag_split_pair(carry, V, k)

                push!(local_results, (
                    k           = k,
                    carry       = carry,
                    V           = V,
                    carry_link  = carry_link,
                    link_l      = link_l,
                    site_l      = site_l,
                    numops_two  = numops_two,
                ))
            end
            local_results
        end
    end

    all_results = Dict{Int,NamedTuple}()
    for t in tasks
        for res in fetch(t)
            all_results[res.k] = res
        end
    end

    # Serial assembly: write V for each bond, carry only for site 1.
    # V_k has left link = carry_link_k (new), right link = psi_frozen[k+1].right (frozen).
    # Relink right-to-left: replace V_k's frozen right link with carry_link_{k+1}.
    for (part_idx, krange) in enumerate(rev_ranges)
        for k in reverse(collect(krange))
            res = all_results[k]
            push!(info.site_numops, res.numops_two)
            psi[k + 1] = res.V
            if k == 1
                psi[1] = res.carry
            end
        end
    end

    for k in 1:(N - 2)
        carry_link_kp1   = all_results[k + 1].carry_link
        frozen_right_kp1 = only(commoninds(psi[k + 1], psi_frozen[k + 2]; tags="Link"))
        psi[k + 1] = replaceind(psi[k + 1], frozen_right_kp1, carry_link_kp1)
    end

    orthogonalize!(psi, 1)
    return psi
end

function _apply_tdvp2_sweep_schedule!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    maxdim::Int,
    cutoff::Float64,
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
    step_mode_cfg,
    step_index::Int = 0,
    npartitions::Int = 1,
    parallel_partitions::Bool = false,
)
    for (direction, coeff) in _tdvp_schedule_for_step(step_mode_cfg, step_index)
        sweep_dt = coeff * dt
        if direction === :forward
            info.forward_sweep_elapsed += @elapsed begin
                _tdvp2_forward_sweep!(psi, W, sweep_dt, info;
                    maxdim = maxdim, cutoff = cutoff,
                    lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
                    lanczos_restart = lanczos_restart, substep_method = substep_method,
                    npartitions = npartitions,
                )
            end
        else
            info.reverse_sweep_elapsed += @elapsed begin
                _tdvp2_reverse_sweep!(psi, W, sweep_dt, info;
                    maxdim = maxdim, cutoff = cutoff,
                    lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
                    lanczos_restart = lanczos_restart, substep_method = substep_method,
                    npartitions = npartitions,
                )
            end
        end
    end
    return psi
end

"""
    tdvp2_step!(psi, W; dt, kwargs...) -> TDVPInfo

Advance the quantum 2-site TDVP integrator by one public step.
Supported step modes include :symmetric_fr, :symmetric_rf, :third_order_frf, :third_order_rfr,
and higher-order Yoshida schemes.
"""
function tdvp2_step!(
    psi::TensorTrain,
    W::TensorTrainOperator;
    dt::Number,
    maxdim::Int          = typemax(Int),
    cutoff::Float64      = 1e-12,
    lanczos_tol::Float64 = 1e-15,
    lanczos_maxiter::Int = 30,
    lanczos_restart::Int = 1,
    substep_method::Symbol = :expv,
    expv_backend::Symbol = :auto,
    step_mode::Symbol    = :symmetric_fr,
    step_index::Int      = 0,
)
    N = length(psi)
    N < 2 && error("tdvp2_step! requires at least 2 sites")

    allowed_backends = (:krylovkit, :native_hermitian_lanczos)
    effective_backend = expv_backend === :auto ? :native_hermitian_lanczos : expv_backend
    effective_backend in allowed_backends ||
        error("Unknown TDVP expv_backend: $expv_backend.")

    step_mode_cfg = _resolve_tdvp_step_mode(step_mode)
    info = TDVPInfo(site_order = 2)
    info.bond_dims_before = [dim(linkind(psi, k)) for k in 1:(N - 1)]

    info.elapsed = @elapsed begin
        _with_bug_expv_backend(effective_backend) do
            _apply_tdvp2_sweep_schedule!(psi, W, dt, info;
                maxdim = maxdim,
                cutoff = cutoff,
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
                step_mode_cfg = step_mode_cfg,
                step_index = step_index,
            )
        end
    end

    info.bond_dims_after = [dim(linkind(psi, k)) for k in 1:(N - 1)]
    return info
end

"""
    tdvp2_parallel_step!(psi, W; dt, kwargs...) -> TDVPInfo

2-site TDVP integrator with optional concurrent-partition execution.
When `parallel_partitions=true` and `npartitions > 1`, bonds are split into
`npartitions` blocks and evolved concurrently via `Threads.@spawn` against
frozen start-of-step environments (Secular et al. arXiv:1912.06127 scheme).
Default (`parallel_partitions=false`) falls through to the sequential path.
"""
function tdvp2_parallel_step!(
    psi::TensorTrain,
    W::TensorTrainOperator;
    dt::Number,
    maxdim::Int           = typemax(Int),
    cutoff::Float64       = 1e-12,
    lanczos_tol::Float64  = 1e-15,
    lanczos_maxiter::Int  = 30,
    lanczos_restart::Int  = 1,
    substep_method::Symbol = :expv,
    expv_backend::Symbol  = :auto,
    step_mode::Symbol     = :symmetric_fr,
    step_index::Int       = 0,
    npartitions::Int      = 1,
    parallel_partitions::Bool = false,
    kwargs...,
)
    N = length(psi)
    N < 2 && error("tdvp2_parallel_step! requires at least 2 sites")
    npartitions >= 1 || error("npartitions must be >= 1; got $npartitions")

    allowed_backends = (:krylovkit, :native_hermitian_lanczos)
    effective_backend = expv_backend === :auto ? :native_hermitian_lanczos : expv_backend
    effective_backend in allowed_backends ||
        error("Unknown TDVP expv_backend: $expv_backend.")

    step_mode_cfg = _resolve_tdvp_step_mode(step_mode)
    info = TDVPInfo(site_order = 2)
    info.bond_dims_before = [dim(linkind(psi, k)) for k in 1:(N - 1)]

    info.elapsed = @elapsed begin
        _with_bug_expv_backend(effective_backend) do
            _apply_tdvp2_sweep_schedule!(psi, W, dt, info;
                maxdim = maxdim,
                cutoff = cutoff,
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
                step_mode_cfg = step_mode_cfg,
                step_index = step_index,
                npartitions = npartitions,
                parallel_partitions = parallel_partitions,
            )
        end
    end

    info.bond_dims_after = [dim(linkind(psi, k)) for k in 1:(N - 1)]
    return info
end
