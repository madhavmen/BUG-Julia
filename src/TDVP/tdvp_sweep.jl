# tdvp_sweep.jl
#
# Forward and reverse TDVP sweeps, composition schedule application,
# and the public `tdvp_step!` entry point.

# ── Forward sweep ─────────────────────────────────────────────────────────────

"""
    _tdvp_forward_sweep!(psi, W, dt, info; kwargs...)

Advance one left-to-right TDVP sweep:
  For k = 1 … N-1:  1-site update (+dt) → QR → 0-site backward (-dt)
  For k = N:         1-site update (+dt)
"""
function _tdvp_forward_sweep!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
)
    N = length(psi)
    orthogonalize!(psi, 1)
    R_mpo     = nothing
    info.env_advance_elapsed += @elapsed begin
        R_mpo = _build_right_envs_mpo(psi, W)
    end
    L_mpo_cur = _left_env_boundary_mpo(psi, W)

    for k in 1:(N - 1)
        link_l = k == 1 ? linkinds(psi)[1] : commonind(psi[k-1], psi[k])
        site_k = siteinds(psi, k)
        link_r = commonind(psi[k], psi[k+1])

        # 1-site forward update
        A_new = nothing
        numops_site = 0
        info.site_update_elapsed += @elapsed begin
            A_new, numops_site = _tdvp_site_update(
                psi, k, dt, L_mpo_cur, R_mpo[k+2], W[k];
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.site_numops, numops_site)

        # QR gauge move
        Q_k = nothing
        C_k = nothing
        canon_link = nothing
        info.gauge_qr_elapsed += @elapsed begin
            Q_k, C_k, canon_link = _tdvp_forward_qr(A_new, link_l, site_k, "Link,l=$k")
        end
        psi[k]    = Q_k
        info.env_advance_elapsed += @elapsed begin
            L_mpo_cur = _advance_left_env_mpo(L_mpo_cur, psi[k], W[k])
        end

        # 0-site backward step
        C_new = nothing
        numops_bond = 0
        info.bond_backward_elapsed += @elapsed begin
            C_new, numops_bond = _tdvp_bond_backward(
                C_k, canon_link, link_r, dt, L_mpo_cur, R_mpo[k+2];
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.bond_numops, numops_bond)
        psi[k+1] = C_new * psi[k+1]
    end

    # Last site: 1-site update only (no backward step at boundary)
    link_l = commonind(psi[N-1], psi[N])
    site_N = siteinds(psi, N)
    link_r = linkinds(psi)[N+1]
    A_new = nothing
    numops_site = 0
    info.site_update_elapsed += @elapsed begin
        A_new, numops_site = _tdvp_site_update(
            psi, N, dt, L_mpo_cur, _right_env_boundary_mpo(psi, W), W[N];
            lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter,
            lanczos_restart = lanczos_restart,
            substep_method = substep_method,
        )
    end
    push!(info.site_numops, numops_site)
    psi[N] = A_new
    return psi
end

# ── Reverse sweep ─────────────────────────────────────────────────────────────

"""
    _tdvp_reverse_sweep!(psi, W, dt, info; kwargs...)

Advance one right-to-left TDVP sweep:
  For k = N … 2:  1-site update (+dt) → LQ → 0-site backward (-dt)
  For k = 1:      1-site update (+dt)
"""
function _tdvp_reverse_sweep!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
)
    N = length(psi)
    orthogonalize!(psi, N)
    L_mpo     = nothing
    info.env_advance_elapsed += @elapsed begin
        L_mpo = _build_left_envs_mpo(psi, W)
    end
    R_mpo_cur = _right_env_boundary_mpo(psi, W)

    for k in N:-1:2
        link_l = commonind(psi[k-1], psi[k])
        site_k = siteinds(psi, k)
        link_r = k == N ? linkinds(psi)[N+1] : commonind(psi[k], psi[k+1])

        # 1-site reverse update
        A_new = nothing
        numops_site = 0
        info.site_update_elapsed += @elapsed begin
            A_new, numops_site = _tdvp_site_update(
                psi, k, dt, L_mpo[k], R_mpo_cur, W[k];
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.site_numops, numops_site)

        # LQ gauge move
        C_km1 = nothing
        Q_k = nothing
        canon_link = nothing
        info.gauge_qr_elapsed += @elapsed begin
            C_km1, Q_k, canon_link = _tdvp_reverse_lq(A_new, site_k, link_r, "Link,l=$(k-1)")
        end
        psi[k]    = Q_k
        info.env_advance_elapsed += @elapsed begin
            R_mpo_cur = _advance_right_env_mpo(R_mpo_cur, psi[k], W[k])
        end

        # 0-site backward step
        C_new = nothing
        numops_bond = 0
        info.bond_backward_elapsed += @elapsed begin
            C_new, numops_bond = _tdvp_bond_backward(
                C_km1, link_l, canon_link, dt, L_mpo[k], R_mpo_cur;
                lanczos_tol = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method = substep_method,
            )
        end
        push!(info.bond_numops, numops_bond)
        psi[k-1] = psi[k-1] * C_new
    end

    # First site: 1-site update only
    link_l = linkinds(psi)[1]
    site_1 = siteinds(psi, 1)
    link_r = commonind(psi[1], psi[2])
    A_new = nothing
    numops_site = 0
    info.site_update_elapsed += @elapsed begin
        A_new, numops_site = _tdvp_site_update(
            psi, 1, dt, _left_env_boundary_mpo(psi, W), R_mpo_cur, W[1];
            lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter,
            lanczos_restart = lanczos_restart,
            substep_method = substep_method,
        )
    end
    push!(info.site_numops, numops_site)
    psi[1] = A_new
    return psi
end

# ── Composition schedule dispatcher ──────────────────────────────────────────

function _apply_tdvp_sweep_schedule!(
    psi::TensorTrain,
    W::TensorTrainOperator,
    dt::Number,
    info::TDVPInfo;
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
    step_mode_cfg,
    step_index::Int = 0,
)
    for (direction, coeff) in _tdvp_schedule_for_step(step_mode_cfg, step_index)
        sweep_dt = coeff * dt
        if direction === :forward
            info.forward_sweep_elapsed += @elapsed begin
                _tdvp_forward_sweep!(psi, W, sweep_dt, info;
                    lanczos_tol = lanczos_tol,
                    lanczos_maxiter = lanczos_maxiter,
                    lanczos_restart = lanczos_restart,
                    substep_method = substep_method,
                )
            end
        else
            info.reverse_sweep_elapsed += @elapsed begin
                _tdvp_reverse_sweep!(psi, W, sweep_dt, info;
                    lanczos_tol = lanczos_tol,
                    lanczos_maxiter = lanczos_maxiter,
                    lanczos_restart = lanczos_restart,
                    substep_method = substep_method,
                )
            end
        end
    end
    return psi
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    tdvp_step!(psi, W; dt, kwargs...) -> TDVPInfo

Advance the quantum 1-site TDVP integrator by one public step.

Arguments
- `psi`  : TensorTrain state (mutated in place)
- `W`    : TensorTrainOperator MPO Hamiltonian
- `dt`   : generator coefficient. For real-time evolution `dψ/dt = -iHψ`,
           pass `dt = -im * Δt`. Positive real `dt` gives imaginary-time evolution.

Keyword arguments
- `step_mode`       : composition order (default `:symmetric_fr`)
                      Options: `:symmetric_fr/rf`, `:third_order_frf/rfr`,
                               `:fourth_order_yoshida_fr/rf`
- `substep_method`  : `:expv` (Krylov), `:euler`, or `:rk4`
- `lanczos_tol`     : Krylov tolerance (default `1e-15`)
- `lanczos_maxiter` : Krylov dimension (default 30)
- `expv_backend`    : `:auto`, `:krylovkit`, or `:native_hermitian_lanczos`

Returns a `TDVPInfo` diagnostics record.
"""
function tdvp_step!(
    psi::TensorTrain,
    W::TensorTrainOperator;
    dt::Number,
    maxdim::Int     = 200,   # accepted but unused (TDVP is rank-preserving)
    cutoff::Float64 = 1e-16, # accepted but unused
    lanczos_tol::Float64    = 1e-15,
    lanczos_maxiter::Int    = 30,
    lanczos_restart::Int    = 1,
    substep_method::Symbol  = :expv,
    expv_backend::Symbol    = :auto,
    step_mode::Symbol       = :symmetric_fr,
    step_index::Int         = 0,
)
    _ = maxdim; _ = cutoff
    N = length(psi)
    N < 2 && error("tdvp_step! requires at least 2 sites")

    allowed_backends = (:krylovkit, :native_hermitian_lanczos)
    effective_backend = expv_backend === :auto ? :native_hermitian_lanczos : expv_backend
    effective_backend in allowed_backends ||
        error("Unknown TDVP expv_backend: $expv_backend.")

    step_mode_cfg = _resolve_tdvp_step_mode(step_mode)
    info = TDVPInfo(site_order = 1)
    info.bond_dims_before = [dim(linkind(psi, k)) for k in 1:(N-1)]

    info.elapsed = @elapsed begin
        _with_bug_expv_backend(effective_backend) do
            _apply_tdvp_sweep_schedule!(psi, W, dt, info;
                lanczos_tol     = lanczos_tol,
                lanczos_maxiter = lanczos_maxiter,
                lanczos_restart = lanczos_restart,
                substep_method  = substep_method,
                step_mode_cfg   = step_mode_cfg,
                step_index      = step_index,
            )
        end
    end

    info.bond_dims_after = [dim(linkind(psi, k)) for k in 1:(N-1)]
    return info
end
