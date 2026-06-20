# tdvp_local.jl
#
# Local 1-site and 0-site TDVP update steps.
# Each site update:
#   1. Builds and applies the 1-site effective Hamiltonian (forward: +dt, reverse: +dt)
#   2. Performs QR (forward) or LQ (reverse) gauge move
#   3. Builds and applies the 0-site bond Hamiltonian with -dt (backward step)
#
# The Krylov exponential is supplied by TTutils._linear_substep.

# ── QR/LQ gauge moves ─────────────────────────────────────────────────────────

"""
    _tdvp_forward_qr(A, link_l, site_k, link_tag) -> (Q, C, canon_link)

QR-factorize tensor A with Q ← left-isometric and C ← right (bond center).
"""
function _tdvp_forward_qr(
    A::ITensor,
    link_l::Index,
    site_k::Index,
    link_tag::AbstractString,
)
    Q, C = qr(A, link_l, site_k; tags=link_tag, positive=false)
    return Q, C, commonind(Q, C)
end

"""
    _tdvp_reverse_lq(A, site_k, link_r, link_tag) -> (C, Q, canon_link)

LQ-factorize tensor A with Q ← right-isometric and C ← left (bond center).
"""
function _tdvp_reverse_lq(
    A::ITensor,
    site_k::Index,
    link_r::Index,
    link_tag::AbstractString,
)
    C, Q = lq(A, site_k, link_r)
    old_link = commonind(C, Q)
    new_link = settags(old_link, link_tag)
    C = replaceind(C, old_link, new_link)
    Q = replaceind(Q, old_link, new_link)
    return C, Q, new_link
end

# ── Site update (1-site Hamiltonian) ─────────────────────────────────────────

"""
    _tdvp_site_update(psi, k, dt, L_mpo, R_mpo, W; kwargs...) -> A_new, numops

Evolve site tensor `psi[k]` under the local 1-site effective Hamiltonian.
Returns the updated ITensor and Krylov work count.
"""
function _tdvp_site_update(
    psi::TensorTrain,
    k::Int,
    dt::Number,
    L_mpo::ITensor,
    R_mpo::ITensor,
    W_k::ITensor;
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
)
    N      = length(psi)
    link_l = k == 1 ? linkinds(psi)[1] : commonind(psi[k-1], psi[k])
    site_k = siteinds(psi, k)
    link_r = k == N ? linkinds(psi)[N+1] : commonind(psi[k], psi[k+1])

    H_site = _build_1site_Heff_mat(link_l, site_k, link_r, L_mpo, R_mpo, W_k)
    a_vec  = _complex_tensor_vec(psi[k], link_l, site_k, link_r)
    a_new_vec, numops = _linear_substep(H_site, dt, a_vec;
        method = substep_method,
        lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter,
        restart = lanczos_restart,
    )
    return itensor(reshape(a_new_vec, dim(link_l), dim(site_k), dim(link_r)),
                   link_l, site_k, link_r), numops
end

# ── Bond backward step (0-site Hamiltonian) ───────────────────────────────────

"""
    _tdvp_bond_backward(C, left_ind, right_ind, dt, L_mpo, R_mpo; kwargs...) -> C_new, numops

Backward-evolve bond center tensor C under the 0-site effective Hamiltonian with `-dt`.
"""
function _tdvp_bond_backward(
    C::ITensor,
    left_ind::Index,
    right_ind::Index,
    dt::Number,
    L_mpo::ITensor,
    R_mpo::ITensor;
    lanczos_tol::Float64,
    lanczos_maxiter::Int,
    lanczos_restart::Int,
    substep_method::Symbol,
)
    H_bond = _build_0site_Heff_mat(left_ind, right_ind, L_mpo, R_mpo)
    c_vec  = _complex_tensor_vec(C, left_ind, right_ind)
    c_new_vec, numops = _linear_substep(H_bond, -dt, c_vec;
        method = substep_method,
        lanczos_tol = lanczos_tol,
        lanczos_maxiter = lanczos_maxiter,
        restart = lanczos_restart,
    )
    return itensor(reshape(c_new_vec, dim(left_ind), dim(right_ind)),
                   left_ind, right_ind), numops
end
