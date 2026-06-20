# mpo.jl
#
# TensorTrainOperator (MPO) type, constructors, and MPO environment helpers.
# The environment helpers are needed by both TDVP and BUG integrators.

"""
    _partition_ranges(num_bonds, npartitions) -> Vector{UnitRange{Int}}

Split `num_bonds` bonds into at most `npartitions` contiguous ranges.
Used by concurrent-partition sweep implementations (BUG and TDVP).
"""
function _partition_ranges(num_bonds::Int, npartitions::Int)
    npartitions >= 1 || error("npartitions must be >= 1; got $npartitions")
    npartitions = min(npartitions, max(num_bonds, 1))
    base, extra = divrem(num_bonds, npartitions)
    ranges = Vector{UnitRange{Int}}()
    start = 1
    for part in 1:npartitions
        len = base + (part <= extra ? 1 : 0)
        len == 0 && continue
        stop = start + len - 1
        push!(ranges, start:stop)
        start = stop + 1
    end
    return ranges
end

const _QM_TTT_MPO = joinpath(@__DIR__, "TensorTrainTools")

include(joinpath(_QM_TTT_MPO, "tensortrainoperator.jl"))

# ── ITensors MPO → TensorTrainOperator conversion ─────────────────────────────
# ITensors MPO boundary tensors are rank-3 (no dim-1 boundary link). We add
# explicit dim-1 boundary links so _mpo_boundary_link finds them correctly.
function TensorTrainOperator(mpo::MPO)
    N = length(mpo)
    tensors = ITensor[mpo[i] for i in 1:N]
    # Add left boundary link to site 1 if it only has one Link index
    links_1 = filter(hastags("Link"), inds(tensors[1]))
    if length(links_1) < 2
        lnk_l = Index(1, "Link,l=0")
        tensors[1] = tensors[1] * itensor([1.0], lnk_l)
    end
    # Add right boundary link to site N if it only has one Link index
    links_N = filter(hastags("Link"), inds(tensors[N]))
    if length(links_N) < 2
        lnk_r = Index(1, "Link,l=$N")
        tensors[N] = tensors[N] * itensor([1.0], lnk_r)
    end
    return TensorTrainOperator(tensors)
end

# ── MPO environment helpers (shared by BUG and TDVP) ─────────────────────────
# These functions build left/right MPO contraction environments that are used
# in both the BUG K/L/S steps and the TDVP 1-site/0-site updates.

"""
    _mpo_boundary_link(W, side) -> Index

Return the left or right MPO boundary link that remains open in environment contractions.
"""
function _mpo_boundary_link(W::TensorTrainOperator, side::Symbol)
    N = length(W)
    N < 2 && error("_mpo_boundary_link requires at least two sites")
    side === :left  && return only(uniqueinds(W[1], W[2]; tags="Link"))
    side === :right && return only(uniqueinds(W[N], W[N-1]; tags="Link"))
    error("side must be :left or :right")
end

function _mpo_identity_transfer(W::TensorTrainOperator, site::Int)
    s_bra = siteinds(W, site; plev=0)
    s_ket = siteinds(W, site; plev=1)
    return W[site] * delta(s_bra, s_ket)
end

function _mpo_identity_left_channel(W::TensorTrainOperator, site::Int)
    1 <= site <= length(W) || error("site out of range")
    channel = itensor([1.0], _mpo_boundary_link(W, :left))
    for k in 1:(site - 1)
        channel = channel * _mpo_identity_transfer(W, k)
    end
    return channel
end

function _mpo_identity_right_channel(W::TensorTrainOperator, site::Int)
    1 <= site <= length(W) || error("site out of range")
    channel = itensor([1.0], _mpo_boundary_link(W, :right))
    for k in length(W):-1:(site + 1)
        channel = channel * _mpo_identity_transfer(W, k)
    end
    return channel
end

"""
    _left_env_boundary_mpo(psi, W) -> ITensor

Return the trivial left boundary environment for a BUG or TDVP forward sweep.
"""
function _left_env_boundary_mpo(psi::TensorTrain, W::TensorTrainOperator)
    link_l = only(uniqueinds(psi[1], psi[2]; tags="Link"))
    mpo_l  = _mpo_boundary_link(W, :left)
    return delta(prime(link_l), link_l) * itensor([1.0], mpo_l)
end

"""
    _right_env_boundary_mpo(psi, W) -> ITensor

Return the trivial right boundary environment for a reverse sweep.
"""
function _right_env_boundary_mpo(psi::TensorTrain, W::TensorTrainOperator)
    N      = length(psi)
    link_r = only(uniqueinds(psi[N], psi[N-1]; tags="Link"))
    mpo_r  = _mpo_boundary_link(W, :right)
    return delta(prime(link_r), link_r) * itensor([1.0], mpo_r)
end

"""
    _advance_left_env_mpo(L_cur, psi_site, W_site) -> ITensor

Grow the left MPO environment by one site to the right.
"""
function _advance_left_env_mpo(L_cur::ITensor, psi_site::ITensor, W_site::ITensor)
    return L_cur * dag(prime(psi_site)) * W_site * psi_site
end

"""
    _advance_right_env_mpo(R_cur, psi_site, W_site) -> ITensor

Grow the right MPO environment by one site to the left.
"""
function _advance_right_env_mpo(R_cur::ITensor, psi_site::ITensor, W_site::ITensor)
    return R_cur * dag(prime(psi_site)) * W_site * psi_site
end

"""
    _build_right_envs_mpo(psi, W) -> Vector{ITensor}

Build all right MPO environments R[k] for a left-to-right sweep.
    R[k] contracts everything strictly to the right of site k-2.
"""
function _build_right_envs_mpo(psi::TensorTrain, W::TensorTrainOperator)
    N   = length(psi)
    R   = Vector{ITensor}(undef, N + 2)
    R[N + 2] = _right_env_boundary_mpo(psi, W)
    for k in N:-1:1
        R[k + 1] = _advance_right_env_mpo(R[k + 2], psi[k], W[k])
    end
    return R
end

"""
    _build_left_envs_mpo(psi, W) -> Vector{ITensor}

Build all left MPO environments L[k] for a right-to-left sweep.
L[k] contracts everything strictly to the left of site k+1.
"""
function _build_left_envs_mpo(psi::TensorTrain, W::TensorTrainOperator)
    N   = length(psi)
    L   = Vector{ITensor}(undef, N + 2)
    L[1] = _left_env_boundary_mpo(psi, W)
    for k in 1:N
        L[k + 1] = _advance_left_env_mpo(L[k], psi[k], W[k])
    end
    return L
end

"""
    _build_right_envs_mpo_for_bonds(psi, W, bonds) -> Vector{ITensor}

Build right MPO environments for a specific sorted list of bond indices.
Returns one environment per bond in the same order.
"""
function _build_right_envs_mpo_for_bonds(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bonds::Vector{Int},
)
    isempty(bonds) && return ITensor[]
    N   = length(psi)
    R_all = _build_right_envs_mpo(psi, W)
    # R[bond+3] is the right env for a 2-site bond update at position `bond`
    return [R_all[b + 3] for b in bonds]
end

"""
    _build_left_envs_mpo_for_bonds(psi, W, bonds) -> Vector{ITensor}

Build left MPO environments for a specific sorted list of bond indices.
"""
function _build_left_envs_mpo_for_bonds(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bonds::Vector{Int},
)
    isempty(bonds) && return ITensor[]
    L_all = _build_left_envs_mpo(psi, W)
    # L[bond] is the left env for a bond update at position `bond`
    return [L_all[b] for b in bonds]
end

"""
    _owned_two_site_mpo_envs(psi, W, bond) -> (L, R)

Compute the left and right MPO environments for bond `bond` using full
MPO contraction via the incremental environment builders. Returns
(L_all[bond], R_all[bond+3]) from `_build_left_envs_mpo` /
`_build_right_envs_mpo`.
"""
function _owned_two_site_mpo_envs(
    psi::TensorTrain,
    W::TensorTrainOperator,
    bond::Int,
)
    L_all = _build_left_envs_mpo(psi, W)
    R_all = _build_right_envs_mpo(psi, W)
    return L_all[bond], R_all[bond + 3]
end

# ── 1-site and 0-site effective Hamiltonian builders (used by TDVP) ──────────

"""
    _build_1site_Heff_mat(link_l, site, link_r, L_mpo, R_mpo, W) -> Matrix{ComplexF64}

Materialize the dense 1-site effective Hamiltonian for TDVP site evolution.
"""
function _build_1site_Heff_mat(
    link_l::Index,
    site_k::Index,
    link_r::Index,
    L_mpo::ITensor,
    R_mpo::ITensor,
    W_k::ITensor,
)
    d_l  = dim(link_l)
    d_s  = dim(site_k)
    d_r  = dim(link_r)
    d_tot = d_l * d_s * d_r
    HW_env = complex(L_mpo) * W_k * R_mpo
    H_mat = zeros(ComplexF64, d_tot, d_tot)
    bra_inds = (prime(link_l), prime(site_k), prime(link_r))
    e_vec = zeros(ComplexF64, d_tot)
    for col in 1:d_tot
        fill!(e_vec, 0.0 + 0.0im)
        e_vec[col] = 1.0
        A_col = itensor(reshape(e_vec, d_l, d_s, d_r), link_l, site_k, link_r)
        HA_col = HW_env * A_col
        H_mat[:, col] = vec(Array(HA_col, bra_inds...))
    end
    return H_mat
end

"""
    _build_0site_Heff_mat(left_ind, right_ind, L_mpo, R_mpo) -> Matrix{ComplexF64}

Materialize the dense 0-site effective Hamiltonian for TDVP bond (center) evolution.
"""
function _build_0site_Heff_mat(
    left_ind::Index,
    right_ind::Index,
    L_mpo::ITensor,
    R_mpo::ITensor,
)
    d_left  = dim(left_ind)
    d_right = dim(right_ind)
    d_tot   = d_left * d_right
    HW_env  = complex(L_mpo) * R_mpo
    H_eff   = zeros(ComplexF64, d_tot, d_tot)
    bra_inds = (prime(left_ind), prime(right_ind))
    e_vec = zeros(ComplexF64, d_tot)
    for col in 1:d_tot
        fill!(e_vec, 0.0 + 0.0im)
        e_vec[col] = 1.0
        C_col = itensor(reshape(e_vec, d_left, d_right), left_ind, right_ind)
        HC_col = HW_env * C_col
        H_eff[:, col] = vec(Array(HC_col, bra_inds...))
    end
    return H_eff
end
