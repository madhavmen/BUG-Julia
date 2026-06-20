# tests/BUG/test_bug_kls.jl
#
# Unit tests for faithful KLS steps: K/L augmentation (isometry property),
# S-step advance, and S-step truncation.
# Tolerances at machine precision for algebraic properties.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "BUG", "BUG.jl"))
using .BUG

function _make_xx_mpo(sites)
    os = OpSum()
    for i in 1:(length(sites)-1)
        os += 0.5, "S+", i, "S-", i+1
        os += 0.5, "S-", i, "S+", i+1
    end
    return TensorTrainOperator(MPO(os, sites))
end

@testset "Augmentation: _pick_left_update" begin
    # Left update should return an orthonormal column matrix
    d  = 4;  rm = 2
    U0 = qr(rand(ComplexF64, d, rm)).Q |> Matrix
    K1 = rand(ComplexF64, d, rm) |> x -> qr(x).Q |> Matrix

    A_aug, n_new, overlap = BUG._pick_left_update(U0, K1;)
    @test size(A_aug, 1) == d
    @test size(A_aug, 2) >= rm
    # Orthonormality
    @test norm(A_aug' * A_aug - I) < 1e-12
    # Overlap preserves old basis
    @test norm(A_aug * overlap - U0) < 1e-12
end

@testset "Augmentation: _pick_right_update" begin
    d  = 4;  rm = 2
    # V0: rm row-isometries of length d  (rm × d)
    V0 = Matrix(qr(rand(ComplexF64, d, rm)).Q)'
    # L1: evolved L tensor in row form (rm × d); must have same ncols as V0
    L1 = rand(ComplexF64, rm, d)

    B_aug, n_new, overlap = BUG._pick_right_update(V0, L1;)
    @test size(B_aug, 2) == d
    @test size(B_aug, 1) >= rm
    # Row orthonormality
    @test norm(B_aug * B_aug' - I) < 1e-12
end

@testset "_augmented_left_isometry_from_k isometry" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 41)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)

    K1_tens = snap.U0_tens * snap.S0_tens   # shape: (link_l, site_l, canon_v0)
    # evolved_right is the right index of K1_tens = the V-side bond (canon_v0)
    evolved_right = snap.canon_v0
    U1, n_new = BUG._augmented_left_isometry_from_k(
        snap.U0_tens, K1_tens,
        snap.link_l, snap.site_l, snap.canon_u0, evolved_right;
    )
    mid = _left_site_bond_index(U1, snap.link_l, snap.site_l)
    U1_mat = reshape(ComplexF64.(Array(U1, snap.link_l, snap.site_l, mid)),
                     dim(snap.link_l)*dim(snap.site_l), dim(mid))
    @test norm(U1_mat' * U1_mat - I) < 1e-10
end

@testset "_augmented_right_isometry_from_l isometry" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 43)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)

    L1_tens = snap.S0_tens * snap.V0_tens   # shape: (canon_u0, site_r, link_r)
    # evolved_left is the left index of L1_tens = the U-side bond (canon_u0)
    evolved_left = snap.canon_u0
    V1, n_new = BUG._augmented_right_isometry_from_l(
        snap.V0_tens, L1_tens,
        snap.canon_v0, evolved_left, snap.site_r, snap.link_r;
    )
    mid = _site_bond_index(V1, snap.site_r, snap.link_r)
    V1_mat = reshape(ComplexF64.(Array(V1, mid, snap.site_r, snap.link_r)),
                     dim(mid), dim(snap.site_r)*dim(snap.link_r))
    @test norm(V1_mat * V1_mat' - I) < 1e-10
end

@testset "_transported_s_start preserves norm" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 45)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)

    # Identity transport: rename bond index so aug != old to avoid index aliasing.
    aug_u = sim(snap.canon_u0)
    aug_v = sim(snap.canon_v0)
    U_aug = replaceind(snap.U0_tens, snap.canon_u0, aug_u)
    V_aug = replaceind(snap.V0_tens, snap.canon_v0, aug_v)
    transported = BUG._transported_s_start_from_augmented_bases(
        U_aug, snap.U0_tens, snap.S0_tens, snap.V0_tens, V_aug,
    )
    # Identity transport preserves the Frobenius norm of S
    @test norm(transported.S_transport) ≈ norm(snap.S0_tens) rtol=1e-10
end

@testset "S-step advance: S-step output shape" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 47)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)
    HW    = snap.L_mpo_cur * snap.W_left * snap.W_right * snap.R_mpo_cur

    result = BUG._advance_s_tensor_in_bases(
        snap.U0_tens, snap.V0_tens, snap.S0_tens, HW, -0.01im;
        lanczos_tol = 1e-12, lanczos_maxiter = 30, substep_method = :expv,
        matrixfree_sstep = false,
    )
    @test hasfield(typeof(result), :S_new)
    @test inds(result.S_new) == inds(snap.S0_tens)
end

@testset "S-step truncation" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 4, seed = 49)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)
    HW    = snap.L_mpo_cur * snap.W_left * snap.W_right * snap.R_mpo_cur

    sstep = BUG._advance_s_tensor_in_bases(
        snap.U0_tens, snap.V0_tens, snap.S0_tens, HW, -0.01im;
        lanczos_tol = 1e-12, lanczos_maxiter = 30, substep_method = :expv,
        matrixfree_sstep = false,
    )
    s_inds = inds(sstep.S_new)
    U_s, SV, keep, svals = BUG._truncate_quantum_s_step(
        sstep.S_new, s_inds[1], s_inds[2];
        maxdim = 10,
    )
    @test keep == min(length(svals), 10)
    # U_s should be column-isometric: U_s' * U_s = I
    p_ind = commonind(U_s, SV)
    left_ind = only(setdiff(inds(U_s), [p_ind]))
    U_s_mat = reshape(ComplexF64.(Array(U_s, left_ind, p_ind)), dim(left_ind), dim(p_ind))
    @test norm(U_s_mat' * U_s_mat - I) < 1e-10
end

@testset "S-step truncation: trunc_thresh (relative SVD cutoff)" begin
    # _svd_keep_count mirrors the Python two_site_bug rule:
    #   thresh = cutoff * |s[1]|;  keep = max(count(|s| > thresh), 1);  min(keep, maxdim, n)
    svals = [1.0, 0.5, 1e-3, 1e-8, 1e-14]
    @test BUG._svd_keep_count(svals, 100, 0.0)  == 5      # cutoff off -> all
    @test BUG._svd_keep_count(svals, 100, 1e-2) == 2      # drop 1e-3,1e-8,1e-14
    @test BUG._svd_keep_count(svals, 100, 1e-6) == 3      # keep down to 1e-3
    @test BUG._svd_keep_count(svals, 2,   0.0)  == 2      # maxdim cap dominates
    @test BUG._svd_keep_count(svals, 100, 10.0) == 1      # everything below thresh -> at least 1
    @test BUG._svd_keep_count(Float64[], 4, 1e-6) == 0    # empty spectrum

    # End-to-end on the truncation function: a coarse cutoff keeps fewer
    # singular values than the pure-maxdim call on the same S tensor.
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 4, seed = 49)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)
    HW    = snap.L_mpo_cur * snap.W_left * snap.W_right * snap.R_mpo_cur
    sstep = BUG._advance_s_tensor_in_bases(
        snap.U0_tens, snap.V0_tens, snap.S0_tens, HW, -0.01im;
        lanczos_tol = 1e-12, lanczos_maxiter = 30, substep_method = :expv,
        matrixfree_sstep = false,
    )
    s_inds = inds(sstep.S_new)
    _, _, keep_full, svals_t = BUG._truncate_quantum_s_step(sstep.S_new, s_inds[1], s_inds[2]; maxdim = 10, cutoff = 0.0)
    _, _, keep_cut,  _       = BUG._truncate_quantum_s_step(sstep.S_new, s_inds[1], s_inds[2]; maxdim = 10, cutoff = 1e-1)
    @test keep_cut ≤ keep_full
    @test keep_cut == BUG._svd_keep_count(svals_t, 10, 1e-1)
end

@testset "Faithful KLS candidate: output fields" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 51)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)
    HW    = snap.L_mpo_cur * snap.W_left * snap.W_right * snap.R_mpo_cur

    cand = BUG._faithful_kls_local_bond_candidate(snap;
        dt = -0.01im,
        lanczos_tol = 1e-12, lanczos_maxiter = 30, substep_method = :expv,
        matrixfree_sstep = false, HW_env_override = HW,
    )
    @test hasfield(typeof(cand), :U_aug_tens)
    @test hasfield(typeof(cand), :V_aug_tens)
    @test hasfield(typeof(cand), :S_new)
    @test cand.n_new_k >= 0
    @test cand.n_new_l >= 0
end

@testset "Faithful reverse KLS candidate: U_aug isometry" begin
    N     = 5
    sites = siteinds("S=1/2", N)
    psi   = random_tt(sites; maxdim = 3, seed = 53)
    W     = _make_xx_mpo(sites)
    orthogonalize!(psi, 1)
    bond  = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap  = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)
    HW    = snap.L_mpo_cur * snap.W_left * snap.W_right * snap.R_mpo_cur

    cand = BUG._faithful_reverse_kls_local_bond_candidate(snap;
        dt = -0.01im,
        lanczos_tol = 1e-12, lanczos_maxiter = 30, substep_method = :expv,
        matrixfree_sstep = false, HW_env_override = HW,
    )
    U_aug = cand.U_aug_tens
    mid   = _left_site_bond_index(U_aug, snap.link_l, snap.site_l)
    U_aug_mat = reshape(ComplexF64.(Array(U_aug, snap.link_l, snap.site_l, mid)),
                        dim(snap.link_l)*dim(snap.site_l), dim(mid))
    @test norm(U_aug_mat' * U_aug_mat - I) < 1e-10
end
