using Test
using ITensors
using ITensorMPS
using LinearAlgebra

const _SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "BUG", "BUG.jl"))
using .BUG

function _make_xx_mpo_local_sym(sites)
    os = OpSum()
    for i in 1:(length(sites)-1)
        os += 0.5, "S+", i, "S-", i+1
        os += 0.5, "S-", i, "S+", i+1
    end
    return TensorTrainOperator(MPO(os, sites))
end

@testset "Local KLS forward/reverse use the same dt sign" begin
    N = 4
    sites = siteinds("S=1/2", N)
    W = _make_xx_mpo_local_sym(sites)
    psi = random_tt(sites; maxdim = 3, seed = 81)
    normalize!(psi)
    orthogonalize!(psi, 1)

    bond = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)
    HW = snap.L_mpo_cur * snap.W_left * snap.W_right * snap.R_mpo_cur

    cand_f = BUG._faithful_kls_local_bond_candidate(
        snap;
        dt = -0.01im,
        lanczos_tol = 1e-12,
        lanczos_maxiter = 30,
        substep_method = :expv,
        matrixfree_sstep = false,
        HW_env_override = HW,
    )
    cand_r = BUG._faithful_reverse_kls_local_bond_candidate(
        snap;
        dt = -0.01im,
        lanczos_tol = 1e-12,
        lanczos_maxiter = 30,
        substep_method = :expv,
        matrixfree_sstep = false,
        HW_env_override = HW,
    )

    theta_f = cand_f.U_aug_tens * cand_f.S_new * cand_f.V_aug_tens
    theta_r = cand_r.U_aug_tens * cand_r.S_new * cand_r.V_aug_tens
    theta_diff = Array(theta_f - theta_r, snap.link_l, snap.site_l, snap.site_r, snap.link_r)
    # The local reverse candidate uses the same bond-update sign as the forward
    # candidate; the adjoint structure lives in the sweep ordering, not a local
    # sign flip.
    @test norm(theta_diff) < 1e-10
end
