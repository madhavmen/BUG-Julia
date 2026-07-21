"""
    BondUpdateBUG

The single consolidated symmetry-native Basis-Update-and-Galerkin integrator.

Built on Telum/LurCGT for U(1) and non-Abelian symmetric tensors. Every Telum
call in this module is one recorded in `docs/telum_api_contract.md`.
"""
module BondUpdateBUG

using LinearAlgebra, Random, Printf
using LurCGT, Telum

include("symmetric_mps.jl")
include("frame.jl")
include("expv.jl")
include("sectors.jl")
include("augment.jl")
include("gates.jl")
include("kls_step.jl")

export SymMPS, canonical!, move_left!, move_right!, bond_dims, leg_dim,
       product_state, domain_wall_state, neel_state,
       total_sz, sz_expectation, site_expval,
       left_gram, right_gram, left_isometry_defect, right_isometry_defect,
       local_space,
       SECTOR_UP, SECTOR_DOWN,
       BondFrame, bond_frame, frame_theta, two_site_block,
       expv, lanczos_expv, arnoldi_expv, tensor_inner,
       hermitian_tridiagonal_exp_coeffs,
       enable_krylov_log, disable_krylov_log, get_krylov_log, KRYLOV_LOG,
       reachable_sectors, fusion_basis, fuse_spaces, add_charge, dual_charge, sector_dim,
       SectorReport, sector_report, missing_charges, perp_component, align_charge,
       sector_report_right, perp_component_right,
       augmented_left_isometry, augmented_right_isometry, random_sector_seed,
       heisenberg_bond_gate, xx_bond_gate, magnetisation_gate, apply_gate,
       kls_bond_update

end # module
