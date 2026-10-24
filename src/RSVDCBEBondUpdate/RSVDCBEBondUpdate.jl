"""
    RSVDCBEBondUpdate

Controlled bond expansion (CBE) as the basis update of the rank-adaptive BUG integrator, on MPO
environments, following the QSMPSLib routines `RSVDpreBE0SiQS.m`, `CBE1SiQS.m` and
`TDVPSweepCBE1Si.m` (design in `docs/cbe_bug.md`).

A separate submodule: `BondUpdateBUG` is the validated gate-based integrator and is not mutated here.

Entry points: `cbe_bug_step!`, `tdvp_cbe1s_step!`, `tdvp2_step!` (reference), `dmrg_cbe1s!` and
`GroundState.solve!`. `tau = -im*dt` is real time and `tau = -dt` imaginary time.
"""
module RSVDCBEBondUpdate

using LinearAlgebra, Random, Printf
using LurCGT, Telum

using ..BondUpdateBUG: SymMPS, canonical!, bond_dims, leg_dim, local_space,
                       symmetry_mode, BondFrame, bond_frame, frame_theta,
                       expv, hermitian_tridiagonal_exp_coeffs, ROOT_METHODS, overlap,
                       tensor_inner, perp_component, perp_component_right,
                       fusion_basis, reachable_sectors, sector_dim, dual_charge,
                       align_charge, left_isometry_defect, right_isometry_defect,
                       energy, bond_gates, apply_gate, heisenberg_bond_gate,
                       magnetisation, center_bond, parity_bonds,
                       # models/oat.jl: the x-polarised start is assembled from two
                       # `product_state`s (same leg spaces under `:none`, so they add tensor by
                       # tensor) and `S^x_tot` is read out with `site_expval`.
                       product_state, site_expval,
                       # models/spin_xx.jl: the spin-s domain wall on `spin_space`
                       sector_product_state,
                       # models/oat.jl under `:Z2` -- the symmetry arXiv:2208.10972 Fig. 2
                       # actually uses. The sigma^x basis of `z2_ising.jl` is what makes it work:
                       # `S^x` is DIAGONAL there (hence measurable) and the x-polarised start is
                       # a single charge-0 sector. `local_space()` cannot build a Z2 space at
                       # all, so every Z2 tensor has to come from here.
                       ising_local_space, ising_polarised_state

# ── the order of the includes is load-bearing ────────────────────────────────────────────────
# Files are not grouped strictly by directory: `operators/mpo*.jl` adds methods to
# `cbe_expand` and the sketches, so it comes after `expansion/cbe_*.jl`, and the DD-QR CBE
# wraps the MPO operators, so it comes after `operators/zero_site_core.jl`. Every split file
# (`*_stacks`, `*_env`, `*_controller`, ...) is included right after its parent.

include("operators/henv.jl")
include("operators/henv_stacks.jl")
include("budget/multiplets.jl")
# Before `cbe_core.jl`, which derives its rank decisions from the allocation.
include("budget/adaptive_budget.jl")
include("expansion/cbe_core.jl")
include("expansion/cbe_sketch.jl")
include("expansion/cbe_range.jl")
include("expansion/cbe_expand.jl")
include("expansion/cbe_options.jl")
include("expansion/cbe_bond_update.jl")
include("operators/mpo.jl")
include("operators/mpo_env.jl")
include("operators/mpo_fold.jl")
include("operators/zero_site_core.jl")
include("expansion/dd_cbe.jl")
include("expansion/dd_cbe_controller.jl")

# models/ -- `pair_mpo.jl` first: it is the exact MPO for an arbitrary two-body coupling
# matrix (charged vertices included) that every lattice model below is written through.
include("models/pair_mpo.jl")
include("models/haldane_shastry.jl")
include("models/square_cylinder.jl")
include("models/triangular_cylinder.jl")
include("models/kagome.jl")
include("models/breathing_kagome.jl")
include("models/oat.jl")
include("models/tfim.jl")
include("models/mbl_scars.jl")
include("models/pxp_scars.jl")
include("models/hubbard_d4.jl")
include("models/liouville.jl")
include("models/liouville_blocks.jl")
include("models/spin_xx.jl")

# baselines/ -- files of this module rather than submodules: they need the same internals.
# `tdvp2.jl` before `bond_expansion.jl`, which uses its `tensor_elements`.
include("baselines/tdvp2.jl")
include("baselines/tdvp1_cbe.jl")
include("baselines/tdvp1_cbe_variants.jl")
include("expansion/bond_expansion.jl")
include("baselines/dmrg_cbe1s.jl")
include("baselines/dmrg_cbe1s_driver.jl")
include("baselines/tdvp_cbe1s.jl")

# bug/ -- the solver. The step (`cbe_bug.jl`) needs the Krylov enrichment, the half-sweeps and
# its record type before it; the order-p closure, the parallel K/L/S step and imaginary time
# build on the step.
include("bug/krylov_enrich.jl")
include("bug/half_sweeps.jl")
include("bug/cbe_bug_info.jl")
include("bug/cbe_bug.jl")
include("bug/cbe_bug_midpoint.jl")
include("bug/krylov_closure.jl")
include("bug/frontier_sweeps.jl")
include("bug/krylov_closure_loop.jl")
include("bug/parallel_kls.jl")
include("bug/imaginary_time.jl")

# Last: a submodule that imports from the parent, so everything it names must exist already.
include("baselines/ground_state.jl")

# ── exports, by directory ────────────────────────────────────────────────────────────────────
# operators/
export XXZTerm, XXZChain, xxz_chain, heisenberg_su2_chain, hamiltonian_terms, bond_gate,
       ChannelSet, boundary_channels, left_channels, right_channels,
       push_left_channels, push_right_channels,
       LeftEnvStack, RightEnvStack, left_env_stack, right_env_stack, env_energy, env_energy_right,
       apply_h_two_site, sketch_h_left, sketch_h_right,
       MPO, mpo_from_terms, xxz_mpo, heisenberg_su2_mpo, mpo_virtual_dims, mpo_energy, mpo_variance,
       onsite_mpo, mpo_sum, no_jump_mpo,
       LongRangeTerm, long_range_mpo, long_range_zz_mpo,
       LongRangeFit, fit_long_range, default_decays, long_range_terms, power_law_zz_mpo,
       MPOLink, MPOLeftEnvStack, MPORightEnvStack, MPOZeroSiteH, MPOOneSiteH,
       shift_h, ZeroSiteH, zero_site_h, apply_zero_site, truncate_sweep!, truncate_recursive!

# budget/ -- the three-tolerance budget, the ledger that reports when a rank ceiling rather
# than a tolerance ended a search, and multiplet-vs-state accounting (`maxdim` is in
# MULTIPLETS, so cross-symmetry comparisons at fixed rank must convert)
export AdaptiveBudget, BasisAllocation, basis_allocation, indicator,
       AdaptiveLedger, merge_ledger!, work_ceiling, budget_status, budget_diagnostic,
       state_dim, state_bond_dims, multiplet_ratio

# expansion/ -- RSVD CBE, the DD-QR CBE with its operator interface and step record, and the
# per-bond expansion
export sector_graded_sketch, full_local_basis, CBEExpansion, cbe_expand,
       CBEBugOptions, CBEBugRunInfo, SStepSolver, Propagate, LowestEigen,
       sketch_bond_left, sketch_bond_right, cbe_expand_bond, cbe_bond_update,
       dd_cbe_expand, dd_reference_sketch,
       TwoSiteSketchOp, OneSiteOp, RootOp,
       apply_two_site_sketch!, apply_one_site!, apply_root!,
       KernelCounters, StepRecord,
       BondExpansionInfo, expand_bond!, lanczos_lowest

# models/
export PairVertex, pair_vertices, pair_mpo,
       nn_coupling_matrix, distance_coupling_matrix, spin_vertices,
       haldane_shastry_couplings, haldane_shastry_mpo,
       heisenberg_chain_mpo, heisenberg_ring_couplings, heisenberg_ring_mpo,
       square_cylinder_couplings, square_cylinder_mpo, square_cylinder_bonds,
       triangular_cylinder_couplings, triangular_cylinder_mpo,
       triangular_cylinder_bonds, triangular_coordination,
       triangular_site_position, triangular_wrap_vector,
       kagome_cylinder_couplings, kagome_cylinder_mpo, kagome_bonds, kagome_triangles,
       breathing_kagome_positions, breathing_kagome_couplings, breathing_kagome_mpo,
       # oat.jl -- `x_polarized_state`, `sx_operator`, `sx_profile` and `total_sx` are
       # :none-only BY PHYSICS: S^x is charge-raising, so a U(1) run reports zero.
       oat_couplings, oat_energy_shift, oat_mpo, oat_mpo_pairs,
       x_polarized_state, sx_operator, sx_profile, total_sx,
       tfim_mpo,
       # mbl_scars.jl / pxp_scars.jl -- disordered and quasiperiodic XXZ (U(1)), the
       # interleaved Aubry-André-Hubbard chain, and PXP (:none)
       disorder_fields, field_mpo_from_terms, disordered_xxz_mpo,
       quasiperiodic_fields, aubry_andre_xxz_mpo,
       hubbard_aa_mpo, hubbard_cdw_state,
       pxp_operators, pxp_mpo, z2_state, zk_state,
       # hubbard_d4.jl -- the fused d=4 spinful chain; `hubbard_aa_mpo` above cannot be evolved
       # from a product state (range-2 hops => H_eff == 0 on every bond)
       hub4_space, hub4_basis_index, hub4_sector, hub4_product_state,
       HUB4_EMPTY, HUB4_UP, HUB4_DOWN, HUB4_BOTH,
       hubbard_aa_mpo_d4, hub4_density, hub4_imbalance, hub4_cdw_occupations,
       # liouville*.jl -- symmetry-aware MPDOs
       LiouvilleSpace, liouville_space, liouville_index, liouville_pair, liouville_charge,
       liouville_op, ket_super, bra_super, dissipator_super, lindblad_mpo, mpdo_apply!,
       mpdo_product_state, mpdo_trace, mpdo_expect, mpdo_rdm, mpdo_dense,
       mpdo_normalise_trace!, mpdo_checks, mpdo_identity, mpdo_identity_sector,
       spin_space, spin_xx_mpo, spin_current_mpo, spin_domain_wall_state, spin_sz_profile

# baselines/
export OneSiteH, one_site_h, apply_one_site, TDVP2Info, tdvp2_step!, tdvp2_composed_step!,
       tensor_elements,
       TDVP1CBEInfo, tdvp1_cbe_step!, tdvp1_rsvd_cbe_step!, tdvp1_cbe_exact_step!,
       TDVPCBEInfo, tdvp_cbe1s_step!,
       DMRGCBEInfo, DMRGCBERun, dmrg_cbe1s_sweep!, dmrg_cbe1s!,
       rsvd_cbe_dmrg1s!, exact_cbe_dmrg1s!, rsvd_cbe_dmrg1s_seeded!,
       GroundState

# bug/
export CBEBugSweepInfo, cbe_bug_step!, cbe_bug_midpoint_step!, parallel_kls_step!,
       ite_step!, ite_evolve!, log_panel_steps, uniform_beta_steps, beta_grid

end # module
