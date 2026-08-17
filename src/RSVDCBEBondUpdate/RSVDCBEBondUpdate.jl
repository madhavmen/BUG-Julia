"""
    RSVDCBEBondUpdate

Exploratory integrator: **controlled bond expansion (CBE) as the BUG basis update**.

Follows the von Delft group's QSMPSLib routines `RSVDpreBE0SiQS.m` (randomized-SVD
bond expansion before a 0-site update), `CBE1SiQS.m` and `TDVPSweepCBE1Si.m`. The
design, and the reasons each piece of `BondUpdateBUG` it replaces is being replaced,
are in `docs/cbe_bug.md`.

Deliberately a SEPARATE submodule. `BondUpdateBUG.bond_update_bug!` is the validated
gate-based Trotter integrator and the unitarity-critical path; nothing here mutates
it. This module reuses its containers and primitives (`SymMPS`, `BondFrame`, `expv`,
the sector tooling) by import only.

Contents, in dependency order:

  - `henv.jl`  — channel environments for a nearest-neighbour XXZ chain: `H` globally,
    which neither the CBE selection nor a single centre-bond Galerkin step can do
    without. Replaces the per-bond gate the Trotter sweep uses.
  - `mpo.jl` — the same `H` as a GENUINE MPO: one rank-4 `W^[i]` per site (Eq. 1.3 of
    arXiv:2606.28169) and one rank-3 `(bra, mpo, ket)` environment per link (Eq. 1.8), from
    which the effective Hamiltonians follow as a single contraction (Eq. 1.7). This is the
    representation the parallel-BUG sweep of that paper is written in, and it is what
    `cbe_lubich_sweep` runs on; `henv.jl` is kept as the independent witness that the two
    are one operator (`test_mpo.jl` pins them elementwise). Unlike the factored form it is
    site-dependent, so it is not restricted to a uniform nearest-neighbour term list.

  - `cbe_core.jl` — THE MAIN CODE, and what all three modes share. Part I is the RSVD bond
    expansion (sector-graded sketch, preselection, final selection); Part II drives it from
    a bond frame and solves the local problem in the expanded basis (`_expanding_krylov`),
    plus the shared `CBEBugOptions` / `CBEBugRunInfo`.
  - `cbe_lubich.jl` — `cbe_lubich_sweep`: basis sweep + one centre-bond Galerkin step, on
    the environment-dressed effective `H`.
  - `modes/` — the three modes. Each is a THIN submodule: it fixes the operator the local
    problem is posed with and the reduction applied to the Krylov tridiagonal, and adds no
    algorithm of its own.

THE THREE MODES, and what actually distinguishes them:

| mode             | `applyH` at a bond      | S-step reduction        | path        |
|------------------|-------------------------|-------------------------|-------------|
| `RealTime`       | gate, or env-dressed H  | `exp(-i dt H)`          | gate + MPO  |
| `ImaginaryTime`  | gate, or env-dressed H  | `exp(-dt H)`            | gate + MPO  |
| `GroundState`    | env-dressed H ONLY      | lowest Ritz vector      | MPO         |

The Krylov space, and every RSVD-CBE growth pass that builds it, is IDENTICAL in all three:
the ground-state solver is a drop-in replacement for the S step, not a second algorithm.

Why `GroundState` is MPO-only: the gate path's environments are the identity by canonical
form, which is exact for a Trotter factor but means the local operator is the bare two-site
term. Minimising that at each bond independently does not have the global ground state as
its fixed point. Imaginary time is unaffected -- `exp(-dt H)` Trotterises exactly as
`exp(-im dt H)` does -- so it runs on both paths.
"""
module RSVDCBEBondUpdate

using LinearAlgebra, Random, Printf
using LurCGT, Telum

using ..BondUpdateBUG: SymMPS, canonical!, bond_dims, leg_dim, local_space,
                       symmetry_mode, BondFrame, bond_frame, frame_theta,
                       expv, hermitian_tridiagonal_exp_coeffs,
                       tensor_inner, perp_component, perp_component_right,
                       fusion_basis, reachable_sectors, sector_dim, dual_charge,
                       align_charge, left_isometry_defect, right_isometry_defect,
                       energy, bond_gates, apply_gate, heisenberg_bond_gate,
                       magnetisation, center_bond, parity_bonds

include("henv.jl")
include("cbe_core.jl")
# The Hamiltonian as a genuine MPO (rank-4 `W^[i]`) plus its rank-3 environments, i.e. the
# objects arXiv:2606.28169 Eqs. (1.3), (1.7) and (1.8) name. It provides a METHOD for every
# verb `cbe_lubich_sweep` uses -- `boundary_channels`, `left/right_env_stack`,
# `push_left/right_channels`, `apply_h_two_site`, `sketch_h_*`, `cbe_expand`,
# `zero_site_h` -- so the sweep runs on either representation unchanged and the channel
# path in `henv.jl` stays as the independent witness that the two are one operator.
# After `cbe_core.jl`, since it adds methods to `cbe_expand` and the sketches.
include("mpo.jl")
include("cbe_lubich.jl")
# Jan BUG: the full KLS step at EVERY bond across the whole chain -- CBE for the basis update and
# the 0-site S step taken as well, then QR'd with `R` discarded so no evolution accumulates -- and
# at the far bond the S step is KEPT, which is the step. One truncation after the sweep,
# direction alternating between steps. MPO-only, so a long-range or site-dependent H needs a
# different `MPO` and nothing else. After `cbe_lubich.jl`, whose `_frame_from` /
# `_absorb_left!` / `_absorb_right!` it reuses.
include("jan_bug.jl")

# The two TDVP integrators live in their own directories. They are FILES of this module, not
# submodules: both are built on `henv.jl` (environments) and `cbe_core.jl` (the expansion)
# above, so splitting them into modules would only add import ceremony around code that has
# to see the same internals anyway.
#
#   tdvp2/       2-site TDVP, the accuracy/memory baseline. No CBE -- the two-site block
#                supplies its own rank, which is exactly what makes it the baseline.
#   tdvp1_cbe/   1-site TDVP + CBE, where CBE is MANDATORY rather than optional: a 1-site
#                sweep cannot change the bond dimension by itself, so without the expansion
#                the rank is frozen at whatever it started as. `variants.jl` names the two
#                selection rules (plain CBE and the RSVD extension) over one implementation.
include("tdvp2/tdvp2_baseline.jl")
include("tdvp1_cbe/tdvp1_cbe.jl")
include("tdvp1_cbe/variants.jl")

# The modes come last: each imports from the parent, so everything they name must already be
# defined. They are submodules rather than files so `RealTime.evolve!` and
# `GroundState.solve!` read as what they are at the call site.
include("modes/real_time.jl")
include("modes/imaginary_time.jl")
include("modes/ground_state.jl")

export XXZTerm, XXZChain, xxz_chain, heisenberg_su2_chain, hamiltonian_terms, bond_gate,
       MPO, mpo_from_terms, xxz_mpo, heisenberg_su2_mpo, mpo_virtual_dims, mpo_energy,
       LongRangeTerm, long_range_mpo, long_range_zz_mpo,
       LongRangeFit, fit_long_range, default_decays, long_range_terms, power_law_zz_mpo,
       MPOLink, MPOLeftEnvStack, MPORightEnvStack, MPOZeroSiteH, MPOOneSiteH,
       JanBugInfo, jan_bug_step!, jan_bug!,
       LeftEnvStack, RightEnvStack,
       left_env_stack, right_env_stack, env_energy, env_energy_right,
       ChannelSet, boundary_channels, left_channels, right_channels,
       push_left_channels, push_right_channels,
       apply_h_two_site,
       sketch_h_left, sketch_h_right,
       sector_graded_sketch, full_local_basis,
       CBEExpansion, cbe_expand,
       CBEBugInfo, cbe_lubich_sweep, truncate_sweep!,
       CBEBugOptions, CBEBugRunInfo, cbe_lubich_bug!,
       sketch_bond_left, sketch_bond_right, cbe_expand_bond,
       cbe_bond_update, cbe_bond_update_sweep!, cbe_bond_update_bug!,
       cbe_gate_evolve!, SStepSolver, Propagate, LowestEigen,
       RealTime, ImaginaryTime, GroundState,
       ZeroSiteH, zero_site_h, apply_zero_site,
       OneSiteH, one_site_h, apply_one_site, TDVP2Info, tdvp2_step!, tensor_elements,
       TDVP1CBEInfo, tdvp1_cbe_step!,
       tdvp1_rsvd_cbe_step!, tdvp1_cbe_exact_step!

end # module
