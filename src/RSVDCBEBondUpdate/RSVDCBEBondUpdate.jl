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
include("cbe_lubich.jl")
include("tdvp2_baseline.jl")
include("tdvp1_cbe.jl")

# The modes come last: each imports from the parent, so everything they name must already be
# defined. They are submodules rather than files so `RealTime.evolve!` and
# `GroundState.solve!` read as what they are at the call site.
include("modes/real_time.jl")
include("modes/imaginary_time.jl")
include("modes/ground_state.jl")

export XXZTerm, XXZChain, xxz_chain, heisenberg_su2_chain, hamiltonian_terms, bond_gate,
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
       TDVP1CBEInfo, tdvp1_cbe_step!

end # module
