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
  - `cbe.jl`   — the RSVD bond expansion itself (sector-graded sketch, preselection,
    final selection).
  - `cbe_bug.jl` — `cbe_bug_bond_update`: basis sweep + one centre-bond Galerkin step.
"""
module RSVDCBEBondUpdate

using LinearAlgebra, Random, Printf
using LurCGT, Telum

using ..BondUpdateBUG: SymMPS, canonical!, bond_dims, leg_dim, local_space,
                       symmetry_mode, BondFrame, bond_frame, frame_theta,
                       expv, tensor_inner, perp_component, perp_component_right,
                       fusion_basis, reachable_sectors, sector_dim, dual_charge,
                       align_charge, left_isometry_defect, right_isometry_defect,
                       energy, bond_gates, apply_gate, heisenberg_bond_gate

include("henv.jl")

export XXZTerm, XXZChain, xxz_chain, hamiltonian_terms, bond_gate,
       LeftEnvStack, RightEnvStack,
       left_env_stack, right_env_stack, env_energy, env_energy_right,
       apply_h_two_site

end # module
