__precompile__(false)

"""
    ProjectorSplitting

Top-level module providing tensor-train tools plus quantum and classical-PDE
time integrators.
"""
module ProjectorSplitting

using LinearAlgebra
using ITensors
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# New restructured quantum submodules
# ──────────────────────────────────────────────────────────────────────────────
include("TTutils/TTutils.jl")
using .TTutils

include("TDVP/TDVP.jl")
using .TDVP

include("BUG/BUG.jl")
using .BUG
const BUGQuantum = BUG

include("LubichBUG/LubichBUG.jl")
using .LubichBUG

include("LubichTreeMPS/LubichTreeMPS.jl")
using .LubichTreeMPS

# ──────────────────────────────────────────────────────────────────────────────
# Classical PDE submodules
# ──────────────────────────────────────────────────────────────────────────────
include("../Classical_PDEs/tensor_train_utils/TensorTrainUtils.jl")
using .TensorTrainUtils

include("../Classical_PDEs/integrators/BUGPDE/BUGPDE.jl")
using .BUGPDE

include("../Classical_PDEs/integrators/ProjectorSplittingPDE/ProjectorSplittingPDE.jl")
using .ProjectorSplittingPDE

include("../Classical_PDEs/integrators/BUGSweepPDE/BUGSweepPDE.jl")
using .BUGSweepPDE

include("../Classical_PDEs/integrators/LubichBUGPDE/LubichBUGPDE.jl")
using .LubichBUGPDE

include("../Classical_PDEs/integrators/LubichTreeMPSPDE/LubichTreeMPSPDE.jl")
using .LubichTreeMPSPDE

include("../Classical_PDEs/quantics/Quantics.jl")
using .Quantics

include("../Classical_PDEs/models/HeatEquation.jl")
using .HeatEquation

include("../Classical_PDEs/models/Burgers.jl")
using .Burgers

include("../Classical_PDEs/models/ColeHopf.jl")
using .ColeHopf

include("../Classical_PDEs/models/GrossPitaevskii2D.jl")
using .GrossPitaevskii2D

include("../Classical_PDEs/models/BurgersColHopf.jl")
using .BurgersColHopf

include("../Classical_PDEs/reference/GrossPitaevskii2DReference.jl")
using .GrossPitaevskii2DReference

include("../Classical_PDEs/reference/HeatAnalytical.jl")
using .HeatAnalytical

include("../Classical_PDEs/reference/BurgersReference.jl")
using .BurgersReference

include("../Classical_PDEs/reference/BurgersColeHopf.jl")
using .BurgersColeHopf

function _as_quantum_tt(psi::TensorTrainUtils.TensorTrain)
    return TTutils.TensorTrain(copy(psi.data))
end

function _as_quantum_tto(W::TensorTrainUtils.TensorTrainOperator)
    return TTutils.TensorTrainOperator(copy(W.data))
end

function _copy_quantum_tt_data!(
    dest::TensorTrainUtils.TensorTrain,
    src::TTutils.TensorTrain,
)
    length(dest) == length(src) || throw(ArgumentError("Tensor trains have different lengths."))
    for i in eachindex(dest.data, src.data)
        dest.data[i] = src.data[i]
    end
    return dest
end

function TTutils._build_right_envs_mpo(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator,
)
    return TTutils._build_right_envs_mpo(_as_quantum_tt(psi), _as_quantum_tto(W))
end

function TTutils._build_left_envs_mpo(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator,
)
    return TTutils._build_left_envs_mpo(_as_quantum_tt(psi), _as_quantum_tto(W))
end

function TTutils._build_right_envs_mpo_for_bonds(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator,
    bonds::Vector{Int},
)
    return TTutils._build_right_envs_mpo_for_bonds(_as_quantum_tt(psi), _as_quantum_tto(W), bonds)
end

function TTutils._build_left_envs_mpo_for_bonds(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator,
    bonds::Vector{Int},
)
    return TTutils._build_left_envs_mpo_for_bonds(_as_quantum_tt(psi), _as_quantum_tto(W), bonds)
end

function TTutils._owned_two_site_mpo_envs(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator,
    bond::Int,
)
    return TTutils._owned_two_site_mpo_envs(_as_quantum_tt(psi), _as_quantum_tto(W), bond)
end

function TDVP.tdvp_step!(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator;
    kwargs...,
)
    psi_q = _as_quantum_tt(psi)
    info = TDVP.tdvp_step!(psi_q, _as_quantum_tto(W); kwargs...)
    _copy_quantum_tt_data!(psi, psi_q)
    return info
end

function TDVP.tdvp2_parallel_step!(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator;
    kwargs...,
)
    psi_q = _as_quantum_tt(psi)
    info = TDVP.tdvp2_parallel_step!(psi_q, _as_quantum_tto(W); kwargs...)
    _copy_quantum_tt_data!(psi, psi_q)
    return info
end

function TDVP.tdvp2_step!(
    psi::TensorTrainUtils.TensorTrain,
    W::TensorTrainUtils.TensorTrainOperator;
    kwargs...,
)
    psi_q = _as_quantum_tt(psi)
    info = TDVP.tdvp2_step!(psi_q, _as_quantum_tto(W); kwargs...)
    _copy_quantum_tt_data!(psi, psi_q)
    return info
end

# ── Submodule re-exports ──────────────────────────────────────────────────────
export TTutils
export TDVP
export BUG
export BUGQuantum
export LubichBUG
export LubichTreeMPS
export TensorTrainUtils
export BUGPDE
export ProjectorSplittingPDE
export BUGSweepPDE
export LubichBUGPDE
export LubichTreeMPSPDE
export Quantics
export HeatEquation
export Burgers
export ColeHopf
export GrossPitaevskii2D
export GrossPitaevskii2DReference
export BurgersColHopf
export HeatAnalytical
export BurgersReference
export BurgersColeHopf

# ── New quantum integrator API ────────────────────────────────────────────────
export TDVPInfo
export tdvp_step!
export tdvp2_step!

export BUGInfo
export LubichBUGInfo
export lubich_bug_step!
export LubichTreeMPSInfo
export MPSParallelIntegrator
export lubich_tree_mps_step!

# ── Classical PDE integrator API ─────────────────────────────────────────────
export PSInfo
export ps_step!
export BUGOperatorFieldTerm
export MPOCallableRHS
export mpo_rhs
export bug_step_euler!
export bug_step_rk4!
export bug_step_lanczos!
export quantics_sites_1d
export quantics_sites_2d
export quantics_grid_1d
export quantics_grid_2d
export tt_from_grid_function_1d
export tt_from_grid_function_2d
export tt_to_grid_1d
export tt_to_grid_2d
export make_heat_rhs_1d
export make_heat_rhs_2d
export pde_heat_generator_1d
export pde_heat_generator_2d
export bug_pde_step!
export lubich_bug_pde_step!
export lubich_tree_mps_pde_step!
export tdvp1_pde_step!
export tdvp2_pde_step!
export make_burgers_rhs_1d
export make_burgers_rhs_2d
export cole_hopf_periodic_factor_1d
export cole_hopf_encode_1d
export cole_hopf_encode_2d
export cole_hopf_readout_1d
export cole_hopf_readout_2d
export heat_periodic_exact
export heat_periodic_sampled
export heat_periodic_exact_2d
export heat_periodic_sampled_2d
export burgers_reference_step
export burgers_reference_solve
export burgers_reference_step_2d
export burgers_reference_solve_2d
export burgers_colehopf_reference_1d
export burgers_colehopf_reference_2d
export gp_2d_dns_step!
export cole_hopf_forward
export cole_hopf_inverse
export cole_hopf_bug_solve_states

end # module ProjectorSplitting
