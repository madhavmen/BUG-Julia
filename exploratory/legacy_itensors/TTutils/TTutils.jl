isdefined(@__MODULE__, :TTutils) || @eval begin

"""
    TTutils

Tensor-train (MPS/MPO) utilities for the quantum integrators.

Organized into logical files:
- `tensor_algebra.jl`  — Low-level ITensor QR/LQ/SVD overloads and truncation
- `mps.jl`             — TensorTrain (MPS) type, constructors, index utilities, norms, orthogonalization
- `mpo.jl`             — TensorTrainOperator (MPO) type, constructors, MPO environment helpers
- `compression.jl`     — SVD compression, addition (variational sum), zipup, linear solve
- `krylov_utils.jl`    — Krylov/expv helpers, tensor index helpers, shared by BUG and TDVP
- `tt_algebra.jl`      — TT-level algebraic operations: direct sum, scaled copy, CBE augmentation

All BUG and TDVP integrators import from this module for tensor-network construction
and manipulation. No integrator logic lives here.
"""
module TTutils

using ITensors
using ITensorMPS
using LinearAlgebra
using TensorOperations: @tensor
import KrylovKit

# Import names that TTutils adds methods to, so the definitions in the
# Quantum_models source files extend the canonical functions instead of
# creating separate TTutils-local copies that shadow them for callers.
import ITensors: siteinds
import ITensorMPS: orthogonalize, orthogonalize!

# ── Core tensor algebra (QR/LQ/SVD overloads for ITensors) ────────────────────
include("tensor_algebra.jl")

# ── MPS (TensorTrain) type, constructors, ops ─────────────────────────────────
include("mps.jl")

# ── MPO (TensorTrainOperator) type, constructors, environment helpers ─────────
include("mpo.jl")

# ── math.jl: dot/norm/distance — must follow mpo.jl (uses TensorTrainOperator) ─
const _QM_TTT_MATH = joinpath(@__DIR__, "TensorTrainTools")
include(joinpath(_QM_TTT_MATH, "math.jl"))



# ── Krylov/expv helpers and low-level index utilities ─────────────────────────
include("krylov_utils.jl")

# ── TT-level algebraic operations ─────────────────────────────────────────────
include("tt_algebra.jl")

# ── Compression / addition / linear solves ────────────────────────────────────
include(joinpath(_QM_TTT_MATH, "compression.jl"))

# ── Re-exports ────────────────────────────────────────────────────────────────

# Tensor algebra
export qr, lq, svd, random_unitary, truncate

# MPS
export AbstractTensorTrain, TensorTrain
export linkind, linkinds, siteind, maxlinkdim
export replacelinks, replacelinks!, rescale!, bitreverse, connect
export random_tt, empty_tt, uniform_tt
# orthogonalize / orthogonalize! extend ITensorMPS — not re-exported to avoid shadow
export vector, dot, norm, distance
export front, back, check_unique_inds
export linkdims

# MPO
export TensorTrainOperator
export matrix, contract, tto_direct_sum
export identity_op, empty_op, random_op
export _build_right_envs_mpo, _build_left_envs_mpo
export _build_right_envs_mpo_for_bonds, _build_left_envs_mpo_for_bonds
export _left_env_boundary_mpo, _right_env_boundary_mpo
export _advance_left_env_mpo, _advance_right_env_mpo
export _mpo_boundary_link, _mpo_identity_transfer
export _build_1site_Heff_mat, _build_0site_Heff_mat
export _owned_two_site_mpo_envs
export _partition_ranges

# Compression / addition
export svd_compress, svd_compress!, compress, compress!
export add, add!, multiply!, square!
export svd_compress_rsvd, svd_compress_rsvd!
export svd_compress_reverse!
export zipup
export LinearSolveInfo, solve_linear, solve_linear!

# Krylov utils
export _linear_substep, _general_linear_substep
export _native_hermitian_lanczos_exponentiate
export _with_bug_expv_backend, _with_bug_time_prefactor, _active_time_prefactor
export _tensortrain_bond_indices, _tensortrain_site_index, _tensortrain_boundary_link
export _complex_tensor_array, _complex_tensor_vec
export _qr_nonzero_diagonal_rank, _qr_column_basis, _qr_row_basis
export _identity_overlap_matrix, _complete_column_basis, _complete_row_basis

# TT algebra
export tt_direct_sum, tt_scaled_copy, tt_cbe_augment

end # module TTutils

end
