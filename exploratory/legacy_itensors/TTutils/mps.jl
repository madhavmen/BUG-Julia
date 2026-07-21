# mps.jl
#
# TensorTrain (MPS) type, constructors, index utilities, norms, orthogonalization.
# Pulled from the Quantum_models canonical tensor_train_utils source.

# ─────────────────────────────────────────────────────────────────────────────
# AbstractTensorTrain (supertype + container interface + broadcast overloads)
const _QM_TTT = joinpath(@__DIR__, "TensorTrainTools")

include(joinpath(_QM_TTT, "utils.jl"))
include(joinpath(_QM_TTT, "abstracttensortrain.jl"))
include(joinpath(_QM_TTT, "tensortrain.jl"))
# math.jl is included in TTutils.jl after mpo.jl since it uses TensorTrainOperator

# â”€â”€ Convenience overload: random_tt(sites; maxdim, seed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import Random
function random_tt(sites::Vector{<:Index}; maxdim::Integer, seed::Union{Integer,Nothing}=nothing, kwargs...)
    isnothing(seed) || Random.seed!(seed)
    return random_tt(sites, maxdim; kwargs...)
end

# â”€â”€ normalize! for TensorTrain (missing from upstream) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function uniform_tt(sites::Vector{<:Index}, c::Number)
    psi = uniform_tt(promote_type(Float64, typeof(c)), sites)
    rescale!(psi, c)
    return psi
end

import LinearAlgebra: normalize!
function normalize!(psi::TensorTrain)
    n = norm(psi)
    n > 0 && rescale!(psi, 1 / n)
    return psi
end