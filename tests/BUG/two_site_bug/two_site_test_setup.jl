# two_site_test_setup.jl
#
# Shared fixtures for the two-site BUG odd/even sweep tests.
# Include this once per test file (after `using Test, ITensors, ÔÇª`).

const _SRC = joinpath(@__DIR__, "..", "..", "..", "src")
if !isdefined(Main, :TTutils)
    include(joinpath(_SRC, "TTutils", "TTutils.jl"))
end
using .TTutils
if !isdefined(Main, :BUG)
    include(joinpath(_SRC, "BUG", "BUG.jl"))
end
using .BUG

# Exact dense propagator exp(-i H t) applied to a state vector.
function two_site_exact(v::AbstractVector, H::AbstractMatrix, t::Real)
    F = eigen(Hermitian(H))
    return F.vectors * (exp.(-im .* F.values .* t) .* (F.vectors' * v))
end

# Infidelity 1 - |Ôƒ¿v,refÔƒ®| / (ÔÇûvÔÇûÔÇûrefÔÇû); phase- and norm-tolerant.
function two_site_infidelity(v::AbstractVector, ref::AbstractVector)
    denom = norm(v) * norm(ref)
    iszero(denom) && return 1.0
    return 1.0 - clamp(real(abs(dot(v, ref)) / denom), 0.0, 1.0)
end

two_site_norm_error(v::AbstractVector, ref_norm::Real) = abs(norm(v) - ref_norm) / ref_norm

# Rank-3 random normalized product-train state on the given sites.
function two_site_rank3_state(sites; seed::Int)
    Random.seed!(seed)
    return TensorTrain(normalize!(random_mps(sites; linkdims = 3)))
end

# Dense matrix of a TensorTrainOperator (warn-order safe).
function two_site_dense(W)
    ITensors.disable_warn_order()
    M = ComplexF64.(TTutils.matrix(W))
    ITensors.reset_warn_order()
    return M
end

# Dense vector of a TensorTrain (warn-order safe).
function two_site_vec(psi)
    ITensors.disable_warn_order()
    v = ComplexF64.(TTutils.vector(psi))
    ITensors.reset_warn_order()
    return v
end
