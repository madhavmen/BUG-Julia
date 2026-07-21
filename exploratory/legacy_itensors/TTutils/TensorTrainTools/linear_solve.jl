export LinearSolveInfo, solve_linear, solve_linear!

"""
    LinearSolveInfo

Diagnostics returned by the dense fallback tensor-train linear solver.

The current implementation forms the dense linear system, solves it directly,
and then compresses the result back into TT form, so the fields mostly record
residual quality and timing rather than sweep-by-sweep ALS statistics.
"""
mutable struct LinearSolveInfo
    converged::Bool
    residual::Float64
    residual_history::Vector{Float64}
    sweeps::Int
    elapsed::Float64
    warning::Bool
end

"""
    LinearSolveInfo()

Construct an empty linear-solver diagnostics record.
"""
LinearSolveInfo() = LinearSolveInfo(false, Float64(Inf), Float64[], 0, 0.0, false)

"""
    _dense_residual(A, x, b)

Evaluate a dense residual vector or tensor for the linear-solve utility.

Arguments
- `A`: Tensor, matrix, or tensor-train operator supplied to the helper.
- `x`: Local tensor or vector variable used by the helper routine.
- `b`: Right-hand side or second tensor-train argument in the local computation.

Returns
- Returns the residual of the dense linear system represented by the supplied arguments.

Description
- Even when the production algorithm works in tensor-train form, dense residuals remain useful for validation, debugging, and small-problem reference calculations.
"""
function _dense_residual(A::TensorTrainOperator, x::AbstractVector, b::AbstractVector)
    denom = max(norm(b), eps(Float64))
    return LinearAlgebra.norm(matrix(A) * x .- b) / denom
end

"""
    solve_linear(A, b, bonddim; tol=1e-10, max_sweeps=8, warntol=1e-6, cutoff=0.0) -> TensorTrain

Allocate a TT ansatz with bond dimension `bonddim`, solve the dense linear
system represented by `A * x = b`, and return the solution compressed back into
tensor-train form.
"""
function solve_linear(
    A::TensorTrainOperator,
    b::TensorTrain,
    bonddim::Integer;
    tol::AbstractFloat = 1e-10,
    max_sweeps::Integer = 8,
    warntol::AbstractFloat = 1e-6,
    cutoff::AbstractFloat = 0.0,
)
    ansatz = empty_tt(promote_itensor_eltype(b), siteinds(b), bonddim)
    solve_linear!(
        ansatz,
        A,
        b;
        tol = tol,
        max_sweeps = max_sweeps,
        warntol = warntol,
        cutoff = cutoff,
    )
    return ansatz
end

"""
    solve_linear!(ansatz, A, b; tol=1e-10, max_sweeps=8, warntol=1e-6, cutoff=0.0) -> LinearSolveInfo

Overwrite `ansatz` with a TT-compressed solution of `A * x = b`.

At the moment this routine is a dense fallback:
1. build the dense matrix and right-hand side,
2. solve them with Julia's backslash operator,
3. convert the dense solution back into TT form,
4. report the relative dense residual.
"""
function solve_linear!(
    ansatz::TensorTrain,
    A::TensorTrainOperator,
    b::TensorTrain;
    tol::AbstractFloat = 1e-10,
    max_sweeps::Integer = 8,
    warntol::AbstractFloat = 1e-6,
    cutoff::AbstractFloat = 0.0,
)
    length(ansatz) == length(b) == length(A) ||
        throw(ArgumentError("ansatz, A, and b must have the same length."))
    siteinds(ansatz) == siteinds(b) == siteinds(A; plev = 0) ||
        throw(ArgumentError("ansatz, A, and b must share the same site indices."))

    info = LinearSolveInfo()
    info.elapsed = @elapsed begin
        # Dense fallback path: simple and robust for the small Poisson and model
        # problems used by the library today.
        A_dense = matrix(A)
        b_vec = vector(b)
        x_vec = A_dense \ b_vec

        # Recompress the dense solution back into the caller-provided TT ansatz.
        x_tt = TensorTrain(
            x_vec,
            siteinds(ansatz);
            maxdim = max(maxlinkdim(ansatz), 1),
            cutoff = cutoff,
        )
        copyto!(ansatz, x_tt)

        info.residual = _dense_residual(A, vector(ansatz), b_vec)
        info.residual_history = [info.residual]
        info.sweeps = min(max_sweeps, 1)
        info.converged = info.residual < tol
        info.warning = info.residual > warntol
    end

    return info
end
