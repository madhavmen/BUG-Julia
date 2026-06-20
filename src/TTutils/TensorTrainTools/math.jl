# math.jl ÔÇö inner products, norms, and distances for TensorTrain objects

export distance
import LinearAlgebra: dot, norm


"""
    dot(x::TensorTrain, y::TensorTrain) -> Number

Compute the inner product Ôƒ¿x, yÔƒ® by contracting from left to right.
"""
function LinearAlgebra.dot(x::TensorTrain, y::TensorTrain)
    length(x) != length(y) && throw(ArgumentError("TensorTrains have different lengths."))
    # replacelinks ensures x and y share only boundaries (dim-1), not interior bonds
    T = contract_tensors(x, replacelinks(y))
    return scalar(T)
end


"""
    dot(x::TensorTrain, B::TensorTrainOperator, y::TensorTrain) -> Number

Compute the expectation value Ôƒ¿x|B|yÔƒ®.
"""
function LinearAlgebra.dot(x::TensorTrain, B::TensorTrainOperator, y::TensorTrain)
    By = contract(B, y)
    return dot(x, By)
end


"""
    dot(A::TensorTrain, x::TensorTrain, B::TensorTrainOperator, y::TensorTrain) -> Number

Compute Ôƒ¿Ax|ByÔƒ®.
"""
function LinearAlgebra.dot(A::TensorTrain, x::TensorTrain, B::TensorTrainOperator, y::TensorTrain)
    T = dag(A[1]) * x[1] * B[1] * y[1]
    for k in 2:length(x)
        T = T * dag(A[k]) * x[k] * B[k] * y[k]
    end
    return scalar(T)
end


"""
    norm(f::TensorTrain) -> Real

Compute the 2-norm of the tensor train.
"""
function LinearAlgebra.norm(f::TensorTrain)
    g = replacelinks(f)
    return sqrt(abs(real(dot(f, g))))
end


"""
    distance(f::TensorTrain, g::TensorTrain) -> Real

Compute ÔÇûf - gÔÇûÔéé without explicitly forming the difference.
"""
function distance(f::TensorTrain, g::TensorTrain)
    ff = dot(replacelinks(f), f)
    gg = dot(replacelinks(g), g)
    fg = dot(f, g)
    return sqrt(abs(real(ff) + real(gg) - 2*real(fg)))
end


# ÔöÇÔöÇ helper ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

"""
    contract_tensors(x, y)

Contract a collection of tensors according to their shared indices.

Arguments
- `x`: Local tensor or vector variable used by the helper routine.
- `y`: Local tensor or vector variable used by the helper routine.

Returns
- Returns the tensor produced by the full contraction sequence.

Description
- This is the basic algebraic reduction underlying TT environment assembly: shared indices are summed while open indices are preserved, implementing multilinear tensor composition.
"""
function contract_tensors(x::TensorTrain, y::TensorTrain)
    T = dag(x[1]) * y[1]
    for k in 2:length(x)
        T = T * dag(x[k]) * y[k]
    end
    return T
end
