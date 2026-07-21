# compression.jl
#
# SVD compression, TT addition (variational sum), zipup, and linear solve.

const _QM_TTT_COMP = joinpath(@__DIR__, "TensorTrainTools")

include(joinpath(_QM_TTT_COMP, "compression.jl"))
include(joinpath(_QM_TTT_COMP, "addition.jl"))
include(joinpath(_QM_TTT_COMP, "linear_solve.jl"))
include(joinpath(_QM_TTT_COMP, "zipup.jl"))