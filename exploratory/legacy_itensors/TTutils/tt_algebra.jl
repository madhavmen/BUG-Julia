# tt_algebra.jl
#
# TT-level algebraic operations: direct sum, scaled copy, CBE augmentation.
# These are used by BUG augmentation and by benchmark state preparation.

include(joinpath(@__DIR__, "TensorTrainTools", "tt_algebra_upstream.jl"))