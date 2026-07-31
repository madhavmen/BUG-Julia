module BUGJulia

# `BondUpdateBUG` is the single consolidated symmetry-native BUG integrator.
include("BondUpdateBUG/BondUpdateBUG.jl")

# `RSVDCBEBondUpdate` is the exploratory CBE-BUG variant (docs/cbe_bug.md): the
# randomized-SVD controlled bond expansion in place of the K/L discarded-projector
# augmentation. Separate submodule -- it imports from `BondUpdateBUG` and changes
# nothing there.
include("RSVDCBEBondUpdate/RSVDCBEBondUpdate.jl")

end # module
