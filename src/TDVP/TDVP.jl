isdefined(@__MODULE__, :TDVP) || @eval begin
"""
    TDVP

Native 1-site and 2-site TDVP integrators for quantum tensor trains.

Evolves TensorTrain states against TensorTrainOperator MPOs using the
standard projector-splitting one-site TDVP sweep (forward: site → QR → 0-site
backward; reverse: site → LQ → 0-site backward).

Public API:
- `tdvp_step!(psi, W; dt, ...)` — advance one 1-site TDVP step
- `tdvp2_step!(psi, W; dt, ...)` — advance one 2-site TDVP step

Supported composition orders (step_mode):
- `:symmetric_fr` / `:symmetric_rf`        — 2nd order (Strang splitting)
- `:third_order_frf` / `:third_order_rfr`  — 3rd order
- `:fourth_order_yoshida_fr` / `_rf`       — 4th order (Yoshida)
"""
module TDVP

using LinearAlgebra
using ITensors
import KrylovKit

using ..TTutils

include("tdvp_init.jl")
include("tdvp_local.jl")
include("tdvp_sweep.jl")
include("tdvp2_sweep.jl")

export TDVPInfo
export tdvp_step!
export tdvp2_step!
export tdvp2_parallel_step!

end # module TDVP
end # isdefined guard
