isdefined(@__MODULE__, :BUG) || @eval begin
"""
    BUG

Bond-Update-and-Galerkin (BUG) integrator for quantum systems.

**Quantum method:**

- `bug_two_site!` — 2-Site-BUG: applies 2-site-local layers via faithful-KLS with
  adaptive rank, using an odd/even (checkerboard) Strang sweep.

The local KLS update uses faithful simultaneous K+L augmentation with Krylov-based
S-step. Orders: `:lie` (1st), `:strang` (2nd, default).

Public API:
- `bug_two_site!(psi, gates; dt, order=:strang, maxdim, ...)` → BUGInfo
- Helpers: `two_site_xx_parity_mpos`, `two_site_xx_bond_gates`, `_faithful_kls_local_bond_candidate`
"""
module BUG

using LinearAlgebra
using ITensors
using ITensorMPS: MPS, MPO, OpSum, op
import KrylovKit

using ..TTutils

include("bug_init.jl")
include("bug_kls.jl")
include("discarded_bug.jl")
let two_site_file = joinpath(@__DIR__, "two_site_bug", "two_site_bug.jl")
    if isfile(two_site_file)
        include("two_site_bug/two_site_bug.jl")
    else
        function bug_two_site!(args...; kwargs...)
            throw(ArgumentError("BUG two-site support is not available in this worktree."))
        end
        function two_site_xx_parity_mpos(args...; kwargs...)
            throw(ArgumentError("BUG two-site support is not available in this worktree."))
        end
        function two_site_xx_bond_gates(args...; kwargs...)
            throw(ArgumentError("BUG two-site support is not available in this worktree."))
        end
    end
end

export BUGInfo
export bug_two_site!
export discarded_bug_step!
export two_site_xx_parity_mpos, two_site_xx_bond_gates
export _left_site_bond_index, _site_bond_index

end # module BUG
end # isdefined guard
