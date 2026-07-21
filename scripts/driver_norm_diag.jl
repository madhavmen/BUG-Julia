using LinearAlgebra, Printf, LurCGT, Telum
using BUGJulia.BondUpdateBUG

function show(label; kwargs...)
    psi = domain_wall_state(6)
    g = bond_gates(psi)
    info = bond_update_bug!(psi, g; opts = BondUpdateOptions(; kwargs...))
    println("--- $label ---")
    for k in eachindex(info.times)
        @printf("  step %2d  norm=%.15f  1-norm=%9.2e  disc=%9.2e  maxb=%d  bd=%s\n",
                k, info.norms[k], 1 - info.norms[k], info.discarded[k],
                info.max_bond_dims[k], info.bond_dims[k])
    end
end

show("dt=0.02 n=10 maxdim=8 thresh=1e-12 (default)";
     dt = 0.02, n_steps = 10, order = :strang, maxdim = 8, normalize = false)
show("dt=0.02 n=10 maxdim=8 thresh=1e-14";
     dt = 0.02, n_steps = 10, order = :strang, maxdim = 8, normalize = false,
     trunc_thresh = 1e-14)
show("dt=0.05 n=5 maxdim=2 normalize=false";
     dt = 0.05, n_steps = 5, maxdim = 2, normalize = false)
