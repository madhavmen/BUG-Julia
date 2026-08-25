# HOW EXPENSIVE IS ONE 8x8 CYLINDER STEP? Sized before committing to a scan, because an earlier
# L=16 estimate read a COLD-START JIT number as the per-step cost and was wrong by ~30x.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
const OUT = joinpath(@__DIR__, "results", "cyl88_probe.txt")
mkpath(dirname(OUT)); io = open(OUT, "w")
say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:U1)
LX, LY = 8, 8
L = LX * LY
W = square_cylinder_mpo(LX, LY)
say(@sprintf("8x8 cylinder: %d sites, %d bonds, MPO max virtual dim = %d",
             L, length(square_cylinder_bonds(LX, LY)), maximum(mpo_virtual_dims(W))))
psi0 = dimer_state(L)
for cap in (32, 64, 128)
    psi = copy(psi0)
    # one WARM-UP step first so the timing below is not measuring JIT
    cbe_bug_step!(psi, W, ComplexF64(-im * 0.05); exact = true, krylov_basis = 3,
                  krylov_tol = 0.0, maxdim = cap, trunc_thresh = 1e-10, maxiter = 16)
    t0 = time(); kry = 0
    for _ in 1:2
        info = cbe_bug_step!(psi, W, ComplexF64(-im * 0.05); exact = true, krylov_basis = 3,
                             krylov_tol = 0.0, maxdim = cap, trunc_thresh = 1e-10, maxiter = 16)
        kry += info.krylov_dims
    end
    el = (time() - t0) / 2
    say(@sprintf("  cbe_bug m=3 cap=%3d: %6.1f s/step (warm)  states=%3d  kry/step=%d",
                 cap, el, maximum(state_bond_dims(psi)), kry ÷ 2))
end
psi = copy(psi0)
tdvp2_step!(psi, W, ComplexF64(-im * 0.05); maxdim = 64, trunc_thresh = 1e-10, maxiter = 16)
t0 = time()
tdvp2_step!(psi, W, ComplexF64(-im * 0.05); maxdim = 64, trunc_thresh = 1e-10, maxiter = 16)
say(@sprintf("  tdvp2       cap= 64: %6.1f s/step (warm)  states=%3d",
             time() - t0, maximum(state_bond_dims(psi))))
close(io)
