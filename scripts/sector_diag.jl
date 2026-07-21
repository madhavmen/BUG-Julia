using LurCGT, Telum, LinearAlgebra
using BUGJulia.BondUpdateBUG

psi = neel_state(6)
for i in 1:5
    canonical!(psi, i)
    f = bond_frame(psi, i)
    t = f.U0
    fused = to_concrete(getIdentity((t,1),(t,2); itag="fused"))
    K1 = to_concrete(f.U0 * f.S0)
    perp = perp_component(f.U0, K1)
    perp_dir = norm(perp) <= 1e-12 ? '0' : to_concrete(svd(perp, (1,2); cutoff=1e-12).U).inds[3].dir
    println("bond ", i,
            " | U0 dirs=", join([x.dir for x in t.inds]),
            " | fused out dir=", fused.inds[end].dir,
            " | U0 bond dir=", t.inds[3].dir,
            " | K1 bond dir=", K1.inds[3].dir,
            " | perpU bond dir=", perp_dir, " |perp|=", round(norm(perp), digits=15))
    println("     fused charges = ", [q for (q,_) in fused.spaces[end]])
    println("     U0[3] charges = ", [q for (q,_) in t.spaces[3]])
    println("     K1[3] charges = ", [q for (q,_) in K1.spaces[3]])
end
println("\nDIAG_DONE")
