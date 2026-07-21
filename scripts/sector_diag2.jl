using LurCGT, Telum, LinearAlgebra
using BUGJulia.BondUpdateBUG

function report(lbl, psi, i)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    t = f.U0
    fused = to_concrete(getIdentity((t,1),(t,2); itag="fused"))
    mine = fuse_spaces(symm(t), t.spaces[1], t.inds[1].dir, t.spaces[2], t.inds[2].dir)
    ok = sort([q for (q,_) in fused.spaces[end]], by=string) == sort([q for (q,_) in mine], by=string)
    println(lbl, " bond ", i, " dirs=", join([x.dir for x in t.inds]),
            " fusedout=", fused.inds[end].dir, "  AGREE=", ok)
    println("    leg1 sp=", t.spaces[1], "  leg2 sp=", t.spaces[2])
    println("    telum  =", [q for (q,_) in fused.spaces[end]])
    println("    mine   =", [q for (q,_) in mine])
end

psi = domain_wall_state(4)
for i in 1:3; report("DW4", psi, i); end
println("\nDIAG2_DONE")
