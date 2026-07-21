using LurCGT, Telum, LinearAlgebra, Random
using BUGJulia.BondUpdateBUG

psi = domain_wall_state(6); canonical!(psi, 2)
f = bond_frame(psi, 2)
K1 = to_concrete(f.U0 * f.S0)
bond_tag = f.U0.inds[3].itags

println("U0   inds = ", f.U0.inds)
println("K1   inds = ", K1.inds)
perp = perp_component(f.U0, K1)
println("perp inds = ", perp.inds, "   |perp| = ", norm(perp))

rep = sector_report(f.U0, K1)
for r in rep; println("  ", r); end

F = to_concrete(getIdentity((f.U0,1),(f.U0,2); itag=bond_tag))
println("F    inds = ", F.inds)
println("F sp3 = ", F.spaces[3])

missed = [r for r in rep if r.missing]
for r in missed
    seed = random_sector_seed(F, r.charge, 1; rng=MersenneTwister(0x5EED))
    println("seed for ", r.charge, " inds = ", seed === nothing ? "nothing" : seed.inds)
    if seed !== nothing
        st = to_concrete(setitag(seed, 3, bond_tag))
        println("  retagged inds = ", st.inds)
        println("  vs U0    inds = ", f.U0.inds)
        for l in 1:3
            a, b = f.U0.inds[l], st.inds[l]
            println("   leg", l, " U0=(", a.itags, ",", a.dir, ",", a.plev, ",", a.lock, ",", a.dual,
                    ")  seed=(", b.itags, ",", b.dir, ",", b.plev, ",", b.lock, ",", b.dual, ")  eq=", a==b)
        end
        try
            o = to_concrete(oplus(f.U0, st, (3,)))
            println("  oplus OK -> ", o.spaces[3])
        catch e
            println("  oplus THREW: ", first(split(sprint(showerror,e),'\n')))
        end
    end
end
println("\nAUGDIAG_DONE")
