# Probe: how to build and apply a two-site U(1) bond gate in Telum.
using LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG

ok(f) = try (f(); true) catch e; println("      ! ", sprint(showerror, e)[1:min(end,200)]); false end
show_inds(name, t) = begin
    println("  $name: rank=$(length(t.inds))")
    for (k, ix) in enumerate(t.inds)
        println("    leg $k: itag=$(ix.itags) dir=$(ix.dir) spaces=$(t.spaces[k])")
    end
end

q = local_space()

println("=== 1. local operators ===")
for nm in (:I, :Sz, :Sp, :Sm)
    show_inds(string(nm), getfield(q, nm))
end

println("\n=== 2. what does Telum export that looks like permute/outer? ===")
cands = filter(n -> occursin(r"perm|outer|kron|tprod|transpose|swap"i, string(n)), names(Telum))
println("  ", cands)

println("\n=== 3. outer product forms ===")
println("  contract(Sz,(),Sz,()) : ", ok(() -> begin
    t = contract(q.Sz, (), q.Sz, ()); show_inds("    result", t) end))

println("\n=== 4. Sp-Sm coupling via the op leg ===")
for (lbl, f) in (
        ("contract(Sp,(3,),Sm,(3,))",  () -> contract(q.Sp, (3,), q.Sm, (3,))),
        ("contract(Sp,(3,),Sm',(3,))", () -> contract(q.Sp, (3,), q.Sm', (3,))),
        ("contract(Sp,(3,),Sp',(3,))", () -> contract(q.Sp, (3,), q.Sp', (3,))),
        ("contract(Sm,(3,),Sm',(3,))", () -> contract(q.Sm, (3,), q.Sm', (3,))),
    )
    print("  $lbl : ")
    r = try f() catch e; println("! ", sprint(showerror, e)[1:min(end,200)]); nothing end
    r === nothing || (println(); show_inds("    result", r); println("    norm=", norm(r)))
end

println("\n=== 5. theta leg structure from a real frame ===")
psi = domain_wall_state(4); canonical!(psi, 2)
f = bond_frame(psi, 2)
th = frame_theta(f)
show_inds("theta", th)
println("  frame.site_l: itag=$(f.site_l.itags) dir=$(f.site_l.dir)")
println("  frame.site_r: itag=$(f.site_r.itags) dir=$(f.site_r.dir)")

println("\n=== 6. can a retagged Sz contract theta's site leg? ===")
Ol = setitag(q.Sz, f.site_l.itags)
show_inds("Sz retagged to site_l", Ol)
println("  contract(Ol,(1,),theta,(2,)) : ", ok(() -> begin
    t = contract(Ol, (1,), th, (2,)); show_inds("    result", t) end))
println("  contract(Ol,(2,),theta,(2,)) : ", ok(() -> contract(Ol, (2,), th, (2,))))
