# Focused probe for the handful of Telum behaviours the smoke run left ambiguous,
# plus the primitives Tasks 5-7 (SymMPS / BondFrame / expv) will need.
# Findings get transcribed into docs/telum_api_contract.md.

using LurCGT, Telum, LinearAlgebra

q = getLocalSpace(SpinOptions(:U1, 1), ("site", "site", "op"))
hdr(s) = println("\n--- ", s, " ", "-"^(58 - length(s)))

# ── A. Does explicit `contract` really require opposite arrow dirs? ──────────
hdr("A. contract arrow-direction rule")
println("Sz leg dirs        : ", [i.dir for i in q.Sz.inds])
for (label, call) in [
        ("contract(Sz,(1,2),Sz,(1,2))",   () -> contract(q.Sz, (1, 2), q.Sz, (1, 2))),
        ("contract(Sz,(1,),Sz,(1,))",     () -> contract(q.Sz, (1,), q.Sz, (1,))),
        ("contract(Sz,(1,),Sz,(2,))",     () -> contract(q.Sz, (1,), q.Sz, (2,))),
        ("contract(Sz',(1,2),Sz,(1,2))",  () -> contract(q.Sz', (1, 2), q.Sz, (1, 2))),
    ]
    try
        r = call()
        println(rpad(label, 32), " -> OK, rank ", length(r.inds))
    catch err
        println(rpad(label, 32), " -> THREW: ", first(split(sprint(showerror, err), '\n')))
    end
end

# ── B. Reading the number out of a rank-0 result ────────────────────────────
hdr("B. rank-0 scalar extraction")
s0 = contract(q.Sz', (1, 2), q.Sz, (1, 2))
println("rank               : ", length(s0.inds))
println("s0[]               : ", s0[])
println("norm(s0)           : ", norm(s0))
sm = contract(q.Sz', (1, 2), q.I, (1, 2))
println("tr(Sz'I) = s[]     : ", sm[], "   (expect 0 -- Sz is traceless)")

# ── C. Addition / subtraction / complex promotion (expv needs these) ────────
hdr("C. TLArray arithmetic")
a = q.Sz + q.Sz
println("norm(Sz+Sz)        : ", norm(a), "  (expect 2*", norm(q.Sz), ")")
b = q.Sz - q.Sz
println("norm(Sz-Sz)        : ", norm(b))
c = q.Sz * (0.0 + 1.0im) + q.I * (2.0 + 0.0im)
println("eltype(i*Sz + 2*I) : ", eltype(c))
println("norm(i*Sz + 2*I)   : ", norm(c))
try
    bad = q.Sz + q.Sp
    println("Sz + Sp            : OK rank ", length(bad.inds))
catch err
    println("Sz + Sp            : THREW: ", first(split(sprint(showerror, err), '\n')))
end

# ── D. permutedims (leg reordering in the sweep) ─────────────────────────────
hdr("D. permutedims")
p = permutedims(q.Sp, (3, 1, 2))
println("Sp legs            : ", q.Sp.inds)
println("perm(3,1,2) legs   : ", p.inds)
println("norm preserved     : ", isapprox(norm(p), norm(q.Sp); atol=1e-12))

# ── E. Building a rank-3 site tensor from scratch (SymMPS needs this) ───────
hdr("E. rank-3 construction via getvac + contract")
vac = getvac(q.I, ("Lbnd", "Rbnd"))
println("vac legs           : ", vac.inds)
println("vac spaces         : ", vac.spaces)
# A boundary "site tensor": vac(1x1) x site. addSingleton is the cheap route.
site3 = addSingleton(q.I; nlegs=1)
println("addSingleton legs  : ", site3.inds)
println("addSingleton spaces: ", site3.spaces)
println("empty_tlarray(I)   : ", (et = empty_tlarray(q.I; T=ComplexF64); (length(et.inds), eltype(et))))

# ── F. Sector arithmetic: what charges can a two-site block carry? ──────────
hdr("F. sector fusion via oplus/contract on the op leg")
println("Sp op-leg charge   : ", q.Sp.spaces[3])
println("Sm op-leg charge   : ", q.Sm.spaces[3])
println("site charges       : ", [s for (s, _) in q.I.spaces[1]])
# add_qn is per-symmetry and takes PLAIN Ints, not the nested sector tuples.
println("add_qn(U1,+1,+1)   : ", add_qn(U1, 1, 1))
println("add_qn(U1,+1,-1)   : ", add_qn(U1, 1, -1))
# Fusing two site charges the way a two-site block does:
fuse(a, b) = ((add_qn(U1, a[1][1], b[1][1]),),)
println("fuse(up,up)        : ", fuse(((1,),), ((1,),)))
println("fuse(up,down)      : ", fuse(((1,),), ((-1,),)))
println("fuse(down,down)    : ", fuse(((-1,),), ((-1,),)))

# ── G. Does `svd` on a rank-3 tensor split the way an MPS bond needs? ───────
hdr("G. svd bipartition on rank-3")
r3 = svd(q.Sp, (1, 2); get_lists=true)
println("left=(1,2) U legs  : ", r3.U.inds)
println("left=(1,2) U spaces: ", r3.U.spaces)
println("Vd legs            : ", r3.Vd.inds)
println("kept_list          : ", r3.kept_list)
println("recon err          : ", norm(r3.U * r3.S * r3.Vd - q.Sp))

println("\nPROBE_DONE")
