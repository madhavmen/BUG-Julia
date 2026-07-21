# Probe 2: the exact construction Task 5 (SymMPS) needs.
# Question: how do you build a rank-3 MPS site tensor (link_l, site, link_r)
# whose link charges accumulate, using only documented Telum calls?

using LurCGT, Telum, LinearAlgebra

q = getLocalSpace(SpinOptions(:U1, 1), ("s", "s", "op"))
hdr(s) = println("\n--- ", s, " ", "-"^max(1, 58 - length(s)))
show3(name, t0) = begin
    # NOTE: several Telum ops (getIdentity, adjoint, scaling) return a lazy
    # TLArrayView, which exposes only `inds`/`spaces`. `to_concrete` is the
    # documented way to get a real TLArray with `.qlabels`/`.RMTs`.
    t = to_concrete(t0)
    println(name, " type   : ", typeof(t0).name.name)
    println(name, " legs   : ", t.inds)
    for l in 1:length(t.inds)
        println(name, " sp[", l, "] : ", t.spaces[l])
    end
    println(name, " nsect  : ", length(t.qlabels), "  qlabels: ", t.qlabels)
end

# ── A. Vacuum boundary link ──────────────────────────────────────────────────
hdr("A. getvac as the left boundary link")
V = getvac(q.I, ("L1", "L1b"))
show3("V", V)

# ── B. Two-leg fusion: link (x) site -> new link ─────────────────────────────
hdr("B. getIdentity((V,2),(I,1); itag=\"L2\")")
F = to_concrete(getIdentity((V, 2), (q.I, 1); itag = "L2"))  # getsub needs a real TLArray
show3("F", F)

# ── C. Restrict the physical leg to one spin -> product-state site tensor ────
hdr("C. getsub on the physical leg of F")
sectors = [s for (s, _) in q.I.spaces[1]]
up, dn = sectors[end], sectors[1]
println("up = ", up, "   dn = ", dn)
Fup = getsub(F, 2, s -> s == up ? Colon() : nothing)
show3("Fup", Fup)
Fdn = getsub(F, 2, s -> s == dn ? Colon() : nothing)
show3("Fdn", Fdn)

# ── D. Chain two site tensors: does the link contract? ──────────────────────
hdr("D. building site 2 on top of site 1's right link")
F2 = to_concrete(getIdentity((Fup, 3), (q.I, 1); itag = "L3"))
show3("F2", F2)
F2up = getsub(F2, 2, s -> s == up ? Colon() : nothing)
show3("F2up", F2up)
println("\nlink dirs: Fup leg3 = ", Fup.inds[3].dir, " ; F2up leg1 = ", F2up.inds[1].dir)
for (lbl, f) in [("contract(Fup,3,F2up,1)", () -> contract(Fup, (3,), F2up, (1,))),
                 ("contract(Fup,3,F2up',1)", () -> contract(Fup, (3,), F2up', (1,)))]
    try
        r = f(); println(rpad(lbl, 26), " -> OK rank ", length(r.inds), " legs ", r.inds)
    catch e
        println(rpad(lbl, 26), " -> THREW: ", first(split(sprint(showerror, e), '\n')))
    end
end

# ── E. Norm and isometry of a single product site tensor ────────────────────
hdr("E. norms")
println("norm(V)    = ", norm(V))
println("norm(F)    = ", norm(F))
println("norm(Fup)  = ", norm(Fup))
println("norm(Fdn)  = ", norm(Fdn))

# ── F. Gram matrix: does `gram - 1` work for the isometry test? ─────────────
hdr("F. gram via primed contraction, and `gram - 1`")
A = Fup
gram_greedy = A' * A
println("greedy A'*A rank : ", length(gram_greedy.inds))
Ap = prime(A, 3)
gram = contract(Ap', (1, 2), A, (1, 2))
show3("gram", gram)
println("gram[] if rank0  : ", length(gram.inds) == 0 ? string(gram[]) : "n/a")
for (lbl, f) in [("gram - 1", () -> gram - 1),
                 ("norm(gram)", () -> norm(gram))]
    try
        println(rpad(lbl, 16), " -> ", f() isa Number ? f() : norm(f()))
    catch e
        println(rpad(lbl, 16), " -> THREW: ", first(split(sprint(showerror, e), '\n')))
    end
end

# ── G. <Sz> on a single site tensor ─────────────────────────────────────────
hdr("G. <Sz> for the up site tensor")
# A has legs (link_l, site, link_r). Sz has (s '+', s '-').
println("A legs  : ", A.inds)
println("Sz legs : ", q.Sz.inds)
for (lbl, f) in [
      ("contract(A,(2,),Sz,(1,))",  () -> contract(A, (2,), q.Sz, (1,))),
      ("contract(A,(2,),Sz,(2,))",  () -> contract(A, (2,), q.Sz, (2,))),
    ]
    try
        r = f(); println(rpad(lbl, 28), " -> OK rank ", length(r.inds), " legs ", r.inds)
    catch e
        println(rpad(lbl, 28), " -> THREW: ", first(split(sprint(showerror, e), '\n')))
    end
end

println("\nPROBE2_DONE")
