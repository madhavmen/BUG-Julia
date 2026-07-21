# Probe 2: build the two-site gate and check it against the analytic
# Heisenberg matrix elements <bra|H|ket> in the Sz product basis.
using LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG

q = local_space()
TL, TR = "1,S", "2,S"
retag2(O, tag) = to_concrete(setitag(setitag(O, 1, tag), 2, tag))

println("=== adjoint leg structure ===")
SpR = retag2(q.Sp, TR)
for (k, ix) in enumerate(SpR.inds);  println("  SpR  leg $k: $(ix.itags) $(ix.dir) $(SpR.spaces[k])"); end
A = SpR'
for (k, ix) in enumerate(A.inds);    println("  SpR' leg $k: $(ix.itags) $(ix.dir) $(A.spaces[k])"); end

# candidate builders -> gate with legs (ket_l, ket_r, bra_l, bra_r)
function build_terms()
    SpL = retag2(q.Sp, TL); SpR = retag2(q.Sp, TR)
    SmL = retag2(q.Sm, TL); SmR = retag2(q.Sm, TR)
    SzL = retag2(q.Sz, TL); SzR = retag2(q.Sz, TR)
    d = Dict{String,Any}()
    for (nm, f) in (
        ("pm_adj",  () -> permutedims(contract(SpL, (3,), SpR', (3,)), (1, 4, 2, 3))),
        ("mp_adj",  () -> permutedims(contract(SmL, (3,), SmR', (3,)), (1, 4, 2, 3))),
        ("pm_alt",  () -> permutedims(contract(SpL, (3,), SpR', (3,)), (1, 3, 2, 4))),
        ("zz",      () -> permutedims(contract(SzL, (), SzR, ()), (1, 3, 2, 4))),
    )
        try; d[nm] = to_concrete(f()) catch e
            println("  build $nm FAILED: ", sprint(showerror, e)[1:min(end,160)]) end
    end
    return d
end

apply_g(G, th) = to_concrete(permutedims(contract(G, (1, 2), th, (2, 3)), (3, 1, 2, 4)))

function elem(G, bra, ket)
    pk = product_state(ket); pb = product_state(bra)
    thk = to_concrete(pk[1] * pk[2]); thb = to_concrete(pb[1] * pb[2])
    ev = apply_g(G, thk)
    return contract(ev, (1, 2, 3, 4), thb', (1, 2, 3, 4))[]
end

basis = [[:up, :up], [:up, :down], [:down, :up], [:down, :down]]
lbl = ["uu", "ud", "du", "dd"]

terms = build_terms()
for nm in sort(collect(keys(terms)))
    println("\n=== $nm : rows=bra, cols=ket ===")
    G = terms[nm]
    println("  legs: ", [(ix.itags, ix.dir) for ix in G.inds])
    for (bi, b) in enumerate(basis)
        row = String[]
        for (ki, k) in enumerate(basis)
            v = try elem(G, b, k) catch e; NaN end
            push!(row, string(round(real(v), digits = 6)))
        end
        println("   <$(lbl[bi])| : ", join(row, "  "))
    end
end
