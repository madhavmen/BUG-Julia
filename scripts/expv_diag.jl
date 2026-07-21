using LurCGT, Telum, LinearAlgebra
using BUGJulia.BondUpdateBUG

function mps_sum(a::SymMPS, b::SymMPS)
    L = length(a); ts = Any[]
    for i in 1:L
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(ts, to_concrete(oplus(a[i], b[i], dims)))
    end
    SymMPS(ts, L)
end

ent = mps_sum(product_state([:up,:down,:up,:down]), product_state([:down,:up,:up,:down]))
canonical!(ent, 1)
fe = bond_frame(ent, 1)
xe = to_concrete(fe.U0 * fe.S0)

println("leg3 spaces : ", xe.spaces[3])
sector_parts(v) = [(s, getsub(v, 3, q -> q == s ? Colon() : nothing; preserve_space=true))
                   for (s,_) in v.spaces[3]]
for (s,p) in sector_parts(xe)
    println("  sector ", s, "  norm ", norm(p))
end
println("sum of parts == xe ? ", norm(to_concrete(reduce(+,(p for (_,p) in sector_parts(xe)))) - xe))

diag_apply(sc) = v -> to_concrete(reduce(+, (sc(s)*p for (s,p) in sector_parts(v))))
diag_exact(sc,tau,v) = to_concrete(reduce(+, (exp(tau*sc(s))*p for (s,p) in sector_parts(v))))

# instrumented Lanczos: report alpha/beta actually built
function probe_lanczos(apply, tau, x; tol=1e-13, maxiter=60)
    xc = to_concrete(x*(1.0+0.0im)); b0 = norm(xc)
    v = to_concrete((1/b0)*xc); basis=Any[v]; alpha=Float64[]; betas=Float64[]
    w = apply(v); a = real(tensor_inner(v,w)); push!(alpha,a); w = to_concrete(w + (-a)*v)
    for _ in 2:maxiter
        b = norm(w); b < tol && break
        push!(betas,b); v = to_concrete((1/b)*w); push!(basis,v)
        w = apply(v); a = real(tensor_inner(v,w)); push!(alpha,a)
        w = to_concrete(w + (-a)*v + (-b)*basis[end-1])
    end
    (alpha, betas, length(basis))
end

for (lbl, sc) in [
    ("real scales            ", s -> 0.5*s[1][1] + 1.0),
    ("complex, CONSTANT imag ", s -> (0.5*s[1][1] + 1.0) + 0.8im),
    ("complex, VARYING imag  ", s -> (0.5*s[1][1] + 1.0) + 0.3im*s[1][1]),
  ]
    println("\n=== ", lbl, " ===")
    for (s,_) in xe.spaces[3]; println("   scale(", s, ") = ", sc(s)); end
    al, be, k = probe_lanczos(diag_apply(sc), -0.35+0.0im, xe)
    println("   krylov dim = ", k, "   alpha = ", al, "   beta = ", be)
    tau = -0.35+0.0im
    ex = diag_exact(sc,tau,xe)
    yl = expv(diag_apply(sc), tau, xe; hermitian=true)
    ya = expv(diag_apply(sc), tau, xe; hermitian=false)
    println("   |lanczos - exact| = ", norm(yl-ex))
    println("   |arnoldi - exact| = ", norm(ya-ex))
end

# structurally-zero inner product: SAME spaces, DISJOINT stored sectors
s1 = first(xe.spaces[3])[1]
a = getsub(xe, 3, q -> q == s1 ? Colon() : nothing; preserve_space=true)
b = getsub(xe, 3, q -> q == s1 ? nothing : Colon(); preserve_space=true)
println("\nspaces equal? ", a.spaces[3] == b.spaces[3])
println("nsectors a=", length(to_concrete(a)), " b=", length(to_concrete(b)))
println("tensor_inner(a,b) = ", tensor_inner(a,b))
println("\nDIAG_DONE")
