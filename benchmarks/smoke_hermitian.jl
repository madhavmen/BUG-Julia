# SMOKE TEST 1: the Hessenberg rewrite and the `hermitian` flag must change NOTHING closed.
#
# `_krylov_frame` now accumulates the full upper-Hessenberg projection instead of rebuilding a
# symmetric tridiagonal from `real(<v,Hv>)`. For a Hermitian H the Hessenberg IS that tridiagonal,
# so every closed-system number must be reproduced BIT FOR BIT. If it is not, the rewrite has
# changed the closed-system results and the open-system work would be built on a moved baseline.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
const OUT = joinpath(@__DIR__, "results", "smoke_hermitian.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io,l); flush(io); println(stdout,l); flush(stdout))
set_symmetry!(:U1)
L = 12
mpo, psi0 = xxz_mpo(L; J=1.0, delta=1.0), neel_state(L)
say("REFERENCE VALUES (record these; they must not move):")
for (nm, kw) in (("m=0",   (krylov_basis=0,  krylov_tol=0.0)),
                 ("m=3",   (krylov_basis=3,  krylov_tol=0.0)),
                 ("m=30",  (krylov_basis=30, krylov_tol=0.0)),
                 ("tol1e-6",(krylov_basis=30, krylov_tol=1e-6)),
                 ("tol1e-8",(krylov_basis=30, krylov_tol=1e-8)))
    psi, kry = copy(psi0), 0
    for _ in 1:10
        i = cbe_bug_step!(psi, mpo, ComplexF64(-im*0.05); exact=true, maxdim=32,
                          trunc_thresh=1e-10, maxiter=16, kw...)
        kry += i.krylov_dims
    end
    m = magnetisation(copy(psi)) ./ max(norm(psi)^2, eps())
    say(@sprintf("  cbe_bug %-8s  sum|Sz| = %.14e   chi=%d  kry=%d", nm, sum(abs.(m)),
                 maximum(state_bond_dims(psi)), kry))
end
# the explicit flag on a HERMITIAN problem must be a no-op
for h in (true, false)
    psi = copy(psi0)
    for _ in 1:10
        cbe_bug_step!(psi, mpo, ComplexF64(-im*0.05); exact=true, krylov_basis=3,
                      krylov_tol=0.0, maxdim=32, trunc_thresh=1e-10, maxiter=16, hermitian=h)
    end
    m = magnetisation(copy(psi)) ./ max(norm(psi)^2, eps())
    say(@sprintf("  cbe_bug hermitian=%-5s sum|Sz| = %.14e", h, sum(abs.(m))))
end
for (nm, f) in (("tdvp2", (p,t,h)->tdvp2_step!(p,mpo,t; maxdim=32, trunc_thresh=1e-10,
                                               maxiter=16, hermitian=h)),
                ("tdvp_cbe1s", (p,t,h)->tdvp_cbe1s_step!(p,mpo,t; maxdim=32, trunc_thresh=1e-10,
                                                          maxiter=16, hermitian=h)))
    for h in (true, false)
        psi = copy(psi0)
        for _ in 1:10; f(psi, ComplexF64(-im*0.05), h); end
        m = magnetisation(copy(psi)) ./ max(norm(psi)^2, eps())
        say(@sprintf("  %-11s hermitian=%-5s sum|Sz| = %.14e", nm, h, sum(abs.(m))))
    end
end
say("")
say("GATE: within each scheme, hermitian=true and hermitian=false must agree to ~1e-13 on a")
say("HERMITIAN problem (Arnoldi and Lanczos span the same space there). A large gap means the")
say("Arnoldi branch is broken and every open-system result would inherit it.")
close(io)
