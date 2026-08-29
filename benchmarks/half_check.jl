# WHICH HALF OF THE DOUBLED CHAIN IS WRONG? At gamma = 0 the Lindbladian FACTORISES:
#     rho(t) = e^{-iHt} rho0 e^{+iHt},   so for rho0 = |psi><psi| the superket is
#     |psi(t)> (x) |psi*(t)>   with psi(t) = e^{-iHt} psi.
# So evolving the KET half alone must reproduce a plain closed-system evolution of psi, and the
# BRA half alone its conjugate. Testing them separately says which term is at fault instead of
# only that the pair is.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "half_check.txt"); mkpath(dirname(OUT))
io = open(OUT,"w"); say(l)=(println(io,l);flush(io);println(stdout,l);flush(stdout))
set_symmetry!(:none)
L, DT, NST = 3, 0.05, 6
Jnn = [abs(i-j)==1 && i<j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
SPINS = [:up, :down, :up]

# (a) is the SINGLE-CHAIN XX MPO from pair_mpo the same operator as the validated xxz_mpo?
v0  = pair_vertices(xxz_chain(L; J=1.0, delta=1.0))
izz = findall(t -> t.left === t.right, v0)
Jxy = [k == izz[1] ? zeros(Float64,L,L) : copy(Jnn) for k in eachindex(v0)]
Wp  = pair_mpo(L, Vector{AbstractMatrix}(Jxy), v0)
Wx  = xxz_mpo(L; J=1.0, delta=0.0)
p   = product_state(SPINS)
say(@sprintf("(a) pair_mpo XY vs xxz_mpo(delta=0) on a product state: %.3e / %.3e",
             real(mpo_energy(copy(p), Wp)), real(mpo_energy(copy(p), Wx))))
# a product state has zero XY weight, so evolve one step and compare again -- that DOES probe it
q1 = copy(p); tdvp2_step!(q1, Wp, ComplexF64(-im*DT); maxdim=32, trunc_thresh=1e-12, maxiter=16)
q2 = copy(p); tdvp2_step!(q2, Wx, ComplexF64(-im*DT); maxdim=32, trunc_thresh=1e-12, maxiter=16)
say(@sprintf("    after one step, |<q1|q2>| = %.12f   (1.0 means the SAME operator)",
             abs(overlap(q1,q2))/(norm(q1)*norm(q2))))

# (b) the KET half of the doubled chain, in isolation: bra couplings zeroed
N = 2L
ket(j)=2j-1; bra(j)=2j
vN = pair_vertices(xxz_chain(N; J=1.0, delta=1.0))
izN = findall(t -> t.left === t.right, vN)
Jk = zeros(ComplexF64,N,N)
for i in 1:L, j in (i+1):L
    Jnn[i,j]==0 && continue
    a,b = minmax(ket(i),ket(j)); Jk[a,b] += -im*Jnn[i,j]
end
Js = Any[copy(Jk) for _ in vN]; Js[izN[1]] = zeros(ComplexF64,N,N)
Wk = pair_mpo(N, Vector{AbstractMatrix}(Js), vN)
sk = product_state(collect(Iterators.flatten((s,s) for s in SPINS)))
for _ in 1:NST
    cbe_bug_step!(sk, Wk, ComplexF64(DT); exact=true, krylov_basis=3, krylov_tol=0.0,
                  hermitian=false, maxdim=64, trunc_thresh=1e-12, maxiter=16)
end
# reference: evolve psi alone, then form psi(t) (x) psi(0)  (bra untouched)
pt = copy(p)
for _ in 1:NST
    cbe_bug_step!(pt, Wx, ComplexF64(-im*DT); exact=true, krylov_basis=3, krylov_tol=0.0,
                  maxdim=64, trunc_thresh=1e-12, maxiter=16)
end
dk, dp = dense_state(sk), dense_state(pt); d0 = dense_state(p)
# ⚠ WRAPPED IN A FUNCTION. A top-level `for` body is its own soft scope, so an accumulator
# `dev = max(dev, ...)` inside one creates a NEW local and the outer name stays undefined. This
# is the second time in this session -- a function (or a comprehension) is the fix, not care.
function _split(idx, L, N)
    a = 0; b = 0
    for j in 1:L
        a |= ((idx >> (N - (2j - 1))) & 1) << (L - j)
        b |= ((idx >> (N - 2j)) & 1) << (L - j)
    end
    return a, b
end
dev = maximum(begin
                  a, b = _split(idx, L, N)
                  abs(dk[idx + 1] - dp[a + 1] * d0[b + 1])
              end for idx in 0:(4^L - 1))
say(@sprintf("(b) KET half alone vs psi(t) (x) psi(0): max dev = %.3e   %s", dev,
             dev < 1e-8 ? "PASS -- the -i(H(x)I) term is right" : "FAIL -- the ket term is wrong"))
close(io)
