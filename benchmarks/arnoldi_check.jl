# WHICH SCHEME'S NON-HERMITIAN PATH IS WRONG?
#
# The ket-half of the doubled-chain Lindbladian disagrees with `psi(t) (x) psi(0)` by 1.489e-01,
# while `pair_mpo` was proven identical to `xxz_mpo(delta=0)`. Two suspects remain: the OPERATOR
# (settled separately by `mpo_vs_dense.jl`, which needs no solver) and the SOLVER. This isolates
# the solver by running the SAME operator through all three schemes.
#
# `tdvp2` has neither CBE nor a Krylov growth frame -- only `expv`. So:
#   all three fail  -> `expv`'s Arnoldi branch, or the operator
#   only cbe_bug    -> the CBE / `_krylov_frame` non-Hermitian path
#
# ⚠ Truncation is NOT a confound here: the bra half is untouched, so the interleaved chain needs
# only chi = chi_ket <= 2^(L/2), far under the cap.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "arnoldi_check.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

const L, DT, NST = 3, 0.05, 6
const N = 2L
ket(j) = 2j - 1
const JNN = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
const SPINS = [:up, :down, :up]

vN  = pair_vertices(xxz_chain(N; J = 1.0, delta = 1.0))
izN = findall(t -> t.left === t.right, vN)
length(izN) == 1 || error("expected exactly one S^zS^z vertex")
Jk = zeros(ComplexF64, N, N)
for i in 1:L, j in (i + 1):L
    JNN[i, j] == 0.0 && continue
    a, b = minmax(ket(i), ket(j)); Jk[a, b] += -im * JNN[i, j]
end
Jsv = Any[copy(Jk) for _ in vN]; Jsv[izN[1]] = zeros(ComplexF64, N, N)
const WK = pair_mpo(N, Vector{AbstractMatrix}(Jsv), vN)   # -i * H_XY on the KET sites only
const WX = xxz_mpo(L; J = 1.0, delta = 0.0)               # the same H on the plain chain

pt = product_state(SPINS)
for _ in 1:NST
    tdvp2_step!(pt, WX, ComplexF64(-im * DT); maxdim = 64, trunc_thresh = 1e-12, maxiter = 16)
end
const D0 = dense_state(product_state(SPINS))
const DP = dense_state(pt)

function split(idx)
    a = 0; b = 0
    for j in 1:L
        a |= ((idx >> (N - (2j - 1))) & 1) << (L - j)
        b |= ((idx >> (N - 2j))       & 1) << (L - j)
    end
    return a, b
end

function dev_of(step!)
    sk = product_state(collect(Iterators.flatten((s, s) for s in SPINS)))
    for _ in 1:NST
        step!(sk)
    end
    dk = dense_state(sk)
    return maximum(begin
                       a, b = split(idx)
                       abs(dk[idx + 1] - DP[a + 1] * D0[b + 1])
                   end for idx in 0:(4^L - 1))
end

say("ket half alone, evolved by each scheme, against the exact psi(t) (x) psi(0):")
say("")
say(@sprintf("  %-14s %14s %10s", "scheme", "max dev", "verdict"))
for (nm, f) in (
    ("tdvp2",      p -> tdvp2_step!(p, WK, ComplexF64(DT); maxdim = 64, trunc_thresh = 1e-12,
                                    maxiter = 16, hermitian = false)),
    ("tdvp_cbe1s", p -> tdvp_cbe1s_step!(p, WK, ComplexF64(DT); maxdim = 64,
                                         trunc_thresh = 1e-12, maxiter = 16, hermitian = false)),
    ("cbe_bug m=3", p -> cbe_bug_step!(p, WK, ComplexF64(DT); exact = true, krylov_basis = 3,
                                       krylov_tol = 0.0, hermitian = false, maxdim = 64,
                                       trunc_thresh = 1e-12, maxiter = 16)),
    ("cbe_bug m=0", p -> cbe_bug_step!(p, WK, ComplexF64(DT); exact = true, krylov_basis = 0,
                                       krylov_tol = 0.0, hermitian = false, maxdim = 64,
                                       trunc_thresh = 1e-12, maxiter = 16)),
    ("tdvp2 HERM",  p -> tdvp2_step!(p, WK, ComplexF64(DT); maxdim = 64, trunc_thresh = 1e-12,
                                     maxiter = 16, hermitian = true)))
    d = try
        dev_of(f)
    catch e
        say("  $nm THREW: " * sprint(showerror, e)[1:min(end, 200)]); continue
    end
    say(@sprintf("  %-14s %14.3e %10s", nm, d, d < 1e-8 ? "PASS" : "FAIL"))
end
say("")
say("The last row runs the NON-Hermitian operator through the LANCZOS branch on purpose: if it")
say("matches the `hermitian = false` rows, the branch flag is not reaching the arithmetic.")
close(io)
