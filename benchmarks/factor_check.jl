# WHAT NORMALISATION DOES `pair_mpo` + `xxz_chain` ACTUALLY USE? Measured, not assumed.
# Builds the MPO's DENSE matrix by applying it to each basis state, and compares against an
# explicitly written XX Hamiltonian. Any constant ratio is the convention factor.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "factor_check.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io,l); flush(io); println(stdout,l); flush(stdout))
set_symmetry!(:none)
L = 4
SX = ComplexF64[0 .5; .5 0]; SY = ComplexF64[0 -.5im; .5im 0]; SZ = ComplexF64[.5 0; 0 -.5]
ID = ComplexF64[1 0; 0 1]
sop(o,j) = (m = ComplexF64[1;;]; for k in 1:L; m = kron(m, k==j ? o : ID); end; m)
Jnn = [abs(i-j)==1 && i<j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
HD_xx  = sum(sop(SX,i)*sop(SX,j) + sop(SY,i)*sop(SY,j) for i in 1:L for j in (i+1):L if Jnn[i,j]!=0)
HD_zz  = sum(sop(SZ,i)*sop(SZ,j) for i in 1:L for j in (i+1):L if Jnn[i,j]!=0)
"Dense matrix of an MPO, by applying it to each computational basis state."
function mpo_dense(W, L)
    D = 2^L; M = zeros(ComplexF64, D, D)
    for c in 0:(D-1)
        spins = [((c >> (L-j)) & 1)==0 ? :up : :down for j in 1:L]
        M[:, c+1] = dense_state(apply_mpo(W, product_state(spins)))
    end
    M
end
for (nm, delta, want) in (("XX  (delta=0)", 0.0, HD_xx), ("ZZ-only", nothing, HD_zz))
    v = delta === nothing ? pair_vertices(xxz_chain(L; J=1.0, delta=1.0)) :
                            pair_vertices(xxz_chain(L; J=1.0, delta=delta))
    Js = if delta === nothing
        izz = findall(t -> t.left === t.right, v)
        z = [zeros(Float64,L,L) for _ in v]; z[izz[1]] = copy(Jnn); Vector{AbstractMatrix}(z)
    else
        Vector{AbstractMatrix}([copy(Jnn) for _ in v])
    end
    try
        M = mpo_dense(pair_mpo(L, Js, v), L)
        r = [abs(want[i,j]) > 1e-12 ? M[i,j]/want[i,j] : missing for i in 1:2^L, j in 1:2^L]
        rs = unique(round.(ComplexF64[x for x in r if x !== missing], digits=8))
        say(@sprintf("  %-14s  ratio MPO/explicit = %s", nm, string(rs)))
    catch e
        say("  $nm THREW: " * sprint(showerror,e)[1:min(end,200)])
    end
end
close(io)
