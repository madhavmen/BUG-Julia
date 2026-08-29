# WHICH MATRIX ELEMENTS OF THE DOUBLED-CHAIN MPO ARE WRONG?
#
# ⛔ WHY THE EARLIER OPERATOR CHECK MISSED THIS. `mpo_vs_dense.jl` compared the SCALAR `<v|W|v>`
# for six probes -- and all six were snapshots of ONE trajectory, hence nearly collinear. Six
# nearly-parallel quadratic forms in a 64-dimensional space cannot pin down a 64x64 operator, so
# it passed at 1e-15 while the operators differ. (`dt`-refinement then showed a FLAT 1.478e-01
# error from dt=0.1 to dt=0.0125 -- order 0.00 -- and a consistent integrator's error vanishes
# with dt, so the fault cannot be the solver.)
#
# THIS COMPARES A FULL VECTOR INSTEAD OF A SCALAR. For a tiny step,
#     (psi(dt) - psi(0)) / dt  ->  W psi(0)
# component by component, and the splitting error is O(dt) in that quotient. Comparing against
# `Schain * v0` therefore tests EVERY row of `W` that `v0` reaches -- and the largest-deviation
# component is reported with its (ket, bra) labels, which says WHICH term is wrong rather than
# only THAT something is.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "response_check.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

const SX = ComplexF64[0 .5; .5 0]; const SY = ComplexF64[0 -.5im; .5im 0]
const SZ = ComplexF64[.5 0; 0 -.5];  const ID = ComplexF64[1 0; 0 1]
function site_op(op, j, n)
    m = ComplexF64[1;;]
    for k in 1:n
        m = kron(m, k == j ? op : ID)
    end
    return m
end
bits(x, n) = join(((x >> (n - k)) & 1) == 1 ? "d" : "u" for k in 1:n)

const L = 2; const N = 2L; const D = 2^L
ket(j) = 2j - 1; bra(j) = 2j
const JM = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
const HD = sum(JM[i,j] * (site_op(SX,i,L)*site_op(SX,j,L) + site_op(SY,i,L)*site_op(SY,j,L))
               for i in 1:L for j in (i+1):L if JM[i,j] != 0.0)
function c2ab(idx)
    a = 0; b = 0
    for j in 1:L
        a |= ((idx >> (N - ket(j))) & 1) << (L - j)
        b |= ((idx >> (N - bra(j))) & 1) << (L - j)
    end
    return a, b
end
const PERM = [(ab = c2ab(idx); ab[2]*D + ab[1] + 1) for idx in 0:(4^L - 1)]

function dense_super(ket_h, bra_h, gamma)
    S = zeros(ComplexF64, D*D, D*D)
    for a in 1:D, b in 1:D
        E = zeros(ComplexF64, D, D); E[a,b] = 1.0
        R = zeros(ComplexF64, D, D)
        ket_h && (R .-= im .* (HD*E))
        bra_h && (R .+= im .* (E*HD))
        if gamma != 0.0
            for j in 1:L
                Z = site_op(SZ, j, L); R .+= gamma .* (Z*E*Z .- 0.5 .* (Z*Z*E .+ E*Z*Z))
            end
        end
        S[:, (b-1)*D + a] = vec(R)
    end
    return S[PERM, PERM]
end

function mpo_of(ket_h, bra_h, gamma)
    v = pair_vertices(xxz_chain(N; J = 1.0, delta = 1.0))
    izz = findall(t -> t.left === t.right, v); length(izz) == 1 || error("vertex count")
    Jh = zeros(ComplexF64, N, N)
    for i in 1:L, j in (i+1):L
        JM[i,j] == 0.0 && continue
        if ket_h
            a, b = minmax(ket(i), ket(j)); Jh[a,b] += -im * JM[i,j]
        end
        if bra_h
            c, d = minmax(bra(i), bra(j)); Jh[c,d] += +im * JM[i,j]
        end
    end
    Js = Any[copy(Jh) for _ in v]
    Jz = zeros(ComplexF64, N, N)
    for j in 1:L
        a, b = minmax(ket(j), bra(j)); Jz[a,b] += gamma
    end
    Js[izz[1]] = Jz
    return pair_mpo(N, Vector{AbstractMatrix}(Js), v)
end

# probe states: PRODUCT states over the whole doubled chain -- deliberately spanning, not one
# trajectory. Each reaches a different row of W.
const PROBES = [[:up,:up,:down,:down], [:up,:down,:up,:down], [:down,:up,:up,:up],
                [:up,:up,:up,:down]]
const DT = 1e-6
for (nm, kh, bh, g) in (("ket half", true, false, 0.0), ("bra half", false, true, 0.0),
                        ("gamma=0 both", true, true, 0.0), ("gamma=0.5 full", true, true, 0.5))
    S = dense_super(kh, bh, g)
    W = mpo_of(kh, bh, g)
    cshift = -g * L / 4
    worst = 0.0; wl = ""
    for sp in PROBES
        p0 = product_state(sp); v0 = dense_state(p0)
        p = copy(p0)
        tdvp2_step!(p, W, ComplexF64(DT); maxdim = 256, trunc_thresh = 1e-14, maxiter = 30,
                    hermitian = false)
        cshift != 0.0 && (p[1] = to_concrete(ComplexF64(exp(DT * cshift)) * p[1]))
        got  = (dense_state(p) .- v0) ./ DT
        want = S * v0
        for idx in 0:(4^L - 1)
            d = abs(got[idx+1] - want[idx+1])
            if d > worst
                a, b = c2ab(idx)
                worst = d
                wl = @sprintf("start %s -> component (ket %s, bra %s): got %+.6f%+.6fi  want %+.6f%+.6fi",
                              join(string.(sp), ""), bits(a, L), bits(b, L),
                              real(got[idx+1]), imag(got[idx+1]),
                              real(want[idx+1]), imag(want[idx+1]))
            end
        end
    end
    say(@sprintf("%-16s worst |W_mps v - W_dense v| = %.4e  %s", nm, worst,
                 worst < 1e-5 ? "PASS" : "FAIL"))
    worst >= 1e-5 && say("     " * wl)
end
say("")
say("A PASS here with a FAIL in lindblad_diag would mean the operator is right after all and the")
say("fault is downstream. A FAIL names the exact matrix element that is wrong.")
close(io)
