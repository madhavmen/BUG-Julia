# THE DOUBLED-CHAIN LAYOUT DECIDES WHETHER THE DYNAMICS CAN START AT ALL.
#
# ⛔ MEASURED: THE INTERLEAVED SUPERKET IS FROZEN. Its error against the exact propagator is FLAT
# in dt (order 0.00 from dt=0.1 to 0.0125), identical for `tdvp2` and `cbe_bug`, and equal to
# `|rho(T) - rho(0)|` to three digits -- at L=2, gamma=0, T=0.3 the largest change in rho is the
# coherence `|sin(0.15)*cos(0.15)| = 0.14776` and the measured "error" was 1.478e-01. The state
# never moves.
#
# ⛔ WHY, AND IT IS STRUCTURAL, NOT A BUG. Interleaving (ket at 2j-1, ancilla at 2j) puts every
# system-system term at RANGE 2 with the partner's ancilla in between, so NO two-site block ever
# contains both operators of a term. The block feels the term only through an MPO environment
# channel -- which for a half-open term is a ONE-SITE EXPECTATION of a charge-changing operator,
# and `<S^->` = 0 on ANY product state. Nothing can start. This is the same bootstrap failure the
# repo has hit before, arriving through the lattice layout rather than through the integrator.
#
# ⛔ THE BLOCKED LAYOUT DOES NOT HAVE IT. With ket at 1..L and ancilla at L+1..2L:
#     H on the ket   -> NEAREST NEIGHBOUR inside the ket block, fully inside a two-site update
#     H on the ancilla -> likewise
#     the jump term  -> range L, but it is `S^z (x) S^z`, and `<S^z>` is NONZERO on a product
#                       state, so a long-range DIAGONAL term does not suffer the freeze
# The cost is MPO width: L jump channels instead of 1. ⚠ That cost is not a regret here -- a wide
# MPO is exactly the regime the rSVD sketch needs, and §6 already wants one.
#
# Reported per layout: whether the state MOVED at all, the error against `exp(T*L_super)`, and the
# dt-refinement order (2 = a working integrator, 0 = frozen).
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "layout_bootstrap.txt"); mkpath(dirname(OUT))
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

const L = 3; const N = 2L; const D = 2^L; const T = 0.3
const SPINS = [:up, :down, :up]
const JM = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
const HD = sum(JM[i,j] * (site_op(SX,i,L)*site_op(SX,j,L) + site_op(SY,i,L)*site_op(SY,j,L))
               for i in 1:L for j in (i+1):L if JM[i,j] != 0.0)

"Site index of the ket / bra copy of system site `j`, per layout."
ket_of(lay, j) = lay === :interleaved ? 2j - 1 : j
bra_of(lay, j) = lay === :interleaved ? 2j     : L + j

function pieces(lay::Symbol, gamma::Float64)
    kf, bf = j -> ket_of(lay, j), j -> bra_of(lay, j)
    # dense superoperator, then permuted into THIS layout's chain order
    S = zeros(ComplexF64, D*D, D*D)
    for a in 1:D, b in 1:D
        E = zeros(ComplexF64, D, D); E[a,b] = 1.0
        R = -im * (HD*E - E*HD)
        if gamma != 0.0
            for j in 1:L
                Z = site_op(SZ, j, L); R += gamma * (Z*E*Z - 0.5*(Z*Z*E + E*Z*Z))
            end
        end
        S[:, (b-1)*D + a] = vec(R)
    end
    function c2v(idx)
        a = 0; b = 0
        for j in 1:L
            a |= ((idx >> (N - kf(j))) & 1) << (L - j)
            b |= ((idx >> (N - bf(j))) & 1) << (L - j)
        end
        return b*D + a + 1
    end
    perm = [c2v(idx) for idx in 0:(4^L - 1)]
    v = pair_vertices(xxz_chain(N; J = 1.0, delta = 1.0))
    izz = findall(t -> t.left === t.right, v); length(izz) == 1 || error("vertex count")
    Jh = zeros(ComplexF64, N, N)
    for i in 1:L, j in (i+1):L
        JM[i,j] == 0.0 && continue
        a, b = minmax(kf(i), kf(j)); Jh[a,b] += -im * JM[i,j]
        c, d = minmax(bf(i), bf(j)); Jh[c,d] += +im * JM[i,j]
    end
    Js = Any[copy(Jh) for _ in v]
    Jz = zeros(ComplexF64, N, N)
    for j in 1:L
        a, b = minmax(kf(j), bf(j)); Jz[a,b] += gamma
    end
    Js[izz[1]] = Jz
    W = pair_mpo(N, Vector{AbstractMatrix}(Js), v)
    # the start state, laid out the same way
    spins = Vector{Symbol}(undef, N)
    for j in 1:L
        spins[kf(j)] = SPINS[j]; spins[bf(j)] = SPINS[j]
    end
    return S[perm, perm], W, product_state(spins), -gamma * L / 4,
           maximum(mpo_virtual_dims(W))
end

for lay in (:interleaved, :blocked), gamma in (0.0, 0.5)
    Schain, W, p0, cshift, wdim = pieces(lay, gamma)
    v0 = dense_state(p0)
    want = exp(Schain * T) * v0
    frozen = maximum(abs.(want .- v0))
    say(@sprintf("── %-12s gamma = %.2f   MPO width %d   |rho(T)-rho(0)| = %.4e",
                 string(lay), gamma, wdim, frozen))
    say(@sprintf("     %-12s %6s %12s %12s %8s", "scheme", "steps", "moved by", "max dev", "order"))
    for (nm, f) in (
        ("tdvp2",       (p, dt) -> tdvp2_step!(p, W, ComplexF64(dt); maxdim = 256,
                            trunc_thresh = 1e-14, maxiter = 30, hermitian = false)),
        ("cbe_bug m=0", (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                            krylov_basis = 0, krylov_tol = 0.0, hermitian = false,
                            maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)))
        prev = NaN
        for n in (3, 12, 48)
            dt = T / n
            p = copy(p0)
            d, mv = try
                for _ in 1:n
                    f(p, dt)
                    cshift != 0.0 && (p[1] = to_concrete(ComplexF64(exp(dt * cshift)) * p[1]))
                end
                dk = dense_state(p)
                (maximum(abs.(dk .- want)), maximum(abs.(dk .- v0)))
            catch e
                say("       $nm THREW: " * sprint(showerror, e)[1:min(end, 150)]); break
            end
            ord = isnan(prev) ? "" : @sprintf("%.2f", log2(prev / d) / 2)
            say(@sprintf("     %-12s %6d %12.4e %12.4e %8s", nm, n, mv, d, ord))
            prev = d
        end
    end
    say("")
end
say("`moved by` ~ 0 means the state never left rho(0) -- the bootstrap freeze, not an accuracy")
say("problem. `order` ~ 2 with `moved by` ~ |rho(T)-rho(0)| is a working integrator.")
close(io)
