# IS THE DOUBLED-CHAIN DISAGREEMENT A BUG, OR IS IT SPLITTING ERROR?
#
# ⛔ THE EARLIER DIAGNOSTICS COULD NOT TELL. `half_check` compared the doubled chain against a
# reference that was ITSELF produced by `tdvp2`, on a DIFFERENT chain (L sites vs 2L), so it
# subtracted one Trotter splitting error from another and called the remainder a fault. `tdvp2`
# and `cbe_bug` then returned the SAME 1.489e-01 -- two structurally different schemes cannot
# share a solver bug to four digits, but they can share a splitting error.
#
# The test that separates them is dt-REFINEMENT AT FIXED TOTAL TIME against the EXACT dense
# propagator `exp(T*L)`:
#
#     deviation ~ dt^2  -> splitting error. Not a bug; the doubled chain just needs smaller dt.
#     deviation ~ const -> a real fault, and it survives dt -> 0.
#
# ⚠ THE DOUBLED CHAIN IS INTRINSICALLY HARDER FOR A SWEEP. Interleaving puts the system-system
# coupling at RANGE 2 with the partner's ancilla sitting in between, so no two-site block ever
# contains a whole interaction term and every term is split. A larger prefactor is expected here
# than on the plain chain -- what must NOT happen is a non-vanishing floor.
#
# ⛔ AND THE DISSIPATOR'S CONSTANT IS NOT IN THE MPO. For `L_j = S^z_j`, `L†L = 1/4`, so the
# anticommutator is `-(gamma*L/4)` times the IDENTITY -- a zero-body term, which `pair_mpo` (two-
# body vertices only) structurally cannot hold, and it was silently dropped. It needs no MPO
# change: `exp(tau*(W + c*I)) = e^(tau*c) * exp(tau*W)`, so it is an EXACT scalar prefactor,
# applied here per step. Without it `Tr rho` inflates -- MEASURED 1.1618 at gamma = 0.5.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "lindblad_diag.txt"); mkpath(dirname(OUT))
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

"Dense superoperator in CHAIN basis, the MPO, and the zero-body constant that the MPO omits."
function build(L::Int, gamma::Float64)
    N = 2L; D = 2^L
    ket(j) = 2j - 1; bra(j) = 2j
    JM = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    HD = sum(JM[i,j] * (site_op(SX,i,L)*site_op(SX,j,L) + site_op(SY,i,L)*site_op(SY,j,L))
             for i in 1:L for j in (i+1):L if JM[i,j] != 0.0)
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
            a |= ((idx >> (N - ket(j))) & 1) << (L - j)
            b |= ((idx >> (N - bra(j))) & 1) << (L - j)
        end
        return b*D + a + 1
    end
    perm = [c2v(idx) for idx in 0:(4^L - 1)]
    v = pair_vertices(xxz_chain(N; J = 1.0, delta = 1.0))
    izz = findall(t -> t.left === t.right, v); length(izz) == 1 || error("vertex count")
    Jh = zeros(ComplexF64, N, N)
    for i in 1:L, j in (i+1):L
        JM[i,j] == 0.0 && continue
        a, b = minmax(ket(i), ket(j)); Jh[a,b] += -im * JM[i,j]
        c, d = minmax(bra(i), bra(j)); Jh[c,d] += +im * JM[i,j]
    end
    Js = Any[copy(Jh) for _ in v]
    Jz = zeros(ComplexF64, N, N)
    for j in 1:L
        a, b = minmax(ket(j), bra(j)); Jz[a,b] += gamma
    end
    Js[izz[1]] = Jz
    return S[perm, perm], pair_mpo(N, Vector{AbstractMatrix}(Js), v), -gamma * L / 4
end

const SPINS = Dict(2 => [:up, :down], 3 => [:up, :down, :up])
sk0(L) = product_state(collect(Iterators.flatten((s, s) for s in SPINS[L])))
"Multiply the whole MPS by a scalar (the omitted zero-body term)."
scale!(p, z) = (p[1] = to_concrete(ComplexF64(z) * p[1]); p)

const T = 0.3
for L in (2, 3), gamma in (0.0, 0.5)
    Schain, W, cshift = build(L, gamma)
    N = 2L
    v0 = dense_state(sk0(L))
    want = exp(Schain * T) * v0
    # trace of the exact answer, read straight off the chain-basis vector
    diagidx = [idx for idx in 0:(4^L - 1) if
               all(((idx >> (N - (2j-1))) & 1) == ((idx >> (N - 2j)) & 1) for j in 1:L)]
    say(@sprintf("── L = %d system sites (%d chain sites), gamma = %.2f, T = %.2f, exact Tr rho = %.10f",
                 L, N, gamma, T, real(sum(want[i+1] for i in diagidx))))
    say(@sprintf("     %-12s %6s %14s %14s %8s", "scheme", "steps", "max dev", "Tr rho", "order"))
    for (nm, f) in (
        ("tdvp2",       (p, dt) -> tdvp2_step!(p, W, ComplexF64(dt); maxdim = 256,
                            trunc_thresh = 1e-14, maxiter = 30, hermitian = false)),
        ("cbe_bug m=0", (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                            krylov_basis = 0, krylov_tol = 0.0, hermitian = false,
                            maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)))
        prev = NaN
        for n in (3, 6, 12, 24)
            dt = T / n
            p = copy(sk0(L))
            d, tr = try
                for _ in 1:n
                    f(p, dt)
                    cshift != 0.0 && scale!(p, exp(dt * cshift))
                end
                dk = dense_state(p)
                (maximum(abs.(dk .- want)), real(sum(dk[i+1] for i in diagidx)))
            catch e
                say("       $nm THREW: " * sprint(showerror, e)[1:min(end, 160)]); break
            end
            ord = isnan(prev) ? "" : @sprintf("%.2f", log2(prev / d))
            say(@sprintf("     %-12s %6d %14.3e %14.10f %8s", nm, n, d, tr, ord))
            prev = d
        end
    end
    say("")
end
say("order ~ 2.0 down the column = SPLITTING error and no fault. A flat column = a real bug.")
say("Tr rho must sit at the exact value in every row; that is the omitted constant doing its job.")
close(io)
