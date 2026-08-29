# SMOKE TEST: IS THE VECTORISED LINDBLADIAN THE OPERATOR WE THINK IT IS?
#
# Validated DENSELY at L = 4 (superket dimension 4^4 = 256), against a dense superoperator built
# by applying the Lindbladian to each basis matrix. That is a strictly stronger check than
# matching a few traced scalars: it compares EVERY component.
#
# ⛔ AND IT NEEDS NO "CUP". Tracing the superket means contracting two KET-type legs, which is a
# `sum_i |i,i>` object with both arrows the same direction -- NOT the identity operator, whose
# legs point opposite ways. `getIdentity` gives a fusion isometry, so contracting with it is a
# CHANGE OF BASIS and not a trace, and hand-rolling the cup risks the false pass `sectors.jl`
# warns about ("with a vacuum link the reachable set is its own dual, so the mislabelled version
# still agrees"). The dense comparison sidesteps the whole question.
#
# ── THE THREE GATES, IN THE ORDER A FAILURE WOULD MATTER ──────────────────────────────────
#
#   GATE 0  LAYOUT.  Before any dynamics: does the MPS superket for `rho0 = |psi><psi|` equal the
#           dense `psi psi^dag`, component by component, under the interleaved mapping? This
#           isolates an index-convention error from a dynamics error. ⚠ `psi` is deliberately
#           ASYMMETRIC (`up,down,up,up`) -- this repo's notes record that Neel and domain-wall
#           states are fixed points of the mirrored convention and HIDE exactly this bug.
#   GATE 1  gamma = 0 must reproduce CLOSED-system dynamics. The doubled chain factorises there,
#           so a disagreement is a construction error and not physics.
#   GATE 2  gamma > 0 must match the dense Lindbladian, and `Tr rho` must stay at 1 -- the trace
#           term `-gamma*L/4` is what makes that true, and dropping it inflates the trace.
#
# Run:  julia --project=. benchmarks/smoke_lindblad.jl [L] [gammas]

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))

const L      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const GAMMAS = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.0, 0.5]
const DT     = 0.05
const NST    = 6
const N      = 2L

const OUT = joinpath(@__DIR__, "results", "smoke_lindblad.txt"); mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:none)

ket(j) = 2j - 1
bra(j) = 2j

# ── the dense side ────────────────────────────────────────────────────────────────────────
const SX = ComplexF64[0 0.5; 0.5 0]
const SY = ComplexF64[0 -0.5im; 0.5im 0]
const SZ = ComplexF64[0.5 0; 0 -0.5]
const ID = ComplexF64[1 0; 0 1]

"`op` on system site `j` of an `L`-site chain, in `dense_state`'s convention (site 1 = MSB)."
function site_op(op, j, L)
    m = ComplexF64[1;;]
    for k in 1:L
        m = kron(m, k == j ? op : ID)
    end
    return m
end

"Chain index -> (ket config, bra config) under the INTERLEAVED layout."
function split_idx(idx::Int, L::Int)
    a = 0; b = 0
    for j in 1:L
        kbit = (idx >> (2L - ket(j))) & 1        # dense_state: chain site k reads bit (N-k)
        bbit = (idx >> (2L - bra(j))) & 1
        a |= kbit << (L - j)                     # system config: site j reads bit (L-j)
        b |= bbit << (L - j)
    end
    return a, b
end

# ── the MPO side ──────────────────────────────────────────────────────────────────────────
"Per-vertex couplings for `-i(H(x)I) + i(I(x)H) + g*sum_j Sz_j(x)Sz_j` on the doubled chain."
function lindblad_couplings(L, Jm, gamma, delta, v)
    izz = findall(t -> t.left === t.right, v)
    length(izz) == 1 || error("expected one S^zS^z vertex, found $(length(izz))")
    Jh = zeros(ComplexF64, 2L, 2L)
    for i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        a, b = minmax(ket(i), ket(j)); Jh[a, b] += -im * Jm[i, j]
        c, d = minmax(bra(i), bra(j)); Jh[c, d] += +im * Jm[i, j]
    end
    Js = Any[copy(Jh) for _ in v]
    Jz = delta .* Jh
    for j in 1:L
        a, b = minmax(ket(j), bra(j)); Jz[a, b] += gamma
    end
    Js[izz[1]] = Jz
    return Vector{AbstractMatrix}(Js)
end

const JM = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
const HD = sum(JM[i, j] * (site_op(SX, i, L) * site_op(SX, j, L) +
                           site_op(SY, i, L) * site_op(SY, j, L))
               for i in 1:L for j in (i + 1):L if JM[i, j] != 0.0)   # XX (delta = 0)

say(@sprintf("SMOKE: dephasing Lindbladian, L = %d system sites -> %d chain sites, dim 4^%d = %d",
             L, N, L, 4^L))
say("")

# ══ GATE 0: THE LAYOUT, before any dynamics ══════════════════════════════════════════════
const SPINS = [:up, :down, :up, :up][1:L]
psi_mps = product_state(collect(Iterators.flatten((s, s) for s in SPINS)))
sk = dense_state(psi_mps)
psi_d = zeros(ComplexF64, 2^L)
let a = 0
    for j in 1:L
        SPINS[j] === :down && (a |= 1 << (L - j))
    end
    psi_d[a + 1] = 1.0
end
rho_d = psi_d * psi_d'
# ⚠ A COMPREHENSION, NOT AN ACCUMULATOR LOOP. At top level a `for` body is its OWN soft scope,
# so `maxdev = max(maxdev, ...)` inside one creates a NEW local and the outer name stays
# undefined -- `UndefVarError: maxdev not defined in local scope`.
maxdev = maximum(abs(sk[idx + 1] - rho_d[split_idx(idx, L)[1] + 1, split_idx(idx, L)[2] + 1])
                 for idx in 0:(4^L - 1))
say(@sprintf("GATE 0 (layout, asymmetric psi = %s): max |MPS - dense| = %.3e   %s",
             join(string.(SPINS), ","), maxdev, maxdev < 1e-12 ? "PASS" : "FAIL"))
maxdev < 1e-12 || say("  ⛔ the interleaved index mapping is wrong; nothing below would mean anything.")
say("")

# ══ GATES 1 and 2: the dynamics ══════════════════════════════════════════════════════════
say(@sprintf("  %6s %14s %14s %14s %10s", "gamma", "max|MPS-dense|", "Tr rho", "|dTr|", "verdict"))
for g in GAMMAS
    v  = pair_vertices(xxz_chain(N; J = 1.0, delta = 1.0))
    W  = pair_mpo(N, lindblad_couplings(L, JM, g, 0.0, v), v)
    # dense superoperator: apply L to each basis matrix E_ab and read off the column
    D = 2^L
    Lsup = zeros(ComplexF64, D * D, D * D)
    for a in 1:D, b in 1:D
        E = zeros(ComplexF64, D, D); E[a, b] = 1.0
        R = -im * (HD * E - E * HD)
        for j in 1:L
            Z = site_op(SZ, j, L)
            R += g * (Z * E * Z - 0.5 * (Z * Z * E + E * Z * Z))
        end
        Lsup[:, (b - 1) * D + a] = vec(R)      # column-stacked vec
    end
    rho0v = vec(rho_d)
    rhot  = exp(Lsup * (DT * NST)) * rho0v
    # the MPS side
    #
    # ⛔ THE ZERO-BODY TERM IS NOT IN THE MPO, AND CANNOT BE. `L†L = 1/4` for `L = S^z`, so the
    # anticommutator is `-(gamma*L/4)` times the IDENTITY. `pair_mpo` builds TWO-BODY vertices
    # only, so it has nowhere to put a zero-body term and dropped it silently -- MEASURED as
    # `Tr rho = 1.1618` at `gamma = 0.5`, and as an 8.572e-01 operator-level disagreement in
    # `mpo_vs_dense.jl` (whose gamma = 0 rows pass at 3.5e-15, pinning the fault to this term).
    #
    # It needs no MPO change and costs nothing: `exp(tau*(W + c*I)) = e^(tau*c) * exp(tau*W)`, so
    # the constant is an EXACT scalar prefactor on the state. It is applied per step so that
    # `Tr rho` is right at every step and not only at the end.
    cshift = -g * L / 4
    psi = copy(psi_mps)
    for _ in 1:NST
        cbe_bug_step!(psi, W, ComplexF64(DT); exact = true, krylov_basis = 3, krylov_tol = 0.0,
                      hermitian = false, maxdim = 64, trunc_thresh = 1e-12, maxiter = 16)
        cshift != 0.0 && (psi[1] = to_concrete(ComplexF64(exp(DT * cshift)) * psi[1]))
    end
    skt = dense_state(psi)
    dev = maximum(abs(skt[idx + 1] - rhot[split_idx(idx, L)[2] * D + split_idx(idx, L)[1] + 1])
                  for idx in 0:(4^L - 1))
    tr = sum(ComplexF64[split_idx(idx, L)[1] == split_idx(idx, L)[2] ? skt[idx + 1] : 0.0 + 0im
                        for idx in 0:(4^L - 1)])
    say(@sprintf("  %6.2f %14.3e %14s %14.3e %10s", g, dev,
                 @sprintf("%.10f", real(tr)), abs(real(tr) - 1),
                 dev < 1e-8 ? "PASS" : "FAIL"))
end
say("")
say("GATE 1 is the gamma = 0 row (closed dynamics on a factorising chain); GATE 2 the rest.")
close(IO_)
