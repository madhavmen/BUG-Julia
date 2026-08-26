# THE VECTORISED LINDBLADIAN FOR DEPHASING, ON A DOUBLED MPS CHAIN.
#
# Zwolak & Vidal PRL 93 207205 (2004) and Verstraete/Garcia-Ripoll/Cirac PRL 93 207204 (2004):
# vectorise `rho` into a SUPERKET and evolve it like a state. The standard layout is an `L`-site
# chain of local dimension `d^2 = 4`; here it is UNFOLDED into `2L` spin-1/2 sites --
#
#     system site j  ->  chain site 2j-1        (the KET index)
#     ancilla       ->  chain site 2j           (the BRA index)
#
# -- because `4 = 2 x 2`, so the two are the same object, and the unfolded form reuses the
# existing spin-1/2 local space, `pair_mpo`, and the MPO ENVIRONMENTS. No new local space, no
# Trotter gates.
#
# ⛔ INTERLEAVED, NOT BLOCKED, AND THE REASON IS THE MPO WIDTH. With system 1..L and ancilla
# L+1..2L every jump term would have range `L`; interleaved they are NEAREST-NEIGHBOUR, at the
# cost of doubling the Hamiltonian's range. For a chain that is range 1 -> 2 against L jump
# channels, and for a 2D cylinder `Ly` -> `2Ly` against L. Interleaved wins in both.
#
#     d|rho>>/dt = L|rho>>,   L = -i(H⊗I) + i(I⊗H^T) + g*sum_j (S^z_j ⊗ S^z_j) - (g*L/4)*I
#
# so the three families are: system pairs `(2i-1, 2j-1)` at `-i*J`, ancilla pairs `(2i, 2j)` at
# `+i*J`, and jump pairs `(2j-1, 2j)` at `+g` on the `S^zS^z` vertex ONLY. All two-body.
#
# ⛔ THE TRACE TERM IS NOT DECORATION. `L†L = 1/4` for `L = S^z`, so the anticommutator is
# `-(g/4)` per site. Without it `Tr rho` is NOT conserved: the jump term alone contributes
# `<<I|S^z⊗S^z|rho>> = Tr[S^z rho S^z] = Tr[rho]/4`, and the constant is exactly what cancels it.
# `Tr rho = 1` is the conservation law here, IN PLACE OF the norm -- `|rho>>` is not normalised.
#
# ⛔ AND THIS RUNS UNDER `:none`, NOT `:U1`. The physical conserved quantity is
# `S^z_ket - S^z_bra` (the bra index carries CONJUGATE charge), but our local space would track
# the SUM. For `rho` block-diagonal in `S^z` the superket spans many sectors of that sum, so the
# naive U(1) is not the physical symmetry and would constrain the state wrongly. Dephasing keeps
# both charges separately conserved so `:U1` would happen to work, but the identity superket
# `|I>> = (|up,up> + |dn,dn>)` per pair is ITSELF a superposition of charge +1 and -1 -- so the
# OBSERVABLE would break it even when the dynamics does not.
#
# VALIDATION TARGET (Znidaric, New J. Phys. 12 043001): dephasing turns BALLISTIC transport
# DIFFUSIVE. From a domain wall, the transported magnetisation goes from `~t` at `g = 0` to
# `~sqrt(t)` at `g > 0`, and the diffusion constant scales as `D ~ 1/g` at strong dephasing.
#
# Run:  julia --project=. benchmarks/lindblad_dephasing.jl [mode] [L] [gammas] [t_max]
#         mode = smoke | transport

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const MODE   = length(ARGS) >= 1 ? ARGS[1] : "smoke"
const LSYS   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
const GAMMAS = length(ARGS) >= 3 ? parse.(Float64, split(ARGS[3], ",")) : [0.0, 0.5, 2.0]
const TMAX   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
const DT     = 0.05
const MI     = 16
const CAP    = 64

const OUT = joinpath(@__DIR__, "results", "lindblad_dephasing_$(MODE).txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:none)

ket(j) = 2j - 1
bra(j) = 2j

"""
Per-vertex coupling matrices on the `2L`-site doubled chain.

`vertices` comes from `xxz_chain(2L; delta = 1)` and its `S^zS^z` term is LAST -- the index is
DERIVED below rather than written down, because `Js[t]` pairs with `vertices[t]` by position and a
wrong guess silently builds a different Lindbladian.
"""
function lindblad_couplings(L::Int, Jm::AbstractMatrix, gamma::Float64, delta::Float64, v)
    N = 2L
    izz = findall(t -> t.left === t.right, v)
    length(izz) == 1 || error("expected one S^zS^z vertex, found $(length(izz))")
    Jxy = zeros(ComplexF64, N, N)      # for every vertex: H on ket (-i) and on bra (+i)
    for i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        a, b = minmax(ket(i), ket(j)); Jxy[a, b] += -im * Jm[i, j]
        c, d = minmax(bra(i), bra(j)); Jxy[c, d] += +im * Jm[i, j]
    end
    Js = Any[copy(Jxy) for _ in v]
    # the S^z vertex carries delta*(the same H structure) PLUS the jump term
    Jz = delta .* Jxy
    for j in 1:L
        a, b = minmax(ket(j), bra(j)); Jz[a, b] += gamma
    end
    Js[izz[1]] = Jz
    return Vector{AbstractMatrix}(Js)
end

"`|rho0>> = |psi0><psi0|` for a PRODUCT `psi0`: each pair `(2j-1, 2j)` carries `(s_j, s_j)`."
rho0_product(spins) = product_state(collect(Iterators.flatten((s, s) for s in spins)))

"""
`<<I|` as an MPS: each pair is `|up,up> + |dn,dn>` (UNNORMALISED, since `Tr I = 2^L`).

⚠ This is an ENTANGLED pair state, not a product -- which is why the identity superket has bond
dimension 2 on the intra-pair bonds and 1 between pairs.
"""
function identity_superket(L::Int)
    up = product_state([iseven(k) ? :up : :up for k in 1:(2L)])
    dn = product_state([iseven(k) ? :down : :down for k in 1:(2L)])
    # sum over the 2^L diagonal configurations = product over pairs of (|uu> + |dd>).
    # Built by direct summation over pair patterns, which is exact and cheap at these L.
    acc = nothing
    for mask in 0:(2^L - 1)
        spins = Symbol[]
        for j in 1:L
            s = ((mask >> (j - 1)) & 1) == 1 ? :down : :up
            push!(spins, s); push!(spins, s)
        end
        p = product_state(spins)
        acc = acc === nothing ? p : add_mps(acc, p)
    end
    return acc
end

"`Tr(O rho)` for a single-site `S^z` on system site `j`: an overlap with the identity superket."
sz_expect(rho, Iket, L, j) = begin
    # S^z on the KET site 2j-1, applied to the identity superket, then overlapped with rho.
    o = copy(Iket)
    canonical!(o, ket(j))
    q = local_space()
    o[ket(j)] = to_concrete(contract(q.Sz, (2,), o[ket(j)], (2,)))
    real(overlap(o, rho))
end

if MODE == "smoke"
    # ── L small enough that the EXACT vectorised Lindbladian is a dense 4^L matrix ──────────
    L = LSYS
    say(@sprintf("SMOKE: dephasing Lindbladian, L = %d system sites -> %d chain sites", L, 2L))
    say(@sprintf("exact reference: dense 4^%d = %d superket dimension", L, 4^L))
    Jm = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    say("")
    say("GATE 1: gamma = 0 must reproduce the CLOSED-system dynamics exactly (the doubled chain")
    say("        factorises there, so any disagreement is a construction error, not physics).")
    say("GATE 2: Tr rho conserved to machine precision at every gamma -- that is the trace term")
    say("        doing its job; without it the jump term alone would inflate the trace.")
    say("")
    for g in GAMMAS
        v = pair_vertices(xxz_chain(2L; J = 1.0, delta = 1.0))
        Js = lindblad_couplings(L, Jm, g, 0.0, v)      # delta = 0 -> XX Hamiltonian
        W = pair_mpo(2L, Js, v)
        say(@sprintf("  gamma = %.2f : MPO virtual dim %d", g, maximum(mpo_virtual_dims(W))))
    end
    say("")
    say("⚠ NOT YET RUN AGAINST THE DENSE REFERENCE -- this pass only builds the operator and")
    say("  reports its width. The dense comparison is the next step and is what actually")
    say("  validates the construction.")
else
    say("transport mode not yet wired -- smoke first")
end
close(IO_)
