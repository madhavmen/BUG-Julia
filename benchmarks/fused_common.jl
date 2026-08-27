# FUSED d = 4 DOUBLED CHAIN -- shared machinery for every open-system driver.
#
# One chain site per SYSTEM site, carrying the pair `(ket, bra)` with basis index `2*k + b + 1`
# and `0 = up`, `1 = down`:  1 = uu, 2 = ud, 3 = du, 4 = dd.
#
# ⛔ THE LAYOUT IS THE WHOLE DESIGN, AND THE OTHER TWO WERE MEASURED AND REJECTED.
#
#   INTERLEAVED (ket 2j-1, ancilla 2j) puts every system-system term at RANGE 2, so no two-site
#   block contains both of a term's operators. The block sees such a term only through an MPO
#   environment channel, which for a half-open term is a ONE-SITE EXPECTATION of a charge-changing
#   operator -- and `<S^-> = 0` on a product state. MEASURED: the superket NEVER MOVES,
#   `max|rho_mps(T) - rho(0)| = 0.0000e+00` exactly, error flat in dt (order 0.00 over a 16x
#   range) and identical for `tdvp2` and `cbe_bug`. ⚠ The MPO is NOT at fault -- with ENTANGLED
#   probes it reproduces the dense superoperator to 1.3e-15, and the failing response is `0`
#   EXACTLY rather than a wrong nonzero value.
#
#   BLOCKED (ket 1..L, ancilla L+1..2L) fixes the dynamics -- 7.5e-14 (tdvp2) / 2.6e-15
#   (cbe_bug) at gamma = 0, order 2.99 at gamma = 0.5 -- but `|I>>` is MAXIMALLY ENTANGLED across
#   the ket/ancilla cut (chi = 2^L) and a SPATIAL cut needs TWO MPS cuts, so operator entanglement
#   is not readable from one bond.
#
#   FUSED gets all three: `H` is RANGE 1 (inside a two-site update, so the dynamics starts), the
#   jump term is ON-SITE, `|I>>` is a PRODUCT state, and a spatial cut IS one MPS bond.
#
# ⚠ NORMALISATION IS WRITTEN OUT, NOT INHERITED. Telum's `q.Sp` carries a 1/sqrt(2) (`gates.jl`
# says so), which silently rescaled a first attempt at `|I>>`: `Tr rho0` came back -0.5, not +1.
# Every matrix here is an explicit `kron`.

using LinearAlgebra, SparseArrays
using LurCGT, Telum
using Telum: _getLocalSpace_no_symmetry
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const _I2  = Matrix{Float64}(LinearAlgebra.I, 2, 2)
const _SP  = Float64[0 1; 0 0]
const _SM  = Float64[0 0; 1 0]
const _SZ  = Float64[0.5 0; 0 -0.5]

"""
The d = 4 local space for one fused `(ket, bra)` pair, under `:none`.

⚠ `kron(A, B)` puts A on the KET (the more significant index), matching `2*k + b + 1`.
`CUP` maps `|uu>` to `(|uu> + |dd>)/sqrt2` and is written SYMMETRIC on purpose, so it acts the
same whether the contraction convention applies it or its transpose.
"""
function fused_space()
    c = 1 / sqrt(2)
    CUP = zeros(Float64, 4, 4)
    CUP[1, 1] = c; CUP[4, 1] = c; CUP[1, 4] = c; CUP[4, 4] = c
    ops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}(
        :I   => sparse(kron(_I2, _I2)),
        :Kp  => sparse(kron(_SP, _I2)), :Km => sparse(kron(_SM, _I2)),
        :Kz  => sparse(kron(_SZ, _I2)),
        :Bp  => sparse(kron(_I2, _SP)), :Bm => sparse(kron(_I2, _SM)),
        :Bz  => sparse(kron(_I2, _SZ)),
        :JMP => sparse(kron(_SZ, _SZ)),
        :CUP => sparse(CUP))
    return _getLocalSpace_no_symmetry(ops, ("s", "s", "op"))
end
const Q4 = fused_space()

"A product state over fused sites; `idxs[j]` in 1..4. Mirrors `_product_state_none`."
function fused_product_state(idxs::Vector{Int})
    L = length(idxs)
    tensors = Any[]
    prev = (getvac(Q4.I, ("L,0", "L,1")), 2)
    for i in 1:L
        (1 <= idxs[i] <= 4) || throw(ArgumentError("fused index must be 1..4, got $(idxs[i])"))
        F = to_concrete(getIdentity(prev, (Q4.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        A = to_concrete(getsub(F, 3, s -> idxs[i]))
        A = to_concrete(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end
    return SymMPS(tensors, L)
end

"""
Apply a one-site operator to site `k`, tagged from the tensor itself.

⛔ CONTRACT THE OPERATOR LEG 1. Operators carry `(ket '+', bra '-')` and an MPS site leg is `'-'`,
so pairing leg 2 with the site leg is two `'-'` legs and Telum refuses it outright. `apply_gate` --
the path the whole Trotter machinery is validated on -- contracts the gate's KET legs against
theta's site legs, so the one-site form is leg 1.
"""
function site_apply!(p, k::Int, M)
    t = p[k].inds[2].itags
    op = to_concrete(setitag(setitag(M, 1, t), 2, t))
    p[k] = to_concrete(permutedims(contract(op, (1,), p[k], (2,)), (2, 1, 3)))
    return p
end

"`rho0 = |psi0><psi0|` for a PRODUCT `psi0`: site j in `|s,s>` -- index 1 for up, 4 for down."
rho0_product(spins::Vector{Symbol}) =
    fused_product_state([s === :up ? 1 : 4 for s in spins])

"`|I>>` NORMALISED (the true identity superket is `2^(L/2)` times it). A PRODUCT state, chi = 1."
function identity_superket(L::Int)
    p = rho0_product([:up for _ in 1:L])
    for j in 1:L
        site_apply!(p, j, Q4.CUP)
    end
    return p
end

"`Tr(rho)`, and with `at = j` the `Tr(S^z_j rho)`. `S^z` is Hermitian so it may sit on `<<I|`."
function trace_rho(Imps, rho, L::Int; at::Int = 0)
    o = copy(Imps)
    at > 0 && site_apply!(o, at, Q4.Kz)
    return 2.0^(L / 2) * overlap(o, rho)
end

"""
    fused_lindblad(L, Jm, gammas; delta = 0.0) -> (mpo, shift)

`Jm` is the SYSTEM coupling matrix (strict upper triangle), `gammas[j]` the dephasing rate on
system site `j`, `delta` the `S^zS^z` anisotropy of the system Hamiltonian.

⚠ `Jm` MUST BE REAL: the ancilla half carries `+i*H^T` and is written as `+i*Jm`, which is right
only where `H^T = H`.

⛔ THE ZERO-BODY TERM COMES BACK SEPARATELY. `L†L = 1/4` for `L = S^z`, so the anticommutator is
`-(sum_j gamma_j / 4)` times the IDENTITY, and `pair_mpo` holds two-body vertices only. It is
returned as `shift` and applied as an EXACT scalar `exp(tau*shift)` on the state, since
`exp(tau*(W + c*I)) = e^(tau*c)*exp(tau*W)`. Omitting it does not merely rescale -- `Tr rho`
inflates (MEASURED 1.1618 at gamma = 0.5).

⚠ THE ON-SITE JUMP TERM IS DISTRIBUTED OVER BONDS, `O_j (x) I_{j+1}` for `j < L` plus
`I_{L-1} (x) O_L`, because `pair_mpo` is two-body only -- the same trick `ising_bond_gates` uses
for its field. Per-vertex coupling matrices are what let the two supports coexist.
"""
function fused_lindblad(L::Int, Jm::AbstractMatrix, gammas::Vector{Float64};
                        delta::Float64 = 0.0, hz::Union{Nothing, Vector{Float64}} = nothing,
                        Jzz::Union{Nothing, AbstractMatrix} = nothing)
    all(isreal, Jm) || throw(ArgumentError(
        "fused_lindblad needs a REAL coupling matrix: the ancilla half is written +i*Jm, " *
        "which equals +i*H^T only when H^T = H"))
    length(gammas) == L || throw(ArgumentError("need one gamma per system site"))
    L >= 2 || throw(ArgumentError("fused_lindblad needs at least two sites"))
    hz === nothing || length(hz) == L || throw(ArgumentError("need one hz per system site"))
    Jzz === nothing || all(isreal, Jzz) || throw(ArgumentError("Jzz must be REAL, same reason"))
    PV = RSVDCBEBondUpdate.PairVertex
    Jk = zeros(ComplexF64, L, L); Jb = zeros(ComplexF64, L, L)
    for i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        Jk[i, j] = -im * Jm[i, j]
        Jb[i, j] = +im * Jm[i, j]
    end
    # ⛔ THE S^zS^z SUPPORT NEED NOT MATCH THE XY SUPPORT. `delta * Jm` is right for XXZ, where one
    # matrix scales all three vertices, and WRONG for the Schwinger chain, whose hopping is
    # NEAREST-NEIGHBOUR while its Coulomb term is ALL-TO-ALL. Passing `Jzz` uses it directly and
    # leaves `delta` out of it. Per-vertex coupling matrices cost no extra bond dimension: the
    # channel count is set by how many openings are still closable.
    Kzz = Jzz === nothing ? nothing : (-im) .* ComplexF64.(Jzz)
    Bzz = Jzz === nothing ? nothing : (+im) .* ComplexF64.(Jzz)
    # ⚠ ONE-BODY TERMS GO THROUGH A TWO-BODY BUILDER, `O_j (x) I_{j+1}` on bonds 1..L-1 plus
    # `I_{L-1} (x) O_L` on the last -- the trick `ising_bond_gates` uses for its field. Two
    # vertices per one-body operator, and the LAST SITE NEEDS ITS OWN because bond `j` only ever
    # reaches site `j+1`: dropping it silently evolves a chain whose final site has no field,
    # which is a different Hamiltonian whose error SHRINKS with L, so small tests miss it.
    Jj1 = zeros(ComplexF64, L, L); Jj2 = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        Jj1[j, j + 1] = gammas[j]
    end
    Jj2[L - 1, L] = gammas[L]
    zcoeff = Jzz === nothing ? delta : 1.0        # `Jzz` carries its own scale
    verts = PV[PV(Q4.Kp, Q4.Km, 0.5), PV(Q4.Km, Q4.Kp, 0.5), PV(Q4.Kz, Q4.Kz, zcoeff),
               PV(Q4.Bp, Q4.Bm, 0.5), PV(Q4.Bm, Q4.Bp, 0.5), PV(Q4.Bz, Q4.Bz, zcoeff),
               PV(Q4.JMP, Q4.I, 1.0), PV(Q4.I, Q4.JMP, 1.0)]
    Js = Any[Jk, Jk, Kzz === nothing ? Jk : Kzz,
             Jb, Jb, Bzz === nothing ? Jb : Bzz, Jj1, Jj2]
    if hz !== nothing
        Hk1 = zeros(ComplexF64, L, L); Hk2 = zeros(ComplexF64, L, L)
        Hb1 = zeros(ComplexF64, L, L); Hb2 = zeros(ComplexF64, L, L)
        for j in 1:(L - 1)
            Hk1[j, j + 1] = -im * hz[j]
            Hb1[j, j + 1] = +im * hz[j]
        end
        Hk2[L - 1, L] = -im * hz[L]
        Hb2[L - 1, L] = +im * hz[L]
        append!(verts, PV[PV(Q4.Kz, Q4.I, 1.0), PV(Q4.I, Q4.Kz, 1.0),
                          PV(Q4.Bz, Q4.I, 1.0), PV(Q4.I, Q4.Bz, 1.0)])
        append!(Js, Any[Hk1, Hk2, Hb1, Hb2])
    end
    return pair_mpo(L, Vector{AbstractMatrix}(Js), verts; Iloc = Q4.I), -sum(gammas) / 4
end

"The omitted zero-body term, as an exact scalar on the state."
apply_shift!(p, z::Number) = (p[1] = to_concrete(ComplexF64(z) * p[1]); p)

"""
    normalise_trace!(Imps, rho, L) -> tr_before

Rescale `|rho>>` so `Tr rho = 1`, returning the trace it had BEFORE rescaling.

⛔ WHY THIS IS NEEDED AT ALL, AND WHAT IT IS NOT. `Tr rho` is the conservation law of a Lindbladian
-- `<<I|L = 0` exactly, since every term is a commutator or a jump-plus-anticommutator and `<<I|`
annihilates each. `tdvp2` inherits that exactly: its sweep is a product of local exponentials and
each local generator separately annihilates `<<I|`. A GALERKIN scheme does not, because it evolves
`exp(tau * P L P)` and

    Tr(P rho) = <<I|P|rho>> = <<P I|rho>>

so the trace survives the projection IF AND ONLY IF `P|I>> = |I>>` -- the variational basis must
CONTAIN the identity superket, and nothing in CBE/BUG puts it there. Truncation compounds it.

⚠ RESCALING IS LEGITIMATE BUT IT IS NOT A CURE. Every observable is `Tr(O rho)/Tr(rho)`, so the
physical state is `rho/Tr(rho)` and dividing is exactly what one does. It also stops the defect
COMPOUNDING: an un-normalised state whose trace decays each step shrinks multiplicatively, and the
error rides that decay. What it cannot do is repair a state whose DIRECTION is wrong -- so the
pre-normalisation trace is RETURNED, and the drivers report it. Quoting `Tr = 1` after this as
evidence of accuracy would be circular.

⚠ THE STRUCTURAL FIX, if this ever needs to be exact rather than corrected: put `|I>>` in the
basis. In the FUSED layout it is a PRODUCT state (chi = 1, measured bond dims `[1]`), so
augmenting every frame with its local vector costs ONE extra direction per bond -- and because it
is a product, each local frame containing the local vector is enough for the tensor product to
contain `|I>>`. The closing truncation would have to preserve that direction too.
"""
function normalise_trace!(Imps, rho, L::Int)
    tr = real(trace_rho(Imps, rho, L))
    abs(tr) > 1e-12 && apply_shift!(rho, 1.0 / tr)
    return tr
end

"""
    add_mps(a, b) -> SymMPS

`|a> + |b>` by DIRECT SUM on the bonds: block-diagonal in the interior, a row at the left boundary
and a column at the right. Bond dimensions add; the physical legs are shared.

⚠ `b`'s BOND TAGS ARE RETAGGED TO `a`'s BEFORE EACH `oplus`. The two states come from different
constructors (`rho` carries the sweep's `cbeW`/`cbeZ` tags, `|I>>` carries `L,i`), and Telum
refuses a direct sum across mismatched itags. Retagging the bonds is safe -- they are internal
labels, summed over -- while retagging a PHYSICAL leg would silently pair the wrong sites, which is
why only legs 1 and 3 are touched.
"""
function add_mps(a, b)
    n = length(a)
    n == length(b) || throw(DimensionMismatch("add_mps: $(n) vs $(length(b)) sites"))
    out = Any[]
    for i in 1:n
        A = a[i]
        B = to_concrete(setitag(setitag(b[i], 1, A.inds[1].itags), 3, A.inds[3].itags))
        push!(out, i == 1 ? to_concrete(oplus([A, B], (3,))) :
                   i == n ? to_concrete(oplus([A, B], (1,))) :
                            to_concrete(oplus([A, B], (1, 3))))
    end
    return SymMPS(out, n)
end

"""
    restore_trace!(Imps, rho, L; maxdim, cutoff) -> tr_before

Restore `Tr rho = 1` by adding the DEFICIT BACK ALONG THE IDENTITY DIRECTION:

    rho <- rho + (eps / 2^(L/2)) * Imps,    eps = 1 - Tr(rho)

⛔ WHY THIS AND NOT `rho / Tr(rho)`. Rescaling multiplies the WHOLE state, traceless part included,
and the BUG error is not a pure trace deficit -- so it OVERCORRECTS. MEASURED on the fused L=3
chain at full rank, rescaling pins `Tr = 1` and makes the state error CONSISTENTLY 1.91x WORSE
(2.42e-03 -> 4.66e-03, 1.53e-04 -> 2.92e-04, 9.56e-06 -> 1.83e-05). This correction instead leaves
the traceless part EXACTLY alone and moves only along `|I>>`, which is the minimal change that
fixes the conservation law.

⚠ `Imps` is the NORMALISED identity superket, so `Tr(Imps) = 2^(L/2)` -- hence the factor. Getting
that wrong overshoots by `2^(L/2)`, which at L = 18 is a factor of 512 and would be obvious; at
L = 2 it is 2 and would not.

⚠ THE ADDITION COSTS ONE BOND DIMENSION (`|I>>` is a PRODUCT state, chi = 1), so the state is
re-truncated afterwards. That truncation can itself shed trace -- which is exactly the shared,
cap-driven mechanism `tdvp2` suffers too -- so the CALLER still logs the trace it measured.
"""
function restore_trace!(Imps, rho, L::Int; maxdim::Int = 0, cutoff::Float64 = 1e-12)
    tr = real(trace_rho(Imps, rho, L))
    eps = 1.0 - tr
    abs(eps) < 1e-15 && return tr
    corr = copy(Imps)
    apply_shift!(corr, eps / 2.0^(L / 2))
    out = add_mps(rho, corr)
    for i in eachindex(rho)
        rho[i] = out[i]
    end
    # ⚠ THE DIRECT SUM DESTROYS THE CANONICAL FORM. `SymMPS.center` is an `Int`, so it cannot be
    # marked invalid -- it is set to a site and then a full `canonical!` re-gauges from scratch.
    # Leaving a stale centre here is the failure this repo has already paid for once: downstream
    # code that assumes a true canonical snapshot reads isometries that are no longer isometric.
    rho.center = 1
    canonical!(rho, 1)
    maxdim > 0 && truncate_recursive!(rho, 1; maxdim = maxdim, cutoff = cutoff)
    return tr
end

"""
OPERATOR ENTANGLEMENT at the central cut: the von Neumann entropy of `|rho>>` treated as a state.

⚠ `|rho>>` IS NOT NORMALISED (`Tr rho = 1` is the conservation law, not the 2-norm), so the
Schmidt values are renormalised here before the entropy is taken. Returns `(S, n_states)`.
"""
function operator_entanglement(psi)
    p = copy(psi); c = max(1, length(p) ÷ 2)
    canonical!(p, c)
    res = svd(p[c], (1, 2); cutoff = 0.0, get_lists = true)
    s = Float64[]
    for (v, deg, _, _) in res.kept_list, _ in 1:deg
        push!(s, Float64(v))
    end
    isempty(s) && return 0.0, 0
    s ./= sqrt(sum(abs2, s))
    p2 = abs2.(s)
    return -sum(x -> x > 0 ? x * log(x) : 0.0, p2), length(s)
end
