# EXACT dense Liouvillian for the fused d = 4 open models -- the reference that turns
# `square_open` and `lgt` from "validated against a published trend" into "scored against a number".
#
# WHY THIS IS A LEGITIMATE REFERENCE AND NOT OUR-OWN-FINEST-GRID. The standing rule is that a model
# with no analytic solution needs a PUBLISHED anchor rather than a finer run of the same code
# (`external-reference-anchors`). This is neither: the Liouvillian is a `4^N x 4^N` MATRIX and
# `exp(L*T)*v0` is its exact matrix exponential, computed by a dense linear-algebra routine that
# shares no code path with the MPO, the sweeps, the truncation or the Krylov solver. It is an exact
# diagonalisation, in the same sense that `xx_free_fermion_sz` is exact -- the prohibition is on
# calibrating against a coarser version of the thing under test, which this is not.
#
# ⛔ IT MUST BE BUILT FROM `S.params`, NOT FROM A RE-DERIVED LATTICE. `setup` hands back the exact
# `(Jm, gammas, delta, hz, Jzz)` that went into `fused_lindblad`. A driver that rebuilds the square
# patch or the Schwinger terms itself is comparing two lattice constructions, and a mismatch there
# looks exactly like an integrator error. That is why `_open` carries `params` at all.
#
# ⛔ THE COST IS `4^N x 4^N`, SO N <= 6. At N = 6 the superoperator is 4096 x 4096 (268 MB complex)
# and `exp` takes seconds. N = 7 is 16384 x 16384 = 4.3 GB and will not fit; `dense_lindblad_super`
# throws rather than trying. This is a CONVENTION AND OPERATOR gate, not a scaling benchmark -- a
# wrong sign or a mirrored bit ordering is N-independent, which is exactly why a small N settles it.
#
# ⚠ THE BIT CONVENTION IS THE LANDMINE HERE. `idx_to_ab` decodes a fused index `d` in 0..3 as
# ket bit `d >> 1` and bra bit `d & 1`, and `perm` maps that ordering onto column-major `vec(rho)`.
# Dense and MPS orderings in this repo have been MIRRORED before, and Neel, domain-wall and dimer
# states all hide it because they are symmetric under the mirror. `dense_lindblad_gate` therefore
# validates the mapping on an ASYMMETRIC state before it validates anything else.

using LinearAlgebra

# ⛔ THIS FILE DECLARES ITS OWN DEPENDENCY. `fused_dense_state` needs `_full_tensor`, which lives in
# `dense_reference.jl`; `tensor_inner` comes from `BUGJulia.BondUpdateBUG`, and `fused_product_state`
# from `benchmarks/fused_common.jl`. Leaving the first one to the caller is how four committed
# drivers ended up broken from a fresh clone once already (fixed in 1084c00) -- a driver that
# happens to include the right file in the right ORDER works, and one that does not fails with an
# `UndefVarError` a long way from the cause.
#
# The guard makes a double include harmless: `trace_preservation.jl` includes `dense_reference.jl`
# directly, and a driver pulling in both would otherwise redefine every method in it.
if !isdefined(@__MODULE__, :_full_tensor)
    include(joinpath(@__DIR__, "dense_reference.jl"))
end

"""
    dense_lindblad_super(L, Jm, gammas; delta = 0.0, hz = nothing, Jzz = nothing)

The exact `4^L x 4^L` Lindbladian superoperator, in the SAME fused index order as
`fused_dense_state`.

Mirrors `fused_lindblad` (benchmarks/fused_common.jl) term for term:

    H = sum_{i<j} Jm[i,j] * (Sx_i Sx_j + Sy_i Sy_j)
      + zcoeff * sum_{i<j} (Jzz === nothing ? Jm : Jzz)[i,j] * Sz_i Sz_j
      + sum_j hz[j] * Sz_j
    L[rho] = -i[H, rho] + sum_j gammas[j] * (Z_j rho Z_j - 1/2 {Z_j Z_j, rho}),   Z_j = S^z_j

⛔ `zcoeff` FOLLOWS `fused_lindblad`: `delta` when `Jzz` is absent, and `1.0` when it is present,
because `Jzz` carries its own scale. Getting that backwards double-counts the Schwinger Coulomb
term, which is all-to-all and would dominate.

⚠ THE ZERO-BODY TERM IS INCLUDED HERE, unlike in the MPO. `L^dag L = 1/4` for `L = S^z`, so the
anticommutator contributes `-sum(gammas)/4` times the identity; `fused_lindblad` returns that
separately as `shift` because `pair_mpo` has nowhere to put a constant. The dense operator has
somewhere to put it, so it carries the FULL generator and needs no shift applied to it. A driver
comparing the two must therefore compare against the SHIFTED MPS state (after `S.shift!`).
"""
function dense_lindblad_super(L::Int, Jm::AbstractMatrix, gammas::Vector{Float64};
                              delta::Float64 = 0.0,
                              hz::Union{Nothing, Vector{Float64}} = nothing,
                              Jzz::Union{Nothing, AbstractMatrix} = nothing)
    L >= 2 || throw(ArgumentError("dense_lindblad_super needs at least two sites"))
    L <= 6 || throw(ArgumentError(
        "dense_lindblad_super is $(4^L) x $(4^L) at L = $L -- " *
        "N <= 6 only (N = 7 is 4.3 GB). This is a convention gate, not a scaling benchmark."))
    length(gammas) == L || throw(ArgumentError("need one gamma per site"))
    hz === nothing || length(hz) == L || throw(ArgumentError("need one hz per site"))

    D  = 2^L
    I2 = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
    SXm = ComplexF64[0 0.5; 0.5 0]
    SYm = ComplexF64[0 -0.5im; 0.5im 0]
    SZm = ComplexF64[0.5 0; 0 -0.5]
    sop(op, j) = (m = ComplexF64[1;;]; for k in 1:L; m = kron(m, k == j ? op : I2); end; m)

    H = zeros(ComplexF64, D, D)
    for i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        H .+= Jm[i, j] .* (sop(SXm, i) * sop(SXm, j) + sop(SYm, i) * sop(SYm, j))
    end
    # `Jzz` carries its own scale; `delta` only applies when it is absent -- see the docstring.
    zc = Jzz === nothing ? delta : 1.0
    Zm = Jzz === nothing ? Jm : Jzz
    if zc != 0.0
        for i in 1:L, j in (i + 1):L
            Zm[i, j] == 0.0 && continue
            H .+= (zc * Zm[i, j]) .* (sop(SZm, i) * sop(SZm, j))
        end
    end
    if hz !== nothing
        for j in 1:L
            hz[j] == 0.0 && continue
            H .+= hz[j] .* sop(SZm, j)
        end
    end

    Zs = [sop(SZm, j) for j in 1:L]
    S = zeros(ComplexF64, D * D, D * D)
    for a in 1:D, b in 1:D
        E = zeros(ComplexF64, D, D); E[a, b] = 1.0
        R = -im * (H * E - E * H)
        for j in 1:L
            g = gammas[j]
            g == 0.0 && continue
            Z = Zs[j]
            R .+= g .* (Z * E * Z .- 0.5 .* (Z * Z * E .+ E * Z * Z))
        end
        S[:, (b - 1) * D + a] = vec(R)
    end
    perm = [(ab = _fused_idx_to_ab(idx, L); ab[2] * D + ab[1] + 1) for idx in 0:(4^L - 1)]
    return S[perm, perm]
end

"""
    _fused_idx_to_ab(idx, n) -> (ket_index, bra_index)

Split a fused configuration index into its ket and bra halves.

⛔ THIS FUNCTION IS THE CONVENTION. Fused site value `d` in 0..3 is `(ket, bra) = (d >> 1, d & 1)`,
matching `rho0_product` (index 1 = |up,up>, index 4 = |down,down>). Mirroring it -- swapping the two
bits -- transposes every density matrix and is INVISIBLE on Neel, domain-wall and dimer states.
"""
function _fused_idx_to_ab(idx::Int, n::Int)
    a = 0; b = 0
    for j in 1:n
        d = (idx ÷ 4^(n - j)) % 4
        a |= (d >> 1) << (n - j)
        b |= (d & 1) << (n - j)
    end
    return a, b
end

"""
    fused_dense_state(psi) -> Vector{ComplexF64}

The fused MPS as a dense superket of length `4^n`, in the index order `dense_lindblad_super` uses.

⚠ COSTS `4^n` FULL-TENSOR OVERLAPS. Fine at n = 6 (4096), unusable beyond -- which is the same
bound the superoperator itself imposes.
"""
function fused_dense_state(psi)
    n = length(psi)
    P = _full_tensor(psi)
    v = zeros(ComplexF64, 4^n)
    for idx in 0:(4^n - 1)
        cfg = fused_product_state([(idx ÷ 4^(n - j)) % 4 + 1 for j in 1:n])
        v[idx + 1] = tensor_inner(_full_tensor(cfg), P)
    end
    return v
end

"""
    dense_sz_profile(v, L) -> Vector{Float64}

`Tr(S^z_j rho)` for every site, read off a dense superket -- the dense counterpart of
`S.profile`, so the two can be compared elementwise.

The superket is `vec(rho)` in the fused order, so undoing `perm` recovers `rho` as a `D x D`
matrix and the observable is an ordinary trace.
"""
function dense_sz_profile(v::Vector{ComplexF64}, L::Int)
    D = 2^L
    rho = zeros(ComplexF64, D, D)
    for idx in 0:(4^L - 1)
        a, b = _fused_idx_to_ab(idx, L)
        rho[a + 1, b + 1] = v[idx + 1]
    end
    out = zeros(Float64, L)
    for j in 1:L
        # <S^z_j> = sum over configurations of (+/- 1/2) * diagonal weight -- no kron needed.
        s = 0.0 + 0.0im
        for a in 0:(D - 1)
            bit = (a >> (L - j)) & 1
            s += (bit == 0 ? 0.5 : -0.5) * rho[a + 1, a + 1]
        end
        out[j] = real(s)
    end
    return out
end

"`Tr rho` from a dense superket -- a conservation law, so its drift is an error with no reference."
function dense_trace(v::Vector{ComplexF64}, L::Int)
    D = 2^L
    s = 0.0 + 0.0im
    for a in 0:(D - 1)
        idx = 0
        for j in 1:L
            bit = (a >> (L - j)) & 1
            idx += ((bit << 1) | bit) * 4^(L - j)
        end
        s += v[idx + 1]
    end
    return real(s)
end
