# RANDOM MPO AT A TARGET VIRTUAL BOND DIMENSION — for cost benchmarking only.
#
# WHY A SYNTHETIC ONE. Cost in `H·Θ` is linear in the MPO bond dimension D, so D is an axis
# in its own right — and the one where the randomised sketch has most to gain, since the
# object being probed grows with D while the number of directions we actually want does not.
# But no physical builder in this repo reaches D = 1024 at L = 30:
#
#   * `xxz_mpo`                D = 5, fixed.
#   * `long_range_zz_mpo`      one self-loop channel; D = 6.
#   * `power_law_zz_mpo`       D = 2 + |nn| + n_exp, so D = 1024 needs n_exp ≈ 1018 —
#                              a least-squares fit of 1018 exponentials to L-1 = 29 data
#                              points. That is not a large MPO, it is a degenerate fit.
#   * `pair_mpo`               exact for arbitrary J[i,j], but D = O(L) ≈ 30.
#
# So D is grown the same way `random_mps.jl` grows chi: DIRECT SUM with a randomised copy,
# which doubles the virtual bond while reproducing the sparsity pattern exactly. The sector
# structure is INHERITED from a real MPO, never invented — a wrong per-sector split gives the
# right D with the wrong block shapes, and block shapes are exactly what a per-sector cost
# measurement measures.
#
# ⛔ THE RESULT IS NOT HERMITIAN AND NOT A HAMILTONIAN. Gaussian channel payloads have no
# adjoint symmetry, so:
#   * nothing built here may be scored for accuracy, energy, or conservation;
#   * runs keep `hermitian = true` anyway. Lanczos on a non-Hermitian generator does not fail,
#     it LIES — but it lies while doing bit-for-bit the same arithmetic per iteration, which
#     is the only thing being measured. Switching to Arnoldi to be "correct" would silently
#     change the code path and measure a different algorithm.
#   * `mpo_energy` and every observable are meaningless. Do not call them.

using Random
using Printf
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: MPO, mpo_virtual_dims
using Telum: TLArray, oplus, symm, to_concrete

"Gaussian payloads on `W`'s exact sparsity pattern, leaving qlabels/spaces untouched."
function randomize_mpo_tensor(W, rng::AbstractRNG, scale::Float64)
    T = eltype(W) <: Complex ? eltype(W) : ComplexF64
    return TLArray(symm(W), copy(W.qlabels), copy(W.wmatdata), copy(W.wmatinfo),
                   [scale .* randn(rng, T, size(r)) for r in W.RMTs], W.inds, W.spaces)
end

"""
    double_D!(mpo, rng; scale) -> mpo

`W_i <- W_i (+) W_i_random` on the VIRTUAL legs only, doubling every internal MPO bond.

Site tensors are rank-4 `(w_l, s_ket, s_bra, w_r)`; legs 2 and 3 are physical and must stay
`d`, so only 1 and 4 enter the `oplus`. The boundary tensors have a trivial leg at one end
(`W[1]`'s `w_l` and `W[L]`'s `w_r` are the dim-1 vacuum that forbids terms running off the
chain), and growing it would turn the MPO into a non-scalar operator — so each boundary
grows on one side only.

`scale` shrinks the random channels. Left at 1.0 the added channels dominate the physical
ones after a few doublings and `exp(tau H)` overflows; the point is to add COST, not norm.
"""
function double_D!(mpo::MPO, rng::AbstractRNG; scale::Float64 = 0.1)
    L = length(mpo)
    for i in 1:L
        W = mpo[i]
        R = randomize_mpo_tensor(W, rng, scale)
        legs = i == 1 ? (4,) : (i == L ? (1,) : (1, 4))
        mpo.W[i] = to_concrete(oplus([W, R], legs))
    end
    return mpo
end

"""
    random_mpo(L, target_D; seed = 1, delta = 1.0, scale = 0.1) -> mpo

An MPO on `L` sites whose virtual bond dimension is at least `target_D`, seeded from
`xxz_mpo` so that the charge structure of every channel is one the code itself produced.

⚠ Doubling OVERSHOOTS — `xxz_mpo` starts at D = 5, so the ladder is 5, 10, 20, ... and there
is no trimming step: unlike an MPS bond there is no canonical form to truncate an MPO bond
against without changing the operator. Take the D that comes back (reported by
`mpo_virtual_dims`) as the x-axis value rather than the D that was asked for.
"""
function random_mpo(L::Int, target_D::Int; seed::Int = 1, delta::Float64 = 1.0,
                    scale::Float64 = 0.1, verbose::Bool = true)
    rng = MersenneTwister(seed)
    t0 = time()
    log(msg) = verbose && (@printf("[random_mpo %7.1fs] %s\n", time() - t0, msg); flush(stdout))

    mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = delta)
    D() = maximum(mpo_virtual_dims(mpo))
    log(@sprintf("seed xxz_mpo D=%d", D()))
    while D() < target_D
        double_D!(mpo, rng; scale = scale)
        log(@sprintf("doubled -> D=%d (target %d)", D(), target_D))
    end
    log(@sprintf("done D=%d", D()))
    return mpo
end
