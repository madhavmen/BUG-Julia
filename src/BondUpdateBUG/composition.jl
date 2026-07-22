# Third- and fourth-order sweeps, by EXTRAPOLATION rather than composition.
#
# MEASURED OUTCOME (see test_composition.jl for the numbers). Under the strict
# Sulz <=2r bound these extrapolated orders do NOT raise the global order past 2:
# the rank-2r Galerkin bond update carries an O(dt^2) projection error whose
# coefficient depends on the augmented basis, which differs between the n=1 and
# n=2 refinement levels, so the linear combination cannot cancel it. They still
# lower the error CONSTANT (richardson4 ~23x, extrap3 ~5.7x vs Strang) and remain
# useful for that. Verified constructively: with the frames COMPLETED (2r bound
# dropped -> full 2-site TDVP) the same machinery reaches Yoshida 4th order, so
# the ceiling is the 2r augmentation, not this code. The forward-only reasoning
# below is retained because it is why composition was rejected in favour of
# extrapolation in the first place.
#
# WHY NOT COMPOSITION. BUG is inverse-free by design and a backward sub-step is
# unstable for parabolic generators (heat, GP), so every BUG order here must be
# FORWARD-ONLY: all sub-step times >= 0. The Sheng-Suzuki theorem says any
# composition splitting of order >= 3 contains at least one NEGATIVE time
# coefficient -- so there is no forward-only 3rd- or 4th-order composition at all.
# Higher order with positive times is reachable only by extrapolation, where the
# negative weight sits in the linear combination (Richardson's -1/3) while every
# time step stays positive.
#
# Corollary worth stating: any all-positive "third-order sweep coefficient" tuple
# is not third order. Derive the weights, never copy them.
#
# DERIVATION. Run the base method with `n` sub-steps of size dt/n and combine the
# results with weights w. Consistency needs sum(w) = 1; killing the error term of
# order k needs sum(w_i (1/n_i)^k) = 0. Solving those small systems:
#
#   :richardson4  base Strang (SYMMETRIC -> even powers only), n = 1,2, kill k=2
#                 -> w = (-1/3, 4/3)
#   :extrap3      base Lie (p=1),                n = 1,2,3, kill k=1,2
#                 -> w = (1/2, -4, 9/2)
#
# Third order is not natural from Strang: a time-symmetric method has only even
# powers, so two-level extrapolation jumps 2 -> 4. Third order comes from Lie.
# Strang's symmetry is ASSERTED BY TEST rather than assumed -- BUG's truncation
# and augmentation are not time-reversible, so odd powers could survive; if they
# did, `:richardson4` would degrade to 3rd and the 3-level (1/12, -4/3, 9/4)
# recipe would be the correct 4th-order one instead.
#
# Cost: `:extrap3` is 1+2+3 = 6 base sweeps per step, `:richardson4` is 1+2 = 3.
# Both need a TT addition and a recompression, which inflates the bond before it
# is cut back. That is the price of staying forward-only.

"Extrapolation table: base method, sub-step counts, and the derived weights."
const _EXTRAPOLATION = Dict{Symbol, NamedTuple}(
    :richardson4 => (base = :strang, levels = (1, 2),    weights = (-1 / 3, 4 / 3)),
    :extrap3     => (base = :lie,    levels = (1, 2, 3), weights = (1 / 2, -4.0, 9 / 2)),
)

"""
Yoshida's fourth-order composition weights, `S2(w1 dt) S2(w0 dt) S2(w1 dt)`.

`w0 < 0`: this steps BACKWARD. It is here only so a BUG-vs-TDVP comparison can be
made at equal order -- TDVP2's validated 4th-order path is this composition, and
comparing a 4th-order TDVP against a 2nd-order BUG would not be a fair test. It
must never be a BUG default, and its coefficients must never be mixed with the
extrapolation weights above: each follows from its own base method's order and
symmetry.
"""
function _yoshida_weights()
    w1 = 1 / (2 - 2^(1 / 3))
    return (w1, 1 - 2 * w1, w1)
end

"""
    extrapolation_weights(order) -> NTuple

The linear-combination weights for an extrapolated order. Satisfies `sum(w) == 1`
and `sum(w_i (1/n_i)^k) == 0` for each error order `k` being removed.
"""
function extrapolation_weights(order::Symbol)
    haskey(_EXTRAPOLATION, order) ||
        throw(ArgumentError("$order is not an extrapolated order; " *
                            "expected one of $(sort(collect(keys(_EXTRAPOLATION))))"))
    return _EXTRAPOLATION[order].weights
end

"""
    substep_times(order, dt) -> Vector{Float64}

Every physical time increment one step of `order` applies, in order.

The forward-only guard reads this: for `:lie`, `:strang`, `:extrap3` and
`:richardson4` every entry is `>= 0`. `:yoshida4` contains a negative entry by
construction, which is exactly why it is not a BUG order.
"""
function substep_times(order::Symbol, dt::Real)
    order === :lie    && return Float64[dt, dt]
    order === :strang && return Float64[dt / 2, dt, dt / 2]
    if haskey(_EXTRAPOLATION, order)
        spec = _EXTRAPOLATION[order]
        out = Float64[]
        for n in spec.levels, _ in 1:n
            append!(out, substep_times(spec.base, dt / n))
        end
        return out
    end
    if order === :yoshida4
        out = Float64[]
        for w in _yoshida_weights()
            append!(out, substep_times(:strang, w * dt))
        end
        return out
    end
    throw(ArgumentError("unknown order $order"))
end

"Orders this module knows about, including the ones `driver.jl` handles directly."
const SUPPORTED_ORDERS = (:lie, :strang, :extrap3, :richardson4, :yoshida4)

"""
    compress!(psi; maxdim, cutoff) -> SymMPS

Truncate `psi` back to `maxdim` with a right-to-left sweep, leaving the centre on
site 1. Needed after a linear combination: `oplus` adds the bond dimensions, so an
`n`-term combination inflates every bond `n`-fold before this cuts it back.
"""
function compress!(psi::SymMPS; maxdim::Int, cutoff::Float64)
    L = length(psi)
    canonical!(psi, L)                       # exact, so the truncation below is local
    for i in L:-1:2
        res = svd(psi[i], (1,); cutoff = cutoff, Nkeep = maxdim)
        M = res.U * res.S
        B = contract(psi[i - 1], (3,), M, (1,))
        psi[i - 1] = to_concrete(setitag(B, 3, "L,$i"))
        psi[i]     = to_concrete(setitag(res.Vd, 1, "L,$i"))
    end
    psi.center = 1
    return psi
end

"""
    linear_combination(states, coeffs; maxdim, cutoff) -> SymMPS

`sum_k coeffs[k] * states[k]` as an MPS, via direct sum then recompression.

The coefficient is folded into site 1 only -- scaling one tensor scales the whole
product state -- so the sum is a single `oplus` per site with no extra arithmetic.
"""
function linear_combination(states::Vector{SymMPS}, coeffs; maxdim::Int, cutoff::Float64)
    length(states) == length(coeffs) ||
        throw(ArgumentError("got $(length(states)) states and $(length(coeffs)) coefficients"))
    L = length(states[1])
    tensors = Any[]
    for i in 1:L
        blocks = Any[]
        for (k, st) in enumerate(states)
            push!(blocks, i == 1 ? to_concrete(ComplexF64(coeffs[k]) * st[i]) : st[i])
        end
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(tensors, to_concrete(oplus(blocks, dims)))
    end
    return compress!(SymMPS(tensors, L); maxdim = maxdim, cutoff = cutoff)
end
