# ── the U(1) XX chain, H = J sum (S^x S^x + S^y S^y), from a domain wall ──
# Spins are `twoS = 2s` (s = 1/2 or 3, d = 2s + 1). Telum's Clebsch-Gordan convention gives the same J
# at every s, pinned by test_spin_xx.jl.
# Only s = 1/2 has a free-fermion closed form; the s = 3 references are sector ED and converged MPS runs.

const _SPIN_SPACES = Dict{Int, Any}()

"""
    spin_space(twoS) -> NamedTuple

The U(1) spin-`twoS/2` local space `(; I, Sp, Sz, Sm)`, charge `2 S^z` (`twoS:-2:-twoS`, each a
one-dimensional sector). `twoS = 1` IS `local_space(:U1)`, so the s = 1/2 chain is the package's
own. Cached, like `local_space`: build it before any sweep runs.
"""
function spin_space(twoS::Int)
    twoS >= 1 || throw(ArgumentError("twoS = 2s must be >= 1, got $twoS"))
    twoS == 1 && return local_space(:U1)
    return get!(() -> getLocalSpace(SpinOptions(:U1, twoS), ("s", "s", "op")), _SPIN_SPACES, twoS)
end

"The nearest-neighbour coupling matrix `w` on the bonds `bonds` (strict upper triangle)."
function _nn_couplings(L::Int, w::Number, bonds = 1:(L - 1))
    M = zeros(typeof(w), L, L)
    for b in bonds
        M[b, b + 1] = w
    end
    return M
end

"""
`pair(Sp, Sp')` and `pair(Sm, Sm')` (U(1) op-legs): stored as `S^+_i S^-_{i+1} / 2` and
`S^-_i S^+_{i+1} / 2`, APPLIED by `pair_mpo` as their transposes (the `liouville.jl` header), i.e.
the other one. The sum `H_s` is symmetric and does not see it; `spin_current_mpo` does.
"""
_spin_xx_vertices(q) = [PairVertex(q.Sp, to_concrete(q.Sp'), 1.0),
                        PairVertex(q.Sm, to_concrete(q.Sm'), 1.0)]

"""
    spin_xx_mpo(L, twoS; J = 1.0) -> MPO

`H_s = J sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1})` on an open chain of `L` spin-`twoS/2` sites,
under U(1). At `twoS = 1` it is the same operator as `xxz_mpo(L; J, delta = 0)`.
"""
function spin_xx_mpo(L::Int, twoS::Int; J::Float64 = 1.0)
    q = spin_space(twoS)
    return pair_mpo(L, _nn_couplings(L, J), _spin_xx_vertices(q); Iloc = q.I)
end

"""
    spin_current_mpo(L, twoS, b; J = 1.0) -> MPO

The spin current across bond `b`, `j_b = J (S^x_b S^y_{b+1} - S^y_b S^x_{b+1})
= (iJ/2)(S^+_b S^-_{b+1} - S^-_b S^+_{b+1})`, signed so that `d<S^z_b>/dt = j_{b-1} - j_b`
under `spin_xx_mpo` (positive for spin flowing to larger `i`).
"""
function spin_current_mpo(L::Int, twoS::Int, b::Int; J::Float64 = 1.0)
    1 <= b <= L - 1 || throw(ArgumentError("bond must be in 1:$(L - 1), got $b"))
    q = spin_space(twoS)
    w = ComplexF64(0, J)
    # ⛔ `j_b` is imaginary ANTISYMMETRIC, so the transpose `pair_mpo` applies is `-j_b`: `+w` goes
    # on the vertex that APPLIES `S^+_b S^-_{b+1}`, which is `(Sm, Sm')`. Measured (hq24a, L = 4
    # against dense): `+w` on `(Sp, Sp')` gave `-j_b` exactly, and evolved as `exp(+i tau j_b)`.
    return pair_mpo(L, [_nn_couplings(L, -w, b:b), _nn_couplings(L, w, b:b)],
                    _spin_xx_vertices(q); Iloc = q.I)
end

"""
    spin_domain_wall_state(L, twoS; eltype = ComplexF64) -> SymMPS

`|+s, ..., +s, -s, ..., -s>`, the first `L ÷ 2` sites at `S^z = +s`. Total `S^z = 0` for even
`L`. At `twoS = 1` it is `domain_wall_state(L)`.
"""
spin_domain_wall_state(L::Int, twoS::Int; eltype::Type{<:Number} = ComplexF64) =
    sector_product_state(spin_space(twoS), [(((i <= L ÷ 2 ? twoS : -twoS),),) for i in 1:L];
                         eltype = eltype)

"`<S^z_i>` on every site of a spin-`twoS/2` chain."
spin_sz_profile(psi::SymMPS, twoS::Int) =
    [real(site_expval(psi, j, spin_space(twoS).Sz)) for j in 1:length(psi)]
