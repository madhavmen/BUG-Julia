# tests/common/spin_ed.jl -- sector-resolved exact dynamics of the open spin-s XX chain,
#
#     H_s = (J/2) sum_{b<L} (S^+_b S^-_{b+1} + S^-_b S^+_{b+1}),
#
# in one total-`S^z` sector, written from the spin algebra on paper:
#
#     S^+ |m> = sqrt(s(s+1) - m(m+1)) |m+1>.
#
# Shares nothing with the package (no Telum operator, no MPO, no MPS basis order), so agreement
# with `spin_xx_mpo` is evidence about its normalisation and not a restatement of it (plan T7.2b).
# Observables are compared, never amplitudes, so the basis order cannot matter.
#
# The propagator is a Taylor series on steps with `||H|| dt <= 1/2`, summed to roundoff: exact
# for the sector sizes used here (L <= 6 at s = 3, ~1e4 states).

using LinearAlgebra, SparseArrays

"""
    spin_xx_sector(L, twoS; J = 1.0, twoM = 0) -> (H, basis)

`H_s` in the sector `2 S^z_tot = twoM`; `basis[k]` is the vector of `2 m_i` of state `k`.
"""
function spin_xx_sector(L::Int, twoS::Int; J::Real = 1.0, twoM::Int = 0)
    vals = twoS:-2:-twoS
    basis = [collect(c) for c in Iterators.product(ntuple(_ -> vals, L)...) if sum(c) == twoM]
    index = Dict(b => k for (k, b) in enumerate(basis))
    s = twoS / 2
    raise(m) = sqrt(s * (s + 1) - m * (m + 1))           # <m+1|S^+|m>, m in units of 1
    I, K, V = Int[], Int[], Float64[]
    for (k, b) in enumerate(basis), i in 1:(L - 1), (u, d) in ((i, i + 1), (i + 1, i))
        # S^+_u S^-_d: raise site u, lower site d
        (b[u] < twoS && b[d] > -twoS) || continue
        nb = copy(b)
        nb[u] += 2
        nb[d] -= 2
        push!(I, index[nb]); push!(K, k)
        push!(V, (J / 2) * raise(b[u] / 2) * raise(nb[d] / 2))
    end
    return sparse(I, K, V, length(basis), length(basis)), basis
end

"The sector vector of the product state with `2 m_i = twoms[i]`."
spin_sector_state(basis, twoms) =
    (v = zeros(ComplexF64, length(basis)); v[findfirst(==(collect(twoms)), basis)] = 1; v)

"`exp(-i H t) v` by Taylor steps with `||H||_1 dt <= 1/2`, each summed until the term is below eps."
function spin_ed_propagate(H, v::AbstractVector, t::Real)
    nh = max(opnorm(H, 1), eps())
    n = max(1, ceil(Int, 2 * nh * abs(t)))
    dt = t / n
    for _ in 1:n
        term, acc = copy(v), copy(v)
        for k in 1:200
            term = (-1im * dt / k) * (H * term)
            acc .+= term
            norm(term) < 1e-17 * norm(acc) && break
        end
        v = acc
    end
    return v
end

"`<S^z_i>` (units of 1) of a sector vector."
spin_ed_sz(v, basis) = [sum(abs2(v[k]) * basis[k][i] / 2 for k in eachindex(basis))
                        for i in eachindex(basis[1])]

"The bond current `j_b = (iJ/2) <S^+_b S^-_{b+1} - S^-_b S^+_{b+1}>` (the `spin_current_mpo` sign)."
function spin_ed_current(v, basis, twoS::Int, b::Int; J::Real = 1.0)
    index = Dict(x => k for (k, x) in enumerate(basis))
    s = twoS / 2
    raise(m) = sqrt(s * (s + 1) - m * (m + 1))
    acc = 0.0im
    for (k, x) in enumerate(basis), (u, d, sg) in ((b, b + 1, 1), (b + 1, b, -1))
        (x[u] < twoS && x[d] > -twoS) || continue
        nx = copy(x)
        nx[u] += 2
        nx[d] -= 2
        acc += sg * conj(v[index[nx]]) * raise(x[u] / 2) * raise(nx[d] / 2) * v[k]
    end
    return real(1im * J / 2 * acc)
end
