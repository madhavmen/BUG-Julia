# tests/common/spin_jw.jl -- the generalized Jordan-Wigner map of an open spin-s chain
# (Batista-Ortiz, PRL 86, 1082 (2001)) on chains small enough for the full Fock space
# (2^(2s L) states: L <= 3 at s = 3). Plan "(2)" T7.2 / U31: build the isometry, check that it
# carries the spin algebra and H_s, then ASK whether the mapped H_s is quadratic.
#
# Site j's 2s+1 states (local index n = 0..2s, m = -s + n) are the vacuum (n = `vac`) and one
# hard-core fermion of 2s flavours; the modes (j, f) are site-major with the ordinary string.
# A local operator |a><b| maps to
#     vac, vac -> prod_f (1 - n_{jf})                      even
#     a,   vac -> cbar^dag_{j f(a)} K_j                    odd,  K_j = (-1)^{N_{<j}}
#     vac, b   -> K_j cbar_{j f(b)}                        odd
#     a,   b   -> cbar^dag_{j f(a)} cbar_{j f(b)}          even
# with cbar_{jf} = c_{jf} prod_{g != f} (1 - n_{jg}), the hard-core constraint. Shares nothing
# with the package; spin operators are built from the algebra as in `spin_ed.jl`.

using LinearAlgebra, SparseArrays

"`S^+` and `S^z` of spin `twoS/2` on the local basis n = 0..2s (m = -s + n)."
function spin_local_ops(twoS::Int)
    s = twoS / 2
    m = [-s + n for n in 0:twoS]
    Sp = zeros(twoS + 1, twoS + 1)
    for n in 1:twoS
        Sp[n + 1, n] = sqrt(s * (s + 1) - m[n] * (m[n] + 1))
    end
    return Sp, Matrix(Diagonal(m))
end

"`O` acting on site `j` of `L` sites of dimension `d` (site 1 is the most significant digit)."
spin_site_op(O, j, L, d) = kron(sparse(1.0I, d^(j - 1), d^(j - 1)), sparse(O),
                                sparse(1.0I, d^(L - j), d^(L - j)))

"`H_s = (J/2) sum_b (S^+_b S^-_{b+1} + h.c.)` and the bond-`b` current operators, full space."
function spin_chain_ops(L::Int, twoS::Int; J::Real = 1.0)
    Sp, _ = spin_local_ops(twoS)
    d = twoS + 1
    hop(b) = spin_site_op(Sp, b, L, d) * spin_site_op(Sp', b + 1, L, d)
    H = (J / 2) * sum(hop(b) + hop(b)' for b in 1:(L - 1))
    cur(b) = (1im * J / 2) * (hop(b) - hop(b)')
    return H, cur
end

"""
    JWMap(L, twoS; vac = 0)

The map's Fock space: `nf = 2s` flavours per site, `M = L nf` modes, annihilators `c[k]`
(sparse, `2^M` square), mode `(j, f)` at `k = (j - 1) nf + f`.
"""
struct JWMap
    L::Int
    twoS::Int
    vac::Int
    nf::Int
    c::Vector{SparseMatrixCSC{Float64, Int}}
end

function JWMap(L::Int, twoS::Int; vac::Int = 0)
    0 <= vac <= twoS || throw(ArgumentError("vac must be a local index 0..$twoS"))
    nf = twoS
    M = L * nf
    n = 2^M
    c = map(1:M) do k
        I, J, V = Int[], Int[], Float64[]
        for s in 0:(n - 1)
            (s >> (k - 1)) & 1 == 1 || continue
            push!(I, (s ⊻ (1 << (k - 1))) + 1); push!(J, s + 1)
            push!(V, iseven(count_ones(s & ((1 << (k - 1)) - 1))) ? 1.0 : -1.0)
        end
        sparse(I, J, V, n, n)
    end
    return JWMap(L, twoS, vac, nf, c)
end

_mode(m::JWMap, j, f) = (j - 1) * m.nf + f
_flavour(m::JWMap, a) = a < m.vac ? a + 1 : a           # local index a != vac -> 1..2s
_fockdim(m::JWMap) = 2^(m.L * m.nf)

"Diagonal operator with entries `f(s)` over the Fock states `s = 0..2^M - 1`."
_fockdiag(m::JWMap, f) = spdiagm(0 => [Float64(f(s)) for s in 0:(_fockdim(m) - 1)])

"`(-1)^{N_{<j}}`, the string of site `j`."
jw_string(m::JWMap, j) =
    _fockdiag(m, s -> (-1)^count_ones(s & ((1 << ((j - 1) * m.nf)) - 1)))

"`prod_{g != f} (1 - n_{jg})` (`f = 0`: over every flavour, the site-vacuum projector)."
_holes(m::JWMap, j, f) = _fockdiag(m, s -> all(g == f || (s >> (_mode(m, j, g) - 1)) & 1 == 0
                                                for g in 1:m.nf))

"The fermionic image of the local operator `O` (basis n = 0..2s) at site `j`."
function jw_local(m::JWMap, O::AbstractMatrix, j::Int)
    K = jw_string(m, j)
    cb(a) = (k = _mode(m, j, _flavour(m, a)); m.c[k] * _holes(m, j, _flavour(m, a)))
    acc = spzeros(_fockdim(m), _fockdim(m))
    for a in 0:m.twoS, b in 0:m.twoS
        o = O[a + 1, b + 1]
        iszero(o) && continue
        acc += o * (a == m.vac && b == m.vac ? _holes(m, j, 0) :
                    b == m.vac ? sparse(cb(a)') * K :
                    a == m.vac ? K * cb(b) : sparse(cb(a)') * cb(b))
    end
    return acc
end

"""
    jw_isometry(m) -> T

`T[:, k]` is the Fock state `prod_j cbar^dag_{j f(n_j)} |0>` (site 1 leftmost; vacuum sites
contribute nothing) of the spin configuration `k`, in the `kron` order of `spin_site_op`.
"""
function jw_isometry(m::JWMap)
    d = m.twoS + 1
    cols = map(1:(d^m.L)) do k
        ns = reverse(digits(k - 1; base = d, pad = m.L))   # site 1 is the most significant digit
        v = sparsevec([1], [1.0], _fockdim(m))
        for j in m.L:-1:1
            ns[j] == m.vac && continue
            v = sparse(m.c[_mode(m, j, _flavour(m, ns[j]))]') * v
        end
        v
    end
    return reduce(hcat, cols)
end

"""
    jw_quadratic_residual(m, T, Hspin; V = I) -> (residual, ncoef)

The smallest `||Hq T V - T Hspin V||_F / ||T Hspin V||_F` over every quadratic `Hq = x0 +
sum x_ab c_a^dag c_b + sum y_ab c_a c_b + sum z_ab c_a^dag c_b^dag` (all coefficients free, so a
Hermitian or particle-conserving `Hq` is included): ~0 iff the mapped `H_s` acts as a
free-fermion operator on the physical states `T V` in this map (`V`: a spin subspace, e.g. the
Krylov space of the domain wall; default the whole space). Least squares on the nonzero rows.
"""
function jw_quadratic_residual(m::JWMap, T, Hspin; V = sparse(1.0I, size(T, 2), size(T, 2)))
    M = length(m.c)
    col(X) = reshape(X, size(X, 1) * size(X, 2), 1)     # stays sparse; `vec` would not
    TV, rhs = T * sparse(V), col(T * sparse(Hspin * V))
    cd = [sparse(c') for c in m.c]
    cT = [m.c[a] * TV for a in 1:M]                     # c_a T V and c_a^dag T V, the columns'
    dT = [cd[a] * TV for a in 1:M]                      # factors: no 2^M-square product formed
    cols = [col(TV)]
    for a in 1:M, b in 1:M
        push!(cols, col(cd[a] * cT[b]))
    end
    for a in 1:M, b in (a + 1):M
        push!(cols, col(m.c[a] * cT[b])); push!(cols, col(cd[a] * dT[b]))
    end
    A = reduce(hcat, cols)
    rows = sort!(unique!([rowvals(A); rowvals(rhs)]))
    A, rhs = A[rows, :], Vector(rhs[rows, 1])
    x = qr(A) \ rhs                                     # sparse least squares (rank-revealing)
    return norm(A * x - rhs) / norm(rhs), length(cols)
end
