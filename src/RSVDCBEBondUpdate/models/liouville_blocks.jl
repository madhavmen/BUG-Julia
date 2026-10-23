# liouville_blocks.jl: MPDO contractions by walking the abelian blocks (from `liouville.jl`, included right after it).

# ── contractions by walking the abelian blocks ─────────────────────────────────────────────────

"""
    _mpdo_walk(rho, ls, cov) -> ComplexF64

`sum over configurations of prod_j cov[j][g_j] * rho(g_1 ... g_L)`: the state contracted with a
PRODUCT covector, one length-`d^2` vector per site. Done block by block over the abelian sectors
(an environment row vector per link label), so it needs no covector MPS and works for every
grading. The trace is `cov[j] = vec(I)`; `Tr(O_j rho)` puts `vec(transpose(O))` at site `j`.
"""
function _mpdo_walk(rho::SymMPS, ls::LiouvilleSpace, cov::AbstractVector)
    L = length(rho)
    length(cov) == L || throw(DimensionMismatch("need one covector per site"))
    env = nothing                                      # label => row vector over the link
    for i in 1:L
        A = to_concrete(rho[i])
        new = Dict{Any, Vector{ComplexF64}}()
        for (k, ql) in pairs(A.qlabels)
            cl, s, cr = ql
            R = _rmt3(A.RMTs[k])
            dl, dp, dr = size(R)
            el = if env === nothing
                ones(ComplexF64, dl)
            else
                haskey(env, cl) || continue
                env[cl]
            end
            gs = ls.members[s]
            t = zeros(ComplexF64, dl, dr)
            for p in 1:dp
                w = cov[i][gs[p]]
                iszero(w) || (t .+= w .* @view R[:, p, :])
            end
            iszero(t) && continue
            acc = get!(new, cr, zeros(ComplexF64, dr))
            acc .+= transpose(t) * el
        end
        env = new
    end
    return sum((sum(v) for v in values(env)); init = zero(ComplexF64))
end

"A block's RMT as a dense `(link_l, phys, link_r)` array: RMTs carry one trailing singleton
multiplicity dimension per abelian symmetry (rank `QD + N`), and may be a `DiagRMT`."
_rmt3(R) = (X = Array(R); reshape(X, size(X, 1), size(X, 2), size(X, 3)))

"`vec(I)` as a Liouville covector: picks the diagonal `|a><a|` entries."
_lv_trace_cov(ls::LiouvilleSpace) =
    ComplexF64[a == b ? 1 : 0 for a in 1:ls.d for b in 1:ls.d]

"`Tr(O rho_site)` as a Liouville covector: `sum_ab O[b,a] rho[a,b]`."
_lv_op_cov(ls::LiouvilleSpace, O::AbstractMatrix) =
    ComplexF64[O[b, a] for a in 1:ls.d for b in 1:ls.d]

"`Tr rho` of an MPDO."
mpdo_trace(rho::SymMPS, ls::LiouvilleSpace) =
    _mpdo_walk(rho, ls, fill(_lv_trace_cov(ls), length(rho)))

"""
    mpdo_expect(rho, ls, ops::Dict{Int, <:AbstractMatrix}) -> ComplexF64

`Tr(prod_j O_j rho) / Tr(rho)` for one-site physical operators on distinct sites (`Dict` site =>
`d x d` matrix); NOT divided when `normalise = false`.
"""
function mpdo_expect(rho::SymMPS, ls::LiouvilleSpace, ops::AbstractDict; normalise::Bool = true)
    cov = [haskey(ops, j) ? _lv_op_cov(ls, ops[j]) : _lv_trace_cov(ls) for j in 1:length(rho)]
    v = _mpdo_walk(rho, ls, cov)
    return normalise ? v / mpdo_trace(rho, ls) : v
end

"""
    mpdo_rdm(rho, ls, sites) -> Matrix{ComplexF64}

The reduced density matrix of `sites` (sorted, distinct), normalised to unit trace, from
`d^(2k)` product-covector walks: `rho_S[a, b] = Tr(|b><a|_S rho) / Tr rho`. For local
Hermiticity / positivity diagnostics on states too large for [`mpdo_dense`](@ref).
"""
function mpdo_rdm(rho::SymMPS, ls::LiouvilleSpace, sites::AbstractVector{<:Integer})
    issorted(sites) && allunique(sites) || throw(ArgumentError("sites must be sorted and distinct"))
    d, k = ls.d, length(sites)
    tr = mpdo_trace(rho, ls)
    out = zeros(ComplexF64, d^k, d^k)
    for ia in CartesianIndices(ntuple(_ -> d, k)), ib in CartesianIndices(ntuple(_ -> d, k))
        ops = Dict{Int, Matrix{Float64}}()
        for (m, j) in pairs(sites)
            E = zeros(d, d); E[ib[m], ia[m]] = 1.0         # Tr(|b><a| rho) = rho[a, b]
            ops[j] = E
        end
        a = sum((ia[m] - 1) * d^(k - m) for m in 1:k) + 1
        b = sum((ib[m] - 1) * d^(k - m) for m in 1:k) + 1
        out[a, b] = mpdo_expect(rho, ls, ops; normalise = false) / tr
    end
    return out
end

"""
    mpdo_dense(rho, ls) -> Matrix{ComplexF64}

The full `d^L x d^L` density matrix (site 1 the most significant index, ket rows, bra columns),
NOT normalised. Small `L` only: it enumerates every configuration through the same block walk.
"""
function mpdo_dense(rho::SymMPS, ls::LiouvilleSpace)
    L, d = length(rho), ls.d
    d^L <= 4096 || throw(ArgumentError("mpdo_dense is for small systems (d^L <= 4096)"))
    D = d^2
    env = nothing                                      # label => (nacc x dlink) amplitudes
    nacc = 1
    for i in 1:L
        A = to_concrete(rho[i])
        new = Dict{Any, Matrix{ComplexF64}}()
        for (k, ql) in pairs(A.qlabels)
            cl, s, cr = ql
            R = _rmt3(A.RMTs[k])
            dl, dp, dr = size(R)
            E = env === nothing ? ones(ComplexF64, 1, dl) :
                (haskey(env, cl) ? env[cl] : continue)
            acc = get!(new, cr, zeros(ComplexF64, nacc * D, dr))
            for p in 1:dp
                g = ls.members[s][p]
                rows = ((0:(nacc - 1)) .* D) .+ g
                acc[rows, :] .+= E * R[:, p, :]
            end
        end
        env = new
        nacc *= D
    end
    v = sum(sum(m; dims = 2)[:, 1] for m in values(env))
    rhoD = zeros(ComplexF64, d^L, d^L)
    for idx in 1:(D^L)
        gs = reverse(digits(idx - 1; base = D, pad = L)) .+ 1        # site 1 most significant
        ka = 1; kb = 1
        for g in gs
            a, b = liouville_pair(d, g)
            ka = (ka - 1) * d + a
            kb = (kb - 1) * d + b
        end
        rhoD[ka, kb] = v[idx]
    end
    return rhoD
end

"""
    mpdo_normalise_trace!(rho, ls) -> trace before

Rescale the orthogonality centre so `Tr rho = 1` (the physical normalisation of an MPDO; its
Frobenius norm is the purity's square root, not 1).
"""
function mpdo_normalise_trace!(rho::SymMPS, ls::LiouvilleSpace)
    t = mpdo_trace(rho, ls)
    (isfinite(abs(t)) && abs(t) > 0) || throw(ErrorException("MPDO trace is $t"))
    c = (1 <= rho.center <= length(rho)) ? rho.center : 1
    rho[c] = to_concrete((1 / t) * rho[c])
    return t
end

"""
    mpdo_checks(rho, ls; dense = length(rho) * log2(ls.d) <= 10) -> NamedTuple

Trace, Hermiticity and positivity of an MPDO. `dense = true`: from the full matrix -- `herm`
`= ||rho - rho'|| / ||rho||` and `min_eig`, the smallest eigenvalue of the Hermitian part of
`rho / Tr rho`. Otherwise the same two numbers from every nearest-neighbour two-site reduced
density matrix (`local = true`), which can only witness a violation, not certify its absence.
"""
function mpdo_checks(rho::SymMPS, ls::LiouvilleSpace; dense::Bool = length(rho) * log2(ls.d) <= 10)
    tr = mpdo_trace(rho, ls)
    if dense
        R = mpdo_dense(rho, ls) / tr
        H = (R + R') / 2
        return (trace = tr, herm = norm(R - R') / max(norm(R), eps()),
                min_eig = minimum(eigvals(Hermitian(H))), local_only = false)
    end
    herm, mn = 0.0, Inf
    for j in 1:(length(rho) - 1)
        R = mpdo_rdm(rho, ls, [j, j + 1])
        herm = max(herm, norm(R - R') / max(norm(R), eps()))
        mn = min(mn, minimum(eigvals(Hermitian((R + R') / 2))))
    end
    return (trace = tr, herm = herm, min_eig = mn, local_only = true)
end
