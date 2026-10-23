# tests/common/liouville_fixtures.jl -- dense Liouvillian references and the XX / Bose-Hubbard
# Lindblad models of the MPDO tests.

# ── dense references, written from the textbook formulas ───────────────────────────────────────

_lv_emb(o, i, L, d) = kron(Matrix{ComplexF64}(I, d^(i - 1), d^(i - 1)), o,
                           Matrix{ComplexF64}(I, d^(L - i), d^(L - i)))
_vecr(r) = vec(permutedims(r))                       # row-major: index (ket - 1) D + bra
_unvecr(v, D) = permutedims(reshape(v, D, D))

"Dense Liouvillian in the row-major vec basis of the full space."
function _lv_dense(L, d, pairs, hs, jumps)
    D = d^L
    H = zeros(ComplexF64, D, D)
    for (A, B, Jm) in pairs, i in 1:L, j in (i + 1):L
        iszero(Jm[i, j]) || (H += Jm[i, j] * _lv_emb(A, i, L, d) * _lv_emb(B, j, L, d))
    end
    for i in 1:L
        H += _lv_emb(hs[i], i, L, d)
    end
    Id = Matrix{ComplexF64}(I, D, D)
    S = -1im * (kron(H, Id) - kron(Id, transpose(H)))
    for (J, rates) in jumps, i in 1:L
        g = rates isa Number ? rates : rates[i]
        iszero(g) && continue
        X = sqrt(g) * _lv_emb(J, i, L, d)
        XX = X' * X
        S += kron(X, conj(X)) - kron(XX, Id) / 2 - kron(Id, transpose(XX)) / 2
    end
    return S
end

"Index map: interleaved Liouville configuration (site 1 most significant) -> row-major (ka, kb)."
function _lv_perm(L, d)
    Dl = d^2
    p = zeros(Int, Dl^L)
    for idx in 1:(Dl^L)
        gs = reverse(digits(idx - 1; base = Dl, pad = L)) .+ 1
        ka = 1; kb = 1
        for g in gs
            a, b = liouville_pair(d, g)
            ka = (ka - 1) * d + a; kb = (kb - 1) * d + b
        end
        p[idx] = (ka - 1) * d^L + kb
    end
    return p
end

"The MPO as a dense matrix over interleaved Liouville configurations (rows = s_ket legs), by
walking its abelian blocks exactly as `mpdo_dense` walks a state."
function _lv_mpo_dense(W, ls, L)
    Dl = ls.d^2
    env = nothing
    n = 1
    for i in 1:L
        T = to_concrete(W[i])
        new = Dict{Any, Array{ComplexF64, 3}}()
        for (k, ql) in pairs(T.qlabels)
            wl, sk, sb, wr = ql
            X = Array(T.RMTs[k])
            R = reshape(X, size(X, 1), size(X, 2), size(X, 3), size(X, 4))
            E = env === nothing ? ones(ComplexF64, 1, 1, size(R, 1)) :
                (haskey(env, wl) ? env[wl] : continue)
            acc = get!(new, wr, zeros(ComplexF64, n * Dl, n * Dl, size(R, 4)))
            for p in 1:size(R, 2), q in 1:size(R, 3)
                gk, gb = ls.members[sk][p], ls.members[sb][q]
                rk = ((0:(n - 1)) .* Dl) .+ gk
                rb = ((0:(n - 1)) .* Dl) .+ gb
                for w in 1:size(R, 4), v in 1:size(R, 1)
                    c = R[v, p, q, w]
                    iszero(c) || (acc[rk, rb, w] .+= c .* E[:, :, v])
                end
            end
        end
        env = new
        n *= Dl
    end
    return sum(dropdims(sum(m; dims = 3); dims = 3) for m in values(env))
end

# ── models ─────────────────────────────────────────────────────────────────────────────────────

const _SPm = ComplexF64[0 1; 0 0]                      # sigma^+ (state 1 = up)
const _SMm = ComplexF64[0 0; 1 0]
const _NUP = ComplexF64[1 0; 0 0]

"Spin-1/2 XX chain + dephasing + boundary drive (gain at 1, loss at L): weak U(1) only."
function _xx_model(L; J = 1.0, gam = 0.3, gL = 0.5, gR = 0.7, drive = true)
    Jm = zeros(L, L)
    for i in 1:(L - 1); Jm[i, i + 1] = J; end
    pairs = Any[(_SPm, _SMm, Jm), (_SMm, _SPm, Jm)]
    hs = [0.2 * i * _NUP for i in 1:L]                    # a site-dependent field
    jumps = Any[(_NUP, gam)]
    if drive
        push!(jumps, (_SPm, [i == 1 ? gL : 0.0 for i in 1:L]))
        push!(jumps, (_SMm, [i == L ? gR : 0.0 for i in 1:L]))
    end
    return pairs, hs, jumps
end

"Bose-Hubbard, cutoff nmax, with on-site two-body loss b^2."
function _bh_model(L; nmax = 2, J = 1.0, U = 2.0, gam = 0.4)
    d = nmax + 1
    b = zeros(ComplexF64, d, d)
    for n in 1:nmax; b[n, n + 1] = sqrt(n); end           # state k holds k-1 bosons
    nn = Diagonal(ComplexF64.(0:nmax))
    Jm = zeros(L, L)
    for i in 1:(L - 1); Jm[i, i + 1] = -J; end
    pairs = Any[(Matrix(b'), b, Jm), (b, Matrix(b'), Jm)]
    hs = [Matrix(U / 2 * nn * (nn - I)) for _ in 1:L]
    return pairs, hs, Any[(b * b, gam)], collect(0:nmax), d
end
