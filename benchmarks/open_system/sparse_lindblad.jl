# EXACT SMALL-L REFERENCE FOR THE OPEN-SYSTEM MODELS (plan Tasks 6-7): the Lindbladian as a SPARSE
# superoperator on vec(rho), propagated by Arnoldi with Saad's error estimate. No 4^L / 9^L dense
# matrix is formed, so OS-2 at L = 6, nmax = 2 (531441-dimensional Liouville space) is affordable.
#
# The model is given exactly as `lindblad_mpo` takes it -- `pairs = [(A, B, Jmat)]`, `onsite_h`,
# `jumps = [(Jop, rates)]` (a rate g is D[sqrt(g) Jop]) -- so the MPDO and its reference cannot
# describe different models. vec is ROW-MAJOR, `(ket - 1) D + bra` (liouville.jl and the dense
# `_lv_dense` of test_liouville.jl), site 1 the most significant index.

using LinearAlgebra, SparseArrays

_sp_emb(o, i, L, d) = kron(sparse(1.0I, d^(i - 1), d^(i - 1)), sparse(ComplexF64.(o)),
                           sparse(1.0I, d^(L - i), d^(L - i)))

"Sparse `H` on the full `d^L` space from the `lindblad_mpo` model description."
function sparse_hamiltonian(L, d, pairs, onsite_h)
    D = d^L
    H = spzeros(ComplexF64, D, D)
    for (A, B, Jm) in pairs, i in 1:L, j in (i + 1):L
        iszero(Jm[i, j]) || (H += Jm[i, j] * _sp_emb(A, i, L, d) * _sp_emb(B, j, L, d))
    end
    if onsite_h !== nothing
        hs = onsite_h isa AbstractMatrix ? fill(onsite_h, L) : onsite_h
        for i in 1:L
            iszero(hs[i]) || (H += _sp_emb(hs[i], i, L, d))
        end
    end
    return H
end

"""
    sparse_lindbladian(L, d; pairs, onsite_h = nothing, jumps = []) -> SparseMatrixCSC

`vec(L rho)` for row-major `vec`: `A rho B -> kron(A, transpose(B))`, so
`-i[H, rho] -> -i (H (x) I - I (x) H^T)` and `D[X] -> X (x) conj(X) - (X'X (x) I + I (x) (X'X)^T)/2`.
"""
function sparse_lindbladian(L, d; pairs, onsite_h = nothing, jumps = Any[])
    D = d^L
    H = sparse_hamiltonian(L, d, pairs, onsite_h)
    Id = sparse(1.0I, D, D)
    S = -1im * (kron(H, Id) - kron(Id, transpose(H)))
    for (Jop, rates) in jumps, i in 1:L
        g = rates isa Number ? rates : rates[i]
        iszero(g) && continue
        X = sqrt(g) * _sp_emb(Jop, i, L, d)
        XX = X' * X
        S += kron(X, conj(X)) - kron(XX, Id) / 2 - kron(Id, transpose(XX)) / 2
    end
    return dropzeros!(S)
end

"Row-major `vec` of a `D x D` matrix and back."
vec_rm(r) = vec(permutedims(r))
unvec_rm(v, D) = permutedims(reshape(v, D, D))

"""
    arnoldi_expv_sparse(A, v, t; m = 40, tol = 1e-12) -> Vector

`exp(t A) v` for a sparse NON-Hermitian `A` (a Lindbladian), by Arnoldi with full double
Gram-Schmidt, restarted in substeps chosen up front from `||A||_1` (`|h| ||A|| <= m/4`), each
checked by Saad's estimate `h_{m+1,m} |[exp(h H_m)]_{m,1}|` and HALVED when it fails -- a reference
that cannot detect its own failure is worse than none (`expv_sparse`'s rule, exact_sparse.jl).
"""
function arnoldi_expv_sparse(A, v::Vector{ComplexF64}, t::Real; m::Int = 40, tol::Float64 = 1e-12)
    total = float(t)
    (total == 0 || norm(v) == 0) && return copy(v)
    nrm = opnorm(A, 1)
    step = nrm > 0 ? min(total, 0.25 * m / nrm) : total
    w, done = copy(v), 0.0
    while done < total * (1 - 1e-12)
        h = min(step, total - done)
        b0 = norm(w)
        V = Vector{Vector{ComplexF64}}(undef, m + 1)
        Hm = zeros(ComplexF64, m + 1, m)
        V[1] = w / b0
        k = m
        for j in 1:m
            x = A * V[j]
            for _pass in 1:2, i in 1:j
                c = dot(V[i], x)
                Hm[i, j] += c
                x .-= c .* V[i]
            end
            Hm[j + 1, j] = norm(x)
            if Hm[j + 1, j] < 1e-14 * b0
                k = j; break
            end
            V[j + 1] = x / Hm[j + 1, j]
        end
        E = exp(h * Hm[1:k, 1:k])
        err = k < m ? 0.0 : abs(Hm[k + 1, k]) * abs(E[k, 1])
        if err > tol && h > 1e-10 * total
            step = h / 2
            continue
        end
        w = b0 .* sum(E[i, 1] .* V[i] for i in 1:k)
        done += h
    end
    return w
end

"`exp(t L) vec(rho0)` at every time of an ascending list, stepping from the previous one."
function sparse_lindblad_trajectory(S, rho0::AbstractMatrix, ts; kwargs...)
    D = size(rho0, 1)
    v = vec_rm(ComplexF64.(rho0))
    out = Matrix{ComplexF64}[]
    tprev = 0.0
    for t in ts
        v = arnoldi_expv_sparse(S, v, t - tprev; kwargs...)
        push!(out, unvec_rm(v, D))
        tprev = t
    end
    return out
end

"`Tr(O_i rho)` for a one-site operator, from a full density matrix."
site_expect(rho, O, i, L, d) = tr(_sp_emb(O, i, L, d) * rho)
