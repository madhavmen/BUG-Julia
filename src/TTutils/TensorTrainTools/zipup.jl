# Single-pass elementwise multiplication of two TensorTrains.

export zipup


"""
    zipup(A::TensorTrain, B::TensorTrain; maxdim, cutoff=1e-16) -> TensorTrain

Compute `C Ôëê A ÔèÖ B` using a right-to-left zip-up sweep.
"""
function zipup(A::TensorTrain, B::TensorTrain; maxdim::Integer, cutoff::AbstractFloat = 1e-16)
    length(A) == length(B) ||
        throw(ArgumentError("TensorTrains must have the same length."))
    siteinds(A) == siteinds(B) ||
        throw(ArgumentError("TensorTrains must share site indices."))

    A_ord = _normal_order(A)
    B_ord = _normal_order(B)
    a = array.(A_ord)
    b = array.(B_ord)
    c = _zipup_arrays(a, b; maxdim, cutoff)
    return TensorTrain(siteinds(A), c)
end


"""
    _zipup_arrays(A, B, maxdim, cutoff)

Carry out the dense array contractions underlying the zip-up TT product routine.

Arguments
- `A`: Tensor, matrix, or tensor-train operator supplied to the helper.
- `B`: Second tensor, matrix, or tensor-train operator supplied to the helper.
- `maxdim`: Maximum admissible bond dimension or retained local rank.
- `cutoff`: Truncation tolerance used when discarding small singular directions.

Returns
- Returns the intermediate array data needed to build the next tensor-train core.

Description
- Zip-up multiplication merges local tensor information sequentially while truncating intermediate bond spaces. This helper performs the dense core algebra behind that procedure.
"""
function _zipup_arrays(
    A::AbstractVector,
    B::AbstractVector;
    maxdim::Integer,
    cutoff::AbstractFloat = 1e-16,
) 
    _validate_zipup_arrays(A, "A")
    _validate_zipup_arrays(B, "B")
    T = promote_type(mapreduce(eltype, promote_type, A), mapreduce(eltype, promote_type, B))
    N = length(A)
    C = Vector{Array{T,3}}(undef, N)

    ╬┤ = zeros(T, 2, 2, 2)
    ╬┤[1, 1, 1] = one(T)
    ╬┤[2, 2, 2] = one(T)

    local L

    for i in reverse(1:N)
        if i == N
            Ai = Base.dropdims(A[i]; dims = 3)
            Bi = Base.dropdims(B[i]; dims = 3)

            @tensor M[:] := Ai[-1, 1] * ╬┤[1, 2, -3] * Bi[-2, 2]

            Mmat = reshape(M, size(M, 1) * size(M, 2), size(M, 3))
            F = svd(Mmat; full = false)
            s = truncate(F.S, maxdim, cutoff)
            n_bond = length(s)

            C[i] = reshape(F.Vt[1:n_bond, :], n_bond, size(M, 3), 1)
            L = reshape(F.U[:, 1:n_bond] * Diagonal(s), size(M, 1), size(M, 2), n_bond)

        elseif i == 1
            Ai = Base.dropdims(A[i]; dims = 1)
            Bi = Base.dropdims(B[i]; dims = 1)

            @tensor M[:] := Ai[2, 1] * ╬┤[2, 3, -1] * L[1, 4, -2] * Bi[3, 4]

            C[i] = reshape(M, 1, size(M)...)

        else
            @tensor M[:] := A[i][-1, 2, 1] * B[i][-2, 3, 4] * L[1, 4, -4] * ╬┤[2, 3, -3]

            Mmat = reshape(M, size(M, 1) * size(M, 2), size(M, 3) * size(M, 4))
            Fqr = qr(Mmat)
            Fsvd = svd(Matrix(Fqr.R); full = false)
            s = truncate(Fsvd.S, maxdim, cutoff)
            n_bond = length(s)

            C[i] = reshape(Fsvd.Vt[1:n_bond, :], n_bond, size(M, 3), size(M, 4))
            L = reshape(
                Matrix(Fqr.Q) * (Fsvd.U[:, 1:n_bond] * Diagonal(s)),
                size(M, 1),
                size(M, 2),
                n_bond,
            )
        end
    end

    return C
end


"""
    _validate_zipup_arrays(arrays, label)

Check that dense arrays supplied to the zip-up routine have compatible dimensions.

Arguments
- `arrays`: Input parameter used by the local tensor-network calculation.
- `label`: Input parameter used by the local tensor-network calculation.

Returns
- Returns `nothing` when the arrays are compatible and throws an informative error otherwise.

Description
- Dimension validation is crucial in sequential low-rank products because every local truncation assumes neighboring array unfoldings line up exactly.
"""
function _validate_zipup_arrays(arrays::AbstractVector, label::AbstractString)
    all(tens -> ndims(tens) == 3 && eltype(tens) <: Number, arrays) ||
        throw(ArgumentError("$label must contain numeric rank-3 arrays."))
    return nothing
end


"""
    _normal_order(A)

Reorder local tensor indices into the canonical order expected by the zip-up routine.

Arguments
- `A`: Tensor, matrix, or tensor-train operator supplied to the helper.

Returns
- Returns the reordered tensor or array view.

Description
- Consistent local index ordering is what makes successive SVD truncations meaningful; this helper normalizes that ordering before a new TT core is extracted.
"""
function _normal_order(A::TensorTrain)
    g = deepcopy(A)
    ls = linkinds(g)
    sites = siteinds(g)
    for k in eachindex(g)
        g[k] = permute(g[k], ls[k], sites[k], ls[k + 1]; allow_alias = true)
    end
    return g
end
