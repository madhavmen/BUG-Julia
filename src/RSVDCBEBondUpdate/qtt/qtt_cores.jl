# ── dense core algebra for quantics fields and operators (QTT plan, Tasks 1.1 and 1.3A) ──
# Fields and operators are built here as plain Julia arrays and only then wrapped as Telum tensors
# (`qtt_telum.jl`): under `:none` a Telum tensor is ONE dense block, so nothing is lost, and the
# construction never touches a 2^L object. Conventions, pinned by tests/sweeps/test_qtt_u01_u02.jl:
#   TT core       (chi_l, d, chi_r);  site 1 is the FASTEST index of a dense contraction
#   operator core (w_l, out, in, w_r); (O psi)[out] = sum_in O[out, in] psi[in]
# A 1D chain is listed in LOGICAL order, bit 0 (least significant) first; `qtt_embed` places it on
# the grid's sites.

# ── TT basics ──

"Smallest rank `r >= 1` whose discarded tail `sqrt(sum S[r+1:end]^2)` is at most `delta`, capped."
function _qtt_rank(S::AbstractVector{<:Real}, delta::Real, maxdim::Int)
    r = length(S)
    tail = 0.0
    while r > 1 && tail + S[r]^2 <= delta^2
        tail += S[r]^2
        r -= 1
    end
    return max(min(r, maxdim), 1)
end

"""
    tt_dense(C) -> Array

The full tensor `(d, ..., d)` of a TT, site 1 first (fastest). Small sizes only.
"""
function tt_dense(C::AbstractVector)
    L = length(C)
    d = size(C[1], 2)
    size(C[1], 1) == 1 || throw(ArgumentError("tt_dense: left boundary rank is $(size(C[1], 1))"))
    M = reshape(C[1], d, size(C[1], 3))
    for k in 2:L
        chi, dk, chr = size(C[k])
        M = reshape(M * reshape(C[k], chi, dk * chr), :, chr)
    end
    size(M, 2) == 1 || throw(ArgumentError("tt_dense: right boundary rank is $(size(M, 2))"))
    return reshape(M, ntuple(k -> size(C[k], 2), L))
end

"""
    tt_from_dense(A; tol = 1e-14, maxdim) -> Vector{Array{T,3}}

TT-SVD of a `(d, ..., d)` array (site 1 = dimension 1) at relative Frobenius tolerance `tol`,
split evenly over the bonds. Fixture construction and 1D functions only.
"""
function tt_from_dense(A::AbstractArray{T}; tol::Real = 1e-14,
                       maxdim::Int = typemax(Int)) where {T}
    L = ndims(A)
    nrm = norm(A)
    delta = tol * nrm / sqrt(max(L - 1, 1))
    C = Vector{Array{T, 3}}(undef, L)
    L == 1 && (C[1] = reshape(Array(A), 1, size(A, 1), 1); return C)
    M = reshape(Array(A), size(A, 1), :)
    chi = 1
    for k in 1:(L - 1)
        d = size(A, k)
        M = reshape(M, chi * d, :)
        F = svd(M)
        r = _qtt_rank(F.S, delta, maxdim)
        C[k] = reshape(F.U[:, 1:r], chi, d, r)
        M = Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        chi = r
    end
    C[L] = reshape(M, chi, size(A, L), 1)
    return C
end

"""
    tt_from_samples(vals; tol = 1e-14, maxdim) -> TT

The 1D quantics TT of `vals[i + 1] = f(x_i)`, `length(vals) = 2^n`, in logical order (bit 0 first).
"""
function tt_from_samples(vals::AbstractVector; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    n = round(Int, log2(length(vals)))
    2^n == length(vals) || throw(ArgumentError("need 2^n samples, got $(length(vals))"))
    return tt_from_dense(reshape(collect(vals), ntuple(_ -> 2, n)); tol = tol, maxdim = maxdim)
end

"""
    tt_compress(C; tol = 1e-14, maxdim, relative = true) -> (C, discarded)

Left QR sweep, then a right-to-left SVD sweep keeping each bond's smallest rank whose tail fits
`tol / sqrt(L - 1)` (of the norm when `relative`). The result is right-canonical (site 1 holds the
norm). `discarded` is the relative Frobenius error bound (the sum of the bonds' tails over the
norm). Works for any physical dimension `d` (operators go through `op_compress`).
"""
function tt_compress(C0::AbstractVector; tol::Real = 1e-14, maxdim::Int = typemax(Int),
                     relative::Bool = true)
    L = length(C0)
    T = mapreduce(eltype, promote_type, C0)
    C = [Array{T, 3}(c) for c in C0]
    for k in 1:(L - 1)
        chi, d, r = size(C[k])
        F = qr(reshape(C[k], chi * d, r))
        Q = Matrix(F.Q)
        m = size(Q, 2)
        C[k] = reshape(Q, chi, d, m)
        c2, d2, r2 = size(C[k + 1])
        C[k + 1] = reshape(F.R * reshape(C[k + 1], c2, d2 * r2), m, d2, r2)
    end
    nrm = norm(C[L])
    delta = (relative ? tol * nrm : tol) / sqrt(max(L - 1, 1))
    disc = 0.0
    for k in L:-1:2
        chi, d, r = size(C[k])
        F = svd(reshape(C[k], chi, d * r))
        rr = _qtt_rank(F.S, delta, maxdim)
        disc += sqrt(sum(abs2, @view(F.S[(rr + 1):end]); init = 0.0))
        C[k] = reshape(F.Vt[1:rr, :], rr, d, r)
        c1, d1, _ = size(C[k - 1])
        C[k - 1] = reshape(reshape(C[k - 1], c1 * d1, chi) * (F.U[:, 1:rr] * Diagonal(F.S[1:rr])),
                           c1, d1, rr)
    end
    return C, (nrm > 0 ? disc / nrm : 0.0)
end

"Bond dimensions `chi_1 ... chi_{L-1}` of a TT or operator chain."
tt_ranks(C::AbstractVector) = [size(C[k], ndims(C[k])) for k in 1:(length(C) - 1)]

"`<A|B>` (A conjugated) of two TTs with the same physical dimensions."
function tt_dot(A::AbstractVector, B::AbstractVector)
    E = ones(promote_type(eltype(A[1]), eltype(B[1])), 1, 1)
    for k in eachindex(A)
        a, b = A[k], B[k]
        X = reshape(E * reshape(b, size(b, 1), :), size(a, 1) * size(a, 2), size(b, 3))
        E = reshape(a, size(a, 1) * size(a, 2), size(a, 3))' * X
    end
    return E[1, 1]
end

tt_norm(A::AbstractVector) = sqrt(max(real(tt_dot(A, A)), 0.0))

"`c A` (the scalar on site 1)."
tt_scale(A::AbstractVector, c::Number) = [k == 1 ? c .* A[1] : copy(A[k]) for k in eachindex(A)]

tt_conj(A::AbstractVector) = [conj.(a) for a in A]

"""
    tt_add(A, B) -> TT

`A + B` by direct sum of the bonds (rank adds; compress afterwards).
"""
function tt_add(A::AbstractVector, B::AbstractVector)
    L = length(A)
    length(B) == L || throw(DimensionMismatch("tt_add: $L sites against $(length(B))"))
    T = promote_type(eltype(A[1]), eltype(B[1]))
    L == 1 && return [T.(A[1]) .+ B[1]]
    C = Vector{Array{T, 3}}(undef, L)
    for k in 1:L
        a, b = A[k], B[k]
        size(a, 2) == size(b, 2) || throw(DimensionMismatch("tt_add: site $k dimensions differ"))
        C[k] = if k == 1
            cat(T.(a), T.(b); dims = 3)
        elseif k == L
            cat(T.(a), T.(b); dims = 1)
        else
            c = zeros(T, size(a, 1) + size(b, 1), size(a, 2), size(a, 3) + size(b, 3))
            c[1:size(a, 1), :, 1:size(a, 3)] .= a
            c[(size(a, 1) + 1):end, :, (size(a, 3) + 1):end] .= b
            c
        end
    end
    return C
end

"`sum_t c_t A_t`, compressed once at `tol`."
function tt_lincomb(terms::AbstractVector; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    acc = nothing
    for (c, A) in terms
        X = tt_scale(A, c)
        acc = acc === nothing ? X : tt_add(acc, X)
    end
    return first(tt_compress(acc; tol = tol, maxdim = maxdim))
end

"""
    tt_hadamard(A, B) -> TT

The elementwise product `A .* B`, exact: rank `chi_A chi_B` (compress afterwards).
"""
function tt_hadamard(A::AbstractVector, B::AbstractVector)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    C = Vector{Array{T, 3}}(undef, length(A))
    for k in eachindex(A)
        a, b = A[k], B[k]
        c = zeros(T, size(a, 1) * size(b, 1), size(a, 2), size(a, 3) * size(b, 3))
        for s in 1:size(a, 2)
            c[:, s, :] .= kron(b[:, s, :], a[:, s, :])        # A's bond index fastest
        end
        C[k] = c
    end
    return C
end

"""
    tt_density(A; tol = 1e-12, maxdim) -> (rho, info)

The density `|A|^2 = A .* conj(A)` as a compressed TT WITHOUT forming the `chi^2` cores: a left
zip-up over a right-canonical copy of `A` (each Hadamard right part is then a contraction, so the
sum of the per-bond tails bounds the error), then a canonical recompression. The relative error
bound is checked against `tol` and the build repeated once at a tighter cut if it misses.
`info = (chi, err_bound, seconds, retried)`. Chen et al. (arXiv:2507.04279v5 Sec. 3.1) report this
density build as a dominant `O(D^4)` cost, so its time is returned for the step ledger (`f_rho`).
"""
function tt_density(A0::AbstractVector; tol::Real = 1e-12, maxdim::Int = typemax(Int))
    t0 = time_ns()
    A, _ = tt_compress(A0; tol = 0.0)
    return _product_rc(A, tt_conj(A), tol, maxdim, t0)
end

"""
    tt_product(A, B; tol = 1e-12, maxdim) -> (C, info)

`A .* B` compressed, by the zip-up of [`tt_density`](@ref) (same error bound, retry and `info`):
never the exact `chi_A chi_B` cores of [`tt_hadamard`](@ref), which at the GP's `psi^2` Jacobian
term reach GBs per core (t3g1/t3v1 OOM at 12 G).
"""
function tt_product(A0::AbstractVector, B0::AbstractVector; tol::Real = 1e-12,
                    maxdim::Int = typemax(Int))
    t0 = time_ns()
    A, _ = tt_compress(A0; tol = 0.0)
    B = B0 === A0 ? A : first(tt_compress(B0; tol = 0.0))
    return _product_rc(A, B, tol, maxdim, t0)
end

"The zip-up product of right-canonical `A`, `B`, retried once at a tighter cut, recompressed."
function _product_rc(A, B, tol::Real, maxdim::Int, t0)
    C, bound = _product_zipup(A, B, tol, maxdim)
    retried = false
    if bound > tol && tol > 0
        C, bound = _product_zipup(A, B, 0.25 * tol * tol / bound, maxdim)
        retried = true
    end
    C, d2 = tt_compress(C; tol = 0.1 * tol, maxdim = maxdim)
    return C, (chi = maximum(tt_ranks(C); init = 1), err_bound = bound + d2,
               seconds = (time_ns() - t0) / 1e9, retried = retried)
end

function _product_zipup(A::AbstractVector, B::AbstractVector, tol::Real, maxdim::Int)
    L = length(A)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    R = ones(T, 1, 1, 1)                                 # (r, a, b) -- a for A, b for B
    C = Vector{Array{T, 3}}(undef, L)
    disc = 0.0
    for k in 1:L
        a, b = A[k], B[k]
        cal, d, car = size(a)
        cbl, _, cbr = size(b)
        r = size(R, 1)
        # Y[r, a, s, b'] = sum_b R[r, a, b] B[b, s, b']
        Y = reshape(reshape(R, r * cal, cbl) * reshape(b, cbl, d * cbr), r, cal, d, cbr)
        Tk = zeros(T, r, d, car, cbr)                     # (r, s, a', b')
        for s in 1:d
            Ys = reshape(permutedims(Y[:, :, s, :], (1, 3, 2)), r * cbr, cal)   # (r b', a)
            Ts = reshape(Ys * a[:, s, :], r, cbr, car)                          # (r, b', a')
            Tk[:, s, :, :] .= permutedims(Ts, (1, 3, 2))
        end
        if k == L
            C[k] = reshape(Tk, r, d, 1)
            break
        end
        M = reshape(Tk, r * d, car * cbr)
        F = svd(M)
        delta = tol * norm(F.S) / sqrt(max(L - 1, 1))
        rr = _qtt_rank(F.S, delta, maxdim)
        disc += sqrt(sum(abs2, @view(F.S[(rr + 1):end]); init = 0.0))
        C[k] = reshape(F.U[:, 1:rr], r, d, rr)
        R = reshape(Diagonal(F.S[1:rr]) * F.Vt[1:rr, :], rr, car, cbr)
    end
    nr = norm(C[L])
    return C, (nr > disc ? disc / (nr - disc) : (disc > 0 ? Inf : 0.0))
end

"""
    tt_evaluate(C, bits) -> scalar

One amplitude: the product of the cores' `bits[k] + 1` slices. Never densifies.
"""
function tt_evaluate(C::AbstractVector, bits::AbstractVector{<:Integer})
    v = C[1][:, bits[1] + 1, :]
    for k in 2:length(C)
        v = v * C[k][:, bits[k] + 1, :]
    end
    return v[1, 1]
end

# ── operator cores ──

_as3(W::AbstractArray{<:Any, 4}) = reshape(W, size(W, 1), size(W, 2) * size(W, 3), size(W, 4))
_as4(C::AbstractArray{<:Any, 3}, d::Int = 2) = reshape(C, size(C, 1), d, d, size(C, 3))

"The `L`-site identity, bond dimension 1."
op_identity(L::Int; T::Type = Float64) = [reshape(Matrix{T}(I, 2, 2), 1, 2, 2, 1) for _ in 1:L]

op_scale(W::AbstractVector, c::Number) = [k == 1 ? c .* W[1] : copy(W[k]) for k in eachindex(W)]

"`A + B` (bond dimensions add)."
op_add(A::AbstractVector, B::AbstractVector) = [_as4(c) for c in tt_add(_as3.(A), _as3.(B))]

"`sum_t c_t W_t`, compressed once."
function op_lincomb(terms::AbstractVector; tol::Real = 1e-13, maxdim::Int = typemax(Int))
    acc = nothing
    for (c, W) in terms
        X = op_scale(W, c)
        acc = acc === nothing ? X : op_add(acc, X)
    end
    return op_compress(acc; tol = tol, maxdim = maxdim)
end

"SVD rounding of an operator chain (its cores read as a TT with `d = 4`)."
op_compress(W::AbstractVector; tol::Real = 1e-13, maxdim::Int = typemax(Int)) =
    [_as4(c) for c in first(tt_compress(_as3.(W); tol = tol, maxdim = maxdim))]

"The adjoint `O'` (conjugate transpose)."
op_adjoint(W::AbstractVector) = [conj.(permutedims(w, (1, 3, 2, 4))) for w in W]

"The transpose `O^T`."
op_transpose(W::AbstractVector) = [permutedims(w, (1, 3, 2, 4)) for w in W]

"""
    op_compose(A, B) -> operator

The product `A * B` (B acts first): bonds multiply, A's index fastest.
"""
function op_compose(A::AbstractVector, B::AbstractVector)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    C = Vector{Array{T, 4}}(undef, length(A))
    for k in eachindex(A)
        a, b = A[k], B[k]
        c = zeros(T, size(a, 1) * size(b, 1), 2, 2, size(a, 4) * size(b, 4))
        for o in 1:2, i in 1:2, m in 1:2
            c[:, o, i, :] .+= kron(b[:, m, i, :], a[:, o, m, :])
        end
        C[k] = c
    end
    return C
end

"`diag(f)` of a TT `f`: `W[a, o, i, b] = delta_{oi} f[a, o, b]`."
function op_diag(F::AbstractVector)
    W = Vector{Array{eltype(F[1]), 4}}(undef, length(F))
    for k in eachindex(F)
        f = F[k]
        w = zeros(eltype(f), size(f, 1), 2, 2, size(f, 3))
        for s in 1:2
            w[:, s, s, :] .= f[:, s, :]
        end
        W[k] = w
    end
    return W
end

"""
    op_apply(W, A) -> TT

`W * A` exactly (bond `w chi`, W's index fastest); compress afterwards.
"""
function op_apply(W::AbstractVector, A::AbstractVector)
    T = promote_type(eltype(W[1]), eltype(A[1]))
    C = Vector{Array{T, 3}}(undef, length(A))
    for k in eachindex(A)
        w, a = W[k], A[k]
        c = zeros(T, size(w, 1) * size(a, 1), 2, size(w, 4) * size(a, 3))
        for o in 1:2, i in 1:2
            c[:, o, :] .+= kron(a[:, i, :], w[:, o, i, :])
        end
        C[k] = c
    end
    return C
end

"""
    op_dense(W) -> Matrix

The `2^L x 2^L` matrix, rows and columns in the order of `tt_dense` (site 1 fastest). Small sizes
only -- tests.
"""
function op_dense(W::AbstractVector)
    size(W[1], 1) == 1 || throw(ArgumentError("op_dense: left boundary dimension $(size(W[1], 1))"))
    M = reshape(W[1], 2, 2, size(W[1], 4))
    P = 2
    for k in 2:length(W)
        wl, _, _, wr = size(W[k])
        X = reshape(reshape(M, P * P, wl) * reshape(W[k], wl, 4 * wr), P, P, 2, 2, wr)
        M = reshape(permutedims(X, (1, 3, 2, 4, 5)), 2P, 2P, wr)
        P *= 2
    end
    size(M, 3) == 1 || throw(ArgumentError("op_dense: right boundary dimension $(size(M, 3))"))
    return reshape(M, P, P)
end

"The rank-1 operator `|r><c|` on `n` bits (logical order), `r, c` 0-based."
function op_ket_bra(n::Int, r::Integer, c::Integer; T::Type = Float64)
    return [begin
                w = zeros(T, 1, 2, 2, 1)
                w[1, ((r >> k) & 1) + 1, ((c >> k) & 1) + 1, 1] = one(T)
                w
            end for k in 0:(n - 1)]
end

# ── placing 1D chains on a 2D grid ──

_bond_reverse(c::AbstractArray{<:Any, 3}) = permutedims(c, (3, 2, 1))
_bond_reverse(c::AbstractArray{<:Any, 4}) = permutedims(c, (4, 2, 3, 1))

"`site => core` of a 1D chain (logical order) on `coord`'s sites, oriented in site order."
function _site_chain(g::QTTGrid, coord::Symbol, C::AbstractVector)
    sites = coord === :x ? g.sites_x : g.sites_y
    length(C) == length(sites) || throw(DimensionMismatch(
        "a $coord chain needs $(length(sites)) cores, got $(length(C))"))
    p = sortperm(sites)                                   # logical bits in site order
    if p == collect(1:length(sites))
        return Dict(sites[k] => C[k] for k in eachindex(C))
    elseif p == collect(length(sites):-1:1)
        return Dict(sites[k] => _bond_reverse(C[k]) for k in eachindex(C))
    end
    throw(ArgumentError("coordinate $coord's bits are not monotone in site order"))
end

"""
    qtt_embed(g; x = nothing, y = nothing) -> chain

The 2D chain `X (x) Y` on `g`'s sites from 1D chains in logical order: TT cores (3-dim, a separable
field `f(x) g(y)`) or operator cores (4-dim, `O_x (x) O_y`). A missing coordinate is the constant 1
(fields) or the identity (operators). At every site the combined bond is `(x bond, y bond)`, x
fastest, so the interleaved order carries the other coordinate's bond through.
"""
function qtt_embed(g::QTTGrid; x = nothing, y = nothing)
    x === nothing && y === nothing && throw(ArgumentError("qtt_embed needs a chain"))
    isop = ndims((x === nothing ? y : x)[1]) == 4
    T = promote_type((x === nothing ? Float64 : eltype(x[1])), (y === nothing ? Float64 : eltype(y[1])))
    one_core = isop ? reshape(Matrix{T}(I, 2, 2), 1, 2, 2, 1) : ones(T, 1, 2, 1)
    cx = _site_chain(g, :x, x === nothing ? [copy(one_core) for _ in 1:g.nx] : x)
    cy = _site_chain(g, :y, y === nothing ? [copy(one_core) for _ in 1:g.ny] : y)
    L = qtt_length(g)
    out = Vector{Array{T, isop ? 4 : 3}}(undef, L)
    bx = by = 1
    for s in 1:L
        own = haskey(cx, s) ? cx[s] : cy[s]
        m = haskey(cx, s) ? by : bx                       # the passing coordinate's bond
        Im = Matrix{T}(I, m, m)
        sl, sr = size(own, 1), size(own, ndims(own))
        if isop
            c = zeros(T, sl * m, 2, 2, sr * m)
            for o in 1:2, i in 1:2
                c[:, o, i, :] .= haskey(cx, s) ? kron(Im, own[:, o, i, :]) : kron(own[:, o, i, :], Im)
            end
        else
            c = zeros(T, sl * m, size(own, 2), sr * m)
            for b in 1:size(own, 2)
                c[:, b, :] .= haskey(cx, s) ? kron(Im, own[:, b, :]) : kron(own[:, b, :], Im)
            end
        end
        out[s] = c
        haskey(cx, s) ? (bx = sr) : (by = sr)
    end
    return out
end

"""
    qtt_function(g, f; tol = 1e-14, maxdim) -> TT

The field `f(x, y)` on the whole grid by TT-SVD of its dense samples, in site order. Fixture
construction on small grids only (`L <= 22`); a production field never goes through this.
"""
function qtt_function(g::QTTGrid, f; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    L = qtt_length(g)
    L <= 22 || throw(ArgumentError("qtt_function densifies; L = $L is not a fixture size"))
    xs, ys = qtt_coords(g)
    p = qtt_site_permutation(g)
    v0 = f(xs[1], ys[1])
    A = Array{typeof(float(v0))}(undef, ntuple(_ -> 2, L))
    nx = 2^g.nx
    for c in eachindex(p)
        gi = p[c] - 1
        A[c] = f(xs[gi % nx + 1], ys[gi ÷ nx + 1])
    end
    return tt_from_dense(A; tol = tol, maxdim = maxdim)
end

"""
    qtt_separable(g, fx, fy; tol = 1e-14) -> TT

`fx(x) fy(y)` from two 1D TT-SVDs (`2^nx + 2^ny` samples; valid at production sizes). Pass
`fy = nothing` for a function of `x` alone, and vice versa.
"""
function qtt_separable(g::QTTGrid, fx, fy; tol::Real = 1e-14, maxdim::Int = typemax(Int))
    xs, ys = qtt_coords(g)
    tx = fx === nothing ? nothing : tt_from_samples(fx.(xs); tol = tol, maxdim = maxdim)
    ty = fy === nothing ? nothing : tt_from_samples(fy.(ys); tol = tol, maxdim = maxdim)
    return qtt_embed(g; x = tx, y = ty)
end

"""
    qtt_values(C, g) -> Matrix

The field on the grid, `M[ix + 1, iy + 1]`. Small grids only (tests, figures of fixtures).
"""
function qtt_values(C::AbstractVector, g::QTTGrid)
    A = tt_dense(C)
    p = qtt_site_permutation(g)
    M = zeros(eltype(A), 2^g.nx, 2^g.ny)
    for c in eachindex(p)
        M[p[c]] = A[c]
    end
    return M
end

"The value at grid point `(ix, iy)` (0-based) by one core contraction -- any size."
qtt_value(C::AbstractVector, g::QTTGrid, ix::Integer, iy::Integer) =
    tt_evaluate(C, qtt_bits(g, ix, iy))
