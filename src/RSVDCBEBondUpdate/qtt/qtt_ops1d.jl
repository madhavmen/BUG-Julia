# ── 1D quantics operators (QTT plan, Task 1.1) ──
# Matrix-free MPOs on n bits, logical order (bit 0 first), from the binary carry automaton: the
# shift `(S+ psi)_i = psi_{i+1}` carries from the least significant bit up, so its bond is the carry
# (dimension 2) and nothing of size 2^n is ever formed. Finite differences are sums of shifts.
#
# Boundary fixtures, kept separate as the plan requires:
#   :periodic   -- the shift wraps (GP)
#   :dirichlet0 -- the shift drops what leaves the box (zero outside)
#   :onesided   -- :dirichlet0 plus rank-1 patches so the first and last rows are the second-order
#                  one-sided stencils, which need no boundary data (Heston; at v = 0 the variance
#                  coefficients vanish there, leaving the degenerate first-order boundary equation)

const QTT_BCS = (:periodic, :dirichlet0, :onesided)

"""
    qtt_shift(n; bc = :periodic) -> operator cores

`(S psi)_i = psi_{i+1}` on `n` bits. `:periodic` wraps `i = 2^n - 1` onto `0`; otherwise the row
`2^n - 1` is zero. The backward shift is `op_transpose(qtt_shift(n; bc))`.
"""
function qtt_shift(n::Int; bc::Symbol = :periodic)
    n >= 1 || throw(ArgumentError("qtt_shift needs n >= 1"))
    # W[c_in + 1, o + 1, i + 1, c_out + 1] = 1 iff o + c_in = i + 2 c_out: the input bit is the
    # output bit plus the incoming carry, the carry passes up.
    W = zeros(Float64, 2, 2, 2, 2)
    for c in 0:1, o in 0:1
        t = o + c
        W[c + 1, o + 1, (t % 2) + 1, (t ÷ 2) + 1] = 1.0
    end
    cores = [copy(W) for _ in 1:n]
    cores[1] = W[2:2, :, :, :]                            # the +1 enters at bit 0
    last = cores[n]
    cores[n] = bc === :periodic ? sum(last; dims = 4) :  # the carry out of the top bit wraps ...
                                  last[:, :, :, 1:1]       # ... or leaves the box
    return cores
end

"`sum (row, col, value)` of rank-1 corrections on `n` bits, compressed."
function _qtt_patch(n::Int, entries)
    terms = [(v, op_ket_bra(n, r, c)) for (r, c, v) in entries if v != 0]
    return op_lincomb(terms)
end

"""
    qtt_derivative(n, h; order = 1, bc = :periodic) -> operator cores

Second-order finite differences: `order = 1` the central `(S+ - S-) / 2h`, `order = 2` the
three-point `(S+ - 2I + S-) / h^2`, with the boundary fixture `bc` (file header). `:onesided`
needs `n >= 2` (four points).
"""
function qtt_derivative(n::Int, h::Real; order::Int = 1, bc::Symbol = :periodic)
    bc in QTT_BCS || throw(ArgumentError("bc must be one of $QTT_BCS, got :$bc"))
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    sb = bc === :periodic ? :periodic : :dirichlet0
    Sp = qtt_shift(n; bc = sb)
    Sm = op_transpose(Sp)
    Id = op_identity(n)
    D = order == 1 ? op_lincomb([(1 / (2h), Sp), (-1 / (2h), Sm)]) :
                     op_lincomb([(1 / h^2, Sp), (-2 / h^2, Id), (1 / h^2, Sm)])
    bc === :onesided || return D
    N = 2^n
    N >= 4 || throw(ArgumentError(":onesided stencils need at least 4 points, got $N"))
    # patch = (one-sided stencil) - (the truncated central row), first and last rows
    P = order == 1 ?
        [(0, 0, -3 / (2h)), (0, 1, 3 / (2h)), (0, 2, -1 / (2h)),
         (N - 1, N - 1, 3 / (2h)), (N - 1, N - 2, -3 / (2h)), (N - 1, N - 3, 1 / (2h))] :
        [(0, 0, 4 / h^2), (0, 1, -6 / h^2), (0, 2, 4 / h^2), (0, 3, -1 / h^2),
         (N - 1, N - 1, 4 / h^2), (N - 1, N - 2, -6 / h^2), (N - 1, N - 3, 4 / h^2),
         (N - 1, N - 4, -1 / h^2)]
    return op_lincomb([(1.0, D), (1.0, _qtt_patch(n, P))])
end

"""
    qtt_diag_1d(vals; tol = 1e-14) -> operator cores

`diag(vals)` on `n = log2(length(vals))` bits, from the 1D TT-SVD of the samples (rank 2 for a
linear function, 3 for a quadratic).
"""
qtt_diag_1d(vals::AbstractVector; tol::Real = 1e-14) = op_diag(tt_from_samples(vals; tol = tol))
