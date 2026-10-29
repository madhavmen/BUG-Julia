# tests/common/qtt_fixtures.jl -- independent references for the QTT plan's tests.
#
# Nothing here calls the QTT operator builders: the stencils are written out row by row on the grid
# index (ix, iy), x fastest, as SPARSE matrices, so a test compares two constructions that share no
# code. Dense site-basis objects are small-grid only.

using LinearAlgebra, SparseArrays, Random

"1D finite-difference matrix on `N` points, written row by row (no shifts, no QTT)."
function _ref_d(N::Int, h::Real, order::Int, bc::Symbol)
    D = zeros(N, N)
    st = order == 1 ? ((1, 1 / (2h)), (-1, -1 / (2h))) :
                      ((1, 1 / h^2), (0, -2 / h^2), (-1, 1 / h^2))
    for i in 1:N, (off, c) in st
        j = i + off
        if bc === :periodic
            j = mod1(j, N)
        elseif !(1 <= j <= N)
            continue
        end
        D[i, j] += c
    end
    if bc === :onesided
        D[1, :] .= 0
        D[N, :] .= 0
        if order == 1
            D[1, 1:3] .= [-3, 4, -1] ./ (2h)
            D[N, (N - 2):N] .= [1, -4, 3] ./ (2h)
        else
            D[1, 1:4] .= [2, -5, 4, -1] ./ h^2
            D[N, (N - 3):N] .= [-1, 4, -5, 2] ./ h^2
        end
    end
    return sparse(D)
end

"The 2D grid matrix `Ax (x) Ay` on the index `ix + 1 + Nx iy` (`nothing` = identity)."
function _ref2(g, Ax, Ay)
    Nx, Ny = 2^g.nx, 2^g.ny
    X = Ax === nothing ? sparse(1.0I, Nx, Nx) : Ax
    Y = Ay === nothing ? sparse(1.0I, Ny, Ny) : Ay
    return kron(Y, X)
end

"A site-basis dense operator re-indexed onto the grid index."
function _grid_matrix(g, M)
    p = qtt_site_permutation(g)
    G = zeros(eltype(M), length(p), length(p))
    G[p, p] .= M
    return G
end

"A site-basis dense vector re-indexed onto the grid index."
function _grid_vector(g, v)
    p = qtt_site_permutation(g)
    w = zeros(eltype(v), length(p))
    w[p] .= vec(v)
    return w
end

_relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)

"A random TT with bond ranks `ranks` (length `L - 1`), complex unless `T` says otherwise."
function _random_tt(ranks::AbstractVector{<:Integer}; rng = MersenneTwister(1), T = ComplexF64)
    r = [1; collect(ranks); 1]
    return [randn(rng, T, r[k], 2, r[k + 1]) ./ sqrt(2 * r[k]) for k in 1:(length(r) - 1)]
end

"Generic ranks `min(chi, 2^k, 2^(L-k))`."
_tt_ranks(L, chi) = [min(chi, 2^k, 2^(L - k)) for k in 1:(L - 1)]

"""
`H Theta` at bond `(i, i+1)` projected by the state's own canonical frames, from the DENSE operator
`Hs` (site basis): the reference for `apply_h_two_site`. `psi` must be canonical at `i`.
"""
function _dense_eff(psi, Hs, i)
    C = qtt_cores(psi)
    L = length(C)
    Lm = ones(ComplexF64, 1, 1)
    for k in 1:(i - 1)
        Lm = reshape(Lm * reshape(C[k], size(C[k], 1), :), :, size(C[k], 3))
    end
    Rm = ones(ComplexF64, 1, 1)
    for k in L:-1:(i + 2)
        c = C[k]
        Rm = reshape(reshape(c, size(c, 1) * 2, size(c, 3)) * Rm, size(c, 1), :)
    end
    A, B = C[i], C[i + 1]
    Th = reshape(reshape(A, size(A, 1) * 2, size(A, 3)) * reshape(B, size(B, 1), :),
                 size(A, 1), 2, 2, size(B, 3))
    X = reshape(Lm * reshape(Th, size(Th, 1), :), :, size(Th, 4)) * Rm
    Hv = Hs * vec(X)
    Pl, Pr = size(Lm, 1), size(Rm, 2)
    Y = Lm' * reshape(Hv, Pl, 4 * Pr)
    Y = reshape(reshape(Y, size(Lm, 2) * 4, Pr) * Rm', size(Lm, 2), 2, 2, size(Rm, 1))
    return Y
end

"The local DD matrix data of a bond: `Xm`, the frame matrices, `P_L^D`, `P_R^D`, `A_DD`."
function _dd_dense(f, mpo, i, lenv, renv)
    Xd = RC._none_block(apply_h_two_site(frame_theta(f), mpo, i, lenv, renv))
    cl, cr = size(Xd, 1), size(Xd, 4)
    Xm = reshape(Xd, 2cl, 2cr)
    Um = reshape(RC._none_block(f.U0), 2cl, :)
    Vm = reshape(RC._none_block(f.V0), :, 2cr)
    PL = I - Um * Um'
    PR = I - Vm' * Vm
    return (; Xm, Um, Vm, PL, PR, A = PL * Xm * PR)
end

"The QTT bond fixture: grid `g`, state cores `C` (gauged to `i`), MPO, frame and environments."
function _qtt_bond(g, W, C, i)
    psi = qtt_mps(C; center = i)
    mpo = qtt_to_mpo(W)
    f = bond_frame(psi, i)
    lenv, renv = _envs(psi, mpo, i, qtt_length(g))
    return (; psi, mpo, f, lenv, renv)
end

"A Heston parameter set in the Feller regime with a strong negative correlation."
const _HP = HestonParams(0.03, 0.01, 2.0, 0.09, 0.4, -0.7)
"Its grid: log-price `x` in [-1, 1], variance in [0, 0.5], end points on the grid."
_heston_grid(nx, ny; kw...) = qtt_grid(nx, ny; x = (-1.0, 1.0), y = (0.0, 0.5), periodic_x = false,
                                       periodic_y = false, kw...)
