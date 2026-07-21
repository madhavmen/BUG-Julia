# Matrix-free exp(tau*A)*x for Telum tensors.
#
# Ports Alice's two_site_bug/_kernel/krylov.py::tensor_lanczos_expv and its
# hermitian_tridiagonal_exp_coeffs. Both paths are needed: the K and L
# generators of the discarded-projector variant are NON-Hermitian, because the
# projector P_perp = I - U0 U0^dagger is applied before the exponential, so
# Lanczos is invalid there and Arnoldi must be used.
#
# PARITY NOTE. Alice uses the classic three-term Lanczos recurrence
#
#     w = A v_i - alpha_i v_i - beta_{i-1} v_{i-1}
#
# with `LanczosOptions(tol=1e-13, krylovdim=60)` and a breakdown-only exit
# (`if norm(w) < tol: break`). This file reproduces that exactly, including the
# defaults, because Task 16 compares the two implementations trace-by-trace. The
# plan text asked for double-orthogonalisation against all previous vectors;
# that spans the same Krylov space and yields the same tridiagonal in exact
# arithmetic, but it is NOT what the verified kernel does, so it is offered as
# opt-in `reorth=true` rather than imposed. Arnoldi always orthogonalises
# against the full basis -- that is what makes it Arnoldi.

# Opt-in Krylov-depth instrumentation, mirroring krylov.py's KRYLOV_LOG /
# enable_krylov_log / disable_krylov_log / get_krylov_log. Off by default, so
# zero overhead. Every expv call appends its Krylov dimension (the number of
# matrix-free applications it actually performed).
const KRYLOV_LOG = Int[]
const _KRYLOV_RECORD = Ref(false)

enable_krylov_log() = (_KRYLOV_RECORD[] = true; empty!(KRYLOV_LOG); nothing)
disable_krylov_log() = (_KRYLOV_RECORD[] = false; nothing)
get_krylov_log() = copy(KRYLOV_LOG)
_record_krylov(k::Int) = (_KRYLOV_RECORD[] && push!(KRYLOV_LOG, k); nothing)

"""
    tensor_inner(a, b) -> ComplexF64

The canonical inner product `<a|b>` for two tensors of identical leg structure.

Ports `krylov.py::tensor_inner`, including its structurally-zero guard: an
inner product with no matching charge blocks is **exactly zero**, not an error.
That case is reachable -- e.g. `<v|H|v>` for a purely off-diagonal (pure XX
flip-flop) `H` on an Sz-basis product state -- and must not need a diagonal
regulariser.
"""
function tensor_inner(a, b)
    n = length(a.inds)
    legs = ntuple(identity, n)
    r = contract(a', legs, b, legs)
    rc = to_concrete(r)
    length(rc) == 0 && return 0.0 + 0.0im   # `length` of a TLArray is its sector count
    return ComplexF64(rc[])
end

"Promote a tensor to ComplexF64 so real inputs can take a complex `tau`."
_as_complex(x) = to_concrete(x * (1.0 + 0.0im))

"""
    hermitian_tridiagonal_exp_coeffs(alpha, beta, tau) -> Vector{ComplexF64}

Krylov coefficients of `exp(tau*T)e_1` for the Hermitian tridiagonal `T`.
Ports `krylov.py::hermitian_tridiagonal_exp_coeffs`.
"""
function hermitian_tridiagonal_exp_coeffs(alpha::Vector{Float64},
                                          beta::Vector{Float64},
                                          tau::ComplexF64)
    isempty(alpha) && return ComplexF64[]
    T = isempty(beta) ? Matrix{Float64}(reshape(alpha, 1, 1)) :
                        Matrix(SymTridiagonal(alpha, beta))
    vals, vecs = eigen(Symmetric(T))
    weights = exp.(tau .* ComplexF64.(vals)) .* ComplexF64.(vecs[1, :])
    return ComplexF64.(vecs) * weights
end

"Recombine `out = beta0 * sum_i coeff[i] * basis[i]`."
function _combine(coeff, basis, beta0::Float64)
    out = (coeff[1] * beta0) * basis[1]
    for i in 2:length(coeff)
        out = out + (coeff[i] * beta0) * basis[i]
    end
    return to_concrete(out)
end

"""
    lanczos_expv(apply, tau, x; tol=1e-13, maxiter=60, reorth=false)

`exp(tau*A)*x` for a **Hermitian** action `apply(v) = A*v`.

Three-term recurrence with a breakdown-only exit, matching
`krylov.py::tensor_lanczos_expv` including its `tol=1e-13`, `krylovdim=60`
defaults. `reorth=true` additionally orthogonalises twice against the whole
basis; that is more stable but is *not* what the reference kernel does.
"""
function lanczos_expv(apply, tau::ComplexF64, x;
                      tol::Float64 = 1e-13, maxiter::Int = 60, reorth::Bool = false)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        return xc
    end

    v = to_concrete((1.0 / beta0) * xc)
    basis = Any[v]
    alpha = Float64[]
    betas = Float64[]

    w = apply(v)
    a = real(tensor_inner(v, w))
    push!(alpha, a)
    w = to_concrete(w + (-a) * v)

    for _ in 2:maxiter
        b = norm(w)
        b < tol && break
        push!(betas, b)
        v = to_concrete((1.0 / b) * w)
        push!(basis, v)
        w = apply(v)
        a = real(tensor_inner(v, w))
        push!(alpha, a)
        w = w + (-a) * v + (-b) * basis[end - 1]
        if reorth
            for _pass in 1:2, u in basis
                w = w - tensor_inner(u, w) * u
            end
        end
        w = to_concrete(w)
    end

    _record_krylov(length(alpha))
    coeff = hermitian_tridiagonal_exp_coeffs(alpha, betas, tau)
    return _combine(coeff, basis, beta0)
end

"""
    arnoldi_expv(apply, tau, x; tol=1e-13, maxiter=60)

`exp(tau*A)*x` for a **non-Hermitian** action, via the upper-Hessenberg Arnoldi
factorisation. This is the path the K and L generators require: the discarded
projector `P_perp = I - U0 U0^dagger` is applied before the exponential and
breaks Hermiticity, so the Lanczos assumption `alpha in R` is false.

Measured on a sector-diagonal generator with complex eigenvalues (job 93396):
Arnoldi terminates at the true Krylov dimension (2) and is exact to 4e-16,
while Lanczos on the same generator **never terminates** -- it burns the whole
`maxiter` budget with `beta` diverging (0.94, 1.6, ... 47.2) and builds a badly
conditioned tridiagonal. On that particular *normal* operator the answer still
came out right, so the failure mode to guard against is cost and conditioning,
not necessarily a wrong number; nothing guarantees the number stays right.

Each new vector is orthogonalised **twice** against the whole basis -- one pass
of modified Gram-Schmidt loses orthogonality badly once the Krylov vectors
start to align.
"""
function arnoldi_expv(apply, tau::ComplexF64, x;
                      tol::Float64 = 1e-13, maxiter::Int = 60)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        return xc
    end

    v = to_concrete((1.0 / beta0) * xc)
    basis = Any[v]
    H = zeros(ComplexF64, maxiter + 1, maxiter)
    m = 0

    for j in 1:maxiter
        w = apply(basis[j])
        for _pass in 1:2                       # double Gram-Schmidt
            for i in 1:length(basis)
                h = tensor_inner(basis[i], w)
                H[i, j] += h
                w = w - h * basis[i]
            end
        end
        w = to_concrete(w)
        m = j
        b = norm(w)
        b < tol && break
        H[j + 1, j] = b
        j == maxiter && break
        push!(basis, to_concrete((1.0 / b) * w))
    end

    _record_krylov(m)
    Hm = H[1:m, 1:m]
    coeff = exp(tau * Hm)[:, 1]
    return _combine(coeff, basis[1:m], beta0)
end

"""
    expv(apply, tau, x; hermitian=true, maxiter=60, tol=1e-13, reorth=false)

`exp(tau*A)*x` where `apply(v) = A*v` is matrix-free.

`hermitian=true` uses Lanczos (three-term, Alice-faithful); `hermitian=false`
uses Arnoldi. Use `hermitian=false` for the K and L generators -- they are not
Hermitian.
"""
function expv(apply, tau::ComplexF64, x;
              hermitian::Bool = true, maxiter::Int = 60, tol::Float64 = 1e-13,
              reorth::Bool = false)
    return hermitian ?
        lanczos_expv(apply, tau, x; tol = tol, maxiter = maxiter, reorth = reorth) :
        arnoldi_expv(apply, tau, x; tol = tol, maxiter = maxiter)
end

expv(apply, tau::Number, x; kwargs...) = expv(apply, ComplexF64(tau), x; kwargs...)
