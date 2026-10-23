# Matrix-free exp(tau*A)*x for Telum tensors, ported from Alice's tensor_lanczos_expv for
# trace-by-trace parity. Arnoldi serves the non-Hermitian K and L generators, where P_perp breaks
# Hermiticity. The breakdown-only exit is not a convergence test: a generic A never breaks down, so
# a large tau silently truncates the series at maxiter. `conv_tol` (Saad's estimate) and `substeps`
# (split tau, which also keeps imaginary-time exponents finite) fix it.

# Opt-in Krylov-depth instrumentation, mirroring krylov.py's KRYLOV_LOG /
# enable_krylov_log / disable_krylov_log / get_krylov_log. Off by default, so
# zero overhead. Every expv call appends its Krylov dimension (the number of
# matrix-free applications it actually performed).
const KRYLOV_LOG = Int[]
const _KRYLOV_RECORD = Ref(false)
# The parallel K/L/S step (plan Task 6) runs three `expv` solves at once; a racy `push!` would
# silently drop entries rather than fail.
const _KRYLOV_LOCK = ReentrantLock()

enable_krylov_log() = (_KRYLOV_RECORD[] = true; empty!(KRYLOV_LOG); nothing)
disable_krylov_log() = (_KRYLOV_RECORD[] = false; nothing)
get_krylov_log() = copy(KRYLOV_LOG)
_record_krylov(k::Int) =
    (_KRYLOV_RECORD[] && lock(() -> push!(KRYLOV_LOG, k), _KRYLOV_LOCK); nothing)

"""
Largest `|Re(tau*lambda)|` tolerated inside one substep's matrix exponential.

`exp(60)` is `1e26` -- large, but 280 orders of magnitude clear of the Float64 ceiling, so the
small coefficients keep their relative accuracy. This is the target the adaptive substep search
aims at when `conv_tol` is off and the only thing bounding the step is representability.
"""
const _EXPV_SAFE_EXPONENT = 60.0

"Exponent past which the spectral shift switches on. Below it the shift is exactly `0`, so short
steps reproduce the unshifted arithmetic bit for bit and nothing already measured moves."
const _EXPV_SHIFT_TRIGGER = 30.0

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
    # `to_concrete!`, not `to_concrete`: `r` is a fresh contract result nothing else sees, so a copy
    # would be waste. This is the hottest allocation site, since re-orthogonalised Lanczos calls it
    # O(maxiter^2) times per solve.
    rc = to_concrete!(r)
    length(rc) == 0 && return 0.0 + 0.0im   # `length` of a TLArray is its sector count
    return ComplexF64(rc[])
end

"Promote a tensor to ComplexF64 so real inputs can take a complex `tau`."
_as_complex(x) = to_concrete(x * (1.0 + 0.0im))

"""
    _shift(vals, tau) -> Float64

The spectral shift that keeps `exp(tau*(lambda - sigma))` representable: the eigenvalue whose
exponent `Re(tau*lambda)` is largest, so every shifted exponent is `<= 0`.

Returns `0.0` while the largest exponent is below [`_EXPV_SHIFT_TRIGGER`], so ordinary short
steps are untouched and stay bit-identical to the unshifted code.
"""
function _shift(vals::AbstractVector, tau::ComplexF64)
    isempty(vals) && return 0.0
    # an explicit vector, not a generator: `argmax` over a generator has no well-defined index
    ex = [real(tau * ComplexF64(v)) for v in vals]
    k = argmax(ex)
    return ex[k] > _EXPV_SHIFT_TRIGGER ? real(vals[k]) : 0.0
end

"""
    hermitian_tridiagonal_exp_coeffs(alpha, beta, tau) -> Vector{ComplexF64}

Krylov coefficients of `exp(tau*T)e_1` for the Hermitian tridiagonal `T`.
Ports `krylov.py::hermitian_tridiagonal_exp_coeffs`.
"""
function hermitian_tridiagonal_exp_coeffs(alpha::Vector{Float64},
                                          beta::Vector{Float64},
                                          tau::ComplexF64)
    isempty(alpha) && return ComplexF64[]
    vals, vecs = _tridiag_eigen(alpha, beta)
    c, ls = _coeffs_from_eigen(vals, vecs, tau)
    # this entry point's contract is the FULL coefficients, so the factored scale goes back in
    # here; an unrepresentable one is inherent to asking for them, and the substepping path
    # exists precisely so the integrators never do
    return c .* exp(ls)
end

"Eigendecomposition of the leading tridiagonal, done ONCE so the substep search can evaluate
`exp(tau'*T)e_1` for any trial `tau'` in `O(k^2)` instead of re-exponentiating the matrix."
function _tridiag_eigen(alpha::Vector{Float64}, beta::Vector{Float64})
    T = isempty(beta) ? Matrix{Float64}(reshape(alpha, 1, 1)) :
                        Matrix(SymTridiagonal(alpha, beta))
    return eigen(Symmetric(T))
end

"""
    _coeffs_from_eigen(vals, vecs, tau) -> (coeff, logscale)

`exp(tau*T)e_1` split into a SHIFTED coefficient vector and the LOG of the scalar that was
factored out, `logscale = tau*sigma`. Keeping them apart is what lets the caller carry a huge
imaginary-time scale in log space rather than overflowing it into the vector.

⛔ THE LOG IS RETURNED, NOT THE SCALAR, and that is the whole point. Returning `exp(tau*sigma)`
for the caller to `log` again would overflow to `Inf` FIRST -- at exactly the large `tau` the
shift exists to survive -- and `log(Inf)` is `Inf`, so the accumulated scale would be destroyed
by the very round trip meant to preserve it.
"""
function _coeffs_from_eigen(vals, vecs, tau::ComplexF64)
    sigma = _shift(vals, tau)
    w = exp.(tau .* (ComplexF64.(vals) .- sigma)) .* ComplexF64.(vecs[1, :])
    return ComplexF64.(vecs) * w, tau * sigma
end
