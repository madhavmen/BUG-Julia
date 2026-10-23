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
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# ⛔ THE BREAKDOWN-ONLY EXIT IS NOT A CONVERGENCE TEST, AND ON A LARGE `tau` IT SILENTLY
# TRUNCATES THE SERIES.
#
# `beta < tol` reports that the Krylov space has run out of independent DIRECTIONS. On a generic
# `A` that essentially never happens, so every solve runs to `maxiter` and then returns whatever
# the `maxiter`-dimensional projection gives -- CONVERGED OR NOT, with no diagnostic either way.
# For small `tau` that is merely wasteful (measured elsewhere: `maxiter = 8` was 3.64x faster at
# 2e-13 on a model where the default 30 was being burned in full). For LARGE `tau` it is wrong:
# the depth a Krylov exponential needs grows with `tau*||A||`, so past some step size the fixed
# cap stops being enough and the error appears with no warning.
#
# MEASURED, L=20 SU(2) Heisenberg, imaginary time, depth needed for ~1e-14 in the HALF-SWEEP
# basis (`benchmarks/imaginary_time/krylov_depth_scan.jl`):
#
#     dt      0.1     1       5
#     m       3       8       16
#
# against a default cap of 30 -- so the cap is comfortable at dt=1 and is being approached by
# dt=5. A logarithmic time grid spans three orders of magnitude of `dt` WITHIN ONE RUN, which is
# exactly the setting in which one fixed `m` cannot be right everywhere.
#
# TWO FIXES, and they compose:
#
#   `conv_tol`   Saad's a-posteriori estimate `beta_k * |[exp(tau*T_k)]_{k,1}|` -- the actual
#                contribution the next basis vector would make. Stops when the answer has
#                converged instead of when the basis degenerates.
#   `substeps`   SPLIT `tau` ITSELF. `exp(tau*A) = exp(tau_s*A)^s`, so a step too large for
#                `maxiter` vectors becomes `s` steps that each fit. Cost then grows LINEARLY in
#                `tau` instead of demanding `m ~ tau*||A||`, and -- the point -- the caller never
#                has to guess `m`. `substeps = 0` picks `s` adaptively from the Krylov spectrum
#                that has already been built, so it costs no extra operator applications.
#
# ⛔ AND SUBSTEPPING IS WHAT MAKES THE EXPONENTIAL REPRESENTABLE AT ALL. In IMAGINARY time
# `exp(tau*lambda)` with `tau = -50` and `lambda ~ -20` is `exp(1000)`, well past the Float64
# ceiling of `exp(709)`: the small components of the coefficient vector are then annihilated by
# the large ones long before anything reports an overflow. Bounding `|tau_s|*rho` per substep and
# shifting the spectrum (below) keeps every exponent near zero. In REAL time `tau = -i*dt` makes
# `exp(tau*lambda)` a pure PHASE, so this failure is imaginary-time-only -- which is why it went
# unseen until the log-grid runs.
#
# DEFAULTS ARE THE OLD BEHAVIOUR (`conv_tol = 0.0`, `substeps = 1`) so nothing already measured
# moves under this file; the shift likewise switches on only once an exponent is genuinely large.
# ─────────────────────────────────────────────────────────────────────────────────────────────

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
    # ⛔ `to_concrete!`, NOT `to_concrete`. `_eager_tlarray(q::TLArray) = copy(q)`, so the
    # copying form DEEP-COPIES the tensor just to read ONE SCALAR out of it. `r` is a fresh
    # `contract` result aliasing neither argument, so nobody else can observe the mutation.
    #
    # This is the hottest allocation site in the whole package. Allocation attribution at
    # chi=1024 (job 16333509) put it FIRST for every arm: 39.9% of tdvp2's sampled bytes and
    # 20.1% of cbe1s's. That is because re-orthogonalised Lanczos calls it O(maxiter^2) times
    # per solve -- two passes over the whole basis every iteration -- against ~4(L-1) solves
    # per step, so it runs tens of thousands of times where a matvec runs hundreds.
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
