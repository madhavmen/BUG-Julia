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

enable_krylov_log() = (_KRYLOV_RECORD[] = true; empty!(KRYLOV_LOG); nothing)
disable_krylov_log() = (_KRYLOV_RECORD[] = false; nothing)
get_krylov_log() = copy(KRYLOV_LOG)
_record_krylov(k::Int) = (_KRYLOV_RECORD[] && push!(KRYLOV_LOG, k); nothing)

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
    rc = to_concrete(r)
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

"Saad's a-posteriori estimate of the next basis vector's contribution: `beta_k*|[exp(tau*T_k)]_k1|`.
This is the CONVERGENCE measure. `beta` alone is a BREAKDOWN detector and does not fire on a
generic operator -- measured: identical results for thresholds from 1e-6 to 1e-13."
_tail(coeff::Vector{ComplexF64}, betak::Float64) = isempty(coeff) ? 0.0 : betak * abs(coeff[end])

"""
    _pick_tau(evalcoeff, tau_rem, betak, conv_tol) -> ComplexF64

Largest fraction of `tau_rem` this already-built basis can integrate accurately.

Bisects on the SMALL matrix only -- `evalcoeff(tau')` is `O(k^2)` from the cached
eigendecomposition -- so choosing the substep costs NO operator applications. That is the whole
economy of the scheme: the basis is built once and then asked how far it can go.
"""
function _pick_tau(evalcoeff, tau_rem::ComplexF64, betak::Float64, conv_tol::Float64)
    ok(t) = begin
        c, _ = evalcoeff(t)
        tl = _tail(c, betak)
        # representability is a hard constraint even when `conv_tol` is off: an exponent past
        # `_EXPV_SAFE_EXPONENT` destroys the small coefficients whatever the tail estimate says
        isfinite(tl) && (conv_tol <= 0 || tl <= conv_tol)
    end
    ok(tau_rem) && return tau_rem
    lo, hi = 0.0, 1.0                       # fraction of tau_rem
    for _ in 1:40
        mid = 0.5 * (lo + hi)
        ok(mid * tau_rem) ? (lo = mid) : (hi = mid)
    end
    # never return zero: a step that cannot be taken at all must still make progress, and the
    # caller's `maxiter` is then the thing to raise
    return max(lo, 1e-6) * tau_rem
end

"""
    lanczos_expv(apply, tau, x; tol=1e-13, maxiter=60, reorth=false,
                 conv_tol=0.0, substeps=1)

`exp(tau*A)*x` for a **Hermitian** action `apply(v) = A*v`.

Three-term recurrence with a breakdown-only exit, matching
`krylov.py::tensor_lanczos_expv` including its `tol=1e-13`, `krylovdim=60`
defaults. `reorth=true` additionally orthogonalises twice against the whole
basis; that is more stable but is *not* what the reference kernel does.

`conv_tol > 0` adds the genuine convergence test (Saad's estimate) on top of the breakdown exit;
`substeps != 1` splits `tau`. See the file header for why both exist and what they fix.
"""
function lanczos_expv(apply, tau::ComplexF64, x;
                      tol::Float64 = 1e-13, maxiter::Int = 60, reorth::Bool = false,
                      conv_tol::Float64 = 0.0, substeps::Int = 1)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        return xc
    end

    cur = xc
    tau_rem = tau
    logscale = 0.0 + 0.0im         # imaginary-time growth carried in log space, never overflowed
    total = 0
    nsub = 0

    while true
        nsub += 1
        b0 = norm(cur)
        b0 == 0 && break
        v = to_concrete((1.0 / b0) * cur)
        basis = Any[v]
        alpha = Float64[]
        betas = Float64[]

        w = apply(v); total += 1
        a = real(tensor_inner(v, w))
        push!(alpha, a)
        w = to_concrete(w + (-a) * v)

        # the step this basis will actually take; set once the loop ends (or converges early)
        for _ in 2:maxiter
            b = norm(w)
            b < tol && break                       # genuine breakdown: the space is exhausted
            if conv_tol > 0                        # genuine convergence: the answer stopped moving
                vals, vecs = _tridiag_eigen(alpha, betas)
                c, _ = _coeffs_from_eigen(vals, vecs, tau_rem)
                _tail(c, b) < conv_tol && break
            end
            push!(betas, b)
            v = to_concrete((1.0 / b) * w)
            push!(basis, v)
            w = apply(v); total += 1
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

        vals, vecs = _tridiag_eigen(alpha, betas)
        evalcoeff(t) = _coeffs_from_eigen(vals, vecs, ComplexF64(t))
        betak = norm(w)

        tau_s = if substeps == 1
            tau_rem                                # single shot: exactly the old behaviour
        elseif substeps > 1
            tau_rem / (substeps - nsub + 1)        # fixed, equal split of what is left
        else
            _pick_tau(evalcoeff, tau_rem, betak, conv_tol)   # adaptive, free of extra matvecs
        end

        coeff, ls = evalcoeff(tau_s)
        cur = _combine(coeff, basis, b0)
        logscale += ls                 # already a log: never round-trips through a huge scalar
        tau_rem -= tau_s
        (substeps == 1 || abs(tau_rem) <= 1e-14 * max(abs(tau), 1.0)) && break
        nsub >= 10_000 && error("expv: substep search failed to advance (tau_rem = $tau_rem)")
    end

    _record_krylov(total)
    return _apply_scale(cur, logscale)
end

"""
Largest norm a returned state may carry.

⛔ THE LIMIT IS `sqrt(floatmax)`, NOT `floatmax`, AND GETTING THAT WRONG IS WHAT MADE THE FIRST
VERSION OF THIS FILE FAIL. Downstream tensor operations form UNSCALED SUMS OF SQUARES -- inner
products, and the Gram-like products inside the splitting SVD -- so a state whose norm exceeds
`sqrt(floatmax) = 1.34e154` overflows THEM while its own entries are still perfectly finite.
(Julia's `norm` on a plain array is safe: it rescales. The tensor-level routines do not.)

MEASURED, and the evidence is a NON-MONOTONE failure, which is what identified it:

    dt      30          50      80      200
    result  1.352e-08   NaN     NaN     1.352e-08   <- works again

A bug that appears at 50 and DISAPPEARS at 200 cannot be overflow in the exponential. It was the
guard: at dt=200 the factored scale `exp(1736)` is `Inf`, so it was dropped and the state stayed
unit-norm; at dt=50 the scale `exp(434) = 1e188` is FINITE, so it was applied -- and 1e188 is
past `sqrt(floatmax)`. Predicted threshold `log(1.34e154)/|E_0| = 40.87`; MEASURED, the log grid
succeeded at `dt = 38.2` and failed at `dt = 46`.
"""
const _EXPV_MAX_NORM = 1e150

"""
    _apply_scale(cur, logscale) -> tensor

Put the factored scale back, unless doing so would push the state past [`_EXPV_MAX_NORM`].

⚠ WHEN IT IS DROPPED THE DIRECTION IS RIGHT AND THE NORM IS NOT. That is safe for IMAGINARY
time, where the norm grows like `exp(-tau*E_0)`, is not physical, and every caller renormalises.
In REAL time `logscale` is imaginary, `|sc| = 1` exactly, and this can never trigger -- so a
real-time norm, which IS physics, is never silently rescaled.
"""
function _apply_scale(cur, logscale::ComplexF64)
    sc = exp(logscale)
    (isfinite(abs(sc)) && sc != 0) || return cur
    n = norm(cur)
    (isfinite(n) && n > 0 && abs(sc) * n > _EXPV_MAX_NORM) && return cur
    return to_concrete(sc * cur)
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
    arnoldi_expv(apply, tau, x; tol=1e-13, maxiter=60, conv_tol=0.0, substeps=1)

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

`conv_tol` and `substeps` mean what they do for [`lanczos_expv`](@ref).
"""
function arnoldi_expv(apply, tau::ComplexF64, x;
                      tol::Float64 = 1e-13, maxiter::Int = 60,
                      conv_tol::Float64 = 0.0, substeps::Int = 1)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        return xc
    end

    cur = xc
    tau_rem = tau
    logscale = 0.0 + 0.0im
    total = 0
    nsub = 0

    while true
        nsub += 1
        b0 = norm(cur)
        b0 == 0 && break
        v = to_concrete((1.0 / b0) * cur)
        basis = Any[v]
        H = zeros(ComplexF64, maxiter + 1, maxiter)
        m = 0
        betak = 0.0

        for j in 1:maxiter
            w = apply(basis[j]); total += 1
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
            betak = b
            b < tol && break
            if conv_tol > 0
                c = _hess_coeffs(H[1:m, 1:m], tau_rem)[1]
                _tail(c, b) < conv_tol && break
            end
            H[j + 1, j] = b
            j == maxiter && break
            push!(basis, to_concrete((1.0 / b) * w))
        end

        Hm = H[1:m, 1:m]
        evalcoeff(t) = _hess_coeffs(Hm, ComplexF64(t))

        tau_s = if substeps == 1
            tau_rem
        elseif substeps > 1
            tau_rem / (substeps - nsub + 1)
        else
            _pick_tau(evalcoeff, tau_rem, betak, conv_tol)
        end

        coeff, ls = evalcoeff(tau_s)
        cur = _combine(coeff, basis[1:m], b0)
        logscale += ls                 # already a log: never round-trips through a huge scalar
        tau_rem -= tau_s
        (substeps == 1 || abs(tau_rem) <= 1e-14 * max(abs(tau), 1.0)) && break
        nsub >= 10_000 && error("expv: substep search failed to advance (tau_rem = $tau_rem)")
    end

    _record_krylov(total)
    return _apply_scale(cur, logscale)
end

"`exp(tau*Hm)e_1` for an upper-Hessenberg `Hm`, shifted the same way the Hermitian path is.
Returns the LOG of the factored scale, for the reason [`_coeffs_from_eigen`](@ref) gives."
function _hess_coeffs(Hm::Matrix{ComplexF64}, tau::ComplexF64)
    isempty(Hm) && return (ComplexF64[], zero(ComplexF64))
    sigma = _shift(eigvals(Hm), tau)
    E = exp(tau * (Hm - sigma * I))
    return E[:, 1], tau * sigma
end

"""
    expv(apply, tau, x; hermitian=true, maxiter=60, tol=1e-13, reorth=false,
         conv_tol=0.0, substeps=1)

`exp(tau*A)*x` where `apply(v) = A*v` is matrix-free.

`hermitian=true` uses Lanczos (three-term, Alice-faithful); `hermitian=false`
uses Arnoldi. Use `hermitian=false` for the K and L generators -- they are not
Hermitian.

`conv_tol > 0` stops on Saad's contribution estimate rather than on breakdown; `substeps = 0`
splits `tau` adaptively so a large step needs no larger `maxiter`. Both default to the historical
behaviour -- see the file header.
"""
function expv(apply, tau::ComplexF64, x;
              hermitian::Bool = true, maxiter::Int = 60, tol::Float64 = 1e-13,
              reorth::Bool = false, conv_tol::Float64 = 0.0, substeps::Int = 1)
    return hermitian ?
        lanczos_expv(apply, tau, x; tol = tol, maxiter = maxiter, reorth = reorth,
                     conv_tol = conv_tol, substeps = substeps) :
        arnoldi_expv(apply, tau, x; tol = tol, maxiter = maxiter,
                     conv_tol = conv_tol, substeps = substeps)
end

expv(apply, tau::Number, x; kwargs...) = expv(apply, ComplexF64(tau), x; kwargs...)
