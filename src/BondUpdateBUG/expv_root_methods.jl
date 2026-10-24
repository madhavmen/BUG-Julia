# expv_root_methods.jl: the root integrators other than the exponential and the scaled-Krylov
# helpers (from `expv.jl`, included right after it).

# ── root integrators other than `:exp` (imaginary-time plan, Task 1) ──
# `:trapezoid`, `:radau2`, `:radau3` are the rational stability functions of one implicit RK step,
# solved on the Krylov basis. The spectrum is always shifted to the extreme Ritz value: R is not
# shift-covariant, and an unshifted imaginary-time step hits R's pole. The global order is the
# smaller of root and closure.

"Root integrators [`expv`](@ref) accepts, see the block above."
const ROOT_METHODS = (:exp, :trapezoid, :radau2, :radau3, :radau2_gmres)

# Numerator / denominator coefficients of each stability function, constant term first.
# `:radau2_gmres` is the same Runge-Kutta method as `:radau2`, solved by its coupled stages.
const _ROOT_PADE = Dict(
    :trapezoid    => ([1.0, 1 / 2], [1.0, -1 / 2]),
    :radau2       => ([1.0, 1 / 3], [1.0, -2 / 3, 1 / 6]),
    :radau2_gmres => ([1.0, 1 / 3], [1.0, -2 / 3, 1 / 6]),
    :radau3       => ([1.0, 2 / 5, 1 / 20], [1.0, -3 / 5, 3 / 20, -1 / 60]))

"Classical order of a root integrator (`typemax(Int)` for the exponential)."
root_order(m::Symbol) = m === :exp ? typemax(Int) : m === :trapezoid ? 2 :
                        m in (:radau2, :radau2_gmres) ? 3 : m === :radau3 ? 5 : _bad_root(m)

_bad_root(m) = throw(ArgumentError("root_method must be one of $(ROOT_METHODS), got :$m"))

"The stability function `R(z)` of root integrator `m` (`exp(z)` for `:exp`)."
function root_stability(m::Symbol, z::Number)
    m === :exp && return exp(ComplexF64(z))
    haskey(_ROOT_PADE, m) || _bad_root(m)
    p, q = _ROOT_PADE[m]
    return evalpoly(ComplexF64(z), p) / evalpoly(ComplexF64(z), q)
end

"`R(Z) b` for a small dense `Z`: one solve with the denominator polynomial."
function _rational_apply(m::Symbol, Z::AbstractMatrix, b::AbstractVector)
    p, q = _ROOT_PADE[m]
    poly(c) = (P = c[end] * Matrix{ComplexF64}(I, size(Z)...);
               for k in (length(c) - 1):-1:1; P = P * Z + c[k] * I; end; P)
    return poly(q) \ (poly(p) * b)
end

"The shift of a rational root: the Ritz value maximising `Re(tau*lambda)` (none in real time)."
_rational_shift(vals, tau::ComplexF64) =
    (isempty(vals) || real(tau) == 0) ? zero(ComplexF64) :
    ComplexF64(vals[argmax([real(tau * ComplexF64(v)) for v in vals])])

"`R(tau*(T - sigma))e_1` for the Hermitian tridiagonal's eigendecomposition; scale not returned."
function _rational_coeffs(vals, vecs, tau::ComplexF64, m::Symbol)
    sigma = _rational_shift(vals, tau)
    w = root_stability.(m, tau .* (ComplexF64.(vals) .- sigma)) .* ComplexF64.(vecs[1, :])
    return ComplexF64.(vecs) * w, zero(ComplexF64)
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

`exp(tau*A)*x` for a **Hermitian** action `apply(v) = A*v`. Three-term recurrence with a
breakdown-only exit, matching `krylov.py::tensor_lanczos_expv` and its defaults. `reorth=true`
orthogonalises twice against the whole basis: more stable, but not what the reference kernel does.
`conv_tol > 0` adds Saad's convergence estimate; `substeps != 1` splits `tau`.
"""
function lanczos_expv(apply, tau::ComplexF64, x;
                      tol::Float64 = 1e-13, maxiter::Int = 60, reorth::Bool = false,
                      conv_tol::Float64 = 0.0, substeps::Int = 1,
                      report::Union{Nothing, Base.RefValue} = nothing,
                      root_method::Symbol = :exp)
    # GMRES needs the full orthogonalisation whatever `B` is: the shifted stage systems are
    # complex non-Hermitian even for a Hermitian generator
    root_method === :radau2_gmres && return radau2_gmres_expv(apply, tau, x; tol = tol,
        maxiter = maxiter, conv_tol = conv_tol, report = report)
    root_method in ROOT_METHODS || _bad_root(root_method)
    # An implicit root takes the WHOLE step in one stage -- that is what it is for -- so the
    # exponential's substep split does not apply to it.
    root_method === :exp || (substeps = 1)
    coeffs(vals, vecs, t) = root_method === :exp ? _coeffs_from_eigen(vals, vecs, t) :
                                                   _rational_coeffs(vals, vecs, t, root_method)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        _report!(report, 0, 0, 0.0)
        return xc
    end

    cur = xc
    tau_rem = tau
    logscale = 0.0 + 0.0im         # imaginary-time growth carried in log space, never overflowed
    total = 0
    nsub = 0
    err = 0.0

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
                c, _ = coeffs(vals, vecs, tau_rem)
                _tail(c, b) < conv_tol && break
            end
            push!(betas, b)
            v = to_concrete((1.0 / b) * w)
            push!(basis, v)
            w = apply(v); total += 1
            a = real(tensor_inner(v, w))
            push!(alpha, a)
            # Materialise `w` before the reorth loop: Telum's +/- are lazy, so a symbolic `w` grew
            # by one term per step and made the loop O(k^3 N). Modified Gram-Schmidt is sequential,
            # so this changes only when each subtraction is evaluated, not the arithmetic.
            w = to_concrete(w + (-a) * v + (-b) * basis[end - 1])
            if reorth
                # One materialisation per pass, not per basis vector. `w` stays concrete for the
                # whole pass and only the correction is lazy, so every inner product is against a
                # fixed `w`. That makes the pass classical Gram-Schmidt, applied twice (as stable as
                # MGS). The same fix is in `cbe_bug.jl`, which keeps BUG and tdvp2 on equal footing.
                for _pass in 1:2
                    corr = nothing
                    for u in basis
                        c = tensor_inner(u, w)
                        corr = corr === nothing ? c * u : corr + c * u
                    end
                    corr === nothing || (w = to_concrete(w - corr))
                end
            end
        end

        vals, vecs = _tridiag_eigen(alpha, betas)
        evalcoeff(t) = coeffs(vals, vecs, ComplexF64(t))
        betak = norm(w)

        tau_s = if substeps == 1
            tau_rem                                # single shot: exactly the old behaviour
        elseif substeps > 1
            tau_rem / (substeps - nsub + 1)        # fixed, equal split of what is left
        else
            _pick_tau(evalcoeff, tau_rem, betak, conv_tol)   # adaptive, free of extra matvecs
        end

        coeff, ls = evalcoeff(tau_s)
        err += _tail(coeff, betak) * b0 / beta0
        cur = _combine(coeff, basis, b0)
        logscale += ls                 # already a log: never round-trips through a huge scalar
        tau_rem -= tau_s
        (substeps == 1 || abs(tau_rem) <= 1e-14 * max(abs(tau), 1.0)) && break
        nsub >= 10_000 && error("expv: substep search failed to advance (tau_rem = $tau_rem)")
    end

    _record_krylov(total)
    _report!(report, total, nsub, err)
    return _apply_scale(cur, logscale)
end

"""
    _report!(report, iterations, substeps, err)

Hand a solve's own account back to a caller that passed `report = Ref{Any}()`:
`(iterations, substeps, err)`. `err` is Saad's a-posteriori estimate summed over the substeps,
relative to `||x||` -- exact bookkeeping in REAL time, where every substep's propagator is
unitary so the local errors cannot grow; in imaginary time the shift makes it a per-substep
relative figure. It is what `tol_expv` is checked against (plan §4, UT15).
"""
_report!(::Nothing, ::Int, ::Int, ::Float64) = nothing
_report!(r::Base.RefValue, total::Int, nsub::Int, err::Float64) =
    (r[] = (iterations = total, substeps = nsub, err = err); nothing)

"""
Largest norm a returned state may carry. The limit is `sqrt(floatmax)`, not `floatmax`: tensor-level
routines form unscaled sums of squares (inner products, the Gram products in the splitting SVD),
which overflow while entries are still finite. Julia's plain-array `norm` rescales and is safe. The
constant (1e150) sits below that limit.
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
    _cgs2!(H, j, basis, w) -> w

`w` orthogonalised against `basis` by classical Gram-Schmidt applied twice, the coefficients added
to column `j` of `H`. One materialisation per pass, as in `lanczos_expv`: a symbolic `w` would make
each inner product force a sum one term longer, so `w` stays concrete for the whole pass and only
`corr` is lazy.
"""
function _cgs2!(H::AbstractMatrix{ComplexF64}, j::Int, basis, w)
    for _pass in 1:2
        corr = nothing
        for i in 1:length(basis)
            h = tensor_inner(basis[i], w)
            H[i, j] += h
            corr = corr === nothing ? h * basis[i] : corr + h * basis[i]
        end
        corr === nothing || (w = to_concrete(w - corr))
    end
    return w
end

"""
    arnoldi_expv(apply, tau, x; tol=1e-13, maxiter=60, conv_tol=0.0, substeps=1)

`exp(tau*A)*x` for a **non-Hermitian** action, via the upper-Hessenberg Arnoldi factorisation. This
is the path for the K and L generators, where `P_perp = I - U0 U0'` breaks Hermiticity and Lanczos
is invalid. Each vector is orthogonalised twice against the full basis, since one MGS pass loses
orthogonality once the Krylov vectors align. `conv_tol` and `substeps` are as for
[`lanczos_expv`](@ref).
"""
function arnoldi_expv(apply, tau::ComplexF64, x;
                      tol::Float64 = 1e-13, maxiter::Int = 60,
                      conv_tol::Float64 = 0.0, substeps::Int = 1,
                      report::Union{Nothing, Base.RefValue} = nothing,
                      root_method::Symbol = :exp)
    root_method === :radau2_gmres && return radau2_gmres_expv(apply, tau, x; tol = tol,
        maxiter = maxiter, conv_tol = conv_tol, report = report)
    root_method in ROOT_METHODS || _bad_root(root_method)
    root_method === :exp || (substeps = 1)     # an implicit root takes the whole step at once
    hcoeffs(Hm, t) = root_method === :exp ? _hess_coeffs(Hm, t) :
                                            _hess_rational(Hm, t, root_method)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        _report!(report, 0, 0, 0.0)
        return xc
    end

    cur = xc
    tau_rem = tau
    logscale = 0.0 + 0.0im
    total = 0
    nsub = 0
    err = 0.0

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
            w = _cgs2!(H, j, basis, apply(basis[j])); total += 1
            m = j
            b = norm(w)
            betak = b
            b < tol && break
            if conv_tol > 0
                c = hcoeffs(H[1:m, 1:m], tau_rem)[1]
                _tail(c, b) < conv_tol && break
            end
            H[j + 1, j] = b
            j == maxiter && break
            push!(basis, to_concrete((1.0 / b) * w))
        end

        Hm = H[1:m, 1:m]
        evalcoeff(t) = hcoeffs(Hm, ComplexF64(t))

        tau_s = if substeps == 1
            tau_rem
        elseif substeps > 1
            tau_rem / (substeps - nsub + 1)
        else
            _pick_tau(evalcoeff, tau_rem, betak, conv_tol)
        end

        coeff, ls = evalcoeff(tau_s)
        err += _tail(coeff, betak) * b0 / beta0
        cur = _combine(coeff, basis[1:m], b0)
        logscale += ls                 # already a log: never round-trips through a huge scalar
        tau_rem -= tau_s
        (substeps == 1 || abs(tau_rem) <= 1e-14 * max(abs(tau), 1.0)) && break
        nsub >= 10_000 && error("expv: substep search failed to advance (tau_rem = $tau_rem)")
    end

    _record_krylov(total)
    _report!(report, total, nsub, err)
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

"`R(tau*(Hm - sigma))e_1` on the Arnoldi Hessenberg, `sigma` the eigenvalue maximising
`Re(tau*lambda)` (see the root-integrator block); scale not returned."
function _hess_rational(Hm::Matrix{ComplexF64}, tau::ComplexF64, m::Symbol)
    isempty(Hm) && return (ComplexF64[], zero(ComplexF64))
    sigma = _rational_shift(eigvals(Hm), tau)
    e1 = zeros(ComplexF64, size(Hm, 1)); e1[1] = 1
    return _rational_apply(m, tau * (Hm - sigma * I), e1), zero(ComplexF64)
end

# ── `:radau2_gmres`: Radau IIA(2) on its coupled stages, matrix-free GMRES (dissipative plan §1.2) ──
# The stages solve `(I - tau A ⊗ B) Y = 1 ⊗ x`. Under `A = P Λ P^{-1}` they decouple into the
# shifted systems `(I - tau λ_k B) Z_k = d_k x`, `d = P^{-1} 1`, `Y = (P ⊗ I) Z`. A Krylov space is
# shift invariant and both right-hand sides are multiples of `x`, so ONE Arnoldi basis of `B` is
# recycled for both stages: an iteration costs one action, not the coupled system's two. The
# method is stiffly accurate, so the end state is `Y_2`. The spectral shift is `:radau2`'s
# (`_rational_shift`, none in real time): converged, both give the same rational step.

const _RADAU2_A = [5 / 12 -1 / 12; 3 / 4 1 / 4]
const _RADAU2_EIG = let F = eigen(ComplexF64.(_RADAU2_A))
    (F.values, F.vectors, inv(F.vectors))
end
"Bound on the restarted GMRES cycles of one Radau solve; missing `conv_tol` after it is reported."
const _RADAU_MAXCYCLE = 4

"""
    _shifted_gmres(apply, r, tau, zs, ds, Wt, sigma, target, maxiter, tol)
        -> (Zs, res, nmv, sigma)

One GMRES cycle for the systems `(I - z_k (B - sigma)) Z_k = d_k r` on one Arnoldi basis of `B`
from `r`. Each `Z_k` minimises its own residual `g_k`; the cycle stops when the stage residual they
make, `‖Σ_k Wt[:, k] ⊗ g_k‖` (`Wt` the columns of `P` they enter with), reaches `target`, at
breakdown (`tol`) or at `maxiter`. `sigma = nothing` takes `:radau2`'s shift from the current Ritz
values; a restart passes the first cycle's, so every cycle solves the same system.
"""
function _shifted_gmres(apply, r, tau::ComplexF64, zs, ds, Wt, sigma, target::Float64,
                        maxiter::Int, tol::Float64)
    beta = norm(r)
    basis = Any[to_concrete((1.0 / beta) * r)]
    H = zeros(ComplexF64, maxiter + 1, maxiter)
    ys, res, sig, nmv = Vector{Vector{ComplexF64}}(), Inf, zero(ComplexF64), 0
    for j in 1:maxiter
        w = _cgs2!(H, j, basis, apply(basis[j])); nmv += 1
        b = norm(w)
        H[j + 1, j] = b
        Hb = H[1:(j + 1), 1:j]
        Ib = Matrix{ComplexF64}(I, j + 1, j)
        sig = sigma === nothing ? _rational_shift(eigvals(H[1:j, 1:j]), tau) : sigma
        e1 = zeros(ComplexF64, j + 1); e1[1] = beta
        G = zeros(ComplexF64, j + 1, length(zs))
        empty!(ys)
        for (k, z) in pairs(zs)
            M = Ib - z * (Hb - sig * Ib)
            y = M \ (ds[k] * e1)                     # the small least-squares problem
            push!(ys, y)
            G[:, k] = ds[k] * e1 - M * y
        end
        res = norm(G * transpose(Wt))
        (b < tol || res <= target || j == maxiter) && break
        push!(basis, to_concrete((1.0 / b) * w))
    end
    return ([_combine(y, basis[1:length(y)], 1.0) for y in ys], res, nmv, sig)
end

"""
    radau2_gmres_expv(apply, tau, x; tol = 1e-13, maxiter = 60, conv_tol = 0.0, report = nothing)

One Radau IIA(2) step (order 3, L-stable) of `dx/dt = B x` over `tau`, the coupled stages solved
matrix-free by GMRES (block above) to the relative stage residual `conv_tol` (the plan's
`tol_root`; `0` runs one cycle to breakdown or `maxiter`), restarted at most `_RADAU_MAXCYCLE`
times. `report` receives `(iterations, substeps = 1, err, cycles, stages)`: `err` is the TRUE
relative stage residual `‖1 ⊗ x - (I - tau A ⊗ (B - sigma)) Y‖ / ‖x‖`, recomputed with two actions
rather than read off the recurrence, and `stages = (Y_1, Y_2)`. As for `:radau2` the shift's scale
is not returned.
"""
function radau2_gmres_expv(apply, tau::ComplexF64, x; tol::Float64 = 1e-13, maxiter::Int = 60,
                           conv_tol::Float64 = 0.0,
                           report::Union{Nothing, Base.RefValue} = nothing)
    xc = _as_complex(x)
    beta0 = norm(xc)
    if beta0 == 0
        _record_krylov(0)
        _report!(report, 0, 0, 0.0)
        return xc
    end
    lam, P, Pinv = _RADAU2_EIG
    zs = tau .* lam
    target = conv_tol > 0 ? conv_tol * beta0 : 0.0
    Zs, _, nmv, sigma = _shifted_gmres(apply, xc, tau, zs, Pinv * ones(ComplexF64, 2), P,
                                       nothing, target, maxiter, tol)
    stage(Zs, i) = to_concrete(P[i, 1] * Zs[1] + P[i, 2] * Zs[2])
    Y = [stage(Zs, 1), stage(Zs, 2)]
    function residual(Y)
        BY = [(w = apply(y); iszero(sigma) ? w : to_concrete(w - sigma * y)) for y in Y]
        nmv += 2
        R = [to_concrete(xc - Y[i] + tau * (_RADAU2_A[i, 1] * BY[1] + _RADAU2_A[i, 2] * BY[2]))
             for i in 1:2]
        return R, sqrt(norm(R[1])^2 + norm(R[2])^2)
    end
    R, res = residual(Y)
    cycles = 1
    while target > 0 && res > target && cycles < _RADAU_MAXCYCLE
        # Restart on the true residual: decoupled again, but the two right-hand sides are no longer
        # parallel, so each stage system gets its own basis.
        cycles += 1
        dZ = Any[]
        for k in 1:2
            rho = to_concrete(Pinv[k, 1] * R[1] + Pinv[k, 2] * R[2])
            if norm(rho) == 0
                push!(dZ, rho)
                continue
            end
            Zk, _, n, _ = _shifted_gmres(apply, rho, tau, zs[k:k], ComplexF64[1], P[:, k:k],
                                         sigma, target / sqrt(2), maxiter, tol)
            nmv += n
            push!(dZ, Zk[1])
        end
        Y = [to_concrete(Y[i] + stage(dZ, i)) for i in 1:2]
        R, res = residual(Y)
    end
    _record_krylov(nmv)
    report === nothing || (report[] = (iterations = nmv, substeps = 1, err = res / beta0,
                                       cycles = cycles, stages = (Y[1], Y[2])))
    return Y[2]
end

"""
    expv(apply, tau, x; hermitian=true, maxiter=60, tol=1e-13, reorth=false,
         conv_tol=0.0, substeps=1)

`exp(tau*A)*x` where `apply(v) = A*v` is matrix-free. `hermitian=true` uses Lanczos, `false` Arnoldi
(needed for the non-Hermitian K and L generators). `conv_tol > 0` stops on Saad's estimate;
`substeps = 0` splits `tau` adaptively. `report = Ref{Any}()` receives `(iterations, substeps, err)`
([`_report!`](@ref)). `root_method` (`:exp` default) can also be `:trapezoid`, `:radau2` or
`:radau3`, which apply that implicit stability function in one stage and ignore `substeps`.
"""
function expv(apply, tau::ComplexF64, x;
              hermitian::Bool = true, maxiter::Int = 60, tol::Float64 = 1e-13,
              reorth::Bool = false, conv_tol::Float64 = 0.0, substeps::Int = 1,
              report::Union{Nothing, Base.RefValue} = nothing,
              root_method::Symbol = :exp)
    return hermitian ?
        lanczos_expv(apply, tau, x; tol = tol, maxiter = maxiter, reorth = reorth,
                     conv_tol = conv_tol, substeps = substeps, report = report,
                     root_method = root_method) :
        arnoldi_expv(apply, tau, x; tol = tol, maxiter = maxiter,
                     conv_tol = conv_tol, substeps = substeps, report = report,
                     root_method = root_method)
end

expv(apply, tau::Number, x; kwargs...) = expv(apply, ComplexF64(tau), x; kwargs...)
