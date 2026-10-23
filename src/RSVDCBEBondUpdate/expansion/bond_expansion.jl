# ── the per-bond expansion, at the reference's granularity ──
# One function for every sweep: expand the bond, keep the state unchanged, and push the far
# environment through the widened tensor. The expansion is lossless; new columns carry no amplitude yet.

"""
    BondExpansionInfo

What one [`expand_bond!`](@ref) did at one bond, with field names following the reference's `Info`
struct so Julia and MATLAB traces can be read side by side.

  - `dim_in`, `dim_out`: bond dimension before and after, counting MULTIPLETS (not comparable with a
    U(1) number under SU(2)). Equal means the expansion declined.
  - `n_new`: directions admitted. `err_pre`/`err_fnl`: weight discarded (watch `err_fnl`).
"""
struct BondExpansionInfo
    bond::Int
    dir::Symbol
    dim_in::Int
    dim_out::Int
    n_new::Int
    err_pre::Float64
    err_fnl::Float64
end

"""
    expand_bond!(psi, mpo, i, dir, lch, rch; exkw...) -> (env_far, info)

Expand bond `(i, i+1)` in place, leaving the state unchanged, and return the environment on the side
the sweep is heading toward, pushed through the widened frame. `dir = :right` widens `psi[i+1]`, so
the centre ends on `i` and the returned environment is the right one at link `i+1`; `:left` mirrors it.

`lch` and `rch` are the caller's current environments, so the call costs one push and no rebuild.
"""
function expand_bond!(psi::SymMPS, mpo::MPO, i::Int, dir::Symbol, lch, rch; exkw...)
    dir in (:right, :left) || throw(ArgumentError("dir must be :right or :left, got $dir"))
    L = length(psi)
    1 <= i <= L - 1 || throw(ArgumentError("bond must be in 1:$(L - 1), got $i"))
    psi.center in (i, i + 1) || throw(ArgumentError(
        "expand_bond! needs the orthogonality centre on site $i or $(i + 1), " *
        "found it on $(psi.center)"))

    dim_in = leg_dim(psi[i], 3)
    f  = _frame_from(psi[i], psi[i + 1])
    ex = cbe_expand(f, mpo, i, lch, rch; exkw...)

    if dir === :right
        _absorb_right!(psi, i, ex.V_ex)              # psi[i+1] = V_ex, centre -> i
        env = push_right_channels(rch, mpo, psi[i + 1], i + 1)
        n_new = ex.n_new_r
    else
        _absorb_left!(psi, i, ex.U_ex)               # psi[i] = U_ex, centre -> i+1
        env = push_left_channels(lch, mpo, psi[i], i)
        n_new = ex.n_new_l
    end

    return env, BondExpansionInfo(i, dir, dim_in, leg_dim(psi[i], 3), n_new,
                                  ex.err_pre, ex.err_fnl)
end

# ── the local eigensolve the DMRG sweep needs ──
# Same Krylov space as `lanczos_expv`, reduced to the lowest Ritz pair. It adds a Ritz-residual exit
# (beta_k |c_k|): `lanczos_expv` stops only on breakdown, which wastes the full `maxiter` in DMRG.

"""
    lanczos_lowest(apply, x; maxiter=30, tol=1e-13, restol=1e-10, reorth=true)
        -> (vector, ritz_value, krylov_dim)

Lowest eigenvector of a Hermitian `apply(v) = A*v` in the Krylov space of `x`, normalised. The Ritz
value is returned too, since recomputing it costs another application. It is a variational upper
bound, because the subspace is fixed.

`reorth = true` is the default: without it the recurrence loses orthogonality silently, giving a Ritz
value below the true eigenvalue that looks like better convergence.
"""
function lanczos_lowest(apply, x; maxiter::Int = 30, tol::Float64 = 1e-13,
                        restol::Float64 = 1e-10, reorth::Bool = true)
    xc = to_concrete(ComplexF64(1.0) * x)
    beta0 = norm(xc)
    beta0 == 0 && throw(ArgumentError("lanczos_lowest: start vector has zero norm"))

    v = to_concrete((1.0 / beta0) * xc)
    basis = Any[v]
    alpha = Float64[]
    betas = Float64[]

    w = apply(v)
    a = real(tensor_inner(v, w))
    push!(alpha, a)
    w = to_concrete(w + (-a) * v)

    coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
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
        coeff, value = _reduce_krylov(LowestEigen(), alpha, betas, beta0)
        # Ritz residual: `beta_k |c_k|` bounds `||A v - theta v||`. Cheap -- both factors are
        # already computed -- and it is the ANSWER's convergence, not the basis's.
        b * abs(coeff[end]) < restol && break
    end

    out = to_concrete(coeff[1] * basis[1])
    for k in 2:length(coeff)
        out = to_concrete(out + coeff[k] * basis[k])
    end
    n = norm(out)
    n > 0 && (out = to_concrete((1.0 / n) * out))
    return out, real(value), length(alpha)
end
