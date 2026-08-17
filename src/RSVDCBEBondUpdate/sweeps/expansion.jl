# THE PER-BOND EXPANSION, AT THE REFERENCE'S OWN GRANULARITY.
#
# `cbe_matlab/RSVDpreBE1SiQS.m` is called from both reference sweeps with one signature,
#
#     [TLex, TRex, Vex, Info] = CBE(TL, TR, VL, VR, HL, HR, direction)
#
# and it does three things: expand the bond, hand back BOTH site tensors re-expressed so the
# STATE IS UNCHANGED, and push the environment on the far side through the tensor it just
# widened (`PushVR` / `PushVL`, `RSVDpreBE1SiQS.m:93/96`). This file is that one function, so
# every sweep in this directory expands its bonds through the SAME code path and a difference
# between two sweeps can never be a difference in how they expanded.
#
# WHERE IT IS CALLED FROM, and this is the answer to "is the shrewd selection done once or
# many times": INSIDE THE BOND LOOP OF EVERY HALF-SWEEP, at every time step or DMRG sweep.
# `TDVPSweepCBE1Si.m:30` and `:70`, `DMRGSweepCBE1Si.m:29` and `:64` -- four call sites, all
# inside `for si = ...`. It cannot be otherwise: a ONE-SITE update cannot change a bond
# dimension at all, so the expansion is the only thing that can, and a bond not expanded in a
# given step has its rank frozen for that step. For a time integrator a direction not admitted
# is not deferred -- it is gone, and the state carries the error forward.
#
# THE EXPANSION IS LOSSLESS, which is what makes it safe to do before evolving anything.
# `oplus` puts the old frame first, so `range(U_ex) ⊇ range(A_i)`, and `_absorb_*!` writes the
# widened frame into one site and the (zero-padded) remainder into its neighbour. The new
# columns carry NO amplitude yet -- and that is precisely what the update afterwards is able to
# fill. Nothing is approximated here; `err_pre`/`err_fnl` measure what the SELECTION declined
# to offer, not what the state lost.
#
# ON THE MPO PATH ONLY. The `XXZChain` channel form cannot express a long-range or
# site-dependent `H` at all (`henv.jl`'s `_left_step` refuses to carry an open channel past a
# site), and Haldane-Shastry is both. Everything here therefore takes an `MPO`.

"""
    BondExpansionInfo

What one [`expand_bond!`](@ref) did at one bond. Field names follow the reference's `Info`
struct so a Julia trace and a MATLAB trace can be read side by side:

  - `bond`, `dir` -- which bond, and which way the sweep was going.
  - `dim_in`, `dim_out` -- `MDi` and `MDf`, the bond dimension before and after. **These
    count MULTIPLETS, not states**, so under SU(2) they are not comparable with a U(1) number
    (see `state_dim`). A `dim_out == dim_in` means the expansion declined -- either the bond
    is already complete (`room == 0`) or the selection found nothing outside the frame.
  - `n_new` -- directions admitted, the honest measure of what the expansion contributed.
  - `err_pre` -- weight the preselection SVD discarded (the reference's `err_pre`).
  - `err_fnl` -- weight the final selection discarded (`err_fnl`). **This is the number to
    watch**: it is the tail of the RANKED candidate spectrum, so a large value means the
    budget, not the selection, was the accuracy limiter.
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

Expand bond `(i, i+1)` in place, leaving the state unchanged, and return the environment on
the side the sweep is heading TOWARDS -- pushed through the frame just widened, never rebuilt.

`dir = :right` widens `psi[i+1]` (the isometry being swept toward) and leaves the padded
remainder in `psi[i]`, so the orthogonality centre ends on site `i` and the returned
environment is the RIGHT one at link `i+1`. `dir = :left` is the mirror. That is the
reference's `'>>'` / `'<<'`, including which of the two environments comes back.

`lch` must be the left environment at link `i` and `rch` the right one at link `i+2`; both are
what the caller is already carrying, so this costs one `push_*_channels` and no rebuild.

The state's gauge is not touched: [`_frame_from`](@ref) needs the amplitude on exactly one
side of the bond, which is true whenever the orthogonality centre is on site `i` or `i+1`, and
both sweep directions satisfy that on arrival.
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

# ── the local eigensolve the DMRG sweep needs ────────────────────────────────
#
# `lanczos_expv` propagates; DMRG minimises. Same Krylov space, different reduction -- exactly
# the `Propagate` / `LowestEigen` split `cbe_core.jl` already draws, so the reduction is
# `_reduce_krylov(LowestEigen(), ...)` and not a second eigensolver to keep in step.
#
# ONE DELIBERATE ADDITION over `lanczos_expv`: A CONVERGENCE EXIT. `lanczos_expv` leaves on
# BREAKDOWN only, so every call burns its whole `maxiter` -- measured at 27.4 of 30 iterations
# for a solve converged long before. That is tolerable for a propagator, where the extra
# vectors are nearly free relative to the sweep, and it is not tolerable for DMRG, where a
# local eigensolve runs `2(L-1)` times per sweep for tens of sweeps. The exit used is the
# standard Ritz residual estimate `beta_k * |c_k|` -- the last Lanczos beta times the last
# component of the Ritz vector, which bounds `||H v - theta v||` -- so it stops when the
# ANSWER has converged rather than when the basis stops growing. This is what the reference's
# `(K, Ktol)` pair is (`mainDMRG_CBE.m:87`: `'K',K, 'Ktol',Ktol`).

"""
    lanczos_lowest(apply, x; maxiter=30, tol=1e-13, restol=1e-10, reorth=true)
        -> (vector, ritz_value, krylov_dim)

Lowest eigenvector of a **Hermitian** `apply(v) = A*v` in the Krylov space of `x`, normalised.

Returns the Ritz value alongside the vector because a DMRG sweep's report IS that number, and
recomputing `<v|H|v>` afterwards would cost another operator application per bond.

THE RITZ VALUE IS A VARIATIONAL UPPER BOUND on the lowest eigenvalue of `A`, because the
tridiagonal is the exact Galerkin matrix of `A` in a FIXED subspace. (That is not true of
`_expanding_krylov`, whose subspace moves under it as it iterates -- see the two-phase
argument in `modes/ground_state.jl`. Nothing here expands, so nothing here loses the bound.)

`reorth = true` by default, and for the same measured reason `tdvp1_cbe_step!` defaults it on:
without it the recurrence loses orthogonality and the failure is silent -- a Ritz value below
the true eigenvalue, which then looks like better convergence rather than a broken solve.
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
