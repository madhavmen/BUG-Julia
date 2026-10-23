# tests/common/krylov_closure_fixtures.jl -- the dense closure loop, fresh environments and
# nesting defect of the Krylov-closure tests.

"""
The production closure loop on a DENSE space: `Q` holds orthonormal columns, the frontier is
`Q (tau^j/j!) (Q'AQ)^j Q'psi`, and a sweep appends the normalised outside-space action when its
time-weighted size exceeds `allow` -- the dense analogue of the MPS frontier sweep.
"""
function _kc_dense_closure(A, psi, Q0, tau, allow, order; include_zero = true,
                           maxpass = RC._closure_maxpass(order))
    Q = Ref(Matrix{ComplexF64}(Q0))
    s0() = Q[]' * psi
    applyB = x -> Q[]' * (A * (Q[] * x))
    frontier(j) = j == 0 ? s0() : RC._projected_frontiers(applyB, s0(), tau, j)[j]
    function sweep!(F, j, pass)
        Av = A * (Q[] * F)
        r = Av - Q[] * (Q[]' * Av)
        leak = abs(tau) * norm(r) / norm(psi)
        added = 0
        if leak > allow
            for _ in 1:2
                r -= Q[] * (Q[]' * r)
            end
            Q[] = hcat(Q[], r / norm(r))
            added = 1
        end
        return (added = added, leak = leak, met = true)
    end
    res = RC._closure_loop!(frontier, sweep!, order; maxpass = maxpass,
                            include_zero = include_zero)
    return Q[], res
end

_kc_unit(n, k) = (v = zeros(ComplexF64, n); v[k] = 1; v)

"The expansion settings of a step (`cbe_bug_step!`'s `exkw`), for driving the closure directly."
_kc_ckw(alloc, wcap; exact = true) =
    (growth = RC.ADAPTIVE_GROWTH, exact = exact, alloc = alloc, workcap = wcap)

"Fresh root environments through the frames' CURRENT tensors (what `fr.lenv/renv` must equal)."
function _kc_fresh_envs(mpo, fr, L)
    lenv = boundary_channels(mpo)
    for i in 1:(fr.c - 1)
        lenv = push_left_channels(lenv, mpo, fr.W[i], i)
    end
    renv = boundary_channels(mpo)
    for j in (L - 1):-1:(fr.c + 1)
        renv = push_right_channels(renv, mpo, fr.Z[j], j + 1)
    end
    return lenv, renv
end

"Is every old frame chain contained in the new one, bond by bond? Max violation of ||O||^2 = r_old."
function _kc_nesting_defect(Wold, Wnew, Zold, Znew, c, L)
    worst = 0.0
    O = nothing
    for i in 1:c
        T = O === nothing ? Wold[i] : to_concrete(contract(O, (2,), Wold[i], (1,)))
        O = to_concrete(contract(Wnew[i]', (1, 2), T, (1, 2)))           # (new_i, old_i)
        worst = max(worst, abs(norm(O)^2 - leg_dim(Wold[i], 3)))
    end
    O = nothing
    for j in (L - 1):-1:c
        T = O === nothing ? Zold[j] : to_concrete(contract(Zold[j], (3,), O, (1,)))
        O = to_concrete(contract(T, (2, 3), Znew[j]', (2, 3)))            # (old_j, new_j)
        worst = max(worst, abs(norm(O)^2 - leg_dim(Zold[j], 1)))
    end
    return worst
end

_kc_magnon(L, k) = product_state([i == k ? :up : :down for i in 1:L])
