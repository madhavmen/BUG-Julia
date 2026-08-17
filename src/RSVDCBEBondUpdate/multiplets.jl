# MULTIPLETS VERSUS STATES, and why every rank number in this module needs a unit.
#
# MEASURED, and this is the one real semantic gap the non-abelian audit turned up:
#
#   * `leg_dim` sums the per-sector dimensions, so under SU(2) it counts MULTIPLETS.
#   * Telum's `Nkeep` counts multiplets too -- `svd(T, (1,2); Nkeep = 2)` on a tensor whose bond
#     carries `[(S=1/2, mult 2), (S=3/2, mult 1)]` returns a bond of `[(S=1/2, mult 2)]`, i.e.
#     TWO multiplets and FOUR states.
#   * Telum's singular-value weights, on the other hand, ARE physical: `kept_list` rows carry a
#     `degeneracy` field and `sum(sigma^2 * degeneracy) == norm(T)^2` exactly (measured: 1.0
#     against 0.4999 for the unweighted sum). So `_trunc_weight` -- which does multiply by
#     `degeneracy` -- is correct, and the truncation ORDERING is correct too: keeping a multiplet
#     costs `dim` states and gains `sigma^2 * dim` weight, so gain per state is `sigma^2` and
#     ranking by `sigma` is exactly optimal.
#
# THE CONSEQUENCE, stated plainly because it is easy to get wrong in a benchmark: `maxdim = 16`
# is NOT the same truncation under `:U1` and under `:SU2`. The physics agrees between symmetry
# modes at FULL RANK to machine precision (measured: `tdvp2` 2.3e-14, `cbe_lubich_sweep`
# 3.1e-15, `jan_bug_step!` 6.7e-15, and a variational ground energy agreeing to 8.9e-16 -- and
# that last one matters most, because unlike a conserved energy it is a non-trivial number). But
# a TRUNCATED comparison at equal `maxdim` compares two different truncations, and the SU(2) run
# is the more generous one. Convert with `state_dim` before quoting an error-at-fixed-rank across
# symmetries.
#
# Nothing here is a fix, because nothing here is broken. It is the accounting that makes a
# cross-symmetry claim checkable.

"""
    state_dim(t, leg) -> Int

Physical dimension of leg `leg` of `t`: each sector's multiplet count weighted by its irrep
dimension. Equals `leg_dim(t, leg)` for an abelian symmetry (every irrep is one-dimensional) and
exceeds it under SU(2).

`leg_dim` is the number the code's budgets, `Nkeep` and reported bond dimensions are in;
`state_dim` is the number a U(1) or unsymmetric run would report for the same physical state.
Quote whichever the comparison needs -- but quote which one.
"""
function state_dim(t, leg::Int)
    S = symm(t)
    n = length(S)
    tot = 0
    for (q, d) in t.spaces[leg]
        w = 1
        for k in 1:n
            w *= dimension(S[k], q[k])
        end
        tot += d * w
    end
    return tot
end

"""
    state_bond_dims(psi) -> Vector{Int}

The interior bond dimensions in STATES rather than multiplets -- the SU(2) analogue of
`bond_dims(psi)`, and the quantity comparable with a U(1) run of the same physics.

Under `:none` and `:U1` this is `bond_dims(psi)` exactly, so it is safe to call unconditionally
and is the right thing to log when a script may run in more than one mode.
"""
state_bond_dims(psi::SymMPS) = [state_dim(psi[i], 3) for i in 1:(length(psi) - 1)]

"""
    multiplet_ratio(psi) -> Float64

Mean states per multiplet over the interior bonds: `1.0` under an abelian symmetry, and how much
smaller an SU(2) tensor is than the U(1) tensor carrying the same physics.

The honest way to state the SU(2) saving, and the factor by which a `maxdim` in multiplets is
more generous than the same number in states.
"""
function multiplet_ratio(psi::SymMPS)
    m = bond_dims(psi)
    s = state_bond_dims(psi)
    isempty(m) && return 1.0
    tot = sum(m)
    return tot == 0 ? 1.0 : sum(s) / tot
end
