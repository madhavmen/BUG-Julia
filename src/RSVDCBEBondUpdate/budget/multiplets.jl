# ── MULTIPLETS VERSUS STATES ──
# Under SU(2) `leg_dim` and `Nkeep` count multiplets, so `maxdim` cuts differently than under `:U1`.
# Weights are physical (`sum sigma^2 * degeneracy == norm^2`); convert via `state_dim` to compare.

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
