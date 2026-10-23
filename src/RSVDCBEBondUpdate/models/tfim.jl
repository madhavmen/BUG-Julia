# ── the transverse-field Ising chain as an explicit MPO ──
# Open-chain MPO of virtual dimension 3 (id, open, done) with no self-loop on `open`, which keeps the
# coupling nearest-neighbour. The first and last sites keep only the row or column that can enter or exit.
# The basis is sigma^x (see `z2_ising.jl`): X is diagonal, Z is the Z2-odd flip of rank 3. The coupling
# is the adjoint pair Z/Z', not Z/Z, which would give both legs the same arrow.

"""
    tfim_mpo(L; J = 1.0, h = 1.0) -> MPO

Explicit MPO for `H = -J Σ Z_i Z_{i+1} - h Σ X_i` on `L` sites, virtual dimension 3.

Valid under `:Z2` (the symmetric form, where the global spin flip is the conserved charge) and
`:none` (the dense control). Both use `ising_local_space`, so the operators are the same tensors
`ising_bond_gates` builds its gates from and the two representations describe one Hamiltonian.

The sign convention is the ferromagnetic one: `J > 0` favours aligned neighbours, `h > 0` is a
field along the axis `X` measures. The critical point of the infinite chain is `h = J`.
"""
function tfim_mpo(L::Int; J::Float64 = 1.0, h::Float64 = 1.0)
    L >= 2 || throw(ArgumentError("tfim_mpo needs at least 2 sites, got $L"))
    sym = symmetry_mode()
    (sym === :Z2 || sym === :none) || throw(ArgumentError(
        "tfim_mpo is defined for :Z2 and :none, got $sym. The TFIM has no U(1) spin symmetry " *
        "(Z_i Z_{i+1} does not conserve Sz) and no SU(2) symmetry (the field picks an axis)."))
    ops = ising_local_space(sym)
    # The two halves of `Z_i Z_{i+1}`, as an ADJOINT pair -- see the arrow note above.
    Zl, Zr = ops.Z, to_concrete(ops.Z')
    W = 3

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid  = _mpo_block(ops.I, i, tl, tr, :none)
        mat  = Matrix{Any}(nothing, W, W)

        i < L && (mat[1, 1] = Iid)                                   # stay `id`
        i > 1 && (mat[W, W] = Iid)                                   # stay `done`
        # `id -> open`: place the first Z. Charged under :Z2, so its op-leg becomes w_r.
        i < L && (mat[1, 2] = _mpo_block(Zl, i, tl, tr, _side(Zl, :left)))
        # `open -> done`: the partner Z', one site later, carrying the coupling.
        i > 1 && (mat[2, W] = to_concrete(-J *
            _mpo_block(Zr, i, tl, tr, _side(Zr, :right))))
        # `id -> done`: the on-site field. Neutral, so no channel is opened at all -- this is
        # why the field costs nothing in virtual dimension.
        mat[1, W] = to_concrete(-h * _mpo_block(ops.X, i, tl, tr, :none))

        # No boundary zeroing, unlike `long_range_mpo`: mat[1,W] is the on-site field here, and blanking it
        # would silently weaken the fields at sites 1 and L.
        if i == 1
            mat = reshape(mat[1, :], 1, W)
        elseif i == L
            mat = reshape(mat[:, W], W, 1)
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end
