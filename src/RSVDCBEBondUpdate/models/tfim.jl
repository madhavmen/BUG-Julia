# THE TRANSVERSE-FIELD ISING CHAIN, as an explicit MPO.
#
#     H = -J Σ_i Z_i Z_{i+1}  -  h Σ_i X_i
#
# WHY AN MPO AND NOT THE EXISTING GATES. `ising_bond_gates` already builds this Hamiltonian as
# one gate per bond, but every sweep in `sweeps/` -- `cbe_bug_step!`, `tdvp_cbe1s_step!`,
# `tdvp2_step!` -- takes an `MPO` and drives its environments through `henv.jl`. Gates cannot be
# used there. Building a genuine MPO means the environment machinery, the one-site and two-site
# effective Hamiltonians and the Krylov solves are all shared with every other model, so a TFIM
# run differs from an OAT run only in the operator.
#
# THE AUTOMATON, and why the virtual dimension is 3 regardless of `L`. Reading left to right, a
# column of the block matrix is a state:
#
#     1 = id     nothing has been opened yet
#     2 = open   a `Z` has been placed and its partner is owed on the NEXT site
#     3 = done   the term has been completed (or the field was taken) and only `I` follows
#
#        ⎡ I      Z      -h·X ⎤        row = incoming state, column = outgoing
#   W  = ⎢ 0      0      -J·Z ⎥        `open` must close immediately, so there is no
#        ⎣ 0      0       I   ⎦        self-loop on state 2 -- that is what makes this
#                                       nearest-neighbour rather than long-ranged.
#
# The first site keeps only row 1 (nothing can be `open` or `done` on entry) and the last only
# column 3 (nothing may still be owed on exit), which is the same boundary trimming
# `long_range_mpo` does.
#
# ⛔ THE BASIS IS σˣ, SO `X` IS DIAGONAL AND `Z` IS THE FLIP -- THE OPPOSITE OF THE USUAL
# CONVENTION, and reading it the usual way silently builds a different Hamiltonian.
# `z2_ising.jl` works in the eigenbasis of the ℤ₂ charge, where the conserved global spin flip
# is diagonal. In that basis:
#
#     X  is `_z2_diag_op(1.0, -1.0)`  -- DIAGONAL, charge-neutral, rank 2  -> the field term
#     Z  is `_z2_flip_op()`           -- the FLIP, charge 1, rank 3        -> the coupling
#
# So the ℤ₂-odd operator is `Z`, and `Z_i Z_{i+1}` is charge-neutral overall (1 + 1 = 0 mod 2)
# while each factor alone is charged. That is exactly the CHARGED CHANNEL case: the `id -> open`
# block carries a rank-3 operator whose op-leg becomes the right virtual leg, and the
# `open -> done` block carries its partner with the op-leg on the left. `_side` picks that per
# operator, so the same code is correct under `:none`, where both are rank 2 and no channel is
# charged.
#
# ⛔ THE TWO HALVES OF THE COUPLING ARE THE ADJOINT PAIR `Z`/`Z'`, NOT `Z`/`Z`. Mod 2 the flip is
# its own charge partner (1 + 1 = 0), so `Z Z` looks like the obvious way to write `Z_i Z_{i+1}`
# and is charge-correct -- the failure is an ARROW, not a charge. An op-leg is created with
# direction `'-'`, so `Z`/`Z` gives the OPENING block's outgoing virtual leg and the CLOSING
# block's incoming one the same direction, and Telum will not contract two `'-'` legs. It does
# not surface as an arrow error either: `oplus` fails first, with
#     "entry 4 has no leg matching reference leg 1"
# which reads like a missing mod-2 capability and is actually a malformed vertex.
# `_oat_sz_vertex` (models/oat.jl:122) documents the identical trap for `(Σ S^z)²`, and
# `ising_bond_gate` builds its `Z⊗Z` the same way -- `contract(Z, (3,), Z', (3,))`. Under `:none`
# both are the rank-2 real symmetric `Z`, where `Z' == Z` and the adjoint costs nothing.
#
# ⛔ THERE IS NO U(1) OR SU(2) FORM. `Z_i Z_{i+1}` does not conserve `S^z`, and the transverse
# field singles out an axis, so `ising_local_space` throws for anything but `:Z2` and `:none`.
# That is the reason this model is the ℤ₂ example and the XX chain is the U(1) one.

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

        # ⛔ NO BOUNDARY ZEROING HERE, unlike `long_range_mpo`. That builder blanks `mat[1,W]` at
        # both ends ("cannot be done before anything ran" / "nothing may stay id past the end")
        # because for it that entry is unused. Here `mat[1,W]` IS the on-site field, so blanking
        # it would drop the field on sites 1 and L -- a chain with weakened boundary fields, which
        # is a different Hamiltonian that still looks entirely plausible and whose error shrinks
        # with L, so small-size tests would miss it. Row 1 at `i=1` is [I, Z, -h·X] and column W
        # at `i=L` is [-h·X, -J·Z, I]; every entry of both is a transition that must survive.
        if i == 1
            mat = reshape(mat[1, :], 1, W)
        elseif i == L
            mat = reshape(mat[:, W], W, 1)
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end
