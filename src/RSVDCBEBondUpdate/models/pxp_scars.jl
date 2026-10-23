# pxp_scars.jl: scars: the PXP model of a Rydberg blockade chain (from `mbl_scars.jl`, included right after it).

# ── SCARS: the PXP model of a Rydberg blockade chain ──
# Virtual dimension 4 for every L: states id, P placed, X placed, done. The two boundary entries
# (id -> X placed at site 1, P placed -> done at site L) are part of the model, since P_0 = P_{L+1} = 1,
# not a correction; dropping them freezes the end atoms.

"""
    pxp_operators() -> (; I, P, X)

The local blockade operators, `:none` only. `P = 1/2 - S^z` projects on the unexcited atom and
`X = 2 S^x` is the Rabi flip; `|•>` is spin up, so a `|Z_2>` quench starts from [`neel_state`](@ref).

`X` must come from [`sx_operator`](@ref): `S^+ + S^-` is anti-Hermitian in Telum's normalisation,
which makes `exp(-iHt)` non-unitary without any error being raised. `P` needs no such check beyond
the +-1/2 eigenvalues asserted below.
"""
function pxp_operators()
    symmetry_mode() === :none || throw(ArgumentError(
        "pxp_operators is :none only, got $(symmetry_mode()): σ^x does not conserve S^z, so " *
        "the PXP model has no U(1) or SU(2) form"))
    q = local_space(:none)
    # `S^z` must be the spin-1/2 generator, not the Pauli matrix, or `P = 1/2 - S^z` is not a
    # projector at all (`P² ≠ P`) and every blockade in the model is silently wrong.
    Mz = Matrix(q.Sz.RMTs[1])
    (maximum(abs.(sort(real.(diag(Mz))) .- [-0.5, 0.5])) < 1e-12 &&
     maximum(abs.(Mz .- Diagonal(diag(Mz)))) < 1e-12) || throw(ErrorException(
        "pxp_operators: :none S^z came back as $Mz, expected diagonal with eigenvalues ±1/2. " *
        "`P = 1/2 - S^z` is only the |∘⟩ projector for that normalisation."))
    P = to_concrete(0.5 * q.I - q.Sz)               # projector on |∘⟩ = the S^z = -1/2 state
    X = to_concrete(2.0 * sx_operator())            # σ^x = 2 S^x -- see the docstring
    MX = Matrix(X.RMTs[1])
    maximum(abs.(MX .- [0.0 1.0; 1.0 0.0])) < 1e-12 || throw(ErrorException(
        "pxp_operators: built X = $MX, expected σ^x = [0 1; 1 0]. A non-symmetric X makes the " *
        "PXP MPO non-Hermitian and `exp(-iHt)` non-unitary; ‖ψ‖ then grows without anything " *
        "throwing (measured 3.95 after 20 steps at L=8)."))
    return (; I = q.I, P = P, X = X)
end

"""
    pxp_mpo(L; Omega = 1.0) -> MPO

`H = Omega sum_{i=1}^{L} P_{i-1} X_i P_{i+1}` (with `P_0 = P_{L+1} = 1`) as an exact MPO of virtual
dimension 4: the PXP model of arXiv:2011.09486, where a neighbouring excitation is forbidden outright.

`Omega` is the Rabi frequency and sets the time unit. The `|Z_2>` revival period is an O(1) multiple
of 1/Omega and is not 2pi/Omega. No period is asserted here: it is measured against ED propagation
in `benchmarks/mbl_scars.jl`, so it is an output of the campaign, not an input.

Requires `L >= 3` (at L = 2 there is no three-site term) and the `:none` symmetry mode.
"""
function pxp_mpo(L::Int; Omega::Float64 = 1.0)
    L >= 3 || throw(ArgumentError("pxp_mpo needs at least three sites, got $L"))
    ops = pxp_operators()
    W = 4

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid  = _mpo_block(ops.I, i, tl, tr, :none)
        Pblk = _mpo_block(ops.P, i, tl, tr, :none)
        Xblk = _mpo_block(ops.X, i, tl, tr, :none)
        Znil = to_concrete(0.0 * Iid)
        mat  = Matrix{Any}(nothing, W, W)

        i < L && (mat[1, 1] = Iid)                                  # stay `id`
        i > 1 && (mat[W, W] = Iid)                                  # stay `done`
        i <= L - 1 && (mat[1, 2] = Pblk)                            # open with the left P
        i >= 2 && i <= L - 1 && (mat[2, 3] = Xblk)                  # the bulk flip
        i >= 2 && (mat[3, W] = to_concrete(Omega * Pblk))           # close with the right P
        i == 1 && (mat[1, 3] = Xblk)                                # X_1 P_2: no left P exists
        i == L && (mat[2, W] = to_concrete(Omega * Xblk))           # P_{L-1} X_L: no right P
        # `id -> done` is EMPTY BY PHYSICS -- PXP has no on-site term -- but it is the only entry
        # of column `W` at `i = 1` and of row 1 at `i = L`, and `oplus` cannot infer a zero block
        # for a row/column it has no other entry in. An explicit zero, not a gap.
        mat[1, W] = Znil

        if i == 1
            mat = reshape(mat[1, :], 1, W)
        elseif i == L
            mat = reshape(mat[:, W], W, 1)
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    z2_state(L) -> SymMPS

`|Z_2> = |•∘•∘…>`, the period-2 density wave the scar revivals are measured from.

It IS [`neel_state`](@ref) -- `|•> = up` -- and the alias exists so a PXP driver reads in the
language of the model rather than of the spin chain. Both are `:none`/`:U1` product states of
bond dimension 1, which is the entire difficulty: the integrator has to build every bit of the
rank it will ever need out of a state that has none.
"""
z2_state(L::Int) = product_state([isodd(i) ? :up : :down for i in 1:L])

"""
    zk_state(L, k) -> SymMPS

`|Z_k>` has an excitation every `k` sites; `k = 0` is the polarised vacuum.

A revival of `|Z_2>` is evidence of weak ergodicity breaking only against a control at the same size
and bond dimension. Ladder (arXiv:2011.09486): `k = 2` strong, long-lived revivals (the scarred
state); `k = 3` weaker but visible; `k = 0` no revivals, the control.

`k = 1` violates the blockade and is rejected. All are bond-dimension-1 product states, so the
integrator must build its own rank.
"""
function zk_state(L::Int, k::Int)
    k == 1 && throw(ArgumentError(
        "zk_state: k=1 is all-up, which violates the Rydberg blockade on every bond"))
    k < 0 && throw(ArgumentError("zk_state: k must be >= 0, got $k"))
    k == 0 && return product_state(fill(:down, L))
    return product_state([(i - 1) % k == 0 ? :up : :down for i in 1:L])
end
