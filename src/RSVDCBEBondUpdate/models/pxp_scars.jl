# pxp_scars.jl: scars: the PXP model of a Rydberg blockade chain (from `mbl_scars.jl`, included right after it).

# ── SCARS: the PXP model of a Rydberg blockade chain ──────────────────────────────────────────
#
# THE AUTOMATON, virtual dimension 4 whatever `L` is. A three-site term needs one more state
# than a two-site one: the middle operator has to be remembered as well as the opening.
#
#     1 = id       nothing opened
#     2 = P placed the left projector is down, `X` is owed on the NEXT site
#     3 = X placed the flip is down, the right projector is owed on the NEXT site
#     4 = done     a complete term lies behind us
#
#          ⎡ I   P   X*   0  ⎤       * `id -> X placed` fires ONLY at site 1
#     W =  ⎢ 0   0   X    X† ⎥       † `P placed -> done` fires ONLY at site L
#          ⎢ 0   0   0    P  ⎥
#          ⎣ 0   0   0    I  ⎦
#
# NO SELF-LOOP on states 2 or 3, which is what makes the three operators land on three
# CONSECUTIVE sites rather than at any three increasing positions.
#
# ⛔ THE TWO BOUNDARY ENTRIES ARE THE MODEL, NOT A CORRECTION TO IT. `H = Σ_i P_{i-1} X_i P_{i+1}`
# with `P_0 = P_{L+1} = 1` contains `X_1 P_2` and `P_{L-1} X_L`, which are TWO-site terms and
# cannot be produced by the bulk path. Dropping them gives an open chain whose two end atoms are
# frozen out of the dynamics; the `|Z2>` revival period shifts by O(1/L) and the state stays
# perfectly plausible. They are entered as the starred/daggered blocks above and both are
# exercised by the `L = 3` case in the tests (`H = X_1 P_2 + P_1 X_2 P_3 + P_2 X_3`).

"""
    pxp_operators() -> (; I, P, X)

The local blockade operators, `:none` only.

`P = 1/2 - S^z` projects on the UNEXCITED atom (`|∘>`, spin down) and `X = 2 S^x = σ^x` is the
Rabi flip. The convention is `|•> = up`, so a `|Z2> = |•∘•∘…>` quench starts from
[`neel_state`](@ref) and `⟨n_i⟩ = ⟨S^z_i⟩ + 1/2`.

⛔⛔ `X` IS **NOT** `S^+ + S^-`, AND WRITING IT THAT WAY MAKES `H` NON-HERMITIAN. Telum's `:none`
raising/lowering operators carry a normalisation AND A SIGN that `sx_operator` records as
MEASURED: `S^x = (S^- - S^+)/√2`, with the resulting matrix asserted to be `[0 1/2; 1/2 0]`.
Solving that for the two operators gives

    S^+ = -(1/√2) |↑⟩⟨↓|        S^- = +(1/√2) |↓⟩⟨↑|

so `S^+ + S^-` is the real ANTISYMMETRIC matrix `[0 -1/√2; 1/√2 0]` -- proportional to `-i σ^y`,
hence ANTI-Hermitian. The consequence is not a wrong coefficient, it is a non-unitary
`exp(-iHt)`: MEASURED on the `L = 8` gate, `‖ψ‖` reached 3.95 (BUG) and 5.07 (TDVP2) after 20
steps of `dt = 0.05`, and the fidelity came out 0.25-0.34 against an exact 0.0076. Nothing threw;
Lanczos on a non-Hermitian generator does not fail, it lies.

⚠ THE SIGN ALONE WOULD HAVE BEEN INVISIBLE. `X -> -X` gives `H -> -H`, i.e. `exp(+iHt)`, which
is the complex conjugate evolution and leaves `|⟨Z_2|ψ(t)⟩|²` and every `⟨S^z_j⟩` unchanged. It
is the LOSS OF SYMMETRY, not the sign, that breaks unitarity -- which is why the failure had to
be found through `‖ψ‖` rather than through a wrong-looking observable.

So `X` is built from [`sx_operator`](@ref), which carries its own assertion against the measured
`[0 1/2; 1/2 0]` and throws if Telum's normalisation ever changes. `P` needs no such check: it is
a FUNCTION of `S^z`, so `P = 1/2 - S^z` annihilates the `S^z = +1/2` state by algebra whichever
dense index that happens to be -- the only thing that could break it is `S^z` having eigenvalues
`±1` instead of `±1/2`, and that is asserted below.

⛔ `:none` IS FORCED. `σ^x` does not conserve `S^z`, so there is no U(1) form of this model at
all; under `:U1` the raising and lowering operators are rank-3 (their op-leg carries the ±2
charge) and the sum is not even a well-formed sum of tensors, so the failure is loud rather than
a silent zero. It is still worth naming, because the MBL model in this same file is `:U1`-native
and a driver that sets the symmetry once for both gets one of them wrong.
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

`H = Ω Σ_{i=1}^{L} P_{i-1} X_i P_{i+1}` (with `P_0 = P_{L+1} = 1`) as an exact MPO of virtual
dimension 4 -- the PXP model of arXiv:2011.09486, the Rydberg blockade chain in the limit where
a neighbouring excitation is forbidden outright rather than merely costly.

`Ω` is the Rabi frequency and sets the time unit. The `|Z2>` quench revives with a period that is
an `O(1)` multiple of `1/Ω` and is NOT `2π/Ω` -- arXiv:2011.09486 makes exactly that point: the
observed frequency "did not coincide with the bare Rabi frequency, signalling the importance of
many-body effects".

⛔ NO PERIOD IS ASSERTED HERE, DELIBERATELY. A specific figure (`4.79/Ω` is the number usually
quoted, and it could not be confirmed from the review itself) hard-coded in a docstring becomes
the thing a plot is aligned to, and then a revival at the wrong period reads as agreement. The
period is MEASURED against the ED propagation in `benchmarks/mbl_scars.jl`, whose reference and
whose MPS arms are both graded on `|⟨Z_2|ψ(t)⟩|²` sampled at the same times -- so the period is an
output of the campaign, not an input to it.

Requires `L >= 3` (at `L = 2` there is no three-site term and the model degenerates) and the
`:none` symmetry mode.
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

`|Z_k> = |•∘…∘•∘…>`, an excitation every `k` sites; `k = 0` is the POLARISED VACUUM `|∘∘∘…>`.

⛔⛔ **THE CONTRAST STATE IS NOT OPTIONAL -- IT IS WHAT MAKES A REVIVAL A SCAR.** `|Z_2>` reviving
says nothing on its own: a revival is only evidence of weak ergodicity breaking if a GENERIC
state in the same Hamiltonian, at the same size and the same bond dimension, does NOT revive.
Reporting the `|Z_2>` fidelity alone would be a curve with no control, and it would look
identical whether or not the physics were there.

The ladder the scar literature uses (arXiv:2011.09486 and Turner et al., Nature Physics 14, 745):

  * `k = 2`  `|•∘•∘…>`   STRONG, long-lived revivals -- the scarred state.
  * `k = 3`  `|•∘∘•∘∘…>` WEAKER but still visible revivals -- an intermediate case, and the one
             that shows the effect is not a peculiarity of the period-2 pattern alone.
  * `k = 0`  `|∘∘∘…>`    the polarised state: NO revivals, fast thermalisation. THE CONTROL.

⚠ Every `k >= 2` pattern satisfies the Rydberg blockade (no two adjacent excitations), so all of
these live inside the constrained basis and are directly comparable. `k = 1` would be all-up,
which violates the blockade at every bond, and is rejected rather than silently evolved.

⚠ All are bond-dimension-1 product states, which is the whole difficulty for the integrator: it
must build every bit of rank it will ever need starting from none. That is also why this is the
right place to separate a rank-adaptive integrator from a fixed-rank one -- the scarred
trajectory stays LOW rank while the control's rank grows.
"""
function zk_state(L::Int, k::Int)
    k == 1 && throw(ArgumentError(
        "zk_state: k=1 is all-up, which violates the Rydberg blockade on every bond"))
    k < 0 && throw(ArgumentError("zk_state: k must be >= 0, got $k"))
    k == 0 && return product_state(fill(:down, L))
    return product_state([(i - 1) % k == 0 ? :up : :down for i in 1:L])
end
