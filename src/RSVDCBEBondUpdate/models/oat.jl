# ── the one-axis twisting model (arXiv:2208.10972, Fig. 2) ──
# H = (sum_l S^z_l)^2 / 2 couples every pair equally, so there is no decay or locality to exploit. Since
# (S^z)^2 = I/4 on a spin-1/2 site, H = sum_{l<l'} S^z_l S^z_l' + (L/8) I. The square is never formed,
# and the constant is returned by `oat_energy_shift`.

# ── the Hamiltonian ──────────────────────────────────────────────────────────

"""
    oat_couplings(L) -> Matrix{Float64}

`J[ℓ,ℓ'] = 1` on the strict upper triangle: the uniform all-to-all coupling of
[`oat_mpo`](@ref), in the form [`pair_mpo`](@ref) reads.

Only the strict upper triangle is filled, so no pair is counted twice and the diagonal (which
would be the `(S^z_ℓ)^2` term already absorbed into [`oat_energy_shift`](@ref)) stays zero.
"""
function oat_couplings(L::Int)
    L >= 2 || throw(ArgumentError("oat_couplings needs L >= 2, got $L"))
    J = zeros(Float64, L, L)
    for i in 1:(L - 1), j in (i + 1):L
        J[i, j] = 1.0
    end
    return J
end

"""
    oat_energy_shift(L) -> Float64

`L/8`, the c-number by which [`oat_mpo`](@ref) differs from the paper's `(Σ_ℓ S^z_ℓ)^2 / 2`.

It comes from `Σ_ℓ (S^z_ℓ)^2 = L/4 · I`, exact on spin-1/2. Being proportional to the identity
it is invisible to every expectation value and to `δE(t) = E(t) - E(0)`, so nothing in the
benchmark needs it; it is here so that a quoted ABSOLUTE energy can be put on the paper's scale.
"""
oat_energy_shift(L::Int) = L / 8

"""
    oat_mpo(L; chi=1.0) -> MPO

`H = chi * sum_{l<l'} S^z_l S^z_l'`, the one-axis twisting Hamiltonian up to the c-number
[`oat_energy_shift`](@ref). `chi = 1` is the paper's `(sum S^z)^2 / 2`.

Exact, with virtual dimension 3 independent of `L`: a constant coupling is the lambda = 1 case of the
geometric tail in [`long_range_mpo`](@ref). Available in `:none` and `:U1`, where S^z is
charge-neutral; not under `:SU2`. Under `:Z2` S^z is a charge-1 flip, so the virtual dimension is
O(L); [`pair_mpo`](@ref) builds the charged transport block and the operator stays exact.
"""
function oat_mpo(L::Int; chi::Float64 = 1.0)
    L >= 2 || throw(ArgumentError("oat_mpo needs L >= 2, got $L"))
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "oat_mpo needs a rank-2 Sz, which SU(2) does not have -- run OAT under :none " *
        "(the mode that can also hold the x-polarised state and measure S^x)"))
    # `:Z2`'s `S^z` is rank 3, which the self-loop cannot carry -- see the docstring.
    symmetry_mode() === :Z2 && return oat_mpo_pairs(L; chi = chi)
    q = local_space()
    # NO NEAREST-NEIGHBOUR TERMS: every pair, near or far, is carried by the one self-loop
    # channel at `λ = 1`. An empty `XXZChain` is what says so -- `xxz_chain(L; J=0)` would
    # instead emit two XY terms with coefficient zero, spending two virtual dimensions on
    # blocks that are identically zero.
    empty_nn = XXZChain(L, XXZTerm[], 0.0, 0.0)
    return long_range_mpo(L, empty_nn, [LongRangeTerm(q.Sz, q.Sz, chi, 1.0)])
end

"""
    oat_mpo_pairs(L; chi=1.0) -> MPO

The same operator as [`oat_mpo`](@ref), built through [`pair_mpo`](@ref) from
[`oat_couplings`](@ref): one channel per opening site, virtual dimension O(L).

It is an independent check: `oat_mpo` relies on the lambda = 1 self-loop, which no other test
exercises, while `pair_mpo` reaches the same operator by a different automaton. Use `oat_mpo` for
runs and this for verification. Under `:Z2` the two are the same construction.
"""
function oat_mpo_pairs(L::Int; chi::Float64 = 1.0)
    left, right, scale = _oat_sz_vertex()
    return pair_mpo(L, (chi * scale) * oat_couplings(L), [PairVertex(left, right, 1.0)])
end

"""
    _oat_sz_vertex() -> (left, right, scale)

The local S^z as the two halves of a pair term, plus a `scale` for normalisation.

Under `:none`/`:U1` both halves are Telum's `Sz` and `scale = 1`. Under `:Z2` S^z is the charge-1
flip, equal to 2 S^z, so `scale = 1/4`, placed on the coupling rather than the operator.
The halves must be an adjoint pair (Z and Z'): two Z halves give both legs the same arrow.
"""
function _oat_sz_vertex()
    if symmetry_mode() === :Z2
        Z = ising_local_space(:Z2).Z
        return Z, to_concrete(Z'), 0.25
    end
    Sz = local_space().Sz
    return Sz, Sz, 1.0
end

# ── the initial state and the observable: `:none` or `:Z2` ───────────────────

"""
    _require_sx_measurable(what)

Refuse `what` where <S^x_tot> is not measurable: `:none` and `:Z2` pass, `:U1` and `:SU2` do not.

Under `:U1`, S^x changes S^z_tot, so its expectation on a definite-charge SymMPS is exactly zero by
symmetry, and the local S^+/S^- are rank-3 operators that cannot be contracted. Both failures are
silent, so the guard runs once at the entry point. `:Z2` works because H is even under the global
spin flip, which is diagonal in the sigma^x basis of `z2_ising.jl`.
"""
function _require_sx_measurable(what::AbstractString)
    symmetry_mode() in (:none, :Z2) || throw(ArgumentError(
        "$what needs symmetry_mode() to be :none or :Z2, got $(symmetry_mode()). The OAT " *
        "initial state ⊗|→⟩ has weight in EVERY total-Sz sector and the observable S^x_tot " *
        "is charge-raising, so a U(1)/SU(2) run would report ⟨S^x_tot⟩ = 0 by symmetry rather " *
        "than the collapse-and-revival signal. Call set_symmetry!(:Z2) for the paper's own " *
        "symmetry, or set_symmetry!(:none) for the dense control."))
end

"""
    sx_operator() -> TLArray

The local S^x as a rank-2 operator on the `:none` dense site, eigenvalues +-1/2.

Telum's raising/lowering pair is asymmetric in sign and normalisation, so the naive (S^+ + S^-)/2
is -i S^y. The 2x2 block is therefore checked against [0 1/2; 1/2 0] before it is returned, and the
check throws if Telum's normalisation changes. Real symmetric test states cannot catch this.
"""
function sx_operator()
    _require_sx_measurable("sx_operator")
    if symmetry_mode() === :Z2
        # THE SIGMA^X BASIS PUTS S^x ON THE DIAGONAL: `X = diag(+1,-1)` is `2 S^x`, one 1x1
        # block per Z2 charge, charge-neutral and rank 2 -- exactly what `site_expval` wants.
        # Nothing here needs Telum's Sp/Sm normalisation, so nothing here can be caught by it.
        Sx = to_concrete(0.5 * ising_local_space(:Z2).X)
        # ⛔ DO NOT REACH FOR `Matrix` HERE. `_z2_diag_op` builds each sector block as
        # `reshape([v], 1, 1, 1)`, so the RMTs are 1x1x1 THREE-dimensional arrays and
        # `Matrix(R)` throws `MethodError: no method matching Matrix(::Array{Float64, 3})`
        # -- measured. Every Z2 irrep is one-dimensional, so each block holds exactly one
        # number and `only` reads it whatever the block's rank happens to be.
        blocks = collect(Sx.RMTs)
        (length(blocks) == 2 && all(length(R) == 1 for R in blocks)) || throw(ErrorException(
            "sx_operator(:Z2): expected two single-entry sector blocks, got $(length(blocks)) " *
            "of sizes $([size(R) for R in blocks]). The Z2 local space is built below Telum's " *
            "IROP layer (z2_ising.jl) -- recalibrate `_z2_diag_op` before trusting any number."))
        # Sorted, so the check does not depend on which charge sector Telum lists first.
        got = sort([only(R) for R in blocks])
        maximum(abs.(got .- [-0.5, 0.5])) < 1e-12 || throw(ErrorException(
            "sx_operator(:Z2): built S^x with eigenvalues $got, expected ±1/2. `X` is meant " *
            "to be `diag(+1,-1)` in the sigma^x basis, i.e. `2 S^x`."))
        return Sx
    end
    q = local_space(:none)
    Sx = to_concrete((1 / sqrt(2)) * (q.Sm - q.Sp))
    M = Matrix(Sx.RMTs[1])
    # `_dense_up_index` fixes which index is up; S^x is off-diagonal either way, so the check is
    # basis-order independent as written.
    want = [0.0 0.5; 0.5 0.0]
    maximum(abs.(M .- want)) < 1e-12 || throw(ErrorException(
        "sx_operator: built S^x = $M, expected $want. Telum's :none Sp/Sm normalisation has " *
        "changed -- recalibrate the (Sm - Sp)/√2 combination against the measured matrices " *
        "before trusting any OAT number."))
    return Sx
end

"""
    x_polarized_state(L) -> SymMPS

The OAT initial state, the product of |+> = (|up> + |down>)/sqrt2 on every site (an MPS with D = 1).

  * `:Z2`: [`ising_polarised_state`](@ref), the paper's own representation.
  * `:none`: the sum of the all-up and all-down `product_state`s. Their site tensors share leg
    spaces, so the sum is the x-polarised site tensor.
`:U1` and `:SU2` are refused (see [`_require_sx_measurable`](@ref)): the summands lie in different
charge sectors.
"""
function x_polarized_state(L::Int)
    _require_sx_measurable("x_polarized_state")
    L >= 1 || throw(ArgumentError("x_polarized_state needs at least one site, got $L"))
    # Under `:Z2` this state is the sigma^x basis's own vacuum: every site `|+>`, Z2 charge 0,
    # `chi = 1`, ONE sector. That is the paper's "an MPS with D = 1" exactly, and it is the
    # reason Fig. 2 could be run with a symmetry at all -- see `_require_sx_measurable`.
    symmetry_mode() === :Z2 && return ising_polarised_state(L)
    up = product_state(fill(:up, L))
    dn = product_state(fill(:down, L))
    ts = Any[to_concrete((1 / sqrt(2)) * (up[i] + dn[i])) for i in 1:L]
    return SymMPS(ts, L)
end

"""
    sx_profile(psi) -> Vector{Float64}

`⟨S^x_ℓ⟩` at every site, the `S^x` twin of [`magnetisation`](@ref).

Like `magnetisation` it walks the orthogonality centre across the chain once, so the whole
profile costs one sweep. It CANONICALISES `psi` (a gauge change, not a state change) -- pass a
`copy` if the centre must stay put.
"""
function sx_profile(psi::SymMPS)
    Sx = sx_operator()
    return [real(site_expval(psi, j, Sx)) for j in 1:length(psi)]
end

"""
    total_sx(psi) -> Float64

`S^x_tot = Σ_ℓ ⟨S^x_ℓ⟩`, THE OAT observable -- panel (a) of arXiv:2208.10972 Fig. 2.

NOT NORMALISED BY `‖ψ‖^2`, deliberately, and this matters for any fixed-rank sweep. A projected
step can lose norm, and dividing that out would hide the loss in exactly the panel whose claim
is that the collapse-and-revival amplitude is reproduced. Divide by `norm(psi)^2` at the call
site if the Rayleigh-quotient form is what is wanted, and say which one is plotted.
"""
total_sx(psi::SymMPS) = sum(sx_profile(psi))
