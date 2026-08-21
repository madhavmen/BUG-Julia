# THE ONE-AXIS TWISTING MODEL of arXiv:2208.10972 Fig. 2 -- the paper's SECOND benchmark, and
# the one that stresses a real-space MPS hardest, because every pair of sites is coupled with
# the SAME strength. There is no decay to exploit and no locality to exploit.
#
#     H_OAT = ( Σ_ℓ S^z_ℓ )^2 / 2                                      (the paper's form)
#
# WHAT IT ACTUALLY IS AS AN MPO, and why the square is never formed. Expand it, and use the fact
# that a spin-1/2 site has `(S^z)^2 = I/4` EXACTLY:
#
#     ( Σ_ℓ S^z_ℓ )^2  =  Σ_ℓ (S^z_ℓ)^2  +  2 Σ_{ℓ<ℓ'} S^z_ℓ S^z_ℓ'
#                      =  L/4 · I        +  2 Σ_{ℓ<ℓ'} S^z_ℓ S^z_ℓ'
#
#     ⇒   H_OAT  =  Σ_{ℓ<ℓ'} S^z_ℓ S^z_ℓ'  +  (L/8) · I
#

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

`H = chi · Σ_{ℓ<ℓ'} S^z_ℓ S^z_ℓ'` -- the one-axis twisting Hamiltonian of arXiv:2208.10972,
up to the c-number [`oat_energy_shift`](@ref). `chi = 1` is the paper's `(Σ S^z)^2 / 2`.

VIRTUAL DIMENSION 3, INDEPENDENT OF `L`, and EXACT: a constant coupling is the `λ = 1`
degenerate case of the geometric tail that [`long_range_mpo`](@ref) carries on a single
self-loop channel, so there is no fit and no truncation anywhere in the operator. See the header
of this file for the expansion of the square, and [`oat_mpo_pairs`](@ref) for the independent
`O(L)` construction the tests pin this against.

Available in `:none` and `:U1` -- `S^z` is charge-neutral (rank 2) in both, which is exactly
what the self-loop needs. It is NOT available under `:SU2`, which has no `S^z` on its own; and
note that a `:U1` run cannot measure the paper's observable even though it can hold this
operator (see [`x_polarized_state`](@ref)).

⛔ UNDER `:Z2` THE VIRTUAL DIMENSION IS `O(L)`, NOT 3, and the reason is structural rather than
an oversight. The paper's Z2 lives in the sigma^x basis, where `S^z` is the CHARGE-1 FLIP and
therefore rank 3; a self-loop carrying it would need a charged transport identity, which
[`long_range_mpo`](@ref) refuses by an explicit check. [`pair_mpo`](@ref) builds exactly that
charged transport block, so `:Z2` is routed there and the operator is still EXACT -- it just
costs `O(L)` virtual dimension. Note that this makes `oat_mpo` and [`oat_mpo_pairs`](@ref) the
SAME construction under `:Z2`, so comparing them there is a tautology, not a cross-check.
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

The SAME operator as [`oat_mpo`](@ref), built through [`pair_mpo`](@ref) from
[`oat_couplings`](@ref): one channel per opening site, so virtual dimension `O(L)`.

Deliberately redundant, and the redundancy is the point. `oat_mpo` leans on the `λ = 1`
self-loop, the DEGENERATE end of `long_range_mpo`'s geometric tail -- `λ = 0` is pinned against
the nearest-neighbour builder in `test_mpo.jl` and intermediate `λ` against a dense pair sum in
`test_long_range.jl`, but `λ = 1` (a channel that never decays, open across the whole chain) is
exercised by nothing else. `pair_mpo` reaches the same operator by a completely different
automaton, so agreement between the two is a real check rather than a restatement.

Use `oat_mpo` for anything that runs; use this to check it -- EXCEPT under `:Z2`, where
`oat_mpo` is this function (see its docstring) and the two are not independent.
"""
function oat_mpo_pairs(L::Int; chi::Float64 = 1.0)
    left, right, scale = _oat_sz_vertex()
    return pair_mpo(L, (chi * scale) * oat_couplings(L), [PairVertex(left, right, 1.0)])
end

"""
    _oat_sz_vertex() -> (left, right, scale)

The current mode's local `S^z` as the two HALVES of a pair term plus a `scale` carrying whatever
normalisation the operators themselves do not.

Under `:none`/`:U1` both halves are Telum's rank-2 `Sz` (eigenvalues `±1/2`) and `scale = 1`.

Under `:Z2` the sigma^x basis makes `S^z` the CHARGE-1 FLIP `Z = [0 1; 1 0]`, which is `2 S^z`
and rank 3 -- its op leg carries the charge. So `scale = 1/4`, one factor of `1/2` per end. THE
SCALE GOES ON THE COUPLING, NOT THE OPERATOR: rescaling a rank-3 tensor would have to preserve
its op-leg sector bookkeeping, and a number on `J[i,j]` does the same job.

⛔ THE TWO HALVES ARE AN ADJOINT PAIR `Z`/`Z'`, NOT `Z`/`Z`, AND THE DIFFERENCE IS NOT COSMETIC.
`XXZTerm`'s own contract is documented as needing `Sp`/`Sp'` rather than `Sp`/`Sm` (see
`henv.jl`), and `ising_bond_gate` builds `Z⊗Z` the same way -- `contract(Z, (3,), Z', (3,))`.
The reason shows up as an ARROW, not a charge: an op leg is created with direction `'-'`, so
`Z`/`Z` gives the opening block's outgoing virtual leg and the closing block's incoming one the
SAME direction, and Telum refuses to contract two `'-'` legs. MEASURED, before the fix:

    AssertionError: Contracted legs must have opposite arrow directions:
                    q1 leg 4 has dir='-', q2 leg 1 has dir='-'

That failure is not specific to long range -- `Z`/`Z` cannot contract at distance 1 either -- so
it surfaced as `_charged_transport` finding no valid transport block, which reads like a missing
mod-2 capability and is actually a malformed vertex. Charge-wise mod 2 is genuinely unusual here
(`1 + 1 = 0`, so the flip IS its own partner, where U(1) needs `+2` and `-2`); that is exactly
what makes the arrow the only thing left to get wrong.
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

Refuse `what` in a mode where `⟨S^x_tot⟩` is not measurable, naming the physics rather than the
implementation limit. `:none` and `:Z2` pass; `:U1` and `:SU2` do not.

⛔ THIS GUARD REPLACES A SILENT ZERO. `S^x = (S^+ + S^-)/2` changes `S^z_tot` by ±1, so on a
`SymMPS` of definite total charge `⟨S^x_tot⟩` is not approximately zero, it is EXACTLY zero by
symmetry -- and `site_expval` cannot even be reached to notice, because under `:U1` the local
`S^+`/`S^-` are rank-3 (their op-leg carries the charge) and are not rank-2 operators at all.
Both failures are quiet in the wrong way: one returns a plausible flat line, the other throws
somewhere in `contract` about leg counts. OAT's whole signal is `S^x_tot(t)`, so this is checked
once, at the entry point, with the reason attached.

WHY `:Z2` IS THE ONE THAT WORKS, and it is the symmetry arXiv:2208.10972 Fig. 2 declares
("computed using δ=0.01 and Z2 spin symmetry"). The conserved quantity is the GLOBAL SPIN FLIP
`P = Π_ℓ 2 S^x_ℓ`, not `S^z_tot`: `H = (Σ S^z)^2/2` is even in `S^z`, so `P H P = H`. In the
sigma^x basis of `z2_ising.jl` that flip is diagonal, which turns the whole difficulty inside
out -- `S^x` becomes the DIAGONAL, charge-neutral operator (so `site_expval` reads it directly)
and `⊗|→⟩` becomes a single charge-0 sector at `chi = 1`. Under `:U1` the same state has weight
in every sector at once and cannot be a `SymMPS` of definite charge at all.
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

The local `S^x` as a rank-2 operator on the `:none` dense site, eigenvalues `±1/2`.

MEASURED, NOT ASSUMED, AND THEN ASSERTED. Telum normalises the `:none` raising/lowering pair
ASYMMETRICALLY -- measured, `Sp = -(1/√2) S^+_std` and `Sm = +(1/√2) S^-_std`, note the sign on
one and not the other -- so `S^x = (S^+ + S^-)/2 = (Sm - Sp)/√2` and the plausible-looking
`(Sp + Sm)/√2` is the operator `(S^- - S^+)/2`, which is `-i S^y`. That mistake is invisible in
any test whose state is real and symmetric under spin flip, and it is exactly wrong for this
benchmark. So the 2x2 block is built and then CHECKED against `[0 1/2; 1/2 0]` before it is
returned; if Telum's normalisation ever changes, this throws instead of quietly measuring `S^y`.
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

`|Ψ(0)⟩ = ⊗_ℓ |→⟩_ℓ`, with `|→⟩ = (|↑⟩ + |↓⟩)/√2`: the OAT initial state of arXiv:2208.10972,
"an MPS with `D = 1`", normalised.

TWO MODES, TWO CONSTRUCTIONS, ONE STATE.

Under `:Z2` it is [`ising_polarised_state`](@ref) -- all sites `|+>` in the sigma^x basis, Z2
charge 0, a single sector at `chi = 1`. This is the paper's own representation.

Under `:none` it is a SUM OF TWO `product_state`s rather than a new fusion recursion. The
physical leg is ONE dense sector of dimension 2 and `_product_state_none` trims each outgoing
link to the single selected spin index, so `product_state(all :up)` and
`product_state(all :down)` produce site tensors with IDENTICAL leg spaces (every link dim 1,
one trivial sector). Their sum is therefore well defined tensor by tensor, and
`(A↑ + A↓)/√2` is exactly the site tensor of `⊗|→⟩`. Reusing the validated builder this way
means the physical leg still carries its full 2-dimensional space, which is what
[`site_expval`](@ref) needs to contract a local operator against.

⛔ `:U1`/`:SU2` ARE REFUSED FOR PHYSICS REASONS, NOT CONVENIENCE -- see
[`_require_sx_measurable`](@ref). Under `:U1` the two summands live in different total-charge
sectors and their sum is not a `SymMPS` of definite charge at all.
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
