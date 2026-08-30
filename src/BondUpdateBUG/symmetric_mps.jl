# Symmetry-native MPS container backed by Telum TLArrays.
#
# Site tensor layout is fixed at (link_l, site, link_r) with itags
# "L,i" / "S,i" / "L,i+1", matching the kernel layout Alice's
# scheme.py::_to_kernel_layout produces.
#
# ARROW CONVENTION. Telum legs carry an arrow ('+' in, '-' out) that has no
# Python analogue. `svd` always emits '-' on the new legs of U and Vd and '+'
# on both legs of S, so the side that absorbs S gets a '+'. A rightward sweep
# therefore yields (link_r '-', link_l '+') across the new bond and a leftward
# sweep yields the mirror image. Rather than fight that with `legflip` (which
# also toggles the `dual` flag and would make bond arrows depend on sweep
# parity), the invariant maintained here is the weaker, sufficient one:
#
#     the two legs sharing a bond always carry OPPOSITE arrows.
#
# Leg *order* is what parity with Python depends on, and that is fixed.
# Every contraction below pairs legs explicitly and is arrow-agnostic, so
# nothing downstream needs to know which side happens to be '+'.

# ── local space ─────────────────────────────────────────────────────────────

# ── symmetry mode ────────────────────────────────────────────────────────────
#
# The integrator runs either symmetry-native (:U1 charge sectors) or without any
# symmetry (:none, one dense block). The mode is a module-level setting because
# the STATE and the GATES must agree on it, and both are built through
# `local_space()`; set it once, before building either. Every tensor is
# self-describing afterwards (`symm(t)`), so the kernel (frame/augment/kls/sweep)
# needs no mode flag -- it already dispatches on the tensors' own symmetry.

const _SYMMETRY = Ref{Symbol}(:U1)
const _LOCAL_SPACES = Dict{Symbol, Any}()

"The current symmetry mode, `:U1` or `:none`."
symmetry_mode() = _SYMMETRY[]

"""
    set_symmetry!(sym) -> Symbol

Select the symmetry the local space is built with: `:SU2` (total-spin multiplets),
`:U1` (charge-conserving spin sectors) or `:none` (a single dense block). Set this
before building the state and its gates; they must share a mode.

`:SU2` IS NOT JUST A FASTER `:U1`, AND THE DIFFERENCE IS ABOUT WHICH STATES EXIST.
A non-abelian MPS stores REDUCED matrix elements on total-spin multiplets, so every
state it can represent has a definite total spin. That is a strict win where the
target has one -- the even-`L` Heisenberg ground state is a singlet, and the
multiplet basis reaches it with far fewer stored numbers -- but it means a Neel or
domain-wall state is not slow to write down, it is UNWRITABLE: those have weight
across many total-`S` sectors. `product_state` therefore refuses under `:SU2`; use
[`dimer_state`](@ref) for a representable `S = 0` starting point.

Consequences worth knowing before switching a benchmark over:
  * `magnetisation` is identically zero for any `S = 0` state, so an `<Sz_j>`
    profile carries no information here -- use the energy or a bond correlator.
  * bond dimensions are counted in MULTIPLETS, so they are not comparable
    digit-for-digit with `:U1` numbers; a multiplet stands for `2S+1` states.
"""
function set_symmetry!(sym::Symbol)
    sym in (:SU2, :U1, :Z2, :none) || throw(ArgumentError(
        "symmetry must be :SU2, :U1, :Z2 or :none, got $sym"))
    _SYMMETRY[] = sym
    return sym
end

"""
    local_space(sym = symmetry_mode())

The spin-½ local space for mode `sym`, built once per mode and cached.
`SpinOptions` takes 2·S as an `Int`, so spin-½ is `SpinOptions(:U1, 1)` /
`SpinOptions(nothing, 1)`. Both branches expose `(:I, :Sp, :Sz, :Sm)`; under
`:U1` the raising/lowering operators are rank-3 (their op-leg carries the ±2
charge), under `:none` they are rank-2 plain matrices. See
`docs/telum_api_contract.md`.
"""
function local_space(sym::Symbol = symmetry_mode())
    haskey(_LOCAL_SPACES, sym) && return _LOCAL_SPACES[sym]
    # `:Z2` IS NOT A SPIN LOCAL SPACE. It is the transverse-field Ising space, in the sigma^x
    # basis, with the Z2 spin-flip parity as its charge -- a different local space with
    # different operators (`I, X, Z`, no `Sp/Sz/Sm`). Asking for it here is asking for a
    # Heisenberg-style space that does not exist under Z2, so it refuses rather than returning
    # something spin-like. See `ising_local_space` in z2_ising.jl.
    sym === :Z2 && throw(ArgumentError(
        "local_space is the spin-1/2 (Heisenberg) space and has no :Z2 form -- the TFIM has " *
        "no U(1)/SU(2) spin symmetry. Use ising_local_space() for the Z2 Ising space."))
    opts = sym === :SU2  ? SpinOptions(:SU2, 1)    :
           sym === :U1   ? SpinOptions(:U1, 1)     :
           sym === :none ? SpinOptions(nothing, 1) :
           throw(ArgumentError("symmetry must be :SU2, :U1 or :none, got $sym"))
    _LOCAL_SPACES[sym] = getLocalSpace(opts, ("s", "s", "op"))
    return _LOCAL_SPACES[sym]
end

"U(1) sector label for a single up spin (charge 2·Sz = +1)."
const SECTOR_UP = ((1,),)
"U(1) sector label for a single down spin (charge 2·Sz = -1)."
const SECTOR_DOWN = ((-1,),)

# ── container ───────────────────────────────────────────────────────────────

"""
    SymMPS(tensors, center)

A symmetric MPS. `tensors[i]` is rank-3 with legs `(link_l, site, link_r)`.
`center` is the orthogonality centre: tensors left of it are left isometries,
tensors right of it are right isometries.
"""
mutable struct SymMPS
    tensors::Vector{Any}
    center::Int
end

Base.length(psi::SymMPS) = length(psi.tensors)
Base.getindex(psi::SymMPS, i::Int) = psi.tensors[i]
Base.setindex!(psi::SymMPS, A, i::Int) = (psi.tensors[i] = A)
Base.eachindex(psi::SymMPS) = eachindex(psi.tensors)
Base.copy(psi::SymMPS) = SymMPS(copy(psi.tensors), psi.center)

"Total dimension of leg `l` of `t`, summed over its charge sectors."
leg_dim(t, l::Int) = sum(d for (_, d) in t.spaces[l]; init = 0)

"""
    bond_dims(psi) -> Vector{Int}

The `length(psi) - 1` interior bond dimensions.
"""
bond_dims(psi::SymMPS) = [leg_dim(psi[i], 3) for i in 1:(length(psi) - 1)]

"""
    norm(psi::SymMPS)

Valid in canonical form, where every tensor other than the centre is an
isometry and the state norm is the Frobenius norm of the centre tensor.
"""
LinearAlgebra.norm(psi::SymMPS) = norm(psi[psi.center])

"""
    overlap(bra::SymMPS, ket::SymMPS) -> ComplexF64

The full MPS inner product `<bra|ket>` (`bra` conjugated), contracted bond by
bond from the left. The two states must have the same length and physical
spaces; their bond dimensions may differ. `overlap(psi, psi)` equals
`norm(psi)^2`, and `overlap` is conjugate-symmetric: `overlap(a,b) ==
conj(overlap(b,a))`.

Each site pairs `bra[i]'` (adjoint) against `ket[i]`, the same arrow-consistent
contraction `tensor_inner` uses, so it is correct with OR without symmetry. A
structurally-zero overlap (no matching charge blocks, e.g. orthogonal U(1)
sectors) is exactly `0`, not an error.

NOTE: BUG-Julia does not define or export `inner`. ITensorMPS owns that name, so
in a session with `using ITensorMPS` a call like `inner(::SymMPS, ::SymMPS)`
resolves to ITensorMPS and throws a `MethodError`. Use `overlap` (exported) or
`LinearAlgebra.dot` (extended below) instead.
"""
function overlap(bra::SymMPS, ket::SymMPS)
    length(bra) == length(ket) || throw(DimensionMismatch(
        "overlap needs equal-length states, got $(length(bra)) and $(length(ket))"))
    L = length(bra)
    # One site: every leg pairs (both links are dim-1 boundaries), so this is
    # exactly the single-tensor inner product.
    L == 1 && return tensor_inner(bra[1], ket[1])
    # The transfer tensor carries (bra_link_r, ket_link_r), which inherit the SAME
    # link itag "L,i+1" -- Telum rejects a tensor with two identical indices, and
    # `contract` also requires paired legs to share an itag, so the two link
    # namespaces cannot just be renamed away. Retag only the bra's INTERNAL link
    # legs into a private "BL,i" namespace: adjacent bra tensors still match each
    # other (so the chain contracts), the bra/ket transfer legs become distinct
    # (so no duplicate index), and the two BOUNDARY links -- contracted directly
    # bra-with-ket at sites 1 and L -- keep their original tag so those pair. Site
    # legs keep "S,i" and still pair with ket.
    braR = Vector{Any}(undef, L)
    for i in 1:L
        t = bra[i]
        i > 1 && (t = setitag(t, 1, "BL,$i"))
        i < L && (t = setitag(t, 3, "BL,$(i + 1)"))
        braR[i] = to_concrete(t)
    end
    # Left boundary link (leg 1) pairs here; carry (bra_link_r, ket_link_r) right.
    E = contract(braR[1]', (1, 2), ket[1], (1, 2))
    for i in 2:(L - 1)
        tmp = contract(E, (1,), braR[i]', (1,))      # (ket_link, bra_site, bra_link_r)
        E   = contract(tmp, (1, 2), ket[i], (1, 2))  # (bra_link_r, ket_link_r)
    end
    # Last site closes ALL remaining legs -- including the dim-1 RIGHT boundary
    # link (leg 3) -- so the result is a rank-0 scalar, not a dangling rank-2.
    tmp = contract(E, (1,), braR[L]', (1,))          # (ket_link, bra_site, bra_link_r)
    # The right boundary carries each state's TOTAL charge. If bra and ket share
    # no sector there they live in different charge sectors and the overlap is
    # structurally zero -- return it rather than let `contract` assert on the
    # mismatched space (the same zero `tensor_inner` gives for no matching blocks).
    braq = Set(first(sp) for sp in braR[L].spaces[3])
    ketq = Set(first(sp) for sp in ket[L].spaces[3])
    isdisjoint(braq, ketq) && return 0.0 + 0.0im
    s = contract(tmp, (1, 2, 3), ket[L], (1, 2, 3))
    r = to_concrete(s)
    length(r) == 0 && return 0.0 + 0.0im             # `length` of a TLArray is its sector count
    return ComplexF64(r[])
end

"`dot(bra, ket)` is `overlap(bra, ket)` -- extends `LinearAlgebra.dot` so it
composes in a mixed session without the `inner` name clash (distinct types, same
generic). See `overlap`."
LinearAlgebra.dot(bra::SymMPS, ket::SymMPS) = overlap(bra, ket)

# ── construction ────────────────────────────────────────────────────────────

"""
    product_state(spins) -> SymMPS

Build a product state from `spins::Vector{Symbol}`, each `:up` or `:down`.

Each site tensor is a two-leg fusion isometry `getIdentity((prev_link), (site))`
restricted to the requested physical sector. The physical leg keeps its FULL
space (`preserve_space=true`) so local operators stay contractible against it;
the outgoing link is trimmed to the sectors actually populated, which is what
makes a product state report bond dimension 1.
"""
function product_state(spins::Vector{Symbol})
    L = length(spins)
    L >= 1 || throw(ArgumentError("product_state needs at least one site"))
    # NOT a missing feature. An `:SU2` MPS carries total-spin multiplets, and a definite-Sz
    # product state is a superposition across many total-S sectors, so there is no bond
    # dimension at which this is representable. See [`dimer_state`](@ref).
    #
    # THIS GUARD REPLACES A SILENT WRONG ANSWER, WHICH IS WHY IT IS A THROW AND NOT A WARNING.
    # Without it the U(1) path ran to completion under `:SU2` and returned `bond_dims = [1,1,1]`
    # with no error (MEASURED). The reason is a coincidence of labels: `SECTOR_UP` is `((1,),)`
    # and the SU(2) spin-1/2 irrep is ALSO `((1,),)`, so `:up` sites match the `getsub` filter
    # while `:down` sites (`((-1,),)`, a label that does not exist under SU(2)) silently select
    # nothing. The result looks like a state and is not one.
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "product_state is not representable under :SU2 -- a definite-Sz product state has " *
        "weight in many total-spin sectors. Use dimer_state(L) for an S = 0 start."))
    return symmetry_mode() === :none ? _product_state_none(spins) :
                                       _product_state_u1(spins)
end

function _product_state_u1(spins::Vector{Symbol})
    L = length(spins)
    q = local_space(:U1)

    tensors = Any[]
    # Left boundary: a one-dimensional vacuum link.
    prev = (getvac(q.I, ("L,0", "L,1")), 2)

    for i in 1:L
        target = if spins[i] === :up
            SECTOR_UP
        elseif spins[i] === :down
            SECTOR_DOWN
        else
            throw(ArgumentError("spin $i must be :up or :down, got $(spins[i])"))
        end

        # Fuse the incoming link with a fresh physical leg. getIdentity returns
        # a lazy TLArrayView and getsub only accepts a concrete TLArray.
        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))

        # Keep only the requested spin, but preserve the physical leg's space.
        A = getsub(F, 2, s -> s == target ? Colon() : nothing; preserve_space = true)

        # Trim the outgoing link to the sectors that survived.
        A = to_concrete(A)
        present = Set(ql[3] for ql in A.qlabels)
        A = to_concrete(getsub(A, 3, s -> s in present ? Colon() : nothing))

        # BUG evolves complex amplitudes; promote once, here.
        A = to_concrete(A * (1.0 + 0.0im))

        push!(tensors, A)
        prev = (A, 3)
    end

    return SymMPS(tensors, L)
end

"SU(2) irrep label for a single spin-½ site (2S = 1). MEASURED from `local_space(:SU2)`."
const SECTOR_SU2_HALF = ((1,),)
"SU(2) irrep label for a singlet (2S = 0) -- what two spin-½ multiplets fuse down to."
const SECTOR_SU2_SINGLET = ((0,),)

"""
    dimer_state(L) -> SymMPS

The product of nearest-neighbour singlets, `(1,2)(3,4)…`, under `:SU2`. `L` must be even.

WHY THIS EXISTS RATHER THAN `product_state`. Under SU(2) an MPS stores reduced matrix elements
on TOTAL-SPIN multiplets, so every representable state has a definite total spin -- and a Néel
or domain-wall state does not. `product_state` cannot express one here at any bond dimension,
which is why it refuses under `:SU2`. A product of singlets is `S = 0`, so it is representable,
and it is the natural starting point for both cooling and real-time evolution.

HOW IT IS BUILT. Same fusion recursion as `_product_state_u1`, but the selection happens on the
OUTGOING LINK rather than the physical leg -- under SU(2) the physical leg has only one sector
(`S = 1/2`), so there is nothing to select there. Fusing alternately gives

    vac (S=0) (x) 1/2  ->  1/2                    keep 1/2   (odd  sites)
    1/2       (x) 1/2  ->  0 (+) 1                keep 0     (even sites)

so after every even site the running total spin is a singlet, which is exactly a chain of
independent dimers. Every link carries ONE multiplet, so `bond_dims` reports all ones -- but a
multiplet is not a state, and this is a genuinely entangled state in the physical basis.
"""
function dimer_state(L::Int)
    # Available in EVERY mode, unlike `product_state` (which `:SU2` cannot represent) and unlike
    # the old `:SU2`-only version of this function. That matters because an SU(2)-vs-no-symmetry
    # comparison is only meaningful if both sides evolve the SAME state: the claim under test is
    # that symmetry changes the cost and not the physics. Under `:none`/`:U1` the construction is
    # by projection — see `_dimer_state_projected`.
    symmetry_mode() === :SU2 || return _dimer_state_projected(L)
    L >= 2 && iseven(L) || throw(ArgumentError(
        "dimer_state needs an even number of sites >= 2, got $L"))
    q = local_space(:SU2)

    tensors = Any[]
    prev = (getvac(q.I, ("L,0", "L,1")), 2)

    for i in 1:L
        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        # Select on leg 3, the FUSED total spin, not on the physical leg.
        target = isodd(i) ? SECTOR_SU2_HALF : SECTOR_SU2_SINGLET
        A = to_concrete(getsub(F, 3, s -> s == target ? Colon() : nothing))
        A = to_concrete(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end

    return SymMPS(tensors, L)
end

"""
    _dense_up_index(q) -> Int

Which index (1 or 2) of the `:none` dense physical sector is the up spin,
read off the sign of `Sz`'s diagonal. Avoids hard-coding the basis order that
`get_SU2_symmops` happens to produce.
"""
function _dense_up_index(q)
    Sz = Matrix(q.Sz.RMTs[1])
    return real(Sz[1, 1]) > real(Sz[2, 2]) ? 1 : 2
end

"""
    _product_state_none(spins) -> SymMPS

No-symmetry product state. The physical leg is a single dense sector of
dimension 2, so a spin is selected by INDEX on the outgoing link (`getsub`
returns an `Int`), not by charge sector. The physical leg keeps its full
2-dimensional space so local operators stay contractible.
"""
function _product_state_none(spins::Vector{Symbol})
    L = length(spins)
    q = local_space(:none)
    up = _dense_up_index(q)

    tensors = Any[]
    prev = (getvac(q.I, ("L,0", "L,1")), 2)
    for i in 1:L
        idx = if spins[i] === :up
            up
        elseif spins[i] === :down
            3 - up
        else
            throw(ArgumentError("spin $i must be :up or :down, got $(spins[i])"))
        end

        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        # One trivial sector on the outgoing link; keep the chosen spin index,
        # which trims that link to dimension 1 (product state ⇒ bond dim 1).
        A = to_concrete(getsub(F, 3, s -> idx))
        A = to_concrete(A * (1.0 + 0.0im))

        push!(tensors, A)
        prev = (A, 3)
    end

    return SymMPS(tensors, L)
end

"""
    domain_wall_state(L) -> SymMPS

`↑…↑↓…↓` with the wall in the middle. Total Sz is 0 for even `L`.
"""
domain_wall_state(L::Int) =
    product_state([i <= L ÷ 2 ? :up : :down for i in 1:L])

"""
    neel_state(L) -> SymMPS

`↑↓↑↓…`.
"""
neel_state(L::Int) =
    product_state([isodd(i) ? :up : :down for i in 1:L])

# ── canonical form ──────────────────────────────────────────────────────────

"""
    move_right!(psi, i)

Move the orthogonality centre from `i` to `i+1`. `psi[i]` becomes a left
isometry; `S·Vd` is absorbed into `psi[i+1]`.
"""
function move_right!(psi::SymMPS, i::Int)
    res = svd(psi[i], (1, 2); cutoff = 0.0)
    # U  : (L,i, S,i, svdL)      -- the left isometry
    # Vd : (svdR, L,i+1)
    M = res.S * res.Vd                       # (svdL, L,i+1)
    B = contract(M, (2,), psi[i + 1], (1,))  # (svdL, S,i+1, L,i+2)
    psi[i]     = to_concrete(setitag(res.U, 3, "L,$(i + 1)"))
    psi[i + 1] = to_concrete(setitag(B, 1, "L,$(i + 1)"))
    return psi
end

"""
    move_left!(psi, i)

Move the orthogonality centre from `i` to `i-1`. `psi[i]` becomes a right
isometry; `U·S` is absorbed into `psi[i-1]`.
"""
function move_left!(psi::SymMPS, i::Int)
    res = svd(psi[i], (1,); cutoff = 0.0)
    # U  : (L,i, svdL)
    # Vd : (svdR, S,i, L,i+1)    -- the right isometry
    M = res.U * res.S                          # (L,i, svdR)
    B = contract(psi[i - 1], (3,), M, (1,))    # (L,i-1, S,i-1, svdR)
    psi[i - 1] = to_concrete(setitag(B, 3, "L,$i"))
    psi[i]     = to_concrete(setitag(res.Vd, 1, "L,$i"))
    return psi
end

"""
    canonical!(psi, i)

Sweep the orthogonality centre to site `i`. Norm-preserving.
"""
function canonical!(psi::SymMPS, i::Int)
    1 <= i <= length(psi) || throw(BoundsError(psi, i))
    while psi.center < i
        move_right!(psi, psi.center)
        psi.center += 1
    end
    while psi.center > i
        move_left!(psi, psi.center)
        psi.center -= 1
    end
    return psi
end

# ── observables ─────────────────────────────────────────────────────────────

"""
    site_expval(psi, j, O) -> ComplexF64

`⟨ψ|O_j|ψ⟩` for a rank-2 local operator `O` with legs `(site '+', site '-')`.
Canonicalises to `j` first, so the contraction is purely local.

The operator's site leg must span the FULL local space, which is why
`product_state` builds site tensors with `preserve_space=true`.

`contract` asserts matching itags as well as opposite arrows, so the operator
is retagged onto site `j` first. Setting both its legs to the same tag is safe:
they stay distinct because `TLIndex` equality includes the arrow.
"""
function site_expval(psi::SymMPS, j::Int, O)
    canonical!(psi, j)
    A = psi[j]
    Oj = setitag(O, "S,$j")
    T = contract(Oj, (1,), A, (2,))         # (op-out, L,j, L,j+1)
    return contract(T, (1, 2, 3), A', (2, 1, 3))[]
end

"""
    sz_expectation(psi, j) -> Float64

`⟨S_z^j⟩`, in units where an up spin gives `+1/2`.
"""
sz_expectation(psi::SymMPS, j::Int) = real(site_expval(psi, j, local_space().Sz))

"""
    total_sz(psi) -> Float64

`Σ_j ⟨S_z^j⟩`.
"""
total_sz(psi::SymMPS) = sum(sz_expectation(psi, j) for j in 1:length(psi))

"""
    apply_sz!(psi, j) -> psi

`S^z_j |ψ⟩` in place, NOT renormalised, by SECTOR PROJECTION rather than by contracting the
operator onto the site leg.

⛔ THE OBVIOUS ROUTE IS WRONG AND IT FAILS SILENTLY. Contracting `local_space().Sz` onto the site
leg returns a correctly SHAPED rank-3 tensor whose site leg has had its charge sectors RE-SORTED
by `contract`, so the state it encodes is not `S^z_j|ψ⟩`. MEASURED against the dense elementwise
`S^z`: residual 2.02e-1 on a state of norm 0.5, and not a sign flip -- there is a genuinely
orthogonal component. [`site_expval`](@ref) uses that same contraction and is fine only because it
closes the loop with the SAME tensor, so the re-sort cancels; producing a NEW MPS tensor is where
it bites. Same class of bug as the augmenter's QR re-sorting bond sectors.

`S^z = ½P↑ − ½P↓`, and `getsub(...; preserve_space = true)` keeps the FULL site space on each
projection -- the flag `product_state` uses for exactly this reason -- so the two halves stay
addable and the site leg comes back with its original sector order.

Lives here rather than in a benchmark because more than one benchmark needs it
(`cbe_sweeps_l16.jl` for the `S^z|GS⟩` real-time start, `hs_structure_factor.jl` for the Li et al.
correlator) and duplicating a function whose whole point is a silent-failure trap invites exactly
one copy being fixed.
"""
function apply_sz!(psi::SymMPS, j::Int)
    # ⛔ UNDER `:SU2` THE TWO `getsub` FILTERS BELOW DEGRADE `S^z` TO `(1/2) * I` WITH NO ERROR,
    # by the SAME label coincidence documented in `product_state`: `SECTOR_UP` is `((1,),)` and so
    # is `SECTOR_SU2_HALF`, so the `up` filter selects the WHOLE physical leg, while
    # `SECTOR_DOWN = ((-1,),)` is a label SU(2) does not have and selects nothing. The result is
    # `0.5 * psi` -- a perfectly valid state, which is why nothing downstream trips.
    #
    # What that does to a structure factor: `G(x,t)` collapses to `0.25` for EVERY site and time,
    # so `G(c,0) = 0.25` still PASSES its sum rule by coincidence and only `sum_i G(i,0) = 0`
    # catches it (it would read `0.25*N`). One gate away from a published-looking flat spectrum.
    #
    # This is not a missing feature. `S^z` is the `m = 0` component of a rank-1 tensor operator,
    # not an SU(2) scalar: it takes the state out of its total-spin sector, and an `:SU2` MPS
    # stores reduced matrix elements on multiplets and has nowhere to put the result. The SU(2)
    # route is the VECTOR correlator `<S.S> = 3 x <S^z S^z>` via a Wigner-Eckart reduced matrix
    # element -- a different observable and a different code path, exactly the factor of 3 that
    # arXiv:2209.00739 Eq. (5) drops.
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "apply_sz! is not available under :SU2 -- S^z is the m=0 component of a rank-1 tensor " *
        "operator, not a scalar, so it leaves the total-spin sector an :SU2 MPS is built on. " *
        "Without this guard the sector filters silently return (1/2)*psi. Use :U1 for any S^z " *
        "observable (the S(q,w) drivers already force it), or implement the vector correlator."))
    canonical!(psi, j)
    A = psi[j]
    up = to_concrete(getsub(A, 2, s -> s == SECTOR_UP   ? Colon() : nothing;
                            preserve_space = true))
    dn = to_concrete(getsub(A, 2, s -> s == SECTOR_DOWN ? Colon() : nothing;
                            preserve_space = true))
    psi[j] = to_concrete(to_concrete(0.5 * up) + to_concrete(-0.5 * dn))
    canonical!(psi, 1)
    return psi
end

"""
    left_gram(A) -> TLArray

`A†A` traced over `(link_l, site)`, leaving the two `link_r` legs open. `A` is
a left isometry iff this is the identity.

`A' * A` will NOT do: `*` is greedy and closes the bond too, collapsing to a
rank-0 scalar (`docs/telum_api_contract.md` §7a). Priming keeps it open.
"""
left_gram(A) = contract(prime(A, 3)', (1, 2), A, (1, 2))

"""
    right_gram(A) -> TLArray

`AA†` traced over `(site, link_r)`, leaving the two `link_l` legs open.
"""
right_gram(A) = contract(prime(A, 1)', (2, 3), A, (2, 3))

"""
    left_isometry_defect(A) -> Float64

`||A (I - A'A)||_F` over the `(link_l, site)` legs: zero exactly when `A` is a
left isometry, and computed WITHOUT catastrophic cancellation.

Two tempting alternatives are both worse:

  - `norm(left_gram(A) - 1)` builds its identity from the tensor's own legs via
    `Base.:-(q, ::Number)` and fails to align them for frames that came out of
    `oplus` ("No leg in TLArray matches leg 2 ...").
  - the Frobenius expansion `||G-I||^2 = ||G||^2 - 2||A||^2 + d` needs no
    identity, but subtracts two quantities of size `d`, so its absolute error is
    `~d*eps` and after the square root the floor is `~1e-8`. A genuinely exact
    isometry measured that way reports 3.7e-8, which is indistinguishable from
    a real defect.

`A - A(A'A)` is a difference of two O(1) tensors whose true value is small, so
its norm is accurate to full precision.
"""
left_isometry_defect(A) = norm(perp_component(A, A))

"""
    right_isometry_defect(A) -> Float64

`||(I - AA') A||_F` over the `(site, link_r)` legs. Mirror of
[`left_isometry_defect`](@ref); the bond is leg 1.
"""
right_isometry_defect(A) = norm(perp_component_right(A, A))
