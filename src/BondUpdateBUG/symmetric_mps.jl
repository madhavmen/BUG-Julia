# Symmetry-native MPS container backed by Telum TLArrays. Site tensors are (link_l, site, link_r)
# with itags "L,i" / "S,i" / "L,i+1", the layout of Alice's scheme.py::_to_kernel_layout. Only the
# invariant that the two legs of a bond carry opposite arrows is kept, since svd's arrows depend on
# sweep direction. Contractions pair legs explicitly, so no code assumes which side is '+'.

# ── local space ─────────────────────────────────────────────────────────────

# ── symmetry mode ──
# Set once, before building the state and its gates: both go through `local_space()` and must match.
# Tensors carry their own symmetry (`symm(t)`), so the kernel needs no mode flag.

const _SYMMETRY = Ref{Symbol}(:U1)
const _LOCAL_SPACES = Dict{Symbol, Any}()

"The current symmetry mode, `:U1` or `:none`."
symmetry_mode() = _SYMMETRY[]

"""
    set_symmetry!(sym) -> Symbol

Select the local-space symmetry: `:SU2` (total-spin multiplets), `:U1` (charge-conserving spin
sectors) or `:none` (one dense block). Call it before building the state and its gates; they must
share a mode.

`:SU2` is not just a faster `:U1`: it stores reduced elements on multiplets, so every state it can
represent has definite total spin. A Neel or domain-wall state is unwritable there, which is why
`product_state` refuses under `:SU2`; use [`dimer_state`](@ref). Bond dimensions then count
multiplets, and `magnetisation` is identically zero for any S = 0 state.
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

The full MPS inner product `<bra|ket>` (`bra` conjugated), contracted bond by bond from the left.
Lengths and physical spaces must match; bond dimensions may differ. `overlap(psi, psi)` is
`norm(psi)^2`. A structurally zero overlap (no matching charge blocks) is exactly `0`.

BUG-Julia does not export `inner`: with `using ITensorMPS` that name resolves to ITensorMPS and
throws a MethodError. Use `overlap` or `LinearAlgebra.dot`.
"""
function overlap(bra::SymMPS, ket::SymMPS)
    length(bra) == length(ket) || throw(DimensionMismatch(
        "overlap needs equal-length states, got $(length(bra)) and $(length(ket))"))
    L = length(bra)
    # One site: every leg pairs (both links are dim-1 boundaries), so this is
    # exactly the single-tensor inner product.
    L == 1 && return tensor_inner(bra[1], ket[1])
    # The bra's internal links are retagged into a private "BL,i" namespace. Telum rejects two
    # identical indices in one tensor, and the transfer tensor carries bra and ket link_r with the
    # same "L,i+1" tag. The two boundary links keep their tags so bra and ket contract directly
    # there.
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
    product_state(spins; eltype = ComplexF64) -> SymMPS

Product state from `spins::Vector{Symbol}`, each `:up` or `:down`. `eltype` sets the site-tensor
type: complex by default because BUG evolves complex amplitudes; `Float64` gives a real state.

Each site is a fusion isometry restricted to the requested physical sector. The physical leg keeps
its full space (`preserve_space=true`) so local operators still contract, and the outgoing link is
trimmed to the populated sectors, which is why the bond dimension is 1.
"""
function product_state(spins::Vector{Symbol}; eltype::Type{<:Number} = ComplexF64)
    L = length(spins)
    L >= 1 || throw(ArgumentError("product_state needs at least one site"))
    # A definite-Sz product state spans many total-S sectors, so it has no bond dimension under :SU2
    # and must throw. Under :SU2 the `:down` filter silently selects nothing, because SU(2)'s
    # spin-1/2 label coincides with `SECTOR_UP`, and a warning would let the wrong state through.
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "product_state is not representable under :SU2 -- a definite-Sz product state has " *
        "weight in many total-spin sectors. Use dimer_state(L) for an S = 0 start."))
    return symmetry_mode() === :none ? _product_state_none(spins, eltype) :
                                       _product_state_u1(spins, eltype)
end

function _product_state_u1(spins::Vector{Symbol}, T::Type{<:Number} = ComplexF64)
    targets = map(enumerate(spins)) do (i, s)
        s === :up ? SECTOR_UP : s === :down ? SECTOR_DOWN :
            throw(ArgumentError("spin $i must be :up or :down, got $s"))
    end
    return sector_product_state(local_space(:U1), targets; eltype = T)
end

"""
    sector_product_state(q, sectors; eltype = ComplexF64) -> SymMPS

The product state with site `i` in the ONE-DIMENSIONAL U(1) sector `sectors[i]` of the local
space `q` -- `product_state` for any abelian site whose local states are distinct charges, such
as the spin-`s` sites of `spin_space` (charge `2 S^z`).
"""
function sector_product_state(q, sectors::AbstractVector; eltype::Type{<:Number} = ComplexF64)
    L, T = length(sectors), eltype
    L >= 1 || throw(ArgumentError("sector_product_state needs at least one site"))

    tensors = Any[]
    # Left boundary: a one-dimensional vacuum link.
    prev = (getvac(q.I, ("L,0", "L,1")), 2)

    for i in 1:L
        target = sectors[i]

        # Fuse the incoming link with a fresh physical leg. getIdentity returns
        # a lazy TLArrayView and getsub only accepts a concrete TLArray.
        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))

        # Keep only the requested spin, but preserve the physical leg's space.
        A = getsub(F, 2, s -> s == target ? Colon() : nothing; preserve_space = true)

        # Trim the outgoing link to the sectors that survived.
        A = to_concrete(A)
        # a label the site does not carry selects nothing, silently -- refuse it
        isempty(A.qlabels) && throw(ArgumentError(
            "site $i: sector $target is not a state of the local space"))
        present = Set(ql[3] for ql in A.qlabels)
        A = to_concrete(getsub(A, 3, s -> s in present ? Colon() : nothing))

        # BUG evolves complex amplitudes; promote once, here (unless a real state was asked for).
        A = to_concrete(A * one(T))

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

The product of nearest-neighbour singlets `(1,2)(3,4)...` under `:SU2`; `L` must be even. It is the
representable S = 0 starting point that replaces `product_state` (a Neel state has no SU(2) form).

Built by the fusion recursion of `_product_state_u1`, but the selection acts on the outgoing link,
since the physical leg has a single sector under SU(2). Odd sites keep spin 1/2 and even sites keep
spin 0, so after each even site the running total is a singlet: a chain of independent dimers.
Every link carries one multiplet, so `bond_dims` reads all ones, yet the state is entangled.
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
    _product_state_none(spins, T = ComplexF64) -> SymMPS

No-symmetry product state. The physical leg is a single dense sector of
dimension 2, so a spin is selected by INDEX on the outgoing link (`getsub`
returns an `Int`), not by charge sector. The physical leg keeps its full
2-dimensional space so local operators stay contractible.
"""
function _product_state_none(spins::Vector{Symbol}, T::Type{<:Number} = ComplexF64)
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
        A = to_concrete(A * one(T))

        push!(tensors, A)
        prev = (A, 3)
    end

    return SymMPS(tensors, L)
end

"""
    domain_wall_state(L; eltype = ComplexF64) -> SymMPS

`↑…↑↓…↓` with the wall in the middle. Total Sz is 0 for even `L`.
"""
domain_wall_state(L::Int; eltype::Type{<:Number} = ComplexF64) =
    product_state([i <= L ÷ 2 ? :up : :down for i in 1:L]; eltype = eltype)

"""
    neel_state(L; eltype = ComplexF64) -> SymMPS

`↑↓↑↓…`.
"""
neel_state(L::Int; eltype::Type{<:Number} = ComplexF64) =
    product_state([isodd(i) ? :up : :down for i in 1:L]; eltype = eltype)
