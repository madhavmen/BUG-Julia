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

Select the symmetry the local space is built with: `:U1` (charge-conserving
spin sectors) or `:none` (a single dense block). Set this before building the
state and its gates; they must share a mode.
"""
function set_symmetry!(sym::Symbol)
    sym in (:U1, :none) || throw(ArgumentError("symmetry must be :U1 or :none, got $sym"))
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
    opts = sym === :U1   ? SpinOptions(:U1, 1)     :
           sym === :none ? SpinOptions(nothing, 1) :
           throw(ArgumentError("symmetry must be :U1 or :none, got $sym"))
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
