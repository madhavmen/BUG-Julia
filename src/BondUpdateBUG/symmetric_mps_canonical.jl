# symmetric_mps_canonical.jl: canonical form and observables of `SymMPS` (from `symmetric_mps.jl`, included right after it).

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
    psi[j] = _sz_site(psi[j])
    canonical!(psi, 1)
    return psi
end

"`S^z` on the site leg (leg 2) of one MPS tensor, by sector projection ([`apply_sz!`](@ref))."
function _sz_site(A)
    up = to_concrete(getsub(A, 2, s -> s == SECTOR_UP   ? Colon() : nothing;
                            preserve_space = true))
    dn = to_concrete(getsub(A, 2, s -> s == SECTOR_DOWN ? Colon() : nothing;
                            preserve_space = true))
    return to_concrete(to_concrete(0.5 * up) + to_concrete(-0.5 * dn))
end

"""
    sz_overlaps(bra, ket) -> Vector{ComplexF64}

`[<bra| S^z_x |ket> for x in 1:L]` in ONE left and one right transfer pass, i.e. about three
[`overlap`](@ref)s instead of `L`. This is the correlator row `G(x,t) ~ <Om| S^z_x |phi(t)>`; the
route through `L` stored states `S^z_x|Om>` costs `L` overlaps per time and `L` copies of the
ground state (several GB at chi >= 512 on 40 sites). `S^z_x` acts on one site leg in any gauge,
so it is applied to the bra tensor alone ([`apply_sz!`](@ref)'s projection, Hermitian). States in
different total charge sectors give zeros, as `overlap` does.
"""
function sz_overlaps(bra::SymMPS, ket::SymMPS)
    length(bra) == length(ket) || throw(DimensionMismatch(
        "sz_overlaps needs equal-length states, got $(length(bra)) and $(length(ket))"))
    symmetry_mode() === :SU2 && throw(ArgumentError("sz_overlaps needs :U1, see apply_sz!"))
    L = length(bra)
    L >= 2 || return [ComplexF64(tensor_inner(_sz_site(bra[1]), ket[1]))]
    # bra links retagged into "BL,i" exactly as `overlap` does, so the transfer legs are distinct
    braR = Vector{Any}(undef, L)
    for i in 1:L
        t = bra[i]
        i > 1 && (t = setitag(t, 1, "BL,$i"))
        i < L && (t = setitag(t, 3, "BL,$(i + 1)"))
        braR[i] = to_concrete(t)
    end
    braq = Set(first(sp) for sp in braR[L].spaces[3])
    ketq = Set(first(sp) for sp in ket[L].spaces[3])
    isdisjoint(braq, ketq) && return zeros(ComplexF64, L)
    scalar(s) = (r = to_concrete(s); length(r) == 0 ? 0.0 + 0.0im : ComplexF64(r[]))
    # left: E[i] = sites 1..i, legs (bra_link_r, ket_link_r)
    lstep(E, b, i) = contract(contract(E, (1,), b', (1,)), (1, 2), ket[i], (1, 2))
    E = Vector{Any}(undef, L - 1)
    E[1] = contract(braR[1]', (1, 2), ket[1], (1, 2))
    for i in 2:(L - 1)
        E[i] = lstep(E[i - 1], braR[i], i)
    end
    # right: R[i] = sites i..L, legs (bra_link_l, ket_link_l)
    R = Vector{Any}(undef, L)
    R[L] = contract(braR[L]', (2, 3), ket[L], (2, 3))
    for i in (L - 1):-1:2
        R[i] = contract(braR[i]', (2, 3), contract(ket[i], (3,), R[i + 1], (2,)), (2, 3))
    end
    G = zeros(ComplexF64, L)
    for x in 1:L
        b = _sz_site(braR[x])
        if x == L
            G[x] = scalar(contract(contract(E[L - 1], (1,), b', (1,)), (1, 2, 3), ket[L],
                                   (1, 2, 3)))
        else
            Ex = x == 1 ? contract(b', (1, 2), ket[1], (1, 2)) : lstep(E[x - 1], b, x)
            G[x] = scalar(contract(Ex, (1, 2), R[x + 1], (1, 2)))
        end
    end
    return G
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
