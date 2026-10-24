# The MPO-MPS product, for a quantum jump `L_μ ψ` (dissipative plan Task 4). A charged MPO (a jump
# changes the U(1) sector) has a charged right boundary leg, which the product fuses into the
# state's own, so the result carries the shifted total charge.

"""
    mpo_product(mpo, psi) -> SymMPS

`mpo * psi`, exact: each site's MPS and MPO links are fused by a `getIdentity` isometry, so the bond
dimension is the product of the two. The result is NOT canonical (`norm` reads the centre, so use
`overlap`, or [`apply_mpo`](@ref) for a gauged and compressed state). The legs follow the
environments (`_mpo_left_step`): the ket's site leg contracts with `w`'s leg 2, and leg 3 is the
product's site leg.
"""
function mpo_product(mpo::MPO, psi::SymMPS)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch("MPO has $(length(mpo)) sites, state has $L"))
    out = Vector{Any}(undef, L)
    F = nothing
    for i in 1:L
        T = to_concrete!(contract(mpo[i], (2,), psi[i], (2,)))       # (w_l, s, w_r, ℓ_l, ℓ_r)
        T = if i == 1
            # the trimmed `w_l` of site 1 is the dim-1 vacuum; the state's own boundary stays
            to_concrete(permutedims(to_concrete(deleteSingleton(T, 1)), (3, 1, 4, 2)))
        else
            # undo the previous site's fusion on (ℓ_l, w_l): F'F is the identity on the pair
            to_concrete(permutedims(to_concrete!(contract(F', (1, 2), T, (4, 1))), (1, 2, 4, 3)))
        end                                                           # (ℓ_l, s, ℓ_r, w_r)
        # Placeholder tag: the fused link must not share `L,i+1` with the leg it replaces.
        F = to_concrete(getIdentity((T, 3), (T, 4); itag = "Jf,$(i + 1)"))
        out[i] = to_concrete!(contract(T, (3, 4), F, (1, 2)))        # (ℓ_l, s, fused)
    end
    for i in 1:L
        A = to_concrete(setitag(out[i], 3, "L,$(i + 1)"))
        out[i] = i == 1 ? A : to_concrete(setitag(A, 1, "L,$i"))
    end
    return SymMPS(out, L)
end

"""
    apply_mpo(mpo, psi; tol_trunc = 0.0, maxdim = 10_000, root = L ÷ 2 + 1) -> (SymMPS, discarded)

[`mpo_product`](@ref), gauged (a right-to-left sweep from site `L` makes every site an isometry,
whatever the product's gauge) and compressed once by `truncate_recursive!` at relative
discarded norm `tol_trunc` and rank `maxdim` (the step's own compression rule). Centre `root`.
Returns the state and its relative discarded norm. A product that is exactly zero (a jump that
annihilates every component) is refused: its probability is zero, so it can never be drawn.
"""
function apply_mpo(mpo::MPO, psi::SymMPS; tol_trunc::Float64 = 0.0, maxdim::Int = 10_000,
                   root::Int = length(psi) ÷ 2 + 1)
    phi = mpo_product(mpo, psi)
    real(overlap(phi, phi)) > 0 || throw(ArgumentError(
        "apply_mpo: the product is exactly zero (the operator annihilates the state)"))
    canonical!(phi, 1)
    canonical!(phi, root)
    _, _, cum, _ = truncate_recursive!(phi, root; maxdim = maxdim, weight_tol = tol_trunc)
    return phi, cum
end
