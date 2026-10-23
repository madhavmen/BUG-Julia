# mpo_fold.jl: fold-omega-first: the sketch that never forms `H*Theta` (from `mpo.jl`, included right after it).

# ── fold-omega-first: the sketch that never forms H*Theta ──
# Contracting `Om` in first makes the cost scale with the sketch width g, not d*chi. It is exact, since
# P_perp and Om act on disjoint legs and commute.
# The three sketches use different Om and cannot share one folded chain, so it wins only for small dex (3g < chi).

"""
    _fold_sketch_closures(f, mpo, i, lenv, renv) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, with `Om` folded into the contraction chain
instead of applied to a finished `H*Theta`.

`skl(Om)` takes `Om` in `V0`'s layout `(g, site_r, link_r)` and returns `(link_l, site_l, g)`;
`skr(Om)` takes `U0`'s layout `(link_l, site_l, g)` and returns `(g, site_r, link_r)`. Both
return the DISCARDED-space component, i.e. with `P_perp` already applied -- identical to
[`_sketch_closures`](@ref)'s output.
"""
function _fold_sketch_closures(f::BondFrame, mpo::MPO, i::Int,
                               lenv::MPOLink, renv::MPOLink)
    W1, W2 = mpo[i], mpo[i + 1]
    # ⚠ `Theta` IS SHARED, `H*Theta` IS NOT. `frame_theta` is `U0*S0*V0` -- an `O(chi^2 d^2)`
    # object the frame already implies, with no `H` in it. Rebuilding it per call would be pure
    # waste; caching the thing this function exists NOT to build would defeat the purpose.
    Theta = frame_theta(f)                       # (ℓ_l, s_l, s_r, ℓ_r)

    # ── LEFT: fold `Om` through renv and W2, then Theta, then W1 and lenv ────────────────
    function skl(Om)
        Omd = Om'                                # (g, s_bra_r, bra_r)
        # 1. the right cap. At the boundary `renv.E` is absent and `W2`'s `w_r` is the dim-1
        #    leg left by trimming, dropped rather than contracted -- the same asymmetry
        #    `apply_h_two_site` handles, and the reason this is not one expression.
        C = if renv.E === nothing
            X = to_concrete!(contract(Omd, (2,), W2, (3,)))   # (g,bra_r,w_mid,s_ket_r,w_r)
            to_concrete(deleteSingleton(X, 5))               # (g, bra_r, w_mid, s_ket_r)
        else
            X = to_concrete!(contract(Omd, (3,), renv.E, (1,)))  # (g, s_bra_r, w, ket_r)
            # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close `s_bra_r` and `w_r`.
            to_concrete!(contract(X, (2, 3), W2, (3, 4)))     # (g, ket_r, w_mid, s_ket_r)
        end
        # 2. into Theta: close the ket site and ket link on the right.
        D = to_concrete!(contract(Theta, (3, 4), C, (4, 2)))  # (ℓ_l, s_l, g, w_mid)
        # 3. W1 = (w_l, s_ket_l, s_bra_l, w_mid); close the ket site and the shared MPO leg.
        E = to_concrete!(contract(D, (2, 4), W1, (2, 4)))     # (ℓ_l, g, w_l, s_bra_l)
        Y = if lenv.E === nothing
            X = to_concrete(deleteSingleton(E, 3))           # (ℓ_l, g, s_bra_l)
            to_concrete(permutedims(X, (1, 3, 2)))           # (ℓ_l, s_bra_l, g)
        else
            # lenv.E = (bra, w, ket); close the ket link and `w_l`.
            X = to_concrete!(contract(E, (1, 3), lenv.E, (3, 2)))  # (g, s_bra_l, bra)
            _unprime(to_concrete(permutedims(X, (3, 2, 1))), 1)   # (bra, s_bra_l, g)
        end
        return _project_left(f, Y)
    end

    # ── RIGHT: the mirror -- fold `Om` through lenv and W1 first ────────────────────────
    function skr(Om)
        Omd = Om'                                # (bra_l, s_bra_l, g)
        C = if lenv.E === nothing
            X = to_concrete!(contract(Omd, (2,), W1, (3,)))   # (bra_l,g,w_l,s_ket_l,w_mid)
            to_concrete(deleteSingleton(X, 3))               # (bra_l, g, s_ket_l, w_mid)
        else
            X = to_concrete!(contract(Omd, (1,), lenv.E, (1,)))  # (s_bra_l, g, w, ket_l)
            # W1 = (w_l, s_ket_l, s_bra_l, w_mid); close `s_bra_l` and `w_l`.
            Z = to_concrete!(contract(X, (1, 3), W1, (3, 1)))    # (g, ket_l, s_ket_l, w_mid)
            to_concrete(permutedims(Z, (2, 1, 3, 4)))           # (ket_l, g, s_ket_l, w_mid)
        end
        # into Theta: close the ket link and ket site on the left.
        D = to_concrete!(contract(Theta, (1, 2), C, (1, 3)))  # (s_r, ℓ_r, g, w_mid)
        # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close the ket site and the shared MPO leg.
        E = to_concrete!(contract(D, (1, 4), W2, (2, 1)))     # (ℓ_r, g, s_bra_r, w_r)
        Y = if renv.E === nothing
            X = to_concrete(deleteSingleton(E, 4))           # (ℓ_r, g, s_bra_r)
            to_concrete(permutedims(X, (2, 3, 1)))           # (g, s_bra_r, ℓ_r)
        else
            X = to_concrete!(contract(E, (1, 4), renv.E, (3, 2)))  # (g, s_bra_r, bra_r)
            _unprime(X, 3)
        end
        return _project_right(f, Y)
    end

    return skl, skr
end

"`P_perp^L Y` with `P_perp^L = I - U0 U0'`, for `Y` of layout `(link_l, site_l, g)`."
function _project_left(f::BondFrame, Y)
    c = to_concrete!(contract(f.U0', (1, 2), Y, (1, 2)))      # (bond, g)
    return to_concrete(Y - to_concrete!(contract(f.U0, (3,), c, (1,))))
end

"`Y P_perp^R` with `P_perp^R = I - V0' V0`, for `Y` of layout `(g, site_r, link_r)`."
function _project_right(f::BondFrame, Y)
    c = to_concrete!(contract(Y, (2, 3), f.V0', (2, 3)))      # (g, bond)
    return to_concrete(Y - to_concrete!(contract(c, (2,), f.V0, (1,))))
end

"""
The bond expansion at bond `i`, driven by MPO environments.

`fold_omega = true` selects [`_fold_sketch_closures`](@ref), which folds the probe into the
contraction instead of building `H*Theta`. ⚠ It makes `share_ht` MEANINGLESS -- there is no
shared object to cache -- so passing both is refused rather than silently ignoring one.
"""
function cbe_expand(f::BondFrame, mpo::MPO, i::Int,
                    lenv::MPOLink, renv::MPOLink;
                    # ⛔ `nothing`, NOT `true`, SO "NOT PASSED" IS DISTINGUISHABLE FROM "PASSED
                    # true". With a `Bool` default the contradiction check below cannot fire at
                    # all -- `share_ht` is a NAMED kwarg, so it never appears in `kwargs...` and
                    # `haskey(kwargs, :share_ht)` is false even when the caller passed it. That
                    # is a guard that reads as protection and provides none.
                    share_ht::Union{Nothing, Bool} = nothing,
                    fold_omega::Bool = false, kwargs...)
    if fold_omega
        # ⛔ REFUSE THE CONTRADICTION RATHER THAN RESOLVING IT. A caller who sets `fold_omega`
        # alone is asserting nothing about sharing and is not warned; a caller who ALSO passes
        # `share_ht` explicitly believes something false about the run they are about to time,
        # and silently honouring one of the two is how a benchmark reports the wrong arm.
        share_ht === nothing || throw(ArgumentError(
            "fold_omega = true never forms H*Theta, so there is nothing for share_ht to " *
            "share; pass one or the other"))
        return cbe_expand(f, _fold_sketch_closures(f, mpo, i, lenv, renv)...; kwargs...)
    end
    return cbe_expand(f,
                      _sketch_closures(f,
                          () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv);
                          share = share_ht === nothing ? true : share_ht)...; kwargs...)
end

"""
    MPOOneSiteH

The single-site effective Hamiltonian of Eq. (1.7), `H_eff^[j] = L^[j-1] W^[j] R^[j+1]`, held
as its three factors rather than contracted: the operand is a site tensor, so contracting them
first would build a `(chi d) x (chi d)` matrix where three sequential contractions cost
`O(chi^2 d w)`.

`nothing` on either environment is the chain boundary, where the MPO's own dim-1 leg is
dropped instead of contracted -- see the file header.
"""
struct MPOOneSiteH
    l::Any                    # (bra, mpo, ket) at link j, or `nothing`
    w::Any                    # W^[j]
    r::Any                    # (bra, mpo, ket) at link j+1, or `nothing`
end

"""
    one_site_h(mpo, j, lenv, renv) -> MPOOneSiteH

`H_eff` at site `j`. `lenv` must sit on link `j` and `renv` on link `j+1` -- the same
convention the channel-path method of this function uses.
"""
one_site_h(mpo::MPO, j::Int, lenv::MPOLink, renv::MPOLink) =
    MPOOneSiteH(lenv.E, mpo[j], renv.E)

one_site_h(mpo::MPO, j::Int, lenv::MPOLeftEnvStack, renv::MPORightEnvStack) =
    one_site_h(mpo, j, left_channels(lenv, j), right_channels(renv, j + 1))

"""
    apply_one_site(H1::MPOOneSiteH, A) -> TLArray

`H_eff A` for a site tensor `A` with legs `(link_l, site, link_r)`, in and out.
"""
function apply_one_site(H1::MPOOneSiteH, A)
    # `to_concrete!` on the contraction results: this is the hot path, and `to_concrete` on an already
    # concrete tensor would deep-copy every block.
    # Safe only because every argument is a fresh `contract` result; the inputs belong to callers and keep `to_concrete`.
    T = if H1.l === nothing
        X = to_concrete!(contract(A, (2,), H1.w, (2,)))     # (ℓ_l, ℓ_r, w_l, s_bra, w_r)
        to_concrete(deleteSingleton(X, 3))                  # (ℓ_l, ℓ_r, s_bra, w_r)
    else
        X = to_concrete!(contract(H1.l, (3,), A, (1,)))      # (bra_l, w, s, ℓ_r)
        X = to_concrete!(contract(X, (2, 3), H1.w, (1, 2)))  # (bra_l, ℓ_r, s_bra, w_r)
        X
    end                                                    # (link_l, ℓ_r, s_bra, w_r)

    T = if H1.r === nothing
        X = to_concrete(deleteSingleton(T, 4))             # (link_l, ℓ_r, s_bra)
        to_concrete(permutedims(X, (1, 3, 2)))
    else
        to_concrete!(contract(T, (4, 2), H1.r, (2, 3)))    # (link_l, s_bra, bra_r)
    end

    H1.l === nothing || (T = _unprime(T, 1))
    H1.r === nothing || (T = _unprime(T, 3))
    return T
end

"""
    MPOZeroSiteH

The centre bond's effective Hamiltonian, as the two environments already pushed through
the expanded frames:

    H_eff^{(0)} S = sum_w L[w] S R[w]

One contraction over the MPO leg -- Eq. (1.7) at a bond, where the site tensor of the
1-site form is absent. Acting with it never touches a site tensor, so a Krylov matvec is
`O(chi^2 w)` rather than the `O(chi^2 d^2)` of a two-site apply.
"""
struct MPOZeroSiteH
    l::Any                    # (bra_bL, mpo, bL)
    r::Any                    # (bra_bR, mpo, bR)
end

"""
    zero_site_h(mpo, i, lenv, renv, U_ex, V_ex) -> MPOZeroSiteH

Push the environments at links `i` and `i+2` onto the centre bond through the EXPANDED
frames. `U_ex` carries `(link_l, site_l, bond)` and `V_ex` carries `(bond, site_r,
link_r)` -- an MPS tensor's layout in each case -- so this is one more step of Eq. (1.8)
with the frame in place of a state tensor, and needs no separate code.
"""
zero_site_h(mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, U_ex, V_ex) =
    MPOZeroSiteH(_mpo_left_step(lenv.E, U_ex, mpo[i]),
                 _mpo_right_step(renv.E, V_ex, mpo[i + 1]))

"""
    apply_zero_site(H0::MPOZeroSiteH, S) -> TLArray

`H_eff S` for the centre core `S`, legs `(bond_l, bond_r)` in and out. Each environment
contributes its bra leg in place of the ket leg it consumes, so the result carries `S`'s
own legs after unpriming.
"""
function apply_zero_site(H0::MPOZeroSiteH, S)
    T = to_concrete!(contract(H0.l, (3,), S, (1,)))        # (bra_l, mpo, bond_r)
    T = to_concrete!(contract(T, (2, 3), H0.r, (2, 3)))    # (bra_l, bra_r)
    return _unprime(_unprime(T, 1), 2)
end

"""
    shift_h(H, s) -> H or ShiftedH

The effective Hamiltonian `H - s` for the generators of an imaginary-time step. `s = 0` returns `H`
itself, so an unshifted step is bit-identical.

A shift by the identity leaves every Krylov space unchanged and only changes the weights read off
them. The CBE sketches are exact as they stand, since `P_perp (H - s) Theta = P_perp H Theta`.
"""
struct ShiftedH{T}
    h::T
    s::Float64
end

shift_h(h, s::Real) = iszero(s) ? h : ShiftedH(h, Float64(s))

apply_one_site(H::ShiftedH, A) = to_concrete(apply_one_site(H.h, A) - H.s * A)
apply_zero_site(H::ShiftedH, S) = to_concrete(apply_zero_site(H.h, S) - H.s * S)
