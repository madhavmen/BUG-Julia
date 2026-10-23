# mpo_env.jl: MPO environments (Eq. 1.8) and the effective Hamiltonians (Eq. 1.7) (from `mpo.jl`, included right after it).

# ── environments: Eq. (1.8) ──────────────────────────────────────────────────

"""
    MPOLink(E)

The MPO environment on ONE link: a rank-3 `(bra, mpo, ket)` tensor, or `nothing` at a
chain boundary (where it is the identity on a dim-1 vacuum leg).

The counterpart of [`ChannelSet`](@ref), and deliberately the same shape of object: a
sweep either CARRIES it bond to bond (`push_left_channels`) or reads a link out of a
prebuilt stack (`left_channels`).
"""
struct MPOLink
    E::Any
end

"The environment on a chain boundary: `L^[0] = 1` of Eq. (1.8)."
boundary_channels(::MPO) = MPOLink(nothing)

"""
    _mpo_left_step(E, A, W) -> TLArray

One step of Eq. (1.8): push the left environment on link `i` through site tensor `A` and
MPO tensor `W`, returning the environment on link `i+1` with legs `(bra, mpo, ket)`.

`A` may be a state tensor OR an expanded frame -- both carry an MPS tensor's
`(link_l, site, link_r)` layout, which is what lets `zero_site_h` push an environment
through `U_ex` with this same function.
"""
function _mpo_left_step(E, A, W)
    if E === nothing
        # Boundary: `link_l` is the dim-1 vacuum, contracted bra-to-ket, and `W`'s `w_l`
        # is the dim-1 leg left by trimming, dropped rather than contracted.
        T = contract(W, (2,), A, (2,))                  # (w_l, s_bra, w_r, ℓ_l, ℓ_r)
        T = contract(to_concrete!(T), (2, 4), _bra_left_boundary(A), (2, 1))
        T = to_concrete(deleteSingleton(to_concrete!(T), 1))   # (w_r, ℓ_r, ℓ_r')
        return to_concrete(permutedims(T, (3, 1, 2)))
    end
    # `to_concrete!` on each fresh `contract` result: the copying form deep-copied the
    # chi^2*w*d intermediates nobody else holds, on the most-called kernel of every sweep.
    # The outer `to_concrete` stays -- `permutedims`/`deleteSingleton` return views.
    T = contract(E, (3,), A, (1,))                       # (bra, w, s, ℓ_r)
    T = contract(to_concrete!(T), (2, 3), W, (1, 2))      # (bra, ℓ_r, s_bra, w_r)
    T = contract(to_concrete!(T), (1, 3), _bra_interior(A), (1, 2))  # (ℓ_r, w_r, ℓ_r')
    return to_concrete(permutedims(to_concrete!(T), (3, 2, 1)))
end

"""
    _mpo_right_step(E, A, W) -> TLArray

The mirror: push the right environment on link `i+1` through site `i`, giving link `i`.

No asymmetry to get backwards here -- unlike the channel recursion, which has to OPEN
with `term.right` and CLOSE with `term.left` on the way back. The MPO tensor is the same
object read the other way, which is the second reason this file exists.
"""
function _mpo_right_step(E, A, W)
    if E === nothing
        T = contract(W, (2,), A, (2,))                   # (w_l, s_bra, w_r, ℓ_l, ℓ_r)
        T = contract(to_concrete!(T), (2, 5), _bra_right_boundary(A), (2, 3))
        T = to_concrete(deleteSingleton(to_concrete!(T), 2))   # (w_l, ℓ_l, ℓ_l')
        return to_concrete(permutedims(T, (3, 1, 2)))
    end
    T = contract(A, (3,), E, (3,))                       # (ℓ_l, s, bra, w)
    T = contract(to_concrete!(T), (2, 4), W, (2, 4))      # (ℓ_l, bra, w_l, s_bra)
    T = contract(to_concrete!(T), (2, 4), _bra_interior(A), (3, 2))  # (ℓ_l, w_l, ℓ_l')
    return to_concrete(permutedims(to_concrete!(T), (3, 2, 1)))
end

"Carry the left environment on link `i` through site `i` onto link `i+1`. `open_next` is
accepted for signature compatibility with the channel path and ignored: an MPO cannot
leave a term dangling past the boundary, the trimmed `W^[L]` forbids it."
push_left_channels(cs::MPOLink, mpo::MPO, A, i::Int; open_next::Bool = true) =
    MPOLink(_mpo_left_step(cs.E, A, mpo[i]))

"Mirror of [`push_left_channels`](@ref) for the right environment."
push_right_channels(cs::MPOLink, mpo::MPO, A, i::Int; open_next::Bool = true) =
    MPOLink(_mpo_right_step(cs.E, A, mpo[i]))

"""
    MPOLeftEnvStack / MPORightEnvStack

Prebuilt environments, indexed by LINK exactly as [`LeftEnvStack`](@ref) /
[`RightEnvStack`](@ref) are: entry `i` of the left stack is everything at sites `< i`,
entry `i` of the right stack everything at sites `>= i`. Links outside the sweep's reach
hold `missing`, so reading one by mistake throws instead of being taken for a boundary.
"""
struct MPOLeftEnvStack
    E::Vector{Any}
end

struct MPORightEnvStack
    E::Vector{Any}
end

"""
    left_env_stack(psi, mpo; upto=length(psi)-1) -> MPOLeftEnvStack

Build the left MPO environments on links `1 … upto+1`. The stack IS the carry, recorded
link by link, so a sweep that carries and a sweep that prebuilds cannot drift.
"""
function left_env_stack(psi::SymMPS, mpo::MPO; upto::Int = length(psi) - 1)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    0 <= upto <= L || throw(ArgumentError("upto must be in 0:$L, got $upto"))
    E = Any[missing for _ in 1:(L + 1)]
    E[1] = nothing
    cur = nothing
    for i in 1:upto
        cur = _mpo_left_step(cur, psi[i], mpo[i])
        E[i + 1] = cur
    end
    return MPOLeftEnvStack(E)
end

"""
    right_env_stack(psi, mpo; downto=2) -> MPORightEnvStack

Build the right MPO environments on links `downto … L+1`.
"""
function right_env_stack(psi::SymMPS, mpo::MPO; downto::Int = 2)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    1 <= downto <= L + 1 || throw(ArgumentError(
        "downto must be in 1:$(L + 1), got $downto"))
    E = Any[missing for _ in 1:(L + 1)]
    E[L + 1] = nothing
    cur = nothing
    for i in L:-1:downto
        cur = _mpo_right_step(cur, psi[i], mpo[i])
        E[i] = cur
    end
    return MPORightEnvStack(E)
end

"Link `i` of a prebuilt left stack."
left_channels(st::MPOLeftEnvStack, i::Int) = MPOLink(st.E[i])

"Link `i` of a prebuilt right stack."
right_channels(st::MPORightEnvStack, i::Int) = MPOLink(st.E[i])

"""
    _mpo_left_close(E, A, W) -> ComplexF64

As [`_mpo_left_step`](@ref) at the LAST site: `link_r` is the dim-1 vacuum boundary so it
is contracted bra-to-ket, and `W`'s `w_r` is the dim-1 leg left by trimming to the `done`
column, so it is dropped. The result is a scalar. `deleteSingleton` doubles as the
assertion that the trimming really did leave a singleton there.
"""
function _mpo_left_close(E, A, W)
    T = contract(E, (3,), A, (1,))                       # (bra, w, s, ℓ_r)
    T = contract(to_concrete!(T), (2, 3), W, (1, 2))       # (bra, ℓ_r, s_bra, w_r)
    T = to_concrete(deleteSingleton(to_concrete!(T), 4))   # (bra, ℓ_r, s_bra)
    s = contract(T, (1, 3, 2), _bra_right_boundary(A), (1, 2, 3))
    return ComplexF64(to_concrete(s)[])
end

"""
    mpo_energy(psi, mpo) -> ComplexF64

`⟨ψ|H|ψ⟩` from the MPO recursion, closing both boundaries. The independent check on
[`env_energy`](@ref): the two share no contraction.

Unnormalised, like `env_energy` -- compare against `energy(psi, gates) * norm(psi)^2`
unless the state is normalised. Exact in any gauge: bra and ket are contracted
explicitly and no isometry is assumed anywhere.
"""
function mpo_energy(psi::SymMPS, mpo::MPO)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    E = nothing
    for i in 1:(L - 1)
        E = _mpo_left_step(E, psi[i], mpo[i])
    end
    return _mpo_left_close(E, psi[L], mpo[L])
end

"""
    _mpo2_left_step(E, A, W) -> TLArray

[`_mpo_left_step`](@ref) with TWO copies of `W` between bra and ket: the left environment of
`<psi|H H|psi>` on link `i+1`, legs `(bra, w_a, w_b, ket)` (`w_b` is the copy nearer the ket).
`W` is `(w_l, s_in, s_out, w_r)`; at the left boundary both copies' trimmed dim-1 `w_l` legs are
dropped exactly as the single-layer step drops one.

⛔ THE OUTER COPY'S VIRTUAL LEGS ARE PRIMED (`_mpo_outer`). Two copies of one `W` carry identical
leg tags, and Telum refuses a tensor holding two legs with the same tag ("Duplicate TLIndex ...
W") -- MEASURED, jobs 17208864/17208865. Priming `w_l`, `w_r` of the `a` copy keeps the layers
distinct; the physical legs need nothing, since the inner copy's `s_out` is contracted away.
"""
# `inc = 5`: far above any prime level the inner copy, the state or the bra carries, so the
# outer layer's virtual legs cannot coincide with any other leg whatever their direction.
_mpo_outer(W) = to_concrete(prime(prime(W, 1; inc = 5), 4; inc = 5))

function _mpo2_left_step(E, A, W)
    Wa = _mpo_outer(W)
    if E === nothing
        T = contract(W, (2,), A, (2,))                    # (wb_l, s1, wb_r, ℓ_l, ℓ_r)
        T = contract(to_concrete!(T), (2,), Wa, (2,))      # (wb_l, wb_r, ℓ_l, ℓ_r, wa_l, s2, wa_r)
        T = contract(to_concrete!(T), (6, 3), _bra_left_boundary(A), (2, 1))
        #                                                   (wb_l, wb_r, ℓ_r, wa_l, wa_r, ℓ_r')
        T = to_concrete(deleteSingleton(to_concrete!(T), 1))   # (wb_r, ℓ_r, wa_l, wa_r, ℓ_r')
        T = to_concrete(deleteSingleton(T, 3))                # (wb_r, ℓ_r, wa_r, ℓ_r')
        return to_concrete(permutedims(T, (4, 3, 1, 2)))
    end
    T = contract(E, (4,), A, (1,))                        # (bra, wa, wb, s, ℓ_r)
    T = contract(to_concrete!(T), (3, 4), W, (1, 2))       # (bra, wa, ℓ_r, s1, wb_r)
    T = contract(to_concrete!(T), (2, 4), Wa, (1, 2))      # (bra, ℓ_r, wb_r, s2, wa_r)
    T = contract(to_concrete!(T), (1, 4), _bra_interior(A), (1, 2))  # (ℓ_r, wb_r, wa_r, ℓ_r')
    return to_concrete(permutedims(to_concrete!(T), (4, 3, 2, 1)))
end

"""
    mpo_variance(psi, mpo) -> Float64

The exact energy variance `<H^2> - <H>^2` of the NORMALISED state, `<H^2> = <psi|H H|psi>` from a
two-layer MPO environment (no `H^2` MPO is formed). The accuracy measure a ground state carries
where exact diagonalisation is out of reach (simulation_plan_2 Task 4): zero exactly on an
eigenstate, and `variance / gap` bounds the energy error.

⚠ A difference of two `O(E^2)` numbers: its floor is `~eps * E^2`, i.e. ~1e-13 at `E ~ 30`.
"""
function mpo_variance(psi::SymMPS, mpo::MPO)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("mpo_variance needs at least two sites"))
    E = nothing
    for i in 1:(L - 1)
        E = _mpo2_left_step(E, psi[i], mpo[i])
    end
    A, W = psi[L], mpo[L]
    T = contract(E, (4,), A, (1,))                        # (bra, wa, wb, s, ℓ_r)
    T = contract(to_concrete!(T), (3, 4), W, (1, 2))       # (bra, wa, ℓ_r, s1, wb_r)
    T = to_concrete(deleteSingleton(to_concrete!(T), 5))   # (bra, wa, ℓ_r, s1)
    T = contract(T, (2, 4), _mpo_outer(W), (1, 2))        # (bra, ℓ_r, s2, wa_r)
    T = to_concrete(deleteSingleton(to_concrete!(T), 4))   # (bra, ℓ_r, s2)
    h2 = real(ComplexF64(to_concrete(contract(T, (1, 3, 2), _bra_right_boundary(A),
                                              (1, 2, 3)))[]))
    n2 = norm(psi)^2
    e = real(mpo_energy(psi, mpo)) / n2
    return h2 / n2 - e^2
end

# ── the effective Hamiltonians: Eq. (1.7) ────────────────────────────────────

"""
    apply_h_two_site(Theta, mpo, i, lenv, renv) -> TLArray

`H Theta` for the two-site block at bond `(i, i+1)`, legs
`(link_l, site_l, site_r, link_r)` in and out -- Eq. (1.7) with two MPO tensors between
the environments.

ONE contraction chain, against the channel version's five-case sum: the automaton's
states are summed over inside the MPO leg instead of being enumerated here. `lenv` must
sit on link `i` and `renv` on link `i+2`, and the state they were built from must be
canonical at the bond, since a boundary environment is the identity only in that gauge.
"""
function apply_h_two_site(Theta, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink)
    W1, W2 = mpo[i], mpo[i + 1]

    # Left end: attach the environment, or -- at the boundary -- keep `Theta`'s own link
    # leg and drop the MPO's dim-1 boundary leg.
    T = if lenv.E === nothing
        X = to_concrete!(contract(Theta, (2,), W1, (2,)))  # (ℓ_l,s_r,ℓ_r,w_l,s_bra,w_mid)
        to_concrete(deleteSingleton(X, 4))
    else
        X = to_concrete!(contract(lenv.E, (3,), Theta, (1,)))       # (bra,w,s_l,s_r,ℓ_r)
        to_concrete!(contract(X, (2, 3), W1, (1, 2)))
    end                                                  # (link_l, s_r, ℓ_r, s_bra_l, w)
    T = to_concrete!(contract(T, (5, 2), W2, (1, 2)))      # (link_l,ℓ_r,s_bra_l,s_bra_r,w)

    T = if renv.E === nothing
        X = to_concrete(deleteSingleton(T, 5))            # (link_l, ℓ_r, s_bra_l, s_bra_r)
        to_concrete(permutedims(X, (1, 3, 4, 2)))
    else
        to_concrete!(contract(T, (5, 2), renv.E, (2, 3)))  # (link_l,s_bra_l,s_bra_r,bra_r)
    end

    lenv.E === nothing || (T = _unprime(T, 1))
    renv.E === nothing || (T = _unprime(T, 4))
    return T
end

apply_h_two_site(Theta, mpo::MPO, i::Int,
                 lenv::MPOLeftEnvStack, renv::MPORightEnvStack) =
    apply_h_two_site(Theta, mpo, i, left_channels(lenv, i), right_channels(renv, i + 2))

# The CBE sketch, unchanged in substance: only `H Theta` comes from the MPO now. The
# projector and the sketch are the reference's (`RSVDpreBE0SiQS.m`), and nothing in the
# selection knows how `H` is stored.

"See the [`sketch_h_left`](@ref) for a term list; this is the same sketch on MPO
environments. ⚠ A standalone call rebuilds `H*Theta`; inside a sweep the shared cache in
[`_sketch_closures`](@ref) is what runs."
sketch_h_left(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    first(_sketch_closures(f,
        () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)

"Mirror of [`sketch_h_left`](@ref) on MPO environments."
sketch_h_right(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    last(_sketch_closures(f,
        () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)
