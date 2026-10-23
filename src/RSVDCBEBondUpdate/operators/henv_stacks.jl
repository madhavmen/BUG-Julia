# henv_stacks.jl: the left and right environment stacks and the two-site effective action (from `henv.jl`, included right after it).

# ── the left stack ───────────────────────────────────────────────────────────

"""
    LeftEnvStack

Left environments at links `1 … n`, one entry per link (`id[i]`, `done[i]`,
`open[i][t]`). `id[1] === nothing` is the boundary identity; `done`/`open` entries are
`ZERO` where the channel has no contribution yet.

`id` is stored even though it is the identity for a left-canonical chain: computing it
explicitly keeps the recursion valid with the orthogonality centre anywhere, and
`test_henv.jl` asserts it IS the identity in the canonical case rather than assuming it.
"""
struct LeftEnvStack
    id::Vector{Any}
    done::Vector{Any}
    open::Vector{Vector{Any}}
end

"""
    left_env_stack(psi, h; upto=length(psi)-1) -> LeftEnvStack

Build the left channel environments at links `1 … upto+1` by sweeping sites `1 … upto`.

`upto = L-1` (the default) fills every link a bond update can need on its left, i.e.
up to link `L`. `upto = 0` is legal and gives just the boundary at link 1 -- what bond 1
needs, where there is nothing to its left.
"""
function left_env_stack(psi::SymMPS, h::XXZChain; upto::Int = length(psi) - 1)
    L = length(psi)
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    0 <= upto <= L || throw(ArgumentError("upto must be in 0:$L, got $upto"))

    # The stack IS the carry, recorded link by link -- one implementation of the
    # recursion, so a sweep that carries and a sweep that prebuilds can never drift.
    cs = boundary_channels(h)
    id   = Any[cs.id]
    done = Any[cs.done]
    open = Vector{Any}[cs.open]
    for i in 1:upto
        # A channel opened at site i is closed at site i+1, so nothing opens on the last
        # site of the chain.
        cs = push_left_channels(cs, h, psi[i], i; open_next = i < L)
        push!(id, cs.id); push!(done, cs.done); push!(open, cs.open)
    end

    return LeftEnvStack(id, done, open)
end

"""
    env_energy(psi, h) -> ComplexF64

`⟨ψ|H|ψ⟩` from the channel recursion, closing the right boundary at site `L`.

Requires `psi` left-canonical up to `L` in the sense that the recursion is exact for
any gauge -- it contracts bra against ket explicitly and never assumes an isometry.
Valid for an unnormalised state: this is the unnormalised expectation value, so
compare against `energy(psi, gates) * norm(psi)^2` unless the state is normalised.
"""
function env_energy(psi::SymMPS, h::XXZChain)
    L = length(psi)
    st = left_env_stack(psi, h; upto = L - 1)
    A = psi[L]
    E = st.done[L] === ZERO ? ComplexF64(0) : _left_close(st.done[L], A, nothing, L)
    for (t, term) in enumerate(h.terms)
        st.open[L][t] === ZERO && continue
        E += term.coeff * _left_close(st.open[L][t], A, term.right, L)
    end
    return E
end

# ── one right step ───────────────────────────────────────────────────────────
#
# The mirror, with one asymmetry that is easy to get backwards: a right channel OPENS
# with `term.right` (the right half of a term, sitting at site i and waiting for its
# partner on its LEFT) and CLOSES with `term.left`. Under U(1) that also puts the op-leg
# arrows the other way round -- `Sp'` dangles a '+' leg here and `Sp` closes it -- which
# is exactly why the term list stores both halves instead of deriving one by transpose.

"""
    _right_step(E, A, O, i) -> TLArray

Push the right environment `E` at link `i+1` through site `i`, inserting `O` on the ket.
`E === nothing` means the right boundary identity. Returns `(bra, ket [, op])` on link
`i`, the same convention as [`_left_step`](@ref).
"""
function _right_step(E, A, O, i::Int)
    AO = _apply_site_op(A, O, i)
    nAO = length(AO.inds)

    if E === nothing
        out = contract(AO, (2, 3), _bra_right_boundary(A), (2, 3))
    else
        nE = length(E.inds)
        if nE == 3 && nAO == 4
            tmp = contract(AO, (3, 4), E, (2, 3))
        elseif nE == 2
            tmp = contract(AO, (3,), E, (2,))
        else
            error("_right_step: rank-$nE environment with a rank-$nAO operator would " *
                  "carry an open channel past site $i")
        end
        # tmp = (link_l, site, [op], bra) -- the bra leg is always last.
        out = contract(tmp, (2, length(tmp.inds)), _bra_interior(A), (2, 3))
    end

    n = length(out.inds)
    return to_concrete(permutedims(out, n == 2 ? (2, 1) : (3, 1, 2)))
end

"""
    _right_close(E, A, O, i) -> ComplexF64

As `_right_step` at site 1, where `link_l` is the dim-1 vacuum boundary and is
contracted bra-to-ket, giving a scalar.
"""
function _right_close(E, A, O, i::Int)
    AO = _apply_site_op(A, O, i)
    nAO, nE = length(AO.inds), length(E.inds)
    tmp = (nE == 3 && nAO == 4) ? contract(AO, (3, 4), E, (2, 3)) :
          nE == 2               ? contract(AO, (3,), E, (2,))     :
          error("_right_close: rank-$nE environment with a rank-$nAO operator at site $i")
    length(tmp.inds) == 3 || error(
        "_right_close: an operator channel is still open at the left boundary")
    s = to_concrete!(contract(tmp, (1, 2, 3), _bra_left_boundary(A), (1, 2, 3)))
    return ComplexF64(s[])
end

# ── the right stack ──────────────────────────────────────────────────────────

"""
    RightEnvStack

Right environments, indexed by LINK like [`LeftEnvStack`](@ref) but filled from the
right: entry `i` is the environment on link `i`, i.e. everything at sites `i … L`.
`id[L+1] === nothing` is the boundary identity. Links below the sweep's reach hold
`missing`, so using one by mistake throws instead of being read as a boundary.
"""
struct RightEnvStack
    id::Vector{Any}
    done::Vector{Any}
    open::Vector{Vector{Any}}
end

"""
    right_env_stack(psi, h; downto=2) -> RightEnvStack

Build the right channel environments on links `downto … L+1` by sweeping sites
`L … downto`. `downto = 2` (the default) fills every link a bond update can need on its
right, i.e. down to link 2. `downto = L+1` is legal and gives just the boundary -- what
bond `L-1` needs, where there is nothing to its right.
"""
function right_env_stack(psi::SymMPS, h::XXZChain; downto::Int = 2)
    L = length(psi)
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    1 <= downto <= L + 1 || throw(ArgumentError(
        "downto must be in 1:$(L + 1), got $downto"))
    nt = length(h.terms)

    id   = Any[missing for _ in 1:(L + 1)]
    done = Any[missing for _ in 1:(L + 1)]
    open = Vector{Any}[Any[missing for _ in 1:nt] for _ in 1:(L + 1)]

    cs = boundary_channels(h)
    id[L + 1] = cs.id; done[L + 1] = cs.done; open[L + 1] = cs.open
    for i in L:-1:downto
        # A channel opened at site i is closed at site i-1, so nothing opens at site 1.
        cs = push_right_channels(cs, h, psi[i], i; open_next = i > 1)
        id[i] = cs.id; done[i] = cs.done; open[i] = cs.open
    end

    return RightEnvStack(id, done, open)
end

"""
    env_energy_right(psi, h) -> ComplexF64

`⟨ψ|H|ψ⟩` swept from the right instead of the left. Same number as
[`env_energy`](@ref); its purpose is to pin the right recursion independently, since
the two sweeps share no contraction.
"""
function env_energy_right(psi::SymMPS, h::XXZChain)
    L = length(psi)
    st = right_env_stack(psi, h; downto = 2)
    A = psi[1]
    E = st.done[2] === ZERO ? ComplexF64(0) : _right_close(st.done[2], A, nothing, 1)
    for (t, term) in enumerate(h.terms)
        st.open[2][t] === ZERO && continue
        E += term.coeff * _right_close(st.open[2][t], A, term.left, 1)
    end
    return E
end

# ── the two-site effective action ────────────────────────────────────────────
#
# `H` restricted to bond `(i, i+1)`, the object both halves of CBE-BUG consume: the
# sketch applies it to a random block (`RSVDpreBE0SiQS.m:336`) and the Galerkin step
# applies it to the centre core. Five contributions, and the two-channel structure is
# what makes them exhaustive for a nearest-neighbour H:
#
#   (a) done_L ⊗ 1          terms entirely left of the block
#   (b) 1 ⊗ done_R          terms entirely right of it
#   (c) open_L[t] closed by term.right at site i      straddling link i
#   (d) open_R[t] closed by term.left  at site i+1    straddling link i+2
#   (e) the bond term itself
#
# (e) is delegated to `apply_gate` with the same gate the Trotter path uses, rather than
# rebuilt from the term list: that code is pinned to the analytic Heisenberg block
# element by element (`test_gates.jl`), and stage 1a establishes that the term list and
# the gate are the same operator.
#
# CANONICALITY IS REQUIRED, NOT OPTIONAL. (a) and (b) each carry an implicit identity on
# the far side, which is only what `id_L`/`id_R` are when sites `1…i-1` are left
# isometries and `i+2…L` are right isometries -- i.e. when the centre is at `i` or `i+1`,
# exactly what `bond_frame` already demands. `test_henv.jl` asserts the `id` channels are
# identities in that gauge rather than trusting it.

"""
    _apply_theta_op(Theta, O, leg, i) -> TLArray

`O_i` acting on `leg` (2 for the left site, 3 for the right) of the two-site block,
returned in `Theta`'s layout `(link_l, site_l, site_r, link_r)` with any op-leg appended.

As in [`_apply_site_op`](@ref), the ket-facing leg of `O` is found by arrow, so an
adjoint half-term works unchanged.
"""
function _apply_theta_op(Theta, O, leg::Int, i::Int)
    O === nothing && return Theta
    Oi = _retag_sites(O, i)
    ket_leg = Oi.inds[1].dir == '+' ? 1 : 2
    T = contract(Oi, (ket_leg,), Theta, (leg,))
    rank3 = length(Oi.inds) == 3
    # T = (site_out, [op], Theta's legs except `leg`, in order).
    perm = if leg == 2
        rank3 ? (3, 1, 4, 5, 2) : (2, 1, 3, 4)
    elseif leg == 3
        rank3 ? (3, 4, 1, 5, 2) : (2, 3, 1, 4)
    else
        throw(ArgumentError("_apply_theta_op: leg must be 2 or 3, got $leg"))
    end
    return to_concrete(permutedims(T, perm))
end

"""
    _apply_theta_env(T, E, link_leg) -> TLArray

Contract environment `E` onto `link_leg` (1 or 4) of `T`, where `T` carries
`(link_l, site_l, site_r, link_r [, op])`. If both `T` and `E` have an op-leg they are
contracted (the channel closes); the environment's bra leg replaces `link_leg` and is
unprimed so the result has exactly `Theta`'s legs again.
"""
function _apply_theta_env(T, E, link_leg::Int)
    link_leg in (1, 4) || throw(ArgumentError("link_leg must be 1 or 4, got $link_leg"))
    nT, nE = length(T.inds), length(E.inds)
    out = if nT == 5 && nE == 3
        contract(T, (link_leg, 5), E, (2, 3))
    elseif nE == 2 && nT == 4
        contract(T, (link_leg,), E, (2,))
    else
        error("_apply_theta_env: rank-$nT block against a rank-$nE environment leaves " *
              "an operator channel dangling")
    end
    # `out` = T's remaining legs in order, then E's bra leg.
    #   link_leg == 1: (site_l, site_r, link_r, bra) -> bra first
    #   link_leg == 4: (link_l, site_l, site_r, bra) -> already in place
    out = link_leg == 1 ? to_concrete(permutedims(out, (4, 1, 2, 3))) : to_concrete(out)
    return to_concrete(prime(out, link_leg; inc = -1))
end

"""
    apply_h_two_site(Theta, h, i, lenv, renv) -> TLArray

`H Theta` for the two-site block at bond `(i, i+1)`, legs
`(link_l, site_l, site_r, link_r)` in and out.

`lenv` must be filled at link `i` and `renv` at link `i+2`; the state they were built
from must be canonical at the bond (see the note above). This is the whole chain's `H`,
not the bond's gate -- which is the point of the module.
"""
function apply_h_two_site(Theta, h::XXZChain, i::Int,
                          lch::ChannelSet, rch::ChannelSet)
    tl, tr = Theta.inds[2].itags, Theta.inds[3].itags
    # ⛔ COLLECT, THEN SUM ONCE. This used to accumulate pairwise --
    # `acc = to_concrete(acc + x)` -- which for a nearest-neighbour chain is 2 + 2*|terms| + 1
    # ~ 9 separate `_sum_tlarrays` calls, each followed by its own `to_concrete`. Every
    # `to_concrete` runs `materialize` + `_eager_tlarray` and rebuilds the qlabel / wmatinfo /
    # RMT vectors from scratch, so the metadata churn was paid nine times per call.
    #
    # MEASURED COST OF THAT (`benchmarks/rsvd_probe_microbench.jl`): the CBE probe, which is one
    # `apply_h_two_site` plus a projection and a sketch contraction, allocated 1.53 MB at chi=16
    # on a two-site block holding 16 KB of data -- 96x the tensor's own size -- and 84% of that
    # was this function. It is what makes the probe allocation-bound rather than flop-bound, and
    # therefore what makes the randomised sketch unable to show a speedup: RSVD reduces flops,
    # and flops were not the cost.
    #
    # Telum's `sum` over a tuple/vector aligns all summands ONCE and adds them in a single pass
    # (`_sum_tlarrays`), so one materialisation replaces nine.
    terms = AbstractTLArray[]
    add!(x) = push!(terms, x)

    # (a), (b): the completed channels, with an implicit identity on the far side.
    lch.done === ZERO || add!(_apply_theta_env(Theta, lch.done, 1))
    rch.done === ZERO || add!(_apply_theta_env(Theta, rch.done, 4))

    # (c), (d): the half-open channels, closed inside the block.
    for (t, term) in enumerate(h.terms)
        o = lch.open[t]
        if o !== ZERO
            T = _apply_theta_op(Theta, term.right, 2, i)
            add!(to_concrete(term.coeff * _apply_theta_env(T, o, 1)))
        end
        o = rch.open[t]
        if o !== ZERO
            T = _apply_theta_op(Theta, term.left, 3, i + 1)
            add!(to_concrete(term.coeff * _apply_theta_env(T, o, 4)))
        end
    end

    # (e) the bond's own term, via the gate the Trotter path is pinned against.
    add!(apply_gate(bond_gate(h, tl, tr), Theta, tl, tr))

    # `nothing` when every channel was ZERO and there is no gate -- preserved from the pairwise
    # version, which returned an unset `acc`.
    isempty(terms) && return nothing
    return length(terms) == 1 ? terms[1] : to_concrete(sum(terms))
end

apply_h_two_site(Theta, h::XXZChain, i::Int,
                 lenv::LeftEnvStack, renv::RightEnvStack) =
    apply_h_two_site(Theta, h, i, left_channels(lenv, i), right_channels(renv, i + 2))

"""
    bond_gate(h, site_l, site_r) -> TLArray

The two-site gate of `h` on one bond -- `heisenberg_bond_gate` with `h`'s couplings, so
the term list and the gate can never disagree about `J` or `delta`.
"""
bond_gate(h::XXZChain, site_l, site_r) =
    heisenberg_bond_gate(site_l, site_r; J = h.J, delta = h.delta)

# ── reading a link out of a prebuilt stack ───────────────────────────────────
#
# Defined here rather than beside `ChannelSet`, because a method signature's types are
# resolved when the method is defined and `LeftEnvStack` / `RightEnvStack` are declared
# further up this file.

"Link `i` of a prebuilt left stack, as a `ChannelSet`."
left_channels(lenv::LeftEnvStack, i::Int) =
    ChannelSet(lenv.id[i], lenv.done[i], lenv.open[i])

"Link `i` of a prebuilt right stack, as a `ChannelSet`."
right_channels(renv::RightEnvStack, i::Int) =
    ChannelSet(renv.id[i], renv.done[i], renv.open[i])
