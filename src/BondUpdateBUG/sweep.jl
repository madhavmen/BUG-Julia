# Odd/even Trotter sweep over the bonds.
#
# Mirrors two_site_bug/scheme.py::parity_bonds / kls_bond / parity_sweep. Bonds
# of one parity act on disjoint site pairs, so the group is an EXACT factor of
# the Trotter step -- the splitting error lives entirely between the two groups,
# never inside one.

"""
    parity_bonds(L, parity) -> Vector{Int}

Left-site indices of one commuting bond group. `:even` is `1, 3, 5, …` and
`:odd` is `2, 4, …` in Julia's 1-based indexing, matching Python's 0-based
`range(0, L-1, 2)` / `range(1, L-1, 2)` bond for bond.
"""
function parity_bonds(L::Int, parity::Symbol)
    parity === :even && return collect(1:2:(L - 1))
    parity === :odd  && return collect(2:2:(L - 1))
    throw(ArgumentError("parity must be :even or :odd, got $parity"))
end

"""
    bond_gates(psi; J=1.0, delta=1.0) -> Vector

One gate per bond, tagged for that bond's own site legs. `delta = 0` gives XX.

Entries may be `nothing` for bonds carrying no Hamiltonian term; `parity_sweep!`
skips those, matching Python's `build_bond_generators`, which leaves untouched
bonds `None`.
"""
bond_gates(psi::SymMPS; J::Float64 = 1.0, delta::Float64 = 1.0) =
    Any[heisenberg_bond_gate(psi[i].inds[2], psi[i + 1].inds[2]; J = J, delta = delta)
        for i in 1:(length(psi) - 1)]

"""
    energy(psi, gates) -> Float64

`Σ_b ⟨ψ|h_b|ψ⟩ / ⟨ψ|ψ⟩` over the bonds carrying a gate.

Divides by the norm bond by bond, so it is a true expectation value even if the
state is not normalised. Canonicalises `psi` as it goes -- exactly as
`sz_expectation` does -- which changes the gauge but not the state.
"""
function energy(psi::SymMPS, gates)
    E = 0.0
    for i in 1:(length(psi) - 1)
        gates[i] === nothing && continue
        canonical!(psi, i)
        f = bond_frame(psi, i)
        th = frame_theta(f)
        num = tensor_inner(th, apply_gate(gates[i], th, f.site_l, f.site_r))
        E += real(num) / real(tensor_inner(th, th))
    end
    return E
end

"""
    _dimer_state_projected(L) -> SymMPS

The dimer covering `⊗_k |singlet⟩_{2k-1,2k}` for `:none` and `:U1`, built by PROJECTING a Néel
state rather than by writing tensors out by hand.

WHY THIS EXISTS. [`dimer_state`](@ref) is `:SU2`-only, and `neel_state`/`product_state` are
guarded off under `:SU2` — so before this there was no state constructible in BOTH a symmetric
and a non-symmetric run, and "same physics, just faster" could not be tested at all. The dimer
covering is representable everywhere, so it is the common start.

HOW: POWER ITERATION ON THE GATE, applied to each odd two-site block directly.

On a pair, `S·S` has eigenvalue `-3/4` on the singlet and `+1/4` on the triplet, and
`|↑↓⟩ = (|s⟩ + |t₀⟩)/√2` has weight on both. Applying `S·S` repeatedly therefore amplifies the
singlet over the triplet by `3ⁿ`, so `n = 60` leaves a relative triplet amplitude `3^-60 ≈
1e-29` — far below double precision. Renormalising each application keeps it away from
overflow; the alternating sign of `(-3/4)ⁿ` only sets a global phase per pair, which no
observable sees.

WHY NOT `parity_sweep!` WITH `exp(-τ S·S)`, WHICH IS THE OBVIOUS ROUTE. Because what
`kls_bond_update` does from a product state DEPENDS ON THE SYMMETRY, and under `:none` — the
mode this function has to serve — it cannot grow rank at all. MEASURED on the `:even` group,
`maxdim = 8`, 60 steps of `τ = -0.5`:

| mode    | kernel                | result                                   |
|---------|-----------------------|------------------------------------------|
| `:none` | kls, every variant    | frozen `[1,1,…]`, 5.0e-01 from target    |
| `:U1`   | kls default           | grows `[2,1,2,…]`, 1.1e-16               |
| `:U1`   | kls `missing_fill=0`  | frozen `[1,1,…]`, 5.0e-01                |
| both    | cbe, either rule      | grows `[2,1,2,…]`, 1e-16                 |

The U(1) growth is ENTIRELY `missing_fill` — a random column seeded into an empty-but-reachable
charge sector. A Néel bond carries one U(1) sector and a singlet needs a second, so `:U1` has
somewhere to put the seed and `:none` (one dense sector) does not. CBE needs no such thing: it
draws directions through the gate and grows in every mode. That contrast is pinned in
`tests/RSVDCBEBondUpdate/test_dimer.jl`. CBE is not callable from here (it lives in
`RSVDCBEBondUpdate`, which depends on this module), hence doing the block update explicitly.

MIND THE PARITY NAMES: `parity_bonds(L, :even)` is `1,3,5,…` and `:odd` is `2,4,6,…`, following
Python's 0-based convention rather than the Julia index. Passing `:odd` while placing gates on
bonds `1,3,5,…` makes every gate in the requested group `nothing`, so the sweep skips every bond
and returns the input UNCHANGED AND UNFLAGGED. That is indistinguishable from the rank freeze
above by looking at the output state, and it is how the freeze above first came to be
"diagnosed" from a run that had in fact done nothing at all.

Splitting the evolved block with `svd` is what supplies the rank: the SVD is free to return two
singular values, which is exactly the `χ=2` a singlet needs. Odd bonds are disjoint, so the
pairs are independent and no Trotter error arises.
"""
function _dimer_state_projected(L::Int)
    L >= 2 && iseven(L) || throw(ArgumentError(
        "dimer state needs an even number of sites >= 2, got $L"))
    psi = neel_state(L)
    g = bond_gates(psi; delta = 1.0)
    for i in 1:2:(L - 1)
        canonical!(psi, i)
        f = bond_frame(psi, i)
        th = frame_theta(f)
        for _ in 1:60
            th = to_concrete(apply_gate(g[i], th, f.site_l, f.site_r))
            nn = sqrt(real(tensor_inner(th, th)))
            nn > 0 || error("dimer projection collapsed at bond $i")
            th = to_concrete((1.0 / nn) * th)
        end
        res = svd(th, (1, 2); cutoff = 1e-14, Nkeep = 4, get_lists = true)
        # ⛔ RESTORE THE BOND TAG. `svd` labels the bond it creates with Telum's DEFAULT (`svdL`),
        # discarding `L,$(i+1)`. Writing that back unretagged hands every caller a state whose ODD
        # bonds -- exactly this loop's range -- are tagged `svdL`, and that is a live landmine
        # rather than a cosmetic one: `move_right!` does `res.S * res.Vd` where `S` is
        # `(svdL, svdR)` and `Vd` is `(svdR, <leg 3 of psi[i]>)`, so an incoming `svdL` on leg 3
        # puts `svdL` into one TLArray twice and Telum asserts
        #     Duplicate TLIndex with non-empty itag: TLIndex(svdL, '+', 0, 0, false)
        # ⚠ IT ONLY BIT `:U1`/`:none`, because `dimer_state` reaches here only when the mode is not
        # `:SU2` -- which is why every BUG result in this repo had been `:SU2` or `:none` and the
        # U(1) arms of `logtime_suite.jl` returned `E = NaN, mv = 0, chi = 0` at every size. The
        # state itself was always CORRECT (with `truncate = false` the U(1) step succeeds at
        # `chi = 8` and neighbouring tags agree, so it contracts fine); only the labelling was
        # wrong. Every other `psi[...] = <svd result>` in `src` already retags -- this was the one
        # that did not.
        bt = "L,$(i + 1)"
        psi[i] = to_concrete(setitag(res.U, 3, bt))
        psi[i + 1] = to_concrete(setitag(to_concrete(res.S * res.Vd), 1, bt))
        psi.center = i + 1
    end
    n = norm(psi)
    n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
    return psi
end
