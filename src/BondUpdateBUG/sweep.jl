# Odd/even Trotter sweep over the bonds, mirroring two_site_bug/scheme.py::parity_bonds / kls_bond /
# parity_sweep. Bonds of one parity act on disjoint site pairs, so each group is an exact factor of
# the Trotter step; the splitting error lies only between the groups.

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

The dimer covering for `:none` and `:U1`, made by projecting a Neel state onto singlets.
`dimer_state` is `:SU2`-only and `product_state` is guarded off under `:SU2`, so this is the start
state buildable in every mode. Each odd two-site block is refined by power iteration on `S·S` (the
singlet grows 3x over the triplet per application); the SVD supplies the rank-2 a singlet needs. Odd
bonds are disjoint, so no Trotter error arises.

Parity is 0-based as in Python: `parity_bonds(L, :even)` gives bonds 1,3,5,... A wrong parity
silently skips every gate and returns the input unchanged, which looks like a rank freeze.
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
        # Restore the bond tag: svd labels the new bond `svdL`, and `move_right!` then sees that tag
        # on two legs, which Telum rejects as a duplicate TLIndex. This fired only under
        # `:U1`/`:none`, since `dimer_state` reaches here only outside `:SU2`; every other svd
        # write-back already retags.
        bt = "L,$(i + 1)"
        psi[i] = to_concrete(setitag(res.U, 3, bt))
        psi[i + 1] = to_concrete(setitag(to_concrete(res.S * res.Vd), 1, bt))
        psi.center = i + 1
    end
    n = norm(psi)
    n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
    return psi
end
