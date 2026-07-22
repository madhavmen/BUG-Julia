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
    parity_sweep!(psi, gates, parity, tau; kwargs...) -> NamedTuple

Advance every bond of one commuting group by `tau`, in place. Returns
`(; aug_k, aug_l, discarded)`, each the **maximum** over the group's bonds --
the diagnostics Python's `parity_sweep` returns.

`kwargs` forward to [`kls_bond_update`](@ref): `maxdim`, `trunc_thresh`,
`augment`, `missing_fill`, `maxiter`, `tol`, `rng`.

No re-canonicalisation after a bond: `kls_bond_update` returns `left_core`
already a left isometry with the centre in `right_core`, so setting
`psi.center = i + 1` is exact, and the next bond of the group is one move to the
right.
"""
function parity_sweep!(psi::SymMPS, gates, parity::Symbol, tau::ComplexF64; kwargs...)
    aug_k = 0
    aug_l = 0
    discarded = 0.0
    for i in parity_bonds(length(psi), parity)
        gates[i] === nothing && continue
        canonical!(psi, i)
        f = bond_frame(psi, i)
        r = kls_bond_update(f, gates[i], tau; kwargs...)
        psi[i] = r.left_core
        psi[i + 1] = r.right_core
        psi.center = i + 1
        aug_k = max(aug_k, r.aug_k)
        aug_l = max(aug_l, r.aug_l)
        discarded = max(discarded, r.discarded)
    end
    return (; aug_k, aug_l, discarded)
end

parity_sweep!(psi::SymMPS, gates, parity::Symbol, tau::Number; kwargs...) =
    parity_sweep!(psi, gates, parity, ComplexF64(tau); kwargs...)

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
