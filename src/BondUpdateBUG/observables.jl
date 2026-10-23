# Read-out of physical and entanglement observables. None of it changes the represented vector, but
# it moves the orthogonality centre (`canonical!`, which preserves norm and state). Pass `copy(psi)`
# if the centre must stay put; the driver records from a copy for this reason.

"""
    magnetisation(psi) -> Vector{Float64}

`⟨S_z^j⟩` at every site `j`, up spin = `+1/2`. This is the site-resolved profile behind a light-cone
heat-map and the basis of most error metrics. One centre sweep gives the whole profile.

```julia
mz = magnetisation(psi)          # length N, one ⟨Sz⟩ per site
```
"""
magnetisation(psi::SymMPS) = [sz_expectation(psi, j) for j in 1:length(psi)]

"""
    center_bond(psi) -> Int

Index of the central interior bond, `length(psi) ÷ 2` (the cut between sites
`N÷2` and `N÷2 + 1`). Interior bonds are numbered `1 : length(psi)-1`, the same
numbering [`bond_dims`](@ref) uses.
"""
center_bond(psi::SymMPS) = length(psi) ÷ 2

"""
    center_bond_dimension(psi) -> Int

Bond dimension of the central cut -- the usual headline entanglement number,
equal to `bond_dims(psi)[center_bond(psi)]`.
"""
center_bond_dimension(psi::SymMPS) = leg_dim(psi[center_bond(psi)], 3)

"""
    bond_spectrum(psi, bond = center_bond(psi)) -> Vector{Float64}

Schmidt values of the cut at interior `bond`, largest first, expanded by symmetry-sector degeneracy
so the length equals the bond dimension. Their squares sum to the norm² across the cut (1 for a
normalised state).

```julia
s_center = bond_spectrum(psi)        # spectrum of the central cut
s_3      = bond_spectrum(psi, 3)     # spectrum of the cut between sites 3 and 4
```
"""
function bond_spectrum(psi::SymMPS, bond::Int = center_bond(psi))
    1 <= bond <= length(psi) - 1 || throw(ArgumentError(
        "bond $bond is outside the interior-bond range 1:$(length(psi) - 1)"))
    canonical!(psi, bond)
    # Split (left link, site) from the right link: the right-link singular values
    # ARE the Schmidt spectrum of the cut just right of `bond`. cutoff=0 keeps the
    # full spectrum; get_lists exposes it without densifying the symmetric tensor.
    res = svd(psi[bond], (1, 2); cutoff = 0.0, get_lists = true)
    svals = Float64[]
    for (sigma, degeneracy, _, _) in res.kept_list
        for _ in 1:Int(degeneracy)          # multiplet size (1 under U(1))
            push!(svals, Float64(sigma))
        end
    end
    return sort!(svals; rev = true)
end

"""
    entanglement_spectrum(psi) -> Vector{Vector{Float64}}

[`bond_spectrum`](@ref) at every interior bond, from one centre sweep: `spectrum[b]` is the cut
between sites `b` and `b+1`. It is far heavier than the scalar diagnostics, so record it at a few
timestamps.

```julia
spec = entanglement_spectrum(psi)    # spec[b] = Schmidt values of bond b
```
"""
entanglement_spectrum(psi::SymMPS) =
    [bond_spectrum(psi, b) for b in 1:(length(psi) - 1)]

"""
    bond_energies(psi, gates) -> Vector{Float64}

`⟨ψ|h_b|ψ⟩ / ⟨ψ|ψ⟩` resolved by bond, i.e. [`energy`](@ref) without the final sum. `out[b]` is the
bond between sites `b` and `b+1`; bonds with a `nothing` gate give `0.0`.

This is the light-cone observable for symmetric runs. Under `:SU2` `⟨Sz_j⟩` is identically zero and
`Sz` is not exposed, but `S_j·S_{j+1}` is an SU(2) scalar, nonzero and defined in every mode, so
`:none`, `:U1` and `:SU2` runs are directly comparable. Like `energy`, it canonicalises `psi`: the
gauge changes, not the state.
"""
function bond_energies(psi::SymMPS, gates)
    out = zeros(Float64, length(psi) - 1)
    for i in 1:(length(psi) - 1)
        gates[i] === nothing && continue
        canonical!(psi, i)
        f = bond_frame(psi, i)
        th = frame_theta(f)
        num = tensor_inner(th, apply_gate(gates[i], th, f.site_l, f.site_r))
        out[i] = real(num) / real(tensor_inner(th, th))
    end
    return out
end
