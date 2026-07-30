# User-facing read-out of the physical and entanglement observables of a state.
#
# Everything here is READ-ONLY in the sense that matters: none of it changes the
# vector the MPS represents. The functions DO move the orthogonality centre (via
# `canonical!`), which is norm- and state-preserving. If you need the centre left
# where it was, pass a `copy(psi)`; the driver records from a copy for exactly
# this reason, so a diagnostic read can never perturb the validated integrator.

"""
    magnetisation(psi) -> Vector{Float64}

`⟨S_z^j⟩` at every site `j = 1:length(psi)`, up spin = `+1/2`.

This is the site-resolved profile behind a light-cone heat-map (stack one row per
time) and the quantity most error metrics are built on. It walks the orthogonality
centre across the chain once, so the whole profile is a single sweep.

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

Schmidt (singular-value) spectrum of the cut at interior `bond`, largest first
and expanded by symmetry-sector degeneracy so its length equals the bond
dimension. These are the numbers behind an entanglement-spectrum plot; the sum
of their squares is the state's norm² across that cut (`1` for a normalised
state).

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

[`bond_spectrum`](@ref) at every interior bond. `spectrum[b]` is the cut between
sites `b` and `b+1`, so `length(spectrum) == length(psi) - 1`. Computed in one
centre sweep. This is the input to the "spectral spectrum" plots -- record it at
a handful of timestamps rather than every step (it is far heavier than the
scalar diagnostics).

```julia
spec = entanglement_spectrum(psi)    # spec[b] = Schmidt values of bond b
```
"""
entanglement_spectrum(psi::SymMPS) =
    [bond_spectrum(psi, b) for b in 1:(length(psi) - 1)]

"""
    bond_energies(psi, gates) -> Vector{Float64}

`⟨ψ|h_b|ψ⟩ / ⟨ψ|ψ⟩` resolved BY BOND, i.e. [`energy`](@ref) without the final sum.
`out[b]` is the bond between sites `b` and `b+1`; bonds whose gate is `nothing` give `0.0`.

THIS IS THE LIGHT-CONE OBSERVABLE FOR A SYMMETRIC RUN, and the reason it exists is that
`magnetisation` cannot play that role under `:SU2`. An `:SU2` state carries total-spin
multiplets, so `⟨Sz_j⟩` is identically zero by symmetry — and the operator is not even exposed
(`local_space(:SU2)` has `(:I, :S)`, no `Sz`), so asking for it throws rather than returning the
zero. `S_j·S_{j+1}` is an SU(2) SCALAR: it is invariant under the symmetry, nonzero, and defined
in every mode. A dimer quench therefore produces a genuine light cone in `:none`, `:U1` and
`:SU2` alike, and the three are directly comparable because it is the same operator.

Like `energy`, this canonicalises `psi` as it goes — the gauge changes, the state does not.
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
