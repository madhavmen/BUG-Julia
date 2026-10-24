# TWO-SITE TDVP WITH THE LOSS SPLIT OFF EXACTLY (dissipative plan, Task 3 A1 strong-loss check).
# When Q = Σ_j Q_j is onsite and constant on each site charge sector (pair loss: κ_j on the doublon
# sector), e^{-hQ/4} is a product of exact local factors, and a Strang step leaves TDVP2 only the
# Hermitian H. The baseline beside direct non-Hermitian TDVP2 on G = H - (i/2)Q.

"""
    scale_site_sectors!(psi, j, f) -> psi

Multiply each site-leg sector `s` of `psi[j]` by `f(s)`, in place and not renormalised: an
onsite operator constant on each charge sector. Built by sector projection, as
[`apply_sz!`](@ref), since contracting an operator onto the site leg re-sorts its sectors. Under
`:none` the site leg is one sector, so it is refused.
"""
function scale_site_sectors!(psi::SymMPS, j::Int, f)
    symmetry_mode() === :none && throw(ArgumentError(
        "scale_site_sectors! needs a charge-resolved site leg; under :none it is one sector"))
    canonical!(psi, j)
    A = psi[j]
    terms = Any[]
    for (s, _) in A.spaces[2]
        B = to_concrete(getsub(A, 2, t -> t == s ? Colon() : nothing; preserve_space = true))
        isempty(B.qlabels) || push!(terms, to_concrete(f(s) * B))
    end
    psi[j] = reduce((a, b) -> to_concrete(a + b), terms)
    return psi
end

"""
    tdvp2_split_loss_step!(psi, h, tau, qloc; kwargs...) -> TDVP2Info

One Strang step `e^{-dt Q/4} TDVP2_H(tau) e^{-dt Q/4}` of `G = -iH - Q/2`, `tau = -im*dt`, in
place. `h` is the HERMITIAN `H`; `qloc[j](s)` is `Q_j` on site sector `s` (e.g.
[`doublon_loss_local`](@ref)). Second order in `dt` through the splitting, on top of TDVP2's own
projection error; `kwargs` go to [`tdvp2_step!`](@ref).
"""
function tdvp2_split_loss_step!(psi::SymMPS, h::Union{XXZChain, MPO}, tau::Number,
                                qloc::AbstractVector; kwargs...)
    L = length(psi)
    length(qloc) == L || throw(DimensionMismatch("qloc has $(length(qloc)) sites, state has $L"))
    t = ComplexF64(tau)
    # e^{(tau/2)(-i/2) Q_j}: tau * (-i/2) Q is the loss part of tau * (H - (i/2) Q)
    loss!() = for j in 1:L
        scale_site_sectors!(psi, j, s -> exp(-0.25im * t * qloc[j](s)))
    end
    loss!()
    info = tdvp2_step!(psi, h, t; kwargs...)
    loss!()
    return info
end
