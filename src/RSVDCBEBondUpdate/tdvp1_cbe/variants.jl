# THE TWO SELECTION RULES OF 1-SITE TDVP + CBE, NAMED.
#
# `tdvp1_cbe.jl` next door holds ONE step. These are its two variants, and they are named
# functions rather than two copies of the algorithm ON PURPOSE:
#
#   tdvp1_cbe_exact_step!   normal (plain) CBE   -- `exact = true`
#   tdvp1_rsvd_cbe_step!    the RSVD extension   -- `exact = false`
#


"""
    tdvp1_rsvd_cbe_step!(psi, h, tau; kwargs...) -> TDVP1CBEInfo

One 1-site TDVP + **RSVD-CBE** step (the benchmark arm `1site_tdvp_cbe_rsvd`).

**This is the method.** The bond expansion is chosen from a thin Gaussian sketch of `H*Theta`,
the sketch probes the projector P_perp(H Theta), which forms the rank-4 object. `tau = -im*dt` is real time, `tau = -dt` imaginary;
nothing in the step cares which.

Identical to [`tdvp1_cbe_step!`](@ref) with `exact = false`, which is that function's default —
this name exists so the two selection rules are discoverable by name rather than by a boolean.
"""
tdvp1_rsvd_cbe_step!(psi::SymMPS, h::XXZChain, tau::ComplexF64; kwargs...) =
    tdvp1_cbe_step!(psi, h, tau; exact = false, kwargs...)

tdvp1_rsvd_cbe_step!(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    tdvp1_rsvd_cbe_step!(psi, h, ComplexF64(tau); kwargs...)

"""
    tdvp1_cbe_exact_step!(psi, h, tau; kwargs...) -> TDVP1CBEInfo

One 1-site TDVP + **plain (exact) CBE** step (the benchmark arm `1site_tdvp_cbe`).

The expansion uses the complete fused basis instead of a sketch, so it forms the rank-4
`H*Theta` — the cost the RSVD variant exists to avoid. **This is the A/B reference arm, not a
better default and not a fallback.** Reach for it to check that the sketch is selecting the
right directions; ship [`tdvp1_rsvd_cbe_step!`](@ref).

Identical to [`tdvp1_cbe_step!`](@ref) with `exact = true`.
"""
tdvp1_cbe_exact_step!(psi::SymMPS, h::XXZChain, tau::ComplexF64; kwargs...) =
    tdvp1_cbe_step!(psi, h, tau; exact = true, kwargs...)

tdvp1_cbe_exact_step!(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    tdvp1_cbe_exact_step!(psi, h, ComplexF64(tau); kwargs...)
