# THE TWO SELECTION RULES OF 1-SITE TDVP + CBE, NAMED.
#
# `tdvp1_cbe.jl` next door holds ONE step. These are its two variants, and they are named
# functions rather than two copies of the algorithm ON PURPOSE:
#
#   tdvp1_cbe_exact_step!   normal (plain) CBE   -- `exact = true`
#   tdvp1_rsvd_cbe_step!    the RSVD extension   -- `exact = false`
#
# WHY NOT TWO IMPLEMENTATIONS. The variants differ in ONE boolean, threaded down to
# `cbe_expand_bond`, which decides how the expansion basis is chosen:
#
#   exact = true    the COMPLETE FUSED BASIS (`full_local_basis`). The preselection sees the
#                   true `H*Theta` and its SVD returns the exact leading directions. This
#                   forms the rank-4 object of size `d*chi`.
#   exact = false   a thin Gaussian SKETCH of that same object, cost `~1.2*budget` instead of
#                   `d*chi`. Forming the rank-4 object is PRECISELY what the sketch exists to
#                   avoid, so this is the method and `exact = true` is its A/B reference.
#
# Everything else -- the sweep, the environments, the zero-site back-step, the truncation --
# is identical. Copying ~250 lines to express one flag would let the copies drift, and the
# whole point of the A/B is that the two paths are otherwise bit-for-bit the same code. Any
# difference in their output must come from the selection rule, and nothing else.
#
# WHICH IS THE DEFAULT: `exact = false`, the sketch. Plain CBE is not a "safer" or "more
# accurate" setting to fall back on -- it is the reference arm, and it costs more. Measured:
# the two agree to every printed digit on all time-evolution arms, and part only where the
# rank cap binds (`dmrg_cbe_rsvd` 7.53e-07 vs `dmrg_cbe` 1.78e-15 at chi=8, gone by chi=16),
# which is evidence the sketch genuinely selects rather than merely inflating rank.

"""
    tdvp1_rsvd_cbe_step!(psi, h, tau; kwargs...) -> TDVP1CBEInfo

One 1-site TDVP + **RSVD-CBE** step (the benchmark arm `1site_tdvp_cbe_rsvd`).

**This is the method.** The bond expansion is chosen from a thin Gaussian sketch of `H*Theta`,
so the rank-4 object is never formed. `tau = -im*dt` is real time, `tau = -dt` imaginary;
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
