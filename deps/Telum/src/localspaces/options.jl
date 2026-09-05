# ── Option types for getLocalSpace ──────────────────────────────────────────
#
# Each concrete option struct encodes the valid configuration for one class
# of physical local space.  The three classes mirror the three branches in
# the original TLArray Matlab code:
#
#   SpinOptions        – pure spin models (no charge degree of freedom)
#   FermionOptions     – spinless (single-component) fermions
#   FermionSOptions    – spinful (multi-component) fermions
#
# Keeping the options in separate types means that
#   • every (option-type, function) pair can be dispatched independently,
#   • invalid combinations are caught at construction time or via the type
#     system rather than as run-time errors, and
#   • users can add a new space type by defining a new option struct and the
#     two generic hooks getSymmetryOps / getIROPs for it (see getLocalSpace.jl).
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Abstract base ────────────────────────────────────────────────────────────

"""
    LocalSpaceOptions

Abstract supertype for all option structs passed to `getLocalSpace`.
"""
abstract type LocalSpaceOptions end

# ── Spin ─────────────────────────────────────────────────────────────────────

"""
    SpinOptions(; symmetry, spin)

Options for a pure spin-`spin` local space with `symmetry` ∈ {`:SU2`, `:U1`, `nothing`, …}.

# Fields
- `symmetry :: Union{Symbol, Nothing}` – symmetry group for the spin degree of
  freedom, or `nothing` for an unsymmetrized local space.
- `spin     :: Int`    – 2 * spin quantum number (e.g. `1`, `2`, `3`).
"""
struct SpinOptions <: LocalSpaceOptions
    symmetry::Union{Symbol, Nothing}
    spin::Int 
end

# ── Spinless fermion ──────────────────────────────────────────────────────────

"""
    FermionOptions(N::Int=1, charge_symm=:U1, channel_symm=nothing)

Options for a spinless fermionic local space with `N` channels.

Each symmetry option is either `nothing`, a `Symbol`, or a vector of
`(symmetry_symbol, channel_indices)` tuples. A plain symbol is applied to all
channels. Charge symmetry is currently Abelian only, so `:U1` is supported.

For compatibility, `FermionOptions(U1)` is equivalent to
`FermionOptions(1, :U1, nothing)`.

# Fields
- `nchannels` – number of channels / flavors. The local Hilbert-space
  dimension is `2^nchannels`.
- `charge_symm` – Abelian charge symmetries on selected channel groups.
- `channel_symm` – channel / flavor symmetry groups on selected channels.
  Symbols of the form `:SU3`, `:SU4`, ... request the corresponding SU(N)
  channel symmetry; `nothing` disables channel symmetry.
"""
struct FermionOptions <: LocalSpaceOptions
    nchannels::Int
    charge_symm::Union{Vector{Tuple{Symbol, Vector{Int}}}, Nothing}
    channel_symm::Union{Vector{Tuple{Symbol, Vector{Int}}}, Nothing}
end

function FermionOptions(N::Int=1, charge_symm=:U1, channel_symm=nothing)
    @assert N > 0 "Number of channels must be positive"
    charge_symm_ = standardize_option(charge_symm, N)
    channel_symm_ = standardize_option(channel_symm, N)
    return FermionOptions(N, charge_symm_, channel_symm_)
end

function FermionOptions(symmetry::Type{<:Symmetry})
    @assert symmetry == U1 "FermionOptions currently only supports U1 charge symmetry"
    return FermionOptions(1, :U1, nothing)
end

# ── Spinful fermion ───────────────────────────────────────────────────────────

"""
    FermionSOptions(N::Int, charge_symm=nothing, spin_symm=nothing, channel_symm=nothing)

Options for a spinful (multi-component) fermionic site.

Each symmetry option is either `nothing`, a `Symbol`, or a vector of
`(symmetry_symbol, channel_indices)` tuples.  A plain symbol is applied to all
channels, so `FermionSOptions(3, :U1, :SU2, :SU3)` is equivalent to
`FermionSOptions(3, [(:U1, [1, 2, 3])], [(:SU2, [1, 2, 3])],
[(:SU3, [1, 2, 3])])`.

# Fields
- `nchannels` – number of channels / flavors.  The local Hilbert-space
  dimension is `4^nchannels`.
- `charge_symm` – charge symmetries on selected channel groups.  Supported
  symbols are currently `:U1` and `:SU2`; `nothing` disables charge symmetry.
- `spin_symm` – spin symmetries on selected channel groups.  Supported symbols
  are currently `:U1` and `:SU2`; `nothing` disables spin symmetry.
- `channel_symm` – channel / flavor symmetry groups on selected channels.
  Symbols of the form `:SU3`, `:SU4`, ... request the corresponding SU(N)
  channel symmetry; `nothing` disables channel symmetry.
"""
struct FermionSOptions <: LocalSpaceOptions
    nchannels   ::Int
    charge_symm ::Union{Vector{Tuple{Symbol, Vector{Int}}}, Nothing}
    spin_symm   ::Union{Vector{Tuple{Symbol, Vector{Int}}}, Nothing}
    channel_symm::Union{Vector{Tuple{Symbol, Vector{Int}}}, Nothing}
end

function standardize_option(opt, N)
    if opt === nothing return nothing
    elseif opt isa Symbol return [(Symbol(opt), collect(1:N))]
    else @assert opt isa Vector{Tuple{Symbol, Vector{Int}}} "Option must be a Symbol or a Vector{Tuple{Symbol, Vector{Int}}}"
        return opt
    end
end

function FermionSOptions(N::Int, charge_symm=nothing, spin_symm=nothing, channel_symm=nothing)
    charge_symm_ = standardize_option(charge_symm, N)
    spin_symm_ = standardize_option(spin_symm, N)
    channel_symm_ = standardize_option(channel_symm, N)
    return FermionSOptions(N, charge_symm_, spin_symm_, channel_symm_)
end
