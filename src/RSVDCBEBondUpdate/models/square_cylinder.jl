# ── the square lattice on a cylinder, as a coupling matrix ───────────────────
#
# An MPS is a 1D object, so a 2D lattice reaches it only through an ORDERING of its sites, and
# every 2D result is a statement about that ordering as much as about the physics. This file
# builds the coupling matrix and nothing else: the operator is then `pair_mpo`, which already
# handles arbitrary `J[i,j]` including the charged rank-3 vertices SU(2) needs.
#
# ⛔ THE COLUMN-MAJOR SNAKE IS NOT A STYLE CHOICE. Wrapping the SHORT direction (the
# circumference `Ly`) is what keeps the MPS bond cutting only `Ly` bonds instead of `Lx`, and it
# is why cylinders are run with `Ly << Lx`. Numbering down each rung in turn puts the two
# neighbours of a site at distance 1 (same rung) and `Ly` (next rung) -- so the longest-range
# coupling in the chain is `Ly`, and the entanglement across a cut grows with `Ly`, not with the
# total site count. Numbering along the LONG direction instead would put every rung neighbour at
# distance `Lx`, and the bond dimension would blow up for the same physics.
#
# ⛔ AND THE WRAP IS A SEPARATE COUPLING, NOT A RE-INDEXING. On a cylinder rung `y = Ly` bonds
# back to `y = 1`. That bond has chain distance `Ly - 1`, which is the LONGEST-range term in the
# model and the one most easily dropped by an off-by-one. `Ly = 2` is the trap: there the wrap
# bond `(1,2)` coincides with the rung bond `(1,2)` and adding both DOUBLES it, which is why
# `periodic_y` is refused below for `Ly < 3`.

"""
    square_cylinder_couplings(Lx, Ly; J=1.0, periodic_y=true) -> Matrix{Float64}

`J[i,j]` for a `Lx x Ly` square lattice wrapped into a cylinder, under the column-major snake
`site(x, y) = (x - 1) * Ly + y`.

Bonds, all with the same `J`:

  * **rung** `(x, y) - (x, y+1)`, chain distance 1
  * **wrap** `(x, Ly) - (x, 1)` when `periodic_y`, chain distance `Ly - 1`
  * **leg**  `(x, y) - (x+1, y)`, chain distance `Ly`

Open in `x`, periodic in `y` -- the standard cylinder. Returns the STRICTLY UPPER triangle, which
is what [`pair_mpo`](@ref) reads; the lower triangle stays zero so no bond is counted twice.

⚠️ `periodic_y` is refused for `Ly < 3`. At `Ly = 2` the wrap bond is the rung bond, and adding
both silently doubles that coupling -- a factor of two in the Hamiltonian that looks like a
physics result rather than a geometry bug.
"""
function square_cylinder_couplings(Lx::Int, Ly::Int; J::Float64 = 1.0,
                                   periodic_y::Bool = true)
    Lx >= 1 || throw(ArgumentError("Lx must be >= 1, got $Lx"))
    Ly >= 2 || throw(ArgumentError("Ly must be >= 2, got $Ly"))
    (periodic_y && Ly < 3) && throw(ArgumentError(
        "periodic_y needs Ly >= 3, got $Ly: at Ly = 2 the wrap bond IS the rung bond and " *
        "adding both doubles it. Pass periodic_y = false for a two-leg ladder."))
    L = Lx * Ly
    site(x, y) = (x - 1) * Ly + y
    Jm = zeros(Float64, L, L)
    # ⚠ WRITTEN OUT, NOT AS `(i, j = minmax(a, b); ...)`. That one-liner is ambiguous: Julia can
    # read it as the TUPLE `(i, (j = ...))` rather than a destructuring assignment, which leaves
    # `i` undefined and `j` holding the pair. `+=` rather than `=` so a geometry that proposes
    # the same bond twice shows up as a doubled coupling in a test instead of being silently
    # overwritten -- that is exactly the `Ly = 2` wrap trap the docstring refuses.
    function put!(a::Int, b::Int)
        i, j = minmax(a, b)
        Jm[i, j] += J
        return nothing
    end
    for x in 1:Lx
        for y in 1:(Ly - 1)
            put!(site(x, y), site(x, y + 1))          # rung
        end
        periodic_y && put!(site(x, Ly), site(x, 1))   # wrap
        if x < Lx
            for y in 1:Ly
                put!(site(x, y), site(x + 1, y))      # leg
            end
        end
    end
    return Jm
end

"""
    square_cylinder_mpo(Lx, Ly; J=1.0, periodic_y=true) -> MPO

Heisenberg `H = J Σ_<ij> S_i · S_j` on the cylinder, in whatever symmetry mode is active.

Built through [`pair_mpo`](@ref) and [`spin_vertices`](@ref), so `:none`, `:U1` and `:SU2` all
route through the same code -- under `:SU2` that is the single `S`/`S'` vertex, under `:U1` the
`Sz Sz` diagonal plus the two CHARGED `Sp`/`Sm` vertices that `long_range_mpo` refuses.

⚠️ THE MPO BOND DIMENSION GROWS WITH `Ly`, NOT WITH `Lx`. `pair_mpo` carries one transport channel
per open bond, and under the column-major snake at most `Ly` bonds are open at any cut. That is
the whole reason for wrapping the short direction, and it is worth checking with
`mpo_virtual_dims` on any new geometry rather than assuming.
"""
square_cylinder_mpo(Lx::Int, Ly::Int; J::Float64 = 1.0, periodic_y::Bool = true) =
    pair_mpo(Lx * Ly, square_cylinder_couplings(Lx, Ly; J = J, periodic_y = periodic_y),
             spin_vertices(Lx * Ly))

"""
    square_cylinder_bonds(Lx, Ly; periodic_y=true) -> Vector{Tuple{Int,Int}}

The bond list, for tests and for anything that needs to iterate bonds rather than read `J[i,j]`.
Same ordering the couplings are built in: rung, wrap, leg, column by column.
"""
function square_cylinder_bonds(Lx::Int, Ly::Int; periodic_y::Bool = true)
    site(x, y) = (x - 1) * Ly + y
    out = Tuple{Int, Int}[]
    for x in 1:Lx
        for y in 1:(Ly - 1)
            push!(out, minmax(site(x, y), site(x, y + 1)))
        end
        periodic_y && Ly >= 3 && push!(out, minmax(site(x, Ly), site(x, 1)))
        if x < Lx
            for y in 1:Ly
                push!(out, minmax(site(x, y), site(x + 1, y)))
            end
        end
    end
    return out
end
