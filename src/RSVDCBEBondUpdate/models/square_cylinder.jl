# ── the square lattice on a cylinder, as a coupling matrix ──
# The column-major snake wraps the short direction, so an MPS cut severs only Ly bonds and the
# longest-range coupling is Ly, not Lx.
# The periodic wrap is a separate coupling at chain distance Ly - 1; at Ly = 2 it coincides with the
# rung bond, so periodic_y is refused below Ly = 3.

"""
    square_cylinder_couplings(Lx, Ly; J=1.0, J2=0.0, periodic_y=true) -> Matrix{Float64}

`J[i,j]` for an `Lx x Ly` square lattice on a cylinder, under the column-major snake
`site(x, y) = (x - 1) * Ly + y`. Open in `x`, periodic in `y` when `periodic_y`.
Bonds: rungs (distance 1), wrap (distance Ly - 1), legs (distance Ly) and, when `J2 != 0`, diagonals
(next-nearest neighbours, which set the MPO bond dimension).
Returns the strictly upper triangle, which [`pair_mpo`](@ref) reads.
"""
function square_cylinder_couplings(Lx::Int, Ly::Int; J::Float64 = 1.0, J2::Float64 = 0.0,
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
    function put!(a::Int, b::Int, c::Float64 = J)
        i, j = minmax(a, b)
        Jm[i, j] += c
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
            if J2 != 0.0
                # Diagonals: each (x, y) -> (x+1, y+-1) is a distinct bond, so counted once.
                for y in 1:Ly, dy in (1, -1)
                    yy = periodic_y ? mod1(y + dy, Ly) : y + dy
                    1 <= yy <= Ly || continue
                    put!(site(x, y), site(x + 1, yy), J2)
                end
            end
        end
    end
    return Jm
end

"""
    square_cylinder_mpo(Lx, Ly; J=1.0, J2=0.0, periodic_y=true) -> MPO

Heisenberg `H = J sum S_i . S_j + J2 sum S_i . S_j` (second sum over next-nearest neighbours) on the
cylinder, built through [`pair_mpo`](@ref) and [`spin_vertices`](@ref). All symmetry modes share this
code path; under `:U1` the charged Sp/Sm vertices are used.

The MPO bond dimension grows with `Ly`, not `Lx`, since at most `Ly` bonds are open at any cut.
Check `mpo_virtual_dims` on a new geometry rather than assuming it.
"""
square_cylinder_mpo(Lx::Int, Ly::Int; J::Float64 = 1.0, J2::Float64 = 0.0,
                    periodic_y::Bool = true) =
    pair_mpo(Lx * Ly, square_cylinder_couplings(Lx, Ly; J = J, J2 = J2, periodic_y = periodic_y),
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
