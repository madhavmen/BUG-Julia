# ── the KAGOME lattice on a cylinder, as a coupling matrix ───────────────────
#
# Corner-sharing triangles: three sublattices A, B, C per unit cell, every site 4-coordinate,
# and the frustration that makes the Heisenberg antiferromagnet here a spin-liquid candidate
# rather than a Neel magnet. Like `square_cylinder.jl` this file builds ONLY the coupling matrix;
# the operator is `pair_mpo`, which already handles arbitrary `J[i,j]` including the charged
# rank-3 vertices SU(2) needs.
#
# GEOMETRY, and it is worth writing down because every kagome convention in the literature
# differs by a relabelling. Lattice vectors `a1 = (2, 0)`, `a2 = (1, sqrt3)`, sublattices at
#
#     A = (0, 0)        B = (1, 0) = a1/2        C = (1/2, sqrt3/2) = a2/2
#
# so that all six neighbour bonds have length exactly 1:
#
#     within the cell (the UP triangle):   A-B,  B-C,  C-A
#     to the next cells (DOWN triangle):   B(i,j)-A(i+1,j)
#                                          C(i,j)-A(i,j+1)
#                                          C(i,j)-B(i-1,j+1)
#
# Six bonds per unit cell over three sites is coordination 4, which is the check that the bond
# list is complete -- `test_kagome.jl` asserts it rather than trusting this comment.
#
# ⛔ 3 SITES PER UNIT CELL MEANS THE SITE COUNT IS ALWAYS A MULTIPLE OF 3. There is no 16-site
# kagome cluster with whole unit cells; 12, 18 and 24 are the reachable sizes near it. Asking for
# an arbitrary count and getting silently rounded is how a "kagome" result ends up being measured
# on a lattice nobody specified, so `kagome_cylinder_couplings` takes CELLS, never a site count.
#
# ⛔ AND THE WRAP DIRECTION IS THE SHORT ONE, for the same reason as the square cylinder: the MPS
# cut severs O(Ly) bonds, so wrapping `a2` (circumference `Ly` cells) and leaving `a1` open keeps
# the entanglement bounded by `Ly` and not by the total length.

"""
    kagome_cylinder_couplings(Lx, Ly; J=1.0, periodic_y=true) -> Matrix{Float64}

`J[i,j]` for a kagome cylinder of `Lx x Ly` unit cells (`3*Lx*Ly` sites), open along `a1`,
periodic along `a2` when `periodic_y`.

Site index: `site(i, j, s) = 3*((i-1)*Ly + (j-1)) + s` with `s = 1, 2, 3` for `A, B, C` — i.e. the
three sublattices of a cell are ADJACENT in the chain, and cells advance along the circumference
first. That ordering is what keeps the longest-range coupling `O(Ly)` rather than `O(Lx)`.

Returns the strictly upper triangle, which is what [`pair_mpo`](@ref) reads.

⚠️ `periodic_y` needs `Ly >= 3`: at `Ly = 2` the wrap bonds coincide with the intra-cell ones and
adding both DOUBLES those couplings — a factor of two in `H` that reads as a physics result. Pass
`periodic_y = false` for a kagome strip.
"""
function kagome_cylinder_couplings(Lx::Int, Ly::Int; J::Float64 = 1.0,
                                   periodic_y::Bool = true)
    Lx >= 1 || throw(ArgumentError("Lx must be >= 1, got $Lx"))
    Ly >= 1 || throw(ArgumentError("Ly must be >= 1, got $Ly"))
    (periodic_y && Ly < 3) && throw(ArgumentError(
        "kagome periodic_y needs Ly >= 3, got $Ly: at Ly = 2 the wrap bonds coincide with the " *
        "intra-cell bonds and adding both doubles them. Pass periodic_y = false for a strip."))
    L = 3 * Lx * Ly
    site(i, j, s) = 3 * ((i - 1) * Ly + (j - 1)) + s
    Jm = zeros(Float64, L, L)
    # `+=`, not `=`: a geometry that proposes the same bond twice must show up as a DOUBLED
    # coupling in a test rather than being silently overwritten.
    function put!(a::Int, b::Int)
        p, q = minmax(a, b)
        p == q && throw(ArgumentError("kagome: self-bond at site $p -- geometry is wrong"))
        Jm[p, q] += J
        return nothing
    end
    wrap(j) = periodic_y ? mod1(j, Ly) : j
    for i in 1:Lx, j in 1:Ly
        A, B, C = site(i, j, 1), site(i, j, 2), site(i, j, 3)
        put!(A, B); put!(B, C); put!(C, A)                 # the UP triangle
        i < Lx && put!(B, site(i + 1, j, 1))               # B -> A of the next cell along a1
        if periodic_y || j < Ly
            put!(C, site(i, wrap(j + 1), 1))               # C -> A of the next cell along a2
        end
        if (periodic_y || j < Ly) && i > 1
            put!(C, site(i - 1, wrap(j + 1), 2))           # C -> B of the cell up-and-back
        end
    end
    return Jm
end

"""
    kagome_cylinder_mpo(Lx, Ly; J=1.0, periodic_y=true) -> MPO

Heisenberg `H = J Σ_<ij> S_i · S_j` on the kagome cylinder, in whatever symmetry mode is active.
Routed through [`pair_mpo`](@ref) / [`spin_vertices`](@ref), so `:none`, `:U1` and `:SU2` all use
the same code path.
"""
kagome_cylinder_mpo(Lx::Int, Ly::Int; J::Float64 = 1.0, periodic_y::Bool = true) =
    pair_mpo(3 * Lx * Ly,
             kagome_cylinder_couplings(Lx, Ly; J = J, periodic_y = periodic_y),
             spin_vertices(3 * Lx * Ly))

"""
    kagome_triangles(Lx, Ly; periodic_y=true) -> Vector{NTuple{3,Int}}

The corner-sharing triangles, up and down, as site triples.

⛔ THIS IS THE SPIN-LIQUID DIAGNOSTIC, not decoration. On kagome the Heisenberg ground state is
expected to show NO magnetic order and NEAR-UNIFORM bond energies: a `<S_i.S_j>` that is uniform
across the triangles, with `<S_i.S_j> ~ -0.22` per bond and no static moment, is the signature to
look for. A VALENCE-BOND-SOLID instead shows the bonds splitting into strong/weak sets, and a
Neel-like state shows a finite staggered moment. Reporting the bond-energy SPREAD over these
triangles is what distinguishes them.
"""
function kagome_triangles(Lx::Int, Ly::Int; periodic_y::Bool = true)
    site(i, j, s) = 3 * ((i - 1) * Ly + (j - 1)) + s
    wrap(j) = periodic_y ? mod1(j, Ly) : j
    out = NTuple{3, Int}[]
    for i in 1:Lx, j in 1:Ly
        push!(out, (site(i, j, 1), site(i, j, 2), site(i, j, 3)))          # UP triangle
        # DOWN triangle: B(i,j) - A(i+1,j) - C(i+1,j-1).
        #
        # ⛔ DERIVED FROM THE NEIGHBOUR LISTS, NOT GUESSED. `B(i,j)` neighbours
        # {A(i,j), C(i,j), A(i+1,j), C(i+1,j-1)} and `A(i+1,j)` neighbours
        # {B(i+1,j), C(i+1,j), B(i,j), C(i+1,j-1)}; their ONLY common neighbour is C(i+1,j-1).
        # A first version used `A(i,j+1)` here, which is not adjacent to `B(i,j)` at all -- the
        # triangle test caught it by checking that all three pairs are real bonds, which is the
        # whole reason that check exists.
        if i < Lx && (periodic_y || j > 1)
            push!(out, (site(i, j, 2), site(i + 1, j, 1), site(i + 1, wrap(j - 1), 3)))
        end
    end
    return out
end

"""
    kagome_bonds(Lx, Ly; periodic_y=true) -> Vector{Tuple{Int,Int}}

The bond list, matching [`kagome_cylinder_couplings`](@ref) entry for entry.
"""
function kagome_bonds(Lx::Int, Ly::Int; periodic_y::Bool = true)
    Jm = kagome_cylinder_couplings(Lx, Ly; periodic_y = periodic_y)
    L = size(Jm, 1)
    return [(i, j) for i in 1:L for j in (i + 1):L if Jm[i, j] != 0.0]
end
