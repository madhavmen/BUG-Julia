# ── the kagome lattice on a cylinder, as a coupling matrix ──
# Three sites per unit cell, every site with coordination 4. The site count is always a multiple of 3,
# so the builder takes unit cells, never a site count.
# The short direction a2 is wrapped, which keeps the MPS cut at O(Ly) bonds.

"""
    kagome_cylinder_couplings(Lx, Ly; J=1.0, periodic_y=true) -> Matrix{Float64}

`J[i,j]` for a kagome cylinder of `Lx x Ly` unit cells (`3*Lx*Ly` sites), open along `a1` and
periodic along `a2` when `periodic_y`. Site index `site(i, j, s) = 3*((i-1)*Ly + (j-1)) + s` with
`s = 1, 2, 3` for A, B, C, so a cell's sublattices are adjacent and the longest coupling is O(Ly).
`periodic_y` needs `Ly >= 3`: at `Ly = 2` the wrap bonds coincide with intra-cell bonds and double them.
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
        # DOWN triangle B(i,j) - A(i+1,j) - C(i+1,j-1), derived from the neighbour lists. A version using
        # A(i,j+1) is not adjacent to B(i,j); the triangle test (all three pairs must be bonds) caught it.
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
