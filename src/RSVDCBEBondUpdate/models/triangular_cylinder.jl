# ── the triangular lattice on a cylinder, as a coupling matrix ──
# `:yc` and `:xc` are different Hamiltonians, not relabellings; `:xc` (arXiv:2209.00739) needs an even
# circumference and admits K = (4pi/3, 0) at every even C.
# Check the geometry with `triangular_coordination`: a diagonal on the wrong corner still runs, but it
# builds the mirror lattice with the same isotropic energy.

"""
The three NEAREST-neighbour steps of the YC wrapping, as `(dx, dy)` in the 120-degree integer
basis, `|n e_x + m e_y|^2 = n^2 + m^2 - n m = 1`. Only one representative per `+-` pair is listed:
the loop visits every site, so adding the negatives too would put each bond in twice.
"""
const _TRI_NN  = ((0, 1), (1, 0), (1, 1))

"The three NEXT-nearest steps of the YC wrapping, `|.|^2 = 3`, same one-per-pair convention."
const _TRI_NNN = ((1, -1), (2, 1), (1, 2))

"""
The three NEAREST-neighbour steps of the XC wrapping, as `(dX, dY)` in the doubled-Cartesian
coordinates of the header (`X = 2x`, `Y = y/(sqrt3/2)`). `(2,0)` is the horizontal bond along the
cylinder axis; `(1,1)` and `(-1,1)` are the two legs of the zigzag.
"""
const _TRI_NN_XC  = ((2, 0), (1, 1), (-1, 1))

"""
The three NEXT-nearest steps of the XC wrapping, `(dX, dY)`, at Cartesian length `sqrt3`:
`(0,2)` is vertical, `(3,1)` and `(3,-1)` the two slanted ones.
"""
const _TRI_NNN_XC = ((0, 2), (3, 1), (3, -1))

"""
    _tri_site_xc(X, Y, Lx, Ly, periodic_x) -> Int | Nothing

Invert the XC embedding: the site index at doubled-Cartesian `(X, Y)`, or `nothing` if the point
falls off the open end. `Y` is reduced modulo the circumference (that IS the wrap); `X` is reduced
modulo `2*Lx` only on a torus.

Throws if `X != Y (mod 2)`, which cannot happen for a legal step from a legal site and therefore
means the caller used an odd circumference or a malformed offset.
"""
function _tri_site_xc(X::Int, Y::Int, Lx::Int, Ly::Int, periodic_x::Bool)
    Yr = mod(Y, Ly)
    Xr = periodic_x ? mod(X, 2 * Lx) : X
    isodd(Xr - Yr) && throw(ArgumentError(
        "XC embedding landed between sites at (X=$Xr, Y=$Yr): X and Y must share parity. " *
        "This is what an ODD circumference produces -- the zigzag ring has a two-site period " *
        "and cannot close on Ly = $Ly."))
    m = div(Xr - mod(Yr, 2), 2) + 1
    (periodic_x || 1 <= m <= Lx) || return nothing
    periodic_x && (m = mod1(m, Lx))
    return (m - 1) * Ly + (Yr + 1)
end

"""
    _tri_walk(Lx, Ly, offsets, geometry, periodic_x, periodic_y, f)

Visit every `(site, offset)` pair once and hand `f(a, b)` the resulting bond. One representative
per `+-` pair is assumed to be in `offsets`, so each bond is produced exactly once per offset that
reaches it -- multiplicity at small circumference is intentional and is the caller's to keep.
"""
function _tri_walk(Lx::Int, Ly::Int, offsets, geometry::Symbol,
                   periodic_x::Bool, periodic_y::Bool, f)
    site(u, v) = (u - 1) * Ly + v
    if geometry === :yc
        for (dx, dy) in offsets, x in 1:Lx, y in 1:Ly
            x2, y2 = x + dx, y + dy
            if periodic_x
                x2 = mod1(x2, Lx)
            elseif !(1 <= x2 <= Lx)
                continue
            end
            if periodic_y
                y2 = mod1(y2, Ly)
            elseif !(1 <= y2 <= Ly)
                continue
            end
            f(site(x, y), site(x2, y2))
        end
    else
        periodic_y || throw(ArgumentError(
            "geometry = :xc describes a WRAPPED zigzag ring; periodic_y = false is not that " *
            "geometry. Use geometry = :yc for an open second direction."))
        for (dX, dY) in offsets, m in 1:Lx, c in 1:Ly
            X, Y = 2 * (m - 1) + mod(c - 1, 2), c - 1
            b = _tri_site_xc(X + dX, Y + dY, Lx, Ly, periodic_x)
            b === nothing && continue
            f(site(m, c), b)
        end
    end
    return nothing
end

"Reject an offset that reaches the same site by `+d` and `-d`, which would double-count the bond."
function _tri_check_alias(Lx::Int, Ly::Int, offsets, geometry::Symbol,
                          periodic_x::Bool, periodic_y::Bool)
    # The alias test is joint across both directions, not one rule per direction. Walking +delta from
    # every site double-counts a bond only when +delta and -delta reach the same site. The open direction
    # alone keeps the two ends apart (e.g. (1,2) at Ly = 4), so a per-direction bound would wrongly refuse
    # valid bonds.
    xper = geometry === :yc ? Lx : 2 * Lx
    for (da, db) in offsets
        ax = periodic_x ? mod(2 * da, xper) == 0 : da == 0
        ay = periodic_y ? mod(2 * db, Ly) == 0 : db == 0
        (ax && ay) && throw(ArgumentError(
            "offset $((da, db)) aliases onto its own negative at Lx = $Lx, Ly = $Ly " *
            "(geometry = $geometry, periodic_x = $periodic_x, periodic_y = $periodic_y): every " *
            "such bond would be counted TWICE. Enlarge that direction or open it."))
    end
    return nothing
end

"Offsets for one neighbour shell under one wrapping."
_tri_offsets(geometry::Symbol, nnn::Bool) =
    geometry === :yc ? (nnn ? _TRI_NNN : _TRI_NN) :
    geometry === :xc ? (nnn ? _TRI_NNN_XC : _TRI_NN_XC) :
    throw(ArgumentError("unknown geometry $geometry (:xc | :yc)"))

"Reject an odd circumference under XC before any bond is built."
function _tri_check_geometry(Ly::Int, geometry::Symbol)
    geometry in (:xc, :yc) || throw(ArgumentError("unknown geometry $geometry (:xc | :yc)"))
    (geometry === :xc && isodd(Ly)) && throw(ArgumentError(
        "geometry = :xc needs an EVEN circumference; got Ly = $Ly. The XC identification is " *
        "T = C*a1 - (C/2)*a2, which is a lattice vector only for even C -- the ring is a zigzag " *
        "with a two-site period. Use an even Ly, or geometry = :yc."))
    return nothing
end

"""
    triangular_cylinder_couplings(Lx, Ly; J=1.0, J2=0.0, geometry=:yc,
                                  periodic_y=true, periodic_x=false) -> Matrix{Float64}

`J[i,j]` for the J1-J2 triangular lattice, column-major snake `site(u, v) = (u - 1) * Ly + v`, with
`Lx` the length and `Ly` the circumference (Eq. (1) of arXiv:2209.00739, `J1 = J`).

`geometry = :yc` (default) closes left-right; `:xc` (the paper's, `Ly` even) closes top-bottom.

Returns the strictly upper triangle that [`pair_mpo`](@ref) reads. `periodic_x = true` closes a torus;
a periodic direction is refused when +d and -d land on the same site.
"""
function triangular_cylinder_couplings(Lx::Int, Ly::Int; J::Float64 = 1.0, J2::Float64 = 0.0,
                                       geometry::Symbol = :yc,
                                       periodic_y::Bool = true, periodic_x::Bool = false)
    Lx >= 1 || throw(ArgumentError("Lx must be >= 1, got $Lx"))
    Ly >= 2 || throw(ArgumentError("Ly must be >= 2, got $Ly"))
    _tri_check_geometry(Ly, geometry)
    fams = J2 == 0.0 ? ((_tri_offsets(geometry, false), J),) :
                       ((_tri_offsets(geometry, false), J), (_tri_offsets(geometry, true), J2))
    for (offs, _) in fams
        _tri_check_alias(Lx, Ly, offs, geometry, periodic_x, periodic_y)
    end
    Jm = zeros(Float64, Lx * Ly, Lx * Ly)
    # `+=`, so a pair reachable by TWO distinct neighbour vectors accumulates both -- which is the
    # physically right answer at small circumference (see the header) and makes an accidental
    # duplicate visible as a doubled coupling instead of a silent overwrite.
    for (offs, w) in fams
        _tri_walk(Lx, Ly, offs, geometry, periodic_x, periodic_y, function (a, b)
            a == b && return nothing        # a step that wrapped onto its own site
            i, j = minmax(a, b)
            Jm[i, j] += w
            return nothing
        end)
    end
    return Jm
end

"""
    triangular_cylinder_mpo(Lx, Ly; J=1.0, J2=0.0, geometry=:yc, periodic_y=true,
                            periodic_x=false) -> MPO

Heisenberg `H = J sum_<ij> S_i . S_j` on the triangular cylinder in the active symmetry mode, built
through [`pair_mpo`](@ref) and [`spin_vertices`](@ref). `geometry` is as in
[`triangular_cylinder_couplings`](@ref); pass `:xc` to match arXiv:2209.00739.

The lattice is frustrated: a Neel start is a poor guess and the bond dimension needed is markedly
higher than on the square cylinder at the same site count. That is physics, not an integrator bug.
"""
triangular_cylinder_mpo(Lx::Int, Ly::Int; J::Float64 = 1.0, J2::Float64 = 0.0,
                        geometry::Symbol = :yc,
                        periodic_y::Bool = true, periodic_x::Bool = false) =
    pair_mpo(Lx * Ly,
             triangular_cylinder_couplings(Lx, Ly; J = J, J2 = J2, geometry = geometry,
                                           periodic_y = periodic_y, periodic_x = periodic_x),
             spin_vertices(Lx * Ly))

"""
    triangular_cylinder_bonds(Lx, Ly; geometry=:yc, periodic_y=true, periodic_x=false,
                              nnn=false) -> Vector{Tuple{Int,Int}}

The bond list, for tests and for anything that needs to iterate bonds rather than read `J[i,j]`.
`nnn = false` gives the `J1` bonds, `nnn = true` the `J2` ones.

⚠️ A pair reachable by two distinct neighbour vectors appears TWICE here, matching the `+=` in
the coupling matrix. `unique` it if a distinct-pair list is what is wanted, but do not do that
before counting coordination -- the multiplicity is the physics at small circumference.
"""
function triangular_cylinder_bonds(Lx::Int, Ly::Int; geometry::Symbol = :yc,
                                   periodic_y::Bool = true,
                                   periodic_x::Bool = false, nnn::Bool = false)
    _tri_check_geometry(Ly, geometry)
    out = Tuple{Int, Int}[]
    _tri_walk(Lx, Ly, _tri_offsets(geometry, nnn), geometry, periodic_x, periodic_y,
              (a, b) -> (a == b || push!(out, minmax(a, b)); nothing))
    return out
end

"""
    triangular_coordination(Lx, Ly; geometry=:yc, periodic_y=true, periodic_x=false,
                            nnn=false) -> Vector{Int}

Neighbour count per site. Every bulk site of a wrapped cylinder must have six (on a torus, every
site), which separates a triangular lattice from a square one with a stray bond. `nnn = true` counts
the J2 shell, which must also be six in the bulk.

Under `:xc` the open ends are ragged, since the odd rings of the end column stick out half a spacing.
Select bulk sites by coordination, not by column index.
"""
function triangular_coordination(Lx::Int, Ly::Int; geometry::Symbol = :yc,
                                 periodic_y::Bool = true,
                                 periodic_x::Bool = false, nnn::Bool = false)
    z = zeros(Int, Lx * Ly)
    for (i, j) in triangular_cylinder_bonds(Lx, Ly; geometry = geometry, periodic_y = periodic_y,
                                            periodic_x = periodic_x, nnn = nnn)
        z[i] += 1; z[j] += 1
    end
    return z
end

"""
    triangular_site_position(Lx, Ly, i; geometry=:yc) -> NTuple{2, Float64}

CARTESIAN position of site `i`, in units of the nearest-neighbour distance. This, not the integer
index pair, is what a Fourier transform must use -- transforming against integer coordinates
computes the structure factor of a SHEARED square lattice, which is wrong without looking wrong.

  - `:yc` -- `r = x*a1 + y*a2` with `a1 = (1,0)`, `a2 = (-1/2, sqrt3/2)`.
  - `:xc` -- the zigzag embedding of the header: `r = (X/2, Y*sqrt3/2)` with `Y = c-1` and
    `X = 2*(m-1) + ((c-1) mod 2)`.
"""
function triangular_site_position(Lx::Int, Ly::Int, i::Int; geometry::Symbol = :yc)
    _tri_check_geometry(Ly, geometry)
    (1 <= i <= Lx * Ly) || throw(BoundsError((Lx, Ly), i))
    u, v = div(i - 1, Ly) + 1, mod(i - 1, Ly) + 1
    if geometry === :yc
        return (u - 0.5 * v, v * sqrt(3.0) / 2)
    end
    return (((2 * (u - 1) + mod(v - 1, 2))) / 2, (v - 1) * sqrt(3.0) / 2)
end

"""
    triangular_wrap_vector(Ly; geometry=:yc) -> NTuple{2, Float64}

The CARTESIAN identification vector `T` of the cylinder: the geometry is the plane modulo `T`, and
`q` is an exact quantum number iff `q.T` is a multiple of `2*pi`. `:yc` gives `Ly*a2`, of length
`Ly`; `:xc` gives the vertical `(0, sqrt3*Ly/2)`. Everything the two wrappings disagree about
follows from this one vector.
"""
function triangular_wrap_vector(Ly::Int; geometry::Symbol = :yc)
    _tri_check_geometry(Ly, geometry)
    return geometry === :yc ? (-0.5 * Ly, Ly * sqrt(3.0) / 2) : (0.0, Ly * sqrt(3.0) / 2)
end
