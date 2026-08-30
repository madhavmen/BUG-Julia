# ── the triangular lattice on a cylinder, as a coupling matrix ───────────────
#
# TWO INEQUIVALENT WRAPPINGS, and arXiv:2209.00739 uses the second one. This file builds the
# coupling matrix and nothing else; `pair_mpo` turns it into the operator.
#
# ⛔ `XC` AND `YC` ARE DIFFERENT HAMILTONIANS, NOT DIFFERENT LABELS FOR ONE. A cylinder is the
# quotient of the plane by ONE lattice translation `T`, and the two conventions choose different
# `T`. Sec. IIB of arXiv:2209.00739: "The boundary conditions in Fig. 1 is the XC boundary
# condition, and the YC boundary condition would be if we identified the left and right edges
# rather than the top and bottom."
#
#     YC   T = C*a2                 a straight row of C nearest-neighbour bonds,  |T| = C
#     XC   T = C*a1 - (C/2)*a2      a ZIGZAG of C bonds, vertical,  |T| = sqrt(3)*C/2
#
# with the paper's `a1 = (1/2, sqrt3/2)`, `a2 = (1, 0)`. They are not related by any lattice
# symmetry -- the lengths differ by `sqrt(3)/2` -- so a run cannot be relabelled from one to the
# other after the fact. **`geometry = :xc` REQUIRES AN EVEN `C`**, because `T` is not a lattice
# vector otherwise; the zigzag has a two-site period and cannot close on an odd ring.
#
# ⛔⛔ THE WRAPPING DECIDES WHICH MOMENTA EXIST, AND THAT IS WHY THE PAPER PICKED XC. Allowed `q`
# satisfy `q.T in 2*pi*Z`. For XC, `T` is vertical, so the condition constrains ONLY `q_y`
# (Eq. 8), and `q_x` is set by the open direction alone (Eq. 9):
#
#     XC:   q_y = 4*pi*n/(sqrt3*C)          q_x = 2*pi*n/L
#     YC:   q.(C*a2) in 2*pi*Z              -- lines TILTED at 120 deg, mixing q_x and q_y
#
# The ordering wavevector `K = (4*pi/3, 0)` has `q_y = 0`, so **XC admits `K` at every even `C`,
# and the whole `K -> Gamma` segment lies on the `n = 0` line** -- which is what makes Fig. 5 c)
# (the dispersion measured from `K` towards `Gamma`) an exact cut rather than an interpolation.
# Under YC, `K` is allowed only when `3 | C`, and `K -> Gamma` crosses the allowed tilted lines at
# isolated points only. Paper, Sec. IIB: "Since most of the high symmetry q values are permitted
# by the XC geometry, we use the XC geometry throughout this work."
#
# ⚠ THE REVERSE TRAP AT THE PAPER'S OWN `C = 6`: XC admits `M = (pi, pi/sqrt3)` only when `4 | C`,
# so at `C = 6` only the image `M = (0, 2*pi/sqrt3)` is exact. The paper covers this by restoring
# the `C6` symmetry (Eq. 10) rather than by measuring every image directly -- see the Fig. 2
# caption, "the orange lines ... are the q values that are rotations of the allowed q values".
#
# ── INDEXING ─────────────────────────────────────────────────────────────────────────────────
#
# Both geometries use the same column-major snake `site(u, v) = (u - 1) * Ly + v`, `v` running
# around the circumference, so `Lx` is always the LENGTH `L` in columns and `Ly` the
# CIRCUMFERENCE `C`, and `N = Lx*Ly` either way.
#
# YC steps in the 120-degree integer basis `(dx, dy)`, where `|n e_x + m e_y|^2 = n^2 + m^2 - n m`
# gives the NN shell `+-(1,0), +-(0,1), +-(1,1)` and the NNN shell `+-(1,-1), +-(2,1), +-(1,2)`.
#
# XC steps in HALF-INTEGER CARTESIAN coordinates scaled to integers,
#
#     X = 2*x_cartesian     Y = y_cartesian / (sqrt3/2)
#
# in which a site `(m, c)` sits at `Y = c - 1`, `X = 2*(m - 1) + ((c - 1) mod 2)` -- the odd rings
# offset by half a lattice spacing, which IS the zigzag. In these coordinates the wrap is simply
# `Y -> Y + C` with `X` unchanged, so **the ring bonds stay at chain distance ~C exactly as in YC**
# and the MPO bond dimension does not blow up. Writing XC instead as a screw shift on the
# 120-degree basis, `(x,y) == (x + C/2, y + C)`, is the SAME lattice but puts the wrap bond at
# chain distance `C^2/2 - C + 1` (13 instead of 5 at `C = 6`) and inflates the MPO for nothing.
#
# ⚠ EVERY LATTICE SITE SATISFIES `X == Y (mod 2)`. Every NN/NNN step preserves that parity, and so
# does the wrap `dY = C` ONLY BECAUSE `C` IS EVEN. `_tri_site_xc` asserts it rather than rounding,
# so an odd `C` fails loudly instead of silently landing between sites.
#
# ⛔ EVERY BULK SITE MUST END WITH SIX NEIGHBOURS. That is the definition of the triangular
# lattice, and it is the single cheapest thing to assert. A diagonal put on the WRONG corner still
# gives six neighbours and still runs; it just builds the mirror lattice, which on an isotropic
# Heisenberg model has the SAME energy -- invisible in the one number most likely to be checked.
# `triangular_cylinder_bonds` exists so the geometry can be inspected directly instead.
#
# ⚠ AT SMALL CIRCUMFERENCE TWO DISTINCT NEIGHBOUR VECTORS CAN LAND ON THE SAME PAIR, and then that
# pair genuinely carries twice the coupling -- both vectors exist in the infinite lattice and
# wrapping does not remove one. `put!` uses `+=` precisely so this ACCUMULATES and is visible,
# instead of one vector silently overwriting the other. It is a property of the geometry, not a
# bug -- but it does mean a small-`C` cylinder is not a small version of a large-`C` one.
#
# ⚠ `J2` OPENS THE LONGEST CHANNEL and is what sets the MPO bond dimension once `J2 != 0`. Read it
# off `mpo_virtual_dims`; do not assume it is the `J1` value plus a constant.

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
    # ⛔ THE ALIAS TEST IS JOINT ACROSS THE TWO DIRECTIONS, NOT ONE RULE PER DIRECTION. Walking
    # `+delta` from every site double-counts a bond only when `+delta` and `-delta` reach the SAME
    # site, and the OPEN direction alone is enough to keep them apart: `(1,2)` at `Ly = 4` is a
    # perfectly good YC bond because the `x` index still distinguishes the two ends, even though
    # `+2 == -2` around a ring of 4. A naive per-direction bound (`Ly >= 2|dy|+1`) would refuse it
    # -- and that is exactly the `C = 4` column of Fig. 3 in arXiv:2209.00739.
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

`J[i,j]` for the `J1-J2` triangular lattice on `Lx x Ly` sites under the column-major snake
`site(u, v) = (u - 1) * Ly + v`, with `Lx` the LENGTH in columns and `Ly` the CIRCUMFERENCE.
This is Eq. (1) of arXiv:2209.00739 with `J1 = J`.

`geometry` picks the wrapping, and the two are DIFFERENT HAMILTONIANS -- see the file header.

  - `:yc` (default, backwards-compatible) identifies the left and right edges: the circumference
    is a straight row of `Ly` nearest-neighbour bonds.
  - `:xc` is **the paper's choice throughout**: it identifies top and bottom, the circumference is
    a zigzag, and `Ly` must be EVEN. Use this for anything reproducing Figs. 3-8, because it is
    what makes `K = (4*pi/3, 0)` an exact momentum.

`periodic_x = true` closes the cylinder into a **torus**, the geometry the exact-diagonalisation
anchors are quoted on -- see `benchmarks/ground_state/`. `periodic_y = false` is meaningful only
for `:yc`.

Returns the STRICTLY UPPER triangle, which is what [`pair_mpo`](@ref) reads.

⚠️ A periodic direction is refused when a step of `+d` and one of `-d` land on the same site, since
the ring would then double-count the bond rather than wrap it.
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

Heisenberg `H = J Σ_<ij> S_i · S_j` on the triangular cylinder, in whatever symmetry mode is
active. Built through [`pair_mpo`](@ref) and [`spin_vertices`](@ref), so `:none`, `:U1` and `:SU2`
all route through the same code. `geometry` is as in
[`triangular_cylinder_couplings`](@ref) -- pass `:xc` to match arXiv:2209.00739.

⚠️ THIS LATTICE IS FRUSTRATED. Nearest-neighbour antiferromagnetic Heisenberg on a triangle has no
classical Néel ground state, so a `neel_state` start is a poor guess and the bond dimension needed
for a given accuracy is markedly higher than on the square cylinder at the same site count. That is
physics, not an integrator problem -- do not read it as one.
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

Neighbour count per site. **Every site of a wrapped cylinder away from the two open ends must have
six** -- this is the assertion that separates a triangular lattice from a square one with a stray
bond, and it is cheap enough to run in any test that builds the geometry. On a torus
(`periodic_x = true`) EVERY site must have six, with no ends to exclude. `nnn = true` counts the
`J2` shell instead, which must also be six in the bulk.

⚠️ Under `:xc` the two open ends are RAGGED, not flat: the zigzag means the odd rings of the end
column stick out half a lattice spacing, so the under-coordinated set is not simply "column 1 and
column Lx". Select bulk sites by coordination, not by column index.
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
