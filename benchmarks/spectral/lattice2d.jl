# THE 2D LATTICE GEOMETRY SHARED BY FIG. 4 (SQUARE) AND FIG. 5 (TRIANGULAR) OF arXiv:2209.00739.
#
# One file so the two figures cannot drift apart: the site ordering, the real-space positions, the
# Brillouin-zone path and the allowed-momentum test are each defined ONCE and read by both the
# spectral driver and the plotters.
#
# ⛔ THE TRIANGLE USES THE `XC` WRAPPING, WHICH IS NOT WHAT THIS FILE ORIGINALLY BUILT. The paper,
# Sec. IIB: "The boundary conditions in Fig. 1 is the XC boundary condition, and the YC boundary
# condition would be if we identified the left and right edges rather than the top and bottom ...
# Since most of the high symmetry q values are permitted by the XC geometry, we use the XC
# geometry throughout this work." XC and YC are DIFFERENT HAMILTONIANS, not different labels:
#
#     YC   T = C*a2              a straight row of C bonds,   |T| = C
#     XC   T = C*a1 - (C/2)*a2   a vertical ZIGZAG of C bonds, |T| = sqrt(3)*C/2,  C EVEN
#
# ⛔⛔ EVERYTHING THE TWO DISAGREE ABOUT FOLLOWS FROM `T`, VIA `q.T in 2*pi*Z`. Under XC the
# condition constrains only `q_y` (the paper's Eq. 8) and `q_x` is set by the open direction alone
# (Eq. 9); under YC it mixes the two into lines tilted at 120 degrees.
#
#     XC:   q_y = 4*pi*n/(sqrt3*C)     q_x = 2*pi*n/L
#     YC:   q.(C*a2) in 2*pi*Z
#
# so `K = (4*pi/3, 0)` -- which has `q_y = 0` -- is allowed under XC at EVERY even `C`, and the
# whole `K -> Gamma` segment lies on the `n = 0` line, which is what makes Fig. 5 c) an exact cut.
# Under YC, `K` needs `3 | C` and `K -> Gamma` meets the allowed set at isolated points only.
#
# ⚠ THE REVERSE TRAP AT THE PAPER'S OWN `C = 6`: XC admits `M = (pi, pi/sqrt3)` only when `4 | C`,
# so at `C = 6` only the image `M = (0, 2*pi/sqrt3)` is exact. The paper handles this by RESTORING
# THE C6 SYMMETRY (Eq. 10, `O(Rq) = O(q)`) rather than by measuring every image -- Fig. 2 caption:
# "The orange lines in both figures are the q values that are rotations of the allowed q values by
# the C6 symmetry of the triangular lattice." `q_status` below reports which of the three a given
# path point is, so a figure never presents an interpolation as a measurement.
#
# ⛔ THE INTEGER `(x, y)` ARE NOT CARTESIAN, in either wrapping. Transforming `e^{-i q.(x,y)}`
# against integer indices computes the structure factor of a SHEARED square lattice -- wrong
# without looking wrong. Every position here is Cartesian and comes from `triangular_site_position`
# in `src/RSVDCBEBondUpdate/models/triangular_cylinder.jl`, which is the SAME function the coupling
# matrix is built from. Do not re-derive the embedding here: that duplication is precisely how the
# bond list and the Fourier transform came to describe different lattices.
#
#     reachable N at C=6 (XC, the paper's circumference):  L=3 -> 18, L=4 -> 24, L=6 -> 36 ...
#     ⚠ the paper runs L = C^2 = 36 (N = 216); at L = 3 Eq. (9) gives only THREE distinct q_x
#       values, {0, +-2pi/3}, so `Gamma -> K` carries 3 points. That is a resolution limit of the
#       cylinder, not of the bond dimension, and belongs in the caption.

using LinearAlgebra

const SQRT3 = sqrt(3.0)

"""
    Lattice2D

A cylinder geometry: primitive vectors, the wrapping, the Gaussian broadening the paper uses for
it, and the Brillouin-zone path its figure is plotted along.

`wrap` is `:square`, `:xc` or `:yc` and is the ONLY thing that distinguishes the two triangular
conventions. `Lx` is always the LENGTH `L` in columns and `Ly` the CIRCUMFERENCE `C`, matching the
column-major snake `site(u,v) = (u-1)*Ly + v` used by the coupling-matrix builders.
"""
struct Lattice2D
    name::String
    a1::NTuple{2, Float64}
    a2::NTuple{2, Float64}
    wrap::Symbol
    eta2::Float64                                    # Eq. (12) Gaussian damping
    corners::Vector{NTuple{2, Float64}}
    labels::Vector{String}
end

"""
Square lattice, Fig. 4. `eta^2 = 0.03` (Sec. IV A). The paper's path visits
`Gamma -> X = (0,pi) -> M = (pi,pi) -> Gamma`, both points named in the Fig. 4 caption.

There is no XC/YC distinction here: the square cylinder wraps along `a2 = (0,1)` and `M = (pi,pi)`
and `X = (0,pi)` are exact at every even circumference, which is why a 20-site 5x4 does Fig. 4 in
full.
"""
square_lattice() = Lattice2D("square", (1.0, 0.0), (0.0, 1.0), :square, 0.03,
    [(0.0, 0.0), (0.0, pi), (pi, pi), (0.0, 0.0)], ["Γ", "X", "M", "Γ"])

"""
Triangular lattice, Fig. 5, in the paper's **XC** wrapping. `eta^2 = 0.02` for the 120-degree
ordered phase (the paper uses 0.05 for the striped phase, which this campaign does not visit).

`K = (4*pi/3, 0)` is quoted directly from the Fig. 5 caption. `M = (pi, pi/sqrt3)` is the midpoint
of the adjacent zone edge, so `Gamma -> M -> K -> Gamma` is a closed path along the zone boundary
and back through the interior.

⚠ At `C = 6` this `M` is a C6 IMAGE of an exact momentum rather than an exact one itself -- see the
header. `q_status` labels it, and `geometry_report` prints the breakdown before anything runs.

`a1`/`a2` are kept in the repo's own 120-degree convention (`a1 = (1,0)`, `a2 = (-1/2, sqrt3/2)`)
because the reciprocal lattice and the C6 images are built from them; the SITE POSITIONS come from
`triangular_site_position`, not from these vectors, and under `:xc` the two are not the same map.
"""
triangular_lattice(wrap::Symbol = :xc) =
    Lattice2D("triangle", (1.0, 0.0), (-0.5, SQRT3 / 2), wrap, 0.02,
              [(0.0, 0.0), (pi, pi / SQRT3), (4pi / 3, 0.0), (0.0, 0.0)], ["Γ", "M", "K", "Γ"])

lattice(name::AbstractString) =
    name == "square"      ? square_lattice() :
    name == "triangle"    ? triangular_lattice(:xc) :
    name == "triangle_yc" ? triangular_lattice(:yc) :
    error("unknown lattice $name (square | triangle | triangle_yc)")

"""
    geom_key(name, Lx, Ly) -> String

The identity a tuned-knob file is filed under, and the `geom` column of every knob scan.

⛔ THE CIRCUMFERENCE **AND THE WRAPPING** ARE PART OF THE IDENTITY. "triangle" at `C = 3` and at
`C = 6` are different problems -- different MPO bond dimension, different entanglement across a
cut, different optimal sketch width -- and XC vs YC at the same `C` are different Hamiltonians
outright. Keying on the lattice NAME alone is exactly how a knob tuned on one gets silently loaded
for the other.
"""
geom_key(name::AbstractString, Lx::Int, Ly::Int) =
    name == "chain" ? "chain_$(Lx * Ly)" : "$(name)_$(Lx)x$(Ly)"

"Integer lattice coordinates of site `i` under the column-major snake: `(column, ring index)`."
coords(i::Int, Ly::Int) = (div(i - 1, Ly) + 1, mod(i - 1, Ly) + 1)

"""
    site_position(lat, i, Ly) -> NTuple{2, Float64}

CARTESIAN position of site `i`. This, not `(x,y)`, is what the Fourier transform must use.

Delegates to `triangular_site_position` for the triangle so the embedding cannot drift from the
one the coupling matrix was built with -- under `:xc` the position is the ZIGZAG
`r = (X/2, Y*sqrt3/2)` with `Y = v-1`, `X = 2*(u-1) + ((v-1) mod 2)`, which is NOT `u*a1 + v*a2`.
"""
function site_position(lat::Lattice2D, i::Int, Ly::Int)
    # The position depends only on `(i, Ly)`; `Lx` is the callee's bounds check, so pass the
    # smallest column count that contains `i` rather than plumbing the real `Lx` through here.
    lat.name == "triangle" &&
        return triangular_site_position(cld(i, Ly), Ly, i; geometry = lat.wrap)
    u, v = coords(i, Ly)
    return (u * lat.a1[1] + v * lat.a2[1], u * lat.a1[2] + v * lat.a2[2])
end

"""
    wrap_vector(lat, Ly) -> NTuple{2, Float64}

The cylinder's identification vector `T`: the geometry is the plane modulo `T`, and `q` is an exact
quantum number iff `q.T` is a multiple of `2*pi`. Every difference between XC and YC follows from
this one vector.
"""
wrap_vector(lat::Lattice2D, Ly::Int) =
    lat.name == "triangle" ? triangular_wrap_vector(Ly; geometry = lat.wrap) :
                             (Ly * lat.a2[1], Ly * lat.a2[2])

"""
    allowed_q(lat, q, Ly) -> Bool

Is `q` an EXACT quantum number of the cylinder? True iff `q.T` is a multiple of `2*pi`.
See the header: this is the paper's Eq. (7)-(8) for `:xc`, and a tilted-line condition for `:yc`.
"""
function allowed_q(lat::Lattice2D, q::NTuple{2, Float64}, Ly::Int)
    T = wrap_vector(lat, Ly)
    ph = (q[1] * T[1] + q[2] * T[2]) / (2pi)
    return abs(ph - round(ph)) < 1e-9
end

"""
    c6_images(lat, q) -> Vector{NTuple{2,Float64}}

The point-group orbit of `q`: six-fold rotations for the triangle, four-fold for the square. This
is the paper's Eq. (10), `O(Rq) = O(q)` for any `R` leaving the lattice invariant, and it is how
Fig. 2's orange points are generated from the blue ones.
"""
function c6_images(lat::Lattice2D, q::NTuple{2, Float64})
    n = lat.name == "triangle" ? 6 : 4
    out = NTuple{2, Float64}[]
    for k in 0:(n - 1)
        t = 2pi * k / n
        push!(out, (cos(t) * q[1] - sin(t) * q[2], sin(t) * q[1] + cos(t) * q[2]))
    end
    return out
end

"""
    q_status(lat, q, Ly) -> Symbol

`:exact` -- `q` is a quantum number of the cylinder and `S(q)` there is a measurement.
`:image` -- `q` is not, but a C6/C4 rotation of it is, so `S(q)` follows from Eq. (10). This is the
            paper's orange set, and it is a legitimate value, not an interpolation.
`:interp` -- neither; whatever is plotted there is smooth continuation off the allowed set.

⛔ A FIGURE THAT DOES NOT DISTINGUISH THESE PRESENTS INTERPOLATION AS DATA. On a `C = 6` XC
cylinder most of the plane is `:interp`, and the ordering peak is only meaningful because `K` is
`:exact`.
"""
function q_status(lat::Lattice2D, q::NTuple{2, Float64}, Ly::Int)
    allowed_q(lat, q, Ly) && return :exact
    any(p -> allowed_q(lat, p, Ly), c6_images(lat, q)) && return :image
    return :interp
end

"""
    bz_path(lat, npts) -> (qs, labels, dists)

The figure's horizontal axis: `npts` samples per segment plus the endpoints, with the cumulative
CARTESIAN distance so segment lengths are drawn to scale (the triangular `Gamma->M` and `M->K`
segments differ in length by a factor of ~2.7, which an index axis would hide).
"""
function bz_path(lat::Lattice2D, npts::Int)
    qs = NTuple{2, Float64}[]; labels = String[]; dists = Float64[]
    d = 0.0
    for s in 1:(length(lat.corners) - 1)
        a, b = lat.corners[s], lat.corners[s + 1]
        seg = hypot(b[1] - a[1], b[2] - a[2])
        for k in 0:(npts - 1)
            f = k / npts
            push!(qs, (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])))
            push!(labels, k == 0 ? lat.labels[s] : "")
            push!(dists, d + f * seg)
        end
        d += seg
    end
    push!(qs, lat.corners[end]); push!(labels, lat.labels[end]); push!(dists, d)
    return qs, labels, dists
end

"""
    geometry_report(io, lat, Lx, Ly) -> Bool

Print, BEFORE any physics runs, which of the figure's high-symmetry points this cylinder can
actually represent, and how much of the plotted path is a measurement rather than a continuation.
Returns `true` when every named corner is `:exact` or `:image`.

A run at a circumference that cannot carry `K` is still a valid COST measurement and an invalid
Fig. 5 -- the two must not be confused after the fact.
"""
function geometry_report(io::IO, lat::Lattice2D, Lx::Int, Ly::Int)
    T = wrap_vector(lat, Ly)
    println(io, "  lattice $(lat.name)  wrap=:$(lat.wrap)  a1=$(lat.a1)  a2=$(lat.a2)")
    println(io, "    L=$Lx (length, columns)  C=$Ly (circumference)  N=$(Lx*Ly)  " *
                "T=$(round.(T; digits=4))  |T|=$(round(hypot(T...); digits=4))")
    if lat.name == "triangle" && lat.wrap === :xc && isodd(Ly)
        println(io, "  ⛔ XC needs an EVEN circumference; C=$Ly cannot close the zigzag ring.")
        return false
    end
    bad = String[]
    for (nm, q) in zip(lat.labels, lat.corners)
        nm == "Γ" && continue
        st = q_status(lat, q, Ly)
        msg = st === :exact  ? "EXACT (a quantum number of this cylinder)" :
              st === :image  ? "C6 IMAGE of an exact q (Eq. 10) -- a value, not an interpolation" :
                               "⛔ NOT REPRESENTABLE at C=$Ly"
        println(io, "    $nm = $(round.(q; digits=4))  $msg")
        st === :interp && push!(bad, nm)
    end
    qs, _, _ = bz_path(lat, 24)
    ne = count(q -> q_status(lat, q, Ly) === :exact, qs)
    ni = count(q -> q_status(lat, q, Ly) === :image, qs)
    println(io, "    path: $ne/$(length(qs)) exact, $ni/$(length(qs)) C6 images, " *
                "$(length(qs)-ne-ni)/$(length(qs)) interpolated")
    isempty(bad) || println(io, "  ⛔ $(join(bad, ", ")) is not representable at circumference " *
                                "$Ly -- any ordering peak there CANNOT appear. Cost numbers " *
                                "from this run stay valid.")
    return isempty(bad)
end

"""
    centre_site(Lx, Ly) -> Int

"the center site in the lattice" (paper, Eq. 6). On even sides there is no exact centre, so this
takes the lower-middle column and row; every `x` in `G(x,t)` is measured from it, so the choice
is written into the output rather than assumed.
"""
centre_site(Lx::Int, Ly::Int) = (cld(Lx, 2) - 1) * Ly + cld(Ly, 2)
