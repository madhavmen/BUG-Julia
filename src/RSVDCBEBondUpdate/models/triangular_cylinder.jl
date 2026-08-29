# ── the triangular lattice on a cylinder, as a coupling matrix ───────────────
#
# The square cylinder plus ONE diagonal per plaquette. Same column-major snake, same reasons: this
# file builds the coupling matrix and nothing else, and `pair_mpo` turns it into the operator.
#
# ⛔ THE DIAGONAL IS WHAT MAKES IT TRIANGULAR, AND IT IS THE LONGEST-RANGE BOND. Under
# `site(x, y) = (x - 1) * Ly + y` the three bond families sit at chain distances
#
#     rung  (x,y)-(x,y+1)        1
#     wrap  (x,Ly)-(x,1)         Ly - 1     (periodic_y only)
#     leg   (x,y)-(x+1,y)        Ly
#     diag  (x,y)-(x+1,y+1)      Ly + 1     <- the new one, and the longest
#
# so the MPO bond dimension grows with `Ly` exactly as on the square cylinder, but with one more
# transport channel open across a cut. Check it with `mpo_virtual_dims` rather than assuming.
#
# ⛔ EVERY BULK SITE MUST END WITH SIX NEIGHBOURS. That is the definition of the triangular
# lattice, and it is the single cheapest thing to assert: `(x,y±1)`, `(x±1,y)`, `(x+1,y+1)` and
# `(x-1,y-1)`. A diagonal put on the WRONG corner -- `(x+1,y-1)` instead of `(x+1,y+1)` -- still
# gives six neighbours and still runs; it just builds the mirror lattice. On an isotropic
# Heisenberg model the two are related by reflection and give the SAME energy, so the mistake is
# invisible in the one number most likely to be checked. `triangular_cylinder_bonds` exists so the
# geometry can be inspected directly instead.
#
# ⚠ THE DIAGONAL WRAPS TOO. On a cylinder the plaquette at `y = Ly` closes onto `y = 1`, so the
# diagonal from `(x, Ly)` goes to `(x+1, 1)`. Dropping that one bond leaves a SEAM: a line of
# under-coordinated sites running the length of the cylinder, which lowers the energy smoothly and
# reads as a physics result rather than as a missing bond.
#
# ── THE NEXT-NEAREST NEIGHBOURS, i.e. `J2` ───────────────────────────────────────────────────
#
# `x` and `y` here step along lattice vectors at 120 degrees, which is what makes `(1,1)` a
# NEAREST neighbour alongside `(1,0)` and `(0,1)`. With `e_x . e_y = -1/2`,
#
#     |n e_x + m e_y|^2 = n^2 + m^2 - n m
#
# so the six vectors of length 1 are +-(1,0), +-(0,1), +-(1,1) -- the NN set above -- and the six
# of length sqrt(3) are +-(1,-1), +-(2,1), +-(1,2). Those are `J2`. Deriving them rather than
# guessing matters: `(1,-1)` has |.|^2 = 3 and is a genuine NNN, while the visually similar
# `(1,-2)` has |.|^2 = 7 and is NOT a neighbour at all.
#
# ⛔ `J2` OPENS A CHANNEL AT CHAIN DISTANCE `2*Ly + 1` (the `(2,1)` vector), which is LONGER than
# any `J1` bond and is what sets the MPO bond dimension once `J2 != 0`. Read it off
# `mpo_virtual_dims`; do not assume it is the `J1` value plus a constant.
#
# ⚠ AT SMALL CIRCUMFERENCE TWO DISTINCT NNN VECTORS CAN LAND ON THE SAME PAIR, and then that pair
# genuinely carries `2*J2` -- both vectors exist in the infinite lattice and wrapping does not
# remove one. At `Ly = 3`, `(1,-1)` and `(1,2)` are the same bond mod 3, so `Ly = 3` doubles that
# family. `put!` uses `+=` precisely so this ACCUMULATES and is visible, instead of one vector
# silently overwriting the other. It is a property of the geometry, not a bug -- but it does mean
# a `C = 3` cylinder is not a small version of a `C = 6` one.

"The six NEAREST-neighbour steps, as `(dx, dy)` with `|n e_x + m e_y|^2 = n^2 + m^2 - n m = 1`.
Only the three with a positive representative are listed: the loop visits every site, so adding
the negatives too would put each bond in twice."
const _TRI_NN  = ((0, 1), (1, 0), (1, 1))

"The six NEXT-nearest steps, `|.|^2 = 3`, same one-representative-per-pair convention."
const _TRI_NNN = ((1, -1), (2, 1), (1, 2))

"""
    triangular_cylinder_couplings(Lx, Ly; J=1.0, J2=0.0, periodic_y=true, periodic_x=false)
        -> Matrix{Float64}

`J[i,j]` for the `J1-J2` triangular lattice on `Lx x Ly` sites under the column-major snake
`site(x, y) = (x - 1) * Ly + y`.

`J` weights the six nearest neighbours `+-(1,0), +-(0,1), +-(1,1)`; `J2` the six next-nearest
`+-(1,-1), +-(2,1), +-(1,2)`. This is Eq. (1) of arXiv:2209.00739 with `J1 = J`.

Periodic in `y` (the circumference, the paper's `XC` wrapping: top and bottom identified) and by
default open in `x`. `periodic_x = true` closes the cylinder into a **torus**, which is the
geometry the exact-diagonalisation anchors are quoted on -- see `benchmarks/ground_state/`.

Returns the STRICTLY UPPER triangle, which is what [`pair_mpo`](@ref) reads.

⚠️ A periodic direction of length `n` is refused when `n < 2*max|offset| + 1`, because below that
a step of `+d` and one of `-d` land on the same site and the ring double-counts the bond rather
than wrapping it. That is `Ly >= 3` for `J1`, and `Ly >= 5` for `J2` -- and it is why `C = 4`
takes `J2` while `Lx = 4` with `periodic_x` does not.
"""
function triangular_cylinder_couplings(Lx::Int, Ly::Int; J::Float64 = 1.0, J2::Float64 = 0.0,
                                       periodic_y::Bool = true, periodic_x::Bool = false)
    Lx >= 1 || throw(ArgumentError("Lx must be >= 1, got $Lx"))
    Ly >= 2 || throw(ArgumentError("Ly must be >= 2, got $Ly"))
    fams = J2 == 0.0 ? ((_TRI_NN, J),) : ((_TRI_NN, J), (_TRI_NNN, J2))
    # ⛔ THE ALIAS TEST IS JOINT ACROSS THE TWO DIRECTIONS, NOT ONE RULE PER DIRECTION. Walking
    # `+delta` from every site double-counts a bond only when `+delta` and `-delta` reach the SAME
    # site, and on a cylinder the OPEN direction alone is enough to keep them apart: `(1,2)` at
    # `Ly = 4` is a perfectly good bond because the `x` index still distinguishes the two ends,
    # even though `+2 == -2` around a ring of 4. A naive per-direction bound (`Ly >= 2|dy|+1`)
    # would refuse it -- and that is exactly the `C = 4` column of Fig. 3 in arXiv:2209.00739.
    for (offs, _) in fams, (dx, dy) in offs
        ax = periodic_x ? mod(2 * dx, Lx) == 0 : dx == 0
        ay = periodic_y ? mod(2 * dy, Ly) == 0 : dy == 0
        (ax && ay) && throw(ArgumentError(
            "offset $((dx, dy)) aliases onto its own negative at Lx = $Lx, Ly = $Ly " *
            "(periodic_x = $periodic_x, periodic_y = $periodic_y): every such bond would be " *
            "counted TWICE. Enlarge that direction or open it."))
    end
    L = Lx * Ly
    site(x, y) = (x - 1) * Ly + y
    Jm = zeros(Float64, L, L)
    # `+=`, so a pair reachable by TWO distinct neighbour vectors accumulates both -- which is the
    # physically right answer at small circumference (see the header) and makes an accidental
    # duplicate visible as a doubled coupling instead of a silent overwrite.
    function put!(a::Int, b::Int, w::Float64)
        a == b && return nothing            # a step that wrapped onto its own site
        i, j = minmax(a, b)
        Jm[i, j] += w
        return nothing
    end
    for (offs, w) in fams, (dx, dy) in offs, x in 1:Lx, y in 1:Ly
        x2, y2 = x + dx, y + dy
        # Off the open end in `x` (or off an unwrapped `y` edge) means the bond does not exist.
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
        put!(site(x, y), site(x2, y2), w)
    end
    return Jm
end

"""
    triangular_cylinder_mpo(Lx, Ly; J=1.0, periodic_y=true) -> MPO

Heisenberg `H = J Σ_<ij> S_i · S_j` on the triangular cylinder, in whatever symmetry mode is
active. Built through [`pair_mpo`](@ref) and [`spin_vertices`](@ref), so `:none`, `:U1` and `:SU2`
all route through the same code.

⚠️ THIS LATTICE IS FRUSTRATED. Nearest-neighbour antiferromagnetic Heisenberg on a triangle has no
classical Néel ground state, so a `neel_state` start is a poor guess and the bond dimension needed
for a given accuracy is markedly higher than on the square cylinder at the same site count. That is
physics, not an integrator problem -- do not read it as one.
"""
triangular_cylinder_mpo(Lx::Int, Ly::Int; J::Float64 = 1.0, J2::Float64 = 0.0,
                        periodic_y::Bool = true, periodic_x::Bool = false) =
    pair_mpo(Lx * Ly,
             triangular_cylinder_couplings(Lx, Ly; J = J, J2 = J2,
                                           periodic_y = periodic_y, periodic_x = periodic_x),
             spin_vertices(Lx * Ly))

"""
    triangular_cylinder_bonds(Lx, Ly; periodic_y=true, periodic_x=false, nnn=false)
        -> Vector{Tuple{Int,Int}}

The bond list, for tests and for anything that needs to iterate bonds rather than read `J[i,j]`.
`nnn = false` gives the `J1` bonds, `nnn = true` the `J2` ones.

⚠️ A pair reachable by two distinct neighbour vectors appears TWICE here, matching the `+=` in
the coupling matrix. `unique` it if a distinct-pair list is what is wanted, but do not do that
before counting coordination -- the multiplicity is the physics at small circumference.
"""
function triangular_cylinder_bonds(Lx::Int, Ly::Int; periodic_y::Bool = true,
                                   periodic_x::Bool = false, nnn::Bool = false)
    site(x, y) = (x - 1) * Ly + y
    out = Tuple{Int, Int}[]
    for (dx, dy) in (nnn ? _TRI_NNN : _TRI_NN), x in 1:Lx, y in 1:Ly
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
        a, b = site(x, y), site(x2, y2)
        a == b || push!(out, minmax(a, b))
    end
    return out
end

"""
    triangular_coordination(Lx, Ly; periodic_y=true, periodic_x=false, nnn=false) -> Vector{Int}

Neighbour count per site. **Every site of a wrapped cylinder away from the two `x` ends must have
six** -- this is the assertion that separates a triangular lattice from a square one with a stray
bond, and it is cheap enough to run in any test that builds the geometry. On a torus
(`periodic_x = true`) EVERY site must have six, with no ends to exclude. `nnn = true` counts the
`J2` shell instead, which must also be six in the bulk.
"""
function triangular_coordination(Lx::Int, Ly::Int; periodic_y::Bool = true,
                                 periodic_x::Bool = false, nnn::Bool = false)
    z = zeros(Int, Lx * Ly)
    for (i, j) in triangular_cylinder_bonds(Lx, Ly; periodic_y = periodic_y,
                                            periodic_x = periodic_x, nnn = nnn)
        z[i] += 1; z[j] += 1
    end
    return z
end
