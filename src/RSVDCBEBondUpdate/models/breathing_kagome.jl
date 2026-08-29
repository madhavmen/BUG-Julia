# ── the DIPOLAR XY MODEL ON A BREATHING KAGOME LATTICE (arXiv:2603.21147) ────
#
#     H = (1/2) sum_{i<j} V_ij (sigma^x_i sigma^x_j + sigma^y_i sigma^y_j),   V_ij = d^2 / R_ij^3
#
# with the BREATHING parameter `h` displacing the B and C sublattices outward along the lattice
# vectors, so the up-triangles expand by `(1+h)` and the down-triangles shrink to compensate. The
# paper reports a continuous Dirac-spin-liquid -> chiral-spin-liquid transition near `h ~ 0.22`,
# with the chiral order parameter turning on there.
#
# ⛔ sigma VERSUS S, AND THE FACTOR THAT HIDES IN IT. The paper writes Pauli operators;
# `spin_vertices` builds spin-1/2 operators, `sigma = 2S`. So
#
#     (1/2) V (sigma^x sigma^x + sigma^y sigma^y) = 2 V (S^x S^x + S^y S^y)
#
# and the coupling handed to `pair_mpo` must be `J_ij = 2 * V_ij`, NOT `V_ij`. A factor of two in
# `H` rescales the entire energy axis and every time in a quench, which reads as a physics result
# rather than a units slip -- `test_breathing_kagome.jl` pins it against a two-site closed form.

"""
    breathing_kagome_positions(Lx, Ly, h) -> Vector{NTuple{2,Float64}}

Real-space positions of the `3*Lx*Ly` sites, in units of the isotropic nearest-neighbour distance.

Sublattices `A = (0,0)`, `B = (1,0)`, `C = (1/2, sqrt3/2)` on lattice vectors `a1 = (2,0)`,
`a2 = (1, sqrt3)`; the breathing displaces `B` and `C` radially outward from `A` by a factor
`(1+h)`, which expands every up-triangle uniformly and shrinks the down-triangles.

Site index matches [`kagome_cylinder_couplings`](@ref): `3*((i-1)*Ly + (j-1)) + s`.
"""
function breathing_kagome_positions(Lx::Int, Ly::Int, h::Float64)
    a1 = (2.0, 0.0)
    a2 = (1.0, sqrt(3.0))
    # (1+h) scaling of the two displaced sublattices: |A-B| = |A-C| = |B-C| = 1+h exactly, so the
    # up-triangle stays EQUILATERAL and only its size breathes. Checked in the test.
    sub = ((0.0, 0.0), ((1.0 + h) * 1.0, 0.0),
           ((1.0 + h) * 0.5, (1.0 + h) * sqrt(3.0) / 2))
    pos = NTuple{2, Float64}[]
    for i in 1:Lx, j in 1:Ly, s in 1:3
        push!(pos, (sub[s][1] + (i - 1) * a1[1] + (j - 1) * a2[1],
                    sub[s][2] + (i - 1) * a1[2] + (j - 1) * a2[2]))
    end
    return pos
end

"""
    breathing_kagome_couplings(Lx, Ly; h=0.0, d2=1.0, shells=7, periodic_y=true) -> Matrix

`J[i,j] = 2 * d2 / R_ij^3` for every pair within the first `shells` distance shells — the factor
2 converting the paper's Pauli convention to this code's spin-1/2 operators (see the header).

⛔ `shells` COUNTS DISTINCT DISTANCES, NOT PAIRS. "Up to the seventh-nearest neighbour" means the
seven smallest distinct separations present in the cluster, so the cutoff is set by BINNING the
pairwise distances and keeping the first seven bins. Taking the seven nearest *pairs per site*
instead would give a different, geometry-dependent cutoff on a lattice with three inequivalent
sublattices, and would silently change with `h`.

⚠ `periodic_y` wraps the `a2` direction by MINIMUM IMAGE: the distance used is the shortest over
the wrap, which is what a cylinder means. Without it the two edges of the circumference would be
`Ly` cells apart instead of adjacent, and the dipolar tail would connect the wrong sites.
"""
function breathing_kagome_couplings(Lx::Int, Ly::Int; h::Float64 = 0.0, d2::Float64 = 1.0,
                                    shells::Int = 7, periodic_y::Bool = true,
                                    ref_h::Float64 = 0.0)
    # ⛔ THE CUTOFF RADIUS IS FIXED AT `ref_h`, NOT RE-DERIVED AT EVERY `h`. At h = 0 the lattice
    # is maximally symmetric so its distance shells are highly degenerate; breathing LIFTS that
    # degeneracy, so "the first 7 distinct distances" covers a DIFFERENT SET OF PAIRS at each h.
    # MEASURED on a 2x3 cluster: 153 couplings at h = 0 against 132 at h = 0.22. A model whose
    # interaction set jumps as the tuning parameter is swept can produce a perfectly sharp
    # "transition" that is nothing but the Hamiltonian changing shape underneath the scan --
    # exactly the artefact a chiral-order-parameter sweep would report as physics.
    #
    # So the shells are counted ONCE in the reference (isotropic) geometry and the resulting
    # RADIUS is applied at every `h`. Pass `ref_h = h` to recover the old per-h behaviour.
    pos = breathing_kagome_positions(Lx, Ly, h)
    L = length(pos)
    a2 = (1.0, sqrt(3.0))
    wrapv = (Ly * a2[1], Ly * a2[2])
    function dist(p, q)
        best = Inf
        # minimum image over the wrapped direction only; `a1` stays open (a cylinder, not a torus)
        for n in (periodic_y ? (-1, 0, 1) : (0,))
            dx = p[1] - q[1] + n * wrapv[1]
            dy = p[2] - q[2] + n * wrapv[2]
            best = min(best, sqrt(dx * dx + dy * dy))
        end
        return best
    end
    D = [i == j ? 0.0 : dist(pos[i], pos[j]) for i in 1:L, j in 1:L]
    # the cutoff RADIUS, from the reference geometry's distance shells
    rpos = ref_h == h ? pos : breathing_kagome_positions(Lx, Ly, ref_h)
    RD = [i == j ? 0.0 : dist(rpos[i], rpos[j]) for i in 1:L, j in 1:L]
    ds = sort!(unique(round.([RD[i, j] for i in 1:L for j in (i + 1):L], digits = 6)))
    cutoff = length(ds) <= shells ? ds[end] : ds[shells]
    J = zeros(Float64, L, L)
    # ⚠ MEMBERSHIP is decided in the REFERENCE geometry; the STRENGTH uses the actual one. That
    # keeps the interaction GRAPH fixed as `h` is swept while the couplings themselves breathe,
    # which is the only way a scan over `h` measures the physics rather than the graph.
    for i in 1:L, j in (i + 1):L
        (RD[i, j] > 0 && round(RD[i, j], digits = 6) <= cutoff + 1e-9) || continue
        r = D[i, j]
        r > 0 || continue
        J[i, j] = 2 * d2 / r^3
    end
    return J
end

"""
    breathing_kagome_mpo(Lx, Ly; h=0.0, d2=1.0, shells=7, periodic_y=true) -> MPO

The dipolar XY Hamiltonian as an exact MPO. `delta = 0` drops the `S^z S^z` vertex, leaving the
two CHARGED `S^+ / S^-` vertices — the case `pair_mpo` exists to handle and `long_range_mpo`
refuses.

⚠ REQUIRES `:U1` or `:none`. The XY model has no SU(2) symmetry (it is not `S_i . S_j`), so
`spin_vertices`' SU(2) branch would build the WRONG operator rather than fail.
"""
function breathing_kagome_mpo(Lx::Int, Ly::Int; h::Float64 = 0.0, d2::Float64 = 1.0,
                              shells::Int = 7, periodic_y::Bool = true)
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "the dipolar XY model has no SU(2) symmetry (it is not S_i.S_j); use :U1 or :none"))
    L = 3 * Lx * Ly
    J = breathing_kagome_couplings(Lx, Ly; h = h, d2 = d2, shells = shells,
                                   periodic_y = periodic_y)
    return pair_mpo(L, J, xxz_chain(L; J = 1.0, delta = 0.0))
end
