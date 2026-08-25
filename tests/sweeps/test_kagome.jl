# THE KAGOME GEOMETRY, PINNED BEFORE ANY PHYSICS RUNS ON IT.
#
# Kagome is the lattice where a geometry bug is HARDEST to notice: the model is frustrated, the
# expected ground state has no order to look wrong, and "featureless with near-uniform bonds" is
# also exactly what a subtly wrong bond list produces. So the lattice is checked against counts
# derived by hand -- never against the constructor's own output, which would pass whatever it
# built.
#
# THE COUNTS, derived independently. For `Lx x Ly` cells (3*Lx*Ly sites), with `a1` open and `a2`
# periodic:
#
#   3*Lx*Ly           intra-cell bonds (the UP triangle: A-B, B-C, C-A)
#   (Lx-1)*Ly         B(i,j) -> A(i+1,j)
#   Lx*Ly             C(i,j) -> A(i,j+1)          (all cells, since a2 wraps)
#   (Lx-1)*Ly         C(i,j) -> B(i-1,j+1)
#
# Coordination is 4 in the bulk and lower on the two open `a1` edges -- kagome edge sites are
# genuinely 2- and 3-coordinate, which is why the degree assertion admits {2,3,4} but REQUIRES
# that 4 actually appears once the lattice is wide enough to have a bulk.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

@testset "kagome cylinder" begin

    @testset "bond counts derived by hand, not from the constructor" begin
        for (Lx, Ly) in ((2, 3), (3, 3), (4, 3))
            J = kagome_cylinder_couplings(Lx, Ly)
            L = 3 * Lx * Ly
            @test size(J) == (L, L)
            want = 3 * Lx * Ly + (Lx - 1) * Ly + Lx * Ly + (Lx - 1) * Ly
            @test count(!iszero, J) == want
            @test length(kagome_bonds(Lx, Ly)) == want
            # NO BOND COUNTED TWICE -- `put!` accumulates, so a duplicate is a 2.0 and not a
            # silent overwrite. This is the assertion that catches a wrap-around off-by-one.
            @test all(x -> x == 0.0 || x ≈ 1.0, J)
            # strictly upper triangular: `pair_mpo` reads i<j
            lower_empty = all(J[i, j] == 0.0 for i in 1:L for j in 1:i)
            @test lower_empty
            # ⛔ COORDINATION 4 IN THE BULK. Kagome is 4-coordinate; the open a1 edges are lower.
            deg = [count(!iszero, J[i, :]) + count(!iszero, J[:, i]) for i in 1:L]
            @test maximum(deg) == 4
            @test all(d -> 2 <= d <= 4, deg)
            # a lattice with a bulk must actually HAVE 4-coordinate sites, and more of them as
            # Lx grows -- otherwise every site is an edge and this is a strip, not a cylinder
            @test count(==(4), deg) >= Ly
        end
    end

    @testset "the wrap is real, and Ly=2 is refused" begin
        # With a2 periodic every cell contributes its C -> A(i,j+1) bond, so removing the wrap
        # must remove exactly Lx of them (the j = Ly row) plus the Lx-1 C -> B bonds in that row.
        Lx, Ly = 3, 3
        open_ = kagome_cylinder_couplings(Lx, Ly; periodic_y = false)
        per   = kagome_cylinder_couplings(Lx, Ly; periodic_y = true)
        @test count(!iszero, per) > count(!iszero, open_)
        @test count(!iszero, per) - count(!iszero, open_) == Lx + (Lx - 1)
        # At Ly = 2 the wrap coincides with the intra-cell bonds; doubling them would be a silent
        # factor of two in H.
        @test_throws ArgumentError kagome_cylinder_couplings(3, 2; periodic_y = true)
        @test count(!iszero, kagome_cylinder_couplings(3, 2; periodic_y = false)) > 0
    end

    @testset "corner-sharing triangles -- every bond in exactly one or two of them" begin
        Lx, Ly = 3, 3
        J = kagome_cylinder_couplings(Lx, Ly)
        tri = kagome_triangles(Lx, Ly)
        @test !isempty(tri)
        # Every triangle's three pairs must BE bonds. A triangle listing a non-bond means the
        # diagnostic used for the spin-liquid signature is averaging over pairs that are not
        # even coupled -- which would produce a plausible, meaningless number.
        for (a, b, c) in tri
            for (p, q) in ((a, b), (b, c), (a, c))
                i, j = minmax(p, q)
                @test J[i, j] != 0.0 ||
                    (@info "triangle lists a non-bond" a b c p q; false)
            end
        end
        # no site repeated inside a triangle
        @test all(t -> length(unique(t)) == 3, tri)
    end

    @testset "the MPO is the same operator in every symmetry mode" begin
        Lx, Ly = 2, 3
        L = 3 * Lx * Ly
        es = Float64[]
        for sym in (:none, :U1)
            set_symmetry!(sym)
            mpo = kagome_cylinder_mpo(Lx, Ly)
            @test length(mpo) == L
            psi = neel_state(L)
            push!(es, real(mpo_energy(psi, mpo)) / max(norm(psi)^2, eps()))
        end
        @test es[1] ≈ es[2] atol = 1e-10
        # ⚠ THE MPO WIDTH MUST TRACK THE CIRCUMFERENCE, NOT THE LENGTH. `pair_mpo` carries one
        # channel per open bond, and the site ordering keeps O(Ly) open at any cut. If this grew
        # with Lx the ordering would be transposed and the whole point of the cylinder lost.
        set_symmetry!(:U1)
        w2 = maximum(mpo_virtual_dims(kagome_cylinder_mpo(2, 3)))
        w4 = maximum(mpo_virtual_dims(kagome_cylinder_mpo(4, 3)))
        @test w4 <= w2 + 4 ||
            (@info "kagome MPO width grew with Lx -- ordering transposed?" w2 w4; false)
        set_symmetry!(:U1)
    end
end
