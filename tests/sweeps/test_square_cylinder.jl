# THE CYLINDER GEOMETRY, PINNED BEFORE ANY PHYSICS RUNS ON IT.
#
# A 2D model reaches an MPS only through an ordering of its sites, so the coupling matrix IS the
# model. Every failure mode here is silent -- a dropped wrap bond, a doubled one, or a leg bond
# joining the wrong pair all produce a perfectly well-formed Hamiltonian that is simply not the
# one intended, and the run converges happily to the wrong answer.
#
# The counts are derived independently of the constructor rather than read off it: for an
# `Lx x Ly` cylinder there are `Lx` rungs of `Ly - 1` bonds, `Lx` wrap bonds, and `Lx - 1` legs of
# `Ly` bonds. Checking the code against its own output would pass no matter what it built.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

@testset "square lattice on a cylinder" begin

    @testset "bond counts and geometry" begin
        for (Lx, Ly) in ((4, 3), (5, 4), (3, 5))
            J = square_cylinder_couplings(Lx, Ly)
            bonds = square_cylinder_bonds(Lx, Ly)
            # rungs + wraps + legs, counted from the lattice and not from the constructor
            want = Lx * (Ly - 1) + Lx + (Lx - 1) * Ly
            @test length(bonds) == want
            @test count(!iszero, J) == want
            # NO BOND COUNTED TWICE. `put!` accumulates, so a duplicated pair shows up as 2.0
            # rather than being silently overwritten -- that is what makes this check meaningful.
            @test all(x -> x == 0.0 || x ≈ 1.0, J)
            # Strictly upper triangular: `pair_mpo` reads i<j, and a stray lower entry would be
            # a second copy of the same bond.
            # ⚠ COMPUTED OUTSIDE `@test`. A generator with a DEPENDENT range (`j in 1:i`) does
            # not survive the `@test` macro's rewriting -- it reports `UndefVarError: i`, which
            # looks like a broken geometry and is actually a broken assertion.
            lower_is_empty = all(J[i, j] == 0.0 for i in 1:(Lx * Ly) for j in 1:i)
            @test lower_is_empty

            # DEGREE: 2 ring neighbours (y±1, periodic) + 1 leg at an end column, 2 in the
            # interior. So 3 or 4 -- NOT 4 or 5, which is what a first draft of this asserted
            # and is the kind of arithmetic slip a test exists to catch in the first place.
            deg = [count(!iszero, J[i, :]) + count(!iszero, J[:, i]) for i in 1:(Lx * Ly)]
            @test all(d -> d in (3, 4), deg) ||
                (@info "unexpected cylinder degrees" Lx Ly sort(unique(deg)); false)
            @test count(==(3), deg) == 2 * Ly      # exactly the two end columns
            @test count(==(4), deg) == (Lx - 2) * Ly
        end
    end

    @testset "the wrap is the LONGEST-range term, and the snake wraps the SHORT direction" begin
        Lx, Ly = 5, 4
        J = square_cylinder_couplings(Lx, Ly)
        rng = [abs(j - i) for i in 1:(Lx * Ly), j in 1:(Lx * Ly) if J[i, j] != 0.0]
        # Under the column-major snake: rung = 1, wrap = Ly-1, leg = Ly. So nothing may reach
        # further than Ly -- that is the entire reason for wrapping the short direction, and a
        # row-major ordering would put leg bonds at distance Lx instead.
        @test maximum(rng) == Ly
        @test Ly - 1 in rng          # the wrap bond exists at all
        @test count(==(Ly), rng) == (Lx - 1) * Ly    # every leg bond
    end

    @testset "Ly = 2 is REFUSED, because the wrap would double the rung" begin
        # At Ly = 2 the wrap bond (x,2)-(x,1) IS the rung bond, and adding both silently doubles
        # that coupling -- a factor of two in H that reads as a physics result.
        @test_throws ArgumentError square_cylinder_couplings(4, 2)
        # ...and the two-leg ladder is reachable by saying so explicitly.
        J = square_cylinder_couplings(4, 2; periodic_y = false)
        @test count(!iszero, J) == 4 * 1 + 3 * 2
        @test all(x -> x == 0.0 || x ≈ 1.0, J)
    end

    @testset "the MPO is the same operator in every symmetry mode" begin
        # 3x3 with the wrap: 9 sites, small enough that the energy of a product state is a
        # closed form. A Neel-like state on a FRUSTRATED odd cylinder is not special, so the
        # check is cross-symmetry agreement rather than a known number.
        Lx, Ly = 3, 3
        es = Float64[]
        for sym in (:none, :U1)
            set_symmetry!(sym)
            mpo = square_cylinder_mpo(Lx, Ly)
            @test length(mpo) == Lx * Ly
            psi = neel_state(Lx * Ly)
            push!(es, real(mpo_energy(psi, mpo)) / max(norm(psi)^2, eps()))
        end
        @test es[1] ≈ es[2] atol = 1e-10
        # ⚠ THE MPO WIDTH MUST TRACK Ly, NOT THE SITE COUNT. `pair_mpo` carries one transport
        # channel per OPEN bond, and the snake keeps at most ~Ly open at any cut. If this grows
        # with Lx instead, the ordering has been transposed and the whole point is lost.
        set_symmetry!(:U1)
        w33 = maximum(mpo_virtual_dims(square_cylinder_mpo(3, 3)))
        w63 = maximum(mpo_virtual_dims(square_cylinder_mpo(6, 3)))
        @test w63 <= w33 + 2 ||
            (@info "cylinder MPO width grew with Lx -- ordering transposed?" w33 w63; false)
        set_symmetry!(:U1)
    end
end
