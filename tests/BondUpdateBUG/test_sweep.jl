using Test, LinearAlgebra, Random, LurCGT, Telum
using BUGJulia.BondUpdateBUG

RNG() = MersenneTwister(0x5EED)
KW = (maxdim = 64, trunc_thresh = 1e-14)

@testset "parity sweep" begin

    @testset "even and odd bonds partition the chain" begin
        @test parity_bonds(6, :even) == [1, 3, 5]
        @test parity_bonds(6, :odd)  == [2, 4]
        @test parity_bonds(7, :even) == [1, 3, 5]
        @test parity_bonds(7, :odd)  == [2, 4, 6]
        for L in (4, 5, 6, 7, 8)
            e, o = parity_bonds(L, :even), parity_bonds(L, :odd)
            @test isempty(intersect(e, o))               # the groups commute
            @test sort(vcat(e, o)) == collect(1:(L - 1)) # and cover every bond
        end
        @test_throws ArgumentError parity_bonds(6, :sideways)
    end

    @testset "bond_gates gives one gate per bond, tagged for that bond" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        @test length(g) == 5
        for i in 1:5
            @test g[i].inds[1].itags == psi[i].inds[2].itags
            @test g[i].inds[2].itags == psi[i + 1].inds[2].itags
        end
    end

    @testset "a full sweep preserves the norm in real time" begin
        psi = domain_wall_state(6); canonical!(psi, 1)
        g = bond_gates(psi; J = 1.0, delta = 1.0)
        n0 = norm(psi)
        parity_sweep!(psi, g, :even, -0.01im; KW..., rng = RNG())
        @test isapprox(norm(psi), n0; atol = 1e-11)
        parity_sweep!(psi, g, :odd, -0.01im; KW..., rng = RNG())
        @test isapprox(norm(psi), n0; atol = 1e-11)
    end

    @testset "the U(1) gate conserves total Sz" begin
        psi = domain_wall_state(6); canonical!(psi, 1)
        g = bond_gates(psi; J = 1.0, delta = 1.0)
        sz0 = total_sz(psi)
        for _ in 1:3, p in (:even, :odd)
            parity_sweep!(psi, g, p, -0.02im; KW..., rng = RNG())
        end
        @test isapprox(total_sz(psi), sz0; atol = 1e-12)
    end

    @testset "energy of the domain wall matches the analytic value" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi; J = 1.0, delta = 1.0)
        # Sz-basis product state: <SxSx + SySy> = 0 on every bond (purely
        # off-diagonal). Of the 5 bonds, 4 are aligned at <SzSz> = +1/4 and the
        # one straddling the wall is at -1/4: E = 4*(1/4) - 1/4 = 0.75.
        @test isapprox(energy(psi, g), 0.75; atol = 1e-12)
        # Neel: every bond anti-aligned -> 5 * (-1/4)
        @test isapprox(energy(neel_state(6), bond_gates(neel_state(6))), -1.25; atol = 1e-12)
        # XX has no SzSz at all, so a product state has exactly zero energy
        @test isapprox(energy(psi, bond_gates(psi; delta = 0.0)), 0.0; atol = 1e-12)
    end

    @testset "energy is unchanged by canonicalisation and by J scaling" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        e0 = energy(psi, g)
        canonical!(psi, 6); canonical!(psi, 2)
        @test isapprox(energy(psi, g), e0; atol = 1e-12)
        @test isapprox(energy(psi, bond_gates(psi; J = 3.0)), 3 * e0; atol = 1e-12)
    end

    @testset "a nothing gate is skipped, not applied" begin
        psi = domain_wall_state(6); canonical!(psi, 1)
        g = bond_gates(psi)
        g[3] = nothing                       # the only bond the gate can move
        before = [sz_expectation(psi, j) for j in 1:6]
        for _ in 1:2, p in (:even, :odd)
            parity_sweep!(psi, g, p, -0.05im; KW..., rng = RNG())
        end
        # every OTHER bond is aligned, so with bond 3 muted nothing can flip
        @test all(isapprox(sz_expectation(psi, j), before[j]; atol = 1e-10) for j in 1:6)
    end

    @testset "real time conserves the energy to Trotter order" begin
        # WARM START, ON PURPOSE. From a product state the bond rank can only
        # grow one step at a time (on a product bond the discarded projector
        # annihilates K1 outright, so only the seed opens anything), which means
        # a run with more steps has built more rank by the same physical time.
        # Measured cold, halving the step made the drift 19x WORSE -- that was
        # rank growth masquerading as time-step error, not a defect. Warmed up,
        # the scaling is textbook first order.
        function warm()
            psi = domain_wall_state(6); canonical!(psi, 1)
            g = bond_gates(psi)
            for _ in 1:4, p in (:even, :odd)
                parity_sweep!(psi, g, p, ComplexF64(-0.05im); KW..., rng = RNG())
            end
            return psi, g
        end
        function drift(tau, nsteps)
            psi, g = warm()
            e0 = energy(psi, g)
            for _ in 1:nsteps, p in (:even, :odd)
                parity_sweep!(psi, g, p, ComplexF64(-tau * im); KW..., rng = RNG())
            end
            return abs(energy(psi, g) - e0)
        end
        d1 = drift(0.04, 10)          # all three cover the same total time, 0.4
        d2 = drift(0.02, 20)
        d3 = drift(0.01, 40)
        @test d1 < 1e-2
        @test d2 < d1 && d3 < d2
        @test 1.6 < d1 / d2 < 2.6     # Lie-Trotter: halving tau halves the drift
        @test 1.6 < d2 / d3 < 2.6
    end

    @testset "many sweeps preserve the norm, not just one" begin
        # The single-sweep version of this test passed throughout while real-time
        # evolution was blowing the norm up by 128x over five steps: the defect
        # needs a bond of rank > 1 AND an O(tau) perp to appear at all.
        psi = domain_wall_state(6); canonical!(psi, 1)
        g = bond_gates(psi)
        for _ in 1:12, p in (:even, :odd)
            parity_sweep!(psi, g, p, ComplexF64(-0.02im); KW..., rng = RNG())
            @test isapprox(norm(psi), 1.0; atol = 1e-10)
        end
        @test maximum(bond_dims(psi)) > 1        # and it was not trivially frozen
    end

    @testset "the sweep reports its augmentation and truncation" begin
        psi = domain_wall_state(6); canonical!(psi, 1)
        g = bond_gates(psi)
        info = parity_sweep!(psi, g, :even, -0.05im; KW..., rng = RNG())
        @test info.aug_k >= 1
        @test info.aug_l >= 1
        @test info.discarded >= 0.0
        # a hard cap has to show up as discarded weight somewhere in the group
        psi2 = domain_wall_state(6); canonical!(psi2, 1)
        g2 = bond_gates(psi2)
        for _ in 1:4, p in (:even, :odd)
            parity_sweep!(psi2, g2, p, -0.1im; maxdim = 64, trunc_thresh = 1e-14,
                          rng = RNG())
        end
        tight = parity_sweep!(psi2, g2, :even, -0.1im; maxdim = 1,
                              trunc_thresh = 1e-14, rng = RNG())
        @test tight.discarded > 1e-6
    end

    @testset "the bond dimension grows, and maxdim stops it" begin
        psi = domain_wall_state(8); canonical!(psi, 1)
        g = bond_gates(psi)
        @test all(bond_dims(psi) .== 1)
        for _ in 1:4, p in (:even, :odd)
            parity_sweep!(psi, g, p, -0.1im; maxdim = 4, trunc_thresh = 1e-14,
                          rng = RNG())
        end
        @test maximum(bond_dims(psi)) > 1        # it grew
        @test maximum(bond_dims(psi)) <= 4       # but not past the cap
    end

    @testset "the chain stays canonical across a sweep" begin
        psi = domain_wall_state(6); canonical!(psi, 1)
        g = bond_gates(psi)
        for p in (:even, :odd)
            parity_sweep!(psi, g, p, -0.02im; KW..., rng = RNG())
        end
        canonical!(psi, 4)
        for i in 1:3
            @test isapprox(left_isometry_defect(psi[i]), 0.0; atol = 1e-11)
        end
        for i in 5:6
            @test isapprox(right_isometry_defect(psi[i]), 0.0; atol = 1e-11)
        end
        # bonds still line up
        for i in 1:5
            @test psi[i].inds[3].itags == psi[i + 1].inds[1].itags
            @test psi[i].inds[3].dir != psi[i + 1].inds[1].dir
        end
    end
end

@testset "directional sweep (adjoint L/R primitive)" begin
    include("../common/dense_reference.jl")

    @testset "applies every bond once, conserves charge, grows the bond" begin
        psi = domain_wall_state(6); canonical!(psi, 1)
        sz0 = total_sz(psi)
        info = directional_sweep!(psi, bond_gates(psi), :left_to_right,
                                  ComplexF64(-im * 0.05); KW..., rng = RNG())
        @test isapprox(total_sz(psi), sz0; atol = 1e-10)   # U(1) charge held
        @test maximum(bond_dims(psi)) > 1                  # entanglement grew
        @test info.aug_k >= 1
        @test_throws ArgumentError directional_sweep!(psi, bond_gates(psi),
                                                      :sideways, ComplexF64(-im * 0.05); KW...)
    end

    # The local adjoint property: one 2-site bond update inverts EXACTLY --
    # kls(+tau) undoes kls(-tau) to machine precision -- WHERE the augmented frame
    # is complete (2r >= ambient). Bond 3 of the L=6 domain wall is such a bond.
    # This is what makes L and R adjoint sweeps (L_{-tau} o R_{tau} = 1) and hence
    # a symmetric composition of them a valid higher-order building block. On a
    # full truncating sweep the relation holds only up to the 2r-Galerkin O(dt^2)
    # floor (see test_composition.jl) -- the ceiling that keeps this integrator at
    # second order under the strict Sulz bound.
    @testset "a complete-frame bond update is exactly invertible" begin
        for tau in (0.05, 0.1)
            psi = domain_wall_state(6); canonical!(psi, 3)
            f = bond_frame(psi, 3); th0 = frame_theta(f)
            gate = heisenberg_bond_gate(f.site_l, f.site_r)
            r1 = kls_bond_update(f, gate, ComplexF64(-im * tau); KW..., rng = RNG())
            psi[3] = r1.left_core; psi[4] = r1.right_core; psi.center = 4
            canonical!(psi, 3)
            f2 = bond_frame(psi, 3)
            r2 = kls_bond_update(f2, gate, ComplexF64(+im * tau); KW..., rng = RNG())
            th2 = to_concrete(r2.left_core * r2.right_core)
            @test isapprox(norm(th2 - th0), 0.0; atol = 1e-10)
        end
    end
end
