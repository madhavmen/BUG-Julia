using Test, LinearAlgebra
using LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("dense_reference.jl")

@testset "dense reference" begin

    @testset "Heisenberg matrix is Hermitian and Sz-block-diagonal" begin
        H = dense_heisenberg(6; J = 1.0, delta = 1.0)
        @test isapprox(norm(H - H'), 0.0; atol = 1e-14)
        Sz = dense_total_sz(6)
        @test isapprox(norm(H * Sz - Sz * H), 0.0; atol = 1e-12)
    end

    @testset "exact propagator is unitary" begin
        H = dense_heisenberg(6)
        U = exp(-im * 0.3 * Matrix(H))
        @test isapprox(norm(U * U' - I), 0.0; atol = 1e-12)
    end

    # THE CONVENTION ANCHOR. Site 1 is the most significant index and up is the
    # first local index; the assertions below pin that on both sides at once, so
    # the Kronecker ordering and the state ordering cannot drift apart.
    @testset "dense_state puts site 1 in the most significant bit, up first" begin
        # up up up down down down -> bits 000111 -> index 7 (0-based)
        v = dense_state(domain_wall_state(6))
        @test isapprox(norm(v), 1.0; atol = 1e-12)
        @test isapprox(abs(v[0b000111 + 1]), 1.0; atol = 1e-12)
        @test count(x -> abs(x) > 1e-12, v) == 1
        # all up -> index 0
        @test isapprox(abs(dense_state(product_state(fill(:up, 6)))[1]), 1.0; atol = 1e-12)
        # all down -> last index
        @test isapprox(abs(dense_state(product_state(fill(:down, 6)))[end]), 1.0; atol = 1e-12)
        # Neel up-first -> 010101
        @test isapprox(abs(dense_state(neel_state(6))[0b010101 + 1]), 1.0; atol = 1e-12)
    end

    @testset "dense Sz agrees with the MPS observable" begin
        Sz = dense_total_sz(6)
        for psi in (domain_wall_state(6), neel_state(6),
                    product_state([:up, :up, :down, :up, :down, :down]))
            v = dense_state(psi)
            @test isapprox(real(v' * Sz * v), total_sz(psi); atol = 1e-12)
        end
    end

    # The single assertion that ties the operator ordering to the state ordering.
    # If either convention were wrong on its own, these would disagree.
    @testset "dense energy agrees with the MPS energy" begin
        for (J, delta) in ((1.0, 1.0), (1.0, 0.0), (2.0, 0.5))
            H = dense_heisenberg(6; J = J, delta = delta)
            for psi in (domain_wall_state(6), neel_state(6))
                g = bond_gates(psi; J = J, delta = delta)
                v = dense_state(psi)
                @test isapprox(real(v' * H * v), energy(psi, g); atol = 1e-11)
            end
        end
    end

    @testset "the parity groups partition the Hamiltonian" begin
        H = dense_heisenberg(6)
        He = dense_parity_hamiltonian(6, :even)
        Ho = dense_parity_hamiltonian(6, :odd)
        @test isapprox(norm(He + Ho - H), 0.0; atol = 1e-14)
        # the two GROUPS do not commute -- that is the whole source of the
        # splitting error, so assert it rather than leave it implied
        @test norm(He * Ho - Ho * He) > 1e-6
        # but within a group the bonds are disjoint, hence commuting, which is
        # why exponentiating a group as one matrix is exact
        for i in parity_bonds(6, :even), j in parity_bonds(6, :even)
            i == j && continue
            a = dense_bond_term(6, i); b = dense_bond_term(6, j)
            @test isapprox(norm(a * b - b * a), 0.0; atol = 1e-12)
        end
    end

    @testset "Trotter converges to exact at second order" begin
        H = dense_heisenberg(6)
        v = normalize!(dense_state(domain_wall_state(6)))
        exact = dense_exact_propagate(H, v, 0.4)
        e1 = norm(dense_trotter_propagate(6, v, 0.04, 10) - exact)
        e2 = norm(dense_trotter_propagate(6, v, 0.02, 20) - exact)
        @test 3.0 < e1 / e2 < 5.0          # ratio ~4 => second order
    end

    @testset "Lie converges at first order, Strang at second" begin
        H = dense_heisenberg(6)
        v = normalize!(dense_state(domain_wall_state(6)))
        exact = dense_exact_propagate(H, v, 0.4)
        l1 = norm(dense_trotter_propagate(6, v, 0.04, 10; order = :lie) - exact)
        l2 = norm(dense_trotter_propagate(6, v, 0.02, 20; order = :lie) - exact)
        @test 1.6 < l1 / l2 < 2.6
        @test l1 > norm(dense_trotter_propagate(6, v, 0.04, 10; order = :strang) - exact)
    end

    @testset "both propagators are unitary on the state" begin
        H = dense_heisenberg(6)
        v = normalize!(dense_state(domain_wall_state(6)))
        @test isapprox(norm(dense_exact_propagate(H, v, 0.4)), 1.0; atol = 1e-12)
        for order in (:strang, :lie)
            @test isapprox(norm(dense_trotter_propagate(6, v, 0.04, 10; order = order)),
                           1.0; atol = 1e-12)
        end
        @test_throws ArgumentError dense_trotter_propagate(6, v, 0.04, 1; order = :sideways)
    end
end
