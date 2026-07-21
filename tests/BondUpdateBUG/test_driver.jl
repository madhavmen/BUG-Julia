using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG

# REAL TIME ONLY. The plan's second testset ran imaginary time; it is replaced
# here by real-time equivalents (energy conservation, Trotter order), which say
# more about the driver anyway.

@testset "bond_update_bug! driver" begin

    @testset "real-time evolution conserves the norm over 10 steps" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi; J = 1.0, delta = 1.0)
        info = bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.02, n_steps = 10, order = :strang,
                                     maxdim = 8, normalize = false))
        @test all(n -> isapprox(n, 1.0; atol = 1e-10), info.norms)
        @test length(info.times) == 10
        @test length(info) == 10
    end

    @testset "every record has one entry per step, and times advance" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        info = bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.03, n_steps = 7, maxdim = 8))
        for v in (info.times, info.norms, info.bond_dims, info.max_bond_dims,
                  info.aug_k_dims, info.aug_l_dims, info.discarded)
            @test length(v) == 7
        end
        @test info.times ≈ 0.03 .* (1:7)
        @test all(info.max_bond_dims[k] == maximum(info.bond_dims[k]) for k in 1:7)
        @test all(length(bd) == 5 for bd in info.bond_dims)   # L-1 bonds
    end

    @testset "normalize=true pins the norm to 1, and is recorded before rescaling" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        info = bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.05, n_steps = 5, maxdim = 2,
                                     normalize = true))
        @test isapprox(norm(psi), 1.0; atol = 1e-12)
        # maxdim=2 truncates, so the pre-rescale norms must dip below 1
        @test any(n -> n < 1.0 - 1e-9, info.norms)
    end

    # The direct end-to-end check that the missing-QN fill works: without it a
    # product state can never leave bond dimension 1, because on a product bond
    # the discarded projector annihilates K1 outright.
    @testset "a product state grows its bond dimension" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi; J = 1.0, delta = 1.0)
        @test all(bond_dims(psi) .== 1)
        info = bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.05, n_steps = 5, maxdim = 8))
        @test maximum(info.max_bond_dims) > 1
        @test issorted(info.max_bond_dims)          # monotone growth
    end

    @testset "missing_fill=0 leaves a product state at bond dimension 1" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        info = bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.05, n_steps = 5, maxdim = 8,
                                     missing_fill = 0))
        @test maximum(info.max_bond_dims) == 1      # negative control
    end

    @testset "maxdim caps the bond dimension" begin
        for cap in (2, 4)
            psi = domain_wall_state(8)
            g = bond_gates(psi)
            info = bond_update_bug!(psi, g;
                opts = BondUpdateOptions(dt = 0.1, n_steps = 8, maxdim = cap))
            @test maximum(info.max_bond_dims) <= cap
            @test all(maximum(bd) <= cap for bd in info.bond_dims)
        end
    end

    @testset "the U(1) charge survives the whole run" begin
        psi = domain_wall_state(6)
        sz0 = total_sz(psi)
        g = bond_gates(psi)
        bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.05, n_steps = 10, maxdim = 8))
        @test isapprox(total_sz(psi), sz0; atol = 1e-11)
    end

    @testset "a run is reproducible from its seed" begin
        function run(seed)
            psi = domain_wall_state(6)
            g = bond_gates(psi)
            info = bond_update_bug!(psi, g;
                opts = BondUpdateOptions(dt = 0.05, n_steps = 6, maxdim = 8,
                                         seed = seed))
            return psi, g, info
        end
        p1, g1, i1 = run(0x5EED)
        p2, _, i2 = run(0x5EED)
        @test i1.norms == i2.norms
        @test i1.bond_dims == i2.bond_dims
        @test isapprox(energy(p1, g1), energy(p2, g1); atol = 1e-13)

        # a different seed picks different fill directions but must still give a
        # legitimate state -- same charge, same norm
        p3, g3, _ = run(0x1234)
        @test isapprox(total_sz(p3), total_sz(p1); atol = 1e-11)
        @test isapprox(norm(p3), 1.0; atol = 1e-12)
    end

    @testset "Strang beats Lie at the same step" begin
        # Warm up first: from a product state the rank grows per STEP, so a cold
        # comparison measures rank growth rather than splitting error.
        function drift(order, dt, nsteps)
            psi = domain_wall_state(6)
            g = bond_gates(psi)
            bond_update_bug!(psi, g; opts = BondUpdateOptions(
                dt = 0.05, n_steps = 4, maxdim = 16, normalize = false))
            e0 = energy(psi, g)
            bond_update_bug!(psi, g; opts = BondUpdateOptions(
                dt = dt, n_steps = nsteps, order = order, maxdim = 16,
                normalize = false))
            return abs(energy(psi, g) - e0)
        end
        d_lie = drift(:lie, 0.04, 10)
        d_str = drift(:strang, 0.04, 10)
        @test d_str < d_lie
        # and Strang converges faster in the step size than Lie does
        @test drift(:strang, 0.02, 20) / d_str < drift(:lie, 0.02, 20) / d_lie
    end

    @testset "the Sulz bound holds at every bond of every step" begin
        # aug_k/aug_l are the per-step MAXIMA over the group's bonds, and the
        # kept rank is what the previous step left, so aug <= 2 * previous keep
        # is the run-level form of rank([U0|K1]) <= 2r.
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        info = bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.05, n_steps = 8, maxdim = 8,
                                     missing_fill = 4))
        for k in 2:8
            @test info.aug_k_dims[k] <= 2 * info.max_bond_dims[k - 1]
            @test info.aug_l_dims[k] <= 2 * info.max_bond_dims[k - 1]
        end
    end

    @testset "n_steps=0 is a no-op that still returns empty records" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        e0 = energy(psi, g)
        info = bond_update_bug!(psi, g; opts = BondUpdateOptions(n_steps = 0))
        @test isempty(info.times) && isempty(info.norms) && isempty(info.bond_dims)
        @test isapprox(energy(psi, g), e0; atol = 1e-13)
    end

    @testset "a bad order is rejected before the state is touched" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        before = bond_dims(psi)
        @test_throws ArgumentError bond_update_bug!(psi, g;
            opts = BondUpdateOptions(order = :sideways, n_steps = 3))
        @test bond_dims(psi) == before
    end

    @testset "the run reports its augmentation" begin
        psi = domain_wall_state(6)
        g = bond_gates(psi)
        info = bond_update_bug!(psi, g;
            opts = BondUpdateOptions(dt = 0.05, n_steps = 6, maxdim = 8))
        @test all(info.aug_k_dims .>= 1)
        @test all(info.aug_l_dims .>= 1)
        @test all(info.discarded .>= 0.0)
        # the proposed rank must exceed the kept rank at some point -- that is
        # what rank adaptation looks like
        @test any(info.aug_k_dims[k] >= info.max_bond_dims[k] for k in 1:6)
    end
end
