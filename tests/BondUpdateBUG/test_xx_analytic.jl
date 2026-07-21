using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
include("../common/free_fermion.jl")
include("../common/dense_reference.jl")

@testset "XX domain wall vs the free-fermion analytic solution" begin

    # Validate the REFERENCE before trusting it. The free-fermion formula is
    # derived on paper from the Hamiltonian; the dense path is built from the
    # integrator's own gate and ordering conventions. Agreement between two
    # independent derivations is what makes the comparison below meaningful.
    @testset "the analytic reference agrees with dense ED" begin
        L = 6
        H = dense_heisenberg(L; J = 1.0, delta = 0.0)
        v0 = dense_state(domain_wall_state(L))
        for t in (0.0, 0.05, 0.3, 1.0)
            v = dense_exact_propagate(H, v0, t)
            Sz = [real(v' * _embed(_SZ, j, L) * v) for j in 1:L]
            @test maximum(abs.(Sz .- xx_free_fermion_sz(L, t))) < 1e-11
        end
    end

    @testset "t=0 reproduces the domain wall exactly" begin
        @test xx_free_fermion_sz(6, 0.0) ≈ [0.5, 0.5, 0.5, -0.5, -0.5, -0.5]
        # and the total stays at zero: XX conserves particle number
        for t in (0.1, 0.5, 2.0)
            @test isapprox(sum(xx_free_fermion_sz(6, t)), 0.0; atol = 1e-12)
        end
    end

    @testset "bond_update_bug! matches the analytic profile" begin
        L, DT, N = 6, 0.01, 5
        psi = domain_wall_state(L)
        g = bond_gates(psi; J = 1.0, delta = 0.0)   # delta=0 => pure XX => solvable
        bond_update_bug!(psi, g; opts = BondUpdateOptions(
            dt = DT, n_steps = N, order = :strang, maxdim = 8,
            trunc_thresh = 1e-14, normalize = false))

        got  = [sz_expectation(psi, j) for j in 1:L]
        want = xx_free_fermion_sz(L, DT * N)
        @test maximum(abs.(got .- want)) < 1e-6     # O(dt^2) Trotter, not rank
        @test isapprox(sum(got), 0.0; atol = 1e-11)
    end

    @testset "the profile converges as the step is refined" begin
        # STEP RANGE CHOSEN SO TROTTER ERROR DOMINATES. XX is accurate enough here
        # that at dt <= 0.02 the error is already down at ~2e-8, where it is set by
        # how much rank the run has grown rather than by the step: from a product
        # state the bond can only grow one step at a time, so a finer run has more
        # rank at the same physical time. Measured at T=0.2, dt=0.01 came out
        # 3.58e-8 against dt=0.02's 2.32e-8 -- the finer step marginally WORSE,
        # which is that artifact and not a convergence failure (same trap as the
        # cold-start Trotter measurement in test_sweep.jl). Coarser steps put the
        # splitting error well above the floor and the ordering is unambiguous.
        L, T = 6, 0.2
        want = xx_free_fermion_sz(L, T)
        function err(dt)
            p = domain_wall_state(L)
            bond_update_bug!(p, bond_gates(p; J = 1.0, delta = 0.0);
                opts = BondUpdateOptions(dt = dt, n_steps = round(Int, T / dt),
                                         order = :strang, maxdim = 8,
                                         trunc_thresh = 1e-14, normalize = false))
            return maximum(abs.([sz_expectation(p, j) for j in 1:L] .- want))
        end
        e = [err(0.1), err(0.05), err(0.02)]
        @test all(e .< 1e-4)
        @test e[1] > e[2] > e[3]          # coarse is worse, monotonically
    end

    @testset "the wavefront is symmetric about the wall" begin
        # XX on a domain wall spreads symmetrically, so <Sz_j> = -<Sz_{L+1-j}>.
        # A gate wrong in one direction only would break this while still
        # conserving total Sz, so it is worth asserting separately.
        #
        # It is NOT exact, and should not be: the sweep visits bonds left to
        # right and the fill draws its random directions in that same fixed
        # order, so the two halves are not treated identically. Measured residual
        # asymmetry 3.7e-10 on a 0.49 signal -- a real artifact of the sweep
        # order, nine orders below the physics, but above machine precision.
        L = 6
        psi = domain_wall_state(L)
        bond_update_bug!(psi, bond_gates(psi; J = 1.0, delta = 0.0);
            opts = BondUpdateOptions(dt = 0.02, n_steps = 10, order = :strang,
                                     maxdim = 8, trunc_thresh = 1e-14,
                                     normalize = false))
        got = [sz_expectation(psi, j) for j in 1:L]
        for j in 1:(L ÷ 2)
            @test isapprox(got[j], -got[L + 1 - j]; atol = 1e-9)
        end
        @test got[3] < 0.5                # the wall has actually moved
    end
end
