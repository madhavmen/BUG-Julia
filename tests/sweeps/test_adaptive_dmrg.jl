# tests/sweeps/test_adaptive_dmrg.jl
#
# simulation_plan_2 Task 4: the CBE-DMRG ground-state search runs on the SAME adaptive RSVD
# engine as `cbe_bug_step!` -- Tests A/B against `tol_basis`, the `maxdim`-derived work ceiling,
# the budgeted split against `tol_trunc`. The fixed-knob path is removed.
#
# References are exact: dense ED of the open Heisenberg chain, and sparse ED in the S^z = 0
# sector of a 4 x 3 square cylinder (`pairs_sparse` and `square_cylinder_mpo` read ONE coupling
# matrix, so the two cannot describe different lattices).

include(joinpath(@__DIR__, "..", "..", "benchmarks", "exact_sparse.jl"))

@testset "adaptive CBE-DMRG (simulation_plan_2 Task 4)" begin
    set_symmetry!(:U1)

    @testset "chain: exact energy, ledger per bond, variational" begin
        L = 10
        mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
        e_ed = eigen(Hermitian(Matrix(dense_heisenberg(L)))).values[1]
        psi = neel_state(L)
        r = dmrg_cbe1s!(psi, mpo; n_sweeps = 20, etol = 1e-11, tol_basis = 1e-6,
                        tol_trunc = 1e-10, maxdim = 64)
        @test r.energies[end] ≈ e_ed atol = 1e-8
        @test r.energies[end] >= e_ed - 1e-9                 # variational
        @test maximum(r.max_bond_dims) > 1                   # CBE grew the rank from Neel
        @test all(s in (:met, :budget_unmet) for s in r.status)
        lg = r.ledgers[end]
        @test lg isa AdaptiveLedger
        # One log entry per bond visit: L-1 left-to-right, L-1 right-to-left.
        @test length(lg.bond_log) == 2 * (L - 1)
        @test all(haskey(e, :sample) && haskey(e, :tail) && haskey(e, :discarded)
                  for e in lg.bond_log)
        @test Set(e.side for e in lg.bond_log) == Set((:right, :left))
        # The adaptive sketch drew probes (RSVD path, not the exact basis).
        @test lg.n_probes > 0
        # The split honoured its budget: no bond discarded more than its share.
        @test all(e.discarded <= 1e-10 / sqrt(L - 1) * (1 + 1e-8) for e in lg.bond_log)
    end

    @testset "4x3 square cylinder: exact S^z = 0 energy" begin
        Lx, Ly = 4, 3
        N = Lx * Ly
        Jm = square_cylinder_couplings(Lx, Ly)
        H, _, _ = pairs_sparse(N, Jm)
        e_ed = eigen(Hermitian(Matrix(H))).values[1]
        for exact in (false, true)
            psi = neel_state(N)
            r = dmrg_cbe1s!(psi, square_cylinder_mpo(Lx, Ly); n_sweeps = 30, etol = 1e-11,
                            tol_basis = 1e-6, tol_trunc = 1e-10, maxdim = 128, exact = exact)
            @test r.energies[end] ≈ e_ed atol = 1e-7
            @test r.energies[end] >= e_ed - 1e-9
        end
    end

    @testset "square J1-J2: diagonal bonds (U17) and exact energy" begin
        Lx, Ly = 4, 3
        N = Lx * Ly
        J1m = square_cylinder_couplings(Lx, Ly)
        Jm = square_cylinder_couplings(Lx, Ly; J2 = 0.5)
        D = Jm - J1m                                # the J2 part alone
        # Independent edge list: every diagonal joins (x, y) to (x+1, y+-1 mod Ly), and there
        # are exactly 2 Ly (Lx - 1) of them on a periodic cylinder.
        xy(s) = ((s - 1) ÷ Ly + 1, (s - 1) % Ly + 1)
        pairs = [(i, j) for i in 1:N, j in 1:N if D[i, j] != 0]
        @test length(pairs) == 2 * Ly * (Lx - 1)
        @test all(D[i, j] == 0.5 for (i, j) in pairs)
        @test all(begin
                      (xi, yi), (xj, yj) = xy(i), xy(j)
                      abs(xj - xi) == 1 && mod(yj - yi, Ly) in (1, Ly - 1)
                  end for (i, j) in pairs)
        @test square_cylinder_couplings(Lx, Ly; J2 = 0.0) == J1m   # default unchanged

        H, _, _ = pairs_sparse(N, Jm)
        e_ed = eigen(Hermitian(Matrix(H))).values[1]
        psi = neel_state(N)
        r = dmrg_cbe1s!(psi, square_cylinder_mpo(Lx, Ly; J2 = 0.5); n_sweeps = 30,
                        etol = 1e-11, tol_basis = 1e-6, tol_trunc = 1e-10, maxdim = 128)
        @test r.energies[end] ≈ e_ed atol = 1e-7
        @test r.energies[end] >= e_ed - 1e-9
    end

    @testset "tol_basis is the expansion knob; maxdim binding is reported" begin
        L = 12
        mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
        widths = Dict{Float64, Int}()
        for tb in (1e-2, 1e-6)
            psi = neel_state(L)
            r = dmrg_cbe1s!(psi, mpo; n_sweeps = 2, tol_basis = tb, tol_trunc = 1e-10,
                            maxdim = 64)
            widths[tb] = sum(e.width for lg in r.ledgers for e in lg.bond_log)
        end
        # A looser basis budget keeps fewer CBE directions in the first sweep.
        @test widths[1e-2] < widths[1e-6]

        psi = neel_state(L)
        r = dmrg_cbe1s!(psi, mpo; n_sweeps = 4, tol_basis = 1e-6, tol_trunc = 1e-12, maxdim = 4)
        @test maximum(bond_dims(psi)) <= 4
        @test r.status[end] === :budget_unmet
        @test :trunc in r.ledgers[end].unmet
        @test r.ledgers[end].maxdim_reached
    end

    @testset "mpo_variance matches dense <H^2> - <H>^2" begin
        # A NON-eigenstate (DMRG at a binding cap), so the variance is not trivially zero; then
        # the converged state, where it must vanish. Chain and a frustrated 3x3 J1-J2 cylinder.
        for (mpo, Hd, N) in ((xxz_mpo(8; J = 1.0, delta = 1.0), Matrix(dense_heisenberg(8)), 8),
                             (square_cylinder_mpo(3, 3; J2 = 0.5),
                              dense_long_range_pairs(9, square_cylinder_couplings(3, 3; J2 = 0.5)), 9))
            psi = neel_state(N)
            dmrg_cbe1s!(psi, mpo; n_sweeps = 2, tol_basis = 1e-6, tol_trunc = 1e-12, maxdim = 3)
            v = dense_state(psi); v ./= norm(v)
            e = real(dot(v, Hd * v))
            want = real(dot(Hd * v, Hd * v)) - e^2
            @test want > 1e-4                                     # really not an eigenstate
            @test mpo_variance(psi, mpo) ≈ want rtol = 1e-9 atol = 1e-11
            psi = neel_state(N)
            dmrg_cbe1s!(psi, mpo; n_sweeps = 20, etol = 1e-12, tol_basis = 1e-7,
                        tol_trunc = 1e-12, maxdim = 64)
            @test abs(mpo_variance(psi, mpo)) < 1e-8
        end
    end

    @testset "no tolerance named: adaptive at the DMRG's own (medium-loose) defaults" begin
        @test RSVDCBEBondUpdate.DMRG_TOL_DEFAULTS.basis > RSVDCBEBondUpdate.TOL_DEFAULTS.basis
        @test RSVDCBEBondUpdate.DMRG_TOL_DEFAULTS.trunc > RSVDCBEBondUpdate.TOL_DEFAULTS.trunc
        L = 10
        mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
        psi = neel_state(L)
        r = dmrg_cbe1s!(psi, mpo; n_sweeps = 20, maxdim = 64)
        @test r.ledgers[end] isa AdaptiveLedger
        e_ed = eigen(Hermitian(Matrix(dense_heisenberg(L)))).values[1]
        @test r.energies[end] >= e_ed - 1e-9
        @test r.energies[end] - e_ed < 1e-4 * L
        # The split budget really is the default's: no bond above tol_trunc / sqrt(L - 1).
        @test all(e.discarded <= RSVDCBEBondUpdate.DMRG_TOL_DEFAULTS.trunc / sqrt(L - 1) * (1 + 1e-8)
                  for e in r.ledgers[end].bond_log)
    end

    @testset "stopping threshold derived from the tolerances (dmrg_etol)" begin
        @test RSVDCBEBondUpdate.dmrg_etol(1e-3, 1e-4, 20) ≈ 0.1 * 20 * (2.5e-9 + 2e-9)
        @test RSVDCBEBondUpdate.dmrg_etol(1e-8, 1e-10, 20) == 1e-12 * 20      # round-off floor
        # Small on purpose: the fixed-etol arm is the slow one by construction (job 17265882 took
        # 12 min at L = 12 / 40 sweeps); L = 8 shows the same ordering in seconds.
        L = 8
        mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
        e_ed = eigen(Hermitian(Matrix(dense_heisenberg(L)))).values[1]
        tb, tt = 0.1, 1e-8
        auto = dmrg_cbe1s!(neel_state(L), mpo; n_sweeps = 20, tol_basis = tb, tol_trunc = tt,
                           maxdim = 16)
        fixed = dmrg_cbe1s!(neel_state(L), mpo; n_sweeps = 20, etol = 1e-10, tol_basis = tb,
                            tol_trunc = tt, maxdim = 16)
        @test length(auto.energies) <= length(fixed.energies)
        # ...and still within what the tolerances allow (square law, 10x slack)
        @test auto.energies[end] - e_ed <= 10 * L * (2.5e-3 * tb^2 + 0.2 * tt^2)
    end

    @testset "the fixed-knob path is gone" begin
        L = 8
        mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
        @test_throws MethodError dmrg_cbe1s!(neel_state(L), mpo; n_sweeps = 1,
                                             tol_basis = 1e-6, dex = 4)
        @test_throws MethodError dmrg_cbe1s!(neel_state(L), mpo; n_sweeps = 1,
                                             trunc_thresh = 1e-14)
    end
end
