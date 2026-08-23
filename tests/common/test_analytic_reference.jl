# The analytic references, pinned against exact diagonalisation BEFORE anything is allowed to
# use them as a reference.
#
# WHY THIS FILE EXISTS SEPARATELY. Every other test in the suite compares an integrator against
# `analytic_reference.jl`. If a formula in there is wrong, every one of those tests is wrong in
# the same direction and they all agree with each other -- the worst possible failure mode,
# because a consistent wrong answer looks like a validated one. So the formulas are checked
# here, against a dense diagonalisation that shares nothing with them, and only the constants
# that survive get used elsewhere.
#
# The risk being guarded against is CONVENTION, not physics: the Haldane-Shastry constant depends
# on the coupling prefactor and on whether each pair carries a `-1/4` shift, and several
# conventions circulate. Likewise the Bethe quantum numbers `I_j` have a parity convention that a
# plausibility argument will not catch.

using Test
using LinearAlgebra

@testset "analytic references" begin

    # Dense two-body Hamiltonian, by an explicit pair loop -- deliberately NOT the automaton and
    # deliberately NOT `analytic_reference.jl`'s own coupling function.
    SP = ComplexF64[0 1; 0 0]; SM = ComplexF64[0 0; 1 0]; SZ = ComplexF64[0.5 0; 0 -0.5]
    emb(o, i, L) = kron(Matrix{ComplexF64}(I, 2^(i - 1), 2^(i - 1)), o,
                        Matrix{ComplexF64}(I, 2^(L - i), 2^(L - i)))
    function dense_pairs(L, J)
        H = zeros(ComplexF64, 2^L, 2^L)
        for i in 1:(L - 1), j in (i + 1):L
            J[i, j] == 0 && continue
            H .+= J[i, j] * (0.5 * (emb(SP, i, L) * emb(SM, j, L) +
                                    emb(SM, i, L) * emb(SP, j, L)) +
                             emb(SZ, i, L) * emb(SZ, j, L))
        end
        return H
    end
    hs_matrix(L) = [i < j ? pi^2 / (L^2 * sin(pi * (i - j) / L)^2) : 0.0
                    for i in 1:L, j in 1:L]
    ring_matrix(L) = [(j == i + 1 || (i == 1 && j == L)) ? 1.0 : 0.0 for i in 1:L, j in 1:L]

    @testset "the HS coupling is periodic, i.e. the ring is in J" begin
        for L in (6, 8, 10)
            Jr = hs_couplings_analytic(L)
            # J(r) = J(L-r): the pair (1,L) is as STRONG as a nearest-neighbour pair, which is
            # the whole reason no wrap-around term has to be added on top.
            for r in 1:(L - 1)
                @test Jr[r] ≈ Jr[L - r] atol = 1e-12
            end
            @test Jr[1] ≈ pi^2 / (L^2 * sin(pi / L)^2) atol = 1e-12
            # It is also the second implementation of `haldane_shastry_couplings`, so the two
            # must agree -- if they do not, one of them is the bug and this says which.
            Jm = haldane_shastry_couplings(L)
            for i in 1:(L - 1), j in (i + 1):L
                @test Jm[i, j] ≈ Jr[j - i] atol = 1e-12
            end
        end
    end

    @testset "hs_ground_energy == exact diagonalisation" begin
        # THE CHECK THE FORMULA'S USE DEPENDS ON. `-pi^2 (L^2+5)/(24L)`, against a dense
        # diagonalisation of the same coupling matrix. L=4,6,8 only -- 2^10 x 2^10 dense is
        # affordable but adds nothing here, since a convention error is L-independent.
        for L in (4, 6, 8)
            e_ed = eigen(Hermitian(Matrix(dense_pairs(L, hs_matrix(L))))).values[1]
            @test hs_ground_energy(L) ≈ e_ed atol = 1e-9
        end
    end

    @testset "one-magnon dispersion == exact diagonalisation in that sector" begin
        # The one-magnon states are exact eigenstates of ANY translation-invariant two-body H on
        # a ring, so the dispersion can be checked by diagonalising the L x L block of H in the
        # one-down-spin sector -- no integrability needed and no formula shared with the source.
        for L in (6, 8), Jm in (hs_matrix(L), ring_matrix(L))
            # Read `J(r)` off the coupling MATRIX by ring translation rather than from a formula,
            # so the dispersion is checked against the operator that was actually diagonalised.
            # A matrix that is not translation invariant on the ring would then show up as a
            # mismatch below instead of being silently symmetrised into one.
            Jr = [maximum(Jm[i, j] for i in 1:L, j in 1:L
                          if j > i && (j - i == r || j - i == L - r); init = 0.0)
                  for r in 1:(L - 1)]
            H = Matrix(dense_pairs(L, Jm))
            # basis: |j> = S_j^- |up...up>, index = the state with bit j set (down)
            idx(j) = 1 + (1 << (L - j))
            B = [H[idx(a), idx(b)] for a in 1:L, b in 1:L]
            evals = sort(real.(eigen(Hermitian(B)).values))
            analytic = sort([magnon_energy(Jr, n; L = L) for n in 0:(L - 1)])
            @test evals ≈ analytic atol = 1e-9
            # `n = 0` is the polarised state rotated, so it must be degenerate with |F>.
            @test magnon_energy(Jr, 0; L = L) ≈ magnon_reference_energy(Jr; L = L) atol = 1e-12
            fullup = 1
            @test real(H[fullup, fullup]) ≈ magnon_reference_energy(Jr; L = L) atol = 1e-10
        end
    end

    @testset "bethe_xxx_ring_energy == exact diagonalisation" begin
        for L in (4, 6, 8, 10)
            e, lam, res = bethe_xxx_ring_energy(L)
            # The residual is asserted FIRST: an unconverged iterative solve must not be able to
            # pass as an analytic reference by accident.
            @test res < 1e-12
            @test length(lam) == L ÷ 2
            e_ed = eigen(Hermitian(Matrix(dense_pairs(L, ring_matrix(L))))).values[1]
            @test e ≈ e_ed atol = 1e-8
        end
    end

    @testset "bethe_xxx_open_energy == exact diagonalisation" begin
        # THE OPEN CHAIN IS THE CAMPAIGN'S HEISENBERG REFERENCE, so this is the test that decides
        # whether every Heisenberg error number means anything. It is also the one that caught a
        # solver which converged to residual 1e-14 and was 0.24 BELOW ED at every L -- so the
        # residual is asserted AND the value is checked, because the first does not imply the
        # second.
        for L in (4, 6, 8, 10, 12)
            e, lam, res = bethe_xxx_open_energy(L)
            @test res < 1e-12
            @test length(lam) == L ÷ 2
            # open chain: L-1 bonds, so the nearest-neighbour matrix without the wrap-around
            Jm = zeros(Float64, L, L)
            for i in 1:(L - 1)
                Jm[i, i + 1] = 1.0
            end
            e_ed = eigen(Hermitian(Matrix(dense_pairs(L, Jm)))).values[1]
            @test e ≈ e_ed atol = 1e-10
        end
    end

    @testset "the open chain approaches Hulthen from ABOVE, and more slowly than the ring" begin
        # Not decoration: the open chain carries an O(1/L) BOUNDARY correction where the ring's
        # leading correction is O(1/L^2). If a future edit silently turned the open solver into
        # the ring one, the energies would still look plausible -- this is what would catch it.
        e64, _, r64 = bethe_xxx_open_energy(64)
        @test r64 < 1e-12
        @test e64 / 64 > hulthen_energy_density()            # from above
        @test abs(e64 / 64 - hulthen_energy_density()) > 1e-3  # slower than the ring's 5e-4
    end

    @testset "Hulthen is a LIMIT and is labelled as one" begin
        @test hulthen_energy_density() ≈ 0.25 - log(2) atol = 1e-15
        # Finite-size corrections fall off as 1/L^2, so the L=10 ring is still ~2% away. This
        # test exists to record that the limit is NOT a finite-L reference, so that no later
        # test compares an L=6 number against it.
        e10, _, _ = bethe_xxx_ring_energy(10)
        @test abs(e10 / 10 - hulthen_energy_density()) > 1e-3
        e64, _, r64 = bethe_xxx_ring_energy(64)
        @test r64 < 1e-12
        @test abs(e64 / 64 - hulthen_energy_density()) < 5e-4
    end

    # ── the TFIM Majorana reference ──────────────────────────────────────────────────────────
    #
    # Same contract as everything above: pinned HERE against a dense propagator, and used
    # everywhere else INSTEAD of one. The integrator tests must not build 2^L objects -- that is
    # what caps them at L~10, and the `rexpand` defect this reference exists to catch is four
    # orders larger at L=16 than at the L a dense propagator affords.
    #
    # The risk is again CONVENTION and not physics: the Majorana factors of two, and the σˣ basis
    # in which `X` is DIAGONAL and `Z` is the flip.
    @testset "tfim_x_profile matches dense propagation" begin
        # Dense TFIM by an explicit bit loop, sharing no code with `free_fermion.jl`.
        function dense_tfim(L, J, h)
            D = 2^L
            H = zeros(ComplexF64, D, D)
            for s in 0:(D - 1)
                H[s + 1, s + 1] = -h * sum(((s >> i) & 1) == 0 ? 1.0 : -1.0 for i in 0:(L - 1))
                for i in 0:(L - 2)
                    H[(s ⊻ (1 << i) ⊻ (1 << (i + 1))) + 1, s + 1] += -J
                end
            end
            return H
        end

        # ⛔ ASYMMETRIC STATES ARE THE POINT. A state symmetric under chain reversal is a FIXED
        # POINT of a mirrored site convention, so it agrees whether or not the two orderings
        # match -- the polarised and kink cases below would pass a reference with its sites
        # indexed backwards. The ragged one is what actually tests the ordering.
        L, J, h, t = 6, 1.0, 0.7, 0.9
        F = eigen(Hermitian(dense_tfim(L, J, h)))
        for (nm, dirs) in ("polarised"  => fill(:plus, L),
                           "kink"       => [i <= L ÷ 2 ? :plus : :minus for i in 1:L],
                           "asymmetric" => [:plus, :minus, :minus, :plus, :minus, :plus])
            v0 = zeros(ComplexF64, 2^L)
            s  = 0
            for (i, d) in enumerate(dirs); d === :minus && (s |= 1 << (i - 1)); end
            v0[s + 1] = 1.0
            vt = F.vectors * (cis.(-t .* F.values) .* (F.vectors' * v0))
            want = [sum(abs2(vt[b + 1]) * (((b >> (j - 1)) & 1) == 0 ? 1.0 : -1.0)
                        for b in 0:(2^L - 1)) for j in 1:L]
            @test tfim_x_profile(L, t, dirs; J = J, h = h) ≈ want atol = 1e-12
        end

        # `t = 0` returns the state it was handed -- catches a generator that is accidentally
        # applied twice, or a covariance built with the wrong sign.
        dirs = [:plus, :minus, :minus, :plus, :minus, :plus]
        @test tfim_x_profile(L, 0.0, dirs) ≈ [d === :plus ? 1.0 : -1.0 for d in dirs] atol = 1e-14

        # The generator is REAL ANTISYMMETRIC, which is what makes `exp(At)` orthogonal and the
        # congruence norm-preserving. A wrong factor of `i` anywhere would break this.
        A = tfim_majorana_generator(8; J = 1.0, h = 1.0)
        @test A ≈ -transpose(A) atol = 1e-15

        # And the reference runs where a dense propagator cannot -- the regime it exists for.
        prof = tfim_x_profile(64, 1.0, [i <= 32 ? :plus : :minus for i in 1:64])
        @test all(abs.(prof) .<= 1 + 1e-12)
        @test length(prof) == 64
    end
end
