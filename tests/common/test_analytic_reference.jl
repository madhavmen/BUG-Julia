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
end
