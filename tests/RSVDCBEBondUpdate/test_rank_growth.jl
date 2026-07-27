using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

# WHY THE RANK GROWS: a Galerkin step at every bond, not just at the centre.
#
# The freeze this file guards against was subtle enough to survive 888 passing tests, so
# the mechanism is pinned directly rather than only its symptom.
#
# An expansion is LOSSLESS: `U_ex` spans `[U0 | C]`, and `Ŝ₀ = U_ex†Θ₀V_ex†` is zero on the
# new directions, so the state is bit-for-bit what it was and its Schmidt rank is unchanged.
# A widened bond therefore collapses at the next gauge move, and that is CORRECT REPORTING,
# not a lost expansion. Rank grows only where the state is EVOLVED in the widened space.
#
# With a Galerkin step at the centre alone, `room_l = d·χ_{c-1} − r` pinned the centre at 2
# forever because χ_{c-1} never moved: L=18 gave maxbond 2 and an error IDENTICAL at
# dt = 0.1, 0.05 and 0.025 (2.3314e-01) -- dt-independence being the signature of a pinned
# manifold rather than an integration error.

secset(t, l) = Set((q, d) for (q, d) in t.spaces[l])

@testset "rank growth needs a Galerkin step per bond" begin

    @testset "an expansion alone cannot change the Schmidt rank" begin
        # The property that makes the single-centre-step design impossible. Stated as a
        # test so nobody "fixes" the collapse by reaching for the gauge move again.
        set_symmetry!(:U1)
        L, c = 8, 4
        h = xxz_chain(L; delta = 0.0)
        psi = domain_wall_state(L)
        canonical!(psi, c)
        f = bond_frame(psi, c)
        ex = cbe_expand(f, h, c,
                        left_env_stack(psi, h; upto = c - 1),
                        right_env_stack(psi, h; downto = c + 2); dex = 0)
        @test ex.n_new_l > 0                          # directions ARE found
        RSVDCBEBondUpdate._absorb_left!(psi, c, ex.U_ex)
        @test leg_dim(psi[c], 3) == leg_dim(ex.U_ex, 3)   # bond is wider

        # ...but the state is unchanged, so the rank is not. `M = U_ex†psi[c]` is
        # (r+dex) x r, rank r, and canonicalising reports exactly that.
        canonical!(psi, 1)
        @test leg_dim(psi[c], 3) == f.old_rank
    end

    @testset "the rank grows, and keeps growing, with the per-bond step" begin
        # The headline regression: a melting domain wall MUST widen the centre cut.
        set_symmetry!(:U1)
        L = 8
        h = xxz_chain(L; delta = 0.0)
        psi = domain_wall_state(L)
        widths = Int[]
        for _ in 1:10
            cbe_bug_bond_update(psi, h, ComplexF64(0, -0.05); maxdim = 64,
                                trunc_thresh = 1e-12)
            push!(widths, maximum(bond_dims(psi)))
        end
        @test widths[end] > 2                         # the freeze is gone
        @test issorted(widths)                        # monotone, no ratchet-then-collapse
        canonical!(psi, L ÷ 2)
        @test length(secset(psi[L ÷ 2], 3)) > 1       # U(1) growth means new charge sectors
    end

    @testset "the error responds to dt" begin
        # The sharpest statement of the old bug: it was dt-INDEPENDENT. Refining dt must
        # now change the answer, whatever the accuracy happens to be.
        set_symmetry!(:U1)
        L = 8
        h = xxz_chain(L; delta = 0.0)
        final = Float64[]
        for dt in (0.1, 0.05)
            psi = domain_wall_state(L)
            for _ in 1:round(Int, 1.0 / dt)
                cbe_bug_bond_update(psi, h, ComplexF64(0, -dt); maxdim = 64,
                                    trunc_thresh = 1e-12)
            end
            p = copy(psi)
            push!(final, sum(magnetisation(p)) / norm(p)^2)
        end
        @test abs(final[1] - final[2]) > 1e-10
    end

    @testset "U(1) charge is conserved through the expansion" begin
        # The expansion opens new charge sectors; total Sz must still be exactly conserved,
        # or the sketch is populating sectors the dynamics cannot reach.
        set_symmetry!(:U1)
        L = 8
        h = xxz_chain(L; delta = 0.0)
        psi = domain_wall_state(L)
        q0 = total_sz(psi)
        for _ in 1:8
            cbe_bug_bond_update(psi, h, ComplexF64(0, -0.05); maxdim = 64,
                                trunc_thresh = 1e-12)
            p = copy(psi)
            @test abs(total_sz(p) / norm(p)^2 - q0) < 1e-10
        end
    end
end
