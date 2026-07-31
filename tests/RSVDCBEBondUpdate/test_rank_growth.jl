using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

# WHY THE RANK GROWS: the evolved centre core propagates out through RETAINED frames.
#
# The freeze this file guards against was subtle enough to survive 888 passing tests, so
# the mechanism is pinned directly rather than only its symptom.
#
# An expansion is LOSSLESS: `U_ex` spans `[U0 | C]`, and `Ŝ₀ = U_ex†Θ₀V_ex†` is zero on the
# new directions, so the state is bit-for-bit what it was and its Schmidt rank is unchanged.
# A widened bond therefore collapses at the next gauge move, and that is CORRECT REPORTING,
# not a lost expansion. Rank grows only where the state is EVOLVED in the widened space.
#
# THERE IS EXACTLY ONE GALERKIN STEP, at the centre, and that is enough: the evolved `S(t1)`
# is DENSE and the augmented frames are RETAINED, so it propagates out through
# `W_{i+1}…W_c` / `Z_c…Z_{L-1}` and every bond ends up carrying `r + Dex`. A Galerkin step
# per bond over-counts `tau` (it applies the global `H` at `L−1` bonds per sweep) and was
# abandoned -- docs/current_work.md section 5. An earlier note here claimed the per-bond
# step was the mechanism; it is not, and the tests below never depended on it.
#
# What DID freeze, and is what these tests guard: a centre Galerkin whose frames were
# written straight back into the state, so `room_l = d·χ_{c-1} − r` pinned the centre at 2
# forever because χ_{c-1} never moved. L=18 gave maxbond 2 and an error IDENTICAL at
# dt = 0.1, 0.05 and 0.025 (2.3314e-01) -- dt-independence being the signature of a pinned
# manifold rather than an integration error. The fix was the carry-based sweep (frames
# accumulated in `W`/`Z`, state assembled only after the step), not more Galerkin steps.

secset(t, l) = Set((q, d) for (q, d) in t.spaces[l])

@testset "rank grows through retained frames, not through a per-bond Galerkin" begin

    @testset "an expansion alone cannot change the Schmidt rank" begin
        # The property behind the whole design. Stated as a test so nobody "fixes" the
        # collapse by reaching for the gauge move again: it is the state's evolution in the
        # widened space that grows the rank, never the widening.
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

    @testset "the rank grows, and keeps growing, over repeated steps" begin
        # The headline regression: a melting domain wall MUST widen the centre cut.
        set_symmetry!(:U1)
        L = 8
        h = xxz_chain(L; delta = 0.0)
        psi = domain_wall_state(L)
        widths = Int[]
        for _ in 1:10
            cbe_lubich_sweep(psi, h, ComplexF64(0, -0.05); maxdim = 64,
                                trunc_thresh = 1e-12)
            push!(widths, maximum(bond_dims(psi)))
        end
        @test widths[end] > 2                         # the freeze is gone
        @test issorted(widths)                        # monotone, no ratchet-then-collapse
        canonical!(psi, L ÷ 2)
        @test length(secset(psi[L ÷ 2], 3)) > 1       # U(1) growth means new charge sectors
    end

    @testset "the answer responds to dt" begin
        # The sharpest statement of the old bug: it was dt-INDEPENDENT. Refining dt must now
        # change the answer, whatever the accuracy happens to be.
        #
        # MUST BE A NON-CONSERVED OBSERVABLE. An earlier version of this test used
        # `sum(magnetisation(...))`, i.e. total Sz -- which U(1) conserves exactly, so it is
        # dt-independent for a CORRECT integrator too and the test proved nothing (it read
        # 1.3e-15 and "failed" a working sweep). The site-resolved profile is what actually
        # carries the dynamics.
        set_symmetry!(:U1)
        L = 8
        h = xxz_chain(L; delta = 0.0)
        profiles = Vector{Vector{Float64}}()
        for dt in (0.1, 0.05)
            psi = domain_wall_state(L)
            for _ in 1:round(Int, 1.0 / dt)
                cbe_lubich_sweep(psi, h, ComplexF64(0, -dt); maxdim = 64,
                                    trunc_thresh = 1e-12)
            end
            p = copy(psi)
            push!(profiles, magnetisation(p) ./ norm(p)^2)
        end
        @test maximum(abs.(profiles[1] .- profiles[2])) > 1e-10
        # and total Sz is conserved at BOTH steps, which is the thing the old test measured
        for pr in profiles
            @test abs(sum(pr) - 0.0) < 1e-10
        end
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
            cbe_lubich_sweep(psi, h, ComplexF64(0, -0.05); maxdim = 64,
                                trunc_thresh = 1e-12)
            p = copy(psi)
            @test abs(total_sz(p) / norm(p)^2 - q0) < 1e-10
        end
    end
end
