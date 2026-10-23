using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

# ── WHY THIS FILE EXISTS ─────────────────────────────────────────────────────────────────────
#
# `fusion_basis` was 45% of a fused d=4 step, recomputing a PURE function. Flat profile of one
# truncating step (job 16342051 stage 3, L=20, chi=32, d=4), 57598 samples in the step:
#
#     svd_std 68.6% | cbe_expand 50.1% | sector_graded_sketch 48.1% | fusion_basis 44.8%
#     gesdd! (LAPACK, SELF) 40.7%      getIdentity 0.05%
#
# i.e. building the isometry is free and taking an SVD OF IT is nearly half the step. The result
# depends only on the legs' spaces, arrows, tags and prime levels, never on the tensor's data, so
# it is now memoised.
#
# ⛔⛔ THE DEFECT WAS INVISIBLE FOR A WHOLE CAMPAIGN, AND THAT IS WHAT THESE TESTS GUARD. Nothing
# measured allocation or call counts, so a step that churned 28.5 GiB at bond dimension 32 looked
# exactly like a step that was simply "slow", and the L=40 production runs were written off as
# expensive rather than broken. Correctness gates alone would never have caught it -- every number
# the campaign printed was RIGHT, just unaffordable.

norm_diff(a, b) = sqrt(abs(tensor_inner(to_concrete(a - b), to_concrete(a - b))))

# Bond frames spanning several states, deliberately including links that carry a NON-ZERO charge.
# ⚠ A vacuum link is a FALSE PASS here: its reachable set {+q,-q} is its own dual, so a wrong
# arrow or a key that drops the itag still agrees. That exact false pass hid a real sign bug
# (jobs 93413, 93458), and it would equally hide a colliding cache key.
function sample_frames()
    out = Any[]
    charged = Ref(false)
    for psi in (domain_wall_state(4), neel_state(6),
                product_state([:up, :down, :down, :up, :up, :down]))
        for i in 1:(length(psi) - 1)
            canonical!(psi, i)
            t = bond_frame(psi, i).U0
            any(q != dual_charge(symm(t), q) for (q, _) in t.spaces[1]) && (charged[] = true)
            push!(out, t)
        end
    end
    return out, charged[]
end

@testset "fusion_basis memo" begin
    set_symmetry!(:U1)

    @testset "a cache HIT returns exactly what a cold MISS returns" begin
        frames, saw_charged = sample_frames()
        @test saw_charged            # the test is vacuous on vacuum links only

        # Cold: every call is a miss, computed against an empty cache.
        cold = map(frames) do t
            clear_fusion_cache!()
            fusion_basis(t, 1, 2; tag = "cbeG")
        end

        # Warm: one shared cache across all of them, so later calls hit -- and a key that fails to
        # discriminate two different frames shows up here as a mismatch, not as a crash.
        clear_fusion_cache!()
        warm = map(t -> fusion_basis(t, 1, 2; tag = "cbeG"), frames)

        st = fusion_stats()
        # ⛔ THE CACHE MUST ACTUALLY HAVE HIT, or everything below passes for the wrong reason.
        @test st.calls == length(frames)
        @test st.misses < st.calls

        for (c, w) in zip(cold, warm)
            @test c.spaces == w.spaces
            @test string.(c.inds) == string.(w.inds)
            @test norm_diff(c, w) < 1e-12
        end
    end

    @testset "the tag is part of the key" begin
        frames, _ = sample_frames()
        t = first(frames)
        clear_fusion_cache!()
        a = fusion_basis(t, 1, 2; tag = "cbeF")
        b = fusion_basis(t, 1, 2; tag = "cbeG")
        # Two distinct tags must not share an entry: the fused leg is labelled with the tag, and
        # handing back the wrong one is how `cbe_bug` broke under U(1) before ("Duplicate TLIndex
        # with non-empty itag"). Two keys, and the labels genuinely differ.
        @test fusion_stats().misses == 2
        @test string(a.inds[3]) != string(b.inds[3])
    end

    @testset "the symmetry mode is part of the key" begin
        clear_fusion_cache!()
        set_symmetry!(:U1)
        fusion_basis(bond_frame(canonical!(neel_state(4), 1), 1).U0, 1, 2; tag = "cbeG")
        n1 = fusion_stats().keys
        set_symmetry!(:none)
        fusion_basis(bond_frame(canonical!(neel_state(4), 1), 1).U0, 1, 2; tag = "cbeG")
        # A :none tensor must never be served a :U1 entry -- the sector structure differs.
        @test fusion_stats().keys > n1
        set_symmetry!(:U1)
    end
end

# ── THE REGRESSION GUARD ─────────────────────────────────────────────────────────────────────
#
# ⛔ THIS IS THE TEST THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT, AND IT IS THE REASON THE FILE
# IS WORTH ITS RUNTIME. It asserts a BUDGET, not a wall-clock time, so it does not depend on the
# machine, on contention, or on how many other jobs share the node -- all of which made the
# original symptom look like ordinary slowness.
#
# MEASURED BEFORE THE MEMO (job 16342051 stage 1, `bug`, d=4, chi=32, cutoff 1e-8), step 2 --
# the first step whose rank reaches the cap and therefore truncates:
#
#     L=6   0.19 GiB      L=10   1.52 GiB      L=14  15.44 GiB      L=20  28.49 GiB
#
# L=14 is the cell used here: far enough up the curve that the defect is unmistakable, small
# enough that a HEALTHY run finishes in seconds.
@testset "a d=4 step stays inside its allocation budget" begin
    set_symmetry!(:none)
    clear_fusion_cache!()
    L = 14
    occ = hub4_cdw_occupations(L; doublon_frac = 0.5, seed = 1)
    psi = hub4_product_state(occ)
    mpo = hubbard_aa_mpo_d4(L; J = 1.0, Delta = 8.0, U = 4.0, beta = 532 / 738, phi = 0.0)
    step!(p) = cbe_bug_step!(p, mpo, ComplexF64(-0.1im); maxdim = 32, tol_basis = 1e-6,
                             tol_trunc = 1e-8, tol_expv = 1e-10)

    p = copy(psi)
    step!(p)                                  # step 1 never truncates -- not representative
    reset_fusion_stats!()
    gib = @allocated(step!(p)) / 2^30         # step 2 is the first truncating one
    st = fusion_stats()

    @info "d=4 step-2 budget" L gib fusion_calls=st.calls fusion_misses=st.misses

    # ⚠ 2.0 GiB is a CEILING, not a target. It sits ~7.7x below the 15.44 GiB this cell measured
    # before the memo, so it catches the regression with a wide margin, and it is loose enough
    # that ordinary changes to the sweep will not make it flaky. TIGHTEN IT once the post-memo
    # number is measured -- a ceiling far above the real cost stops guarding anything.
    @test gib < 2.0

    # And the memo must be doing the work: with 13 bonds x 2 sides x 2 half-sweeps the distinct
    # leg configurations are O(50), so a miss count in the thousands means the keys are churning
    # and the cache is decorative.
    @test st.misses < st.calls
end
