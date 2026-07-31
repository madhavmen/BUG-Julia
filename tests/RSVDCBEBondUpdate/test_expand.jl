using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

# Stage 2c: the expansion -- preselect, final select, direct-sum expand.
#
# Four properties carry the design, and each has a specific failure it is guarding:
#
#   ISOMETRY          `oplus` concatenates blocks without orthogonalising them, so
#                     [U0 | new] is a frame only if the new block is EXACTLY orthogonal to
#                     U0. Missing the RepairOrtho second pass is what duplicated a column
#                     in `augment.jl` and grew the norm 128x over five steps.
#   LOSSLESS          the expansion contains the old frame, so U_ex† Θ₀ V_ex† loses
#                     nothing and tau = 0 is the identity on the state. This is also what
#                     makes the faithful and no-overlap forms coincide (docs/cbe_bug.md §3)
#                     and what makes `ExpandOrthoCenter`'s explicit zero-padding needless.
#   SECTOR OPENING    the point of the whole exercise. Tested as an A/B against the K/L
#                     complement, which provably cannot open an empty sector.
#   2-SITE LIMIT      with the cap lifted and dex = the whole complement, the expansion
#                     spans the full local space, so the bond update becomes exact.

"The K/L complement's reach, without the random fill: `P⊥ (HΘ V0†)`, one application.
No `expv` needed -- the charge sectors of `K1` are set by the first order already, which
is the whole content of the claim that K/L cannot open an empty sector."
function kl_complement_charges(f::BondFrame, gate, tl, tr)
    th = frame_theta(f)
    HK = to_concrete(contract(apply_gate(gate, th, tl, tr), (3, 4), f.V0', (2, 3)))
    K1 = to_concrete(perp_component(f.U0, HK))
    norm(K1) < 1e-14 && return Set()
    return Set(q for (q, _) in to_concrete(svd(K1, (1, 2); cutoff = 1e-10).U).spaces[3])
end

"Reassemble the state's two-site block from an expanded frame pair."
function project_and_rebuild(ex::CBEExpansion, th)
    S = to_concrete(contract(contract(ex.U_ex', (1, 2), th, (1, 2)),
                             (2, 3), ex.V_ex', (2, 3)))
    return to_concrete((ex.U_ex * S) * ex.V_ex), S
end

function evolved(L::Int; n_steps = 3, dt = 0.05, maxdim = 16)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

@testset "CBE expansion" begin

    @testset "expanded frames are isometries" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                ex = cbe_expand(f, h, i,
                                left_env_stack(psi, h; upto = i - 1),
                                right_env_stack(psi, h; downto = i + 2);
                                rng = Random.MersenneTwister(0x5EED))
                @test left_isometry_defect(ex.U_ex) < 1e-11
                @test right_isometry_defect(ex.V_ex) < 1e-11
            end
        end
    end

    @testset "projection onto the expansion is lossless" begin
        set_symmetry!(:U1)
        for L in (4, 6)
            h = xxz_chain(L)
            psi = evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                th = frame_theta(f)
                ex = cbe_expand(f, h, i,
                                left_env_stack(psi, h; upto = i - 1),
                                right_env_stack(psi, h; downto = i + 2);
                                rng = Random.MersenneTwister(0x5EED))
                rebuilt, S = project_and_rebuild(ex, th)
                @test norm(to_concrete(rebuilt - th)) < 1e-11 * max(1.0, norm(th))
                # ... and the core picked up only zeros, so there was nothing to pad.
                @test abs(norm(S) - norm(f.S0)) < 1e-11
            end
        end
    end

    @testset "respects the Sulz bound and the dex budget" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        for i in 1:(L - 1)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            r = f.old_rank
            lenv = left_env_stack(psi, h; upto = i - 1)
            renv = right_env_stack(psi, h; downto = i + 2)
            for dex in (1, 2, 0)                  # 0 == the full Sulz budget r
                ex = cbe_expand(f, h, i, lenv, renv; dex = dex,
                                rng = Random.MersenneTwister(0x5EED))
                @test leg_dim(ex.U_ex, 3) <= 2r
                @test leg_dim(ex.V_ex, 1) <= 2r
                @test ex.n_new_l <= (dex <= 0 ? r : dex)
                @test ex.n_new_r <= (dex <= 0 ? r : dex)
                @test leg_dim(ex.U_ex, 3) == r + ex.n_new_l
                @test left_isometry_defect(ex.U_ex) < 1e-11
            end
        end
    end

    @testset "opens a charge sector the K/L complement cannot" begin
        # THE CLAIM, at integrator level. At the domain wall the frame is missing a
        # reachable sector; the K/L complement stays inside U0's sectors (charge
        # conservation with the opposite frame frozen), which is exactly why `augment.jl`
        # needs `random_sector_seed`. CBE reaches it through H, with no fill anywhere.
        set_symmetry!(:U1)
        L = 6
        wall = L ÷ 2
        h = xxz_chain(L)
        psi = domain_wall_state(L)
        canonical!(psi, wall)
        f = bond_frame(psi, wall)

        have = Set(q for (q, _) in f.U0.spaces[3])
        reach = Set(q for (q, d) in reachable_sectors(f.U0, 1, 2) if d > 0)
        missing_q = setdiff(reach, have)
        @test !isempty(missing_q)

        gate = bond_gate(h, f.site_l, f.site_r)
        kl_new = kl_complement_charges(f, gate, f.site_l, f.site_r)
        @test isempty(intersect(kl_new, missing_q))       # K/L cannot get there

        ex = cbe_expand(f, h, wall,
                        left_env_stack(psi, h; upto = wall - 1),
                        right_env_stack(psi, h; downto = wall + 2);
                        rng = Random.MersenneTwister(0xC0FFEE))
        got = Set(q for (q, _) in ex.U_ex.spaces[3])
        @test !isempty(intersect(got, missing_q))         # CBE does
        @test left_isometry_defect(ex.U_ex) < 1e-11
    end

    @testset "with the cap lifted, dex = full complement spans the local space" begin
        # The 2-site limit: expanding by the whole complement makes the frame a basis of
        # link (x) site, so the bond update is the exact two-site update.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L; n_steps = 2)
        for i in 1:(L - 1)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            dfull_l = leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
            dfull_r = leg_dim(f.V0, 2) * leg_dim(f.V0, 3)
            room = min(dfull_l, dfull_r) - f.old_rank
            room > 0 || continue
            ex = cbe_expand(f, h, i,
                            left_env_stack(psi, h; upto = i - 1),
                            right_env_stack(psi, h; downto = i + 2);
                            dex = room, sulz_cap = false,
                            rng = Random.MersenneTwister(0xBEEF))
            @test leg_dim(ex.U_ex, 3) == f.old_rank + ex.n_new_l
            @test ex.n_new_l <= room && ex.n_new_r <= room
            @test left_isometry_defect(ex.U_ex) < 1e-11
            @test right_isometry_defect(ex.V_ex) < 1e-11
        end
    end

    @testset "preselect-only mode" begin
        # The reference's '-p': skip the weight ranking and keep the sketch's own basis.
        # Isolates how much the final selection is worth; must still give a valid frame.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        for i in 1:(L - 1)
            canonical!(psi, i)
            f = bond_frame(psi, i)
            ex = cbe_expand(f, h, i,
                            left_env_stack(psi, h; upto = i - 1),
                            right_env_stack(psi, h; downto = i + 2);
                            dex = 2, preselect_only = true,
                            rng = Random.MersenneTwister(11))
            @test ex.n_new_l <= 2 && ex.n_new_r <= 2
            @test ex.err_fnl == 0.0                       # no final selection was made
            @test left_isometry_defect(ex.U_ex) < 1e-11
            @test right_isometry_defect(ex.V_ex) < 1e-11
        end
    end

    @testset "a saturated bond expands by nothing" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L; n_steps = 4)
        canonical!(psi, 1)
        f = bond_frame(psi, 1)
        if f.old_rank == leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
            ex = cbe_expand(f, h, 1,
                            left_env_stack(psi, h; upto = 0),
                            right_env_stack(psi, h; downto = 3))
            @test ex.n_new_l == 0
            @test leg_dim(ex.U_ex, 3) == f.old_rank
        end
    end

    @testset "no symmetry" begin
        set_symmetry!(:none)
        try
            L = 6
            h = xxz_chain(L)
            psi = evolved(L; n_steps = 2)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                th = frame_theta(f)
                ex = cbe_expand(f, h, i,
                                left_env_stack(psi, h; upto = i - 1),
                                right_env_stack(psi, h; downto = i + 2);
                                rng = Random.MersenneTwister(0xF00D))
                @test left_isometry_defect(ex.U_ex) < 1e-11
                @test right_isometry_defect(ex.V_ex) < 1e-11
                rebuilt, _ = project_and_rebuild(ex, th)
                @test norm(to_concrete(rebuilt - th)) < 1e-11 * max(1.0, norm(th))
            end
        finally
            set_symmetry!(:U1)
        end
    end
end
