using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/free_fermion.jl")

# CBE inside the VALIDATED gate-based Trotter sweep.
#
# WHY THIS FILE IS THE DECIDING ONE. In `cbe_lubich.jl` a rank stall has four possible homes:
# the Lubich basis sweep, the channel environments, the CBE selection, or the placement of
# the single centre Galerkin step. Here there are no environments (the Hamiltonian at a bond
# IS the gate), no basis sweep (`canonical!` before every bond, as the validated integrator
# does), and one Galerkin per bond (the step the validated integrator already takes). The
# only thing left that can be wrong is the SELECTION -- which is shared code, not a
# reimplementation. So:
#
#   rank grows here  =>  the selection is sound, and the fault in cbe_lubich.jl is the sweep.
#   rank stalls here =>  the selection is the fault, isolated with nothing else to blame.
#
# The order of the testsets is that argument: pin the sketch, pin the expansion, pin
# exactness, and only then read the rank.
#
# U(1) THROUGHOUT, ASSERTED RATHER THAN INHERITED. Every state here is sector-resolved and
# every gate is the charge-conserving XXZ/XX term. The sector machinery is the whole reason
# CBE is being tried: opening an empty-but-reachable charge sector is what the K/L
# augmentation provably cannot do, and what `random_sector_seed` exists to paper over. A
# silent fall back to `:none` would make every sector claim below a tautology, so the mode is
# checked first and the sector-opening claim is carried in its own testset.

const TL_TAG(f) = f.site_l.itags
const TR_TAG(f) = f.site_r.itags

"The rank-4 reference: form `HΘ` explicitly and sketch it AFTERWARDS. This is exactly the
object the production path refuses to build, which is why it belongs only in a test."
# The sketch probes the PROJECTOR: `P_perp (H Theta)`, not `H Theta`. The reference
# therefore forms the rank-4 object, projects it EXPLICITLY with `perp_component`
# (`BondUpdateBUG`'s, independent of anything under test), and only then folds `Om` in.
function reference_sketch_left(f, gate, Om)
    HT = apply_gate(gate, frame_theta(f), TL_TAG(f), TR_TAG(f))
    P  = to_concrete(perp_component(f.U0, HT))
    return to_concrete(contract(P, (3, 4), Om', (2, 3)))
end

function reference_sketch_right(f, gate, Om)
    HT = apply_gate(gate, frame_theta(f), TL_TAG(f), TR_TAG(f))
    P  = to_concrete(perp_component_right(f.V0, HT))
    return to_concrete(permutedims(contract(P, (1, 2), Om', (1, 2)), (3, 1, 2)))
end

"`‖HΘ − U U† HΘ‖ / ‖HΘ‖` -- how much of the gate's action on the block the left frame `U`
cannot represent. Zero means the frame spans the whole left image of `HΘ`, which is the
strongest thing an expansion at this bond can achieve."
function left_residual(U, f, gate)
    HT = apply_gate(gate, frame_theta(f), TL_TAG(f), TR_TAG(f))
    R = to_concrete(perp_component(U, HT))
    return norm(R) / norm(HT)
end

function right_residual(V, f, gate)
    HT = apply_gate(gate, frame_theta(f), TL_TAG(f), TR_TAG(f))
    ov = contract(HT, (3, 4), V', (2, 3))
    R = to_concrete(HT - to_concrete(contract(ov, (3,), V, (1,))))
    return norm(R) / norm(HT)
end

"A state with real structure on every bond: the validated integrator, a few steps in."
function warm_state(L::Int; n_steps = 4, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

"Charge sectors carried by a frame's bond leg."
frame_sectors(U, leg) = Set(q for (q, _) in U.spaces[leg])

"Snapshot bond `i` of `psi` together with its gate."
function frame_and_gate(psi, i; delta = 1.0)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    return f, heisenberg_bond_gate(f.site_l, f.site_r; delta = delta)
end

@testset "cbe_bond_update" begin

    # ── 0. the symmetry these tests claim to be exercising ────────────────
    @testset "the model under test really is U(1)" begin
        @test symmetry_mode() === :U1
        # Under :U1 the raising/lowering operators are RANK-3 -- an op-leg carries the ±2
        # charge. Under :none they are plain matrices and every sector assertion below would
        # be vacuous.
        @test length(local_space().Sp.inds) == 3
        psi = domain_wall_state(8)
        @test all(length(psi[i].spaces[3]) >= 1 for i in 1:7)
        # And the gate conserves the charge: the two-site magnetisation commutes with it.
        f, gate = frame_and_gate(psi, 4)
        tl, tr = TL_TAG(f), TR_TAG(f)
        th = frame_theta(f)
        mg = magnetisation_gate(f.site_l, f.site_r)
        a = apply_gate(mg, apply_gate(gate, th, tl, tr), tl, tr)
        b = apply_gate(gate, apply_gate(mg, th, tl, tr), tl, tr)
        @test norm(to_concrete(a - b)) < 1e-12
    end

    # ── 1. the sketch is the gate action, never the rank-4 object ──────────────
    #
    # The whole memory claim of the method rests on `HΘ` never being formed, so the sketched
    # contraction is a DIFFERENT network from the obvious one and has to be pinned against
    # it. `apply_gate` is itself pinned to the analytic Heisenberg block element-for-element
    # (tests/BondUpdateBUG/test_gates.jl), so this chains onto a validated reference rather
    # than onto a sibling of the code under test.
    @testset "sketch_gate_* == P_perp(gate Θ) Ω†" begin
        psi = warm_state(8)
        for i in (1, 3, 4, 6)
            f, gate = frame_and_gate(psi, i)

            # (a) the EXACT probe: Ω = the frame itself recovers the exact factor.
            @test norm(to_concrete(sketch_bond_left(f, gate, f.V0) -
                                   reference_sketch_left(f, gate, f.V0))) < 1e-12
            @test norm(to_concrete(sketch_bond_right(f, gate, f.U0) -
                                   reference_sketch_right(f, gate, f.U0))) < 1e-12

            # (b) an ARBITRARY probe, the one the production path actually passes. A sketch
            #     with more columns than the frame has rank is the case that would expose a
            #     wrong contraction order or a mis-set arrow.
            OmR = sector_graded_sketch(f.V0, :right, 5; rng = Random.MersenneTwister(11))
            OmL = sector_graded_sketch(f.U0, :left,  5; rng = Random.MersenneTwister(12))
            OmR === nothing || @test norm(to_concrete(
                sketch_bond_left(f, gate, OmR) - reference_sketch_left(f, gate, OmR))) < 1e-12
            OmL === nothing || @test norm(to_concrete(
                sketch_bond_right(f, gate, OmL) - reference_sketch_right(f, gate, OmL))) < 1e-12
        end
    end

    @testset "a gate tagged for the wrong bond is refused" begin
        psi = warm_state(6)
        f, _ = frame_and_gate(psi, 2)
        wrong = heisenberg_bond_gate(psi[4].inds[2], psi[5].inds[2])
        @test_throws ArgumentError sketch_bond_left(f, wrong, f.V0)
        @test_throws ArgumentError sketch_bond_right(f, wrong, f.U0)
    end

    # ── 2. the expansion is an isometry that CONTAINS the old frame ────────────
    #
    # `oplus` concatenates without orthogonalising, so [U0 | new] is a frame only if the new
    # block is exactly orthogonal to U0 -- the omission that duplicated a column in
    # `augment.jl` and grew the norm 128x over five steps. Containment is what makes
    # `Ŝ₀ = U_ex† Θ₀ V_ex†` lossless, hence `tau = 0` the identity on the state.
    @testset "cbe_expand_bond: isometry, contains the frame, lossless" begin
        psi = warm_state(8)
        for i in (1, 2, 4, 5, 7)
            f, gate = frame_and_gate(psi, i)
            ex = cbe_expand_bond(f, gate; rng = Random.MersenneTwister(0xC0FFEE))

            @test left_isometry_defect(ex.U_ex) < 1e-11
            @test right_isometry_defect(ex.V_ex) < 1e-11
            @test leg_dim(ex.U_ex, 3) >= f.old_rank
            @test leg_dim(ex.V_ex, 1) >= f.old_rank

            th = frame_theta(f)
            S = to_concrete(contract(contract(ex.U_ex', (1, 2), th, (1, 2)),
                                     (2, 3), ex.V_ex', (2, 3)))
            back = to_concrete((ex.U_ex * S) * ex.V_ex)
            @test norm(to_concrete(back - th)) / norm(th) < 1e-11
        end
    end

    # ── 2b. the sector claim: CBE opens an empty-but-reachable charge sector ───
    #
    # THE REASON THIS METHOD EXISTS. On a product state every bond carries exactly ONE charge
    # sector, and with the opposite frame held fixed the K/L complement can never open
    # another (`sectors.jl` header) -- which is why `augment.jl` needs `random_sector_seed`.
    # The sector-graded sketch is supposed to reach the empty sector THROUGH the gate, so what
    # comes back is the direction the dynamics wants rather than a random column the S-step
    # SVD then discards. If this fails, nothing downstream matters.
    # Asserted against the PHYSICS, not against an assumption about which bonds are
    # interesting. On the L=8 domain wall only bond 4 straddles the wall: at bonds 3 and 5 both
    # sites are aligned, the XY hop annihilates the block and `HΘ ∝ Θ`, so there is no new
    # direction to find and refusing to expand is CORRECT. An earlier version of this test
    # demanded growth at 3 and 5 and failed on its own premise. `P⊥HΘ` decides which case a
    # bond is in, so the test asks the expansion for exactly what the gate makes available.
    @testset "the expansion opens a new charge sector where the gate can hop" begin
        psi = domain_wall_state(8)
        for i in 1:7
            f, gate = frame_and_gate(psi, i)
            @test f.old_rank == 1
            before = frame_sectors(f.U0, 3)
            @test length(before) == 1
            HT = apply_gate(gate, frame_theta(f), TL_TAG(f), TR_TAG(f))
            reachable = norm(to_concrete(perp_component(f.U0, HT))) / norm(HT)
            ex = cbe_expand_bond(f, gate; rng = Random.MersenneTwister(0xD00D))
            after = frame_sectors(ex.U_ex, 3)
            @info "bond $i product state: P⊥HΘ = $reachable, " *
                  "sectors $(length(before)) -> $(length(after)), " *
                  "rank 1 -> $(leg_dim(ex.U_ex, 3))"
            @test before ⊆ after
            if reachable > 1e-10
                # A new direction exists. Under U(1) it necessarily lives in a DIFFERENT charge
                # sector -- a hop moves one unit of Sz across the bond -- so this is the sector
                # opening the K/L complement provably cannot do and `random_sector_seed` exists
                # to fake.
                @test length(after) > length(before)
                @test leg_dim(ex.U_ex, 3) > 1
            else
                # Nothing to find: the expansion must not invent a direction the dynamics has
                # no weight in, which is precisely the random fill's failure mode.
                @test length(after) == length(before)
            end
        end
    end

    # ── 3. exactness at full rank -- the property that makes it an integrator ──
    #
    # With the frames spanning the whole local space the Galerkin generator IS the gate, so
    # the bond update must reproduce the exact two-site propagator. This is the single test
    # that a truncation bug, a missing projector or a wrong `tau` cannot survive, and it is
    # checked against `expv` on the block directly rather than against `kls_bond_update` --
    # an independent reference, not a sibling.
    @testset "one bond update at full rank IS the exact two-site propagator" begin
        psi = warm_state(6)
        tau = -im * 0.05
        for i in (2, 3)
            f, gate = frame_and_gate(psi, i)
            tl, tr = TL_TAG(f), TR_TAG(f)
            exact = expv(x -> apply_gate(gate, x, tl, tr), tau, frame_theta(f);
                         hermitian = true, maxiter = 60, tol = 1e-16)
            r = cbe_bond_update(f, gate, tau;
                                     maxdim = 400, trunc_thresh = 1e-16, dex = 64,
                                     rng = Random.MersenneTwister(7))
            got = to_concrete(r.left_core * r.right_core)
            @test norm(to_concrete(got - exact)) / norm(exact) < 1e-10
        end
    end

    @testset "tau = 0 is the identity on the state" begin
        psi = warm_state(8)
        f, gate = frame_and_gate(psi, 4)
        th = frame_theta(f)
        r = cbe_bond_update(f, gate, 0.0 + 0.0im; rng = Random.MersenneTwister(3))
        got = to_concrete(r.left_core * r.right_core)
        @test norm(to_concrete(got - th)) / norm(th) < 1e-11
    end

    # ── 4. does the SELECTION admit what the dynamics needs? ───────────────────
    #
    # THE DISCRIMINATING MEASUREMENT. `left_residual` is the part of the gate's action on the
    # block that the expanded frame cannot represent. Given the full budget and the whole
    # complement, CBE should drive it to zero -- that is the exact criterion-2 enrichment
    # (`kls_step.jl::_criterion2_left`) computed by sketching instead of by an exact SVD.
    # If it does NOT, the selection is refusing directions it should keep, and that is the
    # rank stall's cause rather than anything in the sweep. Reported as numbers, not just a
    # pass/fail, because the SIZE of the residual is the diagnostic.
    #
    # `comp_ratio` IS LEFT AT ITS DEFAULT, and an earlier version of this test was wrong to
    # force 1.0. The complement/isometry split governs the PROBE, not what is admitted -- the
    # admitted block is always outside the frame, since `_preselect_*` projects the frame out.
    # Probing with the complement alone therefore does not "keep only the discarded space", it
    # BLINDS the sketch to the 1-site component ("do not project into complement space, 1-site
    # components are also important for bond update", `RSVDpreBE0SiQS.m:339`) and caps the
    # probe's width at the opposite frame's complement dimension, which near an edge is 0 or 1.
    @testset "with the full budget the expansion spans the gate's whole image" begin
        psi = warm_state(8)
        for i in (1, 3, 4, 6)
            f, gate = frame_and_gate(psi, i)
            before_l = left_residual(f.U0, f, gate)
            before_r = right_residual(f.V0, f, gate)
            ex = cbe_expand_bond(f, gate; dex = 64, rng = Random.MersenneTwister(0xBEEF))
            after_l = left_residual(ex.U_ex, f, gate)
            after_r = right_residual(ex.V_ex, f, gate)
            @info "bond $i: left residual $(before_l) -> $(after_l), " *
                  "right $(before_r) -> $(after_r), " *
                  "rank $(f.old_rank) -> ($(leg_dim(ex.U_ex,3)), $(leg_dim(ex.V_ex,1)))"
            @test after_l <= before_l + 1e-12
            @test after_r <= before_r + 1e-12
            @test after_l < 1e-9
            @test after_r < 1e-9
        end
    end

    # ── 5. THE RANK. What the whole exercise is about ──────────────────────────
    #
    # A product state has every bond at dimension 1, so there is nowhere for rank to come
    # from except the expansion. `> 1` after one step is the weakest possible statement and
    # is deliberately separate from the growth test: it distinguishes "expands, slowly" from
    # "does not expand at all", which is the distinction the MPO path never got past.
    @testset "the rank leaves 1 on the first step" begin
        psi = domain_wall_state(8)
        @test maximum(bond_dims(psi)) == 1
        info = cbe_bond_update_bug!(psi, bond_gates(psi);
                             opts = CBEBugOptions(dt = 0.05, n_steps = 1, maxdim = 32))
        @info "after one gate-CBE step: bond_dims = $(info.bond_dims[end])"
        @test maximum(info.bond_dims[end]) > 1
    end

    @testset "the rank keeps growing, and tracks the validated integrator" begin
        gates_of(p) = bond_gates(p)
        L, dt, n = 8, 0.05, 12

        pc = domain_wall_state(L)
        ic = cbe_bond_update_bug!(pc, gates_of(pc);
                           opts = CBEBugOptions(dt = dt, n_steps = n, maxdim = 32,
                                                trunc_thresh = 1e-12))
        pk = domain_wall_state(L)
        ik = bond_update_bug!(pk, gates_of(pk);
                              opts = BondUpdateOptions(dt = dt, n_steps = n, maxdim = 32,
                                                       trunc_thresh = 1e-12))
        @info "gate-CBE  max bond per step: $(ic.max_bond_dims)"
        @info "validated max bond per step: $(ik.max_bond_dims)"
        @info "gate-CBE  final profile: $(ic.bond_dims[end])"
        @info "validated final profile: $(ik.bond_dims[end])"

        # Monotone in the mean: rank must not be frozen at its first-step value.
        @test ic.max_bond_dims[end] > ic.max_bond_dims[1]
        # And it must reach the same order as the validated integrator's, which is the
        # honest bar -- CBE is meant to need FEWER directions for the same accuracy, so
        # falling far short of the validated rank means it is under-expanding, not being
        # efficient. Half is generous and still catches a stall.
        @test ic.max_bond_dims[end] >= ik.max_bond_dims[end] ÷ 2
    end

    # ── 6. the symmetry, and the physics ──────────────────────────────────────
    @testset "U(1) charge is conserved through the run" begin
        # NOT the half-filled domain wall: its total Sz is 0, so a broken charge sector
        # would have to conspire to move it and the test would pass on a zero target. Five
        # up against three down gives Sz = +1, a number that can actually drift.
        psi = product_state([:up, :up, :up, :up, :up, :down, :down, :down])
        m0 = sum(magnetisation(copy(psi)))
        @test abs(m0 - 1.0) < 1e-12
        cbe_bond_update_bug!(psi, bond_gates(psi);
                      opts = CBEBugOptions(dt = 0.05, n_steps = 8, maxdim = 32))
        @test abs(sum(magnetisation(copy(psi))) - m0) < 1e-10
    end

    # The XX domain wall against the free-fermion solution, ALONGSIDE the validated
    # integrator at identical settings. Both are the same Strang composition on the same
    # gates with the same truncation, so the only difference is the basis update -- which
    # makes the two errors side by side, not the absolute one, the number worth reading.
    @testset "XX domain wall follows the free-fermion profile" begin
        L, dt, n = 10, 0.02, 25
        xxg(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(L - 1)]
        prof(p) = (q = copy(p); magnetisation(q) ./ norm(q)^2)

        pc = domain_wall_state(L)
        cbe_bond_update_bug!(pc, xxg(pc);
                      opts = CBEBugOptions(dt = dt, n_steps = n, maxdim = 64,
                                           trunc_thresh = 1e-14))
        pk = domain_wall_state(L)
        bond_update_bug!(pk, xxg(pk);
                         opts = BondUpdateOptions(dt = dt, n_steps = n, maxdim = 64,
                                                  trunc_thresh = 1e-14))
        want = xx_free_fermion_sz(L, dt * n)
        err_c = maximum(abs.(prof(pc) .- want))
        err_k = maximum(abs.(prof(pk) .- want))
        @info "XX profile error at t = $(dt*n): gate-CBE $err_c, validated $err_k"
        @test err_c < 1e-2      # gross-breakage guard; the accuracy GAP is an open item
        @test err_k < 1e-6      # the validated integrator's own bar, for context
    end
end
