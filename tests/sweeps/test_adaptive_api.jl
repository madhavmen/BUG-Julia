# tests/sweeps/test_adaptive_api.jl -- UT01 and UT02 of the adaptive CBE-Krylov plan.
#
# STAGE 1 ONLY: the three-tolerance budget object, the allocation rule, and the migration off
# the retired tuning knobs. NO PHYSICS HAS CHANGED YET, and these tests are written so that
# they would still pass if it had -- they assert the API's contract, not a rank or an error.
#
# ⛔ THE POINT OF UT01 IS THAT THE OLD KNOBS CANNOT SILENTLY AFFECT AN ADAPTIVE RUN. That is a
# stronger claim than "the new keywords exist", and it is the one that actually protects a
# measurement: a benchmark that passes `dex = 8` alongside `tol_basis = 1e-5` and gets a run
# governed by one of them, with no way to tell which, is how a tolerance study ends up
# reporting the behaviour of a fixed rank.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))

# One outer testset so a failure cannot silently skip the later sets in a standalone run.
@testset "adaptive API (Stage 1)" begin

@testset "UT01: public API cleanup and knob migration" begin

    @testset "the PUBLIC entry refuses the mixture before touching the state" begin
        # `resolve_budget` refusing is necessary, not sufficient: what a caller meets is
        # `cbe_bug_step!`, and the refusal must happen there, before any evolution.
        set_symmetry!(:U1)
        mpo, psi = xxz_mpo(6; J = 1.0, delta = 0.0), neel_state(6)
        dims0 = bond_dims(psi)
        dt = ComplexF64(-0.05im)
        # The fixed-knob path is removed from the solver: a retired knob is no keyword.
        @test_throws MethodError cbe_bug_step!(psi, mpo, dt; tol_basis = 1e-5, dex = 4)
        @test_throws MethodError cbe_bug_step!(psi, mpo, dt; trunc_thresh = 1e-10)
        @test_throws MethodError cbe_bug_midpoint_step!(psi, mpo, dt; tol_expv = 1e-10,
                                                        stol_pre = 1e-10)
        @test bond_dims(psi) == dims0                    # refused, not half-run
    end

    @testset "the budget object validates its three tolerances" begin
        b = AdaptiveBudget(1e-5, 1e-7, 1e-10)
        @test b.basis == 1e-5
        @test b.trunc == 1e-7
        @test b.expv  == 1e-10

        # A non-positive basis or expv tolerance is a caller error, not a "keep everything"
        # instruction: both divide or bound a search, and 0 would make every residual fail.
        @test_throws ArgumentError AdaptiveBudget(0.0, 1e-7, 1e-10)
        @test_throws ArgumentError AdaptiveBudget(-1e-5, 1e-7, 1e-10)
        @test_throws ArgumentError AdaptiveBudget(1e-5, 1e-7, 0.0)
        # `trunc = 0` IS meaningful -- "discard nothing" -- so it must be accepted.
        @test AdaptiveBudget(1e-5, 0.0, 1e-10).trunc == 0.0
        @test_throws ArgumentError AdaptiveBudget(1e-5, -1e-7, 1e-10)
    end

    @testset "mixing the two APIs is refused, not resolved" begin
        # Each of these names one tolerance and one retired knob. There is no precedence
        # between them, so the only correct answer is to refuse.
        for knob in (:dex, :growth, :comp_ratio, :krylov_basis, :stol_pre, :stol_fnl,
                     :split_cutoff, :root_cutoff, :trunc_thresh)
            legacy = merge((dex = nothing, dover = nothing, comp_ratio = nothing,
                            growth = nothing, krylov_basis = nothing,
                            split_cutoff = nothing, split_maxdim = nothing,
                            root_cutoff = nothing, root_maxdim = nothing,
                            trunc_thresh = nothing, stol_pre = nothing, stol_fnl = nothing),
                           NamedTuple{(knob,)}((knob === :dex || knob === :krylov_basis ||
                                                knob === :split_maxdim ||
                                                knob === :root_maxdim ? 8 : 1e-6,)))
            @test_throws ArgumentError RSVDCBEBondUpdate.resolve_budget(1e-5, nothing,
                                                                        nothing, legacy)
        end
    end

    @testset "naming nothing selects the adaptive path with the documented defaults" begin
        none = (dex = nothing, dover = nothing, comp_ratio = nothing, growth = nothing,
                krylov_basis = nothing, split_cutoff = nothing, split_maxdim = nothing,
                root_cutoff = nothing, root_maxdim = nothing, trunc_thresh = nothing,
                stol_pre = nothing, stol_fnl = nothing)
        budget, adaptive, named = RSVDCBEBondUpdate.resolve_budget(nothing, nothing,
                                                                   nothing, none)
        @test adaptive
        @test isempty(named)
        @test budget.basis == RSVDCBEBondUpdate.TOL_DEFAULTS.basis
        @test budget.trunc == RSVDCBEBondUpdate.TOL_DEFAULTS.trunc
        @test budget.expv  == RSVDCBEBondUpdate.TOL_DEFAULTS.expv
    end

    @testset "naming a retired knob selects the legacy path and says which" begin
        legacy = (dex = 8, dover = nothing, comp_ratio = nothing, growth = nothing,
                  krylov_basis = nothing, split_cutoff = nothing, split_maxdim = nothing,
                  root_cutoff = nothing, root_maxdim = nothing, trunc_thresh = nothing,
                  stol_pre = nothing, stol_fnl = nothing)
        _, adaptive, named = RSVDCBEBondUpdate.resolve_budget(nothing, nothing, nothing,
                                                              legacy)
        @test !adaptive
        @test named == (:dex,)
    end

    @testset "dover alone carries no signal" begin
        # `nothing` is `dover`'s OWN historical default, so it cannot distinguish "not asked
        # for" from "asked for the default" the way the others can. A caller who wants it is
        # already naming `dex`; treating it as a mode selector would make `dover = nothing`
        # ambiguous in exactly the place the inference has to be certain.
        only_dover = (dex = nothing, dover = 4, comp_ratio = nothing, growth = nothing,
                      krylov_basis = nothing, split_cutoff = nothing, split_maxdim = nothing,
                      root_cutoff = nothing, root_maxdim = nothing, trunc_thresh = nothing,
                      stol_pre = nothing, stol_fnl = nothing)
        _, adaptive, named = RSVDCBEBondUpdate.resolve_budget(nothing, nothing, nothing,
                                                              only_dover)
        @test adaptive
        @test isempty(named)
    end

    @testset "legacy values resolve to their historical defaults" begin
        # The migration must not move a single recorded number: a call naming none of the
        # knobs has to produce what it produced before the sentinels were introduced.
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :dex) == 0
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :growth) == 2.0
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :krylov_basis) == 30
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :stol_pre) == 1e-10
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :stol_fnl) == 1e-13
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :split_cutoff) == 1e-14
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :root_cutoff) == 0.0
        @test RSVDCBEBondUpdate.legacy_or_default(nothing, :trunc_thresh) == 1e-12
        # An explicit value always wins over the default.
        @test RSVDCBEBondUpdate.legacy_or_default(16, :dex) == 16
        @test RSVDCBEBondUpdate.legacy_or_default(0.0, :root_cutoff) == 0.0
    end

    @testset "the allocation rule is the documented one" begin
        b = AdaptiveBudget(1e-4, 1e-7, 1e-10)
        a = basis_allocation(b, 19; tau = 0.05, nrm = 1.0)       # L = 20 -> 19 bonds
        @test a.per_bond ≈ 1e-4 / sqrt(19)
        @test a.sample ≈ a.per_bond / 2
        @test a.tail   ≈ a.per_bond / 2
        @test a.krylov ≈ a.per_bond / 2
        @test a.scale  ≈ 0.05

        # More bonds means a tighter per-bond allowance, and in quadrature rather than
        # linearly -- four times the bonds halves the allowance.
        a4 = basis_allocation(b, 76; tau = 0.05, nrm = 1.0)
        @test a4.per_bond ≈ a.per_bond / 2

        # The norm makes the comparison relative.
        @test basis_allocation(b, 19; tau = 0.05, nrm = 2.0).scale ≈ 0.025
    end

    @testset "tau = 0 is a basis question, not a degenerate one" begin
        # A zero timestep is how a caller asks "is this subspace closed under H" with no
        # evolution. Scaling by |tau| there would make every residual negligible and return
        # the unexpanded frame -- a silent refusal to expand, which is the failure mode.
        a = basis_allocation(AdaptiveBudget(1e-5, 1e-7, 1e-10), 11; tau = 0, nrm = 1.0)
        @test a.scale == 1.0
        @test indicator(a, 0.25) == 0.25
    end

    @testset "the indicator applies the scale exactly once" begin
        a = basis_allocation(AdaptiveBudget(1e-5, 1e-7, 1e-10), 9; tau = 0.1, nrm = 4.0)
        @test a.scale ≈ 0.025
        @test indicator(a, 8.0) ≈ 0.2
    end

    @testset "the work ceiling bounds growth without freezing a wide bond" begin
        # Twice the resource limit is the headroom; the `2r` floor is what keeps a bond
        # already at or above `maxdim` from being pinned at its current rank.
        @test work_ceiling(128, 10) == 256
        @test work_ceiling(128, 200) == 400        # r > maxdim: the floor governs
        @test work_ceiling(0, 0) == 1              # never zero, which would admit nothing
    end

    @testset "the ledger starts clean and merges conservatively" begin
        a, b = AdaptiveLedger(), AdaptiveLedger()
        @test a.tolerance_met
        @test !a.workcap_reached

        a.max_sample, a.n_probes = 1e-6, 12
        b.max_sample, b.n_probes = 3e-6, 7
        b.workcap_reached = true
        b.tolerance_met = false
        merge_ledger!(a, b)

        # Indicators take the WORST of the two, counts ADD, and both status flags are
        # pessimistic: one half-sweep failing its budget fails the step.
        @test a.max_sample == 3e-6
        @test a.n_probes == 19
        @test a.workcap_reached
        @test !a.tolerance_met
    end
end

@testset "UT02: numerical preselection without an absolute cutoff" begin
    # THE CLAIM (plan §1, UT02): near-dependent candidates are removed by a SCALE-AWARE
    # NUMERICAL rank test, and which small directions MATTER is left to Tests A/B. Tested on the
    # PRODUCTION `_preselect_left` with the adaptive path's `stol = 0.0` -- the numerical floor
    # `_rank_cut` -- on a real bond.
    #
    # ⛔ An earlier version tested a rank helper DEFINED IN THIS FILE (`rtol = sqrt(eps)`) and
    # said that was "what the adaptive path uses"; the path used `stol_pre = 1e-10`. It
    # certified nothing about the code (audit row 1.15, `ADAPTIVE_PLAN_AUDIT.md`).
    f, skl, _ = _xx_bond()
    U0 = f.U0
    Y = to_concrete(skl(full_local_basis(f.V0, :right)))    # P_perp G on the complete input
    prank(Y; stol = 0.0) = (Q = first(RC._preselect_left(U0, Y, stol));
                            Q === nothing ? 0 : leg_dim(Q, 3))
    r0 = prank(Y)
    @test r0 >= 2                                           # non-vacuous
    secs = _dims(Y, 3)
    @test length(secs) >= 2
    qbig = argmax(q -> norm(_only(Y, 3, q)), collect(keys(secs)))

    @testset "the verdict is identical across twelve decades of amplitude" begin
        for a in (1e-6, 1e-3, 1e3, 1e6)
            @test prank(to_concrete(a * Y)) == r0
        end
    end

    @testset "a redundant direction adds nothing, at any amplitude" begin
        for a in (1.0, 1e-6)
            @test prank(to_concrete(oplus([Y, to_concrete(a * Y)], (3,)))) == r0
        end
    end

    @testset "a weak but genuine direction survives" begin
        # The strongest sector scaled down by 1e-4: small, but independent of the rest.
        weak = to_concrete(oplus([_drop(Y, 3, qbig),
                                  to_concrete(1e-4 * _only(Y, 3, qbig))], (3,)))
        @test prank(weak) == r0
        # And the numerical floor never drops what the retired 1e-10 cut kept.
        weaker = to_concrete(oplus([_drop(Y, 3, qbig),
                                    to_concrete(1e-9 * _only(Y, 3, qbig))], (3,)))
        @test prank(weaker) >= prank(weaker; stol = 1e-10)
    end
end

end # adaptive API (Stage 1)
