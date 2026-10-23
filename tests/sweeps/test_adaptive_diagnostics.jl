# tests/sweeps/test_adaptive_diagnostics.jl -- the three requirements of the adaptive CBE-Krylov
# plan that a line-by-line re-read found outside UT01-UT15:
#
#   §8  "Track by bond and timestep": the ledger's per-bond log, and that the step-level numbers
#       ARE the aggregate of it (one bookkeeping path, not two that can drift apart);
#   §7  "Use both Float64 and complex state tensors where applicable": a real start state runs
#       through both schemes, in real and imaginary time, and agrees with its complex twin;
#   §1  the optional migration hint for the retired knobs (audit row 1.5).

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))

_sz(p) = real.(magnetisation(copy(p))) ./ norm(copy(p))^2

@testset "adaptive diagnostics, Float64 states, migration hint" begin

@testset "§8 per-bond diagnostics: one entry per half-sweep bond visit" begin
    L = 8
    mpo, psi = _xx_warm(L, 2)
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.05im); tol_basis = 1e-6,
                         rng = Random.MersenneTwister(1))
    lg, c = info.ledger, info.root
    blog = lg.bond_log
    # The left sweep visits 1..c, the right sweep L-1..c: L visits, the root twice.
    @test length(blog) == L
    @test sort([e.bond for e in blog if e.side === :left]) == collect(1:c)
    @test sort([e.bond for e in blog if e.side === :right]) == collect(c:(L - 1))
    @test sum(e.n_probes for e in blog) > 0                 # non-vacuous: the sketch ran

    # THE STEP'S NUMBERS ARE THE AGGREGATE OF THE LOGGED BONDS, by construction.
    for f in (:n_probes, :n_holdout, :n_selected, :n_targeted, :n_krylov, :n_h1)
        @test getfield(lg, f) == sum(getfield(e, f) for e in blog)
    end
    @test lg.n_c_pass == count(e -> e.c_pass, blog)
    @test lg.n_d_after_c == count(e -> e.d_after_c, blog)
    @test lg.max_sample == maximum(e.sample for e in blog)
    @test lg.max_tail == maximum(e.tail for e in blog)
    @test lg.max_krylov_c == maximum(e.krylov_c for e in blog)
    @test lg.max_krylov_d == maximum(e.krylov_d for e in blog)
    @test lg.workcap_reached == any(e -> e.workcap, blog)

    # Each entry's width is the frame the step actually built at that bond.
    for e in blog
        want = e.side === :left ? info.expanded[e.bond] :
               e.bond == c ? info.root_right : info.expanded[e.bond]
        @test e.width == want
    end

    @testset "per-bond ranks, sectors and discarded norm (simulation_plan §5)" begin
        # A loose `tol_trunc` so the compression really cuts: the per-bond discarded norms must
        # then rebuild the step's cumulative estimate exactly (Pythagoras within each subtree,
        # triangle across -- `truncate_recursive!`'s rule).
        mpo4, psi4 = _xx_warm(L, 2)
        bd0 = bond_dims(psi4)
        info4 = cbe_bug_step!(psi4, mpo4, ComplexF64(-0.05im); tol_basis = 1e-6,
                              tol_trunc = 1e-3, rng = Random.MersenneTwister(1))
        lg4, c4 = info4.ledger, info4.root
        bd1 = bond_dims(psi4)
        @test all(e -> e.r_old == bd0[e.bond] && e.r_final == bd1[e.bond], lg4.bond_log)
        d = Dict(e.bond => e.discarded for e in lg4.bond_log)
        @test sort(collect(keys(d))) == collect(1:(L - 1))
        @test lg4.discarded > 0                                     # non-vacuous: it cut
        left  = sqrt(sum(d[b]^2 for b in 1:c4))                    # root c4+1: bonds c4..1
        right = sqrt(sum((d[b]^2 for b in (c4 + 1):(L - 1)); init = 0.0))
        @test lg4.discarded ≈ left + right rtol = 1e-10
        # Reachable sectors: every bond that sampled has at least one, and its probes are > 0.
        for e in lg4.bond_log
            e.n_probes > 0 || continue
            @test e.n_sectors >= 1
            @test all(v -> sum(values(v); init = 0) > 0, values(e.probes_per_sector))
        end
    end

    @testset "a failed bond is named; only step-level checks fail without one" begin
        # maxdim = 2 binds the work ceiling (UT11). Whatever fails, every per-bond failure is in
        # the step's verdict, and every step failure no bond explains is a step-level check.
        mpo2, psi2 = _xx_warm(L, 2)
        lg2 = cbe_bug_step!(psi2, mpo2, ComplexF64(-0.05im); tol_basis = 1e-8, maxdim = 2,
                            rng = Random.MersenneTwister(1)).ledger
        @test budget_status(lg2) === :budget_unmet
        per_bond = union(Symbol[], (e.unmet for e in lg2.bond_log)...)
        @test per_bond ⊆ lg2.unmet
        @test setdiff(lg2.unmet, per_bond) ⊆ [:trunc, :projection, :expv]
        @test all(e -> e.met == isempty(e.unmet), lg2.bond_log)
    end

    @testset "midpoint: both stages logged and tagged" begin
        mpo3, psi3 = _xx_warm(L, 2)
        lg3 = cbe_bug_midpoint_step!(psi3, mpo3, ComplexF64(-0.05im); tol_basis = 1e-6,
                                     rng = Random.MersenneTwister(1)).ledger
        @test count(e -> e.stage == 1, lg3.bond_log) == L
        @test count(e -> e.stage == 2, lg3.bond_log) == L
        @test lg3.n_probes == sum(e.n_probes for e in lg3.bond_log)
        @test lg3.n_krylov == sum(e.n_krylov for e in lg3.bond_log)
    end
end

@testset "§7 Float64 and complex start states" begin
    set_symmetry!(:U1)
    L = 8
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    p64, pc = neel_state(L; eltype = Float64), neel_state(L)
    @test eltype(p64[1]) == Float64          # precondition: really a real state
    @test eltype(pc[1]) == ComplexF64
    @test _sz(p64) ≈ _sz(pc)

    # `exact = true` (no random sketch), so the two runs make the same decisions up to roundoff.
    for (scheme, step!) in (("ordinary", cbe_bug_step!), ("midpoint", cbe_bug_midpoint_step!)),
        (time, tau) in (("real", ComplexF64(-0.05im)), ("imaginary", ComplexF64(-0.05)))
        @testset "$scheme, $time time" begin
            a, b = copy(p64), copy(pc)
            for _ in 1:4
                sa = budget_status(step!(a, mpo, tau; tol_basis = 1e-6, exact = true).ledger)
                sb = budget_status(step!(b, mpo, tau; tol_basis = 1e-6, exact = true).ledger)
                @test sa === :met
                @test sb === :met
            end
            @test maximum(abs.(_sz(a) .- _sz(b))) < 1e-6
            @test maximum(bond_dims(a)) > 1                  # it evolved
        end
    end

    @testset "RSVD path from a real state, graded against free fermions" begin
        a = neel_state(L; eltype = Float64)
        rng = Random.MersenneTwister(3)
        for _ in 1:10
            cbe_bug_step!(a, mpo, ComplexF64(-0.05im); tol_basis = 1e-6, rng = rng)
        end
        @test maximum(abs.(_sz(a) .- xx_free_fermion_sz(L, 0.5; J = 1.0, occupied = 1:2:L))) < 1e-4
    end
end

@testset "§1 migration hint for the retired knobs (audit row 1.5)" begin
    @test RC._legacy_hint(()) === nothing
    msg = @test_logs (:warn, r"fixed-knob legacy path") RC._legacy_hint((:dex, :trunc_thresh))
    @test occursin("dex, trunc_thresh", msg)
    @test occursin("tol_basis", msg)

    # Through the public entry: the BUG step has no legacy path (a retired knob is no keyword),
    # and an adaptive step gives no warning.
    set_symmetry!(:U1)
    mpo = xxz_mpo(6; J = 1.0, delta = 0.0)
    @test_throws MethodError cbe_bug_step!(neel_state(6), mpo, ComplexF64(-0.05im);
                                           trunc_thresh = 1e-10)
    @test_logs min_level=Base.CoreLogging.Warn cbe_bug_step!(
        neel_state(6), mpo, ComplexF64(-0.05im); tol_basis = 1e-5)
end

end # adaptive diagnostics
