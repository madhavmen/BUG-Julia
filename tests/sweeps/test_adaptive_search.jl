# tests/sweeps/test_adaptive_search.jl -- the adaptive RSVD search WIDTH and the folded probe.
#
#   thin search   Test B, not the numerical rank, sets the sketch width (`_certified_range(...;
#                 thin = true)`, the default through `thin_search`). It must still certify a
#                 blind sector (the UT06 fixture), draw only what Test B asks for, and give the
#                 same physics as the old full-rank widening and as exact CBE, with fewer probes.
#   fold_omega    the probe folded into the environment contractions, so `H*Theta` is never
#                 formed -- in the half-sweeps AND in the order-p closure, whose per-bond leak is
#                 then Test B's holdout estimate. Same states as the shared build to tolerance.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))

@testset "adaptive search width and folded probe" begin

@testset "thin _certified_range: Test B decides the width" begin
    f, skl, _ = _xx_bond()
    U0, V0 = f.U0, f.V0
    proj = X -> perp_component(U0, X)
    Fr = full_local_basis(V0, :right)
    A = to_concrete(proj(skl(Fr)))
    nA = norm(A)
    exact(Q) = Q === nothing ? nA : norm(to_concrete(proj(to_concrete(perp_component(Q, A)))))
    select = Om -> RC._preselect_left(U0, skl(Om), 0.0)       # the adaptive stol_pre
    wcap = leg_dim(Fr, 1)
    ncap = leg_dim(U0, 1) * leg_dim(U0, 2) - leg_dim(U0, 3)
    rng = Random.MersenneTwister(0x7A1)
    draw(w, b) = sector_graded_sketch(V0, :right, w; comp_ratio = nothing, rng = rng, boost = b)

    # ⚠ `<=`, not `<`, ON THIS BOND: the fixture's sketched range is numerically exhausted at ~5
    # columns, so the wide search also stops after its first pass here (job 17283342: 5 vs 5).
    # The strict "fewer probes" claim is asserted end to end below, on full steps.
    @testset "a loose allowance stops at the first pass; thin never draws more" begin
        tol = 0.5 * nA
        alloc = RC.BasisAllocation(1.0, tol, tol, tol, 1.0)
        lt, lw = AdaptiveLedger(), AdaptiveLedger()
        RC._certified_range(draw, 1, select, 3, proj, skl, :left, 2, wcap, ncap, alloc, lt;
                            thin = true)
        RC._certified_range(draw, 1, select, 3, proj, skl, :left, 2, wcap, ncap, alloc, lw;
                            thin = false)
        @test lt.tolerance_met
        @test lt.n_probes <= lw.n_probes                        # the width followed tol
    end

    @testset "a tight allowance is still met, by widening" begin
        tol = 1e-6 * nA
        alloc = RC.BasisAllocation(1.0, tol, tol, tol, 1.0)
        lg = AdaptiveLedger()
        Q, _, _ = RC._certified_range(draw, 1, select, 3, proj, skl, :left, 2, wcap, ncap,
                                      alloc, lg; thin = true)
        @test lg.tolerance_met
        @test exact(Q) < 1e-3 * nA
    end

    @testset "a blind sector is still found and certified (UT06 under thin)" begin
        tol = 1e-6 * nA
        alloc = RC.BasisAllocation(1.0, tol, tol, tol, 1.0)
        _, perF, colsF = RC._sample_residual(nothing, proj, skl, w -> Fr, 1, 1, :left)
        qs = argmax(q -> perF[q] * colsF[q], collect(keys(perF)))
        honest = w -> sector_graded_sketch(V0, :right, w; comp_ratio = nothing, rng = rng)
        blind(w, b) = (b !== nothing && haskey(b, qs)) ? draw(w, b) : _drop(draw(w, b), 1, qs)
        lg = AdaptiveLedger()
        Q, _, _ = RC._certified_range(blind, 1, select, 3, proj, skl, :left, 2, wcap, ncap,
                                      alloc, lg; holdout = honest, thin = true)
        @test lg.n_targeted >= 1
        @test lg.tolerance_met
        @test exact(Q) < 1e-3 * nA
    end
end

@testset "cbe_bug_step!: thin == wide == exact CBE to tolerance, with fewer probes" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    tau = ComplexF64(-0.05im)
    akw = (tol_basis = 1e-7, tol_trunc = 1e-9, tol_expv = 1e-10, maxdim = 64)
    fid(a, b) = abs(overlap(a, b)) / (norm(a) * norm(b))
    for (name, kw) in (("ordinary", NamedTuple()), ("order 2", (order = 2,)))
        @testset "$name" begin
            ex, th, wd = neel_state(L), neel_state(L), neel_state(L)
            pt, pw = 0, 0
            for _ in 1:4
                cbe_bug_step!(ex, mpo, tau; exact = true, kw..., akw...)
                it = cbe_bug_step!(th, mpo, tau; thin_search = true, kw..., akw...)
                iw = cbe_bug_step!(wd, mpo, tau; thin_search = false, kw..., akw...)
                pt += it.ledger.n_probes
                pw += iw.ledger.n_probes
            end
            @test 1 - fid(th, ex) < 1e-8
            @test 1 - fid(wd, ex) < 1e-8
            @test pt < pw
        end
    end
end

@testset "fold_omega: same states as the shared H*Theta, half-sweeps and closure" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    tau = ComplexF64(-0.05im)
    akw = (tol_basis = 1e-7, tol_trunc = 1e-9, tol_expv = 1e-10, maxdim = 64)
    fid(a, b) = abs(overlap(a, b)) / (norm(a) * norm(b))
    for (name, kw) in (("ordinary", NamedTuple()), ("order 2", (order = 2,)),
                       ("order 4", (order = 4,)))
        @testset "$name" begin
            ex, fo = neel_state(L), neel_state(L)
            for _ in 1:4
                cbe_bug_step!(ex, mpo, tau; exact = true, kw..., akw...)
                info = cbe_bug_step!(fo, mpo, tau; fold_omega = true, kw..., akw...)
                @test info.ledger isa AdaptiveLedger
                # A folded closure never forms the two-site action.
                haskey(kw, :order) && @test info.ledger.n_h2_leak == 0
            end
            @test 1 - fid(fo, ex) < 1e-8
        end
    end
end

@testset "adaptive DMRG: thin search, same energy, fewer probes" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    run(thin) = (p = neel_state(L);
                 r = rsvd_cbe_dmrg1s!(p, mpo; n_sweeps = 12, maxdim = 32, tol_basis = 1e-5,
                                      tol_trunc = 1e-6, thin_search = thin);
                 (real(mpo_energy(copy(p), mpo)) / norm(p)^2,
                  sum(lg.n_probes for lg in r.ledgers)))
    et, nt = run(true)
    ew, nw = run(false)
    @test abs(et - ew) < 1e-6 * abs(ew)
    @test nt < nw
end

end # adaptive search width and folded probe
