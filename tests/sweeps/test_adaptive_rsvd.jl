# tests/sweeps/test_adaptive_rsvd.jl -- UT03..UT06 of the adaptive CBE-Krylov plan (§3, §7),
# plus the Stage 2 end-to-end checks on `cbe_bug_step!`.
#
#   UT03  sector coverage: every reachable complement sector is probed, nothing forbidden is
#   UT04  TEST B, the holdout sampling test, on the production sketch of a real bond
#   UT05  TEST A, the retained-rank tail test
#   UT06  the misleading tail: a sketch blind to a strong sector passes A and fails B, and
#         the controller widens THAT sector rather than accepting
#
# ⛔ TEST B IS TESTED THROUGH THE CODE THAT RUNS, not through a matrix restatement of it. An
# earlier version checked `||A*Om||/sqrt(n)` on plain matrices, which is true and was never
# the code's estimator: the production holdout is SECTOR-GRADED, and dividing its pooled sum
# by the total width under-reported a missed small sector by `sqrt(w/n_q)`. Only a test on a
# real block-sparse bond can see that.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))

# A Telum `kept_list` entry is `(sigma, degeneracy, _, _)`; only the first two are read.
_mklist(sigmas, degs = ones(Int, length(sigmas))) =
    [(sigmas[i], degs[i], nothing, nothing) for i in eachindex(sigmas)]

# ONE OUTER TESTSET: run standalone, a failing top-level testset throws at its end and every
# later one is silently skipped (job 17188740 lost UT05, UT06 and the end-to-end set that way).
@testset "adaptive RSVD (Stage 2)" begin

@testset "UT03: the probe reaches every admissible sector" begin

    @testset "every sector with room receives at least one column" begin
        # THE EMPTY-BUT-REACHABLE SECTOR IS THE WHOLE POINT. A proportional split rounds a
        # sector holding 1 direction out of 1000 down to zero columns, and that sector is
        # precisely the one the expansion exists to find -- it is empty in the frame BECAUSE
        # the state has not reached it yet. `_alloc_columns` ceils per sector for this reason.
        dims = Dict(:a => 1000, :b => 1, :c => 3)
        for budget in (1, 2, 4, 17, 64)
            cols = RC._alloc_columns(dims, budget)
            for (q, d) in dims
                @test cols[q] >= 1
                @test cols[q] <= d          # never more columns than the sector has room for
            end
        end
    end

    @testset "no sector is over-allocated beyond its own dimension" begin
        dims = Dict(:a => 2, :b => 3)
        cols = RC._alloc_columns(dims, 10_000)
        @test cols[:a] == 2
        @test cols[:b] == 3
    end

    @testset "a zero or negative budget allocates nothing, and does not throw" begin
        dims = Dict(:a => 4, :b => 2)
        for budget in (0, -1)
            cols = RC._alloc_columns(dims, budget)
            @test all(==(0), values(cols))
        end
        # An empty sector table is a real state (a frame that already spans everything).
        @test isempty(RC._alloc_columns(Dict{Symbol, Int}(), 8))
    end

    @testset "on a real frame: the empty sector is probed, a forbidden one never is" begin
        f, _, _ = _xx_bond(steps = 0)
        # (frame, side, fused legs, frame bond leg, probe width leg)
        for (frame, side, a, b, fl, wl) in ((f.V0, :right, 2, 3, 1, 1),
                                            (f.U0, :left, 1, 2, 3, 3))
            F = fusion_basis(frame, a, b; tag = "cbeG")
            reach = _dims(F, 3)
            room = Dict(q => d - sector_dim(frame, fl, q) for (q, d) in reach)
            # Not vacuous: the product state's rank-1 frame leaves a reachable sector EMPTY.
            @test any(sector_dim(frame, fl, q) == 0 for q in keys(reach))
            for npre in (1, 2, 5)
                Om = sector_graded_sketch(frame, side, npre; comp_ratio = 1.0,
                                          rng = Random.MersenneTwister(npre))
                got = _dims(Om, wl)
                @test issubset(keys(got), keys(reach))       # never a forbidden charge
                for (q, r) in room
                    r > 0 && @test get(got, q, 0) >= 1        # every sector with room
                end
            end
        end
    end

    @testset "a Test B boost lands in the named sector only, within its dimension" begin
        f, _, _ = _xx_bond()
        V0 = f.V0
        reach = _dims(fusion_basis(V0, 2, 3; tag = "cbeG"), 3)
        sk(boost) = sector_graded_sketch(V0, :right, 2; comp_ratio = 1.0, boost = boost,
                                         rng = Random.MersenneTwister(3))
        base = _dims(sk(nothing), 1)
        q = first(k for (k, d) in base if d < reach[k])     # a sector with room left
        one = _dims(sk(Dict{Any, Int}(q => 1)), 1)
        @test one[q] == base[q] + 1
        @test all(one[k] == d for (k, d) in base if k != q) # nothing else moved
        # Capped at the sector's own dimension, however large the request.
        @test _dims(sk(Dict{Any, Int}(q => 10^6)), 1)[q] == reach[q]
        # A label the fusion cannot produce is ignored, never created.
        @test _dims(sk(Dict{Any, Int}(:not_a_charge => 5)), 1) == base
    end
end

@testset "UT04: Test B, the holdout sampling test, on a real bond" begin
    f, skl, _ = _xx_bond()
    U0, V0 = f.U0, f.V0
    proj = X -> perp_component(U0, X)
    # `A = P_perp G` over the COMPLETE input basis; that basis is orthonormal and spans the
    # whole space, so `||(I - QQ') A||_F` is the exact quantity Test B estimates.
    Fr = full_local_basis(V0, :right)
    A = to_concrete(proj(skl(Fr)))
    nA = norm(A)
    @test nA > 0
    exact(Q) = Q === nothing ? nA : norm(to_concrete(proj(to_concrete(perp_component(Q, A)))))
    Qx, _ = RC._preselect_left(U0, skl(Fr), 1e-14)
    @test exact(Qx) < 1e-10 * nA                            # the reference basis is complete
    # The holdout over the FULL input space, which is what `A` was built on.
    rng = Random.MersenneTwister(0x5EED)
    honest = w -> sector_graded_sketch(V0, :right, w; comp_ratio = nothing, rng = rng)

    # A basis missing its strongest output sector: the residual it leaves is large and lives
    # in ONE input sector (U(1) pairs input and output charge one-to-one at fixed total N).
    qsec = collect(keys(_dims(Qx, 3)))
    @test length(qsec) >= 2
    Qmiss = argmax(Q -> exact(Q), [_drop(Qx, 3, q) for q in qsec])
    miss = exact(Qmiss)
    @test miss > 1e-3 * nA

    @testset "a complete basis leaves no residual" begin
        est, _, _ = RC._sample_residual(Qx, proj, skl, honest, 8, 1, :left)
        @test est < 1e-8 * nA
    end

    @testset "a sketch cannot certify its own range" begin
        # ⛔ THE COUNTEREXAMPLE THE TEST EXISTS FOR. `Q` spans `A*Om` by construction, so
        # re-measuring with `Om` reports nothing missed however much of the range was.
        # The sketch is made blind to the strongest input sector so that it MUST miss range:
        # a merely narrow one is not enough -- at this bond each sector's range has rank <= 1,
        # so one column per sector already captured all of it (job 17188740).
        _, perF, colsF = RC._sample_residual(nothing, proj, skl, w -> Fr, 1, 1, :left)
        qs = argmax(q -> perF[q] * colsF[q], collect(keys(perF)))
        Om = _drop(honest(4), 1, qs)
        Qs, _ = RC._preselect_left(U0, skl(Om), 1e-14)
        @test exact(Qs) > 1e-6 * nA                         # the narrow sketch DID miss range
        self, _, _ = RC._sample_residual(Qs, proj, skl, w -> Om, 1, 1, :left)
        @test self < 1e-10 * nA
        held = [first(RC._sample_residual(Qs, proj, skl, honest, 4, 1, :left)) for _ in 1:20]
        @test sum(held) / length(held) > 0.1 * exact(Qs)
    end

    @testset "per-sector normalisation is unbiased; pooling is not" begin
        # E[est^2] = exact^2 exactly. The pooled estimator the first version used weights each
        # sector by its share of the holdout's columns: the miss lives in ONE input sector
        # holding `n_in` of the `w` columns, so pooling converges to `(n_in/w) * exact^2`.
        nrep = 200
        e2, p2, share = 0.0, 0.0, 1.0
        for _ in 1:nrep
            est, per, cols = RC._sample_residual(Qmiss, proj, skl, honest, 4, 1, :left)
            e2 += est^2
            p2 += sum(cols[q] * s for (q, s) in per; init = 0.0) / sum(values(cols))
            share = cols[argmax(per)] / sum(values(cols))     # deterministic allocation
        end
        @test abs(e2 / nrep - miss^2) / miss^2 < 0.25
        # The regression this replaces: pooling is biased by exactly the column share, and
        # the share is below one whenever the holdout spans more than one sector.
        @test share < 1
        @test abs(p2 / nrep - share * miss^2) / (share * miss^2) < 0.25
    end

    @testset "the widening is aimed at the sector that holds the miss" begin
        alloc = RC.BasisAllocation(1.0, 1e-8 * nA, 1.0, 1.0, 1.0)
        _, per, _ = RC._sample_residual(Qmiss, proj, skl, honest, 4, 1, :left)
        boost = Dict{Any, Int}()
        Om = honest(4)
        @test RC._target_sectors!(boost, per, Om, 1, alloc) == 1
        @test collect(keys(boost)) == [argmax(per)]
        # Doubled: the boost is the sector's current width in the sketch.
        @test boost[argmax(per)] == max(get(_dims(Om, 1), argmax(per), 0), 1)
    end
end

@testset "UT05: Test A, the retained-rank tail test" begin

    @testset "the plan's spectrum (10, 9, 3, 0.1): the smallest k that fits" begin
        list = _mklist([10.0, 9.0, 3.0, 0.1])
        # tails: k=3 -> 0.1, k=2 -> sqrt(9.01) = 3.0017, k=1 -> sqrt(90.01) = 9.487
        @test first(RC._tail_rank(list, 1.0, 1.0, 4)) == 3
        @test first(RC._tail_rank(list, 1.0, 4.0, 4)) == 2
        @test first(RC._tail_rank(list, 1.0, 9.6, 4)) == 1
        @test first(RC._tail_rank(list, 1.0, 0.05, 4)) == 4
    end

    @testset "it keeps the smallest rank whose tail fits the allowance" begin
        # Weights 1, 1e-2, 1e-4, 1e-6: each tail is dominated by its first discarded entry.
        list = _mklist([1.0, 1e-2, 1e-4, 1e-6])
        k, tail = RC._tail_rank(list, 1.0, 2e-4, 4)
        @test k == 2
        @test tail <= 2e-4
        k, _ = RC._tail_rank(list, 1.0, 5e-5, 4)
        @test k == 3
        k, _ = RC._tail_rank(list, 1.0, 2e-2, 4)
        @test k == 1
    end

    @testset "the retained rank is monotone in the allowance" begin
        list = _mklist([1.0, 0.5, 0.2, 0.05, 0.01, 1e-3])
        ks = [first(RC._tail_rank(list, 1.0, a, 6))
              for a in (1e-6, 1e-3, 1e-2, 0.1, 0.5, 2.0)]
        # A looser budget can never demand MORE rank.
        @test issorted(ks; rev = true)
        @test ks[1] == 6        # unmeetable: nothing may be discarded
        @test ks[end] == 0      # everything fits inside the allowance
    end

    @testset "degeneracy is counted, not ignored" begin
        # One entry standing for 100 states carries 100x the weight of a lone one. Ranking on
        # sigma alone would discard it as "the same size as its neighbour" and lose 100 states.
        plain = _mklist([1.0, 0.1])
        multi = _mklist([1.0, 0.1], [1, 100])
        a = 0.5                       # fits a lone 0.1 (0.1 <= 0.5) but not 100 of them (1.0)
        @test first(RC._tail_rank(plain, 1.0, a, 2)) == 1
        @test first(RC._tail_rank(multi, 1.0, a, 2)) == 2
    end

    @testset "weight the SVD already cut is part of the tail" begin
        # ⛔ THE REGRESSION THIS GUARDS: measuring only the surviving list reports a rank as
        # certified while the weight below the SVD's own cutoff has already breached the
        # budget. The tail must be read against the WHOLE spectrum.
        list = _mklist([1.0, 1e-3])
        base = (0.5)^2
        k_nobase, _ = RC._tail_rank(list, 1.0, 0.1, 2)
        k_base, tail = RC._tail_rank(list, 1.0, 0.1, 2; base = base)
        @test k_nobase == 1
        @test k_base == 2
        @test tail > 0.1
    end

    @testset "the scale is applied, so the test is on the step's error" begin
        list = _mklist([1.0, 1e-2])
        @test first(RC._tail_rank(list, 1.0, 2e-2, 2)) == 1
        @test first(RC._tail_rank(list, 10.0, 2e-2, 2)) == 2
    end

    @testset "an unmeetable allowance returns kmax, never more" begin
        list = _mklist([1.0, 1.0, 1.0, 1.0])
        k, tail = RC._tail_rank(list, 1.0, 1e-12, 2)
        @test k == 2
        @test tail > 1e-12
    end

    @testset "degenerate inputs do not throw" begin
        @test first(RC._tail_rank(_mklist(Float64[]), 1.0, 1e-3, 4)) == 0
        @test first(RC._tail_rank(_mklist([1.0]), 1.0, 1e-3, 0)) == 0
    end

    @testset "Test A also binds a ONE-SIDED expansion (chain-edge bonds)" begin
        # At an edge bond one frame already spans its whole local space, so that side has no
        # candidates and there is no joint SVD. The other side used to keep EVERY preselected
        # direction there; it must now answer to the same tail test as a two-sided bond.
        # (allocation fields: per_bond, sample, tail, krylov, scale; Test B made lenient so
        # only Test A varies.)
        for (i, side) in ((1, :right), (7, :left))
            f, skl, skr = _xx_bond(L = 8, i = i)
            nnew(tail) = (ex = RC.cbe_expand(f, skl, skr;
                                             alloc = RC.BasisAllocation(1.0, 1e6, tail, 1.0, 1.0),
                                             ledger = AdaptiveLedger(),
                                             rng = Random.MersenneTwister(5));
                          (ex.n_new_l, ex.n_new_r))
            tight, mid, loose = nnew(1e-14), nnew(1e-3), nnew(1e6)
            pick(t) = side === :left ? t[1] : t[2]
            other(t) = side === :left ? t[2] : t[1]
            @test other(tight) == 0                          # the edge side is complete
            @test pick(tight) >= 1                           # non-vacuous: there were candidates
            @test pick(tight) >= pick(mid) >= pick(loose)    # monotone in the allowance
            @test pick(loose) == 0                           # everything fits: nothing kept
        end
    end
end

@testset "UT06: a sketch blind to a strong sector passes A, fails B, and is repaired" begin
    f, skl, _ = _xx_bond()
    U0, V0 = f.U0, f.V0
    proj = X -> perp_component(U0, X)
    Fr = full_local_basis(V0, :right)
    A = to_concrete(proj(skl(Fr)))
    nA = norm(A)
    exact(Q) = Q === nothing ? nA : norm(to_concrete(proj(to_concrete(perp_component(Q, A)))))
    tol = 1e-6 * nA
    alloc = RC.BasisAllocation(1.0, tol, tol, tol, 1.0)
    select = Om -> RC._preselect_left(U0, skl(Om), 1e-10)
    wcap = leg_dim(Fr, 1)
    ncap = leg_dim(U0, 1) * leg_dim(U0, 2) - leg_dim(U0, 3)

    # The STRONGEST input sector: the one carrying most of `A`'s weight.
    _, perF, colsF = RC._sample_residual(nothing, proj, skl, w -> Fr, 1, 1, :left)
    @test length(perF) >= 2
    qs = argmax(q -> perF[q] * colsF[q], collect(keys(perF)))

    rng = Random.MersenneTwister(0xB11D)
    honest = w -> sector_graded_sketch(V0, :right, w; comp_ratio = nothing, rng = rng)
    full(w, b) = sector_graded_sketch(V0, :right, w; comp_ratio = nothing, rng = rng,
                                      boost = b)
    # Blind to `qs` UNTIL Test B names it -- the deterministic stand-in for an unlucky draw.
    blind(w, b) = (b !== nothing && haskey(b, qs)) ? full(w, b) : _drop(full(w, b), 1, qs)

    # Drawn at FULL width, so every sector it can see is covered completely -- the only thing
    # missing is `qs`. (A narrow draw could also miss range inside the visible sectors, since
    # `_range_find`'s stop is a global count; Test B catches that too, but it would blur this.)
    Q0, _, _ = RC._range_find(w -> blind(w, nothing), 1, select, 3, wcap, wcap, ncap)
    miss0 = exact(Q0)
    @test miss0 > 1e3 * tol                                 # it really missed a strong sector

    @testset "Test A passes on the blind basis" begin
        # A ranks only what the sketch found, and the sketch found its own range completely:
        # the projected tail is small even though a strong sector is absent.
        res = svd(to_concrete(contract(Q0', (1, 2), A, (1, 2))), (1,); get_lists = true)
        n = length(res.kept_list)
        _, tail = RC._tail_rank(res.kept_list, 1.0, tol, n; base = RC._sq_weight(res.trunc_list))
        @test tail <= tol
    end

    @testset "...an audit sharing the blind spot agrees, the independent one does not" begin
        shared, _, _ = RC._sample_residual(Q0, proj, skl, w -> blind(w, nothing), 4, 1, :left)
        @test shared <= tol
        est, per, _ = RC._sample_residual(Q0, proj, skl, honest, 4, 1, :left)
        @test est > tol
        @test argmax(per) == qs                              # and it names the right sector
    end

    @testset "the controller widens that sector and certifies the result" begin
        ledger = AdaptiveLedger()
        Q1, _, _ = RC._certified_range(blind, 1, select, 3, proj, skl, :left, 2, wcap, ncap,
                                       alloc, ledger; holdout = honest)
        @test ledger.n_targeted >= 1
        @test ledger.tolerance_met
        @test !ledger.workcap_reached
        @test ledger.max_sample <= tol
        @test exact(Q1) < 1e-3 * miss0                      # exact, not just the estimate
    end

    @testset "without Test B the blind basis is accepted unchanged" begin
        Ql, _, _ = RC._certified_range(blind, 1, select, 3, proj, skl, :left, 2, wcap, ncap,
                                       nothing, nothing)
        @test exact(Ql) > 1e3 * tol
    end

    @testset "an exhausted sector is reported, not accepted" begin
        # The sector can never be drawn, so targeting cannot help: the loop must stop.
        never(w, b) = _drop(full(w, b), 1, qs)
        ledger = AdaptiveLedger()
        RC._certified_range(never, 1, select, 3, proj, skl, :left, 2, wcap, ncap,
                            alloc, ledger; holdout = honest)
        @test !ledger.tolerance_met
        @test ledger.max_sample > tol
    end

    @testset "the work ceiling stops the search and says so" begin
        ledger = AdaptiveLedger()
        # `wcap = 1`: every sketch is at least one column wide, so the cap binds on the first
        # pass, before targeting could recover the blind sector.
        RC._certified_range(blind, 1, select, 3, proj, skl, :left, 1, 1, ncap,
                            alloc, ledger; holdout = honest)
        @test ledger.workcap_reached
        @test !ledger.tolerance_met
    end
end

@testset "Stage 2 end to end: cbe_bug_step! on XX / U(1)" begin
    # ⚠ PARALLEL HALF-SWEEPS ARE ON. That is the production configuration: each half-sweep
    # carries its own ledger and RNG stream, merged after the join.
    set_symmetry!(:U1)
    L = 12
    mpo, psi0 = xxz_mpo(L; J = 1.0, delta = 0.0), neel_state(L)
    dt = ComplexF64(-im * 0.05)

    @testset "the adaptive step reports its ledger; the legacy step does not" begin
        psi = copy(psi0)
        info = cbe_bug_step!(psi, mpo, dt; tol_basis = 1e-5, tol_trunc = 1e-7,
                             tol_expv = 1e-10, maxdim = 64, parallel = true)
        @test info.ledger isa AdaptiveLedger
        @test info.ledger.n_probes > 0
        @test info.ledger.n_holdout > 0                      # Test B actually ran
        @test info.ledger.n_selected > 0
        @test isfinite(info.ledger.max_sample) && isfinite(info.ledger.max_tail)
        # Plan §3/§8: Test B is recorded PER SIDE and PER SECTOR, with its uncertainty.
        lg = info.ledger
        @test lg.max_sample == max(lg.max_sample_l, lg.max_sample_r)
        @test any(e.side === :left for e in lg.sample_log)
        @test any(e.side === :right for e in lg.sample_log)
        for e in lg.sample_log
            @test issubset(keys(e.per), keys(e.holdout))     # a sector only from its own probes
            @test all(>=(1), values(e.holdout))              # n_reachable_sectors, each probed
            @test isfinite(e.relunc) && 0 <= e.relunc
            @test e.est <= (e.side === :left ? lg.max_sample_l : lg.max_sample_r)
        end
        @test all(isfinite, bond_dims(psi))
        @test isfinite(norm(psi)) && norm(psi) > 0

        legacy = cbe_bug_step!(copy(psi0), mpo, dt; dex = 8, maxdim = 64)
        @test legacy.ledger === nothing
    end

    @testset "parallel half-sweeps are reproducible for a fixed seed" begin
        # A shared ledger loses `+=` updates and a shared MersenneTwister is not thread-safe,
        # so either race would make two identical calls disagree.
        function run_once()
            psi = copy(psi0)
            infos = [cbe_bug_step!(psi, mpo, dt; tol_basis = 1e-5, tol_trunc = 1e-7,
                                   tol_expv = 1e-10, maxdim = 64, parallel = true,
                                   rng = Random.MersenneTwister(11)) for _ in 1:3]
            lg = last(infos).ledger
            return (bond_dims(psi), lg.n_probes, lg.n_holdout, lg.n_selected, lg.n_targeted)
        end
        @test run_once() == run_once()
    end

    @testset "the midpoint step folds its predictor's ledger in" begin
        psi = copy(psi0)
        info = cbe_bug_midpoint_step!(psi, mpo, dt; tol_basis = 1e-5, tol_trunc = 1e-7,
                                      tol_expv = 1e-10, maxdim = 64, parallel = true)
        @test info.ledger isa AdaptiveLedger
        @test info.ledger.n_holdout > 0
    end

    @testset "a tighter basis tolerance never buys less rank" begin
        # MONOTONICITY IS THE CONTRACT A TOLERANCE MUST HONOUR: asking for more accuracy may
        # cost more rank, but it must never silently cost less.
        function _rank(tb)
            psi = copy(psi0)
            for _ in 1:4
                cbe_bug_step!(psi, mpo, dt; tol_basis = tb, tol_trunc = 1e-10,
                              tol_expv = 1e-10, maxdim = 64, parallel = true)
            end
            maximum(bond_dims(psi); init = 0)
        end
        @test _rank(1e-8) >= _rank(1e-3)
    end

    @testset "maxdim is still a hard ceiling under the adaptive path" begin
        psi = copy(psi0)
        for _ in 1:6
            cbe_bug_step!(psi, mpo, dt; tol_basis = 1e-10, tol_trunc = 0.0,
                          tol_expv = 1e-12, maxdim = 8, parallel = true)
        end
        @test maximum(bond_dims(psi); init = 0) <= 8
    end
end

end # adaptive RSVD (Stage 2)
