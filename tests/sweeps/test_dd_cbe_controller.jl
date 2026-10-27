# tests/sweeps/test_dd_cbe_controller.jl -- updated_cbe_high_order_parallel.md, Tasks 3 and 4.
#
#   Task 4  the order-p frontier sweeps expand by the step's DD-QR CBE (`_frontier_expand`), the
#           one-site Krylov stage keeps the DD directions verbatim and adds a direction once, and
#           order-3 steps match the exact-CBE closure. P07's reuse-off ablation (`krylov_reuse =
#           false`) keeps the H1 count and both stages' directions, and still integrates.
#   Task 3  the incremental DD controller (`_dd_controller`): every streamed column cached and
#           counted once, a nonincreasing cached holdout residual, the range error against the
#           exact DD range following the share, a candidate block blind to the dominant sector
#           caught by holdouts of width 2/4/8, explicit `:work_limited`/`:basis_limited`/`:empty`
#           endings, the linear budget rule and its record, and the Taylor leak of the step's
#           frames against dense `H` (L = 8).

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "dd_fixtures.jl"))

@testset "DD-QR CBE (plan Tasks 3-4)" begin

# ── Task 3: the incremental DD controller ─────────────────────────────────────────────────

@testset "T3.1: incremental controller, every streamed column cached once, $side" for side in
        (:left, :right)
    mpo, psi, f, lenv, renv = _hb_wide()
    @test _unsaturated(f)
    c = _dd_closures(f, mpo, lenv, renv, side; rng = Random.MersenneTwister(21))
    share = 1e-3 * norm(c.G)
    Q, st, lg, tr = _ctrl(c, share)
    @info("T3.1 controller", side, wcap = c.wcap, ncap = c.ncap, rank = _rank(Q, c.fl),
          checks = length(tr), probes = lg.n_probes, holdout = lg.n_holdout)
    @test st === :met
    @test length(tr) >= 2                                        # it had to grow
    # no repeated MPO action: each streamed column is a candidate or a holdout, never both
    @test c.kc.n_stream_cols == lg.n_probes + lg.n_holdout
    # U04: every check runs on its own fresh holdout (>= s columns each), so a holdout promoted
    # into Q never validates the Q it joined
    @test lg.n_holdout >= RC._DD_HOLDOUT * length(tr)
    first_ind = [t[3] for t in tr]                               # first holdout vs each Q
    @test all(first_ind[k + 1] <= first_ind[k] * (1 + 1e-10) + 1e-14
              for k in 1:(length(first_ind) - 1))
    @test norm(c.perpQ(Q, c.G)) <= 10 * share                    # the exact DD range
    k = _rank(Q, c.fl)
    QQ = side === :left ? to_concrete(contract(Q', (1, 2), Q, (1, 2))) :
                          to_concrete(contract(Q, (2, 3), Q', (2, 3)))
    @test abs(norm(QQ)^2 - k) < 1e-10                            # Q'Q = I
    ov = side === :left ? contract(c.frame', (1, 2), Q, (1, 2)) :
                          contract(Q, (2, 3), c.frame', (2, 3))
    @test norm(to_concrete(ov)) < 1e-12                          # Q is outside the frame
    @test lg.max_sample <= share && isempty(lg.unmet)
end

@testset "T3 acceptance: the range error follows the share, $side" for side in (:left, :right)
    mpo, psi, f, lenv, renv = _hb_wide()
    errs, ranks = Float64[], Int[]
    for frac in (1e-1, 1e-2, 1e-3, 1e-5)
        c = _dd_closures(f, mpo, lenv, renv, side; rng = Random.MersenneTwister(31))
        share = frac * norm(c.G)
        Q, st, _, _ = _ctrl(c, share)
        push!(errs, norm(c.perpQ(Q, c.G)) / norm(c.G))
        push!(ranks, _rank(Q, c.fl))
        @test st === :met
        @test errs[end] <= 10 * frac
    end
    @info "T3 acceptance" side errs ranks
    @test ranks[end] >= ranks[1] && errs[end] < errs[1]
end

@testset "T3.3: candidates blind to the dominant sector, holdout s = 2, 4, 8" begin
    mpo, psi, f, lenv, renv = _hb_wide()
    c0 = _dd_closures(f, mpo, lenv, renv, :left; rng = Random.MersenneTwister(1))
    secs = [(q, _sector_norm(c0.G, c0.fl, q, d)) for (q, d) in c0.G.spaces[c0.fl]]
    @test length(secs) >= 2
    qstar = secs[argmax(last.(secs))][1]
    share = 1e-3 * norm(c0.G)
    # Control: holdout blind too -- the controller cannot see q* and accepts without it, so the
    # dropped sector carries real weight and the runs below test the holdout, not the fixture.
    c = _dd_closures(f, mpo, lenv, renv, :left; rng = Random.MersenneTwister(2))
    blind(w, b) = _drop(c.draw(w, b), c.wleg, qstar)
    Q, _, _, _ = _ctrl(c, share; draw = blind, holdout = w -> blind(w, nothing))
    @test norm(c.perpQ(Q, c.G)) > 10 * share
    false_acc = Dict{Int, Int}()
    status = Dict{Int, Vector{Symbol}}()
    for s in (2, 4, 8)
        false_acc[s] = 0
        status[s] = Symbol[]
        for seed in 1:10
            cs = _dd_closures(f, mpo, lenv, renv, :left; rng = Random.MersenneTwister(100 + seed))
            Qs, st, _, _ = _ctrl(cs, share; s = s,
                                 draw = (w, b) -> _drop(cs.draw(w, b), cs.wleg, qstar))
            push!(status[s], st)
            st === :met && norm(cs.perpQ(Qs, cs.G)) > 10 * share && (false_acc[s] += 1)
        end
    end
    @info "T3.3 false acceptances of 10 (missed dominant sector)" qstar false_acc status
    @test false_acc[8] == 0
end

@testset "T3.4: an unreachable share ends with an explicit limit, $side" for side in
        (:left, :right)
    mpo, psi, f, lenv, renv = _hb_wide()
    s = RC._DD_HOLDOUT
    c = _dd_closures(f, mpo, lenv, renv, side; rng = Random.MersenneTwister(41))
    Q, st, lg, _ = _ctrl(c, 1e-300)
    err = norm(c.perpQ(Q, c.G)) / norm(c.G)
    @info("T3.4", side, st, err, rank = _rank(Q, c.fl), streamed = c.kc.n_stream_cols,
          wcap = c.wcap)
    # `:met` only if Q holds the range to an exactly zero holdout residual
    @test st in (:work_limited, :basis_limited) || (st === :met && err <= 1e-12)
    st === :met || @test :sample in lg.unmet && !lg.tolerance_met
    st === :basis_limited && @test err <= 1e-8            # rank exhausted: nothing left to add
    @test c.kc.n_stream_cols < 2 * c.wcap + 3 * s          # each pass is bounded, so is the loop
    # a work ceiling below the first holdout
    c2 = _dd_closures(f, mpo, lenv, renv, side; rng = Random.MersenneTwister(42))
    _, st2, lg2, _ = _ctrl(c2, 1e-300; wcap = 3)
    @test st2 === :work_limited && lg2.workcap_reached && :sample in lg2.unmet
    @test c2.kc.n_stream_cols < 2 * 3 + 3 * s
    # no discarded space: nothing to draw
    c3 = _dd_closures(f, mpo, lenv, renv, side; rng = Random.MersenneTwister(43))
    Q3, st3, _, _ = _ctrl(c3, 1.0; draw = (w, b) -> nothing)
    @test st3 === :empty && Q3 === nothing && c3.kc.n_stream_cols == 0
end

# A binding work ceiling keeps `cap` of the controller's directions. `:sector` (`_trim_total`)
# fills sectors in storage order; `:weight` keeps the heaviest across sectors, so it must hold at
# least as much of the exact DD range and sit near its best rank-`cap` subspace (rp7f: the
# sector cut dropped whole sectors, obs error 2.2e-4 against 1.3e-12 uncut).
@testset "T2.5: a binding work ceiling keeps the heaviest directions, $side" for side in
        (:left, :right)
    mpo, psi, f, lenv, renv = _hb_wide()
    left = side === :left
    G = _dd_closures(f, mpo, lenv, renv, side; rng = Random.MersenneTwister(1)).G
    perp = left ? perp_component : perp_component_right
    r = f.old_rank
    alloc = RC.BasisAllocation(1.0, 1e-300, 1.0, 1.0, 1.0)      # unreachable: the ceiling binds
    run(order) = dd_cbe_expand(f, mpo, 6, lenv, renv, side; alloc = alloc, workcap = r + 4,
                               ledger = RC.AdaptiveLedger(), rng = Random.MersenneTwister(51),
                               cap_order = order)
    es, ew = run(:sector), run(:weight)
    Us, Uw = left ? (es.U_ex, ew.U_ex) : (es.V_ex, ew.V_ex)
    fl = left ? 3 : 1
    @test leg_dim(Us, fl) == leg_dim(Uw, fl) == r + 4
    res_s, res_w = norm(perp(Us, G)), norm(perp(Uw, G))
    best = svd(G, left ? (1, 2) : (2, 3); cutoff = 0.0, Nkeep = 4, get_lists = true)
    res_opt = sqrt(RC._sq_weight(best.trunc_list))
    @info("T2.5 binding ceiling: residual of the exact DD range", side, res_s, res_w, res_opt,
          norm_G = norm(G))
    @test res_w <= res_s * (1 + 1e-10)
    @test res_w <= 2 * res_opt + 1e-12 * norm(G)
    @test_throws ArgumentError dd_cbe_expand(f, mpo, 6, lenv, renv, side; cap_order = :bogus)
end

@testset "T3.2: the linear budget sums to tol_basis, and the step records its rule" begin
    b = RC.AdaptiveBudget(1e-6, 1e-8, 1e-8)
    for nb in (1, 7, 15), nf in (1, 3)
        a = RC.basis_allocation(b, nb; tau = 0.1, nrm = 2.0, nfront = nf, rule = :linear)
        @test a.rule === :linear
        @test 2 * (nb + 1) * nf * a.sample ≈ 1e-6                 # the shares add up
        @test a.per_bond ≈ 2 * a.sample && a.krylov == a.sample && a.tail == a.sample
        @test a.scale ≈ 0.05
    end
    @test RC.basis_allocation(b, 7).rule === :rss
    @test_throws ArgumentError RC.basis_allocation(b, 7; rule = :bogus)
    mpo, psi0, _, _, _ = _hb_bond()
    kw = (tol_basis = 1e-7, maxdim = 32)
    @test cbe_bug_step!(copy(psi0), mpo, -0.05im; kw...).record.budget_rule === :linear
    # USER 2026-10-27: the recycled local Krylov share (Tests C/D) is set on its own, `tol_krylov`;
    # only that share moves, and `tol_krylov = tol_basis` is the default allocation exactly
    for rule in (:linear, :rss)
        a = RC.basis_allocation(b, 7; tau = 0.1, rule = rule)
        @test RC.basis_allocation(b, 7; tau = 0.1, rule = rule, krylov = 1e-6) === a
        ak = RC.basis_allocation(b, 7; tau = 0.1, rule = rule, krylov = 1e-3)
        @test ak.krylov ≈ 1e3 * a.krylov
        @test (ak.per_bond, ak.sample, ak.tail, ak.scale) == (a.per_bond, a.sample, a.tail, a.scale)
    end
    @test_throws ArgumentError RC.basis_allocation(b, 7; krylov = 0.0)
    step(; skw...) = cbe_bug_step!(copy(psi0), mpo, -0.05im; kw..., skw...).ledger
    l0, l1, l2 = step(), step(tol_krylov = 1e-7), step(tol_krylov = 1e-3)
    @test l1.n_krylov == l0.n_krylov && l1.r_aug == l0.r_aug
    @test l0.n_krylov > 0                                       # non-vacuous
    @test l2.n_krylov < l0.n_krylov
end

# Plan T3.5: eta_m = h^m/m! ||(I - P) H^m v|| / ||v|| for the root space P of the step's frames.
# The DD step's basis error is its first-order leak; with that at the floor the higher terms are
# what is left, i.e. the step is temporally limited (the HQGAP h^2 finding, hq12b 17409443).
@testset "T3.5: Taylor leak of the step's frames against dense H, L = 8" begin
    h = 0.1
    mpo, psi0, _, _, _ = _hb_bond()
    H = dense_heisenberg(length(psi0); J = 1.0, delta = 1.0)
    v = dense_state(psi0)
    v ./= norm(v)
    rows = Any[]
    for tol in (1e-4, 1e-8)
        rep = Ref{Any}(nothing)
        cbe_bug_step!(copy(psi0), mpo, -im * h; tol_basis = tol, tol_trunc = 1e-12,
                      tol_expv = 1e-12, maxdim = 64, frames_report = rep)
        U = _root_basis(psi0, mpo, rep[])
        @test norm(v - U * (U' * v)) < 1e-8                      # the frames hold psi
        leak, w = Float64[], v
        for m in 1:3
            w = H * w
            push!(leak, h^m / factorial(m) * norm(w - U * (U' * w)))
        end
        push!(rows, (tol = tol, dim = size(U, 2), leak = leak))
        @test leak[1] <= 10 * tol + 1e-12
        # The dominant leak is a higher-order term, well above the first-order one. Not
        # `leak[2] > leak[1]`: the frames can hold H^2 psi too, and then both sit at the rounding
        # floor (hq13 17418199 at tol 1e-8: 3.4e-16, 5.1e-17, 1.9e-11).
        @test argmax(leak) > 1 && maximum(leak) > 100 * leak[1]
    end
    @info "T3.5 Taylor leak h^m/m! ||(I-P) H^m v||, m = 1..3" h rows
end

# ── Task 4: the one-site Krylov stage after DD-CBE ────────────────────────────────────────

@testset "T4.1: frontier CBE is the half-sweeps' DD-QR expansion" begin
    mpo, psi, f, lenv, renv = _hb_bond()
    i = 4
    alloc = RC.BasisAllocation(1e-6, 1e-6, 1e-6, 1e-6, 1.0)
    run(side) = RC._frontier_expand(f, mpo, i, lenv, renv, side, (alloc = alloc,),
                                    RC.AdaptiveLedger(), Random.MersenneTwister(9))
    for side in (:left, :right)
        e = run(side)
        ref = dd_cbe_expand(f, mpo, i, lenv, renv, side; alloc = alloc, version = i,
                            ledger = RC.AdaptiveLedger(), rng = Random.MersenneTwister(9))
        @test _rel(e.U_ex, ref.U_ex) < 1e-14 && _rel(e.V_ex, ref.V_ex) < 1e-14
        # one-sided: the frame opposite `side` comes back as it went in
        @test side === :left ? e.V_ex === f.V0 : e.U_ex === f.U0
    end
end

# Plan T4.1: the Krylov stage starts from the DD frame and must not regenerate what it holds.
@testset "T4.1: Krylov after DD-CBE keeps its directions and adds each once, $side" for side in
        (:left, :right)
    # `_hb_wide`: at `_hb_bond`'s cap 4 the local room is 8 and CBE alone fills it. A loose CBE
    # share and a tight Krylov one leave the Krylov stage directions to find.
    mpo, psi, f, lenv, renv = _hb_wide()
    i, tau = 6, ComplexF64(-0.05im)
    left = side === :left
    alloc = RC.BasisAllocation(1e-2, 1e-2, 1e-2, 1e-9, 1.0)
    ex = dd_cbe_expand(f, mpo, i, lenv, renv, side; alloc = alloc,
                       rng = Random.MersenneTwister(31))
    seed, Q0, applyH = _kl_stage(side, mpo, i, psi[i], psi[i + 1], lenv, renv, ex)
    perp, fl = left ? (perp_component, 3) : (perp_component_right, 1)
    Q1, st1 = RC._krylov_enrich(seed, applyH, Q0, side, tau, alloc)
    @test st1.met && st1.added > 0
    @test norm(perp(Q1, Q0)) < 1e-12 * norm(Q0)                 # the DD directions, verbatim
    @test leg_dim(Q1, fl) == leg_dim(Q0, fl) + st1.added
    Q2, st2 = RC._krylov_enrich(seed, applyH, Q1, side, tau, alloc)   # a recycled round
    @test st2.met && st2.added == 0 && leg_dim(Q2, fl) == leg_dim(Q1, fl)
    @test st2.eta_c <= alloc.krylov && st2.eta_d <= alloc.krylov
    @info("T4.1 Krylov after DD-CBE", side, old = f.old_rank, cbe = leg_dim(Q0, fl),
          krylov = st1.added, n_h1 = (st1.n_h1, st2.n_h1))
end

# The order-p frontier sweeps integrate like the exact-CBE closure. Not the
# T3.5 Taylor leak at L = 8: there the frames nearly fill the sector (dim 57 / 69 of 70) and
# order 1 already holds H^2, H^3 psi below the budget (hq13b 17419262: 3.5e-7, 1.6e-6 at 1e-4).
@testset "T4.1: order-3 steps == exact CBE to tolerance" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    tau = ComplexF64(-0.05im)
    akw = (tol_basis = 1e-7, tol_trunc = 1e-9, tol_expv = 1e-10, maxdim = 64, order = 3)
    ex, ps = neel_state(L), neel_state(L)
    for _ in 1:4
        cbe_bug_step!(ex, mpo, tau; exact = true, akw...)
        @test cbe_bug_step!(ps, mpo, tau; akw...).record.closure_policy === :dd_qr_streamed
    end
    @test 1 - _fid(ps, ex) < 1e-8
end

# Plan P07, the reuse-off ablation (`krylov_reuse = false`): the Krylov stage grows from the old
# frame and the CBE directions are merged after it. The depth is set by the propagator weight,
# not by the frame, so the `H1` count is the same; the merged frame holds both stages' directions.
@testset "P07: krylov_reuse = false, same H1 count, merged frame holds CBE and Krylov, $side" for
        side in (:left, :right)
    mpo, psi, f, lenv, renv = _hb_wide()
    i, tau = 6, ComplexF64(-0.05im)
    left = side === :left
    alloc = RC.BasisAllocation(1e-2, 1e-2, 1e-2, 1e-9, 1.0)
    ex = dd_cbe_expand(f, mpo, i, lenv, renv, side; alloc = alloc,
                       rng = Random.MersenneTwister(31))
    seed, Q0, applyH = _kl_stage(side, mpo, i, psi[i], psi[i + 1], lenv, renv, ex)
    perp, fl = left ? (perp_component, 3) : (perp_component_right, 1)
    _, son = RC._krylov_enrich(seed, applyH, Q0, side, tau, alloc)
    Qk, soff = RC._krylov_enrich(seed, applyH, left ? f.U0 : f.V0, side, tau, alloc)
    Qoff = RC._merge_frame(Qk, Q0, side)
    @test son.met && soff.met
    @test soff.n_h1 == son.n_h1
    @test norm(perp(Qoff, Q0)) < 1e-12 * norm(Q0)
    @test norm(perp(Qoff, Qk)) < 1e-12 * norm(Qk)
    # nothing outside: the frame comes back as it was (hq34b: a blockless residual crashed the SVD)
    @test leg_dim(RC._merge_frame(Q0, Q0, side), fl) == leg_dim(Q0, fl)
    @test leg_dim(RC._merge_frame(Q0, left ? f.U0 : f.V0, side), fl) == leg_dim(Q0, fl)
    @info("P07 reuse on/off", side, cbe = leg_dim(Q0, fl) - f.old_rank,
          krylov_added = (son.added, soff.added), merged = leg_dim(Qoff, fl), n_h1 = son.n_h1)
end

@testset "P07: order-3 steps with krylov_reuse = false == exact CBE to tolerance" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    tau = ComplexF64(-0.05im)
    akw = (tol_basis = 1e-7, tol_trunc = 1e-9, tol_expv = 1e-10, maxdim = 64, order = 3)
    ex, ps = neel_state(L), neel_state(L)
    nk = 0
    for _ in 1:4
        cbe_bug_step!(ex, mpo, tau; exact = true, akw...)
        nk += cbe_bug_step!(ps, mpo, tau; krylov_reuse = false, akw...).ledger.n_krylov
    end
    @test 1 - _fid(ps, ex) < 1e-8
    @info "P07 order 3, reuse off: Krylov directions added over 4 steps" nk
end

end # DD-QR CBE
