# tests/sweeps/test_dd_cbe.jl -- updated_cbe_high_order_parallel.md, Tasks 1 to 4.
#
#   Task 4  the order-p frontier sweeps expand under the step's policy (`_frontier_expand`), the
#           one-site Krylov stage keeps the DD directions verbatim and adds a direction once, and
#           order-3 steps match the exact-CBE closure. P07's reuse-off ablation (`krylov_reuse =
#           false`) keeps the H1 count and both stages' directions, and still integrates.
#   Task 3  the incremental DD controller (`_dd_controller`): every streamed column cached and
#           counted once, a nonincreasing cached holdout residual, the range error against the
#           exact DD range following the share, a candidate block blind to the dominant sector
#           caught by holdouts of width 2/4/8, explicit `:work_limited`/`:basis_limited`/`:empty`
#           endings, the linear budget rule and its record, and the Taylor leak of the step's
#           frames against dense `H` (L = 8).
#   Task 2  the implicit DD-projected, MPO-streamed, QR-only CBE (`dd_cbe.jl`): the streamed
#           sketch against the explicit tiny P_L^D X P_R^D Om (both directions, bulk and
#           boundary bonds, complex tensors, a w = 5 MPO), the thin projections, the two-pass
#           thin QR, missing-sector discovery, the DK-elimination fixture, no SVD in the
#           expansion, and the two-spin entangling case.
#   Task 1  original-state projection into the new frames (both gauges), the policy and limit
#           statuses in the typed step record, reproducibility from a fixed seed, and the plan's
#           API spellings (`tol_root`, `parallel = :half_sweeps`, `:full` refused on the legacy path).

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))

"""
A warmed, complex Heisenberg/U(1) chain (delta = 1: MPO width 5) and bond `i`'s frame.

The warm-up rank cap binds (`maxdim = 4`) so the bulk bond keeps a discarded space on BOTH sides.
At L = 8 an unbinding cap fills the centre bond (D_4 = 16 = 2^4): both `P^D` vanish there, the DD
sketch is identically zero, and every bond-4 check compares roundoff (hq12br 17409190).
"""
function _hb_bond(; L = 8, i = 4, steps = 2, maxdim = 4)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    psi = neel_state(L)
    for _ in 1:steps
        tdvp2_step!(psi, mpo, -0.2im; maxdim = maxdim, trunc_thresh = 1e-14)
    end
    canonical!(psi, i)
    f = bond_frame(psi, i)
    lenv, renv = _envs(psi, mpo, i, L)
    return mpo, psi, f, lenv, renv
end

_rel(a, b) = norm(to_concrete(a - b)) / max(norm(b), 1e-300)
"Both frames of `f` leave a discarded space, so a DD check at this bond is not vacuous."
_unsaturated(f) = leg_dim(f.U0, 1) * leg_dim(f.U0, 2) > f.old_rank &&
                  leg_dim(f.V0, 2) * leg_dim(f.V0, 3) > f.old_rank
_fid(a, b) = abs(overlap(a, b)) / (norm(a) * norm(b))

"""
The controller fixture: L = 12 warmed at cap 16, bond 6. At L = 8 / cap 4 each side's discarded
probe space is 4-dimensional, so one width-4 holdout spans it and the controller is trivial.
"""
_hb_wide() = _hb_bond(; L = 12, i = 6, steps = 4, maxdim = 16)

"""
`_dd_controller`'s closures for one side of bond `i`, built as `dd_cbe_expand` builds them, and
the exact DD range `G` (the complete discarded probe) its `Q` is graded against.
"""
function _dd_closures(f, mpo, lenv, renv, side; i = 6, rng)
    left = side === :left
    frame, opp = left ? (f.U0, f.V0) : (f.V0, f.U0)
    oside = left ? :right : :left
    kc = KernelCounters()
    op = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = !left, counters = kc)
    out = Ref{Any}(nothing)
    draw(w, b) = RC._dd_probe(opp, oside, w; rng = rng, boost = b, kc = kc)
    sketch(Om) = (apply_two_site_sketch!(out, Om, op); out[])
    project = left ? (X -> perp_component(f.U0, X)) : (X -> perp_component_right(f.V0, X))
    G = dd_reference_sketch(f, mpo, i, lenv, renv, full_local_basis(opp, oside), side)
    fused(t) = left ? leg_dim(t, 1) * leg_dim(t, 2) : leg_dim(t, 2) * leg_dim(t, 3)
    wcap = left ? leg_dim(opp, 2) * leg_dim(opp, 3) : leg_dim(opp, 1) * leg_dim(opp, 2)
    perpQ(Q, X) = Q === nothing ? X : left ? perp_component(Q, X) : perp_component_right(Q, X)
    return (; side, frame, kc, draw, sketch, project, G, perpQ, wcap,
            ncap = fused(frame) - f.old_rank, fl = left ? 3 : 1, wleg = left ? 1 : 3)
end

"Run the controller at `share` (scale 1: the indicator is the raw residual)."
function _ctrl(c, share; w0 = 2, s = RC._DD_HOLDOUT, wcap = c.wcap, draw = c.draw,
               holdout = w -> c.draw(w, nothing))
    lg, tr = RC.AdaptiveLedger(), Any[]
    Q, st = RC._dd_controller(c.frame, c.side, draw, c.sketch, c.project, w0, wcap, c.ncap,
                              RC.BasisAllocation(1.0, share, 1.0, 1.0, 1.0), lg;
                              kc = c.kc, s = s, holdout = holdout, trace = tr)
    return Q, st, lg, tr
end

_rank(Q, fl) = Q === nothing ? 0 : leg_dim(Q, fl)
_sector_norm(T, leg, q, d) = norm(to_concrete(getsub(T, leg, s -> s == q ? (1:d) : nothing)))

"`Om` without its sector `q` on `leg`; `nothing` when nothing else is left."
function _drop(Om, leg, q)
    Om === nothing && return nothing
    dims = Dict(Om.spaces[leg])
    any(k -> k != q, keys(dims)) || return nothing
    return to_concrete(getsub(Om, leg, s -> s == q ? nothing : (1:dims[s])))
end

@testset "DD-QR CBE (plan Tasks 1-3)" begin

@testset "U01: streamed DD sketch == explicit P^D X P^D Om, bond $i" for i in (1, 4, 7)
    mpo, psi, f, lenv, renv = _hb_bond(; i = i)
    i == 4 && @test _unsaturated(f)                              # the bulk case is not vacuous
    rng = Random.MersenneTwister(11 + i)
    out = Ref{Any}(nothing)
    HT = apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)

    # K-sweep orientation: probe on the right, A grows.
    Om = sector_graded_sketch(f.V0, :right, 6; comp_ratio = nothing, rng = rng)
    @test eltype(Om) <: Complex
    OmD = perp_component_right(f.V0, Om)                         # P_R^D Om
    @test _rel(perp_component_right(f.V0, OmD), OmD) < 1e-12     # P_R^D idempotent on it
    @test norm(to_concrete(contract(OmD, (2, 3), f.V0', (2, 3)))) < 1e-12 * norm(Om)
    apply_two_site_sketch!(out, OmD, TwoSiteSketchOp(f, mpo, i, lenv, renv))
    # the raw streamed action is X Om~ of the MATERIALISED two-site action, all channels
    @test _rel(out[], to_concrete(contract(HT, (3, 4), OmD', (2, 3)))) < 1e-12
    Y = perp_component(f.U0, out[])                              # P_L^D on the thin output
    @test _rel(Y, dd_reference_sketch(f, mpo, i, lenv, renv, Om, :left)) < 1e-12
    @test _rel(perp_component(f.U0, Y), Y) < 1e-12               # P_L^D Y == Y

    # L-sweep orientation: probe on the left, B grows; the complex adjoint path.
    OmL = sector_graded_sketch(f.U0, :left, 6; comp_ratio = nothing, rng = rng)
    OmLD = perp_component(f.U0, OmL)
    apply_two_site_sketch!(out, OmLD, TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = true))
    raw = to_concrete(permutedims(contract(HT, (1, 2), OmLD', (1, 2)), (3, 1, 2)))
    @test _rel(out[], raw) < 1e-12
    YR = perp_component_right(f.V0, out[])
    @test _rel(YR, dd_reference_sketch(f, mpo, i, lenv, renv, OmL, :right)) < 1e-12
end

@testset "U03/T2.4: two-pass thin QR, both sides" begin
    mpo, psi, f, lenv, renv = _hb_bond()
    i = 4
    @test _unsaturated(f)
    rng = Random.MersenneTwister(5)
    out = Ref{Any}(nothing)
    for side in (:left, :right)
        left = side === :left
        frame, opp = left ? (f.U0, f.V0) : (f.V0, f.U0)
        Om = RC._dd_probe(opp, left ? :right : :left, 8; rng = rng)
        apply_two_site_sketch!(out, Om, TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = !left))
        kc = KernelCounters()
        Q, backend = RC._dd_select(frame, out[], side, kc)
        @test backend === :householder
        @test Q !== nothing
        fl = left ? 3 : 1
        k = leg_dim(Q, fl)
        @test kc.qr_rank == k && kc.n_qr == 1
        G = left ? to_concrete(contract(Q', (1, 2), Q, (1, 2))) :
                   to_concrete(contract(Q, (2, 3), Q', (2, 3)))
        @test abs(norm(Q)^2 - k) < 1e-10 && abs(norm(G)^2 - k) < 1e-10   # Q'Q = I
        ov = left ? contract(frame', (1, 2), Q, (1, 2)) : contract(Q, (2, 3), frame', (2, 3))
        @test norm(to_concrete(ov)) < 1e-12                               # A'Q = 0, B Q = 0
        Y = left ? perp_component(frame, out[]) : perp_component_right(frame, out[])
        miss = left ? perp_component(Q, Y) : perp_component_right(Q, Y)
        @test norm(miss) < 1e-10 * norm(Y)                                # range kept whole
        @test k <= leg_dim(Om, left ? 1 : 3)                             # QR never widens
    end
end

@testset "T2.3: a reachable sector the frame does not hold is found" begin
    for side in (:left, :right)
        mpo, psi, f, lenv, renv = _hb_bond(; steps = 0)            # Neel: rank-1 frames
        kc = KernelCounters()
        ex = dd_cbe_expand(f, mpo, 4, lenv, renv, side; kc = kc,
                           rng = Random.MersenneTwister(3))
        grown = side === :left ? leg_dim(ex.U_ex, 3) : leg_dim(ex.V_ex, 1)
        @test grown > f.old_rank
        @test kc.n_sector_new >= 1
        # the other frame is untouched: one QR per half-sweep orientation
        @test side === :left ? ex.V_ex === f.V0 : ex.U_ex === f.U0
    end
end

@testset "DK/DD fixture: DD drops the one-site-accessible part, keeps the DD range" begin
    mpo, psi, f, lenv, renv = _hb_bond()
    i = 4
    @test _unsaturated(f)
    rng = Random.MersenneTwister(9)
    out = Ref{Any}(nothing)
    op = TwoSiteSketchOp(f, mpo, i, lenv, renv)
    # A probe in the KEPT right space: its one-sided image is P_L^D X P_R^K Om (DK) ...
    Om = sector_graded_sketch(f.V0, :right, 8; comp_ratio = nothing, rng = rng)
    OmK = to_concrete(Om - perp_component_right(f.V0, Om))
    apply_two_site_sketch!(out, OmK, op)
    dk = norm(perp_component(f.U0, out[]))
    @test dk > 1e-6                                              # a genuine DK component
    # ... and its DD image vanishes: P_R^D removes it before the MPO is applied
    @test norm(perp_component_right(f.V0, OmK)) < 1e-12 * norm(OmK)
    # The exact DD range (complete discarded probe) is captured by the exact DD expansion,
    # and the one-sided exact expansion keeps at least as many directions.
    Gdd = dd_reference_sketch(f, mpo, i, lenv, renv, full_local_basis(f.V0, :right), :left)
    exd = dd_cbe_expand(f, mpo, i, lenv, renv, :left; exact = true)
    ex1 = dd_cbe_expand(f, mpo, i, lenv, renv, :left; exact = true,
                        policy = :one_sided_qr_streamed)
    @test norm(perp_component(exd.U_ex, Gdd)) < 1e-10 * norm(Gdd)
    @test ex1.n_new_l >= exd.n_new_l
    @info "DK/DD fixture" dk dd = norm(Gdd) n_dd = exd.n_new_l n_one_sided = ex1.n_new_l
end

@testset "U02: no SVD inside the DD expansion (rank stop and T3 controller)" begin
    mpo, psi, f, lenv, renv = _hb_bond()
    ctrl = RC.BasisAllocation(1.0, 1e-6, 1.0, 1.0, 1.0)
    for side in (:left, :right), alloc in (nothing, ctrl)
        lg = alloc === nothing ? nothing : RC.AdaptiveLedger()
        dd_cbe_expand(f, mpo, 4, lenv, renv, side)                # warms the fusion cache
        set_accumul_costs!(true)
        read_reset_costs!()
        dd_cbe_expand(f, mpo, 4, lenv, renv, side; alloc = alloc, ledger = lg,
                      rng = Random.MersenneTwister(2))
        _, svdc = read_reset_costs!()
        set_accumul_costs!(false)
        @test svdc == 0
    end
end

@testset "two-spin entangling step: DD finds |1> on both sides, exact to roundoff" begin
    set_symmetry!(:U1)
    mpo = xxz_mpo(2; J = 1.0, delta = 0.0)
    psi = product_state([:up, :down])
    t = 0.3
    info = cbe_bug_step!(psi, mpo, -im * t; tol_basis = 1e-10, tol_trunc = 1e-14,
                         tol_expv = 1e-13, maxdim = 4)
    @test info.record.policy === :dd_qr_streamed
    a = abs(overlap(product_state([:up, :down]), psi)) / norm(psi)
    b = abs(overlap(product_state([:down, :up]), psi)) / norm(psi)
    @test abs(a - cos(t / 2)) < 1e-10
    @test abs(b - sin(t / 2)) < 1e-10
end

"""
`T = 0.2` of Heisenberg L = 8 from Neel with `policy` at step `dt`: the phase-free error vs exact.
"""
function _hb_end_to_end(pol, dt; tol_basis = 1e-8, T = 0.2, L = 8)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    vref = dense_exact_propagate(dense_heisenberg(L; J = 1.0, delta = 1.0),
                                 dense_state(neel_state(L)), T)
    p = neel_state(L)
    for _ in 1:round(Int, T / dt)
        info = cbe_bug_step!(p, mpo, -im * dt; cbe_policy = pol, tol_basis = tol_basis,
                             tol_trunc = 1e-12, tol_expv = 1e-12, maxdim = 64)
        @test info.record.policy === pol
    end
    v = dense_state(p)
    return sqrt(max(2 * (1 - abs(dot(v, vref)) / (norm(v) * norm(vref))), 0.0))
end

# hq12b 17409443 (HQGAP, benchmarks/hq_dd_gap_l8.jl): DD converges as h^2 and is tolerance-blind;
# svd_select as h^3 and ~100x smaller. The gap is the K/L-step wiring, not the selection: the svd
# path's one-site Krylov runs through the CBE-widened `V_ex`, the plan's DD path through `V0`
# (one QR per orientation) and its QR adds only first-order DD directions. USER 2026-10-21: keep
# the plan design; parity is deferred to the Taylor-leak (T3.5) and recycled-round (T4) work.
@testset "end to end: Heisenberg L=8 against exact propagation" begin
    errs = Dict(pol => _hb_end_to_end(pol, 0.05) for pol in CBE_POLICIES)
    half = _hb_end_to_end(:dd_qr_streamed, 0.025)
    loose = _hb_end_to_end(:dd_qr_streamed, 0.05; tol_basis = 1e-6)
    @info "L=8 Heisenberg, T 0.2, error vs exact" errs half loose
    @test errs[:svd_select] < 1e-4
    @test errs[:dd_qr_streamed] / half >= 3.5                    # converges in dt
    @test abs(loose - errs[:dd_qr_streamed]) <= 1e-2 * errs[:dd_qr_streamed]
    @test abs(errs[:one_sided_qr_streamed] - errs[:dd_qr_streamed]) <=
          1e-6 * errs[:dd_qr_streamed]
    @test_broken errs[:dd_qr_streamed] <= 10 * errs[:svd_select] + 1e-8   # T3.5 / T4
end

@testset "T1.2: the original state projects into the new frames losslessly, both gauges" begin
    for centre in (:first, :last)
        mpo, psi, _, _, _ = _hb_bond()
        L = length(psi)
        canonical!(psi, centre === :first ? 1 : L)
        n0 = norm(psi)
        info = cbe_bug_step!(psi, mpo, -0.05im; tol_basis = 1e-8, tol_trunc = 1e-12,
                             tol_expv = 1e-13, maxdim = 64, truncate = false)
        @test info.ledger.proj_err < 1e-12            # ||(1 - P) psi_n|| / ||psi_n||
        @test abs(norm(psi) / n0 - 1) < 1e-10         # lossless root split, unitary root
    end
end

@testset "T1.4: typed step record, reproducible from a fixed seed" begin
    mpo, psi0, _, _, _ = _hb_bond()
    run1() = (p = copy(psi0);
              i = cbe_bug_step!(p, mpo, -0.05im; tol_basis = 1e-7, maxdim = 32,
                                rng = Random.MersenneTwister(77)); (p, i.record))
    p1, r1 = run1()
    p2, r2 = run1()
    @test 1 - _fid(p1, p2) < 1e-14
    for fld in (:n_stream, :n_stream_cols, :n_one_site, :n_root, :n_qr, :qr_rank, :n_sector_new)
        @test getfield(r1.counters, fld) == getfield(r2.counters, fld)
    end
    @test (r1.D_old, r1.D_stage, r1.D_final) == (r2.D_old, r2.D_stage, r2.D_final)
    @test r1.policy === :dd_qr_streamed && r1.qr_backend === :householder
    @test r1.closure_policy === :none && r1.parallel === :serial
    @test r1.counters.n_stream > 0 && r1.counters.n_qr > 0 && r1.counters.qr_rank > 0
    @test r1.counters.n_one_site > 0 && r1.counters.n_root > 0
    @test r1.counters.t_stream > 0 && r1.counters.t_qr > 0
    @test r1.D_stage >= r1.D_old && r1.root_iters > 0 && r1.peak_rss > 0
    @test r1.limits isa Vector{Symbol}
end

@testset "API: policies, limits, tol_root, parallel" begin
    mpo, psi0, _, _, _ = _hb_bond()
    kw = (tol_basis = 1e-7, maxdim = 32)
    i1 = cbe_bug_step!(copy(psi0), mpo, -0.05im; cbe_policy = :one_sided_qr_streamed, kw...)
    @test i1.record.policy === :one_sided_qr_streamed
    il = cbe_bug_step!(copy(psi0), mpo, -0.05im; dex = 4)        # legacy path
    @test il.record.policy === :svd_select && il.record.limits == [:legacy]
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, -0.05im; dex = 4,
                                             cbe_policy = :dd_qr_streamed)
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, -0.05im; cbe_policy = :bogus)
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, -0.05im; parallel = :full, dex = 4)
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, -0.05im; parallel = :bogus, kw...)
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, -0.05im; tol_root = 1e-9,
                                             tol_expv = 1e-9)
    ia = cbe_bug_step!(copy(psi0), mpo, -0.05im; tol_root = 1e-9, kw...)
    ib = cbe_bug_step!(copy(psi0), mpo, -0.05im; tol_expv = 1e-9, kw...)
    @test ia.ledger.expv_err == ib.ledger.expv_err
    ih = cbe_bug_step!(copy(psi0), mpo, -0.05im; parallel = :half_sweeps, kw...)
    @test ih.record.parallel === :half_sweeps
    io = cbe_bug_step!(copy(psi0), mpo, -0.05im; order = 2, kw...)
    @test io.record.closure_policy === :dd_qr_streamed  # frontier sweeps: plan T4.1
end

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
    @test cbe_bug_step!(copy(psi0), mpo, -0.05im; cbe_policy = :one_sided_qr_streamed,
                        kw...).record.budget_rule === :linear
    @test cbe_bug_step!(copy(psi0), mpo, -0.05im; cbe_policy = :svd_select,
                        kw...).record.budget_rule === :rss
    @test cbe_bug_step!(copy(psi0), mpo, -0.05im; dex = 4).record.budget_rule === :legacy
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

@testset "T4.1: frontier CBE runs the step's policy" begin
    mpo, psi, f, lenv, renv = _hb_bond()
    i = 4
    alloc = RC.BasisAllocation(1e-6, 1e-6, 1e-6, 1e-6, 1.0)
    skl, skr = RC._sketch_closures(f, () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv))
    run(pol, side) = RC._frontier_expand(f, skl, skr, mpo, i, lenv, renv, side,
                                         (cbe_policy = pol, alloc = alloc), RC.AdaptiveLedger(),
                                         Random.MersenneTwister(9))
    for side in (:left, :right)
        e = run(:dd_qr_streamed, side)
        ref = dd_cbe_expand(f, mpo, i, lenv, renv, side; alloc = alloc, version = i,
                            ledger = RC.AdaptiveLedger(), rng = Random.MersenneTwister(9))
        @test _rel(e.U_ex, ref.U_ex) < 1e-14 && _rel(e.V_ex, ref.V_ex) < 1e-14
        # one-sided: the frame opposite `side` comes back as it went in
        @test side === :left ? e.V_ex === f.V0 : e.U_ex === f.U0
    end
    s = run(:svd_select, :left)                     # two-sided: both frames grow
    @test leg_dim(s.U_ex, 3) > f.old_rank && leg_dim(s.V_ex, 1) > f.old_rank
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

# The order-p frontier sweeps under each policy integrate like the exact-CBE closure. Not the
# T3.5 Taylor leak at L = 8: there the frames nearly fill the sector (dim 57 / 69 of 70) and
# order 1 already holds H^2, H^3 psi below the budget (hq13b 17419262: 3.5e-7, 1.6e-6 at 1e-4).
@testset "T4.1: order-3 steps under $pol == exact CBE to tolerance" for pol in
        (:dd_qr_streamed, :svd_select)
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    tau = ComplexF64(-0.05im)
    akw = (tol_basis = 1e-7, tol_trunc = 1e-9, tol_expv = 1e-10, maxdim = 64, order = 3)
    ex, ps = neel_state(L), neel_state(L)
    for _ in 1:4
        cbe_bug_step!(ex, mpo, tau; exact = true, akw...)
        @test cbe_bug_step!(ps, mpo, tau; cbe_policy = pol, akw...).record.closure_policy === pol
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
