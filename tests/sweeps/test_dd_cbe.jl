# tests/sweeps/test_dd_cbe.jl -- updated_cbe_high_order_parallel.md, Tasks 1, 2 and 9 (Tasks 3
# and 4: test_dd_cbe_controller.jl).
#
#   Task 9  U02: the thin DD kernels' flops are exactly linear in the probe width (no X, X_mu,
#           G_DD or dense P^D) and the root generator is matrix-free.
#   Task 2  the implicit DD-projected, MPO-streamed, QR-only CBE (`dd_cbe.jl`): the streamed
#           sketch against the explicit tiny P_L^D X P_R^D Om (both directions, bulk and
#           boundary bonds, complex tensors, a w = 5 MPO), the thin projections, the two-pass
#           thin QR, missing-sector discovery, the DK-elimination fixture, no SVD in the
#           expansion, and the two-spin entangling case.
#   Task 1  original-state projection into the new frames (both gauges), the policy and limit
#           statuses in the typed step record, reproducibility from a fixed seed, and the plan's
#           API spellings (`tol_root`, `parallel = :half_sweeps`; a retired knob is no keyword).

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

@testset "DD-QR CBE (plan Tasks 1-2)" begin

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
    # The exact DD range (complete discarded probe) is captured by the exact DD expansion.
    Gdd = dd_reference_sketch(f, mpo, i, lenv, renv, full_local_basis(f.V0, :right), :left)
    exd = dd_cbe_expand(f, mpo, i, lenv, renv, :left; exact = true)
    @test norm(perp_component(exd.U_ex, Gdd)) < 1e-10 * norm(Gdd)
    @info "DK/DD fixture" dk dd = norm(Gdd) n_dd = exd.n_new_l
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

@testset "U02: every DD kernel contraction carries the probe leg; the root is matrix-free" begin
    for (fx, i) in ((_hb_bond, 4), (_hb_wide, 6))
        mpo, psi, f, lenv, renv = fx()
        @test _unsaturated(f)
        for side in (:left, :right)
            fl = _dd_kernel_flops(f, mpo, i, lenv, renv, side, 4; rng = Random.MersenneTwister(4))
            for (k, (f1, f2)) in pairs(fl)
                @test f1 > 0
                @test f2 == 2 * f1               # no X, X_mu, G_DD or dense P^D in kernel k
            end
        end
    end
    # The root generator is the two environments pushed through the frames, (bra, mpo, ket) each.
    # At r = 4 those pushes are as large as a dense root matrix would be, so only the wide bond
    # (r = 16) separates them.
    mpo, psi, f, lenv, renv = _hb_wide()
    H0 = RC.zero_site_h(mpo, 6, lenv, renv, f.U0, f.V0)
    @test H0 isa RC.MPOZeroSiteH
    @test length(H0.l.inds) == 3 && length(H0.r.inds) == 3
    N = RC.tensor_elements(apply_zero_site(H0, f.S0))       # a generic core's entries
    @test RC.tensor_elements(H0.l) + RC.tensor_elements(H0.r) < N^2
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
`T = 0.2` of Heisenberg L = 8 from Neel at step `dt`: the phase-free error vs exact.
"""
function _hb_end_to_end(dt; tol_basis = 1e-8, T = 0.2, L = 8)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    vref = dense_exact_propagate(dense_heisenberg(L; J = 1.0, delta = 1.0),
                                 dense_state(neel_state(L)), T)
    p = neel_state(L)
    for _ in 1:round(Int, T / dt)
        info = cbe_bug_step!(p, mpo, -im * dt; tol_basis = tol_basis,
                             tol_trunc = 1e-12, tol_expv = 1e-12, maxdim = 64)
        @test info.record.policy === :dd_qr_streamed
    end
    v = dense_state(p)
    return sqrt(max(2 * (1 - abs(dot(v, vref)) / (norm(v) * norm(vref))), 0.0))
end

# hq12b 17409443 (HQGAP, benchmarks/hq_dd_gap_l8.jl): DD converges as h^2 and is tolerance-blind
# (USER 2026-10-21: keep the plan design).
@testset "end to end: Heisenberg L=8 against exact propagation" begin
    err = _hb_end_to_end(0.05)
    half = _hb_end_to_end(0.025)
    loose = _hb_end_to_end(0.05; tol_basis = 1e-6)
    @info "L=8 Heisenberg, T 0.2, error vs exact" err half loose
    @test err / half >= 3.5                                      # converges in dt
    @test abs(loose - err) <= 1e-2 * err
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
    # DD+QR is the only selection and the fixed-knob path is gone: neither is a keyword
    @test_throws MethodError cbe_bug_step!(copy(psi0), mpo, -0.05im; cbe_policy = :dd_qr_streamed,
                                           kw...)
    @test_throws MethodError cbe_bug_step!(copy(psi0), mpo, -0.05im; dex = 4)
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

end # DD-QR CBE
