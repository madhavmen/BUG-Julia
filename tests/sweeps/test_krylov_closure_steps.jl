# tests/sweeps/test_krylov_closure_steps.jl -- UT6..UT9 of the general-order Krylov-closure plan
# (krylov_closure_cbe_bug_realtime_L16_plan.md §4) on full steps, a multi-seed stress test, and
# the local-room guards of the frame extensions.
#
#   UT6  eigenstates: no growth, exact root phase
#   UT7  undersized maxdim: finite, :budget_unmet, nothing leaks into the state
#   UT8  one final compression; tol_trunc respected; cumulative weight != worst bond
#   UT9  order = 1 is the ordinary step bit for bit; refusals
#   closure_checks: off (default) = one enriching sweep per frontier; on = the validation loop

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "krylov_closure_fixtures.jl"))

@testset "Krylov closure (general order): steps" begin

@testset "UT6: eigenstates -- no growth, exact root phase" begin
    set_symmetry!(:U1)
    L, dt = 8, 0.3
    for delta in (0.0, 1.0), checks in (false, true)  # H psi = 0, then H psi = E psi, E != 0
        mpo = xxz_mpo(L; J = 1.0, delta = delta)
        E = delta * (L - 1) / 4
        psi = product_state(fill(:up, L))
        v0 = copy(psi)
        info = cbe_bug_step!(psi, mpo, ComplexF64(-1im * dt); order = 4, tol_basis = 1e-10,
                             tol_trunc = 1e-12, tol_expv = 1e-13, closure_checks = checks)
        lg = info.ledger
        @test bond_dims(psi) == ones(Int, L - 1)
        @test lg.n_selected == 0 && lg.n_krylov == 0
        @test all(l.added == 0 for l in lg.closure_log)
        if checks
            # checked first and already closed: no closure sweep runs at all
            @test lg.n_passes == 0
            @test lg.leak_initial == lg.leak && length(lg.rho_initial) == 4
        else
            # one sweep per frontier j = 1..3, each finding nothing; no measurement
            @test lg.n_passes == 1 && length(lg.closure_log) == 3
            @test isempty(lg.leak_initial) && isnan(lg.leak[1])
        end
        @test budget_status(lg) === :met
        @test abs(overlap(v0, psi) - exp(-1im * dt * E)) < 1e-10
        @test lg.defect < 1e-12
    end
end

@testset "UT7: undersized maxdim terminates and reports" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = neel_state(L)
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.5im); order = 4, tol_basis = 1e-10,
                         tol_trunc = 1e-12, maxdim = 2)
    lg = info.ledger
    @test budget_status(lg) === :budget_unmet
    @test occursin("budget unmet", budget_diagnostic(lg))
    @test lg.workcap_reached
    @test maximum(bond_dims(psi)) <= 2                       # nothing past maxdim survives
    @test lg.r_aug <= work_ceiling(2, 1)                     # frames never passed the ceiling
    @test lg.n_passes <= RC._closure_maxpass(4)
    @test norm(psi) <= 1 + 1e-10
    @info "UT7: unmet $(lg.unmet), passes $(lg.n_passes), r_aug $(lg.r_aug), " *
          "diagnostic: $(budget_diagnostic(lg))"
    # the loop's own bound: a sweep that always grows is stopped and reported
    res = RC._closure_loop!(j -> nothing, (F, j, pass) -> (added = 1, leak = 1.0, met = true),
                            3; maxpass = 4)
    @test res.exhausted && !res.converged && res.passes == 4
    # ...and the working-rank limit (plan step 3): a pass that grows but hits the ceiling with a
    # share unmet stops at once, not converged and not exhausted
    res = RC._closure_loop!(j -> nothing,
                            (F, j, pass) -> (added = 1, leak = 1.0, met = false, capped = true),
                            3; maxpass = 4)
    @test res.capped && !res.converged && !res.exhausted && res.passes == 1
    # control: the same sweep without the ceiling runs to the pass bound
    res = RC._closure_loop!(j -> nothing,
                            (F, j, pass) -> (added = 1, leak = 1.0, met = false, capped = false),
                            3; maxpass = 4)
    @test !res.capped && res.exhausted && res.passes == 4
    # a capped sweep that meets its share is not stopped early
    res = RC._closure_loop!(j -> nothing, (F, j, pass) -> (added = pass == 1 ? 1 : 0, leak = 0.0,
                                                           met = true, capped = true),
                            3; maxpass = 4)
    @test res.converged && !res.capped && res.passes == 2
    # the step reports the stop: UT7's undersized step stopped at the ceiling, not at the bound
    @test lg.n_passes < RC._closure_maxpass(4)
end

@testset "UT8: one root_out compression, tol_trunc, cumulative vs worst" begin
    L = 10
    mpo, psi = _xx_warm(L, 3)
    tol = 1e-3
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.1im); order = 3, tol_basis = 1e-8,
                         tol_trunc = tol, maxdim = 64)
    lg = info.ledger
    @test lg.n_compress == 1
    @test lg.discarded <= tol * (1 + 1e-8)
    # per-bond weights (one value per bond, whichever sweep logged it) rebuild the estimate
    d = Dict{Int, Float64}()
    for e in lg.bond_log
        d[e.bond] = e.discarded
    end
    c = info.root
    left = sqrt(sum(d[b]^2 for b in 1:c; init = 0.0))
    right = sqrt(sum(d[b]^2 for b in (c + 1):(L - 1); init = 0.0))
    @test isapprox(left + right, lg.discarded; rtol = 1e-10)
    @test count(>(0), values(d)) >= 2                       # non-vacuous: several bonds cut
    @test lg.discarded > info.discarded                     # cumulative, not the worst bond
    @test budget_status(lg) === :met
end

@testset "UT9: order = 1 is the ordinary step; refusals" begin
    L = 8
    mpo, psi0 = _xx_warm(L, 2)
    dt = ComplexF64(-0.05im)
    a, b = copy(psi0), copy(psi0)
    ia = cbe_bug_step!(a, mpo, dt; tol_basis = 1e-6, rng = Random.MersenneTwister(7))
    ib = cbe_bug_step!(b, mpo, dt; tol_basis = 1e-6, order = 1,
                       rng = Random.MersenneTwister(7))
    @test all(norm(to_concrete(a[i] - b[i])) == 0.0 for i in 1:L)
    @test ia.krylov_dims == ib.krylov_dims
    @test ib.ledger.n_passes == 0 && isempty(ib.ledger.leak) && ib.ledger.order == 1
    # `closure_probe` only measures: same state bit for bit, and the numbers are what an
    # order-3 closure sees on the same half-sweep frames before it decides to enrich
    pc = copy(psi0)
    ic = cbe_bug_step!(pc, mpo, dt; tol_basis = 1e-6, closure_probe = 3,
                       rng = Random.MersenneTwister(7))
    @test all(norm(to_concrete(a[i] - pc[i])) == 0.0 for i in 1:L)
    @test ic.ledger.n_passes == 0 && isempty(ic.ledger.leak) && length(ic.ledger.rho_initial) == 3
    # the probe measures with the exact `H*Theta` leak, so the closure must too
    id = cbe_bug_step!(copy(psi0), mpo, dt; tol_basis = 1e-6, order = 3, closure_checks = true,
                       fold_omega = false, rng = Random.MersenneTwister(7))
    @test isapprox(ic.ledger.leak_initial, id.ledger.leak_initial; rtol = 1e-12)
    @test isapprox(ic.ledger.rho_initial, id.ledger.rho_initial; rtol = 1e-12)
    @info "UT9 probe: rho_initial $(ic.ledger.rho_initial), leak_initial " *
          "$(ic.ledger.leak_initial); order 3 passes $(id.ledger.n_passes)"
    @test_throws MethodError cbe_bug_step!(copy(psi0), mpo, dt; closure_probe = 2,
                                           krylov_basis = 3)
    am, bm = copy(psi0), copy(psi0)
    cbe_bug_midpoint_step!(am, mpo, dt; tol_basis = 1e-6, rng = Random.MersenneTwister(7))
    cbe_bug_midpoint_step!(bm, mpo, dt; tol_basis = 1e-6, order = 1,
                           rng = Random.MersenneTwister(7))
    @test all(norm(to_concrete(am[i] - bm[i])) == 0.0 for i in 1:L)
    # the unified Test A/B allocation is unchanged at order 1
    bud = AdaptiveBudget(1e-6, 1e-7, 1e-10)
    @test basis_allocation(bud, L - 1; tau = dt, nrm = 1.0) ===
          basis_allocation(bud, L - 1; tau = dt, nrm = 1.0, nfront = 1)
    @test basis_allocation(bud, L - 1; nfront = 4).per_bond ≈
          basis_allocation(bud, L - 1).per_bond / 2
    @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, dt; order = 0)
    @test_throws MethodError cbe_bug_step!(copy(psi0), mpo, dt; order = 2, krylov_basis = 3)
    @test_throws ArgumentError cbe_bug_midpoint_step!(copy(psi0), mpo, dt; order = 2)
end

@testset "closure_checks: off = one enriching sweep per frontier, on = the validation loop" begin
    # USER 2026-10-17: an order-2 step is the two half-sweeps plus ONE left || right enriching
    # sweep of j = 1 -- no check first, no re-check, no final j = 0. The checks stay as a switch.
    # A case where the half-sweep frames do NOT already close j = 1 (the warm XX state at dt 0.1
    # does, job 17332175: then neither path adds anything and the comparison is vacuous).
    L = 8
    mpo, warm = _xx_warm(L, 2)
    stepc(psi, dt, p, checks) = cbe_bug_step!(copy(psi), mpo, ComplexF64(-1im * dt); order = p,
                                            tol_basis = 1e-10, parallel = true,
                                            closure_checks = checks,
                                            rng = Random.MersenneTwister(3))
    for p in (2, 3)
        case = nothing
        for psi0 in (warm, domain_wall_state(L)), dt in (0.1, 0.3, 0.6)
            on = stepc(psi0, dt, p, true)
            if sum(e.added for e in on.ledger.closure_log; init = 0) > 0
                case = (psi0 = psi0, dt = dt, on = on)
                break
            end
        end
        @test case !== nothing                                   # non-vacuous: a case grows
        case === nothing && continue
        off = stepc(case.psi0, case.dt, p, false)
        lo, ln = off.ledger, case.on.ledger
        @test [(e.frontier, e.enrich) for e in lo.closure_log] == [(j, true) for j in 1:(p - 1)]
        @test lo.n_passes == 1 && isempty(lo.leak_initial) && isnan(lo.leak[1])
        @test sum(e.added for e in lo.closure_log) > 0          # non-vacuous: j = 1 enriched
        @test !(:closure in lo.unmet)                         # each enriching sweep met A-D
        # the checks measure first (enrich = false) and end on a measurement of j = 0
        @test !ln.closure_log[1].enrich && length(ln.closure_log) > length(lo.closure_log)
        @test ln.closure_log[end].frontier == 0 && !ln.closure_log[end].enrich
        @info "closure_checks p=$p (dt $(case.dt)): off $(length(lo.closure_log)) sweeps adding " *
              "$(sum(e.added for e in lo.closure_log)), on $(length(ln.closure_log)) sweeps " *
              "(passes $(ln.n_passes)); off: $(budget_diagnostic(lo))"
    end
end

@testset "stress: several RSVD seeds, order 2 and 3, against exp(-iHt)" begin
    L = 10
    mpo, psi0 = _xx_warm(L, 2)
    Hd = dense_heisenberg(L; J = 1.0, delta = 0.0)
    nsteps, dt = 4, 0.1
    vref = exp(-1im * nsteps * dt * Hd) * dense_state(psi0)
    for p in (2, 3)
        errs = Float64[]
        for arm in (:exact, 1, 2, 3)
            psi = copy(psi0)
            met = true
            for _ in 1:nsteps
                info = cbe_bug_step!(psi, mpo, ComplexF64(-1im * dt); order = p,
                                     tol_basis = 1e-7, parallel = true, exact = arm === :exact,
                                     rng = Random.MersenneTwister(arm === :exact ? 0 : arm))
                met &= budget_status(info.ledger) === :met
            end
            @test met
            push!(errs, norm(dense_state(psi) - vref))
        end
        @info "stress p=$p: errors exact/seed1/seed2/seed3 = $(errs)"
        @test maximum(errs) < 1e-3
        @test maximum(abs.(errs[2:end] .- errs[1])) < 1e-5     # sampling is not luck
    end
end

# A nearly saturated frame: the complete local basis of one site tensor of an evolved L=8 chain minus
# ONE column of its largest sector, with 1e-11 orthogonality loss (an accumulated-frame level).
# `side = :left` uses the centre tensor `(link, site, bond)`, `:right` the next one `(bond, site,
# link)`. Returns the frame, the source tensor, the local sector dims, the unsaturated sector, rng.
function saturated_frame(side::Symbol)
    set_symmetry!(:U1)
    L = 8
    psi = neel_state(L)
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    for _ in 1:3
        cbe_bug_step!(psi, mpo, ComplexF64(-0.3); exact = true, tol_basis = 1e-10,
                      tol_trunc = 1e-12, tol_expv = 1e-12, maxdim = 64)
    end
    canonical!(psi, side === :left ? L : 1)
    A = psi[side === :left ? L ÷ 2 : L ÷ 2 + 1]
    F = RC.full_local_basis(A, side)
    fl = side === :left ? 3 : 1
    dims = Dict(q => d for (q, d) in F.spaces[fl])
    qbig = argmax(q -> dims[q], collect(keys(dims)))
    Q = RC._trim_per_sector(F, fl, Dict(q => d - (q == qbig ? 1 : 0) for (q, d) in dims))
    rng = Random.MersenneTwister(11)
    Q = to_concrete(Q + 1e-11 * RC._randomize_like(Q, rng))
    return Q, A, dims, qbig, rng
end

@testset "_extend_frame never exceeds the local space (saturated sector + roundoff)" begin
    # s38test 17341708: a nearly saturated frame (253 of 256) gained 72 > 70 directions in one
    # sector -- roundoff cleared Telum's GLOBALLY relative cutoff because another sector carried a
    # real direction. Reproduced on `saturated_frame`, with a residual with one genuine direction
    # (the unsaturated sector) and roundoff everywhere else.
    Q, A, dims, _, rng = saturated_frame(:left)
    v = RC._randomize_like(A, rng)
    v = to_concrete((1.0 / norm(v)) * v)
    R = to_concrete(RC.perp_component(Q, v))
    Qn, nadd, _ = RC._extend_frame(Q, R, v, :left, 1.0, 0.0)
    nloc = sum(values(dims))
    @info "saturated-frame extension: local $nloc, frame $(RC.leg_dim(Q, 3)) -> $(RC.leg_dim(Qn, 3)) (+$nadd)"
    @test nadd >= 1                                                     # the genuine one is kept
    @test RC.leg_dim(Qn, 3) <= nloc
    RC._frame_sector_dims(Qn, :left, RC.fusion_basis(Qn, 1, 2; tag = "chk"), "test")  # no throw
end

@testset "cbe_expand's appended block never exceeds the local space (_guarded_block)" begin
    # d50f_trap 17346775 / d50f_rad3 17346774 (IT_FRAME_CHECKS, IT-1 L=20 domain wall): the first
    # over-wide frame was `cbe_expand V_ex` (10 > 9) and `U_ex` (253 > 252), whose re-orthogonalised
    # block had no local-room cap. A candidate block spanning the whole local space is appended to
    # `saturated_frame`: only the one free direction may survive.
    for side in (:left, :right)
        Q, _, dims, _, rng = saturated_frame(side)
        fl = side === :left ? 3 : 1
        C = RC._randomize_like(Q, rng)
        # the control: re-orthogonalisation alone turns the saturated sectors' roundoff into columns
        bare = side === :left ? RC._reortho_left(Q, C) : RC._reortho_right(Q, C)
        @test RC.leg_dim(bare, fl) > 1
        U = RC._guarded_block(Q, C, side)
        @test U !== nothing
        @test RC.leg_dim(U, fl) == 1
        # width == the local dimension and no sector over its own: the one column is the free one
        Qn = to_concrete(oplus([Q, to_concrete(setitag(U, fl, Q.inds[fl].itags))], (fl,)))
        @test RC.leg_dim(Qn, fl) == sum(values(dims))
        a, b = side === :left ? (1, 2) : (2, 3)
        RC._frame_sector_dims(Qn, side, RC.fusion_basis(Qn, a, b; tag = "chk"), "test")  # no throw
    end
end

@testset "Telum's dense SVD falls back from gesdd (_robust_svd)" begin
    # d50g16 17346776 / d50g20 17346777: gesdd threw LAPACKException(1) in `_frame_onto_left` on a
    # FINITE exact isometry (every sigma 1); the vendored Telum now retries with QR iteration.
    @test isdefined(Telum, :_robust_svd)
    M = randn(Random.MersenneTwister(3), ComplexF64, 7, 5)
    @test Telum._robust_svd(M).S ≈ svd(M).S
end

end # Krylov closure
