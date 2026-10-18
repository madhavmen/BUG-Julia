# tests/sweeps/test_adaptive_compression.jl -- UT11..UT14 of the adaptive CBE-Krylov plan (§5, §7):
# Stage 4, the rank cap and the one budgeted compression.
#
#   UT11  work ceiling / infeasible accuracy: growth stays bounded and the step says
#         `:budget_unmet`, with a reason, instead of passing silently
#   UT12  no hidden K/L truncation (the predictor keeps its rank and all of `psi`); the midpoint's
#         full step integrates from the ORIGINAL state, not the half-step one
#   UT13  the root structural SVD is lossless with the final compression off
#   UT14  `tol_trunc` bounds the ACCUMULATED discarded norm; a `maxdim` cap is reported, not
#         masked by the largest single SVD

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))

"`||a - b|| / ||a||` from overlaps -- no dense state. Resolves down to ~1e-8 (cancellation)."
_reldist(a, b) = sqrt(max(norm(a)^2 + norm(b)^2 - 2 * real(overlap(a, b)), 0.0)) / norm(a)

"Smallest rank whose discarded tail `sqrt(sum s^2)` fits `share` -- computed here, not by `_tail_rank`."
function _min_rank(svals, share)
    w = sort(abs2.(svals); rev = true)
    for k in 0:length(w)
        sqrt(sum(w[(k + 1):end]; init = 0.0)) <= share && return k
    end
    return length(w)
end

@testset "adaptive compression (Stage 4)" begin

@testset "UT11: work ceiling and infeasible accuracy" begin
    @testset "status and diagnostic are read off the ledger" begin
        lg = AdaptiveLedger()
        @test budget_status(lg) === :met
        @test budget_diagnostic(lg) == ""
        @test budget_status(nothing) === :legacy
        RC._unmet!(lg, :trunc)
        lg.maxdim_reached, lg.discarded = true, 3e-4
        @test budget_status(lg) === :budget_unmet
        @test occursin("maxdim", budget_diagnostic(lg))
        lg2 = AdaptiveLedger()
        RC._unmet!(lg2, :sample)
        lg2.workcap_reached = true
        merge_ledger!(lg2, lg)
        @test Set(lg2.unmet) == Set([:sample, :trunc])
        @test occursin("work ceiling", budget_diagnostic(lg2))
    end

    @testset "CBE: the work ceiling bounds the RETAINED directions" begin
        f, skl, skr = _xx_bond(; steps = 2)
        r = f.old_rank
        alloc = basis_allocation(AdaptiveBudget(1e-12, 1e-7, 1e-10), 7;
                                 tau = ComplexF64(-0.05im), nrm = 1.0)
        free = RC.cbe_expand(f, skl, skr; alloc = alloc, ledger = AdaptiveLedger(),
                             stol_pre = 0.0, rng = Random.MersenneTwister(3))
        @test leg_dim(free.U_ex, 3) - r >= 2            # non-vacuous: a cap of 1 must bind
        lg = AdaptiveLedger()
        capd = RC.cbe_expand(f, skl, skr; alloc = alloc, ledger = lg, workcap = r + 1,
                             stol_pre = 0.0, rng = Random.MersenneTwister(3))
        @test leg_dim(capd.U_ex, 3) <= r + 1
        @test leg_dim(capd.V_ex, 1) <= r + 1
        @test lg.workcap_reached
        @test budget_status(lg) === :budget_unmet       # the cut directions mattered: said so
    end

    @testset "Krylov: a frame extension never crosses the ceiling" begin
        # `dex = 1`: at the default budget CBE fills this bond's local space (job 17193333),
        # so a ceiling would have nothing to cut -- the same reason UT10 narrows it.
        kl = _xx_kl(:left; steps = 2, dex = 1)
        w0 = leg_dim(kl.Q0, 3)
        @test leg_dim(kl.Q0, 1) * leg_dim(kl.Q0, 2) - w0 >= 2   # room for the cap to bind
        Q, st = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, :left, ComplexF64(-1im),
                                  RC.BasisAllocation(1.0, 1.0, 1.0, 1e-12, 1.0);
                                  workcap = w0 + 1)
        @test leg_dim(Q, 3) <= w0 + 1
        @test st.capped && !st.met
    end

    @testset "the step: infeasible at a tiny maxdim -> bounded, :budget_unmet, with a reason" begin
        mpo, psi = _xx_warm(10, 3)
        r0 = maximum(bond_dims(psi))
        @test r0 > 2                                    # non-vacuous: maxdim = 2 must cut
        info = cbe_bug_step!(psi, mpo, ComplexF64(-0.1im); tol_basis = 1e-10,
                             tol_trunc = 1e-12, maxdim = 2, parallel = true)
        lg = info.ledger
        @test maximum(bond_dims(psi)) <= 2
        @test lg.r_aug <= work_ceiling(2, r0)           # the trial space is bounded too
        @test budget_status(lg) === :budget_unmet
        @test :trunc in lg.unmet && lg.maxdim_reached
        @test occursin("maxdim", budget_diagnostic(lg))
        @info "UT11 diagnostic: $(budget_diagnostic(lg))"
    end

    @testset "a state H annihilates is :met, with no growth (plan IT04 control)" begin
        set_symmetry!(:U1)
        L = 8
        mpo, psi = xxz_mpo(L; J = 1.0, delta = 0.0), product_state(fill(:up, L))
        info = cbe_bug_step!(psi, mpo, ComplexF64(-0.1im); parallel = true)
        @test budget_status(info.ledger) === :met
        @test isempty(info.ledger.unmet)
        @test maximum(bond_dims(psi)) == 1
        @test info.ledger.n_krylov == 0
    end
end

@testset "UT12: no hidden K/L truncation; the midpoint starts from Y_0" begin
    mpo, psi0 = _xx_warm(8, 2)
    tau = ComplexF64(-0.1im)
    half = copy(psi0)
    info1 = cbe_bug_step!(half, mpo, tau / 2; truncate = false, parallel = true)

    @testset "predictor (truncate = false): the K/L frames ARE the state's bonds" begin
        lg = info1.ledger
        @test lg.proj_err < 1e-12                       # the frames hold all of psi (no sqrt(eps) floor)
        @test lg.discarded == 0.0 && info1.discarded == 0.0
        c, bd = info1.root, bond_dims(half)
        @test all(bd[i] == info1.expanded[i] for i in 1:(c - 1))
        @test all(bd[j] == info1.expanded[j] for j in (c + 1):(length(half) - 1))
        @test lg.r_old <= lg.r_aug && lg.r_final <= lg.r_aug
    end

    @testset "full step: the projection of Y_0 is measured and charged to tol_basis" begin
        p = copy(psi0)
        info2 = cbe_bug_step!(p, mpo, tau; frame_state = half, parallel = true)
        lg = info2.ledger
        @test isfinite(lg.proj_err)
        @test (:projection in lg.unmet) == (lg.proj_err > 1e-5)
        @info "UT12 midpoint projection error $(lg.proj_err) (tol_basis 1e-5)"
        # The midpoint step is exactly these two stages, and keeps the FINAL stage's rank.
        q = copy(psi0)
        im2 = cbe_bug_midpoint_step!(q, mpo, tau; parallel = true)
        @test bond_dims(q) == bond_dims(p)
        @test im2.ledger.r_final == maximum(bond_dims(q))
    end

    @testset "the full step integrates from Y_0: accuracy over T = 1" begin
        set_symmetry!(:U1)
        L, dt, T = 10, 0.05, 1.0
        mpo10, psi = xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L)
        sz(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())
        for _ in 1:round(Int, T / dt)
            cbe_bug_midpoint_step!(psi, mpo10, ComplexF64(-im * dt); parallel = true)
        end
        ref = xx_free_fermion_sz(L, T; J = 1.0, occupied = 1:(L ÷ 2))
        # Starting stage 2 from `Y_{1/2}` evolves 1.5 dt per step, i.e. to 1.5 T; the reference
        # must be able to tell (non-vacuous).
        @test maximum(abs.(xx_free_fermion_sz(L, 1.5T; J = 1.0, occupied = 1:(L ÷ 2)) .- ref)) > 1e-2
        err = maximum(abs.(sz(psi) .- ref))
        @info "UT12 midpoint XX L=$L T=$T dt=$dt: err $err"
        @test err < 5e-4
    end
end

@testset "UT13: the root structural SVD is lossless with the compression off" begin
    mpo, psi0 = _xx_warm(8, 2)
    tau = ComplexF64(-0.05im)
    p = copy(psi0)
    info = cbe_bug_step!(p, mpo, tau; truncate = false, parallel = true)
    @test info.discarded == 0.0                         # the root split cut nothing
    @test info.ledger.discarded == 0.0                  # and no compression ran
    # Real time on frames that hold `psi`, nothing cut anywhere: the norm is conserved.
    @test isapprox(norm(p), norm(psi0); rtol = 1e-10)
    # `tol_trunc = 0` at an ample `maxdim` is the same state: the compression is lossless too.
    q = copy(psi0)
    iq = cbe_bug_step!(q, mpo, tau; tol_trunc = 0.0, maxdim = 10_000, parallel = true)
    @test iq.ledger.discarded == 0.0 && !iq.ledger.maxdim_reached
    @test _reldist(p, q) < 1e-7
    # simulation_plan §3: a pure K/L basis-construction sweep (`dt = 0`, `truncate = false`)
    # reconstructs the original state to numerical precision.
    z = copy(psi0)
    cbe_bug_step!(z, mpo, ComplexF64(0); truncate = false, parallel = true)
    @test _reldist(psi0, z) < 1e-7
end

@testset "UT14: tol_trunc bounds the accumulated discarded norm" begin
    L = 10
    _, psi0 = _xx_warm(L, 4)
    root = L ÷ 2 + 1
    canonical!(psi0, root)
    @test maximum(bond_dims(psi0)) > 4                  # non-vacuous: there is rank to cut
    nrm0 = norm(psi0)
    ranks = Int[]
    for tol in (1e-2, 1e-3, 1e-4)
        p = copy(psi0)
        _, worst, cum, capped = truncate_recursive!(p, root; maxdim = 10_000, weight_tol = tol)
        @test !capped
        @test cum <= tol * (1 + 1e-8)
        @test _reldist(psi0, p) <= cum + 1e-7           # the estimate is conservative
        # The norm removed is the summed Schmidt weight, between cum^2/2 and cum^2.
        loss = 1 - norm(p)^2 / nrm0^2
        @test cum^2 / 2 * (1 - 1e-8) - 1e-13 <= loss <= cum^2 * (1 + 1e-8) + 1e-13
        @test cum >= worst * sqrt(1 - cum^2) * (1 - 1e-12)
        # The prescribed spectrum: the first cut (bond root-1, at the centre) is the exact
        # Schmidt decomposition of psi0, so its rank is the minimal one for the bond's share.
        sv = [s for (s, _, _, _) in svd(psi0[root], (1,); cutoff = 0.0,
                                        get_lists = true).kept_list]
        share = tol * nrm0 / (sqrt(root - 1) + sqrt(L - root))
        @test bond_dims(p)[root - 1] == max(_min_rank(sv, share), 1)
        push!(ranks, maximum(bond_dims(p)))
    end
    @test issorted(ranks)                               # tighter tol_trunc never keeps less

    @testset "a binding maxdim is reported, not masked by the largest single SVD" begin
        p = copy(psi0)
        _, worst, cum, capped = truncate_recursive!(p, root; maxdim = 2, weight_tol = 1e-4)
        @test capped
        @test maximum(bond_dims(p)) <= 2
        @test cum > 1e-4                                # the cap, not tol_trunc, set the rank
        @test cum > worst                               # accumulation over bonds is visible
        @test _reldist(psi0, p) <= cum + 1e-7
    end
end

end # adaptive compression (Stage 4)
