# tests/sweeps/test_xx_efficiency.jl -- xx_domain_wall_cbe_bug_efficiency_standalone_plan.md
# ("the plan"), Sec. 5 S1-S5, on the open XX domain wall. Kernel-level coverage already in the
# suite is not repeated: U01 (streamed DD sketch == explicit, both sides), U02 (no dense projector,
# no SVD), T2.3 (missing sector found) and T3.x in test_dd_cbe*.jl; UT07-UT10 (Test C zero, Test D
# finds the H1^2 direction) in test_adaptive_krylov.jl. Here the same kernels meet the plan's model:
#
#   S1  exact reference: Slater amplitudes == sector ED (signs included), norm, energy, S^z sum and
#       the correlation matrix, L = 4, 6, 8, 16; the MPS basis against the Slater state
#   S2  on a compressed domain-wall snapshot: streamed DD sketch == explicit, both audit routes to
#       ||G_DD|| agree, and the product-state bond discovers its empty reachable sectors
#   S3  the adaptive DD controller against the offline exact DD residual over seeds: met to the
#       share or an explicit limit, holdout cost counted, false accepts recorded
#   S4  local Krylov stopping on the snapshot against the independent eps_K, and fixed depths
#       (`fixed_depth`); the recycled recurrence's kernel identities
#   S5  krylov_recycle on order-3 steps against exact sector propagation
#
# S2-S5 run on the spin-1/2, spin-1 and spin-2 XX chains (`XX_SPINS`). Only s = 1/2 has the Slater
# oracle; at s = 1, 2 the exact reference is sector ED (spin_ed.jl), at L chosen so the S^z = 0
# sector stays ~1e4 states, and S1 pins the MPS basis against it.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "common", "spin_ed.jl"))
include(joinpath(@__DIR__, "..", "common", "ramped_tdvp.jl"))
include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dd_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "xx_efficiency_fixtures.jl"))

_wall(L, twoS = 1) = [i <= L ÷ 2 ? twoS : -twoS for i in 1:L]
const XX_SPINS = ((1, 12), (2, 10), (4, 8))              # (2s, L)
_kalloc(kry) = RC.BasisAllocation(1.0, 1.0, 1.0, kry, 1.0)     # only the Krylov share matters

@testset "XX domain-wall efficiency study: S1-S5" begin
set_symmetry!(:U1)

@testset "S1: exact XX reference, L = $L" for L in (4, 6, 8, 16)
    H, basis = spin_xx_sector(L, 1)
    v0 = spin_sector_state(basis, _wall(L))
    for t in (0.0, 0.7, 2.3)
        ex = xx_slater_amplitudes(L, t, basis)
        # phase-sensitive: a JW sign or ordering error is O(1) here, not a small number
        @test norm(ex - spin_ed_propagate(H, v0, t)) < 1e-10
        @test abs(norm(ex) - 1) < 1e-12                          # Cauchy-Binet: unit norm
        @test abs(dot(ex, H * ex)) < 1e-12                       # <H> of the wall is 0, conserved
        sz = spin_ed_sz(ex, basis)
        @test abs(sum(sz)) < 1e-12
        C = xx_open_correlation(L, t)
        @test maximum(abs.(sz .- (real(diag(C)) .- 0.5))) < 1e-12
    end
end

@testset "S1: the MPS basis is the Slater basis (signs included)" begin
    L = 8
    _, basis = spin_xx_sector(L, 1)
    @test xx_state_error(domain_wall_state(L), 0.0, basis).err < 1e-14
    psi = xx_reference_state(L, 0.5; dt = 0.025)
    e = xx_state_error(psi, 0.5, basis)
    @info "S1 MPS vs Slater, L = 8, Jt = 0.5" e
    @test e.err < 1e-6                                           # a sign error would be O(1)
    @test e.err_phase <= e.err + 1e-15
end

@testset "S1: spin-s sector ED reference and the MPS basis, 2s = $twoS, L = $L" for (twoS, L) in
        XX_SPINS[2:end]
    H, basis = spin_xx_sector(L, twoS)
    v0 = spin_sector_state(basis, _wall(L, twoS))
    @test xx_vector_error(xx_sector_amplitudes(last(xx_model(L, twoS)), basis; twoS = twoS), v0).err < 1e-14
    vt = spin_ed_propagate(H, v0, 0.5)
    @test abs(norm(vt) - 1) < 1e-12                              # unitary
    @test abs(dot(vt, H * vt)) < 1e-12                           # <H> of the wall is 0, conserved
    @test abs(sum(spin_ed_sz(vt, basis))) < 1e-12
    # the package's local basis and CG convention against the paper spin algebra, phases included
    psi = xx_reference_state(L, 0.5; dt = 0.025, twoS = twoS)
    e = xx_vector_error(xx_sector_amplitudes(psi, basis; twoS = twoS), vt)
    @info "S1 MPS vs sector ED, 2s = $twoS, L = $L, Jt = 0.5" e
    @test e.err < 1e-6
    @test maximum(abs.(spin_sz_profile(psi, twoS) .- spin_ed_sz(vt, basis))) < 1e-6
end

# S2-S5 at every spin: the test snapshot is the domain wall at J t = 2, compressed to rank 8
# (Fixture M at test size); `jc` is the front bond, `jo` the off-centre one.
for (twoS, L) in XX_SPINS
jc, jo = L ÷ 2, L ÷ 2 - 2
psi_ref = xx_reference_state(L, 2.0; dt = 0.05, twoS = twoS)
snap, disc = xx_snapshot(psi_ref, 8)
mpo = first(xx_model(L, twoS))
@info "test snapshot 2s = $twoS, L = $L, Jt = 2, rank 8" disc bonds = bond_dims(snap)

@testset "S2: streamed DD sketch and the two routes to ||G_DD||, 2s = $twoS, bond $i" for i in (jc, jo)
    bc = xx_bond_content(copy(snap), mpo, i)
    f, lenv, renv = bc.frame, bc.lenv, bc.renv
    @info "S2 bond $i content" bc.dd bc.dk bc.kd bc.kk bc.rank_dd bc.old_rank bc.disc_l bc.disc_r
    # The front bond carries real DD content; the off-centre one is nearly one-site reachable
    # (xe3: ||G_DD|| = 2.6e-8 at ||X|| = 0.37) but still far above roundoff.
    @test bc.dd > (i == jc ? 1e-6 : 1e-12 * bc.x)
    for side in (:left, :right)
        c = _dd_closures(f, mpo, lenv, renv, side; i = i, rng = Random.MersenneTwister(5))
        # the audit's G (complete discarded probe) and the materialised P^D X P^D agree
        @test abs(norm(c.G) - bc.dd) < 1e-10 * bc.dd
        Om = c.draw(6, nothing)                                  # P^D Om
        Y = c.project(c.sketch(Om))
        Yr = dd_reference_sketch(f, mpo, i, lenv, renv, Om, side)
        # P^D cancels X's KK/KD/DK part, so the floor is eps * ||X|| * ||Om||, not eps * ||Y||
        # (xe3 bond 4: relative 1.3e-10 = that floor over ||G_DD|| / ||X|| = 7e-8)
        @test norm(to_concrete(Y - Yr)) < 1e-12 * norm(Yr) + 16 * eps() * bc.x * norm(Om)
    end
end

@testset "S2: fixed-width ablation (dd_width) is one probe through the same DD+QR, 2s = $twoS" begin
    bc = xx_bond_content(copy(snap), mpo, jc)
    @test min(bc.disc_l, bc.disc_r) >= 6                     # both widths fit the discarded space
    for side in (:left, :right), l in (2, 6)
        kc, lg = KernelCounters(), RC.AdaptiveLedger()
        alloc = basis_allocation(AdaptiveBudget(1e-5, 1e-9, 1e-10), L - 1; tau = ComplexF64(-0.1im),
                                 nrm = 1.0, rule = :linear)
        ex = dd_cbe_expand(bc.frame, mpo, jc, bc.lenv, bc.renv, side; alloc = alloc, ledger = lg,
                           kc = kc, dd_width = l, rng = Random.MersenneTwister(2))
        nnew = side === :left ? ex.n_new_l : ex.n_new_r
        @test kc.n_stream_cols == l == lg.n_probes                # one block, nothing else
        @test lg.n_holdout == 0
        @test 1 <= nnew <= l
    end
end

@testset "S2: the product-state wall discovers its empty reachable sectors, 2s = $twoS" begin
    psi = last(xx_model(L, twoS))
    canonical!(psi, jc)
    f = bond_frame(psi, jc)
    lenv, renv = _envs(psi, mpo, jc, L)
    for side in (:left, :right)
        kc = KernelCounters()
        ex = dd_cbe_expand(f, mpo, jc, lenv, renv, side; kc = kc, rng = Random.MersenneTwister(3))
        @test (side === :left ? ex.n_new_l : ex.n_new_r) >= 1
        @test kc.n_sector_new >= 1
    end
end

@testset "S3: adaptive DD stopping against the exact DD residual, 2s = $twoS, bond $i" for i in (jc, jo)
    bc = xx_bond_content(copy(snap), mpo, i)
    rows = Any[]
    for side in (:left, :right), frac in (1e-2, 1e-4), seed in 1:20
        c = _dd_closures(bc.frame, mpo, bc.lenv, bc.renv, side; i = i,
                         rng = Random.MersenneTwister(1000 * i + seed))
        share = frac * norm(c.G)
        Q, st, lg, _ = _ctrl(c, share; w0 = 4)
        res = norm(c.perpQ(Q, c.G))
        push!(rows, (side = side, frac = frac, st = st, rel = res / norm(c.G), met_exact = res <= share,
                     cols = c.kc.n_stream_cols, holdout = lg.n_holdout))
        @test st in (:met, :basis_limited, :work_limited)
        @test c.kc.n_stream_cols == lg.n_probes + lg.n_holdout   # holdout cost is counted
        st === :met && @test res <= 10 * share
    end
    nmet = count(r -> r.st === :met, rows)
    fa = count(r -> r.st === :met && !r.met_exact, rows)
    @info "S3 2s = $twoS bond $i: $(length(rows)) runs, met $nmet, false accepts (exact > share) $fa, " *
          "mean streamed columns $(sum(r -> r.cols, rows) / length(rows))"
    @test nmet >= length(rows) ÷ 2                               # the controller is not vacuous
end

@testset "S4: Krylov stopping against eps_K on the snapshot, 2s = $twoS, bond $jc, $side" for side in
        (:left, :right)
    psi = copy(snap)
    canonical!(psi, side === :left ? jc : jc + 1)
    f = RC._frame_from(psi[jc], psi[jc + 1])
    lch, rch = _envs(psi, mpo, jc, L)
    tau = ComplexF64(-0.1im)
    # a loose CBE share and a tight Krylov one (as T4.1): at one share the rank-8 DD frame already
    # held the response (xe1 17580834: eps_K 7e-7 against a guard of 4e-5), nothing left to test
    alloc = basis_allocation(AdaptiveBudget(1e-7, 1e-9, 1e-10), L - 1; tau = tau, nrm = 1.0,
                             rule = :linear)
    acbe = basis_allocation(AdaptiveBudget(1e-2, 1e-9, 1e-10), L - 1; tau = tau, nrm = 1.0,
                            rule = :linear)
    ex = dd_cbe_expand(f, mpo, jc, lch, rch, side; alloc = acbe, rng = Random.MersenneTwister(8))
    kl = _kl_stage(side, mpo, jc, psi[jc], psi[jc + 1], lch, rch, ex)
    kry = alloc.krylov
    e0 = eps_k(kl.Q0, kl.applyH, kl.seed, tau, side)
    @test e0 > 100 * kry                                         # CBE alone is not enough
    Q, st = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, alloc)
    ek = eps_k(Q, kl.applyH, kl.seed, tau, side)
    @test st.met
    @test ek <= 10 * kry                                         # the stop meets the response
    # Conventional fixed depth: `_kalloc(0.0)` was NOT one -- an allowance of 0 never "fits", so
    # every depth stopped after one H1 application (xe3 17580882: n_h1 = 1 at m = 2..16).
    fixed = Any[]
    for m in (2, 4, 8, 16)
        Qm, stm = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, alloc; fixed_depth = m)
        push!(fixed, (m = m, n_h1 = stm.n_h1, met = stm.met, width = leg_dim(Qm, side === :left ? 3 : 1),
                      eps = eps_k(Qm, kl.applyH, kl.seed, tau, side)))
        @test stm.n_h1 <= m
        m <= st.n_h1 && @test stm.n_h1 == m                      # no breakdown before the adaptive stop
        @test !stm.capped
    end
    # a deeper fixed space contains the shallower one, so its miss cannot grow
    @test all(fixed[k + 1].eps <= fixed[k].eps + 1e-14 for k in 1:3)
    @info "S4 2s = $twoS $side: allowance $kry, eps_K CBE $e0 -> adaptive $ek with $(st.n_h1) H1 " *
          "applications; fixed depths" fixed

    # S5 kernel: the recycled recurrence with nothing cached IS the production one (Saad weight),
    # and with this round's own basis cached under the same operator it adds no new direction.
    Q0r, s0 = RC._krylov_enrich_recycled(kl.seed, kl.applyH, kl.Q0, side, tau, alloc, Any[])
    @test s0.n_h1 == st.n_h1 && s0.added == st.added && s0.n_recycled == 0
    @test isapprox(s0.weights, st.weights; rtol = 1e-8, atol = 1e-14)
    P = side === :left ? perp_component : perp_component_right
    @test norm(P(Q, Q0r)) < 1e-8 * norm(Q0r) && norm(P(Q0r, Q)) < 1e-8 * norm(Q)
    Q1r, s1 = RC._krylov_enrich_recycled(kl.seed, kl.applyH, kl.Q0, side, tau, alloc,
                                         st.basis[2:end])
    @test s1.n_new == 0 && s1.n_refresh == length(st.basis) && s1.met
    @test eps_k(Q1r, kl.applyH, kl.seed, tau, side) <= 10 * kry
end

@testset "S5: recycled Krylov rounds, changed gauge and operator, order 3 vs exact, 2s = $twoS" begin
    Hs, basis = spin_xx_sector(L, twoS)
    tau = ComplexF64(-0.1im)
    akw = (tol_basis = 1e-8, tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = twoS == 1 ? 64 : 128,
           order = 3)
    # recycle off / on (refreshed images) / on without images (frame seeding only), and the
    # Fig. 4 conventional controls (fixed DD width, fixed Krylov depth)
    arms = Dict{Any, Any}()
    for rc in (false, true, :frame, :conventional)
        conv = rc === :conventional
        psi, v = copy(snap), xx_sector_amplitudes(snap, basis; twoS = twoS)
        logs, nh, errs = Any[], 0, Float64[]
        for _ in 1:3
            rep = Ref{Any}(nothing)
            info = cbe_bug_step!(psi, mpo, tau; akw..., krylov_recycle = rc in (true, :frame),
                                 recycle_mode = rc === :frame ? :frame : :images,
                                 recycle_report = rep, dd_width = conv ? 8 : 0,
                                 krylov_depth = conv ? 8 : 0, rng = Random.MersenneTwister(4))
            conv && @test info.ledger.n_holdout == 0                # no adaptive check ran
            v = spin_ed_propagate(Hs, v, 0.1)                   # the exact step of the same input
            push!(errs, norm(xx_sector_amplitudes(psi, basis; twoS = twoS) - v))
            nh += info.ledger.n_h1                              # half-sweep and frontier stages
            rep[] === nothing || push!(logs, rep[])
        end
        arms[rc] = (errs = errs, n_h1 = nh, logs = logs)
    end
    F, R, Rf, Cv = arms[false], arms[true], arms[:frame], arms[:conventional]
    nrecof(a) = sum(r -> r.recycled, vcat((vcat(l[:left], l[:right]) for l in a.logs)...); init = 0)
    nrec, nrecf = nrecof(R), nrecof(Rf)
    @info "S5 2s = $twoS order 3, 3 steps from the rank-8 snapshot" fresh_err = F.errs recycled_err = R.errs fresh_h1 = F.n_h1 recycled_h1 = R.n_h1 recycled_vectors = nrec frame_err = Rf.errs frame_h1 = Rf.n_h1 frame_seeded_directions = nrecf conventional_err = Cv.errs conventional_h1 = Cv.n_h1
    # image-free recycling: the transported vectors reached the frames, and nothing was refreshed
    @test !isempty(Rf.logs)
    @test all(r.n_refresh == 0 for l in Rf.logs for r in vcat(l[:left], l[:right]))
    @test maximum(Rf.errs) < 1e-3
    @test all(abs.(Rf.errs .- F.errs) .<= 0.1 .* F.errs .+ 1e-9)
    @test isempty(F.logs) && isempty(Cv.logs)                    # off means off
    @test maximum(Cv.errs) < 1e-2                                # the control integrates too
    @test nrec > 0                                               # the arm really recycled
    @test maximum(F.errs) < 1e-3 && maximum(R.errs) < 1e-3       # both integrate (sanity) ...
    # ... to the same accuracy: the arms differ by less than a tenth of the step's own error
    @test all(abs.(R.errs .- F.errs) .<= 0.1 .* F.errs .+ 1e-9)
end

end    # for (twoS, L)

end
