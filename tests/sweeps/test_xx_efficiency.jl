# tests/sweeps/test_xx_efficiency.jl -- xx_domain_wall_cbe_bug_efficiency_standalone_plan.md
# ("the plan"), Sec. 5 S1-S4, on the open XX domain wall. Kernel-level coverage already in the
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

_wall(L) = [i <= L ÷ 2 ? 1 : -1 for i in 1:L]
_kalloc(kry) = RC.BasisAllocation(1.0, 1.0, 1.0, kry, 1.0)     # only the Krylov share matters

@testset "XX domain-wall efficiency study: S1-S4" begin
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

# The test snapshot: L = 12 at J t = 2, compressed to rank 8 (Fixture M at test size).
psi_ref = xx_reference_state(12, 2.0; dt = 0.05)
snap, disc = xx_snapshot(psi_ref, 8)
mpo = xxz_mpo(12; J = 1.0, delta = 0.0)
@info "test snapshot L = 12, Jt = 2, rank 8" disc bonds = bond_dims(snap)

@testset "S2: streamed DD sketch and the two routes to ||G_DD||, bond $i" for i in (6, 4)
    bc = xx_bond_content(copy(snap), mpo, i)
    f, lenv, renv = bc.frame, bc.lenv, bc.renv
    @info "S2 bond $i content" bc.dd bc.dk bc.kd bc.kk bc.rank_dd bc.old_rank bc.disc_l bc.disc_r
    @test bc.dd > 1e-6                                           # not a trivial bond
    for side in (:left, :right)
        c = _dd_closures(f, mpo, lenv, renv, side; i = i, rng = Random.MersenneTwister(5))
        # the audit's G (complete discarded probe) and the materialised P^D X P^D agree
        @test abs(norm(c.G) - bc.dd) < 1e-10 * bc.dd
        Om = c.draw(6, nothing)                                  # P^D Om
        Y = c.project(c.sketch(Om))
        @test _rel(Y, dd_reference_sketch(f, mpo, i, lenv, renv, Om, side)) < 1e-12
    end
end

@testset "S2: the product-state wall discovers its empty reachable sectors" begin
    psi = domain_wall_state(12)
    canonical!(psi, 6)
    f = bond_frame(psi, 6)
    lenv, renv = _envs(psi, mpo, 6, 12)
    for side in (:left, :right)
        kc = KernelCounters()
        ex = dd_cbe_expand(f, mpo, 6, lenv, renv, side; kc = kc, rng = Random.MersenneTwister(3))
        @test (side === :left ? ex.n_new_l : ex.n_new_r) >= 1
        @test kc.n_sector_new >= 1
    end
end

@testset "S3: adaptive DD stopping against the exact DD residual, bond $i" for i in (6, 4)
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
    @info "S3 bond $i: $(length(rows)) runs, met $nmet, false accepts (exact > share) $fa, " *
          "mean streamed columns $(sum(r -> r.cols, rows) / length(rows))"
    @test nmet >= length(rows) ÷ 2                               # the controller is not vacuous
end

@testset "S4: Krylov stopping against eps_K on the snapshot, bond 6, $side" for side in
        (:left, :right)
    psi = copy(snap)
    canonical!(psi, side === :left ? 6 : 7)
    f = RC._frame_from(psi[6], psi[7])
    lch, rch = _envs(psi, mpo, 6, 12)
    tau = ComplexF64(-0.1im)
    # the step's own allocation at tol_basis 1e-5: a DD frame that leaves the DK part to Krylov
    alloc = basis_allocation(AdaptiveBudget(1e-5, 1e-7, 1e-10), 11; tau = tau, nrm = 1.0,
                             rule = :linear)
    ex = dd_cbe_expand(f, mpo, 6, lch, rch, side; alloc = alloc, rng = Random.MersenneTwister(8))
    kl = _kl_stage(side, mpo, 6, psi[6], psi[7], lch, rch, ex)
    kry = alloc.krylov
    e0 = eps_k(kl.Q0, kl.applyH, kl.seed, tau, side)
    @test e0 > 100 * kry                                         # CBE alone is not enough
    Q, st = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, alloc)
    ek = eps_k(Q, kl.applyH, kl.seed, tau, side)
    @test st.met
    @test ek <= 10 * kry                                         # the stop meets the response
    fixed = Any[]
    for m in (2, 4, 8, 16)
        Qm, stm = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(0.0);
                                    maxdepth = m)
        push!(fixed, (m = m, n_h1 = stm.n_h1, eps = eps_k(Qm, kl.applyH, kl.seed, tau, side)))
    end
    @info "S4 $side: allowance $kry, eps_K CBE $e0 -> adaptive $ek with $(st.n_h1) H1 " *
          "applications; fixed depths" fixed
end

end
