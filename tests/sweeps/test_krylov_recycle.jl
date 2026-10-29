# tests/sweeps/test_krylov_recycle.jl -- the recycled local Krylov stage of `order > 1` steps
# (dissipative plan §1.2: "transport/reorthogonalize old useful directions into the new gauge, and
# extend only from active frontiers"; `krylov_recycle`, `recycled_krylov.jl`).
#
#   R1  no recycled vectors: `_krylov_enrich_recycled` IS the fresh stage (same H1 count, same
#       directions appended, same span) -- the defect weight equals Saad's Arnoldi weight
#   R2  the previous solve's own vectors: the evolved seed is captured with at most one
#       continuation image; unrelated vectors cost at most their own count and never accuracy
#   R3  transport into an unchanged tail is the identity onto the new expanded frame
#   R4  the store: what a round keeps (heaviest, frontier first, bounded) and when it is usable
#   R5  end to end: order-3 A1 (doublon loss) agrees with recycling on and off, recycling ran;
#       imaginary time never recycles

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))

_perp(side) = side === :left ? perp_component : perp_component_right
_fleg(side) = side === :left ? 3 : 1
_kalloc(kry) = RC.BasisAllocation(1.0, 1.0, 1.0, kry, 1.0)     # only the Krylov share matters
_isodef(Q, side) = side === :left ? left_isometry_defect(Q) : right_isometry_defect(Q)

"Distance `||a - b||` of two MPS from their overlaps."
_mps_dist(a, b) =
    sqrt(max(norm(a)^2 + norm(b)^2 - 2 * real(BUGJulia.BondUpdateBUG.overlap(a, b)), 0.0))

@testset "recycled local Krylov" begin

tau, kry = ComplexF64(-0.05im), 1e-7

@testset "R1: no recycled vectors -- the fresh stage" begin
    for side in (:left, :right)
        kl = _xx_kl(side; steps = 2, dex = 1)
        Qf, sf = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(kry);
                                   state_seed = false)
        Qr, sr = RC._krylov_enrich_recycled(kl.seed, Any[], kl.applyH, kl.Q0, side, tau,
                                            _kalloc(kry))
        @test sf.added >= 1                             # non-vacuous: the stage appends
        @test sr.n_recycled == 0
        @test sr.n_h1 == sf.n_h1
        @test sr.added == sf.added
        @test sr.met == sf.met
        @test leg_dim(Qr, _fleg(side)) == leg_dim(Qf, _fleg(side))
        @test norm(_perp(side)(Qr, Qf)) < 1e-8 * norm(Qf)
        @test length(sr.basis) == length(sr.weights)
    end
end

@testset "R2: recycled vectors are reused, never trusted" begin
    for side in (:left, :right)
        kl = _xx_kl(side; steps = 2, dex = 1)
        P = _perp(side)
        Y = expv(kl.applyH, tau, kl.seed; maxiter = 40, tol = 1e-15)
        _, sf = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(kry);
                                  state_seed = false)
        rec = sf.basis[2:end]
        Q, st = RC._krylov_enrich_recycled(kl.seed, rec, kl.applyH, kl.Q0, side, tau,
                                           _kalloc(kry))
        @test st.n_recycled == length(rec)
        @test st.n_h1 - (1 + st.n_recycled) <= 1        # the block already holds the evolution
        @test st.met
        @test norm(P(Q, Y)) / norm(kl.seed) <= 10 * kry
        @test norm(P(Q, kl.Q0)) < 1e-12                 # the CBE frame is kept whole
        @test _isodef(Q, side) < 1e-12
        @test length(st.basis) == length(st.weights)
        # vectors with no relation to this bond's evolution: correct, and bounded in cost
        rng = Random.MersenneTwister(11)
        junk = Any[to_concrete(RC._randomize_like(kl.seed, rng)) for _ in 1:3]
        Qj, sj = RC._krylov_enrich_recycled(kl.seed, junk, kl.applyH, kl.Q0, side, tau,
                                            _kalloc(kry))
        @test sj.met
        @test norm(P(Qj, Y)) / norm(kl.seed) <= 10 * kry
        @test _isodef(Qj, side) < 1e-12
        @test sj.n_recycled == length(junk)
        @info "R2 $side: fresh H1 $(sf.n_h1) added $(sf.added); recycled H1 $(st.n_h1) " *
              "added $(st.added); junk H1 $(sj.n_h1) added $(sj.added)"
    end
end

@testset "R3: transport into an unchanged tail" begin
    mpo, psi = _xx_warm(8, 2)
    L = length(psi)
    # left, bond 1: the vector `psi[1] (psi[2] V0')` re-expressed against the CBE frame `V_ex`
    canonical!(psi, 1)
    f = RC._frame_from(psi[1], psi[2])
    lch, rch = _envs(psi, mpo, 1, L)
    ex = RC.cbe_expand(f, mpo, 1, lch, rch; dex = 1)
    @test leg_dim(ex.V_ex, 1) > leg_dim(f.V0, 1)        # non-vacuous: the frame grew
    v = to_concrete(contract(psi[1], (3,), to_concrete(contract(psi[2], (2, 3), f.V0', (2, 3))),
                             (1,)))
    want = _kl_stage(:left, mpo, 1, psi[1], psi[2], lch, rch, ex).seed
    E = RC._rk_overlaps(psi, copy(psi), :left)
    w = RC._rk_transport((vecs = Any[v], frame = f.V0), :left, nothing, ex.V_ex, E, 1)[1]
    @test norm(to_concrete(w - want)) < 1e-12 * norm(want)
    # right, bond L-1: the mirror against `U_ex`
    canonical!(psi, L)
    f = RC._frame_from(psi[L - 1], psi[L])
    lch, rch = _envs(psi, mpo, L - 1, L)
    ex = RC.cbe_expand(f, mpo, L - 1, lch, rch; dex = 1)
    @test leg_dim(ex.U_ex, 3) > leg_dim(f.U0, 3)
    v = to_concrete(contract(to_concrete(contract(f.U0', (1, 2), psi[L - 1], (1, 2))), (2,),
                             psi[L], (1,)))
    want = _kl_stage(:right, mpo, L - 1, psi[L - 1], psi[L], lch, rch, ex).seed
    E = RC._rk_overlaps(psi, copy(psi), :right)
    w = RC._rk_transport((vecs = Any[v], frame = f.U0), :right, nothing, ex.U_ex, E, L - 1)[1]
    @test norm(to_concrete(w - want)) < 1e-12 * norm(want)
end

@testset "R4: what a round keeps, and when it is usable" begin
    rk = RC.RecycledKrylov()
    RC._rk_begin!(rk, :left)
    RC._rk_keep!(rk, :left, 3, (basis = Any[1, 2, 3, 4, 5], weights = [1, 0.5, 1e-9, 0.7, 0.2]),
                 :frame, 1e-6)
    @test rk.pend[1][3].vecs == [5, 4, 2]               # frontier first, then by weight
    RC._rk_keep!(rk, :left, 4, (basis = Any[1:20...], weights = ones(20)), :frame, 1e-6)
    @test length(rk.pend[1][4].vecs) == RC._RECYCLE_KEEP && rk.pend[1][4].vecs[1] == 20
    RC._rk_commit!(rk, :left, :tail, 1, false)          # frames not replaced: nothing usable
    @test RC._rk_round(rk, :left, 1) === nothing && rk.pend[1] === nothing
    RC._rk_begin!(rk, :left)
    RC._rk_keep!(rk, :left, 3, (basis = Any[1, 2], weights = [1, 0.5]), :frame, 1e-6)
    RC._rk_commit!(rk, :left, :tail, 1, true)
    @test RC._rk_round(rk, :left, 1).entries[3].vecs == [2]
    @test RC._rk_round(rk, :left, 2) === nothing        # written in another generation
    @test RC._rk_round(rk, :right, 1) === nothing       # the halves are separate
    @test RC._rk_keep!(nothing, :left, 1, nothing, nothing, 0.0) === nothing
end

@testset "R5: order-3 steps with and without recycling" begin
    set_symmetry!(:U1)
    L, dt, nstep = 6, 0.05, 6
    G = no_jump_mpo(hubbard_aa_mpo_d4(L; J = 1.0, U = 4.0), doublon_loss_mpo(L; kappa = 1.0))
    kw = (hermitian = false, order = 3, tol_basis = 1e-8, tol_root = 1e-12, tol_trunc = 1e-10,
          maxdim = 256)
    occ = [:both, :up, :empty, :down, :both, :empty]
    on, off = hub4_product_state(occ), hub4_product_state(occ)
    nrec, nh1 = [0, 0], [0, 0]
    for _ in 1:nstep
        ion = cbe_bug_step!(on, G, ComplexF64(-im * dt); kw..., krylov_recycle = true)
        ioff = cbe_bug_step!(off, G, ComplexF64(-im * dt); kw...)
        nrec[1] += sum(get(e, :recycled, 0) for e in ion.ledger.bond_log; init = 0)
        nrec[2] += sum(get(e, :recycled, 0) for e in ioff.ledger.bond_log; init = 0)
        nh1 .+= (ion.ledger.n_h1_closure, ioff.ledger.n_h1_closure)
    end
    d = _mps_dist(on, off) / norm(off)
    @info "R5 A1 L=$L: |on - off|/|off| $(d), recycled vectors $(nrec[1]), closure H1 " *
          "on $(nh1[1]) off $(nh1[2]), chi on $(maximum(bond_dims(on))) off " *
          "$(maximum(bond_dims(off)))"
    @test nrec[1] > 0                                   # the closure rounds recycled
    @test nrec[2] == 0                                  # and the default does not
    @test d < 1e-6                                      # same step to the tolerances
    @test isfinite(norm(on))

    # imaginary time: the store is never read
    mpo, psi = _xx_warm(8, 2)
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.05); order = 3, krylov_recycle = true,
                         tol_basis = 1e-8)
    @test all(get(e, :recycled, 0) == 0 for e in info.ledger.bond_log)
end

end # recycled local Krylov
