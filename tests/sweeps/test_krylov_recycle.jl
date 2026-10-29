# tests/sweeps/test_krylov_recycle.jl -- the recycled local Krylov stage of `order > 1` steps
# (dissipative plan §1.2: "transport/reorthogonalize old useful directions into the new gauge, and
# extend only from active frontiers"; `krylov_recycle` with `recycle_mode = :complement`,
# `krylov_recycle.jl` and `_rk_complement` in `krylov_enrich.jl`).
#
#   R1  no recycled vectors: `_krylov_enrich(...; recycled)` IS the fresh stage (same H1 count,
#       same directions appended, same span)
#   R2  the previous solve's own input and vectors: reused, nothing extended; an unrelated set
#       is rejected by its defect and costs at most its own count, never accuracy
#   R3  transport into an unchanged tail is the identity onto the new expanded frame
#   R4  the store: what a round keeps (its input, then frontier and heaviest, bounded) and when
#       it is usable
#   R5  end to end: order-3 A1 (doublon loss) agrees with recycling on and off, recycling ran;
#       imaginary time never recycles
#   R6  the complement solve on plain matrices (`_krylov_basis`) with committed frames: inputs
#       near and far from the previous one, generator unchanged or moved, against the fresh solve
#       and `exp(tau H) v0`

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
        Qr, sr = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(kry);
                                   state_seed = false, recycled = Any[])
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
        rec = sf.basis                                  # its input first, then its vectors
        Q, st = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(kry);
                                  state_seed = false, recycled = rec)
        @test st.n_recycled == length(rec)
        @test st.rec_closed                             # the block holds the input's evolution
        @test st.n_h1 == st.n_recycled                  # and its complement is empty
        @test st.met
        @test norm(P(Q, Y)) / norm(kl.seed) <= 10 * kry
        @test norm(P(Q, kl.Q0)) < 1e-12                 # the CBE frame is kept whole
        @test _isodef(Q, side) < 1e-12
        @test length(st.basis) == length(st.weights)
        # vectors with no relation to this bond's evolution: correct, and bounded in cost
        rng = Random.MersenneTwister(11)
        junk = Any[to_concrete(RC._randomize_like(kl.seed, rng)) for _ in 1:3]
        Qj, sj = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(kry);
                                   state_seed = false, recycled = junk)
        @test sj.met
        @test norm(P(Qj, Y)) / norm(kl.seed) <= 10 * kry
        @test _isodef(Qj, side) < 1e-12
        @test !sj.rec_closed                            # rejected by its defect
        @test sj.n_h1 <= sf.n_h1 + length(junk)         # one image each, the chain no longer
        @test sj.added == sf.added                      # then exactly the fresh solve
        @test !sj.capped
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
    E = RC._tail_overlaps(psi, copy(psi), :left)
    w = RC._recycle_transport((basis = Any[v], frame = f.V0), :left, nothing, ex.V_ex, E, 1)[1]
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
    E = RC._tail_overlaps(psi, copy(psi), :right)
    w = RC._recycle_transport((basis = Any[v], frame = f.U0), :right, nothing, ex.U_ex, E,
                              L - 1)[1]
    @test norm(to_concrete(w - want)) < 1e-12 * norm(want)
end

@testset "R4: what a round keeps, and when it is usable" begin
    rk = RC.KrylovRecycle(:complement)
    RC._recycle_begin!(rk, :left, :tail, 1)
    RC._recycle_store!(rk, :left, 3,
                       (basis = Any[1, 2, 3, 4, 5], weights = [1, 0.5, 1e-9, 0.7, 0.2]),
                       :frame, 1e-6)
    @test rk.pend[:left].entries[3].basis == [1, 5, 4, 2]   # input, frontier, then by weight
    @test rk.pend[:left].entries[3].weights == [1, 0.2, 0.7, 0.5]
    RC._recycle_store!(rk, :left, 4, (basis = Any[1:20...], weights = ones(20)), :frame, 1e-6)
    v4 = rk.pend[:left].entries[4].basis
    @test length(v4) == RC._RECYCLE_KEEP + 1 && v4[1] == 1 && v4[2] == 20
    RC._recycle_commit!(rk, :left, false)               # frames not replaced: nothing usable
    @test RC._recycle_entries(rk, :left, 1) === nothing && rk.pend[:left] === nothing
    RC._recycle_begin!(rk, :left, :tail, 1)
    RC._recycle_store!(rk, :left, 3, (basis = Any[1, 2], weights = [1, 0.5]), :frame, 1e-6)
    RC._recycle_commit!(rk, :left, true)
    @test RC._recycle_entries(rk, :left, 1).entries[3].basis == [1, 2]
    @test RC._recycle_entries(rk, :left, 2) === nothing     # written in another generation
    @test RC._recycle_entries(rk, :right, 1) === nothing    # the halves are separate
    @test RC._recycle_store!(nothing, :left, 1, nothing, nothing, 0.0) === nothing
    @test_throws ArgumentError RC.KrylovRecycle(:bogus)
end

@testset "R5: order-3 steps with and without recycling" begin
    set_symmetry!(:U1)
    L, dt, nstep = 6, 0.05, 6
    G = no_jump_mpo(hubbard_aa_mpo_d4(L; J = 1.0, U = 4.0), doublon_loss_mpo(L; kappa = 1.0))
    kw = (hermitian = false, order = 3, tol_basis = 1e-8, tol_root = 1e-12, tol_trunc = 1e-10,
          maxdim = 256, recycle_mode = :complement)
    occ = [:both, :up, :empty, :down, :both, :empty]
    on, off = hub4_product_state(occ), hub4_product_state(occ)
    nrec, nh1, nkry, raug, nrc = [0, 0], [0, 0], [0, 0], [0, 0], 0
    for _ in 1:nstep
        ion = cbe_bug_step!(on, G, ComplexF64(-im * dt); kw..., krylov_recycle = true)
        ioff = cbe_bug_step!(off, G, ComplexF64(-im * dt); kw...)     # the default: off
        nrec[1] += sum(get(e, :recycled, 0) for e in ion.ledger.bond_log; init = 0)
        nrec[2] += sum(get(e, :recycled, 0) for e in ioff.ledger.bond_log; init = 0)
        nrc += count(e -> get(e, :rec_closed, false), ion.ledger.bond_log)
        nh1 .+= (ion.ledger.n_h1_closure, ioff.ledger.n_h1_closure)
        nkry .+= (ion.ledger.n_krylov, ioff.ledger.n_krylov)
        raug .= max.(raug, (ion.ledger.r_aug, ioff.ledger.r_aug))
    end
    d = _mps_dist(on, off) / norm(off)
    @info "R5 A1 L=$L: |on - off|/|off| $(d), recycled vectors $(nrec[1]) (bonds closed by " *
          "them $(nrc)), closure H1 on $(nh1[1]) off $(nh1[2]), Krylov directions appended " *
          "on $(nkry[1]) off $(nkry[2]), r_aug on $(raug[1]) off $(raug[2]), chi on " *
          "$(maximum(bond_dims(on))) off $(maximum(bond_dims(off)))"
    @test nrec[1] > 0                                   # the closure rounds recycled
    @test nrec[2] == 0                                  # and the default does not
    @test d < 1e-6                                      # same step to the tolerances
    @test isfinite(norm(on))

    # imaginary time: the store is never read
    mpo, psi = _xx_warm(8, 2)
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.05); order = 3, krylov_recycle = true,
                         recycle_mode = :complement, tol_basis = 1e-8)
    @test all(get(e, :recycled, 0) == 0 for e in info.ledger.bond_log)
end

@testset "R6: the complement solve on plain matrices" begin
    # a banded generator `H - (i/2) loss Q` and a localised seed, as a closure frontier is
    function model(n, loss, rng)
        H = zeros(ComplexF64, n, n)
        for k in 1:5
            b = randn(rng, ComplexF64, n - k)
            H .+= diagm(k => b) .+ diagm(-k => conj.(b))
        end
        H .+= diagm(0 => 3 .* randn(rng, n) .- 0.5im * loss .* abs.(randn(rng, n)))
        v0 = randn(rng, ComplexF64, n) .* exp.(-((1:n) .- n / 2) .^ 2 ./ (n / 7)^2)
        return H, v0 / norm(v0)
    end
    # the frame: a scored direction appends its out-of-frame part (`_extend_frame` in miniature)
    function solve(H, v0, tau, rec, Q0, allow)
        Q, nadd = Ref(Q0), Ref(0)
        sc = function (v, w, first)
            r = v - Q[] * (Q[]' * v)
            r -= Q[] * (Q[]' * r)
            nr = norm(r)
            w * nr > allow && (Q[] = hcat(Q[], r / nr); nadd[] += 1)
            return true
        end
        kb = RC._krylov_basis(v0, x -> H * x, tau, allow, sc; recycled = rec, state_seed = false)
        Y = exp(tau * H) * (v0 / norm(v0))
        return merge(kb, (added = nadd[], Q = Q[], err = norm(Y - Q[] * (Q[]' * Y))))
    end
    allow = 1e-7
    for (n, loss, tau) in ((120, 0.0, -0.1im), (200, 1.0, -0.2im), (200, 3.0, -0.05im))
        rng = Random.MersenneTwister(n + round(Int, 10 * loss))
        H, vprev = model(n, loss, rng)
        tau = ComplexF64(tau)
        prev = solve(H, vprev, tau, Any[], reshape(vprev, n, 1), allow)   # the previous round
        @test prev.closed && prev.basis[1] ≈ vprev                         # input kept first
        E = randn(rng, ComplexF64, n, n)
        E ./= opnorm(E)
        for eps in (0.0, 1e-3, 1e-1, 1.0), dH in (0.0, 1e-6, 1e-3)
            z = randn(rng, ComplexF64, n)
            v0 = vprev .+ eps .* z ./ norm(z)
            Hn = H .+ dH * opnorm(H) .* E                                  # this round's generator
            Q0 = Matrix(qr(hcat(v0 / norm(v0), prev.Q)).Q)                 # committed frames
            f = solve(Hn, v0, tau, Any[], Q0, allow)
            r = solve(Hn, v0, tau, prev.basis, Q0, allow)
            @info "R6 n=$n loss=$loss eps=$eps dH=$dH: fresh H1 $(f.n_h1) added $(f.added); " *
                  "complement H1 $(r.n_h1) added $(r.added) err $(r.err) (reused $(r.rec_closed)," *
                  " overlap $(r.rec_overlap), defect $(r.rec_defect))"
            @test r.closed && !r.capped                 # never runs to the depth bound
            @test r.err <= 10 * allow                   # the reused part is tested, never trusted
            @test r.n_h1 <= f.n_h1 + r.n_recycled       # one image per recycled vector at most
            @test r.basis[1] ≈ v0 / norm(v0)            # the next round's complement is of this
            r.rec_closed || @test r.added == f.added    # rejected: exactly the fresh solve
            if eps == 0 && dH == 0                      # the same input, the same generator
                @test r.rec_closed && r.n_h1 == r.n_recycled && r.added == 0
            end
            dH >= 1e-6 && @test !r.rec_closed           # a moved generator is caught
        end
    end
end

end # recycled local Krylov
