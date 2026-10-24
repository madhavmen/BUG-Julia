# tests/sweeps/test_dissipative_t2.jl -- dissipative_cbe_bug_implementation_plan.md, Task 2, the
# G0 gate (U01-U05, U08, U12, U17), on the no-jump generator G = -iH - Q/2 run as the MPO
# H - (i/2)Q at tau = -im*dt.
#
#   U01  G + G' = -Q for the pair-loss and collective-emission jump sets, Q >= 0, Q = Σ L'L; the
#        right sketch is the conjugate transpose of the materialised lossy two-site action (U03)
#   U02  the d = 4 Hubbard G (fermion signs, pair loss) equals the Fock-space G up to basis phases,
#        also under U(1) x U(1), and a Hermitian sign error does not; the collective-emission G
#        equals the dense one with complex phases, the conjugated phases do not; its virtual
#        dimension does not grow with L
#   U03  streamed DD sketch == explicit P^D X P^D Om on lossy bonds, both orientations; two-pass
#        thin QR (Q'Q = I, orthogonal to the kept frame, range kept whole); no SVD in the expansion
#   U04  on G: the DD range error follows the share, holdout false acceptances over seeds, an
#        unreachable share ends with an explicit limit and bounded work, and the order-p frontier
#        sweeps carry a magnon to sites 1..p+1 (the delayed G^2 psi, G^3 psi directions)
#   U05  the root on a nonnormal G: Arnoldi expv error <= tol_root over a ladder, Radau IIA(2) +
#        GMRES true stage residual <= tol_root and one-step error O(h^4) against the exponential
#   U08  P psi_n = psi_n in both gauges, W'W = Z'Z = I, a lossless root split and compression, and
#        a dt = 0 step that reconstructs psi_n
#   U12  tol_trunc bounds the accumulated discarded norm of a decayed state; a binding maxdim is
#        :rank_limited and :budget_unmet; U(1) sector ranks sum to the bond dimension
#   U17  every phase measured in seconds AND bytes; kernel flops linear in the probe width (no
#        dense two-site action), no H*Theta leak, a matrix-free root generator

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
include(joinpath(@__DIR__, "..", "common", "krylov_closure_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dissipative_fixtures.jl"))

"`||a - b|| / ||a||` from overlaps -- no dense state."
_reldist(a, b) = sqrt(max(norm(a)^2 + norm(b)^2 - 2 * real(overlap(a, b)), 0.0)) / norm(a)

@testset "dissipative Task 2 (G0): U01-U05, U08, U12, U17" begin

@testset "U01/U02: Hubbard pair loss, the d = 4 MPO against the Fock space" begin
    L, U, kappa = 3, 4.0, [1.0, 0.3, 2.0]
    D = 4^L
    Hf, Qf, jumps = _hub_fock(L; U = U, kappa = kappa)
    Gf = Hf - 0.5im * Qf
    # (c↓c↑)'(c↓c↑) = n↑n↓: the jumps' signs cancel in Q, which is the doublon count
    ndbl = [sum(kappa[j] * hub_occ(b, hub_mode(j, 0)) * hub_occ(b, hub_mode(j, 1)) for j in 1:L)
            for b in 0:(D - 1)]
    @test norm(Qf - Diagonal(ndbl)) < 1e-14 && maximum(ndbl) > 0
    @test count(!iszero, jumps[1]) > 0
    # :none first: plain 4x4 blocks, nothing but the algebra between the MPO and the reference
    set_symmetry!(:none)
    Gm = no_jump_mpo(hubbard_aa_mpo_d4(L; J = 1.0, U = U), doublon_loss_mpo(L; kappa = kappa))
    Qm = doublon_loss_mpo(L; kappa = kappa)
    states = [_hub4_state(b, L) for b in 0:(D - 1)]
    MG, MQ = _mpo_matrix(Gm, states), _mpo_matrix(Qm, states)
    G = -1im * MG                                   # dψ/dt = Gψ, G = -i(H - iQ/2)
    @test norm(G + G' + MQ) < 1e-12 * norm(MQ)      # U01: G + G' = -Q
    @test eigmin(Hermitian(MQ)) > -1e-12            # U01: Q >= 0
    @test norm(MQ - Qf) < 1e-12 * norm(Qf)          # U01: Q = Σ L'L (diagonal: no basis phase)
    @test _gauge_residual(MG, Gf) < 1e-12           # U02: G up to the basis' state phases
    @test count(!iszero, MG - Diagonal(MG)) > 0     # the hops are there
    # CONTROL: an up hop with an extra (-1)^{n_j↓}. Hermitian, number conserving, and not a basis
    # convention (the two orders of moving a doublon one site differ by a sign), so it must fail.
    Hw, _, _ = _hub_fock(L; U = U, kappa = kappa, wrong = true)
    @test _gauge_residual(MG, Hw - 0.5im * Qf) > 1e-3
    # The same operator under U(1)_charge x U(1)_spin, i.e. U(1)↑ x U(1)↓: matrix elements only
    # inside a (N↑, N↓) sector, which is all the generator has.
    set_symmetry!(:U1)
    Gu = no_jump_mpo(hubbard_aa_mpo_d4(L; J = 1.0, U = U), doublon_loss_mpo(L; kappa = kappa))
    su = [_hub4_state(b, L) for b in 0:(D - 1)]
    MGu = _mpo_matrix(Gu, su; skip = (a, b) -> _hub_charges(a - 1, L) != _hub_charges(b - 1, L))
    @test _gauge_residual(MGu, Gf) < 1e-12
end

@testset "U01/U02: collective emission, complex phases, compact MPO" begin
    set_symmetry!(:U1)
    L, gp, gm, k, Jz = 5, 1.0, 0.4, π / 3, 1.0
    h = [0.2 * (-1)^j for j in 1:L]
    Hm = field_mpo_from_terms(xxz_chain(L; J = 1.0, delta = Jz), h)
    Qm = collective_emission_mpo(L; gamma_plus = gp, gamma_minus = gm, k = k)
    Gm = no_jump_mpo(Hm, Qm)
    Lp, Lm = _emission_jumps(L; gp = gp, gm = gm, k = k)
    Qd = Lp' * Lp + Lm' * Lm
    Hd = dense_heisenberg(L; J = 1.0, delta = Jz) + sum(h[j] * _embed(_SZ, j, L) for j in 1:L)
    @test norm(imag(Qd)) > 0.1 * norm(Qd)                # Γ₊ ≠ Γ₋: genuinely complex couplings
    states = _spin_states(L)
    skip(a, b) = _spin_charge(a - 1, L) != _spin_charge(b - 1, L)
    MQ, MG = _mpo_matrix(Qm, states; skip = skip), _mpo_matrix(Gm, states; skip = skip)
    @test norm(MQ - Qd) < 1e-12 * norm(Qd)                # U02: the dense L₊'L₊ + L₋'L₋
    @test norm(MG - (Hd - 0.5im * Qd)) < 1e-12 * norm(Qd)
    G = -1im * MG
    @test norm(G + G' + MQ) < 1e-12 * norm(MQ)            # U01
    @test eigmin(Hermitian(MQ)) > -1e-12
    # CONTROL: the conjugated phases are Q transposed, a different operator at Γ₊ ≠ Γ₋
    Lp2, Lm2 = _emission_jumps(L; gp = gp, gm = gm, k = -k)
    @test norm(MQ - (Lp2' * Lp2 + Lm2' * Lm2)) > 1e-3 * norm(Qd)
    # compact: one channel per (hop direction, phase), not one per opening site
    @test maximum(mpo_virtual_dims(Qm)) <= 6
    @test maximum(mpo_virtual_dims(collective_emission_mpo(40; gamma_plus = gp,
                                                           gamma_minus = gm, k = k))) <= 6
end

@testset "U03: streamed DD sketch == explicit on a lossy bond, bond $i" for i in (1, 4, 7)
    G, psi, f, lenv, renv, H = _lossy_bond(; i = i)
    i == 4 && @test _unsaturated(f)
    rng = Random.MersenneTwister(11 + i)
    out = Ref{Any}(nothing)
    GT = apply_h_two_site(frame_theta(f), G, i, lenv, renv)
    lH, rH = _envs(psi, H, i, length(psi))
    @test _rel(GT, apply_h_two_site(frame_theta(f), H, i, lH, rH)) > 1e-3   # the loss acts here
    # K-sweep orientation: probe on the right
    Om = sector_graded_sketch(f.V0, :right, 6; comp_ratio = nothing, rng = rng)
    OmD = perp_component_right(f.V0, Om)
    apply_two_site_sketch!(out, OmD, TwoSiteSketchOp(f, G, i, lenv, renv))
    @test _rel(out[], to_concrete(contract(GT, (3, 4), OmD', (2, 3)))) < 1e-12
    @test _rel(perp_component(f.U0, out[]), dd_reference_sketch(f, G, i, lenv, renv, Om, :left)) < 1e-12
    # L-sweep orientation: the rows Om' X of the NON-Hermitian action, i.e. (X' Om)' (U01's X')
    OmL = sector_graded_sketch(f.U0, :left, 6; comp_ratio = nothing, rng = rng)
    OmLD = perp_component(f.U0, OmL)
    apply_two_site_sketch!(out, OmLD, TwoSiteSketchOp(f, G, i, lenv, renv; adjoint = true))
    raw = to_concrete(permutedims(contract(GT, (1, 2), OmLD', (1, 2)), (3, 1, 2)))
    @test _rel(out[], raw) < 1e-12
    @test _rel(perp_component_right(f.V0, out[]),
               dd_reference_sketch(f, G, i, lenv, renv, OmL, :right)) < 1e-12
end

@testset "U03: two-pass thin QR of the lossy sketch, both sides; no SVD" begin
    G, psi, f, lenv, renv, _ = _lossy_bond()
    i = 4
    rng = Random.MersenneTwister(5)
    out = Ref{Any}(nothing)
    for side in (:left, :right)
        left = side === :left
        frame, opp = left ? (f.U0, f.V0) : (f.V0, f.U0)
        Om = RC._dd_probe(opp, left ? :right : :left, 8; rng = rng)
        apply_two_site_sketch!(out, Om, TwoSiteSketchOp(f, G, i, lenv, renv; adjoint = !left))
        # charge-pure by construction: the QR runs per symmetry sector of the block-sparse tensor
        Q, backend = RC._dd_select(frame, out[], side, KernelCounters())
        @test backend === :householder && Q !== nothing
        k = leg_dim(Q, left ? 3 : 1)
        Gram = left ? to_concrete(contract(Q', (1, 2), Q, (1, 2))) :
                      to_concrete(contract(Q, (2, 3), Q', (2, 3)))
        @test abs(norm(Q)^2 - k) < 1e-10 && abs(norm(Gram)^2 - k) < 1e-10      # Q'Q = I
        ov = left ? contract(frame', (1, 2), Q, (1, 2)) : contract(Q, (2, 3), frame', (2, 3))
        @test norm(to_concrete(ov)) < 1e-12                                     # A'Q = 0
        Y = left ? perp_component(frame, out[]) : perp_component_right(frame, out[])
        miss = left ? perp_component(Q, Y) : perp_component_right(Q, Y)
        @test norm(miss) < 1e-10 * norm(Y)                                      # range kept whole
    end
    for side in (:left, :right)
        dd_cbe_expand(f, G, i, lenv, renv, side)                    # warms the fusion cache
        set_accumul_costs!(true)
        read_reset_costs!()
        dd_cbe_expand(f, G, i, lenv, renv, side; rng = Random.MersenneTwister(2))
        _, svdc = read_reset_costs!()
        set_accumul_costs!(false)
        @test svdc == 0
    end
end

@testset "U04: the DD range error on G follows the share, $side" for side in (:left, :right)
    G, psi, f, lenv, renv, _ = _lossy_wide()
    @test _unsaturated(f)
    errs, ranks = Float64[], Int[]
    for frac in (1e-1, 1e-2, 1e-3, 1e-5)
        c = _dd_closures(f, G, lenv, renv, side; rng = Random.MersenneTwister(31))
        Q, st, _, _ = _ctrl(c, frac * norm(c.G))
        push!(errs, norm(c.perpQ(Q, c.G)) / norm(c.G))
        push!(ranks, _rank(Q, c.fl))
        @test st === :met
        @test errs[end] <= 10 * frac
    end
    @info "U04 DD range error vs share" side errs ranks
    @test ranks[end] >= ranks[1] && errs[end] < errs[1]
end

@testset "U04: holdout false acceptances over seeds, candidates blind to the dominant sector" begin
    G, psi, f, lenv, renv, _ = _lossy_wide()
    c0 = _dd_closures(f, G, lenv, renv, :left; rng = Random.MersenneTwister(1))
    secs = [(q, _sector_norm(c0.G, c0.fl, q, d)) for (q, d) in c0.G.spaces[c0.fl]]
    @test length(secs) >= 2
    qstar = secs[argmax(last.(secs))][1]
    share = 1e-3 * norm(c0.G)
    # CONTROL: holdout blind too -- the dropped sector carries real weight
    c = _dd_closures(f, G, lenv, renv, :left; rng = Random.MersenneTwister(2))
    blind(w, b) = _drop(c.draw(w, b), c.wleg, qstar)
    Q, _, _, _ = _ctrl(c, share; draw = blind, holdout = w -> blind(w, nothing))
    @test norm(c.perpQ(Q, c.G)) > 10 * share
    false_acc = Dict{Int, Int}()
    for s in (2, 4, 8)
        false_acc[s] = 0
        for seed in 1:10
            cs = _dd_closures(f, G, lenv, renv, :left; rng = Random.MersenneTwister(100 + seed))
            Qs, st, _, _ = _ctrl(cs, share; s = s,
                                 draw = (w, b) -> _drop(cs.draw(w, b), cs.wleg, qstar))
            st === :met && norm(cs.perpQ(Qs, cs.G)) > 10 * share && (false_acc[s] += 1)
        end
    end
    @info "U04 false acceptances of 10 (missed dominant sector)" qstar false_acc
    @test false_acc[8] == 0
end

@testset "U04: an unreachable share stops with an explicit limit, bounded work, $side" for side in
        (:left, :right)
    G, psi, f, lenv, renv, _ = _lossy_wide()
    s = RC._DD_HOLDOUT
    c = _dd_closures(f, G, lenv, renv, side; rng = Random.MersenneTwister(41))
    Q, st, lg, _ = _ctrl(c, 1e-300)
    err = norm(c.perpQ(Q, c.G)) / norm(c.G)
    @test st in (:work_limited, :basis_limited) || (st === :met && err <= 1e-12)
    st === :met || @test :sample in lg.unmet && !lg.tolerance_met
    @test c.kc.n_stream_cols < 2 * c.wcap + 3 * s
    c2 = _dd_closures(f, G, lenv, renv, side; rng = Random.MersenneTwister(42))
    _, st2, lg2, _ = _ctrl(c2, 1e-300; wcap = 3)
    @test st2 === :work_limited && lg2.workcap_reached && :sample in lg2.unmet
    @test c2.kc.n_stream_cols < 2 * 3 + 3 * s
end

@testset "U04: the order-p frontiers carry a magnon to sites 1..p+1 under loss" begin
    set_symmetry!(:U1)
    L, c = 6, 3
    _, _, G, _, _ = _lossy_xx(L)
    psi = _kc_magnon(L, 1)
    tau = ComplexF64(-1im)
    held = Dict{Int, Vector{Float64}}()
    for p in 1:3
        fr = RC.frames_from_state(psi, G, c)
        alloc = basis_allocation(AdaptiveBudget(1e-10, 0.0, 1.0), L - 1; tau = tau, nrm = 1.0,
                                 nfront = p)
        cl = RC._krylov_closure!(fr, psi, G, RC._project_root(psi, fr), tau, alloc,
                                 _kc_ckw(alloc, 64); order = p, workcap = 64, nrm = 1.0,
                                 include_zero = true)
        @test cl.converged
        held[p] = [norm(RC._project_root(_kc_magnon(L, k), fr)) for k in 1:L]
        @test all(h -> abs(h - 1) < 1e-8, held[p][1:(p + 1)])   # G^j psi, j <= p, in the space
        p1 = [l for l in cl.log if l.pass == 1]
        @test length(p1) == p && all(l.added > 0 for l in p1)   # each frontier found its level
        @test all(left_isometry_defect(fr.W[i]) < 1e-12 for i in 1:c)
        @test all(right_isometry_defect(fr.Z[j]) < 1e-12 for j in c:(L - 1))
    end
    @info "U04 magnon held by order p (sites 1..L)" held
    @test held[1][3] < 0.5                    # non-vacuous: order 1 does not reach G^2 psi
end

@testset "U05: root solvers on a nonnormal root" begin
    set_symmetry!(:U1)
    L = 6
    _, psi0 = _xx_warm(L, 2)
    _, _, G, Gd, _ = _lossy_xx(L)
    v0 = dense_state(psi0)
    tau = ComplexF64(-0.5im)
    rows = Any[]
    for tolx in (1e-4, 1e-6, 1e-8, 1e-10)
        fr, rr = Ref{Any}(nothing), Ref{Any}(nothing)
        psi = copy(psi0)
        cbe_bug_step!(psi, G, tau; hermitian = false, tol_basis = 1e-8, tol_root = tolx,
                      truncate = false, frames_report = fr, root_report = rr)
        U = _root_basis(psi0, G, fr[]; mmax = 10)
        M = U' * Gd * U
        @test norm(M * M' - M' * M) > 1e-3 * norm(M)^2              # the root is nonnormal
        err = norm(dense_state(psi) - U * (exp(tau * M) * (U' * v0))) / norm(v0)
        @test rr[].solve.err <= tolx * (1 + 1e-8)
        @test err <= 10 * tolx + 1e-12
        push!(rows, (tol = tolx, est = rr[].solve.err, err = err))
    end
    # Radau IIA(2) + GMRES: the TRUE stage residual meets tol_root, and the step is order 3
    rerr = Float64[]
    for h in (0.2, 0.1)
        fr, rr = Ref{Any}(nothing), Ref{Any}(nothing)
        psi = copy(psi0)
        cbe_bug_step!(psi, G, ComplexF64(-im * h); hermitian = false, root_method = :radau2_gmres,
                      tol_basis = 1e-10, tol_root = 1e-12, truncate = false,
                      frames_report = fr, root_report = rr)
        @test rr[].solve.err <= 1e-12 * (1 + 1e-8)
        U = _root_basis(psi0, G, fr[]; mmax = 10)
        M = U' * Gd * U
        push!(rerr, norm(dense_state(psi) - U * (exp(-im * h * M) * (U' * v0))) / norm(v0))
    end
    @info "U05 root ladder and Radau one-step error" rows rerr slope = log2(rerr[1] / rerr[2])
    @test log2(rerr[1] / rerr[2]) > 3.5                # R(z) - e^z = O(z^4)
end

@testset "U08: projection and lossless reconstruction under loss" begin
    G, psi0, _, _, _, _ = _lossy_bond()
    L = length(psi0)
    tau = ComplexF64(-0.05im)
    for centre in (1, L)
        psi = copy(psi0)
        canonical!(psi, centre)
        fr = Ref{Any}(nothing)
        info = cbe_bug_step!(psi, G, tau; hermitian = false, tol_basis = 1e-8, tol_trunc = 1e-12,
                             tol_root = 1e-13, maxdim = 64, truncate = false, frames_report = fr)
        @test info.ledger.proj_err < 1e-12                   # ||(1 - P) psi_n|| / ||psi_n||
        @test info.discarded == 0.0 && info.ledger.discarded == 0.0   # lossless root split
        @test all(left_isometry_defect(fr[].W[i]) < 1e-12 for i in 1:fr[].c)        # W'W = I
        @test all(right_isometry_defect(fr[].Z[j]) < 1e-12 for j in fr[].c:(L - 1))  # Z'Z = I
        @test norm(psi) < norm(psi0) * (1 - 1e-6)           # the lossy step contracts
    end
    z = copy(psi0)
    cbe_bug_step!(z, G, ComplexF64(0); hermitian = false, truncate = false)
    @test _reldist(psi0, z) < 1e-7                           # dt = 0 reconstructs psi_n
    p, q = copy(psi0), copy(psi0)
    cbe_bug_step!(p, G, tau; hermitian = false, truncate = false)
    iq = cbe_bug_step!(q, G, tau; hermitian = false, tol_trunc = 0.0, maxdim = 10_000)
    @test iq.ledger.discarded == 0.0 && !iq.ledger.maxdim_reached
    @test _reldist(p, q) < 1e-7                              # tol_trunc = 0 compresses losslessly
end

@testset "U12: compression and rank limits on a decayed state" begin
    set_symmetry!(:U1)
    L = 10
    _, psi = _xx_warm(L, 4)
    _, _, G, _, _ = _lossy_xx(L)
    cbe_bug_step!(psi, G, ComplexF64(-0.2im); hermitian = false, tol_basis = 1e-8)
    root = L ÷ 2 + 1
    canonical!(psi, root)
    @test maximum(bond_dims(psi)) > 4 && norm(psi) < 0.99    # rank to cut, and loss happened
    for tol in (1e-2, 1e-4)
        p = copy(psi)
        _, worst, cum, capped = truncate_recursive!(p, root; maxdim = 10_000, weight_tol = tol)
        @test !capped && cum <= tol * (1 + 1e-8)
        @test _reldist(psi, p) <= cum + 1e-7
    end
    q = copy(psi)
    info = cbe_bug_step!(q, G, ComplexF64(-0.1im); hermitian = false, tol_basis = 1e-10,
                         tol_trunc = 1e-12, maxdim = 2)
    r = info.record
    @test maximum(bond_dims(q)) <= 2 && r.D_max == 2 && r.D_final <= 2
    @test r.D_stage <= r.D_work
    @test :rank_limited in r.limits && info.ledger.maxdim_reached
    @test budget_status(info.ledger) === :budget_unmet      # no false success at a binding cap
    nsec = Int[]
    for i in 1:(L - 1)
        dims = [d for (_, d) in psi[i].spaces[3]]
        @test sum(dims) == bond_dims(psi)[i]
        push!(nsec, length(dims))
    end
    @test maximum(nsec) >= 2                                  # sector-resolved bonds
end

@testset "U17: every phase timed and sized; no dense two-site action, matrix-free root" begin
    G, psi, f, lenv, renv, _ = _lossy_wide()
    info = cbe_bug_step!(copy(psi), G, ComplexF64(-0.05im); order = 3, hermitian = false,
                         tol_basis = 1e-8, maxdim = 64)
    kc = info.record.counters
    for fld in (:t_input_dd, :t_stream, :t_output_dd, :t_qr, :t_one_site, :t_root)
        @test getfield(kc, fld) > 0
    end
    for fld in (:b_input_dd, :b_stream, :b_output_dd, :b_qr, :b_one_site, :b_root, :b_env,
                :b_compress)
        @test getfield(kc, fld) > 0
    end
    @test info.t_cbe > 0 && info.t_kry > 0 && info.t_root > 0 && info.t_trunc > 0 && info.t_env > 0
    @test info.record.bytes_step > 0 && info.record.peak_rss > 0
    @test info.ledger.n_h2_leak == 0                          # order 3 folds: no H*Theta
    for (fx, i) in ((_lossy_bond, 4), (_lossy_wide, 6))
        Gi, _, fi, li, ri, _ = fx()
        @test _unsaturated(fi)
        for side in (:left, :right)
            fl = _dd_kernel_flops(fi, Gi, i, li, ri, side, 4; rng = Random.MersenneTwister(4))
            for (_, (f1, f2)) in pairs(fl)
                @test f1 > 0
                @test f2 == 2 * f1                   # linear in the probe width: no dense X
            end
        end
    end
    H0 = RC.zero_site_h(G, 6, lenv, renv, f.U0, f.V0)
    @test H0 isa RC.MPOZeroSiteH
    N = RC.tensor_elements(apply_zero_site(H0, f.S0))
    @test RC.tensor_elements(H0.l) + RC.tensor_elements(H0.r) < N^2   # no (R_L R_R)^2 matrix
end

end
