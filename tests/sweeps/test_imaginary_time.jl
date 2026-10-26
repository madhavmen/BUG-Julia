# tests/sweeps/test_imaginary_time.jl -- IT-U1..IT-U6 of the imaginary-time / open-system plan
# (cbe_bug_imaginary_open_benchmark_research (1).md, Task 1): normalised imaginary-time CBE-BUG.
#
#   IT-U1  the panel and uniform grids: endpoints, no gaps/duplicates, dyadic panels, early
#          resolution, reproducibility
#   IT-U2  root integrators: stability functions (A-/L-stability, classical order) and the
#          production Lanczos/Arnoldi root against the dense R(tau (M - lambda_min)) on tensors
#   IT-U3  GLOBAL order of BUG p = 1,2,3 against dense exp(-beta H), compression off; the root's
#          order caps it (trapezoid)
#   IT-U4  energy decreases in an uncompressed exponential step; a compression's energy change is
#          bounded by its measured discarded norm
#   IT-U5  total Sz, the old state in the root space, root-space energy = MPS energy, and the
#          outside-Galerkin defect: separate from the root residual, shrinking with dbeta, and
#          triggering re-enrichment / rejection when it fails
#   IT-U6  fail safely: a binding rank, an unmeetable root tolerance -> flagged or thrown, never
#          a silent success
#
# Small chains (L = 8, 10) against dense references only.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))

"Phase-aligned distance of two vectors after normalising both."
function _it_dist(a, b)
    a, b = a / norm(a), b / norm(b)
    ov = dot(b, a)
    return norm(a - b * (abs(ov) > 0 ? ov / abs(ov) : 1))
end

"`n` orthonormal tensors sharing `seed`'s charge blocks (as `test_adaptive_expv.jl`)."
function _it_onb(seed, n; rng = Random.MersenneTwister(5))
    out = Any[]
    for _ in 1:n
        Y = RC._randomize_like(seed, rng)
        for _pass in 1:2, u in out
            Y = to_concrete(Y - tensor_inner(u, Y) * u)
        end
        push!(out, to_concrete((1.0 / norm(Y)) * Y))
    end
    return out
end

_it_coords(y, e) = [tensor_inner(u, y) for u in e]

@testset "imaginary-time CBE-BUG (plan Task 1)" begin

@testset "IT-U1: grids" begin
    for (bmax, P, k) in ((8.0, 5, 4), (1.0, 1, 4), (30.0, 7, 2))
        s = log_panel_steps(bmax, P; per_panel = k)
        g = beta_grid(s)
        @test length(s) == P * k
        @test all(>(0), s)
        @test g[1] == 0.0
        @test g[end] ≈ bmax rtol = 1e-14
        @test all(diff(g) .> 0)                                     # no gaps, no duplicates
        panels = [s[(j - 1) * k + 1:j * k] for j in 1:P]
        @test all(all(x -> x ≈ p[1], p) for p in panels)            # uniform inside a panel
        widths = [sum(p) for p in panels]
        P >= 2 && @test widths[2] ≈ widths[1]                       # first two panels equal
        P >= 3 && @test all(widths[j + 1] ≈ 2 * widths[j] for j in 2:(P - 1))   # then doubling
        @test s[1] ≈ bmax / (2.0^(P - 1) * k)                       # early resolution
        @test maximum(s) ≈ (P == 1 ? bmax : bmax / 2) / k           # coarsest step: per_panel
        @test log_panel_steps(bmax, P; per_panel = k) == s          # reproducible
    end
    u = uniform_beta_steps(2.0, 8)
    @test all(==(0.25), u) && beta_grid(u)[end] ≈ 2.0
    @test_throws ArgumentError log_panel_steps(1.0, 0)
    @test_throws ArgumentError log_panel_steps(-1.0, 3)
    @test_throws ArgumentError uniform_beta_steps(1.0, 0)
end

@testset "IT-U2: root integrators" begin
    @testset "stability functions: A-stability, L-stability, classical order" begin
        zs = vcat(-10.0 .^ range(-3, 8; length = 60), 0.0)
        for m in (:trapezoid, :radau2, :radau3)
            @test all(abs(root_stability(m, z)) <= 1 + 1e-14 for z in zs)          # z <= 0
            @test all(abs(root_stability(m, 1im * y)) <= 1 + 1e-12 for y in -50:0.5:50)
            q = root_order(m)
            e(z) = abs(root_stability(m, z) - exp(z))
            @test log2(e(-0.1) / e(-0.05)) ≈ q + 1 atol = 0.25                      # order q
        end
        @test abs(root_stability(:radau2, -1e8)) < 1e-6                            # L-stable
        @test abs(root_stability(:radau3, -1e8)) < 1e-6
        @test abs(root_stability(:trapezoid, -1e8) + 1) < 1e-6                     # A- not L-
        @test root_stability(:exp, -0.3) ≈ exp(-0.3)
        @test_throws ArgumentError root_order(:euler)
        # stiff scalar y' = -lambda y over h: the implicit roots damp, never amplify
        for m in (:trapezoid, :radau2, :radau3), lh in (0.1, 10.0, 1e4)
            @test abs(root_stability(m, -lh)) < 1
        end
    end

    @testset "matrix-free root on tensors = dense R(tau (M - lambda_min)) e1" begin
        kl = _xx_kl(:left; steps = 2)
        n = 6
        e = _it_onb(kl.seed, n)
        rng = Random.MersenneTwister(11)
        A = randn(rng, ComplexF64, n, n)
        M = (A + A') / 2
        terms = [(i == j ? M[i, i] / 2 : M[i, j], e[i], e[j]) for i in 1:n for j in i:n]
        H = _hop(terms)
        lmin = minimum(eigvals(Hermitian(M)))
        e1 = zeros(ComplexF64, n); e1[1] = 1
        for tau in (-0.3, -5.0, -200.0), m in (:trapezoid, :radau2, :radau3)
            # the Krylov space closes at n, so the Ritz shift IS lambda_min and the root is exact
            want = BondUpdateBUG._rational_apply(m, ComplexF64(tau) * (M - lmin * I), e1)
            for herm in (true, false)
                rep = Ref{Any}(nothing)
                y = expv(H, ComplexF64(tau), e[1]; hermitian = herm, maxiter = 30, tol = 1e-15,
                         reorth = true, root_method = m, report = rep)
                @test norm(_it_coords(y, e) - want) < 1e-10 * max(norm(want), 1)
                @test rep[].substeps == 1                 # an implicit root is one stage
            end
        end
        # :exp is still the exponential (shift trigger off at small tau)
        y = expv(H, ComplexF64(-0.3), e[1]; maxiter = 30, tol = 1e-15, reorth = true)
        @test norm(_it_coords(y, e) - exp(-0.3 * M)[:, 1]) < 1e-10
        @test_throws ArgumentError expv(H, ComplexF64(-0.3), e[1]; root_method = :euler)
    end
end

set_symmetry!(:U1)
_IT_L = 10
_IT_MPO = xxz_mpo(_IT_L; J = 1.0, delta = 1.0)
_IT_HD = dense_heisenberg(_IT_L)
_IT_HNORM = opnorm(_IT_HD)
_IT_TIGHT = (tol_basis = 1e-12, tol_trunc = 1e-14, tol_root = 1e-14, maxdim = 256,
                   outside_residual = false, retries = 0)

@testset "IT-U3: global order against dense exp(-beta H), compression off" begin
    # ⛔ THE START STATE DECIDES WHETHER THE TEST CAN SEE THE ORDER. From Neel at L = 10 the
    # half-sweeps' Krylov enrichment at tol_basis 1e-12 already holds the moments the closure would
    # add: p = 1 and p = 2 came out BIT-IDENTICAL with slope 3, p = 3 with slope 6 (s29test
    # 17340094) -- consistent with "order >= p" and unable to tell p apart. The domain wall (the
    # real-time UT3 start) spreads a front the local enrichment cannot pre-empt. Neel is kept as
    # a logged comparison.
    beta = 0.4
    ns = (2, 4, 8)
    function run(psi0, p, m)
        v0 = dense_state(psi0)
        vref = exp(-beta * _IT_HD) * v0
        errs = Float64[]
        for n in ns
            psi = copy(psi0)
            hist = ite_evolve!(psi, _IT_MPO, uniform_beta_steps(beta, n); order = p,
                               root_method = m, exact = true, _IT_TIGHT...)
            @test all(r.status === :met for r in hist)
            push!(errs, _it_dist(dense_state(psi), vref))
        end
        return errs, log2.(errs[1:(end - 1)] ./ errs[2:end])
    end
    errs1 = nothing
    for (p, m, want) in ((1, :exp, 1), (2, :exp, 2), (3, :exp, 3), (4, :exp, 4),
                         (3, :radau2, 3), (3, :trapezoid, 2), (4, :radau3, 4))
        errs, slopes = run(domain_wall_state(_IT_L), p, m)
        @info "IT-U3 domain wall order $p root $m: errors $(errs), slopes $(slopes) (want $want)"
        p == 1 && (errs1 = errs; @test errs[end] > 1e-10)            # non-vacuous
        @test errs[end] < 1e-11 || slopes[end] >= want - 0.4
        m === :trapezoid && @test slopes[end] <= want + 0.5          # the root caps the order
        # the closure is what buys the accuracy: BUG2 must beat BUG1 at the same grid
        (p, m) == (2, :exp) && @test errs[end] < errs1[end] / 3
    end
    for p in (1, 2, 3)
        errs, slopes = run(neel_state(_IT_L), p, :exp)
        @info "IT-U3 Neel (logged only) order $p: errors $(errs), slopes $(slopes)"
    end
end

@testset "IT-U4: energy in the root solve and in the compression" begin
    psi = neel_state(_IT_L)
    es = Float64[]
    for _ in 1:6
        r = ite_step!(psi, _IT_MPO, 0.1; order = 2, _IT_TIGHT...)
        @test r.status === :met
        @test r.e_root <= r.e_pre_root + 1e-12                      # exact exp: never up
        @test abs(r.de_compression) < 1e-9                          # nothing was compressed
        push!(es, r.e_after)
    end
    @test issorted(es; rev = true)
    # A BINDING compression: its energy change is bounded by what it discarded.
    r = ite_step!(psi, _IT_MPO, 0.2; order = 2, tol_basis = 1e-10, tol_trunc = 1e-14,
                  tol_root = 1e-14, maxdim = 6, outside_residual = false, retries = 0)
    @test r.discarded > 1e-8                                        # non-vacuous
    @test abs(r.de_compression) <= 2 * sqrt(2) * _IT_HNORM * r.discarded * 1.01
    @info "IT-U4 compression: dE $(r.de_compression), discarded $(r.discarded), bound $(2 * sqrt(2) * _IT_HNORM * r.discarded)"
end

@testset "IT-U5: symmetry, old state, root energies, outside-Galerkin defect" begin
    psi = neel_state(_IT_L)
    r = ite_step!(psi, _IT_MPO, 0.1; order = 2, tol_basis = 1e-6, tol_trunc = 1e-10,
                  tol_root = 1e-12, maxdim = 256, outside_residual = true, retries = 0)
    a = r.attempts[1]
    @test abs(total_sz(psi)) < 1e-12                                # strong U(1) kept
    @test a.info.ledger.proj_err < 1e-7                              # old state in the root space
    @test abs(r.e_before - r.e_pre_root) < 1e-10                    # <S0|H0|S0> = <psi|H|psi>
    @test isfinite(r.resid) && r.resid >= 0
    @test r.root_err <= 1e-12 * (1 + 1e-8)                           # projected residual: tiny

    # the outside defect is the BASIS error, not the root solve's: it shrinks with dbeta at a
    # converged root (order 1: dbeta * ||(I-P) H psi_1|| ~ dbeta^2). Measured on the basis the
    # step built (`resid_initial`): a defect over tol_basis is re-enriched to within it, and that
    # controlled value no longer scales with dbeta (s49test 17347098: 1.0e-4 vs 9.9e-5).
    rs = Float64[]
    for db in (0.2, 0.05)
        q = ite_step!(neel_state(_IT_L), _IT_MPO, db; order = 1, tol_basis = 1e-3,
                      tol_trunc = 1e-12, tol_root = 1e-14, maxdim = 256, retries = 0)
        push!(rs, q.attempts[1].resid_initial)
    end
    @info "IT-U5 outside defect at dbeta 0.2 / 0.05: $(rs)"
    @test rs[1] > 4 * rs[2] > 0

    # a coarse step whose defect fails tol_basis is re-enriched on that defect (frames grown by
    # the evolved state's outside directions, root re-solved), then split -- never accepted as
    # it stood
    psi = neel_state(_IT_L)
    q = ite_step!(psi, _IT_MPO, 0.5; order = 2, tol_basis = 1e-7, tol_trunc = 1e-12,
                  tol_root = 1e-14, maxdim = 256, retries = 2)
    a1 = q.attempts[1]
    @info "IT-U5 coarse step: status $(q.status), attempts $(length(q.attempts)), first unmet $(a1.unmet), resid $(a1.resid_initial) -> $(a1.resid) in $(a1.outside_passes) passes (+$(a1.outside_added)), nsub $(q.nsub)"
    @test a1.resid_initial > 1e-7                                 # non-vacuous
    @test a1.outside_passes >= 1                                  # re-enriched on the defect
    a1.outside_added > 0 && @test a1.resid < a1.resid_initial     # ... and it helped
    @test a1.ok || q.nsub > 1                                     # else split
    q.status === :failed || @test q.resid <= 1e-7 * (1 + 1e-8)
    @test abs(total_sz(psi)) < 1e-12
    # a defect already within tol_basis is not re-enriched
    q = ite_step!(neel_state(_IT_L), _IT_MPO, 0.1; order = 2, tol_basis = 1e-3,
                  tol_trunc = 1e-12, tol_root = 1e-14, maxdim = 256, retries = 0)
    q.attempts[1].resid_initial <= 1e-3 && @test q.attempts[1].outside_passes == 0
end

@testset "IT-U6: fail safely" begin
    # insufficient working rank: the compression cannot meet tol_trunc at maxdim = 2
    psi = neel_state(_IT_L)
    ite_step!(psi, _IT_MPO, 0.1; order = 1, maxdim = 32, outside_residual = false)  # grow first
    q = ite_step!(psi, _IT_MPO, 0.3; order = 1, tol_trunc = 1e-10, maxdim = 2,
                  outside_residual = false, retries = 1)
    @test q.status === :failed
    @test :trunc in q.unmet
    @test_throws ErrorException ite_step!(copy(psi), _IT_MPO, 0.3; order = 1, tol_trunc = 1e-10,
                                          maxdim = 2, outside_residual = false, retries = 0,
                                          on_fail = :error)
    # a root tolerance the solve cannot reach: an implicit root has one stage and no substeps.
    # It needs a root wider than the Krylov space: since the DD-QR CBE (d717376) Neel's order-1
    # root is 4-dimensional, Krylov exhausts it and 1e-30 is met honestly (err 3e-68, u6bis). A
    # warmed state's root is not exhaustible; Neel stays as the control that is NOT flagged.
    warm = neel_state(_IT_L)
    for _ in 1:3
        ite_step!(warm, _IT_MPO, 0.2; order = 2, maxdim = 64, outside_residual = false)
    end
    q = ite_step!(warm, _IT_MPO, 3.0; order = 1, root_method = :radau2,
                  tol_root = 1e-30, outside_residual = false, retries = 0)
    @test q.status === :failed
    @test :expv in q.unmet
    qn = ite_step!(neel_state(_IT_L), _IT_MPO, 3.0; order = 1, root_method = :radau2,
                   tol_root = 1e-30, outside_residual = false, retries = 0)
    @info "IT-U6 root tolerance 1e-30: warmed $(q.status) $(q.unmet) root_err $(q.root_err); Neel control $(qn.status) $(qn.unmet) root_err $(qn.root_err)"
    @test !(:expv in qn.unmet)                     # control: an exhausted root meets any tol_root
    # never silent: an accepted record carries no reasons, a failed one carries some
    for rec in (q, ite_step!(neel_state(_IT_L), _IT_MPO, 0.1; order = 2, retries = 0))
        rec.status === :failed ? @test(!isempty(rec.unmet)) : @test(isempty(rec.unmet))
    end
    @test_throws ArgumentError ite_step!(neel_state(_IT_L), _IT_MPO, 0.1; root_method = :euler)
    @test_throws ArgumentError ite_step!(neel_state(_IT_L), _IT_MPO, -0.1)
end

@testset "physics: panel and uniform grids follow dense exp(-beta H) to the ground state" begin
    E0 = minimum(eigvals(Hermitian(Matrix(_IT_HD))))
    beta = 16.0
    v = exp(-beta * _IT_HD) * dense_state(neel_state(_IT_L))
    Ebeta = real(dot(v, _IT_HD * v)) / real(dot(v, v))     # the exact energy AT beta, not E0
    for (name, steps) in (("panel", log_panel_steps(beta, 6; per_panel = 4)),
                          ("uniform", uniform_beta_steps(beta, 64)))
        psi = neel_state(_IT_L)
        hist = ite_evolve!(psi, _IT_MPO, steps; order = 2, tol_basis = 1e-6, tol_trunc = 1e-9,
                           tol_root = 1e-10, maxdim = 64, retries = 3)
        bad = [(beta = r.beta, dbeta = r.dbeta, unmet = r.unmet, resid = r.resid, nsub = r.nsub)
               for r in hist if r.status === :failed]
        @info "IT physics $name: $(length(steps)) steps, E = $(hist[end].e_after), exact at beta $Ebeta, E0 $E0, retried $(count(r -> r.status === :retried, hist)), failed $(bad)"
        # ⛔ NO STEP MAY FAIL, THE FIRST ONE INCLUDED. The first step from a product state used to be
        # flagged and called "the fail-safe working" (s30test 17340117: 2.1e-6 panel, 1.6e-5
        # uniform after 4-5 sub-steps). It was the fixed halving: the defect does shrink like
        # dbeta^(p+1), only from a large constant (fp1, job 17342386), and the split is now
        # sized from it (`_ite_nsplit`). USER 2026-10-18: "this hints at a bug, fix it".
        @test isempty(bad)
        @test hist[end].e_after ≈ Ebeta atol = 1e-7
        @test hist[end].e_after >= E0 - 1e-8                         # variational
    end
end

@testset "first step from a product state: the split is sized from the defect" begin
    # Neel and the domain wall, order 2, a tight tol_basis: two halvings cannot reach it (the old
    # ladder flagged this step); the in-step re-enrichment and/or a defect-sized split do
    for (name, st) in (("neel", neel_state(_IT_L)), ("dw", domain_wall_state(_IT_L)))
        psi = st
        q = ite_step!(psi, _IT_MPO, 0.125; order = 2, tol_basis = 1e-7, tol_trunc = 1e-10,
                      tol_root = 1e-12, maxdim = 256, retries = 2)
        a1 = q.attempts[1]
        @info "first step $name: status $(q.status), first resid $(a1.resid_initial) -> $(a1.resid) ($(a1.outside_passes) re-enrich passes), nsub $(q.nsub), final resid $(q.resid)"
        @test a1.resid_initial > 1e-7                              # non-vacuous: it IS over
        # the split is sized from the defect: more than two halvings give
        @test RC._ite_nsplit((unmet = [:outside], resid = a1.resid_initial), 1e-7, 2) > 4
        @test q.status !== :failed
        @test q.resid <= 1e-7 * (1 + 1e-8)
    end
    # a RESOURCE cap is not split: the binding maxdim stays binding at any dbeta
    psi = neel_state(_IT_L)
    ite_step!(psi, _IT_MPO, 0.1; order = 1, maxdim = 32, outside_residual = false)
    q = ite_step!(psi, _IT_MPO, 0.3; order = 1, tol_trunc = 1e-10, maxdim = 2,
                  outside_residual = false, retries = 3)
    @test q.status === :failed && q.nsub == 1 && length(q.attempts) == 1
end

@testset "Krylov weight in imaginary time: relative to the kept state, not a decaying frontier" begin
    # one Krylov step, [tau*h 0; tau*b 0]: the direction's weight in exp(tau*H) e1 is
    # |tau b phi_1(tau h)| = b |1 - e^{tau h}| / |h| in the Eshift frame
    Hm = zeros(ComplexF64, 3, 3)
    Hm[2, 1] = 0.5
    Hm[1, 1] = 3.0                       # h = 3 above Eshift: the seed DECAYS by e^{-24}
    Ex = exp(ComplexF64[-24 0; -4 0])
    # a closure FRONTIER: the absolute weight, not that inflated by its own decay (the d49 flags
    # at dbeta 8: 1e8 to 3e15)
    wf = RC._kry_weight(Hm, 1, ComplexF64(-8); state_seed = false)
    @test wf ≈ 0.5 * (1 - exp(-24)) / 3 rtol = 1e-10
    @test wf ≈ abs(Ex[2, 1]) rtol = 1e-10
    # the state's own tensor (half-sweeps): relative to its growth, unchanged
    @test RC._kry_weight(Hm, 1, ComplexF64(-8)) ≈ abs(Ex[2, 1]) / abs(Ex[1, 1]) rtol = 1e-10
    @test RC._kry_weight(Hm, 1, ComplexF64(-8)) > 1e9 * wf          # what the frontier used to get
    # real time: one expression for both, unchanged
    @test RC._kry_weight(Hm, 1, ComplexF64(-0.1im); state_seed = false) ==
          RC._kry_weight(Hm, 1, ComplexF64(-0.1im)) ==
          abs(exp(ComplexF64[-0.3im 0; -0.05im 0])[2, 1])
    # No end-to-end case here: at L = 10 the old weight flags nothing (s45 control, order 3,
    # panel:9:4 to beta 64), so it could not discriminate. The L = 16 reproducer is that check.
end

@testset "tdvp2 partner: eshift and the backward-substep amplification" begin
    tkw = (maxdim = 64, maxiter = 30, conv_tol = 1e-14, substeps = 0, weight_tol = 1e-12)
    en(p) = real(mpo_energy(p, _IT_MPO)) / real(overlap(p, p))
    nrm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)
    # the shift changes only the normalisation, so the renormalised states agree
    a, b = neel_state(_IT_L), neel_state(_IT_L)
    for _ in 1:3
        tdvp2_step!(a, _IT_MPO, ComplexF64(-0.2); tkw...); nrm!(a)
        tdvp2_step!(b, _IT_MPO, ComplexF64(-0.2); eshift = en(b), tkw...); nrm!(b)
    end
    @test _it_dist(dense_state(a), dense_state(b)) < 1e-9
    # the backward stage amplifies more at a larger step (the plan's stability metric)
    amps = Float64[]
    for db in (0.05, 1.0)
        rep = Ref{Any}(nothing)
        tdvp2_step!(copy(b), _IT_MPO, ComplexF64(-db); eshift = en(b), backward_report = rep,
                    tkw...)
        push!(amps, rep[])
    end
    @info "tdvp2 backward log-amplification at dbeta 0.05 / 1.0: $(amps)"
    @test all(isfinite, amps)
    @test amps[2] > amps[1]
    # a step of dbeta |E| ~ 400: the shifted step stays representable and variational
    E0 = minimum(eigvals(Hermitian(Matrix(_IT_HD))))
    tdvp2_step!(b, _IT_MPO, ComplexF64(-100.0); eshift = en(b), tkw...)
    @test isfinite(norm(b))
    nrm!(b)
    @test en(b) >= E0 - 1e-8
    @test en(b) < en(a)
end

end # imaginary-time CBE-BUG
