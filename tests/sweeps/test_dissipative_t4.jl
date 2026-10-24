# tests/sweeps/test_dissipative_t4.jl -- dissipative_cbe_bug_implementation_plan.md, Task 4: the
# quantum-jump engine (`bug/quantum_jumps.jl`, the jump MPOs of `models/dissipative.jl`,
# `operators/mpo_apply.jl`).
#
#   J0   the jumps as charged MPOs: the collective L± and the site L_j equal the dense operators on
#        an entangled state (CONTROL: the conjugate phase does not); the pair loss equals the Fock
#        jump in the MPS basis whose state phases H fixes, sign structure included (CONTROL: an
#        extra Jordan-Wigner sign does not); Σ_μ ||L_μ ψ||² = <ψ|Q|ψ> for each model's Q MPO
#   U07  a jump moves N_exc -> N_exc - 1 and (N↑, N↓) -> (N↑-1, N↓-1), and BUG continues in the new
#        sector against ED
#   U13  the jump time solves P0 = u against ED, seeds reproduce, the drawn channels follow
#        p_μ = ||L_μ ψ||² / <Q> over many events (CONTROL: not a uniform draw), the conditional
#        states are normalised, and an integrator that grows the norm is flagged (CONTROL)
#   U14  ensembles against the Lindblad solution within their confidence intervals, occupations and
#        mean jump counts: A2 at L = 4, A1 at L = 2 and 4 (Task 3's trajectory part); CONTROL: the
#        no-jump conditional values are outside them
#
# U15 (convergence under refinement) is test_dissipative_t4_conv.jl.

using Test
using LinearAlgebra
using SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "dd_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dissipative_fixtures.jl"))
include(joinpath(@__DIR__, "..", "..", "benchmarks", "open_system", "sparse_lindblad.jl"))

const RC = BUGJulia.RSVDCBEBondUpdate
# the no-jump step of every trajectory: collective emission needs the validated closure (USER
# 2026-10-24), and it costs nothing on local loss at these sizes
const _T4_STEP = (closure_checks = true, tol_basis = 1e-10, tol_root = 1e-12)
_t4_step!(p, G, h) = cbe_bug_step!(p, G, ComplexF64(-im * h); hermitian = false, order = 3,
                                   tol_trunc = 1e-12, _T4_STEP...)
_sz_obs(L) = p -> [real(site_expval(p, j, local_space().Sz)) for j in 1:L]

"The L = 3 Hubbard fixture of J0/U07: `H`, `G`, Fock `(Hf, Qf, jumps)`, product `states` and the
state phases `s` that `H` fixes (MPS amplitude = s ∘ Fock amplitude)."
function _t4_hub(L, U, kap)
    set_symmetry!(:U1)
    Hf, Qf, jumps = _hub_fock(L; U = U, kappa = kap)
    states = [_hub4_state(b, L) for b in 0:(4^L - 1)]
    Hm = hubbard_aa_mpo_d4(L; J = 1.0, U = U)
    MH = _mpo_matrix(Hm, states; skip = (a, b) -> _hub_charges(a - 1, L) != _hub_charges(b - 1, L))
    s = _gauge_phases(MH, Hf)
    return (; Hm, G = no_jump_mpo(Hm, doublon_loss_mpo(L; kappa = kap)), Hf, Qf, jumps, states,
            s, res = norm(MH - Diagonal(s) * Hf * Diagonal(s)') / norm(Hf))
end

@testset "dissipative Task 4: the quantum-jump engine (J0, U07, U13, U14)" begin

@testset "J0: collective and site jumps against the dense operators and Q" begin
    set_symmetry!(:U1)
    L, gp, gm, k = 5, 1.0, 0.4, π / 3
    _, psi = _xx_warm(L, 2)
    v = dense_state(psi)
    n = only(_spin_support(v, L))
    Lp, Lm = _emission_jumps(L; gp = gp, gm = gm, k = k)
    ch = collective_emission_channels(L; gamma_plus = gp, gamma_minus = gm, k = k)
    @test [c.name for c in ch] == ["L+", "L-"]
    for (c, Ld) in zip(ch, (Lp, Lm))
        w = Ld * v
        @test norm(dense_state(factor_product(c.factors[1], psi)) - w) < 1e-12 * norm(w)
        @test abs(jump_weight(c, psi) - norm(w)^2) < 1e-12 * norm(w)^2
        phi, disc = apply_jump(c, psi)
        @test disc < 1e-14 && norm(dense_state(phi) - w) < 1e-10 * norm(w)
        @test _spin_support(dense_state(phi), L) == Set([n - 1])     # N_exc -> N_exc - 1
    end
    Lp2, _ = _emission_jumps(L; gp = gp, gm = gm, k = -k)
    @test norm(dense_state(factor_product(ch[1].factors[1], psi)) - Lp2 * v) > 1e-3 * norm(Lp * v)  # CONTROL
    Qm = collective_emission_mpo(L; gamma_plus = gp, gamma_minus = gm, k = k)
    @test abs(sum(jump_weight(c, psi) for c in ch) - real(mpo_energy(psi, Qm))) < 1e-12
    g = _loss_rates(L)
    sc = site_emission_channels(L; gamma = g)
    @test all(norm(dense_state(factor_product(c.factors[1], psi)) - sqrt(g[j]) * _embed(_SM, j, L) * v) <
              1e-12 for (j, c) in enumerate(sc))
    @test abs(sum(jump_weight(c, psi) for c in sc) - real(mpo_energy(psi, _site_loss_mpo(L)))) < 1e-12
    @test sc[3].factors[1] isa SiteOp                               # a local jump needs no MPO
end

@testset "J0: pair loss against the Fock jump, sign structure included" begin
    L, U, kap = 3, 4.0, [1.0, 0.3, 2.0]
    f = _t4_hub(L, U, kap)
    @test f.res < 1e-12
    psi = hub4_product_state([:both, :up, :both])                    # (N↑, N↓) = (3, 2)
    for _ in 1:2
        tdvp2_step!(psi, f.Hm, ComplexF64(-0.3im); maxdim = 64, trunc_thresh = 1e-14)
    end
    a = _amplitudes(psi, f.states)
    v = Diagonal(f.s)' * a                                           # the Fock-basis vector
    ch = pair_loss_channels(L; kappa = kap)
    @test [c.name for c in ch] == ["pair1", "pair2", "pair3"]
    @test abs(sum(jump_weight(c, psi) for c in ch) -
              real(mpo_energy(psi, doublon_loss_mpo(L; kappa = kap)))) < 1e-12
    D = 4^L
    string1 = Diagonal([(-1)^hub_occ(b, hub_mode(1, 0)) for b in 0:(D - 1)])   # (-1)^{n_1↑}
    for (j, c) in enumerate(ch)
        phi, _ = apply_jump(c, psi)
        aj = _amplitudes(phi, f.states)
        e = Diagonal(f.s) * (f.jumps[j] * v)                         # the Fock jump, MPS basis
        @test abs(norm(aj) - norm(e)) < 1e-10 * norm(e)
        @test abs(abs(dot(e, aj)) - norm(e) * norm(aj)) < 1e-10 * norm(e)^2  # one global phase
        @test all(_hub_charges(b, L) == (2, 1) for b in 0:(D - 1) if abs(aj[b + 1]) > 1e-12)
        if j > 1                                                     # CONTROL: a sign error
            ew = Diagonal(f.s) * (string1 * (f.jumps[j] * v))
            @test abs(dot(ew, aj)) < (1 - 1e-3) * norm(ew) * norm(aj)
        end
    end
end

@testset "U07: a jump moves the sector, and BUG continues in it against ED" begin
    L = 6
    f = _emission_model(L)
    n = only(_spin_support(dense_state(f.psi0), L))
    phi, _ = apply_jump(f.ch[1], f.psi0)
    RC._jt_normalise!(phi)
    w = dense_state(phi)
    @test _spin_support(w, L) == Set([n - 1]) && abs(norm(w) - 1) < 1e-12
    dt, m = 0.05, 8
    for _ in 1:m
        _t4_step!(phi, f.G, dt)
    end
    ref = exp(-1im * m * dt * f.Gd) * w
    @test norm(dense_state(phi) - ref) < 1e-6 * norm(ref)
    # pair loss: (3, 2) -> (2, 1), then the no-jump flow there in the H-fixed basis
    Lh, kap = 3, [1.0, 0.3, 2.0]
    h = _t4_hub(Lh, 4.0, kap)
    psi = hub4_product_state([:both, :up, :both])
    for _ in 1:2
        tdvp2_step!(psi, h.Hm, ComplexF64(-0.3im); maxdim = 64, trunc_thresh = 1e-14)
    end
    q, _ = apply_jump(pair_loss_channels(Lh; kappa = kap)[3], psi)
    RC._jt_normalise!(q)
    a0 = _amplitudes(q, h.states)
    @test all(_hub_charges(b, Lh) == (2, 1) for b in 0:(4^Lh - 1) if abs(a0[b + 1]) > 1e-12)
    for _ in 1:m
        _t4_step!(q, h.G, dt)
    end
    S = Diagonal(h.s)
    ref = S * (exp(m * dt * (-1im * h.Hf - 0.5 * h.Qf)) * (S' * a0))
    @test norm(_amplitudes(q, h.states) - ref) < 1e-6 * norm(ref)
end

@testset "U13: the jump-time sampler" begin
    L = 6
    f = _emission_model(L; gm = 0.3)
    v0 = dense_state(f.psi0)
    obs(p) = vcat(real(overlap(p, p)), _sz_obs(L)(p))
    tg = collect(0.0:0.5:4.0)
    kw = (dt = 0.1, observe = obs, step_kw = _T4_STEP)
    t1 = jump_trajectory(f.psi0, f.G, f.ch, tg; seed = 7, kw...)
    t2 = jump_trajectory(f.psi0, f.G, f.ch, tg; seed = 7, kw...)
    t3 = jump_trajectory(f.psi0, f.G, f.ch, tg; seed = 8, kw...)
    @test !isempty(t1.events)
    tstar = _ed_jump_time(_ed_flow(-1im * f.Gd, v0), v0, rand(Random.MersenneTwister(7));
                          tmax = tg[end])
    @info "U13 first jump: engine $(t1.events[1].t), ED $(tstar), resid $(t1.events[1].resid), " *
          "evaluations $(t1.events[1].evals); events $(length(t1.events)), steps $(t1.nsteps)"
    @test abs(t1.events[1].t - tstar) < 1e-6
    @test abs(t1.events[1].resid) <= 1e-10
    @test [(e.t, e.channel) for e in t1.events] == [(e.t, e.channel) for e in t2.events]
    @test [e.t for e in t3.events] != [e.t for e in t1.events]
    @test all(abs(o[1] - 1) < 1e-10 for o in t1.obs)                 # normalised conditional states
    @test isempty(t1.flags) && t1.njumps[end] == length(t1.events)
    # CONTROL: an integrator that raises the norm is flagged, not renormalised away
    grow(p, h) = (_t4_step!(p, f.G, h); p[p.center] = to_concrete(2.0 * p[p.center]); p)
    tgw = jump_trajectory(f.psi0, f.G, f.ch, [0.0, 0.3]; dt = 0.1, seed = 7, step = grow)
    @test !isempty(tgw.flags) && all(fl.kind === :norm_growth for fl in tgw.flags)
    # the channel draw follows p_μ, recorded with every event; CONTROL: not a uniform draw
    ens = jump_ensemble(f.psi0, f.G, f.ch, [0.0, 4.0]; ntraj = 80, seed = 100, observe = obs,
                        dt = 0.1, step_kw = _T4_STEP)
    evs = reduce(vcat, [tr.events for tr in ens.trajs])
    p1 = [e.weights[1] / sum(e.weights) for e in evs]
    c1 = count(e -> e.channel == f.ch[1].name, evs)
    sig = sqrt(sum(p1 .* (1 .- p1)))
    @info "U13 channel draw: $(length(evs)) events, L+ drawn $c1, expected $(sum(p1)) ± $sig"
    @test length(evs) >= 100
    @test abs(c1 - sum(p1)) <= 4 * sig + 1
    @test abs(c1 - length(evs) / 2) > 4 * sqrt(length(evs)) / 2
end

@testset "U14: A2 collective emission, L = 4, against the Lindblad solution" begin
    L = 4
    f = _emission_model(L; gm = 0.3, start = [:up, :down, :up, :up])
    tg = collect(0.0:0.25:1.5)
    ens = jump_ensemble(f.psi0, f.G, f.ch, tg; ntraj = 300, seed = 1, observe = _sz_obs(L),
                        dt = 0.05, step_kw = _T4_STEP)
    v0 = dense_state(f.psi0)
    S = lindbladian_from(f.Hd, f.Ld)
    rhos, nex = _lindblad_counts(S, f.Qd, v0 * v0', tg)
    sz(r) = [real(tr(_embed(_SZ, j, L) * r)) for j in 1:L]
    flow = _ed_flow(-1im * f.Gd, v0)
    cond = [(w = flow(t); sz(w * w') ./ real(dot(w, w))) for t in tg]
    ok, dev = _within_ci(reduce(vcat, ens.mean), reduce(vcat, ens.stderr), reduce(vcat, sz.(rhos)))
    okn, devn = _within_ci(ens.jumps_mean, ens.jumps_stderr, nex)
    okc, devc = _within_ci(reduce(vcat, ens.mean), reduce(vcat, ens.stderr), reduce(vcat, cond))
    @info "U14 A2 L=4: worst |mean - Lindblad| / (4 se + 2e-3): S^z $dev, jumps $devn; " *
          "no-jump control $devc; jumps $(ens.jumps_mean) vs $(nex)"
    @test all(abs(tr(r) - 1) < 1e-10 for r in rhos)
    @test ok && okn
    @test !okc
end

@testset "U14: A1 pair loss, L = $L, against the Lindblad solution" for L in (2, 4)
    set_symmetry!(:U1)
    κ, U = 1.0, 4.0
    occ = L == 2 ? [:both, :up] : [:both, :up, :empty, :both]
    G = no_jump_mpo(hubbard_aa_mpo_d4(L; J = 1.0, U = U), doublon_loss_mpo(L; kappa = κ))
    ch = pair_loss_channels(L; kappa = κ)
    psi0 = hub4_product_state(occ)
    nupdn = _hub4_nupdn()
    obs = p -> vcat([hub4_density(p, j) for j in 1:L], [real(site_expval(p, j, nupdn)) for j in 1:L])
    tg = collect(0.0:0.25:1.5)
    ens = jump_ensemble(psi0, G, ch, tg; ntraj = L == 2 ? 300 : 150, seed = 1, observe = obs,
                        dt = 0.05, step_kw = _T4_STEP)
    D = 4^L
    Hf, Qf, jumps = _hub_fock(L; U = U, kappa = fill(κ, L))
    b0 = _hub_pattern(occ)
    rho0 = zeros(ComplexF64, D, D)
    rho0[b0 + 1, b0 + 1] = 1
    rhos, nex = _lindblad_counts(lindbladian_from(Hf, jumps), Qf, rho0, tg)
    nj = [Diagonal([hub_occ(b, hub_mode(j, 0)) + hub_occ(b, hub_mode(j, 1)) for b in 0:(D - 1)])
          for j in 1:L]
    dj = [Diagonal([hub_occ(b, hub_mode(j, 0)) * hub_occ(b, hub_mode(j, 1)) for b in 0:(D - 1)])
          for j in 1:L]
    ex(r) = vcat([real(tr(n * r)) for n in nj], [real(tr(n * r)) for n in dj])
    e0 = zeros(ComplexF64, D)
    e0[b0 + 1] = 1
    Gf = -1im * Hf - 0.5 * Qf
    cond = [(w = exp(t * Gf) * e0; ex(w * w') ./ real(dot(w, w))) for t in tg]
    ok, dev = _within_ci(reduce(vcat, ens.mean), reduce(vcat, ens.stderr), reduce(vcat, ex.(rhos)))
    okn, devn = _within_ci(ens.jumps_mean, ens.jumps_stderr, nex)
    okc, devc = _within_ci(reduce(vcat, ens.mean), reduce(vcat, ens.stderr), reduce(vcat, cond))
    @info "U14 A1 L=$L: worst |mean - Lindblad| / (4 se + 2e-3): n, n↑n↓ $dev, jumps $devn; " *
          "no-jump control $devc; jumps $(ens.jumps_mean) vs $(nex)"
    @test all(abs(tr(r) - 1) < 1e-10 for r in rhos)
    @test ok && okn
    @test !okc
end

end
