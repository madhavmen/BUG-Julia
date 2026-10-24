# tests/sweeps/test_dissipative_t4_b.jl -- dissipative_cbe_bug_implementation_plan.md, Task 4: the
# small physics-project smoke tests, B1 (Hubbard junction, localized pair loss) and B2 (many-body
# subradiance, the conditional extension of A2).
#
#   B1 current   `hub4_current_mpo` equals the Fock current on an entangled state, and with H's own
#                hoppings d<n_{b,σ}>/dt = j_{b-1} - j_b (ED)
#   B1 prep      leads prepared decoupled (J_link = 0) by imaginary time reach the ED ground energy of
#                their per-lead sectors; the imbalances <N_L,σ - N_R,σ> are the prepared ones
#   B1 smoke     L = 4: only the sink's channel exists; per trajectory S^z_tot is constant and N drops
#                by 2 per jump; the ensemble <N>, <S^z>, <n0↑n0↓> and the currents on BOTH sides of
#                the sink agree with the master equation (CONTROL: the no-jump values do not); on ED,
#                d<N>/dt = -2κ<n0↑n0↓> and d<S^z>/dt = 0
#   B1 cond.     L = 6: the no-jump BUG run against sector ED (P0, conditional densities)
#   B2 cond.     L = 6, 8, N_exc = 1, 2: P0, the dark-state weight, λ = -d log P0/dt and <S^z S^z>_c
#                against ED; at N_exc = 1 the Jz term is a one-body edge potential (ED identity),
#                at N_exc = 2 it is not (CONTROL)
#   B2 counts    L = 6, N_exc = 2: the trajectory photon-count distribution matches the count-resolved
#                master equation; CONTROL: independent emitters at the same total rate do not

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

const _TB_STEP = (closure_checks = true, tol_basis = 1e-10, tol_root = 1e-12)
_tb_step!(p, G, h) = cbe_bug_step!(p, G, ComplexF64(-im * h); hermitian = false, order = 3,
                                   tol_trunc = 1e-12, _TB_STEP...)

"The per-lead charges `(N_L↑, N_L↓, N_R↑, N_R↓)` of Fock pattern `b` (leads `< j0` and `> j0`)."
_lead_charges(b, L, j0) = (sum(hub_occ(b, hub_mode(j, 0)) for j in 1:(j0 - 1)),
                          sum(hub_occ(b, hub_mode(j, 1)) for j in 1:(j0 - 1)),
                          sum(hub_occ(b, hub_mode(j, 0)) for j in (j0 + 1):L),
                          sum(hub_occ(b, hub_mode(j, 1)) for j in (j0 + 1):L))

"""
B1 preparation: the ground state of the decoupled leads (`J_link = 0`) in the per-lead sectors of
the product state `occ`, by normalised imaginary time to `beta`. Returns the state and its energy.
"""
function _b1_prepare(L, occ; J = 1.0, U = 4.0, j0 = L ÷ 2, beta = 20.0)
    set_symmetry!(:U1)
    m = pair_loss_junction(L; J = J, J_link = 0.0, U = U, kappa = 1.0, j0 = j0)
    psi = hub4_product_state(occ)
    hist = ite_evolve!(psi, m.H, log_panel_steps(beta, 8; per_panel = 4); order = 2,
                       tol_basis = 1e-9, tol_trunc = 1e-12, tol_root = 1e-12, maxdim = 128, retries = 3)
    return psi, hist
end

@testset "dissipative Task 4: B1 and B2 smoke tests" begin

@testset "B1 current: the MPO current against Fock, and continuity" begin
    set_symmetry!(:U1)
    L, U = 4, 4.0
    m = pair_loss_junction(L; J = 1.0, J_link = 0.7, U = U, kappa = 1.0)
    states = [_hub4_state(b, L) for b in 0:(4^L - 1)]
    Hf, _, _ = _hub_fock(L; J = m.Jb, U = U)
    MH = _mpo_matrix(m.H, states; skip = (a, b) -> _hub_charges(a - 1, L) != _hub_charges(b - 1, L))
    s = _gauge_phases(MH, Hf)
    @test norm(MH - Diagonal(s) * Hf * Diagonal(s)') < 1e-12 * norm(Hf)   # the bond J's are H's
    psi = hub4_product_state([:both, :up, :empty, :down])
    for _ in 1:2
        tdvp2_step!(psi, m.H, ComplexF64(-0.3im); maxdim = 64, trunc_thresh = 1e-14)
    end
    v = Diagonal(s)' * _amplitudes(psi, states)
    P = real(dot(v, v))
    jd = Dict((b, σ) => real(dot(v, _fock_current(L, b, σ, m.Jb[b]) * v)) / P
              for b in 1:(L - 1), σ in 0:1)
    jm = Dict((b, σ) => real(mpo_energy(psi, hub4_current_mpo(L, b, σ; J = m.Jb[b]))) / _p0(psi)
              for b in 1:(L - 1), σ in 0:1)
    @info "B1 currents: MPO $(jm), Fock $(jd)"
    @test all(abs(jm[k] - jd[k]) < 1e-10 for k in keys(jd))
    @test maximum(abs, values(jd)) > 1e-2                                # the currents flow
    for j in 1:L, σ in 0:1                                               # continuity (ED)
        n = _fock_n(L, j, σ)
        dn = real(dot(v, 1im * (Hf * n - n * Hf) * v)) / P
        @test abs(dn - (get(jd, (j - 1, σ), 0.0) - get(jd, (j, σ), 0.0))) < 1e-10
    end
end

@testset "B1 preparation: decoupled leads reach their ground state" begin
    L, U, j0 = 6, 4.0, 3
    occ = [:up, :down, :empty, :down, :empty, :down]          # leads (1↑1↓) | sink | (0↑2↓)
    psi, hist = _b1_prepare(L, occ; U = U, j0 = j0)
    b0 = _hub_pattern(occ)
    sec = [b for b in 0:(4^L - 1) if _lead_charges(b, L, j0) == _lead_charges(b0, L, j0) &&
                                     hub_occ(b, hub_mode(j0, 0)) + hub_occ(b, hub_mode(j0, 1)) == 0]
    Jb = [b in (j0 - 1, j0) ? 0.0 : 1.0 for b in 1:(L - 1)]
    Hs, _, _ = _hub_fock(L; J = Jb, U = U, basis = sec)
    E0 = minimum(eigvals(Hermitian(Hs)))
    bad = [r for r in hist if r.status === :failed]
    nL = [sum(real(site_expval(psi, j, to_concrete(lock(hub4_space().Fu', 2) * hub4_space().Fu)))
              for j in 1:(j0 - 1)), sum(hub4_density(psi, j) for j in (j0 + 1):L)]
    @info "B1 prep: E $(hist[end].e_after), ED $(E0), failed steps $(length(bad)); N_L↑ $(nL[1]), N_R $(nL[2])"
    @test isempty(bad)
    @test abs(hist[end].e_after - E0) < 1e-7
    @test abs(nL[1] - 1) < 1e-10 && abs(nL[2] - 2) < 1e-10          # the prepared imbalance
end

@testset "B1 smoke, L = 4: trajectories against the master equation" begin
    L, U, κ, j0, Jl = 4, 4.0, 2.0, 2, 0.7
    occ = [:both, :empty, :up, :down]                         # leads (1↑1↓) | sink | (1↑1↓)
    psi0, _ = _b1_prepare(L, occ; U = U, j0 = j0)
    m = pair_loss_junction(L; J = 1.0, J_link = Jl, U = U, kappa = κ, j0 = j0)
    @test length(m.channels) == 1 && m.channels[1].name == "pair$j0"   # only the sink loses
    nupdn = _hub4_nupdn()
    nup = to_concrete(lock(hub4_space().Fu', 2) * hub4_space().Fu)
    obs = p -> vcat(sum(hub4_density(p, j) for j in 1:L),
                    sum(real(site_expval(p, j, nup)) for j in 1:L) - 0.5 * sum(hub4_density(p, j) for j in 1:L),
                    real(site_expval(p, j0, nupdn)),
                    [real(mpo_energy(p, hub4_current_mpo(L, b, σ; J = m.Jb[b]))) for b in (j0 - 1, j0)
                     for σ in 0:1])
    tg = collect(0.0:0.25:2.0)
    ens = jump_ensemble(psi0, m.G, m.channels, tg; ntraj = 200, seed = 1, observe = obs, dt = 0.05,
                        step_kw = _TB_STEP)
    for tr in ens.trajs                                       # exact, per trajectory
        @test all(abs(tr.obs[k][2] - tr.obs[1][2]) < 1e-9 for k in eachindex(tr.obs))
        @test all(abs(tr.obs[k][1] - (tr.obs[1][1] - 2 * tr.njumps[k])) < 1e-9 for k in eachindex(tr.obs))
    end
    # the master equation from the ED ground state of the same decoupled leads
    D = 4^L
    Hp, _, _ = _hub_fock(L; J = [b in (j0 - 1, j0) ? 0.0 : 1.0 for b in 1:(L - 1)], U = U)
    b0 = _hub_pattern(occ)
    sec = [b + 1 for b in 0:(D - 1) if _lead_charges(b, L, j0) == _lead_charges(b0, L, j0) &&
                                       hub_occ(b, hub_mode(j0, 0)) + hub_occ(b, hub_mode(j0, 1)) == 0]
    F = eigen(Hermitian(Hp[sec, sec]))
    v0 = zeros(ComplexF64, D)
    v0[sec] = F.vectors[:, 1]
    Hf, Qf, jumps = _hub_fock(L; J = m.Jb, U = U, kappa = [j == j0 ? κ : 0.0 for j in 1:L])
    Ls = [jumps[j0]]
    S = lindbladian_from(Hf, Ls)
    rhos, nex = _lindblad_counts(S, Qf, v0 * v0', tg)
    Nd = sum(_fock_n(L, j, σ) for j in 1:L, σ in 0:1)
    Szd = 0.5 * sum(_fock_n(L, j, 0) - _fock_n(L, j, 1) for j in 1:L)
    dd = _fock_n(L, j0, 0) * _fock_n(L, j0, 1)
    cur = [_fock_current(L, b, σ, m.Jb[b]) for b in (j0 - 1, j0) for σ in 0:1]
    ex(r) = real.(vcat(tr(Nd * r), tr(Szd * r), tr(dd * r), [tr(c * r) for c in cur]))
    for r in rhos                                             # the ED balances
        dr = unvec_rm(S * vec_rm(r), D)
        @test abs(real(tr(Nd * dr)) + 2κ * real(tr(dd * r))) < 1e-10
        @test abs(real(tr(Szd * dr))) < 1e-10
    end
    Gf = -1im * Hf - 0.5 * Qf
    cond = [(w = exp(t * Gf) * v0; ex(w * w') ./ real(dot(w, w))) for t in tg]
    ok, dev = _within_ci(reduce(vcat, ens.mean), reduce(vcat, ens.stderr), reduce(vcat, ex.(rhos)))
    okn, devn = _within_ci(ens.jumps_mean, ens.jumps_stderr, nex)
    okc, devc = _within_ci(reduce(vcat, ens.mean), reduce(vcat, ens.stderr), reduce(vcat, cond))
    @info "B1 L=4: worst |mean - Lindblad| / (4 se + 2e-3): obs $dev, jumps $devn; no-jump control " *
          "$devc; final ensemble [N, Sz, d0, j(j0-1)↑↓, j(j0)↑↓] $(ens.mean[end]) ± $(ens.stderr[end]), " *
          "Lindblad $(ex(rhos[end])); jumps $(ens.jumps_mean[end]) vs $(nex[end])"
    @test ok && okn
    @test !okc
end

@testset "B1 conditional, L = 6: no-jump BUG against sector ED" begin
    set_symmetry!(:U1)
    L, U, κ, j0 = 6, -4.0, 5.0, 3
    occ = [:up, :down, :both, :down, :up, :empty]
    m = pair_loss_junction(L; J = 1.0, J_link = 0.5, U = U, kappa = κ, j0 = j0)
    sec = _hub_sector(L, _hub_charges(_hub_pattern(occ), L)...)
    Hs, Qs, _ = _hub_fock(L; J = m.Jb, U = U, kappa = [j == j0 ? κ : 0.0 for j in 1:L], basis = sec)
    k0 = findfirst(==(_hub_pattern(occ)), sec)
    w = zeros(ComplexF64, length(sec))
    w[k0] = 1
    Gs = -1im * Hs - 0.5 * Qs
    psi = hub4_product_state(occ)
    dt, n = 0.05, 20
    dens(b) = [hub_occ(b, hub_mode(j, 0)) + hub_occ(b, hub_mode(j, 1)) for j in 1:L]
    nd = reduce(hcat, dens.(sec))                              # (L, |sector|)
    worst = 0.0
    for step in 1:n
        _tb_step!(psi, m.G, dt)
        x = exp(step * dt * Gs) * w
        P = real(dot(x, x))
        worst = max(worst, abs(_p0(psi) - P),
                    maximum(abs(hub4_density(psi, j) / _p0(psi) - dot(nd[j, :], abs2.(x)) / P) for j in 1:L))
    end
    @info "B1 conditional L=6: worst |P0| and conditional density error $worst; P0(T) $(_p0(psi))"
    @test worst < 1e-6
    @test _p0(psi) < 0.95                                      # the sink acts
end

@testset "B2 conditional, L = $L, N_exc = $ne" for L in (6, 8), ne in (1, 2)
    occ = [j in (L ÷ 2 - ne + 1):(L ÷ 2) ? :up : :down for j in 1:L]     # a central block
    res = Dict{Float64, Any}()
    for Jz in (0.0, 1.0)
        f = _emission_model(L; Jz = Jz, gp = 1.0, gm = 1.0, start = occ)
        v0 = dense_state(f.psi0)
        Pd = _dark_projector(f.Ld)
        flow = _ed_flow(-1im * f.Gd, v0)
        psi = copy(f.psi0)
        dt, n = 0.05, 40
        e = Dict(:p0 => 0.0, :dark => 0.0, :lam => 0.0, :zz => 0.0)
        P = [1.0]
        Pex = [1.0]
        for step in 1:n
            _tb_step!(psi, f.G, dt)
            x, y = flow(step * dt), dense_state(psi)
            push!(P, real(dot(y, y))); push!(Pex, real(dot(x, x)))
            e[:p0] = max(e[:p0], abs(P[end] - Pex[end]))
            e[:dark] = max(e[:dark], abs(real(dot(y, Pd * y)) / P[end] - real(dot(x, Pd * x)) / Pex[end]))
            zz(z) = real(dot(z, _embed(_SZ, 1, L) * _embed(_SZ, L, L) * z)) / real(dot(z, z))
            e[:zz] = max(e[:zz], abs(zz(y) - zz(x)))
        end
        lam(p) = [-(log(p[k + 1]) - log(p[k])) / dt for k in 1:n]
        e[:lam] = maximum(abs.(lam(P) .- lam(Pex)))
        res[Jz] = (err = e, P = Pex, H = f.Hd, f = f)
        @info "B2 L=$L N_exc=$ne Jz=$Jz: worst errors $(e); P0(T) $(Pex[end]), dark weight " *
              "$(real(dot(flow(n * dt), Pd * flow(n * dt))) / Pex[end]), λ(T) $(lam(Pex)[end])"
        @test all(<(1e-6), values(e))
    end
    # interactions act only with two or more excitations: at N_exc = 1 the Jz term equals the
    # one-body edge potential -(Jz/2)(bonds at j) up to a constant, so P0 agrees exactly
    f1 = res[1.0].f
    v0 = dense_state(f1.psi0)
    edge = sum(-0.5 * (j in (1, L) ? 1 : 2) * (_embed(_SZ, j, L) + 0.5I) for j in 1:L)
    Hone = res[0.0].H + Matrix(edge)
    flow_one = _ed_flow(-1im * (Hone - 0.5im * f1.Qd), v0)
    P1 = [real(norm(flow_one(k * 0.05))^2) for k in 0:40]
    gap = maximum(abs.(P1 .- res[1.0].P))
    @info "B2 L=$L N_exc=$ne: |P0(Jz=1) - P0(Jz=0 + edge potential)| = $gap"
    ne == 1 ? @test(gap < 1e-10) : @test(gap > 1e-4)
end

@testset "B2 counts, L = 6, N_exc = 2: the photon-count distribution" begin
    L, T, ne = 6, 3.0, 2
    occ = [j in (L ÷ 2 - ne + 1):(L ÷ 2) ? :up : :down for j in 1:L]
    f = _emission_model(L; gp = 1.0, gm = 1.0, start = occ)
    v0 = dense_state(f.psi0)
    Pn = _count_distribution(f.Hd, f.Ld, v0 * v0', T; nmax = ne)
    ens = jump_ensemble(f.psi0, f.G, f.ch, [0.0, T]; ntraj = 300, seed = 11,
                        observe = p -> [0.0], dt = 0.05, step_kw = _TB_STEP)
    cnt = [count(tr -> tr.njumps[end] == k, ens.trajs) for k in 0:ne]
    N = length(ens.trajs)
    ok(P) = all(abs(cnt[k] - N * P[k]) <= 4 * sqrt(N * P[k] * (1 - P[k])) + 1 for k in eachindex(P))
    # CONTROL: independent emitters with the same total rate (Γ₊ + Γ₋ per site) emit differently
    Lind = [sqrt(2.0) * _embed(_SM, j, L) for j in 1:L]
    Pind = _count_distribution(f.Hd, Lind, v0 * v0', T; nmax = ne)
    @info "B2 counts L=$L: trajectories $(cnt) of $N, collective master equation $(N .* Pn), " *
          "independent emitters $(N .* Pind)"
    @test abs(sum(Pn) - 1) < 1e-10 && abs(sum(Pind) - 1) < 1e-10
    @test ok(Pn)
    @test !ok(Pind)
end

end
