# tests/sweeps/test_dissipative_t3_a2.jl -- dissipative_cbe_bug_implementation_plan.md, Task 3, A2:
# the XXZ chain with collective emission, H = Σ (S^x S^x + S^y S^y + Jz S^z S^z) + Σ h_j S^z_j,
# h_j = 0.2 (-1)^j, L_± = √Γ Σ_j e^{±ikj} S⁻_j with Γ₊ = Γ₋ = Γ and ka = π/3, against dense ED
# restricted to the excitation sector (the spin product basis carries no phases) and the exact
# Lindblad reference. Under U(1).
#
#   MPO          L = 6: Q against the dense L₊'L₊ + L₋'L₋, G + G' = -Q, Q ⪰ 0, at Γ₊ = Γ₋ and
#                Γ₊ ≠ Γ₋ (complex couplings; k -> -k must differ there)
#   smoke        L = 6, 8; Jz ∈ {0, 1}; Γ ∈ {0.2, 2}; 2 and 3 excitations: order-3 BUG ψ(t), P0,
#                λ = <Q>_c, <S^z_i>_c, <S^z S^z>_c and sector ranks against ED, no norm growth.
#                Strict at Γ = 0.2; Γ = 2 is graded by refinement.
#   bright/dark  L = 6, Jz = 1, Γ = 1: seeds e^{∓βQ/2} ψ0 (BUG on Q alone); the dark seed's λ(0)
#                far below the bright one's and its P0(T) above, both runs against ED, and the
#                first step's sector fidelity
#   refine       Jz = 1, Γ = 2 (interaction and dissipation increased): dt halved twice, converges
#   Lindblad     L = 4, 6: trace 1, ρ ⪰ 0, d<N>/dt = -<Q>_ρ exactly (one excitation per jump), and
#                the initial-sector block of ρ(t) is the no-jump |ψ̃><ψ̃|, at L = 6 the BUG run's.
#                One-excitation runs are a baseline only (plan A2); trajectory averages need the
#                jump engine (plan Task 4).

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

const _A2_BUG = (hermitian = false, order = 3, tol_basis = 1e-8, tol_root = 1e-12,
                 tol_trunc = 1e-10, maxdim = 256)
const _A2_K = π / 3
const _A2_OCC = Dict(4 => [:up, :down, :up, :down],
                     6 => [:up, :down, :down, :up, :down, :down],
                     8 => [:up, :down, :up, :down, :down, :up, :down, :down])

_a2_field(L) = [0.2 * (-1)^j for j in 1:L]
_a2_dense_h(L, Jz) = dense_heisenberg(L; J = 1.0, delta = Jz) +
                     sum(_a2_field(L)[j] * _embed(_SZ, j, L) for j in 1:L)

"""
The A2 model at `(Jz, Γ)` on `occ`'s excitation sector: MPOs `H`, `Q`, `G`, the sector's indices
`bs` (`dense_state` order) and product states, and the sector blocks of the dense `G`, `Q` and
`S^z_j` (diagonal), with the start's index `k0`.
"""
function _a2_model(L, occ; Jz, gamma)
    Hm = field_mpo_from_terms(xxz_chain(L; J = 1.0, delta = Jz), _a2_field(L))
    Qm = collective_emission_mpo(L; gamma_plus = gamma, k = _A2_K)
    Lp, Lm = _emission_jumps(L; gp = gamma, gm = gamma, k = _A2_K)
    Qd = Lp' * Lp + Lm' * Lm
    Gd = -1im * _a2_dense_h(L, Jz) - 0.5 * Qd
    i0 = sum((occ[j] === :down ? 1 : 0) << (L - j) for j in 1:L)
    N = _spin_charge(i0, L)
    bs = [idx for idx in 0:(2^L - 1) if _spin_charge(idx, L) == N]
    ix = bs .+ 1
    states = _spin_states(L)[ix]
    sz = [[((idx >> (L - j)) & 1) == 0 ? 0.5 : -0.5 for idx in bs] for j in 1:L]
    return (; L, N, H = Hm, Q = Qm, G = no_jump_mpo(Hm, Qm), bs, states, sz,
            Gd = Gd[ix, ix], Qd = Qd[ix, ix], k0 = findfirst(==(i0), bs),
            Hfull = _a2_dense_h(L, Jz), jumps = [Lp, Lm])
end

"MPS `(P0, <S^z_j>_c, λ = <Q>_c, <S^z_1 S^z_2>_c, <S^z_1 S^z_L>_c)`."
function _a2_obs(psi, Qm)
    L, p = length(psi), _p0(psi)
    sz = [sz_expectation(psi, j) / p for j in 1:L]
    zz = [_szsz(psi, 1, 2, apply_sz!), _szsz(psi, 1, L, apply_sz!)] ./ p
    return p, sz, real(mpo_energy(psi, Qm)) / p, zz
end

"The same from a sector amplitude vector `v`."
function _a2_obs_ed(v, m)
    w = abs2.(v)
    p = sum(w)
    sz = [dot(w, m.sz[j]) / p for j in 1:m.L]
    zz(i, j) = sum(w .* m.sz[i] .* m.sz[j]) / p
    return p, sz, real(v' * m.Qd * v) / p, [zz(1, 2), zz(1, m.L)]
end

"""
`nsteps` of `step!(psi)` against `exp(dt Gd) v` from `v0`, in place: the worst `:psi`, `:amp`,
`:p0`, `:logp0`, `:sz`, `:lambda`, `:zz`, `:grow` over the steps, `:fid1` the first step's
sector infidelity, and the final ED vector.
"""
function _a2_compare!(psi, m, v0, dt, nsteps, step!)
    E = exp(dt * m.Gd)
    v, d = copy(v0), Dict{Symbol, Float64}()
    pprev = _p0(psi)
    for n in 1:nsteps
        step!(psi)
        v = E * v
        a = _amplitudes(psi, m.states)
        p, sz, q, zz = _a2_obs(psi, m.Q)
        pe, sze, qe, zze = _a2_obs_ed(v, m)
        n == 1 && (d[:fid1] = 1 - abs2(dot(v, a)) / (sum(abs2, v) * sum(abs2, a)))
        _upd!(d, :psi, norm(a - v) / norm(v))
        _upd!(d, :amp, abs(sum(abs2, a) - p) / p)
        _upd!(d, :p0, abs(p - pe) / pe)
        _upd!(d, :logp0, abs(log(p) - log(pe)))
        _upd!(d, :sz, maximum(abs.(sz - sze)))
        _upd!(d, :lambda, abs(q - qe) / max(qe, 1.0))
        _upd!(d, :zz, maximum(abs.(zz - zze)))
        _upd!(d, :grow, max(0.0, p - pprev) / pprev)
        pprev = p
    end
    return d, v
end

"Renormalise in place (canonical at site 1, where the norm lives)."
_normalize!(psi) = (canonical!(psi, 1); psi[1] = to_concrete(psi[1] / norm(psi[1])); psi)

"`nsteps` BUG steps of `exp(tau Q)` on `Q` alone, renormalised: `tau < 0` darkens, `> 0` brightens."
function _a2_seed!(psi, Qm, tau, nsteps)
    for _ in 1:nsteps
        cbe_bug_step!(psi, Qm, ComplexF64(tau); hermitian = true, tol_basis = 1e-8,
                      tol_root = 1e-12, tol_trunc = 1e-12, maxdim = 256)
        _normalize!(psi)
    end
    return psi
end

set_symmetry!(:U1)

@testset "dissipative Task 3 (G1), A2: XXZ chain with collective emission" begin

@testset "A2 nonlocal MPO on L = 6, Γ₊ = 1, Γ₋ = $gm" for gm in (1.0, 0.4)
    L, gp = 6, 1.0
    Qm = collective_emission_mpo(L; gamma_plus = gp, gamma_minus = gm, k = _A2_K)
    Gm = no_jump_mpo(field_mpo_from_terms(xxz_chain(L; J = 1.0, delta = 1.0), _a2_field(L)), Qm)
    Lp, Lm = _emission_jumps(L; gp = gp, gm = gm, k = _A2_K)
    Qd = Lp' * Lp + Lm' * Lm
    skip(a, b) = _spin_charge(a - 1, L) != _spin_charge(b - 1, L)
    states = _spin_states(L)
    MQ, MG = _mpo_matrix(Qm, states; skip = skip), _mpo_matrix(Gm, states; skip = skip)
    @test norm(MQ - Qd) < 1e-12 * norm(Qd)
    @test norm(MG - (_a2_dense_h(L, 1.0) - 0.5im * Qd)) < 1e-12 * norm(Qd)
    G = -1im * MG
    @test norm(G + G' + MQ) < 1e-12 * norm(MQ) && eigmin(Hermitian(MQ)) > -1e-12
    @test count(x -> abs(x) > 1e-12, MQ - Diagonal(MQ)) > 0     # the cross-site terms are there
    @test maximum(mpo_virtual_dims(Qm)) <= 6     # compact: no O(L^2) pair terms
    if gm != gp                                  # CONTROL: conjugated phases are another Q
        Lp2, Lm2 = _emission_jumps(L; gp = gp, gm = gm, k = -_A2_K)
        @test norm(MQ - (Lp2' * Lp2 + Lm2' * Lm2)) > 1e-3 * norm(Qd)
    end
end

@testset "A2 fixture: bond charges, L = $L" for L in (4, 6, 8)
    psi = product_state(_A2_OCC[L])
    m = _a2_model(L, _A2_OCC[L]; Jz = 1.0, gamma = 1.0)
    a = _amplitudes(psi, m.states)
    @test abs(abs(a[m.k0]) - 1) < 1e-12 && sum(abs2, a) ≈ 1
    @test m.N >= 2                               # interactions matter (plan A2)
    for i in 1:(L - 1)
        @test _mps_sector_dims(psi, i, _spin_decode) == Dict(last(_spin_split(i, L)(m.bs[m.k0])) => 1)
    end
end

@testset "A2 smoke: L = $L, Jz = $Jz, Γ = $Γ" for L in (6, 8), Jz in (0.0, 1.0), Γ in (0.2, 2.0)
    m = _a2_model(L, _A2_OCC[L]; Jz = Jz, gamma = Γ)
    psi = product_state(_A2_OCC[L])
    v0 = _amplitudes(psi, m.states)
    dt, n = 0.025, 20
    d, v = _a2_compare!(psi, m, v0, dt, n,
                        p -> cbe_bug_step!(p, m.G, ComplexF64(-dt * im); _A2_BUG...))
    @info "A2 smoke L=$L Jz=$Jz Γ=$Γ" errors = d P0 = _p0(psi)
    @test d[:amp] < 1e-10 && d[:grow] < 1e-10
    tol = Γ <= 0.2 ? 1e-6 : 1e-3                 # Γ = 2: a sanity bound; graded by refinement
    @test d[:psi] < tol && d[:p0] < tol && d[:sz] < tol && d[:zz] < tol && d[:lambda] < 10tol
    @test _p0(psi) < 1 - 1e-3                    # the emission acts
    for i in 1:(L - 1)
        miss, excess = _rank_check(_mps_sector_dims(psi, i, _spin_decode),
                                   _sector_schmidt(v, m.bs, _spin_split(i, L)), norm(v))
        @test miss == 0 && excess == 0
    end
end

@testset "A2 bright vs dark seeds, L = 6, Jz = 1, Γ = 1" begin
    L = 6
    m = _a2_model(L, _A2_OCC[L]; Jz = 1.0, gamma = 1.0)
    dark = _a2_seed!(product_state(_A2_OCC[L]), m.Q, -2.0, 10)     # e^{-20 Q} ψ0
    bright = _a2_seed!(product_state(_A2_OCC[L]), m.Q, 0.25, 8)    # e^{+2 Q} ψ0
    lam(psi) = real(mpo_energy(psi, m.Q)) / _p0(psi)
    lam0 = (dark = lam(dark), bright = lam(bright))
    @test lam0.dark < 0.1 * lam0.bright
    dt, n = 0.01, 50
    out = Dict{Symbol, Any}()
    for (name, psi) in ((:dark, dark), (:bright, bright))
        v0 = _amplitudes(psi, m.states)
        @test abs(lam0[name] - real(v0' * m.Qd * v0) / sum(abs2, v0)) < 1e-10   # λ(0) is ED's
        d, _ = _a2_compare!(psi, m, v0, dt, n,
                            p -> cbe_bug_step!(p, m.G, ComplexF64(-dt * im); _A2_BUG...))
        @test d[:psi] < 1e-5 && d[:p0] < 1e-5 && d[:lambda] < 1e-4 && d[:grow] < 1e-10
        @test d[:fid1] < 1e-10                   # one-step sector fidelity
        out[name] = (P0 = _p0(psi), errors = d)
    end
    @info "A2 bright/dark" lam0 out
    @test out[:dark].P0 > out[:bright].P0        # delayed decay of the dark seed
end

@testset "A2 refinement at Jz = 1, Γ = 2" begin
    L, T = 6, 0.2
    m = _a2_model(L, _A2_OCC[L]; Jz = 1.0, gamma = 2.0)
    psi0 = product_state(_A2_OCC[L])
    v0 = _amplitudes(psi0, m.states)
    errs = Float64[]
    for dt in (0.05, 0.025, 0.0125)
        d, _ = _a2_compare!(copy(psi0), m, v0, dt, round(Int, T / dt),
                            p -> cbe_bug_step!(p, m.G, ComplexF64(-dt * im); _A2_BUG...))
        @test d[:grow] < 1e-10
        push!(errs, max(d[:psi], d[:sz], d[:lambda]))
    end
    @info "A2 refinement" errs
    @test all(errs[k + 1] <= max(errs[k] / 2, 1e-9) for k in 1:2) && errs[end] < 1e-4
end

@testset "A2 unconditional: Lindblad, L = $L" for L in (4, 6)
    Jz, Γ, ts = 1.0, 0.5, [0.25, 0.5]
    m = _a2_model(L, _A2_OCC[L]; Jz = Jz, gamma = Γ)
    D = 2^L
    S = lindbladian_from(m.Hfull, m.jumps)
    i0 = m.bs[m.k0]
    rho0 = zeros(ComplexF64, D, D)
    rho0[i0 + 1, i0 + 1] = 1
    rhos = sparse_lindblad_trajectory(S, rho0, ts; tol = 1e-13)
    Nd = sum(_embed(_SZ, j, L) + 0.5I for j in 1:L)
    Qfull = sum(X' * X for X in m.jumps)
    ix = m.bs .+ 1
    for (t, rho) in zip(ts, rhos)
        @test abs(tr(rho) - 1) < 1e-10
        @test norm(rho - rho') < 1e-10 && eigmin(Hermitian(rho)) > -1e-10
        drho = unvec_rm(S * vec_rm(rho), D)
        @test abs(real(tr(Nd * drho)) + real(tr(Qfull * rho))) < 1e-10   # one excitation per jump
        @test abs(real(tr(Nd * drho)) + 2 * real(tr(Qfull * rho))) > 1e-3  # CONTROL
        psit = exp(t * m.Gd)[:, m.k0]
        @test norm(rho[ix, ix] - psit * psit') < 1e-10
    end
    if L == 6
        psi = product_state(_A2_OCC[L])
        dt = 0.025
        _a2_compare!(psi, m, _amplitudes(psi, m.states), dt, round(Int, ts[end] / dt),
                     p -> cbe_bug_step!(p, m.G, ComplexF64(-dt * im); _A2_BUG...))
        a = _amplitudes(psi, m.states)
        @test norm(rhos[end][ix, ix] - a * a') < 1e-6 * sum(abs2, a)
        @test abs(real(tr(rhos[end][ix, ix])) - _p0(psi)) < 1e-6
    end
end

end
