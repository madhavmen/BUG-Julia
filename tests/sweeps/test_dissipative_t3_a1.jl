# tests/sweeps/test_dissipative_t3_a1.jl -- dissipative_cbe_bug_implementation_plan.md, Task 3, A1:
# the Hubbard chain with bulk doublon loss, G = -iH - (κ/2) Σ n↑n↓, against symmetry-resolved ED
# (one (N↑, N↓) sector of the Fock space, carried into the d = 4 product basis by the basis
# phases that U02 fixes), and the unconditional Lindblad reference at L = 2, 4. Under U(1) x U(1).
#
#   smoke     L = 4, 6; U/J ∈ {0, 4}; κ/J ∈ {0, 1, 8, 20}; a product state with a doublon and an
#             empty site, and an entangled one (a lossless evolution of it): order-3 BUG ψ(t), P0,
#             conditional doublon density, <S^z S^z>_c, <Q>_c and the sector ranks against ED;
#             κ = 0 keeps the norm (the Hermitian control). The strict gate is κ ≤ 1 at this step;
#             κ ≥ 8 is graded by refinement below.
#   split     the split-loss baseline is the Strang product it claims, at full rank, and that
#             product differs from exp(dt G) (its splitting error is real)
#   strong    κ ∈ {8, 20}, dt halved twice: BUG, direct non-Hermitian TDVP2 and TDVP2 with the loss
#             split off exactly all converge to ED, P0 never grows, TDVP2's backward-substep
#             amplification is recorded; the split arm is second order
#   Lindblad  L = 2, 4: trace 1, ρ ⪰ 0, d<N>/dt = -2κ Σ<n↑n↓> exactly, and the initial-sector block
#             of ρ(t) is the no-jump |ψ̃><ψ̃| (loss only lowers N, so nothing re-enters the sector),
#             at L = 4 the BUG run's. Trajectory averages and jump counts need the jump engine
#             (plan Task 4).

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

const _A1_BUG = (hermitian = false, order = 3, tol_basis = 1e-8, tol_root = 1e-12,
                 tol_trunc = 1e-10, maxdim = 256)
const _A1_TDVP = (hermitian = false, maxdim = 256, trunc_thresh = 1e-12, maxiter = 40,
                  conv_tol = 1e-12)
# the split arm's TDVP2 sees only the Hermitian H: Lanczos, same settings otherwise
const _A1_SPLIT = merge(_A1_TDVP, (hermitian = true,))
# a doublon and an empty site at every L
const _A1_OCC = Dict(2 => [:both, :empty], 4 => [:both, :up, :empty, :down],
                     6 => [:both, :up, :empty, :down, :both, :empty])

"""
`occ`'s `(N↑, N↓)` sector: its Fock patterns `bs`, the d = 4 product states, the basis phases
`s` (`<b'|H_mpo|b> = s_b' H_f[b',b] conj(s_b)`, from the U = 4 Hamiltonian so the diagonal is
graded too) and their residual, and the start's index `k0`.
"""
function _a1_sector(L, occ)
    b0 = _hub_pattern(occ)
    bs = _hub_sector(L, _hub_charges(b0, L)...)
    states = [_hub4_state(b, L) for b in bs]
    Hf, _, _ = _hub_fock(L; U = 4.0, basis = bs)
    s, res = _edge_gauge(hubbard_aa_mpo_d4(L; J = 1.0, U = 4.0), states, Hf)
    return (; L, bs, states, s, res, k0 = findfirst(==(b0), bs))
end

"The MPOs and the product-basis ED generator `Gp = S (-i H_f - Q_f/2) S'` at `(U, κ)`."
function _a1_ops(sec; U, kappa)
    L = sec.L
    Hf, Qf, _ = _hub_fock(L; U = U, kappa = fill(kappa, L), basis = sec.bs)
    S = Diagonal(sec.s)
    Hm, Qm = hubbard_aa_mpo_d4(L; J = 1.0, U = U), doublon_loss_mpo(L; kappa = kappa)
    return (; H = Hm, Q = Qm, G = no_jump_mpo(Hm, Qm), Hp = S * Hf * S', Qp = S * Qf * S',
            Gp = S * (-1im * Hf - 0.5 * Qf) * S')
end

"MPS `(P0, conditional doublon densities, <Q>_c, <S^z_1 S^z_2>_c, <S^z_1 S^z_L>_c)`."
function _a1_obs(psi, Qm, nupdn)
    L, p = length(psi), _p0(psi)
    nd = [real(site_expval(psi, j, nupdn)) / p for j in 1:L]
    zz = [_szsz(psi, 1, 2, _hub4_sz!), _szsz(psi, 1, L, _hub4_sz!)] ./ p
    return p, nd, real(mpo_energy(psi, Qm)) / p, zz
end

"The same from an ED amplitude vector `v` over the sector patterns `bs`."
function _a1_obs_ed(v, bs, L, Qp)
    w = abs2.(v)
    p = sum(w)
    occ(b, j, σ) = hub_occ(b, hub_mode(j, σ))
    sz(b, j) = (occ(b, j, 0) - occ(b, j, 1)) / 2
    nd = [sum(w[k] * occ(bs[k], j, 0) * occ(bs[k], j, 1) for k in eachindex(bs)) / p for j in 1:L]
    zz(i, j) = sum(w[k] * sz(bs[k], i) * sz(bs[k], j) for k in eachindex(bs)) / p
    return p, nd, real(v' * Qp * v) / p, [zz(1, 2), zz(1, L)]
end

"""
`nsteps` of `step!(psi)` against the exact flow `exp(dt Gp) v` from `v0`, in place. Returns the
worst errors over the steps -- `:psi` (relative state), `:amp` (the product basis spans the MPS:
`||a||² = P0`), `:p0`, `:logp0`, `:nd`, `:q`, `:zz`, and `:grow`, the largest relative P0 increase
-- and the final ED vector.
"""
function _a1_compare!(psi, sec, ops, v0, dt, nsteps, step!)
    L, nupdn = sec.L, _hub4_nupdn()
    E = exp(dt * ops.Gp)
    v, d = copy(v0), Dict{Symbol, Float64}()
    pprev = _p0(psi)
    for _ in 1:nsteps
        step!(psi)
        v = E * v
        a = _amplitudes(psi, sec.states)
        p, nd, q, zz = _a1_obs(psi, ops.Q, nupdn)
        pe, nde, qe, zze = _a1_obs_ed(v, sec.bs, L, ops.Qp)
        _upd!(d, :psi, norm(a - v) / norm(v))
        _upd!(d, :amp, abs(sum(abs2, a) - p) / p)
        _upd!(d, :p0, abs(p - pe) / pe)
        _upd!(d, :logp0, abs(log(p) - log(pe)))
        _upd!(d, :nd, maximum(abs.(nd - nde)))
        _upd!(d, :q, abs(q - qe) / max(qe, 1.0))
        _upd!(d, :zz, maximum(abs.(zz - zze)))
        _upd!(d, :grow, max(0.0, p - pprev) / pprev)
        pprev = p
    end
    return d, v
end

"A lossless evolution of `psi`: an entangled start, which the ED takes from its amplitudes."
function _entangle!(psi, H; t = 0.3, dt = 0.1)
    for _ in 1:round(Int, t / dt)
        cbe_bug_step!(psi, H, ComplexF64(-dt * im); hermitian = true, tol_basis = 1e-8,
                      tol_root = 1e-12, tol_trunc = 1e-12, maxdim = 256)
    end
    return psi
end

set_symmetry!(:U1)
const _A1_SEC = Dict(L => _a1_sector(L, _A1_OCC[L]) for L in (2, 4, 6))

@testset "dissipative Task 3 (G1), A1: Hubbard chain with bulk doublon loss" begin

@testset "A1 fixture: basis phases and bond charges, L = $L" for L in (2, 4, 6)
    sec = _A1_SEC[L]
    @test sec.res < 1e-12                       # the d = 4 H is the Fock H up to state phases
    @test all(abs.(abs.(sec.s) .- 1) .< 1e-12)
    psi = hub4_product_state(_A1_OCC[L])
    a = _amplitudes(psi, sec.states)
    @test abs(abs(a[sec.k0]) - 1) < 1e-12 && sum(abs2, a) ≈ 1   # the start IS one sector state
    b0 = sec.bs[sec.k0]
    for i in 1:(L - 1)                          # the bond-label decoder, where charges are known
        @test _mps_sector_dims(psi, i, _hub_decode) == Dict(last(_hub_split(i)(b0)) => 1)
    end
end

@testset "A1 smoke: L = $L, U = $U, κ = $κ, $start" for (L, U, κ, start) in
        [(L, U, κ, st) for L in (4, 6) for U in (0.0, 4.0) for κ in (0.0, 1.0, 8.0, 20.0)
         for st in (:product, :entangled) if st === :product || U == 4.0]
    sec = _A1_SEC[L]
    ops = _a1_ops(sec; U = U, kappa = κ)
    psi = hub4_product_state(_A1_OCC[L])
    start === :entangled && _entangle!(psi, ops.H)
    start === :entangled && @test maximum(bond_dims(psi)) > 1
    v0 = _amplitudes(psi, sec.states)
    dt, n = 0.025, 20
    d, v = _a1_compare!(psi, sec, ops, v0, dt, n,
                        p -> cbe_bug_step!(p, ops.G, ComplexF64(-dt * im); _A1_BUG...))
    @info "A1 smoke L=$L U=$U κ=$κ $start" errors = d P0 = _p0(psi)
    @test d[:amp] < 1e-10                       # the sector basis spans the MPS
    @test d[:grow] < 1e-10                      # no false norm growth
    tol = κ <= 1 ? 1e-6 : 1e-3                  # κ ≥ 8: a sanity bound; graded by refinement
    @test d[:psi] < tol && d[:p0] < tol && d[:nd] < tol && d[:zz] < tol && d[:q] < 10tol
    κ == 0 && @test abs(_p0(psi) - sum(abs2, v0)) < 1e-8      # Hermitian control: norm kept
    κ > 0 && @test _p0(psi) < sum(abs2, v0) * (1 - 1e-3)      # ... and the loss acts
    for i in 1:(L - 1)                          # sector ranks against the ED Schmidt values
        miss, excess = _rank_check(_mps_sector_dims(psi, i, _hub_decode),
                                   _sector_schmidt(v, sec.bs, _hub_split(i)), norm(v))
        @test miss == 0 && excess == 0
    end
end

@testset "A1 split baseline: the Strang product, at full rank" begin
    L, κ, dt = 4, 8.0, 0.05
    sec = _A1_SEC[L]
    ops = _a1_ops(sec; U = 4.0, kappa = κ)
    psi = _entangle!(hub4_product_state(_A1_OCC[L]), ops.H; t = 2.0)
    v0 = _amplitudes(psi, sec.states)
    full = all(1:(L - 1)) do i
        sv = _sector_schmidt(v0, sec.bs, _hub_split(i))
        _mps_sector_dims(psi, i, _hub_decode) == Dict(c => length(s) for (c, s) in sv)
    end
    @test full                                   # TDVP2's projector is then the identity
    tdvp2_split_loss_step!(psi, ops.H, ComplexF64(-dt * im), doublon_loss_local(L; kappa = κ);
                           _A1_SPLIT...)
    D = exp(-dt / 4 * ops.Qp)
    strang = D * exp(-1im * dt * ops.Hp) * D * v0
    exact = exp(dt * ops.Gp) * v0
    @test norm(_amplitudes(psi, sec.states) - strang) < 1e-8 * norm(strang)
    @test norm(strang - exact) > 1e-6 * norm(exact)          # CONTROL: the splitting is real
end

@testset "A1 strong loss, κ = $κ: BUG, TDVP2 and split TDVP2 under refinement" for κ in (8.0, 20.0)
    L, T = 6, 0.2
    sec = _A1_SEC[L]
    ops = _a1_ops(sec; U = 4.0, kappa = κ)
    psi0 = hub4_product_state(_A1_OCC[L])
    v0 = _amplitudes(psi0, sec.states)
    qloc = doublon_loss_local(L; kappa = κ)
    dts = (0.05, 0.025, 0.0125)
    errs = Dict(a => Float64[] for a in (:bug, :tdvp, :split))
    logp = Dict(a => Float64[] for a in (:bug, :tdvp, :split))
    amp = Float64[]
    for dt in dts
        tau = ComplexF64(-dt * im)
        arms = (bug = p -> cbe_bug_step!(p, ops.G, tau; _A1_BUG...),
                tdvp = p -> (r = Ref(0.0);
                             tdvp2_step!(p, ops.G, tau; _A1_TDVP..., backward_report = r);
                             push!(amp, r[])),
                split = p -> tdvp2_split_loss_step!(p, ops.H, tau, qloc; _A1_SPLIT...))
        for a in keys(arms)
            d, _ = _a1_compare!(copy(psi0), sec, ops, v0, dt, round(Int, T / dt), arms[a])
            push!(errs[a], d[:psi])
            push!(logp[a], d[:logp0])
            @test d[:grow] < 1e-10               # P0 never increases, every arm, every dt
            @test d[:amp] < 1e-10
        end
    end
    @info "A1 strong κ=$κ (κh = $(κ .* dts))" errs logp tdvp_backward_amplification = maximum(amp)
    @test all(isfinite, amp)
    # converges: each halving at least halves the error, down to a 1e-9 floor
    conv(e, r) = all(e[k + 1] <= max(e[k] / r, 1e-9) for k in 1:(length(e) - 1))
    @test conv(errs[:bug], 2) && errs[:bug][end] < 1e-4
    @test conv(errs[:tdvp], 2) && errs[:tdvp][end] < 1e-2
    # the split arm is second order, and its splitting error is above the floor (a control)
    @test errs[:split][1] > 1e-6 && conv(errs[:split], 3) && errs[:split][end] < 1e-2
    @test logp[:bug][end] < 1e-4                 # -log P0, the hazard integral, after refinement
end

@testset "A1 unconditional: sparse Lindblad, L = $L" for L in (2, 4)
    κ, U, ts = 1.0, 4.0, [0.25, 0.5]
    D = 4^L
    Hf, Qf, jumps = _hub_fock(L; U = U, kappa = fill(κ, L))
    S = lindbladian_from(Hf, jumps)
    b0 = _hub_pattern(_A1_OCC[L])
    rho0 = zeros(ComplexF64, D, D)
    rho0[b0 + 1, b0 + 1] = 1
    rhos = sparse_lindblad_trajectory(S, rho0, ts; tol = 1e-13)
    Nd = Diagonal([count_ones(b) for b in 0:(D - 1)])
    nn = Qf / κ                                  # Σ_j n↑n↓
    sec = _A1_SEC[L]
    idx = sec.bs .+ 1
    Gf = -1im * Hf[idx, idx] - 0.5 * Qf[idx, idx]
    for (t, rho) in zip(ts, rhos)
        @test abs(tr(rho) - 1) < 1e-10
        @test norm(rho - rho') < 1e-10 && eigmin(Hermitian(rho)) > -1e-10
        drho = unvec_rm(S * vec_rm(rho), D)
        @test abs(real(tr(Nd * drho)) + 2κ * real(tr(nn * rho))) < 1e-10   # each jump removes 2
        @test abs(real(tr(Nd * drho)) + κ * real(tr(nn * rho))) > 1e-3     # CONTROL: not 1
        # the initial-sector block is the no-jump state: nothing re-enters the sector
        psit = exp(t * Gf) * Matrix{ComplexF64}(I, length(idx), length(idx))[:, sec.k0]
        @test norm(rho[idx, idx] - psit * psit') < 1e-10
        # unconditional <n↑n↓> (jumps included) is not the conditional no-jump one
        cond = real(psit' * Diagonal(diag(nn)[idx]) * psit) / real(psit' * psit)
        @test abs(real(tr(nn * rho)) - cond) > 1e-3
    end
    if L == 4                                    # the BUG no-jump run is that block
        ops = _a1_ops(sec; U = U, kappa = κ)
        psi = hub4_product_state(_A1_OCC[L])
        dt = 0.025
        _a1_compare!(psi, sec, ops, _amplitudes(psi, sec.states), dt, round(Int, ts[end] / dt),
                     p -> cbe_bug_step!(p, ops.G, ComplexF64(-dt * im); _A1_BUG...))
        vf = Diagonal(sec.s)' * _amplitudes(psi, sec.states)   # product basis -> Fock basis
        @test norm(rhos[end][idx, idx] - vf * vf') < 1e-6 * real(vf' * vf)
        @test abs(real(tr(rhos[end][idx, idx])) - _p0(psi)) < 1e-6        # P0 = Tr P ρ P
    end
end

end
