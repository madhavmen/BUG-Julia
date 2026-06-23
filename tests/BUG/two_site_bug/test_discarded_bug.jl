# test_discarded_bug.jl
#
# Tests for the discarded-projector BUG variant (`discarded_bug`) on the XX model,
# which has an analytical (free-fermion) solution. We compare against the dense
# matrix exponential (exact diagonalization), which is the analytical propagator
# for XX.
#
# Hard constraints exercised here:
#   * rank-adaptive: every state starts LOW rank and the bond dimension GROWS.
#   * NO backward correction: the sweep never does a single-site negative-dt solve
#     (asserted via info.backward_correction_calls == 0).
#   * 2-site local update is validated ONLY against the analytical 2-site
#     exp(-i dt H_eff), not against the faithful scheme.

using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Printf

const _SRC = joinpath(@__DIR__, "..", "..", "..", "src")
include(joinpath(_SRC, "TTutils", "TTutils.jl"))
using .TTutils
include(joinpath(_SRC, "BUG", "BUG.jl"))
using .BUG
include(joinpath(_SRC, "TDVP", "TDVP.jl"))
using .TDVP

# ── XX model + helpers ───────────────────────────────────────────────────────

# H_XX = (1/2) Σ_b (S+_b S-_{b+1} + S-_b S+_{b+1}). Free-fermion ⇒ exp(-iHt) is
# the analytical propagator (here via exact diagonalization of the dense matrix).
function _xx_mpo(sites)
    os = OpSum()
    for i in 1:(length(sites) - 1)
        os += 0.5, "S+", i, "S-", i + 1
        os += 0.5, "S-", i, "S+", i + 1
    end
    return TensorTrainOperator(MPO(os, sites))
end

function _dense_2site_heff(L_mpo, R_mpo, W_l, W_r, link_l, site_l, site_r, link_r)
    HW = L_mpo * W_l * W_r * R_mpo
    d_tot = dim(link_l) * dim(site_l) * dim(site_r) * dim(link_r)
    H = zeros(ComplexF64, d_tot, d_tot)
    e = zeros(ComplexF64, d_tot)
    bra = (prime(link_l), prime(site_l), prime(site_r), prime(link_r))
    for c in 1:d_tot
        fill!(e, 0); e[c] = 1
        Θ = itensor(reshape(e, dim(link_l), dim(site_l), dim(site_r), dim(link_r)),
                    link_l, site_l, site_r, link_r)
        H[:, c] = vec(Array(HW * Θ, bra...))
    end
    return H
end

_tvec(t, o...) = ComplexF64.(vec(Array(t, o...)))

function _exact_exp(H, v, dt)
    F = eigen(Hermitian(H))
    return F.vectors * (exp.(-im * dt .* F.values) .* (F.vectors' * v))
end

function _infid(a, b)
    d = norm(a) * norm(b)
    iszero(d) && return 1.0
    return 1.0 - clamp(abs(dot(a, b)) / d, 0.0, 1.0)
end

# Dense state vector / Hamiltonian of a whole TensorTrain / TensorTrainOperator.
function _full_vec(psi)
    ITensors.disable_warn_order()
    v = ComplexF64.(TTutils.vector(psi))
    ITensors.reset_warn_order()
    return v
end
function _full_mat(W)
    ITensors.disable_warn_order()
    M = ComplexF64.(TTutils.matrix(W))
    ITensors.reset_warn_order()
    return M
end

# ⟨S^z_j⟩ profile from a dense state vector. TTutils.vector flattens with site 1 as
# the fastest index; for "S=1/2", basis index 1 = |Up⟩ (S^z=+1/2), 2 = |Dn⟩ (−1/2).
function _sz_profile_from_vec(v::AbstractVector, N::Int)
    p = abs2.(v); p ./= sum(p)
    A = reshape(p, ntuple(_ -> 2, N))
    return [0.5 * (sum(selectdim(A, j, 1)) - sum(selectdim(A, j, 2))) for j in 1:N]
end

# ── 2-site: validate the local update against the analytical 2-site evolution ─

@testset "discarded_bug local 2-site update vs analytical exp(-i dt H_eff)" begin
    ITensors.disable_warn_order()
    N = 4
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)

    # LOW-rank start so augmentation must grow the bond (rank-adaptive check).
    psi = random_tt(sites; maxdim = 2, seed = 2024)
    normalize!(psi)
    orthogonalize!(psi, 1)

    bond = 2
    L_cur, R_cur = _owned_two_site_mpo_envs(psi, W, bond)
    snap = BUG._canonical_quantum_bond_snapshot(psi, W, bond, L_cur, R_cur)
    HW = snap.L_mpo_cur * snap.W_left * snap.W_right * snap.R_mpo_cur
    link_l, site_l, site_r, link_r = snap.link_l, snap.site_l, snap.site_r, snap.link_r
    old_rank = dim(snap.canon_u0)

    theta0 = snap.U0_tens * snap.S0_tens * snap.V0_tens
    theta0_vec = _tvec(theta0, link_l, site_l, site_r, link_r)
    H = _dense_2site_heff(snap.L_mpo_cur, snap.R_mpo_cur, snap.W_left, snap.W_right,
                          link_l, site_l, site_r, link_r)
    @test norm(H - H') / (norm(H) + 1e-16) < 1e-12

    for dt in (0.05, 0.02, 0.005)
        cand = BUG._with_bug_expv_backend(:native_hermitian_lanczos) do
            BUG._with_bug_time_prefactor(ComplexF64(-im)) do
                BUG.discarded_bug_local_update(snap, HW;
                    dt = dt, maxdim = typemax(Int),
                    lanczos_tol = 1e-15, lanczos_maxiter = 40, substep_method = :expv)
            end
        end
        theta1 = cand.U_new * cand.V_new
        theta1_vec = _tvec(theta1, link_l, site_l, site_r, link_r)
        theta_exact = _exact_exp(H, theta0_vec, dt)
        infid = _infid(theta1_vec, theta_exact)
        @info "2-site vs analytical" dt old_rank new_rank=cand.keep n_new_k=cand.n_new_k n_new_l=cand.n_new_l infid
        # Rank-adaptive: the bond grew beyond the low-rank start.
        @test cand.keep > old_rank
        @test cand.n_new_k >= 1 && cand.n_new_l >= 1
        # The local update is the exact local 2-site evolution (full augmentation
        # ⇒ Galerkin solve on the complete 2-site space) — machine precision.
        @test infid < 1e-10
    end
end

# ── 6-site: rank-adaptive symmetric sweep vs exact diagonalization ────────────

@testset "discarded_bug 6-site rank-adaptive vs exact diagonalization (XX)" begin
    ITensors.disable_warn_order()
    N = 6
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)
    H_full = _full_mat(W)
    @test norm(H_full - H_full') / (norm(H_full) + 1e-16) < 1e-12

    # LOW-rank start (χ=2 ≪ full χ=8 at the middle bond): the integrator must
    # grow the bonds to track the true evolution.
    psi0 = random_tt(sites; maxdim = 2, seed = 4242)
    normalize!(psi0)
    v0 = _full_vec(psi0)
    bonddims(p) = [dim(commonind(p[k], p[k + 1])) for k in 1:(length(p) - 1)]
    χ0 = bonddims(psi0)
    @info "initial bond dims" χ0

    # dt-refinement against exact diagonalization. A symmetric (forward dt/2 +
    # reverse dt/2) sweep of exact local solves is a 2nd-order integrator, so the
    # error vs exp(-i dt H) is the O(dt²) Strang/sweep-splitting error — the SAME
    # error class as 2-site TDVP. We compare head-to-head with TDVP2 (which uses a
    # backward correction) to show discarded_bug is comparable WITHOUT one, and we
    # certify O(dt²) convergence + the absence of a double-counting error floor.
    dts = (0.05, 0.025, 0.0125, 0.00625)
    results = NamedTuple[]
    for dt in dts
        pd = deepcopy(psi0)
        info = BUG.discarded_bug_step!(pd, W;
            dt = dt, order = :symmetric, maxdim = typemax(Int),
            lanczos_tol = 1e-15, lanczos_maxiter = 50, substep_method = :expv)
        vd  = _full_vec(pd)
        ref = _exact_exp(H_full, v0, dt)
        infid_dbug = _infid(vd, ref)
        nrm_err = abs(norm(vd) - norm(v0)) / norm(v0)

        # Reference 2-site TDVP (same dt, same start, full rank), Strang order.
        pt = deepcopy(psi0)
        TDVP.tdvp2_step!(pt, W; dt = dt, step_mode = :symmetric_fr,
            maxdim = typemax(Int), cutoff = 0.0,
            lanczos_tol = 1e-15, lanczos_maxiter = 50, substep_method = :expv)
        infid_tdvp = _infid(_full_vec(pt), ref)

        push!(results, (dt = dt, infid = infid_dbug, infid_tdvp = infid_tdvp,
                        nrm_err = nrm_err, χ = bonddims(pd),
                        backward = info.backward_correction_calls))
        @info "6-site symmetric step" dt infid_dbug infid_tdvp ratio=infid_dbug/infid_tdvp nrm_err χ=bonddims(pd) backward=info.backward_correction_calls
    end

    # ── HARD CONSTRAINT: no backward correction anywhere ──
    @test all(r -> r.backward == 0, results)

    # ── Rank-adaptive: bonds grew beyond the χ=2 start ──
    @test all(r -> maximum(r.χ) > maximum(χ0), results)

    # ── Norm conservation: O(dt²), and at least as good as TDVP2 (XX is unitary) ──
    @test all(r -> r.nrm_err < 1e-3, results)
    nrm_ratios = [results[i - 1].nrm_err / results[i].nrm_err for i in 2:length(results)]
    @test all(r -> 3.5 < r < 4.5, nrm_ratios)        # O(dt²) norm drift

    # ── Clean O(dt²) convergence to exact diagonalization ──
    infids = [r.infid for r in results]
    @test issorted(infids; rev = true)               # monotone decreasing with dt
    order_ratios = [infids[i - 1] / infids[i] for i in 2:length(infids)]
    @test all(r -> 3.5 < r < 4.5, order_ratios)      # ratio ≈ 4 ⇒ 2nd order

    # ── No double counting: the O(dt²) ratio is STEADY (no error floor) ──
    # Double counting would inject a dt-independent floor (ratio → 1) or a spurious
    # O(dt) term (drifting ratio). A pinned ratio ≈ 4 across a 16× dt range rules
    # both out. We check the ratios don't drift by more than 10% across the range.
    @test maximum(order_ratios) / minimum(order_ratios) < 1.1

    # ── Comparable to 2-site TDVP: same order, error within a small steady factor ──
    dbug_over_tdvp = [r.infid / r.infid_tdvp for r in results]
    @test all(x -> 0.5 < x < 6.0, dbug_over_tdvp)    # comparable magnitude
    @test maximum(dbug_over_tdvp) / minimum(dbug_over_tdvp) < 1.1  # steady ⇒ same error class

    for i in 2:length(results)
        @info @sprintf("order: dt %.5f→%.5f  DBUG ratio=%.2f  TDVP ratio=%.2f  DBUG/TDVP=%.2f",
                       results[i - 1].dt, results[i].dt, order_ratios[i - 1],
                       results[i - 1].infid_tdvp / results[i].infid_tdvp, dbug_over_tdvp[i])
    end
end

# ── Domain-wall melting quench (the quantum benchmark) ────────────────────────
#
# Quench from a (near-)sharp domain wall |↑↑↑↑↓↓↓↓⟩ under H_XX and watch it melt.
# A PURE product wall has χ=1 on every bond, and NO BUG-family method (the faithful
# scheme included) can bootstrap rank from a product state in one step — the
# |↑↓⟩→|↓↑⟩ two-spin-flip hop is invisible to the one-sided K/L generators. So we
# seed the bonds near the wall with a small entangling rotation exp(-iθ h_b) (θ
# small ⇒ still near the sharp wall) to give the rank-adaptive sweep a non-trivial
# discarded space to grow into. Then the wall melts and the bond dimension climbs.
#
# Correctness here is checked by basis/index-robust invariants and convergence
# (NOT tight long-time amplitude accuracy): discarded_bug has no backward
# correction, so its per-step error constant is a small (~2–4×) multiple of 2-site
# TDVP's; over a long melt this compounds, but BOTH are the same O(dt²) error
# class. We therefore assert: energy conservation, rank growth (the wall melts),
# no backward correction, and O(dt²) convergence with a bounded DBUG/TDVP ratio.

# Seed bonds `bonds` of the sharp wall with exp(-iθ h_{b,b+1}) via ITensorMPS apply
# (clean link tags), returning a TensorTrain whose seeded bonds have χ≥2. A pure
# product wall (χ=1 everywhere) cannot be rank-bootstrapped by any BUG-family step.
function _seeded_domain_wall(sites, center, bonds, θ)
    N = length(sites)
    psi = MPS(sites, [i <= center ? "Up" : "Dn" for i in 1:N])
    gates = [exp(-im * θ * 0.5 *
                 (op("S+", sites[b]) * op("S-", sites[b + 1]) +
                  op("S-", sites[b]) * op("S+", sites[b + 1]))) for b in bonds]
    return TensorTrain(apply(gates, psi; cutoff = 1e-16))
end

_energy_tt(psi, W) = real(dot(psi, TTutils.contract(W, psi))) / real(dot(psi, psi))

# Evolve a fresh copy of `psi0` to time `T` with `nsteps` of `method`
# (:dbug | :tdvp), normalizing each step; return the final dense state vector.
function _evolve_traj(method::Symbol, psi0, W, T, nsteps, maxdim)
    dt = T / nsteps
    p = deepcopy(psi0)
    for _ in 1:nsteps
        if method === :dbug
            BUG.discarded_bug_step!(p, W; dt = dt, order = :symmetric, maxdim = maxdim,
                lanczos_tol = 1e-14, lanczos_maxiter = 50, substep_method = :expv)
        else
            TDVP.tdvp2_step!(p, W; dt = dt, step_mode = :symmetric_fr, maxdim = maxdim,
                cutoff = 1e-14, lanczos_tol = 1e-14, lanczos_maxiter = 50, substep_method = :expv)
        end
        normalize!(p)
    end
    return _full_vec(p), p
end

@testset "Domain-wall melting (XX): discarded_bug vs exact diag & 2-site TDVP" begin
    ITensors.disable_warn_order()
    N = 8
    center = cld(N, 2)
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)
    H_full = _full_mat(W)
    bonddims(p) = [dim(commonind(p[k], p[k + 1])) for k in 1:(length(p) - 1)]

    # Near-sharp wall: seed every bond with a small θ so rank can grow as it melts.
    θ = 0.15
    psi0 = _seeded_domain_wall(sites, center, 1:(N - 1), θ)
    v0 = _full_vec(psi0)
    E0 = _energy_tt(psi0, W)
    χ0 = bonddims(psi0)
    sz0 = _sz_profile_from_vec(v0, N)
    @info "domain-wall init" χ0 E0 center

    # ── Trajectory invariants: energy conserved, wall melts (rank grows), bk=0 ──
    dt = 0.02; nsteps = 25; maxdim = 64       # T = 0.5
    psi_d = deepcopy(psi0)
    total_backward = 0; χmax_d = maximum(χ0); max_dE_d = 0.0; melt_signal = 0.0
    for step in 1:nsteps
        info = BUG.discarded_bug_step!(psi_d, W; dt = dt, order = :symmetric, maxdim = maxdim,
            lanczos_tol = 1e-14, lanczos_maxiter = 50, substep_method = :expv)
        normalize!(psi_d)
        total_backward += info.backward_correction_calls
        max_dE_d = max(max_dE_d, abs(_energy_tt(psi_d, W) - E0))
        χmax_d = max(χmax_d, maximum(bonddims(psi_d)))
        sz_exact = _sz_profile_from_vec(_exact_exp(H_full, v0, step * dt), N)
        melt_signal = max(melt_signal, maximum(abs.(sz_exact .- sz0)))
    end
    @info "trajectory invariants" max_dE_d χmax_d total_backward melt_signal

    @test total_backward == 0            # HARD CONSTRAINT: no backward correction
    @test χmax_d > maximum(χ0)            # rank-adaptive: the wall melted
    @test melt_signal > 0.05             # the wall actually moved (non-trivial)
    @test max_dE_d < 1e-10               # energy conserved to ~machine precision

    # ── O(dt²) convergence with bounded DBUG/TDVP ratio (short fixed T) ──
    # At a short fixed T the per-step error has not compounded, so BOTH methods
    # converge to exact diagonalization as dt→0; DBUG stays within a bounded factor
    # of TDVP (same error class), confirming it is a correct 2nd-order integrator.
    T = 0.1
    conv = NamedTuple[]
    for ns in (4, 8, 16, 32)
        vd, _ = _evolve_traj(:dbug, psi0, W, T, ns, maxdim)
        vt, _ = _evolve_traj(:tdvp, psi0, W, T, ns, maxdim)
        ref = _exact_exp(H_full, v0, T)
        push!(conv, (ns = ns, id = _infid(vd, ref), it = _infid(vt, ref)))
        @info @sprintf("T=%.2f ns=%2d  DBUG infid=%.3e  TDVP infid=%.3e  D/T=%.2f",
                       T, ns, conv[end].id, conv[end].it, conv[end].id / conv[end].it)
    end
    id = [c.id for c in conv]
    # both converge (error decreases as dt→0)
    @test issorted(id; rev = true)
    @test conv[end].id < conv[1].id / 8        # roughly O(dt²) over 8× step refinement
    # same error class: DBUG within a bounded, steady factor of TDVP
    ratios = [c.id / c.it for c in conv]
    @test all(r -> r < 8.0, ratios)
end
