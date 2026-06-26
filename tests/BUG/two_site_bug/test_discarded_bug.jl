# test_discarded_bug.jl
#
# Tests for the discarded-projector BUG variant (`discarded_bug`) on the XX model,
# which has an analytical (free-fermion) solution. We compare ONLY against the dense
# matrix exponential (exact diagonalization), the analytical propagator for XX —
# never against the other BUGs.
#
# `discarded_bug` is the MPS realisation of the rank-adaptive tree-tensor-network BUG
# (Ceruti–Lubich–Walach / Sulz, Alg. 5–7), specialised to the MPS tree and run as a
# single GLOBAL sweep per step (mirror of the validated Python `sweep.py`): form
# phi = H·psi, build augmented left/right isometries that keep psi EXACT and admit
# only the DISCARDED part (I − U0 U0^dagger) of phi (never an M/N overlap matrix),
# then integrate ONE Galerkin centre connecting tensor under the two-site effective
# Hamiltonian.
#
# Properties exercised here:
#   * EXACT at full bond dimension (the Galerkin core is the exact evolution in the
#     augmented basis) — validated ONLY against the analytical propagator.
#   * SINGLE-STEP infidelity is O(dt^4) (ratio ~16 per dt-halving) ⇒ the step is
#     2nd order in the state and CONVERGENT — refining dt to a fixed time keeps
#     reducing the error (NO forward-only floor).
#   * NO backward correction (info.backward_correction_calls == 0).
#   * rank-adaptive: a low-rank / pure-product-wall start melts into the ballistic
#     light cone, the bond dimension growing along the chain (bounded by 2^(N/2)).

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

# ── XX model + helpers ───────────────────────────────────────────────────────
function _xx_mpo(sites)
    os = OpSum()
    for i in 1:(length(sites) - 1)
        os += 0.5, "S+", i, "S-", i + 1
        os += 0.5, "S-", i, "S+", i + 1
    end
    return TensorTrainOperator(MPO(os, sites))
end
function _exact_exp(H, v, dt)
    F = eigen(Hermitian(H))
    return F.vectors * (exp.(-im * dt .* F.values) .* (F.vectors' * v))
end
function _infid(a, b)
    d = norm(a) * norm(b)
    iszero(d) && return 1.0
    return 1.0 - clamp(abs(dot(a, b)) / d, 0.0, 1.0)
end
function _full_vec(psi)
    ITensors.disable_warn_order(); v = ComplexF64.(TTutils.vector(psi)); ITensors.reset_warn_order(); return v
end
function _full_mat(W)
    ITensors.disable_warn_order(); M = ComplexF64.(TTutils.matrix(W)); ITensors.reset_warn_order(); return M
end
function _sz_profile_from_vec(v::AbstractVector, N::Int)
    p = abs2.(v); p ./= sum(p)
    A = reshape(p, ntuple(_ -> 2, N))
    return [0.5 * (sum(selectdim(A, j, 1)) - sum(selectdim(A, j, 2))) for j in 1:N]
end
bonddims(p) = [dim(commonind(p[k], p[k + 1])) for k in 1:(length(p) - 1)]
_energy_tt(psi, W) = real(dot(psi, TTutils.contract(W, psi))) / real(dot(psi, psi))

# ── Full-rank step is the exact evolution (vs analytical exp(-i dt H)) ─────────
@testset "discarded_bug full-rank step == analytical exp(-i dt H) (XX)" begin
    ITensors.disable_warn_order()
    N = 4
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)
    H = _full_mat(W)
    @test norm(H - H') / (norm(H) + 1e-16) < 1e-12

    psi0 = random_tt(sites; maxdim = 2, seed = 2024)
    normalize!(psi0)
    v0 = _full_vec(psi0)
    old_max = maximum(bonddims(psi0))

    for dt in (0.05, 0.02, 0.005)
        pd = deepcopy(psi0)
        info = discarded_bug_step!(pd, W; dt = dt, maxdim = typemax(Int),
            lanczos_tol = 1e-15, lanczos_maxiter = 40, substep_method = :expv)
        infid = _infid(_full_vec(pd), _exact_exp(H, v0, dt))
        @info "full-rank step vs analytical" dt new_max=maximum(bonddims(pd)) infid backward=info.backward_correction_calls
        @test info.backward_correction_calls == 0
        @test maximum(bonddims(pd)) > old_max          # rank grew from the χ=2 start
        @test infid < 1e-10                            # exact at full rank
    end
end

# ── Single-step error is O(dt^4) (infidelity) ⇒ 2nd order, no backward correction ─
@testset "discarded_bug 8-site single-step O(dt^4) vs exact diagonalization (XX)" begin
    ITensors.disable_warn_order()
    N = 8
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)
    H_full = _full_mat(W)
    @test norm(H_full - H_full') / (norm(H_full) + 1e-16) < 1e-12

    psi0 = random_tt(sites; maxdim = 2, seed = 4242)
    normalize!(psi0)
    v0 = _full_vec(psi0)

    dts = (0.05, 0.025, 0.0125)
    results = NamedTuple[]
    for dt in dts
        pd = deepcopy(psi0)
        info = discarded_bug_step!(pd, W; dt = dt, maxdim = typemax(Int),
            lanczos_tol = 1e-15, lanczos_maxiter = 50, substep_method = :expv)
        ref = _exact_exp(H_full, v0, dt)
        push!(results, (dt = dt, infid = _infid(_full_vec(pd), ref),
                        nrm_err = abs(norm(_full_vec(pd)) - norm(v0)) / norm(v0),
                        backward = info.backward_correction_calls))
        @info "6-site single step" dt infid=results[end].infid backward=results[end].backward
    end

    @test all(r -> r.backward == 0, results)           # HARD CONSTRAINT: no backward correction
    @test all(r -> r.nrm_err < 1e-10, results)         # unitary ⇒ norm conserved

    infids = [r.infid for r in results]
    @test issorted(infids; rev = true)                 # monotone decreasing with dt
    ratios = [infids[i - 1] / infids[i] for i in 2:length(infids)]
    @test all(r -> 8.0 < r < 30.0, ratios)             # ratio ≈ 16 ⇒ O(dt^4) infidelity (2nd order)
    for i in 2:length(results)
        @info @sprintf("single-step order: dt %.5f→%.5f  ratio=%.2f",
                       results[i - 1].dt, results[i].dt, ratios[i - 1])
    end
end

# ── Fixed-time: refining dt CONVERGES (no forward-only floor) + vs 2-site TDVP ─
@testset "discarded_bug fixed-time convergence vs exact diagonalization (XX)" begin
    ITensors.disable_warn_order()
    N = 8
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)
    H_full = _full_mat(W)
    psi0 = random_tt(sites; maxdim = 2, seed = 4242)
    normalize!(psi0)
    v0 = _full_vec(psi0)

    function _evolve_dbug(T, nsteps)
        dt = T / nsteps
        p = deepcopy(psi0)
        for _ in 1:nsteps
            discarded_bug_step!(p, W; dt = dt, maxdim = typemax(Int),
                lanczos_tol = 1e-14, lanczos_maxiter = 50, substep_method = :expv)
            normalize!(p)
        end
        return _full_vec(p)
    end

    # Full rank ⇒ no truncation floor; the only error is the time discretisation. A
    # forward-only method with a per-step projection floor would NOT shrink here.
    T = 0.2
    id = Float64[]
    for ns in (4, 8, 16)
        push!(id, _infid(_evolve_dbug(T, ns), _exact_exp(H_full, v0, T)))
        @info @sprintf("fixed-time T=%.2f ns=%2d  DBUG infid=%.3e", T, ns, id[end])
    end
    @test issorted(id; rev = true)                     # refining dt keeps reducing the error
    @test id[1] > 8.0 * id[end]                        # CONVERGES (no forward-only floor)
    ratios = [id[i - 1] / id[i] for i in 2:length(id)]
    @test all(r -> 8.0 < r < 30.0, ratios)             # ~16× per dt-halving ⇒ O(dt^4) global infidelity
end

# ── Ballistic light cone from a PURE product wall (rank-adaptive growth) ───────
@testset "discarded_bug ballistic light cone (pure product wall, XX)" begin
    ITensors.disable_warn_order()
    N = 8
    center = cld(N, 2)
    sites = siteinds("S=1/2", N)
    W = _xx_mpo(sites)
    H_full = _full_mat(W)

    psi0 = TensorTrain(MPS(sites, [i <= center ? "Up" : "Dn" for i in 1:N]))
    v0 = _full_vec(psi0)
    E0 = _energy_tt(psi0, W)
    χ0 = bonddims(psi0)
    @info "pure product wall χ0" χ0 E0
    @test all(==(1), χ0)                               # genuinely a product state

    dt, nsteps = 0.1, 40                               # T = 4.0: light cone reaches the edges
    psi = deepcopy(psi0)
    total_backward = 0; max_dE = 0.0
    for step in 1:nsteps
        info = discarded_bug_step!(psi, W; dt = dt, maxdim = 64,
            lanczos_tol = 1e-14, lanczos_maxiter = 50, substep_method = :expv)
        normalize!(psi)
        total_backward += info.backward_correction_calls
        max_dE = max(max_dE, abs(_energy_tt(psi, W) - E0))
    end
    χ = bonddims(psi)
    @info "light cone after melt" χ max_dE total_backward

    c = N ÷ 2                                          # central bond (1-based)
    @test total_backward == 0                          # HARD CONSTRAINT: no backward correction
    @test max_dE < 1e-9                                # energy conserved (unitary, exact local solve)
    @test maximum(χ) > maximum(χ0)                     # rank-adaptive: the wall melted
    @test maximum(χ) <= 2^(N ÷ 2)                      # bounded by the half-chain Schmidt rank
    @test argmax(χ) in (c - 1, c, c + 1)               # peaked at the centre
    for b in 1:(c - 1);  @test χ[b] <= χ[b + 1];  end  # rises to the centre …
    for b in c:(N - 2);  @test χ[b] >= χ[b + 1];  end  # … and falls from it
    @test minimum(χ) > 1                               # the full light cone (every bond grew)
end
