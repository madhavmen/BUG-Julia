# tests/sweeps/test_dissipative_t1.jl -- dissipative_cbe_bug_implementation_plan.md, Task 1 (G0):
# the solver on the no-jump generator G = -iH - Q/2, as the MPO H - (i/2)Q at tau = -im*dt.
#
#   D1.1  no_jump_mpo: on a root space the MPO action is the dense H - (i/2)Q, and its
#         anti-Hermitian part is -(i/2)(Q + Q') = -iQ (the sign that makes the step contract)
#   D1.2  the non-Hermitian root expv (hermitian = false, Arnoldi) against the exact exponential of
#         the dense root matrix U'(H - iQ/2)U; Lanczos on the same root is measurably wrong
#   D1.3  two-site TDVP on the same generator MPO: at L = 2 the step is one exact exponential
#   D1.4  lossless: hermitian = false on a Hermitian H is the Hermitian step, and so is
#         no_jump_mpo(H, Q) at Q = 0
#   D1.5  order = 3 is an explicit dispatch: recorded, frontiers j = 1, 2 swept, the root starts
#         from P psi_n; no dense two-site action outside the frontier leak measurement
#   D1.6  StepRecord keeps D_stage / D_final apart from D_max / D_work
#   D1.7  the local Krylov weight of a contracting no-jump step is relative to the decayed state
#   D1.8  `root_method = :radau2_gmres`: the GMRES stages equal the direct coupled Radau IIA(2)
#         solve, after the same CBE / Krylov / order-3 frontier enrichment
#
# The fixture is the XX chain with site-dependent single-site loss L_j = sqrt(g_j) S^-_j, so
# Q = Σ g_j (S^z_j + 1/2) does not commute with H and G is non-normal.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))

_loss_rates(L) = [isodd(j) ? 1.0 : 0.2 for j in 1:L]

"`(H, Q, G)` as MPOs and the dense `H - (i/2) Q`, with `delta` the XXZ anisotropy."
function _lossy_xx(L; delta = 0.0, gamma = _loss_rates(L))
    q = local_space()
    n = to_concrete(q.Sz + 0.5 * q.I)                       # S^+ S^- = |up><up|
    H = xxz_mpo(L; J = 1.0, delta = delta)
    Q = onsite_mpo(L, [to_concrete(gamma[j] * n) for j in 1:L])
    Qd = sum(gamma[j] * (_embed(_SZ, j, L) + 0.5 * I) for j in 1:L)
    Hd = dense_heisenberg(L; J = 1.0, delta = delta)
    return H, Q, no_jump_mpo(H, Q), Hd - 0.5im * Qd, Qd
end

@testset "dissipative Task 1: the solver on G = -iH - Q/2" begin

L = 6
_, psi0 = _xx_warm(L, 2)
H, Q, G, Gd, Qd = _lossy_xx(L)
v0 = dense_state(psi0)
tau = ComplexF64(-0.5im)

@testset "D1.1: no_jump_mpo is H - (i/2)Q on the root space" begin
    fr = Ref{Any}(nothing)
    cbe_bug_step!(copy(psi0), G, tau; hermitian = false, tol_basis = 1e-8, tol_root = 1e-12,
                  truncate = false, frames_report = fr)
    U = _root_basis(psi0, G, fr[]; mmax = 10)
    B = RC._root_generator(G, fr[])
    S0 = RC._project_root(psi0, fr[])
    X, Y = (RC._randomize_like(S0, Random.MersenneTwister(s)) for s in (1, 2))
    dX = dense_state(RC._frontier_mps(psi0, fr[], X))
    dY = dense_state(RC._frontier_mps(psi0, fr[], Y))
    dBX = dense_state(RC._frontier_mps(psi0, fr[], RC.apply_zero_site(B, X)))
    @test norm(dBX - U * ((U' * Gd * U) * (U' * dX))) < 1e-10 * norm(dX)
    # <X|BY> - conj(<Y|BX>) = <X|(B - B')|Y> = -i <X|Q|Y> for X, Y in the root space
    skew = tensor_inner(X, RC.apply_zero_site(B, Y)) - conj(tensor_inner(Y, RC.apply_zero_site(B, X)))
    @test abs(skew - (-1im) * (dX' * Qd * dY)) < 1e-10 * norm(dX) * norm(dY)
    @test abs(dX' * Qd * dY) > 1e-3 * norm(dX) * norm(dY)    # the check is not vacuous
end

@testset "D1.2: non-Hermitian root expv against the exact root exponential" begin
    rows = Any[]
    for tolx in (1e-6, 1e-10)
        fr, rr = Ref{Any}(nothing), Ref{Any}(nothing)
        psi = copy(psi0)
        cbe_bug_step!(psi, G, tau; hermitian = false, tol_basis = 1e-8, tol_root = tolx,
                      truncate = false, frames_report = fr, root_report = rr)
        U = _root_basis(psi0, G, fr[]; mmax = 10)
        @test norm(v0 - U * (U' * v0)) < 1e-12 * norm(v0)              # P psi_n = psi_n
        ref = U * (exp(tau * (U' * Gd * U)) * (U' * v0))
        err = norm(dense_state(psi) - ref) / norm(v0)
        est = rr[].solve.err
        @test est <= tolx * (1 + 1e-8)
        @test err <= 10 * tolx + 1e-12
        @test norm(ref) < 0.99 * norm(v0)                                # the generator contracts
        push!(rows, (tol = tolx, est = est, err = err, dim = size(U, 2)))
    end
    # CONTROL: the same root solved by Lanczos (hermitian = true) is wrong by far more than tol_root
    fr = Ref{Any}(nothing)
    psiH = copy(psi0)
    cbe_bug_step!(psiH, G, tau; hermitian = true, tol_basis = 1e-8, tol_root = 1e-10,
                  truncate = false, frames_report = fr)
    U = _root_basis(psi0, G, fr[]; mmax = 10)
    errH = norm(dense_state(psiH) - U * (exp(tau * (U' * Gd * U)) * (U' * v0))) / norm(v0)
    @test errH > 1e-3
    @info "D1.2 non-Hermitian root vs exp(tau U'GU)" rows lanczos_err = errH
end

@testset "D1.3: two-site TDVP on the same generator, exact at L = 2" begin
    kw = (maxdim = 8, trunc_thresh = 1e-14, maxiter = 30, tol = 1e-15)
    _, _, G2, Gd2, _ = _lossy_xx(2; delta = 0.5)
    err(herm, k) = (psi = neel_state(2); w0 = dense_state(psi);
                    tdvp2_step!(psi, G2, tau; hermitian = herm, merge(kw, (maxiter = k,))...);
                    norm(dense_state(psi) - exp(tau * Gd2) * w0) / norm(w0))
    @test err(false, 30) < 1e-12
    # CONTROL. At maxiter = 30 Lanczos is accurate here too (measured 6e-15): its recurrence
    # `G V = V T + b v e'` holds for any `alpha`, so `V exp(tau T) e1` is a degree-29 polynomial in
    # G, which converges at ||tau G|| ~ 1 although `real(alpha)` drops the loss. At maxiter = 2
    # Arnoldi spans the whole two-dimensional N = 1 sector and is exact, while Lanczos's T is not
    # the projected G (~2e-2 by hand).
    @test err(false, 2) < 1e-12
    @test err(true, 2) > 1e-3
end

@testset "D1.4: lossless runs are the Hermitian step" begin
    kw = (order = 3, exact = true, tol_basis = 1e-10, tol_trunc = 1e-12, tol_root = 1e-13)
    a, b = copy(psi0), copy(psi0)
    cbe_bug_step!(a, H, tau; hermitian = true, kw...)
    cbe_bug_step!(b, H, tau; hermitian = false, kw...)
    @test norm(dense_state(a) - dense_state(b)) < 1e-9 * norm(v0)
    _, _, G0, _, _ = _lossy_xx(L; gamma = zeros(L))
    c = copy(psi0)
    cbe_bug_step!(c, G0, tau; hermitian = false, kw...)
    @test norm(dense_state(c) - dense_state(b)) < 1e-8 * norm(v0)
    @test abs(norm(dense_state(c)) - norm(v0)) < 1e-9 * norm(v0)     # Q = 0 keeps the norm
end

@testset "D1.5: order = 3 dispatch and no dense two-site CBE action" begin
    kw = (hermitian = false, tol_basis = 1e-8, tol_root = 1e-12, maxdim = 64)
    # from the Neel state: psi0 is full rank (its root space is the whole N = 3 sector), so the
    # half-sweep CBE has no room and would stream nothing
    i1 = cbe_bug_step!(neel_state(L), G, tau; order = 1, kw...)
    @test i1.record.order == 1 && i1.record.closure_policy === :none
    @test i1.ledger.n_h2_leak == 0                   # the half-sweep CBE streams, never H*Theta
    @test i1.record.counters.n_stream > 0 && i1.record.counters.n_qr > 0
    for fold in (false, true)
        i3 = cbe_bug_step!(copy(psi0), G, tau; order = 3, fold_omega = fold, kw...)
        @test i3.record.order == 3 && i3.record.closure_policy === :dd_qr_streamed
        @test Set(e.frontier for e in i3.ledger.closure_log) == Set([1, 2])
        @test i3.ledger.proj_err < 1e-12             # the root starts from P psi_n = psi_n
        # the frontier leak test is the only place a two-site action is formed, and only
        # with `fold_omega = false` (the default)
        fold ? (@test i3.ledger.n_h2_leak == 0) : (@test i3.ledger.n_h2_leak > 0)
    end
end

@testset "D1.6: temporary, final and allowed ranks are recorded apart" begin
    md = 6
    r = cbe_bug_step!(copy(psi0), G, tau; hermitian = false, order = 3, tol_basis = 1e-8,
                      maxdim = md).record
    @test r.D_max == md
    @test r.D_work == work_ceiling(md, r.D_old)
    @test r.D_stage <= r.D_work && r.D_final <= r.D_max
end

@testset "D1.7: local Krylov weights under loss are relative to the decayed state" begin
    k = 3
    Hh = ComplexF64[0.3 0.7 0 0; 0.7 -0.2 0.5 0; 0 0.5 0.1 0.4; 0 0 0.4 0]   # Hermitian Hessenberg
    Hl = Hh - 0.5im * Diagonal([1.0, 0.2, 1.0, 0.2])                         # H - (i/2) Q
    function expA(Hm)
        A = zeros(ComplexF64, k + 1, k + 1)
        A[1:k, 1:k] .= tau .* Hm[1:k, 1:k]
        A[k + 1, k] = tau * Hm[k + 1, k]
        return exp(A)
    end
    E = expA(Hh)
    @test RC._kry_weight(Hh, k, tau) ≈ abs(E[k + 1, 1]) rtol = 1e-12            # unitary: unchanged
    E = expA(Hl)
    w = RC._kry_weight(Hl, k, tau)
    @test w ≈ abs(E[k + 1, 1]) / norm(E[1:k, 1]) rtol = 1e-10
    @test w > 1.05 * abs(E[k + 1, 1])                         # not against the start norm
    @test RC._kry_weight(Hl, k, tau; state_seed = false) ≈ abs(E[k + 1, 1]) rtol = 1e-10
end

@testset "D1.8: Radau IIA(2) + GMRES root against the direct coupled stage solve" begin
    A = [5 / 12 -1 / 12; 3 / 4 1 / 4]
    for order in (1, 3)
        fr, rr = Ref{Any}(nothing), Ref{Any}(nothing)
        psi = copy(psi0)
        cbe_bug_step!(psi, G, tau; order = order, hermitian = false, root_method = :radau2_gmres,
                      tol_basis = 1e-8, tol_root = 1e-10, truncate = false,
                      frames_report = fr, root_report = rr)
        U = _root_basis(psi0, G, fr[]; mmax = 10)
        M = U' * Gd * U
        s0 = U' * v0
        n = length(s0)
        Y = (Matrix{ComplexF64}(I, 2n, 2n) - tau * kron(A, M)) \ vcat(s0, s0)
        sol = rr[].solve
        @test sol.err <= 1e-10 * (1 + 1e-8)                      # the TRUE stage residual
        for i in 1:2
            dYi = dense_state(RC._frontier_mps(psi0, fr[], sol.stages[i]))
            @test norm(dYi - U * Y[((i - 1) * n + 1):(i * n)]) < 1e-8 * norm(v0)
        end
        y2 = U * Y[(n + 1):(2n)]
        @test norm(dense_state(psi) - y2) < 1e-8 * norm(v0)      # stiffly accurate: S1 = Y_2
        @test norm(y2 - U * (exp(tau * M) * s0)) > 1e-6 * norm(v0)   # a Radau step, not exp
        if order == 1
            # converged, the stability-function form on the Arnoldi space is the same step
            q = copy(psi0)
            cbe_bug_step!(q, G, tau; hermitian = false, root_method = :radau2, tol_basis = 1e-8,
                          tol_root = 1e-12, truncate = false)
            @test norm(dense_state(q) - y2) < 1e-8 * norm(v0)
        end
        @info "D1.8 Radau IIA(2)+GMRES" order iterations = sol.iterations cycles = sol.cycles err = sol.err
    end
end

end
