# tests/sweeps/test_dissipative_u18.jl -- dissipative_cbe_bug_implementation_plan.md, Task 2 U18
# (optional; the prerequisite of Task 7's efficiency/parallel subset): the fully parallel K/L/S step
# and the concurrent half-sweeps on G = -iH - Q/2, run as the MPO H - (i/2)Q at tau = -im*dt.
# test_parallel_kls.jl is the Hermitian original of U18a-c.
#
#   U18a  K, L and S integrate the same complex generator: K(0), L(0) are the state, K(h), L(h) are
#         exp(tau P'GP) on their one-site spaces; CONTROL: the stage operators are not Hermitian
#   U18b  block assembly and the omitted new-new block on G: the leading term tau U_+' G Theta V_+
#         leaves an O(h^2) remainder, the four-block core has no new-new component, and an
#         over-limit block falls back to the augmented Galerkin step in the grown frames
#   U18c  independent workspaces: the three K/L/S schedules return one state, the concurrent one
#         reproducibly; `parallel = :half_sweeps` returns the serial step's state
#   U18d  observed order 3 (global) and 4 (one step) of `parallel = :full` from the domain wall, on
#         local loss and collective emission (as U10), at the standard step's accuracy

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
include(joinpath(@__DIR__, "..", "common", "dissipative_fixtures.jl"))

"The state's own (unenriched) frames at the root bond of a truncated lossy XXZ chain."
function _u18_schmidt(L; delta = 0.5)
    _, psi0 = _xx_warm(L, 4; delta = delta, maxdim = 4)
    _, _, G, Gd, _ = _lossy_xx(L; delta = delta)
    return G, Matrix(Gd), psi0, RC.frames_from_state(psi0, G, L ÷ 2)
end

@testset "dissipative U18: parallel K/L/S and concurrent half-sweeps on G" begin

@testset "U18a: the K and L stages integrate the complex generator" begin
    f = _g12_local(8)
    tau = ComplexF64(-0.2im)
    fr = Ref{Any}(nothing)
    cbe_bug_step!(copy(f.psi0), f.G, tau; hermitian = false, tol_basis = 1e-4, tol_root = 1e-12,
                  truncate = false, frames_report = fr)
    fr = fr[]
    c, v0 = fr.c, dense_state(f.psi0)
    n0 = norm(v0)
    S0 = RC._project_root(f.psi0, fr)
    st = RC._kls_start(f.psi0, f.G, fr.W, fr.Z, c, fr.lenv, fr.renv,
                       _stage(tau; hermitian = false), 0.0, :sequential)
    @test norm(dense_state(_site_mps(f.psi0, fr, st.K0, c)) - v0) < 1e-12 * n0
    @test norm(dense_state(_site_mps(f.psi0, fr, st.L0, c + 1)) - v0) < 1e-12 * n0
    @test norm(contract(fr.W[c]', (1, 2), st.K0, (1, 2)) - S0) < 1e-12 * n0
    @test norm(contract(st.L0, (2, 3), fr.Z[c]', (2, 3)) - S0) < 1e-12 * n0
    for (op, X0, at) in ((st.HK, st.K0, c), (st.HL, st.L0, c + 1))
        X, Y = (RC._randomize_like(X0, Random.MersenneTwister(s)) for s in (1, 2))
        # CONTROL: <X, G Y> - conj(<Y, G X>) = -i <X, Q Y>, so Lanczos would be wrong here
        @test abs(tensor_inner(X, RC.apply_one_site(op, Y)) -
                  conj(tensor_inner(Y, RC.apply_one_site(op, X)))) > 1e-6 * norm(X) * norm(Y)
        P = _span_basis(X0, x -> RC.apply_one_site(op.h, x),
                        X -> dense_state(_site_mps(f.psi0, fr, X, at)); mmax = 10)
        @test norm(v0 - P * (P' * v0)) < 1e-12 * n0
        Xh = first(RC._join(at == c ? st.K : st.L))
        @test norm(dense_state(_site_mps(f.psi0, fr, Xh, at)) -
                   P * (exp(tau * (P' * f.Gd * P)) * (P' * v0))) < 1e-10 * n0
    end
end

# As T6.3: the Schmidt frames leave new directions on both sides, and h = 1e-3 puts the leading
# term above the h^2 remainder.
@testset "U18b: block assembly and the omitted new-new block on G" begin
    L, tau = 8, ComplexF64(-1e-3im)
    G, Gd, psi0, fr = _u18_schmidt(L)
    c, v0 = fr.c, dense_state(psi0)
    n0 = norm(v0)
    S0 = RC._project_root(psi0, fr)
    stg = _stage(tau; hermitian = false)
    r = RC._parallel_kls(psi0, G, fr, S0, tau, stg; nrm = n0)
    @test r.account.n_new_l > 0 && r.account.n_new_r > 0
    @test !r.account.fallback
    @test r.account.missing ≈ norm(r.M11) / n0
    frp = _with_root(fr, r.Up, r.Vp)
    Bp = RC._root_generator(G, frp)
    Qp = _span_basis(r.M11, x -> RC.apply_zero_site(Bp, x),
                     X -> dense_state(RC._frontier_mps(psi0, frp, X)); mmax = 6)
    dM = dense_state(RC._frontier_mps(psi0, frp, r.M11))
    rems = [norm(Qp * (Qp' * (exp((tau / s) * Gd) * v0)) - dM / s) for s in (1, 2, 4, 8)]
    slopes = log2.(rems[1:(end - 1)] ./ rems[2:end])
    @info "U18b missing block: ||M11||/||psi|| $(r.account.missing), remainders $(rems), " *
          "slopes $(slopes) (want 2)"
    @test all(>=(1.8), slopes)
    @test rems[1] < norm(dM)
    frn = _with_root(fr, r.W, r.Z)
    @test norm(Qp' * dense_state(RC._frontier_mps(psi0, frn, r.S1))) < 1e-12 * n0
    rf = RC._parallel_kls(psi0, G, fr, S0, tau, stg; nrm = n0, tol_missing = 0.0)
    @test rf.account.fallback
    Un = _root_basis(psi0, G, frn; mmax = 10)
    @test norm(dense_state(RC._frontier_mps(psi0, frn, rf.S1)) -
               Un * (exp(tau * (Un' * Gd * Un)) * (Un' * v0))) < 1e-10 * n0
end

@testset "U18c: independent workspaces -- schedules and concurrent half-sweeps" begin
    f = _g12_local(10)
    tau = ComplexF64(-0.2im)
    kw = (hermitian = false, order = 3, tol_basis = 1e-6, tol_root = 1e-12)
    step(; skw...) = (psi = copy(f.psi0); info = cbe_bug_step!(psi, f.G, tau; kw..., skw...);
                      (dense_state(psi), info))
    outs = [step(kls = s) for s in RC.KLS_SCHEDULES]
    for k in 2:length(outs)
        @test norm(outs[k][1] - outs[1][1]) < 1e-10
    end
    @test norm(first(step(kls = :concurrent)) - outs[end][1]) < 1e-12   # reproducible
    vs, is = step()
    vh, ih = step(parallel = :half_sweeps)
    vf, iff = step(parallel = :full)
    @test ih.record.parallel === :half_sweeps && iff.record.parallel === :full
    @test norm(vh - vs) < 1e-10                    # concurrent half-sweeps: the serial step
    @test norm(vf - outs[end][1]) < 1e-10          # :full = concurrent half-sweeps + K/L/S
    k = iff.ledger.kls
    @info "U18c" Threads.nthreads() k
    # no new direction outside the root frames: the four-block core is the standard root solve
    k.n_new_l == 0 && k.n_new_r == 0 && @test norm(vf - vs) < 1e-10
    G, _, p8, fr8 = _u18_schmidt(8)
    S8 = RC._project_root(p8, fr8)
    ks = [RC._parallel_kls(p8, G, fr8, S8, tau, _stage(tau; hermitian = false); schedule = s)
          for s in RC.KLS_SCHEDULES]
    @test all(k -> k.account.n_new_l > 0 && k.account.n_new_r > 0, ks)
    ws = [dense_state(RC._frontier_mps(p8, _with_root(fr8, k.W, k.Z), k.S1)) for k in ks]
    for w in ws[2:end]
        @test norm(w - ws[1]) < 1e-10 * norm(ws[1])
    end
end

# As U10, with `parallel = :full`; the standard step's errors from the same start are the accuracy
# the parallel integrator has to match.
@testset "U18d: order 3 of the K/L/S step on G, $name" for name in (:local_loss, :collective)
    coll = name === :collective
    f = coll ? _emission_model(10; start = [j <= 5 ? :up : :down for j in 1:10]) :
               _g12_local(10; start = :dw)
    kw = (hermitian = false, tol_basis = 1e-12, tol_trunc = 1e-14, tol_root = 1e-14,
          maxdim = 256, exact = true, closure_checks = coll)
    dts = coll ? (0.1, 0.05, 0.025) : (0.4, 0.2, 0.1)
    fk = coll ? 3 : 0
    kf = merge(kw, (parallel = :full,))
    eg, fg = _g12_errors(f, 3, dts, kf; T = 0.8, first_keep = fk)
    el, fl = _g12_errors(f, 3, dts, kf; first_keep = fk)
    es, fs = _g12_errors(f, 3, dts, kw; T = 0.8, first_keep = fk)
    keep = fk > 0 ? global_krylov_states(f.G, f.psi0, fk) : Any[]
    acc = cbe_bug_step!(copy(f.psi0), f.G, ComplexF64(-1im * dts[1]); kf..., order = 3,
                        keep_states = keep).ledger.kls
    sg, sl = _finest_order(eg, fg), _finest_order(el, fl)
    @info "U18d $name: :full global $(eg) floors $(fg) p_obs $(_observed_orders(eg, fg)); one step " *
          "$(el) p_obs $(_observed_orders(el, fl)); serial global $(es); first step K/L/S $(acc)"
    @test sg !== nothing && sg >= 3 - 0.4
    @test sl !== nothing && sl >= 4 - 0.4
    @test all(eg[k] <= 1.5 * es[k] + 10 * max(fg[k], fs[k]) for k in eachindex(eg))
end

end
