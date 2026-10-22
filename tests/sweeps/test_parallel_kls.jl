# tests/sweeps/test_parallel_kls.jl -- updated_cbe_high_order_parallel.md, Task 6: the fully
# parallel K/L/S integrator after the preliminary CBE half-sweeps (`parallel_kls_step!`).
#
#   T6.2  K(0), L(0), S(0) are the original state in the preliminary frames; the K/L operators
#         are Hermitian and K(h) is the exact exponential of the projected K problem
#   T6.3  the omitted new-new block: the kernel's leading term `tau U_+' H Y_n V_+` matches the
#         exact block to O(h^2); the four-block core has no new-new component; the fallback is
#         the augmented root Galerkin step in the grown frames (on the state's own Schmidt
#         frames: real preliminary frames leave no new direction at these sizes)
#   T6.4  |00>, H = J XX: unenriched frames miss the O(h) DD amplitude even with K and L; the
#         preliminary CBE finds it on both sides and the step is exact (and the U(1) analogue)
#   T6.5  the three schedules return one state (the step, and the kernel where U_+, V_+ exist);
#         the concurrent schedule is reproducible; `parallel = :full`
#   orders 1..5 of the K/L/S recurrence on its own (local errors, as UT3)

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))

"The step's root integrator at `tol`, as `cbe_bug_step!` hands it to the K/L/S stages."
_stage(tau; tol = 1e-12) = (apply, x, rep) -> BUGJulia.BondUpdateBUG.expv(apply, tau, x;
    maxiter = RC._ROOT_MAXITER, tol = 1e-15, conv_tol = tol, substeps = 0, report = rep)

"`psi`'s frames with the one-site tensor `X` at site `at` (`c` for K, `c+1` for L)."
function _site_mps(psi, fr, X, at)
    T = Any[i == at ? X : i <= fr.c ? fr.W[i] : fr.Z[i - 1] for i in 1:length(psi)]
    return SymMPS(Any[RC._relink(T[i], psi[i].inds[1].itags, psi[i].inds[3].itags)
                      for i in eachindex(T)], at)
end

"`fr` with the root frames replaced."
_with_root(fr, Wc, Zc) = (W = copy(fr.W); Z = copy(fr.Z); W[fr.c] = Wc; Z[fr.c] = Zc;
                          RC.KLFrames(W, Z, fr.c, fr.lenv, fr.renv, 0))

"The preliminary frames of a real order-`p` step from the warmed XX chain."
function _prelim(L, steps, tau; order = 1, tol_basis = 1e-8)
    mpo, psi0 = _xx_warm(L, steps)
    fr = Ref{Any}(nothing)
    cbe_bug_step!(copy(psi0), mpo, tau; order = order, tol_basis = tol_basis, tol_root = 1e-12,
                  truncate = false, frames_report = fr)
    return mpo, psi0, fr[]
end

@testset "HQ Task 6: parallel K/L/S" begin

@testset "T6.2: initial values, adjoints, the K and L solves" begin
    L, tau = 8, ComplexF64(-0.2im)
    mpo, psi0, fr = _prelim(L, 2, tau; tol_basis = 1e-4)
    c, H, v0 = fr.c, dense_heisenberg(L; J = 1.0, delta = 0.0), dense_state(psi0)
    n0 = norm(v0)
    S0 = RC._project_root(psi0, fr)
    st = RC._kls_start(psi0, mpo, fr.W, fr.Z, c, fr.lenv, fr.renv, _stage(tau), 0.0, :sequential)
    @test norm(dense_state(psi0) - v0) == 0                     # psi_n is not touched
    # K(0) = Y_n V_p and L(0) = U_p' Y_n ARE the state; U_p' K(0) = S(0) = L(0) V_p'
    @test norm(dense_state(_site_mps(psi0, fr, st.K0, c)) - v0) < 1e-12 * n0
    @test norm(dense_state(_site_mps(psi0, fr, st.L0, c + 1)) - v0) < 1e-12 * n0
    @test norm(contract(fr.W[c]', (1, 2), st.K0, (1, 2)) - S0) < 1e-12 * n0
    @test norm(contract(st.L0, (2, 3), fr.Z[c]', (2, 3)) - S0) < 1e-12 * n0
    for (op, X0, at) in ((st.HK, st.K0, c), (st.HL, st.L0, c + 1))
        X, Y = (RC._randomize_like(X0, Random.MersenneTwister(s)) for s in (1, 2))
        @test abs(tensor_inner(X, RC.apply_one_site(op, Y)) -
                  conj(tensor_inner(Y, RC.apply_one_site(op, X)))) < 1e-11 * norm(X) * norm(Y)
        # the stage is exp(tau Q'HQ) on its own one-site space, which contains psi_n
        Q = _span_basis(X0, x -> RC.apply_one_site(op.h, x),
                        X -> dense_state(_site_mps(psi0, fr, X, at)); mmax = 10)
        @test norm(v0 - Q * (Q' * v0)) < 1e-12 * n0
        Xh = first(RC._join(at == c ? st.K : st.L))
        @test norm(dense_state(_site_mps(psi0, fr, Xh, at)) -
                   Q * (exp(tau * (Q' * H * Q)) * (Q' * v0))) < 1e-10 * n0
    end
    Kh, Lh = first(st.K), first(st.L)
    @test leg_dim(Kh, 3) == leg_dim(fr.Z[c], 1) && leg_dim(Lh, 1) == leg_dim(fr.W[c], 3)
end

# ⛔ NOT the preliminary frames of a real step. At L <= 10 the half-sweeps' local Krylov fills the
# one-site spaces K and L live in, so K(h), L(h) have no direction outside them (hq16a 17422381:
# `Up`/`Vp` were `nothing`; T6.5's ledger shows n_new = 0) and there is no omitted block to measure.
# The state's own Schmidt frames are strictly smaller than the one-site spaces at this size, so
# U_+ and V_+ exist at O(h) on both sides -- the case the kernel's claims are about.
"The state's own (unenriched) frames at the root bond of the warmed XX(Z) chain."
function _schmidt_frames(L, steps; kw...)
    mpo, psi0 = _xx_warm(L, steps; kw...)
    return mpo, psi0, RC.frames_from_state(psi0, mpo, L ÷ 2)
end

# ⛔ Not the XX chain, and not h = 0.2. The XX Neel state is Gaussian and its exact new-new block
# has no O(h) term (hq17d 17429012: ||M11|| = 5e-14). On a truncated XXZ state the O(h) term is
# real but small: at h = 0.2 the h^2 remainder is 35-1600x the leading term (hq19b 17429851,
# slopes 1.99). The ratio falls as h, so the leading term dominates only at small h; h = 1e-3
# puts it near 0.2 on this fixture (Delta = 0.5, D = 4: ratio 35 at h = 0.2).
@testset "T6.3: the omitted new-new block" begin
    L, tau, delta = 8, ComplexF64(-1e-3im), 0.5
    mpo, psi0, fr = _schmidt_frames(L, 4; delta = delta, maxdim = 4)
    c, H, v0 = fr.c, dense_heisenberg(L; J = 1.0, delta = delta), dense_state(psi0)
    n0 = norm(v0)
    S0 = RC._project_root(psi0, fr)
    r = RC._parallel_kls(psi0, mpo, fr, S0, tau, _stage(tau); nrm = n0)
    @test r.Up !== nothing && r.Vp !== nothing                   # both sides found directions
    @test r.account.n_new_l > 0 && r.account.n_new_r > 0
    @test !r.account.fallback
    @test r.account.missing ≈ norm(r.M11) / n0
    # S11 = U_+' Y(t+h) V_+ = h U_+' F(Y_n) V_+ + O(h^2) for U_+, V_+ fixed: the remainder after
    # the kernel's leading term falls as h^2
    frp = _with_root(fr, r.Up, r.Vp)
    Bp = RC._root_generator(mpo, frp)
    Qp = _span_basis(r.M11, x -> RC.apply_zero_site(Bp, x),
                     X -> dense_state(RC._frontier_mps(psi0, frp, X)); mmax = 6)
    dM = dense_state(RC._frontier_mps(psi0, frp, r.M11))
    rems = [norm(Qp * (Qp' * (exp((tau / s) * H) * v0)) - dM / s) for s in (1, 2, 4, 8)]
    slopes = log2.(rems[1:(end - 1)] ./ rems[2:end])
    @info "T6.3 missing block: ||M11||/||psi|| $(r.account.missing), remainders $(rems), " *
          "slopes $(slopes) (want 2)"
    @test all(>=(1.8), slopes)
    @test rems[1] < norm(dM)                                     # the leading term dominates
    # the four-block core: nothing in the new-new block
    frn = _with_root(fr, r.W, r.Z)
    @test norm(Qp' * dense_state(RC._frontier_mps(psi0, frn, r.S1))) < 1e-12 * n0
    # an over-limit block is never accepted: the augmented root Galerkin step in the grown frames
    rf = RC._parallel_kls(psi0, mpo, fr, S0, tau, _stage(tau); nrm = n0, tol_missing = 0.0)
    @test rf.account.fallback
    Un = _root_basis(psi0, mpo, frn; mmax = 10)
    @test norm(dense_state(RC._frontier_mps(psi0, frn, rf.S1)) -
               Un * (exp(tau * (Un' * H * Un)) * (Un' * v0))) < 1e-10 * n0
end

@testset "T6.4: |00>, H = J XX -- the DD amplitude needs the preliminary CBE" begin
    J, h = 0.7, 0.3
    tau = ComplexF64(-1im * h)
    set_symmetry!(:Z2)
    # Z2 basis: `Z` is the flip (tfim.jl), so `-J Z Z` at zero field is J sigma^x sigma^x up to
    # its sign, and |++> is |00>: the exact state is cos(Jh)|00> -/+ i sin(Jh)|11>
    mpo = tfim_mpo(2; J = J, h = 0.0)
    psi0 = ising_polarised_state(2)
    pp, mm = ising_product_state([:plus, :plus]), ising_product_state([:minus, :minus])
    fr0 = RC.frames_from_state(psi0, mpo, 1)                     # unenriched U = V = |0>
    r0 = RC._parallel_kls(psi0, mpo, fr0, RC._project_root(psi0, fr0), tau, _stage(tau))
    @test r0.Up === nothing && r0.Vp === nothing                 # K and L see no new direction
    out0 = RC._frontier_mps(psi0, _with_root(fr0, r0.W, r0.Z), r0.S1)
    @test abs(overlap(mm, out0)) < 1e-14                         # the O(h) amplitude is missed
    psi = copy(psi0)
    info = parallel_kls_step!(psi, mpo, tau; tol_basis = 1e-12, tol_root = 1e-14, truncate = false)
    @test abs(abs(overlap(pp, psi)) - cos(J * h)) < 1e-12
    @test abs(abs(overlap(mm, psi)) - sin(J * h)) < 1e-12
    @test info.ledger.kls.n_new_l == 0 && info.ledger.kls.n_new_r == 0   # complete frames
    # the U(1) analogue: |up,down> under XX, against the dense exponential
    set_symmetry!(:U1)
    mpo = xxz_mpo(2; J = 1.0, delta = 0.0)
    psi0 = neel_state(2)
    want = exp(tau * dense_heisenberg(2; J = 1.0, delta = 0.0)) * dense_state(psi0)
    psi = copy(psi0)
    parallel_kls_step!(psi, mpo, tau; tol_basis = 1e-12, tol_root = 1e-14, truncate = false)
    @test norm(dense_state(psi) - want) < 1e-12
end

@testset "T6.5: three schedules, one state" begin
    mpo, psi0 = _xx_warm(10, 3)
    tau = ComplexF64(-0.2im)
    step(s) = (psi = copy(psi0);
               info = parallel_kls_step!(psi, mpo, tau; schedule = s, order = 2,
                                         tol_basis = 1e-6, tol_root = 1e-12);
               (dense_state(psi), info))
    outs = [step(s) for s in RC.KLS_SCHEDULES]
    for k in 2:length(outs)
        @test norm(outs[k][1] - outs[1][1]) < 1e-10
    end
    @test [o[2].ledger.kls.schedule for o in outs] == collect(RC.KLS_SCHEDULES)
    @test all(o -> o[2].record.parallel === :full, outs)
    @test norm(first(step(:concurrent)) - outs[end][1]) < 1e-12  # reproducible
    # the plan's `parallel = :full`: concurrent half-sweeps (their own gauge path, so 1e-10 as
    # in tests/rsvd_cbe/) and the concurrent K/L/S schedule
    psi = copy(psi0)
    info = cbe_bug_step!(psi, mpo, tau; parallel = :full, order = 2, tol_basis = 1e-6,
                         tol_root = 1e-12)
    @test info.record.parallel === :full && info.ledger.kls.schedule === :concurrent
    @test norm(dense_state(psi) - outs[end][1]) < 1e-10
    @info "T6.5 schedules" Threads.nthreads() [o[2].ledger.kls for o in outs]
    # the step above finds no new direction at this size (see T6.3), so the schedules are also
    # compared where the new blocks exist: the kernel on the state's own frames
    mpo8, p8, fr8 = _schmidt_frames(8, 2)
    S8 = RC._project_root(p8, fr8)
    ks = [RC._parallel_kls(p8, mpo8, fr8, S8, tau, _stage(tau); schedule = s)
          for s in RC.KLS_SCHEDULES]
    @test all(k -> k.account.n_new_l > 0 && k.account.n_new_r > 0, ks)
    # as states: each schedule orthogonalises its own U_+, V_+
    vs = [dense_state(RC._frontier_mps(p8, _with_root(fr8, k.W, k.Z), k.S1)) for k in ks]
    for v in vs[2:end]
        @test norm(v - vs[1]) < 1e-10 * norm(vs[1])
    end
end

@testset "local order of the K/L/S recurrence, p = $p" for p in 1:5
    errs, floors = _dw_errors(p, (0.4, 0.2, 0.1); kls = :sequential)
    slope = _finest_order(errs, floors)
    @info "K/L/S p=$p: local errors $(errs), floors $(floors), slope $(slope) (want $(p + 1))"
    @test slope !== nothing && slope >= p + 0.6
end

end # HQ Task 6
