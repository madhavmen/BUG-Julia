# tests/sweeps/test_dissipative_t2_g12.jl -- dissipative_cbe_bug_implementation_plan.md, Task 2, the
# G1/G2 tests that need no jump engine, on G = -iH - Q/2 run as the MPO H - (i/2)Q at tau = -im*dt.
#
#   U06  P0 never grows, before and after the recorded compression, and -Δ log P0 = ∫<Q>_c dt
#        (Simpson) along a BUG trajectory; CONTROL: the unnormalised <Q> does not integrate to it.
#        Flagging a nonphysical norm increase apart from the survival rescaling belongs to the jump
#        engine, which does that rescaling: LATER(T4)
#   U09  the norm balance (2) and the conditional-energy balance (3) by finite differences, on the
#        BUG trajectory and on ED; <H>_c against ED directly; CONTROL: <H>_c is not conserved, and
#        (3) without its <H>_c <Q>_c term fails
#   U10  global slope ~3 and one-step slope ~4 against exp(T G), truncation off, on local loss and
#        on collective emission (closure_checks, USER 2026-10-24); CONTROL: order 1 fails the same
#        slope criterion
#   U11  each of tol_basis, tol_root, tol_trunc moves its own error and cost with the other two
#        tight; the retired fixed knobs are refused
#   U16  two-site TDVP and BUG on the same MPO: with Q = 0 both converge to the Hermitian reference,
#        with Q ≠ 0 both converge to ED under the same cap and per-cut discard rule, in the start's
#        U(1) sector
#
# U07, U13-U15 test the jump engine (plan Task 4). U18 (full parallel K/L/S) is optional and not the
# production integrator (plan §1.2), so it waits until that backend is proposed for production.

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

# tolerances well below the time errors measured here
const _G12_BUG = (hermitian = false, order = 3, tol_basis = 1e-10, tol_root = 1e-12,
                  tol_trunc = 1e-12, maxdim = 256)

"Local loss: `_lossy_xx` (loss 1.0 / 0.2 on odd / even sites) and Neel warmed by two TDVP2 steps."
function _g12_local(L; delta = 0.5, gamma = _loss_rates(L))
    set_symmetry!(:U1)
    H, Q, G, Gd, Qd = _lossy_xx(L; delta = delta, gamma = gamma)
    _, psi0 = _xx_warm(L, 2; delta = delta)
    return (; H, Q, G, Gd = Matrix(Gd), Qd = Matrix(Qd), Hd = Matrix(Gd + 0.5im * Qd), psi0)
end

"Collective emission (A2): XXZ in a staggered field, `L_±` at `ka = π/3`, Neel warmed by `H`."
function _g12_emission(L; Jz = 1.0, g = 1.0, k = π / 3)
    set_symmetry!(:U1)
    h = [0.2 * (-1)^j for j in 1:L]
    H = field_mpo_from_terms(xxz_chain(L; J = 1.0, delta = Jz), h)
    Q = collective_emission_mpo(L; gamma_plus = g, gamma_minus = g, k = k)
    Lp, Lm = _emission_jumps(L; gp = g, gm = g, k = k)
    Qd = Matrix(Lp' * Lp + Lm' * Lm)
    Hd = Matrix(dense_heisenberg(L; J = 1.0, delta = Jz) +
                sum(h[j] * _embed(_SZ, j, L) for j in 1:L))
    psi0 = neel_state(L)
    for _ in 1:2
        tdvp2_step!(psi0, H, ComplexF64(-0.2im); maxdim = 64, trunc_thresh = 1e-14)
    end
    return (; H, Q, G = no_jump_mpo(H, Q), Gd = Hd - 0.5im * Qd, Qd, Hd, psi0)
end

_P0(psi) = norm(psi)^2
_qc(psi, Q) = real(mpo_energy(psi, Q)) / _P0(psi)
_slope(e, k) = log2(e[k] / e[k + 1])

"`n` steps `step!` from a copy of `psi0`: the final state and the dense states, the start first."
function _g12_run(step!, psi0, n)
    psi = copy(psi0)
    vs = [dense_state(psi)]
    for _ in 1:n
        step!(psi)
        push!(vs, dense_state(psi))
    end
    return psi, vs
end

"`||psi - exp(T G) v0|| / ||v0||` in the dense basis."
_g12_err(psi, Gd, T, v0) = norm(dense_state(psi) - exp(-1im * T * Gd) * v0) / norm(v0)

@testset "dissipative Task 2 (G1/G2): U06, U09, U10, U11, U16" begin

@testset "U06: P0 never grows, before and after the compression; the hazard integral" begin
    f = _g12_local(8)
    # the identity holds exactly on the exact flow; on the numerical one its residual is the local
    # error, O(dt^4) per step
    dt, n = 0.01, 100
    tau = ComplexF64(-im * dt)
    psi = copy(f.psi0)
    P, q, qraw = [_P0(psi)], [_qc(psi, f.Q)], [real(mpo_energy(psi, f.Q))]
    grow_pre, grow_post, cut = -Inf, -Inf, -Inf
    for _ in 1:n
        pre = copy(psi)
        cbe_bug_step!(pre, f.G, tau; _G12_BUG..., truncate = false)   # same rng: same frames
        cbe_bug_step!(psi, f.G, tau; _G12_BUG...)
        grow_pre = max(grow_pre, _P0(pre) / P[end] - 1)
        grow_post = max(grow_post, _P0(psi) / P[end] - 1)
        cut = max(cut, _P0(psi) / _P0(pre) - 1)
        push!(P, _P0(psi))
        push!(q, _qc(psi, f.Q))
        push!(qraw, real(mpo_energy(psi, f.Q)))
    end
    @test grow_pre <= 1e-12 && grow_post <= 1e-12        # P0(t + dt) <= P0(t), every step
    @test cut <= 1e-12                                   # the compression only removes weight
    @test P[end] < 0.9 * P[1]                            # the loss acts
    # -Δ log P0 over each pair of steps against Simpson's rule on <Q>_c (quadrature error O(dt^5))
    simpson(y, m) = (2dt / 6) * (y[m] + 4y[m + 1] + y[m + 2])
    lp = -log.(P ./ P[1])
    segs = 1:2:(n - 1)
    inc = lp[end] / length(segs)                         # the mean increment per segment
    res = maximum(abs(lp[m + 2] - lp[m] - simpson(q, m)) for m in segs)
    bad = maximum(abs(lp[m + 2] - lp[m] - simpson(qraw, m)) for m in segs)
    v0 = dense_state(f.psi0)
    lex = -log(norm(exp(-1im * n * dt * f.Gd) * v0)^2 / norm(v0)^2)
    @info "U06 contractivity and hazard" grow_pre grow_post cut res bad inc lp[end] lex
    @test res <= 1e-4 * inc                              # -d log P0 / dt = <Q>_c
    @test bad > 0.1 * inc                                # CONTROL: the unnormalised <Q> is not it
    @test abs(lp[end] - lex) <= 1e-4 * lex               # the survival itself, against ED
end

@testset "U09: the norm and conditional-energy balances against ED finite differences" begin
    f = _g12_local(8)
    dt, n = 0.02, 25
    v0 = dense_state(f.psi0)
    _, vm = _g12_run(p -> cbe_bug_step!(p, f.G, ComplexF64(-im * dt); _G12_BUG...), f.psi0, n)
    U = exp(-1im * dt * f.Gd)
    ve = [v0]
    for _ in 1:n
        push!(ve, U * ve[end])
    end
    HQ = f.Hd * f.Qd + f.Qd * f.Hd
    # P0, <H>_c and the right-hand sides of (2) and (3), with (3) also missing its last term
    function obs(v)
        P = real(dot(v, v))
        E = real(dot(v, f.Hd * v)) / P
        q = real(dot(v, f.Qd * v)) / P
        a = -0.5 * real(dot(v, HQ * v)) / P
        return (P = P, E = E, rhsP = -q * P, rhsE = a + E * q, rhsE_noq = a)
    end
    om, oe = obs.(vm), obs.(ve)
    # five-point central differences, error O(dt^4)
    d(y, m) = (y[m - 2] - 8y[m - 1] + 8y[m + 1] - y[m + 2]) / (12dt)
    inner = 3:(n - 1)
    res(o, x, r) = maximum(abs(d(getfield.(o, x), m) - getfield(o[m], r)) for m in inner)
    scaleE = maximum(abs(oe[m].rhsE) for m in inner)
    scaleP = maximum(abs(oe[m].rhsP) for m in inner)
    rE_ed, rE, rP_ed, rP = res(oe, :E, :rhsE), res(om, :E, :rhsE), res(oe, :P, :rhsP), res(om, :P, :rhsP)
    rE_bad = res(om, :E, :rhsE_noq)
    dE = maximum(abs(om[m].E - oe[m].E) for m in eachindex(om))
    drift = abs(om[end].E - om[1].E)
    @info "U09 balances" rE_ed rE rP_ed rP rE_bad scaleE scaleP dE drift
    @test rE_ed <= 1e-6 * scaleE && rP_ed <= 1e-6 * scaleP    # (2), (3) hold on ED to the FD error
    @test rE <= 1e-3 * scaleE && rP <= 1e-3 * scaleP          # ... and along the BUG trajectory
    @test dE <= 1e-4                                          # <H>_c(t) against ED directly
    @test drift > 1e-2                                        # CONTROL: energy is not conserved
    @test rE_bad > 0.1 * scaleE                               # CONTROL: (3) needs <H>_c <Q>_c
end

@testset "U10: order 3 on G without a truncation floor, $name" for name in (:local_loss, :collective)
    f = name === :local_loss ? _g12_local(10) : _g12_emission(10)
    kw = merge(_G12_BUG, (truncate = false, maxdim = 10_000),
               name === :collective ? (closure_checks = true,) : NamedTuple())
    v0 = dense_state(f.psi0)
    T, dts = 0.4, (0.1, 0.05, 0.025)
    glob = Dict(p => Float64[] for p in (1, 3))
    lte = Float64[]
    for dt in dts
        tau = ComplexF64(-im * dt)
        for p in (1, 3)
            psi = copy(f.psi0)
            for _ in 1:round(Int, T / dt)
                cbe_bug_step!(psi, f.G, tau; kw..., order = p)
            end
            push!(glob[p], _g12_err(psi, f.Gd, T, v0))
        end
        psi = copy(f.psi0)
        cbe_bug_step!(psi, f.G, tau; kw...)
        push!(lte, _g12_err(psi, f.Gd, dt, v0))
    end
    @info("U10 $name", dts, glob3 = glob[3], glob1 = glob[1], lte,
          slopes3 = [_slope(glob[3], k) for k in 1:2], slopes1 = [_slope(glob[1], k) for k in 1:2],
          slopes_lte = [_slope(lte, k) for k in 1:2])
    @test glob[3][end] > 1e-10 && lte[end] > 1e-12          # above the floors: slopes are measured
    @test all(_slope(glob[3], k) >= 2.6 for k in 1:2)       # global O(dt^3)
    @test all(_slope(lte, k) >= 3.5 for k in 1:2)           # one step O(dt^4)
    @test _slope(glob[1], 2) < 2.6                          # CONTROL: order 1 fails the criterion
end

@testset "U11: each tolerance controls its own error and cost" begin
    f = _g12_local(10)
    tau = ComplexF64(-0.1im)
    v0 = dense_state(f.psi0)
    tight = (tol_basis = 1e-10, tol_root = 1e-13, tol_trunc = 1e-13)
    function run1(kw)
        psi = copy(f.psi0)
        info = cbe_bug_step!(psi, f.G, tau; hermitian = false, order = 3, maxdim = 256,
                             merge(tight, kw)...)
        return (err = _g12_err(psi, f.Gd, 0.1, v0), lg = info.ledger, rec = info.record)
    end
    rb = [run1((tol_basis = t,)) for t in (1e-4, 1e-6, 1e-8)]
    rr = [run1((tol_root = t,)) for t in (1e-6, 1e-8, 1e-10)]
    rt = [run1((tol_trunc = t,)) for t in (1e-5, 1e-7, 1e-9)]
    row(r) = (err = r.err, r_aug = r.lg.r_aug, expv = r.lg.expv_err, iters = r.lg.expv_iters,
              disc = r.lg.discarded, D = r.rec.D_final)
    @info "U11 tol_basis / tol_root / tol_trunc ladders" basis = row.(rb) root = row.(rr) trunc = row.(rt)
    # tol_basis: the error falls and the frames widen as it tightens; root and compression stay tight
    @test rb[3].err < rb[1].err && all(rb[k + 1].err <= 1.5 * rb[k].err for k in 1:2)
    @test all(rb[k + 1].lg.r_aug >= rb[k].lg.r_aug for k in 1:2) && rb[3].lg.r_aug > rb[1].lg.r_aug
    @test all(r.lg.expv_err <= 1e-13 * (1 + 1e-8) && r.lg.discarded <= 1e-13 * (1 + 1e-8) for r in rb)
    # tol_root: its estimate is within each setting and the Arnoldi work grows as it tightens; the
    # frames do not depend on it
    @test all(rr[k].lg.expv_err <= t * (1 + 1e-8) for (k, t) in enumerate((1e-6, 1e-8, 1e-10)))
    @test all(rr[k + 1].lg.expv_iters >= rr[k].lg.expv_iters for k in 1:2)
    @test rr[3].lg.expv_iters > rr[1].lg.expv_iters
    @test all(r.lg.r_aug == rr[1].lg.r_aug for r in rr)
    # tol_trunc: the discarded norm is within each setting and the final rank falls as it loosens;
    # the frames and the root do not depend on it
    @test all(rt[k].lg.discarded <= t * (1 + 1e-8) for (k, t) in enumerate((1e-5, 1e-7, 1e-9)))
    @test rt[1].lg.discarded > 0 && rt[1].rec.D_final < rt[3].rec.D_final
    @test all(rt[k + 1].rec.D_final >= rt[k].rec.D_final for k in 1:2)
    @test all(r.lg.r_aug == rt[1].lg.r_aug && r.lg.expv_err <= 1e-13 * (1 + 1e-8) for r in rt)
    # no hidden fixed knob: the retired selection and Krylov-depth keywords are refused
    for kw in (:stol_pre, :dex, :dover, :krylov_basis, :krylov_tol, :grow_iters)
        @test_throws MethodError cbe_bug_step!(copy(f.psi0), f.G, tau; hermitian = false,
                                               NamedTuple{(kw,)}((1,))...)
    end
end

@testset "U16: two-site TDVP and BUG on the same generator" begin
    L, T, dts = 10, 0.4, (0.1, 0.05)
    tdvp_kw(w) = (hermitian = false, maxdim = 256, trunc_thresh = 1e-12, maxiter = 40,
                  conv_tol = 1e-12, weight_tol = w)
    for (name, gamma, wt) in ((:hermitian, zeros(L), 1e-12), (:lossy, _loss_rates(L), 1e-8))
        f = _g12_local(L; gamma = gamma)
        v0 = dense_state(f.psi0)
        nup = _spin_charge(argmax(abs.(v0)) - 1, L)
        off = [b for b in 1:(2^L) if _spin_charge(b - 1, L) != nup]
        eb, et, disc_b, disc_t = Float64[], Float64[], 0.0, 0.0
        for dt in dts
            tau = ComplexF64(-im * dt)
            pb, pt = copy(f.psi0), copy(f.psi0)
            for _ in 1:round(Int, T / dt)
                ib = cbe_bug_step!(pb, f.G, tau; _G12_BUG..., tol_trunc = wt)     # the same MPO
                it = tdvp2_step!(pt, f.G, tau; tdvp_kw(wt)...)
                disc_b = max(disc_b, ib.ledger.discarded)
                disc_t = max(disc_t, it.cumulative)
            end
            push!(eb, _g12_err(pb, f.Gd, T, v0))
            push!(et, _g12_err(pt, f.Gd, T, v0))
            for p in (pb, pt)
                @test norm(dense_state(p)[off]) < 1e-10 * norm(p)                 # one U(1) sector
                @test maximum(bond_dims(p)) <= 256                                # the same cap
                # Q = 0 keeps the norm (up to the discards); Q ≠ 0 loses it
                name === :hermitian ? @test(abs(_P0(p) - _P0(f.psi0)) < 1e-8) :
                                      @test(_P0(p) < 0.99 * _P0(f.psi0))
            end
        end
        @info "U16 $name" eb et disc_b disc_t
        @test eb[2] < eb[1] / 6 && et[2] < et[1] / 3          # both converge (orders 3 and 2)
        @test eb[end] < 1e-3 && et[end] < 1e-2
        @test disc_b <= wt * (1 + 1e-8)                        # the per-cut rule, both arms
        @test disc_t <= sqrt(2) * wt * (1 + 1e-8)              # 2(L-1) cuts against BUG's L-1
    end
end

end
