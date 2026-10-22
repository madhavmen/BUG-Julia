# tests/sweeps/test_spin_xx.jl -- updated_cbe_high_order_parallel.md, Task 7: the two U(1) XX
# models (s = 1/2, d = 2 and s = 3, d = 7) on one exchange convention, and their references.
#
#   T7.0  the models: d = 2s + 1 sectors, s = 1/2 is `xxz_mpo(delta = 0)`, <H> = 0 and the exact
#         variance on product states (the normalisation of the exchange at both spins)
#   T7.1  s = 1/2: the closed-form open-chain propagator, continuity, energy and particle number;
#         a converged order-4 TDVP2 run at L = 16 to J t = 8 against it
#   T7.2  s = 3 against sector ED at L = 4 and 6 (profile, centre current, conserved quantities)
#   T7.3  the short-time coefficient <S^z_{L/2}> = s - J^2 s^2 t^2 (1/4 and 9), MPS and ED
#
# The reference solver is order-4 TDVP2 (`tdvp2_composed_step!`), not the integrator under study.
# The s = 3 references are never analytical: s = 3 XX is not free-fermionic.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "common", "spin_ed.jl"))

const _TD = (maxdim = 512, trunc_thresh = 1e-14)

"`n` order-4 TDVP2 steps of `dt` in place; `f(psi, k)` after step `k`."
function _evolve!(psi, mpo, dt, n, f = (_, _) -> nothing)
    for k in 1:n
        tdvp2_composed_step!(psi, mpo, ComplexF64(-1im * dt); order = 4, _TD...)
        f(psi, k)
    end
    return psi
end

"""
`<H_s^2>` on the product state `2 m_i = twoms[i]` (`J = 1`): each bond's two flips give states
orthogonal to every other bond's, so the variance is the sum of the squared flip amplitudes,
`|<m+1|S^+|m>|^2 = s(s+1) - m(m+1)`.
"""
function _product_variance(twoms, twoS)
    s = twoS / 2
    r(m) = s * (s + 1) - m * (m + 1)
    return sum(r(twoms[b] / 2) * r(-twoms[b + 1] / 2) + r(-twoms[b] / 2) * r(twoms[b + 1] / 2)
               for b in 1:(length(twoms) - 1)) / 4
end

_wall(L, twoS) = [i <= L ÷ 2 ? twoS : -twoS for i in 1:L]

@testset "HQ Task 7: spin-1/2 and spin-3 XX" begin
set_symmetry!(:U1)

@testset "T7.0: the two models" begin
    @test spin_space(1) === local_space(:U1)
    rng = Random.MersenneTwister(7)
    for twoS in (1, 6)
        q = spin_space(twoS)
        @test leg_dim(q.I, 1) == twoS + 1 && length(q.I.spaces[1]) == twoS + 1   # d = 2s + 1
        L = 5
        mpo = spin_xx_mpo(L, twoS)
        for twoms in (_wall(L, twoS), rand(rng, twoS:-2:-twoS, L), rand(rng, twoS:-2:-twoS, L))
            psi = sector_product_state(q, [((m,),) for m in twoms])
            @test spin_sz_profile(psi, twoS) ≈ twoms ./ 2 atol = 1e-14
            @test abs(mpo_energy(psi, mpo)) < 1e-13
            @test mpo_variance(psi, mpo) ≈ _product_variance(twoms, twoS) atol = 1e-12
        end
        # Var_0(H_s) = J^2 s^2 on the L = 16 domain wall: only the central bond acts
        @test mpo_variance(spin_domain_wall_state(16, twoS), spin_xx_mpo(16, twoS)) ≈
              (twoS / 2)^2 atol = 1e-12
        @test_throws ArgumentError sector_product_state(q, [((twoS + 2,),)])
    end
    # s = 1/2 is the package's own chain and state
    @test abs(overlap(spin_domain_wall_state(8, 1), domain_wall_state(8))) ≈ 1 atol = 1e-14
    psi = neel_state(8)
    for _ in 1:3
        tdvp2_step!(psi, xxz_mpo(8; J = 1.0, delta = 0.0), -0.2im; maxdim = 64, trunc_thresh = 1e-14)
    end
    @test mpo_energy(psi, spin_xx_mpo(8, 1)) ≈ mpo_energy(psi, xxz_mpo(8; delta = 0.0)) atol = 1e-12
    @test mpo_variance(psi, spin_xx_mpo(8, 1)) ≈ mpo_variance(psi, xxz_mpo(8; delta = 0.0)) atol = 1e-12
end

@testset "T7.1: s = 1/2, the analytical finite-chain reference" begin
    L = 16
    h = xx_single_particle_hamiltonian(L)
    for t in (0.5, 3.0, 8.0)
        @test norm(xx_open_propagator(L, t) - exp(-1im * t * h)) < 1e-12
        C = xx_open_correlation(L, t)
        @test real(diag(C)) .- 0.5 ≈ xx_free_fermion_sz(L, t) atol = 1e-12
        @test real(tr(C)) ≈ L ÷ 2 atol = 1e-12                      # norm of the sector
        @test abs(tr(h * C)) < 1e-12                                # E(t) = E(0) = 0
        d = 1e-4                                                    # continuity, central difference
        dm = real(diag(xx_open_correlation(L, t + d)) - diag(xx_open_correlation(L, t - d))) / 2d
        j = xx_open_current(C)
        @test maximum(abs.(dm - ([0; j] - [j; 0]))) < 1e-7
    end
    # the converged MPS reference against it, to J t = 8
    psi, mpo = spin_domain_wall_state(L, 1), spin_xx_mpo(L, 1)
    jmpo = spin_current_mpo(L, 1, L ÷ 2)
    sector = psi[L].spaces[3]
    dt, every = 0.025, 40
    rows = []
    _evolve!(psi, mpo, dt, round(Int, 8 / dt), (p, k) -> k % every == 0 && begin
        t = k * dt
        C = xx_open_correlation(L, t)
        m = spin_sz_profile(p, 1)
        push!(rows, (t = t, err_m = maximum(abs.(m - (real(diag(C)) .- 0.5))),
                     err_j = abs(real(mpo_energy(p, jmpo)) - xx_open_current(C)[L ÷ 2]),
                     norm = abs(overlap(p, p) - 1), E = abs(mpo_energy(p, mpo)),
                     var = abs(mpo_variance(p, mpo) - 0.25), sz = abs(sum(m)),
                     sector = p[L].spaces[3] == sector, D = maximum(bond_dims(p))))
    end)
    @info "T7.1 s = 1/2, L = 16, order-4 TDVP2 dt = $dt" rows
    @test all(r -> r.err_m < 1e-6 && r.err_j < 1e-6, rows)
    @test all(r -> r.norm < 1e-10 && r.E < 1e-8 && r.var < 1e-6 && r.sz < 1e-10 && r.sector, rows)
end

@testset "T7.2: s = 3 against sector ED, L = $L" for L in (4, 6)
    twoS, dt, every = 6, 0.01, 25
    H, basis = spin_xx_sector(L, twoS)
    v0 = spin_sector_state(basis, _wall(L, twoS))
    psi, mpo = spin_domain_wall_state(L, twoS), spin_xx_mpo(L, twoS)
    b = L ÷ 2
    jmpo = spin_current_mpo(L, twoS, b)
    sector = psi[L].spaces[3]
    rows = []
    _evolve!(psi, mpo, dt, round(Int, 1 / dt), (p, k) -> k % every == 0 && begin
        v = spin_ed_propagate(H, v0, k * dt)
        m = spin_sz_profile(p, twoS)
        push!(rows, (t = k * dt, err_m = maximum(abs.(m - spin_ed_sz(v, basis))),
                     err_j = abs(real(mpo_energy(p, jmpo)) - spin_ed_current(v, basis, twoS, b)),
                     E = abs(mpo_energy(p, mpo)), var = abs(mpo_variance(p, mpo) - 9),
                     sz = abs(sum(m)), sector = p[L].spaces[3] == sector))
    end)
    @info "T7.2 s = 3, L = $L against ED, order-4 TDVP2 dt = $dt" rows
    @test all(r -> r.err_m < 1e-6 && r.err_j < 1e-6, rows)
    @test all(r -> r.E < 1e-8 && r.var < 1e-6 && r.sz < 1e-10 && r.sector, rows)
    # the ED itself: unitary, and its current obeys continuity (which pins the current's sign)
    t, d = 0.7, 1e-4
    v = spin_ed_propagate(H, v0, t)
    @test norm(v) ≈ 1 atol = 1e-12
    dm = (spin_ed_sz(spin_ed_propagate(H, v0, t + d), basis) -
          spin_ed_sz(spin_ed_propagate(H, v0, t - d), basis)) / 2d
    j = [spin_ed_current(v, basis, twoS, c) for c in 1:(L - 1)]
    @test maximum(abs.(dm - ([0; j] - [j; 0]))) < 1e-6
end

@testset "T7.3: short-time coefficient J^2 s^2, s = $(twoS / 2)" for twoS in (1, 6)
    s, ts = twoS / 2, (0.01, 0.005)
    rich(c) = (4 * c[2] - c[1]) / 3                                 # removes the t^2 correction
    L = 16
    c = map(ts) do t
        psi = _evolve!(spin_domain_wall_state(L, twoS), spin_xx_mpo(L, twoS), t / 4, 4)
        m = spin_sz_profile(psi, twoS)
        @test abs(m[L ÷ 2 + 1] + m[L ÷ 2]) < 1e-10                   # reflection x spin flip
        @test abs(sum(m)) < 1e-10
        (s - m[L ÷ 2]) / t^2
    end
    H, basis = spin_xx_sector(4, twoS)
    v0 = spin_sector_state(basis, _wall(4, twoS))
    ced = [(s - spin_ed_sz(spin_ed_propagate(H, v0, t), basis)[2]) / t^2 for t in ts]
    @info "T7.3 s = $s: MPS L = 16 $(c) -> $(rich(c)), ED L = 4 $(ced) -> $(rich(ced)), " *
          "want $(s^2)"
    @test abs(rich(c) - s^2) < 1e-5 * s^2
    @test abs(rich(ced) - s^2) < 1e-5 * s^2
end

end # HQ Task 7
