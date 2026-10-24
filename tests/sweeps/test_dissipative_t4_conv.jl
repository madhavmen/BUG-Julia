# tests/sweeps/test_dissipative_t4_conv.jl -- dissipative_cbe_bug_implementation_plan.md, Task 4,
# U15: convergence of jumping trajectories under step refinement, on local loss (site S⁻ jumps) from
# the L = 10 domain wall, where the no-jump step's time error is above its floors (see U10).
#
#   U15a  the first jump time of one seed converges to the exact crossing P0(t*) = u, at the
#         segment's order (the crossing is located, not snapped to the grid)
#   U15b  weak convergence: paired-seed ensembles (same thresholds and channel draws) converge to a
#         fine-step ensemble in the mean excitation number and the mean jump count. Pathwise
#         global order 3 across jumps is NOT asserted (plan U15)

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

@testset "dissipative Task 4: U15, convergence of jumping trajectories" begin

set_symmetry!(:U1)
L = 10
_, _, G, Gd, _ = _lossy_xx(L)
ch = site_emission_channels(L; gamma = _loss_rates(L))
psi0 = domain_wall_state(L)
v0 = dense_state(psi0)

@testset "U15a: the first jump time converges at the segment's order" begin
    seed = 3
    tstar = _ed_jump_time(_ed_flow(-1im * Matrix(Gd), v0), v0, rand(Random.MersenneTwister(seed));
                          tmax = 4.0)
    @test isfinite(tstar)
    kw = (tol_basis = 1e-12, tol_root = 1e-14, exact = true)
    errs = Float64[]
    for dt in (0.4, 0.2, 0.1)
        tr = jump_trajectory(psi0, G, ch, [0.0, 4.0]; dt = dt, seed = seed, step_kw = kw,
                             tol_trunc = 1e-14)
        push!(errs, abs(tr.events[1].t - tstar))
    end
    floors = fill(1e-10, 3)              # the event tolerance, in time, at the hazard rate here
    @info "U15a first jump: ED $tstar, |t(dt) - t*| $(errs), p_obs $(_observed_orders(errs, floors))"
    p = _finest_order(errs, floors)
    @test p !== nothing && p >= 2.6
end

@testset "U15b: paired-seed ensembles converge weakly" begin
    nexc(p) = [sum(real(site_expval(p, j, to_concrete(local_space().Sz + 0.5 * local_space().I)))
                   for j in 1:L)]
    kw = (tol_basis = 1e-9, tol_root = 1e-12)
    T = 1.0
    ens_at(dt) = jump_ensemble(psi0, G, ch, [0.0, T]; ntraj = 12, seed = 50, observe = nexc, dt = dt,
                            step_kw = kw, tol_trunc = 1e-11)
    ref = ens_at(0.025)
    d = Float64[]
    for dt in (0.2, 0.1, 0.05)
        e = ens_at(dt)
        push!(d, abs(e.mean[end][1] - ref.mean[end][1]) + abs(e.jumps_mean[end] - ref.jumps_mean[end]))
    end
    @info "U15b weak error against dt = 0.025: $(d); reference <N_exc> $(ref.mean[end][1]), " *
          "jumps $(ref.jumps_mean[end])"
    @test ref.jumps_mean[end] > 0.5                       # the ensemble really jumps
    @test d[3] < d[1] / 4 && d[2] < d[1] && d[3] < d[2]   # converges as dt is refined
end

end
