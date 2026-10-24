# tests/sweeps/test_dissipative_t4_conv.jl -- dissipative_cbe_bug_implementation_plan.md, Task 4,
# U15: convergence of jumping trajectories under step refinement, on local loss (site S⁻ jumps) from
# the L = 10 domain wall, where the no-jump step's time error is above its floors (see U10).
#
#   U15a  the first jump time of one seed, whose exact crossing P0(t*) = u lies several steps in,
#         is LOCATED: its error is far below snapping to the step grid (CONTROL: snapping is off by
#         >= 1e-2), and where it is above the floor it falls at the segment's order
#   U15b  weak convergence: paired-seed ensembles (same thresholds and channel draws) converge to a
#         fine-step ensemble in the left-half occupation, which jumps do not fix (the excitation
#         number is conserved between jumps, so it would be identical at every dt, dt4b). Pathwise
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
# a quarter of `_loss_rates`: the first jump falls several steps in (dt4b: t* = 0.38 < dt at the
# full rates, so every dt located it with ONE step of the same size and nothing could converge)
g = 0.25 .* _loss_rates(L)
_, _, G, Gd, _ = _lossy_xx(L; gamma = g)
ch = site_emission_channels(L; gamma = g)
psi0 = domain_wall_state(L)
v0 = dense_state(psi0)
flow = _ed_flow(-1im * Matrix(Gd), v0)

@testset "U15a: the first jump time is located, and converges" begin
    tstar(s) = _ed_jump_time(flow, v0, rand(Random.MersenneTwister(s)); tmax = 6.0)
    seed = findfirst(s -> 1.5 < tstar(s) < 3.5, 1:200)
    @test seed !== nothing
    ts = tstar(seed)
    kw = (tol_basis = 1e-12, tol_root = 1e-14, exact = true)
    dts = (0.4, 0.2, 0.1)
    errs = Float64[]
    for dt in dts
        tr = jump_trajectory(psi0, G, ch, [0.0, 6.0]; dt = dt, seed = seed, step_kw = kw,
                             tol_trunc = 1e-14)
        push!(errs, abs(tr.events[1].t - ts))
    end
    snap = [abs(ceil(ts / dt) * dt - ts) for dt in dts]
    floors = fill(1e-10, 3)
    @info "U15a first jump (seed $seed): ED $ts, |t(dt) - t*| $(errs), p_obs " *
          "$(_observed_orders(errs, floors)); snapped to the grid $(snap)"
    # Located, not snapped. The crossing itself is solved to `event_tol` (U13: residual 1e-14);
    # what remains is the segment's own time error in P0, which no absolute bound at dt = 0.4 can
    # sit under (dt4g: 1.1e-5, 8.5e-7, 5.8e-8), so it is compared with the grid instead.
    @test all(errs .< 1e-2 .* snap)
    @test errs[end] <= 1e-7
    @test maximum(snap) >= 1e-2                             # CONTROL: the grid would not do
    p = _finest_order(errs, floors)
    p === nothing || @test p >= 2.6                         # converges where measurable
end

@testset "U15b: paired-seed ensembles converge weakly" begin
    occ = to_concrete(local_space().Sz + 0.5 * local_space().I)
    left(p) = [sum(real(site_expval(p, j, occ)) for j in 1:(L ÷ 2))]
    kw = (tol_basis = 1e-6, tol_root = 1e-12)
    T = 2.0
    ens_at(dt) = jump_ensemble(psi0, G, ch, [0.0, T]; ntraj = 12, seed = 50, observe = left, dt = dt,
                               step_kw = kw, tol_trunc = 1e-11)
    ref = ens_at(0.025)
    d = Float64[]
    for dt in (0.2, 0.1, 0.05)
        e = ens_at(dt)
        push!(d, abs(e.mean[end][1] - ref.mean[end][1]))
    end
    @info "U15b weak error in the left-half occupation against dt = 0.025: $(d); reference " *
          "$(ref.mean[end][1]), jumps $(ref.jumps_mean[end])"
    @test ref.jumps_mean[end] > 0.5                          # the ensemble really jumps
    @test d[1] > 1e-8                                        # the error is measurable
    @test d[3] < d[1] / 4 && d[3] < d[2]                     # and falls as dt is refined
end

end
