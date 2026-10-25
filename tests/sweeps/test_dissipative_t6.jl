# tests/sweeps/test_dissipative_t6.jl -- dissipative_cbe_bug_implementation_plan.md §4.3 (needed
# before Task 7): checkpoint/restart of `jump_trajectory` and `jump_ensemble`, on the U13 fixture
# (A2 collective emission, L = 6), every checkpoint through a Serialization round trip as the
# driver writes it.
#
#   R1  a trajectory with jumps and the eq. (7) controller, resumed from a grid time before its last
#       jump, reproduces the uninterrupted one exactly (records, log survival, events, steps,
#       checks); CONTROL: the same resume with a fresh RNG draws different jumps
#   R2  a resume refuses another seed or a grid that does not extend the checkpoint's, and on a
#       longer grid equals the longer uninterrupted run
#   R3  an ensemble resumed inside its second trajectory, and between trajectories, equals the
#       uninterrupted ensemble

using Test
using LinearAlgebra
using Serialization
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "dd_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dissipative_fixtures.jl"))

# the driver's checkpoint: serialised while the state is live
function _roundtrip(x)
    io = IOBuffer()
    serialize(io, x)
    seekstart(io)
    return deserialize(io)
end
_same(a, b) = a.obs == b.obs && a.logp == b.logp && a.njumps == b.njumps && a.steps == b.steps &&
              a.checks == b.checks && [(e.t, e.channel) for e in a.events] ==
              [(e.t, e.channel) for e in b.events] && (a.nsteps, a.nevals, a.nreject) ==
              (b.nsteps, b.nevals, b.nreject)

@testset "dissipative §4.3: checkpoint/restart of trajectories and ensembles" begin

set_symmetry!(:U1)
L = 6
f = _emission_model(L; gm = 0.3)
obs(p) = vcat(real(overlap(p, p)), [real(site_expval(p, j, local_space().Sz)) for j in 1:L])
kw = (dt = 0.1, observe = obs, target_step_error = 1e-6,
      step_kw = (closure_checks = true, tol_basis = 1e-10, tol_root = 1e-12))
tg = collect(0.0:0.5:4.0)

@testset "R1: a resumed trajectory is the uninterrupted one" begin
    # a fine grid: seed 7 jumps at t = 0.18 and 0.50 (dt6r), both inside the first 0.5 interval
    tg1 = collect(0.0:0.05:2.0)
    snaps = Dict{Int, Any}()
    full = jump_trajectory(f.psi0, f.G, f.ch, tg1; seed = 7, kw...,
                           checkpoint = st -> (snaps[st.k] = _roundtrip(st); nothing))
    @test sort(collect(keys(snaps))) == collect(2:length(tg1))
    @test !isempty(full.events) && !isempty(full.checks)
    # the last grid time before the last jump: the resumed part draws a channel and a threshold
    k = findlast(t -> t < full.events[end].t, tg1)
    @info "R1: $(length(full.events)) jumps at $([e.t for e in full.events]), resumed from t = $(tg1[k])"
    @test k >= 2
    res = jump_trajectory(f.psi0, f.G, f.ch, tg1; seed = 7, kw..., resume = snaps[k])
    @test _same(res, full)
    # the checkpoint survives a second resume (resume copies it)
    @test _same(jump_trajectory(f.psi0, f.G, f.ch, tg1; seed = 7, kw..., resume = snaps[k]), full)
    # CONTROL: without the RNG's state the resumed jumps differ
    bad = merge(snaps[k], (rng = Random.MersenneTwister(7),))
    tbad = jump_trajectory(f.psi0, f.G, f.ch, tg1; seed = 7, kw..., resume = bad)
    @test [e.t for e in tbad.events] != [e.t for e in full.events]
end

@testset "R2: inconsistent resumes are refused; a longer grid extends" begin
    snaps = Dict{Int, Any}()
    full = jump_trajectory(f.psi0, f.G, f.ch, tg; seed = 7, kw...,
                           checkpoint = st -> (snaps[st.k] = _roundtrip(st); nothing))
    @test_throws ArgumentError jump_trajectory(f.psi0, f.G, f.ch, tg; seed = 8, kw...,
                                               resume = snaps[4])
    @test_throws ArgumentError jump_trajectory(f.psi0, f.G, f.ch, collect(0.0:0.25:4.0); seed = 7,
                                               kw..., resume = snaps[4])
    @test_throws ArgumentError jump_trajectory(f.psi0, f.G, f.ch, tg[1:3]; seed = 7, kw...,
                                               resume = snaps[4])
    tg6 = collect(0.0:0.5:6.0)
    long = jump_trajectory(f.psi0, f.G, f.ch, tg6; seed = 7, kw...)
    @test long.obs[1:length(tg)] == full.obs
    @test _same(jump_trajectory(f.psi0, f.G, f.ch, tg6; seed = 7, kw..., resume = snaps[length(tg)]),
                long)
end

@testset "R3: a resumed ensemble is the uninterrupted one" begin
    tge = collect(0.0:0.5:2.0)
    ekw = (ntraj = 3, kw...)
    snaps = Any[]
    full = jump_ensemble(f.psi0, f.G, f.ch, tge; seed = 100, ekw...,
                         checkpoint = st -> (push!(snaps, _roundtrip(st)); nothing))
    # per trajectory: one state per grid time after the first, then one between trajectories
    @test length(snaps) == 3 * length(tge)
    inside = findfirst(s -> length(s.done) == 1 && s.current !== nothing && s.current.k == 3, snaps)
    between = findfirst(s -> length(s.done) == 2 && s.current === nothing, snaps)
    @test inside !== nothing && between !== nothing
    for s in (snaps[inside], snaps[between])
        res = jump_ensemble(f.psi0, f.G, f.ch, tge; seed = 100, ekw..., resume = s)
        @test res.mean == full.mean && res.stderr == full.stderr && res.jumps_mean == full.jumps_mean
        @test all(_same(a, b) for (a, b) in zip(res.trajs, full.trajs))
    end
    @test_throws ArgumentError jump_ensemble(f.psi0, f.G, f.ch, tge; seed = 101, ekw...,
                                             resume = snaps[between])
end

end
