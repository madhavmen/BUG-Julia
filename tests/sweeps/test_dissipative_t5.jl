# tests/sweeps/test_dissipative_t5.jl -- dissipative_cbe_bug_implementation_plan.md, Task 5: the
# step-doubling controller of plan eq. (7) in `jump_trajectory`, on local loss from the L = 10 domain
# wall. Only the first step has a visible time error there (C2: 5.4e-7 at dt 0.1); once entangled
# the chain's bonds are at full rank and a step is exact to ~1e-12 (dt5s).
#
#   C1  `_mps_distance` resolves `ε ||X||` down to ε = 1e-12 (a centre-tensor perturbation)
#   C2  on real steps it equals the dense distance, also at ~1e-12 where the overlap form
#       `||a||² + ||b||² - 2 Re<a,b>` reads 0 or sqrt(eps) (CONTROL: that form is replayed and fails);
#       the eq. (7) estimate of the first step's error matches the dense error
#   C3  `jump_trajectory`: the first step is rejected and shrunk, a check is accepted exactly when its
#       estimate is within τ, the step grows once the steps are exact, and the final state beats the
#       fixed step (CONTROL) by far

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "dd_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dissipative_fixtures.jl"))

const RC = BUGJulia.RSVDCBEBondUpdate

@testset "dissipative Task 5: the eq. (7) step-doubling controller" begin

set_symmetry!(:U1)
L = 10
_, _, G, Gd, _ = _lossy_xx(L; gamma = _loss_rates(L))
Gd = Matrix(Gd)
ch = site_emission_channels(L; gamma = _loss_rates(L))
step!(p, h) = cbe_bug_step!(p, G, ComplexF64(-im * h); hermitian = false, order = 3, tol_basis = 1e-10,
                            tol_root = 1e-12, tol_trunc = 1e-12, maxdim = 256)
overlap_distance(a, b) =
    sqrt(max(real(overlap(a, a)) + real(overlap(b, b)) - 2 * real(overlap(a, b)), 0.0))
psi0 = domain_wall_state(L)
v0 = dense_state(psi0)
warm = copy(psi0)
step!(warm, 0.1); step!(warm, 0.1)
RC._jt_normalise!(warm)

@testset "C1: the distance resolves a small perturbation" begin
    c = 5
    a = copy(warm)
    canonical!(a, c)
    X = BUGJulia.BondUpdateBUG._sz_site(a[c])
    for eps_ in (1e-3, 1e-9, 1e-12)
        b = copy(a)
        b[c] = to_concrete(a[c] + to_concrete(ComplexF64(eps_) * X))
        d = RC._mps_distance(a, b)
        # + ~20 eps of ||a|| = 1: backward-stable SVDs leave an absolute roundoff (dt5t: 1.1e-15 at
        # ε = 1e-12), still ~1% of the perturbation there, where the overlap form reads 0 or 1.5e-8
        @test abs(d - eps_ * norm(X)) <= 1e-3 * eps_ * norm(X) + 5e-15
    end
end

@testset "C2: the distance and the eq. (7) estimate on real steps" begin
    # an entangled start: the step is exact to ~1e-12
    for h in (0.2, 0.05)
        full = copy(warm); step!(full, h)
        half = copy(warm); step!(half, h / 2); step!(half, h / 2)
        dden = norm(dense_state(full) - dense_state(half))
        dmps = RC._mps_distance(full, half)
        dold = overlap_distance(full, half)
        @info "C2 entangled h = $h: dense $dden, _mps_distance $dmps, overlap form $dold"
        @test dden < 1e-9                                    # the regime the overlap form cannot see
        @test abs(dmps - dden) <= 0.05 * dden + 1e-14
        @test abs(dold - dden) > 0.5 * dden                  # CONTROL: the replaced form fails here
    end
    # the domain wall: the first step's real error
    for h in (0.1, 0.05)
        full = copy(psi0); step!(full, h)
        half = copy(psi0); step!(half, h / 2); step!(half, h / 2)
        ex = exp(-im * h * Gd) * v0
        dden = norm(dense_state(full) - dense_state(half))
        e = step_doubling_error(full, half)
        efull = norm(dense_state(full) - ex)
        @info "C2 domain wall h = $h: eq. (7) estimate $e, dense full-step error $efull"
        @test abs(RC._mps_distance(full, half) - dden) <= 1e-6 * dden + 1e-14
        @test efull > 1e-8                                   # a measurable error
        @test 1 / 3 < e / efull < 3
    end
end

@testset "C3: the controller in jump_trajectory" begin
    dt, tau, T = 0.1, 1e-9, 1.0
    final_error(tr_psi) = norm(tr_psi / norm(tr_psi) - normalize(exp(-im * T * Gd) * v0))
    # `local`: the testset's own `tr` would otherwise be captured, and the second call's result
    # overwrote the first's (dt5t: the controlled run's `tr` came back as the fixed run's)
    segment(; kw...) = begin
        local final = Ref{Any}(nothing)
        local seg = jump_trajectory(psi0, G, ch, [0.0, T]; dt = dt, seed = 1, jumps = false,
                                    step = step!, observe = p -> (final[] = dense_state(p); [0.0]),
                                    kw...)
        seg, final[]
    end
    tr, v = segment(target_step_error = tau, check_every = 1)
    _, vf = segment()
    ectl, efix = final_error(v), final_error(vf)
    hs = [h for (_, h) in tr.steps]
    @info "C3 checks $([(round(c.t; digits = 4), round(c.h; digits = 4), c.err, c.accepted) for c in tr.checks])"
    @info "C3 final error: controlled $ectl ($(length(hs)) steps, $(tr.nreject) rejected, h in " *
          "[$(minimum(hs)), $(maximum(hs))]), fixed dt $dt $efix"
    c1 = tr.checks[1]
    @test !c1.accepted && c1.err > tau && tr.nreject >= 1    # the domain-wall step is too large
    @test all(c.h > dt / 64 * (1 + 1e-12) for c in tr.checks) # never forced through at dt_min
    @test all(c.accepted == (c.err <= tau) for c in tr.checks)
    @test all(c.err > 0 for c in tr.checks)                  # the estimate resolves exact steps
    @test maximum(hs) >= 2dt                                 # and the step grows on them
    # CONTROL: the fixed step is visibly off (dt5u: 6.4e-7, C2's first-step error 5.4e-7)
    @test efix > 1e-7
    @test ectl < 1e-7 && ectl < efix / 100
end

end
