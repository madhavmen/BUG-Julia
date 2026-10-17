# tests/sweeps/test_tdvp2_truncation.jl -- `tdvp2_step!(...; weight_tol)` discards singular values
# by the BUG's own rule, CUT FOR CUT (USER 2026-10-16/17: the singular values discarded must be the
# same for both): every two-site update is cut by `_budget_svd` with the allowance
# `weight_tol / _cut_divisor(L, root)` that `truncate_recursive!` gives each bond of the BUG's
# S-step result. The step's relative discarded norm (`info.cumulative`) is reported, not budgeted.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

@testset "tdvp2_step!: weight_tol is the BUG's per-cut discard rule" begin
    set_symmetry!(:U1)
    L = 12
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    psi0 = neel_state(L)
    for _ in 1:10                       # entangle: a spectrum with a tail to cut
        tdvp2_step!(psi0, mpo, ComplexF64(-0.1im); maxdim = 64, trunc_thresh = 1e-14)
    end
    tau = ComplexF64(-0.1im)
    root = BUGJulia.RSVDCBEBondUpdate._bug_root_link(L) + 1
    div = BUGJulia.RSVDCBEBondUpdate._cut_divisor(L, root)
    infos = Dict{Float64, Any}()
    for wt in (1e-2, 1e-8)
        share = wt / div                # relative allowance of ONE cut, both integrators
        p = copy(psi0)
        info = tdvp2_step!(p, mpo, tau; maxdim = 1024, weight_tol = wt)
        # the BUG's S-step compression of the same kind of state, by the same rule
        q = copy(psi0)
        tdvp2_step!(q, mpo, tau; maxdim = 1024, trunc_thresh = 1e-14)
        canonical!(q, root)                  # `truncate_recursive!` cuts from the centre
        _, worst_bug, _, _ = truncate_recursive!(q, root; maxdim = 1024, weight_tol = wt)
        @info "weight_tol $wt: share $share, tdvp2 worst cut $(info.discarded), BUG worst cut " *
              "$(worst_bug), tdvp2 cumulative $(info.cumulative), max bond $(info.max_bond)"
        # every cut of either integrator fits the same allowance (maxdim 1024 never binds)
        @test info.discarded <= share * (1 + 1e-10)
        @test worst_bug <= share * (1 + 1e-10)
        # 2(L-1) cuts of at most `share`, quadrature per half-sweep, the halves added
        @test info.cumulative <= 2 * sqrt(L - 1) * share * (1 + 1e-10)
        infos[wt] = (info = info, share = share)
    end
    # tdvp2 cuts with the BUG's allowance, not the tighter per-step share 1/(2 sqrt(L-1)) it
    # used before (a cut past that old share is only allowed by the new rule)
    @test infos[1e-2].info.discarded > 1e-2 / (2 * sqrt(L - 1))
    infos = Dict(k => v.info for (k, v) in infos)
    # non-vacuous: the loose budget discarded something and cut the rank below the tight one
    @test infos[1e-2].cumulative > 0
    @test infos[1e-2].max_bond < infos[1e-8].max_bond
    # the legacy relative cutoff is untouched by the new keyword's default
    a, b = copy(psi0), copy(psi0)
    ia = tdvp2_step!(a, mpo, tau; maxdim = 64, trunc_thresh = 1e-6)
    ib = tdvp2_step!(b, mpo, tau; maxdim = 64, trunc_thresh = 1e-6, weight_tol = nothing)
    @test bond_dims(a) == bond_dims(b) && isfinite(ia.cumulative)
    @test_throws ArgumentError tdvp2_step!(copy(psi0), mpo, tau; weight_tol = -1.0)
end
