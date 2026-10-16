# tests/sweeps/test_tdvp2_truncation.jl -- `tdvp2_step!(...; weight_tol)` discards singular values
# by the BUG's own rule (USER 2026-10-16: "discard the singular values the same for both"): the
# budgeted cut `_budget_svd` of `truncate_recursive!`, with the step's relative discarded norm
# accounted the same way (`info.cumulative`), so `weight_tol` means what `tol_trunc` means.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

@testset "tdvp2_step!: weight_tol is the BUG's discard budget" begin
    set_symmetry!(:U1)
    L = 12
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    psi0 = neel_state(L)
    for _ in 1:10                       # entangle: a spectrum with a tail to cut
        tdvp2_step!(psi0, mpo, ComplexF64(-0.1im); maxdim = 64, trunc_thresh = 1e-14)
    end
    tau = ComplexF64(-0.1im)
    infos = Dict{Float64, Any}()
    for wt in (1e-2, 1e-8)
        p = copy(psi0)
        info = tdvp2_step!(p, mpo, tau; maxdim = 1024, weight_tol = wt)
        @info "weight_tol $wt: cumulative $(info.cumulative), max bond $(info.max_bond), " *
              "worst bond weight $(info.discarded)"
        # the budget holds for the step (maxdim 1024 never binds at L = 12)
        @test info.cumulative <= wt * (1 + 1e-10)
        infos[wt] = info
    end
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
