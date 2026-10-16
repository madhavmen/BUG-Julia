# tests/sweeps/test_tdvp2_order.jl -- `tdvp2_composed_step!`: 2-site TDVP at order 2 and 4 in the
# time step.
#
# ⛔ THE ORDER SHOWS ONLY WHERE THE STEP ERROR DOMINATES. At a binding cap (L = 8, cap 6) every
# composition converges at slope ~1.2 to its own dt -> 0 limit (job 17283495): the per-bond
# truncation, not the splitting, sets the error. At full rank the projector-splitting step is exact
# and there is no step error at all. So this is the TOLERANCE regime of
# benchmarks/tdvp2_order_probe.jl (job 17284044): L = 24, a quench from the gapped Delta = 3 ground
# state to Delta = 1, rank set by trunc_thresh 1e-10 far below both the cap and full rank.
#
# Error = max_j |<S^z_j>_dt(T) - <S^z_j>_ref(T)|; the state distance from overlaps cannot resolve
# below ~sqrt(eps). Measured there against Yoshida at dt 0.05/8, coarsest pair dt 0.2 -> 0.1:
# order 2 1.75e-10 -> 4.07e-11 (slope 2.11), order 4 Suzuki 3.91e-11 -> 7.87e-13 (slope 5.64),
# floor ~7e-13. The reference here is order 4 Suzuki at dt 0.05, which sat at that floor (6.88e-13),
# so it resolves both pairs at a fraction of the cost. (Yoshida's -1.70 tau stage needs a finer
# step: slopes 2.80 then 4.80, so it is not asserted at this pair.)

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

@testset "tdvp2_composed_step!: order 2 and 4 in the step" begin
    set_symmetry!(:U1)
    L, T = 24, 1.0
    gs = neel_state(L)
    hg = xxz_mpo(L; J = 1.0, delta = 3.0)
    for _ in 1:40          # imaginary time to the gapped ground state, renormalised every step
        tdvp2_step!(gs, hg, ComplexF64(-0.1); maxdim = 256, trunc_thresh = 1e-12, maxiter = 30)
        canonical!(gs, 1)
        gs[1] = to_concrete((1.0 / norm(gs[1])) * gs[1])
    end
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    kw = (maxdim = 256, trunc_thresh = 1e-10, maxiter = 30)
    run(order, dt) = (p = copy(gs); cap = 0;
        for _ in 1:round(Int, T / dt)
            info = tdvp2_composed_step!(p, mpo, ComplexF64(-1im * dt); order = order, kw...)
            cap = max(cap, info.max_bond)
        end; (p, cap))
    sz(a) = magnetisation(copy(a)) ./ real(overlap(a, a))
    ref, _ = run(4, 0.05)
    szref = sz(ref)
    dts = (0.2, 0.1)
    for order in (2, 4)
        runs = [run(order, dt) for dt in dts]
        errs = [maximum(abs.(sz(p) .- szref)) for (p, _) in runs]
        slope = log2(errs[1] / errs[2])
        @info "TDVP2 order $order (suzuki): max|dSz| $(errs), slope $(slope), " *
              "max bond $([c for (_, c) in runs])"
        @test all(c < kw.maxdim for (_, c) in runs)  # the tolerance, not the cap, set the rank
        @test errs[1] > 1e-11                          # non-vacuous: a step error above the floor
        @test slope >= order - 0.5
    end
    psi0 = neel_state(8)
    mpo8 = xxz_mpo(8; J = 1.0, delta = 1.0)
    @test_throws ArgumentError tdvp2_composed_step!(psi0, mpo8, -0.1im; order = 3)
    @test_throws ArgumentError tdvp2_composed_step!(psi0, mpo8, -0.1im; composition = :x)
end
