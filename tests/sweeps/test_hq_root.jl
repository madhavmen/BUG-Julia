# tests/sweeps/test_hq_root.jl -- updated_cbe_high_order_parallel.md, Task 5 (root accuracy,
# orders 1..5, final compression). The local orders p = 1..5 (T5.1) are UT3 of
# test_krylov_closure.jl.
#
#   T5.2  the root solve against the exact exponential of the dense root matrix U'HU of a real
#         step's frames: tol_root bounds Saad's estimate, the estimate is not optimistic, the
#         root action count grows with the tolerance; the MPO root action is Hermitian and equals
#         the dense one; the original state lies in the frames
#   T5.3  root-order ablation at p = 3: trapezoid (q = 2 < p) caps the local order at q + 1,
#         radau2 (q = p), radau3 and the exponential reach p + 1
#   T5.4  no hidden truncation: with the compression off nothing is cut, the root split is
#         lossless and the frames hold psi_n, p = 1..5
#   T5.5  global order at fixed T for p = 1..5, read only where the other errors are below the
#         temporal one

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))

@testset "HQ Task 5: root and order" begin

@testset "T5.2: root solve against the exact root exponential" begin
    L = 8
    mpo, psi0 = _xx_warm(L, 2)
    H = dense_heisenberg(L; J = 1.0, delta = 0.0)
    v0 = dense_state(psi0)
    tau = ComplexF64(-0.5im)
    rows, its = Any[], Int[]
    for tolx in (1e-4, 1e-8, 1e-12)
        fr, rr = Ref{Any}(nothing), Ref{Any}(nothing)
        psi = copy(psi0)
        info = cbe_bug_step!(psi, mpo, tau; tol_basis = 1e-8, tol_root = tolx, truncate = false,
                             frames_report = fr, root_report = rr)
        U = _root_basis(psi0, mpo, fr[]; mmax = 10)
        Hr = U' * H * U
        @test norm(v0 - U * (U' * v0)) < 1e-12 * norm(v0)             # P_p psi_n = psi_n
        if tolx == 1e-4
            # the MPO root action IS the dense U'HU on the root space, and Hermitian
            B = RC._root_generator(mpo, fr[])
            S0 = RC._project_root(psi0, fr[])
            X, Y = (RC._randomize_like(S0, Random.MersenneTwister(s)) for s in (1, 2))
            dX = dense_state(RC._frontier_mps(psi0, fr[], X))
            dBX = dense_state(RC._frontier_mps(psi0, fr[], RC.apply_zero_site(B, X)))
            @test norm(dBX - U * (Hr * (U' * dX))) < 1e-10 * norm(dX)
            @test abs(tensor_inner(X, RC.apply_zero_site(B, Y)) -
                      conj(tensor_inner(Y, RC.apply_zero_site(B, X)))) <
                  1e-11 * norm(X) * norm(Y)
        end
        err = norm(dense_state(psi) - U * (exp(tau * Hr) * (U' * v0))) / norm(v0)
        est = rr[].solve.err
        @test est <= tolx * (1 + 1e-8)
        @test err <= 10 * tolx + 1e-13
        @test err <= 10 * est + 1e-13                         # the estimate is not optimistic
        push!(its, info.record.root_iters)
        push!(rows, (tol = tolx, est = est, err = err, n_root = its[end], dim = size(U, 2)))
    end
    @test issorted(its) && its[1] < its[end]
    @test its[1] < rows[1].dim                       # a loose tolerance stops before closure
    @info "T5.2 root solve vs exact exp(tau U'HU)" rows
end

@testset "T5.3: root-order ablation at p = 3" begin
    p = 3
    # local order min(p, q) + 1
    for (m, want) in ((:trapezoid, 3), (:radau2, 4), (:radau3, 4), (:exp, 4))
        errs, floors = _dw_errors(p, (0.4, 0.2, 0.1); root_method = m)
        slope = _finest_order(errs, floors)
        @info "T5.3 p=$p root $m: local errors $(errs), floors $(floors), slope $(slope) " *
              "(want $want)"
        @test slope !== nothing && slope >= want - 0.4
        m === :trapezoid && @test slope !== nothing && slope <= want + 0.4   # q < p caps it
    end
end

@testset "T5.4: nothing cut before the one compression, p = $p" for p in 1:5
    mpo, psi = _xx_warm(10, 3)
    n0 = sqrt(real(overlap(psi, psi)))
    info = cbe_bug_step!(psi, mpo, ComplexF64(-0.1im); order = p, tol_basis = 1e-8,
                         tol_root = 1e-12, truncate = false)
    @test info.discarded < 1e-14                                 # the root split is lossless
    @test info.ledger.proj_err < 1e-12                           # the frames hold psi_n
    @test info.ledger.n_compress == 0
    @test abs(sqrt(real(overlap(psi, psi))) - n0) < 1e-10 * n0   # unitary: nothing cut anywhere
end

@testset "T5.5: global order at fixed T, p = $p" for p in 1:5
    errs, floors = _dw_errors(p, (0.4, 0.2, 0.1); T = 0.8)
    slope = _finest_order(errs, floors)
    @info "T5.5 p=$p: global errors $(errs), floors $(floors), p_obs " *
          "$(_observed_orders(errs, floors)) (target $p)"
    @test slope !== nothing && slope >= p - 0.4
end

end # HQ Task 5
