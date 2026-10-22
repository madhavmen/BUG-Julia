# tests/sweeps/test_spin_jw.jl -- cbe_bug_high_order_rsvd_parallel_tasks (2).md, T7.2 / U31: the
# generalized Jordan-Wigner map (Batista-Ortiz) of the open spin-s XX chain on small chains.
#
#   map     T'T = I; T carries S^+_j, S^z_j, H_s, the centre current and S^z_tot (H_JW T = T H_s)
#   free?   the best quadratic fermion operator on T's range, and on the domain wall's Krylov
#           space: residual ~0 at s = 1/2 (the control: ordinary JW is free), measured at s = 3
#
# An analytical L = 16 spin-3 curve is allowed only if the s = 3 residual is ~0 (plan line 454).

using Test
using LinearAlgebra, SparseArrays

include(joinpath(@__DIR__, "..", "common", "spin_jw.jl"))

"Orthonormal basis of the Krylov space of `v` under `H` (to numerical rank)."
function _krylov_basis(H, v)
    K = [v / norm(v)]
    for _ in 2:size(H, 1)
        w = H * K[end]
        push!(K, w / max(norm(w), eps()))
    end
    F = svd(reduce(hcat, K))
    return F.U[:, F.S .> 1e-10 * F.S[1]]
end

@testset "U31: generalized Jordan-Wigner map, s = $(twoS / 2), L = $L, vacuum n = $vac" for
        (twoS, L, vac) in [(1, 2, 0), (1, 3, 0), (1, 4, 0), (1, 3, 1),
                           [(6, 2, v) for v in 0:6]..., (6, 3, 0), (6, 3, 3)]
    m = JWMap(L, twoS; vac = vac)
    T = jw_isometry(m)
    d = twoS + 1
    Sp, Sz = spin_local_ops(twoS)
    H, cur = spin_chain_ops(L, twoS)
    @test norm(Matrix(T' * T) - I) < 1e-14
    jSp = [jw_local(m, Sp, j) for j in 1:L]
    for j in 1:L
        @test norm(jSp[j] * T - T * spin_site_op(Sp, j, L, d)) < 1e-12
        @test norm(jw_local(m, Sz, j) * T - T * spin_site_op(Sz, j, L, d)) < 1e-12
    end
    hop(b) = jSp[b] * sparse(jSp[b + 1]')
    HJW = (1 / 2) * sum(hop(b) + sparse(hop(b)') for b in 1:(L - 1))
    @test norm(HJW * T - T * H) < 1e-12
    b = max(1, L ÷ 2)
    @test norm((1im / 2) * (hop(b) - sparse(hop(b)')) * T - T * cur(b)) < 1e-12
    Sztot = sum(jw_local(m, Sz, j) for j in 1:L)
    @test norm(Sztot * HJW * T - HJW * Sztot * T) < 1e-12

    res, ncoef = jw_quadratic_residual(m, T, H)
    rows = [(subspace = "all", residual = res, ncoef = ncoef)]
    if iseven(L)                                # the plan's domain wall needs even L
        wall = [j <= L ÷ 2 ? twoS : 0 for j in 1:L]           # local index n: +s, then -s
        v0 = zeros(d^L); v0[1 + sum(wall[j] * d^(L - j) for j in 1:L)] = 1
        @test spin_site_op(Sz, 1, L, d) * v0 ≈ (twoS / 2) * v0
        rdw, _ = jw_quadratic_residual(m, T, H; V = _krylov_basis(H, v0))
        push!(rows, (subspace = "domain-wall Krylov", residual = rdw, ncoef = ncoef))
    end
    @info "U31 s = $(twoS / 2), L = $L, vacuum n = $vac: best quadratic fit of the mapped H_s" rows
    if twoS == 1
        @test all(r -> r.residual < 1e-10, rows)          # the control: ordinary JW is free
    else
        @test all(r -> r.residual > 1e-3, rows)           # s = 3: not free in this map
    end
end
