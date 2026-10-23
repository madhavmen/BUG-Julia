# tests/sweeps/test_adaptive_expv.jl -- UT15 of the adaptive CBE-Krylov plan (§1, §4, §7):
# Stage 5, the root `expv` bound to `tol_expv`.
#
# The operator is a known small Hermitian matrix `M`, realised matrix-free on orthonormal
# block-sparse tensors (`_hop`), so `exp(tau*M)` from `LinearAlgebra` is the exact answer and the
# solver under test is the production `expv` on real Telum tensors.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))

"`n` orthonormal tensors sharing `seed`'s charge blocks (Gram-Schmidt, twice)."
function _onb(seed, n; rng = Random.MersenneTwister(5))
    out = Any[]
    for _ in 1:n
        Y = RC._randomize_like(seed, rng)
        for _pass in 1:2, u in out
            Y = to_concrete(Y - tensor_inner(u, Y) * u)
        end
        push!(out, to_concrete((1.0 / norm(Y)) * Y))
    end
    return out
end

"The matrix-free `M` on the basis `e`, and the exact `exp(tau*M) e_1`."
function _known(M, e)
    n = size(M, 1)
    terms = [(i == j ? M[i, i] / 2 : M[i, j], e[i], e[j]) for i in 1:n for j in i:n]
    exact(tau) = (c = exp(tau * M)[:, 1]; to_concrete(sum(c[i] * e[i] for i in 2:n; init = c[1] * e[1])))
    return _hop(terms), exact
end

@testset "adaptive root expv (Stage 5)" begin

kl = _xx_kl(:left; steps = 2)
n = 6
e = _onb(kl.seed, n)
@test maximum(abs(tensor_inner(e[i], e[j]) - (i == j)) for i in 1:n, j in 1:n) < 1e-12
rng = Random.MersenneTwister(9)
A = randn(rng, ComplexF64, n, n)
# ||M|| = 0.2 EXACTLY, so at tau = -i a loose tolerance converges (error ~ 0.2^k/k!, 7e-5 at
# k = 4) before the n = 6 space closes. A random scale left ||tau M|| ~ 1 and every tolerance ran
# to closure (job 17193333), which makes the iteration-count test vacuous.
M = (A + A') / 2
M .*= 0.2 / opnorm(M)
H, exact = _known(M, e)

@testset "UT15: iterations follow tol_expv and agree with the dense exponential" begin
    tau = ComplexF64(-1.0im)
    its = Int[]
    for tolx in (1e-4, 1e-8, 1e-12)
        rep = Ref{Any}(nothing)
        y = expv(H, tau, e[1]; conv_tol = tolx, substeps = 0, maxiter = 30, tol = 1e-15,
                 reorth = true, report = rep)
        err = norm(to_concrete(y - exact(tau)))
        @test err <= 10 * tolx + 1e-13
        @test rep[].err <= tolx * (1 + 1e-8)
        push!(its, rep[].iterations)
        @info "UT15 tol_expv $tolx: $(rep[].iterations) applications, estimate $(rep[].err), true $err"
    end
    @test issorted(its)                                  # tighter tolerance, never fewer
    @test its[1] < n                                     # a loose one stops before closure

    @testset "a basis too small for the step restarts instead of truncating the series" begin
        big = ComplexF64(-6.0im)
        rep = Ref{Any}(nothing)
        y = expv(H, big, e[1]; conv_tol = 1e-10, substeps = 0, maxiter = 4, tol = 1e-15,
                 reorth = true, report = rep)
        @test rep[].substeps > 1
        @test norm(to_concrete(y - exact(big))) < 1e-8
        # The historical single shot at the same cap is wrong, and its own estimate says so.
        rep1 = Ref{Any}(nothing)
        y1 = expv(H, big, e[1]; maxiter = 4, tol = 1e-15, reorth = true, report = rep1)
        @test norm(to_concrete(y1 - exact(big))) > 1e-3
        @test rep1[].err > 1e-3
    end
end

# U09 (plan T4.3): the stopping test is the coupling-weighted tail `beta_m |[exp(tau T_m)]_m1|`,
# never a Ritz value. On the zero-diagonal hopping chain `T_m` is the path graph, so every ODD m
# has a Ritz value exactly 0 while the error is still 0.2 (m = 1) or 1e-3 (m = 3), and
# `beta_m = g` at every m, so the coupling alone would not stop before the space closes.
@testset "U09: the tail estimate tracks the exponential's error; a zero Ritz value stops nothing" begin
    g = 0.2
    Mp = ComplexF64.(diagm(1 => fill(g, n - 1), -1 => fill(g, n - 1)))
    Hp, exactp = _known(Mp, e)
    tau = ComplexF64(-1.0im)
    for m in 1:(n - 1)
        rep = Ref{Any}(nothing)
        y = expv(Hp, tau, e[1]; maxiter = m, tol = 1e-15, reorth = true, report = rep)
        err = norm(to_concrete(y - exactp(tau)))
        ritz = minimum(abs, eigvals(SymTridiagonal(zeros(m), fill(g, m - 1))))
        @test rep[].iterations == m
        @test err <= rep[].err * (1 + 1e-8)          # an upper estimate of the true error ...
        @test rep[].err <= (m + 1) * err             # ... and a sharp one (ratio ~ m / |tau|)
        isodd(m) && @test ritz < 1e-12               # a Ritz-value test would stop here ...
        isodd(m) && m <= 3 && @test err > 1e-4       # ... with this error
        @info "U09 m = $m: estimate $(rep[].err), true $err, min |Ritz| $ritz"
    end
    rep = Ref{Any}(nothing)
    y = expv(Hp, tau, e[1]; conv_tol = 1e-4, maxiter = 30, tol = 1e-15, reorth = true,
             report = rep)
    @test rep[].iterations == 5                      # tails: m = 4 2.7e-4, m = 5 1.3e-5
    @test norm(to_concrete(y - exactp(tau))) <= rep[].err <= 1e-4
end

@testset "the step: tol_expv reaches the root solve and its ledger" begin
    mpo, psi0 = _xx_warm(8, 2)
    tau = ComplexF64(-0.1im)
    lg = Dict{Float64, Any}()
    for tolx in (1e-4, 1e-12)
        info = cbe_bug_step!(copy(psi0), mpo, tau; tol_expv = tolx, parallel = true)
        lg[tolx] = info.ledger
        @test info.ledger.expv_iters > 0
        @test info.ledger.expv_err <= tolx * (1 + 1e-8)
        @test !(:expv in info.ledger.unmet)
    end
    @test lg[1e-4].expv_iters <= lg[1e-12].expv_iters
    # The iteration controls are legacy knobs now: refused alongside a tolerance ...
    for kw in ((maxiter = 8,), (tol = 1e-12,), (root_conv_tol = 1e-8,), (root_substeps = 0,))
        @test_throws ArgumentError cbe_bug_step!(copy(psi0), mpo, tau; tol_expv = 1e-10, kw...)
    end
    # ... and still drive the legacy path on their own.
    @test cbe_bug_step!(copy(psi0), mpo, tau; maxiter = 8).ledger === nothing
end

end # adaptive root expv (Stage 5)
