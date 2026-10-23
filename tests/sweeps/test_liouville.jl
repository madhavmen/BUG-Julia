# tests/sweeps/test_liouville.jl -- OS-U1..OS-U7 of the imaginary-time / open-system plan
# (cbe_bug_imaginary_open_benchmark_research (1).md, Task 2): symmetry-aware vectorised density
# matrices and CBE-BUG on them.
#
#   OS-U1  vectorisation identities in the declared (row-major, ket-major) convention, and the
#          product-state / one-site-application path against dense matrices
#   OS-U2  the Lindblad MPO contracted to a dense matrix == the dense Liouvillian (spin XX +
#          dephasing + boundary drive; Bose-Hubbard nmax = 2 with two-body loss)
#   OS-U3  Tr L(rho) = 0, Hermiticity preservation, positivity of dense propagation; MPDO trace /
#          Hermiticity / positivity violations measured after compressed steps
#   OS-U4  weak / strong U(1): covariance refusals, forbidden blocks never present, empty
#          reachable blocks discovered
#   OS-U5  purity law for Hermitian jumps; a nonunital counterexample (purity INCREASES)
#   OS-U6  root solves on a Liouvillian against dense exp / Pade; whole-MPDO BUG temporal order
#   OS-U7  adaptive CBE/Krylov reuse on the Liouville space: counters, rank cap and failed solve
#          are explicit

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

const _LV = BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "common", "liouville_fixtures.jl"))

@testset "symmetry-aware MPDO (plan Task 2)" begin
set_symmetry!(:U1)

@testset "OS-U1: vectorisation convention" begin
    rng = Random.MersenneTwister(1)
    d = 3
    A, B, r = (randn(rng, ComplexF64, d, d) for _ in 1:3)
    @test kron(A, transpose(B)) * _vecr(r) ≈ _vecr(A * r * B)        # row-major (ours)
    @test kron(transpose(B), A) * vec(r) ≈ vec(A * r * B)            # column-major identity
    @test all(liouville_index(d, liouville_pair(d, g)...) == g for g in 1:(d^2))
    @test ket_super(A) * _vecr(r) ≈ _vecr(A * r)
    @test bra_super(B) * _vecr(r) ≈ _vecr(r * B)
    J = randn(rng, ComplexF64, d, d)
    @test dissipator_super(J) * _vecr(r) ≈ _vecr(J * r * J' - (J' * J * r + r * J' * J) / 2)
    # the MPDO path: a product state and a one-site application, against dense matrices
    ls = liouville_space([0, 1, 2]; sym = :none)
    r1, r2 = (x -> x * x')(randn(rng, ComplexF64, d, d)), (x -> x * x')(randn(rng, ComplexF64, d, d))
    rho = mpdo_product_state(ls, [r1, r2])
    @test mpdo_dense(rho, ls) ≈ kron(r1, r2)
    @test mpdo_trace(rho, ls) ≈ tr(r1) * tr(r2)
    mpdo_apply!(rho, 1, ls, ket_super(A) * bra_super(B))             # stored-transpose rule
    @test mpdo_dense(rho, ls) ≈ kron(A * r1 * B, r2)
    @test_throws ArgumentError liouville_space([0, 1]; sym = :bogus)
end

@testset "OS-U2: Lindblad MPO == dense Liouvillian" begin
    cases = Any[]
    for sym in (:weak, :none)
        L = 3
        pairs, hs, jumps = _xx_model(L)
        push!(cases, ("XX+dephasing+drive $sym", liouville_space([1, -1]; sym = sym), L, 2,
                      pairs, hs, jumps))
    end
    L = 3
    pairs, hs, jumps, q, d = _bh_model(L)
    push!(cases, ("Bose-Hubbard nmax=2 loss weak", liouville_space(q; sym = :weak), L, d,
                  pairs, hs, jumps))
    for (name, ls, L, d, pairs, hs, jumps) in cases
        W = lindblad_mpo(ls, L; pairs = pairs, onsite_h = hs, jumps = jumps)
        Wd = _lv_mpo_dense(W, ls, L)
        Ld = _lv_dense(L, d, pairs, hs, jumps)
        P = _lv_perm(L, d)
        # H_eff contracts the state with the MPO's s_ket leg: the stored matrix is the transpose
        # of the applied superoperator, which must be the Liouvillian.
        Lmpo = zeros(ComplexF64, size(Ld))
        Lmpo[P, P] = transpose(Wd)
        err = norm(Lmpo - Ld) / norm(Ld)
        @info "OS-U2 $name: ||L_mpo - L_dense|| / ||L|| = $err"
        @test err < 1e-12
    end
end

@testset "OS-U3: trace, Hermiticity, positivity" begin
    L, d = 3, 2
    pairs, hs, jumps = _xx_model(L)
    Ld = _lv_dense(L, d, pairs, hs, jumps)
    D = d^L
    Iv = _vecr(Matrix{ComplexF64}(I, D, D))
    @test norm(transpose(Iv) * Ld) < 1e-12                          # Tr L(rho) = 0 for all rho
    rng = Random.MersenneTwister(2)
    X = randn(rng, ComplexF64, D, D); r = X * X'; r /= tr(r)
    Lr = _unvecr(Ld * _vecr(r), D)
    @test norm(Lr - Lr') < 1e-12                                    # Hermiticity preserved
    rt = _unvecr(exp(0.7 * Ld) * _vecr(r), D)
    @test minimum(eigvals(Hermitian((rt + rt') / 2))) > -1e-12      # positivity, dense
    @test real(tr(rt)) ≈ 1 atol = 1e-12
    # MPDO violations after COMPRESSED adaptive steps are measured, and within the budgets
    L = 6
    pairs, hs, jumps = _xx_model(L)
    ls = liouville_space([1, -1]; sym = :weak)
    W = lindblad_mpo(ls, L; pairs = pairs, onsite_h = hs, jumps = jumps)
    rho = mpdo_product_state(ls, [isodd(i) ? _NUP : I - _NUP for i in 1:L])
    for _ in 1:4
        cbe_bug_step!(rho, W, ComplexF64(0.1); order = 2, hermitian = false, tol_basis = 1e-6,
                      tol_trunc = 1e-6, tol_expv = 1e-8, maxdim = 64)
    end
    c = mpdo_checks(rho, ls)
    @info "OS-U3 MPDO after 4 compressed steps: $(c)"
    @test !c.local_only
    @test abs(c.trace - 1) < 1e-4
    @test c.herm < 1e-4
    @test c.min_eig > -1e-4
    cl = mpdo_checks(rho, ls; dense = false)                         # the local diagnostics
    @test cl.local_only && cl.herm < 1e-4 && cl.min_eig > -1e-4
end

@testset "OS-U4: weak / strong U(1) blocks" begin
    L = 4
    pairs, hs, jumps = _xx_model(L)
    # the boundary drive breaks the strong symmetry: refused at build time
    @test_throws ArgumentError lindblad_mpo(liouville_space([1, -1]; sym = :strong), L;
                                            pairs = pairs, onsite_h = hs, jumps = jumps)
    pb, hb, jb, qb, db = _bh_model(L)
    @test_throws ArgumentError lindblad_mpo(liouville_space(qb; sym = :strong), L;
                                            pairs = pb, onsite_h = hb, jumps = jb)
    # a symmetry-breaking local state is refused rather than silently projected
    lsw = liouville_space([1, -1]; sym = :weak)
    @test_throws ArgumentError mpdo_product_state(lsw, [fill(0.5 + 0im, 2, 2) for _ in 1:L])
    # number-conserving hopping + dephasing: strong U(1), against dense, block labels fixed
    pairs0, hs0, jumps0 = _xx_model(L; drive = false)
    dense_by_sym = Dict{Symbol, Matrix{ComplexF64}}()
    for sym in (:weak, :strong)
        ls = liouville_space([1, 0]; sym = sym)                      # q = particle number
        W = lindblad_mpo(ls, L; pairs = pairs0, onsite_h = hs0, jumps = jumps0)
        r0 = [isodd(i) ? _NUP : I - _NUP for i in 1:L]
        rho = mpdo_product_state(ls, r0)
        nb0 = length(unique(ql[3] for ql in to_concrete(rho[2]).qlabels))
        # dt = 0.025: BUG3's own temporal error at dt = 0.1 is ~4e-5 here (OS-U6 measures the
        # order separately, s33test 17340370), which would hide a symmetry error of that size
        ts = 0.0
        for _ in 1:24
            cbe_bug_step!(rho, W, ComplexF64(0.025); order = 3, hermitian = false, exact = true,
                          tol_basis = 1e-12, tol_trunc = 1e-14, tol_expv = 1e-13, maxdim = 256)
            ts += 0.025
        end
        Ld = _lv_dense(L, 2, pairs0, hs0, jumps0)
        want = _unvecr(exp(ts * Ld) * _vecr(kron(r0...)), 2^L)
        got = mpdo_dense(rho, ls)
        @info "OS-U4 $sym: dense error $(norm(got - want) / norm(want))"
        @test norm(got - want) / norm(want) < 1e-6
        dense_by_sym[sym] = got
        allowed = Set(first.(ls.sectors))
        @test all(ql[2] in allowed for i in 1:L for ql in to_concrete(rho[i]).qlabels)
        fin = unique(ql[3] for ql in to_concrete(rho[L]).qlabels)
        @test length(fin) == 1                                       # one total charge
        # the total charge, up to Telum's arrow sign on the last link (measured: (-N, -N))
        N = L ÷ 2
        @test fin[1] in (sym === :weak ? (((0,),),) : (((N,), (N,)), ((-N,), (-N,))))
        nb1 = length(unique(ql[3] for ql in to_concrete(rho[2]).qlabels))
        @test nb1 > nb0                                              # empty blocks discovered
    end
    # the grading is storage, not physics: weak and strong give the same state
    @test norm(dense_by_sym[:weak] - dense_by_sym[:strong]) < 1e-10
end

@testset "OS-U5: purity" begin
    L = 4
    # Hermitian jumps (dephasing): d Tr(rho^2)/dt = 2 Re <<rho|L rho>> = -sum_mu ||[L_mu, rho]||_F^2,
    # evaluated EXACTLY with the MPO's own dense matrix (validated in OS-U2) on an evolved state
    pairs0, hs0, jumps0 = _xx_model(L; drive = false)
    ls = liouville_space([1, 0]; sym = :weak)
    W = lindblad_mpo(ls, L; pairs = pairs0, onsite_h = hs0, jumps = jumps0)
    Lm = zeros(ComplexF64, 4^L, 4^L)
    P = _lv_perm(L, 2)
    Lm[P, P] = transpose(_lv_mpo_dense(W, ls, L))
    rho = mpdo_product_state(ls, [isodd(i) ? _NUP : I - _NUP for i in 1:L])
    for _ in 1:3
        cbe_bug_step!(rho, W, ComplexF64(0.1); order = 2, hermitian = false, tol_basis = 1e-8,
                      tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = 256)
    end
    R = mpdo_dense(rho, ls)
    R = (R + R') / 2
    pur(R) = real(tr(R * R))
    gam = jumps0[1][2]
    rate = 2 * real(dot(_vecr(R), Lm * _vecr(R)))
    law = -sum(gam * norm(_lv_emb(_NUP, i, L, 2) * R - R * _lv_emb(_NUP, i, L, 2))^2 for i in 1:L)
    @info "OS-U5 purity rate: MPO $rate, law $law"
    @test rate ≈ law rtol = 1e-10
    @test rate < 0
    # nonunital: amplitude damping from the maximally mixed state PURIFIES -- purity is not monotone
    # (a weak hopping, not zero: pair_mpo must not be handed all-zero vertices)
    Jw = zeros(L, L); for i in 1:(L - 1); Jw[i, i + 1] = 0.2; end
    pairs1 = Any[(_SPm, _SMm, Jw), (_SMm, _SPm, Jw)]
    lsd = liouville_space([1, -1]; sym = :weak)
    Wd = lindblad_mpo(lsd, L; pairs = pairs1, jumps = Any[(_SMm, 1.0)])
    rho = mpdo_product_state(lsd, [Matrix{ComplexF64}(I, 2, 2) / 2 for _ in 1:L])
    p0 = pur(mpdo_dense(rho, lsd))
    cbe_bug_step!(rho, Wd, ComplexF64(0.5); hermitian = false, exact = true, tol_basis = 1e-12,
                  tol_trunc = 1e-14, tol_expv = 1e-14, maxdim = 256)
    p1 = pur(mpdo_dense(rho, lsd))
    @info "OS-U5 amplitude damping purity $p0 -> $p1"
    @test p1 > p0 + 1e-3
end

@testset "OS-U6: roots and temporal order on a Liouvillian" begin
    # L = 2: one bond whose frames span the whole space, so a step IS its root solve
    L = 2
    pairs, hs, jumps = _xx_model(L)
    ls = liouville_space([1, -1]; sym = :weak)
    W = lindblad_mpo(ls, L; pairs = pairs, onsite_h = hs, jumps = jumps)
    Ld = _lv_dense(L, 2, pairs, hs, jumps)
    # ⛔ FROM A PRODUCT STATE THE FRAMES ARE NOT THE WHOLE SPACE (s33test: even :exp was 0.12
    # off), so first grow the bond to its full rank d^2 = 4: then both frames are the complete local
    # spaces and the step IS its root solve.
    warm = mpdo_product_state(ls, [_NUP, I - _NUP])
    for _ in 1:10
        cbe_bug_step!(warm, W, ComplexF64(0.05); order = 2, hermitian = false, exact = true,
                      tol_basis = 1e-12, tol_trunc = 1e-14, tol_expv = 1e-14, maxdim = 64)
    end
    @test maximum(bond_dims(warm)) == 4                              # full rank, non-vacuous
    v0 = _vecr(mpdo_dense(warm, ls))
    for (m, tau) in ((:exp, 0.4), (:radau2, 0.4), (:radau3, 0.4), (:trapezoid, 0.4), (:radau2, 5.0))
        rho = copy(warm)
        # tol_expv at roundoff: the Arnoldi space then closes, so its Ritz shift IS the steady
        # state's eigenvalue 0 and the rational root equals the dense R(tau L)
        cbe_bug_step!(rho, W, ComplexF64(tau); hermitian = false, exact = true, root_method = m,
                      tol_basis = 1e-12, tol_trunc = 1e-14, tol_expv = 1e-15, maxdim = 64)
        got = _vecr(mpdo_dense(rho, ls))
        # the Ritz shift of a converged Arnoldi space is the eigenvalue maximising Re(tau lambda):
        # the steady state's 0 for a trace-preserving generator
        want = m === :exp ? exp(tau * Ld) * v0 :
               BondUpdateBUG._rational_apply(m, tau * Ld, v0)
        e = norm(got - want) / norm(want)
        @info "OS-U6 root $m tau $tau: error vs dense $e"
        @test e < 1e-8
    end
    # whole-MPDO BUG order, L = 5 (L = 4 can saturate to full rank and make every order exact),
    # against dense exp(t L)
    L = 5
    pairs, hs, jumps = _xx_model(L)
    W = lindblad_mpo(ls, L; pairs = pairs, onsite_h = hs, jumps = jumps)
    Ld = _lv_dense(L, 2, pairs, hs, jumps)
    r0 = [isodd(i) ? _NUP : I - _NUP for i in 1:L]
    T = 0.4
    want = exp(T * Ld) * _vecr(kron(r0...))
    errs = Dict{Int, Vector{Float64}}()
    for p in (1, 2, 3)
        errs[p] = Float64[]
        for n in (2, 4, 8)
            rho = mpdo_product_state(ls, r0)
            for _ in 1:n
                cbe_bug_step!(rho, W, ComplexF64(T / n); order = p, hermitian = false,
                              exact = true, tol_basis = 1e-12, tol_trunc = 1e-14,
                              tol_expv = 1e-14, maxdim = 256)
            end
            push!(errs[p], norm(_vecr(mpdo_dense(rho, ls)) - want) / norm(want))
        end
        sl = log2.(errs[p][1:(end - 1)] ./ errs[p][2:end])
        @info "OS-U6 MPDO order $p: errors $(errs[p]), slopes $(sl)"
        @test errs[p][end] < 1e-11 || sl[end] >= p - 0.4
    end
    @test errs[1][end] > 1e-9                                        # non-vacuous: not full rank
    @test errs[2][end] < errs[1][end] / 3                            # the closure buys accuracy
end

@testset "OS-U7: adaptive reuse on the Liouville space" begin
    L = 6
    pairs, hs, jumps = _xx_model(L)
    ls = liouville_space([1, -1]; sym = :weak)
    W = lindblad_mpo(ls, L; pairs = pairs, onsite_h = hs, jumps = jumps)
    rho = mpdo_product_state(ls, [isodd(i) ? _NUP : I - _NUP for i in 1:L])
    sel, cpass, kry = 0, 0, 0
    for _ in 1:3
        info = cbe_bug_step!(rho, W, ComplexF64(0.1); order = 2, hermitian = false,
                             tol_basis = 1e-6, tol_trunc = 1e-8, tol_expv = 1e-10, maxdim = 128)
        lg = info.ledger
        @test budget_status(lg) === :met
        @test lg.proj_err < 1e-7                         # the old state is in the root space
        sel += lg.n_selected; cpass += lg.n_c_pass; kry += lg.n_krylov
    end
    @info "OS-U7 counters over 3 steps: CBE selected $sel, Test C passes $cpass, Krylov added $kry"
    @test sel > 0                                        # CBE found directions from a product state
    @test cpass > 0                                      # Test C skipped what CBE already held
    # a binding rank cap and an unreachable root tolerance are REPORTED
    info = cbe_bug_step!(copy(rho), W, ComplexF64(0.1); hermitian = false, tol_basis = 1e-8,
                         tol_trunc = 1e-12, tol_expv = 1e-10, maxdim = 2)
    @test :trunc in info.ledger.unmet && info.ledger.maxdim_reached
    info = cbe_bug_step!(copy(rho), W, ComplexF64(2.0); hermitian = false, root_method = :radau2,
                         tol_basis = 1e-6, tol_trunc = 1e-8, tol_expv = 1e-30, maxdim = 128)
    @test :expv in info.ledger.unmet
end

@testset "trace vector in the trial space (keep_states)" begin
    L = 6
    pairs, hs, jumps = _xx_model(L)
    ls = liouville_space([1, -1]; sym = :weak)
    W = lindblad_mpo(ls, L; pairs = pairs, onsite_h = hs, jumps = jumps)
    Iket = mpdo_identity(ls, L)
    @test mpdo_trace(Iket, ls) ≈ 2^L
    r0 = mpdo_product_state(ls, [isodd(i) ? _NUP : I - _NUP for i in 1:L])
    # a LOOSE basis so the trial space misses directions; no compression to speak of
    drift = Dict{Bool, Float64}()
    for keep in (false, true)
        rho = copy(r0)
        rep = Ref{Any}(nothing)
        info = cbe_bug_step!(rho, W, ComplexF64(0.3); order = 1, hermitian = false,
                             tol_basis = 1e-2, tol_trunc = 1e-14, tol_expv = 1e-14, maxdim = 512,
                             keep_states = keep ? [Iket] : Any[])
        drift[keep] = abs(mpdo_trace(rho, ls) - 1)
        @test info.ledger.proj_err < 1e-7
    end
    @info "trace drift of one step at tol_basis 1e-2: without |I>> $(drift[false]), with $(drift[true])"
    @test drift[true] < 1e-10                                        # exact before compression
    @test drift[true] < drift[false]
    # ... and the ordinary step is untouched by an empty keep list
    a, b = copy(r0), copy(r0)
    cbe_bug_step!(a, W, ComplexF64(0.1); hermitian = false, tol_basis = 1e-6, keep_states = Any[])
    cbe_bug_step!(b, W, ComplexF64(0.1); hermitian = false, tol_basis = 1e-6)
    @test norm(mpdo_dense(a, ls) - mpdo_dense(b, ls)) == 0
end

@testset "OS-3 model: three-site East term == dense Liouvillian, strong U(1), frozen states" begin
    # H_E = J sum_{j=2}^{L-1} n_{j-1} (s+_j s-_{j+1} + h.c.) + dephasing sqrt(gamma) n_j: the
    # facilitated hop as a TripleTerm (A = n neutral), no two-body term at all
    L, J, gam = 5, 0.9, 0.3
    w = fill(J, L - 2)
    triples = Any[(_NUP, _SPm, _SMm, w), (_NUP, _SMm, _SPm, w)]
    jumps = Any[(_NUP, gam)]
    H = zeros(ComplexF64, 2^L, 2^L)
    for (A, B, C, ww) in triples, j in 1:(L - 2)
        H += ww[j] * _lv_emb(A, j, L, 2) * _lv_emb(B, j + 1, L, 2) * _lv_emb(C, j + 2, L, 2)
    end
    Nop = sum(_lv_emb(_NUP, j, L, 2) for j in 1:L)
    @test norm(H * Nop - Nop * H) < 1e-14                            # [H, N] = 0
    @test norm(H) > 0
    Ld = _lv_dense(L, 2, Any[], [zeros(ComplexF64, 2, 2) for _ in 1:L], jumps)
    Ld += -1im * (kron(H, I(2^L)) - kron(I(2^L), transpose(H)))
    P = _lv_perm(L, 2)
    for sym in (:weak, :strong)
        ls = liouville_space([1, 0]; sym = sym)
        W = lindblad_mpo(ls, L; triples = triples, jumps = jumps)  # strong is accepted: [L_j, N] = 0
        Lmpo = zeros(ComplexF64, size(Ld))
        Lmpo[P, P] = transpose(_lv_mpo_dense(W, ls, L))
        err = norm(Lmpo - Ld) / norm(Ld)
        @info "OS-3 $sym: ||L_mpo - L_dense|| / ||L|| = $err, MPO dims $(mpo_virtual_dims(W))"
        @test err < 1e-12
    end
    # frozen: nothing occupied left of a particle, or a lone particle at site 1, never moves
    for occ in ([1], [L], [1, L])
        r = kron([j in occ ? _NUP : I - _NUP for j in 1:L]...)
        @test norm(Ld * _vecr(r)) < 1e-14
    end
    r = kron([j in (1, 2) ? _NUP : I - _NUP for j in 1:L]...)       # mobile: 1 facilitates 2 -> 3
    @test norm(Ld * _vecr(r)) > 0.1
    # and only facilitated hops: from |1,1,0,0,0> the only state H reaches is |1,0,1,0,0>
    v = zeros(ComplexF64, 2^L); v[1 + sum(1 << (L - j) for j in 1:L if !(j in (1, 2)))] = 1
    reach = findall(x -> abs(x) > 1e-14, H * v)
    @test length(reach) == 1
end

@testset "strong U(1): the block trace vector |I_N>> in the trial space" begin
    L = 6
    triples = Any[(_NUP, _SPm, _SMm, fill(1.0, L - 2)), (_NUP, _SMm, _SPm, fill(1.0, L - 2))]
    jumps = Any[(_NUP, 0.3)]
    ls = liouville_space([1, 0]; sym = :strong)
    W = lindblad_mpo(ls, L; triples = triples, jumps = jumps)
    occ = [1, 2, 4]
    r0 = mpdo_product_state(ls, [j in occ ? _NUP : I - _NUP for j in 1:L])
    IN = mpdo_identity_sector(ls, L, r0)
    @test maximum(bond_dims(IN)) <= L                                # one dimension per sector
    @test real(mpdo_trace(IN, ls)) ≈ binomial(L, length(occ)) atol = 1e-10
    @test_throws ArgumentError mpdo_identity_sector(liouville_space([1, 0]; sym = :weak), L, r0)
    # <<I_N|rho>> = Tr rho on an evolved (entangled, non-product) MPDO of the block
    rho = copy(r0)
    for _ in 1:3
        cbe_bug_step!(rho, W, ComplexF64(0.2); order = 2, hermitian = false, exact = true,
                      tol_basis = 1e-10, tol_trunc = 1e-12, tol_expv = 1e-12, maxdim = 128)
    end
    @test maximum(bond_dims(rho)) > 1                                # non-vacuous
    @test abs(overlap(IN, rho) - mpdo_trace(rho, ls)) < 1e-10 * abs(mpdo_trace(rho, ls))
    # and as keep_states it makes a LOOSE step trace-exact before the compression (cf. |I>>, weak)
    drift = Dict{Bool, Float64}()
    for keep in (false, true)
        r = copy(rho)
        t0 = mpdo_trace(r, ls)
        cbe_bug_step!(r, W, ComplexF64(0.3); order = 1, hermitian = false, tol_basis = 1e-2,
                      tol_trunc = 1e-14, tol_expv = 1e-14, maxdim = 512,
                      keep_states = keep ? [IN] : Any[])
        drift[keep] = abs(mpdo_trace(r, ls) / t0 - 1)
    end
    @info "strong |I_N>>: one-step trace drift at tol_basis 1e-2 without $(drift[false]), with $(drift[true])"
    @test drift[true] < 1e-10
end

@testset "OS-1 reference: dephased boundary-driven XX covariance == dense Liouvillian" begin
    # H = J sum (s+_j s-_{j+1} + h.c.) = J sum (c'_j c_{j+1} + h.c.) (Jordan-Wigner, up = occupied),
    # gain/loss sqrt(eps (1 +- mu)/2) s+- at each end, dephasing sqrt(gamma) n_j: the plan's OS-1
    # model in the jump convention `lindblad_mpo` uses (rate g multiplies D[J]); the closed
    # covariance ODE must reproduce the dense Liouvillian's <n_j> and <c'_k c_{k+1}>.
    L, J, gam, eL, eR, muL, muR = 4, 0.8, 0.35, 0.5, 0.7, 0.6, -0.4
    Jm = zeros(L, L); for i in 1:(L - 1); Jm[i, i + 1] = J; end
    pairs = Any[(_SPm, _SMm, Jm), (_SMm, _SPm, Jm)]
    at(i, g) = [k == i ? g : 0.0 for k in 1:L]
    jumps = Any[(_NUP, gam), (_SPm, at(1, eL * (1 + muL) / 2)), (_SMm, at(1, eL * (1 - muL) / 2)),
                (_SPm, at(L, eR * (1 + muR) / 2)), (_SMm, at(L, eR * (1 - muR) / 2))]
    Ld = _lv_dense(L, 2, pairs, [zeros(ComplexF64, 2, 2) for _ in 1:L], jumps)
    occ = [1, 3]
    r0 = [i in occ ? _NUP : I - _NUP for i in 1:L]
    C0 = Diagonal([i in occ ? 1.0 : 0.0 for i in 1:L])
    ts = [0.3, 0.6, 2.0]
    Cs = xx_open_covariance(L, ts; C0 = C0, J = J, gamma = gam, eps_L = eL, eps_R = eR,
                            mu_L = muL, mu_R = muR)
    for (t, C) in zip(ts, Cs)
        R = _unvecr(exp(t * Ld) * _vecr(kron(r0...)), 2^L)
        n = [real(tr(_lv_emb(_NUP, j, L, 2) * R)) for j in 1:L]
        hop = [tr(_lv_emb(_SPm, k, L, 2) * _lv_emb(_SMm, k + 1, L, 2) * R) for k in 1:(L - 1)]
        @test maximum(abs.(n .- real.(diag(C)))) < 1e-12
        @test maximum(abs.(hop .- [C[k, k + 1] for k in 1:(L - 1)])) < 1e-12
        @test norm(hop) > 1e-3                                       # non-vacuous: a current flows
    end
    # gamma = 0 is the published boundary-driven reference; the NESS is the long-time limit
    C1 = xx_open_covariance(L, [1.5]; J = J, eps_L = eL, eps_R = eR, mu_L = muL, mu_R = muR)[1]
    @test norm(C1 - xx_boundary_covariance(L, 1.5; J = J, eps_L = eL, eps_R = eR, mu_L = muL,
                                           mu_R = muR)) < 1e-12
    Cinf = xx_open_covariance(L, [400.0]; J = J, gamma = gam, eps_L = eL, eps_R = eR,
                              mu_L = muL, mu_R = muR)[1]
    @test norm(Cinf - xx_open_ness(L; J = J, gamma = gam, eps_L = eL, eps_R = eR, mu_L = muL,
                                   mu_R = muR)) < 1e-10
end

end # symmetry-aware MPDO
