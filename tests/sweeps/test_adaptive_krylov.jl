# tests/sweeps/test_adaptive_krylov.jl -- UT07..UT10 of the adaptive CBE-Krylov plan (§4, §7),
# plus the Stage 3 end-to-end checks on `cbe_bug_step!`.
#
#   UT07  CBE already holds `H1*Kex`: Test C is zero and nothing redundant is appended
#   UT08  the plan's counterexample -- Test C zero, Test D finds `e3` (subject to budget)
#   UT09  a genuinely new direction: orthonormal, rank grows only by what is significant
#   UT10  old-subspace preservation and index transport on BOTH half-sweeps, with the real `H1`
#
# ⛔ UT07-UT09 RUN THE PRODUCTION `_krylov_enrich` ON REAL BLOCK-SPARSE TENSORS. The plan's
# examples are statements about an operator (`H e1 = e2`, `H e2 = e1 + e3`), so the operator is
# built as a matrix-free map from orthonormal one-site tensors of a real bond -- the frame
# projector, the charge blocks and the SVDs are the ones the sweep uses, only `H1` is chosen.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))
include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))

_perp(side) = side === :left ? perp_component : perp_component_right
_fleg(side) = side === :left ? 3 : 1
_nrmz(x) = to_concrete((1.0 / norm(x)) * x)
_kalloc(kry) = RC.BasisAllocation(1.0, 1.0, 1.0, kry, 1.0)     # only the Krylov share matters

"Number of local directions `T` spans on `side`'s frame legs (numerical rank)."
_framerank(T, side) = length(svd(T, side === :left ? (1, 2) : (2, 3);
                                 cutoff = 1e-10, get_lists = true).kept_list)

"""
Orthonormal one-site tensors on a real bond: `e1` along the seed, `e2` inside `frame` and
orthogonal to `e1`, then `nout` more whose local range lies OUTSIDE `frame` (and which are
therefore orthogonal to both). All share the seed's charge blocks.
"""
function _synthetic(seed, frame, side, nout; rng = Random.MersenneTwister(17))
    P = _perp(side)
    e1 = _nrmz(seed)
    X = RC._randomize_like(seed, rng)
    inside = to_concrete(X - to_concrete(P(frame, X)))
    e2 = _nrmz(to_concrete(inside - tensor_inner(e1, inside) * e1))
    outs = Any[]
    for _ in 1:nout
        Y = to_concrete(P(frame, RC._randomize_like(seed, rng)))
        for u in outs
            Y = to_concrete(Y - tensor_inner(u, Y) * u)
        end
        push!(outs, _nrmz(Y))
    end
    return e1, e2, outs
end

@testset "adaptive Krylov (Stage 3)" begin

@testset "UT07: CBE already holds H1*Kex -- no redundant vector" begin
    for side in (:left, :right)
        kl = _xx_kl(side; steps = 1)
        e1, e2, _ = _synthetic(kl.seed, kl.old, side, 0)
        H = _hop([(1.0, e2, e1)])                       # H e1 = e2, inside the frame
        Q, st = RC._krylov_enrich(kl.seed, H, kl.old, side, ComplexF64(-1im), _kalloc(1e-8))
        @test st.eta_c < 1e-12                          # Test C: nothing new
        @test st.added == 0                             # and nothing appended
        @test leg_dim(Q, _fleg(side)) == leg_dim(kl.old, _fleg(side))
        @test st.met
        @test st.n_h1 == 2                              # H e2 = e1 closes the recursion
    end
end

@testset "UT08: Test C is zero, Test D still finds the second direction" begin
    for side in (:left, :right)
        kl = _xx_kl(side; steps = 1)
        e1, e2, outs = _synthetic(kl.seed, kl.old, side, 1)
        e3 = outs[1]
        @test norm(_perp(side)(kl.old, e3)) > 0.5        # non-vacuous: e3 is outside the frame
        H = _hop([(1.0, e2, e1), (1.0, e3, e2)])       # H e1 = e2, H e2 = e1 + e3

        @testset "at unit time e3 matters and is appended ($side)" begin
            Q, st = RC._krylov_enrich(kl.seed, H, kl.old, side, ComplexF64(-1im),
                                      _kalloc(1e-6))
            @test st.eta_c < 1e-12                      # a sufficient-looking Test C ...
            @test st.eta_d > 1e-6                       # ... and Test D disagrees
            @test st.added >= 1
            @test st.d_after_c
            # The frame now holds e3 -- to the allowance, not exactly: the tail rule may drop
            # directions whose WEIGHTED tail fits (weight of e3 ~ 0.46 here, so <= ~2e-6).
            @test norm(_perp(side)(Q, e3)) < 5e-6
            @test st.met
            lg = AdaptiveLedger()
            RC._record_krylov!(lg, st, _kalloc(1e-6))
            @test lg.n_c_pass == 1 && lg.n_d_after_c == 1 && lg.n_krylov == st.added
        end

        @testset "at a tiny step e3 is below budget and is NOT appended ($side)" begin
            # weight of e3 ~ tau^2/2 = 5e-9 < 1e-6: "subject to budget", per the plan
            Q, st = RC._krylov_enrich(kl.seed, H, kl.old, side, ComplexF64(-1e-4im),
                                      _kalloc(1e-6))
            @test st.added == 0
            @test st.met
        end
    end
end

@testset "UT09: a genuinely new direction, and only the significant part of it" begin
    for side in (:left, :right)
        kl = _xx_kl(side; steps = 1)
        e1, _, outs = _synthetic(kl.seed, kl.old, side, 2)
        e3, e4 = outs
        # H e1 = e3 + 1e-12 e4: one real new direction and a negligible second one.
        H = _hop([(1.0, e3, e1), (1e-12, e4, e1)])
        Q, st = RC._krylov_enrich(kl.seed, H, kl.old, side, ComplexF64(-1im), _kalloc(1e-6))
        P = _perp(side)
        @test st.added >= 1
        @test st.added <= _framerank(e3, side)         # e4 bought no extra rank
        @test norm(P(Q, e3)) < 2e-6                     # captured to the allowance
        @test norm(P(Q, kl.old)) < 1e-12                # old frame kept
        # Orthonormal, and orthogonal to what was there: the guard before the direct sum.
        @test (side === :left ? left_isometry_defect(Q) : right_isometry_defect(Q)) < 1e-12
        # Monotone in the allowance: a looser budget never buys more rank.
        _, st_loose = RC._krylov_enrich(kl.seed, H, kl.old, side, ComplexF64(-1im),
                                        _kalloc(1e-1))
        @test st_loose.added <= st.added
    end
end

@testset "UT10: old subspace and transport on both half-sweeps, real H1" begin
    tau = ComplexF64(-0.05im)
    kry = 1e-7
    for side in (:left, :right)
        # `dex = 1`: at the default budget CBE fills the whole local space at this bond (job
        # 17192249: frame 16 -> 16, CBE alone missed 1.8e-16), which left nothing to test.
        kl = _xx_kl(side; steps = 2, dex = 1)
        P = _perp(side)
        Y = expv(kl.applyH, tau, kl.seed; maxiter = 40, tol = 1e-15)
        miss0 = norm(P(kl.Q0, Y)) / norm(kl.seed)
        @test miss0 > 100 * kry                         # non-vacuous: CBE alone is not enough
        Q, st = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(kry))
        @test st.met
        @test norm(P(Q, kl.Q0)) < 1e-12                 # the CBE frame is kept whole ...
        @test norm(P(Q, kl.old)) < 1e-12                # ... and with it the old directions
        @test norm(P(Q, kl.seed)) < 1e-12 * norm(kl.seed)   # the seed is in the frame
        @test (side === :left ? left_isometry_defect(Q) : right_isometry_defect(Q)) < 1e-12
        # THE POINT OF TESTS C/D: the frame captures the locally EVOLVED seed to the allowance.
        miss = norm(P(Q, Y)) / norm(kl.seed)
        @test miss <= 10 * kry
        @test st.added >= 1
        @info "UT10 $side: frame $(leg_dim(kl.Q0, _fleg(side))) -> $(leg_dim(Q, _fleg(side))), " *
              "eta_C = $(st.eta_c), eta_D = $(st.eta_d), H1 applications $(st.n_h1), " *
              "evolved-seed miss $(miss) (CBE frame alone: " *
              "$(miss0))"
        # An allowance everything fits: the CBE frame is returned unchanged.
        Qloose, stl = RC._krylov_enrich(kl.seed, kl.applyH, kl.Q0, side, tau, _kalloc(1e3))
        @test stl.added == 0
        @test leg_dim(Qloose, _fleg(side)) == leg_dim(kl.Q0, _fleg(side))
    end
end

@testset "Stage 3 end to end: cbe_bug_step! on XX / U(1)" begin
    set_symmetry!(:U1)
    L = 12
    mpo, psi0 = xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L)
    sz(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())

    @testset "accuracy against the free-fermion closed form, parallel half-sweeps" begin
        psi = copy(psi0)
        dt, T = 0.05, 1.0
        sz0 = sum(sz(psi))
        lg = AdaptiveLedger()
        for _ in 1:round(Int, T / dt)
            info = cbe_bug_step!(psi, mpo, ComplexF64(-im * dt); parallel = true)
            merge_ledger!(lg, info.ledger)
        end
        err = maximum(abs.(sz(psi) .- xx_free_fermion_sz(L, T; J = 1.0, occupied = 1:(L ÷ 2))))
        @info "Stage 3 XX L=$L T=$T dt=$dt: err $(err), chi $(maximum(bond_dims(psi))), " *
              "H1 applications $(lg.n_h1), Krylov directions $(lg.n_krylov), " *
              "Test C passed $(lg.n_c_pass), D grew after C $(lg.n_d_after_c), " *
              "eta_C max $(lg.max_krylov_c), eta_D max $(lg.max_krylov_d), " *
              "tolerance_met $(lg.tolerance_met), workcap $(lg.workcap_reached)"
        @test err < 5e-4                                # the legacy arm's bar (measured 1.7e-4)
        @test abs(sum(sz(psi)) - sz0) < 1e-10           # U(1) charge exactly conserved
        @test isapprox(norm(psi), 1.0; atol = 1e-8)
        @test lg.n_h1 > 0                               # the Krylov stage actually ran
        @test isfinite(lg.max_krylov_c) && isfinite(lg.max_krylov_d)
    end

    @testset "the fixed-depth Krylov knobs are legacy-only" begin
        psi = copy(psi0)
        dt = ComplexF64(-0.05im)
        for kw in ((krylov_tol = 1e-6,), (grow_iters = 2,), (krylov_grow = true,))
            @test_throws ArgumentError cbe_bug_step!(psi, mpo, dt; tol_basis = 1e-5, kw...)
        end
        # ...and still drive the legacy path on their own.
        @test cbe_bug_step!(copy(psi0), mpo, dt; krylov_basis = 3).ledger === nothing
        @test cbe_bug_step!(copy(psi0), mpo, dt; krylov_tol = 1e-6).ledger === nothing
    end
end

end # adaptive Krylov (Stage 3)
