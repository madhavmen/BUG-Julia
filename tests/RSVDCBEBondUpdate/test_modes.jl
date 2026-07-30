using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime, ImaginaryTime, GroundState
import Random

# The three modes share ONE engine (`_expanding_krylov`), differing only in the operator the
# local problem is posed with and the reduction applied to the tridiagonal. These tests are
# therefore aimed at the seams rather than at the engine, which the other files already cover:
#
#   * the mode wrappers add NO behaviour -- `RealTime.evolve!` must be `==` to the historical
#     `cbe_bond_update_bug!`, not merely close to it, or the refactor moved something;
#   * `Propagate` and `LowestEigen` must reduce the SAME Krylov space consistently;
#   * the ground-state energy must be a VARIATIONAL UPPER BOUND, which is the one property the
#     two-phase local solve exists to preserve and the one a single run can silently lose.

"Exact ground energy in the half-filling sector, by dense diagonalisation."
function exact_ground_energy(L::Int; delta::Float64 = 1.0, J::Float64 = 1.0)
    sz(s) = s == 1 ? 0.5 : -0.5
    bit(b, j) = (b >> (j - 1)) & 1
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    idx = Dict(s => k for (k, s) in enumerate(states))
    H = zeros(Float64, length(states), length(states))
    for (k, b) in enumerate(states), j in 1:(L - 1)
        sj, sj1 = bit(b, j), bit(b, j + 1)
        H[k, k] += J * delta * sz(sj) * sz(sj1)
        sj == sj1 || (H[idx[b ⊻ (1 << (j - 1)) ⊻ (1 << j)], k] += J / 2)
    end
    return eigen(Symmetric(H)).values[1]
end

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]
hb_gates(p, d) = Any[heisenberg_bond_gate(p[i].inds[2], p[i + 1].inds[2]; delta = d)
                     for i in 1:(length(p) - 1)]

@testset "modes" begin

    @testset "RealTime.evolve! IS cbe_bond_update_bug!" begin
        set_symmetry!(:U1)
        L = 6
        # Both solver paths: the one-shot default and the expanding-basis solver, since the
        # refactor rewired the call into `_expanding_krylov` only on the second.
        for kw in (NamedTuple(), (; s_iters = -2))
            opts = CBEBugOptions(; dt = 0.02, n_steps = 3, maxdim = 16,
                                 trunc_thresh = 1e-13, seed = 0xABCD, kw...)
            a = domain_wall_state(L); cbe_bond_update_bug!(a, xx_gates(a); opts = opts)
            b = domain_wall_state(L); RealTime.evolve!(b, xx_gates(b); opts = opts)
            # `==` on the observable, not isapprox: the wrapper must not perturb a single bit.
            @test magnetisation(copy(a)) == magnetisation(copy(b))
        end
    end

    @testset "cbe_gate_evolve! at tau = -im*dt reproduces the real-time driver" begin
        set_symmetry!(:U1)
        L = 6
        opts = CBEBugOptions(dt = 0.02, n_steps = 3, maxdim = 16, trunc_thresh = 1e-13,
                             seed = 0xABCD)
        a = domain_wall_state(L); cbe_bond_update_bug!(a, xx_gates(a); opts = opts)
        b = domain_wall_state(L)
        cbe_gate_evolve!(b, xx_gates(b), ComplexF64(-im * opts.dt); opts = opts)
        @test magnetisation(copy(a)) == magnetisation(copy(b))
    end

    @testset "imaginary time refuses to run unnormalised" begin
        set_symmetry!(:U1)
        p = neel_state(6)
        # Not a style preference: exp(-dt*H) is a contraction, so without the rescale the
        # state decays to underflow and the run silently returns garbage.
        @test_throws ArgumentError ImaginaryTime.evolve!(p, hb_gates(p, 1.0);
            opts = CBEBugOptions(dt = 0.05, n_steps = 2, normalize = false))
    end

    @testset "imaginary time records a decaying norm and a falling energy" begin
        set_symmetry!(:U1)
        L = 6
        p = neel_state(L)
        info = ImaginaryTime.evolve!(p, hb_gates(p, 1.0);
            opts = CBEBugOptions(dt = 0.05, n_steps = 40, maxdim = 16,
                                 trunc_thresh = 1e-12, s_iters = -2))
        @test length(info.energies) == 40
        # Monotone descent is the whole point of cooling; allow a hair of slack for the
        # truncation, which is not variational.
        @test all(info.energies[k + 1] <= info.energies[k] + 1e-9
                  for k in 1:(length(info.energies) - 1))
        @test info.energies[end] < info.energies[1]
        # Real time leaves this empty; imaginary time must not.
        q = domain_wall_state(L)
        rt = RealTime.evolve!(q, xx_gates(q);
                              opts = CBEBugOptions(dt = 0.02, n_steps = 2, maxdim = 16))
        @test isempty(rt.energies)
    end

    @testset "imaginary time cools to the exact ground energy" begin
        set_symmetry!(:U1)
        L = 6
        e_exact = exact_ground_energy(L; delta = 1.0)
        p = neel_state(L)
        info = ImaginaryTime.evolve!(p, hb_gates(p, 1.0);
            opts = CBEBugOptions(dt = 0.02, n_steps = 1500, maxdim = 24,
                                 trunc_thresh = 1e-13, s_iters = -2))
        # beta = 30 at dt = 0.02: past the point where residual excited-state weight
        # dominates, so what is left is the O(dt^2) Trotter error of the state -- which shows
        # up in the ENERGY as O(dt^4) because the energy is stationary about the ground state.
        @test info.energies[end] ≈ e_exact atol = 1e-6
    end

    @testset "ground state is a variational upper bound" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L; delta = 1.0)
        e_exact = exact_ground_energy(L; delta = 1.0)
        # `grow_iters = 0` is EXCLUDED here on purpose -- it is the no-growth control and has
        # its own testset below, because it legitimately does not reach the ground state.
        for grow_iters in (1, 2, 3)
            p = neel_state(L)
            info = GroundState.solve!(p, h;
                opts = CBEBugOptions(maxdim = 24, trunc_thresh = 1e-13),
                n_sweeps = 30, etol = 1e-12, grow_iters = grow_iters)
            e = info.energies[end]
            # THE ASSERTION THE TWO-PHASE LOCAL SOLVE EXISTS FOR. Phase B runs in FIXED
            # frames, so the Ritz value is an exact Rayleigh-Ritz and cannot fall below the
            # true ground energy. A negative residual here means the tridiagonal went
            # inconsistent -- it is a bug, not a lucky sweep. The tolerance is for the
            # truncation, which is the one non-variational step in the sweep.
            @test e >= e_exact - 1e-10
            @test e ≈ e_exact atol = 1e-8
        end
    end

    @testset "grow_iters = 0 freezes at the product state -- CBE growth is load-bearing" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L; delta = 1.0)
        p = neel_state(L)
        info = GroundState.solve!(p, h;
            opts = CBEBugOptions(maxdim = 24, trunc_thresh = 1e-13),
            n_sweeps = 30, etol = 1e-12, grow_iters = 0)

        # With no growth pass the frames never widen, so the local eigensolve has a
        # one-dimensional space to minimise over and the sweep cannot leave the start state.
        # It then "converges" immediately, which is why `info.converged` is NOT evidence of
        # anything on its own.
        #
        # The energy it reports is exactly the Neel product-state energy: each of the L-1
        # bonds contributes delta*sz*sz = -0.25 with antialigned neighbours.
        @test maximum(info.max_bond_dims) == 1
        @test info.energies[end] ≈ -0.25 * (L - 1) atol = 1e-12
        @test info.energies[end] > exact_ground_energy(L; delta = 1.0) + 0.1
        @test sum(info.grow_passes) == 0

        # This is the DMRG analogue of the standard-BUG chi=1 freeze: the CBE expansion is
        # doing ALL the rank generation, so one growth pass is the minimum viable setting.
        q = neel_state(L)
        one = GroundState.solve!(q, h;
            opts = CBEBugOptions(maxdim = 24, trunc_thresh = 1e-13),
            n_sweeps = 30, etol = 1e-12, grow_iters = 1)
        @test one.energies[end] ≈ exact_ground_energy(L; delta = 1.0) atol = 1e-8
        @test maximum(one.max_bond_dims) > 1
    end

    @testset "ground state converges and reports it" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L; delta = 0.5)
        p = neel_state(L)
        info = GroundState.solve!(p, h;
            opts = CBEBugOptions(maxdim = 24, trunc_thresh = 1e-13),
            n_sweeps = 40, etol = 1e-11, grow_iters = 2)
        @test info.converged
        @test length(info.energies) < 40           # stopped early, did not run out of sweeps
        @test info.energies[end] ≈ exact_ground_energy(L; delta = 0.5) atol = 1e-8
        # Growth passes actually fired, i.e. the CBE expansion is in the loop and not skipped.
        @test sum(info.grow_passes) > 0
    end

    @testset "ground state matches imaginary-time cooling" begin
        set_symmetry!(:U1)
        L = 6
        delta = 1.0
        h = xxz_chain(L; delta = delta)
        a = neel_state(L)
        gs = GroundState.solve!(a, h;
            opts = CBEBugOptions(maxdim = 24, trunc_thresh = 1e-13),
            n_sweeps = 30, etol = 1e-12, grow_iters = 2)

        b = neel_state(L)
        it = ImaginaryTime.evolve!(b, hb_gates(b, delta);
            opts = CBEBugOptions(dt = 0.02, n_steps = 1500, maxdim = 24,
                                 trunc_thresh = 1e-13, s_iters = -2))

        # Two genuinely different routes to the same state: a local eigensolve on the
        # environment-dressed H, and Trotterised cooling on bare gates. They agree only to
        # the cooling route's O(dt^2) splitting error, so this is a loose but meaningful
        # cross-check -- the ground state has no splitting error at all.
        @test gs.energies[end] ≈ it.energies[end] atol = 1e-5
        # And the eigensolve should be the tighter of the two.
        @test gs.energies[end] <= it.energies[end] + 1e-9
    end
end
