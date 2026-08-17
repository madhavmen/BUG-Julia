using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# A GENUINELY LONG-RANGE HAMILTONIAN through the whole stack, which is what the MPO layer
# was built for and what the factored channel form of `henv.jl` cannot express at all.
#
#     H = J sum_i (Sx Sx + Sy Sy)_{i,i+1}  +  Jz sum_{i<j} lambda^(j-i-1) Sz_i Sz_j
#
# Every pair of sites interacts. The MPO carries that with ONE self-loop channel, so the
# virtual dimension is 5 whatever L is -- there is no per-distance channel and nothing in
# the sweep, the environments or the CBE selection knows the range changed.
#
# WHAT MAKES THIS A TEST RATHER THAN A DEMONSTRATION. The dense reference sums the pairs
# with an explicit double loop that shares nothing with the automaton, so the geometric
# tail `lambda^(j-i-1)` has to come out right at EVERY distance for the energy to match --
# a wrong self-loop (missing, applied at the wrong site, or off by one power) changes the
# tail and cannot cancel. The `lambda = 0` arm then ties the new builder to the validated
# nearest-neighbour one, and the full-rank arm ties the ENVIRONMENTS to the operator: at
# L=4 with saturated bonds a 0-site Galerkin step must reproduce `exp(-i H dt)` exactly,
# and `H` there is reached only through Eq. (1.8)'s recursion.

"A state with real entanglement: a few validated BUG steps off a product state."
function _lr_evolved(L::Int; n_steps = 6, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

@testset "long-range H: exponential tail as one self-loop channel" begin

    @testset "the virtual dimension does not grow with L or with the range" begin
        set_symmetry!(:U1)
        # 2 XY terms + 1 long-range ZZ channel + `id` + `done` = 5, for every L.
        for L in (3, 4, 6, 8)
            mpo = long_range_zz_mpo(L; lambda = 0.5)
            dims = mpo_virtual_dims(mpo)
            @test length(mpo) == L
            @test dims[1] == 1 && dims[end] == 1
            @test all(==(5), dims[2:(end - 1)])
            for i in 1:L
                @test length(mpo[i].inds) == 4
            end
        end
    end

    @testset "the energy matches the explicit pair sum at every distance" begin
        # THE CENTRAL ASSERTION. `dense_long_range_zz` loops over all i<j independently of
        # the automaton, so agreement means the self-loop reproduces `lambda^(j-i-1)` for
        # every separation, not just the nearest neighbour.
        set_symmetry!(:U1)
        for L in (3, 4, 6), lambda in (0.0, 0.3, 0.5, 0.9), Jz in (1.0, -0.7)
            psi = _lr_evolved(L)
            v = dense_state(psi)
            H = dense_long_range_zz(L; Jz = Jz, lambda = lambda)
            got = mpo_energy(psi, long_range_zz_mpo(L; Jz = Jz, lambda = lambda))
            @test abs(got - dot(v, H * v)) < 1e-10 * max(1.0, abs(dot(v, H * v)))
        end
    end

    @testset "the tail is really there: lambda changes the operator" begin
        # Guards against the whole long-range block being inert -- a self-loop that never
        # fires would still pass the `lambda = 0` arm and the structure arm.
        set_symmetry!(:U1)
        L = 6
        psi = _lr_evolved(L)
        e0 = mpo_energy(psi, long_range_zz_mpo(L; lambda = 0.0))
        e5 = mpo_energy(psi, long_range_zz_mpo(L; lambda = 0.5))
        e9 = mpo_energy(psi, long_range_zz_mpo(L; lambda = 0.9))
        @test abs(e5 - e0) > 1e-3
        @test abs(e9 - e5) > 1e-3
        # Monotone in lambda for this state, and the L=2 case has no tail to add at all.
        psi2 = _lr_evolved(2)
        @test abs(mpo_energy(psi2, long_range_zz_mpo(2; lambda = 0.0)) -
                  mpo_energy(psi2, long_range_zz_mpo(2; lambda = 0.9))) < 1e-12
    end

    @testset "lambda = 0 IS the nearest-neighbour MPO" begin
        # Ties the new builder to the already-validated one: same state, same operator.
        set_symmetry!(:U1)
        for L in (3, 4, 6)
            psi = _lr_evolved(L)
            lr = long_range_zz_mpo(L; J = 1.0, Jz = 0.7, lambda = 0.0)
            nn = xxz_mpo(L; J = 1.0, delta = 0.7)
            @test mpo_virtual_dims(lr) == mpo_virtual_dims(nn)
            @test abs(mpo_energy(psi, lr) - mpo_energy(psi, nn)) < 1e-12
            # Elementwise, not just the scalar: the two-site action at every bond.
            lst, rst = left_env_stack(psi, lr), right_env_stack(psi, lr)
            lsn, rsn = left_env_stack(psi, nn), right_env_stack(psi, nn)
            for i in 1:(L - 1)
                Theta = to_concrete(contract(psi[i], (3,), psi[i + 1], (1,)))
                a = apply_h_two_site(Theta, lr, i, left_channels(lst, i),
                                     right_channels(rst, i + 2))
                b = apply_h_two_site(Theta, nn, i, left_channels(lsn, i),
                                     right_channels(rsn, i + 2))
                @test norm(to_concrete(a - b)) < 1e-12
            end
        end
    end

    @testset "one full-rank step reproduces exp(-i H dt) through the environments" begin
        # The environments are the only route to H here, so this is Eq. (1.8) under test:
        # at L=4 with saturated bonds the centre projector is the identity and the single
        # 0-site Galerkin step must be the exact propagator of the LONG-RANGE H.
        set_symmetry!(:U1)
        L, dt = 4, 0.02
        for lambda in (0.0, 0.5, 0.9)
            mpo = long_range_zz_mpo(L; lambda = lambda)
            psi = _lr_evolved(L; n_steps = 8, dt = 0.05, maxdim = 32)
            @test bond_dims(psi) == [min(2^i, 2^(L - i)) for i in 1:(L - 1)]
            v0 = dense_state(psi)
            want = exp(-im * dt * dense_long_range_zz(L; lambda = lambda)) * v0
            cbe_lubich_sweep(psi, mpo, ComplexF64(-im * dt);
                             maxdim = 32, rng = Random.MersenneTwister(0x5EED))
            @test norm(dense_state(psi) - want) < 1e-9
        end
    end

    @testset "the integrators run on it and track the exact propagator" begin
        set_symmetry!(:U1)
        L, dt, n = 6, 0.01, 6
        lambda = 0.5
        H = dense_long_range_zz(L; lambda = lambda)
        for (name, run!) in (("cbe_lubich", cbe_lubich_bug!), ("jan", jan_bug!))
            psi = _lr_evolved(L)
            v = dense_state(psi)
            run!(psi, long_range_zz_mpo(L; lambda = lambda);
                 opts = CBEBugOptions(dt = dt, n_steps = n, maxdim = 32,
                                      normalize = false))
            err = norm(dense_state(psi) - exp(-im * (n * dt) * H) * v)
            @info "long-range multi-step" name err standstill =
                norm(exp(-im * (n * dt) * H) * v - v)
            @test err < 0.2 * norm(exp(-im * (n * dt) * H) * v - v)
        end
    end

    @testset "a charged long-range channel is refused, not silently mis-built" begin
        # `Sp` is rank 3 under U(1): its op-leg would BE the channel's virtual space, so the
        # self-loop would have to be the identity on a charged space rather than the trivial
        # `lambda * I` this builds. Refusing is the honest behaviour; see `LongRangeTerm`.
        set_symmetry!(:U1)
        q = local_space()
        @test length(q.Sp.inds) == 3                       # the premise of the refusal
        @test_throws ArgumentError long_range_mpo(
            4, xxz_chain(4; delta = 0.0),
            [LongRangeTerm(q.Sp, to_concrete(q.Sp'), 1.0, 0.5)])
        # ... and the supported case is accepted.
        @test long_range_mpo(4, xxz_chain(4; delta = 0.0),
                             [LongRangeTerm(q.Sz, q.Sz, 1.0, 0.5)]) isa MPO
    end

    # ── the APPROXIMATE build, and its error ─────────────────────────────────────
    #
    # A geometric tail is EXACT, so the tests above assert machine precision and there is no
    # error to quote. A power law is not: it has no finite-w MPO, and fitting it by `n`
    # geometrics introduces an error in the HAMILTONIAN that no integrator can undo and that
    # does not show up in any convergence test of the integrator itself. These testsets
    # measure it at three levels -- the coupling, the operator, and the propagated state --
    # and check the one thing that makes the construction usable: that all three shrink
    # together as `n_exp` grows.

    @testset "the fit error is reported, and shrinks with the number of channels" begin
        set_symmetry!(:U1)
        L, alpha = 8, 3.0
        errs = Float64[]
        for n_exp in (1, 2, 3, 4, 6)
            _, fit = power_law_zz_mpo(L; alpha = alpha, n_exp = n_exp)
            @test length(fit.coeffs) == n_exp
            @test length(fit.rel_errs) == L - 1
            # `fitted` really is the sum the MPO will carry, not a stored guess.
            for r in 1:(L - 1)
                s = sum(fit.coeffs[k] * fit.decays[k]^(r - 1) for k in 1:n_exp)
                @test abs(s - fit.fitted[r]) < 1e-12
            end
            push!(errs, fit.max_rel_err)
        end
        @info "power-law fit error vs channels" alpha n_exp = [1, 2, 3, 4, 6] errs
        @test issorted(errs; rev = true)               # more channels, never worse
        @test errs[end] < 0.05                         # 6 channels resolve 1/r^3 to <5%
    end

    @testset "the MPO carries the FITTED H exactly; the gap to the true H is the fit error" begin
        # Two distinct statements, and separating them is the point:
        #   (a) the MPO is an exact representation of the fitted coupling -- machine
        #       precision, because the automaton is exact for any sum of geometrics;
        #   (b) the fitted coupling is not the power law -- and THAT discrepancy is what
        #       `fit.max_rel_err` predicts.
        # Conflating them would let an automaton bug hide inside the fit error.
        set_symmetry!(:U1)
        L, alpha = 6, 3.0
        psi = _lr_evolved(L)
        v = dense_state(psi)
        gaps = Float64[]
        for n_exp in (1, 2, 4, 6)
            mpo, fit = power_law_zz_mpo(L; alpha = alpha, n_exp = n_exp)
            got = mpo_energy(psi, mpo)

            # (a) against the coupling the MPO actually carries
            H_fit = dense_xy_chain(L) .+ dense_zz_tail(L, fit.fitted)
            @test abs(got - dot(v, H_fit * v)) < 1e-10 * max(1.0, abs(dot(v, H_fit * v)))

            # (b) against the power law it is meant to approximate
            H_true = dense_power_law_zz(L; alpha = alpha)
            push!(gaps, abs(got - dot(v, H_true * v)) / abs(dot(v, H_true * v)))
        end
        @info "power-law energy gap vs channels" n_exp = [1, 2, 4, 6] gaps
        @test issorted(gaps; rev = true)
        @test gaps[end] < 1e-3
    end

    @testset "the fit error propagates to the state, and bounds the run" begin
        # What the fit error costs an actual simulation. Evolving under the fitted MPO and
        # comparing with the TRUE power-law propagator: the residual is the model error, not
        # an integrator error, so it does NOT go away with smaller dt -- it goes away with
        # more channels. Measured both ways so the two cannot be confused.
        set_symmetry!(:U1)
        L, alpha, dt, n = 6, 3.0, 0.01, 6
        H_true = dense_power_law_zz(L; alpha = alpha)
        base = _lr_evolved(L)
        v = dense_state(base)
        want = exp(-im * (n * dt) * H_true) * v
        model_errs = Float64[]
        for n_exp in (1, 2, 4, 6)
            mpo, fit = power_law_zz_mpo(L; alpha = alpha, n_exp = n_exp)
            psi = copy(base)
            cbe_lubich_bug!(psi, mpo;
                            opts = CBEBugOptions(dt = dt, n_steps = n, maxdim = 32,
                                                 normalize = false))
            push!(model_errs, norm(dense_state(psi) - want))
        end
        # Halving dt must NOT help: the residual is the Hamiltonian, not the step.
        mpo4, _ = power_law_zz_mpo(L; alpha = alpha, n_exp = 4)
        psi_fine = copy(base)
        cbe_lubich_bug!(psi_fine, mpo4;
                        opts = CBEBugOptions(dt = dt / 4, n_steps = 4n, maxdim = 32,
                                             normalize = false))
        fine = norm(dense_state(psi_fine) - want)
        @info "power-law state error" n_exp = [1, 2, 4, 6] model_errs dt_refined = fine
        @test issorted(model_errs; rev = true)         # more channels, closer to the truth
        @test fine > 0.5 * model_errs[3]               # a finer dt does not fix a model error
    end

    @testset "several long-range terms coexist" begin
        # `w = 2 + |nn| + |lr|`: two independent geometric tails with different decays.
        set_symmetry!(:U1)
        L = 4
        q = local_space()
        mpo = long_range_mpo(L, xxz_chain(L; delta = 0.0),
                             [LongRangeTerm(q.Sz, q.Sz, 1.0, 0.3),
                              LongRangeTerm(q.Sz, q.Sz, 0.5, 0.8)])
        @test all(==(6), mpo_virtual_dims(mpo)[2:(end - 1)])
        psi = _lr_evolved(L)
        v = dense_state(psi)
        H = dense_xy_chain(L) .+
            dense_zz_tail(L, [1.0 * 0.3^(r - 1) + 0.5 * 0.8^(r - 1) for r in 1:(L - 1)])
        @test abs(mpo_energy(psi, mpo) - dot(v, H * v)) < 1e-10
    end
end
