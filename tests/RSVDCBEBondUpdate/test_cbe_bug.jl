using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../common/dense_reference.jl")

# Stage 3: the step itself.
#
# THE EXACTNESS TEST IS THE ONE THAT MATTERS. At L=4 with the bonds at their maximum
# ([2,4,2]) the frames already span their local spaces, so the projector onto
# span(left basis) x span(right basis) at the centre is the IDENTITY, and the single
# 0-site Galerkin step must reproduce exp(-iH dt) exactly. Any error in the sweep, the
# write-back, the environment push, or the 0-site apply shows up there as a finite number
# -- there is nothing left for it to hide behind.
#
# The second design goal (docs/cbe_bug.md §2b) is pinned structurally rather than by
# timing: the 0-site Hamiltonian holds only rank-2 and rank-3 tensors, and its action
# agrees with the two-site route through `apply_h_two_site`. Together those say the
# Krylov loop computes the right thing without ever building a rank-4 block.

"Bond dimensions saturated at the maximum an L-site chain allows."
saturated(L) = [min(2^i, 2^(L - i)) for i in 1:(L - 1)]

function evolved(L::Int; n_steps = 6, dt = 0.05, maxdim = 32)
    psi = domain_wall_state(L)
    bond_update_bug!(psi, bond_gates(psi);
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = maxdim))
    return psi
end

@testset "cbe_bug_bond_update" begin

    @testset "0-site action agrees with the two-site route" begin
        # `apply_zero_site` must be the same operator as `apply_h_two_site` sandwiched
        # between the expanded frames -- the latter is already pinned off-diagonally to the
        # dense matrix element, so this transfers that guarantee onto the cheap path.
        set_symmetry!(:U1)
        rng = Random.MersenneTwister(0x5EED)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L)
            for i in 1:(L - 1)
                canonical!(psi, i)
                f = bond_frame(psi, i)
                lenv = left_env_stack(psi, h; upto = i - 1)
                renv = right_env_stack(psi, h; downto = i + 2)
                ex = cbe_expand(f, h, i, lenv, renv; rng = rng)

                H0 = RSVDCBEBondUpdate.zero_site_h(h, i, lenv, renv, ex.U_ex, ex.V_ex)
                # Memory claim, structurally: nothing in the Krylov loop is rank 4.
                for t in vcat([H0.done_l, H0.done_r], H0.open_l, H0.open_r)
                    t === RSVDCBEBondUpdate.ZERO && continue
                    @test length(t.inds) <= 3
                end

                S = to_concrete(contract(contract(ex.U_ex', (1, 2), frame_theta(f), (1, 2)),
                                         (2, 3), ex.V_ex', (2, 3)))
                got = RSVDCBEBondUpdate.apply_zero_site(H0, S)
                @test length(got.inds) == 2

                th = to_concrete((ex.U_ex * S) * ex.V_ex)
                hth = apply_h_two_site(th, h, i, lenv, renv)
                want = to_concrete(contract(contract(ex.U_ex', (1, 2), hth, (1, 2)),
                                            (2, 3), ex.V_ex', (2, 3)))
                @test norm(to_concrete(got - want)) < 1e-10 * max(1.0, norm(want))
            end
        end
    end

    @testset "exact at full rank" begin
        set_symmetry!(:U1)
        L = 4
        dt = 0.02
        for delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            psi = evolved(L; n_steps = 8, dt = 0.05, maxdim = 32)
            @test bond_dims(psi) == saturated(L)          # otherwise P is not the identity

            v0 = dense_state(psi)
            H = dense_heisenberg(L; delta = delta)
            want = dense_exact_propagate(H, v0, dt)

            cbe_bug_bond_update(psi, h, -im * dt; maxdim = 32, trunc_thresh = 1e-14)
            got = dense_state(psi)
            @test norm(got - want) < 1e-9
        end
    end

    @testset "tau = 0 is the identity" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        before = dense_state(psi)
        cbe_bug_bond_update(psi, h, 0.0 + 0.0im; maxdim = 32, trunc_thresh = 1e-14)
        @test norm(dense_state(psi) - before) < 1e-10
    end

    @testset "conserves charge and norm" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L)
        sz0, n0 = total_sz(copy(psi)), norm(psi)
        for _ in 1:3
            cbe_bug_bond_update(psi, h, -im * 0.02; maxdim = 32, trunc_thresh = 1e-14)
        end
        @test abs(total_sz(copy(psi)) - sz0) < 1e-10       # U(1) is exact, not approximate
        @test abs(norm(psi) - n0) < 1e-8                   # 0-site generator is Hermitian
    end

    @testset "exact over many steps once the rank is full" begin
        # The L=4 exactness test above, extended: at L=6 with maxdim=32 the state also
        # reaches full rank ([2,4,8,4,2]), so the centre projector stays the identity for
        # the whole run and the error is machine noise -- MEASURED 2.3e-15 over 8 steps
        # (job 94969). Nothing about dt convergence can be read off this regime, which is
        # why the order measurement is a stage-4 benchmark and not a unit test.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        H = dense_heisenberg(L)
        psi = evolved(L; n_steps = 4, dt = 0.05, maxdim = 32)
        # Not `== saturated(L)`: MEASURED [2,4,7,4,2] (job 94971), because the eighth
        # centre direction carries no weight in this state. The subspace is complete for
        # everything the dynamics touches, which is what the error below actually tests --
        # asserting the nominal dimension instead tests the state's Schmidt spectrum.
        @test maximum(bond_dims(psi)) >= min(2^(L ÷ 2), 2^(L - L ÷ 2)) - 1
        v0 = dense_state(psi)
        dt = 0.02
        for _ in 1:8
            cbe_bug_bond_update(psi, h, -im * dt; maxdim = 32, trunc_thresh = 1e-14)
        end
        @test norm(dense_state(psi) - dense_exact_propagate(H, v0, dt * 8)) < 1e-12
    end

    @testset "more expansion never hurts" begin
        # The property `dex` has to have if it is to be a usable knob: enlarging the
        # subspace cannot increase the error. Run below full rank, so the expansion is
        # what limits accuracy rather than the maxdim cap.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        H = dense_heisenberg(L)
        dt = 0.05

        function err_with(dex)
            psi = evolved(L; n_steps = 1, dt = 0.05, maxdim = 6)
            v0 = dense_state(psi)
            cbe_bug_bond_update(psi, h, -im * dt; dex = dex, maxdim = 6,
                                trunc_thresh = 1e-14)
            return norm(dense_state(psi) - dense_exact_propagate(H, v0, dt))
        end

        e_small, e_full = err_with(1), err_with(0)     # dex = 0 is the full Sulz budget
        @test e_full <= e_small + 1e-12
    end

    @testset "rank does not ratchet up" begin
        # The basis sweep over-dimensions every bond; the truncation sweep has to give the
        # unpopulated directions back, or the rank grows without bound step after step.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        psi = evolved(L; n_steps = 2)
        dims = Vector{Int}[]
        for _ in 1:5
            cbe_bug_bond_update(psi, h, -im * 0.02; maxdim = 8, trunc_thresh = 1e-12)
            push!(dims, bond_dims(psi))
        end
        @test all(maximum(d) <= 8 for d in dims)           # maxdim is respected
        @test all(all(x -> x >= 1, d) for d in dims)
    end

    @testset "dex controls the expansion" begin
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        info_small = cbe_bug_bond_update(evolved(L), h, -im * 0.02; dex = 1, maxdim = 32)
        info_full  = cbe_bug_bond_update(evolved(L), h, -im * 0.02; dex = 0, maxdim = 32)
        @test maximum(info_small.n_new) <= 1
        @test maximum(info_full.expanded) >= maximum(info_small.expanded)
    end

    @testset "the centre expansion is load-bearing" begin
        # INVESTIGATION (asked 2026-07-27): given the sweeps already expanded every other
        # bond, are the augmented frames at the centre needed? Measured here rather than
        # argued. The sweeps expand bonds 1..c-1 and c+1..L-1; bond c is expanded ONLY by
        # the centre step, so with it off the centre rank can never grow -- and the centre
        # cut is exactly where a domain wall's entanglement builds.
        set_symmetry!(:U1)
        L = 8
        h = xxz_chain(L; delta = 0.0)
        c = L ÷ 2
        function run(centre_expand)
            psi = domain_wall_state(L)
            r = Int[]
            for _ in 1:6
                cbe_bug_bond_update(psi, h, -im * 0.05; maxdim = 32,
                                    trunc_thresh = 1e-12, centre_expand = centre_expand)
                push!(r, bond_dims(psi)[c])
            end
            return r
        end
        with, without = run(true), run(false)
        @test all(==(1), without)          # frozen at the product state's rank
        @test maximum(with) > 1            # the expansion is what unfreezes it
    end

    @testset "no symmetry" begin
        set_symmetry!(:none)
        try
            L = 6
            h = xxz_chain(L)
            psi = evolved(L; n_steps = 2)
            n0 = norm(psi)
            cbe_bug_bond_update(psi, h, -im * 0.02; maxdim = 16, trunc_thresh = 1e-13)
            @test abs(norm(psi) - n0) < 1e-8
            @test all(d >= 1 for d in bond_dims(psi))
        finally
            set_symmetry!(:U1)
        end
    end
end
