# The three structured sweeps: the shared expansion, the DMRG port, the TDVP port, and the new
# CBE + fixed-rank-BUG sweep.
#
# WHAT EACH SWEEP IS TESTED AGAINST, chosen so a failure says WHICH piece is wrong:
#
#   expand_bond!      that it changes NOTHING about the state. The expansion's whole licence to
#                     run before any evolution is that it is lossless; if it is not, every result
#                     downstream is contaminated and no integrator test would localise it.
#   dmrg_cbe1s!       the exact ground energy, and the VARIATIONAL BOUND. A value below the exact
#                     one is a bug and cannot be a lucky sweep.
#   tdvp_cbe1s_step!  its own channel-path twin `tdvp1_cbe_step!` on a nearest-neighbour chain
#                     (where both apply), the `dt` order, and unitarity.
#   cbe_bug_step!     `dt` order, exactness at full rank, and -- the attribution test --
#                     `kstep = false` reproducing `cbe_lubich_sweep`, which isolates the K-step's
#                     contribution with everything else held fixed.

using Test
using LinearAlgebra
import Random

@testset "sweeps" begin

    nn_chain(sym, L) = sym === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; delta = 1.0)
    function warm(L; steps = 3)
        psi = dimer_state(L)
        h = nn_chain(symmetry_mode(), L)
        for _ in 1:steps
            tdvp2_step!(psi, h, -0.1im; maxdim = 64, trunc_thresh = 1e-14)
            psi[psi.center] = to_concrete((1.0 / norm(psi)) * psi[psi.center])
        end
        return psi
    end
    renorm!(psi) = (psi[psi.center] = to_concrete((1.0 / norm(psi)) * psi[psi.center]); psi)

    # ── the shared expansion ──────────────────────────────────────────────────────────────
    @testset "expand_bond! is LOSSLESS" begin
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            L = 6
            mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(sym, L))
            for dir in (:right, :left)
                psi = warm(L)
                before = real(mpo_energy(psi, mpo))
                nb = norm(psi)
                ov = copy(psi)
                canonical!(psi, 1)
                rstack = right_env_stack(psi, mpo; downto = 3)
                _, info = expand_bond!(psi, mpo, 2, dir, boundary_channels(mpo),
                                       right_channels(rstack, 4))
                # The STATE is unchanged: same energy, same norm, unit overlap with itself.
                @test real(mpo_energy(psi, mpo)) ≈ before atol = 1e-10
                @test norm(psi) ≈ nb atol = 1e-12
                @test abs(overlap(ov, psi)) ≈ nb^2 atol = 1e-10
                # ... but the bond is WIDER, which is the only thing it was supposed to do.
                @test info.dim_out >= info.dim_in
                @test info.bond == 2 && info.dir === dir
            end
        end
    end

    @testset "expand_bond! refuses a wrong gauge rather than reading a stale frame" begin
        set_symmetry!(:U1)
        L = 6
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        psi = warm(L)
        canonical!(psi, 1)
        rstack = right_env_stack(psi, mpo; downto = 3)
        # Centre at 1, asked about bond 4: `_frame_from` would silently treat a non-isometry as
        # one, so this must throw instead.
        @test_throws ArgumentError expand_bond!(psi, mpo, 4, :right, boundary_channels(mpo),
                                                right_channels(rstack, 6))
        @test_throws ArgumentError expand_bond!(psi, mpo, 1, :sideways,
                                                boundary_channels(mpo),
                                                right_channels(rstack, 3))
    end

    @testset "lanczos_lowest finds the lowest eigenvalue and stays variational" begin
        set_symmetry!(:U1)
        L = 6
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        psi = warm(L)
        canonical!(psi, 3)
        lst = left_env_stack(psi, mpo; upto = 2)
        rst = right_env_stack(psi, mpo; downto = 4)
        H1 = one_site_h(mpo, 3, left_channels(lst, 3), right_channels(rst, 4))
        A, e, kd = lanczos_lowest(x -> apply_one_site(H1, x), psi[3];
                                  maxiter = 60, restol = 1e-14)
        @test norm(A) ≈ 1.0 atol = 1e-12
        # The Ritz value IS the Rayleigh quotient of what it returned -- if those disagree the
        # reduction and the vector reconstruction have drifted apart.
        @test real(tensor_inner(A, apply_one_site(H1, A))) ≈ e atol = 1e-9
        # And it is a lower value than the state it started from, or equal if already optimal.
        e0 = real(tensor_inner(psi[3], apply_one_site(H1, psi[3]))) / norm(psi[3])^2
        @test e <= e0 + 1e-10
        @test 1 <= kd <= 60
    end

    # ── DMRG ──────────────────────────────────────────────────────────────────────────────
    @testset "dmrg_cbe1s! reaches the exact ground energy" begin
        for sym in (:none, :U1, :SU2)
            set_symmetry!(sym)
            L = 6
            h = nn_chain(sym, L)
            mpo = RSVDCBEBondUpdate.mpo_from_terms(h)
            psi = dimer_state(L)
            r = dmrg_cbe1s!(psi, mpo; n_sweeps = 20, maxdim = 64, trunc_thresh = 1e-14)
            # The reference is the OPEN-chain XXX ground energy in this sector, by dense
            # diagonalisation -- Bethe is for the ring, so it is not the reference here.
            e_ed = eigen(Hermitian(Matrix(dense_heisenberg(L)))).values[1]
            @test r.energies[end] ≈ e_ed atol = 1e-8
            # VARIATIONAL: never below the exact value, beyond round-off.
            @test r.energies[end] >= e_ed - 1e-9
            # Converged means the bonds agree with each other, not just that the number stopped
            # moving -- the reference's `err_swp`.
            @test r.sweep_spread[end] < 1e-6
            # The expansion did the rank generation: from a dimer product (bond dims all small)
            # the sweep must have grown the bond, and it can only have done so via CBE.
            @test maximum(r.max_bond_dims) > 1
        end
    end

    @testset "dmrg_cbe1s!: without the expansion the rank cannot move" begin
        # The structural claim behind "CBE is mandatory, not an accelerator": a one-site update
        # varies one tensor with both bond legs FIXED, so with the expansion budget crushed to
        # nothing the rank is frozen and the energy stalls above the true ground state. This is
        # the same signature `modes/ground_state.jl` records for `grow_iters = 0`.
        set_symmetry!(:U1)
        L = 6
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        e_ed = eigen(Hermitian(Matrix(dense_heisenberg(L)))).values[1]

        psi_free = dimer_state(L)
        r_free = dmrg_cbe1s!(psi_free, mpo; n_sweeps = 20, maxdim = 64, trunc_thresh = 1e-14)

        # `maxdim` at the starting rank leaves the expansion nothing it may keep.
        psi_cap = dimer_state(L)
        start = maximum(bond_dims(psi_cap))
        r_cap = dmrg_cbe1s!(psi_cap, mpo; n_sweeps = 20, maxdim = start,
                            trunc_thresh = 1e-14)
        @test maximum(bond_dims(psi_cap)) <= start
        @test r_cap.energies[end] > r_free.energies[end] + 1e-6
        @test r_free.energies[end] ≈ e_ed atol = 1e-8
    end

    # ── TDVP ──────────────────────────────────────────────────────────────────────────────
    @testset "tdvp_cbe1s_step! == its channel-path twin on a NN chain" begin
        # Both paths apply on a nearest-neighbour chain and must agree; that agreement is what
        # licenses reading long-range results off the MPO path, where the channel form cannot
        # follow.
        set_symmetry!(:U1)
        L = 6
        h = nn_chain(:U1, L)
        mpo = RSVDCBEBondUpdate.mpo_from_terms(h)
        a = warm(L); b = copy(a); b.tensors .= copy.(a.tensors)
        for _ in 1:4
            tdvp_cbe1s_step!(a, mpo, -0.02im; maxdim = 64, trunc_thresh = 1e-14)
            tdvp1_cbe_step!(b, h, -0.02im; maxdim = 64, trunc_thresh = 1e-14)
        end
        @test real(mpo_energy(a, mpo)) ≈ real(mpo_energy(b, mpo)) atol = 1e-8
        @test abs(overlap(a, b)) ≈ norm(a) * norm(b) atol = 1e-6
    end

    @testset "tdvp_cbe1s_step! is unitary and second order" begin
        set_symmetry!(:U1)
        L = 6
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        psi = warm(L)
        n0 = norm(psi)
        for _ in 1:8
            tdvp_cbe1s_step!(psi, mpo, -0.02im; maxdim = 64, trunc_thresh = 1e-14)
        end
        # Real-time TDVP is unitary, so ANY norm loss is error.
        @test norm(psi) ≈ n0 atol = 1e-9

        # dt order, against the exact propagator on a state that is full rank at L=6 so the
        # residual is the integrator's and not the truncation's.
        ref = warm(L)
        v0 = dense_state(ref)
        T = 0.16
        errs = Float64[]
        for nst in (4, 8, 16)
            p = copy(ref); p.tensors .= copy.(ref.tensors)
            for _ in 1:nst
                tdvp_cbe1s_step!(p, mpo, -im * (T / nst); maxdim = 64, trunc_thresh = 1e-14)
            end
            want = dense_exact_propagate(dense_heisenberg(L), v0, T)
            push!(errs, norm(dense_state(p) - want))
        end
        # Halving `dt` must cut the error by ~4. Bounds are loose because the constant is not
        # the claim -- the EXPONENT is.
        p1 = log2(errs[1] / errs[2]); p2 = log2(errs[2] / errs[3])
        @info "tdvp_cbe1s dt-order: errs = $errs, p = ($p1, $p2)"
        @test 1.7 < p1 < 2.4
        @test 1.7 < p2 < 2.4
    end

    # ── the new sweep ─────────────────────────────────────────────────────────────────────
    @testset "cbe_bug_step! with kstep=false reproduces cbe_lubich_sweep" begin
        # THE ATTRIBUTION TEST. Everything except the K-step is shared, so this pins that the
        # only difference between the two schemes is the basis update -- which is what makes any
        # measured difference between them attributable to the K-step and to nothing else.
        set_symmetry!(:U1)
        L = 6
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        a = warm(L); b = copy(a); b.tensors .= copy.(a.tensors)
        for _ in 1:4
            cbe_bug_step!(a, mpo, -0.02im; kstep = false, maxdim = 64, trunc_thresh = 1e-14)
            cbe_lubich_sweep(b, mpo, -0.02im; maxdim = 64, trunc_thresh = 1e-14)
        end
        @test real(mpo_energy(a, mpo)) ≈ real(mpo_energy(b, mpo)) atol = 1e-9
        @test abs(overlap(a, b)) ≈ norm(a) * norm(b) atol = 1e-7
    end

    @testset "cbe_bug_step! is exact at full rank" begin
        # A fixed-rank BUG sounds like it should not be exact anywhere, and the reason it is here
        # is worth pinning: once the expanded bond reaches the whole local space, `orth(K(tau))`
        # spans it whatever `K(tau)` is, so the basis is complete, the seed is lossless and the
        # root solve is `exp(tau H)` on the full space. The `augment = false` approximation only
        # bites once the expansion is capped.
        set_symmetry!(:U1)
        L = 4
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        psi = warm(L)
        v0 = dense_state(psi)
        dt = 0.05
        cbe_bug_step!(psi, mpo, -im * dt; maxdim = 256, trunc_thresh = 0.0)
        want = dense_exact_propagate(dense_heisenberg(L), v0, dt)
        @test norm(dense_state(psi) - want) < 1e-9
    end

    @testset "cbe_bug_step! is second order" begin
        set_symmetry!(:U1)
        L = 6
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        ref = warm(L)
        v0 = dense_state(ref)
        T = 0.16
        errs = Float64[]
        for nst in (4, 8, 16)
            p = copy(ref); p.tensors .= copy.(ref.tensors)
            for _ in 1:nst
                cbe_bug_step!(p, mpo, -im * (T / nst); maxdim = 8, trunc_thresh = 1e-14)
            end
            push!(errs, norm(dense_state(p) -
                             dense_exact_propagate(dense_heisenberg(L), v0, T)))
        end
        p1 = log2(errs[1] / errs[2]); p2 = log2(errs[2] / errs[3])
        @info "cbe_bug dt-order: errs = $errs, p = ($p1, $p2)"
        # LOCAL order 2. Whether the GLOBAL order is 2 as well depends on the root: the sibling
        # `jan_bug` is locally O(dt^2) and globally order 1 because its Galerkin step sits on a
        # boundary bond. Here the root is the centre, so the frames span both halves. The
        # exponent is measured, not asserted from the structure.
        @test 1.5 < p1 < 2.5
        @test 1.5 < p2 < 2.5
    end

    @testset "cbe_bug_step! reports its expansion at EVERY bond" begin
        # The shrewd selection runs once per bond per half-sweep, not once per step -- the
        # structural claim from the reference sweeps. Assert the bookkeeping shows it.
        set_symmetry!(:U1)
        L = 8
        mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(:U1, L))
        psi = warm(L)
        info = cbe_bug_step!(psi, mpo, -0.02im; maxdim = 64, trunc_thresh = 1e-14)
        @test length(info.expanded) == L - 1
        @test all(>(0), info.expanded)
        @test info.root == L ÷ 2
        # Which bonds saturate depends on the warm-up state, so a per-bond claim is not a
        # property of the scheme -- what is asserted is that SOME bond was genuinely expanded.
        @test any(>(0), info.n_new)
        @test info.krylov_dims > 0
        @test info.peak_elements > 0
    end

    @testset "cbe_bug_step! in imaginary time descends to the ground state" begin
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            L = 6
            mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(sym, L))
            psi = dimer_state(L)
            e_prev = real(mpo_energy(psi, mpo))
            for _ in 1:60
                cbe_bug_step!(psi, mpo, -0.1; maxdim = 64, trunc_thresh = 1e-14)
                renorm!(psi)
            end
            e = real(mpo_energy(psi, mpo))
            e_ed = eigen(Hermitian(Matrix(dense_heisenberg(L)))).values[1]
            @test e < e_prev
            @test e ≈ e_ed atol = 1e-6
        end
    end
end
