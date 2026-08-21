# `tfim_mpo`: the transverse-field Ising chain as an explicit MPO.
#
#     H = -J Σ Z_i Z_{i+1} - h Σ X_i
#
# Sits beside `test_pair_mpo.jl` for the same reason: this is a test of the OPERATOR, with no
# integrator anywhere in it. If the MPO is wrong, every sweep run on it is wrong in the same
# direction and the sweeps agree with each other -- which is the failure mode the file ordering
# in `runtests.jl` exists to prevent.
#
# THE REFERENCE IS A DENSE `2^L` TFIM BUILT FROM THE PAULI ALGEBRA, sharing no code with the
# tensor stack. It is the same construction `tests/BondUpdateBUG/test_z2_ising.jl` pins the GATES
# against, so gates and MPO are pinned to one independent reference rather than to each other.
#
# ⛔ THE BASIS IS σˣ: `X` is DIAGONAL and `Z` is the FLIP. Reading it the usual way describes a
# different Hamiltonian -- see the header of `src/RSVDCBEBondUpdate/models/tfim.jl`.

"Dense TFIM in the sigma^x product basis: site i <-> bit (i-1), bit CLEAR = |+>."
function _tfim_dense_ref(L::Int; J::Float64 = 1.0, h::Float64 = 1.0)
    D = 2^L
    H = zeros(ComplexF64, D, D)
    for s in 0:(D - 1)
        d = 0.0
        for i in 0:(L - 1)
            d += ((s >> i) & 1) == 0 ? 1.0 : -1.0
        end
        H[s + 1, s + 1] = -h * d
        for i in 0:(L - 2)
            t = s ⊻ (1 << i) ⊻ (1 << (i + 1))
            H[t + 1, s + 1] += -J
        end
    end
    return H
end

"A sigma^x product state as a dense vector, in `_tfim_dense_ref`'s convention."
function _x_product_dense(dirs::Vector{Symbol})
    v = zeros(ComplexF64, 2^length(dirs))
    s = 0
    for (i, d) in enumerate(dirs)
        d === :minus && (s |= 1 << (i - 1))
    end
    v[s + 1] = 1.0
    return v
end

@testset "tfim_mpo" begin
    L, J, h = 6, 1.0, 0.7
    HD = _tfim_dense_ref(L; J = J, h = h)

    # ⛔ ASYMMETRIC STATES ARE THE POINT. A state symmetric under reversal is a fixed point of a
    # MIRRORED site convention, so it passes whether or not the two orderings agree -- the same
    # landmine `dense_state` vs `exact_sparse` carries. The ragged cases are what test it.
    cases = ["polarised"  => fill(:plus, L),
             "kink"       => [i <= L ÷ 2 ? :plus : :minus for i in 1:L],
             "asymmetric" => [:plus, :minus, :minus, :plus, :minus, :plus],
             "one flip"   => [i == 2 ? :minus : :plus for i in 1:L]]

    @testset "the operator, under $sym" for sym in (:none, :Z2)
        set_symmetry!(sym)
        W = tfim_mpo(L; J = J, h = h)

        # Virtual dimension 3 on EVERY internal link, independent of L: `id`, `open`, `done`.
        # `mpo_virtual_dims` returns `L+1` entries, the outer two being the trimmed boundaries.
        # MEASURED at L=6: [1, 3, 3, 3, 3, 3, 1].
        @test length(W) == L
        @test mpo_virtual_dims(W) == [1; fill(3, L - 1); 1]

        for (name, dirs) in cases
            psi = ising_product_state(dirs)
            v   = _x_product_dense(dirs)
            @test real(mpo_energy(copy(psi), W)) ≈ real(dot(v, HD * v)) atol = 1e-10
        end

        # The analytic pin: <+...+|H|+...+> = -h*L for ANY J, because `Z_i Z_{i+1}` is purely
        # off-diagonal in this basis. A wrong ZZ charge or a dropped boundary field breaks it.
        @test real(mpo_energy(ising_polarised_state(L), W)) ≈ -h * L atol = 1e-10

        # ⛔ THE FIELD MUST SURVIVE AT BOTH ENDS. `mat[1,W]` is the on-site field here, unlike in
        # `long_range_mpo` where the same entry is unused and gets blanked at the boundaries.
        # Blanking it would weaken the field on sites 1 and L only -- a different Hamiltonian
        # whose error SHRINKS with L, so it would pass at large L and only fail here. `-h*L`
        # above already catches it; this states the size of the miss it would cause.
        @test !isapprox(real(mpo_energy(ising_polarised_state(L), W)), -h * (L - 2); atol = 1e-6)
    end

    # ⛔ EVERY ASSERTION ABOVE PINS THE FIELD TERM AND NONE OF THEM PINS THE COUPLING. `Z` is the
    # FLIP in this basis, so `⟨s|Z_i Z_{i+1}|s⟩ = 0` for EVERY σˣ product state `s` -- the ZZ
    # channel contributes exactly nothing to any of those energies, and an MPO built with the
    # coupling deleted, mis-signed or at the wrong `J` passes all of them. (Measured: the
    # `polarised`, `kink` and `asymmetric` energies are `-4.2`, `0` and `0`, i.e. `-h Σ⟨X_j⟩` and
    # nothing else.) The coupling needs a state that is not diagonal in this basis, which is what
    # evolving produces.
    #
    # THIS TESTSET THEREFORE USES AN INTEGRATOR, against the rule the rest of the file follows.
    # It is the same trade `test_z2_ising.jl` makes for the GATES ("L=8 real-time quench matches
    # exact diagonalisation"), and the direction of blame is still unambiguous: `tdvp2` at full
    # rank on `L = 6` is exact to the time-step error, and it is already pinned elsewhere against
    # models this file does not build. A failure here is the MPO's.
    @testset "the ZZ coupling, via a short quench under $sym" for sym in (:none, :Z2)
        set_symmetry!(sym)
        W    = tfim_mpo(L; J = J, h = h)
        dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
        v    = exp(-im * 0.2 * HD) * _x_product_dense(dirs)
        p    = abs2.(v)
        ref  = [sum(p[s + 1] * (((s >> (j - 1)) & 1) == 0 ? 1.0 : -1.0) for s in 0:(2^L - 1))
                for j in 1:L]

        psi = ising_kink_state(L)
        for _ in 1:10
            tdvp2_step!(psi, W, ComplexF64(-im * 0.02); maxdim = 16,
                        trunc_thresh = 1e-14, maxiter = 16)
        end
        # Full rank at L=6 (the cut can hold at most 8), so the only error left is O(dt²) Strang.
        @test maximum(abs.(x_profile(psi) - ref)) < 1e-6

        # And the coupling is the RIGHT SIZE, not merely present: `J = 0` must give a different
        # profile. Without this, an MPO whose ZZ channel carried the wrong `J` could still pass
        # the tolerance above if the two happened to be close over this short a window.
        psi0 = ising_kink_state(L)
        W0   = tfim_mpo(L; J = 0.0, h = h)
        for _ in 1:10
            tdvp2_step!(psi0, W0, ComplexF64(-im * 0.02); maxdim = 16,
                        trunc_thresh = 1e-14, maxiter = 16)
        end
        @test maximum(abs.(x_profile(psi0) - ref)) > 1e-3
    end

    @testset "the two symmetries describe one operator" begin
        # Same numbers, different block structure. A gap here is charge bookkeeping, not physics.
        e = Dict{Symbol, Vector{Float64}}()
        for sym in (:none, :Z2)
            set_symmetry!(sym)
            W = tfim_mpo(L; J = J, h = h)
            e[sym] = [real(mpo_energy(ising_product_state(d), W)) for (_, d) in cases]
        end
        @test maximum(abs.(e[:none] - e[:Z2])) < 1e-10
    end

    @testset "unsupported symmetries are refused, not approximated" begin
        # `Z_i Z_{i+1}` does not conserve S^z and the field picks an axis, so there is no U(1) or
        # SU(2) form. Refusing is the whole point: a silent projection would return a plausible
        # number for a different model.
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            @test_throws ArgumentError tfim_mpo(L; J = J, h = h)
        end
        set_symmetry!(:Z2)
        @test_throws ArgumentError tfim_mpo(1; J = J, h = h)
    end
end
