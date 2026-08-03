using Test, LinearAlgebra, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime

# Z2 SYMMETRY FOR THE TRANSVERSE-FIELD ISING MODEL.
#
#     H = -J Σ Z_i Z_{i+1} - h Σ X_i
#
# The Z2 local space is built BELOW Telum's IROP layer (see the header of `z2_ising.jl`:
# `getweight_irop` derives an operator's charge from a Lie-algebra commutator, and the Z2 flip
# admits no such charge because +1 and -1 are the same mod 2). That means this file rests on
# Telum-PRIVATE API, so these tests are the tripwire: if a Telum upgrade changes the private
# contract, this must fail loudly rather than quietly producing a wrong Hamiltonian.
#
# The tests are therefore on PHYSICS that an incorrect charge assignment could not fake:
# the Pauli algebra, an analytic energy, and the exact spectrum at L = 8.

# ── an exact reference, built independently of the tensor code ───────────────────────
"Dense TFIM in the sigma^x product basis: |+> = 0 bit, |-> = 1 bit."
function tfim_dense(L::Int; J::Float64 = 1.0, h::Float64 = 1.0)
    D = 2^L
    H = zeros(Float64, D, D)
    for s in 0:(D - 1)
        # -h Σ X_i is DIAGONAL in this basis: bit 0 -> +1, bit 1 -> -1.
        d = 0.0
        for i in 0:(L - 1)
            d += ((s >> i) & 1) == 0 ? 1.0 : -1.0
        end
        H[s + 1, s + 1] = -h * d
        # -J Σ Z_i Z_{i+1} FLIPS both bits: Z is off-diagonal in the x basis.
        for i in 0:(L - 2)
            t = s ⊻ (1 << i) ⊻ (1 << (i + 1))
            H[t + 1, s + 1] += -J
        end
    end
    return H
end

"<X_j> for every site, from a dense state vector."
function x_profile_dense(v::Vector{Float64}, L::Int)
    p = abs2.(v)
    return [sum(p[s + 1] * (((s >> (j - 1)) & 1) == 0 ? 1.0 : -1.0) for s in 0:(2^L - 1))
            for j in 1:L]
end

@testset "Z2 transverse-field Ising" begin

    @testset "the local space is a genuine Z2 space" begin
        set_symmetry!(:Z2)
        q = ising_local_space()
        @test Set(propertynames(q)) == Set((:I, :X, :Z))
        # Two charge sectors, dim 1 each -- the Z2 parity of sigma^x.
        @test length(q.X.spaces[1]) == 2
        # THE FLIP CARRIES AN OP LEG, and that is the whole reason Z2 is representable here:
        # both matrix elements share ONE op-leg charge because 1 = 0+1 and 0 = 1+1 mod 2.
        # Under U(1) this would need charges +1 and -1 and would not be one operator at all.
        @test length(q.Z.inds) == 3
        @test length(q.Z.spaces[3]) == 1
        # `local_space` is the HEISENBERG space and must refuse :Z2 rather than return something
        # spin-like -- the TFIM has no Sp/Sz/Sm.
        @test_throws ArgumentError local_space(:Z2)
    end

    @testset "states, and the sector they live in" begin
        for sym in (:Z2, :none)
            set_symmetry!(sym)
            for L in (4, 8)
                p = ising_polarised_state(L)
                @test norm(p) ≈ 1.0 atol = 1e-12
                @test bond_dims(p) == ones(Int, L - 1)      # a genuine product state
                @test x_profile(p) ≈ ones(L) atol = 1e-12   # every spin along +x
                k = ising_kink_state(L)
                @test x_profile(k) ≈ [i <= L ÷ 2 ? 1.0 : -1.0 for i in 1:L] atol = 1e-12
            end
        end
    end

    @testset "the analytic energy of the polarised state" begin
        # |+...+> has <Z_i Z_j> = 0 (Z is off-diagonal in this basis) and <X_j> = 1, so
        # E = -h*L exactly, for any J. A wrong ZZ charge assignment would break this.
        for sym in (:Z2, :none), L in (4, 8), h in (0.5, 1.0, 2.0)
            set_symmetry!(sym)
            psi = ising_polarised_state(L)
            g = ising_bond_gates(psi; J = 1.0, h = h)
            @test energy(psi, g) ≈ -h * L atol = 1e-10
        end
    end

    @testset "L=8: the gates reproduce the exact spectrum" begin
        # THE LOAD-BEARING TEST. The bond gates distribute the on-site field over bonds
        # (interior sites h/2 from each of two bonds, boundary sites the full h from their
        # single bond). Getting that wrong gives a plausible chain with weakened boundary
        # fields, whose error SHRINKS with L -- so it is checked against a dense H built
        # independently, not against another tensor calculation.
        L = 8
        for (J, h) in ((1.0, 1.0), (1.0, 0.5), (0.7, 1.3))
            Hd = tfim_dense(L; J = J, h = h)
            E0 = minimum(eigvals(Symmetric(Hd)))
            for sym in (:Z2, :none)
                set_symmetry!(sym)
                psi = ising_polarised_state(L)
                g = ising_bond_gates(psi; J = J, h = h)
                # Cool to the ground state by imaginary time. CBE has to GROW the rank from
                # chi = 1 for this to converge at all.
                opts = CBEBugOptions(; dt = 0.05, n_steps = 400, maxdim = 32,
                                     s_iters = -1, maxiter = 8, normalize = true)
                # IMAGINARY time: tau = -dt. `RealTime.evolve!` takes no tau (it forms
                # -im*dt itself), so the shared engine is called directly.
                cbe_gate_evolve!(psi, ising_bond_gates(psi; J = J, h = h),
                                 ComplexF64(-0.05); opts = opts)
                E = energy(psi, ising_bond_gates(psi; J = J, h = h))
                # TOLERANCE SET FROM A MEASURED CONVERGENCE ORDER, NOT PICKED TO PASS.
                # Imaginary-time cooling converges to the ground state of the TROTTERISED
                # propagator, which differs from H at O(dt^2); at dt = 0.05 that is ~2e-6
                # (measured: -9.837949414887 vs -9.837951447459).
                #
                # The reason this is splitting error and not a charge error is the PATTERN:
                # :Z2 and :none agree with EACH OTHER to 1e-15 while both sit 2e-6 from
                # exact. A wrong Z2 charge assignment would separate the two symmetries.
                # Independently confirmed by halving dt in a real-time quench, where the
                # error ratio is 4.01 then 4.00 -- clean second order.
                @test E ≈ E0 atol = 1e-5
                @test maximum(bond_dims(psi)) > 1        # the rank actually grew
            end
        end
    end

    @testset "Z2 and :none agree: same physics, different storage" begin
        # The claim the symmetry is FOR. Same basis, same gates, same state -- only the block
        # structure differs, so every observable must agree to round-off.
        L = 8
        profiles = Dict{Symbol, Vector{Float64}}()
        for sym in (:Z2, :none)
            set_symmetry!(sym)
            psi = ising_kink_state(L)
            g = ising_bond_gates(psi; J = 1.0, h = 1.0)
            RealTime.evolve!(psi, g; opts = CBEBugOptions(;
                dt = 0.05, n_steps = 40, maxdim = 32, s_iters = -1, maxiter = 8))
            profiles[sym] = x_profile(psi)
        end
        @test maximum(abs.(profiles[:Z2] - profiles[:none])) < 1e-10
    end

    @testset "L=8 real-time quench matches exact diagonalisation" begin
        # The strongest check: evolve the kink and compare <X_j>(t) against exp(-iHt) on the
        # dense vector. Accuracy here can only come from the Hamiltonian, the charges and the
        # integrator all being right together.
        L, J, h, t = 8, 1.0, 1.0, 1.0
        Hd = tfim_dense(L; J = J, h = h)
        v0 = zeros(Float64, 2^L)
        v0[(sum(1 << (i - 1) for i in (L ÷ 2 + 1):L)) + 1] = 1.0   # the kink, as a bit string
        F = eigen(Symmetric(Hd))
        vt = F.vectors * (cis.(-t .* F.values) .* (F.vectors' * v0))
        exact = [sum(abs2(vt[s + 1]) * (((s >> (j - 1)) & 1) == 0 ? 1.0 : -1.0)
                     for s in 0:(2^L - 1)) for j in 1:L]

        set_symmetry!(:Z2)
        psi = ising_kink_state(L)
        g = ising_bond_gates(psi; J = J, h = h)
        RealTime.evolve!(psi, g; opts = CBEBugOptions(;
            dt = 0.01, n_steps = 100, maxdim = 64, s_iters = -1, maxiter = 8))
        # O(dt^2) Strang error, MEASURED to be exactly that (t = 0.5, same state and gates):
        #
        #     dt = 0.0500   4.7053e-04
        #     dt = 0.0250   1.1747e-04   ratio 4.01
        #     dt = 0.0125   2.9374e-05   ratio 4.00
        #
        # so the threshold below is the splitting error at this dt with margin, not a number
        # chosen to make the test green. Tighten dt if a tighter bound is ever wanted.
        @test maximum(abs.(x_profile(psi) - exact)) < 5e-4
    end

    set_symmetry!(:U1)
end
