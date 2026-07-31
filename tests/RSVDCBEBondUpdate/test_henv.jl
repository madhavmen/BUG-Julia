using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include("../common/dense_reference.jl")

# Stage 1 of CBE-BUG (docs/cbe_bug.md §4): the channel environments that give this
# module `H` GLOBALLY, which the gate-based Trotter path never needs.
#
# TWO INDEPENDENT REFERENCES, deliberately:
#
#   - `dense_heisenberg` / `dense_state` -- the exact matrix element. This is the real
#     gate: it shares no code with the environment recursion, so a leg/arrow/prime slip
#     cannot cancel between them.
#   - `energy(psi, gates)` -- the gate path, itself pinned element by element to the
#     analytic Heisenberg block in `test_gates.jl`. Agreement here says the term list
#     `xxz_chain` builds is the same operator `bond_gates` builds, including Telum's
#     Sp normalisation and the `:none` sign.
#
# A wrong prime level does NOT throw -- `*` would silently trace the pair (contract
# §7a) -- so these numbers are the only thing standing between a plausible answer and
# a correct one.

"Unnormalised ⟨ψ|H|ψ⟩ from the dense reference."
function dense_expectation(psi::SymMPS, L::Int; J = 1.0, delta = 1.0)
    v = dense_state(psi)
    H = dense_heisenberg(L; J = J, delta = delta)
    return v' * (H * v)
end

"A state with real entanglement: a few BUG steps off a product state."
function entangled_state(L::Int; n_steps = 3, dt = 0.05)
    psi = domain_wall_state(L)
    gates = bond_gates(psi)
    bond_update_bug!(psi, gates;
                     opts = BondUpdateOptions(dt = dt, n_steps = n_steps, maxdim = 16))
    return psi
end

@testset "henv (channel environments)" begin

    @testset "xxz_chain term list" begin
        set_symmetry!(:U1)
        h = xxz_chain(6; J = 1.0, delta = 1.0)
        @test length(h) == 6
        ts = hamiltonian_terms(h)
        @test length(ts) == 3                       # Sp-pair, Sm-pair, SzSz
        # Under U(1) the XY halves are rank-3 adjoint pairs: the op-leg carries the
        # ±2 charge that makes the hop allowed, and it is contracted between them.
        @test length(ts[1].left.inds) == 3
        @test length(ts[1].right.inds) == 3
        @test ts[1].left.inds[3].dir == '-'         # dangles when the channel opens
        @test ts[1].right.inds[3].dir == '+'        # closes against it
        @test length(ts[3].left.inds) == 2          # Sz has no op-leg
        @test length(xxz_chain(6; delta = 0.0).terms) == 2   # XX drops the SzSz term
        @test_throws ArgumentError xxz_chain(1)
    end

    @testset "energy vs dense reference, U(1)" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            h = xxz_chain(L; delta = delta)
            for (name, psi) in ("domain wall" => domain_wall_state(L),
                                "neel"        => neel_state(L))
                got = env_energy(psi, h)
                want = dense_expectation(psi, L; delta = delta)
                @test abs(got - want) < 1e-12
                @test abs(imag(got)) < 1e-12        # H is Hermitian; ⟨H⟩ is real
            end
        end
    end

    @testset "energy vs dense reference, entangled state" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            psi = entangled_state(L)
            @test maximum(bond_dims(psi)) > 1       # the test is vacuous otherwise
            h = xxz_chain(L; delta = delta)
            got = env_energy(psi, h)
            want = dense_expectation(psi, L; delta = delta)
            @test abs(got - want) < 1e-11
        end
    end

    @testset "agrees with the gate path" begin
        set_symmetry!(:U1)
        for L in (4, 6), delta in (1.0, 0.0)
            psi = entangled_state(L)
            n2 = norm(psi)^2
            h = xxz_chain(L; delta = delta)
            got = env_energy(psi, h)
            # `energy` divides by the norm bond by bond and canonicalises as it goes,
            # so it gets a copy and its result is the NORMALISED expectation value.
            want = energy(copy(psi), bond_gates(psi; delta = delta))
            @test abs(got / n2 - want) < 1e-11
        end
    end

    @testset "gauge invariance" begin
        # The recursion contracts bra against ket explicitly and never assumes an
        # isometry, so moving the orthogonality centre must not move the answer.
        set_symmetry!(:U1)
        L = 6
        h = xxz_chain(L)
        ref = env_energy(entangled_state(L), h)
        for c in (1, 3, L)
            psi = entangled_state(L)
            canonical!(psi, c)
            @test abs(env_energy(psi, h) - ref) < 1e-11
        end
    end

    @testset "identity channel is the identity" begin
        # `id[i]` is stored so the recursion stays valid with the centre anywhere; for a
        # left-canonical chain it must come out as the bond identity. Checked by norm
        # rather than by subtraction: building an identity on an oplus-ed leg is exactly
        # the alignment failure documented in `left_isometry_defect`.
        set_symmetry!(:U1)
        L = 6
        psi = entangled_state(L)
        canonical!(psi, L)                         # left-canonical for sites 1 … L-1
        st = left_env_stack(psi, xxz_chain(L))
        for i in 2:L
            d = leg_dim(psi[i], 1)
            @test abs(norm(st.id[i])^2 - d) < 1e-10
        end
    end

    @testset "no symmetry" begin
        # The XY term factorises differently without the op-legs, with a -1 that comes
        # from Telum's Sp normalisation (gates.jl). Same two references.
        set_symmetry!(:none)
        try
            for L in (4, 6), delta in (1.0, 0.0)
                h = xxz_chain(L; delta = delta)
                @test all(length(t.left.inds) == 2 for t in hamiltonian_terms(h))
                psi = neel_state(L)
                @test abs(env_energy(psi, h) - dense_expectation(psi, L; delta = delta)) < 1e-12
                psi2 = entangled_state(L)
                @test abs(env_energy(psi2, h) -
                          dense_expectation(psi2, L; delta = delta)) < 1e-11
            end
        finally
            set_symmetry!(:U1)                     # global setting; never leak it
        end
    end
end
