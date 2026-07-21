using Test, LinearAlgebra, JSON, LurCGT, Telum
using BUGJulia.BondUpdateBUG

# Python <-> Julia parity on the observable.
#
# The coarse gate: same model, same initial condition, same settings, same
# <Sz_j> profile. The reference JSON is produced by
# tests/crosscheck/export_python_reference.py running the VERIFIED Alice
# discarded-projector kernel, and carries the Alice commit it came from.
#
# JUDGE BY THE PROFILE, NOT A FIDELITY. A vec()-based fidelity helper has been
# wrong in this project before -- it reported disagreement that was really a
# sector-ordering artifact. <Sz_j> is basis-independent and site-resolved, so a
# genuine algorithmic difference shows up as a specific site moving, not as a
# single opaque number.
#
# THE TOLERANCE IS 1e-6, AND THE RESIDUAL IS FULLY ACCOUNTED FOR.
# Measured gap: 6.33e-8. It is not a porting error and not randomness:
#
#   - Not the fill's RNG. Four different seeds give a max pairwise spread of
#     EXACTLY 0.0 -- the S-step and the post-S SVD make the answer independent of
#     the seed's orientation, as Alice's own `_random_orthonormal_columns`
#     docstring claims. Confirmed, not assumed.
#   - It is the Sulz bound. Re-running this same comparison with the pre-Sulz
#     augmentation (`aug = 2r + seeds`, which is what Alice does) gives
#     4.27e-11 -- essentially exact agreement between two independent
#     implementations, which is what makes the port itself trustworthy. Enforcing
#     `aug <= 2r` on the Julia side is a deliberate divergence from Alice, so the
#     residual 6.33e-8 is the measured price of that decision and nothing else.
#
# Consequence for the trace-parity work: BIT-FOR-BIT agreement with Alice is no
# longer achievable by construction, because the augmentation step is
# intentionally different. A step-by-step comparator has to treat the augmented
# rank as an expected divergence rather than a mismatch.

const REF_PATH = joinpath(@__DIR__, "reference_l6_heisenberg.json")

@testset "Python parity: L=6 Heisenberg" begin
    @test isfile(REF_PATH)
    ref = JSON.parsefile(REF_PATH)

    L        = Int(ref["L"])
    dt       = Float64(ref["dt"])
    nsteps   = Int(ref["n_steps"])
    delta    = Float64(ref["delta"])
    maxdim   = Int(ref["maxdim"])
    thresh   = Float64(ref["trunc_thresh"])
    normal   = Bool(ref["normalize"])
    sz_want  = Float64.(ref["sz_final"])
    sz_start = Float64.(ref["sz_initial"])

    @testset "the reference is the one this comparison was pinned to" begin
        # A regenerated reference from a different Alice would otherwise be
        # compared silently, and any gap misread as an algorithmic difference.
        @test ref["alice_sha"] == "1f2de1b7b93c6c63876862407698af6abecc9fa3"
        @test ref["variant"] == "discarded"
        @test ref["order"] == "strang"
    end

    @testset "both sides start from the same state" begin
        psi0 = domain_wall_state(L)
        @test maximum(abs.([sz_expectation(psi0, j) for j in 1:L] .- sz_start)) < 1e-13
    end

    @testset "the profile matches after the run" begin
        psi = domain_wall_state(L)
        g = bond_gates(psi; J = 1.0, delta = delta)
        bond_update_bug!(psi, g; opts = BondUpdateOptions(
            dt = dt, n_steps = nsteps, order = :strang, maxdim = maxdim,
            trunc_thresh = thresh, normalize = normal))
        got = [sz_expectation(psi, j) for j in 1:L]

        @test maximum(abs.(got .- sz_want)) < 1e-6
        # and the qualitative structure the two must share
        @test isapprox(sum(got), 0.0; atol = 1e-11)
        for j in 1:(L ÷ 2)
            @test isapprox(got[j], -got[L + 1 - j]; atol = 1e-9)
        end
    end

    @testset "the wall has actually moved, so this is not a trivial match" begin
        # If both sides simply failed to evolve, the profiles would agree
        # perfectly and mean nothing.
        @test maximum(abs.(sz_want .- sz_start)) > 1e-5
    end
end
