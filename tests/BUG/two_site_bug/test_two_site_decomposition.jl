# test_two_site_decomposition.jl
#
# The Hamiltonian must be decomposed PROPERLY into the odd/even Trotter pieces:
#   * the two parity MPOs sum to the full Hamiltonian,
#   * each parity's terms live on disjoint site pairs (so they commute), while
#     the two parities do NOT commute (the genuine Trotter-error source),
#   * the per-bond gate generators sum (lifted with identities) to H,
#   * the local effective Hamiltonian fed to the KLS step at a bond acts as that
#     bond's bare 2-site term (the canonical environment is the identity).

using Test, ITensors, ITensorMPS, LinearAlgebra, Random
include(joinpath(@__DIR__, "two_site_test_setup.jl"))

# Lift a 2-site operator on sites (b,b+1) to the full Hilbert-space matrix.
function _lift_gate(gate, sites, b)
    N = length(sites)
    full = gate
    for k in 1:N
        (k == b || k == b + 1) && continue
        full = full * delta(sites[k], prime(sites[k]))
    end
    ITensors.disable_warn_order()
    M = reshape(Array(full, prime.(sites)..., sites...), 2^N, 2^N)
    ITensors.reset_warn_order()
    return ComplexF64.(M)
end

@testset "two-site Hamiltonian decomposition (N=6 XX)" begin
    N = 6
    sites = siteinds("S=1/2", N)
    J = 1.0
    W_odd, W_even, W_full = two_site_xx_parity_mpos(sites; J = J)
    gates = two_site_xx_bond_gates(sites; J = J)

    Hodd  = two_site_dense(W_odd)
    Heven = two_site_dense(W_even)
    Hfull = two_site_dense(W_full)

    @testset "parity MPOs sum to the full Hamiltonian" begin
        @test norm(Hodd + Heven - Hfull) <= 1e-12 * max(norm(Hfull), 1.0)
        @test norm(Hfull - Hfull') <= 1e-12 * norm(Hfull)   # Hermitian
    end

    @testset "within-parity terms commute, the two parities do not" begin
        # Odd pairs (1,2),(3,4),(5,6) are disjoint ÔçÆ their lifted terms commute.
        h1 = _lift_gate(gates[1], sites, 1)
        h3 = _lift_gate(gates[3], sites, 3)
        @test norm(h1 * h3 - h3 * h1) <= 1e-12 * max(norm(h1) * norm(h3), 1.0)
        # H_odd and H_even share sites (bond 2 = (2,3) overlaps bond 1 and 3) ÔçÆ
        # they do NOT commute. This non-commutativity is the Trotter-error source.
        @test norm(Hodd * Heven - Heven * Hodd) > 1e-6
        @test length(gates) == N - 1
    end

    @testset "per-bond gate generators sum to H" begin
        Hsum = sum(_lift_gate(gates[b], sites, b) for b in 1:(N - 1))
        @test norm(Hsum - Hfull) <= 1e-12 * max(norm(Hfull), 1.0)
    end

    @testset "local effective Hamiltonian acts as the bare bond gate (identity env)" begin
        psi = two_site_rank3_state(sites; seed = 11)
        for b in (1, 3)
            orthogonalize!(psi, b)
            bd = BUG._two_site_bond_snapshot(psi, b)
            HW = BUG._two_site_local_effective_hamiltonian(gates[b], bd)
            theta = bd.U0_tens * bd.S0_tens * bd.V0_tens
            # Dressed effective operator must act on the block exactly like the
            # bare gate (gate on the two sites, identity on the link legs).
            via_HW   = noprime(HW * theta)
            via_gate = noprime(gates[b] * theta)
            @test norm(via_HW - via_gate) <= 1e-12 * max(norm(via_gate), 1.0)
        end
    end
end
