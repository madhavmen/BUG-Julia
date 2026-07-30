using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

# ONE-SITE TDVP WITH CBE.
#
# TWO ARMS ARE UNDER TEST, and they are the only two: `1site_tdvp_cbe` (exact = true, the
# complete fused basis) and `1site_tdvp_cbe_rsvd` (exact = false, the sketch). Bare 1-site TDVP
# is not an arm and is not benchmarked anywhere -- it is only the reason CBE has to be here at
# all, since splitting a `(chi_l, d, chi_r)` site tensor cannot produce a bond wider than
# `chi_r` and so cannot grow entanglement.
#
# The first test is the one that would invalidate every other number in this file, so it comes
# first: DOES THE BOND GROW, on BOTH arms? `maxbd > 1` from a product start is not a quality
# measure, it is the check that the expansion is expanding at all.
#
# The rest pins the composition (`tau/2` + `tau/2`, second order), agreement with an INDEPENDENT
# scheme (2-site TDVP and the analytic free-fermion answer), and that the exact and sketched
# expansions land in the same place.

"Analytic <Sz_j>(t) for the XX chain from a product start, via Jordan-Wigner."
function _xx_analytic(L::Int, t::Float64, n0::Vector{Float64}; J::Float64 = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    U = exp(-im * h * t)
    return [sum(abs2(U[j, k]) * n0[k] for k in 1:L) - 0.5 for j in 1:L]
end

function _run_tdvp1(L, h, tau, nst; normalise = false, kwargs...)
    psi = domain_wall_state(L)
    mb, mx = 0, 0
    for _ in 1:nst
        info = tdvp1_cbe_step!(psi, h, tau; maxdim = 32, trunc_thresh = 1e-12, kwargs...)
        mb = max(mb, info.max_bond); mx = max(mx, info.max_expanded)
        if normalise
            n = norm(psi)
            n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
        end
    end
    return psi, mb, mx
end

@testset "tdvp1_cbe" begin

    @testset "the expansion expands -- both arms" begin
        set_symmetry!(:U1)
        L = 8
        h = xxz_chain(L; delta = 0.0)
        # Started from a domain wall, i.e. bond dimension 1 everywhere.
        @test bond_dims(domain_wall_state(L)) == ones(Int, L - 1)
        # BOTH arms, not just the default: `exact` swaps the whole selection rule (complete
        # fused basis vs Gaussian sketch), so "the sketched path grows the bond" says nothing
        # about the exact path. Testing one and shipping two is how a dead arm survives.
        for exact in (false, true)
            psi, mb, mx = _run_tdvp1(L, h, ComplexF64(-im * 0.02), 20; exact = exact)
            @test mb > 1                 # without a working expansion this stays 1
            @test mx >= mb               # proposed at least what survived truncation
            @test norm(psi) ≈ 1.0 atol = 1e-8    # real time is unitary
        end
    end

    @testset "matches the analytic XX answer and 2-site TDVP" begin
        set_symmetry!(:U1)
        L, dt, tmax = 8, 0.02, 0.4
        h = xxz_chain(L; delta = 0.0)
        ref = _xx_analytic(L, tmax, magnetisation(domain_wall_state(L)) .+ 0.5)

        psi, _, _ = _run_tdvp1(L, h, ComplexF64(-im * dt), round(Int, tmax / dt))
        err1 = norm(magnetisation(copy(psi)) - ref)

        q = domain_wall_state(L)
        for _ in 1:round(Int, tmax / dt)
            tdvp2_step!(q, h, ComplexF64(-im * dt); maxdim = 32, trunc_thresh = 1e-12)
        end
        err2 = norm(magnetisation(copy(q)) - ref)

        # Loose absolute bound, plus a bound RELATIVE to 2-site TDVP. Measured POST-FIX at
        # 2.092258e-06 against tdvp2's 1.152198e-06, so a factor 1.82 -- the price of 1-site and
        # 0-site solves instead of evolving the rank-4 block. The factor of 10 leaves room for a
        # different machine without letting a real regression through.
        #
        # THESE FIGURES ARE BIT-FOR-BIT UNCHANGED BY THE BOUNDARY FIX, and that is the point:
        # the fix alters sites 1 and L, a domain wall's edges are frozen, so this setup cannot
        # register it either way. Do not read this test as evidence about boundary behaviour --
        # that is the Neel testset below. Kept because the analytic free-fermion reference is
        # still the cleanest check of the bulk.
        @test err1 < 1e-4
        @test err1 < 10 * err2
    end

    @testset "the BOUNDARY sites get a full step -- Neel start, edges active" begin
        set_symmetry!(:U1)
        # REGRESSION TEST FOR A BUG THE DOMAIN-WALL TESTS ABOVE CANNOT SEE.
        #
        # The rightward loop runs over bonds 1..L-1 and evolves each bond's LEFT site, so it
        # never touches site L; the leftward loop never touches site 1. Without the closing
        # one-site update at the far end of each pass, sites 1 and L receive tau/2 while every
        # interior site receives tau.
        #
        # A DOMAIN WALL HIDES THIS COMPLETELY -- its edges are frozen, so mis-integrating them
        # changes nothing, and the XX tests above passed at 2.09e-06 with the bug present. A
        # NEEL start at delta = 1 makes every site active and the same code was 5.73e-02 against
        # 2-site TDVP's 6.99e-09 at L = 12. Hence this test uses Neel, delta = 1, and compares
        # against an independent scheme rather than against itself.
        L, dt, tmax = 10, 0.02, 0.3
        h = xxz_chain(L; delta = 1.0)
        nst = round(Int, tmax / dt)

        psi = neel_state(L)
        for _ in 1:nst
            tdvp1_cbe_step!(psi, h, ComplexF64(-im * dt); maxdim = 48,
                            trunc_thresh = 1e-12)
        end
        q = neel_state(L)
        for _ in 1:nst
            tdvp2_step!(q, h, ComplexF64(-im * dt); maxdim = 48, trunc_thresh = 1e-12)
        end

        a, b = magnetisation(copy(psi)), magnetisation(copy(q))
        # With the boundary bug this was ~5e-2; with it fixed the two schemes agree far closer.
        @test norm(a - b) < 1e-3
        # And specifically at the ENDS, which is where the bug lived.
        @test abs(a[1] - b[1]) < 1e-4
        @test abs(a[end] - b[end]) < 1e-4
        @test norm(psi) ≈ 1.0 atol = 1e-8
    end

    @testset "second order in dt" begin
        set_symmetry!(:U1)
        L, tmax = 8, 0.4
        h = xxz_chain(L; delta = 0.0)
        ref = _xx_analytic(L, tmax, magnetisation(domain_wall_state(L)) .+ 0.5)
        errs = Float64[]
        for dt in (0.04, 0.02, 0.01)
            psi, _, _ = _run_tdvp1(L, h, ComplexF64(-im * dt), round(Int, tmax / dt))
            push!(errs, norm(magnetisation(copy(psi)) - ref))
        end
        # `tau/2` forward + `tau/2` backward sweep, so ratio ~4 per halving. Measured 3.92, 3.96.
        for k in 1:(length(errs) - 1)
            @test 3.3 < errs[k] / errs[k + 1] < 4.7
        end
    end

    @testset "exact and sketched expansions agree" begin
        set_symmetry!(:U1)
        L, dt, tmax = 8, 0.02, 0.4
        h = xxz_chain(L; delta = 0.0)
        ref = _xx_analytic(L, tmax, magnetisation(domain_wall_state(L)) .+ 0.5)
        nst = round(Int, tmax / dt)
        a, _, _ = _run_tdvp1(L, h, ComplexF64(-im * dt), nst; exact = false)
        b, _, _ = _run_tdvp1(L, h, ComplexF64(-im * dt), nst; exact = true)
        # THE CENTRAL CORRECTNESS CHECK FOR THE PAIR OF ARMS: `exact` swaps the entire selection
        # rule -- complete fused basis vs Gaussian sketch -- so if both are implemented right
        # they must land on the same state, and if either is wrong they will not. Measured
        # 1.917285e-12, i.e. agreement at the Krylov tolerance rather than merely "close".
        #
        # Note this says the sketch LOSES NOTHING here; it says nothing about which is faster or
        # about behaviour when rank is scarce. At L=8, t=0.4 the cap never binds, so no
        # selection pressure is being applied to either arm (docs 7.8.2.2).
        @test norm(magnetisation(copy(a)) - magnetisation(copy(b))) < 1e-8
        @test norm(magnetisation(copy(b)) - ref) < 1e-4
    end

    @testset "imaginary time descends toward the ground state" begin
        set_symmetry!(:U1)
        L, dt = 8, 0.05
        h = xxz_chain(L; delta = 1.0)
        gates = Any[heisenberg_bond_gate(neel_state(L)[i].inds[2],
                                         neel_state(L)[i + 1].inds[2]; delta = 1.0)
                    for i in 1:(L - 1)]
        psi = neel_state(L)
        e_start = real(energy(psi, gates))
        es = Float64[]
        for k in 1:120
            tdvp1_cbe_step!(psi, h, ComplexF64(-dt); maxdim = 32, trunc_thresh = 1e-12)
            n = norm(psi)
            n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
            k % 20 == 0 && push!(es, real(energy(psi, gates)))
        end
        @test es[end] < e_start                     # cooling lowers the energy
        @test issorted(es; rev = true) || all(es[k + 1] <= es[k] + 1e-8
                                              for k in 1:(length(es) - 1))
        # beta = 6 here, deliberately short for test time, so this is NOT a converged-energy
        # assertion -- at that beta the residual is dominated by leftover excited-state weight
        # (docs 7.7.3). The claim is only that cooling works and stays stable.
        @test es[end] < -3.0
    end

    @testset "rejects a mismatched chain" begin
        set_symmetry!(:U1)
        psi = domain_wall_state(6)
        @test_throws DimensionMismatch tdvp1_cbe_step!(psi, xxz_chain(8), ComplexF64(-im * 0.01))
    end
end
