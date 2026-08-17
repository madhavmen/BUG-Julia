# THE PHYSICS: Haldane-Shastry and Heisenberg, ground state / imaginary time / real time, for the
# new CBE + fixed-rank-BUG sweep and for `jan_bug_step!`, against ANALYTIC references.
#
# WHAT IS ANALYTIC AND WHAT IS NOT -- stated up front, because the honest scope of each test is
# part of the test:
#
#   GROUND STATE     Haldane-Shastry has the closed form `E_0 = -pi^2(L^2+5)/(24L)`; the
#                    Heisenberg RING has the Bethe ansatz (exact, solved numerically from the
#                    Bethe equations, not fitted). Both are pinned against exact diagonalisation
#                    in `tests/common/test_analytic_reference.jl` BEFORE being used here.
#   IMAGINARY TIME   converges to the ground state, so the same references apply -- and the
#                    convergence itself is the test, since a wrong sign or a lost normalisation
#                    shows up as ascent rather than descent.
#   REAL TIME        SU(2) XXX and Haldane-Shastry have NO analytic solution from a generic
#                    state. Two things are analytic and both are used:
#
#                      (a) MAGNON PROPAGATION. A single down spin among ups is a PRODUCT state
#                          (bond dimension 1, trivially constructible) lying entirely in the
#                          one-magnon sector, which for ANY translation-invariant two-body `H` on
#                          a ring is an exact eigenspace. So its evolution is a closed-form
#                          Fourier sum -- `magnon_amplitudes` -- and `<S_j^z>(t)` can be compared
#                          site by site. This is a genuine dynamical test: the magnon actually
#                          propagates and disperses. It is U(1)/`:none` only, because a single
#                          flipped spin is NOT a definite-total-spin state and SU(2) cannot
#                          represent it at any bond dimension.
#                      (b) EIGENSTATE PHASE. An eigenstate evolves by a pure phase, so
#                          `|<psi(0)|psi(t)>| = 1` and the energy is constant at the analytic
#                          value. Available in every symmetry mode, and it is the SU(2) real-time
#                          reference.
#
#                    What is NOT claimed: an analytic real-time reference for a generic,
#                    high-entanglement SU(2) state. There is none, and no test here pretends
#                    otherwise.
#
# The magnon test's weakness is stated too: a one-magnon state has bond dimension 2, so it does
# not exercise rank growth. It tests the DYNAMICS exactly; the rank machinery is tested by the
# `dt`-order and full-rank-exactness tests in `test_sweeps.jl` and by the ground-state runs here,
# where the rank does grow.

using Test
using LinearAlgebra
import Random

@testset "physics: Haldane-Shastry and Heisenberg" begin

    renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

    # ── ground state ──────────────────────────────────────────────────────────────────────
    @testset "Haldane-Shastry ground energy == the analytic formula" begin
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            for L in (6, 8)
                m = haldane_shastry_mpo(L)
                psi = dimer_state(L)
                r = dmrg_cbe1s!(psi, m; n_sweeps = 30, maxdim = 128, trunc_thresh = 1e-14)
                want = hs_ground_energy(L)
                @test r.energies[end] ≈ want atol = 1e-6
                # VARIATIONAL: a value below the exact ground energy is a bug.
                @test r.energies[end] >= want - 1e-8
                @info "HS $sym L=$L: E = $(r.energies[end]), exact = $want, " *
                      "bd = $(r.bond_dims[end]), states = $(state_bond_dims(psi))"
            end
        end
    end

    @testset "Heisenberg RING ground energy == the Bethe ansatz" begin
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            for L in (6, 8)
                m = heisenberg_ring_mpo(L)
                psi = dimer_state(L)
                r = dmrg_cbe1s!(psi, m; n_sweeps = 30, maxdim = 128, trunc_thresh = 1e-14)
                want, _, res = bethe_xxx_ring_energy(L)
                @test res < 1e-12
                @test r.energies[end] ≈ want atol = 1e-6
                @test r.energies[end] >= want - 1e-8
            end
        end
    end

    # ── imaginary time ────────────────────────────────────────────────────────────────────
    @testset "imaginary time descends to the analytic ground energy" begin
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            L = 6
            for (nm, m, want) in (("HS", haldane_shastry_mpo(L), hs_ground_energy(L)),
                                  ("ring", heisenberg_ring_mpo(L),
                                   first(bethe_xxx_ring_energy(L))))
                for (sname, step!) in
                    (("cbe_bug", (p, t) -> cbe_bug_step!(p, m, t; maxdim = 128,
                                                         trunc_thresh = 1e-14)),
                     ("jan_bug", (p, t) -> jan_bug_step!(p, m, t; lossless = true,
                                                         maxdim = 128,
                                                         trunc_thresh = 1e-14)))
                    psi = dimer_state(L)
                    e0 = real(mpo_energy(psi, m))
                    es = Float64[]
                    for _ in 1:120
                        step!(psi, -0.1)
                        renorm!(psi)
                        push!(es, real(mpo_energy(psi, m)))
                    end
                    # It descended, and it descended TO the analytic value.
                    @test es[end] < e0
                    @test es[end] ≈ want atol = 1e-4
                    # Never below: imaginary-time evolution of a normalised state is a
                    # variational descent, so undershooting the exact energy is a defect.
                    @test es[end] >= want - 1e-6
                    @info "ITE $nm $sname $sym L=$L: E = $(es[end]), exact = $want"
                end
            end
        end
    end

    # ── real time (a): magnon propagation, exactly ────────────────────────────────────────
    @testset "magnon propagation == the exact Fourier solution" begin
        # A single down spin at site 1 among ups: bond dimension 1, entirely inside the
        # one-magnon sector, and its exact evolution is `magnon_amplitudes` with `c_n` the
        # Fourier transform of a delta at site 1.
        set_symmetry!(:U1)
        L = 8
        for (nm, m, Jr) in (("HS", haldane_shastry_mpo(L), hs_couplings_analytic(L)),
                            ("ring", heisenberg_ring_mpo(L),
                             [r == 1 || r == L - 1 ? 1.0 : 0.0 for r in 1:(L - 1)]))
            c = ComplexF64[cis(-2 * pi * n / L) / sqrt(L) for n in 0:(L - 1)]
            # Sanity on the convention: at `t = 0` the wavepacket IS the delta at site 1.
            a0 = magnon_amplitudes(c, 0.0, Jr; L = L)
            @test abs(a0[1]) ≈ 1.0 atol = 1e-10
            @test maximum(abs.(a0[2:end])) < 1e-10

            for (sname, step!) in
                (("cbe_bug", (p, t) -> cbe_bug_step!(p, m, t; maxdim = 64,
                                                     trunc_thresh = 1e-14)),
                 ("tdvp_cbe1s", (p, t) -> tdvp_cbe1s_step!(p, m, t; maxdim = 64,
                                                           trunc_thresh = 1e-14)),
                 ("jan_bug", (p, t) -> jan_bug_step!(p, m, t; lossless = true, maxdim = 64,
                                                     trunc_thresh = 1e-14)))
                psi = product_state([j == 1 ? :down : :up for j in 1:L])
                dt = 0.01; nst = 20
                for _ in 1:nst
                    step!(psi, -im * dt)
                    renorm!(psi)
                end
                T = dt * nst
                a = magnon_amplitudes(c, T, Jr; L = L)
                # <S_j^z> = 1/2 - |a_j|^2, exactly, inside the one-magnon sector.
                want = [0.5 - abs2(a[j]) for j in 1:L]
                got = magnetisation(copy(psi))
                err = maximum(abs.(got .- want))
                @info "RTE $nm $sname L=$L T=$T: max |dSz| = $err"
                # The magnon must actually have MOVED, or this is not a dynamical test.
                @test maximum(abs.(want .- [0.5 - (j == 1 ? 1.0 : 0.0) for j in 1:L])) > 1e-3
                @test err < 1e-4
            end
        end
    end

    # ── real time (b): an eigenstate evolves by a pure phase ──────────────────────────────
    @testset "an eigenstate keeps |<psi(0)|psi(t)>| = 1 and its analytic energy" begin
        # The real-time reference available in EVERY symmetry mode, SU(2) included: converge to
        # the ground state, then evolve it in real time. It is an eigenstate, so the state is
        # invariant up to a phase and the energy sits at the analytic value for the whole run.
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            L = 6
            m = haldane_shastry_mpo(L)
            psi = dimer_state(L)
            dmrg_cbe1s!(psi, m; n_sweeps = 30, maxdim = 128, trunc_thresh = 1e-14)
            renorm!(psi)
            psi0 = copy(psi); psi0.tensors .= copy.(psi.tensors)
            want = hs_ground_energy(L)
            @test real(mpo_energy(psi, m)) ≈ want atol = 1e-6
            for _ in 1:10
                cbe_bug_step!(psi, m, -0.05im; maxdim = 128, trunc_thresh = 1e-14)
                renorm!(psi)
            end
            @test abs(overlap(psi0, psi)) ≈ 1.0 atol = 1e-6
            @test real(mpo_energy(psi, m)) ≈ want atol = 1e-6
        end
    end

    # ── the fully polarised state, an exactly known eigenstate of any two-body H ───────────
    @testset "the polarised state is an eigenstate at the analytic energy" begin
        # The cheapest available check that the HS MPO's overall NORMALISATION is right, and it
        # is worth having separately from the ground-state test: Telum scales SU(2) generators
        # through Clebsch-Gordan factors, and `<F|H|F> = (1/4) * (number of pairs weighted by J)`
        # is a closed form with no diagonalisation in it at all.
        set_symmetry!(:U1)
        for L in (6, 8)
            m = haldane_shastry_mpo(L)
            Jr = hs_couplings_analytic(L)
            psi = product_state([:up for _ in 1:L])
            @test real(mpo_energy(psi, m)) ≈ magnon_reference_energy(Jr; L = L) atol = 1e-10
            # And it stays put under real time, since it is an eigenstate.
            psi0 = copy(psi); psi0.tensors .= copy.(psi.tensors)
            for _ in 1:5
                cbe_bug_step!(psi, m, -0.05im; maxdim = 32, trunc_thresh = 1e-14)
                renorm!(psi)
            end
            @test abs(overlap(psi0, psi)) ≈ 1.0 atol = 1e-9
        end
    end
end
