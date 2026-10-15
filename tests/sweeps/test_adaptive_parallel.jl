# tests/sweeps/test_adaptive_parallel.jl
#
# `parallel = true` runs the two half-sweeps (and the closure's frontier half-sweeps) as two
# tasks. It must be the SAME METHOD as the serial sweep on every adaptive path the campaign uses:
# ordinary, midpoint, and the order-p Krylov closure.
#
#   exact = true  no random numbers are drawn, so parallel and serial must agree to round-off,
#                 ledger verdict included -- any difference is a race or a shared-state bug.
#   RSVD          the right half-sweep draws from its own seeded stream when parallel, so the
#                 sketches differ; the states must agree to the tolerance, and repeated parallel
#                 runs must agree EXACTLY (independent tasks, not "usually wins the race").

@testset "adaptive BUG: parallel half-sweeps == serial" begin
    set_symmetry!(:U1)
    L = 10
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    tau = ComplexF64(-0.05im)
    akw = (tol_basis = 1e-7, tol_trunc = 1e-9, tol_expv = 1e-10, maxdim = 64)
    fid(a, b) = abs(overlap(a, b)) / (norm(a) * norm(b))
    schemes = (("ordinary", (p; kw...) -> cbe_bug_step!(p, mpo, tau; kw...)),
               ("midpoint", (p; kw...) -> cbe_bug_midpoint_step!(p, mpo, tau; kw...)),
               ("order 2",  (p; kw...) -> cbe_bug_step!(p, mpo, tau; order = 2, kw...)),
               ("order 4",  (p; kw...) -> cbe_bug_step!(p, mpo, 2 * tau; order = 4, kw...)))
    for (name, step!) in schemes
        @testset "$name" begin
            # exact selection: bit-level agreement expected
            s, p = neel_state(L), neel_state(L)
            for _ in 1:4
                is = step!(s; parallel = false, exact = true, akw...)
                ip = step!(p; parallel = true,  exact = true, akw...)
                @test budget_status(is.ledger) === budget_status(ip.ledger)
            end
            @test 1 - fid(s, p) < 1e-12
            @test bond_dims(s) == bond_dims(p)

            # RSVD selection: equal to tolerance, and parallel reproducible run to run
            s = neel_state(L)
            ps = [neel_state(L) for _ in 1:2]
            for _ in 1:4
                step!(s; parallel = false, akw...)
                for q in ps
                    step!(q; parallel = true, akw...)
                end
            end
            @test 1 - fid(s, ps[1]) < 1e-8
            @test 1 - fid(ps[1], ps[2]) < 1e-13
        end
    end
end
