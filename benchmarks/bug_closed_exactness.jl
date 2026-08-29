# DOES THE FULL-RANK DEFECT REACH THE CLOSED-SYSTEM RESULTS TOO?
#
# ⛔ WHAT WAS MEASURED ON THE OPEN SIDE (`bug_exactness_probe.jl`, fused L=2, d=4): the Galerkin
# basis is the COMPLETE space -- `b_L = b_R = 4`, so `b_L x b_R = 16 = 4^2` -- and `cbe_bug` is
# STILL wrong by 9.55e-06, scaling as O(dt^2). With a complete basis `exp(tau*H0)S0` IS the exact
# propagator (`W S0 Z = psi` for unitary frames), so the step must be exact to roundoff.
# ⛔ AND `growth` STILL CHANGES THE ANSWER at `b_L = b_R = 4` (2.63e-03 at growth 2 vs 9.55e-06 at
# growth 4). Once the basis is complete there is no degree of freedom left for it to act on. Two
# independent signatures of a DEFECT, not of a manifold projection.
#
# THE QUESTION HERE: is it specific to the open-system/`hermitian = false` path, or does the same
# thing happen on a plain CLOSED spin-1/2 chain? If the latter, every `cbe_bug` number in this
# study carries an error where the method should be exact.
#
# ⚠ THE TEST IS "EXACT AT FULL RANK", NOT "ACCURATE". At `L = 2`, `d = 2`, the bond is at most 2
# and the Galerkin space `b_L x b_R <= 2 x 2 = 4 = 2^2` is the whole Hilbert space. At `L = 3` it
# is not (bond 1 carries at most 2 while the space is 8), so L=3 is EXPECTED to be inexact and is
# included only as the contrast -- reading it as a failure would be the trap.
#
# ⚠ REFERENCE IS THE DENSE PROPAGATOR, not another integrator: comparing two approximations cannot
# tell which one moved.
#
#   julia --project=. benchmarks/bug_closed_exactness.jl

using Printf, LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "bug_closed_exactness.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

const T = 0.3

for L in (2, 3)
    full = 2^L
    # the Galerkin space at the root bond: b_L <= d^min(c, L-c) capped by d, b_R likewise
    say(@sprintf("== L = %d spin-1/2 (dim %d). Root bond c = %d; Galerkin space b_L x b_R.",
                 L, full, max(1, L ÷ 2)))
    L == 2 && say("   b_L, b_R <= 2 so the space is 2x2 = 4 = the WHOLE Hilbert space -> BUG MUST be exact.")
    L == 3 && say("   b_L <= 2 while the space is 8 -> NOT complete, so inexactness here is EXPECTED.")
    H = dense_heisenberg(L; J = 1.0, delta = 1.0)
    W = xxz_mpo(L; J = 1.0, delta = 1.0)
    spins = [isodd(k) ? :up : :down for k in 1:L]
    v0 = dense_state(product_state(spins))
    want = exp(-im * T * H) * v0
    say(@sprintf("  %-26s %6s %13s %6s %6s %6s", "arm", "steps", "state dev", "chi", "b_L", "b_R"))
    for (nm, mk) in (
        ("tdvp2", (dt) -> (p) -> tdvp2_step!(p, W, ComplexF64(-im * dt);
                        maxdim = 64, trunc_thresh = 1e-14, maxiter = 30)),
        ("bug_interleaved g=2", (dt) -> (p) -> cbe_bug_step!(p, W, ComplexF64(-im * dt);
                        growth = 2.0, maxdim = 64, trunc_thresh = 1e-14, maxiter = 30)),
        ("bug_interleaved g=8", (dt) -> (p) -> cbe_bug_step!(p, W, ComplexF64(-im * dt);
                        growth = 8.0, maxdim = 64, trunc_thresh = 1e-14, maxiter = 30)),
        ("bug_interleaved g=32", (dt) -> (p) -> cbe_bug_step!(p, W, ComplexF64(-im * dt);
                        growth = 32.0, maxdim = 64, trunc_thresh = 1e-14, maxiter = 30)))
        for n in (3, 48)
            p = copy(product_state(spins)); dt = T / n
            bl = 0; br = 0
            f = mk(dt)
            for _ in 1:n
                info = f(p)
                if info !== nothing && hasproperty(info, :expanded)
                    bl = max(bl, maximum(info.expanded; init = 0))
                    hasproperty(info, :root_right) && (br = max(br, info.root_right))
                end
            end
            d = maximum(abs.(dense_state(p) .- want))
            say(@sprintf("  %-26s %6d %13.4e %6d %6d %6d", nm, n, d,
                         maximum(state_bond_dims(p)), bl, br))
        end
    end
    say("")
end
say("VERDICT (the L = 2 block is the one that decides):")
say("  bug dev ~ 1e-14 at L=2  -> the defect is specific to the open-system path, and the closed")
say("     campaign results are unaffected.")
say("  bug dev >> 1e-14 at L=2 with b_L = b_R = 2 -> the defect is GENERAL. Every cbe_bug number")
say("     in this study then carries an error in a regime where the method should be exact, and")
say("     `growth` changing the answer at a complete basis is the second proof.")
close(io)
