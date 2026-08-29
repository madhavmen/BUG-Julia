# MINIMAL REPRODUCER FOR THE U(1) CYLINDER FAILURE, WITH A STACK TRACE.
#
# ⛔ THE OBSERVATION. `logtime_suite.jl cylinder` at `symmetry = :U1`: bug_rsvd, bug_par AND
# midpoint all return `E = NaN, mv = 0, chi = 0` in under two seconds, at EVERY `Lx` and every
# step count, while `tdvp2` runs normally (`E = -4.820089374, mv = 8012, chi = 16`). The same
# arms are fine under `:SU2` on the same geometry.
#
#     AssertionError: Duplicate TLIndex with non-empty itag in TLArray.inds:
#     TLIndex(svdL, '+', 0, 0, false)
#
# ⚠ NOT A THREADING BUG, WHICH IS WHERE IT LOOKS LIKE POINTING. `bug_par` reports it wrapped in a
# `TaskFailedException` because it launches its half-sweeps as tasks, but `bug_rsvd` and
# `midpoint` are single-threaded and fail identically -- so the task wrapper is the messenger.
#
# ⚠ AND `run_arm` SWALLOWS THE STACK TRACE. It catches, logs `err`, and records NaN so one broken
# arm cannot take the suite down -- correct for a campaign, useless for a diagnosis. This file
# exists to let the exception escape.
#
# ⛔ THE COLLISION CLASS IS ALREADY KNOWN AND WAS ALREADY FIXED ONCE. `cbe_core.jl` (~line 410)
# records the identical assertion from job 94959: the final-selection matrix `U0L' (HTheta) V0R'`
# carries the two bond legs the preselection SVDs made, and on Telum's DEFAULT tags both come back
# `svdL` with the same arrow -- a TLArray may not hold two identical indices. That was cured with
# distinct tag pairs (`_TAG_L`/`_TAG_LR`, `_TAG_R`/`_TAG_RR`, ...). Its reappearance means ANOTHER
# SVD in the BUG path is still taking the default tags; the candidates are the frame splits in
# `cbe_bug.jl` (~383-387) and the root split (~864), none of which pass tag arguments.
#
# ⚠ WHY U(1) AND NOT SU(2). Under `:SU2` `spin_vertices` returns ONE `S.S` vertex; under `:U1` it
# returns THREE (`Sp`, `Sm`, `Sz`). More vertices means more MPO channels and more bond legs alive
# at once, so a duplicate tag has more chances to meet its twin. The bug is therefore likely
# LATENT under SU(2) rather than absent -- do not read "SU(2) works" as "SU(2) is safe".
#
# Run:  julia -t 2 --project=. benchmarks/imaginary_time/u1_cylinder_repro.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

say(s) = (println(s); flush(stdout))

# ⚠ SMALLEST GEOMETRY THAT STILL REPRODUCES: the suite fails already at `Lx = 2, Ly = 4` (8 sites),
# so there is no need to pay for 16. If this passes and the suite still fails, the trigger is size
# or rank dependent and the next step is to raise `Lx`.
const LY = 4
const D  = 32

function try_arm(nm, sym, Lx; twosite = false)
    set_symmetry!(sym)
    L = Lx * LY
    Jm = square_cylinder_couplings(Lx, LY; J = 1.0, periodic_y = true)
    W = pair_mpo(L, Jm, spin_vertices(L))
    psi = dimer_state(L)
    dt = ComplexF64(0.05)
    try
        if twosite
            tdvp2_step!(psi, W, dt; maxdim = D, trunc_thresh = 1e-12, maxiter = 20)
        else
            cbe_bug_step!(psi, W, dt; exact = false, dover = 4, growth = 4.0, maxdim = D,
                          trunc_thresh = 1e-12, maxiter = 20)
        end
        say(@sprintf("  %-9s %-5s Lx=%d  OK    chi=%s", nm, string(sym), Lx,
                     string(maximum(bond_dims(psi)))))
        return true
    catch err
        say(@sprintf("  %-9s %-5s Lx=%d  FAILED", nm, string(sym), Lx))
        say("      " * sprint(showerror, err))
        for (i, fr) in enumerate(stacktrace(catch_backtrace()))
            i > 14 && break
            say("      [$i] $fr")
        end
        return false
    end
end

say("\n=== 1. IS IT U(1) AT ALL, OR U(1) ON THE CYLINDER? ===")
say("  A U(1) CHAIN uses the same three vertices as the U(1) cylinder but only")
say("  nearest-neighbour couplings, so it separates 'U(1) is broken' from")
say("  '2D coupling matrix is broken'.")
for sym in (:SU2, :U1)
    set_symmetry!(sym)
    L = 8
    Jm = [i + 1 == j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    W = pair_mpo(L, Jm, spin_vertices(L))
    psi = dimer_state(L)
    try
        cbe_bug_step!(psi, W, ComplexF64(0.05); exact = false, dover = 4, growth = 4.0,
                      maxdim = D, trunc_thresh = 1e-12, maxiter = 20)
        say(@sprintf("  chain     %-5s        OK    chi=%s", string(sym),
                     string(maximum(bond_dims(psi)))))
    catch err
        say(@sprintf("  chain     %-5s        FAILED  %s", string(sym), sprint(showerror, err)))
    end
end

say("\n=== 2. THE CYLINDER, BOTH SYMMETRIES, WITH A STACK TRACE ===")
for sym in (:SU2, :U1)
    try_arm("cbe_bug", sym, 2)
end

say("\n=== 3. tdvp2 ON THE SAME GEOMETRY (the arm the suite says survives) ===")
for sym in (:SU2, :U1)
    try_arm("tdvp2", sym, 2; twosite = true)
end

say("\n=== 4. ARE THE STATE'S OWN BOND TAGS CLEAN?  (both defects now fixed) ===")
say("  Defect 1, FIXED: `fusion_basis` discarded its `tag` through the final SVD, so every CBE")
say("  probe came back `svdL` and collided inside `cbe_expand`.")
say("  Defect 2, FIXED: `_dimer_state_projected` (sweep.jl:121) wrote `res.U` / `res.S*res.Vd`")
say("  back into the state WITHOUT restoring `L,i+1`, so the start state carried `svdL` on its")
say("  ODD bonds -- exactly that loop's range. `move_right!` then built `res.S * res.Vd` with")
say("  `S = (svdL, svdR)` and `Vd = (svdR, svdL)`: the same itag twice in one TLArray.")
say("  ⚠ ONLY `:U1`/`:none` REACHED IT -- `dimer_state` delegates here only when NOT `:SU2`,")
say("  which is why every BUG result in this repo had been `:SU2` or `:none`.")
say("  Below: NO site should report an svd tag, before OR after a step, in ANY mode.\n")
badtags(t) = [string(t.inds[k].itags) for k in 1:length(t.inds)]
dirty(t)   = any(occursin("svd", s) for s in badtags(t))

for sym in (:SU2, :U1, :none)
    set_symmetry!(sym)
    L = 8
    Jm = [i + 1 == j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    W = pair_mpo(L, Jm, spin_vertices(L))
    psi = dimer_state(L)
    nb = count(dirty, [psi[i] for i in 1:L])
    say(@sprintf("  %-5s dimer_state BEFORE any step: bond_dims = %-22s svd-tagged sites = %d %s",
                 string(sym), string(bond_dims(psi)), nb, nb == 0 ? "OK" : "*** STILL DIRTY ***"))
    for i in 1:L
        dirty(psi[i]) && say(@sprintf("      site %-2d legs = %s", i, join(badtags(psi[i]), " | ")))
    end
    # ⛔ NOW WITH TRUNCATION ON -- the DEFAULT path, and the one that used to throw. Running with
    # `truncate = false` was a diagnostic to reach the tags; it is not the thing that must work.
    try
        cbe_bug_step!(psi, W, ComplexF64(0.05); exact = false, dover = 4, growth = 4.0,
                      maxdim = D, trunc_thresh = 1e-12, maxiter = 20)
        na = count(dirty, [psi[i] for i in 1:L])
        say(@sprintf("  %-5s after a step, truncate=ON: chi = %-3s  svd-tagged sites = %d %s",
                     string(sym), string(maximum(bond_dims(psi))), na,
                     na == 0 ? "OK" : "*** DIRTY ***"))
    catch err
        say(@sprintf("  %-5s after a step, truncate=ON: FAILED: %s", string(sym),
                     first(sprint(showerror, err), 160)))
    end
end

say("\n=== 5. DOES U(1) NOW GIVE THE RIGHT PHYSICS, NOT MERELY A STEP THAT STOPS THROWING? ===")
say("  ⛔ A STEP THAT NO LONGER CRASHES IS NOT A FIXED STEP. The claim the symmetry arm rests on")
say("  is 'symmetry changes the COST, not the PHYSICS', so the three modes must agree on the")
say("  energy of the SAME state after the SAME evolution. `:none` is the control: it never")
say("  reached defect 2's crash but it DOES go through the same patched constructor, so if the")
say("  retag broke anything numerically, `:none` moves too and the agreement below breaks.")
say("  Heisenberg chain, L=8, 20 imaginary-time steps of dt=0.05.\n")

const EREF = Dict{Symbol, Float64}()
for sym in (:SU2, :U1, :none)
    set_symmetry!(sym)
    L = 8
    Jm = [i + 1 == j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    W = pair_mpo(L, Jm, spin_vertices(L))
    psi = dimer_state(L)
    try
        e0 = real(mpo_energy(psi, W))
        for _ in 1:20
            cbe_bug_step!(psi, W, ComplexF64(-0.05); exact = false, dover = 4, growth = 4.0,
                          maxdim = D, trunc_thresh = 1e-12, maxiter = 20)
            n = norm(psi); n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
        end
        e = real(mpo_energy(psi, W))
        EREF[sym] = e
        say(@sprintf("  %-5s  E(start) = %+.10f   E(after 20 steps) = %+.10f   chi = %s",
                     string(sym), e0, e, string(maximum(bond_dims(psi)))))
    catch err
        say(@sprintf("  %-5s  FAILED: %s", string(sym), first(sprint(showerror, err), 160)))
    end
end
if length(EREF) == 3
    d1 = abs(EREF[:U1] - EREF[:SU2])
    d2 = abs(EREF[:none] - EREF[:SU2])
    say(@sprintf("\n  |U1 - SU2|   = %.3e", d1))
    say(@sprintf("  |none - SU2| = %.3e", d2))
    say(max(d1, d2) < 1e-8 ? "  VERDICT: PASSED -- all three modes agree, U(1) is usable." :
        "  VERDICT: *** MODES DISAGREE -- the crash is gone but the physics is not right. ***")
else
    say("\n  VERDICT: *** an arm did not complete; see above. ***")
end

say("\nDONE")
