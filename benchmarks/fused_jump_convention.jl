# GATE 10 -- SETTLE `jump_transpose` BY STEPPING, AND AUDIT `norm` / `apply_shift!` / `add_mps`.
#
# ⛔ WHY A NEW FILE RATHER THAN ANOTHER RUNG ON `fused_freeze_probe.jl`. That ladder's GATE 5 was
# the designated arbiter of `jump_transpose` and it CANNOT arbitrate: it forms the perturbed state
# `p = rho0 + eta*Kz*rho0` with `add_mps` and reads `2^L <p|W|p>/eta`, and it returned the SAME
# constant `-149999.5` for BOTH settings -- a number that scales as `1/eta`, i.e. `eta` never
# entered the state at all. A gate whose answer does not depend on the variable it is scanning has
# measured nothing, and GATE 1.5 (same construction) is void for the same reason. Both were quoted
# as evidence AGAINST the swap hypothesis; neither is.
#
# ⚠ THE ALGEBRA THE STEPPING TEST IS CHECKING. At a boundary site with gain/loss rates G+/G-,
# acting on the maximally mixed fused vector `u = (e1 + e4)/sqrt2`:
#
#   correct   jump -> G+ e1 + G- e4 ;  anticommutator -> -G- e1 - G+ e4
#             sum   =  (G+ - G-)(e1 - e4)          =>  d<S^z>/dt  ~  2(G+ - G-)   NONZERO
#   swapped   jump -> G- e1 + G+ e4 ;  anticommutator UNCHANGED
#             sum   =  0 EXACTLY, for every mu and every eps
#
# So the wrong convention does not tilt the profile the wrong way -- it makes `|I>>` an EXACT fixed
# point. That is the observed symptom to the digit (`1-|<I|q>|^2 = 4.4e-16`, every arm, every dt,
# and unchanged when GATE 9 moved `mu` off +-1, which is why scanning `mu` could never break it).
#
# ⚠ AND WHY THE PROFILE, NOT A RESPONSE FUNCTION. Stepping and reading `<S^z_j>` back through
# `trace_rho` uses only machinery GATE 4 already proved sound (a pure product state evolves
# correctly through both arms). It does not touch `add_mps`, so it is independent of the defect
# audited below.
#
# Run:  julia -t 2 --project=. benchmarks/fused_jump_convention.jl

using LinearAlgebra, Printf
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

set_symmetry!(:none)
say(s) = (println(s); flush(stdout))
fmt(v) = join([@sprintf("%+.8f", x) for x in v], " ")

const KW = (J = 1.0, eps_L = 1.0, eps_R = 1.0, mu_L = 1.0, mu_R = -1.0)

tdvp2_arm(p, W, dt) = tdvp2_step!(p, W, ComplexF64(dt); maxdim = 128, trunc_thresh = 1e-12,
                                  maxiter = 30, hermitian = false)
bug_arm(p, W, dt)   = cbe_bug_step!(p, W, ComplexF64(dt); exact = false, dover = 4, growth = 4.0,
                                    maxdim = 128, trunc_thresh = 1e-12, maxiter = 30,
                                    hermitian = false)

"""
Evolve the maximally mixed state for `nsteps` steps of `dt` and return `(Tr, profile, chi)`.

⚠ `restore_trace!` IS OFF BY DEFAULT. It routes through `add_mps`, which is the very thing under
audit further down, so leaving it on would couple the convention question to an unrelated suspect.
The zero-body `shift` is still applied every step -- that one is a plain scalar and dropping it
would break the trace by construction (`fused_common.jl` §3.2 FAULT 1).
"""
function evolve(L, dt, nsteps, arm; jt::Bool, restore::Bool = false)
    W, shift = fused_boundary_lindblad(L; KW..., jump_transpose = jt)
    Imps = identity_superket(L)
    rho = copy(Imps); apply_shift!(rho, 1.0 / 2.0^(L / 2))
    for _ in 1:nsteps
        arm(rho, W, dt)
        apply_shift!(rho, exp(dt * shift))
        restore && restore_trace!(Imps, rho, L; maxdim = 128, cutoff = 1e-12)
    end
    trr = real(trace_rho(Imps, rho, L))
    return trr, [real(trace_rho(Imps, rho, L; at = j)) / trr for j in 1:L], bond_dims(rho)
end

# ── GATE 10a: one step, both conventions, both arms, L = 2 and L = 3 ────────────────────────────
say("\n=== GATE 10a: ONE STEP, BOTH JUMP CONVENTIONS, vs the exact Lyapunov profile ===")
say("  The wrong convention gives EXACTLY zero (|I>> is a fixed point), not a wrong sign.")
for L in (2, 3)
    dt = 0.01
    exact = xx_boundary_sz(L, dt; KW...)
    say(@sprintf("\n  L = %d, dt = %g   exact <S^z_j> = %s", L, dt, fmt(exact)))
    for (nm, arm) in (("tdvp2", tdvp2_arm), ("bug_rsvd", bug_arm))
        for jt in (false, true)
            try
                trr, prof, chi = evolve(L, dt, 1, arm; jt = jt)
                dev = maximum(abs.(prof .- exact))
                say(@sprintf("    %-9s jump_transpose=%-5s Tr=%.10f  <S^z_j> = %s  max|dev| = %.3e  %s",
                             nm, string(jt), trr, fmt(prof), dev,
                             dev < 1e-4 ? "<-- MATCHES the reference" :
                             maximum(abs.(prof)) < 1e-12 ? "FROZEN (exact fixed point)" : "wrong"))
            catch e
                say("    $nm jt=$jt THREW: " * sprint(showerror, e))
            end
        end
    end
end

# ── GATE 10b: convergence in dt for whichever convention moved ─────────────────────────────────
# A single step landing near the reference is necessary but not sufficient -- it would also happen
# for a generator that is right to O(dt) and wrong at O(dt^2). Fixed horizon, halving dt: the error
# must fall like the integrator's order, not sit on a floor.
say("\n=== GATE 10b: fixed horizon T = 0.1, halving dt (convergence, not a single step) ===")
for jt in (false, true)
    say("  jump_transpose = $jt")
    for L in (3,)
        exact = xx_boundary_sz(L, 0.1; KW...)
        prev = NaN
        for n in (2, 4, 8, 16)
            dt = 0.1 / n
            try
                _, prof, _ = evolve(L, dt, n, tdvp2_arm; jt = jt)
                err = maximum(abs.(prof .- exact))
                say(@sprintf("    L=%d n=%-3d dt=%.5f  max|dev| = %.6e   ratio = %s",
                             L, n, dt, err, isnan(prev) ? "--" : @sprintf("%.2f", prev / err)))
                prev = err
            catch e
                say("    n=$n THREW: " * sprint(showerror, e))
            end
        end
    end
end

# ── GATE 10c: audit `apply_shift!` / `norm` / `add_mps` USING THE TRACE ─────────────────────────
# ⛔ THE TRACE, NOT `norm`. GATE 8 concluded "add_mps normalises its arguments" from `|b|/|a| = 1`
# after `b = 0.001*a` -- but that reading came from `norm`, and
# `norm(psi::SymMPS) = norm(psi[psi.center])` is the Frobenius norm of the CENTRE TENSOR ALONE,
# which equals the state norm only when every other tensor is an isometry. `apply_shift!` scales
# SITE 1. When `center != 1` the scalar lands on a tensor the norm reading never looks at, so the
# STATE is scaled correctly and the NORM is misreported. The conclusion drawn from it does not
# follow, and the fix it implied would have been applied to the wrong function.
#
# `trace_rho` is linear in the state and contracts every site, so it cannot miss a scalar wherever
# it sits. That makes it the right instrument for this question.
say("\n=== GATE 10c: does `apply_shift!` scale the STATE? (measured by the trace, not by `norm`) ===")
let L = 3
    Imps = identity_superket(L)
    a = copy(Imps); apply_shift!(a, 1.0 / 2.0^(L / 2))
    tr_a = real(trace_rho(Imps, a, L))
    b = copy(a); apply_shift!(b, 1e-3)
    tr_b = real(trace_rho(Imps, b, L))
    say(@sprintf("  Tr(a) = %.12f   Tr(0.001*a) = %.12e   ratio = %.8f  [want 0.001]  %s",
                 tr_a, tr_b, tr_b / tr_a,
                 abs(tr_b / tr_a - 1e-3) < 1e-9 ? "apply_shift! is CORRECT" :
                                                  "*** apply_shift! DROPS the scalar ***"))
    say(@sprintf("  norm(a) = %.8f   norm(0.001*a) = %.8f   ratio = %.8f  [%s]",
                 norm(a), norm(b), norm(b) / norm(a),
                 abs(norm(b) / norm(a) - 1e-3) < 1e-9 ? "norm agrees" :
                 "norm MISREPORTS -- centre-tensor norm, see the note above"))
    say("\n  and does `add_mps` preserve the relative weight? (trace is linear: Tr(a+b) = Tr a + Tr b)")
    try
        c = add_mps(copy(a), copy(b))
        tr_c = real(trace_rho(Imps, c, L))
        say(@sprintf("  Tr(a+b) = %.12f   want %.12f   ratio = %.8f   %s",
                     tr_c, tr_a + tr_b, tr_c / (tr_a + tr_b),
                     abs(tr_c / (tr_a + tr_b) - 1) < 1e-9 ? "add_mps is CORRECT" :
                                                            "*** add_mps LOSES the weight ***"))
    catch e
        say("  add_mps THREW: " * sprint(showerror, e))
    end
end

say("\nDONE")
