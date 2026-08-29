# DOES SPLITTING `tau` INSIDE THE KRYLOV SOLVE FIX THE LARGE-STEP FAILURE, AND WHAT DOES IT COST?
#
# THE FAILURE THIS IS ABOUT. On a logarithmic imaginary-time grid at L=20 the last step reaches
# `dt ~ 46-54`, and there both integrators broke: BUG returned exactly `obs = 0`, 2-site TDVP threw
# `UndefRefError`. Three explanations were on the table and only one survives measurement:
#
#   (a) a ROBUSTNESS failure of BUG        -- would contradict Ceruti-Lubich (the BUG error bound
#                                             does not contain `1/sigma_min`, unlike
#                                             projector-splitting TDVP)
#   (b) too few KRYLOV VECTORS             -- ⛔ REFUTED, and it was my prime suspect.
#                                             `benchmarks/imaginary_time/krylov_depth_scan.jl` measured the depth
#                                             needed for ~1e-14 in the half-sweep basis:
#                                               dt   0.1   1    5    20
#                                               m    3     8    16   16-24
#                                             The depth SATURATES -- it does not grow like
#                                             `tau*||H||` -- and 24 is still under the default cap
#                                             of 30. Reading a trend off the first three rows was
#                                             the error; the fourth row settles it.
#   (c) FLOATING-POINT OVERFLOW in the exponential -- what is left, and it is structural:
#       `hermitian_tridiagonal_exp_coeffs` formed `exp.(tau .* vals)` UNSHIFTED, so at `tau = -50`
#       with a root spectrum reaching `-20` that is `exp(1000)` against a Float64 ceiling of
#       `exp(709)`. IMAGINARY TIME ONLY: in real time `tau = -i*dt` makes it a pure phase, which
#       is why nothing saw this until the log grid.
#
# So the fix is not a bigger `m`. It is to stop asking for one huge exponential:
# `exp(tau*A) = exp(tau_s*A)^s` with `|tau_s|*rho` bounded, plus a spectral shift. Neither needs
# the caller to know `m`, which was the point.
#
# ⛔ SECTION 1 IS THE REGRESSION GUARD AND MATTERS MORE THAN SECTION 2. A change to the shared
# `expv` touches BUG, 2-site TDVP, `tdvp_cbe1s`, `tdvp1_cbe` and the DMRG eigensolver at once. If
# the adaptive path moved the answer at ORDINARY step sizes, every number this project has
# measured would be in question. Section 1 checks agreement at `dt` where the old path is known
# good; only then is Section 2's rescue worth anything.
#
# ⛔ MPO ONLY, NEVER GATES.
#
# Run:  julia --project=. benchmarks/imaginary_time/expv_substep_check.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))

renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)
_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

function infidelity(a, b)
    ov = overlap(copy(a), copy(b))
    na, nb = norm(a), norm(b)
    (isfinite(na) && isfinite(nb) && na > 0 && nb > 0) || return NaN
    return max(0.0, 1 - abs2(ov) / (na^2 * nb^2))
end

energy(p, W) = (n = norm(p); isfinite(n) && n > 0 ? real(mpo_energy(copy(p), W)) / n^2 : NaN)

function main(; L = 20, D = 64, maxiter = 40)
    set_symmetry!(:SU2)
    W = heisenberg_su2_mpo(L)
    eref, _, res = bethe_xxx_open_energy(L)
    res < 1e-12 || error("Bethe residual $res")
    @printf("HEISENBERG SU(2) L=%d D=%d  E_Bethe=%.10f\n\n", L, D, eref)

    warm = dimer_state(L)
    for _ in 1:3
        cbe_bug_step!(warm, W, ComplexF64(-0.1); maxdim = D, trunc_thresh = 1e-12,
                      maxiter = maxiter, exact = true)
        renorm!(warm)
    end

    # arms: (label, kwargs for the root solve)
    old  = (root_conv_tol = 0.0,   root_substeps = 1)   # historical behaviour
    conv = (root_conv_tol = 1e-12, root_substeps = 1)   # convergence test only
    sub  = (root_conv_tol = 1e-12, root_substeps = 0)   # + adaptive tau splitting

    step(p, dt, kw) = cbe_bug_step!(p, W, ComplexF64(-dt); maxdim = D, trunc_thresh = 1e-12,
                                    maxiter = maxiter, exact = true, kw...)

    println("== 1. REGRESSION GUARD: ordinary step sizes, where the old path is known good ==")
    @printf("  %-6s %-10s %14s %10s %12s %8s\n",
            "dt", "arm", "infid vs old", "matvec", "E", "secs")
    for dt in (0.05, 0.5, 2.0)
        ref = copy(warm); t0 = time(); iref = step(ref, dt, old); tref = time() - t0
        renorm!(ref)
        @printf("  %-6g %-10s %14s %10d %12.8f %7.2fs\n",
                dt, "old", "-", _kry(iref), energy(ref, W), tref)
        for (nm, kw) in (("conv", conv), ("substep", sub))
            p = copy(warm); t0 = time(); i = step(p, dt, kw); el = time() - t0
            renorm!(p)
            @printf("  %-6g %-10s %14.3e %10d %12.8f %7.2fs\n",
                    dt, nm, infidelity(p, ref), _kry(i), energy(p, W), el)
        end
        flush(stdout)
    end

    println("\n== 2. THE RESCUE: steps large enough to break the old path ==")
    # ⛔ THE REFERENCE HERE IS THE ANALYTIC BETHE ENERGY, not another run of the same code.
    # `exp(-tau*H)` for tau >= 30 from a warm state IS the ground state to far better than the
    # truncation floor, so a correct large step must land on `eref`. Comparing against another
    # numerical arm could only show two arms agreeing, including on being wrong.
    @printf("  %-6s %-10s %14s %10s %12s %8s\n",
            "dt", "arm", "|E - Bethe|", "matvec", "E", "secs")
    for dt in (30.0, 50.0, 80.0, 200.0)
        for (nm, kw) in (("old", old), ("conv", conv), ("substep", sub))
            p = copy(warm)
            t0 = time()
            e, mv = try
                i = step(p, dt, kw); renorm!(p); (energy(p, W), _kry(i))
            catch err
                @printf("  %-6.0f %-10s %14s %10s %12s  THREW: %s\n", dt, nm, "-", "-", "-",
                        first(split(sprint(showerror, err), '\n')))
                flush(stdout); continue
            end
            @printf("  %-6.0f %-10s %14.3e %10d %12.8f %7.2fs\n",
                    dt, nm, abs(e - eref), mv, e, time() - t0)
            flush(stdout)
        end
    end
    println("\ndone")
end

main()
