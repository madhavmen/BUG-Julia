# HOW MANY KRYLOV VECTORS `m` DOES ONE IMAGINARY-TIME STEP NEED, AS A FUNCTION OF `dt`?
#
# THE QUESTION A LOG GRID FORCES. `exp(-tau*H)v` is built from `span{v, Hv, H^2 v, ...}`, and the
# depth needed grows with `tau*||H||`. On a logarithmic grid `tau` spans THREE ORDERS OF MAGNITUDE
# within a single run -- 0.05 to 50 here -- so a depth that is right for the first step cannot
# also be right for the last. A single `krylov_basis` for the whole run is therefore a compromise,
# and this file measures what it is compromising between.
#
# ⛔ THIS IS THE PRIME SUSPECT FOR THE LARGE-STEP FAILURES, ahead of the two explanations I
# reached for first. On the log grid at L=20 both integrators broke at `dt ~ 46-54`: BUG returned
# exactly `obs = 0`, TDVP2 threw from `_state_stored`. I proposed (a) a robustness failure and
# then (b) floating-point overflow. But `cbe_bug_step!` caps the half-sweep basis at
# `krylov_basis = 30` and stops early on `krylov_tol = 1e-6`, and at `dt = 50` with `||H|| ~ 20`
# the argument is `tau*||H|| ~ 1000` -- a 30-vector basis need not represent that at all.
#
# If depth is the cause then BUG's robustness (Ceruti-Lubich: the error bound does not involve
# `sigma_min`, unlike projector-splitting TDVP's `1/sigma_min`) is UNTOUCHED, and the ceiling I
# measured was a parameter choice of mine rather than a property of the integrator. That
# distinction matters for the claim being made against arXiv:2606.02930 and cannot be settled by
# an end-of-run number.
#
# ⛔ `krylov_tol` IS SET TO ZERO IN THE SCAN. Otherwise the early-exit fires and the arm labelled
# `m = 64` may actually have built four vectors -- the scan would then measure the TOLERANCE and
# report it as a depth. Depth is forced; the tolerance is a separate axis.
#
# THE REFERENCE IS A VERY DEEP RUN AT THE SAME `dt`, not a closed form: the question is
# CONVERGENCE IN `m`, so the right reference is the same step with `m` large enough to have
# stopped moving. Convergence of the reference itself is checked by comparing two deep values.
#
# ⛔ MPO ONLY, NEVER GATES.
#
# Run:  julia --project=. benchmarks/imaginary_time/krylov_depth_scan.jl
# Out:  benchmarks/results/krylov_depth.csv -> plotted by plot_logtime_traj.py

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const OUT  = joinpath(@__DIR__, "..", "results", "krylov_depth.csv")
const COLS = ["L", "dt", "m", "err_vs_deep", "matvec", "chi", "seconds"]

renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)
_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

"Distance between two states, gauge-independent: 1 - |<a|b>|^2 / (<a|a><b|b>)."
function infidelity(a, b)
    ov = overlap(copy(a), copy(b))
    na, nb = norm(a), norm(b)
    (isfinite(na) && isfinite(nb) && na > 0 && nb > 0) || return NaN
    return max(0.0, 1 - abs2(ov) / (na^2 * nb^2))
end

function main(; L = 20, D = 64, maxiter = 40)
    set_symmetry!(:SU2)
    W = heisenberg_su2_mpo(L)
    warm = dimer_state(L)
    for _ in 1:3
        cbe_bug_step!(warm, W, ComplexF64(-0.1); maxdim = D, trunc_thresh = 1e-12,
                      maxiter = maxiter, exact = true, krylov_basis = 40, krylov_tol = 0.0)
        renorm!(warm)
    end
    @printf("HEISENBERG SU(2) L=%d D=%d   warm E=%.6f\n", L, D,
            real(mpo_energy(copy(warm), W)) / norm(warm)^2)

    step(p, dt, m) = cbe_bug_step!(p, W, ComplexF64(-dt); maxdim = D, trunc_thresh = 1e-12,
                                   maxiter = maxiter, exact = true,
                                   krylov_basis = m, krylov_tol = 0.0)

    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, join(COLS, ","))
        for dt in (0.1, 1.0, 5.0, 20.0, 50.0)
            # reference: deep, and CHECKED against a second deep value so a drifting reference
            # cannot masquerade as convergence
            ref  = copy(warm); step(ref, dt, 300); renorm!(ref)
            ref2 = copy(warm); step(ref2, dt, 220); renorm!(ref2)
            dref = infidelity(ref, ref2)
            @printf("\ndt = %-6g   reference m=300 vs m=220: %.3e %s\n", dt, dref,
                    isfinite(dref) && dref < 1e-13 ? "(converged)" : "(NOT CONVERGED -- suspect)")
            @printf("  %-5s %14s %9s %5s %8s\n", "m", "infid vs deep", "matvec", "chi", "time")
            for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
                p = copy(warm); t0 = time()
                mv = 0
                d = try
                    mv = _kry(step(p, dt, m)); renorm!(p); infidelity(p, ref)
                catch err
                    @info "depth arm threw" dt m err
                    NaN
                end
                el = time() - t0
                @printf(io, "%d,%g,%d,%.6e,%d,%d,%.3f\n", L, dt, m, d, mv,
                        isfinite(d) ? maximum(bond_dims(p)) : 0, el)
                flush(io)
                @printf("  %-5d %14.4e %9d %5d %7.2fs%s\n", m, d, mv,
                        isfinite(d) ? maximum(bond_dims(p)) : 0, el,
                        isfinite(d) && d < 1e-12 ? "   <- converged" : "")
                flush(stdout)
                isfinite(d) && d < 1e-14 && break     # deeper is pure cost once converged
            end
        end
    end
    println("\nwrote ", OUT)
end

main()
