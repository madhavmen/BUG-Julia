# WHY IS THE FIRST IMAGINARY-TIME STEP FROM A PRODUCT STATE FLAGGED? (USER 2026-10-18: "this hints
# at a bug, fix it.") Diagnostic only.
#
# The flag is the outside-Galerkin defect `|tau| ||(I - P) H psi_1|| / ||psi_1||` above tol_basis.
# If the frames held what the step needs it would scale like tau^(p+1) (order p closes H^k psi up to
# k = p); a defect that survives every split means it does NOT shrink, i.e. the frames built from
# chi = 1 miss a direction. This measures the first step from Neel and from the domain wall at
# several dbeta / orders / tol_basis, RSVD and exact CBE, and the SECOND step (chi > 1) as the
# control: defect, its left/right parts, the frame widths per bond and the CBE/Krylov counters.
#
# Run: julia -t 4 --project=PKG benchmarks/imaginary_time/first_step_probe.jl
# Env: FP_L (10) FP_OUT (csv path)

using LinearAlgebra, Printf, Random
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

const L   = parse(Int, get(ENV, "FP_L", "10"))
const OUT = get(ENV, "FP_OUT", joinpath(@__DIR__, "results", "first_step_probe.csv"))

energy(p, W) = real(mpo_energy(p, W)) / real(overlap(p, p))
renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

function one_step(io, W, psi0, name, stepno, order, exact, tb, db)
    psi = copy(psi0)
    rep = Ref{Any}(nothing)
    info = cbe_bug_step!(psi, W, ComplexF64(-db); order = order, root_method = :exp,
                         eshift = energy(psi, W), tol_basis = tb, tol_trunc = 1e-10,
                         tol_expv = 1e-12, maxdim = 256, root_report = rep,
                         outside_residual = true, exact = exact, rng = MersenneTwister(1))
    renorm!(psi)
    r, lg = rep[], info.ledger
    pl = hasproperty(r, :per_left) ? join(round.(abs(db) .* r.per_left; sigdigits = 2), ";") : ""
    pr = hasproperty(r, :per_right) ? join(round.(abs(db) .* r.per_right; sigdigits = 2), ";") : ""
    @printf(io, "%s,%d,%d,%s,%.0e,%.6g,%s,%s,%.4e,%.4e,%.4e,%.3e,%d,%d,%s,%s,%d,%d,%.3e,%.3e,%s,%s\n",
            name, stepno, order, exact, tb, db, budget_status(lg), join(string.(lg.unmet), "+"),
            r.resid, abs(db) * r.outside_left, abs(db) * r.outside_right, lg.proj_err, lg.r_aug,
            maximum(bond_dims(psi)), join(info.expanded, ";"), join(info.n_new, ";"),
            lg.n_selected, lg.n_krylov, lg.max_krylov_c, lg.max_krylov_d, pl, pr)
    flush(io)
    @printf("%-5s step %d p%d exact=%-5s tb %.0e db %-8g resid %.3e (L %.2e R %.2e) r_aug %d chi %d %s %s\n",
            name, stepno, order, exact, tb, db, r.resid, abs(db) * r.outside_left,
            abs(db) * r.outside_right, lg.r_aug, maximum(bond_dims(psi)), budget_status(lg),
            join(string.(lg.unmet), "+"))
    flush(stdout)
    return psi
end

function main()
    set_symmetry!(:U1)
    mkpath(dirname(OUT))
    W = xxz_mpo(L; J = 1.0, delta = 1.0)
    io = open(OUT, "w")
    println(io, "state,step,order,exact,tol_basis,dbeta,status,unmet,resid,resid_left,resid_right," *
                "proj_err,r_aug,chi,expanded,n_new,n_selected,n_krylov,max_krylov_c,max_krylov_d," *
                "per_left,per_right")
    one_step(devnull, W, neel_state(L), "warm", 1, 2, false, 1e-6, 0.1)      # JIT, discarded
    for (name, st) in (("neel", neel_state(L)), ("dw", domain_wall_state(L)))
        for exact in (false, true), order in (1, 2), tb in (1e-4, 1e-6, 1e-8),
            db in (0.2, 0.1, 0.05, 0.025, 0.0125)
            one_step(io, W, st, name, 1, order, exact, tb, db)
        end
        # the control: the SECOND step, from the state one tight step grew (chi > 1)
        grown = one_step(io, W, st, name, 1, 2, true, 1e-10, 0.1)
        for exact in (false, true), order in (1, 2), tb in (1e-6, 1e-8),
            db in (0.2, 0.1, 0.05, 0.025, 0.0125)
            one_step(io, W, grown, name, 2, order, exact, tb, db)
        end
    end
    close(io)
    println("FP-DONE")
end

main()
