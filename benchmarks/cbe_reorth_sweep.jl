# Does full reorthogonalisation in the S-step pay across a whole evolution?
#
#     julia --project=. benchmarks/cbe_reorth_sweep.jl
#
# `benchmarks/cbe_s_iterations.jl` measured a ~5.7x matvec drop at ONE bond of a warmed state.
# One bond is not a sweep: the saving depends on how close the Galerkin generator is to exact
# there, and a bond whose basis is starved may behave the other way. This runs the full driver
# both ways and compares the quantities that would justify changing a default:
#
#   err      profile error against 2-site TDVP at the final time (the accuracy that must not move)
#   matvec   TOTAL operator applications over the run (the cost that should fall)
#   maxbd    final max bond dimension (must not change -- reorth is a solver detail, not a
#            basis change; if this moves, something other than orthogonality is going on)
#
# Small sizes only, per the standing rule: L <= 12 locally, anything larger goes to the cluster.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf

set_symmetry!(:U1)

const L     = 8
const DT    = 0.02
const TMAX  = 0.5
const NSTEP = round(Int, TMAX / DT)
const MAXDIM = 32

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

# Reference: 2-site TDVP at the same dt, full rank.
ref = domain_wall_state(L)
let h = xxz_chain(L; jz = 0.0)
    for _ in 1:NSTEP
        tdvp2_step!(ref, h, ComplexF64(-im * DT); maxdim = MAXDIM, trunc_thresh = 1e-14)
    end
end
ref_mag = magnetisation(copy(ref))

function run(label; kwargs...)
    psi = domain_wall_state(L)
    gates = xx_gates(psi)
    info = cbe_bond_update_bug!(psi, gates;
                                opts = CBEBugOptions(dt = DT, n_steps = NSTEP,
                                                     maxdim = MAXDIM,
                                                     trunc_thresh = 1e-14,
                                                     record_magnetisation = true,
                                                     kwargs...))
    err = norm(magnetisation(copy(psi)) - ref_mag)
    mv = sum(info.krylov_dims)
    @printf("%-30s %-11.3e %-9d %-7d %s\n", label, err, mv,
            maximum(info.max_bond), string(info.bond_dims[end]))
    return (err, mv)
end

@printf("L = %d, dt = %.3f, t = %.2f (%d steps), maxdim = %d\n",
        L, DT, TMAX, NSTEP, MAXDIM)
@printf("reference: 2-site TDVP, same dt and maxdim\n\n")
@printf("%-30s %-11s %-9s %-7s %s\n", "setting", "err vs tdvp2", "matvec", "maxbd", "profile")
println("-"^86)

e0, m0 = run("reorth = false (default)")
e1, m1 = run("reorth = true"; s_reorth = true)

println("-"^86)
@printf("matvec ratio  reorth/plain = %.3f      err ratio = %.3f\n", m1 / m0, e1 / e0)

# The expanding-basis solver over a whole run, for completeness. `s_iters = -g` grows the basis
# around the first g Krylov vectors of every bond's exponential.
println()
for g in (1, 2)
    run("expanding lanczos grow=$g"; s_iters = -g)
end

println("\nREORTH_DONE")
