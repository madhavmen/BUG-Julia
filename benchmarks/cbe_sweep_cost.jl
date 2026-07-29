# SWEEP-LEVEL cost/accuracy of the S-step schemes. This is the benchmark that decides.
#
#     julia --project=. benchmarks/cbe_sweep_cost.jl
#
# Everything before this measured ONE bond of a warmed state against an exact rank-4
# propagator. That is a diagnostic, not a decision: it says nothing about bonds near the chain
# edge, nothing about how the schemes compound over a run, and -- because the reference itself
# dominates the runtime -- nothing about cost. This runs whole evolutions and reports the three
# quantities a scheme has to win on:
#
#   err     magnetisation error against 2-site TDVP at the same dt and maxdim
#   wall    seconds for the evolution, AFTER warm-up
#   matvec  total operator applications, immune to machine load
#   iters   mean growth passes per bond -- shows the early stop actually firing
#
# WARM-UP IS NOT OPTIONAL. Julia compiles on first call, and charging that to the first timed
# row is exactly the artefact that produced the retracted speedup claim (docs 7.5). Every
# scheme is run once untimed before any timing starts.
#
# Small sizes only, per the standing rule: L <= 12 locally; larger goes to the cluster.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf

set_symmetry!(:U1)

const L      = parse(Int, get(ENV, "L", "8"))
const DT     = 0.02
const TMAX   = 0.5
const NSTEP  = round(Int, TMAX / DT)
const MAXDIM = 32

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

ref = domain_wall_state(L)
let h = xxz_chain(L; delta = 0.0)
    for _ in 1:NSTEP
        tdvp2_step!(ref, h, ComplexF64(-im * DT); maxdim = MAXDIM, trunc_thresh = 1e-14)
    end
end
const REF_MAG = magnetisation(copy(ref))

function evolve(nsteps; kwargs...)
    psi = domain_wall_state(L)
    gates = xx_gates(psi)
    info = cbe_bond_update_bug!(psi, gates;
                                opts = CBEBugOptions(dt = DT, n_steps = nsteps,
                                                     maxdim = MAXDIM, trunc_thresh = 1e-14,
                                                     kwargs...))
    return psi, info
end

rows = Any[]
function bench(label; kwargs...)
    evolve(2; kwargs...)                                   # warm-up, untimed
    t = @elapsed ((psi, info) = evolve(NSTEP; kwargs...))
    err = norm(magnetisation(copy(psi)) - REF_MAG)
    mv = sum(info.krylov_dims)
    push!(rows, (label, err, t, mv, maximum(info.max_bond_dims)))
    @printf("%-32s %-11.3e %-8.2f %-9d %-7d %s\n", label, err, t, mv,
            maximum(info.max_bond_dims), string(info.bond_dims[end]))
end

@printf("L = %d, dt = %.3f, t = %.2f (%d steps), maxdim = %d\n", L, DT, TMAX, NSTEP, MAXDIM)
println("reference: 2-site TDVP, same dt and maxdim.  Timings are post-warm-up.\n")
@printf("%-32s %-11s %-8s %-9s %-7s %s\n",
        "scheme", "err", "wall(s)", "matvec", "maxbd", "profile")
println("-"^94)

bench("growth=2.0 one-shot (default)")
bench("growth=2.0 + reorth"; s_reorth = true)
bench("growth=1.25 one-shot"; growth = 1.25)
bench("growth=1.25 + reorth"; growth = 1.25, s_reorth = true)
bench("growth=1.25 outer s_iters=3"; growth = 1.25, s_iters = 3)
bench("growth=1.25 expanding grow=2"; growth = 1.25, s_iters = -2)
bench("growth=1.5  one-shot"; growth = 1.5)
bench("growth=1.5  expanding grow=2"; growth = 1.5, s_iters = -2)
bench("growth=1.5  expanding grow=8"; growth = 1.5, s_iters = -8)

println("-"^94)
base = rows[1]
println("\nrelative to the default (err x, wall x, matvec x):")
for r in rows
    @printf("  %-32s %6.2fx  %6.2fx  %6.2fx\n", r[1], r[2] / base[2], r[3] / base[3],
            r[4] / base[4])
end

println("\nSWEEP_COST_DONE")
