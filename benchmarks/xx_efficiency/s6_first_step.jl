# benchmarks/xx_efficiency/s6_first_step.jl -- diagnostic for S6 (xs6 17582549): steps from the
# graded entangled references were exact to the input floor (~1e-12) at every h, while the L = 16
# domain-wall run to J T = 2 converged as h^4 globally (10/20/40/80 steps: 1.0e-5, 6.4e-7, 4.0e-8,
# 2.5e-9) with every step reporting tolerance_met at tol_basis 1e-10 -- i.e. a one-time error,
# presumably from the product-state start. Measured here, accuracy only:
#   A  the exact error after each of the first K steps from the domain wall, per h, with each step's
#      ledger (Test B/C/D indicators, frontier defect, widths, closure log, limits)
#   B  the first step's error against the order p = 1..4
#   C  the global run started from the graded J t = 0.5 reference instead of the product state
#
# Env (also common.jl): XXE_DTS (0.2,0.1,0.05,0.025), XXE_K (4)

include(joinpath(@__DIR__, "common.jl"))

const DTS = parse.(Float64, split(get(ENV, "XXE_DTS", "0.2,0.1,0.05,0.025"), ","))
const K = parse(Int, get(ENV, "XXE_K", "4"))
const TIGHT = (tol_basis = 1e-10, tol_trunc = 1e-12, tol_expv = 1e-13, maxdim = 512, order = 3)

basis = spin_xx_sector(L, 1)[2]
load_ref(t) = deserialize(joinpath(SNAP, "ref_L$(L)_t$(t).jls"))
slopes(hs, es) = [log(es[k] / es[k + 1]) / log(hs[k] / hs[k + 1]) for k in 1:(length(hs) - 1)]
ledger_row(info) = (lg = info.ledger;
    (met = lg.tolerance_met, unmet = string(lg.unmet), limits = string(info.record.limits),
     max_sample = lg.max_sample, max_tail = lg.max_tail, max_krylov_c = lg.max_krylov_c,
     max_krylov_d = lg.max_krylov_d, defect = lg.defect, leak = string(lg.leak),
     expv_err = lg.expv_err, proj_err = lg.proj_err, n_h1 = lg.n_h1, n_selected = lg.n_selected,
     n_krylov = lg.n_krylov, r_aug = lg.r_aug, r_final = lg.r_final,
     closure = string([(e.frontier, e.added, round(e.leak; sigdigits = 3)) for e in lg.closure_log
                       if get(e, :enrich, false)])))

out = Dict{String, Any}()
rowsA = Any[]
for h in DTS
    psi = domain_wall_state(L)
    eprev = 0.0
    for k in 1:K
        info = cbe_bug_step!(psi, MPO_XX, ComplexF64(-h * im); TIGHT...,
                             rng = MersenneTwister(SEEDS[1] + k))
        e = xx_state_error(psi, k * h, basis).err
        r = merge((h = h, step = k, err = e, increment = e - eprev, bonds = string(bond_dims(psi))),
                  ledger_row(info))
        push!(rowsA, r)
        eprev = e
        @printf("XXE-FS-A h=%.3f step=%d err=%.3e incr=%.3e met=%s Bmax=%.1e Cmax=%.1e Dmax=%.1e defect=%.1e r_aug=%d r_final=%d closure=%s\n",
                h, k, e, r.increment, r.met, r.max_sample, r.max_krylov_c, r.max_krylov_d, r.defect,
                r.r_aug, r.r_final, r.closure)
    end
end
out["A"] = rowsA
flush(stdout)

rowsB = Any[]
for p in 1:4
    es = Float64[]
    for h in DTS
        psi = domain_wall_state(L)
        info = cbe_bug_step!(psi, MPO_XX, ComplexF64(-h * im); merge(TIGHT, (order = p,))...,
                             rng = MersenneTwister(SEEDS[1]))
        e = xx_state_error(psi, h, basis).err
        push!(es, e)
        push!(rowsB, merge((order = p, h = h, err = e), ledger_row(info)))
    end
    @printf("XXE-FS-B order=%d first-step errors %s slopes %s (local target %d)\n", p,
            string(round.(es; sigdigits = 3)), string(round.(slopes(DTS, es); digits = 2)), p + 1)
end
out["B"] = rowsB
flush(stdout)

rowsC = Any[]
es = Float64[]
for h in DTS
    psi = load_ref(0.5)
    n = round(Int, 1.5 / h)
    for k in 1:n
        cbe_bug_step!(psi, MPO_XX, ComplexF64(-h * im); TIGHT..., rng = MersenneTwister(SEEDS[1] + k))
    end
    e = xx_state_error(psi, 0.5 + n * h, basis).err
    push!(es, e); push!(rowsC, (h = h, nstep = n, err = e))
    @printf("XXE-FS-C from ref t=0.5, %d steps of h=%.3f to t=2: err=%.3e\n", n, h, e)
end
@printf("XXE-FS-C slopes %s\n", string(round.(slopes(DTS, es); digits = 2)))
out["C"] = rowsC
open(joinpath(OUT, "s6_first_step.json"), "w") do io
    JSON.json(io, out; allownan = true)
end
println("XXE-FS-DONE")
