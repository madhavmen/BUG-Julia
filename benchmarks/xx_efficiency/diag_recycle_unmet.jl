# benchmarks/xx_efficiency/diag_recycle_unmet.jl -- diagnostic for the Fig. 4 full arm (xf4
# 17583862): with `krylov_recycle = true` (`:images`) 3 of the 10 dt = 0.2 steps reported
# `tolerance_met = false`, fresh Krylov none, the final error identical. Accuracy only: the Fig. 4
# run at one dt for fresh / `:images` / `:frame`, each step's ledger (which checks failed, Test C/D
# indicators, work ceiling) and every recycled bond visit that was not met.
#
# Env (also common.jl): XXE_DTS (0.2), XXE_T (2.0), XXE_TOLS (1e-7,1e-10,1e-12)

include(joinpath(@__DIR__, "common.jl"))

const DT = parse(Float64, get(ENV, "XXE_DTS", "0.2"))
const TEND = parse(Float64, get(ENV, "XXE_T", "2.0"))
const TB, TT, TE = parse.(Float64, split(get(ENV, "XXE_TOLS", "1e-7,1e-10,1e-12"), ","))
const ARMS = (("fresh", (krylov_recycle = false,)),
              ("images", (krylov_recycle = true, recycle_mode = :images)),
              ("frame", (krylov_recycle = true, recycle_mode = :frame)))

basis = spin_xx_sector(L, 1)[2]
for (name, kw) in ARMS
    psi = domain_wall_state(L)
    for k in 1:round(Int, TEND / DT)
        rep = Ref{Any}(nothing)
        info = cbe_bug_step!(psi, MPO_XX, ComplexF64(-DT * im); tol_basis = TB, tol_trunc = TT,
                             tol_expv = TE, maxdim = 512, order = 3, kw...,
                             recycle_report = rep, rng = MersenneTwister(SEEDS[1] + k))
        lg = info.ledger
        e = xx_state_error(psi, k * DT, basis).err
        @printf("XXE-RU arm=%s step=%d err=%.3e met=%s unmet=%s Cmax=%.2e Dmax=%.2e workcap=%s n_h1=%d bonds=%s\n",
                name, k, e, lg.tolerance_met, string(lg.unmet), lg.max_krylov_c, lg.max_krylov_d,
                lg.workcap_reached, lg.n_h1, string(bond_dims(psi)))
        rep[] === nothing && continue
        for side in (:left, :right), v in rep[][side]
            get(v, :met, true) && continue
            println("XXE-RU-VISIT arm=$name step=$k side=$side ", v)
        end
    end
end
println("XXE-RU-DONE")
