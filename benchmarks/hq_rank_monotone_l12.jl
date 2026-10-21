# benchmarks/hq_rank_monotone_l12.jl -- does a looser tol_basis buy MORE rank on the DD path?
#
# hq13a 17418200, test_adaptive_rsvd.jl "a tighter basis tolerance never buys less rank": XX L = 12
# from Neel, 4 x dt 0.05, tol_trunc 1e-10, maxdim 64, parallel half-sweeps; the final max bond was
# 14 at tol_basis 1e-8 and 15 at 1e-3 (the default policy is :dd_qr_streamed since T1-2). Two
# readings:
#   (a) the FRAMES are monotone (D_stage grows as tol_basis tightens) and the extra final rank at
#       1e-3 is the error of the looser step kept above tol_trunc by the compression;
#   (b) the frames themselves shrink as tol_basis tightens -- a controller defect.
# Per step: D_stage, D_final, and at the end the error vs exact CBE on the same time grid (so the
# BASIS error alone, which is what tol_basis buys), for every policy and tolerance. Not the dense
# exp(-iHt) reference: at L = 12 building it ran 40 min without finishing (17418617).
#
# Env: HQ_L (12), HQ_NSTEP (4), HQ_DT (0.05), HQ_TOLS ("1e-3,1e-5,1e-8").

using LinearAlgebra
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

const L = parse(Int, get(ENV, "HQ_L", "12"))
const NSTEP = parse(Int, get(ENV, "HQ_NSTEP", "4"))
const DT = parse(Float64, get(ENV, "HQ_DT", "0.05"))
const TOLS = parse.(Float64, split(get(ENV, "HQ_TOLS", "1e-3,1e-5,1e-8"), ","))

set_symmetry!(:U1)
const MPO_ = xxz_mpo(L; J = 1.0, delta = 0.0)
const STEPKW = (tol_trunc = 1e-10, tol_expv = 1e-10, maxdim = 64, parallel = true)
const PREF = let p = neel_state(L)
    for _ in 1:NSTEP
        cbe_bug_step!(p, MPO_, ComplexF64(-im * DT); exact = true, tol_basis = 1e-12, STEPKW...)
    end
    p
end

println("HQRANK-HEADER policy tol_basis err D_stage_per_step D_final_per_step final_bonds limits")
println("HQRANK-REF exact ", join(bond_dims(PREF), ","))
flush(stdout)
for pol in CBE_POLICIES, tol in TOLS
    p = neel_state(L)
    ds, df = Int[], Int[]
    lim = Symbol[]
    for _ in 1:NSTEP
        r = cbe_bug_step!(p, MPO_, ComplexF64(-im * DT); cbe_policy = pol, tol_basis = tol,
                          STEPKW...).record
        push!(ds, r.D_stage)
        push!(df, r.D_final)
        union!(lim, r.limits)
    end
    err = sqrt(max(2 * (1 - abs(overlap(p, PREF)) / (norm(p) * norm(PREF))), 0.0))
    println("HQRANK ", pol, " ", tol, " ", err, " ", join(ds, ","), " ", join(df, ","), " ",
            join(bond_dims(p), ","), " ", isempty(lim) ? "none" : join(lim, ","))
    flush(stdout)
end
