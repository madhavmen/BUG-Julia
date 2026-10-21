# benchmarks/hq_dd_gap_l8.jl -- why is :dd_qr_streamed less accurate than :svd_select at L = 8?
#
# hq12br 17409190, test_dd_cbe.jl "end to end": Heisenberg L = 8 from Neel, 4 x dt 0.05,
# tol_basis 1e-8 gave err(dd) = 1.5e-3 against err(svd) = 1.2e-4. Two readings:
#   (a) temporal: both are order 1 and DD has the larger constant (it grows one frame per
#       orientation, and its one-site Krylov runs through V0, not a CBE-widened V_ex), so the error
#       halves with dt and does not move with tol_basis;
#   (b) basis: DD misses directions, so the error does not follow dt and/or the ledger is silent.
# The grid separates them: three policies x dt x tol_basis, error vs exact at a fixed T, with the
# step records' ranks, QR / one-site counts and limit statuses.
#
# Env: HQ_L (8), HQ_T (0.2), HQ_DTS ("0.1,0.05,0.025"), HQ_TOLS ("1e-6,1e-8,1e-10").

using LinearAlgebra
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))

const L = parse(Int, get(ENV, "HQ_L", "8"))
const T = parse(Float64, get(ENV, "HQ_T", "0.2"))
const DTS = parse.(Float64, split(get(ENV, "HQ_DTS", "0.1,0.05,0.025"), ","))
const TOLS = parse.(Float64, split(get(ENV, "HQ_TOLS", "1e-6,1e-8,1e-10"), ","))

set_symmetry!(:U1)
const MPO_ = xxz_mpo(L; J = 1.0, delta = 1.0)
const V0 = dense_state(neel_state(L))
const VREF = dense_exact_propagate(dense_heisenberg(L; J = 1.0, delta = 1.0), V0, T)

println("HQGAP-HEADER policy dt tol_basis err maxD_stage maxD_final qr_rank n_qr n_one_site ",
        "n_root limits")
for tol in TOLS, dt in DTS, pol in CBE_POLICIES
    n = round(Int, T / dt)
    p = neel_state(L)
    dstage, qr_rank, n_qr, n_one, n_root = 0, 0, 0, 0, 0
    lim = Symbol[]
    for _ in 1:n
        r = cbe_bug_step!(p, MPO_, -im * dt; cbe_policy = pol, tol_basis = tol,
                          tol_trunc = 1e-12, tol_expv = 1e-12, maxdim = 64).record
        dstage = max(dstage, r.D_stage)
        qr_rank += r.counters.qr_rank
        n_qr += r.counters.n_qr
        n_one += r.counters.n_one_site
        n_root += r.counters.n_root
        union!(lim, r.limits)
    end
    v = dense_state(p)
    err = sqrt(max(2 * (1 - abs(dot(v, VREF)) / (norm(v) * norm(VREF))), 0.0))
    println("HQGAP ", pol, " ", dt, " ", tol, " ", err, " ", dstage, " ",
            maximum(bond_dims(p); init = 0), " ", qr_rank, " ", n_qr, " ", n_one, " ", n_root,
            " ", isempty(lim) ? "none" : join(lim, ","))
    flush(stdout)
end
