# HOW LOOSE CAN THE ADAPTIVE CBE-DMRG TOLERANCES BE? -- the ground-state twin of
# `benchmarks/adaptive_xx_l20.jl` (ADAPTIVE_XX_L20_RESULTS.md), simulation_plan_2 Task 4.
#
# Each model at N = 20 is scored against its EXACT ground energy (sparse Lanczos in the S^z = 0
# sector, C(20,10) = 184756 states; `pairs_sparse` / `heisenberg_sparse` read the same couplings as
# the MPO). Per run: energy error of the returned STATE (after its truncation), exact variance
# `mpo_variance`, sweeps to convergence, matvecs, wall time, ranks, and the last sweep's ledger.
#
# Studies (one process per model, so wall-clock ratios inside a model are comparable):
#   basis   tol_basis 1e-1 .. 1e-6 at tol_trunc = 1e-8           -- the expansion knob
#   trunc   tol_trunc 1e-2 .. 1e-8 at tol_basis = 1e-5           -- the split knob
#   combo   both loose together (do they compound?)
#   exact   exact = true at two settings (RSVD vs the full local basis)
#   legacy  the fixed-knob DMRG (`trunc_thresh = 1e-10`) as the cost baseline
#
# usage:  julia -t 2 --project=. benchmarks/ground_state/adaptive_dmrg_tolerance.jl MODEL OUTDIR [smoke]
#   MODEL = heis | xx | square | j1j2   (smoke: L = 10 chain / 4x3 cylinder, two settings)
# Output: OUTDIR/dmrgtol_<model>.csv, one row per run, streamed.

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))

const MODEL  = isempty(ARGS) ? "heis" : ARGS[1]
const OUTDIR = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results")
const SMOKE  = length(ARGS) >= 3 && ARGS[3] == "smoke"
const NSW    = 40
const ETOL   = 1e-10

function chain_couplings(L)
    Jm = zeros(L, L)
    for i in 1:(L - 1)
        Jm[i, i + 1] = 1.0
    end
    return Jm
end

"(N, MPO, exact sparse H, maxdim) for the model; one coupling source for MPO and ED."
function build(model)
    if model in ("heis", "xx")
        L = SMOKE ? 10 : 20
        delta = model == "heis" ? 1.0 : 0.0
        H, _, _ = heisenberg_sparse(L; J = 1.0, delta = delta)
        return (N = L, W = xxz_mpo(L; J = 1.0, delta = delta), H = H, maxdim = 256)
    elseif model in ("square", "j1j2")
        Lx, Ly = SMOKE ? (4, 3) : (5, 4)
        J2 = model == "j1j2" ? 0.5 : 0.0
        Jm = square_cylinder_couplings(Lx, Ly; J2 = J2)
        H, _, _ = pairs_sparse(Lx * Ly, Jm)
        return (N = Lx * Ly, W = square_cylinder_mpo(Lx, Ly; J2 = J2), H = H, maxdim = 512)
    end
    error("unknown model $model (heis | xx | square | j1j2)")
end

const COLS = ["model", "study", "tol_basis", "tol_trunc", "exact", "maxdim", "sweeps", "converged",
              "E", "dE", "dE_site", "variance", "chi", "r_aug_max", "matvec", "probes",
              "selected", "wall", "status_last", "unmet_last", "maxdim_reached",
              "testB_last", "testA_last", "discarded_last", "dE_ritz"]

_f(x) = x isa AbstractFloat ? @sprintf("%.6e", x) : string(x)

function run_one(io, S, E0, study; tb = nothing, tt = nothing, exact = false, legacy = false)
    psi = neel_state(S.N)
    kw = legacy ? (maxdim = S.maxdim, trunc_thresh = 1e-10) :
                  (maxdim = S.maxdim, tol_basis = tb, tol_trunc = tt, exact = exact)
    r = dmrg_cbe1s!(psi, S.W; n_sweeps = NSW, etol = ETOL, kw...)
    n2 = norm(psi)^2
    E = real(mpo_energy(psi, S.W)) / n2
    var = mpo_variance(psi, S.W)
    lg = r.ledgers[end]
    lgs = filter(!isnothing, r.ledgers)
    row = Any[MODEL, study, legacy ? "legacy" : tb, legacy ? "legacy" : tt, exact, S.maxdim,
              length(r.energies), r.converged, E, E - E0, (E - E0) / S.N, var,
              maximum(r.max_state_bond_dims), maximum(r.max_expanded), sum(r.krylov_dims),
              sum((l.n_probes for l in lgs); init = 0), sum((l.n_selected for l in lgs); init = 0),
              sum(r.sweep_seconds), r.status[end],
              lg === nothing ? "" : join(lg.unmet, "+"),
              lg === nothing ? false : lg.maxdim_reached,
              lg === nothing ? NaN : lg.max_sample, lg === nothing ? NaN : lg.max_tail,
              lg === nothing ? NaN : lg.discarded, r.energies[end] - E0]
    println(io, join(_f.(row), ","))
    flush(io)
    @printf("  %-6s tb=%-8s tt=%-8s exact=%-5s  dE/N=%.2e var=%.2e  sweeps=%2d chi=%4d matvec=%8d  %7.1fs  %s\n",
            study, string(legacy ? "legacy" : tb), string(legacy ? "legacy" : tt), string(exact),
            (E - E0) / S.N, var, length(r.energies), maximum(r.max_state_bond_dims),
            sum(r.krylov_dims), sum(r.sweep_seconds), string(r.status[end]))
    flush(stdout)
    return nothing
end

function main()
    mkpath(OUTDIR)
    set_symmetry!(:U1)
    S = build(MODEL)
    E0, _, res, E1, _ = sparse_ground(S.H, size(S.H, 1))
    @printf("model %s  N=%d  maxdim=%d  exact E0=%.12f (resid %.1e, E1~%.6f)  threads=%d\n",
            MODEL, S.N, S.maxdim, E0, res, E1, Threads.nthreads())
    res < 1e-8 || error("exact reference unconverged (resid $res)")

    # JIT warm-up on a tiny chain, never inside a timed run.
    let w = xxz_mpo(6; J = 1.0, delta = 1.0)
        dmrg_cbe1s!(neel_state(6), w; n_sweeps = 2, tol_basis = 1e-3, tol_trunc = 1e-4, maxdim = 8)
        dmrg_cbe1s!(neel_state(6), w; n_sweeps = 2, tol_basis = 1e-3, tol_trunc = 1e-4, maxdim = 8,
                    exact = true)
        dmrg_cbe1s!(neel_state(6), w; n_sweeps = 2, maxdim = 8, trunc_thresh = 1e-10)
        mpo_variance(neel_state(6), w)
    end

    io = open(joinpath(OUTDIR, "dmrgtol_$(MODEL)$(SMOKE ? "_smoke" : "").csv"), "w")
    println(io, join(COLS, ","))
    if SMOKE
        run_one(io, S, E0, "basis"; tb = 1e-3, tt = 1e-4)
        run_one(io, S, E0, "legacy"; legacy = true)
    else
        run_one(io, S, E0, "legacy"; legacy = true)
        for tb in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6)
            run_one(io, S, E0, "basis"; tb = tb, tt = 1e-8)
        end
        for tt in (1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8)
            run_one(io, S, E0, "trunc"; tb = 1e-5, tt = tt)
        end
        for (tb, tt) in ((1e-2, 1e-3), (1e-2, 1e-4), (1e-3, 1e-3), (1e-3, 1e-4), (1e-3, 1e-5),
                         (1e-4, 1e-5))
            run_one(io, S, E0, "combo"; tb = tb, tt = tt)
        end
        for (tb, tt) in ((1e-3, 1e-4), (1e-5, 1e-8))
            run_one(io, S, E0, "exact"; tb = tb, tt = tt, exact = true)
        end
    end
    close(io)
    println("GRADED-RUNS-DONE")
end

main()
