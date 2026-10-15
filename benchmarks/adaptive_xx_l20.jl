# XX chain, U(1), L = 20 -- the adaptive CBE-Krylov benchmark (adaptive plan §8, simulation_plan
# §4; Stage 6). Every row is graded against the EXACT free-fermion occupations (an `L x L`
# single-particle propagator, `tests/common/free_fermion.jl`), never against another integrator.
#
# PHASES (first argument):
#
#   profiles   IT01-IT04, IT06: {loose, tight} x {ordinary, midpoint} x {Neel, domain wall}
#              x 5 RNG seeds, plus the all-up no-growth control.
#   sweeps     IT07 and the question this campaign exists for -- HOW LOOSE CAN EACH TOLERANCE
#              BE? One tolerance is swept over several decades with the other two held tight,
#              so each sweep isolates ONE component of the step:
#                basis   tol_basis: the RSVD sampling/retention (Tests A/B) and the K/L Krylov
#                        enrichment (Tests C/D) -- the TRIAL SPACE
#                exact   the same tol_basis sweep with `exact = true` (complete local basis, no
#                        random sketch): the gap to `basis` is what the RANDOMISATION costs
#                trunc   tol_trunc: the final compression of the evolved state
#                expv    tol_expv: the root Galerkin solve
#                mid     the basis sweep under the MIDPOINT BUG, the other integrator
#   baseline   IT05: the fixed-knob legacy solver at several settings, for matched-accuracy cost.
#   combos     the sweeps vary ONE tolerance; this checks that loose settings do not COMPOUND:
#              a grid of tol_basis x tol_trunc x tol_expv, ordinary and midpoint, Neel.
#   maxdim     IT07 "failures from maxdim are visible": tight tolerances at small caps must end
#              in `:budget_unmet` (`:trunc`, `maxdim_reached`), never a silent pass.
#   matched    IT05 / simulation_plan §6 acceptance, IN ONE PROCESS so wall clocks compare: the
#              legacy fixed-knob arms next to cheap adaptive settings (RSVD and exact CBE), both
#              schemes, Neel and DW; compare H applications, trial-space width, rank and time at
#              matched error.
#   rsvdscan   WHERE DOES THE RANDOM SKETCH PAY? RSVD vs exact CBE (`exact = true`) at maxdim
#              256 and 512, Neel, in ONE process so `t_cbe` per step compares as chi grows.
#   all        profiles + sweeps + baseline.
#
# usage:  julia -t 2 --project=. benchmarks/adaptive_xx_l20.jl PHASE [OUTDIR] [L] [T] [dt]
#
# Output: one streamed CSV row per (run, step) in OUTDIR/adaptive_xx_<phase>.csv, so a killed job
# keeps everything up to the kill. The script exits non-zero if it graded nothing.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const PHASE  = isempty(ARGS) ? "all" : ARGS[1]
const OUTDIR = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results")
const L      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
const TMAX   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.5
const DT     = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.05
const NSTEP  = round(Int, TMAX / DT)
const MAXDIM = 128

PHASE in ("profiles", "sweeps", "baseline", "combos", "maxdim", "matched", "rsvdscan", "all") ||
    error("unknown phase $PHASE (profiles | sweeps | baseline | combos | maxdim | matched | rsvdscan | all)")

const PROFILES = Dict("loose" => (1e-3, 1e-5, 1e-8), "tight" => (1e-5, 1e-7, 1e-10))

"Initial state and its occupied (up) sites, for the free-fermion reference."
function initial(state::String)
    state == "neel"   && return (neel_state(L), collect(1:2:L))
    state == "dw"     && return (domain_wall_state(L), collect(1:(L ÷ 2)))
    state == "allup"  && return (product_state(fill(:up, L)), collect(1:L))
    error("unknown state $state")
end

const COLS = ["phase", "arm", "state", "scheme", "seed", "path", "exact",
              "tol_basis", "tol_trunc", "tol_expv", "maxdim", "step", "t",
              "err_sz", "err_sz_1step", "norm_drift", "sz_drift", "chi_max",
              "r_old", "r_aug", "r_final", "krylov_dims", "n_h1", "n_probes", "n_holdout",
              "n_selected", "n_krylov", "n_c_pass", "n_d_after_c", "max_sample", "max_tail",
              "max_krylov_c", "max_krylov_d", "discarded_cum", "discarded_max", "proj_err",
              "expv_err", "expv_iters", "peak_elements", "status", "unmet", "workcap",
              "maxdim_reached", "t_step", "t_cbe", "t_kry", "t_root", "t_trunc"]

_f(x) = x === nothing ? "" : x isa AbstractFloat ? @sprintf("%.6e", x) : string(x)

"""
    run_arm(io, phase, arm, state, scheme; seed, kw...) -> Int

Evolve `NSTEP` steps and write one CSV row per step. `kw` goes to the step VERBATIM, so the
knob under test is exactly what the integrator received (and is echoed in the row).
"""
function run_arm(io, phase, arm, state, scheme; seed::Int, legacy::Bool = false, kw...)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi, occ = initial(state)
    rng = Random.MersenneTwister(seed)
    step! = scheme == "midpoint" ? cbe_bug_midpoint_step! : cbe_bug_step!
    sz(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())
    sz0 = sum(sz(psi))
    nkw = values(kw)
    tb = get(nkw, :tol_basis, nothing); tt = get(nkw, :tol_trunc, nothing)
    te = get(nkw, :tol_expv, nothing)
    ex = get(nkw, :exact, false)
    graded = 0
    err1 = errl = NaN
    for n in 1:NSTEP
        info = step!(psi, mpo, ComplexF64(-im * DT); parallel = true, rng = rng, kw...)
        lg = info.ledger
        # THE PATH THE CALL TOOK MUST BE THE ONE THE ARM CLAIMS -- a legacy knob slipping into an
        # adaptive arm (or the reverse) would silently grade the wrong solver.
        (lg === nothing) == legacy ||
            error("arm $arm expected $(legacy ? "legacy" : "adaptive") path, got the other")
        t = n * DT
        err = maximum(abs.(sz(psi) .- xx_free_fermion_sz(L, t; J = 1.0, occupied = occ)))
        n == 1 && (err1 = err)
        errl = err
        g(f) = lg === nothing ? nothing : getfield(lg, f)
        # The trial-space width exists on BOTH paths (the half-sweep frames); the ledger only
        # carries it on the adaptive one, so the legacy arms read it off the step info.
        raug = lg === nothing ? max(maximum(info.expanded; init = 0), info.root_right) : lg.r_aug
        row = Any[phase, arm, state, scheme, seed, legacy ? "legacy" : "adaptive", ex,
                  tb, tt, te, get(nkw, :maxdim, MAXDIM), n, t,
                  err, err1, norm(psi) - 1, sum(sz(psi)) - sz0, maximum(bond_dims(psi); init = 0),
                  g(:r_old), raug, g(:r_final), info.krylov_dims, g(:n_h1), g(:n_probes),
                  g(:n_holdout), g(:n_selected), g(:n_krylov), g(:n_c_pass), g(:n_d_after_c),
                  g(:max_sample), g(:max_tail), g(:max_krylov_c), g(:max_krylov_d),
                  g(:discarded), info.discarded, g(:proj_err), g(:expv_err), g(:expv_iters),
                  info.peak_elements, budget_status(lg),
                  lg === nothing ? "" : join(lg.unmet, "+"),
                  g(:workcap_reached), g(:maxdim_reached),
                  info.t_step, info.t_cbe, info.t_kry, info.t_root, info.t_trunc]
        println(io, join(_f.(row), ","))
        flush(io)
        graded += 1
    end
    @info "$phase $arm $state $scheme seed=$seed tb=$tb tt=$tt te=$te exact=$ex: " *
          "err(1 step) $err1, err(T) $errl, chi $(maximum(bond_dims(psi); init = 0))"
    return graded
end

function main()
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, "adaptive_xx_$(PHASE)_L$(L).csv")
    graded = 0
    open(path, "w") do io
        println(io, join(COLS, ","))
        go(args...; kw...) = (graded += run_arm(io, args...; kw...))
        phases = PHASE == "all" ? ("profiles", "sweeps", "baseline") : (PHASE,)

        if "profiles" in phases
            for (pname, (tb, tt, te)) in sort(collect(PROFILES); by = first),
                scheme in ("ordinary", "midpoint"), state in ("neel", "dw"), seed in 1:5
                go("profiles", pname, state, scheme; seed = seed,
                   tol_basis = tb, tol_trunc = tt, tol_expv = te, maxdim = MAXDIM)
            end
            for scheme in ("ordinary", "midpoint")          # IT04: H annihilates all-up
                go("profiles", "tight", "allup", scheme; seed = 1, tol_basis = 1e-5,
                   tol_trunc = 1e-7, tol_expv = 1e-10, maxdim = MAXDIM)
            end
        end

        if "sweeps" in phases
            for tb in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
                go("sweeps", "basis", "neel", "ordinary"; seed = 1, tol_basis = tb,
                   tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = MAXDIM)
                go("sweeps", "exact", "neel", "ordinary"; seed = 1, tol_basis = tb,
                   tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = MAXDIM, exact = true)
                go("sweeps", "mid", "neel", "midpoint"; seed = 1, tol_basis = tb,
                   tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = MAXDIM)
            end
            for tt in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
                go("sweeps", "trunc", "neel", "ordinary"; seed = 1, tol_basis = 1e-7,
                   tol_trunc = tt, tol_expv = 1e-12, maxdim = MAXDIM)
            end
            for te in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
                go("sweeps", "expv", "neel", "ordinary"; seed = 1, tol_basis = 1e-7,
                   tol_trunc = 1e-10, tol_expv = te, maxdim = MAXDIM)
            end
        end

        if "combos" in phases
            for tb in (1e-3, 1e-4, 1e-5), tt in (1e-3, 1e-5), te in (1e-2, 1e-6),
                scheme in ("ordinary", "midpoint")
                go("combos", "combo", "neel", scheme; seed = 1, tol_basis = tb,
                   tol_trunc = tt, tol_expv = te, maxdim = MAXDIM)
            end
        end

        if "maxdim" in phases
            for md in (4, 8, 16), scheme in ("ordinary", "midpoint")
                go("maxdim", "cap", "neel", scheme; seed = 1, tol_basis = 1e-6,
                   tol_trunc = 1e-8, tol_expv = 1e-10, maxdim = md)
            end
        end

        if "matched" in phases
            for state in ("neel", "dw"), scheme in ("ordinary", "midpoint")
                go("matched", "legacy_default", state, scheme; seed = 1, legacy = true,
                   trunc_thresh = 1e-12, maxdim = MAXDIM)
                go("matched", "legacy_kry3", state, scheme; seed = 1, legacy = true,
                   trunc_thresh = 1e-10, krylov_basis = 3, krylov_tol = 0.0, maxdim = MAXDIM)
                for tb in (1e-4, 1e-5, 1e-6), ex in (false, true)
                    go("matched", ex ? "adaptive_exact" : "adaptive", state, scheme; seed = 1,
                       tol_basis = tb, tol_trunc = 1e-4, tol_expv = 1e-2, maxdim = MAXDIM,
                       exact = ex)
                end
            end
        end

        if "rsvdscan" in phases
            for md in (256, 512), ex in (false, true)
                go("rsvdscan", ex ? "exact" : "rsvd", "neel", "ordinary"; seed = 1,
                   tol_basis = 1e-5, tol_trunc = 1e-8, tol_expv = 1e-2, maxdim = md, exact = ex)
            end
        end

        if "baseline" in phases
            # The fixed-knob solver (IT05): its historical defaults, a looser cut, and the pinned
            # shallow Krylov frame. `trunc_thresh` is named in every arm, which is what selects
            # the legacy path.
            for state in ("neel", "dw"), scheme in ("ordinary", "midpoint")
                go("baseline", "legacy_default", state, scheme; seed = 1, legacy = true,
                   trunc_thresh = 1e-12, maxdim = MAXDIM)
                go("baseline", "legacy_cut1e-8", state, scheme; seed = 1, legacy = true,
                   trunc_thresh = 1e-8, maxdim = MAXDIM)
                go("baseline", "legacy_kry3", state, scheme; seed = 1, legacy = true,
                   trunc_thresh = 1e-10, krylov_basis = 3, krylov_tol = 0.0, maxdim = MAXDIM)
            end
        end
    end
    println("GRADED-ROWS=$graded FILE=$path")
    graded > 0 || (println("EXIT-GRADED-ZERO"); exit(1))
    return nothing
end

main()
