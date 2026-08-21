# WHAT THE RSVD PARAMETERS ACTUALLY DO TO ERROR AND COST, at L = 16, on OAT and XX.
#
# ⛔ THE TWO SCHEMES SHIP WITH DIFFERENT SKETCH DEFAULTS, AND THAT INVALIDATES EVERY COMPARISON
# MADE AT DEFAULTS. Measured from the signatures:
#
#     tdvp_cbe1s_step!   growth = 2.0   comp_ratio = 0.5
#     cbe_bug_step!      growth = 1.1   comp_ratio = 1.0
#
# `growth` sets the expansion budget as `ceil(growth*dmax) - r` (cbe_core.jl:568), so at a bond
# sitting at the neighbourhood max, 1.1 admits ONE direction per step (2 once r >= 10) while 2.0
# admits r of them. Comparing the two at their defaults compares a STARVED arm against a fed one
# and attributes the difference to the integrator. Every arm below therefore states its budget
# EXPLICITLY, and both schemes are swept over the SAME grid.
#
# THREE PHASES, in this order, because each one decides whether the next is worth running.
#
#   0  SATURATION GATE   `exact=true` (the exact SVD) against `exact=false` (the sketch), same
#      seed, everything else fixed. If they AGREE the sketch already captures the whole relevant
#      subspace -- it is SATURATED -- and no sampling knob can matter, because a different draw
#      is just a different basis of the same space. Scanning `dover`/`comp_ratio`/seed on a
#      saturated model measures the RNG, not the integrator. MEASURED already on OAT at L=10:
#      saturated at three ranks. This asks whether that survives to L=16 and whether XX, whose
#      rank actually grows, is saturated at all.
#
#   1  NOISE FLOOR       the same arm over several seeds. THE SEED SPREAD IS THE RESOLUTION OF
#      EVERY OTHER NUMBER IN THIS FILE: an effect smaller than it is not an effect. Running the
#      budget scan without this is how a tuning artifact gets published.
#
#   2  BUDGET SCAN       `dex` / `growth`, both schemes, one grid.
#
#      ⛔ `maxdim` IS DELIBERATELY TIGHT ENOUGH THAT EVERY ARM SATURATES IT. Left generous, a
#      bigger `dex` simply reaches a bigger rank and the scan measures "more rank is better",
#      which is trivially true and says nothing about the sketch. Pinned at the cap, every arm
#      ends at the SAME chi and `dex` controls only WHICH directions survive -- the best of 2
#      candidates per bond per step against the best of 16. That is search quality at fixed
#      rank, which is the question.
#
# THE OUTPUT IS AN ERROR-VS-COST PARETO FRONT, NOT ERROR VS PARAMETER. "Does dex=16 lower the
# error" is trivially yes; "does it lower the error per unit of work, better than tdvp_cbe1s
# does" is the claim `cbe_bug` actually rests on, and only a cost axis can answer it.
#
# ⛔ TWO COST AXES, AND THE DISAGREEMENT BETWEEN THEM IS THE SIGNAL. `lanczos_expv` exits on
# BREAKDOWN only, so every solve burns `maxiter` and wall time is very nearly proportional to
# Krylov count -- normally it adds noise without information. That argument FAILS for these
# parameters: the sketch's cost is QR and matmul, not Krylov, so `dex`/`dover` can move seconds
# without moving `krylov`. When the two axes diverge, the sketch itself has become the
# bottleneck. Both are recorded.
#
# ⛔ WALL TIME FROM A LOCAL RUN IS NOT DATA. Use `submit_rsvd_scan.slurm`: one job at a time,
# single-threaded. Locally this file is for the ERROR columns only, and `phase=0` at a short
# `t_over_pi` is the part worth running interactively.
#
#   julia --project=. benchmarks/rsvd_param_scan.jl phase=0 model=xx
#   julia --project=. benchmarks/rsvd_param_scan.jl phase=2 model=oat t_over_pi=1.0
#   sbatch benchmarks/submit_rsvd_scan.slurm

using LinearAlgebra, Printf, Random
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const HERE    = dirname(abspath(@__FILE__))
const RESULTS = joinpath(HERE, "results")

# ── parameters ────────────────────────────────────────────────────────────────────────────
function parse_params(defaults::NamedTuple)
    vals = Dict{Symbol, Any}(pairs(defaults))
    for a in ARGS
        occursin('=', a) || error("argument $(repr(a)) is not name=value")
        name, val = split(a, '='; limit = 2)
        key = Symbol(strip(name))
        haskey(vals, key) || error("unknown parameter $(repr(String(key))). Valid: " *
                                   join(sort(String.(collect(keys(defaults)))), ", "))
        d = defaults[key]
        vals[key] = d isa Int ? parse(Int, val) : d isa Float64 ? parse(Float64, val) : String(val)
    end
    return NamedTuple(vals)
end

const P = parse_params((
    model     = "oat",   # "oat" | "xx"
    L         = 16,
    phase     = 0,       # 0 = saturation gate, 1 = noise floor, 2 = budget scan
    t_over_pi = 1.0,     # keep short: this is a PARAMETER scan, not a physics run
    dt        = 0.05,
    maxdim    = 0,       # 0 = the per-model rank-limited default (see below)
    nseeds    = 6,       # phase 1 only
))

const T_MAX  = P.t_over_pi * π
const NSTEPS = round(Int, T_MAX / P.dt)

# ── the two models, each with an EXACT reference and no 2^L object anywhere ────────────────
#
# Both references are closed form / free fermions, so the error column is an error rather than a
# disagreement, and both stay available at L = 16 where exact diagonalisation does not.

oat_sx_exact(L, t) = (L / 2) * cos(t / 2)^(L - 1)

function xx_hopping(L; J = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    return h
end
xx_left_mag_exact(L, t; J = 1.0) =
    (U = exp(-im * t * xx_hopping(L; J = J));
     sum(sum(abs2(U[j, k]) for k in 1:(L ÷ 2)) - 0.5 for j in 1:(L ÷ 2)))

function build_model(model::String, L::Int)
    if model == "oat"
        set_symmetry!(:Z2)
        # OAT's exact rank is min(n,L-n)+1 = 9 at the centre for L=16, CONSTANT in time. maxdim
        # 6 sits BELOW it, so the run is genuinely rank-limited and the budget has work to do --
        # at or above 9 every arm is exact and the scan would measure roundoff.
        return (mpo = oat_mpo(L), psi0 = x_polarized_state(L),
                obs = psi -> total_sx(psi), exact = t -> oat_sx_exact(L, t),
                cap = 6, hnorm = L^2 / 8, label = "OAT")
    elseif model == "xx"
        set_symmetry!(:U1)
        # XX has NO rank ceiling -- the wall spreads and entanglement grows -- so any cap is a
        # genuine truncation and every arm should ratchet into it. That is the regime where the
        # sketch is under pressure, and the one OAT structurally cannot probe.
        return (mpo = xxz_mpo(L; J = 1.0, delta = 0.0), psi0 = domain_wall_state(L),
                obs = psi -> sum(sz_expectation(psi, j) for j in 1:(L ÷ 2)),
                exact = t -> xx_left_mag_exact(L, t),
                cap = 24, hnorm = L / 2, label = "XX")
    end
    error("model must be \"oat\" or \"xx\", got $(repr(model))")
end

const M      = build_model(P.model, P.L)
const MAXDIM = P.maxdim == 0 ? M.cap : P.maxdim

# Krylov depth from ||H||*dt/2, so the SOLVER is never the limiter. A depth that is too shallow
# does not error -- it caps every arm at the same floor and makes the scan look flat.
function krylov_depth(hnorm, dt; thresh = 1e-14, lo = 12, hi = 40)
    arg, m = hnorm * dt / 2, lo
    while m < hi && arg^m / factorial(big(m)) > thresh
        m += 1
    end
    return m
end
const MAXITER = krylov_depth(M.hnorm, P.dt)

# ── one arm ───────────────────────────────────────────────────────────────────────────────

"""
    run_arm(scheme, kw, seed) -> NamedTuple

Evolve a fresh copy of the initial state to `T_MAX` and report error, both cost axes, and the
achieved rank.

`kw` is spliced into BOTH schemes identically -- they take the same sketch keywords -- which is
what makes the grid a fair comparison rather than two separate tunings.
"""
function run_arm(scheme::String, kw::NamedTuple, seed::UInt32)
    psi = copy(M.psi0)
    rng = MersenneTwister(seed)
    step! = scheme == "cbe_bug" ?
        (p, tau) -> cbe_bug_step!(p, M.mpo, tau; maxdim = MAXDIM, trunc_thresh = 1e-12,
                                  maxiter = MAXITER, rng = rng, kw...) :
        (p, tau) -> tdvp_cbe1s_step!(p, M.mpo, tau; maxdim = MAXDIM, trunc_thresh = 1e-12,
                                     maxiter = MAXITER, rng = rng, kw...)
    kry, chi, efnl = 0, 0, 0.0
    # Warm the JIT on a throwaway copy so the timing is the ALGORITHM, not compilation. Without
    # it the first arm of every scan absorbs the whole compile and looks catastrophically slow.
    step!(copy(M.psi0), ComplexF64(-im * P.dt))
    t0 = time_ns()
    for _ in 1:NSTEPS
        info = step!(psi, ComplexF64(-im * P.dt))
        kry += info.krylov_dims
        chi  = max(chi, maximum(bond_dims(psi)))
        efnl = max(efnl, info.err_fnl)
    end
    secs = (time_ns() - t0) / 1e9
    o    = M.obs(copy(psi))
    xt   = M.exact(NSTEPS * P.dt)          # the state ends at NSTEPS*dt, which need not be T_MAX
    return (err = abs(o - xt), krylov = kry, secs = secs, chi = chi, err_fnl = efnl, obs = o)
end

# ── the phases ────────────────────────────────────────────────────────────────────────────

const COLS = "model,L,scheme,phase,arm,dex,growth,dover,comp_ratio,exact,seed," *
             "err,krylov,secs,chi,err_fnl"

emit(io, scheme, phase, arm, kw, seed, r) = begin
    @printf(io, "%s,%d,%s,%d,%s,%d,%g,%s,%s,%s,0x%X,%.8e,%d,%.4f,%d,%.3e\n",
            M.label, P.L, scheme, phase, arm,
            get(kw, :dex, 0), get(kw, :growth, NaN),
            string(get(kw, :dover, nothing)), string(get(kw, :comp_ratio, nothing)),
            string(get(kw, :exact, false)), seed,
            r.err, r.krylov, r.secs, r.chi, r.err_fnl)
    flush(io)
    @printf("  %-11s %-22s err %.3e   krylov %6d   %7.2fs   chi %2d   err_fnl %.1e\n",
            scheme, arm, r.err, r.krylov, r.secs, r.chi, r.err_fnl)
    flush(stdout)
end

# A common budget for the gate and the noise floor, stated explicitly so neither scheme runs at
# its own default and the two are comparable from the first line of output.
const BASE = (dex = 0, growth = 1.5, comp_ratio = 1.0)

function phase0(io)
    println("\nPHASE 0 -- saturation gate: does the exact SVD differ from the sketch?\n")
    for scheme in ("tdvp_cbe1s", "cbe_bug"), ex in (false, true)
        kw = merge(BASE, (exact = ex,))
        emit(io, scheme, 0, ex ? "exact-svd" : "sketch", kw, 0x5EED, run_arm(scheme, kw, 0x5EED))
    end
    println("\n  AGREE  -> saturated: no sampling knob can matter, go straight to phase 2.")
    println("  DIFFER -> the sketch is lossy here; dover/comp_ratio/seed are live knobs.\n")
end

function phase1(io)
    println("\nPHASE 1 -- noise floor: seed spread IS the resolution of every other number.\n")
    for scheme in ("tdvp_cbe1s", "cbe_bug")
        errs = Float64[]
        for k in 1:P.nseeds
            seed = UInt32(0x5EED + 977k)          # spread apart, not consecutive
            r = run_arm(scheme, BASE, seed)
            push!(errs, r.err)
            emit(io, scheme, 1, "seed$k", BASE, seed, r)
        end
        m = sum(errs) / length(errs)
        @printf("  -> %-11s mean %.4e   spread %.3e   relative %.2e\n\n",
                scheme, m, maximum(errs) - minimum(errs),
                m == 0 ? 0.0 : (maximum(errs) - minimum(errs)) / m)
    end
end

function phase2(io)
    println("\nPHASE 2 -- budget scan. maxdim = $MAXDIM, tight enough that arms saturate it,")
    println("so dex measures WHICH directions survive at fixed rank, not how many.\n")
    for scheme in ("tdvp_cbe1s", "cbe_bug")
        for g in (1.1, 1.25, 1.5, 2.0)
            kw = (dex = 0, growth = g, comp_ratio = 1.0)
            emit(io, scheme, 2, @sprintf("growth=%.2f", g), kw, 0x5EED, run_arm(scheme, kw, 0x5EED))
        end
        for d in (2, 4, 8, 16)
            kw = (dex = d, growth = 1.5, comp_ratio = 1.0)
            emit(io, scheme, 2, "dex=$d", kw, 0x5EED, run_arm(scheme, kw, 0x5EED))
        end
        println()
    end
end

# ── run ───────────────────────────────────────────────────────────────────────────────────

@printf("RSVD parameter scan   %s  L=%d  maxdim=%d  dt=%g  t=%gπ (%d steps)  krylov depth=%d\n",
        M.label, P.L, MAXDIM, P.dt, P.t_over_pi, NSTEPS, MAXITER)
@printf("initial chi = %d   obs(0) = %.6f (exact %.6f)\n",
        maximum(bond_dims(M.psi0)), M.obs(copy(M.psi0)), M.exact(0.0))
flush(stdout)

mkpath(RESULTS)
const OUT = joinpath(RESULTS, @sprintf("rsvd_scan_%s_L%d_p%d.csv", P.model, P.L, P.phase))
io = open(OUT, "w")
try
    println(io, COLS)
    P.phase == 0 ? phase0(io) : P.phase == 1 ? phase1(io) :
    P.phase == 2 ? phase2(io) : error("phase must be 0, 1 or 2, got $(P.phase)")
finally
    close(io)
end
println("wrote ", OUT)
