# L = 18 BENCHMARK: no symmetry vs SU(2), across every arm, against an EXACT reference.
#
#   julia --project=. benchmarks/l18_matrix.jl                    # full grid
#   L=8 TMAX=1 BETA=1 MAXDIMS=16 SMOKE=1 julia --project=. benchmarks/l18_matrix.jl
#
# WHAT IS BEING TESTED. Symmetry should change the COST and not the PHYSICS: a symmetric MPO is
# block diagonal, hence sparser, and an SU(2) multiplet stands for `2S+1` states. So the two
# symmetry runs must start from the same state, evolve the same Hamiltonian, and agree on every
# observable -- and they are compared against one reference computed once, independent of both.
#
# THE COMMON STATE IS THE DIMER COVERING, and it has to be, because nothing else is representable
# on both sides: `product_state`/`neel_state` are guarded off under `:SU2` (a definite-Sz product
# state spans many total-spin sectors) while `dimer_state` used to be `:SU2`-only. It is now
# available in every mode.
#
# THE OBSERVABLE IS THE BOND ENERGY `<S_j . S_{j+1}>`, not `<Sz_j>`. Under `:SU2` the state is a
# total-spin multiplet, so `<Sz_j>` is identically zero by symmetry -- and `Sz` is not even
# exposed, so asking for it throws. `S.S` is an SU(2) scalar: nonzero, defined in every mode, and
# the same operator on both sides, so the light cones are directly comparable.
#
# THE REFERENCE IS EXACT, by sparse Krylov -- see `exact_sparse.jl`. `exp(-iHt)|psi>` needs the
# VECTOR, not the spectrum: at L=18 the Sz=0 sector is C(18,9) = 48620 states, so the state is
# ~780 kB and H is sparse, while a dense `eigen` would be ~19 GB. The reference is therefore
# exact to Krylov tolerance rather than a better-converged MPS, and the errors below are
# ACCURACY, not mutual consistency.
#
# RESUMABLE. Every finished configuration is appended to the output JSON and skipped on a rerun,
# so an interrupted laptop run continues instead of restarting. Delete the file to force a
# recompute, or narrow the grid with the ARMS / MODES / SYMS / MAXDIMS / CUTOFFS variables.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime, ImaginaryTime, GroundState
using LinearAlgebra, Printf, JSON, Telum
include(joinpath(@__DIR__, "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

# ── configuration ────────────────────────────────────────────────────────────────────
const L       = parse(Int, get(ENV, "L", "18"))
const DT      = parse(Float64, get(ENV, "DT", "0.05"))
const TMAX    = parse(Float64, get(ENV, "TMAX", "15.0"))
const BETA    = parse(Float64, get(ENV, "BETA", "15.0"))
const SNAP    = parse(Float64, get(ENV, "SNAP", "0.25"))   # record every SNAP of time
const MAXDIMS = [parse(Int, s) for s in split(get(ENV, "MAXDIMS", "32,64,128"), ",")]
# "no cutoff" is 0.0. NOTE it is not literally zero everywhere downstream: the CBE paths clamp
# with `max(trunc_thresh, 1e-14)` and Telum's SVD selector drops exactly-zero directions anyway,
# so 0.0 means "the smallest cutoff the code can express", not "keep every direction".
const CUTOFFS = [parse(Float64, s) for s in split(get(ENV, "CUTOFFS", "0.0,1e-12"), ",")]
# KRYLOV DEPTH FOR THE TIME ARMS. The library default is 30 and that is 3.6x more work than
# this problem needs, because `lanczos_expv` has a BREAKDOWN-ONLY exit -- `b < tol && break`
# tests the off-diagonal Lanczos norm, NOT convergence of the exponential -- so for a generic
# Theta it never fires and every solve runs the full budget. MEASURED at L=18, one step:
#
#   maxiter=30  66 expv calls  1806 matvecs  mean dim 27.4   6.04 s/step
#   maxiter=8   66 expv calls   486 matvecs  mean dim  7.4   1.38 s/step
#
# i.e. the mean Krylov dimension sits at 27.4 out of a cap of 30: the budget is spent, not
# converged into. What that extra depth buys, over 40 steps at dt=0.05 against the maxiter=30
# trajectory:
#
#   maxiter=16  2.02x faster  max|dSz| = 1.7e-13
#   maxiter=12  2.50x faster  max|dSz| = 2.0e-13
#   maxiter=8   3.64x faster  max|dSz| = 2.1e-13      <- default here
#   maxiter=6   4.75x faster  max|dSz| = 3.5e-12
#   maxiter=4   5.77x faster  max|dSz| = 3.2e-07      <- the cliff
#
# 8 sits an ample margin above the cliff at machine precision. It is NOT a universal constant:
# Krylov depth is set by ||tau*H||, which here is ~0.3 (dt/2 = 0.025, ||H|| ~ L/2). A larger
# dt or a wider spectrum needs more, so this is an ENV knob rather than a hardcoded literal,
# and the real fix is a convergence test inside `lanczos_expv` (deliberately not done here --
# src/BondUpdateBUG is not to be modified without asking, and it would change numerics for
# every arm, not just this benchmark).
#
# GROUND-STATE ARMS DELIBERATELY EXCLUDED: their `maxiter` drives a Lanczos EIGENSOLVER, not
# an exponential, and nothing above measures that. They keep the library default.
const MAXITER = parse(Int, get(ENV, "MAXITER", "8"))
# COMP_RATIO and S_ITERS are the two knobs the CBE arms are actually tuned on, so they are
# reachable from the environment rather than frozen at the option defaults.
#
# `S_ITERS` selects the S-step scheme and its SIGN matters: <= 0 is the expanding-basis
# Lanczos (-1 was hardcoded here before), 1 is one expansion then a plain Lanczos, > 1 is the
# iterated expansion that re-sketches each pass from the basis the previous pass expanded.
# NOTE the iterated scheme is INERT at the default growth = 2.0 -- the first pass already
# saturates the local space -- so scanning S_ITERS without also lowering GROWTH measures the
# saturation and not the scheme. Measured at L=8/L=10: growth 1.25 + s_iters 2 is where it pays.
const COMP_RATIO = parse(Float64, get(ENV, "COMP_RATIO", "0.5"))
const S_ITERS    = parse(Int,     get(ENV, "S_ITERS", "-1"))
const GROWTH     = parse(Float64, get(ENV, "GROWTH", "2.0"))
const SYMS    = [Symbol(s) for s in split(get(ENV, "SYMS", "none,SU2"), ",")]
# INITIAL STATE AND OBSERVABLE ARE ONE CHOICE, NOT TWO -- each state has the observable that
# actually shows its light cone, and each pairing has a different symmetry reach:
#
#   dimer       <S_j.S_{j+1}>   an SU(2) SCALAR, so it runs in :none, :U1 AND :SU2. This is the
#                               only pairing that can compare symmetric against non-symmetric,
#                               which is why it is the default.
#   domainwall  <Sz_j>          the canonical magnetisation light cone, but ABELIAN ONLY: under
#                               :SU2 the state is unrepresentable (a definite-Sz product state
#                               spans many total-spin sectors) AND the operator is not exposed
#                               (local_space(:SU2) has :I,:S and no :Sz), so <Sz_j> is
#                               identically zero by symmetry. Asking for it under :SU2 throws
#                               rather than silently returning a blank cone.
const INIT    = Symbol(get(ENV, "INIT", "dimer"))
INIT in (:dimer, :domainwall) || error("INIT must be dimer or domainwall, got $INIT")
const MODES   = [String(s) for s in split(get(ENV, "MODES", "real,imag,ground"), ",")]
const OUT     = get(ENV, "OUT", joinpath(@__DIR__, "l18_results.json"))
const SMOKE   = get(ENV, "SMOKE", "0") != "0"
# DELTA = 0 IS THE XX CHAIN, AND IT CHANGES WHAT THE REFERENCE COSTS. A domain wall under XX is
# a free-fermion problem by Jordan-Wigner, so <Sz_j(t)> is known in CLOSED FORM from an L x L
# single-particle matrix exponential -- O(L^3), independent of the Hilbert space. The sparse
# Krylov reference is only needed for delta != 0.
#
# The difference is not marginal. The Sz=0 sector is C(L, L/2):
#
#   L=18   48,620 states     H as COO triplets  0.0 GB    <- sparse reference is free
#   L=30  155,117,520        H as COO triplets 69.3 GB    <- ~150 GB with the CSC, plus
#                                                            2.31 GB per Krylov vector
#
# So at L=30 the sparse path is what breaks, NOT the arms -- they are O(L*chi^3) and do not
# care. Running XX gives an exact reference at any L for free. Symmetry is orthogonal to this:
# :none / :U1 / :SU2 are the same physics and only change the solver cost.
const DELTA    = parse(Float64, get(ENV, "DELTA", "1.0"))
const ANALYTIC = (DELTA == 0.0 && INIT === :domainwall)

const NSTEP  = round(Int, TMAX / DT)
const NBETA  = round(Int, BETA / DT)
const EVERY  = max(1, round(Int, SNAP / DT))

# ── the arms ─────────────────────────────────────────────────────────────────────────
# Each entry advances `psi` by `nchunk` steps of `tau`. Chunking (rather than one long call)
# is what lets every arm -- bare steppers and whole-sweep drivers alike -- be snapshotted on
# the same grid without special-casing the driver arms.
const TIME_ARMS = [
    "tdvp2",
    "1site_tdvp_cbe", "1site_tdvp_cbe_rsvd",
    "bug_cbe_gate", "bug_cbe_rsvd_gate",
    "bug_cbe_lubich", "bug_cbe_rsvd_lubich",
]
const GROUND_ARMS = ["dmrg_cbe", "dmrg_cbe_rsvd"]

function make_advance(arm::String, h, gates, maxdim::Int, cutoff::Float64)
    # `n_steps` lives INSIDE CBEBugOptions for the gate driver -- it is not a keyword on
    # `cbe_gate_evolve!` -- so the options object is rebuilt per chunk rather than hoisted.
    # `s_iters = -1` selects the expanding-basis Lanczos; note `cbe_lubich_sweep` does NOT
    # accept it (it has no S-step solver choice), which is why only the gate arm passes it.
    # `maxiter = MAXITER` on EVERY time arm, so the Krylov depth is one number across the
    # comparison. Leaving it at the library default for some arms and not others would make
    # the timings incomparable -- the arms differ in how MANY expv calls they issue per step
    # (measured: 66 for tdvp2, 1 for a lubich sweep), so a per-arm depth would confound
    # "does less Krylov work" with "was given a smaller budget".
    o(exact, n) = CBEBugOptions(; dt = DT, n_steps = n, maxdim = maxdim, trunc_thresh = cutoff,
                                s_iters = S_ITERS, exact = exact, normalize = false,
                                maxiter = MAXITER, comp_ratio = COMP_RATIO, growth = GROWTH)
    if arm == "tdvp2"
        return (psi, tau, n) -> for _ in 1:n
            tdvp2_step!(psi, h, tau; maxdim = maxdim, trunc_thresh = cutoff, maxiter = MAXITER)
        end
    elseif arm in ("1site_tdvp_cbe", "1site_tdvp_cbe_rsvd")
        ex = arm == "1site_tdvp_cbe"
        return (psi, tau, n) -> for _ in 1:n
            tdvp1_cbe_step!(psi, h, tau; maxdim = maxdim, trunc_thresh = cutoff, exact = ex,
                            maxiter = MAXITER)
        end
    elseif arm in ("bug_cbe_gate", "bug_cbe_rsvd_gate")
        ex = arm == "bug_cbe_gate"
        return (psi, tau, n) -> cbe_gate_evolve!(psi, gates, tau; opts = o(ex, n))
    elseif arm in ("bug_cbe_lubich", "bug_cbe_rsvd_lubich")
        ex = arm == "bug_cbe_lubich"
        return (psi, tau, n) -> for _ in 1:n
            cbe_lubich_sweep(psi, h, tau; maxdim = maxdim, trunc_thresh = cutoff, exact = ex,
                             maxiter = MAXITER)
        end
    end
    error("unknown arm $arm")
end

# ── reference, computed once and shared by both symmetries ───────────────────────────
if ANALYTIC
    ("imag" in MODES || "ground" in MODES) && error(
        "DELTA=0 uses the analytic free-fermion reference, which gives <Sz_j(t)> for REAL " *
        "time only. Use MODES=real, or DELTA!=0 to get the sparse reference back.")
    :SU2 in SYMS && error("DELTA=0 is anisotropic; xxz_chain refuses under :SU2. Use SYMS=none,U1.")
    println("reference: ANALYTIC free-fermion (XX, delta=0) -- exact, O(L^3), no Hilbert space")
else
    println("building the exact reference (sparse Krylov, Sz=0 sector)...")
end
flush(stdout)
const A, STATES, IDX = ANALYTIC ? (nothing, nothing, nothing) :
                                  heisenberg_sparse(L; delta = DELTA)
const V0 = ANALYTIC ? nothing :
           INIT === :dimer ? dimer_vector(L, STATES, IDX) :
                             domain_wall_vector(L, STATES, IDX)
# The observable is chosen WITH the state, on both sides of the comparison, so the MPS and the
# dense reference are always measuring the same operator.
obs_dense(v) = INIT === :dimer ? bond_energies_dense(v, L, STATES, IDX) :
                                 magnetisation_dense(v, L, STATES)
obs_mps(psi, gates) = INIT === :dimer ? bond_energies(psi, gates) :
                                        magnetisation(copy(psi))
const OBSNAME = INIT === :dimer ? "<S_j.S_{j+1}>" : "<Sz_j>"
ANALYTIC || @printf("  sector = %d states, nnz(H) = %d\n", length(STATES), nnz(A))
@printf("  init = %s, observable = %s\n", INIT, OBSNAME)
if INIT === :domainwall && :SU2 in SYMS
    error("INIT=domainwall cannot run under :SU2 -- the state is unrepresentable there and " *
          "<Sz_j> is identically zero by symmetry. Use SYMS=none,U1 for a domain wall, or " *
          "INIT=dimer for the cross-symmetry comparison.")
end
flush(stdout)

const SNAP_T = [k * EVERY * DT for k in 1:(NSTEP ÷ EVERY)]
ref_real = Dict{Int, Vector{Float64}}()
if ANALYTIC
    # Closed form, evaluated straight onto the snapshot grid. `xx_free_fermion_sz` defaults to
    # `occupied = 1:(L/2)`, which IS `domain_wall_state(L)`, so both sides start from the same
    # state -- the thing that makes a reference a reference.
    for k in 1:(NSTEP ÷ EVERY)
        ref_real[k * EVERY] = xx_free_fermion_sz(L, k * EVERY * DT)
    end
else
    let step = 0
        propagate!(A, V0, ComplexF64(-im * DT), NSTEP, (s, v) -> begin
            s % EVERY == 0 && (ref_real[s] = obs_dense(v))
        end)
    end
end
println("  real-time reference done ($(length(ref_real)) snapshots)")
flush(stdout)

ref_imag = Dict{Int, Vector{Float64}}()
ref_imag_E = Dict{Int, Float64}()
if "imag" in MODES
    propagate!(A, V0, ComplexF64(-DT), NBETA, (s, v) -> begin
        if s % EVERY == 0
            ref_imag[s] = obs_dense(v)
            ref_imag_E[s] = sum(ref_imag[s])
        end
    end; normalise = true)
    println("  imaginary-time reference done")
    flush(stdout)
end

const E0 = "ground" in MODES ? first(ground_energy_sparse(A)) : NaN
isnan(E0) || @printf("  exact E0 = %.12f\n", E0)
flush(stdout)

# ── resumability ─────────────────────────────────────────────────────────────────────
# JSON has no NaN. Two things here can legitimately be "not measured" -- `E0` when the ground
# mode was not requested, and an error at a step that is not a reference snapshot -- and both
# must serialise as `null` rather than crash the run at the first `save()`. Silently substituting
# a NUMBER would be worse: a 0.0 error would read as a perfect result.
jsonsafe(x::Float64) = isfinite(x) ? x : nothing
jsonsafe(v::Vector{Float64}) = Any[isfinite(x) ? x : nothing for x in v]
jsonsafe(x) = x

results = isfile(OUT) ? JSON.parsefile(OUT) : Dict{String, Any}()
# MAXITER IS PART OF THE KEY, and it has to be. Results are resumed by key, so without it a
# rerun at a different Krylov depth would SKIP every row already present and leave the file
# holding a mixture -- rows at maxiter=30 (measured: tdvp2 4159 s) sitting beside rows at
# maxiter=8 (~1150 s) with nothing on either to say which is which. The errors would still be
# comparable (the depths agree to 2e-13) but the TIMINGS would not, and timing is the point.
# Including it means a depth change recomputes instead of silently mixing.
key(mode, sym, arm, md, ct) = "$(INIT)|$(mode)|$(sym)|$(arm)|$(md)|$(ct)|mi$(MAXITER)"

# The reference goes in the file too, not just the errors computed from it. Without it the
# plots can show how far each arm is from the truth but never what the truth LOOKS like, and
# the exact light cone is the most informative panel there is -- it is what every arm is trying
# to reproduce. Stored once; it is symmetry-independent by construction.
let snaps = sort(collect(keys(ref_real)))
    results["__reference__$(INIT)"] = Dict(
        "init" => String(INIT), "obs_name" => OBSNAME, "L" => L, "dt" => DT, "tmax" => TMAX, "beta" => BETA, "snap_every" => EVERY,
        "t" => [s * DT for s in snaps],
        "obs_real" => [ref_real[s] for s in snaps],
        "obs_imag" => isempty(ref_imag) ? nothing :
                      [ref_imag[s] for s in sort(collect(keys(ref_imag)))],
        "E0" => jsonsafe(E0),
        # `nothing` on the analytic path: there is no Hilbert-space sector, which is the
        # entire reason that path exists. Recorded as null rather than 0, so a reader cannot
        # mistake "not applicable" for "empty".
        "sector" => ANALYTIC ? nothing : length(STATES),
        "reference" => ANALYTIC ? "analytic_free_fermion" : "sparse_krylov",
        "delta" => DELTA)
end
save() = open(OUT, "w") do io
    JSON.print(io, results)
end

function start_state(sym::Symbol)
    set_symmetry!(sym)
    return INIT === :dimer ? dimer_state(L) : domain_wall_state(L)
end

# ── run ──────────────────────────────────────────────────────────────────────────────
function run_time_arm(mode, sym, arm, md, ct)
    k = key(mode, sym, arm, md, ct)
    haskey(results, k) && return false
    imag_mode = mode == "imag"
    tau = imag_mode ? ComplexF64(-DT) : ComplexF64(-im * DT)
    nsteps = imag_mode ? NBETA : NSTEP
    ref = imag_mode ? ref_imag : ref_real

    psi = start_state(sym)
    gates = bond_gates(psi; delta = DELTA)
    h = sym === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; delta = DELTA)
    adv = make_advance(arm, h, gates, md, ct)

    ts, errs, bds, obs = Float64[], Float64[], Int[], Vector{Float64}[]
    t0 = time()
    done = 0
    while done < nsteps
        n = min(EVERY, nsteps - done)
        adv(psi, tau, n)
        done += n
        if imag_mode
            nrm = norm(psi)
            nrm > 0 && (psi[psi.center] = to_concrete((1.0 / nrm) * psi[psi.center]))
        end
        be = obs_mps(psi, gates)
        push!(ts, done * DT); push!(obs, be)
        push!(bds, maximum(bond_dims(psi)))
        push!(errs, haskey(ref, done) ? norm(be - ref[done]) : NaN)
    end
    results[k] = Dict("mode" => mode, "sym" => String(sym), "arm" => arm, "init" => String(INIT), "obs_name" => OBSNAME,
                      "maxdim" => md, "cutoff" => ct, "t" => ts, "maxiter" => MAXITER,
                      "err" => jsonsafe(errs), "maxbd" => bds, "obs" => obs,
                      "seconds" => time() - t0)
    save()
    @printf("  %-6s %-5s %-22s chi=%-4d ct=%-8.0e  err_end=%-11.3e maxbd=%-4d %6.1fs\n",
            mode, sym, arm, md, ct, errs[end], bds[end], time() - t0)
    flush(stdout)
    return true
end

function run_ground_arm(sym, arm, md, ct)
    k = key("ground", sym, arm, md, ct)
    haskey(results, k) && return false
    psi = start_state(sym)
    h = sym === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; delta = DELTA)
    f = arm == "dmrg_cbe" ? GroundState.cbe_dmrg! : GroundState.rsvd_cbe_dmrg!
    t0 = time()
    info = f(psi, h; opts = CBEBugOptions(maxdim = md, trunc_thresh = ct),
             n_sweeps = 40, etol = 1e-12)
    e = info.energies[end]
    results[k] = Dict("mode" => "ground", "sym" => String(sym), "arm" => arm, "init" => String(INIT),
                      "maxdim" => md, "cutoff" => ct,
                      "energies" => info.energies, "E" => e, "E0" => jsonsafe(E0),
                      "err" => jsonsafe(e - E0), "maxbd" => maximum(info.max_bond_dims),
                      "sweeps" => length(info.energies),
                      "matvecs" => sum(info.krylov_dims), "seconds" => time() - t0)
    save()
    @printf("  ground %-5s %-22s chi=%-4d ct=%-8.0e  E-E0=%-11.3e maxbd=%-4d %6.1fs\n",
            sym, arm, md, ct, e - E0, maximum(info.max_bond_dims), time() - t0)
    flush(stdout)
    return true
end

@printf("\nL=%d dt=%.3f t=%.1f beta=%.1f snap=%.2f  maxdims=%s cutoffs=%s syms=%s maxiter=%d\n\n",
        L, DT, TMAX, BETA, SNAP, MAXDIMS, CUTOFFS, SYMS, MAXITER)
flush(stdout)

for mode in MODES, sym in SYMS, md in MAXDIMS, ct in CUTOFFS
    if mode == "ground"
        for arm in GROUND_ARMS
            run_ground_arm(sym, arm, md, ct)
        end
    else
        for arm in TIME_ARMS
            run_time_arm(mode, sym, arm, md, ct)
        end
    end
end

set_symmetry!(:U1)
println("\nwrote $OUT")
println("L18_MATRIX_DONE")
