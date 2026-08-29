# LOGARITHMIC TIME GRIDS: WHERE THEY WORK, WHERE THEY FAIL, AND WHAT THAT COSTS EACH INTEGRATOR.
#
# Against arXiv:2606.02930, which proposes imaginary-time MPS evolution on a LOGARITHMIC grid with
# A-STABLE IMPLICIT stepping (matrix-vector products plus a LINEAR SOLVE per step), benchmarked on
# a Heisenberg chain and an Anderson impurity model against uniform-step TDVP.
#
# ⛔ EVERY MODEL HERE IS SCORED AGAINST SOMETHING WE DID NOT PRODUCE. That is the rule this file
# was rewritten to obey, and the earlier version broke it: models 2-4 used "the best of our own
# arms" or "our own finest uniform grid" as the reference. That is a GRID control -- it isolates
# the spacing by holding the integrator fixed -- but it CANNOT certify the integrator, because a
# wrong integrator is wrong in both arms and the table still looks clean. Two grids agreeing to
# four figures is the signature, and it happened: n=12 and n=24 both returned err = 5.479e+04.
#
# THE THREE MODELS, and the published result each one is trying to recreate:
#
#   1  Heisenberg chain, SU(2)      imaginary time. Log grid WORKS.
#                                   REFERENCE: Bethe ansatz, EXACT (analytic, ours to check
#                                   against for free). Also the paper's own headline model.
#
#   2  Heisenberg square CYLINDER   imaginary time, 2D via a snake. Log grid WORKS, harder.
#                                   REFERENCE: Loop QMC, arXiv:1705.03222 Table 1 --
#                                   e_bulk = -0.683282(2) per site at Ly = 4. Same geometry
#                                   (periodic in y, open in x) and same normalisation
#                                   (H = sum_<ij> S_i . S_j) as `square_cylinder_couplings`.
#
#   3  XXZ NEEL QUENCH              REAL time. Log grid FAILS -- by ALIASING.
#                                   REFERENCE: Barmettler et al., arXiv:0810.4845 (PRL 2009).
#                                   `m_s(t) = (1/N) sum_j (-1)^j <S^z_j(t)>` from a Neel state,
#                                   and at Delta = 0 the EXACT closed form `m_s = J_0(2Jt)/2`,
#                                   asymptotically `~ t^{-1/2} cos(2 J t - pi/4)`.
#
# ⛔ WHY MODEL 3 IS THE RIGHT FAILURE MODEL AND THE OLD ONE WAS NOT. The old model 3 was an
# invented non-Hermitian chain whose "log grid fails" claim rested on disagreement with our own
# finer grid. Here the OSCILLATION FREQUENCY IS PUBLISHED: `cos(2 J t - pi/4)` has period
# `pi / J`, so a step past `pi / (2 J) ~ 1.571` provably steps over it. The failure becomes
# arithmetic about a known frequency instead of an assertion about our own output. And Delta = 0
# additionally has a closed form, so the SAME model certifies the integrator and demonstrates the
# grid failure -- two jobs that previously needed two models and a self-reference.
#
# ⛔ MPO ONLY, NEVER GATES. `apply_h_two_site(Theta, h::XXZChain, ...)` closes the on-bond term
# with `apply_gate(bond_gate(h, ...))`, so passing an `XXZChain` silently takes the gate path.
# Every model below is an `MPO` built by `pair_mpo` or `heisenberg_su2_mpo`.
#
# Run:  julia --project=. benchmarks/imaginary_time/logtime_suite.jl [chain] [cylinder] [neel]
# Out:  benchmarks/results/logtime_suite.csv   ->  benchmarks/imaginary_time/plot_logtime.py

using LinearAlgebra, Printf, Statistics
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))
include(joinpath(@__DIR__, "..", "..", "tests", "common", "free_fermion.jl"))

# ⛔ THE OUTPUT PATH IS OVERRIDABLE BECAUSE THE CLUSTER RUNS THE MODELS AS AN ARRAY, and three
# concurrent tasks appending to one CSV interleave their writes into a file that still parses.
# `LOGTIME_OUT` gives each task its own file; concatenate afterwards (keeping one header).
const OUT = get(ENV, "LOGTIME_OUT", joinpath(@__DIR__, "..", "results", "logtime_suite.csv"))
# ⛔ THE GRIDS, `run_arm`, `trajectory_error` AND THE CSV SCHEMA NOW LIVE IN ONE SHARED FILE.
# `benchmarks/imaginary_time/dissipative_xx.jl` runs the SAME grid comparison on the open model, and a second copy
# of `log_steps`/`panel_steps` would drift from this one silently -- while `plot_logtime.py` rebuilds
# the log grid from this construction to draw the Nyquist bound, so a drifted copy would put a bound
# on the figure that does not describe the runs plotted beneath it.
include(joinpath(@__DIR__, "grids_common.jl"))


"""
The three arms. `tau = -dt` is imaginary time, `tau = -im*dt` real time -- one sweep, one call,
either way.

⛔ ALL THREE GET THE SAME `expv` KNOBS. `cbe_bug_midpoint_step!` runs two sweeps per step and so
costs ~2x `cbe_bug_step!` at the same `dt`; that is NOT hidden, it is the point of also reporting
wall time and operator applications. Handing one arm a better exponential (or a deeper Krylov
budget) than another would make every number here a tuning difference dressed as a method
difference.
"""
# ⛔ `exact = false` IS THE METHOD; `exact = true` IS AN A/B REFERENCE. rSVD-CBE is what the sweep
# IS -- rank grows only through `rsvd_cbe` -- and the deterministic CBE arm exists to check it,
# not to be benchmarked as if it were the product. Every earlier row in this suite passed
# `exact = true`, so no BUG number produced before now is a measurement of the shipped method.
#
# ⛔ `dover = 4`, NOT THE DEFAULT. `dover = nothing` resolves to `ceil(0.2*dex)` (the reference's
# `Dpre = ceil(1.2*dex)`), which is a handful of extra columns when `dex` is small -- too thin an
# oversampling for the sketch to find the directions reliably. `dover = 4` is what the
# triple-`H*Theta` fix was measured with (1.3-2.4x over pre-fix, 4-14x over tdvp2).
const RSVD_KW = (exact = false, dover = 4)

const DEFAULT_ARMS = ("bug_rsvd", "bug_par", "midpoint", "tdvp2")

"""
The arms. `tau = -dt` is imaginary time, `tau = -im*dt` real time -- one sweep, one call, either
way. `arms` selects which to run, by name.

⛔ ALL ARMS GET THE SAME `expv` KNOBS. `cbe_bug_midpoint_step!` runs two sweeps per step and so
costs ~2x `cbe_bug_step!` at the same `dt`; that is NOT hidden, it is the point of also reporting
wall time and operator applications. Handing one arm a better exponential (or a deeper Krylov
budget) than another would make every number here a tuning difference dressed as a method
difference.

⚠ `bug_par` NEEDS `julia -t 2` OR MORE AND IS A PESSIMISATION WITHOUT IT. The two half-sweeps are
launched as tasks; on a single worker they interleave and the arm measures the sequential sweep
plus scheduling overhead. It is also NOT bit-identical to `bug_rsvd` -- each sweep re-gauges its
own copy, so the floating-point path differs while the spaces do not.
"""
steppers(W, D, maxiter, imag_time; hermitian::Bool = true, arms = DEFAULT_ARMS) = begin
    tau(dt) = imag_time ? ComplexF64(-dt) : ComplexF64(-im * dt)
    kw = (maxdim = D, trunc_thresh = 1e-12, maxiter = maxiter, hermitian = hermitian)
    tbl = Dict{String, Function}(
        "bug_rsvd"  => (p, dt) -> cbe_bug_step!(p, W, tau(dt); kw..., RSVD_KW...),
        "bug_par"   => (p, dt) -> cbe_bug_step!(p, W, tau(dt); kw..., RSVD_KW...,
                                               parallel = true),
        "bug_exact" => (p, dt) -> cbe_bug_step!(p, W, tau(dt); kw..., exact = true),
        "midpoint"  => (p, dt) -> cbe_bug_midpoint_step!(p, W, tau(dt); kw..., RSVD_KW...),
        "tdvp2"     => (p, dt) -> tdvp2_step!(p, W, tau(dt); kw...))
    for a in arms
        haskey(tbl, a) || throw(ArgumentError(
            "unknown arm $a; have $(sort(collect(keys(tbl))))"))
    end
    return [a => tbl[a] for a in arms]
end


# ── 1. Heisenberg chain, SU(2), imaginary time. Log grid WORKS. ──────────────────────────────
#
# REFERENCE: `bethe_xxx_open_energy` -- the analytic Bethe ansatz for the open XXX chain. Exact,
# independent of every line of the integrator, and asserted to a residual below 1e-12 before it is
# used (a Bethe solve that converged to the WRONG root has been observed at residual 1e-14 while
# 0.24 off; the residual alone is not enough, but a bad residual is disqualifying).
#
# ⛔ NO UNIFORM BASELINE HERE, DELIBERATELY. "Log beats uniform" is arXiv:2606.02930's result and
# is a property of the PROBLEM (in imaginary time a Hermitian generator gives
# `sum_n w_n exp(-tau E_n)` with `w_n >= 0` -- completely monotone, which is exactly what
# geometric spacing is built for), not of any integrator. Re-deriving it would cost the most
# expensive arms in the suite (1000 uniform steps against 12 log ones, twice) to confirm someone
# else's finding. What is ours to show is BUG against 2-site TDVP ON THE SAME LOG GRID.
#
# ⛔ AND THE STEP COUNTS GO BELOW 12 BECAUSE 12 WAS ALREADY SATURATED. Measured: n=12/24/48 gave
# 5.329e-14 / 4.263e-14 / 3.020e-14 -- all at the double-precision floor, so the extra steps were
# pure cost. The interesting question is how FEW steps suffice, which is what n=4,6,8 answers.
#
# ⛔ THE PANEL SCAN LIVES HERE BECAUSE THIS IS THE MODEL WITH AN EXACT REFERENCE, so "the minimum
# number of panels" is a question with an answer rather than a judgement call. `Ps` sweeps
# arXiv:2606.02930's own grid at `per_panel = 4`; the smallest `P` whose energy reaches the Bethe
# value IS the minimum, read off the table rather than argued.
#
# ⚠ WHAT `P` CAN AND CANNOT FIX, BEFORE ANY NUMBER IS READ. From `panel_steps`, the LARGEST step
# is `tau_max / (2 * per_panel)` for EVERY `P` -- so `P` buys resolution near `tau = 0` and
# nothing at all at the coarse end. Two consequences:
#   * a `P` scan at fixed `per_panel` has a FLOOR it cannot go below, set by the last panel;
#   * total cost is `P * per_panel`, i.e. LINEAR in `P` while the refinement it buys is
#     EXPONENTIAL (first step `tau_max / (2^(P-1) * per_panel)`).
# That asymmetry is the whole reason the grid is worth anything, and it is also why "increase P"
# and "increase per_panel" are not interchangeable fixes.
function model_chain(io; L = 20, D = 64, tau_max = 60.0, tau_0 = 0.05, maxiter = 40,
                     Ps = (1, 2, 3, 4, 6, 8), per_panel = 4, arms = DEFAULT_ARMS,
                     syms = (:SU2, :U1))
    # ⚠ THE REFERENCE IS SYMMETRY-INDEPENDENT -- it is the same Hamiltonian either way -- so it is
    # computed ONCE, outside the loop, and every symmetry is scored against the same number.
    eref, _, res = bethe_xxx_open_energy(L)
    res < 1e-12 || error("Bethe residual $res -- reference not trustworthy")
    println("\n== 1. HEISENBERG CHAIN  L=$L D=$D tau_max=$tau_max  syms=$(collect(syms))")
    println("   reference: Bethe ansatz (exact)   E = $(round(eref, digits = 10))")
    @printf("   panel grid (arXiv:2606.02930, per_panel=%d): largest step %.4f for EVERY P\n",
            per_panel, tau_max / (2 * per_panel))
    for P in Ps
        d = panel_steps(tau_max, P; per_panel = per_panel)
        @printf("     P=%-3d steps=%-4d first %10.3e  last %8.3f\n",
                P, length(d), d[1], d[end])
    end
    # ⛔ FLUSH THE HEADER. Redirected stdout is block-buffered, and the FIRST arm of a model can
    # take many minutes -- so without this an early death is indistinguishable from a hang, and a
    # 13-minute run that died in model 1 looks exactly like a 13-minute run still working. That
    # ambiguity has already cost several diagnostic cycles here.
    flush(stdout)
    for sym in syms
        set_symmetry!(sym)
        # ⛔ TWO MPO BUILDERS, AND THE CHOICE IS NOT COSMETIC. `heisenberg_su2_mpo` is the
        # SU(2)-specialised path every existing chain number was produced with, and it is kept so
        # those numbers stay reproducible bit for bit. It has no U(1) analogue, so the other modes
        # use the generic `pair_mpo` + `spin_vertices` construction -- exactly what `model_cylinder`
        # already uses for both symmetries.
        # ⚠ THEY MUST REPRESENT THE SAME HAMILTONIAN, AND THIS IS SELF-CHECKING: both are scored
        # against the SAME Bethe energy, so if the two builders disagreed the U(1) `err` column
        # would blow up immediately rather than quietly reporting a different model.
        W = sym === :SU2 ? heisenberg_su2_mpo(L) :
            pair_mpo(L, [i + 1 == j ? 1.0 : 0.0 for i in 1:L, j in 1:L], spin_vertices(L))
        # ⚠ AFTER `set_symmetry!`, never before -- the start state's tensors carry the mode.
        psi0 = dimer_state(L)                    # :SU2 admits the dimer, not Neel
        obs(p) = real(mpo_energy(copy(p), W)) / max(norm(p)^2, eps())
        # ⛔ `D` COUNTS MULTIPLETS UNDER SU(2) AND STATES OTHERWISE, so the same `D` is NOT the same
        # variational space and the two rows are NOT a like-for-like accuracy comparison. MEASURED
        # on the cylinder: `tdvp2` at `:U1, maxdim=128` gave `err = 2.9e-02` against `2.76e-03` at
        # `:SU2, maxdim=70` -- the U(1) run was UNDER-RESOLVED, not worse-behaved. Compare within a
        # symmetry, or match the EFFECTIVE dimension; never read across at fixed `D`.
        println("\n   -- symmetry = $sym --"); flush(stdout)
        for (nm, st) in steppers(W, D, maxiter, true; arms = arms)
            for n in (4, 6, 8, 12)
                o, k, c, s, mn, fin = run_arm(st, psi0, log_steps(tau_0, tau_max, n), obs;
                                              imag_time = true)
                emit(io, "heis_chain", String(sym), L, "log", n, nm, o, eref, k, c, s, mn, fin)
            end
            # ⚠ `nsteps` RECORDS `P * per_panel`, THE ACTUAL STEP COUNT, NOT `P`. The two are the
            # thing most easily confused between this file and the paper (see `panel_steps`), and a
            # cost axis that plotted `P` would flatter the panel grid by a factor of `per_panel`.
            for P in Ps
                dts = panel_steps(tau_max, P; per_panel = per_panel)
                o, k, c, s, mn, fin = run_arm(st, psi0, dts, obs; imag_time = true)
                emit(io, "heis_chain", String(sym), L, "panel", length(dts), nm, o, eref,
                     k, c, s, mn, fin)
            end
        end
    end
end

# ── 2. Heisenberg square CYLINDER, SU(2), imaginary time. Log grid WORKS, 2D. ────────────────
#
# REFERENCE: Loop QMC, arXiv:1705.03222 Table 1 -- Juan Osorio Iregui, MATTHIAS TROYER, Philippe
# Corboz, "Infinite Matrix Product States vs Infinite Projected Entangled-Pair States on the
# Cylinder: a comparative study". Ground-state energy per site of the S=1/2 Heisenberg
# antiferromagnet on INFINITE-LENGTH cylinders, periodic across the width, open along the length:
#
# ⛔ TABLE 1 IS QMC, NOT ONE OF THE PAPER'S OWN VARIATIONAL ENERGIES, AND THE DISTINCTION DECIDES
# WHETHER IT CAN ANCHOR ANYTHING. The paper's SUBJECT is comparing iMPS against iPEPS, so most of
# its numbers are variational -- upper bounds, which our energy could legitimately beat and which
# therefore cannot certify us. Table 1 is different: its caption reads "Loop QMC estimates for the
# infinite-length finite-width ground state energies of the S=1/2 Heisenberg model on square
# lattice cylinders", i.e. unbiased QMC used as the paper's OWN yardstick. Quoting a variational
# row from this paper as "the reference" would install an upper bound as an exact anchor.
#
#     Ly    4            6            8            10           12
#     e     -0.683282(2) -0.672788(1) -0.670760(2) -0.670101(2) -0.669815(2)
#
# ⛔ THE GEOMETRY AND THE NORMALISATION BOTH MATCH OURS, WHICH IS WHY THIS TABLE IS USABLE AT ALL.
# The paper's `H = sum_<ij> S_i . S_j` counts each bond once with no prefactor, which is exactly
# what `pair_mpo(L, Jm, spin_vertices(L))` builds from `square_cylinder_couplings(Lx, Ly;
# periodic_y = true)`. A table for a TORUS, or one carrying a `J/2` or a `1/N` in the definition,
# would have been off by a clean factor and looked like a physics disagreement.
#
# ⛔ THE ONE MISMATCH IS LENGTH, AND IT IS HANDLED BY A FIT RATHER THAN IGNORED. The QMC number is
# a BULK density; our cylinders are finite in `x` and therefore carry two open ends whose missing
# bonds RAISE the energy per site. Comparing a finite `E/N` directly against `-0.683282` would
# report the edge correction as an integrator error -- MEASURED at Ly=4 the two differ in the
# second decimal. Instead, `E(Lx)` is linear in `Lx` at fixed `Ly`,
#
#     E(Lx) = e_bulk * Ly * Lx + 2 * e_edge,
#
# so the SLOPE across two or more `Lx` divides out the ends, and `slope / Ly` is what the QMC
# value is compared against. Three lengths let the linearity itself be checked rather than
# assumed -- if the residual is not tiny, the fit is not yet in the asymptotic regime and the
# comparison must not be quoted.
#
# ⚠ AND `e_bulk` IS A PROPERTY OF THE STATE, NOT OF THE GRID, so every arm is fitted separately:
# an integrator that converged to a different state gives a different slope, which is precisely
# the signal we want. The per-arm `err` column stays the deviation of that arm's fitted `e_bulk`
# from the published number.
const QMC_CYLINDER = Dict(4 => -0.683282, 6 => -0.672788, 8 => -0.670760,
                          10 => -0.670101, 12 => -0.669815)      # arXiv:1705.03222 Table 1

function model_cylinder(io; Ly = 4, Lxs = (4, 6, 8), D = 96, tau_max = 40.0, tau_0 = 0.05,
                        maxiter = 40, ns = (8, 12), arms = DEFAULT_ARMS,
                        syms = (:SU2, :U1))
    haskey(QMC_CYLINDER, Ly) || error("no published QMC value for Ly = $Ly")
    eref = QMC_CYLINDER[Ly]
    println("\n== 2. HEISENBERG SQUARE CYLINDER  Ly=$Ly  Lx=$(collect(Lxs))  D=$D  syms=$(collect(syms))")
    println("   reference: Loop QMC arXiv:1705.03222 Table 1   e_bulk = $eref per site")
    # ⛔ `D` DOES NOT MEAN THE SAME THING IN THE TWO SYMMETRY MODES, and this is the trap that
    # makes a symmetry comparison look like a physics result. Under `:SU2` the cap counts
    # MULTIPLETS; under `:U1` it counts STATES. An `S` multiplet carries `2S+1` states, so
    # `:SU2` at `D` is a STRICTLY LARGER variational space than `:U1` at the same `D` -- by a
    # factor of roughly 2-3 on a Heisenberg model. Reading "SU(2) is more accurate at equal D" off
    # this table would therefore be reading the units, not the method.
    #
    # The honest comparisons are: same D -> compare COST at whatever accuracy each reaches, and
    # ACCURACY -> compare at matched `chi` (the reported bond dimension), which the `chi` column
    # carries per row precisely so this can be checked afterwards rather than assumed here.
    println("   ⚠ D counts MULTIPLETS under :SU2 and STATES under :U1 -- compare at matched chi")
    flush(stdout)
    for sym in syms
        set_symmetry!(sym)
        @printf("\n   -- symmetry = %s --\n", sym)
        flush(stdout)
        # energies[(scheme, n)][Lx] -- one total energy per length, per arm, per step count
        energies = Dict{Tuple{String, Int}, Dict{Int, Float64}}()
        stats    = Dict{Tuple{String, Int}, Vector{Any}}()
        for Lx in Lxs
            L = Lx * Ly
            Jm = square_cylinder_couplings(Lx, Ly; J = 1.0, periodic_y = true)
            # ⚠ THE SINGLE-MATRIX `pair_mpo` IS WHAT MAKES THIS SYMMETRY-PARAMETRIC. It applies
            # the same `J[i,j]` to EVERY vertex scaled by `vertex.coeff`, and `spin_vertices`
            # returns whatever vertex set the active mode has -- one `S.S` vertex under `:SU2`,
            # three (`Sp`, `Sm`, `Sz`) under `:U1`. Same Hamiltonian either way, no per-mode
            # branch, and the coefficient conventions stay inside `spin_vertices` where they are
            # already pinned by tests.
            W = pair_mpo(L, Jm, spin_vertices(L))
            # ⛔ THE DIMER, IN BOTH MODES, so the initial state is IDENTICAL across the symmetry
            # axis. `:SU2` cannot represent a Neel state at all (it is not an SU(2) eigenstate),
            # so using Neel for `:U1` and dimer for `:SU2` would confound the symmetry comparison
            # with a different starting point.
            psi0 = dimer_state(L)
            obs(p) = real(mpo_energy(copy(p), W)) / max(norm(p)^2, eps())
            for (nm, st) in steppers(W, D, maxiter, true; arms = arms)
                for n in ns
                    o, k, c, s, mn, fin = run_arm(st, psi0, log_steps(tau_0, tau_max, n), obs;
                                                  imag_time = true)
                    @printf("   %-4s Lx=%-3d %-9s n=%-3d  E=%-16.10g  mv=%-7d chi=%-4d %7.1fs\n",
                            sym, Lx, nm, n, o, k, c, s)
                    flush(stdout)
                    get!(energies, (nm, n), Dict{Int, Float64}())[Lx] = o
                    push!(get!(stats, (nm, n), Any[]), (k, c, s, mn, fin))
                end
            end
        end
        # ── the fit: slope of E against Lx, divided by Ly, is the bulk energy per site ───────
        @printf("\n   -- %s bulk fit  E(Lx) = e_bulk * Ly * Lx + 2 * e_edge --\n", sym)
        for ((nm, n), byLx) in sort(collect(energies); by = first)
            xs = Float64[]; ys = Float64[]
            for Lx in Lxs
                haskey(byLx, Lx) && isfinite(byLx[Lx]) && (push!(xs, Lx); push!(ys, byLx[Lx]))
            end
            # need two lengths to have a slope at all; a single surviving arm cannot be fitted
            if length(xs) < 2
                @warn "cylinder: arm has <2 finite lengths, no bulk fit" sym scheme=nm nsteps=n
                continue
            end
            A = hcat(xs, ones(length(xs)))
            coef = A \ ys
            ebulk = coef[1] / Ly
            resid = maximum(abs.(A * coef .- ys))
            st = stats[(nm, n)]
            kry = sum(x[1] for x in st); chi = maximum(x[2] for x in st)
            sec = sum(x[3] for x in st); mn = maximum(x[4] for x in st)
            fin = all(x[5] for x in st)
            # ⚠ TWO POINTS DEFINE A LINE, so `resid` is identically ~1e-16 whenever `Lxs` has two
            # entries and proves NOTHING about being in the asymptotic regime. It only becomes a
            # test at three or more lengths -- which is why the l16 mode uses three.
            # ⛔ "DO NOT QUOTE" MUST REACH THE CSV, NOT JUST THE CONSOLE. This warning used to be
            # printed and then `emit`ted as an ordinary `err`, so the FILE recorded
            # `err=2.756969e-03` against `ref=-0.683282` with nothing marking it unusable -- and a
            # CSV outlives the terminal it was produced in. MEASURED at `Lxs = (2,3,4)`: every arm
            # gave `resid = 8.0e-03` and a 2.757e-03 "deviation from QMC" that is pure finite-length
            # artifact. Read off the file alone that is indistinguishable from a validated result.
            #
            # ⚠ `obs` AND `ref` ARE STILL WRITTEN -- only `err` becomes NaN. The fitted `e_bulk` is
            # a real measurement worth keeping (all four integrators agree on it to 10 digits); what
            # is not available is its DISTANCE FROM THE PUBLISHED NUMBER, because the fit is not yet
            # in the asymptotic regime. `plot_logtime.py` drops NaN rows AND reports the count on
            # the figure, so the omission is visible rather than silent.
            #
            # ⛔ THE REAL FIX IS LONGER CYLINDERS, NOT A LOOSER THRESHOLD. `Lxs = (2,3,4)` exists to
            # hold the run to 16 sites; the QMC bulk density needs `Lxs = (4,6,8)` (16/24/32 sites),
            # which is what `model_cylinder`'s DEFAULTS and `submit_logtime_suite.slurm` already run.
            trust = length(xs) >= 3 && resid <= 1e-6
            @printf("   %-4s %-9s n=%-3d e_bulk=%-14.10g QMC=%-11.6f dev=%-10.3e resid=%.1e%s%s\n",
                    sym, nm, n, ebulk, eref, abs(ebulk - eref), resid,
                    length(xs) < 3 ? " [2pt: resid meaningless]" : "",
                    trust ? "" : "  <- NOT LINEAR, do not quote")
            flush(stdout)
            emit(io, "heis_cylinder", String(sym), Ly, "log", n, nm, ebulk, eref,
                 kry, chi, sec, mn, fin;
                 err = trust ? nothing : NaN,
                 tag = trust ? "" : @sprintf("  <- FIT NOT LINEAR (resid %.1e), err=NaN", resid))
        end
    end
end

# ── 3. XXZ NEEL QUENCH, U(1), REAL time. Log grid FAILS by ALIASING. ────────────────────────
#
# REFERENCE: Barmettler, Punk, Gritsev, Demler, Altman, arXiv:0810.4845 (PRL 102, 130603 (2009)),
# "Relaxation of antiferromagnetic order in spin-1/2 chains following a quantum quench".
#
#     H = J sum_j [ S^x_j S^x_{j+1} + S^y_j S^y_{j+1} + Delta S^z_j S^z_{j+1} ]
#     |psi_0> = |up down up ... >                      (perfect Neel)
#     m_s(t)  = (1/N) sum_j (-1)^j <psi_0| S^z_j(t) |psi_0>
#
# and at `Delta = 0` the EXACT closed form
#
#     m_s(t) = J_0(2 J t) / 2   ~   t^{-1/2} cos(2 J t - pi/4)
#
# ⛔ THE REFERENCE USED HERE IS THE FINITE-L FREE-FERMION SOLUTION, NOT THE BESSEL FUNCTION. The
# Bessel form is the THERMODYNAMIC limit; our chain is finite and open, so it has boundary
# reflections the Bessel curve does not, and scoring against it would charge the integrator for a
# finite-size effect. `xx_free_fermion_sz(L, t; occupied = 1:2:L)` is exact for the chain we
# actually run. The Bessel asymptote is still what makes the ALIASING claim quantitative -- see
# below -- and the two must agree in the chain's interior at short time, which is the cheapest
# check that the coupling convention is right.
#
# ⛔ WHY THIS IS THE ALIASING DEMONSTRATION AND NOT AN ASSERTION. The published asymptote
# `cos(2 J t - pi/4)` has period `pi / J`. At `J = 1` a grid whose step exceeds `pi/2 ~ 1.571`
# cannot resolve the oscillation AT ALL -- Nyquist, on a frequency somebody else measured. The log
# grid reaching `t_max = 16` in `n = 12` steps has a LAST step of ~6.3, four times over. So the
# failure is arithmetic about a known frequency; the uniform arms are there to show the same
# integrator succeeding when the spacing respects it, not to define what "right" means.
#
# ⛔ AND THE ERROR REPORTED FOR THIS MODEL IS THE **TRAJECTORY** ERROR, NOT THE ENDPOINT. Scoring
# `m_s(t_max)` alone said the OPPOSITE -- see [`trajectory_error`](@ref) for the measurement that
# forced this. A single huge step still lands near the right state at that instant, because `expv`
# is exact for the local generator; what it cannot do is report what happened in between. The
# endpoint value and the exact endpoint are still written to the `obs` and `ref` columns, so both
# readings stay available and the difference between them is itself part of the result.
#
# ⚠ `Delta = 1` IS THE INTERESTING CASE AND HAS NO CLOSED FORM. The paper's headline result is
# that the relaxation time has a MINIMUM near `Delta = 1` (tau ~ 1/J), against classical critical
# slowing down. That arm is run and reported, but scored against the paper's published curve by
# eye rather than by a number -- it is a recreate attempt, not a certified error.
#
# ⛔ THE STEP COUNTS BRACKET THE NYQUIST CROSSOVER, WHICH IS THE WHOLE MEASUREMENT. Computed from
# the published period and this file's own `log_steps`: the largest step of a log grid on
# `[0.02, 16]` first drops under `pi/2` at **n = 66**, whereas a UNIFORM grid needs only
# `ceil(16 / (pi/2)) = 11` steps for the same bound.
#
#     log n     12      24      50   |   100     200
#     max step  7.29    4.04    2.04 |   1.04    0.53
#               ---- ALIASES ----    |   ---- resolves ----
#
# So the log grid costs ~6x MORE steps than uniform here -- the exact inversion of
# arXiv:2606.02930's imaginary-time advantage, and derived from Barmettler's frequency rather
# than from any run of ours. Arms are placed either side of 66 so the crossover is MEASURED; a
# set of step counts entirely on one side would only re-assert what the arithmetic already says.
# The uniform `n = 11` arm sits just under the bound on purpose: Nyquist is NECESSARY, not
# sufficient, and that arm is what shows the remaining dt error.
function model_neel(io; L = 20, D = 128, t_max = 16.0, t_0 = 0.02, J = 1.0,
                    deltas = (0.0, 1.0), maxiter = 40,
                    logs = (12, 24, 50, 100, 200), unis = (11, 50, 200),
                    arms = DEFAULT_ARMS)
    set_symmetry!(:U1)
    psi0 = neel_state(L)
    # ⛔ PIN THE STAGGERING SIGN AGAINST THE INITIAL STATE INSTEAD OF GUESSING IT. Barmettler's
    # `m_s(0)` is `+1/2`; whether that is `(-1)^j` or `(-1)^(j+1)` depends on which sublattice
    # `neel_state` puts up, and getting it backwards flips the whole curve into a mirror image
    # that still decays and still oscillates -- a wrong answer that looks like a right one.
    sz0 = magnetisation(copy(psi0))
    sgn = sum((-1)^(j + 1) * sz0[j] for j in 1:L) > 0 ? +1 : -1
    m_s(sz) = sgn * sum((-1)^(j + 1) * sz[j] for j in 1:L) / L
    # ⛔ AND THE REFERENCE'S OCCUPIED SITES ARE READ OFF THE SAME STATE, NOT HARDCODED. `sgn` above
    # fixes the sign of OUR m_s, but the free-fermion reference needs the SUBLATTICE. Writing
    # `occupied = 1:2:L` would silently describe the MIRROR state whenever `neel_state` puts the
    # up spins on the even sublattice -- and the mirror evolves to the mirror profile, which
    # decays and oscillates exactly like the right one. `S^z = n - 1/2`, so up spin = occupied.
    occ = [j for j in 1:L if sz0[j] > 0]
    length(occ) == L ÷ 2 || error("neel_state($L) has $(length(occ)) up spins, expected $(L ÷ 2)")
    @printf("\n== 3. XXZ NEEL QUENCH U(1)  L=%d D=%d t_max=%g (REAL time)   m_s(0)=%+.4f\n",
            L, D, t_max, m_s(sz0))
    println("   reference: Barmettler et al. arXiv:0810.4845;  Delta=0 exact free fermion")
    nyq = pi / (2J)
    @printf("   published asymptote cos(2Jt - pi/4): period %.4f, Nyquist step %.4f\n",
            pi / J, nyq)
    # ⛔ PRINT THE VERDICT PER ARM BEFORE RUNNING IT. Which grids can resolve the oscillation is
    # decided by arithmetic on a PUBLISHED frequency, not by the numbers that come back -- so it
    # is stated up front. A run that then disagrees with this table is a bug in the grid, and a
    # run that agrees is not evidence of anything: the point is that the ERROR should track it.
    for (g, ns) in (("log", logs), ("uniform", unis))
        for n in ns
            dts = g == "log" ? log_steps(t_0, t_max, n) : uniform_steps(t_max, n)
            b = maximum(dts)
            @printf("     %-8s n=%-4d largest step %8.4f   %s\n",
                    g, n, b, b <= nyq ? "resolves" : "ALIASES")
        end
    end
    flush(stdout)
    for Delta in deltas
        xy = [i + 1 == j ? J : 0.0 for i in 1:L, j in 1:L]
        zz = [i + 1 == j ? J * Delta : 0.0 for i in 1:L, j in 1:L]
        # ⛔ THE ZZ MATRIX IS THIRD. `xxz_chain` pushes the two XY channels and THEN `SzSz`, and
        # `pair_mpo` pairs `Js` with `vertices` BY POSITION, so a different order silently builds
        # a different Hamiltonian.
        W = pair_mpo(L, AbstractMatrix[xy, xy, zz], spin_vertices(L))
        # ⛔ DIVIDE BY THE NORM even though a Hermitian real-time evolution conserves it: an arm
        # that is DIVERGING does not, and an unnormalised observable would then report the
        # divergence as a physics value. `sz_expectation` contracts a canonicalised state against
        # itself, so it carries the norm^2 and is NOT already normalised.
        obs(p) = m_s(magnetisation(copy(p))) / max(norm(p)^2, eps())
        # exact only at Delta = 0; at Delta = 1 there is no closed form and `ref` is NaN, which
        # makes `err` NaN -- deliberately, so no accuracy claim can be read off that row.
        eref = Delta == 0.0 ?
               m_s(xx_free_fermion_sz(L, t_max; J = J, occupied = occ)) : NaN
        @printf("\n   -- Delta = %g   reference %s --\n", Delta,
                Delta == 0.0 ? @sprintf("m_s(%g) = %+.10f (exact)", t_max, eref) :
                               "none (no closed form; recreate Barmettler Fig. by eye)")
        flush(stdout)
        # The exact curve at ANY t, for the trajectory metric. `O(L^3)` per evaluation and 401
        # evaluations per arm, which is nothing beside one sweep.
        exact_ms = Delta == 0.0 ?
                   (t -> m_s(xx_free_fermion_sz(L, t; J = J, occupied = occ))) : nothing
        tag = @sprintf("neel_d%g", Delta)
        for (nm, st) in steppers(W, D, maxiter, false; arms = arms)
            for (g, ns) in (("log", logs), ("uniform", unis))
                for n in ns
                    dts = g == "log" ? log_steps(t_0, t_max, n) : uniform_steps(t_max, n)
                    # ⚠ THE TRACE COSTS ONE OBSERVABLE PER STEP, which for `magnetisation` is a
                    # full centre sweep. That is the price of measuring a curve instead of a
                    # point, and it is charged to every arm equally so the `seconds` column stays
                    # a fair comparison BETWEEN arms -- it is not comparable with the
                    # imaginary-time models, which take no trace.
                    tr = Tuple{Float64, Float64}[]
                    o, k, c, s, mn, fin = run_arm(st, psi0, dts, obs;
                                                  imag_time = false, trace = tr)
                    terr = exact_ms === nothing ? NaN : trajectory_error(tr, exact_ms)
                    emit(io, tag, "U1", L, g, n, nm, o, eref, k, c, s, mn, fin;
                         err = terr,
                         tag = maximum(dts) <= nyq ? "" : "  [aliases]")
                end
            end
        end
    end
end

# ⛔ SELECT MODELS FROM THE COMMAND LINE, AND APPEND RATHER THAN TRUNCATE.
#
# `julia --project=. benchmarks/imaginary_time/logtime_suite.jl cylinder neel`
#
# Two reasons, both learned the hard way. (1) A completed model's rows are EXPENSIVE -- model 1 is
# twelve SU(2) arms and roughly an hour -- and re-running the whole file to reach model 3 both
# wastes that hour and OVERWRITES results already in hand. (2) The runs keep dying part-way, so
# being able to resume at the model that failed is the difference between progress and starting
# over. With no arguments the behaviour is the old one: all models, fresh file.
#
# ⛔ AND `smoke` IS NOT AN OPTIMISATION, IT IS THE ONLY THING THAT SHOULD RUN LOCALLY. The
# standing rule here is that a local run means `L <= ~12` and the question is CORRECTNESS against
# an exact reference; every full size, every sweep and ALL wall-clock numbers belong on the
# cluster. `smoke` shrinks all three models to that regime so the wiring -- references, MPO
# layouts, observable conventions, the bulk fit -- can be proven cheaply, and the numbers it
# produces are for VALIDATION ONLY and must never be quoted as results.
const SMOKE = "smoke" in ARGS
const WANT = let a = filter(!=("smoke"), ARGS)
    isempty(a) ? nothing : Set(a)
end
const APPEND = WANT !== nothing && isfile(OUT)

# ⛔ `l16` IS THE SIZE THE PANEL AND KNOB STUDIES ARE QUOTED AT, and it is a separate mode from
# `smoke` rather than a tweak to it. `smoke` answers "is the wiring right" at `L <= 12` against an
# exact reference; `l16` is the actual comparison size, big enough that the arms separate and
# small enough to finish. The cylinder's `l16` means `Lx = 4` at `Ly = 4` -- 16 sites -- with the
# two shorter lengths kept because the QMC comparison is a SLOPE and needs three points.
const L16 = "l16" in ARGS

# ⛔ `l16cap` EXISTS BECAUSE `l16` CANNOT DISCRIMINATE BETWEEN INTEGRATORS AT ALL, AND THAT IS A
# MEASURED FACT ABOUT IT, NOT A SUSPICION. At L = 16 the Heisenberg chain saturates at `chi = 46`
# against a `D = 128` cap, so NOTHING IS EVER TRUNCATED and every arm returns the same answer:
# MEASURED `bug_rsvd` 1.4-4.0e-14, `bug_par` 1.1-5.2e-14, `midpoint` 1.2-4.3e-14 -- one shared
# machine-precision floor across twenty-three runs. On that grid the only real column is `matvec`.
#
# An accuracy comparison needs the cap to BIND. `D = 20` sits well below the natural 46, so the
# truncation each arm chooses is what the error measures -- which is the question "is BUG more
# accurate than 2-site TDVP" actually asks. ⚠ Quoting an accuracy ranking from the UNCAPPED run
# would be reporting the double-precision floor as a method difference.
const L16CAP = "l16cap" in ARGS
const CAPD = parse(Int, get(ENV, "LOGTIME_D", "20"))

# ⚠ THE CHAIN'S `P` SCAN IS DELIBERATELY COARSER THAN IT WAS. The DENSE sweep
# `Ps = (1,2,3,4,5,6,8,10)` has already been run to completion for `bug_rsvd`
# (`results/logtime_l16b.csv`) and settled the panel question: every `P` sits on the SAME
# machine-precision floor (2.1e-14 .. 4.0e-14) while cost climbs linearly, so the minimum is
# `P = 1` and what `P` actually buys is REPRESENTABILITY (`|psi|max` 3.05e+136 on the log grid vs
# 3.26e+22 for every `P >= 3`), not accuracy. Re-running that scan for three more arms would spend
# hours confirming the same floor from four directions. Four `P` values keep a panel axis on every
# arm for the figure; the resolution lives in the file above.
l16_kw(model) =
    model == "chain"    ? (; L = 16, D = 128, Ps = (1, 2, 4, 8)) :
    model == "cylinder" ? (; Ly = 4, Lxs = (2, 3, 4), D = 128, ns = (8, 12)) :
                          (; L = 16, D = 128)

smoke_kw(model) = !SMOKE ? (L16 ? l16_kw(model) : NamedTuple()) :
    model == "chain"    ? (; L = 8, D = 32, tau_max = 30.0, maxiter = 20, Ps = (1, 2, 4)) :
    model == "cylinder" ? (; Ly = 4, Lxs = (2, 3), D = 48, tau_max = 20.0, maxiter = 20,
                             ns = (8,)) :
                          (; L = 10, D = 48, t_max = 8.0, maxiter = 20,
                             logs = (12, 100), unis = (11, 100))

# ⚠ `bug_par` LAUNCHES THE TWO HALF-SWEEPS AS TASKS, so on one worker it measures the sequential
# sweep PLUS scheduling overhead -- a small pessimisation reported as a method difference. Say so
# once, loudly, rather than letting a slow `bug_par` row be read as evidence against the parallel
# sweep.
if "bug_par" in DEFAULT_ARMS && Threads.nthreads() < 2
    @warn """`bug_par` is in the arm list but this process has $(Threads.nthreads()) thread(s).
             Its wall-clock column is MEANINGLESS here -- relaunch with `julia -t 2` (or more)
             before quoting any parallel speedup.""" nthreads=Threads.nthreads()
end

mkpath(dirname(OUT))
open(OUT, APPEND ? "a" : "w") do io
    APPEND || println(io, join(COLS, ","))
    SMOKE && println("\n*** SMOKE MODE -- wiring validation only, NOT results ***")
    # ⛔ ONE MODEL FAILING MUST NOT TAKE THE SUITE WITH IT. This already happened: an exception out
    # of model 1's smallest arm meant models 2-4 were never reached at all, so four models' worth
    # of compute produced one model's worth of rows. `run_arm` already absorbs per-ARM failures;
    # this absorbs a failure in the model SETUP (an MPO builder, a reference, a symmetry mode).
    #
    # ⚠ ALL THREE RUN IN ONE PROCESS DELIBERATELY: SU(2) specialisation costs ~165-350 s per
    # distinct code path PER PROCESS, so splitting these into three jobs would pay it three times.
    for (nm, f) in ("chain" => model_chain, "cylinder" => model_cylinder,
                    "neel" => model_neel)
        WANT === nothing || nm in WANT || continue
        try
            f(io; smoke_kw(nm)...)
        catch err
            @error "model failed entirely; continuing with the rest" model=nm err
            flush(stdout)
        end
    end
end
println("\nwrote ", OUT)
