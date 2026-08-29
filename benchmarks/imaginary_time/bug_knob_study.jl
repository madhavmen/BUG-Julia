# WHERE THE rSVD SWEEP'S SPEEDUP COMES FROM, AND HOW THE TWO TRUNCATIONS INTERACT.
#
# Two studies, on the same three models as `benchmarks/imaginary_time/logtime_suite.jl`:
#
#   A  rSVD KNOBS      one factor at a time from a baseline: `dover`, `growth`, `comp_ratio`,
#                      `share_ht`, `parallel`, and `exact` as the A/B control.
#   B  TRUNCATIONS     the FULL CROSS of the half-sweep split against the root split, because the
#                      question is their INTERACTION and a one-at-a-time scan cannot see it.
#
# ⛔ THE TWO STUDIES NEED DIFFERENT REFERENCES AND MIXING THEM UP MAKES BOTH MEANINGLESS.
#
#   Study A asks: does the RANDOMISED sketch reproduce the DETERMINISTIC one more cheaply? That is
#   a question about the sketch, so the control is `exact = true` AT THE SAME SETTINGS -- not
#   nature. A row that differs from exact CBE by 1e-13 has reproduced it; whether both are 1e-6
#   from the true answer is a different question, answered by the suite.
#
#   Study B asks: how much accuracy does each truncation cost? That IS a question about nature, so
#   it is scored against the model's EXACT reference (Bethe / free fermion). Scoring it against
#   another of our own runs would only measure how far two truncations sit apart.
#
# ⛔ THERE ARE EXACTLY THREE TRUNCATIONS IN A STEP AND THEY ARE NOT INTERCHANGEABLE. From
# `cbe_bug_step!`'s docstring:
#
#   split_cutoff / split_maxdim   the HALF-SWEEP frame split, applied once per bond as the sweep
#                                 passes. Default 1e-14 / 0.
#   root_cutoff  / root_maxdim    the ROOT CORE split -- the SVD that separates the Galerkin core
#                                 into the two centre tensors. Default 0.0 / 0, i.e. NONE.
#   trunc_thresh / maxdim         the closing `truncate_recursive!`. Default 1e-12 / 200.
#
# The root split defaulting to NONE is the load-bearing fact for study B: the sweep is designed to
# have TWO truncations, and turning the third on is a deliberate choice whose cost is what the
# cross measures. A study that varied `trunc_thresh` at the same time could not attribute anything.
#
# Run:  julia -t 2 --project=. benchmarks/imaginary_time/bug_knob_study.jl [A|B] [chain|cylinder|neel]
# Out:  benchmarks/results/bug_knob_A.csv , bug_knob_B.csv

using LinearAlgebra, Printf, Statistics
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "fused_common.jl"))
include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))
include(joinpath(@__DIR__, "..", "..", "tests", "common", "free_fermion.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "results"); mkpath(OUTDIR)

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0
renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

log_steps(t0, tmax, n) = (r = (tmax / t0)^(1 / (n - 1));
                          ts = [t0 * r^(k - 1) for k in 1:n]; vcat(ts[1], diff(ts)))

"""
One model, reduced to what a knob study needs: an MPO, a start state, an observable, an EXACT
reference (or `NaN`), a step list, and whether it is imaginary time.

⚠ SIZES ARE DELIBERATELY SMALL. A knob study is a comparison BETWEEN CONFIGURATIONS at fixed
model, so it wants many cheap points rather than few expensive ones; the absolute accuracy at
these sizes is not the deliverable and must not be quoted as one.
"""
function build(model::String; smoke::Bool = false)
    if model == "chain"
        set_symmetry!(:SU2)
        L = smoke ? 8 : 16
        W = heisenberg_su2_mpo(L)
        e, _, res = bethe_xxx_open_energy(L)
        res < 1e-12 || error("Bethe residual $res")
        return (; name = "chain", W = W, psi0 = dimer_state(L), L = L, herm = true,
                obs = p -> real(mpo_energy(copy(p), W)) / max(norm(p)^2, eps()),
                eref = e, steps = log_steps(0.05, 30.0, 12), imag_time = true, D = 64)
    elseif model == "cylinder"
        set_symmetry!(:SU2)
        Ly = 4; Lx = smoke ? 3 : 4; L = Lx * Ly
        Jm = square_cylinder_couplings(Lx, Ly; J = 1.0, periodic_y = true)
        W = pair_mpo(L, Jm, spin_vertices(L))
        # ⛔ NO EXACT REFERENCE AT FINITE Lx. The QMC number is a BULK density reached by a
        # multi-length fit (see `logtime_suite.jl`), which is far too expensive to redo inside a
        # knob scan. `NaN` here means study B cannot score this model against nature -- study A
        # still can, because its control is the exact-CBE arm at the same size.
        return (; name = "cylinder", W = W, psi0 = dimer_state(L), L = L, herm = true,
                obs = p -> real(mpo_energy(copy(p), W)) / max(norm(p)^2, eps()),
                eref = NaN, steps = log_steps(0.05, 20.0, 12), imag_time = true, D = 96)
    elseif model == "neel"
        set_symmetry!(:U1)
        L = smoke ? 10 : 16
        J = 1.0; tmax = 8.0
        xy = [i + 1 == j ? J : 0.0 for i in 1:L, j in 1:L]
        zz = zeros(Float64, L, L)                       # Delta = 0: the EXACTLY SOLVABLE arm
        W = pair_mpo(L, AbstractMatrix[xy, xy, zz], spin_vertices(L))
        psi0 = neel_state(L)
        sz0 = magnetisation(copy(psi0))
        sgn = sum((-1)^(j + 1) * sz0[j] for j in 1:L) > 0 ? +1 : -1
        ms(sz) = sgn * sum((-1)^(j + 1) * sz[j] for j in 1:L) / L
        occ = [j for j in 1:L if sz0[j] > 0]
        return (; name = "neel", W = W, psi0 = psi0, L = L, herm = true,
                obs = p -> ms(magnetisation(copy(p))) / max(norm(p)^2, eps()),
                eref = ms(xx_free_fermion_sz(L, tmax; J = J, occupied = occ)),
                # uniform and UNDER Nyquist (pi/2): a knob study must not be confounded by the
                # aliasing that `logtime_suite.jl` is separately measuring.
                steps = fill(tmax / 100, 100), imag_time = false, D = 96)
    elseif model == "dissipative"
        # ⛔ THE OPEN-SYSTEM GENERATOR IS WHY THIS MODEL IS IN A **KNOB** STUDY AT ALL.
        # `PLAN-open-systems.md` §7 puts rSVD last precisely because the closed models' MPOs are
        # too NARROW for a sketch to pay off -- `npre/(d*chi)` never gets small. The fused
        # Liouvillian is the first generator here wide enough (d = 4, and a jump channel per
        # boundary on top of the doubled Hamiltonian), so this is the model where a fixed-`dex`
        # rSVD arm has a chance of beating the deterministic one.
        set_symmetry!(:none)
        L = smoke ? 3 : 8
        W, shift = fused_boundary_lindblad(L; J = 1.0, eps_L = 1.0, eps_R = 1.0,
                                           mu_L = 1.0, mu_R = -1.0)
        Imps = identity_superket(L)
        # ⛔ THE MAXIMALLY MIXED STATE IS `|I>> / 2^(L/2)`. `identity_superket` is NORMALISED, so
        # its trace is `2^(L/2)` (16 at L = 8) -- and feeding that in as `rho` makes
        # `restore_trace!` see a deficit of `1 - 2^(L/2)` and project the state straight back onto
        # `|I>>` on every step. The run then reports a profile of EXACTLY zero, `chi = 1`,
        # `Tr = 1.0000000000`, identically for every arm and at every `dt`. See
        # `dissipative_xx.jl`'s `evolve_arm` for the full symptom list; `restore_trace!` now throws
        # above a 0.5 deficit rather than silently doing this.
        rho0 = copy(Imps); apply_shift!(rho0, 1.0 / 2.0^(L / 2))
        let t0 = real(trace_rho(Imps, rho0, L))
            abs(t0 - 1) < 1e-10 || error("dissipative psi0 has Tr = $t0, expected 1")
        end
        # ✅ THE EXACT TIME-DEPENDENT REFERENCE NOW EXISTS, so study B IS quotable on this model.
        # `xx_boundary_sz` is the analytic Lyapunov solution (`O(L^3)`, no dense object), certified
        # against a dense Liouvillian to 8.9e-16 at L = 2..5 in `certify_xx_reference.jl`. The
        # observable is `<S^z_1>`, so the reference is that profile's first entry at `t = sum(steps)`.
        return (; name = "dissipative", W = W, psi0 = rho0, L = L, herm = false,
                obs = p -> (t = real(trace_rho(Imps, p, L));
                            abs(t) < 1e-12 ? NaN : real(trace_rho(Imps, p, L; at = 1)) / t),
                eref = xx_boundary_sz(L, 2.0; J = 1.0, eps_L = 1.0, eps_R = 1.0,
                                      mu_L = 1.0, mu_R = -1.0)[1],
                steps = fill(0.02, 100), imag_time = false, D = 128,
                # ⛔ growth 4, NOT the shipped 2.0: on a d = 4 chain `growth = 2.0` caps the
                # achievable ORDER at 1 (PLAN §3.7) while `Tr rho` still reads 0.99999.
                growth = 4.0, shift = shift, Imps = Imps, liouville = true)
    end
    throw(ArgumentError("unknown model $model"))
end

"""
Run one configuration; returns `(obs, matvec, chi, seconds, ok)`.

⛔ THREE DIFFERENT `tau` CONVENTIONS LIVE HERE AND MIXING THEM EVOLVES A DIFFERENT GENERATOR
WITHOUT ERRORING:

    imaginary time   tau = -dt        renormalise after each step
    real time        tau = -i*dt      do NOT renormalise (the norm is physics)
    LIOUVILLIAN      tau = +dt        the MPO already carries the -i, so it IS the generator;
                                      `exp(+t L)` is the propagator, and the zero-body scalar and
                                      the trace restoration are applied per step

⚠ THE MODEL'S OWN `growth` WINS OVER THE GLOBAL DEFAULT BUT LOSES TO THE VARIANT. A d=4 fused
chain needs `growth = 4.0` (PLAN §3.7), while a variant that is explicitly scanning `growth` must
still override it -- otherwise the scan silently measures one value on that model.
"""
function run_cfg(m, kw)
    psi = copy(m.psi0); mv = 0; t0 = time()
    liou = hasproperty(m, :liouville) && m.liouville
    kw = merge((growth = hasproperty(m, :growth) ? m.growth : 2.0,), kw)
    try
        for dt in m.steps
            tau = liou ? ComplexF64(dt) :
                  m.imag_time ? ComplexF64(-dt) : ComplexF64(-im * dt)
            mv += _kry(cbe_bug_step!(psi, m.W, tau; maxiter = 40, hermitian = m.herm,
                                     maxdim = m.D, kw...))
            if liou
                apply_shift!(psi, exp(dt * m.shift))
                restore_trace!(m.Imps, psi, m.L; maxdim = m.D)   # §3.11: along |I>>, not rescale
            elseif m.imag_time
                renorm!(psi)
            end
        end
        o = m.obs(psi)
        return (isfinite(o) && o != 0.0 ? o : NaN), mv, maximum(bond_dims(psi)),
               time() - t0, true
    catch err
        @info "config broke (recorded NaN)" kw err
        return NaN, mv, 0, time() - t0, false
    end
end

# ── STUDY A: where the rSVD speedup comes from ───────────────────────────────────────────────
#
# ⛔ ONE FACTOR AT A TIME FROM A FIXED BASELINE, not a full grid. The full cross of the five knobs
# is ~100 configurations per model and would answer a question nobody asked; what is wanted is
# WHICH KNOB MOVES THE CLOCK, and that is a main-effects question. The pairs that are expected to
# interact (`share_ht` x `parallel`, both being pure-performance switches that touch the same
# sketch) are added explicitly rather than inferred.
#
# ⚠ `share_ht = false` AND `share_ht = true` MUST AGREE TO ROUNDOFF. It is a performance fix, not
# a knob: `cbe_expand` used to rebuild `H*Theta` for each of its three sketch calls. If those two
# rows differ in the observable, the sharing is WRONG and every timing below is uninterpretable --
# so that agreement is checked first and reported as a gate.
function study_A(io, m)
    # ⛔ `growth` IS DELIBERATELY ABSENT FROM THE BASELINE. Variants merge OVER `base`, and
    # `run_cfg` lets the variant win over the model's own default -- so putting `growth = 2.0`
    # here would silently override the d=4 chain's MANDATORY `growth = 4.0` (PLAN §3.7) on every
    # row except the two that scan `growth` explicitly, i.e. exactly the model where the shipped
    # value caps the achievable order at 1. Omitting it lets each model's own default stand.
    base = (exact = false, dover = 4, comp_ratio = 1.0,
            share_ht = true, parallel = false)
    # ⛔ `dex` IS THE KNOB THAT DECIDES WHETHER rSVD CAN WIN AT ALL, and it is easy to leave out of
    # a scan because it defaults to 0 (= "use the growth schedule"). `PLAN-open-systems.md` §7:
    # the sketch pays off only when `npre/(d*chi)` is SMALL, which needs a FIXED per-side budget at
    # large `chi` -- and the growth schedule gives the opposite, `budget = ceil(growth*dmax) - r`,
    # i.e. a CONSTANT FRACTION of full (about 60% at `growth = 2.0`). A scan that moved only
    # `growth`, `dover` and `comp_ratio` would conclude "rSVD never wins" while never once putting
    # it in the regime where it could. `cbe_core.jl:501` -- "any positive value is an absolute
    # per-side budget and overrides the schedule entirely".
    #
    # ⚠ SO THE `dex` ROWS ARE NOT INTERCHANGEABLE WITH THE `growth` ROWS: they are the same axis
    # measured in different units (absolute vs proportional), and only the absolute one can shrink
    # as a fraction of `chi`. Read them against the `chi` column, not against each other.
    variants = Pair{String, NamedTuple}[
        "baseline"          => NamedTuple(),
        "exact (A/B ctrl)"  => (; exact = true),
        "dover=auto"        => (; dover = nothing),
        "dover=2"           => (; dover = 2),
        "dover=8"           => (; dover = 8),
        "growth=1.5"        => (; growth = 1.5),
        "growth=3.0"        => (; growth = 3.0),
        # fixed ABSOLUTE per-side budgets -- the regime where npre/(d*chi) can actually fall
        "dex=4 (fixed)"     => (; dex = 4),
        "dex=8 (fixed)"     => (; dex = 8),
        "dex=16 (fixed)"    => (; dex = 16),
        "dex=8 exact"       => (; dex = 8, exact = true),   # same budget, DETERMINISTIC sketch
        "comp_ratio=0.5"    => (; comp_ratio = 0.5),
        "share_ht=false"    => (; share_ht = false),
        "parallel"          => (; parallel = true),
        "preselect_only"    => (; preselect_only = true),
    ]
    @printf("\n== STUDY A  rSVD knobs  --  model=%s L=%d D=%d steps=%d threads=%d\n",
            m.name, m.L, m.D, length(m.steps), Threads.nthreads())
    println("   control = `exact = true` at the same settings; err_vs_exact is the deliverable")
    flush(stdout)
    ctrl = nothing
    @printf("   %-18s %-16s %-12s %-9s %-8s %-8s %s\n",
            "variant", "obs", "err_vs_exact", "matvec", "chi", "secs", "speedup")
    rows = Any[]
    for (nm, over) in variants
        kw = merge(base, over)
        o, mv, chi, sec, ok = run_cfg(m, kw)
        nm == "exact (A/B ctrl)" && (ctrl = o)
        push!(rows, (nm, o, mv, chi, sec, ok))
    end
    bl = rows[1]
    for (nm, o, mv, chi, sec, ok) in rows
        dev = (ctrl === nothing || !isfinite(ctrl) || !isfinite(o)) ? NaN : abs(o - ctrl)
        sp = (isfinite(sec) && sec > 0) ? bl[5] / sec : NaN
        @printf("   %-18s %-16.10g %-12.3e %-9d %-8d %-8.2f %.2fx\n",
                nm, o, dev, mv, chi, sec, sp)
        @printf(io, "%s,A,%s,%.10g,%.6e,%d,%d,%.4f,%d\n",
                m.name, replace(nm, "," => ";"), o, dev, mv, chi, sec, ok ? 1 : 0)
        flush(io); flush(stdout)
    end
    # the gate, stated as a verdict rather than left for the reader to spot
    i_s = findfirst(r -> r[1] == "share_ht=false", rows)
    if i_s !== nothing && isfinite(rows[1][2]) && isfinite(rows[i_s][2])
        d = abs(rows[1][2] - rows[i_s][2])
        @printf("\n   GATE  share_ht true vs false: |d obs| = %.3e  %s\n", d,
                d < 1e-10 ? "OK (performance-only, as documented)" :
                            "<- FAILS: sharing changes the RESULT, timings uninterpretable")
    end
    flush(stdout)
end

# ── STUDY B: half-sweep split against root split, as a full cross ────────────────────────────
#
# ⛔ THE CROSS IS THE POINT. Each truncation alone looks like a monotone accuracy-for-rank trade;
# what a one-at-a-time scan cannot show is whether they are REDUNDANT (the tighter one dominates
# and the other is free) or COMPOUNDING (both bite, and loosening either costs accuracy the other
# cannot recover). Those two have opposite engineering consequences and the same one-dimensional
# slices.
#
# ⚠ `root_cutoff = 0.0` IS "NO ROOT TRUNCATION" AND IS THE DEFAULT, so it is the first column
# rather than an extreme: the question is what turning the third truncation ON costs, not where in
# a range it should sit.
function study_B(io, m; splits = (1e-14, 1e-12, 1e-10, 1e-8),
                 roots = (0.0, 1e-12, 1e-10, 1e-8))
    @printf("\n== STUDY B  half-sweep split x root split  --  model=%s L=%d D=%d steps=%d\n",
            m.name, m.L, m.D, length(m.steps))
    if isfinite(m.eref)
        @printf("   reference: EXACT, %.12g\n", m.eref)
    else
        println("   ⚠ no exact reference for this model -- err is NaN; use `chain` or `neel`")
    end
    flush(stdout)
    @printf("\n   %-12s", "split \\ root")
    for r in roots; @printf(" %-13s", r == 0.0 ? "none" : @sprintf("%.0e", r)); end
    println("\n   " * "-"^(12 + 14 * length(roots)))
    for sc in splits
        @printf("   %-12s", @sprintf("%.0e", sc))
        for rc in roots
            o, mv, chi, sec, ok = run_cfg(m, (exact = false, dover = 4,
                                              split_cutoff = sc, root_cutoff = rc))
            e = isfinite(m.eref) ? abs(o - m.eref) : NaN
            @printf(" %-13s", @sprintf("%.2e/%d", e, chi))
            @printf(io, "%s,B,split=%.0e;root=%.0e,%.10g,%.6e,%d,%d,%.4f,%d\n",
                    m.name, sc, rc, o, e, mv, chi, sec, ok ? 1 : 0)
            flush(io)
        end
        println(); flush(stdout)
    end
    println("\n   cells are  err/chi.  Read DOWN a column for the half-sweep split's cost at " *
            "fixed root,\n   and ACROSS a row for the root split's cost at fixed half-sweep.")
    flush(stdout)
end

const WHICH  = isempty(ARGS) ? "A" : uppercase(ARGS[1])
const MODELS = length(ARGS) >= 2 ? ARGS[2:end] : ["chain", "neel"]
const SMOKE  = "smoke" in ARGS

# ⛔ THE MODEL IS IN THE FILENAME, AND `KNOB_OUT` OVERRIDES IT. This was
# `bug_knob_$(WHICH).csv` -- keyed on the STUDY only -- so two runs of the same study on different
# models wrote to the SAME file and the second silently destroyed the first. On a cluster array
# that is not hypothetical: `submit_bug_knob_study.slurm` runs `A chain` and `A cylinder` as
# CONCURRENT tasks, which would have interleaved their writes into rows that still parse and then
# reached a figure as if they were one run.
#
# ⚠ The filename now carries the models actually run, so a partial invocation cannot masquerade as
# a full one either.
out = get(ENV, "KNOB_OUT",
          joinpath(OUTDIR, "bug_knob_$(WHICH)_$(join(filter(!=("smoke"), MODELS), "-")).csv"))
open(out, "w") do io
    println(io, "model,study,variant,obs,err,matvec,chi,seconds,ok")
    for mn in MODELS
        mn == "smoke" && continue
        try
            m = build(mn; smoke = SMOKE)
            WHICH == "A" ? study_A(io, m) : study_B(io, m)
        catch err
            @error "model failed; continuing" model=mn err
            flush(stdout)
        end
    end
end
println("\nwrote ", out)
