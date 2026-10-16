# WHERE DOES THE ADAPTIVE CBE-BUG STEP SPEND ITS TIME AT LARGE BOND DIMENSION, AND HOW DOES ITS
# PER-STEP COST SCALE AGAINST 2-SITE TDVP? -- on the production workload of Phase 6.
#
# The workload is exactly the spectral driver's (`fig45_2209.jl`): the square Heisenberg cylinder
# built by the same coupling builder and `pair_mpo`, the adaptive CBE-DMRG ground state, S^z
# applied at `centre_site`, and the adaptive arms at the Task-4 tolerances with parallel
# half-sweeps. Nothing here is a random MPS: the adaptive engine sizes its sketches from the
# physical spectrum, so a flat-spectrum state would profile a different code path.
#
# Per cap D (ascending) and per arm it reports:
#   1. per-step WALL under the production configuration (`parallel = true`, threads 2), and the
#      ratio to the TDVP arm of the SAME order at the same dt (bug2/bugmid vs tdvp2, bug4 vs
#      tdvp4), with the pair's infidelity at the same simulated time -- the cost axis of the
#      "BUG beats TDVP at large chi" claim;
#   2. the PHASE BREAKDOWN from `CBEBugSweepInfo` (cbe / kry / split / frame / env / root /
#      trunc / closure) on ONE SERIAL step, where the phases are one clock and must sum to the
#      step -- the unattributed remainder is printed, not hidden;
#   3. a sampled `Profile` of that serial step: kernel buckets (gemm / svd / permute / GC / ...)
#      and the call sites in OUR source that own the samples (`owner_top`).
#
# Modes: BP_MODE=gs solves and caches the ground-state ladder only (accuracy-only, may run beside
# a timing job); BP_MODE=profile loads the cache (solving whatever is missing) and times.
#
# Run:  julia -t 2 --project=PKG benchmarks/spectral/bug_profile_2d.jl Lx Ly OUTDIR
# Env:  BP_MODE (profile) BP_DS ("256,512,1024") BP_ARMS ("tdvp2,bug2")
#       BP_PROF ("bug2"); tdvp4 = `tdvp2_composed_step!` order 4 (Suzuki, 5 stages) at BP_DT4,
#       the partner of bug4 (also at BP_DT4)
#       BP_NSTEP (3) BP_TDVP_NSTEP (= BP_NSTEP) BP_DT (0.1) BP_DT4 (0.2)
#       BP_VARIANTS ("wide:1e-4:0,thin:1e-4:0,thin:1e-4:1,thin:1e-3:0,thin:1e-3:1")
#             each BUG arm runs once per `search:tol_basis:fold`. search = thin (Test B sets the
#             width, the default) | wide (the old full-rank widening, `thin_search = false`);
#             tol_basis looser = the speed knob (USER 2026-10-16); fold = 1 folds the probe into
#             the environment contractions so H*Theta is never formed (mpo.jl, krylov_closure.jl)
#       BP_PROFV ("wide:1e-4:0,thin:1e-4:1")  the variants that also get a sampled profile
#       BP_TT/BP_TE (1e-6/1e-6: medium-to-tight, USER 2026-10-16 -- tol_trunc 1e-4 let the
#             per-step compression losses accumulate, TASK4_VALIDATION_RESULTS.md)
#       BP_GS_TB/BP_GS_TT (1e-4/1e-6: tight enough that the ground state reaches the cap)
#       BP_CLOSURE_LOG (0) 1 prints the order-p closure ledger after every step (`closure_lines`)

using LinearAlgebra, Printf, Serialization, Profile
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "lattice2d.jl"))
include(joinpath(@__DIR__, "..", "profile_buckets.jl"))

const LX    = parse(Int, get(ARGS, 1, "10"))
const LY    = parse(Int, get(ARGS, 2, "4"))
const OUT   = get(ARGS, 3, joinpath(@__DIR__, "..", "results"))
const MODE  = get(ENV, "BP_MODE", "profile")
const DS    = parse.(Int, split(get(ENV, "BP_DS", "256,512,1024"), ","))
const ARMS  = split(get(ENV, "BP_ARMS", "tdvp2,bug2"), ",")
const PROF  = split(get(ENV, "BP_PROF", "bug2"), ",")
const NSTEP = parse(Int, get(ENV, "BP_NSTEP", "3"))
# Same step count as the BUG arms by default, so a pair is compared at one simulated time.
const NTDVP = parse(Int, get(ENV, "BP_TDVP_NSTEP", string(NSTEP)))
const DT    = parse(Float64, get(ENV, "BP_DT", "0.1"))
const DT4   = parse(Float64, get(ENV, "BP_DT4", "0.2"))
function parse_variant(v)
    a = split(v, ":")
    a[1] in ("thin", "wide") || error("search must be thin or wide in $v")
    return (thin = a[1] == "thin", tb = parse(Float64, a[2]), fold = a[3] == "1")
end
const VARIANTS = parse_variant.(split(get(ENV, "BP_VARIANTS",
    "wide:1e-4:0,thin:1e-4:0,thin:1e-4:1,thin:1e-3:0,thin:1e-3:1"), ","))
const PROFV = parse_variant.(split(get(ENV, "BP_PROFV", "wide:1e-4:0,thin:1e-4:1"), ","))
const TT    = parse(Float64, get(ENV, "BP_TT", "1e-6"))
const TE    = parse(Float64, get(ENV, "BP_TE", "1e-6"))
const GS_TB = parse(Float64, get(ENV, "BP_GS_TB", "1e-4"))
const GS_TT = parse(Float64, get(ENV, "BP_GS_TT", "1e-6"))
const CLOSURE_LOG = get(ENV, "BP_CLOSURE_LOG", "0") == "1"
const PARALLEL = Threads.nthreads() >= 2
MODE in ("gs", "profile") || error("BP_MODE must be gs or profile, got $MODE")

const T0 = time()
say(msg) = (@printf("[%8.1fs] %s\n", time() - T0, msg); flush(stdout))

# ⛔ A TIMING RATIO IS ONLY MEANINGFUL BETWEEN ARMS OF THE SAME ORDER (USER 2026-10-16). bug4 vs
# tdvp2 compared a 4th-order step at 2x dt with a 2nd-order one: different methods, not a cost
# comparison. Each BUG arm is ratioed against the TDVP arm of its own nominal order, at the same
# dt; bug1 has no first-order TDVP partner and gets no ratio.
const ARM_ORDER = Dict("tdvp2" => 2, "tdvp4" => 4, "bug1" => 1, "bugmid" => 2, "bug2" => 2,
                       "bug4" => 4)
const TDVP_OF_ORDER = Dict(2 => "tdvp2", 4 => "tdvp4")

armdt(arm) = ARM_ORDER[arm] == 4 ? DT4 : DT
istdvp(arm) = startswith(arm, "tdvp")

"Every (arm, variant) run: tdvp2 once, each BUG arm once per `VARIANTS` entry."
runs() = [(arm, v) for arm in ARMS for v in (istdvp(arm) ? [nothing] : VARIANTS)]
label(arm, v) = v === nothing ? arm :
    @sprintf("%s/%s/tb%.0e%s", arm, v.thin ? "thin" : "wide", v.tb, v.fold ? "/fold" : "")

function stepper(arm, W, D; parallel::Bool, v = VARIANTS[1])
    akw = (tol_basis = v.tb, tol_trunc = TT, tol_expv = TE, maxdim = D, parallel = parallel,
           fold_omega = v.fold, thin_search = v.thin)
    arm == "tdvp2"  && return (p, tau) -> tdvp2_step!(p, W, tau; maxdim = D, trunc_thresh = 1e-12,
                                                       maxiter = 8)
    arm == "tdvp4"  && return (p, tau) -> tdvp2_composed_step!(p, W, tau; order = 4,
                                                                maxdim = D, trunc_thresh = 1e-12,
                                                                maxiter = 8)
    arm == "bug1"   && return (p, tau) -> cbe_bug_step!(p, W, tau; akw...)
    arm == "bugmid" && return (p, tau) -> cbe_bug_midpoint_step!(p, W, tau; akw...)
    arm == "bug2"   && return (p, tau) -> cbe_bug_step!(p, W, tau; order = 2, akw...)
    arm == "bug4"   && return (p, tau) -> cbe_bug_step!(p, W, tau; order = 4, akw...)
    error("unknown arm $arm")
end

energy(p, W) = real(mpo_energy(copy(p), W)) / max(norm(p)^2, eps())

"""
The ground state at cap `D`, warm-started from the previous rung. Cached per D with the inputs
that change it INSIDE the payload, and the energy recomputed on load (the `fig45_2209.jl` rule).
"""
function ground_state(W, N, D, prev)
    key  = ("square", LX, LY, N, D, :U1, :adaptive, GS_TB, GS_TT)
    file = joinpath(OUT, "bp_gs_square_$(LX)x$(LY)_D$(D).jls")
    if isfile(file)
        try
            blob = Serialization.deserialize(file)
            if blob.key == key && abs(energy(blob.psi, W) - blob.E0) < 1e-8 * max(1.0, abs(blob.E0))
                say(@sprintf("GS D=%d REUSED  E0=%.10f  chi=%d", D, blob.E0,
                             maximum(state_bond_dims(blob.psi))))
                return blob.psi
            end
            say("GS cache $(basename(file)) rejected (key or energy mismatch), re-solving")
        catch err
            say("GS cache $(basename(file)) unreadable ($(err)), re-solving")
        end
    end
    psi = prev === nothing ? product_state([isodd(j) ? :up : :down for j in 1:N]) : deepcopy(prev)
    t = @elapsed gs = rsvd_cbe_dmrg1s!(psi, W; n_sweeps = 60, maxdim = D, tol_basis = GS_TB,
                                       tol_trunc = GS_TT, maxiter = 30, restol = 1e-10)
    E0 = energy(psi, W)
    say(@sprintf("GS D=%d  E0=%.10f  E0/N=%.10f  chi=%d  sweeps=%d  conv=%s  %s  %.1fs", D, E0,
                 E0 / N, maximum(state_bond_dims(psi)), length(gs.energies),
                 string(gs.converged), string(gs.status[end]), t))
    try
        Serialization.serialize(file, (key = key, psi = psi, E0 = E0))
    catch err
        say("WARNING GS not cached ($(err))")
    end
    return psi
end

const PHASES = (:t_cbe, :t_kry, :t_split, :t_frame, :t_env, :t_root, :t_trunc)

closure_time(info) = (hasproperty(info, :ledger) && info.ledger isa AdaptiveLedger) ?
                     info.ledger.t_closure : 0.0

function phase_line(info)
    hasproperty(info, :t_step) || return ""
    ts = [getproperty(info, f) for f in PHASES]
    tc = closure_time(info)
    rest = info.t_step - sum(ts) - tc
    parts = [@sprintf("%s %.2f", string(f)[3:end], t) for (f, t) in zip(PHASES, ts)]
    return join(parts, "  ") * @sprintf("  closure %.2f  unattributed %.2f  (step %.2f s)", tc,
                                        rest, info.t_step)
end

"""
The order-p closure's own record of one step (`BP_CLOSURE_LOG=1`): passes, the per-frontier
leaks before (`leak_initial`, measured on the half-sweep frames) and after, and one line per
frontier sweep -- so a step that spends its time in `closure` says whether it ran to the pass
bound and whether the leaks were still falling when it stopped.
"""
function closure_lines(info)
    (hasproperty(info, :ledger) && info.ledger isa AdaptiveLedger) || return
    lg = info.ledger
    lg.order > 1 || return
    fmt(v) = join((@sprintf("%.2e", x) for x in v), " ")
    @printf("         closure: order %d  passes %d  defect %.2e  workcap %s  r_aug %d  leak_initial [%s]  rho_initial [%s]  leak [%s]\n",
            lg.order, lg.n_passes, lg.defect, string(lg.workcap_reached), lg.r_aug,
            fmt(lg.leak_initial), fmt(lg.rho_initial), fmt(lg.leak))
    for e in lg.closure_log
        g(k) = get(e, k, missing)
        @printf("           pass %2d  j %d  enrich %-5s  leak %.2e (L %.2e R %.2e)  met %-7s  added %4s (L %4s R %4s)  width %4s -> %4s  n_h2 %4s  n_h1 %5s  t %7.1f s\n",
                e.pass, e.frontier, string(e.enrich), e.leak, coalesce(g(:leak_left), NaN),
                coalesce(g(:leak_right), NaN), string(g(:met)), string(g(:added)),
                string(g(:added_left)), string(g(:added_right)), string(g(:width_before)),
                string(g(:width_after)), string(g(:n_h2)), string(g(:n_h1)), e.t)
    end
end

unmet_reasons(info) = (hasproperty(info, :ledger) && info.ledger isa AdaptiveLedger &&
                       budget_status(info.ledger) === :budget_unmet) ?
                      join(string.(info.ledger.unmet), "+") : ""

function profile_step(arm, W, D, p; v = VARIANTS[1])
    step! = stepper(arm, W, D; parallel = false, v = v)
    lab = label(arm, v)
    q = deepcopy(p)
    Profile.clear()
    Profile.init(n = 10^7, delay = 0.01)
    g0 = Base.gc_num()
    tw = @elapsed info = @profile step!(q, ComplexF64(-im * armdt(arm)))
    gd = Base.GC_Diff(Base.gc_num(), g0)
    say(@sprintf("PROFILE %s D=%d  serial step %.1f s  gc %.1f s  alloc %.1f GiB  pauses %d full %d",
                 lab, D, tw, gd.total_time / 1e9, gd.allocd / 2^30, gd.pause, gd.full_sweep))
    pl = phase_line(info)
    isempty(pl) || println("  phases (serial, one clock): ", pl)
    b, active, tot = profile_buckets()
    @printf("  kernel buckets (%d active / %d samples):\n", active, tot)
    for name in BUCKETS
        name == "IDLE" && continue
        b[name] > 0 && @printf("    %-22s %5.1f%%\n", name, 100 * b[name] / max(active, 1))
    end
    println("  call sites in our source (innermost package frame), by function:")
    for (k, c) in owner_top(15; withline = false)
        @printf("    %5.1f%%  %s\n", 100 * c / max(active, 1), k)
    end
    println("  ... and by line:")
    for (k, c) in owner_top(25)
        @printf("    %5.1f%%  %s\n", 100 * c / max(active, 1), k)
    end
    println("  GC/alloc samples by call site in our source (share of active):")
    for (k, c) in owner_top(15; bucket = "GC/alloc")
        @printf("    %5.1f%%  %s\n", 100 * c / max(active, 1), k)
    end
    println("  unclassified innermost frames:")
    for (k, c) in unclassified_top(8)
        @printf("    %5.1f%%  %s\n", 100 * c / max(active, 1), k)
    end
    open(joinpath(OUT, "bp_profile_$(replace(lab, "/" => "_"))_D$(D).txt"), "w") do io
        Profile.print(IOContext(io, :displaysize => (100000, 400)); format = :tree,
                      mincount = 20, noisefloor = 2.0)
    end
    flush(stdout)
end

function main()
    for (k, arm) in enumerate(ARMS)
        haskey(ARM_ORDER, arm) || error("unknown arm $arm")
        pr = get(TDVP_OF_ORDER, ARM_ORDER[arm], nothing)
        istdvp(arm) || pr === nothing || !(pr in ARMS) || findfirst(==(pr), ARMS) < k ||
            error("BP_ARMS: $pr must come before $arm, which is graded against it")
    end
    mkpath(OUT)
    set_symmetry!(:U1)
    N  = LX * LY
    Jm = square_cylinder_couplings(LX, LY; periodic_y = true)
    W  = pair_mpo(N, Jm, spin_vertices(N))
    c  = centre_site(LX, LY)
    say(@sprintf("square %dx%d N=%d  centre %d  caps %s  mode %s  threads %d  BLAS %d", LX, LY, N,
                 c, string(DS), MODE, Threads.nthreads(), BLAS.get_num_threads()))
    say(@sprintf("evolution %s  tol_trunc %.0e  tol_expv %.0e  dt %.3f  bug4 dt %.3f  GS tol (%.0e, %.0e)",
                 join(label.("bug", VARIANTS), " "), TT, TE, DT, DT4, GS_TB, GS_TT))

    gs = Dict{Int, Any}()
    prev = nothing
    for D in DS
        prev = gs[D] = ground_state(W, N, D, prev)
    end
    MODE == "gs" && (println("BP-GS-DONE"); return)

    io = open(joinpath(OUT, "bp_cost_$(LX)x$(LY).csv"), "w")
    println(io, "D,arm,search,tol_basis,fold,dt,step,wall,chi,krylov_dims," * join(string.(PHASES), ",") *
                ",t_closure,t_step,unmet,gc_s,alloc_gib,gc_pauses,gc_full")
    for D in DS
        phi = deepcopy(gs[D])
        apply_sz!(phi, c)
        say(@sprintf("== D=%d  start chi %d", D, maximum(state_bond_dims(phi))))
        perstep = Dict{String, Float64}()
        # Each arm's FIRST variant (by default the old wide search) is the reference its other
        # variants are compared with at the same simulated time -- a cheaper variant is only a
        # saving if it lands on the same state.
        refs = Dict{String, Any}()
        tdvpfinal = Dict{String, Any}()         # TDVP arm => (simulated time, final state)
        pairinf = Dict{String, Float64}()       # BUG label => infidelity vs its same-order TDVP
        for (arm, v) in runs()
            lab = label(arm, v)
            step! = stepper(arm, W, D; parallel = PARALLEL,
                            v = v === nothing ? VARIANTS[1] : v)
            # JIT, untimed. tdvp4 is five tdvp2 stages -- nothing new to compile, and at large chi
            # a warm-up would cost five full TDVP2 steps.
            arm == "tdvp4" || let q = deepcopy(phi); step!(q, ComplexF64(-im * armdt(arm))); end
            p = deepcopy(phi)
            n = istdvp(arm) ? NTDVP : NSTEP
            walls = Float64[]
            for k in 1:n
                # GC time and bytes from the runtime's own counters: the sampled profile put GC at
                # ~40% of active samples at D=256 (job 17283685) but lost most main-thread samples,
                # so the bucket share alone cannot say how much wall the allocations cost.
                g0 = Base.gc_num()
                tw = @elapsed info = step!(p, ComplexF64(-im * armdt(arm)))
                gd = Base.GC_Diff(Base.gc_num(), g0)
                push!(walls, tw)
                chi = maximum(state_bond_dims(p))
                kd = hasproperty(info, :krylov_dims) ? info.krylov_dims : -1
                ph = hasproperty(info, :t_step) ? [getproperty(info, f) for f in PHASES] :
                     fill(NaN, length(PHASES))
                ts = hasproperty(info, :t_step) ? info.t_step : NaN
                um = unmet_reasons(info)
                println(io, join([D, arm, v === nothing ? "" : (v.thin ? "thin" : "wide"),
                                  v === nothing ? NaN : v.tb, v === nothing ? false : v.fold,
                                  armdt(arm), k, @sprintf("%.3f", tw), chi, kd,
                                  (@sprintf("%.3f", x) for x in ph)...,
                                  @sprintf("%.3f", closure_time(info)), @sprintf("%.3f", ts),
                                  um, @sprintf("%.3f", gd.total_time / 1e9),
                                  @sprintf("%.3f", gd.allocd / 2^30), gd.pause, gd.full_sweep], ","))
                flush(io)
                @printf("  %-12s D=%4d step %d  %8.1f s  chi %4d  matvec %6d  gc %6.1f s (%4.1f%%) alloc %7.1f GiB pauses %d full %d  %s\n",
                        lab, D, k, tw, chi, kd, gd.total_time / 1e9, 100 * gd.total_time / 1e9 / tw,
                        gd.allocd / 2^30, gd.pause, gd.full_sweep, isempty(um) ? "" : "unmet " * um)
                pl = phase_line(info)
                isempty(pl) || println("         CPU phases: ", pl)
                CLOSURE_LOG && closure_lines(info)
                flush(stdout)
            end
            perstep[lab] = sum(walls) / (n * armdt(arm))
            # Only the TDVP finals are kept (a D=1024 state is ~1 GB); a BUG arm is graded against
            # its same-order partner here, which must therefore come first in BP_ARMS.
            if istdvp(arm)
                tdvpfinal[arm] = (n * armdt(arm), deepcopy(p))
            elseif haskey(tdvpfinal, get(TDVP_OF_ORDER, ARM_ORDER[arm], ""))
                tt, pt = tdvpfinal[TDVP_OF_ORDER[ARM_ORDER[arm]]]
                pairinf[lab] = isapprox(tt, n * armdt(arm)) ?
                               1 - abs(overlap(pt, p)) / (norm(pt) * norm(p)) : NaN
            end
            if !haskey(refs, arm)
                refs[arm] = (lab, deepcopy(p))
            else
                rl, rp = refs[arm]
                inf = 1 - abs(overlap(rp, p)) / (norm(rp) * norm(p))
                @printf("  %-12s D=%4d  infidelity vs %s at t=%.2f: %.2e\n", lab, D, rl,
                        n * armdt(arm), inf)
            end
            arm in PROF && (v === nothing || v in PROFV) && profile_step(arm, W, D, p;
                                                                       v = something(v, VARIANTS[1]))
        end
        for (arm, v) in runs()
            istdvp(arm) && continue
            lab = label(arm, v)
            ord = ARM_ORDER[arm]
            partner = get(TDVP_OF_ORDER, ord, nothing)
            if partner === nothing || !haskey(perstep, partner)
                @printf("  D=%4d  %-12s order %d: no same-order TDVP arm in this run -- no ratio\n",
                        D, lab, ord)
                continue
            end
            # The pair must land on the same state for the ratio to be a cost comparison.
            @printf("  D=%4d  order %d dt %.3f  wall per unit time: %-12s %.1f s  %s %.1f s  -> %s/%s = %.2fx  infidelity vs %s at t=%.2f: %.2e\n",
                    D, ord, armdt(arm), lab, perstep[lab], partner, perstep[partner], partner,
                    lab, perstep[partner] / perstep[lab], partner, NSTEP * armdt(arm),
                    get(pairinf, lab, NaN))
        end
        flush(stdout)
    end
    close(io)
    println("BP-PROFILE-DONE")
end

main()
