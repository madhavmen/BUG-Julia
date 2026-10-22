# benchmarks/hq_bench.jl -- the HQ plan's Task 8 driver: ONE setting of the CBE-BUG step under
# study per run, on the Task 7 model (open spin-s XX chain `spin_xx_mpo`, L = 16, domain wall,
# U(1), J = 1) to `J t = HQB_TF`. Every figure P01--P13 is read from these JSON files by
# `benchmarks/hq_task8_figures.py`; nothing here grades or plots.
#
# Every step records the step's own telemetry (`CBEBugSweepInfo`, its `AdaptiveLedger` and
# `StepRecord`): the ranks per bond before (`D_old`), after CBE (`D_CBE`), after the local Krylov
# (`D_joint`, the frame width), of the trial space (`D_working`, `info.expanded`) and after the
# compression (`D_final`); the DD-QR kernel counters and seconds; the Test A/B/C/D indicators,
# probes per sector; the closure leak; the root iterations; the limit statuses; the K/L/S account.
# Every `HQB_EVERY` (in time units) it also records the physics: `<S^z_i>`, the centre current,
# `<H>`, `Var(H)`, the norm, `S^z_tot`, the sector. s = 1/2 records are graded against the closed
# form here (`err_exact_*`); s = 3 against the T7 reference band in the figure script.
#
# `HQB_TWIN = 1` evolves a second copy with the STANDARD step (same tolerances, same seed) next to a
# K/L/S run and records `||psi - psi_std||` (P12).
#
# Env: HQB_TWOS (1), HQB_L (16), HQB_TF (8.0), HQB_H (0.1), HQB_EVERY (0.1, or h if larger),
#      HQB_ORDER (1), HQB_TOLB (1e-5), HQB_TOLR (1e-12), HQB_TOLT (1e-12), HQB_MAXDIM (1024),
#      HQB_SEED (1), HQB_POLICY (dd_qr_streamed | one_sided_qr_streamed | svd_select),
#      HQB_PARALLEL (serial | half_sweeps | full), HQB_KLS (none | sequential | kl_then_s |
#      concurrent), HQB_TWIN (0), HQB_BONDLOG (1), HQB_OUT (hq_bench_<tag>.json)

using LinearAlgebra, Printf, JSON, Random
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

const TWOS = parse(Int, get(ENV, "HQB_TWOS", "1"))
const L = parse(Int, get(ENV, "HQB_L", "16"))
const TF = parse(Float64, get(ENV, "HQB_TF", "8.0"))
const H = parse(Float64, get(ENV, "HQB_H", "0.1"))
const EVERY = max(parse(Float64, get(ENV, "HQB_EVERY", "0.1")), H)
const ORDER = parse(Int, get(ENV, "HQB_ORDER", "1"))
const TOLB = parse(Float64, get(ENV, "HQB_TOLB", "1e-5"))
const TOLR = parse(Float64, get(ENV, "HQB_TOLR", "1e-12"))
const TOLT = parse(Float64, get(ENV, "HQB_TOLT", "1e-12"))
const MAXDIM = parse(Int, get(ENV, "HQB_MAXDIM", "1024"))
const SEED = parse(Int, get(ENV, "HQB_SEED", "1"))
const POLICY = Symbol(get(ENV, "HQB_POLICY", "dd_qr_streamed"))
const PARALLEL = Symbol(get(ENV, "HQB_PARALLEL", "serial"))
const KLS = (k = get(ENV, "HQB_KLS", "none"); k == "none" ? nothing : Symbol(k))
const TWIN = get(ENV, "HQB_TWIN", "0") == "1"
const BONDLOG = get(ENV, "HQB_BONDLOG", "1") == "1"
const TAG = "s$(TWOS)_p$(ORDER)_h$(H)_b$(TOLB)_$(POLICY)_$(PARALLEL)_$(something(KLS, :std))_seed$(SEED)"
const OUT = get(ENV, "HQB_OUT", "hq_bench_$(TAG).json")

TWOS == 1 && include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

set_symmetry!(:U1)
const S = TWOS / 2
const MPO_ = spin_xx_mpo(L, TWOS)
const JMPO = spin_current_mpo(L, TWOS, L ÷ 2)
const NSTEP = round(Int, TF / H)
const KREC = round(Int, EVERY / H)
abs(NSTEP * H - TF) < 1e-12 && abs(KREC * H - EVERY) < 1e-12 ||
    error("HQB_TF and HQB_EVERY must be multiples of HQB_H")
const STEPKW = (order = ORDER, tol_basis = TOLB, tol_root = TOLR, tol_trunc = TOLT,
                maxdim = MAXDIM, cbe_policy = POLICY, parallel = PARALLEL, kls = KLS)

psi = spin_domain_wall_state(L, TWOS)
twin = TWIN ? copy(psi) : nothing
rng, rng_twin = MersenneTwister(SEED), MersenneTwister(SEED)
sector = psi[L].spaces[3]
config = Dict("twoS" => TWOS, "L" => L, "tf" => TF, "h" => H, "every" => EVERY, "order" => ORDER,
              "tol_basis" => TOLB, "tol_root" => TOLR, "tol_trunc" => TOLT, "maxdim" => MAXDIM,
              "seed" => SEED, "policy" => POLICY, "parallel" => PARALLEL,
              "kls" => something(KLS, :none), "twin" => TWIN, "J" => 1.0,
              "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
              "solver" => "cbe_bug_step!")
steps, recs = Any[], Any[]

save() = open(OUT, "w") do io
    JSON.print(io, Dict("config" => config, "steps" => steps, "records" => recs))
end

_fields(x) = Dict(string(f) => getfield(x, f) for f in fieldnames(typeof(x)))
_sectors(d) = Dict(string(k) => Dict(string(q) => n for (q, n) in v) for (k, v) in d)

"The step's telemetry, JSON-ready (P04--P07, P09--P11, P13)."
function step_entry(k, info, bonds_old, wall)
    lg, rec = info.ledger, info.record
    e = Dict{String, Any}("k" => k, "t" => k * H, "wall" => wall,
        "D_old" => bonds_old, "D_final" => bond_dims(psi), "D_working" => info.expanded,
        "n_new" => info.n_new, "root" => info.root, "root_rank" => info.root_rank,
        "root_right" => info.root_right, "krylov_dims" => info.krylov_dims,
        "peak_elements" => info.peak_elements,
        "t_step" => info.t_step, "t_cbe" => info.t_cbe, "t_kry" => info.t_kry,
        "t_root" => info.t_root, "t_trunc" => info.t_trunc, "t_split" => info.t_split,
        "t_frame" => info.t_frame, "t_env" => info.t_env,
        "record" => Dict("policy" => rec.policy, "qr_backend" => rec.qr_backend,
                         "closure_policy" => rec.closure_policy, "order" => rec.order,
                         "parallel" => rec.parallel, "counters" => _fields(rec.counters),
                         "D_old" => rec.D_old, "D_stage" => rec.D_stage, "D_final" => rec.D_final,
                         "root_iters" => rec.root_iters, "t_step" => rec.t_step,
                         "bytes_step" => rec.bytes_step, "peak_rss" => rec.peak_rss,
                         "limits" => rec.limits, "budget_rule" => rec.budget_rule))
    if lg !== nothing
        e["status"] = budget_status(lg)
        e["diagnostic"] = budget_diagnostic(lg)
        scal = Dict(string(f) => getfield(lg, f) for f in fieldnames(AdaptiveLedger)
                    if getfield(lg, f) isa Union{Number, Bool, Symbol, Vector{Float64}, Vector{Symbol}})
        e["ledger"] = scal
        e["kls"] = lg.kls === nothing ? nothing : Dict(string(k) => v for (k, v) in pairs(lg.kls))
        e["closure_log"] = [Dict(string(k) => v for (k, v) in pairs(c)) for c in lg.closure_log]
        BONDLOG && (e["bond_log"] = [Dict(string(k) => (k === :probes_per_sector ? _sectors(v) : v)
                                          for (k, v) in pairs(b)) for b in lg.bond_log])
    end
    return e
end

record(t, wall) = begin
    m = spin_sz_profile(psi, TWOS)
    r = Dict{String, Any}("t" => t, "m" => m, "j" => real(mpo_energy(psi, JMPO)),
                          "E" => real(mpo_energy(psi, MPO_)), "var" => mpo_variance(psi, MPO_),
                          "norm" => real(overlap(psi, psi)), "sz" => sum(m),
                          "sector" => psi[L].spaces[3] == sector, "D" => maximum(bond_dims(psi)),
                          "bonds" => bond_dims(psi), "wall" => wall)
    if TWOS == 1
        C = xx_open_correlation(L, t)
        r["err_exact_m"] = maximum(abs.(m - (real(diag(C)) .- 0.5)))
        r["err_exact_l2"] = norm(m - (real(diag(C)) .- 0.5))
        r["err_exact_j"] = abs(r["j"] - xx_open_current(C)[L ÷ 2])
    end
    if twin !== nothing
        ov = overlap(twin, psi)
        r["dist_twin"] = sqrt(max(real(overlap(psi, psi) + overlap(twin, twin) - 2 * real(ov)), 0.0))
        r["twin_m"] = spin_sz_profile(twin, TWOS)
        r["twin_j"] = real(mpo_energy(twin, JMPO))
    end
    push!(recs, r)
    save()
    @printf("HQB t=%.3f D=%d m_c=%.12f j=%.12f E=%.2e dvar=%.2e dnorm=%.2e%s%s wall=%.1f\n",
            t, r["D"], m[L ÷ 2], r["j"], r["E"], r["var"] - S^2, r["norm"] - 1,
            TWOS == 1 ? @sprintf(" exact_m=%.2e exact_j=%.2e", r["err_exact_m"], r["err_exact_j"]) : "",
            twin === nothing ? "" : @sprintf(" dist_twin=%.2e", r["dist_twin"]), wall)
    flush(stdout)
end

println("HQB-CONFIG ", join(("$k=$v" for (k, v) in sort(collect(config); by = first)), " "))
t0 = time()
record(0.0, 0.0)
for k in 1:NSTEP
    bonds_old = bond_dims(psi)
    info = cbe_bug_step!(psi, MPO_, ComplexF64(-1im * H); STEPKW..., rng = rng)
    twin === nothing || cbe_bug_step!(twin, MPO_, ComplexF64(-1im * H);
                                      STEPKW..., parallel = :serial, kls = nothing, rng = rng_twin)
    push!(steps, step_entry(k, info, bonds_old, time() - t0))
    k % KREC == 0 ? record(k * H, time() - t0) : (k % 10 == 0 && save())
end
save()
println("HQB-DONE $OUT wall=$(round(time() - t0; digits = 1)) s")
