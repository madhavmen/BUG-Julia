# benchmarks/qtt/production.jl -- QTT plan Task 4: the gated long-time campaigns, and (with a smaller
# P_N / P_T) the Stage A-C pilots, scaling runs and the pre-production smoke.
#
#   P_MODEL   gp_shedding | heston_forward | heston_backward
#   P_ARM     bug | tdvp2_lagged | split            (the method under test)
#   P_N       bits per coordinate (15 -> L = 30, grouped x | y, never densified)
#   P_T, P_HMAX, P_H0, P_ORDER, P_TOLB, P_TOLT, P_TOLROOT, P_CHIMAX
#   P_CHECK   every P_CHECK steps a full vs two-half-step check: delta_h, e_hat = delta/(1 - 2^-p);
#             reject (h/2) above P_TARGET, otherwise grow h by min(2, 0.9 (target/e)^(1/(p+1)))
#   P_SNAPS   snapshot times (sampled density/phase, vortex counts, observables, cores)
#   P_OUT, P_CKPT_SEC (1800), P_RESUME (1), P_STOP_AT (wall seconds: checkpoint and stop -- the
#             restart check), P_SAMPLE (max samples per direction, 256)
#
# Files: manifest.json, ledger.csv (one row per attempted step, the plan §4.4 minimum ledger),
# snapshots.csv, snap_<t>.jls (sampled fields + cores), ckpt.jls.

include(joinpath(@__DIR__, "common.jl"))
using JSON

const MODEL = Symbol(envs("P_MODEL", "gp_shedding"))
const ARM = Symbol(envs("P_ARM", "bug"))
const NB = envi("P_N", 15)
const TEND = envf("P_T", 500.0)
const HMAX = envf("P_HMAX", 0.1)
const ORD = envi("P_ORDER", 3)
const TOLB, TOLT, TOLR = envf("P_TOLB", 1e-6), envf("P_TOLT", 1e-8), envf("P_TOLROOT", 1e-10)
const CHIMAX = envi("P_CHIMAX", 256)
const CHECK = envi("P_CHECK", 20)
const TARGET = envf("P_TARGET", 1e-6)
const OUT = envs("P_OUT", "prod_out")
const CKPT_SEC = envf("P_CKPT_SEC", 1800.0)
const STOP_AT = envf("P_STOP_AT", Inf)
const NSAMP = envi("P_SAMPLE", 256)
const SNAPS = envlist("P_SNAPS", "0,5,10,20,50,100,200,350,500")
mkpath(OUT)

# ── the problem ──
function setup()
    if MODEL === :gp_shedding
        box = envf("P_BOX", 64.0)
        g = qtt_grid(NB, NB; x = (-box / 2, box / 2), y = (-box / 2, box / 2))
        pad = GPPaddle(10.0, -box / 4, 1.5, box / 3 / 1.5, 2.0, 1.0, box / 6, 0.0, 0.5)
        m = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 1.0, omega = 0.0, paddle = pad)
        A0 = qtt_separable(g, x -> 1.0 + 0im, y -> 1.0 + 0im)
        return g, GPFlow(m), m, A0
    end
    g = qtt_grid(NB, NB; x = (-4.0, 4.0), y = (0.0, 1.0), periodic_x = false, periodic_y = false)
    p = HestonParams(0.03, 0.0, 1.5, 0.09, 0.5, -0.7)
    if MODEL === :heston_forward
        A0 = qtt_separable(g, x -> exp(-(x / 0.1)^2 / 2), v -> exp(-((v - 0.09) / 0.02)^2 / 2))
        A0 = tt_scale(A0, 1 / heston_mass(A0, g))
        return g, LinearFlow(heston_op(g, p; adjoint = true)), p, A0
    end
    A0 = qtt_separable(g, x -> 0.02 * log1p(exp((exp(x) - 1) / 0.02)), v -> 1.0)
    return g, LinearFlow(heston_op(g, p)), p, A0
end
g, fl, model, A0 = setup()
keep = MODEL === :heston_forward ? [qtt_mps(qtt_separable(g, x -> 1.0, v -> 1.0))] : Any[]

config = (model = MODEL, arm = ARM, bits_per_coordinate = NB, L = qtt_length(g), T = TEND,
          h_max = HMAX, order = ORD, tol_basis = TOLB, tol_trunc = TOLT, tol_root = TOLR,
          chi_max = CHIMAX, check_every = CHECK, target = TARGET, snapshots = SNAPS,
          domain = (x = (g.x0, g.x0 + g.dx * (2^g.nx - 1)), y = (g.y0, g.y0 + g.dy * (2^g.ny - 1))),
          spacing = (dx = g.dx, dy = g.dy), order_of_bits = g.order, endian = g.endian,
          sites_x = g.sites_x, sites_y = g.sites_y,
          boundaries = MODEL === :gp_shedding ? "periodic" : "onesided (Heston)",
          nonlinear_policy = MODEL === :gp_shedding ? "stage-consistent density (qtt_bug_step!)" : "linear")
config_hash = string(hash(string(config)); base = 16)
op_hash = string(hash(fl isa LinearFlow ? fl.H : model.H0); base = 16)
manifest = merge(config, (git_commit = git_commit(), config_hash = config_hash,
                          operator_hash = op_hash, initial_state_hash = string(qtt_state_tag(A0); base = 16),
                          rng_seed = "MersenneTwister(0x5EED) per step (cbe_bug_step! default)",
                          host = gethostname(), julia_threads = Threads.nthreads(),
                          blas_threads = BLAS.get_num_threads(),
                          cpu = Sys.CPU_NAME, source_methods = "qtt_bug_step!, qtt_tdvp2_step!, " *
                          "qtt_split_tdvp_step!, cbe_bug_step! (DD-CBE + local Krylov + order-p closure)"))
open(io -> JSON.print(io, Dict(String(k) => string(v) for (k, v) in pairs(manifest))),
     joinpath(OUT, "manifest.json"), "w")

# ── state, checkpoint ──
ck = joinpath(OUT, "ckpt.jls")
state = if envi("P_RESUME", 1) == 1 && isfile(ck)
    s = deserialize(ck)
    s.config_hash == config_hash || error("checkpoint config hash $(s.config_hash) != $config_hash")
    println("RESUME at t = $(s.t), step $(s.k)")
    s
else
    (config_hash = config_hash, t = 0.0, h = envf("P_H0", HMAX), k = 0, cores = A0,
     done_snaps = Float64[], ledger = NamedTuple[], snaps = NamedTuple[], wall = 0.0)
end
t, h, k = state.t, state.h, state.k
psi = qtt_mps(state.cores)
ledger, snaprows, done = copy(state.ledger), copy(state.snaps), copy(state.done_snaps)
wall0 = state.wall
tstart = time()
lastck = time()
checkpoint() = serialize(ck, (config_hash = config_hash, t = t, h = h, k = k, cores = qtt_cores(psi),
                              done_snaps = done, ledger = ledger, snaps = snaprows,
                              wall = wall0 + time() - tstart))

function one_step!(ps, t, hh)
    ARM === :bug && return qtt_bug_step!(ps, fl, t, hh; order = ORD, maxdim = CHIMAX,
                                         tol_basis = TOLB, tol_trunc = TOLT, tol_stage = TOLR,
                                         keep = keep)
    ARM === :tdvp2_lagged && return qtt_tdvp2_step!(ps, fl, t, hh; maxdim = CHIMAX, tol_trunc = TOLT)
    ARM === :split && return qtt_split_tdvp_step!(ps, model, t, hh; maxdim = CHIMAX, tol_trunc = TOLT)
    error("unknown arm $ARM")
end
"`||a - b|| / ||b||` without forming `a - b` (direct sum, as the dissipative driver's distance)."
function dist(a, b)
    A, B = qtt_cores(a), qtt_cores(b)
    return tt_norm(tt_add(A, tt_scale(B, -1.0))) / max(tt_norm(B), eps())
end
scalar(ps) = MODEL === :gp_shedding ? qtt_l2norm(ps, g)^2 : heston_mass(qtt_cores(ps), g)

function snapshot!(ts)
    A = qtt_cores(psi)
    st = max(1, 2^NB ÷ NSAMP)
    xs, ys, M = qtt_sample(A, g; stride = st)
    row = Dict{Symbol, Any}(:t => ts, :chi_max => maximum(bond_dims(psi)),
                            :profile => join(bond_dims(psi), ";"), :scalar => scalar(psi))
    if MODEL === :gp_shedding
        o = gp_observables(model, A, ts; residual = qtt_length(g) <= 20)
        vp, vm, _ = vortex_count(M)
        merge!(row, Dict(:N => o.N, :E_kin => o.E_kin, :E_int => o.E_int, :E_pad => o.E_pad,
                         :F_gc => o.F_gc, :P_ext => o.P_ext, :vortices_plus => vp,
                         :vortices_minus => vm))
    else
        for (j, (a, b)) in enumerate(((0.5, 0.0), (0.0, -1.0)))
            row[Symbol("moment$j")] = heston_moment(A, g, a, b)
        end
        row[:min_rel] = minimum(real, M) / maximum(abs, M)
    end
    push!(snaprows, (; sort(collect(row); by = first)...))
    serialize(joinpath(OUT, @sprintf("snap_%08.3f.jls", ts)), (t = ts, xs = xs, ys = ys, M = M, cores = A))
    write_csv(joinpath(OUT, "snapshots.csv"), snaprows)
    push!(done, ts)
    println("SNAP ", snaprows[end]); flush(stdout)
end

0.0 in SNAPS && !(0.0 in done) && snapshot!(0.0)
while t < TEND - 1e-12
    nxt = minimum(s for s in SNAPS if s > t + 1e-12; init = TEND)
    hh = min(h, HMAX, nxt - t)
    t1 = time()
    check = CHECK > 0 && k % CHECK == 0
    accepted, reason, ehat = true, :ok, NaN
    if check
        full, halfs = copy(psi), copy(psi)
        r1 = one_step!(full, t, hh)
        one_step!(halfs, t, hh / 2)
        r = one_step!(halfs, t + hh / 2, hh / 2)
        ehat = dist(full, halfs) / (1 - 2.0^(-ORD))
        if ehat > TARGET || (hasproperty(r, :converged) && !r.converged)
            accepted, reason = false, ehat > TARGET ? :step_error : :stage
        else
            RC._replace_state!(psi, halfs)
        end
    else
        r = one_step!(psi, t, hh)
        if hasproperty(r, :converged) && !r.converged
            reason = :stage_unconverged          # recorded, accepted (no check this step)
        end
    end
    w = time() - t1
    push!(ledger, (t = t, h = hh, p = ORD, accepted = accepted, reason = reason, e_hat = ehat,
                   scalar = scalar(psi), chi_max = maximum(bond_dims(psi)),
                   frame = hasproperty(r, :frame_width) ? join(r.frame_width, "x") : "",
                   newton = hasproperty(r, :stats) ? sum(s.newton for s in r.stats) : 0,
                   gmres = hasproperty(r, :stats) ? sum(s.gmres for s in r.stats) : 0,
                   stage_res = hasproperty(r, :stats) ? r.stats[end].residual : NaN,
                   discarded = hasproperty(r, :discarded) ? r.discarded : NaN,
                   A_max = hasproperty(r, :A_max) ? r.A_max : NaN,
                   t_basis = hasproperty(r, :t_basis) ? r.t_basis : NaN,
                   t_root = hasproperty(r, :t_root) ? r.t_root : NaN,
                   t_rho = hasproperty(r, :t_rho0) ? r.t_rho0 : NaN, wall = w,
                   maxrss_gb = Sys.maxrss() / 2^30))
    if accepted
        global t += hh
        global k += 1
        check && isfinite(ehat) && ehat > 0 &&
            (global h = min(HMAX, hh * clamp(0.9 * (TARGET / ehat)^(1 / (ORD + 1)), 0.5, 2.0)))
    else
        global h = hh / 2
    end
    any(abs(t - s) < 1e-9 for s in SNAPS) && !any(abs(t - s) < 1e-9 for s in done) && snapshot!(t)
    if k % 10 == 0 || !accepted
        write_csv(joinpath(OUT, "ledger.csv"), ledger)
        println(ledger[end]); flush(stdout)
    end
    if time() - lastck > CKPT_SEC
        checkpoint(); global lastck = time()
    end
    if wall0 + time() - tstart > STOP_AT
        checkpoint()
        println("STOP-AT reached at t = $t, step $k: checkpointed")
        break
    end
end
write_csv(joinpath(OUT, "ledger.csv"), ledger)
checkpoint()
println("PROD-DONE t = $t steps $k wall $(round(wall0 + time() - tstart; digits = 1)) s")
