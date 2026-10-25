# benchmarks/dissipative_run.jl -- the driver of dissipative_cbe_bug_implementation_plan.md Tasks 5-7
# (its "suggested driver"): ONE run of one model, representation and integrator per call, every
# figure read from the JSON it writes by benchmarks/dissipative_figures.py. Nothing here grades.
#
# Models (plan §3/§4): A1 Hubbard chain, bulk pair loss; A2 XXZ, collective emission; B1 junction,
# localized pair loss; B2 XXZ, collective emission at a chosen excitation density. Representations:
# `no_jump` (the conditional state, P0, log P0) or `traj` (a quantum-jump ensemble). Integrators:
# `bug` (cbe_bug_step!, order DR_ORDER) or `tdvp2` (two-site TDVP on the same G, same cut rule).
#
# no_jump records, every DR_EVERY: P0, log P0, <Q>_c (= -d log P0/dt), <H>_c, Re<HQ>_c (the
# energy balance (3) is checked offline), the model's observables, the centre entropy, the bond
# dimensions and the centre bond's charge sectors; per step: the step size, wall seconds and, for
# BUG, the ledger (frame width, root iterations, probes, selected and Krylov directions, discarded
# norm, closure passes), the step record (D_stage, D_work, D_final, bytes) and its limit flags
# (basis/root/truncation/rank/work_limited, StepRecord.limits); for TDVP2 its
# discarded norm, Krylov applications and backward-substep amplification. With DR_TARGET the
# step-doubling controller of plan eq. (7) sets the step (jump_trajectory with jumps = false).
# For L small enough an ED reference is computed here and the errors recorded.
#
# traj records the ensemble mean / standard error of the observables and of the jump count, and
# per trajectory the steps, event-location evaluations, rejections and norm-growth flags.
#
# Env (defaults): DR_MODEL (A2), DR_L (8), DR_MODE (no_jump), DR_INT (bug), DR_DT (0.05), DR_T (1),
#   DR_EVERY (= DR_DT), DR_ORDER (3), DR_TOLB (1e-8), DR_TOLR (1e-12), DR_TOLT (1e-10),
#   DR_MAXDIM (256), DR_CLOSURE (1 for A2/B2, else 0: USER 2026-10-24),
#   DR_KEEP_KRYLOV (DR_ORDER for A2/B2, else 0: G psi..G^m psi kept on steps from a product state,
#   dt5m), DR_KEEP_EVERY (0; 1 = on every step, the ablation),
#   DR_EXACT (0; 1 = the complete DD probe, the lossless arm of the Task 5 order ablation),
#   DR_WARM (0; t_w > 0 = start from the state evolved to t_w by tight BUG steps, see below),
#   DR_TARGET (none), DR_CHECK (4), DR_NTRAJ (100), DR_SEED (1), DR_REF (auto: ed when small),
#   A1/B1: DR_U (4), DR_KAPPA (1), DR_START (dbl_rich | low_dbl; B1: biased leads), DR_JLINK (1);
#   A2/B2: DR_JZ (1), DR_GAMMA (1), DR_GM (= DR_GAMMA), DR_KA (pi/3), DR_NEXC (L/2 for A2,
#   L/4 for B2), DR_START (block | alt | neel | bright | dark); DR_OUT (dissipative_<tag>.json)

using LinearAlgebra, Printf, JSON, Random
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

env(k, d) = get(ENV, k, string(d))
const MODEL = Symbol(env("DR_MODEL", "A2"))
const L = parse(Int, env("DR_L", 8))
const MODE = Symbol(env("DR_MODE", "no_jump"))
const INT = Symbol(env("DR_INT", "bug"))
const DT = parse(Float64, env("DR_DT", 0.05))
const TF = parse(Float64, env("DR_T", 1.0))
const EVERY = parse(Float64, env("DR_EVERY", DT))
const ORDER = parse(Int, env("DR_ORDER", 3))
const TOLB = parse(Float64, env("DR_TOLB", 1e-8))
const TOLR = parse(Float64, env("DR_TOLR", 1e-12))
const TOLT = parse(Float64, env("DR_TOLT", 1e-10))
const MAXDIM = parse(Int, env("DR_MAXDIM", 256))
const NONLOCAL = MODEL in (:A2, :B2)
const CLOSURE = env("DR_CLOSURE", NONLOCAL ? 1 : 0) == "1"
const KEEPK = parse(Int, env("DR_KEEP_KRYLOV", NONLOCAL ? ORDER : 0))
const KEEP_EVERY = env("DR_KEEP_EVERY", 0) == "1"
const EXACT = env("DR_EXACT", 0) == "1"
const WARM = parse(Float64, env("DR_WARM", 0))
const TARGET = (x = env("DR_TARGET", "none"); x == "none" ? nothing : parse(Float64, x))
const CHECK = parse(Int, env("DR_CHECK", 4))
const NTRAJ = parse(Int, env("DR_NTRAJ", 100))
const SEED = parse(Int, env("DR_SEED", 1))
const START = Symbol(env("DR_START", MODEL in (:A1,) ? "dbl_rich" : MODEL === :B1 ? "biased" : "block"))

set_symmetry!(:U1)

# ── models ──────────────────────────────────────────────────────────────────────────────────
function build_model()
    if MODEL in (:A1, :B1)
        U, κ = parse(Float64, env("DR_U", 4.0)), parse(Float64, env("DR_KAPPA", 1.0))
        q = hub4_space()
        nup = to_concrete(lock(q.Fu', 2) * q.Fu)
        ndn = to_concrete(lock(q.Fd', 2) * q.Fd)
        dbl = to_concrete(lock(nup, 1) * ndn)
        if MODEL === :A1
            H = hubbard_aa_mpo_d4(L; J = 1.0, U = U)
            Q = doublon_loss_mpo(L; kappa = κ)
            ch = pair_loss_channels(L; kappa = κ)
            pat = START === :dbl_rich ? [:both, :empty] : [:both, :up, :empty, :down]
            psi = hub4_product_state([pat[mod1(j, length(pat))] for j in 1:L])
            par = Dict("U" => U, "kappa" => κ)
        else
            Jl = parse(Float64, env("DR_JLINK", 1.0))
            j0 = L ÷ 2
            m = pair_loss_junction(L; J = 1.0, J_link = Jl, U = U, kappa = κ, j0 = j0)
            H, Q, ch = m.H, m.Q, m.channels
            lead(n, s) = [isodd(k) ? s : :both for k in 1:n]
            psi = hub4_product_state(vcat(lead(j0 - 1, :up), [:empty], lead(L - j0, :down)))
            prep = pair_loss_junction(L; J = 1.0, J_link = 0.0, U = U, kappa = 1.0, j0 = j0)
            ite_evolve!(psi, prep.H, log_panel_steps(20.0, 8; per_panel = 4); order = 2,
                        tol_basis = 1e-9, tol_trunc = 1e-12, tol_root = 1e-12, maxdim = MAXDIM,
                        retries = 3)
            par = Dict("U" => U, "kappa" => κ, "J_link" => Jl, "j0" => j0)
        end
        obs = p -> vcat([real(site_expval(p, j, dbl)) for j in 1:L],
                        [real(site_expval(p, j, nup)) + real(site_expval(p, j, ndn)) for j in 1:L])
        names = vcat(["dbl_$j" for j in 1:L], ["n_$j" for j in 1:L])
        return (; H, Q, ch, psi, obs, names, par)
    end
    Jz, g = parse(Float64, env("DR_JZ", 1.0)), parse(Float64, env("DR_GAMMA", 1.0))
    gm, ka = parse(Float64, env("DR_GM", g)), parse(Float64, env("DR_KA", π / 3))
    nexc = parse(Int, env("DR_NEXC", MODEL === :A2 ? L ÷ 2 : max(1, L ÷ 4)))
    h = [0.2 * (-1)^j for j in 1:L]
    H = field_mpo_from_terms(xxz_chain(L; J = 1.0, delta = Jz), h)
    Q = collective_emission_mpo(L; gamma_plus = g, gamma_minus = gm, k = ka)
    ch = collective_emission_channels(L; gamma_plus = g, gamma_minus = gm, k = ka)
    up = START === :alt ? Set(1 + 2k for k in 0:(nexc - 1)) :
         START === :neel ? Set(1:2:L) : Set((L ÷ 2 - nexc ÷ 2 + 1):(L ÷ 2 - nexc ÷ 2 + nexc))
    psi = product_state([j in up ? :up : :down for j in 1:L])
    if START in (:bright, :dark)                  # e^{∓βQ/2} on the block: Task 3's seeds
        tau, n = START === :dark ? (-2.0, 10) : (0.25, 8)
        for _ in 1:n
            cbe_bug_step!(psi, Q, ComplexF64(tau); hermitian = true, tol_basis = 1e-8,
                          tol_root = 1e-12, tol_trunc = 1e-12, maxdim = MAXDIM)
            canonical!(psi, 1)
            psi[1] = to_concrete(psi[1] / norm(psi[1]))
        end
    end
    sz = local_space().Sz
    c = L ÷ 2
    function obs(p)
        P = real(overlap(p, p))
        zz = real(overlap(apply_sz!(copy(p), c), apply_sz!(copy(p), c + 1))) / P
        return vcat([real(site_expval(p, j, sz)) for j in 1:L], zz)
    end
    names = vcat(["sz_$j" for j in 1:L], ["zz_c"])
    par = Dict("Jz" => Jz, "gamma_plus" => g, "gamma_minus" => gm, "ka" => ka, "nexc" => length(up))
    return (; H, Q, ch, psi, obs, names, par)
end

const M = build_model()
const G = no_jump_mpo(M.H, M.Q)
const TAG = @sprintf("%s_L%d_%s_%s_dt%g_b%g_r%g_t%g_%s%s%s%s%s_seed%d", MODEL, L, MODE, INT, DT, TOLB,
                     TOLR, TOLT, START, WARM > 0 ? "_warm$(WARM)" : "", TARGET === nothing ? "" : "_tau$(TARGET)",
                     KEEPK > 0 ? "_keep$(KEEPK)$(KEEP_EVERY ? "e" : "")" : "", EXACT ? "_exact" : "", SEED)
const OUT = env("DR_OUT", "dissipative_$(TAG).json")

# ── the step ────────────────────────────────────────────────────────────────────────────────
const LOG = Dict{String, Vector{Float64}}()
logv!(k, v) = push!(get!(LOG, k, Float64[]), Float64(v))

# From a U(1) product state the closure cannot see a nonlocal Q's long-range hops, so the global
# Krylov vectors are kept on such a step (plan U10, dt5m); on any later step they change nothing.
keep_now(psi) = KEEPK > 0 && (KEEP_EVERY || maximum(bond_dims(psi); init = 1) == 1)

function step!(psi, h)
    t0 = time_ns()
    if INT === :bug
        keep = keep_now(psi) ? global_krylov_states(G, psi, KEEPK; tol_trunc = TOLT, maxdim = MAXDIM) :
               Any[]
        info = cbe_bug_step!(psi, G, ComplexF64(-im * h); hermitian = false, order = ORDER,
                             tol_basis = TOLB, tol_root = TOLR, tol_trunc = TOLT, maxdim = MAXDIM,
                             closure_checks = CLOSURE, exact = EXACT, keep_states = keep)
        lg, rc = info.ledger, info.record
        logv!("r_aug", lg.r_aug); logv!("expv_iters", lg.expv_iters); logv!("n_probes", lg.n_probes)
        logv!("n_selected", lg.n_selected); logv!("n_krylov", lg.n_krylov)
        logv!("discarded", lg.discarded); logv!("closure_sweeps", lg.n_closure_sweeps)
        logv!("D_stage", rc.D_stage); logv!("D_work", rc.D_work); logv!("D_final", rc.D_final)
        logv!("bytes", rc.bytes_step); logv!("budget_met", budget_status(lg) === :met ? 1 : 0)
        # which limit held the step (dr5e: a missed budget was the work ceiling; dr6t: also at full
        # rank, far below it) -- a cap (work/rank) is marked rank-limited, a tolerance miss is not
        for lim in (:basis_limited, :root_limited, :truncation_limited, :rank_limited, :work_limited)
            logv!(string(lim), lim in rc.limits ? 1 : 0)
        end
        # plan Task 6 "measure before scaling": the phase timers and kernel counters of the step
        for k in (:t_cbe, :t_kry, :t_root, :t_trunc, :t_frame, :t_env, :t_step)
            logv!(string(k), getfield(info, k))
        end
        kc = rc.counters
        logv!("matvecs", info.krylov_dims)
        logv!("t_dd",kc.t_input_dd + kc.t_output_dd); logv!("t_qr", kc.t_qr)
        for k in (:n_stream, :n_stream_cols, :n_one_site, :n_root, :n_qr, :n_sector_new)
            logv!(string(k), getfield(kc, k))
        end
    else
        amp = Ref(0.0)
        info = tdvp2_step!(psi, G, ComplexF64(-im * h); hermitian = false, maxdim = MAXDIM,
                           trunc_thresh = 1e-14, maxiter = 40, conv_tol = TOLR, weight_tol = TOLT,
                           backward_report = amp)
        logv!("discarded", info.cumulative); logv!("krylov", info.krylov_dims)
        logv!("backward_amp", amp[]); logv!("D_final", info.max_bond)
    end
    logv!("h", h); logv!("wall", (time_ns() - t0) / 1e9)
    return psi
end

# An entangled start (DR_WARM): from the product state the first step's O(dt^{p+1}) error dominates a
# short run (dr5d, A1 L = 12, T = 0.5: BUG slopes 4-5, TDVP2 3), which hides the global order. Every
# run of a point evolves the same warm state (same steps, tolerances and maxdim, not DR_*), then
# normalises it: t = 0 and log P0 = 0 are re-based there.
if WARM > 0
    let nw = max(1, round(Int, WARM / 0.0125))
        for _ in 1:nw
            keep = KEEPK > 0 && maximum(bond_dims(M.psi); init = 1) == 1 ?
                   global_krylov_states(G, M.psi, KEEPK; tol_trunc = 1e-13, maxdim = MAXDIM) : Any[]
            cbe_bug_step!(M.psi, G, ComplexF64(-im * WARM / nw); hermitian = false, order = 3,
                          tol_basis = 1e-11, tol_root = 1e-13, tol_trunc = 1e-13, maxdim = MAXDIM,
                          closure_checks = CLOSURE, keep_states = keep)
        end
        canonical!(M.psi, 1)
        M.psi[1] = to_concrete(M.psi[1] / norm(M.psi[1]))
    end
end

centre_entropy(p) =(s = abs2.(bond_spectrum(copy(p), L ÷ 2)); s ./= sum(s);
                     -sum(x * log(x) for x in s if x > 1e-300; init = 0.0))
centre_sectors(p) = length(p[L ÷ 2].spaces[3])

# ── an ED reference for small systems ──────────────────────────────────────────────────────
const REF = env("DR_REF", (NONLOCAL && L <= 12) ? "ed" : "none")
dense_amps = nothing
if REF == "ed" && NONLOCAL
    include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
    include(joinpath(@__DIR__, "..", "tests", "common", "dissipative_fixtures.jl"))
    em = _emission_model(L; Jz = M.par["Jz"], gp = M.par["gamma_plus"], gm = M.par["gamma_minus"],
                         k = M.par["ka"], start = [:down for _ in 1:L])
    v0 = dense_state(M.psi)
    global dense_amps = (flow = _ed_flow(-1im * em.Gd, v0), v0 = v0)
end

# ── the run ─────────────────────────────────────────────────────────────────────────────────
tg = collect(0.0:EVERY:(TF + 1e-12))
result = Dict{String, Any}()
nrec = Ref(0)                     # observe runs once per grid time, in order
t_start = time()
if MODE === :no_jump
    tr = jump_trajectory(M.psi, G, M.ch, tg; dt = DT, seed = SEED, jumps = false,
                         step = step!, target_step_error = TARGET, check_every = CHECK,
                         observe = p -> begin
                             P = real(overlap(p, p))
                             E = real(mpo_energy(p, M.H)) / P
                             q = real(mpo_energy(p, M.Q)) / P
                             hq = real(overlap(mpo_product(M.H, p), mpo_product(M.Q, p))) / P
                             x = Dict("obs" => M.obs(p), "E" => E, "Q" => q, "HQ" => hq, "t" => tg[nrec[] + 1],
                                      "S" => centre_entropy(p), "bonds" => bond_dims(p),
                                      "sectors" => centre_sectors(p))
                             if dense_amps !== nothing
                                 # the state is the normalised conditional one: compare directions
                                 t = tg[nrec[] + 1]
                                 w = dense_amps.flow(t)
                                 v = dense_state(p)
                                 x["err_state"] = norm(v / norm(v) - w / norm(w) * (dot(w, v) / abs(dot(w, v))))
                                 x["logp_exact"] = log(real(dot(w, w)) / real(dot(dense_amps.v0, dense_amps.v0)))
                             end
                             nrec[] += 1
                             x
                         end)
    result["times"] = tr.times
    result["logp"] = tr.logp
    result["records"] = tr.obs
    result["checks"] = tr.checks
    result["steps"] = tr.steps
    result["nreject"] = tr.nreject
    result["flags"] = [Dict("t" => f.t, "kind" => string(f.kind), "value" => f.value) for f in tr.flags]
else
    ens = jump_ensemble(M.psi, G, M.ch, tg; ntraj = NTRAJ, seed = SEED, observe = M.obs, dt = DT,
                        step = step!, target_step_error = TARGET, check_every = CHECK)
    result["times"] = ens.times
    result["mean"] = ens.mean
    result["stderr"] = ens.stderr
    result["jumps_mean"] = ens.jumps_mean
    result["jumps_stderr"] = ens.jumps_stderr
    result["traj"] = [Dict("njumps" => tr.njumps[end], "nsteps" => tr.nsteps, "nevals" => tr.nevals,
                           "nreject" => tr.nreject, "flags" => length(tr.flags),
                           "events" => [Dict("t" => e.t, "channel" => e.channel) for e in tr.events])
                      for tr in ens.trajs]
end
result["wall"] = time() - t_start
result["names"] = M.names
result["steplog"] = LOG
result["peak_rss"] = Sys.maxrss()
result["config"] = Dict("model" => string(MODEL), "L" => L, "mode" => string(MODE), "int" => string(INT),
                        "dt" => DT, "T" => TF, "every" => EVERY, "order" => ORDER, "tol_basis" => TOLB,
                        "tol_root" => TOLR, "tol_trunc" => TOLT, "maxdim" => MAXDIM,
                        "closure_checks" => CLOSURE, "keep_krylov" => KEEPK, "keep_every" => KEEP_EVERY, "exact" => EXACT,
                        "start" => string(START), "warm" => WARM,
                        "target" => TARGET === nothing ? "none" : TARGET, "check_every" => CHECK,
                        "ntraj" => NTRAJ, "seed" => SEED, "ref" => REF, "params" => M.par,
                        "threads" => Threads.nthreads())
open(OUT, "w") do io
    JSON.print(io, result)
end
@printf("wrote %s (%.1f s, %d steps logged, peak RSS %.2f GB)\n", OUT, result["wall"],
        length(get(LOG, "h", Float64[])), Sys.maxrss() / 2^30)
