# benchmarks/qtt/first_impressions.jl -- QTT plan §1.2A / U08: the no-symmetry DD-CBE
# first-impressions study, BEFORE any p = 4 stage or large trajectory.
#
# Four fixtures (GP-quiet, GP-structured, Heston-smooth, Heston-hard), each at its plan sizes and
# target ranks (capped at the field's TRUE rank -- nothing is padded), an active bond at the x|y
# interface and a quiet bond. Every arm sees the SAME incoming tensors and frames:
#   D0  exhaustive DD (exact = true)           D1  current adaptive holdout controller
#   D2  one-shot fixed width l in {8, 16, 32}   D3  hybrid (cold; and warm = predictor from a call)
# Offline audit from the materialised local A_DD: rho_L, absolute time-weighted miss, false accepts
# over QI_SEEDS seeds; cost = streamed columns, QR calls, useful directions, warm seconds, bytes.
# Then QI_STEPS full steps on the two hard fixtures, arms D1 / D3 / D2-16 / off (no CBE), with the
# GP density rebuild timed separately (f_rho) and f_CBE = t_cbe / t_step.
#
# ENV: QI_OUT (output dir), QI_QUICK=1 (smoke sizes), QI_SEEDS (30), QI_STEPS (10), QI_H (0.1).
# Writes QI_OUT/qtt_cbe_nosym_ablation.csv, dims.csv, steps.csv.

using LinearAlgebra, Printf, Random, Statistics
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate
const RC = BUGJulia.RSVDCBEBondUpdate
set_symmetry!(:none)

const OUT = get(ENV, "QI_OUT", "qi_out")
const QUICK = get(ENV, "QI_QUICK", "0") == "1"
const NSEED = parse(Int, get(ENV, "QI_SEEDS", QUICK ? "4" : "30"))
const NSTEP = parse(Int, get(ENV, "QI_STEPS", QUICK ? "3" : "10"))
const H = parse(Float64, get(ENV, "QI_H", "0.1"))
const TOLS = QUICK ? (1e-5,) : (1e-3, 1e-5, 1e-7)
mkpath(OUT)

_envs(psi, mpo, i, L) =
    (i == 1 ? boundary_channels(mpo) : left_channels(left_env_stack(psi, mpo; upto = i), i),
     i + 2 > L ? boundary_channels(mpo) :
                 right_channels(right_env_stack(psi, mpo; downto = i + 2), i + 2))

# ── fixtures ──

phi0(x) = pi^(-0.25) * exp(-x^2 / 2)
phi1(x) = sqrt(2) * x * phi0(x)
const HP = HestonParams(0.03, 0.01, 2.0, 0.09, 0.4, -0.7)

"(name, grid, cores, generator cores, tau, model-or-nothing)"
function fixture(name::Symbol, L::Int, chi::Int)
    n = L ÷ 2
    if name === :gp_quiet
        g = qtt_grid(n, n; x = (-8.0, 8.0), y = (-8.0, 8.0))
        C = tt_lincomb([(1.0, qtt_separable(g, phi0, phi0)), (0.25, qtt_separable(g, phi1, phi0)),
                        (0.15im, qtt_separable(g, phi0, phi1))]; maxdim = chi)
        m = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 1.0, omega = 1.0)
        return g, C, m, ComplexF64((-1im - m.gamma) * H)
    elseif name === :gp_structured
        g = qtt_grid(n, n; x = (-8.0, 8.0), y = (-8.0, 8.0))
        vort = ((-1.5, 0.5), (1.5, -0.5))
        # anonymous: two named local `f`s in one method are ONE function (the last wins)
        fv = (x, y) -> exp(-(x^2 + y^2) / 32) *
                       prod(((x - a) + 1im * (y - b)) / sqrt((x - a)^2 + (y - b)^2 + 0.25)
                            for (a, b) in vort)
        C = qtt_function(g, fv; tol = 1e-12, maxdim = chi)
        m = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 0.5, omega = 0.25)
        return g, C, m, ComplexF64((-1im - m.gamma) * H)
    elseif name === :heston_smooth
        g = qtt_grid(n, n; x = (-1.0, 1.0), y = (0.0, 0.5), periodic_x = false, periodic_y = false)
        C = qtt_separable(g, x -> exp(0.7x), v -> exp(-0.5v); maxdim = chi)
        return g, C, nothing, ComplexF64(H)
    elseif name === :heston_hard
        g = qtt_grid(n, n; x = (-1.0, 1.0), y = (0.0, 0.5), periodic_x = false, periodic_y = false)
        r, sx, sv = -0.7, 0.15, 0.05
        fh = (x, v) -> exp(-((x / sx)^2 - 2r * (x / sx) * ((v - 0.1) / sv) + ((v - 0.1) / sv)^2) /
                           (2 * (1 - r^2)))
        C = qtt_function(g, fh; tol = 1e-12, maxdim = chi)
        return g, C, nothing, ComplexF64(H)
    end
    error("unknown fixture $name")
end

"The solver MPO at the fixture state (GP: H[psi], density rebuilt and timed)."
function generator(g, C, m)
    m === nothing && return qtt_to_mpo(heston_op(g, HP)), (t_rho = 0.0, chi_rho = 0)
    psi = qtt_mps(C)
    mpo, info = gp_hamiltonian(m, psi, 0.0)
    return mpo, (t_rho = info.t_rho, chi_rho = info.chi_rho)
end

# ── one local problem, every arm ──

function local_dd(f, mpo, i, lenv, renv)
    Xd = RC._none_block(apply_h_two_site(frame_theta(f), mpo, i, lenv, renv))
    cl, cr = size(Xd, 1), size(Xd, 4)
    Xm = reshape(Xd, 2cl, 2cr)
    Um = reshape(RC._none_block(f.U0), 2cl, :)
    Vm = reshape(RC._none_block(f.V0), :, 2cr)
    A = (I - Um * Um') * Xm * (I - Vm' * Vm)
    return Xm, A
end

function run_arm(arm, b, alloc, seed; mem = nothing)
    pol, width, exact = arm === :D3 || arm === :D3warm ? (:hybrid, 0, false) :
                        arm === :D0 ? (:adaptive, 0, true) :
                        arm === :D1 ? (:adaptive, 0, false) :
                        (:adaptive, parse(Int, String(arm)[4:end]), false)        # :D2_8 etc.
    kc = KernelCounters()
    lg = RC.AdaptiveLedger()
    ex = dd_cbe_expand(b.f, b.mpo, b.i, b.lenv, b.renv, :left; alloc = alloc, ledger = lg,
                       exact = exact, dd_width = width, dd_policy = pol, dd_memory = mem,
                       rng = MersenneTwister(seed), kc = kc)
    return ex, kc, lg
end

const ARMS = (:D0, :D1, :D2_8, :D2_16, :D2_32, :D3, :D3warm)

function bond_rows!(rows, name, L, chi, g, C, m, tau, i, tag)
    mpo, ginfo = generator(g, C, m)
    psi = qtt_mps(C; center = i)
    f = bond_frame(psi, i)
    lenv, renv = _envs(psi, mpo, i, qtt_length(g))
    b = (; f, mpo, i, lenv, renv)
    Xm, A = local_dd(f, mpo, i, lenv, renv)
    nA, nX = norm(A), norm(Xm)
    trivial = nA <= 1e-10 * max(nX, eps())
    r = f.old_rank
    mdim = leg_dim(f.U0, 1) * 2 - r
    ndim = 2 * leg_dim(f.V0, 3) - r
    nrm = norm(psi)
    for tb in TOLS
        alloc = basis_allocation(AdaptiveBudget(tb, 1e-7, 1e-10), qtt_length(g) - 1; tau = tau,
                                 nrm = nrm, rule = :linear)
        for arm in ARMS
            if startswith(String(arm), "D2") && parse(Int, String(arm)[4:end]) > ndim
                continue                                     # admissible widths only
            end
            nfalse, nmet, miss_all = 0, 0, Float64[]
            st_cols = Int[]; st_dirs = Int[]; st_qr = Int[]; branch = ""
            for seed in 1:(arm === :D0 ? 1 : NSEED)
                mem = arm === :D3warm ? DDMemory() : arm === :D3 ? DDMemory() : nothing
                arm === :D3warm && run_arm(:D3, b, alloc, seed + 1000; mem = mem)
                ex, kc, lg = run_arm(arm, b, alloc, seed; mem = mem)
                U = reshape(RC._none_block(ex.U_ex), size(Xm, 1), :)
                miss = norm((I - U * U') * A)
                push!(miss_all, miss)
                met = lg.tolerance_met
                nmet += met
                nfalse += (met && alloc.scale * miss > alloc.sample) ? 1 : 0
                push!(st_cols, kc.n_stream_cols); push!(st_dirs, ex.n_new_l); push!(st_qr, kc.n_qr)
                mem === nothing || (branch = String(mem.trace[end].branch))
            end
            # warm timing: 3 repeats after one JIT call, same seed
            run_arm(arm, b, alloc, 1; mem = arm in (:D3, :D3warm) ? DDMemory() : nothing)
            ts = Float64[]; bs = Int[]
            for rep in 1:3
                mem = arm === :D3warm ? DDMemory() : arm === :D3 ? DDMemory() : nothing
                arm === :D3warm && run_arm(:D3, b, alloc, 1000; mem = mem)
                t = @timed run_arm(arm, b, alloc, 1; mem = mem)
                push!(ts, t.time); push!(bs, t.bytes)
            end
            push!(rows, (fixture = name, L = L, chi_target = chi, chi_true = r, bond = i, tag = tag,
                         m = mdim, n = ndim, normA_rel = nA / max(nX, eps()), trivial = trivial,
                         tol_basis = tb, share = alloc.sample, scale = alloc.scale, arm = arm,
                         branch = branch, seeds = length(miss_all),
                         rhoL_med = trivial ? NaN : median(miss_all) / nA,
                         rhoL_max = trivial ? NaN : maximum(miss_all) / nA,
                         miss_tw_med = alloc.scale * median(miss_all),
                         met_frac = nmet / length(miss_all), false_accepts = nfalse,
                         cols_med = median(st_cols), dirs_med = median(st_dirs),
                         qr_med = median(st_qr), t_med = median(ts), bytes_med = median(bs),
                         t_rho = ginfo.t_rho, chi_rho = ginfo.chi_rho))
            println(rows[end]); flush(stdout)
        end
    end
end

function write_csv(path, rows)
    isempty(rows) && return
    ks = keys(rows[1])
    open(path, "w") do io
        println(io, join(ks, ","))
        for r in rows
            println(io, join((string(getfield(r, k)) for k in ks), ","))
        end
    end
end

# ── 1. local problems ──

const SIZES = QUICK ? Dict(:gp_quiet => [(12, 8)], :gp_structured => [(12, 16)],
                           :heston_smooth => [(12, 8)], :heston_hard => [(12, 16)]) :
              Dict(:gp_quiet => [(12, 8), (16, 16)], :gp_structured => [(16, 16), (16, 32), (20, 64)],
                   :heston_smooth => [(12, 8), (16, 16)], :heston_hard => [(16, 16), (16, 32), (20, 64)])

rows = NamedTuple[]
dims = NamedTuple[]
for name in (:gp_quiet, :gp_structured, :heston_smooth, :heston_hard), (L, chi) in SIZES[name]
    t0 = time()
    g, C, m, tau = fixture(name, L, chi)
    rk = tt_ranks(C)
    for k in 1:(L - 1)                               # the actual non-sector DD sizes, every bond
        chl = k == 1 ? 1 : rk[k - 1]
        chr = k == L - 1 ? 1 : rk[k + 1]
        push!(dims, (fixture = name, L = L, chi_target = chi, bond = k, chi = rk[k],
                     nL = 2chl - rk[k], nR = 2chr - rk[k]))
    end
    println("FIXTURE $name L=$L chi=$chi ranks=$rk built in $(round(time() - t0; digits = 1)) s")
    for (i, tag) in ((g.nx, :active), (2, :quiet))
        bond_rows!(rows, name, L, chi, g, C, m, tau, i, tag)
    end
    write_csv(joinpath(OUT, "qtt_cbe_nosym_ablation.csv"), rows)
    write_csv(joinpath(OUT, "dims.csv"), dims)
end

# ── 2. full steps on the two hard fixtures ──

steps = NamedTuple[]
for name in (:gp_structured, :heston_hard)
    L, chi = QUICK ? (12, 16) : (16, 32)
    g, C, m, tau = fixture(name, L, chi)
    Ahes = m === nothing ? qtt_to_mpo(heston_op(g, HP)) : nothing
    for arm in (:ref, :D1, :D3, :D2_16, :off)
        psi = qtt_mps(C)
        mem = arm === :D3 ? DDMemory() : nothing
        kw = arm === :ref ? (exact = true, tol_basis = 1e-9) :
             arm === :D1 ? (tol_basis = 1e-5,) :
             arm === :D3 ? (tol_basis = 1e-5, dd_policy = :hybrid, dd_memory = mem) :
             arm === :D2_16 ? (tol_basis = 1e-5, dd_width = 16) :
             (tol_basis = 1e-5, dd_policy = :off)
        for s in 1:NSTEP
            t = (s - 1) * H
            trho = 0.0
            tw = @elapsed begin
                if m === nothing
                    info = cbe_bug_step!(psi, Ahes, tau; kw...,
                                         tol_trunc = 1e-8, maxdim = 128, hermitian = false)
                else
                    info, hinfo = gp_bug_step!(psi, m, t, H; kw..., tol_trunc = 1e-8, maxdim = 128)
                    trho = hinfo.t_rho
                end
            end
            push!(steps, (fixture = name, arm = arm, step = s, t = s * H, wall = tw,
                          t_step = info.t_step, t_cbe = info.t_cbe, t_kry = info.t_kry,
                          t_root = info.t_root, t_rho = trho,
                          f_cbe = info.t_cbe / max(info.t_step, eps()),
                          f_rho = trho / max(tw, eps()), chi_max = maximum(bond_dims(psi)),
                          D_stage = info.record.D_stage, limits = join(info.record.limits, "|"),
                          cols = info.record.counters.n_stream_cols,
                          state = qtt_cores(psi)))
        end
    end
    # errors against the reference arm, step by step
    ref = Dict(r.step => r.state for r in steps if r.fixture === name && r.arm === :ref)
    for (k, r) in pairs(steps)
        r.fixture === name || continue
        e = norm(vec(tt_dense(r.state)) - vec(tt_dense(ref[r.step]))) /
            norm(vec(tt_dense(ref[r.step])))
        steps[k] = merge(r, (err_vs_ref = e,))
    end
end
write_csv(joinpath(OUT, "steps.csv"),
          [Base.structdiff(r, NamedTuple{(:state,)}) for r in steps])
println("QI-DONE")
