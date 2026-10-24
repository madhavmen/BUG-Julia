# benchmarks/dissipative_b1_smoke.jl -- dissipative_cbe_bug_implementation_plan.md, Task 4, B1
# physics smoke: ONE (U, J_link, kappa) point of the Hubbard junction with localized pair loss
# (`pair_loss_junction`). The leads are prepared decoupled (J_link = 0) in their ground states with a
# spin bias (left lead up-rich, right lead down-rich, sink empty) by normalised imaginary time, joined
# at t = 0, and evolved as a quantum-jump ensemble (`jump_ensemble`, BUG no-jump steps). Writes JSON
# for benchmarks/dissipative_t4_figures.py; grades nothing.
#
# Per output time, the ensemble mean and standard error over independent trajectories of every
# component of: n↑_j, n↓_j, (n↑n↓)_j for every site; j↑, j↓ on the sink's left bond (j0-1, j0) and
# right bond (j0, j0+1); N_L, N_R, S^z_L, S^z_R (leads exclude the sink); plus the mean jump count
# (each jump loses one pair) and the preparation's energy history.
#
# Env: B1_L (6), B1_U (4.0), B1_JLINK (1.0), B1_KAPPA (2.0), B1_NTRAJ (100), B1_T (4.0),
#      B1_DT (0.05), B1_EVERY (0.25), B1_SEED (1), B1_TOLB (1e-8), B1_MAXDIM (128), B1_OUT

using LinearAlgebra, Printf, JSON
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

const L = parse(Int, get(ENV, "B1_L", "6"))
const U = parse(Float64, get(ENV, "B1_U", "4.0"))
const JLINK = parse(Float64, get(ENV, "B1_JLINK", "1.0"))
const KAPPA = parse(Float64, get(ENV, "B1_KAPPA", "2.0"))
const NTRAJ = parse(Int, get(ENV, "B1_NTRAJ", "100"))
const TF = parse(Float64, get(ENV, "B1_T", "4.0"))
const DT = parse(Float64, get(ENV, "B1_DT", "0.05"))
const EVERY = parse(Float64, get(ENV, "B1_EVERY", "0.25"))
const SEED = parse(Int, get(ENV, "B1_SEED", "1"))
const TOLB = parse(Float64, get(ENV, "B1_TOLB", "1e-8"))
const MAXDIM = parse(Int, get(ENV, "B1_MAXDIM", "128"))
const TAG = @sprintf("L%d_U%+.1f_Jl%.2f_k%.2f_seed%d", L, U, JLINK, KAPPA, SEED)
const OUT = get(ENV, "B1_OUT", "b1_smoke_$(TAG).json")

set_symmetry!(:U1)
j0 = L ÷ 2
lead(n, s) = [isodd(k) ? s : :both for k in 1:n]          # (s, both, s, ...): a spin bias
occ = vcat(lead(j0 - 1, :up), [:empty], lead(L - j0, :down))

# ── preparation: decoupled leads, normalised imaginary time ──
prep = pair_loss_junction(L; J = 1.0, J_link = 0.0, U = U, kappa = 1.0, j0 = j0)
psi0 = hub4_product_state(occ)
t0 = time()
hist = ite_evolve!(psi0, prep.H, log_panel_steps(20.0, 8; per_panel = 4); order = 2,
                   tol_basis = 1e-9, tol_trunc = 1e-12, tol_root = 1e-12, maxdim = MAXDIM, retries = 3)
@printf("prep: E = %.12f after beta 20, %d failed steps, %.1f s\n", hist[end].e_after,
        count(r -> r.status === :failed, hist), time() - t0)

# ── the junction and its observables ──
m = pair_loss_junction(L; J = 1.0, J_link = JLINK, U = U, kappa = KAPPA, j0 = j0)
q = hub4_space()
nup = to_concrete(lock(q.Fu', 2) * q.Fu)
ndn = to_concrete(lock(q.Fd', 2) * q.Fd)
dbl = to_concrete(lock(nup, 1) * ndn)
cur = [hub4_current_mpo(L, b, s; J = m.Jb[b]) for b in (j0 - 1, j0) for s in 0:1]
left, right = 1:(j0 - 1), (j0 + 1):L
function observe(p)
    u = [real(site_expval(p, j, nup)) for j in 1:L]
    d = [real(site_expval(p, j, ndn)) for j in 1:L]
    x = [real(site_expval(p, j, dbl)) for j in 1:L]
    jc = [real(mpo_energy(p, c)) for c in cur]
    return vcat(u, d, x, jc, sum(u[left] + d[left]), sum(u[right] + d[right]),
                0.5 * sum(u[left] - d[left]), 0.5 * sum(u[right] - d[right]))
end
names = vcat(["nup_$j" for j in 1:L], ["ndn_$j" for j in 1:L], ["dbl_$j" for j in 1:L],
             ["jup_left", "jdn_left", "jup_right", "jdn_right"], ["N_L", "N_R", "Sz_L", "Sz_R"])

tg = collect(0.0:EVERY:TF)
t1 = time()
ens = jump_ensemble(psi0, m.G, m.channels, tg; ntraj = NTRAJ, seed = SEED, observe = observe,
                    dt = DT, step_kw = (tol_basis = TOLB, tol_root = 1e-12), tol_trunc = 1e-10,
                    maxdim = MAXDIM)
wall = time() - t1
nflags = sum(length(tr.flags) for tr in ens.trajs)
@printf("ensemble: %d trajectories, %.1f s, mean jumps %.3f, norm-growth flags %d\n", NTRAJ, wall,
        ens.jumps_mean[end], nflags)

open(OUT, "w") do io
    JSON.print(io, Dict(
        "config" => Dict("L" => L, "U" => U, "J_link" => JLINK, "kappa" => KAPPA, "j0" => j0,
                         "occ" => string.(occ), "ntraj" => NTRAJ, "T" => TF, "dt" => DT,
                         "seed" => SEED, "tol_basis" => TOLB, "maxdim" => MAXDIM),
        "prep" => Dict("beta" => cumsum([r.dbeta for r in hist]), "energy" => [r.e_after for r in hist],
                       "failed" => count(r -> r.status === :failed, hist)),
        "times" => tg, "names" => names,
        "mean" => ens.mean, "stderr" => ens.stderr,
        "jumps_mean" => ens.jumps_mean, "jumps_stderr" => ens.jumps_stderr,
        "events" => [length(tr.events) for tr in ens.trajs],
        "nsteps" => [tr.nsteps for tr in ens.trajs], "nevals" => [tr.nevals for tr in ens.trajs],
        "norm_growth_flags" => nflags, "wall" => wall))
end
println("wrote $OUT")
