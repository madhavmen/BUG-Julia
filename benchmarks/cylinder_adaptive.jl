# ADAPTIVE DEPTH AND ADAPTIVE EXPANSION BUDGET ON THE CYLINDER -- does either change the answer?
#
# Two DIFFERENT knobs are both reasonably called "adaptive growth", so both are scanned here
# rather than one being guessed at:
#
#   krylov_tol   the DEPTH stopping rule. `> 0` makes the Krylov recursion adaptive: it stops
#                when the next vector's contribution to `exp(tau*H)Theta` falls below the
#                tolerance (Saad's estimate). `0.0` pins the depth at `krylov_basis`.
#   growth       the EXPANSION BUDGET. `budget = ceil(growth*dmax) - r` decides how many new
#                directions CBE may admit per bond. Raising it lets the bond widen faster.
#
# ⛔ RUN WHERE A REFERENCE EXISTS. 4x4 with a binding cap is the ONLY cylinder configuration in
# this study that has both an exact answer (12870-state Sz=0 sector) and severe truncation -- and
# it is where `cbe_bug` was measured to BEAT both TDVPs. On 8x8 there is no reference at all, so
# "does it make a difference" could only be answered as a spread between schemes, which says they
# disagree and not which is right. Accuracy questions belong here.
#
# ⚠ THE BAR IS RE-RUN, NOT QUOTED. Both TDVPs are measured at the same cap in the same process,
# because their error moves with `maxdim` too.
#
# ⚠ AND `err_fnl` IS REPORTED FOR EVERY ARM. Raising `growth` can only matter if the budget was
# binding, and every measurement in this study so far says it is not (`err_fnl` ~ 1e-16 even when
# fully rank-starved). If `growth` changes nothing, that column is the reason.
#
# Run:  julia --project=. benchmarks/cylinder_adaptive.jl [Lx] [Ly] [maxdim] [t_max]

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const LX   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const LY   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
const CAP  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 48
const TMAX = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.4
const DT   = 0.05
const L    = LX * LY
const MI   = 16

const OUT = joinpath(@__DIR__, "results",
                     @sprintf("cylinder_adaptive_%dx%d_D%d.txt", LX, LY, CAP))
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:U1)
const BONDS = square_cylinder_bonds(LX, LY)
const JM    = square_cylinder_couplings(LX, LY)
const W     = square_cylinder_mpo(LX, LY)
const PSI0  = dimer_state(L)
const OPS   = [begin
                   Jm = zeros(Float64, L, L); Jm[i, j] = 1.0
                   pair_mpo(L, Jm, spin_vertices(L))
               end for (i, j) in BONDS]

bond_profile(psi) = (n2 = max(norm(copy(psi))^2, eps());
                     [real(mpo_energy(copy(psi), op)) / n2 for op in OPS])

function pair_profile_dense(v, states, idx, bonds)
    bit(b, j) = (b >> (j - 1)) & 1
    nrm = real(dot(v, v))
    out = zeros(Float64, length(bonds))
    for (n, (i, j)) in pairs(bonds)
        acc = 0.0 + 0im
        for (k, b) in enumerate(states)
            si, sj = bit(b, i), bit(b, j)
            acc += conj(v[k]) * v[k] * (si == sj ? 0.25 : -0.25)
            if si != sj
                bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                acc += conj(v[idx[bb]]) * v[k] * 0.5
            end
        end
        out[n] = real(acc) / nrm
    end
    return out
end

Hs, STATES, IDX = pairs_sparse(L, JM)
const REF = pair_profile_dense(
    expv_sparse(Hs, ComplexF64(-im * TMAX), dimer_vector(L, STATES, IDX); m = 30, tol = 1e-13),
    STATES, IDX, BONDS)

say(@sprintf("Heisenberg on a %dx%d cylinder -- %d sites, maxdim=%d, dt=%g, T=%g",
             LX, LY, L, CAP, DT, TMAX))
say(@sprintf("exact reference: %d states;  full rank %d -> cap %d BINDS (ratio %.2f)",
             length(STATES), 1 << (L ÷ 2), CAP, CAP / (1 << (L ÷ 2))))
say("")

function run(step!)
    psi, kry, secs, efnl = copy(PSI0), 0, 0.0, 0.0
    for _ in 1:round(Int, TMAX / DT)
        t0 = time_ns()
        info = step!(psi, ComplexF64(-im * DT))
        secs += (time_ns() - t0) / 1e9
        hasproperty(info, :krylov_dims) && (kry += info.krylov_dims)
        hasproperty(info, :err_fnl) && (efnl = max(efnl, info.err_fnl))
    end
    return (err = maximum(abs.(bond_profile(psi) .- REF)),
            states = maximum(state_bond_dims(psi)), kry = kry, secs = secs, efnl = efnl)
end

bug(; kw...) = (p, t) -> cbe_bug_step!(p, W, t; exact = true, maxdim = CAP,
                                       trunc_thresh = 1e-10, maxiter = MI, kw...)

const ARMS = [
    ("tdvp2 (bar)",              (p, t) -> tdvp2_step!(p, W, t; maxdim = CAP,
                                                       trunc_thresh = 1e-10, maxiter = MI)),
    ("tdvp_cbe1s (bar)",         (p, t) -> tdvp_cbe1s_step!(p, W, t; maxdim = CAP,
                                                            trunc_thresh = 1e-10, maxiter = MI)),
    # ── pinned depth: the control ────────────────────────────────────────────────────────
    ("m=3 pinned",               bug(krylov_basis = 3,  krylov_tol = 0.0)),
    ("m=30 pinned (to the cap)", bug(krylov_basis = 30, krylov_tol = 0.0)),
    # ── ADAPTIVE DEPTH: krylov_tol > 0 stops the recursion on Saad's contribution ─────────
    ("adaptive tol=1e-4",        bug(krylov_basis = 30, krylov_tol = 1e-4)),
    ("adaptive tol=1e-6 (DEF)",  bug(krylov_basis = 30, krylov_tol = 1e-6)),
    ("adaptive tol=1e-8",        bug(krylov_basis = 30, krylov_tol = 1e-8)),
    ("adaptive tol=1e-10",       bug(krylov_basis = 30, krylov_tol = 1e-10)),
    # ── ADAPTIVE EXPANSION BUDGET: `growth` decides how fast a bond may widen ─────────────
    ("growth=2 (DEF) tol=1e-8",  bug(krylov_basis = 30, krylov_tol = 1e-8, growth = 2.0)),
    ("growth=4        tol=1e-8", bug(krylov_basis = 30, krylov_tol = 1e-8, growth = 4.0)),
    ("growth=8        tol=1e-8", bug(krylov_basis = 30, krylov_tol = 1e-8, growth = 8.0)),
    ("growth=8        m=3",      bug(krylov_basis = 3,  krylov_tol = 0.0, growth = 8.0)),
]

say("="^108)
say("ADAPTIVE DEPTH AND ADAPTIVE BUDGET  (err = max |dSS| against the exact sector)")
say("="^108)
say(@sprintf("  %-26s %12s %8s %9s %11s %8s %8s",
             "arm", "err", "states", "krylov", "err_fnl", "sec", "vs bar"))
bar = Inf
res = Dict{String, Any}()
for (nm, st) in ARMS
    r = run(st)
    res[nm] = r
    endswith(nm, "(bar)") && (global bar = min(bar, r.err))
    say(@sprintf("  %-26s %12.4e %8d %9s %11s %8.1f %8s",
                 nm, r.err, r.states, r.kry == 0 ? "-" : string(r.kry),
                 r.kry == 0 ? "-" : @sprintf("%.3e", r.efnl), r.secs,
                 isfinite(bar) && !endswith(nm, "(bar)") ? @sprintf("%.2fx", r.err / bar) : "-"))
end

say("")
say("="^108)
say("DOES EITHER KNOB MOVE THE ANSWER?")
say("="^108)
pin3 = res["m=3 pinned"].err
for (label, arms) in (("adaptive DEPTH (tol)", ["adaptive tol=1e-4", "adaptive tol=1e-6 (DEF)",
                                                "adaptive tol=1e-8", "adaptive tol=1e-10"]),
                      ("adaptive BUDGET (growth)", ["growth=2 (DEF) tol=1e-8",
                                                    "growth=4        tol=1e-8",
                                                    "growth=8        tol=1e-8"]))
    errs = [res[k].err for k in arms]
    krys = [res[k].kry for k in arms]
    spread = maximum(errs) / minimum(errs)
    say(@sprintf("  %-26s error spread %.4f x   cost %d -> %d", label, spread,
                 minimum(krys), maximum(krys)))
    say(@sprintf("     %s", spread < 1.01 ? "INERT: the knob does not change the answer at all" :
                            spread < 1.1  ? "marginal (<10%)" : "LIVE: worth tuning"))
end
say("")
bugarms = [nm for (nm, _) in ARMS if !endswith(nm, "(bar)")]
bestnm = argmin(k -> res[k].err, bugarms)
say(@sprintf("  best cbe_bug arm: %s at %.4e (%.2fx bar)",
             bestnm, res[bestnm].err, res[bestnm].err / bar))
say(@sprintf("  the bar (best TDVP): %.4e", bar))
say("")
say("⚠ `growth` can only matter if the BUDGET was binding. Read the err_fnl column: every")
say("  measurement in this study has it at ~1e-16 even when fully rank-starved, which means the")
say("  CBE selection discards nothing and the closing truncation is what enforces the rank.")
close(IO_)
