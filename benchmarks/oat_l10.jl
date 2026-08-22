# RECREATION OF arXiv:2208.10972 FIG. 2 -- the one-axis twisting model -- at L = 10.
#
#     H_OAT = (Σ_ℓ S^z_ℓ)^2 / 2 ,      |Ψ(0)⟩ = ⊗_ℓ |→⟩_ℓ ,      observable  S^x_tot(t)
#
# The paper runs L = 100, δ = 0.01, D_max = 500, Z2 spin symmetry, to t = 12π (three revival
# cycles). This runs the SAME protocol at L = 10 under `:none`, which is the size at which the
# question is CORRECTNESS AGAINST AN EXACT REFERENCE rather than cost.
#
# WHY L = 10 IS A SHARPER TEST THAN L = 100, NOT A WEAKER ONE. At L = 100 the paper can only
# check `S^x_tot` against its closed form, and must judge panels (b) and (c) by eye. At L = 10
# EVERYTHING is closed form -- `S^x_tot(t) = (L/2)cos^(L-1)(t/2)`, the Schmidt spectrum, the
# entanglement entropy and the exact bond dimension all come from
# `tests/common/analytic_reference.jl` with no diagonalisation and no propagation anywhere. Every
# panel below therefore carries its own exact curve, and panel (d) is a true error, not a drift.
#
# THE RANK CEILING, and why it sets every `maxdim` chosen here. OAT's only entangling factor is
# `exp(-i t a b)` with `a`, `b` the two block magnetisations, so the exact Schmidt rank at the
# cut after site `n` is `min(n, L-n) + 1` FOR ALL TIME -- 6 at the centre for L = 10. That is a
# hard ceiling, not a tolerance: `maxdim >= 6` is full rank and the sweeps are exact, `maxdim < 6`
# is a genuine truncation. Both regimes are run, because a plot showing only one of them would be
# either a roundoff measurement or a truncation measurement passing itself off as an integrator
# comparison.
#
# usage:  julia --project=. benchmarks/oat_l10.jl \
#             [L] [dt] [tmax] [schemes] [maxdims] [every] [warmup] [sym]
#   e.g.  julia --project=. benchmarks/oat_l10.jl 10 0.01 12.566 all 2,3,4,6,16
#         julia --project=. benchmarks/oat_l10.jl 10 0.05 12.566 cbe_bug 6 4 16 Z2
#
# `warmup` grades the FIRST step into that many sub-steps (default 1 = plain) and `sym` selects
# the symmetry (default `Z2`, the paper's own; `none` is the dense control). Both reach the
# output filename, so arms never overwrite each other.
#
# Writes ONE streamed CSV to benchmarks/results/, plotted by benchmarks/plot_oat.py.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
Random.seed!(42)

const HERE = dirname(abspath(@__FILE__))
# The closed forms. TEST-SIDE FILE ON PURPOSE: it shares no code with `src/`, which is the only
# reason the curves it produces count as a reference for the curves `src/` produces.
include(joinpath(HERE, "..", "tests", "common", "analytic_reference.jl"))

const L       = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 10
const DT      = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
# 4π is ONE full revival cycle: `cos^(L-1)(t/2)` returns to `+1` there (at 2π it is at `-1` for
# even `L`, the negative revival). The paper's 12π is three of these.
const TMAX    = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 4π
const ALL_SCHEMES = ("tdvp_cbe1s", "tdvp2", "cbe_bug")
const SCHEMES = length(ARGS) >= 4 && ARGS[4] != "all" ?
    Tuple(String.(split(ARGS[4], ","))) : ALL_SCHEMES
const MAXDIMS = length(ARGS) >= 5 ?
    Tuple(parse.(Int, split(ARGS[5], ","))) : (2, 3, 4, 6, 16)
const CUTOFF  = 1e-14
# Sample every `EVERY` steps, i.e. every 0.2 in time by default -- ~60 points per curve over
# `4π`, which is more than a smooth `cos^(L-1)(t/2)` needs. The diagnostics are not free (an
# `S^x` profile is a centre sweep, the entropy another, the energy a full contraction), so
# sampling every step spends a large fraction of the run measuring rather than integrating.
const EVERY   = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : max(1, round(Int, 0.2 / DT))
# ⛔ KRYLOV DEPTH IS THE COST KNOB HERE, AND IT IS NOT THE DEFAULT 30. `lanczos_expv` exits on
# BREAKDOWN only, so every solve burns its full `maxiter` whether or not it converged long
# before -- measured in this repo at 27.4/30 used on average. A sweep is `O(L)` solves, so the
# depth multiplies the whole run.
#
# 12 IS CHOSEN, NOT GUESSED -- BUT ONLY AT L = 10. `‖H_OAT‖ ≈ L²/8` (12.5 at L = 10) and each
# half-sweep carries `tau/2`, so the Krylov argument is `‖H‖·dt/2 ≈ 0.31` at `dt = 0.05`. The
# Lanczos error is bounded by roughly `(‖H‖t)^m/m!`, i.e. `0.31^12/12! ≈ 1e-15` -- below the
# `1e-14` truncation threshold, so the solve is not what limits any number reported here.
#
# ⛔ THAT ARGUMENT DOES NOT SURVIVE LARGER `L`, AND A FIXED 12 WOULD SILENTLY BECOME THE
# LIMITER. `‖H‖` grows like `L²`, so the Krylov argument grows with it and the bound degrades
# fast:
#
#     L = 10   arg 0.31   12 gives 1e-15   (fine -- this is where 12 came from)
#     L = 12   arg 0.45   12 gives 1e-13
#     L = 16   arg 0.80   12 gives 1e-10
#     L = 20   arg 1.25   12 gives 3e-08   <- above cbe_bug's measured 5e-06? no, but only just
#
# At L = 20 the depth is within two orders of the accuracy being claimed, which is no margin at
# all -- and the failure mode is invisible, because a too-shallow solve does not error, it just
# quietly caps every arm at the same floor and makes the schemes look equal. So the depth is
# DERIVED from `(‖H‖·dt/2)^m/m! < 1e-14` rather than fixed, and floored at 12 so nothing
# regresses against the L = 10 numbers already in hand.
"Smallest Krylov depth whose Lanczos bound `(‖H‖·dt/2)^m/m!` clears the truncation threshold."
function _krylov_depth(L::Int, dt::Float64; thresh = 1e-14, lo = 12, hi = 40)
    arg = (L^2 / 8) * dt / 2
    m = lo
    while m < hi && arg^m / factorial(big(m)) > thresh
        m += 1
    end
    return m
end
const MAXITER = length(ARGS) >= 10 ? parse(Int, ARGS[10]) : _krylov_depth(L, DT)

# ── the graded first step ────────────────────────────────────────────────────
#
# THE FIRST STEP IS THE WHOLE ERROR BUDGET FOR CBE-BUG, and `warmup` is what pays it down.
# MEASURED at L = 10, `dt = 0.05`, `maxdim = 6` (the exact rank ceiling): `cbe_bug`'s energy
# error is 1.561e-3 after the FIRST step alone and 1.565e-3 after a full 4π run -- 99.7% of it is
# paid in step one, where the bond has to grow 1 -> 6 out of the product state `⊗|→⟩`. 2-site
# TDVP pays 5e-15 at the same point.
#
# `warmup = m` takes that first step as `m` sub-steps of `dt/m`. Elapsed physical time is
# unchanged (`m · dt/m = dt`), so arms at different `m` are compared at the same `t` and the
# integrator itself is untouched -- this changes how the run is STARTED, not what a step does.
#
# MEASURED EFFECT (T = 2, same settings), and note the SECOND column is the one that justifies
# the knob, since an energy-only improvement would not be worth having:
#
#     m         dE(T)        |δS^x|(T)
#     1         1.565e-3     1.026e-2
#     2         3.908e-4     2.446e-3
#     4         9.767e-5     5.956e-4
#     8         2.441e-5     1.469e-4
#     16        6.104e-6     3.646e-5
#
# Both fall by 4.00 per doubling, i.e. as `m^-2` -- NOT the `m^-1` that accumulating `m` local
# errors of size `(dt/m)^2` would give. So the error is not spread over the ramp at all: it is
# one localised event whose size is set by the FIRST sub-step alone.
#
# ⛔ THIS KNOB IS OBSOLETE WITH `kaug = true`, AND THE EXPLANATION IT ONCE CARRIED WAS WRONG.
#
# The numbers above are real, but the mechanism attributed to them was not. The claim was that a
# `chi = 1` start forces a structural rank-DOUBLING ceiling on any frozen-basis scheme. It does
# not: the actual cause is that `cbe_bug`'s half-sweep basis update REPLACED the CBE frame
# instead of unioning with it, so CBE's admitted directions never reached the state (see `KAUG`
# below and `docs/USAGE.md` §6b). The graded start helped only because taking smaller first
# steps limits how much a too-narrow basis can cost.
#
# WITH THE DEFECT FIXED THERE IS NO BOOTSTRAP ERROR TO PAY: `kaug = true` gives dE = 1.4e-15 on
# the FIRST step from the product state, against 1.561e-3 historically. So `warmup` buys nothing
# and the default stays 1. The knob is kept because grading a start is occasionally useful on its
# own terms, not because this benchmark needs it.
const WARMUP  = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 1
WARMUP >= 1 || error("warmup must be >= 1, got $WARMUP")

# ── the symmetry ─────────────────────────────────────────────────────────────
#
# `:Z2` IS THE PAPER'S OWN CHOICE -- arXiv:2208.10972 Fig. 2 caption: "computed using δ=0.01 and
# Z2 spin symmetry". The conserved quantity is the GLOBAL SPIN FLIP `P = Π_ℓ 2 S^x_ℓ`, not
# `S^z_tot`: `H = (Σ S^z)²/2` is even in `S^z`. In the sigma^x basis of `z2_ising.jl` that flip
# is diagonal, so `S^x` is the diagonal charge-neutral operator and `⊗|→⟩` is a single charge-0
# sector at `chi = 1` -- the paper's "an MPS with D = 1", literally.
#
# THE VIRTUAL DIMENSION IS O(L) HERE, NOT 3, and that is structural. In the sigma^x basis `S^z`
# is the CHARGE-1 FLIP and hence rank 3, so its channel needs a charge-1 transport block;
# `long_range_mpo` refuses rank-3 terms outright, so `:Z2` routes through `pair_mpo`, which
# builds charged transport. Exact either way -- at L = 10 the cost is irrelevant.
#
# `:none` is the dense control in the sigma^z basis: a DIFFERENT BASIS for the same physics, so
# every observable must agree digit for digit. MEASURED over a 40-step run at `maxdim = 6`,
# `max|Z2 - none|` on `S^x_tot(t)`: 1.6e-12 (tdvp_cbe1s), 5.4e-13 (tdvp2), 1.5e-13 (cbe_bug),
# and `E(0)`, `S^x_tot(0)` and the norm are bit-identical. `tests/sweeps/test_oat.jl` asserts it.
# `:U1` is refused for physics reasons -- see `_require_sx_measurable`.
const SYM = length(ARGS) >= 8 ? Symbol(ARGS[8]) : :Z2

# ── the basis-update fix ─────────────────────────────────────────────────────
#
# `kaug = true` UNIONS the CBE frame into `cbe_bug`'s half-sweep basis instead of letting the
# basis update REPLACE it. Default TRUE here, because the historical behaviour is a defect and
# plotting it as "CBE-BUG" misrepresents the method -- see `docs/USAGE.md` §6b.
#
# WHAT IT FIXES. The assembly writes `psi[i] = W[i]`, so the bases ARE the state, but each
# half-sweep kept the CBE frame on the OPPOSITE side from the basis it was building and dropped
# the one that becomes the state. MEASURED at L=10, step 1: `n_new = [1,2,2,2,2,2,2,2,1]` either
# way, but `expanded` = [2,2,2,...] with the update and [2,3,3,3,...] without -- one admitted
# direction per interior bond discarded. Same signature on Heisenberg and Haldane-Shastry, so
# this is not an OAT pathology; the DAMAGE is ordered by the Hamiltonian's range (dE 4.9e-7 nn,
# 1.5e-3 HS and OAT), which is why nearest-neighbour benchmarks never showed it.
#
# MEASURED EFFECT, infidelity against the exact state (L=10, maxdim=6, dt=0.05, 20 steps):
#
#     tdvp2 1.99e-06    tdvp_cbe1s 1.99e-06    historical 3.97e-05    kaug 3.27e-11
#
# and `kaug` converges to chi = [2,3,4,5,6,5,4,3,2], which is EXACTLY `min(i, L-i)+1`, the exact
# Schmidt rank at every bond -- the historical arm over-fills the interior to [2,3,6,6,6,6,6,3,2].
# It adds NO `expv` calls; the union is two projections and two small SVDs per site.
#
# ⛔ DO NOT RANK THESE BY ENERGY DRIFT. `kstep = false` conserves energy to 7.4e-15 -- the best
# of any arm -- and is 12x WORSE than tdvp2 on state infidelity. The orderings disagree.
const KAUG = length(ARGS) >= 9 ? parse(Bool, ARGS[9]) : true

# THE SCHEME SUBSET TAGS THE FILENAME. Without it, running the three schemes as three concurrent
# processes -- which is the point of accepting a subset at all, and what makes a 15-arm campaign
# finish in the time of a 5-arm one on this machine -- has all three open the SAME path `"w"` and
# clobber each other. `plot_oat.py` globs `oat_L*.csv` and unions the rows, so a tagged run and an
# untagged one plot identically.
const TAG = (length(ARGS) >= 4 && ARGS[4] != "all" ? "_" * replace(ARGS[4], "," => "-") : "") *
            (length(ARGS) >= 5 ? "_D" * replace(ARGS[5], "," => "-") : "") *
            # `warmup` and the symmetry BOTH change the numbers, so both have to reach the
            # filename or two arms overwrite each other and the plot silently shows one of them.
            (WARMUP > 1 ? "_w$(WARMUP)" : "") * "_$(SYM)" *
            # `kaug` changes `cbe_bug`'s numbers by orders of magnitude, so a fixed run must not
            # land on top of a historical one.
            (KAUG ? "" : "_nokaug")
const OUT = joinpath(HERE, "results",
                     @sprintf("oat_L%d_dt%.4f_T%.3f%s.csv", L, DT, TMAX, TAG))
mkpath(dirname(OUT))

# ── the integrators ──────────────────────────────────────────────────────────

"""
    stepper(scheme, mpo, maxdim) -> (psi, tau) -> info

One step of each integrator behind one signature.

`cbe_bug` is INTERLEAVED (`decoupled = false`): that is the reference's own ordering
(`TDVPSweepCBE1Si.m` -- expand bond `si`, immediately evolve site `si`), so the environment chain
runs through the EVOLVED frames. It is also `cbe_bug_step!`'s default; it is passed explicitly
here so the choice is visible at the call site rather than inherited silently.
"""
function stepper(scheme::String, mpo, maxdim::Int)
    if scheme == "tdvp_cbe1s"
        return (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = maxdim,
                                            trunc_thresh = CUTOFF, maxiter = MAXITER)
    elseif scheme == "tdvp2"
        return (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = CUTOFF,
                                       maxiter = MAXITER)
    elseif scheme == "cbe_bug"
        # ⛔ `rexpand = false` KEEPS `KAUG` MEANINGFUL. `rexpand` is now the default and overrides
        # `kaug`; without pinning it off, every value of `KAUG` would run the same computation and
        # this benchmark's union A/B would silently compare an arm with itself.
        return (p, tau) -> cbe_bug_step!(p, mpo, tau; kaug = KAUG, rexpand = false,
                                         maxdim = maxdim,
                                         trunc_thresh = CUTOFF, maxiter = MAXITER)
    end
    error("unknown scheme $scheme -- one of $(ALL_SCHEMES)")
end

"`err_fnl` / `discarded` / `krylov_dims` from whichever Info type came back. Probed under BOTH
spellings of the Krylov counter: `JanBugInfo` says `krylov_dim`, the rest say `krylov_dims`, and a
missing count must never be indistinguishable from a zero count (it would read as a free step)."
info_fields(info) = (
    hasproperty(info, :err_fnl)     ? info.err_fnl     : 0.0,
    hasproperty(info, :discarded)   ? info.discarded   : 0.0,
    hasproperty(info, :krylov_dims) ? info.krylov_dims :
    hasproperty(info, :krylov_dim)  ? info.krylov_dim  : 0,
)

# ── observables ──────────────────────────────────────────────────────────────

"Von Neumann entropy of the centre cut, natural log -- the base arXiv:2208.10972 Fig. 2(b) uses."
function centre_entropy(psi::SymMPS)
    p = abs2.(bond_spectrum(copy(psi), length(psi) ÷ 2))
    s = sum(p)
    s > 0 || return 0.0
    p ./= s
    return -sum(q * log(q) for q in p if q > 1e-300; init = 0.0)
end

"""
    energy(psi, mpo) -> Float64

The RAYLEIGH QUOTIENT `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`, not the bare `mpo_energy`.

⛔ THE DIVISION IS LOAD-BEARING IN PANEL (d). A fixed-rank BUG step projects onto the frame
product basis and drops whatever it cannot span, so `‖ψ‖` decays; reporting the bare
`⟨ψ|H|ψ⟩` would print that norm loss as an energy drift, in the panel whose entire claim is that
real-time energy is conserved and any drift is integrator error. The state itself is left
UN-normalised so the `S^x_tot` amplitude still shows the loss where it belongs -- in panel (a).
"""
energy(psi, mpo) = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())

# ── the run ──────────────────────────────────────────────────────────────────

const COLS = ("scheme", "sym", "warmup", "L", "maxdim", "dt", "t",
              "sx", "sx_exact", "sx_err", "sx_density_err",
              "ee", "ee_exact", "maxbond", "centrebond", "rank_exact",
              "energy", "dE", "discarded", "err_fnl", "krylov", "elapsed")

row(io; kw...) = (println(io, join((string(kw[Symbol(c)]) for c in COLS), ","));  flush(io))

function main()
    set_symmetry!(SYM)
    n = L ÷ 2
    mpo = oat_mpo(L)
    @printf("OAT L=%d dt=%g tmax=%g nsteps=%d every=%d\n", L, DT, TMAX, round(Int, TMAX / DT),
            EVERY)
    # ⛔ THE VIRTUAL DIMENSION DEPENDS ON THE SYMMETRY, so do not print a constant. Under
    # `:none` the λ=1 self-loop gives 3 whatever `L` is. Under `:Z2` the sigma^x basis makes
    # `S^z` the rank-3 charge-1 flip, which no self-loop can carry, so the operator is built by
    # `pair_mpo` at O(L) instead -- exact either way, and at L = 10 the cost is irrelevant.
    @printf("MPO virtual dims = %s   (%s)\n", string(mpo_virtual_dims(mpo)),
            SYM === :Z2 ? "O(L) -- pair_mpo, charged Sz channel" :
                          "3 whatever L is -- the λ=1 self-loop")
    @printf("exact centre rank ceiling = %d  (min(n, L-n)+1)\n", min(n, L - n) + 1)
    @printf("schemes = %s   maxdims = %s\n", string(SCHEMES), string(MAXDIMS))
    @printf("symmetry = %s%s   warmup = %d%s   kaug = %s\n", SYM,
            SYM === :Z2 ? " (the paper's own)" : " (dense control)", WARMUP,
            WARMUP > 1 ? " sub-steps on step 1" : "", KAUG)
    # The Krylov depth is derived from ‖H‖·dt/2, not fixed -- print the bound so a run whose
    # accuracy is limited by the SOLVE rather than by the integrator is visible in its own log.
    @printf("krylov depth = %d   (arg %.3f, Lanczos bound %.1e -- must sit under the 1e-14 cutoff)\n",
            MAXITER, (L^2 / 8) * DT / 2,
            Float64(((L^2 / 8) * DT / 2)^MAXITER / factorial(big(MAXITER))))
    flush(stdout)

    nsteps = round(Int, TMAX / DT)
    open(OUT, "w") do io
        println(io, join(COLS, ","))
        for scheme in SCHEMES, maxdim in MAXDIMS
            psi = x_polarized_state(L)
            step = stepper(scheme, mpo, maxdim)
            E0 = energy(psi, mpo)
            t0 = time()
            # t = 0 sample, so every curve starts from the product state rather than from the
            # first sampled step.
            row(io; scheme, sym = SYM, warmup = WARMUP, L, maxdim, dt = DT, t = 0.0,
                sx = total_sx(copy(psi)), sx_exact = oat_total_sx(L, 0.0), sx_err = 0.0,
                sx_density_err = 0.0, ee = 0.0, ee_exact = oat_entropy(L, n, 0.0),
                maxbond = maximum(bond_dims(psi)), centrebond = leg_dim(psi[n], 3),
                rank_exact = oat_rank(L, n, 0.0), energy = E0, dE = 0.0,
                discarded = 0.0, err_fnl = 0.0, krylov = 0, elapsed = 0.0)

            disc = 0.0; efnl = 0.0; kd = 0
            ok = true
            for k in 1:nsteps
                # THE GRADED FIRST STEP. `m` sub-steps of `dt/m` cover exactly `dt` of physical
                # time, so `t = k*DT` stays correct and every arm is sampled at the same `t`.
                taus = (k == 1 && WARMUP > 1) ?
                    fill(ComplexF64(-im * DT / WARMUP), WARMUP) : [ComplexF64(-im * DT)]
                try
                    for tau in taus
                        info = step(psi, tau)
                        f = info_fields(info)
                        efnl = max(efnl, f[1]); disc = max(disc, f[2]); kd += f[3]
                    end
                catch e
                    @printf("  %s maxdim=%d THREW at step %d: %s\n", scheme, maxdim, k,
                            sprint(showerror, e)[1:min(end, 140)]); flush(stdout)
                    ok = false
                    break
                end
                k % EVERY == 0 || continue
                t = k * DT
                sx = total_sx(copy(psi))
                sxe = oat_total_sx(L, t)
                ee = centre_entropy(psi)
                E = energy(psi, mpo)
                row(io; scheme, sym = SYM, warmup = WARMUP, L, maxdim, dt = DT, t = t,
                    sx = sx, sx_exact = sxe, sx_err = abs(sx - sxe),
                    sx_density_err = abs(sx - sxe) / L,
                    ee = ee, ee_exact = oat_entropy(L, n, t),
                    maxbond = maximum(bond_dims(psi)), centrebond = leg_dim(psi[n], 3),
                    rank_exact = oat_rank(L, n, t), energy = E, dE = abs(E - E0),
                    discarded = disc, err_fnl = efnl, krylov = kd,
                    elapsed = time() - t0)
            end
            ok || continue
            @printf("  %-11s maxdim=%-3d  |δS^x| at t=%.2f = %.3e   xi=%.1e   %.1f s\n",
                    scheme, maxdim, nsteps * DT,
                    abs(total_sx(copy(psi)) - oat_total_sx(L, nsteps * DT)), disc,
                    time() - t0)
            flush(stdout)
        end
    end
    println("wrote ", OUT)
end

main()
