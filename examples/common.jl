# Shared plumbing for the example scripts: argument parsing, the run loop, and CSV output.
#
# Kept in one place so each example file is only its PHYSICS -- the Hamiltonian, the initial
# state, the observable and its exact value -- and a reader can see what the model is without
# reading a driver.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const EXDIR = dirname(abspath(@__FILE__))
const RESULTS = joinpath(EXDIR, "results")

"""
    parse_params(defaults) -> NamedTuple

Read `name=value` command-line arguments over a NamedTuple of defaults.

"""
function parse_params(defaults::NamedTuple)
    vals = Dict{Symbol, Any}(pairs(defaults))
    for a in ARGS
        occursin('=', a) || error("argument $(repr(a)) is not name=value")
        name, val = split(a, '='; limit = 2)
        key = Symbol(strip(name))
        haskey(vals, key) || error("unknown parameter $(repr(String(key))). Valid: " *
                                   join(sort(String.(collect(keys(defaults)))), ", "))
        d = defaults[key]
        vals[key] = d isa Int     ? parse(Int, val)     :
                    d isa Float64 ? parse(Float64, val) :
                    d isa Bool    ? parse(Bool, val)    : String(val)
    end
    return NamedTuple(vals)
end

"""
    krylov_depth(hnorm, dt; thresh = 1e-14, lo = 12, hi = 40) -> Int

The Lanczos depth at which `(‖H‖·dt/2)^m / m!` drops below `thresh`.

"""
function krylov_depth(hnorm::Real, dt::Real; thresh = 1e-14, lo = 12, hi = 40)
    arg, m = hnorm * dt / 2, lo
    while m < hi && arg^m / factorial(big(m)) > thresh
        m += 1
    end
    return m
end

"Report the bound `krylov_depth` achieved, so a run that is Krylov-limited says so in its header."
krylov_bound(hnorm::Real, dt::Real, m::Int) =
    Float64((hnorm * dt / 2)^m / factorial(big(m)))

"""
    steppers(mpo; maxdim, trunc_thresh, maxiter)

The three integrators, as closures over one MPO. Any difference between them is the integrator.

⛔ `cbe_bug` NOW RUNS THE MEASURED-BEST CONFIGURATION: the RANDOMISED SKETCH (`exact = false`) at
a PINNED Krylov depth of `m = 3` (`krylov_basis = 3, krylov_tol = 0.0`), with `dex = 8, dover = 4,
comp_ratio = 1.0`. Every part of that is measured, not assumed:

  * `m = 3` is the accuracy/cost optimum. From `docs/PLAN-tolerance-control.md` at L=12, dt=0.05
    (error / operator applications):

        arm            XX                 Heisenberg         OAT
        cap, no test   1.6948e-04 / 1770  1.0800e-06 / 2280  5.9952e-15 / 2158
        tol 1e-6 (old default)
                       1.6948e-04 / 1180  1.4748e-06 / 1622  1.0880e-14 / 1766
        pinned m = 3   1.6948e-04 /  950  2.6697e-06 / 1038  7.0610e-14 / 1036
        pinned m = 0   1.6948e-04 /  289  8.8424e-05 /  308  2.1329e-02 /  313

    `m = 3` holds XX and OAT at the deep-basis answer for ~0.6x the operator work, and gives up
    1.8x on Heisenberg for 0.64x the cost. `m = 0` is eight orders worse on OAT -- ⛔ DEPTH IS THE
    ONE KNOB THAT MUST NOT BE CUT TO ZERO.
  * The SKETCH is worth taking because the expansion it shrinks is no longer paid three times over
    (see `_sketch_closures`). It buys `t_cbe` 1.3-1.9x, the most where `t_cbe` is the largest share
    of the step (the `d = 4` fused open systems, ~72%).
  * ⛔ `dover = 4` IS NOT A ROUNDING OF THE DEFAULT -- IT IS WORTH SEVEN ORDERS, FOR FREE.
    `npre = dover === nothing ? ceil(1.2*dex) : dex + dover` (`cbe_core.jl:587`), so this is the
    PROBE WIDTH, and at `dex = 8` the probe is what binds rather than the admitted budget.
    MEASURED (Heisenberg, 20 steps of dt=0.05, deviation from the EXACT-CBE arm):

        dover     Dpre   L=10 vs exact CBE   L=12 vs exact CBE   chi (exact = 26)   secs (L=10)
        default     10        1.0234e-07          1.0234e-07             28             11.16
        0            8        1.0234e-07          1.0234e-07             28             11.01
        2           10        1.0234e-07          1.0234e-07             28             11.27
        4           12        6.0507e-15          5.1043e-14             26             10.96

    Three columns say the same thing. An under-sampled probe is not merely less accurate: it also
    ADMITS SPURIOUS DIRECTIONS (chi 28 against the exact 26), so it costs rank as well. And the
    seconds are flat across the whole column -- two extra probe columns are free.
    ⚠ THIS SUPERSEDES `dover = 0`, which an earlier sweep (`rsvd_optimal.jl`, Schwinger d=4,
    chi=32) picked as "measured-best". That sweep optimised SPEED on one model and never compared
    the arms against the exact-CBE reference, so it could not see the seven orders it was paying.

⚠ `parallel` IS LEFT OFF HERE. It is worth 1.2-1.7x but needs `julia -t 2` or more, and an example
script that silently reads as a pessimisation on a single-threaded run is worse than one that is
honest about its defaults. Pass `parallel = true` for production runs with threads.

⛔ DO NOT VALIDATE A BASIS-CONSTRUCTION CHANGE ON XX. It is a control, never the evidence. The XX
domain wall stays low-rank enough that basis choices do not bind, and it has now failed to show
FOUR separate real effects while TFIM or OAT showed each of them plainly:

    effect                              XX               TFIM / OAT
    the old projection defect           invisible        2.5e-01  (TFIM)
    the old midpoint basis (retired)    bit-identical    cost an ORDER (TFIM)
    width restoration (retired)         free             five orders (TFIM)
    KRYLOV DEPTH  m=0 -> m=3            bit-identical    eight orders (OAT)

⛔ AND DO NOT RANK ARMS BY ENERGY DRIFT OR BY THE BOND PROFILE. On OAT at L=10, D=6 (the exact
ceiling), a `krylov_basis = 0` run conserves energy to 1.5e-14, lands EXACTLY on the analytic
Schmidt profile `min(k, L-k) + 1` = [2,3,4,5,6,5,4,3,2], exits 0 and writes its CSV -- while
sitting eight orders from the answer. Both diagnostics say it is perfect. Only a comparison
against an exact reference finds it.

"""
function steppers(mpo; maxdim::Int, trunc_thresh::Float64, maxiter::Int)
    return (
        "tdvp2"      => (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = maxdim,
                                                trunc_thresh = trunc_thresh, maxiter = maxiter),
        "tdvp_cbe1s" => (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = maxdim,
                                                     trunc_thresh = trunc_thresh,
                                                     maxiter = maxiter),

        # THE MEASURED-BEST RUNNER -- see the docstring above for every number behind it.
        "cbe_bug"    => (p, tau) -> cbe_bug_step!(p, mpo, tau;
                                                  exact = false,        # randomised sketch
                                                  dex = 8, dover = 4,  # Dpre = 12; see above
                                                  comp_ratio = 1.0,
                                                  krylov_basis = 3,     # m = 3
                                                  krylov_tol = 0.0,     # pinned, not a tolerance
                                                  maxdim = maxdim,
                                                  trunc_thresh = trunc_thresh, maxiter = maxiter),
    )
end

const COLS = ["model", "scheme", "symmetry", "L", "maxdim", "dt", "t",
              "obs", "obs_exact", "err", "maxbond", "energy", "dE", "krylov"]

"""
    run_model(io, model, symmetry, psi0, mpo, observable, exact, p)

Evolve `psi0` under `mpo` with each of the three integrators, streaming one CSV row per sample.

`observable(psi) -> Float64` is measured on a COPY (the measurement canonicalises, which changes
the gauge but not the state). `exact(t) -> Float64` is the closed form it is compared against.

Rows are written as they are produced rather than collected, so a run that is interrupted still
leaves usable data and a long run can be plotted while it is still going.
"""
function run_model(io::IO, model::String, symmetry::String, psi0, mpo,
                   observable, exact, p)
    nsteps = round(Int, p.t_max / p.dt)
    every  = max(1, round(Int, p.sample_every / p.dt))

    t_final = nsteps * p.dt
    for (name, step!) in steppers(mpo; maxdim = p.maxdim, trunc_thresh = p.trunc_thresh,
                                  maxiter = p.maxiter)
        psi = copy(psi0)
        e0  = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
        kry = 0
        emit(t, o, k) = begin
            e = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
            @printf(io, "%s,%s,%s,%d,%d,%g,%g,%.10g,%.10g,%.6e,%d,%.10g,%.6e,%d\n",
                    model, name, symmetry, length(psi), p.maxdim, p.dt, t,
                    o, exact(t), abs(o - exact(t)), maximum(bond_dims(psi)), e, abs(e - e0), k)
            flush(io)
        end
        emit(0.0, observable(copy(psi)), 0)
        for k in 1:nsteps
            info = step!(psi, ComplexF64(-im * p.dt))
            kry += info.krylov_dims
            if k % every == 0
                emit(k * p.dt, observable(copy(psi)), kry)
            end
        end
        # ⛔ PRINT THE WHOLE BOND PROFILE, NOT JUST ITS MAXIMUM. One number cannot distinguish
        # "the rank GREW to the ceiling" from "the rank is FROZEN", and that ambiguity was read as
        # a bug in `cbe_bug` on OAT: `chi = 6` never moves there, which looks like a freeze and is
        # in fact CORRECT -- OAT's exact Schmidt rank `min(n, L-n) + 1` is CONSTANT IN TIME, and
        # `maxdim` on that example defaults to exactly that ceiling. The profile settles it at a
        # glance: `[2,3,4,5,6,5,4,3,2]` is the analytic answer, `[1,1,2,4,2,1,1]` is the freeze
        # this repo actually had once (`cbe_core.jl:638`).
        #
        # ⚠ AND THE PROFILE IS STILL NOT AN ACCURACY DIAGNOSTIC -- see the `steppers` docstring: a
        # `krylov_basis = 0` run on OAT lands EXACTLY on the analytic profile and conserves energy
        # to 1.5e-14 while sitting eight orders from the answer. It answers "did the rank grow",
        # nothing more; `final err` is the only column that ranks the arms.
        @printf("  %-11s maxdim=%-3d  final err = %.3e   chi = %d   krylov = %d\n",
                name, p.maxdim, abs(observable(copy(psi)) - exact(t_final)),
                maximum(bond_dims(psi)), kry)
        @printf("  %-11s bond profile = %s\n", "", string(bond_dims(psi)))
        flush(stdout)
    end
end

"""
    open_csv(path) -> IO

Open `path` for writing and emit the header row.


"""
function open_csv(path::String)
    flush(stdout)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, join(COLS, ","))
    return io
end
