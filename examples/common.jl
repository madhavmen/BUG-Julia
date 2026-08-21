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

⛔ AN UNRECOGNISED NAME IS AN ERROR, not a silently ignored argument. A typo like `maxdim=16`
written as `max_dim=16` would otherwise run the DEFAULT and produce a figure that looks fine and
answers a different question. Types come from the defaults, so `L=12` parses as an `Int` and
`dt=0.05` as a `Float64` without either being declared twice.
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

⛔ A FIXED DEPTH SILENTLY BECOMES THE LIMITER AS `L` OR `dt` GROWS, and it does not error when
it does -- it caps every scheme at the SAME floor, which reads as "the integrators are
equivalent" when it actually means "none of them was measured". Each example passes its own
`‖H‖` estimate because the models scale differently: OAT's `‖H‖ ≈ L²/8` degrades fast with `L`
while the XX chain's is linear, so one shared constant cannot serve both.

`dt/2` is the half-sweep's step, which is what a single `expv` actually carries.
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

`cbe_bug` runs here in its SIMPLEST configuration: the exact SVD rather than the randomised
sketch, and `kaug = false`. See the two notes on the call itself -- one of those choices is free
and the other is not.
"""
function steppers(mpo; maxdim::Int, trunc_thresh::Float64, maxiter::Int)
    return (
        "tdvp2"      => (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = maxdim,
                                                trunc_thresh = trunc_thresh, maxiter = maxiter),
        "tdvp_cbe1s" => (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = maxdim,
                                                     trunc_thresh = trunc_thresh,
                                                     maxiter = maxiter),
        # ⛔ `exact = true` MEANS NO RANDOMISED SVD, AND IT COSTS NOTHING HERE. Measured at L=16
        # on both OAT and XX (benchmarks/rsvd_param_scan.jl phase 0), the sketch and the exact
        # SVD give `cbe_bug` results that are BIT-IDENTICAL -- relative difference 0.000e+00, not
        # "agrees to N digits". The sketch is SATURATED: it already spans the whole relevant
        # subspace, so a different draw is just a different basis of the same space. That is also
        # why `dover`, `comp_ratio` and the RNG seed are inert for this scheme, and why the
        # oversampling knob was dropped from these scripts. On OAT the exact SVD was even
        # slightly FASTER (68.9s vs 72.8s) -- at small rank the sketch's fixed overhead does not
        # amortise. Randomisation is a cost optimisation, never an accuracy one.
        #
        # ⛔ `kaug = false` IS THE HISTORIC BEHAVIOUR AND IT IS MEASURABLY WORSE -- this one is a
        # real trade, unlike `exact`. With the union off, each half-sweep's basis update REPLACES
        # the CBE frame instead of unioning with it, so the directions CBE just paid to find are
        # discarded before they can reach the state. MEASURED on OAT (L=10, T=2):
        #
        #     cbe_bug kaug=true    2.2901e-06
        #     cbe_bug kaug=false   1.8271e-02      <- 8000x worse, and 4.2x worse than tdvp2
        #
        # It is set here because the simpler scheme -- expand, then a plain 1-site update -- is
        # what these examples are meant to show. Pass `kaug = true` for the accurate variant;
        # docs/USAGE.md 6b has the full comparison.
        "cbe_bug"    => (p, tau) -> cbe_bug_step!(p, mpo, tau; kstep = true, kaug = false,
                                                  exact = true, maxdim = maxdim,
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
    # ⛔ THE STATE ENDS AT `nsteps*dt`, WHICH IS NOT `t_max` UNLESS dt DIVIDES IT. Comparing the
    # final state against `exact(t_max)` instead measures the time MISMATCH, not the integrator:
    # at t=0.628 with dt=0.05, `round(12.57)=13` steps land on 0.65, and |dS^x/dt|~3.6 there turns
    # that 0.022 gap into a 7.9e-02 "error" -- which is what every scheme then appears to have,
    # equally, because it is not their error at all. The CSV rows were always right (they emit
    # `exact(k*dt)` at the same `k*dt`); this is the summary line.
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
        @printf("  %-11s maxdim=%-3d  final err = %.3e   chi = %d   krylov = %d\n",
                name, p.maxdim, abs(observable(copy(psi)) - exact(t_final)),
                maximum(bond_dims(psi)), kry)
        flush(stdout)
    end
end

"""
    open_csv(path) -> IO

Open `path` for writing and emit the header row.

⛔ THE `flush(stdout)` IS LOAD-BEARING, and not for this file's sake. Julia BUFFERS stdout when
it is not a terminal, so every example's header -- the parameters, the Krylov depth, the rank
ceiling -- sits unseen in the buffer while the run JIT-compiles, which on a cold cache is
minutes. Piping to `tail` or a log file makes it worse: the reader buffers too, and the run
looks HUNG when it is merely quiet. Flushing here empties whatever the caller already wrote
(flush is about the buffer, not about who filled it), so one line covers all three scripts and
the header appears before the first long wait rather than after it.
"""
function open_csv(path::String)
    flush(stdout)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, join(COLS, ","))
    return io
end
