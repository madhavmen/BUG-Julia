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

`cbe_bug` runs with the exact SVD rather than the randomised sketch. That simplification IS free:
the sketch is measured bit-identical to the exact SVD on both example models
(`benchmarks/rsvd_param_scan.jl` phase 0, relative difference 0.000e+00).

`rexpand = true` is the scheme's default and is passed explicitly so the choice is visible here
rather than inherited: every bond is expanded TWICE per half-sweep -- once as the right bond of
site `i`, once as the left bond of site `i+1` -- so every site is evolved with both of its bonds
widened, and no basis is ever formed by merging two others. It costs a second `cbe_expand` per
bond and no extra `expv`.

⛔ THE WIDTH RESTORATION IS NOT OPTIONAL ON TFIM, AND IS IRRELEVANT ON XX. Without it (`kaug =
false, rexpand = false`) the half-sweep split leaves the next site's LEFT bond narrow. Measured at
L=16, maxdim=64, trunc=1e-10 (`benchmarks/example_defaults.jl`) -- ⚠️ the numbers below were taken
with the RANDOMISED sketch (`exact = false`, the package default) rather than the `exact = true`
these scripts pass; the arms have since been corrected to pass `exact = true` and the table is due
a re-measure. Expect it to move very little if the sketch really is free, which is the other
claim in this docstring that wants re-pinning:

    mechanism                  TFIM :Z2     TFIM :none    XX :U1
    rexpand = true (default)   3.8455e-06   4.3036e-06    8.1911e-05
    kaug    = true             4.2717e-06   4.0839e-06    8.1911e-05
    neither                    1.9699e-01   1.9474e-01    8.0282e-05

On TFIM the unrestored arm is not degraded but WRONG -- five orders, and the run still exits 0 and
still writes its CSV, so the failure is SILENT. On XX it costs NOTHING; `neither` is marginally
BEST there.

⛔ SO DO NOT VALIDATE A BASIS-MECHANISM CHANGE ON XX. That is not a quirk of this table, it is the
pattern this model shows every time: the `rexpand` projection defect was invisible on XX and
glaring on TFIM (2.5e-01); `basis_frac` is bit-identical across every value on XX and costs an
ORDER on TFIM; width restoration is free on XX and worth five orders on TFIM. The XX domain wall
stays low-rank enough that basis-construction choices do not bind, so it cannot discriminate
between them -- it is a control, never the evidence.

⛔ THE TWO MECHANISMS ARE AT PARITY, AND THE DIRECTION FLIPS WITH THE SYMMETRY MODE (`rexpand`
1.11x better under `:Z2`, `kaug` 1.05x better under `:none`). An earlier version of this table
had `kaug` ahead by 2-3x; that was measured at `growth = 1.1`, which STARVES `rexpand` -- its
second expansion runs against a larger `U0`, and `budget = ceil(growth*dmax) - r` SUBTRACTS that
`r`. At the current `growth = 2.0` the gap closes. Any future re-measurement of this table must
re-check `growth` first, because that is the knob it is sensitive to.

So `rexpand` is the default for its STRUCTURE, not for a win: no basis is ever formed by merging
two others, and every site is evolved with both of its bonds widened -- the ordering
`tdvp_cbe1s` gets for free from its forward and backward passes. The measurement says that
choice costs nothing in accuracy.

⛔ ON COST THEY ARE NOT ESTABLISHED AS EQUAL. `rexpand`'s second `cbe_expand` per bond is
QR/matmul work that adds no `expv`, so `krylov` -- the cost axis used everywhere else here -- is
BLIND to exactly the work `rexpand` adds. Wall time on the run above was 556s vs 275s under
`:Z2`, which is directionally consistent but is NOT evidence: local wall time spreads 2.6-2.8x on
bit-identical work. Treat the cost comparison as unmeasured.

ON OAT THE ERROR AND THE BOND PROFILE MEASURE DIFFERENT THINGS, AND NEITHER ALONE IS THE TELL.
At L=10, D=6 (the exact ceiling), t=10π -- the settings `oat_z2.jl` actually runs:

    mechanism                  infidelity   bond profile
    rexpand = true (default)   3.2748e-11   [2, 3, 5, 5, 6, 5, 4, 3, 2]
    kaug    = true             3.2750e-11   [2, 3, 6, 6, 6, 6, 6, 3, 2]
    neither                    3.9347e-05   [2, 3, 6, 6, 6, 6, 6, 3, 2]

The exact Schmidt profile is `min(k, L-k) + 1` = [2,3,4,5,6,5,4,3,2] for all time.

It is also the one place `rexpand` is clearly AHEAD rather than at parity: the same infidelity as
`kaug` on a leaner state (bond sum 35 against 40), which is some of its extra sketch cost bought
back. Both TDVPs over-fill this profile too, so tracking it at all is a property of the BUG
half-sweeps rather than of either restoration mechanism.
"""
function steppers(mpo; maxdim::Int, trunc_thresh::Float64, maxiter::Int)
    return (
        "tdvp2"      => (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = maxdim,
                                                trunc_thresh = trunc_thresh, maxiter = maxiter),
        "tdvp_cbe1s" => (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = maxdim,
                                                     trunc_thresh = trunc_thresh,
                                                     maxiter = maxiter),

        "cbe_bug"    => (p, tau) -> cbe_bug_step!(p, mpo, tau; kstep = true, rexpand = true,
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


"""
function open_csv(path::String)
    flush(stdout)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, join(COLS, ","))
    return io
end
