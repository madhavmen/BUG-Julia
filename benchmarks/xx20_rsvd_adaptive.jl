# XX (Delta = 0), U(1), L = 20 -- the acceptance run for the ADAPTIVE `dex`/`dover` in
# `cbe_core.jl`. The domain wall's magnetisation profile is known in closed form
# (`xx_free_fermion_sz`), so "did the adaptive budget cost accuracy?" is answered against an
# exact reference rather than against another approximation.
#
# ⛔ MPO PATH ONLY. `cbe_bug_step!` also accepts an `XXZChain` term list, which routes through
# `mpo_from_terms` and is a DIFFERENT sketch (`xx_l18_campaign.jl` uses it). The Hamiltonian
# under study here is the MPO construction, so the MPO is built once with `xxz_mpo` and handed
# in directly -- never a term list.
#
# TWO ARMS, and they answer two different questions:
#
#   `default` -- `dex = 0, dover = nothing`. BEFORE the patch this is the `growth = 2.0`
#                schedule (`budget = ceil(2*rmax) - r`, `npre = ceil(1.2*budget)`); AFTER it is
#                the adaptive rule. Run in both trees, this is the measurement.
#   `fixed`   -- `dex = 8, dover = 4`. Both trees must take the SAME code path here (an explicit
#                `dex` is still an absolute cap, an explicit `dover` still pins one probe pass),
#                so this arm is the BACKWARDS-COMPATIBILITY CONTROL: baseline and patched must
#                agree to the last digit. It is run at `parallel = false` because the two
#                half-sweeps share one RNG, and under `parallel = true` their draw ORDER is not
#                reproducible -- a control that cannot be exactly equal cannot catch anything.
#
# Usage: julia --project=. benchmarks/xx20_rsvd_adaptive.jl <tag> <tmax> <outdir>

using LinearAlgebra, Printf, Random
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include("../tests/common/free_fermion.jl")

const L       = 20
const DT      = 0.05
const MAXDIM  = 128
const TRUNC   = 1e-12
const SAMPLE  = 4

exact_profile(t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:(L ÷ 2))

"""
L-infinity error of the MPS magnetisation profile against the exact free-fermion one.

⚠ DIVIDED BY `norm(psi)^2`. The expansion changes how much norm the step loses, so an
un-normalised profile would charge norm loss to the magnetisation and would do so UNEQUALLY
between the arms -- i.e. exactly the difference the run is trying to measure.
"""
function profile_err(psi, t)
    n2 = norm(psi)^2
    return maximum(abs.(magnetisation(psi) ./ n2 .- exact_profile(t)))
end

const HEADER = "arm,parallel,t,err,maxbond,centrebond,maxexpanded,err_pre,err_fnl," *
               "krylov_dims,peak_elements,norm,elapsed"

"""
One arm: `nsteps` of `cbe_bug_step!` on the XX MPO, sampling the profile error every
`SAMPLE` steps. Returns the CSV lines.

`kw` is splatted into the step, so the arm definition is the ONLY place the two differ.
"""
function run_arm(arm::String, tmax::Float64, parallel::Bool; kw...)
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    nsteps = round(Int, tmax / DT)
    c = L ÷ 2
    lines = String[]
    maxex = 0
    epre = 0.0
    efnl = 0.0
    kdims = 0
    peak = 0
    t0 = time()
    for k in 1:nsteps
        info = cbe_bug_step!(psi, mpo, -im * DT; parallel = parallel, maxdim = MAXDIM,
                             trunc_thresh = TRUNC, rng = MersenneTwister(0x5EED + k), kw...)
        maxex = max(maxex, maximum(info.expanded; init = 0))
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
        kdims += info.krylov_dims
        peak = max(peak, info.peak_elements)
        if k % SAMPLE == 0 || k == nsteps
            t = k * DT
            bd = bond_dims(psi)
            push!(lines, @sprintf("%s,%s,%.4f,%.6e,%d,%d,%d,%.6e,%.6e,%d,%d,%.12f,%.3f",
                                  arm, parallel, t, profile_err(psi, t), maximum(bd), bd[c],
                                  maxex, epre, efnl, kdims, peak, norm(psi), time() - t0))
        end
    end
    return lines
end

function main()
    tag    = length(ARGS) >= 1 ? ARGS[1] : "run"
    tmax   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
    outdir = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    path = joinpath(outdir, "xx20_$(tag).csv")

    lines = String[]
    append!(lines, run_arm("default", tmax, true;  dex = 0, dover = nothing))
    append!(lines, run_arm("fixed",   tmax, false; dex = 8, dover = 4))

    open(path, "w") do io
        println(io, HEADER)
        for l in lines
            println(io, l)
        end
    end
    println(HEADER)
    for l in lines
        println(l)
    end
    println("WROTE $path")
end

main()
