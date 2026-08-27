# THE L = 18 TRAJECTORY CAMPAIGN: 2-site TDVP vs 1-site CBE-TDVP vs CBE-BUG, on all six models.
#
#   julia -t 2 --project=. benchmarks/l18_campaign.jl
#   MODELS=xx SITES=8 TMAX=0.5 SMOKE=1 julia -t 2 --project=. benchmarks/l18_campaign.jl
#
# ⛔ THIS IS A TRAJECTORY CAMPAIGN, NOT `rsvd_parallel.jl`. That driver answers "how long does ONE
# step take, from the SAME state" and deliberately pins the root solve so the arms do identical
# operator work. This one answers a different question -- does the PHYSICS come out right, and what
# does a whole trajectory cost -- so nothing is pinned and each arm runs its own shipped
# configuration. The two files' seconds are therefore NOT interchangeable and must not be put in
# one table.
#
# ── THE THREE ARMS ───────────────────────────────────────────────────────────────────────────
#
#   tdvp2        2-site TDVP. The accuracy reference: it is the method these results have to
#                match, and the one with no basis-expansion machinery to get wrong.
#   tdvp_cbe1s   1-site TDVP with controlled bond expansion. ⛔ THIS IS THE REAL BASELINE, and
#                the whole reason this file exists. Every earlier CBE-BUG margin in
#                `docs/current_work.md` was quoted against `tdvp2`, which is the WEAKER of the
#                two: measured at L=16, `tdvp_cbe1s` matched `tdvp2`'s accuracy to every printed
#                digit at 1.22x FEWER operator applications. Beating `tdvp2` is therefore not
#                evidence of beating the state of the art.
#   cbe_bug      the shipped configuration from `examples/common.jl` -- randomised sketch,
#                `dex = 8, dover = 4`, `m = 3` -- plus `parallel = true`, which is what a
#                threaded production run uses.
#
# ── COST HAS TWO AXES AND THEY DISAGREE. BOTH ARE RECORDED. ──────────────────────────────────
#
# ⛔ `krylov_dims` IS NOT A FAIR CROSS-SCHEME COST, and each scheme breaks the comparison in a
# different direction:
#
#   * `tdvp2` counts TWO-SITE matvecs, `O(d^2 chi^2 w)`; `tdvp_cbe1s` counts ONE-SITE and
#     ZERO-SITE ones, `O(d chi^2 w)` and `O(chi^2 w)`. Its own docstring says so. A raw ratio
#     between them therefore flatters the 1-site scheme by roughly `d`.
#   * `cbe_bug`'s count is an UNDERCOUNT: `krylov_dims` sees `apply_one_site`, while the
#     expansion's `H*Theta` work happens inside `sketch_h_*` and is invisible to it.
#
# So WALL CLOCK is the honest axis and `krylov` is the structural one, and a claim that rests on
# only one of them is not a claim. MEASURED on an 8x8 cylinder (docs/PLAN-open-systems.md §5):
# `cbe_bug` used 19x FEWER applications and took 1.91x LONGER. They have agreed at 18 sites so
# far; that is not a reason to quote one.
#
# ── THE PHYSICS VALIDATION, PER MODEL ────────────────────────────────────────────────────────
#
# ⛔ AN INTEGRATOR COMPARISON IS NOT A VALIDATION. Three arms agreeing with each other says they
# share a convention, not that the convention is right -- and `arnoldi_check.jl` already caught
# five code paths agreeing to four digits on a number one of them had produced. Each model
# therefore carries a target that is INDEPENDENT of every arm:
#
#   xx           EXACT: free-fermion `<Sz_j(t)>` by Jordan-Wigner (`tests/common/free_fermion.jl`),
#                an L x L matrix exponential, derived from H on paper rather than from our gates.
#                This is the only model here where `err` is an ERROR; everywhere else it is a
#                DEVIATION from `tdvp2`.
#   square       no closed form at L=18 -> deviation from `tdvp2`, plus energy conservation.
#   kagome       ⚠ NO published REAL-TIME trend exists. arXiv:2603.21147's signature (entropy
#                feature near h = 0.22) is an iDMRG GROUND-STATE statement at D = 10^4 on an
#                infinite cylinder. Reported as a cost/consistency model only; do NOT write it up
#                as validated dynamics.
#   xx_open      Znidaric (NJP 12 043001): dephasing turns BALLISTIC transport DIFFUSIVE.
#                ⚠ needs a DOMAIN WALL, not a Neel start -- `INIT=domain_wall`.
#   square_open  arXiv:2408.10304 ZENO SLOWING + arXiv:2410.18468 operator entanglement.
#                See `l18_square_open_protocol.jl`: the trend needs a gamma SWEEP, so it is not
#                decidable from this file's single gamma.
#   lgt          arXiv:2501.13675: thermalisation time rises with dissipation.
#
# ⚠ `err` FOR THE OPEN MODELS IS A DEVIATION FROM `tdvp2` ON A GENERATOR WHERE `tdvp2` ITSELF WAS
# WRONG UNTIL 2026-08-27 (it accepted `hermitian` and then ignored it, running Lanczos on a
# non-Hermitian Lindbladian). Fixed in `8160827`; recorded because any open-system number from
# before that commit is void.

using LinearAlgebra, Printf, Random
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const OUT    = joinpath(@__DIR__, "results")
const MODELS = split(get(ENV, "MODELS", "xx,square,kagome,xx_open,square_open,lgt"), ",")
const SITES  = parse(Int,     get(ENV, "SITES",  "18"))
const DT     = parse(Float64, get(ENV, "DT",     "0.05"))
const TMAX   = parse(Float64, get(ENV, "TMAX",   "5.0"))
const SNAP   = parse(Float64, get(ENV, "SNAP",   "0.25"))
const CAP    = parse(Int,     get(ENV, "CAP",    "64"))
const GAMMA  = parse(Float64, get(ENV, "GAMMA",  "0.10"))
const INIT   = get(ENV, "INIT", "neel")
const SMOKE  = get(ENV, "SMOKE", "0") == "1"

# ⛔ ONE KRYLOV BUDGET FOR THE TWO TDVP ARMS, AND IT IS NOT THE LIBRARY DEFAULT. `lanczos_expv`
# has a BREAKDOWN-ONLY exit, so at the default `maxiter = 30` every solve spends its whole budget
# (measured mean dimension 27.4 out of 30). `benchmarks/l18_matrix.jl` measured the ladder at this
# exact L and dt: `maxiter = 8` is 3.64x faster than 30 at `max|dSz| = 2.1e-13`, and the cliff is
# at 4 (3.2e-07). Handing one arm a deeper exponential than another would make the seconds
# meaningless, so both TDVP arms get 8.
#
# ⚠ `cbe_bug` IS **NOT** GIVEN THE SAME NUMBER, AND THAT IS THE POINT OF THE ARM. Its `m = 3`
# pinned half-sweep basis is part of the method (see `examples/common.jl`), not a budget knob, and
# its root solve is the only place `maxiter` applies. This is a comparison of SHIPPED
# CONFIGURATIONS, not of one algorithm at matched depth.
const MAXITER = parse(Int, get(ENV, "MAXITER", "8"))

_stag(v) = sum((isodd(j) ? 1.0 : -1.0) * v[j] for j in eachindex(v)) / length(v)

"The three arms as closures over one model, each in the configuration it actually ships in."
function arms(S, cap::Int)
    herm = (S.d == 2)                      # the fused Lindbladian is NOT Hermitian
    return [
        ("tdvp2", (p, tau) -> tdvp2_step!(p, S.W, tau; maxdim = cap,
                                          trunc_thresh = 1e-10, maxiter = MAXITER,
                                          hermitian = herm)),
        ("tdvp_cbe1s", (p, tau) -> tdvp_cbe1s_step!(p, S.W, tau; maxdim = cap,
                                                    trunc_thresh = 1e-10, maxiter = MAXITER,
                                                    hermitian = herm)),
        ("cbe_bug", (p, tau) -> cbe_bug_step!(p, S.W, tau;
                                              exact = false, dex = 8, dover = 4,
                                              comp_ratio = 1.0,
                                              krylov_basis = 3, krylov_tol = 0.0,
                                              parallel = true, share_ht = true,
                                              maxdim = cap, trunc_thresh = 1e-10,
                                              maxiter = MAXITER, hermitian = herm)),
    ]
end

"""
The EXACT `<Sz_j(t)>` for the XX chain, or `nothing` when the model has no closed form.

⛔ THE OCCUPATION SET MUST MATCH THE MPS START OR THIS SILENTLY BECOMES A DIFFERENT PROBLEM.
`neel_state` is up on ODD sites and up is an occupied fermion (`Sz = n - 1/2`), so `occupied` is
`1:2:L`. `domain_wall_state` is the free-fermion default `1:(L/2)`. The `t = 0` row is asserted
against the MPS below rather than trusted -- `dense_state` and `exact_sparse` in this repo use
MIRRORED bit conventions, and Neel is exactly the kind of state symmetric enough to hide it.
"""
function exact_sz(nm, L, t, init; gamma::Float64 = 0.0)
    occ = init == "domain_wall" ? (1:(L ÷ 2)) : (1:2:L)
    nm == "xx" && return xx_free_fermion_sz(L, t; J = 1.0, occupied = occ)
    # ⛔ `xx_open` HAS A CLOSED FORM TOO, AND IT IS THE ONLY OPEN MODEL THAT DOES. Onsite dephasing
    # closes the single-particle correlation matrix (`dC/dt = -i[h,C] - gamma(1-delta_ij)C_ij`), so
    # `<S^z_j(t)>` is an L x L ODE at ANY L and ANY gamma -- see `xx_dephasing_sz`. That makes this
    # the one open model where "TDVP degraded" can be told apart from "BUG drifted": every other
    # open model is scored against a published TREND, and a trend cannot distinguish an arm that
    # survived correctly from one that merely survived.
    nm == "xx_open" && return xx_dephasing_sz(L, t; J = 1.0, gamma = gamma, occupied = occ)
    return nothing
end

function main()
    mkpath(OUT)
    tag = replace(join(MODELS, "-"), "/" => "_") * (SMOKE ? "_smoke" : "")
    csv = open(joinpath(OUT, "l18_campaign_$tag.csv"), "w")
    prof = open(joinpath(OUT, "l18_profile_$tag.csv"), "w")
    println(csv, "model,arm,L,d,cap,gamma,dt,t,step,mstag,energy,err,maxbond,sop,nstates," *
                 "krylov,secs")
    println(prof, "model,arm,t,site,value")
    flush(csv); flush(prof)

    nsteps = max(1, round(Int, TMAX / DT))
    every  = max(1, round(Int, SNAP / DT))

    @printf("L=%d  dt=%.3f  tmax=%.2f (%d steps)  cap=%d  gamma=%.3f  init=%s  maxiter=%d\n",
            SITES, DT, TMAX, nsteps, CAP, GAMMA, INIT, MAXITER)
    @printf("julia threads = %d, BLAS threads = %d\n\n",
            Threads.nthreads(), BLAS.get_num_threads())
    flush(stdout)

    for nm in MODELS
        S = setup(nm; sites = SITES, dt = DT, gamma = GAMMA, cap = CAP)
        L = S.L
        psi0 = S.psi0
        if INIT == "domain_wall"
            # ⚠ ONLY THE OPEN MODELS CAN BE RE-STARTED HERE. The closed ones come out of
            # `_closed` with a Neel `SymMPS` under a GLOBAL `:U1` symmetry setting, and swapping
            # the start would silently change the Sz sector the whole run lives in.
            S.d == 4 || error("INIT=domain_wall is only wired for the fused open models")
            psi0 = rho0_product([k <= L ÷ 2 ? :up : :down for k in 1:L])
        end

        @printf("MODEL %-12s %s\n", nm, S.label)
        @printf("   %d sites, d = %d, MPO virtual dim %d, cap %d\n",
                L, S.d, maximum(mpo_virtual_dims(S.W)), CAP)
        flush(stdout)

        # ── the t = 0 convention gate, for the models that have a closed form ────────────────
        #
        # ⛔ READ THE PROFILE THROUGH `S.profile`, NEVER `sz_expectation`. The fused open models
        # have d = 4 SITES (a `(ket, bra)` pair per system site), so a d = 2 `S^z` cannot contract
        # against them: `sz_expectation` threw `Contracted legs must have matching space info:
        # q1 leg 1 spaces != q2 leg 2 spaces` the moment `xx_open` gained a closed form and
        # started running this gate. `S.profile` is `<S^z_j>` for the closed models and
        # `Tr(S^z_j rho)` for the fused ones -- the SAME physical quantity, obtained the way each
        # layout requires, which is exactly why the models expose it rather than a raw operator.
        ex0 = exact_sz(nm, L, 0.0, INIT; gamma = GAMMA)
        if ex0 !== nothing
            got = S.profile(copy(psi0))[1:L]
            dev = maximum(abs.(got .- ex0))
            @printf("   exact-reference convention gate at t=0: max|dSz| = %.3e  %s\n",
                    dev, dev < 1e-10 ? "PASS" : "FAIL -- occupation set does not match the MPS")
            dev < 1e-10 || error("$nm: the exact reference describes a different initial state")
        end

        # ⛔ WARMUP, AND IT IS NOT OPTIONAL. Without it the FIRST arm pays this model's entire JIT
        # bill and the seconds column is fiction. MEASURED on the L=8 smoke run before this was
        # added: `tdvp2` 61.6 s, `tdvp_cbe1s` 15.1 s, `cbe_bug` 2.7 s -- and `tdvp2` only led
        # because it runs first. The package's `@compile_workload` covers `cbe_bug_step!` only, so
        # the two TDVP arms specialise here, on this model's element types, the first time they
        # are called. One step each, on a throwaway copy, at the real cap.
        for (arm, step!) in arms(S, CAP)
            try
                step!(copy(psi0), S.tau)
            catch e
                @printf("   warmup: %s threw -- %s\n", arm,
                        sprint(showerror, e)[1:min(end, 160)])
            end
        end

        ref = Dict{Int, Vector{Float64}}()   # step -> tdvp2 profile, the fallback reference
        for (arm, step!) in arms(S, CAP)
            psi = copy(psi0)
            kry = 0
            wall = 0.0
            e0 = NaN
            emit = function (k)
                t = k * DT
                p = S.profile(psi)
                sz = p[1:L]
                e  = length(p) > L ? p[L + 1] : NaN
                isnan(e0) && (e0 = e)
                sop, nst = operator_entanglement(psi; base = 2)   # arXiv:2410.18468 Eq. (3)
                exv = exact_sz(nm, L, t, INIT; gamma = GAMMA)
                # ⛔ THE REFERENCE FALLS BACK TO `tdvp2` AND THE COLUMN NAME DOES NOT CHANGE, so
                # the plotting script and the write-up must read `err` through the model: it is an
                # ERROR for `xx` and a DEVIATION everywhere else. `tdvp2` against itself is 0 by
                # construction and is NOT evidence of anything.
                err = if exv !== nothing
                    maximum(abs.(sz .- exv))
                elseif arm == "tdvp2"
                    (ref[k] = copy(sz); 0.0)
                else
                    haskey(ref, k) ? maximum(abs.(sz .- ref[k])) : NaN
                end
                @printf(csv, "%s,%s,%d,%d,%d,%g,%g,%g,%d,%.10g,%.10g,%.6e,%d,%.8g,%d,%d,%.4f\n",
                        nm, arm, L, S.d, CAP, GAMMA, DT, t, k, _stag(sz), e, err,
                        maximum(bond_dims(psi)), sop, nst, kry, wall)
                for j in 1:L
                    @printf(prof, "%s,%s,%g,%d,%.10g\n", nm, arm, t, j, sz[j])
                end
                flush(csv); flush(prof)
                return nothing
            end

            emit(0)
            for k in 1:nsteps
                t0 = time_ns()
                info = step!(psi, S.tau)
                wall += (time_ns() - t0) / 1e9        # the STEP only
                kry += info.krylov_dims
                S.post!(psi)                          # trace/shift: common to all arms, untimed
                # ⚠ THE LAST STEP ALWAYS EMITS, even when `nsteps` is not a multiple of `every`.
                # Without that the final summary line reads `ref[nsteps]`, which the `tdvp2` arm
                # never wrote -- so every later arm's last-row deviation came out `NaN` for any
                # `TMAX/DT` that does not divide by `SNAP/DT`.
                (k % every == 0 || k == nsteps) && emit(k)
            end
            @printf("   %-12s  chi %3d  krylov %7d  %8.1f s   err/dev %.3e\n",
                    arm, maximum(bond_dims(psi)), kry, wall,
                    let exv = exact_sz(nm, L, nsteps * DT, INIT; gamma = GAMMA)
                        szf = S.profile(psi)[1:L]
                        exv !== nothing ? maximum(abs.(szf .- exv)) :
                            (haskey(ref, nsteps) ? maximum(abs.(szf .- ref[nsteps])) : NaN)
                    end)
            flush(stdout)
        end
        println()
    end
    close(csv); close(prof)
    println("wrote results/l18_campaign_$tag.csv and results/l18_profile_$tag.csv")
end

main()
