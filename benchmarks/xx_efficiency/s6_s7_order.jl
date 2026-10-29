# benchmarks/xx_efficiency/s6_s7_order.jl -- plan Sec. 5 S6 and S7, accuracy only (not timed).
#
# S6 local: one order-3 step from the high-accuracy references (snapshots.jl ref_*.jls, graded to
#    ~1e-11) against the exact Slater state at t + h; slope of the one-step error in h (target 4).
# S6 global: L = 16 from the uncompressed domain wall to J T = 2, error against the Slater state
#    (raw and phase-aligned) and eps_z, per dt; slope in dt (target 3). Tolerances are tight so the
#    time-step error is the measured one; each step's limit statuses are recorded, not assumed.
# S7: one tolerance changed at a time on the one-step problem (h = 0.1, ref t = 4): tol_basis
#    sweep at tight tol_trunc/tol_expv, then tol_trunc sweep at tight tol_basis.
# The genuine-recurrence half of S6 (order p gives local slope p + 1, not p repeats of order 1) is
# UT3 in test_krylov_closure.jl.
#
# Env (also common.jl): XXE_DTS (0.2,0.1,0.05,0.025), XXE_T (2.0), XXE_PARTS (local,global,s7)

include(joinpath(@__DIR__, "common.jl"))

const DTS = parse.(Float64, split(get(ENV, "XXE_DTS", "0.2,0.1,0.05,0.025"), ","))
const TEND = parse(Float64, get(ENV, "XXE_T", "2.0"))
const PARTS = split(get(ENV, "XXE_PARTS", "local,global,s7"), ",")
const TIGHT = (tol_basis = 1e-10, tol_trunc = 1e-12, tol_expv = 1e-13, maxdim = 512, order = 3)

basis = spin_xx_sector(L, 1)[2]
load_ref(t) = deserialize(joinpath(SNAP, "ref_L$(L)_t$(t).jls"))
eps_z(psi, t) = sqrt(sum(abs2, spin_sz_profile(psi, 1) .- (real(diag(xx_open_correlation(L, t))) .- 0.5)) / L)
slopes(hs, es) = [log(es[k] / es[k + 1]) / log(hs[k] / hs[k + 1]) for k in 1:(length(hs) - 1)]
verdict(info) = (met = info.ledger.tolerance_met, unmet = string(info.ledger.unmet),
                 limits = string(info.record.limits), workcap = info.ledger.workcap_reached,
                 maxdim_hit = info.ledger.maxdim_reached, rmax = info.ledger.r_final)

function one_step(t0, h; kw...)
    psi = load_ref(t0)
    e_in = xx_state_error(psi, t0, basis).err
    info = cbe_bug_step!(psi, MPO_XX, ComplexF64(-h * im); merge(TIGHT, values(kw))...,
                         rng = MersenneTwister(SEEDS[1]))
    e = xx_state_error(psi, t0 + h, basis)
    return (t0 = t0, h = h, err = e.err, err_phase = e.err_phase, err_in = e_in, verdict(info)...)
end

out = Dict{String, Any}()
if "local" in PARTS
    rows = Any[]
    for t0 in (0.5, 4.0)
        es = Float64[]
        for h in DTS
            r = one_step(t0, h)
            push!(rows, r); push!(es, r.err)
            @printf("XXE-S6L t0=%.1f h=%.4f err=%.3e (input %.1e) met=%s limits=%s rmax=%d\n", t0, h,
                    r.err, r.err_in, r.met, r.limits, r.rmax)
        end
        @printf("XXE-S6L-SLOPES t0=%.1f %s (target 4)\n", t0, string(round.(slopes(DTS, es); digits = 2)))
    end
    out["local"] = rows
end
if "global" in PARTS
    rows = Any[]
    es = Float64[]
    for dt in DTS
        psi = domain_wall_state(L)
        n = round(Int, TEND / dt)
        nunmet, lim = 0, Set{String}()
        for k in 1:n
            info = cbe_bug_step!(psi, MPO_XX, ComplexF64(-dt * im); TIGHT...,
                                 rng = MersenneTwister(SEEDS[1] + k))
            v = verdict(info)
            v.met || (nunmet += 1)
            push!(lim, v.limits)
        end
        e = xx_state_error(psi, TEND, basis)
        ez = eps_z(psi, TEND)
        push!(rows, (dt = dt, nstep = n, err = e.err, err_phase = e.err_phase, fid = e.fid, eps_z = ez,
                     n_unmet = nunmet, limits = collect(lim), bonds = bond_dims(psi)))
        push!(es, e.err)
        @printf("XXE-S6G dt=%.4f steps=%d err=%.3e err_phase=%.3e eps_z=%.3e unmet_steps=%d maxbond=%d\n",
                dt, n, e.err, e.err_phase, ez, nunmet, maximum(bond_dims(psi)))
    end
    @printf("XXE-S6G-SLOPES %s (target 3)\n", string(round.(slopes(DTS, es); digits = 2)))
    out["global"] = rows
end
if "s7" in PARTS
    rows = Any[]
    for tb in (1e-3, 1e-5, 1e-7, 1e-9)
        r = one_step(4.0, 0.1; tol_basis = tb)
        push!(rows, merge((knob = "tol_basis", value = tb), r))
        @printf("XXE-S7 tol_basis=%.0e err=%.3e met=%s unmet=%s rmax=%d\n", tb, r.err, r.met, r.unmet, r.rmax)
    end
    for tt in (1e-6, 1e-8, 1e-10, 1e-12)
        r = one_step(4.0, 0.1; tol_trunc = tt)
        push!(rows, merge((knob = "tol_trunc", value = tt), r))
        @printf("XXE-S7 tol_trunc=%.0e err=%.3e met=%s unmet=%s rmax=%d\n", tt, r.err, r.met, r.unmet, r.rmax)
    end
    out["s7"] = rows
end
open(joinpath(OUT, "s6_s7_order.json"), "w") do io
    JSON.json(io, out; allownan = true)
end
println("XXE-S6S7-DONE")
