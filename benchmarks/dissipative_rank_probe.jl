# benchmarks/dissipative_rank_probe.jl -- why does BUG's rank outgrow TDVP2's on A1 once the working
# frame hits its ceiling? (Task 7 A1 L = 24, maxdim 512: BUG D_final 1.5-3x TDVP2 at the same tol_trunc
# while observables agree to 3e-6; the bond profile dips at the root bond instead of being flat.)
#
# ONE dt step of each arm from the same state psi0 (A1 L = 12, U = 4, kappa = 2, dbl_rich, evolved to
# t0 by fine TDVP2), against psi_ex = the same step by 32 TDVP2 substeps at tol_trunc 1e-12:
#   bug512 / bug2048       BUG order 3, maxdim 512 (work ceiling 1024, binds) / 2048 (ceiling 4096)
#   bug512u / bug2048u     the same with truncate = false: the evolved state before the compression
#   bug512o1u              order 1, maxdim 512, untruncated: is it the frontier closure?
#   tdvp512                TDVP2, two steps of dt/2, maxdim 512
#   bug2048b10 / ...b10u   tol_basis 1e-10 instead of 1e-8 (suffix bN = tol_basis 1e-N): is the excess
#                          the Galerkin error of tol_basis kept by a tighter tol_trunc?
#   bug512w / bug512wu     cap_order = :weight: the capped frame keeps its heaviest directions
# Printed per arm: distance to psi_ex, bond dims, per-bond counts of normalised Schmidt values above
# 1e-2 .. 1e-10, frame widths per bond (BUG) and the step's limits. A spurious tail shows as counts at
# 1e-6 .. 1e-10 above psi_ex's at the same bonds.
#
# Env: RP_L (12), RP_T0 ("0.2,0.4"), RP_DT (0.1), RP_TOLT (1e-9), RP_OUT (rank_probe.json),
#      RP_ARMS (comma list, default the six arms above without the bN ones)

using LinearAlgebra, Printf, JSON
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

env(k, d) = get(ENV, k, string(d))
const L = parse(Int, env("RP_L", 12))
const T0S = parse.(Float64, split(env("RP_T0", "0.2,0.4"), ","))
const DT = parse(Float64, env("RP_DT", 0.1))
const TOLT = parse(Float64, env("RP_TOLT", 1e-9))
const OUT = env("RP_OUT", "rank_probe.json")
const THRESH = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
const ARMS = split(env("RP_ARMS", "tdvp512,bug512,bug2048,bug512u,bug2048u,bug512o1u"), ",")

set_symmetry!(:U1)
const G = no_jump_mpo(hubbard_aa_mpo_d4(L; J = 1.0, U = 4.0), doublon_loss_mpo(L; kappa = 2.0))

tdvp!(psi, h; maxdim, tolt) = tdvp2_step!(psi, G, ComplexF64(-im * h); hermitian = false, maxdim = maxdim,
                                           trunc_thresh = 1e-14, maxiter = 40, conv_tol = 1e-12,
                                           weight_tol = tolt)
function normalise!(psi)
    canonical!(psi, 1)
    psi[1] = to_concrete(psi[1] / norm(psi[1]))
    return psi
end
dist(a, b) = sqrt(abs(real(overlap(a, a)) + real(overlap(b, b)) - 2 * real(overlap(a, b)))) /
             sqrt(real(overlap(b, b)))
# `dist` floors at ~1e-8 (cancellation of overlaps, rp7e): the local observables and log P0 resolve
# far below it, as the driver's errors do
const Q4 = hub4_space()
const NUP = to_concrete(lock(Q4.Fu', 2) * Q4.Fu)
const NDN = to_concrete(lock(Q4.Fd', 2) * Q4.Fd)
const DBL = to_concrete(lock(NUP, 1) * NDN)
obs(p) = vcat([real(site_expval(p, j, DBL)) for j in 1:L], [real(site_expval(p, j, NUP)) for j in 1:L],
              [real(site_expval(p, j, NDN)) for j in 1:L])
obserr(a, b) = max(maximum(abs.(obs(a) .- obs(b))),
                   abs(log(real(overlap(a, a))) - log(real(overlap(b, b)))))
function tails(psi)
    n = norm(psi)
    return [[count(>(x), bond_spectrum(copy(psi), b) ./ n) for x in THRESH] for b in 1:(L - 1)]
end

function arm(psi0, name)
    psi = copy(psi0)
    t0 = time_ns()
    rec = Dict{String, Any}("arm" => name)
    if startswith(name, "tdvp")
        for _ in 1:2
            tdvp!(psi, DT / 2; maxdim = 512, tolt = TOLT)
        end
    else
        md = occursin("2048", name) ? 2048 : 512
        mb = match(r"b(\d+)", name)
        tb = mb === nothing ? 1e-8 : 10.0^(-parse(Int, mb[1]))
        rec["tol_basis"] = tb
        rec["cap_order"] = occursin("w", name) ? "weight" : "sector"
        info = cbe_bug_step!(psi, G, ComplexF64(-im * DT); hermitian = false,
                             order = occursin("o1", name) ? 1 : 3, tol_basis = tb, tol_root = 1e-12,
                             tol_trunc = TOLT, maxdim = md, truncate = !endswith(name, "u"),
                             cap_order = Symbol(rec["cap_order"]))
        rec["frames"] = info.expanded
        rec["root_right"] = info.root_right
        rec["limits"] = string.(info.record.limits)
        rec["D_stage"] = info.record.D_stage
        rec["D_work"] = info.record.D_work
        # who filled the frame: DD-QR CBE (`n_selected`) vs Krylov C/D per half-sweep bond, then
        # what each frontier sweep added and whether the ceiling stopped it
        rec["bond_log"] = [Dict("bond" => e.bond, "side" => string(e.side), "width" => e.width,
                                "n_selected" => e.n_selected, "n_krylov" => e.n_krylov,
                                "workcap" => e.workcap, "pass" => get(e, :pass, 0),
                                "frontier" => get(e, :frontier, 0))
                           for e in info.ledger.bond_log]
        rec["closure_log"] = [Dict(string(k) => (v isa Number || v isa Bool ? v : string(v))
                                   for (k, v) in pairs(e)) for e in info.ledger.closure_log]
    end
    rec["wall"] = (time_ns() - t0) / 1e9
    return psi, rec
end

results = Any[]
psi = hub4_product_state([isodd(j) ? :both : :empty for j in 1:L])
t = 0.0
for t0 in T0S
    global psi, t
    while t < t0 - 1e-12                         # the start state: fine TDVP2, tight cut
        tdvp!(psi, 0.0125; maxdim = 2048, tolt = 1e-11)
        t += 0.0125
    end
    psi0 = normalise!(copy(psi))
    ex = copy(psi0)
    for _ in 1:32
        tdvp!(ex, DT / 32; maxdim = 2048, tolt = 1e-12)
    end
    @printf("== t0 = %.2f  psi0 bonds %s\n   psi_ex bonds %s\n", t0, bond_dims(psi0), bond_dims(ex))
    tex = tails(ex)
    println("   psi_ex tails (>", join(THRESH, ","), ") per bond: ", tex)
    push!(results, Dict("t0" => t0, "arm" => "psi_ex", "bonds" => bond_dims(ex), "tails" => tex,
                        "psi0_bonds" => bond_dims(psi0)))
    for name in ARMS
        p, rec = arm(psi0, name)
        rec["t0"] = t0
        rec["dist"] = dist(p, ex)
        rec["obserr"] = obserr(p, ex)
        rec["bonds"] = bond_dims(p)
        rec["tails"] = tails(p)
        @printf("-- %-10s dist %.3e  obs/logP0 err %.3e  wall %.1f s  bonds %s\n", name, rec["dist"],
                rec["obserr"], rec["wall"], rec["bonds"])
        haskey(rec, "frames") && @printf("   frames %s root_right %d D_stage %d D_work %d limits %s\n",
                                         rec["frames"], rec["root_right"], rec["D_stage"], rec["D_work"],
                                         rec["limits"])
        if haskey(rec, "bond_log")
            for e in rec["bond_log"]
                e["pass"] == 0 || continue
                @printf("   half %-5s bond %2d width %5d cbe %5d krylov %5d workcap %s\n", e["side"],
                        e["bond"], e["width"], e["n_selected"], e["n_krylov"], e["workcap"])
            end
            for e in rec["closure_log"]
                println("   frontier ", get(e, "frontier", -1), " pass ", get(e, "pass", -1),
                        " added ", get(e, "added", 0), " width ", get(e, "width_before", "-"), "->",
                        get(e, "width_after", "-"), " capped ", get(e, "capped", false),
                        " leak ", get(e, "leak", NaN))
            end
        end
        println("   tails: ", rec["tails"])
        push!(results, rec)
        open(OUT, "w") do io
            JSON.print(io, results)
        end
        flush(stdout)
    end
end
println("RANK-PROBE DONE")
