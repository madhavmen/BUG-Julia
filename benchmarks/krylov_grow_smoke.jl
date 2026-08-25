# SMOKE TEST FOR `krylov_grow` (Jan's point 4) -- DOES IT WORK, AND IS IT COMPETITIVE?
#
# This runs BEFORE any matrix job. `krylov_grow` re-expands the bond between Lanczos steps and
# embeds the accumulated Krylov vectors into the widened space, which involves leg indices that
# are NOT mirror images between the two half-sweeps:
#
#   :left   opposite frame `V_ex = (g, site, link)`  bond leg 1;  vectors `(link, site, g)` leg 3
#   :right  opposite frame `U_ex = (link, site, g)`  bond leg 3;  vectors `(g, site, link)` leg 1
#
# Getting that backwards is SILENT -- the shapes still contract, they just contract the wrong leg,
# and the answer comes out plausible and wrong. So the first three checks below are not about
# accuracy at all; they are about whether the thing is doing algebra or nonsense.
#
# THE FOUR GATES, in the order they can fail:
#
#   1. RUNS            no exception, and `n_grow > 0` -- if it never grew, nothing was tested.
#   2. WELL-FORMED     norm preserved and energy conserved. A wrong-leg embedding breaks these
#                      long before it breaks the error, because it silently mixes bond sectors.
#   3. NOT WORSE       at the SAME depth `m`, growing must not be less accurate than not growing.
#                      Growing strictly enlarges the space the Krylov vectors live in, so an
#                      error that goes UP means the embedding is wrong, not that the method is.
#   4. COMPETITIVE     it has to beat `grow = false` at EQUAL COST, not at equal `m` -- growing
#                      spends extra `cbe_expand` passes that `krylov_dims` cannot see. The
#                      honest comparison is against a DEEPER fixed-width run, and `n_grow` is
#                      printed so the extra work is visible rather than free.
#
# ⛔ GATE 4 IS THE ONE THAT MATTERS AND THE ONE MOST LIKELY TO FAIL. `cbe_lubich_sweep`'s
# `grow_iters` already showed that re-expanding per pass can buy 1.8x and then go flat; if
# `krylov_grow` lands there too, the honest answer is that point 4 does not pay on these models.
#
# Run:  julia --project=. benchmarks/krylov_grow_smoke.jl

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "analytic_reference.jl"))
include(joinpath(@__DIR__, "exact_sparse.jl"))

const L    = 12
const DT   = 0.05
const TMAX = 0.5              # 10 steps: enough to separate the arms, short enough to iterate
const CAP  = 64
const MI   = 16

"One arm. Returns error, rank, cost, growth passes, and the two well-formedness numbers."
function arm(mpo, psi0, obs, exact_obs; m, grow, tol = 0.0)
    psi, kry, ng = copy(psi0), 0, 0
    e0 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
    for _ in 1:round(Int, TMAX / DT)
        i = cbe_bug_step!(psi, mpo, ComplexF64(-im * DT);
                          krylov_basis = m, krylov_tol = tol, krylov_grow = grow,
                          exact = true, maxdim = CAP, trunc_thresh = 1e-12, maxiter = MI)
        kry += i.krylov_dims
        ng  += i.n_grow
    end
    e1 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
    return (err = maximum(abs.(obs(psi) .- exact_obs)), chi = maximum(bond_dims(psi)),
            kry = kry, ng = ng, dE = abs(e1 - e0), nrm = norm(psi))
end

_sz(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())

failures = String[]
note(ok, msg) = (ok || push!(failures, msg); ok)

function report(title, mpo, psi0, obs, exact_obs)
    println("\n" * "="^92); println(title); println("="^92)
    @printf("  %-26s %13s %6s %9s %7s %11s %9s\n",
            "arm", "err", "chi", "krylov", "n_grow", "dE", "norm-1")
    res = Dict{String, Any}()
    for m in (2, 3), grow in (false, true)
        r = arm(mpo, psi0, obs, exact_obs; m = m, grow = grow)
        nm = "m=$m grow=$grow"
        res[nm] = r
        @printf("  %-26s %13.4e %6d %9d %7d %11.2e %9.1e\n",
                nm, r.err, r.chi, r.kry, r.ng, r.dE, abs(r.nrm - 1))
        flush(stdout)
    end
    # A deeper FIXED-WIDTH arm, as the equal-cost yardstick for gate 4.
    for m in (5, 8)
        r = arm(mpo, psi0, obs, exact_obs; m = m, grow = false)
        res["m=$m grow=false"] = r
        @printf("  %-26s %13.4e %6d %9d %7d %11.2e %9.1e\n",
                "m=$m grow=false (yardstick)", r.err, r.chi, r.kry, r.ng, r.dE, abs(r.nrm - 1))
        flush(stdout)
    end

    for m in (2, 3)
        g, f = res["m=$m grow=true"], res["m=$m grow=false"]
        note(g.ng > 0, "$title m=$m: GATE 1 -- n_grow == 0, growth never happened")
        note(abs(g.nrm - 1) < 1e-8, "$title m=$m: GATE 2 -- norm off by $(abs(g.nrm-1))")
        note(g.dE < 1e-8, "$title m=$m: GATE 2 -- energy drift $(g.dE)")
        # 1.5x slack: growing changes the space, so tiny non-monotonicity is not a defect.
        note(g.err <= f.err * 1.5,
             "$title m=$m: GATE 3 -- growing is WORSE ($(g.err) vs $(f.err)) => embedding suspect")
    end
    return res
end

set_symmetry!(:U1)
r_xx = report("XX nearest neighbour, L=$L  (control: depth is inert here)",
              xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L), _sz,
              xx_free_fermion_sz(L, TMAX; J = 1.0, occupied = 1:(L ÷ 2)))

set_symmetry!(:U1)
r_hei = let
    H, states, idx = heisenberg_sparse(L)
    v = expv_sparse(H, ComplexF64(-im * TMAX), neel_vector(L, idx); m = 30, tol = 1e-13)
    report("Heisenberg XXX, L=$L, Neel quench  (exact sparse reference)",
           xxz_mpo(L; J = 1.0, delta = 1.0), neel_state(L), _sz,
           magnetisation_dense(v, L, states))
end

set_symmetry!(:none)
r_oat = report("OAT, L=$L  (depth is worth 8 orders here -- the discriminating model)",
               oat_mpo(L), x_polarized_state(L),
               p -> [total_sx(copy(p)) / max(norm(copy(p))^2, eps())],
               [oat_total_sx(L, TMAX)])
set_symmetry!(:U1)

# ── GATE 4, stated as a verdict rather than left to the reader ────────────────────────────
println("\n" * "="^92)
println("GATE 4 -- is growing COMPETITIVE, i.e. better than fixed width AT EQUAL COST?")
println("="^92)
for (nm, r) in (("XX", r_xx), ("Heisenberg", r_hei), ("OAT", r_oat))
    g = r["m=3 grow=true"]
    # the fixed-width arm whose cost is closest from ABOVE, so growing is not flattered
    cands = [(k, v) for (k, v) in r if endswith(k, "grow=false") && v.kry >= g.kry]
    best = isempty(cands) ? nothing : argmin(x -> x[2].kry, cands)
    if best === nothing
        @printf("  %-12s no fixed-width arm reaches %d krylov -- growing is the cheapest here\n",
                nm, g.kry)
    else
        k, f = best
        # ⛔ A TIE IS A TIE, NOT A WIN. The first version of this line used `<=`, which reported
        # "GROWING WINS" on two models whose numbers were BIT-IDENTICAL -- and on OAT, where both
        # arms sit at machine precision so their ratio is roundoff over roundoff. Require a real
        # margin and say "no effect" when there is none, or the verdict launders a null result
        # into a feature.
        ratio = g.err / max(f.err, 1e-300)
        verdict = max(g.err, f.err) < 1e-12 ? "both at machine precision -- NO EFFECT" :
                  ratio < 0.7 ? "GROWING WINS" :
                  ratio > 1.4 ? "fixed width wins" : "NO EFFECT (tie)"
        @printf("  %-12s grow m=3: %.4e @ %d (+%d widenings)  vs  %-16s %.4e @ %d   -> %s\n",
                nm, g.err, g.kry, g.ng, k, f.err, f.kry, verdict)
    end
end

println("\n" * "="^92)
if isempty(failures)
    println("ALL CORRECTNESS GATES PASSED (1-3). Gate 4 is a judgement -- read the verdicts above.")
else
    println("FAILURES:")
    for f in failures
        println("  x ", f)
    end
end
println("="^92)
exit(isempty(failures) ? 0 : 1)
