# DOES THE RANDOMISED SKETCH ACTUALLY BUY SPEED IN THE GROUND-STATE SEARCH?
#
# ⛔ `matvec` CANNOT ANSWER THIS, AND THAT IS THE WHOLE DIFFICULTY. `exact = false` (the
# randomised sketch, the package default and the method) and `exact = true` (the exact SVD of the
# same candidate block) differ ONLY inside `cbe_expand`. They issue the SAME number of `expv`
# calls, so the operator-application count -- the cost axis used everywhere else in this repo,
# precisely because it is machine-independent -- is BLIND to the difference by construction.
# The same blindness bit `rexpand` earlier: its second `cbe_expand` is QR/matmul that adds no
# `expv`, so `krylov` could not price it either.
#
# So this has to be WALL TIME, and wall time here is unreliable: it has been measured to spread
# 2.6-2.8x on BIT-IDENTICAL work on this machine. Three controls make it usable anyway:
#
#   1. A DISCARDED WARM-UP before anything is timed. A first arm pays JIT -- measured at 263.5s
#      against 18.4s for the identical second arm -- so an un-warmed table ranks compilation.
#   2. REPEATS, reported as median and (min, max). A single number cannot be distinguished from
#      a scheduling spike; the spread is printed so the reader can judge whether a gap is real.
#   3. THE ACCURACY COLUMN BESIDE IT. A sketch that is faster and less accurate is not a speedup,
#      it is a different method. `err` must match the exact arm to be comparable at all.
#
# ⛔ AND IT MUST RUN WHERE THE EXPANSION IS ACTUALLY WORKING. At a loose cap the state converges
# before the sketch matters; at D=64 (full rank at L=12) the two arms were bit-identical in the
# defer study. D=8 and D=16 are where the cap binds on these models.
#
# Run:  julia --project=. benchmarks/gse_rsvd_vs_exact.jl

using LinearAlgebra, Printf, Random, Statistics
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate.GroundState

include(joinpath(@__DIR__, "..", "tests", "common", "analytic_reference.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

tfim_ground_energy(L::Int; J = 1.0, h = 1.0) =
    -0.5 * sum(sort(real.(eigvals(im * tfim_majorana_generator(L; J = J, h = h))); rev = true)[1:L])

const REPEATS = 3

"Run one configuration `REPEATS` times; return (median_time, min, max, err, matvec, sweeps)."
function timed_arm(psi0, h, eref; defer, exact, grow_iters, maxdim, n_sweeps = 30)
    ts = Float64[]; errf = NaN; mv = 0; swp = 0; chi = 0
    hm = h isa MPO ? h : mpo_from_terms(h)
    for _ in 1:REPEATS
        psi = copy(psi0)
        opts = CBEBugOptions(; growth = 2.0, maxdim = maxdim, trunc_thresh = 1e-12,
                             maxiter = 30, tol = 1e-14, exact = exact)
        t0 = time()
        info = solve!(psi, h; opts = opts, n_sweeps = n_sweeps, etol = 1e-12,
                      grow_iters = grow_iters, defer_truncation = defer)
        push!(ts, time() - t0)
        errf = abs(real(mpo_energy(copy(psi), hm)) / max(norm(psi)^2, eps()) - eref)
        mv = sum(info.krylov_dims); swp = length(info.energies); chi = maximum(bond_dims(psi))
    end
    return (median(ts), minimum(ts), maximum(ts), errf, mv, swp, chi)
end

function compare(title, psi0, h, eref; caps = (8, 16), grow_iters = 1)
    println("\n" * "="^96)
    println(title, "   (median of $REPEATS, warm)")
    println("="^96)
    @printf("  %-30s %8s %16s %12s %7s %4s\n",
            "arm", "median", "(min , max)", "|E-Eex|", "matvec", "chi")
    for D in caps, defer in (false, true)
        rows = Dict{Bool, Any}()
        for ex in (false, true)          # false = RSVD (the method), true = exact SVD (the A/B)
            rows[ex] = timed_arm(psi0, h, eref; defer = defer, exact = ex,
                                 grow_iters = grow_iters, maxdim = D)
            md, lo, hi, er, mv, swp, chi = rows[ex]
            @printf("  D=%-2d %-6s %-13s %7.2fs (%5.2f,%5.2f) %12.4e %7d %4d\n",
                    D, defer ? "DEFER" : "trunc", ex ? "exact SVD" : "RSVD",
                    md, lo, hi, er, mv, chi)
            flush(stdout)
        end
        # the comparison, stated rather than left to the reader
        r, e = rows[false], rows[true]
        same = abs(r[4] - e[4]) <= 0.05 * max(e[4], 1e-16)
        @printf("      -> RSVD is %.2fx %s%s\n\n", e[1] / r[1],
                e[1] > r[1] ? "FASTER" : "SLOWER",
                same ? "" : "  ⚠ ACCURACY DIFFERS -- not a like-for-like speedup")
        flush(stdout)
    end
end

# ── warm-up, discarded: the first arm in a session pays JIT and would rank compilation ───────
set_symmetry!(:U1)
let L = 8
    timed_arm(neel_state(L), xxz_chain(L; delta = 1.0), 0.0;
              defer = false, exact = false, grow_iters = 1, maxdim = 8, n_sweeps = 2)
end
println("(warm-up done)")

set_symmetry!(:U1)
e12, _, r12 = bethe_xxx_open_energy(12)
r12 < 1e-12 || error("Bethe residual $r12 too large")
compare("HEISENBERG L=12 :U1  delta=1", neel_state(12), xxz_chain(12; delta = 1.0), e12)

set_symmetry!(:Z2)
compare("TFIM L=12 :Z2  J=h=1 (critical)", ising_polarised_state(12),
        tfim_mpo(12; J = 1.0, h = 1.0), tfim_ground_energy(12))
