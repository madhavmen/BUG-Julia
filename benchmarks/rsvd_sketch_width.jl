# WHY IS RSVD-CBE NOT FASTER THAN THE EXACT SVD, AND WHAT WOULD MAKE IT SO?
#
# A randomised SVD is asymptotically faster than an exact one only when the number of directions
# wanted is SMALL COMPARED TO THE AMBIENT DIMENSION -- `k << n`. Below that ratio it is a strict
# loss: you draw and project a Gaussian nearly as wide as the exact basis, then factor something
# barely narrower, and pay the random draw on top.
#
# WHAT CBE ACTUALLY ASKS FOR (cbe_core.jl):
#
#     dmax   = max(chi_l, r, chi_r)
#     budget = ceil(growth * dmax) - r          # directions wanted
#     npre   = ceil(1.2 * budget)               # the SKETCH WIDTH (20% oversampling)
#     room   = d*chi - r                        # the FULL candidate space; d = 2 for spin-1/2
#
# On a bond sitting at rank `r = chi` with `dmax = chi`:
#
#     growth = 2.0  ->  budget = chi        ->  npre = 1.2*chi   vs full 2*chi   ->  60% OF FULL
#     growth = 1.1  ->  budget = 0.1*chi    ->  npre = 0.12*chi  vs full 2*chi   ->   6% OF FULL
#
# ⛔ SO AT THE SHIPPED `growth = 2.0` THERE IS NO ASYMPTOTIC REGIME LEFT TO EXPLOIT. The sketch is
# a constant fraction of the space it is meant to avoid touching. That is the whole explanation
# for the measured result -- RSVD 0.88-0.92x, i.e. SLOWER, on U(1) Heisenberg L=12 at D=8/16.
#
# ⛔ AND `growth` WAS RAISED 1.1 -> 2.0 IN THIS REPO, for a width-restoration mechanism that has
# since been RETIRED (2026-08-24): it expanded each bond twice, so its second expansion ran
# against a larger `U0` and `budget` SUBTRACTS that rank, which the old schedule starved. That
# change fixed the mechanism and, unnoticed, removed the regime in which the randomised sketch
# pays. The mechanism is gone; the `growth = 2.0` it was raised for is still the default, so THE
# CONCLUSION BELOW STILL HOLDS -- but the reason `growth` sits at 2.0 no longer exists, and
# re-deriving it against the current sweep would be worth doing before tuning the sketch again.
#
# ⛔ THE d = 2 CEILING IS THE HARD PART, and no tuning removes it. The candidate space for a
# spin-1/2 bond is `d*chi = 2*chi`. Asking for `k` new directions out of `2*chi` can only satisfy
# `k << n` if `k << 2*chi`, i.e. if the growth per sweep is a SMALL FRACTION of the current rank.
# A scheme that wants to roughly double the rank in one pass is asking for half the space and
# cannot be accelerated by sketching, whatever the implementation quality.
#
# THIS FILE MEASURES THE PREDICTION: RSVD/exact should cross 1.0 as `growth` falls, and the
# crossing point locates the `k/n` at which sketching starts paying on this problem.
#
# Run:  julia --project=. benchmarks/rsvd_sketch_width.jl

using LinearAlgebra, Printf, Random, Statistics
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate.GroundState

include(joinpath(@__DIR__, "..", "tests", "common", "analytic_reference.jl"))

const REPEATS = 3

function timed(psi0, h, eref; growth, exact, maxdim, grow_iters = 1, n_sweeps = 30)
    ts = Float64[]; errf = NaN; mv = 0; swp = 0
    hm = h isa MPO ? h : mpo_from_terms(h)
    for _ in 1:REPEATS
        psi = copy(psi0)
        opts = CBEBugOptions(; growth = growth, maxdim = maxdim, trunc_thresh = 1e-12,
                             maxiter = 30, tol = 1e-14, exact = exact)
        t0 = time()
        info = solve!(psi, h; opts = opts, n_sweeps = n_sweeps, etol = 1e-12,
                      grow_iters = grow_iters)
        push!(ts, time() - t0)
        errf = abs(real(mpo_energy(copy(psi), hm)) / max(norm(psi)^2, eps()) - eref)
        mv = sum(info.krylov_dims); swp = length(info.energies)
    end
    return (median(ts), errf, mv, swp)
end

function growth_scan(title, psi0, h, eref; maxdim = 16)
    println("\n" * "="^92)
    println(title, "   maxdim=$maxdim   (median of $REPEATS, warm)")
    println("="^92)
    # `npre/full` is the ratio that decides whether sketching can pay at all, printed so the
    # speedup column can be read against the mechanism rather than as a bare number.
    @printf("  %-8s %9s %9s %8s   %11s %11s   %6s\n",
            "growth", "RSVD (s)", "exact(s)", "RSVD/ex", "err RSVD", "err exact", "mv R/e")
    for g in (1.05, 1.1, 1.25, 1.5, 2.0, 3.0)
        r = timed(psi0, h, eref; growth = g, exact = false, maxdim = maxdim)
        e = timed(psi0, h, eref; growth = g, exact = true,  maxdim = maxdim)
        @printf("  %-8.2f %9.2f %9.2f %8.2fx   %11.4e %11.4e   %d/%d%s\n",
                g, r[1], e[1], e[1] / r[1], r[2], e[2], r[3], e[3],
                e[1] / r[1] > 1.0 ? "   <- RSVD FASTER" : "")
        flush(stdout)
    end
end

# warm-up, discarded -- a first arm pays JIT and would rank compilation, not the sketch
set_symmetry!(:U1)
timed(neel_state(8), xxz_chain(8; delta = 1.0), 0.0;
      growth = 2.0, exact = false, maxdim = 8, n_sweeps = 2)
println("(warm-up done)")

set_symmetry!(:U1)
e12, _, r12 = bethe_xxx_open_energy(12)
r12 < 1e-12 || error("Bethe residual $r12 too large")
growth_scan("HEISENBERG L=12 :U1  delta=1  -- DMRG-CBE + grow_iters, RSVD vs exact",
            neel_state(12), xxz_chain(12; delta = 1.0), e12; maxdim = 16)
growth_scan("HEISENBERG L=12 :U1  delta=1  -- larger cap, more room for the sketch to matter",
            neel_state(12), xxz_chain(12; delta = 1.0), e12; maxdim = 48)
