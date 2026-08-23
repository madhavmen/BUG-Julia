# DOES BUILDING THE BASES AT THE MIDPOINT RAISE `cbe_bug`'s ORDER?
#
# WHY THIS AND NOT A REFLECTED COMPOSITION. `benchmarks/symmetric_composition.jl` closed the
# spatial question: the K and L half-sweeps commute (both read the FROZEN start-of-step state), so
# the ROOT is the only spatial asymmetry -- and the root is a BOND in `1:L-1`, on which reflection
# acts as `b -> L-b`, whose FIXED POINT on an even chain is exactly the default root `L/2`. The
# sweep is therefore already spatially symmetric where it ships. Measured confirmation: mirroring
# the root was BIT-IDENTICAL to plain substepping, 2.0258e-05 at 43546 Krylov for both.
#
# So the remaining symmetrisation is TEMPORAL: the half-sweeps build the bases, the bases are the
# O(tau) object, and centring them is what a midpoint rule does. `basis_frac = 0.5` evolves the
# half-sweeps by `tau/2` and still takes the ROOT Galerkin step over the full `tau`.
#
# ⛔ THE BAR IS 1x, NOT 4x, AND THAT IS THE WHOLE POINT. A 2x-cost symmetrisation must beat what
# spending the same compute on substepping already buys -- measured at 4.04x on XX (8.1911e-05 ->
# 2.0258e-05 for 21526 -> 43546 Krylov). `basis_frac` changes no sweep count and no solve count;
# only the argument of the half-sweep `expv`. If it helps at all, it is free.
#
# ⛔ THE DECISIVE COLUMN IS THE RATIO, NOT THE ERROR. A single error value at one `dt` cannot
# distinguish "raised the order" from "smaller constant, same order", and those have opposite
# consequences for whether this matters at the `dt` anyone would actually run. Halving `dt` and
# reading the ratio separates them: ~4 is second order, ~8 third, ~16 fourth.
#
# ⛔ AND IT MUST BE READ IN A REGIME WHERE `dt` IS WHAT BINDS. On a tight cap the error is
# TRUNCATION-bound and every scheme gets WORSE as `dt` shrinks (measured on TFIM D=16), so a
# ratio there measures the cap, not the order. Both models below are run loose enough to scale.
#
# ⛔ SIZED FOR A LOCAL CORRECTNESS QUESTION, NOT A CAMPAIGN. "Does the order change" is answered
# by the RATIO, and a ratio is just as readable at L=12 as at L=16 -- so this runs at L=12 with a
# cap loose enough that `dt` and not truncation is what binds, which the baseline row's ~4 then
# CONFIRMS rather than assumes. The first draft ran L=16 / T=6 / 420 steps per arm and took 25
# MINUTES PER ARM; that is a cluster job (`scripts/run_julia.sbatch`), and it is the confirmation
# run, not the exploratory one.
#
# ══ RESULT: NO. SHORTENING THE BASIS EVOLUTION COSTS AN ORDER. LEAVE `basis_frac` AT 1.0. ══
#
#   XX   D=32   1.0 / 0.75 / 0.5 / 0.25 -> BIT-IDENTICAL (6.4211e-05), ratio 3.99, same krylov
#   TFIM D=32   1.0    4.8595e-05  3.1491e-06  1.9996e-07   ratio 15.43   <- best, 4th order
#               0.75   2.0043e-04  2.6416e-05  3.3209e-06   ratio  7.59
#               0.5    4.1845e-04  5.4065e-05  6.6882e-06   ratio  7.74
#               0.25   6.3668e-04  8.1329e-05  1.0051e-05   ratio  7.83
#
# 4th order (~15.5) down to 3rd (~7.8), and 21x worse at the finest dt. Not neutral -- harmful.
#
# WHY, and it is the same property that makes this a BUG and not a TDVP: the half-sweeps are
# BASIS GENERATORS. `C = contract(W[i]', K)` uses `K`, not `K1` -- the evolved AMPLITUDE is
# discarded and only the COLUMN SPAN survives. To leading order that span is
#
#     col(Kex) + col(exp(btau·H)·Kex) = col(Kex) + col(H·Kex) + O(btau)
#
# INDEPENDENT of how far the half-sweep evolved. Where nothing truncates the union, `basis_frac`
# is therefore exactly inert -- and the bit-identical XX column is that statement MEASURED rather
# than argued. Where the union IS truncated, a shorter `btau` does not select a different span,
# only a worse-resolved one: the `O(btau)` content separating good directions from marginal ones
# shrinks, so the truncation decides on less information. Hence monotone degradation as
# `basis_frac` falls, and hence the lost order.
#
# A midpoint rule needs the basis to CARRY the half-step state. This sweep discards amplitude by
# construction, so the midpoint BUG is not available to it at all. That is structural, not tuning.
#
# ⛔ SO THE XX COLUMN IS A LIVE INVARIANT, NOT A NULL RESULT. If a future change to the half-sweeps
# ever makes those four XX numbers differ, the sweep has started keeping amplitude somewhere it
# should not, and this file will say so.
#
# Run:  julia --project=. benchmarks/midpoint_basis.jl
#       julia --project=. benchmarks/midpoint_basis.jl big     # the L=16 confirmation, ~3h

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

# ── the dt-order scan: the same arm at three dt, reported with its ratios ────────────────────
function order_scan(title, mpo, psi0, T, dts, D, maxiter, exact_at, measure)
    println("\n" * "="^94)
    @printf("%s   maxdim=%d  T=%g\n", title, D, T)
    println("="^94)
    flush(stdout)              # or the header sits in the buffer until the first arm finishes
    ref = exact_at(T)

    arms = ["basis_frac=1.0  (endpoint, current default)" => 1.0,
            "basis_frac=0.75"                             => 0.75,
            "basis_frac=0.5   (MIDPOINT BUG)"             => 0.5,
            "basis_frac=0.25"                             => 0.25]

    @printf("\n  %-44s", "arm")
    for dt in dts; @printf("%12s", "dt=$dt"); end
    @printf("%10s%10s\n", "ratio", "krylov")

    for (nm, bf) in arms
        errs, krys = Float64[], Int[]
        for dt in dts
            psi, kry = copy(psi0), 0
            for _ in 1:round(Int, T / dt)
                kry += _kry(cbe_bug_step!(psi, mpo, ComplexF64(-im * dt);
                                          basis_frac = bf, maxdim = D,
                                          trunc_thresh = 1e-12, maxiter = maxiter))
            end
            push!(errs, measure(psi, ref)); push!(krys, kry)
        end
        # ratio across the FIRST halving; a second ratio would need a third dt in geometric
        # sequence, which `dts` supplies.
        r = length(errs) >= 2 && errs[2] > 0 ? errs[1] / errs[2] : NaN
        @printf("  %-44s", nm)
        for e in errs; @printf("%12.4e", e); end
        @printf("%10.2f%10d\n", r, krys[end])
        flush(stdout)
    end

    # the reference the whole comparison is against
    for dt in dts[end:end]
        psi, kry = copy(psi0), 0
        for _ in 1:round(Int, T / dt)
            kry += _kry(tdvp_cbe1s_step!(psi, mpo, ComplexF64(-im * dt); maxdim = D,
                                         trunc_thresh = 1e-12, maxiter = maxiter))
        end
        @printf("  %-44s%36s%10s%10d\n", "tdvp_cbe1s (finest dt only)",
                @sprintf("%12.4e", measure(psi, ref)), "", kry)
        flush(stdout)
    end
end

function run_xx(; L = 12, T = 2.0, D = 32, maxiter = 12)
    set_symmetry!(:U1)
    order_scan("XX domain wall  L=$L  :U1  (baseline ratio ~4 CONFIRMS dt is what binds)",
               xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L), T,
               [0.1, 0.05, 0.025], D, maxiter,
               t -> xx_free_fermion_sz(L, t; J = 1.0),
               (p, r) -> maximum(abs.(magnetisation(copy(p)) - r)))
end

function run_tfim(; L = 12, T = 1.5, D = 32, maxiter = 16)
    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    order_scan("TFIM  L=$L  :Z2  D=$D",
               tfim_mpo(L; J = 1.0, h = 1.0), ising_kink_state(L), T,
               [0.1, 0.05, 0.025], D, maxiter,
               t -> tfim_x_profile(L, t, dirs),
               (p, r) -> maximum(abs.(x_profile(copy(p)) - r)))
end

# `big` reproduces the first draft's sizes: the CONFIRMATION run, for the cluster.
if "big" in ARGS
    run_xx(; L = 16, T = 6.0, D = 24); run_tfim(; L = 16, T = 3.0, D = 32)
else
    run_xx(); run_tfim()
end
println()
