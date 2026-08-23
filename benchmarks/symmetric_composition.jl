# DOES A SYMMETRIC (REFLECTED) COMPOSITION IMPROVE `cbe_bug`'s TIME-STEPPING ERROR?
#
# WHAT IS AND IS NOT ASYMMETRIC IN THIS SWEEP, because it decides what "symmetrise" can mean:
#
#   * The K and L half-sweeps are NOT an asymmetry. Both read the FROZEN start-of-step state --
#     `rstack` and `lstack` are built once, before either runs -- and write into independent
#     `W[]` / `Z[]`. Swapping their order is a literal no-op, not a symmetrisation. (This is why
#     "reverse the sweep" does not mean here what it means for TDVP, whose sweeps mutate `psi`
#     as they go and therefore genuinely do not commute.)
#   * The ROOT POSITION IS the asymmetry. `c = L/2` is one distinguished bond that keeps the
#     single Galerkin step, and everything is built inward toward it. MEASURED elsewhere:
#     centre 8.5e-16, off-centre interior 1.2e-5, boundary 1.7e-5 -- the root position is the
#     largest structural lever in the method.
#
# SO THE REFLECTED COMPOSITION IS: S(tau/2) at root `c`, then S(tau/2) at root `L-c`. Standard
# symmetrisation of a one-step method, `S*(tau/2) . S(tau/2)`, with the mirror supplying the
# adjoint. It needs NO new code -- `root` is already a keyword.
#
# ══ RESULT: THE DEFAULT ROOT IS A FIXED POINT OF THE REFLECTION, SO THERE IS NOTHING TO CANCEL ══
#
# `root` is a BOND, `1:L-1` (see the `ArgumentError` in `cbe_bug_step!`), and reflection acts on
# bonds as `b -> L - b`. The default root is `L/2`, and on an EVEN chain `L - L/2 == L/2`. The
# default is its own mirror. MEASURED, XX L=16 D=24, and the arms came out BIT-IDENTICAL:
#
#     cbe_bug  full dt                 8.1911e-05     21526
#     cbe_bug  2x dt/2 (same root)     2.0258e-05     43546
#     cbe_bug  2x dt/2 MIRRORED        2.0258e-05     43546   <- same bond, so the same run
#
# Combined with the commuting half-sweeps above, that says `cbe_bug_step!` AT ITS DEFAULT ROOT IS
# ALREADY SPATIALLY REFLECTION-SYMMETRIC. Spatial symmetrisation is exhausted -- not because a
# close call went against it, but because the symmetry group has a fixed point exactly where the
# default sits.
#
# ⛔ AND THAT BIT-IDENTITY IS ITSELF THE MEASUREMENT OF THE COMMUTING CLAIM, which was an argument
# and not a number until this run. Were the half-sweeps order-dependent, the K-before-L ordering
# would STILL break the symmetry at a fixed-point root and the two arms would differ. They agree
# to the last printed digit at identical Krylov count.
#
# The substepping arm is retained because it is the BAR: 8.1911e-05 -> 2.0258e-05 for 21526 ->
# 43546 Krylov is 4.04x for 2x the cost, and that is what any 2x-cost scheme must beat.
#
# WHERE THIS WENT NEXT: `benchmarks/midpoint_basis.jl`, temporal rather than spatial
# symmetrisation (`basis_frac`), which costs 1x and so only has to beat 1x.
#
# The off-centre pair below is the one genuinely reflected composition available -- `c-1` and its
# true mirror `c+1` -- kept to test whether composing two mirrored OFF-CENTRE roots recovers what
# the single centred root already gives. The root-position scan (centre 8.5e-16, off-centre
# interior 1.2e-5) predicts it will not.
#
# ⛔ THE CONTROL IS PLAIN SUBSTEPPING, NOT THE SINGLE FULL STEP. Two half-steps are ~2x the cost
# of one full step, and at second order that alone buys ~4x. Comparing mirrored-substepping
# against the FULL step would credit the mirror with the substepping. The question is whether
# mirroring beats plain substepping AT IDENTICAL COST, and the two differ by nothing but which
# bond holds the root.
#
# ⛔ AND THE REAL BAR IS HIGHER STILL. `benchmarks/cost_fair.jl` measured `cbe_bug` at a 1.7x
# COST ADVANTAGE over `tdvp_cbe1s` at matched error. Any symmetrisation that doubles the cost
# must buy more than the ~4x that spending the same compute on substepping already buys, or it
# is a worse use of the budget than doing nothing. That is the number to beat, not the baseline.
#
# ⛔ AND ON A BINDING CAP IT MAY BE MOOT. `cost_fair.jl` also found the D=16 TFIM error is
# TRUNCATION-bound -- every scheme gets WORSE as dt shrinks. Where that holds, no improvement to
# the time-stepping error can help, so the D=16 arm is included to find out whether the regime
# that matters is even reachable by this.
#
# Run:  julia --project=. benchmarks/symmetric_composition.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

function compare(title, mpo, psi0, L, T, dt, D, maxiter, exact_at, measure)
    println("\n" * "="^84)
    @printf("%s   maxdim=%d  T=%g  dt=%g\n", title, D, T, dt)
    println("="^84)
    # ⛔ `root` IS A BOND IN `1:L-1` AND REFLECTION IS `b -> L-b`, so the default `L/2` is a FIXED
    # POINT on an even chain and `L - L÷2 == L÷2` gives back the SAME BOND. The only genuinely
    # reflected pair available is therefore an OFF-CENTRE one: `c-1` and its mirror `L-(c-1)`.
    c  = max(1, min(L - 1, L ÷ 2))
    lo = max(1, c - 1)
    hi = min(L - 1, L - lo)                       # the true mirror of `lo`
    n = round(Int, T / dt)
    bug(p, t; kw...) = cbe_bug_step!(p, mpo, t; maxdim = D, trunc_thresh = 1e-12,
                                     maxiter = maxiter, kw...)

    # ⛔ EVERY ARM RETURNS ITS OWN TOTAL KRYLOV, and a two-half-step arm must SUM both. Written
    # as `(bug(...); bug(...))` the closure returns only the SECOND info, so the substepped arms
    # report HALF their true cost -- which would make them look free in exactly the comparison
    # their cost is the point of. (Caught against `cost_fair.jl`: dt=0.025 on XX is 43546, and
    # the buggy version reported 21818 for the same work.)
    half = ComplexF64(-im * dt / 2)
    arms = ["cbe_bug  full dt"        => (p) -> _kry(bug(p, ComplexF64(-im * dt))),
            "cbe_bug  2x dt/2 (same root)" =>
                (p) -> _kry(bug(p, half)) + _kry(bug(p, half)),
            "cbe_bug  2x dt/2 off-centre MIRROR" =>
                (p) -> _kry(bug(p, half; root = lo)) + _kry(bug(p, half; root = hi)),
            "tdvp_cbe1s full dt"      =>
                (p) -> _kry(tdvp_cbe1s_step!(p, mpo, ComplexF64(-im * dt); maxdim = D,
                                             trunc_thresh = 1e-12, maxiter = maxiter))]

    @printf("\n  %-30s %12s %9s  %s\n", "arm", "err", "krylov", "chi")
    for (nm, st) in arms
        psi, kry = copy(psi0), 0
        for _ in 1:n
            kry += st(psi)
        end
        @printf("  %-30s %12.4e %9d  %d\n", nm, measure(psi, exact_at(T)), kry,
                maximum(bond_dims(psi)))
        flush(stdout)
    end
end

function run_xx(; L = 16, T = 6.0, dt = 0.05, D = 24, maxiter = 12)
    set_symmetry!(:U1)
    compare("XX domain wall  L=$L  :U1  (honest dt^2 regime)",
            xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L), L, T, dt, D, maxiter,
            t -> xx_free_fermion_sz(L, t; J = 1.0),
            (p, r) -> maximum(abs.(magnetisation(copy(p)) - r)))
end

function run_tfim(; L = 16, T = 3.0, dt = 0.05, D = 32, maxiter = 16)
    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    compare("TFIM  L=$L  :Z2  D=$D  (mild truncation)",
            tfim_mpo(L; J = 1.0, h = 1.0), ising_kink_state(L), L, T, dt, D, maxiter,
            t -> tfim_x_profile(L, t, dirs),
            (p, r) -> maximum(abs.(x_profile(copy(p)) - r)))
end

function run_tfim_tight(; L = 16, T = 3.0, dt = 0.05, D = 16, maxiter = 16)
    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    compare("TFIM  L=$L  :Z2  D=$D  (TRUNCATION-BOUND: dt improvements cannot help here)",
            tfim_mpo(L; J = 1.0, h = 1.0), ising_kink_state(L), L, T, dt, D, maxiter,
            t -> tfim_x_profile(L, t, dirs),
            (p, r) -> maximum(abs.(x_profile(copy(p)) - r)))
end

run_xx(); run_tfim(); run_tfim_tight()
println()
