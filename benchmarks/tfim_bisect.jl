# WHERE INSIDE ONE `cbe_bug` STEP DOES THE TFIM ERROR COME FROM?
#
# ESTABLISHED BY `tfim_anomaly.jl`, and this file starts from it:
#   * L=8, rank UNCONSTRAINED (maxdim 1024, never reached): `tdvp_cbe1s` and `cbe_bug` both end
#     at chi = 9 and are 2.24e-14 against 2.12e-10 -- FOUR ORDERS APART AT THE SAME RANK with no
#     truncation binding. So it is not rank, not truncation, not selection.
#   * dt ratios 15-16 on TFIM (better than 2nd order) and no plateau -- so it is not a broken
#     composition either. A large CONSTANT.
#   * Long-range XX at matched alpha puts `cbe_bug` 10-32x behind `tdvp2` at EVERY range
#     including nearest-neighbour, nowhere near TFIM's 1775x. So it is not interaction range.
#   * `:none` reproduces it, so it is not Z2 charge bookkeeping.
#
# WHAT IS LEFT is the sweep itself on this model, and one step is enough to see it. Each row
# below switches off exactly one part of the step, so a row that COLLAPSES the error names the
# culprit and a row that does not exonerates that part.
#
# ⛔ THE METRIC IS `max|dX|` AGAINST THE MAJORANA PROPAGATOR, NOT A STATE INFIDELITY. The dense
# TFIM basis convention and `dense_state`'s are MIRRORED (site 1 is the low bit in one and the
# high bit in the other, and a set bit means the opposite spin), and the kink state is close
# enough to symmetric that a convention error would be nearly invisible. The Majorana reference
# is already pinned against `x_profile` by the examples, so it sidesteps that entirely.
#
# Run:  julia --project=. benchmarks/tfim_bisect.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

function tfim_majorana_generator(L::Int; J::Real = 1.0, h::Real = 1.0)
    A = zeros(Float64, 2L, 2L)
    for j in 1:L
        A[2j - 1, 2j] = -2h; A[2j, 2j - 1] = +2h
    end
    for j in 1:(L - 1)
        A[2j, 2j + 1] = -2J; A[2j + 1, 2j] = +2J
    end
    return A
end

function tfim_x_exact(L::Int, t::Real, dirs::Vector{Symbol}; J = 1.0, h = 1.0)
    G = zeros(Float64, 2L, 2L)
    for (j, d) in enumerate(dirs)
        s = d === :plus ? 1.0 : -1.0
        G[2j - 1, 2j] = s; G[2j, 2j - 1] = -s
    end
    R  = exp(tfim_majorana_generator(L; J = J, h = h) * t)
    Gt = R * G * transpose(R)
    return [Gt[2j - 1, 2j] for j in 1:L]
end

const L  = 8
const D  = 1024                       # far above the L=8 ceiling of 16: never binds
const MI = 40

set_symmetry!(:Z2)
const DIRS = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
const MPO  = tfim_mpo(L; J = 1.0, h = 1.0)
const PSI0 = ising_kink_state(L)

"One step of `dt` from the initial state; returns `(max|dX|, chi)`."
function one_step(dt, step!)
    psi = copy(PSI0)
    step!(psi, ComplexF64(-im * dt))
    return maximum(abs.(x_profile(copy(psi)) - tfim_x_exact(L, dt, DIRS))),
           maximum(bond_dims(psi))
end

bug(; kw...) = (p, tau) -> cbe_bug_step!(p, MPO, tau; maxdim = D, trunc_thresh = 1e-14,
                                         maxiter = MI, kw...)

println("="^80)
@printf("ONE STEP  TFIM L=%d  maxdim=%d (never binds; the L=8 ceiling is 16)  dt=0.01\n", L, D)
println("A row that COLLAPSES the error names the culprit; one that does not exonerates it.")
println("="^80)

rows = ["tdvp2"                  => (p, t) -> tdvp2_step!(p, MPO, t; maxdim = D,
                                                          trunc_thresh = 1e-14, maxiter = MI),
        "tdvp_cbe1s"             => (p, t) -> tdvp_cbe1s_step!(p, MPO, t; maxdim = D,
                                                               trunc_thresh = 1e-14, maxiter = MI),
        "cbe_bug DEFAULT"        => bug(),
        "  maxiter=200"          => (p, t) -> cbe_bug_step!(p, MPO, t; maxdim = D,
                                                            trunc_thresh = 1e-14, maxiter = 200),
        "  truncate=false"       => bug(truncate = false),
        "  trunc_thresh=0"       => (p, t) -> cbe_bug_step!(p, MPO, t; maxdim = D,
                                                            trunc_thresh = 0.0, maxiter = MI),
        "  kaug (no rexpand)"    => bug(rexpand = false, kaug = true),
        "  kstep=false"          => bug(kstep = false),
        "  exact CBE"            => bug(exact = true),
        "  growth=8.0"           => bug(growth = 8.0),
        "  root=2"               => bug(root = 2),
        "  root=6"               => bug(root = 6)]

@printf("\n  %-24s %12s  %s\n", "arm", "max|dX|", "chi")
for (nm, st) in rows
    e, c = one_step(0.01, st)
    @printf("  %-24s %12.4e  %d\n", nm, e, c)
end

# ── does the single-step error SCALE with dt, or is there a floor? ────────────────────────
#
# A correct step of a 2nd-order scheme has local error O(dt^3), so each 10x cut in dt should
# drop it ~1000x. A FLOOR that dt cannot reach is a representation defect and not a time-
# stepping one -- it would mean the step is wrong even as the propagator becomes the identity.
println("\n" * "="^80)
println("SINGLE-STEP dt SCALING.  A floor means the defect is not about time stepping at all.")
println("="^80)
@printf("\n  %-24s %12s %12s %12s   %s\n", "arm", "dt=1e-2", "dt=1e-3", "dt=1e-4", "ratios")
for (nm, st) in ["tdvp_cbe1s" => (p, t) -> tdvp_cbe1s_step!(p, MPO, t; maxdim = D,
                                                            trunc_thresh = 1e-14, maxiter = MI),
                 "cbe_bug"    => bug()]
    es = [one_step(dt, st)[1] for dt in (1e-2, 1e-3, 1e-4)]
    @printf("  %-24s %12.3e %12.3e %12.3e   %8.1f %8.1f\n",
            nm, es[1], es[2], es[3], es[1] / es[2], es[2] / es[3])
end
println()
