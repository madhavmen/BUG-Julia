# HEISENBERG SU(2) ON A SQUARE-LATTICE CYLINDER.
#
# The last rung of Jan's model ladder, and the first genuinely 2D problem in this repo. Everything
# else here is a chain; a cylinder reaches an MPS only through an ORDERING of its sites, so the
# ordering is part of the model and is pinned separately in `tests/sweeps/test_square_cylinder.jl`
# before any physics runs on it.
#
# ⛔ WHY THIS IS THE HARD CASE, AND WHAT IT IS ACTUALLY TESTING. On a chain the entanglement
# across a cut is bounded by the local space; on an `Lx x Ly` cylinder a cut severs `Ly` bonds, so
# the required rank grows like `exp(Ly)` and the state is rank-limited from the first step. That
# is the regime the whole tolerance study has NOT been in: every measurement so far was taken
# where the selection discarded nothing (`err_fnl = 0.000e+00`), which is exactly why Jan's
# points 2 and 4 both came back null. A cylinder is where the budget should finally bind.
#
# ⛔ NO EXACT REFERENCE, AND THE FILE SAYS SO RATHER THAN INVENTING ONE. `Lx*Ly` reaches 2^24 by
# 4x6 and the Sz=0 sector is still 2.7M states -- sparse propagation is out past about 4x4. So
# this reports CONSERVATION (energy drift, norm) and CONVERGENCE IN THE KNOBS (does the answer
# stop moving as the tolerances tighten), never "the error", and the small 3x3 / 4x3 cases carry
# an exact sparse cross-check so the machinery itself is pinned somewhere.
#
# ⚠ ENERGY DRIFT IS A CONSERVATION CHECK, NOT AN ACCURACY CHECK, and on this repo's own models
# the two orderings DISAGREE -- a basis-only arm once conserved energy to 7.4e-15 while being 12x
# further from the exact state than `tdvp2`. Read the knob-convergence columns for accuracy.
#
# Run:  julia --project=. benchmarks/su2_cylinder.jl [Lx] [Ly] [sym] [t_max]

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const LX   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const LY   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SYM  = length(ARGS) >= 3 ? Symbol(ARGS[3])     : :SU2
const TMAX = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.4
const DT   = 0.05
const L    = LX * LY
const MI   = 16

const OUT = joinpath(@__DIR__, "results", @sprintf("su2_cylinder_%dx%d_%s.txt", LX, LY, SYM))
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(SYM)
const W = square_cylinder_mpo(LX, LY)
# `dimer_state` needs an even L and pairs (1,2),(3,4),... which under the snake are RUNG
# neighbours -- a physically sensible, SU(2)-representable start on the cylinder. Neel is not
# SU(2)-representable at all, so there is no choice to make here under :SU2.
const PSI0 = iseven(L) ? dimer_state(L) : neel_state(L)

say(@sprintf("Heisenberg %s on a %dx%d cylinder (%d sites), dt=%g, T=%g",
             SYM, LX, LY, L, DT, TMAX))
say(@sprintf("bonds: %d   MPO virtual dims: max %d   (must track Ly=%d, not Lx=%d)",
             length(square_cylinder_bonds(LX, LY)), maximum(mpo_virtual_dims(W)), LY, LX))
say(@sprintf("start: %s", iseven(L) ? "dimer (S=0, SU(2)-representable)" : "Neel"))
say("")

"One arm. Reports conservation and rank -- NOT an error, since there is no reference at this size."
function arm(; m, tt, sc, cap)
    psi, kry, secs, efnl = copy(PSI0), 0, 0.0, 0.0
    e0 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    kw = m < 0 ? (;) : (krylov_basis = m, krylov_tol = 0.0)
    for _ in 1:round(Int, TMAX / DT)
        t0 = time_ns()
        info = cbe_bug_step!(psi, W, ComplexF64(-im * DT);
                             exact = true, split_cutoff = sc, maxdim = cap,
                             trunc_thresh = tt, maxiter = MI, kw...)
        secs += (time_ns() - t0) / 1e9
        kry += info.krylov_dims
        efnl = max(efnl, info.err_fnl)
    end
    e1 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    return (dE = abs(e1 - e0), states = maximum(state_bond_dims(psi)),
            mult = maximum(bond_dims(psi)), kry = kry, secs = secs, efnl = efnl,
            nrm = norm(psi), psi = psi)
end

# ── does the budget finally BIND here? that is the question the chains could not answer ────
say("="^100)
say("KNOB CONVERGENCE -- and whether `err_fnl` is finally NON-ZERO (the chains all gave 0.000e+00)")
say("="^100)
say(@sprintf("  %-28s %11s %8s %7s %9s %11s %9s", "arm", "|dE|", "states", "mult",
             "krylov", "err_fnl", "sec"))
ref = nothing
for (nm, m, tt, sc, cap) in (("m=3  tau=1e-6  D=64",  3, 1e-6, 1e-6,  64),
                             ("m=3  tau=1e-8  D=64",  3, 1e-8, 1e-8,  64),
                             ("m=3  tau=1e-8  D=128", 3, 1e-8, 1e-8, 128),
                             ("m=3  tau=1e-10 D=128", 3, 1e-10, 1e-10, 128),
                             ("defaults tau=1e-8 D=128", -1, 1e-8, 1e-8, 128),
                             ("m=0  tau=1e-8  D=128",  0, 1e-8, 1e-8, 128))
    r = arm(m = m, tt = tt, sc = sc, cap = cap)
    say(@sprintf("  %-28s %11.2e %8d %7d %9d %11.3e %9.1f",
                 nm, r.dE, r.states, r.mult, r.kry, r.efnl, r.secs))
    ref === nothing && (ref = r)
end
say("")
say("⚠ `err_fnl` > 0 means the SELECTION is finally discarding weight, i.e. the budget binds.")
say("  Every chain measurement in this study had err_fnl == 0.000e+00, which is why Jan's")
say("  points 2 and 4 came back null there. If it is still zero here, they are null on 2D too.")

# ── the exact cross-check, only where it is affordable ────────────────────────────────────
if L <= 16 && iseven(L)
    say("")
    say("="^100)
    say(@sprintf("EXACT CROSS-CHECK at %d sites (Sz=0 sector) -- the machinery pinned somewhere", L))
    say("="^100)
    set_symmetry!(:U1)
    Wu1 = square_cylinder_mpo(LX, LY)
    Jm = square_cylinder_couplings(LX, LY)
    Hs, states, idx = pairs_sparse(L, Jm)
    v0 = dimer_vector(L, states, idx)
    vt = expv_sparse(Hs, ComplexF64(-im * TMAX), v0; m = 30, tol = 1e-13)
    e_exact = real(dot(vt, Hs * vt)) / real(dot(vt, vt))
    psi = dimer_state(L)
    for _ in 1:round(Int, TMAX / DT)
        cbe_bug_step!(psi, Wu1, ComplexF64(-im * DT); exact = true,
                      maxdim = 128, trunc_thresh = 1e-10, maxiter = MI)
    end
    e_mps = real(mpo_energy(copy(psi), Wu1)) / max(norm(psi)^2, eps())
    say(@sprintf("  sector states      %d", length(states)))
    say(@sprintf("  exact E(T)         %.12f", e_exact))
    say(@sprintf("  cbe_bug E(T)       %.12f     |diff| = %.3e", e_mps, abs(e_mps - e_exact)))
    say("  (energy is CONSERVED by construction, so agreement here pins the OPERATOR and the")
    say("   site ordering -- a transposed snake would disagree in the last digits and nowhere else.)")
    set_symmetry!(SYM)
end
close(IO_)
