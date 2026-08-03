# RSVD-CBE BUG in 40 lines: an L=8 Heisenberg domain wall, with and without symmetry.
#
#   julia --project=. examples/heisenberg_domain_wall.jl
#   L=10 TMAX=3 SYMS=none,U1 julia --project=. examples/heisenberg_domain_wall.jl
#
# WHAT THIS SHOWS
#
#   1. how to switch symmetry           -- one call, `set_symmetry!(:none | :U1 | :SU2)`
#   2. how to run the default method    -- `RealTime.evolve!(psi, gates; opts)`
#   3. that symmetry changes the STORAGE, not the physics
#   4. that the answer is right         -- checked against an exact sparse-Krylov reference
#
# THE PHYSICS. A domain wall |↑↑↑↑↓↓↓↓> is not an eigenstate of the Heisenberg chain, so it
# melts: <Sz_j> spreads from the interface outward at a finite velocity. That spreading front
# is the light cone, and it is what the printout below draws.
#
# WHY THE DOMAIN WALL AND NOT A NÉEL STATE. It starts at chi=1 -- a product state. A method
# that cannot GROW the bond dimension is frozen there forever and will report a wall that
# never melts. RSVD-CBE grows the bond from directions drawn through the gate, so it can
# bootstrap from chi=1 with no random seeding anywhere. Watch the chi column climb from 1.
#
# NOTE ON SU(2): this example is abelian-only (`:none` and `:U1`) and that is not an
# oversight. A domain wall has definite Sz on every site, so it spans many total-spin sectors
# and is not representable under `:SU2`; `<Sz_j>` is also identically zero there by symmetry,
# so the light cone would be blank. For an SU(2) comparison use `dimer_state(L)` with
# `bond_energies`, which is an SU(2) scalar -- see `benchmarks/l18_matrix.jl`.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime
using LinearAlgebra, Printf

const L     = parse(Int,     get(ENV, "L",     "8"))
const DT    = parse(Float64, get(ENV, "DT",    "0.05"))
const TMAX  = parse(Float64, get(ENV, "TMAX",  "2.0"))
const CHI   = parse(Int,     get(ENV, "CHI",   "32"))
const SYMS  = [Symbol(s) for s in split(get(ENV, "SYMS", "none,U1"), ",")]
const NSTEP = round(Int, TMAX / DT)

# ── the run ──────────────────────────────────────────────────────────────────────────
"""
Evolve a domain wall with the RSVD-CBE bond-update BUG and return `(magnetisation, chi, info)`.

This is the whole API surface: pick a symmetry, build a state, build one gate per bond,
call `evolve!`. Everything else below is printing.
"""
function run_domain_wall(sym::Symbol; path::Symbol = :gate)
    set_symmetry!(sym)                                   # <-- THE SYMMETRY SWITCH

    psi   = domain_wall_state(L)                         # |↑↑↑↑↓↓↓↓>, chi = 1
    gates = bond_gates(psi; J = 1.0, delta = 1.0)        # Heisenberg; delta = 0 would be XX

    opts = CBEBugOptions(;
        dt      = DT,
        n_steps = NSTEP,
        maxdim  = CHI,          # rank cap
        s_iters = -1,           # expanding-basis Lanczos -- keeps the step 2nd order in dt
        maxiter = 8,            # Krylov depth; the default 30 is ~4x more than dt=0.05 needs
        exact   = false,        # false = RSVD-CBE (the method). true = exact CBE, an A/B only
    )

    # THE TWO PATHS. Same expansion, same S-step, different way of applying H:
    #   :gate   -- Trotter/Strang over bond gates. No MPO environments. Cheapest per step.
    #   :lubich -- one projected exponential per step over a globally expanded subspace,
    #              via MPO environments. No Trotter splitting error, more work per step.
    info = if path === :gate
        RealTime.evolve!(psi, gates; opts = opts)
    else
        RealTime.evolve_mpo!(psi, xxz_chain(L; delta = 1.0); opts = opts)
    end
    return magnetisation(copy(psi)), bond_dims(psi), info
end

# ── an exact reference, because "it ran" is not "it is right" ────────────────────────
# At L=8 the Sz=0 sector is tiny, so exp(-iHt)|psi> is available exactly by sparse Krylov.
# Without this the script could print a beautiful, wrong light cone and look convincing.
include(joinpath(@__DIR__, "..", "benchmarks", "exact_sparse.jl"))

function exact_reference()
    A, states, idx = heisenberg_sparse(L)
    v = domain_wall_vector(L, states, idx)
    v = expv_sparse(A, -im * TMAX, v)
    return magnetisation_dense(v, L, states)
end

# ── report ───────────────────────────────────────────────────────────────────────────
bar(x) = (n = clamp(round(Int, (x + 0.5) * 20), 0, 20); "|" * repeat("#", n) * repeat(".", 20 - n))

@printf("Heisenberg domain wall — L=%d, dt=%.3f, t=%.1f (%d steps), chi cap %d\n",
        L, DT, TMAX, NSTEP, CHI)
println("method: RSVD-CBE bond-update BUG (gate path, Strang, expanding-basis S-step)\n")

exact = exact_reference()
results = Dict{Symbol, Any}()

for sym in SYMS
    mag, chis, info = run_domain_wall(sym)
    results[sym] = mag
    @printf("── symmetry = :%-4s ─────────────────────────────────────────────\n", sym)
    @printf("   bond dims : %s   (started all 1)\n", chis)
    @printf("   max chi   : %d over the run\n", maximum(info.max_bond_dims))
    @printf("   norm      : %.12f   (unitary; drift means truncation is biting)\n",
            info.norms[end])
    @printf("   %-4s %-9s %-9s %s\n", "site", "<Sz_j>", "exact", "")
    for j in 1:L
        @printf("   %-4d %+9.5f %+9.5f  %s\n", j, mag[j], exact[j], bar(mag[j]))
    end
    @printf("   max |error vs exact| = %.3e\n\n", maximum(abs.(mag - exact)))
end

# ── the cross-symmetry check ─────────────────────────────────────────────────────────
# THE POINT OF THE WHOLE SCRIPT. :none stores one dense block per bond; :U1 stores separate
# charge sectors. Different storage, different code paths, SAME STATE -- so these profiles
# must agree to round-off. A visible difference is a bug, not a finding.
if length(SYMS) > 1 && all(haskey(results, s) for s in (:none, :U1))
    d = maximum(abs.(results[:none] - results[:U1]))
    @printf("SAME PHYSICS CHECK   max |<Sz>_none - <Sz>_U1| = %.3e  %s\n",
            d, d < 1e-10 ? "OK — symmetry changed the storage, not the state" : "!! DIVERGED")
end

# ── the other path: Lubich sweep (MPO environments, no Trotter splitting) ────────────
# Same expansion and same S-step; H is applied through MPO environments over a globally
# expanded subspace instead of bond by bond. It should land on the same physics by a
# different route -- so it is a genuinely independent check, not a repeat.
println()
mag_l, chis_l, info_l = run_domain_wall(first(SYMS); path = :lubich)
@printf("── Lubich sweep, symmetry = :%-4s ───────────────────────────────\n", first(SYMS))
@printf("   bond dims : %s\n", chis_l)
@printf("   max |error vs exact|          = %.3e\n", maximum(abs.(mag_l - exact)))
@printf("   max |lubich - gate| (same sym) = %.3e\n",
        maximum(abs.(mag_l - results[first(SYMS)])))
println("""
   The two paths need NOT agree to round-off: the gate path carries a Trotter splitting
   error the Lubich path does not have, so a gap of order dt^2 here is expected physics,
   not a bug. Both are checked against the exact reference above, which is what settles it.""")

# ── A DIFFERENT MODEL AND A DIFFERENT SYMMETRY: TFIM with Z2 ────────────────────────
#
#     H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i
#
# The symmetry here is the global spin flip `P = Π_i X_i`, NOT a spin rotation -- so it is
# Z2, not U(1), and `set_symmetry!(:U1)` would be simply wrong for this model (Z_i Z_{i+1}
# does not conserve Sz). The switch is the same one call; only the value changes.
#
# THE BASIS IS THE TRICK. In the usual sigma^z basis P is off-diagonal and nothing is block
# sparse. In the sigma^x eigenbasis, |+> is Z2 charge 0 and |-> is charge 1, so X is diagonal
# and Z flips the charge by 1 -- and the ZZ bond term changes it by 1+1 = 0 mod 2. H conserves
# the charge exactly, which is what makes the tensors block-sparse.
#
# `<Z_j>` is identically ZERO in any Z2-symmetric state (Z changes the charge, so it has no
# diagonal element at all), so the observable here is `<X_j>` -- the transverse magnetisation.
println("\n" * "="^72)
println("TRANSVERSE-FIELD ISING, Z2 vs no symmetry   H = -J ZZ - h X")
println("="^72)

function run_tfim(sym::Symbol; J = 1.0, h = 1.0)
    set_symmetry!(sym)                                   # :Z2 or :none
    psi   = ising_kink_state(L)                          # |+...+ -...->, chi = 1
    gates = ising_bond_gates(psi; J = J, h = h)
    opts  = CBEBugOptions(; dt = DT, n_steps = NSTEP, maxdim = CHI,
                          s_iters = -1, maxiter = 8, exact = false)
    info = RealTime.evolve!(psi, gates; opts = opts)
    return x_profile(psi), bond_dims(psi), info
end

tfim = Dict{Symbol, Vector{Float64}}()
for sym in (:Z2, :none)
    mag, chis, info = run_tfim(sym)
    tfim[sym] = mag
    @printf("\n── symmetry = :%-4s ─────────────────────────────────────────────\n", sym)
    @printf("   bond dims : %s   (started all 1)\n", chis)
    @printf("   norm      : %.12f\n", info.norms[end])
    @printf("   %-4s %-10s %s\n", "site", "<X_j>", "")
    for j in 1:L
        @printf("   %-4d %+10.5f  %s\n", j, mag[j], bar(mag[j] / 2))
    end
end

d = maximum(abs.(tfim[:Z2] - tfim[:none]))
@printf("\nSAME PHYSICS CHECK   max |<X>_Z2 - <X>_none| = %.3e  %s\n",
        d, d < 1e-10 ? "OK — Z2 changed the storage, not the state" : "!! DIVERGED")
println("""
Note the Z2 bond dimensions count states per charge sector, so they ARE comparable with
:none here (unlike SU(2), where a bond dimension counts multiplets).""")
