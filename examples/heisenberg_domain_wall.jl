# RSVD-CBE BUG in 40 lines: an L=8 XXZ domain wall, with and without symmetry.
#
#   julia --project=. examples/heisenberg_domain_wall.jl
#   L=10 TMAX=3 SYMS=none,U1 julia --project=. examples/heisenberg_domain_wall.jl
#   L=18 TMAX=20 CHI=64 DELTA=0 julia --project=. examples/heisenberg_domain_wall.jl
#
# WHAT THIS SHOWS
#
#   1. how to switch symmetry           -- one call, `set_symmetry!(:none | :U1 | :SU2)`
#   2. how to run the default method    -- `RealTime.evolve!(psi, gates; opts)`
#   3. that symmetry changes the STORAGE, not the physics
#   4. that the answer is right         -- checked against the CLOSED-FORM solution
#
# THE PHYSICS. A domain wall |↑↑↑↑↓↓↓↓> is not an eigenstate of the XXZ chain, so it melts:
# <Sz_j> spreads from the interface outward at a finite velocity. That spreading front is the
# light cone, and it is what the printout below draws.
#
# WHY `DELTA` DEFAULTS TO 0 (XX) AND NOT 1 (HEISENBERG). The reference. At `delta = 0` the chain
# is free-fermion integrable, so `<Sz_j(t)>` is known in CLOSED FORM -- an L x L matrix
# exponential, `xx_free_fermion_sz`, exact at ANY `L` and ANY `t` with no convergence parameter
# to get wrong. At `delta != 0` there is no such formula for this observable, so the only
# reference available is numerical (sparse Krylov in the Sz=0 sector), which is bounded by
# `L <~ 20` and, being iterative, can itself be under-converged. Running Heisenberg is one env
# var away and still checked -- but the DEFAULT is the case where the yardstick cannot lie,
# because a wrong reference is reported as an error in the method.
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
const DELTA = parse(Float64, get(ENV, "DELTA", "0.0"))   # 0 = XX (closed form), 1 = Heisenberg
const SYMS  = [Symbol(s) for s in split(get(ENV, "SYMS", "none,U1"), ",")]
const NSTEP = round(Int, TMAX / DT)

# Which reference is even available is a property of `DELTA`, not a choice -- see the header.
const ANALYTIC = (DELTA == 0.0)

# ── the run ──────────────────────────────────────────────────────────────────────────
"""
Evolve a domain wall with the RSVD-CBE bond-update BUG and return `(magnetisation, chi, info)`.

This is the whole API surface: pick a symmetry, build a state, build one gate per bond,
call `evolve!`. Everything else below is printing.
"""
function run_domain_wall(sym::Symbol; path::Symbol = :gate)
    set_symmetry!(sym)                                   # <-- THE SYMMETRY SWITCH

    psi   = domain_wall_state(L)                         # |↑↑↑↑↓↓↓↓>, chi = 1
    gates = bond_gates(psi; J = 1.0, delta = DELTA)      # delta = 0 is XX, delta = 1 Heisenberg

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
        RealTime.evolve_mpo!(psi, xxz_chain(L; delta = DELTA); opts = opts)
    end
    return magnetisation(copy(psi)), bond_dims(psi), info
end

# ── an exact reference, because "it ran" is not "it is right" ────────────────────────
# Without this the script could print a beautiful, wrong light cone and look entirely
# convincing: a melting domain wall is smooth and antisymmetric about the interface whether or
# not the numbers are right, so the eye cannot referee it.
#
# THE CLOSED FORM (delta = 0). Jordan-Wigner maps the XX chain to free fermions, so the whole
# problem collapses to the L x L single-particle hopping matrix:
#
#     <Sz_j(t)> = sum_{k occupied} |[exp(-i h t)]_{jk}|^2 - 1/2,     h_{j,j+1} = J/2
#
# That is O(L^3) and EXACT -- no Krylov space, no time step, no tolerance, nothing that can be
# set too loose. It is valid at any L and any t, which is what makes the L=18/t=20 regime
# checkable at all.
#
# THE FALLBACK (delta != 0). No closed form for <Sz_j(t)>, so the reference is sparse Krylov in
# the Sz=0 sector -- correct but bounded by C(L, L/2), and iterative, so it has convergence
# parameters of its own. `expv_sparse` substeps from ||H||_inf and checks a Saad residual
# precisely so that it cannot quietly return an under-converged vector; a reference that cannot
# detect its own failure is worse than no reference, because its error is charged to the method.
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))
ANALYTIC || include(joinpath(@__DIR__, "..", "benchmarks", "exact_sparse.jl"))

function exact_reference()
    # `xx_free_fermion_sz` defaults to `occupied = 1:(L÷2)`, which IS `domain_wall_state(L)`.
    # Closed form: nothing to converge, so nothing to check.
    ANALYTIC && return xx_free_fermion_sz(L, TMAX; J = 1.0)

    A, states, idx = heisenberg_sparse(L; delta = DELTA)
    v0 = domain_wall_vector(L, states, idx)

    # SELF-CHECK, because this path is iterative and its failure mode is silent. Walk the same
    # interval on two different substep schedules: one call, and 400 steps. Both are exact if
    # converged, so a disagreement means the reference is NOT converged -- and an under-converged
    # reference is reported as method error, which is exactly how this example once turned a
    # 1.08e-04 result into a 7.21e-02 one. Refuse to hand back a number we cannot vouch for.
    one = magnetisation_dense(expv_sparse(A, -im * TMAX, v0), L, states)
    walked = magnetisation_dense(
        propagate!(A, v0, ComplexF64(-im * TMAX / 400), 400, (s, v) -> nothing), L, states)
    d = maximum(abs.(one - walked))
    d < 1e-9 || error("""
        sparse reference is NOT converged: two substep schedules disagree by $d.
        Do not trust any error reported against it. Raise `m` or lower the substep size in
        `expv_sparse` (benchmarks/exact_sparse.jl).""")
    return one
end

# ── report ───────────────────────────────────────────────────────────────────────────
bar(x) = (n = clamp(round(Int, (x + 0.5) * 20), 0, 20); "|" * repeat("#", n) * repeat(".", 20 - n))

@printf("%s domain wall — L=%d, delta=%.2f, dt=%.3f, t=%.1f (%d steps), chi cap %d\n",
        ANALYTIC ? "XX" : "XXZ", L, DELTA, DT, TMAX, NSTEP, CHI)
println("method:    RSVD-CBE bond-update BUG (gate path, Strang, expanding-basis S-step)")
println(ANALYTIC ?
        "reference: xx_free_fermion_sz — CLOSED FORM, exact at any L and t" :
        "reference: sparse Krylov in the Sz=0 sector — no closed form exists at delta != 0\n" *
        "           (substepped and residual-checked; needs C(L,L/2) memory, so L <~ 20)")
println()

exact = exact_reference()
results = Dict{Symbol, Any}()

capped = Dict{Symbol, Bool}()
errs   = Dict{Symbol, Float64}()

for sym in SYMS
    mag, chis, info = run_domain_wall(sym)
    results[sym] = mag
    capped[sym] = maximum(info.max_bond_dims) >= CHI
    errs[sym]   = maximum(abs.(mag - exact))
    @printf("── symmetry = :%-4s ─────────────────────────────────────────────\n", sym)
    @printf("   bond dims : %s   (started all 1)\n", chis)
    @printf("   max chi   : %d over the run\n", maximum(info.max_bond_dims))
    @printf("   norm      : %.12f   (unitary; drift means truncation is biting)\n",
            info.norms[end])
    @printf("   %-4s %-9s %-9s %s\n", "site", "<Sz_j>", "exact", "")
    for j in 1:L
        @printf("   %-4d %+9.5f %+9.5f  %s\n", j, mag[j], exact[j], bar(mag[j]))
    end
    # Say WHICH error this is. Once the bond saturates `CHI` the number is dominated by discarded
    # Schmidt weight, so it is a statement about the budget and not about the integrator -- and
    # raising CHI is the response, not a bug hunt.
    @printf("   max |error vs exact| = %.3e   %s\n\n", errs[sym],
            capped[sym] ? "<- TRUNCATION-LIMITED (bond pinned at the cap $CHI): this is the \
                           rank budget, not the method. Raise CHI." :
                          "(bond never hit the cap, so this is the method's own error)")
end

# ── the cross-symmetry check ─────────────────────────────────────────────────────────
# THE POINT OF THE WHOLE SCRIPT. :none stores one dense block per bond; :U1 stores separate
# charge sectors. Different storage, different code paths, SAME STATE.
#
# WHY THIS IS NOT A ROUND-OFF CHECK, THOUGH. "Same state" holds for the EXACT states, and neither
# run computes the exact state. Both are approximations, and the two approximate DIFFERENTLY:
#
#   - while the rank is growing, `:none` selects expansion directions from one dense block and
#     `:U1` from a block-diagonal one per charge sector, so they span different subspaces;
#   - once the rank saturates `maxdim`, `:none` truncates by one dense SVD per bond and `:U1`
#     per charge sector, so they discard different directions.
#
# Neither is a defect, and neither is switched off by `maxdim` failing to bind -- the state starts
# at chi=1, so what limits its rank mid-run is the growth schedule and the selection, not the cap.
# An earlier version asserted `< 1e-10` whenever the cap did not bind and duly fired on a healthy
# L=8 run; measurement (verify_symmetry_gap.jl) showed the gap falling as O(dt) in lockstep with
# both runs' own errors, i.e. approximation error, converging to zero. So the criterion below is
# RELATIVE to the accuracy the two runs actually achieved, which is the only thing the triangle
# inequality lets us assert.
if length(SYMS) > 1 && all(haskey(results, s) for s in (:none, :U1))
    d  = maximum(abs.(results[:none] - results[:U1]))
    e  = max(errs[:none], errs[:U1])
    @printf("SAME PHYSICS CHECK   max |<Sz>_none - <Sz>_U1| = %.3e\n", d)
    @printf("                     each run's own error      = %.3e (none), %.3e (U1)\n",
            errs[:none], errs[:U1])
    # THE CRITERION. Two approximations of the same state, each off by ~e, may differ from EACH
    # OTHER by up to ~2e without either being wrong -- that is the triangle inequality, not a
    # defect. So the gap is only evidence of a storage bug when it EXCEEDS what the two runs'
    # own accuracy already allows. A fixed threshold cannot express that, which is why the old
    # `< 1e-10` fired on a perfectly healthy L=8 run.
    if d <= 3 * e
        @printf("                     => CONSISTENT: the gap is within the runs' own accuracy \
                 (%.1fx e). Symmetry\n                        changed the storage, not the \
                 state. Refine dt and all three shrink together.\n", d / max(e, eps()))
    else
        @printf("                     => !! SUSPECT: the gap (%.1fx e) is larger than either \
                 run's error, so it\n                        cannot be explained by their \
                 accuracy. Investigate.\n", d / max(e, eps()))
    end
    # The gap being ~e says the two are mutually consistent; it does NOT say either is accurate.
    # That is what the per-symmetry `error vs exact` lines above are for.
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
