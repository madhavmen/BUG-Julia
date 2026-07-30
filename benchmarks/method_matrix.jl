# THE METHOD MATRIX: every arm, all three modes, against ONE exact reference.
#
#     julia --project=. benchmarks/method_matrix.jl            # L = 12, the smoke test
#     L=16 julia --project=. benchmarks/method_matrix.jl       # cluster
#
# ARMS
#   time evolution (real and imaginary)
#     rsvd_cbe_bond_update  gate     BUG, odd/even Trotter, identity environments
#     rsvd_cbe_bond_update  lubich   BUG, one projected exponential on env-dressed H
#     tdvp2                          2-site TDVP. NO discarded projectors, by design
#     1site_tdvp_cbe_rsvd            1-site TDVP + sketched CBE
#     1site_tdvp_cbe                 1-site TDVP + exact CBE (forms the rank-4 object)
#   ground state
#     rsvd_cbe_dmrg                  sketched CBE + expanding-Krylov local eigensolve
#     cbe_dmrg                       exact CBE, same solve
#
# THE REFERENCE IS EXACT DIAGONALISATION, not another integrator, and at L = 12 that is
# affordable: the Sz = 0 sector of a 12-site chain is C(12,6) = 924 states, so both the
# real-time propagator and the ground state come from one dense eigendecomposition. That
# matters because every previous comparison in this work that used an integrator as its own
# reference measured agreement rather than correctness.
#
# The reference degrades honestly rather than silently: past `MAXDENSE` basis states the dense
# route is abandoned and 2-site TDVP stands in, which the output LABELS as such. A number
# compared against tdvp2 is a consistency check, not an accuracy measurement, and the two must
# not be read the same way.
#
# HEISENBERG (delta = 1) throughout, since that is the point the SU(2) arm will eventually have
# to reproduce -- delta != 1 breaks SU(2) down to U(1) and cannot be part of that comparison.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime, ImaginaryTime, GroundState
using LinearAlgebra, Printf
using Telum: to_concrete      # normalising a state rescales the centre tensor in place

set_symmetry!(:U1)

const L        = parse(Int, get(ENV, "L", "12"))
const DELTA    = 1.0
const DT       = parse(Float64, get(ENV, "DT", "0.02"))
const TMAX     = parse(Float64, get(ENV, "TMAX", "0.4"))
const BETA     = parse(Float64, get(ENV, "BETA", "20.0"))
const MAXDIM   = parse(Int, get(ENV, "MAXDIM", "64"))
const TRUNC    = 1e-12
const MAXDENSE = 20_000

nsteps(t, dt) = round(Int, t / dt)

hb_gates(p) = Any[heisenberg_bond_gate(p[i].inds[2], p[i + 1].inds[2]; delta = DELTA)
                  for i in 1:(length(p) - 1)]

# ── exact reference in the Sz = 0 sector ─────────────────────────────────────────────
"Dense H in the half-filling sector, plus the basis, so a start state can be embedded."
function dense_sector(L::Int; delta = DELTA, J = 1.0)
    sz(s) = s == 1 ? 0.5 : -0.5
    bit(b, j) = (b >> (j - 1)) & 1
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    idx = Dict(s => k for (k, s) in enumerate(states))
    H = zeros(Float64, length(states), length(states))
    for (k, b) in enumerate(states), j in 1:(L - 1)
        sj, sj1 = bit(b, j), bit(b, j + 1)
        H[k, k] += J * delta * sz(sj) * sz(sj1)
        sj == sj1 || (H[idx[b ⊻ (1 << (j - 1)) ⊻ (1 << j)], k] += J / 2)
    end
    return Symmetric(H), states, idx
end

# Diagonalise ONCE and derive every reference from it, so a profile at a second time costs a
# phase multiply rather than another eigensolve. Needed because the fixed-rank table below
# evaluates at a LATER time than the headline table.
const NBASIS = binomial(L, L ÷ 2)
const HAVE_EXACT = NBASIS <= MAXDENSE

struct ExactRef
    vals::Vector{Float64}
    vecs::Matrix{Float64}
    states::Vector{Int}
    coef0::Vector{ComplexF64}     # Neel start in the eigenbasis
end

function build_exact_ref(L::Int)
    H, states, idx = dense_sector(L)
    F = eigen(H)
    neel = sum((isodd(j) ? 1 : 0) << (j - 1) for j in 1:L)
    v = zeros(ComplexF64, length(states)); v[idx[neel]] = 1.0
    return ExactRef(F.values, F.vectors, states, F.vectors' * v)
end

"Exact <Sz_j>(t) for the Neel start."
function exact_profile(R::ExactRef, t::Real)
    vt = R.vecs * (cis.(-R.vals * t) .* R.coef0)
    p = abs2.(vt)
    return [sum(p[k] * (((R.states[k] >> (j - 1)) & 1) - 0.5) for k in eachindex(R.states))
            for j in 1:L]
end

@printf("L = %d, delta = %.1f, dt = %.3f, t = %.2f, beta = %.1f, maxdim = %d\n",
        L, DELTA, DT, TMAX, BETA, MAXDIM)
@printf("half-filling sector: %d states -> reference is %s\n\n", NBASIS,
        HAVE_EXACT ? "EXACT diagonalisation" : "2-site TDVP (CONSISTENCY ONLY, not accuracy)")

const EREF = HAVE_EXACT ? build_exact_ref(L) : nothing
const REF_PROF = HAVE_EXACT ? exact_profile(EREF, TMAX) : Float64[]
const REF_E0 = HAVE_EXACT ? EREF.vals[1] : NaN

# ── the arms ─────────────────────────────────────────────────────────────────────────
h = xxz_chain(L; delta = DELTA)

"Loop a bare stepper (tdvp2 / tdvp1_cbe) and report the profile plus the widest bond."
function loop_stepper(step!, tau; nst, normalise::Bool)
    psi = neel_state(L)
    mb = 0
    for _ in 1:nst
        info = step!(psi, h, tau)
        mb = max(mb, info.max_bond)
        if normalise
            n = norm(psi)
            n > 0 && (psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center]))
        end
    end
    return psi, mb
end

tdvp2 = (psi, hh, tau) -> tdvp2_step!(psi, hh, tau; maxdim = MAXDIM, trunc_thresh = TRUNC)
tdvp1r = (psi, hh, tau) -> tdvp1_cbe_step!(psi, hh, tau; maxdim = MAXDIM,
                                           trunc_thresh = TRUNC, exact = false)
tdvp1e = (psi, hh, tau) -> tdvp1_cbe_step!(psi, hh, tau; maxdim = MAXDIM,
                                           trunc_thresh = TRUNC, exact = true)

opts(; kw...) = CBEBugOptions(; dt = DT, maxdim = MAXDIM, trunc_thresh = TRUNC,
                             s_iters = -1, kw...)

# ── REAL TIME ────────────────────────────────────────────────────────────────────────
#
# READ THE `maxbd` COLUMN BEFORE THE `err` COLUMN. If every arm's widest bond is well under
# `MAXDIM`, the cap never engaged, nobody truncated anything, and the errors below are all the
# same number -- each arm's floor, not a ranking. That is exactly what L=12, t=0.4, maxdim=48
# produces: maxbd 20-22 against a cap of 48, and the three TDVP arms agreeing to six digits
# (6.989474e-09 / 6.989480e-09 / 6.989867e-09). Six-digit agreement between a 2-site block
# evolve and a 1-site+0-site splitting is not a coincidence, it is a saturated test.
#
# This table is therefore a CORRECTNESS check -- it is what caught the boundary bug, which
# showed up here as 5.73e-02 against 6.99e-09. It is NOT an accuracy comparison. The
# fixed-rank table below is the one that ranks the arms, because there the cap binds.
println("REAL TIME -- <Sz_j> profile error at t = $TMAX")
println("  (correctness check: cap $MAXDIM vs maxbd ~20, so this SATURATES -- see the")
println("   fixed-rank table for the actual ranking)")
@printf("  %-30s %-14s %-8s %s\n", "arm", "err", "maxbd", "note")
println("  " * "-"^70)
flush(stdout)

prof_err(psi) = HAVE_EXACT ? norm(magnetisation(copy(psi)) - REF_PROF) : NaN

let nst = nsteps(TMAX, DT), tau = ComplexF64(-im * DT)
    p, _ = loop_stepper(tdvp2, tau; nst = nst, normalise = false)
    ref_tdvp2 = magnetisation(copy(p))
    err_of(psi) = HAVE_EXACT ? norm(magnetisation(copy(psi)) - REF_PROF) :
                               norm(magnetisation(copy(psi)) - ref_tdvp2)

    @printf("  %-30s %-14.6e %-8d %s\n", "tdvp2", err_of(p),
            maximum(bond_dims(p)), HAVE_EXACT ? "" : "IS the reference")
    flush(stdout)

    for (name, st) in (("1site_tdvp_cbe_rsvd", tdvp1r), ("1site_tdvp_cbe (exact)", tdvp1e))
        q, mb = loop_stepper(st, tau; nst = nst, normalise = false)
        @printf("  %-30s %-14.6e %-8d\n", name, err_of(q), mb)
        flush(stdout)
    end

    g = neel_state(L)
    info = RealTime.evolve!(g, hb_gates(g); opts = opts(n_steps = nst))
    @printf("  %-30s %-14.6e %-8d %s\n", "rsvd_cbe_bond_update gate", err_of(g),
            maximum(info.max_bond_dims), "Trotter")
    flush(stdout)

    m = neel_state(L)
    info = RealTime.evolve_mpo!(m, h; opts = opts(n_steps = nst))
    @printf("  %-30s %-14.6e %-8d %s\n", "rsvd_cbe_bond_update lubich", err_of(m),
            maximum(info.max_bond_dims), "no splitting")
    flush(stdout)
end

# ── REAL TIME AT FIXED RANK -- THE DISCRIMINATING TABLE ──────────────────────────────
#
# A rank-adaptive integrator can only be judged where rank is SCARCE. Give every arm the same
# hard cap and evolve long enough that the exact state needs more than the cap allows, then the
# question each arm is answering is the real one: given `chi` numbers to spend, which
# directions do you keep? At an ample cap they all keep everything and the comparison is empty.
#
# `t = TDISC` is deliberately past the headline `TMAX`: at t = 0.4 from a Neel state the exact
# state still fits in bond 20, so even a cap of 16 barely bites. Later, entanglement has grown
# and the caps genuinely compete.
const TDISC = parse(Float64, get(ENV, "TDISC", "2.0"))
const CAPS  = [parse(Int, s) for s in split(get(ENV, "CAPS", "4,8,16"), ",")]

if HAVE_EXACT
    ref_disc = exact_profile(EREF, TDISC)
    nst = nsteps(TDISC, DT)
    tau = ComplexF64(-im * DT)
    @printf("\nREAL TIME AT FIXED RANK -- <Sz_j> error at t = %.2f, cap BINDS\n", TDISC)
    @printf("  %-30s %s\n", "arm", join([@sprintf("%-13s", "chi=$c") for c in CAPS]))
    println("  " * "-"^70)
    flush(stdout)

    err_at(psi) = norm(magnetisation(copy(psi)) - ref_disc)

    # Bare steppers, parameterised by cap.
    for (name, mk) in (("tdvp2", cap -> ((p, hh, t) -> tdvp2_step!(p, hh, t;
                            maxdim = cap, trunc_thresh = TRUNC))),
                       ("1site_tdvp_cbe_rsvd", cap -> ((p, hh, t) -> tdvp1_cbe_step!(p, hh, t;
                            maxdim = cap, trunc_thresh = TRUNC, exact = false))),
                       ("1site_tdvp_cbe (exact)", cap -> ((p, hh, t) -> tdvp1_cbe_step!(p, hh, t;
                            maxdim = cap, trunc_thresh = TRUNC, exact = true))))
        cells = String[]
        for cap in CAPS
            q, _ = loop_stepper(mk(cap), tau; nst = nst, normalise = false)
            push!(cells, @sprintf("%-13.4e", err_at(q)))
        end
        @printf("  %-30s %s\n", name, join(cells))
        flush(stdout)
    end

    # The BUG arms, both paths.
    for (name, run) in (("rsvd_cbe_bond_update gate",
                         (cap, n) -> begin
                             g = neel_state(L)
                             RealTime.evolve!(g, hb_gates(g);
                                 opts = CBEBugOptions(; dt = DT, n_steps = n, maxdim = cap,
                                     trunc_thresh = TRUNC, s_iters = -1))
                             g
                         end),
                        ("rsvd_cbe_bond_update lubich",
                         (cap, n) -> begin
                             m = neel_state(L)
                             RealTime.evolve_mpo!(m, h;
                                 opts = CBEBugOptions(; dt = DT, n_steps = n, maxdim = cap,
                                     trunc_thresh = TRUNC, s_iters = -1))
                             m
                         end))
        cells = String[]
        for cap in CAPS
            push!(cells, @sprintf("%-13.4e", err_at(run(cap, nst))))
        end
        @printf("  %-30s %s\n", name, join(cells))
        flush(stdout)
    end
end

# ── IMAGINARY TIME ───────────────────────────────────────────────────────────────────
@printf("\nIMAGINARY TIME -- E at beta = %.1f, exact E0 = %s\n", BETA,
        HAVE_EXACT ? @sprintf("%.10f", REF_E0) : "unavailable")
@printf("  %-30s %-16s %-13s %s\n", "arm", "energy", "E - E0", "maxbd")
println("  " * "-"^70)
flush(stdout)

let nst = nsteps(BETA, DT * 2.5), dtb = DT * 2.5, tau = ComplexF64(-DT * 2.5)
    gates_ref = hb_gates(neel_state(L))
    for (name, st) in (("1site_tdvp_cbe_rsvd", tdvp1r), ("1site_tdvp_cbe (exact)", tdvp1e))
        psi, mb = loop_stepper(st, tau; nst = nst, normalise = true)
        e = real(energy(psi, gates_ref))
        @printf("  %-30s %-16.10f %-13s %d\n", name, e,
                HAVE_EXACT ? @sprintf("%.3e", e - REF_E0) : "-", mb)
        flush(stdout)
    end

    g = neel_state(L)
    info = ImaginaryTime.evolve!(g, hb_gates(g);
                                 opts = opts(dt = dtb, n_steps = nst))
    e = info.energies[end]
    @printf("  %-30s %-16.10f %-13s %d\n", "rsvd_cbe_bond_update gate", e,
            HAVE_EXACT ? @sprintf("%.3e", e - REF_E0) : "-",
            maximum(info.max_bond_dims))
    flush(stdout)
end

# ── GROUND STATE ─────────────────────────────────────────────────────────────────────
println("\nGROUND STATE -- E - E0 MUST be >= 0 (the local solve is variational)")
@printf("  %-30s %-16s %-13s %-8s %-7s %s\n",
        "arm", "energy", "E - E0", "sweeps", "maxbd", "matvec")
println("  " * "-"^70)
flush(stdout)
for (name, f) in (("rsvd_cbe_dmrg", GroundState.rsvd_cbe_dmrg!),
                  ("cbe_dmrg (exact)", GroundState.cbe_dmrg!))
    p = neel_state(L)
    info = f(p, h; opts = CBEBugOptions(maxdim = MAXDIM, trunc_thresh = TRUNC),
             n_sweeps = 40, etol = 1e-12)
    e = info.energies[end]
    @printf("  %-30s %-16.10f %-13s %-8d %-7d %d\n", name, e,
            HAVE_EXACT ? @sprintf("%.3e", e - REF_E0) : "-",
            length(info.energies), maximum(info.max_bond_dims), sum(info.krylov_dims))
    flush(stdout)
end

println("\nMATRIX_DONE")
