# HIGHER-ORDER BUG BY COMPOSITE ROOT TIME STEPS -- Heisenberg SU(2) chain and cylinder.
#
#   julia -t 2 --project=. benchmarks/composite_order.jl [chain|cylinder|both]
#
# ── WHAT IS BEING COMPOSED, AND WHY ONLY THE ROOT STEP ──────────────────────────────────────────
#
# A BUG step has two parts. The half-sweeps build the augmented basis; the ROOT Galerkin `expv` is
# what actually advances `psi`. So a composition's coefficients multiply the ROOT `tau`.
#
# ⛔ COMPOSING ROOT SOLVES INSIDE ONE FROZEN BASIS IS A NO-OP. They all share the same projected
# `H_0` and therefore commute: `exp(c1 tau H0) exp(c2 tau H0) = exp((c1+c2) tau H0)`. Whatever the
# stage count, a frozen basis gives back exactly the one-shot step. A composite scheme MUST rebuild
# the basis per stage, which is why an s-stage scheme costs s sweeps and not one.
#
# ⛔ RICHARDSON IS NOT USED. It belonged to an older BUG; this version does not need it.
#
# ── ⛔ THE ORDER OF A BUG STEP CANNOT BE MEASURED IN TWO OTHERWISE-OBVIOUS PLACES ────────────────
#
#   1. NOT AT CONVERGED IMAGINARY TIME. Imaginary-time evolution forgets its path: the fixed point
#      of `exp(-H tau)` is the ground state for EVERY `dt`, so the converged energy error is set by
#      the rank and the convergence, not by the discretisation. That is exactly why the existing
#      L=16 chain campaign finds every BUG arm on the same 1e-14 floor at every grid -- that floor
#      is a REFERENCE-quality result and an ORDER-blind one. Here `tau_max` is deliberately SHORT,
#      so the state is nowhere near converged and the path still matters.
#   2. NOT AT FULL RANK. With `D` large enough to span the sector, BUG's augmented basis is complete,
#      the Galerkin projection is the identity, and the step is EXACT -- there is no discretisation
#      error left to have an order. `D` below is deliberately SMALL for the same reason a dt-order
#      test needs a rank-limited regime.
#
# ⇒ read the order off the LARGE-`dt` end of the ladder, before the curve flattens onto the
#   rank floor. A ladder that has already flattened reports order ~0 and means nothing.
#
# ── THE COMPOSITION COEFFICIENTS, AND THE ONE THEOREM THAT CONSTRAINS THEM ──────────────────────
#
# For a base method of order `p`, a composition `prod_i Psi(gamma_i tau)` gains an order when
#
#       sum_i gamma_i = 1        (consistency)      and      sum_i gamma_i^(p+1) = 0
#
# ⛔ SHENG-SUZUKI: for `p >= 1` those two cannot both hold with all `gamma_i > 0`. ANY composition
# of order > 2 therefore has a NEGATIVE stage. Consequences differ completely by time direction:
#
#   * REAL time -- a negative stage is backward unitary evolution. Harmless.
#   * IMAGINARY time -- a negative stage is `exp(+|gamma| tau H)`, which AMPLIFIES the top of the
#     spectrum and destroys the monotone decay a ground-state search relies on. It is measured here
#     rather than assumed fatal, but do not ship it for ITE without reading `maxnorm`.
#
# The triple jump below is the sharpest available probe of the base method itself: with
# `sum gamma = 1` and `sum gamma^3 = 0` it lifts a SYMMETRIC 2nd-order base to 4th order and a
# NON-symmetric one only to 3rd. So the MEASURED order tells us whether `cbe_bug_step!` is
# symmetric -- a property nothing in this repo currently records.

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
const LOG = open(joinpath(OUT, "composite_order.txt"), "a")
say(l) = (println(LOG, l); flush(LOG); println(stdout, l); flush(stdout))

# ── composition coefficient sets ────────────────────────────────────────────────────────────────
#
# `triple` is Yoshida/Suzuki's three-stage jump; `suzuki5` is the five-stage one, which spreads the
# same negative weight over two smaller backward stages (max |gamma| 0.68 against 1.70) and is the
# one to reach for if the triple jump's backward stage is what breaks an imaginary-time run.
const C_TRIPLE = let g1 = 1 / (2 - 2^(1 / 3)); [g1, -2^(1 / 3) * g1, g1] end
const C_SUZUKI5 = let g = 1 / (4 - 4^(1 / 3)); [g, g, -4^(1 / 3) * g, g, g] end
const C_HALVES  = [0.5, 0.5]        # sanity arm: must stay at the base order, never gain one

"""
    check_coeffs(name, c, p)

Assert the order conditions a coefficient set claims, so a typo in a constant is caught here rather
than showing up as a disappointing order two hours into a sweep.
"""
function check_coeffs(name, c, p)
    s1 = sum(c); sp = sum(x -> x^(p + 1), c)
    abs(s1 - 1) < 1e-12 || error("$name: sum(gamma) = $s1, must be 1")
    say(@sprintf("  %-9s stages=%d  sum=%.15f  sum^%d=%+.3e  max|g|=%.4f  %s",
                 name, length(c), s1, p + 1, sp, maximum(abs, c),
                 abs(sp) < 1e-12 ? "<- gains an order" : "(no order gain)"))
    return abs(sp) < 1e-12
end

# ── the arms ────────────────────────────────────────────────────────────────────────────────────

"""
    composite(base!, coeffs) -> (psi, tau) -> nothing

Apply `base!` once per stage with `gamma_i * tau`. Each stage rebuilds its own basis -- see the
header for why a frozen basis would make this identical to a single step.
"""
composite(base!, coeffs) = (psi, tau) -> for g in coeffs
    base!(psi, g * tau)
end

# ── the exact reference ─────────────────────────────────────────────────────────────────────────

"""
    exact_energy(L, t; imag_time) -> Float64

`<H>` of `exp(-H tau) |dimer>` (or `exp(-i H t)`), in the `S^z = 0` sector, by dense exponential.

⚠ THE OBSERVABLE IS THE ENERGY ON PURPOSE. `dense_state` and `exact_sparse` use MIRRORED site
conventions here, and a dimer start cannot catch that -- but the open Heisenberg chain is invariant
under site reversal, so its ENERGY is the same on either convention. Comparing energies sidesteps
the whole ordering question; comparing amplitudes would not.
"""
function exact_energy(L::Int, t::Float64; imag_time::Bool)
    # ⚠ `heisenberg_sparse` returns `(H, states, idx)` -- it builds the `S^z = 0` basis itself, so
    # calling `sz0_basis` separately and passing only the matrix silently drops the pairing between
    # `idx` and the dimer vector.
    Hs, states, idx = heisenberg_sparse(L)
    H = Matrix(Hs)
    v = dimer_vector(L, states, idx)
    U = imag_time ? exp(-t .* H) : exp((-im * t) .* H)
    w = U * v
    return real(dot(w, H * w) / dot(w, w))
end

# ── the ladder ──────────────────────────────────────────────────────────────────────────────────

function ladder(; L = 10, D = 8, tmax = 1.0, imag_time = true, maxiter = 20,
                dts = (0.25, 0.125, 0.0625, 0.03125), tag = "chain")
    set_symmetry!(:SU2)
    W = heisenberg_su2_mpo(L)
    psi0 = dimer_state(L)
    kw = (maxdim = D, trunc_thresh = 1e-12, maxiter = maxiter,
          exact = false, dover = 4, growth = 4.0,
          # ⛔ NOT the library defaults: `krylov_basis = 30, dex = 0` is 14x slower for the SAME
          # accuracy -- measured on the open model, and it is the sweep cost, not the root solve.
          dex = 8, krylov_basis = 3, krylov_tol = 0.0)
    base!(p, tau) = cbe_bug_step!(p, W, ComplexF64(imag_time ? -tau : -im * tau); kw...)
    mid!(p, tau)  = cbe_bug_midpoint_step!(p, W, ComplexF64(imag_time ? -tau : -im * tau); kw...)

    # ⛔ GATE THE REFERENCE PAIRING AT t = 0 BEFORE TRUSTING ANY ERROR BELOW. The MPS `dimer_state`
    # and the sparse `dimer_vector` are documented to be the same state, but if they ever were not,
    # every `|dE|` in the table would be an offset rather than a discretisation error -- and it
    # would still fall with `dt`, so the order column would look perfectly healthy while measuring
    # the wrong thing.
    e0_mps = real(mpo_energy(copy(psi0), W)) / max(norm(psi0)^2, eps())
    e0_ref = exact_energy(L, 0.0; imag_time = imag_time)
    abs(e0_mps - e0_ref) < 1e-10 ||
        error("dimer start mismatch: MPS <H> = $e0_mps, sparse <H> = $e0_ref " *
              "(difference $(abs(e0_mps - e0_ref))) -- the reference pairing is wrong")

    eref = exact_energy(L, tmax; imag_time = imag_time)
    say("\n" * "="^100)
    say(@sprintf("COMPOSITE ORDER -- Heisenberg SU(2) %s  L=%d  D=%d  %s time  T=%g",
                 tag, L, D, imag_time ? "IMAGINARY" : "REAL", tmax))
    say("="^100)
    say("  coefficient sets (base order p = 2):")
    check_coeffs("halves", C_HALVES, 2)
    check_coeffs("triple", C_TRIPLE, 2)
    check_coeffs("suzuki5", C_SUZUKI5, 2)
    say(@sprintf("  exact <H> at T = %g : %.12f      (dense exp in the S^z=0 sector)", tmax, eref))
    say("  ⚠ order is read at the LARGE-dt end; a flat tail is the D=$D rank floor, not the scheme.")

    arms = [("bug",      base!,                      1),
            ("midpoint", mid!,                       2),
            ("halves",   composite(base!, C_HALVES), 2),
            ("triple",   composite(base!, C_TRIPLE), 3),
            ("suzuki5",  composite(base!, C_SUZUKI5), 5)]

    say(@sprintf("\n  %-10s %-9s %6s %13s %8s %6s %10s %9s",
                 "scheme", "dt", "steps", "|dE|", "order", "chi", "maxnorm", "secs"))
    for (nm, st, nstage) in arms
        prev = NaN
        for dt in dts
            n = max(1, round(Int, tmax / dt))
            psi = copy(psi0); t0 = time(); mx = 0.0; ok = true
            try
                for _ in 1:n
                    st(psi, dt)
                    nz = norm(psi); mx = max(mx, nz)
                    isfinite(nz) || (ok = false; break)
                    imag_time && (psi ./= nz)
                end
            catch err
                @info "arm broke" nm dt err; ok = false
            end
            el = time() - t0
            e = NaN
            if ok
                o = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
                isfinite(o) && o != 0.0 && (e = abs(o - eref))
            end
            ord = (isfinite(prev) && isfinite(e) && e > 0) ? log2(prev / e) : NaN
            say(@sprintf("  %-10s %-9g %6d %13.4e %8s %6d %10.2e %8.2fs",
                         nm, dt, n * nstage, e,
                         isfinite(ord) ? @sprintf("%.2f", ord) : "-",
                         maximum(bond_dims(psi); init = 0), mx, el))
            isfinite(e) && (prev = e)
        end
    end
    say("\n  ⚠ `steps` counts BASE SWEEPS (n x stages), so it is the honest cost axis: a 3-stage")
    say("    scheme at dt is compared against the base at dt/3, not at dt.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    which = isempty(ARGS) ? "chain" : ARGS[1]
    L  = parse(Int,     get(ENV, "CO_L", "10"))
    D  = parse(Int,     get(ENV, "CO_D", "8"))
    T  = parse(Float64, get(ENV, "CO_T", "1.0"))
    mi = parse(Int,     get(ENV, "CO_MAXITER", "20"))
    if which in ("chain", "both")
        ladder(; L = L, D = D, tmax = T, imag_time = true,  maxiter = mi, tag = "chain")
        ladder(; L = L, D = D, tmax = T, imag_time = false, maxiter = mi, tag = "chain")
    end
end
