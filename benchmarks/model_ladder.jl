# THE MODEL LADDER (Jan's point 7): Heisenberg U(1) -> SU(2) -> Haldane-Shastry.
#
# ⛔ ARM A IS A STANDING RED TEST, NOT A NEW EXPERIMENT. `tests/sweeps/test_physics.jl:153` has
# been failing on `cbe_bug` for the Haldane-Shastry magnon since before this work started --
# `max|dSz| = 4.25e-03` against a threshold of `1e-4`, while `tdvp_cbe1s` passes the SAME case at
# `2.60e-08`. That was recorded as "not caused by any cbe_bug change" and left open.
#
# It has never been scanned against the KRYLOV DEPTH, because until 2026-08-24 that knob was
# inert (thresholding `beta` never fired). HS is long-ranged, and the one model class where depth
# was already measured to matter is exactly the one where `tau*||H_loc||` is not small. So the
# question this arm asks is: IS THE STANDING FAILURE A DEPTH PROBLEM?
#
# ⚠ THE MAGNON SECTOR IS THE ONLY ANALYTIC REAL-TIME REFERENCE HERE. Neither SU(2) XXX from a
# generic state nor Haldane-Shastry has a closed form, but for ANY translation-invariant coupling
# the one-magnon sector does: a single down spin among ups evolves by the Fourier solution
# `magnon_amplitudes`, and `<S^z_j> = 1/2 - |a_j|^2` exactly. Everything else on this ladder is
# graded on conservation laws, which is weaker and is labelled as such.
#
# ⛔ SU(2) TRAPS, BOTH LOAD-BEARING (arm B):
#   1. `maxdim` COUNTS MULTIPLETS under a non-abelian symmetry, so a cross-symmetry rank
#      comparison MUST index by `state_bond_dims` (states) and never by `maxdim` or `bond_dims`.
#      Comparing a multiplet count against a state count makes SU(2) look free.
#   2. NEEL IS NOT SU(2)-REPRESENTABLE -- a definite-`Sz` product state spans many total-spin
#      sectors. The start has to be `dimer_state`, and the U(1) arm has to be RE-RUN on that same
#      start or the two are not the same physics and the comparison is meaningless.
#
# Run:  julia --project=. benchmarks/model_ladder.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "analytic_reference.jl"))

const OUT = joinpath(@__DIR__, "results", "model_ladder.txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

# ── ARM A: does Krylov depth move the standing Haldane-Shastry magnon failure? ─────────────
say("="^96)
say("ARM A  Haldane-Shastry magnon, L=8, T=0.2  --  the STANDING RED TEST at 4.25e-03")
say("        threshold 1e-4;  tdvp_cbe1s passes the same case at 2.60e-08")
say("="^96)
set_symmetry!(:U1)
let L = 8, dt = 0.01, nst = 20
    mpo = haldane_shastry_mpo(L)
    Jr  = hs_couplings_analytic(L)
    c   = ComplexF64[cis(-2 * pi * n / L) / sqrt(L) for n in 0:(L - 1)]
    T   = dt * nst
    a   = magnon_amplitudes(c, T, Jr; L = L)
    want = [0.5 - abs2(a[j]) for j in 1:L]

    say(@sprintf("  %-34s %13s %8s %9s", "arm", "max|dSz|", "chi", "krylov"))
    # `tdvp_cbe1s` first: it is the arm that already passes, so it is the bar.
    let psi = product_state([j == 1 ? :down : :up for j in 1:L]), kry = 0
        for _ in 1:nst
            tdvp_cbe1s_step!(psi, mpo, ComplexF64(-im * dt); maxdim = 64, trunc_thresh = 1e-14)
            renorm!(psi)
        end
        e = maximum(abs.(magnetisation(copy(psi)) .- want))
        say(@sprintf("  %-34s %13.4e %8d %9s", "tdvp_cbe1s (the bar)", e,
                     maximum(bond_dims(psi)), "-"))
    end
    for (nm, kw) in (("cbe_bug DEFAULTS (the failure)", (;)),
                     ("cbe_bug krylov_tol=0 (to cap)", (krylov_tol = 0.0,)),
                     ("cbe_bug m=2 pinned",            (krylov_basis = 2, krylov_tol = 0.0)),
                     ("cbe_bug m=3 pinned",            (krylov_basis = 3, krylov_tol = 0.0)),
                     ("cbe_bug m=8 pinned",            (krylov_basis = 8, krylov_tol = 0.0)),
                     ("cbe_bug m=16 pinned",           (krylov_basis = 16, krylov_tol = 0.0)))
        psi, kry = product_state([j == 1 ? :down : :up for j in 1:L]), 0
        for _ in 1:nst
            info = cbe_bug_step!(psi, mpo, ComplexF64(-im * dt);
                                 maxdim = 64, trunc_thresh = 1e-14, kw...)
            kry += info.krylov_dims
            renorm!(psi)
        end
        e = maximum(abs.(magnetisation(copy(psi)) .- want))
        say(@sprintf("  %-34s %13.4e %8d %9d%s", nm, e, maximum(bond_dims(psi)), kry,
                     e < 1e-4 ? "   <- PASSES" : ""))
    end
end

# ── ARM B: SU(2) reproduces U(1), and what the symmetry actually costs ────────────────────
say("")
say("="^96)
say("ARM B  SU(2) vs U(1) on the SAME start (dimer), L=8  --  rank indexed by STATES")
say("        `maxdim` counts MULTIPLETS under SU(2); comparing that to a state count is the trap")
say("="^96)
let L = 8, dt = 0.05, nst = 10
    say(@sprintf("  %-22s %15s %15s %11s %11s", "symmetry", "E(0)", "E(T)", "max chi", "max states"))
    for sym in (:U1, :SU2)
        set_symmetry!(sym)
        h   = sym === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; J = 1.0, delta = 1.0)
        mpo = RSVDCBEBondUpdate.mpo_from_terms(h)
        psi = dimer_state(L)                      # the ONLY SU(2)-representable S=0 start
        e0  = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
        for _ in 1:nst
            cbe_bug_step!(psi, mpo, ComplexF64(-im * dt); maxdim = 64, trunc_thresh = 1e-12)
        end
        e1 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
        say(@sprintf("  %-22s %15.10f %15.10f %11d %11d", string(sym), e0, e1,
                     maximum(bond_dims(psi)), maximum(state_bond_dims(psi))))
    end
    say("  (dimer is an S=0 eigenstate of neither model, so E(T) == E(0) is the CONSERVATION")
    say("   check, and the two symmetries must agree on both columns to be the same physics.)")
end

# ── ARM C: Haldane-Shastry conservation over a longer window ──────────────────────────────
say("")
say("="^96)
say("ARM C  Haldane-Shastry, L=12, energy drift to T=1.0  (no closed form -- conservation only)")
say("="^96)
set_symmetry!(:U1)
let L = 12, dt = 0.05, nst = 20
    mpo = haldane_shastry_mpo(L)
    say(@sprintf("  %-30s %13s %8s %9s", "arm", "|dE|", "chi", "krylov"))
    for (nm, kw) in (("cbe_bug DEFAULTS", (;)),
                     ("cbe_bug m=3 pinned", (krylov_basis = 3, krylov_tol = 0.0)),
                     ("cbe_bug m=16 pinned", (krylov_basis = 16, krylov_tol = 0.0)))
        psi, kry = neel_state(L), 0
        e0 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
        for _ in 1:nst
            info = cbe_bug_step!(psi, mpo, ComplexF64(-im * dt);
                                 maxdim = 64, trunc_thresh = 1e-12, kw...)
            kry += info.krylov_dims
        end
        e1 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
        say(@sprintf("  %-30s %13.3e %8d %9d", nm, abs(e1 - e0),
                     maximum(bond_dims(psi)), kry))
    end
end
set_symmetry!(:U1)
close(IO_)
