# NON-UNITARY EVOLUTION: WHERE TDVP'S BACKWARD STEP TURNS INTO AN AMPLIFIER.
#
# ⛔ WHAT THIS FILE IS: MIXED REAL/IMAGINARY TIME, NOT A LINDBLAD BATH. With a REAL `H`,
# `tau = -(eta + i)*dt` gives `exp(-eta*dt*H)` alongside the rotation -- imaginary-time evolution,
# which AMPLIFIES negative-energy states rather than damping them. MEASURED: the norm GROWS,
# |psi| = 1.101 at eta = 0.05. That is a legitimate non-unitary stress test of the backward step,
# and it is NOT an open quantum system. A true bath needs a NON-HERMITIAN `H_eff = H - i*Gamma*O`,
# which needs complex MPO coefficients -- `XXZTerm.coeff` is `Float64`, so that is a separate
# change and is where the Schwinger/Lindblad work goes.
#
# ⚠ AND UNIFORM SINGLE-SITE LOSS IS TRIVIAL UNDER U(1). `sum_j n_j` is CONSERVED, so
# `-i*Gamma*sum_j n_j` is a constant times the identity: a global decay factor and no dynamics at
# all. A meaningful dissipator has to be staggered (the paper's `O(n) = (-1)^n (Z_n+1)/2`) or
# two-body.
#
# ⛔ THE MECHANISM, STATED BEFORE MEASURING SO THE RESULT CAN CONTRADICT IT.
#
# TDVP is a PROJECTOR SPLITTING: it takes a FORWARD step on the site tensor and a BACKWARD step
# on the bond, `+tau/2` then `-tau/2`. Under unitary evolution `tau = -i*dt` both are pure
# rotations and the backward step is harmless. Drive the same sweep with
#
#     tau = -(eta + i) * dt          eta > 0
#
# -- the no-jump trajectory evolution of an open system, damping at rate `eta` -- and the
# backward step becomes `+(eta + i)*dt/2`, whose REAL PART IS POSITIVE. It evolves
# `exp(+eta*dt*H/2)`, which AMPLIFIES precisely the high-energy modes the forward step just
# damped, by a factor `exp(+eta*dt*E_max/2)` per step. That compounds.
#
# `cbe_bug` has no backward step. Its half-sweeps build BASES and evolve nothing; the single
# Galerkin solve at the root is forward, over the full `tau`. So there is nothing in it to
# amplify, and the prediction is that it stays finite where TDVP does not.
#
# ⚠ THIS IS A PREDICTION, NOT A RESULT, UNTIL THE TABLE BELOW SAYS SO. The honest failure modes:
# TDVP might survive at these `eta` and `dt` (the amplification is bounded by `exp(eta*dt*E_max/2)`
# and could simply be small), or `cbe_bug` might blow up for an unrelated reason. Both are
# reported as they come.
#
# ⚠ AND THE NORM DECAYING IS NOT A BLOW-UP. Non-unitary evolution is SUPPOSED to lose norm --
# that is the physics, not a defect. What marks a failure is the norm or the energy DIVERGING, or
# going non-finite. The table therefore tracks the norm, the (real part of the) energy, and an
# explicit finite/non-finite verdict, rather than calling any norm change instability.
#
# ⚠ THE REFERENCE IS EXACT. `expv_sparse` takes a complex `t`, so the same non-unitary propagator
# is available in the Sz = 0 sector and the comparison is against the truth rather than between
# schemes -- which matters here because a scheme that blows up would otherwise "disagree" with a
# scheme that also blew up.
#
# Run:  julia --project=. benchmarks/nonunitary_blowup.jl [model] [etas] [t_max]
#         model = chain | square | kagome | all

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const WHICH = length(ARGS) >= 1 ? ARGS[1] : "all"
const ETAS  = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) :
              [0.0, 0.05, 0.2, 0.5, 1.0]
const TMAX  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5
const DT    = 0.05
const MI    = 16
const CAP   = 32

const OUT = joinpath(@__DIR__, "results", "nonunitary_blowup.txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
const CSV = open(joinpath(@__DIR__, "results", "nonunitary_blowup.csv"), "w")
println(CSV, "model,scheme,eta,t,norm,energy,err,states,finite")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:U1)

function setup(nm)
    if nm == "chain"
        L = 16
        Jm = [abs(i - j) == 1 ? 1.0 : 0.0 for i in 1:L, j in 1:L]
        W  = xxz_mpo(L; J = 1.0, delta = 1.0)
        Hs, states, idx = heisenberg_sparse(L)
        heis = true
    elseif nm == "square"
        L = 16
        Jm = square_cylinder_couplings(4, 4)
        W  = square_cylinder_mpo(4, 4)
        Hs, states, idx = pairs_sparse(L, Jm)
        heis = true
    else
        L = 18
        Jm = breathing_kagome_couplings(2, 3; h = 0.30)
        W  = breathing_kagome_mpo(2, 3; h = 0.30)
        Hs, states, idx = _xy_sparse(L, Jm)
        heis = false
    end
    # ⛔ USE `neel_vector`, NEVER A HAND-ROLLED BIT PATTERN. `neel_state` puts ODD sites up
    # (`1 << (j-1)` for `j in 1:2:L`); a hand-rolled `1 << (2j-1)` puts EVEN sites up -- the
    # MIRRORED state. The two agree on |Sz| and on the energy, so the reference looks healthy and
    # only the observable comparison is wrong: MEASURED as err = 8.807e-01 at eta = 0, in the
    # UNITARY control, which is where it was caught. This repo's own notes record the dense/sparse
    # bit conventions as mirrored and Neel as the state that normally HIDES it.
    v0 = neel_vector(L, idx)
    return (nm = nm, L = L, W = W, psi0 = neel_state(L), Hs = Hs, states = states,
            idx = idx, v0 = v0, heis = heis)
end

function _xy_sparse(L, Jm)
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    idx = Dict(s => k for (k, s) in enumerate(states))
    I, Jd, V = Int[], Int[], Float64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states), i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        bit(b, i) == bit(b, j) && continue
        bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
        push!(I, idx[bb]); push!(Jd, k); push!(V, Jm[i, j] / 2)
    end
    return sparse(I, Jd, V, length(states), length(states)), states, idx
end

"`<S^z_j>` from a sector vector -- the observable both sides can compute."
function sz_dense(S, v)
    p = abs2.(v); n = sum(p)
    return [sum(p[k] * (((S.states[k] >> (j - 1)) & 1) - 0.5) for k in eachindex(S.states)) / n
            for j in 1:S.L]
end
sz_mps(psi) = magnetisation(copy(psi)) ./ max(norm(copy(psi))^2, eps())

for nm in (WHICH == "all" ? ("chain", "square", "kagome") : (WHICH,))
    S = setup(nm)
    say("")
    say("="^108)
    say(@sprintf("%s -- %d sites.  tau = -(eta + i)*dt,  dt = %g,  T = %g,  maxdim = %d",
                 uppercase(nm), S.L, DT, TMAX, CAP))
    say("  eta = 0 is ordinary unitary evolution (the control). eta > 0 damps, and is where")
    say("  TDVP's BACKWARD half-step becomes exp(+eta*dt*H/2) -- an amplifier.")
    say("="^108)
    say(@sprintf("  %-14s %6s %11s %13s %12s %7s %8s",
                 "scheme", "eta", "|psi|", "Re E", "err vs exact", "states", "finite"))
    for eta in ETAS
        tau = ComplexF64(-(eta + im) * DT)
        nst = round(Int, TMAX / DT)
        # the exact non-unitary propagator, same tau
        vt = copy(S.v0)
        for _ in 1:nst
            vt = expv_sparse(S.Hs, tau, vt; m = 30, tol = 1e-13)
        end
        ref = sz_dense(S, vt)
        for (snm, mk) in (("tdvp2", (p, t) -> tdvp2_step!(p, S.W, t; maxdim = CAP,
                                          trunc_thresh = 1e-10, maxiter = MI)),
                          ("tdvp_cbe1s", (p, t) -> tdvp_cbe1s_step!(p, S.W, t; maxdim = CAP,
                                          trunc_thresh = 1e-10, maxiter = MI)),
                          ("cbe_bug", (p, t) -> cbe_bug_step!(p, S.W, t; exact = true,
                                          krylov_basis = 3, krylov_tol = 0.0, maxdim = CAP,
                                          trunc_thresh = 1e-10, maxiter = MI)))
            psi = copy(S.psi0); ok = true
            for k in 1:nst
                try
                    mk(psi, tau)
                catch e
                    ok = false; break
                end
                isfinite(norm(psi)) || (ok = false; break)
            end
            n = ok ? norm(psi) : NaN
            e = ok && isfinite(n) && n > 0 ?
                real(mpo_energy(copy(psi), S.W)) / max(n^2, eps()) : NaN
            er = ok && isfinite(n) && n > 0 ? maximum(abs.(sz_mps(psi) .- ref)) : NaN
            st = ok && isfinite(n) ? maximum(state_bond_dims(psi)) : 0
            say(@sprintf("  %-14s %6.2f %11.3e %13.5f %12s %7d %8s", snm, eta, n, e,
                         isnan(er) ? "--" : @sprintf("%.3e", er), st,
                         ok && isfinite(n) ? "yes" : "NO -- BLEW UP"))
            @printf(CSV, "%s,%s,%g,%g,%.6e,%.6e,%.6e,%d,%s\n",
                    nm, snm, eta, TMAX, n, e, er, st, ok && isfinite(n))
            flush(CSV)
        end
        say("  " * "-"^100)
    end
end
say("")
say("READ IT AS: `eta = 0` must be identical for all three -- that is the control that says the")
say("difference at eta > 0 is the NON-UNITARITY and not a pre-existing gap. A scheme that goes")
say("non-finite has been amplified by its own backward step; one that merely loses norm is")
say("doing what the physics says.")
close(IO_); close(CSV)
