# AN OPEN SYSTEM ON AN MPS: A GENUINELY NON-HERMITIAN GENERATOR, AND TDVP'S BACKWARD STEP.
#
# ⛔ WHY A COMPLEX `tau` IS NOT ENOUGH, AND WHY THIS FILE EXISTS. Driving a REAL `H` with
# `tau = -(eta + i)dt` gives `exp(-eta*dt*H)` -- imaginary-time evolution, which AMPLIFIES
# negative-energy states rather than damping them (measured: the norm GROWS, |psi| = 1.101 at
# eta = 0.05). It is also just a complex RESCALING of H, so it cannot represent a bath that acts
# differently on different terms. A real open system needs a NON-HERMITIAN generator:
#
#     H_eff = H_XY  +  (Delta - i*Gamma) * sum_<ij> S^z_i S^z_j
#
# the no-jump part of a Lindblad evolution with a two-body (correlated-loss) dissipator.
#
# ⛔ AND IT MUST BE TWO-BODY, BECAUSE ONE-BODY LOSS IS TRIVIAL HERE. `sum_j n_j` is CONSERVED
# under U(1), so `-i*Gamma*sum_j n_j` is a constant times the identity: a global decay factor and
# no dynamics whatsoever. `n_i n_j = S^z_i S^z_j + (one-body) + const`, and the one-body and
# constant parts are again conserved, so the whole dissipator reduces to an IMAGINARY `S^z S^z`
# coupling -- which is exactly what is built below.
#
# ⛔ THIS NEEDS ONE COUPLING MATRIX PER VERTEX. A single complex `J` would make the XY vertices
# complex too, giving `(1 - i*g)*H` -- a complex rescaling, i.e. the complex-`tau` case again and
# NOT a dissipator. The XY vertices must stay REAL while the `S^zS^z` vertex goes complex, which
# is what `pair_mpo(L, Js::Vector, vertices)` is for.
#
# ── THE PREDICTION, WRITTEN DOWN BEFORE MEASURING ────────────────────────────────────────
#
# TDVP is a projector splitting: FORWARD `+tau/2` on the site, BACKWARD `-tau/2` on the bond.
# With `H_eff = H_r - i*Gamma*K` (K = sum S^zS^z, Hermitian) and `tau = -i*dt`:
#
#     forward   exp(-i*H_r*dt/2) * exp(-Gamma*K*dt/2)     decays
#     backward  exp(+i*H_r*dt/2) * exp(+Gamma*K*dt/2)     GROWS, by exp(+Gamma*|K|*dt/2)
#
# `K` has eigenvalues up to ~L/4, so the backward step amplifies by `exp(+Gamma*L*dt/8)` per step
# and it COMPOUNDS. `cbe_bug` has no backward step at all -- its half-sweeps build bases and
# evolve nothing, and the single root solve is forward over the full `tau`.
#
# ⚠ THE PREDICTION CAN FAIL, and the table is built to show it: TDVP may simply survive at these
# `Gamma` and `dt` (the amplification is bounded and could be small), or `cbe_bug` may diverge for
# an unrelated reason. `Gamma = 0` is the Hermitian control -- all three MUST agree there, or any
# difference at `Gamma > 0` is a pre-existing gap and not the non-Hermiticity.
#
# ⚠ AND LOSING NORM IS NOT A BLOW-UP. A dissipative generator is SUPPOSED to lose norm. Only
# DIVERGENCE or a non-finite value is a failure, which is what the verdict column reports.
#
# Run:  julia --project=. benchmarks/open_system.jl [model] [gammas] [t_max]

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const WHICH  = length(ARGS) >= 1 ? ARGS[1] : "all"
const GAMMAS = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.0, 0.2, 1.0, 3.0]
const TMAX   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5
const DT     = 0.05
const MI     = 16
const CAP    = 32

const OUT = joinpath(@__DIR__, "results", "open_system.txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
const CSV = open(joinpath(@__DIR__, "results", "open_system.csv"), "w")
println(CSV, "model,scheme,gamma,norm,reE,imE,err,states,finite")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:U1)

"Real XY couplings and a SEPARATE `S^zS^z` matrix, so only the latter goes complex."
function geometry(nm)
    if nm == "chain"
        L = 18; Jm = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    elseif nm == "square"
        # 5x5 cylinder = 25 sites, snake-mapped to an MPS (no PEPS).
        # ⚠ 25 IS ODD, so there is no half-filled Sz = 0 sector and no `dimer_state`; and the
        # nearest sector, C(25,12) = 5.2e6 states, needs ~2.5 GB for a depth-30 Krylov basis. So
        # this size has NO EXACT REFERENCE and is reported on conservation and cross-scheme
        # agreement only -- `HAVE_REF` below decides, rather than a comment promising it.
        L = 25; Jm = square_cylinder_couplings(5, 5)
    else
        L = 18; Jm = kagome_cylinder_couplings(2, 3)
    end
    return L, Jm
end

"`H_eff = XY(real) + (1 - i*Gamma) * S^zS^z`, as an MPO."
function heff(L, Jm, gamma)
    v = pair_vertices(xxz_chain(L; J = 1.0, delta = 1.0))
    # ⛔ `S^zS^z` IS THE LAST VERTEX, NOT THE FIRST. `xxz_chain` pushes the two charged
    # `S^+`/`S^-` terms and only THEN the `S^z` term (`henv.jl`: `delta == 0.0 || push!(...)`).
    # A first version of this used `Js[1]` and would have made the HOPPING dissipative while
    # leaving `S^zS^z` real -- a different model that still runs, still conserves U(1), and still
    # looks Hermitian-ish. This is exactly the "pairs by position" hazard the `pair_mpo`
    # docstring warns about, so the index is DERIVED and CHECKED rather than written down.
    izz = findall(t -> t.left === t.right, v)
    length(izz) == 1 || error("expected exactly one S^zS^z vertex, found $(length(izz)) of " *
                              "$(length(v)) -- the vertex list has changed shape")
    Js = Any[ComplexF64.(Jm) for _ in v]
    Js[izz[1]] = ComplexF64.(Jm) .* (1.0 - im * gamma)
    return pair_mpo(L, Vector{AbstractMatrix}(Js), v)
end

"The same generator as a sparse matrix in the Sz=0 sector -- the exact reference."
function heff_sparse(L, Jm, gamma)
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    idx = Dict(s => k for (k, s) in enumerate(states))
    I, Jd, V = Int[], Int[], ComplexF64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states), i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        si, sj = bit(b, i), bit(b, j)
        # S^zS^z: diagonal, and the ONLY term carrying the imaginary part
        push!(I, k); push!(Jd, k)
        push!(V, Jm[i, j] * (1.0 - im * gamma) * (si == sj ? 0.25 : -0.25))
        if si != sj                                   # the real XY flip-flop
            bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
            push!(I, idx[bb]); push!(Jd, k); push!(V, ComplexF64(Jm[i, j] / 2))
        end
    end
    return sparse(I, Jd, V, length(states), length(states)), states, idx
end

sz_mps(psi) = magnetisation(copy(psi)) ./ max(norm(copy(psi))^2, eps())
function sz_dense(v, L, states)
    p = abs2.(v); n = sum(p)
    return [sum(p[k] * (((states[k] >> (j - 1)) & 1) - 0.5) for k in eachindex(states)) / n
            for j in 1:L]
end

for nm in (WHICH == "all" ? ("chain", "square", "kagome") : (WHICH,))
    L, Jm = geometry(nm)
    say("")
    say("="^110)
    say(@sprintf("%s -- %d sites.  H_eff = XY(real) + (1 - i*Gamma) S^zS^z,  dt=%g, T=%g, D=%d",
                 uppercase(nm), L, DT, TMAX, CAP))
    say("  Gamma = 0 is the HERMITIAN control: all three schemes must agree there.")
    say("  Gamma > 0 makes TDVP's backward bond step exp(+Gamma*K*dt/2) -- an amplifier.")
    say("="^110)
    say(@sprintf("  %-14s %7s %12s %12s %12s %12s %7s %8s",
                 "scheme", "gamma", "|psi|", "Re E", "Im E", "err vs exact", "states", "finite"))
    for g in GAMMAS
        W = heff(L, Jm, g)
        nst = round(Int, TMAX / DT)
        # ⚠ ONLY WHERE IT IS AFFORDABLE. `H_eff` acts on a STATE, so its reference is a sector
        # vector and is reachable to L ~ 20 -- unlike a vectorised Lindbladian, whose reference is
        # a DENSITY MATRIX and is out of reach past L ~ 8.
        have_ref = iseven(L) && binomial(L, L ÷ 2) <= 200_000
        ref, states = if have_ref
            Hs, sts, idx = heff_sparse(L, Jm, g)
            vt = neel_vector(L, idx)
            for _ in 1:nst
                vt = expv_sparse(Hs, ComplexF64(-im * DT), vt; m = 30, tol = 1e-13)
            end
            (sz_dense(vt, L, sts), sts)
        else
            (Float64[], Int[])
        end
        # ⛔ `hermitian = false` IS THE WHOLE POINT. `H_eff` is NOT Hermitian, and Lanczos on it
        # returns a plausible vector with no error at all -- every number below would be quietly
        # wrong. `g = 0` makes it Hermitian again, which is why that row is the control.
        for (snm, st) in (("tdvp2", (p, t) -> tdvp2_step!(p, W, t; maxdim = CAP, hermitian = false,
                                        trunc_thresh = 1e-10, maxiter = MI)),
                          ("tdvp_cbe1s", (p, t) -> tdvp_cbe1s_step!(p, W, t; maxdim = CAP,
                                        hermitian = false,
                                        trunc_thresh = 1e-10, maxiter = MI)),
# ⛔ ONE NAME FOR THE SWEEP: `bug_interleaved`. This is the sweep behind the competitive
# OAT / XX / TFIM results and the square-Heisenberg advantage -- it was labelled `cbe_bug`
# here and `bug_interleaved` in the L=16 campaign, so the SAME arm wore two names across
# CSVs and could not be joined between campaigns without a lookup nobody applies.
                          ("bug_interleaved", (p, t) -> cbe_bug_step!(p, W, t; exact = true,
                                        krylov_basis = 3, krylov_tol = 0.0, maxdim = CAP,
                                        hermitian = false,
                                        trunc_thresh = 1e-10, maxiter = MI)))
            psi = neel_state(L); ok = true
            for _ in 1:nst
                try
                    st(psi, ComplexF64(-im * DT))
                catch
                    ok = false; break
                end
                isfinite(norm(psi)) || (ok = false; break)
            end
            n = ok ? norm(psi) : NaN
            e = ok && isfinite(n) && n > 0 ? mpo_energy(copy(psi), W) / max(n^2, eps()) :
                                             ComplexF64(NaN)
            er = (have_ref && ok && isfinite(n) && n > 0) ?
                 maximum(abs.(sz_mps(psi) .- ref)) : NaN
            stt = ok && isfinite(n) ? maximum(state_bond_dims(psi)) : 0
            say(@sprintf("  %-14s %7.2f %12.4e %12.5f %12.5f %12s %7d %8s",
                         snm, g, n, real(e), imag(e),
                         isnan(er) ? "--" : @sprintf("%.3e", er), stt,
                         ok && isfinite(n) ? "yes" : "NO -- BLEW UP"))
            @printf(CSV, "%s,%s,%g,%.6e,%.6e,%.6e,%.6e,%d,%s\n",
                    nm, snm, g, n, real(e), imag(e), er, stt, ok && isfinite(n))
            flush(CSV)
        end
        say("  " * "-"^102)
    end
end
close(IO_); close(CSV)
