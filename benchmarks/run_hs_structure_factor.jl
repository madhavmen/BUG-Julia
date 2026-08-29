# Run the Li et al. Fig-3 protocol on the Haldane-Shastry ring and write `S(k, w)` to CSV.
#
#   julia --project=. benchmarks/run_hs_structure_factor.jl [L] [T] [dt] [scheme]
#
# DEFAULTS ARE SMALL ON PURPOSE. The point of the first run is the SUM RULES, which are exact at
# any L and therefore decide whether the pipeline is right; the physics comparison against the
# paper wants a bigger ring and belongs on the cluster. `S^z_{j0}|0>` on the HS ring is close to
# full rank the moment it is formed (measured at L=16: centre bond 247 of a 256 cap), so cost
# climbs steeply with L and this is NOT a laptop job past about L=12.
#
# ⚠ `T` SETS THE FREQUENCY RESOLUTION, `dt` ONLY SETS THE CEILING. The record is finite, so
# `dw ~ 2 pi / T` regardless of how finely it is sampled: T=8 resolves 0.79, and the whole
# two-spinon band tops out at `w_+(pi) = pi^2/4 = 2.47`, i.e. about THREE resolvable bins across
# it. Enough to see the support and the edges, far too coarse to read a lineshape. For a figure
# meant to be compared with the paper use T >= 32 (dw ~ 0.20); at L=8 the state is full rank at
# chi=16 and the extra steps are cheap. Halving `dt` buys nothing here -- it raises the Nyquist
# ceiling `pi/dt`, already 63 at dt=0.05, which was never the binding constraint.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate
import Random
Random.seed!(42)

const HERE = dirname(@__FILE__)
include(joinpath(HERE, "hs_structure_factor.jl"))

const L      = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 8
const TMAX   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 8.0
const DT     = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05
const SCHEME = length(ARGS) >= 4 ? String(ARGS[4])         : "bug_interleaved"
const MAXDIM = 256
const CUT    = 1e-8

# ⛔ ONE BUG SCHEME ACROSS EVERY RUN: `bug_interleaved`. `bug_decoupled` is GONE -- its flag lived
# on the K-step path, removed with `kaug`/`rexpand` on 2026-08-24, after which the two branches
# here called `cbe_bug_step!` with IDENTICAL arguments. This driver even DEFAULTED to the dead
# name, so its whole campaign was labelled with an ordering that no longer exists.
stepper(name, mpo) =
    name == "bug_interleaved" ? (p, t) -> cbe_bug_step!(p, mpo, t; maxdim = MAXDIM,
                                                        trunc_thresh = CUT) :
    name == "tdvp2"           ? (p, t) -> tdvp2_step!(p, mpo, t; maxdim = MAXDIM,
                                                      trunc_thresh = CUT) :
    name == "tdvp_cbe1s"      ? (p, t) -> tdvp_cbe1s_step!(p, mpo, t; maxdim = MAXDIM,
                                                           trunc_thresh = CUT) :
    throw(ArgumentError("unknown scheme $name"))

function main()
    set_symmetry!(:U1)
    mpo = haldane_shastry_mpo(L)

    # Ground state by OUR OWN CBE-RSVD DMRG, checked against the closed form before it is used
    # for anything -- a wrong |0> would corrupt C(x,t) in a way the sum rules below would not
    # obviously catch, since they are statements about the transform, not about the state.
    psi0 = dimer_state(L)
    run  = rsvd_cbe_dmrg1s!(psi0, mpo; n_sweeps = 40, maxdim = MAXDIM, trunc_thresh = 1e-12)
    E0   = run.energies[end]
    Eref = -pi^2 * (L^2 + 5) / (24 * L)
    @printf("  |0>: E=%.12f  closed form=%.12f  diff=%.2e  chi=%d\n",
            E0, Eref, abs(E0 - Eref), run.max_bond_dims[end]); flush(stdout)
    abs(E0 - Eref) < 1e-8 || error("ground state is not the HS ground state; aborting")

    j0 = L ÷ 2
    nt = round(Int, TMAX / DT)
    @printf("  protocol: j0=%d, %d steps of dt=%g to T=%g, scheme=%s\n",
            j0, nt, DT, TMAX, SCHEME); flush(stdout)

    t0 = time()
    ts, C = sz_correlation(psi0, E0, stepper(SCHEME, mpo), j0, nt, DT)
    @printf("  C(x,t) done in %.1f s\n", time() - t0); flush(stdout)

    ks, ws, S = structure_factor(C, ts, L)
    r = check_sum_rules(C, L, S, ws)

    println("\n  EXACT IDENTITIES (these decide whether the pipeline is right):")
    @printf("    max |Im C(d,0)|                 = %.3e   (want 0)\n", r.c0_imag)
    @printf("    C(0,0) = <(S^z)^2>              = %.6f    (want 0.25)\n", r.onsite)
    @printf("    total weight  sum S dw / (2pi L)= %.6f    (want 0.25)\n", r.total)
    @printf("    max |S(k,w<0)| / max|S|         = %.3e   (want 0)\n", r.negw)
    @printf("    ring reflection |C(d)-C(-d)|    = %.3e   (want 0)\n", r.refl)
    @printf("    k=0 channel  max_t |sum_d C|    = %.3e   (want 0: S^z_tot|0>=0)\n", r.ktot)
    flush(stdout)

    out = joinpath(HERE, "results", "hs_skw_L$(L)_T$(TMAX)_$(SCHEME).csv")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "k,omega,S,w_lower,w_upper")
        for (ik, k) in enumerate(ks)
            lo, hi = two_spinon_band(k)
            for (iw, om) in enumerate(ws)
                @printf(io, "%.10f,%.10f,%.10e,%.10f,%.10f\n", k, om, S[ik, iw], lo, hi)
            end
        end
    end
    println("\n  wrote ", out); flush(stdout)

    # ENFORCED, and only AFTER the CSV is on disk: these are exact identities, so a violation is a
    # pipeline bug and the run must not pass quietly -- but the data is what makes the violation
    # diagnosable, so it gets written first. The `negw` tolerance is the loose one on purpose: the
    # finite Hann-windowed record leaks a little weight below w = 0 even for a perfect spectrum,
    # whereas the other four are algebraic and hold to roundoff.
    bad = String[]
    r.c0_imag  < 1e-10 || push!(bad, @sprintf("Im C(d,0) = %.3e", r.c0_imag))
    abs(r.onsite - 0.25) < 1e-10 || push!(bad, @sprintf("C(0,0) = %.8f != 0.25", r.onsite))
    abs(r.total  - 0.25) < 1e-3  || push!(bad, @sprintf("total weight = %.6f != 0.25", r.total))
    r.refl     < 1e-8  || push!(bad, @sprintf("ring reflection = %.3e", r.refl))
    # Looser than the algebraic identities because this one accumulates the integrator's own
    # truncation over the whole record, not just roundoff -- but it is still an EXACT statement,
    # so anything above 1e-6 means charge is leaking, not that the tolerance is tight.
    r.ktot     < 1e-6  || push!(bad, @sprintf("k=0 channel = %.3e", r.ktot))
    r.negw     < 5e-2  || push!(bad, @sprintf("weight at w<0 = %.3e", r.negw))
    isempty(bad) ? println("  all identities hold.") :
        error("EXACT IDENTITIES VIOLATED -- the pipeline is wrong, not the physics:\n    " *
              join(bad, "\n    "))
end

main()
