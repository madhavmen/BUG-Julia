# THE 8x8 CYLINDER: WHAT TRUNCATION ACTUALLY DOES, AT A SIZE WHERE IT DOMINATES.
#
# 64 sites, Ly = 8. A cut severs 8 bonds, so the rank needed for an exact answer grows like 2^8
# and the state is SEVERELY truncated at any affordable `maxdim`. That is the point: every other
# measurement in this study was taken where truncation was mild or absent, and the ranking of the
# schemes turned out to DEPEND on that.
#
# ⛔ THERE IS NO REFERENCE HERE AND THERE CANNOT BE. 2^64 is out of reach and so is the Sz=0
# sector (1.8e18 states). Nothing in this file is an "error". What IS available, and what it can
# honestly support:
#
#   CONVERGENCE IN `maxdim`   the same run at increasing rank. Where two consecutive caps agree,
#                             the answer has stopped depending on the truncation -- that is the
#                             only self-contained accuracy statement available without a
#                             reference, and it is a NECESSARY not sufficient condition.
#   CROSS-SCHEME AGREEMENT    three independent integrators landing on the same profile is
#                             evidence; one of them disagreeing names a suspect.
#   ENERGY DRIFT              a conservation check ONLY. ⚠ On this repo's own models the
#                             conservation and accuracy orderings DISAGREE outright -- a
#                             basis-only arm once conserved energy to 7.4e-15 while sitting 12x
#                             further from the exact state than tdvp2. Never rank by dE.
#
# ⛔ AND THE HEADLINE QUESTION IS WHETHER THE 4x4 RESULT SURVIVES. At 4x4 with a binding cap,
# `cbe_bug` BEAT both TDVPs (2.81e-02 against 4.31e-02) at 9.6x fewer operator applications --
# the reverse of the full-rank 4x3 result, where TDVP won. If that reversal is real and not an
# artefact of one small lattice, it should be LARGER here, where truncation is far more severe.
# If it vanishes, the 4x4 result was a coincidence of that size.
#
# Run:  julia --project=. benchmarks/cylinder_8x8.jl [Lx] [Ly] [t_max] [caps]

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const LX   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
const LY   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
const TMAX = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.2
const CAPS = length(ARGS) >= 4 ? parse.(Int, split(ARGS[4], ",")) : [16, 32, 64]
const DT   = 0.05
const L    = LX * LY
const MI   = 16

const OUT = joinpath(@__DIR__, "results", @sprintf("cylinder_%dx%d.txt", LX, LY))
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:U1)
const BONDS = square_cylinder_bonds(LX, LY)
const W     = square_cylinder_mpo(LX, LY)
const PSI0  = dimer_state(L)

# ⛔ SUBSAMPLED, AND THE OPERATORS ARE BUILT ONCE. `bond_profile` needs one single-entry
# `pair_mpo` per bond, and at 64 sites with 120 bonds that is 120 full MPO CONSTRUCTIONS per
# measurement -- across nine runs it rivals the physics it is supposed to measure. A stride keeps
# a representative spread of rung / wrap / leg bonds (they are interleaved in `BONDS` by
# construction) at a tenth of the cost, and the operators are hoisted out of the run loop so each
# is built exactly once for the whole scan.
const PROBE_BONDS = BONDS[1:max(1, length(BONDS) ÷ 24):end]
const PROBE_OPS = [begin
                       Jm = zeros(Float64, L, L); Jm[i, j] = 1.0
                       pair_mpo(L, Jm, spin_vertices(L))
                   end for (i, j) in PROBE_BONDS]

"`<S_i . S_j>` on a representative subset of the cylinder's bonds, measured ONCE per run."
function bond_profile(psi)
    n2 = max(norm(copy(psi))^2, eps())
    return [real(mpo_energy(copy(psi), op)) / n2 for op in PROBE_OPS]
end

say(@sprintf("Heisenberg on a %dx%d cylinder -- %d sites, %d bonds, dt=%g, T=%g",
             LX, LY, L, length(BONDS), DT, TMAX))
say(@sprintf("MPO max virtual dim = %d   (tracks Ly=%d)", maximum(mpo_virtual_dims(W)), LY))
say(@sprintf("full rank at this L would be 2^%d -- UNREACHABLE, so every cap below TRUNCATES HARD",
             L ÷ 2))
say("⚠ NO EXACT REFERENCE EXISTS AT THIS SIZE. Nothing here is an error; read convergence in")
say("  maxdim and agreement between schemes, and never rank by energy drift.")

function run(step!)
    psi, kry, secs = copy(PSI0), 0, 0.0
    e0 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    for _ in 1:round(Int, TMAX / DT)
        t0 = time_ns()
        info = step!(psi, ComplexF64(-im * DT))
        secs += (time_ns() - t0) / 1e9
        hasproperty(info, :krylov_dims) && (kry += info.krylov_dims)
    end
    e1 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    return (prof = bond_profile(psi), dE = abs(e1 - e0),
            states = maximum(state_bond_dims(psi)), kry = kry, secs = secs,
            nrm = norm(psi))
end

schemes(cap) = [
    ("tdvp2",      (p, t) -> tdvp2_step!(p, W, t; maxdim = cap, trunc_thresh = 1e-10, maxiter = MI)),
    ("tdvp_cbe1s", (p, t) -> tdvp_cbe1s_step!(p, W, t; maxdim = cap, trunc_thresh = 1e-10,
                                              maxiter = MI)),
    ("cbe_bug m=3", (p, t) -> cbe_bug_step!(p, W, t; exact = true, krylov_basis = 3,
                                            krylov_tol = 0.0, maxdim = cap,
                                            trunc_thresh = 1e-10, maxiter = MI)),
]

results = Dict{Tuple{String, Int}, Any}()
for cap in CAPS
    say("")
    say("="^100)
    say(@sprintf("maxdim = %d", cap))
    say("="^100)
    say(@sprintf("  %-14s %11s %8s %9s %10s %9s", "scheme", "|dE|", "states", "krylov", "sec", "|psi|"))
    for (nm, st) in schemes(cap)
        r = run(st)
        results[(nm, cap)] = r
        say(@sprintf("  %-14s %11.3e %8d %9s %10.1f %9.6f",
                     nm, r.dE, r.states, r.kry == 0 ? "-" : string(r.kry), r.secs, r.nrm))
    end
    for (a, b) in (("tdvp2", "tdvp_cbe1s"), ("tdvp2", "cbe_bug m=3"), ("tdvp_cbe1s", "cbe_bug m=3"))
        say(@sprintf("    agreement %-14s vs %-14s  %.3e", a, b,
                     maximum(abs.(results[(a, cap)].prof .- results[(b, cap)].prof))))
    end
end

# ── CONVERGENCE IN maxdim: the only self-contained accuracy statement available here ───────
say("")
say("="^100)
say("CONVERGENCE IN maxdim -- max |profile(cap) - profile(largest cap)| per scheme")
say("="^100)
say("  A scheme whose answer stops moving as the cap grows has CONVERGED at this T; one that")
say("  keeps moving is still truncation-dominated. ⚠ NECESSARY, NOT SUFFICIENT: two caps can")
say("  agree on a wrong answer if both are limited the same way.")
big = maximum(CAPS)
for nm in ("tdvp2", "tdvp_cbe1s", "cbe_bug m=3")
    line = @sprintf("  %-14s", nm)
    for cap in CAPS
        d = maximum(abs.(results[(nm, cap)].prof .- results[(nm, big)].prof))
        line *= @sprintf("  D=%-4d %.3e", cap, d)
    end
    say(line)
end

# ── the cross-scheme spread AT THE LARGEST CAP, which is the closest thing to a verdict ────
say("")
say("="^100)
say("AT THE LARGEST CAP -- do the three schemes agree, and at what cost?")
say("="^100)
base = results[("tdvp2", big)]
for nm in ("tdvp2", "tdvp_cbe1s", "cbe_bug m=3")
    r = results[(nm, big)]
    say(@sprintf("  %-14s  spread vs tdvp2 %.3e   krylov %8s   %7.1fs   (%.2fx tdvp2 time)",
                 nm, maximum(abs.(r.prof .- base.prof)),
                 r.kry == 0 ? "-" : string(r.kry), r.secs, r.secs / base.secs))
end
say("")
say("⚠ A SPREAD IS NOT AN ERROR. Without a reference this says the schemes disagree, not which")
say("  one is right. The 4x4 run -- which DID have an exact reference -- found cbe_bug closer to")
say("  the truth than both TDVPs under a binding cap; that is the prior this size cannot re-test.")
close(IO_)
