# IS THE KAGOME SCHMIDT SPECTRUM REALLY FLAT, OR DID THE SETUP SUPPRESS THE DYNAMICS?
#
# The scheme comparison found `cbe_bug`'s advantage much smaller on kagome (0.91x the TDVP bar)
# than on the square lattice (0.61x), and the explanation offered was "kagome's Schmidt spectrum
# is flatter, so a rank-adaptive basis has less to exploit". THAT EXPLANATION IS A GUESS, and
# three features of the run could produce the same observation with no lattice physics involved:
#
#   1. THE DIMER START MAY BE NEARLY AN EIGENSTATE ON KAGOME. The kagome ground state is
#      RVB-like -- a superposition of dimer coverings -- so quenching FROM a dimer covering is
#      close to quenching from an eigenstate, and an eigenstate generates NO entanglement at all.
#      On the square lattice a dimer covering is far from the ground state, so the same start is
#      a genuine quench there and not here. That asymmetry alone could explain everything.
#   2. T = 0.3 IS SIX STEPS. Barely into the growth regime on either lattice.
#   3. Lx = 2 MEANS EVERY CELL IS AN EDGE CELL. An 18-site kagome at 2x3 has essentially no bulk.
#
# ⛔ SO THIS FILE MEASURES THE SPECTRUM ITSELF rather than inferring it from an integrator's
# error. Entanglement entropy and Schmidt-value decay at the central cut, for several STARTS and
# out to a longer T, at a cap loose enough not to bind -- because a spectrum measured through a
# binding cap is the cap's spectrum.
#
# ⚠ AND THE STARTS ARE CHOSEN TO SPAN THE QUESTION. `dimer` is the one already used; `neel` is a
# definite-Sz product state that is FAR from the kagome ground state and should quench hard;
# `stagger` is a third product state that shares neither's symmetry. If kagome's spectrum is
# genuinely flat, all three should show it. If only `dimer` does, the earlier conclusion was an
# artefact of the start and the scheme comparison needs re-running from a state that actually
# generates entanglement.
#
# Run:  julia --project=. benchmarks/entanglement_dynamics.jl [t_max] [maxdim]

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const TMAX = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
const CAP  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
const DT   = 0.05
const MI   = 16

const OUT = joinpath(@__DIR__, "results", "entanglement_dynamics.txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:U1)

"""
Schmidt spectrum at the central cut: entropy, and how many values clear each tolerance.

⛔ THE COUNTS ARE THE POINT, not the entropy. "Flat spectrum" means MANY values of comparable
size -- i.e. `n(1e-6)` close to the full rank -- and that is precisely the situation a
rank-adaptive scheme cannot exploit. Entropy alone cannot distinguish "many tiny values" from
"few large ones" at the same total.
"""
function spectrum(psi)
    L = length(psi)
    c = max(1, L ÷ 2)
    p = copy(psi)
    canonical!(p, c)
    res = svd(p[c], (1, 2); cutoff = 0.0, get_lists = true)
    s = Float64[]
    for (val, deg, _, _) in res.kept_list, _ in 1:deg
        push!(s, Float64(val))
    end
    isempty(s) && return (S = 0.0, n = zeros(Int, 3), top = 0.0, ratio = 0.0)
    s ./= sqrt(sum(abs2, s))
    p2 = abs2.(s)
    S = -sum(x -> x > 0 ? x * log(x) : 0.0, p2)
    return (S = S, n = [count(>(t), s) for t in (1e-2, 1e-4, 1e-6)],
            top = maximum(s),
            # decay ratio: the 10th value over the 1st. Near 1 = FLAT, near 0 = fast decay.
            ratio = length(s) >= 10 ? s[min(10, length(s))] / s[1] : 0.0)
end

function make(name, lat)
    L = lat.L
    name == "dimer"   && return dimer_state(L)
    name == "neel"    && return neel_state(L)
    # a third product state with neither symmetry: blocks of two up then two down
    return product_state([iseven((j - 1) ÷ 2) ? :up : :down for j in 1:L])
end

lattices = [
    (nm = "square 4x4", L = 16, W = () -> square_cylinder_mpo(4, 4)),
    (nm = "kagome 2x3", L = 18, W = () -> kagome_cylinder_mpo(2, 3)),
    (nm = "kagome 3x3", L = 27, W = () -> kagome_cylinder_mpo(3, 3)),
]

say(@sprintf("Entanglement growth at the central cut -- dt=%g, T=%g, maxdim=%d (loose)",
             DT, TMAX, CAP))
say("n(t) = how many Schmidt values exceed t.  ratio = s[10]/s[1]: near 1 is FLAT.")

for lat in lattices
    W = lat.W()
    say("")
    say("="^100)
    say(@sprintf("%s -- %d sites", lat.nm, lat.L))
    say("="^100)
    for start in ("dimer", "neel", "stagger")
        (start == "dimer" && isodd(lat.L)) && continue
        psi = make(start, lat)
        e0 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
        say(@sprintf("  --- start = %-8s  E/site = %+.6f ---", start, e0 / lat.L))
        say(@sprintf("      %6s %9s %8s %8s %8s %9s %8s",
                     "t", "S_vN", "n(1e-2)", "n(1e-4)", "n(1e-6)", "s10/s1", "chi"))
        for k in 0:round(Int, TMAX / DT)
            if k > 0
                cbe_bug_step!(psi, W, ComplexF64(-im * DT); exact = true, krylov_basis = 3,
                              krylov_tol = 0.0, maxdim = CAP, trunc_thresh = 1e-12, maxiter = MI)
            end
            if k % 4 == 0
                sp = spectrum(psi)
                say(@sprintf("      %6.2f %9.4f %8d %8d %8d %9.3e %8d",
                             k * DT, sp.S, sp.n[1], sp.n[2], sp.n[3], sp.ratio,
                             maximum(state_bond_dims(psi))))
            end
        end
    end
end

say("")
say("="^100)
say("HOW TO READ IT")
say("="^100)
say("  If KAGOME's `n(1e-6)` and `s10/s1` are comparable to the square lattice's from a start")
say("  that genuinely quenches (neel / stagger), then the spectrum is NOT intrinsically flat and")
say("  the earlier 0.91x-vs-0.61x result was an artefact of the DIMER start being close to")
say("  kagome's low-energy manifold. The scheme comparison would then need re-running from a")
say("  start that actually generates entanglement on both lattices.")
say("")
say("  If kagome is flat from EVERY start, the original reading stands.")
close(IO_)
