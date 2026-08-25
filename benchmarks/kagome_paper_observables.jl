# THE PAPER'S OBSERVABLES ON THE BREATHING KAGOME (arXiv:2603.21147), AT A SIZE WE CAN CHECK.
#
# The paper measures: the chiral order parameter, spin correlations <sigma^x sigma^x> and
# <sigma^z sigma^z>, the structure factors S^xx(k) / S^zz(k), the correlation length
# (xi ~ 1.6a in the chiral phase), the Chern number, and the entanglement spectrum.
#
# ⛔ THE CHIRAL ORDER PARAMETER IS NOT MEASURABLE HERE, AND THAT IS EXACT, NOT A LIMITATION OF
# OUR SOLVERS. `chi = sum eps_abc sigma1^a sigma2^b sigma3^c` has exactly ONE `sigma^y` per term
# -- the only imaginary Pauli -- so chi is a purely imaginary Hermitian operator. The XY
# Hamiltonian is REAL in the Sz basis, so a non-degenerate ground state can be chosen real, and
# for real |psi>, <chi> = -<chi>* = 0. VERIFIED by exact diagonalisation: 0.000e+00 at every h
# (`benchmarks/results/chirality_check.txt`). The paper sees chi != 0 because iDMRG on an
# INFINITE cylinder selects one state from a degenerate manifold; a finite cluster with a unique
# ground state cannot. Reporting our zero as "no chiral order" would be reporting a theorem about
# finite real matrices as a physics result.
#
# ⛔ SO THE REPRODUCIBLE OBSERVABLES ARE THE T-EVEN ONES: the two correlation functions, their
# structure factors, and the correlation length. Those ARE the paper's other measurements and
# they carry the DSL-vs-CSL distinction: a Dirac spin liquid has ALGEBRAICALLY decaying
# correlations, a chiral spin liquid EXPONENTIAL ones with a finite xi. That contrast is
# qualitative and survives a small cluster; the critical value h_c does not, and the paper itself
# says h_c shifts with cylinder width.
#
# ⚠ HOW `C^zz` AND `C^xx` ARE MEASURED, since neither has a single-vertex MPO here.
# `pair_mpo(..., spin_vertices)` gives `S_i . S_j` and `pair_mpo(..., xxz_chain(delta=0))` gives
# `S^x S^x + S^y S^y`. So
#       C^xx = <XY> / 2      (U(1) makes <SxSx> = <SySy>)
#       C^zz = <S.S> - <XY>
# Two MPOs per pair rather than one, and no new operator machinery.
#
# Run:  julia --project=. benchmarks/kagome_paper_observables.jl [Lx] [Ly] [hs]

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const LX = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const LY = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const HS = length(ARGS) >= 3 ? parse.(Float64, split(ARGS[3], ",")) :
           [0.0, 0.10, 0.22, 0.30, 0.45]
const L  = 3 * LX * LY
const CAP = 96
const MI  = 16

const OUTDIR = joinpath(@__DIR__, "results"); mkpath(OUTDIR)
const CSV = open(joinpath(OUTDIR, "kagome_paper_observables.csv"), "w")
println(CSV, "h,i,j,r,Cxx,Czz,E0,S_vN,chi_states")
const TXT = open(joinpath(OUTDIR, "kagome_paper_observables.txt"), "w")
say(l) = (println(TXT, l); flush(TXT); println(stdout, l); flush(stdout))

set_symmetry!(:U1)

"""
Ground state by imaginary time. `cbe_bug` in imaginary time is a variational descent, so the
energy falling monotonically to a plateau is the convergence check -- reported, not assumed.
"""
function ground_state(W; nsteps = 200, dtau = 0.05)
    psi = neel_state(L)
    es = Float64[]
    for k in 1:nsteps
        cbe_bug_step!(psi, W, ComplexF64(-dtau); exact = true, krylov_basis = 30,
                      krylov_tol = 1e-8, maxdim = CAP, trunc_thresh = 1e-10, maxiter = MI)
        psi[psi.center] = to_concrete((1.0 / norm(psi)) * psi[psi.center])
        k % 20 == 0 && push!(es, real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps()))
    end
    return psi, es
end

"Entanglement entropy at the central cut."
function entropy(psi)
    p = copy(psi); c = max(1, length(p) ÷ 2)
    canonical!(p, c)
    res = svd(p[c], (1, 2); cutoff = 0.0, get_lists = true)
    s = Float64[]
    for (v, deg, _, _) in res.kept_list, _ in 1:deg
        push!(s, Float64(v))
    end
    isempty(s) && return 0.0, 0
    s ./= sqrt(sum(abs2, s)); p2 = abs2.(s)
    return -sum(x -> x > 0 ? x * log(x) : 0.0, p2), length(s)
end

say(@sprintf("Dipolar XY on a breathing kagome %dx%d = %d sites, maxdim=%d", LX, LY, L, CAP))
say("⛔ <chi> is EXACTLY ZERO on any finite real ground state (verified by ED) -- the T-even")
say("   correlations, structure factors and correlation length are what carry the physics here.")
say("")

pos = breathing_kagome_positions(LX, LY, 0.0)
# reference site: a bulk A sublattice site, so the correlation profile is not dominated by an edge
ref = 3 * ((LX ÷ 2) * LY + (LY ÷ 2)) + 1
ref = clamp(ref, 1, L)

for h in HS
    W = breathing_kagome_mpo(LX, LY; h = h)
    psi, es = ground_state(W)
    e0 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    S, nst = entropy(psi)
    conv = length(es) >= 2 ? abs(es[end] - es[end - 1]) : NaN
    say(@sprintf("h = %.2f   E0/site = %+.8f   S_vN = %.4f   states = %d   |dE last| = %.2e",
                 h, e0 / L, S, nst, conv))

    P = breathing_kagome_positions(LX, LY, h)
    n2 = max(norm(copy(psi))^2, eps())
    for j in 1:L
        j == ref && continue
        A = zeros(Float64, L, L)
        a, b = minmax(ref, j); A[a, b] = 1.0
        ss = real(mpo_energy(copy(psi), pair_mpo(L, A, spin_vertices(L)))) / n2
        xy = real(mpo_energy(copy(psi), pair_mpo(L, A, xxz_chain(L; J = 1.0, delta = 0.0)))) / n2
        cxx = xy / 2                       # U(1): <SxSx> = <SySy> = <XY>/2
        czz = ss - xy                      # <S.S> - <XY>
        r = sqrt((P[ref][1] - P[j][1])^2 + (P[ref][2] - P[j][2])^2)
        @printf(CSV, "%g,%d,%d,%.6f,%.8e,%.8e,%.8e,%.6f,%d\n",
                h, ref, j, r, cxx, czz, e0, S, nst)
    end
    flush(CSV)
end

say("")
say("HOW TO READ THE CORRELATION PLOTS")
say("  Dirac spin liquid  -> ALGEBRAIC decay: a straight line on log-log")
say("  chiral spin liquid -> EXPONENTIAL decay: a straight line on log-linear, finite xi")
say("  The paper quotes xi ~ 1.6a in the chiral phase. On an open cluster only a few distance")
say("  shells exist, so a fitted xi here is indicative and NOT a measurement of the paper's.")
say("")
say("⚠ AND AN OPEN CLUSTER HAS EDGES. Correlations from a reference site run into the boundary")
say("  within a couple of shells, which steepens the apparent decay. Any xi read off this is an")
say("  UPPER bound on the decay rate, not a value to compare against 1.6a numerically.")
close(CSV); close(TXT)
println("wrote to ", OUTDIR)
