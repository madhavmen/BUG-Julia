# WHICH KRYLOV DEPTH DOES `cbe_bug` NEED TO MATCH THE TDVPs ON A CYLINDER?
#
# On the chain, depth was the lever that mattered: m=0 -> m=3 bought 163x at L=12 and 8 ORDERS on
# OAT. The 4x3 cylinder hints that it is NOT the lever there -- `cbe_bug` at m=3 and at the
# package defaults returned the SAME error to five figures (1.0949e-02) while both TDVPs sat at
# 7.2309e-03. This file settles that by scanning `m` all the way to the cap instead of inferring
# it from two points.
#
# ⛔ THE SCAN MUST BE ABLE TO SAY "NO m IS ENOUGH". A depth scan that stops at 3 and reports
# "depth does not help" cannot distinguish a genuine plateau from a scan that stopped too early --
# the exact mistake that hid the Krylov floor on the chain for so long, in reverse. This goes to
# `krylov_basis = 30`, the shipped cap, with `krylov_tol = 0` so nothing stops it early.
#
# ⛔ TWO REGIMES, BECAUSE THEY ASK DIFFERENT QUESTIONS.
#   FULL RANK   (`maxdim` above `2^(L/2)`): nothing truncates, so the residual error is PURE
#               time-integration/projection error. If `cbe_bug` cannot match the TDVPs here, the
#               gap is in the scheme's composition and no amount of rank or depth will close it.
#   RANK-LIMITED(`maxdim` below the ceiling): the regime anyone actually runs 2D in, where the
#               basis has to choose. A scheme can lose the first and win the second.
#
# ⚠ THE TDVP BASELINES ARE RE-RUN IN EACH REGIME rather than quoted across them: their error also
# moves with `maxdim`, and comparing a capped `cbe_bug` against an uncapped TDVP would be a
# comparison of two different problems.
#
# Run:  julia --project=. benchmarks/cylinder_depth_scan.jl [Lx] [Ly] [t_max]

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const LX   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const LY   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const TMAX = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.4
const DT   = 0.05
const L    = LX * LY
const MI   = 16
const DEPTHS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 30)

const OUT = joinpath(@__DIR__, "results", @sprintf("cylinder_depth_%dx%d.txt", LX, LY))
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:U1)
const BONDS = square_cylinder_bonds(LX, LY)
const JM    = square_cylinder_couplings(LX, LY)
const W     = square_cylinder_mpo(LX, LY)
const PSI0  = dimer_state(L)

function bond_profile(psi)
    n2 = max(norm(copy(psi))^2, eps())
    return [begin
                Jm = zeros(Float64, L, L); Jm[i, j] = 1.0
                real(mpo_energy(copy(psi), pair_mpo(L, Jm, spin_vertices(L)))) / n2
            end for (i, j) in BONDS]
end

function pair_profile_dense(v, states, idx, bonds)
    bit(b, j) = (b >> (j - 1)) & 1
    nrm = real(dot(v, v))
    out = zeros(Float64, length(bonds))
    for (n, (i, j)) in pairs(bonds)
        acc = 0.0 + 0im
        for (k, b) in enumerate(states)
            si, sj = bit(b, i), bit(b, j)
            acc += conj(v[k]) * v[k] * (si == sj ? 0.25 : -0.25)
            if si != sj
                bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                acc += conj(v[idx[bb]]) * v[k] * 0.5
            end
        end
        out[n] = real(acc) / nrm
    end
    return out
end

Hs, STATES, IDX = pairs_sparse(L, JM)
const REF = pair_profile_dense(
    expv_sparse(Hs, ComplexF64(-im * TMAX), dimer_vector(L, STATES, IDX); m = 30, tol = 1e-13),
    STATES, IDX, BONDS)

say(@sprintf("Heisenberg on a %dx%d cylinder -- %d sites, dt=%g, T=%g", LX, LY, L, DT, TMAX))
say(@sprintf("exact reference: %d states;  full rank at this L = %d", length(STATES), 1 << (L ÷ 2)))

function run(step!)
    psi, kry, secs = copy(PSI0), 0, 0.0
    for _ in 1:round(Int, TMAX / DT)
        t0 = time_ns()
        info = step!(psi, ComplexF64(-im * DT))
        secs += (time_ns() - t0) / 1e9
        hasproperty(info, :krylov_dims) && (kry += info.krylov_dims)
    end
    return (err = maximum(abs.(bond_profile(psi) .- REF)),
            states = maximum(state_bond_dims(psi)), kry = kry, secs = secs)
end

for (regime, cap) in (("FULL RANK (nothing truncates -- pure time-integration error)", 1 << (L ÷ 2)),
                      ("RANK-LIMITED (the regime 2D is actually run in)",              1 << (L ÷ 2) ÷ 4))
    say("")
    say("="^104)
    say(@sprintf("%s   maxdim = %d", regime, cap))
    say("="^104)

    t2 = run((p, t) -> tdvp2_step!(p, W, t; maxdim = cap, trunc_thresh = 1e-10, maxiter = MI))
    t1 = run((p, t) -> tdvp_cbe1s_step!(p, W, t; maxdim = cap, trunc_thresh = 1e-10, maxiter = MI))
    say(@sprintf("  %-24s err=%.4e  states=%3d  krylov=%6s  %6.1fs   <- THE BAR",
                 "tdvp2 (2-site)", t2.err, t2.states, "-", t2.secs))
    say(@sprintf("  %-24s err=%.4e  states=%3d  krylov=%6s  %6.1fs   <- THE BAR",
                 "tdvp_cbe1s (1-site)", t1.err, t1.states, "-", t1.secs))
    bar = min(t2.err, t1.err)
    say("  " * "-"^100)

    hit = nothing
    for m in DEPTHS
        r = run((p, t) -> cbe_bug_step!(p, W, t; exact = true, krylov_basis = m,
                                        krylov_tol = 0.0, maxdim = cap,
                                        trunc_thresh = 1e-10, maxiter = MI))
        mark = r.err <= bar * 1.05 ? "  <== MATCHES THE BAR" : ""
        hit === nothing && r.err <= bar * 1.05 && (hit = (m, r))
        say(@sprintf("  cbe_bug m=%-3d            err=%.4e  states=%3d  krylov=%6d  %6.1fs  %.2fx bar%s",
                     m, r.err, r.states, r.kry, r.secs, r.err / bar, mark))
    end
    say("")
    if hit === nothing
        say(@sprintf("  ⛔ NO DEPTH UP TO %d REACHES THE BAR. The gap is not a basis-depth problem:",
                     maximum(DEPTHS)))
        say("     at full rank nothing is truncated, so what is left is the SCHEME'S COMPOSITION")
        say("     (one Galerkin solve at the root against a per-bond sweep), and depth cannot fix it.")
    else
        m, r = hit
        say(@sprintf("  ✅ m = %d reaches the bar, at %d operator applications and %.1fs against %.1fs.",
                     m, r.kry, r.secs, min(t2.secs, t1.secs)))
    end
end
close(IO_)
