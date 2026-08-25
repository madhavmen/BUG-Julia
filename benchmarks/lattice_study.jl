# SQUARE vs KAGOME: the three-knob matrix, three schemes, error / operations / wall clock.
#
# Two 2D lattices at comparable size, both with an EXACT reference:
#
#   square cylinder  4x4 = 16 sites,  Sz=0 sector 12870 states
#   kagome cylinder  2x3 = 18 sites,  Sz=0 sector 48620 states
#
# ⛔ KAGOME HAS NO 16-SITE CLUSTER. The unit cell is three sites, so the reachable sizes are 12,
# 18, 24. 18 is the closest to 16 and is still exactly propagable. Rounding a requested 16 down
# to 12 or up to 18 silently would mean reporting "kagome at 16 sites", which is not a lattice.
#
# ── ON THE FAIRNESS OF THE COMPARISON, which is subtler than it first looks ────────────────
#
# `cbe_bug` does NOT have `grow_iters` -- that knob lives on `cbe_lubich_sweep`. Both schemes
# already call the SAME `expv` Lanczos solver with the same `maxiter`, `tol` and `reorth`. What
# differs is where the Krylov content sits:
#
#   tdvp_cbe1s   CBE frame (ONE power of H) + an `expv` PER SITE (all powers, to `maxiter`)
#   cbe_bug      CBE frame + `krylov_basis` powers of H + ONE `expv` at the root
#
# So `tdvp_cbe1s` already carries MORE Krylov content per site, not less: the asymmetry runs the
# other way from the obvious reading. Deepening its frame as well would make it something other
# than CBE-TDVP, so the honest parity check is not to change either algorithm but to verify that
# `maxiter` is not handicapping one of them -- `cbe_bug`'s single root solve acts on a LARGER
# space than a one-site solve, so a shared `maxiter` is exactly where a hidden handicap would
# live. Phase `maxiter` below scans it for both.
#
# ⚠ AND WALL CLOCK IS REPORTED SEPARATELY FROM OPERATOR COUNT BECAUSE THEY DISAGREE AT SCALE.
# MEASURED on the 8x8 cylinder: `cbe_bug` used 19x FEWER operator applications and took 1.91x
# LONGER than tdvp2. `krylov` cannot see the CBE expansion's SVD/QR work or the O(L) frames held
# alive to the root solve. Any cost claim must quote both, and at these sizes wall clock is the
# one that decides.
#
# Run:  julia --project=. benchmarks/lattice_study.jl [phase] [lattice]
#         phase   = matrix | schemes | maxiter | physics
#         lattice = square | kagome | both

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const PHASE = length(ARGS) >= 1 ? ARGS[1] : "schemes"
const WHICH = length(ARGS) >= 2 ? ARGS[2] : "both"
const DT    = 0.05
const TMAX  = 0.3
const MI    = 16
const CAP   = 32          # binds hard on both lattices: full rank is 256 (square) / 512 (kagome)

const OUT = joinpath(@__DIR__, "results", "lattice_study_$(PHASE)_$(WHICH).txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

"Everything one lattice needs: operator, bonds, exact reference, and a start state."
function setup(name)
    set_symmetry!(:U1)
    if name == "square"
        LX, LY = 4, 4
        bonds = square_cylinder_bonds(LX, LY)
        Jm    = square_cylinder_couplings(LX, LY)
        L     = LX * LY
        tri   = NTuple{3, Int}[]
    else
        LX, LY = 2, 3
        bonds = kagome_bonds(LX, LY)
        Jm    = kagome_cylinder_couplings(LX, LY)
        L     = 3 * LX * LY
        tri   = kagome_triangles(LX, LY)
    end
    W   = pair_mpo(L, Jm, spin_vertices(L))
    ops = [begin
               A = zeros(Float64, L, L); A[i, j] = 1.0
               pair_mpo(L, A, spin_vertices(L))
           end for (i, j) in bonds]
    Hs, states, idx = pairs_sparse(L, Jm)
    v0 = dimer_vector(L, states, idx)
    return (name = name, L = L, LX = LX, LY = LY, W = W, bonds = bonds, ops = ops, tri = tri,
            Jm = Jm, Hs = Hs, states = states, idx = idx, v0 = v0, psi0 = dimer_state(L))
end

profile(S, psi) = (n2 = max(norm(copy(psi))^2, eps());
                   [real(mpo_energy(copy(psi), op)) / n2 for op in S.ops])

function dense_profile(S, v)
    bit(b, j) = (b >> (j - 1)) & 1
    nrm = real(dot(v, v))
    out = zeros(Float64, length(S.bonds))
    for (n, (i, j)) in pairs(S.bonds)
        acc = 0.0 + 0im
        for (k, b) in enumerate(S.states)
            si, sj = bit(b, i), bit(b, j)
            acc += conj(v[k]) * v[k] * (si == sj ? 0.25 : -0.25)
            if si != sj
                bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                acc += conj(v[S.idx[bb]]) * v[k] * 0.5
            end
        end
        out[n] = real(acc) / nrm
    end
    return out
end

reference(S, t) = dense_profile(S, t == 0 ? copy(S.v0) :
                                expv_sparse(S.Hs, ComplexF64(-im * t), S.v0; m = 30, tol = 1e-13))

"Run one stepper. Reports the three axes the study follows: error, operations, wall clock."
function run(S, step!, ref; nsteps = round(Int, TMAX / DT))
    psi, kry, secs, efnl = copy(S.psi0), 0, 0.0, 0.0
    e0 = real(mpo_energy(copy(psi), S.W)) / max(norm(psi)^2, eps())
    for _ in 1:nsteps
        t0 = time_ns()
        info = step!(psi, ComplexF64(-im * DT))
        secs += (time_ns() - t0) / 1e9
        hasproperty(info, :krylov_dims) && (kry += info.krylov_dims)
        hasproperty(info, :err_fnl) && (efnl = max(efnl, info.err_fnl))
    end
    e1 = real(mpo_energy(copy(psi), S.W)) / max(norm(psi)^2, eps())
    return (err = maximum(abs.(profile(S, psi) .- ref)), kry = kry, secs = secs,
            states = maximum(state_bond_dims(psi)), dE = abs(e1 - e0), efnl = efnl, psi = psi)
end

bug(S; kw...) = (p, t) -> cbe_bug_step!(p, S.W, t; exact = true, maxdim = CAP,
                                        trunc_thresh = 1e-10, maxiter = MI, kw...)

const LATTICES = WHICH == "both" ? ("square", "kagome") : (WHICH,)

# ══ PHASE `schemes`: the headline comparison on both lattices ═══════════════════════════════
if PHASE == "schemes"
    for nm in LATTICES
        S = setup(nm)
        ref = reference(S, TMAX)
        say("")
        say("="^104)
        say(@sprintf("%s cylinder %dx%d -- %d sites, %d bonds, maxdim=%d (full rank %d), T=%g",
                     uppercase(nm), S.LX, S.LY, S.L, length(S.bonds), CAP, 1 << (S.L ÷ 2), TMAX))
        say(@sprintf("exact reference: %d states in the Sz=0 sector", length(S.states)))
        say("="^104)
        say(@sprintf("  %-24s %12s %9s %9s %8s %11s", "scheme", "err", "krylov", "sec",
                     "states", "|dE|"))
        arms = [("tdvp2 (2-site)",  (p, t) -> tdvp2_step!(p, S.W, t; maxdim = CAP,
                                                          trunc_thresh = 1e-10, maxiter = MI)),
                ("tdvp_cbe1s",      (p, t) -> tdvp_cbe1s_step!(p, S.W, t; maxdim = CAP,
                                                               trunc_thresh = 1e-10, maxiter = MI)),
                ("cbe_bug m=0",     bug(S; krylov_basis = 0,  krylov_tol = 0.0)),
                ("cbe_bug m=3",     bug(S; krylov_basis = 3,  krylov_tol = 0.0)),
                ("cbe_bug tol=1e-8", bug(S; krylov_basis = 30, krylov_tol = 1e-8)),
                ("cbe_bug m=30",    bug(S; krylov_basis = 30, krylov_tol = 0.0))]
        bar = Inf
        for (anm, st) in arms
            r = run(S, st, ref)
            startswith(anm, "tdvp") && (bar = min(bar, r.err))
            say(@sprintf("  %-24s %12.4e %9s %9.1f %8d %11.2e", anm, r.err,
                         r.kry == 0 ? "-" : string(r.kry), r.secs, r.states, r.dE))
        end
        say(@sprintf("  the bar (best TDVP): %.4e", bar))
    end

# ══ PHASE `maxiter`: is a shared `maxiter` handicapping one scheme? ═════════════════════════
elseif PHASE == "maxiter"
    for nm in LATTICES
        S = setup(nm)
        ref = reference(S, TMAX)
        say("")
        say("="^104)
        say(@sprintf("%s -- DOES `maxiter` HANDICAP EITHER SCHEME? (cbe_bug's root solve acts on",
                     uppercase(nm)))
        say(" a LARGER space than a one-site solve, so a shared maxiter is where a hidden")
        say(" handicap would live. If both are flat in maxiter, the comparison is fair.)")
        say("="^104)
        say(@sprintf("  %-22s %6s %12s %9s %9s", "scheme", "maxiter", "err", "krylov", "sec"))
        for mi in (8, 16, 30, 45)
            for (anm, st) in (("tdvp2", (p, t) -> tdvp2_step!(p, S.W, t; maxdim = CAP,
                                                              trunc_thresh = 1e-10, maxiter = mi)),
                              ("tdvp_cbe1s", (p, t) -> tdvp_cbe1s_step!(p, S.W, t; maxdim = CAP,
                                                                        trunc_thresh = 1e-10,
                                                                        maxiter = mi)),
                              ("cbe_bug m=3", (p, t) -> cbe_bug_step!(p, S.W, t; exact = true,
                                                                      krylov_basis = 3,
                                                                      krylov_tol = 0.0,
                                                                      maxdim = CAP,
                                                                      trunc_thresh = 1e-10,
                                                                      maxiter = mi)))
                r = run(S, st, ref)
                say(@sprintf("  %-22s %6d %12.4e %9s %9.1f", anm, mi, r.err,
                             r.kry == 0 ? "-" : string(r.kry), r.secs))
            end
        end
    end

# ══ PHASE `matrix`: the three-knob cube, per lattice ════════════════════════════════════════
elseif PHASE == "matrix"
    for nm in LATTICES
        S = setup(nm)
        ref = reference(S, TMAX)
        say("")
        say("="^104)
        say(@sprintf("%s -- THREE-KNOB MATRIX: m x half-sweep split_cutoff x root-out trunc_thresh",
                     uppercase(nm)))
        say("="^104)
        for m in (0, 3, 30)
            say("")
            say(@sprintf("  --- m = %d %s---", m, m == 30 ? "(to the cap) " : ""))
            say(@sprintf("  %10s %s", "root \\ half", join([@sprintf("%14.0e", x)
                                                            for x in (1e-4, 1e-6, 1e-8, 1e-10)], "")))
            for tt in (1e-4, 1e-6, 1e-8, 1e-10)
                line = @sprintf("  %10.0e", tt)
                for sc in (1e-4, 1e-6, 1e-8, 1e-10)
                    r = run(S, (p, t) -> cbe_bug_step!(p, S.W, t; exact = true,
                                                       krylov_basis = m, krylov_tol = 0.0,
                                                       split_cutoff = sc, maxdim = CAP,
                                                       trunc_thresh = tt, maxiter = MI), ref)
                    line *= @sprintf("  %8.3e/%-3d", r.err, r.states)
                end
                say(line)
            end
        end
    end

# ══ PHASE `physics`: is the kagome state actually a spin liquid? ════════════════════════════
elseif PHASE == "physics"
    for nm in LATTICES
        S = setup(nm)
        say("")
        say("="^104)
        say(@sprintf("%s -- GROUND STATE BY IMAGINARY TIME, and what it looks like", uppercase(nm)))
        say("="^104)
        # imaginary time to the ground state, then the SIGNATURES
        psi = copy(S.psi0)
        for _ in 1:120
            cbe_bug_step!(psi, S.W, ComplexF64(-0.05); exact = true, krylov_basis = 3,
                          krylov_tol = 0.0, maxdim = 64, trunc_thresh = 1e-10, maxiter = MI)
            psi[psi.center] = to_concrete((1.0 / norm(psi)) * psi[psi.center])
        end
        e = real(mpo_energy(copy(psi), S.W)) / max(norm(psi)^2, eps())
        # the exact ground state of the same cluster, for comparison
        vals, _ = try
            (eigen(Matrix(S.Hs)).values, nothing)
        catch
            (Float64[], nothing)
        end
        prof = profile(S, psi)
        mag  = magnetisation(copy(psi)) ./ max(norm(copy(psi))^2, eps())
        say(@sprintf("  E/site (imaginary time)  %.8f", e / S.L))
        isempty(vals) || say(@sprintf("  E/site (exact ground)    %.8f   |diff| %.2e",
                                      minimum(vals) / S.L, abs(e - minimum(vals)) / S.L))
        say("")
        say("  ⛔ SPIN-LIQUID SIGNATURES (kagome) vs ORDER:")
        say(@sprintf("    max |<S^z_j>|            %.3e   %s", maximum(abs.(mag)),
                     maximum(abs.(mag)) < 1e-6 ? "NO static moment -- consistent with a liquid" :
                                                 "a static moment is present -- ORDERED"))
        say(@sprintf("    <S_i.S_j> mean           %.6f", sum(prof) / length(prof)))
        say(@sprintf("    <S_i.S_j> min / max      %.6f / %.6f", minimum(prof), maximum(prof)))
        rel = (maximum(prof) - minimum(prof)) / abs(sum(prof) / length(prof))
        say(@sprintf("    bond-energy SPREAD       %.3f  %s", rel,
                     rel < 0.15 ? "NEAR-UNIFORM -- consistent with a liquid, not a VBS" :
                                  "bonds SPLIT into strong/weak -- valence-bond-solid-like"))
        if !isempty(S.tri)
            tri_e = [sum(prof[findfirst(==(minmax(p, q)), S.bonds)]
                         for (p, q) in ((a, b), (b, c), (a, c))) for (a, b, c) in S.tri]
            say(@sprintf("    per-triangle energy      mean %.6f  spread %.3e",
                         sum(tri_e) / length(tri_e), maximum(tri_e) - minimum(tri_e)))
        end
        say("")
        say("  ⚠ AN OPEN 18-SITE CLUSTER IS NOT THE THERMODYNAMIC LIMIT. Edge sites are 2- and")
        say("    3-coordinate, so some bond-energy spread is GEOMETRY and not physics. These")
        say("    numbers pin that the lattice and the solver behave sensibly; they are not a")
        say("    claim about the kagome ground state.")
    end
else
    error("unknown phase $(repr(PHASE)) -- use schemes, maxiter, matrix or physics")
end
close(IO_)
