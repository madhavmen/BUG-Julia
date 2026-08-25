# HEISENBERG ON A SQUARE-LATTICE CYLINDER: cbe_bug vs 1-site CBE-TDVP vs 2-site TDVP.
#
# The last rung of Jan's model ladder, and the first genuinely 2D problem here. Everything else in
# this study is a chain; a cylinder reaches an MPS only through an ORDERING of its sites, so the
# ordering is part of the model and is pinned separately in `tests/sweeps/test_square_cylinder.jl`
# before any physics runs on it.
#
# ⛔ WHY THIS IS THE CASE THAT MATTERS, AND NOT JUST ANOTHER MODEL. Every chain measurement in
# this study came back with `err_fnl = 0.000e+00` -- the CBE selection discarding nothing, because
# at `growth = 2.0` the budget never binds on a chain. That single fact is why Jan's point 2
# (tolerance-driven selection) and point 4 (expansion inside the Lanczos recursion) BOTH returned
# null: there was nothing for either to improve. On an `Lx x Ly` cylinder a cut severs `Ly` bonds,
# so the rank requirement grows like `exp(Ly)` and the state is rank-limited from the first step.
# `err_fnl` is therefore a FIRST-CLASS COLUMN below: if it is still zero here, points 2 and 4 are
# null in 2D as well; if it is finally non-zero, this is the regime they were designed for.
#
# ⛔ THE REFERENCE, AND WHERE IT STOPS. At `L <= 16` the `Sz = 0` sector is at most 12870 states,
# so `expv_sparse` gives an EXACT answer and the comparison is a real accuracy comparison. Past
# that there is no reference at all (4x6 is 2^24, sector 2.7M), and the file reports CONSERVATION
# and CROSS-SCHEME AGREEMENT instead -- never "the error". Which mode is running is printed, so a
# conservation number is never mistaken for an accuracy number.
#
# ⚠ ENERGY DRIFT IS NOT AN ACCURACY CHECK. On this repo's own models the two orderings disagree
# outright -- a basis-only arm once conserved energy to 7.4e-15 while sitting 12x further from the
# exact state than `tdvp2`. Rank the schemes by `err` where there is a reference, never by `dE`.
#
# ⚠ THE OBSERVABLE IS THE CYLINDER'S OWN BOND ENERGIES `<S_i . S_j>` over its 2D bonds, NOT the
# Sz profile: the dimer start has `<S^z_j> = 0` on every site and SU(2)-symmetric evolution keeps
# it there, so an Sz-based comparison would report a flawless 0.000e+00 for every scheme at every
# tolerance -- a measurement that cannot fail.
#
# Run:  julia --project=. benchmarks/su2_cylinder.jl [Lx] [Ly] [sym] [t_max] [maxdim]

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const LX   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const LY   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SYM  = length(ARGS) >= 3 ? Symbol(ARGS[3])     : :U1
const TMAX = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.4
const CAP  = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 128
const DT   = 0.05
const L    = LX * LY
const MI   = 16

const OUT = joinpath(@__DIR__, "results",
                     @sprintf("su2_cylinder_%dx%d_%s.txt", LX, LY, SYM))
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(SYM)
const BONDS = square_cylinder_bonds(LX, LY)
const JM    = square_cylinder_couplings(LX, LY)
const W     = square_cylinder_mpo(LX, LY)
const PSI0  = iseven(L) ? dimer_state(L) : neel_state(L)

"`<S_i . S_j>` on the cylinder's own bonds -- one single-entry `pair_mpo` each, so it is the same
operator in every symmetry mode (the trick `test_symmetry_parity.jl` uses)."
function bond_profile(psi)
    n2 = max(norm(copy(psi))^2, eps())
    return [begin
                Jm = zeros(Float64, L, L); Jm[i, j] = 1.0
                real(mpo_energy(copy(psi), pair_mpo(L, Jm, spin_vertices(L)))) / n2
            end for (i, j) in BONDS]
end

"""
`<v| S_i . S_j |v>` on ARBITRARY pairs, for the dense reference.

`bond_energies_dense` only does chain-adjacent bonds; a cylinder's leg and wrap bonds are not
adjacent under the snake, so it cannot be reused.
"""
function pair_profile_dense(v, states, idx, bonds)
    bit(b, j) = (b >> (j - 1)) & 1
    nrm = real(dot(v, v))
    out = zeros(Float64, length(bonds))
    for (n, (i, j)) in pairs(bonds)
        acc = 0.0 + 0im
        for (k, b) in enumerate(states)
            si, sj = bit(b, i), bit(b, j)
            acc += conj(v[k]) * v[k] * (si == sj ? 0.25 : -0.25)      # S^z S^z
            if si != sj                                              # flip-flop
                bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                acc += conj(v[idx[bb]]) * v[k] * 0.5
            end
        end
        out[n] = real(acc) / nrm
    end
    return out
end

const HAVE_REF = L <= 16 && iseven(L)
say(@sprintf("Heisenberg %s on a %dx%d cylinder -- %d sites, %d bonds, dt=%g, T=%g, maxdim=%d",
             SYM, LX, LY, L, length(BONDS), DT, TMAX, CAP))
say(@sprintf("MPO virtual dims: max %d  (tracks Ly=%d, not Lx=%d)",
             maximum(mpo_virtual_dims(W)), LY, LX))
say(@sprintf("start: %s", iseven(L) ? "dimer (S=0, SU(2)-representable)" : "Neel"))

REF = Float64[]
if HAVE_REF
    set_symmetry!(:U1)
    Hs, states, idx = pairs_sparse(L, JM)
    v0 = dimer_vector(L, states, idx)
    vt = expv_sparse(Hs, ComplexF64(-im * TMAX), v0; m = 30, tol = 1e-13)
    global REF = pair_profile_dense(vt, states, idx, BONDS)
    set_symmetry!(SYM)
    say(@sprintf("EXACT reference: %d states in the Sz=0 sector -- this IS an accuracy comparison",
                 length(states)))
else
    say("NO exact reference at this size -- conservation and cross-scheme agreement only")
end
say("")

"One scheme, one configuration."
function arm(step!; label)
    psi, kry, secs, efnl = copy(PSI0), 0, 0.0, 0.0
    e0 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    for _ in 1:round(Int, TMAX / DT)
        t0 = time_ns()
        info = step!(psi, ComplexF64(-im * DT))
        secs += (time_ns() - t0) / 1e9
        hasproperty(info, :krylov_dims) && (kry += info.krylov_dims)
        hasproperty(info, :err_fnl) && (efnl = max(efnl, info.err_fnl))
    end
    e1 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    prof = bond_profile(psi)
    return (label = label, prof = prof,
            err = HAVE_REF ? maximum(abs.(prof .- REF)) : NaN,
            dE = abs(e1 - e0), states = maximum(state_bond_dims(psi)),
            mult = maximum(bond_dims(psi)), kry = kry, secs = secs, efnl = efnl,
            nrm = norm(psi))
end

const ARMS = [
    ("tdvp2 (2-site)",        (p, t) -> tdvp2_step!(p, W, t; maxdim = CAP,
                                                    trunc_thresh = 1e-10, maxiter = MI)),
    ("tdvp_cbe1s (1-site)",   (p, t) -> tdvp_cbe1s_step!(p, W, t; maxdim = CAP,
                                                         trunc_thresh = 1e-10, maxiter = MI)),
    ("cbe_bug defaults",      (p, t) -> cbe_bug_step!(p, W, t; exact = true, maxdim = CAP,
                                                      trunc_thresh = 1e-10, maxiter = MI)),
    ("cbe_bug m=3",           (p, t) -> cbe_bug_step!(p, W, t; exact = true, krylov_basis = 3,
                                                      krylov_tol = 0.0, maxdim = CAP,
                                                      trunc_thresh = 1e-10, maxiter = MI)),
    ("cbe_bug m=0",           (p, t) -> cbe_bug_step!(p, W, t; exact = true, krylov_basis = 0,
                                                      krylov_tol = 0.0, maxdim = CAP,
                                                      trunc_thresh = 1e-10, maxiter = MI)),
]

say("="^112)
say("THREE SCHEMES ON THE CYLINDER" * (HAVE_REF ? "  (err = max |dSS| against the exact sector)" :
                                                  "  (no reference -- read dE and agreement)"))
say("="^112)
say(@sprintf("  %-22s %12s %11s %8s %7s %9s %11s %8s",
             "scheme", "err", "|dE|", "states", "mult", "krylov", "err_fnl", "sec"))
results = map(ARMS) do (nm, st)
    r = arm(st; label = nm)
    say(@sprintf("  %-22s %12s %11.2e %8d %7d %9s %11s %8.1f",
                 nm, HAVE_REF ? @sprintf("%.4e", r.err) : "  --",
                 r.dE, r.states, r.mult,
                 r.kry == 0 ? "-" : string(r.kry),
                 isnan(r.efnl) ? "-" : @sprintf("%.3e", r.efnl), r.secs))
    r
end

say("")
say("="^112)
say("DOES THE BUDGET BIND HERE? -- and FIRST, was the run rank-starved at all?")
say("="^112)
# ⛔ `err_fnl == 0` IS ONLY EVIDENCE IF THE RANK WAS ACTUALLY LIMITED. This repo's own rule
# (USAGE.md section 6): a knob is inert unless the run is rank-starved. At `L` sites the middle
# bond cannot exceed `2^(L/2)`, so if `states` sits BELOW both that ceiling and `maxdim`, nothing
# was ever truncated and `err_fnl = 0` means "there was nothing to discard" -- NOT "the selection
# is inert". Reporting the second conclusion from the first is the mistake this block prevents.
const FULLRANK = 1 << (L ÷ 2)
const CEIL = min(FULLRANK, CAP)
say(@sprintf("  full rank at this L = %d,  maxdim = %d  ->  binding ceiling = %d",
             FULLRANK, CAP, CEIL))
bound = all(r -> r.states >= CEIL, results)
if !bound
    say("  ⛔ NOT RANK-STARVED: at least one arm ended below the ceiling, so this run says")
    say("     NOTHING about whether the budget binds. Re-run with a smaller maxdim or a bigger")
    say("     cylinder before reading the err_fnl column as evidence.")
else
    say("  ✅ rank-starved: every arm reached the ceiling, so err_fnl is meaningful here.")
end
for r in results
    r.kry == 0 && continue
    verdict = !bound ? "(inconclusive -- not rank-starved)" :
              r.efnl > 1e-12 ? "NON-ZERO: the selection IS discarding weight here" :
                               "zero even when rank-starved -- the budget does not bind"
    say(@sprintf("  %-22s states=%3d/%3d  err_fnl = %.3e   %s",
                 r.label, r.states, CEIL, r.efnl, verdict))
end
say("")
say("  Every chain measurement in this study had err_fnl == 0.000e+00, which is why Jan's")
say("  points 2 and 4 came back null -- but ONLY the rank-starved case can settle that.")

# ── cross-scheme agreement, which is the only accuracy statement available without a reference ──
say("")
say("="^112)
say("CROSS-SCHEME AGREEMENT  max |profile_a - profile_b|  (schemes must agree even with no reference)")
say("="^112)
for a in 1:length(results), b in (a + 1):length(results)
    say(@sprintf("  %-22s vs %-22s  %.3e", results[a].label, results[b].label,
                 maximum(abs.(results[a].prof .- results[b].prof))))
end
close(IO_)
