# "Refused with room to spare" -- the per-bond probe that located the CBE rank stall.
#
#   sbatch --job-name=edgeprobe scripts/run_julia.sbatch benchmarks/probe_edge_refusal.jl
#
# WHAT IT MEASURES. Per bond: `r`, `room_l`, `room_r`, `n_new` and the resulting dimension, so
# a bond that DECLINES TO EXPAND WHILE ROOM REMAINS is visible at a glance. That signature is
# what the whole diagnosis turned on: one blocked narrow bond staircase-caps every bond inward
# (bond 2 <= 2, bond 3 <= 4, centre <= 4), which reproduced the entire stalled profile
# [1,1,2,4,2,1,1] against 2-site TDVP's [2,4,6,8,6,4,2] at L=8.
#
# RESOLVED -- and the original header of this file named the wrong culprits, so read this
# instead. The cause was TWO defects in the selection, both letting one side of a bond veto
# the other (fixed in `cbe.jl`, commit "validate the selection in the gate path"):
#
#   1. `sector_graded_sketch` DROPPED the probe-width shortfall. `_alloc_columns` caps each
#      sector at its own dimension, so a half with a small subspace returned fewer than `npre`
#      columns -- and `npre` is the expansion's ceiling, since the candidates ARE the sketch's
#      columns. Near an edge the opposite frame's complement is 0- or 1-dimensional, so the
#      probe collapsed and the expansion declined with room to spare.
#   2. The final selection COUPLED the two sides: `Nkeep = min(dex_l, dex_r)` throttled each
#      side to the OTHER's headroom, and the `Empty` verdict was GLOBAL, so a left frame that
#      already spanned its own image made `norm(UMU) < 1e-11` fire and returned nothing on
#      EITHER side.
#
# What this file's original header blamed, and which was NOT the cause: `stol_pre = 1e-10` and
# `stol_fnl = 1e-13` were never implicated (the tolerances are relative and fine), and the
# stale-environment hypothesis was checked against `CBE1SiQS.m` and REJECTED -- CBE feeds
# UNEXPANDED environments into the selection by design. The `norm(UMU)` gate WAS involved, but
# not because it was too tight: because the verdict it produced was global instead of per-side.
#
# STILL WORTH RUNNING as a regression probe: after any change to the selection, every bond with
# `room > 0` and a non-zero `P-perp H-Theta` should admit something. It is ~1 min against the
# ~10 min suite.
#
# CAVEAT ON THE PER-BOND TABLE. It calls `cbe_expand` directly against `right_env_stack(...)`,
# i.e. environments built from the unexpanded state. That is CORRECT rather than stale (see
# above), and it means the table shows what each bond admits in isolation. The end-to-end
# signal is `bond_dims` after the 20-step runs below, which go through the bond update itself.

using LinearAlgebra, Printf
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const L  = 8
const DT = 0.05

set_symmetry!(:U1)
h = xxz_chain(L; delta = 0.0)

"Expand every bond of `psi` at the given tolerances and report what each one admitted."
function probe(psi, label; stol_pre, stol_fnl)
    @printf("\n%s   (stol_pre=%.0e, stol_fnl=%.0e)\n", label, stol_pre, stol_fnl)
    @printf("  %4s %5s %7s %7s %7s %7s %9s\n",
            "bond", "r", "room_l", "room_r", "n_new", "dim->", "err_pre")
    for i in 1:(L - 1)
        p = copy(psi)
        canonical!(p, i)
        f = bond_frame(p, i)
        r = f.old_rank
        room_l = leg_dim(f.U0, 1) * leg_dim(f.U0, 2) - r
        room_r = leg_dim(f.V0, 2) * leg_dim(f.V0, 3) - r
        ex = cbe_expand(f, h, i,
                        left_env_stack(p, h; upto = i - 1),
                        right_env_stack(p, h; downto = i + 2);
                        dex = 0, stol_pre = stol_pre, stol_fnl = stol_fnl)
        @printf("  %4d %5d %7d %7d %7d %7d %9.2e%s\n",
                i, r, room_l, room_r, ex.n_new_l, leg_dim(ex.U_ex, 3), ex.err_pre,
                ex.n_new_l == 0 && room_l > 0 ? "   <-- REFUSED with room to spare" : "")
    end
    flush(stdout)
end

# A product state has nothing to expand at the edges yet, so warm up a little first --
# but only a little, or the question is no longer about early-time edge weight.
psi = domain_wall_state(L)
for _ in 1:4
    cbe_bug_bond_update(psi, h, -im * DT; maxdim = 64, trunc_thresh = 1e-12)
end
println("after 4 CBE-BUG steps:  bond_dims = ", bond_dims(psi))

probe(psi, "DEFAULT tolerances"; stol_pre = 1e-10, stol_fnl = 1e-13)
probe(psi, "LOOSENED tolerances"; stol_pre = 1e-14, stol_fnl = 1e-16)

# And the direct question: does loosening actually unstick the run?
for (sp, sf) in ((1e-10, 1e-13), (1e-14, 1e-16))
    q = domain_wall_state(L)
    for _ in 1:20
        cbe_bug_bond_update(q, h, -im * DT; maxdim = 64, trunc_thresh = 1e-12,
                            stol_pre = sp, stol_fnl = sf)
    end
    @printf("\n20 steps at stol_pre=%.0e: bond_dims = %s\n", sp, string(bond_dims(q)))
end

println("\nPROBE_DONE")

# ── fast invariant check, so this one job replaces a full suite run ──────────
# Only the two properties that can silently break: exactness once the manifold is the whole
# space, and U(1) charge conservation. Everything else is covered by tests/ when it matters.
println("\n=== fast invariants ===")
let Ls = 6
    set_symmetry!(:U1)
    hh = xxz_chain(Ls; delta = 0.0)
    a = domain_wall_state(Ls)
    for _ in 1:4; cbe_bug_bond_update(a, hh, -im * DT; maxdim = 64, trunc_thresh = 1e-14); end
    b = copy(a)
    for _ in 1:6
        cbe_bug_bond_update(a, hh, -im * DT; maxdim = 64, trunc_thresh = 1e-14)
        tdvp2_step!(b, hh, -im * DT; maxdim = 64, trunc_thresh = 1e-14)
    end
    e = maximum(abs.(magnetisation(copy(a)) .- magnetisation(copy(b))))
    q = copy(a)
    @printf("exact-at-full-rank vs tdvp2: %.3e  %s\n", e, e < 1e-8 ? "OK" : "*** BROKEN ***")
    @printf("total Sz: %.3e  %s\n", abs(total_sz(q) / norm(q)^2),
            abs(total_sz(q) / norm(q)^2) < 1e-10 ? "OK" : "*** BROKEN ***")
end
