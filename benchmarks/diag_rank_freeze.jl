# WHERE DOES THE EXPANDED BOND GO? A four-part isolation.
#
#   sbatch --job-name=diagfrz scripts/run_julia.sbatch benchmarks/diag_rank_freeze.jl
#
# Calibration showed CBE-BUG's L=18 error is IDENTICAL at dt = 0.1 / 0.05 / 0.025
# (2.3314e-01, maxbond 2 in all three) while 2-site TDVP is cleanly second order in the
# same harness. A dt-independent error is not an integration error -- the rank is pinned.
#
# The suspect is that the expansion's new directions carry EXACTLY ZERO weight (the core is
# zero-padded on them, M = [I;0]), so any rank-revealing SVD is entitled to drop them --
# and `move_right!`/`move_left!` both call `svd(..., cutoff = 0.0)`, which is called again
# at every bond of both sweeps via `canonical!`. If Telum drops zero blocks at cutoff 0,
# every expansion is undone by the very next gauge move, INCLUDING the whole charge sectors
# CBE opened. That would look exactly like a frozen rank.
#
#   (A) does svd(cutoff=0.0) preserve a bond with zero-weight directions?
#   (B) does cbe_expand actually widen the bond, and in which SECTORS?
#   (C) does the widening survive one canonical! call?
#   (D) which sectors does the true state need at that bond (2-site TDVP as reference),
#       and does the expansion offer them?
#
# (D) is the symmetric-blocks question: it is possible for the dimension bookkeeping to
# look right while the sectors offered are useless, so sectors are printed, not just dims.

# ⚠ PORTED from the REMOVED `cbe_lubich_sweep` to `cbe_bug_step!` (2026-08-29). The two are
# different algorithms -- notably `grow_iters` accumulates here instead of re-ranking -- so
# numbers recorded before that date are NOT comparable to what this file now produces.
using LinearAlgebra, Printf
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using Telum

const L  = 8
const DT = 0.05
const C  = L ÷ 2

set_symmetry!(:U1)
h = xxz_chain(L; delta = 0.0)

secs(t, l) = [(q, d) for (q, d) in t.spaces[l]]
fmt(s) = "{" * join(("$(q):$(d)" for (q, d) in s), ", ") * "}"

println("# L=$L XX U(1), centre bond c=$C, dt=$DT")
println()

# ── (A) does a cutoff-0 SVD keep zero-weight directions? ─────────────────────
# Built without CBE at all, so this isolates Telum's behaviour from the expansion's.
println("=== (A) does svd(cutoff=0.0) keep exactly-zero singular directions? ===")
psi = domain_wall_state(L)
canonical!(psi, C)
before = bond_dims(psi)
# widen bond C by hand with a zero column: pad psi[C]'s right leg via oplus against a
# zero-weight partner is what the expansion does, so mimic it with cbe_expand's own frame.
f = bond_frame(psi, C)
ex = cbe_expand(f, h, C,
                left_env_stack(psi, h; upto = C - 1),
                right_env_stack(psi, h; downto = C + 2); dex = 0)
@printf("bond %d: frame U0 leg = %d, expanded U_ex leg = %d  (n_new_l = %d)\n",
        C, leg_dim(f.U0, 3), leg_dim(ex.U_ex, 3), ex.n_new_l)
@printf("  U0    sectors %s\n", fmt(secs(f.U0, 3)))
@printf("  U_ex  sectors %s\n", fmt(secs(ex.U_ex, 3)))
flush(stdout)

# ── (B)+(C) absorb the expansion, then move the gauge and see what survives ──
println("\n=== (B)/(C) does the widened bond survive a canonical! call? ===")
RSVDCBEBondUpdate._absorb_left!(psi, C, ex.U_ex)
@printf("after _absorb_left!:      bond_dims = %s\n", string(bond_dims(psi)))
@printf("  psi[%d] right-leg sectors %s\n", C, fmt(secs(psi[C], 3)))
flush(stdout)
canonical!(psi, 1)
@printf("after canonical!(psi, 1): bond_dims = %s\n", string(bond_dims(psi)))
@printf("  psi[%d] right-leg sectors %s\n", C, fmt(secs(psi[C], 3)))
println("  ^ if this collapsed back, the gauge move is eating the expansion.")
flush(stdout)

# ── (D) what does the TRUE state need at that bond? ──────────────────────────
println("\n=== (D) sectors the true state uses at bond $C, from 2-site TDVP ===")
ref = domain_wall_state(L)
for k in 1:20
    tdvp2_step!(ref, h, -im * DT; maxdim = 64, trunc_thresh = 1e-12)
    if k in (1, 2, 5, 10, 20)
        canonical!(ref, C)
        @printf("t=%.2f  bond_dims = %-26s  bond %d sectors %s\n",
                k * DT, string(bond_dims(ref)), C, fmt(secs(ref[C], 3)))
        flush(stdout)
    end
end

# and what CBE offers on that same evolved state -- the honest sector A/B
println("\n=== CBE expansion offered ON the TDVP2-evolved state (same bond) ===")
for i in (C - 1, C, C + 1)
    canonical!(ref, i)
    fi = bond_frame(ref, i)
    exi = cbe_expand(fi, h, i,
                     left_env_stack(ref, h; upto = i - 1),
                     right_env_stack(ref, h; downto = i + 2); dex = 0)
    @printf("bond %d: U0 %d -> U_ex %d (n_new %d)\n  U0   %s\n  U_ex %s\n",
            i, leg_dim(fi.U0, 3), leg_dim(exi.U_ex, 3), exi.n_new_l,
            fmt(secs(fi.U0, 3)), fmt(secs(exi.U_ex, 3)))
    flush(stdout)
end

# ── the per-step trace, for the record ───────────────────────────────────────
println("\n=== per-step CBE-BUG trace (expanded is PRE-truncation) ===")
psi3 = domain_wall_state(L)
for k in 1:8
    info = cbe_bug_step!(psi3, h, -im * DT; maxdim = 64, trunc_thresh = 1e-12)
    @printf("%2d expanded=%-22s kept=%-22s centre=%d n_new=%d\n", k,
            string(info.expanded), string(bond_dims(psi3)), info.root_rank,
            sum(info.n_new))
    flush(stdout)
end

println("\nDIAG_DONE")
