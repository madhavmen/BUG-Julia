# The requested configuration: RSVD-CBE bond update whose growth passes feed the PREVIOUS
# EXPANDED BASIS forward as the probe, at comp_ratio = 0.5, with a PLAIN Lanczos S step
# limited to 5 Krylov vectors.
#
# That configuration already exists in the code and needs no new scheme -- it is
#
#     cbe_bond_update(f, gate, tau; s_iters = n > 1, maxiter = 5, comp_ratio = 0.5)
#
# because `s_iters > 1` re-sketches each pass from `_reframe(f, U_aug, S_new, V_aug)`, i.e.
# from the basis the previous pass expanded, and the S step in that branch is `expv(...;
# maxiter)` -- an ordinary Lanczos. So what is open is not whether it can be built but what it
# COSTS and BUYS, which is what this file measures.
#
# WHY maxiter MATTERS HERE, and why it is scanned rather than assumed. The S step is a Galerkin
# solve in a basis of dimension `aug_k x aug_l`. Capping the Lanczos at 5 vectors is only free
# if 5 vectors already resolve `exp(tau*H_eff)` in that basis; if the basis is large and tau is
# not small, truncating the Krylov space is an error source that has nothing to do with CBE and
# would be mis-attributed to it. The `maxiter = 30` column is the control that separates the
# two -- if a row moves between the two columns, the cap is binding and the 5 is doing harm.
#
# TWO MEASUREMENTS, deliberately:
#
#   (A) ONE BOND, against the exact two-site propagator (`expv` on the block directly, no
#       truncation). This isolates the bond update with nothing else in the loop -- no Trotter
#       error, no sweep, no accumulated truncation. It is the only place a scheme's own
#       accuracy is visible.
#   (B) THE FULL CHAIN, XX domain wall against the analytic free-fermion profile. This is what
#       the method is actually for, and it is where a per-bond win can still vanish.
#
# Reported per row: error, the rank the expansion reached, and `n_matvec` -- the COUNTED number
# of operator applications, not a timing. Timings on this cluster are contended and have
# produced non-monotonic nonsense before (see docs/l18 handoff §7); matvec count is the honest
# cost axis and is exactly reproducible.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf, Random
using LurCGT, Telum          # to_concrete / contract live here
include("../tests/common/free_fermion.jl")

const DT    = 0.05
const TAU   = -im * DT
const CR    = 0.5          # comp_ratio, as specified
const ITERS = (1, 2, 3, 5) # growth passes; 1 = the current default (single expansion)
const MITER = (5, 30)      # S-step Lanczos cap: 5 as specified, 30 as the control
# GROWTH IS SCANNED, and must be. `cbe_core.jl` states the iterated scheme is flat at the
# DEFAULT growth = 2.0 because the budget already saturates the room (room = d*chi - r -> r,
# budget -> r for d = 2), so the first pass takes the whole local space and later passes have
# nothing to find. Reporting "reusing the previous basis does nothing" from growth = 2.0 alone
# would therefore be measuring the saturation, not the scheme.
const GROWTHS = (1.25, 2.0)

warm(L; n = 4, dt = 0.05, maxdim = 32) = begin
    p = domain_wall_state(L)
    bond_update_bug!(p, bond_gates(p);
                     opts = BondUpdateOptions(dt = dt, n_steps = n, maxdim = maxdim))
    p
end

# ── (A) one bond, against the exact two-site propagator ──────────────────────
println("(A) ONE BOND vs the exact two-site propagator   L=8, dt=$DT, comp_ratio=$CR")
println("     the bond update alone: no Trotter error, no sweep, no accumulated truncation\n")
@printf("%-7s %-8s %-8s  %-12s  %-6s  %-8s\n", "growth", "s_iters", "maxiter", "rel err", "rank", "n_matvec")
psi = warm(8)
for i in (4,)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    gate = heisenberg_bond_gate(f.site_l, f.site_r)
    tl, tr = f.site_l.itags, f.site_r.itags
    exact = expv(x -> apply_gate(gate, x, tl, tr), TAU, frame_theta(f);
                 hermitian = true, maxiter = 60, tol = 1e-16)
    for g in GROWTHS, mi in MITER, si in ITERS
        r = cbe_bond_update(f, gate, TAU; maxdim = 400, trunc_thresh = 1e-16,
                            s_iters = si, maxiter = mi, comp_ratio = CR, growth = g,
                            rng = MersenneTwister(7))
        got = to_concrete(r.left_core * r.right_core)
        err = norm(to_concrete(got - exact)) / norm(exact)
        @printf("%-7.2f %-8d %-8d  %-12.3e  %-6d  %-8d\n", g, si, mi, err, r.aug_k, r.n_matvec)
    end
end

# ── (B) the full chain, XX vs the analytic free-fermion profile ──────────────
const L, NSTEPS, MAXDIM = 10, 25, 64
const TEND = DT * NSTEPS
xxg(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(L - 1)]
prof(p) = (q = copy(p); magnetisation(q) ./ norm(q)^2)

println("\n(B) FULL CHAIN  XX domain wall L=$L, dt=$DT, $NSTEPS steps (t=$TEND), maxdim=$MAXDIM")
println("     vs xx_free_fermion_sz (analytic)\n")
@printf("%-28s  %-12s  %-8s  %-10s\n", "config", "err", "maxbond", "norm")
want = xx_free_fermion_sz(L, TEND)

pk = domain_wall_state(L)
bond_update_bug!(pk, xxg(pk);
                 opts = BondUpdateOptions(dt = DT, n_steps = NSTEPS, maxdim = MAXDIM,
                                          trunc_thresh = 1e-14))
@printf("%-28s  %-12.3e  %-8d  %-10.7f\n", "bond_update_bug!",
        maximum(abs.(prof(pk) .- want)), maximum(bond_dims(pk)), norm(copy(pk)))

for g in GROWTHS, mi in MITER, si in ITERS
    pc = domain_wall_state(L)
    opts = CBEBugOptions(dt = DT, n_steps = NSTEPS, maxdim = MAXDIM,
                         trunc_thresh = 1e-14, comp_ratio = CR, growth = g,
                         s_iters = si, maxiter = mi)
    cbe_bond_update_bug!(pc, xxg(pc); opts = opts)
    @printf("%-28s  %-12.3e  %-8d  %-10.7f\n",
            "g=$g s_iters=$si maxiter=$mi", maximum(abs.(prof(pc) .- want)),
            maximum(bond_dims(pc)), norm(copy(pc)))
end
