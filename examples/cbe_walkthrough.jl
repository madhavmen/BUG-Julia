# One RSVD-CBE bond update, stage by stage, with every intermediate printed.
#
#     julia --project=. examples/cbe_walkthrough.jl
#
# This is the companion to `docs/rsvd_cbe_writeup.tex`, and the numbers in that document are
# this script's output rather than illustrative invention. It walks the `cbe_bond_update` path
# (bare two-site gate, no environments) because that is the shortest route to seeing the
# selection itself: every quantity below is either a shape, a singular-value list, or a
# residual, and none of it depends on the channel machinery.
#
# SMALL, WARM, AND NOT SATURATED -- all three matter, and the second and third were learned by
# getting them wrong:
#
#   * warm, because a cold product state has nothing to expand anywhere except the one bond
#     straddling the domain wall (elsewhere `P_perp H Theta = 0` exactly, and declining to
#     expand is correct), so a cold walkthrough is a page of zeros;
#   * not saturated, because `room = d*chi - r` is a hard ceiling and a bond already spanning
#     its whole local space has nothing to expand INTO. The centre cut of a short chain is
#     exactly that bond: at L=6 it caps at rank 8 and a warmed state reaches it, so every
#     selection stage returned empty. The bond is therefore CHOSEN by `most_expandable`
#     rather than assumed, and the choice is printed.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf, Random
using LurCGT, Telum          # to_concrete, contract, svd, oplus live here
const R = RSVDCBEBondUpdate

set_symmetry!(:U1)

const L      = 8
const DT     = 0.05
const MAXDIM = 16

hdr(s) = (println(); println("=" ^ 78); println(s); println("=" ^ 78))
shape(t) = "(" * join([string(leg_dim(t, k)) for k in 1:length(t.inds)], " x ") * ")"

"Charge sectors on one leg as `charge => dimension`, in Telum's order."
function sectors(t, leg)
    out = String[]
    for (q, d) in t.spaces[leg]
        c = try string(first(first(q))) catch; string(q) end
        push!(out, "$c => $d")
    end
    return "[" * join(out, ",  ") * "]"
end

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

# ── the state ────────────────────────────────────────────────────────────────
psi = domain_wall_state(L)
gates = xx_gates(psi)
cbe_bond_update_bug!(psi, gates;
                     opts = CBEBugOptions(dt = DT, n_steps = 3, maxdim = MAXDIM,
                                          trunc_thresh = 1e-14))

# CHOOSE AN EXPANDABLE BOND, do not assume one. `room = d*chi - r` is a hard ceiling: a bond
# already at the full dimension of its local space has NOTHING to expand into, and CBE
# correctly declines there. The centre cut of a short chain is exactly such a bond -- for L=6
# it caps at rank 8 and a warmed state reaches it -- so a walkthrough fixed at the centre
# would show every selection stage returning empty and would teach nothing about the method.
function most_expandable(psi)
    best, best_room = 0, -1
    for i in 1:(length(psi) - 1)
        canonical!(psi, i)
        f = bond_frame(psi, i)
        rl = leg_dim(f.U0, 1) * leg_dim(f.U0, 2) - f.old_rank
        rr = leg_dim(f.V0, 2) * leg_dim(f.V0, 3) - f.old_rank
        min(rl, rr) > best_room && ((best, best_room) = (i, min(rl, rr)))
    end
    return best
end

const BOND = most_expandable(psi)
canonical!(psi, BOND)
f = bond_frame(psi, BOND)
gate = gates[BOND]
theta0 = frame_theta(f)

hdr("0.  THE BOND, BEFORE ANYTHING HAPPENS")
@printf("chain          L = %d, XX, U(1), domain wall, 3 steps of dt = %.2f (t = %.2f)\n",
        L, DT, 3 * DT)
@printf("state          bond_dims = %s\n", string(bond_dims(psi)))
@printf("bond           (%d, %d) -- chosen as the most expandable, rank r = %d\n",
        BOND, BOND + 1, f.old_rank)
@printf("U0  %-16s  left frame   (link_l, site_l, bond)\n", shape(f.U0))
@printf("S0  %-16s  core\n", shape(f.S0))
@printf("V0  %-16s  right frame  (bond, site_r, link_r)\n", shape(f.V0))
@printf("Theta0 %-13s  the two-site block, U0*S0*V0\n", shape(theta0))
@printf("bond sectors on U0 leg 3: %s\n", string(sectors(f.U0, 3)))
@printf("left isometry defect  = %.2e   (U0 is an isometry)\n", left_isometry_defect(f.U0))

# The two local spaces the frames live in, and therefore the ROOM each side has to grow.
room_l = leg_dim(f.U0, 1) * leg_dim(f.U0, 2) - f.old_rank
room_r = leg_dim(f.V0, 2) * leg_dim(f.V0, 3) - f.old_rank
@printf("room_l = d*chi_l - r = %d*%d - %d = %d\n",
        leg_dim(f.U0, 1), leg_dim(f.U0, 2), f.old_rank, room_l)
@printf("room_r = d*chi_r - r = %d*%d - %d = %d\n",
        leg_dim(f.V0, 2), leg_dim(f.V0, 3), f.old_rank, room_r)

# ── 1. the budget ────────────────────────────────────────────────────────────
hdr("1.  THE BUDGET:  how many new directions may this bond propose?")
dmax = max(leg_dim(f.U0, 1), f.old_rank, leg_dim(f.V0, 3))
for growth in (1.1, 2.0)
    budget = max(ceil(Int, growth * dmax) - f.old_rank, 1)
    npre = ceil(Int, 1.2 * budget)
    @printf("growth = %.1f :  dmax = %d -> budget = ceil(%.1f*%d) - %d = %-3d  npre = %d\n",
            growth, dmax, growth, dmax, f.old_rank, budget, npre)
end
GROWTH = 2.0
budget = max(ceil(Int, GROWTH * dmax) - f.old_rank, 1)
npre = ceil(Int, 1.2 * budget)
@printf("\nusing growth = %.1f:  budget = %d, clipped per side to (dex_l, dex_r) = (%d, %d)\n",
        GROWTH, budget, min(budget, room_l), min(budget, room_r))
@printf("probe width npre = ceil(1.2*budget) = %d      (the 1.2 is the RSVD oversampling)\n",
        npre)

# ── 2. the random probe ──────────────────────────────────────────────────────
hdr("2.  THE RANDOM PROBE  Omega  (the R in RSVD)")
rng = MersenneTwister(0xC0FFEE)
OmR = sector_graded_sketch(f.V0, :right, npre; comp_ratio = 0.5, rng = rng)
@printf("Omega_R %-16s  a stand-in RIGHT frame with %d columns instead of r = %d\n",
        shape(OmR), leg_dim(OmR, 1), f.old_rank)
@printf("its columns are drawn PER CHARGE SECTOR: %s\n", string(sectors(OmR, 1)))
println("half the columns are random-then-projected-OUT of V0 (the complement half),")
println("half are random-then-projected-IN  (the isometry half, comp_ratio = 0.5).")

# ── 3. one application of H, sketched ────────────────────────────────────────
hdr("3.  ONE APPLICATION OF H, SKETCHED:  Y = (P_perp H Theta) Omega^dag")
Y = sketch_bond_left(f, gate, OmR)
@printf("Y   %-16s  the DISCARDED-space image of the gate's action, %d columns wide\n",
        shape(Y), leg_dim(Y, 3))
# The sketch probes the PROJECTOR, so the guard is against the explicitly projected
# action -- and Y is orthogonal to the frame for any probe, which is the sharper check.
Y_exact = sketch_bond_left(f, gate, f.V0)
HT = apply_gate(gate, theta0, f.site_l.itags, f.site_r.itags)
K1 = to_concrete(contract(to_concrete(perp_component(f.U0, HT)), (3, 4), f.V0', (2, 3)))
@printf("check: sketch with Omega = V0 equals P_perp(H*Theta) contracted:      %.2e\n",
        norm(to_concrete(Y_exact - K1)))
@printf("check: <U0 S0 | Y> vanishes -- the sketch lives in the complement:    %.2e\n",
        abs(tensor_inner(to_concrete(f.U0 * f.S0), Y)))
@printf("H*Theta is rank %d %-14s and IS formed, now that the projector comes first;\n",
        length(HT.inds), shape(HT))
println("it is built here only to check the sketch. Omega is folded into the gate first.")

# ── 4. preselection: project the frame out, orthonormalise ───────────────────
hdr("4.  PRESELECTION:  what part of Y lies OUTSIDE the current frame?")
Rperp = to_concrete(perp_component(f.U0, Y))
# The ratio is printed in SCIENTIFIC notation on purpose: it is ~1e-5 here, and "%.3f" rendered
# it as "0.000", which reads as "there is nothing outside the frame" -- the opposite of the
# truth. That sliver IS the new physics, and it is why every tolerance in this path is relative.
@printf("||Y|| = %.6e     ||P_perp Y|| = %.6e   ->  ratio %.3e\n",
        norm(Y), norm(Rperp), norm(Rperp) / norm(Y))
QL, err_pre_l = R._preselect_left(f.U0, Y, 1e-10)
@printf("Q_L %-16s  %d orthonormal candidate directions, all orthogonal to U0\n",
        shape(QL), leg_dim(QL, 3))
@printf("weight the preselection SVD discarded: err_pre = %.3e\n", err_pre_l)
@printf("Q_L is orthogonal to U0 to %.2e  (the RepairOrtho second pass earns this)\n",
        norm(to_concrete(contract(QL', (1, 2), f.U0, (1, 2)))) / max(1, leg_dim(QL, 3)))

# ── 5. final selection: rank the candidates by weight ────────────────────────
hdr("5.  FINAL SELECTION:  rank the candidates by their weight in H*Theta")
OmL = sector_graded_sketch(f.U0, :left, npre; comp_ratio = 0.5,
                           rng = MersenneTwister(0xC0FFEE))
QR, err_pre_r = R._preselect_right(f.V0, sketch_bond_right(f, gate, OmL), 1e-10)
UMU = to_concrete(permutedims(contract(sketch_bond_left(f, gate, QR), (1, 2), QL', (1, 2)),
                              (2, 1)))
@printf("UMU %-16s = Q_L^dag (H Theta) Q_R   -- a SMALL matrix, %d x %d\n",
        shape(UMU), leg_dim(UMU, 1), leg_dim(UMU, 2))
res = svd(UMU, (1,); Nkeep = min(budget, leg_dim(UMU, 1), leg_dim(UMU, 2)),
          cutoff = 1e-13, get_lists = true)
kept = [Float64(s) for (s, _, _, _) in res.kept_list]
cut = [Float64(s) for (s, _, _, _) in res.trunc_list]
@printf("singular values KEPT  (%d): %s\n", length(kept),
        join([@sprintf("%.3e", s) for s in kept[1:min(end, 6)]], "  "))
@printf("singular values CUT   (%d): %s\n", length(cut),
        isempty(cut) ? "-- none --" :
        join([@sprintf("%.3e", s) for s in cut[1:min(end, 6)]], "  "))
println("These singular values ARE the weights: this is the ranking the K/L augmentation")
println("cannot do, since it admits every direction it finds regardless of weight.")

# ── 6. the expansion ─────────────────────────────────────────────────────────
hdr("6.  THE EXPANSION:  U_ex = [U0 | new],  and it is LOSSLESS")
ex = cbe_expand_bond(f, gate; growth = GROWTH, rng = MersenneTwister(0xC0FFEE))
@printf("U_ex %-16s   r = %d -> %d   (+%d new left directions)\n",
        shape(ex.U_ex), f.old_rank, leg_dim(ex.U_ex, 3), ex.n_new_l)
@printf("V_ex %-16s   r = %d -> %d   (+%d new right directions)\n",
        shape(ex.V_ex), f.old_rank, leg_dim(ex.V_ex, 1), ex.n_new_r)
@printf("err_pre = %.3e   err_fnl = %.3e   (err_fnl = 0 means nothing ranked out)\n",
        ex.err_pre, ex.err_fnl)

S_hat0 = to_concrete(contract(contract(ex.U_ex', (1, 2), theta0, (1, 2)),
                              (2, 3), ex.V_ex', (2, 3)))
rebuilt = to_concrete((ex.U_ex * S_hat0) * ex.V_ex)
@printf("LOSSLESS: ||U_ex S_hat0 V_ex - Theta0|| = %.2e     ||S_hat0|| - ||Theta0|| = %.2e\n",
        norm(to_concrete(rebuilt - theta0)), norm(S_hat0) - norm(theta0))
println("So the expansion changes NO amplitude -- it only widens the space the step may use.")
println("This is also why an expansion alone cannot raise the Schmidt rank.")

lres(U) = norm(to_concrete(perp_component(U, HT))) / norm(HT)
@printf("\nleft residual ||HT - U U^dag HT|| / ||HT||:  before %.3e  ->  after %.3e\n",
        lres(f.U0), lres(ex.U_ex))
println("That is the whole point of the selection: the frame now spans the gate's image.")

# ── 7. the S-step ────────────────────────────────────────────────────────────
hdr("7.  THE S-STEP:  one Galerkin exponential in the expanded basis")
tau = -im * DT
nmv = Ref(0)
apply_s(x) = begin
    nmv[] += 1
    th = to_concrete((ex.U_ex * x) * ex.V_ex)
    ev = apply_gate(gate, th, f.site_l.itags, f.site_r.itags)
    pr = contract(ex.U_ex', (1, 2), ev, (1, 2))
    to_concrete(contract(pr, (2, 3), ex.V_ex', (2, 3)))
end
S1 = expv(apply_s, ComplexF64(tau), S_hat0; hermitian = true, maxiter = 30, tol = 1e-15)
@printf("S_hat0 %-14s  ->  S1 %-14s   in %d Lanczos iterations\n",
        shape(S_hat0), shape(S1), nmv[])
@printf("||S1|| = %.10f   ||S_hat0|| = %.10f    (unitary step: equal)\n", norm(S1), norm(S_hat0))

r2 = svd(S1, (1,); cutoff = 1e-14, Nkeep = MAXDIM, get_lists = true)
sv = [Float64(s) for (s, _, _, _) in r2.kept_list]
@printf("S1 singular values: %s\n", join([@sprintf("%.3e", s) for s in sv[1:min(end, 8)]], "  "))
@printf("kept rank %d of the expanded %d; discarded weight %.2e\n",
        length(sv), leg_dim(ex.U_ex, 3), R._trunc_weight(r2))
println("The rank SURVIVES here and only here: S1 is dense on the new directions, so the")
println("truncation keeps them. Without this step they would collapse at the next gauge move.")

# ── 8. the same thing through the public entry point ─────────────────────────
hdr("8.  THE SAME UPDATE, VIA cbe_bond_update")
out = cbe_bond_update(f, gate, tau; growth = GROWTH, maxdim = MAXDIM,
                      trunc_thresh = 1e-14, rng = MersenneTwister(0xC0FFEE))
@printf("left_core %-16s right_core %-16s\n", shape(out.left_core), shape(out.right_core))
@printf("aug_k = %d  aug_l = %d  keep = %d  n_matvec = %d  discarded = %.2e\n",
        out.aug_k, out.aug_l, out.keep, out.n_matvec, out.discarded)
@printf("err_pre = %.3e  err_fnl = %.3e\n", out.err_pre, out.err_fnl)

# ── 9. tau = 0 must be the identity ──────────────────────────────────────────
hdr("9.  SANITY: tau = 0 is the identity on the state")
z = cbe_bond_update(f, gate, ComplexF64(0); growth = GROWTH, maxdim = MAXDIM,
                    trunc_thresh = 1e-14, rng = MersenneTwister(0xC0FFEE))
th_z = to_concrete(contract(z.left_core, (3,), z.right_core, (1,)))
@printf("||Theta(tau=0) - Theta0|| = %.2e\n", norm(to_concrete(th_z - theta0)))

println()
println("WALKTHROUGH_DONE")
