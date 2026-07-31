# Iterated basis construction vs. a wide one-shot probe, at one bond.
#
#     julia --project=. benchmarks/cbe_s_iterations.jl
#
# THE QUESTION. The one-shot expansion has to guess the right subspace in a single draw, and
# two hyperparameters exist to help it guess: the oversampling (`dover`, default `1.2x`) and
# the complement/isometry split (`comp_ratio`, default 0.5). The alternative is to stop
# guessing: take a NARROW probe, evolve, re-sketch from the evolved block, and repeat. Each
# pass applies H once more, so the admitted directions accumulate span{Θ, HΘ, H²Θ, ...} and
# the basis converges on the relevant subspace instead of being tuned onto it.
#
# THE CONTROL THAT MATTERS. `s_iters` iterates the BASIS, not the time step: every pass
# rebuilds S_start from the ORIGINAL Θ₀, so exp(τH) is applied exactly once regardless of the
# pass count. If that were wrong -- if a pass fed its evolved core forward as the next
# starting core -- the error would GROW with `s_iters` (it would be integrating to
# `s_iters*tau`), not shrink. The exact-propagator column below is what detects that.
#
# The reference is the exact two-site propagator on the full local space, which forms a rank-4
# object. That is legitimate HERE and nowhere in the method: this is a diagnostic.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf, Random
using LurCGT, Telum

set_symmetry!(:U1)

const L      = 8
const DT     = 0.05
const MAXDIM = 32
const WARM   = 3
const TAU    = ComplexF64(-im * DT)

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

psi = domain_wall_state(L)
gates = xx_gates(psi)
cbe_bond_update_bug!(psi, gates;
                     opts = CBEBugOptions(dt = DT, n_steps = WARM, maxdim = MAXDIM,
                                          trunc_thresh = 1e-14))

# The most expandable bond, for the reason the walkthrough documents: a saturated bond has
# nothing to expand into and would show nothing.
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
tl, tr = f.site_l.itags, f.site_r.itags

# Exact two-site step, on the full local space.
theta_exact = expv(x -> apply_gate(gate, x, tl, tr), TAU, theta0;
                   hermitian = true, maxiter = 80, tol = 1e-16)

rebuild(u) = to_concrete(contract(u.left_core, (3,), u.right_core, (1,)))
err_of(u) = norm(to_concrete(rebuild(u) - theta_exact))

@printf("L = %d, bond (%d,%d), r = %d, dt = %.3f, bond_dims = %s\n",
        L, BOND, BOND + 1, f.old_rank, DT, string(bond_dims(psi)))
@printf("reference: exact two-site propagator,  ||theta_exact|| = %.6f\n\n", norm(theta_exact))

# tau = 0 must stay the identity however many passes run -- the first thing that breaks if a
# pass evolves the block instead of only rebuilding the basis.
println("SANITY  tau = 0 is the identity on the block")
for it in (1, 2, 4)
    u = cbe_bond_update(f, gate, ComplexF64(0); dex = 1, s_iters = it, maxdim = MAXDIM)
    @printf("  s_iters = %d :  ||theta(0) - theta0|| = %.3e\n", it,
            norm(to_concrete(rebuild(u) - theta0)))
end

function row(label, u)
    @printf("%-26s %-11.3e %-6d %-6d %-8d %-11.3e %d\n", label, err_of(u),
            u.aug_k, u.aug_l, u.n_matvec, u.iter_weight, u.n_iter)
end

@printf("\n%-26s %-11s %-6s %-6s %-8s %-11s %s\n",
        "setting", "err vs exact", "aug_k", "aug_l", "matvec", "iter_wt", "iters")
println("-"^82)

println("A. STARVED probe (dex = 1), iterated -- buy directions with passes")
for it in (1, 2, 3, 4, 6, 8)
    row("   dex=1  s_iters=$it", cbe_bond_update(f, gate, TAU;
                                                 dex = 1, s_iters = it, maxdim = MAXDIM))
end

println("\nB. STARVED probe (dex = 2), iterated")
for it in (1, 2, 3, 4, 6)
    row("   dex=2  s_iters=$it", cbe_bond_update(f, gate, TAU;
                                                 dex = 2, s_iters = it, maxdim = MAXDIM))
end

println("\nC. ONE-SHOT with hyperparameters tuned instead (s_iters = 1)")
row("   growth=2.0 (default)", cbe_bond_update(f, gate, TAU; maxdim = MAXDIM))
for cr in (0.0, 0.25, 0.5, 0.75, 1.0)
    row("   growth=2.0 comp=$cr", cbe_bond_update(f, gate, TAU;
                                                  comp_ratio = cr, maxdim = MAXDIM))
end
for dv in (0, 2, 4, 8)
    row("   growth=2.0 dover=$dv", cbe_bond_update(f, gate, TAU;
                                                   dover = dv, maxdim = MAXDIM))
end

println("\nD. FULL local space (no truncation) -- the floor the Galerkin step can reach")
row("   dex=64 s_iters=1", cbe_bond_update(f, gate, TAU; dex = 64, maxdim = MAXDIM))
row("   dex=64 s_iters=3", cbe_bond_update(f, gate, TAU; dex = 64, s_iters = 3,
                                           maxdim = MAXDIM))

# ── E. the expanding-basis Lanczos ───────────────────────────────────────────────────
#
# `s_iters = -g` grows the basis around each of the first `g` Krylov vectors, BEFORE H is
# applied to them. Here `v_k` genuinely carries `H^k`, so every growth pass probes a direction
# that did not exist at the previous pass -- which is exactly what the outer loop in A/B
# cannot do. `iter_wt` is now the final Lanczos beta: the small value in the Krylov matrix
# that says the space has closed.
println("\nE. EXPANDING-BASIS LANCZOS (growth is step 0 of each Krylov iteration)")
println("   starved probe, growth bought by Krylov depth instead of by probe width")
for g in (1, 2, 3, 4, 6)
    row("   dex=1  grow=$g", cbe_bond_update(f, gate, TAU;
                                             dex = 1, s_iters = -g, maxdim = MAXDIM))
end
for g in (1, 2, 3, 4)
    row("   dex=2  grow=$g", cbe_bond_update(f, gate, TAU;
                                             dex = 2, s_iters = -g, maxdim = MAXDIM))
end
println("   default growth schedule, growing for the first g Krylov steps")
for g in (1, 2, 3)
    row("   growth=2.0 grow=$g", cbe_bond_update(f, gate, TAU;
                                                 s_iters = -g, maxdim = MAXDIM))
end

# ATTRIBUTION. The expanding solver reorthogonalises fully against the whole basis, which the
# default `expv` does not (`reorth = false`). If the matvec count falls, that alone could
# explain it, and the expanding basis would be getting credit for a Lanczos detail. Isolate it:
# same one-shot expansion, only `reorth` flipped.
println("\nE2. CONTROL -- one-shot expansion, full reorthogonalisation only")
row("   growth=2.0 reorth=false", cbe_bond_update(f, gate, TAU; maxdim = MAXDIM))
row("   growth=2.0 reorth=true", cbe_bond_update(f, gate, TAU; s_reorth = true,
                                                 maxdim = MAXDIM))
row("   dex=1 reorth=true", cbe_bond_update(f, gate, TAU; dex = 1, s_reorth = true,
                                            maxdim = MAXDIM))

# ── G. THE ONLY REGIME WHERE ANY OF THIS CAN MATTER ──────────────────────────────────
#
# Everything above is measured where the budget SATURATES the room, and that is not an
# accident of L=8 -- it is structural for a spin-1/2 chain at the default schedule:
#
#     room   = d*chi - r     -> r    when chi = r and d = 2
#     budget = ceil(2.0*dmax) - r -> r    when dmax = r
#
# so `dex = min(budget, room) = room` and the expansion spans the WHOLE local space. The
# selection never has to choose, err_fnl is 0 everywhere, and every variant -- every
# comp_ratio, every dover, every iteration scheme -- must agree to machine precision because
# they are all computing the exact two-site propagator. No experiment run there can
# discriminate.
#
# The discriminating knob is `growth`, not the system size. Below 2.0 the budget is smaller
# than the room, the ranking has to pick, and the hyperparameters and the iteration finally
# have something to do.
println("\nG. growth < 2.0 -- budget BELOW room, so the selection actually chooses")
@printf("   room_l = %d, room_r = %d, r = %d\n",
        leg_dim(f.U0, 1) * leg_dim(f.U0, 2) - f.old_rank,
        leg_dim(f.V0, 2) * leg_dim(f.V0, 3) - f.old_rank, f.old_rank)
for g in (1.1, 1.25, 1.5)
    dmax = max(leg_dim(f.U0, 1), f.old_rank, leg_dim(f.V0, 3))
    @printf("   -- growth = %.2f : budget = %d\n", g,
            max(ceil(Int, g * dmax) - f.old_rank, 1))
    row("      one-shot", cbe_bond_update(f, gate, TAU; growth = g, maxdim = MAXDIM))
    row("      + reorth", cbe_bond_update(f, gate, TAU; growth = g, s_reorth = true,
                                          maxdim = MAXDIM))
    for cr in (0.0, 0.5, 1.0)
        row("      comp_ratio=$cr", cbe_bond_update(f, gate, TAU; growth = g,
                                                    comp_ratio = cr, maxdim = MAXDIM))
    end
    for dv in (0, 4)
        row("      dover=$dv", cbe_bond_update(f, gate, TAU; growth = g, dover = dv,
                                               maxdim = MAXDIM))
    end
    row("      outer s_iters=3", cbe_bond_update(f, gate, TAU; growth = g, s_iters = 3,
                                                 maxdim = MAXDIM))
    for gg in (1, 2, 4)
        row("      expanding grow=$gg", cbe_bond_update(f, gate, TAU; growth = g,
                                                        s_iters = -gg, maxdim = MAXDIM))
    end
end

println("\nF. tau = 0 identity, expanding solver")
for g in (1, 3)
    u = cbe_bond_update(f, gate, ComplexF64(0); dex = 1, s_iters = -g, maxdim = MAXDIM)
    @printf("   grow=%d :  ||theta(0) - theta0|| = %.3e\n", g,
            norm(to_concrete(rebuild(u) - theta0)))
end

println("\nITER_DONE")
