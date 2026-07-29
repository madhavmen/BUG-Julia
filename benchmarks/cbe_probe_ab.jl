# Which vector should drive the in-Lanczos expansion: the residual, or the output?
#
#     julia --project=. benchmarks/cbe_probe_ab.jl
#
# The expanding-basis Lanczos reached the SAME rank as the outer loop and was four orders less
# accurate (growth = 1.25: rank 12 either way, 1.450e-11 vs 1.978e-15). So the deficit is which
# directions get admitted, not how many, and the suspect is the probe:
#
#   :residual   the orthonormal Lanczos vector v_k -- H*v_{k-1} with the already-captured
#               components SUBTRACTED OFF. Small, delicate, and structurally unlike the state.
#   :image      the previous step's OUTPUT H*v_{k-1}, before that subtraction.
#
# The Lanczos recurrence is identical in both: H is still applied to the orthonormal v_k, so
# alpha/beta and the tridiagonal do not change. Only the basis the next vector lives in differs.
#
# Run at growth < 2.0, because at growth = 2.0 the budget saturates the room on a d = 2 chain
# and every scheme computes the exact two-site propagator (docs/current_work.md 7.6.1).

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf
using LurCGT, Telum          # to_concrete, contract, norm on TLArray

set_symmetry!(:U1)

const L      = 8
const DT     = 0.05
const MAXDIM = 32
const TAU    = ComplexF64(-im * DT)

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

psi = domain_wall_state(L)
gates = xx_gates(psi)
cbe_bond_update_bug!(psi, gates;
                     opts = CBEBugOptions(dt = DT, n_steps = 3, maxdim = MAXDIM,
                                          trunc_thresh = 1e-14))

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

theta_exact = expv(x -> apply_gate(gate, x, tl, tr), TAU, theta0;
                   hermitian = true, maxiter = 80, tol = 1e-16)
rebuild(u) = to_concrete(contract(u.left_core, (3,), u.right_core, (1,)))
err_of(u) = norm(to_concrete(rebuild(u) - theta_exact))

@printf("L = %d, bond (%d,%d), r = %d, dt = %.3f\n", L, BOND, BOND + 1, f.old_rank, DT)

# ── IS THE OUTPUT ALREADY ORTHONORMAL? ───────────────────────────────────────────────
#
# The claim to check: `w = H*v_k` is already orthonormal against the existing Krylov basis, so
# the Lanczos subtraction `w - alpha*v_k - beta*v_{k-1}` is re-doing work already done.
#
# Measured directly below, running the recurrence by hand in a FIXED basis (the expansion is
# irrelevant to this question). For each step it prints, BEFORE any subtraction:
#
#   |<v_k, w>|      the overlap with the vector H was just applied to    (= |alpha|)
#   max_j |<v_j,w>| the largest overlap with ANY earlier basis vector
#   ||w||           for scale, so the overlaps can be read as relative
#
# If those overlaps are at machine epsilon, the output is already orthogonal and the
# subtraction is redundant. If they are O(1) relative to ||w||, it is not.
println("\nCHECK: is w = H*v_k already orthogonal to the Krylov basis, before subtraction?")
let
    Uc, Vc = f.U0, f.V0
    proj(th) = to_concrete(contract(contract(Uc', (1, 2), th, (1, 2)), (2, 3), Vc', (2, 3)))
    S0 = proj(theta0)
    v = to_concrete((1.0 / norm(S0)) * S0)
    bas = Any[v]
    betas = Float64[]
    @printf("  %-4s %-12s %-14s %-14s %s\n", "k", "||w||", "|<v_k,w>|", "max_j |<v_j,w>|",
            "rel. to ||w||")
    for k in 1:5
        th = to_concrete((Uc * bas[k]) * Vc)
        w = proj(apply_gate(gate, th, tl, tr))
        nw = norm(w)
        ak = abs(tensor_inner(bas[k], w))
        mx = maximum(abs(tensor_inner(u, w)) for u in bas)
        @printf("  %-4d %-12.4e %-14.4e %-14.4e %.3e\n", k, nw, ak, mx, mx / nw)
        a = real(tensor_inner(bas[k], w))
        w = to_concrete(w + (-a) * bas[k])
        k > 1 && (w = to_concrete(w + (-betas[k - 1]) * bas[k - 1]))
        b = norm(w)
        b < 1e-15 && (println("  (beta below tol at k = $k)"); break)
        push!(betas, b)
        push!(bas, to_concrete((1.0 / b) * w))
    end
end
println()
@printf("%-34s %-11s %-6s %-8s %s\n", "setting", "err vs exact", "rank", "matvec", "beta")
println("-"^76)

function row(label, u)
    @printf("%-34s %-11.3e %-6d %-8d %.3e\n", label, err_of(u), u.aug_k, u.n_matvec,
            u.iter_weight)
end

for g in (1.1, 1.25, 1.5)
    @printf("\n-- growth = %.2f\n", g)
    row("   one-shot (baseline)", cbe_bond_update(f, gate, TAU; growth = g, maxdim = MAXDIM))
    row("   outer s_iters=3", cbe_bond_update(f, gate, TAU; growth = g, s_iters = 3,
                                              maxdim = MAXDIM))
    # `grow = 30` equals `maxiter`, i.e. grow at EVERY Krylov step until beta falls below the
    # threshold -- the literal form of the scheme, with no cap on how deep the growth goes.
    for gg in (1, 2, 4, 30)
        row("   expanding grow=$gg probe=:residual",
            cbe_bond_update(f, gate, TAU; growth = g, s_iters = -gg,
                            s_probe = :residual, maxdim = MAXDIM))
        row("   expanding grow=$gg probe=:image",
            cbe_bond_update(f, gate, TAU; growth = g, s_iters = -gg,
                            s_probe = :image, maxdim = MAXDIM))
    end
end

println("\nSANITY tau = 0 identity, probe = :image")
for gg in (1, 3)
    u = cbe_bond_update(f, gate, ComplexF64(0); growth = 1.25, s_iters = -gg,
                        s_probe = :image, maxdim = MAXDIM)
    @printf("   grow=%d :  ||theta(0) - theta0|| = %.3e\n", gg,
            norm(to_concrete(rebuild(u) - theta0)))
end

println("\nPROBE_AB_DONE")
