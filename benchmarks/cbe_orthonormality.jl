# Isometry defect of the EXPANDED frames, per bond.
#
#     julia --project=. benchmarks/cbe_orthonormality.jl
#
# `[U0 | C_L]` is an isometry only if `C_L` is orthonormal AND `C_L ⊥ U0`. Both hold by
# construction -- the preselection returns an orthonormal `Q ⊥ U0` and the final selection
# rotates it by an isometry -- and "by construction" in floating point is a claim, not a
# guarantee. This measures the claim.
#
# Reported per bond:
#   defL   ‖P⊥_{U_ex} U_ex‖   how far the expanded LEFT frame is from a left isometry
#   defR                      the mirror on the right
#   loss   ‖U_ex Ŝ0 V_ex − Θ0‖   the expansion's losslessness, which defL/defR feed
#
# Run it before and after changing the expansion to see whether a guard bought anything.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf, Random
using LurCGT, Telum

set_symmetry!(:U1)

const L      = get(ENV, "L", "8")      |> x -> parse(Int, x)
const DT     = 0.05
const MAXDIM = 16
const WARM   = 3

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

psi = domain_wall_state(L)
gates = xx_gates(psi)
cbe_bond_update_bug!(psi, gates;
                     opts = CBEBugOptions(dt = DT, n_steps = WARM, maxdim = MAXDIM,
                                          trunc_thresh = 1e-14))

@printf("L = %d, warmed %d steps of dt = %.2f, bond_dims = %s\n",
        L, WARM, DT, string(bond_dims(psi)))

# TWO REGIMES, because they exercise DIFFERENT code and only one of them carries risk.
#
#   * the growth schedule leaves `dex` far above the candidate count, so `dex_l <= joint` is
#     false and both sides take the FALLBACK branch -- `_trim_total`, a per-sector slice of an
#     already-orthonormal Q. Exact by construction: slicing columns cannot break orthonormality.
#   * a small absolute `dex` makes `joint = dex` and fires the RANKED branch, `C_L = Q_L U_M`.
#     That is a contraction over a summed index, i.e. the only place in the expansion where
#     floating-point error can actually accumulate. If a guard is worth anything it is here.
function report(label; kwargs...)
    @printf("\n%s\n", label)
    @printf("%-6s %-6s %-9s %-11s %-11s %-11s\n",
            "bond", "r", "new(l,r)", "defL", "defR", "loss")
    println("-"^62)
    wl, wr, wloss = 0.0, 0.0, 0.0
    for i in 1:(L - 1)
        canonical!(psi, i)
        f = bond_frame(psi, i)
        theta0 = frame_theta(f)
        ex = cbe_expand_bond(f, gates[i]; kwargs...)

        dl = left_isometry_defect(ex.U_ex)
        dr = right_isometry_defect(ex.V_ex)

        # Losslessness: re-express the core in the widened basis and rebuild the block.
        Shat = to_concrete(contract(contract(ex.U_ex', (1, 2), theta0, (1, 2)),
                                    (2, 3), ex.V_ex', (2, 3)))
        rebuilt = to_concrete(contract(contract(ex.U_ex, (3,), Shat, (1,)),
                                       (3,), ex.V_ex, (1,)))
        loss = norm(to_concrete(rebuilt - theta0))

        wl, wr, wloss = max(wl, dl), max(wr, dr), max(wloss, loss)
        @printf("%-6s %-6d (%d,%d)%-4s %-11.3e %-11.3e %-11.3e\n",
                "($i,$(i+1))", f.old_rank, ex.n_new_l, ex.n_new_r, "", dl, dr, loss)
    end
    println("-"^62)
    @printf("WORST   defL %.3e   defR %.3e   loss %.3e\n", wl, wr, wloss)
    return (wl, wr, wloss)
end

report("A. growth schedule (default) -- fallback branch dominates")
report("B. dex = 2 -- forces the RANKED branch, C_L = Q_L*U_M"; dex = 2)
report("C. dex = 1 -- ranked branch, tightest cap"; dex = 1)
println("\nORTHO_DONE")
