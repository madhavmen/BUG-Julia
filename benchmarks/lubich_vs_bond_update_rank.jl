# Did the two shared selection fixes help the LUBICH sweep, and how much is left?
#
# The gate path proved the CBE selection is sound (it grows rank and tracks
# `bond_update_bug!` bond for bond). Both fixes -- the probe-width shortfall and the
# per-side final selection -- landed in `cbe.jl`, which the Lubich/MPO sweep in
# `cbe_lubich.jl` calls through the same `cbe_expand(f, skl, skr; ...)` core. So the MPO path
# gets them for free, and the only question left is what the SWEEP itself costs.
#
# Three runs on the same state, same dt, same maxdim, same truncation:
#
#   bond_update_bug!  validated odd/even gate Trotter, K/L ODEs + random sector fill
#   cbe_bond_update_bug!     the same validated sweep, CBE basis update            <- confirmed
#   cbe_lubich_bug!          the Lubich basis sweep + one centre Galerkin, H dressed
#                            with channel environments (an EFFECTIVE local operator, not a
#                            global H applied to the state)
#
# Read the BOND PROFILES against each other, not against a target. If `cbe_lubich_bug!` now tracks
# the other two, the fixes closed it and there was never a separate sweep bug. If it still
# stalls while `cbe_bond_update_bug!` does not, the residue is the sweep -- and since the selection
# is identical code driven by identical states, the difference has to be in what the sweep
# feeds it or what it does with the result.
#
# XX, so `xx_free_fermion_sz` gives an analytic error alongside the rank.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf
include("../tests/common/free_fermion.jl")

const L      = 8
const DT     = 0.02
const NSTEPS = 25
const MAXDIM = 32
const TRUNC  = 1e-14
const DEX    = 8              # the budget the gate path showed saturates
const T_END  = DT * NSTEPS

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(L - 1)]

function report(label, psi, prof_t)
    q = copy(psi)
    sz = magnetisation(q) ./ norm(q)^2
    err = maximum(abs.(sz .- xx_free_fermion_sz(L, T_END)))
    @printf("%-22s  err %.3e   profile %-26s  %6.1fs\n",
            label, err, string(bond_dims(psi)), prof_t)
end

println("XX domain wall, L = $L, dt = $DT, $NSTEPS steps (t = $T_END), " *
        "maxdim = $MAXDIM, trunc = $TRUNC, dex = $DEX\n")

pk = domain_wall_state(L)
tk = @elapsed bond_update_bug!(pk, xx_gates(pk);
                               opts = BondUpdateOptions(dt = DT, n_steps = NSTEPS,
                                                        maxdim = MAXDIM,
                                                        trunc_thresh = TRUNC))
report("bond_update_bug!", pk, tk)

pg = domain_wall_state(L)
tg = @elapsed cbe_bond_update_bug!(pg, xx_gates(pg);
                            opts = CBEBugOptions(dt = DT, n_steps = NSTEPS,
                                                 maxdim = MAXDIM, trunc_thresh = TRUNC,
                                                 dex = DEX))
report("cbe_bond_update_bug! (gate)", pg, tg)

# The Lubich sweep dresses H with channel environments rather than using per-bond gates,
# so it takes the whole chain rather than a gate list.
h = xxz_chain(L; J = 1.0, delta = 0.0)
pm = domain_wall_state(L)
tm = @elapsed im = cbe_lubich_bug!(pm, h;
                            opts = CBEBugOptions(dt = DT, n_steps = NSTEPS,
                                                 maxdim = MAXDIM, trunc_thresh = TRUNC,
                                                 dex = DEX))
report("cbe_lubich_bug! (Lubich)", pm, tm)
@printf("\ncbe_lubich_bug! max_expanded per step: %s\n", string(im.max_expanded))
@printf("cbe_lubich_bug! centre ranks   per step: %s\n", string(im.centre_ranks))
@printf("cbe_lubich_bug! err_pre %.2e  err_fnl %.2e  discarded %.2e\n",
        maximum(im.err_pre), maximum(im.err_fnl), maximum(im.discarded))
