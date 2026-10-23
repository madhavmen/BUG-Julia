# tests/common/dd_fixtures.jl -- the warmed Heisenberg bonds, closures and controller driver of
# the DD-QR CBE tests.

"""
A warmed, complex Heisenberg/U(1) chain (delta = 1: MPO width 5) and bond `i`'s frame.

The warm-up rank cap binds (`maxdim = 4`) so the bulk bond keeps a discarded space on BOTH sides.
At L = 8 an unbinding cap fills the centre bond (D_4 = 16 = 2^4): both `P^D` vanish there, the DD
sketch is identically zero, and every bond-4 check compares roundoff (hq12br 17409190).
"""
function _hb_bond(; L = 8, i = 4, steps = 2, maxdim = 4)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    psi = neel_state(L)
    for _ in 1:steps
        tdvp2_step!(psi, mpo, -0.2im; maxdim = maxdim, trunc_thresh = 1e-14)
    end
    canonical!(psi, i)
    f = bond_frame(psi, i)
    lenv, renv = _envs(psi, mpo, i, L)
    return mpo, psi, f, lenv, renv
end

_rel(a, b) = norm(to_concrete(a - b)) / max(norm(b), 1e-300)
"Both frames of `f` leave a discarded space, so a DD check at this bond is not vacuous."
_unsaturated(f) = leg_dim(f.U0, 1) * leg_dim(f.U0, 2) > f.old_rank &&
                  leg_dim(f.V0, 2) * leg_dim(f.V0, 3) > f.old_rank
_fid(a, b) = abs(overlap(a, b)) / (norm(a) * norm(b))

"""
The controller fixture: L = 12 warmed at cap 16, bond 6. At L = 8 / cap 4 each side's discarded
probe space is 4-dimensional, so one width-4 holdout spans it and the controller is trivial.
"""
_hb_wide() = _hb_bond(; L = 12, i = 6, steps = 4, maxdim = 16)

"""
`_dd_controller`'s closures for one side of bond `i`, built as `dd_cbe_expand` builds them, and
the exact DD range `G` (the complete discarded probe) its `Q` is graded against.
"""
function _dd_closures(f, mpo, lenv, renv, side; i = 6, rng)
    left = side === :left
    frame, opp = left ? (f.U0, f.V0) : (f.V0, f.U0)
    oside = left ? :right : :left
    kc = KernelCounters()
    op = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = !left, counters = kc)
    out = Ref{Any}(nothing)
    draw(w, b) = RC._dd_probe(opp, oside, w; rng = rng, boost = b, kc = kc)
    sketch(Om) = (apply_two_site_sketch!(out, Om, op); out[])
    project = left ? (X -> perp_component(f.U0, X)) : (X -> perp_component_right(f.V0, X))
    G = dd_reference_sketch(f, mpo, i, lenv, renv, full_local_basis(opp, oside), side)
    fused(t) = left ? leg_dim(t, 1) * leg_dim(t, 2) : leg_dim(t, 2) * leg_dim(t, 3)
    wcap = left ? leg_dim(opp, 2) * leg_dim(opp, 3) : leg_dim(opp, 1) * leg_dim(opp, 2)
    perpQ(Q, X) = Q === nothing ? X : left ? perp_component(Q, X) : perp_component_right(Q, X)
    return (; side, frame, kc, draw, sketch, project, G, perpQ, wcap,
            ncap = fused(frame) - f.old_rank, fl = left ? 3 : 1, wleg = left ? 1 : 3)
end

"Run the controller at `share` (scale 1: the indicator is the raw residual)."
function _ctrl(c, share; w0 = 2, s = RC._DD_HOLDOUT, wcap = c.wcap, draw = c.draw,
               holdout = w -> c.draw(w, nothing))
    lg, tr = RC.AdaptiveLedger(), Any[]
    Q, st = RC._dd_controller(c.frame, c.side, draw, c.sketch, c.project, w0, wcap, c.ncap,
                              RC.BasisAllocation(1.0, share, 1.0, 1.0, 1.0), lg;
                              kc = c.kc, s = s, holdout = holdout, trace = tr)
    return Q, st, lg, tr
end

_rank(Q, fl) = Q === nothing ? 0 : leg_dim(Q, fl)
_sector_norm(T, leg, q, d) = norm(to_concrete(getsub(T, leg, s -> s == q ? (1:d) : nothing)))

"`Om` without its sector `q` on `leg`; `nothing` when nothing else is left."
function _drop(Om, leg, q)
    Om === nothing && return nothing
    dims = Dict(Om.spaces[leg])
    any(k -> k != q, keys(dims)) || return nothing
    return to_concrete(getsub(Om, leg, s -> s == q ? nothing : (1:dims[s])))
end
