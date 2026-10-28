# tests/common/xx_efficiency_fixtures.jl -- fixtures and OFFLINE audits of the XX domain-wall
# method-efficiency study (xx_domain_wall_cbe_bug_efficiency_standalone_plan.md, "the plan"):
# the exact Slater state graded against an MPS, the compressed domain-wall snapshots (Fixture M),
# a bond's DK/DD content, and the two independent accuracy audits (rho_DD, eps_K). Nothing here is
# timed or reachable from the solver. Needs `free_fermion.jl`, `spin_ed.jl`, `ramped_tdvp.jl`,
# `adaptive_fixtures.jl` and `dd_fixtures.jl` included first.

"`<e_k|psi>` for every product state of `basis` (`spin_xx_sector`'s): the MPS in the sector basis."
xx_sector_amplitudes(psi, basis) =
    ComplexF64[overlap(sector_product_state(spin_space(1), [((s,),) for s in b]), psi) for b in basis]

"""
    xx_state_error(psi, t, basis) -> (err, err_phase, fid)

`||psi - psi_ex(t)||`, the same after the best global phase (plan Fig. 4B: phase-aligned) and
`|<psi_ex|psi>|`, against the exact Slater state. Offline only: one overlap per basis state.
"""
function xx_state_error(psi, t, basis)
    a = xx_sector_amplitudes(psi, basis)
    ex = xx_slater_amplitudes(length(basis[1]), t, basis)
    ov = dot(ex, a)
    return (err = norm(a - ex),
            err_phase = sqrt(max(norm(a)^2 + norm(ex)^2 - 2 * abs(ov), 0.0)),
            fid = abs(ov))
end

"""
    xx_reference_state(L, t; dt = 0.025, maxdim = 512) -> SymMPS

The high-accuracy domain wall at `J t`: the order-4 TDVP2 reference solver of `test_spin_xx.jl`
(first step ramped, `trunc_thresh = 1e-14`), NOT the integrator under study. Grade it with
[`xx_state_error`](@ref) before use.
"""
function xx_reference_state(L::Int, t::Real; dt::Real = 0.025, maxdim::Int = 512)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = domain_wall_state(L)
    n = round(Int, t / dt)
    abs(n * dt - t) < 1e-12 || throw(ArgumentError("t = $t is not a multiple of dt = $dt"))
    for k in 1:n
        ramped_tdvp2_step!(psi, mpo, dt, k; order = 4, maxdim = maxdim, trunc_thresh = 1e-14,
                           ramp = 12)
    end
    return psi
end

"""
    xx_snapshot(psi_ref, rank) -> (psi, discarded)

Fixture M's deliberate compression: every bond cut to at most `rank` by one sectorwise SVD sweep
(`truncate_sweep!`, U(1) blocks ranked across sectors), so the bonds keep real discarded
directions to discover. `discarded = ||psi - psi_ref|| / ||psi_ref||`. A fixture-generation step,
never timed.
"""
function xx_snapshot(psi_ref, rank::Int)
    psi = truncate_sweep!(copy(psi_ref); maxdim = rank, cutoff = 1e-14)
    n2 = real(overlap(psi_ref, psi_ref))
    d2 = real(overlap(psi, psi)) + n2 - 2 * real(overlap(psi_ref, psi))
    return psi, sqrt(max(d2, 0.0) / n2)
end

"""
    xx_bond_content(psi, mpo, i) -> NamedTuple

Bond `(i, i+1)` of `psi` (centre moved there) with its environments, and the norms of the four
blocks of the two-site action `X = H Theta` split by the kept (K) / discarded (D) spaces of both
frames: `kk`, `kd`, `dk`, `dd` (left letter = left frame). `dd` is `||G_DD||`, the range the DD
sketch looks for; `dk + kd` is what a one-sided sketch would also spend rank on (plan Sec. 2).
`rank_dd` is `G_DD`'s numerical rank, `disc_l`/`disc_r` the discarded dimensions. Materialises
`X` and is therefore an offline audit for small bonds only.
"""
function xx_bond_content(psi, mpo, i::Int)
    L = length(psi)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    lenv, renv = _envs(psi, mpo, i, L)
    X = apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)          # (link_l, s_l, s_r, link_r)
    pl(T) = to_concrete(T - to_concrete!(contract(f.U0, (3,), contract(f.U0', (1, 2), T, (1, 2)), (1,))))
    pr(T) = to_concrete(T - to_concrete!(contract(contract(T, (3, 4), f.V0', (2, 3)), (3,), f.V0, (1,))))
    XD = pl(X)                                    # P_L^D X
    XK = to_concrete(X - XD)
    Gdd, Gkd = pr(XD), pr(XK)
    nrm(T) = norm(T)
    sv = first.(svd(Gdd, (1, 2); cutoff = 0.0, get_lists = true).kept_list)   # (value, ...) tuples
    rank_dd = isempty(sv) ? 0 : count(>(1e-12 * maximum(sv)), sv)
    fused_l = leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
    fused_r = leg_dim(f.V0, 2) * leg_dim(f.V0, 3)
    return (frame = f, lenv = lenv, renv = renv, x = nrm(X), dd = nrm(Gdd),
            dk = nrm(to_concrete(XD - Gdd)), kd = nrm(Gkd), kk = nrm(to_concrete(XK - Gkd)),
            rank_dd = rank_dd, old_rank = f.old_rank,
            disc_l = fused_l - f.old_rank, disc_r = fused_r - f.old_rank)
end

"`rho_DD(Q) = ||(I - QQ') G_DD|| / ||G_DD||` (plan Fig. 1), with `c` from `_dd_closures`."
rho_dd(c, Q) = norm(c.perpQ(Q, c.G)) / norm(c.G)

"""
    one_sided_probe(c, w; rng) -> Om

Plan Fig. 1B's one-sided baseline: the same sector-graded Gaussian probe as `_dd_probe`, with
columns allotted by each sector's FULL dimension and NOT projected onto the opposite discarded
space, so `c.sketch(Om)` followed by `_dd_select` samples `P^D X` (DK and DD) at width `w`.
"""
function one_sided_probe(c, w::Int; rng)
    opp, oside = c.opp, c.oside
    a, b = oside === :left ? (1, 2) : (2, 3)
    F = fusion_basis(opp, a, b; tag = "cbeG")
    dfull, _ = RC._frame_sector_dims(opp, oside, F, "one_sided_probe")
    R = RC._trim_per_sector(RC._randomize_like(F, rng), 3, RC._alloc_columns(dfull, w))
    return oside === :left ? R : to_concrete(permutedims(R, (3, 1, 2)))
end

"""
    eps_k(Q, applyH, seed, tau, side) -> Float64

Plan Fig. 2's independent response audit `||(I - QQ') e^{tau H1} K|| / ||K||`, the exponential
action from a tightly converged Lanczos (`maxiter = 120`, `tol = 1e-15`) that shares nothing with
the stopping rule under test.
"""
function eps_k(Q, applyH, seed, tau, side::Symbol)
    Y = expv(applyH, ComplexF64(tau), seed; maxiter = 120, tol = 1e-15)
    P = side === :left ? perp_component : perp_component_right
    return norm(P(Q, Y)) / norm(seed)
end
