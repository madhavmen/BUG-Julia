# The discarded-projector K/L/S bond update -- the one local step the whole
# integrator is built from.
#
# Mirrors discarded_candidate.py::_discarded_local_bond_candidate. The two
# things that make this the DISCARDED variant rather than faithful KLS, quoting
# that file's docstring:
#
#   1. PROJECT-BEFORE. The orthogonal-complement projector is applied to the K/L
#      *generator*, before the exponential, not to the integrated factor:
#      G_K = P_perp_U0 . H_K  with  P_perp_U0 = I - U0 U0'. That makes G_K
#      NON-Hermitian, so the K/L substeps take the Arnoldi path; only the S-step
#      generator stays Hermitian and uses Lanczos.
#
#   2. NO OVERLAP MATRICES. The augmented frames are acted directly --
#      S_hat0 = U_hat' Theta0 V_hat' -- instead of transporting the core through
#      M_hat/N_hat. Alice's `_symmetric_augmented_*` still returns an M_hat; the
#      discarded caller drops it, and so does `augmented_left_isometry` here by
#      never forming one.
#
# TIME ARGUMENT. Python splits the step into `prefactor * dt`, where
# `active_time_prefactor()` reads a global mode flag. The Julia entry point takes
# the single already-multiplied complex time `tau` instead: `tau = -im*dt`. One
# argument, nothing global to get out of sync with the caller.
#
# REAL TIME ONLY. `tau` is a plain complex number and the algebra below does not
# care what is in it, but this integrator is exercised and validated in real time
# only -- every test in tests/BondUpdateBUG passes `tau = -im*dt`.

"""
    pairable_charges(f::BondFrame) -> (left::Set, right::Set)

The charge sectors on each side of the bond that can actually hold amplitude.

`S0` is a charge-neutral rank-2 tensor, so its blocks are exactly the pairs
`(q, dual(q))` -- MEASURED: `S0.qlabels` on a rank-6 bond reads
`[((-3,),(3,)), ((-1,),(1,)), ((1,),(-1,))]`, and `S0.spaces[1]` carries the same
labels as `U0.spaces[3]` despite the opposite arrow, so the two sides' labels
compare directly. A left sector therefore pairs iff its dual is reachable on the
right, and vice versa.

Without this the fill seeds every reachable left sector regardless of the right
frame. Measured on the L=6 domain wall: bond 4 reported `aug_k = 10` while only
4 directions could ever be occupied, and the pruned set equalled the unpaired
set on every sweep. Dead weight rather than a wrong answer -- the S-step leaves
those columns exactly zero and the SVD drops them -- but `aug_k` is the headline
rank diagnostic, and it was 2.5x too large.

NOTE FOR TESTS: at a bond whose reachable set is its own dual (any bond near the
middle of a half-filled chain) EVERY sector pairs, so a dual-direction error here
passes unnoticed. Test bond 5 of the L=6 domain wall, where the right side is a
boundary link and only two of the four left sectors pair.
"""
function pairable_charges(f::BondFrame)
    syms = symm(f.U0)
    lreach = [q for (q, _) in fusion_basis(f.U0, 1, 2).spaces[end]]
    rreach = [q for (q, _) in fusion_basis(f.V0, 2, 3).spaces[end]]
    lset, rset = Set(lreach), Set(rreach)
    return (Set(q for q in lreach if dual_charge(syms, q) in rset),
            Set(q for q in rreach if dual_charge(syms, q) in lset))
end

# ── criterion-2 residual enrichment ──────────────────────────────────────────
# Left/right directions of the full 2-site update `Hth = HΘ` that the current
# augmented frame misses, orthonormal and capped to `budget` (the remaining Sulz
# 2r headroom). `Hth` is rank-4 `(link_l, s_l, s_r, link_r)`. Two projection
# passes drive the overlap with the frame to machine zero so `oplus([frame|Q])`
# stays an isometry (same exactness pass as `_complement_basis`).
const _C2_EPS = 100 * eps(Float64)

function _criterion2_left(U_aug, Hth, budget::Int)
    budget <= 0 && return nothing
    function resid(T)
        ov   = contract(U_aug', (1, 2), T, (1, 2))     # (bondU, s_r, link_r)
        proj = contract(U_aug, (3,), ov, (1,))          # (link_l, s_l, s_r, link_r)
        return to_concrete(T - proj)
    end
    sc = norm(Hth); sc == 0 && return nothing
    R = resid(Hth); norm(R) <= _C2_EPS * sc && return nothing
    Q = to_concrete(svd(R, (1, 2); cutoff = 1e-12).U)   # (link_l, s_l, newbond)
    R2 = resid(Q); norm(R2) <= _C2_EPS * norm(Q) && return nothing
    Q = to_concrete(svd(R2, (1, 2); cutoff = 1e-12).U)
    leg_dim(Q, 3) <= budget ? Q : _trim_total(Q, 3, budget)
end

function _criterion2_right(V_aug, Hth, budget::Int)
    budget <= 0 && return nothing
    # residual of the rank-4 Hth against V_aug over (s_r, link_r)
    function resid4(T)
        ov   = contract(T, (3, 4), V_aug', (2, 3))      # (link_l, s_l, bondV)
        proj = contract(ov, (3,), V_aug, (1,))           # (link_l, s_l, s_r, link_r)
        return to_concrete(T - proj)
    end
    # residual of a rank-3 candidate frame Q=(bond, s_r, link_r) against V_aug
    function resid3(Q)
        ov   = contract(Q, (2, 3), V_aug', (2, 3))       # (Qbond, Vbond)
        proj = contract(ov, (2,), V_aug, (1,))            # (Qbond, s_r, link_r)
        return to_concrete(Q - proj)
    end
    sc = norm(Hth); sc == 0 && return nothing
    R = resid4(Hth); norm(R) <= _C2_EPS * sc && return nothing
    Q = to_concrete(permutedims(svd(R, (3, 4); cutoff = 1e-12).U, (3, 1, 2)))  # (newbond, s_r, link_r)
    R2 = resid3(Q); norm(R2) <= _C2_EPS * norm(Q) && return nothing
    Q = to_concrete(permutedims(svd(R2, (2, 3); cutoff = 1e-12).U, (3, 1, 2)))
    leg_dim(Q, 1) <= budget ? Q : _trim_total(Q, 1, budget)
end

"""
    kls_bond_update(f::BondFrame, gate, tau; kwargs...) -> NamedTuple

One discarded-projector K/L/S update on the bond `f` snapshots.

Returns `(; left_core, right_core, U_aug, V_aug, S_new, n_new_k, n_new_l,
keep, aug_k, aug_l, svals, discarded)`. `left_core` and `right_core` carry the
frame's own `link_l/site_l` and `site_r/link_r` legs with a fresh bond tagged
`f.link_mid`, so they drop straight back into a `SymMPS`.

Keywords:

  - `maxdim` -- hard cap on the kept bond dimension (Telum `Nkeep`).
  - `trunc_thresh` -- singular-value cutoff, floored at `1e-14` as in Python.
  - `s_tau` -- separate S-step time, for Trotter schemes that halve it.
    Defaults to `tau`.
  - `augment` -- `false` freezes the bases, giving a fixed-rank projector step.
    There is deliberately no K/L tolerance: the augmentation keeps every
    direction it finds, bounded only by Sulz's `2r`.
  - `missing_fill` -- random seeds per empty-but-reachable charge sector.
  - `maxiter` / `tol` -- Krylov budget for all three substeps.
"""
function kls_bond_update(f::BondFrame, gate, tau::ComplexF64;
                         maxdim::Int = 200,
                         trunc_thresh::Float64 = 1e-14,
                         s_tau::Union{Nothing, ComplexF64} = nothing,
                         augment::Bool = true,
                         missing_fill::Int = 1,
                         pad::Bool = false,
                         criterion2::Bool = false,
                         maxiter::Int = 30,
                         tol::Float64 = 1e-15,
                         rng::AbstractRNG = MersenneTwister(0x5EED))
    tau_s = s_tau === nothing ? tau : s_tau
    tl, tr = f.site_l.itags, f.site_r.itags
    # Only sectors with a partner on the other side are worth seeding.
    seed_l, seed_r = missing_fill > 0 ? pairable_charges(f) : (nothing, nothing)

    # ---- K-step: project-before, then integrate K0 = U0*S0 -----------------
    K0 = to_concrete(f.U0 * f.S0)              # (link_l, site_l, mid)

    # H_K x = V0'-projected gate action; G_K x = P_perp_U0 (H_K x). Written with
    # `perp_component`, the same primitive the augmentation uses, so the
    # projector in the generator and the projector that isolates the new
    # directions can never drift apart.
    function apply_gk(x)
        theta = to_concrete(x * f.V0)
        evolved = apply_gate(gate, theta, tl, tr)
        HK = to_concrete(evolved * f.V0')      # (link_l, site_l, mid)
        return perp_component(f.U0, HK)
    end

    K1 = expv(apply_gk, tau, K0; hermitian = false, maxiter = maxiter, tol = tol)
    U_aug, n_new_k = augmented_left_isometry(f.U0, K1;
                                             augment = augment,
                                             missing_fill = missing_fill,
                                             pad = pad,
                                             seed_charges = seed_l, rng = rng)

    # ---- L-step: the mirror --------------------------------------------------
    L0 = to_concrete(f.S0 * f.V0)              # (mid, site_r, link_r)

    function apply_gl(x)
        theta = to_concrete(f.U0 * x)
        evolved = apply_gate(gate, theta, tl, tr)
        HL = to_concrete(f.U0' * evolved)      # (mid, site_r, link_r)
        return perp_component_right(f.V0, HL)
    end

    L1 = expv(apply_gl, tau, L0; hermitian = false, maxiter = maxiter, tol = tol)
    V_aug, n_new_l = augmented_right_isometry(f.V0, L1;
                                              augment = augment,
                                              missing_fill = missing_fill,
                                              pad = pad,
                                              seed_charges = seed_r, rng = rng)

    theta0 = frame_theta(f)

    # ---- criterion-2 residual enrichment (targeted rank growth, no padding) --
    # The K/L half-steps hold the opposite frame fixed, so on a product state the
    # off-diagonal generator projects to zero and rank cannot grow (the state
    # freezes -- fatal without U(1) sectors, where the missing-fill can't help).
    # The residual of the FULL 2-site update HΘ captures exactly that missed
    # direction; enriching the frames with it -- orthogonal to what K/L already
    # found, capped at the Sulz 2r -- grows rank by the minimal physical amount.
    if criterion2
        Hth = apply_gate(gate, theta0, tl, tr)
        QL = _criterion2_left(U_aug, Hth, 2 * f.old_rank - leg_dim(U_aug, 3))
        QL === nothing || (U_aug = to_concrete(oplus(
            [U_aug, to_concrete(setitag(QL, 3, U_aug.inds[3].itags))], (3,))))
        QR = _criterion2_right(V_aug, Hth, 2 * leg_dim(f.V0, 1) - leg_dim(V_aug, 1))
        QR === nothing || (V_aug = to_concrete(oplus(
            [V_aug, to_concrete(setitag(QR, 1, V_aug.inds[1].itags))], (1,))))
    end

    # ---- S-step: project Theta0 onto the augmented bases and evolve ----------
    # No M_hat/N_hat: S_hat0 = U_hat' Theta0 V_hat' directly. Because U_aug
    # CONTAINS U0 as its first block and V_aug contains V0, this projection is
    # lossless -- U_aug U_aug' Theta0 V_aug' V_aug == Theta0 exactly -- which is
    # what makes tau = 0 the identity on the state.
    S_start = to_concrete(contract(contract(U_aug', (1, 2), theta0, (1, 2)),
                                   (2, 3), V_aug', (2, 3)))

    function apply_s(x)
        theta = to_concrete((U_aug * x) * V_aug)
        evolved = apply_gate(gate, theta, tl, tr)
        proj = contract(U_aug', (1, 2), evolved, (1, 2))
        return to_concrete(contract(proj, (2, 3), V_aug', (2, 3)))
    end

    # The Galerkin generator on the augmented bases IS Hermitian -- the
    # projectors sit symmetrically on both sides -- so this one is Lanczos.
    S_new = expv(apply_s, tau_s, S_start; hermitian = true, maxiter = maxiter, tol = tol)

    # ---- truncate: the SVD sets the new, rank-adaptive bond dimension --------
    # Symmetry-blocked, so the kept rank respects the U(1) sectors and any seeded
    # sector the dynamics left empty is pruned here.
    res = svd(S_new, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true)
    left_tmp  = to_concrete(U_aug * res.U)
    right_tmp = to_concrete((res.S * res.Vd) * V_aug)

    tag = f.link_mid.itags
    left_core  = to_concrete(setitag(left_tmp, 3, tag))
    right_core = to_concrete(setitag(right_tmp, 1, tag))

    svals = [s for (s, _, _, _) in res.kept_list]
    return (; left_core, right_core, U_aug, V_aug, S_new,
            n_new_k, n_new_l,
            keep = leg_dim(left_core, 3),
            aug_k = f.old_rank + n_new_k,
            aug_l = f.old_rank + n_new_l,
            svals, discarded = _discarded_weight(res))
end

kls_bond_update(f::BondFrame, gate, tau::Number; kwargs...) =
    kls_bond_update(f, gate, ComplexF64(tau); kwargs...)

"""
    _discarded_weight(res) -> Float64

`sqrt(sum of discarded sigma^2 / sum of all sigma^2)` for the S-step split.

Python recomputes this by densifying `S_new` and taking its full spectrum
(`scheme.py::_discarded_weight`). Telum's SVD already reports both halves of the
spectrum in `kept_list` / `trunc_list`, so the same number comes out without ever
leaving the block-sparse representation -- densifying a symmetric tensor is the
one thing this integrator must never do.

Each entry is `(sigma, degeneracy, sector, rank)`; the degeneracy factor is 1
throughout under U(1) and carries the multiplet size under a non-Abelian symmetry.
"""
function _discarded_weight(res)
    w(list) = sum(Float64(s)^2 * Float64(deg) for (s, deg, _, _) in list; init = 0.0)
    kept, cut = w(res.kept_list), w(res.trunc_list)
    total = kept + cut
    return total == 0.0 ? 0.0 : sqrt(cut / total)
end
