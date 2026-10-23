# ONE-SITE TDVP WITH CONTROLLED BOND EXPANSION.
#

"""
    TDVP1CBEInfo

Per-step record. `max_expanded` is the widest bond CBE PROPOSED before the split truncated it,
the same quantity `CBEBugRunInfo.max_expanded` reports, and the one to read when asking whether
the expansion or the truncation is the rank limiter.

`krylov_dims` sums operator applications over both the one-site and zero-site solves, so it is
comparable with the other arms' `krylov_dims` only after remembering that a zero-site matvec is
`O(chi^2)` against a two-site `O(d^2 chi^3)` -- the counts are not the same unit of work.
"""
struct TDVP1CBEInfo
    max_bond::Int
    max_expanded::Int
    discarded::Float64
    err_pre::Float64
    err_fnl::Float64
    krylov_dims::Int
end

"""
One CBE + forward-one-site + backward-zero-site triple at bond `i`, sweeping in `dir`.

Takes the channels on the links it needs so the caller can CARRY them, and returns the ones
the next bond of the same sweep needs -- pushed through the tensor just written, never rebuilt.
`acc` collects diagnostics in place.
"""
function _tdvp1_cbe_bond!(psi::SymMPS, h::XXZChain, i::Int, tau::ComplexF64, dir::Symbol,
                          lch::ChannelSet, rch::ChannelSet;
                          exkw, maxdim, trunc_thresh, maxiter, tol, reorth = false, acc)
    nmv = 0

    # ---- CBE: widen the bond before anything is evolved --------------------------------
    # `_absorb_*!` keeps the STATE identical: the neighbour becomes the expanded frame and the
    # current site takes the (zero-padded) remainder. The new columns carry no amplitude yet,
    # which is precisely what the one-site update is then able to fill.
    canonical!(psi, i)
    f  = bond_frame(psi, i)
    ex = cbe_expand(f, h, i, lch, rch; exkw...)
    acc[2] = max(acc[2], max(leg_dim(ex.U_ex, 3), leg_dim(ex.V_ex, 1)))
    acc[4] = max(acc[4], ex.err_pre)
    acc[5] = max(acc[5], ex.err_fnl)

    if dir === :right
        _absorb_right!(psi, i, ex.V_ex)              # psi[i+1] = V_ex, centre stays at i
        tag = psi[i].inds[3].itags


        rch1 = push_right_channels(rch, h, psi[i + 1], i + 1)

        # ---- forward: one-site at site i, +tau ----
        M = expv(x -> (nmv += 1; apply_one_site(one_site_h(h, i, lch, rch1), x)),
                 tau, psi[i]; hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        res = svd(M, (1, 2); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
                  get_lists = true)
        acc[3] = max(acc[3], _trunc_weight(res))

        psi[i] = res.U
        C = to_concrete(res.S * res.Vd)

        H0 = zero_site_h(h, i, lch, rch, psi[i], psi[i + 1])
        C = expv(x -> (nmv += 1; apply_zero_site(H0, x)), -tau, C;
                 hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        # Absorb, then rename BOTH ends of the bond to `tag` in the same breath. Each result is
        # rank-3 with only one leg wanting `tag`, so neither construction sees a duplicate.
        psi[i + 1] = to_concrete(setitag(
            to_concrete!(contract(C, (2,), psi[i + 1], (1,))), 1, tag))
        psi[i] = to_concrete(setitag(psi[i], 3, tag))
        psi.center = i + 1

        lnext = push_left_channels(lch, h, psi[i], i)
        acc[6] += nmv
        return lnext
    else
        _absorb_left!(psi, i, ex.U_ex)               # psi[i] = U_ex, centre moves to i+1
        tag = psi[i].inds[3].itags
        lch1 = push_left_channels(lch, h, psi[i], i)

        # ---- forward: one-site at site i+1, +tau ----
        M = expv(x -> (nmv += 1; apply_one_site(one_site_h(h, i + 1, lch1, rch), x)),
                 tau, psi[i + 1]; hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        res = svd(M, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
                  get_lists = true)
        acc[3] = max(acc[3], _trunc_weight(res))
        # Mirror of the rightward branch: `C` is `(bond_ex, new)`, two different spaces, so the
        # SVD's tag stays on the new leg until after the zero-site step.
        psi[i + 1] = res.Vd
        C = to_concrete(res.U * res.S)

        # ---- backward: zero-site, -tau ----
        H0 = zero_site_h(h, i, lch, rch, psi[i], psi[i + 1])
        C = expv(x -> (nmv += 1; apply_zero_site(H0, x)), -tau, C;
                 hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        psi[i] = to_concrete(setitag(
            to_concrete!(contract(psi[i], (3,), C, (1,))), 3, tag))
        psi[i + 1] = to_concrete(setitag(psi[i + 1], 1, tag))
        psi.center = i

        rnext = push_right_channels(rch, h, psi[i + 1], i + 1)
        acc[6] += nmv
        return rnext
    end
end

"""
    tdvp1_cbe_step!(psi, h, tau; exact=false, kwargs...) -> TDVP1CBEInfo

One symmetric second-order one-site TDVP-with-CBE step of `tau`, in place, with the time convention
and renormalisation of [`tdvp_cbe1s_step!`](@ref).

`exact = false` (default) is the RSVD selection: the sketch still forms the rank-4 two-site object,
and controls only the width of the candidate set. `exact = true` uses the complete fused basis.
"""
function tdvp1_cbe_step!(psi::SymMPS, h::XXZChain, tau::ComplexF64;
                         exact::Bool = false,
                         maxdim::Int = 200,
                         trunc_thresh::Float64 = 1e-12,
                         maxiter::Int = 30,
                         tol::Float64 = 1e-15,
                         # ── reorthogonalisation on by default: a correctness fix ──
                         # Without it, Lanczos at maxiter = 30 loses norm and the exact and sketched arms diverge silently,
                         # with a spurious extra bond. `maxiter = 200` fixes it at about 6x the matvecs.
                         reorth::Bool = true,
                         rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    L >= 2 || throw(ArgumentError("tdvp1_cbe_step! needs at least two sites"))
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))

    # acc = [unused, max_expanded, discarded, err_pre, err_fnl, nmv]
    acc = Any[0, 0, 0.0, 0.0, 0.0, 0]
    half = tau / 2

    # `rmax` read once, before either sweep widens anything, so it is a statement about the
    # state rather than a per-bond one. The expansion runs at `cbe_expand`'s own settings.
    exkw = (rmax = maximum(bond_dims(psi); init = 0), exact = exact, rng = rng)


    onesite!(j, lc, rc) = begin
        acc[6] += 1
        psi[j] = expv(x -> apply_one_site(one_site_h(h, j, lc, rc), x), half, psi[j];
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)
        psi.center = j
    end

    canonical!(psi, 1)
    rstack = right_env_stack(psi, h; downto = 3)
    lch = boundary_channels(h)
    for i in 1:(L - 1)
        lch = _tdvp1_cbe_bond!(psi, h, i, half, :right,
                               lch, right_channels(rstack, i + 2);
                               exkw, maxdim, trunc_thresh, maxiter, tol, reorth, acc)
    end
    # Site L. `lch` is now the channels on link L, carried through the whole pass; to its right
    # is the chain end, so the right environment is the boundary.
    onesite!(L, lch, boundary_channels(h))

    # Safe to build AFTER the closing update: that update rewrote only `psi[L]`, and this stack
    # reads sites 1..L-2.
    lstack = left_env_stack(psi, h; upto = L - 2)
    rch = boundary_channels(h)
    for i in (L - 1):-1:1
        rch = _tdvp1_cbe_bond!(psi, h, i, half, :left,
                               left_channels(lstack, i), rch;
                               exkw, maxdim, trunc_thresh, maxiter, tol, reorth, acc)
    end
    # Site 1, the mirror.
    onesite!(1, boundary_channels(h), rch)

    return TDVP1CBEInfo(maximum(bond_dims(psi); init = 0), acc[2], acc[3],
                        acc[4], acc[5], acc[6])
end

tdvp1_cbe_step!(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    tdvp1_cbe_step!(psi, h, ComplexF64(tau); kwargs...)
