# ONE-SITE TDVP WITH CONTROLLED BOND EXPANSION.
#
# Structure follows QSMPSLib `TDVPSweepCBE1Si.m`: per bond, CBE first, then a FORWARD one-site
# update at `+tau`, then a BACKWARD zero-site update at `-tau`, then absorb the bond tensor
# into the next site. Two half-sweeps at `tau/2` each, so the step is second order -- the same
# composition `tdvp2_step!` uses.
#
# WHY CBE IS NOT OPTIONAL HERE, unlike in the BUG path where it is one basis update among
# several. BARE ONE-SITE TDVP IS FIXED-RANK: splitting a `(chi_l, d, chi_r)` site tensor can
# never produce a bond wider than `chi_r`, because the bond tensor has to contract back into a
# neighbour whose leg is `chi_r`. So 1-site TDVP cannot grow entanglement AT ALL, and started
# from a product state it stays a product state forever. CBE is what supplies the rank: it
# widens the bond BEFORE the update, by padding the neighbour with complement directions and
# the current site with matching zeros, and the one-site evolution then fills them.
#
# THAT IS THE WHOLE APPEAL over 2-site TDVP. Two-site TDVP grows rank by evolving the rank-4
# block, at `O(d^2 chi^3)` per bond and with a projection error from the two-site truncation.
# One-site TDVP + CBE gets rank adaptivity while every evolution stays one-site (`O(d chi^3)`)
# or zero-site (`O(chi^2)`), and the rank-4 object is never formed -- provided the expansion
# itself is sketched, which is exactly what `exact = false` buys. With `exact = true` the
# preselection DOES form it, which is why that arm is the reference and not the default.
#
# THE ZERO-SITE BACK-STEP IS NOT OPTIONAL EITHER. Evolving site `i` and then site `i+1` counts
# the bond between them twice; the `-tau` zero-site step on the shared bond removes exactly
# that double count. Dropping it does not degrade the order, it makes the scheme wrong.

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

        # Right environment at link i+1: the prebuilt stack is at link i+2 and psi[i+1] has
        # just been REPLACED by the expanded frame, so this push is mandatory, not a shortcut.
        rch1 = push_right_channels(rch, h, psi[i + 1], i + 1)

        # ---- forward: one-site at site i, +tau ----
        M = expv(x -> (nmv += 1; apply_one_site(one_site_h(h, i, lch, rch1), x)),
                 tau, psi[i]; hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        res = svd(M, (1, 2); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
                  get_lists = true)
        acc[3] = max(acc[3], _trunc_weight(res))
        # THE TWO BOND LEGS MUST STAY DISTINCTLY TAGGED UNTIL THE VERY END, and that constraint
        # is what dictates the whole ordering below. `C` is `(new, bond_ex)`: two DIFFERENT
        # spaces, so Telum forbids giving them the same itag -- yet `bond_ex` already carries
        # `tag` from `_absorb_right!`. And `contract` DOES verify itags, not just arrows, so
        # leaving `C` untagged instead makes `apply_zero_site` reject it against the environment
        # built from `psi[i]`.
        #
        # So `psi[i]` keeps the SVD's own tag on its bond leg for now, `C` inherits it, `H0` is
        # built from that pair and matches, and only after the absorption are BOTH ends renamed
        # to `tag` together. Same discipline as `cbe_lubich_sweep`, where `U_ex` and `V_ex` carry
        # different bond tags right up to the write-back.
        psi[i] = res.U
        C = to_concrete(res.S * res.Vd)

        # ---- backward: zero-site on the shared bond, -tau ----
        # `zero_site_h` wants the channels at links i and i+2 and pushes them through the two
        # site tensors itself, which is why `rch` (not `rch1`) goes in here.
        H0 = zero_site_h(h, i, lch, rch, psi[i], psi[i + 1])
        C = expv(x -> (nmv += 1; apply_zero_site(H0, x)), -tau, C;
                 hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        # Absorb, then rename BOTH ends of the bond to `tag` in the same breath. Each result is
        # rank-3 with only one leg wanting `tag`, so neither construction sees a duplicate.
        psi[i + 1] = to_concrete(setitag(
            to_concrete(contract(C, (2,), psi[i + 1], (1,))), 1, tag))
        psi[i] = to_concrete(setitag(psi[i], 3, tag))
        psi.center = i + 1

        # AFTER the retag, not before: these channels are what bond `i+1` will contract its own
        # tensors against, so they have to be pushed through the FINAL `psi[i]`. Built from the
        # SVD-tagged version they would carry a tag nothing downstream still uses.
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
            to_concrete(contract(psi[i], (3,), C, (1,))), 3, tag))
        psi[i + 1] = to_concrete(setitag(psi[i + 1], 1, tag))
        psi.center = i

        rnext = push_right_channels(rch, h, psi[i + 1], i + 1)
        acc[6] += nmv
        return rnext
    end
end

"""
    tdvp1_cbe_step!(psi, h, tau; exact=false, kwargs...) -> TDVP1CBEInfo

One symmetric one-site-TDVP-with-CBE step of `tau`, in place. Second order.

`tau = -im*dt` is real time and `tau = -dt` imaginary time; nothing here cares which, exactly
as for the BUG path. The caller renormalises -- for imaginary time that is load-bearing.

`exact = false` (default) is **1site_tdvp_cbe_rsvd**: the expansion is sketched, so no rank-4
object is ever formed and the whole step is one-site or cheaper.
`exact = true` is **1site_tdvp_cbe**: the expansion uses the complete fused basis, which is
the reference the sketch approximates and DOES form the rank-4 object.

Environment work is `O(L)` per half-sweep, the same structure `tdvp2_step!` and
`cbe_lubich_sweep` use: each half-sweep carries its own side bond to bond and reads the other
from one stack built at the start. That stack survives the half-sweep because a rightward pass
only rewrites sites at or behind the bond it is on.
"""
function tdvp1_cbe_step!(psi::SymMPS, h::XXZChain, tau::ComplexF64;
                         exact::Bool = false,
                         dex::Int = 0,
                         growth::Float64 = 2.0,
                         dover::Union{Nothing, Int} = nothing,
                         comp_ratio::Union{Float64, Nothing} = 0.5,
                         sulz_cap::Bool = false,
                         preselect_only::Bool = false,
                         maxdim::Int = 200,
                         trunc_thresh::Float64 = 1e-12,
                         maxiter::Int = 30,
                         tol::Float64 = 1e-15,
                         # REORTHOGONALISATION IS ON BY DEFAULT HERE, and it is a correctness
                         # fix rather than a tuning choice. Without it the Lanczos does not
                         # converge at `maxiter = 30` on the `exact = true` arm, and the failure
                         # is silent in the worst way: MEASURED at L=8, XX, dt=0.02, 20 steps,
                         # the state lost 4.6e-3 of its norm (real-time TDVP is unitary, so any
                         # loss is error), the exact and sketched arms diverged by 2.03e-02
                         # where they must agree, and the unconverged arm carried a SPURIOUS
                         # extra bond -- maxbd 12 against the sketch's 10 -- so it looked like
                         # richer physics rather than a broken solve. With `reorth = true`, at
                         # the same `maxiter`: norm error 1.3e-13, arms agree to 1.06e-12, and
                         # both sit at maxbd 10. `maxiter = 200` fixes it too, at ~6x the
                         # matvecs; turning truncation off also masks it, which is why the
                         # cause was initially mis-attributed to the SVD.
                         reorth::Bool = true,
                         rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    L >= 2 || throw(ArgumentError("tdvp1_cbe_step! needs at least two sites"))
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))

    # acc = [unused, max_expanded, discarded, err_pre, err_fnl, nmv]
    acc = Any[0, 0, 0.0, 0.0, 0.0, 0]
    half = tau / 2

    # `rmax` read once, before either sweep widens anything, so the (optional) Sulz bound is a
    # global statement about the state rather than a per-bond one. Same as `cbe_lubich_sweep`.
    exkw = (dex = dex, growth = growth, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact, rng = rng)

    # THE END-SITE UPDATES ARE NOT OPTIONAL BOOKKEEPING -- they are half the time step at the
    # two boundary sites, and leaving them out is a silent O(1) error.
    #
    # Count the sites each loop touches. The rightward loop runs over BONDS 1..L-1 and evolves
    # the bond's LEFT site, so it covers sites 1..L-1 and never site L. The leftward loop
    # evolves each bond's RIGHT site, covering L..2 and never site 1. Without a closing update
    # at the far end of each pass, sites 1 and L receive tau/2 where every interior site
    # receives tau -- so the boundary is integrated at half the rate of the bulk.
    #
    # MEASURED, and this is why it was not caught earlier: at L=8 XX from a DOMAIN WALL the
    # error is invisible (2.09e-06, matching the analytic answer) because the edges of a domain
    # wall are frozen and never carry any amplitude to get wrong. At L=12 Heisenberg from a
    # NEEL state, where every site is active, the same code gave 5.73e-02 against 2-site TDVP's
    # 6.99e-09 -- seven orders out, while carrying MORE bond dimension, which is the signature
    # of a wrong scheme rather than a starved basis.
    #
    # `TDVPSweepCBE1Si.m` closes each pass with `Update1Si(..., 'vv')`, the flag meaning "final
    # site, no bond update after it". There is no back-step here precisely because there is no
    # shared bond beyond the chain end to double-count.
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
