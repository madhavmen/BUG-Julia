# ONE-SITE TDVP WITH CONTROLLED BOND EXPANSION, ON THE MPO PATH.
#
# A LINE-FOR-LINE PORT of `cbe_matlab/TDVPSweepCBE1Si.m`:
#
#   MATLAB (`>>` half-sweep, si = 1 … L-1)                     here
#   -------------------------------------------------------    ------------------------------
#   VR = loadV(EnV.VR{si+1})                                   right_channels(rstack, i+2)
#   [TL,TR,VR,InfoCBE] = CBE(TL,TR,VL,VR,HL,HR,'>>')           expand_bond!(…, :right, …)
#   [TL,VL,OrthC,Info1Si] = Update1Si(TL,VL,VR,HL,'>>')        forward 1-site expv, +tau
#   [OrthC,Info0Si] = Update0Si(OrthC, VL, VR)                 backward 0-site expv, -tau
#   TL = AbsorbOrthCent(OrthC, TR, '>>')                       C absorbed into psi[i+1]
#   … closing `[TL,Info1Si] = Update1Si(TL,VL,VR,HL,'vv')`     the site-L update, no split
#
# WHY THE BACKWARD 0-SITE STEP IS THERE, since it is the one thing the DMRG sweep next door does
# NOT have. Two consecutive forward site updates both act on the bond between them, so the bond
# is advanced twice; the `-tau` 0-site step removes exactly one of those. It is not a
# correction bolted on -- it is what makes the composition a projector splitting of the TDVP
# flow, and dropping it does not give a slightly worse integrator, it gives one that is not
# consistent.
#
# WHY THIS EXISTS ALONGSIDE `tdvp1_cbe/tdvp1_cbe.jl`. That file is the same algorithm on the
# CHANNEL path (`XXZChain`), and the channel form cannot express a long-range or site-dependent
# `H` at all: `henv.jl::_left_step` explicitly REFUSES to carry an open channel past a site
# (`"would carry an open channel past site $i"`), which is exactly what a distance-2 or longer
# term needs. Haldane-Shastry is all-pairs, so it needs the MPO. The two are pinned against each
# other on a nearest-neighbour chain, where both apply, and that agreement is what licenses
# reading results from this one.
#
# THE STEP IS SECOND ORDER AND SYMMETRIC: each half-sweep carries `tau/2`, so the pair
# composes to `tau` and the two halves are mirror images. The MATLAB sweep takes `tau` from
# whatever its driver supplied and does not fix the convention itself; the standard one is used
# here and the `dt`-order is asserted in the tests rather than assumed.

"""
    TDVPCBEInfo

What one [`tdvp_cbe1s_step!`](@ref) did.

  - `max_bond`, `max_expanded` -- final rank, and the widest CBE PROPOSED before the split
    truncated it. Compare the two to see whether the expansion or the truncation is limiting.
  - `n_new` -- directions admitted per bond, summed over both half-sweeps.
  - `err_pre`, `err_fnl` -- largest preselection / final-selection weight discarded.
  - `discarded` -- largest truncation weight over the step.
  - `krylov_dims` -- operator applications over both the one-site and zero-site solves. NOT
    comparable bond for bond with a 0-site-only scheme: a one-site matvec is `O(d chi^2 w)`
    against a zero-site `O(chi^2 w)`.
"""
struct TDVPCBEInfo
    max_bond::Int
    max_expanded::Int
    n_new::Vector{Int}
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    peak_elements::Int
end

"""
    tdvp_cbe1s_step!(psi, mpo, tau; kwargs...) -> TDVPCBEInfo

One symmetric second-order 1-site-TDVP-with-CBE step of `tau`, in place.

`tau = -im*dt` is real time and `tau = -dt` imaginary time; nothing here cares which. **The
caller renormalises** -- for imaginary time that is load-bearing, since `exp(-dt H)` is not
norm preserving and the norm is the thing being minimised through.

Keywords: `dex`, `growth`, `dover`, `comp_ratio`, `sulz_cap`, `preselect_only`, `exact`
forward to [`cbe_expand`](@ref); `exact = false` (default) is the RSVD selection and is the
method. `maxdim` / `trunc_thresh` bound the splits, `maxiter` / `tol` the Krylov budget.

`reorth = true` by default, and it is a CORRECTNESS fix rather than a tuning choice: measured
on the channel path at L=8, XX, dt=0.02, 20 steps, the un-reorthogonalised Lanczos lost 4.6e-3
of the state's norm (real-time TDVP is unitary, so any loss is error), the exact and sketched
arms diverged by 2.0e-2 where they must agree, and the unconverged arm carried a SPURIOUS
extra bond -- so it looked like richer physics rather than a broken solve.

Environment work is `O(L)` per half-sweep: each half carries its own side bond to bond and
reads the other from one stack built at the start. That stack survives the half-sweep because
a rightward pass only rewrites sites at or behind the bond it is on.
"""
function tdvp_cbe1s_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                          dex::Int = 0,
                          growth::Float64 = 2.0,
                          dover::Union{Nothing, Int} = nothing,
                          comp_ratio::Union{Float64, Nothing} = 0.5,
                          sulz_cap::Bool = false,
                          preselect_only::Bool = false,
                          exact::Bool = false,
                          maxdim::Int = 200,
                          trunc_thresh::Float64 = 1e-12,
                          maxiter::Int = 30,
                          tol::Float64 = 1e-15,
                          reorth::Bool = true,
                          rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("tdvp_cbe1s_step! needs at least two sites"))

    cut = max(trunc_thresh, 1e-14)
    half = tau / 2
    n_new = zeros(Int, L - 1)
    maxexp = 0; epre = 0.0; efnl = 0.0; disc = 0.0; nmv = 0

    exkw = (dex = dex, growth = growth, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact, rng = rng)

    # Working set at this instant: the state as it stands plus the transients still alive. SAME
    # DEFINITION as `CBEBugSweepInfo.peak_elements` and `TDVP2Info.peak_elements`, which is the
    # only reason the three are comparable -- a memory claim measured three different ways is not
    # a claim. It was missing here, so the campaign's `peak` column read 0 for this sweep.
    peak = 0
    peak!(ts...) = (peak = max(peak, _state_stored(psi) +
                               sum(tensor_elements(t) for t in ts; init = 0)))

    function record!(info)
        maxexp = max(maxexp, info.dim_out)
        n_new[info.bond] += info.n_new
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
    end

    "One-site propagation at site `j`, no split -- the reference's `'vv'` closing update."
    function onesite!(j, lch, rch)
        H1 = one_site_h(mpo, j, lch, rch)
        psi[j] = expv(x -> (nmv += 1; apply_one_site(H1, x)), half, psi[j];
                      hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)
        psi.center = j
    end

    # ======== left to right ========
    canonical!(psi, 1)
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    for i in 1:(L - 1)
        rch1, info = expand_bond!(psi, mpo, i, :right, lch,
                                  right_channels(rstack, i + 2); exkw...)
        record!(info)

        # --- forward one-site update at site i, +tau/2 ---
        H1 = one_site_h(mpo, i, lch, rch1)
        M = expv(x -> (nmv += 1; apply_one_site(H1, x)), half, psi[i];
                 hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        peak!(M)
        res = svd(M, (1, 2); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        psi[i] = res.U
        C = to_concrete(res.S * res.Vd)

        # --- backward bond (0-site) update, -tau/2 ---
        # The environments are pushed through the tensors AS THEY NOW STAND: `psi[i]` is the
        # freshly written left isometry and `psi[i+1]` the expanded right frame, so this is
        # exactly the reference's `Update0Si(OrthC, VL, VR)` with its updated `VL` and its
        # expanded `VR`.
        H0 = zero_site_h(mpo, i, lch, right_channels(rstack, i + 2), psi[i], psi[i + 1])
        C = expv(x -> (nmv += 1; apply_zero_site(H0, x)), -half, C;
                 hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        # Retag BOTH ends of the bond in one breath: each result is rank-3 with only one leg
        # wanting `tag`, so neither construction ever sees a duplicate index.
        psi[i + 1] = to_concrete(setitag(
            to_concrete(contract(C, (2,), psi[i + 1], (1,))), 1, tag))
        psi[i] = to_concrete(setitag(psi[i], 3, tag))
        psi.center = i + 1

        lch = push_left_channels(lch, mpo, psi[i], i)
    end
    onesite!(L, lch, boundary_channels(mpo))

    # ======== right to left ========
    lstack = left_env_stack(psi, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    for i in (L - 1):-1:1
        lch1, info = expand_bond!(psi, mpo, i, :left,
                                  left_channels(lstack, i), rch; exkw...)
        record!(info)

        H1 = one_site_h(mpo, i + 1, lch1, rch)
        M = expv(x -> (nmv += 1; apply_one_site(H1, x)), half, psi[i + 1];
                 hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        res = svd(M, (1,); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        psi[i + 1] = res.Vd
        C = to_concrete(res.U * res.S)

        H0 = zero_site_h(mpo, i, left_channels(lstack, i), rch, psi[i], psi[i + 1])
        C = expv(x -> (nmv += 1; apply_zero_site(H0, x)), -half, C;
                 hermitian = true, maxiter = maxiter, tol = tol, reorth = reorth)

        psi[i] = to_concrete(setitag(
            to_concrete(contract(psi[i], (3,), C, (1,))), 3, tag))
        psi[i + 1] = to_concrete(setitag(psi[i + 1], 1, tag))
        psi.center = i

        rch = push_right_channels(rch, mpo, psi[i + 1], i + 1)
    end
    onesite!(1, boundary_channels(mpo), rch)

    return TDVPCBEInfo(maximum(bond_dims(psi); init = 0), maxexp, n_new,
                       epre, efnl, disc, nmv, peak)
end

tdvp_cbe1s_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    tdvp_cbe1s_step!(psi, mpo, ComplexF64(tau); kwargs...)

tdvp_cbe1s_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    tdvp_cbe1s_step!(psi, mpo_from_terms(h), tau; kwargs...)
