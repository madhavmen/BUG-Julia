
"""
    OneSiteH

Single-site effective Hamiltonian at site `j`, as channel environments on links `j` and
`j+1`. Used only for 2-site TDVP's backward half-step.

A single site cannot hold both halves of a nearest-neighbour term, so there is no
analogue of the two-site action's bond contribution -- four kinds of term, not five.
"""
struct OneSiteH
    site::Int
    done_l::Any
    done_r::Any
    open_l::Vector{Any}
    open_r::Vector{Any}
    terms::Vector{XXZTerm}
end

"""
    one_site_h(h, j, lenv, renv) -> OneSiteH

The single-site effective Hamiltonian at site `j`; `lenv` must be filled at link `j` and
`renv` at link `j+1`.
"""
one_site_h(h::XXZChain, j::Int, lch::ChannelSet, rch::ChannelSet) =
    OneSiteH(j, lch.done, rch.done, lch.open, rch.open, h.terms)

one_site_h(h::XXZChain, j::Int, lenv::LeftEnvStack, renv::RightEnvStack) =
    one_site_h(h, j, left_channels(lenv, j), right_channels(renv, j + 1))

"""
    apply_one_site(H1, M) -> TLArray

`H_eff M` for a site tensor `M` with legs `(link_l, site, link_r)`.

`M` shares its layout with both `K0` and `V0`, so `_env_on_link` applies on leg 1 and
leg 3 unchanged, and it already handles closing an op-leg (or not) by rank.
"""
function apply_one_site(H1::OneSiteH, M)
    j = H1.site
    acc = nothing
    add!(x) = (acc = acc === nothing ? to_concrete(x) : to_concrete(acc + x))

    H1.done_l === ZERO || add!(_env_on_link(M, H1.done_l, 1))
    H1.done_r === ZERO || add!(_env_on_link(M, H1.done_r, 3))
    for (t, term) in enumerate(H1.terms)
        o = H1.open_l[t]
        o === ZERO || add!(to_concrete(term.coeff *
            _env_on_link(_apply_site_op(M, term.right, j), o, 1)))
        o = H1.open_r[t]
        o === ZERO || add!(to_concrete(term.coeff *
            _env_on_link(_apply_site_op(M, term.left, j), o, 3)))
    end
    acc === nothing && throw(ArgumentError(
        "apply_one_site: no contributions at site $j"))
    return acc
end

"""
    TDVP2Info

Per-step record: `max_bond` after the step, `theta_elements` the largest two-site block
built (the memory figure CBE-BUG is measured against), and `discarded` the largest
relative weight a bond truncation threw away.

`peak_elements` is the WORKING SET, defined identically to `CBEBugInfo.peak_elements`:
`state + transients alive at that moment`. Here the transient is `Theta`, and it is alive
*alongside* `psi[i]` and `psi[i+1]` -- the two-site block duplicates them rather than
replacing them, which is the cost CBE-BUG avoids by never forming one.

`krylov_dims` counts OPERATOR APPLICATIONS over the whole step, the same field name and the same
counting rule `CBEBugSweepInfo` uses, so a cost comparison between the two is reading one number
rather than two conventions. ⚠ It counts APPLICATIONS, NOT FLOPS, and the applications are not
the same size on both sides: this sweep applies a TWO-site `H_eff` plus a one-site backward
update per bond, while `cbe_bug_step!` applies `L` one-site operators and one zero-site operator.
So a ratio of these counts understates the two-site sweep's per-application cost, and the wall
clock is the honest arbiter -- the count is here to say WHERE the time goes.
"""
struct TDVP2Info
    max_bond::Int
    theta_elements::Int
    discarded::Float64
    peak_elements::Int
    krylov_dims::Int
end

"Total stored elements of a TLArray -- the honest size of the object, block-sparse."
tensor_elements(t) = sum(length(r) for r in t.RMTs; init = 0)

"""
One forward/backward pair at bond `i`, sweeping in `dir`. Takes the channels on the two
links it uses so the caller can CARRY them; `acc` collects the diagnostics.

Returns the channels the NEXT bond of the sweep needs on this sweep's own side: pushed
through the freshly written tensor, which is the O(1) alternative to rebuilding a stack.
"""
function _tdvp2_bond!(psi::SymMPS, h::Union{XXZChain, MPO}, i::Int, tau::ComplexF64,
                      dir::Symbol, lch, rch;
                      maxdim, trunc_thresh, maxiter, tol, acc, hermitian = true,
                      conv_tol::Float64 = 0.0, substeps::Int = 1)
    Theta = to_concrete(psi[i] * psi[i + 1])        # rank 4 -- the object being compared
    acc[2] = max(acc[2], tensor_elements(Theta))
    acc[4] = max(acc[4], _state_stored(psi) + tensor_elements(Theta))
    # `acc[1]` is the operator-application counter -- see `TDVP2Info.krylov_dims`. It was an
    # unused slot before, so nothing else reads it.
    Theta = expv(x -> (acc[1] += 1; apply_h_two_site(x, h, i, lch, rch)), tau, Theta;
                 hermitian = hermitian, maxiter = maxiter, tol = tol,
                 conv_tol = conv_tol, substeps = substeps)

    res = svd(Theta, (1, 2); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true)
    acc[3] = max(acc[3], _trunc_weight(res))
    tag = psi[i].inds[3].itags

    if dir === :right
        psi[i] = to_concrete(setitag(res.U, 3, tag))
        # The backward half-step sits at site i+1, so its left channels are this bond's
        # pushed through the isometry just written -- no rebuild.
        lnext = push_left_channels(lch, h, psi[i], i)
        M = to_concrete(setitag(to_concrete(res.S * res.Vd), 1, tag))
        if i < length(psi) - 1
            M = expv(x -> (acc[1] += 1; apply_one_site(one_site_h(h, i + 1, lnext, rch), x)),
                     -tau, M; hermitian = hermitian, maxiter = maxiter, tol = tol,
                     conv_tol = conv_tol, substeps = substeps)
        end
        psi[i + 1] = M
        psi.center = i + 1
        return lnext
    else
        psi[i + 1] = to_concrete(setitag(res.Vd, 1, tag))
        rnext = push_right_channels(rch, h, psi[i + 1], i + 1)
        M = to_concrete(setitag(to_concrete(res.U * res.S), 3, tag))
        if i > 1
            M = expv(x -> (acc[1] += 1; apply_one_site(one_site_h(h, i, lch, rnext), x)),
                     -tau, M; hermitian = hermitian, maxiter = maxiter, tol = tol,
                     conv_tol = conv_tol, substeps = substeps)
        end
        psi[i] = M
        psi.center = i
        return rnext
    end
end

"""
    tdvp2_step!(psi, h, tau; kwargs...) -> TDVP2Info

One symmetric two-site TDVP step of `tau = -im*dt`, in place. Second order.

Present as the memory/accuracy baseline for CBE-BUG; it is not part of the CBE scheme and
nothing in `cbe_lubich.jl` calls it. Like `cbe_lubich_sweep`, it carries its channel
environments along each half-sweep, so environment work is `O(L)` per step and a wall-time
comparison between the two is not distorted by one of them rebuilding.

TAKES EITHER AN `XXZChain` OR AN `MPO`, and the body is the same either way -- every helper it
calls (`apply_h_two_site`, `one_site_h`, `push_*_channels`, `boundary_channels`, the env stacks)
already has a method for both representations, so admitting the MPO was a matter of widening this
signature and nothing else. The `XXZChain` path is bit-identical to before.

WHY THAT MATTERS: the channel form cannot express a long-range or site-dependent `H` at all
(`henv.jl`'s `_left_step` refuses to carry an open channel past a site), so with the old
`h::XXZChain` annotation the 2-site TDVP BASELINE could not run on Haldane-Shastry or on a
periodic ring -- i.e. it could not run on the models the CBE sweeps are being judged on, which
would have left the comparison with no baseline exactly where one is most wanted.
"""
function tdvp2_step!(psi::SymMPS, h::Union{XXZChain, MPO}, tau::ComplexF64;
                     maxdim::Int = 200,
                     trunc_thresh::Float64 = 1e-12,
                     maxiter::Int = 30,
                     tol::Float64 = 1e-15,
                     # `hermitian = false` -> Arnoldi. TDVP2 needs it for the same reason the
                     # other two do: Lanczos on a non-Hermitian generator does not fail, it lies.
                     #
                     # ⛔ THIS KEYWORD WAS ACCEPTED AND THEN IGNORED. Both `_tdvp2_bond!` calls
                     # below passed a hardcoded `hermitian = true`, so `tdvp2_step!(...;
                     # hermitian = false)` ran LANCZOS on a non-Hermitian generator -- silently,
                     # since that is the failure mode the comment above describes. Anything
                     # measured on a non-Hermitian model through this entry point before
                     # 2026-08-27 used the wrong solver.
                     hermitian::Bool = true,
                     # See `BondUpdateBUG/expv.jl`: a real convergence test, and splitting `tau`
                     # so a large step does not need a larger `maxiter`. Both integrators take
                     # them, because handing one arm a better exponential than the other would
                     # make the comparison meaningless.
                     conv_tol::Float64 = 0.0,
                     substeps::Int = 1)
    L = length(psi)
    L >= 2 || throw(ArgumentError("tdvp2_step! needs at least two sites"))
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    acc = Any[0, 0, 0.0, 0]
    half = tau / 2

    # O(L) environments, as in `cbe_lubich_sweep`: each half-sweep carries its own side
    # and reads the other from one prebuilt stack. The forward sweep only writes sites
    # <= i+1, so the right stack stays valid; the backward sweep only writes sites >= i,
    # so the left stack (rebuilt once, after the forward sweep) stays valid.
    canonical!(psi, 1)
    rstack = right_env_stack(psi, h; downto = 3)
    lch = boundary_channels(h)
    for i in 1:(L - 1)
        lch = _tdvp2_bond!(psi, h, i, half, :right,
                           lch, right_channels(rstack, i + 2);
                           maxdim, trunc_thresh, maxiter, tol, acc, hermitian = hermitian,
                           conv_tol = conv_tol, substeps = substeps)
    end

    lstack = left_env_stack(psi, h; upto = L - 2)
    rch = boundary_channels(h)
    for i in (L - 1):-1:1
        canonical!(psi, i)
        rch = _tdvp2_bond!(psi, h, i, half, :left,
                           left_channels(lstack, i), rch;
                           maxdim, trunc_thresh, maxiter, tol, acc, hermitian = hermitian,
                           conv_tol = conv_tol, substeps = substeps)
    end

    return TDVP2Info(maximum(bond_dims(psi); init = 0), acc[2], acc[3], acc[4], acc[1])
end

# ⛔ `h` MUST be the same `Union` as the method above. When the main method was widened from
# `h::XXZChain` to `h::Union{XXZChain, MPO}` and this wrapper was left at `XXZChain`, the pair
# became AMBIGUOUS for `(XXZChain, ComplexF64)`: the main method is more specific in `tau`
# (`ComplexF64` < `Number`) while this one is more specific in `h` (`XXZChain` < the Union), so
# neither dominates and Julia refuses the call. It stayed hidden because the campaign always
# passes an `MPO` -- only the `XXZChain` + complex-`tau` combination the tests use hits it.
tdvp2_step!(psi::SymMPS, h::Union{XXZChain, MPO}, tau::Number; kwargs...) =
    tdvp2_step!(psi, h, ComplexF64(tau); kwargs...)
