
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

Per-step record: `max_bond`, `theta_elements` (largest two-site block formed), and `discarded`
(largest relative weight one bond truncation threw away).

`peak_elements` is the working set, as for `CBEBugInfo.peak_elements`; `krylov_dims` uses the same
counting rule as `CBEBugSweepInfo` (applications, not flops). `cumulative` is the relative discarded
norm, accounted as in `truncate_recursive!`; `weight_tol` fixes each cut's allowance, not this
total.
"""
struct TDVP2Info
    max_bond::Int
    theta_elements::Int
    discarded::Float64
    peak_elements::Int
    krylov_dims::Int
    cumulative::Float64
end

TDVP2Info(m, t, d, p, k) = TDVP2Info(m, t, d, p, k, NaN)

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
                      conv_tol::Float64 = 0.0, substeps::Int = 1,
                      weight_tol::Union{Nothing, Float64} = nothing, wdiv::Float64 = 1.0,
                      sweep::Int = 1, eshift::Float64 = 0.0, amp::Bool = false,
                      bnorms::Union{Nothing, Vector{Float64}} = nothing)
    Theta = to_concrete(psi[i] * psi[i + 1])        # rank 4 -- the object being compared
    acc[2] = max(acc[2], tensor_elements(Theta))
    acc[4] = max(acc[4], _state_stored(psi) + tensor_elements(Theta))
    # `acc[1]` is the operator-application counter -- see `TDVP2Info.krylov_dims`. It was an
    # unused slot before, so nothing else reads it. The generator is `H - eshift` (see
    # `tdvp2_step!`); `eshift = 0` skips the subtraction, so the unshifted step is bit-identical.
    h2(x) = (y = apply_h_two_site(x, h, i, lch, rch); iszero(eshift) ? y : to_concrete(y - eshift * x))
    Theta = expv(x -> (acc[1] += 1; h2(x)), tau, Theta;
                 hermitian = hermitian, maxiter = maxiter, tol = tol,
                 conv_tol = conv_tol, substeps = substeps)

    # Two rank rules, as in `truncate_recursive!`: Telum's relative singular-value `cutoff`
    # (legacy), or with `weight_tol` the BUG's own budgeted cut (`_budget_svd`): the smallest
    # rank whose discarded Schmidt norm fits the fixed share `weight_tol ||Theta|| / wdiv`.
    res = weight_tol === nothing ?
          svd(Theta, (1, 2); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true) :
          first(_budget_svd(Theta, (1, 2), maxdim, n2 -> weight_tol * sqrt(n2) / wdiv))
    acc[3] = max(acc[3], _trunc_weight(res))
    acc[4 + sweep] += _sq_weight(res.trunc_list)
    acc[7] = max(acc[7], _sq_weight(res.kept_list) + _sq_weight(res.trunc_list))
    tag = psi[i].inds[3].itags

    if dir === :right
        psi[i] = to_concrete(setitag(res.U, 3, tag))
        # The backward half-step sits at site i+1, so its left channels are this bond's
        # pushed through the isometry just written -- no rebuild.
        lnext = push_left_channels(lch, h, psi[i], i)
        M = to_concrete(setitag(to_concrete(res.S * res.Vd), 1, tag))
        if i < length(psi) - 1
            M = _tdvp2_backward(shift_h(one_site_h(h, i + 1, lnext, rch), eshift), M, tau, acc,
                                amp; hermitian, maxiter, tol, conv_tol, substeps, bnorms)
        end
        psi[i + 1] = M
        psi.center = i + 1
        return lnext
    else
        psi[i + 1] = to_concrete(setitag(res.Vd, 1, tag))
        rnext = push_right_channels(rch, h, psi[i + 1], i + 1)
        M = to_concrete(setitag(to_concrete(res.U * res.S), 3, tag))
        if i > 1
            M = _tdvp2_backward(shift_h(one_site_h(h, i, lch, rnext), eshift), M, tau, acc,
                                amp; hermitian, maxiter, tol, conv_tol, substeps, bnorms)
        end
        psi[i] = M
        psi.center = i
        return rnext
    end
end

"""
The backward one-site half-step `exp(-tau H1) M`. With `amp`, `acc[8]` keeps the largest
amplification `log(||M'|| / ||M||) - Re(-tau) <H1>_M` -- the norm growth beyond the mean-energy
drift. By Jensen it is >= 0 and nondecreasing in the step, the same for any energy shift, and zero
in real time; it costs one extra `H1` action, so it is measured only on request.
"""
function _tdvp2_backward(H1, M, tau::ComplexF64, acc, amp::Bool; hermitian, maxiter, tol,
                         conv_tol, substeps, bnorms = nothing)
    nb0 = bnorms === nothing ? 0.0 : norm(M)
    if amp
        n0 = norm(M)
        e1 = real(tensor_inner(M, apply_one_site(H1, M))) / n0^2
    end
    M = expv(x -> (acc[1] += 1; apply_one_site(H1, x)), -tau, M; hermitian = hermitian,
             maxiter = maxiter, tol = tol, conv_tol = conv_tol, substeps = substeps)
    amp && (acc[8] = max(acc[8], log(norm(M) / n0) - real(-tau) * e1))
    bnorms === nothing || push!(bnorms, norm(M) / nb0)
    return M
end

"""
    tdvp2_step!(psi, h, tau; kwargs...) -> TDVP2Info

One symmetric two-site TDVP step of `tau = -im*dt`, in place; second order. The memory and accuracy
baseline for CBE-BUG. `h` may be an `XXZChain` or an `MPO`; the channel form cannot express
long-range or site-dependent `H` (`henv.jl`), so the MPO signature lets the baseline run on
Haldane-Shastry or periodic models.
"""
function tdvp2_step!(psi::SymMPS, h::Union{XXZChain, MPO}, tau::ComplexF64;
                     maxdim::Int = 200,
                     trunc_thresh::Float64 = 1e-12,
                     maxiter::Int = 30,
                     tol::Float64 = 1e-15,
                     # `hermitian = false` selects Arnoldi, since Lanczos on a non-Hermitian
                     # generator does not fail, it lies. This keyword was once accepted and then
                     # ignored: both `_tdvp2_bond!` calls hard-coded `hermitian = true`, so
                     # non-Hermitian runs used Lanczos until it was wired through.
                     hermitian::Bool = true,
                     # See `BondUpdateBUG/expv.jl`: a real convergence test, and splitting `tau`
                     # so a large step does not need a larger `maxiter`. Both integrators take
                     # them, because handing one arm a better exponential than the other would
                     # make the comparison meaningless.
                     conv_tol::Float64 = 0.0,
                     substeps::Int = 1,
                     # The BUG's discard rule, cut for cut: each two-site update uses the allowance
                     # `weight_tol ||Theta|| / _cut_divisor(L, root)` (as in `truncate_recursive!`);
                     # `nothing` keeps `trunc_thresh`. Its 2(L-1) cuts against the BUG's L-1 mean
                     # `info.cumulative` can reach ~sqrt(2) `weight_tol`.
                     weight_tol::Union{Nothing, Float64} = nothing,
                     # IMAGINARY TIME: integrate `H - eshift`, as `cbe_bug_step!(...; eshift)` does,
                     # so the order-2 partner of `ite_step!` sees the same generator. Without it the
                     # unnormalised step grows by e^(dbeta |E|) and its squared Schmidt weights
                     # overflow Float64 near dbeta |E| ~ 350 (L = 100 Heisenberg, panel dbeta = 8).
                     eshift::Real = 0.0,
                     # `Ref`: receives the plan's transient backward-substep amplification, the
                     # largest over the BACKWARD one-site substeps `exp(-tau/2 H1) M` of their
                     # norm growth beyond the mean-energy drift (`_tdvp2_backward`): how strongly
                     # the backward stage amplifies components above the mean energy, the
                     # projector-splitting instability in imaginary time. `nothing` skips it.
                     backward_report::Union{Nothing, Base.RefValue} = nothing,
                     # QTT plan §2.5: every backward substep's RAW amplification
                     # `||M_after|| / ||M_before||` is pushed here (`A_back(n, i)`); `nothing` skips.
                     backward_norms::Union{Nothing, Vector{Float64}} = nothing)
    L = length(psi)
    L >= 2 || throw(ArgumentError("tdvp2_step! needs at least two sites"))
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    weight_tol === nothing || weight_tol >= 0 ||
        throw(ArgumentError("weight_tol must be non-negative, got $weight_tol"))
    # [applications, Theta elements, worst bond weight, peak, sum w forward, sum w backward,
    #  ||psi||^2, worst backward log growth]
    acc = Any[0, 0, 0.0, 0, 0.0, 0.0, 0.0, -Inf]
    wkw = (weight_tol = weight_tol, wdiv = _cut_divisor(L, _bug_root_link(L) + 1),
           eshift = Float64(eshift), amp = backward_report !== nothing,
           bnorms = backward_norms)
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
                           conv_tol = conv_tol, substeps = substeps, wkw..., sweep = 1)
    end

    lstack = left_env_stack(psi, h; upto = L - 2)
    rch = boundary_channels(h)
    for i in (L - 1):-1:1
        canonical!(psi, i)
        rch = _tdvp2_bond!(psi, h, i, half, :left,
                           left_channels(lstack, i), rch;
                           maxdim, trunc_thresh, maxiter, tol, acc, hermitian = hermitian,
                           conv_tol = conv_tol, substeps = substeps, wkw..., sweep = 2)
    end

    cum = acc[7] > 0 ? (sqrt(acc[5]) + sqrt(acc[6])) / sqrt(acc[7]) : 0.0
    backward_report === nothing || (backward_report[] = acc[8])
    return TDVP2Info(maximum(bond_dims(psi); init = 0), acc[2], acc[3], acc[4], acc[1], cum)
end

# The `h` annotation must match the main method's Union. With `XXZChain` here and the widened Union
# there, the pair is ambiguous for `(XXZChain, ComplexF64)`: neither method dominates in both `h`
# and `tau`, so Julia refuses the call. The campaigns pass an `MPO`, which is why this stayed
# hidden.
tdvp2_step!(psi::SymMPS, h::Union{XXZChain, MPO}, tau::Number; kwargs...) =
    tdvp2_step!(psi, h, ComplexF64(tau); kwargs...)

# ── HIGHER ORDER BY COMPOSITION ──
# Composing the symmetric step (Yoshida / Suzuki) gives 4th order via a backward stage (real time
# only). Only splitting error goes away, not projection error, so gains are limited.
const _TDVP2_COMPOSITIONS = Dict(
    # Yoshida's triple jump: 3 stages, the backward one -1.70 tau.
    :yoshida => let g = 1 / (2 - cbrt(2)); (g, 1 - 2g, g) end,
    # Suzuki's fractal: 5 stages, the backward one only -0.66 tau, a smaller error constant.
    :suzuki  => let p = 1 / (4 - cbrt(4)); (p, p, 1 - 4p, p, p) end)

"""
    tdvp2_composed_step!(psi, h, tau; order = 4, composition = :suzuki, kwargs...) -> TDVP2Info

One 2-site TDVP step of order `order` (2 or 4) in the time step, in place: `order = 2` is
[`tdvp2_step!`](@ref); `order = 4` composes it over the stages of `composition` (`:yoshida`, 3
stages; `:suzuki`, 5 stages). `kwargs` go to every stage. The returned info has the largest bond,
Theta and peak over the stages, the worst discarded weight, the summed operator applications and
the summed per-stage `cumulative` discarded norms.
"""
function tdvp2_composed_step!(psi::SymMPS, h::Union{XXZChain, MPO}, tau::Number;
                              order::Int = 4, composition::Symbol = :suzuki, kwargs...)
    order == 2 && return tdvp2_step!(psi, h, ComplexF64(tau); kwargs...)
    order == 4 || throw(ArgumentError("order must be 2 or 4, got $order"))
    haskey(_TDVP2_COMPOSITIONS, composition) || throw(ArgumentError(
        "composition must be one of $(collect(keys(_TDVP2_COMPOSITIONS))), got $composition"))
    infos = [tdvp2_step!(psi, h, ComplexF64(g * tau); kwargs...)
             for g in _TDVP2_COMPOSITIONS[composition]]
    return TDVP2Info(maximum(i.max_bond for i in infos), maximum(i.theta_elements for i in infos),
                     maximum(i.discarded for i in infos), maximum(i.peak_elements for i in infos),
                     sum(i.krylov_dims for i in infos), sum(i.cumulative for i in infos))
end
