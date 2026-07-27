# Two-site TDVP -- the BASELINE, not part of the method.
#
# The second design goal (docs/cbe_bug.md §2b) is that CBE-BUG expands a bond more cheaply
# than 2-site TDVP does. That claim needs a real competitor in the same stack, with the
# same Hamiltonian and the same environment code, or the comparison measures the stack
# rather than the method. A proxy would not do: "cheaper than 2-site TDVP" has to be
# measured at MATCHED ACCURACY, so the baseline has to be a correct integrator.
#
# It is also the thing CBE-BUG is contrasted with structurally. The forward half-step here
# builds `Theta = psi[i] psi[i+1]` explicitly, rank 4 and O(chi^2 d^2), and EVOLVES it --
# that object is unavoidable in 2-site TDVP and is exactly what CBE-BUG never allocates.
# The bond also grows to the full `d*chi` before the truncating SVD, whereas CBE-BUG
# expands to `r + Dex` with `Dex` a knob.
#
# Scheme: the standard symmetric sweep. Left to right, each bond forward by tau/2 and each
# intermediate site backward by tau/2; then right to left with the same halves. Second
# order in tau.

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
one_site_h(h::XXZChain, j::Int, lenv::LeftEnvStack, renv::RightEnvStack) =
    OneSiteH(j, lenv.done[j], renv.done[j + 1], lenv.open[j], renv.open[j + 1], h.terms)

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
"""
struct TDVP2Info
    max_bond::Int
    theta_elements::Int
    discarded::Float64
end

"Total stored elements of a TLArray -- the honest size of the object, block-sparse."
tensor_elements(t) = sum(length(r) for r in t.RMTs; init = 0)

"One forward/backward pair at bond `i`, sweeping in `dir`. Returns the largest Theta seen."
function _tdvp2_bond!(psi::SymMPS, h::XXZChain, i::Int, tau::ComplexF64, dir::Symbol;
                      maxdim, trunc_thresh, maxiter, tol, acc)
    lenv = left_env_stack(psi, h; upto = i - 1)
    renv = right_env_stack(psi, h; downto = i + 2)

    Theta = to_concrete(psi[i] * psi[i + 1])        # rank 4 -- the object being compared
    acc[2] = max(acc[2], tensor_elements(Theta))
    Theta = expv(x -> apply_h_two_site(x, h, i, lenv, renv), tau, Theta;
                 hermitian = true, maxiter = maxiter, tol = tol)

    res = svd(Theta, (1, 2); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
              get_lists = true)
    acc[3] = max(acc[3], _trunc_weight(res))
    tag = psi[i].inds[3].itags

    if dir === :right
        psi[i] = to_concrete(setitag(res.U, 3, tag))
        M = to_concrete(setitag(to_concrete(res.S * res.Vd), 1, tag))
        if i < length(psi) - 1                      # no back-step off the end of the sweep
            H1 = one_site_h(h, i + 1, left_env_stack(psi, h; upto = i), renv)
            M = expv(x -> apply_one_site(H1, x), -tau, M;
                     hermitian = true, maxiter = maxiter, tol = tol)
        end
        psi[i + 1] = M
        psi.center = i + 1
    else
        psi[i + 1] = to_concrete(setitag(res.Vd, 1, tag))
        M = to_concrete(setitag(to_concrete(res.U * res.S), 3, tag))
        if i > 1
            H1 = one_site_h(h, i, lenv, right_env_stack(psi, h; downto = i + 1))
            M = expv(x -> apply_one_site(H1, x), -tau, M;
                     hermitian = true, maxiter = maxiter, tol = tol)
        end
        psi[i] = M
        psi.center = i
    end
    return psi
end

"""
    tdvp2_step!(psi, h, tau; kwargs...) -> TDVP2Info

One symmetric two-site TDVP step of `tau = -im*dt`, in place. Second order.

Present as the memory/accuracy baseline for CBE-BUG; it is not part of the CBE scheme and
nothing in `cbe_bug.jl` calls it. Like `cbe_bug_bond_update`, it rebuilds its environment
stacks per bond, so the two are handicapped identically and their wall times are
comparable to each other (though neither is the method's true cost).
"""
function tdvp2_step!(psi::SymMPS, h::XXZChain, tau::ComplexF64;
                     maxdim::Int = 200,
                     trunc_thresh::Float64 = 1e-12,
                     maxiter::Int = 30,
                     tol::Float64 = 1e-15)
    L = length(psi)
    L >= 2 || throw(ArgumentError("tdvp2_step! needs at least two sites"))
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    acc = Any[0, 0, 0.0]
    half = tau / 2

    canonical!(psi, 1)
    for i in 1:(L - 1)
        _tdvp2_bond!(psi, h, i, half, :right;
                     maxdim, trunc_thresh, maxiter, tol, acc)
    end
    for i in (L - 1):-1:1
        canonical!(psi, i)
        _tdvp2_bond!(psi, h, i, half, :left;
                     maxdim, trunc_thresh, maxiter, tol, acc)
    end

    return TDVP2Info(maximum(bond_dims(psi); init = 0), acc[2], acc[3])
end

tdvp2_step!(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    tdvp2_step!(psi, h, ComplexF64(tau); kwargs...)
