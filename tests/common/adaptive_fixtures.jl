# tests/common/adaptive_fixtures.jl -- a real bond for the adaptive-CBE unit tests.
#
# The UTs of the adaptive plan test the PRODUCTION kernels (`_preselect_*`, `_sample_residual`,
# `_certified_range`, ...) on a genuine block-sparse bond, never a matrix restatement of them --
# a restatement is what let a biased Test B estimator and a test-local rank helper pass.

const RC = BUGJulia.RSVDCBEBondUpdate

_dims(T, leg) = Dict{Any, Int}(q => d for (q, d) in T.spaces[leg])

"`T` without sector `q` on `leg` (or `nothing` if that was all of it)."
function _drop(T, leg, q)
    d = _dims(T, leg)
    any(k != q for k in keys(d)) || return nothing
    return to_concrete(getsub(T, leg, s -> s == q ? nothing : 1:d[s]))
end

"Only sector `q` of `T` on `leg`."
_only(T, leg, q) = to_concrete(getsub(T, leg, s -> s == q ? (1:_dims(T, leg)[s]) : nothing))

"A warmed XX/U(1) chain: Neel evolved `steps` TDVP2 steps of `0.2` (`steps = 0` keeps Neel)."
function _xx_warm(L, steps)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = neel_state(L)
    for _ in 1:steps
        tdvp2_step!(psi, mpo, -0.2im; maxdim = 64, trunc_thresh = 1e-14)
    end
    return mpo, psi
end

"Environments at links `i` and `i+2`; the chain ends carry the boundary, as the sweeps do."
_envs(psi, mpo, i, L) =
    (i == 1 ? boundary_channels(mpo) : left_channels(left_env_stack(psi, mpo; upto = i), i),
     i + 2 > L ? boundary_channels(mpo) :
                 right_channels(right_env_stack(psi, mpo; downto = i + 2), i + 2))

"""
    _xx_bond(; L = 8, i = 4, steps = 2) -> (frame, skl, skr)

Bond `i` of an XX/U(1) chain with the PRODUCTION sketch closures. `steps = 0` keeps the Neel
product state, whose rank-1 frame leaves a reachable sector empty by construction.
"""
function _xx_bond(; L = 8, i = 4, steps = 2)
    mpo, psi = _xx_warm(L, steps)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    lenv, renv = _envs(psi, mpo, i, L)
    skl, skr = RC._sketch_closures(f,
        () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv))
    return f, skl, skr
end

"""
    _xx_kl(side; L = 8, i = 4, steps = 2, dex = 0) -> NamedTuple

What ONE half-sweep step hands `_krylov_enrich` at bond `i`, built exactly as `half_left!`
(`side = :left`) or `half_right!` (`:right`) build it: the centre-carrying site tensor, the CBE
frame pair from a (legacy, deterministic) `cbe_expand`, the transported seed and the one-site
`H1` whose far environment runs through the WIDENED opposite frame.

Fields: `seed`, `Q0` (the CBE frame on this side), `old` (the frame before CBE), `applyH`.
`dex > 0` caps the CBE expansion per side, so `Q0` is a strict subspace of the local space and
the Krylov stage has something left to find (at the defaults CBE fills the local space).
"""
function _xx_kl(side::Symbol; L = 8, i = 4, steps = 2, dex = 0)
    mpo, psi = _xx_warm(L, steps)
    canonical!(psi, side === :left ? i : i + 1)
    f = RC._frame_from(psi[i], psi[i + 1])
    lch, rch = _envs(psi, mpo, i, L)
    ex = RC.cbe_expand(f, mpo, i, lch, rch; dex = dex)
    return (; _kl_stage(side, mpo, i, psi[i], psi[i + 1], lch, rch, ex)...,
            old = side === :left ? f.U0 : f.V0)
end

"""
The seed and one-site operator `half_left!`/`half_right!` hand `_krylov_enrich` after the CBE
`ex` at bond `i` of the two-site block `left * right` (centre in the block): `(seed, Q0, applyH)`.
"""
function _kl_stage(side::Symbol, mpo, i, left, right, lch, rch, ex)
    if side === :left
        M = to_concrete(contract(right, (2, 3), ex.V_ex', (2, 3)))
        seed = to_concrete(contract(left, (3,), M, (1,)))
        H1 = one_site_h(mpo, i, lch, push_right_channels(rch, mpo, ex.V_ex, i + 1))
        return (seed = seed, Q0 = ex.U_ex, applyH = x -> apply_one_site(H1, x))
    end
    M = to_concrete(contract(ex.U_ex', (1, 2), left, (1, 2)))
    seed = to_concrete(contract(M, (2,), right, (1,)))
    H1 = one_site_h(mpo, i + 1, push_left_channels(lch, mpo, ex.U_ex, i), rch)
    return (seed = seed, Q0 = ex.V_ex, applyH = x -> apply_one_site(H1, x))
end

"The Hermitian map `sum_k c_k (a_k b_k' + conj(c_k) b_k a_k')`, matrix-free on tensors."
function _hop(terms)
    return x -> begin
        acc = nothing
        for (c, a, b) in terms
            t = (c * tensor_inner(b, x)) * a + (conj(c) * tensor_inner(a, x)) * b
            acc = acc === nothing ? t : acc + t
        end
        to_concrete(acc)
    end
end
