# The loss operators of dissipative_cbe_bug_implementation_plan.md (A1/B1 pair loss, A2/B2
# collective emission): `Q = Σ_μ L_μ' L_μ` as a compact MPO, to pass to `no_jump_mpo(H, Q)`.

"""
    doublon_loss_mpo(L; kappa) -> MPO

`Q = Σ_j κ_j n_j↑ n_j↓` on `L` fused d = 4 Hubbard sites ([`hub4_space`](@ref)): `L_j' L_j` for
the pair loss `L_j = √κ_j c_j↓ c_j↑`. `kappa` is one rate (A1, bulk loss) or one per site (B1: a
single nonzero entry). Charge-neutral, virtual dimension 2.
"""
function doublon_loss_mpo(L::Int; kappa)
    k = kappa isa Number ? fill(Float64(kappa), L) : collect(Float64, kappa)
    length(k) == L || throw(DimensionMismatch("doublon_loss_mpo got $(length(k)) rates for $L sites"))
    all(>=(0.0), k) || throw(ArgumentError("doublon_loss_mpo: loss rates must be nonnegative, got $k"))
    q = hub4_space()
    nu = to_concrete(lock(q.Fu', 2) * q.Fu)
    nd = to_concrete(lock(q.Fd', 2) * q.Fd)
    nupdn = to_concrete(lock(nu, 1) * nd)
    return onsite_mpo(L, [to_concrete(k[j] * nupdn) for j in 1:L]; Iloc = q.I)
end

"""
    collective_emission_mpo(L; gamma_plus, gamma_minus = gamma_plus, k, x = 1:L) -> MPO

`Q = L₊'L₊ + L₋'L₋` for the collective jumps `L_± = √Γ_± Σ_j e^{±i k x_j} S⁻_j` on a spin-1/2
chain (plan A2):

    Q = (Γ₊ + Γ₋) Σ_j n_j + Σ_{i<j} (c_ij S⁺_i S⁻_j + conj(c_ij) S⁻_i S⁺_j),
    c_ij = Γ₊ e^{ik(x_j - x_i)} + Γ₋ e^{-ik(x_j - x_i)},   n = S^z + 1/2.

Each `c_ij` is a sum of two separable phases `e^{∓ik x_i} e^{±ik x_j}`, so one transported
channel per (hop direction, phase) carries every pair: virtual dimension 6 at any `L`, with
`pair_mpo`'s charged transport block. No pair term is a separate operator. `Γ₊ ≠ Γ₋` makes the
couplings complex.
"""
function collective_emission_mpo(L::Int; gamma_plus::Real, gamma_minus::Real = gamma_plus,
                                 k::Real, x = 1:L)
    L >= 2 || throw(ArgumentError("collective_emission_mpo needs at least two sites, got $L"))
    length(x) == L || throw(DimensionMismatch(
        "collective_emission_mpo got $(length(x)) positions for $L sites"))
    (gamma_plus >= 0 && gamma_minus >= 0) || throw(ArgumentError(
        "collective_emission_mpo: rates must be nonnegative, got $gamma_plus, $gamma_minus"))
    q = local_space()
    n = to_concrete(q.Sz + 0.5 * q.I)
    # `xxz_chain(J = 2)` gives each hop vertex unit weight. ⛔ The MPO APPLIES a vertex as its
    # transpose (`spin_xx.jl`, the `liouville.jl` header): vertex 1 `(Sp, Sp')` is `S⁻_i S⁺_j`
    # and vertex 2 `(Sm, Sm')` is `S⁺_i S⁻_j`. Only complex couplings see it: with c_ij on
    # vertex 1 this built Q(-k) exactly (dt2a, test_dissipative_t2 U02).
    v = pair_vertices(xxz_chain(L; J = 2.0, delta = 0.0))
    length(v) == 2 || error("xxz_chain(delta = 0) gave $(length(v)) vertices, expected the two hops")
    # (vertex, rate, s): opens with e^{-isk x_i}, closes with rate * e^{isk x_j}. Vertex 2 carries
    # c_ij, vertex 1 its conjugate.
    chans = [(t, g, s) for (t, g, s) in ((2, gamma_plus, 1), (2, gamma_minus, -1),
                                         (1, gamma_plus, -1), (1, gamma_minus, 1)) if g != 0]
    isempty(chans) && throw(ArgumentError("collective_emission_mpo: both rates are zero"))
    m = length(chans)
    ncol = m + 2
    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid = _mpo_block(q.I, i, tl, tr, :none)
        mat = Matrix{Any}(nothing, ncol, ncol)
        i < L && (mat[1, 1] = Iid)
        i > 1 && (mat[ncol, ncol] = Iid)
        mat[1, ncol] = to_concrete((gamma_plus + gamma_minus) * _mpo_block(n, i, tl, tr, :none))
        for (c, (t, g, s)) in pairs(chans)
            vt = v[t]
            i < L && (mat[1, 1 + c] = to_concrete(cis(-s * k * x[i]) *
                          _mpo_block(vt.left, i, tl, tr, _side(vt.left, :left))))
            1 < i < L && (mat[1 + c, 1 + c] = _charged_transport(vt, q.I, i, tl, tr))
            i > 1 && (mat[1 + c, ncol] = to_concrete(g * vt.coeff * cis(s * k * x[i]) *
                          _mpo_block(vt.right, i, tl, tr, _side(vt.right, :right))))
        end
        i == 1 && (mat = reshape(mat[1, :], 1, ncol))
        i == L && (mat = reshape(mat[:, ncol], ncol, 1))
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end
