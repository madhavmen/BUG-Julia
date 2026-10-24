# The loss operators of dissipative_cbe_bug_implementation_plan.md (A1/B1 pair loss, A2/B2
# collective emission): `Q = Σ_μ L_μ' L_μ` as a compact MPO, to pass to `no_jump_mpo(H, Q)`, and the
# jumps `L_μ` themselves as charged MPOs for the trajectory engine (Task 4, `quantum_jumps.jl`).

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

# ── the jumps ───────────────────────────────────────────────────────────────────────────────
# A jump changes the sector, so its MPO is a pair-MPO channel that opens and never closes: `id`
# up to the opening site, the CHARGED channel after it, and the right boundary leg carries the
# jump's charge (`mpo_product` fuses it into the state's). ⛔ The MPO applies a stored block as
# its transpose (`spin_xx.jl`): the stored `Sp` applies `S⁻`, the stored `f'` applies `f`.

"""
    jump_mpo(L, op, close, coeffs; Iloc) -> MPO

`Σ_j coeffs[j] op_j` for a charged one-site operator, applied as the transpose of the stored
`op` (see the section header). `close` is `op`'s closing partner in a hop (`(Sp, Sp')`), used
only to orient the charged transport. Sites outside the first and last nonzero coefficient carry
one channel, so a single-site jump has virtual dimension 1.
"""
function jump_mpo(L::Int, op, close, coeffs::AbstractVector; Iloc)
    length(coeffs) == L || throw(DimensionMismatch("jump_mpo got $(length(coeffs)) coefficients for $L sites"))
    act = findall(!iszero, coeffs)
    isempty(act) && throw(ArgumentError("jump_mpo: every coefficient is zero"))
    j0, j1 = extrema(act)
    vt = PairVertex(op, close, 1.0)
    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        # `id` lives on links 0..j1-1, the charged `done` channel on links j0..L
        rows = [r for r in (:id, :done) if (r === :id ? i <= j1 : i - 1 >= j0)]
        cols = [c for c in (:id, :done) if (c === :id ? i < j1 : i >= j0)]
        mat = Matrix{Any}(nothing, length(rows), length(cols))
        for (a, r) in pairs(rows), (b, c) in pairs(cols)
            mat[a, b] = (r, c) === (:id, :id) ? _mpo_block(Iloc, i, tl, tr, :none) :
                        (r, c) === (:done, :done) ? _charged_transport(vt, Iloc, i, tl, tr) :
                        (r, c) === (:id, :done) && !iszero(coeffs[i]) ?
                            to_concrete(ComplexF64(coeffs[i]) * _mpo_block(op, i, tl, tr, :left)) :
                        nothing
        end
        push!(Ws, to_concrete(size(mat) == (1, 1) ? mat[1, 1] : oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    SiteOp(j, O, c)

The one-site factor `c O_j` of a jump, `O` a charged (rank-3) local operator applied by
`_apply_site_op`, i.e. as the TRANSPOSE of the stored tensor -- the convention of the environments
and the MPO (`spin_xx.jl`, the `liouville.jl` header): stored `Sp` applies `S⁻`, stored `f'` applies
`f` (dt4f: a stored `Sm` applied `S⁺`). Its charge leg is carried through the links right of `j`
into the boundary (`operators/mpo_apply.jl`). A local jump needs no MPO channel, and the d = 4
fermion operators have no charged transport block.
"""
struct SiteOp
    j::Int
    O::Any
    c::ComplexF64
end

"""
    JumpChannel(name, factors)

One Lindblad jump `L_μ`, the product of `factors` (each an `MPO` or a [`SiteOp`](@ref)) applied
first to last (a pair loss is two fermion annihilators on one site). `name` labels it in a
trajectory's event list.
"""
struct JumpChannel
    name::String
    factors::Vector{Any}
end

"""
The factor that makes a stored operator act with unit matrix element: `1 / <dst|F|src>` for one
probe pair of product states. Telum's spin operators are spherical tensor components (`Sp` is
`-S⁺/√2`), and a jump's weight `||L ψ||²` must be `<ψ|L'L|ψ>` with `L'L` the `Q` the generator
uses, not a multiple of it. Computed, not hard-coded, so a convention change cannot pass silently.
Two-site probes: `overlap` of states in different sectors is then an exact 0, not a leg mismatch.
"""
function _unit_scale(F, src::SymMPS, dst::SymMPS)
    x = overlap(dst, factor_product(F, src))
    abs(x) > 1e-12 || error("_unit_scale: the probe matrix element vanishes; wrong probe pair")
    return 1 / x
end

"""
    pair_loss_channels(L; kappa) -> Vector{JumpChannel}

`L_j = √κ_j c_j↓ c_j↑` on fused d = 4 sites, one channel per site with `κ_j > 0`, so that
`Σ_j L_j' L_j = doublon_loss_mpo(L; kappa)`. The two annihilators' Jordan-Wigner strings over
sites `< j` cancel, so each is a bare one-site factor (`c↑` first). Each is normalised on one
matrix element (`<0|c↑|↑>`, `<0|c↓|↓>`), which leaves the intra-site sign structure as Telum's.
"""
function pair_loss_channels(L::Int; kappa)
    k = kappa isa Number ? fill(Float64(kappa), L) : collect(Float64, kappa)
    length(k) == L || throw(DimensionMismatch("pair_loss_channels got $(length(k)) rates for $L sites"))
    all(>=(0.0), k) || throw(ArgumentError("pair_loss_channels: rates must be nonnegative, got $k"))
    q = hub4_space()
    fu, fd = to_concrete(q.Fu'), to_concrete(q.Fd')                  # applied as c↑, c↓
    e2 = hub4_product_state([:empty, :empty])
    su = _unit_scale(SiteOp(1, fu, 1), hub4_product_state([:up, :empty]), e2)
    sd = _unit_scale(SiteOp(1, fd, 1), hub4_product_state([:down, :empty]), e2)
    return [JumpChannel("pair$j", Any[SiteOp(j, fu, sqrt(k[j]) * su), SiteOp(j, fd, sd)])
            for j in 1:L if k[j] > 0]
end

"""
    collective_emission_channels(L; gamma_plus, gamma_minus = gamma_plus, k, x = 1:L)

`L_± = √Γ_± Σ_j e^{±i k x_j} S⁻_j` (plan A2/B2), the jumps whose `L₊'L₊ + L₋'L₋` is
`collective_emission_mpo` with the same arguments; each a charged [`jump_mpo`](@ref) on the stored
`Sp` (applied as `S⁻`), scaled to unit `S⁻` matrix elements. Zero rates give no channel.
"""
function collective_emission_channels(L::Int; gamma_plus::Real, gamma_minus::Real = gamma_plus,
                                      k::Real, x = 1:L)
    length(x) == L || throw(DimensionMismatch("got $(length(x)) positions for $L sites"))
    q = local_space()
    sp = to_concrete(q.Sp')
    s = _unit_scale(jump_mpo(2, q.Sp, sp, [1.0, 0.0]; Iloc = q.I), product_state([:up, :down]),
                    product_state([:down, :down]))
    ch = JumpChannel[]
    for (name, g, sg) in (("L+", gamma_plus, 1), ("L-", gamma_minus, -1))
        g > 0 || continue
        push!(ch, JumpChannel(name, Any[jump_mpo(L, q.Sp, sp,
                                                 [s * sqrt(g) * cis(sg * k * x[j]) for j in 1:L];
                                                 Iloc = q.I)]))
    end
    isempty(ch) && throw(ArgumentError("collective_emission_channels: both rates are zero"))
    return ch
end

"""
    site_emission_channels(L; gamma) -> Vector{JumpChannel}

Independent `L_j = √γ_j S⁻_j` on a spin-1/2 chain (`Q = Σ_j γ_j n_j`), one [`SiteOp`](@ref) channel
per site with `γ_j > 0`: the local-loss counterpart of the collective channels.
"""
function site_emission_channels(L::Int; gamma)
    g = gamma isa Number ? fill(Float64(gamma), L) : collect(Float64, gamma)
    length(g) == L || throw(DimensionMismatch("got $(length(g)) rates for $L sites"))
    q = local_space()
    s = _unit_scale(SiteOp(1, q.Sp, 1), product_state([:up, :down]), product_state([:down, :down]))
    return [JumpChannel("site$j", Any[SiteOp(j, q.Sp, sqrt(g[j]) * s)]) for j in 1:L if g[j] > 0]
end
