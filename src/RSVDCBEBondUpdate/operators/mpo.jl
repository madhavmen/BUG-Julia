# The Hamiltonian as an MPO with fused virtual legs, the form of the parallel-BUG sweep
# (arXiv:2606.28169 §1-2). `henv.jl` is the same operator with the virtual index named, kept as an
# independent witness: `test_mpo.jl` pins `apply_h_two_site` and `mpo_energy` against it.
# The leg and prime conventions are load-bearing: a wrong prime does not throw, it traces the wrong pair.

# ── the MPO ──────────────────────────────────────────────────────────────────

"""
    MPO(W)

`H` as a matrix product operator, one rank-4 tensor per site with legs `(w_l, s_ket, s_bra, w_r)`
(Eq. (1.3) of arXiv:2606.28169). The boundary virtual legs of `W[1]` and `W[end]` are dim 1, from
trimming, so no separate boundary vector is carried.

Built by [`mpo_from_terms`](@ref); interchangeable with [`XXZChain`](@ref) in the CBE-BUG sweep.
"""
struct MPO
    W::Vector{Any}
end

Base.length(mpo::MPO) = length(mpo.W)
Base.getindex(mpo::MPO, i::Int) = mpo.W[i]

"Virtual (MPO bond) dimension on each link `0 … L`, i.e. `w` of Eq. (1.3)."
mpo_virtual_dims(mpo::MPO) =
    vcat([sum(d for (_, d) in mpo[1].spaces[1]; init = 0)],
         [sum(d for (_, d) in mpo[i].spaces[4]; init = 0) for i in 1:length(mpo)])

# Each automaton row is written out literally and `oplus` direct-sums them, inferring the zero blocks
# from the row and column spaces. Rows: 1 = id, 1+t = half-open term t, n+2 = done.
# The op-leg carries the charge (under `:U1` the `Sp` leg is its channel's sector; under `:SU2` the S=1
# irrep), so nothing here is symmetry-specific.

"Retag an operator's two site legs onto site `i`, leaving any op-leg alone."
_mpo_retag(O, i::Int) = to_concrete(setitag(setitag(O, 1, "S,$i"), 2, "S,$i"))

"The ket-facing and bra-facing site legs of `O`, found by ARROW: a plain operator has the
ket leg at 1, an adjoint (`Sp'`) at 2. Hard-coding leg 1 breaks on the adjoint half of
every XY term -- the same trap `_apply_site_op` documents."
function _op_site_legs(O)
    ket = O.inds[1].dir == '+' ? 1 : 2
    return ket, 3 - ket
end

"""
    _mpo_block(O, i, tl, tr, side) -> TLArray

One entry of the automaton matrix, as a rank-4 `(w_l, s_ket, s_bra, w_r)` tensor.

`side` says which virtual leg the operator's op-leg becomes: `:left` for the left half of
a term (op-leg '-' → `w_r`), `:right` for the right half (op-leg '+' → `w_l`), `:none`
for an operator without one (both virtual legs trivial). A rank-2 operator is `:none`
whatever `side` says, which is how `Sz Sz` and the whole `:none` symmetry mode work.
"""
function _mpo_block(O, i::Int, tl::AbstractString, tr::AbstractString, side::Symbol)
    B = _mpo_retag(O, i)
    ket, bra = _op_site_legs(B)
    if length(B.inds) == 2 || side === :none
        length(B.inds) == 2 || throw(ArgumentError(
            "_mpo_block: side=:none needs a rank-2 operator, got rank $(length(B.inds))"))
        B = to_concrete(permutedims(B, (ket, bra)))
        return to_concrete(addSingleton(B, (1, 4); itag = (tl, tr), dir = ('+', '-')))
    elseif side === :left
        B = to_concrete(permutedims(B, (ket, bra, 3)))            # (s_ket, s_bra, op)
        B = to_concrete(addSingleton(B, (1,); itag = tl, dir = '+'))
        return to_concrete(setitag(B, 4, tr))                     # op-leg becomes w_r
    elseif side === :right
        B = to_concrete(permutedims(B, (3, ket, bra)))             # (op, s_ket, s_bra)
        B = to_concrete(addSingleton(B, (4,); itag = tr, dir = '-'))
        return to_concrete(setitag(B, 1, tl))                      # op-leg becomes w_l
    else
        throw(ArgumentError("_mpo_block: side must be :left, :right or :none, got $side"))
    end
end

"`side` for one half of a term: rank-3 operators put their op-leg on the virtual leg."
_side(O, s::Symbol) = length(O.inds) == 3 ? s : :none

"""
    mpo_from_terms(h::XXZChain) -> MPO

The MPO of the same Hamiltonian `h` represents as a term list, so the two paths cannot
disagree about `J`, `delta`, a coefficient or Telum's operator normalisation: the blocks
ARE `h.terms`.

The coefficient is applied once, on the closing (right) half of each term -- exactly where
`push_left_channels` applies it.
"""
function mpo_from_terms(h::XXZChain)
    L, terms = length(h), h.terms
    n = length(terms)
    Iloc = symmetry_mode() === :SU2 ? local_space(:SU2).I : local_space().I

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid = _mpo_block(Iloc, i, tl, tr, :none)
        # A zero block with the SAME spaces as `Iid`; `empty_tlarray` will not do, it
        # returns EMPTY space lists and the matrix oplus then cannot infer the column.
        Znil = to_concrete(0.0 * Iid)
        mat = Matrix{Any}(nothing, n + 2, n + 2)
        i < L && (mat[1, 1] = Iid)                       # transport the identity
        i > 1 && (mat[n + 2, n + 2] = Iid)               # ... on the far side of a term
        for t in 1:n
            i < L && (mat[1, 1 + t] =
                _mpo_block(terms[t].left, i, tl, tr, _side(terms[t].left, :left)))
            i > 1 && (mat[1 + t, n + 2] = to_concrete(terms[t].coeff *
                _mpo_block(terms[t].right, i, tl, tr, _side(terms[t].right, :right))))
        end
        if i == 1
            mat = reshape(mat[1, :], 1, n + 2)           # the `id` row only
            mat[1, n + 2] = Znil                         # ... and it cannot be `done` yet
        elseif i == L
            mat = reshape(mat[:, n + 2], n + 2, 1)       # the `done` column only
            mat[1, 1] = Znil                             # ... and nothing may stay `id`
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    xxz_mpo(L; J=1.0, delta=1.0) -> MPO

`H = J Σ_i (Sx Sx + Sy Sy + delta Sz Sz)` as an MPO. Convenience for
`mpo_from_terms(xxz_chain(L; J, delta))`.
"""
xxz_mpo(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0) =
    mpo_from_terms(xxz_chain(L; J = J, delta = delta))

"""
    heisenberg_su2_mpo(L; J=1.0) -> MPO

The SU(2) Heisenberg chain as an MPO: ONE term, whose `S=1` op-leg becomes the MPO bond,
so the virtual dimension is 3 multiplets rather than U(1)'s 5.
"""
heisenberg_su2_mpo(L::Int; J::Float64 = 1.0) =
    mpo_from_terms(heisenberg_su2_chain(L; J = J))

# ── sums of MPOs: the no-jump generator of a dissipative model ──────────────────────────────
# The integrators read `tau * mpo`, so `G = -iH - Q/2` is the MPO `H - (i/2) Q` at `tau = -im*dt`.
# A direct sum of virtual legs makes any builder's `H` and `Q` one operator without a new
# automaton; its virtual dimension is the sum of the two.

"""
    onsite_mpo(L, ops; Iloc = nothing) -> MPO

`Σ_i ops[i]` for rank-2 (charge-neutral) on-site operators, virtual dimension 2. `Iloc` is the
local identity, by default the mode's own (pass `hub4_space().I` for a fused d = 4 site).
"""
function onsite_mpo(L::Int, ops::AbstractVector; Iloc = nothing)
    L >= 2 || throw(ArgumentError("onsite_mpo needs at least two sites, got $L"))
    length(ops) == L || throw(DimensionMismatch(
        "onsite_mpo got $(length(ops)) operators for $L sites"))
    all(length(O.inds) == 2 for O in ops) || throw(ArgumentError(
        "onsite_mpo: every operator must be rank 2 (charge-neutral)"))
    Iloc = Iloc !== nothing ? Iloc :
           symmetry_mode() === :SU2 ? local_space(:SU2).I : local_space().I
    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid = _mpo_block(Iloc, i, tl, tr, :none)
        O   = _mpo_block(ops[i], i, tl, tr, :none)
        # rows/columns: 1 = id, 2 = done; the operator is the `id -> done` entry at EVERY site.
        # Filled by index: an `Any[Iid O]` literal concatenates the TLArrays instead.
        mat = Matrix{Any}(nothing, 2, 2)
        mat[1, 1], mat[1, 2], mat[2, 2] = Iid, O, Iid
        i == 1 && (mat = reshape(mat[1, :], 1, 2))
        i == L && (mat = reshape(mat[:, 2], 2, 1))
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    mpo_sum(A, B; a = 1, b = 1) -> MPO

`a A + b B` as one MPO: block-diagonal virtual legs, the end sites concatenated. Both must act on
the same site legs; the scalars sit on site 1. Every tensor is complex, as `b` may be.
"""
function mpo_sum(A::MPO, B::MPO; a::Number = 1.0, b::Number = 1.0)
    L = length(A)
    length(B) == L || throw(DimensionMismatch("mpo_sum: $L sites against $(length(B))"))
    L >= 2 || throw(ArgumentError("mpo_sum needs at least two sites, got $L"))
    Ws = Any[]
    for i in 1:L
        sa, sb = i == 1 ? (a, b) : (1, 1)
        Ai = to_concrete(ComplexF64(sa) * A[i])
        Bi = to_concrete(ComplexF64(sb) * B[i])
        W = if i == 1 || i == L
            oplus(Any[Ai, Bi], (i == 1 ? 4 : 1,))
        else
            mat = Matrix{Any}(nothing, 2, 2)
            mat[1, 1], mat[2, 2] = Ai, Bi
            oplus(mat, (1, 4))
        end
        push!(Ws, to_concrete(W))
    end
    return MPO(Ws)
end

"""
    no_jump_mpo(H, Q) -> MPO

`H_eff = H - (i/2) Q`, so `tau = -im*dt` integrates the no-jump equation `dψ/dt = Gψ`,
`G = -iH - Q/2`, with `Q = Σ_μ L_μ' L_μ` (the caller's MPO, positive semidefinite). Not Hermitian:
pass `hermitian = false` to `cbe_bug_step!` / `tdvp2_step!`, since Lanczos on it returns a wrong
vector without an error.
"""
no_jump_mpo(H::MPO, Q::MPO) = mpo_sum(H, Q; b = -0.5im)

# Long range: a geometric tail lambda^(j-i-1) needs one self-loop channel, lambda * I, so the virtual
# dimension stays constant in L and in the range. Restricted to rank-2 operators, since a rank-3
# self-loop would need a charged identity; the constructor throws on a rank-3 operator.
# lambda = 0 reproduces the nearest-neighbour MPO exactly, which `test_mpo.jl` uses as its check.

"""
    LongRangeTerm(left, right, coeff, decay)

One exponentially decaying two-site term: `coeff * decay^(r-1) * left_i right_{i+r}` for
every `r >= 1`. `decay = 0` is the nearest-neighbour term with strength `coeff`.

`left` and `right` must be rank-2 (charge-neutral) operators -- see the block comment
above for why a charged channel is refused rather than approximated.
"""
struct LongRangeTerm
    left::Any
    right::Any
    coeff::Float64
    decay::Float64
end

"""
    long_range_mpo(L, nn, lr) -> MPO

`H = (nearest-neighbour terms `nn`) + (exponentially decaying terms `lr`)` as an MPO with
virtual dimension `2 + |nn| + |lr|`, independent of `L` and of the interaction range.

`nn` is an [`XXZChain`](@ref) (or an empty term list); `lr` a vector of
[`LongRangeTerm`](@ref). The coefficient is applied on the closing half of each term, as
in [`mpo_from_terms`](@ref), so the two builders cannot disagree about normalisation.
"""
function long_range_mpo(L::Int, nn::XXZChain, lr::Vector{LongRangeTerm})
    length(nn) == L || throw(DimensionMismatch(
        "nn term list is for $(length(nn)) sites, asked for $L"))
    for (k, t) in pairs(lr)
        (length(t.left.inds) == 2 && length(t.right.inds) == 2) || throw(ArgumentError(
            "long_range_mpo: term $k carries a rank-3 operator, so its channel would need " *
            "a CHARGED self-loop identity. Only charge-neutral (rank-2) long-range " *
            "operators are supported -- see the block comment above `LongRangeTerm`."))
    end
    terms = nn.terms
    n, m = length(terms), length(lr)
    W = n + m + 2                                   # `id`, the channels, `done`
    Iloc = symmetry_mode() === :SU2 ? local_space(:SU2).I : local_space().I

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid  = _mpo_block(Iloc, i, tl, tr, :none)
        Znil = to_concrete(0.0 * Iid)
        mat = Matrix{Any}(nothing, W, W)
        i < L && (mat[1, 1] = Iid)
        i > 1 && (mat[W, W] = Iid)
        for t in 1:n
            i < L && (mat[1, 1 + t] =
                _mpo_block(terms[t].left, i, tl, tr, _side(terms[t].left, :left)))
            i > 1 && (mat[1 + t, W] = to_concrete(terms[t].coeff *
                _mpo_block(terms[t].right, i, tl, tr, _side(terms[t].right, :right))))
        end
        for k in 1:m
            c = 1 + n + k
            i < L && (mat[1, c] = _mpo_block(lr[k].left, i, tl, tr, :none))
            # THE SELF-LOOP. Needs both a predecessor and a successor site, so it exists
            # only in the interior -- at `i = 1` nothing has opened yet and at `i = L`
            # nothing may still be open, which the boundary trimming below enforces anyway.
            1 < i < L && lr[k].decay != 0.0 &&
                (mat[c, c] = to_concrete(lr[k].decay * Iid))
            i > 1 && (mat[c, W] = to_concrete(lr[k].coeff *
                _mpo_block(lr[k].right, i, tl, tr, :none)))
        end
        if i == 1
            mat = reshape(mat[1, :], 1, W)
            mat[1, W] = Znil                        # cannot be `done` before anything ran
        elseif i == L
            mat = reshape(mat[:, W], W, 1)
            mat[1, 1] = Znil                        # nothing may stay `id` past the end
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

"""
    long_range_zz_mpo(L; J=1.0, Jz=1.0, lambda=0.5) -> MPO

`H = J Σ_i (Sx Sx + Sy Sy)_{i,i+1} + Jz Σ_{i<j} lambda^(j-i-1) Sz_i Sz_j`.

A genuinely long-range, U(1)-symmetric test model: nearest-neighbour hopping with an
exponentially decaying Ising tail reaching every pair of sites. Virtual dimension `5`
whatever `L` is. `lambda = 0` reduces it to `xxz_mpo(L; J, delta = Jz/J)`.
"""
function long_range_zz_mpo(L::Int; J::Float64 = 1.0, Jz::Float64 = 1.0,
                           lambda::Float64 = 0.5)
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "long_range_zz_mpo needs a rank-2 Sz, which SU(2) does not have"))
    q = local_space()
    return long_range_mpo(L, xxz_chain(L; J = J, delta = 0.0),
                          [LongRangeTerm(q.Sz, q.Sz, Jz, lambda)])
end

# A power-law tail has no finite MPO: it is fitted by n geometric self-loops, and that fit error
# belongs to the Hamiltonian, not the integrator, so `fit_long_range` returns it with the MPO.
# The decay rates are prescribed on a log-spaced grid, not fitted: a Prony fit can return complex
# rates, which would make every tensor complex for a real H. They can be overridden.

"""
    LongRangeFit

What [`fit_long_range`](@ref) found, kept together so the error travels with the operator.

  - `coeffs`, `decays` -- the `c_k` and `lambda_k` of `sum_k c_k lambda_k^(r-1)`.
  - `target` -- the coupling asked for, `J(r)` for `r = 1 … R`.
  - `fitted` -- what the sum actually gives at those `r`.
  - `rel_errs` -- `|fitted - target| / |target|` per distance. The TAIL is where a fit
    fails, so the per-distance vector is kept rather than only its maximum.
  - `max_rel_err`, `l2_rel_err` -- the two summaries worth quoting.
"""
struct LongRangeFit
    coeffs::Vector{Float64}
    decays::Vector{Float64}
    target::Vector{Float64}
    fitted::Vector{Float64}
    rel_errs::Vector{Float64}
    max_rel_err::Float64
    l2_rel_err::Float64
end

"""
    default_decays(n) -> Vector{Float64}

`n` decay rates log-spaced in `(0, 1)`: `lambda_k = exp(-x_k)` with `x_k` geometric from
`0.05` to `4`. Short-range and long-range channels in one basis, so the fit has something
to put both the head and the tail of a power law on.
"""
default_decays(n::Int) =
    n <= 0 ? Float64[] :
    n == 1 ? [exp(-0.5)] :
    [exp(-0.05 * (4.0 / 0.05)^((k - 1) / (n - 1))) for k in 1:n]

"""
    fit_long_range(Jr; n_exp=4, decays=default_decays(n_exp)) -> LongRangeFit

Least-squares fit of the coupling sequence `Jr[r] = J(r)`, `r = 1 … length(Jr)`, by
`sum_k c_k lambda_k^(r-1)` at PRESCRIBED rates. One linear solve; see the block comment
above for why the rates are not fitted too.

The residual is reported per distance, because a power-law fit is good at the head and
poor at the tail and a single number hides that.
"""
function fit_long_range(Jr::AbstractVector{<:Real};
                        n_exp::Int = 4,
                        decays::AbstractVector{<:Real} = default_decays(n_exp))
    R = length(Jr)
    R >= 1 || throw(ArgumentError("fit_long_range needs at least one distance"))
    lam = collect(Float64, decays)
    all(0 .< lam .< 1) || throw(ArgumentError("decays must lie strictly in (0, 1)"))
    M = [lam[k]^(r - 1) for r in 1:R, k in 1:length(lam)]
    c = M \ collect(Float64, Jr)
    fitted = M * c
    tgt = collect(Float64, Jr)
    rel = abs.(fitted .- tgt) ./ max.(abs.(tgt), eps())
    l2 = norm(fitted .- tgt) / max(norm(tgt), eps())
    return LongRangeFit(c, lam, tgt, fitted, rel, maximum(rel; init = 0.0), l2)
end

"""
    long_range_terms(fit, left, right) -> Vector{LongRangeTerm}

The fit as one self-loop channel per exponential.
"""
long_range_terms(fit::LongRangeFit, left, right) =
    [LongRangeTerm(left, right, fit.coeffs[k], fit.decays[k])
     for k in 1:length(fit.coeffs)]

"""
    power_law_zz_mpo(L; J=1.0, Jz=1.0, alpha=3.0, n_exp=4, decays=...) -> (MPO, LongRangeFit)

`H = J Σ_i (Sx Sx + Sy Sy)_{i,i+1} + Σ_{i<j} Jz |i-j|^-alpha Sz_i Sz_j`, with the power law
fitted by `n_exp` geometric channels.

RETURNS THE FIT AS WELL AS THE OPERATOR, and the two-tuple is the point: unlike
[`long_range_zz_mpo`](@ref) this MPO is NOT the Hamiltonian asked for, and how far off it is
at each distance is `fit.rel_errs`. Quoting a result from it without quoting that is
quoting an unstated model.
"""
function power_law_zz_mpo(L::Int; J::Float64 = 1.0, Jz::Float64 = 1.0,
                          alpha::Float64 = 3.0, n_exp::Int = 4,
                          decays::AbstractVector{<:Real} = default_decays(n_exp))
    symmetry_mode() === :SU2 && throw(ArgumentError(
        "power_law_zz_mpo needs a rank-2 Sz, which SU(2) does not have"))
    L >= 2 || throw(ArgumentError("power_law_zz_mpo needs at least two sites"))
    q = local_space()
    fit = fit_long_range([Jz * float(r)^(-alpha) for r in 1:(L - 1)];
                         n_exp = n_exp, decays = decays)
    mpo = long_range_mpo(L, xxz_chain(L; J = J, delta = 0.0),
                         long_range_terms(fit, q.Sz, q.Sz))
    return mpo, fit
end
