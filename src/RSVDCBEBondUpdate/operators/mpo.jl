# The Hamiltonian as a genuine MPO, and its environments, in the form the parallel-BUG
# sweep of arXiv:2606.28169 §1--2 is written in.
#
# WHY THIS EXISTS ALONGSIDE `henv.jl`. The channel recursion in `henv.jl` is an MPO
# contraction with the virtual index NAMED rather than fused: `id`, `open[t]`, `done` are
# the three rows of
#
#          ( I  O_1^L ... O_n^L   0  )
#      W = ( 0   0          0   c_1 O_1^R )
#          ( ...                  ...  )
#          ( 0   0          0     I  )
#
# stored as separate tensors. That is the same operator, but it is NOT the object the
# paper's equations name, and it hard-codes the automaton for a nearest-neighbour uniform
# term list -- the cost `docs/cbe_lubich_sweep.tex` §"What the factored form costs" records.
#
# This file fuses that virtual index into a real leg, so the sweep runs on
#
#   Eq. (1.3)   H = sum_w W^[1]_{w_0 s_1 z_1 w_1} ... W^[L]_{w_{L-1} s_L z_L w_L}
#   Eq. (1.8)   L^[i]_{v_i w_i v'_i} = sum L^[i-1] T^[i] W^[i] conj(T^[i]),   L^[0] = 1
#   Eq. (1.7)   H_eff^[i] = L^[i-1] W^[i] R^[i+1]
#
# with `L^[i]`/`R^[i]` rank-3 (bra, mpo, ket) tensors -- one environment per link, not
# `2 + |terms|` of them -- and the effective Hamiltonian a plain contraction with no case
# analysis over channel kinds.
#
# THE TWO PATHS ARE THE SAME OPERATOR AND MUST STAY THAT WAY. `test_mpo.jl` pins
# `apply_h_two_site` element by element against the channel version at every bond, and
# `mpo_energy` against `env_energy` and the dense reference. The channel path is kept
# precisely to be that independent witness: a leg/arrow/prime slip here cannot cancel
# against code that shares none of these contractions.
#
# WHAT IS GAINED, beyond matching the paper's notation:
#
#   * a site-dependent `W^[i]`, so anything expressible as an MPO can be run -- the
#     factored form could only ever express one uniform nearest-neighbour term list;
#   * one environment tensor per link instead of `2 + |terms|`, and one contraction per
#     application instead of the five-case split (a)--(e);
#   * `zero_site_h` becomes `L` and `R` joined over the MPO leg, which is Eq. (1.7) at a
#     bond rather than a hand-derived three-way sum.
#
# LEG CONVENTIONS, and every one of them is load-bearing.
#
#   MPO site tensor   `(w_l '+', s_ket '+', s_bra '-', w_r '-')`
#                     `s_ket` contracts the ket state tensor's physical leg (dir '-'),
#                     `s_bra` the bra's (dir '+' after the adjoint). That is exactly a
#                     local operator's own `(site '+', site '-')` layout with a virtual
#                     leg glued to each side, which is why the blocks below need no
#                     arrow surgery: the op-leg of `Sp` is already '-' (it becomes a
#                     `w_r`) and the op-leg of `Sp'` is already '+' (it becomes a `w_l`).
#
#   environment       `(bra, mpo, ket)` with the BRA leg PRIMED, the same convention
#                     `henv.jl` uses, so the ket leg pairs with the next site tensor's
#                     link leg and the bra leg with the bra tensor's. A wrong prime level
#                     does not throw -- it silently traces the wrong pair.
#
#   boundaries        `nothing`, NOT a materialised tensor, again as in `henv.jl`: at link
#                     1 the environment is the identity on a dim-1 vacuum leg, and the
#                     MPO's own boundary leg is the dim-1 leg left by trimming `W^[1]` to
#                     its `id` row. Both are handled by branch, and the singleton is
#                     dropped with `deleteSingleton` rather than contracted.

# ── the MPO ──────────────────────────────────────────────────────────────────

"""
    MPO(W)

`H` as a matrix product operator: one rank-4 tensor per site with legs
`(w_l, s_ket, s_bra, w_r)`, Eq. (1.3) of arXiv:2606.28169.

`W[1]`'s left virtual leg and `W[end]`'s right virtual leg are dim-1 -- the boundary
vectors are built in by trimming, so no separate boundary vector has to be carried.

Built by [`mpo_from_terms`](@ref). Interchangeable with [`XXZChain`](@ref) everywhere the
CBE-BUG sweep takes a Hamiltonian: `boundary_channels`, `left_env_stack`,
`right_env_stack`, `push_left_channels`, `push_right_channels`, `apply_h_two_site`,
`sketch_h_left`, `sketch_h_right`, `cbe_expand` and `zero_site_h` all have a method for it.
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

# ── building the site tensors from the term list ──────────────────────────────
#
# The automaton matrix is written out literally, one entry per non-zero block, and
# `oplus(mat, (1, 4))` direct-sums the rows on leg 1 and the columns on leg 4 -- Telum
# infers the zero blocks from the row/column spaces. Row/column order is
#
#   1        the `id` channel   (nothing has happened yet)
#   1+t      term `t` half-open (its left operator has been placed)
#   n+2      the `done` channel (a complete term lies behind us)
#
# so a path through the chain that opens at site `i` and closes at site `i+1` is exactly
# one nearest-neighbour term, and `W^[1]` keeping only row 1 while `W^[L]` keeps only
# column `n+2` is what forbids terms running off either end.
#
# The op-leg carries the charge. Under `:U1` the leg of `Sp` is a single `((2,),)` sector
# and that IS the virtual sector of its channel; under `:SU2` it is the `S=1` irrep of the
# one `S·S` term. Nothing here is symmetry-specific -- the virtual space is assembled out
# of the operators' own legs, so whatever charge makes the pair allowed is what the MPO
# bond carries.

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

# ── LONG RANGE: one self-loop channel per exponentially decaying term ─────────
#
# THE POINT OF THE MPO LAYER, and the one thing the factored channel form in `henv.jl`
# cannot express at all. An interaction that decays geometrically,
#
#     H_lr = sum_{i<j} c * lambda^(j-i-1) * A_i B_j
#
# needs no extra channel per distance: give the half-open channel a SELF-LOOP block
# `lambda * I` and a path that opens at `i`, loops over `i+1 … j-1` and closes at `j`
# picks up exactly `lambda^(j-i-1)`. So the virtual dimension is CONSTANT in `L` and in
# the interaction range -- `w = 2 + |nn terms| + |lr terms|` -- and everything downstream
# (the environments of Eq. 1.8, `apply_h_two_site`, `zero_site_h`, the CBE sketches) is
# untouched, because none of it ever inspects `W`'s shape.
#
#   row/col 1              `id`
#   row/col 1+t            nearest-neighbour term `t`, half-open
#   row/col 1+n+k          long-range term `k`, half-open AND self-looping
#   row/col n+m+2          `done`
#
# `lambda = 0` collapses the self-loop and reproduces the nearest-neighbour MPO exactly,
# which is how `test_mpo.jl` pins the construction against the already-validated one.
#
# RESTRICTED TO CHARGE-NEUTRAL OPERATORS, deliberately and with the reason stated rather
# than the case silently mishandled. A rank-3 operator (`Sp` under U(1), `S` under SU(2))
# puts its op-leg ON the virtual leg, so that channel's virtual space is CHARGED, and its
# self-loop would have to be the identity on that charged space rather than the trivial
# `lambda * Iid` built here. Telum's `getIdentity` flips input arrows and reports dual
# labels (`sectors.jl:77`), so that block needs real care and is not written until there
# is a model that wants it. `Sz Sz` -- rank 2, `_side` returns `:none`, virtual space
# trivial -- covers the long-range Ising/XXZ family and is what this builds. The
# constructor THROWS on a rank-3 operator rather than emitting a wrong self-loop.

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

# ── an ARBITRARY coupling: n channels, and a fit error that must be reported ──
#
# A geometric tail is carried EXACTLY by one self-loop, so there is nothing to measure
# there -- `long_range_zz_mpo` is not an approximation of anything. Any other decay is a
# different matter: a power law `J(r) = J r^-alpha` has no finite-dimensional MPO, and the
# standard construction fits it by a sum of `n` geometrics,
#
#     J(r)  ~=  sum_{k=1}^{n} c_k lambda_k^(r-1),
#
# one self-loop channel per `k`, so `w = 2 + |nn| + n` -- still independent of `L`, but now
# carrying a REAL error that grows at the tail and shrinks with `n`. That error is a
# property of the Hamiltonian being simulated, not of the integrator, and it is not
# separable from the result afterwards: it must be reported with the MPO, which is why
# [`fit_long_range`](@ref) returns it and `power_law_zz_mpo` hands it back alongside the
# operator rather than discarding it.
#
# THE DECAY RATES ARE PRESCRIBED, NOT FITTED, and that is a deliberate robustness choice.
# Solving for `lambda_k` too (Prony, matrix pencil) is the sharper fit at a given `n`, but
# it is a nonlinear problem on a Hankel matrix whose conditioning collapses as `n` grows,
# and it happily returns complex or negative rates -- which are legitimate mathematically
# (they cancel in conjugate pairs) but make every MPO tensor complex for a real, Hermitian
# `H`. With the rates FIXED on a log-spaced grid the fit is one linear least-squares solve,
# always real, monotone in `n` in practice, and reproducible. The rates can be overridden
# if a sharper fit is wanted.

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
