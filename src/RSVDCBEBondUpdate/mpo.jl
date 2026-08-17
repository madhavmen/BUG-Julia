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

# ── environments: Eq. (1.8) ──────────────────────────────────────────────────

"""
    MPOLink(E)

The MPO environment on ONE link: a rank-3 `(bra, mpo, ket)` tensor, or `nothing` at a
chain boundary (where it is the identity on a dim-1 vacuum leg).

The counterpart of [`ChannelSet`](@ref), and deliberately the same shape of object: a
sweep either CARRIES it bond to bond (`push_left_channels`) or reads a link out of a
prebuilt stack (`left_channels`).
"""
struct MPOLink
    E::Any
end

"The environment on a chain boundary: `L^[0] = 1` of Eq. (1.8)."
boundary_channels(::MPO) = MPOLink(nothing)

"""
    _mpo_left_step(E, A, W) -> TLArray

One step of Eq. (1.8): push the left environment on link `i` through site tensor `A` and
MPO tensor `W`, returning the environment on link `i+1` with legs `(bra, mpo, ket)`.

`A` may be a state tensor OR an expanded frame -- both carry an MPS tensor's
`(link_l, site, link_r)` layout, which is what lets `zero_site_h` push an environment
through `U_ex` with this same function.
"""
function _mpo_left_step(E, A, W)
    if E === nothing
        # Boundary: `link_l` is the dim-1 vacuum, contracted bra-to-ket, and `W`'s `w_l`
        # is the dim-1 leg left by trimming, dropped rather than contracted.
        T = contract(W, (2,), A, (2,))                  # (w_l, s_bra, w_r, ℓ_l, ℓ_r)
        T = contract(to_concrete(T), (2, 4), _bra_left_boundary(A), (2, 1))
        T = to_concrete(deleteSingleton(to_concrete(T), 1))   # (w_r, ℓ_r, ℓ_r')
        return to_concrete(permutedims(T, (3, 1, 2)))
    end
    T = contract(E, (3,), A, (1,))                       # (bra, w, s, ℓ_r)
    T = contract(to_concrete(T), (2, 3), W, (1, 2))      # (bra, ℓ_r, s_bra, w_r)
    T = contract(to_concrete(T), (1, 3), _bra_interior(A), (1, 2))  # (ℓ_r, w_r, ℓ_r')
    return to_concrete(permutedims(to_concrete(T), (3, 2, 1)))
end

"""
    _mpo_right_step(E, A, W) -> TLArray

The mirror: push the right environment on link `i+1` through site `i`, giving link `i`.

No asymmetry to get backwards here -- unlike the channel recursion, which has to OPEN
with `term.right` and CLOSE with `term.left` on the way back. The MPO tensor is the same
object read the other way, which is the second reason this file exists.
"""
function _mpo_right_step(E, A, W)
    if E === nothing
        T = contract(W, (2,), A, (2,))                   # (w_l, s_bra, w_r, ℓ_l, ℓ_r)
        T = contract(to_concrete(T), (2, 5), _bra_right_boundary(A), (2, 3))
        T = to_concrete(deleteSingleton(to_concrete(T), 2))   # (w_l, ℓ_l, ℓ_l')
        return to_concrete(permutedims(T, (3, 1, 2)))
    end
    T = contract(A, (3,), E, (3,))                       # (ℓ_l, s, bra, w)
    T = contract(to_concrete(T), (2, 4), W, (2, 4))      # (ℓ_l, bra, w_l, s_bra)
    T = contract(to_concrete(T), (2, 4), _bra_interior(A), (3, 2))  # (ℓ_l, w_l, ℓ_l')
    return to_concrete(permutedims(to_concrete(T), (3, 2, 1)))
end

"Carry the left environment on link `i` through site `i` onto link `i+1`. `open_next` is
accepted for signature compatibility with the channel path and ignored: an MPO cannot
leave a term dangling past the boundary, the trimmed `W^[L]` forbids it."
push_left_channels(cs::MPOLink, mpo::MPO, A, i::Int; open_next::Bool = true) =
    MPOLink(_mpo_left_step(cs.E, A, mpo[i]))

"Mirror of [`push_left_channels`](@ref) for the right environment."
push_right_channels(cs::MPOLink, mpo::MPO, A, i::Int; open_next::Bool = true) =
    MPOLink(_mpo_right_step(cs.E, A, mpo[i]))

"""
    MPOLeftEnvStack / MPORightEnvStack

Prebuilt environments, indexed by LINK exactly as [`LeftEnvStack`](@ref) /
[`RightEnvStack`](@ref) are: entry `i` of the left stack is everything at sites `< i`,
entry `i` of the right stack everything at sites `>= i`. Links outside the sweep's reach
hold `missing`, so reading one by mistake throws instead of being taken for a boundary.
"""
struct MPOLeftEnvStack
    E::Vector{Any}
end

struct MPORightEnvStack
    E::Vector{Any}
end

"""
    left_env_stack(psi, mpo; upto=length(psi)-1) -> MPOLeftEnvStack

Build the left MPO environments on links `1 … upto+1`. The stack IS the carry, recorded
link by link, so a sweep that carries and a sweep that prebuilds cannot drift.
"""
function left_env_stack(psi::SymMPS, mpo::MPO; upto::Int = length(psi) - 1)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    0 <= upto <= L || throw(ArgumentError("upto must be in 0:$L, got $upto"))
    E = Any[missing for _ in 1:(L + 1)]
    E[1] = nothing
    cur = nothing
    for i in 1:upto
        cur = _mpo_left_step(cur, psi[i], mpo[i])
        E[i + 1] = cur
    end
    return MPOLeftEnvStack(E)
end

"""
    right_env_stack(psi, mpo; downto=2) -> MPORightEnvStack

Build the right MPO environments on links `downto … L+1`.
"""
function right_env_stack(psi::SymMPS, mpo::MPO; downto::Int = 2)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    1 <= downto <= L + 1 || throw(ArgumentError(
        "downto must be in 1:$(L + 1), got $downto"))
    E = Any[missing for _ in 1:(L + 1)]
    E[L + 1] = nothing
    cur = nothing
    for i in L:-1:downto
        cur = _mpo_right_step(cur, psi[i], mpo[i])
        E[i] = cur
    end
    return MPORightEnvStack(E)
end

"Link `i` of a prebuilt left stack."
left_channels(st::MPOLeftEnvStack, i::Int) = MPOLink(st.E[i])

"Link `i` of a prebuilt right stack."
right_channels(st::MPORightEnvStack, i::Int) = MPOLink(st.E[i])

"""
    _mpo_left_close(E, A, W) -> ComplexF64

As [`_mpo_left_step`](@ref) at the LAST site: `link_r` is the dim-1 vacuum boundary so it
is contracted bra-to-ket, and `W`'s `w_r` is the dim-1 leg left by trimming to the `done`
column, so it is dropped. The result is a scalar. `deleteSingleton` doubles as the
assertion that the trimming really did leave a singleton there.
"""
function _mpo_left_close(E, A, W)
    T = contract(E, (3,), A, (1,))                       # (bra, w, s, ℓ_r)
    T = contract(to_concrete(T), (2, 3), W, (1, 2))       # (bra, ℓ_r, s_bra, w_r)
    T = to_concrete(deleteSingleton(to_concrete(T), 4))   # (bra, ℓ_r, s_bra)
    s = contract(T, (1, 3, 2), _bra_right_boundary(A), (1, 2, 3))
    return ComplexF64(to_concrete(s)[])
end

"""
    mpo_energy(psi, mpo) -> ComplexF64

`⟨ψ|H|ψ⟩` from the MPO recursion, closing both boundaries. The independent check on
[`env_energy`](@ref): the two share no contraction.

Unnormalised, like `env_energy` -- compare against `energy(psi, gates) * norm(psi)^2`
unless the state is normalised. Exact in any gauge: bra and ket are contracted
explicitly and no isometry is assumed anywhere.
"""
function mpo_energy(psi::SymMPS, mpo::MPO)
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    E = nothing
    for i in 1:(L - 1)
        E = _mpo_left_step(E, psi[i], mpo[i])
    end
    return _mpo_left_close(E, psi[L], mpo[L])
end

# ── the effective Hamiltonians: Eq. (1.7) ────────────────────────────────────

"""
    apply_h_two_site(Theta, mpo, i, lenv, renv) -> TLArray

`H Theta` for the two-site block at bond `(i, i+1)`, legs
`(link_l, site_l, site_r, link_r)` in and out -- Eq. (1.7) with two MPO tensors between
the environments.

ONE contraction chain, against the channel version's five-case sum: the automaton's
states are summed over inside the MPO leg instead of being enumerated here. `lenv` must
sit on link `i` and `renv` on link `i+2`, and the state they were built from must be
canonical at the bond, since a boundary environment is the identity only in that gauge.
"""
function apply_h_two_site(Theta, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink)
    W1, W2 = mpo[i], mpo[i + 1]

    # Left end: attach the environment, or -- at the boundary -- keep `Theta`'s own link
    # leg and drop the MPO's dim-1 boundary leg.
    T = if lenv.E === nothing
        X = to_concrete(contract(Theta, (2,), W1, (2,)))  # (ℓ_l,s_r,ℓ_r,w_l,s_bra,w_mid)
        to_concrete(deleteSingleton(X, 4))
    else
        X = to_concrete(contract(lenv.E, (3,), Theta, (1,)))       # (bra,w,s_l,s_r,ℓ_r)
        to_concrete(contract(X, (2, 3), W1, (1, 2)))
    end                                                  # (link_l, s_r, ℓ_r, s_bra_l, w)
    T = to_concrete(contract(T, (5, 2), W2, (1, 2)))      # (link_l,ℓ_r,s_bra_l,s_bra_r,w)

    T = if renv.E === nothing
        X = to_concrete(deleteSingleton(T, 5))            # (link_l, ℓ_r, s_bra_l, s_bra_r)
        to_concrete(permutedims(X, (1, 3, 4, 2)))
    else
        to_concrete(contract(T, (5, 2), renv.E, (2, 3)))  # (link_l,s_bra_l,s_bra_r,bra_r)
    end

    lenv.E === nothing || (T = _unprime(T, 1))
    renv.E === nothing || (T = _unprime(T, 4))
    return T
end

apply_h_two_site(Theta, mpo::MPO, i::Int,
                 lenv::MPOLeftEnvStack, renv::MPORightEnvStack) =
    apply_h_two_site(Theta, mpo, i, left_channels(lenv, i), right_channels(renv, i + 2))

# The CBE sketch, unchanged in substance: only `H Theta` comes from the MPO now. The
# projector and the sketch are the reference's (`RSVDpreBE0SiQS.m`), and nothing in the
# selection knows how `H` is stored.

"See the [`sketch_h_left`](@ref) for a term list; this is the same sketch on MPO
environments."
function sketch_h_left(f::BondFrame, mpo::MPO, i::Int,
                       lenv::MPOLink, renv::MPOLink, Om)
    HT = apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)
    c  = contract(f.U0', (1, 2), HT, (1, 2))               # (bond, site_r, link_r)
    A  = to_concrete(HT - to_concrete(contract(f.U0, (3,), c, (1,))))
    return to_concrete(contract(A, (3, 4), Om', (2, 3)))   # (link_l, site_l, g)
end

"Mirror of [`sketch_h_left`](@ref) on MPO environments."
function sketch_h_right(f::BondFrame, mpo::MPO, i::Int,
                        lenv::MPOLink, renv::MPOLink, Om)
    HT = apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)
    c  = contract(HT, (3, 4), f.V0', (2, 3))               # (link_l, site_l, bond)
    A  = to_concrete(HT - to_concrete(contract(c, (3,), f.V0, (1,))))
    Y  = contract(A, (1, 2), Om', (1, 2))                  # (site_r, link_r, g)
    return to_concrete(permutedims(Y, (3, 1, 2)))          # (g, site_r, link_r)
end

"The bond expansion at bond `i`, driven by MPO environments."
cbe_expand(f::BondFrame, mpo::MPO, i::Int,
           lenv::MPOLink, renv::MPOLink; kwargs...) =
    cbe_expand(f,
               Om -> sketch_h_left(f, mpo, i, lenv, renv, Om),
               Om -> sketch_h_right(f, mpo, i, lenv, renv, Om); kwargs...)

"""
    MPOOneSiteH

The single-site effective Hamiltonian of Eq. (1.7), `H_eff^[j] = L^[j-1] W^[j] R^[j+1]`, held
as its three factors rather than contracted: the operand is a site tensor, so contracting them
first would build a `(chi d) x (chi d)` matrix where three sequential contractions cost
`O(chi^2 d w)`.

`nothing` on either environment is the chain boundary, where the MPO's own dim-1 leg is
dropped instead of contracted -- see the file header.
"""
struct MPOOneSiteH
    l::Any                    # (bra, mpo, ket) at link j, or `nothing`
    w::Any                    # W^[j]
    r::Any                    # (bra, mpo, ket) at link j+1, or `nothing`
end

"""
    one_site_h(mpo, j, lenv, renv) -> MPOOneSiteH

`H_eff` at site `j`. `lenv` must sit on link `j` and `renv` on link `j+1` -- the same
convention the channel-path method of this function uses.
"""
one_site_h(mpo::MPO, j::Int, lenv::MPOLink, renv::MPOLink) =
    MPOOneSiteH(lenv.E, mpo[j], renv.E)

one_site_h(mpo::MPO, j::Int, lenv::MPOLeftEnvStack, renv::MPORightEnvStack) =
    one_site_h(mpo, j, left_channels(lenv, j), right_channels(renv, j + 1))

"""
    apply_one_site(H1::MPOOneSiteH, A) -> TLArray

`H_eff A` for a site tensor `A` with legs `(link_l, site, link_r)`, in and out.
"""
function apply_one_site(H1::MPOOneSiteH, A)
    T = if H1.l === nothing
        X = to_concrete(contract(A, (2,), H1.w, (2,)))     # (ℓ_l, ℓ_r, w_l, s_bra, w_r)
        to_concrete(deleteSingleton(X, 3))                 # (ℓ_l, ℓ_r, s_bra, w_r)
    else
        X = to_concrete(contract(H1.l, (3,), A, (1,)))      # (bra_l, w, s, ℓ_r)
        X = to_concrete(contract(X, (2, 3), H1.w, (1, 2)))  # (bra_l, ℓ_r, s_bra, w_r)
        X
    end                                                    # (link_l, ℓ_r, s_bra, w_r)

    T = if H1.r === nothing
        X = to_concrete(deleteSingleton(T, 4))             # (link_l, ℓ_r, s_bra)
        to_concrete(permutedims(X, (1, 3, 2)))
    else
        to_concrete(contract(T, (4, 2), H1.r, (2, 3)))     # (link_l, s_bra, bra_r)
    end

    H1.l === nothing || (T = _unprime(T, 1))
    H1.r === nothing || (T = _unprime(T, 3))
    return T
end

"""
    MPOZeroSiteH

The centre bond's effective Hamiltonian, as the two environments already pushed through
the expanded frames:

    H_eff^{(0)} S = sum_w L[w] S R[w]

One contraction over the MPO leg -- Eq. (1.7) at a bond, where the site tensor of the
1-site form is absent. Acting with it never touches a site tensor, so a Krylov matvec is
`O(chi^2 w)` rather than the `O(chi^2 d^2)` of a two-site apply.
"""
struct MPOZeroSiteH
    l::Any                    # (bra_bL, mpo, bL)
    r::Any                    # (bra_bR, mpo, bR)
end

"""
    zero_site_h(mpo, i, lenv, renv, U_ex, V_ex) -> MPOZeroSiteH

Push the environments at links `i` and `i+2` onto the centre bond through the EXPANDED
frames. `U_ex` carries `(link_l, site_l, bond)` and `V_ex` carries `(bond, site_r,
link_r)` -- an MPS tensor's layout in each case -- so this is one more step of Eq. (1.8)
with the frame in place of a state tensor, and needs no separate code.
"""
zero_site_h(mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, U_ex, V_ex) =
    MPOZeroSiteH(_mpo_left_step(lenv.E, U_ex, mpo[i]),
                 _mpo_right_step(renv.E, V_ex, mpo[i + 1]))

"""
    apply_zero_site(H0::MPOZeroSiteH, S) -> TLArray

`H_eff S` for the centre core `S`, legs `(bond_l, bond_r)` in and out. Each environment
contributes its bra leg in place of the ket leg it consumes, so the result carries `S`'s
own legs after unpriming.
"""
function apply_zero_site(H0::MPOZeroSiteH, S)
    T = to_concrete(contract(H0.l, (3,), S, (1,)))        # (bra_l, mpo, bond_r)
    T = to_concrete(contract(T, (2, 3), H0.r, (2, 3)))    # (bra_l, bra_r)
    return _unprime(_unprime(T, 1), 2)
end
