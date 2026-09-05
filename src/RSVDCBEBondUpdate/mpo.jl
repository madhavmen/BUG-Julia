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
environments. ⚠ A standalone call rebuilds `H*Theta`; inside a sweep the shared cache in
[`_sketch_closures`](@ref) is what runs."
sketch_h_left(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    first(_sketch_closures(f,
        () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)

"Mirror of [`sketch_h_left`](@ref) on MPO environments."
sketch_h_right(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om) =
    last(_sketch_closures(f,
        () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv)))(Om)

# ⛔ ONE `H*Theta` PER EXPANSION, SHARED BY ALL THREE SKETCH CALLS -- see `_sketch_closures`.
# The MPO path is where this costs most: `apply_h_two_site` scales with the MPO's virtual
# dimension, so the old triple build taxed the WIDE generators (square, kagome, Schwinger)
# hardest, which are exactly the models the rSVD study needed to win on.
# ══ FOLD-OMEGA-FIRST: the sketch that never forms `H*Theta` ═══════════════════════════════
#
# ⛔ THE POINT OF A RANDOMISED SKETCH IS TO MAKE `H` ACT ON `Dpre` COLUMNS INSTEAD OF `d*chi_r`,
# AND THE PROJECT-FIRST PATH ABOVE CANNOT DO THAT. `_sketch_closures` builds the full
# `H*Theta` and contracts `Om` into the RESULT, so the sketch shrinks only the SVD and the final
# contraction -- never the object that dominates. That is why turning the sketch on has been
# measured SLOWER than the exact expansion at chi <= 64 (L=18 XX: `t_cbe` 94.4 s -> 123.2 s over a
# trajectory, and 1.750 -> 2.115 on a single pinned step), and why narrowing `Dpre` from 61% of
# full width to 9.4% did not change that: both arms built the same `H*Theta`.
#
# ⚠ THE SAVING IS ASYMPTOTIC IN `chi`, NOT THE `Dpre/(d*chi)` RATIO USUALLY QUOTED. In
# `apply_h_two_site` the two ENVIRONMENT contractions dominate and cost `O(chi^3 d^2 D)` -- the
# left one contracts `(chi,D,chi)` against `(chi,d,d,chi)`, the right one likewise. Folding `Om`
# in first replaces `chi_r`'s `d*chi_r` columns with `g = Dpre` BEFORE those contractions, so
# every step of the chain is `O(chi^2 d g D)` or smaller. A factor `chi/g`, growing with the
# bond dimension -- which is why this matters more at chi = 128/256 than at the chi = 41 the
# campaign has been running at.
#
# ⛔ THE TRADE, AND IT IS A REAL ONE: FOLD-FIRST CANNOT SHARE. `cbe_expand` calls the sketch
# THREE times with DIFFERENT `Om` (`skl(OmR)`, `skr(OmL)`, `skl(QR)`), and a folded chain is
# specific to its `Om`, so it pays three chains where project-first pays ONE build plus three
# cheap contractions. Fold-first wins when `3g < chi` -- comfortably at `dex = 8` (`g = 12`) for
# `chi >= 64`, marginally at `chi = 41`, and NEVER at the `growth = 2.0` width of `g ~ 50`.
# ⇒ THIS FLAG IS ONLY WORTH SETTING TOGETHER WITH A SMALL `dex`. On its own it can lose.
#
# ⚠ THE ORDERINGS AGREE EXACTLY, so this is a cost switch and nothing else. `P_perp` acts on
# `(link_l, site_l)` and `Om` on `(site_r, link_r)` -- disjoint legs, so they commute:
# `P_perp((H Theta) Om') == (P_perp (H Theta)) Om'`. `P_perp` is applied HERE, after the fold, so
# each closure returns exactly what the project-first closure returns. Any difference beyond
# roundoff is a bug in this code, and `tests/rsvd_cbe/` pins it against the other path.
#
# ⛔ A HISTORICAL NOTE THAT MATTERS FOR ANY COMPARISON WITH OLD NUMBERS. A fold-Omega-first path
# existed for the TERM-LIST Hamiltonian (`sketch_h_left(f, h::XXZChain, ...)`) and was deleted in
# 95263fb (2026-08-06, "sketch the projector, and remove the alternative"). The MPO layer arrived
# in 3452a51 (2026-08-17), ELEVEN DAYS LATER -- so the MPO path has NEVER had a fold-first
# variant, and rSVD cost numbers measured through the MPO cannot be compared with the term-list
# ones from before that deletion. This function is new code, not a revert.

"""
    _fold_sketch_closures(f, mpo, i, lenv, renv) -> (skl, skr)

The `(skl, skr)` pair [`cbe_expand`](@ref) wants, with `Om` folded into the contraction chain
instead of applied to a finished `H*Theta`.

`skl(Om)` takes `Om` in `V0`'s layout `(g, site_r, link_r)` and returns `(link_l, site_l, g)`;
`skr(Om)` takes `U0`'s layout `(link_l, site_l, g)` and returns `(g, site_r, link_r)`. Both
return the DISCARDED-space component, i.e. with `P_perp` already applied -- identical to
[`_sketch_closures`](@ref)'s output.
"""
function _fold_sketch_closures(f::BondFrame, mpo::MPO, i::Int,
                               lenv::MPOLink, renv::MPOLink)
    W1, W2 = mpo[i], mpo[i + 1]
    # ⚠ `Theta` IS SHARED, `H*Theta` IS NOT. `frame_theta` is `U0*S0*V0` -- an `O(chi^2 d^2)`
    # object the frame already implies, with no `H` in it. Rebuilding it per call would be pure
    # waste; caching the thing this function exists NOT to build would defeat the purpose.
    Theta = frame_theta(f)                       # (ℓ_l, s_l, s_r, ℓ_r)

    # ── LEFT: fold `Om` through renv and W2, then Theta, then W1 and lenv ────────────────
    function skl(Om)
        Omd = Om'                                # (g, s_bra_r, bra_r)
        # 1. the right cap. At the boundary `renv.E` is absent and `W2`'s `w_r` is the dim-1
        #    leg left by trimming, dropped rather than contracted -- the same asymmetry
        #    `apply_h_two_site` handles, and the reason this is not one expression.
        C = if renv.E === nothing
            X = to_concrete(contract(Omd, (2,), W2, (3,)))   # (g,bra_r,w_mid,s_ket_r,w_r)
            to_concrete(deleteSingleton(X, 5))               # (g, bra_r, w_mid, s_ket_r)
        else
            X = to_concrete(contract(Omd, (3,), renv.E, (1,)))  # (g, s_bra_r, w, ket_r)
            # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close `s_bra_r` and `w_r`.
            to_concrete(contract(X, (2, 3), W2, (3, 4)))     # (g, ket_r, w_mid, s_ket_r)
        end
        # 2. into Theta: close the ket site and ket link on the right.
        D = to_concrete(contract(Theta, (3, 4), C, (4, 2)))  # (ℓ_l, s_l, g, w_mid)
        # 3. W1 = (w_l, s_ket_l, s_bra_l, w_mid); close the ket site and the shared MPO leg.
        E = to_concrete(contract(D, (2, 4), W1, (2, 4)))     # (ℓ_l, g, w_l, s_bra_l)
        Y = if lenv.E === nothing
            X = to_concrete(deleteSingleton(E, 3))           # (ℓ_l, g, s_bra_l)
            to_concrete(permutedims(X, (1, 3, 2)))           # (ℓ_l, s_bra_l, g)
        else
            # lenv.E = (bra, w, ket); close the ket link and `w_l`.
            X = to_concrete(contract(E, (1, 3), lenv.E, (3, 2)))  # (g, s_bra_l, bra)
            _unprime(to_concrete(permutedims(X, (3, 2, 1))), 1)   # (bra, s_bra_l, g)
        end
        return _project_left(f, Y)
    end

    # ── RIGHT: the mirror -- fold `Om` through lenv and W1 first ────────────────────────
    function skr(Om)
        Omd = Om'                                # (bra_l, s_bra_l, g)
        C = if lenv.E === nothing
            X = to_concrete(contract(Omd, (2,), W1, (3,)))   # (bra_l,g,w_l,s_ket_l,w_mid)
            to_concrete(deleteSingleton(X, 3))               # (bra_l, g, s_ket_l, w_mid)
        else
            X = to_concrete(contract(Omd, (1,), lenv.E, (1,)))  # (s_bra_l, g, w, ket_l)
            # W1 = (w_l, s_ket_l, s_bra_l, w_mid); close `s_bra_l` and `w_l`.
            Z = to_concrete(contract(X, (1, 3), W1, (3, 1)))    # (g, ket_l, s_ket_l, w_mid)
            to_concrete(permutedims(Z, (2, 1, 3, 4)))           # (ket_l, g, s_ket_l, w_mid)
        end
        # into Theta: close the ket link and ket site on the left.
        D = to_concrete(contract(Theta, (1, 2), C, (1, 3)))  # (s_r, ℓ_r, g, w_mid)
        # W2 = (w_mid, s_ket_r, s_bra_r, w_r); close the ket site and the shared MPO leg.
        E = to_concrete(contract(D, (1, 4), W2, (2, 1)))     # (ℓ_r, g, s_bra_r, w_r)
        Y = if renv.E === nothing
            X = to_concrete(deleteSingleton(E, 4))           # (ℓ_r, g, s_bra_r)
            to_concrete(permutedims(X, (2, 3, 1)))           # (g, s_bra_r, ℓ_r)
        else
            X = to_concrete(contract(E, (1, 4), renv.E, (3, 2)))  # (g, s_bra_r, bra_r)
            _unprime(to_concrete(X), 3)
        end
        return _project_right(f, Y)
    end

    return skl, skr
end

"`P_perp^L Y` with `P_perp^L = I - U0 U0'`, for `Y` of layout `(link_l, site_l, g)`."
function _project_left(f::BondFrame, Y)
    c = to_concrete(contract(f.U0', (1, 2), Y, (1, 2)))      # (bond, g)
    return to_concrete(Y - to_concrete(contract(f.U0, (3,), c, (1,))))
end

"`Y P_perp^R` with `P_perp^R = I - V0' V0`, for `Y` of layout `(g, site_r, link_r)`."
function _project_right(f::BondFrame, Y)
    c = to_concrete(contract(Y, (2, 3), f.V0', (2, 3)))      # (g, bond)
    return to_concrete(Y - to_concrete(contract(c, (2,), f.V0, (1,))))
end

"""
The bond expansion at bond `i`, driven by MPO environments.

`fold_omega = true` selects [`_fold_sketch_closures`](@ref), which folds the probe into the
contraction instead of building `H*Theta`. ⚠ It makes `share_ht` MEANINGLESS -- there is no
shared object to cache -- so passing both is refused rather than silently ignoring one.
"""
function cbe_expand(f::BondFrame, mpo::MPO, i::Int,
                    lenv::MPOLink, renv::MPOLink;
                    # ⛔ `nothing`, NOT `true`, SO "NOT PASSED" IS DISTINGUISHABLE FROM "PASSED
                    # true". With a `Bool` default the contradiction check below cannot fire at
                    # all -- `share_ht` is a NAMED kwarg, so it never appears in `kwargs...` and
                    # `haskey(kwargs, :share_ht)` is false even when the caller passed it. That
                    # is a guard that reads as protection and provides none.
                    share_ht::Union{Nothing, Bool} = nothing,
                    fold_omega::Bool = false, kwargs...)
    if fold_omega
        # ⛔ REFUSE THE CONTRADICTION RATHER THAN RESOLVING IT. A caller who sets `fold_omega`
        # alone is asserting nothing about sharing and is not warned; a caller who ALSO passes
        # `share_ht` explicitly believes something false about the run they are about to time,
        # and silently honouring one of the two is how a benchmark reports the wrong arm.
        share_ht === nothing || throw(ArgumentError(
            "fold_omega = true never forms H*Theta, so there is nothing for share_ht to " *
            "share; pass one or the other"))
        return cbe_expand(f, _fold_sketch_closures(f, mpo, i, lenv, renv)...; kwargs...)
    end
    return cbe_expand(f,
                      _sketch_closures(f,
                          () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv);
                          share = share_ht === nothing ? true : share_ht)...; kwargs...)
end

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
